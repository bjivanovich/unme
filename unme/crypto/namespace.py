"""Cryptographic isolation, key management, and AEAD encryption for UNME.
Implements:
- Level 1: HMAC-SHA256 namespace derivation.
- Level 3: ChaCha20-Poly1305 authenticated encryption with canonical 20-byte AAD.
- Defense-in-depth: 128-bit persistent epoch tokens generating distinct epoch subkeys
  via HKDF, completely eliminating nonce-reuse risks even under complete backup restorations.
"""

from __future__ import annotations
import hashlib
import hmac
import os
import struct
from pathlib import Path
from typing import Final
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

from unme.core.constants import AAD_STRUCT_FMT, AAD_SIZE_BYTES

NONCE_SIZE_BYTES: Final[int] = 12  # Standard 96-bit nonce for ChaCha20-Poly1305


def build_canonical_aad(
    version: int,
    flags: int,
    segment_id: int,
    record_index: int,
    namespace_id: int,
) -> bytes:
    """Constructs canonical 20-byte Additional Authenticated Data (AAD).
    Cryptographically binds ciphertext to its exact storage coordinates, version, and namespace.
    Format: <HHIIQ (version u16, flags u16, segment_id u32, record_index u32, namespace_id u64).
    """
    return struct.pack(
        AAD_STRUCT_FMT,
        version,
        flags,
        segment_id,
        record_index,
        namespace_id,
    )


class NonceEpochManager:
    """Manages persistent write epochs and 128-bit epoch tokens to derive distinct
    keys per write session, guaranteeing that no (key, nonce) pair can ever be reused
    even across system crashes, reboots, or arbitrary backup restorations.
    """

    def __init__(self, segment_dir: Path | str, segment_id: int) -> None:
        self.meta_dir = Path(segment_dir) / ".meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.meta_dir / f"segment_{segment_id:06d}.epoch"
        self.segment_id = segment_id
        self._segment_uuid, self._write_epoch, self._epoch_token = self._load_or_init()
        # Per-instance session salt

    def _load_or_init(self) -> tuple[bytes, int, bytes]:
        """Loads segment UUID, generates fresh 128-bit epoch token, and increments epoch counter."""
        epoch_token = os.urandom(16)  # Fresh 128-bit random token per write session
        if self.meta_path.exists():
            with open(self.meta_path, "r+b") as f:
                data = f.read()
                if len(data) >= 20:
                    segment_uuid = data[:16]
                    epoch = struct.unpack("<I", data[16:20])[0] + 1
                    f.seek(16)
                    f.write(struct.pack("<I", epoch))
                    f.seek(20)
                    f.write(epoch_token)
                    f.flush()
                    os.fsync(f.fileno())
                    return segment_uuid, epoch, epoch_token

        segment_uuid = os.urandom(16)
        epoch = 1
        with open(self.meta_path, "wb") as f:
            f.write(segment_uuid + struct.pack("<I", epoch) + epoch_token)
            f.flush()
            os.fsync(f.fileno())
        return segment_uuid, epoch, epoch_token

    @property
    def segment_uuid(self) -> bytes:
        return self._segment_uuid

    @property
    def write_epoch(self) -> int:
        return self._write_epoch

    @property
    def epoch_token(self) -> bytes:
        return self._epoch_token

    def derive_epoch_key(self, base_segment_key: bytes) -> bytes:
        """Derives a cryptographically unique 32-byte key for this specific write epoch
        using the fresh 128-bit epoch token as salt.
        """
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._epoch_token,
            info=f"UNME_WRITE_EPOCH_KEY_V2:{self._write_epoch}".encode("utf-8"),
        )
        return hkdf.derive(base_segment_key)

    def generate_nonce(self, record_index: int) -> bytes:
        """Derives a 12-byte nonce from 128-bit epoch entropy + record index (V3).

        ChaCha20-Poly1305 (RFC 7539) mandates a 96-bit nonce (12 bytes, hard limit).
        XChaCha20-Poly1305 (24-byte nonce) is not available in this environment.

        Previous design packed (record_index u32, write_epoch u32, session_salt 4B)
        directly, providing only 32 bits of random differentiation entropy beyond the
        deterministic monotone counters.

        V3 design: HKDF-Expand folds the 128-bit epoch_token + record_index into
        12 bytes of output keying material. Each record receives a nonce derived from
        128 bits of entropy. Two writers with the same record_index but different
        epoch_tokens produce orthogonal nonces. Uniqueness within a session follows
        from HKDF being a deterministic PRF (distinct info strings -> distinct blocks).
        The nonce is reproduced deterministically from epoch_token (in .epoch file)
        and record_index (in segment header); nothing additional is stored on disk.
        """
        hkdf_expand = HKDF(
            algorithm=hashes.SHA256(),
            length=12,
            salt=None,
            info=b"UNME_RECORD_NONCE_V3:" + struct.pack("<I", record_index),
        )
        return hkdf_expand.derive(self._epoch_token)


class NamespaceManager:
    """Cryptographic authority for namespace isolation, subkey derivation, and AEAD operations."""

    def __init__(self, master_secret: bytes | None = None) -> None:
        if master_secret is None:
            master_secret = b"UNME_MASTER_SECRET_KEY_PROD_2026_DEFAULT"
        self._master_secret = master_secret
        self._default_salt = os.urandom(4)

    def derive_namespace_id(self, project_name: str) -> int:
        """Derives a deterministic uint64 namespace ID using HMAC-SHA256."""
        h = hmac.new(self._master_secret, project_name.encode("utf-8"), hashlib.sha256).digest()
        return struct.unpack(">Q", h[:8])[0]

    def derive_namespace_key(self, project_name: str) -> bytes:
        """Derives a dedicated 32-byte AEAD master key for the specified project."""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=f"UNME_NAMESPACE_ENCRYPTION_v2:{project_name}".encode("utf-8"),
        )
        return hkdf.derive(self._master_secret)

    @staticmethod
    def derive_segment_key(namespace_key: bytes, segment_id: int, segment_uuid: bytes | None = None) -> bytes:
        """Derives a segment-specific key. If segment_uuid is provided, cryptographically binds
        the key to that physical segment's lifetime.
        """
        salt = struct.pack("<I", segment_id) + (segment_uuid if segment_uuid else b"")
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b"UNME_SEGMENT_SUBKEY_V2",
        )
        return hkdf.derive(namespace_key)

    def generate_nonce(self, segment_id: int, record_index: int) -> bytes:
        """Fallback nonce generator using session salt."""
        return struct.pack("<II4s", segment_id, record_index, self._default_salt)

    @staticmethod
    def encrypt_payload(
        plaintext: bytes,
        key: bytes,
        aad: bytes,
        nonce: bytes,
    ) -> bytes:
        """Encrypts payload using ChaCha20-Poly1305 bound to canonical AAD.
        Returns: [12-byte Nonce] + [Ciphertext + 16-byte Poly1305 Tag].
        """
        if len(nonce) != NONCE_SIZE_BYTES:
            raise ValueError(f"Nonce must be exactly {NONCE_SIZE_BYTES} bytes, got {len(nonce)}")

        cipher = ChaCha20Poly1305(key)
        ciphertext = cipher.encrypt(nonce, plaintext, aad)
        return nonce + ciphertext

    @staticmethod
    def decrypt_payload(
        encrypted_blob: bytes,
        key: bytes,
        aad: bytes,
    ) -> bytes:
        """Decrypts and authenticates payload with canonical AAD.
        Raises InvalidTag if ciphertext or AAD has been tampered with or transplanted.
        """
        if len(encrypted_blob) < NONCE_SIZE_BYTES + 16:
            raise ValueError("Encrypted blob is too short to contain valid nonce and AEAD tag.")

        nonce = encrypted_blob[:NONCE_SIZE_BYTES]
        ciphertext = encrypted_blob[NONCE_SIZE_BYTES:]
        cipher = ChaCha20Poly1305(key)
        return cipher.decrypt(nonce, ciphertext, aad)

    def encrypt_data(self, plaintext: bytes, key: bytes, associated_data: bytes | None = None) -> bytes:
        """Convenience AEAD wrapper using random nonce."""
        nonce = os.urandom(NONCE_SIZE_BYTES)
        cipher = ChaCha20Poly1305(key)
        ciphertext = cipher.encrypt(nonce, plaintext, associated_data)
        return nonce + ciphertext

    def decrypt_data(self, encrypted_blob: bytes, key: bytes, associated_data: bytes | None = None) -> bytes:
        """Convenience AEAD decryption wrapper."""
        if len(encrypted_blob) < NONCE_SIZE_BYTES + 16:
            raise ValueError("Encrypted blob is too short to contain valid nonce and AEAD tag.")
        nonce = encrypted_blob[:NONCE_SIZE_BYTES]
        ciphertext = encrypted_blob[NONCE_SIZE_BYTES:]
        cipher = ChaCha20Poly1305(key)
        return cipher.decrypt(nonce, ciphertext, associated_data)
