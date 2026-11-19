"""
Patches NonceEpochManager.generate_nonce in namespace.py to use HKDF-V3.
Replaces the 32-bit session_salt design with a 128-bit-entropy-derived nonce.
"""
import pathlib

target = pathlib.Path("d:/memory/unme/crypto/namespace.py")
content = target.read_text(encoding="utf-8")

# ------------------------------------------------------------------
# 1. Remove unused session_salt field
# ------------------------------------------------------------------
content = content.replace(
    "        # Per-instance session salt\n        self._session_salt = os.urandom(4)\n",
    "",
)
print("session_salt line removed:", "session_salt" not in content)

# ------------------------------------------------------------------
# 2. Replace generate_nonce (NonceEpochManager, takes record_index)
# ------------------------------------------------------------------
OLD_NONCE = (
    '    def generate_nonce(self, record_index: int) -> bytes:\n'
    '        """Constructs a 12-byte composite nonce:\n'
    '        - 4 bytes: record_index (uint32) -> monotonic within write session\n'
    '        - 4 bytes: write_epoch (uint32)   -> monotonic across process restarts\n'
    '        - 4 bytes: session_salt (4 bytes) -> entropy protection\n'
    '        """\n'
    '        return struct.pack("<II4s", record_index, self._write_epoch, self._session_salt)\n'
)

NEW_NONCE = (
    '    def generate_nonce(self, record_index: int) -> bytes:\n'
    '        """Derives a 12-byte nonce from 128-bit epoch entropy + record index (V3).\n'
    '\n'
    '        ChaCha20-Poly1305 (RFC 7539) mandates a 96-bit nonce (12 bytes, hard limit).\n'
    '        XChaCha20-Poly1305 (24-byte nonce) is not available in this environment.\n'
    '\n'
    '        Previous design packed (record_index u32, write_epoch u32, session_salt 4B)\n'
    '        directly, providing only 32 bits of random differentiation entropy beyond the\n'
    '        deterministic monotone counters.\n'
    '\n'
    '        V3 design: HKDF-Expand folds the 128-bit epoch_token + record_index into\n'
    '        12 bytes of output keying material. Each record receives a nonce derived from\n'
    '        128 bits of entropy. Two writers with the same record_index but different\n'
    '        epoch_tokens produce orthogonal nonces. Uniqueness within a session follows\n'
    '        from HKDF being a deterministic PRF (distinct info strings -> distinct blocks).\n'
    '        The nonce is reproduced deterministically from epoch_token (in .epoch file)\n'
    '        and record_index (in segment header); nothing additional is stored on disk.\n'
    '        """\n'
    '        hkdf_expand = HKDF(\n'
    '            algorithm=hashes.SHA256(),\n'
    '            length=12,\n'
    '            salt=None,\n'
    '            info=b"UNME_RECORD_NONCE_V3:" + struct.pack("<I", record_index),\n'
    '        )\n'
    '        return hkdf_expand.derive(self._epoch_token)\n'
)

if OLD_NONCE in content:
    content = content.replace(OLD_NONCE, NEW_NONCE)
    print("generate_nonce replaced with HKDF-V3")
else:
    print("WARNING: OLD_NONCE not found verbatim; searching manually")
    idx = content.find("def generate_nonce(self, record_index")
    print(repr(content[idx:idx+350]))

target.write_text(content, encoding="utf-8")
print("namespace.py written")
