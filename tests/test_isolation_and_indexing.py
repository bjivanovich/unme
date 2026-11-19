"""Adversarial isolation and vector indexing test suite for UNME.
Verifies four-level namespace isolation, AEAD tampering resistance, and confidence thresholding.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unme.crypto.namespace import NamespaceManager
from unme.index.vector_index import PartitionedVectorIndex


def test_cryptographic_namespace_derivation_and_aead():
    mgr = NamespaceManager(master_secret=b"TEST_MASTER_KEY_0123456789ABCDEF")

    ns_alpha = mgr.derive_namespace_id("Project_Alpha")
    ns_beta = mgr.derive_namespace_id("Project_Beta")
    assert ns_alpha != ns_beta, "Namespace collision between Alpha and Beta!"

    key_alpha = mgr.derive_namespace_key("Project_Alpha")
    key_beta = mgr.derive_namespace_key("Project_Beta")
    assert key_alpha != key_beta, "Key derivation collision between Alpha and Beta!"

    # 1. AEAD encryption and decryption
    secret_payload = b"Confidential architectural memory for Alpha."
    encrypted_blob = mgr.encrypt_data(secret_payload, key_alpha)

    decrypted = mgr.decrypt_data(encrypted_blob, key_alpha)
    assert decrypted == secret_payload, "Decrypted text mismatch!"

    # 2. Adversarial decryption using Project Beta's key MUST fail authentication
    beta_decrypt_failed = False
    try:
        _ = mgr.decrypt_data(encrypted_blob, key_beta)
    except Exception:
        beta_decrypt_failed = True
    assert beta_decrypt_failed, "Project Beta was able to decrypt Project Alpha's AEAD ciphertext!"

    print("- Cryptographic derivation and AEAD isolation: PASSED")


def test_adversarial_vector_isolation_and_thresholding():
    mgr = NamespaceManager(master_secret=b"TEST_MASTER_KEY_0123456789ABCDEF")
    ns_alpha = mgr.derive_namespace_id("Project_Alpha")
    ns_beta = mgr.derive_namespace_id("Project_Beta")

    index = PartitionedVectorIndex(dim=512)

    # Base target vector
    target_vec = np.random.randn(512).astype(np.float32)
    target_vec /= np.linalg.norm(target_vec)

    # Populate Project Alpha with target vector and variations
    for i in range(10):
        noise = np.random.randn(512).astype(np.float32) * 0.005
        v = target_vec + noise
        index.add_record(namespace_id=ns_alpha, segment_id=1, record_index=i, key_vector=v)

    # Populate Project Beta with completely orthogonal vectors
    ortho_base = np.random.randn(512).astype(np.float32)
    # Make orthogonal to target_vec
    ortho_base -= np.dot(ortho_base, target_vec) * target_vec
    ortho_base /= np.linalg.norm(ortho_base)

    for i in range(10):
        noise = np.random.randn(512).astype(np.float32) * 0.005
        v = ortho_base + noise
        index.add_record(namespace_id=ns_beta, segment_id=2, record_index=i, key_vector=v)

    # Adversarial Attack 1: Query Project Beta using Project Alpha's target vector
    # Project Beta MUST return 0 results because of thresholding and strict partition isolation
    results_beta = index.search(namespace_id=ns_beta, query_vector=target_vec, top_k=5, threshold=0.75)
    assert len(results_beta) == 0, f"Cross-tenant leak! Project Beta returned {len(results_beta)} results for Alpha query!"

    for r in results_beta:
        assert r.namespace_id == ns_beta, "Leaked foreign namespace ID in search result!"

    # Valid Query 1: Query Project Alpha using Alpha's target vector
    results_alpha = index.search(namespace_id=ns_alpha, query_vector=target_vec, top_k=5, threshold=0.75)
    assert len(results_alpha) > 0, "Failed to retrieve matching memories in Project Alpha!"
    for r in results_alpha:
        assert r.namespace_id == ns_alpha, "Foreign record found in Alpha partition!"
        assert r.similarity >= 0.75, "Result below confidence threshold!"

    # Adversarial Attack 2: Cache Poisoning Test
    # Alpha queries with target_vec (cached)
    _ = index.search(namespace_id=ns_alpha, query_vector=target_vec, top_k=5, threshold=0.75, use_cache=True)
    # Beta queries with EXACT SAME target_vec (must NOT hit Alpha's cache entry)
    results_beta_cached = index.search(namespace_id=ns_beta, query_vector=target_vec, top_k=5, threshold=0.75, use_cache=True)
    assert len(results_beta_cached) == 0, "Cache poisoning leak between namespaces!"

    # Test 3: Confidence threshold rejection for out-of-domain query
    random_irrelevant_query = np.random.randn(512).astype(np.float32)
    results_irrelevant = index.search(namespace_id=ns_alpha, query_vector=random_irrelevant_query, top_k=5, threshold=0.85)
    assert len(results_irrelevant) == 0, "Confidence gate failed to inhibit irrelevant query!"

    print("- Adversarial vector index isolation & thresholding: PASSED")


if __name__ == "__main__":
    print("Running UNME Isolation & Indexing Tests...")
    test_cryptographic_namespace_derivation_and_aead()
    test_adversarial_vector_isolation_and_thresholding()
    print("ALL PHASE 2 TESTS PASSED.")
