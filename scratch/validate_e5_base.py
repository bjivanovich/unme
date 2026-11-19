"""Validation script for intfloat/multilingual-e5-base in UNME.

Verifies:
1. Model loads from local HF cache (local_files_only=True).
2. Output shape is exactly (768,) float16.
3. Vectors are unit-norm after L2-normalization.
4. No NaN or Inf values in output.
5. Semantic similarity ordering: exact > related > unrelated.
6. Empirical thresholds for GAR recalibration.
"""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path("d:/memory")))

from unme.canonical.encoder import MultilingualE5SmallEncoder, CANONICAL_DIM

print("=" * 64)
print("UNME encoder validation: intfloat/multilingual-e5-base")
print("=" * 64)
print()

enc = MultilingualE5SmallEncoder(device="cpu")

# --- Test 1: load and dimension check ---
print("[1] Loading model (first call triggers lazy init)...")
q = enc.encode_query("test query for dimension check")
print(f"    Output shape : {q.shape}")
print(f"    Output dtype : {q.dtype}")
print(f"    CANONICAL_DIM: {CANONICAL_DIM}")
assert q.shape == (768,), f"FAIL: expected (768,) got {q.shape}"
assert q.dtype == np.float16, f"FAIL: expected float16 got {q.dtype}"
print("    [PASS] shape and dtype correct")
print()

# --- Test 2: unit norm ---
norm = float(np.linalg.norm(q.astype(np.float32)))
print(f"[2] L2 norm: {norm:.6f}  (expected ~1.0)")
assert abs(norm - 1.0) < 0.01, f"FAIL: norm {norm} not near 1.0"
print("    [PASS] unit norm")
print()

# --- Test 3: no NaN/Inf ---
print("[3] NaN/Inf check...")
assert not np.any(np.isnan(q)), "FAIL: NaN in query vector"
assert not np.any(np.isinf(q)), "FAIL: Inf in query vector"
print("    [PASS] all values finite")
print()

# --- Test 4: encode_record returns (K, V) pair ---
print("[4] encode_record shape check...")
K, V = enc.encode_record("GNSS Kalman filter satellite navigation", {"domain": "nav"})
assert K.shape == (768,), f"FAIL: K shape {K.shape}"
assert V.shape == (768,), f"FAIL: V shape {V.shape}"
assert K.dtype == np.float16
assert not np.any(np.isnan(K))
print(f"    K shape: {K.shape}, V shape: {V.shape}")
print("    [PASS] record encoding correct")
print()

# --- Test 5: semantic similarity calibration ---
print("[5] Semantic similarity calibration (cosine)...")
print()

def cosine(a, b):
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    return float(np.dot(a32, b32) / (np.linalg.norm(a32) * np.linalg.norm(b32) + 1e-8))

scenarios = [
    # (label, query_text, passage_text, expected_category)
    ("Exact match (ES)",
     "filtro de Kalman para navegacion GNSS",
     "filtro de Kalman para navegacion GNSS",
     "exact"),
    ("Exact match (EN)",
     "Kalman filter for GNSS navigation",
     "Kalman filter for GNSS navigation",
     "exact"),
    ("Semantically related (ES-EN cross-lingual)",
     "filtro de Kalman satelite",
     "satellite Kalman filter",
     "related"),
    ("Semantically related (same domain)",
     "GNSS satellite positioning error estimation",
     "Kalman smoother for inertial navigation fusion",
     "related"),
    ("Weakly related (ML + nav)",
     "neural network training gradient descent",
     "GNSS Kalman filter satellite navigation",
     "weak"),
    ("Completely unrelated (cooking)",
     "GNSS Kalman filter satellite navigation",
     "receta de pollo al horno con patatas",
     "unrelated"),
    ("Completely unrelated (random)",
     "cryptographic key derivation HKDF nonce",
     "receta de pasta carbonara italiana",
     "unrelated"),
    ("Code vs natural language",
     "def forward(self, x): return self.linear(x)",
     "forward pass neural network linear layer",
     "related"),
]

results = {}
for label, q_text, p_text, category in scenarios:
    q_vec = enc.encode_query(q_text)
    p_vec, _ = enc.encode_record(p_text)
    sim = cosine(q_vec, p_vec)
    results[category] = results.get(category, []) + [sim]
    print(f"  [{category:10s}] sim={sim:.4f}  |  q: {q_text[:45]!r}")
    print(f"               {'':10s}         |  p: {p_text[:45]!r}")
    print()

print("=" * 64)
print("SUMMARY by category:")
for cat in ["exact", "related", "weak", "unrelated"]:
    vals = results.get(cat, [])
    if vals:
        print(f"  {cat:12s}: min={min(vals):.4f}  max={max(vals):.4f}  "
              f"mean={sum(vals)/len(vals):.4f}  (n={len(vals)})")

print()
print("GAR THRESHOLD RECOMMENDATIONS:")
print("  threshold_local  = similarity where unrelated content starts appearing")
print("  threshold_global = tighter filter for cross-namespace queries")
print()

unrelated_max = max(results.get("unrelated", [0.0]))
related_min   = min(results.get("related",   [1.0]))
exact_min     = min(results.get("exact",     [1.0]))

print(f"  unrelated MAX  : {unrelated_max:.4f}")
print(f"  related   MIN  : {related_min:.4f}")
print(f"  exact     MIN  : {exact_min:.4f}")
print()

# Recommend threshold as midpoint between unrelated_max and related_min
midpoint = (unrelated_max + related_min) / 2.0
print(f"  Midpoint (unrelated_max + related_min) / 2 = {midpoint:.4f}")
print(f"  RECOMMENDED threshold_local  = {midpoint - 0.03:.2f}  (conservative, recall focus)")
print(f"  RECOMMENDED threshold_global = {midpoint + 0.03:.2f}  (strict, precision focus)")
print()

# Ordering assertion
avg = lambda cat: sum(results.get(cat, [0])) / max(len(results.get(cat, [1])), 1)
assert avg("exact") > avg("related"), "FAIL: exact not > related"
assert avg("related") > avg("unrelated"), "FAIL: related not > unrelated"
print("[PASS] Semantic ordering: exact > related > unrelated")
print()
print("Validation complete.")
