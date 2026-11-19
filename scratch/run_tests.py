import subprocess, sys

# All UNME test suites in phase order
PYTEST_SUITES = [
    # Fase 3A, 3B, 3C, 3D — pytest-based
    "d:/memory/tests/test_canonical_retrieval.py",
    "d:/memory/tests/test_adapter.py",
    "d:/memory/tests/test_calibration.py",
    "d:/memory/tests/test_surrogate_trainer.py",
    "d:/memory/tests/test_prefetcher.py",
]

DIRECT_SUITES = [
    # Fases 1 & 2 — direct script execution
    "d:/memory/tests/test_storage_robustness.py",
    "d:/memory/tests/test_isolation_and_indexing.py",
    "d:/memory/tests/test_storage_aead_recovery.py",
    "d:/memory/tests/test_gar_and_cml_lifecycle.py",
    "d:/memory/tests/test_durability_and_rotation.py",
    "d:/memory/tests/test_multiprocess_concurrency.py",
]

results = []
all_passed = True

# Run pytest suites together for a single combined report
r = subprocess.run(
    [sys.executable, "-m", "pytest"] + PYTEST_SUITES + ["--tb=short", "-q"],
    capture_output=True, text=True,
)
results.append(("pytest_suites", r))
if r.returncode != 0:
    all_passed = False
    print("FAIL: pytest suites")
    print(r.stdout[-1200:])
    print(r.stderr[-400:])
else:
    # Extract summary line
    for line in r.stdout.splitlines():
        if "passed" in line:
            print(f"PASS (pytest): {line.strip()}")
            break

# Run direct suites
for t in DIRECT_SUITES:
    r = subprocess.run([sys.executable, t], capture_output=True, text=True)
    results.append((t, r))
    name = t.split("/")[-1]
    if r.returncode != 0:
        print("FAIL:", name)
        print(r.stdout[-600:])
        print(r.stderr[-400:])
        all_passed = False
    else:
        print("PASS:", name)

print()
print("ALL PASSED" if all_passed else "SOME FAILED")

