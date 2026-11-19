"""Cryptographic benchmark for UNME:
Measures the precise throughput, latency percentiles (P50, P95, P99), and storage overhead
of ChaCha20-Poly1305 AEAD authenticated encryption vs unencrypted raw storage.
"""

from __future__ import annotations
import os
import random
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unme.core.record import UMRRecord
from unme.core.storage import SegmentWriter, SegmentReader
from unme.crypto.namespace import NamespaceManager


def run_crypto_benchmark(target_dir: Path | str, num_records: int = 5000, num_queries: int = 5000):
    bench_dir = Path(target_dir) / "bench_crypto"
    bench_dir.mkdir(parents=True, exist_ok=True)

    dir_plain = bench_dir / "plaintext"
    dir_aead = bench_dir / "aead_encrypted"
    dir_plain.mkdir(parents=True, exist_ok=True)
    dir_aead.mkdir(parents=True, exist_ok=True)

    mgr = NamespaceManager(master_secret=b"BENCH_SECRET_KEY_1234567890ABCDEF")
    ns_id = mgr.derive_namespace_id("Benchmark_Project")
    key = mgr.derive_namespace_key("Benchmark_Project")

    dim = 1536
    sample_text = "Factual architectural benchmark record measuring AEAD vs plaintext storage overhead."

    print("==================================================================")
    print("UNME Cryptographic Overhead Benchmark (ChaCha20-Poly1305 AEAD)")
    print(f"Target:       {bench_dir}")
    print(f"Records:      {num_records} (Key: {dim*2}B, Value: {dim*2}B)")
    print(f"Queries:      {num_queries} random lookups per mode")
    print("==================================================================")

    # 1. Plaintext Write Benchmark
    writer_plain = SegmentWriter(dir_plain, segment_id=1)
    t0_wp = time.perf_counter()
    for i in range(num_records):
        k = np.random.randn(dim).astype(np.float16)
        v = np.random.randn(dim).astype(np.float16)
        rec = UMRRecord(
            namespace_id=ns_id,
            canonical_text=sample_text,
            key_vector=k,
            value_vector=v,
        )
        writer_plain.append(rec, encrypt=False)
    writer_plain.close()
    t_write_plain = time.perf_counter() - t0_wp
    size_plain_bytes = os.path.getsize(dir_plain / "segment_000001.umrt")

    # 2. AEAD Encrypted Write Benchmark
    writer_aead = SegmentWriter(dir_aead, segment_id=1, namespace_mgr=mgr, namespace_key=key)
    t0_we = time.perf_counter()
    for i in range(num_records):
        k = np.random.randn(dim).astype(np.float16)
        v = np.random.randn(dim).astype(np.float16)
        rec = UMRRecord(
            namespace_id=ns_id,
            canonical_text=sample_text,
            key_vector=k,
            value_vector=v,
        )
        writer_aead.append(rec, encrypt=True)
    writer_aead.close()
    t_write_aead = time.perf_counter() - t0_we
    size_aead_bytes = os.path.getsize(dir_aead / "segment_000001.umrt")

    # Storage overhead
    storage_overhead_bytes = size_aead_bytes - size_plain_bytes
    overhead_per_record = storage_overhead_bytes / num_records

    # 3. Read Latency Benchmark (mmap)
    query_indices = [random.randint(0, num_records - 1) for _ in range(num_queries)]

    # Plaintext Reads
    reader_plain = SegmentReader(dir_plain, segment_id=1)
    latencies_plain_us = []
    for idx in query_indices:
        t0 = time.perf_counter()
        _ = reader_plain.get_record(idx, verify_checksum=True)
        latencies_plain_us.append((time.perf_counter() - t0) * 1_000_000.0)
    reader_plain.close()

    # AEAD Reads (verifying CRC32 + Poly1305 authentication + ChaCha20 decryption)
    reader_aead = SegmentReader(dir_aead, segment_id=1, namespace_key=key)
    latencies_aead_us = []
    for idx in query_indices:
        t0 = time.perf_counter()
        _ = reader_aead.get_record(idx, verify_checksum=True)
        latencies_aead_us.append((time.perf_counter() - t0) * 1_000_000.0)
    reader_aead.close()

    lp = np.array(latencies_plain_us)
    la = np.array(latencies_aead_us)

    write_throughput_plain = (size_plain_bytes / (1024 * 1024)) / t_write_plain
    write_throughput_aead = (size_aead_bytes / (1024 * 1024)) / t_write_aead

    print("\n======================== BENCHMARK RESULTS ========================")
    print(f"Metric                        Plaintext Storage       AEAD Encrypted Storage")
    print(f"------------------------------------------------------------------")
    print(f"Write Throughput:             {write_throughput_plain:8.2f} MB/s        {write_throughput_aead:8.2f} MB/s")
    print(f"Write Rate:                   {num_records/t_write_plain:8.1f} rec/s         {num_records/t_write_aead:8.1f} rec/s")
    print(f"Storage Size (.umrt):         {size_plain_bytes/(1024*1024):8.2f} MB          {size_aead_bytes/(1024*1024):8.2f} MB")
    print(f"Storage Overhead / Record:          0 bytes           {overhead_per_record:8.1f} bytes (Nonce+Tag)")
    print(f"Read P50 Latency:             {np.percentile(lp, 50):8.2f} us          {np.percentile(la, 50):8.2f} us")
    print(f"Read P90 Latency:             {np.percentile(lp, 90):8.2f} us          {np.percentile(la, 90):8.2f} us")
    print(f"Read P95 Latency:             {np.percentile(lp, 95):8.2f} us          {np.percentile(la, 95):8.2f} us")
    print(f"Read P99 Latency:             {np.percentile(lp, 99):8.2f} us          {np.percentile(la, 99):8.2f} us")
    print(f"Read Max Latency:             {np.max(lp):8.2f} us          {np.max(la):8.2f} us")
    print(f"Read Mean Latency:            {np.mean(lp):8.2f} us          {np.mean(la):8.2f} us")
    print(f"AEAD Decryption Overhead:           -                 {np.mean(la) - np.mean(lp):8.2f} us / lookup")
    print("==================================================================")


if __name__ == "__main__":
    run_crypto_benchmark(target_dir="d:/memory/data")
