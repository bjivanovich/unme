"""Rigorous benchmark for UNME storage core on NVMe SSD.
Measures sequential write throughput and random-access read latency percentiles (P50, P95, P99, P99.9).
"""

from __future__ import annotations
import gc
import os
import random
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from unme.core.record import UMRRecord
from unme.core.storage import SegmentWriter, SegmentReader


def run_benchmark(
    target_dir: Path | str,
    num_records: int = 10000,
    num_queries: int = 10000,
    dim: int = 1536,
) -> dict[str, float]:
    bench_dir = Path(target_dir) / "bench_storage"
    if bench_dir.exists():
        shutil.rmtree(bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f"UNME Storage Engine NVMe Benchmark")
    print(f"Target Directory: {bench_dir}")
    print(f"Dataset Size:     {num_records} records")
    print(f"Dimensions:       {dim} fp16 (Key: {dim*2}B, Value: {dim*2}B)")
    print(f"Total Queries:    {num_queries} random lookups")
    print(f"============================================================")

    # 1. Sequential Write Benchmark
    print("\n[Phase 1] Executing sequential append benchmark...")
    writer = SegmentWriter(bench_dir, segment_id=1)

    t0_write = time.perf_counter()
    for i in range(num_records):
        key = np.random.randn(dim).astype(np.float16)
        val = np.random.randn(dim).astype(np.float16)
        record = UMRRecord(
            namespace_id=42,
            canonical_text=f"Benchmark factual memory entry #{i:06d} for NVMe validation.",
            structured_data={"index": i, "category": "benchmark"},
            key_vector=key,
            value_vector=val,
        )
        writer.append(record)

    writer.flush()
    writer.close()
    t_write_total = time.perf_counter() - t0_write

    header_bytes = os.path.getsize(bench_dir / "segment_000001.umrh")
    tensor_bytes = os.path.getsize(bench_dir / "segment_000001.umrt")
    total_bytes = header_bytes + tensor_bytes
    total_mb = total_bytes / (1024 * 1024)

    write_throughput_mb_s = total_mb / t_write_total
    write_records_s = num_records / t_write_total

    print(f"Write Completed in:      {t_write_total:.3f} s")
    print(f"Total Data Written:      {total_mb:.2f} MB")
    print(f"Write Throughput:        {write_throughput_mb_s:.2f} MB/s")
    print(f"Write Rate:              {write_records_s:.1f} records/s")

    # Force OS cache flush / release
    gc.collect()

    # 2. Random Read Benchmark (mmap)
    print("\n[Phase 2] Executing random read latency benchmark (mmap)...")
    reader = SegmentReader(bench_dir, segment_id=1)

    query_indices = [random.randint(0, num_records - 1) for _ in range(num_queries)]

    # Latency tracking in microseconds
    latencies_us: list[float] = []

    # Warmup 100 queries
    for idx in query_indices[:100]:
        _ = reader.get_record(idx)

    t0_reads = time.perf_counter()
    for idx in query_indices:
        q_start = time.perf_counter()
        header = reader.get_header(idx)
        k_vec = reader.get_key_vector(header)
        v_vec = reader.get_value_vector(header)
        payload = reader.get_payload(header)
        q_dur = (time.perf_counter() - q_start) * 1_000_000.0  # us
        latencies_us.append(q_dur)

    t_read_total = time.perf_counter() - t0_reads
    reader.close()

    latencies_arr = np.array(latencies_us)
    p50 = float(np.percentile(latencies_arr, 50))
    p90 = float(np.percentile(latencies_arr, 90))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    p999 = float(np.percentile(latencies_arr, 99.9))
    max_lat = float(np.max(latencies_arr))
    mean_lat = float(np.mean(latencies_arr))
    read_throughput_qps = num_queries / t_read_total

    print(f"\n=================== LATENCY RESULTS ===================")
    print(f"Queries Processed:   {num_queries}")
    print(f"Read Rate:           {read_throughput_qps:.1f} lookups/sec")
    print(f"P50 Latency:         {p50:.2f} us ({p50/1000.0:.4f} ms)")
    print(f"P90 Latency:         {p90:.2f} us ({p90/1000.0:.4f} ms)")
    print(f"P95 Latency:         {p95:.2f} us ({p95/1000.0:.4f} ms)")
    print(f"P99 Latency:         {p99:.2f} us ({p99/1000.0:.4f} ms)")
    print(f"P99.9 Latency:       {p999:.2f} us ({p999/1000.0:.4f} ms)")
    print(f"Max Latency:         {max_lat:.2f} us ({max_lat/1000.0:.4f} ms)")
    print(f"Mean Latency:        {mean_lat:.2f} us ({mean_lat/1000.0:.4f} ms)")
    print(f"=======================================================")

    # Verify against Contract Criteria:
    # Contract: P50 < 0.15 ms (150 us), P95 < 2.5 ms (2500 us), P99 < 0.8 ms (800 us)
    passed_p50 = (p50 / 1000.0) < 0.15
    passed_p95 = (p95 / 1000.0) < 2.5
    print("\nContractual Compliance Assessment:")
    print(f"- P50 < 0.15 ms:  {'PASS' if passed_p50 else 'FAIL'} ({p50/1000.0:.4f} ms)")
    print(f"- P95 < 2.50 ms:  {'PASS' if passed_p95 else 'FAIL'} ({p95/1000.0:.4f} ms)")

    return {
        "p50_us": p50,
        "p95_us": p95,
        "p99_us": p99,
        "max_us": max_lat,
        "mean_us": mean_lat,
        "qps": read_throughput_qps,
        "write_mb_s": write_throughput_mb_s,
    }


if __name__ == "__main__":
    run_benchmark(target_dir="d:/memory/data", num_records=10000, num_queries=10000)
