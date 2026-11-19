"""Comparison benchmark: Warm Cache (mmap / OS Page Cache) vs Physical NVMe Cold Read (Unbuffered Direct I/O).
Uses Windows FILE_FLAG_NO_BUFFERING to bypass OS cache and measure true hardware flash latency.
"""

from __future__ import annotations
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import win32file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unme.core.constants import UMR_HEADER_SIZE
from unme.core.record import UMRRecord
from unme.core.storage import SegmentWriter, SegmentReader

SECTOR_SIZE = 4096  # 4 KB sector alignment required for unbuffered I/O


def run_comparison(target_dir: Path | str, num_records: int = 5000, num_queries: int = 2000):
    bench_dir = Path(target_dir) / "bench_cold_warm"
    bench_dir.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f"UNME Storage: Warm-Cache vs Direct Unbuffered NVMe Flash")
    print(f"Target:       {bench_dir}")
    print(f"Dataset:      {num_records} records (~31 MB)")
    print(f"Queries:      {num_queries} random lookups per mode")
    print(f"============================================================")

    # 1. Prepare data
    writer = SegmentWriter(bench_dir, segment_id=99)
    dim = 1536
    for i in range(num_records):
        k = np.random.randn(dim).astype(np.float16)
        v = np.random.randn(dim).astype(np.float16)
        rec = UMRRecord(
            namespace_id=999,
            canonical_text=f"Sample factual memory #{i:05d}",
            key_vector=k,
            value_vector=v,
        )
        writer.append(rec)
    writer.flush()
    writer.close()

    tensor_path = bench_dir / "segment_000099.umrt"
    tensor_file_size = os.path.getsize(tensor_path)

    query_indices = [random.randint(0, num_records - 1) for _ in range(num_queries)]

    # 2. Warm Cache Benchmark (mmap)
    reader = SegmentReader(bench_dir, segment_id=99)
    warm_latencies_us = []
    # Warmup
    for idx in query_indices[:50]:
        _ = reader.get_record(idx)

    for idx in query_indices:
        t0 = time.perf_counter()
        header = reader.get_header(idx)
        _ = reader.get_key_vector(header)
        _ = reader.get_value_vector(header)
        _ = reader.get_payload(header)
        dur = (time.perf_counter() - t0) * 1_000_000.0
        warm_latencies_us.append(dur)
    reader.close()

    # 3. Direct Unbuffered I/O Benchmark (Physical NVMe Flash Read)
    # Open file handle with FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS
    handle = win32file.CreateFile(
        str(tensor_path),
        win32file.GENERIC_READ,
        win32file.FILE_SHARE_READ,
        None,
        win32file.OPEN_EXISTING,
        win32file.FILE_FLAG_NO_BUFFERING | win32file.FILE_FLAG_RANDOM_ACCESS,
        None,
    )

    cold_latencies_us = []
    # In unbuffered I/O, file offset and buffer size must be sector-aligned (4096 bytes)
    # Each record is approximately 6144 bytes of tensors + payload, so reading 8192 bytes (2 sectors)
    read_size = 8192
    max_sector_idx = (tensor_file_size - read_size) // SECTOR_SIZE

    sector_indices = [random.randint(0, max_sector_idx) for _ in range(num_queries)]

    for sec_idx in sector_indices:
        file_offset = sec_idx * SECTOR_SIZE
        t0 = time.perf_counter()
        win32file.SetFilePointer(handle, file_offset, win32file.FILE_BEGIN)
        _, data = win32file.ReadFile(handle, read_size, None)
        dur = (time.perf_counter() - t0) * 1_000_000.0
        cold_latencies_us.append(dur)

    win32file.CloseHandle(handle)

    warm_arr = np.array(warm_latencies_us)
    cold_arr = np.array(cold_latencies_us)

    print(f"\n====================== BENCHMARK COMPARISON ======================")
    print(f"Metric                 Warm Cache (mmap/RAM)    Physical NVMe Flash (Direct I/O)")
    print(f"------------------------------------------------------------------")
    print(f"P50 Latency:           {np.percentile(warm_arr, 50):8.2f} us ({np.percentile(warm_arr, 50)/1000:6.3f} ms)    {np.percentile(cold_arr, 50):8.2f} us ({np.percentile(cold_arr, 50)/1000:6.3f} ms)")
    print(f"P90 Latency:           {np.percentile(warm_arr, 90):8.2f} us ({np.percentile(warm_arr, 90)/1000:6.3f} ms)    {np.percentile(cold_arr, 90):8.2f} us ({np.percentile(cold_arr, 90)/1000:6.3f} ms)")
    print(f"P95 Latency:           {np.percentile(warm_arr, 95):8.2f} us ({np.percentile(warm_arr, 95)/1000:6.3f} ms)    {np.percentile(cold_arr, 95):8.2f} us ({np.percentile(cold_arr, 95)/1000:6.3f} ms)")
    print(f"P99 Latency:           {np.percentile(warm_arr, 99):8.2f} us ({np.percentile(warm_arr, 99)/1000:6.3f} ms)    {np.percentile(cold_arr, 99):8.2f} us ({np.percentile(cold_arr, 99)/1000:6.3f} ms)")
    print(f"Max Latency:           {np.max(warm_arr):8.2f} us ({np.max(warm_arr)/1000:6.3f} ms)    {np.max(cold_arr):8.2f} us ({np.max(cold_arr)/1000:6.3f} ms)")
    print(f"Mean Latency:          {np.mean(warm_arr):8.2f} us ({np.mean(warm_arr)/1000:6.3f} ms)    {np.mean(cold_arr):8.2f} us ({np.mean(cold_arr)/1000:6.3f} ms)")
    print(f"Throughput (QPS):      {num_queries / (np.sum(warm_arr)/1e6):10.1f} lookups/s   {num_queries / (np.sum(cold_arr)/1e6):10.1f} lookups/s")
    print(f"==================================================================")


if __name__ == "__main__":
    run_comparison(target_dir="d:/memory/data")
