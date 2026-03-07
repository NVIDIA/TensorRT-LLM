#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Parse nsys report and extract kernel times within the 'benchmark' NVTX range.

Usage:
    # First, generate nsys report with SQLite export
    nsys profile -t cuda,nvtx -o report --force-overwrite true --export=sqlite \
        python bench_nvfp4_moe.py --moe_backend CUTEDSL --enable_cudagraph

    # Then parse the report
    python parse_nsys_report.py report.sqlite

    # Or specify a different NVTX range name
    python parse_nsys_report.py report.sqlite --nvtx-range "my_range"
"""

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path


def parse_nsys_report(sqlite_path: str, nvtx_range_name: str = "benchmark", verbose: bool = False):
    """Parse nsys SQLite report and extract kernel times within NVTX range."""
    if not Path(sqlite_path).exists():
        print(f"Error: File not found: {sqlite_path}")
        return

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # List available tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    # Find NVTX table
    nvtx_table = None
    for table in ["NVTX_EVENTS", "nvtx_events", "NVTX_RANGE", "nvtx_range"]:
        if table in tables:
            nvtx_table = table
            break
    if nvtx_table is None:
        nvtx_tables = [t for t in tables if "nvtx" in t.lower()]
        if nvtx_tables:
            nvtx_table = nvtx_tables[0]
        else:
            print("Error: No NVTX table found")
            conn.close()
            return

    # Get NVTX table schema
    cursor.execute(f"PRAGMA table_info({nvtx_table})")
    columns = [row[1] for row in cursor.fetchall()]

    # Find column names
    name_col = next(
        (col for col in ["text", "name", "eventName", "rangeName"] if col in columns), None
    )
    start_col = next(
        (col for col in ["start", "startNs", "timestamp", "begin"] if col in columns), None
    )
    end_col = next((col for col in ["end", "endNs", "duration"] if col in columns), None)

    if name_col is None:
        print("Error: Could not find name column")
        conn.close()
        return

    # Query NVTX ranges
    cursor.execute(f"SELECT * FROM {nvtx_table} WHERE {name_col} LIKE '%{nvtx_range_name}%'")
    nvtx_rows = cursor.fetchall()

    if not nvtx_rows:
        print(f"Error: No NVTX range found with name '{nvtx_range_name}'")
        conn.close()
        return

    col_indices = {col: idx for idx, col in enumerate(columns)}
    benchmark_start = None
    benchmark_end = None

    for row in nvtx_rows:
        name = row[col_indices[name_col]]
        if nvtx_range_name in str(name):
            if start_col and start_col in col_indices:
                benchmark_start = row[col_indices[start_col]]
            if end_col and end_col in col_indices:
                end_val = row[col_indices[end_col]]
                benchmark_end = benchmark_start + end_val if end_col == "duration" else end_val
            break

    if benchmark_start is None or benchmark_end is None:
        print("Error: Could not determine benchmark time range")
        conn.close()
        return

    # Find kernel table
    kernel_table = None
    for table in [
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUDA_KERNEL",
        "cuda_kernel",
        "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
        "kernels",
    ]:
        if table in tables:
            kernel_table = table
            break
    if kernel_table is None:
        kernel_tables = [t for t in tables if "kernel" in t.lower()]
        if kernel_tables:
            kernel_table = kernel_tables[0]
        else:
            print("Error: No kernel table found")
            conn.close()
            return

    cursor.execute(f"PRAGMA table_info({kernel_table})")
    kernel_columns = [row[1] for row in cursor.fetchall()]

    # Find string table
    string_lookup = {}
    for table in ["StringIds", "StringTable", "TARGET_INFO_CUDA_STRING_ID", "strings"]:
        if table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            string_cols = [row[1] for row in cursor.fetchall()]
            string_id_col = next(
                (col for col in ["id", "stringId", "Id"] if col in string_cols), None
            )
            string_value_col = next(
                (col for col in ["value", "string", "name", "text"] if col in string_cols), None
            )
            if string_id_col and string_value_col:
                cursor.execute(f"SELECT {string_id_col}, {string_value_col} FROM {table}")
                for row in cursor.fetchall():
                    string_lookup[row[0]] = row[1]
            break

    # Find kernel column names
    kernel_name_col = next(
        (
            col
            for col in ["shortName", "demangledName", "mangledName", "name"]
            if col in kernel_columns
        ),
        None,
    )
    kernel_start_col = next(
        (col for col in ["start", "startNs", "timestamp"] if col in kernel_columns), None
    )
    kernel_end_col = next((col for col in ["end", "endNs"] if col in kernel_columns), None)
    kernel_duration_col = next(
        (col for col in ["duration", "durationNs"] if col in kernel_columns), None
    )

    if kernel_name_col is None or kernel_start_col is None:
        print("Error: Could not find required kernel columns")
        conn.close()
        return

    kernel_col_indices = {col: idx for idx, col in enumerate(kernel_columns)}

    # First, try to get kernels within benchmark time range (works with --cuda-graph-trace=node)
    cursor.execute(f"""
        SELECT * FROM {kernel_table}
        WHERE {kernel_start_col} >= {benchmark_start} AND {kernel_start_col} <= {benchmark_end}
        ORDER BY {kernel_start_col}
    """)
    kernel_rows = cursor.fetchall()

    # If no kernels found, try CUDA Graph capture phase
    if not kernel_rows:
        cursor.execute(
            f"SELECT MIN({kernel_start_col}), MAX({kernel_start_col}) FROM {kernel_table}"
        )
        kernel_time_range = cursor.fetchone()
        kernel_min, kernel_max = kernel_time_range

        if "graphId" in kernel_columns:
            cursor.execute(f"SELECT DISTINCT graphId FROM {kernel_table} WHERE graphId > 0")
            graph_ids = [row[0] for row in cursor.fetchall()]
            if graph_ids:
                graph_id = max(graph_ids)
                cursor.execute(
                    f"SELECT * FROM {kernel_table} WHERE graphId = {graph_id} ORDER BY {kernel_start_col}"
                )
                kernel_rows = cursor.fetchall()

        # Fallback: get kernels from last batch
        if not kernel_rows:
            cursor.execute(f"SELECT * FROM {kernel_table} ORDER BY {kernel_start_col}")
            all_kernels = cursor.fetchall()
            if all_kernels:
                last_kernel_time = all_kernels[-1][kernel_col_indices[kernel_start_col]]
                capture_window = 10_000_000
                capture_start = last_kernel_time - capture_window
                kernel_rows = [
                    k
                    for k in all_kernels
                    if k[kernel_col_indices[kernel_start_col]] >= capture_start
                ]

    if not kernel_rows:
        print("No kernels found")
        conn.close()
        return

    # First pass: collect all kernel instances with their durations, ordered by start time
    kernel_instances = []
    for row in kernel_rows:
        name_id = row[kernel_col_indices[kernel_name_col]]
        name = string_lookup.get(name_id, name_id) if isinstance(name_id, int) else name_id
        start = row[kernel_col_indices[kernel_start_col]]

        duration = None
        if kernel_duration_col and kernel_duration_col in kernel_col_indices:
            duration = row[kernel_col_indices[kernel_duration_col]]
        elif kernel_end_col and kernel_end_col in kernel_col_indices:
            duration = row[kernel_col_indices[kernel_end_col]] - start

        if duration:
            kernel_instances.append({"name": name, "start": start, "duration": duration})

    # Sort by start time
    kernel_instances.sort(key=lambda x: x["start"])

    # Count occurrences for each kernel name
    name_counts = defaultdict(int)
    for k in kernel_instances:
        name_counts[k["name"]] += 1

    # Find iteration count using GCD of all counts
    all_counts = list(name_counts.values())
    if all_counts:
        from functools import reduce
        from math import gcd

        iteration_count = reduce(gcd, all_counts)
    else:
        iteration_count = 1

    # Track occurrence index for each kernel name
    name_occurrence = defaultdict(int)
    kernel_times = defaultdict(lambda: {"count": 0, "total_ns": 0})

    for k in kernel_instances:
        name = k["name"]
        duration = k["duration"]

        # Check if this kernel appears multiple times per iteration
        calls_per_iter = name_counts[name] // iteration_count if iteration_count > 0 else 1

        if calls_per_iter > 1:
            # Add suffix based on position within iteration
            suffix_idx = name_occurrence[name] % calls_per_iter
            display_name = f"{name}_{suffix_idx}"
        else:
            display_name = name

        name_occurrence[name] += 1
        kernel_times[display_name]["count"] += 1
        kernel_times[display_name]["total_ns"] += duration

    # Print results
    total_kernel_time_ns = 0
    total_kernel_count = 0

    sorted_kernels = sorted(kernel_times.items(), key=lambda x: x[1]["total_ns"], reverse=True)

    print(f"\n{'Count':>8} {'Total (us)':>12} {'Avg (us)':>12}  Kernel Name")
    print("-" * 120)

    for name, stats in sorted_kernels:
        count = stats["count"]
        total_ns = stats["total_ns"]
        avg_ns = total_ns / count if count > 0 else 0
        total_us = total_ns / 1000
        avg_us = avg_ns / 1000
        total_kernel_time_ns += total_ns
        total_kernel_count += count
        name_str = str(name) if name is not None else "unknown"
        print(f"{count:>8} {total_us:>12.2f} {avg_us:>12.2f}  {name_str}")

    print("-" * 120)
    total_kernel_time_us = total_kernel_time_ns / 1000
    total_kernel_time_ms = total_kernel_time_us / 1000
    print(f"{total_kernel_count:>8} {total_kernel_time_us:>12.2f}              TOTAL")

    benchmark_duration_ms = (benchmark_end - benchmark_start) / 1e6
    print(f"\nBenchmark duration: {benchmark_duration_ms:.3f} ms")
    print(f"Kernel time: {total_kernel_time_ms:.3f} ms, Unique kernels: {len(kernel_times)}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Parse nsys SQLite report and extract kernel times"
    )
    parser.add_argument("sqlite_path", type=str, help="Path to nsys SQLite report file")
    parser.add_argument(
        "--nvtx-range", type=str, default="benchmark", help="NVTX range name (default: benchmark)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    parse_nsys_report(args.sqlite_path, args.nvtx_range, args.verbose)


if __name__ == "__main__":
    main()
