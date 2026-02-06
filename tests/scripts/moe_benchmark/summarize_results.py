#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Summarize benchmark results across backends and seq_lens.

Usage:
    python summarize_results.py results/
    python summarize_results.py results/ --output summary.csv
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path


def parse_log_file(log_path: str) -> dict[str, float] | None:
    """Parse log file and extract kernel avg times (in us).

    Returns:
        Dict mapping kernel name to avg time in us, or None if parsing fails.
    """
    if not Path(log_path).exists():
        return None

    try:
        with open(log_path) as f:
            content = f.read()

        # Find the kernel table section
        # Format: "   Count   Total (us)     Avg (us)  Kernel Name"
        # Data:   "     100      8773.25        87.73  kernel_name"
        kernel_times = {}

        # Match lines with kernel data
        # Pattern: whitespace + count + total + avg + kernel_name
        pattern = r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+(.+)$"

        in_table = False
        for line in content.split("\n"):
            if "Count" in line and "Total (us)" in line and "Avg (us)" in line:
                in_table = True
                continue
            if in_table:
                if line.startswith("---"):
                    if "TOTAL" not in line:
                        continue
                    else:
                        break
                match = re.match(pattern, line)
                if match:
                    avg_us = float(match.group(3))
                    kernel_name = match.group(4).strip()
                    kernel_times[kernel_name] = avg_us

        return kernel_times if kernel_times else None

    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None


def parse_filename(filename: str) -> tuple[str, int] | None:
    """Parse filename to extract backend and seq_len."""
    # Expected format: BACKEND_seqN.log
    match = re.match(r"(\w+)_seq(\d+)\.log", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None


def simplify_kernel_name(name: str) -> str:
    """Simplify long kernel names for display."""
    # For CUTLASS kernels
    if name.startswith("device_kernel"):
        return name

    # For TRTLLM kernels (very long names)
    if "bmm_" in name:
        if "swiGlu" in name:
            return "bmm_fc1_swiglu"
        else:
            return "bmm_fc2"

    # For cute_dsl kernels
    if "Sm100BlockScaledPersistentDenseGemmKernel" in name:
        if "fc1" in name.lower():
            return "densegemm_fc1"
        elif "fc2" in name.lower():
            return "densegemm_fc2"

    # Truncate very long names
    if len(name) > 40:
        return name[:37] + "..."

    return name


def summarize_results(results_dir: str, output_file: str | None = None):
    """Summarize all benchmark results."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Directory not found: {results_dir}")
        return

    # Collect all results
    # data[backend][seq_len][kernel_name] = avg_time_us
    data = defaultdict(lambda: defaultdict(dict))
    backends = set()
    seq_lens = set()
    kernel_names_by_backend = defaultdict(set)

    log_files = list(results_path.glob("*.log"))
    print(f"Found {len(log_files)} log files\n")

    for log_file in sorted(log_files):
        parsed = parse_filename(log_file.name)
        if parsed is None:
            continue

        backend, seq_len = parsed
        backends.add(backend)
        seq_lens.add(seq_len)

        kernel_times = parse_log_file(str(log_file))
        if kernel_times:
            for kernel_name, avg_us in kernel_times.items():
                simplified_name = simplify_kernel_name(kernel_name)
                data[backend][seq_len][simplified_name] = avg_us
                kernel_names_by_backend[backend].add(simplified_name)

    if not data:
        print("No valid results found")
        return

    # Sort
    backends = sorted(backends)
    seq_lens = sorted(seq_lens)

    # Print summary table for each backend
    all_csv_lines = []

    for backend in backends:
        kernel_names = sorted(kernel_names_by_backend[backend])
        if not kernel_names:
            continue

        print("=" * 100)
        print(f"Backend: {backend}")
        print("=" * 100)

        # Header
        header = f"{'seq_len':>8}"
        for kernel in kernel_names:
            header += f" {kernel:>20}"
        header += f" {'TOTAL':>12}"
        print(header)
        print("-" * (8 + 21 * len(kernel_names) + 13))

        # CSV header for this backend
        csv_lines = [f"# {backend}"]
        csv_lines.append(",".join(["seq_len"] + kernel_names + ["TOTAL"]))

        # Data rows
        for seq_len in seq_lens:
            if seq_len not in data[backend]:
                continue

            row = f"{seq_len:>8}"
            csv_row = [str(seq_len)]
            total = 0.0

            for kernel in kernel_names:
                if kernel in data[backend][seq_len]:
                    val = data[backend][seq_len][kernel]
                    row += f" {val:>20.2f}"
                    csv_row.append(f"{val:.2f}")
                    total += val
                else:
                    row += f" {'N/A':>20}"
                    csv_row.append("N/A")

            row += f" {total:>12.2f}"
            csv_row.append(f"{total:.2f}")
            print(row)
            csv_lines.append(",".join(csv_row))

        print()
        all_csv_lines.extend(csv_lines)
        all_csv_lines.append("")

    # Save to CSV if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(all_csv_lines))
        print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument("results_dir", type=str, help="Directory containing log files")
    parser.add_argument("--output", "-o", type=str, help="Output CSV file (optional)")

    args = parser.parse_args()
    summarize_results(args.results_dir, args.output)


if __name__ == "__main__":
    main()
