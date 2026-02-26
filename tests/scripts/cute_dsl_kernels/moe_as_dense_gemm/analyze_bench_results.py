#!/usr/bin/env python3
"""Analyze benchmark results from bench_fc1.log."""

import csv
import re
from pathlib import Path
from typing import Dict


def parse_bench_log(log_file: str) -> Dict[str, Dict[int, float]]:
    """Parse benchmark log file and extract execution times.

    Returns:
        Dict mapping config to {M: execution_time_us}
    """
    results = {}
    current_config = None
    current_m = None

    with open(log_file, "r") as f:
        for line in f:
            # Match config line: "--- FC1 with mma_tiler_mn=128,256, cluster_shape_mn=1,1 ---"
            config_match = re.search(r"mma_tiler_mn=(\d+,\d+).*cluster_shape_mn=(\d+,\d+)", line)
            if config_match and "---" in line:
                mma = config_match.group(1)
                cluster = config_match.group(2)
                current_config = f"MMA_{mma}_Cluster_{cluster}"
                if current_config not in results:
                    results[current_config] = {}
                continue

            # Match M value: "Running FC1 with M=128, mma_tiler_mn=..."
            m_match = re.search(r"Running FC1 with M=(\d+)", line)
            if m_match:
                current_m = int(m_match.group(1))
                continue

            # Match execution time: "Execution time: 117.80 us"
            time_match = re.search(r"Execution time:\s+([\d.]+)\s+us", line)
            if time_match and current_config and current_m is not None:
                exec_time = float(time_match.group(1))
                results[current_config][current_m] = exec_time
                current_m = None

    return results


def print_summary_table(results: Dict[str, Dict[int, float]]):
    """Print a summary table of results."""
    if not results:
        print("No results found!")
        return

    # Get all M values (should be same for all configs)
    all_m_values = sorted(list(results[list(results.keys())[0]].keys()))

    # Print header
    configs = sorted(results.keys())
    print("\n" + "=" * 100)
    print("FC1 Benchmark Results Summary")
    print("=" * 100)
    print(f"\n{'M':<6}", end="")
    for config in configs:
        print(f"{config:<35}", end="")
    print()
    print("-" * 100)

    # Print each row
    for m in all_m_values:
        print(f"{m:<6}", end="")
        for config in configs:
            time = results[config].get(m, 0.0)
            print(f"{time:>8.2f} us                       ", end="")
        print()

    print("-" * 100)


def find_best_configs(results: Dict[str, Dict[int, float]]):
    """Find best configuration for each M value."""
    if not results:
        return

    all_m_values = sorted(list(results[list(results.keys())[0]].keys()))
    configs = sorted(results.keys())

    print("\n" + "=" * 100)
    print("Best Configuration for Each M Value")
    print("=" * 100)
    print(f"\n{'M':<6}{'Best Config':<40}{'Time (us)':<15}{'Speedup vs Worst'}")
    print("-" * 100)

    for m in all_m_values:
        times = [(config, results[config].get(m, float("inf"))) for config in configs]
        times.sort(key=lambda x: x[1])

        best_config, best_time = times[0]
        worst_time = times[-1][1]
        speedup = worst_time / best_time if best_time > 0 else 1.0

        print(f"{m:<6}{best_config:<40}{best_time:>8.2f}        {speedup:>5.2f}x")

    print("-" * 100)


def compute_statistics(results: Dict[str, Dict[int, float]]):
    """Compute statistics for each configuration."""
    print("\n" + "=" * 100)
    print("Statistics by Configuration")
    print("=" * 100)
    print(f"\n{'Config':<40}{'Min (us)':<12}{'Max (us)':<12}{'Avg (us)':<12}{'Median (us)'}")
    print("-" * 100)

    for config in sorted(results.keys()):
        times = sorted(results[config].values())
        if not times:
            continue

        min_time = min(times)
        max_time = max(times)
        avg_time = sum(times) / len(times)
        median_time = times[len(times) // 2]

        print(
            f"{config:<40}{min_time:>8.2f}    {max_time:>8.2f}    {avg_time:>8.2f}    {median_time:>8.2f}"
        )

    print("-" * 100)


def export_to_csv(results: Dict[str, Dict[int, float]], output_file: str):
    """Export results to CSV file."""
    if not results:
        print("No results to export!")
        return

    # Get all M values
    all_m_values = sorted(list(results[list(results.keys())[0]].keys()))
    configs = sorted(results.keys())

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ["M"] + configs
        writer.writerow(header)

        # Write data rows
        for m in all_m_values:
            row = [m]
            for config in configs:
                time = results[config].get(m, "")
                row.append(f"{time:.2f}" if time else "")
            writer.writerow(row)

    print(f"\nResults exported to: {output_file}")


def main():
    log_file = Path(__file__).parent / "bench_fc1.log"

    if not log_file.exists():
        print(f"Error: {log_file} not found!")
        return

    print(f"Parsing {log_file}...")
    results = parse_bench_log(log_file)

    if not results:
        print("No benchmark results found in log file!")
        return

    print(f"Found {len(results)} configurations:")
    for config in sorted(results.keys()):
        print(f"  - {config}: {len(results[config])} M values tested")

    # Print all results
    print_summary_table(results)

    # Find best configs
    find_best_configs(results)

    # Compute statistics
    compute_statistics(results)

    # Export to CSV
    csv_file = log_file.parent / "bench_fc1_results.csv"
    export_to_csv(results, csv_file)

    print("\n" + "=" * 100)
    print("Analysis Complete!")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
