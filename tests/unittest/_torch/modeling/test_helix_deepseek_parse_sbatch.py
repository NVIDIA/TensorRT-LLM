#!/usr/bin/env python3
"""
Script to parse benchmark.log files from multiple folders and calculate median timing values.

Usage:
    python parse_benchmark_logs.py <folder1> [folder2] [folder3] ...

The script looks for benchmark.log in each folder and extracts timing information
for different configurations (dense/moe, ctx_len, tp, kvp, ep).
It verifies all ranks are present and calculates the median time for each configuration.
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

JOB_ID_SPEC_SEQ = re.compile(r'(\d+)\.\.(\d+)')


def parse_benchmark_log(log_file):
    """
    Parse a benchmark.log file and extract timing information.

    Returns:
        dict: Configuration keys mapped to rank times and total GPU count.
    """
    # Pattern to match lines like:
    # Rank <X> <Y>-GPU: time taken for 10 steps of <dense|moe>,
    # ctx_len/ctx_len_per_gpu/tp/kvp/ep: 65536/65536/8/1/2: 0.08419919013977051 s, ...
    pattern = re.compile(r'Rank\s+(\d+)\s+(\d+)-GPU:\s+'
                         r'time taken for 10 steps of\s+(dense|moe),\s+'
                         r'ctx_len/ctx_len_per_gpu/tp/kvp/ep:\s+'
                         r'(\d+)/(\d+)/(\d+)/(\d+)/(\d+):\s+'
                         r'([\d.]+)\s+ms,\s+expected TPOT:\s+([\d.]+)\s+ms')

    data = defaultdict(lambda: {'ranks': {}, 'total_gpus': None})

    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                rank = int(match.group(1))
                total_gpus = int(match.group(2))
                type_name = match.group(3)
                ctx_len = int(match.group(4))
                ctx_len_per_gpu = int(match.group(5))
                tp = int(match.group(6))
                kvp = int(match.group(7))
                ep = int(match.group(8))
                time_val = float(match.group(9))
                tpot_val = float(match.group(10))

                # Create a key for this configuration
                config_key = (type_name, ctx_len, ctx_len_per_gpu, tp, kvp, ep)

                # Store the rank's time value
                data[config_key]['ranks'][rank] = (time_val, tpot_val)
                data[config_key]['total_gpus'] = total_gpus

    return data


def calculate_stats(data):
    """
    Calculate medians for configurations with complete rank sets.

    Args:
        data: Dictionary mapping configuration keys to rank data.

    Returns:
        list: List of dictionaries containing configuration and median time.
    """
    results = []

    for config_key, config_data in data.items():
        type_name, ctx_len, ctx_len_per_gpu, tp, kvp, ep = config_key
        ranks = config_data['ranks']
        total_gpus = config_data['total_gpus']

        if total_gpus is None:
            print(f"Warning: Missing total_gpus for config {config_key}",
                  file=sys.stderr)
            continue
        if total_gpus <= 0:
            print(
                f"Warning: Total GPUs {total_gpus} for config {config_key} is invalid",
                file=sys.stderr)
            continue

        # Check if all ranks from 0 to total_gpus-1 are present
        expected_ranks = set(range(total_gpus))
        actual_ranks = set(ranks.keys())

        if expected_ranks == actual_ranks:
            # All ranks present, calculate median
            times = sorted(ranks[r][1] for r in sorted(ranks.keys()))
            print(
                f"All ranks present for config {config_key}: times {' '.join(f'{t:.2f}' for t in times)}"
            )
            min_time = times[0]
            if len(times) % 2 == 0:
                median_time = (times[len(times) // 2] +
                               times[len(times) // 2 - 1]) / 2.
            else:
                median_time = times[len(times) // 2]

            results.append({
                'type': type_name,
                'ctx_len': ctx_len,
                'ctx_len_per_gpu': ctx_len_per_gpu,
                'tp': tp,
                'kvp': kvp,
                'ep': ep,
                'median_time_ms': median_time,
                'min_time_ms': min_time
            })
        else:
            missing_ranks = expected_ranks - actual_ranks
            print(
                f"Warning: Missing ranks {missing_ranks} for config {config_key}",
                file=sys.stderr)

    return results


def main():
    """Main function to process benchmark logs from multiple folders."""
    if len(sys.argv) < 2:
        print(
            "Usage: python parse_benchmark_logs.py <job_id_spec|list of folders>",
            file=sys.stderr)
        sys.exit(1)

    job_id_spec = sys.argv[1]
    if JOB_ID_SPEC_SEQ.match(job_id_spec):
        job_id_start, job_id_end = map(int, job_id_spec.split('..'))
        job_ids = list(range(job_id_start, job_id_end + 1))
        folders = [Path(f"slurm-{job_id}") for job_id in job_ids]
    else:
        job_ids = job_id_spec.split(',')
        try:
            job_ids = list(map(int, job_ids))
            folders = [Path(f"slurm-{job_id}") for job_id in job_ids]
        except ValueError:
            folders = [Path(folder) for folder in sys.argv[1:]]

    folders = [folder for folder in folders if folder.is_dir()]
    if len(folders) == 0:
        print(
            "No folders found. Usage: python parse_benchmark_logs.py <job_id_spec|list of folders>",
            file=sys.stderr)
        sys.exit(1)

    all_results = []

    for folder in folders:
        folder_path = Path(folder)
        log_file = folder_path / 'benchmark.log'

        if not log_file.exists():
            print(f"Warning: {log_file} not found, skipping", file=sys.stderr)
            continue

        print(f"Processing {log_file}...", file=sys.stderr)
        data = parse_benchmark_log(log_file)
        results = calculate_stats(data)
        all_results.extend(results)

    # Write results to CSV
    if all_results:
        fieldnames = [
            'type', 'ctx_len', 'ctx_len_per_gpu', 'tp', 'kvp', 'ep',
            'median_time_ms', 'min_time_ms'
        ]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()

        # Sort results for consistent output
        all_results.sort(
            key=lambda x: (x['type'], x['ctx_len'], x['tp'], x['kvp'], x['ep']))
        writer.writerows(all_results)
    else:
        print("No results found", file=sys.stderr)


if __name__ == '__main__':
    main()
