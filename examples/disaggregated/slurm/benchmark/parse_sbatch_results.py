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
from pathlib import Path

JOB_ID_SPEC_SEQ = re.compile(r'(\d+)\.\.(\d+)')
BENCH_LOG_SUB_FOLDER = re.compile(r"benchmark-(\d+)-(\d+)")
TPOT_MEAN_LINE = re.compile(r"Mean TPOT \(ms\):\s+(\d+\.\d+)")
TPOT_MEDIAN_LINE = re.compile(r"Median TPOT \(ms\):\s+(\d+\.\d+)")
TP_YAML = re.compile(r"tensor_parallel_size: (\d+)")
EP_YAML = re.compile(r"moe_expert_parallel_size: (\d+)")
CP_YAML = re.compile(r"context_parallel_size: (\d+)")
PP_YAML = re.compile(r"pipeline_parallel_size: (\d+)")


def parse_benchmark_log(log_file):
    """
    Parse a benchmark.log file and extract timing information.

    Returns:
        dict: Configuration key mapped to rank time.
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()

    timing_lines = [
        (i, line.strip()) for i, line in enumerate(lines)
        if "-----Time per Output Token (excl. 1st token)------" in line
    ]
    if len(timing_lines) != 1:
        print(
            f"Warning: {log_file} does not contain timing information. No line contained the TPOT header.",
            file=sys.stderr)
        return None

    i_line, timing_line = timing_lines[0]
    mean_match = TPOT_MEAN_LINE.search(lines[i_line + 1])
    median_match = TPOT_MEDIAN_LINE.search(lines[i_line + 2])
    if not mean_match or not median_match:
        print(
            f"Warning: {log_file} does not contain timing information. "
            f"Lines {lines[i_line + 1].strip()} and {lines[i_line + 2].strip()} "
            "did not contain the TPOT information.",
            file=sys.stderr)
        return None
    tpot_mean = float(mean_match.group(1))
    tpot_median = float(median_match.group(1))

    return tpot_mean, tpot_median


def main():
    """Main function to process benchmark logs from multiple folders."""
    job_id_pattern = "(<job_id_start>..<job_id_end>|<job_id1,job_id2,job_id3>...)"
    if len(sys.argv) < 2:
        print("Usage: python parse_benchmark_logs.py " + job_id_pattern,
              file=sys.stderr)
        sys.exit(1)

    job_id_spec = sys.argv[1]
    if JOB_ID_SPEC_SEQ.match(job_id_spec):
        job_id_start, job_id_end = map(int, job_id_spec.split('..'))
        job_ids = list(range(job_id_start, job_id_end + 1))
    else:
        job_ids = job_id_spec.split(',')
        try:
            job_ids = list(map(int, job_ids))
        except ValueError:
            print(
                f"Invalid job ID spec: {job_id_spec}, "
                f"expected format: {job_id_pattern}",
                file=sys.stderr)
            sys.exit(1)

    all_results = []

    for job_id in job_ids:
        folder_path = Path(f"slurm-{job_id}")

        if not folder_path.is_dir():
            print(f"Warning: {folder_path} not found, skipping",
                  file=sys.stderr)
            continue

        # we expect the folder to contain exactly one sub-folder
        sub_folders1 = list(folder_path.iterdir())
        if len(sub_folders1) != 1:
            print(
                f"Warning: {folder_path} contains {len(sub_folders1)} sub-folders, expected 1",
                file=sys.stderr)
            continue
        sub_folder1 = sub_folders1[0]
        match = BENCH_LOG_SUB_FOLDER.match(sub_folder1.name)
        if not match:
            print(
                f"Warning: {sub_folder1.name} is not a valid benchmark log sub-folder name",
                file=sys.stderr)
            continue
        isl = int(match.group(1))
        # we expect the sub-folder to contain exactly one sub-folder
        sub_folders2 = list(sub_folder1.iterdir())
        if len(sub_folders2) != 1:
            print(
                f"Warning: {sub_folder1} contains {len(sub_folders2)} sub-folders, expected 1",
                file=sys.stderr)
            continue
        sub_folder2 = sub_folders2[0]
        log_file = sub_folder2 / 'benchmark.log'
        if not log_file.exists():
            print(f"Warning: {log_file} not found, skipping", file=sys.stderr)
            continue
        config_file = sub_folder2 / 'gen_config.yaml'
        if not config_file.exists():
            print(f"Warning: {config_file} not found, skipping",
                  file=sys.stderr)
            continue
        with open(config_file, 'r') as f:
            config = f.read()
        tp_match = TP_YAML.search(config)
        ep_match = EP_YAML.search(config)
        cp_match = CP_YAML.search(config)
        pp_match = PP_YAML.search(config)
        if not tp_match or not ep_match or not cp_match or not pp_match:
            print(
                f"Warning: {config_file} does not contain configuration information",
                file=sys.stderr)
            continue
        tp = int(tp_match.group(1))
        ep = int(ep_match.group(1))
        cp = int(cp_match.group(1))
        pp = int(pp_match.group(1))

        config_key = (tp, cp, ep, pp, isl)

        print(f"Processing {log_file} for config {config_key}...",
              file=sys.stderr)
        data = parse_benchmark_log(log_file)
        if data is None:
            continue
        for i, d in enumerate(data):
            if len(all_results) <= i:
                all_results.append(dict())
            all_results[i][config_key] = d

    # Write results to CSV
    for i, results in enumerate(all_results):
        all_isls = set(config_key[-1] for config_key in results)
        all_isls = sorted(all_isls)

        all_configs = set(config_key[:-1] for config_key in results)
        all_configs = sorted(all_configs)

        fieldnames = [
            'tp',
            'kvp',
            'ep',
            'pp',
        ] + all_isls
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()

        for config in all_configs:
            row = {
                'tp': config[0],
                'kvp': config[1],
                'ep': config[2],
                'pp': config[3]
            }
            for isl in all_isls:
                config_key = config + (isl, )
                if config_key in results:
                    row[isl] = results[config_key]
            writer.writerow(row)


if __name__ == '__main__':
    main()
