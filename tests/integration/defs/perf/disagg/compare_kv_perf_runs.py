#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare disaggregated perf benchmark results across two runs.

Reads result.json from each (config, concurrency) pair across two slurm_logs
directories and outputs a single CSV with metrics from both runs side-by-side.

Usage:
    python compare_kv_perf_runs.py \
        --run0 /path/to/slurm_logs_0 \
        --run1 /path/to/slurm_logs \
        -o comparison.csv
"""

import argparse
import csv
import json
import os
import sys


# (json key, short display name)
METRICS = [
    ("median_ttft_ms", "ttft"),
    ("median_tpot_ms", "tpot"),
    ("median_itl_ms", "itl"),
    ("median_e2el_ms", "e2el"),
    ("output_throughput", "output_tput"),
    ("request_throughput", "request_tput"),
]


def discover_configs(slurm_dir):
    """Return dict of {(base_config, backend, concurrency): result.json path}.

    base_config strips 'disagg_perf_' prefix and '-cpp'/'-python' suffix.
    backend is 'cpp' or 'python'.
    """
    results = {}
    if not os.path.isdir(slurm_dir):
        return results

    for entry in sorted(os.listdir(slurm_dir)):
        full_path = os.path.join(slurm_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if entry.endswith("_ERROR"):
            continue

        config_name = entry.replace("disagg_perf_", "")

        # Determine backend and strip it from config name
        if config_name.endswith("-cpp"):
            backend = "cpp"
            config_name = config_name[:-4]
        elif config_name.endswith("-python"):
            backend = "python"
            config_name = config_name[:-7]
        else:
            continue

        # Find concurrency_* subdirs
        for sub in sorted(os.listdir(full_path)):
            if not sub.startswith("concurrency_"):
                continue
            result_path = os.path.join(full_path, sub, "result.json")
            if not os.path.isfile(result_path):
                continue
            try:
                concurrency = int(sub.replace("concurrency_", ""))
            except ValueError:
                continue
            results[(config_name, backend, concurrency)] = result_path

    return results


def load_metrics(result_path):
    """Load metrics from a result.json file."""
    try:
        with open(result_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: failed to read {result_path}: {e}", file=sys.stderr)
        return None

    row = {}
    for key, _ in METRICS:
        row[key] = data.get(key)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Compare disagg perf results across two runs."
    )
    parser.add_argument("--run0", required=True, help="Path to first slurm_logs dir")
    parser.add_argument("--run1", required=True, help="Path to second slurm_logs dir")
    parser.add_argument("--label0", default="run0", help="Label for first run")
    parser.add_argument("--label1", default="run1", help="Label for second run")
    parser.add_argument(
        "--output", "-o", default="kv_perf_comparison.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    configs0 = discover_configs(args.run0)
    configs1 = discover_configs(args.run1)
    print(f"Run0 ({args.label0}): {len(configs0)} entries")
    print(f"Run1 ({args.label1}): {len(configs1)} entries")

    # Collect all (config, concurrency) pairs across both runs and both backends
    all_keys = set()
    for config_name, backend, concurrency in list(configs0.keys()) + list(configs1.keys()):
        all_keys.add((config_name, concurrency))
    all_keys = sorted(all_keys)
    print(f"Total unique (config, concurrency) pairs: {len(all_keys)}")

    backends = ["cpp", "python"]
    # Column groups: cpp_run0, cpp_run1, cpp_delta, python_run0, python_run1, python_delta
    groups = []
    for b in backends:
        groups.append((b, args.label0))
        groups.append((b, args.label1))
        groups.append((b, "delta_pct"))

    num_metrics = len(METRICS)

    # Row 1: grouping header
    group_header = ["", ""]
    for b, label in groups:
        group_header.append(f"{b}_{label}")
        group_header.extend([""] * (num_metrics - 1))

    # Row 2: metric names
    metric_header = ["config", "concurrency"]
    for _ in groups:
        for _, short in METRICS:
            metric_header.append(short)

    rows = []
    for config_name, concurrency in all_keys:
        row = [config_name, concurrency]

        # Load metrics for each (backend, run) combo
        data = {}
        for b in backends:
            key = (config_name, b, concurrency)
            data[(b, args.label0)] = load_metrics(configs0[key]) if key in configs0 else None
            data[(b, args.label1)] = load_metrics(configs1[key]) if key in configs1 else None

        for b in backends:
            v0_data = data[(b, args.label0)]
            v1_data = data[(b, args.label1)]

            # run0 values
            for key, _ in METRICS:
                val = v0_data.get(key) if v0_data else None
                row.append(f"{val:.4f}" if val is not None else "")

            # run1 values
            for key, _ in METRICS:
                val = v1_data.get(key) if v1_data else None
                row.append(f"{val:.4f}" if val is not None else "")

            # delta %
            for key, _ in METRICS:
                v0 = v0_data.get(key) if v0_data else None
                v1 = v1_data.get(key) if v1_data else None
                if v0 is not None and v1 is not None and v0 != 0:
                    row.append(f"{(v1 - v0) / v0 * 100:.2f}")
                else:
                    row.append("")

        rows.append(row)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(group_header)
        writer.writerow(metric_header)
        writer.writerows(rows)

    print(f"\nWritten {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
