# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys

import yaml

# Metrics where larger is better
MAXIMIZE_METRICS = [
    "d_seq_throughput",
    "d_token_throughput",
    "d_total_token_throughput",
    "d_user_throughput",
    "d_mean_tpot",
    "d_median_tpot",
    "d_p99_tpot",
]

# Metrics where smaller is better
MINIMIZE_METRICS = [
    "d_mean_ttft",
    "d_median_ttft",
    "d_p99_ttft",
    "d_mean_itl",
    "d_median_itl",
    "d_p99_itl",
    "d_mean_e2el",
    "d_median_e2el",
    "d_p99_e2el",
]


def should_skip_execution():
    disagg_type = os.getenv("DISAGG_SERVING_TYPE", "")
    if (
        disagg_type.startswith("GEN")
        or disagg_type.startswith("CTX")
        or disagg_type == "DISAGG_SERVER"
    ):
        return True
    return False


def find_yaml_files(job_workspace, filename):
    yaml_files = []
    for root, dirs, files in os.walk(job_workspace):
        for file in files:
            if file == filename:
                yaml_files.append(os.path.join(root, file))
    return yaml_files


def read_yaml_data(yaml_files):
    all_data = []
    for file_path in yaml_files:
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
                if data:
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return all_data


def get_metric_keys():
    metric_keys = set()
    for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
        metric_suffix = metric[2:]  # Strip "d_" prefix
        metric_keys.add(metric)
        metric_keys.add(f"d_baseline_{metric_suffix}")
        metric_keys.add(f"d_threshold_post_merge_{metric_suffix}")
        metric_keys.add(f"d_threshold_pre_merge_{metric_suffix}")
    return metric_keys


def print_perf_data(data):
    print("=== Metrics ===")
    for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
        if metric in data:
            value = data.get(metric, "N/A")
            print(f'"{metric}": {value}')

    metric_keys = get_metric_keys()
    print("\n=== Config ===")
    config_keys = sorted([key for key in data.keys() if key not in metric_keys])
    for key in config_keys:
        value = data[key]
        print(f'"{key}": {value}')


def print_regression_data(data):
    if "s_regression_info" in data:
        print("=== Regression Info ===")
        print(f"{data['s_regression_info']}")

    metric_keys = get_metric_keys()
    is_post_merge = data.get("b_is_post_merge", False)

    print("=== Metrics ===")
    for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
        metric_suffix = metric[2:]  # Strip "d_" prefix
        baseline_key = f"d_baseline_{metric_suffix}"
        if is_post_merge:
            threshold_key = f"d_threshold_post_merge_{metric_suffix}"
        else:
            threshold_key = f"d_threshold_pre_merge_{metric_suffix}"
        # Only print if at least one of the keys exists
        if metric in data or baseline_key in data or threshold_key in data:
            value = data.get(metric, "N/A")
            baseline = data.get(baseline_key, "N/A")
            threshold = data.get(threshold_key, "N/A")
            # Calculate percentage difference between value and baseline
            # Positive percentage means better perf, negative means regression
            if (
                isinstance(value, (int, float))
                and isinstance(baseline, (int, float))
                and baseline != 0
            ):
                if metric in MAXIMIZE_METRICS:
                    # Larger is better: value > baseline is positive (better)
                    percentage = (value - baseline) / baseline * 100
                else:
                    # Smaller is better: value < baseline is positive (better)
                    percentage = (baseline - value) / baseline * 100
                percentage_str = f"{percentage:+.2f}%"
            else:
                percentage_str = "N/A"
            print(
                f'"{metric}": {value}, "{baseline_key}": {baseline}, '
                f'"{threshold_key}": {threshold}, "diff": {percentage_str}'
            )

    print("\n=== Config ===")
    config_keys = sorted([key for key in data.keys() if key not in metric_keys])
    for key in config_keys:
        if key == "s_regression_info":
            continue
        value = data[key]
        print(f'"{key}": {value}')


def main():
    if should_skip_execution():
        print("Skipping check_perf_regression.py due to DISAGG_SERVING_TYPE")
        return 0

    job_workspace = sys.argv[1]

    if not os.path.isdir(job_workspace):
        print(f"Skipping perf regression check since {job_workspace} is not a valid directory.")
        return 0

    perf_data_files = find_yaml_files(job_workspace, "perf_data.yaml")
    all_perf_data = read_yaml_data(perf_data_files)
    print(f"Found {len(all_perf_data)} perf data")
    for i, data in enumerate(all_perf_data):
        print(f"\n{'=' * 60}")
        print(f"Perf Data #{i + 1}")
        print("=" * 60)
        print_perf_data(data)

    print(f"\n{'=' * 60}\n")

    regression_files = find_yaml_files(job_workspace, "regression.yaml")
    all_regression_data = read_yaml_data(regression_files)
    print(f"Found {len(all_regression_data)} regression data")
    for i, data in enumerate(all_regression_data):
        print(f"\n{'=' * 60}")
        print(f"Regression Data #{i + 1}")
        print("=" * 60)
        print_regression_data(data)

    # Split regression data into post-merge and pre-merge categories
    post_merge_regressions = [
        data for data in all_regression_data if data.get("b_is_post_merge", False)
    ]
    pre_merge_regressions = [
        data for data in all_regression_data if not data.get("b_is_post_merge", False)
    ]

    if len(all_regression_data) == 0:
        print("\n No regression data found. Perf check is successful.")
        return 0

    if len(pre_merge_regressions) > 0:
        print(
            f"\n Warning: Found {len(pre_merge_regressions)} pre-merge regression data. "
            "But we don't fail the check temporarily."
        )

    if len(post_merge_regressions) > 0:
        print(
            f"\n Error: Found {len(post_merge_regressions)} post-merge regression data. Perf check is failed."
        )
        return 1

    print("\n No post-merge regression data found. Perf check is successful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
