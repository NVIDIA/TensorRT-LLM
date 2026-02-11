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
"""
OpenSearch database utilities
"""
import json
import os
import re
import sys
import time
from datetime import datetime

import yaml
from defs.trt_test_alternative import print_info, print_warning

_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from jenkins.scripts.open_search_db import (PERF_SANITY_PROJECT_NAME,
                                            OpenSearchDB)

POC_PROJECT_NAME = "sandbox-temp-trtllm-ci-perf-v1-test_info"
USE_POC_DB = os.environ.get("USE_POC_DB", "false").lower() == "true"
TEST_INFO_PROJECT_NAME = POC_PROJECT_NAME if USE_POC_DB else PERF_SANITY_PROJECT_NAME
MAX_QUERY_SIZE = 5000
QUERY_LOOKBACK_DAYS = 90

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

# Key metrics that determine regression (throughput metrics only)
REGRESSION_METRICS = [
    "d_seq_throughput",
    "d_token_throughput",
    "d_total_token_throughput",
    "d_user_throughput",
]

# Default threshold values for performance regression detection
POST_MERGE_THRESHOLD = 0.05
PRE_MERGE_THRESHOLD = 0.1

# Fields for scenario-only matching for recipe tests.
# Unlike regular tests that match on all config fields, recipes match only on the benchmark
# scenario, allowing the underlying config to change while still comparing against baselines
# for the same scenario.
SCENARIO_MATCH_FIELDS = [
    "s_gpu_type",
    "s_runtime",
    "s_model_name",
    "l_isl",
    "l_osl",
    "l_concurrency",
    "l_num_gpus",
]


def add_id(data):
    OpenSearchDB.add_id_of_json(data)


def get_job_info():
    """
    Get job info from environment variables
    """
    # Read environment variables
    host_node_name = os.getenv("HOST_NODE_NAME", "")
    build_id = os.getenv("BUILD_ID", "")
    build_url = os.getenv("BUILD_URL", "")
    job_name = os.getenv("JOB_NAME", "")
    global_vars_str = os.getenv("globalVars", "{}")
    try:
        global_vars = json.loads(global_vars_str)
    except Exception:
        global_vars = {}

    # Get job_url and job_id
    job_url = ""
    job_id = ""
    parents = global_vars.get("action_info", {}).get("parents", [])
    if parents:
        last_parent = parents[-1]
        job_url = last_parent.get("url", "")
        job_id = str(last_parent.get("build_number", ""))

    # Determine job type from job_url
    is_post_merge = "PostMerge" in job_url
    is_pr_job = not is_post_merge

    # Extract branch from job_url
    # Pattern: LLM/job/main/job -> branch is "main"
    branch = ""
    commit = os.getenv("gitlabCommit", "")
    if job_url:
        branch_match = re.search(r'/job/LLM/job/([^/]+)/job/', job_url)
        if branch_match:
            branch = branch_match.group(1)

    # Initialize PR-specific fields
    trigger_mr_user = ""
    trigger_mr_link = ""
    trigger_mr_id = ""
    trigger_mr_commit = ""
    artifact_url = ""
    if is_pr_job:
        # Get PR info from github_pr_api_url
        github_pr_api_url = global_vars.get("github_pr_api_url", "")
        if github_pr_api_url:
            # Extract PR ID from URL like "https://api.github.com/repos/NVIDIA/TensorRT-LLM/pulls/xxxx"
            pr_match = re.search(r"/pulls/(\d+)", github_pr_api_url)
            if pr_match:
                trigger_mr_id = pr_match.group(1)
                # Convert API URL to web URL
                trigger_mr_link = github_pr_api_url.replace(
                    "api.github.com/repos/",
                    "github.com/").replace("/pulls/", "/pull/")

        # Extract user from trigger_info
        # Pattern: target="_blank">Barry-Delaney</a><br/>Git Commit:
        trigger_info = global_vars.get("action_info",
                                       {}).get("trigger_info", "")
        # Try to extract username from patterns like 'target="_blank">username</a><br/>Git Commit:'
        user_match = re.search(r'target="_blank">([^<]+)</a><br/>Git Commit:',
                               trigger_info)
        if user_match:
            trigger_mr_user = user_match.group(1)

        # Set trigger_mr_commit to commit
        trigger_mr_commit = commit
        artifact_url = f"https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/{job_id}" if job_id else ""
    else:
        artifact_url = f"https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/{job_id}" if job_id else ""

    return {
        "b_is_baseline": False,
        "b_is_valid": True,

        # Unique identifier
        "_id": "",

        # Job Config
        "s_host_node_name": host_node_name,
        "s_build_id": build_id,
        "s_build_url": build_url,
        "s_job_name": job_name,
        "s_job_id": job_id,
        "s_job_url": job_url,
        "s_branch": branch,
        "s_commit": commit,
        "b_is_post_merge": is_post_merge,
        "b_is_pr_job": is_pr_job,
        "s_trigger_mr_user": trigger_mr_user,
        "s_trigger_mr_link": trigger_mr_link,
        "s_trigger_mr_id": trigger_mr_id,
        "s_trigger_mr_commit": trigger_mr_commit,
        "s_artifact_url": artifact_url,
        "b_is_regression": False,
    }


def get_common_values(new_data_dict, match_keys):
    """
    Find keys from match_keys where all data entries in new_data_dict have identical values.
    Returns a dict with those common key-value pairs.
    Skips entries that don't have the key or have None/empty values.
    """
    if not new_data_dict or not match_keys:
        return {}

    data_list = list(new_data_dict.values())
    if not data_list:
        return {}

    common_values_dict = {}
    for key in match_keys:
        # Collect non-None, non-empty values for this key
        values = []
        for data in data_list:
            if key in data and data[key] is not None:
                values.append(data[key])

        # Skip if no valid values found
        if len(values) != len(data_list):
            continue

        # Check if all valid values are identical
        first_value = values[0]
        if all(v == first_value for v in values):
            common_values_dict[key] = first_value

    return common_values_dict


def match(history_data, new_data, match_keys):
    """
    Check if the server and client config of history data match the new data
    """

    def is_empty(value):
        return value is None or value == ""

    for field in match_keys:
        history_value = history_data.get(field, None)
        new_value = new_data.get(field, None)
        if is_empty(history_value) and is_empty(new_value):
            continue
        if history_value != new_value:
            return False
    return True


def calculate_best_perf_result(history_data_list, new_data):
    """
    Get the best performance metrics from history data and new data
    """
    # Combine history data and new data
    all_data = []
    if history_data_list:
        all_data.extend(history_data_list)

    # Handle new_data as either a single dict or list
    if isinstance(new_data, list):
        all_data.extend(new_data)
    elif new_data:
        all_data.append(new_data)

    if not all_data:
        return {}

    best_metrics = {}

    # Calculate best values for maximize metrics
    for metric in MAXIMIZE_METRICS:
        values = []
        for data in all_data:
            # Skip baseline data
            if data.get("b_is_baseline") and data.get("b_is_baseline") == True:
                continue
            if metric not in data:
                continue
            values.append(data.get(metric))
        if values:
            best_metrics[metric] = max(values)

    # Calculate best values for minimize metrics
    for metric in MINIMIZE_METRICS:
        values = []
        for data in all_data:
            # Skip baseline data
            if data.get("b_is_baseline") and data.get("b_is_baseline") == True:
                continue
            if metric not in data:
                continue
            values.append(data.get(metric))
        if values:
            best_metrics[metric] = min(values)

    return best_metrics


def get_history_data(new_data_dict, match_keys, common_values_dict):
    """
    Query history post-merge data for each cmd_idx
    """

    def get_latest_data(data_list):
        if not data_list:
            return None

        # Supported timestamp formats
        time_formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO 8601: 2025-12-11T06:25:25.338Z
            "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601 without ms: 2025-12-11T06:25:25Z
            "%Y-%m-%dT%H:%M:%S.%f",  # ISO 8601 without Z: 2025-12-11T06:25:25.338
            "%Y-%m-%dT%H:%M:%S",  # ISO 8601 basic: 2025-12-11T06:25:25
            "%b %d, %Y @ %H:%M:%S.%f",  # OpenSearch format: Dec 11, 2025 @ 06:25:25.338
        ]

        def parse_timestamp(timestamp):
            if isinstance(timestamp, (int, float)):
                # Handle milliseconds timestamp
                if timestamp > 1e12:
                    timestamp = timestamp / 1000
                return datetime.fromtimestamp(timestamp)
            if isinstance(timestamp, datetime):
                return timestamp

            timestamp_str = str(timestamp)
            for fmt in time_formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue

            print_warning(f"Unable to parse timestamp: {timestamp_str}")
            return datetime.fromtimestamp(0)

        # Find the item with the maximum @timestamp value
        latest_data = max(data_list,
                          key=lambda x: parse_timestamp(x.get("@timestamp", 0)))
        return latest_data

    cmd_idxs = new_data_dict.keys()
    history_data_list = None
    if cmd_idxs:
        last_days = QUERY_LOOKBACK_DAYS
        must_clauses = [
            {
                "term": {
                    "b_is_valid": True
                }
            },
            {
                "term": {
                    "b_is_post_merge": True
                }
            },
            {
                "term": {
                    "b_is_regression": False
                }
            },
            {
                "range": {
                    "ts_created": {
                        "gte":
                        int(time.time() - 24 * 3600 * last_days) //
                        (24 * 3600) * 24 * 3600 * 1000,
                    }
                }
            },
        ]
        for key, value in common_values_dict.items():
            must_clauses.append({"term": {key: value}})
        history_data_list = OpenSearchDB.queryPerfDataFromOpenSearchDB(
            TEST_INFO_PROJECT_NAME, must_clauses, size=MAX_QUERY_SIZE)

    # If query_history_data returned None, it means network failure
    if history_data_list is None:
        return None, None

    # Query was successful (even if empty list), initialize dicts
    history_baseline_dict = {}
    history_data_dict = {}
    for cmd_idx in cmd_idxs:
        history_data_dict[cmd_idx] = []
        history_baseline_dict[cmd_idx] = []

    # Process history data if we have any
    if history_data_list:
        for history_data in history_data_list:
            for cmd_idx in cmd_idxs:
                if match(history_data, new_data_dict[cmd_idx], match_keys):
                    if history_data.get("b_is_baseline") and history_data.get(
                            "b_is_baseline") == True:
                        history_baseline_dict[cmd_idx].append(history_data)
                    else:
                        history_data_dict[cmd_idx].append(history_data)
                    break

    # Sometimes the database has several baselines and we only use the latest one
    # If list is empty, set to None for each cmd_idx
    for cmd_idx, baseline_list in history_baseline_dict.items():
        latest_baseline = get_latest_data(baseline_list)
        history_baseline_dict[cmd_idx] = latest_baseline
    return history_baseline_dict, history_data_dict


def _get_threshold_for_metric(baseline_data, metric, is_post_merge):
    """
    Get the threshold for a metric from baseline data using is_post_merge flag.
    """
    metric_suffix = metric[2:]  # Remove "d_" prefix
    if is_post_merge:
        threshold_key = f"d_threshold_post_merge_{metric_suffix}"
    else:
        threshold_key = f"d_threshold_pre_merge_{metric_suffix}"

    if threshold_key in baseline_data:
        return baseline_data[threshold_key]

    raise KeyError(f"No threshold found for metric '{metric}'. "
                   f"Expected '{threshold_key}' in baseline data.")


def _calculate_diff(metric, new_value, baseline_value):
    """
    Calculate the percentage difference between new and baseline values.
    Returns a positive number if perf is better, negative if worse.
    """
    if baseline_value == 0:
        return 0.0
    if metric in MAXIMIZE_METRICS:
        # Larger is better: new > baseline means positive (better)
        return (new_value - baseline_value) / baseline_value * 100
    else:
        # Smaller is better: new < baseline means positive (better)
        return (baseline_value - new_value) / baseline_value * 100


def prepare_regressive_test_cases(history_baseline_dict, new_data_dict):
    """Update regression info for all data in new_data_dict.
    """
    # If history_baseline_dict is None (network failure), skip regression check
    if history_baseline_dict is None:
        return

    for cmd_idx in new_data_dict:
        new_data = new_data_dict[cmd_idx]
        history_baseline = history_baseline_dict.get(cmd_idx)
        if history_baseline is None:
            new_data["s_regression_info"] = ""
            new_data["b_is_regression"] = False
            continue

        is_post_merge = new_data.get("b_is_post_merge", False)
        info_parts = [
            f"baseline_id: {history_baseline.get('_id', '')}",
            f"baseline_branch: {history_baseline.get('s_branch', '')}",
            f"baseline_commit: {history_baseline.get('s_commit', '')}",
            f"baseline_date: {history_baseline.get('ts_created', '')}",
        ]
        regressive_metrics = []
        # Check all metrics and build info string
        for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
            if metric not in new_data or metric not in history_baseline:
                continue

            baseline_value = history_baseline[metric]
            new_value = new_data[metric]
            threshold = _get_threshold_for_metric(history_baseline, metric,
                                                  is_post_merge)
            diff = _calculate_diff(metric, new_value, baseline_value)

            # Add metric info to s_regression_info
            metric_info = (f"{metric}'s value: {new_value} "
                           f"baseline value: {baseline_value} "
                           f"threshold: {threshold * 100:.2f}% "
                           f"diff: {diff:+.2f}%")
            info_parts.append(metric_info)

            # Check if this metric is regressive (only for key regression metrics)
            if metric in REGRESSION_METRICS:
                if metric in MAXIMIZE_METRICS:
                    # Regressive if new_value < baseline_value * (1 - threshold)
                    if new_value < baseline_value * (1 - threshold):
                        regressive_metrics.append(metric)
                else:
                    # Regressive if new_value > baseline_value * (1 + threshold)
                    if new_value > baseline_value * (1 + threshold):
                        regressive_metrics.append(metric)

        new_data["s_regression_info"] = ", ".join(info_parts)
        new_data["b_is_regression"] = len(regressive_metrics) > 0


def _is_valid_baseline(baseline_data):
    """Check if baseline data is valid (non-empty dict)."""
    if isinstance(baseline_data, dict) and len(baseline_data) > 0:
        return True
    return False


def prepare_baseline_data(history_baseline_dict, history_data_dict,
                          new_data_dict):
    """
    Calculate new baseline from history post-merge data and new data.
    Then return new baseline data.
    """
    # If history_baseline_dict and history_data_dict are None (network failure),
    # return None to indicate we cannot prepare baseline data
    if history_baseline_dict is None and history_data_dict is None:
        return {}

    new_baseline_data_dict = {}
    cmd_idxs = new_data_dict.keys()
    # Find the best history post-merge data for each cmd
    for cmd_idx in cmd_idxs:
        # Calculate best metrics from history post-merge data and new data
        best_metrics = calculate_best_perf_result(history_data_dict[cmd_idx],
                                                  new_data_dict[cmd_idx])

        # Create new_baseline_data from new_data_dict and set b_is_baseline
        new_baseline_data = new_data_dict[cmd_idx].copy()
        new_baseline_data["b_is_baseline"] = True

        # Initialize metric_threshold_dict with default thresholds for all metrics
        metric_threshold_dict = {}
        for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
            metric_suffix = metric[2:]
            post_merge_key = f"d_threshold_post_merge_{metric_suffix}"
            pre_merge_key = f"d_threshold_pre_merge_{metric_suffix}"
            metric_threshold_dict[post_merge_key] = POST_MERGE_THRESHOLD
            metric_threshold_dict[pre_merge_key] = PRE_MERGE_THRESHOLD

        # If history baseline is valid, extract thresholds and update metric_threshold_dict
        history_baseline = history_baseline_dict[cmd_idx]
        if _is_valid_baseline(history_baseline):
            for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
                metric_suffix = metric[2:]
                post_merge_key = f"d_threshold_post_merge_{metric_suffix}"
                pre_merge_key = f"d_threshold_pre_merge_{metric_suffix}"
                if post_merge_key in history_baseline:
                    metric_threshold_dict[post_merge_key] = history_baseline[
                        post_merge_key]
                if pre_merge_key in history_baseline:
                    metric_threshold_dict[pre_merge_key] = history_baseline[
                        pre_merge_key]

        # Update new_baseline_data with best_metrics values
        for metric, value in best_metrics.items():
            new_baseline_data[metric] = value

        # Add all thresholds to new_baseline_data
        for threshold_key, threshold_value in metric_threshold_dict.items():
            new_baseline_data[threshold_key] = threshold_value

        add_id(new_baseline_data)
        new_baseline_data_dict[cmd_idx] = new_baseline_data

    return new_baseline_data_dict


def post_new_perf_data(new_baseline_data_dict, new_data_dict):
    """
    Post new perf results and new baseline to database
    """
    data_list = []
    cmd_idxs = new_data_dict.keys()
    for cmd_idx in cmd_idxs:
        # Only upload baseline data when post-merge.
        if new_baseline_data_dict and cmd_idx in new_baseline_data_dict:
            data_list.append(new_baseline_data_dict[cmd_idx])
        if cmd_idx in new_data_dict:
            data_list.append(new_data_dict[cmd_idx])
    if not data_list:
        return
    try:
        print_info(
            f"Ready to post {len(data_list)} data to {TEST_INFO_PROJECT_NAME}")
        OpenSearchDB.postToOpenSearchDB(data_list, TEST_INFO_PROJECT_NAME)
    except Exception as e:
        print_info(
            f"Failed to post data to {TEST_INFO_PROJECT_NAME}, error: {e}")


def _get_metric_keys():
    """Get all metric-related keys for filtering config keys."""
    metric_keys = set()
    for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
        metric_suffix = metric[2:]  # Strip "d_" prefix
        metric_keys.add(metric)
        metric_keys.add(f"d_baseline_{metric_suffix}")
        metric_keys.add(f"d_threshold_post_merge_{metric_suffix}")
        metric_keys.add(f"d_threshold_pre_merge_{metric_suffix}")
    return metric_keys


def _print_regression_data(data, print_func=None):
    """
    Print regression info and config.
    """
    if print_func is None:
        print_func = print_info

    if "s_regression_info" in data:
        print_func("=== Regression Info ===")
        for item in data["s_regression_info"].split(","):
            print_func(item.strip())

    metric_keys = _get_metric_keys()

    print_func("\n=== Config ===")
    config_keys = sorted([key for key in data.keys() if key not in metric_keys])
    for key in config_keys:
        if key == "s_regression_info":
            continue
        value = data[key]
        print_func(f'"{key}": {value}')


def check_perf_regression(new_data_dict,
                          fail_on_regression=False,
                          output_dir=None):
    """
    Check performance regression by printing regression data from new_data_dict.
    If fail_on_regression is True, raises RuntimeError when regressions are found.
    (This is a temporary feature to fail regression tests. We are observing the stability and will fail them by default soon.)
    If output_dir is provided, saves regression data to regression_data.yaml.
    """
    # Filter regression data from new_data_dict
    regressive_data_list = [
        data for data in new_data_dict.values()
        if data.get("b_is_regression", False)
    ]
    # Split regression data into post-merge and pre-merge
    post_merge_regressions = [
        data for data in regressive_data_list
        if data.get("b_is_post_merge", False)
    ]
    pre_merge_regressions = [
        data for data in regressive_data_list
        if not data.get("b_is_post_merge", False)
    ]

    # Save regression data to yaml file if output_dir is provided
    if output_dir is not None and len(regressive_data_list) > 0:
        regression_data_file = os.path.join(output_dir, "regression_data.yaml")
        with open(regression_data_file, 'w') as f:
            yaml.dump(regressive_data_list, f, default_flow_style=False)
        print_info(
            f"Saved {len(regressive_data_list)} regression data to {regression_data_file}"
        )

    # Print pre-merge regression data with print_warning
    if len(pre_merge_regressions) > 0:
        print_warning(
            f"Found {len(pre_merge_regressions)} pre-merge perf regression data"
        )
        for i, data in enumerate(pre_merge_regressions):
            print_warning(f"\n{'=' * 60}")
            print_warning(f"Pre-merge Regression Data #{i + 1}")
            print_warning("=" * 60)
            _print_regression_data(data, print_func=print_warning)

        if fail_on_regression:
            raise RuntimeError(
                f"Found {len(pre_merge_regressions)} pre-merge perf regression data"
            )

    # Print post-merge regression data with print_warning
    if len(post_merge_regressions) > 0:
        print_warning(
            f"Found {len(post_merge_regressions)} post-merge perf regression data"
        )
        for i, data in enumerate(post_merge_regressions):
            print_warning(f"\n{'=' * 60}")
            print_warning(f"Post-merge Regression Data #{i + 1}")
            print_warning("=" * 60)
            _print_regression_data(data, print_func=print_warning)

        if fail_on_regression:
            raise RuntimeError(
                f"Found {len(post_merge_regressions)} post-merge perf regression data"
            )

    # Print summary if no regressions
    if len(regressive_data_list) == 0:
        print_info("No regression data found.")
