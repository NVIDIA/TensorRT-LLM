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


def _rolling_smooth(values, window=3):
    """Trailing rolling mean with same-length output.

    Early elements use fewer samples (i.e. the first element is itself,
    the second is the mean of the first two, etc.).
    """
    if not values:
        return []
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        w = values[start:i + 1]
        smoothed.append(sum(w) / len(w))
    return smoothed


def _percentile(values, p):
    """Compute the p-th percentile with linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (p / 100.0) * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _daily_aggregate_values(data_list, metric):
    """Aggregate multiple data points on the same day to a single mean value.

    Returns a list of daily-aggregated metric values sorted by date.
    """
    by_day = {}
    for data in data_list:
        if data.get("b_is_baseline"):
            continue
        val = data.get(metric)
        if val is None:
            continue
        ts = data.get("ts_created") or data.get("@timestamp")
        if ts is None:
            continue
        if isinstance(ts, (int, float)):
            if ts > 1e12:
                ts = ts / 1000
            day_key = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        elif isinstance(ts, datetime):
            day_key = ts.strftime("%Y-%m-%d")
        else:
            day_key = str(ts)[:10]
        by_day.setdefault(day_key, []).append(val)
    result = []
    for day in sorted(by_day):
        vals = by_day[day]
        result.append(sum(vals) / len(vals))
    return result


def calculate_baseline_metrics(history_data_list, new_data):
    """Calculate baseline metrics using rolling smooth + percentile algorithm.

    For each metric, aggregates data to daily values, applies a trailing
    rolling mean (window=3), then takes:
      - P95 for MAXIMIZE_METRICS (larger is better, e.g. throughput)
      - P5  for MINIMIZE_METRICS (smaller is better, e.g. latency)
    """
    all_data = []
    if history_data_list:
        all_data.extend(history_data_list)
    if isinstance(new_data, list):
        all_data.extend(new_data)
    elif new_data:
        all_data.append(new_data)

    if not all_data:
        return {}

    baseline_metrics = {}
    for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
        daily_vals = _daily_aggregate_values(all_data, metric)
        if not daily_vals:
            continue
        smoothed = _rolling_smooth(daily_vals, window=3)
        if metric in MAXIMIZE_METRICS:
            baseline_metrics[metric] = _percentile(smoothed, 95)
        else:
            baseline_metrics[metric] = _percentile(smoothed, 5)

    return baseline_metrics


def get_history_data(new_data_dict, match_keys, common_values_dict):
    """
    Query history post-merge data for each cmd_idx.

    Returns (latest_history_data_dict, history_data_dict):
      - latest_history_data_dict: latest post-merge entry per cmd_idx (or None)
      - history_data_dict: all history post-merge entries per cmd_idx
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
                    "b_is_baseline": False
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
    history_data_dict = {}
    for cmd_idx in cmd_idxs:
        history_data_dict[cmd_idx] = []

    # Process history data if we have any
    if history_data_list:
        for history_data in history_data_list:
            for cmd_idx in cmd_idxs:
                if match(history_data, new_data_dict[cmd_idx], match_keys):
                    history_data_dict[cmd_idx].append(history_data)
                    break

    # Find the latest entry per cmd_idx
    latest_history_data_dict = {}
    for cmd_idx in cmd_idxs:
        latest_history_data_dict[cmd_idx] = get_latest_data(
            history_data_dict[cmd_idx])
    return latest_history_data_dict, history_data_dict


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


def prepare_regressive_test_cases(latest_history_data_dict, history_data_dict,
                                  new_data_dict):
    """Update regression info for all data in new_data_dict.

    Uses embedded baseline fields from latest history data when available,
    otherwise falls back to calculating baseline from history data.
    """
    # If latest_history_data_dict is None (network failure), skip regression check
    if latest_history_data_dict is None:
        return

    for cmd_idx in new_data_dict:
        new_data = new_data_dict[cmd_idx]
        latest_history = latest_history_data_dict.get(cmd_idx)
        if latest_history is None:
            new_data["s_regression_info"] = ""
            new_data["b_is_regression"] = False
            continue

        is_post_merge = new_data.get("b_is_post_merge", False)
        regressive_metrics = []
        info_lines = []

        # Pre-calculate fallback baseline from history if needed
        fallback_baseline = None

        # Check all metrics and build info string
        for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
            if metric not in new_data:
                continue

            new_value = new_data[metric]
            metric_suffix = metric[2:]  # Remove "d_" prefix

            # Get baseline value: try embedded field from latest history first
            baseline_key = f"d_baseline_{metric_suffix}"
            baseline_value = latest_history.get(baseline_key)
            if baseline_value is None or baseline_value <= 0:
                # Fallback: calculate from history data
                if fallback_baseline is None:
                    history_list = history_data_dict.get(cmd_idx, [])
                    fallback_baseline = calculate_baseline_metrics(
                        history_list, new_data)
                baseline_value = fallback_baseline.get(metric)
                if baseline_value is None or baseline_value <= 0:
                    continue

            # Get threshold: try embedded field from latest history first
            if is_post_merge:
                threshold_key = f"d_threshold_post_merge_{metric_suffix}"
                default_threshold = POST_MERGE_THRESHOLD
            else:
                threshold_key = f"d_threshold_pre_merge_{metric_suffix}"
                default_threshold = PRE_MERGE_THRESHOLD

            threshold = latest_history.get(threshold_key)
            if threshold is None or threshold <= 0:
                threshold = default_threshold

            diff = _calculate_diff(metric, new_value, baseline_value)
            info_lines.append(
                f"  {metric}: value={new_value:.4f} "
                f"baseline={baseline_value:.4f} "
                f"threshold={threshold * 100:.2f}% "
                f"diff={diff:+.2f}%")

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

        test_case = new_data.get("s_test_case_name", "unknown")
        header = f"Regression in {test_case}:"
        new_data["s_regression_info"] = "\n".join([header] + info_lines)
        new_data["b_is_regression"] = len(regressive_metrics) > 0


def add_baseline_fields_to_post_merge_data(latest_history_data_dict,
                                           new_data_dict):
    """Embed baseline fields directly into each post-merge data entry.

    For each metric, adds:
      - d_baseline_{metric_suffix}: from latest history if available and > 0, else -1
      - d_threshold_post_merge_{metric_suffix}: from latest history if available, else POST_MERGE_THRESHOLD
      - d_threshold_pre_merge_{metric_suffix}: from latest history if available, else PRE_MERGE_THRESHOLD
    """
    if latest_history_data_dict is None:
        return

    for cmd_idx in new_data_dict:
        new_data = new_data_dict[cmd_idx]
        latest_history = latest_history_data_dict.get(cmd_idx)

        for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
            metric_suffix = metric[2:]  # Remove "d_" prefix
            baseline_key = f"d_baseline_{metric_suffix}"
            post_merge_key = f"d_threshold_post_merge_{metric_suffix}"
            pre_merge_key = f"d_threshold_pre_merge_{metric_suffix}"

            # Threshold: inherit from latest history or use defaults
            if latest_history and post_merge_key in latest_history:
                new_data[post_merge_key] = latest_history[post_merge_key]
            else:
                new_data[post_merge_key] = POST_MERGE_THRESHOLD

            if latest_history and pre_merge_key in latest_history:
                new_data[pre_merge_key] = latest_history[pre_merge_key]
            else:
                new_data[pre_merge_key] = PRE_MERGE_THRESHOLD

            # Baseline value: inherit from latest history if positive, else -1
            if (latest_history and baseline_key in latest_history
                    and latest_history[baseline_key] is not None
                    and latest_history[baseline_key] > 0):
                new_data[baseline_key] = latest_history[baseline_key]
            else:
                new_data[baseline_key] = -1


def post_new_perf_data(new_data_dict):
    """
    Post new perf results to database.
    """
    data_list = list(new_data_dict.values())
    if not data_list:
        return
    try:
        print_info(
            f"Ready to post {len(data_list)} data to {TEST_INFO_PROJECT_NAME}")
        OpenSearchDB.postToOpenSearchDB(data_list, TEST_INFO_PROJECT_NAME)
    except Exception as e:
        print_info(
            f"Failed to post data to {TEST_INFO_PROJECT_NAME}, error: {e}")


def check_perf_regression(new_data_dict, fail_on_regression=False):
    """
    Check performance regression by printing s_regression_info for each
    regressive entry. Post-merge regressions log warnings. Pre-merge
    regressions raise RuntimeError when fail_on_regression is True.
    """
    regressive_data_list = [
        data for data in new_data_dict.values()
        if data.get("b_is_regression", False)
    ]

    if not regressive_data_list:
        print_info("No regression data found.")
        return

    post_merge_regressions = [
        data for data in regressive_data_list
        if data.get("b_is_post_merge", False)
    ]
    pre_merge_regressions = [
        data for data in regressive_data_list
        if not data.get("b_is_post_merge", False)
    ]

    # Print post-merge regression details as warnings
    for data in post_merge_regressions:
        print_warning(data.get("s_regression_info", ""))

    # Print pre-merge regression details and raise error
    if pre_merge_regressions:
        error_parts = []
        for data in pre_merge_regressions:
            info = data.get("s_regression_info", "")
            print_warning(info)
            error_parts.append(info)
        if fail_on_regression:
            raise RuntimeError("\n".join(error_parts))
