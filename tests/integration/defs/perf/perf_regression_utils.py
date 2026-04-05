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
"""Performance regression pipeline utilities.

Generic regression detection and upload logic that can be reused across
different test scripts (perf sanity, module perf, accuracy, etc.).
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

from defs.trt_test_alternative import print_info, print_warning

from .open_search_db_utils import add_id, get_history_data, post_new_perf_data

# Default threshold values for performance regression detection
POST_MERGE_THRESHOLD = 0.05
PRE_MERGE_THRESHOLD = 0.1

_URM_BASE = "https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main"


def get_job_info():
    """Get job info from environment variables."""
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
        branch_match = re.search(r"/job/LLM/job/([^/]+)/job/", job_url)
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
                    "api.github.com/repos/", "github.com/"
                ).replace("/pulls/", "/pull/")

        # Extract user from trigger_info
        # Pattern: target="_blank">Barry-Delaney</a><br/>Git Commit:
        trigger_info = global_vars.get("action_info", {}).get("trigger_info", "")
        # Try to extract username from patterns like 'target="_blank">username</a><br/>Git Commit:'
        user_match = re.search(r'target="_blank">([^<]+)</a><br/>Git Commit:', trigger_info)
        if user_match:
            trigger_mr_user = user_match.group(1)

        # Set trigger_mr_commit to commit
        trigger_mr_commit = commit
        artifact_url = f"{_URM_BASE}/L0_MergeRequest_PR/{job_id}" if job_id else ""
    else:
        artifact_url = f"{_URM_BASE}/L0_PostMerge/{job_id}" if job_id else ""

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
    """Find keys from match_keys where all entries have identical values.

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
        w = values[start : i + 1]
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


def calculate_baseline_metrics(history_data_list, new_data, maximize_metrics, minimize_metrics):
    """Calculate baseline metrics using rolling smooth + percentile algorithm.

    For each metric, aggregates data to daily values, applies a trailing
    rolling mean (window=3), then takes:
      - P95 for maximize_metrics (larger is better, e.g. throughput)
      - P5  for minimize_metrics (smaller is better, e.g. latency)
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
    for metric in maximize_metrics + minimize_metrics:
        daily_vals = _daily_aggregate_values(all_data, metric)
        if not daily_vals:
            continue
        smoothed = _rolling_smooth(daily_vals, window=3)
        if metric in maximize_metrics:
            baseline_metrics[metric] = _percentile(smoothed, 95)
        else:
            baseline_metrics[metric] = _percentile(smoothed, 5)

    return baseline_metrics


def _calculate_diff(metric, new_value, baseline_value, maximize_metrics):
    """Calculate the percentage difference between new and baseline values.

    Returns a positive number if perf is better, negative if worse.
    """
    if baseline_value == 0:
        return 0.0
    if metric in maximize_metrics:
        # Larger is better: new > baseline means positive (better)
        return (new_value - baseline_value) / baseline_value * 100
    else:
        # Smaller is better: new < baseline means positive (better)
        return (baseline_value - new_value) / baseline_value * 100


def prepare_regressive_test_cases(
    latest_history_data_dict,
    history_data_dict,
    new_data_dict,
    maximize_metrics,
    minimize_metrics,
    regression_metrics,
):
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
        for metric in maximize_metrics + minimize_metrics:
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
                        history_list, new_data, maximize_metrics, minimize_metrics
                    )
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

            diff = _calculate_diff(metric, new_value, baseline_value, maximize_metrics)
            info_lines.append(
                f"  {metric}: value={new_value:.4f} "
                f"baseline={baseline_value:.4f} "
                f"threshold={threshold * 100:.2f}% "
                f"diff={diff:+.2f}%"
            )

            # Check if this metric is regressive (only for key regression metrics)
            if metric in regression_metrics:
                if metric in maximize_metrics:
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


def add_baseline_fields_to_post_merge_data(
    latest_history_data_dict, new_data_dict, maximize_metrics, minimize_metrics
):
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

        for metric in maximize_metrics + minimize_metrics:
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
            if (
                latest_history
                and baseline_key in latest_history
                and latest_history[baseline_key] is not None
                and latest_history[baseline_key] > 0
            ):
                new_data[baseline_key] = latest_history[baseline_key]
            else:
                new_data[baseline_key] = -1


def check_perf_regression(new_data_dict, fail_on_regression=False):
    """Check performance regression by printing s_regression_info.

    Post-merge regressions log warnings. Pre-merge
    regressions raise RuntimeError when fail_on_regression is True.
    """
    regressive_data_list = [
        data for data in new_data_dict.values() if data.get("b_is_regression", False)
    ]

    if not regressive_data_list:
        print_info("No regression data found.")
        return

    post_merge_regressions = [
        data for data in regressive_data_list if data.get("b_is_post_merge", False)
    ]
    pre_merge_regressions = [
        data for data in regressive_data_list if not data.get("b_is_post_merge", False)
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


def process_and_upload_test_results(
    new_data_dict: Dict[int, dict],
    match_keys: List[str],
    maximize_metrics: List[str],
    minimize_metrics: List[str],
    regression_metrics: List[str],
    extra_fields: Optional[dict] = None,
    upload_to_db: bool = True,
    fail_on_regression: Optional[bool] = None,
):
    """Generic regression pipeline: enrich data, query history, detect regression, upload.

    Args:
        new_data_dict: cmd_idx -> data dict with test config + metric fields.
        match_keys: Fields that uniquely identify a test case for history matching.
        maximize_metrics: Metrics where larger is better (e.g. throughput).
        minimize_metrics: Metrics where smaller is better (e.g. latency).
        regression_metrics: Subset of metrics to check for regression.
        extra_fields: Additional fields to merge into each data entry
            (e.g. {"s_stage_name": ..., "s_test_list": ...}).
        upload_to_db: Whether to post results to the database.
        fail_on_regression: Whether to raise on pre-merge regression.
            None = auto (fail for pre-merge only).
    """
    if not new_data_dict:
        print_info("No data to upload to database.")
        return

    # Step 1: Get job config and determine merge type
    job_config = get_job_info()
    is_post_merge = job_config["b_is_post_merge"]

    # Step 2: Enrich each entry with job config, extra fields, and ID
    for cmd_idx in new_data_dict:
        data = new_data_dict[cmd_idx]
        data.update(job_config)
        if extra_fields:
            data.update(extra_fields)
        add_id(data)

    # Step 3: Find common values to narrow query scope
    common_values_dict = get_common_values(new_data_dict, match_keys)

    # Step 4: Query history data
    latest_history_data_dict, history_data_dict = get_history_data(
        new_data_dict, match_keys, common_values_dict
    )

    # Step 5: Compute regression info
    prepare_regressive_test_cases(
        latest_history_data_dict,
        history_data_dict,
        new_data_dict,
        maximize_metrics,
        minimize_metrics,
        regression_metrics,
    )

    # Step 6: For post-merge, embed baseline fields
    if is_post_merge:
        add_baseline_fields_to_post_merge_data(
            latest_history_data_dict, new_data_dict, maximize_metrics, minimize_metrics
        )

    # Step 7: Upload to DB
    if upload_to_db:
        post_new_perf_data(new_data_dict)

    # Step 8: Check regression (auto-detect fail behavior if not specified)
    if fail_on_regression is None:
        fail_on_regression = not is_post_merge
    check_perf_regression(new_data_dict, fail_on_regression=fail_on_regression)
