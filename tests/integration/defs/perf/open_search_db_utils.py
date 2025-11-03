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

from defs.trt_test_alternative import print_info

_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from jenkins.scripts.open_search_db import OpenSearchDB

PROJECT_ROOT = "sandbox-temp-trtllm-ci-perf-v1"  # "sandbox-trtllm-ci-perf"
TEST_INFO_PROJECT_NAME = f"{PROJECT_ROOT}-test_info"

# Server config fields to compare
SERVER_FIELDS = [
    "s_model_name",
    "l_gpus",
    "l_tp",
    "l_ep",
    "l_pp",
    "l_max_num_tokens",
    "b_enable_chunked_prefill",
    "b_disable_overlap_scheduler",
    "s_attention_backend",
    "s_moe_backend",
    "l_moe_max_num_tokens",
    "l_stream_interval",
    "b_enable_attention_dp",
    "b_attention_dp_balance",
    "l_batching_wait_iters",
    "l_timeout_iters",
    "s_kv_cache_dtype",
    "b_enable_block_reuse",
    "d_free_gpu_memory_fraction",
    "l_max_batch_size",
    "b_enable_padding",
]

# Client config fields to compare
CLIENT_FIELDS = [
    "l_concurrency",
    "l_iterations",
    "l_isl",
    "l_osl",
    "d_random_range_ratio",
]

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
    except:
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
        "b_is_regression": False,
    }


def query_history_data():
    """
    Query post-merge data with specific gpu type and model name
    """
    # Query data from the last 14 days
    last_days = 14
    json_data = {
        "query": {
            "bool": {
                "must": [
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
                        "range": {
                            "ts_created": {
                                "gte":
                                int(time.time() - 24 * 3600 * last_days) //
                                (24 * 3600) * 24 * 3600 * 1000,
                            }
                        }
                    },
                ]
            },
        },
        "size": 3000,
    }
    json_data = json.dumps(json_data)

    data_list = []
    try:
        res = OpenSearchDB.queryFromOpenSearchDB(json_data,
                                                 TEST_INFO_PROJECT_NAME)
        if res is None:
            # No response from database, return None
            print_info(
                f"Fail to query from {TEST_INFO_PROJECT_NAME}, returned no response"
            )
            return []
        else:
            payload = res.json().get("hits", {}).get("hits", [])
            if len(payload) == 0:
                # No history data found in database, return empty list
                print_info(
                    f"Fail to query from {TEST_INFO_PROJECT_NAME}, returned no data"
                )
                return []
            for hit in payload:
                data_dict = hit.get("_source", {})
                data_dict["_id"] = hit.get("_id", "")
                if data_dict["_id"] == "":
                    print_info(
                        f"Fail to query from {TEST_INFO_PROJECT_NAME}, returned data with no _id"
                    )
                    # Invalid data, return None
                    return []
                data_list.append(data_dict)
            print_info(
                f"Successfully query from {TEST_INFO_PROJECT_NAME}, queried {len(data_list)} entries"
            )
            return data_list
    except Exception as e:
        print_info(
            f"Fail to query from {TEST_INFO_PROJECT_NAME}, returned error: {e}")
        return []


def match(history_data, new_data):
    """
    Check if the server and client config of history data matches the new data
    """
    # Combine all fields to compare (excluding log links)
    fields_to_compare = SERVER_FIELDS + CLIENT_FIELDS

    def is_empty(value):
        """Check if a value is empty (None, empty string, etc.)"""
        return value is None or value == ""

    # Compare each field
    for field in fields_to_compare:
        history_value = history_data.get(field)
        new_value = new_data.get(field)

        # If both are empty, consider them equal
        if is_empty(history_value) and is_empty(new_value):
            continue

        # If values don't match, return False
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


def get_history_data(new_data_dict):
    """
    Query history post-merge data for each cmd_idx
    """
    history_baseline_dict = {}
    history_data_dict = {}
    cmd_idxs = new_data_dict.keys()
    for cmd_idx in cmd_idxs:
        history_data_dict[cmd_idx] = []
        history_baseline_dict[cmd_idx] = None
    history_data_list = query_history_data()
    if history_data_list:
        for history_data in history_data_list:
            for cmd_idx in cmd_idxs:
                if match(history_data, new_data_dict[cmd_idx]):
                    if history_data.get("b_is_baseline") and history_data.get(
                            "b_is_baseline") == True:
                        history_baseline_dict[cmd_idx] = history_data
                    else:
                        history_data_dict[cmd_idx].append(history_data)
                    break
    return history_baseline_dict, history_data_dict


def prepare_regressive_test_cases(history_baseline_dict, new_data_dict):
    """Get regressive test cases
    1. For Maximize metrics, if new perf is below baseline * (1 - threshold)
    2. For Minimize metrics, if new perf is above baseline * (1 + threshold)
    Set it as regressive.
    """
    regressive_data_list = []
    # Find regressive test cases
    for cmd_idx in new_data_dict:
        if history_baseline_dict[cmd_idx] is None:
            continue

        baseline_data = history_baseline_dict[cmd_idx]
        new_data = new_data_dict[cmd_idx]
        is_regressive = False
        regressive_metrics = []

        # Check MAXIMIZE_METRICS (new should be >= baseline * (1 - threshold))
        for metric in MAXIMIZE_METRICS:
            if metric not in new_data or metric not in baseline_data:
                continue
            threshold_key = f"d_threshold_{metric[2:]}"
            threshold = baseline_data[threshold_key]
            baseline_value = baseline_data[metric]
            new_value = new_data[metric]
            # Regressive if new_value < baseline_value * (1 - threshold)
            if new_value < baseline_value * (1 - threshold):
                is_regressive = True
                regressive_metrics.append(metric)

        # Check MINIMIZE_METRICS (new should be <= baseline * (1 + threshold))
        for metric in MINIMIZE_METRICS:
            if metric not in new_data or metric not in baseline_data:
                continue
            threshold_key = f"d_threshold_{metric[2:]}"
            threshold = baseline_data.get(threshold_key, 0.1)
            baseline_value = baseline_data[metric]
            new_value = new_data[metric]
            # Regressive if new_value > baseline_value * (1 + threshold)
            if new_value > baseline_value * (1 + threshold):
                is_regressive = True
                regressive_metrics.append(metric)

        if is_regressive:
            # Copy new data and add baseline values, thresholds, and regression info
            regressive_data = new_data.copy()
            # Add baseline values and thresholds for all metrics
            for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
                if metric in baseline_data:
                    baseline_key = f"d_baseline_{metric[2:]}"
                    regressive_data[baseline_key] = baseline_data[metric]

                    threshold_key = f"d_threshold_{metric[2:]}"
                    if threshold_key in baseline_data:
                        regressive_data[threshold_key] = baseline_data[
                            threshold_key]

            # Add regression info string
            regressive_data["s_regression_info"] = ", ".join(regressive_metrics)
            regressive_data["b_is_regression"] = True
            add_id(regressive_data)
            regressive_data_list.append(regressive_data)

    return regressive_data_list


def prepare_baseline_data(history_baseline_dict, history_data_dict,
                          new_data_dict):
    """
    Calculate new baseline from history post-merge data and new data.
    Then return new baseline data.
    """
    new_baseline_data_dict = {}
    cmd_idxs = new_data_dict.keys()
    # Find the best history post-merge data for each cmd
    for cmd_idx in cmd_idxs:
        # Calculate best metrics from history post-merge data and new data
        best_metrics = calculate_best_perf_result(history_data_dict[cmd_idx],
                                                  new_data_dict[cmd_idx])
        new_baseline_data = history_baseline_dict[cmd_idx]
        if new_baseline_data:
            print_info(f"Baseline data found (cmd_idx: {cmd_idx}) in history")
        else:
            print_info(
                f"No baseline data found (cmd_idx: {cmd_idx}), created a new baseline"
            )
            new_baseline_data = new_data_dict[cmd_idx].copy()
            new_baseline_data["b_is_baseline"] = True
            add_id(new_baseline_data)
        # Add or update baseline metrics
        for metric, value in best_metrics.items():
            new_baseline_data[metric] = value
            new_baseline_data[f"d_threshold_{metric[2:]}"] = 0.1
        new_baseline_data_dict[cmd_idx] = new_baseline_data

    return new_baseline_data_dict


def post_new_perf_data(new_baseline_data_dict, new_data_dict,
                       regressive_data_list):
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
    # Only post regressive test cases when post-merge.
    if new_baseline_data_dict:
        data_list.extend(regressive_data_list)
    try:
        print_info(
            f"Ready to post {len(data_list)} data to {TEST_INFO_PROJECT_NAME}")
        OpenSearchDB.postToOpenSearchDB(data_list, TEST_INFO_PROJECT_NAME)
    except Exception as e:
        print_info(f"Fail to post data to {TEST_INFO_PROJECT_NAME}, error: {e}")


def print_regressive_test_cases(regressive_data_list):
    """
    Print regressive test cases
    """
    print_info(f"Found {len(regressive_data_list)} regressive test cases")
    for data in regressive_data_list:
        print_info(f"Regressive test case: {data}")
