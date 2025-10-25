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
NVDataFlow utilities
"""
import hashlib
import json
import os
import re
import subprocess
import time

import requests
from defs.trt_test_alternative import print_error, print_info

NVDF_BASE_URL = "http://gpuwa.nvidia.com"
PROJECT_ROOT = "sandbox-temp-1-perf-test"

# Server config fields to compare
SERVER_FIELDS = [
    "s_model_name",
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


def get_nvdf_config() -> dict:
    (
        gpu_type,
        gpu_count,
        host_node_name,
        build_id,
        build_url,
        job_name,
        job_id,
        job_url,
        branch,
        commit,
        is_post_merge,
        is_pr_job,
        trigger_mr_user,
        trigger_mr_link,
        trigger_mr_id,
        trigger_mr_commit,
    ) = get_job_info()
    return {
        # Unique identifier
        "_id": "",
        "b_is_baseline": False,
        "b_is_valid": True,

        # GPU and Host Config
        "s_gpu_type": gpu_type,
        "l_gpu_count": gpu_count,
        "s_gpu_uuids": "",
        "s_host_node_name": host_node_name,

        # Job Config
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
    }


def get_job_info():
    """Get job info from environment variables

    Returns:
        Tuple of (gpu_type, gpu_count, host_node_name, build_id, build_url, job_name, job_id, job_url,
                  branch, commit, is_post_merge, is_pr_job, trigger_mr_user,
                  trigger_mr_link, trigger_mr_id, trigger_mr_commit)
    """
    # Read environment variables
    build_id = os.getenv("BUILD_ID", "")
    build_url = os.getenv("BUILD_URL", "")
    job_name = os.getenv("JOB_NAME", "")
    host_node_name = os.getenv("HOST_NODE_NAME", "")

    # Get gpu_type and gpu_count from nvidia-smi
    gpu_type = ""
    gpu_count = 1
    stage_name = os.getenv("STAGE_NAME", "")
    if stage_name:
        # 1. Get gpu info from STAGE_NAME
        parts = stage_name.split('-')
        assert len(parts) >= 2, f"Invalid stage_name: {stage_name}"
        gpu_count = int(parts[1].split('_')[0]) if "_GPUs" in parts[1] else 1
        gpu_type = f"{parts[0].replace('_', '-').lower()}-x{gpu_count}"
    else:
        # 2. Get gpu info from nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10)
            if result.returncode == 0:
                gpu_names = result.stdout.strip().split("\n")
                gpu_names = [name.strip() for name in gpu_names if name.strip()]
                gpu_type = gpu_names[0].replace(" ", "").lower()
                gpu_count = len(gpu_names)
            else:
                print_error(f"nvidia-smi query failed: {result.stderr}")
        except Exception as e:
            print_error(f"Failed to get GPU info from nvidia-smi: {e}")
    assert gpu_type and gpu_count > 0, "Failed to get GPU info from nvidia-smi"

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
        # Try to extract username from patterns like 'target="_blank">username</a>'
        user_match = re.search(r'target="_blank">([^<]+)</a>', trigger_info)
        if user_match:
            trigger_mr_user = user_match.group(1)

        # Set trigger_mr_commit to commit
        trigger_mr_commit = commit

    print_info(
        f"gpu_type: {gpu_type}, gpu_count: {gpu_count}, host_node_name: {host_node_name}, build_id: {build_id}, build_url: {build_url}, job_name: {job_name}, job_id: {job_id}, job_url: {job_url}, branch: {branch}, commit: {commit}, is_post_merge: {is_post_merge}, is_pr_job: {is_pr_job}, trigger_mr_user: {trigger_mr_user}, trigger_mr_link: {trigger_mr_link}, trigger_mr_id: {trigger_mr_id}, trigger_mr_commit: {trigger_mr_commit}"
    )

    return (gpu_type, gpu_count, host_node_name, build_id, build_url, job_name,
            job_id, job_url, branch, commit, is_post_merge, is_pr_job,
            trigger_mr_user, trigger_mr_link, trigger_mr_id, trigger_mr_commit)


def _id(data):
    """
    Generate hash for data
    """
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def type_check_for_nvdf(json_data):
    """Type check for NVDataFlow data

    Args:
        json_data: JSON data to check
    """
    # NVDF Post supports both list[Dict] and Dict.
    if isinstance(json_data, list):
        for data in json_data:
            type_check_for_nvdf(data)
        return json_data

    assert isinstance(json_data, dict)
    for key, value in json_data.items():
        if key.startswith("l_"):
            if not isinstance(value, int):
                print_error(
                    f"NVDF type check failed! key:{key}, value:{value} value_type:{type(value)}"
                )
        elif key.startswith("s_"):
            if not isinstance(value, str):
                print_error(
                    f"NVDF type check failed! key:{key}, value:{value} value_type:{type(value)}"
                )
        elif key.startswith("b_"):
            if not isinstance(value, bool):
                print_error(
                    f"NVDF type check failed! key:{key}, value:{value} value_type:{type(value)}"
                )
        elif key.startswith("ts_"):
            if not isinstance(value, int):
                print_error(
                    f"NVDF type check failed! key:{key}, value:{value} value_type:{type(value)}"
                )
        elif key.startswith("flat_"):
            if not isinstance(value, dict):
                print_error(
                    f"NVDF type check failed! key:{key}, value:{value} value_type:{type(value)}"
                )
        elif key.startswith("ni_"):
            if not isinstance(value, (str, int, float)):
                print_error(
                    f"NVDF type check failed! key:{key}, value:{value} value_type:{type(value)}"
                )
        else:
            print_error(
                f"Unknown key type! key:{key}, value:{value} value_type:{type(value)}"
            )
    return json_data


def post(data_list, project):
    """Post data to NVDataFlow database

    Args:
        data_list: List of data to post
        project: Project name to post
    """
    type_check_for_nvdf(data_list)
    url = f"{NVDF_BASE_URL}/dataflow2/{project}/posting"
    headers = {"Content-Type": "application/json", "Accept-Charset": "UTF-8"}
    json_data = json.dumps(data_list)
    retry_time = 5
    while retry_time:
        res = requests.post(url, data=json_data, headers=headers)
        if res.status_code in [200, 201, 202]:
            print_info(
                f"NVDF post successfully to {project}, post data size: {len(data_list)}, status code: {res.status_code}"
            )
            return
        print_info(
            f"NVDF post failed to {project}, will retry, error: {res.status_code} {res.text}"
        )
        retry_time -= 1
    print_error(
        f"Fail to post to {project} after {retry_time} retry: {url}, json: {json_data}, error: {res.text}"
    )


def query(json_data, project):
    """Query data from NVDataFlow database

    Args:
        json_data: JSON data to query
        project: Project name to query
    """
    if not isinstance(json_data, str):
        json_data = json.dumps(json_data)
    url = "{}/opensearch/df-{}-*/_search".format(NVDF_BASE_URL, project)
    headers = {"Content-Type": "application/json", "Accept-Charset": "UTF-8"}
    retry_time = 5
    while retry_time:
        res = requests.get(url, data=json_data, headers=headers, timeout=10)
        if res.status_code in [200, 201, 202]:
            return res
        print_info(
            f"NVDF query failed to {project}, will retry, error: {res.status_code} {res.text}"
        )
        retry_time -= 1
    print_error(
        f"Fail to query from {project} after {retry_time} retry: {url}, json: {json_data}, error: {res.text}"
    )
    return None


def post_data(baseline_data_dict, new_data_dict, model_to_cmd_idx_group,
              gpu_type):
    """Post data to NVDataFlow database

    Args:
        baseline_data_dict: Baseline data dict
        new_data_dict: New data dict
        model_to_cmd_idx_group: Model to command index group
        gpu_type: GPU type
    """
    for model_name, cmd_idxs in model_to_cmd_idx_group.items():
        data_list = []
        for cmd_idx in cmd_idxs:
            data_list.append(baseline_data_dict[cmd_idx])
            data_list.append(new_data_dict[cmd_idx])
        project_name = f"{PROJECT_ROOT}-{gpu_type}-{model_name}"
        try:
            post(data_list, project_name)
        except Exception as e:
            print_error(f"NVDF post data error to {project_name}, error: {e}")
            return []


def query_data(gpu_type, model_name):
    """Query data with specific gpu type and model name

    Args:
        model_name: Model name to filter
        gpu_type: GPU type to filter
    """
    project_name = f"{PROJECT_ROOT}-{gpu_type}-{model_name}"

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
        res = query(json_data, project_name)
        if res is None:
            print_error(f"NVDF query returned no response for {project_name}")
            return []
        else:
            payload = res.json().get("hits", {}).get("hits", [])
            if len(payload) == 0:
                print_error(f"NVDF query returned no data for {project_name}")
                return []
            for hit in payload:
                data_dict = hit.get("_source", {})
                data_dict["_id"] = hit.get("_id", "")
                if data_dict["_id"] == "":
                    print_error(
                        f"NVDF query returned data with no _id for {project_name}"
                    )
                    continue
                data_list.append(data_dict)
            print_info(
                f"NVDF query successfully to {project_name}, queried {len(data_list)} entries"
            )
            return data_list
    except Exception as e:
        print_error(f"NVDF query returned error for {project_name}: {e}")
        return []


def prepare_baseline_data(new_data_dict, model_to_cmd_idx_group, gpu_type):
    """Prepare baseline data from new data and history data queried from database

    Args:
        new_data_dict: New data dict
        model_to_cmd_idx_group: Model to command index group
        gpu_type: GPU type

    Returns:
        Baseline data dict
    """
    # Query history data from database
    for model_name, cmd_idxs in model_to_cmd_idx_group.items():
        history_data_list = query_data(gpu_type, model_name)
        # Put history data that matches each cmd into a group
        cmd_to_history_data_group = {}
        for cmd_idx in cmd_idxs:
            cmd_to_history_data_group[cmd_idx] = []

        for history_data in history_data_list:
            for cmd_idx in cmd_idxs:
                if match(history_data, new_data_dict[cmd_idx]):
                    cmd_to_history_data_group[cmd_idx].append(history_data)
                    break

        baseline_data_dict = {}
        # Find the best history data for each cmd
        for cmd_idx in cmd_idxs:
            best_history_metrics = get_best_perf_result(
                cmd_to_history_data_group[cmd_idx], new_data_dict[cmd_idx])
            # Check if history data has baseline
            baseline_data = get_baseline(cmd_to_history_data_group[cmd_idx])

            new_baseline_data = {}
            if baseline_data:
                # If baseline is in history, update the baseline.
                new_baseline_data = baseline_data
            else:
                # If no baseline in history, create a new baseline.
                new_baseline_data = new_data_dict[cmd_idx].copy()
                new_baseline_data["b_is_baseline"] = True
                new_baseline_data["_id"] = _id(new_baseline_data)

            # Add or update baseline metrics
            for metric, value in best_history_metrics.items():
                new_baseline_data[metric] = value
                new_baseline_data[f"d_threshold_{metric[2:]}"] = 0.1
            baseline_data_dict[cmd_idx] = new_baseline_data

        return baseline_data_dict


def match(history_data, new_data):
    """Check if the server and client config of history data matches the new data

    Args:
        history_data: History data
        new_data: New data
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


def get_best_perf_result(history_data_list, new_data):
    """Get the best performance metrics from history data and new data

    Args:
        history_data_list: List of history data dicts
        new_data: New data dict or list of dicts

    Returns:
        Dict with best metric values
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


def get_baseline(history_data_list):
    """Get baseline data from history data list

    Args:
        history_data_list: List of history data dictionaries

    Returns:
        Baseline data dict if found, None otherwise
    """
    if not history_data_list:
        return None

    for data in history_data_list:
        if data.get("b_is_baseline") and data.get("b_is_baseline") == True:
            return data

    return None
