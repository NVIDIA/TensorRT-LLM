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
OpenSearch database utilities — pure DB operations only.

Business logic (baseline calculation, regression detection) lives in
perf_regression_utils.py.
"""
import os
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


def add_id(data):
    OpenSearchDB.add_id_of_json(data)


def _match(history_data, new_data, match_keys):
    """
    Check if the server and client config of history data match the new data.
    """

    def is_empty(value):
        return value is None or value == ""

    for field in match_keys:
        history_value = history_data.get(field, None)
        new_value = new_data.get(field, None)
        # For boolean fields (b_ prefix), treat None/missing as False.
        # This ensures backward compatibility when new boolean match keys
        # are added — historical data without the field can still match
        # current data where the field defaults to False.
        if field.startswith("b_"):
            if history_value is None:
                history_value = False
            if new_value is None:
                new_value = False
        if is_empty(history_value) and is_empty(new_value):
            continue
        if history_value != new_value:
            return False
    return True


def get_history_data(new_data_dict, match_keys, common_values_dict):
    """
    Query history post-merge data for each cmd_idx.

    Returns (latest_history_data_dict, latest_baseline_threshold_dict, history_data_dict):
      - latest_history_data_dict: latest post-merge entry per cmd_idx (or None)
      - latest_baseline_threshold_dict: baseline/threshold fields from the most
        recent entry that has them, per cmd_idx (or None)
      - history_data_dict: all history post-merge entries per cmd_idx
    """
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

    def get_latest_data(data_list):
        """Return the entry with the most recent @timestamp, or None."""
        if not data_list:
            return None
        sorted_data = sorted(
            data_list,
            key=lambda x: parse_timestamp(x.get("@timestamp", 0)),
            reverse=True)
        return sorted_data[0]

    def get_latest_baseline_threshold(data_list):
        """Return baseline/threshold fields from the most recent entry that has them.

        Returns a dict of just the d_baseline_* and d_threshold_* fields,
        or None if no entry has baseline fields.
        """
        if not data_list:
            return None
        sorted_data = sorted(
            data_list,
            key=lambda x: parse_timestamp(x.get("@timestamp", 0)),
            reverse=True)
        for entry in sorted_data:
            if any(k.startswith("d_baseline_") for k in entry):
                return {
                    k: v
                    for k, v in entry.items() if k.startswith("d_baseline_")
                    or k.startswith("d_threshold_")
                }
        return None

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
            if key.startswith("b_") and value is False:
                # For boolean fields with value False, also match documents
                # where the field is missing (backward compatibility for
                # newly added boolean match keys).
                must_clauses.append({
                    "bool": {
                        "should": [
                            {
                                "term": {
                                    key: False
                                }
                            },
                            {
                                "bool": {
                                    "must_not": [{
                                        "exists": {
                                            "field": key
                                        }
                                    }]
                                }
                            },
                        ],
                        "minimum_should_match":
                        1,
                    }
                })
            else:
                must_clauses.append({"term": {key: value}})
        history_data_list = OpenSearchDB.queryPerfDataFromOpenSearchDB(
            TEST_INFO_PROJECT_NAME, must_clauses, size=MAX_QUERY_SIZE)

    # If query_history_data returned None, it means network failure
    if history_data_list is None:
        return None, None, None

    # Query was successful (even if empty list), initialize dicts
    history_data_dict = {}
    for cmd_idx in cmd_idxs:
        history_data_dict[cmd_idx] = []

    # Process history data if we have any
    if history_data_list:
        for history_data in history_data_list:
            for cmd_idx in cmd_idxs:
                if _match(history_data, new_data_dict[cmd_idx], match_keys):
                    history_data_dict[cmd_idx].append(history_data)
                    break

    # Find the latest entry per cmd_idx
    latest_history_data_dict = {}
    latest_baseline_threshold_dict = {}
    for cmd_idx in cmd_idxs:
        latest_history_data_dict[cmd_idx] = get_latest_data(
            history_data_dict[cmd_idx])
        latest_baseline_threshold_dict[cmd_idx] = get_latest_baseline_threshold(
            history_data_dict[cmd_idx])
    return latest_history_data_dict, latest_baseline_threshold_dict, history_data_dict


def post_new_perf_data(new_data_dict):
    """
    Post new perf results to database.

    Raises RuntimeError if the upload fails so that callers (e.g. pytest)
    can detect and report the failure instead of silently losing data.
    """
    data_list = list(new_data_dict.values())
    if not data_list:
        return
    try:
        print_info(
            f"Ready to post {len(data_list)} data to {TEST_INFO_PROJECT_NAME}")
        success = OpenSearchDB.postToOpenSearchDB(data_list,
                                                  TEST_INFO_PROJECT_NAME)
        if not success:
            raise RuntimeError(
                f"OpenSearchDB.postToOpenSearchDB returned False for "
                f"{TEST_INFO_PROJECT_NAME} ({len(data_list)} entries). "
                f"Check type validation and connection settings.")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to post data to {TEST_INFO_PROJECT_NAME}: {e}") from e
