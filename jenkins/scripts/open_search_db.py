# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# =============================================================================
# open_search_db.py
#
# This module provides a client class for interacting with an OpenSearch database
# used for CI/CD job and test result monitoring. It defines various project index
# names, environment variable settings for the OpenSearch connection, and constants
# for queries and timeouts.
#
# The primary interface is the `OpenSearchDB` class, which provides functionality for
# querying and processing indexed test/job metadata within NVIDIA's CI infrastructure,
# such as job information, stage results, test status, and related analytics for display
# or automation.
#
# Typical usage requires setting environment variables for authentication and server
# address, as used by Jenkins or related services.
#
# =============================================================================

import hashlib
import json
import logging
import os
import time

import requests
from requests.auth import HTTPProxyAuth

PROJECT_ROOT = "swdl-trtllm-infra"
MODE = "prod"
VERSION = "v1"

# CI monitor indexes, now only support read access.
JOB_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-job_info"
STAGE_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-stage_info"
TEST_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-test_info"
JOB_MACHINE_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-job_machine_info"
FAILED_STEP_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-failed_step_info"
PR_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-pr_info"

READ_ACCESS_PROJECT_NAME = [
    JOB_PROJECT_NAME,
    STAGE_PROJECT_NAME,
    TEST_PROJECT_NAME,
    JOB_MACHINE_PROJECT_NAME,
    FAILED_STEP_PROJECT_NAME,
    PR_PROJECT_NAME,
]

WRITE_ACCESS_PROJECT_NAME = []

DISABLE_OPEN_SEARCH_DB_FOR_LOCAL_TEST = False

DEFAULT_QUERY_SIZE = 3000
DEFAULT_RETRY_COUNT = 5
DEFAULT_LOOKBACK_DAYS = 7
POST_TIMEOUT_SECONDS = 20
QUERY_TIMEOUT_SECONDS = 10

OPEN_SEARCH_DB_BASE_URL = os.getenv("OPEN_SEARCH_DB_BASE_URL", "")
OPEN_SEARCH_DB_USERNAME = os.getenv("OPEN_SEARCH_DB_CREDENTIALS_USR", "")
OPEN_SEARCH_DB_PASSWORD = os.getenv("OPEN_SEARCH_DB_CREDENTIALS_PSW", "")


class OpenSearchDB:
    logger = logging.getLogger(__name__)
    query_build_id_cache: dict = {}

    def __init__(self) -> None:
        pass

    @staticmethod
    def typeCheckForOpenSearchDB(json_data) -> bool:
        """
        Check if the data is valid for OpenSearchDB.

        :param json_data: Data to check, type dict or list.
        :return: bool, True if data is valid, False otherwise.
        """
        if isinstance(json_data, list):
            return all(
                OpenSearchDB.typeCheckForOpenSearchDB(item)
                for item in json_data)
        if not isinstance(json_data, dict):
            OpenSearchDB.logger.error(
                f"OpenSearchDB type check failed! Expected dict, got {type(json_data).__name__}"
            )
            return False

        allowed_keys = {"_id", "_project", "_shard", "_version"}
        type_map = {
            "l_": int,
            "d_": float,
            "s_": str,
            "b_": bool,
            "ts_": int,
            "flat_": dict,
            "ni_": (str, int, float),
        }

        for key, value in json_data.items():
            matched = False
            for prefix, expected_type in type_map.items():
                if key.startswith(prefix):
                    if not isinstance(value, expected_type):
                        OpenSearchDB.logger.error(
                            f"OpenSearchDB type check failed! key:{key}, value:{value} value_type:{type(value)}"
                        )
                        return False
                    matched = True
                    break
            if not matched:
                if key not in allowed_keys:
                    OpenSearchDB.logger.error(
                        f"Unknown key type! key:{key}, value_type:{type(value)}"
                    )
                    return False
        return True

    @staticmethod
    def _calculate_timestamp(days_ago) -> int:
        """
        Calculate timestamp in milliseconds.

        :param days_ago: Number of days ago.
        :return: Timestamp in milliseconds.
        """
        return int(time.time() -
                   24 * 3600 * days_ago) // (24 * 3600) * 24 * 3600 * 1000

    @staticmethod
    def add_id_of_json(data) -> None:
        """
        Add _id field to the data.

        :param data: Data to add _id field, type dict or list.
        :return: None.
        """
        if isinstance(data, list):
            for d in data:
                OpenSearchDB.add_id_of_json(d)
            return
        if not isinstance(data, dict):
            raise TypeError("data is not a dict, type:{}".format(type(data)))
        data_str = json.dumps(data, sort_keys=True, indent=2).encode("utf-8")
        data["_id"] = hashlib.md5(data_str).hexdigest()

    @staticmethod
    def postToOpenSearchDB(json_data, project) -> bool:
        """
        Post data to OpenSearchDB.

        :param json_data: Data to post, type dict or list.
        :param project: Name of the project.
        :return: bool, True if post successful, False otherwise.
        """
        if not OPEN_SEARCH_DB_BASE_URL:
            OpenSearchDB.logger.error("OPEN_SEARCH_DB_BASE_URL is not set")
            return False
        if not OPEN_SEARCH_DB_USERNAME or not OPEN_SEARCH_DB_PASSWORD:
            OpenSearchDB.logger.error(
                "OPEN_SEARCH_DB_USERNAME or OPEN_SEARCH_DB_PASSWORD is not set")
            return False
        if project not in WRITE_ACCESS_PROJECT_NAME:
            OpenSearchDB.logger.error(
                f"project {project} is not in write access project list: {json.dumps(WRITE_ACCESS_PROJECT_NAME)}"
            )
            return False
        if not OpenSearchDB.typeCheckForOpenSearchDB(json_data):
            OpenSearchDB.logger.error(
                f"OpenSearchDB type check failed! json_data:{json_data}")
            return False

        OpenSearchDB.add_id_of_json(json_data)
        json_data_dump = json.dumps(json_data)

        if DISABLE_OPEN_SEARCH_DB_FOR_LOCAL_TEST:
            OpenSearchDB.logger.info(
                f"OpenSearchDB is disabled for local test, skip posting to OpenSearchDB: {json_data_dump}"
            )
            return True

        url = f"{OPEN_SEARCH_DB_BASE_URL}/dataflow2/{project}/posting"
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "UTF-8"
        }

        for attempt in range(DEFAULT_RETRY_COUNT):
            try:
                res = requests.post(
                    url,
                    data=json_data_dump,
                    headers=headers,
                    auth=HTTPProxyAuth(OPEN_SEARCH_DB_USERNAME,
                                       OPEN_SEARCH_DB_PASSWORD),
                    timeout=POST_TIMEOUT_SECONDS,
                )
                if res.status_code in (200, 201, 202):
                    if res.status_code != 200 and project == JOB_PROJECT_NAME:
                        OpenSearchDB.logger.info(
                            f"OpenSearchDB post not 200, log:{res.status_code} {res.text}"
                        )
                    return True
                else:
                    OpenSearchDB.logger.info(
                        f"OpenSearchDB post failed, will retry, error:{res.status_code} {res.text}"
                    )
            except Exception as e:
                OpenSearchDB.logger.info(
                    f"OpenSearchDB post exception, attempt {attempt + 1} error: {e}"
                )
        OpenSearchDB.logger.error(
            f"Fail to postToOpenSearchDB after {DEFAULT_RETRY_COUNT} tries: {url}, json: {json_data_dump}, last error: {getattr(res, 'text', 'N/A') if 'res' in locals() else ''}"
        )
        return False

    @staticmethod
    def queryFromOpenSearchDB(json_data, project) -> dict:
        """
        Query data from OpenSearchDB.

        :param json_data: Data to query, type dict or list.
        :param project: Name of the project.
        :return: dict, query result.
        """
        if not OPEN_SEARCH_DB_BASE_URL:
            OpenSearchDB.logger.error("OPEN_SEARCH_DB_BASE_URL is not set")
            return {}
        if project not in READ_ACCESS_PROJECT_NAME:
            OpenSearchDB.logger.error(
                f"project {project} is not in read access project list: {json.dumps(READ_ACCESS_PROJECT_NAME)}"
            )
            return {}
        if not isinstance(json_data, str):
            json_data_dump = json.dumps(json_data)
        else:
            json_data_dump = json_data
        url = f"{OPEN_SEARCH_DB_BASE_URL}/opensearch/df-{project}-*/_search"
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "UTF-8"
        }
        retry_time = DEFAULT_RETRY_COUNT
        while retry_time:
            res = requests.get(url,
                               data=json_data_dump,
                               headers=headers,
                               timeout=QUERY_TIMEOUT_SECONDS)
            if res.status_code in [200, 201, 202]:
                return res.json()
            OpenSearchDB.logger.info(
                f"OpenSearchDB query failed, will retry, error:{res.status_code} {res.text}"
            )
            retry_time -= 1
        OpenSearchDB.logger.error(
            f"Fail to queryFromOpenSearchDB after {retry_time} retry: {url}, json: {json_data_dump}, error: {res.text}"
        )
        return {}

    @staticmethod
    def queryBuildIdFromOpenSearchDB(job_name, last_days=DEFAULT_LOOKBACK_DAYS):
        if DISABLE_OPEN_SEARCH_DB_FOR_LOCAL_TEST:
            return []
        if job_name in OpenSearchDB.query_build_id_cache:
            return OpenSearchDB.query_build_id_cache[job_name]
        json_data = {
            "size": DEFAULT_QUERY_SIZE,
            "query": {
                "range": {
                    "ts_created": {
                        "gte": OpenSearchDB._calculate_timestamp(last_days),
                    }
                }
            },
            "_source": ["ts_created", "s_job_name", "s_status", "s_build_id"],
        }
        build_ids = []
        try:
            query_res = OpenSearchDB.queryFromOpenSearchDB(
                json_data, JOB_PROJECT_NAME)
            for job in query_res["hits"]["hits"]:
                job_info = job.get("_source", {})
                if job_name == job_info.get("s_job_name"):
                    build_ids.append(job_info.get("s_build_id"))
            OpenSearchDB.query_build_id_cache[job_name] = build_ids
            return build_ids
        except Exception as e:
            OpenSearchDB.logger.warning(
                f"Failed to query build IDs from OpenSearchDB: {e}")
            return []

    @staticmethod
    def queryPRIdsFromOpenSearchDB(repo_name, last_days=DEFAULT_LOOKBACK_DAYS):
        """
        Query existing PR IDs from OpenSearchDB for a specific repository.
        Mirrors queryBuildIdFromOpenSearchDB for PR monitoring.
        """
        if DISABLE_OPEN_SEARCH_DB_FOR_LOCAL_TEST:
            return []

        cache_key = f"pr_{repo_name}"
        if cache_key in OpenSearchDB.query_build_id_cache:
            return OpenSearchDB.query_build_id_cache[cache_key]

        json_data = {
            "size": DEFAULT_QUERY_SIZE,
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "ts_created": {
                                    "gte":
                                    OpenSearchDB._calculate_timestamp(
                                        last_days),
                                }
                            }
                        },
                        {
                            "term": {
                                "s_repo_name": repo_name
                            }
                        },
                    ]
                }
            },
            "_source": ["ts_created", "s_repo_name", "l_pr_number", "s_pr_id"],
        }

        pr_numbers = []
        try:
            query_res = OpenSearchDB.queryFromOpenSearchDB(
                json_data, PR_PROJECT_NAME)
            for pr in query_res["hits"]["hits"]:
                pr_info = pr.get("_source", {})
                if repo_name == pr_info.get("s_repo_name"):
                    pr_number = pr_info.get("l_pr_number")
                    if pr_number and pr_number not in pr_numbers:
                        pr_numbers.append(pr_number)
            OpenSearchDB.query_build_id_cache[cache_key] = pr_numbers
            return pr_numbers
        except Exception as e:
            OpenSearchDB.logger.warning(
                f"Failed to query PR IDs from OpenSearchDB: {e}")
            return []
