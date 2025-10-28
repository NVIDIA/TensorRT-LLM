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

DISABLE_NVDF_FOR_LOCAL_TEST = False

DEFAULT_QUERY_SIZE = 3000
DEFAULT_RETRY_COUNT = 5
DEFAULT_LOOKBACK_DAYS = 7
POST_TIMEOUT_SECONDS = 20
QUERY_TIMEOUT_SECONDS = 10

NVDF_BASE_URL = os.getenv("NVDF_BASE_URL", "")
NVDF_USERNAME = os.getenv("NVDF_CREDENTIALS_USR", "")
NVDF_PASSWORD = os.getenv("NVDF_CREDENTIALS_PSW", "")


class NVDF:
    logger = logging.getLogger(__name__)
    query_build_id_cache: dict = {}

    def __init__(self) -> None:
        pass

    @staticmethod
    def typeCheckForNVDataFlow(json_data):
        if isinstance(json_data, list):
            for data in json_data:
                NVDF.typeCheckForNVDataFlow(data)
            return json_data
        if not isinstance(json_data, dict):
            raise TypeError(f"Expected dict, got {type(json_data).__name__}")
        for key, value in json_data.items():
            if key.startswith("l_"):
                if not isinstance(value, int):
                    NVDF.logger.fatal(
                        "NVDF type check failed! key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
                    raise Exception(
                        "typeCheckForNVDataFlow Failed, key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
            elif key.startswith("s_"):
                if not isinstance(value, str):
                    NVDF.logger.fatal(
                        "NVDF type check failed! key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
                    raise Exception(
                        "typeCheckForNVDataFlow Failed, key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
            elif key.startswith("b_"):
                if not isinstance(value, bool):
                    NVDF.logger.fatal(
                        "NVDF type check failed! key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
                    raise Exception(
                        "typeCheckForNVDataFlow Failed, key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
            elif key.startswith("ts_"):
                if not isinstance(value, int):
                    NVDF.logger.fatal(
                        "NVDF type check failed! key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
                    raise Exception(
                        "typeCheckForNVDataFlow Failed, key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
            elif key.startswith("flat_"):
                if not isinstance(value, dict):
                    NVDF.logger.fatal(
                        "NVDF type check failed! key:{}, value_type:{}".format(
                            key, type(value)))
                    raise Exception(
                        "typeCheckForNVDataFlow Failed, key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
            elif key.startswith("ni_"):
                if not isinstance(value, (str, int, float)):
                    NVDF.logger.fatal(
                        "NVDF type check failed! key:{}, value_type:{}".format(
                            key, type(value)))
                    raise Exception(
                        "typeCheckForNVDataFlow Failed, key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
            else:
                NVDF.logger.fatal(
                    "Unknown key type! key:{}, value_type:{}".format(
                        key, type(value)))
        return json_data

    @staticmethod
    def _calculate_timestamp(days_ago):
        return int(time.time() -
                   24 * 3600 * days_ago) // (24 * 3600) * 24 * 3600 * 1000

    @staticmethod
    def add_id_of_json(data):
        if isinstance(data, list):
            for d in data:
                NVDF.add_id_of_json(d)
            return
        if not isinstance(data, dict):
            raise TypeError("data is not a dict, type:{}".format(type(data)))
        data_str = json.dumps(data, sort_keys=True, indent=2).encode("utf-8")
        data["_id"] = hashlib.md5(data_str).hexdigest()

    @staticmethod
    def postToNVDataFlow(json_data, project):
        if not NVDF_BASE_URL:
            raise Exception("NVDF_BASE_URL is not set")
        if not NVDF_USERNAME or not NVDF_PASSWORD:
            raise Exception("NVDF_USERNAME or NVDF_PASSWORD is not set")
        if project not in WRITE_ACCESS_PROJECT_NAME:
            raise Exception(
                "project {} is not in write access project list: {}".format(
                    project, json.dumps(WRITE_ACCESS_PROJECT_NAME)))
        NVDF.typeCheckForNVDataFlow(json_data)
        NVDF.add_id_of_json(json_data)
        json_data = json.dumps(json_data)
        if DISABLE_NVDF_FOR_LOCAL_TEST:
            NVDF.logger.info(
                "NVDF is disabled for local test, skip posting to NVDataFlow: {}"
                .format(json_data))
            return None
        url = "{}/dataflow2/{}/posting".format(NVDF_BASE_URL, project)
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "UTF-8"
        }
        retry_time = DEFAULT_RETRY_COUNT
        while retry_time:
            res = requests.post(
                url,
                data=json_data,
                headers=headers,
                auth=HTTPProxyAuth(NVDF_USERNAME, NVDF_PASSWORD),
                timeout=POST_TIMEOUT_SECONDS,
            )
            if res.status_code in [200, 201, 202]:
                if res.status_code != 200 and project == JOB_PROJECT_NAME:
                    NVDF.logger.info("nvdf post not 200, log:{} {}".format(
                        res.status_code, res.text))
                return
            NVDF.logger.info("nvdf post failed, will retry, error:{} {}".format(
                res.status_code, res.text))
            retry_time -= 1
        NVDF.logger.error(
            "Fail to postToNVDataFlow after {} retry: {}, json: {}, error: {}".
            format(retry_time, url, json_data, res.text))
        return None

    @staticmethod
    def queryFromNVDataFlow(json_data, project):
        if not NVDF_BASE_URL:
            raise Exception("NVDF_BASE_URL is not set")
        if project not in READ_ACCESS_PROJECT_NAME:
            raise Exception(
                "project {} is not in read access project list: {}".format(
                    project, json.dumps(READ_ACCESS_PROJECT_NAME)))
        if not isinstance(json_data, str):
            json_data = json.dumps(json_data)
        url = "{}/opensearch/df-{}-*/_search".format(NVDF_BASE_URL, project)
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "UTF-8"
        }
        retry_time = DEFAULT_RETRY_COUNT
        while retry_time:
            res = requests.get(url,
                               data=json_data,
                               headers=headers,
                               timeout=QUERY_TIMEOUT_SECONDS)
            if res.status_code in [200, 201, 202]:
                return res.json()
            NVDF.logger.info(
                "nvdf query failed, will retry, error:{} {}".format(
                    res.status_code, res.text))
            retry_time -= 1
        NVDF.logger.error(
            "Fail to queryFromNVDataFlow after {} retry: {}, json: {}, error: {}"
            .format(retry_time, url, json_data, res.text))
        return None

    @staticmethod
    def queryBuildIdFromNVDataFlow(job_name, last_days=DEFAULT_LOOKBACK_DAYS):
        if DISABLE_NVDF_FOR_LOCAL_TEST:
            return []
        if job_name in NVDF.query_build_id_cache:
            return NVDF.query_build_id_cache[job_name]
        json_data = {
            "size": DEFAULT_QUERY_SIZE,
            "query": {
                "range": {
                    "ts_created": {
                        "gte": NVDF._calculate_timestamp(last_days),
                    }
                }
            },
            "_source": ["ts_created", "s_job_name", "s_status", "s_build_id"],
        }
        build_ids = []
        try:
            query_res = NVDF.queryFromNVDataFlow(json_data, JOB_PROJECT_NAME)
            for job in query_res["hits"]["hits"]:
                job_info = job.get("_source", {})
                if job_name == job_info.get("s_job_name"):
                    build_ids.append(job_info.get("s_build_id"))
            NVDF.query_build_id_cache[job_name] = build_ids
            return build_ids
        except Exception as e:
            NVDF.logger.warning(f"Failed to query build IDs from NVDF: {e}")
            return []

    @staticmethod
    def queryPRIdsFromNVDataFlow(repo_name, last_days=DEFAULT_LOOKBACK_DAYS):
        """
        Query existing PR IDs from NVDF for a specific repository.
        Mirrors queryBuildIdFromNVDataFlow for PR monitoring.
        """
        if DISABLE_NVDF_FOR_LOCAL_TEST:
            return []

        cache_key = f"pr_{repo_name}"
        if cache_key in NVDF.query_build_id_cache:
            return NVDF.query_build_id_cache[cache_key]

        json_data = {
            "size": DEFAULT_QUERY_SIZE,
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "ts_created": {
                                    "gte": NVDF._calculate_timestamp(last_days),
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
            query_res = NVDF.queryFromNVDataFlow(json_data, PR_PROJECT_NAME)
            for pr in query_res["hits"]["hits"]:
                pr_info = pr.get("_source", {})
                if repo_name == pr_info.get("s_repo_name"):
                    pr_number = pr_info.get("l_pr_number")
                    if pr_number and pr_number not in pr_numbers:
                        pr_numbers.append(pr_number)
            NVDF.query_build_id_cache[cache_key] = pr_numbers
            return pr_numbers
        except Exception as e:
            NVDF.logger.warning(f"Failed to query PR IDs from NVDF: {e}")
            return []


print(NVDF.queryBuildIdFromNVDataFlow("LLM/main/L0_MergeRequest_PR"))
