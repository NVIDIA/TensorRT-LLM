import json
import logging
import os
import time

import requests
from requests.auth import HTTPProxyAuth

PROJECT_ROOT = "swdl-trtllm-infra"
MODE = "prod"
VERSION = "v1"

# CI monitor indexes, now only support query online and post offline.
JOB_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-job_info"
STAGE_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-stage_info"
TEST_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-test_info"
JOB_MACHINE_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-job_machine_info"
FAILED_STEP_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-failed_step_info"
PR_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-pr_info"

# post online indexes
# XXXX

DISABLE_NVDF_FOR_LOCAL_TEST = False

NVDF_BASE_URL = os.getenv("NVDF_BASE_URL", None)
NVDF_USERNAME = os.getenv("NVDF_CREDENTIALS_USR", None)
NVDF_PASSWORD = os.getenv("NVDF_CREDENTIALS_PWD", None)
if not NVDF_BASE_URL or not NVDF_USERNAME or not NVDF_PASSWORD:
    raise Exception(
        "NVDF_BASE_URL or NVDF_USERNAME or NVDF_PASSWORD is not set")


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
        assert isinstance(json_data, dict)
        for key, value in json_data.items():
            if key.startswith("l_"):
                if not isinstance(value, int):
                    NVDF.logger.fatal(
                        "NVDF type check failed! key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
                    raise Exception("typeCheckForNVDataFlow Failed")
            elif key.startswith("s_"):
                if not isinstance(value, str):
                    NVDF.logger.fatal(
                        "NVDF type check failed! key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
                    raise Exception("typeCheckForNVDataFlow Failed")
            elif key.startswith("b_"):
                if not isinstance(value, bool):
                    NVDF.logger.fatal(
                        "NVDF type check failed! key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
                    raise Exception("typeCheckForNVDataFlow Failed")
            elif key.startswith("ts_"):
                if not isinstance(value, int):
                    NVDF.logger.fatal(
                        "NVDF type check failed! key:{}, value:{} value_type:{}"
                        .format(key, value, type(value)))
                    raise Exception("typeCheckForNVDataFlow Failed")
            elif key.startswith("flat_"):
                if not isinstance(value, dict):
                    NVDF.logger.fatal(
                        "NVDF type check failed! key:{}, value_type:{}".format(
                            key, type(value)))
                    raise Exception("typeCheckForNVDataFlow Failed")
            elif key.startswith("ni_"):
                if not isinstance(value, (str, int, float)):
                    NVDF.logger.fatal(
                        "NVDF type check failed! key:{}, value_type:{}".format(
                            key, type(value)))
                    raise Exception("typeCheckForNVDataFlow Failed")
            else:
                NVDF.logger.fatal(
                    "Unknown key type! key:{}, value_type:{}".format(
                        key, type(value)))
        return json_data

    @staticmethod
    def postStageInfoToNVDataFlow(json_data):
        NVDF.postToNVDataFlow(json_data, STAGE_PROJECT_NAME)

    @staticmethod
    def postTestInfoToNVDataFlow(json_data):
        NVDF.postToNVDataFlow(json_data, TEST_PROJECT_NAME)

    @staticmethod
    def postJobInfoToNVDataFlow(json_data):
        NVDF.postToNVDataFlow(json_data, JOB_PROJECT_NAME)

    @staticmethod
    def postJobMachineInfoToNVDataFlow(json_data):
        NVDF.postToNVDataFlow(json_data, JOB_MACHINE_PROJECT_NAME)

    @staticmethod
    def postFailedStepInfoToNVDataFlow(json_data):
        NVDF.postToNVDataFlow(json_data, FAILED_STEP_PROJECT_NAME)

    @staticmethod
    def postPRInfoToNVDataFlow(json_data):
        """Post GitHub PR information to NVDF."""
        NVDF.postToNVDataFlow(json_data, PR_PROJECT_NAME)

    @staticmethod
    def postToNVDataFlow(json_data, project, rtn=True):
        NVDF.typeCheckForNVDataFlow(json_data)
        Utils.add_id_of_json(json_data)
        json_data = json.dumps(json_data)
        if DISABLE_NVDF_FOR_LOCAL_TEST:
            # NVDF.logger.info("NVDF is disabled, skip posting to NVDataFlow: {}".format(json_data))
            return
        url = "{}/dataflow2/{}/posting".format(NVDF_BASE_URL, project)
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "UTF-8"
        }
        retry_time = 5
        while retry_time:
            res = requests.post(
                url,
                data=json_data,
                headers=headers,
                auth=HTTPProxyAuth(NVDF_USERNAME, NVDF_PASSWORD),
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

    @staticmethod
    def queryFromNVDataFlow(json_data, project):
        if not isinstance(json_data, str):
            json_data = json.dumps(json_data)
        url = "{}/opensearch/df-{}-*/_search".format(NVDF_BASE_URL, project)
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "UTF-8"
        }
        retry_time = 5
        while retry_time:
            res = requests.get(url, data=json_data, headers=headers, timeout=10)
            if res.status_code in [200, 201, 202]:
                return res.json()
            NVDF.logger.info(
                "nvdf query failed, will retry, error:{} {}".format(
                    res.status_code, res.text))
            retry_time -= 1
        NVDF.logger.error(
            "Fail to queryFromNVDataFlow after {} retry: {}, json: {}, error: {}"
            .format(retry_time, url, json_data, res.text))

    @staticmethod
    def queryBuildIdFromNVDataFlow(job_name, last_days=10):
        if DISABLE_NVDF_FOR_LOCAL_TEST:
            return []
        if job_name in NVDF.query_build_id_cache:
            return NVDF.query_build_id_cache[job_name]
        json_data = {
            "size": 3000,
            "query": {
                "range": {
                    "ts_created": {
                        "gte":
                        int(time.time() - 24 * 3600 * last_days) //
                        (24 * 3600) * 24 * 3600 * 1000,
                    }
                }
            },
            "_source": ["ts_created", "s_job_name", "s_status", "s_build_id"],
        }
        build_ids = []
        query_res = NVDF.queryFromNVDataFlow(json_data, JOB_PROJECT_NAME)
        for job in query_res["hits"]["hits"]:
            job_info = job.get("_source", {})
            if job_name == job_info.get("s_job_name"):
                build_ids.append(job_info.get("s_build_id"))
        NVDF.query_build_id_cache[job_name] = build_ids
        return build_ids

    @staticmethod
    def queryPRIdsFromNVDataFlow(repo_name, last_days=10):
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
            "size": 3000,
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "ts_created": {
                                    "gte":
                                    int(time.time() - 24 * 3600 * last_days) //
                                    (24 * 3600) * 24 * 3600 * 1000,
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
        except Exception as e:
            NVDF.logger.warning(f"Failed to query PR IDs from NVDF: {e}")
            return []

        NVDF.query_build_id_cache[cache_key] = pr_numbers
        return pr_numbers
