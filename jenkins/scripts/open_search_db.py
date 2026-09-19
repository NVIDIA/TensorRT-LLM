# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

try:
    import requests
    from requests.auth import HTTPProxyAuth
except ImportError:
    # Lightweight CI pods (e.g. the Setup Environment pod) may not ship
    # requests. Keep the module importable so requests-free callers can still
    # use the constants and helpers; postToOpenSearchDB itself needs requests.
    requests = None
    HTTPProxyAuth = None

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
PERF_SANITY_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-perf_sanity_info"
CBTS_PROJECT_NAME = f"{PROJECT_ROOT}-ci-{MODE}-cbts_info"

READ_ACCESS_PROJECT_NAME = [
    JOB_PROJECT_NAME,
    STAGE_PROJECT_NAME,
    TEST_PROJECT_NAME,
    JOB_MACHINE_PROJECT_NAME,
    FAILED_STEP_PROJECT_NAME,
    PR_PROJECT_NAME,
    PERF_SANITY_PROJECT_NAME,
    CBTS_PROJECT_NAME,
]

WRITE_ACCESS_PROJECT_NAME = [
    PERF_SANITY_PROJECT_NAME,
    CBTS_PROJECT_NAME,
]

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

    # Structured log prefixes for OpenSearch DB access.  External log
    # aggregation systems can filter on [OpenSearchDB][ERROR] without
    # relying on the DB itself.  Each log line is guaranteed single-line
    # (newlines replaced) so grep always captures the full message.
    #
    # Tag format after the level: (op=query|post, cat=<category>, db=<project>[, status=N][, attempt=M/N])
    _LOG_TAG = "[OpenSearchDB]"

    def __init__(self) -> None:
        pass

    @staticmethod
    def _sanitize_log(msg: str, max_len: int = 1000) -> str:
        """Collapse newlines and cap length for single-line log output."""
        return msg.replace("\n", " | ").replace("\r", "")[:max_len]

    @staticmethod
    def _log(
        level: str,
        op: str,
        db: str,
        msg: str,
        *,
        cat: str | None = None,
        status: int | None = None,
        attempt: int | None = None,
        total: int | None = None,
        url: str | None = None,
    ) -> None:
        """Emit a single-line structured log entry for an OpenSearch DB operation.

        :param level: Logging level: 'debug', 'info', 'warning', or 'error'.
        :param op: Operation type, e.g. 'query' or 'post'.
        :param db: Target project/index name (sanitized automatically).
        :param msg: Human-readable message body (newlines collapsed automatically).
        :param cat: Error category, e.g. 'config', 'network', 'exhausted'.
        :param status: HTTP status code, if applicable.
        :param attempt: Current retry attempt number (pair with total).
        :param total: Total retry count (pair with attempt).
        :param url: Request URL appended at the end of the message (sanitized).
        """
        _LEVEL_TAG = {
            "debug": "DEBUG",
            "info": "INFO",
            "warning": "WARN",
            "error": "ERROR",
        }
        try:
            level_lower = level.lower()
            log_method = getattr(OpenSearchDB.logger, level_lower, None)
            if log_method is None:
                OpenSearchDB.logger.error(
                    f"[OpenSearchDB][ERROR] _log() called with invalid level={level!r}; "
                    f"op={op!r}, db={db!r}, msg={msg!r}")
                log_method = OpenSearchDB.logger.error
                level_lower = "error"

            parts = [f"op={op}"]
            if cat is not None:
                parts.append(f"cat={cat}")
            if status is not None:
                parts.append(f"status={status}")
            parts.append(f"db={OpenSearchDB._sanitize_log(db, max_len=200)}")
            if attempt is not None and total is not None:
                parts.append(f"attempt={attempt}/{total}")
            elif attempt is not None or total is not None:
                OpenSearchDB.logger.error(
                    f"[OpenSearchDB][ERROR] _log() requires both attempt and total, "
                    f"or neither; got attempt={attempt!r}, total={total!r}")

            body = OpenSearchDB._sanitize_log(msg)
            if url is not None:
                body += f", url: {OpenSearchDB._sanitize_log(url)}"

            tag = _LEVEL_TAG.get(level_lower, level_lower.upper())
            log_line = (f"{OpenSearchDB._LOG_TAG}[{tag}]"
                        f"({', '.join(parts)}) {body}")
            log_method(log_line)
        except Exception as exc:
            # _log itself must never raise — emit a raw fallback record
            try:
                OpenSearchDB.logger.error(
                    f"[OpenSearchDB][ERROR] _log() internal error "
                    f"({type(exc).__name__}: {exc}); "
                    f"original: level={level!r} op={op!r} db={db!r}")
            except Exception:
                pass  # absolute last resort

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
            OpenSearchDB.logger.info(
                f"OpenSearchDB type check failed! Expected dict, got {type(json_data).__name__}"
            )
            return False

        allowed_keys = {
            "@timestamp", "_id", "_index", "_score", "_type", "_project",
            "_shard", "_version"
        }
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
                        OpenSearchDB.logger.info(
                            f"OpenSearchDB type check failed! key:{key}, value:{value} value_type:{type(value)}"
                        )
                        return False
                    matched = True
                    break
            if not matched:
                if key not in allowed_keys:
                    OpenSearchDB.logger.info(
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
    def _postOnce(url, data, headers, use_poc_db):
        """Send a single POST request and return (status_code, body_text).

        Uses requests when available, otherwise falls back to urllib so
        requests-free CI pods (e.g. the Setup Environment pod) can still post.
        Connection-level failures propagate so the caller can retry.

        :param url: Target URL.
        :param data: Request body (str).
        :param headers: Request headers (dict).
        :param use_poc_db: Skip proxy auth when posting to the POC DB.
        :return: tuple(int status_code, str body_text).
        """
        if requests is not None:
            args = {
                "url": url,
                "data": data,
                "headers": headers,
                "timeout": POST_TIMEOUT_SECONDS,
            }
            if not use_poc_db:
                args["auth"] = HTTPProxyAuth(OPEN_SEARCH_DB_USERNAME,
                                             OPEN_SEARCH_DB_PASSWORD)
            res = requests.post(**args)
            return res.status_code, res.text

        import base64
        import urllib.error
        import urllib.request

        post_headers = dict(headers)
        if not use_poc_db:
            token = base64.b64encode(
                f"{OPEN_SEARCH_DB_USERNAME}:{OPEN_SEARCH_DB_PASSWORD}".encode(
                )).decode()
            post_headers["Proxy-Authorization"] = f"Basic {token}"
        req = urllib.request.Request(url,
                                     data=data.encode("utf-8"),
                                     headers=post_headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req,
                                        timeout=POST_TIMEOUT_SECONDS) as resp:
                return resp.status, ""
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

    @staticmethod
    def postToOpenSearchDB(json_data, project) -> bool:
        """
        Post data to OpenSearchDB.

        :param json_data: Data to post, type dict or list.
        :param project: Name of the project.
        :return: bool, True if post successful, False otherwise.
                 This function never raises exceptions.
        """
        use_poc_db = "sandbox" in project

        if DISABLE_OPEN_SEARCH_DB_FOR_LOCAL_TEST:
            OpenSearchDB._log("info", "post", project,
                              "disabled for local test, skipping")
            return True

        if not OPEN_SEARCH_DB_BASE_URL:
            OpenSearchDB._log("error",
                              "post",
                              project,
                              "OPEN_SEARCH_DB_BASE_URL is not set",
                              cat="config")
            return False
        if not use_poc_db and (not OPEN_SEARCH_DB_USERNAME
                               or not OPEN_SEARCH_DB_PASSWORD):
            OpenSearchDB._log(
                "error",
                "post",
                project,
                "OPEN_SEARCH_DB_USERNAME or OPEN_SEARCH_DB_PASSWORD is not set",
                cat="config")
            return False
        if not use_poc_db and project not in WRITE_ACCESS_PROJECT_NAME:
            OpenSearchDB._log(
                "error",
                "post",
                project,
                f"project not in write access list: {json.dumps(WRITE_ACCESS_PROJECT_NAME)}",
                cat="config",
            )
            return False
        if not OpenSearchDB.typeCheckForOpenSearchDB(json_data):
            OpenSearchDB._log("error",
                              "post",
                              project,
                              "data type check failed",
                              cat="type_check")
            return False

        try:
            json_data_dump = json.dumps(json_data)
        except (TypeError, ValueError) as e:
            OpenSearchDB._log("error", "post", project, str(e), cat="serialize")
            return False

        url = f"{OPEN_SEARCH_DB_BASE_URL}/dataflow2/{project}/posting"
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "UTF-8",
        }

        last_error = "N/A"
        for attempt in range(1, DEFAULT_RETRY_COUNT + 1):
            try:
                status_code, last_text = OpenSearchDB._postOnce(
                    url, json_data_dump, headers, use_poc_db)
                if status_code in (200, 201, 202):
                    if status_code != 200 and project == JOB_PROJECT_NAME:
                        OpenSearchDB._log(
                            "info", "post", project,
                            f"accepted with status {status_code}")
                    return True

                last_error = f"HTTP {status_code}: {last_text[:200]}"

                if OpenSearchDB._is_non_retryable(status_code):
                    OpenSearchDB._log("error",
                                      "post",
                                      project,
                                      last_error,
                                      cat="non_retryable",
                                      status=status_code,
                                      url=url)
                    return False

                if attempt < DEFAULT_RETRY_COUNT:
                    OpenSearchDB._log("warning",
                                      "post",
                                      project,
                                      "will retry",
                                      cat="transient",
                                      status=status_code,
                                      attempt=attempt,
                                      total=DEFAULT_RETRY_COUNT)
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                OpenSearchDB._log("warning",
                                  "post",
                                  project,
                                  last_error,
                                  cat="network",
                                  attempt=attempt,
                                  total=DEFAULT_RETRY_COUNT,
                                  url=url)

            if attempt < DEFAULT_RETRY_COUNT:
                time.sleep(min(attempt * 2, 10))

        OpenSearchDB._log(
            "error",
            "post",
            project,
            f"failed after {DEFAULT_RETRY_COUNT} attempts, last error: {last_error}",
            cat="exhausted",
            url=url,
        )
        return False

    # HTTP status codes where retry makes sense (transient server/network issues)
    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    # Any 4xx except 429 is a client error — the request itself is wrong, retry won't help.
    # Use _is_non_retryable() rather than checking this set directly.
    _NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 405, 413, 414}

    @staticmethod
    def _is_non_retryable(status_code: int) -> bool:
        """Return True for 4xx client errors (except 429 Too Many Requests)."""
        return 400 <= status_code < 500 and status_code != 429

    @staticmethod
    def queryFromOpenSearchDB(json_data, project):
        """
        Query data from OpenSearchDB.

        :param json_data: Data to query, type dict or list.
        :param project: Name of the project.
        :return: requests.Response on success, None on failure.
                 This function never raises exceptions.
        """
        if requests is None:
            OpenSearchDB._log("error",
                              "query",
                              project,
                              "requests package is not available",
                              cat="config")
            return None

        use_poc_db = "sandbox" in project
        if not OPEN_SEARCH_DB_BASE_URL:
            OpenSearchDB._log("error",
                              "query",
                              project,
                              "OPEN_SEARCH_DB_BASE_URL is not set",
                              cat="config")
            return None
        if not use_poc_db and project not in READ_ACCESS_PROJECT_NAME:
            OpenSearchDB._log(
                "error",
                "query",
                project,
                f"project not in read access list: {json.dumps(READ_ACCESS_PROJECT_NAME)}",
                cat="config",
            )
            return None
        try:
            if not isinstance(json_data, str):
                json_data_dump = json.dumps(json_data)
            else:
                json_data_dump = json_data
        except (TypeError, ValueError) as e:
            OpenSearchDB._log("error",
                              "query",
                              project,
                              str(e),
                              cat="serialize")
            return None

        url = f"{OPEN_SEARCH_DB_BASE_URL}/opensearch/df-{project}-*/_search"
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "UTF-8",
        }
        last_error = "N/A"
        for attempt in range(1, DEFAULT_RETRY_COUNT + 1):
            try:
                res = requests.get(url,
                                   data=json_data_dump,
                                   headers=headers,
                                   timeout=QUERY_TIMEOUT_SECONDS)
                if res.status_code in (200, 201, 202):
                    return res

                last_error = f"HTTP {res.status_code}: {res.text[:200]}"

                # Client errors (4xx except 429) — input is wrong, retry won't help
                if OpenSearchDB._is_non_retryable(res.status_code):
                    OpenSearchDB._log("error",
                                      "query",
                                      project,
                                      last_error,
                                      cat="non_retryable",
                                      status=res.status_code,
                                      url=url)
                    return None

                # Server errors (5xx, 429) — transient, worth retrying
                if attempt < DEFAULT_RETRY_COUNT:
                    OpenSearchDB._log("warning",
                                      "query",
                                      project,
                                      "will retry",
                                      cat="transient",
                                      status=res.status_code,
                                      attempt=attempt,
                                      total=DEFAULT_RETRY_COUNT)
            except requests.exceptions.RequestException as e:
                last_error = f"{type(e).__name__}: {e}"
                OpenSearchDB._log("warning",
                                  "query",
                                  project,
                                  last_error,
                                  cat="network",
                                  attempt=attempt,
                                  total=DEFAULT_RETRY_COUNT,
                                  url=url)

            if attempt < DEFAULT_RETRY_COUNT:
                time.sleep(min(attempt * 2, 10))

        OpenSearchDB._log(
            "error",
            "query",
            project,
            f"failed after {DEFAULT_RETRY_COUNT} attempts, last error: {last_error}",
            cat="exhausted",
            url=url,
        )
        return None

    @staticmethod
    def queryPerfDataFromOpenSearchDB(project_name,
                                      must_clauses,
                                      size=DEFAULT_QUERY_SIZE,
                                      must_not_clauses=None):
        """
        Query perf data from OpenSearchDB using must and must_not clauses.

        :param project_name: Name of the project.
        :param must_clauses: List of must clauses for query.
        :param size: Query size.
        :param must_not_clauses: List of must_not clauses for query.
        :return: list of data dicts, empty list if no data, None on error.
        """
        if DISABLE_OPEN_SEARCH_DB_FOR_LOCAL_TEST:
            return []
        if must_clauses is None:
            must_clauses = []
        if must_not_clauses is None:
            must_not_clauses = []
        if not isinstance(must_clauses, list):
            OpenSearchDB.logger.info(
                f"Invalid must_clauses type: {type(must_clauses).__name__}")
            return None
        if not isinstance(must_not_clauses, list):
            OpenSearchDB.logger.info(
                f"Invalid must_not_clauses type: {type(must_not_clauses).__name__}"
            )
            return None

        bool_query = {"must": must_clauses}
        if must_not_clauses:
            bool_query["must_not"] = must_not_clauses

        json_data = {
            "query": {
                "bool": bool_query
            },
            "size": size,
        }

        data_list = []
        try:
            res = OpenSearchDB.queryFromOpenSearchDB(json_data, project_name)
            if res is None:
                OpenSearchDB.logger.info(
                    f"Failed to query from {project_name}, returned no response"
                )
                return None
            payload = res.json().get("hits", {}).get("hits", [])
            if len(payload) == 0:
                OpenSearchDB.logger.info(
                    f"No data found in {project_name}, returned empty list")
                return []
            for hit in payload:
                data_dict = hit.get("_source", {})
                data_dict["_id"] = hit.get("_id", "")
                if data_dict["_id"] == "":
                    OpenSearchDB.logger.info(
                        f"Failed to query from {project_name}, returned data with no _id"
                    )
                    return None
                data_list.append(data_dict)
            OpenSearchDB.logger.info(
                f"Successfully queried from {project_name}, queried {len(data_list)} entries"
            )
            return data_list
        except Exception as e:
            OpenSearchDB.logger.warning(
                f"Failed to query from {project_name}, returned error: {e}")
            return None

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
            if query_res is None:
                return []
            query_res = query_res.json()
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
            if query_res is None:
                return []
            query_res = query_res.json()
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
