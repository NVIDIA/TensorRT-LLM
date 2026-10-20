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
# open_search_query.py
#
# This module provides functions to query the OpenSearch database for passed
# test results from previous pipeline runs. It retrieves test names that have
# passed for a given commit ID and stage-name patterns, which can be reused to
# skip redundant test execution in subsequent runs.
#
# Main functionality:
# - queryJobEvents: Queries distinct names with cursor pagination
# - getPassedTestList: Retrieves a sorted, deduplicated passed-test list
# - writeTestListToFile: Writes test list to a file for further processing
#
# =============================================================================

import argparse
import json
import os
import sys

from open_search_db import OpenSearchDB

PARTIAL_QUERY_EXIT_CODE = 3


class _IncompleteQueryError(RuntimeError):
    def __init__(self, message: str, test_names: list[str]) -> None:
        super().__init__(message)
        self.test_names = test_names


def queryJobEvents(commitID="", stageNamePattern="", onlySuccess=True) -> list[str]:
    """Query distinct test names using composite aggregation pagination."""
    mustConditions = []
    if commitID:
        mustConditions.append({"term": {"s_trigger_mr_commit": commitID}})
    if not stageNamePattern:
        raise ValueError("stageNamePattern cannot be empty")
    mustConditions.append({"regexp": {"s_stage_name": stageNamePattern}})
    if onlySuccess:
        mustConditions.append({"term": {"s_status": "PASSED"}})

    testSet = set()
    page_size = 1000
    after_key = None
    try:
        while True:
            composite = {
                "size": page_size,
                "sources": [{"test_name": {"terms": {"field": "s_turtle_name", "order": "asc"}}}],
            }
            if after_key is not None:
                composite["after"] = after_key
            requestBody = {
                "query": {"bool": {"must": mustConditions}},
                "size": 0,
                "aggs": {"passed_tests": {"composite": composite}},
            }
            formattedRequestBody = json.dumps(requestBody)
            response = OpenSearchDB.queryFromOpenSearchDB(
                formattedRequestBody, "swdl-trtllm-infra-ci-prod-test_info"
            )
            if response is None:
                raise RuntimeError("OpenSearch passed-test query failed")
            data = response.json()
            if data.get("timed_out") or data.get("_shards", {}).get("failed", 0):
                raise RuntimeError("OpenSearch passed-test query timed out or had failed shards")

            aggregation = data["aggregations"]["passed_tests"]
            buckets = aggregation["buckets"]
            if not buckets:
                break
            next_after_key = aggregation.get("after_key")
            if (
                after_key is not None
                and next_after_key
                and next_after_key["test_name"] <= after_key["test_name"]
            ):
                raise RuntimeError("OpenSearch passed-test pagination cursor did not advance")
            page_names = {bucket["key"]["test_name"] for bucket in buckets}
            testSet.update(page_names)
            print(f"Fetched {len(testSet)} distinct passed test names...")

            if not next_after_key:
                break
            after_key = next_after_key
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
        raise _IncompleteQueryError(str(error), sorted(testSet)) from error

    return sorted(testSet)


def writeTestListToFile(testList, fileName):
    os.makedirs(os.path.dirname(fileName), exist_ok=True)

    with open(fileName, "w") as f:
        for test in testList:
            f.write(test + "\n")


def getPassedTestList(commitID, stageNamePattern, outputFile) -> bool:
    """Write confirmed passed tests and return whether the query was complete."""
    try:
        testList = queryJobEvents(commitID, stageNamePattern)
    except _IncompleteQueryError as error:
        writeTestListToFile(error.test_names, outputFile)
        print(
            f"WARNING: OpenSearch query incomplete; retained {len(error.test_names)} "
            f"test names from successful pages: {error}",
            file=sys.stderr,
        )
        return False
    writeTestListToFile(testList, outputFile)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--commit-id", required=True, help="Commit ID")
    parser.add_argument(
        "--stage-name-pattern",
        required=True,
        help="Regular-expression stage-name pattern to query",
    )
    parser.add_argument("--output-file", required=True, help="Output File")
    args = parser.parse_args(sys.argv[1:])
    if not args.stage_name_pattern:
        parser.error("--stage-name-pattern cannot be empty")
    complete = getPassedTestList(
        commitID=args.commit_id,
        stageNamePattern=args.stage_name_pattern,
        outputFile=args.output_file,
    )
    sys.exit(0 if complete else PARTIAL_QUERY_EXIT_CODE)
