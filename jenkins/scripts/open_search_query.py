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
# - queryJobEvents: Queries OpenSearch for job events with pagination support
# - getPassedTestList: Retrieves and deduplicates passed test names
# - writeTestListToFile: Writes test list to a file for further processing
#
# =============================================================================

import argparse
import json
import os
import sys

from open_search_db import OpenSearchDB


def queryJobEvents(commitID="", onlySuccess=True, stageNamePattern=""):
    """Query OpenSearch database for job events with pagination.

    Args:
        commitID: Git commit SHA to filter by (optional)
        onlySuccess: If True, only return PASSED tests (default: True)
        stageNamePattern: Non-empty regular-expression stage-name pattern to filter by

    Returns:
        List of all matching test result records
    """
    mustConditions = []
    if commitID:
        mustConditions.append({"term": {"s_trigger_mr_commit": commitID}})
    if not stageNamePattern:
        raise ValueError("stageNamePattern cannot be empty")
    mustConditions.append({"regexp": {"s_stage_name": stageNamePattern}})
    if onlySuccess:
        mustConditions.append({"term": {"s_status": "PASSED"}})

    all_results = []
    page_size = 1000
    from_index = 0

    while True:
        requestBody = {
            "query": {"bool": {"must": mustConditions}},
            "_source": [
                "s_job_name",
                "s_status",
                "s_build_id",
                "s_turtle_name",
                "s_test_name",
                "s_gpu_type",
            ],
            "size": page_size,
            "from": from_index,
        }

        formattedRequestBody = json.dumps(requestBody)
        response = OpenSearchDB.queryFromOpenSearchDB(
            formattedRequestBody, "swdl-trtllm-infra-ci-prod-test_info"
        )
        if response is None:
            print("Failed to query from OpenSearchDB")
            break
        data = response.json()

        hits = data["hits"]["hits"]
        if not hits:
            break

        all_results.extend(hits)
        from_index += page_size

        print(f"Fetched {len(all_results)} records...")

    return all_results


def writeTestListToFile(testList, fileName):
    os.makedirs(os.path.dirname(fileName), exist_ok=True)

    with open(fileName, "w") as f:
        for test in testList:
            f.write(test + "\n")


def getPassedTestList(commitID, outputFile, stageNamePattern=""):
    hits = queryJobEvents(
        commitID=commitID,
        onlySuccess=True,
        stageNamePattern=stageNamePattern,
    )
    # Use set to automatically remove duplicates
    testSet = set()
    for hit in hits:
        testSet.add(hit["_source"]["s_turtle_name"])
    testList = list(testSet)
    writeTestListToFile(testList, outputFile)


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
    getPassedTestList(
        commitID=args.commit_id,
        outputFile=args.output_file,
        stageNamePattern=args.stage_name_pattern,
    )
