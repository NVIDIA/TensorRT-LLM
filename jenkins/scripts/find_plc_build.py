#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# find_plc_build.py
#
# Finds a completed /LLM/helpers/PLCScanningSetup pre-merge source-code-scan
# build that already ran against a given commit hash, mirroring the cache
# lookup in findCachedPLCSourceScanResult() (jenkins/L0_MergeRequest.groovy).
# Useful for checking from the command line whether a rerun at a given commit
# should be able to reuse a prior PLC scan result.
# =============================================================================

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_JENKINS_BASE = (
    "https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/helpers/job/PLCScanningSetup"
)


def fetch_json(url):
    request = urllib.request.Request(url)
    user = os.environ.get("JENKINS_USER")
    token = os.environ.get("JENKINS_API_TOKEN")
    if user and token:
        credentials = base64.b64encode(f"{user}:{token}".encode()).decode()
        request.add_header("Authorization", f"Basic {credentials}")
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def build_parameters(build_info):
    params = {}
    for action in build_info.get("actions", []):
        for param in action.get("parameters", []) or []:
            params[param.get("name")] = param.get("value")
    return params


def find_plc_build(
    jenkins_base, commit, scan_mode="pre_merge", run_source_code_scanning="true", max_builds=1000
):
    # The job's default "builds" field is truncated to Jenkins' recent-builds
    # list. "allBuilds" is needed to walk the full history like
    # plcJob.getBuilds() does in findCachedPLCSourceScanResult(), bounded by
    # an explicit range so the response stays finite.
    url = (
        f"{jenkins_base}/api/json?tree=allBuilds[number,url,result,building,"
        f"actions[parameters[name,value]]]{{0,{max_builds}}}"
    )
    data = fetch_json(url)
    for build_info in data.get("allBuilds", []):
        if build_info.get("building") or build_info.get("result") is None:
            continue
        params = build_parameters(build_info)
        if (
            params.get("ref") == commit
            and params.get("scanMode") == scan_mode
            and params.get("runSourceCodeScanning") == run_source_code_scanning
        ):
            return build_info
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Find a cached PLCScanningSetup build for a given commit."
    )
    parser.add_argument(
        "--commit", required=True, help="Git commit hash (the 'ref' build parameter)"
    )
    parser.add_argument(
        "--jenkins-base",
        default=DEFAULT_JENKINS_BASE,
        help="Jenkins job URL for PLCScanningSetup (default: %(default)s)",
    )
    parser.add_argument(
        "--max-builds",
        type=int,
        default=1000,
        help="Maximum number of builds to scan, oldest to newest cutoff (default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        build_info = find_plc_build(args.jenkins_base, args.commit, max_builds=args.max_builds)
    except urllib.error.URLError as e:
        print(f"Error querying Jenkins at {args.jenkins_base}: {e}", file=sys.stderr)
        sys.exit(1)

    if build_info is None:
        print(f"No cached PLC scan build found for commit {args.commit}", file=sys.stderr)
        sys.exit(1)

    # Emit only JSON on stdout so callers (e.g. the Jenkins pipeline) can parse
    # the result without it being interleaved with diagnostic output.
    print(
        json.dumps(
            {
                "build_id": build_info["number"],
                "result": build_info["result"],
                "url": build_info["url"],
            }
        )
    )


if __name__ == "__main__":
    main()
