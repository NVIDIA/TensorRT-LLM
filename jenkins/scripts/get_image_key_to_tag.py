#!/usr/bin/env python3
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

import json
import sys
import urllib.error
import urllib.request


def fetch_url(url):
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        return None, None


def get_latest_build_number(jenkins_base):
    for build_type in ("lastBuild", "lastCompletedBuild"):
        url = f"{jenkins_base}/{build_type}/api/json"
        status, data = fetch_url(url)
        if status == 200 and data:
            try:
                return json.loads(data)["number"]
            except (json.JSONDecodeError, KeyError):
                pass
        if build_type == "lastBuild":
            print(
                "Failed to get last build number. Trying last completed build...", file=sys.stderr
            )
    return None


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <branch_name>", file=sys.stderr)
        sys.exit(1)

    branch_name = sys.argv[1]
    jenkins_base = (
        f"https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/{branch_name}/job/L0_PostMerge"
    )
    artifactory_base = (
        f"https://urm.nvidia.com/artifactory/sw-tensorrt-generic-local/"
        f"llm-artifacts/LLM/{branch_name}/L0_PostMerge"
    )

    print(f"Fetching latest build number from Jenkins for branch: {branch_name}", file=sys.stderr)

    build_number = get_latest_build_number(jenkins_base)
    if build_number is None:
        print(
            f"Error: Could not determine the latest build number from {jenkins_base}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Latest build number: {build_number}", file=sys.stderr)

    while build_number > 0:
        artifact_url = f"{artifactory_base}/{build_number}/imageKeyToTag.json"
        print(f"Fetching: {artifact_url}", file=sys.stderr)
        status, data = fetch_url(artifact_url)
        if status == 200 and data:
            sys.stdout.write(data.decode())
            sys.exit(0)
        print(
            f"Got HTTP {status} for build {build_number}, trying build {build_number - 1}...",
            file=sys.stderr,
        )
        build_number -= 1

    print("Error: Could not find imageKeyToTag.json in any recent build", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
