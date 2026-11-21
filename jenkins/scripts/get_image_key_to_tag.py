#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# A build's image-tag map is read with the read helper, which looks for it in the primary artifact
# location (LLM_ARTIFACTS_PRIMARY) first and in the other one only when it is absent there,
# authenticating with the ARTIFACTORY_NETRC file. The helper is the one beside this script, so a
# copy of this script needs the helper copied beside it.
_FETCH_BUILD_ARTIFACT = Path(__file__).resolve().parent / "fetch_build_artifact.sh"
# Exit codes of the read helper that this script acts on, by the names the helper gives them;
# tests/unittest/scripts/test_fetch_build_artifact.py checks that they match.
_FETCH_ABSENT = 2
_FETCH_REJECTED = 3
_FETCH_TRANSFER_FAILED = 4
_FETCH_USAGE = 64

_IMAGE_MAP_NAME = "imageKeyToTag.json"

# What each failure exit status of the read helper means.
_HELPER_FAILURES = {
    _FETCH_REJECTED: "the primary location rejected the request",
    _FETCH_TRANSFER_FAILED: (
        "the transfer failed after retries, or the secondary location failed or rejected the request"
    ),
    _FETCH_USAGE: "a usage, path, or configuration error",
}


def _helper_failure(artifact_path: str, status: int) -> RuntimeError:
    """The error for a read of `artifact_path` that the read helper ended with `status`."""
    reason = _HELPER_FAILURES.get(status)
    detail = f": {reason}" if reason else ""
    return RuntimeError(
        f"{_FETCH_BUILD_ARTIFACT.name} could not read {artifact_path}{detail} "
        f"(exit status {status}); its message above says why."
    )


def _image_map_path(pipeline_name: str, build_number: int) -> str:
    """Legacy path of a build's image-tag map, which the read helper maps to each location."""
    return f"sw-tensorrt-generic/llm-artifacts/{pipeline_name}/{build_number}/{_IMAGE_MAP_NAME}"


def _read_image_map(pipeline_name: str, build_number: int) -> bytes | None:
    """Content of a build's image-tag map, or None when neither location has it.

    The helper's stdout is discarded, since this script's stdout carries only the map; its stderr,
    which names the location that served the map or says why it could not, goes to the job log.

    Raises:
        RuntimeError: The helper could not be run, or exited with a status other than success and
            FETCH_ABSENT.
    """
    artifact_path = _image_map_path(pipeline_name, build_number)
    with tempfile.TemporaryDirectory(prefix="image-key-to-tag-") as tmp:
        dest = Path(tmp) / _IMAGE_MAP_NAME
        try:
            status = subprocess.run(
                ["bash", str(_FETCH_BUILD_ARTIFACT), artifact_path, str(dest)],
                stdout=subprocess.DEVNULL,
                check=False,
            ).returncode
            content = dest.read_bytes() if status == 0 else None
        except OSError as e:
            raise RuntimeError(
                f"{_FETCH_BUILD_ARTIFACT.name} could not read {artifact_path}: {e}"
            ) from e
    if status in (0, _FETCH_ABSENT):
        return content
    raise _helper_failure(artifact_path, status)


def _read_image_map_or_exit(pipeline_name: str, build_number: int) -> bytes | None:
    """Content of a build's image-tag map, or None when neither location has it.

    A read that fails any other way ends the script with status 1: a rejected request or a failed
    transfer says nothing about whether the build has a map, and would fail for another build too.
    """
    try:
        return _read_image_map(pipeline_name, build_number)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


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
    if len(sys.argv) == 2:
        # Branch mode: auto-discover latest build number
        branch_name = sys.argv[1]
        jenkins_base = f"https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/{branch_name}/job/L0_PostMerge"
        pipeline_name = f"LLM/{branch_name}/L0_PostMerge"

        print(
            f"Fetching latest build number from Jenkins for branch: {branch_name}", file=sys.stderr
        )

        build_number = get_latest_build_number(jenkins_base)
        if build_number is None:
            print(
                f"Error: Could not determine the latest build number from {jenkins_base}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Latest build number: {build_number}", file=sys.stderr)

        while build_number > 0:
            print(f"Fetching: {_image_map_path(pipeline_name, build_number)}", file=sys.stderr)
            data = _read_image_map_or_exit(pipeline_name, build_number)
            if data:
                sys.stdout.write(data.decode())
                sys.exit(0)
            print(
                f"No imageKeyToTag.json for build {build_number}, "
                f"trying build {build_number - 1}...",
                file=sys.stderr,
            )
            build_number -= 1

        print("Error: Could not find imageKeyToTag.json in any recent build", file=sys.stderr)
        sys.exit(1)

    elif len(sys.argv) == 3:
        # Pipeline mode: pipeline name and build number provided directly
        pipeline_name = sys.argv[1]
        try:
            build_number = int(sys.argv[2])
        except ValueError:
            print(f"Error: build number must be an integer, got '{sys.argv[2]}'", file=sys.stderr)
            sys.exit(1)

        print(f"Fetching: {_image_map_path(pipeline_name, build_number)}", file=sys.stderr)
        data = _read_image_map_or_exit(pipeline_name, build_number)
        if data:
            sys.stdout.write(data.decode())
            sys.exit(0)
        print(
            f"Error: Could not fetch imageKeyToTag.json"
            f" for pipeline '{pipeline_name}' build {build_number}",
            file=sys.stderr,
        )
        sys.exit(1)

    else:
        print(
            f"Usage:\n"
            f"  {sys.argv[0]} <branch_name>\n"
            f"  {sys.argv[0]} <pipeline_name> <build_number>",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
