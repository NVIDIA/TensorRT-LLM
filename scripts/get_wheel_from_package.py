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

import glob
import os
import shutil
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path


def get_project_dir():
    return Path(__file__).parent.resolve().parent


def add_arguments(parser: ArgumentParser):
    parser.add_argument("--arch",
                        "-a",
                        required=True,
                        help="Architecture of the built package")
    parser.add_argument("--artifact_path",
                        "-u",
                        required=True,
                        help="the path of the built package")
    parser.add_argument("--timeout",
                        "-t",
                        type=int,
                        default=60,
                        help="Timeout in minutes")
    parser.add_argument("--bolt-branch",
                        "-b",
                        default=None,
                        help="Comma-separated branches whose promoted BOLT "
                        "profile bundle should be applied to the extracted "
                        "wheel, tried in order. The image installs this wheel, "
                        "so optimizing it here is what makes the released "
                        "container carry optimized binaries. Omit to install "
                        "the wheel as built. Fatal if set and no branch has a "
                        "usable bundle.")
    parser.add_argument("--bolt-profile-ref",
                        default=None,
                        help="Pin to one immutable profile bundle by ref, so "
                        "the image's wheel is optimized with the same profiles "
                        "as the tested build and the released wheel. Collapses "
                        "--bolt-branch to its first entry, since a ref names "
                        "one bundle under one branch. Omit to take whatever is "
                        "currently promoted as latest.")


def bolt_optimize_wheels(build_dir, arch, bolt_branch, bolt_profile_ref=None):
    """Apply the latest promoted BOLT bundle to each wheel in `build_dir`.

    Deliberately uses the branch's last promoted bundle rather than one
    generated from this commit: the optimized tarball for THIS run is not
    published until BoltProfileGen finishes, hours after the image build starts,
    and waiting on it would serialize the release behind a multi-hour GPU job.
    Profiles are function-name-keyed and applied with -infer-stale-profile, so a
    bundle from a nearby commit costs some optimization quality, never
    correctness -- the same trade the pre-merge consume path and the image
    profile overlay already make.
    """
    bolt_internal = get_project_dir() / "scripts" / "bolt" / "internal"
    apply_latest = str(bolt_internal / "apply_latest.sh")
    triple = "x86_64-linux-gnu" if arch == "x86_64" else "aarch64-linux-gnu"
    branches = [b.strip() for b in bolt_branch.split(",") if b.strip()]
    env = os.environ.copy()
    if bolt_profile_ref:
        # The caller already narrowed --bolt-branch to the branch the pin was
        # resolved against, so this list is normally a single entry. Setting the
        # ref makes apply_latest.sh fetch that exact bundle.
        env["BOLT_PROFILE_REF"] = bolt_profile_ref
        print(f"Pinned to BOLT bundle {bolt_profile_ref} on {branches}")

    for wheel in sorted(Path(build_dir).glob("tensorrt_llm*.whl")):
        bolted = wheel.with_suffix(".whl.bolted")
        for branch in branches:
            print(f"Applying BOLT profiles from {branch}/{triple} to "
                  f"{wheel.name}")
            cmd = [
                "bash", apply_latest, branch, triple,
                str(wheel),
                str(bolted)
            ]
            # 3 = that branch has nothing promoted; anything else is decisive.
            rc = subprocess.run(cmd, env=env).returncode
            if rc == 0:
                os.replace(bolted, wheel)
                print(f"BOLT optimized {wheel.name} ({branch}/{triple})")
                break
            if rc != 3:
                raise RuntimeError(
                    f"BOLT apply failed for {wheel.name} (rc={rc})")
            print(f"No promoted bundle for {branch}/{triple}; "
                  "trying next branch")
        else:
            if bolt_profile_ref:
                raise RuntimeError(
                    f"Pinned BOLT bundle {bolt_profile_ref} ({triple}) not "
                    f"found under any of {branches}; refusing to build an "
                    f"image whose wheel is optimized with anything else")
            raise RuntimeError(
                f"No promoted BOLT bundle for any of {branches} ({triple}); "
                f"refusing to build an unoptimized release image")


# Exit codes of jenkins/scripts/fetch_build_artifact.sh that download_package
# acts on, by the names the helper gives them; any other non-zero code stops
# it at once. tests/unittest/scripts/test_fetch_build_artifact.py checks that
# they match.
_FETCH_ABSENT = 2
_FETCH_REJECTED = 3
_FETCH_TRANSFER_FAILED = 4


def _rejected_message(package_path: str) -> str:
    """Describe why the primary location refused to serve `package_path`."""
    if (os.environ.get("LLM_ARTIFACTS_PRIMARY") or "aws") == "urm":
        return ("The primary location (urm) rejected the anonymous request "
                f"for {package_path}.")
    credential = (os.environ.get("ARTIFACTORY_CREDENTIAL_ID")
                  or "<unset: ARTIFACTORY_CREDENTIAL_ID>")
    return (
        f"Artifactory rejected the credentials for {package_path}: the "
        f"Artifactory access token in credential '{credential}' needs "
        "renewal or lacks read permission on sw-tensorrt-llm-generic-local; "
        "an Artifactory access token for artifactory.nvidia.com is required.")


def download_package(artifact_path: str, tarfile_name: str,
                     timeout: int) -> None:
    """Download a packaged build, polling until it has been published.

    jenkins/scripts/fetch_build_artifact.sh reads the primary location
    (LLM_ARTIFACTS_PRIMARY) first, falls back to the other location only when
    the file is absent there, and logs which location served it. It writes to
    a temporary file and renames it only after the SHA-1 matches, so a failed
    attempt never leaves a partial `tarfile_name` for the extract to read.

    Args:
        artifact_path: The build's artifact directory in the legacy layout,
            a repository name followed by llm-artifacts/<job>/<build>. The
            helper maps it to each location.
        tarfile_name: File name of the packaged build, also used for the local
            copy.
        timeout: Maximum number of attempts, one minute apart.

    Raises:
        PermissionError: The primary location rejected the request.
        RuntimeError: The helper failed for a reason other than the package
            being absent or a transfer failure, such as a usage error.
        TimeoutError: The package was not downloaded within `timeout`
            attempts.
    """
    helper = (get_project_dir() / "jenkins" / "scripts" /
              "fetch_build_artifact.sh")
    package_path = f"{artifact_path}/{tarfile_name}"
    for attempt in range(1, timeout + 1):
        # The environment carries ARTIFACTORY_NETRC, LLM_ARTIFACTS_PRIMARY and
        # ARTIFACTORY_CREDENTIAL_ID to the helper.
        rc = subprocess.run(
            ["bash", str(helper), package_path, tarfile_name],
            env=os.environ.copy()).returncode
        if rc == 0:
            return
        if rc == _FETCH_REJECTED:
            raise PermissionError(_rejected_message(package_path))
        if rc not in (_FETCH_ABSENT, _FETCH_TRANSFER_FAILED):
            raise RuntimeError(
                f"fetch_build_artifact.sh failed with exit code {rc} for "
                f"{package_path}")
        if attempt < timeout:
            reason = ("not published yet"
                      if rc == _FETCH_ABSENT else "transfer failed")
            print(
                f"Tarfile {reason}, waiting 60 seconds... "
                f"(attempt {attempt}/{timeout})",
                flush=True)
            time.sleep(60)
    raise TimeoutError(
        f"Failed to download file after {timeout} attempts: {package_path}")


def get_wheel_from_package(arch,
                           artifact_path,
                           timeout,
                           bolt_branch=None,
                           bolt_profile_ref=None):
    if arch == "x86_64":
        tarfile_name = "TensorRT-LLM.tar.gz"
    else:
        tarfile_name = "TensorRT-LLM-GH200.tar.gz"

    download_package(artifact_path, tarfile_name, timeout)

    llm_root = get_project_dir()
    tmp_dir = llm_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["tar", "-zxf", tarfile_name, "-C",
                    str(tmp_dir)],
                   check=True)

    tmp_dir = tmp_dir / "TensorRT-LLM"

    build_dir = llm_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    wheel_files = glob.glob(str(tmp_dir / "tensorrt_llm*.whl"))
    for wheel_file in wheel_files:
        shutil.move(wheel_file, str(build_dir))
        print(f"Moved wheel file: {wheel_file} -> {build_dir}")

    shutil.rmtree(tmp_dir)

    if os.path.exists(tarfile_name):
        os.remove(tarfile_name)

    # After the move, before the Dockerfile's release stage pip installs
    # whatever is in build/.
    if bolt_branch:
        bolt_optimize_wheels(build_dir, arch, bolt_branch, bolt_profile_ref)


if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    get_wheel_from_package(**vars(args))
