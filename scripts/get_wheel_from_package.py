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
        # The branch walk is kept under a pin. The ref is a commit SHA, so the
        # same ref under another branch's promote directory is the bundle built
        # from that same commit -- whichever candidate resolves it, the profiles
        # are the pinned ones. Narrowing the list would instead risk looking in
        # a directory the pin was never resolved against.
        env["BOLT_PROFILE_REF"] = bolt_profile_ref
        print(f"Pinned to BOLT bundle {bolt_profile_ref}")

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


def get_wheel_from_package(arch,
                           artifact_path,
                           timeout,
                           bolt_branch=None,
                           bolt_profile_ref=None):
    if arch == "x86_64":
        tarfile_name = "TensorRT-LLM.tar.gz"
    else:
        tarfile_name = "TensorRT-LLM-GH200.tar.gz"

    tarfile_link = f"https://urm.nvidia.com/artifactory/{artifact_path}/{tarfile_name}"
    for attempt in range(timeout):
        try:
            # -O pins the output name: without it wget falls back to
            # <name>.1 when a previous attempt left a partial file behind,
            # and the extract below would then read stale bytes.
            subprocess.run(["wget", "-nv", "-O", tarfile_name, tarfile_link],
                           check=True)
            print(f"Tarfile is available at {tarfile_link}")
            break
        except Exception:
            if attempt == timeout - 1:
                raise TimeoutError(
                    f"Failed to download file after {timeout} attempts: {tarfile_link}"
                )
            print(
                f"Tarfile not ready yet, waiting 60 seconds... (attempt {attempt + 1}/{timeout})"
            )
            time.sleep(60)

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
