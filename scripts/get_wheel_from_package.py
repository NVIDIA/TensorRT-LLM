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
    parser.add_argument("--bolted",
                        "-b",
                        action="store_true",
                        help="Require the BOLT-optimized build "
                        "(bolted-<tarfile>) that BoltProfileGen publishes "
                        "alongside the plain one. The canonical tarfile "
                        "appears as soon as the build stage finishes, long "
                        "before BOLT has run, so an image that must ship "
                        "optimized binaries has to wait on this distinct "
                        "name instead. Never falls back.")


def get_wheel_from_package(arch, artifact_path, timeout, bolted=False):
    if arch == "x86_64":
        tarfile_name = "TensorRT-LLM.tar.gz"
    else:
        tarfile_name = "TensorRT-LLM-GH200.tar.gz"

    if bolted:
        tarfile_name = f"bolted-{tarfile_name}"

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


if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    get_wheel_from_package(**vars(args))
