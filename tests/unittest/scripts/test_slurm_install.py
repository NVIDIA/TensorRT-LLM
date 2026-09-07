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

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BASH_UTILS = REPO_ROOT / "jenkins" / "scripts" / "bash_utils.sh"
SLURM_INSTALL = REPO_ROOT / "jenkins" / "scripts" / "slurm_install.sh"
L0_TEST = REPO_ROOT / "jenkins" / "L0_Test.groovy"


@pytest.mark.parametrize("with_timeout", (False, True))
def test_retry_command_returns_terminal_status(with_timeout: bool) -> None:
    env = os.environ.copy()
    env["BASH_UTILS"] = str(BASH_UTILS)
    env["WITH_TIMEOUT"] = "1" if with_timeout else "0"

    command = r"""
set -e
source "$BASH_UTILS"
attempt=0

sleep() {
    :
}

timeout() {
    [[ "${1-}" == "1800" ]] || exit 90
    shift
    "$@"
}

fail() {
    attempt=$((attempt + 1))
    return 23
}

retry_args=(3 0)
if [[ "$WITH_TIMEOUT" == "1" ]]; then
    retry_args+=(--timeout 1800)
fi

if retry_command "${retry_args[@]}" fail; then
    exit 99
else
    rc=$?
fi
[[ "$rc" -eq 23 ]]
[[ "$attempt" -eq 3 ]]
"""
    result = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True)

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize(
    "missing", ("tarName", "llmTarfile", "resourcePathNode", "stageName", "pytestCommand")
)
def test_slurm_install_requires_artifact_inputs(missing: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "tarName": "/tmp/TensorRT-LLM.tar.gz",
            "llmTarfile": "https://example.invalid/TensorRT-LLM.tar.gz",
            "resourcePathNode": "/tmp",
            "stageName": "test-stage",
            "pytestCommand": "pytest",
        }
    )
    env.pop(missing)

    result = subprocess.run(
        ["bash", str(SLURM_INSTALL)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert f"{missing} is required" in result.stderr


def test_l0_artifact_download_retries_overwrite_archive() -> None:
    download_commands = [
        line.strip()
        for line in L0_TEST.read_text().splitlines()
        if "wget -nv" in line and "llmTarfile" in line
    ]

    assert len(download_commands) == 4
    assert all("wget -nv -O" in command for command in download_commands)
