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

RUNTIME_SETUP = Path(__file__).resolve().parents[3] / "jenkins/scripts/slurm_env_setup.sh"
NCCL_KEYS = ("NCCL_IB_DISABLE", "NCCL_NET", "NCCL_PROFILER_PLUGIN")
GCP_SETTINGS = ("0", "gIB", "none")


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        pytest.param({"NCCL_NET_PLUGIN": "gcp"}, GCP_SETTINGS, id="net-plugin"),
        pytest.param({"NCCL_ENV_PLUGIN": "gcp"}, GCP_SETTINGS, id="env-plugin"),
        pytest.param(
            {
                "NCCL_NET_PLUGIN": "gcp",
                "NCCL_ENV_PLUGIN": "gcp",
                "NCCL_IB_DISABLE": "1",
                "NCCL_NET": "Socket",
                "NCCL_PROFILER_PLUGIN": "gcp",
            },
            GCP_SETTINGS,
            id="override-inherited-settings",
        ),
        pytest.param(
            {
                "NCCL_NET_PLUGIN": "ofi",
                "NCCL_IB_DISABLE": "0",
                "NCCL_NET": "AWS Libfabric",
                "NCCL_PROFILER_PLUGIN": "custom",
            },
            ("0", "AWS Libfabric", "custom"),
            id="preserve-non-gcp-settings",
        ),
        pytest.param({}, (None, None, None), id="unset"),
        pytest.param(
            {"NCCL_NET_PLUGIN": "", "NCCL_ENV_PLUGIN": ""}, (None, None, None), id="empty"
        ),
        pytest.param({"NCCL_NET_PLUGIN": "GCP"}, (None, None, None), id="exact-plugin-name"),
    ],
)
def test_slurm_nccl_environment(variables: dict[str, str], expected: tuple) -> None:
    command = r"""
set -euo pipefail
pip3() { printf 'Location: /test/site-packages\n'; }
source "$1"
slurm_setup_runtime_env
/usr/bin/env
"""
    env = {"PATH": os.environ["PATH"], "LD_LIBRARY_PATH": "/test/lib", **variables}
    result = subprocess.run(
        ["bash", "-c", command, "test-slurm-env", str(RUNTIME_SETUP)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    actual = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert tuple(actual.get(key) for key in NCCL_KEYS) == expected
    for key in ("NCCL_NET_PLUGIN", "NCCL_ENV_PLUGIN"):
        if key in variables:
            assert actual.get(key) == variables[key]


def test_gcp_plugin_selected_by_shell_initialization(tmp_path: Path) -> None:
    init = tmp_path / "image-init.sh"
    init.write_text("export NCCL_ENV_PLUGIN=gcp\n")
    test_slurm_nccl_environment({"BASH_ENV": str(init)}, GCP_SETTINGS)
