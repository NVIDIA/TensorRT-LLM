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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only


def _cpu_quota_threads() -> int:
    cgroup_root = Path("/sys/fs/cgroup")
    cpu_max = cgroup_root / "cpu.max"
    if cpu_max.is_file():
        quota, period = cpu_max.read_text().split()
    else:
        for cpu_directory in (
            cgroup_root / "cpu",
            cgroup_root / "cpu,cpuacct",
            cgroup_root / "cpuacct,cpu",
            cgroup_root,
        ):
            quota_file = cpu_directory / "cpu.cfs_quota_us"
            if quota_file.is_file():
                quota = quota_file.read_text().strip()
                period = (cpu_directory / "cpu.cfs_period_us").read_text().strip()
                break
        else:
            pytest.fail("CPU-only CI must expose its worker CPU quota")

    assert quota not in ("max", "-1"), "CPU-only CI must have a bounded CPU quota"
    expected_threads, remainder = divmod(int(quota), int(period))
    assert expected_threads > 0 and remainder == 0, "CPU-only CI requires a whole-CPU budget"
    return expected_threads


def test_cpu_ci_subprocess_inherits_thread_budget() -> None:
    if not os.environ.get("stageName", "").startswith("CPU-"):
        pytest.skip("Requires the CPU-only CI launcher and its worker cgroup")

    expected_threads = _cpu_quota_threads()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import os
import torch

print(json.dumps({
    "omp": os.environ.get("OMP_NUM_THREADS"),
    "mkl": os.environ.get("MKL_NUM_THREADS"),
    "torch": torch.get_num_threads(),
}))
""",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert json.loads(result.stdout) == {
        "omp": str(expected_threads),
        "mkl": str(expected_threads),
        "torch": expected_threads,
    }
