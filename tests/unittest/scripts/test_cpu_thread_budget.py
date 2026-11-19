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


def _cpu_worker_threads(cgroup_root: Path | None = None) -> int:
    if cgroup_root is None:
        cgroup_root = Path("/sys/fs/cgroup")
    affinity_threads = len(os.sched_getaffinity(0))
    assert affinity_threads > 0, "CPU-only CI must expose a CPU affinity budget"
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
            return affinity_threads

    if quota in ("max", "-1"):
        return affinity_threads
    expected_threads, remainder = divmod(int(quota), int(period))
    assert expected_threads > 0 and remainder == 0, "CPU-only CI requires a whole-CPU budget"
    return expected_threads


def test_cpu_ci_subprocess_inherits_thread_budget() -> None:
    if not os.environ.get("stageName", "").startswith("CPU-"):
        pytest.skip("Requires the CPU-only CI launcher and its worker cgroup")

    expected_threads = _cpu_worker_threads()
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
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "omp": str(expected_threads),
        "mkl": str(expected_threads),
        "torch": expected_threads,
    }


@pytest.mark.parametrize(
    ("quota_files", "expected_threads"),
    [
        ({"cpu.max": "300000 100000"}, 3),
        ({"cpu.max": "max 100000"}, 5),
        ({"cpu/cpu.cfs_quota_us": "300000", "cpu/cpu.cfs_period_us": "100000"}, 3),
        ({"cpu/cpu.cfs_quota_us": "-1", "cpu/cpu.cfs_period_us": "100000"}, 5),
        ({}, 5),
    ],
    ids=["bounded-v2", "unlimited-v2", "bounded-v1", "unlimited-v1", "affinity-only"],
)
def test_cpu_worker_threads_use_enforced_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quota_files: dict[str, str],
    expected_threads: int,
) -> None:
    for relative_path, value in quota_files.items():
        quota_file = tmp_path / relative_path
        quota_file.parent.mkdir(parents=True, exist_ok=True)
        quota_file.write_text(value)
    monkeypatch.setattr(os, "sched_getaffinity", lambda process_id: set(range(5)))

    assert _cpu_worker_threads(tmp_path) == expected_threads


@pytest.mark.parametrize("quota", ["0", "150000"], ids=["zero", "fractional"])
def test_cpu_worker_threads_reject_invalid_quotas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quota: str
) -> None:
    (tmp_path / "cpu.max").write_text(f"{quota} 100000")
    monkeypatch.setattr(os, "sched_getaffinity", lambda process_id: set(range(5)))

    with pytest.raises(AssertionError, match="whole-CPU budget"):
        _cpu_worker_threads(tmp_path)


def test_cpu_worker_threads_require_available_cpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda process_id: set())

    with pytest.raises(AssertionError, match="CPU affinity budget"):
        _cpu_worker_threads(tmp_path)


def test_cpu_ci_subprocess_reports_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_probe(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        result = subprocess.CompletedProcess(
            command, 1, stdout="", stderr="CPU budget probe import failed"
        )
        if options.get("check"):
            result.check_returncode()
        return result

    (tmp_path / "cpu.max").write_text("300000 100000")
    monkeypatch.setattr(f"{__name__}.Path", lambda cgroup_path: tmp_path)
    monkeypatch.setattr(os, "sched_getaffinity", lambda process_id: set(range(3)))
    monkeypatch.setenv("stageName", "CPU-probe")
    monkeypatch.setattr(subprocess, "run", fail_probe)

    with pytest.raises(AssertionError, match="CPU budget probe import failed"):
        test_cpu_ci_subprocess_inherits_thread_budget()
