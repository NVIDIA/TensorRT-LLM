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
"""GPU tests for the orphaned-GPU-process sweep in trt_test_alternative.

A worker left behind by a test killed with ``os._exit()`` (pytest-timeout with
``--timeout-method=thread``) lives in its own session and is re-parented to init.
These tests start real CUDA processes that way and check, against the real NVML and
/proc, that the harness kills exactly the ones that belong to a failed launch.

Requirements: one GPU with torch CUDA support. NVML must report pids in this PID
namespace, which is also what the sweep itself needs to work.
"""

import os
import subprocess
import sys
import time

import pytest
import torch

from defs import trt_test_alternative as tta

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
    pytest.mark.skipif(sys.platform != "linux", reason="Linux-only process cleanup"),
]

_WAIT_S = 180
# Bounds a whole launcher run: the worker's CUDA start-up plus the interpreter and I/O.
_RUN_TIMEOUT_S = _WAIT_S + 120

# Creates a CUDA context, reports its pid through ``argv[1]`` and idles.
_CUDA_WORKER = """
import os, sys, time, torch
torch.zeros(1, device="cuda")
with open(sys.argv[1], "w") as f:
    f.write(str(os.getpid()))
time.sleep(600)
"""

# Starts the CUDA worker in its own session, waits until it holds the GPU, then
# exits with ``argv[2]``, leaving the worker orphaned like an MPI worker.
_LAUNCHER = """
import os, subprocess, sys, time
ready, code = sys.argv[1], int(sys.argv[2])
subprocess.Popen(
    [sys.executable, sys.argv[3], ready],
    start_new_session=True,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
deadline = time.time() + float(sys.argv[4])
while not os.path.exists(ready) and time.time() < deadline:
    time.sleep(0.2)
sys.exit(code)
"""


@pytest.fixture
def scripts(tmp_path):
    worker = tmp_path / "cuda_worker.py"
    worker.write_text(_CUDA_WORKER)
    launcher = tmp_path / "launcher.py"
    launcher.write_text(_LAUNCHER)
    return worker, launcher


@pytest.fixture
def cleanup_pids():
    """Kill workers a test failed to clean up, so they never leak to later tests."""
    pids = []
    yield pids
    for pid in pids:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def _launcher_cmd(scripts, ready, code):
    worker, launcher = scripts
    return [sys.executable, str(launcher), str(ready), str(code), str(worker), str(_WAIT_S)]


def _read_pid(ready) -> int:
    assert ready.exists(), "the CUDA worker never created its context"
    return int(ready.read_text())


def _is_alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def _wait_gone(pid: int) -> bool:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not _is_alive(pid):
            return True
        time.sleep(0.2)
    return False


def test_sweep_kills_orphaned_cuda_process(scripts, tmp_path, cleanup_pids):
    ready = tmp_path / "ready"
    kwargs, token = tta.tag_launch({})
    subprocess.run(_launcher_cmd(scripts, ready, 0), check=True, timeout=_RUN_TIMEOUT_S, **kwargs)
    pid = _read_pid(ready)
    cleanup_pids.append(pid)

    assert pid in tta.list_gpu_compute_pids(), "NVML does not report the worker's pid"
    assert tta._holds_gpu(pid)

    with pytest.warns(UserWarning, match="leftover GPU processes"):
        tta.kill_orphaned_gpu_processes(token)

    assert _wait_gone(pid)
    assert pid not in tta.list_gpu_compute_pids()


def test_failed_launch_kills_orphaned_cuda_process(scripts, tmp_path, cleanup_pids):
    ready = tmp_path / "ready"

    with pytest.warns(UserWarning, match="leftover GPU processes"):
        with pytest.raises(subprocess.CalledProcessError):
            tta.check_call(_launcher_cmd(scripts, ready, 3))

    pid = _read_pid(ready)
    cleanup_pids.append(pid)
    assert _wait_gone(pid)


def test_successful_launch_keeps_cuda_process(scripts, tmp_path, cleanup_pids):
    ready = tmp_path / "ready"

    tta.check_call(_launcher_cmd(scripts, ready, 0))

    pid = _read_pid(ready)
    cleanup_pids.append(pid)
    assert _is_alive(pid)


def test_sweep_keeps_cuda_process_of_another_launch(scripts, tmp_path, cleanup_pids):
    ready = tmp_path / "ready"
    other_kwargs, _ = tta.tag_launch({})
    subprocess.run(
        _launcher_cmd(scripts, ready, 0), check=True, timeout=_RUN_TIMEOUT_S, **other_kwargs
    )
    pid = _read_pid(ready)
    cleanup_pids.append(pid)
    assert pid in tta.list_gpu_compute_pids()

    _, token = tta.tag_launch({})
    tta.kill_orphaned_gpu_processes(token)

    assert _is_alive(pid)
