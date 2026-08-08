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
import signal
import subprocess
import sys
import time
import warnings

import psutil
import pytest
from integration.defs import process_cleanup
from integration.defs.process_cleanup import (
    PROCESS_OWNER_ENV_VAR,
    cleanup_owned_processes,
    new_process_owner_token,
)

pytestmark = pytest.mark.cpu_only


def test_cleanup_owned_processes_kills_orphan_in_new_session(tmp_path):
    owner_token = new_process_owner_token()
    env = os.environ.copy()
    env[PROCESS_OWNER_ENV_VAR] = owner_token
    ready_path = tmp_path / "worker-ready"
    worker_script = (
        "import pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(ready_path)!r}).write_text('ready'); "
        "time.sleep(60)"
    )
    launcher_script = (
        "import subprocess, sys; "
        f"process = subprocess.Popen([sys.executable, '-c', {worker_script!r}], "
        "start_new_session=True); print(process.pid, flush=True)"
    )
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            launcher_script,
        ],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    worker_pid = int(launcher.stdout.readline())

    try:
        assert launcher.wait(timeout=5.0) == 0
        deadline = time.monotonic() + 5.0
        while not ready_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready_path.exists()

        with pytest.warns(UserWarning, match="Found leftover processes"):
            remaining = cleanup_owned_processes(
                owner_token,
                description="cleanup unit test",
                terminate_grace_seconds=0.1,
                kill_wait_seconds=1.0,
            )

        assert remaining == []
        try:
            worker_status = psutil.Process(worker_pid).status()
        except psutil.NoSuchProcess:
            worker_status = None
        assert worker_status in (None, psutil.STATUS_ZOMBIE)
    finally:
        try:
            os.kill(worker_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_cleanup_owned_processes_does_not_kill_another_owner():
    owner_token = new_process_owner_token()
    other_token = new_process_owner_token()
    env = os.environ.copy()
    env[PROCESS_OWNER_ENV_VAR] = other_token
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env=env,
    )

    try:
        assert cleanup_owned_processes(owner_token, description="cleanup unit test") == []
        assert psutil.Process(process.pid).is_running()
    finally:
        process.terminate()
        process.wait(timeout=5.0)


def test_cleanup_waits_for_killed_gpu_context(monkeypatch):
    gpu_pid_samples = iter([{1234}, {1234}, set()])
    monkeypatch.setattr(process_cleanup, "_gpu_process_pids", lambda: next(gpu_pid_samples))
    monkeypatch.setattr(process_cleanup, "_is_recycled_pid", lambda *_: False)

    assert process_cleanup._wait_until_gpu_contexts_released({1234: 1.0}, 5.0) == []


def test_gpu_context_of_recycled_pid_is_ignored():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        create_time = psutil.Process(process.pid).create_time()
        assert not process_cleanup._is_recycled_pid(process.pid, create_time)
        # Same PID, different start time: an unrelated process took the PID over.
        assert process_cleanup._is_recycled_pid(process.pid, create_time - 60.0)
    finally:
        process.terminate()
        process.wait(timeout=5.0)


def test_gpu_context_outliving_its_process_is_reported():
    process = subprocess.Popen([sys.executable, "-c", ""])
    process.wait(timeout=10.0)

    # Nothing alive holds the PID, so a GPU context under it is our leftover, not PID reuse.
    assert not process_cleanup._is_recycled_pid(process.pid, 1.0)


def test_gpu_verification_is_skipped_when_nvml_pids_are_not_ours(monkeypatch):
    monkeypatch.setattr(process_cleanup, "_gpu_process_pids", lambda: {999999})

    with pytest.warns(UserWarning, match="different namespace"):
        assert (
            process_cleanup._verify_gpu_contexts_released(
                {4321: 1.0},
                {999999},
                1.0,
                description="cleanup unit test",
            )
            == []
        )


def test_gpu_verification_is_conclusive_without_compute_apps(monkeypatch):
    def fail_if_called():
        pytest.fail("must not poll NVML when no compute process held a context")

    monkeypatch.setattr(process_cleanup, "_gpu_process_pids", fail_if_called)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            process_cleanup._verify_gpu_contexts_released(
                {4321: 1.0},
                set(),
                1.0,
                description="cleanup unit test",
            )
            == []
        )


def test_gpu_query_failure_is_reported(monkeypatch):
    result = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=1, stdout="", stderr="driver unavailable"
    )
    monkeypatch.setattr(process_cleanup.subprocess, "run", lambda *_, **__: result)

    with pytest.warns(UserWarning, match="driver unavailable"):
        assert process_cleanup._gpu_process_pids() is None


def test_signal_permission_failure_is_reported(monkeypatch):
    class InaccessibleProcess:
        pid = 1234

        def send_signal(self, _sig):
            raise psutil.AccessDenied(self.pid)

    monkeypatch.setattr(process_cleanup, "_is_live", lambda _process: True)

    with pytest.warns(UserWarning, match="Unable to send SIGKILL to process 1234"):
        process_cleanup._signal_processes([InaccessibleProcess()], signal.SIGKILL)
