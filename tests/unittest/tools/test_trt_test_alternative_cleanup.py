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
"""Unit tests for the orphaned-GPU-process sweep in trt_test_alternative.

A worker that outlives a pytest-timeout ``os._exit()`` lives in its own session and
is re-parented to init, so the session/children scan misses it. The sweep identifies
such workers by the launch token inherited through their environment. NVML and psutil
are mocked, so no GPU is needed.
"""

import os
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

__extra_import_path__ = ["~/tests/integration/defs"]
import trt_test_alternative as tta

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(sys.platform != "linux", reason="Linux-only process cleanup"),
]

TOKEN = "launch-token"
OTHER_UID = os.getuid() + 1


@pytest.fixture(autouse=True)
def no_ambient_launch_token(monkeypatch):
    """CI runs this file inside a harness launch, which already sets the token."""
    monkeypatch.delenv(tta.LAUNCH_TOKEN_ENV, raising=False)


class FakeProc:
    """Stand-in for ``psutil.Process`` backed by the ``FAKE_PROCS`` table."""

    def __init__(self, pid):
        if pid not in FAKE_PROCS:
            raise psutil.NoSuchProcess(pid)
        self.pid = pid
        self._info = FAKE_PROCS[pid]

    def uids(self):
        return SimpleNamespace(real=self._info.get("uid", os.getuid()))

    def environ(self):
        return self._info.get("environ", {})

    def cmdline(self):
        return self._info.get("cmdline", ["worker"])

    def parents(self):
        parents, ppid = [], self._info.get("ppid")
        while ppid in FAKE_PROCS:
            parents.append(FakeProc(ppid))
            ppid = FAKE_PROCS[ppid].get("ppid")
        return parents


FAKE_PROCS = {}


@pytest.fixture
def sweep(monkeypatch):
    """Run the sweep over ``procs``; return the pids passed to ``os.kill``."""
    killed = []

    def run(procs, gpu_pids=None, token=TOKEN, no_gpu=()):
        FAKE_PROCS.clear()
        FAKE_PROCS.update(procs)
        monkeypatch.setattr(tta, "list_gpu_compute_pids", lambda: set(gpu_pids or procs))
        monkeypatch.setattr(tta, "_holds_gpu", lambda pid: pid not in no_gpu)
        monkeypatch.setattr(tta.psutil, "Process", FakeProc)
        monkeypatch.setattr(tta.os, "kill", lambda pid, sig: killed.append(pid))
        tta.kill_orphaned_gpu_processes(token)
        return killed

    yield run
    FAKE_PROCS.clear()


def test_kills_orphan_carrying_launch_token(sweep):
    procs = {1000: {"environ": {tta.LAUNCH_TOKEN_ENV: TOKEN}, "ppid": 1}}

    with pytest.warns(UserWarning, match="leftover GPU processes"):
        assert sweep(procs) == [1000]


def test_kills_orphan_whose_ancestor_carries_token(sweep):
    procs = {
        1000: {"ppid": 900},
        900: {"environ": {tta.LAUNCH_TOKEN_ENV: f"outer:{TOKEN}"}, "ppid": 1},
    }

    with pytest.warns(UserWarning):
        assert sweep(procs, gpu_pids={1000}) == [1000]


def test_keeps_unrelated_same_user_process(sweep):
    procs = {1000: {"environ": {tta.LAUNCH_TOKEN_ENV: "someone-else"}, "ppid": 1}}

    assert sweep(procs) == []


def test_keeps_process_not_holding_gpu_device(sweep):
    """A host pid aliasing an unrelated process in a container."""
    procs = {1000: {"environ": {tta.LAUNCH_TOKEN_ENV: TOKEN}, "ppid": 1}}

    assert sweep(procs, no_gpu={1000}) == []


def test_keeps_process_without_token(sweep):
    assert sweep({1000: {"environ": {}, "ppid": 1}}) == []


def test_keeps_process_of_other_user(sweep):
    procs = {1000: {"uid": OTHER_UID, "environ": {tta.LAUNCH_TOKEN_ENV: TOKEN}}}

    assert sweep(procs) == []


def test_keeps_descendant_of_this_process(sweep):
    procs = {
        1000: {"environ": {tta.LAUNCH_TOKEN_ENV: TOKEN}, "ppid": os.getpid()},
        os.getpid(): {"ppid": 1},
    }

    assert sweep(procs, gpu_pids={1000}) == []


def test_keeps_init_and_self(sweep):
    token_env = {tta.LAUNCH_TOKEN_ENV: TOKEN}
    procs = {1: {"environ": token_env}, os.getpid(): {"environ": token_env}}

    assert sweep(procs) == []


def test_skips_process_that_vanished(sweep):
    assert sweep({}, gpu_pids={4242}) == []


def test_warning_redacts_secrets_in_command_line(sweep):
    procs = {
        1000: {
            "environ": {tta.LAUNCH_TOKEN_ENV: TOKEN},
            "cmdline": ["worker", "--api-key=hunter2"],
        }
    }

    with pytest.warns(UserWarning) as record:
        sweep(procs)

    message = str(record[0].message)
    assert "hunter2" not in message
    assert "--api-key=***" in message


def test_tag_launch_adds_unique_token_and_keeps_env():
    first, token1 = tta.tag_launch({"env": {"KEEP": "1"}, "cwd": "/"})
    _, token2 = tta.tag_launch({})

    assert first["env"]["KEEP"] == "1" and first["cwd"] == "/"
    assert first["env"][tta.LAUNCH_TOKEN_ENV] == token1
    assert token1 != token2


def test_tag_launch_keeps_empty_env_empty():
    tagged, token = tta.tag_launch({"env": {}})

    assert tagged["env"] == {tta.LAUNCH_TOKEN_ENV: token}


def test_tag_launch_defaults_to_parent_env(monkeypatch):
    monkeypatch.setenv("PARENT_ONLY", "1")

    tagged, _ = tta.tag_launch({})
    explicit_none, _ = tta.tag_launch({"env": None})

    assert tagged["env"]["PARENT_ONLY"] == "1"
    assert explicit_none["env"]["PARENT_ONLY"] == "1"


def test_tag_launch_keeps_enclosing_token():
    outer, outer_token = tta.tag_launch({})
    inner, inner_token = tta.tag_launch(outer)

    assert inner["env"][tta.LAUNCH_TOKEN_ENV].split(":") == [outer_token, inner_token]


def test_holds_gpu_is_false_for_unreadable_pid():
    assert not tta._holds_gpu(2**22 + 12345)


def test_list_gpu_compute_pids_without_nvml(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)

    assert tta.list_gpu_compute_pids() == set()


@pytest.mark.parametrize("code, sweeps", [(0, False), (3, True)])
def test_popen_sweeps_only_after_failure(monkeypatch, code, sweeps):
    tokens = []
    monkeypatch.setattr(tta, "kill_orphaned_gpu_processes", tokens.append)

    command = [sys.executable, "-c", f"raise SystemExit({code})"]
    with tta.popen(command, suppress_output_info=True) as p:
        p.wait()

    assert len(tokens) == int(sweeps)


def test_check_output_sweeps_only_after_failure(monkeypatch):
    tokens = []
    monkeypatch.setattr(tta, "kill_orphaned_gpu_processes", tokens.append)

    assert tta.check_output([sys.executable, "-c", "print('ok')"]) == "ok\n"
    assert tokens == []
    with pytest.raises(subprocess.CalledProcessError):
        tta.check_output([sys.executable, "-c", "raise SystemExit(3)"])
    assert len(tokens) == 1


def test_sweep_kills_real_orphan_started_by_tagged_launch(monkeypatch):
    """An orphan in its own session, started by a tagged child, is killed."""
    spawn_orphan = (
        "import subprocess, sys;"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'],"
        " start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,"
        " stderr=subprocess.DEVNULL);"
        "print(p.pid)"
    )
    kwargs, token = tta.tag_launch({})
    pid = int(subprocess.check_output([sys.executable, "-c", spawn_orphan], **kwargs))
    try:
        monkeypatch.setattr(tta, "list_gpu_compute_pids", lambda: {pid})
        monkeypatch.setattr(tta, "_holds_gpu", lambda _: True)

        with pytest.warns(UserWarning, match="leftover GPU processes"):
            tta.kill_orphaned_gpu_processes(token)

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.1)
        else:
            pytest.fail("orphaned process was not killed")
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
