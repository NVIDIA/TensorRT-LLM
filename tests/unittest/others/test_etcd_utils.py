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
"""Unit tests for the shared test_common.etcd_utils launcher."""

import os
import shutil
from pathlib import Path

import pytest
from test_common import etcd_utils
from test_common.etcd_utils import running_etcd

pytestmark = pytest.mark.cpu_only


class _FakePopen:
    def __init__(self, cmd: list[str], *, returncode: int | None = None) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        if self.returncode is None:
            self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_running_etcd_configures_isolated_listeners_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ports = iter([31001, 31002])
    launched_procs: list[_FakePopen] = []
    ready_calls: list[tuple[str, dict[str, object]]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        os.makedirs(_flag_value(cmd, "--data-dir"), exist_ok=True)
        proc = _FakePopen(cmd)
        launched_procs.append(proc)
        return proc

    def fake_wait(url: str, **kwargs: object) -> None:
        ready_calls.append((url, kwargs))

    monkeypatch.setattr(etcd_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(etcd_utils, "wait_for_endpoint_ready", fake_wait)

    with running_etcd(
        data_dir_root=str(tmp_path),
        port_allocator=lambda: next(ports),
        extra_args=["--log-level", "debug"],
    ) as etcd:
        assert etcd.client_port == 31001
        assert etcd.peer_port == 31002
        assert etcd.client_url == "http://127.0.0.1:31001"
        assert etcd.uri == "etcd://127.0.0.1:31001"
        assert os.path.isdir(etcd.data_dir)
        assert Path(etcd.data_dir).parent == tmp_path

        cmd = launched_procs[0].cmd
        assert _flag_value(cmd, "--listen-client-urls") == "http://127.0.0.1:31001"
        assert _flag_value(cmd, "--advertise-client-urls") == "http://127.0.0.1:31001"
        assert _flag_value(cmd, "--listen-peer-urls") == "http://127.0.0.1:31002"
        assert _flag_value(cmd, "--initial-advertise-peer-urls") == "http://127.0.0.1:31002"
        assert _flag_value(cmd, "--initial-cluster") == "default=http://127.0.0.1:31002"
        assert cmd[-2:] == ["--log-level", "debug"]

        assert len(ready_calls) == 1
        health_url, ready_kwargs = ready_calls[0]
        assert health_url == "http://127.0.0.1:31001/health"
        assert ready_kwargs["server_proc"] is launched_procs[0]
        assert ready_kwargs["trust_env"] is False
        assert ready_kwargs["allow_redirects"] is False

    assert launched_procs[0].killed
    assert launched_procs[0].waited
    assert not os.path.exists(etcd.data_dir)


def test_running_etcd_retries_with_fresh_ports_and_data_dir_on_early_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ports = iter([32001, 32002, 32003, 32004])
    launched_procs: list[_FakePopen] = []
    data_dirs: list[str] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        data_dir = _flag_value(cmd, "--data-dir")
        os.makedirs(data_dir, exist_ok=True)
        data_dirs.append(data_dir)
        returncode = 1 if len(launched_procs) == 0 else None
        proc = _FakePopen(cmd, returncode=returncode)
        launched_procs.append(proc)
        return proc

    def fake_wait(url: str, **kwargs: object) -> None:
        proc = kwargs["server_proc"]
        assert isinstance(proc, _FakePopen)
        if proc.poll() is not None:
            raise RuntimeError(f"exited early with code {proc.returncode}")

    monkeypatch.setattr(etcd_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(etcd_utils, "wait_for_endpoint_ready", fake_wait)

    with running_etcd(
        data_dir_root=str(tmp_path),
        port_allocator=lambda: next(ports),
        attempts=3,
    ) as etcd:
        assert len(launched_procs) == 2
        assert launched_procs[0].waited
        assert not launched_procs[0].killed
        assert not os.path.exists(data_dirs[0])
        assert data_dirs[0] != data_dirs[1]
        assert etcd.client_port == 32003
        assert etcd.peer_port == 32004

    assert launched_procs[1].killed
    assert launched_procs[1].waited
    assert not os.path.exists(data_dirs[1])


def test_running_etcd_raises_after_exhausting_early_exit_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ports = iter([33001, 33002, 33003, 33004, 33005, 33006])
    data_dirs: list[str] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        data_dir = _flag_value(cmd, "--data-dir")
        os.makedirs(data_dir, exist_ok=True)
        data_dirs.append(data_dir)
        return _FakePopen(cmd, returncode=1)

    def fake_wait(url: str, **kwargs: object) -> None:
        raise RuntimeError("etcd exited early")

    monkeypatch.setattr(etcd_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(etcd_utils, "wait_for_endpoint_ready", fake_wait)

    with pytest.raises(RuntimeError, match="etcd exited early"):
        with running_etcd(
            data_dir_root=str(tmp_path),
            port_allocator=lambda: next(ports),
            attempts=3,
        ):
            pytest.fail("running_etcd should not yield when all attempts exit early")

    assert len(data_dirs) == 3
    assert all(not os.path.exists(path) for path in data_dirs)


def test_running_etcd_does_not_retry_on_readiness_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ports = iter([34001, 34002, 34003, 34004])
    launched_procs: list[_FakePopen] = []
    data_dirs: list[str] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        data_dir = _flag_value(cmd, "--data-dir")
        os.makedirs(data_dir, exist_ok=True)
        data_dirs.append(data_dir)
        proc = _FakePopen(cmd, returncode=None)
        launched_procs.append(proc)
        return proc

    def fake_wait(url: str, **kwargs: object) -> None:
        raise RuntimeError("timed out waiting for readiness")

    monkeypatch.setattr(etcd_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(etcd_utils, "wait_for_endpoint_ready", fake_wait)

    with pytest.raises(RuntimeError, match="timed out waiting for readiness"):
        with running_etcd(
            data_dir_root=str(tmp_path),
            port_allocator=lambda: next(ports),
            attempts=3,
        ):
            pytest.fail("running_etcd should not yield after a readiness timeout")

    assert len(launched_procs) == 1
    assert launched_procs[0].killed
    assert launched_procs[0].waited
    assert not os.path.exists(data_dirs[0])


def test_running_etcd_fixed_client_port_uses_dynamic_peer_and_loopback_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First peer port collides with the fixed client port and must be skipped;
    # the next peer port fails early and triggers a retry on a fresh peer port.
    peer_candidates = iter([2379, 35001, 35002])
    launched_procs: list[_FakePopen] = []
    probed_urls: list[str] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        returncode = 1 if len(launched_procs) == 0 else None
        proc = _FakePopen(cmd, returncode=returncode)
        launched_procs.append(proc)
        return proc

    def fake_wait(url: str, **kwargs: object) -> None:
        probed_urls.append(url)
        proc = kwargs["server_proc"]
        assert isinstance(proc, _FakePopen)
        if proc.poll() is not None:
            raise RuntimeError("peer port race")

    monkeypatch.setattr(etcd_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(etcd_utils, "wait_for_endpoint_ready", fake_wait)

    with running_etcd(
        client_port=2379,
        listen_client_host="0.0.0.0",
        advertise_client_host="node01",
        port_allocator=lambda: next(peer_candidates),
        attempts=3,
    ) as etcd:
        assert len(launched_procs) == 2
        assert probed_urls == ["http://127.0.0.1:2379/health", "http://127.0.0.1:2379/health"]
        assert etcd.client_port == 2379
        assert etcd.peer_port == 35002
        assert etcd.uri == "etcd://node01:2379"
        cmd = launched_procs[1].cmd
        assert _flag_value(cmd, "--listen-client-urls") == "http://0.0.0.0:2379"
        assert _flag_value(cmd, "--advertise-client-urls") == "http://node01:2379"
        assert _flag_value(cmd, "--listen-peer-urls") == "http://127.0.0.1:35002"
        assert _flag_value(cmd, "--initial-cluster") == "default=http://127.0.0.1:35002"

    assert not os.path.exists(etcd.data_dir)


def test_running_etcd_validates_arguments() -> None:
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        with running_etcd(attempts=0):
            pass

    with pytest.raises(ValueError, match="must be distinct"):
        with running_etcd(client_port=2379, peer_port=2379):
            pass


def test_running_etcd_supports_concurrent_real_instances() -> None:
    if shutil.which("etcd") is None:
        pytest.skip("etcd binary is not installed in this environment")

    with running_etcd() as first, running_etcd() as second:
        assert first.client_port != second.client_port
        assert first.peer_port != second.peer_port
        assert first.process.poll() is None
        assert second.process.poll() is None
        first_dir = first.data_dir
        second_dir = second.data_dir

    assert not os.path.exists(first_dir)
    assert not os.path.exists(second_dir)
