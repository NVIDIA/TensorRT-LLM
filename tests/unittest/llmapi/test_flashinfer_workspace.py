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

import fcntl
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi import mpi_session

_FLASHINFER_WORKSPACE_ENV = "FLASHINFER_WORKSPACE_BASE"
_FLASHINFER_CUBIN_ENV = "FLASHINFER_CUBIN_DIR"
_FLASHINFER_ISOLATION_ENV = "TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS"
_FLASHINFER_MANAGED_ENV = "TRTLLM_FLASHINFER_WORKSPACE_MANAGED"
pytestmark = pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")


@pytest.fixture(autouse=True)
def _isolate_flashinfer_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        _FLASHINFER_WORKSPACE_ENV,
        _FLASHINFER_CUBIN_ENV,
        _FLASHINFER_ISOLATION_ENV,
        _FLASHINFER_MANAGED_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def _worker_flashinfer_env() -> tuple[str | None, str | None]:
    mpi_session.mpi4py.MPI.COMM_WORLD.barrier()
    return (os.environ.get(_FLASHINFER_WORKSPACE_ENV), os.environ.get(_FLASHINFER_CUBIN_ENV))


def _run_worker_bootstrap(monkeypatch: pytest.MonkeyPatch, workspace_root: Path) -> None:
    from mpi4py.futures import server

    monkeypatch.setattr(sys, "argv", ["-c", str(workspace_root)])
    monkeypatch.setattr(server, "main", lambda: None)
    exec(mpi_session._FLASHINFER_WORKER_BOOTSTRAP, {})


def test_worker_bootstrap_reuses_rank_workspace_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(_FLASHINFER_WORKSPACE_ENV, raising=False)
    monkeypatch.delenv(_FLASHINFER_CUBIN_ENV, raising=False)
    monkeypatch.delenv(_FLASHINFER_MANAGED_ENV, raising=False)

    _run_worker_bootstrap(monkeypatch, tmp_path)

    rank = mpi_session.mpi4py.MPI.COMM_WORLD.Get_rank()
    workspace = tmp_path / f"rank-{rank}"
    assert os.environ[_FLASHINFER_WORKSPACE_ENV] == str(workspace)
    assert os.environ[_FLASHINFER_MANAGED_ENV] == "1"
    assert workspace.is_dir()
    assert os.environ[_FLASHINFER_CUBIN_ENV] == str(
        Path.home() / ".cache" / "flashinfer" / "cubins"
    )

    with (workspace / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)

    monkeypatch.delenv(_FLASHINFER_WORKSPACE_ENV)
    _run_worker_bootstrap(monkeypatch, tmp_path)

    assert os.environ[_FLASHINFER_WORKSPACE_ENV] == str(workspace)


def test_worker_bootstrap_uses_unlocked_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(_FLASHINFER_WORKSPACE_ENV, raising=False)
    rank = mpi_session.mpi4py.MPI.COMM_WORLD.Get_rank()
    world_size = mpi_session.mpi4py.MPI.COMM_WORLD.Get_size()
    occupied_workspace = tmp_path / f"rank-{rank}"
    occupied_workspace.mkdir()

    with (occupied_workspace / ".lock").open("a") as occupied_lock:
        fcntl.flock(occupied_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_worker_bootstrap(monkeypatch, tmp_path)
        fcntl.flock(occupied_lock, fcntl.LOCK_UN)

    assert os.environ[_FLASHINFER_WORKSPACE_ENV] == str(tmp_path / f"rank-{rank + world_size}")


def test_worker_bootstrap_replaces_inherited_managed_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inherited_workspace = tmp_path / "parent"
    monkeypatch.setenv(_FLASHINFER_WORKSPACE_ENV, str(inherited_workspace))
    monkeypatch.setenv(_FLASHINFER_MANAGED_ENV, "1")

    _run_worker_bootstrap(monkeypatch, tmp_path)

    rank = mpi_session.mpi4py.MPI.COMM_WORLD.Get_rank()
    assert os.environ[_FLASHINFER_WORKSPACE_ENV] == str(tmp_path / f"rank-{rank}")
    assert os.environ[_FLASHINFER_MANAGED_ENV] == "1"


def test_worker_bootstrap_falls_back_when_isolation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(_FLASHINFER_WORKSPACE_ENV, raising=False)
    monkeypatch.delenv(_FLASHINFER_MANAGED_ENV, raising=False)

    def raise_read_only_error(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(Path, "mkdir", raise_read_only_error)
    _run_worker_bootstrap(monkeypatch, tmp_path)

    temporary_workspace = Path(os.environ[_FLASHINFER_WORKSPACE_ENV])
    assert temporary_workspace.name.startswith("trtllm-flashinfer-rank-")
    assert os.environ[_FLASHINFER_MANAGED_ENV] == "1"
    assert not temporary_workspace.exists()
    assert "using temporary workspace" in capfd.readouterr().err


def test_worker_bootstrap_falls_back_on_non_os_setup_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(_FLASHINFER_WORKSPACE_ENV, raising=False)

    def raise_home_error(_path: Path) -> Path:
        raise RuntimeError("home directory unavailable")

    monkeypatch.setattr(Path, "expanduser", raise_home_error)
    _run_worker_bootstrap(monkeypatch, tmp_path)

    temporary_workspace = Path(os.environ[_FLASHINFER_WORKSPACE_ENV])
    rank = mpi_session.mpi4py.MPI.COMM_WORLD.Get_rank()
    assert temporary_workspace.name.startswith(f"trtllm-flashinfer-rank-{rank}-")
    assert not temporary_workspace.exists()
    error = capfd.readouterr().err
    assert f"rank {rank}" in error
    assert "home directory unavailable" in error
    assert "using temporary workspace" in error


def test_worker_bootstrap_fails_when_no_isolated_workspace_is_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(_FLASHINFER_WORKSPACE_ENV, raising=False)

    def raise_home_error(_path: Path) -> Path:
        raise RuntimeError("home directory unavailable")

    def raise_temporary_error(*_args: object, **_kwargs: object) -> None:
        raise OSError("temporary directory unavailable")

    monkeypatch.setattr(Path, "expanduser", raise_home_error)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", raise_temporary_error)

    with pytest.raises(
        RuntimeError, match="could not create an isolated FlashInfer workspace"
    ) as error:
        _run_worker_bootstrap(monkeypatch, tmp_path)

    assert "FLASHINFER_WORKSPACE_BASE" in str(error.value)
    assert "TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS=0" in str(error.value)


def test_worker_bootstrap_warns_when_unlock_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(_FLASHINFER_WORKSPACE_ENV, raising=False)
    flock = fcntl.flock

    def fail_unlock(file: object, operation: int) -> None:
        if operation == fcntl.LOCK_UN:
            raise OSError("unlock failed")
        flock(file, operation)

    monkeypatch.setattr(fcntl, "flock", fail_unlock)
    _run_worker_bootstrap(monkeypatch, tmp_path)

    assert os.environ[_FLASHINFER_WORKSPACE_ENV].startswith(str(tmp_path))
    assert "could not unlock the FlashInfer workspace" in capfd.readouterr().err


@pytest.mark.parametrize(
    "n_workers, env_overrides, expected",
    [
        (1, {}, None),
        (1, {_FLASHINFER_ISOLATION_ENV: "1"}, None),
        (
            4,
            {},
            [
                "-c",
                mpi_session._FLASHINFER_WORKER_BOOTSTRAP,
                mpi_session._FLASHINFER_WORKSPACE_ROOT,
            ],
        ),
        (4, {_FLASHINFER_ISOLATION_ENV: "0"}, None),
        (
            4,
            {
                _FLASHINFER_WORKSPACE_ENV: "/explicit",
            },
            None,
        ),
    ],
)
def test_mpi_pool_configures_worker_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    n_workers: int,
    env_overrides: dict[str, str],
    expected: list[str] | None,
) -> None:
    for name in (
        _FLASHINFER_WORKSPACE_ENV,
        _FLASHINFER_CUBIN_ENV,
        _FLASHINFER_ISOLATION_ENV,
        _FLASHINFER_MANAGED_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    captured: dict[str, object] = {}

    class FakeMpiPoolExecutor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(mpi_session, "MPIPoolExecutor", FakeMpiPoolExecutor)
    session = SimpleNamespace(
        n_workers=n_workers,
        _env_overrides=env_overrides,
        mpi_pool=None,
    )

    mpi_session.MpiPoolSession._start_mpi_pool(session)

    assert captured["python_args"] == expected
    env = captured["env"]
    assert isinstance(env, dict)
    assert all(env[key] == value for key, value in env_overrides.items())


@pytest.mark.parametrize(
    "env_overrides, expected_workspace, expected_marker, expect_bootstrap",
    [
        ({}, None, "1", True),
        ({_FLASHINFER_WORKSPACE_ENV: "/explicit"}, "/explicit", None, False),
    ],
)
@pytest.mark.parametrize("n_workers", [1, 2])
def test_mpi_pool_distinguishes_managed_and_explicit_workspaces(
    monkeypatch: pytest.MonkeyPatch,
    n_workers: int,
    env_overrides: dict[str, str],
    expected_workspace: str | None,
    expected_marker: str | None,
    expect_bootstrap: bool,
) -> None:
    monkeypatch.setenv(_FLASHINFER_WORKSPACE_ENV, "/trtllm-managed")
    monkeypatch.setenv(_FLASHINFER_MANAGED_ENV, "1")
    monkeypatch.delenv(_FLASHINFER_ISOLATION_ENV, raising=False)
    captured: dict[str, object] = {}

    class FakeMpiPoolExecutor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(mpi_session, "MPIPoolExecutor", FakeMpiPoolExecutor)
    session = SimpleNamespace(
        n_workers=n_workers,
        _env_overrides=env_overrides,
        mpi_pool=None,
    )

    mpi_session.MpiPoolSession._start_mpi_pool(session)

    expected_python_args = (
        [
            "-c",
            mpi_session._FLASHINFER_WORKER_BOOTSTRAP,
            mpi_session._FLASHINFER_WORKSPACE_ROOT,
        ]
        if expect_bootstrap
        else None
    )
    assert captured["python_args"] == expected_python_args
    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get(_FLASHINFER_WORKSPACE_ENV) == expected_workspace
    assert env.get(_FLASHINFER_MANAGED_ENV) == expected_marker


def test_mpi_pool_shares_cubins_but_isolates_workspaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(_FLASHINFER_WORKSPACE_ENV, raising=False)
    monkeypatch.delenv(_FLASHINFER_ISOLATION_ENV, raising=False)
    monkeypatch.setenv(_FLASHINFER_CUBIN_ENV, str(tmp_path / "cubins"))
    workspace_root = tmp_path / "workspaces"
    monkeypatch.setattr(mpi_session, "_FLASHINFER_WORKSPACE_ROOT", str(workspace_root))
    session = mpi_session.MpiPoolSession(n_workers=2)
    try:
        worker_envs = session.submit_sync(_worker_flashinfer_env)
    finally:
        session.shutdown()

    workspaces = {workspace for workspace, _ in worker_envs}
    assert len(workspaces) == 2
    assert all(Path(workspace).is_dir() for workspace in workspaces)
    assert all(Path(workspace).parent == workspace_root for workspace in workspaces)
    assert all(cubin == str(tmp_path / "cubins") for _, cubin in worker_envs)


def test_mpi_pool_propagates_explicit_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = str(tmp_path / "explicit")
    monkeypatch.delenv(_FLASHINFER_ISOLATION_ENV, raising=False)
    monkeypatch.setenv(_FLASHINFER_WORKSPACE_ENV, explicit)
    session = mpi_session.MpiPoolSession(n_workers=2)
    try:
        worker_envs = session.submit_sync(_worker_flashinfer_env)
    finally:
        session.shutdown()

    assert all(workspace == explicit for workspace, _ in worker_envs)
