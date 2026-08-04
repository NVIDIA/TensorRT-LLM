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
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from mpi4py.futures import server

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi import mpi_session

_FLASHINFER_WORKSPACE_ENV = "FLASHINFER_WORKSPACE_BASE"
_FLASHINFER_CUBIN_ENV = "FLASHINFER_CUBIN_DIR"
pytestmark = pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")


def _worker_flashinfer_env() -> tuple[str | None, str | None]:
    mpi_session.mpi4py.MPI.COMM_WORLD.barrier()
    return (os.environ.get(_FLASHINFER_WORKSPACE_ENV), os.environ.get(_FLASHINFER_CUBIN_ENV))


def _run_worker_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "main", lambda: None)
    exec(mpi_session._FLASHINFER_WORKER_BOOTSTRAP, {})


def test_worker_bootstrap_uses_rank_and_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(_FLASHINFER_WORKSPACE_ENV, raising=False)
    workspace = tmp_path / f"trtllm-flashinfer-0-{os.getpid()}"
    prefixes = []

    def fake_mkdtemp(prefix: str) -> str:
        prefixes.append(prefix)
        return str(workspace)

    monkeypatch.setattr(tempfile, "mkdtemp", fake_mkdtemp)

    _run_worker_bootstrap(monkeypatch)

    rank = mpi_session.mpi4py.MPI.COMM_WORLD.Get_rank()
    assert prefixes == [f"trtllm-flashinfer-{rank}-{os.getpid()}-"]
    assert os.environ[_FLASHINFER_WORKSPACE_ENV] == str(workspace)
    assert not workspace.exists()


def test_worker_bootstrap_preserves_explicit_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit_workspace = tmp_path / "explicit"
    monkeypatch.setenv(_FLASHINFER_WORKSPACE_ENV, str(explicit_workspace))

    _run_worker_bootstrap(monkeypatch)

    assert os.environ[_FLASHINFER_WORKSPACE_ENV] == str(explicit_workspace)


@pytest.mark.parametrize(
    "n_workers, env_overrides, expected",
    [
        (1, {}, None),
        (4, {}, ["-c", mpi_session._FLASHINFER_WORKER_BOOTSTRAP]),
        (4, {"TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS": "0"}, None),
        (4, {_FLASHINFER_WORKSPACE_ENV: "/explicit"}, None),
    ],
)
def test_mpi_pool_configures_worker_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    n_workers: int,
    env_overrides: dict[str, str],
    expected: list[str] | None,
) -> None:
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
    if expected:
        assert env[_FLASHINFER_CUBIN_ENV].endswith("/.cache/flashinfer/cubins")


def test_mpi_pool_shares_cubins_but_isolates_workspaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(_FLASHINFER_WORKSPACE_ENV, raising=False)
    monkeypatch.setenv(_FLASHINFER_CUBIN_ENV, str(tmp_path / "cubins"))
    session = mpi_session.MpiPoolSession(n_workers=2)
    try:
        worker_envs = session.submit_sync(_worker_flashinfer_env)
    finally:
        session.shutdown()

    workspaces = {workspace for workspace, _ in worker_envs}
    assert len(workspaces) == 2
    assert all(cubin == str(tmp_path / "cubins") for _, cubin in worker_envs)


def test_mpi_pool_propagates_explicit_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = str(tmp_path / "explicit")
    monkeypatch.setenv(_FLASHINFER_WORKSPACE_ENV, explicit)
    session = mpi_session.MpiPoolSession(n_workers=2)
    try:
        worker_envs = session.submit_sync(_worker_flashinfer_env)
    finally:
        session.shutdown()

    assert all(workspace == explicit for workspace, _ in worker_envs)
