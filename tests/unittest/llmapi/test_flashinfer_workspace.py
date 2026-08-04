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


def _run_worker_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "main", lambda: None)
    exec(mpi_session._FLASHINFER_WORKER_BOOTSTRAP, {})


def test_worker_bootstrap_uses_rank_and_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(_FLASHINFER_WORKSPACE_ENV, raising=False)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    _run_worker_bootstrap(monkeypatch)

    rank = mpi_session.mpi4py.MPI.COMM_WORLD.Get_rank()
    expected = tmp_path / f"trtllm-flashinfer-{rank}-{os.getpid()}"
    assert os.environ[_FLASHINFER_WORKSPACE_ENV] == str(expected)


def test_worker_bootstrap_preserves_explicit_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit_workspace = tmp_path / "explicit"
    monkeypatch.setenv(_FLASHINFER_WORKSPACE_ENV, str(explicit_workspace))

    _run_worker_bootstrap(monkeypatch)

    assert os.environ[_FLASHINFER_WORKSPACE_ENV] == str(explicit_workspace)


@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
@pytest.mark.parametrize(
    "n_workers, expected",
    [
        (1, None),
        (4, ["-c", mpi_session._FLASHINFER_WORKER_BOOTSTRAP]),
    ],
)
def test_mpi_pool_configures_worker_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    n_workers: int,
    expected: list[str] | None,
) -> None:
    captured: dict[str, object] = {}

    class FakeMpiPoolExecutor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(mpi_session, "MPIPoolExecutor", FakeMpiPoolExecutor)
    session = SimpleNamespace(
        n_workers=n_workers,
        _env_overrides={},
        mpi_pool=None,
    )

    mpi_session.MpiPoolSession._start_mpi_pool(session)

    assert captured["python_args"] == expected
