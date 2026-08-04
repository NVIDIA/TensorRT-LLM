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
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm import _flashinfer_workaround as workaround
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi import mpi_session


def test_configure_flashinfer_workspace_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(workaround._FLASHINFER_WORKSPACE_ENV, raising=False)
    monkeypatch.delenv(workaround._FLASHINFER_WORKSPACE_ISOLATION_ENV, raising=False)

    workaround._configure_flashinfer_workspace()

    assert workaround._FLASHINFER_WORKSPACE_ENV not in os.environ


def test_configure_flashinfer_workspace_is_per_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(workaround._FLASHINFER_WORKSPACE_ENV, raising=False)
    monkeypatch.setenv(workaround._FLASHINFER_WORKSPACE_ISOLATION_ENV, "1")
    monkeypatch.setattr(workaround.tempfile, "gettempdir", lambda: str(tmp_path))

    workaround._configure_flashinfer_workspace()

    get_user_id = getattr(os, "getuid", lambda: 0)
    expected = tmp_path / f"trtllm-flashinfer-{get_user_id()}-{os.getpid()}"
    assert os.environ[workaround._FLASHINFER_WORKSPACE_ENV] == str(expected)


def test_configure_flashinfer_workspace_preserves_explicit_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit_workspace = tmp_path / "explicit"
    monkeypatch.setenv(workaround._FLASHINFER_WORKSPACE_ISOLATION_ENV, "1")
    monkeypatch.setenv(workaround._FLASHINFER_WORKSPACE_ENV, str(explicit_workspace))

    workaround._configure_flashinfer_workspace()

    assert os.environ[workaround._FLASHINFER_WORKSPACE_ENV] == str(explicit_workspace)


@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
@pytest.mark.parametrize(
    "n_workers, override, expected",
    [
        (1, None, None),
        (4, None, "1"),
        (4, "0", "0"),
    ],
)
def test_mpi_pool_configures_worker_workspace_isolation(
    monkeypatch: pytest.MonkeyPatch,
    n_workers: int,
    override: str | None,
    expected: str | None,
) -> None:
    captured: dict[str, object] = {}

    class FakeMpiPoolExecutor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.delenv(workaround._FLASHINFER_WORKSPACE_ISOLATION_ENV, raising=False)
    monkeypatch.setattr(mpi_session, "MPIPoolExecutor", FakeMpiPoolExecutor)
    env_overrides = (
        {workaround._FLASHINFER_WORKSPACE_ISOLATION_ENV: override} if override is not None else {}
    )
    session = SimpleNamespace(
        n_workers=n_workers,
        _env_overrides=env_overrides,
        mpi_pool=None,
    )

    mpi_session.MpiPoolSession._start_mpi_pool(session)

    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get(workaround._FLASHINFER_WORKSPACE_ISOLATION_ENV) == expected
