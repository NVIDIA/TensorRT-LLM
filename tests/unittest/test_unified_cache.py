# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm import _bootstrap
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi import mpi_session

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _clear_cache_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_bootstrap._UNIFIED_CACHE_ROOT_ENV, raising=False)
    for name in _bootstrap._UNIFIED_CACHE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_unified_cache_is_disabled_without_root() -> None:
    _bootstrap._setup_unified_cache()

    assert all(name not in os.environ for name in _bootstrap._UNIFIED_CACHE_ENV_VARS)


def test_unified_cache_respects_individual_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_bootstrap._UNIFIED_CACHE_ROOT_ENV, str(tmp_path / "unified"))
    for name in _bootstrap._UNIFIED_CACHE_ENV_VARS:
        monkeypatch.setenv(name, f"/explicit/{name.lower()}")

    _bootstrap._setup_unified_cache()

    for name in _bootstrap._UNIFIED_CACHE_ENV_VARS:
        assert os.environ[name] == f"/explicit/{name.lower()}"


def test_prepare_environment_configures_cache_first(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(_bootstrap, "_setup_unified_cache", lambda: calls.append("cache"))
    monkeypatch.setattr(_bootstrap, "_add_trt_llm_dll_directory", lambda: calls.append("dll"))
    monkeypatch.setattr(_bootstrap, "_preload_python_lib", lambda: calls.append("python"))
    monkeypatch.setattr(
        _bootstrap, "_setup_vendored_triton_kernels", lambda: calls.append("triton")
    )

    _bootstrap._prepare_environment()

    assert calls == ["cache", "dll", "python", "triton"]


def test_mpi_pool_environment_forwards_unified_cache_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not ENABLE_MULTI_DEVICE:
        pytest.skip("multi-device required")

    captured: dict[str, object] = {}

    class FakeMpiPoolExecutor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setenv(_bootstrap._UNIFIED_CACHE_ROOT_ENV, "/cache")
    for name in _bootstrap._UNIFIED_CACHE_ENV_VARS:
        monkeypatch.setenv(name, f"/cache/{name.lower()}")
    monkeypatch.setenv("UNRELATED_CACHE_DIR", "/not-forwarded")
    monkeypatch.setattr(mpi_session, "MPIPoolExecutor", FakeMpiPoolExecutor)
    session = SimpleNamespace(mpi_pool=None, n_workers=1, _env_overrides={})

    mpi_session.MpiPoolSession._start_mpi_pool(session)

    worker_env = captured["env"]
    assert isinstance(worker_env, dict)
    assert worker_env[_bootstrap._UNIFIED_CACHE_ROOT_ENV] == "/cache"
    assert all(worker_env[name] == os.environ[name] for name in _bootstrap._UNIFIED_CACHE_ENV_VARS)
    assert "UNRELATED_CACHE_DIR" not in worker_env
