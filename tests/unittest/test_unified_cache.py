# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm import _bootstrap
from tensorrt_llm.llmapi import mpi_session

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _clear_cache_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_bootstrap._UNIFIED_CACHE_ROOT_ENV, raising=False)
    monkeypatch.delenv(_bootstrap._TRTLLM_DG_DUMP_CUBIN_ENV, raising=False)
    monkeypatch.delenv("TRTLLM_DEEP_GEMM_CACHE_PER_PROCESS", raising=False)
    monkeypatch.delenv("TRTLLM_FLASHINFER_WORKSPACE_MANAGED", raising=False)
    monkeypatch.delenv("TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS", raising=False)
    for name in _bootstrap._UNIFIED_CACHE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_unified_cache_is_disabled_without_root() -> None:
    _bootstrap._setup_unified_cache()

    assert all(name not in os.environ for name in _bootstrap._UNIFIED_CACHE_ENV_VARS)
    assert _bootstrap._TRTLLM_DG_DUMP_CUBIN_ENV not in os.environ


def test_unified_cache_enables_trtllm_deep_gemm_cubin_dump(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "unified"
    monkeypatch.setenv(_bootstrap._UNIFIED_CACHE_ROOT_ENV, str(cache_root))

    _bootstrap._setup_unified_cache()

    assert os.environ[_bootstrap._TRTLLM_DG_CACHE_ENV] == str(cache_root / "trtllm_deep_gemm")
    assert os.environ[_bootstrap._TRTLLM_DG_DUMP_CUBIN_ENV] == "1"


def test_explicit_trtllm_deep_gemm_cache_enables_cubin_dump(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_bootstrap._TRTLLM_DG_CACHE_ENV, str(tmp_path / "deep_gemm"))

    _bootstrap._setup_unified_cache()

    assert os.environ[_bootstrap._TRTLLM_DG_DUMP_CUBIN_ENV] == "1"


def test_explicit_trtllm_deep_gemm_dump_setting_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_bootstrap._TRTLLM_DG_CACHE_ENV, str(tmp_path / "deep_gemm"))
    monkeypatch.setenv(_bootstrap._TRTLLM_DG_DUMP_CUBIN_ENV, "0")

    _bootstrap._setup_unified_cache()

    assert os.environ[_bootstrap._TRTLLM_DG_DUMP_CUBIN_ENV] == "0"


def test_unified_cache_respects_individual_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_bootstrap._UNIFIED_CACHE_ROOT_ENV, str(tmp_path / "unified"))
    for name in _bootstrap._UNIFIED_CACHE_ENV_VARS:
        monkeypatch.setenv(name, f"/explicit/{name.lower()}")

    _bootstrap._setup_unified_cache()

    for name in _bootstrap._UNIFIED_CACHE_ENV_VARS:
        assert os.environ[name] == f"/explicit/{name.lower()}"


@pytest.mark.parametrize(
    "cache_override,isolate,expected",
    [
        (None, True, "isolated"),
        ("unified", True, "isolated"),
        (None, False, "unified"),
        ("explicit", True, "explicit"),
    ],
)
def test_ray_deep_gemm_cache_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cache_override: str | None,
    isolate: bool,
    expected: str,
) -> None:
    from tensorrt_llm.executor.ray.utils import _configure_deep_gemm_cache

    cache_root = tmp_path / "unified"
    cache_dir = cache_root / "deep_gemm"
    monkeypatch.setenv(_bootstrap._UNIFIED_CACHE_ROOT_ENV, str(cache_root))
    if cache_override:
        override = cache_dir if cache_override == "unified" else tmp_path / "explicit"
        monkeypatch.setenv("DG_JIT_CACHE_DIR", str(override))
    if not isolate:
        monkeypatch.setenv("TRTLLM_DEEP_GEMM_CACHE_PER_PROCESS", "0")

    _bootstrap._setup_unified_cache()
    _configure_deep_gemm_cache(rank=2, gpu=3)

    expected_cache = {
        "isolated": cache_dir / "deep_gemm_rank2_gpu3",
        "unified": cache_dir,
        "explicit": tmp_path / "explicit",
    }[expected]
    assert os.environ["DG_JIT_CACHE_DIR"] == str(expected_cache)


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
    captured: dict[str, object] = {}

    class FakeMpiPoolExecutor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setenv(_bootstrap._UNIFIED_CACHE_ROOT_ENV, "/cache")
    for name in _bootstrap._UNIFIED_CACHE_ENV_VARS:
        monkeypatch.setenv(name, f"/cache/{name.lower()}")
    monkeypatch.setenv(_bootstrap._TRTLLM_DG_DUMP_CUBIN_ENV, "1")
    monkeypatch.setenv("UNRELATED_CACHE_DIR", "/not-forwarded")
    monkeypatch.setattr(mpi_session, "MPIPoolExecutor", FakeMpiPoolExecutor, raising=False)
    session = SimpleNamespace(mpi_pool=None, n_workers=1, _env_overrides={})

    mpi_session.MpiPoolSession._start_mpi_pool(session)

    worker_env = captured["env"]
    assert isinstance(worker_env, dict)
    assert worker_env[_bootstrap._UNIFIED_CACHE_ROOT_ENV] == "/cache"
    assert worker_env[_bootstrap._TRTLLM_DG_DUMP_CUBIN_ENV] == "1"
    assert all(worker_env[name] == os.environ[name] for name in _bootstrap._UNIFIED_CACHE_ENV_VARS)
    assert "UNRELATED_CACHE_DIR" not in worker_env


@pytest.mark.parametrize(
    "use_unified_cache,isolate,explicit_workspace",
    [
        (False, True, False),
        (True, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_mpi_pool_flashinfer_isolation_respects_unified_cache_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    use_unified_cache: bool,
    isolate: bool,
    explicit_workspace: bool,
) -> None:
    captured: dict[str, object] = {}

    class FakeMpiPoolExecutor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    cache_root = tmp_path / "unified"
    if use_unified_cache:
        monkeypatch.setenv(_bootstrap._UNIFIED_CACHE_ROOT_ENV, str(cache_root))
    if not isolate:
        monkeypatch.setenv("TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS", "0")
    if explicit_workspace:
        monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(tmp_path / "explicit-flashinfer"))
    monkeypatch.setattr(mpi_session, "MPIPoolExecutor", FakeMpiPoolExecutor, raising=False)
    _bootstrap._setup_unified_cache()
    session = SimpleNamespace(mpi_pool=None, n_workers=2, _env_overrides={})

    mpi_session.MpiPoolSession._start_mpi_pool(session)

    worker_env = captured["env"]
    assert isinstance(worker_env, dict)
    if use_unified_cache:
        workspace_root = str(cache_root / "flashinfer")
    else:
        workspace_root = mpi_session._FLASHINFER_WORKSPACE_ROOT
    if isolate and not explicit_workspace:
        assert captured["python_args"] == [
            "-c",
            mpi_session._FLASHINFER_WORKER_BOOTSTRAP,
            workspace_root,
        ]
        assert "FLASHINFER_WORKSPACE_BASE" not in worker_env
    else:
        expected_workspace = (
            str(tmp_path / "explicit-flashinfer") if explicit_workspace else workspace_root
        )
        assert captured["python_args"] is None
        assert worker_env["FLASHINFER_WORKSPACE_BASE"] == expected_workspace
