# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
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


@pytest.fixture(autouse=True)
def _isolate_mimalloc_thp_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # _prepare_environment() sets the mimalloc default with a plain os.environ
    # assignment. monkeypatch restores only what it recorded, and delenv records
    # nothing for an absent variable, so setenv records the original value first
    # and teardown restores it (or removes the variable) whatever the test set.
    monkeypatch.setenv(_bootstrap._MIMALLOC_ALLOW_THP_ENV, "")
    monkeypatch.delenv(_bootstrap._MIMALLOC_ALLOW_THP_ENV)
    monkeypatch.setattr(_bootstrap, "_mimalloc_thp_default_applied", False)
    monkeypatch.setattr(_bootstrap, "_thp_disabled_before_torch", None)


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


def test_prepare_environment_keeps_inductor_compile_pool_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_bootstrap._INDUCTOR_QUIESCE_POOL_ENV, raising=False)
    for name in (
        "_setup_unified_cache",
        "_add_trt_llm_dll_directory",
        "_preload_python_lib",
        "_setup_vendored_triton_kernels",
    ):
        monkeypatch.setattr(_bootstrap, name, lambda: None)

    _bootstrap._prepare_environment()

    assert os.environ[_bootstrap._INDUCTOR_QUIESCE_POOL_ENV] == "0"


def test_explicit_inductor_compile_pool_quiesce_setting_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_bootstrap._INDUCTOR_QUIESCE_POOL_ENV, "1")

    _bootstrap._keep_inductor_compile_pool_alive()

    assert os.environ[_bootstrap._INDUCTOR_QUIESCE_POOL_ENV] == "1"


@pytest.fixture
def _mimalloc_default_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # Linux, where the default applies; _isolate_mimalloc_thp_state has already
    # removed the variable and reset the module state.
    monkeypatch.setattr(_bootstrap.sys, "platform", "linux")


@pytest.mark.usefixtures("_mimalloc_default_state")
def test_prepare_environment_keeps_mimalloc_arenas_off_thp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_bootstrap, "_thp_disable_flag", lambda: False)
    for name in (
        "_setup_unified_cache",
        "_add_trt_llm_dll_directory",
        "_preload_python_lib",
        "_setup_vendored_triton_kernels",
    ):
        monkeypatch.setattr(_bootstrap, name, lambda: None)

    _bootstrap._prepare_environment()

    assert os.environ[_bootstrap._MIMALLOC_ALLOW_THP_ENV] == "0"
    assert _bootstrap._mimalloc_thp_default_applied
    assert _bootstrap._thp_disabled_before_torch is False


@pytest.mark.usefixtures("_mimalloc_default_state")
def test_explicit_mimalloc_thp_setting_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_bootstrap._MIMALLOC_ALLOW_THP_ENV, "1")
    calls = []
    monkeypatch.setattr(_bootstrap, "_thp_disable_flag", lambda: True)
    monkeypatch.setattr(_bootstrap, "_clear_thp_disable_flag", lambda: calls.append("clear"))

    _bootstrap._keep_mimalloc_arenas_off_thp()
    _bootstrap._restore_process_thp()

    assert os.environ[_bootstrap._MIMALLOC_ALLOW_THP_ENV] == "1"
    assert calls == []


@pytest.mark.usefixtures("_mimalloc_default_state")
def test_mimalloc_thp_default_is_linux_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_bootstrap.sys, "platform", "win32")

    _bootstrap._keep_mimalloc_arenas_off_thp()

    assert _bootstrap._MIMALLOC_ALLOW_THP_ENV not in os.environ
    assert not _bootstrap._mimalloc_thp_default_applied


def test_mimalloc_thp_default_is_set_before_torch_can_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The option only takes effect if it is in the environment when torch loads
    # mimalloc, and the vendored triton_kernels import is the first step that can
    # import torch.
    calls = []
    for name in (
        "_set_numexpr_thread_default",
        "_keep_inductor_compile_pool_alive",
        "_setup_unified_cache",
        "_add_trt_llm_dll_directory",
        "_preload_python_lib",
    ):
        monkeypatch.setattr(_bootstrap, name, lambda: None)
    monkeypatch.setattr(
        _bootstrap, "_keep_mimalloc_arenas_off_thp", lambda: calls.append("mimalloc")
    )
    monkeypatch.setattr(
        _bootstrap, "_setup_vendored_triton_kernels", lambda: calls.append("triton")
    )

    _bootstrap._prepare_environment()

    assert calls == ["mimalloc", "triton"]


@pytest.mark.usefixtures("_mimalloc_default_state")
@pytest.mark.parametrize(
    ("before", "after", "cleared"),
    [
        (False, True, True),  # the torch import disabled THP: undo it
        (False, False, False),  # nothing changed
        (True, True, False),  # disabled before TensorRT LLM was imported: keep it
        (None, True, False),  # flag not readable before the import: keep it
    ],
)
def test_restore_process_thp_withdraws_the_default(
    monkeypatch: pytest.MonkeyPatch, before, after, cleared
) -> None:
    calls = []
    monkeypatch.setattr(_bootstrap, "_thp_disable_flag", lambda: before)
    _bootstrap._keep_mimalloc_arenas_off_thp()
    assert os.environ[_bootstrap._MIMALLOC_ALLOW_THP_ENV] == "0"
    monkeypatch.setattr(_bootstrap, "_thp_disable_flag", lambda: after)
    monkeypatch.setattr(_bootstrap, "_clear_thp_disable_flag", lambda: calls.append("clear"))

    _bootstrap._restore_process_thp()
    _bootstrap._restore_process_thp()

    assert _bootstrap._MIMALLOC_ALLOW_THP_ENV not in os.environ
    assert calls == (["clear"] if cleared else [])


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_THP_DISABLE is Linux-only")
def test_thp_disable_flag_round_trip() -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    original = libc.prctl(_bootstrap._PR_GET_THP_DISABLE, 0, 0, 0, 0)
    if original == 3:
        # PR_THP_DISABLE_EXCEPT_ADVISED (Linux 6.18+), inherited from the parent.
        # The round trip below would restore it as the plain flag, so only check
        # that it reads as disabled.
        assert _bootstrap._thp_disable_flag() is True
        return
    assert original in (0, 1)
    try:
        assert libc.prctl(_bootstrap._PR_SET_THP_DISABLE, 1, 0, 0, 0) == 0
        assert _bootstrap._thp_disable_flag() is True
        _bootstrap._clear_thp_disable_flag()
        assert _bootstrap._thp_disable_flag() is False
    finally:
        libc.prctl(_bootstrap._PR_SET_THP_DISABLE, original, 0, 0, 0)


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_THP_DISABLE is Linux-only")
def test_importing_tensorrt_llm_keeps_process_thp_enabled() -> None:
    # A fresh interpreter, as a worker process starts: importing the package loads
    # torch, whose mimalloc sees the default, so its arenas are not advised
    # MADV_HUGEPAGE; neither the process's THP setting nor a later library with
    # its own mimalloc is affected by it.
    code = (
        "import ctypes, os\n"
        "libc = ctypes.CDLL(None)\n"
        "if libc.prctl(42, 0, 0, 0, 0) != 0:\n"
        "    raise SystemExit(77)\n"
        "import tensorrt_llm\n"
        "assert 'MIMALLOC_ALLOW_THP' not in os.environ\n"
        "assert libc.prctl(42, 0, 0, 0, 0) == 0\n"
        "try:\n"
        "    import pyarrow\n"
        "except ImportError:\n"
        "    pass\n"
        "assert libc.prctl(42, 0, 0, 0, 0) == 0\n"
        "import torch\n"
        "keep = torch.empty(64 << 20, dtype=torch.uint8)\n"
        "arenas = 0\n"
        "advised = []\n"
        "name = None\n"
        "for line in open('/proc/self/smaps'):\n"
        "    if line[:1] in '0123456789abcdef' and '-' in line.split()[0]:\n"
        "        name = line.split()[5] if len(line.split()) > 5 else ''\n"
        "        arenas += name == '[anon:mimalloc]'\n"
        "    elif line.startswith('VmFlags:') and name == '[anon:mimalloc]' and ' hg' in line:\n"
        "        advised.append(line)\n"
        "if not arenas:\n"
        "    raise SystemExit(78)\n"
        "assert not advised, advised\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "MIMALLOC_ALLOW_THP"}
    result = subprocess.run([sys.executable, "-c", code], env=env, timeout=600)
    if result.returncode == 77:
        pytest.skip("THP is already disabled for processes started here")
    if result.returncode == 78:
        # The process-wide checks passed, but there was no arena to inspect:
        # torch's CPU allocator is not mimalloc here (x86), or the kernel does not
        # name anonymous mappings.
        pytest.skip("no [anon:mimalloc] mapping in /proc/self/smaps")
    assert result.returncode == 0


def _import_tensorrt_llm_failing_on(module: str) -> int:
    # Imports the package in a fresh interpreter in which importing `module`
    # raises, checks that the import failed there, and that the mimalloc default
    # is gone afterwards, as a process that handles the error and goes on sees it.
    code = (
        "import os, sys\n"
        "class _Fail:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        f"        if name == {module!r}:\n"
        "            raise ImportError(name + ' cannot be imported')\n"
        "sys.meta_path.insert(0, _Fail())\n"
        "try:\n"
        "    import tensorrt_llm\n"
        "except ImportError as e:\n"
        f"    assert str(e) == {module + ' cannot be imported'!r}, e\n"
        "else:\n"
        "    raise SystemExit('tensorrt_llm was imported')\n"
        "assert 'MIMALLOC_ALLOW_THP' not in os.environ\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "MIMALLOC_ALLOW_THP"}
    return subprocess.run([sys.executable, "-c", code], env=env, timeout=600).returncode


@pytest.mark.skipif(sys.platform != "linux", reason="the mimalloc default is Linux-only")
def test_failed_torch_import_withdraws_the_mimalloc_default() -> None:
    assert _import_tensorrt_llm_failing_on("torch") == 0


@pytest.mark.skipif(sys.platform != "linux", reason="the mimalloc default is Linux-only")
def test_failed_environment_preparation_withdraws_the_mimalloc_default() -> None:
    # Phase 1 sets the default first, and a later step of it, the vendored
    # triton_kernels import, can still raise.
    assert _import_tensorrt_llm_failing_on("triton_kernels") == 0


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
            mpi_session._WORKER_ENV_BOOTSTRAP + mpi_session._FLASHINFER_WORKER_BOOTSTRAP,
            workspace_root,
        ]
        assert "FLASHINFER_WORKSPACE_BASE" not in worker_env
    else:
        expected_workspace = (
            str(tmp_path / "explicit-flashinfer") if explicit_workspace else workspace_root
        )
        assert captured["python_args"] == [
            "-c",
            mpi_session._WORKER_ENV_BOOTSTRAP + mpi_session._WORKER_SERVER_MAIN,
        ]
        assert worker_env["FLASHINFER_WORKSPACE_BASE"] == expected_workspace
