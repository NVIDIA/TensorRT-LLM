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

import contextlib
import importlib
import inspect
import pickle
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import pytest

from tensorrt_llm._torch import flashinfer_artifacts
from tensorrt_llm._torch.flashinfer_artifacts import (
    FlashInferArtifactFailure,
    FlashInferArtifactUnavailableError,
    raise_on_flashinfer_artifact_failure,
)

# Tests marked cpu_only run against a fake FlashInfer package, so every failure
# class can be produced without a GPU or a network. The unmarked tests check the
# installed FlashInfer, so that a FlashInfer upgrade that moves get_artifact fails
# here; they run in the GPU stages, where FlashInfer is installed.
cpu_only = pytest.mark.cpu_only

_CUBIN = "8ec29a98/batched_gemm-fa419f4-31ee4e5/Bmm_Bfloat16_t128x8x128.cubin"
_HEADER = "8ec29a98/batched_gemm-fa419f4-31ee4e5/include/flashinferMetaInfo.h"
_SHA256 = "0" * 64
_CUBIN_DIR = "/cache/flashinfer/cubins"
_REPOSITORY = "https://artifacts.example/flashinfer/"


@dataclass
class _FakeFlashInfer:
    cubin_loader: ModuleType
    fused_moe: ModuleType
    outside_module: ModuleType
    original_get_artifact: Callable[..., bytes]
    artifacts: dict[str, bytes | Exception]
    """What fetching each file returns, or raises. Missing files fetch as b""."""

    def fetch(self, file_name: str) -> bytes:
        return self.cubin_loader.get_artifact(file_name, _SHA256)


def _make_cubin_loader(get_artifact: Callable[..., bytes] | None) -> ModuleType:
    cubin_loader = ModuleType("fakeinfer.jit.cubin_loader")
    if get_artifact is not None:
        cubin_loader.get_artifact = get_artifact
        cubin_loader.get_cubin = get_artifact
    cubin_loader.FLASHINFER_CUBIN_DIR = _CUBIN_DIR
    cubin_loader.FLASHINFER_CUBINS_REPOSITORY = _REPOSITORY
    return cubin_loader


def _fail_like_a_kernel() -> None:
    """Raise what a FlashInfer kernel raises when an artifact it needs was not fetched."""
    raise RuntimeError("Error occurred when running GEMM!")


@pytest.fixture
def fake_flashinfer(monkeypatch) -> _FakeFlashInfer:
    """Install the monitor on a fake FlashInfer package for the duration of a test."""
    artifacts: dict[str, bytes | Exception] = {}

    def get_artifact(file_name: str, sha256: str, session=None) -> bytes:
        result = artifacts.get(file_name, b"")
        if isinstance(result, Exception):
            raise result
        return result

    cubin_loader = _make_cubin_loader(get_artifact)
    fused_moe = ModuleType("fakeinfer.jit.fused_moe")
    fused_moe.get_artifact = get_artifact
    outside_module = ModuleType("fakeinfer_consumer")
    outside_module.get_artifact = get_artifact
    loaded_modules = {
        module.__name__: module for module in (cubin_loader, fused_moe, outside_module)
    }
    monkeypatch.setattr(flashinfer_artifacts, "_monitor", None)
    flashinfer_artifacts._install_monitor(cubin_loader, "0.6.18", loaded_modules)
    return _FakeFlashInfer(cubin_loader, fused_moe, outside_module, get_artifact, artifacts)


@pytest.fixture
def logged(monkeypatch) -> dict[str, list[str]]:
    """Capture what the module logs, by level."""
    messages: dict[str, list[str]] = {"error": [], "warning": []}
    for level, captured in messages.items():
        monkeypatch.setattr(flashinfer_artifacts.logger, level, captured.append)
    return messages


@cpu_only
def test_fetched_artifact_is_returned_unchanged(fake_flashinfer):
    fake_flashinfer.artifacts[_CUBIN] = b"\x7fELF"

    with raise_on_flashinfer_artifact_failure():
        assert fake_flashinfer.fetch(_CUBIN) == b"\x7fELF"


@cpu_only
def test_empty_fetch_is_returned_unchanged_and_reported(fake_flashinfer):
    with pytest.raises(FlashInferArtifactUnavailableError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            assert fake_flashinfer.fetch(_CUBIN) == b""
            _fail_like_a_kernel()

    (failure,) = excinfo.value.failures
    assert failure.file_name == _CUBIN
    assert "download failed or sha256 mismatch" in failure.reason
    assert excinfo.value.source.cubin_dir == _CUBIN_DIR
    assert excinfo.value.source.repository == _REPOSITORY


@cpu_only
def test_fetch_error_is_reraised_unchanged_and_reported(fake_flashinfer):
    error = RuntimeError(f"Artifact not found locally: {_HEADER}")
    fake_flashinfer.artifacts[_HEADER] = error

    with pytest.raises(FlashInferArtifactUnavailableError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            fake_flashinfer.fetch(_HEADER)

    assert excinfo.value.__cause__ is error
    assert excinfo.value.failures == (
        FlashInferArtifactFailure(
            file_name=_HEADER, reason=f"RuntimeError: Artifact not found locally: {_HEADER}"
        ),
    )


@cpu_only
def test_each_failed_fetch_is_logged_with_the_file_name(fake_flashinfer, logged):
    fake_flashinfer.fetch(_CUBIN)

    (message,) = logged["error"]
    assert _CUBIN in message
    assert "FLASHINFER_NO_DOWNLOAD=1" in message


@cpu_only
def test_every_reference_inside_the_flashinfer_package_is_replaced(fake_flashinfer):
    monitor = fake_flashinfer.cubin_loader.get_artifact

    assert isinstance(monitor, flashinfer_artifacts._ArtifactFetchMonitor)
    assert inspect.unwrap(monitor) is fake_flashinfer.original_get_artifact
    assert fake_flashinfer.cubin_loader.get_cubin is monitor
    assert fake_flashinfer.fused_moe.get_artifact is monitor
    assert fake_flashinfer.outside_module.get_artifact is fake_flashinfer.original_get_artifact


@cpu_only
def test_installing_again_keeps_the_installed_monitor(fake_flashinfer):
    cubin_loader = fake_flashinfer.cubin_loader
    monitor = cubin_loader.get_artifact

    flashinfer_artifacts._install_monitor(
        cubin_loader, "0.6.18", {cubin_loader.__name__: cubin_loader}
    )

    assert fake_flashinfer.cubin_loader.get_artifact is monitor
    assert inspect.unwrap(monitor) is fake_flashinfer.original_get_artifact


@cpu_only
def test_flashinfer_without_get_artifact_is_left_unchanged_with_a_warning(monkeypatch, logged):
    monkeypatch.setattr(flashinfer_artifacts, "_monitor", None)
    cubin_loader = _make_cubin_loader(get_artifact=None)

    flashinfer_artifacts._install_monitor(cubin_loader, "9.9.9", {})

    (message,) = logged["warning"]
    assert "fakeinfer.jit.cubin_loader.get_artifact was not found (flashinfer 9.9.9)" in message
    assert flashinfer_artifacts._monitor is None
    assert not hasattr(cubin_loader, "get_artifact")


@cpu_only
def test_block_without_failed_fetches_is_transparent(fake_flashinfer):
    error = ValueError("unsupported shape")

    with raise_on_flashinfer_artifact_failure():
        result = "output"
    with pytest.raises(ValueError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            raise error

    assert result == "output"
    assert excinfo.value is error


@cpu_only
def test_error_raised_after_a_failed_fetch_becomes_the_cause(fake_flashinfer):
    gemm_error = RuntimeError("Error occurred when running GEMM!")

    with pytest.raises(FlashInferArtifactUnavailableError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            fake_flashinfer.fetch(_CUBIN)
            raise gemm_error

    assert excinfo.value.__cause__ is gemm_error
    assert _CUBIN in str(excinfo.value)


@cpu_only
def test_failed_fetch_in_a_block_that_completes_is_only_logged(fake_flashinfer, logged):
    def profile_tactic():
        fake_flashinfer.fetch(_CUBIN)
        _fail_like_a_kernel()

    with raise_on_flashinfer_artifact_failure():
        # An autotuner skips a tactic whose kernel cannot be loaded.
        with contextlib.suppress(RuntimeError):
            profile_tactic()
        result = "output"

    assert result == "output"
    (message,) = logged["error"]
    assert _CUBIN in message


@cpu_only
def test_fetch_that_failed_before_the_block_is_not_reported(fake_flashinfer):
    fake_flashinfer.fetch(_CUBIN)
    error = MemoryError("CUDA out of memory")

    with pytest.raises(MemoryError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            raise error

    assert excinfo.value is error


@cpu_only
def test_nested_blocks_report_a_failed_fetch_once(fake_flashinfer):
    gemm_error = RuntimeError("Error occurred when running GEMM!")

    with pytest.raises(FlashInferArtifactUnavailableError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            with raise_on_flashinfer_artifact_failure():
                fake_flashinfer.fetch(_CUBIN)
                raise gemm_error

    assert excinfo.value.__cause__ is gemm_error
    assert len(excinfo.value.failures) == 1


@cpu_only
def test_repeated_failures_of_one_file_are_listed_once(fake_flashinfer):
    with pytest.raises(FlashInferArtifactUnavailableError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            fake_flashinfer.fetch(_CUBIN)
            fake_flashinfer.fetch(_HEADER)
            fake_flashinfer.fetch(_CUBIN)
            _fail_like_a_kernel()

    assert [failure.file_name for failure in excinfo.value.failures] == [_CUBIN, _HEADER]


@cpu_only
def test_block_is_transparent_before_the_monitor_is_installed(monkeypatch):
    monkeypatch.setattr(flashinfer_artifacts, "_monitor", None)

    with raise_on_flashinfer_artifact_failure():
        result = "output"

    assert result == "output"


@cpu_only
@pytest.mark.parametrize("file_name", [_HEADER, _CUBIN], ids=["header", "cubin"])
def test_error_says_where_artifacts_come_from_and_how_to_provide_them(fake_flashinfer, file_name):
    with pytest.raises(FlashInferArtifactUnavailableError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            fake_flashinfer.fetch(file_name)
            _fail_like_a_kernel()

    message = str(excinfo.value)
    assert file_name in message
    assert _CUBIN_DIR in message
    assert _REPOSITORY in message
    assert "flashinfer-cubin==0.6.18" in message
    assert "FLASHINFER_CUBIN_DIR" in message
    assert "FLASHINFER_NO_DOWNLOAD=1" in message
    assert ("until the process restarts" in message) == (file_name == _CUBIN)


@cpu_only
def test_error_survives_pickling(fake_flashinfer):
    # An executor worker pickles its startup error to report it to the proxy process.
    with pytest.raises(FlashInferArtifactUnavailableError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            fake_flashinfer.fetch(_CUBIN)
            _fail_like_a_kernel()

    restored = pickle.loads(pickle.dumps(excinfo.value))

    assert type(restored) is FlashInferArtifactUnavailableError
    assert str(restored) == str(excinfo.value)
    assert restored.failures == excinfo.value.failures
    assert restored.source == excinfo.value.source


@cpu_only
@pytest.mark.parametrize("entry_point", ["forward", "warmup"])
def test_model_engine_entry_points_report_failed_fetches(fake_flashinfer, entry_point):
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    gemm_error = RuntimeError("Error occurred when running GEMM!")

    def step_that_fetches(*args):
        fake_flashinfer.fetch(_CUBIN)
        raise gemm_error

    engine = object.__new__(PyTorchModelEngine)
    # The warmup decorators read these attributes, which __init__ normally sets.
    engine.moe_load_balancer = None
    engine._metrics = {}
    engine._forward_impl = step_that_fetches
    engine._warmup_impl = step_that_fetches

    with pytest.raises(FlashInferArtifactUnavailableError) as excinfo:
        if entry_point == "forward":
            engine.forward(batch=None)
        else:
            engine.warmup(resource_manager=None)

    assert excinfo.value.__cause__ is gemm_error


def test_installed_monitor_replaces_every_reference_to_flashinfer_get_artifact():
    importlib.import_module("tensorrt_llm._torch.flashinfer_utils")
    from flashinfer.jit import cubin_loader

    monitor = cubin_loader.get_artifact
    assert isinstance(monitor, flashinfer_artifacts._ArtifactFetchMonitor)
    original = inspect.unwrap(monitor)
    stale_references = [
        f"{name}.{attribute}"
        for name, module in list(sys.modules.items())
        if module is not None and name.partition(".")[0] == "flashinfer"
        for attribute, value in list(vars(module).items())
        if value is original
    ]
    assert stale_references == []


def test_installed_monitor_reports_a_flashinfer_fetch_that_fails(monkeypatch):
    importlib.import_module("tensorrt_llm._torch.flashinfer_utils")
    from flashinfer.jit import cubin_loader

    missing = "trtllm-test/missing-artifact.cubin"
    # FlashInfer then raises for a missing file instead of downloading it.
    monkeypatch.setenv("FLASHINFER_NO_DOWNLOAD", "1")

    with pytest.raises(FlashInferArtifactUnavailableError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            cubin_loader.get_artifact(missing, _SHA256)

    assert [failure.file_name for failure in excinfo.value.failures] == [missing]
    assert isinstance(excinfo.value.__cause__, RuntimeError)
