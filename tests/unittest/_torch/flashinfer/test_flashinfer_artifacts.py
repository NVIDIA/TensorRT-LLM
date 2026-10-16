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
from types import FunctionType, ModuleType

import pytest
import torch

from tensorrt_llm._torch import flashinfer_artifacts
from tensorrt_llm._torch.flashinfer_artifacts import (
    FlashInferArtifactUnavailableError,
    raise_on_flashinfer_artifact_failure,
)

# These tests run in the CPU stages. Most use a fake FlashInfer package, and the
# tests of the installed FlashInfer only inspect it or fetch under
# FLASHINFER_NO_DOWNLOAD=1, so none of them needs a GPU or the network.
pytestmark = pytest.mark.cpu_only

_CUBIN = "8ec29a98/batched_gemm-fa419f4-31ee4e5/Bmm_Bfloat16_t128x8x128.cubin"
_HEADER = "8ec29a98/batched_gemm-fa419f4-31ee4e5/include/flashinferMetaInfo.h"
_SHA256 = "0" * 64
_CUBIN_DIR = "/cache/flashinfer/cubins"
_REPOSITORY = "https://artifacts.example/flashinfer/"
_KERNEL_ERROR = "RuntimeError: Error occurred when running GEMM!"


def _move_into(module: ModuleType, function: Callable[..., object]) -> Callable[..., object]:
    """Copy ``function`` into ``module``, so that its frames belong to that module."""
    moved = FunctionType(
        function.__code__,
        vars(module),
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    setattr(module, function.__name__, moved)
    return moved


def _make_cubin_loader() -> ModuleType:
    cubin_loader = ModuleType("fakeinfer.jit.cubin_loader")
    cubin_loader.FLASHINFER_CUBIN_DIR = _CUBIN_DIR
    cubin_loader.FLASHINFER_CUBINS_REPOSITORY = _REPOSITORY
    return cubin_loader


@dataclass
class _FakeFlashInfer:
    cubin_loader: ModuleType
    artifacts: dict[str, bytes | Exception]
    """What fetching each file returns, or raises. Missing files fetch as b""."""
    run_kernel: Callable[[], object]
    """Fails inside the package, as a FlashInfer kernel does without its artifact."""

    def fetch(self, file_name: str) -> bytes:
        return self.cubin_loader.get_artifact(file_name, _SHA256)


@pytest.fixture
def fake_flashinfer(monkeypatch) -> _FakeFlashInfer:
    """Install the monitor on a fake FlashInfer package for the duration of a test."""
    artifacts: dict[str, bytes | Exception] = {}

    def get_artifact(file_name: str, sha256: str, session=None) -> bytes:
        result = artifacts.get(file_name, b"")
        if isinstance(result, Exception):
            raise result
        return result

    def run_kernel() -> None:
        raise RuntimeError("Error occurred when running GEMM!")

    cubin_loader = _make_cubin_loader()
    _move_into(cubin_loader, get_artifact)
    kernel = _move_into(ModuleType("fakeinfer.fused_moe"), run_kernel)
    monkeypatch.setattr(flashinfer_artifacts, "_monitor", None)
    flashinfer_artifacts._install_monitor(
        cubin_loader, "0.6.18", {cubin_loader.__name__: cubin_loader}
    )
    return _FakeFlashInfer(cubin_loader, artifacts, kernel)


@pytest.fixture
def logged(monkeypatch) -> dict[str, list[str]]:
    """Capture what the module logs, by level."""
    messages: dict[str, list[str]] = {"error": [], "warning": []}
    for level, captured in messages.items():
        monkeypatch.setattr(flashinfer_artifacts.logger, level, captured.append)
    return messages


@pytest.mark.parametrize(
    "result, reason",
    [
        (b"", "download failed or sha256 mismatch"),
        (RuntimeError(f"Artifact not found locally: {_CUBIN}"), "RuntimeError: Artifact not found"),
    ],
    ids=["empty", "raising"],
)
def test_failed_fetch_is_passed_through_and_logged(fake_flashinfer, logged, result, reason):
    fake_flashinfer.artifacts[_CUBIN] = result

    if isinstance(result, Exception):
        with pytest.raises(RuntimeError) as excinfo:
            fake_flashinfer.fetch(_CUBIN)
        assert excinfo.value is result
    else:
        assert fake_flashinfer.fetch(_CUBIN) == result

    (message,) = logged["error"]
    assert _CUBIN in message
    assert reason in message
    assert "FLASHINFER_NO_DOWNLOAD=1" in message


def test_flashinfer_without_get_artifact_is_left_unchanged_with_a_warning(monkeypatch, logged):
    monkeypatch.setattr(flashinfer_artifacts, "_monitor", None)
    cubin_loader = _make_cubin_loader()

    flashinfer_artifacts._install_monitor(cubin_loader, "9.9.9", {})

    (message,) = logged["warning"]
    assert "fakeinfer.jit.cubin_loader.get_artifact was not found (flashinfer 9.9.9)" in message
    assert flashinfer_artifacts._monitor is None
    assert not hasattr(cubin_loader, "get_artifact")


def test_block_without_a_failed_fetch_is_transparent(fake_flashinfer, monkeypatch):
    fake_flashinfer.artifacts[_CUBIN] = b"\x7fELF"

    with raise_on_flashinfer_artifact_failure():
        assert fake_flashinfer.fetch(_CUBIN) == b"\x7fELF"
    with pytest.raises(RuntimeError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            fake_flashinfer.fetch(_CUBIN)
            fake_flashinfer.run_kernel()
    assert type(excinfo.value) is RuntimeError

    # As without FlashInfer, where the monitor is never installed.
    monkeypatch.setattr(flashinfer_artifacts, "_monitor", None)
    with raise_on_flashinfer_artifact_failure():
        result = "output"
    assert result == "output"


def test_flashinfer_error_after_a_failed_fetch_names_the_files_and_the_original_error(
    fake_flashinfer,
):
    with pytest.raises(FlashInferArtifactUnavailableError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            fake_flashinfer.fetch(_CUBIN)
            fake_flashinfer.fetch(_HEADER)
            fake_flashinfer.fetch(_CUBIN)
            fake_flashinfer.run_kernel()

    error = excinfo.value
    assert [failure.file_name for failure in error.failures] == [_CUBIN, _HEADER]
    assert "download failed or sha256 mismatch" in error.failures[0].reason
    assert error.source.cubin_dir == _CUBIN_DIR
    assert error.source.repository == _REPOSITORY
    assert error.original_error == _KERNEL_ERROR
    assert type(error.__cause__) is RuntimeError
    message = str(error)
    for expected in (
        _CUBIN,
        _HEADER,
        _CUBIN_DIR,
        _REPOSITORY,
        "flashinfer-cubin==0.6.18",
        "FLASHINFER_CUBIN_DIR",
        "FLASHINFER_NO_DOWNLOAD=1",
        "until the process restarts",
    ):
        assert expected in message
    assert message.endswith(f"Original error: {_KERNEL_ERROR}")


def test_error_from_outside_flashinfer_propagates_unchanged(fake_flashinfer, logged):
    error = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB.")

    with pytest.raises(torch.cuda.OutOfMemoryError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            # An autotuner skips a tactic whose kernel cannot be loaded, and capturing
            # a CUDA graph later in the same warmup runs out of memory.
            with contextlib.suppress(RuntimeError):
                fake_flashinfer.fetch(_CUBIN)
                fake_flashinfer.run_kernel()
            raise error

    assert excinfo.value is error
    (message,) = logged["error"]
    assert _CUBIN in message


def test_failed_fetch_in_a_block_that_completes_is_only_logged(fake_flashinfer, logged):
    with raise_on_flashinfer_artifact_failure():
        # An autotuner skips a tactic whose kernel cannot be loaded.
        with contextlib.suppress(RuntimeError):
            fake_flashinfer.fetch(_CUBIN)
            fake_flashinfer.run_kernel()
        result = "output"

    assert result == "output"
    (message,) = logged["error"]
    assert _CUBIN in message


def test_fetch_that_failed_before_the_block_is_not_reported(fake_flashinfer):
    fake_flashinfer.fetch(_CUBIN)

    with pytest.raises(RuntimeError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            fake_flashinfer.run_kernel()

    assert type(excinfo.value) is RuntimeError


def test_error_survives_pickling(fake_flashinfer):
    # An executor worker pickles its startup error to report it to the proxy process.
    with pytest.raises(FlashInferArtifactUnavailableError) as excinfo:
        with raise_on_flashinfer_artifact_failure():
            fake_flashinfer.fetch(_CUBIN)
            fake_flashinfer.run_kernel()

    restored = pickle.loads(pickle.dumps(excinfo.value))

    assert type(restored) is FlashInferArtifactUnavailableError
    assert str(restored) == str(excinfo.value)
    assert restored.failures == excinfo.value.failures
    assert restored.source == excinfo.value.source
    assert restored.original_error == _KERNEL_ERROR


@pytest.mark.parametrize("entry_point", ["forward", "warmup"])
def test_model_engine_entry_points_report_failed_fetches(fake_flashinfer, entry_point):
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    def step_that_fetches(*args):
        fake_flashinfer.fetch(_CUBIN)
        fake_flashinfer.run_kernel()

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

    assert excinfo.value.original_error == _KERNEL_ERROR


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

    (failure,) = excinfo.value.failures
    assert failure.file_name == missing
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    # FlashInfer raised the fetch error itself, so the message names it once.
    assert failure.reason == excinfo.value.original_error
    assert str(excinfo.value).count(excinfo.value.original_error) == 1
