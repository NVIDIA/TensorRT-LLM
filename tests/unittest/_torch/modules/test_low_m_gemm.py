# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.modules import linear as linear_module
from tensorrt_llm._torch.modules.low_m_gemm import (
    LowMGemmBackend,
    LowMGemmDispatcher,
    _configured_backend,
    _device_sm,
    _infer_op_name,
    _normalize_backend,
    _prefer_cublas_for_auto,
    _ShapeCollector,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("off", LowMGemmBackend.OFF),
        ("auto", LowMGemmBackend.AUTO),
        ("cute-dsl", LowMGemmBackend.FLASHINFER),
        ("flashinfer_cute_dsl", LowMGemmBackend.FLASHINFER),
        ("cublaslt", LowMGemmBackend.CUBLAS),
    ],
)
def test_normalize_backend(value: str, expected: LowMGemmBackend) -> None:
    assert _normalize_backend(value) == expected


def test_normalize_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="TRTLLM_LOW_M_GEMM_BACKEND"):
        _normalize_backend("split-all-shapes")


def test_legacy_exact_m_backend_environment_is_ignored(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_LOW_M_GEMM_BACKEND", raising=False)
    monkeypatch.setenv("TLLM_BF16_GEMM_BACKEND", "heuristic")
    assert _configured_backend() == LowMGemmBackend.OFF


def test_device_sm_capability_is_cached(monkeypatch) -> None:
    get_device_capability = MagicMock(return_value=(10, 3))
    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)
    _device_sm.cache_clear()
    try:
        assert _device_sm(7) == 103
        assert _device_sm(7) == 103
        get_device_capability.assert_called_once_with(7)
    finally:
        _device_sm.cache_clear()


def _install_fake_flashinfer(
    monkeypatch,
    *,
    version: str = "0.6.17.dev20260806",
    cute_dsl_available: bool = True,
) -> MagicMock:
    mm_bf16 = MagicMock()
    mm_bf16.is_backend_supported = MagicMock(return_value=True)

    flashinfer_module = ModuleType("flashinfer")
    flashinfer_module.__path__ = []
    flashinfer_module.__version__ = version
    flashinfer_module.mm_bf16 = mm_bf16
    cute_dsl_module = ModuleType("flashinfer.cute_dsl")
    cute_dsl_module.__path__ = []
    cute_dsl_utils_module = ModuleType("flashinfer.cute_dsl.utils")
    cute_dsl_utils_module.is_cute_dsl_available = lambda: cute_dsl_available

    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_module)
    monkeypatch.setitem(sys.modules, "flashinfer.cute_dsl", cute_dsl_module)
    monkeypatch.setitem(sys.modules, "flashinfer.cute_dsl.utils", cute_dsl_utils_module)
    return mm_bf16


def test_auto_prepare_uses_packaged_flashinfer_without_tuning_cache(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_LOW_M_GEMM_BACKEND", "auto")
    monkeypatch.setenv("TRTLLM_LOW_M_GEMM_TUNING_CACHE", "/missing/dispatch.json")
    monkeypatch.setenv("TRTLLM_FLASHINFER_AUTOTUNER_CACHE", "/missing/flashinfer.json")
    mm_bf16 = _install_fake_flashinfer(monkeypatch)
    module = torch.nn.Linear(8, 8)

    dispatcher = LowMGemmDispatcher()
    dispatcher.prepare(module, cuda_graph_enabled=True)

    assert dispatcher._flashinfer_mm is mm_bf16
    assert dispatcher._prepared
    assert module._low_m_gemm_name == ""


def test_prepare_rejects_flashinfer_before_split_k_nightly(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_LOW_M_GEMM_BACKEND", "auto")
    _install_fake_flashinfer(monkeypatch, version="0.6.15")

    with pytest.raises(RuntimeError, match="0.6.17.dev20260806"):
        LowMGemmDispatcher().prepare(torch.nn.Linear(8, 8), cuda_graph_enabled=False)


@pytest.mark.parametrize(("m", "expected"), ((32, True), (33, False)))
def test_flashinfer_candidate_shape_stops_at_m32(m: int, expected: bool, monkeypatch) -> None:
    device = torch.device("cuda")
    input_tensor = MagicMock(
        ndim=2,
        is_cuda=True,
        device=device,
        dtype=torch.bfloat16,
        shape=(m, 128),
    )
    input_tensor.is_contiguous.return_value = True
    input_tensor.numel.return_value = m * 128
    input_tensor.data_ptr.return_value = 32
    weight = MagicMock(
        ndim=2,
        is_cuda=True,
        device=device,
        dtype=torch.bfloat16,
        shape=(256, 128),
    )
    weight.is_contiguous.return_value = True
    weight.data_ptr.return_value = 32
    monkeypatch.setattr("tensorrt_llm._torch.modules.low_m_gemm._current_sm", lambda unused: 103)

    with torch.inference_mode():
        assert LowMGemmDispatcher()._is_candidate_shape(input_tensor, weight, None) is expected


def test_linear_fast_rejects_m_above_flashinfer_domain(monkeypatch) -> None:
    monkeypatch.setattr(linear_module, "LOW_M_GEMM_ACTIVE", True)
    monkeypatch.setattr(linear_module, "_LOW_M_GEMM_SHAPE_COLLECTION_ACTIVE", False)

    assert linear_module._should_apply_low_m_gemm(torch.empty((32, 128)))
    assert not linear_module._should_apply_low_m_gemm(torch.empty((33, 128)))


def test_linear_fast_reject_preserves_full_shape_collection(monkeypatch) -> None:
    monkeypatch.setattr(linear_module, "LOW_M_GEMM_ACTIVE", True)
    monkeypatch.setattr(linear_module, "_LOW_M_GEMM_SHAPE_COLLECTION_ACTIVE", True)

    assert linear_module._should_apply_low_m_gemm(torch.empty((64, 128)))


@pytest.mark.parametrize(
    "layer,expected",
    [
        ("LMHead", "lm_head"),
        ("model.layers.92.fc", "mtp_fusion_fc"),
        ("model.layers.0.self_attn.qkv_proj", "attention_qkv"),
    ],
)
def test_infer_op_name_for_hot_modules(layer: str, expected: str) -> None:
    assert _infer_op_name(layer) == expected


@pytest.mark.parametrize(
    "m,n,k,expected",
    [
        (7, 8192, 128, False),
        (8, 8192, 128, True),
        (14, 15520, 8192, False),
        (15, 15520, 8192, True),
        (15, 2304, 8192, False),
        (16, 2304, 8192, True),
        (17, 2304, 8192, False),
        (32, 8192, 1024, False),
    ],
)
def test_auto_cublas_crossover_for_blackwell(m: int, n: int, k: int, expected: bool) -> None:
    assert _prefer_cublas_for_auto(m, n, k, sm=103) is expected
    assert not _prefer_cublas_for_auto(m, n, k, sm=100)


def test_apply_uses_public_flashinfer_heuristic(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_LOW_M_GEMM_BACKEND", "auto")
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    dispatcher = LowMGemmDispatcher()
    dispatcher._prepared = True
    dispatcher._flashinfer_mm = MagicMock(return_value=torch.empty((4, 256)))
    monkeypatch.setattr(dispatcher, "_is_candidate_shape", lambda *unused: True)
    monkeypatch.setattr(dispatcher, "_is_flashinfer_supported", lambda unused: True)
    monkeypatch.setattr("tensorrt_llm._torch.modules.low_m_gemm._current_sm", lambda unused: 103)

    input_tensor = torch.empty((2, 2, 128), dtype=torch.bfloat16)
    weight = torch.empty((256, 128), dtype=torch.bfloat16)
    bias = torch.empty((256,), dtype=torch.bfloat16)
    with torch.inference_mode():
        output = dispatcher.apply(torch.nn.Linear(1, 1), input_tensor, weight, bias)

    assert output.shape == (2, 2, 256)
    args, kwargs = dispatcher._flashinfer_mm.call_args
    assert args[0].shape == (4, 128)
    assert args[1].shape == (128, 256)
    assert kwargs["bias"].data_ptr() == bias.data_ptr()
    assert kwargs["bias"].shape == bias.shape
    assert kwargs["bias"].dtype == bias.dtype
    assert kwargs["pdl"] is False
    assert kwargs["out_dtype"] == torch.bfloat16
    assert kwargs["backend"] == "cute-dsl"


@pytest.mark.parametrize(
    "backend,expected_flashinfer_calls",
    [("auto", 0), ("flashinfer", 1)],
)
def test_explicit_flashinfer_overrides_auto_cublas_crossover(
    monkeypatch, backend: str, expected_flashinfer_calls: int
) -> None:
    monkeypatch.setenv("TRTLLM_LOW_M_GEMM_BACKEND", backend)
    dispatcher = LowMGemmDispatcher()
    dispatcher._prepared = True
    dispatcher._flashinfer_mm = MagicMock(return_value=torch.empty((8, 8192)))
    monkeypatch.setattr(dispatcher, "_is_candidate_shape", lambda *unused: True)
    monkeypatch.setattr(dispatcher, "_is_flashinfer_supported", lambda unused: True)
    monkeypatch.setattr("tensorrt_llm._torch.modules.low_m_gemm._current_sm", lambda unused: 103)

    input_tensor = torch.empty((8, 128), dtype=torch.bfloat16)
    weight = torch.empty((8192, 128), dtype=torch.bfloat16)
    with torch.inference_mode():
        output = dispatcher.apply(torch.nn.Linear(1, 1), input_tensor, weight, None)

    assert dispatcher._flashinfer_mm.call_count == expected_flashinfer_calls
    assert (output is not None) is bool(expected_flashinfer_calls)


def test_shape_collector_persists_new_runtime_shape_after_warmup(tmp_path) -> None:
    output = tmp_path / "shapes.json"
    collector = _ShapeCollector(str(output))
    module = torch.nn.Linear(8, 4, bias=False)
    module._low_m_gemm_name = "model.layers.0.self_attn.qkv_proj"
    weight = torch.empty((4, 8))

    try:
        collector.record(module, torch.empty((1, 8)), weight, None, cuda_graph=True)
        assert not output.exists()

        collector.flush(mark_warmup_complete=True)
        warmup_document = json.loads(output.read_text(encoding="utf-8"))
        assert [(row["m"], row["call_count"]) for row in warmup_document["shapes"]] == [(1, 1)]

        collector.record(module, torch.empty((1, 8)), weight, None, cuda_graph=True)
        collector.record(module, torch.empty((2, 8)), weight, None, cuda_graph=True)
        runtime_document = json.loads(output.read_text(encoding="utf-8"))
        assert {(row["m"], row["call_count"]) for row in runtime_document["shapes"]} == {
            (1, 2),
            (2, 1),
        }

        another_module = torch.nn.Linear(8, 4, bias=False)
        another_module._low_m_gemm_name = "model.layers.1.self_attn.qkv_proj"
        collector.record(another_module, torch.empty((2, 8)), weight, None, cuda_graph=True)
        unchanged_document = json.loads(output.read_text(encoding="utf-8"))
        assert unchanged_document == runtime_document

        collector.flush()
        final_document = json.loads(output.read_text(encoding="utf-8"))
        assert {
            (row["m"], row["layer"], row["call_count"]) for row in final_document["shapes"]
        } == {
            (1, "model.layers.0.self_attn.qkv_proj", 2),
            (2, "model.layers.0.self_attn.qkv_proj", 1),
            (2, "model.layers.1.self_attn.qkv_proj", 1),
        }
    finally:
        atexit.unregister(collector.flush)
