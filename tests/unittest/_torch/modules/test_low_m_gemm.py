# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.modules import linear as linear_module
from tensorrt_llm._torch.modules.low_m_gemm import (
    GemmDispatchKey,
    GemmTuningResult,
    LowMGemmBackend,
    LowMGemmDispatcher,
    _configured_backend,
    _flashinfer_commit,
    _infer_op_name,
    _normalize_backend,
    _ShapeCollector,
    write_dispatch_cache,
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


def test_flashinfer_commit_uses_package_identity(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_FLASHINFER_COMMIT", raising=False)
    assert _flashinfer_commit(SimpleNamespace(__git_version__="abc123")) == "abc123"


def test_flashinfer_commit_rejects_false_environment_identity(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_FLASHINFER_COMMIT", "declared")
    with pytest.raises(RuntimeError, match="does not match"):
        _flashinfer_commit(SimpleNamespace(__git_version__="actual"))


def test_dispatch_key_includes_shape_layout_and_graph_mode() -> None:
    key = GemmDispatchKey(sm=103, m=4, n=2304, k=8192)
    assert key.cache_key() == "sm103:bf16:4x2304x8192:nt:nobias:graph"
    assert (
        GemmDispatchKey(sm=103, m=4, n=2304, k=8192, cuda_graph=False)
        .cache_key()
        .endswith(":eager")
    )
    assert (
        GemmDispatchKey(sm=103, m=4, n=2304, k=8192, has_bias=True)
        .cache_key()
        .endswith(":bias:graph")
    )


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


def test_write_dispatch_cache(tmp_path) -> None:
    output = tmp_path / "dispatch.json"
    key = GemmDispatchKey(sm=103, m=1, n=256, k=8192).cache_key()
    write_dispatch_cache(
        output,
        {"flashinfer_commit": "b195c7a8"},
        {
            key: GemmTuningResult(
                backend="flashinfer",
                algorithm="direct",
                latency_us=2.1,
                baseline_us=3.3,
            )
        },
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == 3
    assert document["metadata"]["flashinfer_commit"] == "b195c7a8"
    assert document["entries"][key]["algorithm"] == "direct"


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


def test_auto_cache_can_keep_shape_on_cublas_without_flashinfer(tmp_path, monkeypatch) -> None:
    output = tmp_path / "dispatch.json"
    key = GemmDispatchKey(sm=103, m=2, n=2304, k=8192, cuda_graph=False).cache_key()
    write_dispatch_cache(
        output,
        {"pdl": True, "cuda_version": torch.version.cuda or "none"},
        {key: GemmTuningResult(backend="cublas", algorithm="cublas")},
    )
    monkeypatch.setenv("TRTLLM_LOW_M_GEMM_BACKEND", "auto")
    monkeypatch.setenv("TRTLLM_LOW_M_GEMM_TUNING_CACHE", str(output))
    dispatcher = LowMGemmDispatcher()
    module = torch.nn.Linear(8, 8)
    dispatcher.prepare(module, cuda_graph_enabled=False)
    assert (
        dispatcher._select_backend(GemmDispatchKey(sm=103, m=2, n=2304, k=8192, cuda_graph=False))
        == LowMGemmBackend.CUBLAS
    )
    assert module._low_m_gemm_name == ""


def test_cached_tactic_requires_exact_fields() -> None:
    result = GemmTuningResult(
        backend="flashinfer",
        algorithm="simt",
        tactic={"block_size": 256, "rows_per_block": 4},
    )
    with pytest.raises(RuntimeError, match="Invalid 'simt' tactic"):
        LowMGemmDispatcher._tactic_values(
            result,
            ("block_size", "outputs_per_block", "rows_per_block"),
        )
