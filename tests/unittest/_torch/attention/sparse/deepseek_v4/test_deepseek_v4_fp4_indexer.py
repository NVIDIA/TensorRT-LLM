# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Static checks for the DeepSeek-V4 FP4 indexer K cache wiring.

Heavy end-to-end exercises that allocate the V4 cache manager and run the
indexer forward pass live in ``test_deepseek_v4_cache_manager.py``; those
require a Blackwell GPU and DeepSeek-V4-shaped HF configs that are not part
of every CI lane. This module focuses on cheap config-level guarantees that
catch regressions in the FP4 plumbing without needing GPU memory:

- ``DeepSeekV4SparseAttentionConfig`` inherits the single ``indexer_k_dtype``
  knob ("fp8" / "fp4") from the DSA base config; V4 has no V4-only dtype
  field of its own.
- The V4-specific ``get_token_bytes`` returns 132 B/token under "fp8" and
  68 B/token under "fp4" at index_head_dim=128.
- ``Indexer.use_fp4`` is set from this single knob.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import indexer as dsv4_module
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import get_token_bytes
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.indexer import DeepseekV4Indexer
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    DeepSeekV4SparseAttentionConfig,
)

# ---------------------------------------------------------------------------
# Pydantic config validators
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_runtime_sm_validation(monkeypatch):
    """Keep these schema/layout tests independent of the runner GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_indexer_k_dtype_default_is_fp4():
    cfg = DeepSeekV4SparseAttentionConfig()
    assert cfg.indexer_k_dtype == "fp4"


def test_indexer_k_dtype_accepts_fp4():
    cfg = DeepSeekV4SparseAttentionConfig(indexer_k_dtype="fp4")
    assert cfg.indexer_k_dtype == "fp4"


def test_indexer_k_dtype_rejects_non_128_head_dim():
    with pytest.raises(ValueError, match="index_head_dim=128"):
        DeepSeekV4SparseAttentionConfig(
            indexer_k_dtype="fp4",
            index_head_dim=64,
        )


# ---------------------------------------------------------------------------
# Per-token byte size: "fp8" 132 B vs "fp4" 68 B at index_head_dim=128
# ---------------------------------------------------------------------------


def _indexer_compress_bytes(indexer_k_dtype: str) -> int:
    """Wrap V4's get_token_bytes for the INDEXER_COMPRESS attention type."""
    return get_token_bytes(
        head_dim=512,
        index_head_dim=128,
        compress_ratio=4,
        attn_type=DeepseekV4AttentionType.INDEXER_COMPRESS,
        has_fp8_kv_cache=True,
        indexer_k_dtype=indexer_k_dtype,
    )


def test_indexer_compress_token_bytes_fp8():
    # 1 byte per element (128) + 1 fp32 scale per 128 elements (= 4 bytes)
    assert _indexer_compress_bytes("fp8") == 132


def test_indexer_compress_token_bytes_fp4():
    # ½ byte per element (64) + 1 ue8m0 byte per 32 elements (= 4 bytes)
    assert _indexer_compress_bytes("fp4") == 68


def test_indexer_compress_fp4_halves_pool_footprint():
    fp8 = _indexer_compress_bytes("fp8")
    fp4 = _indexer_compress_bytes("fp4")
    assert fp4 / fp8 < 0.52, (
        f"FP4 indexer K cache footprint did not shrink as expected: {fp4}/{fp8}"
    )


def test_indexer_compress_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="Unsupported indexer_k_dtype"):
        _indexer_compress_bytes("bf16")


# ---------------------------------------------------------------------------
# V4 inherits indexer_k_dtype from the DSA base config — there is no
# V4-only dtype knob — so V3 and V4 round-trip through the same field.
# A regression here previously had V4 carry a separate ``indexer_k_cache_dtype``
# knob that silently dropped on the FP8 branch and tripped DeepGEMM's
# ``q_fp.scalar_type() == torch::kFloat8_e4m3fn`` assertion at runtime.
# ---------------------------------------------------------------------------


def test_v4_inherits_indexer_k_dtype_field():
    """V4 must accept the same FP4 knob as the V3 base config."""
    v3 = DeepSeekSparseAttentionConfig(indexer_k_dtype="fp4")
    v4 = DeepSeekV4SparseAttentionConfig(indexer_k_dtype="fp4")
    assert v3.indexer_k_dtype == "fp4"
    assert v4.indexer_k_dtype == "fp4"


def test_v4_has_no_separate_indexer_k_cache_dtype_field():
    """V4 should expose only ``indexer_k_dtype``; the legacy
    ``indexer_k_cache_dtype`` knob has been removed."""
    cfg = DeepSeekV4SparseAttentionConfig()
    assert "indexer_k_cache_dtype" not in cfg.model_fields


# ---------------------------------------------------------------------------
# Indexer-Q stream scheduling
# ---------------------------------------------------------------------------


def test_fused_indexer_q_uses_serial_prepare(monkeypatch):
    """Without a pre-launched aux half, fused Indexer-Q must not overlap the compressor."""
    monkeypatch.setattr(dsv4_module, "do_multi_stream", lambda: True)
    calls = []

    def serial_prepare(*_args, **_kwargs):
        calls.append("serial")
        return object(), object(), None, None, object()

    def overlapped_prepare(*_args, **_kwargs):
        raise AssertionError("fused Indexer-Q must not use aux-stream overlap")

    indexer = SimpleNamespace(
        aux_stream=object(),
        _is_fused_project_mxfp4_enabled=lambda _dtype: True,
        _run_serial_indexer_prepare=serial_prepare,
        _run_overlapped_indexer_prepare=overlapped_prepare,
    )
    metadata = SimpleNamespace(empty_topk_indices_buffer=torch.full((4, 2), -1))
    qr = torch.empty((2, 4), dtype=torch.bfloat16)
    hidden_states = torch.empty_like(qr)
    position_ids = torch.arange(2)

    result = DeepseekV4Indexer.forward(
        indexer,
        qr,
        hidden_states,
        metadata,
        position_ids,
    )

    assert calls == ["serial"]
    assert torch.equal(result, metadata.empty_topk_indices_buffer[:2])


def test_overlapped_prepare_records_aux_outputs_on_consumer_stream(monkeypatch):
    class Recordable:
        def __init__(self):
            self.streams = []

        def record_stream(self, stream):
            self.streams.append(stream)

    class Event:
        def record(self):
            return None

        def wait(self):
            return None

    consumer_stream = object()
    weights = Recordable()
    k_fp8 = Recordable()
    k_scale = Recordable()
    indexer = SimpleNamespace(
        aux_stream=object(),
        indexer_start_event=Event(),
        weights_proj_event=Event(),
        k_cache_update_event=Event(),
        _project_and_quantize_q=lambda _qr, _position_ids: ("q", "q_scale"),
        weights_proj=lambda _hidden_states: weights,
        compressor=lambda _hidden_states, _metadata: (k_fp8, k_scale),
        _update_k_cache_if_needed=lambda *_args: None,
        _apply_weight_scale=lambda value, _scale: value,
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: consumer_stream)

    result = DeepseekV4Indexer._run_overlapped_indexer_prepare(
        indexer,
        object(),
        object(),
        object(),
        object(),
    )

    assert result == ("q", "q_scale", k_fp8, k_scale, weights)
    assert weights.streams == [consumer_stream]
    assert k_fp8.streams == [consumer_stream]
    assert k_scale.streams == [consumer_stream]
