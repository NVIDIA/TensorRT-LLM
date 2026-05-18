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

import pytest

from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import (
    DeepseekV4AttentionType,
    get_token_bytes,
)
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    DeepSeekV4SparseAttentionConfig,
)

# ---------------------------------------------------------------------------
# Pydantic config validators
# ---------------------------------------------------------------------------


def test_indexer_k_dtype_default_is_fp8():
    cfg = DeepSeekV4SparseAttentionConfig()
    assert cfg.indexer_k_dtype == "fp8"


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
