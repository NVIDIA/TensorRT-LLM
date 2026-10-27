# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 TP2 weight table: the per-rank shapes at tp 2, layer kinds from the config, packed-NVFP4 expert
scalars as FP32 divisors."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
    weights as W,
)

pytestmark = pytest.mark.cpu_only


def _core(sparse_freq, moe_freq, fused=True, tp=2):
    cfg = SimpleNamespace(
        num_hidden_layers=len(sparse_freq),
        moe_layer_freq=moe_freq,
        sparse_attention_config=dict(
            sparse_attention_freq=sparse_freq, sparse_num_index_heads=4, sparse_index_dim=128
        ),
        head_dim=128,
        num_attention_heads=64,
        num_key_value_heads=4,
        hidden_size=6144,
        dense_intermediate_size=12288,
        shared_intermediate_size=3072,
        n_shared_experts=1,
        intermediate_size=3072,
        num_local_experts=128,
        vocab_size=200064,
        torch_dtype=torch.bfloat16,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=cfg,
            mapping=SimpleNamespace(tp_size=tp, tp_rank=0),
            sparse_attention_config=SimpleNamespace(fuse_qkv_index_projection=fused),
        )
    )


def test_the_tp2_per_rank_shapes():
    """32 query / 2 KV / 2 index heads per rank: the five-way projection is 4992 rows (the producer's 39 packed heads),
    the three-way 4608, o_proj reads 4096 columns; dense MLP 6144 and shared / routed experts 1536 per rank (the
    m3_mx_gemv TP2 shapes and the MoE engine's INTER / SHARED_INTER)."""
    d = W.MODEL_WEIGHTS.dims(_core([1] * 4, [0, 1, 1, 1]))
    assert (d.tp, d.heads, d.kv_heads, d.index_heads, d.head_dim, d.index_dim) == (
        2,
        32,
        2,
        2,
        128,
        128,
    )
    assert d.qkv_sparse_rows == 4992 == 39 * 128
    assert d.qkv_dense_rows == 4608
    assert d.q_width == 4096
    assert (2 * d.dense_inter, d.dense_inter) == (12288, 6144)
    assert (2 * d.shared_inter, d.shared_inter) == (3072, 1536)
    assert d.moe_inter == 1536


def test_layer_kinds_come_from_the_config_independently():
    """Sparse attention on every layer while layers 0-1 keep the dense MLP."""
    d = W.MODEL_WEIGHTS.dims(_core([1] * 6, [0, 0, 1, 1, 1, 1]))
    assert list(W._sparse_attn_layers(d)) == [0, 1, 2, 3, 4, 5]
    assert list(W._dense_mlp_layers(d)) == [0, 1]
    assert list(W._moe_layers(d)) == [2, 3, 4, 5]
    assert list(W._five_way_qkv_layers(d)) == [0, 1, 2, 3, 4, 5]
    assert list(W._three_way_qkv_layers(d)) == []


def test_unfused_projection_uses_the_three_way_qkv_everywhere():
    d = W.MODEL_WEIGHTS.dims(_core([0, 0, 1, 1], [0, 0, 1, 1], fused=False))
    assert list(W._three_way_qkv_layers(d)) == [0, 1, 2, 3]
    assert list(W._unfused_sparse_layers(d)) == [2, 3]
    assert list(W._five_way_qkv_layers(d)) == []


def test_expert_scalars_are_the_fp32_reciprocal_of_the_stored_multiplier():
    stored = torch.tensor(12345.678, dtype=torch.float32)
    out = W._divisor(stored, None)
    assert out.dtype == torch.float32
    assert out.view(torch.int32).item() == torch.reciprocal(stored).view(torch.int32).item()
    with pytest.raises(ValueError):
        W._divisor(torch.tensor(0.0), None)
