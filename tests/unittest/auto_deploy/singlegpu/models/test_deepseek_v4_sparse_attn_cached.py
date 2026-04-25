# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached sparse attention op equivalence tests.

Validates that ``torch_deepseek_v4_sparse_attn_with_cache`` produces the
same last-token output when decoding one step at a time as when the whole
sequence is run in a single fresh prefill.

Strategy: run a prefill of length ``N``, seed the caches. Then for ``M``
decode steps, call the cached op with one token at a time and check the
final attention output against a fresh prefill of length ``N + M``.
"""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention import (  # noqa: F401
    deepseek_v4_sparse_attention,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention import (
    _apply_interleaved_rope,
    _fake_fp8_act_quant,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Config,
    _build_freqs_cis,
)


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _small_cfg(ratio: int) -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=200,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        sliding_window=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=0,
        moe_intermediate_size=32,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
        max_position_embeddings=max(64, ratio * 2),
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
        rope_scaling=None,
        compress_ratios=[ratio],
        index_n_heads=4,
        index_head_dim=16,
        index_topk=2,
        swiglu_limit=0.0,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )


def _pack_op_qkv_args(
    attn: DeepseekV4Attention,
    q: torch.Tensor,
    kv: torch.Tensor,
    x: torch.Tensor,
    qr: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_anchor: torch.Tensor,
    sin_anchor: torch.Tensor,
) -> tuple:
    if attn.indexer is not None:
        indexer_args = (
            attn.indexer.wq_b.weight,
            attn.indexer.weights_proj.weight,
            attn.indexer.compressor.wkv.weight,
            attn.indexer.compressor.wgate.weight,
            attn.indexer.compressor.ape,
            attn.indexer.compressor.norm.weight,
        )
    else:
        indexer_args = (None,) * 6
    return (
        q,
        kv,
        x,
        qr,
        cos,
        sin,
        cos_anchor,
        sin_anchor,
        attn.attn_sink,
        attn.compressor.wkv.weight,
        attn.compressor.wgate.weight,
        attn.compressor.ape,
        attn.compressor.norm.weight,
        *indexer_args,
    )


def _pack_op_constants(cfg: DeepseekV4Config, attn: DeepseekV4Attention) -> tuple:
    index_n_heads = attn.indexer.index_n_heads if attn.indexer is not None else 0
    index_head_dim = attn.indexer.index_head_dim if attn.indexer is not None else 0
    index_topk = attn.indexer.index_topk if attn.indexer is not None else 0
    return (
        attn.softmax_scale,
        attn.window_size,
        attn.compress_ratio,
        attn.rope_head_dim,
        attn.head_dim,
        index_n_heads,
        index_head_dim,
        index_topk,
        cfg.rms_norm_eps,
    )


def _prepare_qkv(
    attn: DeepseekV4Attention,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    """Compute (q, kv, qr) with RoPE applied, mirroring DeepseekV4Attention.forward."""
    B, S, _ = x.shape
    qr = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr).view(B, S, attn.n_heads, attn.head_dim)
    q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + attn.rms_eps).to(q.dtype)
    rd = attn.rope_head_dim
    q_nope, q_pe = torch.split(q, [attn.head_dim - rd, rd], dim=-1)
    q_pe = _apply_interleaved_rope(q_pe, cos.unsqueeze(2), sin.unsqueeze(2))
    q = torch.cat([q_nope, q_pe], dim=-1)
    kv = attn.kv_norm(attn.wkv(x)).view(B, S, 1, attn.head_dim)
    kv_nope, kv_pe = torch.split(kv, [attn.head_dim - rd, rd], dim=-1)
    kv_pe = _apply_interleaved_rope(kv_pe, cos.unsqueeze(2), sin.unsqueeze(2))
    kv_nope = _fake_fp8_act_quant(kv_nope, block_size=64)
    kv = torch.cat([kv_nope, kv_pe], dim=-1)
    return q, kv, qr


def _make_caches(cfg: DeepseekV4Config, attn: DeepseekV4Attention, max_batch: int = 1):
    coff = 1 + int(attn.compress_ratio == 4)
    max_seq = cfg.max_position_embeddings
    window = attn.window_size
    head_dim = attn.head_dim
    window_cache = torch.zeros(max_batch, window, head_dim)
    compressed_kv_cache = torch.zeros(max_batch, max_seq, head_dim)
    compressor_kv_state = torch.zeros(
        max_batch, coff * attn.compress_ratio, coff * head_dim, dtype=torch.float32
    )
    compressor_score_state = torch.full(
        (max_batch, coff * attn.compress_ratio, coff * head_dim),
        -1.0e20,
        dtype=torch.float32,
    )
    if attn.indexer is not None:
        idx_d = attn.indexer.index_head_dim
        indexer_compressed_kv_cache = torch.zeros(max_batch, max_seq, idx_d)
        indexer_kv_state = torch.zeros(
            max_batch, coff * attn.compress_ratio, coff * idx_d, dtype=torch.float32
        )
        indexer_score_state = torch.full(
            (max_batch, coff * attn.compress_ratio, coff * idx_d),
            -1.0e20,
            dtype=torch.float32,
        )
    else:
        indexer_compressed_kv_cache = torch.zeros(max_batch, 0, 0)
        indexer_kv_state = torch.zeros(max_batch, 0, 0, dtype=torch.float32)
        indexer_score_state = torch.zeros(max_batch, 0, 0, dtype=torch.float32)
    return (
        window_cache,
        compressed_kv_cache,
        compressor_kv_state,
        compressor_score_state,
        indexer_compressed_kv_cache,
        indexer_kv_state,
        indexer_score_state,
    )


@pytest.mark.parametrize("ratio", [4, 128])
def test_cached_prefill_matches_stateless_prefill(ratio: int) -> None:
    """Running the cached op in prefill mode should match the stateless prefill op."""
    cfg = _small_cfg(ratio)
    attn = DeepseekV4Attention(cfg, layer_idx=0).eval()
    attn.compressor.ape.data.normal_(std=0.02)
    if attn.indexer is not None:
        attn.indexer.compressor.ape.data.normal_(std=0.02)
    attn.attn_sink.data.normal_(std=0.1)

    B, S = (1, 8) if ratio == 4 else (1, 256)
    x = torch.randn(B, S, cfg.hidden_size)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos_tbl, sin_tbl = _build_freqs_cis(
        cfg.qk_rope_head_dim,
        cfg.max_position_embeddings,
        cfg.compress_rope_theta,
        0,
        1.0,
        32,
        1,
    )
    cos = cos_tbl[position_ids]
    sin = sin_tbl[position_ids]
    q, kv, qr = _prepare_qkv(attn, x, cos, sin)

    # Stateless op — reference.
    qkv_args = _pack_op_qkv_args(attn, q, kv, x, qr, cos, sin, cos_tbl, sin_tbl)
    constants = _pack_op_constants(cfg, attn)
    y_stateless = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn(
        *qkv_args, *constants, 0, "mha_sparse"
    )

    # Cached op in prefill mode (s > 1).
    caches = _make_caches(cfg, attn)
    meta = (
        torch.zeros(1, dtype=torch.int64),  # batch_info_host (unused placeholder)
        torch.tensor([S], dtype=torch.int32),  # seq_len
        torch.tensor([0], dtype=torch.int32),  # input_pos
        torch.tensor([0], dtype=torch.int32),  # slot_idx
        torch.tensor([0, S], dtype=torch.int32),  # cu_seqlen
    )
    y_cached = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn_with_cache(
        *qkv_args, *meta, *caches, *constants
    )
    torch.testing.assert_close(y_stateless, y_cached, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("ratio", [4, 128])
def test_decode_matches_fresh_prefill_last_token(ratio: int) -> None:
    """Decoding M tokens via the cached op matches a fresh prefill of length N+M.

    After a prefill of length N, decoding M tokens via the cached op should
    produce the same final-token attention output as a fresh prefill of length N+M.
    """
    cfg = _small_cfg(ratio)
    attn = DeepseekV4Attention(cfg, layer_idx=0).eval()
    attn.compressor.ape.data.normal_(std=0.02)
    if attn.indexer is not None:
        attn.indexer.compressor.ape.data.normal_(std=0.02)
    attn.attn_sink.data.normal_(std=0.1)

    B = 1
    prefill_len, decode_steps = (5, 3) if ratio == 4 else (130, 126)
    total_len = prefill_len + decode_steps
    x_full = torch.randn(B, total_len, cfg.hidden_size)
    position_ids_full = torch.arange(total_len).unsqueeze(0).expand(B, -1)
    cos_tbl, sin_tbl = _build_freqs_cis(
        cfg.qk_rope_head_dim,
        cfg.max_position_embeddings,
        cfg.compress_rope_theta,
        0,
        1.0,
        32,
        1,
    )
    cos_full = cos_tbl[position_ids_full]
    sin_full = sin_tbl[position_ids_full]
    q_full, kv_full, qr_full = _prepare_qkv(attn, x_full, cos_full, sin_full)

    # Reference: single prefill of length total_len.
    constants = _pack_op_constants(cfg, attn)
    qkv_ref = _pack_op_qkv_args(
        attn, q_full, kv_full, x_full, qr_full, cos_full, sin_full, cos_tbl, sin_tbl
    )
    y_ref_full = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn(
        *qkv_ref, *constants, 0, "mha_sparse"
    )

    # Step 1: seed caches with a prefill of length prefill_len.
    x_prefill = x_full[:, :prefill_len]
    q_prefill = q_full[:, :prefill_len]
    kv_prefill = kv_full[:, :prefill_len]
    qr_prefill = qr_full[:, :prefill_len]
    cos_prefill = cos_full[:, :prefill_len]
    sin_prefill = sin_full[:, :prefill_len]
    caches = _make_caches(cfg, attn)
    meta_prefill = (
        torch.zeros(1, dtype=torch.int64),
        torch.tensor([prefill_len], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, prefill_len], dtype=torch.int32),
    )
    qkv_prefill = _pack_op_qkv_args(
        attn,
        q_prefill,
        kv_prefill,
        x_prefill,
        qr_prefill,
        cos_prefill,
        sin_prefill,
        cos_tbl,
        sin_tbl,
    )
    torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn_with_cache(
        *qkv_prefill, *meta_prefill, *caches, *constants
    )
    # Decode must ignore unpopulated rows in overallocated compressed caches.
    # E2E runs can allocate these caches to max_seq_len; attending over the
    # whole allocation is both incorrect and can explode memory.
    initial_num_compressed = prefill_len // ratio
    compressed_kv_cache = caches[1]
    compressed_kv_cache[:, initial_num_compressed:] = torch.nan
    if attn.indexer is not None:
        indexer_compressed_kv_cache = caches[4]
        indexer_compressed_kv_cache[:, initial_num_compressed:] = torch.nan

    # Step 2: decode M tokens one at a time.
    y_last_decoded = None
    for step in range(decode_steps):
        pos = prefill_len + step
        x_step = x_full[:, pos : pos + 1]
        q_step = q_full[:, pos : pos + 1]
        kv_step = kv_full[:, pos : pos + 1]
        qr_step = qr_full[:, pos : pos + 1]
        cos_step = cos_full[:, pos : pos + 1]
        sin_step = sin_full[:, pos : pos + 1]
        meta_step = (
            torch.zeros(1, dtype=torch.int64),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([pos], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
        )
        qkv_step = _pack_op_qkv_args(
            attn, q_step, kv_step, x_step, qr_step, cos_step, sin_step, cos_tbl, sin_tbl
        )
        y_last_decoded = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn_with_cache(
            *qkv_step, *meta_step, *caches, *constants
        )

    # Compare the decoded last-token attention output to the reference prefill's last token.
    assert y_last_decoded is not None
    y_ref_last = y_ref_full[:, -1:]
    # Tolerances are a bit loose because the decode path uses a different computation order
    # (rolling-state updates vs one-shot softmax) which introduces bfloat16-scale drift.
    torch.testing.assert_close(y_ref_last, y_last_decoded, rtol=5e-2, atol=5e-2)
