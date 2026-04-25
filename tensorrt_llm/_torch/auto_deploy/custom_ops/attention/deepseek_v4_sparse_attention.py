# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""DeepSeek V4 sparse attention custom op (prefill + cached decode).

Two ops live here:

* ``torch_deepseek_v4_sparse_attn`` — the prefill-only form that wraps the
  ``compress_ratio != 0`` attention path of ``modeling_deepseek_v4.py`` as a
  single FX node. The AD KV-cache transform pipeline recognises this op and
  replaces each node with the cached variant below.
* ``torch_deepseek_v4_sparse_attn_with_cache`` — the decode-capable variant
  that maintains (per sequence slot):
    - ``window_cache`` — ring buffer for the last ``window_size`` KV tokens
    - ``compressed_kv_cache`` — append-only store of compressed KV tokens
    - ``compressor_kv_state`` / ``compressor_score_state`` — the HF
      reference's rolling ``kv_state`` / ``score_state`` (fp32)
    - three parallel caches for the Indexer's own compressor when the layer
      uses an Indexer (compress_ratio == 4).
  Non-indexer layers pass zero-sized stubs for the indexer caches so the op
  signature is fixed across all sparse layers.

Both ops are stateless in Python — weight tensors and caches are passed in
as arguments. Keeping the Compressor / Indexer submodules as ``nn.Module``
children on the parent ``DeepseekV4Attention`` preserves checkpoint key
names while exposing their weights as graph placeholders after
``torch.export``.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    MHACallable,
    ResourceHandlerDict,
    StateResourceHandler,
    UnpagedResourceHandler,
)

# ----------------------------------------------------------------------------
# Private helpers — ported verbatim from modeling_deepseek_v4.py
# ----------------------------------------------------------------------------


def _apply_interleaved_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Interleaved (complex-style) RoPE on the last dim of ``x`` in pure real ops.

    ``x``: ``[..., rope_head_dim]`` with channel pairs ``[a0, b0, a1, b1, ...]``.
    ``cos``/``sin``: must broadcast against ``x[..., ::2]`` / ``x[..., 1::2]``.
    ``inverse=True`` negates sin, i.e. multiplies by the complex conjugate.
    """
    if inverse:
        sin = -sin
    a = x[..., 0::2]
    b = x[..., 1::2]
    out_a = a * cos - b * sin
    out_b = a * sin + b * cos
    return torch.stack([out_a, out_b], dim=-1).flatten(-2).type_as(x)


def _ceil_pow2_scale(amax: torch.Tensor, max_value: float, min_value: float) -> torch.Tensor:
    amax = amax.clamp_min(min_value)
    return torch.pow(2.0, torch.ceil(torch.log2(amax / max_value)))


def _fake_fp8_act_quant(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Reference in-place ``act_quant`` approximation for DeepSeek V4 KV tensors."""
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x

    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(*x_float.shape[:-1], dim // block_size, block_size)
    scale = _ceil_pow2_scale(grouped.abs().amax(dim=-1, keepdim=True), 448.0, 1.0e-4)
    quant = torch.clamp(grouped / scale, -448.0, 448.0).to(dtype).float()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _fake_fp4_act_quant(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Reference in-place ``fp4_act_quant`` approximation for DeepSeek V4 indexer tensors."""
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x

    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(*x_float.shape[:-1], dim // block_size, block_size)
    scale = _ceil_pow2_scale(grouped.abs().amax(dim=-1, keepdim=True), 6.0, 6.0 * 2.0**-126)
    normalized = torch.clamp(grouped / scale, -6.0, 6.0)
    abs_normalized = normalized.abs()
    quant_abs = torch.where(abs_normalized > 0.25, 0.5, torch.zeros_like(abs_normalized))
    quant_abs = torch.where(abs_normalized > 0.75, 1.0, quant_abs)
    quant_abs = torch.where(abs_normalized > 1.25, 1.5, quant_abs)
    quant_abs = torch.where(abs_normalized > 1.75, 2.0, quant_abs)
    quant_abs = torch.where(abs_normalized > 2.5, 3.0, quant_abs)
    quant_abs = torch.where(abs_normalized > 3.5, 4.0, quant_abs)
    quant_abs = torch.where(abs_normalized > 5.0, 6.0, quant_abs)
    quant = quant_abs * normalized.sign()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """Hadamard rotation used before FP4 simulation in the DeepSeek V4 indexer."""
    dim = x.shape[-1]
    if dim <= 1:
        return x
    if dim & (dim - 1):
        raise ValueError(f"Hadamard rotation requires power-of-two dimension, got {dim}.")

    orig_shape = x.shape
    y = x.float()
    width = 1
    while width < dim:
        y = y.reshape(*y.shape[:-1], dim // (2 * width), 2, width)
        left = y[..., 0, :]
        right = y[..., 1, :]
        y = torch.cat([left + right, left - right], dim=-1).reshape(orig_shape)
        width *= 2
    return (y * (dim**-0.5)).to(x.dtype)


def _overlap_transform(tensor: torch.Tensor, head_dim: int, value: float = 0.0) -> torch.Tensor:
    """HF DeepSeek V4 overlap transform used by ratio-4 compression."""
    bsz, _, ratio, _ = tensor.shape
    previous = tensor[:, :, :, :head_dim]
    current = tensor[:, :, :, head_dim:]
    prefix = tensor.new_full((bsz, 1, ratio, head_dim), value)
    previous = torch.cat([prefix, previous[:, :-1]], dim=1)
    return torch.cat([previous, current], dim=2)


def _window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Prefill (``start_pos == 0``) version of HF ``get_window_topk_idxs``.

    The window axis is kept at static ``window_size`` for export. Entries
    outside the real causal window are marked ``-1`` and ignored downstream.
    """
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    matrix = base - window_size + 1 + torch.arange(window_size, device=device)
    matrix = torch.where((matrix < 0) | (matrix > base), -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def _compress_topk_idxs(
    ratio: int, bsz: int, seqlen: int, offset: int, device: torch.device
) -> torch.Tensor:
    """Prefill version of HF ``get_compress_topk_idxs``.

    Padded to ``seqlen`` so the dynamic export shape doesn't specialise on
    ``seqlen // ratio``.
    """
    matrix = torch.arange(seqlen, device=device).repeat(seqlen, 1)
    valid_count = torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
    mask = matrix >= valid_count
    matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def _build_sparse_attn_mask(topk_idxs: torch.Tensor, total_kv_len: int) -> torch.Tensor:
    """Convert top-k index lists ``[B, S, K]`` into additive attention mask.

    Returns ``[B, 1, S, total_kv_len]`` with 0 for selected positions and
    ``-inf`` elsewhere. ``-1`` entries in ``topk_idxs`` are treated as invalid.
    """
    valid = topk_idxs >= 0
    kv_positions = torch.arange(total_kv_len, device=topk_idxs.device)
    selected = topk_idxs.unsqueeze(-1) == kv_positions.view(1, 1, 1, -1)
    selected = (selected & valid.unsqueeze(-1)).any(dim=2)
    mask = topk_idxs.new_zeros(selected.shape, dtype=torch.float32)
    mask = mask.masked_fill(~selected, -10000.0)
    return mask.unsqueeze(1)


def _ratio_chunk_indices(
    seqlen: int, ratio: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return padded ratio chunk token indices and their validity mask."""
    chunk_ids = torch.arange(seqlen, device=device).unsqueeze(1)
    token_offsets = torch.arange(ratio, device=device)
    indices = chunk_ids * ratio + token_offsets
    valid = indices < seqlen
    indices = torch.where(valid, indices, torch.zeros_like(indices))
    return indices, valid


def _manual_attention_with_sinks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Reference-style attention with learnable sinks used by compressed sparse layers.

    q: ``[B, S, H, D]``; k, v: ``[B, S_kv, 1, D]`` (broadcast over heads). Returns ``[B, S, H, D]``.
    """
    bsz, seqlen, n_heads, head_dim = q.shape
    kv_len = k.shape[1]
    q_bh = q.transpose(1, 2).float()
    k_bh = k.expand(bsz, kv_len, n_heads, head_dim).transpose(1, 2).float()
    v_bh = v.expand(bsz, kv_len, n_heads, head_dim).transpose(1, 2).float()
    scores = torch.matmul(q_bh, k_bh.transpose(-2, -1)) * scale
    scores = scores + attn_mask.float()

    sink_logits = sinks.float().reshape(1, n_heads, 1, 1).expand(bsz, n_heads, seqlen, 1)
    logits_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink_logits)
    exp_scores = torch.exp(scores - logits_max)
    exp_sinks = torch.exp(sink_logits - logits_max)
    attn = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + exp_sinks)
    out = torch.matmul(attn, v_bh)
    return out.transpose(1, 2).contiguous().to(q.dtype)


# ----------------------------------------------------------------------------
# Stateless prefill building blocks: compressor + indexer
# ----------------------------------------------------------------------------


def _sparse_compressor(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wkv: torch.Tensor,  # [coff*head_dim, hidden]
    wgate: torch.Tensor,  # [coff*head_dim, hidden]
    ape: torch.Tensor,  # [ratio, coff*head_dim]
    norm_weight: torch.Tensor,  # [head_dim]
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    rms_norm_eps: float,
    rotate: bool = False,
) -> torch.Tensor:
    """Stateless equivalent of DeepseekV4Compressor.forward (prefill only)."""
    bsz, seqlen, _ = hidden_states.shape
    overlap = compress_ratio == 4
    chunk_indices, chunk_valid = _ratio_chunk_indices(seqlen, compress_ratio, hidden_states.device)
    flat_indices = chunk_indices.reshape(-1)

    kv_all = F.linear(hidden_states, wkv).float()
    score_all = F.linear(hidden_states, wgate).float()
    kv = kv_all[:, flat_indices].view(bsz, seqlen, compress_ratio, -1)
    score = score_all[:, flat_indices].view(bsz, seqlen, compress_ratio, -1) + ape
    score = torch.where(
        chunk_valid.view(1, seqlen, compress_ratio, 1),
        score,
        score.new_full((), -1.0e20),
    )
    if overlap:
        kv = _overlap_transform(kv, head_dim, 0.0)
        score = _overlap_transform(score, head_dim, -1.0e20)

    # Chunks whose scores are all ``-1e20`` (chunk index >= ceil(seqlen / ratio)
    # for non-overlap, or the first chunk in overlap with no previous data) produce
    # ``exp(-1e20) / sum(exp(-1e20)) = 0/0 = NaN`` from softmax. Detect rows that
    # would be all-masked and substitute a safe weight vector so the rest of the
    # op doesn't propagate NaN into ``kv_all`` downstream. Invalid rows of the
    # output are filtered out anyway via the attention mask's top-k indices.
    score_max = score.amax(dim=2, keepdim=True)
    row_all_masked = score_max <= -1.0e19  # true when the whole row is -1e20
    score = torch.where(row_all_masked, torch.zeros_like(score), score)
    compressed = (kv * score.softmax(dim=2)).sum(dim=2)
    compressed = torch.ops.auto_deploy.torch_rmsnorm(
        compressed.to(hidden_states.dtype), norm_weight, rms_norm_eps
    ).to(hidden_states.dtype)

    chunk_start = torch.arange(seqlen, device=hidden_states.device) * compress_ratio
    chunk_start = torch.where(chunk_start < seqlen, chunk_start, torch.zeros_like(chunk_start))
    cos_comp = cos[:, chunk_start]
    sin_comp = sin[:, chunk_start]
    nope, pe = torch.split(compressed, [head_dim - rope_head_dim, rope_head_dim], dim=-1)
    pe = _apply_interleaved_rope(pe, cos_comp, sin_comp)
    compressed = torch.cat([nope, pe], dim=-1)
    if rotate:
        return _fake_fp4_act_quant(_hadamard_rotate(compressed), block_size=32)

    nope, pe = torch.split(compressed, [head_dim - rope_head_dim, rope_head_dim], dim=-1)
    nope = _fake_fp8_act_quant(nope, block_size=64)
    return torch.cat([nope, pe], dim=-1)


def _sparse_indexer(
    hidden_states: torch.Tensor,
    q_lora: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int,
    wq_b: torch.Tensor,  # [index_n_heads*index_head_dim, q_lora_rank]
    weights_proj: torch.Tensor,  # [index_n_heads, hidden]
    compressor_wkv: torch.Tensor,
    compressor_wgate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    *,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope_head_dim: int,
    compress_ratio: int,
    rms_norm_eps: float,
) -> torch.Tensor:
    """Stateless equivalent of DeepseekV4Indexer.forward (prefill only)."""
    bsz, seqlen, _ = hidden_states.shape
    softmax_scale = index_head_dim**-0.5
    rd = rope_head_dim

    q = F.linear(q_lora, wq_b).view(bsz, seqlen, index_n_heads, index_head_dim)
    q_nope, q_pe = torch.split(q, [index_head_dim - rd, rd], dim=-1)
    q_pe = _apply_interleaved_rope(q_pe, cos.unsqueeze(2), sin.unsqueeze(2))
    q = torch.cat([q_nope, q_pe], dim=-1)
    q = _fake_fp4_act_quant(_hadamard_rotate(q), block_size=32)

    index_k = _sparse_compressor(
        hidden_states,
        cos,
        sin,
        compressor_wkv,
        compressor_wgate,
        compressor_ape,
        compressor_norm_weight,
        head_dim=index_head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        rms_norm_eps=rms_norm_eps,
        rotate=True,
    )
    num_compressed = seqlen // compress_ratio
    if index_topk == 0:
        return torch.empty(bsz, seqlen, 0, device=hidden_states.device, dtype=torch.int64)
    if num_compressed == 0:
        return torch.full(
            (bsz, seqlen, index_topk), -1, device=hidden_states.device, dtype=torch.int64
        )
    index_k = index_k[:, :num_compressed]
    weights = F.linear(hidden_states, weights_proj).float() * (softmax_scale * index_n_heads**-0.5)

    index_score = torch.einsum("bshd,btd->bsht", q, index_k).float()
    index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)

    compressed_positions = torch.arange(num_compressed, device=hidden_states.device)
    valid_count = (
        torch.arange(1, seqlen + 1, device=hidden_states.device).unsqueeze(1) // compress_ratio
    )
    future_mask = compressed_positions.unsqueeze(0) >= valid_count
    index_score = index_score.masked_fill(future_mask.unsqueeze(0), -1.0e20)

    k = min(index_topk, num_compressed)
    topk_idxs = index_score.topk(k, dim=-1).indices
    invalid = topk_idxs >= valid_count.unsqueeze(0)
    topk_idxs = torch.where(invalid, -1, topk_idxs + offset)
    if k < index_topk:
        pad = torch.full(
            (bsz, seqlen, index_topk - k),
            -1,
            device=hidden_states.device,
            dtype=topk_idxs.dtype,
        )
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1)
    return topk_idxs.to(torch.int64)


def _sparse_attn_prefill_body(
    q: torch.Tensor,
    kv: torch.Tensor,
    hidden_states: torch.Tensor,
    q_lora: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_anchor: torch.Tensor,
    sin_anchor: torch.Tensor,
    attn_sink: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_wgate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    indexer_wq_b: Optional[torch.Tensor],
    indexer_weights_proj: Optional[torch.Tensor],
    indexer_compressor_wkv: Optional[torch.Tensor],
    indexer_compressor_wgate: Optional[torch.Tensor],
    indexer_compressor_ape: Optional[torch.Tensor],
    indexer_compressor_norm_weight: Optional[torch.Tensor],
    *,
    scale: float,
    window_size: int,
    compress_ratio: int,
    rope_head_dim: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rms_norm_eps: float,
) -> torch.Tensor:
    """Shared prefill attention body reused by the exported op and equivalence tests."""
    bsz, seqlen, _, _ = q.shape

    kv_compress = _sparse_compressor(
        hidden_states,
        cos,
        sin,
        compressor_wkv,
        compressor_wgate,
        compressor_ape,
        compressor_norm_weight,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        rms_norm_eps=rms_norm_eps,
    )

    if indexer_wq_b is not None:
        compress_topk = _sparse_indexer(
            hidden_states,
            q_lora,
            cos,
            sin,
            offset=seqlen,
            wq_b=indexer_wq_b,
            weights_proj=indexer_weights_proj,
            compressor_wkv=indexer_compressor_wkv,
            compressor_wgate=indexer_compressor_wgate,
            compressor_ape=indexer_compressor_ape,
            compressor_norm_weight=indexer_compressor_norm_weight,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            rms_norm_eps=rms_norm_eps,
        )
    else:
        compress_topk = _compress_topk_idxs(
            compress_ratio, bsz, seqlen, seqlen, hidden_states.device
        )

    window_topk = _window_topk_idxs(window_size, bsz, seqlen, hidden_states.device)
    topk_idxs = torch.cat([window_topk, compress_topk], dim=-1)
    kv_all = torch.cat([kv, kv_compress.view(bsz, seqlen, 1, head_dim)], dim=1)
    attn_mask = _build_sparse_attn_mask(topk_idxs, kv_all.shape[1]).to(q.dtype)
    return _manual_attention_with_sinks(q, kv_all, kv_all, attn_mask, scale, attn_sink)


# ----------------------------------------------------------------------------
# Custom op registration (prefill)
# ----------------------------------------------------------------------------


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_sparse_attn", mutates_args=())
def torch_deepseek_v4_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    hidden_states: torch.Tensor,
    q_lora: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_anchor: torch.Tensor,
    sin_anchor: torch.Tensor,
    attn_sink: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_wgate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    indexer_wq_b: Optional[torch.Tensor],
    indexer_weights_proj: Optional[torch.Tensor],
    indexer_compressor_wkv: Optional[torch.Tensor],
    indexer_compressor_wgate: Optional[torch.Tensor],
    indexer_compressor_ape: Optional[torch.Tensor],
    indexer_compressor_norm_weight: Optional[torch.Tensor],
    scale: float,
    window_size: int,
    compress_ratio: int,
    rope_head_dim: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rms_norm_eps: float,
    layer_idx: Optional[int] = None,
    layer_type: str = "mha_sparse",
) -> torch.Tensor:
    """DeepSeek V4 prefill-only sparse (compress_ratio != 0) attention.

    Layout is ``bsnd``. Returns ``[B, S, H, head_dim]`` — the rotary
    inversion on the output happens outside this op to keep the op
    boundary clean.
    """
    del layer_idx, layer_type
    return _sparse_attn_prefill_body(
        q,
        kv,
        hidden_states,
        q_lora,
        cos,
        sin,
        cos_anchor,
        sin_anchor,
        attn_sink,
        compressor_wkv,
        compressor_wgate,
        compressor_ape,
        compressor_norm_weight,
        indexer_wq_b,
        indexer_weights_proj,
        indexer_compressor_wkv,
        indexer_compressor_wgate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        scale=scale,
        window_size=window_size,
        compress_ratio=compress_ratio,
        rope_head_dim=rope_head_dim,
        head_dim=head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        rms_norm_eps=rms_norm_eps,
    )


@torch_deepseek_v4_sparse_attn.register_fake
def torch_deepseek_v4_sparse_attn_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    hidden_states: torch.Tensor,
    q_lora: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_anchor: torch.Tensor,
    sin_anchor: torch.Tensor,
    attn_sink: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_wgate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    indexer_wq_b: Optional[torch.Tensor],
    indexer_weights_proj: Optional[torch.Tensor],
    indexer_compressor_wkv: Optional[torch.Tensor],
    indexer_compressor_wgate: Optional[torch.Tensor],
    indexer_compressor_ape: Optional[torch.Tensor],
    indexer_compressor_norm_weight: Optional[torch.Tensor],
    scale: float,
    window_size: int,
    compress_ratio: int,
    rope_head_dim: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rms_norm_eps: float,
    layer_idx: Optional[int] = None,
    layer_type: str = "mha_sparse",
) -> torch.Tensor:
    return torch.empty_like(q)


# ----------------------------------------------------------------------------
# Decode-capable cached variant
# ----------------------------------------------------------------------------


def _seed_compressor_state(
    hidden_states: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    ape: torch.Tensor,
    slot: int,
    compress_ratio: int,
) -> None:
    """Seed the Compressor's rolling state buffers with the last chunk of a prefill.

    Mirrors the HF reference's ``start_pos == 0`` branch in ``Compressor.forward``:
    when the prefill length is not a multiple of ``ratio``, the leftover tokens
    initialise the rolling buffer so subsequent decode steps continue from the
    correct position inside the ratio window.
    """
    bsz, seqlen, _ = hidden_states.shape
    assert bsz == 1, "Cached op currently supports single-batch inference."
    overlap = compress_ratio == 4
    remainder = seqlen % compress_ratio
    cutoff = seqlen - remainder
    offset = compress_ratio if overlap else 0

    kv_all = F.linear(hidden_states, wkv).float()
    score_all = F.linear(hidden_states, wgate).float()

    # Overlap window: stash the last full chunk of kv/score as the 'previous' slots.
    if overlap and cutoff >= compress_ratio:
        kv_state[slot, :compress_ratio] = kv_all[0, cutoff - compress_ratio : cutoff]
        # score for the previous chunk does not yet get the ape added — matches HF reference.
        score_state[slot, :compress_ratio] = score_all[0, cutoff - compress_ratio : cutoff] + ape
    else:
        # No previous overlap yet; zero out the previous-window slots.
        if overlap:
            kv_state[slot, :compress_ratio].zero_()
            score_state[slot, :compress_ratio].fill_(-1.0e20)

    # Remainder (partial-chunk) tokens go into the current-window slots.
    if overlap:
        # Fully reset the current window.
        kv_state[slot, compress_ratio:].zero_()
        score_state[slot, compress_ratio:].fill_(-1.0e20)
    else:
        kv_state[slot].zero_()
        score_state[slot].fill_(-1.0e20)
    if remainder > 0:
        kv_state[slot, offset : offset + remainder] = kv_all[0, cutoff:]
        score_state[slot, offset : offset + remainder] = score_all[0, cutoff:] + ape[:remainder]


def _decode_compressor_step(
    hidden_states: torch.Tensor,  # [1, 1, hidden]
    cos_anchor_table: torch.Tensor,  # [max_seq_len, rd/2]
    sin_anchor_table: torch.Tensor,  # [max_seq_len, rd/2]
    kv_state: torch.Tensor,  # [max_batch, coff*ratio, coff*head_dim]
    score_state: torch.Tensor,  # [max_batch, coff*ratio, coff*head_dim]
    compressed_kv_cache: torch.Tensor,  # [max_batch, max_compressed, head_dim]
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    slot: int,
    input_pos: int,
    compress_ratio: int,
    head_dim: int,
    rope_head_dim: int,
    rms_norm_eps: float,
    rotate: bool = False,
) -> None:
    """One decode step's Compressor update. Writes to state + (maybe) compressed_kv_cache.

    Note: ``kv_state`` / ``score_state`` may be allocated with a larger axis-1
    (e.g. ``max_seq_len``) than the logical rolling-state size. We only use the
    first ``coff * compress_ratio`` slots and must index with explicit bounds.
    """
    overlap = compress_ratio == 4
    used_slots = (2 if overlap else 1) * compress_ratio  # logical state size
    kv = F.linear(hidden_states, wkv).float()  # [1, 1, coff*head_dim]
    score = F.linear(hidden_states, wgate).float() + ape[input_pos % compress_ratio]

    if overlap:
        slot_idx_in_state = compress_ratio + (input_pos % compress_ratio)
    else:
        slot_idx_in_state = input_pos % compress_ratio
    kv_state[slot, slot_idx_in_state] = kv.squeeze(0).squeeze(0)
    score_state[slot, slot_idx_in_state] = score.squeeze(0).squeeze(0)

    should_compress = (input_pos + 1) % compress_ratio == 0
    if not should_compress:
        return

    if overlap:
        kv_full = torch.cat(
            [
                kv_state[slot, :compress_ratio, :head_dim],
                kv_state[slot, compress_ratio:used_slots, head_dim:],
            ],
            dim=0,
        )
        score_full = torch.cat(
            [
                score_state[slot, :compress_ratio, :head_dim],
                score_state[slot, compress_ratio:used_slots, head_dim:],
            ],
            dim=0,
        )
        # [2*ratio, head_dim] softmax along axis 0 -> weight per slot.
        compressed = (kv_full * score_full.softmax(dim=0)).sum(dim=0)
        # Rotate rolling state: previous <- current, reset current.
        kv_state[slot, :compress_ratio] = kv_state[slot, compress_ratio:used_slots]
        score_state[slot, :compress_ratio] = score_state[slot, compress_ratio:used_slots]
        kv_state[slot, compress_ratio:used_slots].zero_()
        score_state[slot, compress_ratio:used_slots].fill_(-1.0e20)
    else:
        compressed = (
            kv_state[slot, :used_slots] * score_state[slot, :used_slots].softmax(dim=0)
        ).sum(dim=0)
        kv_state[slot, :used_slots].zero_()
        score_state[slot, :used_slots].fill_(-1.0e20)

    compressed = compressed.view(1, 1, head_dim).to(hidden_states.dtype)
    compressed = torch.ops.auto_deploy.torch_rmsnorm(compressed, norm_weight, rms_norm_eps).to(
        hidden_states.dtype
    )
    nope, pe = torch.split(compressed, [head_dim - rope_head_dim, rope_head_dim], dim=-1)
    anchor_pos = input_pos + 1 - compress_ratio
    cos_anchor = cos_anchor_table[anchor_pos].view(1, 1, -1)
    sin_anchor = sin_anchor_table[anchor_pos].view(1, 1, -1)
    pe = _apply_interleaved_rope(pe, cos_anchor, sin_anchor)
    compressed = torch.cat([nope, pe], dim=-1)
    if rotate:
        compressed = _fake_fp4_act_quant(_hadamard_rotate(compressed), block_size=32)
    else:
        nope, pe = torch.split(compressed, [head_dim - rope_head_dim, rope_head_dim], dim=-1)
        nope = _fake_fp8_act_quant(nope, block_size=64)
        compressed = torch.cat([nope, pe], dim=-1)

    write_idx = input_pos // compress_ratio
    compressed_kv_cache[slot, write_idx] = compressed.squeeze(0).squeeze(0)


def _decode_indexer_step(
    q_lora: torch.Tensor,
    cos_last: torch.Tensor,
    sin_last: torch.Tensor,
    wq_b: torch.Tensor,
    weights_proj: torch.Tensor,
    indexer_compressed_kv_cache: torch.Tensor,
    slot: int,
    input_pos: int,
    hidden_states: torch.Tensor,
    compress_ratio: int,
    index_n_heads: int,
    index_head_dim: int,
    rope_head_dim: int,
    index_topk: int,
    offset: int,
) -> torch.Tensor:
    """Compute compressed-KV top-k indices at decode for one token."""
    if index_topk == 0:
        return torch.empty(1, 1, 0, device=hidden_states.device, dtype=torch.int64)
    num_compressed = (input_pos + 1) // compress_ratio
    if num_compressed == 0:
        # No compressed tokens yet — return empty-by-padding (all -1) matching prefill output.
        return torch.full((1, 1, index_topk), -1, device=hidden_states.device, dtype=torch.int64)
    softmax_scale = index_head_dim**-0.5
    q = F.linear(q_lora, wq_b).view(1, 1, index_n_heads, index_head_dim)
    rd = rope_head_dim
    q_nope, q_pe = torch.split(q, [index_head_dim - rd, rd], dim=-1)
    q_pe = _apply_interleaved_rope(q_pe, cos_last.unsqueeze(2), sin_last.unsqueeze(2))
    q = torch.cat([q_nope, q_pe], dim=-1)
    q = _fake_fp4_act_quant(_hadamard_rotate(q), block_size=32)

    weights = F.linear(hidden_states, weights_proj).float() * (
        softmax_scale * index_n_heads**-0.5
    )  # [1, 1, index_n_heads]
    index_k = indexer_compressed_kv_cache[
        slot : slot + 1, :num_compressed
    ]  # [1, num_compressed, index_head_dim]
    index_score = torch.einsum("bshd,btd->bsht", q, index_k).float()
    index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)  # [1, 1, num_compressed]

    k = min(index_topk, num_compressed)
    topk = index_score.topk(k, dim=-1).indices  # [1, 1, k]
    topk = topk + offset
    if k < index_topk:
        pad = torch.full((1, 1, index_topk - k), -1, device=hidden_states.device, dtype=topk.dtype)
        topk = torch.cat([topk, pad], dim=-1)
    return topk.to(torch.int64)


def _seed_indexer_compressed_cache(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    indexer_compressor_wkv: torch.Tensor,
    indexer_compressor_wgate: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    indexer_compressed_kv_cache: torch.Tensor,
    indexer_kv_state: torch.Tensor,
    indexer_score_state: torch.Tensor,
    slot: int,
    compress_ratio: int,
    index_head_dim: int,
    rope_head_dim: int,
    rms_norm_eps: float,
) -> None:
    """Prefill-seed the Indexer's own compressed KV cache and rolling state."""
    bsz, seqlen, _ = hidden_states.shape
    # Reuse the stateless compressor on the Indexer's dimensions.
    idx_compressed = _sparse_compressor(
        hidden_states,
        cos,
        sin,
        indexer_compressor_wkv,
        indexer_compressor_wgate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        head_dim=index_head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        rms_norm_eps=rms_norm_eps,
        rotate=True,
    )  # [1, seqlen, index_head_dim]
    # Zero full slot first — UnpagedResourceHandler.allocate() uses torch.empty()
    # so unused entries contain uninitialized memory that would propagate NaN
    # into the indexer's score path at decode.
    indexer_compressed_kv_cache[slot].zero_()
    num_compressed = seqlen // compress_ratio
    if num_compressed > 0:
        indexer_compressed_kv_cache[slot, :num_compressed] = idx_compressed[0, :num_compressed]
    _seed_compressor_state(
        hidden_states,
        indexer_kv_state,
        indexer_score_state,
        indexer_compressor_wkv,
        indexer_compressor_wgate,
        indexer_compressor_ape,
        slot,
        compress_ratio,
    )


def _seed_window_cache(
    kv: torch.Tensor,
    window_cache: torch.Tensor,
    slot: int,
    window_size: int,
) -> None:
    """Seed the decode ring buffer exactly as HF ``Attention.forward(start_pos=0)``."""
    seqlen = kv.shape[1]
    window_cache[slot].zero_()
    if seqlen <= window_size:
        window_cache[slot, :seqlen] = kv[0, :seqlen, 0, :]
        return

    cutoff = seqlen % window_size
    recent = kv[0, -window_size:, 0, :]
    window_cache[slot, cutoff:window_size] = recent[: window_size - cutoff]
    window_cache[slot, :cutoff] = recent[window_size - cutoff :]


def _decode_sparse_attn_step(
    q: torch.Tensor,  # [1, 1, H, D]
    kv_new: torch.Tensor,  # [1, 1, 1, D]
    attn_sink: torch.Tensor,
    window_cache: torch.Tensor,  # [max_batch, window, D]
    compressed_kv_cache: torch.Tensor,  # [max_batch, max_compressed, D]
    compress_topk: torch.Tensor,  # [1, 1, K] — absolute indices into concat(window, compressed)
    slot: int,
    input_pos: int,
    window_size: int,
    compress_ratio: int,
    scale: float,
    head_dim: int,
) -> torch.Tensor:
    """Compute the sparse attention output for one decode step."""
    # Update window ring buffer with the current-token KV.
    window_cache[slot, input_pos % window_size] = kv_new.squeeze(0).squeeze(0).squeeze(0)
    # Build window top-k (positions of the last min(input_pos+1, window_size) tokens).
    visible_window = min(input_pos + 1, window_size)
    device = q.device
    # Window positions are logical absolute positions max(0, input_pos+1-window)..input_pos.
    # For the attention mask we index into the combined KV tensor, where window rows come first.
    # Ring buffer layout: row i in window_cache corresponds to the token written at some past position
    # where pos % window == i. Easiest is to gather the visible rows in chronological order below.
    first = input_pos + 1 - visible_window
    mod_indices = (torch.arange(visible_window, device=device) + first) % window_size
    gathered_window = window_cache[slot][mod_indices]  # [visible_window, head_dim]

    # Pad window to window_size for static shapes.
    if visible_window < window_size:
        pad = window_cache.new_zeros(window_size - visible_window, head_dim)
        gathered_window = torch.cat([gathered_window, pad], dim=0)
    # Only materialize compressed rows that are visible to this decode step.
    # ``compressed_kv_cache`` is allocated against max_seq_len, not the current
    # prefix length; using the full allocation would make attention scale with
    # max_seq_len and can allocate hundreds of GiB for long-context models.
    num_compressed = min((input_pos + 1) // compress_ratio, compressed_kv_cache.shape[1])
    visible_compressed = compressed_kv_cache[slot, :num_compressed]
    kv_all = torch.cat([gathered_window, visible_compressed], dim=0)
    # Window top-k indices are just [0..visible_window-1]; entries beyond are -1.
    window_topk = torch.full((1, 1, window_size), -1, device=device, dtype=compress_topk.dtype)
    window_topk[0, 0, :visible_window] = torch.arange(visible_window, device=device)
    topk_idxs = torch.cat([window_topk, compress_topk], dim=-1)
    attn_mask = _build_sparse_attn_mask(topk_idxs, kv_all.shape[0]).to(q.dtype)

    kv_all = kv_all.view(1, kv_all.shape[0], 1, head_dim)
    return _manual_attention_with_sinks(q, kv_all, kv_all, attn_mask, scale, attn_sink)


def _decode_compressor_step_static_cuda(
    hidden_states: torch.Tensor,
    cos_anchor_table: torch.Tensor,
    sin_anchor_table: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    slot_idx: torch.Tensor,
    input_pos: torch.Tensor,
    compress_ratio: int,
    head_dim: int,
    rope_head_dim: int,
    rms_norm_eps: float,
    rotate: bool = False,
) -> None:
    """Capture-safe B=1 decode compressor update with fixed-shape tensor work."""
    overlap = compress_ratio == 4
    used_slots = (2 if overlap else 1) * compress_ratio
    slot = slot_idx[:1].to(torch.long)
    pos = input_pos[:1].to(torch.long)
    pos_mod = torch.remainder(pos, compress_ratio)

    kv = F.linear(hidden_states, wkv).float()  # [1, 1, coff*head_dim]
    score = F.linear(hidden_states, wgate).float()
    score = score + ape.index_select(0, pos_mod).view(1, 1, -1)

    state_idx = pos_mod + (compress_ratio if overlap else 0)
    kv_state.index_put_((slot, state_idx), kv.view(1, -1), accumulate=False)
    score_state.index_put_((slot, state_idx), score.view(1, -1), accumulate=False)

    kv_slot = kv_state.index_select(0, slot)[:, :used_slots, :]
    score_slot = score_state.index_select(0, slot)[:, :used_slots, :]
    if overlap:
        kv_full = torch.cat(
            [
                kv_slot[:, :compress_ratio, :head_dim],
                kv_slot[:, compress_ratio:used_slots, head_dim:],
            ],
            dim=1,
        )
        score_full = torch.cat(
            [
                score_slot[:, :compress_ratio, :head_dim],
                score_slot[:, compress_ratio:used_slots, head_dim:],
            ],
            dim=1,
        )
    else:
        kv_full = kv_slot
        score_full = score_slot

    compressed = (kv_full * score_full.softmax(dim=1)).sum(dim=1)
    compressed = compressed.view(1, 1, head_dim).to(hidden_states.dtype)
    compressed = torch.ops.auto_deploy.torch_rmsnorm(compressed, norm_weight, rms_norm_eps).to(
        hidden_states.dtype
    )
    nope, pe = torch.split(compressed, [head_dim - rope_head_dim, rope_head_dim], dim=-1)
    anchor_pos = torch.clamp(pos + 1 - compress_ratio, min=0, max=cos_anchor_table.shape[0] - 1)
    cos_anchor = cos_anchor_table.index_select(0, anchor_pos).view(1, 1, -1)
    sin_anchor = sin_anchor_table.index_select(0, anchor_pos).view(1, 1, -1)
    pe = _apply_interleaved_rope(pe, cos_anchor, sin_anchor)
    compressed = torch.cat([nope, pe], dim=-1)
    if rotate:
        compressed = _fake_fp4_act_quant(_hadamard_rotate(compressed), block_size=32)
    else:
        nope, pe = torch.split(compressed, [head_dim - rope_head_dim, rope_head_dim], dim=-1)
        nope = _fake_fp8_act_quant(nope, block_size=64)
        compressed = torch.cat([nope, pe], dim=-1)

    should_compress = torch.eq(torch.remainder(pos + 1, compress_ratio), 0).view(1, 1, 1)
    write_idx = torch.div(pos, compress_ratio, rounding_mode="floor")
    write_idx = torch.clamp(write_idx, min=0, max=compressed_kv_cache.shape[1] - 1)
    existing_compressed = (
        compressed_kv_cache.index_select(0, slot).squeeze(0).index_select(0, write_idx)
    )
    compressed_to_store = torch.where(
        should_compress.view(1, 1), compressed.view(1, head_dim), existing_compressed
    )
    compressed_kv_cache.index_put_((slot, write_idx), compressed_to_store, accumulate=False)

    if overlap:
        reset_kv = torch.zeros_like(kv_slot[:, compress_ratio:used_slots, :])
        reset_score = torch.full_like(score_slot[:, compress_ratio:used_slots, :], -1.0e20)
        kv_after_compress = torch.cat([kv_slot[:, compress_ratio:used_slots, :], reset_kv], dim=1)
        score_after_compress = torch.cat(
            [score_slot[:, compress_ratio:used_slots, :], reset_score], dim=1
        )
    else:
        kv_after_compress = torch.zeros_like(kv_slot)
        score_after_compress = torch.full_like(score_slot, -1.0e20)

    kv_state.index_copy_(0, slot, torch.where(should_compress, kv_after_compress, kv_slot))
    score_state.index_copy_(0, slot, torch.where(should_compress, score_after_compress, score_slot))


def _decode_indexer_step_static_cuda(
    q_lora: torch.Tensor,
    cos_last: torch.Tensor,
    sin_last: torch.Tensor,
    wq_b: torch.Tensor,
    weights_proj: torch.Tensor,
    indexer_compressed_kv_cache: torch.Tensor,
    slot_idx: torch.Tensor,
    input_pos: torch.Tensor,
    hidden_states: torch.Tensor,
    compress_ratio: int,
    index_n_heads: int,
    index_head_dim: int,
    rope_head_dim: int,
    index_topk: int,
    offset: int,
) -> torch.Tensor:
    """Capture-safe B=1 indexer top-k with a static output width."""
    if index_topk == 0:
        return torch.empty(1, 1, 0, device=hidden_states.device, dtype=torch.int64)

    slot = slot_idx[:1].to(torch.long)
    pos = input_pos[:1].to(torch.long)
    max_compressed = indexer_compressed_kv_cache.shape[1]
    softmax_scale = index_head_dim**-0.5

    q = F.linear(q_lora, wq_b).view(1, 1, index_n_heads, index_head_dim)
    q_nope, q_pe = torch.split(q, [index_head_dim - rope_head_dim, rope_head_dim], dim=-1)
    q_pe = _apply_interleaved_rope(q_pe, cos_last.unsqueeze(2), sin_last.unsqueeze(2))
    q = torch.cat([q_nope, q_pe], dim=-1)
    q = _fake_fp4_act_quant(_hadamard_rotate(q), block_size=32)

    weights = F.linear(hidden_states, weights_proj).float() * (softmax_scale * index_n_heads**-0.5)
    index_k = indexer_compressed_kv_cache.index_select(0, slot)
    num_compressed = torch.div(pos + 1, compress_ratio, rounding_mode="floor")
    num_compressed = torch.clamp(num_compressed, min=0, max=max_compressed)
    compressed_positions = torch.arange(
        max_compressed, device=hidden_states.device, dtype=torch.long
    )
    invalid = compressed_positions.view(1, 1, -1) >= num_compressed.view(1, 1, 1)
    index_k = torch.where(~invalid.transpose(1, 2), index_k, torch.zeros_like(index_k))
    index_score = torch.einsum("bshd,btd->bsht", q, index_k).float()
    index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
    index_score = index_score.masked_fill(invalid, -1.0e20)

    k = min(index_topk, max_compressed)
    topk = index_score.topk(k, dim=-1).indices
    topk = torch.where(topk >= num_compressed.view(1, 1, 1), -1, topk + offset)
    if k < index_topk:
        pad = torch.full((1, 1, index_topk - k), -1, device=hidden_states.device, dtype=topk.dtype)
        topk = torch.cat([topk, pad], dim=-1)
    return topk.to(torch.int64)


def _decode_sparse_attn_step_static_cuda(
    q: torch.Tensor,
    kv_new: torch.Tensor,
    attn_sink: torch.Tensor,
    window_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    compress_topk: Optional[torch.Tensor],
    slot_idx: torch.Tensor,
    input_pos: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    scale: float,
    head_dim: int,
) -> torch.Tensor:
    """Capture-safe B=1 sparse decode attention over fixed masked candidate sets."""
    slot = slot_idx[:1].to(torch.long)
    pos = input_pos[:1].to(torch.long)
    window_pos = torch.remainder(pos, window_size)
    window_cache.index_put_(
        (slot, window_pos), kv_new.squeeze(0).squeeze(0).squeeze(0).view(1, head_dim)
    )

    window_offsets = torch.arange(window_size, device=q.device, dtype=torch.long)
    visible_window = torch.clamp(pos + 1, min=0, max=window_size)
    first_window_pos = pos + 1 - visible_window
    window_indices = torch.remainder(window_offsets + first_window_pos, window_size)
    window_slot = window_cache.index_select(0, slot).squeeze(0)
    gathered_window = window_slot.index_select(0, window_indices)
    window_valid = window_offsets < visible_window
    gathered_window = torch.where(
        window_valid.view(-1, 1), gathered_window, torch.zeros_like(gathered_window)
    )

    compressed_slot = compressed_kv_cache.index_select(0, slot).squeeze(0)
    max_compressed = compressed_kv_cache.shape[1]
    if compress_topk is not None:
        topk = compress_topk.view(-1)
        compressed_indices = torch.clamp(topk - window_size, min=0, max=max_compressed - 1)
        gathered_compressed = compressed_slot.index_select(0, compressed_indices)
        compressed_valid = topk >= window_size
    else:
        compressed_offsets = torch.arange(max_compressed, device=q.device, dtype=torch.long)
        num_compressed = torch.div(pos + 1, compress_ratio, rounding_mode="floor")
        num_compressed = torch.clamp(num_compressed, min=0, max=max_compressed)
        gathered_compressed = compressed_slot
        compressed_valid = compressed_offsets < num_compressed
    gathered_compressed = torch.where(
        compressed_valid.view(-1, 1),
        gathered_compressed,
        torch.zeros_like(gathered_compressed),
    )

    kv_all = torch.cat([gathered_window, gathered_compressed], dim=0)
    valid = torch.cat([window_valid, compressed_valid], dim=0)
    q_heads = q.squeeze(0).squeeze(0).float()
    kv_float = kv_all.float()
    scores = torch.matmul(q_heads, kv_float.transpose(0, 1)) * scale
    scores = scores.masked_fill(~valid.view(1, -1), -10000.0)

    sink_logits = attn_sink.float().view(-1, 1)
    logits_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink_logits)
    exp_scores = torch.exp(scores - logits_max)
    exp_sinks = torch.exp(sink_logits - logits_max)
    attn = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + exp_sinks)
    out = torch.matmul(attn, kv_float)
    return out.view(1, 1, q.shape[2], head_dim).to(q.dtype)


def _decode_sparse_attn_cuda_graph_step(
    q: torch.Tensor,
    kv: torch.Tensor,
    hidden_states: torch.Tensor,
    q_lora: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_anchor: torch.Tensor,
    sin_anchor: torch.Tensor,
    attn_sink: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_wgate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    indexer_wq_b: Optional[torch.Tensor],
    indexer_weights_proj: Optional[torch.Tensor],
    indexer_compressor_wkv: Optional[torch.Tensor],
    indexer_compressor_wgate: Optional[torch.Tensor],
    indexer_compressor_ape: Optional[torch.Tensor],
    indexer_compressor_norm_weight: Optional[torch.Tensor],
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    window_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    compressor_kv_state: torch.Tensor,
    compressor_score_state: torch.Tensor,
    indexer_compressed_kv_cache: torch.Tensor,
    indexer_kv_state: torch.Tensor,
    indexer_score_state: torch.Tensor,
    scale: float,
    window_size: int,
    compress_ratio: int,
    rope_head_dim: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rms_norm_eps: float,
) -> torch.Tensor:
    """B=1 CUDA decode path that avoids host sync and dynamic-shape tensors."""
    _decode_compressor_step_static_cuda(
        hidden_states,
        cos_anchor,
        sin_anchor,
        compressor_kv_state,
        compressor_score_state,
        compressed_kv_cache,
        compressor_wkv,
        compressor_wgate,
        compressor_ape,
        compressor_norm_weight,
        slot_idx,
        input_pos,
        compress_ratio,
        head_dim,
        rope_head_dim,
        rms_norm_eps,
    )

    compress_topk = None
    if indexer_wq_b is not None:
        _decode_compressor_step_static_cuda(
            hidden_states,
            cos_anchor,
            sin_anchor,
            indexer_kv_state,
            indexer_score_state,
            indexer_compressed_kv_cache,
            indexer_compressor_wkv,
            indexer_compressor_wgate,
            indexer_compressor_ape,
            indexer_compressor_norm_weight,
            slot_idx,
            input_pos,
            compress_ratio,
            index_head_dim,
            rope_head_dim,
            rms_norm_eps,
            rotate=True,
        )
        compress_topk = _decode_indexer_step_static_cuda(
            q_lora,
            cos,
            sin,
            indexer_wq_b,
            indexer_weights_proj,
            indexer_compressed_kv_cache,
            slot_idx,
            input_pos,
            hidden_states,
            compress_ratio,
            index_n_heads,
            index_head_dim,
            rope_head_dim,
            index_topk,
            offset=window_size,
        )

    return _decode_sparse_attn_step_static_cuda(
        q,
        kv,
        attn_sink,
        window_cache,
        compressed_kv_cache,
        compress_topk,
        slot_idx,
        input_pos,
        window_size,
        compress_ratio,
        scale,
        head_dim,
    )


_CACHE_NAMES = (
    "window_cache",
    "compressed_kv_cache",
    "compressor_kv_state",
    "compressor_score_state",
    "indexer_compressed_kv_cache",
    "indexer_kv_state",
    "indexer_score_state",
)


@torch.library.custom_op(
    "auto_deploy::torch_deepseek_v4_sparse_attn_with_cache", mutates_args=_CACHE_NAMES
)
def torch_deepseek_v4_sparse_attn_with_cache(
    # --- Source-op tensor args (19). Order MUST match the prefill op's qkv arg layout. ---
    q: torch.Tensor,
    kv: torch.Tensor,
    hidden_states: torch.Tensor,
    q_lora: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_anchor: torch.Tensor,
    sin_anchor: torch.Tensor,
    attn_sink: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_wgate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    indexer_wq_b: Optional[torch.Tensor],
    indexer_weights_proj: Optional[torch.Tensor],
    indexer_compressor_wkv: Optional[torch.Tensor],
    indexer_compressor_wgate: Optional[torch.Tensor],
    indexer_compressor_ape: Optional[torch.Tensor],
    indexer_compressor_norm_weight: Optional[torch.Tensor],
    # --- Standard AD metadata (5). Matches torch backend. ---
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # --- Caches (7, mutated). ---
    window_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    compressor_kv_state: torch.Tensor,
    compressor_score_state: torch.Tensor,
    indexer_compressed_kv_cache: torch.Tensor,
    indexer_kv_state: torch.Tensor,
    indexer_score_state: torch.Tensor,
    # --- Constants (9). Returned by AttentionDescriptor.get_constants. ---
    scale: float,
    window_size: int,
    compress_ratio: int,
    rope_head_dim: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rms_norm_eps: float,
) -> torch.Tensor:
    """DeepSeek V4 sparse attention with KV cache (prefill seeds, decode updates).

    Prefill (``s > 1``) supports ``B >= 1``; each element in the batch is
    processed independently and its KV caches are seeded using the corresponding
    slot from ``slot_idx[b]``.

    Decode (``s == 1``) supports ``B >= 1`` by processing each batch element's
    sequence slot independently.
    """
    del batch_info_host, cu_seqlen
    bsz, s, _, _ = q.shape
    has_indexer = indexer_wq_b is not None

    if s > 1:
        # --- Prefill: run the stateless body and seed caches per batch element. ---
        if bsz == 1:
            out = _sparse_attn_prefill_body(
                q,
                kv,
                hidden_states,
                q_lora,
                cos,
                sin,
                cos_anchor,
                sin_anchor,
                attn_sink,
                compressor_wkv,
                compressor_wgate,
                compressor_ape,
                compressor_norm_weight,
                indexer_wq_b,
                indexer_weights_proj,
                indexer_compressor_wkv,
                indexer_compressor_wgate,
                indexer_compressor_ape,
                indexer_compressor_norm_weight,
                scale=scale,
                window_size=window_size,
                compress_ratio=compress_ratio,
                rope_head_dim=rope_head_dim,
                head_dim=head_dim,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
                rms_norm_eps=rms_norm_eps,
            )
            slot = int(slot_idx[0].item())
            # Seed the window ring buffer with the last min(s, window) tokens.
            # Zero ALL unused slots first: UnpagedResourceHandler.allocate() uses
            # torch.empty() which leaves uninitialized memory (possibly NaN). If
            # we don't zero out the padding, kv_all = cat(window, compressed) at
            # decode time contains NaN garbage, ``q @ kv_all.T`` propagates NaN
            # into even the attention-masked positions, and logits NaN out.
            _seed_window_cache(kv, window_cache, slot, window_size)
            # Seed the compressed KV cache by re-running the stateless compressor.
            compressed = _sparse_compressor(
                hidden_states,
                cos,
                sin,
                compressor_wkv,
                compressor_wgate,
                compressor_ape,
                compressor_norm_weight,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                compress_ratio=compress_ratio,
                rms_norm_eps=rms_norm_eps,
            )  # [1, s, head_dim]
            compressed_kv_cache[slot].zero_()
            num_compressed = s // compress_ratio
            if num_compressed > 0:
                compressed_kv_cache[slot, :num_compressed] = compressed[0, :num_compressed]
            _seed_compressor_state(
                hidden_states,
                compressor_kv_state,
                compressor_score_state,
                compressor_wkv,
                compressor_wgate,
                compressor_ape,
                slot,
                compress_ratio,
            )
            if has_indexer:
                _seed_indexer_compressed_cache(
                    hidden_states,
                    cos,
                    sin,
                    indexer_compressor_wkv,
                    indexer_compressor_wgate,
                    indexer_compressor_ape,
                    indexer_compressor_norm_weight,
                    indexer_compressed_kv_cache,
                    indexer_kv_state,
                    indexer_score_state,
                    slot,
                    compress_ratio,
                    index_head_dim,
                    rope_head_dim,
                    rms_norm_eps,
                )
            return out
        else:
            # Batched prefill (B > 1): process each sequence independently and
            # stack the per-sequence outputs.  KV caches are seeded per-slot.
            outs = []
            for b in range(bsz):
                q_b = q[b : b + 1]
                kv_b = kv[b : b + 1] if kv.shape[0] == bsz else kv
                hs_b = hidden_states[b : b + 1]
                ql_b = q_lora[b : b + 1] if q_lora is not None else None
                cos_b = cos[b : b + 1] if cos.shape[0] == bsz else cos
                sin_b = sin[b : b + 1] if sin.shape[0] == bsz else sin
                slot_b = int(slot_idx[b].item())
                out_b = _sparse_attn_prefill_body(
                    q_b,
                    kv_b,
                    hs_b,
                    ql_b,
                    cos_b,
                    sin_b,
                    cos_anchor,
                    sin_anchor,
                    attn_sink,
                    compressor_wkv,
                    compressor_wgate,
                    compressor_ape,
                    compressor_norm_weight,
                    indexer_wq_b,
                    indexer_weights_proj,
                    indexer_compressor_wkv,
                    indexer_compressor_wgate,
                    indexer_compressor_ape,
                    indexer_compressor_norm_weight,
                    scale=scale,
                    window_size=window_size,
                    compress_ratio=compress_ratio,
                    rope_head_dim=rope_head_dim,
                    head_dim=head_dim,
                    index_n_heads=index_n_heads,
                    index_head_dim=index_head_dim,
                    index_topk=index_topk,
                    rms_norm_eps=rms_norm_eps,
                )
                # Zero full slot first to prevent uninitialized memory (from
                # torch.empty() in the resource handler) from leaking NaN into
                # attention at decode; see also the B==1 branch above.
                _seed_window_cache(kv_b, window_cache, slot_b, window_size)
                compressed = _sparse_compressor(
                    hs_b,
                    cos_b,
                    sin_b,
                    compressor_wkv,
                    compressor_wgate,
                    compressor_ape,
                    compressor_norm_weight,
                    head_dim=head_dim,
                    rope_head_dim=rope_head_dim,
                    compress_ratio=compress_ratio,
                    rms_norm_eps=rms_norm_eps,
                )  # [1, s, head_dim]
                compressed_kv_cache[slot_b].zero_()
                num_compressed = s // compress_ratio
                if num_compressed > 0:
                    compressed_kv_cache[slot_b, :num_compressed] = compressed[0, :num_compressed]
                _seed_compressor_state(
                    hs_b,
                    compressor_kv_state,
                    compressor_score_state,
                    compressor_wkv,
                    compressor_wgate,
                    compressor_ape,
                    slot_b,
                    compress_ratio,
                )
                if has_indexer:
                    _seed_indexer_compressed_cache(
                        hs_b,
                        cos_b,
                        sin_b,
                        indexer_compressor_wkv,
                        indexer_compressor_wgate,
                        indexer_compressor_ape,
                        indexer_compressor_norm_weight,
                        indexer_compressed_kv_cache,
                        indexer_kv_state,
                        indexer_score_state,
                        slot_b,
                        compress_ratio,
                        index_head_dim,
                        rope_head_dim,
                        rms_norm_eps,
                    )
                outs.append(out_b)
            return torch.cat(outs, dim=0)

    # --- Decode step: s == 1. ---
    # Supports bsz >= 1: each batch element is processed independently using its
    # own slot_idx[b] and input_pos[b].  Results are stacked along dim 0.
    if bsz == 1:
        if q.is_cuda:
            return _decode_sparse_attn_cuda_graph_step(
                q,
                kv,
                hidden_states,
                q_lora,
                cos,
                sin,
                cos_anchor,
                sin_anchor,
                attn_sink,
                compressor_wkv,
                compressor_wgate,
                compressor_ape,
                compressor_norm_weight,
                indexer_wq_b,
                indexer_weights_proj,
                indexer_compressor_wkv,
                indexer_compressor_wgate,
                indexer_compressor_ape,
                indexer_compressor_norm_weight,
                input_pos,
                slot_idx,
                window_cache,
                compressed_kv_cache,
                compressor_kv_state,
                compressor_score_state,
                indexer_compressed_kv_cache,
                indexer_kv_state,
                indexer_score_state,
                scale,
                window_size,
                compress_ratio,
                rope_head_dim,
                head_dim,
                index_n_heads,
                index_head_dim,
                index_topk,
                rms_norm_eps,
            )
        slot = int(slot_idx[0].item())
        pos = int(input_pos[0].item())
        # Compressor rolling update (may or may not produce a new compressed token).
        cos_last = cos  # already [1, 1, rd/2]
        sin_last = sin
        _decode_compressor_step(
            hidden_states,
            cos_anchor,
            sin_anchor,
            compressor_kv_state,
            compressor_score_state,
            compressed_kv_cache,
            compressor_wkv,
            compressor_wgate,
            compressor_ape,
            compressor_norm_weight,
            slot,
            pos,
            compress_ratio,
            head_dim,
            rope_head_dim,
            rms_norm_eps,
        )

        # Indexer rolling update + top-k.
        if has_indexer:
            _decode_compressor_step(
                hidden_states,
                cos_anchor,
                sin_anchor,
                indexer_kv_state,
                indexer_score_state,
                indexer_compressed_kv_cache,
                indexer_compressor_wkv,
                indexer_compressor_wgate,
                indexer_compressor_ape,
                indexer_compressor_norm_weight,
                slot,
                pos,
                compress_ratio,
                index_head_dim,
                rope_head_dim,
                rms_norm_eps,
                rotate=True,
            )
            compress_topk = _decode_indexer_step(
                q_lora,
                cos_last,
                sin_last,
                indexer_wq_b,
                indexer_weights_proj,
                indexer_compressed_kv_cache,
                slot,
                pos,
                hidden_states,
                compress_ratio,
                index_n_heads,
                index_head_dim,
                rope_head_dim,
                index_topk,
                offset=window_size,
            )
        else:
            # Deterministic compress top-k: positions 0, 1, ..., (pos+1)//ratio - 1.
            num_compressed = min((pos + 1) // compress_ratio, compressed_kv_cache.shape[1])
            full = torch.full((1, 1, num_compressed), -1, device=q.device, dtype=torch.int64)
            if num_compressed > 0:
                full[0, 0, :num_compressed] = (
                    torch.arange(num_compressed, device=q.device) + window_size
                )
            compress_topk = full

        # Sparse attention over concat(window_cache, compressed_kv_cache).
        out = _decode_sparse_attn_step(
            q,
            kv,
            attn_sink,
            window_cache,
            compressed_kv_cache,
            compress_topk,
            slot,
            pos,
            window_size,
            compress_ratio,
            scale,
            head_dim,
        )
        return out
    else:
        # Batched decode (bsz > 1): process each sequence independently and stack.
        outs = []
        for b in range(bsz):
            slot_b = int(slot_idx[b].item())
            pos_b = int(input_pos[b].item())
            q_b = q[b : b + 1]
            kv_b = kv[b : b + 1] if kv.shape[0] == bsz else kv
            hs_b = hidden_states[b : b + 1]
            ql_b = q_lora[b : b + 1] if q_lora is not None else None
            cos_b = cos[b : b + 1] if cos.shape[0] == bsz else cos
            sin_b = sin[b : b + 1] if sin.shape[0] == bsz else sin
            _decode_compressor_step(
                hs_b,
                cos_anchor,
                sin_anchor,
                compressor_kv_state,
                compressor_score_state,
                compressed_kv_cache,
                compressor_wkv,
                compressor_wgate,
                compressor_ape,
                compressor_norm_weight,
                slot_b,
                pos_b,
                compress_ratio,
                head_dim,
                rope_head_dim,
                rms_norm_eps,
            )
            if has_indexer:
                _decode_compressor_step(
                    hs_b,
                    cos_anchor,
                    sin_anchor,
                    indexer_kv_state,
                    indexer_score_state,
                    indexer_compressed_kv_cache,
                    indexer_compressor_wkv,
                    indexer_compressor_wgate,
                    indexer_compressor_ape,
                    indexer_compressor_norm_weight,
                    slot_b,
                    pos_b,
                    compress_ratio,
                    index_head_dim,
                    rope_head_dim,
                    rms_norm_eps,
                    rotate=True,
                )
                compress_topk_b = _decode_indexer_step(
                    ql_b,
                    cos_b,
                    sin_b,
                    indexer_wq_b,
                    indexer_weights_proj,
                    indexer_compressed_kv_cache,
                    slot_b,
                    pos_b,
                    hs_b,
                    compress_ratio,
                    index_n_heads,
                    index_head_dim,
                    rope_head_dim,
                    index_topk,
                    offset=window_size,
                )
            else:
                num_compressed_b = min((pos_b + 1) // compress_ratio, compressed_kv_cache.shape[1])
                full_b = torch.full(
                    (1, 1, num_compressed_b), -1, device=q.device, dtype=torch.int64
                )
                if num_compressed_b > 0:
                    full_b[0, 0, :num_compressed_b] = (
                        torch.arange(num_compressed_b, device=q.device) + window_size
                    )
                compress_topk_b = full_b
            out_b = _decode_sparse_attn_step(
                q_b,
                kv_b,
                attn_sink,
                window_cache,
                compressed_kv_cache,
                compress_topk_b,
                slot_b,
                pos_b,
                window_size,
                compress_ratio,
                scale,
                head_dim,
            )
            outs.append(out_b)
        return torch.cat(outs, dim=0)


@torch_deepseek_v4_sparse_attn_with_cache.register_fake
def torch_deepseek_v4_sparse_attn_with_cache_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    hidden_states: torch.Tensor,
    q_lora: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_anchor: torch.Tensor,
    sin_anchor: torch.Tensor,
    attn_sink: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_wgate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    indexer_wq_b: Optional[torch.Tensor],
    indexer_weights_proj: Optional[torch.Tensor],
    indexer_compressor_wkv: Optional[torch.Tensor],
    indexer_compressor_wgate: Optional[torch.Tensor],
    indexer_compressor_ape: Optional[torch.Tensor],
    indexer_compressor_norm_weight: Optional[torch.Tensor],
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    window_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    compressor_kv_state: torch.Tensor,
    compressor_score_state: torch.Tensor,
    indexer_compressed_kv_cache: torch.Tensor,
    indexer_kv_state: torch.Tensor,
    indexer_score_state: torch.Tensor,
    scale: float,
    window_size: int,
    compress_ratio: int,
    rope_head_dim: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rms_norm_eps: float,
) -> torch.Tensor:
    return torch.empty_like(q)


# ----------------------------------------------------------------------------
# AttentionDescriptor for the KV-cache transform
# ----------------------------------------------------------------------------


# Maximum compressed KV cache size: overhead for partial ratio chunks at prefill end.
_COMPRESSED_SLACK = 4


def _coff(compress_ratio: int) -> int:
    return 1 + int(compress_ratio == 4)


@AttentionRegistry.register("torch_ds_v4_sparse")
class DeepseekV4SparseAttentionDescriptor(AttentionDescriptor):
    """AttentionDescriptor for the DeepSeek V4 sparse (compress_ratio != 0) layers.

    The cached-op signature is:
        (*qkv_args_19, *std_metadata_5, *caches_7, *constants_9)

    which matches ``_InsertCachedOperator`` 's expected node-replacement contract.
    """

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        # q, kv, hidden_states, q_lora, cos, sin, cos_anchor, sin_anchor, attn_sink,
        # + 4 compressor weights + 6 (optional) indexer weights = 19
        return 19

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "seq_len", "input_pos", "slot_idx", "cu_seqlen"]

    @classmethod
    def _extract_constants(
        cls, source_attn_node: Node
    ) -> Tuple[float, int, int, int, int, int, int, int, float]:
        (
            scale,
            window_size,
            compress_ratio,
            rope_head_dim,
            head_dim,
            index_n_heads,
            index_head_dim,
            index_topk,
            rms_norm_eps,
        ) = extract_op_args(
            source_attn_node,
            "scale",
            "window_size",
            "compress_ratio",
            "rope_head_dim",
            "head_dim",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "rms_norm_eps",
        )
        return (
            scale,
            window_size,
            compress_ratio,
            rope_head_dim,
            head_dim,
            index_n_heads,
            index_head_dim,
            index_topk,
            rms_norm_eps,
        )

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        (
            _scale,
            window_size,
            compress_ratio,
            _rope_head_dim,
            head_dim,
            index_n_heads,
            index_head_dim,
            _index_topk,
            _rms_norm_eps,
        ) = cls._extract_constants(source_attn_node)
        # Fallback dtypes from the KV fake tensor.
        kv_fake = source_attn_node.args[1].meta["val"]
        kv_dtype = cls.resolve_cache_dtype(cache_config.dtype, kv_fake.dtype)
        coff = _coff(compress_ratio)

        # Compressed KV count upper-bound: ceil(max_seq_len / ratio) + slack.
        # We size it via UnpagedResourceHandler by using the max_seq_len dimension (first axis)
        # and collapsing to the head_dim. The handler already allocates
        # [max_num_state_slots, max_seq_len, *token_shape].
        handlers: ResourceHandlerDict = {
            "window_cache": UnpagedResourceHandler(head_dim, dtype=kv_dtype),
            "compressed_kv_cache": UnpagedResourceHandler(head_dim, dtype=kv_dtype),
            # Rolling state buffers: shape [max_batch, coff*compress_ratio, coff*head_dim].
            # Use StateResourceHandler (not UnpagedResourceHandler) because these have a
            # fixed state_shape independent of max_seq_len.
            "compressor_kv_state": StateResourceHandler(
                coff * compress_ratio, coff * head_dim, dtype=torch.float32
            ),
            "compressor_score_state": StateResourceHandler(
                coff * compress_ratio, coff * head_dim, dtype=torch.float32
            ),
        }
        if compress_ratio == 4:
            handlers["indexer_compressed_kv_cache"] = UnpagedResourceHandler(
                index_head_dim, dtype=kv_dtype
            )
            handlers["indexer_kv_state"] = StateResourceHandler(
                coff * compress_ratio, coff * index_head_dim, dtype=torch.float32
            )
            handlers["indexer_score_state"] = StateResourceHandler(
                coff * compress_ratio, coff * index_head_dim, dtype=torch.float32
            )
        else:
            # Zero-width stubs so the op signature stays fixed.
            handlers["indexer_compressed_kv_cache"] = UnpagedResourceHandler(0, dtype=kv_dtype)
            handlers["indexer_kv_state"] = StateResourceHandler(
                coff * compress_ratio, 0, dtype=torch.float32
            )
            handlers["indexer_score_state"] = StateResourceHandler(
                coff * compress_ratio, 0, dtype=torch.float32
            )
        del index_n_heads, window_size  # currently unused for cache sizing
        return handlers

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        return list(cls._extract_constants(source_attn_node))
