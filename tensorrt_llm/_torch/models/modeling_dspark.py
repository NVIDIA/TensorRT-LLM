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
#
# DSpark backbone ported from the DeepSeek-V4-Pro-DSpark reference
# (`inference/model.py`: DSparkBlock / Transformer.forward_spec).
#
# The captured-context attention primitives are ported from DeepSeek's DeepSpec
# reference ``inference/kernel.py`` (``sparse_attn``) and ``inference/model.py``
# (``get_dspark_topk_idxs``). The reference computes these with a TileLang
# kernel; this is a functional-first pure-PyTorch port with the same math
# (index-gather + online softmax + a learnable attention sink that contributes
# only to the softmax denominator).
#
# The draft I/O stages are ported from the same reference
# (`inference/model.py`: DSparkBlock.forward_embed / forward_head).
"""DSpark speculative-decoding drafters.

``DSpark`` names the speculative-decoding *algorithm*: a parallel block draft
over captured target hidden states, refined by a low-rank Markov head and
scheduled by a confidence head. Every ``decoding_type: DSpark`` drafter is built
here, in one of two flavours that differ only in how the draft is delivered:

* **embedded** — the draft weights ship inside the DeepSeek-V4-Pro *target*
  checkpoint under the ``mtp.*`` namespace and reuse the V4 decoder block, so the
  draft inherits the target's EPLB layer namespace and fp8/NVFP4 quantization.
  Most of this module is this flavour.
* **standalone** — the drafter ships as its own checkpoint and shares nothing
  with the target but the vocabulary and the captured hidden states. Its block
  decode is DFlash's, so :class:`GQADSparkForCausalLM` subclasses
  ``DFlashForCausalLM`` and adds the Markov head, the confidence head and the
  shift_label convention. See "Standalone DSpark drafters" near the bottom.

The dependency runs one way, ``modeling_dspark -> modeling_dflash``: DFlash is
DSpark minus those three heads, and :mod:`modeling_dflash` must not name a DSpark
class or the edge would become a cycle.

Four parts live here:

1. **Draft backbone** — ``n_mtp_layers`` (3 for V4-Pro) full DeepSeek-V4 blocks
   (MLA attention + MoE + manifold Hyper-Connections), plus:

     - **stage 0**: ``main_proj`` (Linear, fp8) + ``main_norm`` (RMSNorm) —
       projects the concatenation of captured target-layer hidden states
       ([58,59,60]) into the draft's cross-attention context (``main_x``);
       replaces vanilla-MTP's enorm/hnorm + e_proj/h_proj single-hidden mixing.
     - **last stage**: ``norm`` + ``markov_head`` + ``confidence_head`` + flat
       ``hc_head`` — the block-draft output head. The heads themselves are shared
       with the standalone path and live in :mod:`modeling_speculative`.

2. **Captured-context attention primitives** — the dense sliding-window MLA the
   draft block attends with (``get_dspark_topk_idxs`` / ``dspark_sparse_attn``
   and the rotary helpers). Hardware-agnostic and unit-testable in isolation.

3. **Block draft I/O** — ``build_draft_input_ids`` (the
   ``[bonus_token, noise, ...]`` block input) and ``dspark_propose`` (Markov
   refinement + static confidence truncation).

4. **Standalone DSpark drafters** — :class:`GQADSparkForCausalLM`, plus the
   ``_build_dspark_draft`` dispatch that picks between the two flavours on
   deployment form alone.

The per-stage *backbone* forward (block attention whose K/V derive from
``main_x``, + MoE + mHC) is brought up and numerically validated against the real
fp8 weights separately; ``forward_embed`` (capture) and ``forward_head`` (block
draft) are the reference-faithful, unit-validated I/O stages.
"""

import copy
import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._utils import is_sm_100f
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ..distributed import AllReduceParams
from ..modules.linear import Linear
from ..modules.mhc.hyper_connection import HCHead
from ..modules.rms_norm import RMSNorm
from ..speculative.interface import SpeculativeDecodingMode
from ..utils import AuxStreamType
from .modeling_deepseekv4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4WeightLoader,
    _get_deepseek_v4_routed_moe_scale_name,
    _maybe_view_deepseek_v4_routed_moe_tensor,
    _normalize_deepseek_v4_nvfp4_mixed_precision_config,
    _rename_deepseek_v4_attn_subkey,
    _rename_deepseek_v4_ffn_subkey,
)
from .modeling_dflash import DFlashForCausalLM, resolve_dspark_head_config
from .modeling_speculative import (
    DSparkConfidenceHead,
    build_markov_head,
    confident_prefix_length,
    dspark_markov_chain_logits,
)
from .modeling_utils import register_draft_model

if IS_CUTLASS_DSL_AVAILABLE:
    from ..custom_ops.dspark_attention_custom_op import (
        fused_dsv4_dspark_attention,
        is_fused_dsv4_dspark_attention_supported,
    )
    from ..custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
        cute_dsl_dspark_rmsnorm_rope_cache_write,
        cute_dsl_dspark_rmsnorm_rope_draft_block,
        is_fused_dspark_attention_preparation_supported,
        is_fused_dspark_rmsnorm_rope_supported,
    )

# ----------------------------------------------------------------------------
# Captured-context attention primitives.
# ----------------------------------------------------------------------------


def precompute_dspark_freqs_cis(
    rope_head_dim: int,
    seqlen: int,
    rope_theta: float = 10000.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Plain (non-YaRN) RoPE complex exponentials for the DSpark draft.

    The dense draft attention (``compress_ratio == 0``) disables YaRN and uses the
    base ``rope_theta`` (DeepSpec ``precompute_freqs_cis`` with
    ``original_seq_len == 0``).

    Returns:
        complex64 tensor ``[seqlen, rope_head_dim // 2]``.
    """
    freqs = 1.0 / (
        rope_theta
        ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float32, device=device) / rope_head_dim)
    )
    t = torch.arange(seqlen, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_dspark_rotary(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply (or, with ``inverse``, de-apply) rotary embeddings, DeepSpec-style.

    Functional (non-in-place) port of DeepSpec ``apply_rotary_emb``: treats the
    last dim as adjacent (re, im) pairs, rotates by ``freqs_cis`` indexed along the
    sequence axis, and conjugates for the inverse (de-rotation applied to the
    attention output). ``x`` is the rope-dim slice only: ``[b, s, rd]`` (3D) or
    ``[b, s, h, rd]`` (4D), with ``freqs_cis`` of shape ``[s, rd // 2]``.
    """
    orig_dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        fc = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        fc = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    out = torch.view_as_real(xc * fc).flatten(-2)
    return out.to(orig_dtype)


def apply_dspark_rotary_batched(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Per-row (batched) variant of :func:`apply_dspark_rotary`.

    Identical math, but ``freqs_cis`` carries a leading batch axis so each row of
    ``x`` is rotated by its own per-request phases (the generation draft runs each
    request at a different absolute ``start_pos``). ``x`` is the rope-dim slice
    only: ``[G, s, rd]`` (3D) or ``[G, s, h, rd]`` (4D), with ``freqs_cis`` of shape
    ``[G, s, rd // 2]``.
    """
    orig_dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    g, s, half = freqs_cis.shape
    if xc.ndim == 3:
        fc = freqs_cis.view(g, s, half)
    else:
        fc = freqs_cis.view(g, s, 1, half)
    out = torch.view_as_real(xc * fc).flatten(-2)
    return out.to(orig_dtype)


@lru_cache(maxsize=64)
def _topk_matrix(window_size: int, block_size: int, start_pos: int) -> torch.Tensor:
    # [min(window, start_pos+1)] context positions in the rolling KV window,
    # followed by [block_size] positions for the current block's own K/V (which
    # the caller appends to the window at offset ``window_size``).
    ctx = torch.arange(min(window_size, start_pos + 1))
    blk = window_size + torch.arange(block_size)
    return torch.cat([ctx, blk]).int()


def get_dspark_topk_idxs(
    window_size: int,
    bsz: int,
    block_size: int,
    start_pos: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-query attended-position indices for the DSpark draft block.

    Mirrors DeepSpec ``get_dspark_topk_idxs``: every one of the ``block_size``
    query positions attends to the same set — the ``min(window_size, start_pos+1)``
    most-recent context positions in the rolling KV window, then the
    ``block_size`` positions of the current block (stored at offset
    ``window_size`` in the concatenated KV). Note this is *non-causal* within the
    block (every position sees every block position), matching the reference.

    Args:
        window_size: sliding-window length of the captured-context KV cache.
        bsz: batch size.
        block_size: number of draft positions per request.
        start_pos: absolute decode position (must be > 0); bounds the context.
        device: device for the returned index tensor.

    Returns:
        int32 tensor ``[bsz, block_size, topk]`` with
        ``topk = min(window_size, start_pos+1) + block_size``.
    """
    assert start_pos > 0, "DSpark draft attention runs at generation (start_pos > 0)"
    matrix = _topk_matrix(int(window_size), int(block_size), int(start_pos)).to(device)
    return matrix.view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


def get_dspark_topk_idxs_batched(
    window_size: int,
    block_size: int,
    start_pos: torch.Tensor,
    valid_len: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sync-free, fixed-size (CUDA-graph-safe) batched ``get_dspark_topk_idxs``.

    Unlike the scalar :func:`get_dspark_topk_idxs` (whose ``topk`` width
    ``min(window_size, start_pos+1) + block_size`` depends on the host int
    ``start_pos``), this always returns the **fixed** width ``window_size +
    block_size`` and masks the unfilled context slots with ``-1``. The masked
    slots are excluded by :func:`dspark_sparse_attn` exactly as if they were
    absent while the shape remains CUDA-graph safe.

    Every query attends to the actually written circular-window suffix, followed
    by the current-block positions. Without ``valid_len`` this preserves the
    legacy ``start_pos``-only behavior.

    Args:
        window_size: sliding-window length of the captured-context KV cache.
        block_size: number of draft positions per request.
        start_pos: ``[G]`` int tensor of per-request absolute decode positions.
        valid_len: optional ``[G]`` count of actually written rolling-window
            entries. When omitted, preserve the legacy ``start_pos`` mask.

    Returns:
        int32 tensor ``[G, block_size, window_size + block_size]``.
    """
    device = start_pos.device
    g = start_pos.shape[0]
    ctx_cols = torch.arange(window_size, device=device)  # [win]
    if valid_len is None:
        valid = ctx_cols.unsqueeze(0) <= start_pos.unsqueeze(1)  # [G, win]
    else:
        # The valid entries are the contiguous logical suffix ending at
        # start_pos, but their physical slots wrap modulo window_size.
        valid_len = valid_len.clamp(min=0, max=window_size)
        age = torch.remainder(start_pos.unsqueeze(1) - ctx_cols.unsqueeze(0), window_size)
        valid = age < valid_len.unsqueeze(1)
    ctx_idx = torch.where(
        valid, ctx_cols.unsqueeze(0).expand(g, -1), torch.full_like(valid, -1, dtype=torch.long)
    )
    blk_idx = window_size + torch.arange(block_size, device=device)  # [block]
    blk_idx = blk_idx.unsqueeze(0).expand(g, -1)  # [G, block]
    row = torch.cat([ctx_idx, blk_idx], dim=1).to(torch.int32)  # [G, win+block]
    return row.unsqueeze(1).expand(g, block_size, -1).contiguous()


def dspark_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Index-gathered multi-query attention with an attention sink.

    Functional-first port of the DeepSpec ``sparse_attn`` TileLang kernel. For
    each ``(batch, query, head)`` it gathers the ``topk`` KV rows named by
    ``topk_idxs`` (an index of ``-1`` masks that slot), computes a scaled
    dot-product softmax over them, and adds a per-head learnable *sink* logit that
    participates only in the softmax denominator (i.e. an "attend-to-nothing"
    option with a zero value vector). KV is shared across query heads (MQA).

    Args:
        q: ``[b, m, h, d]`` query (``m`` = block_size, ``h`` = heads).
        kv: ``[b, n, d]`` keys/values (shared across heads).
        attn_sink: ``[h]`` per-head sink logits (fp32).
        topk_idxs: ``[b, m, topk]`` int gather indices into ``kv`` (``-1`` masks).
        softmax_scale: scalar applied to the q·k scores (``head_dim ** -0.5``).

    Returns:
        ``[b, m, h, d]`` attention output, in ``q.dtype``.
    """
    b, m, h, d = q.shape
    idx = topk_idxs.long()  # [b, m, topk]
    valid = idx >= 0
    safe = idx.clamp(min=0)

    # Invalid slots read kv[0, :] (via safe.clamp), but masked_fill below
    # zeros their softmax probs, so the einsum nullifies them.
    kv_exp = kv.unsqueeze(1).expand(b, m, kv.shape[1], d)
    gathered = torch.gather(kv_exp, 2, safe.unsqueeze(-1).expand(b, m, safe.shape[-1], d)).float()

    # Scores [b, m, h, topk]; mask invalid slots to -inf before the softmax.
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    # Online-softmax max is taken over gathered positions only (the sink is added
    # to the denominator afterwards), matching the kernel's reduce order.
    smax = scores.max(dim=-1, keepdim=True).values  # [b, m, h, 1]
    smax = torch.where(torch.isinf(smax), torch.zeros_like(smax), smax)
    probs = torch.exp(scores - smax)  # masked slots -> exp(-inf) = 0
    sink = torch.exp(attn_sink.to(torch.float32).view(1, 1, h) - smax.squeeze(-1))
    denom = probs.sum(dim=-1) + sink  # [b, m, h]
    out = torch.einsum("bmhk,bmkd->bmhd", probs, gathered) / denom.unsqueeze(-1)
    return out.to(q.dtype)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm matching the DeepSpec reference (fp32 reduce, then * weight)."""
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * xf).to(dtype)


def _rope_last_dims(
    t: torch.Tensor, rope_head_dim: int, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply RoPE to the last ``rope_head_dim`` dims; pass the rest through."""
    nope = t[..., :-rope_head_dim]
    rope = apply_dspark_rotary(t[..., -rope_head_dim:], freqs_cis, inverse=inverse)
    return torch.cat([nope, rope], dim=-1)


def _rope_last_dims_batched(
    t: torch.Tensor, rope_head_dim: int, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Per-row variant of :func:`_rope_last_dims` (``freqs_cis`` has a batch axis)."""
    nope = t[..., :-rope_head_dim]
    rope = apply_dspark_rotary_batched(t[..., -rope_head_dim:], freqs_cis, inverse=inverse)
    return torch.cat([nope, rope], dim=-1)


def _rmsnorm_rope_batched(
    t: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rope_head_dim: int,
    freqs_cis: torch.Tensor,
    *,
    num_heads: int = 1,
    apply_weight: bool = True,
    apply_rmsnorm: bool = True,
    inverse_rope: bool = False,
) -> torch.Tensor:
    """Fuse DSpark RMSNorm and last-dimension RoPE when supported."""
    if IS_CUTLASS_DSL_AVAILABLE and is_sm_100f():
        freqs_real = torch.view_as_real(freqs_cis).reshape(-1, freqs_cis.shape[-1], 2)
        if is_fused_dspark_rmsnorm_rope_supported(t, weight, freqs_real, num_heads, rope_head_dim):
            return cute_dsl_dspark_rmsnorm_rope(
                t,
                weight,
                freqs_real,
                num_heads,
                rope_head_dim,
                eps,
                apply_weight,
                apply_rmsnorm,
                inverse_rope,
            )

    if apply_rmsnorm:
        if apply_weight:
            t = _rmsnorm(t, weight, eps)
        else:
            t = t * torch.rsqrt(t.square().mean(-1, keepdim=True) + eps)
    elif apply_weight:
        t = (t.float() * weight.float()).to(t.dtype)
    if rope_head_dim > 0:
        t = _rope_last_dims_batched(t, rope_head_dim, freqs_cis, inverse=inverse_rope)
    return t


def dspark_attention_forward(
    x: torch.Tensor,
    main_x: torch.Tensor,
    start_pos: int,
    kv_cache: torch.Tensor,
    *,
    wq_a: torch.Tensor,
    q_norm_w: torch.Tensor,
    wq_b: torch.Tensor,
    wkv: torch.Tensor,
    kv_norm_w: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    attn_sink: torch.Tensor,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    n_groups: int,
    o_lora_rank: int,
    window_size: int,
    eps: float,
    softmax_scale: float,
    freqs_cis: torch.Tensor,
    persist: bool = False,
) -> torch.Tensor:
    """Captured-context DSpark draft attention (generation path, ``start_pos > 0``).

    Functional port of DeepSpec ``DSparkAttention.forward`` for the dense
    (``compress_ratio == 0``) draft: low-rank Q (``wq_a`` -> ``q_norm`` -> ``wq_b``)
    with a per-head RMS + RoPE, MQA K/V from ``wkv`` (shared across heads), keys
    gathered from a rolling captured-context window (``kv_cache``, into which the
    projected ``main_x`` context is written at ``start_pos % window_size``) plus the
    block's own positions, attention-sink softmax, inverse-RoPE on the output, and a
    grouped low-rank O projection (``wo_a`` einsum + ``wo_b``).

    Weights are plain tensors for ``F.linear`` (the caller supplies the loaded /
    dequantized projection weights); ``wo_a`` is the raw grouped weight matrix
    ``[n_groups * o_lora_rank, n_heads * head_dim // n_groups]``. ``kv_cache`` is
    ``[b, window_size, head_dim]`` and is updated functionally (cloned).

    Returns:
        ``[b, block_size, dim]`` attention output (residual stream contribution).
    """
    assert start_pos > 0, "DSpark draft attention runs at generation (start_pos > 0)"
    b, block, _ = x.shape
    rd = rope_head_dim
    main_freqs = freqs_cis[start_pos : start_pos + 1]
    blk_freqs = freqs_cis[start_pos + 1 : start_pos + 1 + block]

    # Captured-context K/V from main_x (MQA, shared across heads).
    main_kv = _rmsnorm(F.linear(main_x, wkv), kv_norm_w, eps)  # [b, 1, head_dim]
    main_kv = _rope_last_dims(main_kv, rd, main_freqs)

    # Query: low-rank + per-head RMS + RoPE.
    q = _rmsnorm(F.linear(x, wq_a), q_norm_w, eps)
    q = F.linear(q, wq_b).unflatten(-1, (n_heads, head_dim))  # [b, block, h, head_dim]
    # Per-head RMS in the query dtype (matches the reference inline normalization,
    # which is NOT the fp32 RMSNorm path).
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    q = _rope_last_dims(q, rd, blk_freqs)

    # Block K/V.
    kv = _rmsnorm(F.linear(x, wkv), kv_norm_w, eps)  # [b, block, head_dim]
    kv = _rope_last_dims(kv, rd, blk_freqs)

    # Write the context K/V into the rolling window, then attend over
    # [window context | block] with the sink. ``persist=True`` writes through
    # to the caller's buffer (cross-step decode, worker-owned window); the
    # default clones so single-shot callers (golden / unit tests) stay pure.
    cache = kv_cache if persist else kv_cache.clone()
    cache[:, start_pos % window_size] = main_kv.squeeze(1)
    kv_full = torch.cat([cache, kv], dim=1)  # [b, window + block, head_dim]
    topk = get_dspark_topk_idxs(window_size, b, block, start_pos, device=x.device)
    o = dspark_sparse_attn(q, kv_full, attn_sink, topk, softmax_scale)  # [b, block, h, head_dim]
    o = _rope_last_dims(o, rd, blk_freqs, inverse=True)

    # Grouped low-rank O projection.
    o = o.reshape(b, block, n_groups, -1)
    wo_a_v = wo_a.view(n_groups, o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a_v)
    return F.linear(o.flatten(2), wo_b)


def dspark_attention_forward_batched(
    x: torch.Tensor,
    main_x: torch.Tensor,
    start_pos: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    valid_len: torch.Tensor | None = None,
    *,
    wq_a: torch.Tensor,
    q_norm_w: torch.Tensor,
    wq_b: torch.Tensor,
    wkv: torch.Tensor,
    kv_norm_w: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    attn_sink: torch.Tensor,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    n_groups: int,
    o_lora_rank: int,
    window_size: int,
    eps: float,
    softmax_scale: float,
    freqs_cis: torch.Tensor,
    persist: bool = False,
) -> torch.Tensor:
    """Batched, CUDA-graph-safe captured-context DSpark draft attention.

    This path belongs to the embedded DeepSeek-V4-Pro draft. Standalone DSpark
    drafters (including Kimi-K3-DSpark) use the DFlash paged-KV decode instead.

    Numerically identical, per request, to :func:`dspark_attention_forward`, but
    free of host syncs and data-dependent shapes so it can be captured into a CUDA
    graph (the one-engine drafter runs inside the target's graph). The differences
    from the scalar path are purely mechanical:

    * ``start_pos`` is a ``[G]`` int tensor (one absolute decode position per gen
      request) instead of a python int; RoPE phases are *gathered* per request from
      the fixed ``freqs_cis`` table rather than sliced.
    * the rolling-window context K/V is written/read through the ``slots`` index
      into a shared ``kv_cache`` (``persist=True`` writes through to the caller's
      worker-owned buffer; otherwise a clone is used), instead of mutating a
      per-request cache in place.
    * the attended-position list has the fixed width ``window_size + block_size``
      with ``-1`` masking (see :func:`get_dspark_topk_idxs_batched`).

    Args:
        x: ``[G, block, dim]`` block layer input (per gen request).
        main_x: ``[G, 1, hidden]`` projected captured context.
        start_pos: ``[G]`` int tensor of absolute decode positions (> 0).
        kv_cache: ``[N, window_size, head_dim]`` rolling captured-context windows
            (``N`` rows indexed by ``slots``; ``N == G`` for single-shot callers).
        slots: ``[G]`` int tensor mapping each request to its ``kv_cache`` row.
        valid_len: optional ``[G]`` count of actually written context entries;
            masks holes left when absolute positions are bootstrapped without
            receiving the corresponding DSpark rolling-window state.
        freqs_cis: ``[maxlen, rope_head_dim // 2]`` precomputed plain-RoPE table;
            must satisfy ``maxlen > start_pos.max() + block_size``.

    Returns:
        ``[G, block, dim]`` attention output (residual stream contribution).
    """
    g, block, _ = x.shape
    if kv_cache.shape[1] != window_size:
        raise ValueError(
            f"kv_cache window extent {kv_cache.shape[1]} does not match window_size {window_size}"
        )
    rd = rope_head_dim
    # Per-request RoPE phases gathered from the fixed table (no host-int slicing).
    main_freqs = freqs_cis[start_pos].unsqueeze(1)  # [G, 1, rd//2]
    blk_pos = start_pos.unsqueeze(1) + 1 + torch.arange(block, device=x.device)  # [G, block]
    blk_freqs = freqs_cis[blk_pos]  # [G, block, rd//2]

    # Keep the two GEMM outputs live until dispatch: the fused path lets
    # their RMSNorm/RoPE kernels store directly into attention's physical inputs.
    main_kv_input = F.linear(main_x, wkv)

    # Query: low-rank + per-head RMS + RoPE.
    q = _rmsnorm_rope_batched(F.linear(x, wq_a), q_norm_w, eps, 0, blk_freqs)
    q = F.linear(q, wq_b).unflatten(-1, (n_heads, head_dim))  # [G, block, h, head_dim]
    q = _rmsnorm_rope_batched(
        q,
        kv_norm_w,
        eps,
        rd,
        blk_freqs,
        num_heads=n_heads,
        apply_weight=False,
    )

    block_kv_input = F.linear(x, wkv)

    # ``persist=True`` writes through to the worker-owned rolling window;
    # single-shot callers keep the original functional clone behavior.
    write_target = kv_cache if persist else kv_cache.clone()
    main_rope_freqs = torch.view_as_real(main_freqs).reshape(-1, rd // 2, 2).contiguous()
    inverse_rope_freqs = torch.view_as_real(blk_freqs).contiguous()
    block_rope_freqs = inverse_rope_freqs.reshape(-1, rd // 2, 2)

    # The DSV4 fused kernel requires explicit physical-window validity; None
    # keeps the reference path.
    use_fused_dsv4_dspark_attention = (
        valid_len is not None
        and IS_CUTLASS_DSL_AVAILABLE
        and is_fused_dsv4_dspark_attention_supported(
            q, write_target, valid_len, attn_sink, inverse_rope_freqs
        )
        and is_fused_dspark_attention_preparation_supported(
            main_kv_input,
            block_kv_input,
            kv_norm_w,
            main_rope_freqs,
            block_rope_freqs,
            write_target,
            slots,
            start_pos,
        )
    )
    if use_fused_dsv4_dspark_attention:
        slots_i32, cache_seqs = cute_dsl_dspark_rmsnorm_rope_cache_write(
            main_kv_input,
            kv_norm_w,
            main_rope_freqs,
            write_target,
            slots,
            start_pos,
            eps,
        )
        draft_block = cute_dsl_dspark_rmsnorm_rope_draft_block(
            block_kv_input,
            kv_norm_w,
            block_rope_freqs,
            eps,
        )
        o = fused_dsv4_dspark_attention(
            q,
            draft_block,
            write_target,
            slots_i32,
            cache_seqs,
            valid_len,
            attn_sink,
            inverse_rope_freqs,
            softmax_scale,
        )
    else:
        main_kv = _rmsnorm_rope_batched(main_kv_input, kv_norm_w, eps, rd, main_freqs)
        kv = _rmsnorm_rope_batched(block_kv_input, kv_norm_w, eps, rd, blk_freqs)
        main_kv_flat = main_kv.squeeze(1).to(write_target.dtype)

        slot_pos = start_pos % window_size  # [G]
        write_target[slots, slot_pos] = main_kv_flat
        cache_rows = write_target[slots]  # [G, window, head_dim]
        kv_full = torch.cat([cache_rows, kv], dim=1)  # [G, window + block, head_dim]
        topk = get_dspark_topk_idxs_batched(window_size, block, start_pos, valid_len)
        o = dspark_sparse_attn(
            q, kv_full, attn_sink, topk, softmax_scale
        )  # [G, block, h, head_dim]
        o = _rmsnorm_rope_batched(
            o,
            kv_norm_w,
            eps,
            rd,
            blk_freqs,
            num_heads=n_heads,
            apply_weight=False,
            apply_rmsnorm=False,
            inverse_rope=True,
        )

    # Grouped low-rank O projection.
    o = o.reshape(g, block, n_groups, -1)
    wo_a_v = wo_a.view(n_groups, o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a_v)
    return F.linear(o.flatten(2), wo_b)


# ----------------------------------------------------------------------------
# Block draft I/O.
# ----------------------------------------------------------------------------


def build_draft_input_ids(
    bonus_token_ids: torch.Tensor, *, block_size: int, noise_token_id: int
) -> torch.Tensor:
    """``[batch] -> [batch, block_size]`` = ``[bonus, noise, noise, ...]``.

    The first position is the verified bonus token (the target's last accepted
    token); the rest are the DSpark noise/mask token (id 128799 for V4-Pro).
    """
    batch = bonus_token_ids.shape[0]
    out = bonus_token_ids.new_full((batch, block_size), int(noise_token_id))
    out[:, 0] = bonus_token_ids
    return out


def dspark_propose(
    base_logits: torch.Tensor,
    *,
    bonus_token_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    markov_head: Optional[nn.Module],
    confidence_head: Optional[nn.Module],
    block_size: int,
    temperature: float = 0.0,
    confidence_threshold: float = 0.0,
    return_logits: bool = False,
) -> tuple:
    """Produce DSpark draft tokens for one block (functional-first, static length).

    Args:
        base_logits: ``[batch, block_size, vocab]`` from the backbone + lm_head.
        bonus_token_ids: ``[batch]`` the token preceding the first draft position.
        block_hidden: ``[batch, block_size, hidden]`` backbone hidden (feeds the
            confidence head, and the RNN-head variant).
        markov_head / confidence_head: the validated DSpark heads (may be None).
    Returns:
        draft_tokens: ``[batch, block_size]`` sampled tokens (full block; callers
            keep the tensor fixed-width for CUDA-graph safety).
        num_proposed: ``[batch]`` int32 — how many leading tokens survive the
            static confidence-threshold truncation (== block_size when no head /
            threshold<=0).
    """
    batch = base_logits.shape[0]
    # ``draft_logits`` are the per-position distributions the draft token is drawn
    # from (markov-corrected when a head is present, else the raw base logits).
    # Surfaced under ``return_logits`` for the §7.9 probabilistic-acceptance
    # (1-TV) measurement; the normal path ignores them.
    draft_logits = base_logits
    if markov_head is not None:
        draft_tokens, corrected = markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=bonus_token_ids,
            hidden_states=block_hidden,
            temperature=temperature,
        )
        draft_logits = corrected
    else:
        from .modeling_speculative import greedy_or_sample

        draft_tokens = greedy_or_sample(base_logits, temperature)

    # Scaffolding: confidence-based dynamic drafting is NOT enabled in this PR.
    # The worker always calls with confidence_threshold=0.0, so the block below is
    # inert and num_proposed stays == block_size (the full block is proposed). The
    # returned num_proposed is intentionally not yet consumed by the speculative
    # scheduler/verifier; wiring it through is a follow-up (see PR description).
    num_proposed = torch.full(
        (batch,), int(block_size), dtype=torch.int32, device=base_logits.device
    )
    if confidence_head is not None and confidence_threshold > 0.0:
        # prev token at position k is [bonus, draft_0, ..., draft_{k-1}]
        prev_ids = torch.cat([bonus_token_ids.unsqueeze(1), draft_tokens[:, :-1]], dim=1)
        prev_emb = (
            markov_head.get_prev_embeddings(prev_ids)
            if (markov_head is not None and getattr(confidence_head, "with_markov", False))
            else None
        )
        conf_logits = (
            confidence_head(block_hidden, prev_embeddings=prev_emb)
            if prev_emb is not None
            else confidence_head(block_hidden)
        )
        # Per-request prefix truncation (batch handled row-wise to stay simple;
        # functional-first scope typically runs batch=1 for the draft).
        for b in range(batch):
            num_proposed[b] = confident_prefix_length(
                conf_logits[b : b + 1], block_size=block_size, threshold=confidence_threshold
            )
    if return_logits:
        return draft_tokens, num_proposed, draft_logits
    return draft_tokens, num_proposed


# ----------------------------------------------------------------------------
# Draft backbone (``mtp.*`` stages of DeepSeek-V4 blocks).
# ----------------------------------------------------------------------------

# Matches the draft namespace ``mtp.<stage>.<rest>`` in the V4-Pro-DSpark
# checkpoint. Each draft stage is a full DeepSeek-V4 block stored under this
# prefix; the main model's keys (``layers.*``, ``embed.weight``, ``head.weight``,
# top-level ``norm.weight`` / ``hc_head_*``) are loaded by the target model.
_DSPARK_MTP_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")


def _active_moe_load_balancer():
    """The engine-wide ``MoeLoadBalancer``, or None when EPLB is not active.

    Non-None exactly inside ``maybe_create_moe_load_balancer(...)`` when EPLB is
    really enabled for this engine (supported arch, ``moe_ep_size > 1``, no smart
    router, ``moe_load_balancer`` configured) -- i.e. exactly the condition under
    which ``MoE._init_load_balancer`` consumes ``model_config.moe_load_balancer``.
    Gating every DSpark EPLB check on it keeps the non-EPLB path untouched.
    """
    from ..moe.fused_moe.moe_load_balancer import get_moe_load_balancer

    return get_moe_load_balancer()


def validate_dspark_eplb_layer_base(model_config, draft_config) -> None:
    """Require the draft's layer namespace to match the target's, under EPLB.

    DSpark stages take ``layer_idx = draft_config.num_hidden_layers + stage_id``
    and register as extra EPLB layers in the *target* engine's balancer, whose
    ``initial_global_assignments`` are keyed by target layer index. If the draft
    checkpoint's config reports a different depth the stages silently land on the
    wrong keys, so fail fast instead. Only enforced when EPLB is active; a
    draft-only checkpoint config remains valid without EPLB.
    """
    if _active_moe_load_balancer() is None:
        return
    target_layers = model_config.pretrained_config.num_hidden_layers
    draft_layers = draft_config.pretrained_config.num_hidden_layers
    if target_layers != draft_layers:
        raise ValueError(
            "DSpark + EPLB requires the draft checkpoint config to report the "
            f"same num_hidden_layers as the target (target={target_layers}, "
            f"draft={draft_layers}). DSpark stage layer indices are derived as "
            "draft num_hidden_layers + stage_id and must line up with the "
            "target layer namespace that initial_global_assignments is keyed by."
        )


def validate_dspark_eplb_stage_layers(model_config, base: int, num_stages: int) -> None:
    """Validate the EPLB config actually covers the DSpark draft stages.

    DSpark registers each stage as an independent EPLB layer at index
    ``base + stage_id`` (``base = num_hidden_layers``). Two failure modes are
    caught here, before any DSpark MoE layer is built, so the user gets one
    actionable error instead of a bare ``KeyError`` from deep inside MoE init:

      1. online EPLB, which DSpark does not support (see below);
      2. an ``initial_global_assignments`` map generated without DSpark enabled,
         which therefore lacks the draft stage indices.
    """
    if _active_moe_load_balancer() is None:
        return
    lb_config = getattr(model_config, "moe_load_balancer", None)
    if lb_config is None:
        return

    draft_layers = list(range(base, base + num_stages))

    # DSpark supports STATIC EPLB only. Online EPLB requires every registered MoE
    # layer to run exactly once per iteration, but the DSpark draft MoE is skipped
    # on iterations with no generation requests anywhere (context-only batches,
    # warmup), and the balancer's CPU worker then spins forever in its untimed
    # waitCpuStage() waiting for a GPU signal that is only emitted from an MoE
    # forward -- a silent deadlock.
    if getattr(lb_config, "layer_updates_per_iter", 0) > 0:
        raise ValueError(
            "DSpark speculative decoding supports static EPLB only, but "
            f"layer_updates_per_iter={lb_config.layer_updates_per_iter} requests "
            "online EPLB. The DSpark draft MoE does not run on iterations without "
            "generation requests (context-only batches, warmup), which deadlocks "
            "the MoE load balancer worker. Set layer_updates_per_iter=0 in the "
            "load balancer config, or disable DSpark."
        )

    assignments = getattr(lb_config, "initial_global_assignments", None)
    if not assignments:
        # No custom placement: the auto-generated assignment covers every layer.
        return
    missing = [layer_idx for layer_idx in draft_layers if layer_idx not in assignments]
    if missing:
        raise ValueError(
            f"initial_global_assignments is missing DSpark layer(s) {missing}. "
            f"The {num_stages} DSpark draft stages register as additional EPLB "
            f"layers with indices [{base}, {base + num_stages}). Regenerate the "
            "EPLB config from statistics collected with DSpark enabled (see "
            "examples/wide_ep/ep_load_balancer/README.md), or omit "
            "initial_global_assignments to use the auto-generated placement."
        )


def count_dspark_stages(ckpt_dir: str) -> Optional[int]:
    """Count the DSpark draft stages (``mtp.{s}.*``) in a checkpoint index.

    The HF ``config.json`` does not expose ``n_mtp_layers`` (only the reference
    ``inference/config.json`` does), so the authoritative draft stage count is
    the number of distinct ``mtp.<stage>`` prefixes in the weight index. Returns
    ``None`` if the index is missing or has no ``mtp.*`` keys (caller falls back
    to the config-derived default).
    """
    index = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return None
    with open(index, encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})
    stages = {int(m.group(1)) for k in weight_map if (m := _DSPARK_MTP_RE.match(k))}
    return (max(stages) + 1) if stages else None


def _rename_dspark_stage_subkey(rest: str, routed_scale: str) -> str:
    """Map a per-stage checkpoint subkey to the ``DSv4DSparkBlock`` param subkey."""
    if rest == "attn_norm.weight":
        return "input_layernorm.weight"
    if rest == "ffn_norm.weight":
        return "post_attention_layernorm.weight"
    # Flat manifold-Hyper-Connections / draft-head weights are loaded via
    # ``load_flat_hc_weights`` (keyed by the parent module stem), so pass the
    # flat-underscore form through unchanged:
    #   hc_attn_* / hc_ffn_*  -> mHC on every block
    #   hc_head_*             -> HCHead on the last stage
    if rest.startswith(("hc_attn_", "hc_ffn_", "hc_head_")):
        return rest
    # DSpark capture projection (stage 0): fp8 Linear .scale -> .weight_scale_inv.
    if rest == "main_proj.scale":
        return "main_proj.weight_scale_inv"
    if rest.startswith("attn."):
        return f"self_attn.{_rename_deepseek_v4_attn_subkey(rest[len('attn.') :])}"
    if rest.startswith("ffn."):
        return f"mlp.{_rename_deepseek_v4_ffn_subkey(rest[len('ffn.') :], routed_scale)}"
    # main_proj.weight, main_norm.weight, norm.weight, markov_head.*,
    # confidence_head.* map 1:1 onto the DSv4DSparkBlock submodules.
    return rest


def remap_dspark_draft_keys(weights: Dict, num_stages: int) -> Dict:
    """Convert checkpoint ``mtp.{s}.*`` keys to ``mtp_layers.{s}.*`` model keys.

    Only the draft namespace is consumed (stages ``[0, num_stages)``); shared
    ``embed_tokens`` / ``lm_head`` and other top-level keys belong to the target
    model and are skipped here. The routed-expert scale suffix mirrors the V4
    loader: ``weight_scale`` for the packed MXFP4 layout, else ``weight_scale_inv``.
    """
    routed_scale = _get_deepseek_v4_routed_moe_scale_name(weights, "mtp.")
    out: Dict[str, torch.Tensor] = {}
    for k, v in weights.items():
        m = _DSPARK_MTP_RE.match(k)
        if not m:
            continue
        stage = int(m.group(1))
        if stage >= num_stages:
            continue
        sub = _rename_dspark_stage_subkey(m.group(2), routed_scale)
        model_key = f"mtp_layers.{stage}.{sub}"
        v = _maybe_view_deepseek_v4_routed_moe_tensor(model_key, v, routed_scale)
        out[model_key] = v
    return out


# The checkpoint stores
# ``mtp.{s}.attn.wo_a`` as fp8_e4m3 + a UE8M0 128x128 block scale (verified), and
# the reference (`inference/model.py`, ``self.wo_a`` is a bf16 ColumnParallelLinear
# loaded from the fp8 ckpt) uses the DEQUANTIZED bf16 ``wo_a`` (== ``wo_a_fp8 *
# scale`` ~ absmean 0.065). The bf16 captured-context path historically skipped
# this dequant (raw fp8-cast-to-bf16, ~993x too large); the correct behavior is to
# dequantize ``wo_a`` (cos 1.0 vs ``wo_a_fp8 * scale``). Always dequantize now.


class DSv4DSparkBlock(DeepseekV4DecoderLayer):
    """One DSpark draft stage = a DeepSeek-V4 decoder block + DSpark extras.

    ``stage_id`` in ``[0, num_stages)``; only stage 0 owns the capture projection
    and only the last stage owns the draft heads, matching the ``mtp.*`` schema.
    """

    def __init__(
        self,
        model_config,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        *,
        stage_id: int,
        num_stages: int,
        num_capture_layers: int,
    ):
        # The inherited attention uses a draft-local layer index, while the
        # decoder layer keeps its model-level index for weights and captures.
        super().__init__(
            model_config,
            layer_idx,
            aux_stream_dict,
            attention_layer_idx=stage_id,
            disable_post_moe_fusion=True,
        )
        config = model_config.pretrained_config
        spec_cfg = getattr(model_config, "spec_config", None)
        self.stage_id = int(stage_id)
        self.num_stages = int(num_stages)
        # mask_token_id is a user override on the speculative_config; None means
        # fall back to the draft checkpoint's dspark_noise_token_id.
        mask_token_id = getattr(spec_cfg, "mask_token_id", None)
        self.noise_token_id = int(
            mask_token_id
            if mask_token_id is not None
            else getattr(config, "dspark_noise_token_id", config.vocab_size)
        )
        self.markov_rank = int(getattr(config, "dspark_markov_rank", 0))
        self.hc_mult = config.hc_mult
        # markov_head_type is a user override on the speculative_config; None
        # means fall back to the draft checkpoint's dspark_markov_head_type.
        markov_head_type = getattr(spec_cfg, "markov_head_type", None)
        if markov_head_type is None:
            markov_head_type = getattr(config, "dspark_markov_head_type", "vanilla")
        self.markov_head_type = markov_head_type

        # Stage 0: capture projection of the concatenated target-layer hiddens.
        if self.has_capture:
            self.main_proj = Linear(
                config.hidden_size * num_capture_layers,
                config.hidden_size,
                bias=False,
                dtype=config.torch_dtype,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )
            self.main_norm = RMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
            )

        # Last stage: the block-draft output heads + mHC head + final norm.
        if self.has_heads:
            self.norm = RMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
            )
            self.hc_head = HCHead(config.hc_mult, config.hidden_size)
            self.markov_head = build_markov_head(
                markov_head_type=self.markov_head_type,
                vocab_size=config.vocab_size,
                markov_rank=self.markov_rank,
                hidden_size=config.hidden_size,
            )
            self.confidence_head = DSparkConfidenceHead(
                hidden_size=config.hidden_size,
                markov_rank=self.markov_rank,
                # Only concat the Markov prev-token embedding when a Markov head
                # actually exists (build_markov_head returns None for
                # markov_rank <= 0); otherwise dspark_propose passes no
                # prev_embeddings and DSparkConfidenceHead.forward would assert.
                with_markov=self.markov_rank > 0,
            )

    @property
    def has_capture(self) -> bool:
        return self.stage_id == 0

    @property
    def has_heads(self) -> bool:
        return self.stage_id == self.num_stages - 1


class DSv4DSparkDraftModel(nn.Module):
    """The ``n_mtp_layers``-stage DSpark draft stacked on a DeepSeek-V4 target.

    Shares ``embed_tokens`` / ``lm_head`` with the target model. ``forward_embed``
    builds the block input from the captured context; the per-stage backbone runs
    the 3 blocks; ``forward_head`` produces the block draft tokens + confidence.
    """

    def __init__(
        self,
        model_config,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        num_stages: Optional[int] = None,
        block_size: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.config = config
        # The DSpark stage count is NOT the HF ``num_nextn_predict_layers`` (=1).
        # It is ``n_mtp_layers`` (3 for V4-Pro), which lives in the draft
        # sub-checkpoint config (inference/config.json) and is reflected by the
        # ``mtp.{0..n-1}.*`` weight namespace. Resolve it from (in priority):
        # an explicit override, the spec config's ``num_draft_layers``, a
        # pretrained-config ``n_mtp_layers``, else fall back to nextn.
        spec_cfg = getattr(model_config, "spec_config", None)
        self.num_stages = int(
            num_stages
            if num_stages is not None
            else getattr(spec_cfg, "num_draft_layers", None)
            or getattr(config, "n_mtp_layers", None)
            or config.num_nextn_predict_layers
        )
        # Production passes the validated speculative-config value explicitly;
        # direct construction falls back to the checkpoint's trained block size.
        self.block_size = int(
            block_size if block_size is not None else getattr(config, "dspark_block_size", 5)
        )
        # mask_token_id is a user override on the speculative_config; None means
        # fall back to the draft checkpoint's dspark_noise_token_id.
        mask_token_id = getattr(spec_cfg, "mask_token_id", None)
        self.noise_token_id = int(
            mask_token_id
            if mask_token_id is not None
            else getattr(config, "dspark_noise_token_id", config.vocab_size)
        )
        self.hc_mult = config.hc_mult
        target_layer_ids = getattr(config, "dspark_target_layer_ids", [])
        self.num_capture_layers = len(target_layer_ids)
        base = config.num_hidden_layers
        # Each DSpark stage becomes an independent EPLB layer at index base + s.
        # Validate the load-balancer config covers them (and rejects online EPLB)
        # before building any MoE, so a stale config fails with one actionable
        # error instead of a bare KeyError from inside MoE._init_load_balancer.
        validate_dspark_eplb_stage_layers(model_config, base, self.num_stages)
        # Derive a draft-only model_config (a shallow copy so the shared config
        # and the target model are untouched) carrying two draft-specific fixes:
        #
        #  1. compress_ratios SLICE — the draft runs as a separate engine, so the
        #     inherited DeepSeek-V4 block remaps each block's layer_idx to a
        #     draft-local index in [0, num_stages) (the 1-layer-style draft KV
        #     cache). Sparse-attention compress_ratios / RoPE are indexed by that
        #     draft-local id, so they must be the draft slice
        #     (compress_ratios[base : base + num_stages]); otherwise indices
        #     0..n-1 resolve to the first *main* layers' sparse ratios — building
        #     a compressor the DSpark draft lacks and selecting YaRN over the
        #     dense path. For V4-Pro the draft slice is [1, 1, 1] (dense).
        #
        #  2. quant_config_dict EXTENSION — the checkpoint's per-module quant map
        #     only enumerates the base layers, so the draft layers' routed
        #     experts fall back to the global fp8 config and build fp8-shaped
        #     buffers. The draft experts are physically MXFP4 (same as the main
        #     MoE layers), so copy a main MoE layer's experts quant onto the
        #     draft layer keys.
        draft_model_config = self._derive_draft_model_config(model_config, base, self.num_stages)
        self.mtp_layers = nn.ModuleList(
            [
                DSv4DSparkBlock(
                    draft_model_config,
                    base + s,
                    aux_stream_dict,
                    stage_id=s,
                    num_stages=self.num_stages,
                    num_capture_layers=self.num_capture_layers,
                )
                for s in range(self.num_stages)
            ]
        )
        # Shared with target; wired by the spec wrapper after construction.
        self.embed_tokens: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None

        # Scalar attention params for the captured-context draft attention. These
        # are the dense (compress_ratio == 0) DSparkAttention constants — see the
        # reference ``inference/model.py`` ``Attention.__init__``. ``head_dim`` is
        # the MLA latent (MQA) dim; ``softmax_scale = head_dim ** -0.5``; the dense
        # draft disables YaRN and uses the base ``rope_theta``.
        self._attn_params = dict(
            n_heads=int(config.num_attention_heads),
            head_dim=int(
                getattr(config, "head_dim", config.kv_lora_rank + config.qk_rope_head_dim)
            ),
            rope_head_dim=int(config.qk_rope_head_dim),
            n_groups=int(config.o_groups),
            o_lora_rank=int(config.o_lora_rank),
            window_size=int(getattr(config, "window_size", 128)),
            eps=float(config.rms_norm_eps),
        )
        self._attn_params["softmax_scale"] = self._attn_params["head_dim"] ** -0.5
        self._rope_theta = float(getattr(config, "rope_theta", 10000.0))
        # Fixed-cap plain-RoPE table shared by the eager and CUDA-graph-safe
        # batched paths. It is built once per device and gathered/sliced by the
        # runtime decode positions, so the cache does not grow with sequence
        # length and the batched consuming op's shape remains static.
        self._freqs_cap = (
            int(getattr(config, "max_position_embeddings", 163840)) + self.block_size + 2
        )
        self._freqs_table_cache: Dict = {}

    def post_load_weights(self) -> None:
        """Run the one-shot post-load transforms for the draft's quant linears.

        The fp8 UE8M0 linears we invoke as modules (``main_proj``, shared experts,
        the heads) need ``resmooth_to_fp8_e8m0`` + ``transform_sf_into_required_layout``
        before the first forward, or the kernel reads raw scales and emits NaNs.
        ``Linear.transform_weights`` is idempotent; the routed-expert MoE packs
        itself in its own ``load_weights``.

        The bf16 captured-context attention does NOT use the MLA module's forward —
        it runs ``dspark_attention_forward`` on dequantized bf16 weights cached via
        :meth:`cache_attn_weights_from_checkpoint` — so the MLA projection linears are
        skipped here (they would otherwise be transformed into the deep_gemm layout we
        don't consume).
        """
        attn_linear_ids = set()
        for stage in self.mtp_layers:
            for m in stage.self_attn.modules():
                if isinstance(m, Linear):
                    attn_linear_ids.add(id(m))

        for module in self.modules():
            if isinstance(module, Linear) and id(module) not in attn_linear_ids:
                module.transform_weights()

    @staticmethod
    def _block_dequant(w_fp8: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
        """DeepSeek ``block``×``block`` block-scale dequant → bf16: ``real = fp8 * scale``.

        ``scale`` (possibly UE8M0) is broadcast over each ``block``×``block`` tile.
        Pure-torch (matches the golden-validated reference dequant), robust to the
        e8m0 scale dtype.
        """
        wf = w_fp8.float()
        out, inn = wf.shape
        s = scale.float()
        s_full = s.repeat_interleave(block, 0)[:out].repeat_interleave(block, 1)[:, :inn]
        return (wf * s_full).to(torch.bfloat16)

    def _cache_attn_weights(self, src: Dict) -> None:
        """Populate each stage's ``_dspark_attn`` from a dict of raw ``mtp.{s}.attn.*``
        tensors (source-agnostic core shared by the two public entry points).

        The captured-context attention runs the validated ``dspark_attention_forward``
        free function on reference-layout bf16 weights dequantized here. Sourcing the
        separate ``wq_a``/``wkv`` (plain 128×128 block scale) sidesteps the TRT-LLM
        ``MLA`` module's fused + interleaved fp8 storage (``kv_a_proj_with_mqa`` fuses
        ``q_a``+``kv`` and stores the scale interleaved). This mirrors the
        golden-validated dequant exactly.
        """
        for s, stage in enumerate(self.mtp_layers):
            pref = f"mtp.{s}.attn."
            dev = stage.input_layernorm.weight.device

            def deq(name: str, fp8: bool) -> torch.Tensor:
                w = src[f"{pref}{name}.weight"].to(dev)
                if fp8:
                    return self._block_dequant(w, src[f"{pref}{name}.scale"].to(dev))
                return w.to(torch.bfloat16)

            stage._dspark_attn = dict(
                wq_a=deq("wq_a", True),
                q_norm_w=src[f"{pref}q_norm.weight"].to(dev).to(torch.bfloat16),
                wq_b=deq("wq_b", True),
                wkv=deq("wkv", True),
                kv_norm_w=src[f"{pref}kv_norm.weight"].to(dev).to(torch.bfloat16),
                # wo_a IS fp8+scale in the checkpoint (verified); always dequant.
                wo_a=deq("wo_a", True),
                wo_b=deq("wo_b", True),
                attn_sink=src[f"{pref}attn_sink"].to(dev).float(),
            )

    def cache_attn_weights_from_checkpoint(self, ckpt_dir: str, weight_map: Dict[str, str]) -> None:
        """Populate ``_dspark_attn`` by reading the ``mtp.{s}.attn.*`` tensors from the
        checkpoint shards on disk, then dequantizing via :meth:`_cache_attn_weights`.

        TODO(step 3): source these from the loaded ``MLA`` modules instead, once the
        fused/interleaved fp8 scale layout is decoded, to drop the checkpoint I/O.
        """
        from safetensors import safe_open

        prefixes = tuple(f"mtp.{s}.attn." for s in range(len(self.mtp_layers)))
        shards: Dict[str, list] = {}
        for k in weight_map:
            if k.startswith(prefixes):
                shards.setdefault(weight_map[k], []).append(k)
        raw: Dict[str, torch.Tensor] = {}
        for shard, ks in shards.items():
            with safe_open(os.path.join(ckpt_dir, shard), framework="pt", device="cpu") as f:
                for k in ks:
                    raw[k] = f.get_tensor(k)
        self._cache_attn_weights(raw)

    def cache_attn_weights_from_state_dict(self, weights: Dict) -> None:
        """Populate ``_dspark_attn`` from an already-loaded in-memory ``weights`` dict
        (no extra disk I/O); used on the one-engine load path
        (``DSv4DSparkForCausalLM.load_weights``). Delegates to :meth:`_cache_attn_weights`.
        """
        self._cache_attn_weights(weights)

    def _dspark_freqs_table(self, device: torch.device) -> torch.Tensor:
        """Return the fixed-size plain-RoPE table cached for ``device``."""
        key = str(device)
        cached = self._freqs_table_cache.get(key)
        if cached is None:
            cached = precompute_dspark_freqs_cis(
                self._attn_params["rope_head_dim"],
                self._freqs_cap,
                rope_theta=self._rope_theta,
                device=device,
            )
            self._freqs_table_cache[key] = cached
        return cached

    @classmethod
    def _derive_draft_model_config(cls, model_config, base: int, num_stages: int):
        """Return a draft-only ``model_config`` copy with draft-specific fixes.

        Applies (1) the ``compress_ratios`` draft slice and (2) the
        ``quant_config_dict`` MXFP4 extension for the draft layers' routed
        experts. A single shallow copy is made (and only when something needs to
        change) so the shared ``model_config`` and the target model are untouched.

        The draft MoE backend is **inherited** from the target's
        ``model_config.moe_backend`` (carried by the shallow copy) — not pinned —
        matching every other drafter (the MTP module reuses the V4 decoder layer,
        whose MoE is built with ``moe_backend=model_config.moe_backend``; separate
        Eagle3/DFlash drafts resolve it from their own config the same way). The
        draft ``mtp.*`` stages are full V4 blocks, so they share the target's
        MXFP4 ``n_routed_experts=384`` / ``n_group=8`` (= 48 experts/group) layout
        and therefore the same backend constraints: pick a backend that supports
        it (CUTLASS today, DeepGEMM megaMoE once available) on the target and the
        draft follows. Note the TRTLLM-Gen ``blockScaleMoe`` routing kernel asserts
        ``experts/group <= 32`` (warp size), so it is incompatible with this layout
        for both the target and the draft.
        """
        new_sa = cls._draft_sparse_config(model_config, base, num_stages)
        new_qcd = cls._draft_quant_config_dict(model_config, base, num_stages)
        new_qc = cls._draft_normalized_quant_config(model_config)
        if new_sa is None and new_qcd is None and new_qc is None:
            return model_config
        draft_cfg = copy.copy(model_config)
        # ModelConfig is a frozen dataclass; bypass the guard for these fields.
        if new_sa is not None:
            object.__setattr__(draft_cfg, "sparse_attention_config", new_sa)
        if new_qcd is not None:
            object.__setattr__(draft_cfg, "quant_config_dict", new_qcd)
        if new_qc is not None:
            object.__setattr__(draft_cfg, "quant_config", new_qc)
        return draft_cfg

    @staticmethod
    def _draft_normalized_quant_config(model_config):
        """Resolved global ``quant_config`` for NVFP4 DSpark checkpoints, or None.

        NVFP4 DSpark checkpoints declare a global ``MIXED_PRECISION`` quant algo
        (per-layer NVFP4 routed experts over an FP8 base). The target resolves it
        in :func:`_normalize_deepseek_v4_nvfp4_mixed_precision_config`
        (base -> ``FP8_BLOCK_SCALES``); the separately-built draft config needs
        the same, otherwise the inherited ``DeepseekV4DecoderLayer`` asserts
        ``"MIXED_PRECISION is ambiguous"`` when it builds the draft stages.
        Returns the resolved ``quant_config`` to set on the draft copy, or None
        when nothing changes (e.g. the MXFP4 checkpoint, whose global algo is not
        ``MIXED_PRECISION``) so the draft config is left byte-identical.
        """
        qc = getattr(model_config, "quant_config", None)
        if qc is None or getattr(qc, "quant_algo", None) != QuantAlgo.MIXED_PRECISION:
            return None
        # Reuse the target normalizer on a throwaway shallow copy so we only
        # extract the resolved global quant_config; the shared ``model_config``
        # is left untouched (the normalizer rebinds ``.quant_config`` on the copy).
        probe = copy.copy(model_config)
        object.__setattr__(probe, "_frozen", False)
        normalized = _normalize_deepseek_v4_nvfp4_mixed_precision_config(probe)
        resolved = getattr(normalized, "quant_config", qc)
        return resolved if resolved is not qc else None

    @staticmethod
    def _draft_sparse_config(model_config, base: int, num_stages: int):
        """Sparse-attention config sliced to the draft layers, or None if N/A.

        The inherited block remaps ``layer_idx`` to a draft-local index, so the
        sparse config must expose the draft layers' per-layer ratios at indices
        ``[0, num_stages)``.
        """
        sa = getattr(model_config, "sparse_attention_config", None)
        compress_ratios = getattr(sa, "compress_ratios", None) if sa is not None else None
        if not compress_ratios or len(compress_ratios) < base + num_stages:
            return None
        draft_ratios = list(compress_ratios)[base : base + num_stages]
        # Already draft-local (e.g. a draft-only checkpoint config); no slice.
        if (
            draft_ratios == list(compress_ratios)[:num_stages]
            and len(compress_ratios) == num_stages
        ):
            return None
        return sa.model_copy(update={"compress_ratios": draft_ratios})

    @staticmethod
    def _draft_quant_config_dict(model_config, base: int, num_stages: int):
        """quant_config_dict extended to cover draft-layer experts, or None.

        The checkpoint's per-module quant map only enumerates the base layers, so
        ``model.layers.{base+s}.mlp.experts`` would fall back to the global fp8
        config and build fp8-shaped expert buffers. The draft routed experts are
        physically MXFP4 (identical to the main MoE layers), so copy a
        representative main MoE layer's experts quant onto the draft layer keys.
        """
        qcd = getattr(model_config, "quant_config_dict", None)
        if not qcd:
            return None
        src = next(
            (
                qcd[f"model.layers.{li}.mlp.experts"]
                for li in range(base)
                if f"model.layers.{li}.mlp.experts" in qcd
            ),
            None,
        )
        if src is None:
            return None
        new_qcd = dict(qcd)
        changed = False
        for s in range(num_stages):
            key = f"model.layers.{base + s}.mlp.experts"
            if new_qcd.get(key) is not src:
                new_qcd[key] = src
                changed = True
        return new_qcd if changed else None

    @torch.inference_mode()
    def write_context_windows(
        self,
        main_hidden: torch.Tensor,
        positions: torch.Tensor,
        stage_windows: torch.Tensor,
    ) -> None:
        """Write captured-context ``main_kv`` into the rolling per-stage KV windows.

        Replicates exactly the per-position context write that
        :func:`dspark_attention_forward` performs each generation step
        (``main_kv = RoPE_pos(rmsnorm(wkv @ main_x))`` written at
        ``window[pos % window_size]``), but for an arbitrary set of
        ``(captured-hidden, position)`` pairs. Used to (a) seed a request's
        window from its prompt at prefill and (b) back-fill the intermediate
        accepted tokens of a multi-accept step — both of which the per-step
        generation path would otherwise leave as holes, starving the draft
        attention of context (acceptance-rate only; verified decoding keeps
        output correctness regardless).

        Args:
            main_hidden: ``[M, num_capture * hidden]`` captured target hiddens.
            positions: ``[M]`` absolute window positions (used for BOTH the RoPE
                phase and the slot ``pos % window_size``). By the generation-path
                convention this is ``committed_position + 1``. Must hold at most
                ``window_size`` entries with distinct slots (the caller passes a
                contiguous, deduplicated range) so the scatter is well defined.
            stage_windows: ``[num_stages, window_size, head_dim]`` window for one
                request's slot; updated in place.
        """
        if getattr(self.mtp_layers[0], "_dspark_attn", None) is None:
            return
        M = int(main_hidden.shape[0])
        if M == 0:
            return
        win = int(self._attn_params["window_size"])
        rd = int(self._attn_params["rope_head_dim"])
        eps = float(self._attn_params["eps"])
        positions = positions.to(main_hidden.device).long()
        # main_x is stage-invariant (stage 0's projection), matching forward_embed.
        stage0 = self.mtp_layers[0]
        main_x = stage0.main_norm(stage0.main_proj(main_hidden))  # [M, hidden]
        freqs = self._dspark_freqs_table(main_x.device)[positions]
        slots = positions % win  # [M]
        mx = main_x.unsqueeze(0)  # [1, M, hidden] for the per-position RoPE layout
        for s, stage in enumerate(self.mtp_layers):
            a = stage._dspark_attn
            kv = _rmsnorm(F.linear(mx, a["wkv"]), a["kv_norm_w"], eps)  # [1, M, head_dim]
            kv = _rope_last_dims(kv, rd, freqs)  # [1, M, head_dim]
            stage_windows[s, slots] = kv[0].to(stage_windows.dtype)

    def write_context_windows_batched(
        self,
        main_hidden: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        mask: torch.Tensor,
        kv_windows: torch.Tensor,
    ) -> None:
        """CUDA-graph-safe batched + masked variant of :meth:`write_context_windows`.

        Back-fills the intermediate accepted tokens of a multi-accept step into the
        rolling per-stage KV windows for ALL gen requests at once, with a fixed
        ``[G, M]`` shape (``M = max interim per request``) and a validity mask
        (invalid entries are no-ops via a read-modify-write), so nothing depends on
        the per-request accept count. Same per-position math as the scalar version
        (``RoPE_pos(rmsnorm(wkv @ main_x))`` written at ``window[pos % window]``),
        but indexed/scattered through ``slots`` into the shared persistent buffer.

        Args:
            main_hidden: ``[G, M, num_capture * hidden]`` captured target hiddens
                (rows beyond a request's interim count are masked).
            positions: ``[G, M]`` absolute window positions (RoPE phase + slot).
            slots: ``[G]`` row index of each request into ``kv_windows``.
            mask: ``[G, M]`` bool — which ``(g, m)`` entries are real interim writes.
            kv_windows: ``[N, num_stages, window_size, head_dim]`` persistent buffer;
                updated in place.
        """
        if getattr(self.mtp_layers[0], "_dspark_attn", None) is None:
            return
        G, M = positions.shape
        if G == 0 or M == 0:
            return
        win = int(self._attn_params["window_size"])
        rd = int(self._attn_params["rope_head_dim"])
        eps = float(self._attn_params["eps"])
        positions = positions.long()
        slots = slots.long()
        freqs = self._dspark_freqs_table(main_hidden.device)[positions]  # [G, M, rd//2]
        cols = positions % win  # [G, M]
        rows = slots[:, None].expand(-1, M)  # [G, M]
        mask3 = mask.unsqueeze(-1)  # [G, M, 1]
        stage0 = self.mtp_layers[0]
        main_x = stage0.main_norm(stage0.main_proj(main_hidden))  # [G, M, hidden]
        for s, stage in enumerate(self.mtp_layers):
            a = stage._dspark_attn
            kv = _rmsnorm(F.linear(main_x, a["wkv"]), a["kv_norm_w"], eps)  # [G, M, head_dim]
            kv = _rope_last_dims_batched(kv, rd, freqs)  # [G, M, head_dim]
            win_s = kv_windows[:, s]  # [N, win, head_dim] view onto the base buffer
            # Read-modify-write so masked-out (g, m) entries keep their current
            # value — a graph-safe masked scatter (no dynamic-shape compaction).
            cur = win_s[rows, cols]  # [G, M, head_dim]
            win_s[rows, cols] = torch.where(mask3, kv.to(win_s.dtype), cur)

    def forward_embed(
        self, main_hidden: torch.Tensor, bonus_token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the draft block input and the cross-attention context.

        Args:
            main_hidden: ``[num_tokens, num_capture * hidden]`` captured context.
            bonus_token_ids: ``[num_tokens]`` last accepted token per request.
        Returns:
            x: hc-expanded block embeddings ``[num_tokens, block_size, hc_mult, hidden]``
            main_x: projected context ``[num_tokens, hidden]``
            draft_ids: block input token ids ``[num_tokens, block_size]`` (the callers
                reuse these as the per-position MoE routing ids, so we return them
                rather than rebuild).
        """
        stage0 = self.mtp_layers[0]
        main_x = stage0.main_norm(stage0.main_proj(main_hidden))
        draft_ids = build_draft_input_ids(
            bonus_token_ids, block_size=self.block_size, noise_token_id=self.noise_token_id
        )
        x = self.embed_tokens(draft_ids)
        x = x.unsqueeze(-2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x, draft_ids

    def _forward_stage(
        self,
        stage: "DSv4DSparkBlock",
        h: torch.Tensor,
        main_x: torch.Tensor,
        start_pos,
        freqs_cis: torch.Tensor,
        moe_input_ids: torch.Tensor,
        stage_window: Optional[torch.Tensor] = None,
        slots: Optional[torch.Tensor] = None,
        valid_len: Optional[torch.Tensor] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """One DSpark stage = reference ``Block.forward`` with captured-context attn.

        ``h`` is the mHC residual stream ``[T, block, hc_mult, hidden]``. The mHC
        ``pre_mapping``/``post_mapping`` preserve the leading ``[T, block]`` dims;
        the captured-context attention and MoE run on the collapsed token axis.
        Mirrors the reference (unfused) mHC boundaries exactly:
        ``hc_pre → attn_norm → DSparkAttention → hc_post`` then
        ``hc_pre → ffn_norm → MoE → hc_post``.

        ``stage_window`` is this stage's persistent rolling captured-context KV
        window ``[T, window_size, head_dim]`` owned by the worker across decode
        steps; the attention writes the current ``main_kv`` into it in place. When
        ``None`` (golden / single-shot) a zero window is allocated per call.

        When ``slots`` (a ``[G]`` int tensor) is given, the CUDA-graph-safe batched
        attention (:func:`dspark_attention_forward_batched`) is used instead: it
        takes ``start_pos`` as a ``[G]`` tensor and writes/reads ``stage_window``
        (then shaped ``[N, window_size, head_dim]``) through the ``slots`` index.
        """
        T, block, _, hidden = h.shape

        # --- attention sub-block (captured-context, not paged-KV MLA) ---
        residual = h
        post_mix, comb_mix, layer_input = stage.hc_attn.pre_mapping(residual)
        layer_input = stage.input_layernorm(layer_input)  # [T, block, hidden]
        # Rolling-window cache: persist through the worker-owned ``stage_window``
        # for cross-step decode, else a fresh zero window for a single block.
        persist = stage_window is not None
        kv_cache = (
            stage_window
            if persist
            else torch.zeros(
                T,
                self._attn_params["window_size"],
                self._attn_params["head_dim"],
                dtype=torch.bfloat16,
                device=h.device,
            )
        )
        if slots is not None:
            # Batched, CUDA-graph-safe path (start_pos is a [G] tensor; window is
            # written/read through ``slots``).
            attn = dspark_attention_forward_batched(
                layer_input,
                main_x,
                start_pos,
                kv_cache,
                slots,
                valid_len=valid_len,
                freqs_cis=freqs_cis,
                persist=True,
                **stage._dspark_attn,
                **self._attn_params,
            )
        else:
            attn = dspark_attention_forward(
                layer_input,
                main_x,
                start_pos,
                kv_cache,
                freqs_cis=freqs_cis,
                persist=persist,
                **stage._dspark_attn,
                **self._attn_params,
            )
        if stage.enable_fused_hc:
            residual, post_mix, comb_mix, layer_input = stage.hc_ffn.fused_hc(
                x_prev=attn,
                residual_prev=residual,
                post_mix_prev=post_mix,
                comb_mix_prev=comb_mix,
                norm_weight=stage.post_attention_layernorm.weight,
                norm_eps=stage.post_attention_layernorm.variance_epsilon,
            )
        else:
            residual = stage.hc_attn.post_mapping(
                x=attn,
                residual=residual,
                post_layer_mix=post_mix,
                comb_res_mix=comb_mix,
            )
            post_mix, comb_mix, layer_input = stage.hc_ffn.pre_mapping(residual)
            layer_input = stage.post_attention_layernorm(layer_input)
        num_tokens = T * block
        # FUSED_COMM MoE backends (DeepGEMM MegaMoE) size their in-kernel
        # NVLink-barrier chunk loop from ``max(all_rank_num_tokens)`` and index
        # the local slice by ``moe_ep_rank``, so every EP rank must pass the same
        # globally-gathered per-rank list (here: gen tokens = num_gens * block per
        # rank). Passing only the local ``[num_tokens]`` desyncs the phase-flip
        # barrier across ranks (hang / "unspecified launch failure"). Fall back to
        # the local count for single-rank / non-ADP runs where no list is threaded.
        moe_all_rank_num_tokens = (
            all_rank_num_tokens if all_rank_num_tokens is not None else [num_tokens]
        )
        # The draft captured-context MoE must mirror the target DeepseekV4MoE's TP
        # reduction policy. Under attention DP each rank is data-parallel (owns a
        # distinct set of requests) and needs no cross-rank reduction; but under
        # plain tensor parallelism (attention_dp off, tp_size > 1) the expert-sharded
        # MoE output must be all-reduced across ranks -- exactly what the target MoE
        # does (enable_allreduce = not (POST_MOE_FUSION or tp_size == 1)). Previously
        # this was hard-coded to False, which dropped the reduction on the non-ADP
        # path, corrupting the draft block proposals and roughly halving DSpark
        # acceptance length whenever attention_dp was disabled.
        moe_enable_allreduce = (
            not self.model_config.mapping.enable_attention_dp
            and self.model_config.mapping.tp_size > 1
        )
        moe_out = stage.mlp(
            layer_input.reshape(num_tokens, hidden),
            input_ids=moe_input_ids,
            all_rank_num_tokens=moe_all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(enable_allreduce=moe_enable_allreduce),
            do_finalize=True,
        ).reshape(T, block, hidden)
        h = stage.hc_ffn.post_mapping(
            x=moe_out, residual=residual, post_layer_mix=post_mix, comb_res_mix=comb_mix
        )
        return h

    def forward(
        self,
        main_hidden: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        start_pos: int,
        *,
        kv_windows: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        return_logits: bool = False,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> tuple:
        """Full block-draft forward: chain the ``num_stages`` DSpark stages.

        Mirrors the reference ``Transformer.forward_spec`` (generation path,
        ``start_pos > 0``): ``forward_embed`` builds the block input + captured
        context, each stage runs the captured-context backbone, and
        ``forward_head`` emits the block draft tokens + per-position confidence.

        Args:
            main_hidden: ``[T, num_capture * hidden]`` captured target context.
            bonus_token_ids: ``[T]`` last accepted token per request.
            start_pos: absolute decode position (must be > 0).
            kv_windows: optional persistent per-stage rolling captured-context KV
                windows ``[T, num_stages, window_size, head_dim]`` owned by the
                worker; updated in place each call. ``None`` allocates fresh zero
                windows (single-shot golden / test path).
        Returns:
            ``(draft_tokens [T, block], num_proposed [T])`` from ``forward_head``.
        """
        assert start_pos > 0, "DSpark draft runs at generation (start_pos > 0)"
        if getattr(self.mtp_layers[0], "_dspark_attn", None) is None:
            raise RuntimeError(
                "DSpark attention weights not cached; call "
                "cache_attn_weights_from_checkpoint(ckpt_dir, weight_map) after loading."
            )
        x, main_x, draft_ids = self.forward_embed(main_hidden, bonus_token_ids)
        main_x = main_x.unsqueeze(1)  # [T, 1, hidden] for the MQA K/V projection
        freqs_cis = self._dspark_freqs_table(x.device)
        moe_input_ids = draft_ids.reshape(-1)

        h = x
        for s, stage in enumerate(self.mtp_layers):
            stage_window = kv_windows[:, s] if kv_windows is not None else None
            h = self._forward_stage(
                stage,
                h,
                main_x,
                start_pos,
                freqs_cis,
                moe_input_ids,
                stage_window,
                all_rank_num_tokens=all_rank_num_tokens,
            )

        return self.forward_head(
            h,
            bonus_token_ids,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            return_logits=return_logits,
        )

    def forward_batched(
        self,
        main_hidden: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        kv_windows: torch.Tensor,
        slots: torch.Tensor,
        valid_len: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        return_logits: bool = False,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> tuple:
        """CUDA-graph-safe batched block-draft forward (all gen requests at once).

        Same computation as :meth:`forward`, but every host-int / data-dependent
        operation is tensorized so the whole path can be captured into the target's
        CUDA graph (DSpark is a one-engine drafter — its worker runs inside the
        graph). ``start_pos`` is a ``[G]`` tensor (one absolute decode position per
        gen request); the rolling captured-context windows are written/read through
        ``slots`` into the worker-owned ``kv_windows`` buffer; RoPE phases are
        gathered from a fixed table. ``forward_head`` is run with
        ``confidence_threshold == 0`` (the worker proposes the full block), which is
        the graph-safe branch of :func:`dspark_propose`.

        Args:
            main_hidden: ``[G, num_capture * hidden]`` captured target context.
            bonus_token_ids: ``[G]`` last accepted token per gen request.
            start_pos: ``[G]`` int tensor of absolute decode positions (> 0).
            kv_windows: ``[N, num_stages, window_size, head_dim]`` persistent rolling
                windows; written in place through ``slots``.
            slots: ``[G]`` int tensor mapping each request to its ``kv_windows`` row.
            valid_len: ``[G]`` count of actually written rolling-window entries.
        Returns:
            ``(draft_tokens [G, block], num_proposed [G])`` from ``forward_head``.
        """
        if getattr(self.mtp_layers[0], "_dspark_attn", None) is None:
            raise RuntimeError(
                "DSpark attention weights not cached; call "
                "cache_attn_weights_from_checkpoint(ckpt_dir, weight_map) after loading."
            )
        x, main_x, draft_ids = self.forward_embed(main_hidden, bonus_token_ids)
        main_x = main_x.unsqueeze(1)  # [G, 1, hidden] for the MQA K/V projection
        freqs_cis = self._dspark_freqs_table(x.device)
        moe_input_ids = draft_ids.reshape(-1)

        h = x
        for s, stage in enumerate(self.mtp_layers):
            stage_window = kv_windows[:, s]  # [N, window_size, head_dim]
            h = self._forward_stage(
                stage,
                h,
                main_x,
                start_pos,
                freqs_cis,
                moe_input_ids,
                stage_window,
                slots,
                valid_len,
                all_rank_num_tokens=all_rank_num_tokens,
            )

        return self.forward_head(
            h,
            bonus_token_ids,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            return_logits=return_logits,
        )

    def run_moe_lockstep_noop(
        self, all_rank_num_tokens: Optional[List[int]], device: torch.device
    ) -> None:
        """Cross the FUSED_COMM MoE NVLink barrier the same number of times as
        gen-bearing ranks, for an EP rank whose local draft batch is empty.

        DeepGEMM MegaMoE (``scheduler_kind == FUSED_COMM``) synchronizes EP ranks
        with an in-kernel phase-flip NVLink barrier that flips on every kernel
        call, so every rank must invoke the MoE the same number of times or the
        barrier desyncs (hang / "unspecified launch failure"). In the DSpark
        draft only the MoE carries a cross-rank barrier (the captured-context
        attention and the markov/confidence heads are per-rank), so a rank with
        zero local generation requests replays just the per-stage MoE call with a
        single 1-row dummy (its entry in ``all_rank_num_tokens`` is ``1``). The
        scheduler runs its ``max``-derived chunk count, slicing this rank to the
        1 dummy row and zero-padding the remaining chunks, keeping the barrier
        lockstep. No-op when there is no cross-rank work (single-rank / non-ADP,
        or every rank is empty).
        """
        if all_rank_num_tokens is None or max(all_rank_num_tokens) == 0:
            return
        hidden = self.config.hidden_size
        # Use a 1-row dummy, NOT a 0-row tensor: DeepseekV4MoE's router /
        # shared-expert dense GEMMs reject a 0-row input (cuBLAS
        # CUBLAS_STATUS_INVALID_VALUE). The paired ``all_rank_num_tokens`` encodes
        # 1 for this rank, so the FUSED_COMM scheduler slices to this 1 dummy row
        # and still launches ``num_chunks`` cross-rank barrier crossings in
        # lockstep with the gen-bearing ranks.
        dummy_x = torch.zeros((1, hidden), dtype=torch.bfloat16, device=device)
        dummy_ids = torch.zeros((1,), dtype=torch.long, device=device)
        for stage in self.mtp_layers:
            stage.mlp(
                dummy_x,
                input_ids=dummy_ids,
                all_rank_num_tokens=all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(enable_allreduce=False),
                do_finalize=True,
            )

    def forward_head(
        self,
        block_hidden: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        *,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        return_logits: bool = False,
    ) -> tuple:
        """Block-draft head: hc_head + norm + lm_head -> markov refine + confidence.

        ``block_hidden`` is the last stage's mHC residual ``[*, block, hc_mult, hidden]``.
        Returns (draft_tokens [*, block], num_proposed [*]); with ``return_logits``
        also returns the per-position draft logits [*, block, vocab] (§7.9 1-TV).
        """
        last = self.mtp_layers[-1]
        h = last.hc_head(block_hidden)
        h = last.norm(h)
        base_logits = self.lm_head(h)
        return dspark_propose(
            base_logits,
            bonus_token_ids=bonus_token_ids,
            block_hidden=h,
            markov_head=last.markov_head,
            confidence_head=last.confidence_head,
            block_size=self.block_size,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            return_logits=return_logits,
        )


class DSv4DSparkForCausalLM(nn.Module):
    """One-engine draft wrapper for DSpark (mirrors ``DFlashForCausalLM``).

    Wraps :class:`DSv4DSparkDraftModel` (the ``n_mtp_layers``-stage ``mtp.*`` backbone)
    for the single-engine external-drafter flow: created by ``get_draft_model``,
    appended to the target's epilogue, and driven by ``DSv4DSparkWorker``.

    ``embed_tokens`` / ``lm_head`` are shared with the target model
    (:meth:`load_weights_from_target_model`). The draft weights live in the SAME
    checkpoint under ``mtp.*``; :meth:`load_weights` remaps them
    (``remap_dspark_draft_keys``), loads via ``DeepseekV4WeightLoader``, runs the
    fp8 ``post_load_weights`` transforms, and caches the bf16 captured-context
    attention weights from the in-memory state dict.
    """

    def __init__(self, draft_config, aux_stream_dict=None, num_stages=None, block_size=None):
        super().__init__()
        self.dspark_model = DSv4DSparkDraftModel(
            draft_config,
            aux_stream_dict,
            num_stages=num_stages,
            block_size=block_size,
        )
        # Generic handles expected by the loader / weight mappers.
        self.model = self.dspark_model
        self.model_config = draft_config
        self.config = draft_config.pretrained_config
        # Worker-facing interface (the worker receives this wrapper as
        # ``draft_model`` and calls forward()/reads these properties and scalars).
        self.num_stages = self.dspark_model.num_stages
        self._attn_params = self.dspark_model._attn_params
        self.lm_head = None  # shared from the target (load_weights_from_target_model)
        self.logits_processor = None  # set by the caller after construction

    @property
    def block_size(self):
        return self.dspark_model.block_size

    @property
    def embed_tokens(self):
        return self.dspark_model.embed_tokens

    def forward(self, main_hidden, bonus_token_ids, start_pos, **kwargs):
        return self.dspark_model.forward(main_hidden, bonus_token_ids, start_pos, **kwargs)

    def forward_batched(self, main_hidden, bonus_token_ids, start_pos, **kwargs):
        """CUDA-graph-safe batched draft forward (delegates to the draft model)."""
        return self.dspark_model.forward_batched(main_hidden, bonus_token_ids, start_pos, **kwargs)

    def run_moe_lockstep_noop(self, all_rank_num_tokens, device):
        """Empty-batch MoE barrier lockstep (delegates to the draft model)."""
        return self.dspark_model.run_moe_lockstep_noop(all_rank_num_tokens, device)

    def write_context_windows(self, main_hidden, positions, stage_windows):
        """Seed / back-fill the rolling KV windows (delegates to the draft model)."""
        return self.dspark_model.write_context_windows(main_hidden, positions, stage_windows)

    def write_context_windows_batched(self, main_hidden, positions, slots, mask, kv_windows):
        """Batched + masked window back-fill (delegates to the draft model)."""
        return self.dspark_model.write_context_windows_batched(
            main_hidden, positions, slots, mask, kv_windows
        )

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Load the ``mtp.*`` draft weights from the (full) checkpoint dict.

        ``weight_mapper`` is accepted for interface parity with the draft-weight
        loader but unused: DSpark does its own ``mtp.{s}.* -> mtp_layers.{s}.*``
        remap (``remap_dspark_draft_keys``) onto the shared V4 weight loader.
        """
        remapped = remap_dspark_draft_keys(weights, num_stages=self.num_stages)
        logger.info(
            f"[DSpark] loading {len(remapped)} draft params across {self.num_stages} stages"
        )
        DeepseekV4WeightLoader(self.dspark_model).load_weights(remapped)
        self.dspark_model.post_load_weights()
        # bf16 captured-context attention path: dequantize the raw mtp.{s}.attn.*
        # tensors for ``dspark_attention_forward``.
        self.dspark_model.cache_attn_weights_from_state_dict(weights)
        logger.info("[DSpark] draft weight load complete")

    def load_weights_from_target_model(self, target_model):
        """Share the target's embed_tokens / lm_head (DSpark has neither)."""
        if self.dspark_model.embed_tokens is None:
            self.dspark_model.embed_tokens = target_model.model.embed_tokens
        if self.lm_head is None:
            self.lm_head = target_model.lm_head
            self.dspark_model.lm_head = target_model.lm_head


# ----------------------------------------------------------------------------
# Standalone DSpark drafters.
#
# The other DSpark flavour: the drafter ships as its own checkpoint instead of
# living in the target's ``mtp.*`` namespace, so it shares nothing with the
# target but the vocabulary and the captured hidden states. Its block decode is
# DFlash's -- DSpark is DFlash plus a Markov logit bias, a confidence head and
# the shift_label slot convention -- so these subclass ``DFlashForCausalLM`` and
# add exactly those three.
# ----------------------------------------------------------------------------

# Both published drafters (RadixArk/Kimi-K3-DSpark, Inferact/Kimi-K3-DSpark)
# name the head tensors after the submodules that own them: markov_head is a
# VanillaMarkov, confidence_head an AcceptRatePredictor whose linear is ``proj``.
# The bare spellings are kept for drafters exported without that nesting.
_DSPARK_HEAD_WEIGHT_ALIASES = {
    "markov_w1.weight": ("markov_head.markov_w1.weight", "markov_w1.weight"),
    "markov_w2.weight": ("markov_head.markov_w2.weight", "markov_w2.weight"),
    "confidence_proj.weight": ("confidence_head.proj.weight", "confidence_proj.weight"),
    "confidence_proj.bias": ("confidence_head.proj.bias", "confidence_proj.bias"),
}


class GQADSparkForCausalLM(DFlashForCausalLM):
    """DSpark drafter on a GQA-shaped backbone, from a standalone checkpoint.

    Adds the DSpark head set on top of the DFlash block decode:

      - the vanilla Markov intra-block logit bias, applied by ``DSparkWorker``
        through :meth:`apply_markov_chain_logits`;
      - the ``shift_label`` output convention (the hidden state at block slot j
        predicts draft token j+1, so slot 0 holds the anchor token);
      - the confidence head weights.

    Confidence-scheduled verification is not implemented yet: ``confidence_proj``
    is loaded but unused, and drafting always proposes the full K tokens.

    Named for the attention shape, not for a model: the backbone is whatever
    the drafter config resolves to through the model registry, and the
    inherited block decode works for every GQA family the DFlash drafters
    already cover (qwen3, llama, gpt_oss, ...). A per-model subclass would be
    empty. The GQA precondition is inherited, not introduced here -- see
    ``DFlashForCausalLM._validate_gqa_shape``. An MLA-backboned drafter needs
    its own block decode and becomes a sibling, ``MLADSparkForCausalLM``, not a
    subclass of this.

    Reference: arXiv 2607.05147; deepseek-ai/DeepSpec.
    """

    def __init__(self, draft_config, *, dflash_attention_backend: str = "VANILLA"):
        super().__init__(draft_config, dflash_attention_backend=dflash_attention_backend)

        cfg = draft_config.pretrained_config
        # Defaults on, unlike the DFlash base: the shift_label slot layout is
        # part of what DSpark *is*, and both published drafters set
        # block_size == max_draft_len, where the DFlash layout (slots 1..K)
        # runs one slot past the block and reads the next request's anchor.
        # An explicit false still selects the legacy layout.
        shift_label = resolve_dspark_head_config(cfg, "shift_label")
        self._dspark_shift_label = True if shift_label is None else bool(shift_label)
        self._dspark_markov_rank = int(resolve_dspark_head_config(cfg, "markov_rank") or 0)
        self._dspark_markov_head_type = str(
            resolve_dspark_head_config(cfg, "markov_head_type") or "vanilla"
        ).lower()
        self._dspark_use_confidence_head = bool(
            resolve_dspark_head_config(cfg, "use_confidence_head") or False
        )
        # Plain None placeholders rather than nn.Parameter/buffer: the shapes
        # ([vocab, rank]) are checkpoint-dependent, so nothing is pre-allocated
        # and nothing is constructed in the module's default dtype. Using the
        # checkpoint tensor as-is is also what keeps the head in the checkpoint's
        # dtype instead of an nn.Module default. load_weights() fills them in;
        # consumers treat None as "head absent".
        self.markov_w1 = None  # [vocab, rank] (nn.Embedding weight layout)
        self.markov_w2 = None  # [vocab, rank] (nn.Linear(rank->vocab) weight)
        self.confidence_proj_weight = None  # loaded, unused (follow-up MR)
        self.confidence_proj_bias = None

        if self._dspark_markov_rank > 0 and self._dspark_markov_head_type != "vanilla":
            raise ValueError(
                f"DSpark drafter declares markov_head_type="
                f"'{self._dspark_markov_head_type}'; only 'vanilla' is "
                "supported (gated/rnn heads need per-step hidden features)."
            )
        # The block decode only supports the non-causal DSpark convention.
        # Legacy DFlash drafter configs (e.g. Laguna) also carry a causal field
        # and handle it in the legacy decode path, which is why this check lives
        # here rather than in the DFlash base.
        if resolve_dspark_head_config(cfg, "causal"):
            raise ValueError(
                "DSpark drafter sets causal=true; the block decode only "
                "supports the non-causal DSpark convention."
            )
        if self._dspark_use_confidence_head:
            logger.warning(
                "DSpark drafter declares use_confidence_head; "
                "confidence-scheduled verification is not implemented yet "
                "(confidence_proj weights are loaded but unused, drafting "
                "always proposes the full K tokens)."
            )

    @property
    def has_markov_head(self) -> bool:
        return self._dspark_markov_rank > 0 and self.markov_w1 is not None

    def apply_markov_chain_logits(
        self,
        base_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        argmax_fn=None,
        vocab_slice: Optional[slice] = None,
    ) -> torch.Tensor:
        """Apply the vanilla-Markov intra-block bias to block logits.

        No-op (returns ``base_logits`` unchanged) when the checkpoint ships no
        Markov head. See :func:`dspark_markov_chain` for the semantics; when
        ``base_logits`` is a TP vocab shard the caller must pass this rank's
        ``vocab_slice`` (to shard the markov_w2 rows identically) and an
        ``argmax_fn`` returning full-vocab token ids -- ``DFlashWorker`` handles
        both.
        """
        if not self.has_markov_head:
            return base_logits
        markov_w2 = self.markov_w2 if vocab_slice is None else self.markov_w2[vocab_slice]
        return dspark_markov_chain_logits(
            base_logits, first_prev_tokens, self.markov_w1, markov_w2, argmax_fn=argmax_fn
        )

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Take the DSpark head weights, then hand the rest to DFlash.

        The head keys are pulled out before the backbone remap: left in, they
        would pick up a ``model.`` prefix and be dropped by partial loading.
        """
        dspark_weights = {}
        consumed = set()
        for canonical, aliases in _DSPARK_HEAD_WEIGHT_ALIASES.items():
            for name in aliases:
                if name in weights:
                    dspark_weights[canonical] = weights[name]
                    consumed.add(name)
                    break
        if consumed:
            weights = {k: v for k, v in weights.items() if k not in consumed}
        # The inverse of the missing-weights check below. Without it, a config
        # whose head switches this build cannot resolve loads the drafter with
        # the heads silently dropped -- correct output, lower acceptance.
        if self._dspark_markov_rank <= 0 and "markov_w1.weight" in dspark_weights:
            raise ValueError(
                "DSpark drafter ships markov_w1/markov_w2 but markov_rank resolved to 0. "
                "The checkpoint's head switches were not found in dspark_config, "
                "dflash_config, or at the top level; loading it would drop the Markov "
                "head silently."
            )
        if self._dspark_markov_rank > 0:
            vocab = self.config.vocab_size
            rank = self._dspark_markov_rank
            for k in ("markov_w1.weight", "markov_w2.weight"):
                if k not in dspark_weights:
                    raise ValueError(
                        f"DSpark drafter declares markov_rank="
                        f"{self._dspark_markov_rank} but the checkpoint is "
                        f"missing {k}."
                    )
                if tuple(dspark_weights[k].shape) != (vocab, rank):
                    raise ValueError(
                        f"DSpark {k} has shape "
                        f"{tuple(dspark_weights[k].shape)}, expected "
                        f"[vocab, markov_rank] = ({vocab}, {rank})."
                    )
            self.markov_w1 = dspark_weights["markov_w1.weight"].to("cuda")
            self.markov_w2 = dspark_weights["markov_w2.weight"].to("cuda")
        if "confidence_proj.weight" in dspark_weights:
            self.confidence_proj_weight = dspark_weights["confidence_proj.weight"].to("cuda")
        if "confidence_proj.bias" in dspark_weights:
            self.confidence_proj_bias = dspark_weights["confidence_proj.bias"].to("cuda")
        return super().load_weights(weights, weight_mapper=weight_mapper, **kwargs)


def draft_is_embedded_in_target(model_config) -> bool:
    """True when the DSpark draft weights live inside the target checkpoint.

    That is the DeepSeek-V4-Pro layout: the draft is ``mtp.*`` inside the target
    checkpoint and inherits its block definition, EPLB layer namespace and
    quantization.

    The answer comes from ``DSparkDecodingConfig.draft_is_embedded_in_target``
    rather than being re-derived here, because the worker and the spec metadata
    have to make the same call from ``_torch/speculative/`` -- which cannot
    import this package -- and a builder that disagreed with them would hand
    the worker a draft model whose attributes it does not have.
    """
    return bool(model_config.spec_config.draft_is_embedded_in_target)


@register_draft_model(SpeculativeDecodingMode.DSPARK)
def _build_dspark_draft(model_config, draft_config, lm_head, model):
    """Build the DSpark drafter for either flavour.

    Two levels of dispatch:

    1. Are the draft weights embedded in the target checkpoint? If so this is
       the DeepSeek-V4-Pro draft, whose stage count (``n_mtp_layers``) is not in
       the HF config and is derived from the ``mtp.*`` namespace.
    2. Otherwise the drafter is standalone, and its own ``model_type`` selects
       the backbone-specific class.

    Args:
        model_config: the target engine's ``ModelConfig``.
        draft_config: the drafter's own ``ModelConfig``.
        lm_head: unused; DSpark shares the target's head at weight-load time.
        model: the target model, whose aux streams the draft stages reuse.

    Returns:
        The draft ``nn.Module`` for this drafter.
    """
    if draft_is_embedded_in_target(model_config):
        num_stages = count_dspark_stages(model_config.spec_config.speculative_model)
        validate_dspark_eplb_layer_base(model_config, draft_config)
        return DSv4DSparkForCausalLM(
            draft_config,
            getattr(model, "aux_stream_dict", None),
            num_stages=num_stages,
            block_size=model_config.spec_config.block_size,
        )

    # No per-model_type table here. ``DFlashForCausalLM.__init__`` already
    # resolves the backbone from the drafter config through the model registry,
    # so keying on model_type a second time would only duplicate that dispatch
    # and force a new entry for every GQA family that already works. What the
    # table really guarded was the block decode's GQA precondition, which is now
    # checked where it belongs, in the DFlash base. An MLA-backboned drafter
    # (e.g. Inferact/Kimi-K3-DSpark) fails that check with a clear message until
    # ``MLADSparkForCausalLM`` lands as a sibling.
    return GQADSparkForCausalLM(
        draft_config,
        dflash_attention_backend=model_config.spec_config.attention_backend,
    )


__all__ = [
    # Embedded (DeepSeek-V4-Pro) flavour.
    "DSv4DSparkBlock",
    "DSv4DSparkDraftModel",
    "DSv4DSparkForCausalLM",
    # Standalone flavour.
    "GQADSparkForCausalLM",
    "draft_is_embedded_in_target",
    "validate_dspark_eplb_layer_base",
    "validate_dspark_eplb_stage_layers",
    # Captured-context attention primitives.
    "get_dspark_topk_idxs",
    "get_dspark_topk_idxs_batched",
    "dspark_sparse_attn",
    "precompute_dspark_freqs_cis",
    "apply_dspark_rotary",
    "apply_dspark_rotary_batched",
    "dspark_attention_forward",
    "dspark_attention_forward_batched",
    # Block draft I/O.
    "build_draft_input_ids",
    "dspark_propose",
]
