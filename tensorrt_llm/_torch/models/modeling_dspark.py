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
``main_x``, + MoE + mHC) consumes the loaded block-scaled FP8 projections directly by
default when attention projections are unsharded. Plain tensor parallelism
automatically uses dequantized BF16 projections; set
``TRTLLM_DSPARK_LINEAR_BACKEND=bf16`` before model construction to select the same
fallback explicitly.
``forward_embed`` (capture) and ``forward_head`` (block draft) below are the
reference-faithful, unit-validated I/O stages.
"""

import copy
import json
import math
import os
import re
from functools import lru_cache
from typing import Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._utils import get_sm_version, is_sm_100f
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE, IS_CUTLASS_DSL_INTERNAL_AVAILABLE
from ..distributed import AllReduceParams
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.mhc.hyper_connection import HCHead
from ..modules.rms_norm import RMSNorm
from ..pyexecutor.config_utils import is_mla
from ..speculative.interface import SpeculativeDecodingMode
from ..utils import AuxStreamType, swizzle_sf, unswizzle_sf
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
    SpecDecOneEngineForCausalLM,
    build_markov_head,
    confident_prefix_length,
    dspark_markov_chain_logits,
)
from .modeling_utils import DecoderModel, register_auto_model, register_draft_model

if IS_CUTLASS_DSL_INTERNAL_AVAILABLE:
    from ..custom_ops.dspark_attention_custom_op import (
        _run_cute_dsl_dspark_mxfp8_gemm_rubin_prevalidated,
    )

if IS_CUTLASS_DSL_AVAILABLE:
    from ..custom_ops.dspark_attention_custom_op import (
        cute_dsl_dspark_attention,
        cute_dsl_dspark_attention_prepared,
        cute_dsl_dspark_attention_prepared_fp8,
        is_cute_dsl_dspark_attention_prepared_supported,
        is_cute_dsl_dspark_attention_supported,
    )
    from ..custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_concat_fp8_quant,
        cute_dsl_dspark_rmsnorm_fp8_quant,
        cute_dsl_dspark_rmsnorm_rope,
        cute_dsl_dspark_rmsnorm_rope_block_page,
        cute_dsl_dspark_rmsnorm_rope_cache_write,
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
    valid_len: torch.Tensor,
) -> torch.Tensor:
    """Sync-free, fixed-size (CUDA-graph-safe) batched ``get_dspark_topk_idxs``.

    Unlike the scalar :func:`get_dspark_topk_idxs` (whose ``topk`` width
    ``min(window_size, start_pos+1) + block_size`` depends on the host int
    ``start_pos``), this always returns the **fixed** width ``window_size +
    block_size`` and masks the unfilled context slots with ``-1``. The masked
    slots are excluded by :func:`dspark_sparse_attn` exactly as if they were
    absent while the shape remains CUDA-graph safe.

    Every query attends to the actually written circular-window suffix,
    followed by the current-block positions.

    Args:
        window_size: sliding-window length of the captured-context KV cache.
        block_size: number of draft positions per request.
        start_pos: ``[G]`` int tensor of per-request absolute decode positions.
        valid_len: ``[G]`` count of actually written rolling-window entries.

    Returns:
        int32 tensor ``[G, block_size, window_size + block_size]``.
    """
    device = start_pos.device
    g = start_pos.shape[0]
    ctx_cols = torch.arange(window_size, device=device)  # [win]
    # The valid entries are the contiguous logical suffix ending at start_pos,
    # but their physical slots wrap modulo window_size.
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


def _mla_rope_into(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    rope_head_dim: int,
    out_rope_offset: int,
) -> bool:
    """Rotate q's rope tail into ``out[..., out_rope_offset:]``; False if unavailable.

    Returns a bool rather than raising so the caller keeps the eager path for
    builds without the fused kernel, the same way _rmsnorm_rope_batched does.
    """
    if not (IS_CUTLASS_DSL_AVAILABLE and is_sm_100f()):
        return False
    freqs_real = torch.view_as_real(freqs_cis).reshape(-1, freqs_cis.shape[-1], 2)
    if not is_fused_dspark_rmsnorm_rope_supported(
        q, None, freqs_real, num_heads, rope_head_dim, q.shape[-1] - rope_head_dim
    ):
        return False
    torch.ops.trtllm.cute_dsl_dspark_rope_into(
        q, freqs_real, out, num_heads, rope_head_dim, out_rope_offset
    )
    return True


def _rmsnorm_rope_batched(
    t: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    rope_head_dim: int,
    freqs_cis: torch.Tensor,
    *,
    num_heads: int = 1,
    apply_weight: bool = True,
    apply_rmsnorm: bool = True,
    inverse_rope: bool = False,
    norm_dim: Optional[int] = None,
) -> torch.Tensor:
    """Fuse DSpark RMSNorm and last-dimension RoPE when supported.

    ``weight`` may be None when ``apply_weight`` is False -- no path reads it
    then, fused or eager.

    ``norm_dim`` bounds the RMSNorm (and the weight) to a prefix of the row.
    None means the whole row, which is DSpark's own convention. The other
    supported value is ``t.shape[-1] - rope_head_dim``: DeepSeek-style MLA
    normalizes the kv_lora_rank latent and leaves k_pe raw, and then ``weight``
    spans only that latent.
    """
    if IS_CUTLASS_DSL_AVAILABLE and is_sm_100f():
        freqs_real = torch.view_as_real(freqs_cis).reshape(-1, freqs_cis.shape[-1], 2)
        if is_fused_dspark_rmsnorm_rope_supported(
            t, weight, freqs_real, num_heads, rope_head_dim, norm_dim
        ):
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
                norm_dim,
            )

    split_norm = norm_dim is not None and norm_dim != t.shape[-1]
    head, tail = (t[..., :norm_dim], t[..., norm_dim:]) if split_norm else (t, None)
    if apply_rmsnorm:
        if apply_weight:
            head = _rmsnorm(head, weight, eps)
        else:
            head = head * torch.rsqrt(head.square().mean(-1, keepdim=True) + eps)
    elif apply_weight:
        head = (head.float() * weight.float()).to(head.dtype)
    t = head if tail is None else torch.cat([head, tail], dim=-1)
    if rope_head_dim > 0:
        t = _rope_last_dims_batched(t, rope_head_dim, freqs_cis, inverse=inverse_rope)
    return t


def _linear_projection(
    x: torch.Tensor,
    weight_or_projection: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply either a reference tensor weight or a loaded ``Linear`` module."""
    if callable(weight_or_projection):
        return weight_or_projection(x)
    return F.linear(x, weight_or_projection)


def _prequantized_fp8_projection(
    x: tuple[torch.Tensor, torch.Tensor],
    projection: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Run a bind-time-validated Rubin FP8 projection without Linear dispatch."""
    activation, activation_scale = x
    tuning_config = getattr(projection, "_dspark_generation_tuning_config", None)
    return _run_cute_dsl_dspark_mxfp8_gemm_rubin_prevalidated(
        activation,
        projection.weight,
        activation_scale,
        projection.weight_scale,
        materialize_launch=True,
        tuning_config=tuning_config,
    )


def _can_use_fused_fp8_input_projection(
    projection: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    in_features: int,
) -> bool:
    """Return whether a loaded Linear accepts Rubin R128c4 FP8 input."""
    if (
        get_sm_version() != 107
        or not IS_CUTLASS_DSL_INTERNAL_AVAILABLE
        or not callable(projection)
        or in_features % 128 != 0
        or getattr(projection, "in_features", None) != in_features
        or getattr(projection, "has_fp8_block_scales", False) is not True
        or not (
            getattr(projection, "use_cute_dsl_blockscaling_mm", False) is True
            or getattr(projection, "disable_deep_gemm", False) is True
        )
        or getattr(projection, "bias", None) is not None
        or getattr(projection, "lora", None) is not None
        or getattr(projection, "_ugpu_weight_shards", None) is not None
    ):
        return False

    weight = getattr(projection, "weight", None)
    weight_scale = getattr(projection, "weight_scale", None)
    return (
        isinstance(weight, torch.Tensor)
        and weight.dtype == torch.float8_e4m3fn
        and weight.ndim == 2
        and weight.shape[1] == in_features
        and weight.is_contiguous()
        and isinstance(weight_scale, torch.Tensor)
        and weight_scale.dtype == torch.uint8
        and weight_scale.is_contiguous()
    )


def _can_use_fused_fp8_output_projection(
    wo_b: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    o_a_proj: torch.Tensor,
    n_groups: int,
    o_lora_rank: int,
) -> bool:
    """Return whether ``wo_b`` consumes the Rubin R128c4 FP8 layout."""
    if (
        get_sm_version() != 107
        or not IS_CUTLASS_DSL_INTERNAL_AVAILABLE
        or not callable(wo_b)
        or getattr(wo_b, "has_fp8_block_scales", False) is not True
        or not (
            getattr(wo_b, "use_cute_dsl_blockscaling_mm", False) is True
            or getattr(wo_b, "disable_deep_gemm", False) is True
        )
    ):
        return False

    weight = getattr(wo_b, "weight", None)
    weight_scale = getattr(wo_b, "weight_scale", None)
    flat_n = n_groups * o_lora_rank
    return (
        o_a_proj.dtype == torch.float8_e4m3fn
        and o_lora_rank % 128 == 0
        and tuple(o_a_proj.shape[:2]) == (n_groups, o_lora_rank)
        and getattr(wo_b, "in_features", None) == flat_n
        and isinstance(weight, torch.Tensor)
        and weight.dtype == torch.float8_e4m3fn
        and isinstance(weight_scale, torch.Tensor)
        and weight_scale.dtype == torch.uint8
    )


def _qkv_projections(
    x: torch.Tensor,
    main_x: torch.Tensor,
    *,
    wq_a: Optional[torch.Tensor],
    wkv: Optional[torch.Tensor],
    q_norm_w: torch.Tensor,
    qkv_proj: Optional[Callable[[torch.Tensor], torch.Tensor]],
    use_fused_fp8_input: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Q-latent and MQA K/V, fusing q_a+wkv when a module is supplied."""
    if qkv_proj is None:
        if wq_a is None or wkv is None:
            raise ValueError("DSpark attention requires tensor weights or a fused qkv projection")
        return F.linear(x, wq_a), F.linear(x, wkv), F.linear(main_x, wkv)

    q_lora_rank = q_norm_w.shape[0]
    # Group main and draft rows separately so the last-dimension Q/KV splits
    # retain a constant row stride across batches. The views are metadata-only.
    # Rubin folds their concatenation into the projection FP8 input producer;
    # other paths keep the original concatenation.
    main_rows = main_x.view(-1, main_x.shape[-1])
    block_rows = x.view(-1, x.shape[-1])
    if use_fused_fp8_input:
        qkv_input = cute_dsl_dspark_concat_fp8_quant(main_rows, block_rows)
    else:
        qkv_input = torch.cat((main_rows, block_rows), dim=0)
    qkv_rows = (
        _prequantized_fp8_projection(qkv_input, qkv_proj)
        if use_fused_fp8_input
        else qkv_proj(qkv_input)
    )
    main_qkv, qkv = qkv_rows.split((main_rows.shape[0], block_rows.shape[0]), dim=0)
    main_qkv = main_qkv.view(*main_x.shape[:-1], qkv_rows.shape[-1])
    qkv = qkv.view(*x.shape[:-1], qkv_rows.shape[-1])
    q_latent, kv = qkv.split((q_lora_rank, qkv.shape[-1] - q_lora_rank), dim=-1)
    main_kv = main_qkv[..., q_lora_rank:]
    return q_latent, kv, main_kv


def _q_projection(
    q_latent: torch.Tensor,
    q_norm_w: torch.Tensor,
    eps: float,
    wq_b: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    use_fused_fp8_input: bool,
    batched_freqs_cis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply q-latent RMSNorm and q_b, optionally sharing one FP8 producer."""
    if use_fused_fp8_input:
        q_input = cute_dsl_dspark_rmsnorm_fp8_quant(q_latent, q_norm_w, eps)
        q = _prequantized_fp8_projection(q_input, wq_b)
        return q.view(*q_latent.shape[:-1], q.shape[-1])

    if batched_freqs_cis is not None:
        return _linear_projection(
            _rmsnorm_rope_batched(q_latent, q_norm_w, eps, 0, batched_freqs_cis),
            wq_b,
        )
    q = _rmsnorm(q_latent, q_norm_w, eps)
    return _linear_projection(q, wq_b)


def _output_projection(
    o: torch.Tensor,
    *,
    wo_a: Optional[torch.Tensor],
    wo_b: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    n_groups: int,
    o_lora_rank: int,
    o_a_proj: Optional[torch.Tensor],
    o_a_scale: Optional[torch.Tensor],
    o_a_dequant: Optional[torch.Tensor],
    use_fused_output_projection: Optional[bool] = None,
) -> torch.Tensor:
    """Apply the grouped o_a projection followed by o_b."""
    batch, block = o.shape[:2]
    if o_a_proj is None:
        if wo_a is None:
            raise ValueError("DSpark attention requires a tensor wo_a or a loaded FP8 o_a")
        o = o.reshape(batch, block, n_groups, -1)
        wo_a_v = wo_a.view(n_groups, o_lora_rank, -1)
        o_lora = torch.einsum("bsgd,grd->bsgr", o, wo_a_v).flatten(2)
        return _linear_projection(o_lora, wo_b)

    if o_a_proj.dtype != torch.float8_e4m3fn:
        grouped_o = o.reshape(batch * block, n_groups, -1).transpose(0, 1)
        o_lora = torch.bmm(grouped_o, o_a_proj.transpose(1, 2))
        o_lora = o_lora.transpose(0, 1).reshape(batch, block, n_groups * o_lora_rank)
        return _linear_projection(o_lora, wo_b)

    if o_a_scale is None:
        raise ValueError("The DSpark FP8 o_a projection requires block scales")

    grouped_o = o.reshape(batch * block, n_groups, -1)
    if use_fused_output_projection is None:
        use_fused_output_projection = _can_use_fused_fp8_output_projection(
            wo_b, o_a_proj, n_groups, o_lora_rank
        )
    if use_fused_output_projection:
        grouped_o_fp8, grouped_o_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
            grouped_o
        )
        return _fused_fp8_output_projection_from_quantized(
            grouped_o_fp8,
            grouped_o_scale,
            batch=batch,
            block=block,
            wo_b=wo_b,
            n_groups=n_groups,
            o_lora_rank=o_lora_rank,
            o_a_proj=o_a_proj,
            o_a_scale=o_a_scale,
        )

    from ..modules.mla import fp8_block_scaling_bmm_out

    o_lora = o.new_empty((n_groups, batch * block, o_lora_rank))
    fp8_block_scaling_bmm_out(
        grouped_o,
        o_a_proj,
        o_a_scale,
        o_lora,
        mat2_dequant=o_a_dequant,
        use_cute_dsl_blockscaling_bmm=True,
    )
    o_lora = o_lora.transpose(0, 1).reshape(batch, block, n_groups * o_lora_rank)
    return _linear_projection(o_lora, wo_b)


def _fused_fp8_output_projection_from_quantized(
    grouped_o_fp8: torch.Tensor,
    grouped_o_scale: torch.Tensor,
    *,
    batch: int,
    block: int,
    wo_b: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    n_groups: int,
    o_lora_rank: int,
    o_a_proj: torch.Tensor,
    o_a_scale: torch.Tensor,
) -> torch.Tensor:
    """Consume group-major FP8 attention output in the fused OA/OB path."""
    rows = batch * block
    flat_n = n_groups * o_lora_rank
    fp8_output = grouped_o_fp8.new_empty((rows, flat_n))
    fp8_batched = fp8_output.view(rows, n_groups, o_lora_rank).transpose(0, 1)
    scale_m = (rows + 127) // 128 * 128
    scale_k = ((flat_n + 31) // 32 + 3) // 4 * 4
    packed_scale = torch.empty(
        (scale_m * scale_k,),
        dtype=torch.uint8,
        device=grouped_o_fp8.device,
    )
    max_batch = getattr(wo_b, "_dspark_generation_max_batch", 0)
    rows_per_request = getattr(wo_b, "_dspark_generation_rows_per_request", 0)
    torch.ops.trtllm.cute_dsl_dspark_fp8_bmm_quantize_rubin_out(
        grouped_o_fp8,
        o_a_proj,
        grouped_o_scale,
        o_a_scale,
        fp8_batched,
        packed_scale,
        True,
        max_batch,
        rows_per_request,
    )
    projected = _prequantized_fp8_projection((fp8_output, packed_scale), wo_b)
    return projected.view(batch, block, projected.shape[-1])


def dspark_attention_forward(
    x: torch.Tensor,
    main_x: torch.Tensor,
    start_pos: int,
    kv_cache: torch.Tensor,
    *,
    wq_a: Optional[torch.Tensor],
    q_norm_w: torch.Tensor,
    wq_b: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    wkv: Optional[torch.Tensor],
    kv_norm_w: torch.Tensor,
    wo_a: Optional[torch.Tensor],
    wo_b: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
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
    qkv_proj: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    o_a_proj: Optional[torch.Tensor] = None,
    o_a_scale: Optional[torch.Tensor] = None,
    o_a_dequant: Optional[torch.Tensor] = None,
    use_fused_output_projection: Optional[bool] = None,
    use_fused_qkv_input_projection: bool = False,
    use_fused_q_input_projection: bool = False,
) -> torch.Tensor:
    """Captured-context DSpark draft attention (generation path, ``start_pos > 0``).

    Functional port of DeepSpec ``DSparkAttention.forward`` for the dense
    (``compress_ratio == 0``) draft: low-rank Q (``wq_a`` -> ``q_norm`` -> ``wq_b``)
    with a per-head RMS + RoPE, MQA K/V from ``wkv`` (shared across heads), keys
    gathered from a rolling captured-context window (``kv_cache``, into which the
    projected ``main_x`` context is written at ``start_pos % window_size``) plus the
    block's own positions, attention-sink softmax, inverse-RoPE on the output, and a
    grouped low-rank O projection (``wo_a`` einsum + ``wo_b``).

    The reference path accepts plain BF16 tensor weights. Production passes the
    loaded FP8 ``Linear`` modules plus the native grouped o_a weight and scale, so
    all five projections dispatch through block-scaled CuTe DSL kernels on Rubin.

    Returns:
        ``[b, block_size, dim]`` attention output (residual stream contribution).
    """
    assert start_pos > 0, "DSpark draft attention runs at generation (start_pos > 0)"
    b, block, _ = x.shape
    rd = rope_head_dim
    main_freqs = freqs_cis[start_pos : start_pos + 1]
    blk_freqs = freqs_cis[start_pos + 1 : start_pos + 1 + block]

    q_latent, kv, main_kv = _qkv_projections(
        x,
        main_x,
        wq_a=wq_a,
        wkv=wkv,
        q_norm_w=q_norm_w,
        qkv_proj=qkv_proj,
        use_fused_fp8_input=use_fused_qkv_input_projection,
    )

    # Captured-context K/V from main_x (MQA, shared across heads).
    main_kv = _rmsnorm(main_kv, kv_norm_w, eps)  # [b, 1, head_dim]
    main_kv = _rope_last_dims(main_kv, rd, main_freqs)

    # Query: low-rank + per-head RMS + RoPE.
    q = _q_projection(q_latent, q_norm_w, eps, wq_b, use_fused_q_input_projection).unflatten(
        -1, (n_heads, head_dim)
    )
    # Per-head RMS in the query dtype (matches the reference inline normalization,
    # which is NOT the fp32 RMSNorm path).
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    q = _rope_last_dims(q, rd, blk_freqs)

    # Block K/V.
    kv = _rmsnorm(kv, kv_norm_w, eps)  # [b, block, head_dim]
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

    return _output_projection(
        o,
        wo_a=wo_a,
        wo_b=wo_b,
        n_groups=n_groups,
        o_lora_rank=o_lora_rank,
        o_a_proj=o_a_proj,
        o_a_scale=o_a_scale,
        o_a_dequant=o_a_dequant,
        use_fused_output_projection=use_fused_output_projection,
    )


def dspark_attention_forward_batched(
    x: torch.Tensor,
    main_x: torch.Tensor,
    start_pos: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    valid_len: torch.Tensor,
    *,
    wq_a: Optional[torch.Tensor],
    q_norm_w: torch.Tensor,
    wq_b: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    wkv: Optional[torch.Tensor],
    kv_norm_w: torch.Tensor,
    wo_a: Optional[torch.Tensor],
    wo_b: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
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
    qkv_proj: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    o_a_proj: Optional[torch.Tensor] = None,
    o_a_scale: Optional[torch.Tensor] = None,
    o_a_dequant: Optional[torch.Tensor] = None,
    use_fused_output_projection: Optional[bool] = None,
    use_fused_qkv_input_projection: bool = False,
    use_fused_q_input_projection: bool = False,
) -> torch.Tensor:
    """Batched, CUDA-graph-safe captured-context DSpark draft attention.

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
        valid_len: ``[G]`` count of actually written context entries; masks holes left when
            absolute positions are bootstrapped without the corresponding rolling-window state.
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

    q_latent, block_kv_input, main_kv_input = _qkv_projections(
        x,
        main_x,
        wq_a=wq_a,
        wkv=wkv,
        q_norm_w=q_norm_w,
        qkv_proj=qkv_proj,
        use_fused_fp8_input=use_fused_qkv_input_projection,
    )

    # Query: low-rank + per-head RMS + RoPE.
    q = _q_projection(
        q_latent, q_norm_w, eps, wq_b, use_fused_q_input_projection, blk_freqs
    ).unflatten(-1, (n_heads, head_dim))
    # No gain: DSpark's per-head query norm normalizes and stops there. The
    # None used to be kv_norm_w -- a KV weight handed to a query call, which
    # only ever passed review because nothing reads it.
    q = _rmsnorm_rope_batched(
        q,
        None,
        eps,
        rd,
        blk_freqs,
        num_heads=n_heads,
        apply_weight=False,
    )
    # ``persist=True`` writes through to the worker-owned rolling window;
    # single-shot callers keep the original functional clone behavior.
    write_target = kv_cache if persist else kv_cache.clone()
    main_rope_freqs = torch.view_as_real(main_freqs).reshape(-1, rd // 2, 2).contiguous()
    inverse_rope_freqs = torch.view_as_real(blk_freqs).contiguous()
    block_rope_freqs = inverse_rope_freqs.reshape(-1, rd // 2, 2)
    if use_fused_output_projection is None:
        use_fused_output_projection = o_a_proj is not None and _can_use_fused_fp8_output_projection(
            wo_b, o_a_proj, n_groups, o_lora_rank
        )

    # The fused kernel consumes the number of initialized physical-window rows,
    # including short suffixes on disaggregated generation workers. Legacy
    # callers without explicit validity have a dense suffix, so synthesize the
    # equivalent length from their absolute decode position.
    kernel_valid_len = (
        valid_len if valid_len is not None else torch.clamp(start_pos + 1, max=window_size)
    )

    use_prepared_attention = (
        IS_CUTLASS_DSL_AVAILABLE
        and is_cute_dsl_dspark_attention_prepared_supported(
            q, write_target, kernel_valid_len, attn_sink, inverse_rope_freqs
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
    if use_prepared_attention:
        slots_i32, cache_seqs = cute_dsl_dspark_rmsnorm_rope_cache_write(
            main_kv_input,
            kv_norm_w,
            main_rope_freqs,
            write_target,
            slots,
            start_pos,
            eps,
        )
        block_page = cute_dsl_dspark_rmsnorm_rope_block_page(
            block_kv_input,
            kv_norm_w,
            block_rope_freqs,
            eps,
        )
        if use_fused_output_projection:
            if o_a_proj is None or o_a_scale is None:
                raise ValueError("The fused DSpark FP8 output path requires o_a weights and scales")
            grouped_o_fp8, grouped_o_scale = cute_dsl_dspark_attention_prepared_fp8(
                q,
                block_page,
                write_target,
                slots_i32,
                cache_seqs,
                kernel_valid_len,
                attn_sink,
                inverse_rope_freqs,
                softmax_scale,
                n_groups,
            )
            return _fused_fp8_output_projection_from_quantized(
                grouped_o_fp8,
                grouped_o_scale,
                batch=g,
                block=block,
                wo_b=wo_b,
                n_groups=n_groups,
                o_lora_rank=o_lora_rank,
                o_a_proj=o_a_proj,
                o_a_scale=o_a_scale,
            )
        o = cute_dsl_dspark_attention_prepared(
            q,
            block_page,
            write_target,
            slots_i32,
            cache_seqs,
            kernel_valid_len,
            attn_sink,
            inverse_rope_freqs,
            softmax_scale,
        )
    else:
        main_kv = _rmsnorm_rope_batched(main_kv_input, kv_norm_w, eps, rd, main_freqs)
        kv = _rmsnorm_rope_batched(block_kv_input, kv_norm_w, eps, rd, blk_freqs)
        main_kv_flat = main_kv.squeeze(1).to(write_target.dtype)

        # Compatibility path for direct/int32 callers; production uses
        # preparation.
        if IS_CUTLASS_DSL_AVAILABLE and is_cute_dsl_dspark_attention_supported(
            q,
            main_kv_flat,
            kv,
            write_target,
            slots,
            start_pos,
            kernel_valid_len,
            attn_sink,
            inverse_rope_freqs,
        ):
            o = cute_dsl_dspark_attention(
                q,
                main_kv_flat,
                kv,
                write_target,
                slots,
                start_pos,
                kernel_valid_len,
                attn_sink,
                inverse_rope_freqs,
                softmax_scale,
            )
        else:
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
                None,
                eps,
                rd,
                blk_freqs,
                num_heads=n_heads,
                apply_weight=False,
                apply_rmsnorm=False,
                inverse_rope=True,
            )

    return _output_projection(
        o,
        wo_a=wo_a,
        wo_b=wo_b,
        n_groups=n_groups,
        o_lora_rank=o_lora_rank,
        o_a_proj=o_a_proj,
        o_a_scale=o_a_scale,
        o_a_dequant=o_a_dequant,
        use_fused_output_projection=use_fused_output_projection,
    )


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
_DSPARK_LINEAR_BACKEND_ENV = "TRTLLM_DSPARK_LINEAR_BACKEND"


def _use_dspark_bf16_linear(mapping) -> bool:
    """Select BF16 for an explicit override or sharded attention projections."""
    backend = os.environ.get(_DSPARK_LINEAR_BACKEND_ENV, "fp8").lower()
    if backend not in ("fp8", "bf16"):
        raise ValueError(f"{_DSPARK_LINEAR_BACKEND_ENV} must be 'fp8' or 'bf16', got {backend!r}")
    return backend == "bf16" or (mapping.tp_size > 1 and not mapping.enable_attention_dp)


def _stack_fp8_projection_output_rows(
    projections: List[Linear], output_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack trailing output rows and rebuild their shared R128c4 scales."""
    if output_rows <= 0 or output_rows % 128 != 0:
        raise ValueError(
            f"DSpark stacked FP8 output_rows must be a positive multiple of 128, got {output_rows}"
        )

    weights = []
    linear_scales = []
    for projection in projections:
        weight = projection.weight
        if (
            weight.ndim != 2
            or weight.shape[0] % 128 != 0
            or weight.shape[1] % 128 != 0
            or output_rows > weight.shape[0]
        ):
            raise ValueError(
                "DSpark stacked FP8 projections require 2-D 128-aligned weights "
                f"with at least {output_rows} rows, got shape {tuple(weight.shape)}"
            )
        linear_scale = unswizzle_sf(
            projection.weight_scale,
            weight.shape[0],
            weight.shape[1],
            scaling_vector_size=32,
        )
        weights.append(weight[-output_rows:])
        linear_scales.append(linear_scale[-output_rows:])

    stacked_weight = torch.cat(weights, dim=0).contiguous()
    stacked_scale = swizzle_sf(
        torch.cat(linear_scales, dim=0),
        len(projections) * output_rows,
        stacked_weight.shape[1],
        scaling_vector_size=32,
    )
    return stacked_weight, stacked_scale


def _active_moe_load_balancer():
    """The engine-wide ``MoeLoadBalancer``, or None when EPLB is not active.

    Non-None exactly inside ``maybe_create_moe_load_balancer(...)`` when EPLB is
    really enabled for this engine (supported arch, ``moe_ep_size > 1``, no smart
    router, ``moe_load_balancer`` configured) -- i.e. exactly the condition under
    which ``MoE._init_load_balancer`` consumes ``model_config.moe_load_balancer``.
    Gating every DSpark EPLB check on it keeps the non-EPLB path untouched.
    """
    from ..modules.fused_moe.moe_load_balancer import get_moe_load_balancer

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
                use_cute_dsl_blockscaling_mm=model_config.use_cute_dsl_blockscaling_mm,
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


def _runtime_position_cap(model_config, pretrained_config, slack: int = 0) -> int:
    """Fallback RoPE table length: the served length, not the advertised one.

    Both drafter families build a position table, and both were sized from the
    checkpoint's max_position_embeddings. K3 advertises 1,048,576, which costs
    ~256 MiB per rank as complex64 -- built via full-size fp32 cos/sin, so
    ~768 MiB transient -- while the context cache is bounded by the runtime
    max_seq_len. Only the table length is affected; YaRN's correction range is
    computed from original_max_position_embeddings and does not move.

    Shrinking the table this way is also what made its bound matter: sized from
    max_position_embeddings it could not be indexed past, sized from
    model_config.max_seq_len it can. Hence _runtime_position_ceiling, which
    every drafter prefers over this value whenever a worker has published it.
    """
    runtime = getattr(model_config, "max_seq_len", None)
    return int(runtime or getattr(pretrained_config, "max_position_embeddings", 163840)) + slack


# NOTE on slack: this is a construction-time FALLBACK only, for direct
# construction in tests. Every drafter's real bound comes at runtime from
# dflash_position_ceiling (published as _runtime_position_ceiling), because the
# engine raises max_seq_len past model_config's value and never writes it back.
# DSv4 needs it too: not being driven by DFlashWorker keeps its ctx_len from
# being clamped to the engine's value, but its positions still come from the
# target's position_ids, which are bounded by exactly that value.


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
        # Fallback: _dspark_freqs_table prefers _runtime_position_ceiling.
        self._freqs_cap = _runtime_position_cap(model_config, config, self.block_size + 2)
        self._freqs_table_cache: Dict = {}
        self._context_wkv_fp8 = None
        self._use_bf16_linear = _use_dspark_bf16_linear(model_config.mapping)

    def post_load_weights(self, weights: Optional[Dict] = None) -> None:
        """Transform draft linears and bind the selected attention backend."""
        # DSparkForCausalLM.load_weights supplies the complete raw draft
        # checkpoint before the generic zero-argument post-load walk. Only that
        # explicit draft load owns these derived projections; a target-model
        # reload does not reload the separate draft model.
        if weights is None:
            return

        # The separately constructed draft config does not receive the target
        # model's SM107 CuTe block-scaling default. Select the same backend for
        # the draft attention linears before their one-time weight transform;
        # otherwise they materialize DeepGEMM int32 scales and cannot consume
        # the fused Rubin attention/projection layouts.
        if not self._use_bf16_linear and get_sm_version() == 107:
            for stage in self.mtp_layers:
                for module in stage.self_attn.modules():
                    if isinstance(module, Linear):
                        module.use_cute_dsl_blockscaling_mm = True

        attn_linear_ids = set()
        if self._use_bf16_linear:
            for stage in self.mtp_layers:
                for module in stage.self_attn.modules():
                    if isinstance(module, Linear):
                        attn_linear_ids.add(id(module))

        for module in self.modules():
            if isinstance(module, Linear) and id(module) not in attn_linear_ids:
                module.transform_weights()
        if self._use_bf16_linear:
            self._bind_bf16_attn_projections(weights)
        else:
            self._bind_attn_projections()

    @staticmethod
    def _block_dequant(
        weight_fp8: torch.Tensor, scale: torch.Tensor, block: int = 128
    ) -> torch.Tensor:
        """Dequantize a block-scaled FP8 checkpoint weight to BF16."""
        weight = weight_fp8.float()
        output_features, input_features = weight.shape
        scale = scale.float()
        scale = scale.repeat_interleave(block, 0)[:output_features]
        scale = scale.repeat_interleave(block, 1)[:, :input_features]
        return (weight * scale).to(torch.bfloat16)

    def _bind_bf16_attn_projections(self, weights: Dict) -> None:
        """Bind all five captured-context attention projections as BF16 tensors."""
        for stage_id, stage in enumerate(self.mtp_layers):
            prefix = f"mtp.{stage_id}.attn."
            device = stage.input_layernorm.weight.device

            def dequant(name: str) -> torch.Tensor:
                weight = weights[f"{prefix}{name}.weight"].to(device)
                scale = weights[f"{prefix}{name}.scale"].to(device)
                return self._block_dequant(weight, scale)

            stage._dspark_attn = dict(
                wq_a=dequant("wq_a"),
                q_norm_w=weights[f"{prefix}q_norm.weight"].to(device=device, dtype=torch.bfloat16),
                wq_b=dequant("wq_b"),
                wkv=dequant("wkv"),
                kv_norm_w=weights[f"{prefix}kv_norm.weight"].to(
                    device=device, dtype=torch.bfloat16
                ),
                wo_a=dequant("wo_a"),
                wo_b=dequant("wo_b"),
                attn_sink=weights[f"{prefix}attn_sink"].to(device=device, dtype=torch.float32),
                qkv_proj=None,
                o_a_proj=None,
                o_a_scale=None,
                o_a_dequant=None,
                use_fused_output_projection=False,
                use_fused_qkv_input_projection=False,
                use_fused_q_input_projection=False,
            )
        self._context_wkv_fp8 = None

    def _bind_attn_projections(self) -> None:
        """Bind loaded MLA modules to the DSpark attention functional path.

        DeepSeek-V4 loads q_a+wkv as one block-scaled FP8 ``Linear`` and keeps
        o_a as a grouped FP8 weight. Reusing those objects avoids a second BF16
        copy of every attention projection and preserves the Rubin CuTe DSL
        weight/scale layouts produced by ``Linear.transform_weights``.
        """
        expected_heads = self._attn_params["n_heads"]
        expected_groups = self._attn_params["n_groups"]
        head_dim = self._attn_params["head_dim"]
        for stage in self.mtp_layers:
            attn = stage.self_attn
            if attn.num_heads_tp != expected_heads or attn.n_local_groups != expected_groups:
                raise ValueError(
                    "DSpark FP8 attention requires unsharded attention projections; "
                    f"got {attn.num_heads_tp} heads and {attn.n_local_groups} groups, "
                    f"expected {expected_heads} heads and {expected_groups} groups. "
                    "Enable attention DP (or use TP=1)."
                )
            qkv_in_features = attn.kv_a_proj_with_mqa.in_features
            q_lora_rank = attn.q_a_layernorm.weight.shape[0]
            use_fused_qkv_input_projection = _can_use_fused_fp8_input_projection(
                attn.kv_a_proj_with_mqa, qkv_in_features
            )
            use_fused_q_input_projection = _can_use_fused_fp8_input_projection(
                attn.q_b_proj, q_lora_rank
            )
            if use_fused_qkv_input_projection or use_fused_q_input_projection:
                from ..custom_ops.dspark_rmsnorm_rope_custom_op import (
                    precompile_dspark_fp8_input_fusions,
                )

                precompile_dspark_fp8_input_fusions(
                    qkv_in_features if use_fused_qkv_input_projection else None,
                    q_lora_rank if use_fused_q_input_projection else None,
                    self._attn_params["eps"],
                )
            stage._dspark_attn = dict(
                wq_a=None,
                q_norm_w=attn.q_a_layernorm.weight,
                wq_b=attn.q_b_proj,
                wkv=None,
                kv_norm_w=attn.kv_a_layernorm.weight,
                wo_a=None,
                wo_b=attn.o_b_proj,
                attn_sink=attn.mqa.attn_sink,
                qkv_proj=attn.kv_a_proj_with_mqa,
                o_a_proj=attn.o_a_proj,
                o_a_scale=attn.o_a_proj_scale,
                o_a_dequant=attn.o_a_proj_dequant,
                use_fused_output_projection=_can_use_fused_fp8_output_projection(
                    attn.o_b_proj,
                    attn.o_a_proj,
                    attn.n_local_groups,
                    attn.o_a_proj.shape[1],
                ),
                use_fused_qkv_input_projection=use_fused_qkv_input_projection,
                use_fused_q_input_projection=use_fused_q_input_projection,
            )
        # Generation consumes q_a+wkv, while context-window seeding needs only
        # the trailing wkv rows. Stack the three transformed WKV weights once so
        # CTX runs one N=(stages*head_dim) GEMM instead of three small-N GEMMs.
        # Scale layout conversion and the weight copy never enter serving.
        self._context_wkv_fp8 = (
            _stack_fp8_projection_output_rows(
                [stage.self_attn.kv_a_proj_with_mqa for stage in self.mtp_layers],
                head_dim,
            )
            if self.mtp_layers
            and all(
                stage._dspark_attn["use_fused_qkv_input_projection"] for stage in self.mtp_layers
            )
            else None
        )

    def configure_generation_autotuner(self, max_batch: int) -> None:
        """Give DSpark projection runners the worker's real generation capacity."""
        if self._use_bf16_linear:
            return
        if max_batch < 1:
            raise ValueError(f"DSpark max_batch must be positive, got {max_batch}")

        from ..custom_ops.dspark_attention_custom_op import (
            _get_dspark_generation_mxfp8_tuning_config,
        )

        projection_rows = (
            ("qkv_proj", self.block_size + 1),
            ("wq_b", self.block_size),
            ("wo_b", self.block_size),
        )
        for stage in self.mtp_layers:
            for name, rows_per_request in projection_rows:
                projection = stage._dspark_attn[name]
                projection._dspark_generation_tuning_config = (
                    _get_dspark_generation_mxfp8_tuning_config(
                        rows_per_request,
                        max_batch,
                    )
                )
                if name == "wo_b":
                    projection._dspark_generation_max_batch = max_batch
                    projection._dspark_generation_rows_per_request = rows_per_request

    def _dspark_freqs_table(self, device: torch.device) -> torch.Tensor:
        """Return the plain-RoPE table cached for ``(device, cap)``.

        Sized from the ceiling the worker publishes once it knows the runtime
        one, because that is what the decode positions actually reach: they come
        from the target's position_ids, and py_executor_creator raises the
        engine's max_seq_len past model_config's value without writing it back.
        ``_freqs_cap`` is the fallback for direct construction in tests, where no
        worker runs. MLADSparkForCausalLM reads the same attribute in
        ``_mla_freqs_cis``.
        """
        cap = int(getattr(self, "_runtime_position_ceiling", None) or self._freqs_cap)
        key = (str(device), cap)
        cached = self._freqs_table_cache.get(key)
        if cached is None:
            cached = precompute_dspark_freqs_cis(
                self._attn_params["rope_head_dim"],
                cap,
                rope_theta=self._rope_theta,
                device=device,
            )
            self._freqs_table_cache[key] = cached
        return cached

    @classmethod
    def _derive_draft_model_config(cls, model_config, base: int, num_stages: int):
        """Return a draft-only ``model_config`` copy with draft-specific fixes.

        Applies (1) the ``compress_ratios`` draft slice, (2) the
        ``quant_config_dict`` MXFP4 extension for the draft layers' routed
        experts, and (3) default EPLB assignments for draft layers omitted from
        a target-only assignment file. A single shallow copy is made (and only
        when something needs to change) so the shared ``model_config`` and the
        target model are untouched.

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
        new_mlb = cls._draft_moe_load_balancer_config(model_config, base, num_stages)
        if new_sa is None and new_qcd is None and new_qc is None and new_mlb is None:
            return model_config
        draft_cfg = copy.copy(model_config)
        # ModelConfig is a frozen dataclass; bypass the guard for these fields.
        if new_sa is not None:
            object.__setattr__(draft_cfg, "sparse_attention_config", new_sa)
        if new_qcd is not None:
            object.__setattr__(draft_cfg, "quant_config_dict", new_qcd)
        if new_qc is not None:
            object.__setattr__(draft_cfg, "quant_config", new_qc)
        if new_mlb is not None:
            object.__setattr__(draft_cfg, "moe_load_balancer", new_mlb)
        return draft_cfg

    @staticmethod
    def _draft_moe_load_balancer_config(model_config, base: int, num_stages: int):
        """EPLB config with deterministic defaults for omitted draft layers."""
        load_balancer = getattr(model_config, "moe_load_balancer", None)
        assignments = (
            getattr(load_balancer, "initial_global_assignments", None)
            if load_balancer is not None
            else None
        )
        if assignments is None or all(base + stage in assignments for stage in range(num_stages)):
            return None

        num_experts = model_config.pretrained_config.n_routed_experts
        ep_size = model_config.mapping.moe_ep_size
        local_slots = load_balancer.num_slots // ep_size
        default_assignments = [
            (ep_rank * num_experts // ep_size + local_slot) % num_experts
            for ep_rank in range(ep_size)
            for local_slot in range(local_slots)
        ]
        draft_assignments = dict(assignments)
        for stage in range(num_stages):
            draft_assignments.setdefault(base + stage, default_assignments)
        return load_balancer.model_copy(update={"initial_global_assignments": draft_assignments})

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

    def _project_context_wkv(self, main_x: torch.Tensor) -> Optional[torch.Tensor]:
        """Quantize once and project every stage's WKV with one stacked GEMM."""
        if self._context_wkv_fp8 is None:
            return None
        rows = main_x.reshape(-1, main_x.shape[-1])
        input_fp8, input_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(rows)
        weight, weight_scale = self._context_wkv_fp8
        output = torch.ops.trtllm.cute_dsl_dspark_context_wkv_gemm_rubin(
            input_fp8, weight, input_scale, weight_scale
        )
        head_dim = self._attn_params["head_dim"]
        return output.view(*main_x.shape[:-1], len(self.mtp_layers), head_dim)

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
        head_dim = int(self._attn_params["head_dim"])
        rd = int(self._attn_params["rope_head_dim"])
        eps = float(self._attn_params["eps"])
        positions = positions.to(main_hidden.device).long()
        # main_x is stage-invariant (stage 0's projection), matching forward_embed.
        stage0 = self.mtp_layers[0]
        main_x = stage0.main_norm(stage0.main_proj(main_hidden))  # [M, hidden]
        freqs = self._dspark_freqs_table(main_x.device)[positions]
        slots = positions % win  # [M]
        mx = main_x.unsqueeze(0)  # [1, M, hidden] for the per-position RoPE layout
        projected_wkv = self._project_context_wkv(mx)
        for s, stage in enumerate(self.mtp_layers):
            a = stage._dspark_attn
            kv = (
                projected_wkv[..., s, :]
                if projected_wkv is not None
                else (
                    a["qkv_proj"](mx)[..., -head_dim:]
                    if a["qkv_proj"] is not None
                    else F.linear(mx, a["wkv"])
                )
            )
            kv = _rmsnorm(kv, a["kv_norm_w"], eps)  # [1, M, head_dim]
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
        head_dim = int(self._attn_params["head_dim"])
        rd = int(self._attn_params["rope_head_dim"])
        eps = float(self._attn_params["eps"])
        positions = positions.long()
        slots = slots.long()
        safe_positions = torch.where(mask, positions, torch.zeros_like(positions))
        freqs = self._dspark_freqs_table(main_hidden.device)[safe_positions]  # [G, M, rd//2]
        cols = positions % win  # [G, M]
        rows = slots[:, None].expand(-1, M)  # [G, M]
        mask3 = mask.unsqueeze(-1)  # [G, M, 1]
        stage0 = self.mtp_layers[0]
        main_x = stage0.main_norm(stage0.main_proj(main_hidden))  # [G, M, hidden]
        projected_wkv = self._project_context_wkv(main_x)
        for s, stage in enumerate(self.mtp_layers):
            a = stage._dspark_attn
            kv = (
                projected_wkv[..., s, :]
                if projected_wkv is not None
                else (
                    a["qkv_proj"](main_x)[..., -head_dim:]
                    if a["qkv_proj"] is not None
                    else F.linear(main_x, a["wkv"])
                )
            )
            kv = _rmsnorm(kv, a["kv_norm_w"], eps)  # [G, M, head_dim]
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
            if valid_len is None:
                raise ValueError(
                    "DSpark batched attention requires valid_len for each rolling window"
                )
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
                "DSpark attention projections are not bound; call post_load_weights() after loading."
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
        valid_len: torch.Tensor,
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
                "DSpark attention projections are not bound; call post_load_weights() after loading."
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
    selected post-load transforms, and binds the captured-context path to the
    selected attention projection backend.
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

    def configure_generation_autotuner(self, max_batch):
        """Delegate generation autotuner capacity to the DSpark model."""
        self.dspark_model.configure_generation_autotuner(max_batch)

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
        self.dspark_model.post_load_weights(weights)
        logger.info(
            f"[DSpark] draft weight load complete "
            f"(linear_backend={'bf16' if self.dspark_model._use_bf16_linear else 'fp8'})"
        )

    def load_weights_from_target_model(self, target_model):
        """Share the target's embed_tokens / lm_head (DSpark has neither)."""
        if self.dspark_model.embed_tokens is None:
            self.dspark_model.embed_tokens = target_model.model.embed_tokens
        if self.lm_head is None:
            self.lm_head = target_model.lm_head
            self.dspark_model.lm_head = target_model.lm_head


# ----------------------------------------------------------------------------
# MLA-shaped standalone drafter backbone (Inferact/Kimi-K3-DSpark).
#
# A weight container, not a servable model: ``DFlashForCausalLM`` builds this
# through the registry only to own the drafter's modules, then runs its own
# hand-written block decode over them (see ``MLADSparkForCausalLM``). Nothing
# here registers with the attention backend or the KV cache manager, so the
# layers deliberately have no working ``forward``.
#
# Not a ``DeepseekV3`` reuse: that model carries MoE branches, a fused
# ``qkv_a_proj`` switch, quantization and KV-manager registration, none of
# which this bf16 five-layer drafter has.
# ----------------------------------------------------------------------------


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(minimum: float, maximum: float, dim: int) -> torch.Tensor:
    if minimum == maximum:
        maximum += 0.001  # Prevent singularity.
    linear_func = (torch.arange(dim, dtype=torch.float32) - minimum) / (maximum - minimum)
    return torch.clamp(linear_func, 0, 1)


def build_dspark_mla_yarn_rope(
    *,
    dim: int,
    base: float,
    scaling_factor: float,
    original_max_position_embeddings: int,
    max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """YaRN cos/sin tables for an MLA drafter's ``qk_rope_head_dim`` slice.

    Transcribed from HF ``DeepseekV3YarnRotaryEmbedding``: the drafters are
    distilled under HF/vLLM numerics, so the table is built here rather than
    routed through ``RopeParams`` (whose MLA convention carries the fused
    kernel's ``duplicate_data`` / GPT-J packing).

    Returns ``(cos, sin)``, both ``[max_position_embeddings, dim]`` fp32, with
    the frequency half duplicated so ``rotate_half`` applies.
    """
    freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freq_inter = 1.0 / (
        scaling_factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    low, high = _yarn_find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_position_embeddings
    )
    inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2)
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    t = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    # HF divides the two mscales; both published drafters set them equal, so
    # this is 1.0 there. Kept general because the checkpoint declares both.
    scale = _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(
        scaling_factor, mscale_all_dim
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * scale).to(device), (emb.sin() * scale).to(device)


def build_dspark_mla_yarn_freqs_cis(*, device: str = "cuda", **kwargs) -> torch.Tensor:
    """The same YaRN table as :func:`build_dspark_mla_yarn_rope`, adjacent-pair.

    Returns ``[max_position_embeddings, dim // 2]`` complex64: the convention the
    fused ``cute_dsl_dspark_rmsnorm_rope`` kernel consumes, where the last dim of
    the rotated slice is read as (re, im) pairs.

    HF DeepSeek instead de-interleaves and then rotates halves. The two produce
    the *same values in a different order*: with even/odd the pair members,
    HF writes ``even*cos - odd*sin`` to lane i and ``odd*cos + even*sin`` to lane
    i + dim/2, while this one writes them to lanes 2i and 2i+1. That permutation
    of the rope slice cancels inside ``q_rope . k_rope``, so the drafter is free
    to use either -- but every producer of a rope slice must use the SAME one.
    Mixing them changes the scores silently and only shows up as lower AL.

    ``mscale`` rides in the modulus rather than the phase, exactly as the
    cos/sin builder folds it into the table.
    """
    cos, sin = build_dspark_mla_yarn_rope(device=device, **kwargs)
    half = cos.shape[-1] // 2
    return torch.complex(cos[..., :half], sin[..., :half]).contiguous()


def apply_dspark_mla_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the trailing ``qk_rope_head_dim`` of ``x`` the HF DeepSeek way.

    ``x`` is ``[..., dim]`` and ``cos``/``sin`` are ``[..., dim]`` already
    gathered at the query positions. HF interleaves the rope slice, so it first
    reshapes ``(d//2, 2) -> transpose -> (d,)`` and only then applies the
    half-split ``rotate_half``; the net rotation is GPT-J style.
    """
    d = x.shape[-1]
    x = x.reshape(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(*x.shape[:-1], d)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class K3DsparkMLA(nn.Module):
    """MLA projections for one drafter layer. No attention, no KV registration.

    Head sharding follows :class:`~tensorrt_llm._torch.modules.mla.MLA`: the
    low-rank ``*_a`` projections are replicated (every head reads the same
    latent), ``q_b_proj`` / ``kv_b_proj`` are column-sharded by head and
    ``o_proj`` is row-sharded. Under attention-DP ``tp_size`` collapses to 1,
    which is also why the drafter's latent cache is unsharded.
    """

    def __init__(self, model_config, layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        dtype = config.torch_dtype

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        # What the drafter's KV cache stores per token: one MLA latent, no V.
        self.kv_cache_head_dim = self.kv_lora_rank + self.qk_rope_head_dim

        mapping = model_config.mapping
        if mapping.cp_size > 1:
            # MLA's own module folds cp_size into the head split and the o_proj
            # mapping; this drafter does not, and a silently wrong world_size
            # would mis-shard the head projections.
            raise NotImplementedError(
                f"The MLA DSpark drafter does not support context parallelism "
                f"(cp_size={mapping.cp_size})."
            )
        tp_size = mapping.tp_size
        dp_size = 1
        if mapping.enable_attention_dp:
            dp_size = tp_size
            tp_size = 1
        if self.num_heads % tp_size != 0:
            raise ValueError(
                f"DSpark MLA drafter has {self.num_heads} heads, not divisible by tp_size {tp_size}."
            )
        self.num_heads_tp = self.num_heads // tp_size
        attn_mapping = Mapping(
            world_size=mapping.pp_size * dp_size * tp_size,
            tp_size=tp_size,
            pp_size=mapping.pp_size * dp_size,
            rank=mapping.rank,
            gpus_per_node=mapping.gpus_per_node,
            enable_attention_dp=mapping.enable_attention_dp,
        )
        self.mapping = attn_mapping
        # Row-parallel o_proj reduces only when the heads are actually split.
        reduce_output = not mapping.enable_attention_dp and mapping.tp_size > 1

        skip_init = model_config.skip_create_weights_in_init
        self.q_a_proj = Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            dtype=dtype,
            skip_create_weights_in_init=skip_init,
        )
        self.q_a_layernorm = RMSNorm(
            hidden_size=self.q_lora_rank, eps=config.rms_norm_eps, dtype=dtype
        )
        self.q_b_proj = Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            dtype=dtype,
            mapping=attn_mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            skip_create_weights_in_init=skip_init,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        self.kv_a_proj_with_mqa = Linear(
            self.hidden_size,
            self.kv_cache_head_dim,
            bias=False,
            dtype=dtype,
            skip_create_weights_in_init=skip_init,
        )
        self.kv_a_layernorm = RMSNorm(
            hidden_size=self.kv_lora_rank, eps=config.rms_norm_eps, dtype=dtype
        )
        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            dtype=dtype,
            mapping=attn_mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            skip_create_weights_in_init=skip_init,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        self.o_proj = Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            dtype=dtype,
            mapping=attn_mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=reduce_output,
            skip_create_weights_in_init=skip_init,
            allreduce_strategy=model_config.allreduce_strategy,
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "K3DsparkMLA holds the drafter's MLA weights for MLADSparkForCausalLM's "
            "block decode; it has no standalone attention path."
        )


class K3DsparkDecoderLayer(DecoderLayer):
    """One drafter layer: MLA projections + dense gated MLP + the two norms."""

    def __init__(self, model_config, layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.self_attn = K3DsparkMLA(model_config, layer_idx)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
            dtype=config.torch_dtype,
            overridden_tp_size=1 if model_config.mapping.enable_attention_dp else None,
            config=model_config,
            layer_idx=layer_idx,
        )
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "K3DsparkDecoderLayer is driven by MLADSparkForCausalLM.dflash_forward, "
            "not by the generic decoder loop."
        )


class K3DsparkModel(DecoderModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        config = model_config.pretrained_config
        # The drafter ships its own trained embedding rather than sharing the
        # target's -- verified different weights on Inferact/Kimi-K3-DSpark --
        # so this is loaded from the drafter checkpoint, not overwritten.
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList(
            [
                K3DsparkDecoderLayer(model_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "K3DsparkModel is a weight container for the DSpark block decode; "
            "see MLADSparkForCausalLM.dflash_forward."
        )


@register_auto_model("K3DsparkForCausalLM")
class K3DsparkForCausalLM(SpecDecOneEngineForCausalLM[K3DsparkModel, PretrainedConfig]):
    """Registry entry for the MLA DSpark drafter checkpoint.

    The name is the one ``DFlashForCausalLM.__init__`` derives from the
    checkpoint's ``model_type`` ("k3_dspark"), but MLADSparkForCausalLM pins
    ``architectures`` explicitly rather than relying on that derivation --
    the Laguna precedent in ``modeling_dflash.py``.
    """

    def __init__(self, model_config):
        super().__init__(K3DsparkModel(model_config), model_config)


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


_LN2 = math.log(2.0)
# flashinfer sizes its own scratch from this; the MLA decode path is the only
# consumer here and 256 MiB is what the DFlash TRTLLM buffers already reserve.
_MLA_DECODE_WORKSPACE_BYTES = 256 * 1024 * 1024


class _MLABlockFixup(NamedTuple):
    """Layer-invariant state for the MLA block-decode fixup.

    The block's own latents are written into the pool at ``ctx_len + j`` first.
    Those slots belong to this request -- the draft KV manager adds
    ``max(draft_len, max_total_draft_tokens)`` tokens to every generation
    request each step (resource_manager.py:1060) -- and the accepted tokens
    overwrite them next step, so the write is transient rather than a claim on
    the cache. It is what the GQA drafter's TRTLLM backend already does
    (modeling_dflash.py append_paged_kv_cache).

    With ``seq_lens = ctx_len + block_size`` the kernel's in-block causality
    then leaves query j seeing block keys 0..j, so all that is left to fix up is
    the block's strict upper triangle -- ``blk_kv`` itself, already in hand. The
    earlier form kept the pool read-only and paid for it twice: a gather of the
    hidden context tail, and a 15-key rather than 8-key fixup.

    Every field depends only on ctx_len, the page table and the block geometry,
    so it is built once per forward rather than once per layer.
    """

    pages: torch.Tensor  # [B, block] page holding each block position
    offsets: torch.Tensor  # [B, block] offset of that position inside its page
    valid: torch.Tensor  # [1, block, 1, block] strict upper triangle
    page_tables_i32: torch.Tensor  # kernel operands, cast once
    seq_lens_i32: torch.Tensor  # ctx_len + block_size
    kv_bounds_i32: torch.Tensor  # [B * block] seq_lens repeated per query token


@lru_cache(maxsize=1)
def cute_dsl_mla_decode_unavailability_reason() -> Optional[str]:
    """Return why the cute-dsl MLA decode cannot run here, or None.

    Mirrors ``dflash_trtllm_gen_unavailability_reason``: the op itself only
    reports by raising, and ``attention_backend="CUTEDSL"`` has to fail at
    construction rather than five layers into the first draft step.
    """
    if (sm := get_sm_version()) not in (100, 103, 107):
        return f"requires SM100, SM103 or SM107, got SM{sm}"
    try:
        import cutlass  # noqa: F401

        from ..custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner  # noqa: F401
    except ImportError as error:
        return f"the cute-dsl kernels are not importable: {error}"
    return None


def _build_mla_block_fixup(ctx_len, page_tables, block_size, page_size) -> _MLABlockFixup:
    """Where the block's latents go, and the upper-triangle mask, for all layers."""
    device = ctx_len.device
    pos = ctx_len.view(-1, 1) + torch.arange(block_size, device=device)
    idx = torch.arange(block_size, device=device)
    seq_lens = (ctx_len + block_size).to(torch.int32)
    return _MLABlockFixup(
        pages=page_tables.gather(1, pos // page_size),
        offsets=pos % page_size,
        valid=(idx.view(1, block_size) > idx.view(block_size, 1)).view(
            1, block_size, 1, block_size
        ),
        page_tables_i32=page_tables.to(torch.int32),
        seq_lens_i32=seq_lens,
        # repeat_interleave, not repeat: the kernel indexes kv_bounds at
        # b * seq_len_q + q (mla_decode_fp16.py:400-407), so the batch must be
        # the slow axis. B == 1 cannot tell the two apart.
        kv_bounds_i32=seq_lens.repeat_interleave(block_size),
    )


def _resolve_dspark_mla_rope_params(pretrained_config, model_config=None, slack: int = 0) -> Dict:
    """RoPE settings for an MLA drafter, from either HF spelling.

    Transformers v5 checkpoints (TorchSpec) carry ``rope_parameters``; older
    ones carry ``rope_scaling``. Only YaRN is accepted: both published MLA
    drafters use it, and silently treating an unknown scaling as plain RoPE
    would cost acceptance without an error.
    """
    scaling = (
        getattr(pretrained_config, "rope_parameters", None)
        or getattr(pretrained_config, "rope_scaling", None)
        or {}
    )
    rope_type = scaling.get("rope_type") or scaling.get("type") or "default"
    if rope_type != "yarn":
        raise ValueError(
            f"MLA DSpark drafter declares rope_type={rope_type!r}; only 'yarn' is "
            "supported (the block decode builds its own YaRN table)."
        )
    theta = scaling.get("rope_theta") or getattr(pretrained_config, "rope_theta", 10000.0)
    return {
        "theta": float(theta),
        "scaling_factor": float(scaling["factor"]),
        "original_max_positions": int(scaling["original_max_position_embeddings"]),
        # Table length, not YaRN math: the correction range below uses
        # original_max_positions, so capping this at the served length is free.
        # K3 advertises 1,048,576, which would build a ~256 MiB complex64 table
        # per rank -- through full-size fp32 cos/sin, so ~768 MiB transient --
        # for a drafter whose point is 5760 B/token instead of 20480.
        "max_positions": _runtime_position_cap(model_config, pretrained_config, slack),
        "beta_fast": float(scaling.get("beta_fast", 32.0)),
        "beta_slow": float(scaling.get("beta_slow", 1.0)),
        "mscale": float(scaling.get("mscale", 1.0)),
        "mscale_all_dim": float(scaling.get("mscale_all_dim", 0.0)),
    }


class _DSparkHeadMixin:
    """The DSpark head set, shared by the GQA and MLA drafter wrappers.

    DSpark is DFlash plus three things, and only these three are common to both
    backbone shapes: the vanilla Markov intra-block logit bias, the
    ``shift_label`` slot convention (block slot j predicts draft token j+1, so
    slot 0 holds the anchor) and the confidence head. The block decode itself
    has nothing in common between the shapes, which is why this is a mixin and
    not a base class.

    Confidence-scheduled verification is not implemented yet: ``confidence_proj``
    is loaded but unused, and drafting always proposes the full K tokens.
    """

    def _init_dspark_heads(self, cfg) -> None:
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

    def _take_dspark_head_weights(self, weights: Dict) -> tuple[Dict, Dict]:
        """Pull the head tensors out of ``weights`` and load them onto self.

        The head keys are pulled out before the backbone remap: left in, they
        would pick up a ``model.`` prefix and be dropped by partial loading.
        Returns the remaining weights and the extracted head weights.
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
        return weights, dspark_weights


class GQADSparkForCausalLM(_DSparkHeadMixin, DFlashForCausalLM):
    """DSpark drafter on a GQA-shaped backbone, from a standalone checkpoint.

    Adds the DSpark head set (see :class:`_DSparkHeadMixin`) on top of the
    DFlash block decode.

    Named for the attention shape, not for a model: the backbone is whatever
    the drafter config resolves to through the model registry, and the
    inherited block decode works for every GQA family the DFlash drafters
    already cover (qwen3, llama, gpt_oss, ...). A per-model subclass would be
    empty. The GQA precondition is inherited, not introduced here -- see
    ``DFlashForCausalLM._validate_gqa_shape``. The MLA-backboned drafter is the
    sibling :class:`MLADSparkForCausalLM`, not a subclass of this.

    Reference: arXiv 2607.05147; deepseek-ai/DeepSpec.
    """

    # TRTLLM, matching what the shipped K3 configs already pin
    # (examples/kimi_k3/eval_extra_llm_options_dspark{,_nvfp4}.yaml); VANILLA
    # additionally needs the optional flash-attn package. AUTO degrades to
    # VANILLA off SM100/SM103.
    #
    # This also moves the drafter's context KV into the manager's paged pool,
    # and that is the point, not a side effect: dflash.py:592 keys `use_paged`
    # off the backend name, and this class leaves _paged_ctx_cache False, so
    # VANILLA is the only way to get the arena that is dense in max_seq_len and
    # that free_gpu_memory_fraction never bounds.
    _default_attention_backend = "TRTLLM"

    def __init__(self, draft_config, *, dflash_attention_backend: str = "AUTO"):
        super().__init__(draft_config, dflash_attention_backend=dflash_attention_backend)
        self._init_dspark_heads(draft_config.pretrained_config)

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Take the DSpark head weights, then hand the rest to DFlash."""
        weights, _ = self._take_dspark_head_weights(weights)
        return super().load_weights(weights, weight_mapper=weight_mapper, **kwargs)


class MLADSparkForCausalLM(_DSparkHeadMixin, DFlashForCausalLM):
    """DSpark drafter on an MLA-shaped backbone, from a standalone checkpoint.

    The sibling ``modeling_dspark``'s GQA class could not be: DFlash's block
    decode splits a fused ``qkv_proj`` by (num_heads, num_kv_heads, head_dim)
    and fuses per-head K/V across layers, and MLA has neither. What it keeps is
    the DFlash *contract* -- dual-source KV (context from the projected target
    hidden, block from the draft hidden), non-causal block attention, one
    precomputed context cache -- and the DSpark head set on top.

    The cache stores one ``kv_lora_rank + qk_rope_head_dim`` latent per token
    per layer and no V, which is the point: 5 x 576 x 2B = 5760 B/token/rank
    against the GQA drafter's 20480 under attention-DP, where KV heads are
    unsharded.

    ``spec_config.attention_backend`` selects among this class's own three
    block-decode implementations, not among the shared DFlash worker op sets:

    ``CUTEDSL``  one cute-dsl pass with ``kv_bounds``, no fixup
                 (``_cute_dsl_block_decode``). Paged only. What ``AUTO`` picks
                 on a build that can run it.
    ``TRTLLM``   flashinfer's trtllm-gen absorbed-MLA paged decode plus the
                 upper-triangle fixup (``_mla_paged_attention``).
    ``VANILLA``  the eager torch reference. It is also what the other two
                 degrade to on the unpaged arena, where there are no page
                 tables to hand a kernel.

    Reference: arXiv 2607.05147; deepseek-ai/DeepSpec.
    """

    # Attention is _mla_paged_attention, so neither worker OP SET applies and
    # neither backend's shape checks bind -- the absorbed shape (64 heads : 1 KV
    # head, head_dim 576) would fail both. The backend NAME still selects, via
    # _mla_block_decode_variant below.
    _uses_worker_attention_backend = False
    # CUTEDSL by default: 35.7% off the drafter region (947.50 -> 608.95
    # us/iteration, jobs 3006025/3006026) at no cost in acceptance or accuracy,
    # measured on both aggregated and disaggregated gsm8k (jobs 3006856/3006857,
    # 3006673/3006684). It does not degrade to another backend: this drafter
    # needs a 288 GB Blackwell part to hold the target at all, so a build
    # without the cute-dsl kernel cannot run the model either, and raising in
    # __init__ names the reason instead of hiding it behind a slower path.
    _default_attention_backend = "CUTEDSL"
    _supported_attention_backends = ("VANILLA", "TRTLLM", "CUTEDSL")

    # The whole point of the MLA drafter: its context KV comes out of the
    # manager's pool (5760 B/token/rank) instead of an arena dense in
    # max_seq_len that free_gpu_memory_fraction never bounds. Independent of the
    # backend field, which is why either value of it works here.
    _paged_ctx_cache = True

    def __init__(self, draft_config, *, dflash_attention_backend: str = "AUTO"):
        cfg = draft_config.pretrained_config
        # Pin the backbone instead of relying on the model_type-derived name
        # (the Laguna precedent in modeling_dflash.py): the checkpoint labels
        # itself "K3DSparkModel", which is registered nowhere.
        cfg.architectures = ["K3DsparkForCausalLM"]
        super().__init__(draft_config, dflash_attention_backend=dflash_attention_backend)
        self._init_dspark_heads(cfg)

        attn0 = self.model.layers[0].self_attn
        self.qk_nope_head_dim = attn0.qk_nope_head_dim
        self.qk_rope_head_dim = attn0.qk_rope_head_dim
        self.kv_lora_rank = attn0.kv_lora_rank
        self.v_head_dim = attn0.v_head_dim
        # The cache contract the worker sizes its arena from. Config-derived, so
        # it is published here rather than waiting for the lazy buffer build:
        # one MLA latent per token per layer, no V half.
        self._num_attn_layers = len(self.model.layers)
        self._num_heads = attn0.num_heads_tp
        self._num_kv_heads = 1
        self._head_dim = attn0.kv_cache_head_dim
        self._kv_factor = 1

        # No slack: the lookahead lives in dflash_position_ceiling, which the
        # worker publishes as _runtime_position_ceiling before any forward and
        # _mla_freqs_cis builds the table from. This config-derived cap is only
        # the fallback for direct construction, where no worker runs and the
        # positions asked for are whatever the caller passes.
        rope = _resolve_dspark_mla_rope_params(cfg, draft_config)
        self._mla_rope_params = rope
        self._mla_workspace = None
        self._mla_freqs = None
        self._cute_dsl_checked_shape = None
        # HF DeepSeek folds YaRN's attention scaling into the softmax scale:
        # softmax_scale = mscale^2 / sqrt(qk_nope + qk_rope). NOT 1/sqrt(576) --
        # the absorbed product reproduces the 192-dim un-absorbed score.
        mscale = _yarn_get_mscale(rope["scaling_factor"], rope["mscale_all_dim"])
        self.softmax_scale = (mscale * mscale) / math.sqrt(
            self.qk_nope_head_dim + self.qk_rope_head_dim
        )
        # Raise here rather than five layers into the first draft step, and name
        # the reason. Reached for an explicit CUTEDSL and for the AUTO default
        # alike -- neither degrades.
        if self.dflash_attention_backend == "CUTEDSL":
            reason = cute_dsl_mla_decode_unavailability_reason()
            if reason is not None:
                raise ValueError(
                    f"attention_backend='CUTEDSL' was requested but the cute-dsl "
                    f"MLA decode is unavailable: {reason}."
                )

    # -- shape / buffers ---------------------------------------------------

    def _validate_gqa_shape(self):
        """Replace the base's GQA precondition with the MLA one."""
        layers = getattr(self.model, "layers", None)
        if not layers:
            return
        attn0 = layers[0].self_attn
        keys = (
            "num_heads_tp",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "kv_lora_rank",
            "v_head_dim",
        )
        base = tuple(getattr(attn0, k) for k in keys)
        for idx, layer in enumerate(layers[1:], start=1):
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "kv_a_proj_with_mqa"):
                raise ValueError(
                    f"MLA DSpark block decode needs self_attn.kv_a_proj_with_mqa on every "
                    f"draft layer, but layer {idx} of {type(self.config).__name__} has none."
                )
            if tuple(getattr(attn, k) for k in keys) != base:
                raise ValueError(
                    "MLA DSpark fuses the latent projection across layers and needs one "
                    f"uniform MLA shape, but layer {idx} differs from layer 0 "
                    f"({dict(zip(keys, base))})."
                )

    def _init_rope(self):
        """No shared RoPE cache: the MLA path rotates only the rope slice."""
        self._rope_initialized = True

    @staticmethod
    @lru_cache(maxsize=1)
    def _mla_decode_op():
        """flashinfer's absorbed-MLA paged decode, or None when unavailable.

        Loaded lazily so a build without flashinfer still runs the eager path.
        """
        try:
            import flashinfer

            return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla
        except (ImportError, AttributeError):
            return None

    def _mla_block_decode_variant(self, paged: bool) -> str:
        """Which of the three block-decode implementations this step runs.

        ``paged`` is False on the private arena, where there are no page tables
        for either kernel, so both kernel backends degrade to eager there.
        """
        if not paged or self.dflash_attention_backend == "VANILLA":
            return "eager"
        if self.dflash_attention_backend == "CUTEDSL":
            return "cute_dsl"
        return "trtllm_gen" if self._mla_decode_op() is not None else "eager"

    def _mla_decode_workspace(self, device, min_bytes: int = 0):
        want = max(_MLA_DECODE_WORKSPACE_BYTES, int(min_bytes))
        if self._mla_workspace is None or self._mla_workspace.numel() < want:
            self._mla_workspace = torch.zeros(want, dtype=torch.uint8, device=device)
        return self._mla_workspace

    def _assert_cute_dsl_can_implement(self, batch, block_size, page_size, dtype) -> None:
        """Reject a shape the kernel cannot serve, here, naming the reason.

        Without this the rejection is silent until it is not: ``get_valid_tactics``
        returns [] (cute_dsl_custom_ops.py:8508), the AutoTuner falls back to its
        -1 sentinel, and ``forward`` compiles an unvalidated config -- so a draft
        pool with ``tokens_per_block=256`` (``128 % 256 != 0``,
        mla_decode_fp16.py:3798) surfaces as a CUTLASS error out of the fifth
        drafter layer rather than as the configuration problem it is.

        Page size is the only operand here that is not fixed by the checkpoint,
        and it comes from the draft KV pool, so this cannot move to __init__.
        """
        key = (batch, block_size, page_size, dtype)
        if self._cute_dsl_checked_shape == key:
            return
        import cutlass

        from ..custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

        in_dtype = cutlass.Float16 if dtype == torch.float16 else cutlass.BFloat16
        runner = CuteDSLNVMlaDecodeBlackwellRunner
        # Mirror the runner's own probe (cute_dsl_custom_ops.py:8542) through its
        # own constants, so this cannot drift from the tactic it will pick. Its
        # candidate tiler list has one entry; split_kv=1 is always in range.
        ok = runner._KERNEL_CLASS_BY_DTYPE[in_dtype].can_implement(
            batch,
            block_size,
            page_size,
            self._num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            in_dtype,
            in_dtype,
            cutlass.Float32,
            cutlass.Float32,
            (128, 128),
            (128, 256),
            1,
            False,
            runner._IS_VAR_SEQ,
            runner._IS_VAR_SPLIT_KV,
            page_size,
        )
        if not ok:
            raise ValueError(
                f"attention_backend='CUTEDSL' cannot serve this shape: batch={batch}, "
                f"block_size={block_size}, page_size={page_size}, "
                f"num_heads={self._num_heads}, dtype={dtype}. Use "
                f"attention_backend='TRTLLM', or set kv_cache_config.tokens_per_block "
                f"to a divisor of 128 other than 1."
            )
        self._cute_dsl_checked_shape = key

    def _cute_dsl_block_decode(self, query, layer_cache, fixup, block_size):
        """One cute-dsl MLA decode over context AND block, no fixup.

        The trtllm-gen path has to split the work: its q_len semantics are
        causal, so query i cannot see block positions j > i, and the strict
        upper triangle is recovered eagerly afterwards and merged by LSE. The
        cute-dsl kernel takes ``kv_bounds`` -- a per-query-token KV length that
        REPLACES the implicit causal bound (mla_decode_fp16.py:400-407) -- so
        filling it with ctx_len + block_size makes every block position visible
        to every other in one pass and deletes the fixup outright.

        No fallback: what the kernel cannot serve raises. Build-level
        availability is checked at construction through
        ``cute_dsl_mla_decode_unavailability_reason``; the shape depends on the
        draft pool's page size, which is not known until here, so it is checked
        on the first call.
        """
        import cutlass

        from ..custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

        batch, _, num_heads, _ = query.shape
        device = query.device
        pool = layer_cache[:, 0, 0]  # [pages, page_size, head_dim]
        page_size = pool.shape[1]
        d_latent = self.kv_lora_rank
        self._assert_cute_dsl_can_implement(batch, block_size, page_size, query.dtype)

        # stride[1] == 1 is the kernel's only layout demand (mla_decode_fp16.py:465-470);
        # a 576-wide row sliced at 512 keeps that on both halves, so no copy.
        # detach(): cute.runtime.from_dlpack refuses a tensor that requires grad
        # ("Can't export tensors that require gradient"). Production runs under
        # inference_mode so nothing here carries grad, but the in-place pool
        # write above pulls the pool into the autograd graph whenever a caller
        # does not -- a unit test, for one. Views, so no copy.
        q_latent = query.detach()[..., :d_latent].permute(2, 3, 1, 0)
        q_rope = query.detach()[..., d_latent:].permute(2, 3, 1, 0)
        c_latent = pool.detach()[..., :d_latent].permute(1, 2, 0)
        c_rope = pool.detach()[..., d_latent:].permute(1, 2, 0)

        out = torch.empty(batch, block_size, num_heads, d_latent, dtype=query.dtype, device=device)
        workspace_numel = CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
            num_heads, block_size, d_latent, batch, cutlass.Float32
        )
        workspace = self._mla_decode_workspace(device, workspace_numel)
        torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            # .t() WITHOUT .contiguous(): the kernel wants (max_blocks, B) with
            # dim 0 as the leading (stride-1) dimension. contiguous() would make
            # it row-major, i.e. stride (B, 1), and the kernel rejects it with
            # "Expected strides[leading_dim] == 1, but got B" for every B > 1
            # (probe job 3005553; B=1 passes by coincidence). The transposed
            # view of a row-major (B, max_blocks) already has stride (1, max_blocks).
            fixup.page_tables_i32.t(),
            fixup.seq_lens_i32,
            out.permute(2, 3, 1, 0),
            workspace,
            num_heads,
            block_size,
            page_size,
            self.softmax_scale,
            1.0,
            batch,
            None,
            # Every block token sees the whole block, not just its causal prefix.
            fixup.kv_bounds_i32,
        )
        return out

    def _mla_paged_attention(self, query, blk_kv, layer_cache, fixup, block_size, max_kv_len):
        """Absorbed-MLA block attention: paged kernel + a block-local fixup.

        The kernel is causal across the query block and takes no mask (measured,
        not assumed: it matches a causal reference to 1.6e-4 against 1.5e-2 for
        full visibility). Its LSE is base 2 (also measured: exact against
        ``log2(sum exp)``, 1.84 off against the natural log).

        The block's own latents go into the pool first, at ``ctx_len + j`` --
        slots the draft KV manager already reserves for this request each step
        (see _MLABlockFixup). With ``seq_lens = ctx_len + block_size`` the kernel
        then covers context AND block, and in-block causality leaves only the
        block's strict upper triangle: 8 keys, no context-tail gather. Those are
        attended eagerly against ``blk_kv``, already in hand, and merged by
        log-sum-exp.

        Writing into the pool is safe because dflash.py bounds the advertised
        context at ``allocated - block_size``; without that the first block
        position lands on the first unallocated page, whose table entry was
        clamped to physical block 0 -- another request's.

        Returns the latent-space output ``[B, block, heads, kv_lora_rank]``.
        """
        decode = self._mla_decode_op()
        batch, _, num_heads, _ = query.shape
        device = query.device

        # --- the block's own latents into the pool, then one causal pass ---
        # Must precede BOTH decode paths: each reads the block back out of the
        # pool, so writing it afterwards would attend uninitialised slots.
        pool = layer_cache[:, 0, 0]  # [pages, page_size, head_dim]
        pool[fixup.pages.reshape(-1), fixup.offsets.reshape(-1)] = blk_kv.reshape(
            -1, blk_kv.shape[-1]
        )

        if self._mla_block_decode_variant(paged=True) == "cute_dsl":
            return self._cute_dsl_block_decode(query, layer_cache, fixup, block_size)
        out_lower = torch.empty(
            batch, block_size, num_heads, self.kv_lora_rank, dtype=query.dtype, device=device
        )
        lse_lower = torch.empty(batch * block_size, num_heads, dtype=torch.float32, device=device)
        decode(
            query,
            layer_cache[:, 0],  # [pages, 1, page, head_dim]
            self._mla_decode_workspace(device),
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            fixup.page_tables_i32,
            fixup.seq_lens_i32,
            max_kv_len,
            0,
            out_lower,
            self.softmax_scale,
            1.0,
            None,
            None,
            None,
            backend="trtllm-gen",
            is_var_seq=True,
            uses_shared_paged_kv_idx=True,
            lse=lse_lower,
            return_lse=True,
        )

        # --- fixup: the block's strict upper triangle, the only thing the
        # kernel's in-block causality hides once the block is in the pool ---
        scores = torch.einsum("bshc,btc->bsht", query, blk_kv).float() * self.softmax_scale
        scores = scores.masked_fill(~fixup.valid, float("-inf"))
        lse_upper = scores.logsumexp(-1)
        probs = torch.softmax(scores, dim=-1).nan_to_num_().to(blk_kv.dtype)
        out_upper = torch.einsum("bsht,btc->bshc", probs, blk_kv[..., : self.kv_lora_rank])

        # --- log-sum-exp merge (flash-decoding combine), kernel LSE is base 2 ---
        # The weight pair collapses: w_u / (w_l + w_u) == sigmoid(lse_u - lse_l),
        # so the merge is one sigmoid and one lerp instead of a peak subtraction,
        # two exps and a division -- and it needs no explicit max for stability.
        # The last query has an empty triangle, hence lse_upper -inf, sigmoid 0,
        # pure kernel output.
        lse_l = lse_lower.view(batch, block_size, num_heads).float() * _LN2
        weight_upper = torch.sigmoid(lse_upper - lse_l).unsqueeze(-1)
        merged = torch.lerp(out_lower.float(), out_upper.float(), weight_upper)
        return merged.to(query.dtype)

    def _mla_freqs_cis(self, device):
        """Adjacent-pair YaRN table, cached. See build_dspark_mla_yarn_freqs_cis.

        Sized from the ceiling the worker publishes once it knows the runtime
        one (_runtime_position_ceiling, set in DFlashWorker._lazy_init_ctx_buffers
        before any forward), because that is the value ctx_len is clamped to and
        it is strictly above model_config.max_seq_len whenever spec decoding is
        on. The config-derived cap is the fallback for direct construction in
        tests, where no worker runs.
        """
        if self._mla_freqs is None:
            rope = dict(self._mla_rope_params)
            runtime_cap = getattr(self, "_runtime_position_ceiling", None)
            if runtime_cap is not None:
                # Outright, not max(): the published ceiling is already
                # min(runtime, max_position_embeddings) + lookahead, so it is
                # both sufficient and the only thing that keeps the table small
                # when max_seq_len is unset and the config cap is the
                # checkpoint's advertised 1,048,576.
                rope["max_positions"] = int(runtime_cap)
            self._mla_freqs = build_dspark_mla_yarn_freqs_cis(
                dim=self.qk_rope_head_dim,
                base=rope["theta"],
                scaling_factor=rope["scaling_factor"],
                original_max_position_embeddings=rope["original_max_positions"],
                max_position_embeddings=rope["max_positions"],
                beta_fast=rope["beta_fast"],
                beta_slow=rope["beta_slow"],
                mscale=rope["mscale"],
                mscale_all_dim=rope["mscale_all_dim"],
                device=device,
            )
        return self._mla_freqs

    def _build_fused_kv_buffers(self) -> None:
        """Stack the per-layer latent projection and its RMSNorm weight.

        The MLA analogue of the base's fused K/V GEMM: one ``[L*576, hidden]``
        weight, plus the ``[L, kv_lora_rank]`` layernorm scales applied after.
        """
        if self._fused_kv_weight is not None:
            return
        layers_attn = [layer.self_attn for layer in self.model.layers]
        attn0 = layers_attn[0]
        self._fused_kv_weight = torch.cat(
            [a.kv_a_proj_with_mqa.weight for a in layers_attn], dim=0
        ).contiguous()
        self._fused_kv_bias = None
        self._kv_a_norm_stacked = torch.stack([a.kv_a_layernorm.weight.data for a in layers_attn])
        self._kv_a_norm_eps = attn0.kv_a_layernorm.variance_epsilon
        self._input_ln_eps = None
        self._has_qk_norm = False
        self._use_fused_qk_norm_rope = False

        # Split kv_b_proj into the absorption operands once. K absorbs into the
        # query, V un-absorbs the attention output.
        nh = attn0.num_heads_tp
        for a in layers_attn:
            w = a.kv_b_proj.weight.view(nh, self.qk_nope_head_dim + self.v_head_dim, -1)
            a._k_b_proj = w[:, : self.qk_nope_head_dim].contiguous()
            a._v_b_proj = w[:, self.qk_nope_head_dim :].contiguous()

        logger.debug(
            f"MLA DSpark: fused latent projection built for {self._num_attn_layers} layers "
            f"(weight={tuple(self._fused_kv_weight.shape)}, head_dim={self._head_dim})"
        )

    # -- context cache -----------------------------------------------------

    def precompute_context_kv(self, projected_hidden, positions):
        """Post-norm / post-RoPE MLA latent for ALL drafter layers in one GEMM.

        Returns ``(latent, None)``: ``[N, L, 1, kv_lora_rank + qk_rope_head_dim]``
        and no V half, which is what ``_kv_factor == 1`` means to the worker.
        """
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        n = projected_hidden.shape[0]
        num_layers = self._num_attn_layers
        weight_dtype = self._fused_kv_weight.dtype
        if projected_hidden.dtype != weight_dtype:
            projected_hidden = projected_hidden.to(weight_dtype)

        latent = F.linear(projected_hidden, self._fused_kv_weight)
        latent = latent.view(n, num_layers, self._head_dim)

        # One fused RMSNorm+RoPE per layer, not one eager chain for all of them:
        # the kernel takes a single norm weight per call and the layers do not
        # share one. Same op the block path uses, so the two halves of an
        # attention stay numerically consistent -- which is the point here, not
        # speed (this runs per request, not per decode step).
        freqs = self._mla_freqs_cis(latent.device)[positions.view(-1).long()]
        out = torch.empty_like(latent)
        for layer_idx in range(num_layers):
            # 3D throughout: the eager fallback's rotary indexes [G, s, rd].
            out[:, layer_idx] = _rmsnorm_rope_batched(
                latent[:, layer_idx].unsqueeze(0).contiguous(),
                self._kv_a_norm_stacked[layer_idx],
                self._kv_a_norm_eps,
                self.qk_rope_head_dim,
                freqs.unsqueeze(0),
                norm_dim=self.kv_lora_rank,
            ).squeeze(0)
        return out.unsqueeze(2).contiguous(), None

    # -- block decode ------------------------------------------------------

    def dflash_forward(
        self,
        noise_embedding,
        query_positions,
        num_ctx_per_req,
        ctx_k_cache,
        ctx_v_cache,
        ctx_cache_batch_idx,
        ctx_kv_cache=None,
        ctx_page_table=None,
    ):
        """Eager absorbed-MLA block decode over the drafter's latent cache.

        Args mirror the base: ``ctx_k_cache`` is
        ``[pool_slots, L, max_ctx+block_size, 1, kv_cache_head_dim]`` and
        ``ctx_v_cache`` is unused (``None``).
        """
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()

        batch, block_size = noise_embedding.shape[0], noise_embedding.shape[1]
        device = noise_embedding.device
        num_heads = self._num_heads
        latent_dim = self.kv_lora_rank

        # [B, blk, rope/2] complex: one phase per draft position, shared by the
        # query and the block latent of every layer.
        blk_freqs = self._mla_freqs_cis(device)[query_positions.long()]

        slots = ctx_cache_batch_idx.long()
        # Context pages for this batch, resolved once: rows are batch positions
        # when bound to the manager's per-request block table and slots on the
        # private arena, which is exactly what ctx_cache_batch_idx already is.
        page_tables = (
            None if ctx_page_table is None else ctx_page_table.index_select(0, slots).long()
        )

        # Valid-key mask, shared by every layer: context slots below this
        # request's length, then the whole block (DSpark is non-causal). Pages
        # past a request's allocation hold another request's data, so the mask
        # is what keeps them out -- not the block table.
        page_size = None if page_tables is None else ctx_kv_cache[0].shape[-2]
        capacity = ctx_k_cache.shape[2] if page_tables is None else page_tables.shape[1] * page_size
        # attention_backend picks the implementation; both kernel backends need
        # page tables, so the eager gather also covers the private arena.
        variant = self._mla_block_decode_variant(page_tables is not None)
        use_kernel = variant != "eager"
        # Kernel path: index and mask operands are layer-invariant, so build
        # them once here rather than five times inside the decode loop.
        fixup = (
            _build_mla_block_fixup(num_ctx_per_req[:batch], page_tables, block_size, page_size)
            if use_kernel
            else None
        )
        logger.info_once(
            "MLA DSpark block decode: "
            + {
                "trtllm_gen": "trtllm-gen paged kernel, block in pool + upper-triangle fixup",
                "cute_dsl": "one cute-dsl pass over context and block, no fixup",
                "eager": f"eager (backend={self.dflash_attention_backend!r}, "
                f"paged={page_tables is not None}, "
                f"flashinfer={self._mla_decode_op() is not None})",
            }[variant],
            key="mla_dspark_block_decode_variant",
        )
        # Eager path only: dense key mask over the gathered context.
        if not use_kernel:
            ctx_valid = torch.arange(capacity, device=device).unsqueeze(0) < num_ctx_per_req[
                :batch
            ].view(-1, 1)
            key_valid = torch.cat(
                [ctx_valid, torch.ones(batch, block_size, dtype=torch.bool, device=device)], dim=1
            )
        hidden_states = noise_embedding
        residual = None
        for layer_idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            if residual is None:
                residual = hidden_states.clone()
                hs_normed = layer.input_layernorm(hs_flat)
            else:
                res_flat = residual.reshape(-1, residual.shape[-1])
                hs_normed, res_flat = layer.input_layernorm(hs_flat, res_flat)
                residual = res_flat.reshape(batch, block_size, -1)

            q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hs_normed)))
            q = q.view(batch, block_size, num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
            # The 576-wide query is assembled in one buffer rather than
            # cat([absorbed, rope]): q's rope half is a 64-slice of a 192-wide
            # row, so that cat had a non-contiguous input and PyTorch dropped
            # off CatArrayBatchedCopy_alignedK_contig onto the generic kernel --
            # 17.3 us vs 1.6 us per launch, five launches per iteration
            # (measured job 2986420). Mirrors fused_q in modules/mla.py
            # forward_absorption_generation.
            query = torch.empty(
                (batch, block_size, num_heads, self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
            # Rotate the trailing rope slice of every head straight into the
            # destination's rope columns. No norm and no weight here --
            # q_a_layernorm already ran. Writing the rotation into `query`
            # instead of back into `q` removes the strided
            # `query[..., kv_lora_rank:] = q_rope` copy that replaced the cat.
            q_rope_into = _mla_rope_into(
                q, blk_freqs, query, num_heads, self.qk_rope_head_dim, self.kv_lora_rank
            )
            if not q_rope_into:
                q = _rmsnorm_rope_batched(
                    q,
                    None,
                    0.0,
                    self.qk_rope_head_dim,
                    blk_freqs,
                    num_heads=num_heads,
                    apply_weight=False,
                    apply_rmsnorm=False,
                )
                query[..., self.kv_lora_rank :] = q[..., self.qk_nope_head_dim :]
            q_nope = q[..., : self.qk_nope_head_dim]
            # Absorb the nope half into latent space so the block attends the
            # stored latent directly (MQA), instead of expanding it per head.
            # [h, b*s, nope] x [h, nope, kv_lora_rank] straight into the latent
            # half. view(), never reshape(): both of these are strided views and
            # reshape would silently stage a copy that bmm_out then writes into
            # and throws away, whereas view() raises.
            torch.ops.trtllm.bmm_out(
                q_nope.view(batch * block_size, num_heads, self.qk_nope_head_dim).transpose(0, 1),
                attn._k_b_proj.to(q.dtype),
                query[..., : self.kv_lora_rank]
                .view(batch * block_size, num_heads, self.kv_lora_rank)
                .transpose(0, 1),
            )

            # The block's own latent comes from the draft hidden; the context's
            # from the projected target hidden (precompute_context_kv). That
            # dual-source split is the DFlash contract, not an MLA detail.
            blk_latent = attn.kv_a_proj_with_mqa(hs_normed).view(batch, block_size, -1)
            # RMSNorm over the latent, RoPE over k_pe, one kernel. norm_dim keeps
            # k_pe out of the reduction, which is what DeepSeek-style MLA wants
            # and what DSpark's own convention (whole row) does not do.
            blk_kv = _rmsnorm_rope_batched(
                blk_latent,
                attn.kv_a_layernorm.weight,
                attn.kv_a_layernorm.variance_epsilon,
                self.qk_rope_head_dim,
                blk_freqs,
                norm_dim=latent_dim,
            )
            if use_kernel:
                ctx_out = self._mla_paged_attention(
                    query,
                    blk_kv,
                    ctx_kv_cache[layer_idx],
                    fixup,
                    block_size,
                    capacity,
                )
            else:
                if page_tables is None:
                    ctx = ctx_k_cache[slots, layer_idx, :, 0, :]
                else:
                    # [pages, kv_factor=1, nkv=1, page, hd] -> gather this
                    # batch's pages and flatten back to a per-request context.
                    ctx = ctx_kv_cache[layer_idx][:, 0, 0][page_tables].reshape(batch, capacity, -1)
                kv_full = torch.cat([ctx, blk_kv], dim=1)

                scores = torch.einsum("bshc,btc->bsht", query, kv_full).float() * self.softmax_scale
                scores = scores.masked_fill(~key_valid.view(batch, 1, 1, -1), float("-inf"))
                probs = torch.softmax(scores, dim=-1).to(kv_full.dtype)
                ctx_out = torch.einsum("bsht,btc->bshc", probs, kv_full[..., :latent_dim])
            attn_out = torch.einsum("bshc,hvc->bshv", ctx_out, attn._v_b_proj.to(ctx_out.dtype))

            hidden_out = attn.o_proj(attn_out.reshape(batch * block_size, -1))
            res_flat = residual.reshape(-1, residual.shape[-1])
            hidden_out, res_flat = layer.post_attention_layernorm(hidden_out, res_flat)
            hidden_out = layer.mlp(hidden_out)
            hidden_states = hidden_out.reshape(batch, block_size, -1)
            residual = res_flat.reshape(batch, block_size, -1)

        hidden_states_out, _ = self.model.norm(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            residual.reshape(-1, residual.shape[-1]),
        )
        return hidden_states_out

    # -- weights -----------------------------------------------------------

    #: Only the head is shared. This drafter ships its own embed_tokens whose
    #: values differ from the target's -- see load_weights_from_target_model --
    #: so a checkpoint missing it must fail rather than run on random weights.
    WEIGHTS_SHARED_WITH_TARGET = ("lm_head",)

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Take the DSpark heads, rename the TorchSpec spellings, then load.

        The TorchSpec drafters name the capture projection ``context_proj`` /
        ``context_norm`` and the final norm ``final_norm``, where the SpecForge
        ones use ``fc`` / ``hidden_norm`` / ``norm``. Everything below that is
        already HF-standard MLA.
        """
        weights, _ = self._take_dspark_head_weights(weights)
        renames = {
            "context_proj.weight": "fc.weight",
            "context_norm.weight": "hidden_norm.weight",
            "final_norm.weight": "norm.weight",
        }
        weights = {renames.get(k, k): v for k, v in weights.items()}
        return super().load_weights(weights, weight_mapper=weight_mapper, **kwargs)

    def load_weights_from_target_model(self, target_model: torch.nn.Module) -> None:
        """Share only ``lm_head``; the drafter has its own trained embedding.

        Inferact/Kimi-K3-DSpark ships ``embed_tokens`` whose values differ from
        the target's, so adopting the target's would feed the block decode
        embeddings it was not distilled against -- silently, at some acceptance
        cost. The GQA drafters ship no embedding and keep the base behavior.
        """
        self.draft_model_full.lm_head = target_model.lm_head
        self.lm_head = target_model.lm_head


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

    # Dispatch on the attention shape, not on model_type: the block decode is
    # what differs between the two wrappers, and everything below it
    # (``DFlashForCausalLM.__init__``) already resolves the backbone through the
    # model registry, so keying on model_type a second time would force a new
    # entry for every GQA family that already works.
    drafter_cls = (
        MLADSparkForCausalLM if is_mla(draft_config.pretrained_config) else GQADSparkForCausalLM
    )
    return drafter_cls(
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
    "MLADSparkForCausalLM",
    # MLA drafter backbone (a weight container for the block decode).
    "K3DsparkForCausalLM",
    "K3DsparkModel",
    "build_dspark_mla_yarn_rope",
    "build_dspark_mla_yarn_freqs_cis",
    "apply_dspark_mla_rope",
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
