# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton DeepSeek-V4 MoE gate for decode-size batches.

Computes the same routing as ``torch.ops.trtllm.gate_forward`` (sqrt(softplus) scores, top-k on the
biased scores or the ``tid2eid`` hash table, weights renormalized and scaled), with one 4-warp
program per token instead of one warp. The math follows the CUDA kernel built with
``--use_fast_math`` (fast exp, log1p, approximate sqrt), so the selected experts match it exactly
and the weights differ only in summation order.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

TRITON_GATE_MAX_TOKENS = 256
_EXPERT_COUNTS = (256, 384)
_TOP_K = 6


@triton.jit
def _dsv4_gate_kernel(
    logits,
    bias,
    vision_bias,
    image_mask,
    input_ids,
    tid2eid,
    out_weights,
    out_indices,
    n_experts,
    route_scale,
    HASH: tl.constexpr,
    HAS_MASK: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, TOPK_PAD)
    kmask = k < TOPK
    if HASH:
        token = tl.load(input_ids + row).to(tl.int64)
        idx = tl.load(tid2eid + token * TOPK + k, mask=kmask, other=0)
        s = tl.load(logits + row * n_experts + idx, mask=kmask, other=0.0)
        sp = tl.where(s > 20.0, s, libdevice.log1p(tl.exp(s)))
        w = tl.where(kmask, tl.sqrt(sp), 0.0)
    else:
        offs = tl.arange(0, BLOCK)
        valid = offs < n_experts
        s = tl.load(logits + row * n_experts + offs, mask=valid, other=0.0)
        sp = tl.where(s > 20.0, s, libdevice.log1p(tl.exp(s)))
        score = tl.sqrt(sp)
        b = tl.load(bias + offs, mask=valid, other=0.0)
        if HAS_MASK:
            vb = tl.load(vision_bias + offs, mask=valid, other=0.0)
            b = tl.where(tl.load(image_mask + row) != 0, vb, b)
        biased = tl.where(valid, score + b, float("-inf"))
        w = tl.zeros([TOPK_PAD], dtype=tl.float32)
        idx = tl.zeros([TOPK_PAD], dtype=tl.int32)
        for j in tl.static_range(TOPK):
            best = tl.argmax(biased, 0)
            hit = offs == best
            w = tl.where(k == j, tl.sum(tl.where(hit, score, 0.0), 0), w)
            idx = tl.where(k == j, best.to(tl.int32), idx)
            biased = tl.where(hit, float("-inf"), biased)
    total = tl.sum(w, 0)
    tl.store(out_weights + row * TOPK + k, (w / (total + 1e-20)) * route_scale, mask=kmask)
    tl.store(out_indices + row * TOPK + k, idx.to(tl.int32), mask=kmask)


def dsv4_triton_gate(
    logits: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_ids: Optional[torch.Tensor],
    tid2eid: Optional[torch.Tensor],
    out_weights: torch.Tensor,
    out_indices: torch.Tensor,
    top_k: int,
    route_scale: float,
    is_hash: bool,
    image_mask: Optional[torch.Tensor] = None,
    vision_bias: Optional[torch.Tensor] = None,
) -> bool:
    """Run the gate with the Triton kernel if the inputs are supported.

    Args:
        logits: [num_tokens, num_experts] float32, contiguous; num_experts in (256, 384).
        bias: [num_experts] float32 expert-score correction bias (top-k mode).
        input_ids: [num_tokens] int32 token IDs (hash mode).
        tid2eid: [vocab_size, top_k] int32 token-to-expert table (hash mode).
        out_weights: [num_tokens, top_k] float32 output.
        out_indices: [num_tokens, top_k] int32 output.
        top_k: number of experts per token; only 6 is supported.
        route_scale: scale applied to the renormalized weights.
        is_hash: hash routing instead of top-k selection.
        image_mask: optional [num_tokens] mask selecting ``vision_bias`` (top-k mode).
        vision_bias: [num_experts] float32 bias for masked tokens.

    Returns:
        False without launching anything when the inputs are outside what the kernel supports
        (the caller then runs ``gate_forward``), True after writing the outputs.
    """
    m, n = logits.shape
    if (
        not logits.is_cuda
        or m == 0
        or m > TRITON_GATE_MAX_TOKENS
        or n not in _EXPERT_COUNTS
        or top_k != _TOP_K
        or logits.dtype != torch.float32
        or not logits.is_contiguous()
    ):
        return False
    if is_hash and (
        input_ids is None
        or input_ids.dtype != torch.int32
        or tid2eid is None
        or tid2eid.dtype != torch.int32
        or not tid2eid.is_contiguous()
    ):
        return False
    has_mask = image_mask is not None and not is_hash
    _dsv4_gate_kernel[(m,)](
        logits,
        logits if is_hash else bias,
        vision_bias if has_mask else logits,
        image_mask if has_mask else logits,
        input_ids.contiguous() if is_hash else logits,
        tid2eid if is_hash else logits,
        out_weights,
        out_indices,
        n,
        route_scale,
        HASH=is_hash,
        HAS_MASK=has_mask,
        TOPK=top_k,
        TOPK_PAD=8,
        BLOCK=triton.next_power_of_2(n),
        num_warps=4,
    )
    return True
