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
"""Fused DSpark vanilla-Markov greedy block chain.

Bit-identical to
:func:`tensorrt_llm._torch.models.modeling_speculative.dspark_markov_chain_logits`
driven by a plain full-vocab ``argmax``: step ``i`` adds
``markov_w2 @ markov_w1[prev_i]`` (cast to the logits dtype) to block position
``i``, where ``prev_0`` is the anchor token (an out-of-vocab anchor gives a zero
bias) and ``prev_{i>0}`` is the ``torch.argmax`` of step ``i - 1``'s biased
logits.

Each step is three launches instead of about a dozen:

1. ``_markov_prev_embedding_kernel`` -- reduces the previous step's per-tile
   argmax partials to ``prev_i`` (or reads the anchor) and gathers the masked
   ``markov_w1`` row;
2. the same ``F.linear`` bias GEMV as the reference, so the bias is
   bit-identical by construction;
3. ``_markov_add_argmax_kernel`` -- adds the bias to the block logits, stores
   the biased row and writes a per-tile argmax.

Launch configurations depend only on ``(B, vocab)`` and nothing synchronizes
with the host, so the chain is CUDA-graph capturable.
"""

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

FUSED_MARKOV_ENV_VAR = "TLLM_DSPARK_FUSED_MARKOV"

_ADD_BLOCK_V = 2048
_INT32_MAX = tl.constexpr(2**31 - 1)


def fused_markov_enabled() -> bool:
    """Whether ``TLLM_DSPARK_FUSED_MARKOV=1`` selects the fused Markov chain."""
    return os.environ.get(FUSED_MARKOV_ENV_VAR, "0") == "1"


def fused_markov_supported(
    base_logits: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
) -> bool:
    """Whether :func:`fused_markov_chain_logits` handles these operands.

    Requires full-vocab fp32 ``[B, K, vocab]`` block logits with a unit vocab
    stride, a contiguous ``[B]`` anchor (the kernel indexes it by request) and
    contiguous ``[vocab, rank]`` Markov weights of one dtype.
    """
    return (
        base_logits.is_cuda
        and base_logits.dim() == 3
        and base_logits.dtype == torch.float32
        and base_logits.stride(-1) == 1
        and first_prev_tokens.shape == base_logits.shape[:1]
        and first_prev_tokens.is_contiguous()
        and markov_w1.dtype == markov_w2.dtype
        and markov_w1.is_contiguous()
        and markov_w2.is_contiguous()
        and markov_w1.shape == markov_w2.shape
        and markov_w2.shape[0] == base_logits.shape[-1]
    )


@triton.jit
def _argmax_combine(val_a, idx_a, val_b, idx_b):
    # torch.argmax order: NaN beats any number, equal values go to the lower index.
    nan_a = val_a != val_a
    nan_b = val_b != val_b
    take_a = tl.where(
        nan_a,
        (~nan_b) | (idx_a < idx_b),
        (val_a > val_b) | ((val_a == val_b) & (idx_a < idx_b)),
    )
    return tl.where(take_a, val_a, val_b), tl.where(take_a, idx_a, idx_b)


@triton.jit
def _markov_prev_embedding_kernel(
    part_val_ptr,
    part_idx_ptr,
    anchor_ptr,
    w1_ptr,
    emb_ptr,
    vocab,
    num_parts,
    RANK: tl.constexpr,
    FROM_ANCHOR: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    req = tl.program_id(0)
    if FROM_ANCHOR:
        prev = tl.load(anchor_ptr + req).to(tl.int64)
    else:
        offs_p = tl.arange(0, BLOCK_P)
        mask_p = offs_p < num_parts
        part_val = tl.load(
            part_val_ptr + req * num_parts + offs_p, mask=mask_p, other=float("-inf")
        )
        part_idx = tl.load(part_idx_ptr + req * num_parts + offs_p, mask=mask_p, other=_INT32_MAX)
        _, best_idx = tl.reduce((part_val, part_idx), 0, _argmax_combine)
        prev = best_idx.to(tl.int64)
    valid = (prev >= 0) & (prev < vocab)
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < RANK
    row = tl.load(w1_ptr + tl.where(valid, prev, 0) * RANK + offs_r, mask=mask_r & valid, other=0.0)
    tl.store(emb_ptr + req * RANK + offs_r, row, mask=mask_r)


@triton.jit
def _markov_add_argmax_kernel(
    base_ptr,
    out_ptr,
    bias_ptr,
    stride_base_b,
    stride_out_b,
    part_val_ptr,
    part_idx_ptr,
    vocab,
    num_parts,
    BLOCK_V: tl.constexpr,
):
    req = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < vocab
    row = req.to(tl.int64)
    bias = tl.load(bias_ptr + row * vocab + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    base = tl.load(base_ptr + row * stride_base_b + offs_v, mask=mask_v, other=0.0)
    logits = base + bias
    tl.store(out_ptr + row * stride_out_b + offs_v, logits, mask=mask_v)
    vals = tl.where(mask_v, logits, float("-inf"))
    idx = tl.where(mask_v, offs_v, _INT32_MAX)
    best_val, best_idx = tl.reduce((vals, idx), 0, _argmax_combine)
    tl.store(part_val_ptr + req * num_parts + pid_v, best_val)
    tl.store(part_idx_ptr + req * num_parts + pid_v, best_idx)


def fused_markov_chain_logits(
    base_logits: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
) -> torch.Tensor:
    """Greedy-chained vanilla-Markov block logits (fused path).

    Args:
        base_logits: ``[B, K, vocab]`` fp32 shared-lm_head block logits (full
            vocab, unit vocab stride).
        first_prev_tokens: ``[B]`` int32/int64 anchor token ids.
        markov_w1: ``[vocab, rank]``, contiguous.
        markov_w2: ``[vocab, rank]``, contiguous, same dtype as ``markov_w1``.
    Returns:
        ``[B, K, vocab]`` fp32 biased logits (a new tensor), bit-identical to
        ``dspark_markov_chain_logits`` with a plain argmax.
    """
    num_reqs, block_size, vocab = base_logits.shape
    rank = markov_w2.shape[1]
    device = base_logits.device
    out = torch.empty((num_reqs, block_size, vocab), dtype=base_logits.dtype, device=device)
    if num_reqs == 0 or block_size == 0:
        return out

    num_parts = triton.cdiv(vocab, _ADD_BLOCK_V)
    part_val = torch.empty((num_reqs, num_parts), dtype=torch.float32, device=device)
    part_idx = torch.empty((num_reqs, num_parts), dtype=torch.int32, device=device)
    prev_emb = torch.empty((num_reqs, rank), dtype=markov_w1.dtype, device=device)
    for step in range(block_size):
        _markov_prev_embedding_kernel[(num_reqs,)](
            part_val,
            part_idx,
            first_prev_tokens,
            markov_w1,
            prev_emb,
            vocab,
            num_parts,
            RANK=rank,
            FROM_ANCHOR=step == 0,
            BLOCK_P=triton.next_power_of_2(num_parts),
            BLOCK_R=triton.next_power_of_2(rank),
            num_warps=4,
        )
        bias = F.linear(prev_emb, markov_w2)
        _markov_add_argmax_kernel[(num_reqs, num_parts)](
            base_logits[:, step],
            out[:, step],
            bias,
            base_logits.stride(0),
            out.stride(0),
            part_val,
            part_idx,
            vocab,
            num_parts,
            BLOCK_V=_ADD_BLOCK_V,
            num_warps=4,
        )
    return out
