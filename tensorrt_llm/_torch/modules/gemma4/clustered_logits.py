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
"""Clustered-vocabulary logits (Triton): logits only for the tokens of the top-scoring clusters.

Replaces ``topk`` + gather + ``bmm`` + ``scatter_``. One kernel selects the top-``K`` clusters per
row with a radix select and, in the same launch, fills the logits with the fill value; a second
writes each selected token's logit to its vocabulary position. A logit can differ from the bf16
``bmm`` by one ulp; ties at the ``K``-th score go to the lower cluster index.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# Tokens per program of the logits kernel; with 4 warps it reads a 32 x H tile of embedding rows.
_BLOCK_TOKENS = 32
# Logits elements written per fill program of the select kernel.
_FILL_BLOCK = 4096


@triton.jit
def _select_kernel(
    scores_ptr,  # [R, C] float, row stride SS
    idx_ptr,  # [R, K] int32
    fill_ptr,  # [NFILL] fp32, contiguous (unused when NFILL == 0)
    SS,
    R,
    NFILL,
    FILL,
    C: tl.constexpr,
    K: tl.constexpr,
    FB: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= R:
        # Programs past the R selection rows fill the logits; the selection rows are latency
        # bound, so the fill runs alongside them.
        offs = (pid - R).to(tl.int64) * FB + tl.arange(0, FB)
        tl.store(fill_ptr + offs, tl.full([FB], FILL, tl.float32), mask=offs < NFILL)
    else:
        cols = tl.arange(0, C)
        s = tl.load(scores_ptr + pid * SS + cols).to(tl.float32)
        # Order-preserving unsigned 32-bit image of the fp32 score (held in int64).
        bits = s.to(tl.int32, bitcast=True)
        key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64) + 2147483648
        # Radix select, 8 bits per pass: find the K-th largest key and how many entries equal to
        # it are still needed.
        bins = tl.arange(0, 256)
        prefix = tl.zeros([], tl.int64)
        need = tl.full([], K, tl.int32)
        for p in tl.static_range(4):
            digit = ((key >> (24 - 8 * p)) & 255).to(tl.int32)
            if p == 0:
                hist = tl.histogram(digit, 256)
            else:
                hist = tl.histogram(digit, 256, mask=(key >> (32 - 8 * p)) == prefix)
            at_least = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist
            b = tl.max(tl.where(at_least >= need, bins, -1), axis=0)
            need -= tl.sum(tl.where(bins > b, hist, 0), axis=0)
            prefix = (prefix << 8) | b.to(tl.int64)
        # Everything above the K-th key, plus the lowest-index entries equal to it.
        eq = key == prefix
        sel = (key > prefix) | (eq & (tl.cumsum(eq.to(tl.int32), axis=0) <= need))
        pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(idx_ptr + pid * K + pos, cols, mask=sel)


@triton.jit
def _clustered_logits_kernel(
    hidden_ptr,  # [R, H], row stride SH
    weight_ptr,  # [V, H], row stride SW
    ordering_ptr,  # [V] int64: token ids, cluster by cluster
    top_ptr,  # [R, K] int32 selected clusters
    out_ptr,  # [R, V] fp32, row stride SO
    SH,
    SW,
    SO,
    K: tl.constexpr,
    CSIZE: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.program_id(1)
    cluster = tl.load(top_ptr + row * K + slot).to(tl.int64)
    j = tl.program_id(2) * BT + tl.arange(0, BT)
    tokens = tl.load(ordering_ptr + cluster * CSIZE + j)
    d = tl.arange(0, H)
    h = tl.load(hidden_ptr + row * SH + d).to(tl.float32)
    w = tl.load(weight_ptr + tokens[:, None] * SW + d[None, :]).to(tl.float32)
    logit = tl.sum(w * h[None, :], axis=1)
    logit = logit.to(weight_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + row.to(tl.int64) * SO + tokens, logit)


def _select(
    scores: torch.Tensor,
    k: int,
    fill_out: Optional[torch.Tensor] = None,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Top-``k`` indices per row (see ``topk_indices``); also fills ``fill_out`` in the same launch."""
    assert scores.dim() == 2 and scores.stride(-1) == 1
    rows, c = scores.shape
    assert c & (c - 1) == 0 and 0 < k <= c
    idx = torch.empty((rows, k), dtype=torch.int32, device=scores.device)
    nfill = fill_out.numel() if fill_out is not None else 0
    assert fill_out is None or fill_out.is_contiguous()
    if rows or nfill:
        _select_kernel[(rows + triton.cdiv(nfill, _FILL_BLOCK),)](
            scores,
            idx,
            fill_out if fill_out is not None else scores,
            scores.stride(0),
            rows,
            nfill,
            fill_value,
            C=c,
            K=k,
            FB=_FILL_BLOCK,
            num_warps=8,
        )
    return idx


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` largest entries of each row of ``scores`` [R, C] (C a power of two),
    in ascending index order, ties at the ``k``-th score to the lower index. Returns [R, k] int32."""
    return _select(scores, k)


def clustered_logits(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    token_ordering: torch.Tensor,
    cluster_scores: torch.Tensor,
    top_k: int,
    cluster_size: int,
    fill_value: float,
) -> torch.Tensor:
    """Full-vocabulary fp32 logits where only the tokens of the top-scoring clusters are computed.

    Args:
        hidden_states: [R, H] (H a power of two), the head input.
        weight: [V, H] output embedding, same dtype as ``hidden_states``.
        token_ordering: [V] int64 token ids listed cluster by cluster.
        cluster_scores: [R, C] cluster scores (C a power of two); the ``top_k`` best are computed.
        top_k: clusters selected per row.
        cluster_size: tokens per cluster (a power of two).
        fill_value: logit of every token outside the selected clusters.

    Returns:
        [R, V] float32: ``hidden_states[r] . weight[t]`` (rounded to the embedding dtype) for each
        token ``t`` of row ``r``'s selected clusters, ``fill_value`` elsewhere.
    """
    rows, hidden = hidden_states.shape
    vocab = weight.shape[0]
    assert weight.shape[1] == hidden and hidden & (hidden - 1) == 0
    assert cluster_size & (cluster_size - 1) == 0 and vocab % cluster_size == 0
    assert cluster_scores.shape == (rows, vocab // cluster_size)
    assert hidden_states.stride(-1) == 1 and weight.stride(-1) == 1
    assert token_ordering.dtype == torch.int64 and token_ordering.numel() == vocab
    out = torch.empty((rows, vocab), dtype=torch.float32, device=weight.device)
    if rows:
        top_clusters = _select(cluster_scores, top_k, out, fill_value)
        bt = min(cluster_size, _BLOCK_TOKENS)
        _clustered_logits_kernel[(rows, top_k, cluster_size // bt)](
            hidden_states,
            weight,
            token_ordering,
            top_clusters,
            out,
            hidden_states.stride(0),
            weight.stride(0),
            out.stride(0),
            K=top_k,
            CSIZE=cluster_size,
            H=hidden,
            BT=bt,
            num_warps=4,
        )
    return out
