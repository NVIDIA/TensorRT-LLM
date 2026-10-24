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
# =============================================================================
# DSpark vanilla-Markov head: the biased block logits' glue around the bias GEMM, in two launches
# =============================================================================
#
# It replaces, bit for bit, the elementwise lines of the batched biased-logits form (the tp targets' S4-logits path)
#   prevs  = cat([first_prev.view(B, 1), tokens[:, :K - 1].long()], 1).reshape(-1)
#   emb    = markov_prev_embeddings(prevs, w1)     (get_masked_input_and_mask, F.embedding, masked_fill: an anchor
#                                                   outside [0, V) gives a zero row)
#   bias   = F.linear(emb, w2_shard)                (unchanged: the caller's GEMM)
#   out    = shard_logits.float() + bias.view(B, K, N).to(float32)
# with ``prev_embeddings`` (one launch for the 11 index / mask / gather / fill launches) and ``widen_add`` (one launch
# for the two widening copies and the add). The gather copies w1's rows bit for bit (zero rows are +0.0, as
# masked_fill(0)); the add is one fp32 IEEE add of two exactly widened bf16 values, as torch's.

import torch
import triton
import triton.language as tl


@triton.jit
def _prev_embeddings_kernel(
    first_prev_ptr,  # [B] int32 / int64
    tokens_ptr,  # [B, K] int32 / int64, contiguous
    w1_ptr,  # [V, R] bf16, contiguous
    out_ptr,  # out [B * K, R] bf16
    K,
    V,
    R: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // K
    k = row - b * K
    if k == 0:
        prev = tl.load(first_prev_ptr + b).to(tl.int64)
    else:
        prev = tl.load(tokens_ptr + b * K + k - 1).to(tl.int64)
    valid = (prev >= 0) & (prev < V)
    safe = tl.where(valid, prev, 0)
    c = tl.arange(0, R)
    vals = tl.load(w1_ptr + safe * R + c)
    vals = tl.where(valid, vals, tl.zeros_like(vals))
    tl.store(out_ptr + row.to(tl.int64) * R + c, vals)


@triton.jit
def _widen_add_kernel(
    logits_ptr,  # [M, N] bf16, contiguous
    bias_ptr,  # [M, N] bf16, contiguous
    out_ptr,  # out [M, N] fp32
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < N
    a = tl.load(logits_ptr + row * N + c, mask=m, other=0.0).to(tl.float32)
    bb = tl.load(bias_ptr + row * N + c, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * N + c, a + bb, mask=m)


_INT = (torch.int32, torch.int64)
WIDEN_BLOCK = 4096


def supports(
    shard_logits: torch.Tensor, first_prev: torch.Tensor, tokens: torch.Tensor, w1: torch.Tensor
) -> bool:
    """Whether both launches take these operands; a caller that gets False runs the in-tree lines."""
    if shard_logits.dim() != 3 or tokens.dim() != 2 or first_prev.dim() != 1 or w1.dim() != 2:
        return False
    B, K, _ = shard_logits.shape
    R = w1.shape[1]
    return (
        shard_logits.is_cuda
        and shard_logits.dtype == torch.bfloat16
        and shard_logits.is_contiguous()
        and w1.dtype == torch.bfloat16
        and w1.is_contiguous()
        and w1.device == shard_logits.device
        and R >= 16
        and (R & (R - 1)) == 0
        and R <= 1024
        and first_prev.dtype in _INT
        and tokens.dtype in _INT
        and first_prev.is_contiguous()
        and tokens.is_contiguous()
        and first_prev.numel() == B
        and tuple(tokens.shape) == (B, K)
        and B > 0
        and K > 0
    )


def prev_embeddings(
    first_prev: torch.Tensor, tokens: torch.Tensor, w1: torch.Tensor
) -> torch.Tensor:
    """[B * K, R] bf16: row b * K + k is w1[prev] with prev = first_prev[b] (k = 0) or tokens[b, k - 1], and a zero
    row for a prev outside [0, V) -- bitwise markov_prev_embeddings of the in-tree prevs."""
    B, K = tokens.shape
    V, R = w1.shape
    out = torch.empty(B * K, R, dtype=w1.dtype, device=w1.device)
    _prev_embeddings_kernel[(B * K,)](first_prev, tokens, w1, out, K, V, R=R, num_warps=1)
    return out


def widen_add(shard_logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """fp32 ``shard_logits.float() + bias.to(float32)`` ([B, K, N] bf16 each, contiguous), bitwise torch's."""
    B, K, N = shard_logits.shape
    if bias.dtype != torch.bfloat16 or not bias.is_contiguous() or bias.numel() != B * K * N:
        raise ValueError("widen_add: bias must be contiguous bf16 with the logits' element count")
    out = torch.empty(B, K, N, dtype=torch.float32, device=shard_logits.device)
    grid = (B * K, triton.cdiv(N, WIDEN_BLOCK))
    _widen_add_kernel[grid](shard_logits, bias, out, N, BLOCK=WIDEN_BLOCK, num_warps=8)
    return out
