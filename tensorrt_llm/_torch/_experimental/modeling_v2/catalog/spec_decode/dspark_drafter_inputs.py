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
# DSpark / DFlash drafter step inputs: one launch per step
# =============================================================================
#
# It replaces, bit for bit, the start of DFlashWorker.prepare_1st_drafter_inputs' generation branch, for a drafter whose
# token embedding is a plain row gather of a replicated BF16 table:
#   bonus[r]          = accepted[r, max(num_accepted[r] - 1, 0)]                       (int64)
#   query_pos[r, j]   = min(ctx_len[slot[r]] + num_accepted[r], max_ctx) + j,  j < block (int64)
#   ctx_pos[r, j]     = ctx_len[slot[r]] + j,                                 j < K + 1 (int64)
#   noise[r, 0, :]    = table[bonus[r]], or +0 for an id outside [0, vocab) (m3_embedding's masked_fill)
#   noise[r, j>0, :]  = mask_row   (the drafter's trained mask embedding, else table[mask_token_id])
# Integer arithmetic and row copies only, so the result is exact; the stock path spends a launch per torch op on it.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# acc_stride and nacc_ptr depend on the step's batch composition (the accepted-token buffer's row stride; a view
# starting after the context requests), so Triton's divisibility-16 specialization of either would compile a new
# variant at a batch shape warmup never ran (a post-ready late compile, seen at 3 scheduled requests). Both feed
# scalar per-program loads only, so not specializing them costs nothing.
@triton.jit(do_not_specialize=["acc_stride"], do_not_specialize_on_alignment=["nacc_ptr"])
def _inputs_kernel(
    acc_ptr,  # accepted tokens [n, acc_stride] (int32 / int64)
    acc_stride,
    nacc_ptr,  # accepted counts [n] (int)
    slots_ptr,  # [n] int64: the request's ctx slot
    ctx_len_ptr,  # [slots] int64
    max_ctx,
    table_ptr,  # [vocab, hidden] bf16 embedding table
    mask_ptr,  # [hidden] bf16 mask row
    bonus_ptr,  # out [n] int64
    qpos_ptr,  # out [n, BLOCK_Q] int64
    cpos_ptr,  # out [n, KP1] int64
    noise_ptr,  # out [n, BLOCK_Q, hidden] bf16
    hidden,
    vocab,
    BLOCK_Q: tl.constexpr,
    KP1: tl.constexpr,
    BLOCK_QP: tl.constexpr,  # next powers of two of BLOCK_Q / KP1 (tl.arange sizes)
    KP1P: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)  # the block row this program copies
    hb = tl.program_id(2)
    nacc = tl.load(nacc_ptr + r).to(tl.int64)
    bonus = tl.load(acc_ptr + r * acc_stride + tl.maximum(nacc - 1, 0)).to(tl.int64)
    o = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    m = o < hidden
    if j == 0:
        valid = (bonus >= 0) & (bonus < vocab)
        row = tl.load(table_ptr + tl.where(valid, bonus, 0) * hidden + o, mask=m & valid, other=0.0)
    else:
        row = tl.load(mask_ptr + o, mask=m)
    tl.store(noise_ptr + (r * BLOCK_Q + j) * hidden + o, row, mask=m)
    if (j == 0) & (hb == 0):
        ctx = tl.load(ctx_len_ptr + tl.load(slots_ptr + r))
        tl.store(bonus_ptr + r, bonus)
        now = tl.minimum(ctx + nacc, max_ctx)
        q = tl.arange(0, BLOCK_QP)
        tl.store(qpos_ptr + r * BLOCK_Q + q, now + q, mask=q < BLOCK_Q)
        c = tl.arange(0, KP1P)
        tl.store(cpos_ptr + r * KP1 + c, ctx + c, mask=c < KP1)


def drafter_inputs(
    accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    slots: torch.Tensor,
    ctx_len: torch.Tensor,
    max_ctx: int,
    table: torch.Tensor,
    mask_row: torch.Tensor,
    block_size: int,
    tokens_per_request: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """(bonus [n] int64, query positions [n, block_size] int64, ctx positions [n, tokens_per_request] int64,
    noise block [n, block_size, hidden]) in one launch, or None when the inputs are not a layout the kernel reproduces
    (the caller keeps the stock path). ``mask_row`` [hidden] is the row every slot after the first takes."""
    n = num_accepted.shape[0]
    if not (
        accepted.dim() == 2
        and accepted.shape[0] == n
        and accepted.stride(1) == 1
        and accepted.dtype in (torch.int32, torch.int64)
        and num_accepted.dtype in (torch.int32, torch.int64)
        and slots.shape == (n,)
        and ctx_len.dtype == torch.int64
        and table.dim() == 2
        and table.is_contiguous()
        and mask_row.shape == (table.shape[1],)
        and mask_row.dtype == table.dtype
        and mask_row.is_contiguous()
        and block_size >= 1
        and tokens_per_request >= 1
    ):
        return None
    hidden = table.shape[1]
    dev = table.device
    bonus = torch.empty(n, dtype=torch.int64, device=dev)
    qpos = torch.empty(n, block_size, dtype=torch.int64, device=dev)
    cpos = torch.empty(n, tokens_per_request, dtype=torch.int64, device=dev)
    noise = torch.empty(n, block_size, hidden, dtype=table.dtype, device=dev)
    if n == 0:
        return bonus, qpos, cpos, noise
    block_h = 1024
    _inputs_kernel[(n, block_size, triton.cdiv(hidden, block_h))](
        accepted,
        accepted.stride(0),
        num_accepted.contiguous(),
        slots.to(torch.int64).contiguous(),
        ctx_len,
        int(max_ctx),
        table,
        mask_row,
        bonus,
        qpos,
        cpos,
        noise,
        hidden,
        table.shape[0],
        BLOCK_Q=block_size,
        KP1=tokens_per_request,
        BLOCK_QP=triton.next_power_of_2(block_size),
        KP1P=triton.next_power_of_2(tokens_per_request),
        BLOCK_H=block_h,
        num_warps=4,
    )
    return bonus, qpos, cpos, noise
