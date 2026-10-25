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
# DSpark / DFlash drafter forward: the paged block decode's index prelude in one launch
# =============================================================================
#
# It replaces, bit for bit, the integer lines at the top of DFlashForCausalLM.dflash_forward on a paged backend:
#   cache_seqlens     = num_ctx[:B].to(int32)
#   cache_batch_idx   = ctx_cache_batch_idx.to(int32)
#   seq_lens_after    = cache_seqlens + block_size
#   kv_indptr         = arange(0, (B + 1) * pages_per_slot, pages_per_slot)          (int32)
#   kv_last_page_len  = ((seq_lens_after - 1) % page_size) + 1                       (floored modulo, as torch)
#   append_positions  = (cache_seqlens.view(-1, 1) + arange(block_size)).reshape(-1) (int32)
#   query_pos_i32     = query_positions.reshape(-1).to(int32)
# The int64 -> int32 casts keep the low 32 bits, as torch's do. One program writes every output.

import torch
import triton
import triton.language as tl


@triton.jit
def _indices_kernel(
    num_ctx_ptr,  # [B] int32 / int64
    batch_idx_ptr,  # [B] int32 / int64
    qpos_ptr,  # [B, block] int32 / int64, strided
    qpos_s0,
    qpos_s1,
    seqlens_ptr,  # out [B] int32
    batch_out_ptr,  # out [B] int32
    after_ptr,  # out [B] int32
    indptr_ptr,  # out [B + 1] int32
    last_ptr,  # out [B] int32
    append_ptr,  # out [B * block] int32
    qflat_ptr,  # out [B * block] int32
    B,
    block,
    pages_per_slot,
    page_size,
    BLOCK_B: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    b = tl.arange(0, BLOCK_B)
    mb = b < B
    seqlens = tl.load(num_ctx_ptr + b, mask=mb, other=0).to(tl.int32)
    tl.store(seqlens_ptr + b, seqlens, mask=mb)
    tl.store(batch_out_ptr + b, tl.load(batch_idx_ptr + b, mask=mb, other=0).to(tl.int32), mask=mb)
    after = seqlens + block
    tl.store(after_ptr + b, after, mask=mb)
    r = (after - 1) % page_size
    r = tl.where(r < 0, r + page_size, r)  # torch's floored modulo for a positive divisor
    tl.store(last_ptr + b, r + 1, mask=mb)
    mi = b <= B
    tl.store(indptr_ptr + b, b * pages_per_slot, mask=mi)
    t = tl.arange(0, BLOCK_T)
    mt = t < B * block
    rb = t // block
    j = t - rb * block
    base = tl.load(num_ctx_ptr + rb, mask=mt, other=0).to(tl.int32)
    tl.store(append_ptr + t, base + j, mask=mt)
    q = tl.load(qpos_ptr + rb * qpos_s0 + j * qpos_s1, mask=mt, other=0).to(tl.int32)
    tl.store(qflat_ptr + t, q, mask=mt)


def forward_indices(
    num_ctx: torch.Tensor,
    batch_idx: torch.Tensor,
    query_positions: torch.Tensor,
    block_size: int,
    pages_per_slot: int,
    page_size: int,
):
    """The paged dflash_forward's index prelude in one launch: (cache_seqlens, cache_batch_idx, seq_lens_after,
    kv_indptr, kv_last_page_len, append_positions, query_positions_flat), all int32 and bitwise the stock lines; or
    None (nothing computed) for a layout the kernel does not reproduce, so the caller keeps the stock lines."""
    ok_dtypes = (torch.int32, torch.int64)
    B = num_ctx.shape[0] if num_ctx.dim() == 1 else -1
    if not (
        B >= 1
        and num_ctx.dtype in ok_dtypes
        and batch_idx.dtype in ok_dtypes
        and query_positions.dtype in ok_dtypes
        and batch_idx.shape == (B,)
        and query_positions.dim() == 2
        and tuple(query_positions.shape) == (B, block_size)
        and block_size >= 1
        and pages_per_slot >= 1
        and page_size >= 1
        and num_ctx.is_cuda
    ):
        return None
    dev = num_ctx.device
    i32 = dict(dtype=torch.int32, device=dev)
    seqlens = torch.empty(B, **i32)
    batch_out = torch.empty(B, **i32)
    after = torch.empty(B, **i32)
    indptr = torch.empty(B + 1, **i32)
    last = torch.empty(B, **i32)
    append = torch.empty(B * block_size, **i32)
    qflat = torch.empty(B * block_size, **i32)
    _indices_kernel[(1,)](
        num_ctx,
        batch_idx,
        query_positions,
        query_positions.stride(0),
        query_positions.stride(1),
        seqlens,
        batch_out,
        after,
        indptr,
        last,
        append,
        qflat,
        B,
        block_size,
        pages_per_slot,
        page_size,
        BLOCK_B=triton.next_power_of_2(B + 1),
        BLOCK_T=triton.next_power_of_2(B * block_size),
        num_warps=1,
    )
    return seqlens, batch_out, after, indptr, last, append, qflat
