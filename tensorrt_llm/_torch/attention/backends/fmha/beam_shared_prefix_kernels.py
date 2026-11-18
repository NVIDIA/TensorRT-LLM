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

"""Triton kernel of the beam_shared_prefix library: the tail attention, the KV write and the merge.

Kept apart from ``beam_shared_prefix`` so that registering the library (always done, the library
being opt-in) does not compile or import any Triton code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from .beam_shared_prefix import _StepIndices

_LOG2E = 1.4426950408889634


# tail_len and stride_tail_row vary between eager steps; keep Triton from compiling a variant at
# tail_len == 1 and at multiples of 16.
@triton.jit(do_not_specialize=["tail_len", "stride_tail_row"])
def _tail_merge_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pool_ptr,
    prefix_out_ptr,
    prefix_lse_ptr,
    out_ptr,
    write_page_ptr,
    write_slot_ptr,
    tail_page_ptr,
    tail_slot_ptr,
    tail_valid_ptr,
    prefix_empty_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_row,
    stride_k_head,
    stride_v_row,
    stride_v_head,
    stride_pool_page,
    stride_pool_kv,
    stride_pool_head,
    stride_pool_slot,
    stride_prefix_row,
    stride_prefix_head,
    stride_lse_row,
    stride_lse_head,
    stride_out_row,
    stride_out_head,
    stride_tail_row,
    tail_len,
    sm_scale_log2,
    LSE_SCALE: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One program per (row, KV head): write the current K/V, attend the tail, merge.

    Online softmax in the log2 domain, started from the prefix state: the prefix
    output is already normalized, so it enters with weight 1 at max ``lse``.
    """
    row = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    group = tl.arange(0, GROUP_PAD)
    dim = tl.arange(0, HEAD_DIM)
    group_mask = group < GROUP
    head = kv_head * GROUP + group

    q = tl.load(
        q_ptr + row * stride_q_row + head[:, None] * stride_q_head + dim[None, :],
        mask=group_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    k_raw = tl.load(k_ptr + row * stride_k_row + kv_head * stride_k_head + dim)
    v_raw = tl.load(v_ptr + row * stride_v_row + kv_head * stride_v_head + dim)

    write_page = tl.load(write_page_ptr + row)
    write_slot = tl.load(write_slot_ptr + row)
    write_base = (
        pool_ptr
        + write_page * stride_pool_page
        + kv_head * stride_pool_head
        + write_slot * stride_pool_slot
    )
    tl.store(write_base + dim, k_raw)
    tl.store(write_base + stride_pool_kv + dim, v_raw)

    empty = tl.load(prefix_empty_ptr + row) != 0
    lse = (
        tl.load(
            prefix_lse_ptr + row * stride_lse_row + head * stride_lse_head,
            mask=group_mask,
            other=float("-inf"),
        )
        * LSE_SCALE
    )
    acc = tl.load(
        prefix_out_ptr
        + row * stride_prefix_row
        + head[:, None] * stride_prefix_head
        + dim[None, :],
        mask=group_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    m = tl.where(empty, float("-inf"), lse)
    denom = tl.where(group_mask & (~empty), 1.0, 0.0)
    acc = tl.where(empty, 0.0, acc)

    for t in range(0, tail_len):
        valid = tl.load(tail_valid_ptr + row * stride_tail_row + t) != 0
        page = tl.load(tail_page_ptr + row * stride_tail_row + t)
        slot = tl.load(tail_slot_ptr + row * stride_tail_row + t)
        base = (
            pool_ptr
            + page * stride_pool_page
            + kv_head * stride_pool_head
            + slot * stride_pool_slot
        )
        k = tl.load(base + dim, mask=valid, other=0.0).to(tl.float32)
        v = tl.load(base + stride_pool_kv + dim, mask=valid, other=0.0).to(tl.float32)
        score = tl.sum(q * k[None, :], axis=1) * sm_scale_log2
        score = tl.where(valid, score, float("-inf"))
        m_new = tl.maximum(m, score)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m - m_safe)
        p = tl.exp2(score - m_safe)
        acc = acc * alpha[:, None] + p[:, None] * v[None, :]
        denom = denom * alpha + p
        m = m_new

    score = tl.sum(q * k_raw.to(tl.float32)[None, :], axis=1) * sm_scale_log2
    m_new = tl.maximum(m, score)
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m - m_safe)
    p = tl.exp2(score - m_safe)
    acc = acc * alpha[:, None] + p[:, None] * v_raw.to(tl.float32)[None, :]
    denom = denom * alpha + p

    out = acc / denom[:, None]
    tl.store(
        out_ptr + row * stride_out_row + head[:, None] * stride_out_head + dim[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=group_mask[:, None],
    )


def fused_tail_and_merge(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pool: torch.Tensor,
    indices: "_StepIndices",
    prefix_out: torch.Tensor,
    prefix_lse: torch.Tensor,
    output: torch.Tensor,
    *,
    tail_len: int,
    num_kv_heads: int,
    sm_scale: float,
    lse_scale: float = 1.0,
    lse_heads_first: bool = False,
) -> None:
    """Write the current K/V, attend the tail and merge with the prefix, in one launch.

    Same contract as ``merge_prefix_and_tail`` plus the KV write of the current
    token: ``q`` is [rows, heads, dim], ``k`` and ``v`` are [rows, kv_heads, dim],
    ``pool`` is the layer's [pages, 2, kv_heads, tokens_per_block, dim] view, and
    ``output`` receives the merged [rows, heads, dim] result. ``prefix_lse`` is [rows, heads]
    (FlashInfer) or, with ``lse_heads_first``, [heads, capacity >= rows] (FlashAttention-4);
    ``lse_scale`` turns it into base 2 (1 for FlashInfer, log2(e) for FA4's natural log).
    """
    rows, num_heads, head_dim = q.shape
    group = num_heads // num_kv_heads
    for tensor in (q, k, v):
        assert tensor.stride(-1) == 1
    tail_page = indices.tail_page if tail_len > 0 else indices.write_page
    tail_slot = indices.tail_slot if tail_len > 0 else indices.write_slot
    # With no tail the kernel never reads these three; any valid tensors of the right type do.
    tail_valid = (
        indices.tail_valid.view(torch.int8)
        if tail_len > 0
        else indices.prefix_empty.view(torch.int8)
    )
    _tail_merge_kernel[(rows, num_kv_heads)](
        q,
        k,
        v,
        pool,
        prefix_out,
        prefix_lse,
        output,
        indices.write_page,
        indices.write_slot,
        tail_page,
        tail_slot,
        tail_valid,
        indices.prefix_empty.view(torch.int8),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        pool.stride(0),
        pool.stride(1),
        pool.stride(2),
        pool.stride(3),
        prefix_out.stride(0),
        prefix_out.stride(1),
        prefix_lse.stride(1 if lse_heads_first else 0),
        prefix_lse.stride(0 if lse_heads_first else 1),
        output.stride(0),
        output.stride(1),
        tail_page.stride(0) if tail_len > 0 else 0,
        tail_len,
        sm_scale * _LOG2E,
        LSE_SCALE=lse_scale,
        GROUP=group,
        GROUP_PAD=triton.next_power_of_2(group),
        HEAD_DIM=head_dim,
    )
