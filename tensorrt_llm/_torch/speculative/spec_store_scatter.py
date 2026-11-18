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
"""One-launch scatter of a speculative step's outputs into the ``SpecSampler`` store."""

import torch
import triton
import triton.language as tl


@triton.jit
def _spec_store_scatter_kernel(
    slots_ptr,
    new_src,
    new_src_row,
    new_src_col,
    next_src,
    next_src_row,
    next_src_col,
    lens_src,
    lens_src_row,
    draft_src,
    draft_src_row,
    draft_src_col,
    new_dst,
    new_dst_col,
    new_dst_slot,
    next_dst,
    next_dst_col,
    next_dst_slot,
    lens_dst,
    draft_dst,
    draft_dst_slot,
    NEW_WIDTH: tl.constexpr,
    NEXT_WIDTH: tl.constexpr,
    DRAFT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + request).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < NEW_WIDTH
    values = tl.load(new_src + request * new_src_row + cols * new_src_col, mask=mask)
    tl.store(new_dst + cols * new_dst_col + slot * new_dst_slot, values, mask=mask)
    mask = cols < NEXT_WIDTH
    values = tl.load(next_src + request * next_src_row + cols * next_src_col, mask=mask)
    tl.store(next_dst + cols * next_dst_col + slot * next_dst_slot, values, mask=mask)
    if DRAFT_WIDTH > 0:
        mask = cols < DRAFT_WIDTH
        values = tl.load(draft_src + request * draft_src_row + cols * draft_src_col, mask=mask)
        tl.store(draft_dst + slot * draft_dst_slot + cols, values, mask=mask)
    tl.store(lens_dst + slot, tl.load(lens_src + request * lens_src_row))


def spec_store_scatter(
    slots: torch.Tensor,
    new_tokens: torch.Tensor,
    next_new_tokens: torch.Tensor,
    new_tokens_lens: torch.Tensor,
    next_draft_tokens: torch.Tensor,
    store_new_tokens: torch.Tensor,
    store_next_new_tokens: torch.Tensor,
    store_new_tokens_lens: torch.Tensor,
    store_next_draft_tokens: torch.Tensor,
) -> None:
    """``store_*[..., slots[i], ...] = source[i]`` for each request ``i``, in one launch.

    ``slots`` are distinct; ``[R, width]`` int32 sources match the store widths, and
    ``store_next_draft_tokens`` has unit column stride.
    """
    requests = slots.numel()
    if not requests:
        return
    new_width, next_width = store_new_tokens.shape[0], store_next_new_tokens.shape[0]
    draft_width = store_next_draft_tokens.shape[1]
    _spec_store_scatter_kernel[(requests,)](
        slots,
        new_tokens,
        new_tokens.stride(0),
        new_tokens.stride(1),
        next_new_tokens,
        next_new_tokens.stride(0),
        next_new_tokens.stride(1),
        new_tokens_lens,
        new_tokens_lens.stride(0),
        next_draft_tokens,
        next_draft_tokens.stride(0),
        next_draft_tokens.stride(1),
        store_new_tokens,
        store_new_tokens.stride(0),
        store_new_tokens.stride(1),
        store_next_new_tokens,
        store_next_new_tokens.stride(0),
        store_next_new_tokens.stride(1),
        store_new_tokens_lens,
        store_next_draft_tokens,
        store_next_draft_tokens.stride(0),
        NEW_WIDTH=new_width,
        NEXT_WIDTH=next_width,
        DRAFT_WIDTH=draft_width,
        BLOCK=triton.next_power_of_2(max(new_width, next_width, draft_width, 1)),
    )
