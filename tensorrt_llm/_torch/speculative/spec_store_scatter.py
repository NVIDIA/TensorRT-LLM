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
"""One-launch scatter of a speculative step's outputs into the ``SpecSampler`` store.

``SpecSampler.sample_async`` copies each sampled request's accepted tokens, next-step input
tokens, accepted count and next drafts into its slot of four store tensors. Where the
``spec_step_copies.SlotScatter`` kernels are not supported, one Triton kernel writes all four
(instead of four ``index_copy_`` launches), the three host-read store tensors are views of one
contiguous buffer that a single D2H copy snapshots, and the slot index tensor is reused while the
batch's slots are unchanged.
"""

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
    host_new,  # HOST_OUT: host-mapped copies of new_dst / lens_dst / draft_dst (same layout)
    host_lens,
    host_draft,
    NEW_WIDTH: tl.constexpr,
    NEXT_WIDTH: tl.constexpr,
    DRAFT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    HOST_OUT: tl.constexpr = False,
):
    """One sampled request: its four outputs into store slot ``slots[request]``."""
    request = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + request).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < NEW_WIDTH
    values = tl.load(new_src + request * new_src_row + cols * new_src_col, mask=mask)
    tl.store(new_dst + cols * new_dst_col + slot * new_dst_slot, values, mask=mask)
    if HOST_OUT:
        tl.store(host_new + cols * new_dst_col + slot * new_dst_slot, values, mask=mask)
    mask = cols < NEXT_WIDTH
    values = tl.load(next_src + request * next_src_row + cols * next_src_col, mask=mask)
    tl.store(next_dst + cols * next_dst_col + slot * next_dst_slot, values, mask=mask)
    if DRAFT_WIDTH > 0:
        mask = cols < DRAFT_WIDTH
        values = tl.load(draft_src + request * draft_src_row + cols * draft_src_col, mask=mask)
        tl.store(draft_dst + slot * draft_dst_slot + cols, values, mask=mask)
        if HOST_OUT:
            tl.store(host_draft + slot * draft_dst_slot + cols, values, mask=mask)
    count = tl.load(lens_src + request * lens_src_row)
    tl.store(lens_dst + slot, count)
    if HOST_OUT:
        tl.store(host_lens + slot, count)


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
    host_out: "tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None" = None,
) -> None:
    """``store_*[..., slots[i], ...] = source[i]`` for every sampled request ``i``.

    Sources are ``[requests, width]`` (``new_tokens_lens``: ``[requests]``); the token stores
    are ``[width, slots, 1]``, the draft store ``[slots, draft width]``, the count store
    ``[slots]``. Widths must already match the stores, as for ``index_copy_``.
    ``host_out``: pinned host tensors laid out like ``store_new_tokens``, ``store_new_tokens_lens``
    and ``store_next_draft_tokens`` that also receive the written slots (through their device
    mapping), in place of a D2H snapshot of those three stores.
    """
    requests = slots.numel()
    if not requests:
        return
    new_width, next_width = store_new_tokens.shape[0], store_next_new_tokens.shape[0]
    draft_width = store_next_draft_tokens.shape[1]
    if (
        new_tokens.shape != (requests, new_width)
        or next_new_tokens.shape != (requests, next_width)
        or next_draft_tokens.shape != (requests, draft_width)
        or new_tokens_lens.shape != (requests,)
        or store_next_draft_tokens.stride(1) != 1
    ):
        raise ValueError("spec store scatter: source shapes must match the store widths")
    if host_out is not None and (
        host_out[0].stride() != store_new_tokens.stride()
        or host_out[1].stride() != store_new_tokens_lens.stride()
        or host_out[2].stride() != store_next_draft_tokens.stride()
    ):
        raise ValueError("spec store scatter: host outputs must have the store layouts")
    host_new, host_lens, host_draft = (
        host_out
        if host_out is not None
        else (store_new_tokens, store_new_tokens_lens, store_next_draft_tokens)
    )
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
        host_new,
        host_lens,
        host_draft,
        NEW_WIDTH=new_width,
        NEXT_WIDTH=next_width,
        DRAFT_WIDTH=draft_width,
        BLOCK=triton.next_power_of_2(max(new_width, next_width, draft_width, 1)),
        HOST_OUT=host_out is not None,
    )
