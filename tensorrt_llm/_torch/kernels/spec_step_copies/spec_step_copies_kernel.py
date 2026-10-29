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
"""The one-model speculative decoding step's eager copy passes, one kernel each.

* ``scatter_kernel``: the sampler moves the forward's per-row outputs into its slot-indexed stores (``index_copy_``
  of new tokens, next new tokens, accepted lengths and next draft tokens, each padded or cut to its store width;
  a row's new tokens past its accepted length are written as zeros).
* ``gather_kernel``: the overlap scheduler's gathers from those stores into a decode step's inputs, for the rows whose
  request ran in the previous step (``index_select`` into input ids and draft tokens by slot, into the position
  offsets by the per-token index list, and into the KV-length offsets by slot, minus the tokens per step).

Every buffer is int32 and passed as a raw address, so a call passes integers only. Row and element offsets are
arguments. One thread per element.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute

THREADS = 128


def _i32_at(address, index):
    """A one-element int32 tensor at ``address + 4 * index`` in global memory (trace-time helper)."""
    ptr = cute.make_ptr(
        cutlass.Int32,
        address + cutlass.Int64(index) * cutlass.Int64(4),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    return cute.make_tensor(ptr, cute.make_layout((1,)))


@cute.kernel
def scatter_kernel(
    out_new_tokens: cutlass.Int64,  # int32 [rows_total, out_new_width], the forward's accepted tokens
    out_next_new_tokens: cutlass.Int64,  # int32 [rows_total, out_next_width]
    out_lens: cutlass.Int64,  # int32 [rows_total]
    out_next_draft: cutlass.Int64,  # int32 [rows_total, out_draft_width]
    slots: cutlass.Int64,  # int32 [rows]
    store_new_tokens: cutlass.Int64,  # int32 [new_width, num_slots] (token-major)
    store_next_new_tokens: cutlass.Int64,  # int32 [next_width, num_slots]
    store_lens: cutlass.Int64,  # int32 [num_slots]
    store_next_draft: cutlass.Int64,  # int32 [num_slots, draft_width]
    rows: cutlass.Int32,
    row_begin: cutlass.Int32,
    columns: cutlass.Int32,
    num_slots: cutlass.Int32,
    new_width: cutlass.Int32,
    out_new_width: cutlass.Int32,
    next_width: cutlass.Int32,
    out_next_width: cutlass.Int32,
    draft_width: cutlass.Int32,
    out_draft_width: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    t = bidx * cutlass.Int32(THREADS) + tidx
    if t < rows * columns:
        row = t // columns
        col = t - row * columns
        src_row = row_begin + row
        slot = _i32_at(slots, row)[0]
        accepted = _i32_at(out_lens, src_row)[0]
        zero = cutlass.Int32(0)
        if col < new_width:
            # Only the row's accepted tokens are read: past them a context row's forward output is never written.
            value = zero
            if col < out_new_width:
                if col < accepted:
                    value = _i32_at(out_new_tokens, src_row * out_new_width + col)[0]
            _i32_at(store_new_tokens, col * num_slots + slot)[0] = value
        if col < next_width:
            value = zero
            if col < out_next_width:
                value = _i32_at(out_next_new_tokens, src_row * out_next_width + col)[0]
            _i32_at(store_next_new_tokens, col * num_slots + slot)[0] = value
        if col < draft_width:
            value = zero
            if col < out_draft_width:
                value = _i32_at(out_next_draft, src_row * out_draft_width + col)[0]
            _i32_at(store_next_draft, slot * draft_width + col)[0] = value
        if col == zero:
            _i32_at(store_lens, slot)[0] = accepted


@cute.jit
def scatter(
    out_new_tokens: cutlass.Int64,
    out_next_new_tokens: cutlass.Int64,
    out_lens: cutlass.Int64,
    out_next_draft: cutlass.Int64,
    slots: cutlass.Int64,
    store_new_tokens: cutlass.Int64,
    store_next_new_tokens: cutlass.Int64,
    store_lens: cutlass.Int64,
    store_next_draft: cutlass.Int64,
    rows: cutlass.Int32,
    row_begin: cutlass.Int32,
    columns: cutlass.Int32,
    num_slots: cutlass.Int32,
    new_width: cutlass.Int32,
    out_new_width: cutlass.Int32,
    next_width: cutlass.Int32,
    out_next_width: cutlass.Int32,
    draft_width: cutlass.Int32,
    out_draft_width: cutlass.Int32,
    stream: cuda_driver.CUstream,
) -> None:
    scatter_kernel(
        out_new_tokens, out_next_new_tokens, out_lens, out_next_draft, slots, store_new_tokens,
        store_next_new_tokens, store_lens, store_next_draft, rows, row_begin, columns, num_slots, new_width,
        out_new_width, next_width, out_next_width, draft_width, out_draft_width,
    ).launch(
        grid=[(rows * columns + THREADS - 1) // THREADS, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )  # fmt: skip


@cute.kernel
def gather_kernel(
    store_next_new_tokens: cutlass.Int64,  # int32 [next_width, num_slots] (token-major)
    store_next_draft: cutlass.Int64,  # int32 [num_slots, draft_stride]
    store_lens: cutlass.Int64,  # int32 [num_slots]
    slots: cutlass.Int64,  # int32 [rows]
    pos_indices: cutlass.Int64,  # int32 [rows * tokens_per_row]
    input_ids: cutlass.Int64,
    draft_tokens: cutlass.Int64,
    pos_offsets: cutlass.Int64,
    kv_offsets: cutlass.Int64,
    rows: cutlass.Int32,
    tokens_per_row: cutlass.Int32,
    draft_width: cutlass.Int32,
    num_slots: cutlass.Int32,
    draft_stride: cutlass.Int32,
    input_begin: cutlass.Int32,
    draft_begin: cutlass.Int32,
    pos_begin: cutlass.Int32,
    kv_begin: cutlass.Int32,
):
    """For row r < rows, slot s = slots[r] and token j < tokens_per_row: ``input_ids[input_begin + r * tokens_per_row
    + j] = store_next_new_tokens[j, s]``, ``pos_offsets[pos_begin + r * tokens_per_row + j] = store_lens[pos_indices[r
    * tokens_per_row + j]]``, ``draft_tokens[draft_begin + r * draft_width + j] = store_next_draft[s, j]`` for j <
    draft_width and ``kv_offsets[kv_begin + r] = store_lens[s] - tokens_per_row``. One thread per (row, token)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    t = bidx * cutlass.Int32(THREADS) + tidx
    if t < rows * tokens_per_row:
        row = t // tokens_per_row
        col = t - row * tokens_per_row
        slot = _i32_at(slots, row)[0]
        _i32_at(input_ids, input_begin + t)[0] = _i32_at(
            store_next_new_tokens, col * num_slots + slot
        )[0]
        _i32_at(pos_offsets, pos_begin + t)[0] = _i32_at(store_lens, _i32_at(pos_indices, t)[0])[0]
        if col < draft_width:
            _i32_at(draft_tokens, draft_begin + row * draft_width + col)[0] = _i32_at(
                store_next_draft, slot * draft_stride + col
            )[0]
        if col == cutlass.Int32(0):
            _i32_at(kv_offsets, kv_begin + row)[0] = _i32_at(store_lens, slot)[0] - tokens_per_row


@cute.jit
def gather(
    store_next_new_tokens: cutlass.Int64,
    store_next_draft: cutlass.Int64,
    store_lens: cutlass.Int64,
    slots: cutlass.Int64,
    pos_indices: cutlass.Int64,
    input_ids: cutlass.Int64,
    draft_tokens: cutlass.Int64,
    pos_offsets: cutlass.Int64,
    kv_offsets: cutlass.Int64,
    rows: cutlass.Int32,
    tokens_per_row: cutlass.Int32,
    draft_width: cutlass.Int32,
    num_slots: cutlass.Int32,
    draft_stride: cutlass.Int32,
    input_begin: cutlass.Int32,
    draft_begin: cutlass.Int32,
    pos_begin: cutlass.Int32,
    kv_begin: cutlass.Int32,
    stream: cuda_driver.CUstream,
) -> None:
    gather_kernel(
        store_next_new_tokens, store_next_draft, store_lens, slots, pos_indices, input_ids, draft_tokens, pos_offsets,
        kv_offsets, rows, tokens_per_row, draft_width, num_slots, draft_stride, input_begin, draft_begin, pos_begin,
        kv_begin,
    ).launch(
        grid=[(rows * tokens_per_row + THREADS - 1) // THREADS, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )  # fmt: skip
