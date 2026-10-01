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
  of new tokens, next new tokens, accepted lengths and next draft tokens, each padded or cut to its store width).
* ``stage_kernel``: one launch writes a decode step's per-step inputs: the overlap scheduler's gathers from those
  stores for the rows whose request ran in the previous step (``index_select`` into input ids and draft tokens by
  slot, into the position offsets by the per-token index list, and into the KV-length offsets by slot, minus the
  tokens per step), up to four host-to-device copies whose values it reads in place from a pinned host record, and up
  to two KV cache block-offset copies.

Every buffer is int32 and passed as a raw address, so a call passes integers only. Row and element offsets are
arguments. One thread per element.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute

THREADS = 128


def _i32_at(address, index):
    """A one-element int32 tensor at ``address + 4 * index`` (trace-time helper; global or mapped host memory)."""
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
        zero = cutlass.Int32(0)
        if col < new_width:
            value = zero
            if col < out_new_width:
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
            _i32_at(store_lens, slot)[0] = _i32_at(out_lens, src_row)[0]


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


STAGED_COPIES = 4
BLOCK_COPIES = 2


@cute.kernel
def stage_kernel(
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
    record: cutlass.Int64,  # int32 values in pinned host memory, read in place
    dst0: cutlass.Int64,
    dst1: cutlass.Int64,
    dst2: cutlass.Int64,
    dst3: cutlass.Int64,
    off0: cutlass.Int32,
    off1: cutlass.Int32,
    off2: cutlass.Int32,
    off3: cutlass.Int32,
    n0: cutlass.Int32,
    n1: cutlass.Int32,
    n2: cutlass.Int32,
    n3: cutlass.Int32,
    table0: cutlass.Int64,  # block copy 0: int32 [pools, table_seqs, 2, blocks] in pinned host memory
    offsets0: cutlass.Int64,  # int32 [pools, offsets_seqs, 2, blocks] (device)
    copy_index0: cutlass.Int64,  # int32 [seqs], pinned host
    index_scales0: cutlass.Int64,  # int32 [pools], pinned host
    kv_offset0: cutlass.Int64,  # int32 [pools], pinned host
    pools0: cutlass.Int32,
    table_seqs0: cutlass.Int32,
    offsets_seqs0: cutlass.Int32,
    blocks0: cutlass.Int32,
    seqs0: cutlass.Int32,
    table1: cutlass.Int64,
    offsets1: cutlass.Int64,
    copy_index1: cutlass.Int64,
    index_scales1: cutlass.Int64,
    kv_offset1: cutlass.Int64,
    pools1: cutlass.Int32,
    table_seqs1: cutlass.Int32,
    offsets_seqs1: cutlass.Int32,
    blocks1: cutlass.Int32,
    seqs1: cutlass.Int32,
):
    """The overlap gathers for threads [0, rows * tokens_per_row) (for row r, slot s = slots[r] and token j:
    ``input_ids[input_begin + r * tokens_per_row + j] = store_next_new_tokens[j, s]``, ``pos_offsets[pos_begin + r *
    tokens_per_row + j] = store_lens[pos_indices[r * tokens_per_row + j]]``, ``draft_tokens[draft_begin + r *
    draft_width + j] = store_next_draft[s, j]`` for j < draft_width, ``kv_offsets[kv_begin + r] = store_lens[s] -
    tokens_per_row``); then up to four staged copies ``dst_i[j] = record[off_i + j]`` for j < n_i; then up to two KV
    block-offset copies, each the
    ``copyBatchBlockOffsetsToDeviceKernel`` of kvCacheManagerV2Utils.cu (the K and V rows of every (pool, sequence)
    from the pinned host table's row ``copy_index[sequence]``: ``index_scale * page``, and ``+ kv_offset`` for V; a
    bad page (-1) gives 0). One thread per int32 (per K / V pair for the block copies)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    t = bidx * cutlass.Int32(THREADS) + tidx
    gathered = rows * tokens_per_row
    if t < gathered:
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
    else:
        u = t - gathered
        dst = cutlass.Int64(0)
        src = cutlass.Int32(0)
        j = cutlass.Int32(0)
        active = cutlass.Int32(0)
        if u < n0:
            dst = dst0
            src = off0 + u
            j = u
            active = cutlass.Int32(1)
        elif u < n0 + n1:
            dst = dst1
            src = off1 + u - n0
            j = u - n0
            active = cutlass.Int32(1)
        elif u < n0 + n1 + n2:
            dst = dst2
            src = off2 + u - n0 - n1
            j = u - n0 - n1
            active = cutlass.Int32(1)
        elif u < n0 + n1 + n2 + n3:
            dst = dst3
            src = off3 + u - n0 - n1 - n2
            j = u - n0 - n1 - n2
            active = cutlass.Int32(1)
        if active == cutlass.Int32(1):
            _i32_at(dst, j)[0] = _i32_at(record, src)[0]
        v = u - n0 - n1 - n2 - n3
        per0 = pools0 * seqs0 * blocks0
        per1 = pools1 * seqs1 * blocks1
        if v >= cutlass.Int32(0):
            if v < per0 + per1:
                table = table0
                offsets = offsets0
                copy_index = copy_index0
                index_scales = index_scales0
                kv_offset = kv_offset0
                table_seqs = table_seqs0
                offsets_seqs = offsets_seqs0
                blocks = blocks0
                seqs = seqs0
                w = v
                if v >= per0:
                    table = table1
                    offsets = offsets1
                    copy_index = copy_index1
                    index_scales = index_scales1
                    kv_offset = kv_offset1
                    table_seqs = table_seqs1
                    offsets_seqs = offsets_seqs1
                    blocks = blocks1
                    seqs = seqs1
                    w = v - per0
                pool = w // (seqs * blocks)
                rest = w - pool * seqs * blocks
                seq = rest // blocks
                block = rest - seq * blocks
                row = pool * table_seqs + _i32_at(copy_index, seq)[0]
                page = _i32_at(table, row * cutlass.Int32(2) * blocks + block)[0]
                key = cutlass.Int32(0)
                value = cutlass.Int32(0)
                if page != cutlass.Int32(-1):
                    key = _i32_at(index_scales, pool)[0] * page
                    value = key + _i32_at(kv_offset, pool)[0]
                out = (pool * offsets_seqs + seq) * cutlass.Int32(2) * blocks + block
                _i32_at(offsets, out)[0] = key
                _i32_at(offsets, out + blocks)[0] = value


@cute.jit
def stage(
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
    record: cutlass.Int64,
    dst0: cutlass.Int64,
    dst1: cutlass.Int64,
    dst2: cutlass.Int64,
    dst3: cutlass.Int64,
    off0: cutlass.Int32,
    off1: cutlass.Int32,
    off2: cutlass.Int32,
    off3: cutlass.Int32,
    n0: cutlass.Int32,
    n1: cutlass.Int32,
    n2: cutlass.Int32,
    n3: cutlass.Int32,
    table0: cutlass.Int64,
    offsets0: cutlass.Int64,
    copy_index0: cutlass.Int64,
    index_scales0: cutlass.Int64,
    kv_offset0: cutlass.Int64,
    pools0: cutlass.Int32,
    table_seqs0: cutlass.Int32,
    offsets_seqs0: cutlass.Int32,
    blocks0: cutlass.Int32,
    seqs0: cutlass.Int32,
    table1: cutlass.Int64,
    offsets1: cutlass.Int64,
    copy_index1: cutlass.Int64,
    index_scales1: cutlass.Int64,
    kv_offset1: cutlass.Int64,
    pools1: cutlass.Int32,
    table_seqs1: cutlass.Int32,
    offsets_seqs1: cutlass.Int32,
    blocks1: cutlass.Int32,
    seqs1: cutlass.Int32,
    stream: cuda_driver.CUstream,
) -> None:
    total = (
        rows * tokens_per_row
        + n0
        + n1
        + n2
        + n3
        + pools0 * seqs0 * blocks0
        + pools1 * seqs1 * blocks1
    )
    stage_kernel(
        store_next_new_tokens, store_next_draft, store_lens, slots, pos_indices, input_ids, draft_tokens, pos_offsets,
        kv_offsets, rows, tokens_per_row, draft_width, num_slots, draft_stride, input_begin, draft_begin, pos_begin,
        kv_begin, record, dst0, dst1, dst2, dst3, off0, off1, off2, off3, n0, n1, n2, n3, table0, offsets0,
        copy_index0, index_scales0, kv_offset0, pools0, table_seqs0, offsets_seqs0, blocks0, seqs0, table1, offsets1,
        copy_index1, index_scales1, kv_offset1, pools1, table_seqs1, offsets_seqs1, blocks1, seqs1,
    ).launch(
        grid=[(total + THREADS - 1) // THREADS, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )  # fmt: skip
