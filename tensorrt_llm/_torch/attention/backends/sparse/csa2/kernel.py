# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""CSA2 CuTe DSL kernels: index projection, packed attention, decode glue and packed caches.

Sections:

* Metadata and indexer glue (SWA/owner slot refresh, indexer descriptors,
  Top-K offset mapping, packed prefix gathering, candidate publication). Every
  kernel replaces a chain of small PyTorch or Inductor launches that ran once
  per layer (or per KV owner) and follows the integer semantics of the PyTorch
  references in ``metadata.py`` (floor division/modulo, ``-1`` padding,
  ``INT_MAX`` sort sentinels).
* Packed-cache kernels: byte-exact quantized publication, gather/dequantization
  and native staging for the ``main`` / ``index`` / ``swa`` row formats of
  ``quantization.py``.
* Index-Q projection and packed sparse attention (native CuTe GEMM/attention).

Kernels take raw pointers plus geometry so one compilation per configuration
serves every shape, including CUDA Graph capture. CuTe types are constructed
lazily, so importing this module does not import optional Cutlass bindings or
initialize CUDA; the module is loaded standalone by an import probe and imports
no package modules at import time.
"""

import dataclasses
import functools
import math
from typing import Any

import torch

from tensorrt_llm.math_utils import ceil_div

# =============================================================================
# Decode metadata and selection glue
# =============================================================================

_INT32_MAX = 2**31 - 1
# Candidate blocks per row that candidate expansion's shared bitmap covers at least;
# wider rows use the next power of two.
_CANDIDATE_BITMAP_BLOCKS = 1 << 16
# Widest packed selection row the finalize bitmap scan sorts (32 KB of shared memory).
_FINALIZE_BITMAP_COLUMNS = 1 << 18
# Widest paged decode row it sorts; wider rows (graph arenas) keep the sort network.
_PAGED_FINALIZE_BITMAP_COLUMNS = 1 << 17
# Candidate columns per row that the selection map's shared bitmap covers: the widest
# candidate publication (4096 sparse blocks of 16 columns).
_MAP_BITMAP_COLUMNS = 1 << 16
# Index page-footer geometry of quantization.py in 32-bit words: 64 rows of
# 16 data words, then one scale word per row.
INDEX_PAGE_ROWS = 64
INDEX_ROW_WORDS = 16
INDEX_PAGE_WORDS = INDEX_PAGE_ROWS * (INDEX_ROW_WORDS + 1)
# Packed prefill piece: request, key width, key offset, first query row, row in group.
PREFIX_DESC_WIDTH = 5


def dsl_available() -> bool:
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    return IS_CUTLASS_DSL_AVAILABLE


@functools.lru_cache(maxsize=1)
def _glue_kernels():
    """Build the decode-glue kernel classes once; returns a namespace of classes and helpers."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.utils import SmemAllocator

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.block_scan import warp_scan
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
        griddepcontrol_launch_dependents,
        griddepcontrol_wait,
        make_ptr,
    )

    Int32, Int64, Float32 = cutlass.Int32, cutlass.Int64, cutlass.Float32

    # ------------------------------------------------------------------ helpers

    @dsl_user_op
    def bitcast_u32_f32(bits, *, loc=None, ip=None):
        return Float32(llvm.bitcast(T.f32(), cutlass.Uint32(bits).ir_value(loc=loc, ip=ip)))

    @cute.jit
    def floor_div(a: Int64, b: Int64) -> Int64:
        """PyTorch ``//`` for signed integers (round toward negative infinity)."""
        q = a // b
        if ((a % b) != 0) & ((a < 0) != (b < 0)):
            q = q - 1
        return q

    @cute.jit
    def floor_mod(a: Int64, b: Int64) -> Int64:
        """PyTorch ``%`` for signed integers (result takes the divisor's sign)."""
        r = a % b
        if (r != 0) & ((r < 0) != (b < 0)):
            r = r + b
        return r

    @cute.jit
    def tensor1d(ptr: cute.Pointer, count: Int64):
        return cute.make_tensor(ptr, cute.make_layout(count))

    @cute.jit
    def tensor2d(ptr: cute.Pointer, rows: Int64, cols: Int64, row_stride: Int64):
        return cute.make_tensor(ptr, cute.make_layout((rows, cols), stride=(row_stride, 1)))

    @cute.jit
    def _int32_ptr(address: Int64):
        return cute.make_ptr(Int32, Int64(address), cute.AddressSpace.gmem, assumed_align=4)

    # ------------------------------------------------------- K1: SWA refresh

    class SwaRefreshKernel:
        """Per-step SWA read/write slots and visibility for every model layer.

        One CTA per (query tile, layer); one thread per window column. The
        first ``swa_layers`` layers address their page tables through a
        ``[swa_layers, 4]`` int64 descriptor table of ``(pointer, row stride,
        rows, columns)``, so layers may keep independently bound tables; later
        layers own no SWA cache and only get visibility. Window and page size
        are compile-time constants, so the slot arithmetic avoids 64-bit division.
        """

        tile = 8

        def __init__(self, window: int, block: int):
            if window <= 0 or window > 1024:
                raise ValueError("CSA2 SWA window must be within 1..1024")
            self.window = window
            self.block = block
            self.threads = max(32, min(1024, -(-window // 32) * 32))

        @cute.jit
        def __call__(
            self,
            positions: cute.Pointer,  # int32 [tokens]
            requests: cute.Pointer,  # int64 [tokens]
            floors: cute.Pointer,  # int64 [floor_count]: write (and read) floors
            read_floors: cute.Pointer,  # int64 [floor_count]: read-only floors
            tables: cute.Pointer,  # int64 [swa_layers, 4]: ptr, stride, rows, cols
            ratios: cute.Pointer,  # int32 [layers]
            reads: cute.Pointer,  # int64 [swa_layers, tokens, window]
            writes: cute.Pointer,  # int64 [swa_layers, tokens]
            visible: cute.Pointer,  # int64 [layers, tokens]
            tokens: Int32,
            swa_layers: Int32,
            layers: Int32,
            floor_count: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                positions,
                requests,
                floors,
                read_floors,
                tables,
                ratios,
                reads,
                writes,
                visible,
                tokens,
                swa_layers,
                layers,
                floor_count,
            ).launch(
                grid=[(tokens + self.tile - 1) // self.tile, layers, 1],
                block=[self.threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            positions: cute.Pointer,
            requests: cute.Pointer,
            floors: cute.Pointer,
            read_floors: cute.Pointer,
            tables: cute.Pointer,
            ratios: cute.Pointer,
            reads: cute.Pointer,
            writes: cute.Pointer,
            visible: cute.Pointer,
            tokens: Int32,
            swa_layers: Int32,
            layers: Int32,
            floor_count: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            tile_idx, layer, _ = cute.arch.block_idx()
            window, block = self.window, self.block
            first = tile_idx * self.tile
            positions_t = tensor1d(positions, Int64(tokens))
            if tid < self.tile:
                token = first + tid
                if token < tokens:
                    ratio = Int64(tensor1d(ratios, Int64(layers))[layer])
                    visible_value = Int64(0)
                    if ratio > 0:
                        # Padded rows may carry negative positions: floor, as torch.
                        visible_value = floor_div(Int64(positions_t[token]) + 1, ratio)
                    out_index = Int64(layer) * Int64(tokens) + Int64(token)
                    tensor1d(visible, Int64(layers) * Int64(tokens))[out_index] = visible_value
            if layer < swa_layers:
                table_t = tensor2d(tables, Int64(swa_layers), Int64(4), Int64(4))
                table_rows = Int32(table_t[layer, 2])
                table_cols = Int32(table_t[layer, 3])
                pages = cute.make_tensor(
                    cute.make_ptr(
                        Int32, Int64(table_t[layer, 0]), cute.AddressSpace.gmem, assumed_align=4
                    ),
                    cute.make_layout(
                        (Int64(table_rows), Int64(table_cols)), stride=(Int64(table_t[layer, 1]), 1)
                    ),
                )
                request_count = table_rows
                if floor_count < request_count:
                    request_count = floor_count
                floor_t = tensor1d(floors, Int64(floor_count) + 1)
                read_floor_t = tensor1d(read_floors, Int64(floor_count) + 1)
                requests_t = tensor1d(requests, Int64(tokens))
                reads_t = tensor1d(reads, Int64(swa_layers) * Int64(tokens) * Int64(window))
                writes_t = tensor1d(writes, Int64(swa_layers) * Int64(tokens))
                for offset in cutlass.range(self.tile):
                    token = first + offset
                    if token < tokens:
                        request = Int32(requests_t[token])
                        position = Int32(positions_t[token])
                        safe_request = request
                        if safe_request < 0:
                            safe_request = Int32(0)
                        if safe_request > request_count - 1:
                            safe_request = request_count - 1
                        valid_request = (request >= 0) & (request < request_count)
                        floor = Int32(0)
                        read_floor = Int32(0)
                        if request_count > 0:
                            floor = Int32(floor_t[safe_request])
                            read_floor = Int32(read_floor_t[safe_request])
                        row = Int64(layer) * Int64(tokens) + Int64(token)
                        for column in cutlass.range(tid, window, self.threads):
                            logical = position - (window - 1) + column
                            slot = Int64(-1)
                            if (
                                valid_request
                                & (table_cols > 0)
                                & (logical >= floor)
                                & (logical >= 0)
                            ):
                                page_column = logical // block
                                if page_column < table_cols:
                                    physical = Int32(pages[safe_request, page_column])
                                    if physical >= 0:
                                        slot = Int64(physical) * block + (logical % block)
                            # Decoder floors hide older SWA from reads only.
                            read = slot
                            if logical < read_floor:
                                read = Int64(-1)
                            reads_t[row * window + Int64(column)] = read
                            if column == window - 1:
                                writes_t[row] = slot

    # ------------------------------------------------ K1b: page-table convert

    class PageTableConvertKernel:
        """Converted page indices of many (layer, role) tables from base page rows.

        One CTA per (sequence, table); threads stride over the columns. With
        ``with_scratch`` the SWA scratch blocks ``[begin, end)`` of each (pool,
        sequence) map to its scratch slots, as ``PageIndexConverter`` does. With
        ``with_low`` the first ``low_tables`` tables go to ``low_out`` (its own
        width and strides) instead of ``out``.
        """

        threads = 128

        def __init__(self, with_scratch: bool, with_low: bool):
            self.with_scratch = with_scratch
            self.with_low = with_low

        @cute.jit
        def __call__(
            self,
            base: cute.Pointer,  # int32 [pools, rows, blocks]
            rows: cute.Pointer,  # int32 [seqs]: base row of each sequence
            params: cute.Pointer,  # int32 [tables, 5]: pool, scale, offset, scratch pool, pages
            scratch_ranges: cute.Pointer,  # int32 [scratch pools, seqs, 2]: begin, end
            scratch_slots: cute.Pointer,  # int32 [scratch pools, seqs, slot_stride]
            out: cute.Pointer,  # int32 [tables, seqs, width], unit column stride
            seqs: Int32,
            tables: Int32,
            width: Int32,
            slot_stride: Int32,
            pool_stride: Int64,
            row_stride: Int64,
            out_table_stride: Int64,
            out_row_stride: Int64,
            bad: Int32,
            low_out: cute.Pointer,  # int32 [low tables, seqs, low width], unit column stride
            low_tables: Int32,
            low_width: Int32,
            low_table_stride: Int64,
            low_row_stride: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                base,
                rows,
                params,
                scratch_ranges,
                scratch_slots,
                out,
                seqs,
                tables,
                width,
                slot_stride,
                pool_stride,
                row_stride,
                out_table_stride,
                out_row_stride,
                bad,
                low_out,
                low_tables,
                low_width,
                low_table_stride,
                low_row_stride,
            ).launch(grid=[seqs, tables, 1], block=[self.threads, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            base: cute.Pointer,
            rows: cute.Pointer,
            params: cute.Pointer,
            scratch_ranges: cute.Pointer,
            scratch_slots: cute.Pointer,
            out: cute.Pointer,
            seqs: Int32,
            tables: Int32,
            width: Int32,
            slot_stride: Int32,
            pool_stride: Int64,
            row_stride: Int64,
            out_table_stride: Int64,
            out_row_stride: Int64,
            bad: Int32,
            low_out: cute.Pointer,
            low_tables: Int32,
            low_width: Int32,
            low_table_stride: Int64,
            low_row_stride: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            seq, table, _ = cute.arch.block_idx()
            param_t = tensor2d(params, Int64(tables), Int64(5), Int64(5))
            pool = Int64(param_t[table, 0])
            scale = Int32(param_t[table, 1])
            offset = Int32(param_t[table, 2])
            row = Int64(tensor1d(rows, Int64(seqs))[seq])
            src_base = pool * pool_stride + row * row_stride
            dst_base = Int64(table) * out_table_stride + Int64(seq) * out_row_stride
            columns = width
            if cutlass.const_expr(self.with_low):
                # Address the table's row directly in whichever output holds it.
                address = out.toint() + dst_base * 4
                if table < low_tables:
                    address = (
                        low_out.toint()
                        + (Int64(table) * low_table_stride + Int64(seq) * low_row_stride) * 4
                    )
                    columns = low_width
                dst_base = Int64(0)
                dst = tensor1d(_int32_ptr(address), Int64(columns))
            else:
                dst = tensor1d(out, dst_base + Int64(width))
            src = tensor1d(base, src_base + Int64(columns))
            if cutlass.const_expr(self.with_scratch):
                scratch_row = Int64(param_t[table, 3]) * Int64(seqs) + Int64(seq)
                scratch_pages = Int32(param_t[table, 4])
                ranges = tensor1d(scratch_ranges, (scratch_row + 1) * 2)
                begin = Int32(ranges[scratch_row * 2])
                end = Int32(ranges[scratch_row * 2 + 1])
                slots = tensor1d(scratch_slots, (scratch_row + 1) * Int64(slot_stride))
            for column in cutlass.range(tid, columns, self.threads):
                page = Int32(src[src_base + Int64(column)])
                value = bad
                if page != bad:
                    value = page * scale + offset
                if cutlass.const_expr(self.with_scratch):
                    if (page == bad) & (column >= begin) & (column < end):
                        total = (column - begin) * scratch_pages
                        slot = Int32(
                            slots[scratch_row * Int64(slot_stride) + Int64(total // scale)]
                        )
                        value = slot * scale + (total % scale + offset) % scale
                dst[dst_base + Int64(column)] = value

    class TableCopyKernel:
        """Copy many int32 ``[rows, columns]`` tables in one launch: one CTA per (row,
        table) of an int64 ``[tables, 6]`` descriptor table of (source, destination,
        rows, columns, source row stride, destination row stride)."""

        threads = 128

        @cute.jit
        def __call__(
            self,
            descriptors: cute.Pointer,  # int64 [tables, 6]
            tables: Int32,
            max_rows: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(descriptors, tables).launch(
                grid=[max_rows, tables, 1], block=[self.threads, 1, 1], stream=stream
            )

        @cute.kernel
        def kernel(self, descriptors: cute.Pointer, tables: Int32):
            tid, _, _ = cute.arch.thread_idx()
            row, table, _ = cute.arch.block_idx()
            desc = tensor2d(descriptors, Int64(tables), Int64(6), Int64(6))
            if Int64(row) < Int64(desc[table, 2]):
                columns = Int32(desc[table, 3])
                src = tensor1d(_int32_ptr(desc[table, 0]), Int64(2**62))
                dst = tensor1d(_int32_ptr(desc[table, 1]), Int64(2**62))
                src_base = Int64(row) * Int64(desc[table, 4])
                dst_base = Int64(row) * Int64(desc[table, 5])
                for column in cutlass.range(tid, columns, self.threads):
                    dst[dst_base + Int64(column)] = src[src_base + Int64(column)]

    # ------------------------------------------------------ K2: owner slots

    class OwnerSlotsKernel:
        """GLOBAL write slots and compressed groups of every KV owner for live endpoints.

        Mirrors ``CSA2TrtllmMetadata._refresh_owner_slots`` in one launch: grid
        ``y`` selects the owner, whose buffers and geometry come from an int64
        ``[owners, OWNER_DESC_WIDTH]`` descriptor table (see
        ``owner_slot_descriptor``). Per-request group counts and their prefix
        sums are recomputed by every CTA in shared memory (requests are few),
        then each thread resolves one output row. With ``with_positions`` the
        first CTA also refreshes the query positions, as
        ``CSA2TrtllmMetadata._refresh_query_positions`` does.
        """

        MAX_REQUESTS = 4096
        THREADS = 256

        def __init__(self, with_positions: bool = False):
            self.with_positions = with_positions

        @cute.jit
        def __call__(
            self,
            kv_lens: cute.Pointer,  # int32
            lengths: cute.Pointer,  # int32 [requests]
            query_base: cute.Pointer,  # int32 [requests]
            descriptors: cute.Pointer,  # int64 [owners, OWNER_DESC_WIDTH]
            positions: cute.Pointer,  # int32 [tokens] (out)
            token_requests: cute.Pointer,  # int64 [tokens]
            query_offsets: cute.Pointer,  # int32 [tokens]
            requests: Int32,
            owners: Int32,
            max_capacity: Int32,
            tokens: Int32,
            stream: cuda.CUstream,
        ):
            blocks = (max_capacity + self.THREADS - 1) // self.THREADS
            if blocks < 1:
                blocks = Int32(1)
            self.kernel(
                kv_lens,
                lengths,
                query_base,
                descriptors,
                positions,
                token_requests,
                query_offsets,
                requests,
                owners,
                tokens,
            ).launch(grid=[blocks, owners, 1], block=[self.THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            kv_lens: cute.Pointer,
            lengths: cute.Pointer,
            query_base: cute.Pointer,
            descriptors: cute.Pointer,
            positions: cute.Pointer,
            token_requests: cute.Pointer,
            query_offsets: cute.Pointer,
            requests: Int32,
            owners: Int32,
            tokens: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            bid, owner, _ = cute.arch.block_idx()
            if cutlass.const_expr(self.with_positions):
                if (bid == 0) & (owner == 0):
                    kv_t = tensor1d(kv_lens, Int64(requests) + 1)
                    len_t = tensor1d(lengths, Int64(requests) + 1)
                    for t in cutlass.range(tid, tokens, self.THREADS):
                        r = Int32(tensor1d(token_requests, Int64(tokens))[t])
                        tensor1d(positions, Int64(tokens))[t] = (
                            Int32(kv_t[r])
                            - Int32(len_t[r])
                            + Int32(tensor1d(query_offsets, Int64(tokens))[t])
                        )
            smem = SmemAllocator()
            starts_s = smem.allocate_tensor(Int32, cute.make_layout(self.MAX_REQUESTS), 16)
            cumulative_s = smem.allocate_tensor(Int32, cute.make_layout(self.MAX_REQUESTS), 16)
            desc = tensor2d(
                descriptors, Int64(owners), Int64(OWNER_DESC_WIDTH), Int64(OWNER_DESC_WIDTH)
            )
            source_base = _int32_ptr(desc[owner, 0])
            source_lengths = _int32_ptr(desc[owner, 1])
            pages = _int32_ptr(desc[owner, 2])
            columns = Int32(desc[owner, 3])
            page_stride = Int64(desc[owner, 4])
            slots = cute.make_ptr(
                Int64, Int64(desc[owner, 5]), cute.AddressSpace.gmem, assumed_align=8
            )
            compressed = _int32_ptr(desc[owner, 6])
            capacity = Int32(desc[owner, 7])
            ratio = Int32(desc[owner, 8])
            page_size = Int32(desc[owner, 9])
            batch_starts = _int32_ptr(desc[owner, 10])
            batch_lengths = _int32_ptr(desc[owner, 11])
            batch_cu = _int32_ptr(desc[owner, 12])
            with_batch = Int64(desc[owner, 12]) != 0
            n = Int64(requests)
            kv_t = tensor1d(kv_lens, n + 1)
            len_t = tensor1d(lengths, n + 1)
            base_t = tensor1d(query_base, n + 1)
            sbase_t = tensor1d(source_base, n + 1)
            slen_t = tensor1d(source_lengths, n + 1)
            for r in cutlass.range(tid, requests, self.THREADS):
                delta = Int32(kv_t[r]) - Int32(len_t[r]) - Int32(base_t[r])
                start = Int32(sbase_t[r]) + delta
                end = start + Int32(slen_t[r])
                starts_s[r] = start
                count = Int32(
                    floor_div(Int64(end), Int64(ratio)) - floor_div(Int64(start), Int64(ratio))
                )
                cumulative_s[r] = count
                if with_batch & (bid == 0):
                    tensor1d(batch_starts, n + 1)[r] = start
                    tensor1d(batch_lengths, n + 1)[r] = end
            cute.arch.barrier()
            if tid == 0:
                running = Int32(0)
                for r in cutlass.range(0, requests, 1):
                    running = running + cumulative_s[r]
                    cumulative_s[r] = running
            cute.arch.barrier()
            if with_batch & (bid == 0):
                cu_t = tensor1d(batch_cu, n + 2)
                if tid == 0:
                    cu_t[0] = Int32(0)
                for r in cutlass.range(tid, requests, self.THREADS):
                    cu_t[r + 1] = cumulative_s[r]
            offset = bid * self.THREADS + tid
            if (offset < capacity) & (requests > 0):
                # searchsorted(cumulative, offset, right=True): count of cumulative <= offset
                lo = Int32(0)
                hi = requests
                for _ in cutlass.range(0, 32, 1):
                    if lo < hi:
                        mid = (lo + hi) // 2
                        if cumulative_s[mid] <= offset:
                            lo = mid + 1
                        else:
                            hi = mid
                r = lo
                valid = r < requests
                if r > requests - 1:
                    r = requests - 1
                preceding = Int32(0)
                if r > 0:
                    preceding = cumulative_s[r - 1]
                group = (
                    floor_div(Int64(starts_s[r]), Int64(ratio)) + Int64(offset) - Int64(preceding)
                )
                column = group
                if column < 0:
                    column = Int64(0)
                column = column // Int64(page_size)
                safe_column = column
                if safe_column > Int64(columns) - 1:
                    safe_column = Int64(columns) - 1
                physical = Int64(-1)
                if columns > 0:
                    physical = Int64(
                        tensor2d(pages, n, Int64(columns), page_stride)[r, Int32(safe_column)]
                    )
                valid = valid & (group >= 0) & (column < Int64(columns)) & (physical >= 0)
                slot = Int64(-1)
                position = Int32(0)
                if valid:
                    slot = physical * Int64(page_size) + floor_mod(group, Int64(page_size))
                    position = Int32(group * Int64(ratio))
                tensor1d(slots, Int64(capacity))[offset] = slot
                tensor1d(compressed, Int64(capacity))[offset] = position
            elif offset < capacity:
                tensor1d(slots, Int64(capacity))[offset] = Int64(-1)
                tensor1d(compressed, Int64(capacity))[offset] = Int32(0)

    # ------------------------------------------------ K3: indexer descriptors

    class IndexerDescriptorsKernel:
        """Native paged-indexer block tables and page validity for decode queries.

        Each thread writes ``PAGES`` native pages of a query, one per ``THREADS``
        stride: its block-table entry and one validity byte that is set when the
        page is allocated.
        """

        THREADS = 128
        PAGES = 4

        def __init__(self, pages_per_source_page: int):
            self.pages_per_source_page = pages_per_source_page

        @cute.jit
        def __call__(
            self,
            table: cute.Pointer,  # int32 [requests, source_pages] strided
            token_requests: cute.Pointer,  # int64 [count]
            decode_visible: cute.Pointer,  # int64 [count]
            blocks: cute.Pointer,  # int32 [count, page_capacity] strided (out)
            context: cute.Pointer,  # int32 [count, 1] (out)
            visible: cute.Pointer,  # int32 [count] (out)
            valid: cute.Pointer,  # uint8 [count, page_capacity] strided (out)
            count: Int32,
            request_count: Int32,
            source_pages: Int32,
            table_stride: Int64,
            context_requests: Int32,
            page_capacity: Int32,
            blocks_stride: Int64,
            valid_stride: Int64,
            chunks: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                table,
                token_requests,
                decode_visible,
                blocks,
                context,
                visible,
                valid,
                count,
                request_count,
                source_pages,
                table_stride,
                context_requests,
                page_capacity,
                blocks_stride,
                valid_stride,
            ).launch(
                grid=[count, chunks, 1],
                block=[self.THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            table: cute.Pointer,
            token_requests: cute.Pointer,
            decode_visible: cute.Pointer,
            blocks: cute.Pointer,
            context: cute.Pointer,
            visible: cute.Pointer,
            valid: cute.Pointer,
            count: Int32,
            request_count: Int32,
            source_pages: Int32,
            table_stride: Int64,
            context_requests: Int32,
            page_capacity: Int32,
            blocks_stride: Int64,
            valid_stride: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, chunk, _ = cute.arch.block_idx()
            n = Int64(count)
            request = Int32(tensor1d(token_requests, n)[q]) - context_requests
            admitted = (request >= 0) & (request < request_count)
            safe_request = request
            if safe_request < 0:
                safe_request = Int32(0)
            if safe_request > request_count - 1:
                safe_request = request_count - 1
            end = Int32(0)
            if admitted:
                end = Int32(tensor1d(decode_visible, n)[q])
            table_t = tensor2d(table, Int64(request_count), Int64(source_pages), table_stride)
            pages_per_source_page = self.pages_per_source_page
            for item in cutlass.range_constexpr(self.PAGES):
                index = (chunk * self.PAGES + item) * self.THREADS + tid
                if index < page_capacity:
                    source = index // pages_per_source_page
                    native = Int32(0)
                    backed = cutlass.Uint8(0)
                    if admitted & (source < source_pages) & (request_count > 0):
                        physical = Int32(table_t[safe_request, source])
                        if physical >= 0:
                            native = (
                                physical * pages_per_source_page + index % pages_per_source_page
                            )
                            backed = cutlass.Uint8(1)
                    tensor2d(blocks, n, Int64(page_capacity), blocks_stride)[q, index] = native
                    tensor2d(valid, n, Int64(page_capacity), valid_stride)[q, index] = backed
            if (chunk == 0) & (tid == 0):
                tensor1d(visible, n)[q] = end
                clamped = end
                if clamped < 1:
                    clamped = Int32(1)
                tensor1d(context, n)[q] = clamped

    # ------------------------------------- K4: map Top-K candidate offsets

    class MaskLogitsKernel:
        """Mask unallocated pages among each row's leading ``row_lengths`` columns of
        ``[count, width]`` logits, one page per thread under its validity byte.
        """

        THREADS = 256
        COLUMNS = INDEX_PAGE_ROWS

        @cute.jit
        def __call__(
            self,
            logits: cute.Pointer,  # float32 [count, width] strided
            valid: cute.Pointer,  # uint8 [count, >= pages] strided, one byte per page
            row_lengths: cute.Pointer,  # int32 [count] visible prefix lengths
            count: Int32,
            width: Int32,
            logits_stride: Int64,
            valid_stride: Int64,
            stream: cuda.CUstream,
        ):
            span = self.THREADS * self.COLUMNS
            self.kernel(
                logits, valid, row_lengths, count, width, logits_stride, valid_stride
            ).launch(
                grid=[count, (width + span - 1) // span, 1],
                block=[self.THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            logits: cute.Pointer,
            valid: cute.Pointer,
            row_lengths: cute.Pointer,
            count: Int32,
            width: Int32,
            logits_stride: Int64,
            valid_stride: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, chunk, _ = cute.arch.block_idx()
            logits_t = tensor2d(logits, Int64(count), Int64(width), logits_stride)
            length = Int32(tensor1d(row_lengths, Int64(count))[q])
            if length > width:
                length = width
            page = chunk * self.THREADS + tid
            first = page * self.COLUMNS
            if first < length:
                byte = tensor1d(valid, Int64(2**62))[Int64(q) * valid_stride + Int64(page)]
                if cutlass.Uint8(byte) == 0:
                    for column in cutlass.range(first, first + self.COLUMNS, 1):
                        if column < length:
                            logits_t[q, column] = Float32(float("-inf"))

    class MaskPackedLogitsKernel:
        """Mask the in-row columns of unbacked gathered keys in packed prefix logits.

        Row ``q`` scores gathered keys ``row_starts[q] + column`` for ``column <
        row_lengths[q]``, and every consumer bounds its reads by ``row_lengths``.
        A thread owns one gathered key within a block of ``ROWS`` rows: a backed
        key costs one byte load, and only an unbacked key scans those rows to mask
        its column wherever it is visible.
        """

        THREADS = 256
        ROWS = 1024

        @cute.jit
        def __call__(
            self,
            logits: cute.Pointer,  # float32 [count, width] strided
            valid: cute.Pointer,  # uint8 [keys]
            row_starts: cute.Pointer,  # int32 [count]
            row_lengths: cute.Pointer,  # int32 [count]
            count: Int32,
            width: Int32,
            keys: Int32,
            logits_stride: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                logits, valid, row_starts, row_lengths, count, width, keys, logits_stride
            ).launch(
                grid=[
                    (keys + self.THREADS - 1) // self.THREADS,
                    (count + self.ROWS - 1) // self.ROWS,
                    1,
                ],
                block=[self.THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            logits: cute.Pointer,
            valid: cute.Pointer,
            row_starts: cute.Pointer,
            row_lengths: cute.Pointer,
            count: Int32,
            width: Int32,
            keys: Int32,
            logits_stride: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, rows, _ = cute.arch.block_idx()
            key = Int64(block) * self.THREADS + Int64(tid)
            if key < Int64(keys):
                if cutlass.Uint8(tensor1d(valid, Int64(keys))[key]) == 0:
                    n = Int64(count)
                    logits_t = tensor2d(logits, n, Int64(width), logits_stride)
                    last = rows * self.ROWS + self.ROWS
                    if last > count:
                        last = count
                    for q in cutlass.range(rows * self.ROWS, last, 1):
                        column = key - Int64(tensor1d(row_starts, n)[q])
                        length = Int64(tensor1d(row_lengths, n)[q])
                        if (column >= 0) & (column < length) & (column < Int64(width)):
                            logits_t[q, column] = Float32(float("-inf"))

    # ------------------------------------------ K5: candidate-ordered scores

    class CandidateGatherKernel:
        """Gather candidate columns from paged logits with validity and visibility masks."""

        THREADS = 256

        @cute.jit
        def __call__(
            self,
            logits: cute.Pointer,  # float32 [count, width] strided
            valid: cute.Pointer,  # uint8 [count, >= pages] strided, one byte per page
            candidates: cute.Pointer,  # int64/int32 [count, cwidth] strided
            visible: cute.Pointer,  # int32 [count]
            scores: cute.Pointer,  # float32 [count, cwidth] (out)
            positions: cute.Pointer,  # int32 [count, cwidth] (out)
            count: Int32,
            width: Int32,
            cwidth: Int32,
            logits_stride: Int64,
            valid_stride: Int64,
            candidate_stride: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                logits,
                valid,
                candidates,
                visible,
                scores,
                positions,
                count,
                width,
                cwidth,
                logits_stride,
                valid_stride,
                candidate_stride,
            ).launch(
                grid=[count, (cwidth + self.THREADS - 1) // self.THREADS, 1],
                block=[self.THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            logits: cute.Pointer,
            valid: cute.Pointer,
            candidates: cute.Pointer,
            visible: cute.Pointer,
            scores: cute.Pointer,
            positions: cute.Pointer,
            count: Int32,
            width: Int32,
            cwidth: Int32,
            logits_stride: Int64,
            valid_stride: Int64,
            candidate_stride: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, chunk, _ = cute.arch.block_idx()
            column = chunk * self.THREADS + tid
            if column < cwidth:
                n = Int64(count)
                candidate_t = tensor2d(candidates, n, Int64(cwidth), candidate_stride)
                position = Int64(candidate_t[q, column])
                ok = (position >= 0) & (position < Int64(width))
                ok = ok & (position < Int64(tensor1d(visible, n)[q]))
                safe = position
                if safe < 0:
                    safe = Int64(0)
                if safe > Int64(width) - 1:
                    safe = Int64(width) - 1
                score = Float32(float("-inf"))
                out_position = Int32(-1)
                if ok & (width > 0):
                    valid_t = tensor1d(valid, Int64(2**62))
                    page = Int64(q) * valid_stride + safe // INDEX_PAGE_ROWS
                    if cutlass.Uint8(valid_t[page]) != 0:
                        score = Float32(
                            tensor2d(logits, n, Int64(width), logits_stride)[q, Int32(safe)]
                        )
                        out_position = Int32(position)
                tensor2d(scores, n, Int64(cwidth), Int64(cwidth))[q, column] = score
                tensor2d(positions, n, Int64(cwidth), Int64(cwidth))[q, column] = out_position

    # ------------------------------------------- K6: finalize Top-K rows

    class FinalizeSelectionKernel:
        """Validate, map and sort one Top-K row in place; padding becomes ``-1``.

        Mode ``paged``: offsets are positions, valid below ``lengths[q]`` when their
        page's byte ``valid[q, off // INDEX_PAGE_ROWS]`` is set.
        Mode ``mapped``: offsets index ``positions``; valid when the gathered
        score is finite (``> -inf``). Mode ``packed``: offsets are positions,
        valid below ``lengths[q]`` when packed key ``valid[starts[q] + off]`` is.
        A bitonic sort over ``INT32_MAX`` sentinels reproduces ``torch.sort`` of
        the int64 sentinel reference. ``bitmap`` (packed and paged rows) instead flags
        the distinct valid offsets in a bitmap of the row's columns and scans it.
        """

        def __init__(self, top_k: int, mapped: bool, packed: bool = False, bitmap: bool = False):
            if top_k <= 0 or top_k > 4096:
                raise ValueError("CSA2 selection finalize supports 1..4096 selections")
            self.top_k = top_k
            self.padded = 1 << (top_k - 1).bit_length()
            self.threads = max(32, min(1024, self.padded // 2 if self.padded >= 64 else 32))
            self.mapped = mapped
            self.packed = packed
            self.bitmap = bitmap

        @cute.jit
        def __call__(
            self,
            indices: cute.Pointer,  # int32 [count, top_k] strided (in/out)
            valid: cute.Pointer,  # uint8 [count, pages] strided (paged) or [keys] (packed)
            scores: cute.Pointer,  # float32 [count, width] strided (mapped mode)
            positions: cute.Pointer,  # int32/int64 [count, width] strided (mapped mode)
            starts: cute.Pointer,  # int32 [count] (packed mode)
            lengths: cute.Pointer,  # int32 [count] (paged and packed modes)
            count: Int32,
            width: Int32,
            indices_stride: Int64,
            valid_stride: Int64,
            scores_stride: Int64,
            positions_stride: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                indices,
                valid,
                scores,
                positions,
                starts,
                lengths,
                count,
                width,
                indices_stride,
                valid_stride,
                scores_stride,
                positions_stride,
            ).launch(grid=[count, 1, 1], block=[self.threads, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            indices: cute.Pointer,
            valid: cute.Pointer,
            scores: cute.Pointer,
            positions: cute.Pointer,
            starts: cute.Pointer,
            lengths: cute.Pointer,
            count: Int32,
            width: Int32,
            indices_stride: Int64,
            valid_stride: Int64,
            scores_stride: Int64,
            positions_stride: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, _, _ = cute.arch.block_idx()
            smem = SmemAllocator()
            n = Int64(count)
            indices_t = tensor2d(indices, n, Int64(self.top_k), indices_stride)
            if cutlass.const_expr(self.packed):
                valid_t = tensor1d(valid, Int64(2**62))
                row_start = Int64(tensor1d(starts, n)[q])
                row_length = Int32(tensor1d(lengths, n)[q])
            if cutlass.const_expr(self.bitmap):
                if cutlass.const_expr(not self.packed):
                    valid_t = tensor1d(valid, Int64(2**62))
                    row_length = Int32(tensor1d(lengths, n)[q])
                flags = smem.allocate_tensor(
                    cutlass.Uint32, cute.make_layout(_FINALIZE_BITMAP_COLUMNS // 32), 16
                )
                warp_totals = smem.allocate_tensor(Int32, cute.make_layout(self.threads // 32), 16)
                # Thread t owns words [t * chunk, (t + 1) * chunk), word w kept at
                # (w % chunk) * threads + w // chunk so a warp reads distinct banks.
                # Only offsets below the row's length are flagged, so only their words are scanned.
                columns = width
                if row_length < width:
                    columns = row_length
                chunk = (columns + 32 * self.threads - 1) // (32 * self.threads)
                for slot in cutlass.range(tid, chunk * self.threads, self.threads):
                    flags[slot] = cutlass.Uint32(0)
                cute.arch.barrier()
                for k in cutlass.range(tid, self.top_k, self.threads):
                    offset = Int32(indices_t[q, k])
                    if (offset >= 0) & (offset < row_length) & (offset < width):
                        if cutlass.const_expr(self.packed):
                            byte = row_start + Int64(offset)
                        else:
                            byte = Int64(q) * valid_stride + Int64(offset // INDEX_PAGE_ROWS)
                        if cutlass.Uint8(valid_t[byte]) != 0:
                            word = offset // 32
                            cute.arch.atomic_or(
                                ptr=(
                                    flags.iterator + (word % chunk) * self.threads + word // chunk
                                ).llvm_ptr,
                                val=cutlass.Uint32(1) << cutlass.Uint32(offset % 32),
                                sem="relaxed",
                                scope="cta",
                            )
                cute.arch.barrier()
                local = Int32(0)
                for i in cutlass.range(chunk):
                    local = local + Int32(cute.arch.popc(flags[i * self.threads + tid]))
                # Exclusive block scan of the threads' flag counts ranks their words.
                lane = tid % 32
                inclusive = local
                for step in cutlass.range_constexpr(5):
                    other = cute.arch.shuffle_sync_up(
                        inclusive, 1 << step, mask=-1, mask_and_clamp=0
                    )
                    if lane >= (1 << step):
                        inclusive = inclusive + other
                if lane == 31:
                    warp_totals[tid // 32] = inclusive
                cute.arch.barrier()
                rank = inclusive - local
                total = Int32(0)
                for warp in cutlass.range_constexpr(self.threads // 32):
                    if warp < tid // 32:
                        rank = rank + warp_totals[warp]
                    total = total + warp_totals[warp]
                for i in cutlass.range(chunk):
                    bits = flags[i * self.threads + tid]
                    for _ in cutlass.range(Int32(cute.arch.popc(bits))):
                        low = bits & (cutlass.Uint32(0) - bits)
                        bit = Int32(cute.arch.popc(low - cutlass.Uint32(1)))
                        indices_t[q, rank] = (tid * chunk + i) * 32 + bit
                        rank = rank + 1
                        bits = bits ^ low
                for k in cutlass.range(total + tid, self.top_k, self.threads):
                    indices_t[q, k] = Int32(-1)
                return
            keys = smem.allocate_tensor(Int32, cute.make_layout(self.padded), 16)
            for k in cutlass.range(tid, self.padded, self.threads):
                key = Int32(_INT32_MAX)
                if k < self.top_k:
                    offset = Int32(indices_t[q, k])
                    if (offset >= 0) & (offset < width):
                        if cutlass.const_expr(self.mapped):
                            score = Float32(
                                tensor2d(scores, n, Int64(width), scores_stride)[q, offset]
                            )
                            if score > Float32(float("-inf")):
                                key = Int32(
                                    tensor2d(positions, n, Int64(width), positions_stride)[
                                        q, offset
                                    ]
                                )
                        elif cutlass.const_expr(self.packed):
                            if offset < row_length:
                                if cutlass.Uint8(valid_t[row_start + Int64(offset)]) != 0:
                                    key = offset
                        else:
                            if offset < Int32(tensor1d(lengths, n)[q]):
                                page = Int64(q) * valid_stride + Int64(offset // INDEX_PAGE_ROWS)
                                if cutlass.Uint8(tensor1d(valid, Int64(2**62))[page]) != 0:
                                    key = offset
                keys[k] = key
            cute.arch.barrier()
            # Bitonic sort, ascending.
            size = 2
            while size <= self.padded:
                stride = size // 2
                while stride >= 1:
                    for pair in cutlass.range(tid, self.padded // 2, self.threads):
                        i = (pair // stride) * (2 * stride) + (pair % stride)
                        partner = i + stride
                        ascending = ((i // size) % 2) == 0
                        a = keys[i]
                        b = keys[partner]
                        if (a > b) == ascending:
                            keys[i] = b
                            keys[partner] = a
                    cute.arch.barrier()
                    stride = stride // 2
                size = size * 2
            for k in cutlass.range(tid, self.top_k, self.threads):
                key = keys[k]
                if key == Int32(_INT32_MAX):
                    key = Int32(-1)
                indices_t[q, k] = key

    class PackedSortFinalizeKernel:
        """``FinalizeSelectionKernel``'s packed bitmap mode (rows up to ``_FINALIZE_BITMAP_COLUMNS``
        wide, distinct offsets) for ``top_k`` in 128..1024 (a power of two), one warp per row and no
        bitmap, so its cost does not grow with the row width.

        Lane ``j`` loads offsets ``32 i + j``; an offset is kept when it lies below the row's length
        and the width and its packed key ``valid[starts[q] + off]`` is set, others become
        ``INT32_MAX``. A bitonic sort over element ``per * lane + r`` orders the row: steps within a
        lane swap registers, steps across lanes shuffle. A warp scan ranks the distinct kept offsets,
        which are staged in shared memory and written back ascending with ``-1`` padding.
        """

        WARPS = 4

        def __init__(self, top_k: int):
            self.top_k = top_k
            self.per = top_k // 32
            self.steps = []
            size = 2
            while size <= top_k:
                d = size // 2
                while d >= 1:
                    self.steps.append((size, d))
                    d //= 2
                size *= 2

        @cute.jit
        def __call__(
            self,
            indices: cute.Pointer,  # int32 [count, top_k] strided (in/out)
            valid: cute.Pointer,  # uint8 [keys]
            starts: cute.Pointer,  # int32 [count]
            lengths: cute.Pointer,  # int32 [count]
            count: Int32,
            width: Int32,
            indices_stride: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(indices, valid, starts, lengths, count, width, indices_stride).launch(
                grid=[(count + self.WARPS - 1) // self.WARPS, 1, 1],
                block=[self.WARPS * 32, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            indices: cute.Pointer,
            valid: cute.Pointer,
            starts: cute.Pointer,
            lengths: cute.Pointer,
            count: Int32,
            width: Int32,
            indices_stride: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            warp = tid // 32
            lane = tid % 32
            smem = SmemAllocator()
            stage = smem.allocate_tensor(Int32, cute.make_layout(self.WARPS * self.top_k), 16)
            base = warp * self.top_k
            q = block * self.WARPS + warp
            if q < count:
                n = Int64(count)
                indices_t = tensor2d(indices, n, Int64(self.top_k), indices_stride)
                valid_t = tensor1d(valid, Int64(2**62))
                row_start = Int64(tensor1d(starts, n)[q])
                columns = Int32(tensor1d(lengths, n)[q])
                if width < columns:
                    columns = width
                offsets = [Int32(indices_t[q, i * 32 + lane]) for i in range(self.per)]
                vals = []
                for i in cutlass.range_constexpr(self.per):
                    kept = Int32(0)
                    if (offsets[i] >= 0) & (offsets[i] < columns):
                        kept = Int32(cutlass.Uint8(valid_t[row_start + Int64(offsets[i])]) != 0)
                    vals.append(kept * offsets[i] + (1 - kept) * Int32(_INT32_MAX))
                # Pairs (e, e + d) with e & d == 0 go up when e & size == 0; the runtime choices
                # take min / max by arithmetic (values are offsets or INT32_MAX: no overflow).
                for step in cutlass.range_constexpr(len(self.steps)):
                    size, d = self.steps[step]
                    if cutlass.const_expr(d < self.per):
                        for r in cutlass.range_constexpr(self.per):
                            if cutlass.const_expr(r & d == 0):
                                lo = cutlass.min(vals[r], vals[r + d])
                                hi = cutlass.max(vals[r], vals[r + d])
                                if cutlass.const_expr(size < self.per):
                                    if cutlass.const_expr(r & size == 0):
                                        vals[r], vals[r + d] = lo, hi
                                    else:
                                        vals[r], vals[r + d] = hi, lo
                                elif cutlass.const_expr(size == self.top_k):
                                    vals[r], vals[r + d] = lo, hi
                                else:
                                    up = Int32((lane & (size // self.per)) == 0)
                                    vals[r] = hi + (lo - hi) * up
                                    vals[r + d] = lo + (hi - lo) * up
                    else:
                        lanes = d // self.per
                        take_min = Int32(
                            ((lane & lanes) == 0) == ((lane & (size // self.per)) == 0)
                        )
                        for r in cutlass.range_constexpr(self.per):
                            other = cute.arch.shuffle_sync_bfly(vals[r], offset=lanes)
                            lo = cutlass.min(vals[r], other)
                            hi = cutlass.max(vals[r], other)
                            vals[r] = hi + (lo - hi) * take_min
                # A kept offset is listed unless it repeats its predecessor.
                previous = cute.arch.shuffle_sync_up(
                    vals[self.per - 1], 1, mask=-1, mask_and_clamp=0
                )
                listed = []
                local = Int32(0)
                for r in cutlass.range_constexpr(self.per):
                    before = previous if r == 0 else vals[r - 1]
                    first = (lane == 0) & (r == 0)
                    keep = (vals[r] != Int32(_INT32_MAX)) & ((vals[r] != before) | first)
                    listed.append(keep)
                    if keep:
                        local = local + 1
                inclusive = warp_scan(local, tid, lane, 32)
                total = cute.arch.shuffle_sync(inclusive, 31)
                rank = inclusive - local
                for r in cutlass.range_constexpr(self.per):
                    if listed[r]:
                        stage[base + rank] = vals[r]
                        rank = rank + 1
                for slot in cutlass.range(total + lane, self.top_k, 32):
                    stage[base + slot] = Int32(-1)
                cute.arch.sync_warp()
                for i in cutlass.range_constexpr(self.per):
                    indices_t[q, i * 32 + lane] = stage[base + i * 32 + lane]

    class MapSelectionKernel:
        """Map candidate-column Top-K offsets to logical positions in place, in increasing column
        order with ``-1`` padding last, so the list does not depend on the order in which the
        Top-K wrote its selection (candidate columns map to increasing positions, so each row
        is sorted by position).

        An offset is kept when it lies below ``counts[q]`` (each row's valid columns are a
        prefix) and maps to a position ``>= 0``. One CTA per row flags the kept columns in a
        shared bitmap; each kept offset's slot is the number of flagged columns below it (a
        column selected twice is listed once).
        """

        THREADS = 256

        def __init__(self, top_k: int):
            if top_k <= 0 or top_k > 4096:
                raise ValueError("CSA2 selection map supports 1..4096 selections")
            self.top_k = top_k
            self.words = _MAP_BITMAP_COLUMNS // 32  # bitmap capacity in 32-column words
            self.per_thread = (top_k + self.THREADS - 1) // self.THREADS

        @cute.jit
        def __call__(
            self,
            indices: cute.Pointer,  # int32 [count, top_k] strided (in/out)
            positions: cute.Pointer,  # int32 [count, width] strided
            counts: cute.Pointer,  # int32 [count]
            count: Int32,
            width: Int32,
            indices_stride: Int64,
            positions_stride: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                indices,
                positions,
                counts,
                count,
                width,
                indices_stride,
                positions_stride,
            ).launch(grid=[count, 1, 1], block=[self.THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            indices: cute.Pointer,
            positions: cute.Pointer,
            counts: cute.Pointer,
            count: Int32,
            width: Int32,
            indices_stride: Int64,
            positions_stride: Int64,
        ):
            threads: cutlass.Constexpr = self.THREADS
            warps: cutlass.Constexpr = self.THREADS // 32
            tid, _, _ = cute.arch.thread_idx()
            q, _, _ = cute.arch.block_idx()
            warp = tid // 32
            lane = tid % 32
            smem = SmemAllocator()
            flags = smem.allocate_tensor(cutlass.Uint32, cute.make_layout(self.words), 16)
            below = smem.allocate_tensor(Int32, cute.make_layout(self.words), 16)
            totals = smem.allocate_tensor(Int32, cute.make_layout(warps), 16)
            n = Int64(count)
            indices_t = tensor2d(indices, n, Int64(self.top_k), indices_stride)
            positions_t = tensor2d(positions, n, Int64(width), positions_stride)
            # Issue the count and offset loads together, clear the bitmap while they are in
            # flight, then load the kept offsets' positions before the first barrier.
            valid = Int32(tensor1d(counts, n)[q])
            kept = cute.make_rmem_tensor((self.per_thread,), Int32)
            mapped = cute.make_rmem_tensor((self.per_thread,), Int32)
            for j in cutlass.range_constexpr(self.per_thread):
                kept[j] = Int32(-1)
                k = tid + j * threads
                if k < self.top_k:
                    kept[j] = Int32(indices_t[q, k])
            if valid > width:
                valid = width
            used = (valid + 31) // 32
            for word in cutlass.range(tid, used, threads):
                flags[word] = cutlass.Uint32(0)
            for j in cutlass.range_constexpr(self.per_thread):
                offset = kept[j]
                mapped[j] = Int32(-1)
                if (offset >= 0) & (offset < valid):
                    mapped[j] = Int32(positions_t[q, offset])
            cute.arch.barrier()
            for j in cutlass.range_constexpr(self.per_thread):
                offset = kept[j]
                if mapped[j] >= 0:
                    cute.arch.atomic_or(
                        ptr=(flags.iterator + (offset // 32)).llvm_ptr,
                        val=cutlass.Uint32(1) << cutlass.Uint32(offset % 32),
                        sem="relaxed",
                        scope="cta",
                    )
                else:
                    kept[j] = Int32(-1)
            cute.arch.barrier()
            # Flagged columns below each word: each thread counts a run of words, then an
            # exclusive scan over the threads (warp shuffles, then the warp totals).
            chunk = (used + threads - 1) // threads
            begin = tid * chunk
            end = begin + chunk
            if end > used:
                end = used
            local = Int32(0)
            for word in cutlass.range(begin, end, 1):
                local = local + Int32(cute.arch.popc(flags[word]))
            inclusive = warp_scan(local, tid, lane, 32)
            if lane == 31:
                totals[warp] = inclusive
            cute.arch.barrier()
            run = inclusive - local
            total = Int32(0)
            for w in cutlass.range_constexpr(warps):
                if w < warp:
                    run = run + totals[w]
                total = total + totals[w]
            for word in cutlass.range(begin, end, 1):
                below[word] = run
                run = run + Int32(cute.arch.popc(flags[word]))
            cute.arch.barrier()
            for j in cutlass.range_constexpr(self.per_thread):
                offset = kept[j]
                if offset >= 0:
                    word = offset // 32
                    lower = (cutlass.Uint32(1) << cutlass.Uint32(offset % 32)) - cutlass.Uint32(1)
                    slot = below[word] + Int32(cute.arch.popc(flags[word] & lower))
                    indices_t[q, slot] = mapped[j]
            for k in cutlass.range(total + tid, self.top_k, threads):
                indices_t[q, k] = Int32(-1)

    # ------------------------------------------- K5: packed prefill prefix keys

    class PrefixGatherKernel:
        """Compressed prefix keys and query-row bounds of one packed prefill group.

        ``descriptors`` is int32 ``[pieces, PREFIX_DESC_WIDTH]``: request,
        key width, key offset, first query row and its row inside the group,
        with both offsets ascending. Thread ``t < total_keys * 17`` copies
        32-bit word ``t % 17`` of key ``t // 17`` (16 data words, then the scale
        word) from the page-footer index pages, resolving the key's logical
        position through its request's GLOBAL page table; positions without a
        page are zero rows with ``valid = 0``. The next ``rows`` threads write
        ``rows_out[5, rows]``: key start, key end ``start + clamp(visible, 0,
        width)``, that length, the block width and a zero row start.
        """

        THREADS = 256
        WORDS = INDEX_ROW_WORDS + 1

        @cute.jit
        def __call__(
            self,
            pages: cute.Pointer,  # int32 words of the [pages, 64, 1, 68] index pages
            table: cute.Pointer,  # int32 [table_rows, columns] strided
            visible: cute.Pointer,  # int32/int64 [query rows]
            descriptors: cute.Pointer,  # int32 [pieces, PREFIX_DESC_WIDTH]
            keys: cute.Pointer,  # int32 [total_keys, 16] (out)
            scales: cute.Pointer,  # int32 [total_keys] (out)
            valid: cute.Pointer,  # uint8 [total_keys] (out)
            rows_out: cute.Pointer,  # int32 [5, rows] (out)
            pieces: Int32,
            table_rows: Int32,
            columns: Int32,
            table_stride: Int64,
            page_size: Int32,
            max_positions: Int32,
            capacity: Int64,
            total_keys: Int32,
            rows: Int32,
            stream: cuda.CUstream,
        ):
            total = Int64(total_keys) * Int64(self.WORDS) + Int64(rows)
            blocks = Int32((total + Int64(self.THREADS - 1)) // Int64(self.THREADS))
            self.kernel(
                pages,
                table,
                visible,
                descriptors,
                keys,
                scales,
                valid,
                rows_out,
                pieces,
                table_rows,
                columns,
                table_stride,
                page_size,
                max_positions,
                capacity,
                total_keys,
                rows,
            ).launch(grid=[blocks, 1, 1], block=[self.THREADS, 1, 1], stream=stream)

        @cute.jit
        def _piece(self, desc, pieces: Int32, column: Int32, value: Int32) -> Int32:
            """Last piece whose ascending ``desc[:, column]`` is ``<= value``."""
            lo = Int32(0)
            hi = pieces
            for _ in cutlass.range(0, 32, 1):
                if lo < hi:
                    mid = (lo + hi) // 2
                    if Int32(desc[mid, column]) <= value:
                        lo = mid + 1
                    else:
                        hi = mid
            piece = lo - 1
            if piece < 0:
                piece = Int32(0)
            return piece

        @cute.kernel
        def kernel(
            self,
            pages: cute.Pointer,
            table: cute.Pointer,
            visible: cute.Pointer,
            descriptors: cute.Pointer,
            keys: cute.Pointer,
            scales: cute.Pointer,
            valid: cute.Pointer,
            rows_out: cute.Pointer,
            pieces: Int32,
            table_rows: Int32,
            columns: Int32,
            table_stride: Int64,
            page_size: Int32,
            max_positions: Int32,
            capacity: Int64,
            total_keys: Int32,
            rows: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            desc = tensor2d(
                descriptors, Int64(pieces), Int64(PREFIX_DESC_WIDTH), Int64(PREFIX_DESC_WIDTH)
            )
            t = Int64(bid) * Int64(self.THREADS) + Int64(tid)
            key_words = Int64(total_keys) * Int64(self.WORDS)
            if t < key_words:
                key = t // Int64(self.WORDS)
                word = Int32(t % Int64(self.WORDS))
                piece = self._piece(desc, pieces, 2, Int32(key))
                request = Int32(desc[piece, 0])
                position = key - Int64(desc[piece, 2])
                column = position // Int64(page_size)
                slot = Int64(-1)
                if (
                    (position < Int64(max_positions))
                    & (column < Int64(columns))
                    & (request >= 0)
                    & (request < table_rows)
                ):
                    physical = Int64(
                        tensor2d(table, Int64(table_rows), Int64(columns), table_stride)[
                            request, Int32(column)
                        ]
                    )
                    if physical >= 0:
                        slot = physical * Int64(page_size) + position % Int64(page_size)
                backed = (slot >= 0) & (slot < capacity)
                value = Int32(0)
                if backed:
                    page = slot // Int64(INDEX_PAGE_ROWS)
                    row = slot % Int64(INDEX_PAGE_ROWS)
                    base = page * Int64(INDEX_PAGE_WORDS)
                    source = base + row * Int64(INDEX_ROW_WORDS) + Int64(word)
                    if word == INDEX_ROW_WORDS:
                        source = base + Int64(INDEX_PAGE_ROWS * INDEX_ROW_WORDS) + row
                    value = Int32(tensor1d(pages, Int64(2**62))[source])
                if word == INDEX_ROW_WORDS:
                    tensor1d(scales, Int64(2**62))[key] = value
                    flag = cutlass.Uint8(0)
                    if backed:
                        flag = cutlass.Uint8(1)
                    tensor1d(valid, Int64(2**62))[key] = flag
                else:
                    tensor1d(keys, Int64(2**62))[key * Int64(INDEX_ROW_WORDS) + Int64(word)] = value
            elif t < key_words + Int64(rows):
                r = Int32(t - key_words)
                piece = self._piece(desc, pieces, 4, r)
                width = Int32(desc[piece, 1])
                start = Int32(desc[piece, 2])
                query = Int32(desc[piece, 3]) + r - Int32(desc[piece, 4])
                end = Int64(tensor1d(visible, Int64(2**62))[query])
                if end < 0:
                    end = Int64(0)
                if end > Int64(width):
                    end = Int64(width)
                out = tensor2d(rows_out, Int64(5), Int64(rows), Int64(rows))
                out[0, r] = start
                out[1, r] = start + Int32(end)
                out[2, r] = Int32(end)
                out[3, r] = width
                out[4, r] = Int32(0)

    # ------------------------------------------- K6: candidate block maxima

    class CandidateBlockMaxKernel:
        """Block maxima of bounded compressed logits for candidate-source rows.

        Row ``q`` scores column ``c`` (logical position ``c``) for
        ``c < row_ends[q] - row_starts[q]`` where
        ``valid[q * valid_stride + (row_starts[q] + c) // page]`` is set: ``page`` 1
        reads one byte per packed key, a larger one a byte per page of a row whose start
        is 0. Blocks reduce ``block_size`` columns; blocks without a scored column are ``-inf``,
        the newest visible block is pinned with ``+inf`` and blocks at or
        beyond ``ceil(row_widths[q] / block_size)`` stay ``-inf``. The first
        column tile also writes that block count, the row bound of the block
        TopK.
        """

        THREADS = 128
        # Blocks per thread, THREADS apart so each pass stays coalesced.
        BLOCKS = 4

        def __init__(self, block_size: int, vector: bool, page: int = 1):
            self.block_size = block_size
            self.page = page
            # Blocks that tile the pages read one validity byte for all their columns.
            self.paged_blocks = page > 1 and page % block_size == 0
            # 16-byte logits and 4-byte validity loads for complete blocks whose row
            # offsets are aligned; the caller checks the bases and strides.
            self.vector = vector and block_size % 4 == 0 and (page == 1 or self.paged_blocks)

        @cute.jit
        def __call__(
            self,
            logits: cute.Pointer,  # float32 [count, width] strided
            valid: cute.Pointer,  # uint8 rows of ``valid_stride`` keys
            visible: cute.Pointer,  # int32/int64 [>= query_start + count]
            row_starts: cute.Pointer,  # int32 [count]
            row_ends: cute.Pointer,  # int32 [count]
            row_widths: cute.Pointer,  # int32 [count]
            blocks: cute.Pointer,  # float32 [count, block_capacity] (out)
            block_counts: cute.Pointer,  # int32 [count] (out)
            count: Int32,
            width: Int32,
            block_capacity: Int32,
            query_start: Int32,
            logits_stride: Int64,
            valid_stride: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                logits,
                valid,
                visible,
                row_starts,
                row_ends,
                row_widths,
                blocks,
                block_counts,
                count,
                width,
                block_capacity,
                query_start,
                logits_stride,
                valid_stride,
            ).launch(
                grid=[count, -(-block_capacity // (self.THREADS * self.BLOCKS)), 1],
                block=[self.THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            logits: cute.Pointer,
            valid: cute.Pointer,
            visible: cute.Pointer,
            row_starts: cute.Pointer,
            row_ends: cute.Pointer,
            row_widths: cute.Pointer,
            blocks: cute.Pointer,
            block_counts: cute.Pointer,
            count: Int32,
            width: Int32,
            block_capacity: Int32,
            query_start: Int32,
            logits_stride: Int64,
            valid_stride: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, chunk, _ = cute.arch.block_idx()
            n = Int64(count)
            start = Int32(tensor1d(row_starts, n)[q])
            length = Int32(tensor1d(row_ends, n)[q]) - start
            block_size = Int32(self.block_size)
            row_blocks = (Int32(tensor1d(row_widths, n)[q]) + block_size - 1) // block_size
            latest = floor_div(
                Int64(tensor1d(visible, Int64(2**62))[query_start + q]) - 1, Int64(block_size)
            )
            if (chunk == 0) & (tid == 0):
                tensor1d(block_counts, n)[q] = row_blocks
            for i in cutlass.range_constexpr(self.BLOCKS):
                b = (chunk * self.BLOCKS + i) * self.THREADS + tid
                if b < block_capacity:
                    value = Float32(float("-inf"))
                    if b < row_blocks:
                        if Int64(b) == latest:
                            value = Float32(float("inf"))
                        else:
                            logits_t = tensor2d(logits, n, Int64(width), logits_stride)
                            valid_t = tensor1d(valid, Int64(2**62))
                            base = Int64(q) * valid_stride + Int64(start)
                            first = b * block_size
                            last = first + block_size
                            if last > length:
                                last = length
                            scalar = cutlass.Boolean(True)
                            if cutlass.const_expr(self.paged_blocks):
                                page_byte = valid_t[
                                    Int64(q) * valid_stride + Int64(first // self.page)
                                ]
                                if cutlass.Uint8(page_byte) == 0:
                                    scalar = cutlass.Boolean(False)
                            if cutlass.const_expr(self.vector):
                                aligned = (base + Int64(first)) % 4 == 0
                                if cutlass.const_expr(self.paged_blocks):
                                    aligned = scalar
                                if (last - first == block_size) & aligned:
                                    scalar = cutlass.Boolean(False)
                                    row = logits.toint() + (
                                        Int64(q) * logits_stride + Int64(first)
                                    ) * Int64(4)
                                    for part in cutlass.range_constexpr(self.block_size // 4):
                                        values = cute.make_rmem_tensor((4,), Float32)
                                        cute.autovec_copy(
                                            cute.make_tensor(
                                                cute.make_ptr(
                                                    Float32,
                                                    row + 16 * part,
                                                    cute.AddressSpace.gmem,
                                                    assumed_align=16,
                                                ),
                                                cute.make_layout(4),
                                            ),
                                            values,
                                        )
                                        if cutlass.const_expr(self.paged_blocks):
                                            for j in cutlass.range_constexpr(4):
                                                value = cute.arch.fmax(value, values[j])
                                        else:
                                            flags = cutlass.Uint32(
                                                cute.make_tensor(
                                                    cute.make_ptr(
                                                        cutlass.Uint32,
                                                        valid.toint()
                                                        + base
                                                        + Int64(first + 4 * part),
                                                        cute.AddressSpace.gmem,
                                                        assumed_align=4,
                                                    ),
                                                    cute.make_layout(1),
                                                )[0]
                                            )
                                            for j in cutlass.range_constexpr(4):
                                                if ((flags >> (8 * j)) & cutlass.Uint32(255)) != 0:
                                                    value = cute.arch.fmax(value, values[j])
                            if scalar:
                                for column in cutlass.range(first, last, 1):
                                    if cutlass.const_expr(self.paged_blocks):
                                        value = cute.arch.fmax(value, Float32(logits_t[q, column]))
                                    else:
                                        key = base + Int64(column)
                                        if cutlass.const_expr(self.page > 1):
                                            key = Int64(q) * valid_stride + Int64(
                                                column // self.page
                                            )
                                        if cutlass.Uint8(valid_t[key]) != 0:
                                            value = cute.arch.fmax(
                                                value, Float32(logits_t[q, column])
                                            )
                    tensor2d(blocks, n, Int64(block_capacity), Int64(block_capacity))[q, b] = value

    # ------------------------------------------- K7: candidate expansion

    class CandidateExpandKernel:
        """Publish candidate positions of rows whose block Top-K is already selected.

        The first ``min(top_k, row_blocks)`` selections whose block maximum is
        finite are sorted ascending (others last) and expanded to positions;
        positions at or beyond the visible length or the row width publish
        ``-1``. With ``sparse_block`` it also writes the DeepGEMM sparse-logits
        inputs: the selected sparse block ids, padded by repeating the last
        valid one, and the number of valid leading positions.
        """

        def __init__(self, block_size: int, top_k: int, sparse_block: int, bitmap_blocks: int):
            if top_k <= 0 or top_k > 4096:
                raise ValueError("CSA2 candidate expansion supports 1..4096 selected blocks")
            if sparse_block < 0 or (sparse_block and block_size % sparse_block):
                raise ValueError("CSA2 sparse blocks must divide the candidate block")
            self.block_size = block_size
            self.top_k = top_k
            self.padded = 1 << (top_k - 1).bit_length()
            self.sparse_block = sparse_block
            self.ratio = block_size // sparse_block if sparse_block else 1
            self.threads = max(32, min(256, self.padded // 2 if self.padded >= 64 else 32))
            self.bitmap_blocks = bitmap_blocks

        @cute.jit
        def __call__(
            self,
            selected: cute.Pointer,  # int32 [count, top_k] strided
            blocks: cute.Pointer,  # float32 [count, block_capacity]
            visible: cute.Pointer,  # int32/int64 [>= query_start + count]
            row_widths: cute.Pointer,  # int32 [count]
            output: cute.Pointer,  # int32 [count, published_width] strided (out)
            sparse_ids: cute.Pointer,  # int32 [count, published_width // sparse_block] (out)
            sparse_counts: cute.Pointer,  # int32 [count] (out)
            count: Int32,
            block_capacity: Int32,
            query_start: Int32,
            published_width: Int32,
            selected_stride: Int64,
            output_stride: Int64,
            sparse_stride: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                selected,
                blocks,
                visible,
                row_widths,
                output,
                sparse_ids,
                sparse_counts,
                count,
                block_capacity,
                query_start,
                published_width,
                selected_stride,
                output_stride,
                sparse_stride,
            ).launch(grid=[count, 1, 1], block=[self.threads, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            selected: cute.Pointer,
            blocks: cute.Pointer,
            visible: cute.Pointer,
            row_widths: cute.Pointer,
            output: cute.Pointer,
            sparse_ids: cute.Pointer,
            sparse_counts: cute.Pointer,
            count: Int32,
            block_capacity: Int32,
            query_start: Int32,
            published_width: Int32,
            selected_stride: Int64,
            output_stride: Int64,
            sparse_stride: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, _, _ = cute.arch.block_idx()
            smem = SmemAllocator()
            ids = smem.allocate_tensor(Int32, cute.make_layout(self.padded), 16)
            partial = smem.allocate_tensor(Int32, cute.make_layout(max(self.threads // 32, 1)), 16)
            flags = smem.allocate_tensor(
                cutlass.Uint32, cute.make_layout(self.bitmap_blocks // 32), 16
            )
            prefix = smem.allocate_tensor(Int32, cute.make_layout(self.threads), 16)
            n = Int64(count)
            block_size = Int64(self.block_size)
            width = Int64(tensor1d(row_widths, n)[q])
            row_blocks = Int32((width + block_size - 1) // block_size)
            selected_blocks = row_blocks
            if selected_blocks > self.top_k:
                selected_blocks = Int32(self.top_k)
            visible_length = Int64(tensor1d(visible, Int64(2**62))[query_start + q])
            selected_t = tensor2d(selected, n, Int64(self.top_k), selected_stride)
            blocks_t = tensor2d(blocks, n, Int64(block_capacity), Int64(block_capacity))
            # Flag the reachable selected blocks in a bitmap, then compact the set bits in
            # block order: the reachable blocks ascending, followed by sentinels.
            words = (row_blocks + 31) // 32
            for word in cutlass.range(tid, words, self.threads):
                flags[word] = cutlass.Uint32(0)
            cute.arch.barrier()
            for k in cutlass.range(tid, selected_blocks, self.threads):
                block = Int32(selected_t[q, k])
                if (block >= 0) & (block < row_blocks):
                    if Float32(blocks_t[q, block]) > Float32(float("-inf")):
                        cute.arch.atomic_or(
                            ptr=(flags.iterator + (block // 32)).llvm_ptr,
                            val=cutlass.Uint32(1) << cutlass.Uint32(block % 32),
                            sem="relaxed",
                            scope="cta",
                        )
            cute.arch.barrier()
            chunk = (words + self.threads - 1) // self.threads
            begin = tid * chunk
            end = begin + chunk
            if end > words:
                end = words
            local = Int32(0)
            for word in cutlass.range(begin, end, 1):
                local = local + Int32(cute.arch.popc(flags[word]))
            prefix[tid] = local
            cute.arch.barrier()
            offset = 1
            while offset < self.threads:
                addend = Int32(0)
                if tid >= offset:
                    addend = prefix[tid - offset]
                cute.arch.barrier()
                prefix[tid] = prefix[tid] + addend
                cute.arch.barrier()
                offset = offset * 2
            rank = prefix[tid] - local
            for word in cutlass.range(begin, end, 1):
                bits = flags[word]
                for bit in cutlass.range_constexpr(32):
                    if (bits >> cutlass.Uint32(bit)) & cutlass.Uint32(1) != cutlass.Uint32(0):
                        ids[rank] = word * 32 + bit
                        rank = rank + 1
            num_valid = prefix[self.threads - 1]
            for k in cutlass.range(num_valid + tid, selected_blocks, self.threads):
                ids[k] = Int32(_INT32_MAX)
            cute.arch.barrier()
            out_t = tensor2d(output, n, Int64(published_width), output_stride)
            selected_width = Int64(selected_blocks) * block_size
            local_valid = Int32(0)
            for k in cutlass.range(tid, selected_blocks, self.threads):
                block = Int64(ids[k])
                for j in cutlass.range(0, self.block_size, 1):
                    position = block * block_size + Int64(j)
                    published = Int64(-1)
                    if (
                        (block < Int64(_INT32_MAX))
                        & (position < visible_length)
                        & (position < width)
                    ):
                        published = position
                        local_valid = local_valid + 1
                    out_t[q, Int32(Int64(k) * block_size + Int64(j))] = Int32(published)
            for column in cutlass.range(Int32(selected_width) + tid, published_width, self.threads):
                out_t[q, column] = Int32(-1)
            if cutlass.const_expr(self.sparse_block > 0):
                # Block-wide sum of the valid positions.
                offset = 1
                while offset < 32:
                    local_valid = local_valid + cute.arch.shuffle_sync_bfly(
                        local_valid, offset=offset
                    )
                    offset = offset * 2
                if cutlass.const_expr(self.threads > 32):
                    if tid % 32 == 0:
                        partial[tid // 32] = local_valid
                    cute.arch.barrier()
                    local_valid = Int32(0)
                    for warp in cutlass.range_constexpr(self.threads // 32):
                        local_valid = local_valid + partial[warp]
                if tid == 0:
                    tensor1d(sparse_counts, n)[q] = local_valid
                last_block = Int32(0)
                if num_valid > 0:
                    last_block = ids[num_valid - 1]
                ratio = Int32(self.ratio)
                sparse_width = published_width // Int32(self.sparse_block)
                sparse_t = tensor2d(sparse_ids, n, Int64(sparse_width), sparse_stride)
                selected_sparse = selected_blocks * ratio
                for column in cutlass.range(tid, sparse_width, self.threads):
                    identifier = last_block * ratio + (ratio - 1)
                    if column < selected_sparse:
                        k = column // ratio
                        block = last_block
                        if k < num_valid:
                            block = ids[k]
                        identifier = block * ratio + column % ratio
                    sparse_t[q, column] = identifier

    class Namespace:
        pass

    ns = Namespace()
    ns.cuda = cuda
    ns.cutlass = cutlass
    ns.cute = cute
    ns.llvm = llvm
    ns.T = T
    ns.dsl_user_op = dsl_user_op
    ns.SmemAllocator = SmemAllocator
    ns.make_ptr = make_ptr
    ns.griddepcontrol_wait = griddepcontrol_wait
    ns.griddepcontrol_launch_dependents = griddepcontrol_launch_dependents
    ns.bitcast_u32_f32 = bitcast_u32_f32
    ns.tensor1d = tensor1d
    ns.tensor2d = tensor2d
    ns.SwaRefreshKernel = SwaRefreshKernel
    ns.PageTableConvertKernel = PageTableConvertKernel
    ns.TableCopyKernel = TableCopyKernel
    ns.OwnerSlotsKernel = OwnerSlotsKernel
    ns.IndexerDescriptorsKernel = IndexerDescriptorsKernel
    ns.MaskLogitsKernel = MaskLogitsKernel
    ns.MaskPackedLogitsKernel = MaskPackedLogitsKernel
    ns.CandidateGatherKernel = CandidateGatherKernel
    ns.FinalizeSelectionKernel = FinalizeSelectionKernel
    ns.PackedSortFinalizeKernel = PackedSortFinalizeKernel
    ns.MapSelectionKernel = MapSelectionKernel
    ns.PrefixGatherKernel = PrefixGatherKernel
    ns.CandidateBlockMaxKernel = CandidateBlockMaxKernel
    ns.CandidateExpandKernel = CandidateExpandKernel
    return ns


# ----------------------------------------------------------------- host side

_COMPILED: dict[tuple, Any] = {}
_DTYPE_MAP = None
_SIGNATURES: dict[tuple[torch.dtype, int], tuple[Any, int]] = {}


def _cute_dtype(dtype: torch.dtype):
    global _DTYPE_MAP
    if _DTYPE_MAP is None:
        cutlass = _glue_kernels().cutlass
        _DTYPE_MAP = {
            torch.uint8: cutlass.Uint8,
            torch.bool: cutlass.Uint8,
            torch.int32: cutlass.Int32,
            torch.int64: cutlass.Int64,
            torch.float32: cutlass.Float32,
            torch.bfloat16: cutlass.BFloat16,
            torch.float8_e4m3fn: cutlass.Float8E4M3FN,
            torch.uint32: cutlass.Uint32,
        }
    return _DTYPE_MAP[dtype]


def _ptr(tensor: torch.Tensor, dtype: torch.dtype | None = None, align: int = 4):
    """Raw address and interned pointer signature for a TVM-FFI launch."""
    address = tensor.data_ptr()
    while align > 1 and address % align:
        align //= 2
    key = (dtype or tensor.dtype, align)
    signature = _SIGNATURES.get(key)
    if signature is None:
        signature = _SIGNATURES[key] = (_cute_dtype(key[0]), align)
    return (address, signature)


class _Stream(int):
    """The raw handle of a CUDA stream: a ``CUstream`` when compiled, the handle when launched."""


def _stream(device: torch.device) -> _Stream:
    index = torch.cuda.current_device() if device.index is None else device.index
    return _Stream(torch._C._cuda_getCurrentRawStream(index))


def _compile(factory, traced_args):
    """Materialize CuTe pointer types once, then compile the TVM-FFI entry."""
    ns = _glue_kernels()
    traced = []
    for arg in traced_args:
        if type(arg) is tuple:
            address, (element, align) = arg
            arg = ns.make_ptr(
                element,
                address,
                ns.cute.AddressSpace.gmem,
                assumed_align=align,
            )
        elif type(arg) is _Stream:
            arg = ns.cuda.CUstream(int(arg))
        traced.append(arg)
    return ns.cute.compile(factory(), *traced, options="--enable-tvm-ffi")


def _launch(key: tuple, factory, *args) -> None:
    """Compile once per ``key`` and pointer signature (never inside a capture), then launch.

    Pointer element types and assumed alignments are part of the compiled
    signature, so they extend the caller's key: a call site that alternates
    between int32 and int64 tensors gets one specialization per type instead
    of reusing a function compiled for the other.
    """
    signature = tuple(arg[1] if type(arg) is tuple else None for arg in args)
    key = (key, signature)
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up CSA2 CuTe kernels before CUDA Graph capture")
        compiled = _COMPILED[key] = _compile(factory, args)
    launch_args = tuple(
        arg[0] if type(arg) is tuple else int(arg) if type(arg) is _Stream else arg for arg in args
    )
    compiled(*launch_args)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pdl() -> bool:
    """Whether CSA2 staging kernels launch with PDL (``TRTLLM_ENABLE_PDL``)."""
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    return get_env_enable_pdl()


def _as_int64(tensor: torch.Tensor) -> torch.Tensor:
    """Contiguous int64 view of an integer tensor (a copy only when it is not already one)."""
    if tensor.dtype != torch.int64:
        tensor = tensor.long()
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def convert_page_tables(
    base: torch.Tensor,
    rows: torch.Tensor,
    params: torch.Tensor,
    out: torch.Tensor,
    bad: int,
    scratch_ranges: torch.Tensor | None = None,
    scratch_slots: torch.Tensor | None = None,
    low_out: torch.Tensor | None = None,
) -> None:
    """``out[t, s, c] = base[pool_t, rows[s], c] * scale_t + offset_t``, keeping ``bad``.

    ``base`` is device int32 ``[pools, rows, blocks]`` base page rows and ``rows``
    the int32 ``[seqs]`` row of each batch sequence.
    ``params`` is int32 ``[tables, 5]`` of
    (pool, scale, offset, scratch pool, scratch pages per block); ``out`` is
    int32 ``[tables, seqs, width]`` with unit column stride. With
    ``scratch_ranges`` (int32 ``[scratch pools, seqs, 2]``) and
    ``scratch_slots`` (int32 ``[scratch pools, seqs, n]``), columns inside a
    range map to the scratch slots like ``PageIndexConverter``. With ``low_out``
    (int32 ``[L, seqs, low width]``, unit column stride, ``low width <= width``) tables
    ``0..L-1`` are written there instead of ``out``, in the same launch.
    """
    tables, seqs, width = out.shape
    if tables == 0 or seqs == 0 or width == 0:
        return
    _check(base.dtype == torch.int32 and base.dim() == 3 and base.is_contiguous(), "base rows")
    _check(width <= base.shape[2], "base rows cover the table width")
    _check(rows.dtype == torch.int32 and rows.shape == (seqs,) and rows.is_contiguous(), "rows")
    _check(params.shape == (tables, 5) and params.dtype == torch.int32, "params [T, 5] int32")
    _check(params.is_contiguous() and out.dtype == torch.int32 and out.stride(2) == 1, "out")
    with_scratch = scratch_ranges is not None
    if with_scratch:
        _check(
            scratch_ranges.dtype == scratch_slots.dtype == torch.int32
            and scratch_ranges.shape[1:] == (seqs, 2)
            and scratch_slots.shape[:2] == scratch_ranges.shape[:2]
            and scratch_ranges.is_contiguous()
            and scratch_slots.is_contiguous(),
            "CSA2 scratch ranges/slots",
        )
    else:
        scratch_ranges = scratch_slots = params
    with_low = low_out is not None
    if with_low:
        _check(
            low_out.dtype == torch.int32
            and low_out.dim() == 3
            and low_out.shape[0] <= tables
            and low_out.shape[1] == seqs
            and low_out.shape[2] <= width
            and low_out.stride(2) == 1,
            "CSA2 low tables",
        )
    else:
        low_out = out
    ns = _glue_kernels()
    _launch(
        ("page_table_convert", with_scratch, with_low, out.device.index),
        lambda: ns.PageTableConvertKernel(with_scratch, with_low),
        _ptr(base),
        _ptr(rows),
        _ptr(params),
        _ptr(scratch_ranges),
        _ptr(scratch_slots),
        _ptr(out),
        int(seqs),
        int(tables),
        int(width),
        int(scratch_slots.shape[-1]) if with_scratch else 1,
        int(base.stride(0)),
        int(base.stride(1)),
        int(out.stride(0)),
        int(out.stride(1)),
        int(bad),
        _ptr(low_out),
        int(low_out.shape[0]) if with_low else 0,
        int(low_out.shape[2]),
        int(low_out.stride(0)),
        int(low_out.stride(1)),
        _stream(out.device),
    )


def table_copy_descriptor(destination: torch.Tensor, source: torch.Tensor) -> tuple[int, ...]:
    """One ``copy_tables_`` descriptor row for int32 ``[rows, columns]`` tables with unit column stride."""
    _check(
        destination.dtype == source.dtype == torch.int32
        and destination.shape == source.shape
        and destination.dim() == 2
        and destination.stride(1) == source.stride(1) == 1,
        "CSA2 table copies need equal int32 [rows, columns] tables",
    )
    return (
        source.data_ptr(),
        destination.data_ptr(),
        int(destination.shape[0]),
        int(destination.shape[1]),
        int(source.stride(0)),
        int(destination.stride(0)),
    )


def copy_tables_(descriptors: torch.Tensor, max_rows: int) -> None:
    """Copy every table of the device int64 ``[tables, 6]`` ``table_copy_descriptor`` rows,
    none of them taller than ``max_rows``, in one launch."""
    tables = int(descriptors.shape[0])
    if tables == 0 or max_rows == 0:
        return
    _check(
        descriptors.dtype == torch.int64
        and descriptors.is_contiguous()
        and descriptors.shape[1:] == (6,),
        "CSA2 table copy descriptors",
    )
    ns = _glue_kernels()
    _launch(
        ("table_copy", descriptors.device.index),
        ns.TableCopyKernel,
        _ptr(descriptors, align=8),
        tables,
        int(max_rows),
        _stream(descriptors.device),
    )


def refresh_swa_slots(
    positions: torch.Tensor,
    requests: torch.Tensor,
    floors: torch.Tensor,
    read_floors: torch.Tensor,
    tables: torch.Tensor,
    ratios: torch.Tensor,
    reads: torch.Tensor,
    writes: torch.Tensor,
    visible: torch.Tensor,
    block: int,
) -> None:
    """Fill ``reads[S, T, W]`` and ``writes[S, T]`` of the first ``S`` (SWA) layers and
    ``visible[L, T]`` of every model layer in one launch.

    Positions below ``floors`` are neither read nor written; positions below
    ``read_floors`` (at least ``floors``) are written but not read.

    ``tables`` is int64 ``[S, 4]`` holding each SWA layer's page-table pointer,
    row stride (elements), row count and column count; ``ratios`` is int32 ``[L]``.
    """
    swa_layers, tokens, window = reads.shape
    layers = visible.shape[0]
    if tokens == 0 or layers == 0:
        return
    _check(reads.is_contiguous() and writes.is_contiguous() and visible.is_contiguous(), "slabs")
    _check(reads.dtype == writes.dtype == visible.dtype == torch.int64, "CSA2 SWA slabs are int64")
    _check(positions.dtype == torch.int32 and positions.is_contiguous(), "positions int32")
    _check(requests.dtype == torch.int64 and requests.is_contiguous(), "requests int64")
    for floor in (floors, read_floors):
        _check(floor.dtype == torch.int64 and floor.is_contiguous(), "floors int64")
    _check(tables.shape == (swa_layers, 4) and tables.dtype == torch.int64, "tables [S, 4] int64")
    _check(ratios.shape == (layers,) and ratios.dtype == torch.int32, "ratios [L] int32")
    ns = _glue_kernels()
    key = ("swa_refresh", window, block, positions.device.index)
    _launch(
        key,
        lambda: ns.SwaRefreshKernel(window, block),
        _ptr(positions),
        _ptr(requests, align=8),
        _ptr(floors, align=8),
        _ptr(read_floors, align=8),
        _ptr(tables, align=8),
        _ptr(ratios),
        _ptr(reads, align=8),
        _ptr(writes, align=8),
        _ptr(visible, align=8),
        int(tokens),
        int(swa_layers),
        int(layers),
        int(floors.numel()),
        _stream(positions.device),
    )


OWNER_DESC_WIDTH = 16


def owner_slot_descriptor(
    source_base: torch.Tensor,
    source_lengths: torch.Tensor,
    pages: torch.Tensor,
    slots: torch.Tensor,
    compressed: torch.Tensor,
    ratio: int,
    page_size: int,
    batch_starts: torch.Tensor | None = None,
    batch_lengths: torch.Tensor | None = None,
    batch_cu: torch.Tensor | None = None,
) -> tuple[int, ...]:
    """One owner's ``refresh_owner_slots`` descriptor row: (source base, source
    lengths, pages, page columns, page row stride, slots, compressed positions,
    capacity, ratio, page size, batch starts, batch lengths, batch cu), zero padded.
    Without a compression batch the three batch pointers are 0."""
    requests = int(source_base.numel())
    for tensor, dtype in (
        (source_base, torch.int32),
        (source_lengths, torch.int32),
        (compressed, torch.int32),
        (slots, torch.int64),
    ):
        _check(tensor.dtype == dtype and tensor.is_contiguous(), "CSA2 owner refresh dtypes")
    _check(pages.dtype == torch.int32 and pages.ndim == 2 and pages.stride(1) == 1, "pages")
    _check(pages.shape[0] >= requests, "CSA2 owner page table must cover every request")
    batch = (0, 0, 0)
    if batch_starts is not None:
        _check(
            batch_starts.dtype == torch.int32
            and batch_lengths.dtype == torch.int32
            and batch_cu.dtype == torch.int32
            and batch_cu.numel() == requests + 1,
            "CSA2 owner compression batch descriptors",
        )
        batch = (batch_starts.data_ptr(), batch_lengths.data_ptr(), batch_cu.data_ptr())
    row = (
        source_base.data_ptr(),
        source_lengths.data_ptr(),
        pages.data_ptr(),
        int(pages.shape[1]),
        int(pages.stride(0)),
        slots.data_ptr(),
        compressed.data_ptr(),
        int(slots.numel()),
        int(ratio),
        int(page_size),
        *batch,
    )
    return row + (0,) * (OWNER_DESC_WIDTH - len(row))


def refresh_owner_slots(
    kv_lens: torch.Tensor,
    lengths: torch.Tensor,
    query_base: torch.Tensor,
    descriptors: torch.Tensor,
    requests: int,
    max_capacity: int,
    *,
    positions: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Device implementation of ``CSA2TrtllmMetadata._refresh_owner_slots`` for
    every owner of the int64 ``[owners, OWNER_DESC_WIDTH]`` ``descriptors``.

    ``positions`` is ``(positions, token_requests, query_offsets)``: int32 output,
    int64 and int32 ``[tokens]`` inputs; the same launch then also refreshes the
    query positions of ``CSA2TrtllmMetadata._refresh_query_positions``.
    """
    ns = _glue_kernels()
    _check(requests <= ns.OwnerSlotsKernel.MAX_REQUESTS, "CSA2 owner refresh request capacity")
    for tensor in (kv_lens, lengths, query_base):
        _check(tensor.dtype == torch.int32 and tensor.is_contiguous(), "CSA2 owner refresh dtypes")
    _check(
        descriptors.dtype == torch.int64
        and descriptors.is_contiguous()
        and descriptors.shape[1:] == (OWNER_DESC_WIDTH,),
        "CSA2 owner refresh descriptors",
    )
    owners = int(descriptors.shape[0])
    if owners == 0:
        return
    out, token_requests, query_offsets = positions or (lengths, descriptors, lengths)
    tokens = 0
    if positions is not None:
        tokens = int(out.numel())
        for tensor, dtype in (
            (out, torch.int32),
            (token_requests, torch.int64),
            (query_offsets, torch.int32),
        ):
            _check(
                tensor.dtype == dtype and tensor.is_contiguous() and tensor.numel() == tokens,
                "CSA2 position refresh dtypes",
            )
    _launch(
        ("owner_slots", positions is not None, descriptors.device.index),
        lambda: ns.OwnerSlotsKernel(positions is not None),
        _ptr(kv_lens),
        _ptr(lengths),
        _ptr(query_base),
        _ptr(descriptors, align=8),
        _ptr(out),
        _ptr(token_requests, align=8),
        _ptr(query_offsets),
        requests,
        owners,
        max_capacity,
        tokens,
        _stream(descriptors.device),
    )


def fill_indexer_descriptors(
    table: torch.Tensor,
    token_requests: torch.Tensor,
    decode_visible: torch.Tensor,
    context_requests: int,
    pages_per_source_page: int,
    blocks: torch.Tensor,
    context: torch.Tensor,
    visible: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Device implementation of ``CSA2TrtllmMetadata._fill_indexer_descriptors``."""
    count = int(blocks.shape[0])
    if count == 0:
        return
    _check(
        table.dtype == torch.int32
        and table.ndim == 2
        and (table.shape[1] <= 1 or table.stride(1) == 1),
        "table",
    )
    token_requests = _as_int64(token_requests)
    decode_visible = _as_int64(decode_visible)
    _check(blocks.dtype == torch.int32 and blocks.stride(1) == 1, "blocks")
    _check(context.dtype == torch.int32 and context.is_contiguous(), "context")
    _check(visible.dtype == torch.int32 and visible.is_contiguous(), "visible out")
    page_capacity = int(blocks.shape[1])
    _check(
        valid.dtype == torch.bool and valid.stride(1) == 1 and valid.shape[1] >= page_capacity,
        "valid",
    )
    ns = _glue_kernels()
    per_cta = ns.IndexerDescriptorsKernel.THREADS * ns.IndexerDescriptorsKernel.PAGES
    _launch(
        ("indexer_descriptors", blocks.device.index, int(pages_per_source_page)),
        lambda: ns.IndexerDescriptorsKernel(int(pages_per_source_page)),
        _ptr(table),
        _ptr(token_requests, align=8),
        _ptr(decode_visible, align=8),
        _ptr(blocks),
        _ptr(context),
        _ptr(visible),
        _ptr(valid, torch.uint8, align=1),
        count,
        int(table.shape[0]),
        int(table.shape[1]),
        int(table.stride(0)) if table.shape[0] > 1 else int(table.shape[1]),
        int(context_requests),
        page_capacity,
        int(blocks.stride(0)),
        int(valid.stride(0)),
        -(-page_capacity // per_cta),
        _stream(blocks.device),
    )


def mask_logits_(
    logits: torch.Tensor,
    valid: torch.Tensor,
    *,
    row_lengths: torch.Tensor,
) -> torch.Tensor:
    """Mask unallocated pages in place among each row's leading ``row_lengths[q]`` columns.

    Reads ``valid[count, >=pages]``, one byte per ``INDEX_PAGE_ROWS``-column page, and
    leaves the columns past a row's length as they are: the decode TopK never reads them.
    """
    count, width = logits.shape
    if count == 0 or width == 0:
        return logits
    _check(logits.dtype == torch.float32 and logits.stride(1) == 1, "logits")
    _check(
        row_lengths.dtype == torch.int32
        and row_lengths.shape == (count,)
        and row_lengths.is_contiguous(),
        "row bounds int32",
    )
    _check(
        valid.dtype == torch.bool
        and valid.stride(1) == 1
        and valid.shape[1] >= -(-width // INDEX_PAGE_ROWS),
        "valid",
    )
    ns = _glue_kernels()
    _launch(
        ("mask_logits", logits.device.index),
        ns.MaskLogitsKernel,
        _ptr(logits),
        _ptr(valid, torch.uint8, align=1),
        _ptr(row_lengths, torch.int32),
        int(count),
        int(width),
        int(logits.stride(0)),
        int(valid.stride(0)),
        _stream(logits.device),
    )
    return logits


def mask_packed_logits_(
    logits: torch.Tensor,
    valid: torch.Tensor,
    row_starts: torch.Tensor,
    row_lengths: torch.Tensor,
) -> torch.Tensor:
    """Mask the columns of packed prefix logits whose gathered key has no page, in place.

    Row ``q`` of ``logits [count, width]`` scores gathered keys ``row_starts[q] +
    column`` for ``column < row_lengths[q]``; ``valid`` is the gathered ``[keys]``
    backing. Columns at or past ``row_lengths[q]`` are left as they are: every
    consumer of packed logits bounds its reads by the row lengths.
    """
    count, width = logits.shape
    keys = valid.numel()
    if count == 0 or width == 0 or keys == 0:
        return logits
    _check(logits.dtype == torch.float32 and logits.stride(1) == 1, "logits")
    _check(valid.dtype == torch.bool and valid.ndim == 1 and valid.is_contiguous(), "valid")
    for bounds in (row_starts, row_lengths):
        _check(
            bounds.dtype == torch.int32 and bounds.shape == (count,) and bounds.is_contiguous(),
            "packed bounds int32",
        )
    ns = _glue_kernels()
    _launch(
        ("mask_packed_logits", logits.device.index),
        ns.MaskPackedLogitsKernel,
        _ptr(logits),
        _ptr(valid, torch.uint8, align=1),
        _ptr(row_starts),
        _ptr(row_lengths),
        int(count),
        int(width),
        int(keys),
        int(logits.stride(0)),
        _stream(logits.device),
    )
    return logits


def gather_candidate_scores(
    logits: torch.Tensor,
    valid: torch.Tensor,
    candidates: torch.Tensor,
    visible: torch.Tensor,
    scores: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Candidate-ordered scores/positions (``-inf``/``-1`` when unbacked or invisible)."""
    count, cwidth = candidates.shape
    width = int(logits.shape[1])
    if count == 0 or cwidth == 0:
        return
    _check(logits.dtype == torch.float32 and logits.stride(1) == 1, "logits")
    _check(
        valid.dtype == torch.bool
        and valid.stride(1) == 1
        and valid.shape[1] >= -(-width // INDEX_PAGE_ROWS),
        "valid",
    )
    _check(candidates.dtype in (torch.int32, torch.int64) and candidates.stride(1) == 1, "cands")
    _check(visible.dtype == torch.int32 and visible.is_contiguous(), "visible int32")
    _check(positions.dtype == torch.int32, "positions int32")
    ns = _glue_kernels()
    int64 = candidates.dtype == torch.int64
    _launch(
        ("candidate_gather", logits.device.index),
        lambda: ns.CandidateGatherKernel(),
        _ptr(logits),
        _ptr(valid, torch.uint8, align=1),
        _ptr(candidates, align=8 if int64 else 4),
        _ptr(visible),
        _ptr(scores),
        _ptr(positions),
        int(count),
        width,
        int(cwidth),
        int(logits.stride(0)),
        int(valid.stride(0)),
        int(candidates.stride(0)),
        _stream(logits.device),
    )


def finalize_selection_(
    indices: torch.Tensor,
    width: int,
    *,
    valid: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    row_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sort validated selections ascending in place with ``-1`` padding.

    Paged mode passes the pages' ``valid`` with the rows' visible ``row_lengths``;
    mapped mode passes ``scores`` and ``positions``; packed mode passes the gathered
    keys' ``valid`` with the rows' bounds, as ``mask_packed_logits_`` takes them.
    """
    count, top_k = indices.shape
    if count == 0 or width == 0:
        return indices
    _check(indices.dtype == torch.int32 and indices.stride(1) == 1, "indices int32")
    mapped = scores is not None
    packed = row_starts is not None
    if mapped:
        _check(positions is not None, "mapped finalize requires positions")
        _check(scores.dtype == torch.float32 and scores.stride(1) == 1, "scores")
        _check(positions.dtype in (torch.int32, torch.int64) and positions.stride(1) == 1, "pos")
        _check(scores.shape[1] >= width and positions.shape[1] >= width, "mapped widths")
    elif packed:
        _check(
            valid is not None
            and valid.dtype == torch.bool
            and valid.ndim == 1
            and valid.is_contiguous(),
            "valid",
        )
        for bounds in (row_starts, row_lengths):
            _check(
                bounds.dtype == torch.int32 and bounds.shape == (count,) and bounds.is_contiguous(),
                "packed bounds int32",
            )
    else:
        _check(valid is not None and valid.dtype == torch.bool and valid.stride(1) == 1, "valid")
        _check(valid.shape[1] >= -(-width // INDEX_PAGE_ROWS), "valid pages")
        _check(
            row_lengths is not None
            and row_lengths.dtype == torch.int32
            and row_lengths.shape == (count,)
            and row_lengths.is_contiguous(),
            "paged lengths int32",
        )
    ns = _glue_kernels()
    if packed and top_k in (128, 256, 512, 1024) and width <= _FINALIZE_BITMAP_COLUMNS:
        _launch(
            ("finalize_packed_sort", int(top_k), indices.device.index),
            lambda: ns.PackedSortFinalizeKernel(int(top_k)),
            _ptr(indices),
            _ptr(valid, torch.uint8, align=1),
            _ptr(row_starts),
            _ptr(row_lengths),
            int(count),
            int(width),
            int(indices.stride(0)),
            _stream(indices.device),
        )
        return indices
    limit = _FINALIZE_BITMAP_COLUMNS if packed else _PAGED_FINALIZE_BITMAP_COLUMNS
    bitmap = not mapped and width <= limit
    int64 = mapped and positions.dtype == torch.int64
    _launch(
        ("finalize", int(top_k), mapped, packed, bitmap, indices.device.index),
        lambda: ns.FinalizeSelectionKernel(int(top_k), mapped, packed, bitmap),
        _ptr(indices),
        _ptr(valid if not mapped else indices, torch.uint8 if not mapped else None, align=1),
        _ptr(scores if mapped else indices),
        _ptr(positions if mapped else indices, align=8 if int64 else 4),
        _ptr(row_starts if packed else indices),
        _ptr(row_lengths if not mapped else indices),
        int(count),
        int(width),
        int(indices.stride(0)),
        int(valid.stride(0)) if not mapped else 0,
        int(scores.stride(0)) if mapped else 0,
        int(positions.stride(0)) if mapped else 0,
        _stream(indices.device),
    )
    return indices


def map_selection_(
    indices: torch.Tensor,
    positions: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Map Top-K candidate-column offsets to logical positions in place (``-1`` when invalid),
    in increasing column order with ``-1`` padding last.

    Each row's valid columns are its leading ``counts[q]`` ones, mapping to increasing
    positions. The order does not depend on the Top-K's output order.
    """
    count, top_k = indices.shape
    width = int(positions.shape[1])
    if count == 0 or top_k == 0:
        return indices
    _check(indices.dtype == torch.int32 and indices.stride(1) == 1, "indices int32")
    _check(positions.dtype == torch.int32 and positions.stride(1) == 1, "positions")
    counts = counts.reshape(-1)
    _check(counts.dtype == torch.int32 and counts.is_contiguous(), "counts")
    ns = _glue_kernels()
    _check(width <= _MAP_BITMAP_COLUMNS, "CSA2 selection map supports up to 65536 columns")
    _launch(
        ("map_selection", int(top_k), indices.device.index),
        lambda: ns.MapSelectionKernel(int(top_k)),
        _ptr(indices),
        _ptr(positions),
        _ptr(counts),
        int(count),
        width,
        int(indices.stride(0)),
        int(positions.stride(0)),
        _stream(indices.device),
    )
    return indices


def gather_prefix_keys_(
    pages: torch.Tensor,
    table: torch.Tensor,
    visible: torch.Tensor,
    descriptors: torch.Tensor,
    page_size: int,
    max_positions: int,
    keys: torch.Tensor,
    scales: torch.Tensor,
    valid: torch.Tensor,
    rows_out: torch.Tensor,
) -> None:
    """Gather a packed prefill group's compressed prefixes and query-row bounds in one launch.

    ``descriptors`` rows are (request, key width, key offset, first query row,
    row in group). Keys land at ``keys/scales/valid`` as exact packed FP4 rows
    (positions without a page are zero and invalid); ``rows_out[5, rows]``
    receives each query row's key start, key end, length, block width and a
    zero row start.
    """
    total_keys, rows = int(keys.shape[0]), int(rows_out.shape[1])
    if total_keys == 0 and rows == 0:
        return
    _check(pages.dtype == torch.uint8 and pages.is_contiguous(), "index pages")
    _check(table.ndim == 2, "table")
    if table.dtype != torch.int32 or (table.shape[1] > 1 and table.stride(1) != 1):
        table = table.to(torch.int32).contiguous()
    _check(visible.dtype in (torch.int32, torch.int64) and visible.is_contiguous(), "visible")
    _check(
        descriptors.dtype == torch.int32
        and descriptors.is_contiguous()
        and descriptors.shape[1:] == (PREFIX_DESC_WIDTH,),
        "prefix descriptors",
    )
    _check(
        keys.dtype == torch.int8 and keys.is_contiguous() and keys.shape[1] == 4 * INDEX_ROW_WORDS,
        "keys",
    )
    _check(scales.dtype == torch.int32 and scales.is_contiguous(), "scales")
    _check(valid.dtype == torch.bool and valid.is_contiguous(), "valid")
    _check(
        rows_out.dtype == torch.int32 and rows_out.is_contiguous() and rows_out.shape[0] == 5,
        "row bounds",
    )
    ns = _glue_kernels()
    _launch(
        ("prefix_gather", pages.device.index),
        lambda: ns.PrefixGatherKernel(),
        _ptr(pages, torch.int32),
        _ptr(table),
        _ptr(visible, align=8 if visible.dtype == torch.int64 else 4),
        _ptr(descriptors),
        _ptr(keys, torch.int32),
        _ptr(scales),
        _ptr(valid, torch.uint8, align=1),
        _ptr(rows_out),
        int(descriptors.shape[0]),
        int(table.shape[0]),
        int(table.shape[1]),
        int(table.stride(0)) if table.shape[0] > 1 else int(table.shape[1]),
        int(page_size),
        int(max_positions),
        pages.numel() // (4 * INDEX_PAGE_WORDS) * INDEX_PAGE_ROWS,
        total_keys,
        rows,
        _stream(pages.device),
    )


def candidate_block_max_(
    logits: torch.Tensor,
    valid: torch.Tensor,
    valid_stride: int,
    visible: torch.Tensor,
    query_start: int,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    row_widths: torch.Tensor,
    blocks: torch.Tensor,
    block_counts: torch.Tensor,
    block_size: int,
    page: int = 1,
) -> None:
    """Candidate block maxima of bounded logits (see ``CandidateBlockMaxKernel``)."""
    count, width = logits.shape
    if count == 0:
        return
    _check(logits.dtype == torch.float32 and logits.stride(1) == 1, "logits")
    _check(valid.dtype == torch.bool and valid.stride(-1) == 1, "valid")
    _check(visible.dtype in (torch.int32, torch.int64) and visible.is_contiguous(), "visible")
    for tensor in (row_starts, row_ends, row_widths, block_counts):
        _check(tensor.dtype == torch.int32 and tensor.is_contiguous(), "int32 rows")
    _check(
        blocks.dtype == torch.float32 and blocks.is_contiguous() and blocks.shape[0] == count,
        "blocks",
    )
    ns = _glue_kernels()
    vector = logits.data_ptr() % 16 == 0 and logits.stride(0) % 4 == 0 and valid.data_ptr() % 4 == 0
    _launch(
        ("candidate_block_max", int(block_size), vector, int(page), logits.device.index),
        lambda: ns.CandidateBlockMaxKernel(int(block_size), vector, int(page)),
        _ptr(logits),
        _ptr(valid, torch.uint8, align=1),
        _ptr(visible, align=8 if visible.dtype == torch.int64 else 4),
        _ptr(row_starts),
        _ptr(row_ends),
        _ptr(row_widths),
        _ptr(blocks),
        _ptr(block_counts),
        int(count),
        int(width),
        int(blocks.shape[1]),
        int(query_start),
        int(logits.stride(0)),
        int(valid_stride),
        _stream(logits.device),
    )


def expand_candidates_(
    selected: torch.Tensor,
    blocks: torch.Tensor,
    visible: torch.Tensor,
    query_start: int,
    row_widths: torch.Tensor,
    output: torch.Tensor,
    block_size: int,
    sparse_ids: torch.Tensor | None = None,
    sparse_counts: torch.Tensor | None = None,
    sparse_block: int = 0,
) -> None:
    """Publish candidates from a block Top-K selection (see ``CandidateExpandKernel``)."""
    count, top_k = selected.shape
    if count == 0:
        return
    published_width = int(output.shape[1])
    _check(selected.dtype == torch.int32 and selected.stride(1) == 1, "selected")
    _check(blocks.dtype == torch.float32 and blocks.is_contiguous(), "blocks")
    _check(visible.dtype in (torch.int32, torch.int64) and visible.is_contiguous(), "visible")
    _check(row_widths.dtype == torch.int32 and row_widths.is_contiguous(), "row widths")
    _check(output.dtype == torch.int32 and output.stride(1) == 1, "CSA2 candidate output")
    if sparse_block:
        _check(
            sparse_ids is not None
            and sparse_ids.dtype == torch.int32
            and sparse_ids.shape == (count, published_width // sparse_block)
            and sparse_ids.stride(1) == 1,
            "CSA2 sparse block ids",
        )
        _check(
            sparse_counts is not None
            and sparse_counts.dtype == torch.int32
            and sparse_counts.numel() == count,
            "CSA2 sparse counts",
        )
    else:
        sparse_ids = sparse_counts = output  # unread placeholders
    bitmap_blocks = max(_CANDIDATE_BITMAP_BLOCKS, 1 << (int(blocks.shape[1]) - 1).bit_length())
    ns = _glue_kernels()
    _launch(
        (
            "candidate_expand",
            int(block_size),
            int(top_k),
            int(sparse_block),
            bitmap_blocks,
            output.device.index,
        ),
        lambda: ns.CandidateExpandKernel(
            int(block_size), int(top_k), int(sparse_block), bitmap_blocks
        ),
        _ptr(selected),
        _ptr(blocks),
        _ptr(visible, align=8 if visible.dtype == torch.int64 else 4),
        _ptr(row_widths),
        _ptr(output),
        _ptr(sparse_ids),
        _ptr(sparse_counts),
        int(count),
        int(blocks.shape[1]),
        int(query_start),
        published_width,
        int(selected.stride(0)),
        int(output.stride(0)),
        int(sparse_ids.stride(0)),
        _stream(output.device),
    )


# =============================================================================
# Packed-cache kernels: publication, gathering and staging
# =============================================================================

_FORMATS = ("main", "index", "swa")


def _format_geometry(cache_format: str, head_dim: int) -> tuple[int, int, int]:
    """Return ``(group, data_bytes, scale_bytes)`` of one packed row."""
    if cache_format not in _FORMATS or head_dim <= 0 or head_dim % 4:
        raise ValueError("Unsupported CSA2 cache format or head dimension")
    group = 16 if cache_format == "main" else 32
    if head_dim % group:
        raise ValueError("CSA2 head dimension must be a multiple of the quantization group")
    data_bytes = head_dim if cache_format == "swa" else head_dim // 2
    return group, data_bytes, head_dim // group


@functools.lru_cache(maxsize=1)
def _cache_kernels():
    base = _glue_kernels()
    cuda, cutlass, cute, llvm, T = base.cuda, base.cutlass, base.cute, base.llvm, base.T
    dsl_user_op, SmemAllocator = base.dsl_user_op, base.SmemAllocator
    Int32, Int64, Float32 = cutlass.Int32, cutlass.Int64, cutlass.Float32
    Uint8, Uint32 = cutlass.Uint8, cutlass.Uint32
    tensor1d, tensor2d, bitcast_u32_f32 = base.tensor1d, base.tensor2d, base.bitcast_u32_f32

    @cute.jit
    def pdl_prologue(pdl: cutlass.Constexpr, trigger: cutlass.Constexpr):
        """Under PDL, wait for the previous grids; with ``trigger``, let the next kernel launch.

        Nothing is read before the wait, so data from any earlier kernel is complete. Only
        kernels whose successor always waits for them in turn trigger early.
        """
        if cutlass.const_expr(pdl):
            base.griddepcontrol_wait()
            if cutlass.const_expr(trigger):
                base.griddepcontrol_launch_dependents()

    # ---------------------------------------------------------------- numerics

    @dsl_user_op
    def bitcast_f32_u32(value, *, loc=None, ip=None):
        return Uint32(llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip)))

    @dsl_user_op
    def mul_rn(a, b, *, loc=None, ip=None):
        """IEEE round-to-nearest multiply that never flushes subnormals."""
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
                "mul.rn.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def fma_rn(a, b, c, *, loc=None, ip=None):
        """``a * b + c`` with one round-to-nearest rounding."""
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [
                    Float32(a).ir_value(loc=loc, ip=ip),
                    Float32(b).ir_value(loc=loc, ip=ip),
                    Float32(c).ir_value(loc=loc, ip=ip),
                ],
                "fma.rn.f32 $0, $1, $2, $3;",
                "=f,f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _f32_pair(source, body, *, loc=None, ip=None):
        out = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32()] * 2),
            [Int32(source).ir_value(loc=loc, ip=ip)],
            body,
            "=f,=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
        )

    @cute.jit
    def e4m3x2_to_f32(source: Int32):
        """Exact FP32 values of the two E4M3 bytes in the low 16 bits (hardware convert)."""
        return _f32_pair(
            source,
            "{\n.reg .b16 s, lo, hi;\n.reg .b32 h;\ncvt.u16.u32 s, $2;\n"
            "cvt.rn.f16x2.e4m3x2 h, s;\nmov.b32 {lo, hi}, h;\n"
            "cvt.f32.f16 $0, lo;\ncvt.f32.f16 $1, hi;\n}",
        )

    @cute.jit
    def e2m1x2_to_f32(source: Int32):
        """Exact FP32 values of the two E2M1 nibbles in the low byte (hardware convert)."""
        return _f32_pair(
            source,
            "{\n.reg .b8 s;\n.reg .b16 lo, hi;\n.reg .b32 h;\ncvt.u8.u32 s, $2;\n"
            "cvt.rn.f16x2.e2m1x2 h, s;\nmov.b32 {lo, hi}, h;\n"
            "cvt.f32.f16 $0, lo;\ncvt.f32.f16 $1, hi;\n}",
        )

    @dsl_user_op
    def f32x2_to_e4m3x2(low, high, *, loc=None, ip=None):
        """Saturating round-to-nearest E4M3 bytes of two FP32 values, ``low`` in the low byte.

        ``cvt.rn.satfinite`` clamps to +-448 and keeps NaN as NaN, as ``f32_to_e4m3_byte`` does.
        """
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [Float32(low).ir_value(loc=loc, ip=ip), Float32(high).ir_value(loc=loc, ip=ip)],
                "{\n.reg .b16 h;\ncvt.rn.satfinite.e4m3x2.f32 h, $2, $1;\ncvt.u32.u16 $0, h;\n}",
                "=r,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def max_nan(a, b, *, loc=None, ip=None):
        """FP32 maximum that is NaN when either input is NaN."""
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
                "max.NaN.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def f32x4_to_e2m1x4(v0, v1, v2, v3, *, loc=None, ip=None):
        """Saturating round-to-nearest-even E2M1 nibbles of four FP32 values, ``v0`` lowest."""
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
                "{\n.reg .b8 a, b;\n.reg .b16 h;\ncvt.rn.satfinite.e2m1x2.f32 a, $2, $1;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b, $4, $3;\nmov.b16 h, {a, b};\ncvt.u32.u16 $0, h;\n}",
                "=r,f,f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _u32_asm(sources, body, constraints, *, loc=None, ip=None):
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [value.ir_value(loc=loc, ip=ip) for value in sources],
                body,
                constraints,
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.jit
    def f16x2_splat(value: Float32):
        """``value`` rounded to F16 in both halves of a word."""
        return _u32_asm([Float32(value)], "cvt.rn.f16x2.f32 $0, $1, $1;", "=r,f")

    @cute.jit
    def e4m3_f16x2(byte: Int32):
        """An E4M3 byte widened exactly to F16 in both halves of a word."""
        return _u32_asm(
            [Int32(byte | (byte << 8))],
            "{\n.reg .b16 s;\ncvt.u16.u32 s, $1;\ncvt.rn.f16x2.e4m3x2 $0, s;\n}",
            "=r,r",
        )

    @cute.jit
    def scaled_e4m3x4(source: Uint32, scale: Uint32):
        """Four E4M3 bytes times an F16 ``scale`` pair, rounded to saturated E4M3 bytes."""
        return _u32_asm(
            [Uint32(source), Uint32(scale)],
            "{\n.reg .b16 a, b, x, y;\n.reg .b32 h, l;\nmov.b32 {a, b}, $1;\n"
            "cvt.rn.f16x2.e4m3x2 l, a;\ncvt.rn.f16x2.e4m3x2 h, b;\n"
            "mul.rn.f16x2 l, l, $2;\nmul.rn.f16x2 h, h, $2;\n"
            "cvt.rn.satfinite.e4m3x2.f16x2 x, l;\ncvt.rn.satfinite.e4m3x2.f16x2 y, h;\n"
            "mov.b32 $0, {x, y};\n}",
            "=r,r,r",
        )

    @cute.jit
    def scaled_e2m1x4(source: Uint32, group_scale: Uint32, scale: Uint32):
        """The four E2M1 nibbles in the low 16 bits times an F16 ``group_scale`` pair,
        then times an F16 ``scale`` pair, rounded to saturated E4M3 bytes."""
        return _u32_asm(
            [Uint32(source), Uint32(group_scale), Uint32(scale)],
            "{\n.reg .b8 a, b;\n.reg .b16 x, y;\n.reg .b32 h, l, t;\n"
            "cvt.u8.u32 a, $1;\nshr.b32 t, $1, 8;\ncvt.u8.u32 b, t;\n"
            "cvt.rn.f16x2.e2m1x2 l, a;\ncvt.rn.f16x2.e2m1x2 h, b;\n"
            "mul.rn.f16x2 l, l, $2;\nmul.rn.f16x2 h, h, $2;\n"
            "mul.rn.f16x2 l, l, $3;\nmul.rn.f16x2 h, h, $3;\n"
            "cvt.rn.satfinite.e4m3x2.f16x2 x, l;\ncvt.rn.satfinite.e4m3x2.f16x2 y, h;\n"
            "mov.b32 $0, {x, y};\n}",
            "=r,r,r,r",
        )

    @dsl_user_op
    def rsqrt_approx(a, *, loc=None, ip=None):
        """``rsqrtf`` as the native RMSNorm kernels compute it."""
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [Float32(a).ir_value(loc=loc, ip=ip)],
                "rsqrt.approx.f32 $0, $1;",
                "=f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def div_rn(a, b, *, loc=None, ip=None):
        """IEEE round-to-nearest division (``tl.div_rn`` / PyTorch semantics)."""
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
                "div.rn.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.jit
    def e4m3_to_f32(value: Int32) -> Float32:
        """Exact E4M3FN decode (0x7F/0xFF are NaN; no infinities)."""
        byte = value & 255
        sign = (byte >> 7) & 1
        exponent = (byte >> 3) & 15
        mantissa = byte & 7
        bits = Uint32(0)
        if exponent == 0:
            bits = bitcast_f32_u32(Float32(mantissa) * Float32(0.001953125))
        elif (exponent == 15) & (mantissa == 7):
            bits = Uint32(0x7FC00000)
        else:
            bits = (Uint32(exponent + 120) << 23) | (Uint32(mantissa) << 20)
        bits = bits | (Uint32(sign) << 31)
        return bitcast_u32_f32(bits)

    @cute.jit
    def ue8m0_to_f32(value: Int32) -> Float32:
        """Exact ``2 ** (byte - 127)``; byte zero is the FP32 subnormal, 255 is inf."""
        byte = value & 255
        bits = Uint32(byte) << 23
        if byte == 0:
            bits = Uint32(0x00400000)
        return bitcast_u32_f32(bits)

    @cute.jit
    def e2m1_to_f32(code: Int32) -> Float32:
        """E2M1 nibble decode preserving the sign of zero."""
        magnitude = code & 7
        value = Float32(magnitude) * Float32(0.5)
        if magnitude >= 4:
            value = (Float32(1.0) + Float32(magnitude % 2) * Float32(0.5)) * bitcast_u32_f32(
                Uint32((magnitude // 2) + 126) << 23
            )
        return bitcast_u32_f32(bitcast_f32_u32(value) ^ (Uint32(code & 8) << 28))

    @cute.jit
    def f32_to_e4m3_byte(value: Float32) -> Int32:
        """Saturating round-to-nearest E4M3 (``cvt.rn.satfinite``); NaN stays NaN (0x7F)."""
        clamped = value
        if clamped > Float32(448.0):
            clamped = Float32(448.0)
        if clamped < Float32(-448.0):
            clamped = Float32(-448.0)
        fragment = cute.make_rmem_tensor((1,), cutlass.Float8E4M3FN)
        fragment[0] = cutlass.Float8E4M3FN(clamped)
        byte = Int32(cute.recast_tensor(fragment, Uint8)[0]) & 255
        if value != value:
            byte = Int32(127)
        return byte

    @cute.jit
    def ceil_log2_bits(value: Float32) -> Int32:
        """``ceil(log2(value))`` for positive normal floats; +inf maps to 128."""
        bits = bitcast_f32_u32(value)
        exponent = Int32((bits >> 23) & 255) - 127
        if (bits & Uint32(0x7FFFFF)) != 0:
            exponent = exponent + 1
        if exponent > 128:
            exponent = Int32(128)
        return exponent

    # --------------------------------------------------- shared row decoding

    @cute.jit
    def decode_channels(
        pool: cute.Pointer,
        row_base: Int64,
        first_channel: Int32,
        head_dim: cutlass.Constexpr,
        cache_format: cutlass.Constexpr,
        valid,
        result,
    ):
        """Decode channels ``first_channel .. +4`` of one packed row into ``result`` (FP32)."""
        bytes_t = tensor1d(pool, Int64(2**62))
        for j in cutlass.range_constexpr(4):
            channel = first_channel + j
            value = Float32(0.0)
            if valid:
                if cutlass.const_expr(cache_format == "swa"):
                    payload = Int32(bytes_t[row_base + Int64(channel)]) & 255
                    scale_byte = (
                        Int32(bytes_t[row_base + Int64(head_dim) + Int64(channel // 32)]) & 255
                    )
                    value = mul_rn(e4m3_to_f32(payload), ue8m0_to_f32(scale_byte))
                else:
                    packed = Int32(bytes_t[row_base + Int64(channel // 2)]) & 255
                    code = (packed >> ((channel % 2) * 4)) & 15
                    if cutlass.const_expr(cache_format == "main"):
                        scale_byte = (
                            Int32(bytes_t[row_base + Int64(head_dim // 2) + Int64(channel // 16)])
                            & 255
                        )
                        value = mul_rn(e2m1_to_f32(code), e4m3_to_f32(scale_byte))
                    else:
                        scale_byte = (
                            Int32(bytes_t[row_base + Int64(head_dim // 2) + Int64(channel // 32)])
                            & 255
                        )
                        value = mul_rn(e2m1_to_f32(code), ue8m0_to_f32(scale_byte))
            result[j] = value

    @dsl_user_op
    def load_row_x16(payload: Int64, scale: Int64, kind: Int32, *, loc=None, ip=None):
        """Predicated loads of sixteen channels of a packed row and their scale byte.

        ``kind`` 1 loads sixteen SWA payload bytes, 2 eight main payload bytes, and
        either also loads the scale byte at ``scale``; any other kind loads nothing
        and returns zeros. Returns four payload words and the scale byte. The loads
        are predicated rather than branched over, so callers can issue several rows'
        loads before the first result is needed.
        """
        out = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32()] * 5),
            [
                Int64(payload).ir_value(loc=loc, ip=ip),
                Int64(scale).ir_value(loc=loc, ip=ip),
                Int32(kind).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .pred s, m, any;\nsetp.eq.s32 s, $7, 1;\nsetp.eq.s32 m, $7, 2;\n"
            "or.pred any, s, m;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\n"
            "mov.b32 $3, 0;\nmov.b32 $4, 0;\n@s ld.global.v4.u32 {$0, $1, $2, $3}, [$5];\n"
            "@m ld.global.v2.u32 {$0, $1}, [$5];\n@any ld.global.u8 $4, [$6];\n}",
            "=r,=r,=r,=r,=r,l,l,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        return [Uint32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(5)]

    @cute.jit
    def decode_channels_x16(words, scale_byte: Int32, cache_format: cutlass.Constexpr, result):
        """Decode sixteen channels loaded by ``load_row_x16`` into ``result``.

        Hardware pair conversions; SWA values are rounded to BF16, and main values
        (E2M1 times an E4M3 scale) are exact in BF16.
        """
        if cutlass.const_expr(cache_format == "swa"):
            factor = ue8m0_to_f32(scale_byte)
            for pair in cutlass.range_constexpr(8):
                a, b = e4m3x2_to_f32(Int32(words[pair // 2] >> (16 * (pair % 2))) & 65535)
                result[2 * pair] = Float32(cutlass.BFloat16(mul_rn(a, factor)))
                result[2 * pair + 1] = Float32(cutlass.BFloat16(mul_rn(b, factor)))
        else:
            factor, _ = e4m3x2_to_f32(scale_byte)
            for pair in cutlass.range_constexpr(8):
                a, b = e2m1x2_to_f32(Int32(words[pair // 4] >> (8 * (pair % 4))) & 255)
                result[2 * pair] = mul_rn(a, factor)
                result[2 * pair + 1] = mul_rn(b, factor)

    @cute.jit
    def store_bf16x16(pointer: cute.Pointer, element_offset: Int64, result):
        """Store sixteen FP32 values as BF16 with two vector stores."""
        fragment = cute.make_rmem_tensor((16,), cutlass.BFloat16)
        for j in cutlass.range_constexpr(16):
            fragment[j] = cutlass.BFloat16(result[j])
        address = pointer.toint() + element_offset * 2
        target = cute.make_ptr(cutlass.BFloat16, address, cute.AddressSpace.gmem, assumed_align=16)
        cute.autovec_copy(fragment, cute.make_tensor(target, cute.make_layout(16)))

    @cute.jit
    def quantize_e4m3x16(result, quant_scale: Float32, words):
        """Sixteen BF16-exact values as scaled, saturated E4M3 bytes in four words."""
        for word in cutlass.range_constexpr(4):
            value = Uint32(0)
            for pair in cutlass.range_constexpr(2):
                low = result[4 * word + 2 * pair] * quant_scale
                high = result[4 * word + 2 * pair + 1] * quant_scale
                value = value | (f32x2_to_e4m3x2(low, high) << (16 * pair))
            words[word] = value

    @cute.jit
    def store_u32x4(pointer: cute.Pointer, byte_offset: Int64, words):
        """Store four words with one 16-byte vector store."""
        address = pointer.toint() + byte_offset
        target = cute.make_ptr(Uint32, address, cute.AddressSpace.gmem, assumed_align=16)
        cute.autovec_copy(words, cute.make_tensor(target, cute.make_layout(4)))

    @cute.jit
    def load_rows_x16(kind: Int32, row_base: Int64, rows: cutlass.Constexpr, kinds, scales, words):
        """Load the lane's sixteen channels of each of the warp's first ``rows`` rows.

        Lane ``r`` holds row ``r``'s ``load_row_x16`` kind and packed row address. A
        lane reads sixteen SWA bytes or eight main bytes, and the one group-scale byte
        they share. Every row's loads are issued before any of them is converted.
        """
        tid, _, _ = cute.arch.thread_idx()
        first = tid % 32 * 16
        for row in cutlass.range_constexpr(rows):
            row_kind = cute.arch.shuffle_sync(kind, Int32(row))
            base = Int64(
                cute.arch.shuffle_sync(Int32(row_base & Int64(0xFFFFFFFF)), Int32(row))
            ) & Int64(0xFFFFFFFF)
            base = base | (Int64(cute.arch.shuffle_sync(Int32(row_base >> 32), Int32(row))) << 32)
            payload = base + Int64(first // 2)
            scale_byte = base + Int64(256 + first // 16)
            if row_kind == 1:
                payload = base + Int64(first)
                scale_byte = base + Int64(512 + first // 32)
            kinds[row] = row_kind
            loaded = load_row_x16(payload, scale_byte, row_kind)
            for word in cutlass.range_constexpr(4):
                words[row][word] = loaded[word]
            scales[row] = Int32(loaded[4])

    @cute.jit
    def staging_quant(quant_scale: cute.Pointer, fp8: cutlass.Constexpr):
        """The FP8 staging scale with its exponent, whether the F16 conversions apply,
        and its F16 pair; zeros for a BF16 bank."""
        quant = Float32(0.0)
        quant_exponent = Int32(0)
        quant_pow2 = cutlass.Boolean(False)
        quant_pair = Uint32(0)
        if cutlass.const_expr(fp8):
            quant = Float32(tensor1d(quant_scale, Int64(1))[0])
            # With a power-of-two ``quant`` inside the F16 range, every product in
            # stage_row_x16 is exact in F16 (or overflows to a saturating infinity), so
            # one F16 multiply chain rounds to E4M3 exactly as the FP32 path does.
            # SWA rows also need their UE8M0 factor in [2**-117, 2**118], where the FP32
            # path's BF16 rounding is the identity.
            bits = bitcast_f32_u32(quant)
            quant_exponent = Int32((bits >> 23) & 255) - 127
            quant_pow2 = (
                ((bits & Uint32(0x807FFFFF)) == Uint32(0))
                & (quant_exponent >= -24)
                & (quant_exponent <= 15)
            )
            quant_pair = f16x2_splat(quant)
        return quant, quant_exponent, quant_pow2, quant_pair

    @cute.jit
    def stage_row_x16(
        kind: Int32,
        scale_byte: Int32,
        words,
        quant_state,
        fp8: cutlass.Constexpr,
        target: cute.Pointer,
        element: Int64,
        result,
        packed,
    ):
        """Convert one row's sixteen loaded channels to the bank format and store them at
        ``element`` of ``target``; ``quant_state`` comes from ``staging_quant``."""
        quant, quant_exponent, quant_pow2, quant_pair = quant_state
        generic = cutlass.Boolean(True)
        if cutlass.const_expr(fp8):
            if kind == 2:
                if quant_pow2:
                    generic = cutlass.Boolean(False)
                    group_scale = e4m3_f16x2(scale_byte)
                    for word in cutlass.range_constexpr(4):
                        packed[word] = scaled_e2m1x4(
                            words[word // 2] >> (16 * (word % 2)), group_scale, quant_pair
                        )
            elif kind == 1:
                exponent = scale_byte - 127 + quant_exponent
                if (
                    quant_pow2
                    & (scale_byte >= 10)
                    & (scale_byte <= 245)
                    & (exponent >= -24)
                    & (exponent <= 15)
                ):
                    generic = cutlass.Boolean(False)
                    scale = f16x2_splat(ue8m0_to_f32(scale_byte) * quant)
                    for word in cutlass.range_constexpr(4):
                        packed[word] = scaled_e4m3x4(words[word], scale)
        if generic:
            for j in cutlass.range_constexpr(16):
                result[j] = Float32(0.0)
            if kind == 1:
                decode_channels_x16(words, scale_byte, "swa", result)
            elif kind == 2:
                decode_channels_x16(words, scale_byte, "main", result)
            if cutlass.const_expr(fp8):
                quantize_e4m3x16(result, quant, packed)
            else:
                store_bf16x16(target, element, result)
        if cutlass.const_expr(fp8):
            store_u32x4(target, element, packed)

    @cute.jit
    def store_bf16x4(pointer: cute.Pointer, element_offset: Int64, result):
        """Store four FP32 values as BF16 with one vector store."""
        fragment = cute.make_rmem_tensor((4,), cutlass.BFloat16)
        for j in cutlass.range_constexpr(4):
            fragment[j] = cutlass.BFloat16(result[j])
        address = pointer.toint() + element_offset * 2
        target = cute.make_ptr(cutlass.BFloat16, address, cute.AddressSpace.gmem, assumed_align=8)
        cute.autovec_copy(fragment, cute.make_tensor(target, cute.make_layout(4)))

    @cute.jit
    def map_logical_slot(
        logical: Int64,
        request: Int64,
        table: cute.Pointer,
        table_rows: Int32,
        table_cols: Int32,
        table_stride: Int64,
        page_size: Int64,
        max_positions: Int64,
        visible: Int64,
    ) -> Int64:
        """``CSA2TrtllmMetadata._map_global_slots`` for one entry (visibility applied)."""
        slot = Int64(-1)
        if (table_rows > 0) & (table_cols > 0) & (logical >= 0):
            # Positions past the admitted bound are invalid below; the rest fit
            # 32-bit division, which is far cheaper than the 64-bit sequence.
            bounded = logical
            if bounded > max_positions:
                bounded = max_positions
            if bounded > Int64(2**31 - 1):
                bounded = Int64(2**31 - 1)
            page_size32 = Int32(page_size)
            page = Int64(Int32(bounded) // page_size32)
            safe_request = request
            if safe_request < 0:
                safe_request = Int64(0)
            if safe_request > Int64(table_rows) - 1:
                safe_request = Int64(table_rows) - 1
            safe_page = page
            if safe_page > Int64(table_cols) - 1:
                safe_page = Int64(table_cols) - 1
            table_t = tensor2d(table, Int64(table_rows), Int64(table_cols), table_stride)
            physical = Int64(table_t[Int32(safe_request), Int32(safe_page)])
            valid = (logical < max_positions) & (page < Int64(table_cols))
            valid = valid & (request >= 0) & (request < Int64(table_rows)) & (physical >= 0)
            valid = valid & (logical < visible)
            if valid:
                slot = physical * page_size + Int64(Int32(bounded) % page_size32)
        return slot

    @cute.jit
    def highest_scale_byte(bitmap: cute.Pointer, index: Int32) -> Int32:
        """Largest byte value recorded in source ``index``'s 8-word bitmap, or -1.

        The staging-scale reductions record which group-scale byte values occur
        rather than their maximum, so that their CTAs combine with ``atomic_or``. One
        bit per value makes the recovery exact: the highest set bit is the maximum.
        """
        largest = Int32(-1)
        for word in cutlass.range_constexpr(8):
            bits = Uint32(tensor1d(bitmap, Int64(16))[Int64(index) * 8 + word])
            if bits != Uint32(0):
                largest = Int32(word * 32) + Int32(_u32_asm([bits], "bfind.u32 $0, $1;", "=r,r"))
        return largest

    @cute.jit
    def staged_exponent(bitmap: cute.Pointer, cached: cute.Pointer | None = None) -> Int32:
        """Smallest power-of-two exponent mapping the bitmaps' decode ceiling onto 448, or 0.

        Both encoders clamp their payload -- ``main`` codes at the E2M1 maximum of
        6, ``swa`` values at +-448 -- so ``6 * e4m3(byte)`` and ``448 * 2**(byte -
        127)`` are exact ceilings on what a row can decode to, whichever way the
        encoder rounded its group scale. Bitmaps without a byte keep the unit scale.
        ``cached`` is a staging group's recorded main bitmap, which
        stands in for main rows that were not reduced.
        """
        # Sentinel exponent for "no group scale was observed", below every real one.
        absent = -1000
        swa_byte = highest_scale_byte(bitmap, Int32(0))
        main_byte = highest_scale_byte(bitmap, Int32(1))
        if cutlass.const_expr(cached is not None):
            cached_byte = highest_scale_byte(cached, Int32(0))
            if cached_byte > main_byte:
                main_byte = cached_byte
        # 448 * 2**k >= 448 * 2**(byte - 127) holds exactly when k >= byte - 127.
        swa_exponent = Int32(absent)
        if swa_byte >= 0:
            swa_exponent = swa_byte - 127
        main_exponent = Int32(absent)
        if main_byte >= 0:
            # 6 * e4m3(byte) is exact in FP32: it never needs more than
            # seven significant bits. ``e4m3_to_f32`` decodes without the
            # approximate ex2 instruction, keeping the power of two below exact.
            ceiling = Float32(6.0) * e4m3_to_f32(main_byte)
            if ceiling > Float32(0.0):
                # With ceiling = m * 2**e (m in [1, 2)) and 448 = 1.75 * 2**8,
                # the smallest k with 448 * 2**k >= ceiling is e - 8, plus one
                # when m exceeds 1.75.
                bits = bitcast_f32_u32(ceiling)
                main_exponent = Int32(bits >> 23) - 135
                if (bits & Uint32(0x7FFFFF)) > Uint32(0x600000):
                    main_exponent = main_exponent + 1
        exponent = swa_exponent
        if main_exponent > exponent:
            exponent = main_exponent
        # Staging nothing at all keeps the unit scale.
        if exponent == Int32(absent):
            exponent = Int32(0)
        return exponent

    @cute.jit
    def warp_max(value: Int32) -> Int32:
        offset = 1
        while offset < 32:
            other = cute.arch.shuffle_sync_bfly(value, offset=offset)
            if other > value:
                value = other
            offset = offset * 2
        return value

    @cute.jit
    def q_amax_bits(
        words: cute.Pointer, first: Int32, chunks: Int32, step: Int32, batch: cutlass.Constexpr
    ) -> Int32:
        """Largest sign-cleared BF16 half in 16-byte chunks ``first, first + step, ...``, or -1.

        Each round issues ``batch`` independent chunk loads before reducing any of them.
        """
        largest = Int32(-1)
        fragments = [cute.make_rmem_tensor((4,), Int32) for _ in range(batch)]
        for base in cutlass.range(first, chunks, step * batch):
            for k in cutlass.range_constexpr(batch):
                chunk = base + k * step
                if chunk < chunks:
                    address = words.toint() + Int64(chunk) * 16
                    source_t = cute.make_tensor(
                        cute.make_ptr(Int32, address, cute.AddressSpace.gmem, assumed_align=16),
                        cute.make_layout(4),
                    )
                    cute.autovec_copy(source_t, fragments[k])
                else:
                    fragments[k].fill(0)
            for k in cutlass.range_constexpr(batch):
                for j in cutlass.range_constexpr(4):
                    low = fragments[k][j] & 0x7FFF
                    high = (fragments[k][j] >> 16) & 0x7FFF
                    if low > largest:
                        largest = low
                    if high > largest:
                        largest = high
        return largest

    @cute.jit
    def largest_scale_byte(
        pool: cute.Pointer, offset: Int64, words: cutlass.Constexpr, limit: cutlass.Constexpr
    ) -> Int32:
        """Largest finite group-scale byte in the ``words`` u32 words at byte ``offset``, or -1.

        ``limit`` drops the non-finite encodings: 0x7F is E4M3 NaN and also rejects
        every negative E4M3 byte, and byte 255 is the ``swa`` infinity. Neither can be
        bounded by any scale, so they must not drive the reduction.
        """
        largest = Int32(-1)
        for word in cutlass.range_constexpr(words):
            packed = Int32(
                cute.make_tensor(
                    cute.make_ptr(
                        Uint32,
                        pool.toint() + offset + Int64(4 * word),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    cute.make_layout(1),
                )[0]
            )
            for j in cutlass.range_constexpr(4):
                value = (packed >> (8 * j)) & 255
                if (value < Int32(limit)) & (value > largest):
                    largest = value
        return largest

    @cute.jit
    def store_bmm_scales(
        q_dequant_scale: cute.Pointer,
        dequant_scale: cute.Pointer,
        bmm1_scale: cute.Pointer,
        bmm2_scale: cute.Pointer,
        softmax_scale: Float32,
    ):
        """Q and KV may carry different staging scales, so BMM1's dequant is their product."""
        dq_q = Float32(tensor1d(q_dequant_scale, Int64(1))[0])
        dq = Float32(tensor1d(dequant_scale, Int64(1))[0])
        bmm1 = dq_q * dq * softmax_scale
        bmm1_t = tensor1d(bmm1_scale, Int64(2))
        bmm1_t[0] = bmm1
        bmm1_t[1] = bmm1 * Float32(1.4426950408889634)
        tensor1d(bmm2_scale, Int64(1))[0] = dq

    # ---------------------------------------------------- K7: quantize rows

    class QuantizeScatterKernel:
        """Quantize BF16 rows and scatter them into packed cache rows, up to four jobs per launch.

        Each job (``QuantizeJobSpec``) writes in mode ``row`` (``pool[slot, :]``
        with a row stride), ``footer`` (native page-footer pages of ``page_rows``
        rows) or ``split`` (data and scale bytes in two contiguous buffers); with
        ``norm`` the row is RMS-normalized first (``rsqrt`` and one BF16 rounding
        as the native RMSNorm kernel) and with ``rope`` the last ``rope_dim``
        channels are rotated in interleaved GPT-J pairs (one BF16 rounding as
        ``mla_rope_inplace``). A CTA quantizes ``threads // job.threads`` rows of
        one job (one row of a full-width job, one row per warp of a 128D job)
        and the grid concatenates the jobs' CTAs, so a layer's SWA, main, index
        and index-query writes are one launch. Each thread owns four channels;
        quantization groups reduce with butterfly shuffles.
        """

        def __init__(self, jobs, pdl: bool = False):
            if not 1 <= len(jobs) <= _MAX_QUANTIZE_JOBS:
                raise ValueError("A CSA2 quantize launch takes one to four jobs")
            self.jobs = tuple(jobs)
            self.pdl = pdl
            self.threads = max(job.threads for job in self.jobs)
            self.rows_per_cta = tuple(
                self.threads // self.jobs[i].threads if i < len(self.jobs) else 1
                for i in range(_MAX_QUANTIZE_JOBS)
            )

        @cute.jit
        def __call__(
            self,
            pool0: cute.Pointer,
            scale0: cute.Pointer,
            slots0: cute.Pointer,
            values0: cute.Pointer,
            weight0: cute.Pointer,
            positions0: cute.Pointer,
            cos_sin0: cute.Pointer,
            rows0: Int32,
            capacity0: Int64,
            pool_stride0: Int64,
            slot_stride0: Int64,
            value_stride0: Int64,
            page_rows0: Int64,
            eps0: Float32,
            pool1: cute.Pointer,
            scale1: cute.Pointer,
            slots1: cute.Pointer,
            values1: cute.Pointer,
            weight1: cute.Pointer,
            positions1: cute.Pointer,
            cos_sin1: cute.Pointer,
            rows1: Int32,
            capacity1: Int64,
            pool_stride1: Int64,
            slot_stride1: Int64,
            value_stride1: Int64,
            page_rows1: Int64,
            eps1: Float32,
            pool2: cute.Pointer,
            scale2: cute.Pointer,
            slots2: cute.Pointer,
            values2: cute.Pointer,
            weight2: cute.Pointer,
            positions2: cute.Pointer,
            cos_sin2: cute.Pointer,
            rows2: Int32,
            capacity2: Int64,
            pool_stride2: Int64,
            slot_stride2: Int64,
            value_stride2: Int64,
            page_rows2: Int64,
            eps2: Float32,
            pool3: cute.Pointer,
            scale3: cute.Pointer,
            slots3: cute.Pointer,
            values3: cute.Pointer,
            weight3: cute.Pointer,
            positions3: cute.Pointer,
            cos_sin3: cute.Pointer,
            rows3: Int32,
            capacity3: Int64,
            pool_stride3: Int64,
            slot_stride3: Int64,
            value_stride3: Int64,
            page_rows3: Int64,
            eps3: Float32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                pool0,
                scale0,
                slots0,
                values0,
                weight0,
                positions0,
                cos_sin0,
                rows0,
                capacity0,
                pool_stride0,
                slot_stride0,
                value_stride0,
                page_rows0,
                eps0,
                pool1,
                scale1,
                slots1,
                values1,
                weight1,
                positions1,
                cos_sin1,
                rows1,
                capacity1,
                pool_stride1,
                slot_stride1,
                value_stride1,
                page_rows1,
                eps1,
                pool2,
                scale2,
                slots2,
                values2,
                weight2,
                positions2,
                cos_sin2,
                rows2,
                capacity2,
                pool_stride2,
                slot_stride2,
                value_stride2,
                page_rows2,
                eps2,
                pool3,
                scale3,
                slots3,
                values3,
                weight3,
                positions3,
                cos_sin3,
                rows3,
                capacity3,
                pool_stride3,
                slot_stride3,
                value_stride3,
                page_rows3,
                eps3,
            ).launch(
                grid=[self._ctas(rows0, rows1, rows2, rows3), 1, 1],
                block=[self.threads, 1, 1],
                stream=stream,
                use_pdl=self.pdl,
            )

        @cute.kernel
        def kernel(
            self,
            pool0: cute.Pointer,
            scale0: cute.Pointer,
            slots0: cute.Pointer,
            values0: cute.Pointer,
            weight0: cute.Pointer,
            positions0: cute.Pointer,
            cos_sin0: cute.Pointer,
            rows0: Int32,
            capacity0: Int64,
            pool_stride0: Int64,
            slot_stride0: Int64,
            value_stride0: Int64,
            page_rows0: Int64,
            eps0: Float32,
            pool1: cute.Pointer,
            scale1: cute.Pointer,
            slots1: cute.Pointer,
            values1: cute.Pointer,
            weight1: cute.Pointer,
            positions1: cute.Pointer,
            cos_sin1: cute.Pointer,
            rows1: Int32,
            capacity1: Int64,
            pool_stride1: Int64,
            slot_stride1: Int64,
            value_stride1: Int64,
            page_rows1: Int64,
            eps1: Float32,
            pool2: cute.Pointer,
            scale2: cute.Pointer,
            slots2: cute.Pointer,
            values2: cute.Pointer,
            weight2: cute.Pointer,
            positions2: cute.Pointer,
            cos_sin2: cute.Pointer,
            rows2: Int32,
            capacity2: Int64,
            pool_stride2: Int64,
            slot_stride2: Int64,
            value_stride2: Int64,
            page_rows2: Int64,
            eps2: Float32,
            pool3: cute.Pointer,
            scale3: cute.Pointer,
            slots3: cute.Pointer,
            values3: cute.Pointer,
            weight3: cute.Pointer,
            positions3: cute.Pointer,
            cos_sin3: cute.Pointer,
            rows3: Int32,
            capacity3: Int64,
            pool_stride3: Int64,
            slot_stride3: Int64,
            value_stride3: Int64,
            page_rows3: Int64,
            eps3: Float32,
        ):
            cta, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            pdl_prologue(self.pdl, False)
            smem = SmemAllocator()
            partial = smem.allocate_tensor(
                Float32, cute.make_layout(max(self.threads // 32, 1)), 16
            )
            per = self.rows_per_cta
            ctas0 = (rows0 + per[0] - 1) // per[0]
            if cta < ctas0:
                self._job_row(
                    0,
                    cta,
                    tid,
                    pool0,
                    scale0,
                    slots0,
                    values0,
                    weight0,
                    positions0,
                    cos_sin0,
                    rows0,
                    capacity0,
                    pool_stride0,
                    slot_stride0,
                    value_stride0,
                    page_rows0,
                    eps0,
                    partial,
                )
            else:
                if cutlass.const_expr(len(self.jobs) > 1):
                    cta1 = cta - ctas0
                    ctas1 = (rows1 + per[1] - 1) // per[1]
                    if cta1 < ctas1:
                        self._job_row(
                            1,
                            cta1,
                            tid,
                            pool1,
                            scale1,
                            slots1,
                            values1,
                            weight1,
                            positions1,
                            cos_sin1,
                            rows1,
                            capacity1,
                            pool_stride1,
                            slot_stride1,
                            value_stride1,
                            page_rows1,
                            eps1,
                            partial,
                        )
                    else:
                        if cutlass.const_expr(len(self.jobs) > 2):
                            cta2 = cta1 - ctas1
                            ctas2 = (rows2 + per[2] - 1) // per[2]
                            if cta2 < ctas2:
                                self._job_row(
                                    2,
                                    cta2,
                                    tid,
                                    pool2,
                                    scale2,
                                    slots2,
                                    values2,
                                    weight2,
                                    positions2,
                                    cos_sin2,
                                    rows2,
                                    capacity2,
                                    pool_stride2,
                                    slot_stride2,
                                    value_stride2,
                                    page_rows2,
                                    eps2,
                                    partial,
                                )
                            else:
                                if cutlass.const_expr(len(self.jobs) > 3):
                                    self._job_row(
                                        3,
                                        cta2 - ctas2,
                                        tid,
                                        pool3,
                                        scale3,
                                        slots3,
                                        values3,
                                        weight3,
                                        positions3,
                                        cos_sin3,
                                        rows3,
                                        capacity3,
                                        pool_stride3,
                                        slot_stride3,
                                        value_stride3,
                                        page_rows3,
                                        eps3,
                                        partial,
                                    )

        def _ctas(self, rows0, rows1, rows2, rows3):
            per = self.rows_per_cta
            return (
                (rows0 + per[0] - 1) // per[0]
                + (rows1 + per[1] - 1) // per[1]
                + (rows2 + per[2] - 1) // per[2]
                + (rows3 + per[3] - 1) // per[3]
            )

        @cute.jit
        def _job_row(
            self,
            job_index: cutlass.Constexpr,
            cta,
            tid,
            pool,
            scale_pool,
            slots,
            values,
            norm_weight,
            positions,
            cos_sin,
            rows,
            capacity,
            pool_stride,
            slot_stride,
            value_stride,
            page_rows,
            eps,
            partial,
        ):
            """Row ``cta * rows_per_cta + tid // job.threads`` of one job, on lane ``tid % job.threads``."""
            lanes = self.jobs[job_index].threads
            row = cta * self.rows_per_cta[job_index] + tid // lanes
            if row < rows:
                self._quantize_row(
                    job_index,
                    pool,
                    scale_pool,
                    slots,
                    values,
                    norm_weight,
                    positions,
                    cos_sin,
                    row,
                    tid % lanes,
                    capacity,
                    pool_stride,
                    slot_stride,
                    value_stride,
                    page_rows,
                    eps,
                    partial,
                )

        @cute.jit
        def _quantize_row(
            self,
            job_index: cutlass.Constexpr,
            pool: cute.Pointer,  # uint8 data destination
            scale_pool: cute.Pointer,  # uint8 scale destination (== pool unless split)
            slots: cute.Pointer,  # int64 [rows] strided
            values: cute.Pointer,  # bf16 [rows, head_dim] strided
            norm_weight: cute.Pointer,  # bf16 [head_dim] (norm)
            positions: cute.Pointer,  # int32 [rows] (rope)
            cos_sin: cute.Pointer,  # float32 [positions, rope_dim]: cos half then sin half
            row: Int32,
            tid: Int32,
            capacity: Int64,
            pool_stride: Int64,
            slot_stride: Int64,
            value_stride: Int64,
            page_rows: Int64,
            eps: Float32,
            partial,
        ):
            job = self.jobs[job_index]
            slot = Int64(row)
            if cutlass.const_expr(not job.identity):
                slot = Int64(tensor1d(slots, Int64(2**62))[Int64(row) * slot_stride])
            if (slot >= 0) & (slot < capacity):
                first = tid * 4
                active = first < job.head_dim
                value_t = tensor1d(values, Int64(2**62))
                x = cute.make_rmem_tensor((4,), Float32)
                for j in cutlass.range_constexpr(4):
                    v = Float32(0.0)
                    if active:
                        v = Float32(value_t[Int64(row) * value_stride + Int64(first + j)])
                    x[j] = v
                if cutlass.const_expr(job.norm):
                    # RMSNorm over the row: ``x * rsqrt(mean(x^2) + eps) * w``, one BF16 rounding.
                    sumsq = Float32(0.0)
                    for j in cutlass.range_constexpr(4):
                        sumsq = sumsq + x[j] * x[j]
                    offset = 1
                    while offset < 32:
                        sumsq = sumsq + cute.arch.shuffle_sync_bfly(sumsq, offset=offset)
                        offset = offset * 2
                    # A row narrower than the CTA is one warp: no CTA barrier.
                    if cutlass.const_expr(job.threads > 32):
                        if tid % 32 == 0:
                            partial[tid // 32] = sumsq
                        cute.arch.barrier()
                        sumsq = Float32(0.0)
                        for warp in cutlass.range_constexpr(job.threads // 32):
                            sumsq = sumsq + partial[warp]
                    inv = rsqrt_approx(sumsq / Float32(job.head_dim) + eps)
                    weight_t = tensor1d(norm_weight, Int64(job.head_dim))
                    for j in cutlass.range_constexpr(4):
                        if active:
                            w = Float32(weight_t[first + j])
                            x[j] = Float32(cutlass.BFloat16(x[j] * inv * w))
                if cutlass.const_expr(job.rope_dim > 0):
                    # Interleaved GPT-J RoPE on the trailing channels: pair
                    # ``p`` of the row rotates with cos/sin ``p`` of its position,
                    # rounded as mla_rope_inplace's contracted products.
                    nope = job.head_dim - job.rope_dim
                    # ``active`` keeps the spare threads of a wider job's
                    # block from reading past the end of the table.
                    if active & (first >= nope):
                        pos = Int64(tensor1d(positions, Int64(2**62))[Int64(row)])
                        table = tensor1d(cos_sin, Int64(2**62))
                        base = pos * Int64(job.rope_dim)
                        pair = Int64(first - nope) // 2
                        for k in cutlass.range_constexpr(2):
                            c = Float32(table[base + pair + Int64(k)])
                            sn = Float32(table[base + Int64(job.rope_dim // 2) + pair + Int64(k)])
                            a = x[2 * k]
                            b = x[2 * k + 1]
                            x[2 * k] = Float32(cutlass.BFloat16(fma_rn(c, a, -mul_rn(sn, b))))
                            x[2 * k + 1] = Float32(cutlass.BFloat16(fma_rn(sn, a, mul_rn(c, b))))
                local_max = Float32(0.0)
                for j in cutlass.range_constexpr(4):
                    v = x[j]
                    local_max = max_nan(local_max, cute.arch.fmax(v, -v))
                offset = 1
                while offset < job.group_lanes:
                    local_max = max_nan(
                        local_max, cute.arch.shuffle_sync_bfly(local_max, offset=offset)
                    )
                    offset = offset * 2
                group_nan = local_max != local_max
                # Destination addressing.
                data_base = Int64(0)
                scale_base = Int64(0)
                if cutlass.const_expr(job.mode == "row"):
                    data_base = slot * pool_stride
                    scale_base = data_base + Int64(job.data_bytes)
                elif cutlass.const_expr(job.mode == "footer"):
                    page = slot // page_rows
                    position = slot % page_rows
                    base = page * page_rows * Int64(job.data_bytes + job.scale_bytes)
                    data_base = base + position * Int64(job.data_bytes)
                    scale_base = (
                        base + page_rows * Int64(job.data_bytes) + position * Int64(job.scale_bytes)
                    )
                else:
                    data_base = slot * Int64(job.data_bytes)
                    scale_base = slot * Int64(job.scale_bytes)
                pool_t = tensor1d(pool, Int64(2**62))
                scale_t = tensor1d(scale_pool, Int64(2**62))
                scale_byte = Int32(0)
                scale = Float32(1.0)
                # Power-of-two scales divide exactly as a multiply by their reciprocal.
                inv_scale = Float32(1.0)
                if cutlass.const_expr(job.cache_format == "main"):
                    pre = local_max
                    if pre < Float32(6.0 * 2.0**-9):
                        pre = Float32(6.0 * 2.0**-9)
                    scale_byte = f32_to_e4m3_byte(div_rn(pre, Float32(6.0)))
                    scale = e4m3_to_f32(scale_byte)
                    if group_nan:
                        scale_byte = Int32(127)
                else:
                    floor = Float32(6.0 * 2.0**-126)
                    divisor = Float32(6.0)
                    if cutlass.const_expr(job.cache_format == "swa"):
                        floor = Float32(1.0e-4)
                        divisor = Float32(448.0)
                    clamped = local_max
                    if clamped < floor:
                        clamped = floor
                    exponent = ceil_log2_bits(div_rn(clamped, divisor))
                    scale_byte = (exponent + 127) & 255
                    if exponent > 128:
                        scale_byte = Int32(255)
                    scale = ue8m0_to_f32(exponent + 127)
                    inv_scale = ue8m0_to_f32((127 - exponent) & 255)
                    if exponent >= 128:
                        inv_scale = Float32(0.0)
                    if group_nan:
                        scale_byte = Int32(0)
                if active:
                    scaled = cute.make_rmem_tensor((4,), Float32)
                    for j in cutlass.range_constexpr(4):
                        if cutlass.const_expr(job.cache_format == "main"):
                            scaled[j] = div_rn(x[j], scale)
                        else:
                            scaled[j] = mul_rn(x[j], inv_scale)
                    if cutlass.const_expr(job.cache_format == "swa"):
                        word = f32x2_to_e4m3x2(scaled[0], scaled[1]) | (
                            f32x2_to_e4m3x2(scaled[2], scaled[3]) << Uint32(16)
                        )
                        if group_nan:
                            word = Uint32(0x7F7F7F7F)
                        for j in cutlass.range_constexpr(4):
                            pool_t[data_base + Int64(first + j)] = Uint8(word >> Uint32(8 * j))
                    else:
                        nibbles = f32x4_to_e2m1x4(scaled[0], scaled[1], scaled[2], scaled[3])
                        if group_nan:
                            nibbles = Uint32(0x7777)
                        pool_t[data_base + Int64(first // 2)] = Uint8(nibbles)
                        pool_t[data_base + Int64(first // 2) + 1] = Uint8(nibbles >> Uint32(8))
                    if tid % job.group_lanes == 0:
                        scale_t[scale_base + Int64(first // job.group)] = Uint8(scale_byte)

    # ----------------------------------------------------- K8: gather rows

    class GatherDequantKernel:
        """Gather packed rows by slot and decode them into BF16 (``gather_dequant_rows``)."""

        def __init__(self, cache_format: str, head_dim: int):
            _format_geometry(cache_format, head_dim)
            self.cache_format = cache_format
            self.head_dim = head_dim
            self.threads = max(32, -(-(head_dim // 4) // 32) * 32)

        @cute.jit
        def __call__(
            self,
            pool: cute.Pointer,
            slots: cute.Pointer,
            output: cute.Pointer,
            rows: Int32,
            capacity: Int64,
            pool_stride: Int64,
            slot_columns: Int64,
            slot_row_stride: Int64,
            slot_col_stride: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                pool,
                slots,
                output,
                rows,
                capacity,
                pool_stride,
                slot_columns,
                slot_row_stride,
                slot_col_stride,
            ).launch(grid=[rows, 1, 1], block=[self.threads, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            pool: cute.Pointer,
            slots: cute.Pointer,
            output: cute.Pointer,
            rows: Int32,
            capacity: Int64,
            pool_stride: Int64,
            slot_columns: Int64,
            slot_row_stride: Int64,
            slot_col_stride: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            index = Int64(row)
            slot_offset = (index // slot_columns) * slot_row_stride + (
                index % slot_columns
            ) * slot_col_stride
            slot = Int64(tensor1d(slots, Int64(2**62))[slot_offset])
            valid = (slot >= 0) & (slot < capacity)
            first = tid * 4
            if first < self.head_dim:
                result = cute.make_rmem_tensor((4,), Float32)
                decode_channels(
                    pool, slot * pool_stride, first, self.head_dim, self.cache_format, valid, result
                )
                store_bf16x4(output, index * Int64(self.head_dim) + Int64(first), result)

    # ------------------------------------------------- K9: byte-row scatter

    class ScatterPackedKernel:
        """Copy already packed rows into row-strided or page-footer destinations."""

        THREADS = 128

        def __init__(self, footer: bool, data_bytes: int, scale_bytes: int):
            self.footer = footer
            self.data_bytes = data_bytes
            self.scale_bytes = scale_bytes

        @cute.jit
        def __call__(
            self,
            pool: cute.Pointer,
            slots: cute.Pointer,
            rows: cute.Pointer,
            count: Int32,
            capacity: Int64,
            pool_stride: Int64,
            slot_stride: Int64,
            row_stride: Int64,
            page_rows: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                pool, slots, rows, count, capacity, pool_stride, slot_stride, row_stride, page_rows
            ).launch(grid=[count, 1, 1], block=[self.THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            pool: cute.Pointer,
            slots: cute.Pointer,
            rows: cute.Pointer,
            count: Int32,
            capacity: Int64,
            pool_stride: Int64,
            slot_stride: Int64,
            row_stride: Int64,
            page_rows: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            slot = Int64(tensor1d(slots, Int64(2**62))[Int64(row) * slot_stride])
            if (slot >= 0) & (slot < capacity):
                pool_t = tensor1d(pool, Int64(2**62))
                rows_t = tensor1d(rows, Int64(2**62))
                source = Int64(row) * row_stride
                width = self.data_bytes + self.scale_bytes
                if cutlass.const_expr(self.footer):
                    page = slot // page_rows
                    position = slot % page_rows
                    base = page * page_rows * Int64(width)
                    data_base = base + position * Int64(self.data_bytes)
                    scale_base = (
                        base
                        + page_rows * Int64(self.data_bytes)
                        + position * Int64(self.scale_bytes)
                    )
                    for byte in cutlass.range(tid, width, self.THREADS):
                        value = Uint8(rows_t[source + Int64(byte)])
                        if byte < self.data_bytes:
                            pool_t[data_base + Int64(byte)] = value
                        else:
                            pool_t[scale_base + Int64(byte - self.data_bytes)] = value
                else:
                    base = slot * pool_stride
                    for byte in cutlass.range(tid, width, self.THREADS):
                        pool_t[base + Int64(byte)] = Uint8(rows_t[source + Int64(byte)])

    # ------------------------------------- K10/K11: selected-row staging

    class StageCompactKernel:
        """Compact one query's selected SWA/main rows into dense staging positions.

        Main selections arrive as logical positions and are mapped through the
        owner's page table here. Outputs: ``tags`` (source column per
        destination), ``row_slots`` (its slot), ``counts`` and the mapped main
        slots. Layers sharing a KV owner and an index source share one
        compaction per step.
        """

        THREADS = 256

        def __init__(self, fp8: bool, has_main: bool, premapped: bool):
            self.fp8 = fp8
            self.has_main = has_main
            self.premapped = premapped

        @cute.jit
        def __call__(
            self,
            swa_slots: cute.Pointer,  # int64 [count, swa_width] strided
            main_logical: cute.Pointer,  # int32/int64 [count, main_width] strided
            main_slots: cute.Pointer,  # int64 [count, main_width]: in (premapped) or out (mapped)
            requests: cute.Pointer,  # int64 [count]
            visible: cute.Pointer,  # int64 [count]
            table: cute.Pointer,  # int32 [table_rows, table_cols] strided
            tags: cute.Pointer,  # int32 [count, capacity]
            row_slots: cute.Pointer,  # int32 [count, capacity]
            counts: cute.Pointer,  # int32 [count]
            count: Int32,
            swa_width: Int32,
            main_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            main_rs: Int64,
            main_cs: Int64,
            slot_rs: Int64,
            slot_cs: Int64,
            swa_capacity: Int64,
            main_capacity: Int64,
            table_rows: Int32,
            table_cols: Int32,
            table_stride: Int64,
            page_size: Int64,
            max_positions: Int64,
            capacity: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                swa_slots,
                main_logical,
                main_slots,
                requests,
                visible,
                table,
                tags,
                row_slots,
                counts,
                count,
                swa_width,
                main_width,
                swa_rs,
                swa_cs,
                main_rs,
                main_cs,
                slot_rs,
                slot_cs,
                swa_capacity,
                main_capacity,
                table_rows,
                table_cols,
                table_stride,
                page_size,
                max_positions,
                capacity,
            ).launch(grid=[count, 1, 1], block=[self.THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            swa_slots: cute.Pointer,
            main_logical: cute.Pointer,
            main_slots: cute.Pointer,
            requests: cute.Pointer,
            visible: cute.Pointer,
            table: cute.Pointer,
            tags: cute.Pointer,
            row_slots: cute.Pointer,
            counts: cute.Pointer,
            count: Int32,
            swa_width: Int32,
            main_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            main_rs: Int64,
            main_cs: Int64,
            slot_rs: Int64,
            slot_cs: Int64,
            swa_capacity: Int64,
            main_capacity: Int64,
            table_rows: Int32,
            table_cols: Int32,
            table_stride: Int64,
            page_size: Int64,
            max_positions: Int64,
            capacity: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, _, _ = cute.arch.block_idx()
            smem = SmemAllocator()
            prefix = smem.allocate_tensor(Int32, cute.make_layout(self.THREADS), 16)
            chunk = (capacity + self.THREADS - 1) // self.THREADS
            begin = tid * chunk
            end = begin + chunk
            if end > capacity:
                end = capacity
            swa_t = tensor1d(swa_slots, Int64(2**62))
            request = Int64(0)
            visible_length = Int64(0)
            if cutlass.const_expr(self.has_main):
                request = Int64(tensor1d(requests, Int64(count))[q])
                visible_length = Int64(tensor1d(visible, Int64(count))[q])
            main_slot_t = tensor1d(main_slots, Int64(2**62))
            # Pass 1: validity per column (main columns are mapped once and stored).
            local = Int32(0)
            for column in cutlass.range(begin, end, 1):
                valid = cutlass.Boolean(False)
                if column < swa_width:
                    slot = Int64(swa_t[Int64(q) * swa_rs + Int64(column) * swa_cs])
                    valid = (slot >= 0) & (slot < swa_capacity)
                elif column < swa_width + main_width:
                    if cutlass.const_expr(self.has_main):
                        main_column = column - swa_width
                        slot_index = Int64(q) * slot_rs + Int64(main_column) * slot_cs
                        slot = Int64(-1)
                        if cutlass.const_expr(self.premapped):
                            slot = Int64(main_slot_t[slot_index])
                        else:
                            logical_t = tensor1d(main_logical, Int64(2**62))
                            logical = Int64(
                                logical_t[Int64(q) * main_rs + Int64(main_column) * main_cs]
                            )
                            slot = map_logical_slot(
                                logical,
                                request,
                                table,
                                table_rows,
                                table_cols,
                                table_stride,
                                page_size,
                                max_positions,
                                visible_length,
                            )
                            main_slot_t[slot_index] = slot
                        valid = (slot >= 0) & (slot < main_capacity)
                if valid:
                    local = local + 1
            prefix[tid] = local
            cute.arch.barrier()
            offset = 1
            while offset < self.THREADS:
                addend = Int32(0)
                if tid >= offset:
                    addend = prefix[tid - offset]
                cute.arch.barrier()
                prefix[tid] = prefix[tid] + addend
                cute.arch.barrier()
                offset = offset * 2
            total = prefix[self.THREADS - 1]
            destination = prefix[tid] - local
            tags_t = tensor2d(tags, Int64(count), Int64(capacity), Int64(capacity))
            row_slots_t = tensor2d(row_slots, Int64(count), Int64(capacity), Int64(capacity))
            # Pass 2: assign destinations in column order.
            for column in cutlass.range(begin, end, 1):
                valid = cutlass.Boolean(False)
                slot = Int64(-1)
                if column < swa_width:
                    slot = Int64(swa_t[Int64(q) * swa_rs + Int64(column) * swa_cs])
                    valid = (slot >= 0) & (slot < swa_capacity)
                elif column < swa_width + main_width:
                    if cutlass.const_expr(self.has_main):
                        slot = Int64(
                            main_slot_t[Int64(q) * slot_rs + Int64(column - swa_width) * slot_cs]
                        )
                        valid = (slot >= 0) & (slot < main_capacity)
                if valid:
                    tags_t[q, destination] = column
                    row_slots_t[q, destination] = Int32(slot)
                    destination = destination + 1
            if tid == 0:
                tensor1d(counts, Int64(count))[q] = total

    class StageDedupKernel:
        """Map a query's compacted rows onto identical rows its request's anchor query stages.

        A query's anchor is the earliest query of its request within its aligned group
        of eight queries, so anchors stage all their rows. Another query's row takes
        the anchor's row holding the same SWA slot (columns shift with the query) or
        the same main slot, when both rows fall in the same staging pool. ``refs`` holds
        ``anchor * capacity + anchor_row`` for such rows and -1 for rows the query
        stages itself. Bit ``b`` of ``own_blocks[q]`` is set when destination rows
        ``[32 b, 32 b + 32)`` hold an SWA row the query stages itself (``b < 4``).
        """

        THREADS = 256
        PROBES = 32

        def __init__(self, capacity: int, main_width: int):
            self.capacity = capacity
            self.main_width = main_width
            # Open-addressing table of the anchor's main slots, at most half full.
            self.hash_bits = max(2 * main_width - 1, 1).bit_length()

        @cute.jit
        def __call__(
            self,
            swa_slots: cute.Pointer,  # int64 [count, swa_width] strided
            main_slots: cute.Pointer,  # int64 [count, main_width] mapped
            requests: cute.Pointer,  # int64 [count]
            tags: cute.Pointer,  # int32 [count, capacity]
            counts: cute.Pointer,  # int32 [count]
            refs: cute.Pointer,  # int32 [count, capacity] (out)
            own_blocks: cute.Pointer,  # int32 [count] (out)
            count: Int32,
            swa_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            slot_rs: Int64,
            slot_cs: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                swa_slots,
                main_slots,
                requests,
                tags,
                counts,
                refs,
                own_blocks,
                count,
                swa_width,
                swa_rs,
                swa_cs,
                slot_rs,
                slot_cs,
            ).launch(grid=[count, 1, 1], block=[self.THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            swa_slots: cute.Pointer,
            main_slots: cute.Pointer,
            requests: cute.Pointer,
            tags: cute.Pointer,
            counts: cute.Pointer,
            refs: cute.Pointer,
            own_blocks: cute.Pointer,
            count: Int32,
            swa_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            slot_rs: Int64,
            slot_cs: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, _, _ = cute.arch.block_idx()
            smem = SmemAllocator()
            hash_size = 1 << self.hash_bits
            hash_keys = smem.allocate_tensor(Int32, cute.make_layout(hash_size), 16)
            hash_columns = smem.allocate_tensor(Int32, cute.make_layout(hash_size), 16)
            anchor_tags = smem.allocate_tensor(Int32, cute.make_layout(self.capacity), 16)
            block_bits = smem.allocate_tensor(Int32, cute.make_layout(4), 16)
            n = Int64(count)
            capacity = Int32(self.capacity)
            requests_t = tensor1d(requests, n)
            request = Int64(requests_t[q])
            anchor = q
            searching = cutlass.Boolean(True)
            for back in cutlass.range_constexpr(1, 8):
                if searching & (q % 8 >= back):
                    if Int64(requests_t[q - back]) == request:
                        anchor = q - back
                    else:
                        searching = cutlass.Boolean(False)
            tags_t = tensor2d(tags, n, Int64(self.capacity), Int64(self.capacity))
            refs_t = tensor2d(refs, n, Int64(self.capacity), Int64(self.capacity))
            counts_t = tensor1d(counts, n)
            total = Int32(counts_t[q])
            if anchor == q:
                for row in cutlass.range(tid, capacity, self.THREADS):
                    refs_t[q, row] = Int32(-1)
            else:
                swa_t = tensor1d(swa_slots, Int64(2**62))
                slot_t = tensor1d(main_slots, Int64(2**62))
                anchor_total = Int32(counts_t[anchor])
                for entry in cutlass.range(tid, hash_size, self.THREADS):
                    hash_keys[entry] = Int32(-1)
                cute.arch.barrier()
                for column in cutlass.range(tid, self.main_width, self.THREADS):
                    slot = Int64(slot_t[Int64(anchor) * slot_rs + Int64(column) * slot_cs])
                    if (slot >= 0) & (slot < Int64(_INT32_MAX)):
                        key = Int32(slot)
                        entry = Int32(
                            (Uint32(key) * Uint32(2654435761)) >> Uint32(32 - self.hash_bits)
                        )
                        pending = cutlass.Boolean(True)
                        for _ in cutlass.range(self.PROBES):
                            if pending:
                                previous = cute.arch.atomic_cas(
                                    (hash_keys.iterator + entry).llvm_ptr,
                                    cmp=Int32(-1),
                                    val=key,
                                    sem="relaxed",
                                    scope="cta",
                                )
                                if (previous == -1) | (previous == key):
                                    hash_columns[entry] = column
                                    pending = cutlass.Boolean(False)
                                else:
                                    entry = (entry + 1) & (hash_size - 1)
                for row in cutlass.range(tid, capacity, self.THREADS):
                    column = Int32(_INT32_MAX)
                    if row < anchor_total:
                        column = Int32(tags_t[anchor, row])
                    anchor_tags[row] = column
                cute.arch.barrier()
                shift = q - anchor
                for row in cutlass.range(tid, capacity, self.THREADS):
                    reference = Int32(-1)
                    if row < total:
                        column = Int32(tags_t[q, row])
                        match = Int32(-1)
                        if column < swa_width:
                            slot = Int64(swa_t[Int64(q) * swa_rs + Int64(column) * swa_cs])
                            shifted = column + shift
                            if shifted < swa_width:
                                anchor_slot = Int64(
                                    swa_t[Int64(anchor) * swa_rs + Int64(shifted) * swa_cs]
                                )
                                if anchor_slot == slot:
                                    match = shifted
                        elif column < swa_width + self.main_width:
                            slot = Int64(
                                slot_t[Int64(q) * slot_rs + Int64(column - swa_width) * slot_cs]
                            )
                            if slot < Int64(_INT32_MAX):
                                key = Int32(slot)
                                entry = Int32(
                                    (Uint32(key) * Uint32(2654435761))
                                    >> Uint32(32 - self.hash_bits)
                                )
                                pending = cutlass.Boolean(True)
                                for _ in cutlass.range(self.PROBES):
                                    if pending:
                                        stored = hash_keys[entry]
                                        if stored == key:
                                            match = swa_width + hash_columns[entry]
                                            pending = cutlass.Boolean(False)
                                        elif stored == -1:
                                            pending = cutlass.Boolean(False)
                                        else:
                                            entry = (entry + 1) & (hash_size - 1)
                        if match >= 0:
                            # Anchor rows hold its columns in ascending order.
                            low = Int32(0)
                            high = anchor_total
                            for _ in cutlass.range_constexpr(self.capacity.bit_length()):
                                if low < high:
                                    middle = (low + high) // 2
                                    if anchor_tags[middle] < match:
                                        low = middle + 1
                                    else:
                                        high = middle
                            same_pool = ((low < 128) & (row < 128)) | ((low >= 128) & (row >= 128))
                            if (low < anchor_total) & same_pool:
                                if anchor_tags[low] == match:
                                    reference = anchor * capacity + low
                    refs_t[q, row] = reference
            # Warp ``w < 4`` covers destination rows [32 w, 32 w + 32); each thread rereads
            # the row it wrote.
            own = cutlass.Boolean(False)
            if (tid < 128) & (tid < total):
                own = (Int32(tags_t[q, tid]) < swa_width) & (Int32(refs_t[q, tid]) < 0)
            ballot = cute.arch.vote_ballot_sync(own)
            if (tid % 32 == 0) & (tid < 128):
                block_bits[tid // 32] = Int32(ballot != 0) << (tid // 32)
            cute.arch.barrier()
            if tid == 0:
                tensor1d(own_blocks, n)[q] = (
                    block_bits[0] | block_bits[1] | block_bits[2] | block_bits[3]
                )

    class StageDequantKernel:
        """Decode compacted selections into the native primary/extra staging pools.

        Each CTA decodes ``rows_per_cta`` destination rows of one query. CTA ``(q, 0)``
        also publishes the query's ``lengths`` entry (``max(counts[q], 1)``) and native
        ``indices``; CTA ``(0, 0)`` resets the scheduler counter and derives the
        FP8 BMM scales. Each warp stages ``rows_per_cta // 4`` rows and issues all
        their loads before converting any; each lane reads its sixteen channels with
        one payload and one scale load. With ``dedup``, rows
        whose ``refs`` entry names an anchor row are not staged and their indices
        point to that row.

        With ``grouped`` (``dedup`` and 32-row blocks), CTA ``(g, b)`` instead covers row
        block ``b`` of queries ``[4g, 4g + 4)``, a warp per query and a lane per row. The
        rows to stage are listed in shared memory and staged eight per warp; the CTA also
        publishes its block's ``indices``.

        With ``reuse`` the pools already hold this staging group's rows from an earlier
        layer: only SWA rows, which every layer owns privately, are staged again, all
        of them inside the first 128 destination rows, which the grid covers. Main and
        zero rows are kept unless ``staged[slot]``, the KV quant scale they were staged
        with, differs from this layer's; then the CTAs stage every row block. Otherwise,
        with ``dedup`` and one query per CTA, a CTA stops at once when its rows hold none
        of the SWA rows its query stages itself (``own_blocks``). The ``lengths`` and
        ``indices`` the group's first layer published still hold.
        CTA ``(0, 0)`` records this layer's scale in ``staged[1 - slot]``, which no CTA
        of the launch reads.
        """

        THREADS = 128

        def __init__(self, fp8: bool, rows_per_cta: int, dedup: bool, reuse: bool, grouped: bool):
            self.fp8 = fp8
            self.rows_per_cta = rows_per_cta
            self.dedup = dedup
            self.reuse = reuse
            self.queries = self.THREADS // 32 if grouped else 1

        @cute.jit
        def __call__(
            self,
            swa_pool: cute.Pointer,
            main_pool: cute.Pointer,
            swa_slots: cute.Pointer,
            row_slots: cute.Pointer,  # int32 [count, capacity] compacted slot per row
            tags: cute.Pointer,
            counts: cute.Pointer,
            refs: cute.Pointer,  # int32 [count, capacity] anchor rows (dedup)
            primary: cute.Pointer,  # [count, 128, 512]
            extra: cute.Pointer,  # [count, extra_rows, 512]
            quant_scale: cute.Pointer,
            staged: cute.Pointer,  # float32 [2]: KV quant scale the pools' rows carry, by slot
            own_blocks: cute.Pointer,  # int32 [count]: StageDedupKernel's own-SWA-row block bits
            lengths: cute.Pointer,  # int32 [count]
            indices: cute.Pointer,  # int32 [count, capacity]
            counter: cute.Pointer,  # uint32 [1]
            q_dequant_scale: cute.Pointer,  # float32 [1]
            dequant_scale: cute.Pointer,  # float32 [1]
            bmm1_scale: cute.Pointer,  # float32 [2]
            bmm2_scale: cute.Pointer,  # float32 [1]
            count: Int32,
            rows: Int32,
            row_grid: Int32,
            slot: Int32,
            swa_width: Int32,
            main_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            swa_pool_stride: Int64,
            main_pool_stride: Int64,
            swa_capacity: Int64,
            main_capacity: Int64,
            capacity: Int32,
            extra_rows: Int32,
            softmax_scale: Float32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                swa_pool,
                main_pool,
                swa_slots,
                row_slots,
                tags,
                counts,
                refs,
                primary,
                extra,
                quant_scale,
                staged,
                own_blocks,
                lengths,
                indices,
                counter,
                q_dequant_scale,
                dequant_scale,
                bmm1_scale,
                bmm2_scale,
                count,
                rows,
                row_grid,
                slot,
                swa_width,
                main_width,
                swa_rs,
                swa_cs,
                swa_pool_stride,
                main_pool_stride,
                swa_capacity,
                main_capacity,
                capacity,
                extra_rows,
                softmax_scale,
            ).launch(
                grid=[(count + self.queries - 1) // self.queries, row_grid, 1],
                block=[self.THREADS, 1, 1],
                stream=stream,
            )

        @cute.jit
        def _resolve_row(
            self,
            swa_pool: cute.Pointer,
            main_pool: cute.Pointer,
            swa_slots: cute.Pointer,
            row_slots_t: cute.Tensor,
            tags_t: cute.Tensor,
            refs_t: cute.Tensor,
            q: Int32,
            total: Int32,
            rows: Int32,
            swa_width: Int32,
            main_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            swa_pool_stride: Int64,
            main_pool_stride: Int64,
            swa_capacity: Int64,
            main_capacity: Int64,
            capacity: Int32,
            destination: Int32,
            keep: cutlass.Boolean,
        ):
            """Source of query ``q``'s row ``destination``: its kind, 0 for a zero row, 1 for SWA,
            2 for main and 3 for no store, its packed row address and its ``refs`` entry;
            ``keep`` skips main and zero rows."""
            # ``rows`` can exceed ``capacity`` by the one-row extra pool of SWA-only
            # tiles, so the index is clamped and rows past ``total`` ignore the loads.
            index = destination
            if index >= capacity:
                index = capacity - 1
            reference = Int32(-1)
            if cutlass.const_expr(self.dedup):
                reference = Int32(refs_t[q, index])
            tag = Int32(tags_t[q, index])
            slot = Int32(row_slots_t[q, index])
            kind = Int32(0)
            if (destination >= rows) | (reference >= 0):
                kind = Int32(3)
            elif destination < total:
                # Tags hold columns, and the SWA slots backing them are private to each
                # layer, so a column that compaction resolved for the group's first
                # layer can be empty in this one. The pools are addressed without
                # bounds checking, so the slot is re-validated here and a slot outside
                # the pool or a column outside this layer's widths stages a zero row.
                if tag < swa_width:
                    swa_slot = Int64(
                        tensor1d(swa_slots, Int64(2**62))[Int64(q) * swa_rs + Int64(tag) * swa_cs]
                    )
                    if (swa_slot >= 0) & (swa_slot < swa_capacity):
                        kind = Int32(1)
                        slot = Int32(swa_slot)
                elif tag < swa_width + main_width:
                    if (Int64(slot) >= 0) & (Int64(slot) < main_capacity):
                        kind = Int32(2)
            if cutlass.const_expr(self.reuse):
                # The group's main and zero rows are already staged at this scale.
                if keep & ((tag >= swa_width) | (destination >= total)):
                    kind = Int32(3)
            row_base = main_pool.toint() + Int64(slot) * main_pool_stride
            if kind == 1:
                row_base = swa_pool.toint() + Int64(slot) * swa_pool_stride
            return kind, row_base, reference

        @cute.jit
        def _stage_rows(
            self,
            swa_pool: cute.Pointer,
            main_pool: cute.Pointer,
            swa_slots: cute.Pointer,
            row_slots_t: cute.Tensor,
            tags_t: cute.Tensor,
            refs_t: cute.Tensor,
            primary: cute.Pointer,
            extra: cute.Pointer,
            quant_scale: cute.Pointer,
            q: Int32,
            total: Int32,
            rows: Int32,
            swa_width: Int32,
            main_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            swa_pool_stride: Int64,
            main_pool_stride: Int64,
            swa_capacity: Int64,
            main_capacity: Int64,
            capacity: Int32,
            extra_rows: Int32,
            row_block: Int32,
            keep: cutlass.Boolean,
        ):
            """Stage the destination rows of ``row_block``; ``keep`` skips main and zero rows."""
            tid, _, _ = cute.arch.thread_idx()
            # Lane ``r`` resolves the warp's row ``r``.
            destination = (
                row_block * self.rows_per_cta
                + (tid % 32 % (self.rows_per_cta // 4)) * 4
                + tid // 32
            )
            kind, row_base, _ = self._resolve_row(
                swa_pool,
                main_pool,
                swa_slots,
                row_slots_t,
                tags_t,
                refs_t,
                q,
                total,
                rows,
                swa_width,
                main_width,
                swa_rs,
                swa_cs,
                swa_pool_stride,
                main_pool_stride,
                swa_capacity,
                main_capacity,
                capacity,
                destination,
                keep,
            )
            self._stage_resolved(
                primary, extra, quant_scale, extra_rows, q, destination, kind, row_base
            )

        @cute.jit
        def _stage_resolved(
            self,
            primary: cute.Pointer,
            extra: cute.Pointer,
            quant_scale: cute.Pointer,
            extra_rows: Int32,
            q: Int32,
            destination: Int32,
            kind: Int32,
            row_base: Int64,
        ):
            """Stage the warp's rows: lane ``r`` holds row ``r``'s query, destination row,
            kind and source address (``_resolve_row``)."""
            tid, _, _ = cute.arch.thread_idx()
            per_warp = self.rows_per_cta // 4
            first = tid % 32 * 16
            kinds = cute.make_rmem_tensor((per_warp,), Int32)
            scales = cute.make_rmem_tensor((per_warp,), Int32)
            words = [cute.make_rmem_tensor((4,), Uint32) for _ in range(per_warp)]
            load_rows_x16(kind, row_base, per_warp, kinds, scales, words)
            quant_state = staging_quant(quant_scale, self.fp8)
            result = cute.make_rmem_tensor((16,), Float32)
            packed = cute.make_rmem_tensor((4,), Uint32)
            for row in cutlass.range_constexpr(per_warp):
                row_q = cute.arch.shuffle_sync(q, Int32(row))
                row_destination = cute.arch.shuffle_sync(destination, Int32(row))
                if kinds[row] != 3:
                    target = primary
                    element = (Int64(row_q) * 128 + Int64(row_destination)) * 512 + Int64(first)
                    if row_destination >= 128:
                        target = extra
                        element = (
                            Int64(row_q) * Int64(extra_rows) + Int64(row_destination - 128)
                        ) * 512 + Int64(first)
                    stage_row_x16(
                        kinds[row],
                        scales[row],
                        words[row],
                        quant_state,
                        self.fp8,
                        target,
                        element,
                        result,
                        packed,
                    )

        @cute.jit
        def _stage_block(
            self,
            swa_pool: cute.Pointer,
            main_pool: cute.Pointer,
            swa_slots: cute.Pointer,
            row_slots_t: cute.Tensor,
            tags_t: cute.Tensor,
            refs_t: cute.Tensor,
            primary: cute.Pointer,
            extra: cute.Pointer,
            quant_scale: cute.Pointer,
            q: Int32,
            total: Int32,
            rows: Int32,
            swa_width: Int32,
            main_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            swa_pool_stride: Int64,
            main_pool_stride: Int64,
            swa_capacity: Int64,
            main_capacity: Int64,
            capacity: Int32,
            extra_rows: Int32,
            count: Int32,
            indices_t: cute.Tensor,
            row_block: Int32,
            keep: cutlass.Boolean,
            listed: tuple,
        ):
            """Stage row block ``row_block`` of the CTA's queries: warp ``w`` resolves query ``q``'s
            rows, the ones to stage are listed in shared memory (``listed``: kinds, queries,
            rows, source addresses, per-warp counts) and the warps stage them eight at a time.
            Without ``reuse`` the block's ``indices`` are published too."""
            tid, _, _ = cute.arch.thread_idx()
            warp = tid // 32
            lane = tid % 32
            listed_kinds, listed_queries, listed_rows, listed_bases, warp_counts = listed
            destination = row_block * 32 + lane
            kind = Int32(3)
            row_base = Int64(0)
            if q < count:
                kind, row_base, reference = self._resolve_row(
                    swa_pool,
                    main_pool,
                    swa_slots,
                    row_slots_t,
                    tags_t,
                    refs_t,
                    q,
                    total,
                    rows,
                    swa_width,
                    main_width,
                    swa_rs,
                    swa_cs,
                    swa_pool_stride,
                    main_pool_stride,
                    swa_capacity,
                    main_capacity,
                    capacity,
                    destination,
                    keep,
                )
                if cutlass.const_expr(not self.reuse):
                    if destination < capacity:
                        value = Int32(-1)
                        if (destination < total) | (destination == 0):
                            owner = q
                            row = destination
                            if reference >= 0:
                                owner = reference // capacity
                                row = reference - owner * capacity
                            value = owner * (capacity - 128) + row - 128
                            if row < 128:
                                value = owner * 128 + row
                        indices_t[q, destination] = value
            selected = kind != 3
            ballot = cute.arch.vote_ballot_sync(selected)
            if lane == 0:
                warp_counts[warp] = Int32(cute.arch.popc(ballot))
            cute.arch.barrier()
            offset = Int32(0)
            listed_total = Int32(0)
            for w in cutlass.range_constexpr(self.queries):
                listed_w = warp_counts[w]
                if w < warp:
                    offset = offset + listed_w
                listed_total = listed_total + listed_w
            if selected:
                below = (Uint32(1) << Uint32(lane)) - Uint32(1)
                position = offset + Int32(cute.arch.popc(ballot & below))
                listed_kinds[position] = kind
                listed_queries[position] = q
                listed_rows[position] = destination
                listed_bases[position] = row_base
            cute.arch.barrier()
            per_warp = self.rows_per_cta // 4
            for first in cutlass.range(0, listed_total, per_warp * self.queries):
                entry = first + warp * per_warp + lane
                row_kind = Int32(3)
                row_q = Int32(0)
                row_destination = Int32(0)
                row_source = Int64(0)
                if (lane < per_warp) & (entry < listed_total):
                    row_kind = listed_kinds[entry]
                    row_q = listed_queries[entry]
                    row_destination = listed_rows[entry]
                    row_source = listed_bases[entry]
                self._stage_resolved(
                    primary,
                    extra,
                    quant_scale,
                    extra_rows,
                    row_q,
                    row_destination,
                    row_kind,
                    row_source,
                )
            cute.arch.barrier()

        @cute.kernel
        def kernel(
            self,
            swa_pool: cute.Pointer,
            main_pool: cute.Pointer,
            swa_slots: cute.Pointer,
            row_slots: cute.Pointer,
            tags: cute.Pointer,
            counts: cute.Pointer,
            refs: cute.Pointer,
            primary: cute.Pointer,
            extra: cute.Pointer,
            quant_scale: cute.Pointer,
            staged: cute.Pointer,
            own_blocks: cute.Pointer,
            lengths: cute.Pointer,
            indices: cute.Pointer,
            counter: cute.Pointer,
            q_dequant_scale: cute.Pointer,
            dequant_scale: cute.Pointer,
            bmm1_scale: cute.Pointer,
            bmm2_scale: cute.Pointer,
            count: Int32,
            rows: Int32,
            row_grid: Int32,
            slot: Int32,
            swa_width: Int32,
            main_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            swa_pool_stride: Int64,
            main_pool_stride: Int64,
            swa_capacity: Int64,
            main_capacity: Int64,
            capacity: Int32,
            extra_rows: Int32,
            softmax_scale: Float32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, block, _ = cute.arch.block_idx()
            total = Int32(tensor1d(counts, Int64(count))[q])
            refs_t = tensor2d(refs, Int64(count), Int64(capacity), Int64(capacity))
            tags_t = tensor2d(tags, Int64(count), Int64(capacity), Int64(capacity))
            row_slots_t = tensor2d(row_slots, Int64(count), Int64(capacity), Int64(capacity))
            indices_t = tensor2d(indices, Int64(count), Int64(capacity), Int64(capacity))
            extra_capacity = capacity - 128
            stage = (
                swa_pool,
                main_pool,
                swa_slots,
                row_slots_t,
                tags_t,
                refs_t,
                primary,
                extra,
                quant_scale,
                q,
                total,
                rows,
                swa_width,
                main_width,
                swa_rs,
                swa_cs,
                swa_pool_stride,
                main_pool_stride,
                swa_capacity,
                main_capacity,
                capacity,
                extra_rows,
            )
            if cutlass.const_expr(self.queries > 1):
                q = q * self.queries + tid // 32
                total = Int32(0)
                if q < count:
                    total = Int32(tensor1d(counts, Int64(count))[q])
                smem = SmemAllocator()
                listed = (
                    smem.allocate_tensor(Int32, cute.make_layout(self.THREADS), 16),
                    smem.allocate_tensor(Int32, cute.make_layout(self.THREADS), 16),
                    smem.allocate_tensor(Int32, cute.make_layout(self.THREADS), 16),
                    smem.allocate_tensor(Int64, cute.make_layout(self.THREADS), 16),
                    smem.allocate_tensor(Int32, cute.make_layout(self.queries), 16),
                )
                stage = stage[:9] + (q, total) + stage[11:]
                keep = cutlass.Boolean(self.reuse)
                restage = cutlass.Boolean(False)
                if cutlass.const_expr(self.reuse and self.fp8):
                    quant = Float32(tensor1d(quant_scale, Int64(1))[0])
                    held = Float32(tensor1d(staged, Int64(2))[slot])
                    keep = quant == held
                    restage = quant != held
                self._stage_block(*stage, count, indices_t, block, keep, listed)
                if restage:
                    for row_block in cutlass.range(block + row_grid, (rows + 31) // 32, row_grid):
                        self._stage_block(*stage, count, indices_t, row_block, keep, listed)
                if cutlass.const_expr(not self.reuse):
                    if (block == 0) & (q < count) & (tid % 32 == 0):
                        length = total
                        if length < 1:
                            length = Int32(1)
                        tensor1d(lengths, Int64(count))[q] = length
            elif cutlass.const_expr(self.reuse):
                keep = cutlass.Boolean(True)
                row_blocks = row_grid
                bits = Int32(-1)
                if cutlass.const_expr(self.dedup):
                    bits = Int32(tensor1d(own_blocks, Int64(count))[q])
                if cutlass.const_expr(self.fp8):
                    held = Float32(tensor1d(staged, Int64(2))[slot])
                    if Float32(tensor1d(quant_scale, Int64(1))[0]) != held:
                        keep = cutlass.Boolean(False)
                        row_blocks = (rows + self.rows_per_cta - 1) // self.rows_per_cta
                if keep & (((bits >> (block * self.rows_per_cta // 32)) & 1) == 0):
                    row_blocks = Int32(0)
                for row_block in cutlass.range(block, row_blocks, row_grid):
                    self._stage_rows(*stage, row_block, keep)
            else:
                self._stage_rows(*stage, block, cutlass.Boolean(False))
            if (block == 0) & (not self.reuse) & (self.queries == 1):
                length = total
                if length < 1:
                    length = Int32(1)
                if tid == 0:
                    tensor1d(lengths, Int64(count))[q] = length
                for column in cutlass.range(tid, capacity, self.THREADS):
                    physical = Int32(q) * extra_capacity + column - 128
                    if column < 128:
                        physical = Int32(q) * 128 + column
                    value = Int32(-1)
                    if column < length:
                        value = physical
                        if cutlass.const_expr(self.dedup):
                            reference = Int32(refs_t[q, column])
                            if reference >= 0:
                                anchor = reference // capacity
                                anchor_row = reference - anchor * capacity
                                value = anchor * extra_capacity + anchor_row - 128
                                if anchor_row < 128:
                                    value = anchor * 128 + anchor_row
                    indices_t[q, column] = value
            if (q == 0) & (block == 0) & (tid == 0):
                tensor1d(counter, Int64(1))[0] = Uint32(0)
                if cutlass.const_expr(self.fp8):
                    store_bmm_scales(
                        q_dequant_scale, dequant_scale, bmm1_scale, bmm2_scale, softmax_scale
                    )
                    tensor1d(staged, Int64(2))[1 - slot] = Float32(
                        tensor1d(quant_scale, Int64(1))[0]
                    )

    # --------------------------------------- K12/K13: shared prefill staging

    class SharedCompactKernel:
        """Map each query's selections onto the shared bank rows, one warp per query.

        A warp walks its query's columns 32 at a time: a ballot of the valid
        columns gives every lane its compacted destination, so the columns keep
        their order without block-wide scans. Main columns are resolved
        ``UNROLL`` chunks at a time, so their loads are in flight together, and
        with ``shift_pages`` a power-of-two page size maps them with shifts.

        ``selected`` marks the selected SWA bank rows. Its main rows hold
        ``SharedRowScaleKernel``'s words instead: a selected main row's word
        folds into ``selected[0]``, the largest group-scale byte (plus one) of
        every selected main row, so main rows need no marks. A request's first
        SWA bank row is position ``max(positions[first[r]] - window + 1, 0,
        floors[r])``; the request's first query publishes it to ``swa_starts``
        for the decode launch.
        """

        WARPS = 8
        UNROLL = 4

        def __init__(self, has_main: bool, premapped: bool, shift_pages: bool):
            self.has_main = has_main
            self.premapped = premapped
            self.shift_pages = shift_pages

        @cute.jit
        def __call__(
            self,
            swa_slots: cute.Pointer,
            main_logical: cute.Pointer,
            main_slots: cute.Pointer,  # int64 physical [count, main_width] strided (premapped)
            requests: cute.Pointer,  # int64 [count] (plan query requests)
            positions: cute.Pointer,  # int32 [count]
            swa_starts: cute.Pointer,  # int64 [requests] (out)
            swa_first: cute.Pointer,  # int64 [requests]
            swa_floors: cute.Pointer,  # int64 [requests]
            swa_offsets: cute.Pointer,  # int64 [requests + 1]
            main_offsets: cute.Pointer,  # int64 [requests + 1]
            visible: cute.Pointer,  # int64 [count]
            table: cute.Pointer,  # owner page table
            selected: cute.Pointer,  # int32 [bank_rows]
            indices: cute.Pointer,  # int32 [count, capacity]
            lengths: cute.Pointer,  # int32 [count]
            count: Int32,
            swa_width: Int32,
            main_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            main_rs: Int64,
            main_cs: Int64,
            slot_rs: Int64,
            slot_cs: Int64,
            swa_capacity: Int64,
            main_capacity: Int64,
            table_rows: Int32,
            table_cols: Int32,
            table_stride: Int64,
            page_size: Int64,
            page_shift: Int32,
            max_positions: Int64,
            request_count: Int32,
            capacity: Int32,
            window: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                swa_slots,
                main_logical,
                main_slots,
                requests,
                positions,
                swa_starts,
                swa_first,
                swa_floors,
                swa_offsets,
                main_offsets,
                visible,
                table,
                selected,
                indices,
                lengths,
                count,
                swa_width,
                main_width,
                swa_rs,
                swa_cs,
                main_rs,
                main_cs,
                slot_rs,
                slot_cs,
                swa_capacity,
                main_capacity,
                table_rows,
                table_cols,
                table_stride,
                page_size,
                page_shift,
                max_positions,
                request_count,
                capacity,
                window,
            ).launch(
                grid=[(count + self.WARPS - 1) // self.WARPS, 1, 1],
                block=[32 * self.WARPS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            swa_slots: cute.Pointer,
            main_logical: cute.Pointer,
            main_slots: cute.Pointer,
            requests: cute.Pointer,
            positions: cute.Pointer,
            swa_starts: cute.Pointer,
            swa_first: cute.Pointer,
            swa_floors: cute.Pointer,
            swa_offsets: cute.Pointer,
            main_offsets: cute.Pointer,
            visible: cute.Pointer,
            table: cute.Pointer,
            selected: cute.Pointer,
            indices: cute.Pointer,
            lengths: cute.Pointer,
            count: Int32,
            swa_width: Int32,
            main_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            main_rs: Int64,
            main_cs: Int64,
            slot_rs: Int64,
            slot_cs: Int64,
            swa_capacity: Int64,
            main_capacity: Int64,
            table_rows: Int32,
            table_cols: Int32,
            table_stride: Int64,
            page_size: Int64,
            page_shift: Int32,
            max_positions: Int64,
            request_count: Int32,
            capacity: Int32,
            window: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            lane = tid % 32
            q = Int32(block) * self.WARPS + tid // 32
            smem = SmemAllocator()
            warp_words = smem.allocate_tensor(Int32, cute.make_layout(self.WARPS), 16)
            selected_t = tensor1d(selected, Int64(2**62))
            # Largest selected main row word of this lane; 0 is "none".
            main_word = Int32(0)
            if q < count:
                n = Int64(count)
                request = Int64(tensor1d(requests, n)[q])
                request_valid = (request >= 0) & (request < Int64(request_count))
                safe_request = request
                if safe_request < 0:
                    safe_request = Int64(0)
                if safe_request > Int64(request_count) - 1:
                    safe_request = Int64(request_count) - 1
                position = Int64(tensor1d(positions, n)[q])
                swa_start = Int64(0)
                swa_begin = Int64(0)
                swa_end = Int64(0)
                main_begin = Int64(0)
                main_end = Int64(0)
                if request_valid:
                    request_t = Int64(request_count)
                    first = Int64(tensor1d(swa_first, request_t)[Int32(safe_request)])
                    swa_start = Int64(tensor1d(positions, n)[Int32(first)]) - window + 1
                    if swa_start < 0:
                        swa_start = Int64(0)
                    floor = Int64(tensor1d(swa_floors, request_t)[Int32(safe_request)])
                    if swa_start < floor:
                        swa_start = floor
                    if (Int64(q) == first) & (lane == 0):
                        tensor1d(swa_starts, request_t)[Int32(safe_request)] = swa_start
                    swa_offset_t = tensor1d(swa_offsets, Int64(request_count) + 1)
                    swa_begin = Int64(swa_offset_t[Int32(safe_request)])
                    swa_end = Int64(swa_offset_t[Int32(safe_request) + 1])
                    main_offset_t = tensor1d(main_offsets, Int64(request_count) + 1)
                    main_begin = Int64(main_offset_t[Int32(safe_request)])
                    main_end = Int64(main_offset_t[Int32(safe_request) + 1])
                swa_t = tensor1d(swa_slots, Int64(2**62))
                indices_t = tensor2d(indices, n, Int64(capacity), Int64(capacity))
                below = (Uint32(1) << Uint32(lane)) - Uint32(1)
                total = Int32(0)
                for chunk in cutlass.range(0, swa_width, 32, unroll=4):
                    column = chunk + lane
                    valid = cutlass.Boolean(False)
                    bank_row = Int64(0)
                    if column < swa_width:
                        slot = Int64(swa_t[Int64(q) * swa_rs + Int64(column) * swa_cs])
                        logical_swa = position - Int64(swa_width) + 1 + Int64(column)
                        bank_row = swa_begin + logical_swa - swa_start
                        valid = (
                            (slot >= 0)
                            & (slot < swa_capacity)
                            & (bank_row >= swa_begin)
                            & (bank_row < swa_end)
                            & request_valid
                        )
                    if valid:
                        # Every writer stores the same mark.
                        selected_t[bank_row] = Int32(1)
                    ballot = cute.arch.vote_ballot_sync(valid)
                    if valid:
                        indices_t[q, total + Int32(cute.arch.popc(ballot & below))] = Int32(
                            bank_row
                        )
                    total = total + Int32(cute.arch.popc(ballot))
                if cutlass.const_expr(self.has_main):
                    visible_length = Int64(tensor1d(visible, n)[q])
                    logical_t = tensor1d(main_logical, Int64(2**62))
                    slot_t = tensor1d(main_slots, Int64(2**62))
                    domain = main_end - main_begin
                    # ``map_logical_slot``'s per-query terms.
                    table_request = Int32(0)
                    if request > 0:
                        table_request = Int32(request)
                    if request > Int64(table_rows) - 1:
                        table_request = table_rows - 1
                    table_valid = (table_rows > 0) & (table_cols > 0)
                    table_valid = table_valid & (request >= 0) & (request < Int64(table_rows))
                    limit = max_positions
                    if visible_length < limit:
                        limit = visible_length
                    table_t = tensor2d(table, Int64(table_rows), Int64(table_cols), table_stride)
                    for chunk in cutlass.range(0, main_width, 32 * self.UNROLL):
                        logicals, slots = [], []
                        for unroll in cutlass.range_constexpr(self.UNROLL):
                            column = chunk + 32 * unroll + lane
                            logical = Int64(-1)
                            slot = Int64(-1)
                            if column < main_width:
                                logical = Int64(
                                    logical_t[Int64(q) * main_rs + Int64(column) * main_cs]
                                )
                                if cutlass.const_expr(self.premapped):
                                    slot = Int64(
                                        slot_t[Int64(q) * slot_rs + Int64(column) * slot_cs]
                                    )
                            logicals.append(logical)
                            slots.append(slot)
                        if cutlass.const_expr(not self.premapped):
                            for unroll in cutlass.range_constexpr(self.UNROLL):
                                logical = logicals[unroll]
                                if cutlass.const_expr(self.shift_pages):
                                    # ``map_logical_slot`` with shifts for the division.
                                    bounded = logical
                                    if bounded > max_positions:
                                        bounded = max_positions
                                    if bounded > Int64(2**31 - 1):
                                        bounded = Int64(2**31 - 1)
                                    if bounded < 0:
                                        bounded = Int64(0)
                                    page = Int32(bounded) >> page_shift
                                    safe_page = page
                                    if safe_page > table_cols - 1:
                                        safe_page = table_cols - 1
                                    physical = Int64(-1)
                                    if table_valid & (logical >= 0):
                                        physical = Int64(table_t[table_request, safe_page])
                                    mapped = Int64(-1)
                                    if (
                                        table_valid
                                        & (logical >= 0)
                                        & (logical < limit)
                                        & (page < table_cols)
                                        & (physical >= 0)
                                    ):
                                        mapped = (physical << Int64(page_shift)) + Int64(
                                            Int32(bounded) & ((Int32(1) << page_shift) - 1)
                                        )
                                    slots[unroll] = mapped
                                else:
                                    slots[unroll] = map_logical_slot(
                                        logical,
                                        request,
                                        table,
                                        table_rows,
                                        table_cols,
                                        table_stride,
                                        page_size,
                                        max_positions,
                                        visible_length,
                                    )
                        for unroll in cutlass.range_constexpr(self.UNROLL):
                            column = chunk + 32 * unroll + lane
                            logical = logicals[unroll]
                            slot = slots[unroll]
                            bank_row = main_begin + logical
                            valid = (
                                (column < main_width)
                                & (slot >= 0)
                                & (slot < main_capacity)
                                & (logical >= 0)
                                & (logical < domain)
                                & request_valid
                            )
                            if valid:
                                word = Int32(selected_t[bank_row])
                                if word > main_word:
                                    main_word = word
                            ballot = cute.arch.vote_ballot_sync(valid)
                            if valid:
                                indices_t[q, total + Int32(cute.arch.popc(ballot & below))] = Int32(
                                    bank_row
                                )
                            total = total + Int32(cute.arch.popc(ballot))
                for column in cutlass.range(total + lane, capacity, 32):
                    indices_t[q, column] = Int32(-1)
                if lane == 0:
                    length = total
                    if length < 1:
                        length = Int32(1)
                        indices_t[q, 0] = Int32(0)
                    tensor1d(lengths, n)[q] = length
            if cutlass.const_expr(self.has_main):
                value = warp_max(main_word)
                if lane == 0:
                    warp_words[tid // 32] = value
                cute.arch.barrier()
                if tid == 0:
                    for warp in cutlass.range_constexpr(1, self.WARPS):
                        other = Int32(warp_words[warp])
                        if other > value:
                            value = other
                    if value > 0:
                        cute.arch.atomic_max(selected_t.iterator, value)

    class SharedDecodeKernel:
        """Decode every marked SWA bank row and every main bank row once from its owner page
        (``_shared_decode_rows``).

        A warp covers ``ROWS_PER_WARP`` consecutive bank rows. Lane ``r`` resolves row
        ``r``'s owner slot, then every lane loads sixteen channels of each row (sixteen
        SWA or eight main payload bytes and their group-scale byte) before converting
        any, with ``StageDequantKernel``'s arithmetic. Row 0 is zeroed, rows
        ``[1, main_base)`` are SWA rows and the rest are main rows; every main row is
        decoded, including unselected ones that attention never reads. The SWA marks stay set,
        so the later layers of a staging group decode the same rows without compacting
        again. Those layers read the same main pool, so with ``reuse`` they keep the
        bank's main rows when ``main_scales[parity]`` holds this launch's quantization
        scale; every launch records its scale in ``main_scales[1 - parity]``. CTA 0 also
        resets the scheduler counter and derives the FP8 BMM scales from the staging
        scales, which are final by the time this launch runs.
        """

        THREADS = 256
        ROWS_PER_WARP = 4
        ROWS = THREADS // 32 * ROWS_PER_WARP

        def __init__(self, fp8: bool):
            _format_geometry("swa", 512)
            _format_geometry("main", 512)
            self.fp8 = fp8

        @cute.jit
        def __call__(
            self,
            swa_pool: cute.Pointer,
            main_pool: cute.Pointer,
            swa_pages: cute.Pointer,  # int32 [requests, columns] strided
            main_pages: cute.Pointer,  # int32 [requests, columns] strided
            starts: cute.Pointer,  # int64 [requests]
            swa_offsets: cute.Pointer,  # int64 [requests + 1]
            main_offsets: cute.Pointer,  # int64 [requests + 1]
            selected: cute.Pointer,  # int32 [bank rows]
            bank: cute.Pointer,  # bf16/fp8 [bank rows, 512]
            quant_scale: cute.Pointer,
            q_dequant_scale: cute.Pointer,  # float32 [1]
            dequant_scale: cute.Pointer,  # float32 [1]
            bmm1_scale: cute.Pointer,  # float32 [2]
            bmm2_scale: cute.Pointer,  # float32 [1]
            counter: cute.Pointer,
            main_scales: cute.Pointer,  # int32 [2]: main rows' scale bits by parity
            rows: Int32,
            main_base: Int32,
            swa_capacity: Int64,
            swa_stride: Int64,
            main_capacity: Int64,
            main_stride: Int64,
            swa_page_rows: Int32,
            swa_page_cols: Int32,
            swa_page_stride: Int64,
            swa_page_size: Int64,
            main_page_rows: Int32,
            main_page_cols: Int32,
            main_page_stride: Int64,
            main_page_size: Int64,
            request_count: Int32,
            softmax_scale: Float32,
            reuse: Int32,
            parity: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                swa_pool,
                main_pool,
                swa_pages,
                main_pages,
                starts,
                swa_offsets,
                main_offsets,
                selected,
                bank,
                quant_scale,
                q_dequant_scale,
                dequant_scale,
                bmm1_scale,
                bmm2_scale,
                counter,
                main_scales,
                rows,
                main_base,
                swa_capacity,
                swa_stride,
                main_capacity,
                main_stride,
                swa_page_rows,
                swa_page_cols,
                swa_page_stride,
                swa_page_size,
                main_page_rows,
                main_page_cols,
                main_page_stride,
                main_page_size,
                request_count,
                softmax_scale,
                reuse,
                parity,
            ).launch(
                grid=[(rows + self.ROWS - 1) // self.ROWS, 1, 1],
                block=[self.THREADS, 1, 1],
                stream=stream,
            )

        @cute.jit
        def _slot(
            self,
            pages,
            starts,
            offsets,
            row,
            pool_capacity,
            page_rows,
            page_cols,
            page_stride,
            page_size,
            request_count,
            cache_format: cutlass.Constexpr,
        ) -> Int64:
            """The row's pool slot, or -1 when its page is unavailable or the row lies
            past every request's domain."""
            offsets_t = tensor1d(offsets, Int64(request_count) + 1)
            # Select the first domain ending after this row. Repeated
            # offsets from empty requests are skipped by the upper bound.
            request = Int32(0)
            upper = request_count
            while request < upper:
                candidate = (request + upper) // 2
                end = Int64(offsets_t[candidate + 1])
                if row < end:
                    upper = candidate
                else:
                    request = candidate + 1
            slot = Int64(-1)
            if request < request_count:
                logical = row - Int64(offsets_t[request])
                if cutlass.const_expr(cache_format == "swa"):
                    logical = logical + Int64(tensor1d(starts, Int64(request_count))[request])
                # Bank rows and their logical positions fit 32-bit division.
                if (logical >= 0) & (logical < Int64(2**31 - 1)):
                    page_size32 = Int32(page_size)
                    column = Int32(logical) // page_size32
                    if column < page_cols:
                        page = Int64(
                            tensor2d(pages, Int64(page_rows), Int64(page_cols), page_stride)[
                                request, column
                            ]
                        )
                        if page >= 0:
                            slot = page * page_size + Int64(Int32(logical) % page_size32)
                            if slot >= pool_capacity:
                                slot = Int64(-1)
            return slot

        @cute.jit
        def _word(self, pool, byte_offset: Int64) -> Int32:
            address = pool.toint() + byte_offset
            pointer = cute.make_ptr(Uint32, address, cute.AddressSpace.gmem, assumed_align=4)
            return Int32(cute.make_tensor(pointer, cute.make_layout(1))[0])

        @cute.kernel
        def kernel(
            self,
            swa_pool: cute.Pointer,
            main_pool: cute.Pointer,
            swa_pages: cute.Pointer,
            main_pages: cute.Pointer,
            starts: cute.Pointer,
            swa_offsets: cute.Pointer,
            main_offsets: cute.Pointer,
            selected: cute.Pointer,
            bank: cute.Pointer,
            quant_scale: cute.Pointer,
            q_dequant_scale: cute.Pointer,
            dequant_scale: cute.Pointer,
            bmm1_scale: cute.Pointer,
            bmm2_scale: cute.Pointer,
            counter: cute.Pointer,
            main_scales: cute.Pointer,
            rows: Int32,
            main_base: Int32,
            swa_capacity: Int64,
            swa_stride: Int64,
            main_capacity: Int64,
            main_stride: Int64,
            swa_page_rows: Int32,
            swa_page_cols: Int32,
            swa_page_stride: Int64,
            swa_page_size: Int64,
            main_page_rows: Int32,
            main_page_cols: Int32,
            main_page_stride: Int64,
            main_page_size: Int64,
            request_count: Int32,
            softmax_scale: Float32,
            reuse: Int32,
            parity: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            lane = tid % 32
            held = tensor1d(main_scales, Int64(2))
            scale_bits = Int32(0)
            if cutlass.const_expr(self.fp8):
                scale_bits = self._word(quant_scale, Int64(0))
            decode_main = (reuse == 0) | (Int32(held[parity]) != scale_bits)
            if (block == 0) & (tid == 0):
                held[1 - parity] = scale_bits
                tensor1d(counter, Int64(1))[0] = Uint32(0)
                if cutlass.const_expr(self.fp8):
                    store_bmm_scales(
                        q_dequant_scale, dequant_scale, bmm1_scale, bmm2_scale, softmax_scale
                    )
            # Lane ``r`` resolves the warp's row ``r``: kind 0 stages zeros, 1 an SWA row,
            # 2 a main row, and 3 leaves the bank row untouched.
            first_row = (
                Int64(block) * (self.THREADS // 32) + Int64(tid // 32)
            ) * self.ROWS_PER_WARP
            row = first_row + Int64(lane)
            kind = Int32(3)
            row_base = Int64(0)
            if (lane < self.ROWS_PER_WARP) & (row < Int64(rows)):
                if (row < Int64(main_base)) | decode_main:
                    if row >= Int64(main_base):
                        kind = Int32(0)
                        slot = self._slot(
                            main_pages,
                            starts,
                            main_offsets,
                            row,
                            main_capacity,
                            main_page_rows,
                            main_page_cols,
                            main_page_stride,
                            main_page_size,
                            request_count,
                            "main",
                        )
                        if slot >= 0:
                            kind = Int32(2)
                            row_base = main_pool.toint() + slot * main_stride
                    elif row == 0:
                        kind = Int32(0)
                    elif Int32(tensor1d(selected, Int64(2**62))[row]) != 0:
                        kind = Int32(0)
                        slot = self._slot(
                            swa_pages,
                            starts,
                            swa_offsets,
                            row,
                            swa_capacity,
                            swa_page_rows,
                            swa_page_cols,
                            swa_page_stride,
                            swa_page_size,
                            request_count,
                            "swa",
                        )
                        if slot >= 0:
                            kind = Int32(1)
                            row_base = swa_pool.toint() + slot * swa_stride
            first = lane * 16
            kinds = cute.make_rmem_tensor((self.ROWS_PER_WARP,), Int32)
            scales = cute.make_rmem_tensor((self.ROWS_PER_WARP,), Int32)
            words = [cute.make_rmem_tensor((4,), Uint32) for _ in range(self.ROWS_PER_WARP)]
            load_rows_x16(kind, row_base, self.ROWS_PER_WARP, kinds, scales, words)
            quant_state = staging_quant(quant_scale, self.fp8)
            result = cute.make_rmem_tensor((16,), Float32)
            packed = cute.make_rmem_tensor((4,), Uint32)
            for r in cutlass.range_constexpr(self.ROWS_PER_WARP):
                if kinds[r] != 3:
                    element = (first_row + r) * 512 + Int64(first)
                    stage_row_x16(
                        kinds[r],
                        scales[r],
                        words[r],
                        quant_state,
                        self.fp8,
                        bank,
                        element,
                        result,
                        packed,
                    )

    class SharedRowScaleKernel(SharedDecodeKernel):
        """Write every main bank row's largest group-scale byte plus one (0 without a slot)
        into ``selected``, where ``SharedCompactKernel`` folds the selected ones.

        Rows resolve through ``_slot`` as ``SharedDecodeKernel`` decodes them, so the
        folded scale bounds exactly what staging writes.
        """

        THREADS = 256

        def __init__(self):
            self.limit = _SCALE_BYTE_LIMITS["main"]

        @cute.jit
        def __call__(
            self,
            main_pool: cute.Pointer,
            main_pages: cute.Pointer,
            starts: cute.Pointer,
            main_offsets: cute.Pointer,
            selected: cute.Pointer,
            rows: Int32,
            main_base: Int32,
            main_capacity: Int64,
            main_stride: Int64,
            main_page_rows: Int32,
            main_page_cols: Int32,
            main_page_stride: Int64,
            main_page_size: Int64,
            request_count: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                main_pool,
                main_pages,
                starts,
                main_offsets,
                selected,
                rows,
                main_base,
                main_capacity,
                main_stride,
                main_page_rows,
                main_page_cols,
                main_page_stride,
                main_page_size,
                request_count,
            ).launch(
                grid=[(rows - main_base + self.THREADS - 1) // self.THREADS, 1, 1],
                block=[self.THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            main_pool: cute.Pointer,
            main_pages: cute.Pointer,
            starts: cute.Pointer,
            main_offsets: cute.Pointer,
            selected: cute.Pointer,
            rows: Int32,
            main_base: Int32,
            main_capacity: Int64,
            main_stride: Int64,
            main_page_rows: Int32,
            main_page_cols: Int32,
            main_page_stride: Int64,
            main_page_size: Int64,
            request_count: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            row = Int64(main_base) + Int64(block) * self.THREADS + Int64(tid)
            if row < Int64(rows):
                word = Int32(0)
                slot = self._slot(
                    main_pages,
                    starts,
                    main_offsets,
                    row,
                    main_capacity,
                    main_page_rows,
                    main_page_cols,
                    main_page_stride,
                    main_page_size,
                    request_count,
                    "main",
                )
                if slot >= 0:
                    word = (
                        largest_scale_byte(main_pool, slot * main_stride + 256, 8, self.limit) + 1
                    )
                tensor1d(selected, Int64(2**62))[row] = word

    class SharedScaleKernel(SharedDecodeKernel):
        """Reduce the group-scale bytes of every marked SWA bank row, and the main rows'
        byte that the group's compaction folded into ``selected[0]``, into the staging bitmap.

        The marked rows are the ones ``SharedDecodeKernel`` decodes, resolved
        through the same ``_slot``, so the scale bounds exactly what staging
        writes while each row is read once. One thread per SWA bank row; marks
        stay set for ``SharedDecodeKernel``.
        """

        THREADS = 256

        def __init__(self):
            self.limits = (_SCALE_BYTE_LIMITS["swa"], _SCALE_BYTE_LIMITS["main"])
            self.indices = (_SCALE_SOURCE_SLOTS["swa"], _SCALE_SOURCE_SLOTS["main"])

        @cute.jit
        def __call__(
            self,
            swa_pool: cute.Pointer,
            main_pool: cute.Pointer,
            swa_pages: cute.Pointer,
            main_pages: cute.Pointer,
            starts: cute.Pointer,
            swa_offsets: cute.Pointer,
            main_offsets: cute.Pointer,
            selected: cute.Pointer,
            bitmap: cute.Pointer,  # int32 [17]
            rows: Int32,
            main_base: Int32,
            swa_capacity: Int64,
            swa_stride: Int64,
            main_capacity: Int64,
            main_stride: Int64,
            swa_page_rows: Int32,
            swa_page_cols: Int32,
            swa_page_stride: Int64,
            swa_page_size: Int64,
            main_page_rows: Int32,
            main_page_cols: Int32,
            main_page_stride: Int64,
            main_page_size: Int64,
            request_count: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                swa_pool,
                main_pool,
                swa_pages,
                main_pages,
                starts,
                swa_offsets,
                main_offsets,
                selected,
                bitmap,
                rows,
                main_base,
                swa_capacity,
                swa_stride,
                main_capacity,
                main_stride,
                swa_page_rows,
                swa_page_cols,
                swa_page_stride,
                swa_page_size,
                main_page_rows,
                main_page_cols,
                main_page_stride,
                main_page_size,
                request_count,
            ).launch(
                grid=[(main_base + self.THREADS - 1) // self.THREADS, 1, 1],
                block=[self.THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            swa_pool: cute.Pointer,
            main_pool: cute.Pointer,
            swa_pages: cute.Pointer,
            main_pages: cute.Pointer,
            starts: cute.Pointer,
            swa_offsets: cute.Pointer,
            main_offsets: cute.Pointer,
            selected: cute.Pointer,
            bitmap: cute.Pointer,
            rows: Int32,
            main_base: Int32,
            swa_capacity: Int64,
            swa_stride: Int64,
            main_capacity: Int64,
            main_stride: Int64,
            swa_page_rows: Int32,
            swa_page_cols: Int32,
            swa_page_stride: Int64,
            swa_page_size: Int64,
            main_page_rows: Int32,
            main_page_cols: Int32,
            main_page_stride: Int64,
            main_page_size: Int64,
            request_count: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            smem = SmemAllocator()
            warps = smem.allocate_tensor(Int32, cute.make_layout(2 * (self.THREADS // 32)), 16)
            # -1 is "this thread saw no bounded scale", below every real byte.
            largest = cute.make_rmem_tensor((2,), Int32)
            largest[0] = Int32(-1)
            largest[1] = Int32(-1)
            row = Int64(block) * self.THREADS + Int64(tid)
            if (block == 0) & (tid == 0):
                largest[1] = Int32(tensor1d(selected, Int64(1))[0]) - 1
            if (row > 0) & (row < Int64(main_base)):
                if Int32(tensor1d(selected, Int64(2**62))[row]) != 0:
                    slot = self._slot(
                        swa_pages,
                        starts,
                        swa_offsets,
                        row,
                        swa_capacity,
                        swa_page_rows,
                        swa_page_cols,
                        swa_page_stride,
                        swa_page_size,
                        request_count,
                        "swa",
                    )
                    if slot >= 0:
                        largest[0] = largest_scale_byte(
                            swa_pool, slot * swa_stride + 512, 4, self.limits[0]
                        )
            for source in cutlass.range_constexpr(2):
                value = warp_max(largest[source])
                if tid % 32 == 0:
                    warps[source * (self.THREADS // 32) + tid // 32] = value
            cute.arch.barrier()
            if tid == 0:
                bitmap_t = tensor1d(bitmap, Int64(16))
                for source in cutlass.range_constexpr(2):
                    value = Int32(-1)
                    for warp in cutlass.range_constexpr(self.THREADS // 32):
                        other = Int32(warps[source * (self.THREADS // 32) + warp])
                        if other > value:
                            value = other
                    if value >= 0:
                        word = Int32(self.indices[source] * 8) + (value >> 5)
                        cute.arch.atomic_or(bitmap_t.iterator + word, Int32(1) << (value & 31))

    # ------------------------------------------ K14/K15: staged FP8 KV range

    class StageScalesKernel:
        """Reduce Q's amax and the selected rows' group-scale bitmaps in one launch.

        Blocks ``[0, q_blocks)`` raise Q's amax word, and each following block
        reduces 128 selected ``swa`` and then ``main`` columns of one query into
        that format's bitmap; ``StageScaleFinalizeKernel`` turns them into the
        scale. Both persistent encodings store non-negative group scales in the
        tail of each packed row, so the largest scale byte of the selected rows
        bounds every magnitude staging can decode; reading only that tail costs 32
        or 16 bytes per row. The reduction records which byte values occur rather
        than their maximum, so blocks combine with ``atomic_or``. A BF16 with its
        sign bit cleared orders as an integer, NaN above infinity above every
        finite magnitude, so the integer maximum of Q's sign-cleared halves,
        shifted into FP32 position, is exactly ``torch.linalg.vector_norm(q, inf)``.
        Without row blocks the bitmaps were filled beforehand, so the Q blocks
        skip reading Q when their exponent is at most ``minimum``: the finalize
        then returns ``minimum`` whatever Q's amax is.
        """

        THREADS = 128
        Q_SOURCE = 2

        def __init__(
            self,
            has_swa: bool,
            has_main: bool,
            has_q: bool,
            cached_main: bool = False,
            mapped: bool = False,
            pdl: bool = False,
        ):
            self.has_swa = has_swa
            self.has_main = has_main
            self.has_q = has_q
            self.gated = has_q and not (has_swa or has_main)
            self.cached_main = cached_main
            self.mapped = mapped
            self.pdl = pdl
            # Per format: scale tail offset, its 4-byte words and the exclusive byte limit.
            self.geometry = {}
            for cache_format in ("swa", "main"):
                _, data_bytes, scale_bytes = _format_geometry(cache_format, _STAGE_HEAD_DIM)
                self.geometry[cache_format] = (
                    data_bytes,
                    scale_bytes // 4,
                    _SCALE_BYTE_LIMITS[cache_format],
                )

        @cute.jit
        def __call__(
            self,
            q: cute.Pointer,  # int32 [chunks * 4], two BF16 values per word, 16-byte aligned
            swa_pool: cute.Pointer,
            swa_slots: cute.Pointer,
            main_pool: cute.Pointer,
            main_slots: cute.Pointer,
            bitmap: cute.Pointer,  # int32 [17]: two 8-word bitmaps, then Q's amax bits
            cached: cute.Pointer,  # int32 [8]: a staging group's main bitmap (gated, cached_main)
            chunks: Int32,
            q_blocks: Int32,
            minimum: Int32,
            swa_blocks: Int32,
            swa_column_blocks: Int32,
            swa_width: Int32,
            swa_capacity: Int64,
            swa_stride: Int64,
            swa_rs: Int64,
            swa_cs: Int64,
            main_blocks: Int32,
            main_column_blocks: Int32,
            main_width: Int32,
            main_capacity: Int64,
            main_stride: Int64,
            main_rs: Int64,
            main_cs: Int64,
            requests: cute.Pointer,
            visible: cute.Pointer,
            table: cute.Pointer,
            table_rows: Int32,
            table_cols: Int32,
            table_stride: Int64,
            page_size: Int64,
            max_positions: Int64,
            stream: cuda.CUstream,
        ):
            mapping = None
            if cutlass.const_expr(self.mapped):
                mapping = (
                    requests,
                    visible,
                    (table, table_rows, table_cols, table_stride, page_size, max_positions),
                )
            self.kernel(
                q,
                swa_pool,
                swa_slots,
                main_pool,
                main_slots,
                bitmap,
                cached,
                chunks,
                q_blocks,
                minimum,
                swa_blocks,
                swa_column_blocks,
                swa_width,
                swa_capacity,
                swa_stride,
                swa_rs,
                swa_cs,
                main_column_blocks,
                main_width,
                main_capacity,
                main_stride,
                main_rs,
                main_cs,
                mapping,
            ).launch(
                grid=[q_blocks + swa_blocks + main_blocks, 1, 1],
                block=[self.THREADS, 1, 1],
                stream=stream,
                use_pdl=self.pdl,
            )

        @cute.jit
        def _largest_scale(
            self,
            pool: cute.Pointer,
            slots: cute.Pointer,
            query: Int32,
            column: Int32,
            width: Int32,
            capacity: Int64,
            pool_stride: Int64,
            row_stride: Int64,
            col_stride: Int64,
            scale_base: cutlass.Constexpr,
            words: cutlass.Constexpr,
            limit: cutlass.Constexpr,
            mapping: tuple | None,
        ) -> Int32:
            """Largest bounded group-scale byte of one selected row, or -1."""
            largest = Int32(-1)
            if column < width:
                slot = Int64(
                    tensor1d(slots, Int64(2**62))[
                        Int64(query) * row_stride + Int64(column) * col_stride
                    ]
                )
                if cutlass.const_expr(mapping is not None):
                    requests, visible, table_args = mapping
                    slot = map_logical_slot(
                        slot,
                        Int64(tensor1d(requests, Int64(2**62))[query]),
                        *table_args,
                        visible=Int64(tensor1d(visible, Int64(2**62))[query]),
                    )
                if (slot >= 0) & (slot < capacity):
                    largest = largest_scale_byte(
                        pool, slot * pool_stride + scale_base, words, limit
                    )
            return largest

        @cute.kernel
        def kernel(
            self,
            q: cute.Pointer,
            swa_pool: cute.Pointer,
            swa_slots: cute.Pointer,
            main_pool: cute.Pointer,
            main_slots: cute.Pointer,
            bitmap: cute.Pointer,
            cached: cute.Pointer,
            chunks: Int32,
            q_blocks: Int32,
            minimum: Int32,
            swa_blocks: Int32,
            swa_column_blocks: Int32,
            swa_width: Int32,
            swa_capacity: Int64,
            swa_stride: Int64,
            swa_rs: Int64,
            swa_cs: Int64,
            main_column_blocks: Int32,
            main_width: Int32,
            main_capacity: Int64,
            main_stride: Int64,
            main_rs: Int64,
            main_cs: Int64,
            mapping: tuple | None,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            pdl_prologue(self.pdl, True)
            smem = SmemAllocator()
            warps = smem.allocate_tensor(Int32, cute.make_layout(self.THREADS // 32), 16)
            if cutlass.const_expr(self.gated):
                exponent = smem.allocate_tensor(Int32, cute.make_layout(1), 4)
            b = Int32(block)
            # A bitmap slot, or Q's amax word. -1 is "nothing seen", below every
            # real byte and every sign-cleared Q half.
            source = Int32(self.Q_SOURCE)
            largest = Int32(-1)
            if b < q_blocks:
                if cutlass.const_expr(self.has_q):
                    needed = cutlass.Boolean(True)
                    if cutlass.const_expr(self.gated):
                        # One thread decodes the bitmaps for the whole block.
                        if tid == 0:
                            if cutlass.const_expr(self.cached_main):
                                exponent[0] = staged_exponent(bitmap, cached)
                            else:
                                exponent[0] = staged_exponent(bitmap)
                        cute.arch.barrier()
                        needed = Int32(exponent[0]) > minimum
                    if needed:
                        largest = q_amax_bits(
                            q,
                            b * Int32(self.THREADS) + Int32(tid),
                            chunks,
                            q_blocks * Int32(self.THREADS),
                            4,
                        )
            elif b < q_blocks + swa_blocks:
                source = Int32(_SCALE_SOURCE_SLOTS["swa"])
                if cutlass.const_expr(self.has_swa):
                    index = b - q_blocks
                    largest = self._largest_scale(
                        swa_pool,
                        swa_slots,
                        index // swa_column_blocks,
                        (index % swa_column_blocks) * Int32(self.THREADS) + Int32(tid),
                        swa_width,
                        swa_capacity,
                        swa_stride,
                        swa_rs,
                        swa_cs,
                        *self.geometry["swa"],
                        None,
                    )
            else:
                source = Int32(_SCALE_SOURCE_SLOTS["main"])
                if cutlass.const_expr(self.has_main):
                    index = b - q_blocks - swa_blocks
                    largest = self._largest_scale(
                        main_pool,
                        main_slots,
                        index // main_column_blocks,
                        (index % main_column_blocks) * Int32(self.THREADS) + Int32(tid),
                        main_width,
                        main_capacity,
                        main_stride,
                        main_rs,
                        main_cs,
                        *self.geometry["main"],
                        mapping,
                    )
            largest = warp_max(largest)
            if tid % 32 == 0:
                warps[tid // 32] = largest
            cute.arch.barrier()
            if tid == 0:
                for warp in cutlass.range_constexpr(self.THREADS // 32):
                    other = Int32(warps[warp])
                    if other > largest:
                        largest = other
                bitmap_t = tensor1d(bitmap, Int64(17))
                if source == Int32(self.Q_SOURCE):
                    if largest > 0:
                        cute.arch.atomic_max(bitmap_t.iterator + 16, largest << 16)
                elif largest >= 0:
                    word = source * Int32(8) + (largest >> 5)
                    bit = Int32(1) << (largest & 31)
                    # The words only grow until finalize, so a stale read costs at most
                    # a redundant atomic, while most row blocks find their bit set.
                    if (bitmap_t[word] & bit) == 0:
                        cute.arch.atomic_or(bitmap_t.iterator + word, bit)

    class StageScaleFinalizeKernel:
        """Turn the observed group-scale bitmaps into one power-of-two staging scale.

        The bitmaps and the Q amax word are cleared once read, so the next
        derivation starts from zero without a separate reset.

        ``staged_exponent`` gives the smallest power of two that maps the larger
        decode ceiling onto 448, which puts every staged value inside the E4M3
        range, and a power of two costs no relative precision: E4M3's relative
        error is identical in every binade, so the factor only moves the exponent.

        ``main_bits`` ``"record"`` copies the main bitmap into ``cached`` before it
        is cleared; ``"reuse"`` adds ``cached`` to a main bitmap no row block filled.
        With ``external_q``, ``q_word`` is an int32 word already holding Q's amax
        bits, raised by Q's producer; it joins the bitmap's Q word and is cleared.
        """

        def __init__(
            self,
            has_q: bool,
            q_shared: bool,
            main_bits: str | None = None,
            external_q: bool = False,
            pdl: bool = False,
        ):
            self.has_q = has_q
            self.q_shared = q_shared
            self.main_bits = main_bits
            self.external_q = external_q
            self.pdl = pdl

        @cute.jit
        def __call__(
            self,
            bitmap: cute.Pointer,  # int32 [17]: two 8-word bitmaps, then Q's amax bits
            cached: cute.Pointer,  # int32 [8]: a staging group's main bitmap
            dequant: cute.Pointer,  # float32 [1]
            quant: cute.Pointer,  # float32 [1]
            q_word: cute.Pointer,  # int32 [1]: Q's amax bits (external_q)
            minimum: Int32,
            maximum: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(bitmap, cached, dequant, quant, q_word, minimum, maximum).launch(
                grid=[1, 1, 1], block=[32, 1, 1], stream=stream, use_pdl=self.pdl
            )

        @cute.kernel
        def kernel(
            self,
            bitmap: cute.Pointer,
            cached: cute.Pointer,
            dequant: cute.Pointer,
            quant: cute.Pointer,
            q_word: cute.Pointer,
            minimum: Int32,
            maximum: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            pdl_prologue(self.pdl, False)
            bitmap_t = tensor1d(bitmap, Int64(17))
            cached_t = tensor1d(cached, Int64(8))
            if cutlass.const_expr(self.main_bits == "reuse"):
                if tid < 8:
                    bitmap_t[8 + tid] = bitmap_t[8 + tid] | cached_t[tid]
                cute.arch.barrier()
            if tid == 0:
                if cutlass.const_expr(self.external_q):
                    word = tensor1d(q_word, Int64(1))
                    external = Int32(word[0])
                    if external > Int32(bitmap_t[16]):
                        bitmap_t[16] = external
                    word[0] = Int32(0)
                exponent = staged_exponent(bitmap)
                if cutlass.const_expr(self.has_q):
                    amax = bitcast_u32_f32(Uint32(tensor1d(bitmap, Int64(17))[16]))
                    if amax > Float32(0.0):
                        amax_bits = bitcast_f32_u32(amax)
                        if cutlass.const_expr(self.q_shared):
                            # Sharing one tensor for both dequant scales divides Q by
                            # this scale instead of multiplying it by the reciprocal,
                            # so Q cannot saturate at all: it runs out of resolution
                            # underneath. Its codes land at amax(|Q|) / 2**exponent,
                            # so stopping at floor(log2(amax(|Q|))) keeps Q's amax at
                            # unit magnitude or above and leaves it the whole E4M3
                            # normal range below one -- six binades -- plus the
                            # subnormals under that. A subnormal amax yields -127 and,
                            # once clamped below, simply leaves the unit scale in place.
                            allowed = Int32(amax_bits >> 23) - Int32(127)
                        else:
                            # Q carries the reciprocal scale so that BMM1's dequant
                            # product stays exactly one, which turns Q's own headroom
                            # into a ceiling on the exponent: its codes land at
                            # 2**exponent * amax(|Q|), so the exponent may not exceed
                            # log2(448 / amax(|Q|)). Same 1.75 * 2**8 split as above,
                            # one binade tighter when the amax mantissa exceeds 1.75.
                            # A Q that already saturates on its own yields a negative
                            # bound and, once clamped below, simply leaves the unit
                            # scale in place.
                            allowed = Int32(135) - Int32(amax_bits >> 23)
                            if (amax_bits & Uint32(0x7FFFFF)) > Uint32(0x600000):
                                allowed = allowed - 1
                        if allowed < exponent:
                            exponent = allowed
                if exponent < minimum:
                    exponent = minimum
                if exponent > maximum:
                    exponent = maximum
                tensor1d(dequant, Int64(1))[0] = bitcast_u32_f32(Uint32(exponent + 127) << 23)
                tensor1d(quant, Int64(1))[0] = bitcast_u32_f32(Uint32(Int32(127) - exponent) << 23)
            cute.arch.barrier()
            if tid < 17:
                if cutlass.const_expr(self.main_bits == "record"):
                    if (tid >= 8) & (tid < 16):
                        cached_t[tid - 8] = bitmap_t[tid]
                bitmap_t[tid] = Int32(0)

    class StageQueryKernel:
        """Quantize generation Q with the staging scale (``static_quantize_e4m3_per_tensor``).

        Each value is widened to FP32, multiplied by ``1 / scale`` and rounded with
        ``cvt.rn.satfinite``, byte for byte as the native op. A thread converts
        sixteen values per chunk with two 16-byte loads and one 16-byte store, and
        issues ``UNROLL`` chunks' loads before converting any.
        """

        THREADS = 256
        UNROLL = 2

        @cute.jit
        def __call__(
            self,
            q: cute.Pointer,  # BF16 [chunks * 16], 16-byte aligned
            output: cute.Pointer,  # E4M3 [chunks * 16], 16-byte aligned
            scale: cute.Pointer,  # float32 [1]
            chunks: Int32,
            blocks: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(q, output, scale, chunks, blocks).launch(
                grid=[blocks, 1, 1], block=[self.THREADS, 1, 1], stream=stream
            )

        @cute.kernel
        def kernel(
            self,
            q: cute.Pointer,
            output: cute.Pointer,
            scale: cute.Pointer,
            chunks: Int32,
            blocks: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            factor = div_rn(Float32(1.0), Float32(tensor1d(scale, Int64(1))[0]))
            step = blocks * Int32(self.THREADS)
            words = [cute.make_rmem_tensor((8,), Uint32) for _ in range(self.UNROLL)]
            packed = cute.make_rmem_tensor((4,), Uint32)
            for first in cutlass.range(
                block * Int32(self.THREADS) + Int32(tid), chunks, step * self.UNROLL
            ):
                for u in cutlass.range_constexpr(self.UNROLL):
                    chunk = first + u * step
                    if chunk < chunks:
                        source = cute.make_ptr(
                            Uint32,
                            q.toint() + Int64(chunk) * 32,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        cute.autovec_copy(cute.make_tensor(source, cute.make_layout(8)), words[u])
                for u in cutlass.range_constexpr(self.UNROLL):
                    chunk = first + u * step
                    if chunk < chunks:
                        for word in cutlass.range_constexpr(4):
                            value = Uint32(0)
                            for half in cutlass.range_constexpr(2):
                                bits = words[u][2 * word + half]
                                low = mul_rn(bitcast_u32_f32(bits << 16), factor)
                                high = mul_rn(bitcast_u32_f32(bits & Uint32(0xFFFF0000)), factor)
                                value = value | (f32x2_to_e4m3x2(low, high) << (16 * half))
                            packed[word] = value
                        store_u32x4(output, Int64(chunk) * 16, packed)

    class Namespace:
        pass

    ns = Namespace()
    ns.mul_rn = mul_rn
    ns.fma_rn = fma_rn
    ns.QuantizeScatterKernel = QuantizeScatterKernel
    ns.GatherDequantKernel = GatherDequantKernel
    ns.ScatterPackedKernel = ScatterPackedKernel
    ns.StageCompactKernel = StageCompactKernel
    ns.StageDedupKernel = StageDedupKernel
    ns.StageDequantKernel = StageDequantKernel
    ns.SharedCompactKernel = SharedCompactKernel
    ns.SharedDecodeKernel = SharedDecodeKernel
    ns.SharedRowScaleKernel = SharedRowScaleKernel
    ns.SharedScaleKernel = SharedScaleKernel
    ns.StageScalesKernel = StageScalesKernel
    ns.StageScaleFinalizeKernel = StageScaleFinalizeKernel
    ns.StageQueryKernel = StageQueryKernel
    return ns


# ----------------------------------------------------------------- host side


_MAX_QUANTIZE_JOBS = 4


@dataclasses.dataclass(frozen=True)
class QuantizeJobSpec:
    """Compile-time shape of one row set in a quantize launch."""

    cache_format: str
    head_dim: int
    mode: str  # row | footer | split
    identity: bool = False  # row ``r`` writes destination ``r``; no slot array
    norm: bool = False
    rope_dim: int = 0

    def __post_init__(self):
        group, data_bytes, scale_bytes = _format_geometry(self.cache_format, self.head_dim)
        if self.mode not in ("row", "footer", "split"):
            raise ValueError("Unsupported CSA2 scatter mode")
        if self.rope_dim < 0 or self.rope_dim > self.head_dim or self.rope_dim % 4:
            raise ValueError("CSA2 RoPE width must be a multiple of four channels")
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "group_lanes", group // 4)
        object.__setattr__(self, "data_bytes", data_bytes)
        object.__setattr__(self, "scale_bytes", scale_bytes)
        object.__setattr__(self, "threads", max(32, -(-(self.head_dim // 4) // 32) * 32))


@dataclasses.dataclass(frozen=True)
class QuantizeJob:
    """One row set of a quantize launch; build with ``row_job`` / ``footer_job`` / ``split_job``."""

    spec: QuantizeJobSpec
    pointers: tuple  # pool, scale_pool, slots, values, norm_weight, positions, cos_sin
    rows: int
    capacity: int
    pool_stride: int
    slot_stride: int
    value_stride: int
    page_rows: int
    eps: float
    device: torch.device


def _fusion_args(values: torch.Tensor, norm, rope):
    """Kernel arguments for the optional RMSNorm / RoPE prologue of a quantize launch.

    ``norm`` is ``(weight, eps)`` and ``rope`` is ``(positions, cos_sin, rope_dim)``
    with a float32 ``[positions, rope_dim]`` table (cos half, then sin half).
    Absent stages pass unread placeholders.
    """
    rows, head_dim = values.shape
    weight, eps = (None, 0.0) if norm is None else norm
    positions, cos_sin, rope_dim = (None, None, 0) if rope is None else rope
    if weight is not None:
        _check(weight.dtype == torch.bfloat16 and weight.numel() == head_dim, "norm weight")
        weight = weight.contiguous()
    if rope_dim:
        _check(positions.dtype == torch.int32 and positions.numel() == rows, "rope positions")
        _check(cos_sin.dtype == torch.float32 and cos_sin.shape[-1] == rope_dim, "cos/sin table")
        positions, cos_sin = positions.contiguous(), cos_sin.contiguous()
    return (
        weight is not None,
        int(rope_dim),
        _ptr(values if weight is None else weight, align=2),
        _ptr(values if positions is None else positions, torch.int32),
        _ptr(values if cos_sin is None else cos_sin, torch.float32),
        float(eps),
    )


def row_job(
    pool: torch.Tensor,
    slots: torch.Tensor,
    values: torch.Tensor,
    cache_format: str,
    norm=None,
    rope=None,
) -> QuantizeJob | None:
    """Job publishing BF16 ``values[rows, head_dim]`` into ``pool[slot, :]`` rows; padding slots skip.

    ``norm=(weight, eps)`` normalizes and ``rope=(positions, cos_sin, rope_dim)``
    rotates the rows inside the launch. ``None`` when there are no rows.
    """
    rows, head_dim = values.shape
    _, data_bytes, scale_bytes = _format_geometry(cache_format, head_dim)
    if rows == 0:
        return None
    _check(pool.ndim == 2 and pool.dtype == torch.uint8 and pool.stride(1) == 1, "pool")
    _check(pool.shape[1] == data_bytes + scale_bytes, "CSA2 pool row width")
    _check(values.dtype == torch.bfloat16 and values.stride(1) == 1, "values")
    _check(slots.ndim == 1 and slots.dtype == torch.int64 and slots.numel() == rows, "slots")
    has_norm, rope_dim, weight_ptr, positions_ptr, cos_sin_ptr, eps = _fusion_args(
        values, norm, rope
    )
    return QuantizeJob(
        QuantizeJobSpec(cache_format, head_dim, "row", norm=has_norm, rope_dim=rope_dim),
        (
            _ptr(pool, align=1),
            _ptr(pool, align=1),
            _ptr(slots, align=8),
            _ptr(values, align=2),
            weight_ptr,
            positions_ptr,
            cos_sin_ptr,
        ),
        int(rows),
        int(pool.shape[0]),
        int(pool.stride(0)),
        int(slots.stride(0)),
        int(values.stride(0)),
        1,
        eps,
        pool.device,
    )


def footer_job(
    pages: torch.Tensor,
    slots: torch.Tensor,
    values: torch.Tensor,
    page_rows: int,
    norm=None,
    rope=None,
) -> QuantizeJob | None:
    """Job publishing BF16 128D index rows into native page-footer pages (optionally norm + RoPE)."""
    rows, head_dim = values.shape
    _, data_bytes, scale_bytes = _format_geometry("index", head_dim)
    if rows == 0:
        return None
    flat = pages.view(-1)
    page_bytes = page_rows * (data_bytes + scale_bytes)
    _check(pages.dtype == torch.uint8 and flat.numel() % page_bytes == 0, "index pages")
    _check(values.dtype == torch.bfloat16 and values.stride(1) == 1, "values")
    _check(slots.ndim == 1 and slots.dtype == torch.int64 and slots.numel() == rows, "slots")
    has_norm, rope_dim, weight_ptr, positions_ptr, cos_sin_ptr, eps = _fusion_args(
        values, norm, rope
    )
    return QuantizeJob(
        QuantizeJobSpec("index", head_dim, "footer", norm=has_norm, rope_dim=rope_dim),
        (
            _ptr(flat, align=1),
            _ptr(flat, align=1),
            _ptr(slots, align=8),
            _ptr(values, align=2),
            weight_ptr,
            positions_ptr,
            cos_sin_ptr,
        ),
        int(rows),
        flat.numel() // page_bytes * page_rows,
        0,
        int(slots.stride(0)),
        int(values.stride(0)),
        int(page_rows),
        eps,
        pages.device,
    )


def split_job(values: torch.Tensor, data: torch.Tensor, scales: torch.Tensor) -> QuantizeJob | None:
    """Job quantizing ``values[rows, 128]`` into separate ``data[rows, 64]`` and ``scales[rows, 4]`` bytes."""
    rows, head_dim = values.shape
    _, data_bytes, scale_bytes = _format_geometry("index", head_dim)
    if rows == 0:
        return None
    _check(values.dtype == torch.bfloat16 and values.stride(1) == 1, "values")
    _check(
        data.dtype == torch.uint8 and data.is_contiguous() and data.shape == (rows, data_bytes),
        "data",
    )
    _check(
        scales.dtype == torch.uint8
        and scales.is_contiguous()
        and scales.shape == (rows, scale_bytes),
        "scale",
    )
    # Row ``r`` quantizes into row ``r``: the identity specialization reads no
    # slot array, so the slot / norm / rope pointers are unread placeholders.
    values_ptr = _ptr(values, align=2)
    return QuantizeJob(
        QuantizeJobSpec("index", head_dim, "split", identity=True),
        (
            _ptr(data, align=1),
            _ptr(scales, align=1),
            _ptr(data, align=1),
            values_ptr,
            values_ptr,
            _ptr(values, torch.int32),
            _ptr(values, torch.float32),
        ),
        int(rows),
        int(rows),
        0,
        1,
        int(values.stride(0)),
        1,
        0.0,
        values.device,
    )


def quantize_scatter_jobs(jobs) -> None:
    """Run up to four quantize jobs (``None`` entries skipped) as one launch."""
    jobs = [job for job in jobs if job is not None]
    if not jobs:
        return
    device = jobs[0].device
    _check(all(job.device == device for job in jobs), "quantize jobs must share a device")
    ns = _cache_kernels()
    args = []
    for slot in range(_MAX_QUANTIZE_JOBS):
        # Absent job slots reuse the first job's pointers with zero rows.
        job = jobs[slot] if slot < len(jobs) else jobs[0]
        rows = job.rows if slot < len(jobs) else 0
        args += [
            *job.pointers,
            rows,
            job.capacity,
            job.pool_stride,
            job.slot_stride,
            job.value_stride,
            job.page_rows,
            job.eps,
        ]
    pdl = _pdl()
    _launch(
        ("quantize", tuple(job.spec for job in jobs), pdl, device.index),
        lambda: ns.QuantizeScatterKernel([job.spec for job in jobs], pdl),
        *args,
        _stream(device),
    )


def quantize_scatter_rows(pool, slots, values, cache_format: str, *, norm=None, rope=None):
    """Publish BF16 ``values[rows, head_dim]`` into ``pool[slot, :]`` rows (see ``row_job``)."""
    quantize_scatter_jobs([row_job(pool, slots, values, cache_format, norm, rope)])
    return pool


def quantize_scatter_index_pages(pages, slots, values, page_rows: int, *, norm=None, rope=None):
    """Publish BF16 128D index rows into native page-footer pages (see ``footer_job``)."""
    quantize_scatter_jobs([footer_job(pages, slots, values, page_rows, norm, rope)])
    return pages


def quantize_index_queries(values: torch.Tensor, data: torch.Tensor, scales: torch.Tensor) -> None:
    """Quantize ``values[rows, 128]`` into separate data and scale bytes (see ``split_job``)."""
    quantize_scatter_jobs([split_job(values, data, scales)])


def gather_dequant_rows(
    pool: torch.Tensor, slots: torch.Tensor, head_dim: int, cache_format: str
) -> torch.Tensor:
    """Gather row-strided CSA2 bytes into BF16 ``[*slots.shape, head_dim]``."""
    output = torch.empty((*slots.shape, head_dim), dtype=torch.bfloat16, device=pool.device)
    if slots.numel() == 0:
        return output
    _check(pool.ndim == 2 and pool.dtype == torch.uint8 and pool.stride(1) == 1, "pool")
    _check(slots.dtype in (torch.int32, torch.int64) and slots.ndim in (1, 2), "slots")
    if slots.dtype != torch.int64:
        slots = slots.long()
    if slots.ndim == 1:
        columns, row_stride, col_stride = slots.shape[0], 0, slots.stride(0)
    else:
        columns, row_stride, col_stride = slots.shape[1], slots.stride(0), slots.stride(1)
    ns = _cache_kernels()
    _launch(
        ("gather", cache_format, head_dim, pool.device.index),
        lambda: ns.GatherDequantKernel(cache_format, head_dim),
        _ptr(pool, align=1),
        _ptr(slots, align=8),
        _ptr(output, align=8),
        int(slots.numel()),
        int(pool.shape[0]),
        int(pool.stride(0)),
        int(columns),
        int(row_stride),
        int(col_stride),
        _stream(pool.device),
    )
    return output


def scatter_packed_rows(pool: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """Copy packed byte rows into ``pool[slot, :]``; padding slots never write."""
    if rows.shape[0] == 0:
        return
    _check(
        pool.stride(1) == 1 and rows.stride(1) == 1, "CSA2 packed cache columns must be contiguous"
    )
    _check(rows.shape[1] == pool.shape[1], "CSA2 packed rows must match the pool width")
    _check(
        slots.ndim == 1 and slots.dtype == torch.int64 and slots.numel() == rows.shape[0], "slots"
    )
    ns = _cache_kernels()
    width = int(rows.shape[1])
    _launch(
        ("scatter_rows", False, width, pool.device.index),
        lambda: ns.ScatterPackedKernel(False, width, 0),
        _ptr(pool, align=1),
        _ptr(slots, align=8),
        _ptr(rows, align=1),
        int(rows.shape[0]),
        int(pool.shape[0]),
        int(pool.stride(0)),
        int(slots.stride(0)),
        int(rows.stride(0)),
        1,
        _stream(pool.device),
    )


def scatter_packed_index_rows(
    flat: torch.Tensor,
    slots: torch.Tensor,
    rows: torch.Tensor,
    page_rows: int,
    data_bytes: int,
    scale_bytes: int,
) -> None:
    """Scatter packed ``[n, data + scale]`` index rows into flat page-footer pages."""
    if rows.shape[0] == 0:
        return
    _check(rows.stride(1) == 1 and rows.shape[1] == data_bytes + scale_bytes, "packed index rows")
    _check(
        slots.ndim == 1 and slots.dtype == torch.int64 and slots.numel() == rows.shape[0], "slots"
    )
    page_bytes = page_rows * (data_bytes + scale_bytes)
    ns = _cache_kernels()
    _launch(
        ("scatter_rows", True, data_bytes, scale_bytes, flat.device.index),
        lambda: ns.ScatterPackedKernel(True, data_bytes, scale_bytes),
        _ptr(flat, align=1),
        _ptr(slots, align=8),
        _ptr(rows, align=1),
        int(rows.shape[0]),
        flat.numel() // page_bytes * page_rows,
        0,
        int(slots.stride(0)),
        int(rows.stride(0)),
        int(page_rows),
        _stream(flat.device),
    )


_CONSTANT_ZEROS: dict[tuple, torch.Tensor] = {}


def _constant_zeros(device, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Zero tensor whose contents never change, retained for the process lifetime.

    Staging launches pass these as placeholders on every layer; a retained
    constant costs one fill per distinct shape instead of one per launch and
    stays valid for graphs that captured its address.
    """
    key = (str(device), tuple(shape), dtype)
    zeros = _CONSTANT_ZEROS.get(key)
    if zeros is None:
        zeros = _CONSTANT_ZEROS[key] = torch.zeros(shape, dtype=dtype, device=device)
    return zeros


def _page_mapping_args(
    main_mapping: tuple[torch.Tensor, int, int, torch.Tensor, torch.Tensor],
    count: int,
    device: torch.device,
) -> tuple[tuple, torch.Tensor, torch.Tensor]:
    """Validate and lower the live page mapping shared by staging and scale reduction."""
    table, page_size, max_positions, requests, visible = main_mapping
    _check(
        table.dtype == torch.int32
        and table.ndim == 2
        and table.device == device
        and (table.shape[1] <= 1 or table.stride(1) == 1),
        "table",
    )
    _check(page_size > 0, "CSA2 mapping page size must be positive")
    _check(
        requests.device == device and visible.device == device,
        "CSA2 mapping tensors must share the pool device",
    )
    requests = _as_int64(requests)
    visible = _as_int64(visible)
    _check(requests.numel() == count and visible.numel() == count, "CSA2 mapping rows")
    table_args = (
        _ptr(table),
        int(table.shape[0]),
        int(table.shape[1]),
        int(table.stride(0) if table.shape[0] > 1 else table.shape[1]),
        int(page_size),
        int(max_positions),
    )
    return table_args, requests, visible


def _mapping_args(main_mapping, main_slots, main_logical, count: int, device):
    """Resolve the main-selection source for the staging kernels.

    Returns ``(premapped, logical, slots, slot_strides, table_args, requests, visible)``.
    With ``main_mapping`` the logical positions are mapped in-kernel through
    ``(page_table, page_size, max_positions, requests, visible)``; otherwise
    ``main_slots`` already hold physical rows.
    """
    identity = _constant_zeros(device, (1, 1), torch.int32)
    if main_mapping is None:
        _check(main_slots is not None, "CSA2 staging requires physical main slots or a mapping")
        slots = main_slots if main_slots.dtype == torch.int64 else main_slots.long()
        logical = main_logical if main_logical is not None else slots
        table_args = (_ptr(identity), 1, 1, 1, 1, 2**62)
        zeros = _constant_zeros(device, (count,), torch.int64)
        return (
            True,
            logical,
            slots,
            (slots.stride(0), slots.stride(1) if slots.shape[1] else 1),
            table_args,
            zeros,
            zeros,
        )
    table_args, requests, visible = _page_mapping_args(main_mapping, count, device)
    _check(
        main_slots is not None
        and main_slots.dtype == torch.int64
        and main_slots.shape == main_logical.shape
        and (main_slots.shape[1] <= 1 or main_slots.stride(1) == 1),
        "slot scratch",
    )
    return False, main_logical, main_slots, (main_slots.stride(0), 1), table_args, requests, visible


# The staging pools are fixed at the MLA latent width, as every other staging
# launch in this module assumes.
_STAGE_HEAD_DIM = 512

# Bitmap slot per persistent format; ``StageScaleFinalizeKernel`` decodes slot 0
# as UE8M0 exponents and slot 1 as E4M3 codes, so the order is part of its ABI.
_SCALE_SOURCE_SLOTS = {"swa": 0, "main": 1}

# Byte values a group scale may take, exclusive: 255 is the UE8M0 infinity and
# 0x7F is E4M3 NaN (with every byte above it a negative scale). None of them can
# be bounded by a finite staging scale, so they must not drive the reduction.
_SCALE_BYTE_LIMITS = {"swa": 255, "main": 127}


def _stage_q_words(q: torch.Tensor) -> torch.Tensor:
    """``q``'s BF16 pairs as 16-byte aligned int32 words for the staging Q amax."""
    _check(
        q.dtype == torch.bfloat16 and q.numel() % 8 == 0,
        "CSA2 staging scales need a BF16 Q with a multiple of eight elements",
    )
    words = q.contiguous().reshape(-1).view(torch.int32)
    _check(words.data_ptr() % 16 == 0, "CSA2 staging Q amax needs a 16-byte aligned Q")
    return words


def derive_stage_kv_scales(
    sources,
    bitmap: torch.Tensor,
    dequant: torch.Tensor,
    quant: torch.Tensor,
    *,
    main_mapping: tuple | None = None,
    q: torch.Tensor | None = None,
    q_shared: bool = False,
    minimum: int = 0,
    maximum: int = 126,
    main_bits: torch.Tensor | None = None,
    reuse_main: bool = False,
    q_word: torch.Tensor | None = None,
    q_ready: torch.cuda.Event | None = None,
) -> None:
    """Fill ``dequant``/``quant`` with a range-aware power-of-two staging KV scale.

    ``sources`` holds the ``(pool, slots, cache_format)`` triples whose rows the
    staging pass will decode. Staging expands the persistent per-group scales back
    to real magnitudes, so a unit per-tensor scale saturates every channel above
    448; the bound here is read from those same group-scale bytes, which makes it
    track the forward's actual range instead of a calibrated constant. ``bitmap``
    is a 17-word int32 scratch holding one presence bitmap per format and Q's
    amax bits, zeroed unless a caller filled the bitmaps beforehand; the finalize
    kernel clears it again once read. Both outputs are written in place on the
    device, so CUDA Graph replay observes the live value. When ``main_mapping``
    is present, MAIN slots are logical positions; the reduction applies the same
    page and visibility mapping as staging.

    Duplicated or over-covered selections are harmless: the reduction is a maximum
    over an exact per-row ceiling, so any superset of the decoded rows still yields
    a scale that cannot saturate.

    ``q`` is the BF16 query whose ``amax(|Q|)`` caps the exponent, and ``q_shared``
    says which way Q carries the scale. By default Q takes the reciprocal to keep BMM1's
    dequant product at one, so the cap keeps Q from saturating in exchange; with
    ``q_shared`` Q is divided by this very scale, so the cap keeps Q's own amax at
    unit magnitude or above instead. ``q`` is read only when the rows need more than
    ``minimum`` if there are no rows to reduce here or, for generation, if Q is larger
    than half of L2. ``minimum`` defaults to zero because only the saturating end is a
    correctness problem: a forward that needs no extra range keeps the unit scale, byte
    for byte.

    ``main_bits`` (int32 [8]) records the main bitmap of this derivation, or with
    ``reuse_main`` stands in for the main source, whose rows are then not read.

    ``q_word`` (int32, instead of ``q``) already holds the FP32 bits of Q's amax,
    raised by whatever produced Q; the finalize consumes and clears it. The current
    stream waits for ``q_ready`` (the event of Q's producer) only before that finalize,
    so the row reduction does not wait for Q.
    """
    _check(minimum <= maximum, "CSA2 staging scale bounds must not be inverted")
    _check(
        q_word is None
        or (
            q is None
            and q_word.dtype == torch.int32
            and q_word.numel() >= 1
            and q_word.is_contiguous()
            and q_word.device == bitmap.device
        ),
        "CSA2 staging Q amax word must be a device int32 used instead of Q",
    )
    _check(q_ready is None or q_word is not None, "CSA2 staging waits for Q only with its word")
    _check(
        bitmap.dtype == torch.int32 and bitmap.is_contiguous() and bitmap.numel() >= 17,
        "CSA2 staging scale bitmap must be 17 contiguous int32 words",
    )
    for buffer in (dequant, quant):
        _check(
            buffer.dtype == torch.float32 and buffer.numel() == 1 and buffer.is_contiguous(),
            "CSA2 staging scales must be single contiguous float32 elements",
        )
    device = bitmap.device
    ns = _cache_kernels()
    threads = ns.StageScalesKernel.THREADS
    external_q = q_word is not None
    has_q = (q is not None and q.numel() > 0) or external_q
    words = bitmap
    chunks = q_blocks = 0
    if has_q and not external_q:
        words = _stage_q_words(q)
        chunks = words.numel() // 4
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        q_blocks = min(8 * sm_count, ceil_div(chunks, threads))
    placeholder = (
        _ptr(bitmap, align=1),
        _ptr(_constant_zeros(device, (1, 1), torch.int64), align=8),
    )
    args = {cache_format: (*placeholder, 0, 1, 0, 0, 0, 0, 0) for cache_format in ("swa", "main")}
    identity = _constant_zeros(device, (1, 1), torch.int32)
    unused_rows = _constant_zeros(device, (1,), torch.int64)
    unmapped_args = (
        _ptr(unused_rows, align=8),
        _ptr(unused_rows, align=8),
        _ptr(identity),
        1,
        1,
        1,
        1,
        2**62,
    )
    mapping_args = unmapped_args
    mapped = False
    for pool, slots, cache_format in sources:
        if reuse_main and cache_format == "main":
            continue
        _, data_bytes, scale_bytes = _format_geometry(cache_format, _STAGE_HEAD_DIM)
        _check(
            pool.ndim == 2
            and pool.shape[1] == data_bytes + scale_bytes
            and pool.stride(1) == 1
            and pool.data_ptr() % 4 == 0
            and pool.stride(0) % 4 == 0,
            "CSA2 staging scale sources must be 4-byte aligned packed rows with unit column stride",
        )
        rows = int(slots.shape[0])
        width = int(slots.shape[1]) if slots.ndim > 1 else 0
        if not rows or not width:
            continue
        if cache_format == "main" and main_mapping is not None:
            table_args, requests, visible = _page_mapping_args(main_mapping, rows, device)
            mapped = True
            mapping_args = (_ptr(requests, align=8), _ptr(visible, align=8), *table_args)
        column_blocks = -(-width // threads)
        args[cache_format] = (
            _ptr(pool, align=1),
            _ptr(slots, align=8 if slots.dtype == torch.int64 else 4),
            rows * column_blocks,
            column_blocks,
            width,
            int(pool.shape[0]),
            int(pool.stride(0)),
            int(slots.stride(0)),
            int(slots.stride(1)),
        )
    has_swa, has_main = (args[cache_format][2] > 0 for cache_format in ("swa", "main"))
    swa_pool, swa_slots, *swa_geometry = args["swa"]
    main_pool, main_slots, *main_geometry = args["main"]
    # A generation Q larger than half of L2 is read after the rows, only if they need more
    # than ``minimum``.
    gate = (
        has_q
        and not external_q
        and not q_shared
        and (has_swa or has_main or reuse_main)
        and 2 * q.numel() * q.element_size()
        > torch.cuda.get_device_properties(device).L2_cache_size
    )
    row_q = has_q and not gate and not external_q
    cached = main_bits is not None and reuse_main
    pdl = _pdl()
    if (q_blocks if row_q else 0) + swa_geometry[0] + main_geometry[0]:
        q_cached = cached and not (has_swa or has_main)
        _launch(
            ("stage_scales", has_swa, has_main, mapped, row_q, q_cached, pdl, device.index),
            lambda: ns.StageScalesKernel(
                has_swa, has_main, row_q, q_cached, mapped=mapped, pdl=pdl
            ),
            _ptr(words, align=16),
            swa_pool,
            swa_slots,
            main_pool,
            main_slots,
            _ptr(bitmap),
            _ptr(main_bits if q_cached else bitmap),
            chunks if row_q else 0,
            q_blocks if row_q else 0,
            int(minimum),
            *swa_geometry,
            *main_geometry,
            *mapping_args,
            _stream(device),
        )
    if gate:
        pool, slots, *geometry = (*placeholder, 0, 1, 0, 0, 0, 0, 0)
        _launch(
            ("stage_scales", False, False, False, True, cached, pdl, device.index),
            lambda: ns.StageScalesKernel(False, False, True, cached, pdl=pdl),
            _ptr(words, align=16),
            pool,
            slots,
            pool,
            slots,
            _ptr(bitmap),
            _ptr(main_bits if cached else bitmap),
            chunks,
            q_blocks,
            int(minimum),
            *geometry,
            *geometry,
            *unmapped_args,
            _stream(device),
        )
    if q_ready is not None:
        q_ready.wait()
    mode = None if main_bits is None else "reuse" if reuse_main else "record"
    _launch(
        ("stage_scale_finalize", has_q, q_shared, mode, external_q, pdl, device.index),
        lambda: ns.StageScaleFinalizeKernel(has_q, q_shared, mode, external_q, pdl),
        _ptr(bitmap),
        _ptr(bitmap if main_bits is None else main_bits),
        _ptr(dequant),
        _ptr(quant),
        _ptr(q_word if external_q else bitmap, align=4),
        int(minimum),
        int(maximum),
        _stream(device),
    )


def quantize_stage_query(q: torch.Tensor, scale: torch.Tensor, output: torch.Tensor) -> None:
    """Write E4M3 ``q`` to ``output`` exactly as ``static_quantize_e4m3_per_tensor(q, scale)[0]``,
    one launch."""
    _check(
        q.dtype == torch.bfloat16
        and q.is_contiguous()
        and q.numel() % 16 == 0
        and q.data_ptr() % 16 == 0,
        "CSA2 staging Q quantization needs a contiguous 16-byte aligned BF16 Q",
    )
    chunks = q.numel() // 16
    if chunks == 0:
        return
    ns = _cache_kernels()
    kernel = ns.StageQueryKernel
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    blocks = min(8 * sm_count, -(-chunks // (kernel.THREADS * kernel.UNROLL)))
    _launch(
        ("stage_query", q.device.index),
        kernel,
        _ptr(q, torch.int32, align=16),
        _ptr(output, torch.uint8, align=16),
        _ptr(scale),
        chunks,
        blocks,
        _stream(q.device),
    )


def stage_selected_rows(
    swa_pool,
    swa_slots,
    main_pool,
    main_slots,
    metadata,
    scratch,
    *,
    compact=True,
    main_logical=None,
    main_mapping=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    q_scale_quant_orig=None,
    softmax_scale=512**-0.5,
    dedup=False,
):
    """Compact selections (once per staging group) and decode them into the staging pools.

    Main selections are either physical ``main_slots`` or ``main_logical``
    positions resolved in-kernel through ``main_mapping`` (``page_table``,
    ``page_size``, ``max_positions``, ``requests``, ``visible``), exactly as
    ``CSA2TrtllmMetadata.global_slot_tile`` with visibility. ``scratch`` holds
    the group's ``tags``/``row_slots``/``counts`` (and, in the mapped form, ``main_slots``
    receives the physical rows); ``compact=False`` reuses them from an
    earlier layer of the same group in this forward. With ``dedup`` and mapped
    main selections, a query's rows that its request's anchor query also stages
    reference the anchor's rows (``scratch["refs"]``) instead of being staged;
    only consumers that read rows through the published indices may ask for it.
    """
    count = metadata.num_tokens
    capacity = metadata.num_sparse_topk
    fp8 = metadata.swa_pool.dtype == torch.float8_e4m3fn
    device = swa_pool.device
    has_main = main_slots is not None or main_logical is not None
    if not has_main:
        main_pool = swa_pool
        main_slots = swa_slots[:, :0]
        main_logical = None
        main_mapping = None
    premapped, logical, slots, (slot_rs, slot_cs), table_args, requests, visible = _mapping_args(
        main_mapping, main_slots, main_logical, count, device
    )
    main_width = int(slots.shape[1]) if premapped else int(logical.shape[1])
    if swa_slots.dtype != torch.int64:
        swa_slots = swa_slots.long()
    swa_width = int(swa_slots.shape[1])
    tags, counts, row_slots = scratch["tags"], scratch["counts"], scratch["row_slots"]
    dedup = bool(dedup and has_main and not premapped)
    refs = scratch["refs"] if dedup else tags
    own_blocks = scratch["own_blocks"] if dedup else counts
    _check(capacity <= tags.shape[1] and capacity <= 4096, "staging capacity")
    ns = _cache_kernels()
    placeholder = metadata.prepared_counter
    dequant = kv_scale_quant_orig if fp8 else placeholder
    quant = kv_scale_orig_quant if fp8 else placeholder
    q_dequant = q_scale_quant_orig if fp8 else placeholder
    bmm1 = metadata.mla_bmm1_scale if fp8 else placeholder
    bmm2 = metadata.mla_bmm2_scale if fp8 else placeholder
    logical_int64 = logical.dtype == torch.int64
    table_ptr, table_rows, table_cols, table_stride, page_size, max_positions = table_args
    if compact:
        _launch(
            ("stage_compact", fp8, has_main, premapped, device.index),
            lambda: ns.StageCompactKernel(fp8, has_main, premapped),
            _ptr(swa_slots, align=8),
            _ptr(logical, align=8 if logical_int64 else 4),
            _ptr(slots, align=8),
            _ptr(requests, align=8),
            _ptr(visible, align=8),
            table_ptr,
            _ptr(tags),
            _ptr(row_slots),
            _ptr(counts),
            int(count),
            swa_width,
            main_width,
            int(swa_slots.stride(0)),
            int(swa_slots.stride(1)) if swa_width > 0 else 1,
            int(logical.stride(0)) if main_width > 0 else 0,
            int(logical.stride(1)) if main_width > 0 else 1,
            int(slot_rs),
            int(slot_cs),
            int(swa_pool.shape[0]),
            int(main_pool.shape[0]),
            table_rows,
            table_cols,
            table_stride,
            page_size,
            max_positions,
            int(capacity),
            _stream(device),
        )
        if dedup:
            _launch(
                ("stage_dedup", int(capacity), main_width, device.index),
                lambda: ns.StageDedupKernel(int(capacity), main_width),
                _ptr(swa_slots, align=8),
                _ptr(slots, align=8),
                _ptr(requests, align=8),
                _ptr(tags),
                _ptr(counts),
                _ptr(refs),
                _ptr(own_blocks),
                int(count),
                swa_width,
                int(swa_slots.stride(0)),
                int(swa_slots.stride(1)) if swa_width > 0 else 1,
                int(slot_rs),
                int(slot_cs),
                _stream(device),
            )
    extra_rows = int(metadata.extra_pool.shape[1])
    _check(
        swa_pool.data_ptr() % 16 == 0
        and swa_pool.stride(0) % 16 == 0
        and main_pool.data_ptr() % 8 == 0
        and main_pool.stride(0) % 8 == 0,
        "CSA2 staging needs 16-byte aligned SWA rows and 8-byte aligned main rows",
    )
    # A warp stages rows_per_cta / 4 rows. Batch eight rows per warp only when the grid
    # still covers every SM several times over; small decode batches keep one row per warp
    # for latency.
    rows = 128 + extra_rows
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    rows_per_cta = 32 if int(count) * -(-rows // 32) >= 8 * sm_count else 4
    # The layers of a staging group stage the same main rows. While this tile's pools
    # still hold the rows an earlier layer of the group staged in this forward, only
    # the first 128 rows, which hold every SWA row, are staged again.
    token = (id(scratch), scratch.get("serial"))
    reuse = has_main and not compact and metadata.__dict__.get("_csa2_staged_group") == token
    metadata._csa2_staged_group = token
    staged, slot = placeholder, 0
    if fp8:
        staged, slot = metadata._csa2_staged_scales, metadata.__dict__.get("_csa2_staged_slot", 0)
        metadata._csa2_staged_slot = 1 - slot
    row_grid = -(-(128 if reuse else rows) // rows_per_cta)
    # With dedup, a CTA takes a row block of four queries while that grid still covers
    # every SM four times over.
    grouped = dedup and rows_per_cta == 32 and -(-int(count) // 4) * row_grid >= 4 * sm_count
    _launch(
        ("stage_dequant", fp8, rows_per_cta, dedup, reuse, grouped, device.index),
        lambda: ns.StageDequantKernel(fp8, rows_per_cta, dedup, reuse, grouped),
        _ptr(swa_pool, align=1),
        _ptr(main_pool, align=1),
        _ptr(swa_slots, align=8),
        _ptr(row_slots),
        _ptr(tags),
        _ptr(counts),
        _ptr(refs),
        _ptr(metadata.swa_pool, align=8),
        _ptr(metadata.extra_pool, align=8),
        _ptr(quant),
        _ptr(staged),
        _ptr(own_blocks),
        _ptr(metadata.prepared_lens),
        _ptr(metadata.prepared_indices),
        _ptr(metadata.prepared_counter),
        _ptr(q_dequant),
        _ptr(dequant),
        _ptr(bmm1),
        _ptr(bmm2),
        int(count),
        rows,
        row_grid,
        slot,
        swa_width,
        main_width,
        int(swa_slots.stride(0)),
        int(swa_slots.stride(1)) if swa_width > 0 else 1,
        int(swa_pool.stride(0)),
        int(main_pool.stride(0)),
        int(swa_pool.shape[0]),
        int(main_pool.shape[0]),
        int(capacity),
        extra_rows,
        float(softmax_scale),
        _stream(device),
    )


def stage_shared_rows(
    swa_pool,
    swa_slots,
    main_pool,
    main_slots,
    main_logical,
    metadata,
    plan,
    *,
    main_mapping=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    q_scale_quant_orig=None,
    softmax_scale=512**-0.5,
    scale=None,
    compact=True,
    parity=0,
):
    """Decode selected logical rows once into a shared native KV bank.

    ``plan`` supplies per-request logical domains/page tables and per-query
    request/position vectors. Main selections are physical ``main_slots`` plus
    their ``main_logical`` positions, or logical positions resolved through
    ``main_mapping``. Outputs preserve selection order and duplicates.
    ``compact=False`` keeps the bank indices, SWA row marks and main scale fold
    of the previous call, made for a layer of the same staging group in this
    forward, and decodes those rows from this layer's pools; main rows the bank
    already holds at this layer's scale are kept. ``parity`` alternates over a
    group's calls.

    ``scale`` is ``(q, q_shared, minimum, maximum)`` when the FP8 staging scales
    are still to be derived: the marked SWA rows and the selected main rows the
    compaction folded, then Q's amax when they need more than ``minimum``, are reduced before
    decode into ``metadata``'s staging-scale buffers.
    """
    count = metadata.num_tokens
    capacity = metadata.num_sparse_topk
    bank = metadata.shared_pool
    selected = metadata._csa2_shared_selected
    device = swa_pool.device
    _check(
        bank.shape == (1 + plan.swa_rows + plan.main_rows, 512) and bank.is_contiguous(),
        "shared bank",
    )
    _check(
        bank.dtype in (torch.bfloat16, torch.float8_e4m3fn),
        "CSA2 shared bank requires BF16 or E4M3",
    )
    _check(bank.shape[0] <= 2**31 - 1, "CSA2 shared bank exceeds native int32 sparse indices")
    _check(
        selected.shape == (bank.shape[0],) and selected.dtype == torch.int32,
        "shared selection mask",
    )
    fp8 = bank.dtype == torch.float8_e4m3fn
    if fp8:
        _check(
            kv_scale_orig_quant is not None
            and kv_scale_quant_orig is not None
            and q_scale_quant_orig is not None,
            "FP8 scales",
        )
    has_main = main_slots is not None or main_logical is not None
    if not has_main:
        main_pool = swa_pool
    ns = _cache_kernels()
    placeholder = metadata.prepared_counter
    q_dequant = q_scale_quant_orig if fp8 else placeholder
    quant = kv_scale_orig_quant if fp8 else placeholder
    _check(
        swa_pool.data_ptr() % 16 == 0
        and swa_pool.stride(0) % 16 == 0
        and main_pool.data_ptr() % 8 == 0
        and main_pool.stride(0) % 8 == 0,
        "CSA2 shared staging needs 16-byte aligned SWA rows and 8-byte aligned main rows",
    )
    for pages in (plan.swa_pages, plan.main_pages):
        if pages is not None:
            _check(
                pages.dtype == torch.int32
                and pages.ndim == 2
                and (pages.shape[1] <= 1 or pages.stride(1) == 1),
                "pages",
            )
    swa_pages = plan.swa_pages
    main_pages = plan.main_pages if plan.main_rows else swa_pages

    def page_args(pages):
        return (
            int(pages.shape[0]),
            int(pages.shape[1]),
            int(pages.stride(0)) if pages.shape[0] > 1 else int(pages.shape[1]),
        )

    row_args = (
        _ptr(swa_pool, align=1),
        _ptr(main_pool, align=1),
        _ptr(swa_pages),
        _ptr(main_pages),
        _ptr(plan.swa_starts, align=8),
        _ptr(plan.swa_offsets, align=8),
        _ptr(plan.main_offsets, align=8),
        _ptr(selected),
    )
    geometry_args = (
        int(bank.shape[0]),
        int(1 + plan.swa_rows),
        int(swa_pool.shape[0]),
        int(swa_pool.stride(0)),
        int(main_pool.shape[0]),
        int(main_pool.stride(0)),
        *page_args(swa_pages),
        int(plan.swa_page_size),
        *page_args(main_pages),
        int(plan.main_page_size),
        int(plan.num_requests),
    )
    if compact:
        if not has_main:
            main_slots = swa_slots[:, :0]
            main_logical = None
            main_mapping = None
        if main_mapping is not None and main_slots is None:
            main_slots = torch.empty(main_logical.shape, dtype=torch.int64, device=device)
        premapped, logical, slots, (slot_rs, slot_cs), table_args, requests, visible = (
            _mapping_args(main_mapping, main_slots, main_logical, count, device)
        )
        if swa_slots.dtype != torch.int64:
            swa_slots = swa_slots.long()
        main_width = int(logical.shape[1])
        swa_width = int(swa_slots.shape[1])
        logical_int64 = logical.dtype == torch.int64
        table_ptr, table_rows, table_cols, table_stride, page_size, max_positions = table_args
        # SWA marks and the main fold of the previous compaction must not leak into
        # this one; the main rows' words are rewritten before the compaction folds them.
        selected[: 1 + plan.swa_rows].zero_()
        if has_main and plan.main_rows:
            _launch(
                ("shared_row_scale", device.index),
                ns.SharedRowScaleKernel,
                _ptr(main_pool, align=1),
                _ptr(main_pages),
                _ptr(plan.swa_starts, align=8),
                _ptr(plan.main_offsets, align=8),
                _ptr(selected),
                int(bank.shape[0]),
                int(1 + plan.swa_rows),
                int(main_pool.shape[0]),
                int(main_pool.stride(0)),
                *page_args(main_pages),
                int(plan.main_page_size),
                int(plan.num_requests),
                _stream(device),
            )
        rows_per_page = int(main_mapping[1]) if main_mapping is not None else 1
        page_shift = (
            rows_per_page.bit_length() - 1 if rows_per_page & (rows_per_page - 1) == 0 else -1
        )
        _launch(
            ("shared_compact", has_main, premapped, page_shift >= 0, device.index),
            lambda: ns.SharedCompactKernel(has_main, premapped, page_shift >= 0),
            _ptr(swa_slots, align=8),
            _ptr(logical, align=8 if logical_int64 else 4),
            _ptr(slots, align=8),
            _ptr(plan.query_requests, align=8),
            _ptr(plan.query_positions),
            _ptr(plan.swa_starts, align=8),
            _ptr(plan.swa_first, align=8),
            _ptr(plan.swa_floors, align=8),
            _ptr(plan.swa_offsets, align=8),
            _ptr(plan.main_offsets, align=8),
            _ptr(visible, align=8),
            table_ptr,
            _ptr(selected),
            _ptr(metadata.prepared_indices),
            _ptr(metadata.prepared_lens),
            int(count),
            swa_width,
            main_width,
            int(swa_slots.stride(0)),
            int(swa_slots.stride(1)) if swa_width > 0 else 1,
            int(logical.stride(0)) if main_width > 0 else 0,
            int(logical.stride(1)) if main_width > 0 else 1,
            int(slot_rs),
            int(slot_cs),
            int(swa_pool.shape[0]),
            int(main_pool.shape[0]),
            table_rows,
            table_cols,
            table_stride,
            page_size,
            max(page_shift, 0),
            max_positions,
            int(plan.num_requests),
            int(capacity),
            int(plan.window),
            _stream(device),
        )
    if scale is not None:
        q, q_shared, minimum, maximum = scale
        bitmap = metadata._csa2_stage_scale_bitmap
        _launch(
            ("shared_scale", device.index),
            ns.SharedScaleKernel,
            *row_args,
            _ptr(bitmap),
            *geometry_args,
            _stream(device),
        )
        derive_stage_kv_scales(
            [],
            bitmap,
            metadata._csa2_stage_kv_dequant,
            metadata._csa2_stage_kv_quant,
            q=q,
            q_shared=q_shared,
            minimum=minimum,
            maximum=maximum,
        )
    _launch(
        ("shared_decode", fp8, device.index),
        lambda: ns.SharedDecodeKernel(fp8),
        *row_args,
        _ptr(bank, align=8),
        _ptr(quant),
        _ptr(q_dequant),
        _ptr(kv_scale_quant_orig if fp8 else placeholder),
        _ptr(metadata.mla_bmm1_scale if fp8 else placeholder),
        _ptr(metadata.mla_bmm2_scale if fp8 else placeholder),
        _ptr(metadata.prepared_counter),
        _ptr(metadata._csa2_shared_main_scales),
        *geometry_args,
        float(softmax_scale),
        int(not compact),
        int(parity),
        _stream(device),
    )


# =============================================================================
# Index-Q projection and packed sparse attention
# =============================================================================


def _indexer_projection_runner_type():
    """Import optional CuTe dependencies only when fused projection executes."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLIndexerQBlackwellRunner
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import (
        Sm100BlockScaledPersistentDenseGemmKernel,
        _indexer_q_pack_fp4x4,
    )
    from tensorrt_llm.quantization.utils.fp4_utils import pad_up

    class _CSA2IndexerQKernel(Sm100BlockScaledPersistentDenseGemmKernel):
        """Existing small-M MMA with CSA2 nearest-even FP4 and its scale floor."""

        @cute.jit
        def _indexer_q_transform_rows(
            self,
            sC: cute.Tensor,
            c_buffer: cutlass.Int32,
            row_start: cutlass.Int32,
            row_stride: cutlass.Constexpr,
            head_idx: cutlass.Int32,
            token_tile_idx: cutlass.Int32,
            batch_idx: cutlass.Int32,
            real_subtile_idx: cutlass.Int32,
            mPacked_nml: cute.Tensor,
            mIndexerScale_nml: cute.Tensor,
            mPositionIds: cute.Tensor,
            mCosSinCache: cute.Tensor,
        ):
            """Transform complete BF16 token rows from the shared epilogue tile."""
            lane_idx = cute.arch.lane_idx()
            for row in cutlass.range(row_start, self.epi_tile_n, row_stride, unroll_full=True):
                token_idx = (
                    token_tile_idx * self.cta_tile_shape_mnk[1]
                    + real_subtile_idx * self.epi_tile_n
                    + row
                )
                if token_idx < mPositionIds.shape[0]:
                    values = cute.make_rmem_tensor((4,), cutlass.Float32)
                    for value_idx in cutlass.range_constexpr(4):
                        feature_idx = lane_idx * 4 + value_idx
                        values[value_idx] = cutlass.Float32(sC[(feature_idx, row, c_buffer)])

                    if lane_idx >= 16:
                        position = mPositionIds[token_idx]
                        pair_base = (lane_idx * 4 - 64) // 2
                        for value_idx in cutlass.range_constexpr(0, 4, 2):
                            cosine = mCosSinCache[position, pair_base + value_idx // 2]
                            sine = mCosSinCache[position, pair_base + value_idx // 2 + 32]
                            x = values[value_idx]
                            y = values[value_idx + 1]
                            values[value_idx] = (
                                (cosine * x - sine * y).to(cutlass.BFloat16).to(cutlass.Float32)
                            )
                            values[value_idx + 1] = (
                                (cosine * y + sine * x).to(cutlass.BFloat16).to(cutlass.Float32)
                            )

                    amax = cute.arch.fmax(
                        cute.arch.fmax(values[0], -values[0]),
                        cute.arch.fmax(values[1], -values[1]),
                    )
                    amax = cute.arch.fmax(
                        amax,
                        cute.arch.fmax(
                            cute.arch.fmax(values[2], -values[2]),
                            cute.arch.fmax(values[3], -values[3]),
                        ),
                    )
                    amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
                    amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=2))
                    amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=4))
                    if amax < cutlass.Float32(6.0 * 2.0**-126):
                        amax = cutlass.Float32(6.0 * 2.0**-126)

                    scale_reg = cute.make_rmem_tensor((1,), cutlass.Float8E8M0FNU)
                    scale_reg[0] = (amax * cutlass.Float32(1.0 / 6.0)).to(cutlass.Float8E8M0FNU)
                    scale_byte = cute.recast_tensor(scale_reg, cutlass.Uint8)[0]
                    exponent = cutlass.Uint32(scale_byte)
                    inverse_bits = cutlass.Uint32(0)
                    if exponent == cutlass.Uint32(254):
                        inverse_bits = cutlass.Uint32(0x00400000)
                    else:
                        inverse_bits = (cutlass.Uint32(254) - exponent) << 23
                    inverse_scale = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, inverse_bits.ir_value())
                    )
                    for value_idx in cutlass.range_constexpr(4):
                        values[value_idx] = values[value_idx] * inverse_scale

                    packed = _indexer_q_pack_fp4x4(
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                    )
                    mPacked_nml[head_idx * 32 + lane_idx, token_idx, batch_idx] = packed

                    exponent0 = cutlass.Uint32(cute.arch.shuffle_sync(exponent, cutlass.Int32(0)))
                    exponent1 = cutlass.Uint32(cute.arch.shuffle_sync(exponent, cutlass.Int32(8)))
                    exponent2 = cutlass.Uint32(cute.arch.shuffle_sync(exponent, cutlass.Int32(16)))
                    exponent3 = cutlass.Uint32(cute.arch.shuffle_sync(exponent, cutlass.Int32(24)))
                    if lane_idx == 0:
                        mIndexerScale_nml[
                            head_idx,
                            token_idx,
                            batch_idx,
                        ] = exponent0 | (exponent1 << 8) | (exponent2 << 16) | (exponent3 << 24)

    class _CSA2IndexerQRunner(CuteDSLIndexerQBlackwellRunner):
        """Reuse tuning/launch contracts, with local quantization and epilogue."""

        small_m_kernel_class = _CSA2IndexerQKernel
        kernel_cache = {}

        def unique_id(self):
            return ("csa2_rne_32", self.use_tvm_ffi)

        def get_valid_tactics(self, inputs, profile, **kwargs):
            m, k = inputs[0].shape
            n = inputs[1].shape[0]
            return list(self._small_m_tactics) if self._small_m_kernel_is_supported(m, n, k) else []

        def forward(self, inputs, tactic):
            x, weight, weight_scale, positions, cos_sin, alpha = inputs
            m, k = x.shape
            n = weight.shape[0]
            if not self._small_m_kernel_is_supported(m, n, k):
                raise ValueError("Fused CSA2 index Q requires 1..16 rows and N/K divisible by 128")
            if tactic == -1:
                tactic = self._small_m_tactics[0 if m <= 4 else 1]
            kind, tile, cluster, prefetch, warps = tactic
            if kind != "swap_ab":
                raise ValueError("CSA2 index Q supports only the small-M epilogue specialization")
            # Unlike V4, both activation scaling and output FP4 rounding follow the
            # 32-channel CSA2 contract. The GEMM/transform kernel remains fused.
            data, scales = torch.ops.trtllm.mxfp8_quantize(x, True)
            packed = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
            output_scales = torch.empty((m, n // 32), dtype=torch.uint8, device=x.device)
            pointers = (
                self._ptr(data, cutlass.Float8E4M3FN),
                self._ptr(weight, cutlass.Float8E4M3FN),
                self._ptr(scales, cutlass.Float8E8M0FNU),
                self._ptr(weight_scale, cutlass.Float8E8M0FNU),
                self._ptr(packed, cutlass.Uint8),
                self._ptr(output_scales, cutlass.Float8E8M0FNU),
                self._ptr(positions, cutlass.Int32, 4),
                self._ptr(cos_sin, cutlass.Float32, 32),
            )
            alpha_cute = cute.runtime.from_dlpack(alpha)
            stream = (
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
                if self.use_tvm_ffi
                else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            )
            key = (tile, cluster, prefetch, warps, self.use_tvm_ffi)
            dynamic = (
                m,
                n,
                k,
                pad_up(m, 128) // 128,
                pad_up(n, 128) // 128,
                pad_up(k // 32, 4) // 4,
                cos_sin.shape[0],
            )
            if key not in self.kernel_cache:
                kernel = self.small_m_kernel_class(
                    32,
                    tile,
                    cluster,
                    use_prefetch=prefetch,
                    indexer_q_fusion=True,
                    indexer_transform_warps=warps,
                )
                clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
                    cluster[0] * cluster[1]
                )
                self.kernel_cache[key] = cute.compile(
                    kernel.wrapper_indexer_q_swap_ab,
                    *dynamic,
                    1,
                    *pointers,
                    alpha_cute,
                    clusters,
                    stream,
                    options="--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi
                    else "--opt-level 2",
                )
            compiled = self.kernel_cache[key]
            if self.use_tvm_ffi:
                compiled(
                    *dynamic,
                    data.data_ptr(),
                    weight.data_ptr(),
                    scales.data_ptr(),
                    weight_scale.data_ptr(),
                    packed.data_ptr(),
                    output_scales.data_ptr(),
                    positions.data_ptr(),
                    cos_sin.data_ptr(),
                    alpha,
                )
            else:
                compiled(*dynamic, *pointers, alpha_cute, stream)
            return packed.view(torch.int8), output_scales.view(torch.int32)

    return _CSA2IndexerQRunner


@torch.library.custom_op(
    "trtllm::csa2_indexer_q_gemm_rope_fp4", mutates_args=(), device_types="cuda"
)
def csa2_indexer_q_gemm_rope_fp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native small-M projection with exact CSA2 activation/FP4 semantics."""
    from tensorrt_llm._torch.autotuner import AutoTuner

    runner = _indexer_projection_runner_type()()
    inputs = [x, weight, weight_scale, positions, cos_sin, alpha]
    _, tactic = AutoTuner.get().choose_one(
        "trtllm::csa2_indexer_q_gemm_rope_fp4", [runner], runner.tuning_config, inputs
    )
    return runner(inputs, tactic=tactic)


@csa2_indexer_q_gemm_rope_fp4.register_fake
def _fake_indexer_projection(x, weight, weight_scale, positions, cos_sin, alpha):
    return (
        x.new_empty((x.shape[0], weight.shape[0] // 2), dtype=torch.int8),
        x.new_empty((x.shape[0], weight.shape[0] // 128), dtype=torch.int32),
    )


def supports_packed_attention(
    q,
    swa_pool,
    main_pool,
    swa_indices,
    main_indices,
    sink,
    scale,
    *,
    position_ids=None,
    rotary_cos_sin=None,
):
    if not isinstance(q, torch.Tensor) or not q.is_cuda or q.ndim != 3:
        return False
    no_main = main_pool is None and main_indices is None
    if not no_main and (main_pool is None or main_indices is None):
        return False
    pools = (swa_pool,) if no_main else (swa_pool, main_pool)
    indices = (swa_indices,) if no_main else (swa_indices, main_indices)
    tensors = (q, sink, *pools, *indices)
    if position_ids is not None or rotary_cos_sin is not None:
        if position_ids is None or rotary_cos_sin is None:
            return False
        if not (
            isinstance(position_ids, torch.Tensor)
            and isinstance(rotary_cos_sin, torch.Tensor)
            and position_ids.dtype in (torch.int32, torch.int64)
            and position_ids.shape == (q.shape[0],)
            and position_ids.is_contiguous()
            and rotary_cos_sin.dtype == torch.float32
            and rotary_cos_sin.ndim == 3
            and rotary_cos_sin.shape[0] > 0
            and rotary_cos_sin.shape[1:] == (2, 32)
            and rotary_cos_sin.is_contiguous()
        ):
            return False
        tensors += (position_ids, rotary_cos_sin)
    return (
        all(isinstance(t, torch.Tensor) and t.is_cuda and t.device == q.device for t in tensors)
        and torch.cuda.get_device_capability(q.device) == (10, 0)
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[2] == 512
        and q.shape[1] > 0
        and q.shape[1] % 16 == 0
        and q.is_contiguous()
        and all(
            p.dtype == torch.uint8 and p.ndim == 2 and p.shape[0] > 0 and p.stride(1) == 1
            for p in pools
        )
        and swa_pool.shape[1] == 528
        and (no_main or main_pool.shape[1] == 288)
        and all(
            i.dtype in (torch.int32, torch.int64)
            and i.ndim == 2
            and i.shape[0] == q.shape[0]
            and i.is_contiguous()
            for i in indices
        )
        and sink.dtype == torch.float32
        and sink.shape == (q.shape[1],)
        and sink.is_contiguous()
        and isinstance(scale, (int, float))
    )


def packed_attention_workspace_bytes(
    queries: int, heads: int, swa_width: int, main_width: int, num_sms: int
) -> int:
    """Transient FP32 split accumulators, maxima and denominators (no KV staging)."""
    if queries == 0:
        return 0
    splits = max(
        1,
        min(
            math.ceil((swa_width + main_width) / 64), max(1, 2 * num_sms // (queries * heads // 16))
        ),
    )
    return queries * heads * splits * 514 * 4


@functools.lru_cache(maxsize=1)
def _packed_attention_kernel_type():
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.nvgpu import warp

    class PackedAttention:
        @cute.jit
        def __call__(
            self,
            q,
            swa,
            main,
            si,
            mi,
            sink,
            out,
            partial,
            positions,
            cos_sin,
            scale: cutlass.Float32,
            stream,
        ):
            mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
                cute.make_layout((1, 4, 1)),
            )
            self.kernel(q, swa, main, si, mi, sink, partial, scale, mma).launch(
                grid=(q.shape[0], q.shape[1] // 16, partial.shape[2]),
                block=(128, 1, 1),
                stream=stream,
            )

            self.reduce(partial, sink, out, positions, cos_sin).launch(
                grid=(q.shape[0], q.shape[1] // 4, 1), block=(128, 1, 1), stream=stream
            )

        @cute.kernel
        def kernel(
            self, q, swa, main, si, mi, sink, partial, scale: cutlass.Float32, mma: cute.TiledMma
        ):
            tid, _, _ = cute.arch.thread_idx()
            row, head_block, split = cute.arch.block_idx()
            swa_fp8 = cute.recast_tensor(swa, cutlass.Float8E4M3FN)
            main_fp8 = cute.recast_tensor(main, cutlass.Float8E4M3FN)
            smem = utils.SmemAllocator()
            sq = smem.allocate_tensor(
                cutlass.BFloat16, cute.make_layout((16, 512), stride=(512, 1)), 16
            )
            sk = smem.allocate_tensor(
                cutlass.BFloat16, cute.make_layout((64, 512), stride=(512, 1)), 16
            )
            sp = smem.allocate_tensor(
                cutlass.BFloat16, cute.make_layout((16, 64), stride=(64, 1)), 16
            )
            ss = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((16, 64), stride=(64, 1)), 16
            )
            stats = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((16, 3), stride=(3, 1)), 16
            )
            valid = smem.allocate_tensor(cutlass.Int32, cute.make_layout(64), 16)
            for j in cutlass.range_constexpr(64):
                offset = tid + j * 128
                sq[offset // 512, offset % 512] = q[
                    row, head_block * 16 + offset // 512, offset % 512
                ]
            if tid < 16:
                stats[tid, 0] = cutlass.Float32(float("-inf"))
                stats[tid, 1] = 0.0
            thr = mma.get_slice(tid)
            acc = cute.make_rmem_tensor(thr.partition_shape_C((16, 512)), cutlass.Float32)
            acc.fill(0.0)
            co = thr.partition_C(cute.make_identity_tensor((16, 512)))
            cs = thr.partition_C(cute.make_identity_tensor((16, 64)))
            sv = cute.make_tensor(sk.iterator, cute.make_layout((512, 64), stride=(1, 512)))
            cute.arch.barrier()
            for tile in range(
                split, cute.ceil_div(si.shape[1] + mi.shape[1], 64), partial.shape[2]
            ):
                if tid < 64:
                    column = tile * 64 + tid
                    slot = cutlass.Int64(-1)
                    capacity = cutlass.Int64(0)
                    if column < si.shape[1]:
                        slot = cutlass.Int64(si[row, column])
                        capacity = cutlass.Int64(swa.shape[0])
                    elif column < si.shape[1] + mi.shape[1]:
                        slot = cutlass.Int64(mi[row, column - si.shape[1]])
                        capacity = cutlass.Int64(main.shape[0])
                    valid[tid] = cutlass.Int32((slot >= 0) & (slot < capacity))
                for j in range(256):
                    offset = tid + j * 128
                    key = offset // 512
                    d = offset % 512
                    column = tile * 64 + key
                    value = cutlass.Float32(0.0)
                    if column < si.shape[1]:
                        slot = cutlass.Int64(si[row, column])
                        if (slot >= 0) & (slot < swa.shape[0]):
                            exponent = cutlass.Int32(swa[slot, 512 + d // 32]) - 127
                            value = cutlass.Float32(swa_fp8[slot, d]) * cute.exp2(
                                cutlass.Float32(exponent)
                            )
                    elif column < si.shape[1] + mi.shape[1]:
                        slot = cutlass.Int64(mi[row, column - si.shape[1]])
                        if (slot >= 0) & (slot < main.shape[0]):
                            packed = cutlass.Int32(main[slot, d // 2])
                            code = (packed >> ((d % 2) * 4)) & 15
                            mag = code & 7
                            value = cutlass.Float32(mag) * 0.5
                            if mag >= 4:
                                value = (1.0 + cutlass.Float32(mag % 2) * 0.5) * cute.exp2(
                                    cutlass.Float32(mag // 2 - 1)
                                )
                            if (code & 8) != 0:
                                value = -value
                            value = value * cutlass.Float32(main_fp8[slot, 256 + d // 16])
                    sk[key, d] = cutlass.BFloat16(value)
                cute.arch.barrier()
                scores = cute.make_rmem_tensor(thr.partition_shape_C((16, 64)), cutlass.Float32)
                scores.fill(0.0)
                for k in range(32):
                    qa = cute.local_tile(sq, (16, 16), (0, k))
                    kb = cute.local_tile(sk, (64, 16), (0, k))
                    pa = thr.partition_A(qa)
                    pb = thr.partition_B(kb)
                    ra = thr.make_fragment_A(pa)
                    rb = thr.make_fragment_B(pb)
                    cute.autovec_copy(pa, ra)
                    cute.autovec_copy(pb, rb)
                    cute.gemm(mma, scores, ra, rb, scores)
                for j in cutlass.range_constexpr(cute.size(scores)):
                    h, k = cs[j]
                    value = cutlass.Float32(float("-inf"))
                    if valid[k] != 0:
                        value = scores[j] * scale
                    ss[h, k] = value
                cute.arch.barrier()
                # One warp per head, four heads at a time. The reducer adds
                # the sink exactly once after merging disjoint key splits.
                lane = tid % 32
                for j in cutlass.range_constexpr(4):
                    h = tid // 32 + j * 4
                    a = ss[h, lane]
                    b = ss[h, lane + 32]
                    mx = cute.arch.warp_reduction_max(cute.arch.fmax(a, b))
                    mx = cute.arch.fmax(mx, stats[h, 0])
                    old_scale = cutlass.Float32(0.0)
                    if stats[h, 1] != 0.0:
                        old_scale = cute.exp2((stats[h, 0] - mx) * 1.4426950408889634)
                    safe_mx = mx
                    if mx == cutlass.Float32(float("-inf")):
                        safe_mx = cutlass.Float32(0.0)
                    a = cute.exp2((a - safe_mx) * 1.4426950408889634)
                    b = cute.exp2((b - safe_mx) * 1.4426950408889634)
                    denom = cute.arch.warp_reduction_sum(a + b)
                    sp[h, lane] = cutlass.BFloat16(a)
                    sp[h, lane + 32] = cutlass.BFloat16(b)
                    if lane == 0:
                        stats[h, 0] = mx
                        stats[h, 1] = stats[h, 1] * old_scale + denom
                        stats[h, 2] = old_scale
                cute.arch.barrier()
                for j in cutlass.range_constexpr(cute.size(acc)):
                    h, _ = co[j]
                    acc[j] = acc[j] * stats[h, 2]
                for k in range(4):
                    pa = thr.partition_A(cute.local_tile(sp, (16, 16), (0, k)))
                    pb = thr.partition_B(cute.local_tile(sv, (512, 16), (0, k)))
                    ra = thr.make_fragment_A(pa)
                    rb = thr.make_fragment_B(pb)
                    cute.autovec_copy(pa, ra)
                    cute.autovec_copy(pb, rb)
                    cute.gemm(mma, acc, ra, rb, acc)
                cute.arch.barrier()
            for j in cutlass.range_constexpr(cute.size(acc)):
                h, d = co[j]
                partial[row, head_block * 16 + h, split, d] = acc[j]

            if tid < 16:
                partial[row, head_block * 16 + tid, split, 512] = stats[tid, 0]
                partial[row, head_block * 16 + tid, split, 513] = stats[tid, 1]

        @cute.kernel
        def reduce(self, partial, sink, out, positions, cos_sin):
            tid, _, _ = cute.arch.thread_idx()
            row, heads, _ = cute.arch.block_idx()
            head = heads * 4 + tid // 32
            lane = tid % 32
            mx = sink[head]
            for split in range(partial.shape[2]):
                mx = cute.arch.fmax(mx, partial[row, head, split, 512])
            denom = cute.exp2((sink[head] - mx) * 1.4426950408889634)
            for split in range(partial.shape[2]):
                factor = cute.exp2((partial[row, head, split, 512] - mx) * 1.4426950408889634)
                denom += factor * partial[row, head, split, 513]
            for dblock in cutlass.range_constexpr(16):
                d = lane + 32 * dblock
                value = cutlass.Float32(0.0)
                for split in range(partial.shape[2]):
                    factor = cute.exp2((partial[row, head, split, 512] - mx) * 1.4426950408889634)
                    value += factor * partial[row, head, split, d]
                rounded = cutlass.BFloat16(value / denom)
                if cutlass.const_expr(positions is not None and dblock >= 14):
                    # Preserve the BF16 attention boundary before inverse RoPE.
                    own = cutlass.Float32(rounded)
                    other = cute.arch.shuffle_sync_bfly(own, offset=1)
                    position = cutlass.Int64(positions[row])
                    rotated = cutlass.Float32(float("nan"))
                    # Invalid positions never form out-of-bounds cache reads.
                    # Runtime metadata owns position validation.
                    if (position >= 0) & (position < cos_sin.shape[0]):
                        cosine = cos_sin[position, 0, (d - 448) // 2]
                        sine = cos_sin[position, 1, (d - 448) // 2]
                        if lane % 2 != 0:
                            sine = -sine
                        rotated = own * cosine + other * sine
                    rounded = cutlass.BFloat16(rotated)
                out[row, head, d] = rounded

    return PackedAttention


_PACKED_ATTENTION_COMPILED = {}


def packed_sparse_attention(
    q,
    swa_pool,
    main_pool,
    swa_indices,
    main_indices,
    sink,
    scale,
    *,
    output=None,
    workspace=None,
    position_ids=None,
    rotary_cos_sin=None,
):
    """Run direct packed attention; invalid indices have zero probability.

    Optional CUDA positions[Q] and FP32 cosine/sine[max_position,2,32]
    fuse inverse interleaved RoPE on the final 64 channels. Positions must
    be in range; invalid positions produce NaN in the rotated tail.
    """
    if not supports_packed_attention(
        q,
        swa_pool,
        main_pool,
        swa_indices,
        main_indices,
        sink,
        scale,
        position_ids=position_ids,
        rotary_cos_sin=rotary_cos_sin,
    ):
        raise ValueError("Unsupported CSA2 packed attention geometry, dtype or device")
    if main_pool is None:
        main_pool = torch.empty((1, 288), dtype=torch.uint8, device=q.device)
        main_indices = torch.empty((q.shape[0], 0), dtype=torch.int32, device=q.device)
    if output is None:
        output = torch.empty_like(q)
    if (
        output.shape != q.shape
        or output.dtype != q.dtype
        or output.device != q.device
        or not output.is_contiguous()
    ):
        raise ValueError("CSA2 packed output must match contiguous BF16 query shape/device")
    if q.shape[0] == 0:
        return output
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    workspace_bytes = packed_attention_workspace_bytes(
        q.shape[0],
        q.shape[1],
        swa_indices.shape[1],
        main_indices.shape[1],
        torch.cuda.get_device_properties(q.device).multi_processor_count,
    )
    splits = workspace_bytes // (q.shape[0] * q.shape[1] * 514 * 4)
    shape = (q.shape[0], q.shape[1], splits, 514)
    if workspace is None:
        workspace = torch.empty(shape, dtype=torch.float32, device=q.device)
    elif (
        workspace.shape != shape
        or workspace.dtype != torch.float32
        or workspace.device != q.device
        or not workspace.is_contiguous()
    ):
        raise ValueError("CSA2 packed workspace must match FP32 split geometry/device")
    partial = workspace
    tensors = (
        q,
        swa_pool,
        main_pool,
        swa_indices,
        main_indices,
        sink,
        output,
        partial,
        position_ids,
        rotary_cos_sin,
    )
    key = (
        q.device.index,
        tuple(None if t is None else (tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors),
    )
    views = tuple(None if t is None else from_dlpack(t.detach(), assumed_align=1) for t in tensors)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    if key not in _PACKED_ATTENTION_COMPILED:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm CSA2 packed attention before CUDA Graph capture")
        _PACKED_ATTENTION_COMPILED[key] = cute.compile(
            _packed_attention_kernel_type()(), *views, cutlass.Float32(scale), stream
        )
    _PACKED_ATTENTION_COMPILED[key](*views, cutlass.Float32(scale), stream)
    return output
