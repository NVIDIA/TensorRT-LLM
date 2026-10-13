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

* Decode metadata and selection glue (SWA/owner slot refresh, indexer
  descriptors, logits masking, candidate gathering/publication, Top-K
  finalization). Every kernel replaces a chain of small PyTorch or Inductor
  launches that ran once per layer (or per KV owner) on every decode step and
  follows the integer semantics of the PyTorch references in ``metadata.py``
  and ``indexer.py`` (floor division/modulo, ``-1`` padding, ``INT_MAX`` sort
  sentinels).
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

# =============================================================================
# Decode metadata and selection glue
# =============================================================================

_INT32_MAX = 2**31 - 1


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

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

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
        sequence) map to its scratch slots, as ``PageIndexConverter`` does.
        """

        threads = 128

        def __init__(self, with_scratch: bool):
            self.with_scratch = with_scratch

        @cute.jit
        def __call__(
            self,
            base: cute.Pointer,  # int32 [pools, seqs, blocks]
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
            stream: cuda.CUstream,
        ):
            self.kernel(
                base,
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
            ).launch(grid=[seqs, tables, 1], block=[self.threads, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            base: cute.Pointer,
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
        ):
            tid, _, _ = cute.arch.thread_idx()
            seq, table, _ = cute.arch.block_idx()
            param_t = tensor2d(params, Int64(tables), Int64(5), Int64(5))
            pool = Int64(param_t[table, 0])
            scale = Int32(param_t[table, 1])
            offset = Int32(param_t[table, 2])
            src_base = pool * pool_stride + Int64(seq) * row_stride
            dst_base = Int64(table) * out_table_stride + Int64(seq) * out_row_stride
            src = tensor1d(base, src_base + Int64(width))
            dst = tensor1d(out, dst_base + Int64(width))
            if cutlass.const_expr(self.with_scratch):
                scratch_row = Int64(param_t[table, 3]) * Int64(seqs) + Int64(seq)
                scratch_pages = Int32(param_t[table, 4])
                ranges = tensor1d(scratch_ranges, (scratch_row + 1) * 2)
                begin = Int32(ranges[scratch_row * 2])
                end = Int32(ranges[scratch_row * 2 + 1])
                slots = tensor1d(scratch_slots, (scratch_row + 1) * Int64(slot_stride))
            for column in cutlass.range(tid, width, self.threads):
                page = Int32(src[src_base + Int64(column)])
                value = bad
                if page != bad:
                    value = page * scale + offset
                if cutlass.const_expr(self.with_scratch):
                    if (column >= begin) & (column < end):
                        total = (column - begin) * scratch_pages
                        slot = Int32(
                            slots[scratch_row * Int64(slot_stride) + Int64(total // scale)]
                        )
                        value = slot * scale + (total % scale + offset) % scale
                dst[dst_base + Int64(column)] = value

    # ------------------------------------------------------ K2: owner slots

    class OwnerSlotsKernel:
        """GLOBAL write slots and compressed groups of every KV owner for live endpoints.

        Mirrors ``CSA2TrtllmMetadata._refresh_owner_slots`` in one launch: grid
        ``y`` selects the owner, whose buffers and geometry come from an int64
        ``[owners, OWNER_DESC_WIDTH]`` descriptor table (see
        ``owner_slot_descriptor``). Per-request group counts and their prefix
        sums are recomputed by every CTA in shared memory (requests are few),
        then each thread resolves one output row.
        """

        MAX_REQUESTS = 4096
        THREADS = 256

        @cute.jit
        def __call__(
            self,
            kv_lens: cute.Pointer,  # int32
            lengths: cute.Pointer,  # int32 [requests]
            query_base: cute.Pointer,  # int32 [requests]
            descriptors: cute.Pointer,  # int64 [owners, OWNER_DESC_WIDTH]
            requests: Int32,
            owners: Int32,
            max_capacity: Int32,
            stream: cuda.CUstream,
        ):
            blocks = (max_capacity + self.THREADS - 1) // self.THREADS
            if blocks < 1:
                blocks = Int32(1)
            self.kernel(kv_lens, lengths, query_base, descriptors, requests, owners).launch(
                grid=[blocks, owners, 1], block=[self.THREADS, 1, 1], stream=stream
            )

        @cute.kernel
        def kernel(
            self,
            kv_lens: cute.Pointer,
            lengths: cute.Pointer,
            query_base: cute.Pointer,
            descriptors: cute.Pointer,
            requests: Int32,
            owners: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            bid, owner, _ = cute.arch.block_idx()
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
        """Native paged-indexer block tables and position validity for decode queries.

        One thread per native page: it resolves the page's physical block and
        writes the page's ``page_rows`` validity bytes as 32-bit words.
        """

        THREADS = 128

        @cute.jit
        def __call__(
            self,
            table: cute.Pointer,  # int32 [requests, source_pages] strided
            token_requests: cute.Pointer,  # int64 [count]
            decode_visible: cute.Pointer,  # int64 [count]
            blocks: cute.Pointer,  # int32 [count, page_capacity] (out)
            context: cute.Pointer,  # int32 [count, 1] (out)
            visible: cute.Pointer,  # int32 [count] (out)
            valid: cute.Pointer,  # int32 words of bool [count, page_capacity * page_rows] (out)
            count: Int32,
            request_count: Int32,
            source_pages: Int32,
            table_stride: Int64,
            context_requests: Int32,
            pages_per_source_page: Int32,
            page_rows: Int32,
            page_capacity: Int32,
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
                pages_per_source_page,
                page_rows,
                page_capacity,
            ).launch(
                grid=[count, (page_capacity + self.THREADS - 1) // self.THREADS, 1],
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
            pages_per_source_page: Int32,
            page_rows: Int32,
            page_capacity: Int32,
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
            page = chunk * self.THREADS + tid
            if page < page_capacity:
                source = page // pages_per_source_page
                native = Int32(0)
                backed = cutlass.Boolean(False)
                if admitted & (source < source_pages) & (request_count > 0):
                    physical = Int32(table_t[safe_request, source])
                    if physical >= 0:
                        native = physical * pages_per_source_page + page % pages_per_source_page
                        backed = cutlass.Boolean(True)
                tensor2d(blocks, n, Int64(page_capacity), Int64(page_capacity))[q, page] = native
                words_per_page = page_rows // 4
                words = tensor1d(valid, Int64(2**62))
                word_base = (Int64(q) * Int64(page_capacity) + Int64(page)) * Int64(words_per_page)
                first_position = page * page_rows
                for w in cutlass.range(0, words_per_page, 1):
                    position = first_position + w * 4
                    word = Int32(0)
                    if backed:
                        if position + 4 <= end:
                            word = Int32(0x01010101)
                        else:
                            for j in cutlass.range_constexpr(4):
                                if position + j < end:
                                    word = word | (Int32(1) << (8 * j))
                    words[word_base + Int64(w)] = word
            if (chunk == 0) & (tid == 0):
                tensor1d(visible, n)[q] = end
                clamped = end
                if clamped < 1:
                    clamped = Int32(1)
                tensor1d(context, n)[q] = clamped

    # -------------------------------------------- K4: mask in-place logits

    class MaskLogitsKernel:
        """``logits.masked_fill_(~valid, -inf)`` over ``[count, width]`` float32 rows."""

        THREADS = 256
        PER_THREAD = 4

        @cute.jit
        def __call__(
            self,
            logits: cute.Pointer,  # float32 [count, width] strided
            valid: cute.Pointer,  # uint8 [count, >= width] strided
            count: Int32,
            width: Int32,
            logits_stride: Int64,
            valid_stride: Int64,
            stream: cuda.CUstream,
        ):
            span = self.THREADS * self.PER_THREAD
            self.kernel(logits, valid, count, width, logits_stride, valid_stride).launch(
                grid=[count, (width + span - 1) // span, 1],
                block=[self.THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            logits: cute.Pointer,
            valid: cute.Pointer,
            count: Int32,
            width: Int32,
            logits_stride: Int64,
            valid_stride: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, chunk, _ = cute.arch.block_idx()
            logits_t = tensor2d(logits, Int64(count), Int64(width), logits_stride)
            valid_t = tensor2d(valid, Int64(count), Int64(width), valid_stride)
            first = chunk * self.THREADS * self.PER_THREAD + tid
            for j in cutlass.range_constexpr(self.PER_THREAD):
                column = first + j * self.THREADS
                if column < width:
                    if cutlass.Uint8(valid_t[q, column]) == 0:
                        logits_t[q, column] = Float32(float("-inf"))

    # ------------------------------------------ K5: candidate-ordered scores

    class CandidateGatherKernel:
        """Gather candidate columns from paged logits with validity and visibility masks."""

        THREADS = 256

        @cute.jit
        def __call__(
            self,
            logits: cute.Pointer,  # float32 [count, width] strided
            valid: cute.Pointer,  # uint8 [count, >= width] strided
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
                    valid_t = tensor2d(valid, n, Int64(width), valid_stride)
                    if cutlass.Uint8(valid_t[q, Int32(safe)]) != 0:
                        score = Float32(
                            tensor2d(logits, n, Int64(width), logits_stride)[q, Int32(safe)]
                        )
                        out_position = Int32(position)
                tensor2d(scores, n, Int64(cwidth), Int64(cwidth))[q, column] = score
                tensor2d(positions, n, Int64(cwidth), Int64(cwidth))[q, column] = out_position

    # ------------------------------------------- K6: finalize Top-K rows

    class FinalizeSelectionKernel:
        """Validate, map and sort one Top-K row in place; padding becomes ``-1``.

        Mode ``paged``: offsets are positions, valid when ``valid[q, off]``.
        Mode ``mapped``: offsets index ``positions``; valid when the gathered
        score is finite (``> -inf``). A bitonic sort over ``INT32_MAX``
        sentinels reproduces ``torch.sort`` of the int64 sentinel reference.
        """

        def __init__(self, top_k: int, mapped: bool):
            if top_k <= 0 or top_k > 4096:
                raise ValueError("CSA2 selection finalize supports 1..4096 selections")
            self.top_k = top_k
            self.padded = 1 << (top_k - 1).bit_length()
            self.threads = max(32, min(1024, self.padded // 2 if self.padded >= 64 else 32))
            self.mapped = mapped

        @cute.jit
        def __call__(
            self,
            indices: cute.Pointer,  # int32 [count, top_k] strided (in/out)
            valid: cute.Pointer,  # uint8 [count, width] strided (paged mode)
            scores: cute.Pointer,  # float32 [count, width] strided (mapped mode)
            positions: cute.Pointer,  # int32/int64 [count, width] strided (mapped mode)
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
            keys = smem.allocate_tensor(Int32, cute.make_layout(self.padded), 16)
            n = Int64(count)
            indices_t = tensor2d(indices, n, Int64(self.top_k), indices_stride)
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
                        else:
                            if (
                                cutlass.Uint8(
                                    tensor2d(valid, n, Int64(width), valid_stride)[q, offset]
                                )
                                != 0
                            ):
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

    # ------------------------------------------- K7: candidate publication

    class CandidatePublishKernel:
        """Block hierarchy of one candidate-source row: block maxima, Top-K blocks, positions.

        Mirrors ``CSA2Indexer._publish_candidates``: scores are reduced per
        ``block_size`` columns (``-inf`` padding), the newest visible block is
        pinned with ``+inf``, the ``min(top_k, blocks)`` best blocks are chosen
        by a shared-memory bitonic sort (descending value, ascending block id
        on ties; skipped when every block is selected), the reachable selected
        blocks are ordered ascending by id (unreachable ``-inf`` blocks last) and
        expanded to positions: positions at or beyond ``visible`` and columns of
        unreachable or unselected blocks publish ``-1``, so each row's valid
        positions form a prefix. With ``sparse_block`` the row's DeepGEMM
        sparse-logits inputs are written too: every selected block's sparse
        sub-block ids in order, unreachable slots and the tail repeating the last
        reachable block, plus the number of valid positions.
        """

        def __init__(self, block_size: int, padded_blocks: int, sparse_block: int = 0):
            if block_size <= 0 or padded_blocks <= 0 or padded_blocks & (padded_blocks - 1):
                raise ValueError("CSA2 candidate publication needs a power-of-two block capacity")
            if padded_blocks > 4096:
                raise ValueError("CSA2 candidate publication supports up to 4096 blocks")
            if sparse_block < 0 or (sparse_block and block_size % sparse_block):
                raise ValueError("CSA2 sparse blocks must divide the candidate block")
            self.block_size = block_size
            self.padded = padded_blocks
            self.sparse_block = sparse_block
            self.ratio = block_size // sparse_block if sparse_block else 1
            self.threads = max(32, min(1024, padded_blocks // 2 if padded_blocks >= 64 else 32))

        @cute.jit
        def __call__(
            self,
            scores: cute.Pointer,  # float32 [count, width] strided
            visible: cute.Pointer,  # int32/int64 [count]
            output: cute.Pointer,  # int32 [count, >= published_width] strided (out)
            sparse_ids: cute.Pointer,  # int32 [count, published_width // sparse_block] (out)
            sparse_counts: cute.Pointer,  # int32 [count] (out)
            count: Int32,
            width: Int32,
            blocks: Int32,
            selected_blocks: Int32,
            published_width: Int32,
            scores_stride: Int64,
            output_stride: Int64,
            sparse_stride: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                scores,
                visible,
                output,
                sparse_ids,
                sparse_counts,
                count,
                width,
                blocks,
                selected_blocks,
                published_width,
                scores_stride,
                output_stride,
                sparse_stride,
            ).launch(grid=[count, 1, 1], block=[self.threads, 1, 1], stream=stream)

        @cute.jit
        def _bitonic_sort(self, keys, tags, descending_keys: cutlass.Constexpr, tid):
            """Sort ``padded`` (key, tag) pairs; ``descending_keys`` orders by key desc, tag asc."""
            size = 2
            while size <= self.padded:
                stride = size // 2
                while stride >= 1:
                    for pair in cutlass.range(tid, self.padded // 2, self.threads):
                        i = (pair // stride) * (2 * stride) + (pair % stride)
                        partner = i + stride
                        forward = ((i // size) % 2) == 0
                        a_key = keys[i]
                        b_key = keys[partner]
                        a_tag = tags[i]
                        b_tag = tags[partner]
                        if cutlass.const_expr(descending_keys):
                            after = (a_key < b_key) | ((a_key == b_key) & (a_tag > b_tag))
                        else:
                            after = a_tag > b_tag
                        if after == forward:
                            keys[i] = b_key
                            keys[partner] = a_key
                            tags[i] = b_tag
                            tags[partner] = a_tag
                    cute.arch.barrier()
                    stride = stride // 2
                size = size * 2

        @cute.kernel
        def kernel(
            self,
            scores: cute.Pointer,
            visible: cute.Pointer,
            output: cute.Pointer,
            sparse_ids: cute.Pointer,
            sparse_counts: cute.Pointer,
            count: Int32,
            width: Int32,
            blocks: Int32,
            selected_blocks: Int32,
            published_width: Int32,
            scores_stride: Int64,
            output_stride: Int64,
            sparse_stride: Int64,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, _, _ = cute.arch.block_idx()
            smem = SmemAllocator()
            values = smem.allocate_tensor(Float32, cute.make_layout(self.padded), 16)
            ids = smem.allocate_tensor(Int32, cute.make_layout(self.padded), 16)
            partial = smem.allocate_tensor(Int32, cute.make_layout(max(self.threads // 32, 1)), 16)
            n = Int64(count)
            block_size = Int64(self.block_size)
            score_t = tensor2d(scores, n, Int64(width), scores_stride)
            visible_length = Int64(tensor1d(visible, n)[q])
            latest = floor_div(visible_length - 1, block_size)
            for b in cutlass.range(tid, self.padded, self.threads):
                value = Float32(float("-inf"))
                identifier = Int32(_INT32_MAX)
                if b < blocks:
                    identifier = Int32(b)
                    first = Int64(b) * block_size
                    last = first + block_size
                    if last > Int64(width):
                        last = Int64(width)
                    for column in cutlass.range(first, last, 1):
                        value = cute.arch.fmax(value, Float32(score_t[q, Int32(column)]))
                    if Int64(b) == latest:
                        value = Float32(float("inf"))
                values[b] = value
                ids[b] = identifier
            cute.arch.barrier()
            # Top-K blocks: descending value, ascending block id on ties. When
            # every block is selected the set is the whole prefix, so skip it.
            if selected_blocks < blocks:
                self._bitonic_sort(values, ids, True, tid)
            # Publish in ascending block order with unreachable (-inf) and
            # unselected slots last, so each row's valid columns are a prefix.
            for k in cutlass.range(tid, self.padded, self.threads):
                if (k >= selected_blocks) | (values[k] == Float32(float("-inf"))):
                    ids[k] = Int32(_INT32_MAX)
            cute.arch.barrier()
            self._bitonic_sort(values, ids, False, tid)
            out_t = tensor2d(output, n, Int64(published_width), output_stride)
            selected_width = Int64(selected_blocks) * block_size
            local_valid = Int32(0)
            local_blocks = Int32(0)
            for k in cutlass.range(tid, selected_blocks, self.threads):
                block = Int64(ids[k])
                for j in cutlass.range(0, self.block_size, 1):
                    position = block * block_size + Int64(j)
                    published = Int64(-1)
                    if (
                        (block < Int64(_INT32_MAX))
                        & (position < visible_length)
                        & (position < Int64(width))
                    ):
                        published = position
                        local_valid = local_valid + 1
                    out_t[q, Int32(Int64(k) * block_size + Int64(j))] = Int32(published)
                if block < Int64(_INT32_MAX):
                    local_blocks = local_blocks + 1
            for column in cutlass.range(Int32(selected_width) + tid, published_width, self.threads):
                out_t[q, column] = Int32(-1)
            if cutlass.const_expr(self.sparse_block > 0):
                # Block-wide sums of the valid positions and reachable blocks.
                offset = 1
                while offset < 32:
                    local_valid = local_valid + cute.arch.shuffle_sync_bfly(
                        local_valid, offset=offset
                    )
                    local_blocks = local_blocks + cute.arch.shuffle_sync_bfly(
                        local_blocks, offset=offset
                    )
                    offset = offset * 2
                if cutlass.const_expr(self.threads > 32):
                    if tid % 32 == 0:
                        partial[tid // 32] = local_blocks
                    cute.arch.barrier()
                    local_blocks = Int32(0)
                    for warp in cutlass.range_constexpr(self.threads // 32):
                        local_blocks = local_blocks + partial[warp]
                    cute.arch.barrier()
                    if tid % 32 == 0:
                        partial[tid // 32] = local_valid
                    cute.arch.barrier()
                    local_valid = Int32(0)
                    for warp in cutlass.range_constexpr(self.threads // 32):
                        local_valid = local_valid + partial[warp]
                num_valid = local_blocks
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
    ns.bitcast_u32_f32 = bitcast_u32_f32
    ns.tensor1d = tensor1d
    ns.tensor2d = tensor2d
    ns.SwaRefreshKernel = SwaRefreshKernel
    ns.PageTableConvertKernel = PageTableConvertKernel
    ns.OwnerSlotsKernel = OwnerSlotsKernel
    ns.IndexerDescriptorsKernel = IndexerDescriptorsKernel
    ns.MaskLogitsKernel = MaskLogitsKernel
    ns.CandidateGatherKernel = CandidateGatherKernel
    ns.FinalizeSelectionKernel = FinalizeSelectionKernel
    ns.CandidatePublishKernel = CandidatePublishKernel
    return ns


# ----------------------------------------------------------------- host side

_COMPILED: dict[tuple, Any] = {}
_DTYPE_MAP = None


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
    """Raw gmem pointer for ``tensor``, tagged with its compiled signature."""
    ns = _glue_kernels()
    address = tensor.data_ptr()
    while align > 1 and address % align:
        align //= 2
    element = _cute_dtype(dtype or tensor.dtype)
    pointer = ns.make_ptr(element, address, ns.cute.AddressSpace.gmem, assumed_align=align)
    pointer._csa2_signature = (str(element), align)
    return pointer


def _stream(device: torch.device):
    return _glue_kernels().cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _launch(key: tuple, factory, *args) -> None:
    """Compile once per ``key`` and pointer signature (never inside a capture), then launch.

    Pointer element types and assumed alignments are part of the compiled
    signature, so they extend the caller's key: a call site that alternates
    between int32 and int64 tensors gets one specialization per type instead
    of reusing a function compiled for the other.
    """
    ns = _glue_kernels()
    signature = tuple(
        arg._csa2_signature if isinstance(arg, ns.cute.Pointer) else None for arg in args
    )
    key = (key, signature)
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up CSA2 CuTe kernels before CUDA Graph capture")
        compiled = _COMPILED[key] = ns.cute.compile(factory(), *args)
    compiled(*args)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_int64(tensor: torch.Tensor) -> torch.Tensor:
    """Contiguous int64 view of an integer tensor (a copy only when it is not already one)."""
    if tensor.dtype != torch.int64:
        tensor = tensor.long()
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def convert_page_tables(
    base: torch.Tensor,
    params: torch.Tensor,
    out: torch.Tensor,
    bad: int,
    scratch_ranges: torch.Tensor | None = None,
    scratch_slots: torch.Tensor | None = None,
) -> None:
    """``out[t, s, c] = base[pool_t, s, c] * scale_t + offset_t``, keeping ``bad``.

    ``base`` is the device int32 ``[pools, seqs, blocks]`` base page rows of the batch.
    ``params`` is int32 ``[tables, 5]`` of
    (pool, scale, offset, scratch pool, scratch pages per block); ``out`` is
    int32 ``[tables, seqs, width]`` with unit column stride. With
    ``scratch_ranges`` (int32 ``[scratch pools, seqs, 2]``) and
    ``scratch_slots`` (int32 ``[scratch pools, seqs, n]``), columns inside a
    range map to the scratch slots like ``PageIndexConverter``.
    """
    tables, seqs, width = out.shape
    if tables == 0 or seqs == 0 or width == 0:
        return
    _check(base.dtype == torch.int32 and base.dim() == 3 and base.is_contiguous(), "base rows")
    _check(base.shape[1] == seqs and width <= base.shape[2], "base rows cover the batch")
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
    ns = _glue_kernels()
    _launch(
        ("page_table_convert", with_scratch, out.device.index),
        lambda: ns.PageTableConvertKernel(with_scratch),
        _ptr(base),
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
        _stream(out.device),
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
) -> None:
    """Device implementation of ``CSA2TrtllmMetadata._refresh_owner_slots`` for
    every owner of the int64 ``[owners, OWNER_DESC_WIDTH]`` ``descriptors``."""
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
    _launch(
        ("owner_slots", descriptors.device.index),
        ns.OwnerSlotsKernel,
        _ptr(kv_lens),
        _ptr(lengths),
        _ptr(query_base),
        _ptr(descriptors, align=8),
        requests,
        owners,
        max_capacity,
        _stream(descriptors.device),
    )


def fill_indexer_descriptors(
    table: torch.Tensor,
    token_requests: torch.Tensor,
    decode_visible: torch.Tensor,
    context_requests: int,
    pages_per_source_page: int,
    page_rows: int,
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
    _check(blocks.dtype == torch.int32 and blocks.is_contiguous(), "blocks")
    _check(context.dtype == torch.int32 and context.is_contiguous(), "context")
    _check(visible.dtype == torch.int32 and visible.is_contiguous(), "visible out")
    _check(valid.dtype == torch.bool and valid.is_contiguous(), "valid")
    page_capacity = int(blocks.shape[1])
    _check(page_rows % 4 == 0 and valid.data_ptr() % 4 == 0, "valid rows must be word aligned")
    ns = _glue_kernels()
    _launch(
        ("indexer_descriptors", blocks.device.index),
        lambda: ns.IndexerDescriptorsKernel(),
        _ptr(table),
        _ptr(token_requests, align=8),
        _ptr(decode_visible, align=8),
        _ptr(blocks),
        _ptr(context),
        _ptr(visible),
        _ptr(valid, torch.int32),
        count,
        int(table.shape[0]),
        int(table.shape[1]),
        int(table.stride(0)) if table.shape[0] > 1 else int(table.shape[1]),
        int(context_requests),
        int(pages_per_source_page),
        int(page_rows),
        page_capacity,
        _stream(blocks.device),
    )


def mask_logits_(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """In-place ``logits.masked_fill_(~valid[:, :width], -inf)`` in one launch."""
    count, width = logits.shape
    if count == 0 or width == 0:
        return logits
    _check(logits.dtype == torch.float32 and logits.stride(1) == 1, "logits")
    _check(valid.dtype == torch.bool and valid.stride(1) == 1 and valid.shape[1] >= width, "valid")
    ns = _glue_kernels()
    _launch(
        ("mask_logits", logits.device.index),
        lambda: ns.MaskLogitsKernel(),
        _ptr(logits),
        _ptr(valid, torch.uint8, align=1),
        int(count),
        int(width),
        int(logits.stride(0)),
        int(valid.stride(0)),
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
    _check(valid.dtype == torch.bool and valid.stride(1) == 1 and valid.shape[1] >= width, "valid")
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
) -> torch.Tensor:
    """Sort validated selections ascending in place with ``-1`` padding.

    Paged mode passes ``valid``; mapped mode passes ``scores`` and ``positions``.
    """
    count, top_k = indices.shape
    if count == 0 or width == 0:
        return indices
    _check(indices.dtype == torch.int32 and indices.stride(1) == 1, "indices int32")
    mapped = scores is not None
    if mapped:
        _check(positions is not None, "mapped finalize requires positions")
        _check(scores.dtype == torch.float32 and scores.stride(1) == 1, "scores")
        _check(positions.dtype in (torch.int32, torch.int64) and positions.stride(1) == 1, "pos")
        _check(scores.shape[1] >= width and positions.shape[1] >= width, "mapped widths")
    else:
        _check(valid is not None and valid.dtype == torch.bool and valid.stride(1) == 1, "valid")
        _check(valid.shape[1] >= width, "valid width")
    ns = _glue_kernels()
    int64 = mapped and positions.dtype == torch.int64
    _launch(
        ("finalize", int(top_k), mapped, indices.device.index),
        lambda: ns.FinalizeSelectionKernel(int(top_k), mapped),
        _ptr(indices),
        _ptr(valid if not mapped else indices, torch.uint8 if not mapped else None, align=1),
        _ptr(scores if mapped else indices),
        _ptr(positions if mapped else indices, align=8 if int64 else 4),
        int(count),
        int(width),
        int(indices.stride(0)),
        int(valid.stride(0)) if not mapped else 0,
        int(scores.stride(0)) if mapped else 0,
        int(positions.stride(0)) if mapped else 0,
        _stream(indices.device),
    )
    return indices


def publish_candidates_(
    scores: torch.Tensor,
    visible: torch.Tensor,
    output: torch.Tensor,
    top_k_blocks: int,
    block_size: int,
    sparse_ids: torch.Tensor | None = None,
    sparse_counts: torch.Tensor | None = None,
    sparse_block: int = 0,
) -> bool:
    """Write one candidate row per query into ``output[count, published_width]`` (int32).

    Device implementation of ``CSA2Indexer._publish_candidates``; with
    ``sparse_block`` it also writes the DeepGEMM sparse-logits inputs
    ``sparse_ids[count, published_width // sparse_block]`` and
    ``sparse_counts[count]``. Returns ``False`` when the geometry is outside the
    kernel's shared-memory budget so the caller keeps the PyTorch path.
    """
    count, width = scores.shape
    blocks = -(-width // block_size)
    if count == 0:
        return True
    if blocks == 0 or blocks > 4096 or block_size <= 0:
        return False
    published_width = int(output.shape[1])
    selected_blocks = min(int(top_k_blocks), blocks)
    _check(selected_blocks * block_size <= published_width, "CSA2 candidate publication width")
    _check(scores.dtype == torch.float32 and scores.stride(1) == 1, "candidate scores")
    _check(visible.dtype in (torch.int32, torch.int64) and visible.numel() == count, "visible")
    _check(
        output.dtype == torch.int32 and output.shape[0] == count and output.stride(1) == 1,
        "CSA2 candidate output",
    )
    if sparse_block:
        _check(sparse_ids is not None and sparse_counts is not None, "sparse outputs")
        _check(
            sparse_ids.dtype == torch.int32
            and sparse_ids.shape == (count, published_width // sparse_block)
            and sparse_ids.stride(1) == 1,
            "CSA2 sparse block ids",
        )
        _check(
            sparse_counts.dtype == torch.int32
            and sparse_counts.numel() == count
            and sparse_counts.is_contiguous(),
            "CSA2 sparse counts",
        )
    else:
        sparse_ids = sparse_counts = output  # unread placeholders
    padded = 1 << (blocks - 1).bit_length()
    ns = _glue_kernels()
    _launch(
        ("candidate_publish", int(block_size), padded, int(sparse_block), scores.device.index),
        lambda: ns.CandidatePublishKernel(int(block_size), padded, int(sparse_block)),
        _ptr(scores),
        _ptr(visible.contiguous(), align=8 if visible.dtype == torch.int64 else 4),
        _ptr(output, align=4),
        _ptr(sparse_ids, align=4),
        _ptr(sparse_counts, align=4),
        int(count),
        int(width),
        int(blocks),
        int(selected_blocks),
        published_width,
        int(scores.stride(0)),
        int(output.stride(0)),
        int(sparse_ids.stride(0)),
        _stream(scores.device),
    )
    return True


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

    @cute.jit
    def e2m1_code(scaled: Float32) -> Int32:
        """Round-to-nearest-even E2M1 code with sign; NaN encodes as 7."""
        magnitude = scaled
        if magnitude < Float32(0.0):
            magnitude = -magnitude
        code = Int32(0)
        if magnitude > Float32(0.25):
            code = code + 1
        if magnitude >= Float32(0.75):
            code = code + 1
        if magnitude > Float32(1.25):
            code = code + 1
        if magnitude >= Float32(1.75):
            code = code + 1
        if magnitude > Float32(2.5):
            code = code + 1
        if magnitude >= Float32(3.5):
            code = code + 1
        if magnitude > Float32(5.0):
            code = code + 1
        if magnitude != magnitude:
            code = Int32(7)
        return code | Int32((bitcast_f32_u32(scaled) >> 28) & 8)

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
    def store_e4m3x4(pointer: cute.Pointer, element_offset: Int64, result, quant_scale: Float32):
        """Quantize four values as the native static FP8 path does (BF16 round, scale, saturate)."""
        fragment = cute.make_rmem_tensor((4,), Uint8)
        for j in cutlass.range_constexpr(4):
            rounded = Float32(cutlass.BFloat16(result[j])) * quant_scale
            fragment[j] = Uint8(f32_to_e4m3_byte(rounded))
        address = pointer.toint() + element_offset
        target = cute.make_ptr(Uint8, address, cute.AddressSpace.gmem, assumed_align=4)
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
            page = logical // page_size
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
                slot = physical * page_size + logical % page_size
        return slot

    @cute.jit
    def highest_scale_byte(bitmap: cute.Pointer, index: Int32) -> Int32:
        """Largest byte value recorded in source ``index``'s 8-word bitmap, or -1.

        ``StageScaleKernel`` records which group-scale byte values occur rather
        than their maximum, so that its CTAs can combine with ``atomic_or``. One
        bit per value makes the recovery exact: the highest set bit is the maximum.
        """
        largest = Int32(-1)
        for word in cutlass.range_constexpr(8):
            bits = Int32(tensor1d(bitmap, Int64(16))[Int64(index) * 8 + word])
            for bit in cutlass.range_constexpr(32):
                if ((bits >> bit) & 1) != 0:
                    largest = Int32(word * 32 + bit)
        return largest

    # ---------------------------------------------------- K7: quantize rows

    class QuantizeScatterKernel:
        """Quantize BF16 rows and scatter them into packed cache rows, up to four jobs per launch.

        Each job (``QuantizeJobSpec``) writes in mode ``row`` (``pool[slot, :]``
        with a row stride), ``footer`` (native page-footer pages of ``page_rows``
        rows) or ``split`` (data and scale bytes in two contiguous buffers); with
        ``norm`` the row is RMS-normalized first (``rsqrt`` and one BF16 rounding
        as the native RMSNorm kernel) and with ``rope`` the last ``rope_dim``
        channels are rotated in interleaved GPT-J pairs (one BF16 rounding as
        ``mla_rope_inplace``). One CTA quantizes one row and the grid concatenates
        the jobs' rows, so a layer's SWA, main, index and index-query writes are
        one launch. Each thread owns four channels; quantization groups reduce
        with butterfly shuffles.
        """

        def __init__(self, jobs):
            if not 1 <= len(jobs) <= _MAX_QUANTIZE_JOBS:
                raise ValueError("A CSA2 quantize launch takes one to four jobs")
            self.jobs = tuple(jobs)
            self.threads = max(job.threads for job in self.jobs)

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
                grid=[rows0 + rows1 + rows2 + rows3, 1, 1],
                block=[self.threads, 1, 1],
                stream=stream,
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
            row, _, _ = cute.arch.block_idx()
            smem = SmemAllocator()
            partial = smem.allocate_tensor(
                Float32, cute.make_layout(max(self.threads // 32, 1)), 16
            )
            if row < rows0:
                self._quantize_row(
                    0,
                    pool0,
                    scale0,
                    slots0,
                    values0,
                    weight0,
                    positions0,
                    cos_sin0,
                    row,
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
                    row1 = row - rows0
                    if row1 < rows1:
                        self._quantize_row(
                            1,
                            pool1,
                            scale1,
                            slots1,
                            values1,
                            weight1,
                            positions1,
                            cos_sin1,
                            row1,
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
                            row2 = row1 - rows1
                            if row2 < rows2:
                                self._quantize_row(
                                    2,
                                    pool2,
                                    scale2,
                                    slots2,
                                    values2,
                                    weight2,
                                    positions2,
                                    cos_sin2,
                                    row2,
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
                                    row3 = row2 - rows2
                                    self._quantize_row(
                                        3,
                                        pool3,
                                        scale3,
                                        slots3,
                                        values3,
                                        weight3,
                                        positions3,
                                        cos_sin3,
                                        row3,
                                        capacity3,
                                        pool_stride3,
                                        slot_stride3,
                                        value_stride3,
                                        page_rows3,
                                        eps3,
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
            capacity: Int64,
            pool_stride: Int64,
            slot_stride: Int64,
            value_stride: Int64,
            page_rows: Int64,
            eps: Float32,
            partial,
        ):
            job = self.jobs[job_index]
            tid, _, _ = cute.arch.thread_idx()
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
                    if cutlass.const_expr(self.threads > 32):
                        if tid % 32 == 0:
                            partial[tid // 32] = sumsq
                        cute.arch.barrier()
                        sumsq = Float32(0.0)
                        for warp in cutlass.range_constexpr(self.threads // 32):
                            sumsq = sumsq + partial[warp]
                    inv = rsqrt_approx(sumsq / Float32(job.head_dim) + eps)
                    weight_t = tensor1d(norm_weight, Int64(job.head_dim))
                    for j in cutlass.range_constexpr(4):
                        if active:
                            w = Float32(weight_t[first + j])
                            x[j] = Float32(cutlass.BFloat16(x[j] * inv * w))
                if cutlass.const_expr(job.rope_dim > 0):
                    # Interleaved GPT-J RoPE on the trailing channels: pair
                    # ``p`` of the row rotates with cos/sin ``p`` of its position.
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
                            x[2 * k] = Float32(cutlass.BFloat16(c * a - sn * b))
                            x[2 * k + 1] = Float32(cutlass.BFloat16(c * b + sn * a))
                local_max = Float32(0.0)
                local_nan = Float32(0.0)
                for j in cutlass.range_constexpr(4):
                    v = x[j]
                    local_max = cute.arch.fmax(local_max, cute.arch.fmax(v, -v))
                    if v != v:
                        local_nan = Float32(1.0)
                offset = 1
                while offset < job.group_lanes:
                    local_max = cute.arch.fmax(
                        local_max, cute.arch.shuffle_sync_bfly(local_max, offset=offset)
                    )
                    local_nan = cute.arch.fmax(
                        local_nan, cute.arch.shuffle_sync_bfly(local_nan, offset=offset)
                    )
                    offset = offset * 2
                group_nan = local_nan > Float32(0.5)
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
                    if group_nan:
                        scale_byte = Int32(0)
                if active:
                    if cutlass.const_expr(job.cache_format == "swa"):
                        for j in cutlass.range_constexpr(4):
                            scaled = div_rn(x[j], scale)
                            byte = f32_to_e4m3_byte(scaled)
                            if group_nan | (scaled != scaled):
                                byte = Int32(127)
                            pool_t[data_base + Int64(first + j)] = Uint8(byte)
                    else:
                        codes = cute.make_rmem_tensor((4,), Int32)
                        for j in cutlass.range_constexpr(4):
                            code = e2m1_code(div_rn(x[j], scale))
                            if group_nan:
                                code = Int32(7)
                            codes[j] = code
                        pool_t[data_base + Int64(first // 2)] = Uint8(codes[0] | (codes[1] << 4))
                        pool_t[data_base + Int64(first // 2) + 1] = Uint8(
                            codes[2] | (codes[3] << 4)
                        )
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
        destination), ``counts`` and the mapped main slots. Layers sharing a
        KV owner and an index source share one compaction per step.
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
            # Pass 2: assign destinations in column order.
            for column in cutlass.range(begin, end, 1):
                valid = cutlass.Boolean(False)
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
                    destination = destination + 1
            if tid == 0:
                tensor1d(counts, Int64(count))[q] = total

    class StageDequantKernel:
        """Decode compacted selections into the native primary/extra staging pools.

        CTA ``(q, 0)`` also publishes the layer's ``lengths`` (``max(count, 1)``)
        and native ``indices``; CTA ``(0, 0)`` resets the scheduler counter and
        derives the FP8 BMM scales.
        """

        THREADS = 128

        def __init__(self, fp8: bool):
            self.fp8 = fp8

        @cute.jit
        def __call__(
            self,
            swa_pool: cute.Pointer,
            main_pool: cute.Pointer,
            swa_slots: cute.Pointer,
            main_slots: cute.Pointer,  # int64 [count, main_width] mapped
            tags: cute.Pointer,
            counts: cute.Pointer,
            primary: cute.Pointer,  # [count, 128, 512]
            extra: cute.Pointer,  # [count, extra_rows, 512]
            quant_scale: cute.Pointer,
            lengths: cute.Pointer,  # int32 [count]
            indices: cute.Pointer,  # int32 [count, capacity]
            counter: cute.Pointer,  # uint32 [1]
            q_dequant_scale: cute.Pointer,  # float32 [1]
            dequant_scale: cute.Pointer,  # float32 [1]
            bmm1_scale: cute.Pointer,  # float32 [2]
            bmm2_scale: cute.Pointer,  # float32 [1]
            count: Int32,
            rows: Int32,
            swa_width: Int32,
            main_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            slot_rs: Int64,
            slot_cs: Int64,
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
                main_slots,
                tags,
                counts,
                primary,
                extra,
                quant_scale,
                lengths,
                indices,
                counter,
                q_dequant_scale,
                dequant_scale,
                bmm1_scale,
                bmm2_scale,
                count,
                swa_width,
                main_width,
                swa_rs,
                swa_cs,
                slot_rs,
                slot_cs,
                swa_pool_stride,
                main_pool_stride,
                swa_capacity,
                main_capacity,
                capacity,
                extra_rows,
                softmax_scale,
            ).launch(grid=[count, rows, 1], block=[self.THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            swa_pool: cute.Pointer,
            main_pool: cute.Pointer,
            swa_slots: cute.Pointer,
            main_slots: cute.Pointer,
            tags: cute.Pointer,
            counts: cute.Pointer,
            primary: cute.Pointer,
            extra: cute.Pointer,
            quant_scale: cute.Pointer,
            lengths: cute.Pointer,
            indices: cute.Pointer,
            counter: cute.Pointer,
            q_dequant_scale: cute.Pointer,
            dequant_scale: cute.Pointer,
            bmm1_scale: cute.Pointer,
            bmm2_scale: cute.Pointer,
            count: Int32,
            swa_width: Int32,
            main_width: Int32,
            swa_rs: Int64,
            swa_cs: Int64,
            slot_rs: Int64,
            slot_cs: Int64,
            swa_pool_stride: Int64,
            main_pool_stride: Int64,
            swa_capacity: Int64,
            main_capacity: Int64,
            capacity: Int32,
            extra_rows: Int32,
            softmax_scale: Float32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, destination, _ = cute.arch.block_idx()
            total = Int32(tensor1d(counts, Int64(count))[q])
            first = tid * 4
            result = cute.make_rmem_tensor((4,), Float32)
            for j in cutlass.range_constexpr(4):
                result[j] = Float32(0.0)
            if destination < total:
                tag = Int32(
                    tensor2d(tags, Int64(count), Int64(capacity), Int64(capacity))[q, destination]
                )
                # Tags hold columns, and the SWA slots backing them are private to
                # each layer, so a column that compaction resolved for the group's
                # first layer can be empty in this one. The pools are addressed
                # without bounds checking, so the slot is re-validated here and
                # ``decode_channels`` leaves the zero-initialized row alone for a
                # slot outside the pool or a column outside this layer's widths.
                if tag < swa_width:
                    slot = Int64(
                        tensor1d(swa_slots, Int64(2**62))[Int64(q) * swa_rs + Int64(tag) * swa_cs]
                    )
                    decode_channels(
                        swa_pool,
                        slot * swa_pool_stride,
                        first,
                        512,
                        "swa",
                        (slot >= 0) & (slot < swa_capacity),
                        result,
                    )
                elif tag < swa_width + main_width:
                    slot = Int64(
                        tensor1d(main_slots, Int64(2**62))[
                            Int64(q) * slot_rs + Int64(tag - swa_width) * slot_cs
                        ]
                    )
                    decode_channels(
                        main_pool,
                        slot * main_pool_stride,
                        first,
                        512,
                        "main",
                        (slot >= 0) & (slot < main_capacity),
                        result,
                    )
            target = primary
            element = (Int64(q) * 128 + Int64(destination)) * 512 + Int64(first)
            if destination >= 128:
                target = extra
                element = (Int64(q) * Int64(extra_rows) + Int64(destination - 128)) * 512 + Int64(
                    first
                )
            if cutlass.const_expr(self.fp8):
                store_e4m3x4(target, element, result, Float32(tensor1d(quant_scale, Int64(1))[0]))
            else:
                store_bf16x4(target, element, result)
            if destination == 0:
                length = total
                if length < 1:
                    length = Int32(1)
                if tid == 0:
                    tensor1d(lengths, Int64(count))[q] = length
                indices_t = tensor2d(indices, Int64(count), Int64(capacity), Int64(capacity))
                extra_capacity = capacity - 128
                for column in cutlass.range(tid, capacity, self.THREADS):
                    physical = Int32(q) * extra_capacity + column - 128
                    if column < 128:
                        physical = Int32(q) * 128 + column
                    value = Int32(-1)
                    if column < length:
                        value = physical
                    indices_t[q, column] = value
                if (q == 0) & (tid == 0):
                    tensor1d(counter, Int64(1))[0] = Uint32(0)
                    if cutlass.const_expr(self.fp8):
                        # Q and KV may carry different staging scales, so BMM1's
                        # dequant is their product rather than one scale squared.
                        dq_q = Float32(tensor1d(q_dequant_scale, Int64(1))[0])
                        dq = Float32(tensor1d(dequant_scale, Int64(1))[0])
                        bmm1 = (dq_q * dq) * softmax_scale
                        bmm1_t = tensor1d(bmm1_scale, Int64(2))
                        bmm1_t[0] = bmm1
                        bmm1_t[1] = bmm1 * Float32(1.4426950408889634)
                        tensor1d(bmm2_scale, Int64(1))[0] = dq

    # --------------------------------------- K12/K13: shared prefill staging

    class SharedCompactKernel:
        """Map one query's selections onto the shared bank rows and mark used rows."""

        THREADS = 256

        def __init__(self, fp8: bool, has_main: bool, premapped: bool):
            self.fp8 = fp8
            self.has_main = has_main
            self.premapped = premapped

        @cute.jit
        def __call__(
            self,
            swa_slots: cute.Pointer,
            main_logical: cute.Pointer,
            main_slots: cute.Pointer,  # int64 physical [count, main_width] strided (premapped)
            requests: cute.Pointer,  # int64 [count] (plan query requests)
            positions: cute.Pointer,  # int32 [count]
            swa_starts: cute.Pointer,  # int64 [requests]
            swa_offsets: cute.Pointer,  # int64 [requests + 1]
            main_offsets: cute.Pointer,  # int64 [requests + 1]
            visible: cute.Pointer,  # int64 [count]
            table: cute.Pointer,  # owner page table
            selected: cute.Pointer,  # int32 [bank_rows]
            indices: cute.Pointer,  # int32 [count, capacity]
            lengths: cute.Pointer,  # int32 [count]
            counter: cute.Pointer,
            q_dequant_scale: cute.Pointer,
            dequant_scale: cute.Pointer,
            bmm1_scale: cute.Pointer,
            bmm2_scale: cute.Pointer,
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
            request_count: Int32,
            capacity: Int32,
            softmax_scale: Float32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                swa_slots,
                main_logical,
                main_slots,
                requests,
                positions,
                swa_starts,
                swa_offsets,
                main_offsets,
                visible,
                table,
                selected,
                indices,
                lengths,
                counter,
                q_dequant_scale,
                dequant_scale,
                bmm1_scale,
                bmm2_scale,
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
                request_count,
                capacity,
                softmax_scale,
            ).launch(grid=[count, 1, 1], block=[self.THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            swa_slots: cute.Pointer,
            main_logical: cute.Pointer,
            main_slots: cute.Pointer,
            requests: cute.Pointer,
            positions: cute.Pointer,
            swa_starts: cute.Pointer,
            swa_offsets: cute.Pointer,
            main_offsets: cute.Pointer,
            visible: cute.Pointer,
            table: cute.Pointer,
            selected: cute.Pointer,
            indices: cute.Pointer,
            lengths: cute.Pointer,
            counter: cute.Pointer,
            q_dequant_scale: cute.Pointer,
            dequant_scale: cute.Pointer,
            bmm1_scale: cute.Pointer,
            bmm2_scale: cute.Pointer,
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
            request_count: Int32,
            capacity: Int32,
            softmax_scale: Float32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            q, _, _ = cute.arch.block_idx()
            smem = SmemAllocator()
            prefix = smem.allocate_tensor(Int32, cute.make_layout(self.THREADS), 16)
            if (q == 0) & (tid == 0):
                tensor1d(counter, Int64(1))[0] = Uint32(0)
                if cutlass.const_expr(self.fp8):
                    # Q and KV may carry different staging scales, so BMM1's
                    # dequant is their product rather than one scale squared.
                    dq_q = Float32(tensor1d(q_dequant_scale, Int64(1))[0])
                    dq = Float32(tensor1d(dequant_scale, Int64(1))[0])
                    scale = dq_q * dq * softmax_scale
                    bmm1_t = tensor1d(bmm1_scale, Int64(2))
                    bmm1_t[0] = scale
                    bmm1_t[1] = scale * Float32(1.4426950408889634)
                    tensor1d(bmm2_scale, Int64(1))[0] = dq
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
                swa_start = Int64(tensor1d(swa_starts, Int64(request_count))[Int32(safe_request)])
                swa_offset_t = tensor1d(swa_offsets, Int64(request_count) + 1)
                swa_begin = Int64(swa_offset_t[Int32(safe_request)])
                swa_end = Int64(swa_offset_t[Int32(safe_request) + 1])
                main_offset_t = tensor1d(main_offsets, Int64(request_count) + 1)
                main_begin = Int64(main_offset_t[Int32(safe_request)])
                main_end = Int64(main_offset_t[Int32(safe_request) + 1])
            visible_length = Int64(0)
            if cutlass.const_expr(self.has_main):
                visible_length = Int64(tensor1d(visible, n)[q])
            chunk = (capacity + self.THREADS - 1) // self.THREADS
            begin = tid * chunk
            end = begin + chunk
            if end > capacity:
                end = capacity
            swa_t = tensor1d(swa_slots, Int64(2**62))
            logical_t = tensor1d(main_logical, Int64(2**62))
            slot_t = tensor1d(main_slots, Int64(2**62))
            local = Int32(0)
            for column in cutlass.range(begin, end, 1):
                valid = cutlass.Boolean(False)
                if column < swa_width:
                    slot = Int64(swa_t[Int64(q) * swa_rs + Int64(column) * swa_cs])
                    logical_swa = position - Int64(swa_width) + 1 + Int64(column)
                    swa_row = swa_begin + logical_swa - swa_start
                    valid = (
                        (slot >= 0)
                        & (slot < swa_capacity)
                        & (swa_row >= swa_begin)
                        & (swa_row < swa_end)
                    )
                elif column < swa_width + main_width:
                    if cutlass.const_expr(self.has_main):
                        main_column = column - swa_width
                        logical = Int64(
                            logical_t[Int64(q) * main_rs + Int64(main_column) * main_cs]
                        )
                        slot = Int64(-1)
                        if cutlass.const_expr(self.premapped):
                            slot = Int64(slot_t[Int64(q) * slot_rs + Int64(main_column) * slot_cs])
                        else:
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
                        valid = (slot >= 0) & (slot < main_capacity) & (logical >= 0)
                        valid = valid & (logical < main_end - main_begin)
                valid = valid & request_valid
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
            indices_t = tensor2d(indices, n, Int64(capacity), Int64(capacity))
            selected_t = tensor1d(selected, Int64(2**62))
            for column in cutlass.range(begin, end, 1):
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
                    )
                elif column < swa_width + main_width:
                    if cutlass.const_expr(self.has_main):
                        main_column = column - swa_width
                        logical = Int64(
                            logical_t[Int64(q) * main_rs + Int64(main_column) * main_cs]
                        )
                        slot = Int64(-1)
                        if cutlass.const_expr(self.premapped):
                            slot = Int64(slot_t[Int64(q) * slot_rs + Int64(main_column) * slot_cs])
                        else:
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
                        bank_row = main_begin + logical
                        valid = (slot >= 0) & (slot < main_capacity) & (logical >= 0)
                        valid = valid & (logical < main_end - main_begin)
                valid = valid & request_valid
                if valid:
                    indices_t[q, destination] = Int32(bank_row)
                    cute.arch.atomic_or(selected_t.iterator + Int32(bank_row), Int32(1))
                    destination = destination + 1
            for column in cutlass.range(tid, capacity, self.THREADS):
                if column >= total:
                    indices_t[q, column] = Int32(-1)
            if tid == 0:
                length = total
                if length < 1:
                    length = Int32(1)
                    indices_t[q, 0] = Int32(0)
                tensor1d(lengths, n)[q] = length

    class SharedDecodeKernel:
        """Decode every marked shared bank row once from its owner page (``_shared_decode_rows``)."""

        THREADS = 128

        def __init__(self, cache_format: str, fp8: bool):
            _format_geometry(cache_format, 512)
            self.cache_format = cache_format
            self.fp8 = fp8

        @cute.jit
        def __call__(
            self,
            pool: cute.Pointer,
            pages: cute.Pointer,  # int32 [requests, columns] strided
            starts: cute.Pointer,  # int64 [requests] (swa only)
            offsets: cute.Pointer,  # int64 [requests + 1]
            selected: cute.Pointer,  # int32 [bank rows]
            bank: cute.Pointer,  # bf16/fp8 [bank rows, 512]
            quant_scale: cute.Pointer,
            rows: Int32,
            row_base: Int32,
            pool_capacity: Int64,
            pool_stride: Int64,
            page_rows: Int32,
            page_cols: Int32,
            page_stride: Int64,
            page_size: Int64,
            request_count: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                pool,
                pages,
                starts,
                offsets,
                selected,
                bank,
                quant_scale,
                row_base,
                pool_capacity,
                pool_stride,
                page_rows,
                page_cols,
                page_stride,
                page_size,
                request_count,
            ).launch(grid=[rows, 1, 1], block=[self.THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            pool: cute.Pointer,
            pages: cute.Pointer,
            starts: cute.Pointer,
            offsets: cute.Pointer,
            selected: cute.Pointer,
            bank: cute.Pointer,
            quant_scale: cute.Pointer,
            row_base: Int32,
            pool_capacity: Int64,
            pool_stride: Int64,
            page_rows: Int32,
            page_cols: Int32,
            page_stride: Int64,
            page_size: Int64,
            request_count: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            row = Int64(block) + Int64(row_base)
            marked = Int32(tensor1d(selected, Int64(2**62))[row]) != 0
            if marked:
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
                logical = row - Int64(offsets_t[request])
                if cutlass.const_expr(self.cache_format == "swa"):
                    logical = logical + Int64(tensor1d(starts, Int64(request_count))[request])
                column = logical // page_size
                valid = (logical >= 0) & (column >= 0) & (column < Int64(page_cols))
                page = Int64(-1)
                if valid:
                    page = Int64(
                        tensor2d(pages, Int64(page_rows), Int64(page_cols), page_stride)[
                            request, Int32(column)
                        ]
                    )
                slot = page * page_size + logical % page_size
                valid = valid & (page >= 0) & (slot >= 0) & (slot < pool_capacity)
                first = tid * 4
                result = cute.make_rmem_tensor((4,), Float32)
                decode_channels(
                    pool, slot * pool_stride, first, 512, self.cache_format, valid, result
                )
                element = row * 512 + Int64(first)
                if cutlass.const_expr(self.fp8):
                    store_e4m3x4(bank, element, result, Float32(tensor1d(quant_scale, Int64(1))[0]))
                else:
                    store_bf16x4(bank, element, result)

    # ------------------------------------------ K14/K15: staged FP8 KV range

    class StageScaleKernel:
        """Reduce the selected rows' group-scale bytes into a 256-bit presence bitmap.

        Both persistent encodings store non-negative group scales in the tail of
        each packed row, so the largest scale byte of the selected rows bounds
        every magnitude staging can decode. Reading only that tail costs one pass
        over 32 or 16 bytes per row instead of its 256 or 512 payload bytes.

        The reduction records which byte values occur rather than their maximum,
        because a bitmap combines with ``atomic_or`` where a maximum would need an
        atomic this glue does not otherwise use. One bit per byte value is exact:
        ``StageScaleFinalizeKernel`` recovers the maximum as the highest set bit.
        Each CTA reduces its own columns first, so it contributes a single atomic.
        """

        THREADS = 128

        def __init__(self, groups: int, limit: int) -> None:
            self.groups = groups
            self.limit = limit

        @cute.jit
        def __call__(
            self,
            pool: cute.Pointer,
            slots: cute.Pointer,
            bitmap: cute.Pointer,  # int32 [sources, 8]
            rows: Int32,
            width: Int32,
            capacity: Int64,
            pool_stride: Int64,
            row_stride: Int64,
            col_stride: Int64,
            scale_base: Int64,
            index: Int32,
            mapping: tuple | None,
            stream: cuda.CUstream,
        ):
            blocks = (width + Int32(self.THREADS - 1)) // Int32(self.THREADS)
            self.kernel(
                pool,
                slots,
                bitmap,
                width,
                capacity,
                pool_stride,
                row_stride,
                col_stride,
                scale_base,
                index,
                mapping,
            ).launch(grid=[rows, blocks, 1], block=[self.THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            pool: cute.Pointer,
            slots: cute.Pointer,
            bitmap: cute.Pointer,
            width: Int32,
            capacity: Int64,
            pool_stride: Int64,
            row_stride: Int64,
            col_stride: Int64,
            scale_base: Int64,
            index: Int32,
            mapping: tuple | None,
        ):
            tid, _, _ = cute.arch.thread_idx()
            query, block, _ = cute.arch.block_idx()
            smem = SmemAllocator()
            warps = smem.allocate_tensor(Int32, cute.make_layout(self.THREADS // 32), 16)
            column = Int32(block) * Int32(self.THREADS) + Int32(tid)
            # -1 is "this thread saw no bounded scale", below every real byte.
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
                    bytes_t = tensor1d(pool, Int64(2**62))
                    base = slot * pool_stride + scale_base
                    for group in cutlass.range_constexpr(self.groups):
                        # ``limit`` drops the non-finite encodings: 0x7F is E4M3 NaN
                        # and also rejects every negative E4M3 byte, and byte 255 is
                        # the ``swa`` infinity. Neither can be bounded by any scale,
                        # so they must not drive the reduction.
                        value = Int32(bytes_t[base + Int64(group)]) & 255
                        if (value < Int32(self.limit)) & (value > largest):
                            largest = value
            offset = 1
            while offset < 32:
                other = cute.arch.shuffle_sync_bfly(largest, offset=offset)
                if other > largest:
                    largest = other
                offset = offset * 2
            if tid % 32 == 0:
                warps[tid // 32] = largest
            cute.arch.barrier()
            if tid == 0:
                for warp in cutlass.range_constexpr(self.THREADS // 32):
                    other = Int32(warps[warp])
                    if other > largest:
                        largest = other
                if largest >= 0:
                    bitmap_t = tensor1d(bitmap, Int64(16))
                    word = index * Int32(8) + (largest >> 5)
                    cute.arch.atomic_or(bitmap_t.iterator + word, Int32(1) << (largest & 31))

    class StageScaleFinalizeKernel:
        """Turn the observed group-scale bitmaps into one power-of-two staging scale.

        Both encoders clamp their payload -- ``main`` codes at the E2M1 maximum of
        6, ``swa`` values at +-448 -- so ``6 * e4m3(byte)`` and ``448 * 2**(byte -
        127)`` are exact ceilings on what a row can decode to, whichever way the
        encoder rounded its group scale. The smallest power of two that maps the
        larger ceiling onto 448 puts every staged value inside the E4M3 range, and
        a power of two costs no relative precision: E4M3's relative error is
        identical in every binade, so the factor only moves the exponent.
        """

        # Sentinel exponent for "no group scale was observed", below every real one.
        ABSENT = -1000

        def __init__(self, has_q: bool, q_shared: bool):
            self.has_q = has_q
            self.q_shared = q_shared

        @cute.jit
        def __call__(
            self,
            bitmap: cute.Pointer,  # int32 [sources, 8]
            dequant: cute.Pointer,  # float32 [1]
            quant: cute.Pointer,  # float32 [1]
            q_amax: cute.Pointer,  # float32 [1]
            minimum: Int32,
            maximum: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(bitmap, dequant, quant, q_amax, minimum, maximum).launch(
                grid=[1, 1, 1], block=[32, 1, 1], stream=stream
            )

        @cute.kernel
        def kernel(
            self,
            bitmap: cute.Pointer,
            dequant: cute.Pointer,
            quant: cute.Pointer,
            q_amax: cute.Pointer,
            minimum: Int32,
            maximum: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            if tid == 0:
                swa_byte = highest_scale_byte(bitmap, Int32(0))
                main_byte = highest_scale_byte(bitmap, Int32(1))
                # 448 * 2**k >= 448 * 2**(byte - 127) holds exactly when k >= byte - 127.
                swa_exponent = Int32(self.ABSENT)
                if swa_byte >= 0:
                    swa_exponent = swa_byte - 127
                main_exponent = Int32(self.ABSENT)
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
                # Staging nothing at all keeps the historical unit scale.
                if exponent == Int32(self.ABSENT):
                    exponent = Int32(0)
                if cutlass.const_expr(self.has_q):
                    amax = Float32(tensor1d(q_amax, Int64(1))[0])
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
                            # once clamped below, simply leaves the historical unit
                            # scale in place.
                            allowed = Int32(amax_bits >> 23) - Int32(127)
                        else:
                            # Q carries the reciprocal scale so that BMM1's dequant
                            # product stays exactly one, which turns Q's own headroom
                            # into a ceiling on the exponent: its codes land at
                            # 2**exponent * amax(|Q|), so the exponent may not exceed
                            # log2(448 / amax(|Q|)). Same 1.75 * 2**8 split as above,
                            # one binade tighter when the amax mantissa exceeds 1.75.
                            # A Q that already saturates on its own yields a negative
                            # bound and, once clamped below, simply leaves the
                            # historical unit scale in place.
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

    class Namespace:
        pass

    ns = Namespace()
    ns.QuantizeScatterKernel = QuantizeScatterKernel
    ns.GatherDequantKernel = GatherDequantKernel
    ns.ScatterPackedKernel = ScatterPackedKernel
    ns.StageCompactKernel = StageCompactKernel
    ns.StageDequantKernel = StageDequantKernel
    ns.SharedCompactKernel = SharedCompactKernel
    ns.SharedDecodeKernel = SharedDecodeKernel
    ns.StageScaleKernel = StageScaleKernel
    ns.StageScaleFinalizeKernel = StageScaleFinalizeKernel
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
    _launch(
        ("quantize", tuple(job.spec for job in jobs), device.index),
        lambda: ns.QuantizeScatterKernel([job.spec for job in jobs]),
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
    cutlass = _glue_kernels().cutlass
    table_args = (
        _ptr(table),
        cutlass.Int32(table.shape[0]),
        cutlass.Int32(table.shape[1]),
        cutlass.Int64(table.stride(0) if table.shape[0] > 1 else table.shape[1]),
        cutlass.Int64(page_size),
        cutlass.Int64(max_positions),
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


def derive_stage_kv_scales(
    sources,
    bitmap: torch.Tensor,
    dequant: torch.Tensor,
    quant: torch.Tensor,
    *,
    main_mapping: tuple | None = None,
    q_amax: torch.Tensor | None = None,
    q_shared: bool = False,
    minimum: int = 0,
    maximum: int = 126,
) -> None:
    """Fill ``dequant``/``quant`` with a range-aware power-of-two staging KV scale.

    ``sources`` holds the ``(pool, slots, cache_format)`` triples whose rows the
    staging pass will decode. Staging expands the persistent per-group scales back
    to real magnitudes, so a unit per-tensor scale saturates every channel above
    448; the bound here is read from those same group-scale bytes, which makes it
    track the forward's actual range instead of a calibrated constant. ``bitmap``
    is a 16-word int32 scratch holding one presence bitmap per format. Both
    outputs are written in place on the device, so CUDA Graph replay observes the
    live value. When ``main_mapping`` is present, MAIN slots are logical positions;
    the reduction applies the same page and visibility mapping as staging.

    Duplicated or over-covered selections are harmless: the reduction is a maximum
    over an exact per-row ceiling, so any superset of the decoded rows still yields
    a scale that cannot saturate.

    ``q_amax`` is a device scalar holding ``amax(|Q|)``, and ``q_shared`` says
    which way Q carries the scale. By default Q takes the reciprocal to keep BMM1's
    dequant product at one, so the cap keeps Q from saturating in exchange; with
    ``q_shared`` Q is divided by this very scale, so the cap keeps Q's own amax at
    unit magnitude or above instead. ``minimum`` defaults to zero because only the
    saturating end is a correctness problem: a forward that needs no extra range
    keeps the historical unit scale, byte for byte.
    """
    _check(minimum <= maximum, "CSA2 staging scale bounds must not be inverted")
    _check(
        bitmap.dtype == torch.int32 and bitmap.is_contiguous() and bitmap.numel() >= 16,
        "CSA2 staging scale bitmap must be 16 contiguous int32 words",
    )
    for buffer in (dequant, quant):
        _check(
            buffer.dtype == torch.float32 and buffer.numel() == 1 and buffer.is_contiguous(),
            "CSA2 staging scales must be single contiguous float32 elements",
        )
    _check(
        q_amax is None or (q_amax.dtype == torch.float32 and q_amax.numel() == 1),
        "CSA2 staging scales need a single float32 Q amax",
    )
    device = bitmap.device
    ns = _cache_kernels()
    bitmap.zero_()
    for pool, slots, cache_format in sources:
        _, data_bytes, scale_bytes = _format_geometry(cache_format, _STAGE_HEAD_DIM)
        _check(
            pool.ndim == 2 and pool.shape[1] == data_bytes + scale_bytes and pool.stride(1) == 1,
            "CSA2 staging scale sources must be packed rows with unit column stride",
        )
        rows = int(slots.shape[0])
        width = int(slots.shape[1]) if slots.ndim > 1 else 0
        if not rows or not width:
            continue
        mapping = None
        if cache_format == "main" and main_mapping is not None:
            table_args, requests, visible = _page_mapping_args(main_mapping, rows, device)
            mapping = (_ptr(requests, align=8), _ptr(visible, align=8), table_args)
        mapping_signature = (
            None
            if mapping is None
            else tuple(
                pointer._csa2_signature for pointer in (mapping[0], mapping[1], mapping[2][0])
            )
        )
        _launch(
            ("stage_scale", cache_format, mapping_signature, device.index),
            lambda groups=scale_bytes, limit=_SCALE_BYTE_LIMITS[cache_format]: (
                ns.StageScaleKernel(groups, limit)
            ),
            _ptr(pool, align=1),
            _ptr(slots, align=8 if slots.dtype == torch.int64 else 4),
            _ptr(bitmap),
            rows,
            width,
            int(pool.shape[0]),
            int(pool.stride(0)),
            int(slots.stride(0)),
            int(slots.stride(1)),
            data_bytes,
            _SCALE_SOURCE_SLOTS[cache_format],
            mapping,
            _stream(device),
        )
    has_q = q_amax is not None
    _launch(
        ("stage_scale_finalize", has_q, q_shared, device.index),
        lambda: ns.StageScaleFinalizeKernel(has_q, q_shared),
        _ptr(bitmap),
        _ptr(dequant),
        _ptr(quant),
        _ptr(q_amax if has_q else _constant_zeros(device, (1,), torch.float32)),
        int(minimum),
        int(maximum),
        _stream(device),
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
):
    """Compact selections (once per staging group) and decode them into the staging pools.

    Main selections are either physical ``main_slots`` or ``main_logical``
    positions resolved in-kernel through ``main_mapping`` (``page_table``,
    ``page_size``, ``max_positions``, ``requests``, ``visible``), exactly as
    ``CSA2TrtllmMetadata.global_slot_tile`` with visibility. ``scratch`` holds
    the group's ``tags``/``counts`` (and, in the mapped form, ``main_slots``
    receives the physical rows); ``compact=False`` reuses them from an
    earlier layer of the same group in this forward.
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
    tags, counts = scratch["tags"], scratch["counts"]
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
    extra_rows = int(metadata.extra_pool.shape[1])
    _launch(
        ("stage_dequant", fp8, device.index),
        lambda: ns.StageDequantKernel(fp8),
        _ptr(swa_pool, align=1),
        _ptr(main_pool, align=1),
        _ptr(swa_slots, align=8),
        _ptr(slots, align=8),
        _ptr(tags),
        _ptr(counts),
        _ptr(metadata.swa_pool, align=8),
        _ptr(metadata.extra_pool, align=8),
        _ptr(quant),
        _ptr(metadata.prepared_lens),
        _ptr(metadata.prepared_indices),
        _ptr(metadata.prepared_counter),
        _ptr(q_dequant),
        _ptr(dequant),
        _ptr(bmm1),
        _ptr(bmm2),
        int(count),
        128 + extra_rows,
        swa_width,
        main_width,
        int(swa_slots.stride(0)),
        int(swa_slots.stride(1)) if swa_width > 0 else 1,
        int(slot_rs),
        int(slot_cs),
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
):
    """Decode selected logical rows once into a shared native KV bank (three launches).

    ``plan`` supplies per-request logical domains/page tables and per-query
    request/position vectors. Main selections are physical ``main_slots`` plus
    their ``main_logical`` positions, or logical positions resolved through
    ``main_mapping``. Outputs preserve selection order and duplicates.
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
        main_slots = swa_slots[:, :0]
        main_logical = None
        main_mapping = None
    if main_mapping is not None and main_slots is None:
        main_slots = torch.empty(main_logical.shape, dtype=torch.int64, device=device)
    premapped, logical, slots, (slot_rs, slot_cs), table_args, requests, visible = _mapping_args(
        main_mapping, main_slots, main_logical, count, device
    )
    if swa_slots.dtype != torch.int64:
        swa_slots = swa_slots.long()
    main_width = int(logical.shape[1])
    swa_width = int(swa_slots.shape[1])
    logical_int64 = logical.dtype == torch.int64
    table_ptr, table_rows, table_cols, table_stride, page_size, max_positions = table_args
    selected.zero_()
    bank[0].zero_()
    ns = _cache_kernels()
    placeholder = metadata.prepared_counter
    q_dequant = q_scale_quant_orig if fp8 else placeholder
    _launch(
        ("shared_compact", fp8, has_main, premapped, device.index),
        lambda: ns.SharedCompactKernel(fp8, has_main, premapped),
        _ptr(swa_slots, align=8),
        _ptr(logical, align=8 if logical_int64 else 4),
        _ptr(slots, align=8),
        _ptr(plan.query_requests, align=8),
        _ptr(plan.query_positions),
        _ptr(plan.swa_starts, align=8),
        _ptr(plan.swa_offsets, align=8),
        _ptr(plan.main_offsets, align=8),
        _ptr(visible, align=8),
        table_ptr,
        _ptr(selected),
        _ptr(metadata.prepared_indices),
        _ptr(metadata.prepared_lens),
        _ptr(metadata.prepared_counter),
        _ptr(q_dequant),
        _ptr(kv_scale_quant_orig if fp8 else placeholder),
        _ptr(metadata.mla_bmm1_scale if fp8 else placeholder),
        _ptr(metadata.mla_bmm2_scale if fp8 else placeholder),
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
        int(plan.num_requests),
        int(capacity),
        float(softmax_scale),
        _stream(device),
    )
    quant = kv_scale_orig_quant if fp8 else placeholder
    for pool, pages, offsets, row_base, rows, page_size_, cache_format in (
        (swa_pool, plan.swa_pages, plan.swa_offsets, 1, plan.swa_rows, plan.swa_page_size, "swa"),
        (
            main_pool,
            plan.main_pages,
            plan.main_offsets,
            1 + plan.swa_rows,
            plan.main_rows,
            plan.main_page_size,
            "main",
        ),
    ):
        if not rows:
            continue
        _check(
            pages.dtype == torch.int32
            and pages.ndim == 2
            and (pages.shape[1] <= 1 or pages.stride(1) == 1),
            "pages",
        )
        _launch(
            ("shared_decode", cache_format, fp8, device.index),
            lambda cache_format=cache_format: ns.SharedDecodeKernel(cache_format, fp8),
            _ptr(pool, align=1),
            _ptr(pages),
            _ptr(plan.swa_starts, align=8),
            _ptr(offsets, align=8),
            _ptr(selected),
            _ptr(bank, align=8),
            _ptr(quant),
            int(rows),
            int(row_base),
            int(pool.shape[0]),
            int(pool.stride(0)),
            int(pages.shape[0]),
            int(pages.shape[1]),
            int(pages.stride(0)) if pages.shape[0] > 1 else int(pages.shape[1]),
            int(page_size_),
            int(plan.num_requests),
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
