# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rank-local prefixes for block-cyclic context-parallel MLA caches."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from .mask import mask_visible_k_length
from .query import query_batch_bounds


@cute.jit
def local_kv_length(cfg, global_length, cp_rank):
    """Count this rank's tokens in a global logical KV prefix."""
    if cutlass.const_expr(cfg.cp_world == 1):
        return global_length
    global_length = cute.math.max(global_length, Int32(0))
    interleave = Int32(cfg.cp_interleave_size)
    period = Int32(cfg.cp_world * cfg.cp_interleave_size)
    full_blocks = global_length // period
    remainder = global_length % period - Int32(cp_rank) * interleave
    return full_blocks * interleave + cute.math.min(
        cute.math.max(remainder, Int32(0)), interleave
    )


@cute.kernel
def initialize_empty_query_rows(
    output,
    lse,
    cache_seqs,
    cu_seqlens_q,
    cp_rank: Int32,
    cfg: cutlass.Constexpr,
    seq_len_q: Int32,
):
    """Publish identity rows that an empty rank's main schedule can skip."""
    flat_row = Int32(cute.arch.block_idx()[0])
    thread = Int32(cute.arch.thread_idx()[0])
    heads = Int32(output.shape[0])
    rows_per_request = seq_len_q * heads
    batch_idx = flat_row // rows_per_request
    row = flat_row - batch_idx * rows_per_request
    logical_q_idx = row // heads
    q_start, q_length = query_batch_bounds(cu_seqlens_q, batch_idx, seq_len_q)
    if cutlass.const_expr(cu_seqlens_q is None):
        q_start = Int32(batch_idx) * seq_len_q
    if logical_q_idx < q_length:
        global_bound = mask_visible_k_length(
            cfg.mask_type, Int32(cache_seqs[batch_idx]), logical_q_idx, q_length
        )
        if local_kv_length(cfg, global_bound, cp_rank) == Int32(0):
            output_row = Int64(q_start) * Int64(heads) + Int64(row)
            dim = cutlass.const_expr(output.shape[1])
            for column in cutlass.range(thread, dim, 128):
                (
                    output.iterator.raw_ptr() + output_row * Int64(dim) + Int64(column)
                ).store(output.element_type(0))
            if thread == Int32(0):
                if cutlass.const_expr(lse is not None):
                    (lse.iterator.raw_ptr() + output_row).store(Float32(-Float32.inf))


@cute.jit
def initialize_empty_rows(
    output, lse, cache_seqs, cu_seqlens_q, cp_rank, cfg, seq_len_q, stream
):
    """Launch the opt-in neutral writer without materializing query bounds."""
    initialize_empty_query_rows(
        output, lse, cache_seqs, cu_seqlens_q, cp_rank, cfg, seq_len_q
    ).launch(
        grid=(cute.size(cache_seqs) * seq_len_q * output.shape[0], 1, 1),
        block=(128, 1, 1),
        stream=stream,
    )
