# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Runtime per-query prefixes and neutral publication for empty MLA rows."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from .query import query_batch_bounds


@cute.jit
def query_kv_bound(kv_bounds, cu_seqlens_q, batch_idx, logical_q_idx, seq_len_q):
    """Load a logical-token bound without indexing packed or fixed padding."""
    q_start, q_length = query_batch_bounds(cu_seqlens_q, batch_idx, seq_len_q)
    if cutlass.const_expr(cu_seqlens_q is None):
        q_start = Int32(batch_idx) * Int32(seq_len_q)
    bound = Int32(0)
    if logical_q_idx >= Int32(0) and logical_q_idx < q_length:
        bound = Int32(kv_bounds[q_start + logical_q_idx])
    return bound


@cute.kernel
def initialize_empty_query_rows(output, lse, softmax_stats, kv_bounds):
    """Publish identity rows even when the main schedule skips an empty cache.

    The public tensors have contiguous token/head storage before their
    dimension-first kernel views. Bound values remain graph-live device data.
    """
    row = Int64(cute.arch.block_idx()[0])
    thread = Int32(cute.arch.thread_idx()[0])
    heads = Int64(output.shape[0])
    dim = cutlass.const_expr(output.shape[1])
    if kv_bounds[row // heads] == Int32(0):
        for column in cutlass.range(thread, dim, 128):
            (output.iterator.raw_ptr() + row * Int64(dim) + Int64(column)).store(
                output.element_type(0)
            )
        if thread == Int32(0):
            if cutlass.const_expr(lse is not None):
                (lse.iterator.raw_ptr() + row).store(Float32(-Float32.inf))
            if cutlass.const_expr(softmax_stats is not None):
                (softmax_stats.iterator.raw_ptr() + row * Int64(2)).store(
                    Float32(-Float32.inf)
                )
                (softmax_stats.iterator.raw_ptr() + row * Int64(2) + Int64(1)).store(
                    Float32(0)
                )


@cute.jit
def initialize_empty_rows(output, lse, softmax_stats, kv_bounds, stream):
    """Launch the bounded-mode neutral writer on the caller's stream."""
    initialize_empty_query_rows(output, lse, softmax_stats, kv_bounds).launch(
        grid=(cute.size(kv_bounds) * output.shape[0], 1, 1),
        block=(128, 1, 1),
        stream=stream,
    )
