# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row copies between a layer's KV cache blocks and its mixed-precision high layer's blocks.

The row copy is shared with the sparse backends. The copy back belongs to dense attention under
the cold-page first/last-N mixed precision: attention reads and writes the held blocks through
block offsets redirected to the high layer's blocks, and after each layer the step's new rows
are copied to the layer's own blocks so they are complete when the window releases a block.
"""

from typing import TYPE_CHECKING

import triton
import triton.language as tl

if TYPE_CHECKING:
    from .trtllm import TrtllmAttentionMetadata


@triton.jit
def copy_rows(
    kv_cache_ptr,
    slot,
    scratch_ptr,
    scratch_row,
    valid,
    scale_ptr,
    APPLY_SCALE: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Copy mixed-precision high pool rows `slot` to scratch rows `scratch_row` (FP8)."""
    src = kv_cache_ptr + slot[:, None] * ROW_WIDTH
    dst = scratch_ptr + scratch_row.to(tl.int64)[:, None] * ROW_WIDTH
    if APPLY_SCALE:
        scale = tl.load(scale_ptr)
    for start in range(0, ROW_WIDTH, BLOCK_W):
        cols = start + tl.arange(0, BLOCK_W)
        mask = valid[:, None] & (cols < ROW_WIDTH)[None, :]
        row = tl.load(src + cols[None, :], mask=mask)
        if APPLY_SCALE:
            row = row.to(tl.float32) * scale
        tl.store(dst + cols[None, :], row.to(scratch_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _store_mixed_precision_low_rows_kernel(
    rows_ptr,
    high_table_ptr,
    high_table_stride,
    src_ptr,
    dst_ptr,
    offsets_request_stride,
    offsets_kv_stride,
    kv_lens_ptr,
    new_lens_ptr,
    tokens_per_block,
    max_blocks,
    num_heads,
    head_stride,
    token_stride,
    ROW_WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Per (request, BLOCK_T new tokens, K or V): copy a held block's (head, token) rows from `src`
    to `dst`; the strides are the pool layout's row strides inside a block."""
    request = tl.program_id(0).to(tl.int64)
    kv = tl.program_id(2)
    j = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    new = tl.load(new_lens_ptr + request)
    position = tl.load(kv_lens_ptr + request) - new + j
    block = position // tokens_per_block
    valid = (j < new) & (block < max_blocks)
    page = tl.load(high_table_ptr + request * high_table_stride + block, mask=valid, other=-1)
    valid = valid & (page >= 0)
    entry = request * offsets_request_stride + kv * offsets_kv_stride + block
    rows_per_block = num_heads * tokens_per_block
    src_row = tl.load(src_ptr + entry, mask=valid, other=0).to(tl.int64) * rows_per_block
    dst_row = tl.load(dst_ptr + entry, mask=valid, other=0).to(tl.int64) * rows_per_block
    token = (position % tokens_per_block) * token_stride
    for head in range(num_heads):
        offset = head * head_stride + token
        copy_rows(
            rows_ptr,
            src_row + offset,
            rows_ptr,
            dst_row + offset,
            valid,
            rows_ptr,
            False,
            ROW_WIDTH,
            BLOCK_W,
        )


def store_mixed_precision_low_rows(metadata: "TrtllmAttentionMetadata", layer_idx: int) -> None:
    """Copy this step's rows of the held blocks from the high layer's blocks to the layer's own.

    Called only for a layer that has a high layer (TrtllmAttentionMetadata.mixed_precision_kv_staging).
    """
    manager = metadata.kv_cache_manager
    local_layer = manager.layer_offsets[layer_idx]
    num_seqs = metadata.num_seqs
    if num_seqs == 0:
        return
    pool_id = int(manager.impl.get_layer_group_id(local_layer))
    # Block offsets as attention used them (held blocks point at the high layer's blocks) and
    # the layer's own block offsets saved before that redirect.
    redirected_offsets = metadata.kv_cache_block_offsets[pool_id]
    own_offsets = metadata.mixed_precision_low_block_offsets[pool_id]
    high_table = metadata.mixed_precision_high_block_table
    # The pool in its own layout as rows of one head's bytes; block offsets count K/V blocks.
    layout = metadata.kv_layout
    pool = manager._get_mixed_precision_buffers(local_layer, layout, "low")
    heads, tokens = (
        (pool.shape[2], pool.shape[3]) if layout == "HND" else (pool.shape[3], pool.shape[2])
    )
    head_stride, token_stride = (tokens, 1) if layout == "HND" else (1, heads)
    pool_rows = pool.view(-1, pool.shape[4])
    # This step's new tokens per request: the query lengths, which speculative decoding corrects
    # on the device before a draft step (seq_lens_kv keeps a separate, uncorrected copy under
    # CUDA graphs).
    max_new_tokens = int(metadata.seq_lens[:num_seqs].max())
    tokens_per_program = 16
    _store_mixed_precision_low_rows_kernel[
        (num_seqs, triton.cdiv(max_new_tokens, tokens_per_program), pool.shape[1])
    ](
        pool_rows,
        high_table,
        high_table.stride(0),
        redirected_offsets,
        own_offsets,
        redirected_offsets.stride(0),
        redirected_offsets.stride(1),
        metadata.kv_lens_cuda_runtime,
        metadata.seq_lens_cuda[:num_seqs],
        tokens,
        high_table.shape[1],
        heads,
        head_stride,
        token_stride,
        ROW_WIDTH=pool_rows.shape[1],
        BLOCK_T=tokens_per_program,
        BLOCK_W=128,
    )
