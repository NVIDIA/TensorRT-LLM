# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

########################################################
# Index gather kernel
########################################################


@triton.jit
def _deepseek_v4_local_to_global_kernel(
    req_id_ptr,
    block_table_swa_ptr,
    block_table_compressed_ptr,
    swa_local_indices_ptr,
    compressed_local_indices_ptr,
    out_ptr,
    swa_buffer_offset_in_tokens,
    compressed_buffer_offset_in_tokens,
    tokens_per_block_swa: tl.constexpr,
    tokens_per_block_compressed: tl.constexpr,
    max_blocks_swa,
    max_blocks_compressed,
    num_swa_indices: tl.constexpr,
    num_compressed_indices: tl.constexpr,
    total_output_indices,
    has_compressed: tl.constexpr,
    bt_swa_stride0,
    bt_swa_stride1,
    bt_compressed_stride0,
    bt_compressed_stride1,
    swa_indices_stride0,
    swa_indices_stride1,
    compressed_indices_stride0,
    compressed_indices_stride1,
    out_stride0,
    out_stride1,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    """
    Triton kernel for converting local indices to global KV cache pool indices.

    Dual-pool output layout:
    - SWA region [0, num_swa_indices): indices relative to swa_pool_base_ptr
    - Compress region [num_swa_indices, num_swa_indices + num_compressed_indices):
      indices relative to compress_pool_base_ptr
    - Invalid positions padded with -1 at their fixed positions.

    This enables the FMHA kernel to determine which TMA descriptor to use based
    solely on tile index (tile 0 = SWA via tmaKSecondary_, rest = compress via tmaK_).
    """
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    token_id = tl.program_id(0)

    # Load request ID for this token
    req = tl.load(req_id_ptr + token_id)

    # Load all SWA local indices for this token
    swa_ids = tl.arange(0, num_swa_indices)
    swa_ptr = swa_local_indices_ptr + token_id * swa_indices_stride0 + swa_ids * swa_indices_stride1
    swa_local_idx = tl.load(swa_ptr)

    # Compute global indices for all SWA positions
    swa_valid_mask = swa_local_idx >= 0
    swa_block_ordinal = swa_local_idx // tokens_per_block_swa
    swa_token_in_block = swa_local_idx % tokens_per_block_swa
    swa_valid_block = swa_block_ordinal < max_blocks_swa
    swa_full_mask = swa_valid_mask & swa_valid_block

    swa_bt_ptr = block_table_swa_ptr + req * bt_swa_stride0 + swa_block_ordinal * bt_swa_stride1
    swa_page_index = tl.load(swa_bt_ptr, mask=swa_full_mask, other=0)

    swa_global_index = swa_buffer_offset_in_tokens + swa_page_index * tokens_per_block_swa + swa_token_in_block
    swa_global_index = tl.where(swa_full_mask, swa_global_index, -1)

    # Store SWA results at fixed positions [0, num_swa_indices)
    swa_out_ptr = out_ptr + token_id * out_stride0 + swa_ids * out_stride1
    tl.store(swa_out_ptr, swa_global_index)

    if has_compressed:
        # Load all compressed local indices for this token
        compressed_ids = tl.arange(0, num_compressed_indices)
        compressed_ptr = (compressed_local_indices_ptr +
                          token_id * compressed_indices_stride0 +
                          compressed_ids * compressed_indices_stride1)
        compressed_local_idx = tl.load(compressed_ptr)

        # Compute global indices for all compressed positions
        compressed_valid_mask = compressed_local_idx >= 0
        compressed_block_ordinal = compressed_local_idx // tokens_per_block_compressed
        compressed_token_in_block = compressed_local_idx % tokens_per_block_compressed
        compressed_valid_block = compressed_block_ordinal < max_blocks_compressed
        compressed_full_mask = compressed_valid_mask & compressed_valid_block

        compressed_bt_ptr = (block_table_compressed_ptr +
                             req * bt_compressed_stride0 +
                             compressed_block_ordinal * bt_compressed_stride1)
        compressed_page_index = tl.load(compressed_bt_ptr,
                                        mask=compressed_full_mask,
                                        other=0)

        compressed_global_index = (
            compressed_buffer_offset_in_tokens +
            compressed_page_index * tokens_per_block_compressed +
            compressed_token_in_block)
        compressed_global_index = tl.where(compressed_full_mask,
                                           compressed_global_index, -1)

        # Store compressed results at fixed positions [num_swa_indices, total)
        compressed_write_pos = num_swa_indices + compressed_ids
        compressed_out_ptr = out_ptr + token_id * out_stride0 + compressed_write_pos * out_stride1
        tl.store(compressed_out_ptr, compressed_global_index)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def deepseek_v4_local_to_global_indices(
        req_id: torch.Tensor,  # int32 [num_tokens]
        block_table_swa: torch.Tensor,  # int32 [num_requests, max_blocks_swa]
        swa_local_indices: torch.Tensor,  # int32 [num_tokens, num_swa_indices]
        swa_pool_base_ptr: int,  # int64: base address of SWA pool
        swa_buffer_ptr: int,  # int64: base address of SWA buffer
        tokens_per_block: int,  # tokens per block for SWA
        token_stride: int,  # bytes per token (use SWA token stride)
        # Optional compressed arguments (for compress_ratio > 1)
    block_table_compressed: torch.Tensor | None = None,
        compressed_local_indices: torch.Tensor | None = None,
        compress_pool_base_ptr: int = 0,  # int64: base address of compress pool
        compressed_buffer_ptr: int = 0,
        compress_ratio: int = 1,
        num_compressed_indices: int = 0,  # max number of compressed indices
) -> torch.Tensor:
    """
    Convert local token indices to global KV cache pool indices.

    Dual-pool output layout:
    - SWA region [0, window_size): indices relative to swa_pool_base_ptr
    - Compress region [window_size, window_size + num_compressed_indices):
      indices relative to compress_pool_base_ptr
    - Invalid positions padded with -1 at their fixed positions.

    For compress_ratio=1: Only SWA region, all indices relative to swa_pool_base_ptr.
    For compress_ratio>1: SWA region + compress region with separate base pointers.

    Args:
        req_id: Request ID per token [num_tokens], int32
        block_table_swa: SWA block table [num_requests, max_blocks_swa], int32
        swa_local_indices: Local indices for SWA cache [num_tokens, num_swa_indices], int32
            Use -1 for invalid/padding indices.
        swa_pool_base_ptr: Base address of SWA pool
        swa_buffer_ptr: Base address of SWA buffer (base_pool_ptr + buffer_offset_in_slot)
        tokens_per_block: Number of tokens per block for SWA cache
        token_stride: Bytes per token (use SWA token stride)
        block_table_compressed: Compressed block table [num_requests, max_blocks_compressed], int32 (optional)
        compressed_local_indices: Local indices for compressed cache [num_tokens, num_compressed_indices], int32 (optional)
            Use -1 for invalid/padding indices.
        compress_pool_base_ptr: Base address of compress pool
        compressed_buffer_ptr: Base address of compressed buffer (optional)
        compress_ratio: Compression ratio (1: no compression, >1: with compression)
        num_compressed_indices: Max number of compressed indices for CUDA graph compatibility
            Output width = num_swa_indices + num_compressed_indices.

    Returns:
        global_indices: int32 [num_tokens, num_swa_indices + num_compressed_indices]
    """
    assert req_id.dtype == torch.int32, f"req_id must be int32, got {req_id.dtype}"
    assert block_table_swa.dtype == torch.int32, f"block_table_swa must be int32, got {block_table_swa.dtype}"
    assert swa_local_indices.dtype == torch.int32, f"swa_local_indices must be int32, got {swa_local_indices.dtype}"

    num_tokens = req_id.shape[0]
    num_swa_indices = swa_local_indices.shape[1]

    assert swa_local_indices.shape[0] == num_tokens

    has_compressed = compress_ratio > 1

    # Compute SWA buffer offset relative to swa_pool_base_ptr in tokens
    swa_buffer_offset_in_tokens = (swa_buffer_ptr -
                                   swa_pool_base_ptr) // token_stride

    if has_compressed:
        assert block_table_compressed is not None, "block_table_compressed required when compress_ratio > 1"
        assert compressed_local_indices is not None, "compressed_local_indices required when compress_ratio > 1"
        assert (
            block_table_compressed.dtype == torch.int32
        ), f"block_table_compressed must be int32, got {block_table_compressed.dtype}"
        assert (
            compressed_local_indices.dtype == torch.int32
        ), f"compressed_local_indices must be int32, got {compressed_local_indices.dtype}"
        assert compressed_local_indices.shape[0] == num_tokens

        tokens_per_block_compressed = tokens_per_block // compress_ratio
        # Compute compressed buffer offset relative to compress_pool_base_ptr in tokens
        assert (
            compressed_buffer_ptr - compress_pool_base_ptr
        ) % token_stride == 0, "compressed_buffer_ptr must be aligned to token_stride"
        compressed_buffer_offset_in_tokens = (
            compressed_buffer_ptr - compress_pool_base_ptr) // token_stride
        _, max_blocks_compressed = block_table_compressed.shape
        block_table_compressed_c = block_table_compressed.contiguous()
        compressed_local_indices_c = compressed_local_indices.contiguous()
    else:
        # Dummy values
        tokens_per_block_compressed = tokens_per_block
        compressed_buffer_offset_in_tokens = 0
        max_blocks_compressed = 1
        block_table_compressed_c = torch.zeros((1, 1),
                                               dtype=torch.int32,
                                               device=req_id.device)
        compressed_local_indices_c = torch.zeros((1, 1),
                                                 dtype=torch.int32,
                                                 device=req_id.device)

    total_output_indices = num_swa_indices + num_compressed_indices
    _, max_blocks_swa = block_table_swa.shape

    # Ensure contiguous tensors
    req_id_c = req_id.contiguous()
    block_table_swa_c = block_table_swa.contiguous()
    swa_local_indices_c = swa_local_indices.contiguous()

    # Create output tensor
    out = torch.empty((num_tokens, total_output_indices),
                      dtype=torch.int32,
                      device=req_id.device)

    # Grid: one program per token
    grid = (num_tokens, )

    # Get strides
    bt_swa_stride0, bt_swa_stride1 = block_table_swa_c.stride()
    bt_compressed_stride0, bt_compressed_stride1 = block_table_compressed_c.stride(
    )
    swa_indices_stride0, swa_indices_stride1 = swa_local_indices_c.stride()
    compressed_indices_stride0, compressed_indices_stride1 = compressed_local_indices_c.stride(
    )
    out_stride0, out_stride1 = out.stride()
    launch_with_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"

    # Launch kernel
    _deepseek_v4_local_to_global_kernel[grid](
        req_id_c,
        block_table_swa_c,
        block_table_compressed_c,
        swa_local_indices_c,
        compressed_local_indices_c,
        out,
        swa_buffer_offset_in_tokens,
        compressed_buffer_offset_in_tokens,
        tokens_per_block,
        tokens_per_block_compressed,
        max_blocks_swa,
        max_blocks_compressed,
        num_swa_indices,
        num_compressed_indices,
        total_output_indices,
        has_compressed,
        bt_swa_stride0,
        bt_swa_stride1,
        bt_compressed_stride0,
        bt_compressed_stride1,
        swa_indices_stride0,
        swa_indices_stride1,
        compressed_indices_stride0,
        compressed_indices_stride1,
        out_stride0,
        out_stride1,
        LAUNCH_WITH_PDL=launch_with_pdl,
        launch_pdl=launch_with_pdl,
    )

    return out
