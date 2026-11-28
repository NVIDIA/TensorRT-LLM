# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


@triton.jit
def _resmooth_kernel(
    w_ptr,
    s_ptr,
    M,
    K,
    stride_wb,
    stride_wm,
    stride_wk,
    stride_sb,
    stride_sm,
    stride_sk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    curr_w_ptr = w_ptr + batch_idx * stride_wb
    curr_s_ptr = s_ptr + batch_idx * stride_sb

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    s_offset = pid_m * stride_sm + pid_k * stride_sk
    old_scale = tl.load(curr_s_ptr + s_offset)

    w_mask = (rm[:, None] < M) & (rk[None, :] < K)
    w_offsets = rm[:, None] * stride_wm + rk[None, :] * stride_wk
    w_fp8 = tl.load(curr_w_ptr + w_offsets, mask=w_mask, other=0.0)
    w_fp32 = w_fp8.to(tl.float32)

    w_val = w_fp32 * old_scale
    block_amax = tl.maximum(tl.max(tl.abs(w_val)), 1e-4)

    # UE8M0 sf = 2 ^ ceil(log2(sf))
    new_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(block_amax / 448.0)))
    w_requant = w_val * (1.0 / new_scale)

    tl.store(curr_w_ptr + w_offsets, w_requant, mask=w_mask)
    tl.store(curr_s_ptr + s_offset, new_scale)


def resmooth_to_fp8_e8m0(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
):
    assert weight.dtype == torch.float8_e4m3fn
    assert weight_scale.dtype == torch.float32

    orig_shape = weight.shape
    M, K = orig_shape[-2:]
    w_view = weight.view(-1, M, K)
    s_view = weight_scale.view(-1, weight_scale.shape[-2], weight_scale.shape[-1])

    num_batches = w_view.shape[0]
    BLOCK_M, BLOCK_K = block_size

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_k = triton.cdiv(K, BLOCK_K)
    blocks_per_batch = grid_m * grid_k

    # Workaround for Triton compiler/runtime bug on SM100f (Blackwell):
    # CUDA illegal memory access when total grid blocks exceed ~65K.
    # Split launches along the batch dimension to stay under the limit.
    MAX_GRID_BLOCKS = 65536
    if blocks_per_batch > 0:
        max_batches_per_launch = max(1, MAX_GRID_BLOCKS // blocks_per_batch)
    else:
        max_batches_per_launch = num_batches

    for batch_offset in range(0, num_batches, max_batches_per_launch):
        batch_count = min(max_batches_per_launch, num_batches - batch_offset)
        grid = (batch_count, grid_m, grid_k)

        _resmooth_kernel[grid](
            w_view[batch_offset:],
            s_view[batch_offset:],
            M,
            K,
            w_view.stride(0),
            w_view.stride(1),
            w_view.stride(2),
            s_view.stride(0),
            s_view.stride(1),
            s_view.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
        )


def get_tma_aligned_size(x: int, element_size: int) -> int:
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return align(x, alignment)


def get_col_major_tma_aligned_packed_tensor(x: torch.Tensor) -> torch.Tensor:
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dtype == torch.float and x.dim() in (2, 3)

    # First, convert into UE8M0 `uint8_t`
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8)

    # Second, make padded packed tensors
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=torch.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn, aligned_k // 4)

    # Finally, transpose
    transposed = torch.transpose(
        torch.empty((b, aligned_k // 4, aligned_mn), device=x.device, dtype=torch.int), 1, 2
    )
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x
