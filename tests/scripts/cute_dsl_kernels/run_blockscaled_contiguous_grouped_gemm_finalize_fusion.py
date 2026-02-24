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

# TODO(zhichenj): This file is copied and modified from cutlass example

"""Example usage of the kernel.

python run_blockscaled_contiguous_grouped_gemm_finalize_fusion.py \
        --ab_dtype Float4E2M1FN --out_dtype Float32 \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 256,256 --cluster_shape_mn 1,1 --seq_len 4096 \
        --benchmark 128x7168x2048x8 --iterations 1
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
        blockscaled_contiguous_grouped_gemm_finalize_fusion as kernel_module,
    )
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).parents[3] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell import blockscaled_contiguous_grouped_gemm_finalize_fusion as kernel_module

Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel = (
    kernel_module.Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel
)
cvt_sf_MKL_to_M32x4xrm_K4xrk_L = kernel_module.cvt_sf_MKL_to_M32x4xrm_K4xrk_L

try:
    from .testing import benchmark
except ImportError:
    from testing import benchmark


def create_mask(group_m_list, cta_tile_mn, permuted_m=None):
    """Create mask and group mapping for contiguous grouped GEMM.

    :param group_m_list: List of M values for each group (will be aligned to cta_tile_mn[0] dimension)
    :param cta_tile_mn: CTA tile size tuple (M, N) - M dimension used for alignment
    :param permuted_m: Optional padded M dimension for cuda_graph support. If provided,
                     tile_idx_to_expert_idx will be padded to this size.
                     When tile_idx >= num_non_exiting_tiles, the kernel exits.

    Note: For cuda_graph support, set permuted_m to the pre-calculated padded size:
          permuted_m = m * topK + num_local_experts * (256 - 1)
          Example: 4096*8 + (256/32)*255 = 34808
          Only the actual valid rows (aligned_groupm[0]+aligned_groupm[1]+...) contain
          valid data. The kernel will exit when tile_idx >= num_non_exiting_tiles.

    :return: Tuple of (valid_m, aligned_group_m_list, tile_idx_to_expert_idx,
                num_non_exiting_tiles, tile_idx_to_mn_limit)
         - valid_m: total valid M dimension
         - aligned_group_m_list: aligned group M dimension list
         - tile_idx_to_expert_idx: shape (permuted_m/cta_tile_m,) if permuted_m provided, else (valid_m/cta_tile_m,)
             - num_non_exiting_tiles: scalar value = valid_m/cta_tile_m
             - tile_idx_to_mn_limit: M limit for each tile index
    """
    m_aligned = cta_tile_mn[0]
    valid_m = 0
    aligned_group_m_list = []
    tile_idx_to_expert_idx = []
    tile_idx_to_mn_limit = []

    for i, group_m in enumerate(group_m_list):
        aligned_group_m = ((group_m + m_aligned - 1) // m_aligned) * m_aligned
        aligned_group_m_list.append(aligned_group_m)

        # Calculate number of tiles for this group based on CTA tile M size
        # Each tile covers cta_tile_m rows in M dimension
        num_tiles_in_group = aligned_group_m // cta_tile_mn[0]
        # Add expert_idx for each tile in this group
        tile_idx_to_expert_idx.extend([i] * num_tiles_in_group)

        tile_idx_to_mn_limit.extend([group_m + valid_m] * num_tiles_in_group)
        valid_m += aligned_group_m

    # Compute num_non_exiting_tiles (number of valid tiles in M dimension)
    num_non_exiting_tiles = len(tile_idx_to_expert_idx)

    # Apply padding if requested (for cuda_graph support)
    if permuted_m is not None:
        if permuted_m < valid_m:
            raise ValueError(
                f"permuted_m ({permuted_m}) must be >= valid_m ({valid_m}). "
                f"Cannot pad to a smaller size."
            )
        if permuted_m > valid_m:
            # Calculate how many padding tiles are needed based on CTA tile M size
            num_padding_tiles = (permuted_m - valid_m) // cta_tile_mn[0]
            # Pad with 0 (these tiles won't be accessed due to num_non_exiting_tiles check)
            tile_idx_to_expert_idx.extend([0] * num_padding_tiles)

    # Final shape of tile_idx_to_expert_idx: (permuted_m/cta_tile_m,) if permuted_m provided, else (valid_m/cta_tile_m,)
    tile_idx_to_expert_idx = torch.tensor(tile_idx_to_expert_idx, device="cuda", dtype=torch.int32)
    num_non_exiting_tiles_tensor = torch.tensor(
        [num_non_exiting_tiles], device="cuda", dtype=torch.int32
    )

    tile_idx_to_mn_limit_tensor = torch.tensor(
        tile_idx_to_mn_limit, device="cuda", dtype=torch.int32
    )

    return (
        valid_m,
        aligned_group_m_list,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles_tensor,
        tile_idx_to_mn_limit_tensor,
    )


def create_fused_finalize_tensors(
    seq_len, topK, permuted_m, group_m_list, mma_tiler_mn, final_scale_dtype
):
    m_aligned = mma_tiler_mn[0]
    permuted_idx_to_expanded_idx_tensor = torch.empty(
        (permuted_m,), dtype=torch.int32, device="cuda"
    ).fill_(-1)
    # # normalized token final scales
    token_final_scales = (
        torch.rand(seq_len, topK).to(dtype=cutlass_torch.dtype(final_scale_dtype)).cuda()
    )
    token_final_scales = token_final_scales / token_final_scales.sum(dim=1, keepdim=True)

    start_idx = 0
    for group_idx in range(len(group_m_list)):
        m_per_group = group_m_list[group_idx]

        if m_per_group > 0:
            # Sequential/Blocked assignment for better atomic add memory access (Friendly Layout)
            # Experts are grouped into sets of size topK.
            # Expert Set S (experts S*topK ... S*topK+topK-1) serves a contiguous block of tokens.
            # This ensures that within an expert, we process tokens T, T+1, T+2... sequentially.

            expert_set_idx = group_idx // topK
            k_in_set = group_idx % topK

            # Start token index for this expert set
            start_token = expert_set_idx * m_per_group

            # Generate sequential token indices for this expert
            token_indices = torch.arange(
                start_token, start_token + m_per_group, dtype=torch.int32, device="cuda"
            )
            token_indices = token_indices % seq_len

            # expanded_idx = token_idx * topK + k
            expanded_idx = token_indices * topK + k_in_set

            permuted_idx_to_expanded_idx_tensor[start_idx : (start_idx + m_per_group)] = (
                expanded_idx
            )
        # Move to next aligned group
        m_aligned_per_group = ((m_per_group + m_aligned - 1) // m_aligned) * m_aligned
        start_idx += m_aligned_per_group

    return (
        permuted_idx_to_expanded_idx_tensor,
        token_final_scales,
        from_dlpack(permuted_idx_to_expanded_idx_tensor).mark_layout_dynamic(),
        from_dlpack(token_final_scales).mark_layout_dynamic(),
    )


def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype):  # noqa: E741
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(k, sf_vec_size)
    ref_shape = (l, mn, sf_k)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    ref_permute_order = (1, 2, 0)
    mma_permute_order = (3, 4, 1, 5, 2, 0)

    # Create f32 ref torch tensor (cpu)
    ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        ref_shape,
        torch.float32,
        permute_order=ref_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=1,
            max_val=3,
        ),
    )

    # Create f32 cute torch tensor (cpu)
    cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=0,
            max_val=1,
        ),
    )

    # convert ref f32 tensor to cute f32 tensor
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )

    cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

    # reshape makes memory contiguous
    ref_f32_torch_tensor_cpu = (
        ref_f32_torch_tensor_cpu.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, mn, sf_k, sf_vec_size)
        .reshape(l, mn, sf_k * sf_vec_size)
        .permute(*ref_permute_order)
    )
    # prune to mkl for reference check.
    ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

    # Create dtype cute torch tensor (cpu)
    cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
        cute_f32_torch_tensor_cpu,
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )

    # Convert f32 cute tensor to dtype cute tensor
    cute_tensor = cutlass_torch.convert_cute_tensor(
        cute_f32_torch_tensor,
        cute_tensor,
        dtype,
        is_dynamic_layout=True,
    )

    return ref_f32_torch_tensor_cpu, cute_tensor, cute_torch_tensor


def create_tensors(
    l,  # noqa: E741
    group_m_list,
    n,
    k,
    a_major,
    b_major,
    cd_major,
    ab_dtype,
    out_dtype,
    sf_dtype,
    sf_vec_size,
    mma_tiler_mn,
    permuted_m=None,
    seq_len=None,
):
    """Create tensors for contiguous grouped GEMM.

    :param permuted_m: Optional padded M dimension for cuda_graph support. If provided,
                     A matrix, C matrix, and scale factor A will be padded to this size.
                     The kernel exits when tile_idx >= num_non_exiting_tiles.
    :param seq_len: Sequence length (number of output tokens for C tensor)
    :return: Tuple of (a_tensor, b_tensor, out_tensor, sfa_tensor, sfb_tensor,
                      tile_idx_to_expert_idx, num_non_exiting_tiles, alpha,
                      a_torch_cpu, b_torch_cpu, out_torch_cpu, sfa_torch_cpu, sfb_torch_cpu,
                      alpha_torch_cpu, a_torch_gpu, b_torch_gpu, sfa_torch_gpu, sfb_torch_gpu,
                      out_torch_gpu, aligned_group_m_list, valid_m

    Example with CUDA graph padding:
        # For MoE: m=4096, topK=8, num_local_experts=256, experts_per_rank=8
        permuted_m = 4096 * 8 + 8 * 255  # = 34808
        tensors = create_tensors(
            l=8,  # num_local_experts
            group_m_list=[512, 1024, ...],  # actual group sizes
            n=4096, k=7168,
            a_major="k", b_major="k", cd_major="n",
            ab_dtype=cutlass.Float4E2M1FN,
            out_dtype=cutlass.BFloat16,
            sf_dtype=cutlass.Float8E4M3FN,
            sf_vec_size=16,
            mma_tiler_mn=(128, 128),  # MMA tile shape
            permuted_m=34808,  # Enable padding for cuda_graph
        )
        # Returns tensors with A, C, and SFA padded to permuted_m rows,
        # kernel exits early when tile_idx >= num_non_exiting_tiles
        # Also returns permuted_idx_to_expanded_idx and token_final_scales for fusion
    """
    torch.manual_seed(1111)

    alpha_torch_cpu = torch.ones((l,), dtype=torch.float32) * 0.1

    (
        valid_m,
        aligned_group_m_list,
        _tile_idx_to_expert_idx,
        _num_non_exiting_tiles,
        _tile_idx_to_mn_limit,
    ) = create_mask(group_m_list, mma_tiler_mn, permuted_m)

    # Use permuted_m for A/C tensors if provided (for cuda_graph support)
    tensor_m = permuted_m if permuted_m is not None else valid_m

    a_torch_cpu = cutlass_torch.matrix(1, tensor_m, k, a_major == "m", cutlass.Float32)
    b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major == "n", cutlass.Float32)
    out_torch_cpu = cutlass_torch.matrix(1, seq_len, n, cd_major == "m", cutlass.Float32)

    out_torch_cpu.fill_(0)

    a_tensor, a_torch_gpu = cutlass_torch.cute_tensor_like(
        a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch_gpu = cutlass_torch.cute_tensor_like(
        b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    out_tensor, out_torch_gpu = cutlass_torch.cute_tensor_like(
        out_torch_cpu, out_dtype, is_dynamic_layout=True, assumed_align=16
    )

    # Mark tensor with element divisibility for 16B alignment
    a_tensor.mark_compact_shape_dynamic(
        mode=1 if a_major == "k" else 0,
        stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
        divisibility=32 if ab_dtype == cutlass.Float4E2M1FN else 16,
    )
    b_tensor.mark_compact_shape_dynamic(
        mode=1 if b_major == "k" else 0,
        stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
        divisibility=32 if ab_dtype == cutlass.Float4E2M1FN else 16,
    )

    out_tensor.mark_compact_shape_dynamic(
        mode=1 if cd_major == "n" else 0,
        stride_order=(2, 0, 1) if cd_major == "n" else (2, 1, 0),
        divisibility=32 if ab_dtype == cutlass.Float4E2M1FN else 16,
    )

    # Use tensor_m (permuted_m if provided) for scale factor A
    sfa_torch_cpu, sfa_tensor, sfa_torch_gpu = create_scale_factor_tensor(
        1, tensor_m, k, sf_vec_size, sf_dtype
    )
    sfb_torch_cpu, sfb_tensor, sfb_torch_gpu = create_scale_factor_tensor(
        l, n, k, sf_vec_size, sf_dtype
    )

    tile_idx_to_expert_idx = from_dlpack(_tile_idx_to_expert_idx).mark_layout_dynamic()
    num_non_exiting_tiles = from_dlpack(_num_non_exiting_tiles).mark_layout_dynamic()
    tile_idx_to_mn_limit = from_dlpack(_tile_idx_to_mn_limit).mark_layout_dynamic()

    alpha = from_dlpack(alpha_torch_cpu.cuda()).mark_layout_dynamic()

    out_torch_gpu.fill_(0)

    return (
        a_tensor,
        b_tensor,
        out_tensor,
        sfa_tensor,
        sfb_tensor,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles,
        tile_idx_to_mn_limit,
        alpha,
        a_torch_cpu,
        b_torch_cpu,
        out_torch_cpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
        alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        sfa_torch_gpu,
        sfb_torch_gpu,
        out_torch_gpu,
        aligned_group_m_list,
        valid_m,
    )


def verify_reference_result(
    a_torch_cpu: torch.Tensor,
    b_torch_cpu: torch.Tensor,
    sfa_torch_cpu: torch.Tensor,
    sfb_torch_cpu: torch.Tensor,
    alpha_torch_cpu: torch.Tensor,
    permuted_idx_to_expanded_idx_torch: torch.Tensor,
    token_final_scales_torch: torch.Tensor,
    group_m_list: List[int],
    aligned_group_m_list: List[int],
    out_dtype: torch.dtype,
    valid_m: int,
    n: int,
    topK: int,
    seq_len: int,
) -> torch.Tensor:
    gemm_output = torch.empty((1, valid_m, n), dtype=torch.float32)
    valid_mask = torch.zeros((valid_m,), dtype=torch.bool, device="cuda")
    #########  gemm calculation #########
    start = 0
    for i, group_m in enumerate(aligned_group_m_list):
        end = start + group_m
        res_a = torch.einsum(
            "mk,mk->mk",
            a_torch_cpu[start:end, :, 0],
            sfa_torch_cpu[start:end, :, 0],
        )
        res_b = torch.einsum("nk,nk->nk", b_torch_cpu[:, :, i], sfb_torch_cpu[:, :, i])
        gemm_output[0, start:end, :] = torch.einsum("mk,nk->mn", res_a, res_b) * alpha_torch_cpu[i]
        valid_mask[start : start + group_m_list[i]] = 1
        start = end

    #########  finalize calculation #########
    # Considering BF16 accumulator gpu accuracy error, we used gpu to verify the result
    gemm_output = gemm_output.permute((1, 2, 0)).cuda()

    final_output = torch.zeros((seq_len, n), dtype=out_dtype).cuda()

    gemm_output = gemm_output[:valid_m, :, 0].clone()  # (valid_m, n)
    expanded_idx_all = permuted_idx_to_expanded_idx_torch[:valid_m]  # (valid_m,

    # Step 3: Filter valid rows (expanded_idx >= 0)
    expanded_idx_valid = expanded_idx_all[valid_mask]  # (num_valid,)
    gemm_output_valid = gemm_output[valid_mask]  # (num_valid, n)

    token_idx = expanded_idx_valid // topK  # (num_valid,)
    topk_idx = expanded_idx_valid % topK  # (num_valid,)

    scales = token_final_scales_torch[token_idx, topk_idx]  # (num_valid,)
    scaled_output = gemm_output_valid * scales.unsqueeze(1)  # (num_valid, n)
    scaled_output = scaled_output.to(out_dtype)

    for i in range(len(token_idx)):
        final_output[token_idx[i]] += scaled_output[i]

    return final_output


def run(
    nkl: Tuple[int, int, int],
    group_m_list: Tuple[int, ...],
    ab_dtype: Type[cutlass.Numeric],
    out_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    final_scale_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    out_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    permuted_m: int = None,
    topK: int = 8,
    seq_len: int = 4096,
    raster_along_m: bool = False,
    use_cupti: bool = False,
    **kwargs,
):
    """Prepare A/B/C tensors, launch GPU kernel, and reference checking.

    :param nkl: Tuple of (N, K, L) dimensions of the GEMM problem.
    :param group_m_list: List of M values for each group.
    :param ab_dtype: Data type for input tensors A and B.
    :param out_dtype: Data type for the output tensor C.
    :param sf_dtype: Data type for the scale factor tensors.
    :param sf_vec_size: Vector size for the scale factor tensors.
    :param final_scale_dtype: Data type for the final scale factor tensors.
    :param a_major: The major axis of the A tensor.
    :param b_major: The major axis of the B tensor.
    :param out_major: The major axis of the C tensor.
    :param mma_tiler_mn: The shape of the MMA tile.
    :param cluster_shape_mn: The shape of the CTA cluster.
    :param tolerance: Tolerance for result validation.
    :param warmup_iterations: Number of warmup runs.
    :param iterations: Number of benchmark iterations.
    :param skip_ref_check: If True, skips reference checking.
    :param use_cold_l2: If True, uses a circular buffer to ensure a cold L2 cache.
    :param permuted_m: Optional padded M dimension for CUDA graph support. If provided,
                    A/C matrices and scale factor A will be padded to this size.
    :param topK: Number of experts per token (for MoE with fused finalize)
    :param seq_len: Sequence length for MoE, used by fused finalize,
            need to know the sequence length to do finalize correctly.
    :param raster_along_m: If True, raster along M dimension for tile scheduler.
    :param use_cupti (bool, optional): If True, uses CUPTI to measure execution time.
            Defaults to False.
    """
    m_aligned = mma_tiler_mn[0]

    print("Running Blackwell Persistent Dense Contiguous Grouped GEMM test with:")
    print(f"nkl: {nkl}")
    print(f"group_m_list: {group_m_list}")
    print(
        f"AB dtype: {ab_dtype}, C dtype: {out_dtype}, Scale factor dtype: {sf_dtype}, SF Vec size: {sf_vec_size}"
    )
    print(f"Group M alignment: {m_aligned}")
    if permuted_m is not None:
        print(f"Padded M (CUDA graph support): {permuted_m}")
    print(f"Fused finalize enabled with topK={topK}")
    print(f"Sequence length: {seq_len}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, Out: {out_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use TMA prefetch: {'True'}")
    print(f"Raster along M: {raster_along_m}")
    print(f"Use CUPTI: {'True' if use_cupti else 'False'}")

    # Unpack parameters
    n, k, l = nkl  # noqa: E741

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # Skip unsupported testcase
    if not Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        out_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        m_aligned,
        n,
        k,
        l,
        a_major,
        b_major,
        out_major,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {out_dtype}, "
            f"{mma_tiler_mn}, {cluster_shape_mn}, {n}, {k}, {l}, "
            f"{a_major}, {b_major}, {out_major}, {m_aligned}"
        )

    (
        a_tensor,
        b_tensor,
        out_tensor,
        sfa_tensor,
        sfb_tensor,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles,
        tile_idx_to_mn_limit,
        alpha,
        a_torch_cpu,
        b_torch_cpu,
        out_torch_cpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
        alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        sfa_torch_gpu,
        sfb_torch_gpu,
        out_torch_gpu,
        aligned_group_m_list,
        valid_m,
    ) = create_tensors(
        l,
        group_m_list,
        n,
        k,
        a_major,
        b_major,
        out_major,
        ab_dtype,
        out_dtype,
        sf_dtype,
        sf_vec_size,
        mma_tiler_mn,  # cta_tile_m
        permuted_m,
        seq_len,  # Pass seq_len as num_tokens for C tensor shape
    )

    # Calculate actual tensor_m used (with padding if permuted_m provided)
    tensor_m = permuted_m if permuted_m is not None else valid_m

    (
        permuted_idx_to_expanded_idx_torch,
        token_final_scales_torch,
        permuted_idx_to_expanded_idx,
        token_final_scales,
    ) = create_fused_finalize_tensors(
        seq_len,
        topK,
        tensor_m,
        group_m_list,
        mma_tiler_mn,
        final_scale_dtype,
    )

    # Configure gemm kernel
    gemm = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        raster_along_m,
    )

    # Compute max active clusters on current device
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # Initialize Stream
    current_stream = cutlass_torch.default_stream()

    # Compile gemm kernel
    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        out_tensor,
        sfa_tensor,
        sfb_tensor,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles,
        tile_idx_to_mn_limit,
        alpha,
        max_active_clusters,
        current_stream,
        permuted_idx_to_expanded_idx,
        token_final_scales,
        options="--opt-level 2",
    )

    # Compute reference result
    if not skip_ref_check:
        print("Verifying results...")
        # Execution
        compiled_gemm(
            a_tensor,
            b_tensor,
            out_tensor,
            sfa_tensor,
            sfb_tensor,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            current_stream,
            permuted_idx_to_expanded_idx,
            token_final_scales,
        )

        torch.cuda.synchronize()
        ref_result = verify_reference_result(
            a_torch_cpu,
            b_torch_cpu,
            sfa_torch_cpu,
            sfb_torch_cpu,
            alpha_torch_cpu,
            permuted_idx_to_expanded_idx_torch,
            token_final_scales_torch,
            group_m_list,
            aligned_group_m_list,
            out_torch_gpu.dtype,
            valid_m,
            n,
            topK,
            seq_len,
        )

        # Convert c back to f32 for comparison.
        actual_result = out_torch_gpu[:, :, 0]
        if out_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            torch.testing.assert_close(actual_result, ref_result, atol=tolerance, rtol=1e-02)
        elif out_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN):
            # Convert ref : f32 -> f8 -> f32
            ref_f8_ = torch.empty(*(1, valid_m, n), dtype=torch.uint8, device="cuda").permute(
                1, 2, 0
            )
            ref_f8 = from_dlpack(ref_f8_, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            ref_f8.element_type = out_dtype
            ref_device = ref_result.cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_f8)
            cute.testing.convert(ref_f8, ref_tensor)
            torch.testing.assert_close(
                actual_result.cpu(), ref_device.cpu(), atol=tolerance, rtol=1e-02
            )
        elif out_dtype is cutlass.Float4E2M1FN:
            # Convert ref : f32 -> f4 -> f32
            ref_f4_ = torch.empty(*(1, valid_m, n), dtype=torch.uint8, device="cuda").permute(
                1, 2, 0
            )
            ref_f4 = from_dlpack(ref_f4_, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            ref_f4.element_type = out_dtype
            ref_device = ref_result.cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_f4)
            cute.testing.convert(ref_f4, ref_tensor)
            torch.testing.assert_close(
                actual_result.cpu(), ref_device.cpu(), atol=tolerance, rtol=1e-02
            )

    def generate_tensors():
        (
            a_tensor,
            b_tensor,
            out_tensor,
            sfa_tensor,
            sfb_tensor,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            a_torch_cpu,
            b_torch_cpu,
            out_torch_cpu,
            sfa_torch_cpu,
            sfb_torch_cpu,
            alpha_torch_cpu,
            a_torch_gpu,
            b_torch_gpu,
            sfa_torch_gpu,
            sfb_torch_gpu,
            out_torch_gpu,
            aligned_group_m_list,
            valid_m,
        ) = create_tensors(
            l,
            group_m_list,
            n,
            k,
            a_major,
            b_major,
            out_major,
            ab_dtype,
            out_dtype,
            sf_dtype,
            sf_vec_size,
            mma_tiler_mn,  # cta_tile_m
            permuted_m,
            seq_len,  # Pass seq_len as num_tokens for C tensor shape
        )

        (
            permuted_idx_to_expanded_idx_torch,
            token_final_scales_torch,
            permuted_idx_to_expanded_idx,
            token_final_scales,
        ) = create_fused_finalize_tensors(
            seq_len,
            topK,
            tensor_m,
            group_m_list,
            mma_tiler_mn,
            final_scale_dtype,
        )

        return cute.testing.JitArguments(
            a_tensor,
            b_tensor,
            out_tensor,
            sfa_tensor,
            sfb_tensor,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            current_stream,
            permuted_idx_to_expanded_idx,
            token_final_scales,
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch_gpu.numel() * a_torch_gpu.element_size()
            + b_torch_gpu.numel() * b_torch_gpu.element_size()
            + out_torch_gpu.numel() * out_torch_gpu.element_size()
            + sfa_torch_gpu.numel() * sfa_torch_gpu.element_size()
            + sfb_torch_gpu.numel() * sfb_torch_gpu.element_size()
            + (tensor_m // mma_tiler_mn[0])
            * 4  # tile_idx_to_expert_idx length (tiles) * sizeof(int32)
            + 1 * 4  # num_non_exiting_tiles (1 element) * sizeof(int32)
            + alpha_torch_cpu.numel() * alpha_torch_cpu.element_size()
        )
        workspace_count = cute.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cupti=use_cupti,
    )

    return exec_time  # Return execution time in microseconds


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")

    def read_benchmark_file(
        filepath: str,
    ) -> Tuple[Tuple[int, int, int], Tuple[int, ...]]:
        """Read benchmark file and return nkl and group_m_list.

        File format:
            0 256x512x128
            1 512x512x512
            2 1024x256x128
            ...

        Returns:
            Tuple of ((n, k, l), (m0, m1, m2, ...)) where:
            - n, k are from the first problem
            - l is the total number of problems (groups)
            - (m0, m1, m2, ...) are the M values for each group
        """
        problems = []
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    # Parse index and dimensions (e.g., "256x512x128")
                    dims = parts[1].split("x")
                    if len(dims) == 3:
                        m, n, k = int(dims[0]), int(dims[1]), int(dims[2])
                        problems.append((m, n, k))

            if not problems:
                raise ValueError(f"No valid problems found in benchmark file: {filepath}")

            # Use first problem's N, K dimensions
            m_first, n, k = problems[0]
            l = len(problems)  # noqa: E741

            # Extract M values for each group
            m_values = tuple(m for m, _, _ in problems)

            print(f"Loaded {l} problems from benchmark file")
            print(f"Using N={n}, K={k}, L={l}")
            print(f"M values per group: {m_values}")

            return ((n, k, l), m_values)

        except FileNotFoundError:
            raise argparse.ArgumentTypeError(f"Benchmark file not found: {filepath}")
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Error reading benchmark file: {e}")

    def parse_benchmark_arg(
        arg: str,
    ) -> Tuple[Tuple[int, int, int], Tuple[int, ...]]:
        """Parse benchmark argument string.

        Supported formats:
        1. 'MxNxKxL': 128x512x1024x4 -> n=512, k=1024, l=4, m_values=(128, 128, 128, 128)
        2. '[m0,m1,...]xNxK': [128,256]x512x1024 -> n=512, k=1024, l=2, m_values=(128, 256)
        """
        # Try matching [m0, m1, ...]xNxK format
        match_list = re.match(r"\[([\d,\s]+)\]\s*x\s*(\d+)\s*x\s*(\d+)", arg)
        if match_list:
            m_str = match_list.group(1)
            n = int(match_list.group(2))
            k = int(match_list.group(3))
            try:
                m_values = tuple(int(x.strip()) for x in m_str.split(","))
                l = len(m_values)  # noqa: E741
                print(f"Parsed benchmark arg: N={n}, K={k}, L={l}")
                print(f"M values per group: {m_values}")
                return ((n, k, l), m_values)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid integer list in benchmark argument: {arg}"
                )

        # Try matching MxNxKxL format
        parts = arg.split("x")
        if len(parts) == 4:
            try:
                m, n, k, l = [int(x.strip()) for x in parts]  # noqa: E741
                m_values = tuple([m] * l)
                print(f"Parsed benchmark arg: M={m}, N={n}, K={k}, L={l}")
                return ((n, k, l), m_values)
            except ValueError:
                pass

        raise argparse.ArgumentTypeError(
            f"Invalid benchmark argument format. Expected file path, 'MxNxKxL', or '[m0,m1,...]xNxK'. Got: {arg}"
        )

    parser = argparse.ArgumentParser(
        description="Example of BlockScaled Contiguous grouped GEMM finalize fusion kernel on Blackwell."
    )

    parser.add_argument(
        "--nkl",
        type=parse_comma_separated_ints,
        default=(256, 512, 1),
        help="nkl dimensions: N, K, L (number of groups) (comma-separated)",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to benchmark file with problem sizes (format: 'index MxNxK' per line),"
        "or 'MxNxKxL' format or '[m0,m1,...]xNxK' format",
    )

    parser.add_argument(
        "--permuted_m",
        type=int,
        default=None,
        help="Optional padded M dimension for CUDA graph support. If specified,"
        "A/C matrices and scale factor A will be padded to this size."
        "Example: For MoE with m=4096, topK=8, experts_per_rank=8: permuted_m = 4096*8 + 8*255 = 34808",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=4096,
        help="Sequence length for MoE, used by fused finalize, need to know"
        "the sequence length to do finalize correctly",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Top-K experts per token (used for fused finalize)",
    )

    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--out_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--final_scale_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--a_major", choices=["k"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k"], type=str, default="k")
    parser.add_argument("--out_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--tolerance", type=float, default=1e-01, help="Tolerance for validation")
    parser.add_argument("--warmup_iterations", type=int, default=0, help="Warmup iterations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")
    parser.add_argument("--use_cold_l2", action="store_true", default=False, help="Use cold L2")

    parser.add_argument(
        "--raster_along_m",
        action="store_true",
        dest="raster_along_m",
        default=False,
        help="Enable raster along M (default: False)",
    )

    parser.add_argument(
        "--use_cupti",
        action="store_true",
        default=False,
        help="Use CUPTI to measure execution time",
    )

    args = parser.parse_args()

    # Process arguments to generate nkl and group_m_list
    if args.benchmark:
        # Read from benchmark file or parse string
        if os.path.isfile(args.benchmark):
            nkl, group_m_list = read_benchmark_file(args.benchmark)
        else:
            nkl, group_m_list = parse_benchmark_arg(args.benchmark)
    else:
        parser.error("No benchmark file or benchmark argument provided")
    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    exec_time = run(
        nkl,
        group_m_list,
        args.ab_dtype,
        args.out_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.final_scale_dtype,
        args.a_major,
        args.b_major,
        args.out_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        args.permuted_m,
        args.topk,
        args.seq_len,
        args.raster_along_m,
        args.use_cupti,
    )
    print("exec_time: ", exec_time)
    print("PASS")
