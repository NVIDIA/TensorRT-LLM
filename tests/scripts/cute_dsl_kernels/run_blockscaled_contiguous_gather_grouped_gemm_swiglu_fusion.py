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

"""Example usage of the kernel.

Functional testing:
python run_blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py \
        --ab_dtype Float4E2M1FN --c_dtype Float4E2M1FN \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \
        --nkl 4096,7168,8 --fixed_m 128
or use a benchmark file:
python run_blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py \
        --ab_dtype Float4E2M1FN --c_dtype Float4E2M1FN \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \
        --benchmark benchmark.txt
Perf testing:
python run_blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py \
        --ab_dtype Float4E2M1FN --c_dtype Float4E2M1FN \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \
        --benchmark benchmark.txt \
        --skip_ref_check --use_cold_l2 --use_cupti --warmup_iterations 10 --iterations 50
A sample benchmark.txt file is shown below:
0 89x4096x7168
1 200x4096x7168
2 145x4096x7168
3 178x4096x7168
4 241x4096x7168
5 78x4096x7168
6 198x4096x7168
7 60x4096x7168
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
        blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion as kernel_module,
    )
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).parents[3] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell import blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion as kernel_module

BlockScaledContiguousGatherGroupedGemmKernel = (
    kernel_module.BlockScaledContiguousGatherGroupedGemmKernel
)

try:
    from .testing import benchmark
except ImportError:
    from testing import benchmark


cvt_sf_MKL_to_M32x4xrm_K4xrk_L = kernel_module.cvt_sf_MKL_to_M32x4xrm_K4xrk_L
cvt_sf_M32x4xrm_K4xrk_L_to_MKL = kernel_module.cvt_sf_M32x4xrm_K4xrk_L_to_MKL


def create_mask(group_m_list, mma_tiler_m, permuted_m=None):
    """Create mask and group mapping for contiguous grouped GEMM.

    :param group_m_list: List of M values for each group (will be aligned to mma_tiler_m)
    :param mma_tiler_m: MMA tile size in M dimension (from mma_tiler_mn[0]), also used for alignment
    :param permuted_m: Optional padded M dimension for cuda_graph support. If provided,
                     tile_idx_to_expert_idx will be padded to this size.
                     When tile_idx >= num_non_exiting_tiles, the kernel exits.

    Note: For cuda_graph support, set permuted_m to the pre-calculated padded size:
          permuted_m = m * topK + num_local_experts * (256 - 1)
          Example: 4096*8 + (256/32)*255 = 34808
          Only the actual valid rows (aligned_groupm[0]+aligned_groupm[1]+...) contain
          valid data. The kernel will exit when tile_idx >= num_non_exiting_tiles.

    :return: Tuple of (valid_m, aligned_group_m_list, tile_idx_to_expert_idx, num_non_exiting_tiles)
             - tile_idx_to_expert_idx: shape (permuted_m/mma_tiler_m,) if permuted_m provided,
               else (valid_m/mma_tiler_m,)
             - num_non_exiting_tiles: scalar value = valid_m/mma_tiler_m
    """
    valid_m = 0
    aligned_group_m_list = []
    tile_idx_to_expert_idx = []
    tile_idx_to_mn_limit = []

    for i, group_m in enumerate(group_m_list):
        aligned_group_m = ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m
        aligned_group_m_list.append(aligned_group_m)

        # Calculate number of tiles for this group based on MMA tile M size
        # Each tile covers mma_tiler_m rows in M dimension
        num_tiles_in_group = aligned_group_m // mma_tiler_m
        # Add expert_idx for each tile in this group
        tile_idx_to_expert_idx.extend([i] * num_tiles_in_group)
        for tile_idx_in_group in range(num_tiles_in_group):
            tile_idx_to_mn_limit.append(
                valid_m + min(tile_idx_in_group * mma_tiler_m + mma_tiler_m, group_m)
            )
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
            # Calculate how many padding tiles are needed based on MMA tile M size
            num_padding_tiles = (permuted_m - valid_m) // mma_tiler_m
            # Pad with 0 (these tiles won't be accessed due to num_non_exiting_tiles check)
            tile_idx_to_expert_idx.extend([int(-2e9)] * num_padding_tiles)
            tile_idx_to_mn_limit.extend([int(-2e9)] * num_padding_tiles)

    # Final shape of tile_idx_to_expert_idx: (permuted_m/mma_tiler_m,) if permuted_m provided,
    # else (valid_m/mma_tiler_m,)
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


def create_scale_factor_tensor(num_groups, mn, k, sf_vec_size, dtype):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(k, sf_vec_size)
    ref_shape = (num_groups, mn, sf_k)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        num_groups,
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
        .expand(num_groups, mn, sf_k, sf_vec_size)
        .reshape(num_groups, mn, sf_k * sf_vec_size)
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


def create_scale_factor_tensor_unswizzled(num_groups, mn, k, sf_vec_size, dtype):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(k, sf_vec_size)
    sf_ref = cutlass_torch.matrix(
        num_groups,
        mn,
        sf_k,
        False,
        cutlass.Float32,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=1,
            max_val=3,
        ),
    )

    sf_tensor, sf_torch = cutlass_torch.cute_tensor_like(
        sf_ref, dtype, is_dynamic_layout=True, assumed_align=16
    )

    # reshape makes memory contiguous
    sf_ref = (
        sf_ref.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(num_groups, mn, sf_k, sf_vec_size)
        .reshape(num_groups, mn, sf_k * sf_vec_size)
        .permute(1, 2, 0)
    )

    # print(sf_ref[0])
    sf_ref = sf_ref[:, :k, :]
    return sf_ref, sf_tensor, sf_torch


def create_sf_layout_tensor(num_groups, mn, nk, sf_vec_size):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(nk, sf_vec_size)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        num_groups,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    mma_permute_order = (3, 4, 1, 5, 2, 0)

    # Create f32 cute torch tensor (cpu)
    cute_f32_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=0,
            max_val=1,
        ),
    )
    return cute_f32_torch_tensor, sf_k


# Create token_id_mapping tensor for gather operation
# This tensor maps each output row position to the corresponding input row position in A matrix
# Shape: (valid_m,) or (permuted_m,) if padding is requested
# Each element contains the token ID (input row index) for that output position
# Invalid positions (padding) are marked with -1
def create_token_id_mapping_tensor(group_m_list, mma_tiler_m, max_token_id, permuted_m=None):
    """Create token_id_mapping tensor for gather operation with random distribution.

    :param group_m_list: List of M values for each group (will be aligned to mma_tiler_m)
    :param mma_tiler_m: MMA tile size in M dimension (from mma_tiler_mn[0]), also used for alignment
    :param max_token_id: Maximum token ID (typically the number of input tokens)
    :param permuted_m: Optional padded M dimension for cuda_graph support

    Note: In real MoE scenarios, tokens are randomly routed to different experts,
          so we use random sampling (with replacement) to simulate this distribution.
    """
    valid_m = 0
    for group_m in group_m_list:
        valid_m += ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m

    tensor_m = permuted_m if permuted_m is not None else valid_m

    # Initialize all values to -1 (invalid token)
    base_data = torch.full((tensor_m,), -1, dtype=torch.int32)

    accumulated_m = 0
    for group_m in group_m_list:
        start_idx = accumulated_m
        rounded_group_m = ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m
        # Generate random token IDs instead of contiguous ones
        # Random sampling with replacement to simulate real MoE routing
        random_token_ids = torch.randint(0, max_token_id, (group_m,), dtype=torch.int32)
        # perfect_token_ids = torch.arange(0, group_m, dtype=torch.int32)
        base_data[start_idx : start_idx + group_m] = random_token_ids
        accumulated_m += rounded_group_m

    token_id_mapping_ref = base_data.clone()
    token_id_mapping_tensor, token_id_mapping_torch = cutlass_torch.cute_tensor_like(
        token_id_mapping_ref, cutlass.Int32, is_dynamic_layout=True, assumed_align=4
    )
    return token_id_mapping_ref, token_id_mapping_tensor, token_id_mapping_torch


def create_tensors(
    num_groups,
    group_m_list,
    n,
    k,
    a_major,
    b_major,
    cd_major,
    ab_dtype,
    c_dtype,
    sf_dtype,
    sf_vec_size,
    mma_tiler_m,
    permuted_m=None,
):
    """Create tensors for contiguous grouped GEMM with gather operation and SwiGLU fusion.

    Note: Output C has N/2 columns since SwiGLU combines pairs of (up, gate) from interleaved B weights.
    B weights are expected to be interleaved: [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]

    Returns tensors including:
    - A: Input matrix (MxKx1)
    - B: Weight matrix with interleaved up/gate weights (NxKxL)
    - C: Output matrix (Mx(N/2)x1), N is halved due to SwiGLU fusion
    - SFA, SFB: Scale factor matrices for A and B
    - SFC: Scale factor matrix for C (only when c_dtype is Float4E2M1FN)
    - tile_idx_to_expert_idx: Maps tile index to expert/group ID
    - token_id_mapping: Maps output row position to input row position (for gather)
    - num_non_exiting_tiles: Number of valid tiles to process

    :param mma_tiler_m: MMA tile size in M dimension (from mma_tiler_mn[0]), also used for alignment
    :param permuted_m: Optional padded M dimension for cuda_graph support. If provided,
                     A matrix, C matrix, token_id_mapping, and scale factor A will be padded to this size.
                     The kernel exits when tile_idx >= num_non_exiting_tiles.

    Example with CUDA graph padding:
        # For MoE: m=4096, topK=8, num_local_experts=256, experts_per_rank=8
        permuted_m = 4096 * 8 + 8 * 255  # = 34808
        tensors = create_tensors(
            num_groups=8,  # num_local_experts
            group_m_list=[512, 1024, ...],  # actual group sizes
            n=4096, k=7168,
            a_major="k", b_major="k", cd_major="n",
            ab_dtype=cutlass.Float4E2M1FN,
            c_dtype=cutlass.BFloat16,
            sf_dtype=cutlass.Float8E4M3FN,
            sf_vec_size=16,
            mma_tiler_m=128,  # MMA tile size in M dimension, also used for alignment
            permuted_m=34808  # Enable padding for cuda_graph
        )
        # Returns tensors with A, C, SFA, and token_id_mapping padded to permuted_m size,
        # kernel exits early when tile_idx >= num_non_exiting_tiles
    """
    torch.manual_seed(1111)

    alpha_torch_cpu = torch.randn((num_groups,), dtype=torch.float32)

    (
        valid_m,
        aligned_group_m_list,
        _tile_idx_to_expert_idx,
        _num_non_exiting_tiles,
        _tile_idx_to_mn_limit,
    ) = create_mask(group_m_list, mma_tiler_m, permuted_m)

    max_m = max(group_m_list)

    # Use permuted_m for A/C tensors if provided (for cuda_graph support)
    tensor_m = permuted_m if permuted_m is not None else valid_m

    a_torch_cpu = cutlass_torch.matrix(1, max_m, k, a_major == "m", cutlass.Float32)
    b_torch_cpu = cutlass_torch.matrix(num_groups, n, k, b_major == "n", cutlass.Float32)
    # C tensor also uses tensor_m (permuted_m) for cuda_graph support
    c_torch_cpu = cutlass_torch.matrix(1, tensor_m, n // 2, cd_major == "m", cutlass.Float32)

    a_tensor, a_torch_gpu = cutlass_torch.cute_tensor_like(
        a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch_gpu = cutlass_torch.cute_tensor_like(
        b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
        c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
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
    c_tensor.mark_compact_shape_dynamic(
        mode=1 if cd_major == "n" else 0,
        stride_order=(2, 0, 1) if cd_major == "n" else (2, 1, 0),
        divisibility=32 if ab_dtype == cutlass.Float4E2M1FN else 16,
    )

    # Use tensor_m (permuted_m if provided) for scale factor A
    sfa_torch_cpu, sfa_tensor, sfa_torch_gpu = create_scale_factor_tensor_unswizzled(
        1, max_m, k, sf_vec_size, sf_dtype
    )

    sfb_torch_cpu, sfb_tensor, sfb_torch_gpu = create_scale_factor_tensor(
        num_groups, n, k, sf_vec_size, sf_dtype
    )

    token_id_mapping_cpu, token_id_mapping, token_id_mapping_torch = create_token_id_mapping_tensor(
        group_m_list, mma_tiler_m, max_token_id=max_m, permuted_m=permuted_m
    )

    tile_idx_to_expert_idx = from_dlpack(_tile_idx_to_expert_idx).mark_layout_dynamic()
    tile_idx_to_mn_limit = from_dlpack(_tile_idx_to_mn_limit).mark_layout_dynamic()
    num_non_exiting_tiles = from_dlpack(_num_non_exiting_tiles).mark_layout_dynamic()

    alpha = from_dlpack(alpha_torch_cpu.cuda()).mark_layout_dynamic()

    # Create sfc_tensor and norm_const_tensor when c_dtype is Float4E2M1FN
    sfc_torch_cpu = None
    sfc_tensor = None
    sfc_torch_gpu = None
    norm_const_torch_cpu = None
    norm_const_tensor = None
    norm_const_torch_gpu = None
    n_out = n // 2  # Output N dimension after SwiGLU fusion
    if c_dtype == cutlass.Float4E2M1FN:
        # Create scale factor C tensor for quantized output
        sfc_torch_cpu, sfc_tensor, sfc_torch_gpu = create_scale_factor_tensor(
            1, tensor_m, n_out, sf_vec_size, sf_dtype
        )
        # Create norm_const_tensor with value 1.0
        norm_const_torch = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        norm_const_tensor = from_dlpack(norm_const_torch).mark_layout_dynamic()
        norm_const_torch_cpu = norm_const_torch.cpu()

    return (
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
        sfc_torch_cpu,
        norm_const_torch_cpu,
        alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        c_torch_gpu,
        sfa_torch_gpu,
        sfb_torch_gpu,
        sfc_torch_gpu,
        norm_const_torch_gpu,
        aligned_group_m_list,
        valid_m,
        token_id_mapping_cpu,
    )


def run(
    nkl: Tuple[int, int, int],
    group_m_list: Tuple[int, ...],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    permuted_m: int = None,
    use_cupti: bool = False,
    raster_along_m: bool = False,
    **kwargs,
):
    """Run contiguous grouped GEMM with gather operation and SwiGLU fusion for FC1 layer.

    Computation flow:
    1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
    2. SwiGLU: C = up * silu(gate), where up/gate are extracted from interleaved acc (granularity=64)
    3. Optional Quant: When c_dtype is Float4E2M1FN, generates SFC and quantizes output

    Note: Output C has N/2 columns since SwiGLU combines pairs of (up, gate) from interleaved B weights.

    This function:
    - Creates tensors including token_id_mapping for gather operation
    - Uses LDGSTS for loading A and SFA matrices with gather capability
    - Uses TMA for loading B and SFB matrices with multicast
    - Performs SwiGLU activation fusion in epilogue
    - Optionally performs quantization fusion for Float4E2M1FN output
    - Performs reference checking (if not skipped)
    - Benchmarks kernel performance

    :param nkl: (N, K, L) dimensions where L is the number of experts/groups
    :param group_m_list: List of M values for each group
    :param mma_tiler_mn: MMA tile shape (M, N), where mma_tiler_mn[0] is used for group M alignment
    :param permuted_m: Optional padded M dimension for CUDA graph support. If provided,
                     A/C matrices, token_id_mapping, and scale factor A will be padded to this size.
    """
    print("Running Blackwell Persistent Contiguous Grouped GEMM with Gather test:")
    print(f"nkl: {nkl}")
    print(f"group_m_list: {group_m_list}")
    print(
        f"AB dtype: {ab_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}, "
        f"Scale factor dtype: {sf_dtype}, SF Vec size: {sf_vec_size}"
    )
    if permuted_m is not None:
        print(f"Padded M (CUDA graph support): {permuted_m}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"2CTA MMA instructions: {'True' if mma_tiler_mn[0] == 256 else 'False'}")
    print(f"Use TMA Store: {'True'}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {use_cold_l2}")
    print(f"Use CUPTI: {use_cupti}")
    print(f"Raster along M: {raster_along_m}")

    # Unpack parameters
    n, k, num_groups = nkl

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # Skip unsupported testcase
    # Note: For grouped GEMM, we use mma_tiler_mn[0] as the m parameter for can_implement check
    # since individual group M values vary
    if not BlockScaledContiguousGatherGroupedGemmKernel.can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        mma_tiler_mn[0],  # m (use mma_tiler_m as placeholder for grouped GEMM)
        n,
        k,
        num_groups,
        a_major,
        b_major,
        c_major,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {acc_dtype}, "
            f"{c_dtype}, {mma_tiler_mn}, {cluster_shape_mn}, {n}, {k}, {num_groups}, "
            f"{a_major}, {b_major}, {c_major}"
        )

    (
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
        sfc_torch_cpu,
        norm_const_torch_cpu,
        alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        c_torch_gpu,
        sfa_torch_gpu,
        sfb_torch_gpu,
        sfc_torch_gpu,
        norm_const_torch_gpu,
        aligned_group_m_list,
        valid_m,
        token_id_mapping_cpu,
    ) = create_tensors(
        num_groups,
        group_m_list,
        n,
        k,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        c_dtype,
        sf_dtype,
        sf_vec_size,
        mma_tiler_mn[0],  # mma_tiler_m, also used for alignment
        permuted_m,
    )
    # Configure gemm kernel
    gemm = BlockScaledContiguousGatherGroupedGemmKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        True,
        topk=1,
        raster_along_m=raster_along_m,
    )

    # Compute max active clusters on current device
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    # Compile gemm kernel
    # sfc_tensor is optional and can be set as None (Python's None value) if not needed.
    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        max_active_clusters,
        current_stream,
    )

    # Execution
    compiled_gemm(
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        current_stream,
    )

    torch.cuda.synchronize()
    # Compute reference result
    if not skip_ref_check:
        print("Verifying results...")
        # SwiGLU fusion with interleaved weights at granularity 64
        # Output has N/2 columns since pairs of (up, gate) are combined
        interleave_granularity = 64
        n_out = n // 2

        # Step 1: Compute full GEMM first
        gemm_result = torch.empty((1, valid_m, n), dtype=torch.float32)
        start = 0
        a_torch_cpu_f32 = torch.einsum("mk,mk->mk", a_torch_cpu[:, :, 0], sfa_torch_cpu[:, :, 0])
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            res_a = a_torch_cpu_f32[token_id_mapping_cpu[start:end]]
            res_b = torch.einsum("nk,nk->nk", b_torch_cpu[:, :, i], sfb_torch_cpu[:, :, i])
            gemm_result[0, start:end, :] = (
                torch.einsum("mk,nk->mn", res_a, res_b) * alpha_torch_cpu[i]
            )
            start = end

        # Step 2: Apply SwiGLU on interleaved GEMM result
        # Weights are interleaved: [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]
        assert n % (2 * interleave_granularity) == 0
        ref = torch.empty((1, valid_m, n_out), dtype=torch.float32)
        for n_block in range(0, n, 2 * interleave_granularity):
            # Up projection result: columns n_block to n_block+64
            up_result = gemm_result[0, :, n_block : n_block + interleave_granularity]
            # Gate projection result: columns n_block+64 to n_block+128
            gate_result = gemm_result[
                0, :, n_block + interleave_granularity : n_block + 2 * interleave_granularity
            ]

            # SwiGLU: up * silu(gate) where silu(x) = x * sigmoid(x)
            silu_gate = gate_result * torch.sigmoid(gate_result)
            output_block = up_result * silu_gate

            # Store to output at n_block/2 position
            out_start = n_block // 2
            out_end = out_start + interleave_granularity
            ref[0, :, out_start:out_end] = output_block

        ref = ref.permute((1, 2, 0))

        # Convert c back to f32 for comparison.
        res = c_torch_cpu.cuda()
        cute.testing.convert(
            c_tensor,
            from_dlpack(res, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if c_major == "n" else 0)
            ),
        )

        # Only compare valid rows (in case of padding for cuda_graph)
        res = res[:valid_m]
        mask = token_id_mapping_cpu[:valid_m] >= 0
        res = res.cpu()[mask]
        ref = ref[mask]

        print(f"valid_m: {valid_m}, ref.shape: {ref.shape}, res.shape: {res.shape}")

        if c_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            torch.testing.assert_close(res.cpu(), ref.cpu(), atol=tolerance, rtol=1e-02)
        elif c_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN):
            # Convert ref : f32 -> f8 -> f32
            ref_f8_ = torch.empty(*(1, valid_m, n_out), dtype=torch.uint8, device="cuda").permute(
                1, 2, 0
            )
            ref_f8 = from_dlpack(ref_f8_, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            ref_f8.element_type = c_dtype
            ref_device = ref.cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_f8)
            cute.testing.convert(ref_f8, ref_tensor)
            torch.testing.assert_close(res.cpu(), ref_device.cpu(), atol=tolerance, rtol=1e-02)
        elif c_dtype is cutlass.Float4E2M1FN:

            def ceil_div(a, b):
                return (a + b - 1) // b

            def simulate_f8_quantization(tensor_f32: torch.Tensor, f8_dtype) -> torch.Tensor:
                """Simulate f8 quantization: fp32 -> f8 -> fp32.

                This models the precision loss when storing scale factors in f8 format.

                :param tensor_f32: Input fp32 tensor (on CPU), shape (m, n, num_groups)
                :param f8_dtype: Target f8 dtype (e.g., cutlass.Float8E4M3FN)
                :return: Tensor after f32 -> f8 -> f32 round-trip (on CPU)
                """
                shape = tensor_f32.shape
                # Create f8 tensor on GPU
                f8_torch = torch.empty(*shape, dtype=torch.uint8, device="cuda")
                f8_tensor = from_dlpack(f8_torch, assumed_align=16).mark_layout_dynamic(
                    leading_dim=1
                )
                f8_tensor.element_type = f8_dtype
                # Create f32 tensor on GPU
                f32_device = tensor_f32.cuda()
                f32_tensor = from_dlpack(f32_device, assumed_align=16).mark_layout_dynamic(
                    leading_dim=1
                )
                # f32 -> f8 -> f32
                cute.testing.convert(f32_tensor, f8_tensor)
                cute.testing.convert(f8_tensor, f32_tensor)
                return f32_device.cpu()

            def simulate_nvfp4_quantization(tensor_f32: torch.Tensor) -> torch.Tensor:
                """Simulate nvfp4 quantization: fp32 -> nvfp4 -> fp32.

                This models the precision loss when storing output in nvfp4 format.

                :param tensor_f32: Input fp32 tensor (on CPU), shape (m, n, ng)
                :return: Tensor after f32 -> nvfp4 -> f32 round-trip (on CPU)
                """
                m, n, ng = tensor_f32.shape
                # Create properly packed nvfp4 tensor using cutlass_torch utilities
                ref_f32_torch = cutlass_torch.matrix(ng, m, n, False, cutlass.Float32)
                f4_tensor, _ = cutlass_torch.cute_tensor_like(
                    ref_f32_torch, cutlass.Float4E2M1FN, is_dynamic_layout=True, assumed_align=16
                )
                # Create f32 tensor on GPU
                f32_device = tensor_f32.cuda()
                f32_tensor = from_dlpack(f32_device, assumed_align=16).mark_layout_dynamic(
                    leading_dim=1
                )
                # f32 -> f4 -> f32
                cute.testing.convert(f32_tensor, f4_tensor)
                cute.testing.convert(f4_tensor, f32_tensor)
                return f32_device.cpu()

            def compute_scale_factor(
                tensor_f32: torch.Tensor,
                sf_vec_size: int,
                norm_const: float,
                rcp_limits: float,
            ) -> torch.Tensor:
                """Compute scale factor for nvfp4 quantization.

                Scale factor = abs_max_per_vector * norm_const * rcp_limits

                :param tensor_f32: Input fp32 tensor, shape (m, n, ng)
                :param sf_vec_size: Vector size for scale factor (e.g., 16)
                :param norm_const: Normalization constant
                :param rcp_limits: Reciprocal of dtype max value (e.g., 1/6.0 for nvfp4)
                :return: Scale factor tensor, shape (m, sfn, ng) where sfn = ceil(n / sf_vec_size)
                """
                m, n, ng = tensor_f32.shape
                sfn = ceil_div(n, sf_vec_size)
                # Reshape to (m, sfn, sf_vec_size, ng) for abs max computation
                # Pad n dimension if needed
                padded_n = sfn * sf_vec_size
                if padded_n > n:
                    tensor_padded = torch.zeros(m, padded_n, ng, dtype=tensor_f32.dtype)
                    tensor_padded[:, :n, :] = tensor_f32
                else:
                    tensor_padded = tensor_f32
                tensor_reshaped = tensor_padded.view(m, sfn, sf_vec_size, ng)
                # Compute abs max over sf_vec_size dimension
                abs_max, _ = torch.abs(tensor_reshaped).max(dim=2)  # (m, sfn, l)
                # Compute scale factor
                scale_factor = abs_max * norm_const * rcp_limits
                return scale_factor

            def apply_quantization_scale(
                tensor_f32: torch.Tensor,
                scale_factor: torch.Tensor,
                sf_vec_size: int,
                norm_const: float,
            ) -> torch.Tensor:
                """Apply quantization scale to tensor.

                Output = tensor * (norm_const / scale_factor).
                This simulates the kernel's quantization scaling.

                :param tensor_f32: Input fp32 tensor, shape (m, n, ng)
                :param scale_factor: Scale factor tensor, shape (m, sfn, ng)
                :param sf_vec_size: Vector size for scale factor
                :param norm_const: Normalization constant
                :return: Scaled tensor, shape (m, n, ng)
                """
                m, n, ng = tensor_f32.shape
                sfn = scale_factor.shape[1]
                # Compute reciprocal scale, clamping inf to fp32_max (matching kernel fmin behavior)
                fp32_max = torch.tensor(3.40282346638528859812e38, dtype=torch.float32)
                scale_rcp = norm_const * scale_factor.reciprocal()
                scale_rcp = torch.where(torch.isinf(scale_rcp), fp32_max, scale_rcp)
                # Expand scale factor to match tensor dimensions
                # (m, sfn, ng) -> (m, sfn, sf_vec_size, ng) -> (m, sfn * sf_vec_size, ng)
                scale_rcp_expanded = scale_rcp.unsqueeze(2).expand(m, sfn, sf_vec_size, ng)
                scale_rcp_expanded = scale_rcp_expanded.reshape(m, sfn * sf_vec_size, ng)
                # Trim to exact n dimension
                scale_rcp_expanded = scale_rcp_expanded[:, :n, :]
                # Apply scale
                return tensor_f32 * scale_rcp_expanded

            def unswizzle_kernel_sfc(
                sfc_tensor,
                permuted_m: int,
                n_out: int,
                sf_vec_size: int,
            ) -> torch.Tensor:
                """Unswizzle kernel's scale factor tensor from MMA layout to MKL layout.

                :param sfc_tensor: Kernel's scale factor cute tensor (swizzled MMA layout)
                :param permuted_m: Padded M dimension
                :param n_out: Output N dimension
                :param sf_vec_size: Vector size for scale factor
                :return: Unswizzled scale factor tensor, shape (permuted_m, sfn, 1)
                """
                sfn = ceil_div(n_out, sf_vec_size)
                unswizzled_sfc = torch.empty(permuted_m, sfn, 1, dtype=torch.float32)
                # Create swizzled layout tensor and convert from kernel sfc
                swizzled_sfc_cpu, _ = create_sf_layout_tensor(1, permuted_m, n_out, sf_vec_size)
                swizzled_sfc_tensor, swizzled_sfc_torch = cutlass_torch.cute_tensor_like(
                    swizzled_sfc_cpu, cutlass.Float32, is_dynamic_layout=True, assumed_align=16
                )
                cute.testing.convert(sfc_tensor, swizzled_sfc_tensor)
                swizzled_sfc_cpu = swizzled_sfc_torch.cpu()
                # Convert from swizzled to MKL layout
                cvt_sf_M32x4xrm_K4xrk_L_to_MKL(
                    from_dlpack(swizzled_sfc_cpu),
                    from_dlpack(unswizzled_sfc),
                )
                return unswizzled_sfc

            # ============================================================
            # Step 1: Compute reference scale factor (SFC) from SwiGLU output
            # ============================================================
            norm_const = norm_const_torch_cpu.item()
            rcp_limits = gemm.get_dtype_rcp_limits(c_dtype)

            # Compute reference SFC: abs_max * norm_const * rcp_limits
            ref_sfc_f32 = compute_scale_factor(ref, sf_vec_size, norm_const, rcp_limits)
            # Simulate f8 quantization for SFC (kernel stores SFC in f8 format)
            ref_sfc_f32 = simulate_f8_quantization(ref_sfc_f32, sf_dtype)

            # ============================================================
            # Step 2: Verify kernel SFC matches reference SFC
            # ============================================================
            permuted_m = token_id_mapping_cpu.shape[0]
            kernel_sfc = unswizzle_kernel_sfc(sfc_tensor, permuted_m, n_out, sf_vec_size)
            torch.testing.assert_close(
                ref_sfc_f32, kernel_sfc[:valid_m][mask], atol=tolerance, rtol=1e-02
            )
            print("SFC Tensor comparison passed!")

            # ============================================================
            # Step 3: Apply quantization scale and simulate nvfp4 precision loss
            # ============================================================
            # Apply scale: ref_scaled = ref * (norm_const / sfc)
            ref_scaled = apply_quantization_scale(ref, ref_sfc_f32, sf_vec_size, norm_const)
            # Simulate nvfp4 quantization: f32 -> nvfp4 -> f32
            ref_quantized = simulate_nvfp4_quantization(ref_scaled)

            # ============================================================
            # Step 4: Compare kernel output with reference
            # ============================================================
            print("Verifying C Tensor...")
            res_cpu = res.cpu()
            diff = torch.abs(res_cpu - ref_quantized)
            within_tolerance = (diff <= tolerance) | (diff <= torch.abs(ref_quantized) * 1e-02)
            pass_rate = within_tolerance.float().mean().item()
            print(f"C Tensor pass rate: {pass_rate * 100:.2f}% (threshold: 95%)")
            assert pass_rate >= 0.95, (
                f"Only {pass_rate * 100:.2f}% elements within tolerance, expected >= 95%"
            )

    def generate_tensors():
        # Reuse existing CPU reference tensors and create new GPU tensors from them
        (
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            sfc_tensor,
            norm_const_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            a_torch_cpu,
            b_torch_cpu,
            c_torch_cpu,
            sfa_torch_cpu,
            sfb_torch_cpu,
            sfc_torch_cpu,
            norm_const_torch_cpu,
            alpha_torch_cpu,
            a_torch_gpu,
            b_torch_gpu,
            c_torch_gpu,
            sfa_torch_gpu,
            sfb_torch_gpu,
            sfc_torch_gpu,
            norm_const_torch_gpu,
            aligned_group_m_list,
            valid_m,
            token_id_mapping_cpu,
        ) = create_tensors(
            num_groups,
            group_m_list,
            n,
            k,
            a_major,
            b_major,
            c_major,
            ab_dtype,
            c_dtype,
            sf_dtype,
            sf_vec_size,
            mma_tiler_mn[0],  # mma_tiler_m, also used for alignment
            permuted_m,
        )
        return cute.testing.JitArguments(
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            sfc_tensor,
            norm_const_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        # Calculate actual tensor_m used (with padding if permuted_m provided)
        tensor_m = permuted_m if permuted_m is not None else valid_m
        one_workspace_bytes = (
            a_torch_gpu.numel() * a_torch_gpu.element_size()
            + b_torch_gpu.numel() * b_torch_gpu.element_size()
            + c_torch_gpu.numel() * c_torch_gpu.element_size()
            + sfa_torch_gpu.numel() * sfa_torch_gpu.element_size()
            + sfb_torch_gpu.numel() * sfb_torch_gpu.element_size()
            + (tensor_m // mma_tiler_mn[0])
            * 4  # tile_idx_to_expert_idx length (tiles) * sizeof(int32)
            + (tensor_m // mma_tiler_mn[0])
            * 4  # tile_idx_to_mn_limit length (tiles) * sizeof(int32)
            + tensor_m * 4  # token_id_mapping_tensor length (elements) * sizeof(int32)
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
    return exec_time


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
                    _ = int(parts[0])  # index (unused)
                    dims = parts[1].split("x")
                    if len(dims) == 3:
                        m, n, k = int(dims[0]), int(dims[1]), int(dims[2])
                        problems.append((m, n, k))

            if not problems:
                raise ValueError(f"No valid problems found in benchmark file: {filepath}")

            # Use first problem's N, K dimensions
            m_first, n, k = problems[0]
            num_groups = len(problems)

            # Extract M values for each group
            m_values = tuple(m for m, _, _ in problems)

            print(f"Loaded {num_groups} problems from benchmark file")
            print(f"Using N={n}, K={k}, L={num_groups}")
            print(f"M values per group: {m_values}")

            return ((n, k, num_groups), m_values)

        except FileNotFoundError:
            raise argparse.ArgumentTypeError(f"Benchmark file not found: {filepath}")
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Error reading benchmark file: {e}")

    parser = argparse.ArgumentParser(description="Example of Dense Persistent GEMM on Blackwell.")

    parser.add_argument(
        "--nkl",
        type=parse_comma_separated_ints,
        default=(256, 512, 1),
        help="nkl dimensions: N, K, L (number of groups) (comma-separated)",
    )

    parser.add_argument(
        "--fixed_m",
        type=int,
        default=None,
        help="Fixed M value for all groups. If specified, all groups will have this M dimension.",
    )

    parser.add_argument(
        "--custom_mask",
        type=parse_comma_separated_ints,
        default=None,
        help="Custom M values for each group (comma-separated integers). "
        "Length must match L dimension. Overrides --fixed_m.",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to benchmark file with problem sizes (format: 'index MxNxK' per line). "
        "If specified, overrides --nkl, --fixed_m, and --custom_mask.",
    )

    parser.add_argument(
        "--permuted_m",
        type=int,
        default=None,
        help="Optional padded M dimension for CUDA graph support. If specified, A/C matrices "
        "and scale factor A will be padded to this size. "
        "Example: permuted_m = 4096*8 + 8*255 = 34808",
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
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--a_major", choices=["k"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
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
        "--use_cupti", action="store_true", default=False, help="Use CUPTI for profiling"
    )
    parser.add_argument(
        "--raster_along_m", action="store_true", default=False, help="Raster along M dimension"
    )

    args = parser.parse_args()

    # Process arguments to generate nkl and group_m_list
    if args.benchmark:
        # Read from benchmark file
        nkl, group_m_list = read_benchmark_file(args.benchmark)
    else:
        # Use command line arguments
        if len(args.nkl) != 3:
            parser.error("--nkl must contain exactly 3 values")

        n, k, num_groups = args.nkl
        nkl = (n, k, num_groups)

        # Generate group_m_list based on --custom_mask or --fixed_m
        if args.custom_mask is not None:
            group_m_list = args.custom_mask
            if len(group_m_list) != num_groups:
                parser.error(
                    f"--custom_mask must have exactly {num_groups} values (matching L dimension)"
                )
        elif args.fixed_m is not None:
            group_m_list = tuple([args.fixed_m] * num_groups)
        else:
            # Default: use 128 for all groups
            group_m_list = tuple([128] * num_groups)

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    exec_time = run(
        nkl,
        group_m_list,
        args.ab_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        args.permuted_m,
        args.use_cupti,
        args.raster_along_m,
    )
    print(f"Execution time: {exec_time:.2f} us")
    print("PASS")
