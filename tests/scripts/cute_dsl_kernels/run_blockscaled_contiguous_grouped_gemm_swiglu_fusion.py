# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Example usage of the kernel.

Functional testing:

python run_blockscaled_contiguous_grouped_gemm_swiglu_fusion.py \
        --ab_dtype Float4E2M1FN --c_dtype Float4E2M1FN \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \
        --nkl 4096,7168,8 --fixed_m 128

or use a benchmark file:
python run_blockscaled_contiguous_grouped_gemm_swiglu_fusion.py \
        --ab_dtype Float4E2M1FN --c_dtype Float4E2M1FN \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \
        --benchmark benchmark.txt

Perf testing:

python run_blockscaled_contiguous_grouped_gemm_swiglu_fusion.py \
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

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
        blockscaled_contiguous_grouped_gemm_swiglu_fusion as kernel_module,
    )
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).parents[3] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell import blockscaled_contiguous_grouped_gemm_swiglu_fusion as kernel_module

Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel = (
    kernel_module.Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel
)
cvt_sf_MKL_to_M32x4xrm_K4xrk_L = kernel_module.cvt_sf_MKL_to_M32x4xrm_K4xrk_L

try:
    from .testing import benchmark
except ImportError:
    from testing import benchmark


def create_mask(group_m_list, cta_tile_m, m_aligned=128, permuted_m=None):
    """Create mask and group mapping for contiguous grouped GEMM.

    :param group_m_list: List of M values for each group (will be aligned to m_aligned)
    :param m_aligned: Alignment requirement for group M dimension
    :param cta_tile_m: CTA tile size in M dimension (from mma_tiler_mn[0])
    :param permuted_m: Optional padded M dimension for cuda_graph support. If provided,
                     tile_idx_to_expert_idx will be padded to this size.
                     When tile_idx >= num_non_exiting_tiles, the kernel exits.

    Note: m_aligned should be a multiple of the CTA tile M dimension to prevent
          a single tile from spanning multiple groups, which would cause incorrect
          B matrix access.

    Note: For cuda_graph support, set permuted_m to the pre-calculated padded size:
          permuted_m = m * topK + num_local_experts * (256 - 1)
          Example: 4096*8 + (256/32)*255 = 34808
          Only the actual valid rows (aligned_groupm[0]+aligned_groupm[1]+...) contain
          valid data. The kernel will exit when tile_idx >= num_non_exiting_tiles.

    :return: Tuple of (valid_m, aligned_group_m_list, tile_idx_to_expert_idx, num_non_exiting_tiles)
             - tile_idx_to_expert_idx: shape (permuted_m/cta_tile_m,) if permuted_m provided, else (valid_m/cta_tile_m,)
             - num_non_exiting_tiles: scalar value = valid_m/cta_tile_m
    """
    valid_m = 0
    aligned_group_m_list = []
    tile_idx_to_expert_idx = []

    for i, group_m in enumerate(group_m_list):
        aligned_group_m = ((group_m + m_aligned - 1) // m_aligned) * m_aligned
        valid_m += aligned_group_m
        aligned_group_m_list.append(aligned_group_m)

        # Calculate number of tiles for this group based on CTA tile M size
        # Each tile covers cta_tile_m rows in M dimension
        num_tiles_in_group = aligned_group_m // cta_tile_m
        # Add expert_idx for each tile in this group
        tile_idx_to_expert_idx.extend([i] * num_tiles_in_group)

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
            num_padding_tiles = (permuted_m - valid_m) // cta_tile_m
            # Pad with 0 (these tiles won't be accessed due to num_non_exiting_tiles check)
            tile_idx_to_expert_idx.extend([int(-2e9)] * num_padding_tiles)

    # Final shape of tile_idx_to_expert_idx: (permuted_m/cta_tile_m,) if permuted_m provided, else (valid_m/cta_tile_m,)
    tile_idx_to_expert_idx = torch.tensor(tile_idx_to_expert_idx, device="cuda", dtype=torch.int32)
    num_non_exiting_tiles_tensor = torch.tensor(
        [num_non_exiting_tiles], device="cuda", dtype=torch.int32
    )

    return (
        valid_m,
        aligned_group_m_list,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles_tensor,
    )


# Return a SF tensor with SF layout and the SF row dimension size
def create_sf_layout_tensor(l, mn, nk, sf_vec_size):  # noqa: E741
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(nk, sf_vec_size)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,  # noqa: E741
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


# Create scale factor tensor SFA/SFB/SFD
def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype):  # noqa: E741
    cute_f32_torch_tensor_cpu, sf_k = create_sf_layout_tensor(l, mn, k, sf_vec_size)
    ref_shape = (l, mn, sf_k)

    ref_permute_order = (1, 2, 0)

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
    c_dtype,
    sf_dtype,
    sf_vec_size,
    m_aligned,
    cta_tile_m,
    permuted_m=None,
    generate_sfc=False,
):
    """Create tensors for contiguous grouped GEMM.

    :param permuted_m: Optional padded M dimension for cuda_graph support. If provided,
                     A matrix, C matrix, and scale factor A will be padded to this size.
                     The kernel exits when tile_idx >= num_non_exiting_tiles.

    Example with CUDA graph padding:
        # For MoE: m=4096, topK=8, num_local_experts=256, experts_per_rank=8
        permuted_m = 4096 * 8 + 8 * 255  # = 34808
        tensors = create_tensors(
            l=8,  # num_local_experts
            group_m_list=[512, 1024, ...],  # actual group sizes
            n=4096, k=7168,
            a_major="k", b_major="k", cd_major="n",
            ab_dtype=cutlass.Float4E2M1FN,
            c_dtype=cutlass.BFloat16,
            sf_dtype=cutlass.Float8E4M3FN,
            sf_vec_size=16,
            m_aligned=256,
            cta_tile_m=128,  # CTA tile size in M dimension
            permuted_m=34808  # Enable padding for cuda_graph
        )
        # Returns tensors with A, C, and SFA padded to permuted_m rows,
        # kernel exits early when tile_idx >= num_non_exiting_tiles
    """
    torch.manual_seed(1111)

    alpha_torch_cpu = torch.ones((l,), dtype=torch.float32)
    # Initialize alpha with random integer values in [-3, 3]
    # alpha_torch_cpu = torch.randint(-3, 4, (l,), dtype=torch.float32)

    valid_m, aligned_group_m_list, _tile_idx_to_expert_idx, _num_non_exiting_tiles = create_mask(
        group_m_list, cta_tile_m, m_aligned, permuted_m
    )

    # Use permuted_m for A/C tensors if provided (for cuda_graph support)
    tensor_m = permuted_m if permuted_m is not None else valid_m

    a_torch_cpu = cutlass_torch.matrix(1, tensor_m, k, a_major == "m", cutlass.Float32)
    b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major == "n", cutlass.Float32)
    n_after_swiglu = n // 2
    # C tensor also uses tensor_m (permuted_m) for cuda_graph support
    c_torch_cpu = cutlass_torch.matrix(
        1, tensor_m, n_after_swiglu, cd_major == "m", cutlass.Float32
    )

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
    sfa_torch_cpu, sfa_tensor, sfa_torch_gpu = create_scale_factor_tensor(
        1, tensor_m, k, sf_vec_size, sf_dtype
    )
    sfb_torch_cpu, sfb_tensor, sfb_torch_gpu = create_scale_factor_tensor(
        l, n, k, sf_vec_size, sf_dtype
    )

    alpha = from_dlpack(alpha_torch_cpu.cuda()).mark_layout_dynamic()

    # Create sfc_tensor and norm_const_tensor when c_dtype is Float4E2M1FN
    sfc_torch_cpu = None
    sfc_tensor = None
    sfc_torch_gpu = None
    norm_const_tensor = None
    norm_const_torch_gpu = None
    if generate_sfc:
        # Create output scale factor tensor
        sfc_torch_cpu, sfc_tensor, sfc_torch_gpu = create_scale_factor_tensor(
            1, tensor_m, n_after_swiglu, sf_vec_size, sf_dtype
        )

        # Create norm constant tensor
        norm_const = 1.0
        norm_const_torch_gpu = torch.tensor([norm_const], dtype=torch.float32).cuda()
        norm_const_tensor = from_dlpack(norm_const_torch_gpu)

    tile_idx_to_expert_idx = from_dlpack(_tile_idx_to_expert_idx).mark_layout_dynamic()
    num_non_exiting_tiles = from_dlpack(_num_non_exiting_tiles).mark_layout_dynamic()

    return (
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles,
        alpha,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
        sfc_torch_cpu,
        alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        sfa_torch_gpu,
        sfb_torch_gpu,
        sfc_torch_gpu,
        c_torch_gpu,
        norm_const_torch_gpu,
        aligned_group_m_list,
        valid_m,
    )


def run(
    nkl: Tuple[int, int, int],
    group_m_list: Tuple[int, ...],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    vectorized_f32: bool,
    tolerance: float,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    permuted_m: int = None,
    use_cupti: bool = False,
    **kwargs,
):
    """Prepare A/B/C tensors, launch GPU kernel, and reference checking.

    :param permuted_m: Optional padded M dimension for CUDA graph support. If provided,
                     A/C matrices and scale factor A will be padded to this size.
    """
    m_aligned = mma_tiler_mn[0]

    print("Running Blackwell Persistent Dense Contiguous Grouped GEMM test with:")
    print(f"nkl: {nkl}")
    print(f"group_m_list: {group_m_list}")
    print(
        f"AB dtype: {ab_dtype}, C dtype: {c_dtype}, Scale factor dtype: {sf_dtype}, SF Vec size: {sf_vec_size}"
    )
    if permuted_m is not None:
        print(f"Padded M (CUDA graph support): {permuted_m}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")
    print(f"Use CUPTI: {'True' if use_cupti else 'False'}")

    # Unpack parameters
    n, k, l = nkl  # noqa: E741

    # If c_dtype is Float4E2M1FN, SFC will be generated
    generate_sfc = c_dtype == cutlass.Float4E2M1FN

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # Skip unsupported testcase
    if not Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel.can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        m_aligned,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype}, "
            f"{mma_tiler_mn}, {cluster_shape_mn}, {n}, {k}, {l}, "
            f"{a_major}, {b_major}, {c_major}, {m_aligned}"
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
        num_non_exiting_tiles,
        alpha,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
        sfc_torch_cpu,
        alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        sfa_torch_gpu,
        sfb_torch_gpu,
        sfc_torch_gpu,
        c_torch_gpu,
        norm_const_torch_gpu,
        aligned_group_m_list,
        valid_m,
    ) = create_tensors(
        l,
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
        m_aligned,
        mma_tiler_mn[0],  # cta_tile_m
        permuted_m,
        generate_sfc,
    )
    # Configure gemm kernel
    gemm = Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        vectorized_f32,
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
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles,
        alpha,
        max_active_clusters,
        current_stream,
    )

    # Compute reference result
    if not skip_ref_check:
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
            num_non_exiting_tiles,
            alpha,
            current_stream,
        )

        torch.cuda.synchronize()
        print("Verifying results...")
        ref = torch.empty((1, valid_m, n), dtype=torch.float32)
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            res_a = torch.einsum(
                "mk,mk->mk",
                a_torch_cpu[start:end, :, 0],
                sfa_torch_cpu[start:end, :, 0],
            )
            res_b = torch.einsum("nk,nk->nk", b_torch_cpu[:, :, i], sfb_torch_cpu[:, :, i])
            ref[0, start:end, :] = torch.einsum("mk,nk->mn", res_a, res_b) * alpha_torch_cpu[i]
            start = end

            print(f"ref: {ref.shape}, {ref.stride()}")
        ref = ref.permute((1, 2, 0))

        # Reference checking for SwiGLU output
        # group is epi tile size
        group = 64
        print(f" n : {n}, group : {group}")
        assert n % group == 0, "N must be divisible by 64 for GLU block grouping"
        num_blocks = n // group
        assert num_blocks % 2 == 0, "Number of 64-col blocks must be even (pairs of input/gate)"

        cols = torch.arange(n, device=ref.device, dtype=torch.long)
        block_cols = cols.view(num_blocks, group)
        # ref1: blocks 1,3,5,7 (1-based) => indices 0,2,4,6 (0-based)
        # ref2: blocks 2,4,6,8 (1-based) => indices 1,3,5,7 (0-based)
        up_idx = block_cols[0::2].reshape(-1)
        gate_idx = block_cols[1::2].reshape(-1)
        ref_up = ref.index_select(1, up_idx)
        ref_gate = ref.index_select(1, gate_idx)
        ref_after_swiglu = ref_up * (ref_gate * torch.sigmoid(ref_gate))
        print(f"ref: {ref.shape}, {ref.stride()}")

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

        # print("res: {} ref_device :{}", res.cpu()[127,:,0], ref_after_swiglu.cpu()[127,:,0])
        if c_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            print("Verifying C Tensor...")
            torch.testing.assert_close(
                res.cpu(), ref_after_swiglu.cpu(), atol=tolerance, rtol=1e-02
            )
        elif c_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN):
            n_after_swiglu = n // 2
            if generate_sfc:
                print("Verifying SFC Tensor...")
                norm_const = 1.0

                # 1. Compute reference SFC (m, sfn, l) in fp32
                def ceil_div(a, b):
                    return (a + b - 1) // b

                sfn = ceil_div(n_after_swiglu, sf_vec_size)
                print(f"ref_after_swiglu: {ref_after_swiglu.shape}, {ref_after_swiglu.stride()}")
                # Reshape ref to (l, m, sfn, sf_vec_size)
                ref_for_sf = ref_after_swiglu.permute(2, 0, 1).contiguous()  # (l, m, n)
                # l is involved in valid_m
                ref_for_sf = ref_for_sf.view(1, valid_m, sfn, sf_vec_size)
                # Take abs max over sf_vec_size dimension
                ref_for_sf, _ = torch.abs(ref_for_sf).max(dim=3)  # (l, m, sfn)
                # Multiply by norm_const and rcp_limits
                ref_sfc_f32 = ref_for_sf * norm_const * gemm.get_dtype_rcp_limits(c_dtype)
                # Permute to (m, sfn, l)
                ref_sfc_f32 = ref_sfc_f32.permute(1, 2, 0)

                # Convert fp32 -> f8 -> fp32 for ref_sfc_f32
                ref_sfc_f8_torch = torch.empty(
                    *(1, valid_m, sfn), dtype=torch.uint8, device="cuda"
                ).permute(1, 2, 0)
                ref_sfc_f8 = from_dlpack(ref_sfc_f8_torch, assumed_align=16).mark_layout_dynamic(
                    leading_dim=1
                )
                ref_sfc_f8.element_type = sf_dtype
                ref_sfc_f32_device = ref_sfc_f32.cuda()
                ref_sfc_f32_tensor = from_dlpack(
                    ref_sfc_f32_device, assumed_align=16
                ).mark_layout_dynamic(leading_dim=1)
                # fp32 -> f8
                cute.testing.convert(ref_sfc_f32_tensor, ref_sfc_f8)
                # f8 -> fp32
                cute.testing.convert(ref_sfc_f8, ref_sfc_f32_tensor)
                ref_sfc_f32 = ref_sfc_f32_device.cpu()

                # 2.Convert ref_sfc_f32 to scale factor layout and compare with kernel sfc tensor
                ref_sfc_f32_cute_torch_tensor_cpu, _ = create_sf_layout_tensor(
                    1, valid_m, n_after_swiglu, sf_vec_size
                )
                # convert ref_after_swiglu f32 tensor to cute f32 tensor
                cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
                    from_dlpack(ref_sfc_f32),
                    from_dlpack(ref_sfc_f32_cute_torch_tensor_cpu),
                )
                kernel_sfc_cute_torch_tensor_cpu, _ = create_sf_layout_tensor(
                    1, valid_m, n_after_swiglu, sf_vec_size
                )
                kernel_sfc_tensor, kernel_sfc_torch_tensor = cutlass_torch.cute_tensor_like(
                    kernel_sfc_cute_torch_tensor_cpu,
                    cutlass.Float32,
                    is_dynamic_layout=True,
                    assumed_align=16,
                )
                cute.testing.convert(sfc_tensor, kernel_sfc_tensor)
                kernel_sfc_cute_torch_tensor_cpu = kernel_sfc_torch_tensor.cpu()
                torch.testing.assert_close(
                    kernel_sfc_cute_torch_tensor_cpu,
                    ref_sfc_f32_cute_torch_tensor_cpu,
                    atol=tolerance,
                    rtol=1e-02,
                )
                print("SFC Tensor comparison passed!")

                # 3. Quantized output with scale factor
                # Compute reciprocal of ref_sfc_f32 and multiply by norm_const
                ref_sfc_rcp = norm_const * ref_sfc_f32.reciprocal()
                # Expand the sfn dimension by repeating each value sf_vec_size times
                # ref_sfc_rcp: (m, sfn, l) -> (m, sfn, sf_vec_size, l) -> (m, n, l)
                ref_sfc_rcp_expanded = ref_sfc_rcp.unsqueeze(2).expand(valid_m, sfn, sf_vec_size, 1)
                ref_sfc_rcp_expanded = ref_sfc_rcp_expanded.reshape(valid_m, sfn * sf_vec_size, 1)
                # Trim to exact n dimension if needed
                ref_sfc_rcp_expanded = ref_sfc_rcp_expanded[:, :n_after_swiglu, :]
                # Apply scale to reference output: ref = ref * ref_sfc_rcp
                ref_after_swiglu = torch.einsum(
                    "mnl,mnl->mnl", ref_after_swiglu, ref_sfc_rcp_expanded
                )

            print("Verifying C Tensor...")
            # Convert ref_after_swiglu : f32 -> f8 -> f32
            ref_ = torch.empty(
                *(1, valid_m, n_after_swiglu), dtype=torch.uint8, device="cuda"
            ).permute(1, 2, 0)
            ref_ = from_dlpack(ref_, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            ref_.element_type = c_dtype
            ref_device = ref_after_swiglu.cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_)
            cute.testing.convert(ref_, ref_tensor)
            torch.testing.assert_close(res.cpu(), ref_device.cpu(), atol=tolerance, rtol=1e-02)

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
            num_non_exiting_tiles,
            alpha,
            a_torch_cpu,
            b_torch_cpu,
            c_torch_cpu,
            sfa_torch_cpu,
            sfb_torch_cpu,
            sfc_torch_cpu,
            alpha_torch_cpu,
            a_torch_gpu,
            b_torch_gpu,
            sfa_torch_gpu,
            sfb_torch_gpu,
            sfc_torch_gpu,
            c_torch_gpu,
            norm_const_torch_gpu,
            aligned_group_m_list,
            valid_m,
        ) = create_tensors(
            l,
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
            m_aligned,
            mma_tiler_mn[0],  # cta_tile_m
            permuted_m,
            generate_sfc,
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
            num_non_exiting_tiles,
            alpha,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        # Calculate actual tensor_m used (with padding if permuted_m provided)
        tensor_m = permuted_m if permuted_m is not None else valid_m
        if generate_sfc:
            one_workspace_bytes = (
                a_torch_gpu.numel() * a_torch_gpu.element_size()
                + b_torch_gpu.numel() * b_torch_gpu.element_size()
                + c_torch_gpu.numel() * c_torch_gpu.element_size()
                + sfa_torch_gpu.numel() * sfa_torch_gpu.element_size()
                + sfb_torch_gpu.numel() * sfb_torch_gpu.element_size()
                + sfc_torch_gpu.numel() * sfc_torch_gpu.element_size()
                + norm_const_torch_gpu.numel() * norm_const_torch_gpu.element_size()
                + (tensor_m // mma_tiler_mn[0])
                * 4  # tile_idx_to_expert_idx length (tiles) * sizeof(int32)
                + 1 * 4  # num_non_exiting_tiles (1 element) * sizeof(int32)
                + alpha_torch_cpu.numel() * alpha_torch_cpu.element_size()
            )
        else:
            one_workspace_bytes = (
                a_torch_gpu.numel() * a_torch_gpu.element_size()
                + b_torch_gpu.numel() * b_torch_gpu.element_size()
                + c_torch_gpu.numel() * c_torch_gpu.element_size()
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

    parser = argparse.ArgumentParser(
        description="Example of BlockScaled Contiguous grouped GEMM swiglu fusion kernel on Blackwell."
    )

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
        help="Optional padded M dimension for CUDA graph support. If specified, "
        "A/C matrices and scale factor A will be padded to this size. "
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
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--a_major", choices=["k"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--vectorized_f32", action="store_true", help="Use vectorized f32 operations"
    )
    parser.add_argument("--tolerance", type=float, default=1e-01, help="Tolerance for validation")
    parser.add_argument(
        "--use_cupti",
        action="store_true",
        default=False,
        help="Use CUPTI to measure execution time",
    )
    parser.add_argument("--warmup_iterations", type=int, default=0, help="Warmup iterations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")
    parser.add_argument("--use_cold_l2", action="store_true", default=False, help="Use cold L2")

    args = parser.parse_args()

    # Process arguments to generate nkl and group_m_list
    if args.benchmark:
        # Read from benchmark file
        nkl, group_m_list = read_benchmark_file(args.benchmark)
    else:
        # Use command line arguments
        if len(args.nkl) != 3:
            parser.error("--nkl must contain exactly 3 values")

        n, k, l = args.nkl  # noqa: E741
        nkl = (n, k, l)  # noqa: E741

        # Generate group_m_list based on --custom_mask or --fixed_m
        if args.custom_mask is not None:
            group_m_list = args.custom_mask
            if len(group_m_list) != l:
                parser.error(f"--custom_mask must have exactly {l} values (matching L dimension)")
        elif args.fixed_m is not None:
            group_m_list = tuple([args.fixed_m] * l)
        else:
            # Default: use 128 for all groups
            group_m_list = tuple([128] * l)

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    exec_time = run(
        nkl,
        group_m_list,
        args.ab_dtype,
        args.c_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.vectorized_f32,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        args.permuted_m,
        args.use_cupti,
    )

    print(f"Execution time: {exec_time:.2f} us")
    print("PASS")
