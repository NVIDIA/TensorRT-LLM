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

"""Example usage of the MoE as Dense GEMM FC1 kernel.

Functional testing:
python run_moe_as_dense_gemm_fc1.py \
        --ab_dtype Float4E2M1FN --c_dtype Float16 \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \
        --mnkl 128,65536,256,1 --expert_count 256

Perf testing:
python run_moe_as_dense_gemm_fc1.py \
        --ab_dtype Float4E2M1FN --c_dtype Float16 \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \
        --mnkl 128,65536,256,1 --expert_count 256 \
        --skip_ref_check --use_cold_l2 --warmup_iterations 10 --iterations 50
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

# Import kernel module
try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.moe_as_dense_gemm import (
        fc1 as kernel_module,
    )
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).parents[3] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell.moe_as_dense_gemm import fc1 as kernel_module

Sm100BlockScaledPersistentDenseGemmKernel = kernel_module.Sm100BlockScaledPersistentDenseGemmKernel
cvt_sf_MKL_to_M32x4xrm_K4xrk_L = kernel_module.cvt_sf_MKL_to_M32x4xrm_K4xrk_L
cvt_sf_M32x4xrm_K4xrk_L_to_MKL = kernel_module.cvt_sf_M32x4xrm_K4xrk_L_to_MKL

# Add parent directory to path to import testing module
sys.path.insert(0, str(Path(__file__).parent.parent))
from testing import benchmark  # noqa: E402


def create_sf_layout_tensor(l, mn, k, sf_vec_size):  # noqa: E741
    """Create scale factor tensor in MMA layout for SFC verification.

    :param l: Batch dimension (L)
    :param mn: M or N dimension
    :param k: K dimension (for SFC, this is n_out // sf_vec_size)
    :param sf_vec_size: Vector size for scale factor
    :return: Tuple of (swizzled_tensor_cpu, ref_shape)
    """

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

    mma_permute_order = (3, 4, 1, 5, 2, 0)

    # Create f32 cute torch tensor (cpu) with MMA layout
    cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.SCALAR,
        init_config=cutlass_torch.ScalarInitConfig(value=0.0),
    )

    return cute_f32_torch_tensor_cpu, ref_shape


def run(
    mnkl: Tuple[int, int, int, int],
    expert_count: int,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    use_prefetch: bool = False,
    prefetch_dist: int = 3,
    vectorized_f32: bool = True,
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    use_cupti: bool = True,
    no_alpha_post: bool = False,
    **kwargs,
):
    """Execute a persistent batched dense blockscaled GEMM operation on Blackwell architecture.

    This function prepares input tensors, configures and launches the persistent GEMM kernel,
    optionally performs reference validation, and benchmarks the execution performance.

    :param mnkl: Problem size (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param ab_dtype: Data type for input tensors A and B
    :type ab_dtype: Type[cutlass.Numeric]
    :param sf_dtype: Data type for scale factor tensor
    :type sf_dtype: Type[cutlass.Numeric]
    :param sf_vec_size: Vector size for scale factor tensor
    :type sf_vec_size: int
    :param c_dtype: Data type for output tensor C
    :type c_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/c_major: Memory layout of tensor A/B/C
    :type a_major/b_major/c_major: str
    :param mma_tiler_mn: MMA tiling size.
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster shape.
    :type cluster_shape_mn: Tuple[int, int]
    :param tolerance: Tolerance value for reference validation comparison, defaults to 1e-01
    :type tolerance: float, optional
    :param warmup_iterations: Number of warmup iterations before benchmarking, defaults to 0
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations to run, defaults to 1
    :type iterations: int, optional
    :param skip_ref_check: Whether to skip reference result validation, defaults to False
    :type skip_ref_check: bool, optional
    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache, defaults to False
    :type use_cold_l2: bool, optional
    :raises RuntimeError: If CUDA GPU is not available
    :raises ValueError: If the configuration is invalid or unsupported by the kernel
    :return: Execution time of the GEMM kernel
    :rtype: float
    """
    # Unpack parameters
    m, n, k, l = mnkl  # noqa: E741

    # Compute weight_per_expert from n and expert_count
    weight_per_expert = n // expert_count

    print("Running Sm100 Persistent Dense BlockScaled GEMM with SwiGLU fusion test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, SF dtype: {sf_dtype}, SF Vec size: {sf_vec_size}")
    print(f"C dtype: {c_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Expert count: {expert_count}")
    print(f"Weight per expert (n // expert_count): {weight_per_expert}")
    print(f"Use prefetch: {'True' if use_prefetch else 'False'}")
    print(f"Prefetch dist: {prefetch_dist}")
    print(f"Vectorized f32: {'True' if vectorized_f32 else 'False'}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")
    print(f"Use CUPTI: {'True' if use_cupti else 'False'}")
    print(f"No alpha post: {'True' if no_alpha_post else 'False'}")

    # Skip unsupported testcase
    if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
        expert_count,
        weight_per_expert,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype}, "
            f"{mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, "
            f"{a_major}, {b_major}, {c_major}, {expert_count}, {weight_per_expert}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create tensor A/B/C
    # Note: C has n//2 columns due to SwiGLU fusion (pairs of up/gate columns are combined)
    a_ref = cutlass_torch.matrix(l, m, k, a_major == "m", cutlass.Float32)
    b_ref = cutlass_torch.matrix(l, n, k, b_major == "n", cutlass.Float32)
    c_ref = cutlass_torch.matrix(l, m, n // 2, c_major == "m", cutlass.Float32)

    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
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
        mode=1 if c_major == "n" else 0,
        stride_order=(2, 0, 1) if c_major == "n" else (2, 1, 0),
        divisibility=32 if ab_dtype == cutlass.Float4E2M1FN else 16,
    )

    # Create scale factor tensor SFA/SFB
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
                max_val=2,  # Reduced from 3 to 2 to reduce overflow risk
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

    sfa_ref, sfa_tensor, sfa_torch = create_scale_factor_tensor(l, m, k, sf_vec_size, sf_dtype)
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)

    # Create scale factor tensor alpha_scale_tensor
    # Alpha scale is indexed by N dimension (not M) since we don't swap A/B
    def create_alpha_scale_tensor(l, expert_count, weight_per_expert):  # noqa: E741
        ref_shape = (l, expert_count)
        ref_permute_order = (1, 0)

        # Create f32 ref torch tensor (cpu)
        ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            ref_shape,
            torch.float32,
            permute_order=ref_permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=-2,
                max_val=2,
            ),
            # init_type=cutlass_torch.TensorInitType.SCALAR,
            # init_config=cutlass_torch.ScalarInitConfig(
            #     value=1.0,
            # ),
        )

        # Create cute alpha_scale_tensor
        cute_alpha_scale_tensor, cute_alpha_scale_torch_tensor = cutlass_torch.cute_tensor_like(
            ref_f32_torch_tensor_cpu,
            cutlass.Float32,
            is_dynamic_layout=True,
            assumed_align=4,
        )

        # Expand to (n, l) for einsum "mnl,nl->mnl" (alpha indexed by N dimension)
        # n = expert_count * weight_per_expert
        ref_f32_torch_tensor_cpu = (
            ref_f32_torch_tensor_cpu.permute(1, 0)
            .unsqueeze(-1)
            .expand(l, expert_count, weight_per_expert)
            .reshape(l, expert_count * weight_per_expert)
            .permute(*ref_permute_order)
        )

        return (
            ref_f32_torch_tensor_cpu,
            cute_alpha_scale_tensor,
            cute_alpha_scale_torch_tensor,
        )

    alpha_scale_ref, alpha_scale_tensor, alpha_scale_torch = create_alpha_scale_tensor(
        l, expert_count, weight_per_expert
    )

    # Create post-SwiGLU alpha scale tensor
    # Shape: (m, expert_count, l) for per-token per-expert scaling
    def create_alpha_scale_post_swiglu_tensor(l, m, expert_count, weight_per_expert):  # noqa: E741
        """Create alpha scale tensor to apply after SwiGLU.

        Post-SwiGLU alpha has shape (m, expert_count, l) for per-token per-expert scaling.
        This provides fine-grained control over each token's contribution to each expert.

        Args:
            l: batch size
            m: sequence length (number of tokens)
            expert_count: number of experts
            weight_per_expert: weights per expert in input (before SwiGLU)

        Returns:
            tuple: (ref_tensor_expanded, cute_tensor, torch_tensor)
                - ref_tensor_expanded: (m, n_out, l) for reference check
                - cute_tensor: (m, expert_count, l) for kernel input
                - torch_tensor: backing torch tensor
        """
        weight_per_expert_out = weight_per_expert // 2
        n_out = expert_count * weight_per_expert_out

        # Create tensor with shape (m, expert_count, l) in contiguous memory order
        # This matches the convention of other tensors (L as last dimension)
        ref_shape = (m, expert_count, l)

        # Create f32 tensor directly without permutation
        # Use positive range [0.8, 1.2] to minimize numerical variation while still testing functionality
        ref_f32_torch_tensor_cpu = torch.rand(ref_shape, dtype=torch.float32) * 0.4 + 0.8

        # Create cute alpha_scale_tensor with shape (m, expert_count, l)
        cute_alpha_scale_tensor, cute_alpha_scale_torch_tensor = cutlass_torch.cute_tensor_like(
            ref_f32_torch_tensor_cpu,
            cutlass.Float32,
            is_dynamic_layout=True,
            assumed_align=4,
        )

        # Expand to (m, n_out, l) for reference computation
        # Each expert's alpha (per token) is repeated for weight_per_expert_out elements
        # Input: (m, expert_count, l)
        # Need to expand to (m, n_out, l) where n_out = expert_count * weight_per_expert_out
        ref_expanded = (
            ref_f32_torch_tensor_cpu.unsqueeze(2)  # (m, expert_count, 1, l)
            .expand(m, expert_count, weight_per_expert_out, l)  # (m, expert_count, wpe_out, l)
            .reshape(m, n_out, l)  # (m, n_out, l)
        )

        return (
            ref_expanded,
            cute_alpha_scale_tensor,
            cute_alpha_scale_torch_tensor,
        )

    # Create alpha_scale_post tensor (optional, disabled when no_alpha_post=True)
    alpha_scale_post_ref = None
    alpha_scale_post_tensor = None
    alpha_scale_post_torch = None
    if not no_alpha_post:
        alpha_scale_post_ref, alpha_scale_post_tensor, alpha_scale_post_torch = (
            create_alpha_scale_post_swiglu_tensor(l, m, expert_count, weight_per_expert)
        )

        # Debug: print alpha statistics
        print(f"Alpha post shape: {alpha_scale_post_torch.shape}")
        print(
            f"Alpha post min: {alpha_scale_post_torch.min().item():.4f}, max: {alpha_scale_post_torch.max().item():.4f}"
        )
        print(f"Alpha post mean: {alpha_scale_post_torch.mean().item():.4f}")
        print(f"Alpha post with abs < 0.1: {(alpha_scale_post_torch.abs() < 0.1).sum().item()}")
    else:
        print("Alpha post: disabled (no_alpha_post=True)")

    # Create SFC tensor and norm_const tensor for FP4 quantized output
    # SFC (Scale Factor C) is used to store per-block scale factors for FP4 quantization
    generate_sfc = c_dtype is cutlass.Float4E2M1FN
    sfc_tensor = None
    sfc_torch = None
    norm_const_tensor = None
    norm_const = 1.0

    print(f"FP4 quantization with SFC: {'True' if generate_sfc else 'False'}")

    if generate_sfc:
        # SFC shape: (m, n_out // sf_vec_size, l) where n_out = n // 2 due to SwiGLU
        # SFC tensor needs to be created with swizzled MMA layout (same as SFA/SFB)
        n_out = n // 2

        # Create SFC tensor with swizzled MMA layout using create_sf_layout_tensor
        # This creates tensor with ref_shape (l, m, sfc_n) in MMA swizzled layout
        sfc_swizzled_cpu, _ = create_sf_layout_tensor(l, m, n_out, sf_vec_size)
        sfc_tensor, sfc_torch = cutlass_torch.cute_tensor_like(
            sfc_swizzled_cpu, sf_dtype, is_dynamic_layout=True, assumed_align=16
        )

        # Create norm_const tensor (scalar tensor containing normalization constant)
        norm_const_ref = torch.tensor([norm_const], dtype=torch.float32)
        norm_const_tensor, _ = cutlass_torch.cute_tensor_like(
            norm_const_ref, cutlass.Float32, is_dynamic_layout=True, assumed_align=4
        )

    # Configure gemm kernel with SwiGLU fusion
    gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        weight_per_expert,
        use_prefetch,
        prefetch_dist,
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
        sfa_tensor,
        sfb_tensor,
        alpha_scale_tensor,
        alpha_scale_post_tensor,  # Post-SwiGLU alpha: (l, m, expert_count)
        c_tensor,
        sfc_tensor,  # SFC tensor for FP4 quantized output (None if not quantizing)
        norm_const_tensor,  # Normalization constant tensor (None if not quantizing)
        max_active_clusters,
        current_stream,
    )

    # Compute reference result
    if not skip_ref_check:
        # Execute kernel once for reference checking
        compiled_gemm(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            alpha_scale_tensor,
            alpha_scale_post_tensor,  # Post-SwiGLU alpha: (l, m, expert_count)
            c_tensor,
            sfc_tensor,  # SFC tensor for FP4 quantized output
            norm_const_tensor,  # Normalization constant tensor
            current_stream,
        )
        print("Verifying results...")
        res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
        res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
        ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)
        # Alpha scale indexed by N dimension (not M) since we don't swap A/B
        ref = torch.einsum("mnl,nl->mnl", ref, alpha_scale_ref)

        # Apply SwiGLU fusion: output = up * silu(gate)
        # up and gate are interleaved in the N dimension (granularity=64)
        # Extract up (even subtiles) and gate (odd subtiles)
        # ref shape is (m, n, l), we need to reshape to extract up and gate
        # Assuming interleaving at granularity 64
        granularity = 64
        ref_reshaped = ref.view(m, n // granularity, granularity, l)
        ref_up = ref_reshaped[:, 0::2, :, :].contiguous().view(m, n // 2, l)
        ref_gate = ref_reshaped[:, 1::2, :, :].contiguous().view(m, n // 2, l)

        # silu(x) = x * sigmoid(x)
        ref = ref_up * (ref_gate * torch.sigmoid(ref_gate))
        # Now ref shape: (m, n_out, l) where n_out = n // 2

        # Apply post-SwiGLU alpha scale (optional)
        # Alpha is per-token per-expert: (m, n_out, l) * (m, n_out, l)
        if alpha_scale_post_ref is not None:
            ref = ref * alpha_scale_post_ref

        # Convert c back to f32 for comparison.
        c_ref_device = c_ref.cuda()
        cute.testing.convert(
            c_tensor,
            from_dlpack(c_ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if c_major == "n" else 0)
            ),
        )
        c_ref = c_ref_device.cpu()

        # Note: n_out = n // 2 due to SwiGLU fusion
        n_out = n // 2
        if c_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)
        elif c_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN):
            # Convert ref : f32 -> f8 -> f32
            ref_f8_ = torch.empty(*(l, m, n_out), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
            ref_f8 = from_dlpack(ref_f8_, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            ref_f8.element_type = c_dtype
            ref_device = ref.permute(2, 0, 1).contiguous().permute(1, 2, 0).cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_f8)
            cute.testing.convert(ref_f8, ref_tensor)
            ref = ref_device.cpu()
            torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)
        # {$nv-internal-release begin}
        elif c_dtype is cutlass.Float4E2M1FN:
            # FP4 quantization with SFC (Scale Factor C) verification
            # Reference: run_blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py

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
                m_dim, n_dim, ng = tensor_f32.shape
                # Create properly packed nvfp4 tensor using cutlass_torch utilities
                ref_f32_torch = cutlass_torch.matrix(ng, m_dim, n_dim, False, cutlass.Float32)
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
                sf_vec_size_local: int,
                norm_const_local: float,
                rcp_limits: float,
            ) -> torch.Tensor:
                """Compute scale factor for nvfp4 quantization.

                Scale factor = abs_max_per_vector * norm_const * rcp_limits

                :param tensor_f32: Input fp32 tensor, shape (m, n, ng)
                :param sf_vec_size_local: Vector size for scale factor (e.g., 16)
                :param norm_const_local: Normalization constant
                :param rcp_limits: Reciprocal of dtype max value (e.g., 1/6.0 for nvfp4)
                :return: Scale factor tensor, shape (m, sfn, ng) where sfn = ceil(n / sf_vec_size)
                """
                m_dim, n_dim, ng = tensor_f32.shape
                sfn = ceil_div(n_dim, sf_vec_size_local)
                # Reshape to (m, sfn, sf_vec_size, ng) for abs max computation
                # Pad n dimension if needed
                padded_n = sfn * sf_vec_size_local
                if padded_n > n_dim:
                    tensor_padded = torch.zeros(m_dim, padded_n, ng, dtype=tensor_f32.dtype)
                    tensor_padded[:, :n_dim, :] = tensor_f32
                else:
                    tensor_padded = tensor_f32
                tensor_reshaped = tensor_padded.view(m_dim, sfn, sf_vec_size_local, ng)
                # Compute abs max over sf_vec_size dimension
                abs_max, _ = torch.abs(tensor_reshaped).max(dim=2)  # (m, sfn, ng)
                # Compute scale factor
                scale_factor = abs_max * norm_const_local * rcp_limits
                return scale_factor

            def apply_quantization_scale(
                tensor_f32: torch.Tensor,
                scale_factor: torch.Tensor,
                sf_vec_size_local: int,
                norm_const_local: float,
            ) -> torch.Tensor:
                """Apply quantization scale to tensor.

                Output = tensor * (norm_const / scale_factor).
                This simulates the kernel's quantization scaling.

                :param tensor_f32: Input fp32 tensor, shape (m, n, ng)
                :param scale_factor: Scale factor tensor, shape (m, sfn, ng)
                :param sf_vec_size_local: Vector size for scale factor
                :param norm_const_local: Normalization constant
                :return: Scaled tensor, shape (m, n, ng)
                """
                m_dim, n_dim, ng = tensor_f32.shape
                sfn = scale_factor.shape[1]
                # Compute reciprocal scale, clamping inf to fp32_max (matching kernel fmin behavior)
                fp32_max = torch.tensor(3.40282346638528859812e38, dtype=torch.float32)
                scale_rcp = norm_const_local * scale_factor.reciprocal()
                scale_rcp = torch.where(torch.isinf(scale_rcp), fp32_max, scale_rcp)
                # Expand scale factor to match tensor dimensions
                # (m, sfn, ng) -> (m, sfn, sf_vec_size, ng) -> (m, sfn * sf_vec_size, ng)
                scale_rcp_expanded = scale_rcp.unsqueeze(2).expand(
                    m_dim, sfn, sf_vec_size_local, ng
                )
                scale_rcp_expanded = scale_rcp_expanded.reshape(m_dim, sfn * sf_vec_size_local, ng)
                # Trim to exact n dimension
                scale_rcp_expanded = scale_rcp_expanded[:, :n_dim, :]
                # Apply scale
                return tensor_f32 * scale_rcp_expanded

            def unswizzle_kernel_sfc(
                sfc_cute_tensor,
                m_dim: int,
                n_dim: int,
                sf_vec_size_local: int,
                l_dim: int,
            ) -> torch.Tensor:
                """Unswizzle kernel's scale factor tensor from MMA layout to MKL layout.

                :param sfc_cute_tensor: Kernel's scale factor cute tensor (swizzled MMA layout)
                :param m_dim: M dimension
                :param n_dim: Output N dimension (n_out)
                :param sf_vec_size_local: Vector size for scale factor
                :param l_dim: L dimension (batch)
                :return: Unswizzled scale factor tensor, shape (m, sfn, l)
                """
                sfn = ceil_div(n_dim, sf_vec_size_local)

                # Create swizzled layout tensor matching kernel's SFC layout
                swizzled_sfc_cpu, _ = create_sf_layout_tensor(
                    l_dim, m_dim, n_dim, sf_vec_size_local
                )
                swizzled_sfc_tensor, swizzled_sfc_torch = cutlass_torch.cute_tensor_like(
                    swizzled_sfc_cpu, cutlass.Float32, is_dynamic_layout=True, assumed_align=16
                )

                # Copy kernel SFC (sf_dtype) to swizzled layout tensor (Float32)
                cute.testing.convert(sfc_cute_tensor, swizzled_sfc_tensor)
                swizzled_sfc_cpu = swizzled_sfc_torch.cpu()

                # Unswizzle: MMA layout -> MKL layout (m, sfn, l)
                unswizzled_sfc = torch.empty(m_dim, sfn, l_dim, dtype=torch.float32)
                cvt_sf_M32x4xrm_K4xrk_L_to_MKL(
                    from_dlpack(swizzled_sfc_cpu),
                    from_dlpack(unswizzled_sfc),
                )

                return unswizzled_sfc

            # ============================================================
            # Step 1: Compute reference scale factor (SFC) from SwiGLU output
            # ============================================================
            rcp_limits = gemm.get_dtype_rcp_limits(c_dtype)

            # Compute reference SFC: abs_max * norm_const * rcp_limits
            ref_sfc_before_f8 = compute_scale_factor(ref, sf_vec_size, norm_const, rcp_limits)
            # Simulate f8 quantization for SFC (kernel stores SFC in sf_dtype format)
            ref_sfc_f32 = simulate_f8_quantization(ref_sfc_before_f8, sf_dtype)

            # ============================================================
            # Step 2: Verify kernel SFC matches reference SFC (using pass rate)
            # ============================================================
            if sfc_tensor is not None:
                kernel_sfc = unswizzle_kernel_sfc(sfc_tensor, m, n_out, sf_vec_size, l)

                sfc_diff = torch.abs(ref_sfc_f32 - kernel_sfc)
                sfc_within_tolerance = (sfc_diff <= tolerance) | (
                    sfc_diff <= torch.abs(ref_sfc_f32) * 1e-02
                )
                sfc_pass_rate = sfc_within_tolerance.float().mean().item()
                print(f"SFC Tensor pass rate: {sfc_pass_rate * 100:.2f}%")

            # ============================================================
            # Step 3: Apply quantization scale and simulate nvfp4 precision loss
            # ============================================================
            # Apply scale: ref_scaled = ref * (norm_const / sfc)
            ref_scaled = apply_quantization_scale(ref, ref_sfc_f32, sf_vec_size, norm_const)
            # Simulate nvfp4 quantization: f32 -> nvfp4 -> f32
            ref_quantized = simulate_nvfp4_quantization(ref_scaled)

            # ============================================================
            # Step 4: Compare kernel output with reference (using pass rate)
            # ============================================================
            print("Verifying C Tensor...")
            diff = torch.abs(c_ref - ref_quantized)
            within_tolerance = (diff <= tolerance) | (diff <= torch.abs(ref_quantized) * 1e-02)
            pass_rate = within_tolerance.float().mean().item()
            print(f"C Tensor pass rate: {pass_rate * 100:.2f}% (threshold: 95%)")
            assert pass_rate >= 0.95, (
                f"Only {pass_rate * 100:.2f}% elements within tolerance, expected >= 95%"
            )
        # {$nv-internal-release end}

    def generate_tensors():
        a_tensor, _ = cutlass_torch.cute_tensor_like(
            a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_tensor, _ = cutlass_torch.cute_tensor_like(
            b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, _ = cutlass_torch.cute_tensor_like(
            c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
        )

        # Mark tensor to be byte aligned
        a_tensor.mark_compact_shape_dynamic(
            mode=1 if a_major == "k" else 0,
            stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
            divisibility=2 if ab_dtype == cutlass.Float4E2M1FN else 1,
        )
        b_tensor.mark_compact_shape_dynamic(
            mode=1 if b_major == "k" else 0,
            stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
            divisibility=2 if ab_dtype == cutlass.Float4E2M1FN else 1,
        )
        c_tensor.mark_compact_shape_dynamic(
            mode=1 if c_major == "n" else 0,
            stride_order=(2, 0, 1) if c_major == "n" else (2, 1, 0),
            divisibility=2 if c_dtype == cutlass.Float4E2M1FN else 1,
        )

        _, sfa_tensor, _ = create_scale_factor_tensor(l, m, k, sf_vec_size, sf_dtype)
        _, sfb_tensor, _ = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)
        _, alpha_scale_tensor, _ = create_alpha_scale_tensor(l, expert_count, weight_per_expert)
        gen_alpha_scale_post_tensor = None
        if not no_alpha_post:
            _, gen_alpha_scale_post_tensor, _ = create_alpha_scale_post_swiglu_tensor(
                l, m, expert_count, weight_per_expert
            )

        # Create SFC tensor and norm_const tensor for FP4 quantized output
        gen_sfc_tensor = None
        gen_norm_const_tensor = None
        if generate_sfc:
            n_out = n // 2
            # Create SFC tensor with swizzled MMA layout
            sfc_swizzled_cpu_gen, _ = create_sf_layout_tensor(l, m, n_out, sf_vec_size)
            gen_sfc_tensor, _ = cutlass_torch.cute_tensor_like(
                sfc_swizzled_cpu_gen, sf_dtype, is_dynamic_layout=True, assumed_align=16
            )
            norm_const_ref_gen = torch.tensor([norm_const], dtype=torch.float32)
            gen_norm_const_tensor, _ = cutlass_torch.cute_tensor_like(
                norm_const_ref_gen, cutlass.Float32, is_dynamic_layout=True, assumed_align=4
            )

        return cute.testing.JitArguments(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            alpha_scale_tensor,
            gen_alpha_scale_post_tensor,  # Post-SwiGLU alpha: (l, m, expert_count) or None
            c_tensor,
            gen_sfc_tensor,  # SFC tensor for FP4 quantized output
            gen_norm_const_tensor,  # Normalization constant tensor
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch.numel() * a_torch.element_size()
            + b_torch.numel() * b_torch.element_size()
            + sfa_torch.numel() * sfa_torch.element_size()
            + sfb_torch.numel() * sfb_torch.element_size()
            + alpha_scale_torch.numel() * alpha_scale_torch.element_size()
            + (
                alpha_scale_post_torch.numel() * alpha_scale_post_torch.element_size()
                if alpha_scale_post_torch is not None
                else 0
            )
            + c_torch.numel() * c_torch.element_size()
        )
        if sfc_torch is not None:
            one_workspace_bytes += sfc_torch.numel() * sfc_torch.element_size()
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

    parser = argparse.ArgumentParser(
        description="Example of Sm100 Dense Persistent BlockScaled GEMM with SwiGLU fusion."
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(512, 256, 256, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--expert_count",  # should be 256 or 257 normally
        type=int,
        default=257,
        help="expert count",
    )
    parser.add_argument(
        "--use_prefetch",
        action="store_true",
        default=False,
        help="Enable prefetch operations (default: False)",
    )
    parser.add_argument(
        "--prefetch_dist",
        type=int,
        default=7,
        help="Prefetch distance for TMA operations (default: 3)",
    )
    parser.add_argument(
        "--vectorized_f32",
        action="store_true",
        default=True,
        help="Enable vectorized f32x2 operations for better performance (default: True)",
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
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--tolerance", type=float, default=1e-01, help="Tolerance for validation")
    parser.add_argument("--warmup_iterations", type=int, default=10, help="Warmup iterations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )
    parser.add_argument(
        "--use_cupti",
        action="store_true",
        default=True,
        help="Use CUPTI for profiling (default: True)",
    )
    parser.add_argument(
        "--no_alpha_post",
        action="store_true",
        default=False,
        help="Disable post-SwiGLU alpha scaling (alpha_post=None)",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    exec_time = run(
        args.mnkl,
        args.expert_count,
        args.ab_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.c_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.use_prefetch,
        args.prefetch_dist,
        args.vectorized_f32,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        args.use_cupti,
        args.no_alpha_post,
    )
    print(f"Execution time: {exec_time:.2f} us")
    print("PASS")
