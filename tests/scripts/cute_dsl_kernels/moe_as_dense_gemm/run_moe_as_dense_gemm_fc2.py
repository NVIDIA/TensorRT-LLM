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

"""Example usage of the MoE as Dense GEMM FC2 kernel.

Functional testing:
python run_moe_as_dense_gemm_fc2.py \
        --ab_dtype Float4E2M1FN --c_dtype Float16 \
        --sf_dtype Float8E8M0FNU --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \
        --mnkl 512,256,256,1 --expert_count 256

Perf testing:
python run_moe_as_dense_gemm_fc2.py \
        --ab_dtype Float4E2M1FN --c_dtype Float16 \
        --sf_dtype Float8E8M0FNU --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \
        --mnkl 512,256,256,1 --expert_count 256 \
        --skip_ref_check --use_cold_l2 --warmup_iterations 10 --iterations 50

Note: weight_per_expert is automatically computed as k // expert_count.
      Ensure k is divisible by expert_count.
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
        fc2 as kernel_module,
    )
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).parents[3] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell.moe_as_dense_gemm import fc2 as kernel_module

Sm100BlockScaledPersistentDenseGemmKernel = kernel_module.Sm100BlockScaledPersistentDenseGemmKernel
cvt_sf_MKL_to_M32x4xrm_K4xrk_L = kernel_module.cvt_sf_MKL_to_M32x4xrm_K4xrk_L

# Add parent directory to path to import testing module
sys.path.insert(0, str(Path(__file__).parent.parent))
from testing import benchmark  # noqa: E402


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
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    use_cupti: bool = True,
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

    # Compute weight_per_expert from k and expert_count
    if k % expert_count != 0:
        raise ValueError(f"k ({k}) must be divisible by expert_count ({expert_count})")
    weight_per_expert = k // expert_count

    print("Running Sm100 Persistent Dense BlockScaled GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, SF dtype: {sf_dtype}, SF Vec size: {sf_vec_size}")
    print(f"C dtype: {c_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Expert count: {expert_count}")
    print(f"Weight per expert (k // expert_count): {weight_per_expert}")
    print(f"Use prefetch: {'True' if use_prefetch else 'False'}")
    print(f"Prefetch dist: {prefetch_dist}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")
    print(f"Use CUPTI: {'True' if use_cupti else 'False'}")

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
    a_ref = cutlass_torch.matrix(l, m, k, a_major == "m", cutlass.Float32)
    b_ref = cutlass_torch.matrix(l, n, k, b_major == "n", cutlass.Float32)
    c_ref = cutlass_torch.matrix(l, m, n, c_major == "m", cutlass.Float32)

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

    sfa_ref, sfa_tensor, sfa_torch = create_scale_factor_tensor(l, m, k, sf_vec_size, sf_dtype)
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)

    # Additional Alpha scale tensor
    def create_alpha_scale_tensor(l, m, n, expert_count, dtype):  # noqa: E741
        ### if not swapAB
        # True means alpha_scale is expert_count major.
        alpha_scale_ref = cutlass_torch.matrix(
            l,
            m,
            expert_count,
            False,  # expert_count is major
            cutlass.Float32,
            # Debug alpha scale data
            # init_type=cutlass_torch.TensorInitType.SCALAR,
            # init_config=cutlass_torch.ScalarInitConfig(value=1),
        )
        # currently we assume alpha_scale is n major, which means token major and 4B aligned, should be 4B aligned.
        alpha_scale_tensor, alpha_scale_torch = cutlass_torch.cute_tensor_like(
            alpha_scale_ref, cutlass.Float32, is_dynamic_layout=True, assumed_align=4
        )
        # print(f"alpha_scale_ref: {alpha_scale_ref.shape}")
        # reshape makes memory contiguous for reference check.
        alpha_scale_ref = (
            alpha_scale_ref.permute(2, 0, 1)
            .unsqueeze(-1)
            .expand(l, m, expert_count, weight_per_expert)
            .reshape(l, m, expert_count * weight_per_expert)
            .permute((1, 2, 0))
        )

        return alpha_scale_ref, alpha_scale_tensor, alpha_scale_torch

    alpha_scale_ref, alpha_scale_tensor, alpha_scale_torch = create_alpha_scale_tensor(
        l, m, n, expert_count, cutlass.Float32
    )

    # Configure gemm kernel
    gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        expert_count,
        weight_per_expert,
        use_prefetch,
        prefetch_dist,
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
        c_tensor,
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
            c_tensor,
            current_stream,
        )
        print("Verifying results...")
        res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
        res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
        # Final reference result is the product of res_a, res_b and alpha_scale_ref.
        ref = torch.einsum("mkl,nkl,mkl->mnl", res_a, res_b, alpha_scale_ref)

        # Convert c back to f32 for comparison.
        c_ref_device = c_ref.cuda()
        cute.testing.convert(
            c_tensor,
            from_dlpack(c_ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if c_major == "n" else 0)
            ),
        )
        c_ref = c_ref_device.cpu()

        if c_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)
        elif c_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN):
            # Convert ref : f32 -> f8 -> f32
            ref_f8_ = torch.empty(*(l, m, n), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
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
            # Convert ref : f32 -> f4 -> f32
            ref_f4_ = torch.empty(*(l, m, n), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
            ref_f4 = from_dlpack(ref_f4_, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            ref_f4.element_type = c_dtype
            ref_device = ref.permute(2, 0, 1).contiguous().permute(1, 2, 0).cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_f4)
            cute.testing.convert(ref_f4, ref_tensor)
            ref = ref_device.cpu()
            torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)
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
        _, alpha_scale_tensor, _ = create_alpha_scale_tensor(l, m, n, expert_count, cutlass.Float32)
        return cute.testing.JitArguments(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            alpha_scale_tensor,
            c_tensor,
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
            + c_torch.numel() * c_torch.element_size()
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

    parser = argparse.ArgumentParser(
        description="Example of Sm100 Dense Persistent BlockScaled GEMM."
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(512, 256, 256, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--expert_count",
        type=int,
        default=256,
        help="Expert count (k must be divisible by expert_count)",
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
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E8M0FNU)
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
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        args.use_cupti,
    )
    print(f"Execution time: {exec_time:.2f} us")
    print("PASS")
