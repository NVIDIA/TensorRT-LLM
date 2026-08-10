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
        dense_gemm_persistent as kernel_module,
    )
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).parents[3] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell import dense_gemm_persistent as kernel_module

PersistentDenseGemmKernel = kernel_module.PersistentDenseGemmKernel

try:
    from .testing import benchmark
except ImportError:
    from testing import benchmark


# Map string dtype names to cutlass and torch types
_CUTLASS_DTYPE_MAP = {
    "bf16": cutlass.BFloat16,
    "fp16": cutlass.Float16,
    "fp32": cutlass.Float32,
}

_TORCH_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    use_2cta_instrs: bool,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float = 1e-02,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    use_cupti: bool = False,
    use_strided: bool = False,
    **kwargs,
):
    """Runs and benchmarks the persistent dense BF16/FP16 GEMM/BMM.

    This function prepares input tensors, launches the kernel, optionally
    validates the results against a reference implementation, and measures
    performance.

    Args:
        mnkl (Tuple[int, int, int, int]): The dimensions (M, N, K, L) of the
            GEMM problem, where L is the batch size.
        ab_dtype (Type[cutlass.Numeric]): Data type for input tensors A and B
            (BFloat16 or Float16).
        c_dtype (Type[cutlass.Numeric]): Data type for the output tensor C
            (BFloat16 or Float32).
        use_2cta_instrs (bool): Whether to use 2-CTA instructions.
        mma_tiler_mn (Tuple[int, int]): The shape of the MMA tile.
        cluster_shape_mn (Tuple[int, int]): The shape of the CTA cluster.
        tolerance (float, optional): Tolerance for result validation.
            Defaults to 1e-02.
        warmup_iterations (int, optional): Number of warmup runs. Defaults to 0.
        iterations (int, optional): Number of benchmark iterations.
            Defaults to 1.
        skip_ref_check (bool, optional): If True, skips result validation.
            Defaults to False.
        use_cold_l2 (bool, optional): If True, uses a circular buffer to
            ensure a cold L2 cache. Defaults to False.
        use_cupti (bool, optional): If True, uses CUPTI to measure execution time.
            Defaults to False.
        use_strided (bool, optional): If True, uses wrapper_strided (BMM mode)
            instead of wrapper (contiguous mode). Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns:
        float: The execution time of the kernel in microseconds.

    Raises:
        RuntimeError: If no CUDA-capable GPU is available.
        TypeError: If the configuration is not supported by the kernel.
    """
    acc_dtype = cutlass.Float32

    print("Running Sm100 Persistent Dense BF16/FP16 GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, Acc dtype: {acc_dtype}")
    print(f"C dtype: {c_dtype}")
    print(f"Use 2-CTA instructions: {use_2cta_instrs}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")
    print(f"Use CUPTI: {'True' if use_cupti else 'False'}")
    print(f"Use strided (BMM): {'True' if use_strided else 'False'}")

    # Unpack parameters
    m, n, k, batch = mnkl

    # A layout: (M, K, batch) with K innermost
    a_major = "k"
    # B layout: (N, K, batch) with K innermost
    b_major = "k"
    # C layout: (M, N, batch)
    c_major = "n"

    # Skip unsupported testcase
    if not PersistentDenseGemmKernel.can_implement(
        ab_dtype,
        acc_dtype,
        c_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        batch,
        a_major,
        b_major,
        c_major,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {acc_dtype}, {c_dtype}, "
            f"use_2cta={use_2cta_instrs}, "
            f"{mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {batch}, "
            f"{a_major}, {b_major}, {c_major}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Determine torch dtype for A/B
    if ab_dtype == cutlass.BFloat16:
        _ab_torch_dtype = torch.bfloat16  # noqa: F841
    elif ab_dtype == cutlass.Float16:
        _ab_torch_dtype = torch.float16  # noqa: F841
    else:
        raise ValueError(f"Unsupported ab_dtype: {ab_dtype}")

    # Determine torch dtype for C
    if c_dtype == cutlass.BFloat16:
        _c_torch_dtype = torch.bfloat16  # noqa: F841
    elif c_dtype == cutlass.Float32:
        _c_torch_dtype = torch.float32  # noqa: F841
    elif c_dtype == cutlass.Float16:
        _c_torch_dtype = torch.float16  # noqa: F841
    else:
        raise ValueError(f"Unsupported c_dtype: {c_dtype}")

    # Create tensors:
    # A: (M, K, batch) with K innermost -> column-major in (M, K) sense
    # B: (N, K, batch) with K innermost -> column-major in (N, K) sense
    # C: (M, N, batch)
    a_ref = cutlass_torch.matrix(batch, m, k, a_major == "m", cutlass.Float32)
    b_ref = cutlass_torch.matrix(batch, n, k, b_major == "n", cutlass.Float32)
    c_ref = cutlass_torch.matrix(batch, m, n, c_major == "m", cutlass.Float32)

    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    # Mark tensor to be byte aligned
    a_tensor.mark_compact_shape_dynamic(
        mode=1 if a_major == "k" else 0,
        stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
        divisibility=1,
    )
    b_tensor.mark_compact_shape_dynamic(
        mode=1 if b_major == "k" else 0,
        stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
        divisibility=1,
    )
    c_tensor.mark_compact_shape_dynamic(
        mode=1 if c_major == "n" else 0,
        stride_order=(2, 0, 1) if c_major == "n" else (2, 1, 0),
        divisibility=1,
    )

    # Configure gemm kernel
    gemm = PersistentDenseGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store=True,
    )

    # Compute max active clusters on current device
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # Initialize Stream
    current_stream = cutlass_torch.default_stream()

    if use_strided:
        # Strided BMM mode: need a_stride_m and a_stride_batch
        # A is (M, K, batch) with K innermost, so stride along M dimension
        # and stride along batch dimension are needed.
        # For a contiguous (M, K, batch) tensor with K innermost:
        #   stride_k = 1, stride_m = K, stride_batch = M*K
        a_stride_m = k
        a_stride_batch = m * k

        # Compile gemm kernel for strided mode
        compiled_gemm = cute.compile(
            gemm.wrapper_strided,
            m,
            n,
            k,
            batch,
            a_tensor,
            b_tensor,
            c_tensor,
            a_stride_m,
            a_stride_batch,
            max_active_clusters,
            current_stream,
            options="--opt-level 2",
        )

        def run_kernel(a_t, b_t, c_t, stream):
            compiled_gemm(
                m,
                n,
                k,
                batch,
                a_t,
                b_t,
                c_t,
                a_stride_m,
                a_stride_batch,
                max_active_clusters,
                stream,
            )
    else:
        # Contiguous mode
        compiled_gemm = cute.compile(
            gemm.wrapper,
            m,
            n,
            k,
            batch,
            a_tensor,
            b_tensor,
            c_tensor,
            max_active_clusters,
            current_stream,
            options="--opt-level 2",
        )

        def run_kernel(a_t, b_t, c_t, stream):
            compiled_gemm(
                m,
                n,
                k,
                batch,
                a_t,
                b_t,
                c_t,
                max_active_clusters,
                stream,
            )

    # Compute reference result
    if not skip_ref_check:
        # Execute kernel once for reference checking
        run_kernel(a_tensor, b_tensor, c_tensor, current_stream)
        print("Verifying results...")

        # Reference: C = einsum("mkl,nkl->mnl", A, B)
        ref = torch.einsum("mkl,nkl->mnl", a_ref, b_ref)

        # Convert c back to f32 for comparison.
        c_ref_device = c_ref.cuda()
        cute.testing.convert(
            c_tensor,
            from_dlpack(c_ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if c_major == "n" else 0)
            ),
        )
        c_ref = c_ref_device.cpu()

        torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)

    if use_strided:

        def generate_tensors():
            a_tensor_new, _ = cutlass_torch.cute_tensor_like(
                a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
            )
            b_tensor_new, _ = cutlass_torch.cute_tensor_like(
                b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
            )
            c_tensor_new, _ = cutlass_torch.cute_tensor_like(
                c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
            )
            a_tensor_new.mark_compact_shape_dynamic(
                mode=1,
                stride_order=(2, 0, 1),
                divisibility=1,
            )
            b_tensor_new.mark_compact_shape_dynamic(
                mode=1,
                stride_order=(2, 0, 1),
                divisibility=1,
            )
            c_tensor_new.mark_compact_shape_dynamic(
                mode=1,
                stride_order=(2, 0, 1),
                divisibility=1,
            )
            return cute.testing.JitArguments(
                m,
                n,
                k,
                batch,
                a_tensor_new,
                b_tensor_new,
                c_tensor_new,
                a_stride_m,
                a_stride_batch,
                max_active_clusters,
                current_stream,
            )
    else:

        def generate_tensors():
            a_tensor_new, _ = cutlass_torch.cute_tensor_like(
                a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
            )
            b_tensor_new, _ = cutlass_torch.cute_tensor_like(
                b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
            )
            c_tensor_new, _ = cutlass_torch.cute_tensor_like(
                c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
            )
            a_tensor_new.mark_compact_shape_dynamic(
                mode=1,
                stride_order=(2, 0, 1),
                divisibility=1,
            )
            b_tensor_new.mark_compact_shape_dynamic(
                mode=1,
                stride_order=(2, 0, 1),
                divisibility=1,
            )
            c_tensor_new.mark_compact_shape_dynamic(
                mode=1,
                stride_order=(2, 0, 1),
                divisibility=1,
            )
            return cute.testing.JitArguments(
                m,
                n,
                k,
                batch,
                a_tensor_new,
                b_tensor_new,
                c_tensor_new,
                max_active_clusters,
                current_stream,
            )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch.numel() * a_torch.element_size()
            + b_torch.numel() * b_torch.element_size()
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
        description="Functionality and Performance Test for Sm100 Dense Persistent BF16/FP16 GEMM."
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(512, 256, 256, 1),
        help="mnkl dimensions (comma-separated)",
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
        default=(1, 4),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument(
        "--ab_dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type for A and B tensors (bf16 or fp16)",
    )
    parser.add_argument(
        "--c_dtype",
        type=str,
        choices=["bf16", "fp32"],
        default="bf16",
        help="Data type for output tensor C (bf16 or fp32)",
    )
    parser.add_argument(
        "--use_2cta_instrs",
        action="store_true",
        default=False,
        help="Use 2-CTA instructions (default: False)",
    )
    parser.add_argument("--tolerance", type=float, default=1e-02, help="Tolerance for validation")
    parser.add_argument("--warmup_iterations", type=int, default=0, help="Warmup iterations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
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
        default=False,
        help="Use CUPTI to measure execution time",
    )
    parser.add_argument(
        "--use_strided",
        action="store_true",
        default=False,
        help="Use strided wrapper (BMM mode) instead of contiguous wrapper",
    )
    parser.add_argument(
        "--print_duration",
        action="store_true",
        default=False,
        help="Print execution time",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    # Convert string dtype args to cutlass types
    ab_dtype = _CUTLASS_DTYPE_MAP[args.ab_dtype]
    c_dtype = _CUTLASS_DTYPE_MAP[args.c_dtype]

    exec_time = run(
        args.mnkl,
        ab_dtype,
        c_dtype,
        args.use_2cta_instrs,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        args.use_cupti,
        args.use_strided,
    )
    if args.print_duration:
        print(f"Execution time: {exec_time} us")
    print("PASS")
