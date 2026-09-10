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

"""Functionality and performance test for Sm100 Persistent Dense BlockScaled GEMM activation fusion.

The kernel supports two activations selected via ``--activation``:
  * ``swiglu`` (default): gated SiLU; output C has N/2 columns.
  * ``gelu``: non-gated tanh-approx GELU; output C keeps the full N columns,
    with an optional per-N bias added before the activation (``--bias``).

Functional testing (SwiGLU):

python run_dense_blockscaled_gemm_act_fusion.py \
        --mnkl 512,256,256,1 \
        --ab_dtype Float4E2M1FN --c_dtype BFloat16 \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1

Functional testing with FP4 output + SFC:

python run_dense_blockscaled_gemm_act_fusion.py \
        --mnkl 512,256,256,1 \
        --ab_dtype Float4E2M1FN --c_dtype Float4E2M1FN \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1

Functional testing (non-gated GELU, bf16 output):

python run_dense_blockscaled_gemm_act_fusion.py --activation gelu \
        --mnkl 512,256,256,1 \
        --ab_dtype Float4E2M1FN --c_dtype BFloat16 \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1

Functional testing (non-gated GELU + bias, FP4 output + SFC):

python run_dense_blockscaled_gemm_act_fusion.py --activation gelu --bias \
        --mnkl 512,256,256,1 \
        --ab_dtype Float4E2M1FN --c_dtype Float4E2M1FN \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1

Perf testing:

python run_dense_blockscaled_gemm_act_fusion.py \
        --mnkl 4096,7168,2048,1 \
        --ab_dtype Float4E2M1FN --c_dtype BFloat16 \
        --sf_dtype Float8E4M3FN --sf_vec_size 16 \
        --mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \
        --vectorized_f32 \
        --skip_ref_check --use_cold_l2 --use_cupti --warmup_iterations 10 --iterations 50

Note: N is the full B matrix width. For SwiGLU the output C has N/2 columns; for
non-gated GELU the output C keeps the full N columns.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

from tensorrt_llm._torch.utils import ActivationType

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
        dense_blockscaled_gemm_act_fusion as kernel_module,
    )
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).parents[3] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell import dense_blockscaled_gemm_act_fusion as kernel_module

Sm100BlockScaledPersistentDenseGemmActFusionKernel = (
    kernel_module.Sm100BlockScaledPersistentDenseGemmActFusionKernel
)
cvt_sf_MKL_to_M32x4xrm_K4xrk_L = kernel_module.cvt_sf_MKL_to_M32x4xrm_K4xrk_L

try:
    from .testing import benchmark
except ImportError:
    from testing import benchmark


def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    vectorized_f32: bool = False,
    use_prefetch: bool = False,
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    use_cupti: bool = False,
    activation: str = "swiglu",
    use_bias: bool = False,
    **kwargs,
):
    """Runs and benchmarks the persistent batched dense block-scaled GEMM with activation fusion.

    This function prepares input tensors, launches the kernel, optionally
    validates the results against a reference implementation, and measures
    performance.

    N is the full B matrix width. For the gated SwiGLU activation the output C
    tensor has N/2 columns (interleaved up/gate pairs are combined). For the
    non-gated GELU activation the output C tensor keeps the full N columns,
    with an optional per-N bias added before the activation.

    Args:
        mnkl (Tuple[int, int, int, int]): The dimensions (M, N, K, L) of the
            GEMM problem. N is the full B matrix width (must be even).
        ab_dtype (Type[cutlass.Numeric]): Data type for input tensors A and B.
        sf_dtype (Type[cutlass.Numeric]): Data type for the scale factor tensors.
        sf_vec_size (int): Vector size for the scale factor tensors.
        c_dtype (Type[cutlass.Numeric]): Data type for the output tensor C.
        a_major (str): The major layout of tensor A ('k' or 'm').
        b_major (str): The major layout of tensor B ('k' or 'n').
        c_major (str): The major layout of tensor C ('n' or 'm').
        mma_tiler_mn (Tuple[int, int]): The shape of the MMA tile.
        cluster_shape_mn (Tuple[int, int]): The shape of the CTA cluster.
        vectorized_f32 (bool, optional): Use vectorized f32x2 operations.
            Defaults to False.
        use_prefetch (bool, optional): Enable TMA prefetching.
            Defaults to False.
        tolerance (float, optional): Tolerance for result validation.
            Defaults to 1e-01.
        warmup_iterations (int, optional): Number of warmup runs. Defaults to 0.
        iterations (int, optional): Number of benchmark iterations.
            Defaults to 1.
        skip_ref_check (bool, optional): If True, skips result validation.
            Defaults to False.
        use_cold_l2 (bool, optional): If True, uses a circular buffer to
            ensure a cold L2 cache. Defaults to False.
        use_cupti (bool, optional): If True, uses CUPTI to measure execution time.
            Defaults to False.
        activation (str, optional): Fused activation, "swiglu" (gated, output
            N/2) or "gelu" (non-gated tanh-approx GELU, output N). Defaults to
            "swiglu".
        use_bias (bool, optional): If True (only meaningful for "gelu"), adds a
            random per-N bias before the activation. Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns:
        float: The execution time of the kernel in microseconds.

    Raises:
        RuntimeError: If no CUDA-capable GPU is available.
        TypeError: If the configuration is not supported by the kernel.
    """
    print("Running Sm100 Persistent Dense BlockScaled GEMM Activation Fusion test with:")
    print(f"Activation: {activation}")
    print(f"Use bias: {use_bias}")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, SF dtype: {sf_dtype}, SF Vec size: {sf_vec_size}")
    print(f"C dtype: {c_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Vectorized f32: {vectorized_f32}")
    print(f"Use prefetch: {use_prefetch}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")
    print(f"Use CUPTI: {'True' if use_cupti else 'False'}")

    # Unpack parameters
    m, n, k, batch = mnkl
    # Gated SwiGLU halves the output width; non-gated GELU keeps the full N.
    n_out = (n // 2) if activation == "swiglu" else n
    # Alias kept so the SwiGLU reference path below stays byte-identical.
    n_after_swiglu = n_out

    if activation == "swiglu":
        assert n % 2 == 0, f"N must be even for SwiGLU fusion, got {n}"

    # If c_dtype is Float4E2M1FN, SFC will be generated
    generate_sfc = c_dtype == cutlass.Float4E2M1FN

    # Skip unsupported testcase
    if not Sm100BlockScaledPersistentDenseGemmActFusionKernel.can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
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
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype}, "
            f"{mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {batch}, "
            f"{a_major}, {b_major}, {c_major}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create tensor A (m, k, batch) and B (n, k, batch) - B has full N columns
    a_ref = cutlass_torch.matrix(batch, m, k, a_major == "m", cutlass.Float32)
    b_ref = cutlass_torch.matrix(batch, n, k, b_major == "n", cutlass.Float32)
    # C has N/2 columns after SwiGLU fusion
    c_ref = cutlass_torch.matrix(batch, m, n_after_swiglu, c_major == "m", cutlass.Float32)

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

    # Create scale factor tensor SFA/SFB
    def create_scale_factor_tensor(batch, mn, k, sf_vec_size, dtype):
        def ceil_div(a, b):
            return (a + b - 1) // b

        sf_k = ceil_div(k, sf_vec_size)
        ref_shape = (batch, mn, sf_k)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            batch,
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
            .expand(batch, mn, sf_k, sf_vec_size)
            .reshape(batch, mn, sf_k * sf_vec_size)
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

    def create_sf_layout_tensor(batch, mn, nk, sf_vec_size):
        """Create a scale factor layout tensor (for SFC verification)."""

        def ceil_div(a, b):
            return (a + b - 1) // b

        sf_k = ceil_div(nk, sf_vec_size)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            batch,
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        mma_permute_order = (3, 4, 1, 5, 2, 0)

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

    sfa_ref, sfa_tensor, sfa_torch = create_scale_factor_tensor(batch, m, k, sf_vec_size, sf_dtype)
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(batch, n, k, sf_vec_size, sf_dtype)

    # Create SFC tensor and norm_const_tensor for FP4 output quantization
    sfc_tensor = None
    sfc_torch = None
    norm_const_tensor = None
    norm_const_torch = None
    if generate_sfc:
        _, sfc_tensor, sfc_torch = create_scale_factor_tensor(
            batch, m, n_after_swiglu, sf_vec_size, sf_dtype
        )
        norm_const = 1.0
        norm_const_torch = torch.tensor([norm_const], dtype=torch.float32).cuda()
        norm_const_tensor = from_dlpack(norm_const_torch)

    # Create alpha tensor (single-element tensor containing alpha value)
    alpha_value = 1.0
    alpha_torch = torch.tensor([alpha_value], dtype=torch.float32).cuda()
    alpha = from_dlpack(alpha_torch)

    # Create optional per-N bias for the non-gated GELU path. The kernel's
    # __call__ takes a (m, n_out, l) cute.Tensor with M-stride 0 (broadcast over
    # rows). We build it from a torch [n_out] vector expanded to (m, n_out, l)
    # with stride (0, 1, 0) -- the same broadcast layout wrapper() constructs.
    bias_n_torch = None
    bias_tensor = None
    use_gelu_bias = activation == "gelu" and use_bias
    if use_gelu_bias:
        bias_n_torch = (torch.randn(n_out, dtype=torch.float32) * 0.1).cuda()
        bias_bcast = bias_n_torch.view(1, n_out, 1).expand(m, n_out, batch)
        bias_tensor = from_dlpack(bias_bcast, assumed_align=16)

    # Configure gemm kernel
    gemm = Sm100BlockScaledPersistentDenseGemmActFusionKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        vectorized_f32,
        use_prefetch,
        activation_type=(ActivationType.Gelu if activation == "gelu" else ActivationType.Swiglu),
    )

    # Compute max active clusters on current device
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # Initialize Stream
    current_stream = cutlass_torch.default_stream()

    # Compile gemm kernel. bias_tensor is the trailing optional __call__ arg;
    # it is only supplied for the non-gated GELU + bias path, so the SwiGLU
    # (and bias-free GELU) compile signatures stay unchanged.
    compile_bias_kwargs = {"bias_tensor": bias_tensor} if use_gelu_bias else {}
    if generate_sfc:
        compiled_gemm = cute.compile(
            gemm,
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            alpha,
            max_active_clusters,
            current_stream,
            lambda x: x,  # epilogue_op (default)
            sfc_tensor,
            norm_const_tensor,
            **compile_bias_kwargs,
            options="--opt-level 2",
        )
    else:
        compiled_gemm = cute.compile(
            gemm,
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            alpha,
            max_active_clusters,
            current_stream,
            **compile_bias_kwargs,
            options="--opt-level 2",
        )

    # Compute reference result
    if not skip_ref_check:
        # Execute kernel once for reference checking. bias_tensor is the trailing
        # optional arg, supplied only for the non-gated GELU + bias path.
        run_bias_kwargs = {"bias_tensor": bias_tensor} if use_gelu_bias else {}
        if generate_sfc:
            compiled_gemm(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                alpha,
                current_stream,
                sfc_tensor,
                norm_const_tensor,
                **run_bias_kwargs,
            )
        else:
            compiled_gemm(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                alpha,
                current_stream,
                **run_bias_kwargs,
            )

        torch.cuda.synchronize()
        print("Verifying results...")

        # Compute reference GEMM: ref = (A * SFA) @ (B * SFB) with alpha=1.0
        res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
        res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
        ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)
        # alpha = 1.0, so no scaling needed for ref

        # Apply the fused activation, producing ref_after_swiglu with n_out
        # columns (the variable name is shared by both activations so the
        # downstream comparison code is identical).
        if activation == "swiglu":
            # Apply SwiGLU: split N into interleaved up/gate blocks of 64 columns
            # (matching epi_tile N=64), compute output = up * silu(gate)
            group = 64
            assert n % group == 0, f"N ({n}) must be divisible by {group} for SwiGLU block grouping"
            num_blocks = n // group
            assert num_blocks % 2 == 0, "Number of blocks must be even (pairs of up/gate)"

            cols = torch.arange(n, device=ref.device, dtype=torch.long)
            block_cols = cols.view(num_blocks, group)
            # Even blocks (0, 2, 4, ...) are 'up', odd blocks (1, 3, 5, ...) are 'gate'
            up_idx = block_cols[0::2].reshape(-1)
            gate_idx = block_cols[1::2].reshape(-1)
            ref_up = ref.index_select(1, up_idx)
            ref_gate = ref.index_select(1, gate_idx)
            ref_after_swiglu = ref_up * (ref_gate * torch.sigmoid(ref_gate))
        else:
            # Non-gated tanh-approx GELU on full N: x = alpha * acc + bias
            # (bias per-N, broadcast over M), then gelu_tanh(x).
            x = alpha_value * ref
            if use_gelu_bias:
                # bias_n_torch is [n_out] (built on CUDA for the kernel); the
                # reference GEMM `ref` is on CPU, so align the device before add.
                # Broadcast over M (dim 0) and L (dim 2).
                x = x + bias_n_torch.view(1, n_out, 1).to(x.device)
            ref_after_swiglu = (
                0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
            )

        # Convert kernel output C back to f32 for comparison
        res = c_ref.cuda()
        cute.testing.convert(
            c_tensor,
            from_dlpack(res, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if c_major == "n" else 0)
            ),
        )
        res = res.cpu()

        if c_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            print("Verifying C Tensor...")
            torch.testing.assert_close(res, ref_after_swiglu, atol=tolerance, rtol=1e-02)
        elif c_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN):
            if generate_sfc:
                print("Verifying SFC Tensor...")
                norm_const = 1.0

                # 1. Compute reference SFC in fp32
                def ceil_div(a, b):
                    return (a + b - 1) // b

                sfn = ceil_div(n_after_swiglu, sf_vec_size)
                # Reshape ref to (batch, m, sfn, sf_vec_size)
                ref_for_sf = ref_after_swiglu.permute(2, 0, 1).contiguous()  # (batch, m, n_out)
                ref_for_sf = ref_for_sf.view(batch, m, sfn, sf_vec_size)
                # Take abs max over sf_vec_size dimension
                ref_for_sf, _ = torch.abs(ref_for_sf).max(dim=3)  # (batch, m, sfn)
                # Multiply by norm_const and rcp_limits
                ref_sfc_f32 = ref_for_sf * norm_const * gemm.get_dtype_rcp_limits(c_dtype)
                # Permute to (m, sfn, batch)
                ref_sfc_f32 = ref_sfc_f32.permute(1, 2, 0)

                # Convert fp32 -> f8 -> fp32 for ref_sfc_f32 (quantize round-trip)
                ref_sfc_f8_torch = torch.empty(
                    *(batch, m, sfn), dtype=torch.uint8, device="cuda"
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

                # 2. Convert ref_sfc_f32 to scale factor layout and compare with kernel sfc tensor
                ref_sfc_f32_cute_torch_cpu, _ = create_sf_layout_tensor(
                    batch, m, n_after_swiglu, sf_vec_size
                )
                cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
                    from_dlpack(ref_sfc_f32),
                    from_dlpack(ref_sfc_f32_cute_torch_cpu),
                )
                kernel_sfc_cute_torch_cpu, _ = create_sf_layout_tensor(
                    batch, m, n_after_swiglu, sf_vec_size
                )
                kernel_sfc_tensor, kernel_sfc_torch_tensor = cutlass_torch.cute_tensor_like(
                    kernel_sfc_cute_torch_cpu,
                    cutlass.Float32,
                    is_dynamic_layout=True,
                    assumed_align=16,
                )
                cute.testing.convert(sfc_tensor, kernel_sfc_tensor)
                kernel_sfc_cute_torch_cpu = kernel_sfc_torch_tensor.cpu()
                torch.testing.assert_close(
                    kernel_sfc_cute_torch_cpu,
                    ref_sfc_f32_cute_torch_cpu,
                    atol=tolerance,
                    rtol=1e-02,
                )
                print("SFC Tensor comparison passed!")

                # 3. Quantized output with scale factor
                ref_sfc_rcp = norm_const * ref_sfc_f32.reciprocal()
                ref_sfc_rcp_expanded = ref_sfc_rcp.unsqueeze(2).expand(m, sfn, sf_vec_size, batch)
                ref_sfc_rcp_expanded = ref_sfc_rcp_expanded.reshape(m, sfn * sf_vec_size, batch)
                ref_sfc_rcp_expanded = ref_sfc_rcp_expanded[:, :n_after_swiglu, :]
                ref_after_swiglu = torch.einsum(
                    "mnl,mnl->mnl", ref_after_swiglu, ref_sfc_rcp_expanded
                )

            print("Verifying C Tensor...")
            # Convert ref : f32 -> low-precision -> f32 (quantize round-trip)
            ref_ = torch.empty(
                *(batch, m, n_after_swiglu), dtype=torch.uint8, device="cuda"
            ).permute(1, 2, 0)
            ref_ = from_dlpack(ref_, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            ref_.element_type = c_dtype
            ref_device = ref_after_swiglu.permute(2, 0, 1).contiguous().permute(1, 2, 0).cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_)
            cute.testing.convert(ref_, ref_tensor)
            torch.testing.assert_close(res, ref_device.cpu(), atol=tolerance, rtol=1e-02)

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

        _, sfa_tensor, _ = create_scale_factor_tensor(batch, m, k, sf_vec_size, sf_dtype)
        _, sfb_tensor, _ = create_scale_factor_tensor(batch, n, k, sf_vec_size, sf_dtype)

        # bias_tensor (read-only broadcast view) is the trailing runtime arg;
        # appended only for the non-gated GELU + bias path so the bias-free
        # JitArguments signatures stay unchanged.
        extra_bias = (bias_tensor,) if use_gelu_bias else ()
        if generate_sfc:
            _, sfc_tensor, _ = create_scale_factor_tensor(
                batch, m, n_after_swiglu, sf_vec_size, sf_dtype
            )
            norm_const_torch = torch.tensor([1.0], dtype=torch.float32).cuda()
            norm_const_tensor = from_dlpack(norm_const_torch)
            return cute.testing.JitArguments(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                alpha,
                current_stream,
                sfc_tensor,
                norm_const_tensor,
                *extra_bias,
            )
        else:
            return cute.testing.JitArguments(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                alpha,
                current_stream,
                *extra_bias,
            )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch.numel() * a_torch.element_size()
            + b_torch.numel() * b_torch.element_size()
            + sfa_torch.numel() * sfa_torch.element_size()
            + sfb_torch.numel() * sfb_torch.element_size()
            + c_torch.numel() * c_torch.element_size()
        )
        if generate_sfc:
            one_workspace_bytes += (
                sfc_torch.numel() * sfc_torch.element_size()
                + norm_const_torch.numel() * norm_const_torch.element_size()
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
        description="Functionality and Performance Test for "
        "Sm100 Dense Persistent BlockScaled GEMM with SwiGLU Fusion."
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(512, 256, 256, 1),
        help="mnkl dimensions (comma-separated). N is the full B matrix width (must be even). "
        "Output C has N/2 columns after SwiGLU fusion.",
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
    parser.add_argument(
        "--activation",
        choices=["swiglu", "gelu"],
        type=str,
        default="swiglu",
        help="Fused activation: 'swiglu' (gated, output N/2) or 'gelu' "
        "(non-gated tanh-approx GELU, output N).",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Add a random per-N bias before the activation (only meaningful "
        "for --activation gelu).",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--vectorized_f32",
        action="store_true",
        default=False,
        help="Use vectorized f32x2 operations for SwiGLU computation",
    )
    parser.add_argument(
        "--use_prefetch",
        action="store_true",
        default=False,
        help="Enable TMA prefetch for both A and B matrices (default: False)",
    )
    parser.add_argument("--tolerance", type=float, default=1e-01, help="Tolerance for validation")
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

    exec_time = run(
        args.mnkl,
        args.ab_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.c_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.vectorized_f32,
        args.use_prefetch,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        args.use_cupti,
        activation=args.activation,
        use_bias=args.bias,
    )
    if args.print_duration:
        print(f"Execution time: {exec_time:.2f} us")
    print("PASS")
