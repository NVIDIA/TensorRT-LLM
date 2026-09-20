# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""TRT-LLM-backed quantized linear custom ops.

These ops dispatch to ``torch.ops.trtllm.*`` kernels and are registered under
the ``auto_deploy`` namespace so the AutoDeploy graph transforms can route
quantized linears to TRT-LLM implementations.
"""

from typing import List, Optional

import torch
import triton

from ...utils.fp8_dequant import dequant_fp8_weight_two_dim_block_grid


def _dequant_block_fp8_weight(weight_fp8, weight_scale, block_n, block_k, dtype=torch.bfloat16):
    """Dequantize block-scaled FP8 weight to BF16 for tiny projections."""
    return dequant_fp8_weight_two_dim_block_grid(
        weight_fp8, weight_scale, block_n, block_k, dtype=dtype
    )


@torch.library.custom_op("auto_deploy::trtllm_fp8_deepgemm", mutates_args=())
def trtllm_fp8_deepgemm(
    input: torch.Tensor,  # [..., K] bfloat16
    weight: torch.Tensor,  # [N, K] float8_e4m3fn
    bias: Optional[torch.Tensor],  # [N] or None
    weight_scale: torch.Tensor,  # UE8M0 packed int, TMA-aligned col-major
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    """Blackwell (SM100f) FineGrainedFP8 linear via DeepGEMM fp8_swap_ab_gemm.

    Dedicated compile-time-selected path for the UE8M0 fast path. The caller
    (fuse_finegrained_fp8_linear transform) is responsible for routing to this
    op only when:
      * Running on SM100f, and
      * `weight_scale` has been converted to UE8M0 packed int (torch.int) layout
        by FineGrainedFP8LinearQuantization.post_load_hook.

    Keeping this as its own op avoids per-call `is_sm_100f()` / dtype branching
    inside `trtllm_finegrained_fp8_linear` and makes the graph shape explicit.
    """
    if input.dtype != torch.bfloat16:
        raise ValueError("trtllm_fp8_deepgemm expects bfloat16 input")

    input_shape = input.shape
    N = weight.shape[0]
    input_2d = input.reshape(-1, input_shape[-1])
    output = torch.ops.trtllm.fp8_swap_ab_gemm(
        input_2d,
        weight,
        weight_scale,
        output_dtype=input.dtype,
        # Both activation scales (from _fp8_quantize_1x128_ue8m0 inside
        # fp8_swap_ab_gemm) and weight scales (from post_load_hook) are
        # already UE8M0. Skip DeepGEMM's internal conversion.
        disable_ue8m0_cast=True,
    )
    if bias is not None:
        output = output + bias
    return output.reshape(*input_shape[:-1], N)


@trtllm_fp8_deepgemm.register_fake
def _trtllm_fp8_deepgemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_scale: torch.Tensor,
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    """Fake implementation for torch.export tracing."""
    out_features = weight.shape[0]
    return torch.empty((*input.shape[:-1], out_features), dtype=input.dtype, device=input.device)


@torch.library.custom_op("auto_deploy::trtllm_finegrained_fp8_linear", mutates_args=())
def trtllm_finegrained_fp8_linear(
    input: torch.Tensor,  # [..., K] bfloat16
    weight: torch.Tensor,  # [N, K] float8_e4m3fn
    bias: Optional[torch.Tensor],  # [N] or None
    weight_scale: torch.Tensor,  # [N/128, K/128] per-block weight scale (FP32)
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    """TRT-LLM optimized FineGrainedFP8 linear operation.

    Uses TRT-LLM's fp8_block_scaling_gemm kernel with FP32 per-block weight
    scales. The SM100f + UE8M0 fast path is handled by a separate op
    (`trtllm_fp8_deepgemm`) that the `fuse_finegrained_fp8_linear` transform
    dispatches to at compile time.

    - weight_scale: per-block weight scale with shape [ceil(N/128), ceil(K/128)]
      in FP32. UE8M0 packed int scales are NOT handled here.
    - Input is dynamically quantized using fp8_quantize_1x128.
    - For exact 128x128 blocks, uses the TRT-LLM fast path; otherwise falls
      back to BF16 dequant + cuBLAS to avoid underutilizing the FP8 kernel.
    """
    from tensorrt_llm._utils import get_sm_version

    if input.dtype == torch.float8_e4m3fn:
        raise ValueError("trtllm_finegrained_fp8_linear expects bfloat16 input, not FP8")

    input_shape = input.shape
    N, K = weight.shape

    # TRT-LLM fp8_block_scaling_gemm requires float32 scales; HF checkpoints may
    # store weight_scale_inv in bfloat16 to save space, so cast here.
    if weight_scale.dtype != torch.float32:
        weight_scale = weight_scale.float()

    # Derive effective block size from weight and scale shapes.
    scale_n, scale_k = weight_scale.shape
    if scale_n == 0 or scale_k == 0:
        raise ValueError(
            f"trtllm_finegrained_fp8_linear: weight_scale has zero dimension "
            f"(shape={weight_scale.shape}), weight shape={weight.shape}. "
            f"This usually means scale tensor sharding produced an empty tensor."
        )
    # Ceiling division is required because the weight dimension may not be
    # evenly divisible by the number of scale blocks (e.g. after TP sharding).
    block_n = triton.cdiv(N, scale_n)
    block_k = triton.cdiv(K, scale_k)

    # TRT-LLM fp8_block_scaling_gemm requires exact 128x128 blocks.
    # For small layers where a dimension < 128 (e.g. N=64), the derived block
    # size will be < 128.  Fall back to BF16 dequant + cuBLAS.
    if block_n != 128 or block_k != 128:
        # BF16 fallback: the Triton FP8 kernel launches Grid=1x1x1 for tiny N,
        # wasting 99% of SM capacity. Dequantize weight + cuBLAS is faster.
        weight_dequant = _dequant_block_fp8_weight(
            weight, weight_scale, block_n, block_k, dtype=input.dtype
        )
        output = torch.nn.functional.linear(input, weight_dequant, bias)
        return output.reshape(*input_shape[:-1], N) if len(input_shape) > 2 else output

    # Flatten input for GEMM: [..., K] -> [M, K]
    input_2d = input.reshape(-1, input_shape[-1])

    # SM version-specific activation quantization
    if get_sm_version() == 120:
        from tensorrt_llm._torch.modules.linear import per_token_quant_and_transform

        act_fp8, act_sf = per_token_quant_and_transform(input_2d)
    else:
        # Hopper (SM90) and Blackwell (SM100+) share the same path
        act_fp8, act_sf = torch.ops.trtllm.fp8_quantize_1x128(input_2d)
    output = torch.ops.trtllm.fp8_block_scaling_gemm(act_fp8, weight, act_sf, weight_scale)

    if bias is not None:
        output = output + bias

    # Reshape back to original batch dimensions: [M, N] -> [..., N]
    return output.reshape(*input_shape[:-1], weight.shape[0])


@trtllm_finegrained_fp8_linear.register_fake
def _trtllm_finegrained_fp8_linear_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_scale: torch.Tensor,
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    """Fake implementation for torch.export tracing."""
    out_features = weight.shape[0]
    return torch.empty((*input.shape[:-1], out_features), dtype=input.dtype, device=input.device)
