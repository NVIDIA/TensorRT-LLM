# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.quant import TRTLLM_NVFP4_SCALING_VECTOR_SIZE
from tensorrt_llm._torch.utils import ActivationType


@torch.library.custom_op("auto_deploy::trtllm_moe_fused", mutates_args=())
def trtllm_moe_fused(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
) -> torch.Tensor:
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    routing_weights = routing_weights.to(torch.float32)
    selected_experts = selected_experts.to(torch.int32)
    quant_scales = []

    # Determine activation type

    activation_type = ActivationType.Swiglu
    if is_gated_mlp:
        # Gated MLP uses Silu: silu(x @ w1.T) * (x @ w3.T)
        if act_fn in [ActivationType.Silu, ActivationType.Swiglu]:
            activation_type = ActivationType.Swiglu
        else:
            raise ValueError(
                f"Unsupported activation '{ActivationType(act_fn).name}' for gated_mlp. Use 'silu'."
            )
    else:
        # For non-gated MLP with ReLU^2
        if act_fn == ActivationType.Relu2:
            activation_type = ActivationType.Relu2
        else:
            raise ValueError(
                f"Unsupported activation '{ActivationType(act_fn).name}' for mlp. Use 'relu2'."
            )

    return torch.ops.trtllm.fused_moe(
        x,
        selected_experts,
        routing_weights,
        fc1_expert_weights=w3_w1_stacked_weight,
        fc1_expert_biases=None,
        fc2_expert_weights=w2_stacked_weight,
        fc2_expert_biases=None,
        output_dtype=x.dtype,
        quant_scales=quant_scales,
        activation_type=activation_type,
    )[0].view(x_shape)


@trtllm_moe_fused.register_fake
def trtllm_moe_fused_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
) -> torch.Tensor:
    return torch.empty_like(x)


# NOTE(suyogg): If compile ever fails because of this, just write a triton kernel
# for this function and use it as a custom op.
@torch.compile
def _quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to FP8 with clamping (matches torch_quant_fp8_linear)."""
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    return (x / scale).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)


def _validate_mlp_style_and_act_fn(is_gated_mlp: bool, act_fn: int) -> None:
    assert (is_gated_mlp and act_fn in [ActivationType.Silu, ActivationType.Swiglu]) or (
        not is_gated_mlp and act_fn == ActivationType.Relu2
    ), (
        f"Unsupported combination: is_gated_mlp='{is_gated_mlp}', act_fn='{act_fn}'. "
        f"Supported combinations: gated mlp with silu or mlp with relu2."
    )


@torch.library.custom_op("auto_deploy::trtllm_quant_fp8_moe_fused", mutates_args=())
def trtllm_quant_fp8_moe_fused(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    fc1_act_scale: torch.Tensor,
    fc1_dequant_scale: torch.Tensor,
    fc2_act_scale_reciprocal: torch.Tensor,
    fc2_dequant_scale: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
) -> torch.Tensor:
    """TensorRT-LLM Cutlass FP8 (W8A8) MoE for gated and non-gated MLP.

    Computes (per expert):
        For gated_mlp:
            y = (act(x @ w1.T) * (x @ w3.T)) @ w2.T  # act := SiLU
        For mlp:
            y = act(x @ w1.T) @ w2.T                 # act := ReLU^2
    Notes:
        - FC1 implements: fc1_output = (act(x @ w1.T) * (x @ w3.T)) or fc1_output = act(x @ w1.T)
        - FC2 implements: fc2_output = fc1_output @ w2.T
        - FC1 weights are concatenated w3 and w1 if gated_mlp, otherwise w1

    Parameters:
        x: BF16/FP16 input tensor of shape (B, H) or (B, S, H)
        selected_experts: Expert indices (B*S, TOP_K)
        routing_weights: Routing weights (B*S, TOP_K)
        fc1_expert_weights: FC1 weights [E, 2*I, H] for gated_mlp, [E, I, H] for mlp
        fc2_expert_weights: FC2 weights [E, H, I]
        fc1_act_scale: FC1 activation scale [E]
        fc1_dequant_scale: FC1 dequant scale [E]
        fc2_act_scale_reciprocal: FC2 activation scale reciprocal [E]
        fc2_dequant_scale: FC2 dequant scale [E]
        is_gated_mlp: True for gated_mlp, False for mlp
        act_fn: ActivationType.Silu for gated_mlp, ActivationType.Relu2 for mlp

    Returns:
        Output tensor of shape (B, H) or (B, S, H)
    """

    _validate_mlp_style_and_act_fn(is_gated_mlp, act_fn)
    act_fn = ActivationType.Swiglu if act_fn == ActivationType.Silu else act_fn

    # Store original shape and flatten to 2D
    x_shape = x.shape
    x2d = x.view(-1, x_shape[-1])
    # Quantize the input
    x_q_fp8 = _quantize_fp8(x2d, fc1_act_scale[0])

    # Scales are stored in float32
    w1_input_scale = fc1_act_scale[0]

    # Prepare quant_scales for TensorRT-LLM (Cutlass) FP8 format:
    # [fc1_dequant_scale, fc2_act_scale_reciprocal, fc2_dequant_scale, gemm1_input_dequant_scale]
    # For gated MLP:
    # These are precomputed in `fused_moe` transform
    # - fc1_dequant_scale: w1_weight_scale * w1_input_scale (combined for w1 and w3)
    # - fc2_act_scale_reciprocal: 1 / w2_input_scale
    # - fc1_dequant_scale: w2_weight_scale * w2_input_scale
    # - fc1_act_scale: w1_input_scale

    assert fc1_dequant_scale.ndim == 1, "fc1_dequant_scale must be 1D"
    assert fc2_dequant_scale.ndim == 1, "fc2_dequant_scale must be 1D"
    quant_scales = [fc1_dequant_scale, fc2_act_scale_reciprocal, fc2_dequant_scale, w1_input_scale]

    # Ensure contiguous tensors
    selected_experts = selected_experts.int().contiguous()
    routing_weights = routing_weights.contiguous()

    # Note! Outputting Float8_e4m3fn directly is not currently supported
    output = torch.ops.trtllm.fused_moe(
        x_q_fp8,
        selected_experts,
        routing_weights,
        fc1_expert_weights=fc1_expert_weights,
        fc1_expert_biases=None,
        fc2_expert_weights=fc2_expert_weights.contiguous(),
        fc2_expert_biases=None,
        output_dtype=x.dtype,
        quant_scales=quant_scales,
        activation_type=act_fn,
    )

    # Restore original shape
    return output[0].view(x_shape)


@trtllm_quant_fp8_moe_fused.register_fake
def trtllm_quant_fp8_moe_fused_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    fc1_act_scale: torch.Tensor,
    fc1_dequant_scale: torch.Tensor,
    fc2_act_scale_reciprocal: torch.Tensor,
    fc2_dequant_scale: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
) -> torch.Tensor:
    _validate_mlp_style_and_act_fn(is_gated_mlp, act_fn)
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::trtllm_quant_nvfp4_moe_fused", mutates_args=())
def trtllm_quant_nvfp4_moe_fused(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights_fp4: torch.Tensor,
    fc2_expert_weights_fp4: torch.Tensor,
    fc1_weight_blockscale_fp8: torch.Tensor,
    fc2_weight_blockscale_fp8: torch.Tensor,
    fc1_act_global_scale: torch.Tensor,
    fc2_act_global_scale: torch.Tensor,
    fc1_alpha: torch.Tensor,
    fc2_alpha: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
) -> torch.Tensor:
    """TensorRT-LLM Cutlass NVFP4 W8A8 MoE for gated and non-gated MLP.

    Computes (per expert):
        For gated_mlp:
            y = (act(x @ w1.T) * (x @ w3.T)) @ w2.T  # act := SiLU
        For mlp:
            y = act(x @ w1.T) @ w2.T                 # act := ReLU^2

    Notes:
    - FC1 implements: fc1_output = (act(x @ w1.T) * (x @ w3.T)) or fc1_output = act(x @ w1.T)
    - FC2 implements: fc2_output = fc1_output @ w2.T
    - FC1 weights are concatenated w3 and w1 if gated_mlp, otherwise w1
    - FP4 elements pairs are packed as a single uint8 element

    Parameters:
        x: BF16/FP16 input tensor of shape (B, H) or (B, S, H)
        selected_experts: Expert indices (B*S, TOP_K)
        routing_weights: Routing weights (B*S, TOP_K)
        fc1_expert_weights_fp4: FP4 FC1 weights [E, 2*I, H/2] or [E, I, H/2]; packed uint8
        fc2_expert_weights_fp4: FP4 FC2 weights [E, H, I/2]; packed uint8
        fc1_weight_blockscale_fp8: Block scales for FC1 weights (w1 or cat(w3, w1))
        fc2_weight_blockscale_fp8: Block scales for FC2 weights (w2)
        fc1_act_global_scale: Global scale for FC1 activations (scalar)
        fc2_act_global_scale: Global scale for FC2 activations (scalar)
        fc1_alpha: FC1 dequant scales = 1.0 / (fc1_act_global_scale * fc1_weight_global_scale)
        fc2_alpha: FC2 dequant scales = 1.0 / (fc2_act_global_scale * fc2_weight_global_scale)
        mlp_style: "gated_mlp" or "mlp"
        act_fn: "silu" for gated_mlp, "relu2" for mlp
    """

    # Validate block scale tensors are 3D (padding requirements handled below)
    assert fc1_weight_blockscale_fp8.ndim == 3, "fc1_weight_blockscale_fp8 must be 3D"
    assert fc2_weight_blockscale_fp8.ndim == 3, "fc2_weight_blockscale_fp8 must be 3D"

    _validate_mlp_style_and_act_fn(is_gated_mlp, act_fn)
    act_fn = ActivationType.Swiglu if act_fn == ActivationType.Silu else act_fn

    if x.dtype in (torch.float16, torch.bfloat16):
        x_q_fp4, input_blockscale = torch.ops.trtllm.fp4_quantize(
            x, fc1_act_global_scale, TRTLLM_NVFP4_SCALING_VECTOR_SIZE
        )
        output_dtype = x.dtype
    else:
        x_q_fp4 = x
        input_blockscale = None
        output_dtype = x.dtype

    # quant_scales is described by this code:
    # https://github.com/NVIDIA/TensorRT-LLM/blob/c9771ebb997683c08b26bbba796a7fc6aff09d93/cpp/tensorrt_llm/thop/moeOp.cpp#L1015
    quant_scales = [
        fc1_act_global_scale,  # torch.float32; [E] or scalar
        fc1_weight_blockscale_fp8.view(
            torch.int32
        ),  # 4 FP8 as packed int32; [E, I*2, H / 16 / 4] or [E, I, H / 16 / 4]
        fc1_alpha,  # torch.float32; [E]
        fc2_act_global_scale,  # torch.float32; [E] or scalar
        fc2_weight_blockscale_fp8.view(torch.int32),  # 4 FP8 as packed int32; [E, H, I / 16 / 4]
        fc2_alpha,  # torch.float32; [E]
    ]

    trtllm_output = torch.ops.trtllm.fused_moe(
        x_q_fp4.view(torch.long),
        selected_experts.to(torch.int32),
        routing_weights.to(torch.float32),
        # Groups of 16 FP4 weight elements are packed as a single int64 element (see isNvfp4Quant in moeOp.cpp)
        fc1_expert_weights=fc1_expert_weights_fp4.view(torch.long),
        fc1_expert_biases=None,
        fc2_expert_weights=fc2_expert_weights_fp4.view(torch.long),
        fc2_expert_biases=None,
        output_dtype=output_dtype,
        quant_scales=quant_scales,
        input_sf=input_blockscale,
        activation_type=act_fn,
    )[0].view(x.shape)

    return trtllm_output


@trtllm_quant_nvfp4_moe_fused.register_fake
def trtllm_quant_nvfp4_moe_fused_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights_fp4: torch.Tensor,
    fc2_expert_weights_fp4: torch.Tensor,
    fc1_weight_blockscale_fp8: torch.Tensor,
    fc2_weight_blockscale_fp8: torch.Tensor,
    fc1_act_global_scale: torch.Tensor,
    fc2_act_global_scale: torch.Tensor,
    fc1_alpha: torch.Tensor,
    fc2_alpha: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
) -> torch.Tensor:
    return torch.empty_like(x)
