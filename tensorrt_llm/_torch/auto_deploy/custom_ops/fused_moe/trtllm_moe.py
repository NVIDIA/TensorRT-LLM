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

from typing import List, Optional

import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.quant import TRTLLM_NVFP4_SCALING_VECTOR_SIZE
from tensorrt_llm._torch.auto_deploy.utils.mapping_utils import MappingSerializer
from tensorrt_llm._torch.distributed.moe_alltoall import MoeAlltoAll
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm._utils import mpi_allgather


@torch.library.custom_op("auto_deploy::trtllm_moe_fused", mutates_args=())
def trtllm_moe_fused(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
    apply_routing_on_input: bool = False,
    enable_alltoall: bool = False,
    # Sharding configuration (only used when enable_alltoall=True) - see MappingSerializer
    mapping_config: Optional[List[int]] = None,
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

    # Deserialize Mapping from config (or use defaults if None)
    # Note: mapping_config is only populated when enable_alltoall=True
    mapping = MappingSerializer.deserialize(mapping_config)

    # All-to-all paradigm for attention-DP: tokens are DP-sharded, experts are EP-sharded.
    # Expert IDs remain in GLOBAL coordinates, and MoeAlltoAll handles dispatch/combine.
    # Alternative (enable_alltoall=False): Expert IDs localized, remote weights zeroed, all_reduce.
    if enable_alltoall:
        top_k = selected_experts.shape[1]
        hidden_size = x.shape[-1]
        x_dtype = x.dtype

        # Expert weights are sharded - shape[0] gives LOCAL expert count
        local_num_experts = w3_w1_stacked_weight.shape[0]
        global_num_experts = local_num_experts * mapping.moe_ep_size

        # Workspace size for MoeAlltoAll (set to max_batch_size * max_seq_len * ep_size)
        max_num_tokens = MappingSerializer.get_max_num_tokens(mapping_config)

        workspace_size = MoeAlltoAll.calculate_required_workspace_size(
            mapping.moe_ep_size, top_k, max_num_tokens, hidden_size, x_dtype
        )

        # Get runtime tokens per rank for actual dispatch/combine
        local_tokens = int(x.shape[0])
        all_rank_tokens = mpi_allgather(local_tokens)
        if isinstance(all_rank_tokens, list):
            runtime_max_tokens_per_rank = max(all_rank_tokens)
        else:
            runtime_max_tokens_per_rank = local_tokens

        # Build MoeAlltoAll (num_slots = num_experts without EPLB load balancing)
        moe_a2a = MoeAlltoAll(
            mapping=mapping,
            max_num_tokens=max_num_tokens,
            top_k=top_k,
            num_slots=global_num_experts,  # No EPLB: num_slots == num_experts
            workspace_size_per_rank=workspace_size,
            num_experts=None,  # None = EPLB disabled
        )

        # Validate and clamp expert IDs (using GLOBAL range)
        invalid_expert_id = global_num_experts
        invalid_input_mask = (selected_experts < 0) | (selected_experts >= global_num_experts)
        if invalid_input_mask.any():
            selected_experts = selected_experts.clamp(0, global_num_experts - 1)
            routing_weights = routing_weights.masked_fill(invalid_input_mask, 0.0)

        # DISPATCH: Route tokens to correct GPUs based on global expert IDs
        recv_x, recv_selected, recv_weights = moe_a2a.dispatch(
            selected_experts,
            [x.contiguous(), selected_experts.contiguous(), routing_weights.contiguous()],
            runtime_max_tokens_per_rank,
            invalid_token_expert_id=invalid_expert_id,
            expert_id_payload_index=1,
        )

        dispatched_x = recv_x.reshape(-1, hidden_size)
        dispatched_selected = recv_selected.reshape(-1, top_k)
        dispatched_weights = recv_weights.reshape(-1, top_k)

        # Handle invalid/padding tokens from dispatch
        invalid_mask = (dispatched_selected < 0) | (dispatched_selected >= global_num_experts)
        if invalid_mask.any():
            dispatched_weights = dispatched_weights.masked_fill(invalid_mask, 0.0)
            dispatched_selected = dispatched_selected.clamp(0, global_num_experts - 1)

        # Compute: kernel uses GLOBAL expert IDs and handles localization internally
        moe_out = torch.ops.trtllm.fused_moe(
            dispatched_x,
            dispatched_selected,  # GLOBAL expert IDs
            dispatched_weights,
            fc1_expert_weights=w3_w1_stacked_weight,
            fc1_expert_biases=None,
            fc2_expert_weights=w2_stacked_weight,
            fc2_expert_biases=None,
            output_dtype=x.dtype,
            quant_scales=quant_scales,
            tp_size=mapping.moe_tp_size,
            tp_rank=mapping.moe_tp_rank,
            ep_size=mapping.moe_ep_size,
            ep_rank=mapping.moe_ep_rank,
            cluster_size=mapping.moe_cluster_size,
            cluster_rank=mapping.moe_cluster_rank,
            enable_alltoall=True,
            tuner_num_tokens=max_num_tokens,
            tuner_top_k=top_k,
            activation_type=activation_type,
            use_deepseek_fp8_block_scale=False,
            use_w4_group_scaling=False,
            use_int8_woq_per_channel=False,
            use_mxfp8_act_scaling=False,
            min_latency_mode=False,
            use_fused_finalize=True,
        )[0]

        # COMBINE: Gather full results back to original GPUs
        moe_out = moe_out.view(mapping.moe_ep_size, runtime_max_tokens_per_rank, hidden_size)
        combined = moe_a2a.combine(moe_out, runtime_max_tokens_per_rank)

        return combined.view(x_shape)

    else:
        # EP WITH ALL-REDUCE PATH: Expert IDs are in LOCAL coordinates (from sharding.py),
        # routing weights for remote experts are zeroed, all_reduce is added after this op
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
    apply_routing_on_input: bool = False,
    enable_alltoall: bool = False,
    mapping_config: Optional[List[int]] = None,
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
    # Additional kwargs for consistency with torch_quant_fp8_moe
    apply_routing_on_input: bool = False,
    enable_alltoall: bool = False,
    mapping_config: Optional[List[int]] = None,
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
    apply_routing_on_input: bool = False,
    enable_alltoall: bool = False,
    mapping_config: Optional[List[int]] = None,
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
    # Additional kwargs for consistency with torch_quant_nvfp4_moe
    apply_routing_on_input: bool = False,
    enable_alltoall: bool = False,
    mapping_config: Optional[List[int]] = None,
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
    apply_routing_on_input: bool = False,
    enable_alltoall: bool = False,
    mapping_config: Optional[List[int]] = None,
) -> torch.Tensor:
    return torch.empty_like(x)
