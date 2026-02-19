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

from typing import List, Tuple

import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.quant import (
    TRTLLM_NVFP4_SCALING_VECTOR_SIZE,
)
from tensorrt_llm._torch.auto_deploy.utils.mapping_utils import deserialize_mapping
from tensorrt_llm._torch.distributed.moe_alltoall import MoeAlltoAll
from tensorrt_llm._torch.modules.fused_moe.routing import RoutingMethodType
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.mapping import Mapping


def _check_moe_alltoall(mapping_config: str, max_num_tokens: int) -> Tuple[Mapping | None, bool]:
    """Check if MoE all-to-all mode should be used and validate parameters.

    All-to-all is used when attention-DP is enabled and experts are sharded (EP > 1).

    Returns:
        (mapping, enable_alltoall) — mapping is None when mapping_config is empty.
    """
    mapping = deserialize_mapping(mapping_config) if mapping_config else None
    enable_alltoall = (
        mapping is not None and mapping.enable_attention_dp and mapping.moe_ep_size > 1
    )
    if enable_alltoall and max_num_tokens <= 0:
        raise ValueError("max_num_tokens must be > 0 when enable_alltoall is True")
    return mapping, enable_alltoall


def _run_moe_with_alltoall(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    activation_type: ActivationType,
    mapping: Mapping,
    max_num_tokens: int,
    fc1_expert_biases: torch.Tensor | None = None,
    fc2_expert_biases: torch.Tensor | None = None,
    nvfp4_act_global_scale: torch.Tensor | None = None,
    use_deepseek_fp8_block_scale: bool = False,
) -> torch.Tensor:
    """
    Execute MoE with all-to-all dispatch/combine pattern.

    Encapsulates the common all-to-all logic shared by the unquantized, FP8,
    NVFP4, and FineGrained FP8 (Hopper) variants, calling
    ``torch.ops.trtllm.fused_moe`` directly rather than going through a
    caller-provided kernel closure.

    Args:
        x: 2-D input tensor ``(num_tokens, hidden_size)``.
            For unquantized / FP8: pass the (possibly quantized) flattened input.
            For NVFP4: pass the **bf16** flattened input (per-rank FP4 quantisation
            is performed after dispatch when *nvfp4_act_global_scale* is set).
            For FineGrained FP8 (Hopper): pass the **bf16** flattened input
            (dynamic activation quant happens inside the kernel).
        selected_experts: Expert indices in GLOBAL coordinates ``(num_tokens, top_k)``.
        routing_weights: Routing weights ``(num_tokens, top_k)``.
        fc1_expert_weights: FC1 weight tensor (shape[0] = local expert count).
            For NVFP4 pass the packed ``int64`` view.
        fc2_expert_weights: FC2 weight tensor. For NVFP4 pass the packed ``int64`` view.
        output_dtype: Element type of the kernel output (bf16/fp16).  Used to size
            the combine workspace correctly when the dispatched input is quantised.
        quant_scales: Quantisation scale list expected by ``fused_moe``.
        activation_type: Activation function (``Swiglu``, ``Relu2``, …).
        mapping: ``Mapping`` object (already deserialised by the caller via
            ``_check_moe_alltoall``).
        max_num_tokens: Upper-bound token count per rank.
        fc1_expert_biases: Optional FC1 biases (currently always ``None``).
        fc2_expert_biases: Optional FC2 biases (currently always ``None``).
        nvfp4_act_global_scale: When set, the dispatched bf16 input is quantised to
            NVFP4 per-rank before the kernel call (``torch.ops.trtllm.fp4_quantize``).
        use_deepseek_fp8_block_scale: When True, enables DeepSeek FP8 block scale
            mode in ``fused_moe``. Used by FineGrained FP8 on Hopper where
            activation quantization happens dynamically inside the kernel.

    Returns:
        2-D output tensor ``(num_tokens, hidden_size)`` — the caller reshapes to the
        original input shape.
    """
    top_k = selected_experts.shape[1]
    hidden_size = x.shape[-1]

    # Expert weights are sharded — shape[0] gives LOCAL expert count
    local_num_experts = fc1_expert_weights.shape[0]
    global_num_experts = local_num_experts * mapping.moe_ep_size

    # Workspace must be sized for the LARGEST element type used by dispatch or combine.
    # The input x may be quantized (fp8/fp4), but combine outputs in the model dtype
    # (bf16/fp16). We always pass the model dtype so the combine buffer is large enough.
    workspace_size = MoeAlltoAll.calculate_required_workspace_size(
        mapping.moe_ep_size, top_k, max_num_tokens, hidden_size, output_dtype
    )

    # We need runtime_max_tokens_per_rank = max(tokens across all EP ranks).
    # An NCCL all_reduce cannot run inside CUDA-graph capture, so we conservatively
    # use max_num_tokens (the config-level upper bound) as an over-approximation.
    # This causes the dispatch to allocate larger recv buffers (padded with invalid
    # tokens that are skipped by the kernel), trading memory for correctness.
    runtime_max_tokens_per_rank = max_num_tokens

    # Build MoeAlltoAll (num_slots = num_experts without EPLB load balancing)
    moe_a2a = MoeAlltoAll(
        mapping=mapping,
        max_num_tokens=max_num_tokens,
        top_k=top_k,
        num_slots=global_num_experts,  # No EPLB: num_slots == num_experts
        workspace_size_per_rank=workspace_size,
        num_experts=None,  # None = EPLB disabled
    )

    invalid_expert_id = global_num_experts

    # Pad inputs to runtime_max_tokens_per_rank so all ranks send the same number
    # of rows through dispatch.  Padding expert IDs must route to a VALID rank
    # (the local rank's first expert), otherwise the dispatch kernel computes an
    # out-of-bounds target rank.  Zero routing weights ensure padding tokens
    # contribute nothing to the output.
    local_num_tokens = x.shape[0]
    pad_expert_id = mapping.moe_ep_rank * local_num_experts  # routes to local rank
    pad_size = runtime_max_tokens_per_rank - local_num_tokens
    if pad_size > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))  # pad rows with zeros
        selected_experts = torch.nn.functional.pad(
            selected_experts, (0, 0, 0, pad_size), value=pad_expert_id
        )
        routing_weights = torch.nn.functional.pad(routing_weights, (0, 0, 0, pad_size))

    # Build payload list: x, selected_experts, routing_weights
    payloads = [x.contiguous(), selected_experts.contiguous(), routing_weights.contiguous()]

    # DISPATCH: Route tokens to correct GPUs based on global expert IDs.
    recv_results = moe_a2a.dispatch(
        selected_experts,
        payloads,
        runtime_max_tokens_per_rank,
        invalid_token_expert_id=invalid_expert_id,
        expert_id_payload_index=1,
    )

    dispatched_x = recv_results[0].reshape(-1, hidden_size)
    dispatched_selected = recv_results[1].reshape(-1, top_k)
    dispatched_weights = recv_results[2].reshape(-1, top_k)

    # NVFP4: quantise the dispatched bf16 input to FP4 per-rank
    input_sf_kwargs: dict = {}
    if nvfp4_act_global_scale is not None:
        dispatched_x, input_sf = torch.ops.trtllm.fp4_quantize(
            dispatched_x, nvfp4_act_global_scale, TRTLLM_NVFP4_SCALING_VECTOR_SIZE
        )
        dispatched_x = dispatched_x.view(torch.long)
        input_sf_kwargs["input_sf"] = input_sf

    # Call the fused MoE kernel with all-to-all parameters
    moe_out = torch.ops.trtllm.fused_moe(
        dispatched_x,
        dispatched_selected,
        dispatched_weights,
        fc1_expert_weights=fc1_expert_weights,
        fc1_expert_biases=fc1_expert_biases,
        fc2_expert_weights=fc2_expert_weights,
        fc2_expert_biases=fc2_expert_biases,
        output_dtype=output_dtype,
        quant_scales=quant_scales,
        tp_size=mapping.moe_tp_size,
        tp_rank=mapping.moe_tp_rank,
        ep_size=mapping.moe_ep_size,
        ep_rank=mapping.moe_ep_rank,
        cluster_size=mapping.moe_cluster_size,
        cluster_rank=mapping.moe_cluster_rank,
        enable_alltoall=True,
        tuner_num_tokens=dispatched_x.shape[0],
        tuner_top_k=top_k,
        activation_type=activation_type,
        use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        use_w4_group_scaling=False,
        use_int8_woq_per_channel=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        use_fused_finalize=True,
        **input_sf_kwargs,
    )[0]

    # COMBINE: Gather full results back to original GPUs.
    # runtime_max_tokens_per_rank is an over-approximation (max_num_tokens),
    # so the combined result has more rows than the original input.
    # Slice back to local_num_tokens (captured before padding) before returning.
    moe_out = moe_out.view(mapping.moe_ep_size, runtime_max_tokens_per_rank, hidden_size)
    combined = moe_a2a.combine(moe_out, runtime_max_tokens_per_rank)
    return combined[:local_num_tokens]


def _run_finegrained_moe_with_alltoall_blackwell(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    is_gated_mlp: bool,
    mapping: Mapping,
    max_num_tokens: int,
) -> torch.Tensor:
    """Execute FineGrained FP8 MoE with all-to-all on Blackwell (SM100+).

    Wraps ``fp8_block_scale_moe_runner`` with alltoall dispatch/combine.
    Activation quantization (bf16 -> FP8) is performed per-rank after dispatch.

    Args:
        x: 2-D bf16/fp16 input tensor ``(num_tokens, hidden_size)``.
        selected_experts: Expert indices in GLOBAL coordinates ``(num_tokens, top_k)``.
        routing_weights: Routing weights ``(num_tokens, top_k)``.
        fc1_expert_weights: FC1 FP8 weights ``[local_E, 2*I, H]`` or ``[local_E, I, H]``.
        fc2_expert_weights: FC2 FP8 weights ``[local_E, H, I]``.
        fc1_weight_scale: FC1 block weight scales.
        fc2_weight_scale: FC2 block weight scales.
        is_gated_mlp: Whether gated MLP is used.
        mapping: ``Mapping`` object with EP configuration.
        max_num_tokens: Upper-bound token count per rank.

    Returns:
        2-D output tensor ``(num_tokens, hidden_size)``.
    """
    top_k = selected_experts.shape[1]
    hidden_size = x.shape[-1]

    local_num_experts = fc1_expert_weights.shape[0]
    global_num_experts = local_num_experts * mapping.moe_ep_size
    intermediate_size = (
        fc1_expert_weights.shape[1] // 2 if is_gated_mlp else fc1_expert_weights.shape[1]
    )

    workspace_size = MoeAlltoAll.calculate_required_workspace_size(
        mapping.moe_ep_size, top_k, max_num_tokens, hidden_size, x.dtype
    )

    runtime_max_tokens_per_rank = max_num_tokens

    moe_a2a = MoeAlltoAll(
        mapping=mapping,
        max_num_tokens=max_num_tokens,
        top_k=top_k,
        num_slots=global_num_experts,
        workspace_size_per_rank=workspace_size,
        num_experts=None,
    )

    invalid_expert_id = global_num_experts

    local_num_tokens = x.shape[0]
    pad_expert_id = mapping.moe_ep_rank * local_num_experts
    pad_size = runtime_max_tokens_per_rank - local_num_tokens
    if pad_size > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        selected_experts = torch.nn.functional.pad(
            selected_experts, (0, 0, 0, pad_size), value=pad_expert_id
        )
        routing_weights = torch.nn.functional.pad(routing_weights, (0, 0, 0, pad_size))

    payloads = [x.contiguous(), selected_experts.contiguous(), routing_weights.contiguous()]

    recv_results = moe_a2a.dispatch(
        selected_experts,
        payloads,
        runtime_max_tokens_per_rank,
        invalid_token_expert_id=invalid_expert_id,
        expert_id_payload_index=1,
    )

    dispatched_x = recv_results[0].reshape(-1, hidden_size)
    dispatched_selected = recv_results[1].reshape(-1, top_k)
    dispatched_weights = recv_results[2].reshape(-1, top_k)

    # Quantize dispatched bf16 input to FP8 per-rank
    x_fp8, x_sf = torch.ops.trtllm.fp8_quantize_1x128(dispatched_x)

    routing_weights_bf16 = dispatched_weights.to(torch.bfloat16).contiguous()
    fc1_weight_scale_f32 = fc1_weight_scale.to(torch.float32).contiguous()
    fc2_weight_scale_f32 = fc2_weight_scale.to(torch.float32).contiguous()

    local_expert_offset = mapping.moe_ep_rank * local_num_experts

    moe_out = torch.ops.trtllm.fp8_block_scale_moe_runner(
        None,  # routing_logits
        None,  # routing_bias
        x_fp8,
        x_sf,
        fc1_expert_weights.contiguous(),
        fc1_weight_scale_f32,
        fc2_expert_weights.contiguous(),
        fc2_weight_scale_f32,
        global_num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        None,  # routed_scaling_factor
        RoutingMethodType.Renormalize,
        topk_weights=routing_weights_bf16,
        topk_ids=dispatched_selected,
    )

    moe_out = moe_out.view(mapping.moe_ep_size, runtime_max_tokens_per_rank, hidden_size)
    combined = moe_a2a.combine(moe_out, runtime_max_tokens_per_rank)
    return combined[:local_num_tokens]


@torch.library.custom_op("auto_deploy::trtllm_moe_fused", mutates_args=())
def trtllm_moe_fused(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
    mapping_config: str = "",
    max_num_tokens: int = 0,
    apply_routing_on_input: bool = False,
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

    mapping, enable_alltoall = _check_moe_alltoall(mapping_config, max_num_tokens)

    if enable_alltoall:
        return _run_moe_with_alltoall(
            x=x,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            fc1_expert_weights=w3_w1_stacked_weight,
            fc2_expert_weights=w2_stacked_weight,
            output_dtype=x.dtype,
            quant_scales=quant_scales,
            activation_type=activation_type,
            mapping=mapping,
            max_num_tokens=max_num_tokens,
        ).view(x_shape)

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
    mapping_config: str = "",
    max_num_tokens: int = 0,
    apply_routing_on_input: bool = False,
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
    mapping_config: str = "",
    max_num_tokens: int = 0,
    apply_routing_on_input: bool = False,
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
        fc1_act_scale: FC1 activation scalar (scalar)
        fc1_dequant_scale: FC1 dequant scale [E]
        fc2_act_scale_reciprocal: FC2 activation scale reciprocal (scalar)
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

    # Quantize the input using precomputed max scale
    x_q_fp8 = _quantize_fp8(x2d, fc1_act_scale)

    # Prepare quant_scales for TensorRT-LLM (Cutlass) FP8 format:
    # [fc1_dequant_scale, fc2_act_scale_reciprocal, fc2_dequant_scale, gemm1_input_dequant_scale]
    # These are precomputed in `fused_moe` transform:
    # - fc1_dequant_scale: w1_weight_scale * max(w1_input_scale) [E]
    # - fc2_act_scale_reciprocal: 1 / max(w2_input_scale) (scalar)
    # - fc2_dequant_scale: w2_weight_scale * max(w2_input_scale) [E]
    # - fc1_act_scale: max(w1_input_scale) (scalar)

    assert fc1_dequant_scale.ndim == 1, "fc1_dequant_scale must be 1D"
    assert fc2_dequant_scale.ndim == 1, "fc2_dequant_scale must be 1D"
    quant_scales = [
        fc1_dequant_scale,
        fc2_act_scale_reciprocal,
        fc2_dequant_scale,
        fc1_act_scale,
    ]

    # Ensure correct dtypes and contiguous tensors
    selected_experts = selected_experts.int().contiguous()
    routing_weights = routing_weights.to(torch.float32).contiguous()

    mapping, enable_alltoall = _check_moe_alltoall(mapping_config, max_num_tokens)

    if enable_alltoall:
        return _run_moe_with_alltoall(
            x=x_q_fp8,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            fc1_expert_weights=fc1_expert_weights,
            fc2_expert_weights=fc2_expert_weights.contiguous(),
            output_dtype=x.dtype,  # kernel outputs in model dtype (bf16/fp16), not fp8
            quant_scales=quant_scales,
            activation_type=act_fn,
            mapping=mapping,
            max_num_tokens=max_num_tokens,
        ).view(x_shape)

    # EP WITH ALL-REDUCE PATH: Expert IDs are in LOCAL coordinates.
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
    mapping_config: str = "",
    max_num_tokens: int = 0,
    apply_routing_on_input: bool = False,
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
    mapping_config: str = "",
    max_num_tokens: int = 0,
    apply_routing_on_input: bool = False,
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

    mapping, enable_alltoall = _check_moe_alltoall(mapping_config, max_num_tokens)

    if enable_alltoall:
        # Dispatch bf16 input (not FP4-quantized) to avoid padding issues with packed
        # FP4 tensors and blockscales. Per-rank FP4 quantisation happens inside
        # _run_moe_with_alltoall (triggered by nvfp4_act_global_scale).
        return _run_moe_with_alltoall(
            x=x.view(-1, x.shape[-1]),
            selected_experts=selected_experts.to(torch.int32),
            routing_weights=routing_weights.to(torch.float32),
            fc1_expert_weights=fc1_expert_weights_fp4.view(torch.long),
            fc2_expert_weights=fc2_expert_weights_fp4.view(torch.long),
            output_dtype=x.dtype,  # kernel outputs in model dtype (bf16/fp16), not fp4
            quant_scales=quant_scales,
            activation_type=act_fn,
            mapping=mapping,
            max_num_tokens=max_num_tokens,
            nvfp4_act_global_scale=fc1_act_global_scale,
        ).view(x.shape)

    # EP WITH ALL-REDUCE PATH: Expert IDs are in LOCAL coordinates.
    # FP4 quantisation happens here (before the kernel) for the non-alltoall path.
    if x.dtype in (torch.float16, torch.bfloat16):
        x_q_fp4, input_blockscale = torch.ops.trtllm.fp4_quantize(
            x, fc1_act_global_scale, TRTLLM_NVFP4_SCALING_VECTOR_SIZE
        )
    else:
        x_q_fp4 = x
        input_blockscale = None

    return torch.ops.trtllm.fused_moe(
        x_q_fp4.view(torch.long),
        selected_experts.to(torch.int32),
        routing_weights.to(torch.float32),
        # Groups of 16 FP4 weight elements are packed as a single int64 element (see isNvfp4Quant in moeOp.cpp)
        fc1_expert_weights=fc1_expert_weights_fp4.view(torch.long),
        fc1_expert_biases=None,
        fc2_expert_weights=fc2_expert_weights_fp4.view(torch.long),
        fc2_expert_biases=None,
        output_dtype=x.dtype,
        quant_scales=quant_scales,
        input_sf=input_blockscale,
        activation_type=act_fn,
    )[0].view(x.shape)


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
    mapping_config: str = "",
    max_num_tokens: int = 0,
    apply_routing_on_input: bool = False,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::trtllm_quant_finegrained_fp8_moe_fused", mutates_args=())
def trtllm_quant_finegrained_fp8_moe_fused(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
    mapping_config: str = "",
    max_num_tokens: int = 0,
    apply_routing_on_input: bool = False,
) -> torch.Tensor:
    """TensorRT-LLM Cutlass FP8 Block Scale MoE for FineGrainedFP8 format.

    This op uses the DeepSeek FP8 block scale format which is compatible with FineGrained FP8.
    Activations are quantized dynamically at runtime (no pre-computed activation scales).

    Computes (per expert):
        For gated_mlp:
            y = (act(x @ w1.T) * (x @ w3.T)) @ w2.T  # act := SiLU
        For mlp:
            y = act(x @ w1.T) @ w2.T                 # act := ReLU^2

    Notes:
        - FC1 implements: fc1_output = (act(x @ w1.T) * (x @ w3.T)) or fc1_output = act(x @ w1.T)
        - FC2 implements: fc2_output = fc1_output @ w2.T
        - FC1 weights are concatenated w3 and w1 if gated_mlp, otherwise w1
        - Uses per-block weight scales (128x128 blocks)
        - On Hopper (SM90): Activation quantization happens dynamically inside the kernel
        - On Blackwell (SM100+): Uses fp8_block_scale_moe_runner with external activation quantization

    Parameters:
        x: BF16/FP16 input tensor of shape (B, H) or (B, S, H)
        selected_experts: Expert indices (B*S, TOP_K)
        routing_weights: Routing weights (B*S, TOP_K)
        fc1_expert_weights: FC1 FP8 weights [E, 2*I, H] for gated_mlp, [E, I, H] for mlp
        fc2_expert_weights: FC2 FP8 weights [E, H, I]
        fc1_weight_scale: FC1 block weight scales [E, 2*I/128, H/128] or [E, I/128, H/128]
        fc2_weight_scale: FC2 block weight scales [E, H/128, I/128]
        is_gated_mlp: True for gated_mlp, False for mlp
        act_fn: ActivationType.Silu for gated_mlp, ActivationType.Relu2 for mlp
        mapping_config: Serialized Mapping config for distributed all-to-all
        max_num_tokens: Maximum tokens for workspace allocation (all-to-all mode)
        apply_routing_on_input: If True, apply routing weights to input before MLP

    Returns:
        Output tensor of shape (B, H) or (B, S, H)
    """
    _validate_mlp_style_and_act_fn(is_gated_mlp, act_fn)
    act_fn = ActivationType.Swiglu if act_fn == ActivationType.Silu else act_fn

    x_shape = x.shape
    x2d = x.view(-1, x_shape[-1])

    selected_experts = selected_experts.int().contiguous()
    routing_weights = routing_weights.to(torch.float32).contiguous()

    mapping, enable_alltoall = _check_moe_alltoall(mapping_config, max_num_tokens)

    if enable_alltoall:
        if is_sm_100f():
            return _run_finegrained_moe_with_alltoall_blackwell(
                x=x2d,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
                fc1_expert_weights=fc1_expert_weights,
                fc2_expert_weights=fc2_expert_weights,
                fc1_weight_scale=fc1_weight_scale,
                fc2_weight_scale=fc2_weight_scale,
                is_gated_mlp=is_gated_mlp,
                mapping=mapping,
                max_num_tokens=max_num_tokens,
            ).view(x_shape)
        else:
            quant_scales = (fc1_weight_scale, fc2_weight_scale)
            return _run_moe_with_alltoall(
                x=x2d,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
                fc1_expert_weights=fc1_expert_weights.contiguous(),
                fc2_expert_weights=fc2_expert_weights.contiguous(),
                output_dtype=x.dtype,
                quant_scales=quant_scales,
                activation_type=act_fn,
                mapping=mapping,
                max_num_tokens=max_num_tokens,
                use_deepseek_fp8_block_scale=True,
            ).view(x_shape)

    # EP WITH ALL-REDUCE PATH: Expert IDs are in LOCAL coordinates (from sharding.py),
    # routing weights for remote experts are zeroed, all_reduce is added after this op
    if is_sm_100f():
        # --- Blackwell (SM100+) Path ---
        x_fp8, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x2d)

        num_experts = fc1_expert_weights.shape[0]
        top_k = selected_experts.shape[-1]
        intermediate_size = (
            fc1_expert_weights.shape[1] // 2 if is_gated_mlp else fc1_expert_weights.shape[1]
        )

        routing_weights_bf16 = routing_weights.to(torch.bfloat16).contiguous()
        fc1_weight_scale_f32 = fc1_weight_scale.to(torch.float32).contiguous()
        fc2_weight_scale_f32 = fc2_weight_scale.to(torch.float32).contiguous()

        output = torch.ops.trtllm.fp8_block_scale_moe_runner(
            None,  # routing_logits
            None,  # routing_bias
            x_fp8,
            x_sf,
            fc1_expert_weights.contiguous(),
            fc1_weight_scale_f32,
            fc2_expert_weights.contiguous(),
            fc2_weight_scale_f32,
            num_experts,
            top_k,
            None,  # n_group
            None,  # topk_group
            intermediate_size,
            0,  # local_expert_offset
            num_experts,  # local_num_experts
            None,  # routed_scaling_factor
            RoutingMethodType.Renormalize,
            topk_weights=routing_weights_bf16,
            topk_ids=selected_experts,
        )

        return output.view(x_shape)
    else:
        # --- Hopper (SM90) Path ---
        quant_scales = (fc1_weight_scale, fc2_weight_scale)

        output = torch.ops.trtllm.fused_moe(
            x2d,
            selected_experts,
            routing_weights,
            fc1_expert_weights=fc1_expert_weights.contiguous(),
            fc1_expert_biases=None,
            fc2_expert_weights=fc2_expert_weights.contiguous(),
            fc2_expert_biases=None,
            output_dtype=x.dtype,
            quant_scales=quant_scales,
            activation_type=act_fn,
            use_deepseek_fp8_block_scale=True,
        )

        return output[0].view(x_shape)


@trtllm_quant_finegrained_fp8_moe_fused.register_fake
def trtllm_quant_finegrained_fp8_moe_fused_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
    mapping_config: str = "",
    max_num_tokens: int = 0,
    apply_routing_on_input: bool = False,
) -> torch.Tensor:
    _validate_mlp_style_and_act_fn(is_gated_mlp, act_fn)
    return torch.empty_like(x)
