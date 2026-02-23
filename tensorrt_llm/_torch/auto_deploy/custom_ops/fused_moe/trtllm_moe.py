# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def _nvfp4_quantize_scale_from_act_global(act_global_scale: torch.Tensor) -> torch.Tensor:
    """Convert act_global_scale to the scalar scale expected by fp4_quantize.

    The module path uses fc31_input_scale = 1/max(per-expert w1 input scales) for
    fp4_quantize. When act_global_scale is per-expert [E], we use 1/max to match.
    When it is already a scalar, we pass it through.
    Must be capture-safe: no .item() or CPU-GPU sync.
    """
    if act_global_scale.numel() > 1:
        inv_max = (1.0 / act_global_scale.max()).to(torch.float32)
        return inv_max.reshape(1)
    return act_global_scale.flatten()[:1].to(torch.float32).clone()


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
) -> torch.Tensor:
    """
    Execute MoE with all-to-all dispatch/combine pattern.

    Encapsulates the common all-to-all logic shared by the unquantized, FP8 and
    NVFP4 variants, calling ``torch.ops.trtllm.fused_moe`` directly rather than
    going through a caller-provided kernel closure.

    Args:
        x: 2-D input tensor ``(num_tokens, hidden_size)``.
            For unquantized / FP8: pass the (possibly quantized) flattened input.
            For NVFP4: pass the **bf16** flattened input (per-rank FP4 quantisation
            is performed after dispatch when *nvfp4_act_global_scale* is set).
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
        use_deepseek_fp8_block_scale=False,
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

    # Quantize the input using precomputed max scale.
    # Use the optimized CUDA kernel (same as PT backend's static_quantize path) instead of
    # Python-traced ops that compile to a slower Triton kernel.
    x_q_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x2d, fc1_act_scale)

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


@torch.library.custom_op("auto_deploy::trtllm_nvfp4_trtllm_gen_moe_fused", mutates_args=())
def trtllm_nvfp4_trtllm_gen_moe_fused(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights_fp4: torch.Tensor,
    fc2_expert_weights_fp4: torch.Tensor,
    fc1_weight_blockscale_fp8: torch.Tensor,
    fc2_weight_blockscale_fp8: torch.Tensor,
    fc1_act_global_scale: torch.Tensor,
    fc1_scale_c: torch.Tensor,
    fc1_alpha: torch.Tensor,
    fc2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    hidden_size: int,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
) -> torch.Tensor:
    """TensorRT-LLM TRTLLM-Gen NVFP4 MoE for SM100+ (Blackwell).

    This uses the optimized fp4_block_scale_moe_runner kernel which is specifically
    designed for SM100/SM103 architectures. It differs from the Cutlass-based
    trtllm_quant_nvfp4_moe_fused in that:
    - It requires shuffled weight format
    - It supports additional gpt-oss style parameters (bias, alpha, beta, limit)

    Computes (per expert):
        For gated_mlp:
            y = (act(x @ w1.T) * (x @ w3.T)) @ w2.T  # act := SiLU
        For mlp:
            y = act(x @ w1.T) @ w2.T                 # act := ReLU^2

    Parameters:
        x: BF16 input tensor of shape (B, H) or (B, S, H)
        selected_experts: Pre-computed expert indices (B*S, TOP_K)
        routing_weights: Pre-computed routing weights (B*S, TOP_K)
        fc1_expert_weights_fp4: Shuffled FP4 FC1 weights [E, 2*I, H/2] for gated_mlp
        fc2_expert_weights_fp4: Shuffled FP4 FC2 weights [E, H, I/2]
        fc1_weight_blockscale_fp8: Block scales for FC1 weights
        fc2_weight_blockscale_fp8: Block scales for FC2 weights
        fc1_act_global_scale: Global scale for FC1 activations
        fc1_scale_c: Scale for FC1 output quantization
        fc1_alpha: FC1 dequant scale
        fc2_alpha: FC2 dequant scale
        num_experts: Total number of experts
        top_k: Number of experts per token
        intermediate_size: MLP intermediate dimension (padded)
        hidden_size: Original hidden size (before padding)
        is_gated_mlp: True for gated_mlp (SwiGLU), False for mlp (ReLU2)
        act_fn: Activation function type

    Returns:
        Output tensor of shape (B, H) or (B, S, H)
    """
    _validate_mlp_style_and_act_fn(is_gated_mlp, act_fn)

    # Store original shape
    x_shape = x.shape
    x2d = x.view(-1, x_shape[-1])

    # Determine activation type for the kernel
    # 0 = Swiglu, 1 = Relu2
    act_type = 0 if is_gated_mlp else 1

    # Pad input if necessary (hidden_size must match weight dimension)
    padded_hidden_size = fc1_expert_weights_fp4.shape[-1] * 2  # *2 because FP4 is packed
    if x2d.shape[-1] < padded_hidden_size:
        x2d = torch.nn.functional.pad(x2d, (0, padded_hidden_size - x2d.shape[-1]))

    # Quantize input to FP4
    hidden_states_fp4, hidden_states_scale = torch.ops.trtllm.fp4_quantize(
        x2d, fc1_act_global_scale, TRTLLM_NVFP4_SCALING_VECTOR_SIZE, False, False
    )

    # Get number of local experts (for single GPU, this equals num_experts)
    local_num_experts = fc1_expert_weights_fp4.shape[0]
    local_expert_offset = 0

    # Routing parameters - use DeepSeekV3 routing with n_group=1 for external routing
    # This is the ONLY routing method that supports top_k > 10 (Nemotron Super v3 uses top_k=22)
    # When n_group=1, topk_group=1, it behaves like standard routing but supports higher top_k
    # Note: Nemotron uses routed_scaling_factor=5.0 in its router (noaux_tc_op)
    # The routing weights should already have this scaling applied, so we don't pass it here
    routing_method_type = int(RoutingMethodType.DeepSeekV3)  # = 2
    n_group = 1
    topk_group = 1
    routed_scaling_factor = 1.0  # Nemotron Super v3 uses routed_scaling_factor=5.0

    # Prepare topk tensors for external routing
    topk_ids = selected_experts.to(torch.int32)
    topk_weights = routing_weights.to(torch.bfloat16)

    # Call the TRTLLM-Gen kernel with external routing
    outputs = torch.ops.trtllm.fp4_block_scale_moe_runner(
        None,  # routing_logits (None for external routing)
        None,  # routing_bias (optional, for DeepSeek-V3)
        hidden_states_fp4,  # hidden_states (FP4 quantized)
        hidden_states_scale.view(torch.float8_e4m3fn),  # hidden_states_scale
        fc1_expert_weights_fp4,  # gemm1_weights
        fc1_weight_blockscale_fp8.view(torch.float8_e4m3fn),  # gemm1_weights_scale
        None,  # gemm1_bias
        None,  # gemm1_alpha (swiglu alpha)
        None,  # gemm1_beta (swiglu beta)
        None,  # gemm1_clamp_limit
        fc2_expert_weights_fp4,  # gemm2_weights
        fc2_weight_blockscale_fp8.view(torch.float8_e4m3fn),  # gemm2_weights_scale
        None,  # gemm2_bias
        fc1_scale_c,  # output1_scale_scalar
        fc1_alpha,  # output1_scale_gate_scalar
        fc2_alpha,  # output2_scale_scalar
        num_experts,  # num_experts
        top_k,  # top_k
        n_group,  # n_group
        topk_group,  # topk_group
        intermediate_size,  # intermediate_size
        local_expert_offset,  # local_expert_offset
        local_num_experts,  # local_num_experts
        routed_scaling_factor,  # routed_scaling_factor
        routing_method_type,  # routing_method_type
        True,  # do_finalize
        act_type,  # act_type (0=Swiglu, 1=Relu2)
        topk_weights,  # topk_weights (external routing)
        topk_ids,  # topk_ids (external routing)
    )

    final_hidden_states = outputs[0]

    # Slice output if it was padded
    if final_hidden_states.shape[1] > hidden_size:
        final_hidden_states = final_hidden_states[:, :hidden_size].contiguous()

    return final_hidden_states.view(x_shape)


@trtllm_nvfp4_trtllm_gen_moe_fused.register_fake
def trtllm_nvfp4_trtllm_gen_moe_fused_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights_fp4: torch.Tensor,
    fc2_expert_weights_fp4: torch.Tensor,
    fc1_weight_blockscale_fp8: torch.Tensor,
    fc2_weight_blockscale_fp8: torch.Tensor,
    fc1_act_global_scale: torch.Tensor,
    fc1_scale_c: torch.Tensor,
    fc1_alpha: torch.Tensor,
    fc2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    hidden_size: int,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
) -> torch.Tensor:
    _validate_mlp_style_and_act_fn(is_gated_mlp, act_fn)
    return torch.empty_like(x)
