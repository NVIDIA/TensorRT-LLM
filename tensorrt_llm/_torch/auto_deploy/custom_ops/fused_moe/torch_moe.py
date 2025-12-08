from typing import Callable, List

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.auto_deploy.enums import (
    ActivationFunction,
    MLPStyle,
    WeightsFormat,
    WeightsFusion,
    act_fn_from_str,
    mlp_style_from_str,
    weights_format_from_str,
    weights_fusion_from_str,
)


def _resolve_activation(act_fn: ActivationFunction) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns an elementwise activation callable matching the given activation function.
    """
    if act_fn == ActivationFunction.SILU:
        return F.silu
    elif act_fn == ActivationFunction.RELU2:

        def relu2(x: torch.Tensor) -> torch.Tensor:
            return torch.square(F.relu(x))

        return relu2
    else:
        raise ValueError(f"Unsupported activation '{act_fn.value}'.")


def _template_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    mlps: List[Callable[[torch.Tensor], torch.Tensor]],
    apply_routing_on_input: bool = False,
) -> torch.Tensor:
    """Generic MoE template with token-level dispatch, routing tokens to expert MLPs.

    Args:
        x: Input tensor
        selected_experts: Expert indices for each token
        routing_weights: Routing weights for each expert
        mlps: List of MLP functions
        apply_routing_on_input: If True, multiply routing weights with INPUT before MLP (BMM-based pattern).
                                This means: silu(input * routing_weight)
                                If False, multiply routing weights with OUTPUT after MLP (standard pattern).
                                This means: silu(input) * routing_weight
    """
    x_shape = x.shape
    hidden_dim = x_shape[-1]
    x = x.view(-1, hidden_dim)
    num_experts = len(mlps)

    final_hidden_states = torch.zeros_like(x)
    valid_mask = (selected_experts >= 0) & (selected_experts < num_experts)
    # For out-of-range indices, set them to num_experts
    selected_experts_fixed = torch.where(
        valid_mask, selected_experts, torch.full_like(selected_experts, num_experts)
    )
    # Create one-hot encoding with an extra class.
    # NOTE: `F.one_hot` only accepts `LongTensor` as an input, and will throw an error if the tensor is of another
    # dtype, even if `torch.int32`.
    one_hot = F.one_hot(selected_experts_fixed.long(), num_classes=num_experts + 1)
    expert_mask = one_hot[..., :num_experts].permute(2, 1, 0)

    for expert_idx in range(num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        tokens_for_this_expert = x[None, top_x].reshape(-1, hidden_dim)
        if not tokens_for_this_expert.shape[0]:
            continue  # input of shape [0, hidden_dim] breaks fp4 kernel

        if apply_routing_on_input:
            # INPUT-SIDE routing (BMM-based pattern): multiply routing weights with INPUT
            # Result: silu(input * routing_weight)
            scaled_input = tokens_for_this_expert * routing_weights[top_x, idx, None]
            expert_out = mlps[expert_idx](scaled_input)
            current_hidden_states = expert_out
        else:
            # OUTPUT-SIDE routing (standard pattern): multiply routing weights with OUTPUT
            # Result: silu(input) * routing_weight
            expert_out = mlps[expert_idx](tokens_for_this_expert)
            current_hidden_states = expert_out * routing_weights[top_x, idx, None]

        final_hidden_states.index_add_(
            0, top_x, current_hidden_states.to(final_hidden_states.dtype)
        )
    return final_hidden_states.view(x_shape)


@torch.library.custom_op("auto_deploy::torch_moe", mutates_args=())
def torch_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    weights_1: List[torch.Tensor],
    weights_2: List[torch.Tensor],
    weights_3: List[torch.Tensor],
    weights_format: str = WeightsFormat.PER_EXPERT.value,
    weights_fusion: str = WeightsFusion.GATE_UP_DOWN.value,
    mlp_style: str = MLPStyle.GATED_MLP.value,
    act_fn: str = ActivationFunction.SILU.value,
    apply_routing_on_input: bool = False,
) -> torch.Tensor:
    """
    Mixture-of-Experts (MoE) operator with token-level routing and dispatch.

    Supports various MoE architectures (Mixtral, DeepSeek, Llama4, NemotronH, etc.)
    through flexible weight format and MLP style parameters.

    Uses opaque weight parameters (weights_1, weights_2, weights_3) whose interpretation
    depends on weights_format, weights_fusion, and mlp_style parameters.

    WEIGHT INTERPRETATION:
    ======================

    format="per_expert" + fusion="w1_w2_w3_separate" + style="gated_mlp":
        Gated MLP with separate weights in storage order: w1, w2, w3
        Computation: output = w2( act(w1(x)) * w3(x) )
        Where: w1=gate_proj, w2=down_proj, w3=up_proj

        weights_1: List of w1 (gate) weights, each [intermediate_size, hidden_size]
        weights_2: List of w2 (down) weights, each [hidden_size, intermediate_size]
        weights_3: List of w3 (up) weights, each [intermediate_size, hidden_size]

        Note: PyTorch Linear weight shape is [out_features, in_features]
              x [B, H] @ w1.T [I, H] -> [B, I]  (gate projection)
              x [B, H] @ w3.T [I, H] -> [B, I]  (up projection)
              act(gate) * up -> [B, I]          (element-wise gating)
              gated [B, I] @ w2.T [H, I] -> [B, H]  (down projection)

        Example (Mixtral/Llama-style models):
            weights_1 = [expert.gate_proj.weight for expert in model.experts]  # w1 (gate): [I, H]
            weights_2 = [expert.down_proj.weight for expert in model.experts]  # w2 (down): [H, I]
            weights_3 = [expert.up_proj.weight for expert in model.experts]    # w3 (up): [I, H]

    format="per_expert" + fusion="w3w1_w2" + style="gated_mlp":
        weights_1: List of fused [w3, w1], each [2*intermediate_size, hidden_size]
        weights_2: List of w2, each [hidden_size, intermediate_size]
        weights_3: [] (unused, must be empty)

        Example:
            weights_1 = [expert.w3_w1_fused for expert in model.experts]  # [2*I, H]
            weights_2 = [expert.down_proj.weight for expert in model.experts]  # [H, I]
            weights_3 = []

    format="per_expert" + style="mlp":
        weights_1: List of w1, each [intermediate_size, hidden_size]
        weights_2: List of w2, each [hidden_size, intermediate_size]
        weights_3: [] (unused, must be empty)
        Note: weights_fusion is ignored for mlp style

        Example:
            weights_1 = [expert.up_proj.weight for expert in model.experts]  # [I, H]
            weights_2 = [expert.down_proj.weight for expert in model.experts]  # [H, I]
            weights_3 = []

    format="stacked" + fusion="w1_w2_w3_separate" + style="gated_mlp":
        weights_1: Single-element list [w1 stacked], shape [num_experts, intermediate_size, hidden_size]
        weights_2: Single-element list [w2 stacked], shape [num_experts, hidden_size, intermediate_size]
        weights_3: Single-element list [w3 stacked], shape [num_experts, intermediate_size, hidden_size]

        Example:
            weights_1 = [model.w1_stacked]  # [E, I, H]
            weights_2 = [model.w2_stacked]  # [E, H, I]
            weights_3 = [model.w3_stacked]  # [E, I, H]

    format="stacked" + fusion="w3w1_w2" + style="gated_mlp":
        weights_1: Single-element list [w3_w1 fused and stacked], shape [num_experts, 2*intermediate_size, hidden_size]
        weights_2: Single-element list [w2 stacked], shape [num_experts, hidden_size, intermediate_size]
        weights_3: [] (unused, must be empty)

        Example:
            weights_1 = [model.w3_w1_stacked]  # [E, 2*I, H]
            weights_2 = [model.w2_stacked]     # [E, H, I]
            weights_3 = []

    Parameters:
        x: Input tensor of shape (B, H) or (B, S, H)
        selected_experts: Expert indices, shape (B, TOP_K) or (B*S, TOP_K)
        routing_weights: Routing weights, shape (B, TOP_K) or (B*S, TOP_K)
        weights_1: First weight tensor(s) - see WEIGHT INTERPRETATION
        weights_2: Second weight tensor(s) - see WEIGHT INTERPRETATION
        weights_3: Third weight tensor(s) - see WEIGHT INTERPRETATION
        weights_format: "per_expert" (default) or "stacked"
        weights_fusion: "w1_w2_w3_separate" (default), "w3w1_w2", or "w1w3_w2" (only for gated_mlp)
        mlp_style: "gated_mlp" (default) or "mlp"
        act_fn: "silu" (default) or "relu2"
        apply_routing_on_input:
            - False (default): routing applied to output
            - True (Llama4): routing applied to input
    Returns:
        Output tensor with same shape as input x
    """
    # Convert string parameters to enums
    weights_format_enum = weights_format_from_str(weights_format)
    weights_fusion_enum = weights_fusion_from_str(weights_fusion)
    mlp_style_enum = mlp_style_from_str(mlp_style)
    act_fn_enum = act_fn_from_str(act_fn)
    act_fn_callable = _resolve_activation(act_fn_enum)

    # Validate fusion parameter only applies to gated_mlp
    if mlp_style_enum == MLPStyle.MLP and weights_fusion_enum != WeightsFusion.GATE_UP_DOWN:
        raise ValueError(
            f"weights_fusion='{weights_fusion}' only applies to gated_mlp. "
            f"For mlp style, use weights_fusion='w1_w2_w3_separate'."
        )

    # Dispatch based on combination of format + fusion + style
    if weights_format_enum == WeightsFormat.STACKED:
        # === STACKED FORMAT ===
        if mlp_style_enum == MLPStyle.GATED_MLP:
            if weights_fusion_enum == WeightsFusion.UPGATE_DOWN:
                # STACKED + W3W1_W2 + GATED_MLP: weights_1=[w3_w1 E,2*I,H], weights_2=[w2 E,H,I], weights_3=[]
                if len(weights_1) != 1 or weights_1[0].ndim != 3:
                    raise ValueError(
                        f"stacked+w3w1_w2+gated_mlp: weights_1 must be [w3_w1_stacked] with shape [E,2*I,H]. "
                        f"Got {len(weights_1)} elements{', shape: ' + str(weights_1[0].shape) if weights_1 else ''}"
                    )
                if len(weights_2) != 1 or weights_2[0].ndim != 3:
                    raise ValueError(
                        f"stacked+w3w1_w2+gated_mlp: weights_2 must be [w2_stacked] with shape [E,H,I]. "
                        f"Got {len(weights_2)} elements{', shape: ' + str(weights_2[0].shape) if weights_2 else ''}"
                    )
                if len(weights_3) > 0:
                    raise ValueError(
                        f"stacked+w3w1_w2+gated_mlp: weights_3 must be empty []. Got {len(weights_3)} elements."
                    )

                w3_w1_stacked = weights_1[0]  # [E, 2*I, H]
                w2_stacked = weights_2[0]  # [E, H, I]

                if w3_w1_stacked.shape[0] != w2_stacked.shape[0]:
                    raise ValueError(
                        f"Expert count mismatch: weights_1 has {w3_w1_stacked.shape[0]}, "
                        f"weights_2 has {w2_stacked.shape[0]} experts"
                    )

                # Extract per-expert slices and create MLPs
                def make_mlp(i: int):
                    w3_w1 = w3_w1_stacked[i]  # [2*I, H] - ordered as [w3, w1]
                    intermediate_size = w3_w1.shape[0] // 2
                    w3 = w3_w1[:intermediate_size, :]  # [I, H]
                    w1 = w3_w1[intermediate_size:, :]  # [I, H]
                    w2 = w2_stacked[i]  # [H, I]
                    weight_dtype = w1.dtype
                    return lambda inp: F.linear(
                        act_fn_callable(F.linear(inp.to(weight_dtype), w1))
                        * F.linear(inp.to(weight_dtype), w3),
                        w2,
                    )

                mlps = [make_mlp(i) for i in range(w3_w1_stacked.shape[0])]

            elif weights_fusion_enum == WeightsFusion.GATE_UP_DOWN:
                # STACKED + W1_W2_W3_SEPARATE + GATED_MLP:
                # weights_1=[w1 E,I,H], weights_2=[w2 E,H,I], weights_3=[w3 E,I,H]
                if len(weights_1) != 1 or weights_1[0].ndim != 3:
                    raise ValueError(
                        f"stacked+w1_w2_w3_separate+gated_mlp: weights_1 must be [w1_stacked] with shape [E,I,H]. "
                        f"Got {len(weights_1)} elements{', shape: ' + str(weights_1[0].shape) if weights_1 else ''}"
                    )
                if len(weights_2) != 1 or weights_2[0].ndim != 3:
                    raise ValueError(
                        f"stacked+w1_w2_w3_separate+gated_mlp: weights_2 must be [w2_stacked] with shape [E,H,I]. "
                        f"Got {len(weights_2)} elements{', shape: ' + str(weights_2[0].shape) if weights_2 else ''}"
                    )
                if len(weights_3) != 1 or weights_3[0].ndim != 3:
                    raise ValueError(
                        f"stacked+w1_w2_w3_separate+gated_mlp: weights_3 must be [w3_stacked] with shape [E,I,H]. "
                        f"Got {len(weights_3)} elements{', shape: ' + str(weights_3[0].shape) if weights_3 else ''}"
                    )

                w1_stacked = weights_1[0]  # [E, I, H]
                w2_stacked = weights_2[0]  # [E, H, I]
                w3_stacked = weights_3[0]  # [E, I, H]

                num_experts = w1_stacked.shape[0]
                if w2_stacked.shape[0] != num_experts or w3_stacked.shape[0] != num_experts:
                    raise ValueError(
                        f"Expert count mismatch: weights_1={w1_stacked.shape[0]}, "
                        f"weights_2={w2_stacked.shape[0]}, weights_3={w3_stacked.shape[0]}"
                    )

                # Extract per-expert slices and create MLPs
                def make_mlp(i: int):
                    w1 = w1_stacked[i]  # [I, H]
                    w2 = w2_stacked[i]  # [H, I]
                    w3 = w3_stacked[i]  # [I, H]
                    return lambda inp: F.linear(
                        act_fn_callable(F.linear(inp, w1)) * F.linear(inp, w3), w2
                    )

                mlps = [make_mlp(i) for i in range(num_experts)]

        elif mlp_style_enum == MLPStyle.MLP:
            # STACKED + MLP: weights_1=[w_up E,I,H], weights_2=[w_down E,H,I], weights_3=[]
            # (fusion doesn't apply to mlp style)
            if len(weights_1) != 1 or weights_1[0].ndim != 3:
                raise ValueError(
                    f"stacked+mlp: weights_1 must be [w_up_stacked] with shape [E,I,H]. "
                    f"Got {len(weights_1)} elements{', shape: ' + str(weights_1[0].shape) if weights_1 else ''}"
                )
            if len(weights_2) != 1 or weights_2[0].ndim != 3:
                raise ValueError(
                    f"stacked+mlp: weights_2 must be [w_down_stacked] with shape [E,H,I]. "
                    f"Got {len(weights_2)} elements{', shape: ' + str(weights_2[0].shape) if weights_2 else ''}"
                )
            if len(weights_3) > 0:
                raise ValueError(
                    f"stacked+mlp: weights_3 must be empty []. Got {len(weights_3)} elements."
                )

            w1_stacked = weights_1[0]  # [E, I, H]
            w2_stacked = weights_2[0]  # [E, H, I]

            if w1_stacked.shape[0] != w2_stacked.shape[0]:
                raise ValueError(
                    f"Expert count mismatch: weights_1={w1_stacked.shape[0]}, "
                    f"weights_2={w2_stacked.shape[0]}"
                )

            # Extract per-expert slices and create MLPs
            def make_mlp(i: int):
                w1 = w1_stacked[i]  # [I, H]
                w2 = w2_stacked[i]  # [H, I]
                return lambda inp: F.linear(act_fn_callable(F.linear(inp, w1)), w2)

            mlps = [make_mlp(i) for i in range(w1_stacked.shape[0])]

    elif weights_format_enum == WeightsFormat.PER_EXPERT:
        # === PER_EXPERT FORMAT ===
        num_experts = len(weights_1)

        if num_experts == 0:
            raise ValueError("per_expert format: weights_1 cannot be empty")

        if len(weights_2) != num_experts:
            raise ValueError(
                f"per_expert format: weights_1 and weights_2 must have same length. "
                f"weights_1: {num_experts}, weights_2: {len(weights_2)}"
            )

        if mlp_style_enum == MLPStyle.GATED_MLP:
            if weights_fusion_enum == WeightsFusion.UPGATE_DOWN:
                # PER_EXPERT + W3W1_W2 + GATED_MLP: weights_1=[w3_w1 per expert], weights_2=[w2], weights_3=[]
                if len(weights_3) > 0:
                    raise ValueError(
                        f"per_expert+w3w1_w2+gated_mlp: weights_3 must be empty []. Got {len(weights_3)} elements."
                    )

                # Create MLPs from fused weights
                def make_mlp(i: int):
                    w3_w1 = weights_1[i]  # fused [2*I, H] - ordered as [w3, w1]
                    w2 = weights_2[i]  # [H, I]
                    intermediate_size = w3_w1.shape[0] // 2
                    w3 = w3_w1[:intermediate_size, :]  # [I, H]
                    w1 = w3_w1[intermediate_size:, :]  # [I, H]
                    return lambda inp: F.linear(
                        act_fn_callable(F.linear(inp, w1)) * F.linear(inp, w3), w2
                    )

                mlps = [make_mlp(i) for i in range(num_experts)]

            elif weights_fusion_enum == WeightsFusion.GATE_UP_DOWN:
                # PER_EXPERT + W1_W2_W3_SEPARATE + GATED_MLP: weights_1=[w1], weights_2=[w2], weights_3=[w3]
                if len(weights_3) != num_experts:
                    raise ValueError(
                        f"per_expert+w1_w2_w3_separate+gated_mlp: weights_3 must have {num_experts} elements. "
                        f"Got {len(weights_3)}"
                    )

                # Create gated MLPs
                def make_mlp(i: int):
                    w1 = weights_1[i]  # [I, H]
                    w2 = weights_2[i]  # [H, I]
                    w3 = weights_3[i]  # [I, H]
                    return lambda inp: F.linear(
                        act_fn_callable(F.linear(inp, w1)) * F.linear(inp, w3), w2
                    )

                mlps = [make_mlp(i) for i in range(num_experts)]

        elif mlp_style_enum == MLPStyle.MLP:
            # PER_EXPERT + MLP: weights_1=[w_up], weights_2=[w_down], weights_3=[]
            if len(weights_3) > 0:
                raise ValueError(
                    f"per_expert+mlp: weights_3 must be empty []. Got {len(weights_3)} elements."
                )

            # Create simple MLPs
            def make_mlp(i: int):
                w1 = weights_1[i]  # [I, H]
                w2 = weights_2[i]  # [H, I]
                return lambda inp: F.linear(act_fn_callable(F.linear(inp, w1)), w2)

            mlps = [make_mlp(i) for i in range(num_experts)]

    else:
        raise ValueError(f"Unknown weights_format: '{weights_format}'")

    return _template_moe(x, selected_experts, routing_weights, mlps, apply_routing_on_input)


@torch_moe.register_fake
def torch_moe_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    weights_1: List[torch.Tensor],
    weights_2: List[torch.Tensor],
    weights_3: List[torch.Tensor],
    weights_format: str = WeightsFormat.PER_EXPERT.value,
    weights_fusion: str = WeightsFusion.GATE_UP_DOWN.value,
    mlp_style: str = MLPStyle.GATED_MLP.value,
    act_fn: str = ActivationFunction.SILU.value,
    apply_routing_on_input: bool = False,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::torch_moe_fused", mutates_args=())
def torch_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    """
    A reference implementation of a fused MoE layer computation.

    All weights are in TRT-LLM format (conversion from Llama4 happens during graph transformation).

    Parameters:
        x (torch.Tensor): Input tensor of shape (B, H) or (B, S, H), where B is the batch size,
            S is the sequence length, and H is the hidden size.
        selected_experts (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the
            indices of the selected experts for each token.
        routing_weights (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the normalized
            routing weights for the selected experts.
        w3_w1_stacked_weight (torch.Tensor): Stacked w3/w1 weights in TRT-LLM format:
            (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
            Ordered as [w3, w1] along intermediate dimension
        w2_stacked_weight (torch.Tensor): Stacked w2 weights in TRT-LLM format:
            (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    Returns:
        torch.Tensor: Output tensor with the same shape as the input x.
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    num_experts = w2_stacked_weight.shape[0]

    # Standardized on TRT-LLM format (conversion happens during graph transformation)
    # TRT-LLM format: w3_w1 is (2*I, H) ordered as [w3, w1], w2 is (H, I)
    intermediate_size = w3_w1_stacked_weight.shape[1] // 2
    results = torch.zeros_like(x)

    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        if batch_idx.numel() == 0:
            continue

        expert_inputs = x[batch_idx]

        w3_w1 = w3_w1_stacked_weight[expert_id]
        w3 = w3_w1[:intermediate_size, :]
        w1 = w3_w1[intermediate_size:, :]
        w2 = w2_stacked_weight[expert_id]
        expert_out = (F.silu(expert_inputs @ w1.t()) * (expert_inputs @ w3.t())) @ w2.t()

        scaling = routing_weights[batch_idx, nth_expert].unsqueeze(-1)
        results[batch_idx] += scaling * expert_out

    return results.view(x_shape)


@torch_fused_moe.register_fake
def torch_fused_moe_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::torch_quant_fp8_moe", mutates_args=())
def torch_quant_fp8_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
    w1_input_scale: List[torch.Tensor],
    w2_input_scale: List[torch.Tensor],
    w3_input_scale: List[torch.Tensor],
    w1_weight_scale: List[torch.Tensor],
    w2_weight_scale: List[torch.Tensor],
    w3_weight_scale: List[torch.Tensor],
    weights_fusion: str = WeightsFusion.GATE_UP_DOWN.value,
    mlp_style: str = MLPStyle.GATED_MLP.value,
    act_fn: str = ActivationFunction.SILU.value,
) -> torch.Tensor:
    """
    FP8 MoE op using quantized linear operations.

    Computes a Mixture-of-Experts layer similar to the reference auto_deploy::torch_moe op, but uses the
    quantized FP8 linear op for expert computations.

    Args:
        x: Input tensor of shape (B, H) or (B, S, H).
        selected_experts: Tensor (B, TOP_K) or (B*S, TOP_K) containing expert indices.
        routing_weights: Tensor of normalized routing weights.
        w1_weight:
            List of per-expert weight tensors:
              • mlp_style=="gated_mlp": gate with shape (I, H) — gate projection.
              • mlp_style=="mlp":       up with shape (I, H) — up projection.
        w2_weight:
            List of per-expert weight tensors:
              • gated_mlp: down with shape (H, I) — down projection.
              • mlp:       down with shape (H, I) — down projection.
        w3_weight:
            List of per-expert weight tensors:
              • gated_mlp: up with shape (I, H) — up projection in gated MLP.
              • mlp:       pass an empty list []; ignored.
        w1_input_scale, w2_input_scale, w3_input_scale: Lists of input scale tensors for the corresponding ops.
        w1_weight_scale, w2_weight_scale, w3_weight_scale: Lists of weight scale tensors for the corresponding ops.
        mlp_style:
            Selects the per-expert MLP computation:
              • "gated_mlp" (default, Mixtral/DeepSeek-style):
                    y = down( act(gate x) * (up x) )
              • "mlp" (NemotronH-style 2-layer MLP):
                    y = down( act(up x) )
        act_fn:
            Elementwise activation applied inside the expert MLP.
            Supported: "silu" (default), "relu2" (ReLU then square).
    """
    # Convert string parameters to enums
    mlp_style_enum = mlp_style_from_str(mlp_style)
    act_fn_enum = act_fn_from_str(act_fn)
    act_fn_callable = _resolve_activation(act_fn_enum)

    if mlp_style_enum == MLPStyle.GATED_MLP:

        def make_fp8_mlp(i):
            def mlp(inp):
                w1_out = torch.ops.auto_deploy.torch_quant_fp8_linear(
                    inp,
                    w1_weight[i],
                    bias=None,
                    input_scale=w1_input_scale[i],
                    weight_scale=w1_weight_scale[i],
                )
                w3_out = torch.ops.auto_deploy.torch_quant_fp8_linear(
                    inp,
                    w3_weight[i],
                    bias=None,
                    input_scale=w3_input_scale[i],
                    weight_scale=w3_weight_scale[i],
                )
                prod = act_fn_callable(w1_out) * w3_out
                return torch.ops.auto_deploy.torch_quant_fp8_linear(
                    prod,
                    w2_weight[i],
                    bias=None,
                    input_scale=w2_input_scale[i],
                    weight_scale=w2_weight_scale[i],
                )

            return mlp

        mlps = [make_fp8_mlp(i) for i in range(len(w1_weight))]

    elif mlp_style_enum == MLPStyle.MLP:

        def make_fp8_mlp(i):
            def mlp(inp):
                w1_out = torch.ops.auto_deploy.torch_quant_fp8_linear(
                    inp,
                    w1_weight[i],
                    bias=None,
                    input_scale=w1_input_scale[i],
                    weight_scale=w1_weight_scale[i],
                )
                return torch.ops.auto_deploy.torch_quant_fp8_linear(
                    act_fn_callable(w1_out),
                    w2_weight[i],
                    bias=None,
                    input_scale=w2_input_scale[i],
                    weight_scale=w2_weight_scale[i],
                )

            return mlp

        mlps = [make_fp8_mlp(i) for i in range(len(w1_weight))]

    else:
        raise ValueError(f"Unknown mlp_style '{mlp_style}'.")

    return _template_moe(x, selected_experts, routing_weights, mlps)


@torch_quant_fp8_moe.register_fake
def torch_quant_fp8_moe_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
    w1_input_scale: List[torch.Tensor],
    w2_input_scale: List[torch.Tensor],
    w3_input_scale: List[torch.Tensor],
    w1_weight_scale: List[torch.Tensor],
    w2_weight_scale: List[torch.Tensor],
    w3_weight_scale: List[torch.Tensor],
    weights_fusion: str = WeightsFusion.GATE_UP_DOWN.value,
    mlp_style: str = MLPStyle.GATED_MLP.value,
    act_fn: str = ActivationFunction.SILU.value,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::torch_quant_nvfp4_moe", mutates_args=())
def torch_quant_nvfp4_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
    w1_input_scale: List[torch.Tensor],
    w2_input_scale: List[torch.Tensor],
    w3_input_scale: List[torch.Tensor],
    w1_weight_scale: List[torch.Tensor],
    w2_weight_scale: List[torch.Tensor],
    w3_weight_scale: List[torch.Tensor],
    w1_alpha: List[torch.Tensor],
    w2_alpha: List[torch.Tensor],
    w3_alpha: List[torch.Tensor],
    weights_fusion: str = WeightsFusion.GATE_UP_DOWN.value,
    mlp_style: str = MLPStyle.GATED_MLP.value,
    act_fn: str = ActivationFunction.SILU.value,
) -> torch.Tensor:
    """
    FP4 MoE op using quantized linear operations.

    Computes a Mixture-of-Experts layer similar to the reference auto_deploy::torch_moe op,
    but uses the NVFP4 quantized linear op for expert computations.

    Args:
        x: Input tensor of shape (B, H) or (B, S, H).
        selected_experts: Tensor (B, TOP_K) or (B*S, TOP_K) containing expert indices.
        routing_weights: Tensor of normalized routing weights.
        w1_weight:
            List of per-expert weight tensors:
              • mlp_style=="gated_mlp": gate with shape (I, H) — gate projection.
              • mlp_style=="mlp":       up with shape (I, H) — up projection.
        w2_weight:
            List of per-expert weight tensors:
              • gated_mlp: down with shape (H, I) — down projection.
              • mlp:       down with shape (H, I) — down projection.
        w3_weight:
            List of per-expert weight tensors:
              • gated_mlp: up with shape (I, H) — up projection in gated MLP.
              • mlp:       pass an empty list []; ignored.
        w1_input_scale, w2_input_scale, w3_input_scale: Lists of input scale tensors.
        w1_weight_scale, w2_weight_scale, w3_weight_scale: Lists of weight scale tensors.
        w1_alpha, w2_alpha, w3_alpha: Lists of alpha scale tensors for FP4 quantization.
        weights_fusion: Weight fusion strategy (default: "w1_w2_w3_separate")
        mlp_style:
            Selects the per-expert MLP computation:
              • "gated_mlp" (default, Mixtral/DeepSeek-style):
                    y = w2( act(w1 x) * (w3 x) )
              • "mlp" (NemotronH-style 2-layer MLP):
                    y = w2( act(w1 x) )
        act_fn:
            Elementwise activation applied inside the expert MLP.
            Supported: "silu" (default), "relu2" (ReLU then square).
    """
    # Convert string parameters to enums
    mlp_style_enum = mlp_style_from_str(mlp_style)
    act_fn_enum = act_fn_from_str(act_fn)
    act_fn_callable = _resolve_activation(act_fn_enum)

    if mlp_style_enum == MLPStyle.GATED_MLP:

        def make_fp4_mlp(i):
            def mlp(inp):
                if inp.shape[0] == 0:
                    return torch.zeros_like(inp)
                w1_out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
                    inp,
                    w1_weight[i],
                    bias=None,
                    input_scale=w1_input_scale[i],
                    weight_scale=w1_weight_scale[i],
                    alpha=w1_alpha[i],
                )
                w3_out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
                    inp,
                    w3_weight[i],
                    bias=None,
                    input_scale=w3_input_scale[i],
                    weight_scale=w3_weight_scale[i],
                    alpha=w3_alpha[i],
                )
                prod = act_fn_callable(w1_out) * w3_out
                return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
                    prod,
                    w2_weight[i],
                    bias=None,
                    input_scale=w2_input_scale[i],
                    weight_scale=w2_weight_scale[i],
                    alpha=w2_alpha[i],
                )

            return mlp

        mlps = [make_fp4_mlp(i) for i in range(len(w1_weight))]

    elif mlp_style_enum == MLPStyle.MLP:

        def make_fp4_mlp(i):
            def mlp(inp):
                if inp.shape[0] == 0:
                    return torch.zeros_like(inp)
                w1_out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
                    inp,
                    w1_weight[i],
                    bias=None,
                    input_scale=w1_input_scale[i],
                    weight_scale=w1_weight_scale[i],
                    alpha=w1_alpha[i],
                )
                return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
                    act_fn_callable(w1_out),
                    w2_weight[i],
                    bias=None,
                    input_scale=w2_input_scale[i],
                    weight_scale=w2_weight_scale[i],
                    alpha=w2_alpha[i],
                )

            return mlp

        mlps = [make_fp4_mlp(i) for i in range(len(w1_weight))]

    else:
        raise ValueError(f"Unknown mlp_style '{mlp_style}'.")

    return _template_moe(x, selected_experts, routing_weights, mlps)


@torch_quant_nvfp4_moe.register_fake
def torch_quant_nvfp4_moe_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
    w1_input_scale: List[torch.Tensor],
    w2_input_scale: List[torch.Tensor],
    w3_input_scale: List[torch.Tensor],
    w1_weight_scale: List[torch.Tensor],
    w2_weight_scale: List[torch.Tensor],
    w3_weight_scale: List[torch.Tensor],
    w1_alpha: List[torch.Tensor],
    w2_alpha: List[torch.Tensor],
    w3_alpha: List[torch.Tensor],
    weights_fusion: str = WeightsFusion.GATE_UP_DOWN.value,
    mlp_style: str = MLPStyle.GATED_MLP.value,
    act_fn: str = ActivationFunction.SILU.value,
) -> torch.Tensor:
    return torch.empty_like(x)


# GPT-OSS uses this style
@torch.library.custom_op("auto_deploy::torch_moe_dense_mlp", mutates_args=())
def torch_moe_dense_mlp(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    routing_weights: torch.Tensor,  # [B*S, E]
    gate_up_w: torch.Tensor,  # [E, H, 2I] - note: this is interleaved gate/up
    gate_up_b: torch.Tensor,  # [E, 2I] - note: this is interleaved gate/up
    down_w: torch.Tensor,  # [E, I, H]
    down_b: torch.Tensor,  # [E, H]
    alpha: float = 1.0,
    limit: float = 10.0,
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_size)  # (num_tokens, hidden_size)
    num_experts = routing_weights.shape[1]

    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, hidden_size)
    gate_up = torch.bmm(hidden_states, gate_up_w) + gate_up_b[..., None, :]
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]  # interleaved: even=gate, odd=up
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    next_states = torch.bmm(((up + 1) * glu), down_w)
    next_states = next_states + down_b[..., None, :]
    next_states = next_states.view(num_experts, batch_size, -1, hidden_size)
    next_states = (
        next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
    )
    next_states = next_states.sum(dim=0)
    next_states = next_states.reshape(*leading_shape, hidden_size)
    return next_states  # [B, S, H] or [B*S, H]


@torch_moe_dense_mlp.register_fake
def _torch_moe_dense_mlp_fake(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_w: torch.Tensor,
    gate_up_b: torch.Tensor,
    down_w: torch.Tensor,
    down_b: torch.Tensor,
    alpha: float = 1.0,
    limit: float = 10.0,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)
