from typing import Callable, List, Optional

import torch
import torch.nn.functional as F


def _resolve_activation(name: Optional[str]) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns an elementwise activation callable matching the given name.
    Supported: "silu", "relu2".
    Defaults to SiLU when name is None or empty.
    """
    if not name:
        name = "silu"
    key = name.lower()

    if key == "silu":
        return F.silu
    elif key == "relu2":

        def relu2(x: torch.Tensor) -> torch.Tensor:
            return torch.square(F.relu(x))

        return relu2
    else:
        raise ValueError(f"Unsupported activation '{name}'. Use one of: silu, relu2.")


def _template_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    mlps: List[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """Mixtral-style generic MoE template, dispatching tokens to expert MLPs based on routing info."""
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
    one_hot = F.one_hot(selected_experts_fixed, num_classes=num_experts + 1)
    expert_mask = one_hot[..., :num_experts].permute(2, 1, 0)

    for expert_idx in range(num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        tokens_for_this_expert = x[None, top_x].reshape(-1, hidden_dim)
        if not tokens_for_this_expert.shape[0]:
            continue  # input of shape [0, hidden_dim] breaks fp4 kernel

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
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    """
    Unified Mixture-of-Experts (MoE) operator that uses a Mixtral-style dispatch
    (token routing + index_add_ accumulation) and a selectable per-expert MLP.
    Parameters:
        x (torch.Tensor): Input tensor of shape (B, H) or (B, S, H), where B is the batch size,
            S is the sequence length, and H is the hidden size.
        selected_experts (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the indices
            of the selected experts for each token. Only experts within range [0,num_experts) is processed
        routing_weights (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the normalized
            routing weights for the selected experts.
        w1_weight:
            List of per-expert weight tensors:
              • mlp_style=="gated_mlp": W1 with shape (I, H)  — "gate" projection.
              • mlp_style=="mlp":       W_up with shape (I, H) — up projection.
        w2_weight:
            List of per-expert weight tensors:
              • gated_mlp: W2 with shape (H, I) — down projection.
              • mlp:       W_down with shape (H, I) — down projection.
        w3_weight:
            List of per-expert weight tensors:
              • gated_mlp: W3 with shape (I, H) — “up” (second) projection in gated MLP.
              • mlp:       pass an empty list []; ignored.
        mlp_style:
            Selects the per-expert MLP computation:
              • "gated_mlp" (default, Mixtral/DeepSeek-style):
                    y = W2( act(W1 x) * (W3 x) )
              • "mlp" (NemotronH-style 2-layer MLP):
                    y = W_down( act(W_up x) )
        act_fn:
            Elementwise activation applied inside the expert MLP.
            Supported: "silu" (default), "relu2" (ReLU then square).
    Returns:
        torch.Tensor: Output tensor with the same shape as the input x.
    """
    act_fn = _resolve_activation(act_fn)
    style = mlp_style.lower()

    if style == "gated_mlp":

        def make_mlp(i: int):
            W1 = w1_weight[i]  # (I, H)
            W2 = w2_weight[i]  # (H, I)
            W3 = w3_weight[i]  # (I, H)
            return lambda inp: F.linear(act_fn(F.linear(inp, W1)) * F.linear(inp, W3), W2)

        mlps = [make_mlp(i) for i in range(len(w1_weight))]

    elif style == "mlp":

        def make_mlp(i: int):
            W_up = w1_weight[i]  # (I, H)
            W_down = w2_weight[i]  # (H, I)
            return lambda inp: F.linear(act_fn(F.linear(inp, W_up)), W_down)

        mlps = [make_mlp(i) for i in range(len(w1_weight))]

    else:
        raise ValueError(f"Unknown mlp_style '{mlp_style}'. Use 'gated_mlp' or 'mlp'.")

    return _template_moe(x, selected_experts, routing_weights, mlps)


@torch_moe.register_fake
def torch_moe_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
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
    Parameters:
        x (torch.Tensor): Input tensor of shape (B, H) or (B, S, H), where B is the batch size,
            S is the sequence length, and H is the hidden size.
        selected_experts (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the
            indices of the selected experts for each token.
        routing_weights (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the normalized
            routing weights for the selected experts.
        w3_w1_stacked_weight (torch.Tensor): A tensor of shape (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
            containing the fused weights for w3 and w1 for each expert.
        w2_stacked_weight (torch.Tensor): A tensor of shape (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
            containing the weights for w2 for each expert.
    Returns:
        torch.Tensor: Output tensor with the same shape as the input x.
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    num_experts = w2_stacked_weight.shape[0]
    intermediate_size = w3_w1_stacked_weight.shape[1] // 2
    results = torch.zeros_like(x)

    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        if batch_idx.numel() == 0:
            continue

        expert_inputs = x[batch_idx]

        stacked = w3_w1_stacked_weight[expert_id]
        w3 = stacked[:intermediate_size, :]
        w1 = stacked[intermediate_size:, :]
        w2 = w2_stacked_weight[expert_id]

        # Compute expert output:
        #   expert_out = (F.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()
        out_w1 = expert_inputs @ w1.t()
        out_w3 = expert_inputs @ w3.t()
        expert_out = (F.silu(out_w1) * out_w3) @ w2.t()

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
    mlp_style: str = "gated_mlp",  # "gated_mlp" (default) or "mlp"
    act_fn: str = "silu",  # silu or relu2
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
              • mlp_style=="gated_mlp": W1 with shape (I, H)  — "gate" projection.
              • mlp_style=="mlp":       W_up with shape (I, H) — up projection.
        w2_weight:
            List of per-expert weight tensors:
              • gated_mlp: W2 with shape (H, I) — down projection.
              • mlp:       W_down with shape (H, I) — down projection.
        w3_weight:
            List of per-expert weight tensors:
              • gated_mlp: W3 with shape (I, H) — "up" (second) projection in gated MLP.
              • mlp:       pass an empty list []; ignored.
        w1_input_scale, w2_input_scale, w3_input_scale: Lists of input scale tensors for the corresponding ops.
        w1_weight_scale, w2_weight_scale, w3_weight_scale: Lists of weight scale tensors for the corresponding ops.
        mlp_style:
            Selects the per-expert MLP computation:
              • "gated_mlp" (default, Mixtral/DeepSeek-style):
                    y = W2( act(W1 x) * (W3 x) )
              • "mlp" (NemotronH-style 2-layer MLP):
                    y = W_down( act(W_up x) )
        act_fn:
            Elementwise activation applied inside the expert MLP.
            Supported: "silu" (default), "relu2" (ReLU then square).
    """

    act_fn = _resolve_activation(act_fn)
    style = mlp_style.lower()

    if style == "gated_mlp":

        def make_fp8_mlp(i):
            def mlp(inp):
                gate_out = torch.ops.auto_deploy.torch_quant_fp8_linear(
                    inp,
                    w1_weight[i],
                    bias=None,
                    input_scale=w1_input_scale[i],
                    weight_scale=w1_weight_scale[i],
                )
                up_out = torch.ops.auto_deploy.torch_quant_fp8_linear(
                    inp,
                    w3_weight[i],
                    bias=None,
                    input_scale=w3_input_scale[i],
                    weight_scale=w3_weight_scale[i],
                )
                prod = act_fn(gate_out) * up_out
                return torch.ops.auto_deploy.torch_quant_fp8_linear(
                    prod,
                    w2_weight[i],
                    bias=None,
                    input_scale=w2_input_scale[i],
                    weight_scale=w2_weight_scale[i],
                )

            return mlp

        mlps = [make_fp8_mlp(i) for i in range(len(w1_weight))]

    elif style == "mlp":

        def make_fp8_mlp(i):
            def mlp(inp):
                up_out = torch.ops.auto_deploy.torch_quant_fp8_linear(
                    inp,
                    w1_weight[i],
                    bias=None,
                    input_scale=w1_input_scale[i],
                    weight_scale=w1_weight_scale[i],
                )
                return torch.ops.auto_deploy.torch_quant_fp8_linear(
                    act_fn(up_out),
                    w2_weight[i],
                    bias=None,
                    input_scale=w2_input_scale[i],
                    weight_scale=w2_weight_scale[i],
                )

            return mlp

        mlps = [make_fp8_mlp(i) for i in range(len(w1_weight))]

    else:
        raise ValueError(f"Unknown mlp_style '{mlp_style}'. Use 'gated_mlp' or 'mlp'.")

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
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
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
    mlp_style: str = "gated_mlp",  # "gated_mlp" (default) or "mlp"
    act_fn: str = "silu",  # silu or relu2
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
              • mlp_style=="gated_mlp": W1 with shape (I, H)  — "gate" projection.
              • mlp_style=="mlp":       W_up with shape (I, H) — up projection.
        w2_weight:
            List of per-expert weight tensors:
              • gated_mlp: W2 with shape (H, I) — down projection.
              • mlp:       W_down with shape (H, I) — down projection.
        w3_weight:
            List of per-expert weight tensors:
              • gated_mlp: W3 with shape (I, H) — "up" (second) projection in gated MLP.
              • mlp:       pass an empty list []; ignored.
        w1_input_scale, w2_input_scale, w3_input_scale: Lists of input scale tensors.
        w1_weight_scale, w2_weight_scale, w3_weight_scale: Lists of weight scale tensors.
        w1_alpha, w2_alpha, w3_alpha: Lists of alpha scale tensors for FP4 quantization.
        mlp_style:
            Selects the per-expert MLP computation:
              • "gated_mlp" (default, Mixtral/DeepSeek-style):
                    y = W2( act(W1 x) * (W3 x) )
              • "mlp" (NemotronH-style 2-layer MLP):
                    y = W_down( act(W_up x) )
        act_fn:
            Elementwise activation applied inside the expert MLP.
            Supported: "silu" (default), "relu2" (ReLU then square).
    """

    act_fn = _resolve_activation(act_fn)
    style = mlp_style.lower()

    if style == "gated_mlp":

        def make_fp4_mlp(i):
            def mlp(inp):
                if inp.shape[0] == 0:
                    return torch.zeros_like(inp)
                gate_out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
                    inp,
                    w1_weight[i],
                    bias=None,
                    input_scale=w1_input_scale[i],
                    weight_scale=w1_weight_scale[i],
                    alpha=w1_alpha[i],
                )
                up_out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
                    inp,
                    w3_weight[i],
                    bias=None,
                    input_scale=w3_input_scale[i],
                    weight_scale=w3_weight_scale[i],
                    alpha=w3_alpha[i],
                )
                prod = act_fn(gate_out) * up_out
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

    elif style == "mlp":

        def make_fp4_mlp(i):
            def mlp(inp):
                if inp.shape[0] == 0:
                    return torch.zeros_like(inp)
                up_out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
                    inp,
                    w1_weight[i],
                    bias=None,
                    input_scale=w1_input_scale[i],
                    weight_scale=w1_weight_scale[i],
                    alpha=w1_alpha[i],
                )
                return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
                    act_fn(up_out),
                    w2_weight[i],
                    bias=None,
                    input_scale=w2_input_scale[i],
                    weight_scale=w2_weight_scale[i],
                    alpha=w2_alpha[i],
                )

            return mlp

        mlps = [make_fp4_mlp(i) for i in range(len(w1_weight))]

    else:
        raise ValueError(f"Unknown mlp_style '{mlp_style}'. Use 'gated_mlp' or 'mlp'.")

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
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    return torch.empty_like(x)


# GPT-OSS uses this style
@torch.library.custom_op("auto_deploy::torch_moe_dense_mlp", mutates_args=())
def torch_moe_dense_mlp(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    routing_weights: torch.Tensor,  # [B*S, E]
    gate_up_w: torch.Tensor,  # [E, H, 2I]
    gate_up_b: torch.Tensor,  # [E, 2I]
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
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
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
