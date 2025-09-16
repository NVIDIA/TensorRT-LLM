from typing import Callable, List

import torch
import torch.nn.functional as F


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
) -> torch.Tensor:
    """
    A reference implementation of a Mixture-of-Experts (MoE) layer computation in the Mixtral style,
    compatible with DeepSeek by using index_add_ for in-place updates.
    Parameters:
        x (torch.Tensor): Input tensor of shape (B, H) or (B, S, H), where B is the batch size,
            S is the sequence length, and H is the hidden size.
        selected_experts (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the indices
            of the selected experts for each token. Only experts within range [0,num_experts) is processed
        routing_weights (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the normalized
            routing weights for the selected experts.
        w1_weight (List[torch.Tensor]): A list of expert weight tensors for w1, each of shape
            (I, H), where I is the intermediate size.
        w2_weight (List[torch.Tensor]): A list of expert weight tensors for w2, each of shape
            (H, I).
        w3_weight (List[torch.Tensor]): A list of expert weight tensors for w3, each of shape
            (I, H).
    Returns:
        torch.Tensor: Output tensor with the same shape as the input x.
    """

    def make_mlp(i):
        return lambda inp: F.linear(
            F.silu(F.linear(inp, w1_weight[i])) * F.linear(inp, w3_weight[i]), w2_weight[i]
        )

    mlps = [make_mlp(i) for i in range(len(w1_weight))]
    return _template_moe(x, selected_experts, routing_weights, mlps)


@torch_moe.register_fake
def torch_moe_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
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
) -> torch.Tensor:
    """
    FP8 MoE op using quantized linear operations.

    Computes a Mixture-of-Experts layer similar to the reference auto_deploy::torch_moe op, but uses the
    quantized FP8 linear op for expert computations.

    Args:
        x: Input tensor of shape (B, H) or (B, S, H).
        selected_experts: Tensor (B, TOP_K) or (B*S, TOP_K) containing expert indices.
        routing_weights: Tensor of normalized routing weights.
        w1_weight, w2_weight, w3_weight: Lists of pre-quantized weight tensors for the three linear ops.
        w1_input_scale, w2_input_scale, w3_input_scale: Lists of input scale tensors for the corresponding ops.
        w1_weight_scale, w2_weight_scale, w3_weight_scale: Lists of weight scale tensors for the corresponding ops.

    """

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
            prod = F.silu(gate_out) * up_out
            return torch.ops.auto_deploy.torch_quant_fp8_linear(
                prod,
                w2_weight[i],
                bias=None,
                input_scale=w2_input_scale[i],
                weight_scale=w2_weight_scale[i],
            )

        return mlp

    mlps = [make_fp8_mlp(i) for i in range(len(w1_weight))]
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
) -> torch.Tensor:
    """
    FP4 MoE op using quantized linear operations.

    Computes a Mixture-of-Experts layer similar to the reference auto_deploy::torch_moe op,
    but uses the NVFP4 quantized linear op for expert computations.

    Args:
        x: Input tensor of shape (B, H) or (B, S, H).
        selected_experts: Tensor (B, TOP_K) or (B*S, TOP_K) containing expert indices.
        routing_weights: Tensor of normalized routing weights.
        w1_weight, w2_weight, w3_weight: Lists of pre-quantized weight tensors for the three linear ops.
        w1_input_scale, w2_input_scale, w3_input_scale: Lists of input scale tensors.
        w1_weight_scale, w2_weight_scale, w3_weight_scale: Lists of weight scale tensors.
        w1_alpha, w2_alpha, w3_alpha: Lists of alpha scale tensors for FP4 quantization.
    """

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
            prod = F.silu(gate_out) * up_out
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
) -> torch.Tensor:
    return torch.empty_like(x)
