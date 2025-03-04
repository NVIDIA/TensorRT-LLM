from typing import List

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.fused_moe import FusedMoE  # noqa: F401


@torch.library.custom_op("moe::torch_moe", mutates_args=())
def torch_moe(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
    top_k: int,
) -> torch.Tensor:
    """
    A reference implementation of fused MoE with the same signature as torch.ops.trtllm.fused_moe.
      - w1_weight: a list of tensors, each of shape (INTERMEDIATE_SIZE, HIDDEN_SIZE)
      - w2_weight: a list of tensors, each of shape (HIDDEN_SIZE, INTERMEDIATE_SIZE)
      - w3_weight: a list of tensors, each of shape (INTERMEDIATE_SIZE, HIDDEN_SIZE)

    """
    num_experts = len(w1_weight)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(x.dtype)

    results = torch.zeros_like(x)

    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        if batch_idx.numel() == 0:
            continue

        expert_inputs = x[batch_idx]

        # Get the expert-specific weights from the list
        w1 = w1_weight[expert_id]
        w2 = w2_weight[expert_id]
        w3 = w3_weight[expert_id]

        # Compute expert output:
        #   expert_out = (F.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()
        out_w1 = expert_inputs @ w1.t()  # shape: (N, INTERMEDIATE_SIZE)
        out_w3 = expert_inputs @ w3.t()  # shape: (N, INTERMEDIATE_SIZE)
        expert_out = (F.silu(out_w1) * out_w3) @ w2.t()  # shape: (N, HIDDEN_SIZE)

        scaling = routing_weights[batch_idx, nth_expert].unsqueeze(-1)
        results[batch_idx] += scaling * expert_out

    return results.view_as(x)


@torch_moe.register_fake
def torch_moe(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
    top_k: int,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("moe::torch_fused_moe", mutates_args=())
def torch_fused_moe(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    A reference implementation of fused MoE with the same signature as torch.ops.trtllm.fused_moe.
      - w3_w1_stacked_weight: a tensor of shape (NUM_EXPERTS, 2*INTERMEDIATE_SIZE, HIDDEN_SIZE)
        where, for each expert, the first half along dim=1 corresponds to w3 and the second half to w1.
      - w2_weight: a tensor of shape (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    """
    num_experts = router_logits.shape[-1]
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(x.dtype)

    results = torch.zeros_like(x)

    intermediate_size = w3_w1_stacked_weight.shape[1] // 2

    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        if batch_idx.numel() == 0:
            continue

        expert_inputs = x[batch_idx]

        stacked = w3_w1_stacked_weight[expert_id]
        w3 = stacked[:intermediate_size, :]  # shape: (INTERMEDIATE_SIZE, HIDDEN_SIZE)
        w1 = stacked[intermediate_size:, :]  # shape: (INTERMEDIATE_SIZE, HIDDEN_SIZE)
        w2 = w2_weight[expert_id]  # shape: (HIDDEN_SIZE, INTERMEDIATE_SIZE)

        # Compute expert output:
        #   expert_out = (F.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()
        out_w1 = expert_inputs @ w1.t()  # shape: (N, INTERMEDIATE_SIZE)
        out_w3 = expert_inputs @ w3.t()  # shape: (N, INTERMEDIATE_SIZE)
        expert_out = (F.silu(out_w1) * out_w3) @ w2.t()  # shape: (N, HIDDEN_SIZE)

        scaling = routing_weights[batch_idx, nth_expert].unsqueeze(-1)
        results[batch_idx] += scaling * expert_out

    return results.view_as(x)


@torch_fused_moe.register_fake
def torch_fused_moe(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("moe::trtllm_fused_moe", mutates_args=())
def trtllm_fused_moe(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return torch.ops.trtllm.fused_moe(
        x, router_logits.float(), w3_w1_stacked_weight, w2_stacked_weight, x.dtype, top_k
    )


@trtllm_fused_moe.register_fake
def trtllm_fused_moe(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return torch.empty_like(x)
