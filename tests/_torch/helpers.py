from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def reference_moe_torch(x: torch.Tensor, router_logits: torch.Tensor,
                        top_k: int,
                        weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    num_experts = router_logits.shape[-1]
    routing_weights = nn.functional.softmax(router_logits,
                                            dim=1,
                                            dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights,
                                                   top_k,
                                                   dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # cast back to the input dtype
    routing_weights = routing_weights.to(x.dtype)
    results = torch.zeros_like(x)

    # naive looping over experts
    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        w1_weight = weights[f"{expert_id}.w1.weight"]
        w2_weight = weights[f"{expert_id}.w2.weight"]
        w3_weight = weights[f"{expert_id}.w3.weight"]
        expert_inputs = x[batch_idx]
        output = (F.silu(expert_inputs @ w1_weight.t()) *
                  (expert_inputs @ w3_weight.t())) @ w2_weight.t()
        results[batch_idx] += routing_weights[batch_idx, nth_expert,
                                              None] * output

    return results.view_as(x)
