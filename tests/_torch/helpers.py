from typing import Dict

import torch
import torch.nn.functional as F


def reference_moe_torch(x: torch.Tensor, selected_experts: torch.Tensor,
                        final_scales: torch.Tensor, num_experts: int,
                        weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    # cast back to the input dtype
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
        results[batch_idx] += final_scales[batch_idx, nth_expert, None] * output

    return results.view_as(x)
