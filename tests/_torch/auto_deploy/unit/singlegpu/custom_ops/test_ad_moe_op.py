from typing import Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.modules.fused_moe import FusedMoE  # noqa: F401


# Copy from tests/_torch/helpers.py::reference_moe_torch
def reference_moe_torch(
    x: torch.Tensor, router_logits: torch.Tensor, top_k: int, weights: Dict[str, torch.Tensor]
) -> torch.Tensor:
    num_experts = router_logits.shape[-1]
    routing_weights = nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
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
        output = (
            F.silu(expert_inputs @ w1_weight.t()) * (expert_inputs @ w3_weight.t())
        ) @ w2_weight.t()
        results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * output

    return results.view_as(x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_op_run(dtype):
    SEQ_LEN = 8
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = 3
    TOP_K = 2
    TP_SIZE = 1
    TP_RANK = 0
    EP_SIZE = 1
    EP_RANK = 0

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

    w1_weight = []
    w2_weight = []
    w3_weight = []
    weights = {}
    fused_w3_w1_stacked_weight = torch.empty(
        (NUM_EXPERTS, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE), dtype=dtype
    ).cuda()
    fused_w2_weight = torch.empty((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype).cuda()
    for expert_id in range(NUM_EXPERTS):
        w1 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
        w2 = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype).cuda()
        w3 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
        weights[f"{expert_id}.w1.weight"] = w1
        weights[f"{expert_id}.w2.weight"] = w2
        weights[f"{expert_id}.w3.weight"] = w3

        w1_weight.append(w1)
        w2_weight.append(w2)
        w3_weight.append(w3)

        fused_w3_w1_stacked_weight.data[expert_id].copy_(torch.cat([w3, w1], dim=-2))
        fused_w2_weight.data[expert_id].copy_(w2)

    with torch.inference_mode():
        output_torch_moe = torch.ops.moe.torch_moe(
            x,
            router_logits.float(),
            w1_weight,
            w2_weight,
            w3_weight,
            TOP_K,
        )
        output_torch_fused_moe = torch.ops.moe.torch_fused_moe(
            x,
            router_logits.float(),
            fused_w3_w1_stacked_weight,
            fused_w2_weight,
            TOP_K,
        )
        output_trt_fused_moe = torch.ops.trtllm.fused_moe(
            x,
            router_logits.float(),
            fused_w3_w1_stacked_weight,
            fused_w2_weight,
            dtype,
            TOP_K,
            quant_scales=None,
            tp_size=TP_SIZE,
            tp_rank=TP_RANK,
            ep_size=EP_SIZE,
            ep_rank=EP_RANK,
            profile_ids=None,
        )

        ref_output = reference_moe_torch(x, router_logits, TOP_K, weights)

    torch.cuda.synchronize()
    torch.testing.assert_close(output_torch_moe, ref_output, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(output_torch_fused_moe, ref_output, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(output_trt_fused_moe, ref_output, rtol=0.2, atol=0.5)
