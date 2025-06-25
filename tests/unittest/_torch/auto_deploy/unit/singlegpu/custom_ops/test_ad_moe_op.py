import pytest
import torch
import torch.nn.functional as F
from _torch.helpers import reference_moe_torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.modules.fused_moe import MoE  # noqa: F401


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_op_run(dtype):
    SEQ_LEN = 8
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = 3
    TOP_K = 2

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda() * 0.5

    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=torch.float32).cuda()
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    final_scales, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
    final_scales = final_scales / final_scales.sum(dim=-1, keepdim=True)
    final_scales = final_scales.to(x.dtype)

    w1_weight = []
    w2_weight = []
    w3_weight = []
    weights = {}
    fused_w3_w1_stacked_weight = torch.empty(
        (NUM_EXPERTS, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE), dtype=dtype
    ).cuda()
    fused_w2_weight = torch.empty((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype).cuda()
    for expert_id in range(NUM_EXPERTS):
        w1 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda() * 0.5
        w2 = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype).cuda() * 0.5
        w3 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda() * 0.5
        weights[f"{expert_id}.w1.weight"] = w1
        weights[f"{expert_id}.w2.weight"] = w2
        weights[f"{expert_id}.w3.weight"] = w3

        w1_weight.append(w1)
        w2_weight.append(w2)
        w3_weight.append(w3)

        fused_w3_w1_stacked_weight.data[expert_id].copy_(torch.cat([w3, w1], dim=-2))
        fused_w2_weight.data[expert_id].copy_(w2)

    with torch.inference_mode():
        output_torch_moe = torch.ops.auto_deploy.torch_moe(
            x,
            selected_experts,
            final_scales,
            w1_weight,
            w2_weight,
            w3_weight,
        )
        output_torch_fused_moe = torch.ops.auto_deploy.torch_moe_fused(
            x,
            selected_experts,
            final_scales,
            fused_w3_w1_stacked_weight,
            fused_w2_weight,
        )
        output_trt_fused_moe = torch.ops.auto_deploy.trtllm_moe_fused(
            x,
            selected_experts,
            final_scales,
            fused_w3_w1_stacked_weight,
            fused_w2_weight,
        )

        ref_output = reference_moe_torch(x, selected_experts, final_scales, NUM_EXPERTS, weights)

    torch.cuda.synchronize()
    torch.testing.assert_close(output_trt_fused_moe, output_torch_fused_moe, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(output_trt_fused_moe, ref_output, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(output_torch_fused_moe, ref_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(output_torch_moe, ref_output, rtol=1e-5, atol=1e-5)
