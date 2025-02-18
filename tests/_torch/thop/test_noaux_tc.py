import os
import sys

import pytest
import torch

from tensorrt_llm._torch.models.modeling_deepseekv3 import Deepseekv3Gate

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@pytest.mark.parametrize("seq_len", [1, 32, 8192])
@pytest.mark.parametrize("num_experts, n_group, topk_group, top_k", [
    (256, 8, 4, 8),
    (72, 1, 1, 6),
])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_noaux_tc_run(seq_len, num_experts, n_group, topk_group, top_k, dtype):
    ROUTED_SCALING_FACTOR = 2.5
    HIDDEN_SIZE = 7168
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    weight = torch.randn((num_experts, HIDDEN_SIZE), dtype=torch.float32).cuda()
    e_score_correction_bias = torch.randn((num_experts),
                                          dtype=torch.float32).cuda()

    logits = torch.randn((seq_len, HIDDEN_SIZE), dtype=dtype).cuda()

    weights = {}
    weights["weight"] = weight
    weights["e_score_correction_bias"] = e_score_correction_bias

    # Run the thop
    gate = Deepseekv3Gate(hidden_size=HIDDEN_SIZE,
                          num_experts=num_experts,
                          top_k=top_k,
                          n_group=n_group,
                          topk_group=topk_group,
                          routed_scaling_factor=ROUTED_SCALING_FACTOR,
                          is_thop=True)
    gate.load_weights([weights])
    gate.cuda()
    with torch.inference_mode():
        output = gate.forward(logits)

    # Run the original version
    ref_gate = Deepseekv3Gate(hidden_size=HIDDEN_SIZE,
                              num_experts=num_experts,
                              top_k=top_k,
                              n_group=n_group,
                              topk_group=topk_group,
                              routed_scaling_factor=ROUTED_SCALING_FACTOR,
                              is_thop=False)
    ref_gate.load_weights([weights])
    ref_gate.cuda()
    with torch.inference_mode():
        ref_output = ref_gate.forward(logits)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output, rtol=0.01, atol=0.01)
