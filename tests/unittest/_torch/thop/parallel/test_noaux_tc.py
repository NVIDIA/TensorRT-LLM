import pytest
import torch

from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate


@pytest.mark.parametrize("seq_len", [1, 32, 8192])
@pytest.mark.parametrize("num_experts, n_group, topk_group, top_k", [
    (256, 8, 4, 8),
    (72, 1, 1, 6),
    (384, 1, 1, 8),
])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_noaux_tc_run(seq_len, num_experts, n_group, topk_group, top_k, dtype):
    ROUTED_SCALING_FACTOR = 2.5
    HIDDEN_SIZE = 7168
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    weight = torch.randn((num_experts, HIDDEN_SIZE), dtype=dtype).cuda()
    e_score_correction_bias = torch.randn((num_experts),
                                          dtype=torch.float32).cuda()

    logits = torch.randn((seq_len, HIDDEN_SIZE), dtype=dtype).cuda()

    weights = {}
    weights["weight"] = weight
    weights["e_score_correction_bias"] = e_score_correction_bias

    # Run the thop
    gate = DeepseekV3Gate(
        hidden_size=HIDDEN_SIZE,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        dtype=dtype,
        fuse_routing_kernel=True,
        apply_routing=False,
    )
    gate.load_weights([weights])
    gate.cuda()
    with torch.inference_mode():
        selected_indices, selected_values = gate.routing_method.apply(
            gate.forward(logits))

    # Run the original version
    ref_gate = DeepseekV3Gate(
        hidden_size=HIDDEN_SIZE,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        dtype=dtype,
        fuse_routing_kernel=False,
        apply_routing=False,
    )
    ref_gate.load_weights([weights])
    ref_gate.cuda()
    with torch.inference_mode():
        ref_selected_indices, ref_selected_values = ref_gate.routing_method.apply(
            ref_gate.forward(logits))

    # sort before compare
    sorted_selected_values, _ = torch.sort(selected_values)
    ref_sorted_selected_values, _ = torch.sort(ref_selected_values)

    # compare
    torch.cuda.synchronize()

    torch.testing.assert_close(sorted_selected_values,
                               ref_sorted_selected_values,
                               rtol=0.01,
                               atol=0.01)
