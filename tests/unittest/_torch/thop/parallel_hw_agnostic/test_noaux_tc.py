# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate
from tensorrt_llm._torch.modules.linear import Linear


def test_deepseek_gate_dispatches_locality_domain_bf16_with_fp32_output() -> None:
    policy = LocalityDomainPolicy(enabled=False)
    gate = DeepseekV3Gate(
        hidden_size=128,
        num_experts=16,
        top_k=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        dtype=torch.bfloat16,
        use_cute_dsl_bf16_gemm=True,
        locality_domain_policy=policy,
    )
    hidden_states = torch.empty(2, 3, 128, dtype=torch.bfloat16)
    expected = torch.empty(2, 3, 16, dtype=torch.float32)
    run_locality_domain = Mock(return_value=expected)
    gate._run_bf16_linear_locality_domain = run_locality_domain
    gate._locality_domain_weight_shards = ({"weight": object()}, {"weight": object()})

    output = gate(hidden_states)

    assert isinstance(gate, Linear)
    assert gate.enable_locality_domain_bf16_linear
    assert gate._locality_domain_policy is policy
    assert gate._skip_layerwise_quant_config
    assert set(dict(gate.named_parameters())) == {
        "weight",
        "e_score_correction_bias",
    }
    assert output is expected
    run_locality_domain.assert_called_once_with(hidden_states, None, output_dtype=torch.float32)


@pytest.mark.parametrize("seq_len", [1, 32, 8192])
@pytest.mark.parametrize(
    "num_experts, n_group, topk_group, top_k",
    [
        (256, 8, 4, 8),
        (72, 1, 1, 6),
        (384, 1, 1, 8),
        (512, 1, 1, 22),
        (1024, 1, 1, 32),
        (512, 1, 1, 32),
        (256, 8, 2, 8),
        (512, 8, 6, 8),  # fallback
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_noaux_tc_run(seq_len, num_experts, n_group, topk_group, top_k, dtype):
    ROUTED_SCALING_FACTOR = 2.5
    HIDDEN_SIZE = 7168
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    weight = torch.randn((num_experts, HIDDEN_SIZE), dtype=dtype).cuda()
    e_score_correction_bias = torch.randn((num_experts), dtype=torch.float32).cuda()

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
        selected_indices, selected_values = gate.routing_method.apply(gate.forward(logits))

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
            ref_gate.forward(logits)
        )

    # sort before compare
    sorted_selected_values, _ = torch.sort(selected_values)
    ref_sorted_selected_values, _ = torch.sort(ref_selected_values)

    # compare
    torch.cuda.synchronize()

    torch.testing.assert_close(
        sorted_selected_values, ref_sorted_selected_values, rtol=0.01, atol=0.01
    )
