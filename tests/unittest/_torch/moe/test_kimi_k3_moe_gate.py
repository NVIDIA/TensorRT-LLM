# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the production Kimi K3 MoE routing method."""

import dataclasses

import pytest
import torch
from _torch.moe.kimi_k3_ref_moe.kimi_k3_moe_block import KimiK3ReferenceMoEGate

from tensorrt_llm._torch.models.modeling_kimi_linear import KimiK3MoEGate


@dataclasses.dataclass
class _GateConfig:
    """Minimal config carrying the fields ``KimiK3MoEGate`` reads."""

    hidden_size: int = 512
    # 256 experts / top_k 8 / n_group 1 is a canonical noaux_tc shape and
    # sits inside the n_group == 1 kernel bounds (num_experts <= 1024,
    # top_k <= 32) that K3 (896 / 16) also satisfies.
    num_experts: int = 256
    num_experts_per_token: int = 8
    moe_renormalize: bool = True
    moe_router_activation_func: str = "sigmoid"
    routed_scaling_factor: float = 2.5
    num_expert_group: int = 1
    topk_group: int = 1


def _dense_weights(
    topk_idx: torch.Tensor, topk_weight: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Scatter per-token ``(index, weight)`` pairs into a dense
    ``[num_tokens, num_experts]`` map so the fused and eager outputs can be
    compared without depending on the (unsorted) top-k ordering."""
    dense = topk_weight.new_zeros((topk_idx.shape[0], num_experts))
    dense.scatter_(1, topk_idx.to(torch.int64), topk_weight.to(dense.dtype))
    return dense


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noaux_tc_op is a CUDA custom op")
def test_fused_routing_matches_eager_reference():
    torch.manual_seed(0)
    cfg = _GateConfig()
    gate = KimiK3MoEGate(cfg).cuda()
    reference_gate = KimiK3ReferenceMoEGate(cfg).cuda()
    with torch.no_grad():
        gate.weight.normal_(0.0, 0.05)
        gate.e_score_correction_bias.normal_(0.0, 0.1)
        reference_gate.load_state_dict(gate.state_dict())

    num_tokens = 17
    hidden = torch.randn(1, num_tokens, cfg.hidden_size, device="cuda")

    idx_eager, wt_eager = reference_gate(hidden)
    idx_fused, wt_fused = gate.routing_method.apply(gate.compute_logits(hidden))

    assert idx_fused.shape == (num_tokens, cfg.num_experts_per_token)
    assert wt_fused.shape == (num_tokens, cfg.num_experts_per_token)

    # Same experts selected per token (top-k is unsorted, so compare sets).
    sel_fused = torch.sort(idx_fused, dim=-1).values
    sel_eager = torch.sort(idx_eager.to(torch.int64), dim=-1).values
    assert torch.equal(sel_fused, sel_eager)

    # Same renormalized + scaled weight on each selected expert.
    torch.testing.assert_close(
        _dense_weights(idx_fused, wt_fused, cfg.num_experts),
        _dense_weights(idx_eager, wt_eager, cfg.num_experts),
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Kimi fused route+MXFP8 quant requires an SM10x architecture",
)
@pytest.mark.parametrize("num_tokens", [1, 5, 64])
def test_fused_route_quant_matches_unfused_chain(num_tokens):
    torch.manual_seed(0x5EED + num_tokens)
    scores = torch.randn(num_tokens, 896, dtype=torch.float32, device="cuda")
    bias = torch.randn(896, dtype=torch.float32, device="cuda")
    hidden_states = torch.randn(num_tokens, 3584, dtype=torch.bfloat16, device="cuda")
    routed_scaling_factor = 2.446

    ref_scales, ref_experts = torch.ops.trtllm.noaux_tc_op(
        scores, bias, 1, 1, 16, routed_scaling_factor
    )
    ref_quantized, ref_quant_scales = torch.ops.trtllm.mxfp8_quantize(
        hidden_states, False, alignment=256
    )

    experts, scales, quantized, quant_scales = torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(
        scores,
        bias,
        hidden_states,
        routed_scaling_factor,
    )

    assert torch.equal(experts, ref_experts)
    assert torch.equal(scales.view(torch.int16), ref_scales.to(torch.bfloat16).view(torch.int16))
    assert torch.equal(quantized.view(torch.uint8), ref_quantized.view(torch.uint8))
    assert torch.equal(quant_scales, ref_quant_scales.view(num_tokens, -1))
