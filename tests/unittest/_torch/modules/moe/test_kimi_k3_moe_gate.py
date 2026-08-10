# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the Kimi K3 MoE gate fused-routing fast path.

``KimiK3MoEGate.forward`` routes eligible configs (per-expert sigmoid
scoring, ``num_expert_group == 1``, renormalize on, no mutation controls,
within the kernel's supported bounds) through the fused
``torch.ops.trtllm.noaux_tc_op`` kernel via ``Deepseekv3RoutingImpl``.

These tests assert:

* the fused path is numerically equivalent to the eager
  ``sigmoid -> +bias select -> top-k -> gather raw scores -> renormalize
  (sum + 1e-20) -> scale`` reference it replaces (same experts selected,
  same weights), and preserves the eager dtype contract (int64 indices,
  fp32 weights) every downstream consumer relies on; and
* ineligible configs (softmax scoring, grouped routing, renormalize off,
  any mutation control) keep the eager reference path.
"""

import dataclasses

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_k3_moe.kimi_k3_moe_gate import KimiK3MoEGate


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
def test_fused_routing_matches_eager():
    torch.manual_seed(0)
    cfg = _GateConfig()
    gate = KimiK3MoEGate(cfg).cuda()
    with torch.no_grad():
        gate.weight.normal_(0.0, 0.05)
        gate.e_score_correction_bias.normal_(0.0, 0.1)

    # The stock K3 config takes the fused fast path.
    assert gate._use_fused_routing is True

    num_tokens = 17
    hidden = torch.randn(1, num_tokens, cfg.hidden_size, device="cuda")

    idx_fused, wt_fused = gate(hidden)

    # Dtype/shape contract preserved for downstream consumers: the python
    # fallback's ``scatter_`` needs int64 indices; the weighted sum consumes
    # fp32 weights.
    assert idx_fused.dtype == torch.int64
    assert wt_fused.dtype == torch.float32
    assert idx_fused.shape == (num_tokens, cfg.num_experts_per_token)
    assert wt_fused.shape == (num_tokens, cfg.num_experts_per_token)

    # Force the eager reference path on the same gate/weights/input.
    gate._use_fused_routing = False
    idx_eager, wt_eager = gate(hidden)

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


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(softmax_routing_mutation=True),
        dict(biased_weights_mutation=True),
        dict(omit_renormalize_mutation=True),
    ],
)
def test_mutation_controls_disable_fused_routing(kwargs):
    # The mutation controls change the routing math, so they must fall back
    # to the eager reference rather than the fused kernel.
    gate = KimiK3MoEGate(_GateConfig(), **kwargs)
    assert gate._use_fused_routing is False


@pytest.mark.parametrize(
    "cfg",
    [
        _GateConfig(moe_router_activation_func="softmax"),
        _GateConfig(num_expert_group=4, topk_group=2),
        _GateConfig(moe_renormalize=False),
        _GateConfig(num_experts_per_token=1),
    ],
)
def test_ineligible_configs_disable_fused_routing(cfg):
    # softmax scoring, grouped routing, renormalize off, and top_k == 1 all
    # diverge from the fused kernel's fixed contract -> eager path.
    assert KimiK3MoEGate(cfg)._use_fused_routing is False
