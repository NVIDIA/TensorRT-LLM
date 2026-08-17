# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.interface import (
    SharedConfig,
    Stages,
    TransformConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.library.moe_routing import MatchNoAuxTCPattern


class _NoAuxTCRouter(nn.Module):
    def __init__(
        self,
        hidden: int = 1024,
        n_experts: int = 128,
        n_group: int = 8,
        topk_group: int = 4,
        top_k: int = 8,
        scale: float = 2.5,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, hidden))
        self.register_buffer("e_score_correction_bias", torch.zeros(n_experts, dtype=torch.float32))
        self.n_group = n_group
        self.topk_group = topk_group
        self.top_k = top_k
        self.scale = scale
        self.n_experts = n_experts

    def forward(self, hidden_states):
        T = hidden_states.shape[0]
        E, G = self.n_experts, self.n_group
        logits = F.linear(hidden_states.float(), self.weight)
        scores = logits.sigmoid()
        scores_with_bias = scores + self.e_score_correction_bias
        grouped = scores_with_bias.view(T, G, E // G)
        group_scores = grouped.topk(2, dim=-1).values.sum(-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1).indices
        group_mask = torch.zeros_like(group_scores).scatter_(-1, group_idx, 1.0)
        score_mask = group_mask.unsqueeze(-1).expand(T, G, E // G).reshape(T, E)
        masked = torch.where(
            score_mask.bool(),
            scores_with_bias,
            torch.tensor(float("-inf"), dtype=scores_with_bias.dtype),
        )
        topk_idx = torch.topk(masked, k=self.top_k, dim=-1).indices
        topk_w = scores.gather(-1, topk_idx)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True) * self.scale
        return topk_w, topk_idx


class _PlainSoftmaxRouter(nn.Module):
    def __init__(self, hidden: int = 64, n_experts: int = 16, top_k: int = 2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, hidden))
        self.top_k = top_k

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.float(), self.weight)
        weights = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(weights, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, indices


def _apply_matcher(gm: torch.fx.GraphModule) -> int:
    transform = MatchNoAuxTCPattern(TransformConfig(stage=Stages.PATTERN_MATCHER))
    _, info = transform._apply(gm, cm=None, factory=None, shared_config=SharedConfig())
    return info.num_matches


def _is_noaux_tc(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    packet = torch.ops.trtllm.noaux_tc_op
    return node.target is packet or node.target is getattr(packet, "default", None)


def _find_noaux_tc_node(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if _is_noaux_tc(node):
            return node
    return None


def test_noaux_tc_pattern_matches_and_extracts_constants():
    n_group, topk_group, top_k, scale = 8, 4, 8, 2.5
    module = _NoAuxTCRouter(n_group=n_group, topk_group=topk_group, top_k=top_k, scale=scale).eval()
    gm = torch_export_to_gm(module, args=(torch.randn(4, 1024),))

    assert _apply_matcher(gm) == 1

    fused = _find_noaux_tc_node(gm)
    assert fused is not None

    # noaux_tc_op(scores, bias, n_group, topk_group, top_k, routed_scaling_factor)
    assert fused.args[2] == n_group
    assert fused.args[3] == topk_group
    assert fused.args[4] == top_k
    assert fused.args[5] == pytest.approx(scale)


def test_noaux_tc_pattern_replaces_outputs():
    module = _NoAuxTCRouter().eval()
    gm = torch_export_to_gm(module, args=(torch.randn(4, 1024),))
    assert _apply_matcher(gm) == 1

    fused = _find_noaux_tc_node(gm)
    assert fused is not None

    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    out_args = output_node.args[0]
    if not isinstance(out_args, (list, tuple)):
        out_args = (out_args,)

    fused_origins = set()
    for out in out_args:
        if (
            isinstance(out, torch.fx.Node)
            and out.op == "call_function"
            and out.target is operator.getitem
            and out.args[0] is fused
        ):
            fused_origins.add(out.args[1])
    assert fused_origins == {0, 1}


def test_plain_softmax_router_is_not_matched():
    module = _PlainSoftmaxRouter().eval()
    gm = torch_export_to_gm(module, args=(torch.randn(4, 64),))
    assert _apply_matcher(gm) == 0
    assert _find_noaux_tc_node(gm) is None
