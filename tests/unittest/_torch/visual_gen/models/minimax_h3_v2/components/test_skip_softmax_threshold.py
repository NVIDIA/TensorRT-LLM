# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.cuda_graph_runner import resolved_extra_keys_scope
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.components.skip_softmax_threshold import (
    resolve_skip_softmax_threshold_scale_factor,
)
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig

pytestmark = pytest.mark.cpu_only


def test_constant_threshold_compiles_in_one_graph():
    params = SkipSoftmaxAttentionConfig(threshold_scale_factor=4.0).to_sparse_params()

    @torch.compile(backend="eager", fullgraph=True)
    def forward(x):
        return x + resolve_skip_softmax_threshold_scale_factor(None, params, x)

    torch.testing.assert_close(forward(torch.zeros(1)), torch.tensor([4.0]))


def test_compiled_schedule_resolves_timestep_on_host():
    params = SkipSoftmaxAttentionConfig(
        threshold_scale_factor=4.0, disabled_until_timestep=0.6
    ).to_sparse_params()
    graphs = []

    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward

    @torch.compile(backend=capture)
    def forward(x, timestep):
        threshold = resolve_skip_softmax_threshold_scale_factor(None, params, timestep)
        return x + (threshold or 0.0)

    x = torch.ones(1)
    torch.testing.assert_close(forward(x, torch.tensor([0.0, 0.7])), x)
    torch.testing.assert_close(forward(x, torch.tensor([0.0, 0.5])), x + 4.0)
    with resolved_extra_keys_scope({"sparse_attn_phase": 0}):
        torch.testing.assert_close(forward(x, torch.tensor([float("nan")])), x)
    assert graphs
    assert all(node.target not in ("amax", "item") for g in graphs for node in g.graph.nodes)
