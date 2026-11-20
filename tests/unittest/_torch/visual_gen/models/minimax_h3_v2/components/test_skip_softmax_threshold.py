# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.cuda_graph_runner import resolved_extra_keys_scope
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.components.skip_softmax_threshold import (
    H3SkipSoftmax,
    resolve_skip_softmax_threshold_scale_factor,
)
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig

pytestmark = pytest.mark.cpu_only


def test_constant_threshold_compiles_in_one_graph():
    params = H3SkipSoftmax(4.0, None)

    @torch.compile(backend="eager", fullgraph=True)
    def forward(x):
        return x + resolve_skip_softmax_threshold_scale_factor(None, params, x)

    torch.testing.assert_close(forward(torch.zeros(1)), torch.tensor([4.0]))


def test_compiled_schedule_resolves_timestep_on_host():
    params = H3SkipSoftmax(4.0, 0.6)
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


def test_checkpoint_calibration_and_ignored_layers():
    config = {
        "sparse_attention_config": {
            "ignore": ["refiner_*"],
            "config_groups": {
                "attention": {
                    "algorithm": "skip_softmax",
                    "target_sparsity": 0.75,
                    "disabled_until_timestep": 0.6,
                    "ignore": ["transformer_blocks.1.*"],
                    "threshold_scale_factor": {
                        "formula": "coefficient * target_sparsity",
                        "coefficients": {"coefficient": 4.0},
                    },
                }
            },
        }
    }
    recipe = SkipSoftmaxAttentionConfig()
    params = H3SkipSoftmax.from_config(recipe, "transformer.transformer_blocks.0.attn", config)
    assert params == H3SkipSoftmax(3.0, 0.6)
    for name in ("transformer.transformer_blocks.1.attn", "transformer.refiner_blocks.0.attn"):
        assert H3SkipSoftmax.from_config(recipe, name, config) is None
    user_override = SkipSoftmaxAttentionConfig(threshold_scale_factor=0.0)
    assert (
        H3SkipSoftmax.from_config(user_override, "transformer.transformer_blocks.0.attn", config)
        is None
    )


def test_checkpoint_calibration_formula_location():
    recipe = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
    coefficients = {"formula": "a * target_sparsity + b", "a": 4.0, "b": 0.5}
    nested = {"threshold_scale_factor": {"coefficients": coefficients}}
    assert (
        recipe.to_sparse_params(pretrained_config=nested).scheduler.threshold_scale_factor_prefill
        == 2.5
    )
    assert H3SkipSoftmax.from_config(recipe, "attn", nested) == H3SkipSoftmax(2.5, None)

    misplaced = {"coefficients": coefficients}
    with pytest.raises(ValueError):
        recipe.to_sparse_params(pretrained_config=misplaced)
    with pytest.raises(ValueError):
        H3SkipSoftmax.from_config(recipe, "attn", misplaced)


def test_unconfigured_and_ignored_layers_skip_calibration():
    recipe = SkipSoftmaxAttentionConfig()
    assert H3SkipSoftmax.from_config(recipe, "trunk.attn", {}) is None
    missing_formula = {"target_sparsity": 0.25, "ignore": ["refiner.*"]}
    assert H3SkipSoftmax.from_config(recipe, "refiner.attn", missing_formula) is None
    with pytest.raises(ValueError, match="calibration"):
        H3SkipSoftmax.from_config(recipe, "trunk.attn", missing_formula)


def test_trtllm_native_skip_threshold_tracks_timestep(monkeypatch):
    from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.components.trtllm_attention import (
        H3TrtllmAttention,
    )
    from tensorrt_llm.bindings.internal import thop

    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")
    attn = H3TrtllmAttention(
        num_heads=2,
        head_dim=128,
        attention_metadata_state={},
        sparse_params=H3SkipSoftmax(4.0, 0.6),
    )
    thresholds = []

    def record_native(params):
        thresholds.append(params.fwd.sparse_runtime_params.threshold_scale_factor_prefill)
        assert params.attn.skip_softmax_stat is attn.skip_stats
        params.attn.skip_softmax_stat.copy_(torch.tensor([12, 3], dtype=torch.uint32))
        params.output.zero_()

    monkeypatch.setattr(thop, "run_context", record_native)
    qkv = torch.zeros(1, 3, 6 * 128, dtype=torch.bfloat16)
    for timestep in (0.7, 0.5):
        attn.forward(qkv, batch_size=1, seq_len=3, timestep=torch.tensor(timestep))
    assert thresholds == [0.0, 4.0]
    assert attn.skip_stats.tolist() == [12, 3]
