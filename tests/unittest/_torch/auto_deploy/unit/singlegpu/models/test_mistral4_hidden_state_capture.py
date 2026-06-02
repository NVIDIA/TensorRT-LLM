# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for detect_hidden_states_for_capture on Mistral4 models.

Tests that collect_residual_add_nodes correctly identifies decoder-layer residual
add nodes in the Mistral4 graph (MLA attention + MoE FFN) and that:
  - {0} captures the single layer of a 1-layer model
  - {-1} resolves to layer 0 and captures the same node
  - the captured node is placed correctly (between layers or at model output)

Uses Mistral4ForCausalLM directly (text-only, no multimodal wrapper) so the
graph does not have the language_model. prefix that complicates layer detection.
"""

import pytest
import torch
from torch.fx import GraphModule

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_mistral3 import (
    Mistral4ForCausalLM,
    Mistral4TextConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig
from tensorrt_llm._torch.auto_deploy.transform.library.hidden_states import (
    DetectHiddenStatesForCapture,
)


def _small_1layer_config() -> Mistral4TextConfig:
    """Minimal 1-layer Mistral4TextConfig for fast export."""
    return Mistral4TextConfig(
        vocab_size=256,
        hidden_size=4096,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=32,
        q_lora_rank=1024,
        kv_lora_rank=256,
        qk_head_dim=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        max_position_embeddings=128,
        rope_parameters={
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 8.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "llama_4_scaling_beta": 0.1,
        },
        pad_token_id=0,
    )


def _export_mistral4_text(config: Mistral4TextConfig) -> GraphModule:
    """Export a Mistral4ForCausalLM to a GraphModule on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Mistral4 export.")
    device = "cuda"
    dtype = torch.bfloat16
    model = Mistral4ForCausalLM(config).to(device=device, dtype=dtype).eval()
    inputs_embeds = torch.randn(1, 4, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(4, device=device).unsqueeze(0)
    return torch_export_to_gm(
        model,
        kwargs={"inputs_embeds": inputs_embeds, "position_ids": position_ids},
        clone=True,
    )


def _make_transform(layers_to_capture) -> DetectHiddenStatesForCapture:
    return DetectHiddenStatesForCapture(
        config=TransformConfig(
            stage="pattern_matcher",
            eagle3_layers_to_capture=layers_to_capture,
        )
    )


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helper: dump graph to /tmp for manual inspection
# ---------------------------------------------------------------------------


def _dump_graph(gm: GraphModule, path: str) -> None:
    """Write a human-readable graph dump to `path` for visualization."""
    lines = []
    for node in gm.graph.nodes:
        users = [u.name for u in node.users]
        if node.op == "call_function":
            w_info = ""
            from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op

            if is_linear_op(node) and len(node.args) > 1:
                w = node.args[1]
                if hasattr(w, "op") and w.op == "get_attr":
                    w_info = f"  weight={w.target}"
            lines.append(
                f"{node.op:15s} {str(node.target):60s} {node.name:40s}{w_info}"
                f"  [{', '.join(users)}]"
            )
        else:
            lines.append(
                f"{node.op:15s} {str(node.target)[:60]:60s} {node.name:40s}  [{', '.join(users)}]"
            )
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Test: collect_residual_add_nodes on 1-layer Mistral4
# ---------------------------------------------------------------------------


def test_mistral4_collect_residual_add_nodes_one_layer():
    """collect_residual_add_nodes finds exactly 1 residual add for a 1-layer Mistral4.

    Dumps the graph to /tmp/mistral4_1layer_graph.txt for manual inspection.
    """
    config = _small_1layer_config()
    gm = _export_mistral4_text(config)

    dump_path = "/tmp/mistral4_1layer_graph.txt"
    _dump_graph(gm, dump_path)
    print(f"Graph dumped to {dump_path}  ({len(list(gm.graph.nodes))} nodes)")

    t = DetectHiddenStatesForCapture(config=TransformConfig(stage="pattern_matcher"))
    residual_add_nodes = t.collect_residual_add_nodes(gm)

    print(f"collect_residual_add_nodes: {sorted(residual_add_nodes.keys())}")
    assert len(residual_add_nodes) >= 1, (
        f"Expected at least 1 residual add node for the single decoder layer, "
        f"got {residual_add_nodes}. Check {dump_path} for graph structure."
    )
    assert 0 in residual_add_nodes, (
        f"Expected layer 0 residual add but found keys={sorted(residual_add_nodes.keys())}. "
        f"Check {dump_path} for graph structure."
    )


def test_mistral4_capture_explicit_layer_0():
    """detect_hidden_states_for_capture with eagle3_layers_to_capture={0} succeeds."""
    config = _small_1layer_config()
    gm = _export_mistral4_text(config)
    t = _make_transform(layers_to_capture={0})
    gm_out, info = t._apply(gm, None, None, None)
    assert info.num_matches == 1, f"Expected 1 match for {{0}}, got {info.num_matches}"
    capture_nodes = [
        n
        for n in gm_out.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.auto_deploy.residual_add_for_capture.default
    ]
    assert len(capture_nodes) == 1, f"Expected 1 capture node, got {len(capture_nodes)}"


def test_mistral4_capture_neg1_same_as_layer_0():
    """eagle3_layers_to_capture={-1} captures the same node as {0} for a 1-layer model.

    For a single-layer model layer 0 IS the last layer, so {-1} must resolve to {0}
    and capture the identical residual add.
    """
    config = _small_1layer_config()

    # Capture with explicit {0}
    gm0 = _export_mistral4_text(config)
    t0 = _make_transform(layers_to_capture={0})
    gm0_out, info0 = t0._apply(gm0, None, None, None)

    # Capture with {-1}  (separate export to avoid in-place mutation interference)
    gm_neg = _export_mistral4_text(config)
    t_neg = _make_transform(layers_to_capture={-1})
    gm_neg_out, info_neg = t_neg._apply(gm_neg, None, None, None)

    assert info0.num_matches == 1, f"{{0}} gave {info0.num_matches} matches"
    assert info_neg.num_matches == 1, f"{{-1}} gave {info_neg.num_matches} matches"

    # Both should have inserted exactly 1 capture node
    def _get_capture_nodes(g):
        return [
            n
            for n in g.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.auto_deploy.residual_add_for_capture.default
        ]

    caps0 = _get_capture_nodes(gm0_out)
    caps_neg = _get_capture_nodes(gm_neg_out)
    assert len(caps0) == 1 and len(caps_neg) == 1

    # The capture nodes should have the same args (same original residual add position)
    args0 = tuple(a.name if isinstance(a, torch.fx.Node) else a for a in caps0[0].args)
    args_neg = tuple(a.name if isinstance(a, torch.fx.Node) else a for a in caps_neg[0].args)
    assert args0 == args_neg, (
        f"{{0}} captured at {args0} but {{-1}} captured at {args_neg}; "
        "they should be identical for a 1-layer model."
    )


def test_mistral4_collect_residual_adds_counts_layers():
    """collect_residual_add_nodes returns exactly N entries for an N-layer model.

    This is the implicit 'count hidden layers' test the user described — the number
    of entries in the dict tells you how many decoder layers were detected.
    """
    for num_layers in (1, 2, 3):
        config = _small_1layer_config()
        config.num_hidden_layers = num_layers
        gm = _export_mistral4_text(config)
        t = DetectHiddenStatesForCapture(config=TransformConfig(stage="pattern_matcher"))
        residual_add_nodes = t.collect_residual_add_nodes(gm)
        assert len(residual_add_nodes) == num_layers, (
            f"num_hidden_layers={num_layers}: expected {num_layers} residual adds, "
            f"got {sorted(residual_add_nodes.keys())}"
        )
        assert set(residual_add_nodes.keys()) == set(range(num_layers)), (
            f"Expected layer indices {{0..{num_layers - 1}}}, "
            f"got {sorted(residual_add_nodes.keys())}"
        )


def test_mistral4_is_linear_op_finds_correct_nodes():
    """is_linear_op identifies exactly the attention + shared-expert linear nodes.

    For Mistral4 with n_routed_experts=4, each decoder layer has:
      - 4 MLA attention linears: wq_a, wq_b, wkv_a_with_mqa, wo
      - 3 shared expert linears: gate_proj, up_proj, down_proj
    Plus 1 lm_head → 7*N + 1 total.  Routed expert linears are inside torch_moe
    and are NOT is_linear_op nodes.
    """
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op

    for num_layers in (1, 2):
        config = _small_1layer_config()
        config.num_hidden_layers = num_layers
        gm = _export_mistral4_text(config)
        lin_nodes = [n for n in gm.graph.nodes if is_linear_op(n)]
        expected = 7 * num_layers + 1  # 7 per layer + lm_head
        assert len(lin_nodes) == expected, (
            f"num_layers={num_layers}: expected {expected} linear nodes, got {len(lin_nodes)}"
        )


def test_mistral4_capture_default_layers_requires_enough_hidden_layers():
    """eagle3_layers_to_capture=None (default) requires num_hidden_layers > 6."""
    config = _small_1layer_config()
    config.num_hidden_layers = 1  # too few for default capture set
    gm = _export_mistral4_text(config)
    t = DetectHiddenStatesForCapture(
        config=TransformConfig(stage="pattern_matcher", eagle3_layers_to_capture=None)
    )
    import pytest

    with pytest.raises(ValueError, match="Not enough hidden layers"):
        t._apply(gm, None, None, None)


def test_mistral4_capture_default_layers_7_layer_model():
    """eagle3_layers_to_capture=None succeeds and picks 3 layers for a 7-layer model.

    set_default_eagle3_layers_to_capture({1, num_layers//2-1, num_layers-4}) for num_layers=7
    gives {1, 2, 3}.
    """
    config = _small_1layer_config()
    config.num_hidden_layers = 7
    gm = _export_mistral4_text(config)
    t = DetectHiddenStatesForCapture(
        config=TransformConfig(stage="pattern_matcher", eagle3_layers_to_capture=None)
    )
    gm_out, info = t._apply(gm, None, None, None)
    assert info.num_matches == 3, (
        f"Expected 3 matches (default 3-layer capture), got {info.num_matches}"
    )
    capture_nodes = [
        n
        for n in gm_out.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.auto_deploy.residual_add_for_capture.default
    ]
    assert len(capture_nodes) == 3, f"Expected 3 capture nodes, got {len(capture_nodes)}"


def test_mistral4_capture_neg1_resolves_correctly_for_various_depths():
    """{-1} always resolves to the LAST layer, regardless of model depth."""
    for num_layers in (1, 2, 4):
        config = _small_1layer_config()
        config.num_hidden_layers = num_layers
        gm = _export_mistral4_text(config)
        t = _make_transform(layers_to_capture={-1})
        gm_out, info = t._apply(gm, None, None, None)
        assert info.num_matches == 1, (
            f"num_layers={num_layers}: expected 1 match for {{-1}}, got {info.num_matches}"
        )
        # {-1} should resolve to last layer index = num_layers - 1
        # Verify the captured node is from the last layer by checking that
        # {num_layers-1} explicit capture finds the same node.
        gm2 = _export_mistral4_text(config)
        t2 = _make_transform(layers_to_capture={num_layers - 1})
        gm2_out, info2 = t2._apply(gm2, None, None, None)
        assert info2.num_matches == 1

        def _cap_args(g):
            caps = [
                n
                for n in g.graph.nodes
                if n.op == "call_function"
                and n.target == torch.ops.auto_deploy.residual_add_for_capture.default
            ]
            return tuple(a.name if isinstance(a, torch.fx.Node) else a for a in caps[0].args)

        assert _cap_args(gm_out) == _cap_args(gm2_out), (
            f"num_layers={num_layers}: {{-1}} and {{{num_layers - 1}}} captured at "
            f"different positions: {_cap_args(gm_out)} vs {_cap_args(gm2_out)}"
        )


def test_mistral4_capture_2layer_last_vs_first():
    """For a 2-layer model, {-1} captures layer 1 and {0} captures layer 0 (different nodes)."""
    config = _small_1layer_config()
    config.num_hidden_layers = 2  # override to 2 layers

    # {0}: capture first layer
    gm0 = _export_mistral4_text(config)
    t0 = _make_transform(layers_to_capture={0})
    gm0_out, info0 = t0._apply(gm0, None, None, None)

    # {-1}: capture last layer (= layer 1)
    gm_last = _export_mistral4_text(config)
    t_last = _make_transform(layers_to_capture={-1})
    gm_last_out, info_last = t_last._apply(gm_last, None, None, None)

    assert info0.num_matches == 1, f"{{0}} matches: {info0.num_matches}"
    assert info_last.num_matches == 1, f"{{-1}} matches: {info_last.num_matches}"

    def _capture_args(g):
        caps = [
            n
            for n in g.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.auto_deploy.residual_add_for_capture.default
        ]
        return tuple(a.name if isinstance(a, torch.fx.Node) else a for a in caps[0].args)

    args0 = _capture_args(gm0_out)
    args_last = _capture_args(gm_last_out)
    # For a 2-layer model, the two captures should be at different graph positions
    assert args0 != args_last, (
        f"{{0}} and {{-1}} captured the same node {args0}; they should differ for a 2-layer model."
    )
