# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint probes for structure beyond a plain decoder.

Each probe closes the same gap: the config says something structural, no
probe claims it, the key lands in `unrecognized`, and the runtime area that
knows about it is never surfaced to the plan phase. The key showing up in
`unrecognized` is the safety net working — but a net is not a map.

Covered here: the sparse-attention indexer, mrope and plain RoPE,
hyper-connections, the vision tower, huge auxiliary embedding tables, hybrid
stacks, attention sinks, asymmetric qk/v head dims, and MoE routing.
"""

from __future__ import annotations

import pytest

from agent_flow.workflows.trtllm_modeling_v2.onboard.runtime_map import (
    areas_for_demands,
    load_index,
)
from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import read_checkpoint_surface


def _names(config):
    return set(read_checkpoint_surface(config).demand_names)


def _detail(config, name):
    for demand in read_checkpoint_surface(config).demands:
        if demand.name == name:
            return demand.detail
    raise AssertionError(f"{name} not produced; got {sorted(_names(config))}")


def _areas(config):
    return {
        a.name
        for a in areas_for_demands(read_checkpoint_surface(config).demand_names, index=load_index())
    }


# ------------------------------------------------- the sparse-attention indexer


def test_an_indexer_is_a_sparse_attention_demand():
    """A learned selector, not a mask.

    A window leaves cache addressing alone; an indexer decides per query
    which past positions are attended to, so the top-k selection is part of
    the forward and part of what a receipt certifies. A target that computes
    full attention here boots and scores close on short prompts.
    """
    config = {"indexer_budget": 2048, "indexer_n_heads": 4, "indexer_head_dim": 128}
    assert "attention.indexer" in _names(config)
    assert "budget=2048" in _detail(config, "attention.indexer")
    assert "sparse_attention" in _areas(config)


def test_the_deepseek_spelling_is_also_an_indexer():
    assert "attention.indexer" in _names({"index_topk": 2048, "index_n_heads": 64})


def test_a_plain_attention_config_produces_no_indexer_demand():
    assert "attention.indexer" not in _names({"num_attention_heads": 32, "head_dim": 128})


# ----------------------------------------------------------------------- rope


def test_rope_is_read_from_rope_parameters_as_well_as_rope_scaling():
    """Transformers 5.x spells the block `rope_parameters`.

    Read only under `rope_scaling`, the ONLY rope demand such a checkpoint
    produces is `rope.partial`, so mrope — the thing a multimodal target is
    most likely to get wrong — is never mentioned.
    """
    config = {"rope_parameters": {"rope_type": "default", "rope_theta": 10000000}}
    assert "rope.plain" in _names(config)
    assert "theta=10000000" in _detail(config, "rope.plain")


def test_a_scaled_rope_under_the_new_name_is_still_scaling():
    config = {"rope_parameters": {"rope_type": "yarn", "factor": 4}}
    assert "rope.scaling" in _names(config)
    assert "yarn" in _detail(config, "rope.scaling")


def test_mrope_is_detected_under_the_new_name_with_its_sections():
    config = {
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        }
    }
    detail = _detail(config, "rope.mrope")
    assert "[11, 11, 10]" in detail
    assert "interleaved" in detail


def test_the_old_rope_scaling_spelling_still_works():
    """Nothing that already worked may change shape."""
    config = {"rope_scaling": {"rope_type": "linear", "factor": 2}}
    assert "rope.scaling" in _names(config)


def test_partial_rotary_is_read_from_either_place():
    assert "rope.partial" in _names({"partial_rotary_factor": 0.25})
    assert "rope.partial" in _names({"rope_parameters": {"partial_rotary_factor": 0.25}})


# --------------------------------------------------------- hyper-connections


def test_hyper_connections_reach_their_runtime_area():
    """An area no demand routes to is never read.

    The residual stream is not one tensor: `hc_count` parallel streams are
    mixed at every block boundary, which changes the shape of every residual
    add. A target that assumes one stream is a different model.
    """
    config = {"hc_count": 4, "hc_lowrank": 320}
    assert "residual.hyper_connection" in _names(config)
    detail = _detail(config, "residual.hyper_connection")
    assert "4 parallel residual streams" in detail
    assert "320" in detail
    assert "hyper_connection" in _areas(config)


def test_the_generic_spelling_is_also_hyper_connections():
    assert "residual.hyper_connection" in _names({"num_residual_streams": 2})


# ------------------------------------------------------------- vision towers


def test_a_vision_tower_reaches_the_multimodal_area():
    """A vision tower reaches the multimodal area.

    Printing "Multimodal checkpoint" is not enough: without a demand, the
    multimodal runtime area stays unread.
    """
    config = {"vision_config": {"depth": 27}, "image_token_id": 248056}
    assert "multimodal.tower" in _names(config)
    assert "multimodal" in _areas(config)


def test_the_tower_is_read_from_the_outer_config_not_the_text_tower():
    """`scoped` is the text tower by then; the vision config is its sibling."""
    config = {"text_config": {"num_attention_heads": 24}, "vision_config": {"depth": 27}}
    assert "multimodal.tower" in _names(config)


def test_placeholder_tokens_alone_are_enough():
    """Some checkpoints declare the tokens and inline the tower."""
    assert "multimodal.tower" in _names({"image_token_id": 1, "video_token_id": 2})


def test_a_text_only_checkpoint_produces_no_tower_demand():
    assert "multimodal.tower" not in _names({"num_attention_heads": 32, "vocab_size": 1000})


# ------------------------------------------------------- huge auxiliary tables


def test_a_secondary_table_far_larger_than_the_vocabulary_is_a_residency_decision():
    """20 M rows against a 248 K vocabulary can be tens of GB of the checkpoint.

    Whether it is resident decides how much of the device is left for KV, so
    it is a decision rather than a default — which is what the `host_offload`
    feature exists to make explicit.
    """
    config = {"vocab_size": 248320, "ngram_vocab_size_base": 20000000}
    assert "embedding.huge_table" in _names(config)
    detail = _detail(config, "embedding.huge_table")
    assert "20,000,000" in detail
    assert "80x" in detail
    assert {"host_offload", "embedding"} <= _areas(config)


def test_a_merely_larger_table_is_not_huge():
    """The threshold is an order of magnitude, not "different".

    A padded or slightly-extended vocabulary is ordinary and must not
    surface a residency decision nobody has to make.
    """
    assert "embedding.huge_table" not in _names({"vocab_size": 100000, "extra_vocab_size": 120000})


def test_the_primary_vocabulary_is_never_its_own_secondary_table():
    assert "embedding.huge_table" not in _names({"vocab_size": 248320})


@pytest.mark.parametrize("value", ["20000000", None, True])
def test_a_non_integer_table_size_is_ignored_rather_than_crashing(value):
    """Configs in the wild carry strings and nulls.

    A probe that raises on one costs the whole survey, which is never worth it.
    """
    assert "embedding.huge_table" not in _names(
        {"vocab_size": 248320, "ngram_vocab_size_base": value}
    )


# ------------------------------------------------------------ the hybrid stack


def _unrecognized(config):
    return set(read_checkpoint_surface(config).unrecognized)


_MIMO_HYBRID = {
    # 39 of one kind and 9 of the other, in the real checkpoint's own order.
    "hybrid_layer_pattern": [0, 1, 1, 1, 1] + ([0] + [1] * 5) * 7 + [0],
    "num_hidden_layers": 48,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 192,
    "v_head_dim": 128,
    "rope_theta": 1e7,
    "swa_num_attention_heads": 64,
    "swa_num_key_value_heads": 8,
    "swa_head_dim": 192,
    "swa_v_head_dim": 128,
    "swa_rope_theta": 1e4,
}


def test_a_per_layer_pattern_is_a_hybrid_stack_not_a_window():
    """`attention.windowing` alone says `sliding_window=128` and nothing else.

    That reads as one window over a uniform stack. This is 48 layers of two
    different kinds, and the difference decides whether the cache manager is
    asked for one geometry or two.
    """
    assert "attention.hybrid" in _names(_MIMO_HYBRID)
    detail = _detail(_MIMO_HYBRID, "attention.hybrid")
    assert "48 layers" in detail
    assert "39 of `1`" in detail and "9 of `0`" in detail
    assert {"sparse_attention", "kv_cache"} <= _areas(_MIMO_HYBRID)


def test_the_survey_reports_the_pattern_and_refuses_to_interpret_it():
    """Which value means sliding-window is the config author's convention.

    On this checkpoint `1` is sliding-window and `0` is full attention, and
    nothing in `config.json` says so. A survey that guessed the polarity
    would silently window the wrong 39 layers, which is worse than handing
    the plan phase the counts and the question.
    """
    detail = _detail(_MIMO_HYBRID, "attention.hybrid")
    assert "question for the plan phase" in detail
    for leak in ("sliding-window layers:", "39 sliding", "9 full"):
        assert leak not in detail


def test_different_head_geometry_between_the_two_kinds_is_the_headline():
    """Not the same shape under two masks: two shapes in one stack.

    The KV head counts differ (8 against 4) and the RoPE bases differ by
    three orders of magnitude, which is the weight manifest's shard math and
    the attention entry's positional scheme, not a mask.
    """
    detail = _detail(_MIMO_HYBRID, "attention.hybrid")
    assert "DIFFERENT geometry" in detail
    assert "swa_num_key_value_heads=8 against num_key_value_heads=4" in detail
    assert "swa_rope_theta" in detail
    # The ones that agree must not be reported as differences.
    assert "swa_head_dim=192 against head_dim=192" not in detail


def test_per_kind_geometry_that_agrees_is_said_to_agree():
    """A hybrid stack whose kinds share a shape is a different, easier target."""
    config = {
        "hybrid_layer_pattern": [0, 1] * 8,
        "num_key_value_heads": 8,
        "swa_num_key_value_heads": 8,
    }
    detail = _detail(config, "attention.hybrid")
    assert "declared and agrees" in detail
    assert "DIFFERENT" not in detail


def test_an_exact_period_is_reported_and_an_inexact_one_is_not_invented():
    """An exact period is reported, and an inexact one is not invented.

    A stack whose global layers land every sixth position and one whose
    first block is short are different stacks.
    """
    assert "repeating every 2 layer(s)" in _detail(
        {"hybrid_layer_pattern": [0, 1] * 12}, "attention.hybrid"
    )
    assert "no fixed period" in _detail(_MIMO_HYBRID, "attention.hybrid")


def test_a_period_that_occurs_only_once_is_not_a_period():
    """A period that occurs only once is not a period.

    A 48-entry list trivially satisfies p=47 when its last entry equals its
    first, and reporting that as a repeat is worse than reporting none.
    """
    from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import _repeating_period

    assert _repeating_period([0, 1, 1, 1, 0]) is None
    assert _repeating_period([0, 1, 0, 1]) == 2


@pytest.mark.parametrize("key", ["hybrid_layer_pattern", "layer_types", "moe_layer_pattern"])
def test_every_per_layer_pattern_spelling_fires(key):
    """Config authors spell the same concept a dozen ways.

    `*_layer_pattern` catches the ones nobody has written down yet.
    """
    assert "attention.hybrid" in _names({key: ["full", "swa", "full", "swa"]})


def test_an_index_list_names_one_kind_and_implies_the_rest():
    """`full_attention_idx` says which layers are full and leaves the rest."""
    detail = _detail(
        {"full_attention_idx": [0, 9, 18, 27], "num_hidden_layers": 36}, "attention.hybrid"
    )
    assert "36 layers, 4 of them named by `full_attention_idx` and 32 not" in detail


def test_a_per_layer_list_that_is_not_attention_does_not_fire():
    """`moe_layer_freq` is the same shape and a different question."""
    assert "attention.hybrid" not in _names({"moe_layer_freq": [0] + [1] * 47})


def test_a_uniform_stack_produces_no_hybrid_demand():
    assert "attention.hybrid" not in _names({"num_attention_heads": 32, "head_dim": 128})


def test_the_per_kind_geometry_keys_stop_being_reported_as_unrecognized():
    """They are the evidence for a demand, not leftovers."""
    left = _unrecognized(_MIMO_HYBRID)
    assert not {k for k in left if k.startswith("swa_")}
    assert "hybrid_layer_pattern" not in left


# ------------------------------------------------------------ attention sinks


_SINKS = {"add_swa_attention_sink_bias": True, "add_full_attention_sink_bias": False}


def test_a_sink_bias_names_the_layer_kind_that_carries_it():
    """A sink on every layer and a sink on the local layers only are different models.

    The layer kind is in the key name and nowhere else.
    """
    detail = _detail(_SINKS, "attention.sink")
    assert "swa layers=True" in detail
    assert "full layers=False" in detail
    assert "sparse_attention" in _areas(_SINKS)


def test_an_unprefixed_sink_flag_is_read_as_applying_to_every_layer():
    assert "all layers=True" in _detail({"attention_sink": True}, "attention.sink")


def test_the_sink_keys_stop_being_reported_as_unrecognized():
    assert not _unrecognized(_SINKS)


def test_the_count_of_sink_tokens_is_still_its_own_demand():
    """`attention.sinks` reads the count spelling; this probe reads the per-layer-kind boolean.

    Neither may swallow the other.
    """
    assert "attention.sinks" in _names({"sinks": 4})
    assert "attention.sink" not in _names({"sinks": 4})


# ------------------------------------------------- asymmetric qk / v head dims


def test_a_v_head_dim_that_differs_is_stated_as_its_own_clause():
    """The single hardest kernel-feasibility question this family asks.

    It decides whether any existing paged-attention entry can be reused at
    all, or whether the target needs a padded-V fallback — so a drafter that
    reads only `head_dim` sizes the queue without the hardest item in it.
    """
    detail = _detail({"head_dim": 192, "v_head_dim": 128}, "attention.head_dim")
    assert "v_head_dim=128 differs from head_dim=192" in detail
    assert "asymmetric qk/v" in detail
    assert "v_head_dim" not in _unrecognized({"head_dim": 192, "v_head_dim": 128})


def test_a_v_head_dim_equal_to_head_dim_adds_no_clause():
    """The ordinary case must stay one line."""
    detail = _detail({"head_dim": 128, "v_head_dim": 128}, "attention.head_dim")
    assert detail == "128, declared not derived"


def test_a_checkpoint_with_no_v_head_dim_is_unchanged():
    assert _detail({"head_dim": 256}, "attention.head_dim") == "256, declared not derived"


# --------------------------------------------------------------- moe routing


def test_the_routing_method_is_reported_beside_the_scoring_function():
    """The scoring function says how an expert is SCORED; these say how the top-k is SELECTED.

    They pick a different routing entry rather than a different constant.
    """
    detail = _detail(
        {
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "n_group": 1,
            "topk_group": 1,
        },
        "moe.routing",
    )
    assert "scoring=sigmoid" in detail
    assert "topk_method=noaux_tc" in detail
    assert "n_group=1" in detail
    assert "topk_group=1" in detail


def test_a_plain_moe_reports_no_routing_method_it_does_not_have():
    detail = _detail({"n_routed_experts": 8, "num_experts_per_tok": 2}, "moe.routing")
    assert "topk_method" not in detail
    assert "topk_group" not in detail


# -------------------------------------------------- the invariant behind all of it


def test_every_demand_a_runtime_area_declares_is_one_some_probe_can_produce():
    """A dangling demand name is an area that can never be surfaced.

    An area may only claim demands the survey knows how to emit, and a probe
    may only emit demands some area claims.
    """
    import pathlib
    import re

    import yaml

    from agent_flow.workflows.trtllm_modeling_v2.onboard import survey as survey_mod

    index = yaml.safe_load(
        (pathlib.Path(survey_mod.__file__).parent / "runtime_index.yaml").read_text()
    )
    source = pathlib.Path(survey_mod.__file__).read_text()
    produced = set(re.findall(r'Demand\(\s*"([a-z_.]+)"', source))

    declared: dict[str, list[str]] = {}
    for name, area in index.items():
        if name == "_overview":
            continue
        for demand in area.get("demands") or []:
            declared.setdefault(demand, []).append(name)

    dangling = {d: owners for d, owners in declared.items() if d not in produced}
    assert not dangling, f"areas declare demands no probe emits: {dangling}"

    orphan = sorted(d for d in produced if d not in declared)
    assert not orphan, f"probes emit demands no area claims: {orphan}"


# ------------------------------------------- a DSpark drafter under mtp.*

# A DSpark checkpoint's own `text_config`, trimmed to the keys that matter.
_DSPARK = {
    "architectures": ["DeepseekV41ForCausalLM"],
    "text_config": {
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_markov_rank": 256,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39],
    },
}


def test_a_dspark_drafter_is_not_reported_as_mtp():
    """The survey is what the plan drafter reads first.

    Saying "MTP is the checkpoint's own" for a checkpoint whose `mtp.*`
    stages are a DSpark block drafter is how a plan ends up requesting the
    feature that builds the wrong drafter.
    """
    detail = _detail(_DSPARK, "speculation.heads")
    assert "`dspark`, NOT `mtp`" in detail
    assert "block 5" in detail
    assert "MTP is the checkpoint's own" not in detail
    assert not {k for k in _unrecognized(_DSPARK) if k.startswith("dspark_")}


def test_a_plain_mtp_checkpoint_still_reads_as_mtp():
    detail = _detail({"num_nextn_predict_layers": 1}, "speculation.heads")
    assert "MTP is the checkpoint's own" in detail
