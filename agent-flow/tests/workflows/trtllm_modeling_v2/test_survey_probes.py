# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint probes for structure beyond a plain decoder, and the areas they surface."""

from __future__ import annotations

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


def _unrecognized(config):
    return set(read_checkpoint_surface(config).unrecognized)


def test_an_indexer_is_a_sparse_attention_demand():
    config = {"indexer_budget": 2048, "indexer_n_heads": 4, "indexer_head_dim": 128}
    assert "attention.indexer" in _names(config)
    assert "budget=2048" in _detail(config, "attention.indexer")
    assert "sparse_attention" in _areas(config)


def test_rope_is_read_from_rope_parameters_as_well_as_rope_scaling():
    config = {"rope_parameters": {"rope_type": "default", "rope_theta": 10000000}}
    assert "rope.plain" in _names(config)
    assert "theta=10000000" in _detail(config, "rope.plain")


def test_hyper_connections_reach_their_runtime_area():
    config = {"hc_count": 4, "hc_lowrank": 320}
    assert "residual.hyper_connection" in _names(config)
    detail = _detail(config, "residual.hyper_connection")
    assert "4 parallel residual streams" in detail
    assert "320" in detail
    assert "hyper_connection" in _areas(config)


def test_a_vision_tower_reaches_the_multimodal_area():
    config = {"vision_config": {"depth": 27}, "image_token_id": 248056}
    assert "multimodal.tower" in _names(config)
    assert "multimodal" in _areas(config)


def test_a_secondary_table_far_larger_than_the_vocabulary_is_a_residency_decision():
    config = {"vocab_size": 248320, "ngram_vocab_size_base": 20000000}
    assert "embedding.huge_table" in _names(config)
    detail = _detail(config, "embedding.huge_table")
    assert "20,000,000" in detail
    assert "80x" in detail
    assert {"host_offload", "embedding"} <= _areas(config)


# ------------------------------------------------------------ the hybrid stack


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
    assert "attention.hybrid" in _names(_MIMO_HYBRID)
    detail = _detail(_MIMO_HYBRID, "attention.hybrid")
    assert "48 layers" in detail
    assert "39 of `1`" in detail and "9 of `0`" in detail
    assert {"sparse_attention", "kv_cache"} <= _areas(_MIMO_HYBRID)


def test_the_survey_reports_the_pattern_and_refuses_to_interpret_it():
    """On this checkpoint `1` is sliding-window and `0` is full attention; nothing says so."""
    detail = _detail(_MIMO_HYBRID, "attention.hybrid")
    assert "question for the plan phase" in detail
    for leak in ("sliding-window layers:", "39 sliding", "9 full"):
        assert leak not in detail


# ------------------------------------------------------- the rest of attention


def test_a_sink_bias_names_the_layer_kind_that_carries_it():
    sinks = {"add_swa_attention_sink_bias": True, "add_full_attention_sink_bias": False}
    detail = _detail(sinks, "attention.sink")
    assert "swa layers=True" in detail
    assert "full layers=False" in detail
    assert "sparse_attention" in _areas(sinks)


# -------------------------------------------------- the invariant behind all of it


def test_every_demand_a_runtime_area_declares_is_one_some_probe_can_produce():
    """An area may only claim demands the survey emits, and a probe only emit claimed ones."""
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
    detail = _detail(_DSPARK, "speculation.heads")
    assert "`dspark`, NOT `mtp`" in detail
    assert "block 5" in detail
    assert "MTP is the checkpoint's own" not in detail
    assert not {k for k in _unrecognized(_DSPARK) if k.startswith("dspark_")}


def test_a_plain_mtp_checkpoint_still_reads_as_mtp():
    detail = _detail({"num_nextn_predict_layers": 1}, "speculation.heads")
    assert "MTP is the checkpoint's own" in detail
