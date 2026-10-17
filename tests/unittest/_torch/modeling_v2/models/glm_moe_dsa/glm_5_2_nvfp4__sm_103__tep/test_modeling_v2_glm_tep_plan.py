# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fast paths the GLM-5.2 tep target serves at each tensor-parallel size, and what its
plan does with the declines its deployments log.

Host-only: the plan is resolved and installed from the target's ``TARGET_PATHS`` exactly as
the shell does at construction; the declines are the (path, key, reason) triples the kernels
log at those deployments, decided at construction or in the warm-up, and none of them may be
a miss: under ``modeling_v2='require'`` a miss raises.
"""

from __future__ import annotations

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import ModelingV2Mode
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8 import (  # noqa: E501
    modeling as tp8_modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    dispatch,
)

pytestmark = pytest.mark.cpu_only

D = dispatch.Decline
REQUIRE = ModelingV2Mode.REQUIRE

# The paths whose kernels take only tensor-parallel experts (every expert on every rank).
ROUTED_MOE = {
    "routed_moe",
    "routed_moe_into_hop",
    "hop_routing",
    "shared_expert_prefetch",
    "draft_routed_moe",
    "cross_layer_expert_prefetch",
}
# At four ranks: the paths that engage there, and the bases they extend.
TEP4_ENGAGED = {
    "indexer_layer_front",
    "rope_kv_append",
    "indexer_topk",
    "cross_rank_topk",
    "cross_rank_topk_cluster_merge",
    "cross_rank_topk_global_emit",
    "router_gemm",
    "dense_mlp",
    "draft_front",
    "draft_tail",
    "draft_loop",
    "decode_prologue",
    "deferred_input_prep",
    "sampler_store",
    "execution_stream_loop",
    "host_page_retention",
}
TEP4_BASES = {"absorbed_query", "draft_absorbed_query", "post_attention_hop"}
SERVED = {
    4: TEP4_ENGAGED | TEP4_BASES,
    8: set(dispatch.FAST_PATH_NAMES) - ROUTED_MOE,
}


def _layers(spec: str) -> list:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


FULL = "0-2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74"
SHARED = (
    "3-5,7-9,11-13,15-17,19-21,23-25,27-29,31-33,35-37,39-41,43-45,47-49,51-53,55-57,59-61,"
    "63-65,67-69,71-73,75-77"
)
TP8_DOMAIN = "shape outside the GLM-5.2 TP8 domain"

# The declines the served paths log at four ranks (bs 2 and bs 5), on every rank.
DECLINES_TP4 = (
    *(("absorbed_query", f"layer{i}", TP8_DOMAIN) for i in range(78)),
    ("absorbed_query", "layer78", "target layers only"),
    *(("draft_absorbed_query", f"layer{i}", "draft (MTP) layers only") for i in range(78)),
    ("draft_absorbed_query", "layer78", TP8_DOMAIN),
    *(("indexer_layer_front", f"target_layer{i}", TP8_DOMAIN) for i in range(78)),
    ("indexer_layer_front", "draft_layer78", TP8_DOMAIN),
    (
        "post_attention_hop",
        "post_attn_hop:native all-reduce (AUTO) does not use MNNVL",
        "native all-reduce (AUTO) does not use MNNVL",
    ),
    ("router_gemm", "router:not a target layer", "not a target layer"),
    ("indexer_topk", "indexer_decode", "MTP draft step (target layers only)"),
    ("indexer_topk", "indexer_decode", "5 generation requests > 4"),
    ("cross_rank_topk", "cp_indexer_decode", "2 generation requests != 1"),
    ("cross_rank_topk_cluster_merge", "cp_indexer_decode", "4 generation requests != 1"),
    ("cross_rank_topk_global_emit", "cp_indexer_decode", "5 generation requests > 4"),
    ("dense_mlp", "dense_mlp:row count outside [1, 8]", "row count outside [1, 8]"),
    ("draft_loop", "draft_loop:context rows present", "context rows present"),
    ("draft_loop", "draft_step:dense top-k fallback active", "dense top-k fallback active"),
    (
        "deferred_input_prep",
        "prep[not a steady decode-graph replay]",
        "not a steady decode-graph replay",
    ),
    (
        "deferred_input_prep",
        "prep[no previous speculative device tensors]",
        "no previous speculative device tensors",
    ),
    (
        "deferred_input_prep",
        "prep[dummy row or row without previous tensors]",
        "dummy row or row without previous tensors",
    ),
    (
        "decode_prologue",
        "target_prologue[not a generation-only step]",
        "not a generation-only step",
    ),
)

# The declines the served paths log at eight ranks (bs 5), on every rank.
DECLINES_TP8 = (
    ("absorbed_query", "layer78", "target layers only"),
    ("attention_output_tail", "layer78", "draft or unindexed layer"),
    *(
        (
            "shared_indexer_absorbed_query",
            f"layer{i}",
            "full indexer layer (shared_indexer_absorbed_query serves shared layers)",
        )
        for i in _layers(FULL)
    ),
    ("shared_indexer_absorbed_query", "layer78", "target layers only"),
    *(("draft_absorbed_query", f"layer{i}", "draft (MTP) layers only") for i in range(78)),
    *(
        (
            "deferred_norm",
            f"scope[shared]_into_layer{i}",
            "TRTLLM_STAIRCASE_GLM_LL_DEFERRED_NORM_SCOPE=shared: layer "
            f"{i} is a full-indexer layer; post_moe_hop hop",
        )
        for i in _layers(FULL)[3:]
    ),
    *(
        (
            "indexer_layer_front",
            f"target_layer{i}",
            "shared-indexer layer (no q_a RMSNorm launch to remove; "
            "shared_indexer_absorbed_query / absorbed_query serve it)",
        )
        for i in _layers(SHARED)
    ),
    *(
        (
            "fused_kv_indexer_projection",
            f"layer{i}",
            "full layers not selected (TRTLLM_STAIRCASE_GLM_LL_FUSED_KV_INDEXER_PROJECTION_LAYERS)",
        )
        for i in _layers(FULL)
    ),
    (
        "fused_kv_indexer_projection",
        "layer78",
        "draft layers not selected (TRTLLM_STAIRCASE_GLM_LL_FUSED_KV_INDEXER_PROJECTION_LAYERS)",
    ),
    ("router_gemm", "router:not a target layer", "not a target layer"),
    (
        "sparse_mla_decode",
        "request:not generation-only",
        "not generation-only; first seen at layer 3 with 1 rows",
    ),
    ("indexer_topk", "indexer_decode", "MTP draft step (target layers only)"),
    ("indexer_topk", "indexer_decode", "5 generation requests > 4"),
    ("cross_rank_topk", "cp_indexer_decode", "2 generation requests != 1"),
    ("cross_rank_topk_global_emit", "cp_indexer_decode", "5 generation requests > 4"),
    ("dense_mlp", "dense_mlp:row count outside [1, 8]", "row count outside [1, 8]"),
    ("draft_loop", "draft_loop:context rows present", "context rows present"),
    (
        "deferred_input_prep",
        "prep[not a steady decode-graph replay]",
        "not a steady decode-graph replay",
    ),
    (
        "decode_prologue",
        "target_prologue[not a generation-only step]",
        "not a generation-only step",
    ),
)
DECLINES = {4: DECLINES_TP4, 8: DECLINES_TP8}

# The layer-keyed capabilities that engage at eight ranks: (path, key family, layers).
ENGAGED_TP8 = (
    ("absorbed_query", "layer", "0-77"),
    ("attention_output_tail", "layer", "0-77"),
    ("oproj_weight_staging", "layer", "0-77"),
    ("shared_indexer_absorbed_query", "layer", SHARED),
    ("draft_absorbed_query", "layer", "78"),
    ("sparse_mla_decode", "layer", "0-78"),
    ("sparse_mla_decode", "call:layer", "0-78"),
    ("fused_kv_indexer_projection", "layer", "3"),
    ("indexer_layer_front", "target_layer", FULL),
    ("indexer_layer_front", "draft_layer", "78"),
    ("deferred_norm", "scope[shared]_into_layer", SHARED.replace("3-5", "4-5")),
    ("deferred_norm_evict_hints", "projection_layer", SHARED.replace("3-5", "4-5")),
)


@pytest.fixture
def installed(monkeypatch):
    """Resolve and install the target's plan at a tensor-parallel size, as the shell does."""
    for name in ("_INSTALLED", "_VALIDATE", "_require", "_TARGET"):
        monkeypatch.setattr(dispatch, name, list(getattr(dispatch, name)))
    monkeypatch.setattr(dispatch, "_logged", set())
    monkeypatch.setattr(dispatch, "_missed", {})
    monkeypatch.setattr(dispatch, "_CAPABILITY_LOGGED", {})
    monkeypatch.setattr(dispatch, "_log_info", lambda message: None)
    dispatch.classify_decline.cache_clear()

    def install(tp_size, environ=None):
        plan = dispatch.resolve(REQUIRE, environ or {}, modeling.TARGET_PATHS[tp_size])
        dispatch.install(plan, environ or {})
        return plan

    yield install
    dispatch.classify_decline.cache_clear()


def test_the_target_has_a_table_per_tensor_parallel_size():
    assert sorted(modeling.TARGET_PATHS) == [4, 8]


@pytest.mark.parametrize("tp_size", [4, 8])
def test_the_target_serves_its_paths_at_each_size(installed, tp_size):
    plan = installed(tp_size)
    assert set(plan.served) == SERVED[tp_size]
    assert dispatch.installed() == frozenset(SERVED[tp_size])
    assert plan.describe().startswith(
        f"modeling_v2=require, {len(SERVED[tp_size])}/{len(SERVED[tp_size])} fast paths"
    )


def test_the_sizes_serve_nineteen_and_thirty_paths():
    assert len(SERVED[4]) == 19 and len(SERVED[8]) == 30


@pytest.mark.parametrize("tp_size", [4, 8])
def test_no_served_path_extends_one_the_target_does_not_serve(tp_size):
    served = SERVED[tp_size]
    for path in served:
        assert dispatch.REQUIRED_FEATURES.get(path, frozenset()) <= served, path


@pytest.mark.parametrize("tp_size", [4, 8])
def test_the_target_never_serves_a_routed_moe_path(tp_size):
    assert not (SERVED[tp_size] & ROUTED_MOE)


@pytest.mark.parametrize("tp_size", [4, 8])
def test_every_kernel_package_is_the_familys(tp_size):
    """The target has no kernel package of its own: every served path imports from shared."""
    assert dict(modeling.TARGET_PATHS[tp_size].kernel_modules) == {}


@pytest.mark.parametrize("tp_size", [4, 8])
def test_the_record_line_lists_the_served_paths(installed, monkeypatch, tp_size):
    lines = []
    monkeypatch.setattr(dispatch, "_log_info", lines.append)
    installed(tp_size)
    (record,) = [line for line in lines if "record" in line]
    listed = record.split("paths=", 1)[1].split(" ", 1)[0].split(",")
    assert {name.split("[", 1)[0] for name in listed} == SERVED[tp_size]


@pytest.mark.parametrize("tp_size", [4, 8])
def test_disable_may_name_only_a_served_path(installed, tp_size):
    with pytest.raises(ValueError, match="not fast paths of this target"):
        installed(tp_size, {dispatch.DISABLE_ENV: "routed_moe"})
    plan = installed(tp_size, {dispatch.DISABLE_ENV: "draft_tail"})
    assert not plan.enabled("draft_tail") and plan.enabled("draft_loop")


def test_disable_turns_the_dependents_off_at_four_ranks(installed):
    plan = installed(4, {dispatch.DISABLE_ENV: "absorbed_query"})
    assert not plan.enabled("indexer_layer_front")
    assert plan.describe().startswith("modeling_v2=require, 17/19 fast paths")


@pytest.mark.parametrize("tp_size", [4, 8])
def test_the_routed_accumulation_knob_is_refused(installed, tp_size):
    with pytest.raises(ValueError, match="does not serve routed_moe_into_hop"):
        installed(tp_size, {dispatch.ACCUMULATION_ENV: "identity"})


@pytest.mark.parametrize("tp_size", [4, 8])
def test_the_cross_rank_wait_timeout_is_read(installed, tp_size):
    plan = installed(tp_size, {dispatch.CROSS_RANK_TOPK_TIMEOUT_ENV: "60000"})
    assert plan.knobs == ((dispatch.CROSS_RANK_TOPK_TIMEOUT_ENV, "60000"),)


@pytest.mark.parametrize("tp_size", [4, 8])
def test_no_logged_decline_is_a_miss(installed, tp_size):
    """Under require, a decline decided once inside a path's cells would raise."""
    installed(tp_size)
    assert dispatch.require_mode()
    for path, key, reason in DECLINES[tp_size]:
        assert path in SERVED[tp_size], path
        assert dispatch.classify_decline(path, key, reason) is not D.MISSED, (path, key, reason)
        assert dispatch.log_capability(path, key, False, reason) is False
    assert all(count >= 0 for count in dispatch.missed_counts().values())


def test_at_four_ranks_the_bases_decline_as_uncovered(installed):
    installed(4)
    for path, key, reason in DECLINES_TP4:
        if path in TEP4_BASES or key.startswith(("target_layer", "draft_layer")):
            assert dispatch.classify_decline(path, key, reason) is D.UNCOVERED, (path, key)


def test_the_four_rank_tables_would_raise_at_the_tp8_cells(installed):
    """The tp8 cells hold the query fronts and the hop that four ranks decline on: the tep
    table is what keeps a four-rank deployment from raising under require."""
    installed(8)
    misses = [
        (path, key)
        for path, key, reason in DECLINES_TP4
        if dispatch.classify_decline(path, key, reason) is D.MISSED
    ]
    assert ("absorbed_query", "layer0") in misses
    assert ("post_attention_hop", "post_attn_hop:native all-reduce (AUTO) does not use MNNVL") in (
        misses
    )


def test_at_eight_ranks_every_engaged_layer_is_inside_the_cells(installed):
    target = modeling.TARGET_PATHS[8]
    for path, family, layers in ENGAGED_TP8:
        cells = target.layer_cells[path][family]
        assert set(_layers(layers)) <= cells, (path, family)


def test_the_tp8_target_serves_every_path_with_the_family_tables():
    target = tp8_modeling.TARGET_PATHS
    assert target.served == frozenset(dispatch.FAST_PATH_NAMES)
    assert target.layer_cells is dispatch.LAYER_CELLS
    assert dict(target.uncovered) == {}
    assert {path for path in target.kernel_modules} == ROUTED_MOE
    assert all(
        module.startswith(tp8_modeling.__package__ + ".kernels.")
        for module in target.kernel_modules.values()
    )
