# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fast paths the GLM-5.2 dep target serves in each of its cells, how it picks the cell, and
what its plan does with the declines its deployments log.

Host-only: the plan is resolved and installed from the cell's ``TargetPaths`` exactly as the
shell does at construction. The engagement of each deployment is the per-rank decision set the
fast paths log there (the same on every rank): the paths that engage, and the (path, key,
reason) triples of the paths the cell serves that decline. None of the declines may be a miss:
under ``modeling_v2='require'`` a miss raises.
"""

from __future__ import annotations

import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import ModelingV2Mode
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__dep import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    dispatch,
)
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

D = dispatch.Decline
REQUIRE, AUTO = ModelingV2Mode.REQUIRE, ModelingV2Mode.AUTO

# The deployments, by (ranks, LM head tensor-parallel in attention DP, draft tokens per request),
# with their KV-cache dtype and MoE backend.
DEP16 = (16, False, 3)
DEP8 = (8, False, 3)
DEP8_LM_HEAD_TP = (8, True, 5)
DEPLOYMENTS = {
    DEP16: (QuantAlgo.FP8, "TRTLLM"),
    DEP8: (QuantAlgo.FP8, "CUTEDSL"),
    DEP8_LM_HEAD_TP: (QuantAlgo.NVFP4, "CUTEDSL"),
}
IDS = ["dep16", "dep8", "dep8-lm-head-tp"]

# The paths that engage at each deployment (device work on every rank).
ENGAGED = {
    DEP16: {
        "rope_kv_append",
        "indexer_layer_front",
        "draft_loop",
        "decode_prologue",
        "host_page_retention",
    },
    DEP8: {
        "rope_kv_append",
        "indexer_layer_front",
        "draft_loop",
        "decode_prologue",
        "host_page_retention",
    },
    DEP8_LM_HEAD_TP: {"draft_loop", "decode_prologue", "host_page_retention"},
}
# The bases the engaged paths extend; they decline every call at these deployments, or (the
# post-attention hop) are never built.
BASES = {
    DEP16: {
        "absorbed_query",
        "draft_absorbed_query",
        "indexer_topk",
        "cross_rank_topk",
        "post_attention_hop",
    },
    DEP8: {
        "absorbed_query",
        "draft_absorbed_query",
        "indexer_topk",
        "cross_rank_topk",
        "post_attention_hop",
    },
    DEP8_LM_HEAD_TP: {"post_attention_hop"},
}
SERVED = {cell: ENGAGED[cell] | BASES[cell] for cell in DEPLOYMENTS}
# Paths that log an engaged decision at these deployments without device work there: the
# step-input stager's feature check (its stager, store writer and execution-stream loop all
# decline attention DP) and the cross-layer expert prefetch links (the prefetch is issued only
# by the routed-MoE launch, which does not run with expert-parallel experts).
LOG_ONLY = {"deferred_input_prep", "sampler_store", "execution_stream_loop"} | {
    "cross_layer_expert_prefetch"
}

ADP = "attention DP / CP not supported"
ADP_TOPK = "attention data parallelism"

# The declines the served paths log with an FP8 KV cache, on every rank.
DECLINES_FP8_KV = (
    *(("absorbed_query", f"layer{i}", ADP) for i in range(78)),
    ("absorbed_query", "layer78", "target layers only"),
    *(("draft_absorbed_query", f"layer{i}", "draft (MTP) layers only") for i in range(78)),
    ("draft_absorbed_query", "layer78", ADP),
    *(("indexer_layer_front", f"target_layer{i}", ADP) for i in range(78)),
    ("indexer_layer_front", "draft_layer78", ADP),
    ("indexer_layer_front", "cp_indexer_decode_cross_rank_topk", ADP_TOPK),
    ("indexer_topk", "indexer_decode", ADP_TOPK),
    ("indexer_topk", "indexer_decode", "MTP draft step (target layers only)"),
    ("cross_rank_topk", "cp_indexer_decode", ADP_TOPK),
    ("draft_loop", "draft_loop:context rows present", "context rows present"),
    ("draft_loop", "draft_step:dense top-k fallback active", "dense top-k fallback active"),
    (
        "decode_prologue",
        "target_prologue[not a generation-only step]",
        "not a generation-only step",
    ),
)
# The declines the served paths log with an NVFP4 KV cache and the LM head tensor-parallel in
# attention DP, on every rank.
DECLINES_NVFP4_KV = (
    ("draft_loop", "draft_loop:LM-head TP in attention DP", "LM-head TP in attention DP"),
    ("draft_loop", "draft_loop:context rows present", "context rows present"),
    (
        "decode_prologue",
        "target_prologue[not a generation-only step]",
        "not a generation-only step",
    ),
)
DECLINES = {DEP16: DECLINES_FP8_KV, DEP8: DECLINES_FP8_KV, DEP8_LM_HEAD_TP: DECLINES_NVFP4_KV}
# The engaged decisions the served paths log, as (path, key).
ENGAGED_KEYS = {
    DEP16: {
        ("rope_kv_append", "rope"),
        ("rope_kv_append", "bridge"),
        ("indexer_layer_front", "rope"),
        ("draft_loop", "draft_embedding"),
        ("draft_loop", "draft_loop:engaged"),
        ("draft_loop", "draft_step:engaged"),
        ("decode_prologue", "target_prologue"),
        ("decode_prologue", "target_embedding"),
        ("host_page_retention", "host_policy"),
    },
    DEP8_LM_HEAD_TP: {
        ("draft_loop", "draft_embedding"),
        ("decode_prologue", "target_prologue"),
        ("decode_prologue", "target_embedding"),
        ("host_page_retention", "host_policy"),
    },
}
ENGAGED_KEYS[DEP8] = ENGAGED_KEYS[DEP16]


def _model_config(cell, spec=True, **overrides):
    """What ``target_paths`` reads off the model config of a deployment; ``overrides`` replace
    its ``kv_cache`` and ``moe_backend``."""
    ranks, lm_head_tp, draft = cell
    kv_cache, moe_backend = DEPLOYMENTS.get(cell, (QuantAlgo.FP8, "TRTLLM"))
    facts = {"kv_cache": kv_cache, "moe_backend": moe_backend, **overrides}
    return types.SimpleNamespace(
        mapping=types.SimpleNamespace(
            world_size=ranks, enable_attention_dp=True, enable_lm_head_tp_in_adp=lm_head_tp
        ),
        spec_config=types.SimpleNamespace(max_draft_len=draft) if spec else None,
        quant_config=types.SimpleNamespace(kv_cache_quant_algo=facts["kv_cache"]),
        moe_backend=facts["moe_backend"],
    )


@pytest.fixture
def installed(monkeypatch):
    """Resolve and install the plan of a deployment's cell, as the shell does."""
    for name in ("_INSTALLED", "_VALIDATE", "_require", "_TARGET"):
        monkeypatch.setattr(dispatch, name, list(getattr(dispatch, name)))
    monkeypatch.setattr(dispatch, "_logged", set())
    monkeypatch.setattr(dispatch, "_missed", {})
    monkeypatch.setattr(dispatch, "_CAPABILITY_LOGGED", {})
    monkeypatch.setattr(dispatch, "_log_info", lambda message: None)
    dispatch.classify_decline.cache_clear()

    def install(cell, environ=None, mode=REQUIRE):
        paths = modeling.target_paths(_model_config(cell), mode)
        plan = dispatch.resolve(mode, environ or {}, paths)
        dispatch.install(plan, environ or {})
        return plan

    yield install
    dispatch.classify_decline.cache_clear()


# --- the cells ------------------------------------------------------------------------------


def test_the_cells_are_the_three_deployments():
    assert set(modeling.CELLS) == set(DEPLOYMENTS)
    for cell, (kv_cache, moe_backend) in DEPLOYMENTS.items():
        assert (modeling.CELLS[cell].kv_cache, modeling.CELLS[cell].moe_backend) == (
            kv_cache,
            moe_backend,
        )


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
@pytest.mark.parametrize("mode", [REQUIRE, AUTO], ids=["require", "auto"])
def test_each_deployment_selects_its_cells_paths(cell, mode):
    assert modeling.target_paths(_model_config(cell), mode) is modeling.CELLS[cell].paths


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
def test_the_kv_cache_dtype_may_be_the_enum_or_its_value(cell):
    kv_cache = DEPLOYMENTS[cell][0]
    config = _model_config(cell, kv_cache=kv_cache.value)
    assert not isinstance(config.quant_config.kv_cache_quant_algo, QuantAlgo)
    assert modeling.target_paths(config, REQUIRE) is modeling.CELLS[cell].paths


# A deployment ``route`` claims whose KV-cache dtype or MoE backend is not its cell's, and one
# outside the cells: (cell, the facts that differ, what the error names).
_OUTSIDE = {
    "dep16-nvfp4-kv": (DEP16, {"kv_cache": QuantAlgo.NVFP4}, "KV cache NVFP4, MoE backend TRTLLM"),
    "dep16-cutedsl": (DEP16, {"moe_backend": "CUTEDSL"}, "KV cache FP8, MoE backend CUTEDSL"),
    "dep8-trtllm": (DEP8, {"moe_backend": "TRTLLM"}, "KV cache FP8, MoE backend TRTLLM"),
    "dep8-nvfp4-kv": (DEP8, {"kv_cache": QuantAlgo.NVFP4}, "KV cache NVFP4, MoE backend CUTEDSL"),
    "dep8-unquantized-kv": (DEP8, {"kv_cache": None}, "KV cache None, MoE backend CUTEDSL"),
    "dep8-lm-head-tp-fp8-kv": (
        DEP8_LM_HEAD_TP,
        {"kv_cache": QuantAlgo.FP8},
        "KV cache FP8, MoE backend CUTEDSL",
    ),
    "dep8-lm-head-tp-cutlass": (
        DEP8_LM_HEAD_TP,
        {"moe_backend": "CUTLASS"},
        "KV cache NVFP4, MoE backend CUTLASS",
    ),
    "dep16-mtp5": (
        (16, False, 5),
        {},
        "16 ranks, LM head tensor-parallel in attention DP False, 5",
    ),
}


@pytest.mark.parametrize("cell, overrides, named", list(_OUTSIDE.values()), ids=list(_OUTSIDE))
def test_a_deployment_outside_the_cells_raises_under_require(cell, overrides, named):
    config = _model_config(cell, **overrides)
    with pytest.raises(ValueError, match="is not a deployment this target serves") as excinfo:
        modeling.target_paths(config, REQUIRE)
    message = str(excinfo.value)
    assert named in message, message
    # The error names the three cells.
    assert message.count("draft tokens: KV cache") == 3, message


@pytest.mark.parametrize("cell, overrides, named", list(_OUTSIDE.values()), ids=list(_OUTSIDE))
def test_a_deployment_outside_the_cells_serves_nothing_under_auto(
    monkeypatch, cell, overrides, named
):
    warnings = []
    monkeypatch.setattr(modeling.logger, "warning", warnings.append)
    config = _model_config(cell, **overrides)
    assert modeling.target_paths(config, AUTO) is modeling.NO_PATHS
    assert modeling.NO_PATHS.served == frozenset()
    (warning,) = warnings
    assert named in warning and warning.endswith("no fast path runs"), warning


def test_without_speculation_no_cell_matches():
    config = _model_config(DEP8, spec=False)
    with pytest.raises(ValueError, match="None draft tokens"):
        modeling.target_paths(config, REQUIRE)


def test_the_auto_plan_outside_the_cells_runs_no_fast_path(installed, monkeypatch):
    monkeypatch.setattr(modeling.logger, "warning", lambda message: None)
    paths = modeling.target_paths(_model_config(DEP8, moe_backend="TRTLLM"), AUTO)
    plan = dispatch.resolve(AUTO, {}, paths)
    dispatch.install(plan, {})
    assert dispatch.installed() == frozenset()
    assert plan.describe().startswith("modeling_v2=auto, 0/0 fast paths")
    assert not dispatch.require_mode()


# --- the plans ------------------------------------------------------------------------------


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
def test_the_cell_serves_the_engaged_paths_and_their_bases(installed, cell):
    plan = installed(cell)
    assert set(plan.served) == SERVED[cell]
    assert dispatch.installed() == frozenset(SERVED[cell])
    count = len(SERVED[cell])
    assert plan.describe().startswith(f"modeling_v2=require, {count}/{count} fast paths")


def test_the_cells_serve_ten_ten_and_four_paths():
    assert [len(SERVED[cell]) for cell in DEPLOYMENTS] == [10, 10, 4]
    assert modeling.CELLS[DEP16].paths is modeling.CELLS[DEP8].paths is modeling.FP8_KV_PATHS
    assert modeling.CELLS[DEP8_LM_HEAD_TP].paths is modeling.NVFP4_KV_PATHS


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
def test_every_base_of_a_served_path_is_served(cell):
    served = SERVED[cell]
    for path in served:
        assert dispatch.REQUIRED_FEATURES.get(path, frozenset()) <= served, path
    # The bases are exactly what the engaged paths extend.
    required = set().union(*(dispatch.REQUIRED_FEATURES.get(p, frozenset()) for p in ENGAGED[cell]))
    assert required - ENGAGED[cell] == BASES[cell]


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
def test_no_log_only_path_is_served(cell):
    """The step-input stager's parts and the expert prefetch do no device work under attention
    DP; the execution-stream loop has no attention-DP gate of its own, so it must not be
    served."""
    assert not (SERVED[cell] & LOG_ONLY)


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
def test_every_kernel_package_is_the_familys(cell):
    assert dict(modeling.CELLS[cell].paths.kernel_modules) == {}


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
def test_the_record_line_lists_the_served_paths(installed, monkeypatch, cell):
    lines = []
    monkeypatch.setattr(dispatch, "_log_info", lines.append)
    installed(cell)
    (record,) = [line for line in lines if "record" in line]
    listed = record.split("paths=", 1)[1].split(" ", 1)[0].split(",")
    assert {name.split("[", 1)[0] for name in listed} == SERVED[cell]


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
def test_disable_may_name_only_a_served_path(installed, cell):
    with pytest.raises(ValueError, match="not fast paths of this target"):
        installed(cell, {dispatch.DISABLE_ENV: "execution_stream_loop"})
    plan = installed(cell, {dispatch.DISABLE_ENV: "decode_prologue"})
    assert not plan.enabled("decode_prologue") and plan.enabled("draft_loop")


# Each path of the failure drop order with what it turns off (itself and its dependents).
_DROP_ORDER = {
    "decode_prologue": {"decode_prologue"},
    "indexer_layer_front": {"indexer_layer_front"},
    "draft_loop": {"draft_loop", "decode_prologue"},
    "rope_kv_append": {"rope_kv_append", "indexer_layer_front"},
}


@pytest.mark.parametrize("path", list(_DROP_ORDER))
def test_each_path_of_the_drop_order_can_be_disabled(installed, path):
    plan = installed(DEP8, {dispatch.DISABLE_ENV: path})
    off = {name for name in SERVED[DEP8] if not plan.enabled(name)}
    assert off == _DROP_ORDER[path]
    assert plan.describe().startswith(f"modeling_v2=require, {10 - len(off)}/10 fast paths")


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
def test_the_routed_accumulation_knob_is_refused(installed, cell):
    with pytest.raises(ValueError, match="does not serve routed_moe_into_hop"):
        installed(cell, {dispatch.ACCUMULATION_ENV: "identity"})


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
def test_the_cross_rank_wait_timeout_is_read(installed, cell):
    plan = installed(cell, {dispatch.CROSS_RANK_TOPK_TIMEOUT_ENV: "60000"})
    assert plan.knobs == ((dispatch.CROSS_RANK_TOPK_TIMEOUT_ENV, "60000"),)


# --- the declines ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
def test_no_logged_decline_is_a_miss(installed, cell):
    """Under require, a decline decided once inside a path's cells would raise."""
    installed(cell)
    assert dispatch.require_mode()
    for path, key, reason in DECLINES[cell]:
        assert path in SERVED[cell], path
        assert dispatch.classify_decline(path, key, reason) is D.UNCOVERED, (path, key, reason)
        assert dispatch.log_capability(path, key, False, reason) is False
    assert dispatch.missed_counts() == {}


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
def test_the_engaged_decisions_are_of_served_paths(installed, cell):
    installed(cell)
    for path, key in ENGAGED_KEYS[cell]:
        assert path in ENGAGED[cell], path
        assert dispatch.log_capability(path, key, True) is True
    assert {path for path, _ in ENGAGED_KEYS[cell]} == ENGAGED[cell]


@pytest.mark.parametrize("cell", list(DEPLOYMENTS), ids=IDS)
def test_every_base_declines_every_call(cell):
    """A base the cell serves logs no engaged decision; every one but the never-built
    post-attention hop declines."""
    declining = {path for path, _, _ in DECLINES[cell]}
    assert BASES[cell] - {"post_attention_hop"} <= declining
    assert not ({path for path, _ in ENGAGED_KEYS[cell]} & BASES[cell])


def test_the_family_cells_would_raise_on_the_query_fronts(installed):
    """The family's cells hold every target layer for the query fronts; the dep table is what
    keeps an attention-DP deployment from raising on their declines under require."""
    plan = dispatch.resolve(REQUIRE, {}, dispatch.TargetPaths(served=SERVED[DEP8]))
    dispatch.install(plan, {})
    misses = {
        (path, key)
        for path, key, reason in DECLINES_FP8_KV
        if dispatch.classify_decline(path, key, reason) is D.MISSED
    }
    assert ("absorbed_query", "layer0") in misses
    assert ("draft_absorbed_query", "layer78") in misses
    assert ("indexer_layer_front", "target_layer0") in misses
