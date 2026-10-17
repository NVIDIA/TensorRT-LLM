# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which fast paths of the GLM-5.2 tp8 target run: the installed plan, the knobs the target
refuses, the kernel registry, the capability log and the knobs folded into constants.

Host-only: the knobs are set per test, the log goes into a list and the kernel imports are
recorded, so nothing here needs a GPU or a kernel package.
"""

from __future__ import annotations

import json
import os
import re
import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    MODELING_V2_VALIDATE_ENV,
    ModelingV2Mode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    dispatch,
)

pytestmark = pytest.mark.cpu_only

AUTO = ModelingV2Mode.AUTO
ACCUMULATION = dispatch.KNOB_PREFIX + "ROUTED_MOE_INTO_HOP_ACCUMULATION"
SCOPE = dispatch.KNOB_PREFIX + "DEFERRED_NORM_SCOPE"
PARTS = ("sampler_store", "execution_stream_loop")

# The record of every fast path but the parts that only deferred_input_prep brings along.
SERVED_PATHS = (
    "paths=absorbed_query,attention_output_into_hop,attention_output_tail,"
    "attention_output_tail_prefetch,cross_layer_expert_prefetch,cross_rank_topk[rank-consistent],"
    "cross_rank_topk_cluster_merge[rank-consistent],cross_rank_topk_global_emit[rank-consistent],"
    "decode_prologue,deferred_input_prep,deferred_norm[shared],deferred_norm_evict_hints,dense_mlp,"
    "draft_absorbed_query,draft_front,draft_live_rows,draft_loop,draft_routed_moe,draft_tail,"
    "fused_kv_indexer_projection[kw-fused-a-order],hop_routing,host_page_retention,"
    "indexer_layer_front,indexer_topk,oproj_weight_staging,post_attention_hop,post_moe_hop,"
    "rope_kv_append,routed_moe,routed_moe_into_hop[fp32],router_gemm,shared_expert_prefetch,"
    "shared_indexer_absorbed_query,sparse_mla_decode"
)

# The paths each path has no effect without: the closure of ``depends_on``.
REQUIRED = {
    "shared_indexer_absorbed_query": {"absorbed_query"},
    "attention_output_tail_prefetch": {"attention_output_tail"},
    "attention_output_into_hop": {"attention_output_tail", "post_attention_hop"},
    "routed_moe_into_hop": {"post_moe_hop", "routed_moe"},
    "deferred_norm": {"absorbed_query", "post_moe_hop"},
    "cross_layer_expert_prefetch": {"routed_moe"},
    "hop_routing": {
        "attention_output_tail",
        "post_attention_hop",
        "attention_output_into_hop",
        "routed_moe",
    },
    "oproj_weight_staging": {
        "attention_output_tail",
        "post_attention_hop",
        "sparse_mla_decode",
        "attention_output_into_hop",
    },
    "fused_kv_indexer_projection": {"absorbed_query"},
    "host_page_retention": {"post_attention_hop"},
    "shared_expert_prefetch": {
        "attention_output_tail",
        "post_moe_hop",
        "routed_moe",
        "post_attention_hop",
        "attention_output_into_hop",
        "routed_moe_into_hop",
        "hop_routing",
    },
    "cross_rank_topk_cluster_merge": {"cross_rank_topk"},
    "deferred_norm_evict_hints": {"absorbed_query", "post_moe_hop", "deferred_norm"},
    "cross_rank_topk_global_emit": {
        "rope_kv_append",
        "cross_rank_topk",
        "cross_rank_topk_cluster_merge",
    },
    "decode_prologue": {"draft_loop"},
    "draft_live_rows": {"post_attention_hop", "draft_loop"},
    "indexer_layer_front": {
        "absorbed_query",
        "indexer_topk",
        "draft_absorbed_query",
        "rope_kv_append",
        "cross_rank_topk",
    },
}

RECORD_PREFIX = "[glm_moe_dsa] dispatch: record "
_RECORD = re.compile(r"^paths=(\S+) side_selectors=(\{.*\}) shared_env=(\{.*\})$")


@pytest.fixture
def lines(monkeypatch):
    """A fresh selector state that logs into the returned list; no knob of the target is set."""
    for name in list(os.environ):
        if name.startswith(dispatch.SIDE_SELECTOR_PREFIXES) or name in dispatch.SHARED_ENV:
            monkeypatch.delenv(name)
    logged = []
    monkeypatch.setattr(dispatch, "_log_info", logged.append)
    monkeypatch.delenv(MODELING_V2_VALIDATE_ENV, raising=False)
    monkeypatch.setattr(dispatch, "_INSTALLED", [None])
    monkeypatch.setattr(dispatch, "_VALIDATE", [False])
    monkeypatch.setattr(dispatch, "_require", [False])
    monkeypatch.setattr(dispatch, "_REGISTRY", {})
    monkeypatch.setattr(dispatch, "_CAPABILITY_LOGGED", {})
    return logged


def _parse(record):
    m = _RECORD.match(record)
    assert m, record
    return m.group(1), json.loads(m.group(2)), json.loads(m.group(3))


def _install(environ):
    """Resolve and install the plan of ``environ``; return the paths it runs."""
    dispatch.install(dispatch.resolve(AUTO, environ), environ)
    return dispatch.installed()


# --- the vocabulary -----------------------------------------------------------------------


def test_the_selector_takes_every_fast_path_but_the_parts():
    assert len(dispatch.FAST_PATH_NAMES) == 36
    assert dispatch.SELECTOR_PATHS == tuple(n for n in dispatch.FAST_PATH_NAMES if n not in PARTS)
    assert len(dispatch.SELECTOR_PATHS) == 34
    assert dict(dispatch.PART_PATHS) == {
        "prep": "deferred_input_prep",
        "store": "sampler_store",
        "stream": "execution_stream_loop",
    }
    assert dispatch.PARTS_ENV == "TRTLLM_STAIRCASE_GLM_LL_DEFERRED_INPUT_PREP_PARTS"


def test_the_prefetch_schedule_extends_the_attention_tail():
    names = list(dispatch.FAST_PATH_NAMES)
    assert names.index("attention_output_tail_prefetch") == names.index("attention_output_tail") + 1
    prefetch = dispatch.FAST_PATHS[names.index("attention_output_tail_prefetch")]
    assert prefetch.depends_on == ("attention_output_tail",)


def test_every_fast_path_has_its_kernel_directory():
    directories = set(dispatch.FAST_PATH_NAMES) - set(dispatch.KERNEL_DIRECTORY)
    assert len(directories) == 28
    assert len(dispatch.KERNEL_DIRECTORY) == 8
    assert set(dispatch.KERNEL_DIRECTORY) <= set(dispatch.FAST_PATH_NAMES)
    assert set(dispatch.KERNEL_DIRECTORY.values()) <= directories
    assert dispatch.KERNELS_PACKAGE == dispatch.__name__.rpartition(".")[0] + ".kernels"


def test_required_features_are_the_closure_of_the_table():
    assert {k: set(v) for k, v in dispatch.REQUIRED_FEATURES.items()} == REQUIRED


def test_the_variant_table_names_fast_paths_and_knobs():
    for path, (name, choices) in dispatch.NUMERICAL_VARIANTS.items():
        assert path in dispatch.FAST_PATH_NAMES
        assert choices and len(set(choices)) == len(choices)
        assert all(re.fullmatch(r"[a-z0-9][a-z0-9-]*", c) for c in choices)
        assert name is None or name.startswith(dispatch.KNOB_PREFIX)
    assert dispatch.NUMERICAL_VARIANTS["routed_moe_into_hop"] == (
        ACCUMULATION,
        ("fp32", "identity"),
    )
    # The deferred-norm scope is fixed in this target: no knob selects a variant.
    assert dispatch.NUMERICAL_VARIANTS["deferred_norm"] == (None, ("shared",))


# --- a path and its bases -----------------------------------------------------------------


@pytest.mark.parametrize("path", dispatch.FAST_PATH_NAMES)
def test_every_path_runs_on_its_minimal_base(lines, path):
    """Turning off every path outside a path and its bases leaves exactly those running."""
    kept = {path} | set(dispatch.REQUIRED_FEATURES.get(path, ()))
    off = ",".join(sorted(set(dispatch.FAST_PATH_NAMES) - kept))
    assert _install({dispatch.DISABLE_ENV: off} if off else {}) == kept


# --- variants and the record --------------------------------------------------------------


def test_the_served_record_names_every_variant(lines):
    paths = frozenset(dispatch.SELECTOR_PATHS)
    assert (
        dispatch.feature_record(paths, {}) == f"{SERVED_PATHS} side_selectors={{}} shared_env={{}}"
    )
    assert dispatch.feature_variants(paths, {}) == {
        "cross_rank_topk": "rank-consistent",
        "cross_rank_topk_cluster_merge": "rank-consistent",
        "cross_rank_topk_global_emit": "rank-consistent",
        "deferred_norm": "shared",
        "fused_kv_indexer_projection": "kw-fused-a-order",
        "routed_moe_into_hop": "fp32",
    }


@pytest.mark.parametrize("mode", ["fp32", "identity", " Identity "])
def test_the_accumulation_variant_follows_its_knob(mode):
    paths = {"post_moe_hop", "routed_moe", "routed_moe_into_hop"}
    env = {ACCUMULATION: mode}
    listed, side, _ = _parse(dispatch.feature_record(paths, env))
    assert listed == f"post_moe_hop,routed_moe,routed_moe_into_hop[{mode.strip().lower()}]"
    assert side == env


@pytest.mark.parametrize("scope", ["all", "shared", " Shared ", ""])
def test_the_deferred_norm_variant_is_the_fixed_scope(scope):
    """The record names the scope the hop runs; ``resolve`` refuses the scope variable."""
    paths = {"absorbed_query", "post_moe_hop", "deferred_norm"}
    env = {SCOPE: scope}
    listed, side, _ = _parse(dispatch.feature_record(paths, env))
    assert listed == "absorbed_query,deferred_norm[shared],post_moe_hop"
    assert side == env


@pytest.mark.parametrize(
    "raw, env",
    [
        ("post_moe_hop,routed_moe,routed_moe_into_hop", {ACCUMULATION: "fp16"}),
        # The finalized-row input stays a mode of the post-MoE hop, but no knob selects it.
        ("post_moe_hop,routed_moe,routed_moe_into_hop", {ACCUMULATION: "row"}),
    ],
)
def test_an_unknown_variant_of_a_selected_path_fails(raw, env):
    with pytest.raises(ValueError, match="is not a .* variant"):
        dispatch.feature_record(frozenset(raw.split(",")), env)


def test_the_variant_knob_of_an_unselected_path_is_only_recorded():
    env = {ACCUMULATION: "fp16", SCOPE: "x"}
    listed, side, _ = _parse(dispatch.feature_record({"absorbed_query"}, env))
    assert listed == "absorbed_query"
    assert side == env


def test_side_selectors_and_shared_env():
    env = {
        dispatch.DISABLE_ENV: "draft_tail",
        dispatch.KNOB_PREFIX + "CROSS_RANK_TOPK_TIMEOUT_MS": "100",
        ACCUMULATION: "identity",
        "TRTLLM_ENABLE_PDL": "1",
        "TRTLLM_FORCE_MNNVL_AR": "0",
        "PATH": "/usr/bin",
    }
    assert dispatch.side_selectors(env) == {
        dispatch.DISABLE_ENV: "draft_tail",
        dispatch.KNOB_PREFIX + "CROSS_RANK_TOPK_TIMEOUT_MS": "100",
        ACCUMULATION: "identity",
    }
    assert dispatch.shared_env(env) == {"TRTLLM_ENABLE_PDL": "1", "TRTLLM_FORCE_MNNVL_AR": "0"}
    assert _parse(dispatch.feature_record({"absorbed_query"}, env)) == (
        "absorbed_query",
        dispatch.side_selectors(env),
        dispatch.shared_env(env),
    )


# --- without a plan ---------------------------------------------------------------------


def test_without_a_plan_no_fast_path_runs_and_nothing_is_logged(lines, monkeypatch):
    """A process that does not build the target runs no fast path, whatever its environment."""
    monkeypatch.setenv(dispatch.FEATURE_ENV, ",".join(dispatch.SELECTOR_PATHS))
    monkeypatch.setenv(ACCUMULATION, "identity")
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    assert dispatch.installed() is None
    assert dispatch.enabled_features() == frozenset()
    assert not any(dispatch.is_enabled(path) for path in dispatch.FAST_PATH_NAMES)
    assert not dispatch.validate_mode()
    assert lines == []


def test_asking_for_something_that_is_not_a_fast_path_fails(lines):
    with pytest.raises(ValueError, match="'draft_tails' is not a fast path"):
        dispatch.is_enabled("draft_tails")
    assert lines == []


# --- the installed plan -------------------------------------------------------------------


def test_an_installed_plan_answers_every_query(lines):
    plan = dispatch.resolve(AUTO, {})
    lines.clear()
    assert dispatch.installed() is None
    dispatch.install(plan, {})
    assert dispatch.installed() == frozenset(dispatch.FAST_PATH_NAMES)
    assert all(dispatch.is_enabled(path) for path in dispatch.FAST_PATH_NAMES)
    assert len(lines) == 1
    listed, side, shared = _parse(lines[0].removeprefix(RECORD_PREFIX))
    served = SERVED_PATHS.removeprefix("paths=").split(",")
    assert listed.split(",") == sorted(served + list(PARTS))
    assert (side, shared) == ({}, {})


@pytest.mark.parametrize(
    "part, other_parts",
    [
        ("deferred_input_prep", {"sampler_store", "execution_stream_loop"}),
        ("sampler_store", {"deferred_input_prep", "execution_stream_loop"}),
        ("execution_stream_loop", {"deferred_input_prep", "sampler_store"}),
    ],
)
def test_each_part_of_the_input_preparation_is_a_path_of_its_own(lines, part, other_parts):
    paths = _install({dispatch.DISABLE_ENV: part})
    assert paths == frozenset(dispatch.FAST_PATH_NAMES) - {part}
    assert other_parts <= paths


@pytest.mark.parametrize("raw, validate", [("", False), ("0", False), ("1", True), (" On ", True)])
def test_validate_follows_the_modeling_v2_switch(lines, monkeypatch, raw, validate):
    monkeypatch.setenv(MODELING_V2_VALIDATE_ENV, raw)
    plan = dispatch.resolve(AUTO, {})
    assert plan.validate is validate
    assert not dispatch.validate_mode()
    dispatch.install(plan, {})
    assert dispatch.validate_mode() is validate
    dispatch.reset_fallback_state()
    assert not dispatch.validate_mode()


def test_disable_turns_off_a_path_and_its_dependents(lines):
    dispatch.install(dispatch.resolve(AUTO, {dispatch.DISABLE_ENV: "post_attention_hop"}), {})
    off = {path for path in dispatch.FAST_PATH_NAMES if not dispatch.is_enabled(path)}
    assert off == {
        "post_attention_hop",
        "attention_output_into_hop",
        "oproj_weight_staging",
        "hop_routing",
        "shared_expert_prefetch",
        "draft_live_rows",
        "host_page_retention",
    }
    listed, _, _ = _parse(lines[-1].removeprefix(RECORD_PREFIX))
    assert set(listed.split(",")).isdisjoint(off)


def test_a_bad_variant_of_an_enabled_path_fails_at_install(lines):
    plan = dispatch.resolve(AUTO, {})
    with pytest.raises(ValueError, match="is not a routed_moe_into_hop variant"):
        dispatch.install(plan, {ACCUMULATION: "fp16"})
    assert dispatch.installed() is None


def test_the_variant_knob_of_a_disabled_path_is_only_recorded(lines):
    env = {dispatch.DISABLE_ENV: "routed_moe_into_hop", ACCUMULATION: "fp16"}
    dispatch.install(dispatch.resolve(AUTO, env), env)
    assert not dispatch.is_enabled("routed_moe_into_hop")


@pytest.mark.parametrize(
    "knob", sorted(dispatch.REMOVED_KNOBS), ids=lambda k: k.removeprefix(dispatch.KNOB_PREFIX)
)
@pytest.mark.parametrize("value", ["deferred_input_prep", ""], ids=["set", "empty"])
def test_a_removed_knob_is_refused_with_what_to_use_instead(lines, knob, value):
    with pytest.raises(ValueError) as excinfo:
        dispatch.resolve(AUTO, {knob: value})
    assert str(excinfo.value) == (
        f"this target does not read {knob} ({dispatch.REMOVED_KNOBS[knob]}); unset them"
    )
    assert dispatch.installed() is None
    assert lines == []


def test_every_removed_knob_set_is_named_in_one_refusal(lines):
    env = {knob: "1" for knob in dispatch.REMOVED_KNOBS}
    env[dispatch.DISABLE_ENV] = "draft_tail"
    with pytest.raises(ValueError) as excinfo:
        dispatch.resolve(AUTO, env)
    message = str(excinfo.value)
    assert all(f"{knob} (" in message for knob in dispatch.REMOVED_KNOBS)
    assert f"{dispatch.DISABLE_ENV} (" not in message


def test_the_removed_knobs_point_at_what_replaces_them():
    hints = dispatch.REMOVED_KNOBS
    assert set(hints) == {
        dispatch.FEATURE_ENV,
        dispatch.PARTS_ENV,
        dispatch.KNOB_PREFIX + "DEFERRED_INPUT_PREP_DEBUG",
        dispatch.KNOB_PREFIX + "DRAFT_LOOP_VERIFY",
        dispatch.KNOB_PREFIX + "CROSS_LAYER_EXPERT_PREFETCH_STATS",
        dispatch.KNOB_PREFIX + "CROSS_LAYER_EXPERT_PREFETCH_REPORT_EVERY",
        dispatch.KNOB_PREFIX + "SHARED_INDEXER_ABSORBED_QUERY_FORCE_NATIVE",
    }
    assert "DISABLE" in hints[dispatch.KNOB_PREFIX + "SHARED_INDEXER_ABSORBED_QUERY_FORCE_NATIVE"]
    assert dispatch.DISABLE_ENV in hints[dispatch.FEATURE_ENV]
    assert dispatch.DISABLE_ENV in hints[dispatch.PARTS_ENV]
    for part, path in dispatch.PART_PATHS.items():
        assert f"{part} = {path}" in hints[dispatch.PARTS_ENV]
    for knob in ("DEFERRED_INPUT_PREP_DEBUG", "DRAFT_LOOP_VERIFY"):
        assert MODELING_V2_VALIDATE_ENV in hints[dispatch.KNOB_PREFIX + knob]


def test_resetting_the_fallback_state_uninstalls_the_plan(lines):
    dispatch.install(dispatch.resolve(AUTO, {dispatch.DISABLE_ENV: "draft_tail"}), {})
    assert not dispatch.is_enabled("draft_tail")
    dispatch.reset_fallback_state()
    assert dispatch.installed() is None


# --- the kernel registry ------------------------------------------------------------------


def test_register_and_look_up(lines, monkeypatch):
    monkeypatch.setattr(
        dispatch, "importlib", types.SimpleNamespace(import_module=lambda name: pytest.fail(name))
    )

    def impl():
        return 42

    assert dispatch.register("draft_front", "draft_front", impl) is impl
    assert dispatch.get_impl("draft_front", "draft_front")() == 42
    with pytest.raises(KeyError, match="'draft_front' does not register 'missing'"):
        dispatch.get_impl("draft_front", "missing")
    with pytest.raises(ValueError, match="'not_a_path' is not a fast path"):
        dispatch.register("not_a_path", "draft_front", impl)


@pytest.mark.parametrize("path", dispatch.FAST_PATH_NAMES)
def test_get_impl_imports_the_kernel_directory_of_the_path(lines, monkeypatch, path):
    imported = []
    monkeypatch.setattr(dispatch, "importlib", types.SimpleNamespace(import_module=imported.append))
    with pytest.raises(KeyError, match=f"{path!r} does not register 'entry'"):
        dispatch.get_impl(path, "entry")
    directory = dispatch.KERNEL_DIRECTORY.get(path, path)
    assert imported == [f"{dispatch.KERNELS_PACKAGE}.{directory}"]


def test_capability_lines_are_logged_once_per_key(lines):
    assert dispatch.log_capability("draft_front", "draft_front", True)
    assert not dispatch.log_capability("attention_output_tail", "layer3", False, "dense")
    # One line per (path, key) and process; the call still returns its own decision.
    assert not dispatch.log_capability("draft_front", "draft_front", False)
    assert lines == [
        "[glm_moe_dsa] draft_front/draft_front: engaged",
        "[glm_moe_dsa] attention_output_tail/layer3: native fallback (dense)",
    ]


# --- knobs folded into constants --------------------------------------------------------

# Every variable with the knob prefix that the kernels of this target once read; they run at
# fixed launch parameters now, so ``resolve`` refuses each of them when it is set.
FOLDED_KNOBS = tuple(
    dispatch.KNOB_PREFIX + name
    for name in (
        "DEFERRED_NORM_HOP_CARVEOUT",
        "DEFERRED_NORM_PROJECTION_CONFIG",
        "DEFERRED_NORM_EVICT_HINTS_HOP_CONFIG",
        "DEFERRED_NORM_EVICT_HINTS_PROJECTION_CONFIG",
        "ABSORBED_QUERY_FORM",
        "FUSED_KV_INDEXER_PROJECTION_LAYERS",
        "FUSED_KV_INDEXER_PROJECTION_CONFIG",
        "INDEXER_LAYER_FRONT_SKIP_REUSED_INDEXER_Q",
        "CROSS_RANK_TOPK_LOGITS_PDL",
        "ATTENTION_OUTPUT_TAIL_PDL_ENTRY",
        "OPROJ_WEIGHT_STAGING_CONFIG",
        "ROUTER_GEMM_EXPERTS_PER_CTA",
        "ROUTER_GEMM_PAIRED_FMA",
        "ROUTED_MOE_SCHEDULE",
        "ROUTED_MOE_EXPERT_CLUSTER_CTAS",
        "ROUTED_MOE_FINALIZE_MODE",
        "ROUTED_MOE_PDL_TRIGGER",
        "ROUTED_MOE_MIN_CLUSTERS",
        "ROUTED_MOE_STAGES",
        "ROUTED_MOE_GRID_WIDE_L2_PREFETCH",
        "ROUTED_MOE_MIN_FC1_KBLOCKS",
        "ROUTED_MOE_ACCUMULATOR_STAGES",
        "ROUTED_MOE_HANDOFF",
        "ROUTED_MOE_DELAY_SITES",
        "ROUTED_MOE_DELAY_NS",
        "ROUTED_MOE_EXPERT_CLUSTER_L2_PREFETCH",
        "ROUTED_MOE_TWO_ROUND_CLUSTERS",
        "SHARED_EXPERT_PREFETCH_CLUSTERS",
        "SHARED_EXPERT_PREFETCH_WEIGHTS",
        "DRAFT_ROUTED_MOE_FC1_CONFIG",
        "DRAFT_ROUTED_MOE_FC2_CONFIG",
        "DRAFT_ROUTED_MOE_FLAGS",
        "DRAFT_TAIL_MODE",
        "DRAFT_TAIL_EXCHANGE",
        "DRAFT_TAIL_VARIANT",
        "DRAFT_TAIL_CTAS_PER_SM",
        "DRAFT_TAIL_NUM_CTAS",
        "DRAFT_TAIL_PDL",
        "DRAFT_TAIL_TIMEOUT_CYCLES",
        "DRAFT_TAIL_KEEP_PERMILLE",
        "CROSS_LAYER_EXPERT_PREFETCH_START_POINT",
        "CROSS_LAYER_EXPERT_PREFETCH_CHUNK",
        "CROSS_LAYER_EXPERT_PREFETCH_CTAS",
        "CROSS_LAYER_EXPERT_PREFETCH_DELAY_NS",
        "CROSS_LAYER_EXPERT_PREFETCH_FUSED",
        "DRAFT_LOOP_PARTS",
        "DEFERRED_NORM_SCOPE",
        "CROSS_LAYER_EXPERT_PREFETCH_DEPTH",
        "CROSS_LAYER_EXPERT_PREFETCH_PREDICTOR",
        "CROSS_LAYER_EXPERT_PREFETCH_ISSUE_KERNEL",
        "CROSS_LAYER_EXPERT_PREFETCH_ROUTED_KERNEL_CTAS",
        "DRAFT_ABSORBED_QUERY_SKIP_REUSED_INDEXER_Q",
        "SHARED_INDEXER_ABSORBED_QUERY_FORCE_NATIVE",
        "ROPE_KV_APPEND_METADATA_HINTS",
        "ROPE_KV_APPEND_METADATA_PREFETCH",
    )
)


def test_the_knobs_of_the_target_are_the_plan_knobs():
    assert dispatch.KNOBS == {
        dispatch.DISABLE_ENV,
        ACCUMULATION,
        dispatch.KNOB_PREFIX + "CROSS_RANK_TOPK_TIMEOUT_MS",
    }
    assert not set(FOLDED_KNOBS) & dispatch.KNOBS


@pytest.mark.parametrize("knob", FOLDED_KNOBS, ids=lambda k: k.removeprefix(dispatch.KNOB_PREFIX))
def test_a_folded_knob_is_refused_even_at_its_value(lines, knob):
    with pytest.raises(ValueError) as excinfo:
        dispatch.resolve(AUTO, {knob: "0"})
    hint = dispatch.REMOVED_KNOBS.get(
        knob, "the fast paths of this target run at fixed launch parameters"
    )
    assert str(excinfo.value) == f"this target does not read {knob} ({hint}); unset them"
    assert dispatch.installed() is None
    assert lines == []


def test_the_knobs_of_the_target_resolve(lines):
    env = {
        dispatch.DISABLE_ENV: "draft_tail",
        ACCUMULATION: "identity",
        dispatch.KNOB_PREFIX + "CROSS_RANK_TOPK_TIMEOUT_MS": "100",
    }
    plan = dispatch.resolve(AUTO, env)
    assert plan.disabled == {"draft_tail"}
    assert plan.knobs == tuple(sorted(env.items()))


# Shared TensorRT-LLM switches a kernel of this target may read next to the plan.
_SHARED_READS = {"TRTLLM_ENABLE_PDL", "TRTLLM_FORCE_MNNVL_AR", "TLLM_TEST_MNNVL"}


def test_only_dispatch_reads_the_knobs():
    """No module of the family but ``dispatch`` reads the environment, except the shared switches."""
    root = os.path.dirname(os.path.dirname(dispatch.__file__))
    reads = []
    for directory, _, files in os.walk(root):
        for name in sorted(files):
            if not name.endswith(".py") or os.path.join(directory, name) == dispatch.__file__:
                continue
            path = os.path.join(directory, name)
            with open(path) as f:
                text = f.read()
            for match in re.finditer(r"os\.(?:environ|getenv)\b", text):
                read = re.match(
                    r"os\.(?:environ(?:\.get)?|getenv)\s*[\[(]\s*[\"']([A-Z_]+)[\"']",
                    text[match.start() :],
                )
                if read is None or read.group(1) not in _SHARED_READS:
                    line = text.count("\n", 0, match.start()) + 1
                    reads.append(f"{os.path.relpath(path, root)}:{line}")
    assert reads == []
