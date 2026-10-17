# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the GLM-5.2 tp8 target classifies a fast path's decline: the sites, the layer cells, the
rules, and what ``log_capability`` does with each class.

Host-only: the declines are synthetic calls with the keys and reasons the kernels log, the log
goes into lists, and the sources are read as text, so nothing here needs a GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    dispatch,
)

pytestmark = pytest.mark.cpu_only

D = dispatch.Decline
# The family: its shared code and every target of it.
FAMILY = Path(dispatch.__file__).parents[1]
SCOPE = "TRTLLM_STAIRCASE_GLM_LL_DEFERRED_NORM_SCOPE"
LAYERS = "TRTLLM_STAIRCASE_GLM_LL_FUSED_KV_INDEXER_PROJECTION_LAYERS"


def _layers(spec: str) -> frozenset:
    """``"0-2,6,10"`` as a set of layer indices."""
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return frozenset(out)


SHARED = (
    "3-5,7-9,11-13,15-17,19-21,23-25,27-29,31-33,35-37,39-41,43-45,47-49,51-53,55-57,59-61,"
    "63-65,67-69,71-73,75-77"
)
FULL = "0-2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74"

# The layers each per-layer key engages on, on every rank, when this target serves its
# checkpoint: (path, key family, layers).
CENSUS = (
    ("absorbed_query", "layer", "0-77"),
    ("attention_output_tail", "layer", "0-77"),
    ("shared_indexer_absorbed_query", "layer", SHARED),
    ("deferred_norm", "scope[shared]_into_layer", SHARED.replace("3-5", "4-5")),
    ("sparse_mla_decode", "call:layer", "0-78"),
    ("sparse_mla_decode", "layer", "0-78"),
    ("draft_absorbed_query", "layer", "78"),
    ("deferred_norm_evict_hints", "projection_layer", SHARED.replace("3-5", "4-5")),
    ("indexer_layer_front", "draft_layer", "78"),
    ("indexer_layer_front", "target_layer", FULL),
    ("oproj_weight_staging", "layer", "0-77"),
    ("fused_kv_indexer_projection", "layer", "3"),
)

# Every decline the served composition logs, on every rank: (path, key, reason). None of them
# is a miss.
SERVED_DECLINES = (
    ("absorbed_query", "layer78", "target layers only"),
    ("attention_output_tail", "layer78", "draft or unindexed layer"),
    ("router_gemm", "router:not a target layer", "not a target layer"),
    *(
        (
            "shared_indexer_absorbed_query",
            f"layer{i}",
            "full indexer layer (shared_indexer_absorbed_query serves shared layers)",
        )
        for i in sorted(_layers(FULL))
    ),
    ("shared_indexer_absorbed_query", "layer78", "target layers only"),
    *(
        (
            "deferred_norm",
            f"scope[shared]_into_layer{i}",
            f"{SCOPE}=shared: layer {i} is a full-indexer layer; post_moe_hop hop",
        )
        for i in sorted(_layers(FULL) - {0, 1, 2})
    ),
    ("dense_mlp", "dense_mlp:row count outside [1, 8]", "row count outside [1, 8]"),
    ("cross_layer_expert_prefetch", "spec_experts_declined", "128 rows outside [1, 8]"),
    ("routed_moe", "rows=128 128 rows outside [1, 8]", "128 rows outside [1, 8]"),
    (
        "sparse_mla_decode",
        "request:not generation-only",
        "not generation-only; first seen at layer 3 with 1 rows",
    ),
    *(("draft_absorbed_query", f"layer{i}", "draft (MTP) layers only") for i in range(78)),
    ("draft_loop", "draft_loop:context rows present", "context rows present"),
    ("draft_loop", "draft_step:dense top-k fallback active", "dense top-k fallback active"),
    (
        "deferred_input_prep",
        "prep[no previous speculative device tensors]",
        "no previous speculative device tensors",
    ),
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
    *(
        (
            "indexer_layer_front",
            f"target_layer{i}",
            "shared-indexer layer (no q_a RMSNorm launch to remove; "
            "shared_indexer_absorbed_query / absorbed_query serve it)",
        )
        for i in sorted(_layers(SHARED))
    ),
    ("fused_kv_indexer_projection", "layer78", f"draft layers not selected ({LAYERS})"),
    *(
        ("fused_kv_indexer_projection", f"layer{i}", f"full layers not selected ({LAYERS})")
        for i in sorted(_layers(FULL))
    ),
    *(
        ("fused_kv_indexer_projection", f"layer{i}", f"shared layers not selected ({LAYERS})")
        for i in sorted(_layers(SHARED) - {3})
    ),
    *(
        (
            "draft_routed_moe",
            f"rows={rows} unfinalized output requested",
            "unfinalized output requested",
        )
        for rows in (1, 2, 6)
    ),
    ("draft_routed_moe", "rows=128 128 rows outside [1, 8]", "128 rows outside [1, 8]"),
    # Decided per object behind the first object's line: the deferred-norm hops into the
    # full-indexer layers, and a dense attention instance of a sparse-attention layer.
    (
        "deferred_norm",
        "deferred_hop",
        f"{SCOPE}=shared: layer 6 is a full-indexer layer; post_moe_hop hop",
    ),
    (
        "deferred_norm_evict_hints",
        "deferred_hop",
        f"{SCOPE}=shared: layer 6 is a full-indexer layer; post_moe_hop hop",
    ),
    ("sparse_mla_decode", "layer5", "not a DSA layer"),
)

# One decline per rule and site kind: (path, key, reason, class).
RULES = (
    # A path turned off.
    ("sampler_store", "engaged", "turned off by TRTLLM_STAIRCASE_GLM_LL_DISABLE", D.DISABLED),
    # A key no site names, and a path that does not exist.
    ("routed_moe", "a_key_no_site_names", "workspace failed", D.UNCOVERED),
    ("not_a_fast_path", "layer3", "workspace failed", D.UNCOVERED),
    # Decided per call: a missing resource is a miss that is never raised.
    (
        "routed_moe",
        "rows=6 not warmed up before CUDA graph capture",
        "not warmed up before CUDA graph capture",
        D.MISSED_PER_CALL,
    ),
    (
        "rope_kv_append",
        "scatter_defer",
        "call site does not forward the deferred scatter",
        D.MISSED_PER_CALL,
    ),
    (
        "draft_front",
        "draft_front_call_fallback",
        "kernel not compiled before CUDA graph capture",
        D.MISSED_PER_CALL,
    ),
    ("decode_prologue", "take_embedding", "packet does not match", D.MISSED_PER_CALL),
    (
        "indexer_layer_front",
        "cp_indexer_decode_global_emit",
        "workspace unavailable: peer rank",
        D.MISSED_PER_CALL,
    ),
    (
        "dense_mlp",
        "dense_mlp:glm_moe_dsa_dense_mlp op not built",
        "glm_moe_dsa_dense_mlp op not built",
        D.MISSED_PER_CALL,
    ),
    ("indexer_topk", "indexer_decode", "CUTLASS DSL unavailable", D.MISSED_PER_CALL),
    # Decided per call, outside the shapes (rows, metadata layout, step kind, KV band).
    ("rope_kv_append", "rope", "outside the FP8 MLA decode domain", D.UNCOVERED),
    ("sparse_mla_decode", "request:KV length layout", "KV length layout; first seen", D.UNCOVERED),
    (
        "decode_prologue",
        "target_prologue[metadata is not DSAtrtllmAttentionMetadata]",
        "metadata is not DSAtrtllmAttentionMetadata",
        D.UNCOVERED,
    ),
    ("draft_tail", "draft_tail[graph]", "draft logits are not vocab-sharded", D.UNCOVERED),
    (
        "draft_tail",
        "draft_tail[step: 9 rows outside [1, 8]]",
        "identical on every TP rank",
        D.UNCOVERED,
    ),
    ("sparse_mla_decode", "call:layer5", "first call", D.UNCOVERED),
    # Decided once: a layer kind or a composition the path does not serve.
    ("attention_output_into_hop", "layer3", "needs the post_attention_hop hop", D.UNCOVERED),
    (
        "attention_output_tail_prefetch",
        "layer3",
        "post_attention_hop without sparse_mla_decode: native tail is faster",
        D.UNCOVERED,
    ),
    (
        "attention_output_tail_prefetch",
        "schedule",
        "post_attention_hop without sparse_mla_decode",
        D.UNCOVERED,
    ),
    (
        "draft_live_rows",
        "live_row_hop:no post_attention_hop",
        "no post_attention_hop hop",
        D.UNCOVERED,
    ),
    ("sparse_mla_decode", "layer40", "not MLA", D.UNCOVERED),
    # Decided once, on a layer outside the cells.
    ("fused_kv_indexer_projection", "layer4", "kernel compile failed", D.UNCOVERED),
    ("attention_output_tail", "layer78", "staging buffer", D.UNCOVERED),
    # Decided once, inside the cells or on a key without a layer: a miss.
    ("fused_kv_indexer_projection", "layer3", "kernel compile failed", D.MISSED),
    ("sparse_mla_decode", "layer12", "latent dims", D.MISSED),
    ("sparse_mla_decode", "layer78", "KV cache is not FP8", D.MISSED),
    ("sparse_mla_decode", "layer0", "rope_append disabled", D.MISSED),
    ("indexer_layer_front", "rope", "indexer_layer_front rope op not loaded", D.MISSED),
    (
        "post_attention_hop",
        "post_attn_hop:native all-reduce is not MNNVL",
        "native all-reduce is not MNNVL",
        D.MISSED,
    ),
    ("post_attention_hop", "post_attn_hop", "", D.MISSED),
    ("router_gemm", "router:attention DP", "attention DP", D.MISSED),
    ("post_moe_hop", "cluster_kernel", "the cluster kernel needs SM100", D.MISSED),
    ("post_moe_hop", "post_moe_hop[cluster]", "workspace agreement failed", D.MISSED),
    ("routed_moe_into_hop", "post_moe_routed_input_buffer", "no buffer", D.MISSED),
    (
        "cross_layer_expert_prefetch",
        "link[k=8,start=moe]",
        "0 MoE layers, 0 routed-kernel prefetch hooks",
        D.MISSED,
    ),
    ("draft_front", "draft_front", "requires non-ADP tensor parallelism", D.MISSED),
    ("draft_tail", "draft_tail[tp8]", "workspace setup failed: OSError()", D.MISSED),
    ("host_page_retention", "host_policy", "libc10.so: not found", D.MISSED),
    # The loop domain is final only once the overlap loop has declared it.
    (
        "deferred_input_prep",
        "loop_domain",
        "no overlap executor loop declared its domain; the store writer and the single-stream "
        "loop are validated only there",
        D.UNCOVERED,
    ),
    (
        "deferred_input_prep",
        "loop_domain",
        "attention DP: the store writer and the single-stream loop are not validated with it",
        D.MISSED,
    ),
)


@pytest.fixture
def capability(monkeypatch):
    """A fresh capability log and fallback state; returns the info and warning lines."""
    info, warnings = [], []

    class _Logger:
        @staticmethod
        def info(message):
            info.append(message)

        @staticmethod
        def warning(message):
            warnings.append(message)

    monkeypatch.setattr(dispatch, "_log_info", info.append)
    monkeypatch.setattr(dispatch, "logger", _Logger)
    monkeypatch.setattr(dispatch, "_CAPABILITY_LOGGED", {})
    dispatch.reset_fallback_state()
    dispatch.set_require(False)
    dispatch.classify_decline.cache_clear()
    yield info, warnings
    dispatch.reset_fallback_state()
    dispatch.set_require(False)


# --- the tables ---------------------------------------------------------------------------


def test_the_layer_cells_are_the_census():
    expected = {}
    for path, family, layers in CENSUS:
        expected.setdefault(path, {})[family] = _layers(layers)
    assert {path: dict(families) for path, families in dispatch.LAYER_CELLS.items()} == expected


def test_every_site_names_a_fast_path_and_every_fast_path_has_a_site():
    assert {site.path for site in dispatch.DECLINE_KINDS} == set(dispatch.FAST_PATH_NAMES)


def test_every_family_with_cells_is_a_site_of_its_path():
    families = {(site.path, site.cells) for site in dispatch.DECLINE_KINDS if site.cells}
    with_cells = {
        (path, family) for path, cells in dispatch.LAYER_CELLS.items() for family in cells
    }
    assert with_cells <= families
    # The per-layer keys the served composition never logs have no cells: every decline there
    # is outside the path's shapes.
    assert families - with_cells == {
        ("deferred_norm", "projection_layer"),
        ("attention_output_tail_prefetch", "layer"),
        ("attention_output_into_hop", "layer"),
    }


def test_a_site_with_cells_captures_the_layer():
    for site in dispatch.DECLINE_KINDS:
        assert ("layer" in site.key.groupindex) == (site.cells is not None), site


def _kernel_sources() -> list:
    """The sources that log declines: every file of the family except the classifier itself."""
    return [p for p in sorted(FAMILY.rglob("*.py")) if p != Path(dispatch.__file__)]


def _source_text() -> str:
    return "\n".join(p.read_text() for p in _kernel_sources())


@pytest.mark.parametrize("fragment", dispatch.LAYER_KIND_REASONS + dispatch.DISABLED_REASONS)
def test_every_reason_fragment_is_a_reason_the_kernels_give(fragment):
    """A renamed reason must rename its fragment too, or its declines change class."""
    assert fragment in _source_text()


def _string_literals() -> set:
    return {
        node.value
        for source in _kernel_sources()
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_the_served_reasons_are_the_kernels_literals():
    """Each constant reason of a served decline is a whole string literal of the kernels, so a
    reworded reason fails here before its declines change class."""
    literals = _string_literals()
    # The reasons without a formatted field (a number or a knob name).
    reasons = {
        reason
        for _, _, reason in SERVED_DECLINES
        if not any(c.isdigit() for c in reason) and dispatch.KNOB_PREFIX not in reason
    }
    assert len(reasons) == 13
    assert sorted(reasons - literals) == []


def _render(node: ast.AST):
    """A key the call can log: f-string fields become a sample value."""
    if isinstance(node, ast.Constant):
        return node.value
    if not isinstance(node, ast.JoinedStr):
        return None
    out = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            out.append(value.value)
        elif isinstance(value.value, ast.IfExp) and isinstance(value.value.body, ast.Constant):
            out.append(value.value.body.value)
        elif isinstance(value.value, ast.Attribute) and value.value.attr == "scope":
            out.append("shared")
        else:
            out.append("0")
    return "".join(out)


def test_every_logged_key_has_a_site():
    """Each ``log_capability`` call whose path and key can be read off the source logs a key
    that a site of its path matches."""
    checked, missing = 0, []
    for source in _kernel_sources():
        tree = ast.parse(source.read_text())
        constants = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and len(node.args) >= 2):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            if name != "log_capability":
                continue
            first = node.args[0]
            path = (
                first.value
                if isinstance(first, ast.Constant)
                else constants.get(getattr(first, "id", None))
            )
            key = _render(node.args[1])
            if path is None or key is None:
                continue
            checked += 1
            if dispatch._decline_site(path, key) is None:
                missing.append(f"{source.name}:{node.lineno} {path} {key!r}")
    assert missing == []
    assert checked >= 70


# --- the rules ----------------------------------------------------------------------------


@pytest.mark.parametrize("path, key, reason", SERVED_DECLINES)
def test_no_decline_of_the_served_composition_is_a_miss(path, key, reason):
    assert dispatch.classify_decline(path, key, reason) is D.UNCOVERED


@pytest.mark.parametrize("path, key, reason, kind", RULES)
def test_each_rule(path, key, reason, kind):
    assert dispatch.classify_decline(path, key, reason) is kind


def test_a_resource_reason_is_matched_in_any_case():
    assert (
        dispatch.classify_decline("routed_moe", "rows=4 x", "Workspace FAILED") is D.MISSED_PER_CALL
    )


def test_a_repeated_decline_is_one_lookup(capability):
    args = ("routed_moe", "rows=128 128 rows outside [1, 8]", "128 rows outside [1, 8]")
    for _ in range(5):
        dispatch.log_capability(*args[:2], False, args[2])
    info = dispatch.classify_decline.cache_info()
    assert (info.misses, info.hits) == (1, 4)


# --- what log_capability does with each class ---------------------------------------------


def test_an_engaged_decision_is_not_classified(capability):
    assert dispatch.log_capability("post_moe_hop", "post_moe_hop[cluster]", True)
    assert dispatch.classify_decline.cache_info().misses == 0


def test_a_miss_decided_once_raises_under_require_after_its_line(capability):
    info, warnings = capability
    dispatch.set_require(True)
    with pytest.raises(
        dispatch.FastPathUnavailable, match=r"'post_moe_hop'.*post_moe_hop\[cluster\]: workspace"
    ):
        dispatch.log_capability("post_moe_hop", "post_moe_hop[cluster]", False, "workspace failed")
    assert info == [
        "[glm_moe_dsa] post_moe_hop/post_moe_hop[cluster]: native fallback (workspace failed)"
    ]
    assert warnings == []


def test_a_miss_decided_once_is_counted_and_warned_once_under_auto(capability):
    info, warnings = capability
    for _ in range(2):
        assert not dispatch.log_capability("sparse_mla_decode", "layer7", False, "latent dims")
    assert dispatch.missed_counts() == {"sparse_mla_decode": 2}
    assert warnings == [
        "[glm_moe_dsa] sparse_mla_decode/missed: layer7: latent dims; falling back (logged once per reason)"
    ]
    assert len(info) == 1


@pytest.mark.parametrize("require", [False, True], ids=["auto", "require"])
def test_a_miss_decided_per_call_is_counted_and_never_raised(capability, require):
    info, warnings = capability
    dispatch.set_require(require)
    key, reason = (
        "rows=6 not warmed up before CUDA graph capture",
        "not warmed up before CUDA graph capture",
    )
    for _ in range(3):
        assert not dispatch.log_capability("routed_moe", key, False, reason)
    assert dispatch.missed_counts() == {"routed_moe": 3}
    assert warnings == [
        f"[glm_moe_dsa] routed_moe/missed: {reason}; falling back (logged once per reason)"
    ]
    assert info == [f"[glm_moe_dsa] routed_moe/{key}: native fallback ({reason})"]


@pytest.mark.parametrize(
    "path, key, reason",
    [
        ("absorbed_query", "layer78", "target layers only"),
        ("routed_moe", "rows=128 128 rows outside [1, 8]", "128 rows outside [1, 8]"),
        ("sampler_store", "engaged", "turned off by TRTLLM_STAIRCASE_GLM_LL_DISABLE"),
        ("routed_moe", "a_key_no_site_names", "workspace failed"),
    ],
    ids=["uncovered-once", "uncovered-per-call", "disabled", "unknown-site"],
)
@pytest.mark.parametrize("require", [False, True], ids=["auto", "require"])
def test_uncovered_and_disabled_declines_add_nothing_to_their_line(
    capability, path, key, reason, require
):
    info, warnings = capability
    dispatch.set_require(require)
    for _ in range(2):
        assert not dispatch.log_capability(path, key, False, reason)
    assert info == [f"[glm_moe_dsa] {path}/{key}: native fallback ({reason})"]
    assert warnings == [] and dispatch.missed_counts() == {}


def test_the_served_declines_are_not_raised_under_require(capability):
    """Under require, the served composition's declines run as before: logged, not raised."""
    dispatch.set_require(True)
    for path, key, reason in SERVED_DECLINES:
        dispatch.log_capability(path, key, False, reason)
    _, warnings = capability
    assert warnings == [] and dispatch.missed_counts() == {}
