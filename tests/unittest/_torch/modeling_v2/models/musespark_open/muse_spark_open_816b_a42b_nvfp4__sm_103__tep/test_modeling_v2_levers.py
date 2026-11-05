# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The musespark_open sm103 TEP target's lever table, on the host only (no torch, no device).

``levers.py`` and ``dispatch.py`` are loaded from their files, with the three leaf modules
``dispatch`` imports stubbed for the load (``_router_index``, the communicator and the
logger); the compositions are ``modeling.py``'s ``TARGET_PATHS`` statements run over that
``dispatch``. The tests pin:

* the table of every region deployment under ``auto``, which is the table under ``baseline``;
* every decoding role (aggregated, generation, unknown) resolves alike;
* ``baseline``: every key's plan is the composition before the lever table
  (``C55_SERVED`` under ``c55_plan``'s rules) plus the install the plan adds to it;
* a knob naming a lever this target does not have raises, and no stale switch is retired here;
* the knob grammar's errors, the overrides, the facts and the drafter arena;
* ``compose`` agreed over a stubbed TP group, and the ``Plan`` reports.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[7]
_TARGET = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)
_PACKAGE = "_musespark_open_tep_levers_under_test"


class _Mode(str, enum.Enum):
    OFF = "off"
    AUTO = "auto"
    REQUIRE = "require"


class _Log:
    """The logger stub: every call as (level, message)."""

    def __init__(self) -> None:
        self.records: list = []

    def _record(self, level: str, message) -> None:
        self.records.append((level, str(message)))

    def info(self, message, *args, **kwargs) -> None:
        self._record("info", message)

    def warning(self, message, *args, **kwargs) -> None:
        self._record("warning", message)

    def debug(self, message, *args, **kwargs) -> None:
        self._record("debug", message)

    def error(self, message, *args, **kwargs) -> None:
        self._record("error", message)

    def messages(self, level: str) -> list:
        return [message for kind, message in self.records if kind == level]


class _Group:
    """``Distributed.get``'s TP group: this rank's items, and each other rank's as
    ``others`` makes them from this rank's (the same items by default)."""

    def __init__(self) -> None:
        self.world = 1
        self.others = None

    def tp_allgather(self, items, small_payload: bool = False) -> list:
        others = self.others or list
        return [list(items)] + [others(list(items)) for _ in range(self.world - 1)]


class _Distributed:
    group = _Group()

    @classmethod
    def get(cls, mapping) -> _Group:
        cls.group.world = int(mapping.tp_size)
        return cls.group


_LOG = _Log()


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _load():
    stubs = {
        "tensorrt_llm._torch._experimental.modeling_v2._router_index": _module(
            "_router_index", ModelingV2Mode=_Mode
        ),
        "tensorrt_llm._torch.distributed.communicator": _module(
            "communicator", Distributed=_Distributed
        ),
        "tensorrt_llm.logger": _module("logger", logger=_LOG),
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(_TARGET)]
        sys.modules[_PACKAGE] = package
        levers = importlib.import_module(f"{_PACKAGE}.levers")
        dispatch = importlib.import_module(f"{_PACKAGE}.dispatch")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return levers, dispatch


def _assigned(node) -> list:
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _compositions(dispatch):
    """``modeling.py``'s composition tables (its ``_TEP*``, ``_SM10*``, ``_PREFETCH*`` and
    ``TARGET_PATHS`` statements), run over ``dispatch``."""
    path = _TARGET / "modeling.py"
    wanted = re.compile(r"^(_(TEP|SM10|PREFETCH)\w*|TARGET_PATHS)$")
    body = [
        node
        for node in ast.parse(path.read_text()).body
        if _assigned(node) and all(wanted.match(name) for name in _assigned(node))
    ]
    namespace = {"_dispatch": dispatch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["TARGET_PATHS"]


LEVERS, DISPATCH = _load()
TARGET_PATHS = _compositions(DISPATCH)
RULES = LEVERS.RULES
BASELINE = {LEVERS.DEFAULT_KNOB: "baseline"}


def knobs(**values) -> dict:
    """Lever knobs: ``name="on"``, or ``name__KEY="value"`` for a parameter."""
    out = {}
    for spec, value in values.items():
        name, _, key = spec.partition("__")
        out[LEVERS.knob_name(name, key or None)] = str(value)
    return out


def table(key, mbs, role, arena, knob_set=None, facts=None):
    sm, tp = key
    return DISPATCH.table_for(TARGET_PATHS, sm, tp, mbs, role, arena, knob_set, facts)


def rule_levers_on(lever_table) -> set:
    return {name for name in RULES if lever_table.on(name)}


def base_off(lever_table) -> set:
    return {
        name
        for name, decision in lever_table.decisions.items()
        if name not in RULES and not decision.on
    }


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------
#: Where each rule-table lever is served, and its kind and origin.
SERVED_AT = {
    "lm_head_inbox": ({(103, 4), (103, 8)}, "composition", "upstream"),
    "graph_start_qkv_prefetch": ({(103, 8)}, "composition", "upstream"),
    "half_window_experts": ({(103, 4)}, "composition", "upstream"),
    "hot_replicas": ({(103, 4)}, "composition", "upstream"),
    "vocab_parallel_head": ({(103, 4), (103, 8)}, "composition", "upstream"),
    "multi_request_prep_append": ({(103, 4), (103, 8)}, "composition", "upstream"),
    "multi_request_split_kv": ({(103, 4), (103, 8)}, "composition", "upstream"),
    "multi_request_post_attention_hop": ({(103, 4), (103, 8)}, "composition", "upstream"),
    "wide_drafter_hops": ({(103, 4)}, "composition", "upstream"),
    "router_gemm": ({(103, 4), (103, 8)}, "composition", "upstream"),
    "projection_publish": ({(103, 4), (103, 8)}, "composition", "upstream"),
}


def test_the_rule_table_is_the_lever_registry():
    assert set(RULES) == set(SERVED_AT)
    assert set(RULES) <= DISPATCH.FAST_PATH_NAMES
    for name, (keys, kind, origin) in SERVED_AT.items():
        served = {key for key, paths in TARGET_PATHS.items() if name in paths.served}
        assert served == keys, name
        assert (RULES[name].kind.value, RULES[name].origin) == (kind, origin), name


def test_the_compositions_and_the_fast_paths():
    assert sorted(TARGET_PATHS) == [(103, 4), (103, 8)]
    served = TARGET_PATHS[(103, 4)].served | TARGET_PATHS[(103, 8)].served
    assert DISPATCH.FAST_PATH_NAMES == served
    assert (len(TARGET_PATHS[(103, 8)].served), len(TARGET_PATHS[(103, 4)].served)) == (55, 55)
    with pytest.raises(KeyError):
        table((107, 4), 4, "agg", "windowed")


def test_the_lever_records_carry_no_field_of_another_target():
    assert "band" not in LEVERS.SOURCES
    fields = {field.name for field in dataclasses.fields(LEVERS.Deployment)}
    assert not fields & {"draft_logit_transform", "rejection_sampling"}
    # The drafter facts another target reads are accepted and change nothing here.
    for transform in (None, False, True):
        for sampling in (None, False, True):
            lever_table = DISPATCH.table_for(
                TARGET_PATHS,
                103,
                8,
                1,
                "agg",
                "dense",
                draft_logit_transform=transform,
                rejection_sampling=sampling,
            )
            assert lever_table.paths() == table((103, 8), 1, "agg", "dense").paths()


# --------------------------------------------------------------------------
# The regions under auto
# --------------------------------------------------------------------------
VP = {"vocab_parallel_head"}
#: The TP4 paths of the composition that are rule levers, on wherever served.
TEP4_PATHS = set(LEVERS.TEP4_PATHS)
#: The rule levers among the multi-request paths TEP8 serves (``SM103_TEP8_ADDED``).
TEP8_MULTI_REQUEST = {
    "multi_request_prep_append",
    "multi_request_split_kv",
    "multi_request_post_attention_hop",
    "router_gemm",
    "projection_publish",
}

REGIONS = [
    pytest.param(
        (103, 8),
        "agg",
        (1, 8),
        "windowed",
        {"lm_head_inbox", "graph_start_qkv_prefetch"} | VP | TEP8_MULTI_REQUEST,
        id="tep8",
    ),
    pytest.param(
        (103, 4),
        "agg",
        (1, 4, 8),
        "windowed",
        {"lm_head_inbox", "half_window_experts"} | VP | TEP4_PATHS,
        id="tep4-low",
    ),
    pytest.param(
        (103, 4),
        "agg",
        (16, 64),
        "windowed",
        {"lm_head_inbox", "hot_replicas"} | VP | TEP4_PATHS,
        id="tep4",
    ),
]


@pytest.mark.parametrize("key, role, sizes, arena, expected", REGIONS)
def test_region_tables_under_auto(key, role, sizes, arena, expected):
    for mbs in sizes:
        lever_table = table(key, mbs, role, arena)
        assert rule_levers_on(lever_table) == expected, mbs
        assert base_off(lever_table) == set(), mbs
        assert lever_table.paths() == TARGET_PATHS[key].served - {
            name for name, d in lever_table.decisions.items() if not d.on
        }


def test_the_tep4_paths():
    assert TEP4_PATHS == {
        "multi_request_prep_append",
        "multi_request_split_kv",
        "multi_request_post_attention_hop",
        "wide_drafter_hops",
        "router_gemm",
        "projection_publish",
    }


def test_the_vocabulary_parallel_head_rule():
    lever_table = table((103, 4), 64, "agg", "windowed")
    assert lever_table.on("vocab_parallel_head")
    assert lever_table.param("vocab_parallel_head", "MIN_ROWS") == 16
    assert LEVERS.default_param("vocab_parallel_head", "MIN_ROWS") == 16
    floor = table((103, 4), 4, "agg", "windowed", knobs(vocab_parallel_head__MIN_ROWS=8))
    assert floor.param("vocab_parallel_head", "MIN_ROWS") == 8


@pytest.mark.parametrize("key", [(103, 4), (103, 8)])
def test_auto_resolves_as_baseline(key):
    for role in ("agg", "gen"):
        for arena in ("dense", "windowed", "paged"):
            for mbs in (None, 1, 4, 8, 9, 64):
                auto = table(key, mbs, role, arena)
                baseline = table(key, mbs, role, arena, BASELINE)
                assert auto.paths() == baseline.paths(), (role, arena, mbs)


@pytest.mark.parametrize("key", [(103, 4), (103, 8)])
@pytest.mark.parametrize("mbs", [1, 4, 8, 64])
def test_every_decoding_role_resolves_alike(key, mbs):
    def selection(role):
        lever_table = table(key, mbs, role, "windowed")
        return {n: (d.on, dict(d.params)) for n, d in lever_table.decisions.items()}

    agg = selection("agg")
    for role in ("gen", "unknown"):
        assert selection(role) == agg, role


# --------------------------------------------------------------------------
# baseline: the composition before the lever table
# --------------------------------------------------------------------------
#: The served sets of the compositions at c55183f445 (its ``modeling.py`` tables over its
#: ``dispatch.FAST_PATH_NAMES``).
C55_SERVED = {
    (103, 4): frozenset(
        {
            "batch_dense_gemms",
            "bf16_routed_experts",
            "boundary_early_release",
            "capture_fold",
            "context_kv_fork",
            "context_prep",
            "context_sequence_parallel",
            "decode_kv_splits",
            "deferred_input_prep",
            "dense_gemms",
            "draft_head",
            "drafter_block_conv",
            "drafter_block_decode",
            "drafter_fc_shard",
            "drafter_fixed_gemms",
            "drafter_hops",
            "drafter_pdl_chain",
            "drafter_window_compact",
            "finalize_transport",
            "finalize_two_shot",
            "full_attention_prep_append",
            "full_attention_prep_fold",
            "full_attention_split_kv",
            "fused_accept",
            "half_window_experts",
            "hot_replicas",
            "lm_head_inbox",
            "model_boundary",
            "moe_prologue",
            "multi_request_post_attention_hop",
            "multi_request_prep_append",
            "multi_request_split_kv",
            "path_walk_tail",
            "persistent_experts",
            "post_attention_hop",
            "preattn_prep",
            "projection_gathers",
            "projection_publish",
            "replicated_embedding",
            "router_gemm",
            "sampler_store",
            "split_sampler",
            "swa_block",
            "swa_paged_decode",
            "verify_fold",
            "vocab_parallel_head",
            "wide_draft_topk",
            "wide_drafter_hops",
            "wide_drafter_window_attention",
            "wide_moe_allreduce",
            "wide_projection_gathers",
            "wide_split_sampler",
        }
    ),
    (103, 8): frozenset(
        {
            "attention_output_into_hop",
            "bf16_routed_experts",
            "boundary_early_release",
            "capture_fold",
            "context_kv_fork",
            "deferred_input_prep",
            "dense_gemm_prefetch",
            "dense_gemms",
            "draft_head",
            "drafter_block_conv",
            "drafter_block_decode",
            "drafter_fc_shard",
            "drafter_fixed_gemms",
            "drafter_hops",
            "drafter_pdl_chain",
            "early_expert_ids",
            "finalize_transport",
            "full_attention_prep_append",
            "full_attention_prep_fold",
            "full_attention_split_kv",
            "fused_accept",
            "lm_head_inbox",
            "model_boundary",
            "moe_prologue",
            "next_qkv_prefetch",
            "path_walk_tail",
            "persistent_experts",
            "pool_into_route",
            "post_attention_hop",
            "preattn_prep",
            "projection_gathers",
            "replicated_embedding",
            "replicated_experts",
            "sampler_store",
            "split_sampler",
            "swa_block",
            "swa_paged_decode",
            "three_window_placement",
            "unpool_into_boundary",
            "verify_fold",
        }
    ),
}


#: The paths TEP8 serves beyond ``C55_SERVED``: the multi-request decode paths of
#: ``modeling._SM103_TEP8_MULTI_REQUEST_PATHS`` and ``wide_dense_gemms``, TEP8's alone. The
#: baseline serves them like the rest of the composition.
SM103_TEP8_ADDED = frozenset(
    {
        "drafter_window_compact",
        "multi_request_post_attention_hop",
        "multi_request_prep_append",
        "multi_request_split_kv",
        "projection_publish",
        "router_gemm",
        "vocab_parallel_head",
        "wide_dense_gemms",
        "wide_moe_allreduce",
        "wide_projection_gathers",
        "wide_split_sampler",
    }
)


def c55_plan(key, max_batch_size, arena) -> set:
    """``C55_SERVED[key]`` (with ``SM103_TEP8_ADDED`` at TEP8) under that composition's
    rules: ``half_window_experts`` only up to ``max_batch_size`` 8 (left out for an unknown
    one), ``hot_replicas`` left out of a plan that carries the half window, and the drafter
    arena's: over a windowed arena, ``context_kv_fork`` and ``verify_fold`` only with the
    window compaction; any other arena leaves the compaction out. No served path builds on
    one of these names there, so each rule removes the name alone. Then the base paths added
    since, each beside the path it builds on: ``draft_head_sampled`` (``draft_head``),
    ``rejection_fold`` (``verify_fold``) and ``paged_context_commit`` (``context_kv_fork``)."""
    paths = set(C55_SERVED[key])
    if key == (103, 8):
        paths |= SM103_TEP8_ADDED
    if max_batch_size is None or max_batch_size > 8:
        paths.discard("half_window_experts")
    if "half_window_experts" in paths:
        paths.discard("hot_replicas")
    if arena == "windowed":
        if "drafter_window_compact" not in paths:
            paths -= {"context_kv_fork", "verify_fold"}
    else:
        paths.discard("drafter_window_compact")
    for added, base in (
        ("draft_head_sampled", "draft_head"),
        ("rejection_fold", "verify_fold"),
        ("paged_context_commit", "context_kv_fork"),
    ):
        if base in paths:
            paths.add(added)
    return paths


@pytest.mark.parametrize("key", sorted(C55_SERVED))
@pytest.mark.parametrize("arena", ["dense", "windowed", "paged"])
@pytest.mark.parametrize("role", ["agg", "gen", "ctx"])
def test_baseline_is_the_previous_composition(key, arena, role):
    for mbs in (None, 1, 2, 4, 8, 9, 16, 64):
        want = c55_plan(key, mbs, arena)
        # The install the plan adds to c55_plan: the graph-start fork of next_qkv_prefetch.
        if "next_qkv_prefetch" in want:
            want.add("graph_start_qkv_prefetch")
        assert table(key, mbs, role, arena, BASELINE).paths() == want, mbs


# --------------------------------------------------------------------------
# The knob grammar
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "snapshot, message",
    [
        (knobs(lm_head_inbx="on"), "did you mean MUSESPARK_OPEN_LEVER_LM_HEAD_INBOX"),
        (knobs(hot_replicas="yes"), "expected one of ['auto', 'on', 'off']"),
        ({LEVERS.DEFAULT_KNOB: "fast"}, "expected one of ['auto', 'baseline']"),
        ({LEVERS.ROLE_KNOB: "decode"}, "MUSESPARK_OPEN_LEVER__ROLE='decode'"),
        (knobs(vocab_parallel_head__MIN_ROWS=7), "expected an integer in 8..16, got 7"),
        (knobs(vocab_parallel_head__MIN_ROWS="many"), "expected an integer in 8..16"),
        (knobs(half_window_experts__MIN_ROWS=8), "declares the parameters ['MAX_BATCH']"),
        (knobs(hot_replicas__ROWS=8), "declares no parameter"),
        (knobs(half_window_experts__MAX_BATCH=65), "expected an integer in 1..64, got 65"),
        ({"MUSESPARK_OPEN_LEVER_Hot_Replicas": "on"}, "no lever of this target"),
    ],
)
def test_a_malformed_knob_raises(snapshot, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        table((103, 4), 4, "agg", "windowed", snapshot)


def test_every_error_is_named_in_one_message():
    snapshot = {**knobs(lm_head_inbx="on", hot_replicas="yes")}
    with pytest.raises(ValueError) as raised:
        table((103, 4), 4, "agg", "windowed", snapshot)
    text = str(raised.value)
    assert "LM_HEAD_INBX" in text and "'yes'" in text


@pytest.mark.parametrize(
    "name", ["packed_prep_append", "megakernel", "rejection_sampler", "step_window_compact"]
)
@pytest.mark.parametrize("value", ["on", "off", "auto"])
def test_a_knob_naming_a_lever_this_target_does_not_have_raises(name, value):
    with pytest.raises(ValueError, match="no lever of this target is named"):
        table((103, 4), 4, "agg", "windowed", {LEVERS.PREFIX + name.upper(): value})


def test_no_switch_is_retired_here():
    assert LEVERS.RETIRED == {} and not LEVERS.RETIRED_PREFIXES
    stale = {"MUSESPARK_OPEN_PS_SIDE_CHAIN": "1", "MUSESPARK_OPEN_RS_Q_TRANSFORM": "1"}
    assert LEVERS.snapshot_of(stale) == {}
    for name in stale:
        assert LEVERS.retired_replacement(name) is None


@pytest.mark.parametrize(
    "key, mbs, snapshot, message",
    [
        ((103, 8), 1, knobs(hot_replicas="on"), "does not serve hot_replicas"),
        ((103, 4), 4, knobs(hot_replicas="on"), "cannot both be on"),
    ],
)
def test_on_raises_where_the_lever_cannot_be_on(key, mbs, snapshot, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        table(key, mbs, "agg", "windowed", snapshot)


@pytest.mark.parametrize("key", sorted(TARGET_PATHS))
@pytest.mark.parametrize("arena", ["dense", "windowed", "paged"])
def test_every_single_off_gives_a_closed_plan(key, arena):
    for mbs in (1, 4, 8, 48):
        for name in sorted(TARGET_PATHS[key].served):
            paths = table(key, mbs, "agg", arena, knobs(**{name: "off"})).paths()
            DISPATCH.Plan(paths)  # raises when not closed under depends_on
            for lever in paths & set(RULES):
                assert set(RULES[lever].requires) <= paths, (key, mbs, name, lever)


def test_a_parameter_moves_the_band():
    lever_table = table((103, 4), 16, "agg", "windowed", knobs(half_window_experts__MAX_BATCH=16))
    assert lever_table.on("half_window_experts")
    assert lever_table.param("half_window_experts", "MAX_BATCH") == 16
    assert not lever_table.on("hot_replicas")


def test_a_knob_of_an_unserved_lever_set_off_or_auto_only_warns():
    lever_table = table((103, 8), 1, "agg", "dense", knobs(hot_replicas="off"))
    assert any("has no effect" in warning for warning in lever_table.warnings)
    assert lever_table.paths() == table((103, 8), 1, "agg", "dense").paths()


def test_an_empty_knob_is_unset():
    snapshot = LEVERS.snapshot_of({LEVERS.knob_name("hot_replicas"): " ", "PATH": "/bin"})
    assert snapshot == {}


# --------------------------------------------------------------------------
# Facts, the drafter arena, the role
# --------------------------------------------------------------------------
def test_the_facts_of_a_snapshot():
    snapshot = LEVERS.snapshot_of(
        {LEVERS.KV_DEFER_CLOSE_ENV: "1", LEVERS.UCX_RNDV_THRESH_ENV: "8192", "HOME": "/root"}
    )
    assert set(snapshot) == {LEVERS.KV_DEFER_CLOSE_ENV, LEVERS.UCX_RNDV_THRESH_ENV}
    assert LEVERS.facts_of(snapshot) == LEVERS.Facts(kv_defer_close=True, ucx_rndv_thresh_set=True)
    assert LEVERS.facts_of({LEVERS.KV_DEFER_CLOSE_ENV: "0"}) == LEVERS.Facts()


def test_the_arena_of_the_worker_and_its_drafter():
    paged = ("TRTLLM", "FA4")
    worker = types.SimpleNamespace(_dflash_attention_backend="AUTO", windowed_ctx_arena=True)
    drafter = types.SimpleNamespace(dflash_attention_backend="TRTLLM")
    assert LEVERS.arena_of(worker, drafter, paged) is LEVERS.Arena.PAGED
    drafter = types.SimpleNamespace(dflash_attention_backend="FLASHINFER")
    assert LEVERS.arena_of(worker, drafter, paged) is LEVERS.Arena.WINDOWED
    drafter = types.SimpleNamespace(_paged_ctx_cache=True)
    assert LEVERS.arena_of(worker, drafter, paged) is LEVERS.Arena.PAGED
    dense = types.SimpleNamespace(_dflash_attention_backend="FLASHINFER", windowed_ctx_arena=False)
    assert LEVERS.arena_of(dense, None, paged) is LEVERS.Arena.DENSE


def test_the_arena_the_worker_built():
    assert LEVERS.worker_arena(types.SimpleNamespace(_ctx_buf_inited=False)) is None
    built = dict(_ctx_buf_inited=True, _ctx_paged=False, _ctx_base=None)
    assert LEVERS.worker_arena(types.SimpleNamespace(**built)) is LEVERS.Arena.DENSE
    built["_ctx_base"] = object()
    assert LEVERS.worker_arena(types.SimpleNamespace(**built)) is LEVERS.Arena.WINDOWED
    built["_ctx_paged"] = True
    assert LEVERS.worker_arena(types.SimpleNamespace(**built)) is LEVERS.Arena.PAGED


def test_the_window_compaction_needs_a_windowed_arena_and_the_full_context_paths_need_it():
    assert table((103, 4), 1, "agg", "windowed").on("drafter_window_compact")
    dense = table((103, 4), 1, "agg", "dense")
    assert dense.decisions["drafter_window_compact"].source == "structural"
    # TEP8 serves the compaction too, so its windowed arena keeps the arena-writing paths.
    tep8 = table((103, 8), 1, "agg", "windowed")
    assert tep8.on("context_kv_fork") and tep8.on("verify_fold")
    # A composition without the compaction leaves them out over a windowed arena.
    no_compaction = {
        **TARGET_PATHS,
        (103, 8): DISPATCH.TargetPaths(
            served=TARGET_PATHS[103, 8].served - {"drafter_window_compact"},
            topology=TARGET_PATHS[103, 8].topology,
            preconditions=TARGET_PATHS[103, 8].preconditions,
        ),
    }
    windowed = DISPATCH.table_for(no_compaction, 103, 8, 1, "agg", "windowed", None, None)
    assert windowed.decisions["context_kv_fork"].source == "structural"
    assert windowed.decisions["verify_fold"].source == "structural"
    # With the compaction served, a knob that turns it off takes the arena-writing paths along.
    off = table((103, 4), 1, "agg", "windowed", knobs(drafter_window_compact="off"))
    assert off.decisions["context_kv_fork"].source == "dependency"


def test_the_role():
    hint = {LEVERS.ROLE_KNOB: "gen"}
    assert LEVERS.role_of(hint, True) == (LEVERS.Role.GEN, "hint")
    assert LEVERS.role_of({}, False) == (LEVERS.Role.AGG, "no cache transceiver")
    assert LEVERS.role_of({}, True)[0] is LEVERS.Role.UNKNOWN
    assert LEVERS.role_of({}, None)[0] is LEVERS.Role.UNKNOWN
    with pytest.raises(ValueError, match="with a cache transceiver"):
        LEVERS.role_of({LEVERS.ROLE_KNOB: "agg"}, True)
    with pytest.raises(ValueError, match="without one|without a cache transceiver"):
        LEVERS.role_of(hint, False)
    unknown = table((103, 4), 4, "unknown", "windowed")
    assert any("role unknown" in warning for warning in unknown.warnings)
    assert rule_levers_on(unknown) == rule_levers_on(table((103, 4), 4, "gen", "windowed"))


def test_the_table_lines_and_json():
    lever_table = table((103, 4), 4, "gen", "windowed", knobs(router_gemm="off"))
    lines = lever_table.describe()
    assert lines[0].startswith("levers: sm103 tep4, gen (hint)")
    resolved = json.loads(lines[-1].split("lever table json (resolved): ", 1)[1])
    assert resolved == json.loads(lever_table.as_json())
    assert resolved["levers"]["router_gemm"] == {"resolved": "off", "source": "knob"}
    assert resolved["knobs"] == {LEVERS.knob_name("router_gemm"): "off"}
    assert resolved["deployment"]["max_batch_size"] == 4
    installed = json.loads(lever_table.as_json({"projection_publish": False}))
    assert installed["levers"]["projection_publish"]["installed"] is False
    assert installed["levers"]["hot_replicas"]["installed"] is False
    assert installed["levers"]["half_window_experts"]["installed"] is True


# --------------------------------------------------------------------------
# compose over a stubbed TP group, and the plan
# --------------------------------------------------------------------------
@pytest.fixture
def compose(monkeypatch):
    group = _Distributed.group
    monkeypatch.setattr(group, "others", None)
    _LOG.records.clear()
    monkeypatch.setattr(
        DISPATCH,
        "composition",
        lambda model_config, compositions, capability: compositions.get(
            (DISPATCH.sm_of(capability), model_config.mapping.tp_size)
        ),
    )
    monkeypatch.setattr(DISPATCH, "missing_preconditions", lambda *args: [])
    monkeypatch.setattr(DISPATCH, "unserved_reason", lambda *args: "not served")
    dflash = types.ModuleType("tensorrt_llm._torch.speculative.dflash")
    dflash._PAGED_ATTENTION_BACKENDS = ("TRTLLM", "FA4")
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.speculative.dflash", dflash)

    def run(key, mbs, snapshot=None, disaggregated=False, arena="dense", tp_rank=0):
        sm, tp = key
        mapping = types.SimpleNamespace(
            tp_size=tp, tp_rank=tp_rank, gpus_per_node=4, tp_group=list(range(tp))
        )
        worker = types.SimpleNamespace(
            _dflash_attention_backend="TRTLLM" if arena == "paged" else "FLASHINFER",
            windowed_ctx_arena=arena == "windowed",
        )
        facts = types.SimpleNamespace(
            max_batch_size=mbs, max_num_tokens=8192, disaggregated=disaggregated
        )
        return DISPATCH.compose(
            None,
            types.SimpleNamespace(mapping=mapping),
            _Mode.AUTO,
            TARGET_PATHS,
            (sm // 10, sm % 10),
            worker,
            mbs,
            facts=facts,
            snapshot=snapshot or {},
        )

    return run


def test_compose_plans_the_table_and_logs_it_on_rank_0(compose):
    plan = compose((103, 4), 4, arena="windowed")
    assert plan.paths == table((103, 4), 4, "agg", "windowed").paths()
    assert plan.levers.deployment.role_source == "no cache transceiver"
    assert any("lever table json (resolved)" in m for m in _LOG.messages("info"))
    _LOG.records.clear()
    compose((103, 4), 4, arena="windowed", tp_rank=1)
    info = _LOG.messages("info")
    assert any("drafter context arena" in m for m in info)
    assert not any("lever table json" in m for m in info)


def test_compose_raises_when_the_ranks_disagree_on_the_knobs(compose):
    _Distributed.group.others = lambda items: [item.replace("=off", "=on") for item in items]
    with pytest.raises(ValueError, match="disagree"):
        compose((103, 8), 1, knobs(lm_head_inbox="off"))


def test_compose_raises_another_ranks_lever_error(compose):
    _Distributed.group.others = lambda items: items + ["lever table: a rank's own error"]
    with pytest.raises(ValueError, match="a rank's own error"):
        compose((103, 8), 1)


def test_compose_raises_another_ranks_missing_precondition(compose):
    _Distributed.group.others = lambda items: items + ["a precondition"]
    with pytest.raises(ValueError, match="the fast paths need a precondition"):
        compose((103, 8), 1)


def test_compose_raises_this_ranks_lever_error_after_the_agreement(compose):
    with pytest.raises(ValueError, match="cannot both be on"):
        compose((103, 4), 4, knobs(hot_replicas="on"), arena="windowed")


def test_compose_without_a_composition_still_checks_the_knobs(compose, monkeypatch):
    monkeypatch.setattr(DISPATCH, "composition", lambda *args: None)
    monkeypatch.setattr(DISPATCH, "require_served", lambda *args: None)
    assert compose((103, 4), 4, knobs(hot_replicas="on")) is DISPATCH.NO_PATHS
    assert any("no effect without a composition" in m for m in _LOG.messages("info"))
    with pytest.raises(ValueError, match="no lever of this target is named"):
        compose((103, 4), 4, knobs(packed_prep_append="on"))


def test_the_role_hint_on_a_disaggregated_server(compose):
    plan = compose((103, 4), 4, {LEVERS.ROLE_KNOB: "gen"}, disaggregated=True, arena="windowed")
    assert plan.levers.deployment.role is LEVERS.Role.GEN
    with pytest.raises(ValueError, match="with a cache transceiver"):
        compose((103, 4), 4, {LEVERS.ROLE_KNOB: "agg"}, disaggregated=True)


def test_compose_names_a_bad_role_hint_with_every_other_grammar_error(compose):
    snapshot = {LEVERS.ROLE_KNOB: "decode", **knobs(lm_head_inbx="on")}
    with pytest.raises(ValueError) as raised:
        compose((103, 4), 4, snapshot, disaggregated=True, arena="windowed")
    text = str(raised.value)
    assert "'decode'" in text and "LM_HEAD_INBX" in text


def _router_index(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt_llm.logger", _module("logger", logger=_LOG))
    path = _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/_router_index.py"
    spec = importlib.util.spec_from_file_location("_router_index_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "transceiver, disaggregated",
    [
        (None, False),
        (types.SimpleNamespace(backend=None), False),
        (types.SimpleNamespace(backend="NIXL"), True),
    ],
)
def test_a_server_is_disaggregated_only_with_a_transceiver_backend(
    monkeypatch, transceiver, disaggregated
):
    args = types.SimpleNamespace(
        max_batch_size=4,
        max_num_tokens=8192,
        cuda_graph_config=None,
        cache_transceiver_config=transceiver,
    )
    facts = _router_index(monkeypatch)._facts_of(args)
    assert facts.disaggregated is disaggregated


def test_the_plan_reports(compose):
    plan = compose((103, 4), 16, knobs(router_gemm="on"), arena="windowed")
    assert plan.param("vocab_parallel_head", "MIN_ROWS") == 16
    plan.installed("projection_publish", False, "no workspace")
    assert any("projection_publish is not installed" in m for m in _LOG.messages("warning"))
    with pytest.raises(ValueError, match="ROUTER_GEMM=on"):
        plan.installed("router_gemm", False, "refused")
    report = json.loads(plan.report())
    assert report["levers"]["projection_publish"]["installed"] is False
    assert report["levers"]["hot_replicas"]["installed"] is True
    for _ in range(3):
        plan.declined("multi_request_split_kv", "a reason", "8 rows")
    assert plan.declines == {("multi_request_split_kv", "a reason"): 3}
    assert sum("declined a step" in m for m in _LOG.messages("warning")) == 1
    _LOG.records.clear()
    for _ in range(DISPATCH.DECLINE_REPORT_STEPS):
        plan.step()
    assert any("lever declines after" in m for m in _LOG.messages("info"))


def test_a_plan_built_from_names_takes_the_declared_parameters():
    plan = DISPATCH.plan_of("vocab_parallel_head")
    assert plan.param("vocab_parallel_head", "MIN_ROWS") == 16
    assert plan.report() == "{}"
    with pytest.raises(KeyError):
        plan.param("split_sampler", "MAX_ROWS")
    # A name this target does not serve is no fast path of its plan.
    for name in ("megakernel", "packed_prep_append"):
        with pytest.raises(KeyError):
            plan.enabled(name)
