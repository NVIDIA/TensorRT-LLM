# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The musespark_open TEP target's lever table, on the host only (no torch, no device).

``levers.py`` and ``dispatch.py`` are loaded from their files, with the three leaf modules
``dispatch`` imports stubbed for the load (``_router_index``, the communicator and the
logger); the compositions are ``modeling.py``'s ``TARGET_PATHS`` statements run over that
``dispatch``. The tests pin:

* the table of every region deployment under ``auto``, and of the replay knob sets (``R_LL``, ``R_H``, ``B_LL``);
* the ladder shapes, where an agg and a gen server resolve alike;
* ``baseline``: every key's plan is the composition before the lever table
  (``C55_SERVED`` under ``c55_plan``'s rules) plus the two installs the plan adds to it;
* the knob grammar's errors, the overrides, the facts and the drafter arena;
* ``compose`` agreed over a stubbed TP group, and the ``Plan`` reports.
"""

from __future__ import annotations

import ast
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
HC_MIN_BATCH = LEVERS.HC_MIN_BATCH
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
    "packed_prep_append": ({(107, 4)}, "lossless-bitwise", "HC"),
    "windowed_commit_parallel": ({(107, 4), (107, 8)}, "lossless-bitwise", "HC"),
    "context_fp8_dense": ({(107, 4)}, "lossy", "HC"),
    "context_fp8_attn_gather": ({(107, 4)}, "lossy", "HC"),
    "wide_direct_transports": ({(107, 4)}, "lossless-bitwise", "midc"),
    "wide_router_per_row": ({(107, 4)}, "lossless-numeric", "midc"),
    "wide_oproj_publish": ({(107, 4)}, "lossless-bitwise", "midc"),
    "wide_finalize_transport": ({(107, 4)}, "lossless-bitwise", "midc"),
    "lm_head_inbox": ({(103, 4), (103, 8), (107, 8)}, "composition", "upstream"),
    "graph_start_qkv_prefetch": ({(103, 8), (107, 4), (107, 8)}, "composition", "upstream"),
    "step_loop_on_execution_stream": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "incremental_block_offsets": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "drafter_kv_side_chain": ({(107, 8)}, "lossless-bitwise", "LL"),
    "drafter_fc_early_fork": ({(107, 8)}, "lossless-bitwise", "LL"),
    "drafter_preamble_from_commit": ({(107, 8)}, "lossless-bitwise", "LL"),
    "head_sampler_coop_fold": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "head_multimem_gather": ({(107, 8)}, "flagged", "LL"),
    "half_window_experts": ({(103, 4), (107, 4)}, "composition", "upstream"),
    "hot_replicas": ({(103, 4), (107, 4)}, "composition", "upstream"),
    "prefill_cutedsl_experts": ({(107, 4)}, "composition", "upstream"),
}


def test_the_rule_table_is_the_registry_of_twenty_levers():
    assert set(RULES) == set(SERVED_AT)
    assert set(RULES) <= DISPATCH.FAST_PATH_NAMES
    for name, (keys, kind, origin) in SERVED_AT.items():
        served = {key for key, paths in TARGET_PATHS.items() if name in paths.served}
        assert served == keys, name
        assert (RULES[name].kind.value, RULES[name].origin) == (kind, origin), name


def test_no_lever_of_another_origin_is_served_on_sm103():
    others = {name for name, rule in RULES.items() if rule.origin != "upstream"}
    for key in ((103, 4), (103, 8)):
        assert not others & TARGET_PATHS[key].served


# --------------------------------------------------------------------------
# The regions under auto
# --------------------------------------------------------------------------
LL = {"step_loop_on_execution_stream", "incremental_block_offsets", "head_sampler_coop_fold"}
DENSE = {"drafter_kv_side_chain", "drafter_fc_early_fork", "drafter_preamble_from_commit"}
MIDC = {"wide_direct_transports", "wide_oproj_publish", "wide_finalize_transport"}
HC = {"packed_prep_append", "windowed_commit_parallel"}
PREFILL = {"prefill_cutedsl_experts"}

REGIONS = [
    pytest.param((107, 8), "agg", (1,), "dense", LL | DENSE, id="ll-agg-tep8-mbs1"),
    pytest.param(
        (107, 4),
        "agg",
        (1,),
        "windowed",
        LL | {"half_window_experts", "wide_finalize_transport"} | PREFILL,
        id="ll-agg-tep4-mbs1",
    ),
    pytest.param((107, 8), "gen", (1,), "windowed", LL, id="gen-tep8-b1"),
    pytest.param(
        (107, 4),
        "gen",
        (1,),
        "windowed",
        LL | {"half_window_experts", "wide_finalize_transport"} | PREFILL,
        id="gen-tep4-b1",
    ),
    pytest.param(
        (107, 4),
        "gen",
        (2, 3, 4),
        "windowed",
        MIDC | {"wide_router_per_row"} | LL | {"half_window_experts"} | PREFILL,
        id="gen-tep4-b2-b4",
    ),
    pytest.param(
        (107, 4),
        "agg",
        (2, 3, 4),
        "windowed",
        MIDC | LL | {"half_window_experts"} | PREFILL,
        id="agg-caps-2-4",
    ),
    pytest.param(
        (107, 4),
        "agg",
        (6, 8),
        "windowed",
        HC
        | MIDC
        | {"incremental_block_offsets", "head_sampler_coop_fold", "half_window_experts"}
        | PREFILL,
        id="agg-caps-6-8",
    ),
    pytest.param(
        (107, 4),
        "agg",
        (16, 32),
        "windowed",
        HC | MIDC | {"head_sampler_coop_fold", "hot_replicas"} | PREFILL,
        id="agg-mid-high",
    ),
    pytest.param(
        (107, 4),
        "agg",
        (48, 64),
        "windowed",
        HC | {"graph_start_qkv_prefetch", "hot_replicas"} | PREFILL,
        id="agg-hc",
    ),
    pytest.param(
        (107, 4),
        "gen",
        (8,),
        "windowed",
        HC
        | MIDC
        | {"incremental_block_offsets", "head_sampler_coop_fold", "half_window_experts"}
        | PREFILL,
        id="gen-tep4-b8",
    ),
    pytest.param(
        (107, 4),
        "gen",
        (16, 32, 64),
        "windowed",
        HC | MIDC | {"head_sampler_coop_fold", "hot_replicas"} | PREFILL,
        id="gen-tep4-b16-b64",
    ),
    pytest.param((107, 8), "agg", (2, 4), "dense", LL | DENSE, id="agg-tep8-mbs2-4"),
    pytest.param(
        (107, 8),
        "agg",
        (8,),
        "dense",
        {"incremental_block_offsets", "head_sampler_coop_fold"} | DENSE,
        id="agg-tep8-mbs8",
    ),
    pytest.param(
        (107, 4), "ctx", (8, 64), "windowed", {"wide_finalize_transport"} | PREFILL, id="ctx-tep4"
    ),
    pytest.param(
        (103, 8),
        "agg",
        (1, 8),
        "dense",
        {"lm_head_inbox", "graph_start_qkv_prefetch"},
        id="sm103-tep8",
    ),
    pytest.param(
        (103, 4),
        "agg",
        (1, 4, 8),
        "windowed",
        {"lm_head_inbox", "half_window_experts"},
        id="sm103-tep4-low",
    ),
    pytest.param(
        (103, 4), "agg", (16, 64), "windowed", {"lm_head_inbox", "hot_replicas"}, id="sm103-tep4"
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


def test_the_router_per_row_gates():
    on = table((107, 4), 4, "gen", "windowed").decisions["wide_router_per_row"]
    assert on.on and on.source == "auto" and "gate" in on.why
    off = table((107, 4), 8, "agg", "windowed").decisions["wide_router_per_row"]
    assert not off.on and off.source == "gate"
    for mbs in (2, 3, 4):
        off = table((107, 4), mbs, "agg", "windowed").decisions["wide_router_per_row"]
        assert not off.on and off.source == "gate", mbs
    assert not table((107, 4), 1, "agg", "windowed").on("wide_router_per_row")


def test_the_lossy_and_flagged_levers_are_never_on_under_auto():
    for key, paths in TARGET_PATHS.items():
        for mbs in (1, 4, 8, 64):
            lever_table = table(key, mbs, "agg", "dense")
            for name, rule in RULES.items():
                if rule.kind.value in ("lossy", "flagged") and name in paths.served:
                    assert not lever_table.on(name), (key, mbs, name)


@pytest.mark.parametrize("key", [(103, 4), (103, 8)])
def test_sm103_resolves_under_auto_as_under_baseline(key):
    for role in ("agg", "gen"):
        for arena in ("dense", "windowed", "paged"):
            for mbs in (None, 1, 4, 8, 9, 64):
                auto = table(key, mbs, role, arena)
                baseline = table(key, mbs, role, arena, BASELINE)
                assert auto.paths() == baseline.paths(), (role, arena, mbs)


# --------------------------------------------------------------------------
# The replay knob sets and the ladders
# --------------------------------------------------------------------------
R_LL = knobs(head_multimem_gather="on")
R_H = knobs(
    wide_direct_transports="off",
    wide_oproj_publish="off",
    wide_finalize_transport="off",
    head_sampler_coop_fold="off",
    graph_start_qkv_prefetch="on",
    windowed_commit_parallel__MIN_REQUESTS=1,
    context_fp8_dense="on",
    context_fp8_attn_gather="on",
)
B_LL = knobs(
    step_loop_on_execution_stream="off",
    incremental_block_offsets="off",
    head_sampler_coop_fold="off",
    graph_start_qkv_prefetch="on",
)


def test_r_ll_selects_the_low_latency_set():
    lever_table = table((107, 8), 1, "agg", "dense", R_LL)
    assert rule_levers_on(lever_table) == LL | DENSE | {"head_multimem_gather"}
    assert lever_table.decisions["head_multimem_gather"].source == "knob"
    assert not lever_table.on("lm_head_inbox") and not lever_table.on("graph_start_qkv_prefetch")


def test_r_h_selects_the_high_concurrency_set():
    lever_table = table((107, 4), 48, "agg", "windowed", R_H)
    assert (
        rule_levers_on(lever_table)
        == HC
        | {
            "context_fp8_dense",
            "context_fp8_attn_gather",
            "graph_start_qkv_prefetch",
            "hot_replicas",
        }
        | PREFILL
    )
    assert lever_table.param("windowed_commit_parallel", "MIN_REQUESTS") == 1
    assert lever_table.param("packed_prep_append", "MIN_ROWS") == 33
    assert lever_table.param("context_fp8_dense", "MIN_ROWS") == 1024
    assert lever_table.param("context_fp8_dense", "PROJECTIONS") == (
        "qkv",
        "o_proj",
        "pool",
        "unpool",
    )


@pytest.mark.parametrize("mbs", [1, 2, 3, 4])
def test_b_ll_selects_the_generation_set(mbs):
    lever_table = table((107, 4), mbs, "gen", "windowed", B_LL)
    wide = (
        {"wide_direct_transports", "wide_router_per_row", "wide_oproj_publish"}
        if mbs > 1
        else set()
    )
    assert (
        rule_levers_on(lever_table)
        == wide
        | {
            "wide_finalize_transport",
            "graph_start_qkv_prefetch",
            "half_window_experts",
        }
        | PREFILL
    )


def test_baseline_tep8_gen_mbs1_is_the_previous_generation_selection():
    lever_table = table((107, 8), 1, "gen", "windowed", BASELINE)
    assert rule_levers_on(lever_table) == {"lm_head_inbox", "graph_start_qkv_prefetch"}


R_B1 = dict(B_LL, **knobs(wide_finalize_transport="off"))
_LL_REPLAY = LL | DENSE | {"head_multimem_gather"}


@pytest.mark.parametrize(
    "key, mbs, role, arena, knob_set, extra, less",
    [
        # model_boundary has no knob: the replayed code built layer 0's input in the forward.
        pytest.param(
            (107, 8), 1, "agg", "dense", R_LL, _LL_REPLAY, {"lm_head_inbox"}, id="ll-tep8-mbs1"
        ),
        pytest.param(
            (107, 4),
            48,
            "agg",
            "windowed",
            R_H,
            HC
            | PREFILL
            | {"context_fp8_dense", "context_fp8_attn_gather", "graph_start_qkv_prefetch"},
            set(),
            id="hc-tep4-mbs48",
        ),
        *[
            pytest.param(
                (107, 4),
                mbs,
                "gen",
                "windowed",
                B_LL,
                MIDC | PREFILL | {"wide_router_per_row", "graph_start_qkv_prefetch"},
                set(),
                id=f"gen-tep4-mbs{mbs}",
            )
            for mbs in (2, 3, 4)
        ],
        pytest.param(
            (107, 4),
            1,
            "gen",
            "windowed",
            R_B1,
            PREFILL | {"graph_start_qkv_prefetch"},
            set(),
            id="gen-tep4-mbs1",
        ),
        pytest.param(
            (107, 8),
            1,
            "gen",
            "windowed",
            BASELINE,
            {"graph_start_qkv_prefetch"},
            set(),
            id="baseline-tep8-gen-mbs1",
        ),
    ],
)
def test_a_replay_set_is_the_whole_plan_it_replays(key, mbs, role, arena, knob_set, extra, less):
    want = (c55_plan(key, mbs, arena) | extra) - less
    assert set(table(key, mbs, role, arena, knob_set).paths()) == want


def test_auto_at_the_high_concurrency_selection_is_r_h():
    opt_ins = knobs(context_fp8_dense="on", context_fp8_attn_gather="on")
    for mbs in (HC_MIN_BATCH, 64):
        auto = table((107, 4), mbs, "agg", "windowed", opt_ins).decisions
        replay = table((107, 4), mbs, "agg", "windowed", R_H).decisions
        assert {
            name
            for name, d in auto.items()
            if (d.on, dict(d.params)) != (replay[name].on, dict(replay[name].params))
        } == set(), mbs


def test_the_high_concurrency_selection_is_aggregated_tep4_from_its_batch():
    below = table((107, 4), HC_MIN_BATCH - 1, "agg", "windowed")
    assert below.param("windowed_commit_parallel", "MIN_REQUESTS") == 5
    assert below.on("wide_direct_transports") and not below.on("graph_start_qkv_prefetch")
    hc = table((107, 4), HC_MIN_BATCH, "agg", "windowed")
    assert hc.param("windowed_commit_parallel", "MIN_REQUESTS") == 1
    assert not hc.on("wide_direct_transports") and hc.on("graph_start_qkv_prefetch")
    for key, role in (((107, 4), "gen"), ((107, 8), "agg")):
        other = table(key, HC_MIN_BATCH, role, "windowed")
        assert other.param("windowed_commit_parallel", "MIN_REQUESTS") == 5, (key, role)
        assert not other.on("graph_start_qkv_prefetch"), (key, role)


@pytest.mark.parametrize("arm", [None, B_LL, BASELINE], ids=["auto", "b-ll", "baseline"])
@pytest.mark.parametrize("key, mbs", [((107, 4), 4), ((107, 4), 1), ((107, 8), 1)])
def test_the_ladder_shapes_resolve_alike_on_agg_and_gen(key, mbs, arm):
    def selection(role):
        lever_table = table(key, mbs, role, "windowed", arm)
        return {n: (d.on, dict(d.params)) for n, d in lever_table.decisions.items()}

    agg, gen = selection("agg"), selection("gen")
    # The router-per-row accuracy gate holds on generation servers only.
    gated = {"wide_router_per_row"} if gen.get("wide_router_per_row", (False,))[0] else set()
    assert {n for n in agg if agg[n] != gen[n]} == gated
    assert all(not agg[n][0] for n in gated)


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
    (107, 4): frozenset(
        {
            "attention_output_into_hop",
            "batch_dense_gemms",
            "bf16_routed_experts",
            "boundary_early_release",
            "capture_fold",
            "context_kv_fork",
            "context_prep",
            "context_sequence_parallel",
            "deferred_input_prep",
            "dense_gemm_prefetch",
            "dense_gemms",
            "draft_head",
            "drafter_block_conv",
            "drafter_block_decode",
            "drafter_fc_inbox",
            "drafter_fc_shard",
            "drafter_fixed_gemms",
            "drafter_hops",
            "drafter_pdl_chain",
            "early_expert_ids",
            "finalize_transport",
            "finalize_two_shot",
            "full_attention_prep_append",
            "full_attention_prep_fold",
            "full_attention_split_kv",
            "fused_accept",
            "half_window_experts",
            "hot_replicas",
            "mixed_step_gen_rows",
            "model_boundary",
            "moe_prologue",
            "multi_request_post_attention_hop",
            "multi_request_prep_append",
            "multi_request_split_kv",
            "next_qkv_prefetch",
            "path_walk_tail",
            "persistent_experts",
            "pool_into_route",
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
            "unpool_into_boundary",
            "verify_fold",
            "wide_dense_gemms",
            "wide_draft_topk",
            "wide_drafter_hops",
            "wide_drafter_window_attention",
            "wide_full_attention",
            "wide_moe_allreduce",
            "wide_persistent_experts",
            "wide_post_attention_hop",
            "wide_prologue",
            "wide_split_sampler",
            "wide_swa_block",
        }
    ),
    (107, 8): frozenset(
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


def c55_plan(key, max_batch_size, arena) -> set:
    """``C55_SERVED[key]`` under that composition's rules: ``half_window_experts`` only up to
    ``max_batch_size`` 8 (left out for an unknown one), ``hot_replicas`` left out of a plan
    that carries the half window, and the drafter arena's: over a windowed arena on an
    architecture other than sm107, ``context_kv_fork`` and ``verify_fold`` only with the
    window compaction; any other arena leaves the compaction out. No served path builds on
    one of these names there, so each rule removes the name alone."""
    paths = set(C55_SERVED[key])
    if max_batch_size is None or max_batch_size > 8:
        paths.discard("half_window_experts")
    if "half_window_experts" in paths:
        paths.discard("hot_replicas")
    if arena == "windowed":
        if "drafter_window_compact" not in paths and key[0] != 107:
            paths -= {"context_kv_fork", "verify_fold"}
    else:
        paths.discard("drafter_window_compact")
    return paths


@pytest.mark.parametrize("key", sorted(C55_SERVED))
@pytest.mark.parametrize("arena", ["dense", "windowed", "paged"])
@pytest.mark.parametrize("role", ["agg", "gen", "ctx"])
def test_baseline_is_the_previous_composition(key, arena, role):
    for mbs in (None, 1, 2, 4, 8, 9, 16, 64):
        want = c55_plan(key, mbs, arena)
        # The two installs the plan adds to c55_plan: the graph-start fork of next_qkv_prefetch,
        # and the large prefill expert calls the sm107 TEP4 load installed whenever composed.
        if "next_qkv_prefetch" in want:
            want.add("graph_start_qkv_prefetch")
        if key == (107, 4):
            want.add("prefill_cutedsl_experts")
        assert table(key, mbs, role, arena, BASELINE).paths() == want, mbs


def test_baseline_leaves_every_other_origin_off_unless_its_knob_sets_it():
    lever_table = table(
        (107, 4), 4, "agg", "windowed", {**BASELINE, **knobs(head_sampler_coop_fold="on")}
    )
    others = {name for name, rule in RULES.items() if rule.origin != "upstream"}
    assert rule_levers_on(lever_table) & others == {"head_sampler_coop_fold"}
    decision = lever_table.decisions["wide_direct_transports"]
    assert decision.source == "baseline" and "midc" in decision.why


# --------------------------------------------------------------------------
# The knob grammar
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "snapshot, message",
    [
        (knobs(packed_prep_apend="on"), "did you mean MUSESPARK_OPEN_LEVER_PACKED_PREP_APPEND"),
        (knobs(step_loop_on_execution_stream="yes"), "expected one of ['auto', 'on', 'off']"),
        ({LEVERS.DEFAULT_KNOB: "fast"}, "expected one of ['auto', 'baseline']"),
        ({LEVERS.ROLE_KNOB: "decode"}, "MUSESPARK_OPEN_LEVER__ROLE='decode'"),
        (knobs(packed_prep_append__MIN_ROWS=7), "expected an integer in 8..1024, got 7"),
        (knobs(packed_prep_append__MIN_ROWS="many"), "expected an integer in 8..1024"),
        (knobs(packed_prep_append__MAX_ROWS=8), "declares the parameters ['MIN_ROWS']"),
        (knobs(drafter_kv_side_chain__ROWS=8), "declares no parameter"),
        (knobs(context_fp8_dense__MIN_ROWS=512), "expected an integer in 1024..32768"),
        (knobs(context_fp8_dense__PROJECTIONS="qkv,lm_head"), "comma-separated subset"),
        (knobs(head_sampler_coop_fold__MAX_ROWS=129), "expected an integer in 1..128, got 129"),
        (knobs(head_sampler_coop_fold__MAX_ROWS=0), "expected an integer in 1..128, got 0"),
        (
            knobs(windowed_commit_parallel__MIN_REQUESTS=65),
            "expected an integer in 1..64, got 65",
        ),
        (
            knobs(step_loop_on_execution_stream__MAX_BATCH=65),
            "expected an integer in 1..64, got 65",
        ),
        (knobs(incremental_block_offsets__MAX_BATCH=65), "expected an integer in 1..64, got 65"),
        (knobs(half_window_experts__MAX_BATCH=65), "expected an integer in 1..64, got 65"),
        ({"MUSESPARK_OPEN_LEVER_Packed_Prep_Append": "on"}, "no lever of this target"),
    ],
)
def test_a_malformed_knob_raises(snapshot, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        table((107, 4), 4, "agg", "windowed", snapshot)


def test_every_error_is_named_in_one_message():
    snapshot = {**knobs(packed_prep_apend="on", step_loop_on_execution_stream="yes")}
    snapshot["MUSESPARK_OPEN_PS_SIDE_CHAIN"] = "1"
    with pytest.raises(ValueError) as raised:
        table((107, 4), 4, "agg", "windowed", snapshot)
    text = str(raised.value)
    assert "PACKED_PREP_APEND" in text and "'yes'" in text and "is retired" in text


#: The retired switches, pinned here so that a name dropped from ``levers.RETIRED`` fails.
SPEC_RETIRED = (
    "MUSESPARK_OPEN_PREP_PACKED_ROWS",
    "MUSESPARK_OPEN_CTX_COMMIT_TUNED",
    "MUSESPARK_OPEN_CONTEXT_FP8_DENSE",
    "MUSESPARK_OPEN_CONTEXT_FP8_DENSE_MIN_ROWS",
    "MUSESPARK_OPEN_CONTEXT_FP8_DENSE_PROJECTIONS",
    "MUSESPARK_OPEN_CONTEXT_FP8_DENSE_GATHER",
    "TRTLLM_MUSESPARK_TEP_WIDE_DIRECT",
    "TRTLLM_MUSESPARK_TEP_WIDE_ROUTER_ROWS",
    "TRTLLM_MUSESPARK_TEP_WIDE_OPROJ_PUBLISH",
    "MUSESPARK_OPEN_STEP_LOOP_ONE_STREAM",
    "MUSESPARK_OPEN_INCREMENTAL_BLOCK_OFFSETS",
    "MUSESPARK_OPEN_PS_SIDE_CHAIN",
    "MUSESPARK_OPEN_PS_EARLY_FC",
    "MUSESPARK_OPEN_PS_DRAFTER_PREAMBLE",
    "MUSESPARK_OPEN_PS_COOP_FOLD",
    "MUSESPARK_OPEN_PS_SYMM_GATHER",
    "MUSESPARK_OPEN_PS_NO_START_PREFETCH",
    "MUSESPARK_OPEN_PS_NO_LM_HEAD_INBOX",
    "MUSESPARK_OPEN_PS_LAYER0_INPUT",
    "MUSESPARK_OPEN_PS_STEP_COMMIT_PDL",
    "MUSESPARK_OPEN_HEAD_INBOX_POLL_SCOPE",
    "MUSESPARK_OPEN_HEAD_INBOX_RESET_SCOPE",
    "TRTLLM_MUSESPARK_SWA_BLOCKS_PER_PROGRAM",
    "TRTLLM_MUSESPARK_TEP_WIDE_FULL_FOLD",
    "TRTLLM_MUSESPARK_TEP_WIDE_QKV_PREFETCH",
    "TRTLLM_MUSESPARK_TEP_WIDE_CAPTURE_FOLD",
)


def test_retired_is_the_spec_list():
    assert set(LEVERS.RETIRED) == set(SPEC_RETIRED)


@pytest.mark.parametrize("name", sorted(SPEC_RETIRED) + ["MUSESPARK_OPEN_PS_ANYTHING_ELSE"])
def test_a_retired_switch_raises_naming_its_replacement(name):
    replacement = LEVERS.retired_replacement(name)
    assert replacement
    with pytest.raises(ValueError, match=re.escape(f"{name} is retired (use {replacement})")):
        table((107, 4), 4, "agg", "windowed", {name: "0"})


@pytest.mark.parametrize(
    "key, mbs, arena, snapshot, message",
    [
        (
            (103, 8),
            1,
            "dense",
            knobs(step_loop_on_execution_stream="on"),
            "does not serve step_loop_on_execution_stream",
        ),
        ((107, 8), 1, "windowed", knobs(drafter_kv_side_chain="on"), "dense full-length"),
        (
            (107, 8),
            1,
            "dense",
            knobs(drafter_kv_side_chain="off", drafter_fc_early_fork="on"),
            "builds on drafter_kv_side_chain, which is off",
        ),
        ((107, 4), 4, "windowed", knobs(hot_replicas="on"), "cannot both be on"),
        (
            (107, 8),
            8,
            "dense",
            knobs(head_multimem_gather="on"),
            "requires step_loop_on_execution_stream, which is off",
        ),
        (
            (107, 8),
            1,
            "dense",
            knobs(head_multimem_gather="on", lm_head_inbox="on"),
            "cannot both be on",
        ),
        (
            (107, 4),
            1,
            "windowed",
            knobs(
                context_fp8_dense="on",
                context_fp8_dense__PROJECTIONS="o_proj",
                context_fp8_attn_gather="on",
            ),
            "PROJECTIONS leaves out",
        ),
    ],
)
def test_on_raises_where_the_lever_cannot_be_on(key, mbs, arena, snapshot, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        table(key, mbs, "agg", arena, snapshot)


# --------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------
def test_off_turns_a_lever_and_what_builds_on_it_off():
    lever_table = table((107, 8), 1, "agg", "dense", knobs(drafter_kv_side_chain="off"))
    decisions = lever_table.decisions
    assert (decisions["drafter_kv_side_chain"].on, decisions["drafter_kv_side_chain"].source) == (
        False,
        "knob",
    )
    assert (decisions["drafter_fc_early_fork"].on, decisions["drafter_fc_early_fork"].source) == (
        False,
        "dependency",
    )
    assert decisions["drafter_preamble_from_commit"].on


@pytest.mark.parametrize("key", sorted(TARGET_PATHS))
@pytest.mark.parametrize("arena", ["dense", "windowed", "paged"])
def test_every_single_off_gives_a_closed_plan(key, arena):
    for mbs in (1, 4, 8, 48):
        for name in sorted(TARGET_PATHS[key].served):
            paths = table(key, mbs, "agg", arena, knobs(**{name: "off"})).paths()
            DISPATCH.Plan(paths)  # raises when not closed under depends_on
            for lever in paths & set(RULES):
                assert set(RULES[lever].requires) <= paths, (key, mbs, name, lever)


def test_a_base_path_knob_turns_its_dependents_off():
    lever_table = table((107, 4), 8, "agg", "windowed", knobs(multi_request_prep_append="off"))
    assert not lever_table.on("multi_request_prep_append")
    assert lever_table.decisions["packed_prep_append"].source == "dependency"


def test_a_parameter_moves_the_band():
    lever_table = table((107, 4), 1, "agg", "windowed", knobs(packed_prep_append__MIN_ROWS=8))
    assert lever_table.on("packed_prep_append")
    assert lever_table.param("packed_prep_append", "MIN_ROWS") == 8
    lever_table = table(
        (107, 4), 1, "agg", "windowed", knobs(windowed_commit_parallel__MIN_REQUESTS=1)
    )
    assert lever_table.on("windowed_commit_parallel")
    lever_table = table(
        (107, 8), 8, "agg", "dense", knobs(step_loop_on_execution_stream__MAX_BATCH=8)
    )
    assert lever_table.on("step_loop_on_execution_stream")


def test_on_outside_a_numeric_gate_warns():
    lever_table = table((107, 4), 8, "agg", "windowed", knobs(wide_router_per_row="on"))
    assert lever_table.on("wide_router_per_row")
    assert any("outside its accuracy gates" in warning for warning in lever_table.warnings)


def test_a_knob_of_an_unserved_lever_set_off_or_auto_only_warns():
    lever_table = table((103, 8), 1, "agg", "dense", knobs(head_sampler_coop_fold="off"))
    assert any("has no effect" in warning for warning in lever_table.warnings)
    assert lever_table.paths() == table((103, 8), 1, "agg", "dense").paths()


def test_an_empty_knob_is_unset():
    snapshot = LEVERS.snapshot_of({LEVERS.knob_name("hot_replicas"): " ", "PATH": "/bin"})
    assert snapshot == {}


# --------------------------------------------------------------------------
# Facts and the drafter arena
# --------------------------------------------------------------------------
def test_the_facts_of_a_snapshot():
    snapshot = LEVERS.snapshot_of(
        {LEVERS.KV_DEFER_CLOSE_ENV: "1", LEVERS.UCX_RNDV_THRESH_ENV: "8192", "HOME": "/root"}
    )
    assert set(snapshot) == {LEVERS.KV_DEFER_CLOSE_ENV, LEVERS.UCX_RNDV_THRESH_ENV}
    assert LEVERS.facts_of(snapshot) == LEVERS.Facts(kv_defer_close=True, ucx_rndv_thresh_set=True)
    assert LEVERS.facts_of({LEVERS.KV_DEFER_CLOSE_ENV: "0"}) == LEVERS.Facts()


def test_the_deferred_close_leaves_the_one_stream_loop_out():
    facts = LEVERS.Facts(kv_defer_close=True)
    lever_table = table((107, 8), 1, "agg", "dense", facts=facts)
    decision = lever_table.decisions["step_loop_on_execution_stream"]
    assert (decision.on, decision.source) == (False, "structural")
    with pytest.raises(ValueError, match="requires step_loop_on_execution_stream"):
        table((107, 8), 1, "agg", "dense", knobs(head_multimem_gather="on"), facts)


def test_rndv_raises_on_a_disaggregated_server_only():
    facts = LEVERS.Facts(ucx_rndv_thresh_set=True)
    assert table((107, 4), 4, "agg", "windowed", facts=facts).deployment.facts == facts
    for role in ("gen", "ctx"):
        with pytest.raises(ValueError, match=LEVERS.UCX_RNDV_THRESH_ENV):
            table((107, 4), 4, role, "windowed", facts=facts)


def test_rndv_on_an_unhinted_disaggregated_server_raises():
    facts = LEVERS.Facts(ucx_rndv_thresh_set=True)
    with pytest.raises(ValueError, match=LEVERS.UCX_RNDV_THRESH_ENV):
        table((107, 4), 4, "unknown", "windowed", facts=facts)
    lever_table = DISPATCH.table_for(
        TARGET_PATHS, 107, 4, 4, "unknown", "windowed", facts=facts, role_source="no engine facts"
    )
    assert lever_table.deployment.role is LEVERS.Role.UNKNOWN


def test_context_fp8_dense_on_refuses_a_reachable_decode_step():
    with pytest.raises(ValueError, match="decode steps of 1024"):
        table((107, 4), 128, "agg", "windowed", knobs(context_fp8_dense="on"))
    with pytest.raises(ValueError, match="max_batch_size unknown"):
        table((107, 4), None, "agg", "windowed", knobs(context_fp8_dense="on"))
    moved = knobs(context_fp8_dense="on", context_fp8_dense__MIN_ROWS=2048)
    assert table((107, 4), 128, "agg", "windowed", moved).on("context_fp8_dense")
    assert table((107, 4), 48, "agg", "windowed", knobs(context_fp8_dense="on")).on(
        "context_fp8_dense"
    )


def test_a_paged_arena_leaves_the_arena_writing_levers_out():
    lever_table = table((107, 8), 1, "agg", "paged")
    for name in DENSE | {"incremental_block_offsets", "windowed_commit_parallel"}:
        assert lever_table.decisions[name].source == "structural", name
    assert lever_table.on("step_loop_on_execution_stream")
    assert lever_table.on("context_kv_fork")


def test_the_window_compaction_needs_a_windowed_arena_and_the_full_context_paths_need_it():
    assert table((103, 4), 1, "agg", "windowed").on("drafter_window_compact")
    dense = table((103, 4), 1, "agg", "dense")
    assert dense.decisions["drafter_window_compact"].source == "structural"
    sm103_tep8 = table((103, 8), 1, "agg", "windowed")
    assert sm103_tep8.decisions["context_kv_fork"].source == "structural"
    assert sm103_tep8.decisions["verify_fold"].source == "structural"
    assert table((107, 8), 1, "agg", "windowed").on("context_kv_fork")
    # With the compaction served, a knob that turns it off takes the arena-writing paths along.
    off = table((103, 4), 1, "agg", "windowed", knobs(drafter_window_compact="off"))
    assert off.decisions["context_kv_fork"].source == "dependency"


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
    unknown = table((107, 4), 4, "unknown", "windowed")
    assert any("role unknown" in warning for warning in unknown.warnings)
    # Counted as a decoding server; no accuracy gate names an unknown role.
    gen = rule_levers_on(table((107, 4), 4, "gen", "windowed"))
    assert rule_levers_on(unknown) == gen - {"wide_router_per_row"}
    assert unknown.decisions["wide_router_per_row"].source == "gate"


def test_the_table_lines_and_json():
    lever_table = table((107, 4), 4, "gen", "windowed", knobs(hot_replicas="off"))
    lines = lever_table.describe()
    assert lines[0].startswith("levers: sm107 tep4, gen (hint)")
    resolved = json.loads(lines[-1].split("lever table json (resolved): ", 1)[1])
    assert resolved == json.loads(lever_table.as_json())
    assert resolved["levers"]["hot_replicas"] == {"resolved": "off", "source": "knob"}
    assert resolved["knobs"] == {LEVERS.knob_name("hot_replicas"): "off"}
    assert resolved["deployment"]["max_batch_size"] == 4
    installed = json.loads(lever_table.as_json({"wide_oproj_publish": False}))
    assert installed["levers"]["wide_oproj_publish"]["installed"] is False
    assert installed["levers"]["wide_direct_transports"]["installed"] is True


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
    plan = compose((107, 4), 4, arena="windowed")
    assert plan.paths == table((107, 4), 4, "agg", "windowed").paths()
    assert plan.levers.deployment.role_source == "no cache transceiver"
    assert any("lever table json (resolved)" in m for m in _LOG.messages("info"))
    _LOG.records.clear()
    compose((107, 4), 4, arena="windowed", tp_rank=1)
    info = _LOG.messages("info")
    assert any("drafter context arena" in m for m in info)
    assert not any("lever table json" in m for m in info)


def test_compose_raises_when_the_ranks_disagree_on_the_knobs(compose):
    _Distributed.group.others = lambda items: [item.replace("=on", "=off") for item in items]
    with pytest.raises(ValueError, match="disagree"):
        compose((107, 8), 1, knobs(head_multimem_gather="on"))


def test_compose_raises_another_ranks_lever_error(compose):
    _Distributed.group.others = lambda items: items + ["lever table: a rank's own error"]
    with pytest.raises(ValueError, match="a rank's own error"):
        compose((107, 8), 1)


def test_compose_raises_another_ranks_missing_precondition(compose):
    _Distributed.group.others = lambda items: items + ["a precondition"]
    with pytest.raises(ValueError, match="the fast paths need a precondition"):
        compose((107, 8), 1)


def test_compose_raises_this_ranks_lever_error_after_the_agreement(compose):
    with pytest.raises(ValueError, match="cannot both be on"):
        compose((107, 4), 4, knobs(hot_replicas="on"), arena="windowed")


def test_compose_without_a_composition_still_checks_the_knobs(compose, monkeypatch):
    monkeypatch.setattr(DISPATCH, "composition", lambda *args: None)
    monkeypatch.setattr(DISPATCH, "require_served", lambda *args: None)
    assert compose((107, 4), 4, knobs(hot_replicas="on")) is DISPATCH.NO_PATHS
    assert any("no effect without a composition" in m for m in _LOG.messages("info"))
    with pytest.raises(ValueError, match="is retired"):
        compose((107, 4), 4, {"MUSESPARK_OPEN_PS_SIDE_CHAIN": "1"})


def test_the_role_hint_on_a_disaggregated_server(compose):
    plan = compose((107, 4), 4, {LEVERS.ROLE_KNOB: "gen"}, disaggregated=True, arena="windowed")
    assert plan.levers.deployment.role is LEVERS.Role.GEN
    with pytest.raises(ValueError, match="with a cache transceiver"):
        compose((107, 4), 4, {LEVERS.ROLE_KNOB: "agg"}, disaggregated=True)


def test_compose_names_a_bad_role_hint_with_every_other_grammar_error(compose):
    snapshot = {LEVERS.ROLE_KNOB: "decode", **knobs(packed_prep_apend="on")}
    with pytest.raises(ValueError) as raised:
        compose((107, 4), 4, snapshot, disaggregated=True, arena="windowed")
    text = str(raised.value)
    assert "'decode'" in text and "PACKED_PREP_APEND" in text


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
    plan = compose((107, 4), 4, knobs(wide_oproj_publish="on"), arena="windowed")
    assert plan.param("packed_prep_append", "MIN_ROWS") == 33
    plan.installed("wide_direct_transports", False, "no wide workspace")
    assert any("wide_direct_transports is not installed" in m for m in _LOG.messages("warning"))
    with pytest.raises(ValueError, match="WIDE_OPROJ_PUBLISH=on"):
        plan.installed("wide_oproj_publish", False, "refused")
    report = json.loads(plan.report())
    assert report["levers"]["wide_direct_transports"]["installed"] is False
    assert report["levers"]["half_window_experts"]["installed"] is True
    for _ in range(3):
        plan.declined("incremental_block_offsets", "a reason", "8 rows")
    assert plan.declines == {("incremental_block_offsets", "a reason"): 3}
    assert sum("declined a step" in m for m in _LOG.messages("warning")) == 1
    _LOG.records.clear()
    for _ in range(DISPATCH.DECLINE_REPORT_STEPS):
        plan.step()
    assert any("lever declines after" in m for m in _LOG.messages("info"))


def test_a_plan_built_from_names_takes_the_declared_parameters():
    plan = DISPATCH.plan_of("split_sampler")
    assert plan.param("head_sampler_coop_fold", "MAX_ROWS") == 16
    assert plan.report() == "{}"
    with pytest.raises(KeyError):
        plan.param("split_sampler", "MAX_ROWS")
