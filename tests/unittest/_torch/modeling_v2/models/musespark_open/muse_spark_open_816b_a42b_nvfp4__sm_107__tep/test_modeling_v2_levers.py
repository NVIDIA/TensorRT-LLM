# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The musespark_open sm107 TEP target's lever table, on the host only (no torch, no device).

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
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
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
    "lm_head_inbox": ({(107, 8)}, "composition", "upstream"),
    "graph_start_qkv_prefetch": ({(107, 4), (107, 8)}, "composition", "upstream"),
    "step_loop_on_execution_stream": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "incremental_block_offsets": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "drafter_kv_side_chain": ({(107, 8)}, "lossless-bitwise", "LL"),
    "drafter_fc_early_fork": ({(107, 8)}, "lossless-bitwise", "LL"),
    "drafter_preamble_from_commit": ({(107, 8)}, "lossless-bitwise", "LL"),
    "head_sampler_coop_fold": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "head_multimem_gather": ({(107, 8)}, "flagged", "LL"),
    "half_window_experts": ({(107, 4)}, "composition", "upstream"),
    "hot_replicas": ({(107, 4)}, "composition", "upstream"),
    "prefill_cutedsl_experts": ({(107, 4)}, "composition", "upstream"),
    "vocab_parallel_head": ({(107, 4), (107, 8)}, "composition", "upstream"),
    "multi_request_prep_append": ({(107, 4), (107, 8)}, "composition", "upstream"),
    "wide_swa_block": ({(107, 4), (107, 8)}, "composition", "upstream"),
    "wide_full_attention": ({(107, 4), (107, 8)}, "composition", "upstream"),
    "multi_request_split_kv": ({(107, 4), (107, 8)}, "composition", "upstream"),
    "multi_request_post_attention_hop": ({(107, 4), (107, 8)}, "composition", "upstream"),
    "wide_drafter_hops": ({(107, 4), (107, 8)}, "composition", "upstream"),
    "router_gemm": ({(107, 4), (107, 8)}, "composition", "upstream"),
    "wide_dense_gemms": ({(107, 4), (107, 8)}, "composition", "upstream"),
    "projection_publish": ({(107, 4), (107, 8)}, "composition", "upstream"),
    "weight_prefetch_after_join": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "next_qkv_prefetch_skip": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "swa_one_cluster_block": ({(107, 4), (107, 8)}, "lossless-numeric", "LL"),
    "swa_prep_split": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "full_attention_fold_multi": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "split_kv_deep_rings": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "prerouted_experts": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "finalize_entry_trigger": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "hop_entry_trigger": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "step_commit_v2": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "wide_norm_gate_pdl": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "drafter_kv_glue": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "drafter_kv_fork_before_head": ({(107, 4), (107, 8)}, "lossless-bitwise", "LL"),
    "splitk_pool_split8": ({(107, 4)}, "lossless-numeric", "LL"),
    "splitk_frontier_tables": ({(107, 4)}, "lossless-numeric", "LL"),
    "bf16_half_window": ({(107, 4)}, "lossless-numeric", "LL"),
    "context_sp_extra_parts": ({(107, 4)}, "lossless-bitwise", "LL"),
    "tep8_wide64_splitk": ({(107, 8)}, "lossless-numeric", "LL"),
    "tep8_publish_splitk": ({(107, 8)}, "lossless-numeric", "LL"),
    "tep8_drafter_wide_gemms": ({(107, 8)}, "lossless-numeric", "LL"),
    "tep8_fc_shard_wide": ({(107, 8)}, "lossless-numeric", "LL"),
    "swa_eight_requests": ({(107, 4), (107, 8)}, "lossless-numeric", "LL"),
    "tep8_ring_buddy_wide": ({(107, 8)}, "lossless-numeric", "LL"),
    "tep8_ring_buddy_pdl": ({(107, 8)}, "lossless-bitwise", "LL"),
    "tep8_tile_placement": ({(107, 8)}, "lossless-numeric", "LL"),
    "tep8_router_push": ({(107, 8)}, "lossless-numeric", "LL"),
    "tep8_router_prewait": ({(107, 8)}, "lossless-bitwise", "LL"),
    "tep8_finalize_transport_64": ({(107, 8)}, "lossless-bitwise", "LL"),
    "tep8_early_multicast_hop": ({(107, 8)}, "lossless-bitwise", "LL"),
    "tep8_direct_hop": ({(107, 8)}, "lossless-bitwise", "LL"),
    "tep8_publish_join": ({(107, 8)}, "lossless-bitwise", "LL"),
    "tep8_pool_programmatic_fork": ({(107, 8)}, "lossless-bitwise", "LL"),
    "tep8_wide_projection_gathers": ({(107, 8)}, "lossless-numeric", "LL"),
    "step_window_compact": ({(107, 4)}, "lossless-bitwise", "HC"),
    "fmk_context_fmha": ({(107, 4)}, "lossless-numeric", "HC"),
    "context_kv_late_join": ({(107, 4)}, "lossless-bitwise", "HC"),
    "hop_cluster_split": ({(107, 4)}, "lossless-bitwise", "HC"),
    "drafter_fc_inbox_wide": ({(107, 4)}, "lossless-bitwise", "HC"),
    "l061_tile32_pin": ({(107, 4)}, "lossless-numeric", "HC"),
    "router_staged_epilogue": ({(107, 4)}, "lossless-bitwise", "HC"),
    "decode_qkv_mxfp8": ({(107, 4)}, "lossy", "HC"),
    "split_kv_deep_ring": ({(107, 4)}, "lossless-bitwise", "HC"),
    "context_prep_rows_vec": ({(107, 4)}, "lossless-bitwise", "HC"),
    "context_embedding_replicated": ({(107, 4)}, "lossless-bitwise", "HC"),
    "context_sp_router_input": ({(107, 4)}, "lossless-bitwise", "HC"),
    "swa_blocks_per_program": ({(107, 4)}, "lossless-bitwise", "midc"),
    "wide_qkv_prefetch": ({(107, 4)}, "lossless-bitwise", "midc"),
    "wide_capture_fold": ({(107, 4)}, "lossless-bitwise", "midc"),
    "megakernel": ({(107, 8)}, "lossless-bitwise", "LL"),
    "megakernel_one_exchange_boundaries": ({(107, 8)}, "lossless-numeric", "LL"),
    "megakernel_two_term_pool": ({(107, 8)}, "lossless-numeric", "LL"),
    "megakernel_router_by_linearity": ({(107, 8)}, "lossless-numeric", "LL"),
    "megakernel_zero_exchange_boundary": ({(107, 8)}, "lossless-bitwise", "LL"),
    "layer0_qkv_table": ({(107, 8)}, "lossless-bitwise", "LL"),
    "edge_bf16_moe_front": ({(107, 8)}, "lossless-bitwise", "LL"),
    "last_layer_boundary": ({(107, 8)}, "lossless-bitwise", "LL"),
    "context_kv_fork_last_layer": ({(107, 8)}, "lossless-bitwise", "LL"),
    "distributed_head_sampler": ({(107, 8)}, "lossless-numeric", "LL"),
    "drafter_preamble_late_join": ({(107, 8)}, "lossless-bitwise", "LL"),
    "drafter_folds": ({(107, 8)}, "lossless-bitwise", "LL"),
    "draft_head_tail": ({(107, 8)}, "lossless-bitwise", "LL"),
    "step_loop_inline_commits": ({(107, 8)}, "lossless-bitwise", "LL"),
    "rejection_sampler": ({(107, 8)}, "composition", "LL"),
}


def test_the_rule_table_is_the_lever_registry():
    assert set(RULES) == set(SERVED_AT)
    assert set(RULES) <= DISPATCH.FAST_PATH_NAMES
    for name, (keys, kind, origin) in SERVED_AT.items():
        served = {key for key, paths in TARGET_PATHS.items() if name in paths.served}
        assert served == keys, name
        assert (RULES[name].kind.value, RULES[name].origin) == (kind, origin), name


# --------------------------------------------------------------------------
# The regions under auto
# --------------------------------------------------------------------------
LL = {"step_loop_on_execution_stream", "incremental_block_offsets", "head_sampler_coop_fold"}
#: The drafter-arena levers: their stores and commits are a dense full-length arena's or the KV
#: cache manager's pages (``dispatch.UNWINDOWED_ARENA_LEVERS``).
DENSE = {"drafter_kv_side_chain", "drafter_fc_early_fork", "drafter_preamble_from_commit"}
MIDC = {"wide_direct_transports", "wide_oproj_publish", "wide_finalize_transport"}
HC = {"packed_prep_append", "windowed_commit_parallel"}
PREFILL = {"prefill_cutedsl_experts"}
#: The Vera Rubin low-latency levers (origin LL, max_batch_size up to 8): both sm107 keys, the
#: TP4 ones, the TP8 ones; the context lever; and the vocabulary-parallel head's sm107 rule.
VR_LL = {
    "weight_prefetch_after_join",
    "next_qkv_prefetch_skip",
    "swa_one_cluster_block",
    "swa_prep_split",
    "swa_eight_requests",
    "full_attention_fold_multi",
    "split_kv_deep_rings",
    "prerouted_experts",
    "finalize_entry_trigger",
    "hop_entry_trigger",
    "step_commit_v2",
    "wide_norm_gate_pdl",
    "drafter_kv_glue",
    "drafter_kv_fork_before_head",
}
VR_LL_TEP4 = {"splitk_pool_split8", "splitk_frontier_tables", "bf16_half_window"}
VR_LL_TEP8 = {
    "tep8_wide64_splitk",
    "tep8_publish_splitk",
    "tep8_drafter_wide_gemms",
    "tep8_fc_shard_wide",
    "tep8_ring_buddy_wide",
    "tep8_ring_buddy_pdl",
    "tep8_tile_placement",
    "tep8_router_push",
    "tep8_router_prewait",
    "tep8_finalize_transport_64",
    "tep8_early_multicast_hop",
    "tep8_direct_hop",
    "tep8_publish_join",
    "tep8_pool_programmatic_fork",
    "tep8_wide_projection_gathers",
}
CONTEXT_SP = {"context_sp_extra_parts"}
VP = {"vocab_parallel_head"}
#: The low-latency set of an sm107 key at max_batch_size up to 8; over a dense arena the
#: two-launch side chain yields to ``drafter_kv_side_chain``.
VR_TEP4 = VR_LL | VR_LL_TEP4 | VP
VR_TEP8 = VR_LL | VR_LL_TEP8 | VP
VR_TEP8_DENSE = VR_TEP8 - {"drafter_kv_glue"}
#: The TP4 paths TEP8 serves too: rule levers (left out under ``baseline`` there only),
#: on wherever served.
TEP4_PATHS = set(LEVERS.SM107_TEP8_TEP4_PATHS)
#: The numeric low-latency levers: an unknown role passes none of their accuracy gates.
VR_NUMERIC = {name for name in VR_TEP4 | VR_TEP8 if RULES[name].kind.value == "lossless-numeric"}

#: The bitwise levers of 2..4-request decode steps: on aggregated sm107 TEP4 servers of
#: max_batch_size 2..8 (``levers.WIDE_AGG_BATCH``); the default band does not run them.
WIDE_STEPS = {
    "swa_blocks_per_program",
    "wide_qkv_prefetch",
    "wide_capture_fold",
}
#: ... and the full-attention prep fold those servers run on the same steps
#: (``full_attention_fold_multi``'s rule in the aggregated sm107 band).
AGG_WIDE = WIDE_STEPS | {"full_attention_fold_multi"}
#: The levers of aggregated sm107 TEP4 servers of max_batch_size ``HC_LEVERS_MIN_BATCH`` and
#: up (the wider drafter fc inbox, ``INBOX``, also needs decode steps of 129 rows; the layer
#: 0 / 61 tile pin, ``L061``, also needs ``hot_replicas``).
AGG_TEP4_CORE = {
    "step_window_compact",
    "fmk_context_fmha",
    "context_kv_late_join",
    "hop_cluster_split",
    "router_staged_epilogue",
    "split_kv_deep_ring",
    "context_prep_rows_vec",
    "context_embedding_replicated",
    "context_sp_router_input",
}
INBOX = {"drafter_fc_inbox_wide"}
AGG_TEP4 = AGG_TEP4_CORE | INBOX
L061 = {"l061_tile32_pin"}
#: The single-request decode schedule of aggregated sm107 TEP8 servers of max_batch_size 1 over a
#: dense arena, under auto: everything of ``dispatch.SM107_TEP8_ONLY`` but the boundary variant no
#: build has (``megakernel_zero_exchange_boundary``) and the draft head's tail, which waits for a
#: drafter that declares no logit transform (``SINGLE_TAIL``).
SINGLE = {
    "megakernel",
    "megakernel_one_exchange_boundaries",
    "megakernel_two_term_pool",
    "megakernel_router_by_linearity",
    "layer0_qkv_table",
    "edge_bf16_moe_front",
    "last_layer_boundary",
    "context_kv_fork_last_layer",
    "distributed_head_sampler",
    "drafter_preamble_late_join",
    "drafter_folds",
    "step_loop_inline_commits",
}
SINGLE_TAIL = {"draft_head_tail"}
#: The fused rejection sampler, on in that deployment only where the speculative config asks for
#: rejection sampling (forced acceptance leaves it inert), whatever the drafter's logit transform.
SINGLE_RS = {"rejection_sampler"}

#: The low-latency levers that stand down where one of ``WIDE_STEPS`` is on (by band on
#: aggregated sm107 TEP4 servers; elsewhere where a knob turns one on): the sliding-window
#: block's other builds of those steps (``exclusive``) and the dense prefetch and QKV tables
#: that ``wide_qkv_prefetch`` reads.
VR_YIELD_WIDE = {
    "swa_one_cluster_block",
    "swa_prep_split",
    "swa_eight_requests",
    "weight_prefetch_after_join",
    "splitk_frontier_tables",
}
#: The context-K/V levers that stand down where ``context_kv_late_join`` is on (aggregated
#: sm107 TEP4 servers of max_batch_size 4 and up).
VR_YIELD_LATE_JOIN = {"drafter_kv_glue", "drafter_kv_fork_before_head"}
#: The levers that stand down on the single-request decode schedule (``SINGLE``).
VR_YIELD_SINGLE = {
    "vocab_parallel_head",
    "step_commit_v2",
    "drafter_kv_fork_before_head",
    "weight_prefetch_after_join",
}
#: What the default band's sm107 TEP8 set leaves on beside the single-request schedule under
#: ``auto``: nothing, an aggregated sm107 server is the aggregated sm107 band's
#: (``levers.band_of``).
VR_SINGLE: set = set()

REGIONS = [
    pytest.param(
        (107, 8),
        "agg",
        (1,),
        "dense",
        LL | DENSE | VR_SINGLE | SINGLE,
        id="ll-agg-tep8-mbs1",
    ),
    pytest.param(
        (107, 4),
        "agg",
        (1,),
        "windowed",
        LL | {"half_window_experts", "wide_finalize_transport"} | PREFILL | TEP4_PATHS,
        id="ll-agg-tep4-mbs1",
    ),
    pytest.param((107, 8), "gen", (1,), "windowed", LL | VR_TEP8 | TEP4_PATHS, id="gen-tep8-b1"),
    pytest.param(
        (107, 4),
        "gen",
        (1,),
        "windowed",
        LL | {"half_window_experts", "wide_finalize_transport"} | PREFILL | VR_TEP4 | TEP4_PATHS,
        id="gen-tep4-b1",
    ),
    pytest.param(
        (107, 4),
        "gen",
        (2, 3, 4),
        "windowed",
        MIDC
        | {"wide_router_per_row"}
        | LL
        | {"half_window_experts"}
        | PREFILL
        | VR_TEP4
        | TEP4_PATHS,
        id="gen-tep4-b2-b4",
    ),
    pytest.param(
        (107, 4),
        "agg",
        (2, 3),
        "windowed",
        MIDC | LL | {"half_window_experts"} | PREFILL | TEP4_PATHS | AGG_WIDE,
        id="agg-caps-2-3",
    ),
    pytest.param(
        (107, 4),
        "agg",
        (4,),
        "windowed",
        MIDC | LL | {"half_window_experts"} | PREFILL | TEP4_PATHS | AGG_TEP4_CORE | AGG_WIDE,
        id="agg-cap-4",
    ),
    pytest.param(
        (107, 4),
        "agg",
        (5, 6, 8),
        "windowed",
        HC
        | MIDC
        | {"incremental_block_offsets", "head_sampler_coop_fold", "half_window_experts"}
        | PREFILL
        | TEP4_PATHS
        | AGG_TEP4_CORE
        | AGG_WIDE,
        id="agg-caps-5-8",
    ),
    pytest.param(
        (107, 4),
        "agg",
        (9, 15),
        "windowed",
        HC
        | MIDC
        | {"head_sampler_coop_fold", "hot_replicas"}
        | PREFILL
        | TEP4_PATHS
        | AGG_TEP4_CORE
        | L061,
        id="agg-mid-9-15",
    ),
    pytest.param(
        (107, 4),
        "agg",
        (16,),
        "windowed",
        HC
        | MIDC
        | {"head_sampler_coop_fold", "hot_replicas"}
        | PREFILL
        | TEP4_PATHS
        | AGG_TEP4_CORE
        | L061,
        id="agg-mid-16",
    ),
    pytest.param(
        (107, 4),
        "agg",
        (17, 32),
        "windowed",
        HC
        | MIDC
        | {"head_sampler_coop_fold", "hot_replicas"}
        | PREFILL
        | TEP4_PATHS
        | AGG_TEP4
        | L061,
        id="agg-mid-high",
    ),
    pytest.param(
        (107, 4),
        "agg",
        (48, 64),
        "windowed",
        HC | {"graph_start_qkv_prefetch", "hot_replicas"} | PREFILL | TEP4_PATHS | AGG_TEP4 | L061,
        id="agg-hc",
    ),
    pytest.param(
        (107, 4),
        "gen",
        (5, 8),
        "windowed",
        HC
        | MIDC
        | {"incremental_block_offsets", "head_sampler_coop_fold", "half_window_experts"}
        | PREFILL
        | VR_TEP4
        | TEP4_PATHS,
        id="gen-tep4-b5-b8",
    ),
    pytest.param(
        (107, 4),
        "gen",
        (16, 32, 64),
        "windowed",
        HC | MIDC | {"head_sampler_coop_fold", "hot_replicas"} | PREFILL | TEP4_PATHS,
        id="gen-tep4-b16-b64",
    ),
    pytest.param(
        (107, 8),
        "agg",
        (2, 4),
        "dense",
        LL | DENSE,
        id="agg-tep8-mbs2-4",
    ),
    pytest.param(
        (107, 8),
        "agg",
        (8,),
        "dense",
        {"incremental_block_offsets", "head_sampler_coop_fold"} | DENSE,
        id="agg-tep8-mbs8",
    ),
    pytest.param(
        (107, 4),
        "ctx",
        (8, 64),
        "windowed",
        {"wide_finalize_transport"} | PREFILL | CONTEXT_SP | TEP4_PATHS,
        id="ctx-tep4",
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


def test_the_vera_rubin_low_latency_deployments():
    # A TEP8 generation server of eight requests over a windowed arena.
    tep8 = table((107, 8), 8, "gen", "windowed")
    assert rule_levers_on(tep8) == (
        VR_TEP8
        | TEP4_PATHS
        | {"incremental_block_offsets", "head_sampler_coop_fold", "windowed_commit_parallel"}
    )
    # A TEP4 generation server of three requests; its context server.
    tep4 = table((107, 4), 3, "gen", "windowed")
    assert VR_TEP4 <= rule_levers_on(tep4)
    # The 2..4-request step levers are the aggregated sm107 band's: off here.
    assert not rule_levers_on(tep4) & WIDE_STEPS
    assert not rule_levers_on(tep4) & CONTEXT_SP
    ctx = table((107, 4), 64, "ctx", "windowed")
    assert rule_levers_on(ctx) & (VR_TEP4 | CONTEXT_SP) == CONTEXT_SP
    # Above eight requests every low-latency lever is off, on either key.
    for key in ((107, 4), (107, 8)):
        above = table(key, 9, "gen", "windowed")
        assert not rule_levers_on(above) & (VR_TEP4 | VR_TEP8 | CONTEXT_SP), key
        assert above.decisions["swa_prep_split"].why == "max_batch_size 9 is above 8"
    # An aggregated sm107 server runs none of them, at any max_batch_size (band); the
    # full-attention fold there is its 2..4-request step lever.
    for key in ((107, 4), (107, 8)):
        for mbs in (1, 4, 9):
            agg = table(key, mbs, "agg", "windowed")
            low_latency = (VR_TEP4 | VR_TEP8 | CONTEXT_SP) - AGG_WIDE
            assert not rule_levers_on(agg) & low_latency, (key, mbs)
            assert agg.decisions["swa_prep_split"].source == "band", (key, mbs)


def test_the_vocabulary_parallel_head_rule():
    for mbs in (1, 8):
        lever_table = table((107, 4), mbs, "gen", "windowed")
        assert lever_table.on("vocab_parallel_head")
        assert lever_table.param("vocab_parallel_head", "MIN_ROWS") == 8
    assert not table((107, 4), 9, "gen", "windowed").on("vocab_parallel_head")
    # A context server keeps the fabric memory: the built-in head samples its decode steps.
    assert not table((107, 4), 8, "ctx", "windowed").on("vocab_parallel_head")
    floor = table((107, 4), 4, "gen", "windowed", knobs(vocab_parallel_head__MIN_ROWS=16))
    assert floor.param("vocab_parallel_head", "MIN_ROWS") == 16


def test_the_two_launch_side_chain_yields_to_the_fused_one():
    dense = table((107, 8), 4, "gen", "dense")
    assert dense.on("drafter_kv_side_chain") and not dense.on("drafter_kv_glue")
    assert "yields to drafter_kv_side_chain" in dense.decisions["drafter_kv_glue"].why
    for key in ((107, 4), (107, 8)):
        assert table(key, 4, "gen", "windowed").on("drafter_kv_glue"), key
        assert not table(key, 4, "gen", "paged").on("drafter_kv_glue"), key
    with pytest.raises(ValueError, match="drafter_kv_glue"):
        table((107, 8), 4, "gen", "dense", knobs(drafter_kv_glue="on"))


def test_the_context_lever_runs_where_context_steps_run():
    for mbs in (8, 64, None):
        lever_table = table((107, 4), mbs, "ctx", "windowed")
        assert lever_table.on("context_sp_extra_parts"), mbs
        assert lever_table.param("context_sp_extra_parts", "PARTS") == ("router_input", "ce_gather")
    # An aggregated sm107 server runs its own band's context levers instead.
    for mbs in (8, 16):
        agg = table((107, 4), mbs, "agg", "windowed")
        assert not agg.on("context_sp_extra_parts"), mbs
        assert agg.decisions["context_sp_extra_parts"].source == "band", mbs
    assert not table((107, 4), 8, "gen", "windowed").on("context_sp_extra_parts")
    parts = table(
        (107, 4), 64, "ctx", "windowed", knobs(context_sp_extra_parts__PARTS="router_input")
    )
    assert parts.param("context_sp_extra_parts", "PARTS") == ("router_input",)
    off = table((107, 4), 64, "ctx", "windowed", knobs(context_sp_extra_parts="off"))
    assert not off.on("context_sp_extra_parts")


# --------------------------------------------------------------------------
# Band ownership: the low-latency levers beside the aggregated sm107 band's own levers
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mbs", [2, 3, 4])
def test_the_low_latency_levers_yield_to_the_two_to_four_request_steps(mbs):
    # A generation server runs the low-latency levers, not the 2..4-request step levers.
    gen = table((107, 4), mbs, "gen", "windowed")
    assert VR_YIELD_WIDE <= rule_levers_on(gen) and not WIDE_STEPS & rule_levers_on(gen)
    # The step levers turned on by their knobs take those steps: the low-latency ones yield.
    forced = table(
        (107, 4),
        mbs,
        "gen",
        "windowed",
        knobs(swa_blocks_per_program="on", wide_qkv_prefetch="on"),
    )
    for name in VR_YIELD_WIDE:
        decision = forced.decisions[name]
        assert not decision.on and decision.source == "auto", (mbs, name)
        assert "yields to" in decision.why, (mbs, name)
    # An aggregated server runs the step levers; the low-latency ones are off by band.
    agg = table((107, 4), mbs, "agg", "windowed")
    assert WIDE_STEPS <= rule_levers_on(agg), mbs
    for name in VR_YIELD_WIDE:
        assert agg.decisions[name].source == "band", (mbs, name)


def test_the_other_sliding_window_builds_cannot_join_the_split_programs():
    for name in ("swa_one_cluster_block", "swa_prep_split", "swa_eight_requests"):
        with pytest.raises(ValueError, match="cannot both be on"):
            table((107, 4), 3, "agg", "windowed", knobs(**{name: "on"}))
        lever_table = table(
            (107, 4), 3, "agg", "windowed", knobs(**{name: "on", "swa_blocks_per_program": "off"})
        )
        assert lever_table.on(name)


def test_the_context_kv_levers_yield_to_the_late_join():
    for mbs in (4, 8):
        lever_table = table((107, 4), mbs, "agg", "windowed")
        assert lever_table.on("context_kv_late_join")
        assert not rule_levers_on(lever_table) & VR_YIELD_LATE_JOIN, mbs
    # A generation server runs the low-latency context-K/V levers; the late join turned on
    # by its knob takes their steps.
    gen = table((107, 4), 4, "gen", "windowed")
    assert VR_YIELD_LATE_JOIN <= rule_levers_on(gen) and not gen.on("context_kv_late_join")
    forced = table((107, 4), 4, "gen", "windowed", knobs(context_kv_late_join="on"))
    assert forced.on("context_kv_late_join")
    for name in VR_YIELD_LATE_JOIN:
        assert "yields to context_kv_late_join" in forced.decisions[name].why, name


def test_the_low_latency_levers_yield_to_the_single_request_schedule():
    lever_table = table((107, 8), 1, "agg", "dense")
    for name in VR_YIELD_SINGLE:
        assert not lever_table.on(name), name
        assert lever_table.decisions[name].source == "band", name
    # The vocabulary-parallel head would take the distributed sampler's 8-row verify heads.
    with pytest.raises(ValueError, match="cannot both be on"):
        table((107, 8), 1, "agg", "dense", knobs(vocab_parallel_head="on"))
    alone = table(
        (107, 8), 1, "agg", "dense", knobs(distributed_head_sampler="off", vocab_parallel_head="on")
    )
    assert alone.on("vocab_parallel_head")
    # A generation server of one request runs the low-latency levers; the single-request
    # schedule's levers turned on by their knobs take their steps.
    gen = table((107, 8), 1, "gen", "dense")
    assert VR_YIELD_SINGLE <= rule_levers_on(gen)
    forced = table(
        (107, 8),
        1,
        "gen",
        "dense",
        knobs(
            distributed_head_sampler="on",
            step_loop_inline_commits="on",
            layer0_qkv_table="on",
            context_kv_fork_last_layer="on",
        ),
    )
    for name in VR_YIELD_SINGLE:
        assert not forced.on(name) and "yields to" in forced.decisions[name].why, name


class _SettingsPlan:
    """The two ``dispatch.Plan`` reads ``lever_settings.apply`` makes, over a lever table."""

    def __init__(self, lever_table) -> None:
        self.table = lever_table

    def __bool__(self) -> bool:
        return True

    def enabled(self, name: str) -> bool:
        return self.table.on(name)

    def param(self, name: str, key: str):
        return self.table.param(name, key)


def _settings(lever_table) -> dict:
    """``lever_settings.apply``'s ``configure`` calls for ``lever_table``, by module."""
    settings = importlib.import_module(f"{_PACKAGE}.lever_settings")
    calls = {}
    original = settings._configure
    settings._configure = lambda module, values, defaults: calls.__setitem__(
        module, (values, defaults)
    )
    try:
        settings.apply(_SettingsPlan(lever_table))
    finally:
        settings._configure = original
    return calls


def test_the_duplicate_rings_and_context_parts_share_one_configure():
    ring = "ops.attention.multi_request_split_kv"
    fa_ring = "ops.attention.full_attention_split_kv_kernel"
    parts = "kernels.context_sequence_parallel.forward"
    # The aggregated sm107 band, at any max_batch_size from 4: the 20-page multi-request ring
    # and the router input.
    for mbs in (48, 8):
        hc = _settings(table((107, 4), mbs, "agg", "windowed"))
        assert hc[ring][0] == {"sm107_stages": 20}, mbs
        assert hc[fa_ring][0] == {"stages": 16}, mbs
        assert hc[parts][0] == {"sm107_extra_parts": ("router_input",)}, mbs
    # The default band's levers: both rings on a generation server of 8 requests, the router
    # input and the CE gather on a context server.
    gen = _settings(table((107, 4), 8, "gen", "windowed"))
    assert gen[ring][0] == {"sm107_stages": 20} and gen[fa_ring][0] == {"stages": 20}
    ctx = _settings(table((107, 4), 64, "ctx", "windowed"))
    assert ctx[parts][0] == {"sm107_extra_parts": ("router_input", "ce_gather")}
    # Both bands' levers on together: one configure each, the union.
    low = _settings(
        table(
            (107, 4),
            8,
            "agg",
            "windowed",
            knobs(split_kv_deep_rings="on", context_sp_extra_parts="on"),
        )
    )
    assert low[ring][0] == {"sm107_stages": 20}
    assert low[fa_ring][0] == {"stages": 20}
    assert low[parts][0] == {"sm107_extra_parts": ("router_input", "ce_gather")}
    # Each alone.
    only_hc = _settings(table((107, 4), 8, "agg", "windowed", knobs(split_kv_deep_rings="off")))
    assert only_hc[ring][0] == {"sm107_stages": 20} and only_hc[fa_ring][0] == {"stages": 16}
    neither = _settings(
        table(
            (107, 4),
            48,
            "agg",
            "windowed",
            knobs(split_kv_deep_ring="off", context_sp_router_input="off"),
        )
    )
    assert neither[ring][0] == {"sm107_stages": None}
    assert neither[parts][0] == {"sm107_extra_parts": ()}


def test_the_single_request_set_is_the_sm107_tep8_set():
    assert SINGLE | SINGLE_TAIL | SINGLE_RS | {"megakernel_zero_exchange_boundary"} == set(
        DISPATCH.SM107_TEP8_ONLY
    )
    assert SINGLE | SINGLE_TAIL | SINGLE_RS <= TARGET_PATHS[(107, 8)].served
    assert not DISPATCH.SM107_TEP8_ONLY & TARGET_PATHS[(107, 4)].served


def test_agg_tep8_mbs1_dense():
    lever_table = table((107, 8), 1, "agg", "dense")
    assert rule_levers_on(lever_table) == LL | DENSE | SINGLE | VR_SINGLE
    assert base_off(lever_table) == set()


def test_agg_tep8_mbs1_paged():
    lever_table = table((107, 8), 1, "agg", "paged")
    assert rule_levers_on(lever_table) == LL | DENSE | SINGLE | VR_SINGLE
    assert rule_levers_on(rs_table(arena="paged")) == (
        LL | DENSE | SINGLE | VR_SINGLE | SINGLE_TAIL | SINGLE_RS
    )


@pytest.mark.parametrize(
    "transform, expected",
    [(None, set()), (True, set()), (False, SINGLE_TAIL)],
    ids=["unknown", "declared", "none"],
)
def test_the_draft_head_tail_waits_for_a_drafter_without_a_logit_transform(transform, expected):
    sm, tp = 107, 8
    lever_table = DISPATCH.table_for(
        TARGET_PATHS, sm, tp, 1, "agg", "dense", None, None, draft_logit_transform=transform
    )
    assert rule_levers_on(lever_table) == LL | DENSE | SINGLE | VR_SINGLE | expected
    if not expected:
        with pytest.raises(ValueError, match="logit transform"):
            DISPATCH.table_for(
                TARGET_PATHS,
                sm,
                tp,
                1,
                "agg",
                "dense",
                knobs(draft_head_tail="on"),
                None,
                draft_logit_transform=transform,
            )


@pytest.mark.parametrize(
    "key, mbs, role, arena",
    [
        ((107, 8), 2, "agg", "dense"),
        ((107, 8), 1, "gen", "dense"),
        ((107, 8), 1, "agg", "windowed"),
        ((107, 4), 1, "agg", "windowed"),
    ],
)
def test_the_single_request_set_stays_off_outside_its_deployment(key, mbs, role, arena):
    lever_table = table(key, mbs, role, arena, facts=None)
    assert not rule_levers_on(lever_table) & (SINGLE | SINGLE_TAIL | SINGLE_RS)
    if key in TARGET_PATHS:
        rs_table = DISPATCH.table_for(
            TARGET_PATHS,
            *key,
            mbs,
            role,
            arena,
            draft_logit_transform=False,
            rejection_sampling=True,
        )
        assert not rule_levers_on(rs_table) & (SINGLE | SINGLE_TAIL | SINGLE_RS)


def rs_table(key=(107, 8), mbs=1, role="agg", arena="dense", knob_set=None, **facts):
    """``table_for`` with the drafter-transform and rejection-sampling facts (default: no
    transform, rejection sampling asked for)."""
    facts = {"draft_logit_transform": False, "rejection_sampling": True, **facts}
    return DISPATCH.table_for(TARGET_PATHS, *key, mbs, role, arena, knob_set, None, **facts)


@pytest.mark.parametrize(
    "transform, rejection_sampling, expected",
    [
        (False, True, SINGLE_TAIL | SINGLE_RS),
        (False, False, SINGLE_TAIL),
        (False, None, SINGLE_TAIL),
        (True, True, SINGLE_RS),
        (None, True, SINGLE_RS),
        (True, False, set()),
    ],
    ids=[
        "asked",
        "not-asked-or-forced",
        "unknown",
        "declared-transform",
        "unknown-transform",
        "both-off",
    ],
)
def test_the_rejection_sampler_needs_rejection_sampling(transform, rejection_sampling, expected):
    lever_table = rs_table(draft_logit_transform=transform, rejection_sampling=rejection_sampling)
    assert rule_levers_on(lever_table) == LL | DENSE | SINGLE | VR_SINGLE | expected
    decision = lever_table.decisions["rejection_sampler"]
    assert decision.source == ("auto" if rejection_sampling is True else "structural")


def test_the_rejection_sampler_is_the_only_difference_rejection_sampling_makes():
    for key in sorted(TARGET_PATHS):
        for mbs in (1, 2, 8, 48):
            for role in ("agg", "gen", "ctx"):
                for arena in ("dense", "windowed", "paged"):
                    for transform in (None, True, False):
                        tables = [
                            DISPATCH.table_for(
                                TARGET_PATHS,
                                *key,
                                mbs,
                                role,
                                arena,
                                draft_logit_transform=transform,
                                rejection_sampling=rejection_sampling,
                            )
                            for rejection_sampling in (None, False, True)
                        ]
                        on_sets = [rule_levers_on(t) - SINGLE_RS for t in tables]
                        assert on_sets[0] == on_sets[1] == on_sets[2], (
                            key,
                            mbs,
                            role,
                            arena,
                            transform,
                        )
                        assert not rule_levers_on(tables[0]) & SINGLE_RS
                        assert not rule_levers_on(tables[1]) & SINGLE_RS


def test_the_rejection_sampler_follows_its_knob():
    off = rs_table(knob_set=knobs(rejection_sampler="off"))
    assert rule_levers_on(off) == LL | DENSE | SINGLE | VR_SINGLE | SINGLE_TAIL
    assert off.decisions["rejection_sampler"].source == "knob"
    off_tail = rs_table(knob_set=knobs(draft_head_tail="off"))
    assert rule_levers_on(off_tail) == LL | DENSE | SINGLE | VR_SINGLE | SINGLE_RS
    assert off_tail.decisions["rejection_sampler"].source == "auto"
    off_folds = rs_table(knob_set=knobs(drafter_folds="off"))
    assert off_folds.decisions["rejection_sampler"].source == "dependency"
    with pytest.raises(ValueError, match="builds on distributed_head_sampler, which is off"):
        rs_table(mbs=2, knob_set=knobs(rejection_sampler="on"))
    baseline = rs_table(knob_set={**BASELINE})
    assert "rejection_sampler" not in rule_levers_on(baseline)


@pytest.mark.parametrize(
    "facts, message",
    [
        ({"rejection_sampling": False}, "forced acceptance"),
        ({"rejection_sampling": None}, "is not known"),
    ],
    ids=["forced-or-not-asked", "unknown"],
)
def test_the_rejection_sampler_on_raises_where_it_cannot_be_on(facts, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        rs_table(knob_set=knobs(rejection_sampler="on"), **facts)


def test_the_rejection_sampler_beside_the_single_request_yields():
    lever_table = rs_table(draft_logit_transform=True)
    assert lever_table.on("rejection_sampler")
    assert not rule_levers_on(lever_table) & VR_YIELD_SINGLE
    # The vocabulary-parallel head conflicts with the distributed sampler RS builds on.
    with pytest.raises(ValueError, match="cannot both be on"):
        rs_table(knob_set=knobs(vocab_parallel_head="on"))
    vocab_parallel = rs_table(
        knob_set=knobs(vocab_parallel_head="on", distributed_head_sampler="off")
    )
    assert vocab_parallel.on("vocab_parallel_head")
    assert vocab_parallel.decisions["rejection_sampler"].source == "dependency"
    # The commit launch and the context K/V fork point are other launches: turned on by their
    # knobs (the default band's levers), RS stays on beside them.
    for name, knob_set in (
        ("step_commit_v2", knobs(step_commit_v2="on")),
        ("step_commit_v2", knobs(step_loop_inline_commits="off", step_commit_v2="on")),
        ("drafter_kv_fork_before_head", knobs(drafter_kv_fork_before_head="on")),
        (
            "drafter_kv_fork_before_head",
            knobs(context_kv_fork_last_layer="off", drafter_kv_fork_before_head="on"),
        ),
    ):
        beside = rs_table(knob_set=knob_set)
        assert beside.on(name) and beside.on("rejection_sampler"), knob_set


def test_the_deployment_reports_rejection_sampling():
    deployment = rs_table().deployment
    assert deployment.rejection_sampling is True
    assert deployment.as_dict()["rejection_sampling"] is True
    assert deployment.describe().endswith("drafter logit transform none, rejection sampling on")
    unknown = table((107, 8), 1, "agg", "dense").deployment
    assert "rejection_sampling" not in unknown.as_dict()
    assert "rejection sampling" not in unknown.describe()


@pytest.mark.parametrize(
    "spec, forced, expected",
    [
        (None, "off", None),
        (types.SimpleNamespace(use_rejection_sampling=False), "inert", False),
        (types.SimpleNamespace(use_rejection_sampling=True), "off", True),
        (types.SimpleNamespace(use_rejection_sampling=True), "inert", False),
        (types.SimpleNamespace(), "off", False),
    ],
    ids=["no-spec-config", "not-asked", "asked", "asked-forced", "no-flag"],
)
def test_rejection_sampling_of_a_model_config(monkeypatch, spec, forced, expected):
    rs_common = types.ModuleType(f"{_PACKAGE}.kernels.rs_sampler.rs_common")
    rs_common.forced_mode = lambda: forced
    rs_sampler = types.ModuleType(f"{_PACKAGE}.kernels.rs_sampler")
    rs_sampler.__path__ = []
    rs_sampler.rs_common = rs_common
    kernels = types.ModuleType(f"{_PACKAGE}.kernels")
    kernels.__path__ = []
    kernels.rs_sampler = rs_sampler
    for module in (kernels, rs_sampler, rs_common):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    model_config = types.SimpleNamespace(spec_config=spec)
    assert DISPATCH.rejection_sampling_of(model_config) is expected


def test_the_numerics_levers_are_switchable_and_the_megakernel_needs_them():
    """Opting out of one numerics lever runs the stock layers (no other build exists)."""
    nb, bh, rl = (
        "megakernel_one_exchange_boundaries",
        "megakernel_two_term_pool",
        "megakernel_router_by_linearity",
    )
    # bh builds on nb, so opting out of nb takes bh with it.
    for name, off in ((nb, {nb, bh}), (bh, {bh}), (rl, {rl})):
        lever_table = table((107, 8), 1, "agg", "dense", knobs(**{name: "off"}))
        assert rule_levers_on(lever_table) == (
            LL | DENSE | VR_SINGLE | SINGLE - {"megakernel"} - off
        ), name
    with pytest.raises(ValueError):
        table((107, 8), 1, "agg", "dense", knobs(megakernel_zero_exchange_boundary="on"))


def test_the_router_per_row_gates():
    on = table((107, 4), 4, "gen", "windowed").decisions["wide_router_per_row"]
    assert on.on and on.source == "auto" and "gate" in on.why
    off = table((107, 4), 8, "agg", "windowed").decisions["wide_router_per_row"]
    assert not off.on and off.source == "gate"
    for mbs in (2, 3, 4):
        off = table((107, 4), mbs, "agg", "windowed").decisions["wide_router_per_row"]
        assert not off.on and off.source == "gate", mbs
    assert not table((107, 4), 1, "agg", "windowed").on("wide_router_per_row")


@pytest.mark.parametrize(
    "role, sizes, on",
    [
        ("gen", (None, 1, 2, 3, 4, 5, 8, 16, 48), False),
        ("agg", (2, 3, 4, 5, 6, 7, 8), True),
        ("agg", (None, 1, 9, 16, 47, HC_MIN_BATCH, 64), False),
        ("ctx", (1, 4, 8, 64), False),
        ("unknown", (2, 4, 8), False),
    ],
)
def test_the_two_to_four_request_step_levers_under_auto(role, sizes, on):
    for mbs in sizes:
        lever_table = table((107, 4), mbs, role, "windowed")
        for name in WIDE_STEPS:
            decision = lever_table.decisions[name]
            assert decision.on is on, (name, mbs)
            # wide_capture_fold builds on wide_direct_transports, off where no decode step
            # reaches 16 rows and on the high-concurrency selection; the other roles are
            # the default band's.
            first = "auto" if role == "agg" else "band"
            sources = {first} if on else {first, "dependency"}
            assert decision.source in sources, (name, mbs, decision.source)
            assert decision.source == first or name == "wide_capture_fold", (name, mbs)
            assert RULES[name].kind.value == "lossless-bitwise", name
    assert LEVERS.WIDE_AGG_BATCH == (2, 8)


def test_the_two_to_four_request_step_levers_are_tep4_only():
    for name in WIDE_STEPS:
        assert name not in TARGET_PATHS[(107, 8)].served, name
        with pytest.raises(ValueError, match=re.escape(f"does not serve {name}")):
            table((107, 8), 2, "gen", "windowed", knobs(**{name: "on"}))


def test_the_two_to_four_request_step_levers_follow_their_knobs_and_paths():
    # on outside the measured band, off inside it, and off with the path each builds on.
    outside = table((107, 4), 8, "gen", "windowed", knobs(wide_qkv_prefetch="on"))
    assert outside.on("wide_qkv_prefetch")
    assert outside.decisions["wide_qkv_prefetch"].source == "knob"
    inside = table((107, 4), 2, "agg", "windowed", knobs(swa_blocks_per_program="off"))
    assert not inside.on("swa_blocks_per_program")
    assert WIDE_STEPS - {"swa_blocks_per_program"} <= rule_levers_on(inside)
    for base, lever in (
        ("wide_swa_block", "swa_blocks_per_program"),
        ("next_qkv_prefetch", "wide_qkv_prefetch"),
        ("wide_dense_gemms", "wide_qkv_prefetch"),
        ("capture_fold", "wide_capture_fold"),
        ("wide_direct_transports", "wide_capture_fold"),
    ):
        lever_table = table((107, 4), 3, "agg", "windowed", knobs(**{base: "off"}))
        assert lever_table.decisions[lever].source == "dependency", (base, lever)
        assert not lever_table.on(lever), (base, lever)
    baseline = table((107, 4), 3, "agg", "windowed", BASELINE)
    assert not WIDE_STEPS & rule_levers_on(baseline)


def test_the_retired_two_to_four_request_step_switches_name_their_levers():
    for old, name in (
        ("TRTLLM_MUSESPARK_SWA_BLOCKS_PER_PROGRAM", "swa_blocks_per_program"),
        ("TRTLLM_MUSESPARK_TEP_WIDE_FULL_FOLD", "full_attention_fold_multi"),
        ("TRTLLM_MUSESPARK_TEP_WIDE_QKV_PREFETCH", "wide_qkv_prefetch"),
        ("TRTLLM_MUSESPARK_TEP_WIDE_CAPTURE_FOLD", "wide_capture_fold"),
    ):
        assert LEVERS.retired_replacement(old) == LEVERS.knob_name(name), old
        for value in ("0", "1", "auto"):
            with pytest.raises(ValueError, match=re.escape(f"{old} is retired")):
                table((107, 4), 2, "gen", "windowed", {old: value})


def test_the_lossy_and_flagged_levers_are_never_on_under_auto():
    for key, paths in TARGET_PATHS.items():
        for mbs in (1, 4, 8, 64):
            lever_table = table(key, mbs, "agg", "dense")
            for name, rule in RULES.items():
                if rule.kind.value in ("lossy", "flagged") and name in paths.served:
                    assert not lever_table.on(name), (key, mbs, name)


# --------------------------------------------------------------------------
# The replay knob sets and the ladders
# --------------------------------------------------------------------------
#: What the table gained after the replayed measurements (the Vera Rubin low-latency levers,
#: the vocabulary-parallel head on sm107 and the TP4 paths sm107 TEP8 serves): a replay of
#: one of those states turns it off.
VR_OFF_TEP8 = knobs(**{name: "off" for name in VR_TEP8 | TEP4_PATHS})
VR_OFF_TEP4 = knobs(**{name: "off" for name in VR_TEP4 | CONTEXT_SP})
R_LL = {**knobs(head_multimem_gather="on"), **VR_OFF_TEP8}
#: The high-concurrency selection of the wide and fold levers, by knobs.
R_H_CORE = knobs(
    wide_direct_transports="off",
    wide_oproj_publish="off",
    wide_finalize_transport="off",
    head_sampler_coop_fold="off",
    graph_start_qkv_prefetch="on",
    windowed_commit_parallel__MIN_REQUESTS=1,
    context_fp8_dense="on",
    context_fp8_attn_gather="on",
)
#: ``R_H_CORE`` with the aggregated TP4 levers off: the selection before them.
R_H = {**R_H_CORE, **knobs(**{name: "off" for name in sorted(AGG_TEP4 | L061)})}
B_LL = {
    **knobs(
        step_loop_on_execution_stream="off",
        incremental_block_offsets="off",
        head_sampler_coop_fold="off",
        graph_start_qkv_prefetch="on",
    ),
    **VR_OFF_TEP4,
}


def test_r_ll_selects_the_low_latency_set():
    lever_table = table((107, 8), 1, "agg", "dense", R_LL)
    assert rule_levers_on(lever_table) == LL | DENSE | SINGLE | {"head_multimem_gather"}
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
        | TEP4_PATHS
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
        | TEP4_PATHS
    )


def test_baseline_tep8_gen_mbs1_is_the_previous_generation_selection():
    lever_table = table((107, 8), 1, "gen", "windowed", BASELINE)
    assert rule_levers_on(lever_table) == {"lm_head_inbox", "graph_start_qkv_prefetch"}


R_B1 = dict(B_LL, **knobs(wide_finalize_transport="off"))
_LL_REPLAY = LL | DENSE | SINGLE | {"head_multimem_gather"}


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


def test_auto_at_the_high_concurrency_selection_is_r_h_core():
    opt_ins = knobs(context_fp8_dense="on", context_fp8_attn_gather="on")
    for mbs in (HC_MIN_BATCH, 64):
        auto = table((107, 4), mbs, "agg", "windowed", opt_ins).decisions
        replay = table((107, 4), mbs, "agg", "windowed", R_H_CORE).decisions
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
    # Every other difference is a lever of one band's selection, on in its own band only
    # (the full-attention fold serves both bands, with each band's rule).
    banded = {n for n in agg if LEVERS.lever_band(n) is not None} | {"full_attention_fold_multi"}
    differ = {n for n in agg if agg[n] != gen[n]}
    assert differ - banded == gated
    assert all(not agg[n][0] for n in gated)
    for name in differ & banded - {"full_attention_fold_multi"}:
        mine = LEVERS.lever_band(name) is LEVERS.Band.AGGREGATED_SM107
        assert (agg[name][0], gen[name][0]) == (mine, not mine), name


# --------------------------------------------------------------------------
# baseline: the composition before the lever table
# --------------------------------------------------------------------------
#: The served sets of the compositions at c55183f445 (its ``modeling.py`` tables over its
#: ``dispatch.FAST_PATH_NAMES``).
C55_SERVED = {
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
    one of these names there, so each rule removes the name alone. Then the base paths added
    since, each beside the path it builds on: ``draft_head_sampled`` (``draft_head``),
    ``rejection_fold`` (``verify_fold``), ``paged_context_commit``,
    ``paged_ctx_block_tables`` and ``paged_ctx_kv_store`` (``context_kv_fork``)."""
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
    for added, base in (
        ("draft_head_sampled", "draft_head"),
        ("rejection_fold", "verify_fold"),
        ("paged_context_commit", "context_kv_fork"),
        ("paged_ctx_block_tables", "context_kv_fork"),
        ("paged_ctx_kv_store", "context_kv_fork"),
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
    "MUSESPARK_OPEN_SPLITK_MAILBOX_V4",
    "MUSESPARK_OPEN_SPLITK_OVERSIZED_SMEM",
    "MUSESPARK_OPEN_SPLITK_RS",
    "MUSESPARK_OPEN_SPLITK_RS_ONLY",
    "MUSESPARK_OPEN_POOL_SPLITK8",
    "MUSESPARK_OPEN_TEP4_WIDE_GEMMS",
    "MUSESPARK_OPEN_TEP4_NARROW_GEMMS",
    "MUSESPARK_OPEN_TEP8_WIDE64_SPLITK",
    "MUSESPARK_OPEN_TEP8_WIDE64_KERNELS",
    "MUSESPARK_OPEN_TEP8_PUBLISH_SPLITK",
    "MUSESPARK_OPEN_TEP8_DRAFTER_WIDE_GEMMS",
    "MUSESPARK_OPEN_TEP8_FC_SHARD_WIDE",
    "MUSESPARK_OPEN_TEP8_FC_GATHER_PROGRAMS",
    "MUSESPARK_OPEN_PREFETCH_JOIN",
    "MUSESPARK_OPEN_DROP_WEIGHT_PREFETCH",
    "MUSESPARK_OPEN_NEXT_QKV_PREFETCH_SKIP_ROWS",
    "MUSESPARK_OPEN_SWA_CL8",
    "MUSESPARK_OPEN_TEP8_SWA_CL8",
    "MUSESPARK_OPEN_SWA_PREP_SPLIT",
    "MUSESPARK_OPEN_SWA_ONE2",
    "MUSESPARK_OPEN_FA_PREP_FOLD_MULTI",
    "MUSESPARK_OPEN_FA_STAGES",
    "MUSESPARK_OPEN_MR_STAGES",
    "MUSESPARK_OPEN_BF16_HALF_WINDOW",
    "MUSESPARK_OPEN_PREROUTED_META",
    "MUSESPARK_OPEN_TEP8_PREROUTED",
    "MUSESPARK_OPEN_PREROUTED_SO",
    "MUSESPARK_OPEN_ROUTER_ROWS_TILING",
    "MUSESPARK_OPEN_TEP8_RING_BUDDY_MAX_ROWS",
    "MUSESPARK_OPEN_TEP8_RB_PDL",
    "MUSESPARK_OPEN_TEP8_TILE_PLACEMENT",
    "MUSESPARK_OPEN_TEP8_TILE_PLACEMENT_MIN_ROWS",
    "MUSESPARK_OPEN_TEP8_ROUTER_PUSH",
    "MUSESPARK_OPEN_TEP8_ROUTER_PUSH_MIN_ROWS",
    "MUSESPARK_OPEN_TEP8_ROUTER_PUSH_ROUTE",
    "MUSESPARK_OPEN_TEP8_ROUTER_PREWAIT",
    "MUSESPARK_OPEN_TEP8_POOL_PROGRAMMATIC_FORK",
    "MUSESPARK_OPEN_TEP8_WIDE_POOL_GATHER",
    "MUSESPARK_OPEN_TEP8_WIDE_UNPOOL_GATHER",
    "MUSESPARK_OPEN_FINALIZE_TRIGGER",
    "MUSESPARK_OPEN_TEP8_FINALIZE_TRANSPORT_TOKENS",
    "MUSESPARK_OPEN_HOP_ENTRY_TRIGGER",
    "MUSESPARK_OPEN_TEP8_HOP_DIRECT",
    "MUSESPARK_OPEN_TEP8_HOP_EARLY",
    "MUSESPARK_OPEN_TEP8_HOP_MULTICAST",
    "MUSESPARK_OPEN_TEP8_HOP_PAD_ROWS",
    "MUSESPARK_OPEN_TEP8_HOP_R2",
    "MUSESPARK_OPEN_TEP8_HOP_R2_MIN_ROWS",
    "MUSESPARK_OPEN_TEP8_HOP_SMEM_PAD",
    "MUSESPARK_OPEN_TEP8_JOIN_PDL",
    "MUSESPARK_OPEN_PUBLISH_JOIN_CLUSTER",
    "MUSESPARK_OPEN_NORM_GATE_PDL_MAX_ROWS",
    "MUSESPARK_OPEN_STEP_COMMIT_V2",
    "MUSESPARK_OPEN_CTX_KV_GLUE",
    "MUSESPARK_OPEN_CTX_KV_FORK_EARLY",
    "MUSESPARK_OPEN_VP_MIN_ROWS",
    "MUSESPARK_OPEN_SM107_VOCAB_PARALLEL_HEAD",
    "MUSESPARK_OPEN_TEP8_EXTRA_PATHS",
    "MUSESPARK_OPEN_DROP_PATHS",
    "MUSESPARK_OPEN_FC_SHARD_PREWARM",
    "MUSESPARK_OPEN_SP_SM107_PARTS",
    "MUSESPARK_OPEN_SM107_FMK_CONTEXT_FMHA",
    "MUSESPARK_OPEN_CTX_KV_LATE_JOIN",
    "MUSESPARK_OPEN_HOP_CLUSTER",
    "MUSESPARK_OPEN_HOP_CLUSTER_ROWS",
    "MUSESPARK_OPEN_FC_INBOX_MAX_ROWS",
    "MUSESPARK_OPEN_FC_INBOX_PROGRAMS",
    "MUSESPARK_OPEN_L061_TILE32_PIN",
    "MUSESPARK_OPEN_ROUTER_STAGED_EPILOGUE",
    "MUSESPARK_OPEN_DECODE_QKV_MXFP8",
    "MUSESPARK_OPEN_CTX_SP_SM107_ROUTER_INPUT",
    "MUSESPARK_OPEN_CTX_KV_FORK_POINT",
    "MUSESPARK_OPEN_HOP_PAYLOAD",
    "MUSESPARK_OPEN_FINALIZE_PAYLOAD",
    "MUSESPARK_OPEN_ROUTER_WEIGHT_PROLOGUE",
    "MUSESPARK_OPEN_POOL_FORK_AFTER_ROUTER",
    "MUSESPARK_OPEN_SM107_UNPOOL_TGV",
    "MUSESPARK_OPEN_DECODE_ROUTE_RECORD_DIR",
    "MUSESPARK_OPEN_MEGAKERNEL",
    "MUSESPARK_OPEN_MEGAKERNEL_NB",
    "MUSESPARK_OPEN_MEGAKERNEL_BH",
    "MUSESPARK_OPEN_MEGAKERNEL_RL",
    "MUSESPARK_OPEN_MEGAKERNEL_FZ",
    "MUSESPARK_OPEN_TQ_TARGET_L0",
    "MUSESPARK_OPEN_PF_FUSED_PRE",
    "MUSESPARK_OPEN_OB_PRE_VARIANT",
    "MUSESPARK_OPEN_PV_L0_INPUT_MK",
    "MUSESPARK_OPEN_PF_BF16_FRONT",
    "MUSESPARK_OPEN_OB_BF16_DIRECT",
    "MUSESPARK_OPEN_OB_BF16_MOE_PDL",
    "MUSESPARK_OPEN_G2_L61_BOUNDARY",
    "MUSESPARK_OPEN_PV_GLUE_FUSED",
    "MUSESPARK_OPEN_PF_SIDE_FORK",
    "MUSESPARK_OPEN_PV_EARLY_FC_AFTER_HOP1",
    "MUSESPARK_OPEN_S2_SAMPLER",
    "MUSESPARK_OPEN_PV_LOGITS_ONESHOT",
    "MUSESPARK_OPEN_PV_LOGITS_POLL",
    "MUSESPARK_OPEN_PT_VERIFY_PDL",
    "MUSESPARK_OPEN_OX_DEJ",
    "MUSESPARK_OPEN_DR3_CONV_FOLD",
    "MUSESPARK_OPEN_DR3_SILU_FOLD",
    "MUSESPARK_OPEN_SP_CF_V2",
    "MUSESPARK_OPEN_DG_SF_TACTIC",
    "MUSESPARK_OPEN_OX_DE",
    "MUSESPARK_OPEN_PT_ATTN_PDL",
    "MUSESPARK_OPEN_TT_DRAFTER_L0",
    "MUSESPARK_OPEN_DR_DRAFT_TAIL",
    "MUSESPARK_OPEN_GR_INLINE_COMMIT",
    "MUSESPARK_OPEN_IA_FOLD",
    "MUSESPARK_OPEN_RS_FUSED",
    "MUSESPARK_OPEN_RS_FORCED",
)


def test_retired_is_the_spec_list():
    assert set(LEVERS.RETIRED) == set(SPEC_RETIRED)


@pytest.mark.parametrize(
    "name",
    sorted(SPEC_RETIRED)
    + [
        "MUSESPARK_OPEN_PS_ANYTHING_ELSE",
        "MUSESPARK_OPEN_MEGAKERNEL_ANYTHING_ELSE",
        "MUSESPARK_OPEN_RS_Q_TRANSFORM",
    ],
)
def test_a_retired_switch_raises_naming_its_replacement(name):
    replacement = LEVERS.retired_replacement(name)
    assert replacement
    with pytest.raises(ValueError, match=re.escape(f"{name} is retired (use {replacement})")):
        table((107, 4), 4, "agg", "windowed", {name: "0"})


@pytest.mark.parametrize(
    "key, mbs, arena, snapshot, message",
    [
        (
            (107, 8),
            1,
            "windowed",
            knobs(step_window_compact="on"),
            "does not serve step_window_compact",
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


def test_the_aggregated_tep4_levers_and_their_parameters():
    hc = table((107, 4), 48, "agg", "windowed")
    assert hc.param("hop_cluster_split", "CLUSTER") == 2
    assert hc.param("hop_cluster_split", "ROWS") == ((192, 208), (248, 264))
    assert hc.param("drafter_fc_inbox_wide", "MAX_ROWS") == 384
    assert hc.param("drafter_fc_inbox_wide", "PROGRAMS") == 128
    assert "gate: aggregated sm107 tep4 servers" in hc.decisions["fmk_context_fmha"].why
    moved = table(
        (107, 4),
        48,
        "agg",
        "windowed",
        knobs(hop_cluster_split__ROWS=" 264, 200-216 ", hop_cluster_split__CLUSTER=1),
    )
    assert moved.param("hop_cluster_split", "ROWS") == ((200, 216), (264, 264))
    assert moved.param("hop_cluster_split", "CLUSTER") == 1
    for value, message in (("8-16", "outside 9..1024"), ("300-200", "descending"), (",", "lo-hi")):
        with pytest.raises(ValueError, match=message):
            table((107, 4), 48, "agg", "windowed", knobs(hop_cluster_split__ROWS=value))
    with pytest.raises(ValueError, match="1..2"):
        table((107, 4), 48, "agg", "windowed", knobs(hop_cluster_split__CLUSTER=4))
    assert '"ROWS":[[192,208],[248,264]]' in hc.as_json()


def test_the_aggregated_tep4_floor_and_the_inbox_reach():
    assert LEVERS.HC_LEVERS_MIN_BATCH == 4
    for mbs in (1, 2, 3):
        low = table((107, 4), mbs, "agg", "windowed")
        for name in AGG_TEP4:
            assert not low.on(name), (mbs, name)
        assert "max_batch_size 4 and up" in low.decisions["router_staged_epilogue"].why
        assert low.decisions["fmk_context_fmha"].source == "auto"
    unknown = table((107, 4), None, "agg", "windowed")
    assert not unknown.on("router_staged_epilogue")
    capped = table((107, 4), 16, "agg", "windowed")
    assert capped.on("router_staged_epilogue")
    assert not capped.on("drafter_fc_inbox_wide")
    assert "129 rows" in capped.decisions["drafter_fc_inbox_wide"].why
    assert table((107, 4), 17, "agg", "windowed").on("drafter_fc_inbox_wide")


def test_decode_qkv_mxfp8_needs_the_context_fp8_qkv_copies():
    off = table((107, 4), 48, "agg", "windowed", knobs(decode_qkv_mxfp8="auto"))
    assert not off.on("decode_qkv_mxfp8")
    assert off.decisions["decode_qkv_mxfp8"].source == "dependency"
    with pytest.raises(ValueError, match="DECODE_QKV_MXFP8=on"):
        table((107, 4), 48, "agg", "windowed", knobs(decode_qkv_mxfp8="on"))
    on = table((107, 4), 48, "agg", "windowed", {**knobs(decode_qkv_mxfp8="on"), **R_H_CORE})
    assert on.on("decode_qkv_mxfp8") and on.decisions["decode_qkv_mxfp8"].source == "knob"
    no_qkv = {
        **knobs(decode_qkv_mxfp8="on", context_fp8_dense="on"),
        **knobs(context_fp8_dense__PROJECTIONS="o_proj,pool,unpool"),
    }
    with pytest.raises(ValueError, match="leaves out"):
        table((107, 4), 48, "agg", "windowed", no_qkv)


def test_an_aggregated_tep4_lever_on_a_generation_server_only_by_its_knob():
    gen = table((107, 4), 48, "gen", "windowed")
    assert not gen.on("router_staged_epilogue")
    assert gen.decisions["router_staged_epilogue"].source == "band"
    forced = table((107, 4), 48, "gen", "windowed", knobs(fmk_context_fmha="on"))
    assert forced.on("fmk_context_fmha")
    assert any("FMK_CONTEXT_FMHA=on outside its accuracy gates" in w for w in forced.warnings)


def test_on_outside_a_numeric_gate_warns():
    lever_table = table((107, 4), 8, "agg", "windowed", knobs(wide_router_per_row="on"))
    assert lever_table.on("wide_router_per_row")
    assert any("outside its accuracy gates" in warning for warning in lever_table.warnings)


def test_a_knob_of_an_unserved_lever_set_off_or_auto_only_warns():
    lever_table = table((107, 8), 1, "agg", "dense", knobs(packed_prep_append="off"))
    assert any("has no effect" in warning for warning in lever_table.warnings)
    assert lever_table.paths() == table((107, 8), 1, "agg", "dense").paths()


@pytest.mark.parametrize(
    "name", ["decode_kv_splits", "drafter_window_compact", "wide_projection_gathers"]
)
@pytest.mark.parametrize("value", ["on", "off", "auto"])
def test_a_knob_naming_a_lever_this_target_does_not_have_raises(name, value):
    with pytest.raises(ValueError, match="no lever of this target is named"):
        table((107, 4), 4, "agg", "windowed", {LEVERS.PREFIX + name.upper(): value})


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
    assert lever_table.decisions["windowed_commit_parallel"].source == "structural"
    for name in DENSE | {"drafter_preamble_late_join"}:
        assert lever_table.on(name), name
    assert lever_table.on("step_loop_on_execution_stream")
    assert lever_table.on("context_kv_fork")
    # The drafter's pages are in the target's manager, whose tables the stager copies.
    assert lever_table.on("incremental_block_offsets")


def test_the_window_compaction_needs_a_windowed_arena_and_the_full_context_paths_need_it():
    assert table((107, 4), 4, "agg", "windowed").on("step_window_compact")
    dense = table((107, 4), 4, "agg", "dense")
    assert dense.decisions["step_window_compact"].source == "structural"
    # The context-K/V fork's windowed commit compacts its own decode steps: the fork and the
    # verify fold run over a windowed arena.
    assert table((107, 8), 1, "agg", "windowed").on("context_kv_fork")
    assert table((107, 4), 1, "agg", "windowed").on("verify_fold")


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
    # Counted as a decoding server of the default band; no accuracy gate names an unknown
    # role, and the context lever runs on servers that may take context steps.
    gen = rule_levers_on(table((107, 4), 4, "gen", "windowed"))
    assert rule_levers_on(unknown) == (gen - {"wide_router_per_row"} - VR_NUMERIC) | CONTEXT_SP
    assert unknown.decisions["wide_router_per_row"].source == "gate"
    for name in WIDE_STEPS:
        decision = unknown.decisions[name]
        assert decision.source == "band" and "default band" in decision.why, name


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
