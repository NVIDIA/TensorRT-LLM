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
* the context-step reduce-scatter overlaps: off unless their knobs set them, their parameters,
  what the tail pipeline needs and the MXFP8 projections they stand aside for;
* ``compose`` agreed over a stubbed TP group, and the ``Plan`` reports.
"""

from __future__ import annotations

import ast
import enum
import hashlib
import importlib
import importlib.util
import json
import random
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
    "packed_prep_append": ({(107, 4), (107, 8)}, "lossless-bitwise", "HC"),
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
    "context_sp_oproj_overlap": ({(107, 4)}, "lossless-bitwise", "HC"),
    "context_sp_tail_pipeline": ({(107, 4)}, "lossless-bitwise", "HC"),
    "context_sp_moe_rs_overlap": ({(107, 4)}, "lossless-bitwise", "HC"),
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
#: The levers of a context step alone, in no band's selection: on for aggregated sm107 TEP4 servers of
#: ``HC_LEVERS_MIN_BATCH`` and up (``AGG_TEP4_CORE``) and on for the TEP4 ``ctx`` role by rule.
CTX_STEP_LEVERS = {
    "context_prep_rows_vec",
    "context_embedding_replicated",
    "context_sp_router_input",
}
#: The ctx role's defaults (int1): the multi-instance FMHA on pure context steps and the
#: context-step reduce-scatter overlaps 'ovts' (o_proj overlap + tail pipeline).
CTX_ROLE_DEFAULTS = {"fmk_context_fmha", "context_sp_oproj_overlap", "context_sp_tail_pipeline"}
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
        {"wide_finalize_transport"}
        | CTX_STEP_LEVERS
        | CTX_ROLE_DEFAULTS
        | PREFILL
        | CONTEXT_SP
        | TEP4_PATHS,
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
    # Above eight requests every low-latency lever is off on TEP4; a TEP8 generation server
    # runs them up to sixteen requests (MAX_BATCH 16 by default there, the 'wall' set), and
    # none above.
    above = table((107, 4), 9, "gen", "windowed")
    assert not rule_levers_on(above) & (VR_TEP4 | VR_TEP8 | CONTEXT_SP)
    assert above.decisions["swa_prep_split"].why == "max_batch_size 9 is above 8"
    for mbs in (9, 12, 16):
        wall = table((107, 8), mbs, "gen", "windowed")
        # The two-launch side chain was not in the gated set: it stays at 8 requests.
        # ... and samples on the split sampler (int2 X400 composite: the vocabulary-parallel head off).
        assert rule_levers_on(wall) & (VR_TEP4 | VR_TEP8 | CONTEXT_SP) == VR_TEP8 - {
            "drafter_kv_glue",
            "vocab_parallel_head",
        }
        assert wall.param("swa_prep_split", "MAX_BATCH") == 16, mbs
        assert wall.param("drafter_kv_glue", "MAX_BATCH") == 8, mbs
    above = table((107, 8), 17, "gen", "windowed")
    assert not rule_levers_on(above) & (VR_TEP4 | VR_TEP8 | CONTEXT_SP)
    assert above.decisions["swa_prep_split"].why == "max_batch_size 17 is above 8"
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
        # A TEP4 context server adds the shard routing too (the 'ovts' set, int1 default) and
        # leaves the copy engines to the KV transfers: no CE gather (int4 default).
        assert lever_table.param("context_sp_extra_parts", "PARTS") == (
            "shard_routing",
            "router_input",
        )
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
    # The default band's levers: both rings on a generation server of 8 requests, the shard
    # routing and the router input on a context server (int1 default; no CE gather since int4).
    gen = _settings(table((107, 4), 8, "gen", "windowed"))
    assert gen[ring][0] == {"sm107_stages": 20} and gen[fa_ring][0] == {"stages": 20}
    ctx = _settings(table((107, 4), 64, "ctx", "windowed"))
    assert ctx[parts][0] == {"sm107_extra_parts": ("shard_routing", "router_input")}
    # The CE gather is still one knob away on a context server.
    ce = _settings(
        table(
            (107, 4),
            64,
            "ctx",
            "windowed",
            knobs(context_sp_extra_parts__PARTS="shard_routing,router_input,ce_gather"),
        )
    )
    assert ce[parts][0] == {"sm107_extra_parts": ("shard_routing", "router_input", "ce_gather")}
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


#: The context-step reduce-scatter overlaps (``kernels/context_sequence_parallel/overlap``).
CTX_OVERLAPS = ("context_sp_oproj_overlap", "context_sp_tail_pipeline", "context_sp_moe_rs_overlap")
CTX_OVERLAP_MODULE = "kernels.context_sequence_parallel.overlap"
CTX_OVERLAP_OFF = {
    "oproj_chunks": 0,
    "oproj_min_rows": 8192,
    "oproj_high_priority": True,
    "tail_pipeline": False,
    "moe_rs_chunks": 0,
    "moe_rs_min_rows": 8192,
    "moe_rs_high_priority": True,
}


@pytest.mark.parametrize(
    "role, sizes",
    [("agg", (1, 4, 16, 48)), ("gen", (1, 8, 64))],
)
def test_the_context_sp_overlaps_are_off_unless_their_knobs_set_them(role, sizes):
    for mbs in sizes:
        lever_table = table((107, 4), mbs, role, "windowed")
        for name in CTX_OVERLAPS:
            decision = lever_table.decisions[name]
            assert not decision.on, (role, mbs, name)
            # The tail pipeline is off with the projection overlap it builds on.
            if name == "context_sp_tail_pipeline":
                assert decision.source == "dependency", (role, mbs)
            else:
                assert decision.source == "auto", (role, mbs, name)
                assert "not yet measured on sm107" in decision.why, (role, mbs, name)
        settings = _settings(lever_table)
        assert settings[CTX_OVERLAP_MODULE] == (CTX_OVERLAP_OFF, CTX_OVERLAP_OFF), (role, mbs)
        baseline = table((107, 4), mbs, role, "windowed", BASELINE)
        assert not any(baseline.on(name) for name in CTX_OVERLAPS), (role, mbs)
    # Not served at TEP8: its context steps are not sequence-parallel.
    assert not set(CTX_OVERLAPS) & TARGET_PATHS[(107, 8)].served
    with pytest.raises(ValueError, match="does not serve context_sp_oproj_overlap"):
        table((107, 8), 8, "gen", "windowed", knobs(context_sp_oproj_overlap="on"))


def test_a_context_server_runs_the_projection_overlap_and_the_tail_pipeline_by_default():
    # int1: 'ovts' (o_proj overlap + tail pipeline + shard routing) is the TEP4 ctx role's default;
    # the experts' overlap stays off until measured.
    for mbs in (8, 32, 64, None):
        lever_table = table((107, 4), mbs, "ctx", "windowed")
        for name in ("context_sp_oproj_overlap", "context_sp_tail_pipeline"):
            decision = lever_table.decisions[name]
            assert decision.on and decision.source == "auto", (mbs, name)
            assert "context servers" in decision.why, (mbs, name)
        moe = lever_table.decisions["context_sp_moe_rs_overlap"]
        assert not moe.on and "not yet measured on sm107" in moe.why, mbs
        assert _settings(lever_table)[CTX_OVERLAP_MODULE][0] == {
            **CTX_OVERLAP_OFF,
            "oproj_chunks": 2,
            "tail_pipeline": True,
        }, mbs
        # Their knobs still turn them off; baseline leaves them out.
        off = table((107, 4), mbs, "ctx", "windowed", knobs(context_sp_oproj_overlap="off"))
        assert not off.on("context_sp_oproj_overlap") and not off.on("context_sp_tail_pipeline")
        assert off.decisions["context_sp_tail_pipeline"].source == "dependency", mbs
        tail_off = table((107, 4), mbs, "ctx", "windowed", knobs(context_sp_tail_pipeline="off"))
        assert tail_off.on("context_sp_oproj_overlap") and not tail_off.on("context_sp_tail_pipeline")
        baseline = table((107, 4), mbs, "ctx", "windowed", BASELINE)
        assert not any(baseline.on(name) for name in CTX_OVERLAPS), mbs
    # Without the router-input join the tail pipeline stands aside under auto (no error).
    no_join = table(
        (107, 4),
        64,
        "ctx",
        "windowed",
        knobs(context_sp_router_input="off", context_sp_extra_parts="off"),
    )
    assert no_join.on("context_sp_oproj_overlap") and not no_join.on("context_sp_tail_pipeline")
    assert no_join.decisions["context_sp_tail_pipeline"].source == "structural"


def test_the_context_sp_overlaps_follow_their_knobs_and_parameters():
    on = knobs(
        context_sp_oproj_overlap="on",
        context_sp_tail_pipeline="on",
        context_sp_moe_rs_overlap="on",
    )
    for role, mbs in (("ctx", 64), ("agg", 16), ("agg", 48)):
        lever_table = table((107, 4), mbs, role, "windowed", on)
        assert all(lever_table.on(name) for name in CTX_OVERLAPS), (role, mbs)
        assert all(lever_table.decisions[name].source == "knob" for name in CTX_OVERLAPS)
        assert _settings(lever_table)[CTX_OVERLAP_MODULE][0] == {
            "oproj_chunks": 2,
            "oproj_min_rows": 8192,
            "oproj_high_priority": True,
            "tail_pipeline": True,
            "moe_rs_chunks": 2,
            "moe_rs_min_rows": 8192,
            "moe_rs_high_priority": True,
        }, (role, mbs)
    tuned = table(
        (107, 4),
        64,
        "ctx",
        "windowed",
        knobs(
            context_sp_oproj_overlap="on",
            context_sp_oproj_overlap__CHUNKS="3",
            context_sp_oproj_overlap__MIN_ROWS="4096",
            context_sp_oproj_overlap__HIGH_PRIORITY="0",
            context_sp_moe_rs_overlap="on",
            context_sp_moe_rs_overlap__CHUNKS="4",
            context_sp_tail_pipeline="off",
        ),
    )
    assert tuned.param("context_sp_oproj_overlap", "CHUNKS") == 3
    assert not tuned.on("context_sp_tail_pipeline")
    assert _settings(tuned)[CTX_OVERLAP_MODULE][0] == {
        "oproj_chunks": 3,
        "oproj_min_rows": 4096,
        "oproj_high_priority": False,
        "tail_pipeline": False,
        "moe_rs_chunks": 4,
        "moe_rs_min_rows": 8192,
        "moe_rs_high_priority": True,
    }
    # The experts' overlap alone.
    moe_only = table(
        (107, 4),
        64,
        "ctx",
        "windowed",
        knobs(context_sp_moe_rs_overlap="on", context_sp_oproj_overlap="off"),
    )
    assert _settings(moe_only)[CTX_OVERLAP_MODULE][0] == {**CTX_OVERLAP_OFF, "moe_rs_chunks": 2}
    for spec, message in (
        (dict(context_sp_oproj_overlap__CHUNKS="5"), r"1\.\.4"),
        (dict(context_sp_moe_rs_overlap__CHUNKS="0"), r"1\.\.4"),
        (dict(context_sp_oproj_overlap__MIN_ROWS="512"), "1024"),
        (dict(context_sp_moe_rs_overlap__HIGH_PRIORITY="2"), r"0\.\.1"),
    ):
        with pytest.raises(ValueError, match=message):
            table((107, 4), 64, "ctx", "windowed", knobs(**spec))


def test_the_tail_pipeline_needs_the_projection_overlap_and_the_router_input_join():
    # It builds on the projection overlap.
    with pytest.raises(ValueError, match="builds on context_sp_oproj_overlap, which is off"):
        table((107, 4), 16, "agg", "windowed", knobs(context_sp_tail_pipeline="on"))
    both = knobs(context_sp_oproj_overlap="on", context_sp_tail_pipeline="on")
    # The router-input join: context_sp_router_input, or context_sp_extra_parts' router_input part.
    for knob_set in (
        both,
        {**both, **knobs(context_sp_router_input="off")},
        {**both, **knobs(context_sp_extra_parts="off")},
    ):
        assert table((107, 4), 64, "ctx", "windowed", knob_set).on("context_sp_tail_pipeline")
    for knob_set in (
        {**both, **knobs(context_sp_router_input="off", context_sp_extra_parts="off")},
        {
            **both,
            **knobs(context_sp_router_input="off", context_sp_extra_parts__PARTS="ce_gather"),
        },
    ):
        with pytest.raises(ValueError, match="router-input join"):
            table((107, 4), 64, "ctx", "windowed", knob_set)
    # An aggregated sm107 TEP4 server joins the router input by rule.
    assert table((107, 4), 16, "agg", "windowed", both).on("context_sp_tail_pipeline")
    # The projection overlap off by its knob: the tail pipeline cannot be on without it.
    with pytest.raises(ValueError, match="builds on context_sp_oproj_overlap"):
        table(
            (107, 4),
            64,
            "ctx",
            "windowed",
            knobs(context_sp_oproj_overlap="off", context_sp_tail_pipeline="on"),
        )


@pytest.mark.parametrize(
    "projections, refused",
    [
        ("qkv,o_proj,pool,unpool", set(CTX_OVERLAPS)),
        ("qkv", set()),
        ("o_proj", {"context_sp_oproj_overlap"}),
        ("pool", {"context_sp_tail_pipeline"}),
        ("unpool", {"context_sp_moe_rs_overlap"}),
    ],
)
def test_the_context_sp_overlaps_stand_aside_for_the_fp8_projections_they_chunk(
    projections, refused
):
    fp8 = knobs(context_fp8_dense="on", context_fp8_dense__PROJECTIONS=projections)
    on = knobs(context_sp_oproj_overlap="on", context_sp_tail_pipeline="on")
    for name in CTX_OVERLAPS:
        knob_set = {**fp8, **knobs(**{name: "on"})}
        if name == "context_sp_tail_pipeline":
            knob_set = {**fp8, **on}
        if name in refused or (
            name == "context_sp_tail_pipeline" and "context_sp_oproj_overlap" in refused
        ):
            with pytest.raises(ValueError, match="context_fp8_dense picks the"):
                table((107, 4), 64, "ctx", "windowed", knob_set)
        else:
            assert table((107, 4), 64, "ctx", "windowed", knob_set).on(name), (projections, name)


def test_the_built_in_overlap_threshold_is_a_layer_attribute():
    """``context_sp_oproj_overlap__MIN_ROWS`` moves the built-in layer's threshold, whose default is the
    switch's documented 8192 rows."""
    builtin = (_ROOT / "tensorrt_llm/_torch/models/modeling_musespark_open.py").read_text()
    assert "_overlap_sp_projection_min_rows = 8192" in builtin
    assert "sp_num_tokens >= self._overlap_sp_projection_min_rows" in builtin
    assert "sp_num_tokens >= 8192" not in builtin
    overlap = (_TARGET / "kernels/context_sequence_parallel/overlap.py").read_text()
    for attribute in (
        "_overlap_sp_projection_enabled",
        "_overlap_sp_projection_graph",
        "_overlap_sp_projection_min_rows",
        "_sp_projection_chunks",
        "_sp_projection_stream",
    ):
        assert f".{attribute} = " in overlap, attribute
        assert attribute in builtin, attribute


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
    # The TEP8 generation role's own parameter default (int1: the router push from 9 rows) is
    # a role default, not a band difference (test_the_disaggregated_role_defaults).
    role_params = {("tep8_router_push", "MIN_ROWS"), ("swa_one_cluster_block", "TEP8_MIN_REQUESTS")}

    def selection(role):
        lever_table = table(key, mbs, role, "windowed", arm)
        return {
            n: (d.on, {k: v for k, v in d.params.items() if (n, k) not in role_params})
            for n, d in lever_table.decisions.items()
        }

    agg, gen = selection("agg"), selection("gen")
    # The router-per-row accuracy gate holds on generation servers only.
    gated = {"wide_router_per_row"} if gen.get("wide_router_per_row", (False,))[0] else set()
    # Every other difference is a lever of one band's selection, on in its own band only
    # (the full-attention fold serves both bands, with each band's rule).
    banded = {n for n in agg if LEVERS.lever_band(n) is not None} | {"full_attention_fold_multi"}
    # Both bands' by rule: on for aggregated servers and the ctx role.
    by_rule = CTX_STEP_LEVERS | {"fmk_context_fmha"}
    banded |= by_rule
    differ = {n for n in agg if agg[n] != gen[n]}
    assert differ - banded == gated
    assert all(not agg[n][0] for n in gated)
    for name in differ & banded - {"full_attention_fold_multi"} - by_rule:
        mine = LEVERS.lever_band(name) is LEVERS.Band.AGGREGATED_SM107
        assert (agg[name][0], gen[name][0]) == (mine, not mine), name


@pytest.mark.parametrize("mbs", [1, 4, 8, 64])
def test_the_context_step_levers_run_on_the_tep4_context_role(mbs):
    ctx = table((107, 4), mbs, "ctx", "windowed")
    # A context server runs the context steps whatever its max_batch_size: all three are on.
    assert all(ctx.on(name) for name in CTX_STEP_LEVERS), mbs
    # ... as a rule of both bands, not of the aggregated band's selection.
    assert all(LEVERS.lever_band(name) is None for name in CTX_STEP_LEVERS)
    # A generation server runs no context step: off. TEP8 does not serve them: off.
    gen = table((107, 4), mbs, "gen", "windowed")
    assert not any(gen.on(name) for name in CTX_STEP_LEVERS), mbs
    assert not any(table((107, 8), mbs, "ctx", "windowed").on(n) for n in CTX_STEP_LEVERS), mbs
    # The aggregated selection is as before: on from HC_MIN_BATCH.
    for agg_mbs in (LEVERS.HC_LEVERS_MIN_BATCH, 64):
        agg = table((107, 4), agg_mbs, "agg", "windowed")
        assert all(agg.on(name) for name in CTX_STEP_LEVERS), agg_mbs
    below = table((107, 4), LEVERS.HC_LEVERS_MIN_BATCH - 1, "agg", "windowed")
    assert not any(below.on(name) for name in CTX_STEP_LEVERS)


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
        (knobs(drafter_fc_early_fork__ROWS=8), "declares no parameter"),
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
    "MUSESPARK_OPEN_TEP8_SWA_CL8_MIN_REQUESTS",
    "MUSESPARK_OPEN_SWA_PREP_SPLIT",
    "MUSESPARK_OPEN_SWA_ONE2",
    "MUSESPARK_OPEN_FA_PREP_FOLD_MULTI",
    "MUSESPARK_OPEN_FA_STAGES",
    "MUSESPARK_OPEN_MR_STAGES",
    "MUSESPARK_OPEN_BF16_HALF_WINDOW",
    "MUSESPARK_OPEN_PREROUTED_META",
    "MUSESPARK_OPEN_TEP8_PREROUTED",
    "MUSESPARK_OPEN_TEP8_PREROUTED_MAX_ROWS",
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
    "MUSESPARK_OPEN_TEP8_HOP_DIRECT_ROWS",
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


def test_the_tep8_row_floors_are_lever_parameters():
    """The router push, tile placement and ring-buddy row limits are parameters (they were constants),
    with the old values as defaults, so a deployment moves them by knob alone. A TEP8 generation
    server pushes the router from 9 rows by default (int1: rp9); every other deployment from 33."""
    base = table((107, 8), 8, "gen", "paged")
    assert base.on("tep8_router_push") and base.on("tep8_tile_placement") and base.on("tep8_ring_buddy_wide")
    assert base.param("tep8_router_push", "MIN_ROWS") == LEVERS.TEP8_GEN_ROUTER_PUSH_MIN_ROWS == 9
    assert base.param("tep8_tile_placement", "MIN_ROWS") == 41
    assert base.param("tep8_ring_buddy_wide", "MAX_ROWS") == 64
    agg = table((107, 8), 8, "agg", "paged", knobs(tep8_router_push="on"))
    assert agg.param("tep8_router_push", "MIN_ROWS") == 33
    moved = table(
        (107, 8),
        8,
        "gen",
        "paged",
        knobs(
            tep8_router_push__MIN_ROWS=33,
            tep8_tile_placement__MIN_ROWS=33,
            tep8_ring_buddy_wide__MAX_ROWS=48,
        ),
    )
    assert moved.param("tep8_router_push", "MIN_ROWS") == 33
    assert moved.param("tep8_tile_placement", "MIN_ROWS") == 33
    assert moved.param("tep8_ring_buddy_wide", "MAX_ROWS") == 48
    assert moved.paths() == base.paths()  # a parameter moves a band; it turns no lever on or off
    for knob in ("tep8_router_push__MIN_ROWS", "tep8_tile_placement__MIN_ROWS"):
        for bad in (8, 65):
            with pytest.raises(ValueError, match="MIN_ROWS"):
                table((107, 8), 8, "gen", "paged", knobs(**{knob: bad}))


def test_the_next_qkv_prefetch_skip_band_is_a_parameter():
    """The largest step that forks no next-QKV prefetch was the constant 8 rows; it is MAX_ROWS (default 8)."""
    base = table((107, 8), 8, "gen", "paged")
    assert base.on("next_qkv_prefetch_skip")
    assert base.param("next_qkv_prefetch_skip", "MAX_ROWS") == 8
    wide = table((107, 8), 8, "gen", "paged", knobs(next_qkv_prefetch_skip__MAX_ROWS=32))
    assert wide.param("next_qkv_prefetch_skip", "MAX_ROWS") == 32
    assert wide.paths() == base.paths()
    with pytest.raises(ValueError, match="MAX_ROWS"):
        table((107, 8), 8, "gen", "paged", knobs(next_qkv_prefetch_skip__MAX_ROWS=7))


#: The 65..128-row band extensions of the TEP8 low-latency levers (decode steps of 9..16 requests
#: at K = 7): each a parameter whose default is the band the lever served before.
WIDE128 = knobs(
    tep8_ring_buddy_wide__MAX_ROWS=128,
    prerouted_experts__TEP8_MAX_ROWS=128,
    tep8_finalize_transport_64__TOKENS=128,
    tep8_direct_hop__MAX_ROWS=128,
    tep8_publish_splitk__MAX_ROWS=128,
    tep8_fc_shard_wide__MAX_ROWS=128,
)
#: Each WIDE128 parameter, its default (the old band) and its value under WIDE128.
WIDE128_PARAMS = {
    ("tep8_ring_buddy_wide", "MAX_ROWS"): (64, 128),
    ("prerouted_experts", "TEP8_MAX_ROWS"): (64, 128),
    ("tep8_finalize_transport_64", "TOKENS"): (64, 128),
    ("tep8_direct_hop", "MAX_ROWS"): (64, 128),
    ("tep8_publish_splitk", "MAX_ROWS"): (64, 128),
    ("tep8_fc_shard_wide", "MAX_ROWS"): (64, 128),
}


def _ll_on(mbs: int, names) -> dict:
    """``names`` on by knob, each lever's MAX_BATCH (where it declares one) at ``mbs``: the
    low-latency set of a generation server above eight requests."""
    out = {}
    for name in sorted(names):
        out.update(knobs(**{name: "on"}))
        if any(param.key == "MAX_BATCH" for param in RULES[name].params):
            out.update(knobs(**{f"{name}__MAX_BATCH": mbs}))
    return out


def test_the_wide128_parameters_default_to_the_served_bands():
    """Every 65..128-row extension is a parameter whose default is the band served before up to
    eight requests, so the tables, the paths and every ``configure`` call of a server of
    max_batch_size 8 are unchanged unless a knob is set (the 9..16-request defaults of a TEP8
    generation server: test_the_disaggregated_role_defaults)."""
    base = table((107, 8), 8, "gen", "windowed")
    for (name, key), (default, _) in WIDE128_PARAMS.items():
        assert base.on(name), name
        assert base.param(name, key) == default, (name, key)
    assert base.param("tep8_tile_placement", "MAX_ROWS") == 64
    assert base.param("tep8_direct_hop", "PRODUCER") == "grid"
    assert base.param("tep8_direct_hop", "PRODUCER_STAGES") == 0
    calls = _settings(base)
    assert calls["ops.moe.finalize_allreduce_transport"][0]["tep8_max_tokens"] == 64
    ring = calls["kernels.replicated_experts.ring_buddy"][0]
    assert ring["tep8_max_rows"] == 64 and ring["tep8_prerouted_max_rows"] == 64
    assert calls["ops.moe.ring_buddy_assign"][0]["tile_max_rows"] == 64
    hop = calls["ops.allreduce.post_attn_hop_wide"][0]
    assert hop["direct"] and hop["direct_max_rows"] == 64
    assert (hop["direct_producer"], hop["direct_producer_stages"]) == ("grid", 0)
    assert calls["ops.comm.projection_publish"][0]["tep8_publish_splitk_max_rows"] == 64
    assert calls["spec_worker"][0]["fc_shard_wide_rows"] == 64
    assert base.param("swa_one_cluster_block", "TEP8_MAX_REQUESTS") == 8
    assert calls["ops.attention.swa_block"][0]["cl8_tep8_max_requests"] == 8
    # int2: a TEP8 generation server runs CL8 from 2 requests (TEP8_GEN_CL8_MIN_REQUESTS).
    assert base.param("swa_one_cluster_block", "TEP8_MIN_REQUESTS") == 2
    assert calls["ops.attention.swa_block"][0]["cl8_tep8_min_requests"] == 2
    # Off (a server above sixteen requests under auto): every new setting is its module default.
    off = _settings(table((107, 8), 17, "gen", "windowed"))
    for module, key in (
        ("ops.moe.finalize_allreduce_transport", "tep8_max_tokens"),
        ("kernels.replicated_experts.ring_buddy", "tep8_prerouted_max_rows"),
        ("kernels.replicated_experts.ring_buddy", "tep8_max_rows"),
        ("ops.moe.ring_buddy_assign", "tile_max_rows"),
        ("ops.allreduce.post_attn_hop_wide", "direct_max_rows"),
        ("ops.allreduce.post_attn_hop_wide", "direct_producer"),
        ("ops.allreduce.post_attn_hop_wide", "direct_producer_stages"),
        ("ops.comm.projection_publish", "tep8_publish_splitk_max_rows"),
        ("spec_worker", "fc_shard_wide_rows"),
        ("ops.attention.swa_block", "cl8_tep8_max_requests"),
        ("ops.attention.swa_block", "cl8_tep8_min_requests"),
    ):
        values, defaults = off[module]
        assert values[key] == defaults[key], (module, key)


def test_the_wide128_knobs_move_the_tep8_bands_to_128_rows():
    """At max_batch_size 16 with the low-latency levers on by knob (MAX_BATCH 16), WIDE128 moves the
    ring-buddy placement, the prerouted tables, the finalize transport, the direct hop, the split-K
    publication and the sharded drafter fc to 128 rows; it turns no lever on or off. On an
    aggregated server (the rulers), whose defaults stay at the 1..8-request bands."""
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "agg", "windowed", ll16)
    assert VR_TEP8 <= rule_levers_on(plain)
    for (name, key), (default, _) in WIDE128_PARAMS.items():
        assert plain.param(name, key) == default, (name, key)
    # Inside the TEP8 accuracy gates up to max_batch_size 16 (the 'wall' gate): no warning;
    # above, on by knob outside them: a warning, no error.
    assert not any("outside its accuracy gates" in warning for warning in plain.warnings)
    above = table((107, 8), 24, "agg", "windowed", _ll_on(24, VR_TEP8))
    assert any("outside its accuracy gates" in warning for warning in above.warnings)
    wide = table((107, 8), 16, "agg", "windowed", {**ll16, **WIDE128})
    assert wide.paths() == plain.paths()
    for (name, key), (_, value) in WIDE128_PARAMS.items():
        assert wide.param(name, key) == value, (name, key)
    calls = _settings(wide)
    assert calls["ops.moe.finalize_allreduce_transport"][0]["tep8_max_tokens"] == 128
    ring = calls["kernels.replicated_experts.ring_buddy"][0]
    assert ring["tep8_max_rows"] == 128 and ring["tep8_prerouted_max_rows"] == 128
    # The tile placement keeps its measured band unless its own MAX_ROWS moves.
    assert calls["ops.moe.ring_buddy_assign"][0]["tile_max_rows"] == 64
    hop = calls["ops.allreduce.post_attn_hop_wide"][0]
    assert hop["direct_max_rows"] == 128 and hop["direct_producer"] == "grid"
    assert calls["ops.comm.projection_publish"][0]["tep8_publish_splitk_max_rows"] == 128
    assert calls["spec_worker"][0]["fc_shard_wide_rows"] == 128
    tuned = table(
        (107, 8),
        16,
        "agg",
        "windowed",
        {
            **ll16,
            **WIDE128,
            **knobs(
                tep8_tile_placement__MAX_ROWS=128,
                tep8_direct_hop__PRODUCER="chunks",
                tep8_direct_hop__PRODUCER_STAGES=5,
                swa_one_cluster_block__TEP8_MAX_REQUESTS=13,
            ),
        },
    )
    calls = _settings(tuned)
    assert calls["ops.attention.swa_block"][0]["cl8_tep8_max_requests"] == 13
    assert calls["ops.moe.ring_buddy_assign"][0]["tile_max_rows"] == 128
    hop = calls["ops.allreduce.post_attn_hop_wide"][0]
    assert (hop["direct_producer"], hop["direct_producer_stages"]) == ("chunks", 5)
    assert '"PRODUCER":"chunks"' in tuned.as_json()
    # A parameter on a lever that is off changes nothing: its module keeps the default.
    off = _settings(
        table((107, 8), 16, "agg", "windowed", {**ll16, **WIDE128, **knobs(tep8_direct_hop="off")})
    )
    assert off["ops.allreduce.post_attn_hop_wide"][0]["direct_max_rows"] == 64


@pytest.mark.parametrize(
    "knob, value, message",
    [
        ("tep8_ring_buddy_wide__MAX_ROWS", 129, "expected an integer in 9..128"),
        ("prerouted_experts__TEP8_MAX_ROWS", 63, "expected an integer in 64..128"),
        ("prerouted_experts__TEP8_MAX_ROWS", 129, "expected an integer in 64..128"),
        ("tep8_tile_placement__MAX_ROWS", 129, "expected an integer in 9..128"),
        ("tep8_finalize_transport_64__TOKENS", 96, r"expected one of \[16, 32, 64, 128\]"),
        ("tep8_finalize_transport_64__TOKENS", "x", r"expected one of \[16, 32, 64, 128\]"),
        ("tep8_direct_hop__MAX_ROWS", 96, r"expected one of \[64, 128\]"),
        ("tep8_direct_hop__PRODUCER", "fast", "expected one of"),
        ("tep8_direct_hop__PRODUCER", "grid,chunks", "expected one of"),
        ("tep8_direct_hop__PRODUCER_STAGES", 13, "expected an integer in 0..12"),
        ("tep8_publish_splitk__MAX_ROWS", 63, "expected an integer in 64..128"),
        ("tep8_publish_splitk__MAX_ROWS", 129, "expected an integer in 64..128"),
        ("tep8_fc_shard_wide__MAX_ROWS", 7, "expected an integer in 8..128"),
        ("tep8_fc_shard_wide__MAX_ROWS", 129, "expected an integer in 8..128"),
        ("swa_one_cluster_block__TEP8_MAX_REQUESTS", 7, "expected an integer in 8..16"),
        ("swa_one_cluster_block__TEP8_MAX_REQUESTS", 17, "expected an integer in 8..16"),
        ("swa_one_cluster_block__TEP8_MIN_REQUESTS", 1, "expected an integer in 2..5"),
        ("swa_one_cluster_block__TEP8_MIN_REQUESTS", 6, "expected an integer in 2..5"),
        ("swa_one_cluster_block__TEP8_MIN_REQUESTS", "x", "expected an integer in 2..5"),
    ],
)
def test_the_wide128_parameters_refuse_values_outside_their_kernels(knob, value, message):
    with pytest.raises(ValueError, match=message):
        table((107, 8), 8, "gen", "windowed", knobs(**{knob: value}))


# --------------------------------------------------------------------------
# The 2..4-request bands of the 5..8-request TEP8 paths (decode steps of 9..32 rows):
# tep8_publish_splitk__MIN_ROWS, tep8_direct_hop__MIN_ROWS, tep8_wide64_splitk__MIN_ROWS
# --------------------------------------------------------------------------
#: Each MIN_ROWS parameter: (lever, module, configure keyword); default 33, range 9..33.
LOW_BANDS = {
    "tep8_publish_splitk": ("ops.comm.projection_publish", "tep8_publish_splitk_min_rows"),
    "tep8_direct_hop": ("ops.allreduce.post_attn_hop_wide", "direct_min_rows"),
    "tep8_wide64_splitk": ("kernels.dense_gemms.dispatcher", "tep8_wide64_min_rows"),
}
#: The knob set that moves bs3-4 (24 and 32 rows at K = 7) onto the 33..64-row family.
LOW17 = knobs(
    tep8_publish_splitk__MIN_ROWS=17,
    tep8_direct_hop__MIN_ROWS=17,
    tep8_wide64_splitk__MIN_ROWS=17,
)
_PREFETCH_MODULE = "kernels.next_qkv_prefetch.prefetch"


def test_the_low_band_parameters_default_to_the_served_bands():
    """MIN_ROWS of the publication, the direct hop and the wide64 split-K is 33 (the band served
    before) on every TEP8 generation table that serves them; every configure call keeps its
    module default then, and off (a server above eight requests under auto) too."""
    for mbs, arena in ((8, "windowed"), (8, "paged"), (4, "windowed")):
        base = table((107, 8), mbs, "gen", arena)
        calls = _settings(base)
        for name, (module, key) in LOW_BANDS.items():
            assert base.on(name), (name, mbs, arena)
            assert base.param(name, "MIN_ROWS") == 33, (name, mbs, arena)
            assert calls[module][0][key] == calls[module][1][key] == 33, (name, mbs, arena)
        assert calls[_PREFETCH_MODULE][0]["published_min_rows"] == 0
    off = _settings(table((107, 8), 16, "gen", "windowed"))
    for module, key in [*LOW_BANDS.values(), (_PREFETCH_MODULE, "published_min_rows")]:
        values, defaults = off[module]
        assert values[key] == defaults[key], (module, key)


@pytest.mark.parametrize("rows", [9, 17, 25, 32, 33])
def test_the_low_band_knobs_reach_their_modules(rows):
    """Each MIN_ROWS knob turns no lever on or off and changes no path; it reaches its module's
    ``configure`` only while its lever is on. The publication's also stops the next-QKV prefetch
    from forking at the rows it takes (as 33..64-row steps fork none), below 33 only."""
    base = table((107, 8), 8, "gen", "windowed")
    moved = table(
        (107, 8),
        8,
        "gen",
        "windowed",
        knobs(
            tep8_publish_splitk__MIN_ROWS=rows,
            tep8_direct_hop__MIN_ROWS=rows,
            tep8_wide64_splitk__MIN_ROWS=rows,
        ),
    )
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    calls = _settings(moved)
    for name, (module, key) in LOW_BANDS.items():
        assert moved.param(name, "MIN_ROWS") == rows, name
        assert calls[module][0][key] == rows, name
    # The other parameters of the same levers keep their defaults.
    assert calls["ops.comm.projection_publish"][0]["tep8_publish_splitk_max_rows"] == 64
    assert calls["ops.allreduce.post_attn_hop_wide"][0]["direct_max_rows"] == 64
    assert calls["kernels.dense_gemms.dispatcher"][0]["tep8_wide64"]
    published = calls[_PREFETCH_MODULE][0]["published_min_rows"]
    assert published == (rows if rows < 33 else 0)
    for name, (module, key) in LOW_BANDS.items():
        off = table((107, 8), 8, "gen", "windowed", {**LOW17, **knobs(**{name: "off"})})
        assert not off.on(name)
        values, defaults = _settings(off)[module]
        assert values[key] == defaults[key] == 33, name
    # The publication's lever off: no prefetch stand-down either.
    off = _settings(
        table((107, 8), 8, "gen", "windowed", {**LOW17, **knobs(tep8_publish_splitk="off")})
    )
    assert off[_PREFETCH_MODULE][0]["published_min_rows"] == 0


def test_the_low_band_knobs_compose_with_the_wide128_knobs():
    """At max_batch_size 16 with the low-latency levers on by knob, LOW17 and WIDE128 move both ends
    of the same levers' bands (17..128) and nothing else."""
    ll16 = _ll_on(16, VR_TEP8)
    both = table((107, 8), 16, "gen", "windowed", {**ll16, **WIDE128, **LOW17})
    assert both.paths() == table((107, 8), 16, "gen", "windowed", ll16).paths()
    calls = _settings(both)
    publish = calls["ops.comm.projection_publish"][0]
    assert (publish["tep8_publish_splitk_min_rows"], publish["tep8_publish_splitk_max_rows"]) == (
        17,
        128,
    )
    hop = calls["ops.allreduce.post_attn_hop_wide"][0]
    assert (hop["direct_min_rows"], hop["direct_max_rows"]) == (17, 128)
    assert calls["kernels.dense_gemms.dispatcher"][0]["tep8_wide64_min_rows"] == 17
    assert '"MIN_ROWS":17' in both.as_json()


@pytest.mark.parametrize("name", sorted(LOW_BANDS))
@pytest.mark.parametrize("value", [8, 34, 64, "x"])
def test_the_low_band_parameters_refuse_values_outside_9_to_33(name, value):
    with pytest.raises(ValueError, match="expected an integer in 9..33"):
        table((107, 8), 8, "gen", "windowed", knobs(**{f"{name}__MIN_ROWS": value}))


def test_the_gb300_direct_hop_band_switch_is_retired_to_min_rows():
    assert LEVERS.retired_replacement("MUSESPARK_OPEN_TEP8_HOP_DIRECT_ROWS") == (
        LEVERS.knob_name("tep8_direct_hop", "MIN_ROWS")
    )
    with pytest.raises(ValueError, match="MUSESPARK_OPEN_TEP8_HOP_DIRECT_ROWS is retired"):
        table((107, 8), 8, "gen", "windowed", {"MUSESPARK_OPEN_TEP8_HOP_DIRECT_ROWS": "17-64"})


def _host_module(path: Path, names, functions, **stubs) -> types.ModuleType:
    """The top-level assignments of ``names`` and the functions / classes ``functions`` of
    ``path`` (a kernel module that imports torch), run in a module of their own over ``stubs``."""
    import __future__

    tree = ast.parse(path.read_text())
    body = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in functions)
        or (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and (_target_names(node) if isinstance(node, ast.Assign) else {node.target.id})
            <= set(names)
        )
    ]
    host = _module(f"_host_{path.stem}", **stubs)
    code = compile(
        ast.Module(body=body, type_ignores=[]),
        str(path),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, host.__dict__)
    missing = (set(names) | set(functions)) - set(vars(host))
    assert not missing, f"names moved in {path.name}: {sorted(missing)}"
    return host


class _FakeTensor:
    """What ``_splitk_tactic`` and ``publish_projection`` read of a tensor."""

    def __init__(self, rows, cols, dtype="bf16", ptr=1 << 20, contiguous=True, name="") -> None:
        self.shape = (rows, cols)
        self.dtype = dtype
        self.device = "cuda:0"
        self.ptr, self.contiguous, self.name = ptr, contiguous, name
        self.copied_from = None

    def dim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self.contiguous

    def data_ptr(self) -> int:
        return self.ptr

    def t(self) -> "_FakeTensor":
        return self

    def copy_(self, other) -> "_FakeTensor":
        self.copied_from = other
        return self


_PUBLISH = _TARGET / "ops/comm/projection_publish.py"
_PUBLISH_NAMES = (
    "MIN_ROWS",
    "MAX_ROWS",
    "POOL_HIDDEN",
    "UNPOOL_HIDDEN",
    "SPLITK_MIN_ROWS",
    "SPLITK_MAX_ROWS",
    "SPLITK_WIDE_MAX_ROWS",
    "SPLITK_WIDE_LIMIT",
    "PUBLISH_MIN_ROWS",
    "PUBLISH_MIN_LIMIT",
    "TP8_PUBLISH_SPLITK_TACTICS",
    "TEP8_PUBLISH_SPLITK",
    "TEP8_POOL_PROGRAMMATIC",
    "TEP8_JOIN_PDL",
    "JOIN_CLUSTER",
    "_splitk_state",
    "_SPLITK_PDL",
    "_LOW_ROWS_COPIES",
    "POOL_SLICE",
    "SPLITK_WIDE_POOL_TILES",
    "SPLITK_WIDE_POOL_TILE",
    "SPLITK_WIDE_UNPOOL_TILES",
    "SPLITK_WIDE_UNPOOL_TILE",
    "SPLITK_WIDE_TILE_MIN_ROWS",
    "UNPOOL_SLICE",
    "CERTIFIED_SPLITK_PRODUCERS_TP8",
)
_PUBLISH_FUNCTIONS = (
    "configure",
    "splitk_min_rows",
    "publish_min_rows",
    "_splitk_tactic",
    "publish_projection",
    "_low_rows_operands",
    "_wide_tile_tactic",
    "_wide_pool_tactic",
    "_wide_unpool_tactic",
    "_splitk_site_key",
)


def _publish_host(certified_low: bool = True, publisher: bool = True) -> types.ModuleType:
    """``projection_publish``'s band settings and gates; ``_splitk_publisher`` stubbed as the
    real one decides the band below 33 (``_splitk_state["low"]``: the key at MIN_ROWS is the
    certified one)."""
    host = _publish_host_bare()

    def _splitk_publisher():
        if not host.TEP8_PUBLISH_SPLITK or not publisher:
            return None
        if host.PUBLISH_MIN_ROWS < host.SPLITK_MIN_ROWS:
            host._splitk_state["low"] = certified_low
        return (lambda *args: host.launches.append(("splitk",) + args), lambda *fields: fields)

    host._splitk_publisher = _splitk_publisher
    return host


def _publish_host_bare() -> types.ModuleType:
    """The host module over a torch stub: ``torch.grad`` is the grad mode, ``mm`` and ``empty``
    record into ``launches`` / return fresh 512-byte aligned fakes."""
    launches: list = []
    torch_stub = types.SimpleNamespace(bfloat16="bf16", grad=False)
    torch_stub.is_grad_enabled = lambda: torch_stub.grad
    torch_stub.mm = lambda a, b, out: launches.append(("cublas", a, b, out))
    torch_stub.empty = lambda shape, dtype, device: _FakeTensor(*shape, dtype=dtype, ptr=1 << 22)
    host = _host_module(
        _PUBLISH, _PUBLISH_NAMES, _PUBLISH_FUNCTIONS, torch=torch_stub, logger=_Log()
    )
    host.launches = launches
    return host


def _published_rows(host, sites=((8192, 512), (4096, 1024))) -> list:
    """The row counts 1..128 both TP8 sites' split-K producer publishes under ``host``'s settings."""
    out = []
    for rows in range(1, 129):
        taken = [
            host._splitk_tactic(_FakeTensor(rows, k), _FakeTensor(n, k), _FakeTensor(rows, n))
            for k, n in sites
        ]
        if all(t is not None for t in taken):
            out.append(rows)
        else:
            assert all(t is None for t in taken), rows
    return out


def test_projection_publish_keeps_33_rows_by_default():
    host = _publish_host()
    assert (host.MIN_ROWS, host.PUBLISH_MIN_ROWS, host.PUBLISH_MIN_LIMIT) == (33, 33, 9)
    assert host.publish_min_rows() == 33
    assert _published_rows(host) == []  # split-K off
    host.configure(tep8_publish_splitk=True)
    assert host.publish_min_rows() == 33
    assert _published_rows(host) == list(range(33, 65))
    assert "low" not in host._splitk_state


@pytest.mark.parametrize("rows", [9, 17, 24, 32])
def test_projection_publish_min_rows_moves_the_publication_floor(rows):
    """``tep8_publish_splitk_min_rows`` below 33: the publication (``publish_min_rows``, which
    ``PublishedProjections.takes`` reads) and the split-K producer (``_splitk_tactic``) both start at
    it, only while the split-K producer publishes with the certified key at that row count."""
    host = _publish_host()
    host.configure(tep8_publish_splitk=True, tep8_publish_splitk_min_rows=rows)
    assert host.publish_min_rows() == rows
    assert host.splitk_min_rows() == rows
    assert _published_rows(host) == list(range(rows, 65))
    host.configure(
        tep8_publish_splitk=True,
        tep8_publish_splitk_min_rows=rows,
        tep8_publish_splitk_max_rows=128,
    )
    assert _published_rows(host) == list(range(rows, 129))
    # The key at MIN_ROWS is not the certified one: the band below 33 stays off.
    uncertified = _publish_host(certified_low=False)
    uncertified.configure(tep8_publish_splitk=True, tep8_publish_splitk_min_rows=rows)
    assert uncertified.publish_min_rows() == 33
    assert _published_rows(uncertified) == list(range(33, 65))
    # No split-K producer (lever off, or its census refused): never below 33 (cuBLAS's census).
    refused = _publish_host(publisher=False)
    refused.configure(tep8_publish_splitk=True, tep8_publish_splitk_min_rows=rows)
    assert refused.publish_min_rows() == 33
    off = _publish_host()
    off.configure(tep8_publish_splitk=False, tep8_publish_splitk_min_rows=rows)
    assert off.publish_min_rows() == 33
    assert _published_rows(off) == []


def _publish_call(host, rows, k=8192, n=512, **kwargs):
    """``publish_projection`` of a ``rows``-row slice; ``kwargs`` go to the activation's fake."""
    hidden = _FakeTensor(rows, k, name="hidden", **kwargs)
    weight, slot = _FakeTensor(n, k, name="weight"), _FakeTensor(rows, n, name="slot")
    host.launches.clear()
    host.publish_projection(hidden, weight, slot)
    assert len(host.launches) == 1
    return host.launches[0], hidden, slot


@pytest.mark.parametrize("rows", [9, 17, 24, 32])
def test_publish_projection_below_33_rows_never_raises_for_rank_local_reasons(rows):
    """Below 33 rows (MIN_ROWS 9 armed) a slice ``_splitk_tactic`` declines for a rank-local reason
    -- grad mode, a non-contiguous or misaligned activation -- still publishes from split-K (an
    aligned copy of the activation where needed): never cuBLAS, never a raise, and never a per-site
    fallback that would break the transport's buffer alternation."""
    host = _publish_host()
    host.configure(tep8_publish_splitk=True, tep8_publish_splitk_min_rows=9)
    launch, hidden, slot = _publish_call(host, rows)
    assert launch[0] == "splitk" and launch[1] is hidden and launch[4] is slot
    assert launch[6] == (64, 16, 4, 10)  # the pool tactic
    host.torch.grad = True
    launch, hidden, _ = _publish_call(host, rows)
    assert launch[0] == "splitk" and launch[1] is hidden
    host.torch.grad = False
    for kwargs in (dict(ptr=(1 << 20) + 16), dict(contiguous=False)):
        launch, hidden, _ = _publish_call(host, rows, **kwargs)
        assert launch[0] == "splitk" and launch[1] is not hidden, kwargs
        assert launch[1].copied_from is hidden and launch[1].data_ptr() % 32 == 0, kwargs
    assert ("copy", ("activation",)) in host._LOW_ROWS_COPIES
    assert ("grad", True) in host._LOW_ROWS_COPIES
    # The unpool site too.
    launch, _, _ = _publish_call(host, rows, k=4096, n=1024)
    assert launch[0] == "splitk" and launch[6] == (64, 32, 4, 8)
    # 33..64 rows: unchanged (a declined call publishes from cuBLAS, as before).
    host.torch.grad = True
    assert _publish_call(host, 40)[0][0] == "cublas"


def test_publish_projection_below_33_rows_raises_only_when_split_k_cannot_serve():
    host = _publish_host()
    host.configure(tep8_publish_splitk=True, tep8_publish_splitk_min_rows=17)
    for rows, kwargs, reason in (
        (24, dict(k=1000), "no split-K producer serves it"),  # no tactic for the shape
        (16, {}, "no split-K producer serves it"),  # below the armed floor
        (24, dict(dtype="fp16"), "dtypes"),
    ):
        with pytest.raises(RuntimeError, match=reason):
            _publish_call(host, rows, **kwargs)
        assert not any(launch[0] == "cublas" for launch in host.launches)
    refused = _publish_host(certified_low=False)
    refused.configure(tep8_publish_splitk=True, tep8_publish_splitk_min_rows=17)
    with pytest.raises(RuntimeError, match="no split-K producer serves it"):
        _publish_call(refused, 24)


@pytest.mark.parametrize("value", [8, 34, 0])
def test_projection_publish_refuses_min_rows_outside_9_to_33(value):
    host = _publish_host()
    with pytest.raises(ValueError, match="tep8_publish_splitk_min_rows"):
        host.configure(tep8_publish_splitk=True, tep8_publish_splitk_min_rows=value)
    assert host.PUBLISH_MIN_ROWS == 33 and not host.TEP8_PUBLISH_SPLITK


_HOP_WIDE = _TARGET / "ops/allreduce/post_attn_hop_wide.py"
_HOP_NAMES = (
    "MIN_ROWS",
    "SERVED_MULTICAST",
    "SERVED_EARLY_LAUNCH",
    "SERVED_R2",
    "SERVED_R2_WIDE",
    "SERVED_R2_MIN_ROWS",
    "SERVED_SMEM_PAD",
    "SERVED_PAD_ROWS",
    "SERVED_DIRECT",
    "SERVED_DIRECT_ROWS",
    "DIRECT_MIN_ROWS",
    "DIRECT_MAX_ROWS",
    "DIRECT128",
    "DIRECT128_MAX_ROWS",
    "DIRECT128_PAD_ROWS",
    "DIRECT128_PRODUCER_STAGES",
    "DIRECT128_PRODUCER",
    "DIRECT_POLL_FIRST",
)


class _FakeDirectWorkspace:
    ranks, max_rows, fb_layout = 8, 64, True

    def producer_views(self):
        return ["parity0", "parity1"]


def _hop_host() -> types.ModuleType:
    return _host_module(
        _HOP_WIDE,
        _HOP_NAMES,
        ("configure", "DirectPublication"),
        _hop=types.SimpleNamespace(MAX_ROWS=8),
        Tuple=tuple,
        Optional=None,
        logger=_Log(),
    )


def _direct_band(hop) -> list:
    publication = hop.DirectPublication(_FakeDirectWorkspace(), ["d0", "d1"], ["j0", "j1"])
    return [rows for rows in range(1, 129) if publication.takes(rows)]


def test_post_attn_hop_wide_direct_poll_first():
    """``direct_poll_first`` sets ``DIRECT_POLL_FIRST`` only with the direct hop on; default off."""
    hop = _hop_host()
    assert hop.DIRECT_POLL_FIRST is False
    hop.configure(direct=True)
    assert hop.DIRECT_POLL_FIRST is False
    hop.configure(direct=True, direct_poll_first=True)
    assert hop.DIRECT_POLL_FIRST is True
    hop.configure(direct=False, direct_poll_first=True)
    assert hop.DIRECT_POLL_FIRST is False
    hop.configure()
    assert hop.DIRECT_POLL_FIRST is False


def test_the_direct_hop_poll_first_knob_reaches_only_the_hop_module():
    """tep8_direct_hop__POLL_FIRST (0 / 1, default 0): with the lever on it reaches
    ``post_attn_hop_wide.configure``'s ``direct_poll_first`` and nothing else; off, nothing; and
    layers.py builds the direct workspace with ``wait_first`` = not ``DIRECT_POLL_FIRST``."""
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "windowed", {**ll16, **WIDE128})
    assert plain.param("tep8_direct_hop", "POLL_FIRST") == 0
    calls = _settings(plain)
    assert calls["ops.allreduce.post_attn_hop_wide"][0]["direct_poll_first"] is False
    moved = table(
        (107, 8), 16, "gen", "windowed", {**ll16, **WIDE128, **knobs(tep8_direct_hop__POLL_FIRST=1)}
    )
    assert moved.paths() == plain.paths() and rule_levers_on(moved) == rule_levers_on(plain)
    after = _settings(moved)
    hop = after["ops.allreduce.post_attn_hop_wide"][0]
    assert hop["direct_poll_first"] is True
    before = calls["ops.allreduce.post_attn_hop_wide"][0]
    assert {k: v for k, v in hop.items() if k != "direct_poll_first"} == {
        k: v for k, v in before.items() if k != "direct_poll_first"
    }
    assert {m: c for m, c in after.items() if m != "ops.allreduce.post_attn_hop_wide"} == {
        m: c for m, c in calls.items() if m != "ops.allreduce.post_attn_hop_wide"
    }
    # int4w: the 9..16-request gen role runs tep8_direct_hop by default; turn it off by knob.
    off = _settings(
        table(
            (107, 8),
            16,
            "gen",
            "windowed",
            knobs(tep8_direct_hop="off", tep8_direct_hop__POLL_FIRST=1),
        )
    )
    values, defaults = off["ops.allreduce.post_attn_hop_wide"]
    assert values["direct_poll_first"] is defaults["direct_poll_first"] is False
    with pytest.raises(ValueError, match="expected an integer in 0..1"):
        table((107, 8), 8, "gen", "windowed", knobs(tep8_direct_hop__POLL_FIRST=2))
    source = (_TARGET / "layers.py").read_text()
    assert "wait_first=not _wide.DIRECT_POLL_FIRST," in source


@pytest.mark.parametrize("rows", [9, 17, 24, 33])
def test_post_attn_hop_wide_direct_min_rows_moves_the_direct_band(rows):
    """``direct_min_rows`` (9..33) is the direct hop's first row count: ``DirectPublication.takes``
    (what ``WideHopCall.armed`` arms) takes rows..64 on the 64-row workspace; default 33."""
    hop = _hop_host()
    assert hop.SERVED_DIRECT_ROWS == (33, 64) and hop.DIRECT_MIN_ROWS == 33
    hop.configure(direct=True)
    assert _direct_band(hop) == list(range(33, 65))
    hop.configure(direct=True, direct_min_rows=rows)
    assert hop.SERVED_DIRECT_ROWS == (rows, 64)
    assert _direct_band(hop) == list(range(rows, 65))
    hop.configure(direct=True, direct_min_rows=rows, direct_max_rows=128)
    assert hop.SERVED_DIRECT_ROWS == (rows, 128)


@pytest.mark.parametrize("value", [8, 34])
def test_post_attn_hop_wide_refuses_direct_min_rows_outside_9_to_33(value):
    hop = _hop_host()
    with pytest.raises(ValueError, match="direct_min_rows"):
        hop.configure(direct=True, direct_min_rows=value)
    assert hop.SERVED_DIRECT_ROWS == (33, 64) and not hop.SERVED_DIRECT


_DISPATCHER = _TARGET / "kernels/dense_gemms/dispatcher.py"


def _dispatcher_host(functions=(), **stubs) -> types.ModuleType:
    return _host_module(
        _DISPATCHER,
        (
            "TEP8_WIDE64_MIN_ROWS",
            "TEP8_WIDE64_MAX_ROWS",
            "TEP8_WIDE64_FLOOR",
            "TEP8_WIDE64_MIN_LIMIT",
            "SM107_TP8_WIDE64_KERNELS",
            "TEP8_WIDE64_SERVED_SITES",
            "TEP8_WIDE64_SITES",
            "TEP8_WIDE64_KERNELS",
            "TEP8_WIDE64_WIDE_MAX_ROWS",
            "TEP8_WIDE64_WIDE_LIMIT",
            "TEP8_WIDE64_WIDE_SITES",
            "TEP8_WIDE64_WIDE_TILES",
            "TEP8_WIDE64_WIDE_TILE",
            "TEP8_WIDE64_WIDE_SPLIT_KS",
            "TEP8_WIDE64_WIDE_SPLIT_K",
            "TEP8_WIDE64_2CTA",
            "TEP8_WIDE64_WIDE_2CTA",
            "TEP8_WIDE64_2CTA_SPLIT_KS",
            "TEP8_WIDE64_2CTA_RS",
            "TEP8_WIDE64_2CTA_PRESTAGE",
            "TEP8_WIDE64_PREP_FOLD",
            "DIRECT_PREFETCH_MIN_ROWS",
            "DIRECT_PREFETCH_LIMIT",
            "PREFETCH_CONSUMER",
            "DROPPED_PREFETCH_CONSUMERS",
            "PREFETCH_JOIN_AFTER",
        ),
        (
            "configure",
            "_tep8_wide64_kernels",
            "dropped_prefetch_consumers",
            "_2cta_shape_error",
            *functions,
        ),
        MAX_ROWS=8,
        WIDE_MAX_ROWS=32,
        Mapping=dict,
        FrozenSet=frozenset,
        Dict=dict,
        Optional=None,
        Sequence=tuple,
        Tuple=tuple,
        **stubs,
    )


def test_dense_gemms_tep8_wide64_min_rows():
    """``tep8_wide64_min_rows`` (9..33, default 33) is ``TEP8_WIDE64_MIN_ROWS``, the first row count
    ``wide64`` serves; a value outside raises and sets nothing."""
    host = _dispatcher_host()
    assert host.TEP8_WIDE64_MIN_ROWS == host.TEP8_WIDE64_FLOOR == 33
    assert host.TEP8_WIDE64_MIN_LIMIT == 9
    host.configure(tep8_wide64=True)
    assert host.TEP8_WIDE64_MIN_ROWS == 33 and host.TEP8_WIDE64_SITES == {"qkv", "o_proj"}
    host.configure(tep8_wide64=True, tep8_wide64_min_rows=17)
    assert host.TEP8_WIDE64_MIN_ROWS == 17
    for bad in (8, 34):
        with pytest.raises(ValueError, match="tep8_wide64_min_rows"):
            host.configure(tep8_wide64=True, tep8_wide64_min_rows=bad)
        assert host.TEP8_WIDE64_MIN_ROWS == 17


# --------------------------------------------------------------------------
# tep8_wide64_splitk__MAX_ROWS / __WIDE_TILE: the 65..128-row TP8 QKV (9..16 requests at K = 7) in
# one launch of the wide64 tactic instead of cuBLAS
# --------------------------------------------------------------------------
_WIDE64_DISPATCHER = "kernels.dense_gemms.dispatcher"
#: The two knobs' configure keywords and their defaults (the served band).
_WIDE64_MAX_DEFAULTS = {
    "tep8_wide64_max_rows": 64,
    "tep8_wide64_wide_tile": "64x64",
    "tep8_wide64_wide_split_k": 0,
}


def test_the_wide64_max_rows_parameters_default_to_the_served_band():
    """MAX_ROWS 64 and WIDE_TILE 64x64 on every TEP8 generation table that serves the wide64 split-K,
    and the dispatcher's ``configure`` gets its defaults (QKV above 64 rows keeps cuBLAS); off (a
    server above sixteen requests under auto: int1 runs the LL levers up to 16 on a TEP8 gen server) too."""
    for mbs, arena in ((8, "windowed"), (8, "paged"), (4, "windowed")):
        base = table((107, 8), mbs, "gen", arena)
        assert base.on("tep8_wide64_splitk"), (mbs, arena)
        assert base.param("tep8_wide64_splitk", "MAX_ROWS") == 64, (mbs, arena)
        assert base.param("tep8_wide64_splitk", "WIDE_TILE") == "64x64", (mbs, arena)
        values, defaults = _settings(base)[_WIDE64_DISPATCHER]
        assert values["tep8_wide64"]
        for key, default in _WIDE64_MAX_DEFAULTS.items():
            assert values[key] == defaults[key] == default, (key, mbs, arena)
    off = table((107, 8), 17, "gen", "windowed")
    assert not off.on("tep8_wide64_splitk")
    values, defaults = _settings(off)[_WIDE64_DISPATCHER]
    for key in ("tep8_wide64", *_WIDE64_MAX_DEFAULTS):
        assert values[key] == defaults[key], key


@pytest.mark.parametrize(
    "rows, tile, split_k",
    [
        (72, "64x64", 0),
        (96, "128x64", 0),
        (128, "64x64", 4),
        (128, "64x128", 0),
        (65, "128x64", 0),
        (128, "128x128", 8),
        (96, "128x128", 0),
        (128, "2cta", 0),
        (96, "2cta", 4),
    ],
)
def test_the_wide64_max_rows_knobs_reach_the_dispatcher(rows, tile, split_k):
    """At max_batch_size 16 with the low-latency levers on by knob, MAX_ROWS / WIDE_TILE turn no
    lever on or off and change no path; they reach the dispatcher's ``configure`` (and nothing else)
    only while ``tep8_wide64_splitk`` is on."""
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "windowed", ll16)
    set_ = knobs(
        tep8_wide64_splitk__MAX_ROWS=rows,
        tep8_wide64_splitk__WIDE_TILE=tile,
        tep8_wide64_splitk__WIDE_SPLIT_K=split_k,
    )
    moved = table((107, 8), 16, "gen", "windowed", {**ll16, **set_})
    assert moved.paths() == plain.paths()
    assert rule_levers_on(moved) == rule_levers_on(plain)
    assert moved.param("tep8_wide64_splitk", "MAX_ROWS") == rows
    assert moved.param("tep8_wide64_splitk", "WIDE_TILE") == tile
    assert moved.param("tep8_wide64_splitk", "WIDE_SPLIT_K") == split_k
    calls, before = _settings(moved), _settings(plain)
    dispatcher = calls[_WIDE64_DISPATCHER][0]
    assert dispatcher["tep8_wide64"] and calls["ops.gemm.splitk_gemm"][0]["wide64"]
    assert dispatcher["tep8_wide64_max_rows"] == rows
    assert dispatcher["tep8_wide64_wide_tile"] == tile
    assert dispatcher["tep8_wide64_min_rows"] == 33
    # Every other configure call is the one without the knobs.
    assert {m: c for m, c in calls.items() if m != _WIDE64_DISPATCHER} == {
        m: c for m, c in before.items() if m != _WIDE64_DISPATCHER
    }
    assert {k: v for k, v in dispatcher.items() if k not in _WIDE64_MAX_DEFAULTS} == {
        k: v for k, v in before[_WIDE64_DISPATCHER][0].items() if k not in _WIDE64_MAX_DEFAULTS
    }
    # The lever off (max_batch_size 17 under auto, above the TEP8 gen MAX_BATCH 16): the knobs reach nothing.
    off = table((107, 8), 17, "gen", "windowed", set_)
    assert not off.on("tep8_wide64_splitk")
    values, defaults = _settings(off)[_WIDE64_DISPATCHER]
    for key, default in _WIDE64_MAX_DEFAULTS.items():
        assert values[key] == defaults[key] == default, key


@pytest.mark.parametrize(
    "knob, value, message",
    [
        ("tep8_wide64_splitk__MAX_ROWS", 63, "expected an integer in 64..128"),
        ("tep8_wide64_splitk__MAX_ROWS", 129, "expected an integer in 64..128"),
        ("tep8_wide64_splitk__MAX_ROWS", "x", "expected an integer in 64..128"),
        ("tep8_wide64_splitk__WIDE_TILE", "256x64", "expected one of"),
        ("tep8_wide64_splitk__WIDE_SPLIT_K", 2, r"expected one of \[0, 4, 8\]"),
        ("tep8_wide64_splitk__WIDE_SPLIT_K", 16, r"expected one of \[0, 4, 8\]"),
        ("tep8_wide64_splitk__WIDE_SPLIT_K", "x", r"expected one of \[0, 4, 8\]"),
        ("tep8_wide64_splitk__WIDE_TILE", "64", "expected one of"),
        ("tep8_wide64_splitk__WIDE_TILE", "64x64,128x64", "expected one of"),
    ],
)
def test_the_wide64_max_rows_parameters_refuse_values_outside_their_kernel(knob, value, message):
    with pytest.raises(ValueError, match=message):
        table((107, 8), 8, "gen", "windowed", knobs(**{knob: value}))


def test_dense_gemms_tep8_wide64_max_rows():
    """``tep8_wide64_max_rows`` (64..128, default 64) is ``TEP8_WIDE64_WIDE_MAX_ROWS`` and
    ``tep8_wide64_wide_tile`` (default ``64x64``) ``TEP8_WIDE64_WIDE_TILE`` as (mma_m, mma_n); a
    value outside, or a band above 64 without ``tep8_wide64``, raises and sets nothing."""
    host = _dispatcher_host()
    state = lambda: (host.TEP8_WIDE64_WIDE_MAX_ROWS, host.TEP8_WIDE64_WIDE_TILE)  # noqa: E731
    assert host.TEP8_WIDE64_WIDE_MAX_ROWS == host.TEP8_WIDE64_MAX_ROWS == 64
    assert host.TEP8_WIDE64_WIDE_LIMIT == 128 and host.TEP8_WIDE64_WIDE_SITES == {"qkv"}
    assert host.TEP8_WIDE64_WIDE_TILE == host.TEP8_WIDE64_WIDE_TILES["64x64"] == (64, 64)
    host.configure(tep8_wide64=True)
    assert state() == (64, (64, 64))
    for name, tile in (("128x64", (128, 64)), ("64x128", (64, 128)), ("128x128", (128, 128))):
        host.configure(tep8_wide64=True, tep8_wide64_max_rows=96, tep8_wide64_wide_tile=name)
        assert state() == (96, tile)
    assert host.TEP8_WIDE64_WIDE_SPLIT_K == 0 and host.TEP8_WIDE64_WIDE_SPLIT_KS == (0, 4, 8)
    host.configure(tep8_wide64=True, tep8_wide64_wide_split_k=8)
    assert host.TEP8_WIDE64_WIDE_SPLIT_K == 8
    host.configure(tep8_wide64=True, tep8_wide64_max_rows=128, tep8_wide64_wide_tile="128x64")
    for kwargs, match in (
        (dict(tep8_wide64=True, tep8_wide64_max_rows=63), "max_rows must be in 64..128"),
        (dict(tep8_wide64=True, tep8_wide64_max_rows=129), "max_rows must be in 64..128"),
        (dict(tep8_wide64_max_rows=96), "above 64 needs tep8_wide64"),
        (dict(tep8_wide64=True, tep8_wide64_wide_tile="256x64"), "tep8_wide64_wide_tile"),
        (dict(tep8_wide64=True, tep8_wide64_wide_split_k=2), "tep8_wide64_wide_split_k"),
    ):
        with pytest.raises(ValueError, match=match):
            host.configure(**kwargs)
        assert state() == (128, (128, 64))
        assert host.TEP8_WIDE64_SITES == {"qkv", "o_proj"}
    host.configure()
    assert state() == (64, (64, 64))
    assert host.TEP8_WIDE64_SITES == frozenset()


def test_dense_gemms_tep8_wide64_wide_tile_2cta():
    """WIDE_TILE ``2cta`` sets ``TEP8_WIDE64_WIDE_2CTA`` (the pair tile 128x128 as the tile), at
    WIDE_SPLIT_K 0 (the served 4) or 4; 8 is refused with it and nothing is set; any other tile or
    the defaults clear it. ``_2cta_shape_error`` takes TP8 QKV (2304 x 8192) at split_k 2 / 4 only."""
    host = _dispatcher_host()
    assert host.TEP8_WIDE64_2CTA == "2cta" and not host.TEP8_WIDE64_WIDE_2CTA
    assert host.TEP8_WIDE64_WIDE_TILES["2cta"] == (128, 128)
    host.configure(tep8_wide64=True, tep8_wide64_max_rows=128, tep8_wide64_wide_tile="2cta")
    assert host.TEP8_WIDE64_WIDE_2CTA and host.TEP8_WIDE64_WIDE_TILE == (128, 128)
    host.configure(
        tep8_wide64=True,
        tep8_wide64_max_rows=96,
        tep8_wide64_wide_tile="2cta",
        tep8_wide64_wide_split_k=4,
    )
    assert host.TEP8_WIDE64_WIDE_2CTA and host.TEP8_WIDE64_WIDE_MAX_ROWS == 96
    with pytest.raises(ValueError, match="2cta serves split_k"):
        host.configure(
            tep8_wide64=True,
            tep8_wide64_max_rows=128,
            tep8_wide64_wide_tile="2cta",
            tep8_wide64_wide_split_k=8,
        )
    assert host.TEP8_WIDE64_WIDE_2CTA and host.TEP8_WIDE64_WIDE_MAX_ROWS == 96
    host.configure(tep8_wide64=True, tep8_wide64_max_rows=128, tep8_wide64_wide_tile="64x128")
    assert not host.TEP8_WIDE64_WIDE_2CTA
    host.configure()
    assert not host.TEP8_WIDE64_WIDE_2CTA
    assert host._2cta_shape_error(2304, 8192, 4) is None
    assert host._2cta_shape_error(2304, 8192, 2) is None
    assert "split_k 8" in host._2cta_shape_error(2304, 8192, 8)
    assert "pair tiles" in host._2cta_shape_error(2300, 8192, 4)
    assert "pair tiles" in host._2cta_shape_error(2304, 8000, 4)


# --------------------------------------------------------------------------
# tep8_wide64_splitk__PREP_FOLD (C3): the multi-request decode prep in the 2-CTA QKV launch
# --------------------------------------------------------------------------
_PREP_FOLD_KNOB = "MUSESPARK_OPEN_LEVER_TEP8_WIDE64_SPLITK__PREP_FOLD"


def test_dense_gemms_tep8_wide64_prep_fold():
    """``tep8_wide64_prep_fold`` (default off) sets ``TEP8_WIDE64_PREP_FOLD`` only with WIDE_TILE
    ``2cta`` above 64 rows; anything else raises naming the knob and sets nothing; the defaults
    clear it."""
    host = _dispatcher_host()
    assert host.TEP8_WIDE64_PREP_FOLD is False
    wide = dict(tep8_wide64=True, tep8_wide64_max_rows=128, tep8_wide64_wide_tile="2cta")
    host.configure(**wide)
    assert host.TEP8_WIDE64_PREP_FOLD is False and host.TEP8_WIDE64_WIDE_2CTA
    host.configure(**wide, tep8_wide64_prep_fold=True)
    assert host.TEP8_WIDE64_PREP_FOLD is True and host.TEP8_WIDE64_WIDE_2CTA
    for kwargs in (
        dict(tep8_wide64=True, tep8_wide64_max_rows=128, tep8_wide64_wide_tile="64x128"),
        dict(tep8_wide64=True, tep8_wide64_max_rows=64, tep8_wide64_wide_tile="2cta"),
        dict(),
    ):
        with pytest.raises(ValueError, match=_PREP_FOLD_KNOB):
            host.configure(**kwargs, tep8_wide64_prep_fold=True)
        assert host.TEP8_WIDE64_PREP_FOLD is True and host.TEP8_WIDE64_WIDE_MAX_ROWS == 128
    host.configure()
    assert host.TEP8_WIDE64_PREP_FOLD is False


def test_the_prep_fold_knob_reaches_only_the_dispatcher():
    """PREP_FOLD defaults to 0; set to 1 it turns no lever on or off and changes no path, and only the
    dispatcher's ``tep8_wide64_prep_fold`` moves (True), only while ``tep8_wide64_splitk`` is on."""
    ll16 = _ll_on(16, VR_TEP8)
    wide = knobs(tep8_wide64_splitk__MAX_ROWS=128, tep8_wide64_splitk__WIDE_TILE="2cta")
    plain = table((107, 8), 16, "gen", "windowed", {**ll16, **wide})
    folded = table(
        (107, 8), 16, "gen", "windowed", {**ll16, **wide, **knobs(tep8_wide64_splitk__PREP_FOLD=1)}
    )
    assert plain.param("tep8_wide64_splitk", "PREP_FOLD") == 0
    assert folded.param("tep8_wide64_splitk", "PREP_FOLD") == 1
    assert folded.paths() == plain.paths()
    assert rule_levers_on(folded) == rule_levers_on(plain)
    calls, before = _settings(folded), _settings(plain)
    values, defaults = calls[_WIDE64_DISPATCHER]
    assert values["tep8_wide64_prep_fold"] is True and defaults["tep8_wide64_prep_fold"] is False
    assert before[_WIDE64_DISPATCHER][0]["tep8_wide64_prep_fold"] is False
    assert {k: v for k, v in values.items() if k != "tep8_wide64_prep_fold"} == {
        k: v for k, v in before[_WIDE64_DISPATCHER][0].items() if k != "tep8_wide64_prep_fold"
    }
    assert {m: c for m, c in calls.items() if m != _WIDE64_DISPATCHER} == {
        m: c for m, c in before.items() if m != _WIDE64_DISPATCHER
    }
    # Every generation table: off by default.
    for mbs, arena in ((8, "windowed"), (8, "paged"), (4, "windowed")):
        base = table((107, 8), mbs, "gen", arena)
        assert base.param("tep8_wide64_splitk", "PREP_FOLD") == 0, (mbs, arena)
        assert _settings(base)[_WIDE64_DISPATCHER][0]["tep8_wide64_prep_fold"] is False
    # The lever off (an aggregated server of max_batch_size 16 under auto; the TEP8 gen role runs it at
    # 9..16 requests): the knob reaches nothing.
    off = table((107, 8), 16, "agg", "windowed", knobs(tep8_wide64_splitk__PREP_FOLD=1))
    assert not off.on("tep8_wide64_splitk")
    assert _settings(off)[_WIDE64_DISPATCHER][0]["tep8_wide64_prep_fold"] is False


@pytest.mark.parametrize("value", [2, -1, "x"])
def test_the_prep_fold_knob_refuses_values_other_than_0_and_1(value):
    with pytest.raises(ValueError, match="0..1"):
        table((107, 8), 8, "gen", "windowed", knobs(tep8_wide64_splitk__PREP_FOLD=value))


_MR_APPEND = _TARGET / "kernels/multi_request_prep_append/attention.py"


class _FoldDispatcher:
    prep_fold = None


class _FoldTensor:
    def __init__(self, rows, cols=2304, ptr=4096):
        self.shape = (rows, cols)
        self.ptr = ptr
        self.dtype = "bf16"
        self.device = "cuda:0"

    def numel(self):
        return self.shape[0] * self.shape[1]

    def data_ptr(self):
        return self.ptr


def _mr_append_host(dense, errors):
    torch_stub = types.SimpleNamespace(
        empty=lambda shape, dtype=None, device=None: ("empty", tuple(shape)), Tensor=object
    )
    logger = types.SimpleNamespace(
        error=errors.append, warning_once=lambda *a, **k: None, info=lambda *a, **k: None
    )
    return _host_module(
        _MR_APPEND,
        ("PREP_FOLD_KNOB", "MIN_SEQS", "NOT_TAKEN", "NOT_REACHED"),
        ("_PrepFoldRequest", "MultiRequestPrepAppendAttention"),
        torch=torch_stub,
        logger=logger,
        Optional=None,
        Tuple=tuple,
        _dense_gemms=lambda: dense,
        takes_step=lambda md, min_seqs: True,
    )


def _fold_layer(host, dispatcher, decline=None, prefetch=None):
    class _Base:
        def forward(self, position_ids, hidden_states, attn_metadata, *args, **kwargs):
            self.seen = dispatcher.prep_fold if dispatcher is not None else None
            return "out"

        def preprocess_qkv(self, qkv, position_ids):
            return "builtin"

    class _Layer(host.MultiRequestPrepAppendAttention, _Base):
        layer_idx = 3
        num_heads = 16
        num_key_value_heads = 2

        def _fits_preattn_prep(self, qkv, position_ids):
            return True

        def _mr_append_decline(self, qkv, md):
            return decline

        def _mr_append_operands(self, qkv, position_ids, md):
            return dict(num_seqs=12), prefetch, lambda qkv_out: ("folded", qkv_out)

    layer = _Layer()
    layer.qkv_proj = types.SimpleNamespace()
    if dispatcher is not None:
        layer.qkv_proj.__dict__["_low_m_dispatcher"] = dispatcher
    layer.records = []
    layer._mr_append_record = lambda kind, detail="": layer.records.append(kind)
    layer._preattn_prep = object()
    layer._mr_append_prep_fold = True
    layer._mr_append_metadata = None
    return layer


def _fold_dense(**overrides):
    values = dict(
        TEP8_WIDE64_PREP_FOLD=True,
        TEP8_WIDE64_WIDE_2CTA=True,
        TEP8_WIDE64_MAX_ROWS=64,
        TEP8_WIDE64_WIDE_MAX_ROWS=128,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_prep_fold_arms_only_2cta_rows_and_disarms_after_the_forward():
    """A multi-request decode forward of 65..WIDE_MAX_ROWS rows arms its QKV dispatcher with a
    ``_PrepFoldRequest`` for the forward's duration (and clears it after); 64 rows, the knob off at
    the dispatcher or a tile other than 2cta arm nothing; no dispatcher raises naming the knob."""
    errors = []
    dispatcher = _FoldDispatcher()
    host = _mr_append_host(_fold_dense(), errors)
    layer = _fold_layer(host, dispatcher)
    positions = object()
    assert layer.forward(positions, _FoldTensor(96, 8192), "md", preattn_prep=object()) == "out"
    assert isinstance(layer.seen, host._PrepFoldRequest)
    assert layer.seen.position_ids is positions and layer.seen.metadata == "md"
    assert dispatcher.prep_fold is None and layer._mr_append_fold_request is None
    for dense, rows in (
        (_fold_dense(), 64),
        (_fold_dense(), 129),
        (_fold_dense(TEP8_WIDE64_PREP_FOLD=False), 96),
        (_fold_dense(TEP8_WIDE64_WIDE_2CTA=False), 96),
    ):
        layer = _fold_layer(_mr_append_host(dense, errors), dispatcher)
        layer.forward(positions, _FoldTensor(rows, 8192), "md", preattn_prep=object())
        assert layer.seen is None, rows
    # Not a multi-request prep step (no preattn_prep op): nothing armed.
    layer = _fold_layer(host, dispatcher)
    layer.forward(positions, _FoldTensor(96, 8192), "md")
    assert layer.seen is None
    with pytest.raises(RuntimeError, match=_PREP_FOLD_KNOB):
        _fold_layer(host, None).forward(positions, _FoldTensor(96, 8192), "md", preattn_prep=1)
    assert not errors


def test_prep_fold_request_hands_the_kernel_the_prep_operands_and_preprocess_its_outputs():
    """The QKV call's ``operands`` are the packed prep's plus fresh ``qkv_out`` / ``gate_out``;
    ``finish`` records them, and the layer's ``preprocess_qkv`` returns them (no prep launch) for that
    projection only, recording the step as the unfused prep does."""
    errors = []
    host = _mr_append_host(_fold_dense(), errors)
    layer = _fold_layer(host, _FoldDispatcher())
    layer._mr_append_metadata = "md"
    positions = object()
    request = host._PrepFoldRequest(layer, positions, "md")
    qkv = _FoldTensor(96)
    operands = request.operands(qkv)
    assert request.reached and request.declined is None
    assert operands == dict(
        qkv_out=("empty", (96, 20 * 64)), gate_out=("empty", (96, 16 * 64)), num_seqs=12
    )
    request.finish(qkv)
    layer._mr_append_fold_request = request
    qkv_out, k, v, gate = layer.preprocess_qkv(qkv, positions)
    assert (qkv_out, k, v, gate) == (("empty", (96, 1280)), None, None, ("empty", (96, 1024)))
    assert layer._mr_append_folded == ("folded", ("empty", (96, 1280)))
    assert layer.records == [host.NOT_REACHED]
    for other_qkv, other_positions in ((_FoldTensor(96, ptr=8192), positions), (qkv, object())):
        with pytest.raises(RuntimeError, match=_PREP_FOLD_KNOB):
            layer.preprocess_qkv(other_qkv, other_positions)
    assert not errors


def test_prep_fold_request_declines_with_the_step_and_refuses_a_prefetch():
    """A step the prep declines declines the fold (the unfused path then declines it the same way,
    silently here); a call the 2-CTA launch never reached logs an ERROR naming the knob and runs the
    prep unfused; a prep prefetch (which the fold does not issue) raises."""
    errors = []
    host = _mr_append_host(_fold_dense(), errors)
    layer = _fold_layer(host, _FoldDispatcher(), decline="a rotary append")
    request = host._PrepFoldRequest(layer, None, "md")
    assert request.operands(_FoldTensor(96)) is None
    assert request.reached and request.declined == "a rotary append" and request.result is None
    layer._mr_append_fold_request = request
    assert layer.preprocess_qkv(_FoldTensor(96), None) == "builtin"
    assert not errors
    layer._mr_append_fold_request = host._PrepFoldRequest(layer, None, "md")
    assert layer.preprocess_qkv(_FoldTensor(96), None) == "builtin"
    assert len(errors) == 1 and _PREP_FOLD_KNOB in errors[0] and "did not run" in errors[0]
    layer = _fold_layer(host, _FoldDispatcher(), prefetch=object())
    with pytest.raises(RuntimeError, match=_PREP_FOLD_KNOB):
        host._PrepFoldRequest(layer, None, "md").operands(_FoldTensor(96))


def test_dense_gemms_wide_tiles_takes_the_2cta_kernel_and_refuses_loudly():
    """With WIDE_TILE ``2cta`` the one-launch call (``_wide_tiles``) goes to ``_wide_2cta``; that
    raises a ValueError naming the knob for a weight the 2-CTA kernel cannot take (no fallback)."""
    stubs = dict(
        torch=types.SimpleNamespace(
            is_grad_enabled=lambda: False,
            compiler=types.SimpleNamespace(is_compiling=lambda: False),
            bfloat16="bf16",
        ),
        parse_kernel=lambda kernel: ("splitk", tuple(int(v) for v in kernel[7:].split(","))),
    )
    host = _dispatcher_host(("DenseGemmDispatcher",), **stubs)
    host.configure(tep8_wide64=True, tep8_wide64_max_rows=128, tep8_wide64_wide_tile="2cta")
    plan = types.SimpleNamespace(
        tp_size=8, enable_pdl=True, named_forks={}, join=lambda: None, join_role=lambda r: None
    )
    dispatcher = host.DenseGemmDispatcher(
        types.SimpleNamespace(role="qkv", plan=plan, shape=(8192, 2304), layer_idx=0)
    )
    real = dispatcher._wide_2cta
    dispatcher._wide_2cta = lambda x, w, rows, k: "2cta"
    assert dispatcher._wide_tiles(_HostTensor(96, 8192), _HostTensor(2304, 8192), 96, 8192) == "2cta"
    with pytest.raises(ValueError, match="WIDE_TILE 2cta: N 2300"):
        real(_HostTensor(96, 8192), _HostTensor(2300, 8192), 96, 8192)
    host.configure()


class _HostTensor:
    """What ``DenseGemmDispatcher.wide64`` reads of an operand."""

    def __init__(self, *shape) -> None:
        self.shape, self.is_cuda, self.dtype = shape, True, "bf16"

    def dim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return True

    def data_ptr(self) -> int:
        return 1 << 20

    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    def view(self, *shape) -> "_HostTensor":
        return self


def _wide64_route(host, role: str, rows: int):
    """Which path ``wide64`` sends a ``rows``-row call of ``role`` down: ``"wide64"`` (the 33..64-row
    tactic), ``"tiles"`` (the one-launch 65..MAX_ROWS call), ``"hop128"``, or None (cuBLAS)."""
    shape = {"qkv": (8192, 2304), "o_proj": (1024, 8192)}[role]
    plan = types.SimpleNamespace(
        tp_size=8, enable_pdl=True, named_forks={}, join=lambda: None, join_role=lambda r: None
    )
    dispatcher = host.DenseGemmDispatcher(
        types.SimpleNamespace(role=role, plan=plan, shape=shape, layer_idx=0)
    )
    dispatcher._wide_tiles = lambda x, w, n, k: "tiles"
    dispatcher._direct128 = lambda x, w, n, k: "hop128"
    k, n = shape
    return dispatcher.wide64(_HostTensor(rows, k), _HostTensor(n, k), None)


def test_dense_gemms_wide64_routes_65_to_max_rows_qkv_to_one_launch():
    """``DenseGemmDispatcher.wide64``: at the default MAX_ROWS (64) a QKV call above 64 rows keeps
    cuBLAS (``None``), as before; with MAX_ROWS 96 / 128 the 65..MAX_ROWS-row QKV calls take the
    one-launch path and the rest above keeps cuBLAS. o_proj's 65..128-row calls go to hop128
    whatever MAX_ROWS, and 33..64-row calls of both sites keep the wide64 tactic."""
    stubs = dict(
        torch=types.SimpleNamespace(
            is_grad_enabled=lambda: False,
            compiler=types.SimpleNamespace(is_compiling=lambda: False),
            bfloat16="bf16",
        ),
        dense_gemm=lambda x, w, kernel, pdl: types.SimpleNamespace(view=lambda *s: "wide64"),
    )
    host = _dispatcher_host(("DenseGemmDispatcher",), **stubs)
    assert _wide64_route(host, "qkv", 64) is None  # tep8_wide64 off: nothing changes
    for max_rows in (64, 96, 128):
        host.configure(tep8_wide64=True, tep8_wide64_max_rows=max_rows)
        for rows in (32, 33, 64, 65, 72, 96, 97, 128, 129):
            want_qkv = None if rows < 33 or rows > max_rows else "wide64" if rows <= 64 else "tiles"
            assert _wide64_route(host, "qkv", rows) == want_qkv, (max_rows, rows)
            want_o = None if rows < 33 or rows > 128 else "wide64" if rows <= 64 else "hop128"
            if rows > 64:  # o_proj below 65 rows looks up the direct publication (a GPU module)
                assert _wide64_route(host, "o_proj", rows) == want_o, (max_rows, rows)
    host.configure()


# --------------------------------------------------------------------------
# 65..128-row post-attention / publication path: tep8_direct_hop__PREFETCH_MIN_ROWS (the QKV call of
# an armed direct-hop step forks o_proj's L2 prefetch) and tep8_publish_splitk__WIDE_POOL_TILE (the
# pool slice's one-launch tile)
# --------------------------------------------------------------------------
#: Each new parameter: (lever, key) -> (module, configure keyword, default, a moved value).
HOP_POOL_PARAMS = {
    ("tep8_direct_hop", "PREFETCH_MIN_ROWS"): (
        "kernels.dense_gemms.dispatcher",
        "tep8_direct_prefetch_min_rows",
        0,
        65,
    ),
    ("tep8_publish_splitk", "WIDE_POOL_TILE"): (
        "ops.comm.projection_publish",
        "tep8_publish_splitk_wide_pool_tile",
        "64x16",
        "64x32",
    ),
    ("tep8_publish_splitk", "WIDE_UNPOOL_TILE"): (
        "ops.comm.projection_publish",
        "tep8_publish_splitk_wide_unpool_tile",
        "64x32",
        "64x64",
    ),
    ("tep8_publish_splitk", "WIDE_TILE_MIN_ROWS"): (
        "ops.comm.projection_publish",
        "tep8_publish_splitk_wide_tile_min_rows",
        65,
        97,
    ),
}


def test_the_hop_pool_parameters_default_to_the_served_paths():
    """PREFETCH_MIN_ROWS 0 and WIDE_POOL_TILE 64x16 on every TEP8 generation table that serves their
    levers; the configure calls keep their module defaults then, and off too."""
    for mbs, arena in ((8, "windowed"), (8, "paged"), (4, "windowed")):
        base = table((107, 8), mbs, "gen", arena)
        calls = _settings(base)
        for (name, key), (module, kw, default, _) in HOP_POOL_PARAMS.items():
            assert base.on(name), (name, mbs, arena)
            assert base.param(name, key) == default, (name, mbs, arena)
            assert calls[module][0][kw] == calls[module][1][kw] == default, (name, mbs, arena)
    # Off: a TEP8 generation server above the LL gate (9..16 requests run the int5 winners' tiles).
    off = _settings(table((107, 8), 17, "gen", "windowed"))
    for module, kw, default, _ in HOP_POOL_PARAMS.values():
        values, defaults = off[module]
        assert values[kw] == defaults[kw] == default, (module, kw)


def test_the_hop_pool_knobs_reach_only_their_modules():
    """At max_batch_size 16 with the low-latency levers on by knob and WIDE128, each knob turns no
    lever on or off, changes no path and reaches only its module's configure keyword; with its lever
    off it reaches nothing."""
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "windowed", {**ll16, **WIDE128})
    for (name, key), (module, kw, default, value) in HOP_POOL_PARAMS.items():
        moved = table(
            (107, 8),
            16,
            "gen",
            "windowed",
            {**ll16, **WIDE128, **knobs(**{f"{name}__{key}": value})},
        )
        assert moved.paths() == plain.paths(), name
        assert rule_levers_on(moved) == rule_levers_on(plain), name
        assert moved.param(name, key) == value, name
        calls, before = _settings(moved), _settings(plain)
        assert calls[module][0][kw] == value, name
        assert {k: v for k, v in calls[module][0].items() if k != kw} == {
            k: v for k, v in before[module][0].items() if k != kw
        }, name
        assert {m: c for m, c in calls.items() if m != module} == {
            m: c for m, c in before.items() if m != module
        }, name
        # int4w: a 9..16-request gen server runs these levers by default (int1 / int2 role defaults),
        # so the lever is turned off by its knob here.
        off = table(
            (107, 8), 16, "gen", "windowed", knobs(**{name: "off", f"{name}__{key}": value})
        )
        assert not off.on(name), name
        values, defaults = _settings(off)[module]
        assert values[kw] == defaults[kw] == default, name


@pytest.mark.parametrize(
    "knob, value, message",
    [
        ("tep8_direct_hop__PREFETCH_MIN_ROWS", 129, "expected an integer in 0..128"),
        ("tep8_direct_hop__PREFETCH_MIN_ROWS", -1, "expected an integer in 0..128"),
        ("tep8_direct_hop__PREFETCH_MIN_ROWS", "x", "expected an integer in 0..128"),
        ("tep8_publish_splitk__WIDE_POOL_TILE", "128x64", "expected one of"),
        ("tep8_publish_splitk__WIDE_POOL_TILE", "64x16,64x32", "expected one of"),
        ("tep8_publish_splitk__WIDE_UNPOOL_TILE", "128x64", "expected one of"),
        ("tep8_publish_splitk__WIDE_UNPOOL_TILE", "64x16", "expected one of"),
    ],
)
def test_the_hop_pool_parameters_refuse_values_outside_their_kernels(knob, value, message):
    with pytest.raises(ValueError, match=message):
        table((107, 8), 8, "gen", "windowed", knobs(**{knob: value}))


class _ArmedPublication:
    """``post_attn_hop_wide._DIRECT_PUBLICATION`` as ``fork_direct_prefetch`` reads it."""

    def __init__(self, rows) -> None:
        self.rows = rows


def _prefetch_host(monkeypatch, publication):
    """The dispatcher's ``DenseGemmDispatcher`` over stubs; the hop module is a stub whose
    ``_DIRECT_PUBLICATION`` is ``publication``. Returns ``(host, qkv, o_proj, forks)``."""
    torch_stub = types.SimpleNamespace(
        is_grad_enabled=lambda: False,
        compiler=types.SimpleNamespace(is_compiling=lambda: False),
    )
    log = _Log()
    log.info_once = lambda message, key=None: log.info(message)
    host = _dispatcher_host(("DenseGemmDispatcher",), torch=torch_stub, logger=log)
    hop = types.ModuleType("_stub_post_attn_hop_wide")
    hop._DIRECT_PUBLICATION = publication
    hop.DIRECT_MAX_ROWS = 128
    forks = []

    def dispatcher(role, shape):
        plan = types.SimpleNamespace(fork_prefetch=forks.append)
        site = types.SimpleNamespace(
            role=role, plan=plan, shape=shape, layer_idx=0, next_weight=lambda: f"{role}-next"
        )
        out = host.DenseGemmDispatcher(site)
        return out

    # ``from ...ops.allreduce import post_attn_hop_wide`` inside the method: resolve it to the stub.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 3 and name == "ops.allreduce" and fromlist == ("post_attn_hop_wide",):
            return types.SimpleNamespace(post_attn_hop_wide=hop)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return host, dispatcher("qkv", (8192, 2304)), dispatcher("o_proj", (1024, 8192)), forks, log


def test_dense_gemms_direct_prefetch_forks_only_on_armed_wide_steps(monkeypatch):
    """``fork_direct_prefetch``: off by default; with PREFETCH_MIN_ROWS 65 only the QKV call of a
    65..128-row step the direct publication armed forks (o_proj's weight, once, logged once);
    33 also takes 33..64; never at 32 rows or fewer (the module's rule), an unarmed step, another
    step's rows, or o_proj."""
    host, qkv, o_proj, forks, log = _prefetch_host(monkeypatch, _ArmedPublication(72))
    weight = _HostTensor(2304, 8192)
    assert host.DIRECT_PREFETCH_MIN_ROWS == 0 and host.DIRECT_PREFETCH_LIMIT == 128
    assert not qkv.fork_direct_prefetch(_HostTensor(72, 8192), weight)
    host.configure(tep8_wide64=True, tep8_direct_prefetch_min_rows=65)
    assert host.DIRECT_PREFETCH_MIN_ROWS == 65
    assert qkv.fork_direct_prefetch(_HostTensor(72, 8192), weight)
    assert forks == ["qkv-next"]
    assert not o_proj.fork_direct_prefetch(_HostTensor(72, 1024), _HostTensor(8192, 1024))
    for rows in (64, 96):  # below the floor; not the armed step's rows
        assert not qkv.fork_direct_prefetch(_HostTensor(rows, 8192), weight), rows
    for rows in (8, 32, 33, 64, 72, 128):
        low, low_qkv, _, low_forks, _ = _prefetch_host(monkeypatch, _ArmedPublication(rows))
        low.configure(tep8_wide64=True, tep8_direct_prefetch_min_rows=1)
        forked = low_qkv.fork_direct_prefetch(_HostTensor(rows, 8192), weight)
        assert forked == (rows > 32) and low_forks == (["qkv-next"] if forked else []), rows
    host.configure()
    assert host.DIRECT_PREFETCH_MIN_ROWS == 0
    for bad in (-1, 129):
        with pytest.raises(ValueError, match="tep8_direct_prefetch_min_rows"):
            host.configure(tep8_direct_prefetch_min_rows=bad)
        assert host.DIRECT_PREFETCH_MIN_ROWS == 0
    assert sum("PREFETCH_MIN_ROWS" in m for m in log.messages("info")) == 1


def test_dense_gemms_apply_linear_forks_after_the_qkv_call():
    """``apply_linear`` forks the direct prefetch after the call's own GEMM (wide64 or the built-in
    ``Linear``), only while PREFETCH_MIN_ROWS is set."""
    source = _DISPATCHER.read_text()
    body = source[source.index("    def apply_linear(") :]
    body = body[: body.index("    @classmethod")]
    assert body.index("dispatcher.wide64(") < body.index("dispatcher.fork_direct_prefetch(")
    assert body.index("Linear.apply_linear(self") < body.index("dispatcher.fork_direct_prefetch(")
    assert "if DIRECT_PREFETCH_MIN_ROWS and not lora:" in body


def _wide_pool_host(sms=212, certified=True, max_rows=128, tile="64x32", unpool="64x32"):
    """``projection_publish`` with the split-K kernel stubbed: ``producer_key`` names the tactic and
    the device code as the census entry does; ``certified`` False removes the unpool's key."""
    host = _publish_host_bare()
    md5 = "534e6f639ccb0b27b855884d278a2883"
    splitk = types.SimpleNamespace(
        SplitKTactic=lambda *fields: tuple(fields),
        producer_key=lambda rows, tactic, pdl: (md5, tuple(tactic), "owner", "v4", "opt-in", True),
    )
    host.torch.cuda = types.SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda d: types.SimpleNamespace(multi_processor_count=sms),
    )
    if not certified:
        host.CERTIFIED_SPLITK_PRODUCERS_TP8 = frozenset(
            {(212, (md5, (64, 16, 4, 10), "owner", "v4", "opt-in", True), None)}
        )
    host.configure(
        tep8_publish_splitk=True,
        tep8_publish_splitk_max_rows=max_rows,
        tep8_publish_splitk_wide_pool_tile=tile,
        tep8_publish_splitk_wide_unpool_tile=unpool,
    )
    return host, splitk


def test_projection_publish_wide_unpool_tile_is_refused_until_certified():
    """WIDE_UNPOOL_TILE 64x64 is (64, 64, 4, 5), certified by the W10 census entry at 212 SMs; without
    an entry naming its key the gate refuses it with an ERROR (the unpool keeps (64, 32, 4, 8)), with
    one it takes it, logged as info. configure refuses an unknown tile and sets nothing."""
    host, splitk = _wide_pool_host(tile="64x16", unpool="64x64")
    assert host.SPLITK_WIDE_UNPOOL_TILES == {"64x32": None, "64x64": (64, 64, 4, 5)}
    assert host.UNPOOL_SLICE == (4096, 1024)
    md5 = "534e6f639ccb0b27b855884d278a2883"
    key = (md5, (64, 64, 4, 5), "owner", "v4", "opt-in", True)
    # The W10 census entry names it at 212 SMs; without that entry the gate refuses it.
    assert any(key in entry[1:] for entry in host.CERTIFIED_SPLITK_PRODUCERS_TP8 if entry[0] == 212)
    host.CERTIFIED_SPLITK_PRODUCERS_TP8 = frozenset(
        entry for entry in host.CERTIFIED_SPLITK_PRODUCERS_TP8 if key not in entry[1:]
    )
    assert host._wide_unpool_tactic(splitk) is None
    errors = host.logger.messages("error")
    assert any(
        "WIDE_UNPOOL_TILE 64x64 ignored: its producer key" in m
        and "the unpool keeps its 33..64-row tile" in m
        for m in errors
    ), errors
    host.CERTIFIED_SPLITK_PRODUCERS_TP8 = host.CERTIFIED_SPLITK_PRODUCERS_TP8 | {(212, key, key)}
    assert host._wide_unpool_tactic(splitk) == (64, 64, 4, 5)
    assert any(
        "WIDE_UNPOOL_TILE 64x64: the unpool slice" in m for m in host.logger.messages("info")
    )
    with pytest.raises(ValueError, match="wide_unpool_tile"):
        host.configure(tep8_publish_splitk=True, tep8_publish_splitk_wide_unpool_tile="64x16")
    assert host.SPLITK_WIDE_UNPOOL_TILE == "64x64"
    host.configure()
    assert host.SPLITK_WIDE_UNPOOL_TILE == "64x32"


@pytest.mark.parametrize("rows", [65, 96, 128])
def test_projection_publish_wide_unpool_tile_takes_only_the_unpool(rows):
    """An accepted unpool tile launches the 65..128-row unpool slice; the pool slice keeps its own."""
    host = _publish_host()
    host.configure(
        tep8_publish_splitk=True,
        tep8_publish_splitk_max_rows=128,
        tep8_publish_splitk_wide_unpool_tile="64x64",
    )
    wide = []
    host._run_splitk_wide = lambda hidden, weight, slot, tactic: wide.append(
        (hidden.shape, tuple(tactic))
    )
    host._splitk_publisher()
    host._splitk_state["wide_unpool"] = (5, 6, 7, 8)  # what the gate accepted (a marker)
    for k, n, want in ((4096, 1024, (5, 6, 7, 8)), (8192, 512, (64, 16, 4, 10))):
        wide.clear()
        host.publish_projection(_FakeTensor(rows, k), _FakeTensor(n, k), _FakeTensor(rows, n))
        assert wide == [((rows, k), want)], (k, n)


def test_projection_publish_wide_pool_tile_needs_a_certified_key():
    """WIDE_POOL_TILE 64x32 is (64, 32, 4, 8), the census's certified unpool key: taken at 212 SMs
    with MAX_ROWS above 64; refused (warning, the served tile) at another SM count, without the
    certified key, or with MAX_ROWS 64; 64x16 is the served tile (None). configure refuses an
    unknown tile and sets nothing."""
    host, splitk = _wide_pool_host()
    assert host.SPLITK_WIDE_POOL_TILES == {
        "64x16": None,
        "64x32": (64, 32, 4, 8),
        "64x64": (64, 64, 4, 5),
    }
    assert host.POOL_SLICE == (8192, 512) and host.TP8_PUBLISH_SPLITK_TACTICS[(4096, 1024)] == (
        64,
        32,
        4,
        8,
    )
    assert host._wide_pool_tactic(splitk) == (64, 32, 4, 8)
    assert any("WIDE_POOL_TILE 64x32" in m for m in host.logger.messages("info"))
    for kwargs, level in (
        (dict(sms=148), "error"),
        (dict(certified=False), "error"),
        (dict(max_rows=64), "warning"),
    ):
        host, splitk = _wide_pool_host(**kwargs)
        assert host._wide_pool_tactic(splitk) is None, kwargs
        assert any("WIDE_POOL_TILE 64x32 ignored" in m for m in host.logger.messages(level)), kwargs
    # 64x64 is (64, 64, 4, 5): the W10-certified unpool 64x64 key, so taken at 212 SMs too.
    host, splitk = _wide_pool_host(tile="64x64")
    assert host._wide_pool_tactic(splitk) == (64, 64, 4, 5)
    with pytest.raises(ValueError, match="wide_pool_tile"):
        host.configure(tep8_publish_splitk=True, tep8_publish_splitk_wide_pool_tile="128x64")
    assert host.SPLITK_WIDE_POOL_TILE == "64x64"
    host.configure()
    assert host.SPLITK_WIDE_POOL_TILE == "64x16"


@pytest.mark.parametrize("rows", [65, 72, 96, 128])
def test_projection_publish_wide_pool_tile_takes_only_the_pool_above_64_rows(rows):
    """With the certified 64x32 accepted, a 65..128-row pool slice launches on it; the unpool slice
    and every slice of 33..64 rows keep their served tactics."""
    host = _publish_host()
    host.configure(
        tep8_publish_splitk=True,
        tep8_publish_splitk_max_rows=128,
        tep8_publish_splitk_wide_pool_tile="64x32",
    )
    wide = []
    host._run_splitk_wide = lambda hidden, weight, slot, tactic: wide.append(
        (hidden.shape, tuple(tactic))
    )
    host._splitk_publisher()
    # What _wide_pool_tactic accepted (a marker tactic: the unpool must not take it).
    host._splitk_state["wide_pool"] = (1, 2, 3, 4)
    for k, n, want in ((8192, 512, (1, 2, 3, 4)), (4096, 1024, (64, 32, 4, 8))):
        wide.clear()
        hidden, weight, slot = _FakeTensor(rows, k), _FakeTensor(n, k), _FakeTensor(rows, n)
        host.publish_projection(hidden, weight, slot)
        assert wide == [((rows, k), want)], (k, n)
    host._splitk_state["wide_pool"] = None
    wide.clear()
    host.publish_projection(_FakeTensor(rows, 8192), _FakeTensor(512, 8192), _FakeTensor(rows, 512))
    assert wide == [((rows, 8192), (64, 16, 4, 10))]
    (launch, *_), _, _ = _publish_call(host, 48)
    assert launch == "splitk"


@pytest.mark.parametrize("rows", [65, 96, 97, 112, 128])
def test_projection_publish_wide_tile_min_rows_keeps_the_served_tiles_below(rows):
    """WIDE_TILE_MIN_ROWS 97: a 65..96-row pool / unpool slice launches on its served tile (the same
    split_k: the same bits), a 97..128-row one on the accepted wide tile; 65 (default) takes the
    whole band. configure refuses a value outside 65..128 and sets nothing."""
    host = _publish_host()
    assert host.SPLITK_WIDE_TILE_MIN_ROWS == 65
    host.configure(
        tep8_publish_splitk=True,
        tep8_publish_splitk_max_rows=128,
        tep8_publish_splitk_wide_pool_tile="64x64",
        tep8_publish_splitk_wide_unpool_tile="64x64",
        tep8_publish_splitk_wide_tile_min_rows=97,
    )
    assert host.SPLITK_WIDE_TILE_MIN_ROWS == 97
    wide = []
    host._run_splitk_wide = lambda hidden, weight, slot, tactic: wide.append(tuple(tactic))
    host._splitk_publisher()
    host._splitk_state["wide_pool"] = (1, 2, 3, 4)
    host._splitk_state["wide_unpool"] = (5, 6, 7, 8)
    for k, n, served, accepted in (
        (8192, 512, (64, 16, 4, 10), (1, 2, 3, 4)),
        (4096, 1024, (64, 32, 4, 8), (5, 6, 7, 8)),
    ):
        wide.clear()
        host.publish_projection(_FakeTensor(rows, k), _FakeTensor(n, k), _FakeTensor(rows, n))
        assert wide == [accepted if rows >= 97 else served], (rows, k)
    for bad in (64, 129):
        with pytest.raises(ValueError, match="wide_tile_min_rows"):
            host.configure(tep8_publish_splitk=True, tep8_publish_splitk_wide_tile_min_rows=bad)
        assert host.SPLITK_WIDE_TILE_MIN_ROWS == 97
    host.configure()
    assert host.SPLITK_WIDE_TILE_MIN_ROWS == 65


def test_projection_publish_wide_tile_min_rows_above_max_rows_leaves_no_band():
    """WIDE_TILE_MIN_ROWS above MAX_ROWS: the wide tile has no band, a warning (not an ERROR) names
    both bounds and the served tile publishes; within the band the info line names it."""
    host, splitk = _wide_pool_host(max_rows=96, tile="64x64")
    host.configure(
        tep8_publish_splitk=True,
        tep8_publish_splitk_max_rows=96,
        tep8_publish_splitk_wide_pool_tile="64x64",
        tep8_publish_splitk_wide_tile_min_rows=97,
    )
    assert host._wide_pool_tactic(splitk) is None
    assert any(
        "no one-launch band (MAX_ROWS 96, WIDE_TILE_MIN_ROWS 97)" in m
        for m in host.logger.messages("warning")
    )
    host.configure(
        tep8_publish_splitk=True,
        tep8_publish_splitk_max_rows=128,
        tep8_publish_splitk_wide_pool_tile="64x64",
        tep8_publish_splitk_wide_tile_min_rows=97,
    )
    assert host._wide_pool_tactic(splitk) == (64, 64, 4, 5)
    assert any(
        "WIDE_POOL_TILE 64x64: the pool slice of 97..128-row steps" in m
        and "65..96 rows keep the served tile" in m
        for m in host.logger.messages("info")
    )


def test_next_qkv_prefetch_stands_down_at_the_published_rows():
    """``published_min_rows`` (the publication's MIN_ROWS below 33): steps of that many rows or more
    fork no next-QKV prefetch; 0 (default) changes nothing; ``skip_rows`` still applies below."""
    host = _host_module(
        _TARGET / "kernels/next_qkv_prefetch/prefetch.py",
        ("SKIP_ROWS", "PUBLISHED_MIN_ROWS"),
        ("configure", "_forks_at"),
    )
    assert [r for r in range(1, 65) if host._forks_at(r)] == list(range(1, 65))
    host.configure(skip_rows=8, published_min_rows=17)
    assert [r for r in range(1, 65) if host._forks_at(r)] == list(range(9, 17))
    host.configure(skip_rows=8)
    assert [r for r in range(1, 65) if host._forks_at(r)] == list(range(9, 65))


# --------------------------------------------------------------------------
# The TEP8 one-cluster SWA block's band (swa_one_cluster_block__TEP8_MIN/MAX_REQUESTS)
# --------------------------------------------------------------------------
_SWA_BLOCK = _TARGET / "ops/attention/swa_block.py"
#: The module settings and constants the TEP8 CL8 band logic of ``swa_block`` reads.
_SWA_HOST_NAMES = frozenset(
    {
        "_SWA_CL8",
        "_TEP8_SWA_CL8",
        "_SWA_PREP_SPLIT",
        "_SWA_ONE2",
        "_ONE2_MAX_REQUESTS",
        "_CL8_TEP8_MIN_REQUESTS",
        "_CL8_TEP8_MAX_REQUESTS",
        "_CL8_TEP8_WIDE_MIN_REQUESTS",
        "CL8_TEP8_REQUESTS_LIMIT",
        "CL8_TEP8_MIN_REQUESTS_FLOOR",
        "MULTI_MAX_REQUESTS",
        "_CL8_EXACT",
    }
)
#: Its host-side band functions: ``configure`` and the request-count gates.
_SWA_HOST_FUNCTIONS = frozenset(
    {"configure", "cl8_tep8_requests", "cl8_tep8_takes", "cl8_tep8_wide_takes"}
)


def _target_names(node) -> set:
    return {
        name.id
        for target in node.targets
        for name in ast.walk(target)
        if isinstance(name, ast.Name)
    }


def _swa_block_host() -> types.ModuleType:
    """``swa_block``'s band settings, ``configure`` and request-count gates, run from its file
    (the module imports torch and triton) in a module of their own; ``log`` collects
    ``configure``'s ``_cl8_log`` lines."""
    tree = ast.parse(_SWA_BLOCK.read_text())
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in _SWA_HOST_FUNCTIONS)
        or (isinstance(node, ast.Assign) and _target_names(node) <= _SWA_HOST_NAMES)
    ]
    log: list = []
    host = _module("_swa_block_host", log=log)
    host._cl8_log = lambda message, warning=False, error=False: log.append(
        ("ERROR " if error else "") + message
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_SWA_BLOCK), "exec"), host.__dict__)
    assert _SWA_HOST_NAMES | _SWA_HOST_FUNCTIONS <= set(vars(host)), "a name moved in swa_block"
    return host


def _cl8_tep8_band(swa) -> list:
    """The request counts of a TEP8 step (1..17) the CL8 build takes under ``swa``'s settings."""
    return [n for n in range(1, 18) if swa.cl8_tep8_takes(n)]


def test_the_tep8_cl8_min_requests_parameter_defaults_to_the_served_band():
    """TEP8_MIN_REQUESTS defaults to 2 (CL8 on 2..8-request TEP8 steps, gated: int2) on every TEP8
    generation table that serves the lever, and to 5 (the band before it) elsewhere, e.g. an
    aggregated server that turns the lever on by its knob; swa_block's own default stays 5."""
    for mbs, arena in ((8, "windowed"), (4, "windowed"), (4, "paged"), (2, "windowed"), (16, "paged")):
        lever_table = table((107, 8), mbs, "gen", arena)
        assert lever_table.on("swa_one_cluster_block"), (mbs, arena)
        assert lever_table.param("swa_one_cluster_block", "TEP8_MIN_REQUESTS") == 2
        swa_call = _settings(lever_table)["ops.attention.swa_block"][0]
        assert swa_call["cl8_tep8_min_requests"] == 2, (mbs, arena)
    agg = table((107, 8), 8, "agg", "windowed", knobs(swa_one_cluster_block="on"))
    assert agg.param("swa_one_cluster_block", "TEP8_MIN_REQUESTS") == 5
    swa = _swa_block_host()
    assert (swa._CL8_TEP8_MIN_REQUESTS, swa._CL8_TEP8_MAX_REQUESTS) == (5, 8)
    assert swa.CL8_TEP8_MIN_REQUESTS_FLOOR == 2
    assert _cl8_tep8_band(swa) == []  # off by default
    swa.configure(cl8_tep8=True, one2=True)
    assert _cl8_tep8_band(swa) == [5, 6, 7, 8]
    assert any("cl8_tep8 (5-8 requests per TEP8 decode step" in line for line in swa.log)


@pytest.mark.parametrize("value", [2, 3, 4, 5])
def test_the_tep8_cl8_min_requests_knob_reaches_swa_block(value):
    """swa_one_cluster_block__TEP8_MIN_REQUESTS (2..5) turns no lever on or off; it reaches
    ``swa_block.configure`` as ``cl8_tep8_min_requests``, next to TEP8_MAX_REQUESTS, and only
    while the lever is on (off, the module keeps its default 5)."""
    base = table((107, 8), 8, "gen", "windowed")
    moved = table(
        (107, 8), 8, "gen", "windowed", knobs(swa_one_cluster_block__TEP8_MIN_REQUESTS=value)
    )
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    assert moved.param("swa_one_cluster_block", "TEP8_MIN_REQUESTS") == value
    assert moved.param("swa_one_cluster_block", "TEP8_MAX_REQUESTS") == 8
    swa_call = _settings(moved)["ops.attention.swa_block"][0]
    assert (swa_call["cl8_tep8_min_requests"], swa_call["cl8_tep8_max_requests"]) == (value, 8)
    assert swa_call["cl8_tep8"] and swa_call["one2"]
    both = table(
        (107, 8),
        8,
        "gen",
        "windowed",
        knobs(
            swa_one_cluster_block__TEP8_MIN_REQUESTS=value,
            swa_one_cluster_block__TEP8_MAX_REQUESTS=16,
        ),
    )
    swa_call = _settings(both)["ops.attention.swa_block"][0]
    assert (swa_call["cl8_tep8_min_requests"], swa_call["cl8_tep8_max_requests"]) == (value, 16)
    off = table(
        (107, 8),
        8,
        "gen",
        "windowed",
        knobs(swa_one_cluster_block="off", swa_one_cluster_block__TEP8_MIN_REQUESTS=value),
    )
    assert not off.on("swa_one_cluster_block")
    values, defaults = _settings(off)["ops.attention.swa_block"]
    assert values["cl8_tep8_min_requests"] == defaults["cl8_tep8_min_requests"] == 5
    assert not values["cl8_tep8"]
    # The GB300 port's switch for the same band is retired to this parameter.
    assert LEVERS.retired_replacement("MUSESPARK_OPEN_TEP8_SWA_CL8_MIN_REQUESTS") == (
        LEVERS.knob_name("swa_one_cluster_block", "TEP8_MIN_REQUESTS")
    )


@pytest.mark.parametrize("value", [2, 3, 4, 5])
def test_swa_block_configure_takes_the_tep8_cl8_min_requests(value):
    """``configure``'s ``cl8_tep8_min_requests`` (2..5) is the band's first request count: CL8
    takes the TEP8 steps of value..cl8_tep8_max_requests, under one2 only, and
    ``cl8_tep8_requests`` (the caller's cap, the prep-append's stand-down) does not move."""
    swa = _swa_block_host()
    swa.configure(cl8_tep8=True, cl8_tep8_min_requests=value, one2=True)
    assert swa._CL8_TEP8_MIN_REQUESTS == value
    assert _cl8_tep8_band(swa) == list(range(value, 9))
    assert swa.cl8_tep8_requests() == 8
    assert swa.log[-1] == (
        "[musespark_open] swa_block CL8 gate: cl8_tep8 "
        f"({value}-8 requests per TEP8 decode step run the one-cluster SWA block)"
    )
    swa.configure(cl8_tep8=True, cl8_tep8_min_requests=value, cl8_tep8_max_requests=16, one2=True)
    assert _cl8_tep8_band(swa) == list(range(value, 17))
    assert swa.cl8_tep8_requests() == 16
    # Without one2 the whole TEP8 band is off (5..16 never reach the block; 2..4 stay on the split
    # kernels), as the gate line says.
    swa.configure(cl8_tep8=True, cl8_tep8_min_requests=value, one2=False)
    assert _cl8_tep8_band(swa) == []
    assert swa.cl8_tep8_requests() == swa.MULTI_MAX_REQUESTS
    assert swa.log[-1].endswith("inactive: it extends one2 (swa_eight_requests), which is off")
    # CL8-TEP8 off: no step of any count, whatever the band.
    swa.configure(cl8_tep8=False, cl8_tep8_min_requests=value, one2=True)
    assert _cl8_tep8_band(swa) == []
    # Back to the defaults: today's band.
    swa.configure(cl8_tep8=True, one2=True)
    assert swa._CL8_TEP8_MIN_REQUESTS == 5
    assert _cl8_tep8_band(swa) == [5, 6, 7, 8]


@pytest.mark.parametrize("value", [0, 1, 6, 8])
def test_swa_block_configure_refuses_a_tep8_cl8_min_requests_outside_2_5(value):
    swa = _swa_block_host()
    with pytest.raises(ValueError, match=r"cl8_tep8_min_requests is 2\.\.5"):
        swa.configure(cl8_tep8=True, cl8_tep8_min_requests=value, one2=True)
    assert swa._CL8_TEP8_MIN_REQUESTS == 5  # nothing set


def test_swa_block_multi_routes_tep8_steps_to_cl8_only_through_its_band():
    """``swa_block_multi``'s TEP8 CL8 launch is gated by ``cl8_tep8_takes``, the only comparison
    against the band's first count anywhere in ``swa_block``: a step of 2..4 sequences reaches CL8
    only when cl8_tep8_min_requests allows it (``cl8_tep8_takes``, above)."""
    tree = ast.parse(_SWA_BLOCK.read_text())
    (op,) = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SwaBlockMultiAttention"
    ]
    (call,) = [
        node for node in op.body if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    ]

    def calls(node, name):
        return any(
            isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == name
            for sub in ast.walk(node)
        )

    gates = [
        node
        for node in ast.walk(call)
        if isinstance(node, ast.If) and calls(node.test, "_cl8_eligible")
    ]
    tep8 = [node for node in gates if calls(node.test, "cl8_tep8_takes")]
    assert len(tep8) == 1, "one TEP8 CL8 launch, behind cl8_tep8_takes"
    assert calls(tep8[0], "_cl8_launch")
    comparing = {
        function.name
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef)
        for node in ast.walk(function)
        if isinstance(node, ast.Compare)
        and any(
            isinstance(sub, ast.Name) and sub.id == "_CL8_TEP8_MIN_REQUESTS"
            for sub in ast.walk(node)
        )
    }
    assert comparing == {"cl8_tep8_takes"}


def test_a_single_choice_and_a_value_set_parameter():
    one = LEVERS.Param("P", "grid", choices=("chunks", "grid"), one=True)
    assert one.parse(" Chunks ") == "chunks"
    for bad in ("", "chunks,grid", "tile"):
        with pytest.raises(ValueError, match="expected one of"):
            one.parse(bad)
    some = LEVERS.Param("S", ("a",), choices=("a", "b"))
    assert some.parse("b,a") == ("a", "b")  # a subset, as before
    values = LEVERS.Param("V", 64, 16, 128, values=(16, 64, 128))
    assert values.parse("128") == 128
    for bad in ("32", "abc", "256"):
        with pytest.raises(ValueError, match=r"expected one of \[16, 64, 128\]"):
            values.parse(bad)


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
    lever_table = table((107, 8), 1, "agg", "dense", knobs(wide_direct_transports="off"))
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


def test_packed_prep_append_at_tep8_is_on_only_by_its_knob():
    # The composite turns the multi-request prep append on by its knob.
    prep = knobs(multi_request_prep_append="on")
    for mbs in (8, 16):
        decision = table((107, 8), mbs, "agg", "paged", prep).decisions["packed_prep_append"]
        assert (decision.on, decision.source) == (False, "auto"), mbs
        assert "only by its knob" in decision.why
    on = table((107, 8), 16, "agg", "paged", {**prep, **knobs(packed_prep_append="on")})
    assert on.on("packed_prep_append") and on.param("packed_prep_append", "MIN_ROWS") == 33
    moved = {**prep, **knobs(packed_prep_append="on", packed_prep_append__MIN_ROWS=96)}
    assert table((107, 8), 16, "agg", "paged", moved).param("packed_prep_append", "MIN_ROWS") == 96
    # TEP4 keeps its rule.
    assert table((107, 4), 64, "agg", "windowed").on("packed_prep_append")


def test_step_commit_v2_overlap_fold_and_inline_params():
    plain = table((107, 8), 16, "agg", "paged", knobs(step_commit_v2="on"))
    assert plain.param("step_commit_v2", "OVERLAP_FOLD") == 0
    assert plain.param("step_commit_v2", "INLINE_COMMIT") == 0
    for fold in (1, 2):
        folded = table(
            (107, 8),
            16,
            "agg",
            "paged",
            knobs(step_commit_v2="on", step_commit_v2__OVERLAP_FOLD=fold),
        )
        assert folded.on("step_commit_v2")
        assert folded.param("step_commit_v2", "OVERLAP_FOLD") == fold
        # Independent of the step loop's stream.
        assert not folded.on("step_loop_on_execution_stream")
    with pytest.raises(ValueError, match="expected an integer in 0..2"):
        table((107, 8), 8, "agg", "paged", knobs(step_commit_v2__OVERLAP_FOLD=3))
    inline = table(
        (107, 8), 8, "agg", "paged", knobs(step_commit_v2="on", step_commit_v2__INLINE_COMMIT=1)
    )
    assert inline.param("step_commit_v2", "INLINE_COMMIT") == 1
    assert not inline.on("step_loop_inline_commits")


def test_hop_entry_trigger_input_boundary_param():
    base = table((107, 8), 16, "agg", "paged", knobs(hop_entry_trigger="on"))
    assert base.param("hop_entry_trigger", "INPUT_BOUNDARY") == 0
    assert base.param("hop_entry_trigger", "MAX_BATCH") == LEVERS.LL_MAX_BATCH
    on = table(
        (107, 8),
        16,
        "agg",
        "paged",
        knobs(hop_entry_trigger="on", hop_entry_trigger__INPUT_BOUNDARY=1),
    )
    assert on.on("hop_entry_trigger") and on.param("hop_entry_trigger", "INPUT_BOUNDARY") == 1
    with pytest.raises(ValueError, match="expected an integer in 0..1"):
        table((107, 8), 16, "agg", "paged", knobs(hop_entry_trigger__INPUT_BOUNDARY=2))


def test_pool_groups_is_an_alias_over_a_paged_arena():
    # Upstream #21789: the stager copies a paged drafter context's attention-op tables, so a
    # paged arena no longer excludes the lever; POOL_GROUPS is kept only as an accepted alias.
    for mbs in (8, 16):
        plain = table((107, 8), mbs, "agg", "paged")
        groups = table((107, 8), mbs, "agg", "paged", knobs(incremental_block_offsets__POOL_GROUPS=1))
        assert plain.decisions["incremental_block_offsets"].source != "structural"
        assert plain.on("incremental_block_offsets") == groups.on("incremental_block_offsets")
        assert plain.paths() == groups.paths()
        assert groups.param("incremental_block_offsets", "POOL_GROUPS") == 1
    # On by default up to MAX_BATCH (8); the knob turns it on at mbs16 with or without the alias.
    assert table((107, 8), 8, "agg", "paged").on("incremental_block_offsets")
    assert not table((107, 8), 16, "agg", "paged").on("incremental_block_offsets")
    for extra in ({}, {"incremental_block_offsets__POOL_GROUPS": 1}):
        on16 = table((107, 8), 16, "agg", "paged", knobs(incremental_block_offsets="on", **extra))
        assert on16.on("incremental_block_offsets")
    with pytest.raises(ValueError, match="expected an integer in 0..1"):
        table((107, 8), 8, "agg", "paged", knobs(incremental_block_offsets__POOL_GROUPS=2))
    # No staging path of its own: runtime has no separate group copy; the one copy's memo
    # compares a step's rows at once (table_runs) inside it.
    source = (_TARGET / "runtime.py").read_text()
    assert "def _group_block_offsets(" not in source
    assert "memo.table_runs(step_keys, step_pages)" in source






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


# ---------------------------------------------------------------------------
# tep8_router_push__CL16_MAX_ROWS: the push's calls of at most that many rows run 16-CTA clusters
# ---------------------------------------------------------------------------

_ROUTER_PUSH = _TARGET / "ops/moe/router_push.py"
#: The module settings and constants the CL16 band logic of ``router_push`` reads.
_PUSH_HOST_NAMES = frozenset(
    {
        "MAX_ROWS",
        "CLUSTER",
        "TEP8_MAX_ROWS",
        "TEP8_CLUSTER",
        "_TEP8_PUSH",
        "TEP8_MIN_ROWS",
        "_TEP8_ROUTE",
        "TEP8_CL16_MAX_ROWS",
        "TEP8_CL16_CLUSTER",
        "ROUTE_ROW_MAX",
        "TEP8_ROUTE_ROW_MAX",
        "TEP8_ROUTE_ROWS_PER_CTA",
        "TEP8_EARLY_TRIGGER",
        "SRC",
    }
)
_PUSH_HOST_FUNCTIONS = frozenset({"configure", "cl16_takes", "armed", "route_armed", "route_extra"})


class _FakePushKernel:
    """A ``router_push.Kernel`` stand-in: the cluster count (16 rows x NEG expert groups per row block) and the
    residency check of the real one, with ``max_clusters`` set per build."""

    max_clusters_by_cl: dict = {}
    built: list = []

    def __init__(self, device_index, world, cluster=16, tpl=8, max_rows=384, extra=(), timing=False, route=False):
        self.world, self.cluster, self.tpl, self.max_rows, self.route = world, cluster, tpl, max_rows, route
        self.neg = (256 // world) // 16
        self.max_clusters = self.max_clusters_by_cl.get(cluster, 64)
        self.regs, self.local_bytes = 96, 0
        _FakePushKernel.built.append((cluster, route))

    def clusters(self, rows):
        return (rows + 15) // 16 * self.neg

    def fits(self, rows):
        return 1 <= rows <= self.max_rows and self.clusters(rows) <= self.max_clusters


def _router_push_host() -> types.ModuleType:
    """``router_push``'s band settings, ``configure``, ``cl16_takes`` and ``RouterPush``'s build choice, run from
    its file (the module imports torch and the CUDA driver) in a module of their own, with a fake ``Kernel``."""
    tree = ast.parse(_ROUTER_PUSH.read_text())
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in _PUSH_HOST_FUNCTIONS)
        or (isinstance(node, (ast.Assign, ast.AnnAssign)) and _push_target_names(node) <= _PUSH_HOST_NAMES)
        or (isinstance(node, ast.ClassDef) and node.name == "RouterPush")
    ]
    _FakePushKernel.built = []
    host = _module(
        "_router_push_host",
        Kernel=_FakePushKernel,
        Optional=__import__("typing").Optional,
        Dict=dict,
        Tuple=tuple,
        torch=_module("torch", zeros=lambda *a, **k: None, int32=None),
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_ROUTER_PUSH), "exec"), host.__dict__)
    assert _PUSH_HOST_NAMES | _PUSH_HOST_FUNCTIONS | {"RouterPush"} <= set(vars(host)), "a name moved in router_push"
    return host


def _push_target_names(node) -> set:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {name.id for target in targets for name in ast.walk(target) if isinstance(name, ast.Name)}


def _push_object(host, cluster=8):
    """A ``RouterPush`` built without the collective (no torch): the CL 8 builds as ``__init__`` makes them."""
    push = host.RouterPush(8, 0, _module("device", index=0), 0, 0, cluster=cluster)
    return push


def _cl16_band(host) -> list:
    return [n for n in range(1, 130) if host.cl16_takes(n)]


def test_the_router_push_cl16_band_defaults_to_off():
    """CL16_MAX_ROWS defaults to 0 on every TEP8 generation table that serves the push, and the module's own
    default is 0: no call runs the CL16 builds, none is compiled."""
    for mbs, arena in ((8, "windowed"), (8, "paged"), (4, "windowed"), (16, "paged")):
        lever_table = table((107, 8), mbs, "gen", arena)
        if not lever_table.on("tep8_router_push"):
            continue
        assert lever_table.param("tep8_router_push", "CL16_MAX_ROWS") == 0, (mbs, arena)
        call = _settings(lever_table)["ops.moe.router_push"][0]
        assert call["tep8_cl16_max_rows"] == 0, (mbs, arena)
    host = _router_push_host()
    assert (host.TEP8_CL16_MAX_ROWS, host.TEP8_CL16_CLUSTER, host.TEP8_CLUSTER) == (0, 16, 8)
    assert _cl16_band(host) == []
    host.configure(tep8_push=True, tep8_min_rows=9, tep8_route=True)
    assert _cl16_band(host) == []
    push = _push_object(host)
    assert push.cl16_build(32, route=True) is None and push.cl16_build(32, route=False) is None
    assert _FakePushKernel.built == [(8, False)]  # the CL 8 build only


@pytest.mark.parametrize("value", [16, 32, 48, 64])
def test_the_router_push_cl16_knob_reaches_configure(value):
    """tep8_router_push__CL16_MAX_ROWS (0..64) turns no lever on or off; it reaches ``router_push.configure`` as
    ``tep8_cl16_max_rows`` next to MIN_ROWS, only while the push is on (off, the module keeps 0)."""
    base = table((107, 8), 8, "gen", "paged")
    moved = table((107, 8), 8, "gen", "paged", knobs(tep8_router_push__CL16_MAX_ROWS=value))
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    assert moved.param("tep8_router_push", "CL16_MAX_ROWS") == value
    assert moved.param("tep8_router_push", "MIN_ROWS") == base.param("tep8_router_push", "MIN_ROWS")
    call = _settings(moved)["ops.moe.router_push"][0]
    assert call["tep8_cl16_max_rows"] == value and call["tep8_push"] and call["tep8_route"]
    off = table(
        (107, 8), 8, "gen", "paged", knobs(tep8_router_push="off", tep8_router_push__CL16_MAX_ROWS=value)
    )
    assert not off.on("tep8_router_push")
    values, defaults = _settings(off)["ops.moe.router_push"]
    assert values["tep8_cl16_max_rows"] == defaults["tep8_cl16_max_rows"] == 0


@pytest.mark.parametrize("bad", [-1, 65, 128])
def test_the_router_push_cl16_knob_refuses_rows_past_v2(bad):
    with pytest.raises(ValueError, match="CL16_MAX_ROWS"):
        table((107, 8), 8, "gen", "paged", knobs(tep8_router_push__CL16_MAX_ROWS=bad))


@pytest.mark.parametrize("bad", [-1, 65, 128])
def test_router_push_configure_refuses_a_cl16_band_past_v2(bad):
    host = _router_push_host()
    with pytest.raises(ValueError, match=r"tep8_cl16_max_rows is 0\.\.64"):
        host.configure(tep8_push=True, tep8_cl16_max_rows=bad)
    assert host.TEP8_CL16_MAX_ROWS == 0 and not host.armed()  # nothing set


def test_router_push_cl16_band_and_build_choice():
    """``cl16_takes`` is 1..CL16_MAX_ROWS while the push is armed; a call there runs the CL16 build of its kind
    (routed or plain) when ``prepare_cl16`` built it and it keeps the call's clusters resident, else the CL 8
    build; the timed and explicit-tpl calls keep the CL 8 build."""
    host = _router_push_host()
    host.configure(tep8_push=True, tep8_min_rows=9, tep8_route=True, tep8_cl16_max_rows=32)
    assert _cl16_band(host) == list(range(1, 33))
    host.configure(tep8_push=False, tep8_cl16_max_rows=32)
    assert _cl16_band(host) == []  # the push off: no band
    host.configure(tep8_push=True, tep8_min_rows=9, tep8_route=True, tep8_cl16_max_rows=32)
    push = _push_object(host)
    assert push.cl16_build(24, route=True) is None  # not prepared yet
    assert push.prepare_cl16(route=True) is None
    assert _FakePushKernel.built == [(8, False), (16, False), (16, True)]
    assert push.prepare_cl16(route=True) is None and len(_FakePushKernel.built) == 3  # once
    for rows in (9, 16, 17, 24, 32):
        assert push.cl16_build(rows, route=True) is push.cl16_route_kernel
        assert push.cl16_build(rows, route=False) is push.cl16_kernel
    for rows in (33, 64, 128):
        assert push.cl16_build(rows, route=True) is None
    # rbcnt (one counter per (row block, expert group) cluster) is the same set at CL 16
    assert push.cl16_kernel.clusters(384) == push.kernel.clusters(384) == 48
    # a build that cannot keep the call's clusters resident leaves it on CL 8
    push.cl16_route_kernel.max_clusters = 2
    assert push.cl16_build(16, route=True) is push.cl16_route_kernel
    assert push.cl16_build(17, route=True) is None
    # a CL16 build failure (e.g. no non-portable cluster size) keeps every call on CL 8, with a reason
    _FakePushKernel.max_clusters_by_cl = {16: 0}
    try:
        failed = _push_object(host)
        assert "clusters resident" in failed.prepare_cl16(route=True)
        assert failed.cl16_build(16, route=True) is None and failed.cl16_build(16, route=False) is None
    finally:
        _FakePushKernel.max_clusters_by_cl = {}
    # a push built at CL 16 already has nothing to add
    assert "already built at CL 16" in _push_object(host, cluster=16).prepare_cl16(route=True)


def test_router_push_call_picks_the_cl16_build_first():
    """``RouterPush.__call__``: the routed call asks ``cl16_build`` before ``route_build``; the plain call asks it
    only without ``timing`` / ``tpl`` (the timed build stays CL 8)."""
    tree = ast.parse(_ROUTER_PUSH.read_text())
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RouterPush"]
    (call,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"]
    text = ast.unparse(call)
    assert "k = self.cl16_build(rows, route=True) or self.route_build()" in text
    assert "self.cl16_build(rows, route=False) if tpl is None and timing is None else None" in text
    (push_for,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_push_for"]
    text = ast.unparse(push_for)
    assert "if TEP8_CL16_MAX_ROWS > 0:" in text and "push.prepare_cl16(route=route_armed())" in text


def _resolve_cl16_blocks(src: str, world: int, cl: int, row_pairs: int = 0) -> str:
    """``src`` with its ``#if CL == BM && NEG * CL != BM``, ``#if ROW_PAIRS`` and
    ``#if ROW_PAIRS && !(CL == BM && NEG * CL != BM)`` blocks resolved for a (WORLD, CL, ROW_PAIRS) build (BM 16,
    NEG = 256 / WORLD / 16), nested blocks included; every other line (other directives too) kept."""
    neg, bm = 256 // world // 16, 16
    cl16 = cl == bm and neg * cl != bm
    known = {
        "#if CL == BM && NEG * CL != BM": cl16,
        "#if ROW_PAIRS": bool(row_pairs),
        "#if ROW_PAIRS && !(CL == BM && NEG * CL != BM)": bool(row_pairs) and not cl16,
    }
    out, stack = [], []  # stack entries: (known, branch value now)
    for line in src.splitlines():
        stripped = line.strip()
        active = all(value for is_known, value in stack if is_known)
        if stripped.startswith("#if"):
            if stripped in known:
                stack.append((True, known[stripped]))
                continue
            stack.append((False, True))
        elif stripped == "#else" and stack:
            is_known, value = stack[-1]
            if is_known:
                stack[-1] = (True, not value)
                continue
        elif stripped == "#endif" and stack:
            is_known, _ = stack.pop()
            if is_known:
                continue
        if active:
            out.append(line)
    assert not stack
    return "\n".join(out)


def test_router_push_cl16_source_leaves_the_cl8_and_tp16_builds_unchanged():
    """The CL16 per-row branch is preprocessed out of the CL 8 TP8 build and the CL 16 TP16 build (NEG x CL ==
    16 there): their source keeps v2's statements; the CL 16 TP8 build gets expert group 0's mapping."""
    host = _router_push_host()
    src = host.SRC
    assert src.count("#if CL == BM && NEG * CL != BM") == 2
    assert src.count("#if ROW_PAIRS") == 3  # the two v2 sites and the second-row call
    v2 = (
        "  const bool per_row = NEG * CL == BM && rows <= ROUTE_ROW_MAX;\n"
        "  if (crank != 0 && !per_row) { FLUSH(); return; }"
    )
    row = "    const int grow = rb * BM + eg * CL + (int)crank;"
    for world, cl in ((8, 8), (16, 16)):
        resolved = _resolve_cl16_blocks(src, world, cl)
        assert v2 in resolved and row in resolved, (world, cl)
        assert "eg == 0 ? rb * BM + (int)crank : rows" not in resolved
    resolved = _resolve_cl16_blocks(src, 8, 16)
    assert v2 not in resolved and row not in resolved
    assert "if (crank != 0 && !(per_row && eg == 0)) { FLUSH(); return; }" in resolved
    assert "const int grow = eg == 0 ? rb * BM + (int)crank : rows;" in resolved


@pytest.mark.parametrize("cl", [8, 16])
@pytest.mark.parametrize("rows", [9, 16, 17, 24, 32, 40, 64, 65, 72, 80, 96, 127, 128])
def test_router_push_per_row_routing_covers_every_row_once(cl, rows):
    """A model of the kernel's v2 index map at TP8 (grid = clusters x CL, cid = block / CL, rb = cid / NEG,
    eg = cid % NEG): CL 8 routes row 16 rb + 8 eg + crank, CL 16 row 16 rb + crank in expert group 0 only;
    either way each row of the call is routed by exactly one CTA, and every row block's NEG leaders publish."""
    neg, bm = 2, 16
    clusters = (rows + bm - 1) // bm * neg
    routed, leaders = [], set()
    for block in range(clusters * cl):
        cid, crank = block // cl, block % cl
        rb, eg = cid // neg, cid % neg
        if crank == 0:
            leaders.add((rb, eg))
        if cl == bm and neg * cl != bm:
            if crank != 0 and eg != 0:
                continue
            grow = rb * bm + crank if eg == 0 else rows
        else:
            grow = rb * bm + eg * cl + crank
        if grow < rows:
            routed.append(grow)
    assert sorted(routed) == list(range(rows))
    assert leaders == {(rb, eg) for rb in range(clusters // neg) for eg in range(neg)}


# ---------------------------------------------------------------------------
# tep8_tile_placement__COST_A / COST_B: the placement's expert cost model per deployment
# ---------------------------------------------------------------------------

_RING_BUDDY_ASSIGN = _TARGET / "ops/moe/ring_buddy_assign.py"
_RBA_HOST_NAMES = frozenset(
    {
        "_RB_PDL",
        "_TILE_PLACEMENT",
        "_TILE_MIN_ROWS",
        "_TILE_MAX_ROWS",
        "_TILE_COST",
        "TILE_COST_A_RANGE",
        "TILE_COST_B_RANGE",
        "NUM_WARPS",
        "_META_WARPS",
        "_WIDE_WARPS",
        "WIDE_WARPS_CHOICES",
        "WIDE_WARPS_FROM_SLOTS",
        "_CUDA_TABLES",
        "MAX_SLOTS",
        "_BF16_TILE_COST",
        "_BF16_TILE",
        "BF16_TILE_CHOICES",
    }
)
_RBA_HOST_FUNCTIONS = frozenset({"configure", "expert_cost", "program_warps", "cuda_tables_take", "tile_rule"})


def _ring_buddy_assign_host() -> types.ModuleType:
    """``ring_buddy_assign``'s placement settings, ``configure`` and ``expert_cost``, run from its file (the
    module imports torch and triton) in a module of their own."""
    tree = ast.parse(_RING_BUDDY_ASSIGN.read_text())
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in _RBA_HOST_FUNCTIONS)
        or (isinstance(node, ast.Assign) and _target_names(node) <= _RBA_HOST_NAMES)
    ]
    host = _module("_ring_buddy_assign_host", Tuple=tuple, Optional=__import__("typing").Optional)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_RING_BUDDY_ASSIGN), "exec"), host.__dict__)
    assert _RBA_HOST_NAMES | _RBA_HOST_FUNCTIONS <= set(vars(host)), "a name moved in ring_buddy_assign"
    return host


def test_the_tile_cost_parameters_default_to_the_gb300_costs():
    """COST_A / COST_B default to 20 / 11 (0.1 us), the costs placed before; ring_buddy_assign's own default
    is the same, so every table places as before."""
    for mbs, arena in ((8, "paged"), (8, "windowed"), (4, "windowed")):
        lever_table = table((107, 8), mbs, "gen", arena)
        if not lever_table.on("tep8_tile_placement"):
            continue
        assert (
            lever_table.param("tep8_tile_placement", "COST_A"),
            lever_table.param("tep8_tile_placement", "COST_B"),
        ) == (20, 11)
        call = _settings(lever_table)["ops.moe.ring_buddy_assign"][0]
        assert call["tile_cost"] == (20, 11), (mbs, arena)
    rba = _ring_buddy_assign_host()
    assert rba._TILE_COST == (20, 11)
    assert rba.expert_cost(0, 8) == 0 and rba.expert_cost(8, 8) == 20 and rba.expert_cost(17, 8) == 42


@pytest.mark.parametrize("cost", [(14, 11), (14, 14), (10, 20), (1, 0), (255, 255)])
def test_the_tile_cost_knobs_reach_ring_buddy_assign(cost):
    """tep8_tile_placement__COST_A / __COST_B turn no lever on or off; they reach
    ``ring_buddy_assign.configure`` as ``tile_cost`` only while the lever is on (off: (20, 11))."""
    a, b = cost
    base = table((107, 8), 8, "gen", "paged")
    moved = table((107, 8), 8, "gen", "paged", knobs(tep8_tile_placement__COST_A=a, tep8_tile_placement__COST_B=b))
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    call = _settings(moved)["ops.moe.ring_buddy_assign"][0]
    assert call["tile_cost"] == (a, b) and call["tile_placement"]
    off = table(
        (107, 8),
        8,
        "gen",
        "paged",
        knobs(tep8_tile_placement="off", tep8_tile_placement__COST_A=a, tep8_tile_placement__COST_B=b),
    )
    values, defaults = _settings(off)["ops.moe.ring_buddy_assign"]
    assert values["tile_cost"] == defaults["tile_cost"] == (20, 11)
    rba = _ring_buddy_assign_host()
    rba.configure(tile_placement=True, tile_min_rows=41, tile_max_rows=64, tile_cost=(a, b))
    assert rba._TILE_COST == (a, b)
    assert rba.expert_cost(9, 8) == a + b  # two 8-token tiles; the configured cost
    assert rba.expert_cost(9, 8, cost=(20, 11)) == 31  # an explicit cost still wins


@pytest.mark.parametrize("knob,bad", [("COST_A", 0), ("COST_A", 256), ("COST_B", -1), ("COST_B", 256)])
def test_the_tile_cost_knobs_refuse_out_of_range_costs(knob, bad):
    with pytest.raises(ValueError, match=knob):
        table((107, 8), 8, "gen", "paged", knobs(**{f"tep8_tile_placement__{knob}": bad}))


@pytest.mark.parametrize("bad", [(0, 11), (20, -1), (256, 11), (20, 256), (20,), (20, 11, 3)])
def test_ring_buddy_assign_configure_refuses_a_bad_tile_cost(bad):
    rba = _ring_buddy_assign_host()
    with pytest.raises(ValueError, match="tile_cost is"):
        rba.configure(tile_placement=True, tile_cost=bad)
    assert rba._TILE_COST == (20, 11) and not rba._TILE_PLACEMENT  # nothing set


def test_the_placement_kernel_reads_the_configured_cost_at_launch():
    """The launch passes COST_A / COST_B / TILE from ``tile_rule`` at call time (constexpr per value; the
    count rule's launch keeps ``_TILE_COST`` and the tables' tile), and the Python reference defaults to
    the configured cost."""
    text = _RING_BUDDY_ASSIGN.read_text()
    assert "rule = tile_rule(selected.shape[0], meta_tile, balance, half, ranks, bf16)" in text
    assert "rule if tilew else (*_TILE_COST, meta_tile if meta is not None else 8)" in text
    assert "COST_A=cost_a," in text and "COST_B=cost_b," in text and "TILE=tile," in text
    tree = ast.parse(text)
    (ref,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "place_tile_reference"]
    assert "cost = _TILE_COST if cost is None else cost" in ast.unparse(ref)


def test_fmk_context_fmha_pure_context_param():
    # PURE_CONTEXT (0 by default) also offers the multi-instance context FMHA on pure context
    # steps; the lever's decision is unchanged by it, and 0..1 is its whole range.
    for mbs in (16, 64):
        default = table((107, 4), mbs, "agg", "windowed")
        assert default.on("fmk_context_fmha"), mbs
        assert default.param("fmk_context_fmha", "PURE_CONTEXT") == 0, mbs
        pure = table((107, 4), mbs, "agg", "windowed", knobs(fmk_context_fmha__PURE_CONTEXT="1"))
        assert pure.on("fmk_context_fmha"), mbs
        assert pure.param("fmk_context_fmha", "PURE_CONTEXT") == 1, mbs
    with pytest.raises(ValueError):
        table((107, 4), 64, "agg", "windowed", knobs(fmk_context_fmha__PURE_CONTEXT="2"))


# ---------------------------------------------------------------------------
# int1: the gated winners are the disaggregated roles' defaults
# ---------------------------------------------------------------------------
#: The TEP8 'wall' parameters a generation server of max_batch_size 9..16 resolves by default:
#: every low-latency lever at MAX_BATCH 16 and the WIDE128 bands (the gated mbs16 knob set).
def test_the_disaggregated_role_defaults():
    # A TEP8 generation server of 9..16 requests: the wall set, rp9, no knob.
    for mbs in (9, 12, 16):
        gen = table((107, 8), mbs, "gen", "paged")
        # (over a paged arena the two-launch side chain is not served: drafter_kv_glue off; the
        # int2 X400 composite samples on the split sampler: vocab_parallel_head off at MAX_BATCH 8)
        assert VR_TEP8 - {"drafter_kv_glue", "vocab_parallel_head"} <= rule_levers_on(gen), mbs
        assert not gen.on("vocab_parallel_head") and gen.param("vocab_parallel_head", "MAX_BATCH") == 8
        assert gen.param("swa_one_cluster_block", "TEP8_MAX_REQUESTS") == LEVERS.TEP8_GEN_CL8_MAX_REQUESTS == 11
        assert gen.param("tep8_router_push", "ROUTE_ROW_MAX") == 128
        for name in VR_TEP8 - {"drafter_kv_glue", "vocab_parallel_head"}:
            if any(param.key == "MAX_BATCH" for param in RULES[name].params):
                assert gen.param(name, "MAX_BATCH") == LEVERS.LL_MAX_BATCH_TEP8_GEN == 16, name
        # Over a windowed arena too (the E2E gen arena): the gated set, without the side chain.
        windowed = table((107, 8), mbs, "gen", "windowed")
        assert not windowed.on("drafter_kv_glue"), mbs
        assert windowed.param("drafter_kv_glue", "MAX_BATCH") == 8, mbs
        assert VR_TEP8 - {"drafter_kv_glue", "vocab_parallel_head"} <= rule_levers_on(windowed), mbs
        for (name, key), (_, value) in WIDE128_PARAMS.items():
            assert gen.param(name, key) == value, (mbs, name, key)
        assert gen.param("tep8_router_push", "MIN_ROWS") == 9
        assert gen.param("tep8_tile_placement", "MAX_ROWS") == 64
        assert gen.param("swa_one_cluster_block", "TEP8_MIN_REQUESTS") == LEVERS.TEP8_GEN_CL8_MIN_REQUESTS == 2
        assert not any("outside its accuracy gates" in warning for warning in gen.warnings)
        # ... the same table as the knob set the gate ran (on an aggregated ruler).
        wall = {
            **_ll_on(16, VR_TEP8 - {"drafter_kv_glue", "vocab_parallel_head"}),
            **WIDE128,
            **knobs(
                tep8_router_push__MIN_ROWS=9,
                tep8_router_push__ROUTE_ROW_MAX=128,
                swa_one_cluster_block__TEP8_MAX_REQUESTS=11,
            ),
        }
        knobbed = table((107, 8), mbs, "gen", "paged", wall)
        assert knobbed.paths() == gen.paths(), mbs
        assert {n: dict(d.params) for n, d in knobbed.decisions.items()} == {
            n: dict(d.params) for n, d in gen.decisions.items()
        }, mbs
        # Knobs still turn each one off, or move it back.
        back = table((107, 8), mbs, "gen", "paged", knobs(prerouted_experts__TEP8_MAX_ROWS=64))
        assert back.param("prerouted_experts", "TEP8_MAX_ROWS") == 64
        assert not table((107, 8), mbs, "gen", "paged", knobs(prerouted_experts="off")).on(
            "prerouted_experts"
        )
    # Up to eight requests: the 1..8-request bands, as before, plus rp9.
    for mbs in (1, 4, 8):
        gen = table((107, 8), mbs, "gen", "paged")
        for (name, key), (default, _) in WIDE128_PARAMS.items():
            assert gen.param(name, key) == default, (mbs, name, key)
        assert gen.param("swa_prep_split", "MAX_BATCH") == 8
        assert gen.param("tep8_router_push", "MIN_ROWS") == 9
    # A TEP4 generation server and every aggregated server keep the old defaults.
    assert not table((107, 4), 12, "gen", "windowed").on("swa_prep_split")
    for key in ((107, 4), (107, 8)):
        for mbs in (8, 16):
            agg = table(key, mbs, "agg", "windowed")
            assert agg.param("swa_prep_split", "MAX_BATCH") == 8, (key, mbs)
            assert not agg.on("fmk_context_fmha") or key == (107, 4)
            assert agg.param("fmk_context_fmha", "PURE_CONTEXT") == 0, (key, mbs)
            assert not agg.on("context_sp_oproj_overlap"), (key, mbs)
    assert table((107, 8), 16, "agg", "windowed", _ll_on(16, VR_TEP8)).param(
        "tep8_router_push", "MIN_ROWS"
    ) == 33
    # A TEP4 context server: fmk on every pure context step, ovts.
    for mbs in (8, 32, 64):
        ctx = table((107, 4), mbs, "ctx", "windowed")
        assert CTX_ROLE_DEFAULTS <= rule_levers_on(ctx), mbs
        assert ctx.decisions["fmk_context_fmha"].source == "auto", mbs
        assert ctx.param("fmk_context_fmha", "PURE_CONTEXT") == 1, mbs
        assert ctx.param("context_sp_extra_parts", "PARTS") == (
            "shard_routing",
            "router_input",
        )
        assert not any("outside its accuracy gates" in warning for warning in ctx.warnings)
        off = table((107, 4), mbs, "ctx", "windowed", knobs(fmk_context_fmha="off"))
        assert not off.on("fmk_context_fmha")
        mixed_only = table((107, 4), mbs, "ctx", "windowed", knobs(fmk_context_fmha__PURE_CONTEXT=0))
        assert mixed_only.param("fmk_context_fmha", "PURE_CONTEXT") == 0
    # fmk_context_fmha serves both bands by rule; an unhinted or generation server runs none.
    assert LEVERS.lever_band("fmk_context_fmha") is None
    assert not table((107, 4), 8, "gen", "windowed").on("fmk_context_fmha")


# ---------------------------------------------------------------------------
# tep8_tile_placement__BF16_COST_A / BF16_COST_B / BF16_TILE: the tile rule on the two BF16 layers
# ---------------------------------------------------------------------------


def test_the_bf16_tile_rule_defaults_to_off():
    """BF16_COST_A / BF16_COST_B / BF16_TILE default to 0 / 0 / 8: the BF16 layers keep the count rule,
    on the tables and in the module, so every placement is the one served before."""
    for mbs, arena in ((8, "paged"), (16, "paged"), (8, "windowed")):
        lever_table = table((107, 8), mbs, "gen", arena)
        if not lever_table.on("tep8_tile_placement"):
            continue
        assert lever_table.param("tep8_tile_placement", "BF16_COST_A") == 0
        assert lever_table.param("tep8_tile_placement", "BF16_COST_B") == 0
        assert lever_table.param("tep8_tile_placement", "BF16_TILE") == 8
        values, defaults = _settings(lever_table)["ops.moe.ring_buddy_assign"]
        assert values["bf16_tile_cost"] == defaults["bf16_tile_cost"] == (0, 0), (mbs, arena)
        assert values["bf16_tile"] == defaults["bf16_tile"] == 8
    rba = _ring_buddy_assign_host()
    assert rba._BF16_TILE_COST == (0, 0) and rba._BF16_TILE == 8
    rba.configure(tile_placement=True, tile_min_rows=41, tile_max_rows=128)
    # the BF16 layers' calls (no tables) keep the count rule; the NVFP4 prerouted calls take the tile rule
    assert rba.tile_rule(96, None, True, False, 8, bf16=True) is None
    assert rba.tile_rule(96, 16, True, False, 8) == (20, 11, 16)


@pytest.mark.parametrize("cost,tile", [((85, 28), 8), ((1, 0), 16), ((255, 255), 64)])
def test_the_bf16_tile_knobs_reach_ring_buddy_assign(cost, tile):
    """The BF16 knobs turn no lever on or off; they reach ``configure`` only while the lever is on, and
    then only the BF16 layers' table-less balanced TP8 calls of MIN_ROWS..MAX_ROWS rows take the rule."""
    a, b = cost
    base = table((107, 8), 8, "gen", "paged")
    assert base.on("tep8_tile_placement")
    set_ = dict(
        tep8_tile_placement__MAX_ROWS=128,
        tep8_tile_placement__BF16_COST_A=a,
        tep8_tile_placement__BF16_COST_B=b,
        tep8_tile_placement__BF16_TILE=tile,
    )
    moved = table((107, 8), 8, "gen", "paged", knobs(**set_))
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    call = _settings(moved)["ops.moe.ring_buddy_assign"][0]
    assert call["bf16_tile_cost"] == (a, b) and call["bf16_tile"] == tile and call["tile_placement"]
    off = table((107, 8), 8, "gen", "paged", knobs(tep8_tile_placement="off", **set_))
    values, defaults = _settings(off)["ops.moe.ring_buddy_assign"]
    assert values["bf16_tile_cost"] == defaults["bf16_tile_cost"] == (0, 0)
    rba = _ring_buddy_assign_host()
    rba.configure(
        tile_placement=True,
        tile_min_rows=41,
        tile_max_rows=128,
        tile_cost=(20, 11),
        bf16_tile_cost=(a, b),
        bf16_tile=tile,
    )
    assert rba.tile_rule(96, None, True, False, 8, bf16=True) == (a, b, tile)
    assert rba.tile_rule(128, None, True, False, 8, bf16=True) == (a, b, tile)
    # the NVFP4 prerouted calls keep their own costs and the tables' tile
    assert rba.tile_rule(96, 16, True, False, 8, bf16=False) == (20, 11, 16)
    # outside the band, unbalanced, half-window, tep4 or an NVFP4 call without tables: the count rule
    assert rba.tile_rule(40, None, True, False, 8, bf16=True) is None
    assert rba.tile_rule(129, None, True, False, 8, bf16=True) is None
    assert rba.tile_rule(96, None, False, False, 8, bf16=True) is None
    assert rba.tile_rule(96, None, True, True, 8, bf16=True) is None
    assert rba.tile_rule(96, None, True, False, 4, bf16=True) is None
    assert rba.tile_rule(96, None, True, False, 8, bf16=False) is None
    # the lever off: nothing takes the rule, whatever the BF16 costs
    rba.configure(tile_placement=False, bf16_tile_cost=(a, b), bf16_tile=tile)
    assert rba.tile_rule(96, None, True, False, 8, bf16=True) is None


def test_a_zero_bf16_cost_a_is_off_whatever_b():
    rba = _ring_buddy_assign_host()
    rba.configure(tile_placement=True, tile_min_rows=41, tile_max_rows=128, bf16_tile_cost=(0, 30))
    assert rba._BF16_TILE_COST == (0, 0)
    assert rba.tile_rule(96, None, True, False, 8, bf16=True) is None


@pytest.mark.parametrize(
    "knob,bad",
    [("BF16_COST_A", -1), ("BF16_COST_A", 256), ("BF16_COST_B", -1), ("BF16_COST_B", 256), ("BF16_TILE", 12)],
)
def test_the_bf16_tile_knobs_refuse_out_of_range_values(knob, bad):
    with pytest.raises(ValueError, match=knob):
        table((107, 8), 8, "gen", "paged", knobs(**{f"tep8_tile_placement__{knob}": bad}))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(bf16_tile_cost=(256, 0)), "bf16_tile_cost is"),
        (dict(bf16_tile_cost=(20, -1)), "bf16_tile_cost is"),
        (dict(bf16_tile_cost=(20,)), "bf16_tile_cost is"),
        (dict(bf16_tile=12), "bf16_tile is"),
    ],
)
def test_ring_buddy_assign_configure_refuses_a_bad_bf16_rule(kwargs, match):
    rba = _ring_buddy_assign_host()
    with pytest.raises(ValueError, match=match):
        rba.configure(tile_placement=True, **kwargs)
    assert rba._BF16_TILE_COST == (0, 0) and rba._BF16_TILE == 8 and not rba._TILE_PLACEMENT  # nothing set


def test_ring_buddy_marks_the_bf16_layers_calls():
    """The table-less placement call says whether its layer is a BF16 one (``state.kind``); the prerouted
    call (NVFP4 only) does not pass it."""
    text = (_TARGET / "kernels/replicated_experts/ring_buddy.py").read_text()
    assert text.count('bf16=state.kind == "bf16",') == 1
    tree = ast.parse(text)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "ring_buddy_assign"
    ]
    with_meta = [c for c in calls if any(k.arg == "meta_tile" for k in c.keywords)]
    without = [c for c in calls if not any(k.arg == "meta_tile" for k in c.keywords)]
    assert with_meta and without
    assert all(not any(k.arg == "bf16" for k in c.keywords) for c in with_meta)
    assert any(any(k.arg == "bf16" for k in c.keywords) for c in without)


# ---------------------------------------------------------------------------
# tep8_router_push__ROUTE_ROW_MAX: step A's per-row (v2) band past 64 rows
# ---------------------------------------------------------------------------


def test_the_router_push_v2_band_defaults_to_64_rows():
    """ROUTE_ROW_MAX defaults to 64 (v2 up to bs8, v1 above: the band served before) on the TEP8 tables of up
    to 8 requests and in the module, and to 128 on a TEP8 generation server of 9..16 requests (int2 X400
    composite); the routed build compiles with the configured band."""
    for mbs, arena in ((8, "paged"), (16, "paged"), (8, "windowed"), (12, "windowed")):
        lever_table = table((107, 8), mbs, "gen", arena)
        if not lever_table.on("tep8_router_push"):
            continue
        band = 128 if mbs > 8 else 64
        assert lever_table.param("tep8_router_push", "ROUTE_ROW_MAX") == band, (mbs, arena)
        assert _settings(lever_table)["ops.moe.router_push"][0]["tep8_route_row_max"] == band
    agg = table((107, 8), 16, "agg", "paged", knobs(tep8_router_push="on"))
    assert agg.param("tep8_router_push", "ROUTE_ROW_MAX") == 64
    host = _router_push_host()
    assert (host.ROUTE_ROW_MAX, host.TEP8_ROUTE_ROW_MAX, host.TEP8_MAX_ROWS) == (64, 64, 128)
    tree = ast.parse(_ROUTER_PUSH.read_text())
    (kernel,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Kernel"]
    text = ast.unparse(kernel)
    assert "f'-DROUTE_ROW_MAX={TEP8_ROUTE_ROW_MAX}'.encode()" in text
    assert "-DROUTE_ROW_MAX={ROUTE_ROW_MAX}" not in text


@pytest.mark.parametrize("value", [65, 72, 80, 96, 128])
def test_the_router_push_v2_band_knob_reaches_configure(value):
    base = table((107, 8), 8, "gen", "paged")
    moved = table((107, 8), 8, "gen", "paged", knobs(tep8_router_push__ROUTE_ROW_MAX=value))
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    call = _settings(moved)["ops.moe.router_push"][0]
    assert call["tep8_route_row_max"] == value and call["tep8_route"]
    assert call["tep8_cl16_max_rows"] == 0  # independent of the CL16 band
    off = table(
        (107, 8), 8, "gen", "paged", knobs(tep8_router_push="off", tep8_router_push__ROUTE_ROW_MAX=value)
    )
    values, defaults = _settings(off)["ops.moe.router_push"]
    assert values["tep8_route_row_max"] == defaults["tep8_route_row_max"] == 64
    host = _router_push_host()
    host.configure(tep8_push=True, tep8_route=True, tep8_route_row_max=value, tep8_cl16_max_rows=64)
    assert host.TEP8_ROUTE_ROW_MAX == value
    assert [n for n in range(1, 130) if host.cl16_takes(n)] == list(range(1, 65))  # CL16 stays <= 64


@pytest.mark.parametrize("bad", [63, 129, 0])
def test_the_router_push_v2_band_refuses_rows_outside_64_128(bad):
    with pytest.raises(ValueError, match="ROUTE_ROW_MAX"):
        table((107, 8), 8, "gen", "paged", knobs(tep8_router_push__ROUTE_ROW_MAX=bad))
    host = _router_push_host()
    with pytest.raises(ValueError, match=r"tep8_route_row_max is 64\.\.128"):
        host.configure(tep8_push=True, tep8_route=True, tep8_route_row_max=bad)
    assert host.TEP8_ROUTE_ROW_MAX == 64 and not host.armed()


def test_router_push_v2_is_chosen_in_the_kernel_by_the_compiled_band():
    """The kernel's v1 / v2 choice is ``rows <= ROUTE_ROW_MAX`` on the compiled macro (both branches), so the band
    moves with the build's ``-DROUTE_ROW_MAX`` alone; both epilogues write ``sel`` / ``wts`` for every row."""
    src = _router_push_host().SRC
    assert "const bool per_row = NEG * CL == BM && rows <= ROUTE_ROW_MAX;" in src
    assert "const bool per_row = rows <= ROUTE_ROW_MAX;" in src
    assert src.count("route_row_cta(gather, buf, grow, out, red, bias_s, eps, sel, wts, t)") == 1
    assert "route_block(lgb, bias_s, eps, rb * BM, rows, sel, wts, t);" in src


# ---------------------------------------------------------------------------
# tep8_ring_buddy_wide__WIDE_WARPS: warps of the 1024-slot table-writing program
# ---------------------------------------------------------------------------


def test_the_ring_buddy_wide_warps_default_to_the_served_program():
    for mbs, arena in ((8, "paged"), (16, "paged")):
        lever_table = table((107, 8), mbs, "gen", arena)
        if not lever_table.on("tep8_ring_buddy_wide"):
            continue
        assert lever_table.param("tep8_ring_buddy_wide", "WIDE_WARPS") == 8
        assert _settings(lever_table)["ops.moe.ring_buddy_assign"][0]["wide_warps"] == 8
    rba = _ring_buddy_assign_host()
    assert (rba._WIDE_WARPS, rba._META_WARPS, rba.NUM_WARPS, rba.WIDE_WARPS_FROM_SLOTS) == (8, 8, 4, 1024)
    # 512-slot (33..64 rows) and 1024-slot (65..128 rows) table-writing programs: 8 warps, as served
    assert [rba.program_warps(n, True) for n in (16, 256, 512, 1024)] == [8, 8, 8, 8]
    assert [rba.program_warps(n, False) for n in (16, 512, 1024)] == [4, 4, 4]


@pytest.mark.parametrize("warps", [4, 16, 32])
def test_the_ring_buddy_wide_warps_knob_reaches_the_1024_slot_program_only(warps):
    base = table((107, 8), 8, "gen", "paged")
    moved = table((107, 8), 8, "gen", "paged", knobs(tep8_ring_buddy_wide__WIDE_WARPS=warps))
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    assert _settings(moved)["ops.moe.ring_buddy_assign"][0]["wide_warps"] == warps
    off = table(
        (107, 8), 8, "gen", "paged", knobs(tep8_ring_buddy_wide="off", tep8_ring_buddy_wide__WIDE_WARPS=warps)
    )
    values, defaults = _settings(off)["ops.moe.ring_buddy_assign"]
    assert values["wide_warps"] == defaults["wide_warps"] == 8
    rba = _ring_buddy_assign_host()
    rba.configure(wide_warps=warps)
    assert [rba.program_warps(n, True) for n in (256, 512, 1024)] == [8, 8, warps]
    assert rba.program_warps(1024, False) == 4  # the table-free program keeps NUM_WARPS


@pytest.mark.parametrize("bad", [1, 2, 12, 64])
def test_the_ring_buddy_wide_warps_refuse_other_counts(bad):
    with pytest.raises(ValueError, match="WIDE_WARPS"):
        table((107, 8), 8, "gen", "paged", knobs(tep8_ring_buddy_wide__WIDE_WARPS=bad))
    rba = _ring_buddy_assign_host()
    with pytest.raises(ValueError, match="wide_warps is one of"):
        rba.configure(wide_warps=bad)
    assert rba._WIDE_WARPS == 8


def test_the_placement_launch_takes_its_warps_from_program_warps():
    text = _RING_BUDDY_ASSIGN.read_text()
    assert "num_warps=program_warps(slots, meta is not None)," in text
    assert "num_warps=_META_WARPS if meta is not None else NUM_WARPS" not in text


# ---------------------------------------------------------------------------
# tep8_router_push__ROUTE_ROWS_PER_CTA: v2 in row pairs (half the CTAs through the poll)
# ---------------------------------------------------------------------------


def test_the_router_push_rows_per_cta_defaults_to_one():
    for mbs, arena in ((8, "paged"), (16, "paged")):
        lever_table = table((107, 8), mbs, "gen", arena)
        if not lever_table.on("tep8_router_push"):
            continue
        assert lever_table.param("tep8_router_push", "ROUTE_ROWS_PER_CTA") == 1
        assert _settings(lever_table)["ops.moe.router_push"][0]["tep8_route_rows_per_cta"] == 1
    host = _router_push_host()
    assert host.TEP8_ROUTE_ROWS_PER_CTA == 1
    assert host.route_extra(8, 8) == () and host.route_extra(8, 16) == ()


def test_the_router_push_rows_per_cta_knob_reaches_the_cl8_routed_build_only():
    base = table((107, 8), 8, "gen", "paged")
    moved = table(
        (107, 8), 8, "gen", "paged",
        knobs(tep8_router_push__ROUTE_ROWS_PER_CTA=2, tep8_router_push__ROUTE_ROW_MAX=128),
    )
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    call = _settings(moved)["ops.moe.router_push"][0]
    assert (call["tep8_route_rows_per_cta"], call["tep8_route_row_max"]) == (2, 128)
    off = table(
        (107, 8), 8, "gen", "paged", knobs(tep8_router_push="off", tep8_router_push__ROUTE_ROWS_PER_CTA=2)
    )
    values, defaults = _settings(off)["ops.moe.router_push"]
    assert values["tep8_route_rows_per_cta"] == defaults["tep8_route_rows_per_cta"] == 1
    host = _router_push_host()
    host.configure(tep8_push=True, tep8_route=True, tep8_route_rows_per_cta=2)
    assert host.route_extra(8, 8) == ("-DROW_PAIRS=1",)
    assert host.route_extra(8, 16) == () and host.route_extra(16, 16) == ()  # CL16 / TP16 builds: one row per CTA
    tree = ast.parse(_ROUTER_PUSH.read_text())
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RouterPush"]
    (build,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "route_build"]
    assert "extra=route_extra(self.world, self.cluster)" in ast.unparse(build)


@pytest.mark.parametrize("bad", [0, 3, 4])
def test_the_router_push_rows_per_cta_refuses_other_values(bad):
    with pytest.raises(ValueError, match="ROUTE_ROWS_PER_CTA"):
        table((107, 8), 8, "gen", "paged", knobs(tep8_router_push__ROUTE_ROWS_PER_CTA=bad))
    host = _router_push_host()
    with pytest.raises(ValueError, match="tep8_route_rows_per_cta is 1 or 2"):
        host.configure(tep8_push=True, tep8_route=True, tep8_route_rows_per_cta=bad)
    assert host.TEP8_ROUTE_ROWS_PER_CTA == 1 and not host.armed()


def test_router_push_row_pairs_source():
    """ROW_PAIRS changes only the CL 8 TP8 routed build: expert group 0's CTAs route rows 16 rb + c and
    16 rb + 8 + c (a barrier between the two), the other groups' non-leaders leave after the split exchange; with
    ROW_PAIRS off (every build today) the source is v2's."""
    src = _router_push_host().SRC
    v2 = (
        "  const bool per_row = NEG * CL == BM && rows <= ROUTE_ROW_MAX;\n"
        "  if (crank != 0 && !per_row) { FLUSH(); return; }"
    )
    row = "    const int grow = rb * BM + eg * CL + (int)crank;"
    second = "route_row_cta(gather, buf, grow + CL, out, red, bias_s, eps, sel, wts, t);"
    off = _resolve_cl16_blocks(src, 8, 8, row_pairs=0)
    assert v2 in off and row in off and second not in off
    pairs = _resolve_cl16_blocks(src, 8, 8, row_pairs=1)
    assert v2 not in pairs and row not in pairs
    assert "if (crank != 0 && !(per_row && eg == 0)) { FLUSH(); return; }" in pairs
    assert "const int grow = eg == 0 ? rb * BM + (int)crank : rows;  // and row grow + CL, below" in pairs
    assert second in pairs
    i = pairs.index("if (grow + CL < rows) {")
    assert pairs.index("__syncthreads();", i) < pairs.index(second, i)
    assert 'static_assert(CL * 2 == BM && NEG == 2' in pairs
    # the CL16 build ignores ROW_PAIRS (its own branch comes first)
    cl16 = _resolve_cl16_blocks(src, 8, 16, row_pairs=1)
    assert second not in cl16 and "const int grow = eg == 0 ? rb * BM + (int)crank : rows;  // only" in cl16


@pytest.mark.parametrize("rows", [9, 16, 24, 40, 64, 65, 72, 96, 127, 128])
def test_router_push_row_pairs_route_every_row_once(rows):
    """Model of the row-pairs index map (TP8, CL 8, NEG 2): only expert group 0's CTAs route, CTA c of block rb
    rows 16 rb + c and 16 rb + 8 + c; each row of the call exactly once, and expert group 1 keeps only its
    leaders (which publish) past the split exchange."""
    neg, bm, cl = 2, 16, 8
    clusters = (rows + bm - 1) // bm * neg
    routed, staying = [], 0
    for block in range(clusters * cl):
        cid, crank = block // cl, block % cl
        rb, eg = cid // neg, cid % neg
        if crank != 0 and eg != 0:
            continue
        staying += 1
        grow = rb * bm + crank if eg == 0 else rows
        if grow < rows:
            routed.append(grow)
        if grow + cl < rows:
            routed.append(grow + cl)
    assert sorted(routed) == list(range(rows))
    assert staying == clusters // neg * (cl + 1)  # 9 of 16 CTAs per row block


# ---------------------------------------------------------------------------
# tep8_router_push__RB_CUDA: the ring-buddy placement + prerouted tables as one CUDA CTA (ring_buddy_cuda), which
# the routed push triggers right after its dependency wait (router_push EARLY_TRIGGER)
# ---------------------------------------------------------------------------

_RING_BUDDY_CUDA = _TARGET / "ops/moe/ring_buddy_cuda.py"


def _ring_buddy_cuda() -> types.ModuleType:
    """``ring_buddy_cuda`` itself: its module level imports only ctypes and typing (torch and the CUDA driver at
    the first launch)."""
    spec = importlib.util.spec_from_file_location("_ring_buddy_cuda_host", _RING_BUDDY_CUDA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _balance_ring():
    tree = ast.parse(_RING_BUDDY_ASSIGN.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "balance_ring"]
    host = _module("_balance_ring_host", List=list, Tuple=tuple)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_RING_BUDDY_ASSIGN), "exec"), host.__dict__)
    return host.balance_ring


def _rb_reference(flat, rank):
    """``RingBuddyAssign.reference`` (8 ranks, balanced, two windows) in plain Python: local ids and stats."""
    balance_ring = _balance_ring()
    size, ranks = 32, 8
    present = sorted(set(flat))
    load = [0] * ranks
    for e in present:
        load[e // size] += 1
    x = balance_ring(load)[1]
    moved = set()
    for b in range(ranks):
        moved.update(sorted([e for e in present if e // size == b], reverse=True)[: x[b]])
    local = []
    for e in flat:
        owner, index = divmod(e, size)
        runs_on = (owner + (e in moved)) % ranks
        if runs_on != rank:
            local.append(2 * size + e % (256 - 2 * size))
        else:
            local.append(index if owner == rank else size + index)
    after = [load[r] - x[r] + x[(r - 1) % ranks] for r in range(ranks)]
    return local, load + after


def _rb_routing(rows, seed, skew):
    """Row-major top-8 ids (distinct per row); ``skew`` of the picks from 24 hot experts."""
    rng = random.Random(seed)
    hot = rng.sample(range(256), 24)
    flat = []
    for _ in range(rows):
        row = []
        while len(row) < 8:
            e = rng.choice(hot) if rng.random() < skew else rng.randrange(256)
            if e not in row:
                row.append(e)
        flat += row
    return flat


def test_the_router_push_rb_cuda_defaults_to_off():
    for mbs, arena in ((8, "paged"), (16, "paged")):
        lever_table = table((107, 8), mbs, "gen", arena)
        if not lever_table.on("tep8_router_push"):
            continue
        assert lever_table.param("tep8_router_push", "RB_CUDA") == 0
        settings = _settings(lever_table)
        assert settings["ops.moe.router_push"][0]["tep8_early_trigger"] is False
        assert settings["ops.moe.ring_buddy_assign"][0]["cuda_tables"] is False
    host = _router_push_host()
    assert host.TEP8_EARLY_TRIGGER is False and host.route_extra(8, 8) == () and host.route_extra(8, 16) == ()
    rba = _ring_buddy_assign_host()
    assert rba._CUDA_TABLES is False and not rba.cuda_tables_take(8, True, False, 512, 8)


def test_the_router_push_rb_cuda_knob_reaches_the_push_and_the_placement():
    base = table((107, 8), 8, "gen", "paged")
    moved = table((107, 8), 8, "gen", "paged", knobs(tep8_router_push__RB_CUDA=1))
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    settings = _settings(moved)
    assert settings["ops.moe.router_push"][0]["tep8_early_trigger"] is True
    assert settings["ops.moe.ring_buddy_assign"][0]["cuda_tables"] is True
    off = table((107, 8), 8, "gen", "paged", knobs(tep8_router_push="off", tep8_router_push__RB_CUDA=1))
    push_values, push_defaults = _settings(off)["ops.moe.router_push"]
    assert push_values["tep8_early_trigger"] is push_defaults["tep8_early_trigger"] is False
    rb_values, rb_defaults = _settings(off)["ops.moe.ring_buddy_assign"]
    assert rb_values["cuda_tables"] is rb_defaults["cuda_tables"] is False
    host = _router_push_host()
    host.configure(tep8_push=True, tep8_route=True, tep8_early_trigger=True)
    assert host.route_extra(8, 8) == ("-DEARLY_TRIGGER=1",) and host.route_extra(8, 16) == ("-DEARLY_TRIGGER=1",)
    assert host.route_extra(16, 16) == ()  # the TP16 layout is not served
    host.configure(tep8_push=True, tep8_route=True, tep8_route_rows_per_cta=2, tep8_early_trigger=True)
    assert host.route_extra(8, 8) == ("-DROW_PAIRS=1", "-DEARLY_TRIGGER=1")
    host.configure()
    assert host.TEP8_EARLY_TRIGGER is False
    # both routed builds (CL 8 and the CL16 band's) take the extra options
    tree = ast.parse(_ROUTER_PUSH.read_text())
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RouterPush"]
    (prep,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_cl16"]
    assert "extra=route_extra(self.world, TEP8_CL16_CLUSTER)" in ast.unparse(prep)


@pytest.mark.parametrize("bad", [2, -1])
def test_the_router_push_rb_cuda_refuses_other_values(bad):
    with pytest.raises(ValueError, match="RB_CUDA"):
        table((107, 8), 8, "gen", "paged", knobs(tep8_router_push__RB_CUDA=bad))


def test_router_push_early_trigger_source():
    """EARLY_TRIGGER adds one ``griddepcontrol.launch_dependents`` right after the dependency wait; the build
    without it (every build by default) is the source served before, and the end-of-CTA triggers stay."""
    src = _router_push_host().SRC
    assert "#ifndef EARLY_TRIGGER\n#define EARLY_TRIGGER 0\n#endif" in src
    head = 'asm volatile("griddepcontrol.wait;" ::: "memory");\n  STAMP(1);\n#if EARLY_TRIGGER\n'
    assert src.count(head) == 1
    i = src.index(head)
    block = src[i + len(head) : src.index("#endif", i)]
    assert [ln.strip() for ln in block.splitlines() if ln.strip() and not ln.strip().startswith("//")] == [
        'asm volatile("griddepcontrol.launch_dependents;" ::: "memory");'
    ]
    assert src.count("griddepcontrol.launch_dependents;") == 3
    assert src.count("#if EARLY_TRIGGER") == 1


def test_the_cuda_tables_take_only_the_balanced_two_window_tables():
    rba = _ring_buddy_assign_host()
    rbc = _ring_buddy_cuda()
    rba.configure(cuda_tables=True)
    taken = ((8, True, False, 1024, 16), (8, True, False, 72, 8), (8, True, False, 512, 8))
    for args in taken:
        assert rba.cuda_tables_take(*args), args
    refused = (
        (4, True, False, 512, 8),  # tep4
        (8, False, False, 512, 8),  # owner-only / static placement
        (8, True, True, 512, 8),  # the half window
        (8, True, False, 1032, 8),  # past one CTA's slots
        (8, True, False, 0, 8),
        (8, True, False, 512, None),  # no tables
        (8, True, False, 512, 4),
    )
    for args in refused:
        assert not rba.cuda_tables_take(*args), args
    for args in refused + taken:
        assert rba.cuda_tables_take(*args) == rbc.takes(*args), args
    assert rbc.MAX_SLOTS == rba.MAX_SLOTS == 1024
    rba.configure()
    assert not rba.cuda_tables_take(8, True, False, 512, 8)


def test_the_placement_dispatches_the_cuda_tables_before_the_triton_program():
    tree = ast.parse(_RING_BUDDY_ASSIGN.read_text())
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RingBuddyAssign"]
    (call,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"]
    text = ast.unparse(call)
    gate = (
        "if _CUDA_TABLES and meta is not None and cuda_tables_take(ranks, balance or half, half, num_slots, "
        "meta_tile):"
    )
    i = text.index(gate)
    j = text.index("_ring_buddy_assign_kernel[")
    assert i < j
    branch = text[i:j]
    assert (
        "if _rbc.launch(selected, weights, rank, meta_tile, local_ids, weights_bf16, stats, meta, tilew=tilew, "
        "cost=_TILE_COST):" in branch
    )
    assert "return (local_ids, weights_bf16, stats, meta)" in branch
    # the tile placement (the same band and costs as the Triton program's TILEW) is decided before the branch
    assert text.index("tilew = ") < i


def test_rb_tables_source_reads_nothing_before_its_dependency_wait():
    src = _ring_buddy_cuda().SRC
    wait = src.index('asm volatile("griddepcontrol.wait;" ::: "memory");')
    # the predecessor's ids and weights: coherent volatile loads after the wait, never a plain / read-only load
    for read in ("ld_dep(sel + t)", "ld_dep(wts + t)"):
        assert src.count(read) == 1 and src.index(read) > wait
    assert 'asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");' in src
    assert not re.search(r"\b(sel|wts)\s*\[", src) and "__ldg" not in src
    signature = src[src.index("rb_tables(") : src.index("{", src.index("rb_tables("))]
    assert "__restrict__" not in signature  # a const __restrict__ load may be issued as ld.global.nc above the wait
    before = src[src.index("rb_tables(") : wait]
    stores = re.findall(r"(\w+)\[[^\]]*\](?:\[[^\]]*\])? = ", before)
    assert stores and all(name.startswith("s_") for name in stores), stores  # shared memory only
    assert "__launch_bounds__(NT, 1)" in src and "#define NT 1024" in src
    assert "cvt.rn.bf16.f32" in src  # Triton's FP32 -> BF16 cast
    # positions by warp ranking (token order, not arrival order): the one atomicAdd is tilew's token count
    assert src.count("atomicAdd") == 1 and "atomicAdd(&s_tok[" in src
    assert "launch_dependents" not in src  # FC1 launches at its completion, as after the Triton program


@pytest.mark.parametrize("rows", [9, 16, 33, 48, 64, 65, 72, 96, 128])
def test_rb_tables_model_equals_the_placement_reference(rows):
    """The CUDA program's algorithm, step by step (``ring_buddy_cuda.model``): the reference's local ids and stats
    on every rank, each slot run by exactly one rank, the tables of the local ids (counts, CTA offsets, ``c2b`` /
    ``c2m``, ``nnc`` / ``tnp`` as the Triton program writes them), every permuted row back to its token, and an
    expert's rows dense and in token order."""
    rbc = _ring_buddy_cuda()
    tile = 8 if rows <= 64 else 16  # ring_buddy._prerouted_tile
    cases = [(1, 0.0), (2, 0.6), (3, 0.95)]
    for seed, skew in cases:
        flat = _rb_routing(rows, 1000 * rows + seed, skew)
        owners = [0] * len(flat)
        for rank in range(8):
            m = rbc.model(flat, rank, tile)
            local, stats = _rb_reference(flat, rank)
            assert m["local"] == local and m["stats"] == stats, (rows, seed, rank)
            mine = [s for s, e in enumerate(local) if e < 64]
            for s in mine:
                owners[s] += 1
            counts = [local.count(k) for k in range(64)]
            tiles = [(c + tile - 1) // tile for c in counts]
            nnc = sum(tiles)
            offs = [sum(tiles[:k]) for k in range(64)]
            assert m["nnc"] == nnc and m["tnp"] == nnc * tile
            assert m["c2b"] == [k for k in range(64) for _ in range(tiles[k])]
            assert m["c2m"] == [
                min((cta + 1) * tile, offs[k] * tile + counts[k])
                for k in range(64)
                for cta in range(offs[k], offs[k] + tiles[k])
            ]
            assert [p >= 0 for p in m["e2p"]] == [e < 64 for e in local]
            assert sorted(m["p2t"]) == sorted(m["e2p"][s] for s in mine)
            for s in mine:
                assert m["p2t"][m["e2p"][s]] == s // 8
            for k in range(64):
                ps = [m["e2p"][s] for s in mine if local[s] == k]
                assert ps == list(range(offs[k] * tile, offs[k] * tile + counts[k])), (rows, rank, k)
        assert owners == [1] * len(flat), (rows, seed)  # every (token, slot) runs on exactly one rank


def test_rb_tables_model_covers_an_expert_every_row_selects():
    """128 rows that all select the same 8 experts at 8-token tiles: 16 tiles per expert, the most the
    one-thread-per-(expert, tile) step covers."""
    rbc = _ring_buddy_cuda()
    flat = [3, 40, 77, 100, 150, 200, 230, 255] * 128
    for rank in range(8):
        m = rbc.model(flat, rank, 8)
        local, stats = _rb_reference(flat, rank)
        assert m["local"] == local and m["stats"] == stats
        counts = [local.count(k) for k in range(64)]
        assert max(counts) in (0, 128)
        assert m["nnc"] == sum((c + 7) // 8 for c in counts) and len(m["c2b"]) == m["nnc"]
    assert rbc.MAX_TILES == 16


def _place_tile_reference():
    tree = ast.parse(_RING_BUDDY_ASSIGN.read_text())
    names = {"place_tile_reference", "balance_ring", "expert_cost", "block_of"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    host = _module(
        "_place_tile_host", List=list, Tuple=tuple, EXPERTS=256, RING_SIZES={8: 32, 4: 64}, _TILE_COST=(20, 11)
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_RING_BUDDY_ASSIGN), "exec"), host.__dict__)
    return host.place_tile_reference


@pytest.mark.parametrize("rows", [41, 48, 56, 64])
@pytest.mark.parametrize("cost", [(20, 11), (14, 9), (30, 0)])
def test_rb_tables_model_tile_placement_equals_the_reference(rows, cost):
    """tilew (``tep8_tile_placement``, 41..64 rows at 8-token tiles, the bs6-8 band): the model's per-slot rank equals
    ``place_tile_reference``'s (the Triton TILEW rule in plain Python), its stats count the chosen rule's movers,
    and its tables are those of its local ids."""
    rbc = _ring_buddy_cuda()
    place = _place_tile_reference()
    tile, ranks, size = 8, 8, 32
    for seed, skew in ((1, 0.0), (2, 0.6), (3, 0.9), (4, 0.97)):
        flat = _rb_routing(rows, 31 * rows + seed + cost[0], skew)
        runs_on = place(flat, ranks, tile, cost)
        present = sorted(set(flat))
        load = [sum(1 for e in present if e // size == b) for b in range(ranks)]
        moved = {e for e, r in zip(flat, runs_on) if r != e // size}
        xs = [sum(1 for e in moved if e // size == b) for b in range(ranks)]
        stats = load + [load[r] - xs[r] + xs[(r - 1) % ranks] for r in range(ranks)]
        for rank in range(ranks):
            m = rbc.model(flat, rank, tile, tilew=True, cost=cost)
            local = [
                (e % size if e // size == rank else size + e % size) if r == rank else 2 * size + e % (256 - 2 * size)
                for e, r in zip(flat, runs_on)
            ]
            assert m["local"] == local and m["stats"] == stats, (rows, cost, seed, rank)
            counts = [local.count(k) for k in range(64)]
            assert m["nnc"] == sum((c + tile - 1) // tile for c in counts)
# role_env: the TEP4 ctx role's shared-path environment defaults (prefix-aware scheduling skip)
# ---------------------------------------------------------------------------
_PSKIP = {
    "TRTLLM_SCHED_PREFIX_SKIP_PENDING": "1",
    "TRTLLM_SCHED_PREFIX_SKIP_KEY_CACHE": "1",
    "TRTLLM_SCHED_PREFIX_SKIP_PRIORITIZE": "1",
}


class _RoleEnvPlan:
    def __init__(self, lever_table) -> None:
        self.levers = lever_table

    def __bool__(self) -> bool:
        return True


def test_the_ctx_role_defaults_the_prefix_aware_scheduling_skip(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt_llm.logger", _module("logger", logger=_LOG))
    role_env = importlib.import_module(f"{_PACKAGE}.role_env")
    _LOG.records.clear()
    assert dict(role_env.CTX_TEP4_ENV) == _PSKIP
    # A TEP4 context server: all three, whatever its max_batch_size.
    for mbs in (8, 32, 64):
        environ = {}
        chosen = role_env.apply(_RoleEnvPlan(table((107, 4), mbs, "ctx", "paged")), environ)
        assert chosen == _PSKIP and environ == _PSKIP, mbs
    assert any("role environment defaults for the ctx role" in m for m in _LOG.messages("info"))
    # A value the deployment set wins, "0" included; the others still get their default.
    environ = {"TRTLLM_SCHED_PREFIX_SKIP_PENDING": "0"}
    chosen = role_env.apply(_RoleEnvPlan(table((107, 4), 64, "ctx", "paged")), environ)
    assert environ["TRTLLM_SCHED_PREFIX_SKIP_PENDING"] == "0"
    assert chosen == {k: v for k, v in _PSKIP.items() if k != "TRTLLM_SCHED_PREFIX_SKIP_PENDING"}
    # Generation, aggregated (the rulers) and TEP8 servers: nothing.
    for key, mbs, role in (((107, 8), 16, "gen"), ((107, 8), 8, "gen"), ((107, 4), 64, "agg"), ((107, 8), 8, "agg")):
        environ = {}
        assert role_env.apply(_RoleEnvPlan(table(key, mbs, role, "paged")), environ) == {}, (key, role)
        assert environ == {}, (key, role)
    # No plan / no lever table (NO_PATHS): nothing.
    assert role_env.apply(None, {}) == {}
    assert role_env.apply(_RoleEnvPlan(None), {}) == {}


def test_the_model_applies_the_role_environment_after_the_lever_settings():
    """The model constructor calls role_env.apply right after lever_settings.apply, i.e. after compose and
    before the executor builds its scheduler (which reads these variables)."""
    text = (_TARGET / "modeling.py").read_text()
    settings = text.index("_lever_settings.apply(plan)")
    role = text.index("_role_env.apply(plan)")
    assert settings < role < settings + 300


# ---------------------------------------------------------------------------
# head_sampler_coop_fold__TOPP_ROUNDS: the split sampler's top-p-only rows decided by the multi-CTA top-p test
# (head_sampler.draw_topp), default off
# ---------------------------------------------------------------------------


def test_the_coop_fold_topp_rounds_defaults_to_off_up_to_8_requests():
    # Off on a TEP8 generation server of 1..8 requests; 2 rounds at 9..16 requests (int5 winner).
    for mbs, rounds in ((8, 0), (16, 2)):
        lever_table = table((107, 8), mbs, "gen", "paged")
        assert lever_table.on("head_sampler_coop_fold"), mbs
        assert lever_table.param("head_sampler_coop_fold", "TOPP_ROUNDS") == rounds
        values, defaults = _settings(lever_table)["ops.spec.head_sampler"]
        assert values["topp_rounds"] == rounds and defaults["topp_rounds"] == 0






# ---------------------------------------------------------------------------
# head_sampler_coop_fold__DIST_MIN_ROWS: the distributed sampler (head_sampler.ds_sample), default off
# ---------------------------------------------------------------------------


def test_the_coop_fold_dist_min_rows_defaults_to_off_up_to_8_requests():
    # Off on a TEP8 generation server of 1..8 requests; from 16 rows at 9..16 requests (int5 winner).
    for mbs, floor in ((8, 0), (16, 16)):
        lever_table = table((107, 8), mbs, "gen", "paged")
        assert lever_table.param("head_sampler_coop_fold", "DIST_MIN_ROWS") == floor
        values, defaults = _settings(lever_table)["ops.spec.head_sampler"]
        assert values["dist_min_rows"] == floor and defaults["dist_min_rows"] == 0


@pytest.mark.parametrize("floor", [8, 16, 64, 128])
def test_the_coop_fold_dist_min_rows_knob_reaches_the_head_sampler(floor):
    base = table((107, 8), 16, "gen", "paged")
    moved = table((107, 8), 16, "gen", "paged", knobs(head_sampler_coop_fold__DIST_MIN_ROWS=floor))
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    values, defaults = _settings(moved)["ops.spec.head_sampler"]
    assert values["dist_min_rows"] == floor and defaults["dist_min_rows"] == 0
    assert values["topp_rounds"] == 2  # the int5 default at 9..16 requests
    off = table(
        (107, 8),
        16,
        "gen",
        "paged",
        knobs(head_sampler_coop_fold="off", head_sampler_coop_fold__DIST_MIN_ROWS=floor),
    )
    values, defaults = _settings(off)["ops.spec.head_sampler"]
    assert values["dist_min_rows"] == defaults["dist_min_rows"] == 0




# ---------------------------------------------------------------------------
# tep8_pool_programmatic_fork__NORM_PREFETCH: the pool branch's norm + NVFP4 kernels load the norm gain and the
# global scale before their dependency wait / poll
# ---------------------------------------------------------------------------

_POOL_NORM_KERNELS = (_TARGET / "ops/comm/pool_publish_fp4quant.py", _TARGET / "ops/comm/pool_gather_fp4quant.py")


def test_the_norm_prefetch_defaults_to_off_up_to_8_requests():
    # Off on a TEP8 generation server of 1..8 requests; on at 9..16 requests (int5 winner).
    for mbs, arena, on in ((8, "paged", False), (16, "paged", True)):
        lever_table = table((107, 8), mbs, "gen", arena)
        if not lever_table.on("tep8_pool_programmatic_fork"):
            continue
        assert lever_table.param("tep8_pool_programmatic_fork", "NORM_PREFETCH") == int(on)
        settings = _settings(lever_table)
        for module in ("ops.comm.pool_publish_fp4quant", "ops.comm.pool_gather_fp4quant"):
            assert settings[module][0]["norm_prefetch"] is on and settings[module][1]["norm_prefetch"] is False


def test_the_norm_prefetch_knob_reaches_both_kernels_only():
    base = table((107, 8), 8, "gen", "paged")
    moved = table((107, 8), 8, "gen", "paged", knobs(tep8_pool_programmatic_fork__NORM_PREFETCH=1))
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    settings, base_settings = _settings(moved), _settings(base)
    for module in ("ops.comm.pool_publish_fp4quant", "ops.comm.pool_gather_fp4quant"):
        assert settings[module][0]["norm_prefetch"] is True
    for module in settings:
        if module not in ("ops.comm.pool_publish_fp4quant", "ops.comm.pool_gather_fp4quant"):
            assert settings[module] == base_settings[module], module
    off = table(
        (107, 8), 8, "gen", "paged",
        knobs(tep8_pool_programmatic_fork="off", tep8_pool_programmatic_fork__NORM_PREFETCH=1),
    )
    for module in ("ops.comm.pool_publish_fp4quant", "ops.comm.pool_gather_fp4quant"):
        assert _settings(off)[module][0]["norm_prefetch"] is False


@pytest.mark.parametrize("bad", [2, -1])
def test_the_norm_prefetch_refuses_other_values(bad):
    with pytest.raises(ValueError, match="NORM_PREFETCH"):
        table((107, 8), 8, "gen", "paged", knobs(tep8_pool_programmatic_fork__NORM_PREFETCH=bad))


def _norm_host(path):
    tree = ast.parse(path.read_text())
    body = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in {"configure", "_log_prefetch"})
        or (isinstance(n, ast.Assign) and _target_names(n) <= {"_NORM_PREFETCH", "_PREFETCH_LOGGED"})
    ]
    host = _module(f"_norm_host_{path.stem}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), host.__dict__)
    return host


@pytest.mark.parametrize("path", _POOL_NORM_KERNELS, ids=lambda p: p.stem)
def test_the_norm_prefetch_configure_and_compile_key(path):
    host = _norm_host(path)
    assert host._NORM_PREFETCH is False
    host.configure(norm_prefetch=True)
    assert host._NORM_PREFETCH is True
    host.configure()
    assert host._NORM_PREFETCH is False
    text = path.read_text()
    tree = ast.parse(text)
    (compile_fn,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_compile"]
    assert "prefetch: bool = False" in ast.unparse(compile_fn.args) or compile_fn.args.args[-1].arg == "prefetch"
    assert "_log_prefetch(" in ast.unparse(compile_fn)  # the engagement line when the prefetched kernel builds
    # the op passes the setting as the last _compile key
    call = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_compile"]
    assert len(call) == 1 and ast.unparse(call[0].args[-1]) == "_NORM_PREFETCH"


def _kernel_body(text):
    i = text.index("        @cute.kernel\n        def kernel(")
    return text[i : text.index("    @functools.lru_cache", i)]


@pytest.mark.parametrize("path", _POOL_NORM_KERNELS, ids=lambda p: p.stem)
def test_the_norm_prefetch_source(path):
    """The prefetch reads only parameters (gain, global scale) before the dependency wait; the prefetched quantization
    loop is the served loop with its gain loads replaced by the prefetched words (and unrolled); the served loop is the
    else branch (the default build is the served kernel)."""
    body = _kernel_body(path.read_text())
    pre = body.index("if cutlass.const_expr(self.prefetch):")
    wait = body.index("cute.arch.griddepcontrol_wait()")
    assert pre < wait
    prologue = body[pre:wait]
    assert "_fp4.ld_global_v4_u32(wp0)" in prologue and "_fp4.ld_global_v4_u32(wp1)" in prologue
    assert "gs_pre = mGlobalScale[0]" in prologue
    assert re.findall(r"get_ptr_as_int64\((\w+)", prologue) == ["mW", "mW"]  # only the gain is read
    gsv = body.index("global_scale_val = gs_pre")
    assert body.index("global_scale_val = mGlobalScale[0]", gsv) > gsv  # the served load in the else branch
    i = body.index("# NORM_PREFETCH: the same blocks and operations, unrolled, the gain from w_pre")
    j = body.index("                else:\n", i)
    k = body.index("            if enable_pdl:\n                cute.arch.griddepcontrol_launch_dependents()", j)
    fast = [ln.strip() for ln in body[i:j].splitlines()[1:] if ln.strip()]
    served = [ln.strip() for ln in body[j:k].splitlines()[1:] if ln.strip()]
    assert served[0].startswith("num_sf_per_thread = ") and served[1] == "for sf_iter in range(num_sf_per_thread):"
    assert fast[0] == "for sf_iter in cutlass.range_constexpr(self.sf_per_thread):"
    gain_loads = [
        "w_ptr0 = _fp4.get_ptr_as_int64(mW, block_start)",
        "w_ptr1 = _fp4.get_ptr_as_int64(mW, block_start + Int32(8))",
        "w_h2[0], w_h2[1], w_h2[2], w_h2[3] = _fp4.ld_global_v4_u32(w_ptr0)",
        "w_h2[4], w_h2[5], w_h2[6], w_h2[7] = _fp4.ld_global_v4_u32(w_ptr1)",
    ]
    served_ops = [ln for ln in served[2:] if ln != "if sf_idx < num_sf_blocks_per_row:" and ln not in gain_loads]
    fast_ops = [ln for ln in fast[1:] if ln not in ("for k in cutlass.range_constexpr(8):", "w_h2[k] = w_pre[8 * sf_iter + k]")]
    assert fast_ops == served_ops
    assert all(g in served for g in gain_loads)


# ---------------------------------------------------------------------------
# tep8_finalize_transport_64__INDEX_PREFETCH and the finalize timer stamps
# ---------------------------------------------------------------------------

_FINALIZE_TRANSPORT = _TARGET / "ops/moe/finalize_allreduce_transport.py"
_FIN_HOST_NAMES = frozenset({"TP8_MAX_TOKENS", "TP8_LAYOUTS", "DEFAULT", "ENTRY_TRIGGER", "STAMP_STAGES"})


def _finalize_host() -> types.ModuleType:
    """``finalize_allreduce_transport``'s ``Transport``, ``configure`` and settings, run from its file (the module
    imports torch and the CuTe DSL) in a module of their own."""
    tree = ast.parse(_FINALIZE_TRANSPORT.read_text())
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name == "Transport")
        or (isinstance(node, ast.FunctionDef) and node.name == "configure")
        or (isinstance(node, (ast.Assign, ast.AnnAssign)) and _push_target_names(node) <= _FIN_HOST_NAMES)
    ]
    import dataclasses as _dataclasses

    host = _module("_finalize_host", dataclasses=_dataclasses)
    sys.modules["_finalize_host"] = host  # dataclasses resolves the class's module by name
    try:
        exec(compile(ast.Module(body=body, type_ignores=[]), str(_FINALIZE_TRANSPORT), "exec"), host.__dict__)
    finally:
        sys.modules.pop("_finalize_host", None)
    assert _FIN_HOST_NAMES | {"Transport", "configure"} <= set(vars(host)), "a name moved in the finalize transport"
    return host


def _finalize_kernel_text() -> str:
    """The finalize kernel's source as ``ast.unparse`` writes it, with tuple assignment targets unparenthesized
    (Python 3.9 writes ``(a, b) = ...``, 3.11+ ``a, b = ...``)."""
    tree = ast.parse(_FINALIZE_TRANSPORT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "kernel":
            return re.sub(r"\(((?:\w+, )+\w+)\) = ", r"\1 = ", ast.unparse(node))
    raise AssertionError("the finalize kernel moved")


def test_the_finalize_index_prefetch_defaults_to_off():
    for mbs, arena in ((8, "paged"), (16, "paged")):
        lever_table = table((107, 8), mbs, "gen", arena)
        if not lever_table.on("tep8_finalize_transport_64"):
            continue
        assert lever_table.param("tep8_finalize_transport_64", "INDEX_PREFETCH") == 0
        assert _settings(lever_table)["ops.moe.finalize_allreduce_transport"][0]["index_prefetch"] is False
    fin = _finalize_host()
    assert fin.DEFAULT == fin.Transport() and not fin.DEFAULT.prefetch and fin.DEFAULT.stamps == 0
    assert fin.DEFAULT.label() == "multicast/red/warp/late"
    assert fin.ENTRY_TRIGGER.label() == "multicast/red/warp/entry"
    fin.configure(tep8_max_tokens=128, trigger="entry")
    assert fin.DEFAULT == fin.Transport(trigger="entry")  # the transport served before


def test_the_finalize_index_prefetch_knob_reaches_the_default_transport():
    base = table((107, 8), 8, "gen", "paged")
    moved = table((107, 8), 8, "gen", "paged", knobs(tep8_finalize_transport_64__INDEX_PREFETCH=1))
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    call = _settings(moved)["ops.moe.finalize_allreduce_transport"][0]
    assert call["index_prefetch"] is True
    assert call["tep8_max_tokens"] == _settings(base)["ops.moe.finalize_allreduce_transport"][0]["tep8_max_tokens"]
    off = table(
        (107, 8), 8, "gen", "paged",
        knobs(tep8_finalize_transport_64="off", tep8_finalize_transport_64__INDEX_PREFETCH=1),
    )
    values, defaults = _settings(off)["ops.moe.finalize_allreduce_transport"]
    assert values["index_prefetch"] is defaults["index_prefetch"] is False
    fin = _finalize_host()
    fin.configure(tep8_max_tokens=128, trigger="entry", index_prefetch=True)
    assert fin.DEFAULT == fin.Transport(trigger="entry", prefetch=True)
    assert fin.DEFAULT.label() == "multicast/red/warp/entry/prefetch"
    assert fin.TP8_MAX_TOKENS == 128


@pytest.mark.parametrize("bad", [2, -1, "x"])
def test_the_finalize_index_prefetch_knob_refuses_other_values(bad):
    with pytest.raises(ValueError, match="INDEX_PREFETCH"):
        table((107, 8), 8, "gen", "paged", knobs(tep8_finalize_transport_64__INDEX_PREFETCH=bad))


def test_finalize_transport_validates_prefetch_and_stamps():
    fin = _finalize_host()
    assert fin.Transport(stamps=4096).label().endswith("/stamps")
    for bad in (dict(prefetch=1), dict(stamps=-8), dict(stamps=12), dict(stamps=1.0)):
        with pytest.raises(ValueError):
            fin.Transport(**bad)
    assert len(fin.STAMP_STAGES) == 8 and fin.STAMP_STAGES[1] == "dep wait"


def test_finalize_prefetch_reads_the_row_map_before_the_wait_and_verifies_after():
    """With ``prefetch`` the row map is read before ``griddepcontrol_wait``; after it the rows of those entries
    load beside a second read of the map, and a changed entry reloads its row through ``_reload_row_x4`` (load when
    the new entry is valid, zeros when not): the words used are the post-wait entries'. Without it the original
    statements run in their order."""
    text = _finalize_kernel_text()
    wait = text.index("cute.arch.griddepcontrol_wait()")
    early = text.index("early_index.append(self.select_row(indices, rows, token, k)[0])")
    assert text.index("if cutlass.const_expr(self.prefetch):") < early < wait
    after = text[wait:]
    verify = after.index("index, valid = self.select_row(indices, rows, token, k)")
    early_rows = after.index("row_words.append(_load_row_x4(gemm_address + offset * 2, early_valid))")
    reload = after.index("_reload_row_x4(gemm_address + offset * 2, row_valid[k] & changed, changed")
    assert early_rows < verify < reload
    assert "changed = Uint32(row_index[k] != early_index[k])" in after
    # the default path keeps today's statements, in order
    default = after[after.index("else:", reload):]
    seq = [
        "index, valid = self.select_row(indices, rows, token, k)",
        "row_weight.append(_load_f32(weights.toint() + (selection + k) * 4))",
        "address = gemm_address + offset * 2",
        "row_words.append(_load_row_x4(address, valid))",
    ]
    positions = [default.index(s) for s in seq]
    assert positions == sorted(positions)


def _reload_model(old, load, replace, memory):
    """``_reload_row_x4``'s PTX: the given words, zeros under ``replace``, the load under ``load``."""
    words = list(old)
    if replace:
        words = [0, 0, 0, 0]
    if load:
        words = list(memory)
    return words


@pytest.mark.parametrize("early_valid", [False, True])
@pytest.mark.parametrize("same", [False, True])
@pytest.mark.parametrize("new_valid", [False, True])
def test_finalize_prefetch_correction_gives_the_post_wait_entry_words(early_valid, same, new_valid):
    if same and early_valid != new_valid:
        pytest.skip("one entry, one validity")
    row_early, row_new = [1, 2, 3, 4], ([1, 2, 3, 4] if same else [5, 6, 7, 8])
    early_words = row_early if early_valid else [0, 0, 0, 0]  # _load_row_x4: zeros when not valid
    changed = not same
    got = _reload_model(early_words, new_valid and changed, changed, row_new)
    expected = row_new if new_valid else [0, 0, 0, 0]  # the default path's words for the post-wait entry
    assert got == expected
    src = _FINALIZE_TRANSPORT.read_text()
    i = src.index("def _reload_row_x4(")
    asm = src[i : src.index("return _split_words", i)]
    q, z, p, ld = (asm.index(t) for t in ("setp.ne.u32 q, $6, 0", "@q mov.b32 a, 0", "setp.ne.u32 p, $5, 0", "@p ld.global.v4.b32"))
    assert q < z < p < ld


def test_finalize_stamps_are_compiled_out_by_default():
    text = _finalize_kernel_text()
    assert text.count("_stamp(stamp_base") == 8
    # every stamp sits under the compile-time stamps switch, issued by thread 0
    assert text.count("if cutlass.const_expr(self.stamps != 0):") == 8
    src = _FINALIZE_TRANSPORT.read_text()
    assert "self.stamps = int(transport.stamps)" in src and "self.prefetch = transport.prefetch" in src


# --------------------------------------------------------------------------
# The TEP8 one-cluster SWA block, one cluster per request (swa_one_cluster_block__TEP8_WIDE_MIN_REQUESTS)
# --------------------------------------------------------------------------
#: md5 of ``swa_block._CL8_SRC`` (swa_cl8, the per-(request, KV head) build) as the base served it:
#: the wide build leaves it, and so every step it keeps, untouched.
_CL8_SRC_MD5 = "10334d867210ce7cb83cb59337e2eb51"




















# ---------------------------------------------------------------------------
# head_sampler_coop_fold__TOPP_ROUNDS: the split sampler's top-p-only rows decided by the multi-CTA top-p test
# (head_sampler.draw_topp), default off
# ---------------------------------------------------------------------------




@pytest.mark.parametrize("rounds", [1, 2, 3, 4])
def test_the_coop_fold_topp_rounds_knob_reaches_the_head_sampler(rounds):
    base = table((107, 8), 16, "gen", "paged")
    moved = table((107, 8), 16, "gen", "paged", knobs(head_sampler_coop_fold__TOPP_ROUNDS=rounds))
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    values, defaults = _settings(moved)["ops.spec.head_sampler"]
    assert values["topp_rounds"] == rounds and defaults["topp_rounds"] == 0
    # The vocabulary-parallel head's floor is untouched.
    assert values["vp_min_rows"] == _settings(base)["ops.spec.head_sampler"][0]["vp_min_rows"]
    # The coop fold lever off: the knob does not reach the module.
    off = table(
        (107, 8),
        16,
        "gen",
        "paged",
        knobs(head_sampler_coop_fold="off", head_sampler_coop_fold__TOPP_ROUNDS=rounds),
    )
    values, defaults = _settings(off)["ops.spec.head_sampler"]
    assert values["topp_rounds"] == defaults["topp_rounds"] == 0


@pytest.mark.parametrize("bad", [5, -1])
def test_the_coop_fold_topp_rounds_refuses_other_values(bad):
    with pytest.raises(ValueError, match="TOPP_ROUNDS"):
        table((107, 8), 16, "gen", "paged", knobs(head_sampler_coop_fold__TOPP_ROUNDS=bad))


# ---------------------------------------------------------------------------
# head_sampler_coop_fold__DIST_MIN_ROWS: the distributed sampler (head_sampler.ds_sample), default off
# ---------------------------------------------------------------------------






@pytest.mark.parametrize("bad", [129, -1])
def test_the_coop_fold_dist_min_rows_refuses_other_values(bad):
    with pytest.raises(ValueError, match="DIST_MIN_ROWS"):
        table((107, 8), 16, "gen", "paged", knobs(head_sampler_coop_fold__DIST_MIN_ROWS=bad))


# --------------------------------------------------------------------------
# tep8_drafter_wide_gemms__MAX_ROWS / __WIDE_TILE / __SITES: the 65..128-row TP8 drafter GEMMs (9..16
# requests at K = 7) in one launch of the 64-row split-K tactic instead of cuBLAS
# --------------------------------------------------------------------------
_FIXED_GEMM = "ops.gemm.fixed_low_m_gemm"
#: The three knobs' configure keywords and their defaults (the served band).
_DRAFTER_WIDE_DEFAULTS = {
    "tep8_drafter_max_rows": 64,
    "tep8_drafter_wide_tile": "64x128",
    "tep8_drafter_wide_sites": ("conv_coef", "qkv"),
}


def test_the_drafter_wide_max_rows_parameters_default_to_the_served_band():
    """MAX_ROWS 64, WIDE_TILE 64x128 and SITES conv_coef,qkv on every TEP8 generation table that
    serves the drafter's wide tactics, and ``fixed_low_m_gemm.configure`` gets its defaults (above 64
    rows the drafter keeps cuBLAS); off (a server above eight requests under auto) too."""
    for mbs, arena in ((8, "windowed"), (8, "paged"), (4, "windowed")):
        base = table((107, 8), mbs, "gen", arena)
        assert base.on("tep8_drafter_wide_gemms") and base.on("tep8_wide64_splitk"), (mbs, arena)
        assert base.param("tep8_drafter_wide_gemms", "MAX_ROWS") == 64, (mbs, arena)
        assert base.param("tep8_drafter_wide_gemms", "WIDE_TILE") == "64x128", (mbs, arena)
        assert base.param("tep8_drafter_wide_gemms", "SITES") == ("conv_coef", "qkv")
        values, defaults = _settings(base)[_FIXED_GEMM]
        assert values["tep8_drafter_wide"] and values["tep8_wide64"]
        for key, default in _DRAFTER_WIDE_DEFAULTS.items():
            assert values[key] == defaults[key] == default, (key, mbs, arena)
    off = table((107, 8), 16, "agg", "windowed")
    assert not off.on("tep8_drafter_wide_gemms")
    values, defaults = _settings(off)[_FIXED_GEMM]
    assert values == defaults
    settings = importlib.import_module(f"{_PACKAGE}.lever_settings")
    assert settings.TEP8_DRAFTER_WIDE_SITES == _DRAFTER_WIDE_DEFAULTS["tep8_drafter_wide_sites"]


@pytest.mark.parametrize(
    "rows, tile, sites",
    [
        (72, "64x64", "conv_coef"),
        (96, "128x64", "conv_coef,qkv,gate_up"),
        (128, "64x64", "gate_up,qkv,conv_coef,o_proj,down"),
        (128, "64x128", "qkv"),
        (65, "128x64", "down,o_proj"),
    ],
)
def test_the_drafter_wide_max_rows_knobs_reach_only_the_fixed_gemm_table(rows, tile, sites):
    """At max_batch_size 16 with the low-latency levers on by knob, MAX_ROWS / WIDE_TILE / SITES turn no
    lever on or off and change no path; they reach ``fixed_low_m_gemm.configure`` (and nothing else)
    only while ``tep8_drafter_wide_gemms`` is on; SITES keeps the parameter's choice order."""
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "windowed", ll16)
    set_ = knobs(
        tep8_drafter_wide_gemms__MAX_ROWS=rows,
        tep8_drafter_wide_gemms__WIDE_TILE=tile,
        tep8_drafter_wide_gemms__SITES=sites,
    )
    moved = table((107, 8), 16, "gen", "windowed", {**ll16, **set_})
    assert moved.paths() == plain.paths()
    assert rule_levers_on(moved) == rule_levers_on(plain)
    order = ("conv_coef", "qkv", "o_proj", "down", "gate_up")
    want_sites = tuple(site for site in order if site in sites.split(","))
    assert moved.param("tep8_drafter_wide_gemms", "MAX_ROWS") == rows
    assert moved.param("tep8_drafter_wide_gemms", "WIDE_TILE") == tile
    assert moved.param("tep8_drafter_wide_gemms", "SITES") == want_sites
    calls, before = _settings(moved), _settings(plain)
    fixed = calls[_FIXED_GEMM][0]
    assert fixed["tep8_wide64"] and fixed["tep8_drafter_wide"]
    assert calls["ops.gemm.splitk_gemm"][0]["wide64"]  # run_splitk_dense_tiles needs it
    assert fixed["tep8_drafter_max_rows"] == rows
    assert fixed["tep8_drafter_wide_tile"] == tile
    assert fixed["tep8_drafter_wide_sites"] == want_sites
    assert {m: c for m, c in calls.items() if m != _FIXED_GEMM} == {
        m: c for m, c in before.items() if m != _FIXED_GEMM
    }
    assert {k: v for k, v in fixed.items() if k not in _DRAFTER_WIDE_DEFAULTS} == {
        k: v for k, v in before[_FIXED_GEMM][0].items() if k not in _DRAFTER_WIDE_DEFAULTS
    }
    # The lever off (max_batch_size 16 under auto): the knobs reach nothing.
    off = table((107, 8), 16, "agg", "windowed", set_)
    assert not off.on("tep8_drafter_wide_gemms")
    values, defaults = _settings(off)[_FIXED_GEMM]
    for key, default in _DRAFTER_WIDE_DEFAULTS.items():
        assert values[key] == defaults[key] == default, key


@pytest.mark.parametrize(
    "knob, value, message",
    [
        ("tep8_drafter_wide_gemms__MAX_ROWS", 63, "expected an integer in 64..128"),
        ("tep8_drafter_wide_gemms__MAX_ROWS", 129, "expected an integer in 64..128"),
        ("tep8_drafter_wide_gemms__WIDE_TILE", "128x128", "expected one of"),
        ("tep8_drafter_wide_gemms__WIDE_TILE", "64x64,64x128", "expected one of"),
        ("tep8_drafter_wide_gemms__SITES", "fc", "non-empty comma-separated subset"),
        ("tep8_drafter_wide_gemms__SITES", "", "non-empty comma-separated subset"),
    ],
)
def test_the_drafter_wide_max_rows_parameters_refuse_values_outside_their_kernel(knob, value, message):
    with pytest.raises(ValueError, match=message):
        table((107, 8), 8, "gen", "windowed", knobs(**{knob: value}))


def _fixed_gemm_host(**stubs) -> types.ModuleType:
    return _host_module(
        _TARGET / "ops/gemm/fixed_low_m_gemm.py",
        (
            "SM107_SPLITK_TACTICS",
            "_TP8_DRAFTER_WIDE_SHAPES",
            "_TP8_DRAFTER_WIDE_TILES",
            "SM107_TP8_DRAFTER_WIDE_TACTICS",
            "SM107_TP8_FC_WIDE_TACTICS",
            "TEP8_DRAFTER_MAX_ROWS",
            "TEP8_DRAFTER_WIDE_LIMIT",
            "TEP8_DRAFTER_WIDE_TILES",
            "TEP8_DRAFTER_WIDE_TILE",
            "TEP8_DRAFTER_WIDE_SITES",
            "ONE_LAUNCH_ROWS",
            "TEP8_WIDE64",
            "TEP8_DRAFTER_WIDE",
            "TEP8_FC_WIDE",
            "TEP8_DRAFTER_WIDE64",
            "_SM107_SPLITK_TACTICS_BASE",
            "MAX_M",
            "_TP8_DRAFTER_DEEP_STAGES",
            "TEP8_DRAFTER_DEEP_STAGES",
        ),
        ("configure", "wide_launch_tactics"),
        **stubs,
    )


def test_fixed_gemm_tep8_drafter_max_rows_table():
    """``tep8_drafter_max_rows`` above 64 lists every 65..max_rows row count of each SITES shape with
    the site's 64-row tactic on WIDE_TILE (split_k and stages the 64-row tactic's; ``one_launch_tactic``
    lowers stages at launch), and ``MAX_M`` max_rows; no other entry moves (the fc shard and the other
    sites keep F.linear above 64 rows). The default (64) lists nothing above 64 and MAX_M stays 64."""
    host = _fixed_gemm_host()
    assert host.TEP8_DRAFTER_MAX_ROWS == host.ONE_LAUNCH_ROWS == 64
    assert host.TEP8_DRAFTER_WIDE_LIMIT == 128 and host.TEP8_DRAFTER_WIDE_TILE == (64, 128)
    assert host.TEP8_DRAFTER_WIDE_SITES == ("conv_coef", "qkv")
    host.configure(tep8_wide64=True, tep8_drafter_wide=True, tep8_fc_wide=True)
    served = dict(host.SM107_SPLITK_TACTICS)
    assert max(m for m, _, _ in served) == 64 and host.MAX_M == 64
    for tile_name, tile in host.TEP8_DRAFTER_WIDE_TILES.items():
        for max_rows, sites in ((96, ("conv_coef", "qkv", "gate_up")), (128, ("o_proj",))):
            host.configure(
                tep8_wide64=True,
                tep8_drafter_wide=True,
                tep8_fc_wide=True,
                tep8_drafter_max_rows=max_rows,
                tep8_drafter_wide_tile=tile_name,
                tep8_drafter_wide_sites=sites,
            )
            table_ = host.SM107_SPLITK_TACTICS
            assert host.MAX_M == max_rows
            assert {key: t for key, t in table_.items() if key[0] <= 64} == served
            wide = {key: t for key, t in table_.items() if key[0] > 64}
            assert len(wide) == (max_rows - 64) * len(sites)
            for site in sites:
                k, n = host._TP8_DRAFTER_WIDE_SHAPES[site]
                base = host._TP8_DRAFTER_WIDE_TILES[64][site]
                for rows in range(65, max_rows + 1):
                    assert table_[(rows, k, n)] == (*tile, *base[2:]), (site, rows, tile_name)
            assert (max_rows, 40960, 1024) not in table_  # the fc shard: F.linear above 64 rows
            assert host.TEP8_DRAFTER_WIDE_TILE == tile and host.TEP8_DRAFTER_WIDE_SITES == sites
    host.configure()
    assert host.SM107_SPLITK_TACTICS == host._SM107_SPLITK_TACTICS_BASE and host.MAX_M == 32
    assert (host.TEP8_DRAFTER_MAX_ROWS, host.TEP8_DRAFTER_WIDE_TILE) == (64, (64, 128))


def test_fixed_gemm_tep8_drafter_max_rows_refusals():
    """A max_rows outside 64..128, above 64 without tep8_wide64 or tep8_drafter_wide, an unknown tile or
    an empty / unknown site list raises and sets nothing."""
    host = _fixed_gemm_host()
    on = dict(tep8_wide64=True, tep8_drafter_wide=True)
    host.configure(**on, tep8_drafter_max_rows=96, tep8_drafter_wide_tile="64x128")
    before = (dict(host.SM107_SPLITK_TACTICS), host.MAX_M, host.TEP8_DRAFTER_MAX_ROWS,
              host.TEP8_DRAFTER_WIDE_TILE, host.TEP8_DRAFTER_WIDE_SITES)
    for kwargs, match in (
        (dict(on, tep8_drafter_max_rows=63), "must be in 64..128"),
        (dict(on, tep8_drafter_max_rows=129), "must be in 64..128"),
        (dict(tep8_wide64=True, tep8_drafter_max_rows=72), "needs tep8_wide64 and tep8_drafter_wide"),
        (dict(tep8_drafter_wide=True, tep8_drafter_max_rows=72), "needs tep8_wide64 and tep8_drafter_wide"),
        (dict(on, tep8_drafter_wide_tile="128x128"), "tep8_drafter_wide_tile"),
        (dict(on, tep8_drafter_wide_sites=()), "tep8_drafter_wide_sites"),
        (dict(on, tep8_drafter_wide_sites=("qkv", "fc")), "tep8_drafter_wide_sites"),
    ):
        with pytest.raises(ValueError, match=match):
            host.configure(**kwargs)
        assert before == (dict(host.SM107_SPLITK_TACTICS), host.MAX_M, host.TEP8_DRAFTER_MAX_ROWS,
                          host.TEP8_DRAFTER_WIDE_TILE, host.TEP8_DRAFTER_WIDE_SITES), kwargs


def test_fixed_gemm_wide_launch_tactics_resolve_only_the_one_launch_rows():
    """``wide_launch_tactics`` (spec_worker's install) resolves one tactic per (K, N) of the 65..max_rows
    entries through ``one_launch_tactic``, and nothing (no split-K import) at the default band."""
    seen = []

    def fit(tactic, k, n):
        seen.append((tactic, k, n))
        return ("fit", tactic)

    host = _fixed_gemm_host(one_launch_tactic=fit)
    host.configure(tep8_wide64=True, tep8_drafter_wide=True)
    assert host.wide_launch_tactics(host.SM107_SPLITK_TACTICS) == {} and not seen
    host.configure(tep8_wide64=True, tep8_drafter_wide=True, tep8_drafter_max_rows=128)
    wide = host.wide_launch_tactics(host.SM107_SPLITK_TACTICS)
    assert set(wide) == {host._TP8_DRAFTER_WIDE_SHAPES[s] for s in ("conv_coef", "qkv")}
    for (k, n), (_, tactic) in wide.items():
        assert tactic[:2] == (64, 128), (k, n)


def test_the_drafter_deep_stages_knob_reaches_only_the_fixed_gemm_table():
    """tep8_drafter_wide_gemms__DEEP_STAGES (0..1, default 0) turns no lever on or off and changes no path;
    it reaches ``fixed_low_m_gemm.configure`` (tep8_drafter_deep_stages) and nothing else, only while
    tep8_drafter_wide_gemms is on."""
    for mbs, arena in ((8, "windowed"), (8, "paged")):
        base = table((107, 8), mbs, "gen", arena)
        assert base.param("tep8_drafter_wide_gemms", "DEEP_STAGES") == 0
        values, defaults = _settings(base)[_FIXED_GEMM]
        assert values["tep8_drafter_deep_stages"] is defaults["tep8_drafter_deep_stages"] is False
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "windowed", ll16)
    moved = table((107, 8), 16, "gen", "windowed", {**ll16, **knobs(tep8_drafter_wide_gemms__DEEP_STAGES=1)})
    assert moved.paths() == plain.paths() and rule_levers_on(moved) == rule_levers_on(plain)
    calls, before = _settings(moved), _settings(plain)
    assert calls[_FIXED_GEMM][0] == {**before[_FIXED_GEMM][0], "tep8_drafter_deep_stages": True}
    assert {m: c for m, c in calls.items() if m != _FIXED_GEMM} == {
        m: c for m, c in before.items() if m != _FIXED_GEMM
    }
    off = table((107, 8), 16, "agg", "windowed", knobs(tep8_drafter_wide_gemms__DEEP_STAGES=1))
    assert not off.on("tep8_drafter_wide_gemms")
    values, defaults = _settings(off)[_FIXED_GEMM]
    assert values == defaults
    for bad in (2, -1):
        with pytest.raises(ValueError, match="expected an integer in 0..1"):
            table((107, 8), 8, "gen", "windowed", knobs(tep8_drafter_wide_gemms__DEEP_STAGES=bad))


def test_fixed_gemm_deep_stages_move_only_stages_of_the_listed_entries():
    """With tep8_drafter_deep_stages the listed 16..64-row drafter entries take the listed stage counts and keep
    their tile and split_k (the same bits); every other entry (the 8-row cell, the fc shard, 65+ rows, the
    unlisted sites) is the plain table's; without tep8_drafter_wide the knob is refused and nothing is set."""
    host = _fixed_gemm_host()
    host.configure(tep8_wide64=True, tep8_drafter_wide=True, tep8_fc_wide=True, tep8_drafter_max_rows=128)
    plain = dict(host.SM107_SPLITK_TACTICS)
    host.configure(
        tep8_wide64=True, tep8_drafter_wide=True, tep8_fc_wide=True, tep8_drafter_max_rows=128,
        tep8_drafter_deep_stages=True,
    )
    deep = dict(host.SM107_SPLITK_TACTICS)
    assert host.TEP8_DRAFTER_DEEP_STAGES is True and set(deep) == set(plain)
    listed = {
        (rows, *host._TP8_DRAFTER_WIDE_SHAPES[site]): stages
        for rows, sites in host._TP8_DRAFTER_DEEP_STAGES.items()
        for site, stages in sites.items()
    }
    assert all(key in plain for key in listed)
    assert max(rows for rows, _, _ in listed) == 64 and min(rows for rows, _, _ in listed) == 16
    for key, tactic in deep.items():
        if key in listed:
            assert tactic[:3] == plain[key][:3] and tactic[3] == listed[key], key
            assert tactic[3] >= plain[key][3], key
        else:
            assert tactic == plain[key], key
    assert deep[(64, 8192, 7168)] == (64, 64, 1, 10) and deep[(64, 8192, 2048)] == (64, 64, 4, 8)
    host.configure(tep8_wide64=True, tep8_drafter_wide=True)
    assert host.TEP8_DRAFTER_DEEP_STAGES is False
    with pytest.raises(ValueError, match="DEEP_STAGES"):
        host.configure(tep8_wide64=True, tep8_fc_wide=True, tep8_drafter_deep_stages=True)
    assert host.TEP8_DRAFTER_DEEP_STAGES is False and host.SM107_SPLITK_TACTICS == {
        **host._SM107_SPLITK_TACTICS_BASE, **{k: v for k, v in host.SM107_TP8_DRAFTER_WIDE_TACTICS.items()}
    }
    src = (_TARGET / "ops/gemm/fixed_low_m_gemm.py").read_text()
    assert "def deep_stage_tactics(" in src and "validate_tactic(candidate, rows, n, k)" in src
    assert "tep8_drafter_wide_gemms__DEEP_STAGES: the {rows}-row {site} tactic" in src
    worker = (_TARGET / "spec_worker.py").read_text()
    assert "deep = deep_stage_tactics(tactics)" in worker
    assert "(tep8_drafter_wide_gemms__DEEP_STAGES): " in worker


# --------------------------------------------------------------------------
# wide_drafter_hops__TEP8_MAX_ROWS: 65..128-row TP8 drafter forwards on the wide fused one-shot hop
# instead of the built-in two-shot + convolution + fused add-RMSNorm
# --------------------------------------------------------------------------
_DRAFTER_HOPS = "kernels.drafter_hops.hops"


def test_the_wide_drafter_hops_tep8_max_rows_defaults_to_the_served_rows():
    """TEP8_MAX_ROWS 64 wherever wide_drafter_hops is on (TEP8 / TEP4, every max_batch_size), and the
    hops module is not configured (not imported for it); off under baseline at TEP8."""
    for key, mbs in (((107, 8), 8), ((107, 8), 16), ((107, 4), 8), ((107, 4), 48)):
        base = table(key, mbs, "gen", "windowed")
        assert base.on("wide_drafter_hops"), (key, mbs)
        assert base.param("wide_drafter_hops", "TEP8_MAX_ROWS") == 64, (key, mbs)
        values, defaults = _settings(base)[_DRAFTER_HOPS]
        stock = {"tep8_wide_max_rows": 64, "tep8_two_shot_min_rows": 0, "tep8_two_shot_max_rows": 128}
        assert defaults == stock, (key, mbs)
        # int5: a TEP8 generation server of 9..16 requests runs the two-shot hop on 25..64 rows.
        wide = key == (107, 8) and mbs == 16
        assert values == ({**stock, "tep8_two_shot_min_rows": 25, "tep8_two_shot_max_rows": 64} if wide else stock), (key, mbs)
    baseline = table((107, 8), 8, "gen", "windowed", BASELINE)
    assert not baseline.on("wide_drafter_hops")
    values, defaults = _settings(baseline)[_DRAFTER_HOPS]
    assert values == defaults


@pytest.mark.parametrize("rows", [72, 96, 128])
def test_the_wide_drafter_hops_tep8_max_rows_knob_reaches_only_the_hops(rows):
    """At max_batch_size 16 (the measured LL server), TEP8_MAX_ROWS turns no lever on or off and changes
    no path; it reaches the hops module's ``configure`` and nothing else."""
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "windowed", ll16)
    moved = table((107, 8), 16, "gen", "windowed", {**ll16, **knobs(wide_drafter_hops__TEP8_MAX_ROWS=rows)})
    assert moved.paths() == plain.paths()
    assert rule_levers_on(moved) == rule_levers_on(plain)
    assert moved.param("wide_drafter_hops", "TEP8_MAX_ROWS") == rows
    calls, before = _settings(moved), _settings(plain)
    # (the int5 two-shot band 25..64 at 9..16 requests rides along)
    assert calls[_DRAFTER_HOPS][0] == {"tep8_wide_max_rows": rows, "tep8_two_shot_min_rows": 25, "tep8_two_shot_max_rows": 64}
    assert {m: c for m, c in calls.items() if m != _DRAFTER_HOPS} == {
        m: c for m, c in before.items() if m != _DRAFTER_HOPS
    }


@pytest.mark.parametrize("value", [63, 129, "x"])
def test_the_wide_drafter_hops_tep8_max_rows_refuses_values_outside_64_to_128(value):
    with pytest.raises(ValueError, match="expected an integer in 64..128"):
        table((107, 8), 8, "gen", "windowed", knobs(wide_drafter_hops__TEP8_MAX_ROWS=value))


def test_drafter_hops_tep8_wide_max_rows_sizes_only_tep8_workspaces():
    """``configure(tep8_wide_max_rows)`` (64..128) is ``TEP8_WIDE_MAX_ROWS``; ``wide_rows`` gives a TP8
    drafter that many rows when above the device's served rows, a TP4 one the served rows; a value
    outside raises and sets nothing."""
    host = _host_module(
        _TARGET / "kernels/drafter_hops/hops.py",
        ("TEP8_WIDE_MAX_ROWS", "TEP8_WIDE_ROWS_BAND", "TEP8_TWO_SHOT_MIN_ROWS", "TWO_SHOT_MAX_ROWS", "TEP8_TWO_SHOT_MAX_ROWS"),
        ("configure", "wide_rows"),
    )
    assert host.TEP8_WIDE_MAX_ROWS == 64 and host.TEP8_WIDE_ROWS_BAND == (64, 128)
    assert host.wide_rows(64, 8) == host.wide_rows(64, 4) == 64
    for rows in (72, 96, 128):
        host.configure(tep8_wide_max_rows=rows)
        assert host.TEP8_WIDE_MAX_ROWS == rows
        assert host.wide_rows(64, 8) == rows and host.wide_rows(64, 4) == 64
    host.configure(tep8_wide_max_rows=64)
    assert host.wide_rows(64, 8) == 64
    host.configure(tep8_wide_max_rows=96)
    for bad in (63, 129):
        with pytest.raises(ValueError, match="tep8_wide_max_rows must be in 64..128"):
            host.configure(tep8_wide_max_rows=bad)
        assert host.TEP8_WIDE_MAX_ROWS == 96
    host.configure()
    assert host.TEP8_WIDE_MAX_ROWS == 64


# --------------------------------------------------------------------------
# tep8_drafter_wide_gemms__WINDOW_TILES / __WINDOW_MIN_ROWS: multi-tile drafter window-attention splits
# on the 9..16-request TEP8 steps (fewer programs, fewer merged partials)
# --------------------------------------------------------------------------
_WINDOW = "ops.spec.dflash_window_attention"


def test_the_drafter_window_tiles_default_to_the_one_request_geometry():
    """WINDOW_TILES 1 and WINDOW_MIN_ROWS 65 wherever tep8_drafter_wide_gemms is on, and the window
    attention module is not configured (not imported for it); off at mbs16 under auto too."""
    for mbs, arena in ((8, "windowed"), (8, "paged"), (4, "windowed")):
        base = table((107, 8), mbs, "gen", arena)
        assert base.param("tep8_drafter_wide_gemms", "WINDOW_TILES") == 1, (mbs, arena)
        assert base.param("tep8_drafter_wide_gemms", "WINDOW_MIN_ROWS") == 65, (mbs, arena)
        values, defaults = _settings(base)[_WINDOW]
        assert values == defaults == {"multi_tiles": 1, "multi_min_rows": 65}, (mbs, arena)
    values, defaults = _settings(table((107, 8), 16, "gen", "windowed"))[_WINDOW]
    assert values == defaults
    for mbs in (8, 48):
        values, defaults = _settings(table((107, 4), mbs, "gen", "windowed"))[_WINDOW]
        assert values == defaults, mbs  # TEP8-only lever


@pytest.mark.parametrize("tiles, min_rows", [(2, 65), (3, 65), (4, 97), (3, 9)])
def test_the_drafter_window_tiles_knobs_reach_only_the_window_attention(tiles, min_rows):
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "windowed", ll16)
    set_ = knobs(
        tep8_drafter_wide_gemms__WINDOW_TILES=tiles,
        tep8_drafter_wide_gemms__WINDOW_MIN_ROWS=min_rows,
    )
    moved = table((107, 8), 16, "gen", "windowed", {**ll16, **set_})
    assert moved.paths() == plain.paths()
    assert rule_levers_on(moved) == rule_levers_on(plain)
    calls, before = _settings(moved), _settings(plain)
    assert calls[_WINDOW][0] == {"multi_tiles": tiles, "multi_min_rows": min_rows}
    assert {m: c for m, c in calls.items() if m != _WINDOW} == {
        m: c for m, c in before.items() if m != _WINDOW
    }
    off = table((107, 8), 16, "agg", "windowed", set_)
    values, defaults = _settings(off)[_WINDOW]
    assert values == defaults


@pytest.mark.parametrize(
    "knob, value",
    [
        ("tep8_drafter_wide_gemms__WINDOW_TILES", 0),
        ("tep8_drafter_wide_gemms__WINDOW_TILES", 5),
        ("tep8_drafter_wide_gemms__WINDOW_MIN_ROWS", 8),
        ("tep8_drafter_wide_gemms__WINDOW_MIN_ROWS", 129),
    ],
)
def test_the_drafter_window_tiles_refuse_values_outside_their_band(knob, value):
    with pytest.raises(ValueError, match="expected an integer in"):
        table((107, 8), 8, "gen", "windowed", knobs(**{knob: value}))


def _window_host() -> types.ModuleType:
    def cdiv(a, b):
        return -(-a // b)

    return _host_module(
        _TARGET / "ops/spec/dflash_window_attention.py",
        ("SPLIT_BLOCK_N", "MAX_SPLITS", "MULTI_TILES", "MULTI_MIN_ROWS", "MULTI_TILES_MAX"),
        ("configure", "launch_config", "default_launch_config"),
        triton=types.SimpleNamespace(cdiv=cdiv),
    )


def test_window_attention_multi_tiles_take_only_wide_calls():
    """``launch_config``: the one-request geometry (one 32-key tile per split at the drafter's 2,055-key
    span) at the defaults and below MULTI_MIN_ROWS rows; MULTI_TILES tiles per split from
    MULTI_MIN_ROWS rows, never fewer than the default's; ``configure`` refuses values outside its band
    and sets nothing then."""
    host = _window_host()
    span, tile_rows = 2047 + 8, 64
    base = host.default_launch_config(span, tile_rows)
    assert base["tiles"] == 1 and base["block_n"] == 32
    for batch in (1, 8, 9, 16):
        assert host.launch_config(batch, 8, span, tile_rows) == base, batch
    host.configure(multi_tiles=3, multi_min_rows=65)
    for batch, tiles in ((1, 1), (8, 1), (9, 3), (12, 3), (16, 3)):
        assert host.launch_config(batch, 8, span, tile_rows) == dict(base, tiles=tiles), batch
    # A span wider than MAX_SPLITS one-tile splits keeps its own (larger) tile count.
    wide_span = host.MAX_SPLITS * host.SPLIT_BLOCK_N * 5
    assert host.launch_config(16, 8, wide_span, tile_rows)["tiles"] == 5
    for kwargs in (dict(multi_tiles=0), dict(multi_tiles=9), dict(multi_min_rows=8)):
        with pytest.raises(ValueError):
            host.configure(**kwargs)
        assert (host.MULTI_TILES, host.MULTI_MIN_ROWS) == (3, 65), kwargs
    host.configure()
    assert (host.MULTI_TILES, host.MULTI_MIN_ROWS) == (1, 65)


# --------------------------------------------------------------------------
# tep8_drafter_wide_gemms__PREFETCH / __PREFETCH_MIN_ROWS: the drafter's next GEMM weights asked of the
# L2 from a side stream (kernels/drafter_l2_prefetch)
# --------------------------------------------------------------------------
_DRAFTER_PREFETCH = "kernels.drafter_l2_prefetch.prefetch"
_ALL_PREFETCH = ("o_proj", "conv_mlp", "conv_attn", "qkv")


def test_the_drafter_prefetch_defaults_to_off():
    for mbs, arena in ((8, "windowed"), (8, "paged"), (4, "windowed")):
        base = table((107, 8), mbs, "gen", arena)
        assert base.param("tep8_drafter_wide_gemms", "PREFETCH") == 0, (mbs, arena)
        assert base.param("tep8_drafter_wide_gemms", "PREFETCH_MIN_ROWS") == 1, (mbs, arena)
        values, defaults = _settings(base)[_DRAFTER_PREFETCH]
        assert values == defaults == {"level": 0, "min_rows": 1, "weights": _ALL_PREFETCH}, (mbs, arena)
        assert base.param("tep8_drafter_wide_gemms", "PREFETCH_WEIGHTS") == _ALL_PREFETCH
    for key, mbs in (((107, 8), 16), ((107, 4), 8)):
        values, defaults = _settings(table(key, mbs, "gen", "windowed"))[_DRAFTER_PREFETCH]
        assert values == defaults, (key, mbs)


@pytest.mark.parametrize("level, min_rows", [(1, 1), (2, 65), (3, 1), (3, 72), (4, 1)])
def test_the_drafter_prefetch_knobs_reach_only_the_prefetch_module(level, min_rows):
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "windowed", ll16)
    set_ = knobs(
        tep8_drafter_wide_gemms__PREFETCH=level, tep8_drafter_wide_gemms__PREFETCH_MIN_ROWS=min_rows
    )
    moved = table((107, 8), 16, "gen", "windowed", {**ll16, **set_})
    assert moved.paths() == plain.paths()
    assert rule_levers_on(moved) == rule_levers_on(plain)
    calls, before = _settings(moved), _settings(plain)
    assert calls[_DRAFTER_PREFETCH][0] == {"level": level, "min_rows": min_rows, "weights": _ALL_PREFETCH}
    subset = table(
        (107, 8), 16, "gen", "windowed",
        {**ll16, **set_, **knobs(tep8_drafter_wide_gemms__PREFETCH_WEIGHTS="down,o_proj,qkv")},
    )  # fmt: skip
    assert _settings(subset)[_DRAFTER_PREFETCH][0]["weights"] == ("o_proj", "down", "qkv")
    assert {m: c for m, c in calls.items() if m != _DRAFTER_PREFETCH} == {
        m: c for m, c in before.items() if m != _DRAFTER_PREFETCH
    }
    off = table((107, 8), 16, "agg", "windowed", set_)
    values, defaults = _settings(off)[_DRAFTER_PREFETCH]
    assert values == defaults


@pytest.mark.parametrize(
    "knob, value",
    [
        ("tep8_drafter_wide_gemms__PREFETCH", 5),
        ("tep8_drafter_wide_gemms__PREFETCH", -1),
        ("tep8_drafter_wide_gemms__PREFETCH_MIN_ROWS", 0),
        ("tep8_drafter_wide_gemms__PREFETCH_MIN_ROWS", 129),
    ],
)
def test_the_drafter_prefetch_refuses_values_outside_its_band(knob, value):
    with pytest.raises(ValueError, match="expected an integer in"):
        table((107, 8), 8, "gen", "windowed", knobs(**{knob: value}))


def test_drafter_l2_prefetch_schedule_on_the_host():
    """``configure`` (0..3, 1..1024; refusals set nothing), the weights of each fork (o_proj + MLP conv
    coefficients; gate_up + down; the next layer's attention conv coefficients + qkv, none after the
    last layer), and the fork order of a forward per level."""
    host = _host_module(
        _TARGET / "kernels/drafter_l2_prefetch/prefetch.py",
        ("LEVEL", "MIN_ROWS", "STAGES", "WEIGHT_NAMES", "DEFAULT_WEIGHTS", "WEIGHTS"),
        ("configure", "stage_weights", "forks_of"),
    )
    assert (host.LEVEL, host.MIN_ROWS, host.STAGES) == (0, 1, ("attn", "mlp", "next", "first"))
    host.configure(level=3, min_rows=65)
    assert (host.LEVEL, host.MIN_ROWS) == (3, 65)
    for kwargs in (
        dict(level=5), dict(level=-1), dict(min_rows=0), dict(min_rows=1025), dict(weights=()), dict(weights=("fc",)),
    ):  # fmt: skip
        with pytest.raises(ValueError, match="tep8_drafter_wide_gemms__PREFETCH"):
            host.configure(**kwargs)
        assert (host.LEVEL, host.MIN_ROWS) == (3, 65), kwargs
    host.configure()
    assert (host.LEVEL, host.MIN_ROWS) == (0, 1)

    ns = types.SimpleNamespace

    def linear(name):
        return ns(weight=name)

    layers = [
        ns(
            self_attn=ns(qkv_proj=linear(f"qkv{i}"), o_proj=linear(f"o{i}")),
            mlp=ns(gate_up_proj=linear(f"gu{i}"), down_proj=linear(f"dn{i}")),
        )
        for i in range(5)
    ]
    drafter = ns(
        model=ns(layers=layers),
        attention_convs=[ns(kernel_projection=linear(f"ca{i}")) for i in range(5)],
        mlp_convs=[ns(kernel_projection=linear(f"cm{i}")) for i in range(5)],
    )
    assert host.stage_weights(drafter, "attn", 2) == ["o2", "cm2"]
    assert host.stage_weights(drafter, "mlp", 2) == ["gu2", "dn2"]
    assert host.stage_weights(drafter, "next", 2) == ["ca3", "qkv3"]
    assert host.stage_weights(drafter, "next", 4) == []
    assert host.stage_weights(drafter, "mlp", 2, ("down", "qkv")) == ["dn2"]
    assert host.stage_weights(drafter, "attn", 2, ("gate_up",)) == []
    host.configure(level=3, weights=("gate_up", "qkv"))
    assert host.WEIGHTS == ("gate_up", "qkv")
    host.configure()
    assert host.WEIGHTS == host.DEFAULT_WEIGHTS == ("o_proj", "conv_mlp", "conv_attn", "qkv")
    with pytest.raises(ValueError, match="unknown stage"):
        host.stage_weights(drafter, "head", 0)
    assert host.forks_of(0, 5) == []
    assert host.forks_of(1, 5) == [("attn", i) for i in range(5)]
    assert host.forks_of(2, 2) == [("attn", 0), ("mlp", 0), ("attn", 1), ("mlp", 1)]
    assert host.forks_of(3, 2) == [("attn", 0), ("mlp", 0), ("next", 0), ("attn", 1), ("mlp", 1)]
    assert host.forks_of(4, 2) == host.forks_of(3, 2)  # "first" is issued by the side chain
    assert host.stage_weights(drafter, "first", 3) == ["ca0", "qkv0"]


def test_the_block_decode_forks_and_joins_the_drafter_prefetch():
    """``windowed_block_decode`` takes the prefetch from the drafter, forks attn after qkv_proj, mlp after
    the window attention, next after the MLP (both hop paths), and joins before either return."""
    src = (_TARGET / "kernels/drafter_block_decode/block_decode.py").read_text()
    body = src[src.index("def windowed_block_decode(") : src.index("def install_block_decode(")]
    assert body.count('prefetch.fork("attn"') == 1 and body.count('prefetch.fork("mlp"') == 1
    assert body.count('prefetch.fork("next"') == 2
    i_qkv = body.index("qkv_query = attn_mod.qkv_proj(hs_normed_flat)")
    assert i_qkv < body.index('prefetch.fork("attn"') < body.index("dflash_window_attention(")
    i_win = body.index("attn_output = out.reshape(B * block_size, q_size)")
    assert i_win < body.index('prefetch.fork("mlp"') < body.index("hops.attention(")
    i_join = body.index("prefetch.join()")
    assert i_join < body.index("return prenormed") and i_join < body.index("hidden_states_out, _ =")
    assert '_drafter_l2_prefetch' in body


# --------------------------------------------------------------------------
# drafter_kv_fork_before_head__PAGED_COMMIT (an alias of paged_context_commit), and paged_context_commit's
# launch writing the drafter preamble on a forked folded step (kernels/verify_fold, kernels/context_kv_fork,
# ops/spec/context_kv_commit_paged)
# --------------------------------------------------------------------------
_FOLD = "kernels.verify_fold.fold"


def test_the_paged_commit_defaults_to_off():
    for key, mbs, arena in (((107, 8), 8, "paged"), ((107, 8), 8, "windowed"), ((107, 4), 8, "paged")):
        base = table(key, mbs, "gen", arena)
        assert base.param("drafter_kv_fork_before_head", "PAGED_COMMIT") == 0, (key, mbs, arena)
        # The verify fold has no setting of its own: the preamble rides on paged_context_commit.
        assert _FOLD not in _settings(base), (key, mbs, arena)


@pytest.mark.parametrize("arena", ["paged", "windowed"])
def test_the_paged_commit_knob_is_an_alias_that_configures_nothing(arena):
    # gint4-zf: PAGED_COMMIT is an alias of upstream #21789's paged_context_commit (a base path on
    # context_kv_fork): it configures nothing (the preamble rides on paged_context_commit itself).
    ll16 = _ll_on(16, VR_TEP8_DENSE)  # drafter_kv_glue refuses a paged context
    plain = table((107, 8), 16, "gen", arena, ll16)
    moved = table((107, 8), 16, "gen", arena, {**ll16, **knobs(drafter_kv_fork_before_head__PAGED_COMMIT=1)})
    assert moved.on("drafter_kv_fork_before_head")
    assert moved.param("drafter_kv_fork_before_head", "PAGED_COMMIT") == 1
    assert moved.paths() == plain.paths() and rule_levers_on(moved) == rule_levers_on(plain)
    calls, before = _settings(moved), _settings(plain)
    assert _FOLD not in calls
    assert calls == before
    source = (_TARGET / "drafter.py").read_text()
    assert 'PAGED_COMMIT=1: alias of "' in source and 'plan.enabled("paged_context_commit")' in source
    # Off with the lever (an aggregated mbs16 server under auto; the TEP8 gen role runs it at 9..16): nothing.
    off = table((107, 8), 16, "agg", arena, knobs(drafter_kv_fork_before_head__PAGED_COMMIT=1))
    assert not off.on("drafter_kv_fork_before_head")
    assert _FOLD not in _settings(off)


@pytest.mark.parametrize("value", [2, -1, "x"])
def test_the_paged_commit_refuses_values_outside_0_1(value):
    with pytest.raises(ValueError, match="expected an integer in 0..1"):
        table((107, 8), 8, "gen", "paged", knobs(drafter_kv_fork_before_head__PAGED_COMMIT=value))


def test_the_paged_commit_refusal_and_routing_on_the_host():
    """``_paged_preamble_refusal`` takes a paged context with a [requests, block] int64 position block,
    [requests] batch rows (with block tables) and a covering ``_seq_lens_cuda``, and refuses a dense one,
    oversized steps, other positions or rows and block tables without counts; the forked prepare routes
    the commit + preamble launch whenever paged_context_commit takes the step's commit, and logs the
    reason once when the preamble stays out of it."""
    ns = types.SimpleNamespace
    torch_stub = ns(int64="i64", int32="i32")
    host = _host_module(
        _TARGET / "kernels/verify_fold/fold.py",
        ("_PAGED_LOGGED",),
        ("_paged_preamble_refusal",),
        torch=torch_stub,
        Optional=None,
    )

    class _T:
        def __init__(self, shape, dtype, contiguous=True):
            self.shape, self.dtype, self._c = tuple(shape), dtype, contiguous

        def is_contiguous(self):
            return self._c

        def numel(self):
            n = 1
            for s in self.shape:
                n *= s
            return n

    fork = ns(rows=_T((64,), "i64"))
    worker = types.SimpleNamespace(_ctx_paged=True, _ctx_block_tables=object(), _ctx_block_counts=object(),
                                   _ctx_page_size=32)
    worker._kv_fork = fork
    fold = ns(query_positions=_T((12, 8), "i64"), gen_rows=_T((12,), "i64"))
    meta = ns(_seq_lens_cuda=_T((16,), "i32"))
    assert host._paged_preamble_refusal(worker, fold, meta, 12, 8) is None
    for rows in (_T((13,), "i64"), _T((12,), "i64", contiguous=False), _T((12,), "f32")):
        assert "batch rows" in host._paged_preamble_refusal(worker, ns(**{**vars(fold), "gen_rows": rows}), meta, 12, 8)
    assert "not paged" in host._paged_preamble_refusal(ns(**{**vars(worker), "_ctx_paged": False}), fold, meta, 12, 8)
    assert "exceed" in host._paged_preamble_refusal(worker, ns(query_positions=_T((65, 8), "i64")), meta, 65, 8)
    rows = _T((12,), "i64")
    assert "query positions" in host._paged_preamble_refusal(worker, ns(query_positions=_T((12, 7), "i64"), gen_rows=rows), meta, 12, 8)
    assert "query positions" in host._paged_preamble_refusal(worker, ns(query_positions=_T((12, 8), "i32"), gen_rows=rows), meta, 12, 8)
    assert "_seq_lens_cuda" in host._paged_preamble_refusal(worker, fold, ns(_seq_lens_cuda=_T((8,), "i32")), 12, 8)
    no_counts = types.SimpleNamespace(**{**vars(worker), "_ctx_block_counts": None})
    assert "block counts" in host._paged_preamble_refusal(no_counts, fold, meta, 12, 8)
    no_tables = types.SimpleNamespace(**{**vars(worker), "_ctx_block_tables": None, "_ctx_block_counts": None})
    assert host._paged_preamble_refusal(no_tables, fold, meta, 12, 8) is None
    # The prepare's routing (source): the commit + preamble launch whenever paged_context_commit takes
    # the step's commit (fork._commits_paged), the drafter's own batch index (the rows with block tables,
    # as _drafter_inputs gives it), a refusal logged once per reason, the engaged path one INFO line.
    src = (_TARGET / "kernels/verify_fold/fold.py").read_text()
    body = src[src.index("    def _fold_forked_inputs(") : src.index("    def _path_walk_tail_args(")]
    assert "if self._ctx_paged and fork._commits_paged(self, slots, gen_num_accepted):" in body
    assert "num_ctx, i32s = fork.commit_paged_preamble(" in body
    # The drafter's batch index is hoisted into batch_idx (feat's commit_with_preamble takes it too).
    assert "batch_idx = fold.gen_rows if self._ctx_block_tables is not None else slots" in body
    call = body[body.index("fork.commit_paged_preamble(") + len("fork.commit_paged_preamble(") :]
    assert call[: call.index(")")].split(",")[3].strip() == "batch_idx"
    assert "PAGED_PREAMBLE" not in src and "paged_preamble=" not in src
    assert "logger.warning(" in body and "leaves this step's" in body
    assert "paged drafter context commit + drafter preamble + " in body
    # A refusal after an engaged step still logs: one line per reason, one INFO for the engaged path.
    assert "if not paged and paged_refusal not in _PAGED_LOGGED:" in body
    assert "elif paged and None not in _PAGED_LOGGED:" in body


# --------------------------------------------------------------------------
# tep8_drafter_wide_gemms__FINISH_NORM: the non-fused hops' tail (finish conv + add + RMSNorm) in one launch
# --------------------------------------------------------------------------
_BLOCK_DECODE = "kernels.drafter_block_decode.block_decode"


def test_the_finish_norm_defaults_to_off_and_reaches_only_the_block_decode():
    for mbs, arena in ((8, "windowed"), (8, "paged")):
        base = table((107, 8), mbs, "gen", arena)
        assert base.param("tep8_drafter_wide_gemms", "FINISH_NORM") == 0
        values, defaults = _settings(base)[_BLOCK_DECODE]
        assert values == defaults == {"finish_norm": False, "qk_fold": False}
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "windowed", ll16)
    moved = table((107, 8), 16, "gen", "windowed", {**ll16, **knobs(tep8_drafter_wide_gemms__FINISH_NORM=1)})
    assert moved.paths() == plain.paths() and rule_levers_on(moved) == rule_levers_on(plain)
    calls, before = _settings(moved), _settings(plain)
    assert calls[_BLOCK_DECODE][0] == {"finish_norm": True, "qk_fold": False}  # int7: QK fold off again at 9..16
    assert {m: c for m, c in calls.items() if m != _BLOCK_DECODE} == {
        m: c for m, c in before.items() if m != _BLOCK_DECODE
    }
    off = table((107, 8), 16, "agg", "windowed", knobs(tep8_drafter_wide_gemms__FINISH_NORM=1))
    values, defaults = _settings(off)[_BLOCK_DECODE]
    assert values == defaults
    with pytest.raises(ValueError, match="expected an integer in 0..1"):
        table((107, 8), 8, "gen", "windowed", knobs(tep8_drafter_wide_gemms__FINISH_NORM=2))


def test_the_block_decode_routes_the_hop_tails_through_finish_norm():
    """Without the fused hop, both hop tails of a layer go through ``_finish_norm`` under FINISH_NORM (the
    attention side's post-attention norm, the MLP side's next norm / model.norm as the prenormed rows), a
    refusal falls back to the built-in launches, and the forward returns the prenormed rows at the end."""
    src = (_TARGET / "kernels/drafter_block_decode/block_decode.py").read_text()
    body = src[src.index("def windowed_block_decode(") : src.index("def install_block_decode(")]
    assert body.count("_finish_norm(") == 2 and body.count("if FINISH_NORM and has_block_conv") == 2
    i_attn = body.index("hidden_out = attn_mod.o_proj(attn_output)")
    i_fused = body.index("layer.post_attention_layernorm,\n", i_attn)
    assert i_attn < i_fused < body.index("hidden_out, res_flat = layer.post_attention_layernorm(hidden_out, res_flat)")
    i_mlp = body.index("hidden_out = layer.mlp(hidden_out)")
    assert i_mlp < body.index("self.mlp_convs[layer_idx], next_norm, hidden_out, mlp_conv_coefficients", i_mlp)
    assert "prenormed = fused" in body and "prenormed = None" in body
    assert "if hops is not None or prenormed is not None:" in body
    host = _host_module(
        _TARGET / "kernels/drafter_block_decode/block_decode.py",
        ("FINISH_NORM", "_FINISH_NORM_LOGGED"),
        ("configure",),
    )
    assert host.FINISH_NORM is False
    host.configure(finish_norm=True)
    assert host.FINISH_NORM is True
    host.configure()
    assert host.FINISH_NORM is False


def test_the_side_chain_asks_for_layer_0_weights_at_prefetch_level_4():
    """``ContextKvFork.fire`` issues the drafter prefetch's ``first`` stage inside the side stream block, right
    after the side chain (so the main stream's join before the drafter covers it)."""
    src = (_TARGET / "kernels/context_kv_fork/fork.py").read_text()
    body = src[src.index("    def fire(self, point: str)") : src.index("    # -- the worker side")]
    i_with = body.index("with torch.cuda.stream(self.stream):")
    i_chain = body.index("gen_hs = self.side_chain(", i_with)
    i_first = body.index("prefetch.first(num_gens * (k + 1))", i_chain)
    assert i_with < i_chain < i_first < body.index("self._early = None")


def test_the_qk_fold_knob_and_its_block_decode_routing():
    """tep8_drafter_wide_gemms__QK_FOLD (0/1, default 0) reaches only the block decode's configure; the decode
    folds only with the plain marker on the path without drafter_attn_pdl (and never on the wide op), and the
    PDL copy's path ignores a plain marker."""
    base = table((107, 8), 8, "gen", "windowed")
    assert base.param("tep8_drafter_wide_gemms", "QK_FOLD") == 0
    values, defaults = _settings(base)[_BLOCK_DECODE]
    assert values == defaults == {"finish_norm": False, "qk_fold": False}
    ll16 = _ll_on(16, VR_TEP8)
    moved = table((107, 8), 16, "gen", "windowed", {**ll16, **knobs(tep8_drafter_wide_gemms__QK_FOLD=1)})
    assert _settings(moved)[_BLOCK_DECODE][0] == {"finish_norm": False, "qk_fold": True}
    src = (_TARGET / "kernels/drafter_block_decode/block_decode.py").read_text()
    assert 'qk_fold = self.__dict__.get("_dr3_qf") if not plain_qf else None' in src
    assert 'qk_fold = self.__dict__.get("_dr3_qf") if QK_FOLD and plain_qf and wide is None else None' in src
    folds = (_TARGET / "kernels/dr3_folds/folds.py").read_text()
    assert 'draft_model.__dict__["_dr3_qf_plain"] = True' in folds
    assert "tep8_drafter_wide_gemms__QK_FOLD cannot run here" in folds


def test_the_pdl_conv_max_rows_knob_and_candidates():
    """tep8_drafter_wide_gemms__PDL_CONV_MAX_ROWS (32..128, default 32) reaches only the PDL chain's configure;
    the conv trigger's candidate widths run every multiple of 8 up to it; refusals set nothing."""
    chain = "kernels.drafter_pdl_chain.chain"
    base = table((107, 8), 8, "gen", "windowed")
    assert base.param("tep8_drafter_wide_gemms", "PDL_CONV_MAX_ROWS") == 32
    values, defaults = _settings(base)[chain]
    assert values == defaults == {"conv_max_rows": 32, "selector_max_rows": 7}
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "windowed", ll16)
    moved = table((107, 8), 16, "gen", "windowed", {**ll16, **knobs(tep8_drafter_wide_gemms__PDL_CONV_MAX_ROWS=64)})
    calls, before = _settings(moved), _settings(plain)
    assert calls[chain][0] == {"conv_max_rows": 64, "selector_max_rows": 7}
    assert {m: c for m, c in calls.items() if m != chain} == {m: c for m, c in before.items() if m != chain}
    host = _host_module(
        _TARGET / "kernels/drafter_pdl_chain/chain.py",
        ("ROWS", "ENTRY_ROW_CANDIDATES", "CONV_MAX_ROWS", "HEAD_ROWS", "SELECTOR_MAX_ROWS", "SELECTOR_LIMIT"),
        ("configure", "conv_candidates"),
    )
    assert host.conv_candidates() == (8, 16, 24, 32)
    host.configure(conv_max_rows=64)
    assert host.conv_candidates() == (8, 16, 24, 32, 40, 48, 56, 64)
    for bad in (24, 132, 60):
        with pytest.raises(ValueError, match="PDL_CONV_MAX_ROWS"):
            host.configure(conv_max_rows=bad)
        assert host.CONV_MAX_ROWS == 64
    host.configure(conv_max_rows=128)
    assert host.conv_candidates()[-1] == 128 and len(host.conv_candidates()) == 16
    src = (_TARGET / "kernels/drafter_pdl_chain/chain.py").read_text()
    assert "for width in conv_candidates():" in src


def test_the_selector_fork_max_rows_knob_and_rows():
    """tep8_drafter_wide_gemms__SELECTOR_FORK_MAX_ROWS (7..112, default 7) reaches only the PDL chain's configure;
    the selector fork takes the one-request head's 7 rows and, past them, every multiple of 7 up to the knob;
    refusals set nothing; a multi-request fork the projection does not take logs ERROR with the knob's name."""
    chain = "kernels.drafter_pdl_chain.chain"
    base = table((107, 8), 8, "gen", "windowed")
    assert base.param("tep8_drafter_wide_gemms", "SELECTOR_FORK_MAX_ROWS") == 7
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "windowed", ll16)
    moved = table(
        (107, 8), 16, "gen", "windowed", {**ll16, **knobs(tep8_drafter_wide_gemms__SELECTOR_FORK_MAX_ROWS=112)}
    )
    calls, before = _settings(moved), _settings(plain)
    assert calls[chain][0] == {"conv_max_rows": 64, "selector_max_rows": 112}  # conv entry 64: int5 at 9..16
    assert {m: c for m, c in calls.items() if m != chain} == {m: c for m, c in before.items() if m != chain}
    host = _host_module(
        _TARGET / "kernels/drafter_pdl_chain/chain.py",
        ("ROWS", "ENTRY_ROW_CANDIDATES", "CONV_MAX_ROWS", "HEAD_ROWS", "SELECTOR_MAX_ROWS", "SELECTOR_LIMIT"),
        ("configure", "selector_forks"),
    )
    assert host.HEAD_ROWS == 7 and host.SELECTOR_MAX_ROWS == 7 and host.SELECTOR_LIMIT == 112
    assert [r for r in range(0, 130) if host.selector_forks(r)] == [7]
    host.configure(selector_max_rows=56)
    assert [r for r in range(0, 130) if host.selector_forks(r)] == [7, 14, 21, 28, 35, 42, 49, 56]
    for bad in (6, 113, 0):
        with pytest.raises(ValueError, match="SELECTOR_FORK_MAX_ROWS"):
            host.configure(selector_max_rows=bad)
        assert host.SELECTOR_MAX_ROWS == 56 and host.CONV_MAX_ROWS == 32
    host.configure(selector_max_rows=112)
    assert [r for r in range(0, 130) if host.selector_forks(r)][-1] == 112
    src = (_TARGET / "kernels/drafter_pdl_chain/chain.py").read_text()
    assert "and selector_forks(hidden_states.shape[0])" in src
    assert "hidden_states.shape[0] == HEAD_ROWS" not in src
    assert "tep8_drafter_wide_gemms__SELECTOR_FORK_MAX_ROWS: the selector projection" in src
    assert "if source.shape[0] > HEAD_ROWS and not self.refused:" in src



def test_the_drafter_two_shot_hop_knob_and_routing():
    """wide_drafter_hops__TEP8_TWO_SHOT_MIN_ROWS (0 or 9..128, default 0) reaches only the hops' configure; the
    hops route forwards of MIN..TWO_SHOT_MAX_ROWS rows to the two-shot workspace ahead of the wide one; the kernel's
    two-shot path scatters to the owner, polls the gather and the chain sums, and keeps the one-shot's statements."""
    base = table((107, 8), 8, "gen", "windowed")
    assert base.param("wide_drafter_hops", "TEP8_TWO_SHOT_MIN_ROWS") == 0
    ll16 = _ll_on(16, VR_TEP8)
    moved = table((107, 8), 16, "gen", "windowed", {**ll16, **knobs(wide_drafter_hops__TEP8_TWO_SHOT_MIN_ROWS=17)})
    # (the two-shot cap 64 is the int5 default at 9..16 requests)
    assert _settings(moved)[_DRAFTER_HOPS][0] == {"tep8_wide_max_rows": 64, "tep8_two_shot_min_rows": 17, "tep8_two_shot_max_rows": 64}
    for bad in (1, 8, 129):
        with pytest.raises(ValueError, match="TEP8_TWO_SHOT_MIN_ROWS"):
            table((107, 8), 8, "gen", "windowed", knobs(wide_drafter_hops__TEP8_TWO_SHOT_MIN_ROWS=bad))
    host = _host_module(
        _TARGET / "kernels/drafter_hops/hops.py",
        ("TEP8_WIDE_MAX_ROWS", "TEP8_WIDE_ROWS_BAND", "TEP8_TWO_SHOT_MIN_ROWS", "TWO_SHOT_MAX_ROWS", "TEP8_TWO_SHOT_MAX_ROWS"),
        ("configure",),
    )
    host.configure(tep8_two_shot_min_rows=24)
    assert host.TEP8_TWO_SHOT_MIN_ROWS == 24
    for bad in (5, 129):
        with pytest.raises(ValueError, match="TEP8_TWO_SHOT_MIN_ROWS"):
            host.configure(tep8_two_shot_min_rows=bad)
        assert host.TEP8_TWO_SHOT_MIN_ROWS == 24
    # TEP8_TWO_SHOT_MAX_ROWS (9..128, default 128, not below the minimum): routing only, the workspace keeps 128.
    assert host.TEP8_TWO_SHOT_MAX_ROWS == 128
    host.configure(tep8_two_shot_min_rows=25, tep8_two_shot_max_rows=64)
    assert (host.TEP8_TWO_SHOT_MIN_ROWS, host.TEP8_TWO_SHOT_MAX_ROWS) == (25, 64)
    for lo, hi in ((25, 8), (25, 129), (65, 64)):
        with pytest.raises(ValueError, match="TEP8_TWO_SHOT_MAX_ROWS"):
            host.configure(tep8_two_shot_min_rows=lo, tep8_two_shot_max_rows=hi)
        assert (host.TEP8_TWO_SHOT_MIN_ROWS, host.TEP8_TWO_SHOT_MAX_ROWS) == (25, 64)
    capped = table(
        (107, 8), 16, "gen", "windowed",
        {**ll16, **knobs(wide_drafter_hops__TEP8_TWO_SHOT_MIN_ROWS=25, wide_drafter_hops__TEP8_TWO_SHOT_MAX_ROWS=64)},
    )
    assert _settings(capped)[_DRAFTER_HOPS][0] == {
        "tep8_wide_max_rows": 64, "tep8_two_shot_min_rows": 25, "tep8_two_shot_max_rows": 64
    }
    for bad in (8, 129):
        with pytest.raises(ValueError, match="TEP8_TWO_SHOT_MAX_ROWS"):
            table((107, 8), 8, "gen", "windowed", knobs(wide_drafter_hops__TEP8_TWO_SHOT_MAX_ROWS=bad))
    src = (_TARGET / "kernels/drafter_hops/hops.py").read_text()
    takes = src[src.index("    def takes(self, rows: int"): src.index("    def begin_forward(")]
    assert takes.index("self.two_shot is not None") < takes.index("if self.wide is not None")
    assert "self.two_shot_min_rows <= rows <= self.two_shot_max_rows" in takes
    assert "call.two_shot_max_rows = min(TEP8_TWO_SHOT_MAX_ROWS, call.two_shot.max_rows)" in src
    assert "max_rows=TWO_SHOT_MAX_ROWS," in src  # the gated 128-row workspace whatever the cap
    kernel = (_TARGET / "ops/allreduce/drafter_hop.py").read_text()
    for needle in (
        "_store_x4_sys_if(",
        "owner_area = _load_u64(peers + Int64(crank) * 8) + area_offset",
        "on = on & is_owner",
        "x[e] = _select_f32(is_owner, x[e], _f32_from_bits(gather[e // 4, e % 4]))",
        "two_shot_area_bytes(self.max_rows, self.ranks)",
    ):
        assert needle in kernel, needle


# --------------------------------------------------------------------------
# tep8_drafter_wide_gemms__SILU_FOLD: the drafter MLP's activation in the gate_up GEMM's epilogue
# --------------------------------------------------------------------------
_SILU_FOLD = "kernels.drafter_silu_fold.fold"


def test_the_silu_fold_knob_reaches_only_the_fold():
    """SILU_FOLD (0..1, default 0) turns no lever on or off and changes no path; it reaches the fold module's
    ``configure`` (silu_fold) and nothing else, only while tep8_drafter_wide_gemms is on."""
    for mbs, arena in ((8, "windowed"), (8, "paged")):
        base = table((107, 8), mbs, "gen", arena)
        assert base.param("tep8_drafter_wide_gemms", "SILU_FOLD") == 0
        values, defaults = _settings(base)[_SILU_FOLD]
        assert values == defaults == {"silu_fold": False}
    ll16 = _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "windowed", ll16)
    moved = table((107, 8), 16, "gen", "windowed", {**ll16, **knobs(tep8_drafter_wide_gemms__SILU_FOLD=1)})
    assert moved.paths() == plain.paths() and rule_levers_on(moved) == rule_levers_on(plain)
    calls, before = _settings(moved), _settings(plain)
    assert calls[_SILU_FOLD][0] == {"silu_fold": True}
    assert {m: c for m, c in calls.items() if m != _SILU_FOLD} == {m: c for m, c in before.items() if m != _SILU_FOLD}
    off = table((107, 8), 16, "agg", "windowed", knobs(tep8_drafter_wide_gemms__SILU_FOLD=1))
    assert not off.on("tep8_drafter_wide_gemms")
    values, defaults = _settings(off)[_SILU_FOLD]
    assert values == defaults
    for bad in (2, -1):
        with pytest.raises(ValueError, match="expected an integer in 0..1"):
            table((107, 8), 8, "gen", "windowed", knobs(tep8_drafter_wide_gemms__SILU_FOLD=bad))


def test_the_silu_fold_routing_and_its_loud_refusals():
    """The fold's host side: configure; the 8..64-row gate_up entries of the site's table; the forward folds a
    row count the table lists (``silu_fold_gemm`` when it takes the operands), else runs gate_up + the interleaved
    activation (``half`` = mma_m / 2), logging ERROR with the knob for a listed row count it could not fold; the
    install runs after the PDL chain and the fixed GEMMs, refuses with the knob's name, interleaves in place."""
    host = _host_module(
        _TARGET / "kernels/drafter_silu_fold/fold.py",
        ("SILU_FOLD", "ATTRIBUTE", "MIN_ROWS", "MAX_ROWS"),
        ("configure", "SiluFold", "_gate_up_entries"),
        logger=types.SimpleNamespace(error=lambda *a, **k: None, info=lambda *a, **k: None),
        nn=types.SimpleNamespace(Module=object),
    )
    assert host.SILU_FOLD is False and (host.MIN_ROWS, host.MAX_ROWS) == (8, 64)
    host.configure(silu_fold=True)
    assert host.SILU_FOLD is True
    table_ = {
        (8, 8192, 7168): (64, 16, 1, 9), (16, 8192, 7168): (64, 16, 1, 11), (64, 8192, 7168): (64, 64, 1, 7),
        (72, 8192, 7168): (64, 128, 1, 7), (8, 8192, 2048): (64, 8, 4, 10),
    }
    linear = types.SimpleNamespace(quant_method=types.SimpleNamespace(tactics=table_))
    entries = host._gate_up_entries(linear, 8192, 7168)
    assert set(entries) == {8, 16, 64}  # 72 rows: above the fold; other shapes: other sites
    fold = host.SiluFold(64, {8: "t8", 64: "t64"}, 3)
    assert fold.half == 32 and fold.tactic(64) == "t64" and fold.tactic(72) is None and fold.version == 3
    src = (_TARGET / "kernels/drafter_silu_fold/fold.py").read_text()
    body = src[src.index("def folded_forward("):]
    assert "if tactic is not None and takes(flat, weight, tactic):" in body
    assert "fold.refused(rows, " in body and "drafter_silu_and_mul(h.reshape(-1, h.shape[-1]), False, fold.half)" in body
    assert "tep8_drafter_wide_gemms__SILU_FOLD: the {rows}-row drafter MLP did not take the" in src
    assert "tep8_drafter_wide_gemms__SILU_FOLD cannot run here: " in src
    assert "weight.data.copy_(interleave_gate_up(weight.data, mma_m))" in src
    # A write to the weight after the interleave (a reload) is refused, not computed on plain rows.
    assert "if weight._version != fold.version:" in body and "a weight reload?" in body
    # A load hook after a reload through .data (which _version does not see) finds the fingerprint changed and
    # interleaves again instead of keeping the fold on plain rows.
    install = src[src.index("def install("):src.index("def folded_forward(")]
    assert "if isinstance(fold, SiluFold) and not _still_interleaved(layer.mlp, fold):" in install
    assert install.index("_still_interleaved(") < install.index("reason = unsupported_reason(draft_model)")
    assert "return weight.detach()[mma_m // 2, :FINGERPRINT].clone()" in src
    for needle in ("not the PDL chain's MLP (select drafter_pdl_chain)", "already runs drafter_folds' silu fold",
                   "mix kernel-M tiles", "has no fold (split_k > 1 or no stage fits)"):
        assert needle in src, needle
    chain = (_TARGET / "kernels/drafter_pdl_chain/chain.py").read_text()
    forward = chain[chain.index("class EntryTriggerGatedMLP"):chain.index("def _mlp_reason")]
    assert forward.index('self.__dict__.get("_silu_fold")') < forward.index('self.__dict__.get("_dr3_sf")')
    drafter = (_TARGET / "drafter.py").read_text()
    assert drafter.index("_drafter_pdl.install(shell.llm, shell._mapping)") < drafter.index(
        "_silu_fold.install(getattr(shell.llm, \"draft_model\", None))"
    )
    gemm = (_TARGET / "ops/gemm/silu_fold_gemm.py").read_text()
    assert "class SiluFoldKernel(_splitk.SplitKDenseGemmKernel):" in gemm  # this target's kernel (wide64, oversized)
    assert "_splitk._smem_bytes(candidate, stages) + _stage_bytes(candidate) + 1024 <= _splitk._SMEM_CAPACITY_BYTES" in gemm
    assert "if t.split_k != 1 or t.mma_m not in (64, 128):" in gemm


@pytest.mark.parametrize("half, width", [(32, 128), (32, 14336), (64, 512)])
def test_the_interleaved_activation_reads_the_interleave_of_the_fold(half, width):
    """``drafter_silu_and_mul(.., half)``'s gate column of output column j is j // half * 2 half + j % half and its
    up column that + half: exactly where ``interleave_gate_up`` (tile 2 half) puts gate row j and up row j."""
    d = width // 2
    gate_rows = list(range(d))
    up_rows = list(range(d, width))
    # interleave_gate_up's row order: per tile, half gate rows then the matching half up rows
    order = []
    for t in range(d // half):
        order += gate_rows[t * half:(t + 1) * half] + up_rows[t * half:(t + 1) * half]
    position = {row: i for i, row in enumerate(order)}
    for j in range(d):
        gate = (j // half) * (2 * half) + j % half
        assert position[j] == gate and position[d + j] == gate + half
    src = (_TARGET / "ops/spec/drafter_silu_and_mul.py").read_text()
    assert "gate = (offsets // HALF) * (2 * HALF) + offsets % HALF" in src
    assert "b = tl.load(x_row_ptr + gate + HALF, mask=mask).to(tl.float32)" in src
    assert "result = a * tl.sigmoid(swiglu_alpha * a) * (b + swiglu_beta)" in src  # one statement for both orders


# ---------------------------------------------------------------------------
# head_sampler_coop_fold__VERIFY_GRID: the verify fold over one program per request, default off
# ---------------------------------------------------------------------------


def test_the_coop_fold_verify_grid_defaults_to_off():
    for mbs in (8, 16):
        lever_table = table((107, 8), mbs, "gen", "paged")
        assert lever_table.param("head_sampler_coop_fold", "VERIFY_GRID") == 0
        values, defaults = _settings(lever_table)["ops.spec.dflash_verify_fold"]
        assert values == defaults == {"grid": False}


def test_the_coop_fold_verify_grid_knob_reaches_the_fold():
    base = table((107, 8), 16, "gen", "paged")
    moved = table((107, 8), 16, "gen", "paged", knobs(head_sampler_coop_fold__VERIFY_GRID=1))
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    assert _settings(moved)["ops.spec.dflash_verify_fold"] == ({"grid": True}, {"grid": False})
    off = table(
        (107, 8), 16, "gen", "paged", knobs(head_sampler_coop_fold="off", head_sampler_coop_fold__VERIFY_GRID=1)
    )
    assert _settings(off)["ops.spec.dflash_verify_fold"] == ({"grid": False}, {"grid": False})


@pytest.mark.parametrize("bad", [2, -1])
def test_the_coop_fold_verify_grid_refuses_other_values(bad):
    with pytest.raises(ValueError, match="VERIFY_GRID"):
        table((107, 8), 16, "gen", "paged", knobs(head_sampler_coop_fold__VERIFY_GRID=bad))


def test_the_coop_fold_dist_fabric_defaults_off_and_reaches_the_head_sampler():
    base = table((107, 8), 16, "gen", "paged")
    assert base.param("head_sampler_coop_fold", "DIST_FABRIC") == 0
    assert _settings(base)["ops.spec.head_sampler"][0]["dist_fabric"] is False
    moved = table(
        (107, 8),
        16,
        "gen",
        "paged",
        knobs(head_sampler_coop_fold__DIST_MIN_ROWS=16, head_sampler_coop_fold__DIST_FABRIC=1),
    )
    assert moved.paths() == base.paths()
    values, defaults = _settings(moved)["ops.spec.head_sampler"]
    assert values["dist_fabric"] is True and values["dist_min_rows"] == 16 and defaults["dist_fabric"] is False
    with pytest.raises(ValueError, match="DIST_FABRIC"):
        table((107, 8), 16, "gen", "paged", knobs(head_sampler_coop_fold__DIST_FABRIC=2))


# --------------------------------------------------------------------------
# drafter_kv_side_chain__PAGED: the side chain over the KV cache manager's pages, fused
# --------------------------------------------------------------------------
_KV_FORK = "kernels.context_kv_fork.fork"


def test_the_paged_side_chain_knob_reaches_only_the_fork():
    """drafter_kv_fork_before_head__PAGED_CHAIN (0..2, default 0) turns no lever on or off and changes no path; it
    reaches the fork's ``configure`` (paged_side_chain) and nothing else, while drafter_kv_fork_before_head is on --
    which it is on the paged composite. Since feat's UNWINDOWED_ARENA_LEVERS, drafter_kv_side_chain is on there too
    (``_fused_side_chain_paged``); the fork takes the knob's paged chain first when it is set and accepted, so the two
    never both store a step (``ContextKvFork._store``)."""
    for mbs, arena in ((8, "windowed"), (8, "paged"), (16, "paged")):
        base = table((107, 8), mbs, "gen", arena)
        values, defaults = _settings(base)[_KV_FORK]
        assert values == defaults == {"paged_side_chain": 0}
    ll16 = _ll_on(16, VR_TEP8 - {"drafter_kv_glue"}) if "drafter_kv_glue" in VR_TEP8 else _ll_on(16, VR_TEP8)
    plain = table((107, 8), 16, "gen", "paged", ll16)
    assert plain.on("drafter_kv_fork_before_head") and plain.on("drafter_kv_side_chain")
    for value in (1, 2):
        moved = table((107, 8), 16, "gen", "paged", {**ll16, **knobs(drafter_kv_fork_before_head__PAGED_CHAIN=value)})
        assert moved.paths() == plain.paths() and rule_levers_on(moved) == rule_levers_on(plain)
        calls, before = _settings(moved), _settings(plain)
        assert calls[_KV_FORK][0] == {"paged_side_chain": value}
        assert {m: c for m, c in calls.items() if m != _KV_FORK} == {
            m: c for m, c in before.items() if m != _KV_FORK
        }
    for bad in (3, -1):
        with pytest.raises(ValueError, match="expected an integer in 0..2"):
            table((107, 8), 8, "gen", "paged", knobs(drafter_kv_fork_before_head__PAGED_CHAIN=bad))


class _FakePool:
    """What ``_paged_layer_pools_or_reason`` reads of a layer's pool."""

    def __init__(self, shape, strides=None, dtype="bf16", device="cuda:0") -> None:
        self.shape = tuple(shape)
        self._strides = tuple(strides) if strides is not None else tuple(
            int(_prod(self.shape[i + 1:])) for i in range(len(self.shape))
        )
        self.dtype, self.device = dtype, device

    def dim(self) -> int:
        return len(self.shape)

    def stride(self, i=None):
        return self._strides if i is None else self._strides[i]


def _prod(xs) -> int:
    out = 1
    for x in xs:
        out *= x
    return out


def test_the_paged_side_chain_takes_v2_pools_whose_page_counts_differ():
    """KVCacheManagerV2.get_buffers wraps each layer's own base address in a view running to the end of the shared
    pool, so the drafter layers' page counts differ (vrg-zk1b: every rank refused "not uniform"). The pools only need
    the same [2, kv_heads, page, head_dim], strides, dtype and device -- the scatter reads each layer's base pointer
    and the shared strides, never shape[0]; anything else that differs is refused with the layer and both layouts."""
    host = _host_module(
        _TARGET / "kernels/context_kv_fork/fork.py",
        (),
        ("_paged_layer_pools_or_reason", "_paged_layer_pools"),
        torch=types.SimpleNamespace(Tensor=_FakePool),
        Optional=object,
    )
    row = (2, 1, 32, 128)
    worker = types.SimpleNamespace(_ctx_kv_buf=[_FakePool((4096 - i, *row)) for i in range(5)])
    pools, why = host._paged_layer_pools_or_reason(worker)
    assert why is None and pools == worker._ctx_kv_buf
    assert host._paged_layer_pools(worker) == worker._ctx_kv_buf
    strided = _FakePool((4096, *row), strides=(5 * 2 * 32 * 128, 32 * 128, 32 * 128, 128, 1))
    for bad, needle in (
        ([_FakePool((4096, *row)), _FakePool((4096, 2, 2, 32, 128))], "drafter layer 1's pool is (4096, 2, 2, 32, 128)"),
        ([_FakePool((4096, *row)), strided], "drafter layer 1's pool is (4096, 2, 1, 32, 128) / (40960,"),
        ([_FakePool((4096, *row)), _FakePool((4096, *row), dtype="fp8")], "dtype or device differs"),
        ([_FakePool((4096, *row)), _FakePool((4096, 32, 128))], "drafter layer 1's pool is not a 5-d tensor"),
        ([], "has no layers"),
    ):
        pools, why = host._paged_layer_pools_or_reason(types.SimpleNamespace(_ctx_kv_buf=bad))
        assert pools is None and needle in why, (needle, why)
        assert host._paged_layer_pools(types.SimpleNamespace(_ctx_kv_buf=bad)) is None
    pools, why = host._paged_layer_pools_or_reason(types.SimpleNamespace(_ctx_kv_buf=object()))
    assert pools is None and "not a per-layer list" in why


def test_the_paged_side_chain_routes_and_refuses_loudly():
    """The side chain takes the paged path first, only under PAGED_SIDE_CHAIN with the side chain on and a paged
    context; a refusal logs ERROR naming the knob (once per reason) and the built-in store runs; engagement logs one
    INFO line; the refusals cover the private paged arena, the metadata's offsets, the pools' layout, the RoPE, the
    K/V GEMM and a pointer table first asked for inside a capture. The kernels carry the built-in's statements."""
    host = _host_module(
        _TARGET / "kernels/context_kv_fork/fork.py", ("PAGED_SIDE_CHAIN", "_PAGED_LOGGED"), ("configure",)
    )
    assert host.PAGED_SIDE_CHAIN == 0
    host.configure(paged_side_chain=2)
    assert host.PAGED_SIDE_CHAIN == 2
    for bad in (3, -1):
        with pytest.raises(ValueError, match="drafter_kv_fork_before_head__PAGED_CHAIN"):
            host.configure(paged_side_chain=bad)
        assert host.PAGED_SIDE_CHAIN == 2
    src = (_TARGET / "kernels/context_kv_fork/fork.py").read_text()
    body = src[src.index("    def side_chain(self, spec_metadata"):src.index("    def _glue_side_chain(")]
    assert body.index("if PAGED_SIDE_CHAIN and worker._ctx_paged:") < body.index(
        "refusal = _side_chain_refusal(worker, drafter) if self.side_chain_on else None"
    )
    assert "drafter_kv_fork_before_head__PAGED_CHAIN is set but this step's side chain cannot take it" in body
    assert "PAGED_CHAIN is set but the drafter context is not" in body  # set over a dense arena: ERROR, not silence
    assert "logger.error(" in body and "if paged_refusal not in _PAGED_LOGGED:" in body
    assert "the paged context-K/V side chain in two launches around its" in body
    refusal = src[src.index("def _paged_side_chain_refusal("):src.index("def _fused_paged_side_chain(")]
    for needle in ("the private paged arena", "is not [pools, batch, 2, blocks]", "no flashinfer RoPE",
                   "not uniform [pages, 2, kv_heads, page, head_dim]", "precompute_context_kv is replaced",
                   "was not built before the capture"):
        assert needle in refusal, needle
    # PAGED 2: fire_early runs the paged prep + sharded fc early (PAGED 1: no early part over pages); the head fork
    # consumes it after checking it is this step's and capture's.
    fire = src[src.index("    def fire_early(self)"):src.index("    def fire(self, point: str)")]
    assert "paged = PAGED_SIDE_CHAIN >= 2 and self.worker._ctx_paged" in fire
    assert "if not ((self.early_fc_on and self.side_chain_on) or self.paged_early_on):" in fire
    modeling = (_TARGET / "modeling.py").read_text()
    assert 'if not (fork.early_fc_on or getattr(fork, "paged_early_on", False)):' in modeling
    assert "early = _paged_side_chain_early(" in fire
    assert "the early paged side-chain part was launched for" in body
    early_src = src[src.index("def _paged_side_chain_early("):src.index("def _fused_paged_side_chain(")]
    assert early_src.index("prologue = _paged_side_chain_prologue(") < early_src.index("local = _drafter_worker._gemm(")
    chain = src[src.index("def _fused_paged_side_chain("):src.index("class ContextKvForkDFlashWorker")]
    assert "projected = drafter.hidden_norm(early.gather.gather(early.local))" in chain
    # precompute_context_kv written out exactly as _fused_side_chain writes it.
    for needle in ("kv_flat = _kv_gemm(drafter)(projected)", "k_new = kv[:, :, 0].contiguous()",
                   "k_new = torch.nn.functional.rms_norm(k_new, (hd,), eps=drafter._k_norm_eps)",
                   "k_new = k_new * drafter._k_norm_stacked.view(1, layers, 1, hd)",
                   "modeling_dflash._flashinfer_rope(", "projected = drafter.project_target_hidden("):
        assert needle in chain, needle
    assert "pools, why = _paged_layer_pools_or_reason(worker)" in refusal and "tensors: {why}" in refusal
    kern = (_TARGET / "ops/spec/context_kv_store_paged.py").read_text()
    for needle in (".to(tl.int64) - block_offset", "valid = tl.where((encoded >= 0) & keep, (cols + 1).to(tl.int64), 0)",
                   "decoded = tl.maximum(encoded, 0) // divisor", "col = tl.maximum(tl.minimum(col, count * page_size - 1), 0)",
                   "page = tl.load(table_ptr + g.to(tl.int64) * table_row_stride + col // page_size)",
                   "tl.store(dst + plane_stride, v, mask=m)"):
        assert needle in kern, needle
# --------------------------------------------------------------------------
# int5: the gen lane's stacked winners are the TEP8 gen role's defaults at 9..16 requests
# --------------------------------------------------------------------------
INT5_GEN_WIDE_DEFAULTS = {
    ("head_sampler_coop_fold", "TOPP_ROUNDS"): (2, 0),
    ("head_sampler_coop_fold", "DIST_MIN_ROWS"): (16, 0),
    ("tep8_pool_programmatic_fork", "NORM_PREFETCH"): (1, 0),
    ("incremental_block_offsets", "MAX_BATCH"): (16, 8),
    ("tep8_wide64_splitk", "MAX_ROWS"): (128, 64),
    ("tep8_wide64_splitk", "WIDE_TILE"): ("2cta", "64x64"),
    ("tep8_publish_splitk", "WIDE_TILE_MIN_ROWS"): (97, 65),
    ("tep8_publish_splitk", "WIDE_POOL_TILE"): ("64x64", "64x16"),
    ("tep8_publish_splitk", "WIDE_UNPOOL_TILE"): ("64x64", "64x32"),
    ("tep8_drafter_wide_gemms", "DEEP_STAGES"): (1, 0),
    ("tep8_drafter_wide_gemms", "PDL_CONV_MAX_ROWS"): (64, 32),
    ("wide_drafter_hops", "TEP8_TWO_SHOT_MIN_ROWS"): (25, 0),
    ("wide_drafter_hops", "TEP8_TWO_SHOT_MAX_ROWS"): (64, 128),
}


def test_the_tep8_gen_role_runs_the_stacked_winners_at_9_to_16_requests():
    """int5: a TEP8 gen server of 9..16 requests defaults the winners' parameters (the E2E X400 gen
    knob set, int4w_e2e_gen.env); 1..8 requests and the aggregated band keep the stock values; a knob
    still wins; the paged drafter context's pool groups only over a paged arena."""
    for arena in ("paged", "windowed"):
        for mbs in (9, 12, 15, 16):
            wide = table((107, 8), mbs, "gen", arena)
            for (lever, key), (value, _) in INT5_GEN_WIDE_DEFAULTS.items():
                assert wide.param(lever, key) == value, (lever, key, mbs, arena)
            assert wide.on("incremental_block_offsets"), (mbs, arena)
            assert wide.param("incremental_block_offsets", "POOL_GROUPS") == int(arena == "paged"), (mbs, arena)
        narrow = table((107, 8), 8, "gen", arena)
        for (lever, key), (_, stock) in INT5_GEN_WIDE_DEFAULTS.items():
            assert narrow.param(lever, key) == stock, (lever, key, arena)
        assert narrow.param("incremental_block_offsets", "POOL_GROUPS") == 0, arena
    agg = table((107, 8), 16, "agg", "paged")
    for (lever, key), (_, stock) in INT5_GEN_WIDE_DEFAULTS.items():
        assert agg.param(lever, key) == stock, (lever, key)
    # int7: the drafter QK fold is not a default (zia / zca).
    assert table((107, 8), 16, "gen", "windowed").param("tep8_drafter_wide_gemms", "QK_FOLD") == 0
    knobbed = table(
        (107, 8),
        16,
        "gen",
        "windowed",
        knobs(head_sampler_coop_fold__TOPP_ROUNDS=0, tep8_wide64_splitk__WIDE_TILE="64x64"),
    )
    assert knobbed.param("head_sampler_coop_fold", "TOPP_ROUNDS") == 0
    assert knobbed.param("tep8_wide64_splitk", "WIDE_TILE") == "64x64"
# --------------------------------------------------------------------------
# The TEP8 one-cluster SWA block, one cluster per request (swa_one_cluster_block__TEP8_WIDE_MIN_REQUESTS)
# --------------------------------------------------------------------------
#: md5 of ``swa_block._CL8_SRC`` (swa_cl8, the per-(request, KV head) build) as the base served it:
#: the wide build leaves it, and so every step it keeps, untouched.
_CL8_SRC_MD5 = "10334d867210ce7cb83cb59337e2eb51"


def _cl8_exact_view(src: str, exact: bool) -> str:
    """``src`` as the preprocessor sees it with (``exact``) or without ``-DCL8_EXACT``, for the CL8_EXACT
    directives only: each ``#ifdef CL8_EXACT`` block keeps its first branch iff ``exact``, its ``#else``
    branch otherwise; every other line as written."""
    out, stack = [], []
    for line in src.split("\n"):
        if line == "#ifdef CL8_EXACT":
            stack.append(["if", exact])
            continue
        if stack and line == "#else" and stack[-1][0] == "if":
            stack[-1] = ["else", not exact]
            continue
        if stack and line == "#endif" and stack[-1][0] in ("if", "else"):
            stack.pop()
            continue
        if all(keep for _, keep in stack):
            out.append(line)
    assert not stack
    return "\n".join(out)


def _swa_block_sources(exact: bool = False) -> dict:
    """``swa_block``'s CL8 kernel source literals, read from its file (it imports torch), as the default
    build (or, ``exact``, the CL8_EXACT build) compiles them (``_cl8_exact_view``)."""
    tree = ast.parse(_SWA_BLOCK.read_text())
    return {
        node.targets[0].id: _cl8_exact_view(node.value.value, exact)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("_CL8_SRC", "_CL8W_KERNEL")
    }


def _cl8_wide_band(swa) -> list:
    """The request counts of a TEP8 CL8 step (1..17) that launch swa_cl8w under ``swa``'s settings."""
    return [n for n in range(1, 18) if swa.cl8_tep8_takes(n) and swa.cl8_tep8_wide_takes(n)]


def test_the_tep8_cl8_wide_parameter_defaults_off():
    """TEP8_WIDE_MIN_REQUESTS defaults to 0 (no step runs swa_cl8w) on every TEP8 generation table
    that serves the lever, and swa_block's own default is 0; swa_cl8's source is the base's."""
    for mbs, arena in ((8, "windowed"), (4, "paged"), (16, "windowed")):
        lever_table = table((107, 8), mbs, "gen", arena)
        values, defaults = _settings(lever_table)["ops.attention.swa_block"]
        assert values["cl8_tep8_wide_min_requests"] == defaults["cl8_tep8_wide_min_requests"] == 0
        if lever_table.on("swa_one_cluster_block"):
            assert lever_table.param("swa_one_cluster_block", "TEP8_WIDE_MIN_REQUESTS") == 0
    swa = _swa_block_host()
    assert swa._CL8_TEP8_WIDE_MIN_REQUESTS == 0
    swa.configure(cl8_tep8=True, cl8_tep8_min_requests=2, cl8_tep8_max_requests=16, one2=True)
    assert _cl8_wide_band(swa) == []
    sources = _swa_block_sources()
    md5 = hashlib.md5(sources["_CL8_SRC"].encode(), usedforsecurity=False).hexdigest()
    assert md5 == _CL8_SRC_MD5, "swa_cl8's source changed"


@pytest.mark.parametrize("value", [9, 12, 16])
def test_the_tep8_cl8_wide_knob_reaches_swa_block(value):
    """swa_one_cluster_block__TEP8_WIDE_MIN_REQUESTS (0..16) turns no lever on or off and changes no
    path; it reaches ``swa_block.configure`` as ``cl8_tep8_wide_min_requests`` only while the lever is
    on (off, the module keeps 0)."""
    # int7b: CL8 is LOSSY, on only by its knob (both tables turn it on).
    base = table((107, 8), 8, "gen", "windowed", knobs(swa_one_cluster_block="on"))
    knob = knobs(
        swa_one_cluster_block="on",
        swa_one_cluster_block__TEP8_MAX_REQUESTS=16,
        swa_one_cluster_block__TEP8_WIDE_MIN_REQUESTS=value,
    )
    moved = table((107, 8), 8, "gen", "windowed", knob)
    assert moved.paths() == base.paths()
    assert rule_levers_on(moved) == rule_levers_on(base)
    assert moved.param("swa_one_cluster_block", "TEP8_WIDE_MIN_REQUESTS") == value
    swa_call = _settings(moved)["ops.attention.swa_block"][0]
    assert swa_call["cl8_tep8_wide_min_requests"] == value
    assert swa_call["cl8_tep8_max_requests"] == 16
    off = table((107, 8), 8, "gen", "windowed", {**knob, **knobs(swa_one_cluster_block="off")})
    values, defaults = _settings(off)["ops.attention.swa_block"]
    assert values["cl8_tep8_wide_min_requests"] == defaults["cl8_tep8_wide_min_requests"] == 0


@pytest.mark.parametrize("value", [-1, 17, "x"])
def test_the_tep8_cl8_wide_parameter_refuses_values_outside_0_to_16(value):
    with pytest.raises(ValueError, match="expected an integer in 0..16"):
        table(
            (107, 8),
            8,
            "gen",
            "windowed",
            knobs(swa_one_cluster_block__TEP8_WIDE_MIN_REQUESTS=value),
        )


@pytest.mark.parametrize("value", [2, 9, 12, 16])
def test_swa_block_configure_takes_the_tep8_cl8_wide_min_requests(value):
    """``cl8_tep8_wide_min_requests`` (0, or 2..16) picks the CL8 steps that launch swa_cl8w: those of
    the CL8 band with at least that many requests; the band itself does not move, and the steps below
    it keep swa_cl8 (the 5..11-request path unchanged at 12)."""
    swa = _swa_block_host()
    swa.configure(
        cl8_tep8=True,
        cl8_tep8_min_requests=2,
        cl8_tep8_max_requests=16,
        cl8_tep8_wide_min_requests=value,
        one2=True,
    )
    assert swa._CL8_TEP8_WIDE_MIN_REQUESTS == value
    assert _cl8_tep8_band(swa) == list(range(2, 17))
    assert _cl8_wide_band(swa) == list(range(value, 17))
    assert swa.log[-1] == (
        "[musespark_open] swa_block CL8 gate: cl8_tep8_wide_min_requests "
        f"{value}: TEP8 steps of {value}+ requests run one 8-CTA cluster per request over both KV "
        "heads (swa_cl8w, one wave up to 22)"
    )
    # Only inside the CL8 band: at the default 5..8 band, swa_cl8w takes value..8 at most.
    swa.configure(cl8_tep8=True, cl8_tep8_wide_min_requests=value, one2=True)
    assert _cl8_wide_band(swa) == list(range(max(value, 5), 9))
    for bad in (1, 17, -1):
        with pytest.raises(ValueError, match="cl8_tep8_wide_min_requests"):
            swa.configure(cl8_tep8=True, cl8_tep8_wide_min_requests=bad, one2=True)
        assert swa._CL8_TEP8_WIDE_MIN_REQUESTS == value
    swa.configure()
    assert swa._CL8_TEP8_WIDE_MIN_REQUESTS == 0 and _cl8_wide_band(swa) == []


def test_swa_cl8w_is_swa_cl8_per_head():
    """swa_cl8w is swa_cl8's kernel with the fixed edits that run the request's two KV heads in turn
    in one cluster: the shared helpers are swa_cl8's text, the TEP8 build only, head 0 from the
    registers swa_cl8 loads, head 1 staged before the dependency wait, one receive buffer and
    mbarrier per head, the cluster wait once, both merges after both passes."""
    sources = _swa_block_sources()
    cl8, kernel = sources["_CL8_SRC"], sources["_CL8W_KERNEL"]
    assert cl8.count('extern "C" __global__') == 1
    assert kernel.count('extern "C" __global__') == 1 and "swa_cl8w(" in kernel
    assert '#error "swa_cl8w is the TEP8 build (2 KV heads per rank)"' in kernel
    # The kernel body: swa_cl8's, line for line, except the edited lines.
    base = cl8.split('extern "C" __global__')[1].splitlines()
    wide = kernel.split('extern "C" __global__')[1].splitlines()
    added = [line for line in wide if line not in base]
    removed = [line for line in base if line not in wide]
    # The signature, head index, mbarrier init (now per head), head-0 load, staged head-1 read,
    # receive buffers / mbarriers of the push and the merge, and the cluster wait (now once).
    assert len(removed) == 12, removed
    assert any("cp.async.cg.shared.global" in line for line in added)
    assert any("for (int h = 0; h < HKV; ++h) {" in line for line in added)
    assert any(
        'if (h == 0) asm volatile("barrier.cluster.wait.acquire.aligned;"' in line for line in added
    )
    assert sum("BAR_OFF + 8 * h)" in line for line in added) == 2  # push and merge
    assert sum("BAR_OFF + 8 * hb)" in line for line in added) == 1  # one mbarrier per head
    assert sum("h ? RECV1_OFF : RECV_OFF" in line for line in added) == 2


def test_a_cache_hit_of_the_tep8_cl8_build_compiles_swa_cl8w():
    """The TEP8 build compiled before ``cl8_tep8_wide_min_requests`` was set (W4 first, or an earlier
    configure): the next eager call (a cache hit) compiles swa_cl8w too, so a capture finds it; never
    inside a capture, never without the setting, never twice."""
    tree = ast.parse(_SWA_BLOCK.read_text())
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_cl8_function")
        or (
            isinstance(node, ast.Assign)
            and _target_names(node) <= {"_CL8_TEP8_WIDE_MIN_REQUESTS", "_CL8_EXACT"}
        )
    ]
    capturing = [False]
    cuda = types.SimpleNamespace(
        is_current_stream_capturing=lambda: capturing[0], current_device=lambda: 0
    )
    host = _module("_cl8_function_host", torch=types.SimpleNamespace(cuda=cuda), _CL8_FN={})
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_SWA_BLOCK), "exec"), host.__dict__)
    calls = []
    real = host._cl8_function

    def spy(device, pos64, hkv, wide=False):
        calls.append((pos64, hkv, wide))
        if wide:  # stands for the NVRTC build
            host._CL8_FN[(0, bool(pos64), int(hkv), "w")] = "swa_cl8w"
            return "swa_cl8w"
        return real(device, pos64, hkv, wide)

    host._cl8_function = spy
    device = types.SimpleNamespace(index=0)
    host._CL8_FN[(0, False, 2)] = "swa_cl8"
    assert host._cl8_function(device, False, 2) == "swa_cl8" and calls == [(False, 2, False)]
    host._CL8_TEP8_WIDE_MIN_REQUESTS = 12
    capturing[0] = True
    calls.clear()
    assert host._cl8_function(device, False, 2) == "swa_cl8" and calls == [(False, 2, False)]
    capturing[0] = False
    calls.clear()
    assert host._cl8_function(device, False, 2) == "swa_cl8"
    assert calls == [(False, 2, False), (False, 2, True)]
    assert (0, False, 2, "w") in host._CL8_FN
    calls.clear()
    assert host._cl8_function(device, False, 2) == "swa_cl8" and calls == [(False, 2, False)]
    # The TEP4 build (4 KV heads) never compiles swa_cl8w.
    host._CL8_FN[(0, False, 4)] = "swa_cl8 tep4"
    calls.clear()
    assert host._cl8_function(device, False, 4) == "swa_cl8 tep4" and calls == [(False, 4, False)]


# --------------------------------------------------------------------------
# The exact one-cluster SWA block (swa_one_cluster_block__EXACT): the replaced path's arithmetic
# --------------------------------------------------------------------------
def test_the_cl8_exact_parameter_is_fixed_on_and_reaches_swa_block():
    """int9: EXACT is fixed at 1 (the original f16 CL8 build is not selectable): with the lever on, configure
    gets cl8_exact; with it off (a knob, or a table where CL8 does not run), nothing is passed."""
    base = table((107, 8), 8, "gen", "windowed")
    assert base.on("swa_one_cluster_block")
    assert base.param("swa_one_cluster_block", "EXACT") == 1
    values, defaults = _settings(base)["ops.attention.swa_block"]
    assert values["cl8_exact"] is True and values["cl8_tep8"] is True and "cl8_exact" not in defaults
    off = table((107, 8), 8, "gen", "windowed", knobs(swa_one_cluster_block="off"))
    values = _settings(off)["ops.attention.swa_block"][0]
    assert "cl8_exact" not in values and values["cl8_tep8"] is False
    agg = _settings(table((107, 8), 8, "agg", "windowed"))["ops.attention.swa_block"][0]
    assert "cl8_exact" not in agg
    for bad in (0, 2):
        with pytest.raises(ValueError, match="expected an integer in 1..1"):
            table((107, 8), 8, "gen", "windowed", knobs(swa_one_cluster_block__EXACT=bad))


def test_swa_block_cl8_exact_logs_error_without_a_cl8_path():
    swa = _swa_block_host()
    swa.configure(cl8_exact=True)
    assert swa._CL8_EXACT is True
    assert any(line.startswith("ERROR") and "cl8_exact is set but no CL8 path is on" in line for line in swa.log)
    swa.log.clear()
    swa.configure(cl8_tep8=True, one2=True, cl8_exact=True)
    assert any(line.startswith("[musespark_open] swa_block CL8 cl8_exact:") for line in swa.log)
    assert not any(line.startswith("ERROR") for line in swa.log)
    swa.configure()
    assert swa._CL8_EXACT is False


def test_the_cl8_exact_build_is_keyed_and_compiled_apart():
    src = _SWA_BLOCK.read_text()
    assert '+ (("exact", CL8_EXACT_EX2, CL8_EXACT_DIV) if _CL8_EXACT else ())' in src
    assert 'opts.append(b"-DCL8_EXACT")' in src


@pytest.mark.parametrize("name", ["_CL8_SRC", "_CL8W_KERNEL"])
def test_the_cl8_exact_build_uses_the_replaced_paths_operations(name):
    """The CL8_EXACT view of each kernel: sc = s * qk_scale (fp32 mul) before the mask, one row max over
    the CTA's partition (both key halves, floored at -1e30), p = exp2(sc - m) with Triton's exp2 in fp32,
    fp32 row sums of p, bf16 P and V into a bf16 P.V MMA, partial (acc * dequant, m, l) with the halves'
    sums added, the merge's exp2(m - m_max) / fp32 products and sums / Triton's division, dequant by
    Triton's division; no f16 exponential or f16 operand anywhere in the attention."""
    exact = " ".join(_swa_block_sources(exact=True)[name].split())
    default = " ".join(_swa_block_sources()[name].split())
    for statement in (
        "float a = f_mul(sa[n][e], qk_scale), b = f_mul(sa[n][2 + e], qk_scale);",
        "if ((unsigned)(j - jlo_lo) > (unsigned)(jhi_lo - jlo_lo)) a = NEG_INF;",
        "m_lo = fmaxf(m_lo, mx[((rb * 2 + (kh ^ 1)) * 2) * 32 + lane]);",
        "m_lo = fmaxf(m_lo, -1.0e30f);",
        "const float p0 = t_ex2(f_sub(sa[n][0], m_lo)), p1 = t_ex2(f_sub(sa[n][1], m_lo));",
        "l_lo = f_add(f_add(l_lo, p0), p1);",
        "pl[n] = bf16x2(p0, p1);",
        "mma16816bf(oa[2 * q], a0, a1, a2, a3, b0, b1);",
        "const float L_lo = f_add(l_lo, scr[34 * 32]), L_hi = f_add(l_hi, scr[35 * 32]);",
        "const float o0 = f_mul(f_add(oa[n][0], scr[(4 * n + 0) * 32]), dequant);",
        "st_async2(ra_lo + 64 * 4, m_lo, L_lo, rb_lo);",
        "const float w = ls[s] > 0.f ? t_ex2(f_sub(ms[s], mmax)) : 0.f;",
        "den = f_add(den, f_mul(w, ls[s]));",
        "n0 = f_add(n0, f_mul(o.x, w)); n1 = f_add(n1, f_mul(o.y, w));",
        "const float a0 = bf16r(t_div(n0, den)), a1 = bf16r(t_div(n1, den));",
        "const float dequant = t_div(1.0f, scale);",
        "a.x = e4m3x2_bf16x2(v.x);",
        "w[p] = e4m3x2_bf16x2(e);",
    ):
        assert statement in exact, statement
        assert statement not in default, statement
    body = exact.split('extern "C" __global__')[1]
    for gone in ("ex2h2(", "f16x2(fmaf(", "mma16816(", "e4m3x2_f16x2(v.", "f_ex2(", "__fdiv_rn(", "n0 / den"):
        assert gone not in body, gone
    src = _SWA_BLOCK.read_text()
    assert "opts.append(f'-DTRITON_EX2=\"{CL8_EXACT_EX2}\"'.encode())" in src
    assert "opts.append(f'-DTRITON_DIV=\"{CL8_EXACT_DIV}\"'.encode())" in src


def test_verify_grid_leaves_accepted_given_steps_to_the_loop_kernel():
    """gint4-zf: VERIFY_GRID's per-request kernel has no ACCEPTED_GIVEN mode (upstream #21789's
    rejection_fold), so a given-counts step keeps the loop kernel, which reads them."""
    source = (_TARGET / "ops/spec/dflash_verify_fold.py").read_text()
    assert "if GRID and not pre_wait and force_mode != ACCEPTED_GIVEN:" in source
    grid = source[source.index("def _dflash_verify_fold_grid_kernel(") : source.index("def _launch_grid(")]
    assert "given_counts" not in grid.split('"""')[0]
