# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The lever table: which installable pieces of this target a model is built with, and why.

A lever is any installable piece of the target: every entry of ``dispatch.FAST_PATHS``,
the fast paths themselves and the refinements and installs registered beside them. Once
per process ``dispatch.compose`` resolves one ``LeverTable`` from the composition's served
set, the ``Deployment`` (the GPU, the tensor-parallel size, the engine's
``max_batch_size`` and ``max_num_tokens``, the server's role and the drafter's context
arena), the structural rules ``dispatch`` derives from it, and the environment snapshot
(``environment``). The installers then ask the ``dispatch.Plan`` built from the table.

Knobs. These are the only variables the target reads for levers::

    MUSESPARK_OPEN_LEVER_<LEVER>=auto|on|off          one per lever; unset, empty or auto: its default
    MUSESPARK_OPEN_LEVER_<LEVER>__<PARAM>=<value>      a parameter the lever declares (Lever.params)
    MUSESPARK_OPEN_LEVER__DEFAULT=on|off               off: every lever off but the upstream ones' stock rule
    MUSESPARK_OPEN_LEVER__ROLE=auto|agg|ctx|gen        the server's role; unset or auto: derived

``<LEVER>`` is the lever's name upper-cased; ``__`` introduces a parameter or a resolver
setting and never occurs inside a lever name. Values are case-insensitive and
whitespace-stripped. Two shared-path variables are read as facts and never set or retired
here: ``TRTLLM_KV_DEFER_CLOSE`` (on iff exactly ``1``, the KV cache manager's own test) and
``UCX_RNDV_THRESH`` (set or not).

The table. ``LEVERS`` is the one list of this target's levers: each row is a ``Lever`` with its
``kind`` (what it changes, ``Kind``) and ``on_by_default``, the condition under which it is on
when no knob speaks for it. Every other deployment runs it only when its knob says ``on``
(opt-in). The conditions are written from a few combinators: ``ALWAYS`` (wherever it can run),
``NEVER`` (opt-in only), ``unless_on(...)``, ``gated(...)`` (a lossless-numeric lever's own
accuracy gates, ``Lever.gates``, must hold), ``in_band(...)`` (aggregated vs disaggregated
deployments, ``band_of``), and the target's deployment rules (``_at_most``,
``_on_single_request``, ...).

Semantics, in the order a lever is resolved:

* structural: a fact the lever's code needs (an arena kind, a fact variable, the drafter's
  logit transform, rejection sampling; ``dispatch.structural_rules``, ``Lever.excluded``)
  excludes it; nothing overrides that;
* dependency: a lever it builds on (``dispatch.FastPath.depends_on``) or requires
  (``Lever.requires``) is off;
* knob: ``off`` removes it, and with it every lever built on it or requiring it; ``on``
  keeps it whatever its default says, and turns on the levers it builds on or requires;
* ``__DEFAULT=off``: a lever with a ``stock`` rule (the upstream sm107 compositions' own
  levers) runs that rule; every other lever is off;
* default: its row's ``on_by_default``.

``on`` for a lever the composition does not serve, that a structural rule excludes, whose
dependency or requirement is off, or that conflicts with another lever that is on, raises.
So does every grammar error and every retired switch (``RETIRED``); ``UCX_RNDV_THRESH`` on
a server with a cache transceiver (a ``ctx`` or ``gen`` role, or an unhinted one); and
``context_fp8_dense`` on where a decode step can reach its ``MIN_ROWS``. The resolver
reads no device and no module of this target: everything it needs is an argument, so it
runs on any host.
"""

from __future__ import annotations

import dataclasses
import difflib
import enum
import json
import os
import types
from typing import Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

#: The draft tokens per step of the one speculative-decoding mode the compositions serve.
DRAFT_LEN = 7
#: Decode rows one request brings to a step: its last accepted token and its drafts.
ROWS_PER_REQUEST = DRAFT_LEN + 1
#: The fewest decode rows of a step of two requests, where the wide TP4 paths start.
WIDE_MIN_ROWS = 2 * ROWS_PER_REQUEST
#: Aggregated sm107 TEP4 servers of at least this max_batch_size run the high-concurrency
#: selection: no wide decode paths or cooperative head fold, the graph-start QKV prefetch,
#: and the parallel windowed context commit from one request.
HC_MIN_BATCH = 48

#: The prefix of every lever knob.
PREFIX = "MUSESPARK_OPEN_LEVER_"
#: Separates a lever name from a parameter key.
SEP = "__"
#: ``on`` (the default): every lever by its row's ``on_by_default``; ``off``: every lever off but
#: the upstream ones' ``stock`` rule (a stock engine).
DEFAULT_KNOB = PREFIX + "_DEFAULT"
#: The server's role: ``auto`` (derived, the default), ``agg``, ``ctx`` or ``gen``.
ROLE_KNOB = PREFIX + "_ROLE"
#: The shared-path facts the resolver reads.
KV_DEFER_CLOSE_ENV = "TRTLLM_KV_DEFER_CLOSE"
UCX_RNDV_THRESH_ENV = "UCX_RNDV_THRESH"

AUTO = "auto"
ON = "on"
OFF = "off"
LEVER_VALUES = (AUTO, ON, OFF)
DEFAULT_VALUES = (ON, OFF)

#: The decision sources, in the order a lever is resolved.
SOURCES = ("structural", "dependency", "knob", "stock", "default")

#: Retired switches and what replaced them. Setting one, to any value, raises.
RETIRED: Mapping[str, str] = types.MappingProxyType(
    {
        "MUSESPARK_OPEN_PREP_PACKED_ROWS": PREFIX + "PACKED_PREP_APPEND",
        "MUSESPARK_OPEN_CTX_COMMIT_TUNED": PREFIX + "WINDOWED_COMMIT_PARALLEL",
        "MUSESPARK_OPEN_CONTEXT_FP8_DENSE": PREFIX + "CONTEXT_FP8_DENSE",
        "MUSESPARK_OPEN_CONTEXT_FP8_DENSE_MIN_ROWS": PREFIX + "CONTEXT_FP8_DENSE__MIN_ROWS",
        "MUSESPARK_OPEN_CONTEXT_FP8_DENSE_PROJECTIONS": PREFIX + "CONTEXT_FP8_DENSE__PROJECTIONS",
        "MUSESPARK_OPEN_CONTEXT_FP8_DENSE_GATHER": PREFIX + "CONTEXT_FP8_ATTN_GATHER",
        "TRTLLM_MUSESPARK_TEP_WIDE_DIRECT": PREFIX + "WIDE_DIRECT_TRANSPORTS",
        "TRTLLM_MUSESPARK_TEP_WIDE_ROUTER_ROWS": PREFIX + "WIDE_ROUTER_PER_ROW",
        "TRTLLM_MUSESPARK_TEP_WIDE_OPROJ_PUBLISH": PREFIX + "WIDE_OPROJ_PUBLISH",
        "MUSESPARK_OPEN_STEP_LOOP_ONE_STREAM": PREFIX + "STEP_LOOP_ON_EXECUTION_STREAM",
        "MUSESPARK_OPEN_INCREMENTAL_BLOCK_OFFSETS": PREFIX + "INCREMENTAL_BLOCK_OFFSETS",
        "MUSESPARK_OPEN_PS_SIDE_CHAIN": PREFIX + "DRAFTER_KV_SIDE_CHAIN",
        "MUSESPARK_OPEN_PS_EARLY_FC": PREFIX + "DRAFTER_FC_EARLY_FORK",
        "MUSESPARK_OPEN_PS_DRAFTER_PREAMBLE": PREFIX + "DRAFTER_PREAMBLE_FROM_COMMIT",
        "MUSESPARK_OPEN_PS_COOP_FOLD": PREFIX + "HEAD_SAMPLER_COOP_FOLD",
        "MUSESPARK_OPEN_PS_SYMM_GATHER": PREFIX + "HEAD_MULTIMEM_GATHER",
        "MUSESPARK_OPEN_PS_NO_START_PREFETCH": PREFIX + "GRAPH_START_QKV_PREFETCH=off",
        "MUSESPARK_OPEN_PS_NO_LM_HEAD_INBOX": PREFIX + "LM_HEAD_INBOX=off",
        "MUSESPARK_OPEN_PS_LAYER0_INPUT": "nothing: model_boundary builds layer 0's attention input",
        "MUSESPARK_OPEN_PS_STEP_COMMIT_PDL": "nothing: the step commit launch has no PDL variant",
        "MUSESPARK_OPEN_HEAD_INBOX_POLL_SCOPE": "nothing: the inbox polls are system-scope",
        "MUSESPARK_OPEN_HEAD_INBOX_RESET_SCOPE": "nothing: the inbox resets are system-scope",
        "MUSESPARK_OPEN_SPLITK_MAILBOX_V4": "nothing: the sm107 compositions build the split-K mailbox",
        "MUSESPARK_OPEN_SPLITK_OVERSIZED_SMEM": (
            "nothing: the sm107 compositions build split-K in oversized shared memory"
        ),
        "MUSESPARK_OPEN_SPLITK_RS": PREFIX + "SPLITK_POOL_SPLIT8",
        "MUSESPARK_OPEN_SPLITK_RS_ONLY": PREFIX + "SPLITK_POOL_SPLIT8",
        "MUSESPARK_OPEN_POOL_SPLITK8": PREFIX + "SPLITK_POOL_SPLIT8",
        "MUSESPARK_OPEN_TEP4_WIDE_GEMMS": PREFIX + "SPLITK_FRONTIER_TABLES",
        "MUSESPARK_OPEN_TEP4_NARROW_GEMMS": PREFIX + "SPLITK_FRONTIER_TABLES",
        "MUSESPARK_OPEN_TEP8_WIDE64_SPLITK": PREFIX + "TEP8_WIDE64_SPLITK",
        "MUSESPARK_OPEN_TEP8_WIDE64_KERNELS": PREFIX + "TEP8_WIDE64_SPLITK",
        "MUSESPARK_OPEN_TEP8_PUBLISH_SPLITK": PREFIX + "TEP8_PUBLISH_SPLITK",
        "MUSESPARK_OPEN_TEP8_DRAFTER_WIDE_GEMMS": PREFIX + "TEP8_DRAFTER_WIDE_GEMMS",
        "MUSESPARK_OPEN_TEP8_FC_SHARD_WIDE": PREFIX + "TEP8_FC_SHARD_WIDE",
        "MUSESPARK_OPEN_TEP8_FC_GATHER_PROGRAMS": PREFIX + "TEP8_FC_SHARD_WIDE",
        "MUSESPARK_OPEN_PREFETCH_JOIN": PREFIX + "WEIGHT_PREFETCH_AFTER_JOIN",
        "MUSESPARK_OPEN_DROP_WEIGHT_PREFETCH": PREFIX + "WEIGHT_PREFETCH_AFTER_JOIN",
        "MUSESPARK_OPEN_NEXT_QKV_PREFETCH_SKIP_ROWS": PREFIX + "NEXT_QKV_PREFETCH_SKIP",
        "MUSESPARK_OPEN_SWA_CL8": PREFIX + "SWA_ONE_CLUSTER_BLOCK",
        "MUSESPARK_OPEN_TEP8_SWA_CL8": PREFIX + "SWA_ONE_CLUSTER_BLOCK",
        # The GB300 port's name for the TEP8 CL8 band's first request count; here a parameter.
        "MUSESPARK_OPEN_TEP8_SWA_CL8_MIN_REQUESTS": (
            PREFIX + "SWA_ONE_CLUSTER_BLOCK" + SEP + "TEP8_MIN_REQUESTS"
        ),
        "MUSESPARK_OPEN_SWA_PREP_SPLIT": PREFIX + "SWA_PREP_SPLIT",
        "MUSESPARK_OPEN_SWA_ONE2": PREFIX + "SWA_EIGHT_REQUESTS",
        "MUSESPARK_OPEN_FA_PREP_FOLD_MULTI": PREFIX + "FULL_ATTENTION_FOLD_MULTI",
        "MUSESPARK_OPEN_FA_STAGES": PREFIX + "SPLIT_KV_DEEP_RINGS",
        "MUSESPARK_OPEN_MR_STAGES": PREFIX + "SPLIT_KV_DEEP_RINGS",
        "MUSESPARK_OPEN_BF16_HALF_WINDOW": PREFIX + "BF16_HALF_WINDOW",
        "MUSESPARK_OPEN_PREROUTED_META": PREFIX + "PREROUTED_EXPERTS",
        "MUSESPARK_OPEN_TEP8_PREROUTED": PREFIX + "PREROUTED_EXPERTS",
        # The GB300 port's name for the TP8 prerouted band (sm_103__tep8); here a parameter.
        "MUSESPARK_OPEN_TEP8_PREROUTED_MAX_ROWS": PREFIX
        + "PREROUTED_EXPERTS"
        + SEP
        + "TEP8_MAX_ROWS",
        "MUSESPARK_OPEN_PREROUTED_SO": "nothing: the experts run trtllm::fp4_block_scale_moe_prerouted",
        "MUSESPARK_OPEN_ROUTER_ROWS_TILING": "nothing: the sm107 compositions tile the per-row router shard",
        "MUSESPARK_OPEN_TEP8_RING_BUDDY_MAX_ROWS": PREFIX
        + "TEP8_RING_BUDDY_WIDE"
        + SEP
        + "MAX_ROWS",
        "MUSESPARK_OPEN_TEP8_RB_PDL": PREFIX + "TEP8_RING_BUDDY_PDL",
        "MUSESPARK_OPEN_TEP8_TILE_PLACEMENT": PREFIX + "TEP8_TILE_PLACEMENT",
        "MUSESPARK_OPEN_TEP8_TILE_PLACEMENT_MIN_ROWS": PREFIX + "TEP8_TILE_PLACEMENT",
        "MUSESPARK_OPEN_TEP8_ROUTER_PUSH": PREFIX + "TEP8_ROUTER_PUSH",
        "MUSESPARK_OPEN_TEP8_ROUTER_PUSH_MIN_ROWS": PREFIX + "TEP8_ROUTER_PUSH",
        "MUSESPARK_OPEN_TEP8_ROUTER_PUSH_ROUTE": PREFIX + "TEP8_ROUTER_PUSH",
        "MUSESPARK_OPEN_TEP8_ROUTER_PREWAIT": PREFIX + "TEP8_ROUTER_PREWAIT",
        "MUSESPARK_OPEN_TEP8_POOL_PROGRAMMATIC_FORK": PREFIX + "TEP8_POOL_PROGRAMMATIC_FORK",
        "MUSESPARK_OPEN_TEP8_WIDE_POOL_GATHER": PREFIX + "TEP8_WIDE_PROJECTION_GATHERS",
        "MUSESPARK_OPEN_TEP8_WIDE_UNPOOL_GATHER": PREFIX + "TEP8_WIDE_PROJECTION_GATHERS",
        "MUSESPARK_OPEN_FINALIZE_TRIGGER": PREFIX + "FINALIZE_ENTRY_TRIGGER",
        "MUSESPARK_OPEN_TEP8_FINALIZE_TRANSPORT_TOKENS": (
            PREFIX + "TEP8_FINALIZE_TRANSPORT_64" + SEP + "TOKENS"
        ),
        "MUSESPARK_OPEN_HOP_ENTRY_TRIGGER": PREFIX + "HOP_ENTRY_TRIGGER",
        "MUSESPARK_OPEN_TEP8_HOP_DIRECT": PREFIX + "TEP8_DIRECT_HOP",
        # The GB300 port's "lo-hi" band of the direct hop (sm_103__tep8); here MIN_ROWS / MAX_ROWS.
        "MUSESPARK_OPEN_TEP8_HOP_DIRECT_ROWS": PREFIX + "TEP8_DIRECT_HOP" + SEP + "MIN_ROWS",
        "MUSESPARK_OPEN_TEP8_HOP_EARLY": PREFIX + "TEP8_EARLY_MULTICAST_HOP",
        "MUSESPARK_OPEN_TEP8_HOP_MULTICAST": PREFIX + "TEP8_EARLY_MULTICAST_HOP",
        "MUSESPARK_OPEN_TEP8_HOP_PAD_ROWS": PREFIX + "TEP8_EARLY_MULTICAST_HOP",
        "MUSESPARK_OPEN_TEP8_HOP_R2": PREFIX + "TEP8_EARLY_MULTICAST_HOP",
        "MUSESPARK_OPEN_TEP8_HOP_R2_MIN_ROWS": PREFIX + "TEP8_EARLY_MULTICAST_HOP",
        "MUSESPARK_OPEN_TEP8_HOP_SMEM_PAD": PREFIX + "TEP8_EARLY_MULTICAST_HOP",
        "MUSESPARK_OPEN_TEP8_JOIN_PDL": PREFIX + "TEP8_PUBLISH_JOIN",
        "MUSESPARK_OPEN_PUBLISH_JOIN_CLUSTER": PREFIX + "TEP8_PUBLISH_JOIN",
        "MUSESPARK_OPEN_NORM_GATE_PDL_MAX_ROWS": PREFIX + "WIDE_NORM_GATE_PDL",
        "MUSESPARK_OPEN_STEP_COMMIT_V2": PREFIX + "STEP_COMMIT_V2",
        "MUSESPARK_OPEN_CTX_KV_GLUE": PREFIX + "DRAFTER_KV_GLUE",
        "MUSESPARK_OPEN_CTX_KV_FORK_EARLY": PREFIX + "DRAFTER_KV_FORK_BEFORE_HEAD",
        "MUSESPARK_OPEN_VP_MIN_ROWS": PREFIX + "VOCAB_PARALLEL_HEAD__MIN_ROWS",
        "MUSESPARK_OPEN_SM107_VOCAB_PARALLEL_HEAD": PREFIX + "VOCAB_PARALLEL_HEAD",
        "MUSESPARK_OPEN_TEP8_EXTRA_PATHS": "nothing: the sm107 tep8 composition serves those paths",
        "MUSESPARK_OPEN_DROP_PATHS": "the " + PREFIX + "<PATH>=off knobs",
        "MUSESPARK_OPEN_FC_SHARD_PREWARM": "nothing: the sm107 compositions prewarm the drafter fc",
        "MUSESPARK_OPEN_SP_SM107_PARTS": PREFIX + "CONTEXT_SP_EXTRA_PARTS__PARTS",
        "TRTLLM_MUSESPARK_SWA_BLOCKS_PER_PROGRAM": PREFIX + "SWA_BLOCKS_PER_PROGRAM",
        "TRTLLM_MUSESPARK_TEP_WIDE_FULL_FOLD": PREFIX + "FULL_ATTENTION_FOLD_MULTI",
        "TRTLLM_MUSESPARK_TEP_WIDE_QKV_PREFETCH": PREFIX + "WIDE_QKV_PREFETCH",
        "TRTLLM_MUSESPARK_TEP_WIDE_CAPTURE_FOLD": PREFIX + "WIDE_CAPTURE_FOLD",
        "MUSESPARK_OPEN_SM107_FMK_CONTEXT_FMHA": PREFIX + "FMK_CONTEXT_FMHA",
        "MUSESPARK_OPEN_CTX_KV_LATE_JOIN": PREFIX + "CONTEXT_KV_LATE_JOIN",
        "MUSESPARK_OPEN_HOP_CLUSTER": PREFIX + "HOP_CLUSTER_SPLIT" + SEP + "CLUSTER",
        "MUSESPARK_OPEN_HOP_CLUSTER_ROWS": PREFIX + "HOP_CLUSTER_SPLIT" + SEP + "ROWS",
        "MUSESPARK_OPEN_FC_INBOX_MAX_ROWS": PREFIX + "DRAFTER_FC_INBOX_WIDE" + SEP + "MAX_ROWS",
        "MUSESPARK_OPEN_FC_INBOX_PROGRAMS": PREFIX + "DRAFTER_FC_INBOX_WIDE" + SEP + "PROGRAMS",
        "MUSESPARK_OPEN_L061_TILE32_PIN": PREFIX + "L061_TILE32_PIN",
        "MUSESPARK_OPEN_ROUTER_STAGED_EPILOGUE": PREFIX + "ROUTER_STAGED_EPILOGUE",
        "MUSESPARK_OPEN_DECODE_QKV_MXFP8": PREFIX + "DECODE_QKV_MXFP8",
        "MUSESPARK_OPEN_CTX_SP_SM107_ROUTER_INPUT": (
            "nothing: sm107 context sequence parallelism always joins the router input"
        ),
        "MUSESPARK_OPEN_CTX_KV_FORK_POINT": "nothing: the fork point is the LM head's GEMM",
        "MUSESPARK_OPEN_HOP_PAYLOAD": "nothing: the wide hop moves BF16",
        "MUSESPARK_OPEN_FINALIZE_PAYLOAD": "nothing: the expert finalize moves BF16",
        "MUSESPARK_OPEN_ROUTER_WEIGHT_PROLOGUE": "nothing: this build has no such switch",
        "MUSESPARK_OPEN_POOL_FORK_AFTER_ROUTER": "nothing: this build has no such switch",
        "MUSESPARK_OPEN_SM107_UNPOOL_TGV": "nothing: this build has no such switch",
        "MUSESPARK_OPEN_DECODE_ROUTE_RECORD_DIR": "nothing: this build has no such switch",
        # The single-request decode schedule's switches (sm107 TEP8).
        "MUSESPARK_OPEN_MEGAKERNEL": PREFIX + "MEGAKERNEL",
        "MUSESPARK_OPEN_MEGAKERNEL_NB": PREFIX + "MEGAKERNEL_ONE_EXCHANGE_BOUNDARIES",
        "MUSESPARK_OPEN_MEGAKERNEL_BH": PREFIX + "MEGAKERNEL_TWO_TERM_POOL",
        "MUSESPARK_OPEN_MEGAKERNEL_RL": PREFIX + "MEGAKERNEL_ROUTER_BY_LINEARITY",
        "MUSESPARK_OPEN_MEGAKERNEL_FZ": PREFIX + "MEGAKERNEL_ZERO_EXCHANGE_BOUNDARY",
        "MUSESPARK_OPEN_TQ_TARGET_L0": PREFIX + "LAYER0_QKV_TABLE",
        "MUSESPARK_OPEN_PF_FUSED_PRE": "nothing: model_boundary builds layer 0's attention input",
        "MUSESPARK_OPEN_OB_PRE_VARIANT": "nothing: the fused pre has one variant",
        "MUSESPARK_OPEN_PV_L0_INPUT_MK": "nothing: model_boundary builds layer 0's attention input",
        "MUSESPARK_OPEN_PF_BF16_FRONT": PREFIX + "EDGE_BF16_MOE_FRONT",
        "MUSESPARK_OPEN_OB_BF16_DIRECT": PREFIX + "EDGE_BF16_MOE_FRONT",
        "MUSESPARK_OPEN_OB_BF16_MOE_PDL": "nothing: this build has no such switch",
        "MUSESPARK_OPEN_G2_L61_BOUNDARY": PREFIX + "LAST_LAYER_BOUNDARY",
        "MUSESPARK_OPEN_PV_GLUE_FUSED": PREFIX + "LAST_LAYER_BOUNDARY",
        "MUSESPARK_OPEN_PF_SIDE_FORK": PREFIX + "CONTEXT_KV_FORK_LAST_LAYER",
        "MUSESPARK_OPEN_PV_EARLY_FC_AFTER_HOP1": PREFIX + "CONTEXT_KV_FORK_LAST_LAYER",
        "MUSESPARK_OPEN_S2_SAMPLER": PREFIX + "DISTRIBUTED_HEAD_SAMPLER",
        "MUSESPARK_OPEN_PV_LOGITS_ONESHOT": PREFIX + "DISTRIBUTED_HEAD_SAMPLER",
        "MUSESPARK_OPEN_PV_LOGITS_POLL": "nothing: the one-shot logits gather polls one way",
        "MUSESPARK_OPEN_PT_VERIFY_PDL": PREFIX + "DISTRIBUTED_HEAD_SAMPLER",
        "MUSESPARK_OPEN_OX_DEJ": PREFIX + "DRAFTER_PREAMBLE_LATE_JOIN",
        "MUSESPARK_OPEN_DR3_CONV_FOLD": PREFIX + "DRAFTER_FOLDS",
        "MUSESPARK_OPEN_DR3_SILU_FOLD": PREFIX + "DRAFTER_FOLDS",
        "MUSESPARK_OPEN_SP_CF_V2": PREFIX + "DRAFTER_FOLDS",
        "MUSESPARK_OPEN_DG_SF_TACTIC": "nothing: the folded GEMV's tactic is fixed per architecture",
        "MUSESPARK_OPEN_OX_DE": PREFIX + "DRAFTER_FOLDS",
        "MUSESPARK_OPEN_PT_ATTN_PDL": PREFIX + "DRAFTER_FOLDS",
        "MUSESPARK_OPEN_TT_DRAFTER_L0": PREFIX + "DRAFTER_FOLDS",
        "MUSESPARK_OPEN_DR_DRAFT_TAIL": PREFIX + "DRAFT_HEAD_TAIL",
        "MUSESPARK_OPEN_GR_INLINE_COMMIT": PREFIX + "STEP_LOOP_INLINE_COMMITS",
        "MUSESPARK_OPEN_IA_FOLD": PREFIX + "STEP_LOOP_INLINE_COMMITS",
        # The fused rejection sampler's switches (sm107 TEP8).
        "MUSESPARK_OPEN_RS_FUSED": PREFIX + "REJECTION_SAMPLER",
        "MUSESPARK_OPEN_RS_FORCED": (
            "nothing: forced acceptance always leaves rejection sampling inert"
        ),
        # The resolver settings the one lever table replaced.
        PREFIX + "_BITWISE": "nothing: each lever's row in LEVERS says when it is on by default",
        PREFIX + "_NUMERICS": "the " + PREFIX + "<LEVER>=off knobs",
    }
)
#: Name prefixes only retired switches carry; any other name under them raises too.
RETIRED_PREFIXES: Mapping[str, str] = types.MappingProxyType(
    {
        "MUSESPARK_OPEN_PS_": "the " + PREFIX + "* knobs",
        "TRTLLM_MUSESPARK_TEP_WIDE_": "the " + PREFIX + "WIDE_* knobs",
        "MUSESPARK_OPEN_HEAD_INBOX_": "nothing: the inbox scopes are system-scope",
        "MUSESPARK_OPEN_MEGAKERNEL_": (
            "nothing: the megakernel's build options are fixed in its served build"
        ),
        "MUSESPARK_OPEN_RS_": (
            "nothing: the fused rejection sampler's q, chain and filters are fixed"
        ),
    }
)


def retired_replacement(name: str) -> Optional[str]:
    """What replaced the retired switch ``name``, or None for a name that is not retired."""
    if name in RETIRED:
        return RETIRED[name]
    for prefix, replacement in RETIRED_PREFIXES.items():
        if name.startswith(prefix):
            return replacement
    return None


class Kind(str, enum.Enum):
    """What a lever changes, which fixes what evidence ``auto`` needs (module docstring)."""

    #: A fast path validated with its composition: ``auto`` serves it where the rule says.
    COMPOSITION = "composition"
    #: Writes the bytes of the path it replaces.
    BITWISE = "lossless-bitwise"
    #: Same precision, another rounding or summation order: ``auto`` also needs a gate.
    NUMERIC = "lossless-numeric"
    #: Lower precision: never on under ``auto``.
    LOSSY = "lossy"
    #: Lossless, with an open safety check: never on under ``auto``.
    FLAGGED = "flagged"


class Role(str, enum.Enum):
    """The server's role: aggregated, disaggregated context or generation, or not known."""

    AGG = "agg"
    CTX = "ctx"
    GEN = "gen"
    UNKNOWN = "unknown"


class Arena(str, enum.Enum):
    """Where the DFlash worker keeps the drafter's context K/V."""

    #: A worker-owned arena of every context column.
    DENSE = "dense"
    #: A worker-owned arena of each slot's last window (``TRTLLM_DFLASH_WINDOWED_CTX_ARENA``).
    WINDOWED = "windowed"
    #: Pages of a KV cache manager.
    PAGED = "paged"
    #: No drafter: a context server without speculative decoding
    #: (``dispatch.drafterless_context``); every drafter lever is left out.
    NONE = "none"


@dataclasses.dataclass(frozen=True)
class Facts:
    """The shared-path variables the resolver reads; it never sets or retires them."""

    #: ``TRTLLM_KV_DEFER_CLOSE`` is exactly ``1``.
    kv_defer_close: bool = False
    #: ``UCX_RNDV_THRESH`` is set to a non-empty value.
    ucx_rndv_thresh_set: bool = False

    def describe(self) -> str:
        return (
            f"deferred close {'on' if self.kv_defer_close else 'off'}, "
            f"{UCX_RNDV_THRESH_ENV} {'set' if self.ucx_rndv_thresh_set else 'unset'}"
        )


@dataclasses.dataclass(frozen=True)
class Deployment:
    """The facts a lever's deployment rule reads, fixed when the model is built."""

    sm: int
    tp: int
    max_batch_size: Optional[int]
    max_num_tokens: Optional[int]
    role: Role
    #: Where the role came from: ``hint``, ``no cache transceiver``, ``no hint on a
    #: disaggregated server`` or ``no engine facts``.
    role_source: str
    arena: Arena
    facts: Facts = Facts()
    rows_per_request: int = ROWS_PER_REQUEST
    #: Whether the drafter declares a logit transform (``DFlashForCausalLM.logit_multiplier`` /
    #: ``logit_softcap``); None when there is no drafter to read it from.
    draft_logit_transform: Optional[bool] = None
    #: Whether the speculative config asks for rejection sampling (``use_rejection_sampling``)
    #: once forced acceptance (``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS``) has left it
    #: inert; None when there is no speculative config to read it from.
    rejection_sampling: Optional[bool] = None

    def decodes(self) -> bool:
        """Whether this server runs decode steps (an unknown role counts as one that does)."""
        return self.role is not Role.CTX

    def reaches(self, rows: int) -> bool:
        """Whether some decode step can have at least ``rows`` rows (true for an unknown
        ``max_batch_size``)."""
        return self.max_batch_size is None or self.max_batch_size * self.rows_per_request >= rows

    def describe(self) -> str:
        transform = (
            ""
            if self.draft_logit_transform is None
            else f", drafter logit transform {'declared' if self.draft_logit_transform else 'none'}"
        )
        rejection = (
            ""
            if self.rejection_sampling is None
            else f", rejection sampling {'on' if self.rejection_sampling else 'off'}"
        )
        return (
            f"sm{self.sm} tep{self.tp}, {self.role.value} ({self.role_source}), max_batch_size "
            f"{_count(self.max_batch_size)}, max_num_tokens {_count(self.max_num_tokens)}, "
            f"{self.arena.value} arena{transform}{rejection}"
        )

    def as_dict(self) -> dict:
        out = {
            "sm": self.sm,
            "tp": self.tp,
            "max_batch_size": self.max_batch_size,
            "max_num_tokens": self.max_num_tokens,
            "role": self.role.value,
            "role_source": self.role_source,
            "arena": self.arena.value,
        }
        if self.draft_logit_transform is not None:
            out["draft_logit_transform"] = self.draft_logit_transform
        if self.rejection_sampling is not None:
            out["rejection_sampling"] = self.rejection_sampling
        return out


def _count(value: Optional[int]) -> str:
    return "unknown" if value is None else str(value)


class Band(str, enum.Enum):
    """Which band's lever selection ``auto`` serves a deployment (module docstring, ``band_of``)."""

    #: Aggregated sm107 servers, at any TEP size, ``max_batch_size`` and arena.
    AGGREGATED_SM107 = "aggregated sm107"
    #: Every other deployment: context, generation and unhinted disaggregated servers.
    DEFAULT = "default"


def band_of(d: Deployment) -> Band:
    """The band ``d`` belongs to: ``AGGREGATED_SM107`` iff it is an aggregated server.

    Only the role decides; the TEP size, ``max_batch_size``, the arena, the drafter's logit
    transform and rejection sampling do not move a deployment between bands. An unknown role
    (an unhinted disaggregated server, or no engine facts) is ``DEFAULT``.
    """
    if d.role is Role.AGG:
        return Band.AGGREGATED_SM107
    return Band.DEFAULT


@dataclasses.dataclass(frozen=True)
class Param:
    """A heuristic threshold a lever declares, overridable by ``<LEVER>__<KEY>``.

    An integer in ``low..high`` (with ``values``, one of ``values``); with ``choices``, a
    non-empty subset of ``choices`` (comma-separated, kept in ``choices`` order), or with
    ``one`` exactly one of them (the name itself); with ``ranges``, a non-empty list of
    inclusive integer ranges in ``low..high`` (``"lo-hi,lo-hi,n"``, as sorted
    ``(lo, hi)`` pairs). A lossy lever's range is a subset of
    the values its accuracy gate covered. ``default_for``, when set, gives the default
    per deployment; the knob still wins over it.
    """

    key: str
    default: object
    low: int = 1
    high: int = 1 << 30
    choices: Tuple[str, ...] = ()
    default_for: Optional[Callable[["Deployment"], object]] = None
    ranges: bool = False
    #: With ``choices``: the value is exactly one name (a ``str``), not a subset.
    one: bool = False
    #: An integer parameter's allowed values (empty: every integer in ``low..high``).
    values: Tuple[int, ...] = ()

    def default_of(self, deployment: "Deployment") -> object:
        return self.default if self.default_for is None else self.default_for(deployment)

    def parse(self, raw: str) -> object:
        """The value ``raw`` gives this parameter.

        Raises:
            ValueError: for a value outside the parameter's range or choices.
        """
        text = raw.strip().lower()
        if self.choices and self.one:
            if text not in self.choices:
                raise ValueError(f"expected one of {list(self.choices)}, got {raw!r}")
            return text
        if self.choices:
            names = [name.strip() for name in text.split(",") if name.strip()]
            unknown = sorted(set(names) - set(self.choices))
            if not names or unknown:
                raise ValueError(
                    f"expected a non-empty comma-separated subset of {list(self.choices)}, got "
                    f"{raw!r}"
                )
            return tuple(choice for choice in self.choices if choice in names)
        if self.ranges:
            return self._parse_ranges(raw, text)
        if self.values:
            try:
                value = int(text, 10)
            except ValueError:
                value = None
            if value not in self.values:
                raise ValueError(f"expected one of {list(self.values)}, got {raw!r}")
            return value
        try:
            value = int(text, 10)
        except ValueError:
            raise ValueError(
                f"expected an integer in {self.low}..{self.high}, got {raw!r}"
            ) from None
        if not self.low <= value <= self.high:
            raise ValueError(f"expected an integer in {self.low}..{self.high}, got {value}")
        return value

    def _parse_ranges(self, raw: str, text: str) -> Tuple[Tuple[int, int], ...]:
        pairs = []
        for item in (part.strip() for part in text.split(",")):
            if not item:
                continue
            lo_text, _, hi_text = item.partition("-")
            try:
                lo, hi = int(lo_text, 10), int(hi_text or lo_text, 10)
            except ValueError:
                raise ValueError(
                    f"expected ranges lo-hi,... in {self.low}..{self.high}, got {raw!r}"
                ) from None
            if not self.low <= lo <= hi <= self.high:
                raise ValueError(f"range {item!r} is outside {self.low}..{self.high} or descending")
            pairs.append((lo, hi))
        if not pairs:
            raise ValueError(f"expected ranges lo-hi,... in {self.low}..{self.high}, got {raw!r}")
        return tuple(sorted(pairs))


@dataclasses.dataclass(frozen=True)
class Gate:
    """A deployment class where a lossless-numeric lever passed, or must pass, an accuracy gate.

    ``scope`` says the class in words; the evidence lives with the measurements, not here.
    """

    holds: Callable[["Deployment"], bool]
    scope: str


#: ``(on, the rule in words)`` from a deployment, the decisions so far and the lever's parameters.
RuleFn = Callable[["Deployment", "Resolving", Mapping[str, object]], Tuple[bool, str]]


@dataclasses.dataclass(frozen=True)
class Lever:
    """One row of the lever table: what the lever is and when it is on by default.

    ``on_by_default`` decides it for a deployment no knob speaks for; anywhere else it runs only
    when its knob says ``on`` (opt-in). ``stock`` is the upstream target's own rule for a lever
    the upstream sm107 compositions also serve: what ``__DEFAULT=off`` keeps (None: off there).
    ``gates`` are the accuracy gates a lossless-numeric lever's ``gated(...)`` default needs, and
    the scope a knob ``on`` outside them is warned about. ``after`` names the decisions its
    conditions read; ``requires`` levers that must be on; ``conflicts`` levers that may not be on
    with it; ``excluded`` a structural reason it cannot be on, from the decisions so far.
    """

    kind: Kind
    on_by_default: RuleFn
    stock: Optional[RuleFn] = None
    params: Tuple[Param, ...] = ()
    gates: Tuple[Gate, ...] = ()
    after: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()
    conflicts: Tuple[str, ...] = ()
    excluded: Optional[Callable[["Deployment", "Resolving"], Optional[str]]] = None

    def param(self, key: str) -> Param:
        for param in self.params:
            if param.key == key:
                return param
        raise KeyError(key)


@dataclasses.dataclass(frozen=True)
class Decision:
    """One lever's resolution: on or off, by which step, the rule in words, its parameters."""

    name: str
    on: bool
    source: str
    why: str
    params: Mapping[str, object] = dataclasses.field(default_factory=dict)


class Resolving:
    """The decisions made so far, as a lever's conditions read them (``Lever.after``)."""

    def __init__(self, served: FrozenSet[str], decisions: Mapping[str, Decision]) -> None:
        self._served = served
        self._decisions = decisions
        #: The lever being decided and its parameters (``resolve`` sets it): its own
        #: ``Lever.excluded`` reads them through ``param``.
        self.pending: Optional[Tuple[str, Mapping[str, object]]] = None

    def on(self, name: str) -> bool:
        """Whether ``name`` resolved on; False for a lever the composition does not serve.

        Raises:
            RuntimeError: for a served lever not decided yet (a rule that reads it must
                name it in ``Lever.after``).
        """
        if name not in self._served:
            return False
        decision = self._decisions.get(name)
        if decision is None:
            raise RuntimeError(f"lever rule reads {name}, which is not decided yet")
        return decision.on

    def param(self, name: str, key: str) -> object:
        """``name``'s parameter ``key`` as resolved (for the lever being decided, as its knob
        sets it), or its default when ``name`` is not served."""
        decision = self._decisions.get(name)
        if decision is not None and key in decision.params:
            return decision.params[key]
        if self.pending is not None and self.pending[0] == name and key in self.pending[1]:
            return self.pending[1][key]
        return LEVERS[name].param(key).default


# --------------------------------------------------------------------------
# The deployment rules
# --------------------------------------------------------------------------
_NO_DECODE = "the server runs no decode step"
_HC = f"aggregated sm107 tep4 servers of max_batch_size {HC_MIN_BATCH} or more"


def _hc_selection(d: Deployment) -> bool:
    """Whether ``d`` runs the high-concurrency selection (``HC_MIN_BATCH``)."""
    return (
        d.tp == 4
        and d.role is Role.AGG
        and d.max_batch_size is not None
        and d.max_batch_size >= HC_MIN_BATCH
    )


def _off_at_hc(rule: "RuleFn") -> "RuleFn":
    """``rule``, except off on the high-concurrency selection."""

    def wrapped(d: Deployment, t: "Resolving", p: Mapping[str, object]) -> Tuple[bool, str]:
        if _hc_selection(d):
            return False, f"{_HC} run without it"
        return rule(d, t, p)

    return wrapped


def _reaching(rows: int, what: str) -> RuleFn:
    """A rule on while the server decodes and some decode step reaches ``rows`` rows."""

    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if not d.decodes():
            return False, _NO_DECODE
        if d.reaches(rows):
            return True, f"{what} are reachable at max_batch_size {_count(d.max_batch_size)}"
        return False, f"max_batch_size {_count(d.max_batch_size)} never reaches {rows} decode rows"

    return rule


def _packed_prep_append(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    if d.tp == 8:
        # Served at TP8 for its A/B (12..16-request steps: 1728..2304 one-warp programs per
        # call); its geometries were swept at TP4.
        return False, "at tep8 on only by its knob (geometries swept at tep4)"
    rows = int(p["MIN_ROWS"])
    return _reaching(rows, f"prep-append calls of {rows} or more rows")(d, t, p)


def _windowed_commit_parallel(
    d: Deployment, t: Resolving, p: Mapping[str, object]
) -> Tuple[bool, str]:
    requests = int(p["MIN_REQUESTS"])
    return _reaching(
        d.rows_per_request * requests, f"context commits of {requests} or more requests"
    )(d, t, p)


def _wide_steps(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    return _reaching(
        WIDE_MIN_ROWS, f"decode steps of {WIDE_MIN_ROWS}..{4 * ROWS_PER_REQUEST} rows"
    )(d, t, p)


def _at_most(key: str, requires_decode: bool = True) -> RuleFn:
    """A rule on while ``max_batch_size`` is known and at most the parameter ``key``."""

    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        most = int(p[key])
        if requires_decode and not d.decodes():
            return False, _NO_DECODE
        if d.max_batch_size is None:
            return False, "max_batch_size is unknown"
        if d.max_batch_size <= most:
            return True, f"max_batch_size {d.max_batch_size} is at most {most}"
        return False, f"max_batch_size {d.max_batch_size} is above {most}"

    return rule


def _lossy(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    return False, "lossy: on only by its knob"


def _flagged(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    return False, "flagged: on only by its knob, once its open safety check has passed"


def _always(why: str) -> RuleFn:
    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        return True, why

    return rule


def _never(why: str) -> RuleFn:
    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        return False, why

    return rule


def _graph_start_prefetch(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    if _hc_selection(d):
        return True, f"graph starts of split-K row counts on {_HC}"
    return False, (
        "on sm107 a captured decode graph starts without a fork, and layer 0's QKV weight is "
        "not prefetched"
    )


def _coop_fold(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    if not d.decodes():
        return False, _NO_DECODE
    return True, f"head calls of at most {p['MAX_ROWS']} rows"


def _hot_replicas(requires_decode: bool) -> RuleFn:
    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if requires_decode and not d.decodes():
            return False, _NO_DECODE
        if t.on("half_window_experts"):
            return False, "yields to half_window_experts, which places each step's experts"
        return True, "33..512-row decode steps"

    return rule


def _swa_block_long_splits(
    d: Deployment, t: Resolving, p: Mapping[str, object]
) -> Tuple[bool, str]:
    rows = int(p["MIN_ROWS"])
    return _reaching(rows, f"sliding-window decode steps of {rows} or more rows")(d, t, p)


def _attn_gather_excluded(d: Deployment, t: Resolving) -> Optional[str]:
    if "qkv" not in t.param("context_fp8_dense", "PROJECTIONS"):
        return "its gathered rows feed the QKV projection, which context_fp8_dense__PROJECTIONS leaves out"
    return None


def _router_gate(role: Role, scope: str) -> Gate:
    def holds(d: Deployment) -> bool:
        return (
            d.role is role
            and d.tp == 4
            and d.max_batch_size is not None
            and 2 <= d.max_batch_size <= 4
        )

    return Gate(holds, scope)


#: The most requests the Vera Rubin low-latency levers (origin ``LL``, sm107 compositions)
#: were measured on: TEP4 and TEP8 decode of one to eight requests.
LL_MAX_BATCH = 8

#: The most requests the low-latency levers serve by default on a TEP8 generation server (the
#: ``gen`` role hint): the 'wall' set (every LL lever at MAX_BATCH 16, the 65..128-row bands of
#: ``_wide128``) passed its accuracy gate on sm107 TEP8 at max_batch_size 16 (GSM8K-long
#: conc 16, -0.08 pt; natural AL ratio 0.9991) and its ruler (bs10 / 12 / 16 -1.0 / -1.3 /
#: -1.7 ms against MAX_BATCH 16 alone).
LL_MAX_BATCH_TEP8_GEN = 16

#: The fewest rows ``tep8_router_push`` takes by default on a TEP8 generation server (``rp9``:
#: the kernel floor; ruler and accuracy gate passed at max_batch_size 8, ruler at 16).
TEP8_GEN_ROUTER_PUSH_MIN_ROWS = 9


#: The fewest requests of a TEP8 generation server's steps the one-cluster SWA block takes by
#: default (CL8 on 2..4 requests: GSM8K-long -0.15 pt, natural AL 0.9965, W4 batch-invariance bits
#: PASS; bs4 -0.51 ms); every other deployment keeps the 5..8-request ONE2 range.
TEP8_GEN_CL8_MIN_REQUESTS = 2


#: The most requests of a TEP8 generation server's 9..16-request steps the one-cluster SWA block
#: takes by default (the X400 composite: CL8 to 11 requests, one wave; W4 bits PASS, composite GSM8K
#: gate PASS; bs10 -0.38 ms). 12..16 requests keep the split kernels (CL8 would need two waves there).
TEP8_GEN_CL8_MAX_REQUESTS = 11


def _tep8_gen(d: Deployment) -> bool:
    """Whether ``d`` is a TEP8 generation server (the ``gen`` role hint)."""
    return d.tp == 8 and d.role is Role.GEN


def _tep8_gen_wide(d: Deployment) -> bool:
    """Whether ``d`` is a TEP8 generation server whose decode steps reach 65..128 rows:
    ``max_batch_size`` above ``LL_MAX_BATCH`` and at most ``LL_MAX_BATCH_TEP8_GEN``.

    Its low-latency levers default to MAX_BATCH 16 and their 65..128-row bands. Every other
    deployment keeps the 1..8-request defaults, so a generation server of ``max_batch_size``
    up to 8 resolves as before (the 128-row parameters also set kernel layouts there).
    """
    return (
        _tep8_gen(d)
        and d.max_batch_size is not None
        and LL_MAX_BATCH < d.max_batch_size <= LL_MAX_BATCH_TEP8_GEN
    )


def _ll_max_batch(d: Deployment) -> int:
    """The default MAX_BATCH of a low-latency lever on ``d`` (``_tep8_gen_wide``)."""
    return LL_MAX_BATCH_TEP8_GEN if _tep8_gen_wide(d) else LL_MAX_BATCH


def _wide128(default: int) -> Callable[[Deployment], int]:
    """A 65..128-row band parameter's default: 128 on ``_tep8_gen_wide`` deployments, else
    ``default``."""
    return lambda d: 128 if _tep8_gen_wide(d) else default


#: The gen lane's stacked winners (int5): on by default where they were gated and measured, a TEP8
#: ``gen`` server of 9..16 requests (``_tep8_gen_wide``; stack round 1 ABBA at mbs16, LEDGER G92:
#: -0.20 / -0.13 / -0.28 ms at bs8 / 12 / 16). The set the E2E X400 gen role ran as knobs
#: (lanes/gen/vrg/gint/int4w_e2e_gen.env): the verify sampler's multi-CTA top-p and distributed
#: sampling, the pool norm prefetch, incremental block offsets at 9..16 requests (and, over a paged
#: drafter context, its pool groups), the QKV 2-CTA split-K to 128 rows (numeric, GSM8K PASS), the
#: pool / unpool 64x64 tiles from 97 rows, and the drafter's deep split-K stages, PDL conv entry to
#: 64 rows and two-shot fused hop on 25..64 rows. Everything else of the gen lane's tree (window
#: tiles, paged commits, SiLU fold, L061; since int7 the drafter QK fold) stays off. Every value
#: stays a knob away.
def _tep8_gen_wide_winner(value: object, default: object) -> Callable[[Deployment], object]:
    return lambda d: value if _tep8_gen_wide(d) else default


def _tep8_gen_router_push_min_rows(d: Deployment) -> int:
    return TEP8_GEN_ROUTER_PUSH_MIN_ROWS if _tep8_gen(d) else 33


def _tep8_gen_cl8_min_requests(d: Deployment) -> int:
    return TEP8_GEN_CL8_MIN_REQUESTS if _tep8_gen(d) else 5


def _tep8_gen_cl8_max_requests(d: Deployment) -> int:
    return TEP8_GEN_CL8_MAX_REQUESTS if _tep8_gen_wide(d) else 8


#: On wherever the lever can run (no structural, dependency or conflict rule excludes it).
ALWAYS = _always("on by default wherever it can run")
#: Opt-in: on only by its knob.
NEVER = _never("opt-in: on only by its knob")


def unless_on(others: Tuple[str, ...], rule: Optional[RuleFn] = None) -> RuleFn:
    """``rule`` (``ALWAYS`` when not given), off while one of ``others`` is on (they serve the
    same steps, or one may not be on with it); ``others`` go in the row's ``after``."""
    return _yielding(others, rule if rule is not None else ALWAYS)


def gated(rule: RuleFn) -> RuleFn:
    """``rule``, and one of the lever's own accuracy gates (``Lever.gates``) holds."""

    def wrapped(d: Deployment, t: "Resolving", p: Mapping[str, object]) -> Tuple[bool, str]:
        on, why = rule(d, t, p)
        if not on:
            return on, why
        gates = LEVERS[t.pending[0]].gates
        held = [gate for gate in gates if gate.holds(d)]
        if not held:
            scopes = "; ".join(gate.scope for gate in gates)
            return False, f"{why}, but no accuracy gate holds ({scopes})"
        return True, f"{why}; gate: {held[0].scope}"

    return wrapped


def in_band(band: Band, rule: RuleFn, stock: bool = True) -> RuleFn:
    """``rule`` on deployments of ``band`` (``band_of``); elsewhere the lever's ``stock`` rule,
    or off for a lever without one (or with ``stock=False``)."""

    def wrapped(d: Deployment, t: "Resolving", p: Mapping[str, object]) -> Tuple[bool, str]:
        here = band_of(d)
        if here is band:
            return rule(d, t, p)
        stock = LEVERS[t.pending[0]].stock if use_stock else None
        if stock is None:
            return False, f"a {band.value} lever, off on a {here.value} deployment"
        on, why = stock(d, t, p)
        return on, f"a {band.value} lever on a {here.value} deployment, its upstream rule: {why}"

    use_stock = stock
    return wrapped


def either(first: RuleFn, second: RuleFn) -> RuleFn:
    """On where ``first`` is; elsewhere ``second``."""

    def wrapped(d: Deployment, t: "Resolving", p: Mapping[str, object]) -> Tuple[bool, str]:
        on, why = first(d, t, p)
        return (on, why) if on else second(d, t, p)

    return wrapped


def unless_rule(others: Mapping[str, RuleFn], rule: Optional[RuleFn] = None) -> RuleFn:
    """``rule`` (``ALWAYS`` when not given), off where one of ``others`` would be on by the rule
    given for it (evaluated as that lever: its parameters and gates), since this lever would take
    those steps from it."""
    rule = rule if rule is not None else ALWAYS

    def wrapped(d: Deployment, t: "Resolving", p: Mapping[str, object]) -> Tuple[bool, str]:
        own = t.pending
        for name, other_rule in others.items():
            if name not in t._served:
                continue
            decided = t._decisions.get(name)
            params = (
                dict(decided.params)
                if decided is not None
                else {param.key: param.default_of(d) for param in LEVERS[name].params}
            )
            t.pending = (name, params)
            try:
                held = other_rule(d, t, params)[0]
            finally:
                t.pending = own
            if held:
                return False, f"leaves these steps to {name}, on here by its rule"
        return rule(d, t, p)

    return wrapped


def _ll_params() -> Tuple[Param, ...]:
    return (Param("MAX_BATCH", LL_MAX_BATCH, 1, 64, default_for=_ll_max_batch),)


def _ll_gate() -> Gate:
    """Where the numeric low-latency levers passed their accuracy gates."""

    def holds(d: Deployment) -> bool:
        if d.role not in (Role.GEN, Role.AGG) or d.max_batch_size is None:
            return False
        if d.tp == 8:
            return d.max_batch_size <= LL_MAX_BATCH_TEP8_GEN
        return d.tp == 4 and d.max_batch_size <= LL_MAX_BATCH

    return Gate(
        holds,
        f"sm107 tep4 servers of max_batch_size up to {LL_MAX_BATCH} and tep8 servers up to "
        f"{LL_MAX_BATCH_TEP8_GEN}: greedy drift, GSM8K and SPEED-Bench acceptance (tep8 9..16: "
        "the 'wall' set, GSM8K-long conc 16 and natural AL)",
    )


def _yielding(yields_to: Tuple[str, ...], rule: RuleFn) -> RuleFn:
    """``rule``, off while one of ``yields_to`` is on.

    ``yields_to`` are levers of the aggregated sm107 band's selection that serve the same
    steps, or feed what this lever changes (a default-band lever). Under ``auto`` the two
    are never on in the same band (``band_of``); where one of ``yields_to`` is turned on by
    its knob outside its band, this lever steps aside for it instead of joining it.
    """

    def yielding(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        for other in yields_to:
            if t.on(other):
                return False, f"yields to {other}, which serves these steps where it is on"
        return rule(d, t, p)

    return yielding


def _context_workers(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    """On context servers; on aggregated ones only while ``max_batch_size`` is at most MAX_BATCH."""
    if d.role is Role.GEN:
        return False, "the server runs no context step"
    if d.role is Role.AGG:
        return _at_most("MAX_BATCH", requires_decode=False)(d, t, p)
    return True, "large context steps of a context server"


def _not_on_tep8(why_off: str, why_on: str) -> RuleFn:
    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if d.tp == 8:
            return False, why_off
        return True, why_on

    return rule


#: The TP4 paths the sm107 TEP8 composition serves too (``modeling._SM107_TEP8_PATHS``):
#: on wherever served under ``auto``; under ``baseline`` the upstream sm107 TEP8 composition,
#: which did not serve them, leaves them out.
SM107_TEP8_TEP4_PATHS = (
    "multi_request_prep_append",
    "wide_swa_block",
    "wide_full_attention",
    "multi_request_split_kv",
    "multi_request_post_attention_hop",
    "wide_drafter_hops",
    "router_gemm",
    "wide_dense_gemms",
    "projection_publish",
)


#: Parameters of the ``SM107_TEP8_TEP4_PATHS`` rules (each default the band served before).
_TEP4_PATH_PARAMS: Mapping[str, Tuple[Param, ...]] = {
    # The most rows of a TP8 drafter forward on the wide fused hop (``drafter_hops.install_wide``):
    # 64, the device's served wide rows (``drafter_hop.WIDE_SERVED_ROWS_BY_SM``: on Rubin the
    # one-shot hop is the faster one at 64 rows and the slower one at 128); 65..TEP8_MAX_ROWS rows
    # (9..16 requests) then take the one-shot hop (all-reduce + block convolution + add + RMSNorm in
    # one launch) instead of the built-in two-shot + convolution + fused add-RMSNorm (three
    # launches), on a wide workspace of TEP8_MAX_ROWS rows (5 areas x 8 ranks x rows x 16 KiB).
    "wide_drafter_hops": (
        Param("TEP8_MAX_ROWS", 64, 64, 128),
        # TP8 drafter forwards of at least this many rows on the fused hop's two-shot by column slice
        # (the same statements as the one-shot hop: bit for bit), 9..128; 0 (default): none.
        Param(
            "TEP8_TWO_SHOT_MIN_ROWS",
            0,
            0,
            128,
            values=(0, *range(9, 129)),
            default_for=_tep8_gen_wide_winner(25, 0),
        ),
        # The most rows of that two-shot (128: the whole band). At 65..128 rows it replaces the built-in
        # two-shot + convolution + add-RMSNorm, which is faster there (vrg-zdf): 64 keeps those.
        Param("TEP8_TWO_SHOT_MAX_ROWS", 128, 9, 128, default_for=_tep8_gen_wide_winner(64, 128)),
    ),
}


def _drafter_kv_glue(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    if t.on("drafter_kv_side_chain"):
        return False, "yields to drafter_kv_side_chain, which fuses the same store"
    if t.on("context_kv_late_join"):
        return False, "yields to context_kv_late_join, which serves these steps where it is on"
    return _at_most("MAX_BATCH")(d, t, p)


def _vocab_parallel_head(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    if t.on("distributed_head_sampler"):
        # Both sample the 8-row verify head; the vocabulary-parallel head would take its steps.
        return False, "yields to distributed_head_sampler, which samples these verify heads"
    # 151 MiB of fabric buffers per TP4 rank for steps of up to 1024 rows; an engine that
    # admits more requests (a context worker among them) samples on the built-in head.
    return _at_most("MAX_BATCH")(d, t, p)


#: The smallest ``max_batch_size`` the HC levers were measured at on aggregated sm107 TEP4
#: servers; below it they resolve off.
HC_LEVERS_MIN_BATCH = 4

_AGG_TEP4 = f"aggregated sm107 tep4 servers of max_batch_size {HC_LEVERS_MIN_BATCH} and up"


def _agg_sm107_tep4(d: Deployment) -> bool:
    """Whether ``d`` is an aggregated sm107 TEP4 server of a known ``max_batch_size`` of at
    least ``HC_LEVERS_MIN_BATCH`` (where the HC levers were measured)."""
    return (
        d.tp == 4
        and d.role is Role.AGG
        and d.max_batch_size is not None
        and d.max_batch_size >= HC_LEVERS_MIN_BATCH
    )


def _on_agg_tep4(what: str, min_rows: int = 0) -> RuleFn:
    """A rule on for ``what`` on aggregated sm107 TEP4 servers (``_agg_sm107_tep4``) whose
    decode steps can reach ``min_rows`` rows (``Deployment.reaches``), off elsewhere."""

    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if not _agg_sm107_tep4(d):
            if d.tp == 4 and d.role is Role.AGG:
                return False, (
                    f"measured at max_batch_size {HC_LEVERS_MIN_BATCH} and up; max_batch_size "
                    f"is {'unknown' if d.max_batch_size is None else d.max_batch_size}"
                )
            return False, f"measured on {_AGG_TEP4} only; this is a {d.role.value} server"
        if min_rows and not d.reaches(min_rows):
            return False, f"no decode step reaches {min_rows} rows"
        return True, f"{what} on {_AGG_TEP4}"

    return rule


def _ctx_or_agg_tep4(what: str) -> RuleFn:
    """A rule on for ``what`` on sm107 TEP4 context servers (the ``ctx`` role hint) and on
    aggregated sm107 TEP4 servers (``_on_agg_tep4``), off elsewhere.

    For the levers of a context step alone: a context server runs the same context steps an
    aggregated server does, so its band does not decide them (they are in no band's selection).
    A generation or unhinted disaggregated server runs none of them, or cannot tell, and stays off.
    """
    agg = _on_agg_tep4(what)

    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if d.tp == 4 and d.role is Role.CTX:
            return True, f"{what} on sm107 tep4 context servers"
        return agg(d, t, p)

    return rule


#: The accuracy gate of the lossless-numeric HC levers.
_AGG_TEP4_GATE = Gate(_agg_sm107_tep4, _AGG_TEP4)

#: The ``max_batch_size`` range of the aggregated sm107 TEP4 servers whose 2..4-request
#: decode steps (16..32 rows) the levers of ``_two_to_four_request_steps`` serve: each of
#: these levers writes the bytes of the path it replaces, chosen by the step's shape alone.
WIDE_AGG_BATCH = (2, 8)


def _two_to_four_request_steps(what: str) -> RuleFn:
    """A rule for a bitwise lever of the 2..4-request decode steps: on for aggregated sm107
    TEP4 servers of ``max_batch_size`` in ``WIDE_AGG_BATCH``, off elsewhere (TEP8, one-request
    servers, the high-concurrency selection, and the other roles, whose band does not serve
    these levers)."""
    low, high = WIDE_AGG_BATCH

    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if d.tp != 4:
            return False, f"for sm107 tep4 servers only; this is sm{d.sm} tep{d.tp}"
        if d.role is not Role.AGG:
            return False, (
                f"for aggregated servers only; this is a {d.role.value} server ({d.role_source})"
            )
        if d.max_batch_size is None:
            return False, "max_batch_size is unknown"
        if not low <= d.max_batch_size <= high:
            return False, (
                f"for aggregated servers of max_batch_size {low}..{high}; max_batch_size is "
                f"{d.max_batch_size}"
            )
        return True, f"{what} on aggregated servers of max_batch_size {low}..{high}"

    return rule


def _full_attention_fold_multi(
    d: Deployment, t: Resolving, p: Mapping[str, object]
) -> Tuple[bool, str]:
    """In the aggregated sm107 band, the fold of the 2..4-request steps of
    ``wide_full_attention`` (off with it); in the default band, on while ``max_batch_size``
    is at most MAX_BATCH."""
    if band_of(d) is Band.AGGREGATED_SM107:
        if not t.on("wide_full_attention"):
            return False, "aggregated sm107 servers fold it into wide_full_attention, which is off"
        return _two_to_four_request_steps("the multi-sequence full-attention prep fold")(d, t, p)
    return _at_most("MAX_BATCH")(d, t, p)


_CTX_TEP4 = "disaggregated context servers at sm107 tep4"


def _ctx_sm107_tep4(d: Deployment) -> bool:
    """Whether ``d`` is a disaggregated context server at sm107 TEP4."""
    return d.sm == 107 and d.tp == 4 and d.role is Role.CTX


def _on_ctx_sm107_tep4(what: str) -> RuleFn:
    """A rule on for ``what`` on disaggregated sm107 TEP4 context servers, off elsewhere."""

    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if _ctx_sm107_tep4(d):
            return True, f"{what} on {_CTX_TEP4}"
        return False, f"measured on {_CTX_TEP4} only; this server's role is {d.role.value}"

    return rule


#: The accuracy gate of ``fmk_pure_context_fmha``: its class, whose gate is still to run, so
#: it holds nowhere yet and the lever is on only by its knob.
_CTX_TEP4_GATE_PENDING = Gate(lambda d: False, f"{_CTX_TEP4} (gate pending)")
#: The accuracy gate of ``context_last_layer_tail`` (acceptance length and accuracy on the TEP4
#: context server):
#: still to run, so the lever is on only by its knob.
_CTX_TEP4_TAIL_GATE_PENDING = Gate(lambda d: False, f"{_CTX_TEP4} (gate pending)")


_GEN_TEP4 = "disaggregated generation servers at sm107 tep4"

#: The requests the context-K/V fork takes without ``context_kv_fork_wide``
#: (``ops/spec/context_kv_commit.MAX_REQUESTS``).
FORK_MAX_REQUESTS = 64
#: The requests the context-K/V fork takes with ``context_kv_fork_wide``: the parallel
#: windowed commit's bound (``ops/spec/context_kv_commit_windowed.PARALLEL_MAX_REQUESTS``).
FORK_WIDE_MAX_REQUESTS = 128


#: The drafter context arenas of a wide generation server (``_wide_generation_refusal``).
WIDE_GENERATION_ARENAS = (Arena.WINDOWED, Arena.PAGED)


def _wide_generation_refusal(d: Deployment) -> Optional[str]:
    """Why ``d`` is not a wide generation server, or None when it is one.

    A wide generation server is a disaggregated sm107 TEP4 generation server with a windowed
    or a paged drafter context arena (``WIDE_GENERATION_ARENAS``) whose decode steps can
    exceed ``FORK_MAX_REQUESTS`` requests. Its decode steps are served by
    ``multi_request_ffn_join`` and the raised row cap of ``router_gemm`` and
    ``finalize_two_shot`` (``row_cap``), and over a windowed arena also by
    ``context_kv_fork_wide`` (``_wide_fork_refusal``); every other deployment runs those
    steps as without them.

    None of the three reads the drafter context arena: the router GEMM and the two-shot
    finalize take the target's decode rows, and the join stores the DFlash capture into the
    speculative metadata's capture buffer (``capture_fold.destination``), which the worker
    reads after the target forward whatever arena it then writes the drafter's context to.
    """
    if not (d.sm == 107 and d.tp == 4 and d.role is Role.GEN):
        return f"measured on {_GEN_TEP4} only; this server's role is {d.role.value}"
    if d.arena not in WIDE_GENERATION_ARENAS:
        return f"for a windowed or paged drafter context arena; this server's is {d.arena.value}"
    if d.max_batch_size is None or d.max_batch_size <= FORK_MAX_REQUESTS:
        return (
            f"max_batch_size {_count(d.max_batch_size)} never exceeds {FORK_MAX_REQUESTS} requests"
        )
    return None


def _wide_fork_refusal(d: Deployment) -> Optional[str]:
    """Why the context-K/V fork does not take ``d``'s decode steps of more than
    ``FORK_MAX_REQUESTS`` requests (``context_kv_fork_wide``), or None.

    A wide generation server (``_wide_generation_refusal``) with the windowed drafter
    context arena: the fork of those steps commits through the parallel windowed commit.
    """
    refusal = _wide_generation_refusal(d)
    if refusal is not None:
        return refusal
    if d.arena is not Arena.WINDOWED:
        return (
            "its commit is the parallel windowed commit, which needs the windowed drafter "
            f"context arena; this server's is {d.arena.value}"
        )
    return None


def _context_kv_fork_wide(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    """On for wide generation servers over a windowed arena (``_wide_fork_refusal``), off
    elsewhere."""
    refusal = _wide_fork_refusal(d)
    if refusal is not None:
        return False, refusal
    most = min(d.max_batch_size, FORK_WIDE_MAX_REQUESTS)
    return True, (
        f"decode steps of {FORK_MAX_REQUESTS + 1}..{most} requests on the context-K/V fork "
        f"on {_GEN_TEP4}"
    )


def _multi_request_ffn_join(
    d: Deployment, t: Resolving, p: Mapping[str, object]
) -> Tuple[bool, str]:
    """On for wide generation servers (``_wide_generation_refusal``), off elsewhere."""
    refusal = _wide_generation_refusal(d)
    if refusal is not None:
        return False, refusal
    return True, f"the capture layers' post-FFN joins of 33..768-row calls on {_GEN_TEP4}"


#: The most rows of a decode step ``router_gemm`` and ``finalize_two_shot`` arm: ``ROW_CAP``,
#: or ``WIDE_ROW_CAP`` (96 requests at K = 7) on a wide generation server (``row_cap``).
ROW_CAP = 512
WIDE_ROW_CAP = 768


def row_cap(d: Deployment, default: str = AUTO) -> int:
    """The row cap of ``router_gemm`` and ``finalize_two_shot`` on ``d``.

    ``WIDE_ROW_CAP`` on a wide generation server (``_wide_generation_refusal``) under the
    ``auto`` default, ``ROW_CAP`` everywhere else (and under ``baseline``). The rows above
    ``ROW_CAP`` leave the built-in FP32 router and the unfused expert combine for these
    paths, which is lossless-numeric against them.
    """
    if default == AUTO and _wide_generation_refusal(d) is None:
        return WIDE_ROW_CAP
    return ROW_CAP


def _decode_qkv_excluded(d: Deployment, t: Resolving) -> Optional[str]:
    if "qkv" not in t.param("context_fp8_dense", "PROJECTIONS"):
        return "it reads context_fp8_dense's QKV copies, which context_fp8_dense__PROJECTIONS leaves out"
    return None


#: Why ``auto`` leaves the context-step reduce-scatter overlaps off
#: (``kernels/context_sequence_parallel/overlap``): their knobs turn them on.
_CTX_OVERLAP_UNMEASURED = "not yet measured on sm107: on only by its knob"


def _ctx_tep4(d: Deployment) -> bool:
    """Whether ``d`` is an sm107 TEP4 context server (the ``ctx`` role hint)."""
    return d.tp == 4 and d.role is Role.CTX


def _on_ctx_tep4(what: str, elsewhere: str) -> RuleFn:
    """A rule on for ``what`` on sm107 TEP4 context servers (``_ctx_tep4``), off elsewhere
    for ``elsewhere`` (a knob still turns it on there)."""

    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if _ctx_tep4(d):
            return True, f"{what} on sm107 tep4 context servers"
        return False, elsewhere

    return rule


def _fmk_context_fmha(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    """On sm107 TEP4 context servers (every pure context step, PURE_CONTEXT 1) and on
    aggregated sm107 TEP4 servers (the context rows of mixed steps), off elsewhere."""
    if _ctx_tep4(d):
        return True, "the pure context steps of sm107 tep4 context servers"
    return _on_agg_tep4("the context rows of mixed steps")(d, t, p)


#: The accuracy gate ``fmk_context_fmha`` passed on pure context steps (PURE_CONTEXT 1): every
#: prefill of a CONC=1 run is a pure context step.
_CTX_TEP4_FMK_GATE = Gate(
    _ctx_tep4,
    "sm107 tep4 context servers, pure context steps (PURE_CONTEXT 1): GSM8K-long CONC=1 -0.08 pt, "
    "natural AL ratio 0.9974",
)

#: The parts ``context_sp_extra_parts`` adds on a context server: ``shard_routing`` too (the
#: context-step reduce-scatter overlap set, 'ovts'), and no ``ce_gather`` (int4): its
#: copy-engine gathers run during every large context step, exactly while that server's KV
#: blocks go to the generation servers, and they share the copy engines with the cache
#: transceiver's NIXL writes (rhea D1 H4: with it off the slow KV-transfer class drops to
#: IBPL's, 9.4 % of requests vs 28-34 %; the GB300 fix of the same contention). The gather of
#: the attention input then runs as an all-gather, as on an aggregated server; bitwise.
_CTX_TEP4_EXTRA_PARTS = ("shard_routing", "router_input")


def _ctx_overlap_excluded(
    projection: str, needs_router_input: bool = False
) -> Callable[[Deployment, Resolving], Optional[str]]:
    """The structural exclusions of a context-step reduce-scatter overlap that runs the
    ``projection`` GEMM on row chunks: ``context_fp8_dense`` on that projection (its MXFP8 path
    is chosen by a call's rows, and a chunk has fewer than the whole call), and, with
    ``needs_router_input``, a step without the sequence-parallel router-input join (the tail
    pipeline joins through it)."""

    def excluded(d: Deployment, t: Resolving) -> Optional[str]:
        if t.on("context_fp8_dense") and projection in t.param("context_fp8_dense", "PROJECTIONS"):
            return (
                f"context_fp8_dense picks the {projection} GEMM's MXFP8 path by a call's rows, and "
                "the chunked calls have fewer rows than the whole call"
            )
        if needs_router_input and not (
            t.on("context_sp_router_input")
            or (
                t.on("context_sp_extra_parts")
                and "router_input" in t.param("context_sp_extra_parts", "PARTS")
            )
        ):
            return (
                "it joins through the sequence-parallel router-input join, which neither "
                "context_sp_router_input nor context_sp_extra_parts__PARTS turns on"
            )
        return None

    return excluded


def _ctx_overlap_params() -> Tuple[Param, ...]:
    """A chunked reduce-scatter overlap's chunks, its row floor and its stream priority."""
    return (
        # Row chunks per step (1: the unchunked call on the side stream, a control).
        Param("CHUNKS", 2, 1, 4),
        # The fewest rows of a sequence-parallel step it takes.
        Param("MIN_ROWS", 8192, 1024, 1 << 20),
        # 1: the side stream runs at the device's highest priority; 0: the default priority.
        Param("HIGH_PRIORITY", 1, 0, 1),
    )


def _row_owner_excluded(d: Deployment, t: Resolving) -> Optional[str]:
    """Why ``row_owner_moe`` cannot be on ``d``: it builds on the wide generation server's
    row cap of ``router_gemm`` and ``finalize_two_shot`` (``row_cap``), whose 520..768-row
    calls it reproduces per owner row."""
    return _wide_generation_refusal(d)


#: The deployments the single-request decode schedule of sm107 TEP8 was measured on.
_SINGLE_REQUEST = (
    "aggregated sm107 tep8 servers of max_batch_size 1 over a dense or paged drafter context arena"
)


def _single_request_tep8(d: Deployment) -> bool:
    """Whether ``d`` is a deployment the single-request decode schedule was measured on."""
    return (
        d.tp == 8
        and d.role is Role.AGG
        and d.max_batch_size == 1
        and d.arena in (Arena.DENSE, Arena.PAGED)
    )


def _on_single_request(what: str) -> RuleFn:
    """A rule on for ``what`` on ``_single_request_tep8`` deployments, off elsewhere."""

    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if _single_request_tep8(d):
            return True, f"{what} on {_SINGLE_REQUEST}"
        if not d.decodes():
            return False, _NO_DECODE
        if d.role is not Role.AGG:
            return False, f"measured on {_SINGLE_REQUEST} only; this is a {d.role.value} server"
        if d.max_batch_size != 1:
            return False, (
                f"measured on {_SINGLE_REQUEST} only; max_batch_size is {_count(d.max_batch_size)}"
            )
        return False, f"measured on {_SINGLE_REQUEST} only; this is a {d.arena.value} arena"

    return rule


def _on_max_batch_one(what: str) -> RuleFn:
    """On for ``what`` on a server of max_batch_size 1 (any role), off elsewhere."""

    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if d.max_batch_size == 1:
            return True, f"{what} on a server of max_batch_size 1"
        return False, f"max_batch_size is {_count(d.max_batch_size)}, not 1"

    return rule


#: The accuracy gate of the single-request lossless-numeric levers: the megakernel's
#: numerics-budget levers and the distributed head sampler.
_SINGLE_REQUEST_GATE = Gate(_single_request_tep8, _SINGLE_REQUEST)

#: The megakernel's numerics-budget levers: the served build carries all three, and no build
#: of it without one exists, so ``megakernel`` requires them (turning one off runs the stock
#: layers, which is how the numerics budget is opted out of).
MEGAKERNEL_NUMERICS = (
    "megakernel_one_exchange_boundaries",
    "megakernel_two_term_pool",
    "megakernel_router_by_linearity",
)


def _megakernel_variant_excluded(d: Deployment, t: Resolving) -> Optional[str]:
    return "no build of the served megakernel carries it"


#: Every lever whose rule is not the default; every other served lever is on under ``auto``
#: and ``baseline``.
#: The deployment rules of the levers others step aside for: where one holds, that lever is on and
#: the levers yielding to it (``unless_rule``) are off; elsewhere those run, and it fills in where
#: they are off (``either(<rule>, unless_on(...))``).
_STEP_LOOP_INLINE_COMMITS = in_band(
    Band.AGGREGATED_SM107, _on_single_request("the one-stream step loop's commits"), stock=False
)
_LAYER0_QKV_TABLE = in_band(
    Band.AGGREGATED_SM107, _on_single_request("8-row decode steps' layer-0 QKV rows"), stock=False
)
_WIDE_QKV_PREFETCH = in_band(
    Band.AGGREGATED_SM107,
    _two_to_four_request_steps("the 9..32-row next-QKV prefetch after the experts"),
    stock=False,
)
_CONTEXT_KV_LATE_JOIN = in_band(
    Band.AGGREGATED_SM107, _on_agg_tep4("windowed block decodes after a forked commit"), stock=False
)
_CONTEXT_KV_FORK_LAST_LAYER = in_band(
    Band.AGGREGATED_SM107, _on_single_request("forked side-chain steps"), stock=False
)
_SWA_BLOCKS_PER_PROGRAM = in_band(
    Band.AGGREGATED_SM107,
    _two_to_four_request_steps("sliding-window split programs over 2..4 key blocks"),
    stock=False,
)
#: The low-latency rule of the disaggregated deployments' levers (``_at_most`` MAX_BATCH), as the
#: levers stepping aside read it; ``_LL_GATED`` with the lever's own accuracy gate.
_LL_DEFAULT_BAND = in_band(Band.DEFAULT, _at_most("MAX_BATCH"), stock=False)
_LL_GATED = in_band(Band.DEFAULT, gated(_at_most("MAX_BATCH")), stock=False)
_DRAFTER_KV_GLUE_RULE = in_band(
    Band.DEFAULT, _yielding(("drafter_kv_side_chain",), _at_most("MAX_BATCH")), stock=False
)

LEVERS: Mapping[str, Lever] = types.MappingProxyType(
    {
        "packed_prep_append": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=(Param("MIN_ROWS", 33, 8, 1024),),
        ),
        "windowed_commit_parallel": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=(
                Param("MIN_REQUESTS", 5, 1, 64, default_for=lambda d: 1 if _hc_selection(d) else 5),
            ),
        ),
        "context_fp8_dense": Lever(
            Kind.LOSSY,
            _lossy,
            params=(
                Param("MIN_ROWS", 1024, 1024, 32768),
                Param(
                    "PROJECTIONS",
                    ("qkv", "o_proj", "pool", "unpool"),
                    choices=("qkv", "o_proj", "pool", "unpool"),
                ),
            ),
        ),
        "context_fp8_attn_gather": Lever(
            Kind.LOSSY,
            _lossy,
            after=("context_fp8_dense",),
            excluded=_attn_gather_excluded,
        ),
        "wide_direct_transports": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "wide_router_per_row": Lever(
            Kind.NUMERIC,
            gated(_wide_steps),
            gates=(_router_gate(Role.GEN, "generation servers at tep4 with max_batch_size 2..4"),),
        ),
        "wide_oproj_publish": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "wide_finalize_transport": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "lm_head_inbox": Lever(
            Kind.COMPOSITION,
            _never("the sm107 compositions gather the 8-row head without the inbox"),
            stock=_always("8-row heads"),
        ),
        "graph_start_qkv_prefetch": Lever(
            Kind.COMPOSITION,
            _graph_start_prefetch,
            stock=_always("graph starts of split-K row counts"),
        ),
        "step_loop_on_execution_stream": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=(Param("MAX_BATCH", 4, 1, 64),),
        ),
        "incremental_block_offsets": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=(
                Param("MAX_BATCH", 8, 1, 64, default_for=_tep8_gen_wide_winner(16, 8)),
                # Alias only (kept for the round-3 / E2E envs): upstream #21789 stages a manager
                # with attention-op pool groups (a paged drafter context) whenever the lever is on,
                # so the value changes nothing. int5's gen-role default (1 on a TEP8 gen server of
                # 9..16 requests over a paged drafter context) is kept for the record.
                Param(
                    "POOL_GROUPS",
                    0,
                    0,
                    1,
                    default_for=lambda d: int(_tep8_gen_wide(d) and d.arena is Arena.PAGED),
                ),
            ),
        ),
        # w256: served at TEP8 too (modeling._SM107_TEP8_PATHS), but on there only by its knob: decode
        # steps of 129..1024 rows (17+ requests at K = 7) keep their BF16 head row and the split sampler
        # (the shipped 512-thread block bit for bit: the same tokens) instead of the built-in chain
        # (logits cat, FP32 copy, soft cap, fusedSampling). TEP4 as before.
        "wide_split_sampler": Lever(
            Kind.BITWISE,
            ALWAYS,
            stock=_not_on_tep8(
                "on at TEP8 only by its knob (w256: 129..1024-row steps on the split sampler)",
                "the split verify sampler on 129..1024-row decode steps",
            ),
            params=(
                # w256 (0: off): steps of 129..VP_MAX_ROWS rows on the vocabulary-parallel head next to
                # head_sampler_coop_fold's distributed sampler (no logits all-gather; the same tokens).
                Param("VP_MAX_ROWS", 0, 0, 1024),
                # w256 (0: off; order-only, gated): the > 128-row LM-head GEMM as calls of at most this
                # many rows (each row the bits of its <= 128-row call's shape).
                Param("HEAD_CHUNK_ROWS", 0, 0, 128),
                # w256 (0: off): steps of 129..256 rows on head_sampler_coop_fold's distributed sampler in
                # 128-row calls (each row the 1024-thread block's tokens, as at <= 128 rows; gated).
                Param("DIST_CHUNK", 0, 0, 1),
            ),
        ),
        "drafter_kv_side_chain": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "drafter_fc_early_fork": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "drafter_preamble_from_commit": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "head_sampler_coop_fold": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=(
                Param("MAX_ROWS", 16, 1, 128),
                # 1..4: the split sampler's top-p-only rows (calls of at most 128 rows) take the
                # multi-CTA top-p test (head_sampler.TOPP_ROUNDS: that many test rounds spread
                # over 8 CTAs per row, the rest in the row's CTA); the same tokens. 0: off.
                Param("TOPP_ROUNDS", 0, 0, 4, default_for=_tep8_gen_wide_winner(2, 0)),
                # 8..128: split_sampler decode steps of at least that many rows (to 128) skip the logits
                # all-gather: the distributed sampler (head_sampler.ds_sample) samples the ranks' own LM-head
                # shards, kernel B's tokens. 0: off. Not with vocab_parallel_head.
                Param("DIST_MIN_ROWS", 0, 0, 128, default_for=_tep8_gen_wide_winner(16, 0)),
                # 1: DIST_MIN_ROWS' sampler stages its shard as LL elements the peers pull with their flags (no
                # system fence / ready flag) and sends 24-byte states (ds_sample's fabric variant); the same
                # tokens. 0: the gated form.
                Param("DIST_FABRIC", 0, 0, 1),
                # 1: the verify fold after the sampler (ops.spec.dflash_verify_fold) runs one program per request
                # instead of one program looping over them; the same outputs. 0: off.
                Param("VERIFY_GRID", 0, 0, 1),
            ),
        ),
        "head_multimem_gather": Lever(
            Kind.FLAGGED,
            _flagged,
            requires=("step_loop_on_execution_stream",),
            conflicts=("lm_head_inbox",),
        ),
        "half_window_experts": Lever(
            Kind.COMPOSITION,
            _at_most("MAX_BATCH"),
            stock=_at_most("MAX_BATCH", requires_decode=False),
            params=(Param("MAX_BATCH", 8, 1, 64),),
        ),
        "hot_replicas": Lever(
            Kind.COMPOSITION,
            _hot_replicas(requires_decode=True),
            stock=_hot_replicas(requires_decode=False),
            after=("half_window_experts",),
            conflicts=("half_window_experts",),
        ),
        "prefill_cutedsl_experts": Lever(
            Kind.COMPOSITION,
            _always("expert calls of 4096..8192 rows"),
            stock=_always("expert calls of 4096..8192 rows"),
        ),
        "vocab_parallel_head": Lever(
            Kind.COMPOSITION,
            in_band(Band.DEFAULT, _vocab_parallel_head),
            stock=_never("the upstream sm107 compositions sample on the split sampler"),
            params=(
                # Not raised with the LL levers at 9..16 requests: the X400 composite samples a TEP8
                # generation server of max_batch_size 9..16 on the split sampler (the gated path;
                # bs13-16 -0.12..-0.18 ms, bs8 -0.15 ms).
                Param("MAX_BATCH", LL_MAX_BATCH, 1, 64),
                Param("MIN_ROWS", 8, 8, 16),
                # 1: a sampled row walks the ranks' vocabulary slices as one chain (the single-rank
                # sampler's draws, bit for bit; xiaow 58f1bf57d70). 0: independent per-rank partials
                # combined in rank order (the pcastonguay campaign's draws). Greedy rows: the same.
                Param("RANK_CHAIN", 1, 0, 1),
            ),
            after=("distributed_head_sampler",),
            conflicts=("distributed_head_sampler",),
        ),
        **{
            name: Lever(
                Kind.COMPOSITION,
                in_band(Band.DEFAULT, _always("its rows, wherever served")),
                stock=_not_on_tep8(
                    "the upstream sm107 tep8 composition does not serve it",
                    "its rows, wherever served",
                ),
                params=_TEP4_PATH_PARAMS.get(name, ()),
            )
            for name in SM107_TEP8_TEP4_PATHS
        },
        # The Vera Rubin low-latency levers.
        "splitk_pool_split8": Lever(
            Kind.NUMERIC,
            in_band(Band.DEFAULT, gated(_at_most("MAX_BATCH"))),
            params=_ll_params(),
            gates=(_ll_gate(),),
        ),
        "splitk_frontier_tables": Lever(
            Kind.NUMERIC,
            in_band(Band.DEFAULT, gated(_yielding(("wide_qkv_prefetch",), _at_most("MAX_BATCH")))),
            after=("wide_qkv_prefetch",),
            params=_ll_params(),
            gates=(_ll_gate(),),
        ),
        "tep8_wide64_splitk": Lever(
            Kind.NUMERIC,
            in_band(Band.DEFAULT, gated(_at_most("MAX_BATCH"))),
            # The fewest rows of a TP8 QKV / o_proj call on the wide64 tactics (33: the 5..8-request
            # band; down to 9, the 2..4-request steps' calls leave the 9..32-row wide table).
            params=_ll_params()
            + (
                Param("MIN_ROWS", 33, 9, 33),
                # The most rows of a TP8 QKV call on the wide64 tactic (64: cuBLAS above); 65..MAX_ROWS
                # (9..16 requests) in one launch of it, 64-row token tiles on the grid (each row's
                # 1..64-row bits). o_proj keeps its 65..128-row paths (hop128 / cuBLAS).
                # (w256: to 256 with WIDE_TILE 2cta only: two 128-token pair tiles; PREP_FOLD to 128.)
                Param("MAX_ROWS", 64, 64, 256, default_for=_tep8_gen_wide_winner(128, 64)),
                # That launch's tile, output columns x rows: 64x64 (the served tactic as is, two token
                # tiles, 288 CTAs), 128x64 (144 CTAs) or 64x128 (one token tile, 144 CTAs: each weight
                # slab read once); stages lowered to fit; same split_k, same bits. 2cta: the 2-CTA
                # split-K kernel (ops/gemm/splitk_gemm_2cta, 128x128 pair tiles, in-kernel K-slice
                # reduction; split_k 4: the same bits).
                Param(
                    "WIDE_TILE",
                    "64x64",
                    choices=("64x64", "128x64", "64x128", "128x128", "2cta"),
                    one=True,
                    default_for=_tep8_gen_wide_winner("2cta", "64x64"),
                ),
                # That launch's split_k (0: the served tactic's, 4); 8 (reduce-scatter epilogue)
                # changes every row's bits (numeric).
                Param("WIDE_SPLIT_K", 0, 0, 8, values=(0, 4, 8)),
                # 1: with WIDE_TILE 2cta, a multi-request decode step's QKV call runs the packed decode
                # prep (q/k norm, RoPE, gate split, E4M3 query, FP8 K/V append, FMHA scales) in the
                # 2-CTA kernel's epilogue (ops/gemm/splitk_gemm_2cta_prep): no prep launch, no
                # GEMM -> prep edge; the same bytes. Needs multi_request_prep_append.
                Param("PREP_FOLD", 0, 0, 1),
                # (w256) The most rows PREP_FOLD folds: 128 (one token tile, default) .. 256 (both token
                # tiles of a 129..256-row call; the same bytes as the GEMM + the packed prep).
                Param("PREP_FOLD_MAX_ROWS", 128, 128, 256),
                # (w256, 0: off) The 2cta launch's split_k above 128 rows (2: one wave of 144 CTAs instead
                # of 288 on 212 SMs; order-only for those rows, gated).
                Param("WIDE2_SPLIT_K", 0, 0, 4, values=(0, 2, 4)),
            ),
            gates=(_ll_gate(),),
        ),
        "tep8_publish_splitk": Lever(
            Kind.NUMERIC,
            in_band(Band.DEFAULT, gated(_at_most("MAX_BATCH"))),
            params=_ll_params()
            + (
                # The most rows of a TP8 slice the split-K producer publishes; 65..MAX_ROWS in one
                # launch of the 33..64-row tactics (each row keeps its one-request bits).
                # (w256: to 256 in the same one launch.)
                Param("MAX_ROWS", 64, 64, 256, default_for=_wide128(64)),
                # The fewest rows of a decode step the publication takes (33: the cuBLAS census's
                # floor). Below 33 only on the split-K producer: the 2..4-request steps then publish
                # and join as 5..8-request steps do, instead of the 9..32-row gathers.
                Param("MIN_ROWS", 33, 9, 33),
                # The pool slice's tile in that one launch: 64x16 (the 33..64-row tactic: 256 CTAs
                # at 128 rows, two waves) or 64x32 (128 CTAs, the slice read half as often; the
                # census's certified unpool key); split_k 4 either way, so the same bits.
                Param(
                    "WIDE_POOL_TILE",
                    "64x16",
                    choices=("64x16", "64x32", "64x64"),
                    one=True,
                    default_for=_tep8_gen_wide_winner("64x64", "64x16"),
                ),
                # The unpool slice's tile there: 64x32 (the 33..64-row tactic: 256 CTAs at 128
                # rows) or 64x64 (128 CTAs; refused with an ERROR until the census certifies its key).
                Param(
                    "WIDE_UNPOOL_TILE",
                    "64x32",
                    choices=("64x32", "64x64"),
                    one=True,
                    default_for=_tep8_gen_wide_winner("64x64", "64x32"),
                ),
                # The fewest rows of a slice on those two tiles (65: the whole one-launch band; 97:
                # only where the served tiles' grid spills into a second wave of 212 SMs); fewer
                # rows keep the served tile, the same bits.
                Param("WIDE_TILE_MIN_ROWS", 65, 65, 256, default_for=_tep8_gen_wide_winner(97, 65)),
            ),
            gates=(_ll_gate(),),
        ),
        "tep8_drafter_wide_gemms": Lever(
            Kind.NUMERIC,
            in_band(Band.DEFAULT, gated(_at_most("MAX_BATCH"))),
            params=_ll_params()
            + (
                # The most rows of a TP8 drafter GEMM on the fixed split-K tactics (64: cuBLAS
                # above, nvjet + a splitKreduce launch on conv_coef / qkv, and gate_up at 96 rows);
                # 65..MAX_ROWS (9..16 requests) in ONE launch of the site's 64-row tactic, 64-row
                # token tiles on the grid (each row's one-request bits). Needs tep8_wide64_splitk.
                Param("MAX_ROWS", 64, 64, 128),
                # That launch's tile, output columns x rows: 64x128 (one token tile: as many CTAs
                # as the 64-row launch, one wave at every site, each weight slab read once), 64x64
                # (the 64-row tactic as is, same compiled kernel; two token tiles: the conv
                # coefficient GEMM's 256 CTAs are two waves on 212 SMs) or 128x64; stages lowered
                # to fit; same split_k, same bits.
                Param("WIDE_TILE", "64x128", choices=("64x64", "128x64", "64x128"), one=True),
                # The sites that take it (the rest keep cuBLAS above 64 rows): the ones whose
                # cuBLAS kernel at 72..128 rows is split-K + a reduce launch (conv_coef, qkv).
                # gate_up splits only at 96 rows (128x160 + reduce, 18 us; 14 us unsplit at 72 and
                # 128); o_proj / down: cuBLAS at 72..128 rows is already the 64-row tactic's time.
                Param(
                    "SITES",
                    ("conv_coef", "qkv"),
                    choices=("conv_coef", "qkv", "o_proj", "down", "gate_up"),
                ),
                # The drafter window attention's tiles per split on calls of at least
                # WINDOW_MIN_ROWS rows (1: the one-request geometry, 65 one-tile splits per
                # request: 585..1040 programs and 65 merged partials per row at 9..16 requests).
                # 2..4: fewer, longer splits (33 / 22 / 17). Numeric (split boundaries), drafter only.
                Param("WINDOW_TILES", 1, 1, 4),
                Param("WINDOW_MIN_ROWS", 65, 9, 128),
                # The drafter's next GEMM weights asked of the L2 from a side stream while its
                # non-HBM kernels run (kernels/drafter_l2_prefetch): 0 off; 1 o_proj + MLP conv
                # coefficients behind the window attention; 2 also gate_up + down behind o_proj's
                # hop; 3 also the next layer's conv coefficients + qkv behind the MLP's hop. Same
                # bits (a prefetch reads no value). On block decodes of PREFETCH_MIN_ROWS+ rows.
                Param("PREFETCH", 0, 0, 4),
                Param("PREFETCH_MIN_ROWS", 1, 1, 128),
                # The weights those forks ask for. Default without the MLP's gate_up / down (176 MB per
                # rank against Rubin's 126 MiB L2: asking for them evicts what the next GEMMs read; W10
                # prefetch chain on VR: all six +10..+22 us/step, these four -9..-11).
                Param(
                    "PREFETCH_WEIGHTS",
                    ("o_proj", "conv_mlp", "conv_attn", "qkv"),
                    choices=("o_proj", "conv_mlp", "gate_up", "down", "conv_attn", "qkv"),
                ),
                # 1: on block decodes without the fused drafter hop (65..128 rows), each hop's tail
                # (finish convolution + the next norm's residual add + RMSNorm) in one launch
                # (ops/spec/drafter_finish_norm: the fused hop's statements, bit for bit the two).
                Param("FINISH_NORM", 0, 0, 1),
                # 1: each layer's QK norm + RoPE folded into the window attention's split programs
                # (dr3_folds' QF, without the drafter_attn_pdl copy): one launch less per layer.
                # int7: no longer a 9..16-request default (vrg-zia / zca: QK fold costs step time in-model).
                Param("QK_FOLD", 0, 0, 1),
                # The PDL chain's conv entry trigger (releases the next fixed-tactic GEMM to stream its
                # weight during the block convolution) on block widths up to this many rows (32: the
                # measured 1..4 requests); silu / merge triggers keep their measured widths.
                Param("PDL_CONV_MAX_ROWS", 32, 32, 128, default_for=_tep8_gen_wide_winner(64, 32)),
                # The PDL chain's selector fork (the candidate selector's hidden_projection on a side
                # stream under the draft LM-head GEMM) on draft heads of several requests too, every
                # multiple of K = 7 rows up to this (7: the one-request head only, as measured). Same bits.
                Param("SELECTOR_FORK_MAX_ROWS", 7, 7, 112),
                # 1: the 16..64-row drafter split-K entries at deeper A/B pipelines (the split-K kernel's
                # oversized shared memory: 8 / 10 stages at the 64x64 tiles instead of the opt-in's 5 / 7);
                # the same tiles and K splits, so the same bits (fixed_low_m_gemm._TP8_DRAFTER_DEEP_STAGES).
                Param("DEEP_STAGES", 0, 0, 1, default_for=_tep8_gen_wide_winner(1, 0)),
                # 1: every drafter MLP's silu(gate) * up in its gate_up GEMM's epilogue on 8..64-row blocks
                # (kernels/drafter_silu_fold; gate_up weights interleaved in place, the other rows on the
                # interleaved activation kernel). Same bits. Needs drafter_pdl_chain.
                Param("SILU_FOLD", 0, 0, 1),
            ),
            gates=(_ll_gate(),),
        ),
        "tep8_fc_shard_wide": Lever(
            Kind.NUMERIC,
            in_band(Band.DEFAULT, gated(_at_most("MAX_BATCH"))),
            # The most rows the TP8 sharded drafter fc takes (its fabric gather holds 128).
            params=_ll_params() + (Param("MAX_ROWS", 64, 8, 128, default_for=_wide128(64)),),
            gates=(_ll_gate(),),
        ),
        "weight_prefetch_after_join": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ll_params(),
        ),
        "next_qkv_prefetch_skip": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ll_params()
            + (
                Param("MAX_ROWS", 8, 8, 64),
            ),  # the largest decode step (rows) that forks no next-QKV prefetch
        ),
        # int9 (USER approved 2026-11-11 06:00): the one-cluster block is the EXACT build only, an
        # order-only change: the replaced ONE2 / split path's operations at each rounding point
        # (ex2.approx.ftz.f32 / div.full.f32, fp32 sums, bf16 P / V, fp32 merge; gen lane G124: 8-rank
        # gate, mean 0.377 vs 0.372 ulp vs fp64). ``auto`` runs it where the original ran (the default
        # band's TEP8 gen steps of 2..8 requests, 2..11 at 9..16). The original f16 build (int7b: LOSSY)
        # is gone: EXACT is fixed at 1, so no knob selects it.
        "swa_one_cluster_block": Lever(
            Kind.NUMERIC,
            in_band(
                Band.DEFAULT, gated(_yielding(("swa_blocks_per_program",), _at_most("MAX_BATCH")))
            ),
            after=("swa_blocks_per_program",),
            # The fewest and the most requests of a TEP8 step the one-cluster block takes (5..8:
            # the ONE2 range it replaces; from 2, the split kernels' 2..4-request steps, and up to
            # 16, each request's clusters as at 5..8 requests).
            params=_ll_params()
            + (
                Param("TEP8_MIN_REQUESTS", 5, 2, 5, default_for=_tep8_gen_cl8_min_requests),
                Param("TEP8_MAX_REQUESTS", 8, 8, 16, default_for=_tep8_gen_cl8_max_requests),
                # From this many requests (0: none) a TEP8 step runs ONE 8-CTA cluster per request over
                # both KV heads (swa_cl8w, the same bits): one wave up to 22 requests, where the
                # per-(request, KV head) clusters need two from 12.
                Param("TEP8_WIDE_MIN_REQUESTS", 0, 0, 16),
                # From this many requests (0: none) a TEP8 step runs swa_cl8 compiled for two CTAs per SM
                # (registers capped at 102, 100 KB SMEM; the same source, the same bits): 46 co-resident clusters,
                # one wave up to 23 requests (1 GPU: 11.8 vs 14.4 us at 12..16 requests).
                Param("TEP8_TWO_CTA_MIN_REQUESTS", 0, 0, 16),
                # Every CL8 build is the exact variant (int9: fixed at 1; the f16 build is not offered).
                Param("EXACT", 1, 1, 1),
            ),
            gates=(_ll_gate(),),
            conflicts=("swa_blocks_per_program",),
        ),
        "swa_prep_split": Lever(
            Kind.BITWISE,
            unless_rule({"swa_blocks_per_program": _SWA_BLOCKS_PER_PROGRAM}),
            params=_ll_params(),
            conflicts=("swa_blocks_per_program",),
        ),
        "swa_eight_requests": Lever(
            Kind.NUMERIC,
            in_band(
                Band.DEFAULT, gated(_yielding(("swa_blocks_per_program",), _at_most("MAX_BATCH")))
            ),
            after=("swa_blocks_per_program",),
            params=_ll_params(),
            gates=(_ll_gate(),),
            conflicts=("swa_blocks_per_program",),
        ),
        "full_attention_fold_multi": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ll_params(),
            after=("wide_full_attention",),
        ),
        "split_kv_deep_rings": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ll_params(),
        ),
        "bf16_half_window": Lever(
            Kind.NUMERIC,
            in_band(Band.DEFAULT, gated(_at_most("MAX_BATCH"))),
            params=_ll_params(),
            gates=(_ll_gate(),),
        ),
        "prerouted_experts": Lever(
            Kind.BITWISE,
            ALWAYS,
            # The most rows of a TP8 two-window call on the prerouted tables (16-token tiles
            # above 64 rows, the stock call's tile there).
            # (w256: to 256, 32-token tiles at 129..256 rows.)
            params=_ll_params() + (Param("TEP8_MAX_ROWS", 64, 64, 256, default_for=_wide128(64)),),
        ),
        "tep8_ring_buddy_wide": Lever(
            Kind.NUMERIC,
            in_band(Band.DEFAULT, gated(_at_most("MAX_BATCH"))),
            params=_ll_params()
            + (
                # The most rows the TP8 ring-buddy placement takes (its one program covers 256;
                # w256: 129..256 on a 2048-slot tile).
                Param("MAX_ROWS", 64, 9, 256, default_for=_wide128(64)),
                # Warps of its table-writing program on the 1024-slot tile (65..128 rows).
                Param("WIDE_WARPS", 8, 4, 32, values=(4, 8, 16, 32)),
            ),
            gates=(_ll_gate(),),
        ),
        "tep8_ring_buddy_pdl": Lever(
            Kind.BITWISE,
            ALWAYS,
            # 1: the table-writing program releases the expert call (FC1) right after its own
            # dependency wait instead of at exit (launch timing only; s316 lane, default off).
            params=_ll_params() + (Param("TRIGGER", 0, 0, 1),),
        ),
        "tep8_tile_placement": Lever(
            Kind.NUMERIC,
            in_band(Band.DEFAULT, gated(_at_most("MAX_BATCH"))),
            params=_ll_params()
            + (
                Param("MIN_ROWS", 41, 9, 64),  # the fewest rows placed by tile cost
                Param("MAX_ROWS", 64, 9, 256),  # the most rows placed by tile cost (w256: to 256)
                # An expert's cost A + B * (tiles - 1), in 0.1 us (GB300's 20 and 11 by default).
                Param("COST_A", 20, 1, 255),
                Param("COST_B", 11, 0, 255),
                # The two BF16 expert layers (0 and 61): their stacked call (no prerouted tables) placed
                # by the same tile rule on its own costs, BF16_COST_A + BF16_COST_B * (tiles - 1) in
                # 0.1 us over BF16_TILE-token tiles, in the same MIN_ROWS..MAX_ROWS band. 0 (default): the
                # BF16 layers keep the count rule. (VR 8-rank nsys at bs12 / 16: max - median of FC1 + FC2
                # 20 / 23 us a BF16 layer vs 3.1 / 3.7 on the NVFP4 layers.)
                Param("BF16_COST_A", 0, 0, 255),
                Param("BF16_COST_B", 0, 0, 255),
                Param("BF16_TILE", 8, 8, 64, values=(8, 16, 32, 64)),
            ),
            gates=(_ll_gate(),),
        ),
        "tep8_router_push": Lever(
            Kind.NUMERIC,
            in_band(Band.DEFAULT, gated(_at_most("MAX_BATCH"))),
            params=_ll_params()
            + (
                # The fewest rows the router push takes (kernel floor 9; 9 on a TEP8 generation
                # server, ``TEP8_GEN_ROUTER_PUSH_MIN_ROWS``).
                Param("MIN_ROWS", 33, 9, 64, default_for=_tep8_gen_router_push_min_rows),
                # The most rows a push call runs 16-CTA clusters at (half the per-CTA K: same bits; 0 = none;
                # at most step A v2's 64 rows).
                Param("CL16_MAX_ROWS", 0, 0, 64),
                # The most rows step A routes per row in every CTA (v2; above: the row-block leader, v1).
                # (128 on a TEP8 generation server of 9..16 requests: the X400 composite, _wide128.)
                Param("ROUTE_ROW_MAX", 64, 64, 128, default_for=_wide128(64)),
                # v2's rows per CTA: 2 keeps half the push's CTAs (expert group 0's) through the poll.
                Param("ROUTE_ROWS_PER_CTA", 1, 1, 2),
                # 1: the ring-buddy placement + prerouted tables after the push run as one CUDA CTA
                # (ring_buddy_cuda.rb_tables) that the routed push triggers right after its dependency wait.
                Param("RB_CUDA", 0, 0, 1),
                # (w256) The most rows a push call takes (128: the band served before; to 256, 17..32
                # requests; v1 routing above ROUTE_ROW_MAX). The same logits / routing as at <= 128.
                Param("TEP8_MAX_ROWS", 128, 128, 256),
                # (w256, 0: off) Calls above 128 rows route in a v2-only routed build (2 CTAs per SM, so 192 / 256
                # rows stay resident and route in the push instead of the separate routing kernel); same bits.
                Param("ROUTE_WIDE_V2", 0, 0, 1),
                # (w256) the fewest rows ROUTE_WIDE_V2's v2-only build takes (129: every call above 128 rows);
                # calls of 129..(this - 1) rows keep the v1 routed build (same bits: both _full_routing_kernel's).
                Param("ROUTE_WIDE_V2_MIN_ROWS", 129, 129, 256),
            ),
            gates=(_ll_gate(),),
        ),
        "tep8_router_prewait": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ll_params(),
        ),
        "finalize_entry_trigger": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ll_params(),
        ),
        "tep8_finalize_transport_64": Lever(
            Kind.BITWISE,
            ALWAYS,
            # The TP8 transport's token layout: its 17..TOKENS-token calls run the flat build.
            params=_ll_params()
            + (
                # (w256: 256 -- with CHUNKED, the 129..256-token calls in one launch.)
                Param(
                    "TOKENS", 64, 16, 256, values=(16, 32, 64, 128, 256), default_for=_wide128(64)
                ),
                # 1: the row map read before the dependency wait and verified after it
                # (finalize_allreduce_transport.Transport.prefetch; the same bits).
                Param("INDEX_PREFETCH", 0, 0, 1),
                # 1 (w256): the TP8 workspace also takes the built-in's chunked calls of 129..
                # tokens (17+ requests at K = 7) as consecutive calls of at most TOKENS tokens
                # (TP8_CHUNKED), instead of the built-in forward's unfused combine + cast +
                # all-reduce above 128 tokens. Order-only against that path (same BF16 partials,
                # FP32 rank-order sum); every row is what a <= TOKENS-token call returns. 0: off.
                Param("CHUNKED", 0, 0, 1),
                # (w256) TP8 calls of at least this many tokens (to 256) on the two-phase finalize
                # (ops/moe/finalize_two_phase, from the TEP16 target's B(ii)): owner-reduce + multicast gather,
                # this transport's arithmetic (the same bits). 0: off.
                Param("TWO_PHASE_MIN_TOKENS", 0, 0, 256),
            ),
        ),
        "hop_entry_trigger": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ll_params()
            + (
                # 1: the decode input boundary (model_boundary) also releases its dependent,
                # layer 0's QKV GEMM, at entry on steps of INPUT_BOUNDARY_MIN_ROWS (9) rows or
                # more, so the GEMM streams its weight during the boundary as every later
                # layer's QKV does behind its post-FFN join; 0 the exit trigger everywhere.
                Param("INPUT_BOUNDARY", 0, 0, 1),
            ),
        ),
        "tep8_early_multicast_hop": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ll_params(),
        ),
        "tep8_direct_hop": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ll_params()
            + (
                # The direct hop's last row count: 128 also publishes the 65..128-row o_proj;
                # w256: 256 the 129..256-row o_proj too (a 256-row workspace).
                Param("MAX_ROWS", 64, 64, 256, values=(64, 128, 256), default_for=_wide128(64)),
                # How o_proj of a 65..128-row step publishes: two 64-row launches, or one
                # launch of 64-row (grid) or 128-row (tile128) token tiles; split_k 1 in each.
                Param("PRODUCER", "grid", choices=("chunks", "grid", "tile128", "2cta"), one=True),
                # The one-launch producer's stages (0: the mode's, grid 3 / tile128 4).
                Param("PRODUCER_STAGES", 0, 0, 12),
                # The direct hop's first row count (33: the 5..8-request band; down to 9, the
                # 2..4-request steps' o_proj scatters on the served wide64 tactic too).
                Param("MIN_ROWS", 33, 9, 33),
                # The fewest rows of an armed step whose QKV call forks the L2 prefetch of the
                # layer's o_proj weight (0: none; at most 32 rows the module's rule already forks).
                Param("PREFETCH_MIN_ROWS", 0, 0, 128),
                # 1: the direct consumer polls its slots before its dependency wait (same bits).
                Param("POLL_FIRST", 0, 0, 1),
                # 1: the 33..128-row scatter o_proj producer's idle warp prefetches its CTA's
                # weight tile into L2 at kernel entry, before the dependency wait (loads only;
                # same bits). x82: -0.05..-0.14 ms at bs10..16, bs4 unchanged.
                Param("L2_PREFETCH", 0, 0, 1),
            ),
        ),
        "tep8_publish_join": Lever(
            Kind.BITWISE,
            ALWAYS,
            # 1: the post-FFN join releases its dependent (the next QKV GEMM) at entry, before its own
            # dependency wait (launch order only; s316 lane, default off).
            params=_ll_params() + (Param("ENTRY_TRIGGER", 0, 0, 1),),
        ),
        # UNPOOL_DIRECT_WIDE: the 2..4-request (16..32-row) steps at TP8 take the direct unpool
        # transport's wide layout (unpool_into_boundary): the unpool split-K's epilogue publishes its
        # slice and the post-FFN boundary only polls (same GEMM, same boundary arithmetic).
        "tep8_unpool_direct_wide": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ll_params(),
        ),
        "tep8_pool_programmatic_fork": Lever(
            Kind.BITWISE,
            ALWAYS,
            # 1: the pool branch's norm + NVFP4 kernels (pool publication, pool gather) load the norm gain and the
            # global scale before their dependency wait / poll (the same bits).
            params=_ll_params()
            + (Param("NORM_PREFETCH", 0, 0, 1, default_for=_tep8_gen_wide_winner(1, 0)),),
        ),
        "tep8_wide_projection_gathers": Lever(
            Kind.NUMERIC,
            in_band(Band.DEFAULT, gated(_at_most("MAX_BATCH"))),
            params=_ll_params(),
            gates=(_ll_gate(),),
        ),
        "step_commit_v2": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ll_params()
            + (
                # The overlap scheduler's two input corrections of a captured decode graph as
                # one launch (ops/runtime/overlap_offsets, step_loop_inline_commits' applier):
                # 0 the two built-in adds, 1 one launch, 2 one launch that releases its
                # programmatic dependent (the input boundary, which waits first) at entry.
                Param("OVERLAP_FOLD", 0, 0, 2),
                # 1: the step-input and sampler-store launches carry their descriptor table in
                # the kernel parameter block when it fits (ops/runtime/step_commit_inline, as
                # step_loop_inline_commits' commits); 0 the pinned-table kernel.
                Param("INLINE_COMMIT", 0, 0, 1),
            ),
        ),
        "wide_norm_gate_pdl": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ll_params()
            + (
                # (row, head) pairs per program of the PDL norm-gate on its 9..128-row steps (1: the
                # built-in grid). Larger values make the grid one wave, so its after-wait trigger
                # releases the output projection (which streams its weight before its own wait) as
                # soon as the attention completes instead of at the last wave. Same statements per
                # pair: bitwise.
                Param("HEADS_PER_PROGRAM", 1, 1, 32, values=(1, 2, 4, 8, 16, 32)),
                # When it releases the output projection on those steps: after its wait (stock) or at
                # its entry (the projection may also stream its weight under the attention's tail).
                # Launch order only: bitwise.
                Param("TRIGGER", "after_wait", choices=("after_wait", "entry"), one=True),
            ),
        ),
        "drafter_kv_glue": Lever(
            Kind.BITWISE,
            unless_on(("drafter_kv_side_chain",)),
            after=("drafter_kv_side_chain",),
            # Not in the 'wall' set the TEP8 9..16-request gate ran (the rulers emulate the
            # generation role over a windowed arena, where it was off at max_batch_size 16): its
            # default stays at 8 requests on every deployment.
            params=(Param("MAX_BATCH", LL_MAX_BATCH, 1, 64),),
            conflicts=("drafter_kv_side_chain",),
        ),
        "drafter_kv_fork_before_head": Lever(
            Kind.BITWISE,
            ALWAYS,
            # PAGED_COMMIT: alias only (kept for the round-3 / E2E envs) of upstream #21789's
            # paged_context_commit, the one-launch paged commit on the context-K/V fork; the value
            # engages nothing of its own (drafter.py logs the mapping).
            params=_ll_params()
            + (
                Param("PAGED_COMMIT", 0, 0, 1),
                # 1: over a paged drafter context, the forked side chain's block-table decode, columns and
                # positions in one launch and every layer's K / V into the pages in one
                # (ops/spec/context_kv_store_paged) instead of ~45 built-in launches; the same GEMMs, norms, RoPE.
                # 2: also its prep and the sharded fc GEMM right after the last capture layer (the early fork).
                Param("PAGED_CHAIN", 0, 0, 2),
            ),
        ),
        "context_sp_extra_parts": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=(
                Param("MAX_BATCH", LL_MAX_BATCH, 1, 64),
                Param(
                    "PARTS",
                    ("router_input", "ce_gather"),
                    choices=("shard_routing", "router_input", "ce_gather"),
                    default_for=lambda d: (
                        _CTX_TEP4_EXTRA_PARTS if _ctx_tep4(d) else ("router_input", "ce_gather")
                    ),
                ),
            ),
        ),
        "swa_block_long_splits": Lever(
            Kind.NUMERIC,
            NEVER,
            # KEYS: the key tiles the split kernel is built for (swa_block.LONG_SPLIT_KEYS).
            params=(
                Param("KEYS", 128, 64, 256, values=(64, 128, 256)),
                Param("MIN_ROWS", 8, 1, 32),
            ),
            # The 32-key builds of the same steps: the one-cluster block and the multi-block
            # programs take a step before the split path, and run no long split.
            conflicts=("swa_one_cluster_block", "swa_blocks_per_program"),
        ),
        # The half window's placement on the NVFP4 decode calls of 33..64 rows (five to eight
        # requests). On only by its knob.
        "batch_half_window_experts": Lever(Kind.NUMERIC, NEVER),
        # The sliding-window decode block on TP4 decode steps of five to MAX_REQUESTS requests.
        # On only by its knob.
        "batch_swa_block": Lever(
            Kind.NUMERIC,
            NEVER,
            # MAX_REQUESTS stops at 8: above it the split grid's waves make the block slower
            # than the trtllm-gen FMHA path the prep append feeds.
            params=(Param("MAX_REQUESTS", 8, 5, 8), Param("O_PROJ_PREFETCH", 1, 0, 1)),
        ),
        "step_window_compact": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "fmk_context_fmha": Lever(
            Kind.NUMERIC,
            gated(_fmk_context_fmha),
            gates=(_AGG_TEP4_GATE, _CTX_TEP4_FMK_GATE),
            # 1: also every pure context step (ctx-only / prefill steps): the default on sm107
            # TEP4 context servers (gated there, _CTX_TEP4_FMK_GATE); 0 elsewhere, where the
            # accuracy gate covers mixed steps only.
            params=(Param("PURE_CONTEXT", 0, 0, 1, default_for=lambda d: int(_ctx_tep4(d))),),
        ),
        # Pure context steps of a disaggregated context server on the same multi-instance
        # context FMHA (its own install, its own gate); the pipeline stage target serves it at
        # the stage's heads. On only by its knob until its gate holds.
        "fmk_pure_context_fmha": Lever(
            Kind.NUMERIC,
            gated(_on_ctx_sm107_tep4("the context FMHA of pure context steps")),
            gates=(_CTX_TEP4_GATE_PENDING,),
        ),
        # The last layer of a disaggregated context server's sequence-parallel steps past its K/V
        # append on the LM-head rows only (``kernels/context_last_layer_tail``). On only by its
        # knob until its gate holds. ``MAX_DEPTH``: a step whose deepest context request reaches
        # that many keys runs the layer unchanged (past it the tail's one-row-per-request decode
        # FMHA over every key costs more than it saves).
        "context_last_layer_tail": Lever(
            Kind.NUMERIC,
            gated(
                _on_ctx_sm107_tep4("the last layer of large context steps on the LM-head rows only")
            ),
            gates=(_CTX_TEP4_TAIL_GATE_PENDING,),
            params=(Param("MAX_DEPTH", 262144, 1, 1 << 30),),
        ),
        "context_kv_late_join": Lever(
            Kind.BITWISE,
            either(
                _CONTEXT_KV_LATE_JOIN,
                unless_rule(
                    {
                        "drafter_kv_fork_before_head": _LL_DEFAULT_BAND,
                        "drafter_kv_glue": _DRAFTER_KV_GLUE_RULE,
                    }
                ),
            ),
            after=("drafter_kv_side_chain",),
        ),
        "context_kv_fork_wide": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        # Writes the bytes of the built-in Triton join (the arithmetic of ``tep8_publish_join``'s
        # bitwise cluster join) and of the built-in capture cast. It takes only the capture
        # layers' joins of 33..768-row calls (``kernels/multi_request_ffn_join``'s install
        # defaults), where it also replaces the built-in capture's cast and copy.
        "multi_request_ffn_join": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "hop_cluster_split": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=(
                Param("CLUSTER", 2, 1, 2),
                Param("ROWS", ((192, 208), (248, 264)), 9, 1024, ranges=True),
            ),
        ),
        "drafter_fc_inbox_wide": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=(Param("MAX_ROWS", 384, 128, 512), Param("PROGRAMS", 128, 1, 256)),
        ),
        "l061_tile32_pin": Lever(
            Kind.NUMERIC,
            in_band(
                Band.AGGREGATED_SM107,
                gated(_on_agg_tep4("layer 0 / 61 owner-window expert calls of 256..384 rows")),
            ),
            gates=(_AGG_TEP4_GATE,),
        ),
        "router_staged_epilogue": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        # Writes the bytes of the built-in pool GEMM (cuBLAS's split-K-1 kernels share its K
        # order; bitwise at every served row count 584..768 on the checkpoint's weights).
        # ROWS: the decode row counts whose pool GEMM runs the pinned algorithm; the default is
        # the 744..768-row graphs, which only a wide generation server's row cap reaches.
        # Lossless-bitwise, so on by default wherever it can run.
        "pool_gemm_pin": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=(Param("ROWS", ((744, 768),), 520, 768, ranges=True),),
        ),
        # Pins one TRTLLM-gen [TILE_N, CONFIG] for the NVFP4 expert calls of 512..1023 rows (the
        # AutoTuner's 512-row bucket) on every rank, in place of each rank's own profiled pick.
        # The tactics of these calls write the same expert rows.
        # Lossless-bitwise, so on by default wherever it can run.
        "expert_tactic_pin": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=(Param("TILE_N", 64, 8, 256), Param("CONFIG", 26, 0, 4095)),
        ),
        # The context-K/V fork over a paged drafter context on decode steps of 65..128 requests,
        # forked behind the LM head's logits gather (``context_kv_fork.GATHER_POINT``) so the side
        # chain's fc gather is ordered after it on the TP communicator; the store, the table
        # decode and the commit are the built-in's or the existing paged launches. Same ops on
        # another stream: bitwise, so on by default wherever it can run.
        "context_kv_fork_paged_wide": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        # The LM head of decode steps of 65..96 requests written by every rank straight into the
        # owner ranks' pages (each request's rows owned by one rank of four), the owners soft-cap,
        # take the target probabilities of and chain-verify their slice, and the tokens and
        # counts are all-gathered: no logits gather, a quarter of the verify per rank. The same
        # kernels on the same rows (the chain kernel with the request's batch index as its
        # Philox subsequence) as the built-in verify on the owner rank. Each request is decided
        # with the owner's draft-probability rows, which are not bitwise equal across TP ranks,
        # where the built-in emits rank 0's decision: numeric. On only by its knob.
        "lm_head_row_owner": Lever(Kind.NUMERIC, _never("opt-in: on only by its knob")),
        # The NVFP4 MoE layers of the ROWS decode row counts as a row-owner island
        # (``kernels/row_owner_moe``): each rank runs the post-attention join, the MoE
        # prologue, the expert finalize, the unpool GEMM and the post-FFN join on the rows it
        # owns, and the expert dispatch and the next layer's input travel as owner-row
        # pushes. Every stored value is the built-in kernels' (same kernels, same per-row
        # launch choices); bitwise. Not with a lever that changes the router GEMM or the pool
        # GEMM it reproduces; ROWS starts above hot_replicas' 512 rows. On by default where
        # neither of its conflicts is on (in practice: by its knob, on wide generation servers).
        "row_owner_moe": Lever(
            Kind.BITWISE,
            unless_on(("router_staged_epilogue", "pool_gemm_pin")),
            params=(Param("ROWS", ((584, 768),), 520, 768, ranges=True),),
            conflicts=("router_staged_epilogue", "pool_gemm_pin"),
            excluded=_row_owner_excluded,
        ),
        # Integer words of the built-in decode, bitwise (block_table.py). On a generation
        # server whose drafter context is in the KV cache manager's pages, the worker's own
        # page-table decode of steps the context-K/V fork does not take (more than 64
        # requests) in one launch instead of the built-in's eleven. Lossless-bitwise, so on by default wherever it can
        # run.
        "drafter_block_table_decode": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        # The device words of the staged copy are unchanged: past a row's readable columns it
        # holds the bad-page word either way (runtime._BlockTailMarks). The copy of a manager
        # with attention-op pool groups fills only the columns a row may not already hold as
        # that word, instead of every column past the readable ones every step.
        # Lossless-bitwise, so on by default wherever it can run.
        "pool_group_tail_marks": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "split_kv_deep_ring": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "context_prep_rows_vec": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "context_embedding_replicated": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "context_sp_router_input": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        # The context-step reduce-scatter overlaps: bitwise where the chunked BF16 GEMMs are
        # row-split invariant and NCCL reduces a chunk as the whole buffer (both held on GB300;
        # their sm107 GPU tests decide it), else lossless-numeric.
        # The o_proj overlap and the tail pipeline ('ovts', bitwise on sm107) are on by default
        # on sm107 TEP4 context servers; the experts' overlap is not measured yet.
        "context_sp_oproj_overlap": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ctx_overlap_params(),
            after=("context_fp8_dense",),
            excluded=_ctx_overlap_excluded("o_proj"),
        ),
        "context_sp_tail_pipeline": Lever(
            Kind.BITWISE,
            ALWAYS,
            after=("context_fp8_dense", "context_sp_router_input", "context_sp_extra_parts"),
            excluded=_ctx_overlap_excluded("pool", needs_router_input=True),
        ),
        "context_sp_moe_rs_overlap": Lever(
            Kind.BITWISE,
            ALWAYS,
            params=_ctx_overlap_params(),
            after=("context_fp8_dense",),
            excluded=_ctx_overlap_excluded("unpool"),
        ),
        "decode_qkv_mxfp8": Lever(
            Kind.LOSSY,
            in_band(Band.AGGREGATED_SM107, _lossy),
            after=("context_fp8_dense",),
            excluded=_decode_qkv_excluded,
        ),
        "swa_blocks_per_program": Lever(
            Kind.BITWISE,
            either(
                _SWA_BLOCKS_PER_PROGRAM,
                unless_rule(
                    {"swa_one_cluster_block": _LL_GATED, "swa_eight_requests": _LL_GATED},
                    unless_on(("swa_prep_split",)),
                ),
            ),
            after=("swa_prep_split",),
        ),
        "wide_qkv_prefetch": Lever(
            Kind.BITWISE,
            either(
                _WIDE_QKV_PREFETCH,
                unless_rule(
                    {
                        "weight_prefetch_after_join": _LL_DEFAULT_BAND,
                        "splitk_frontier_tables": _LL_GATED,
                    }
                ),
            ),
        ),
        "wide_capture_fold": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "megakernel": Lever(
            Kind.BITWISE,
            ALWAYS,
            requires=MEGAKERNEL_NUMERICS,
        ),
        "megakernel_one_exchange_boundaries": Lever(
            Kind.NUMERIC,
            in_band(
                Band.AGGREGATED_SM107, gated(_on_single_request("the megakernel's MoE boundaries"))
            ),
            gates=(_SINGLE_REQUEST_GATE,),
        ),
        "megakernel_two_term_pool": Lever(
            Kind.NUMERIC,
            in_band(
                Band.AGGREGATED_SM107,
                gated(_on_single_request("the megakernel's pool and router GEMVs")),
            ),
            gates=(_SINGLE_REQUEST_GATE,),
            requires=("megakernel_one_exchange_boundaries",),
        ),
        "megakernel_router_by_linearity": Lever(
            Kind.NUMERIC,
            in_band(
                Band.AGGREGATED_SM107, gated(_on_single_request("the megakernel's router logits"))
            ),
            gates=(_SINGLE_REQUEST_GATE,),
        ),
        "megakernel_zero_exchange_boundary": Lever(
            Kind.BITWISE,
            ALWAYS,
            requires=("megakernel_one_exchange_boundaries",),
            excluded=_megakernel_variant_excluded,
        ),
        "layer0_qkv_table": Lever(
            Kind.BITWISE,
            either(
                _LAYER0_QKV_TABLE, unless_rule({"weight_prefetch_after_join": _LL_DEFAULT_BAND})
            ),
        ),
        "edge_bf16_moe_front": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "last_layer_boundary": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "context_kv_fork_last_layer": Lever(
            Kind.BITWISE,
            either(
                _CONTEXT_KV_FORK_LAST_LAYER,
                unless_rule({"drafter_kv_fork_before_head": _LL_DEFAULT_BAND}),
            ),
        ),
        # Its LM-head GEMM runs at one fixed split-K tactic on every rank, which can differ from
        # the autotuner's per-rank choice for the stock 8-row head it replaces.
        "distributed_head_sampler": Lever(
            Kind.NUMERIC,
            in_band(Band.AGGREGATED_SM107, gated(_on_single_request("8-row verify heads"))),
            gates=(_SINGLE_REQUEST_GATE,),
            requires=("step_loop_on_execution_stream",),
            conflicts=("lm_head_inbox",),
        ),
        "drafter_preamble_late_join": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        # On by default at max_batch_size 1 only: its CF / SF folds relayout drafter weights at install, so every
        # multi-request drafter step pays an unpermute / deinterleave per layer. Hecate TEP8 gens
        # 2026-11-17 (max batch 4 and 16, leave-one-out, two cycles): with it on the decode step is
        # 0.9-1.1 % faster at bs1 and 0.35-1.95 % slower at bs2-16 than with it off.
        "drafter_folds": Lever(
            Kind.BITWISE,
            _on_max_batch_one("one-request 8-row drafter block decodes"),
        ),
        "draft_head_tail": Lever(
            Kind.BITWISE,
            ALWAYS,
        ),
        "step_loop_inline_commits": Lever(
            Kind.BITWISE,
            either(_STEP_LOOP_INLINE_COMMITS, unless_rule({"step_commit_v2": _LL_DEFAULT_BAND})),
        ),
        "rejection_sampler": Lever(
            Kind.COMPOSITION,
            in_band(
                Band.AGGREGATED_SM107,
                _on_single_request("rejection-sampled one-request decode steps"),
            ),
        ),
        # The compositions' own paths: on wherever the composition serves them, in the stock engine too.
        "attention_output_into_hop": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "batch_dense_gemms": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "bf16_routed_experts": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "boundary_early_release": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "capture_fold": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "context_kv_fork": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "context_prep": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "context_sequence_parallel": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "deferred_input_prep": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "dense_gemm_prefetch": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "dense_gemms": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "draft_head": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "draft_head_sampled": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "drafter_block_conv": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "drafter_block_decode": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "drafter_fc_inbox": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "drafter_fc_shard": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "drafter_fixed_gemms": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "drafter_hops": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "drafter_pdl_chain": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "early_expert_ids": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "finalize_transport": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "finalize_two_shot": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "full_attention_prep_append": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "full_attention_prep_fold": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "full_attention_split_kv": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "fused_accept": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "mixed_step_gen_rows": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "model_boundary": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "moe_prologue": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "next_qkv_prefetch": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "paged_context_commit": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "paged_ctx_block_tables": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "paged_ctx_kv_store": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "path_walk_tail": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "persistent_experts": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "pool_into_route": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "post_attention_hop": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "preattn_prep": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "projection_gathers": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "rejection_fold": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "replicated_embedding": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "replicated_experts": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "sampler_store": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "split_sampler": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "swa_block": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "swa_paged_decode": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "three_window_placement": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "unpool_into_boundary": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "verify_fold": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "wide_draft_topk": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "wide_drafter_window_attention": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "wide_moe_allreduce": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "wide_persistent_experts": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "wide_post_attention_hop": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
        "wide_prologue": Lever(Kind.COMPOSITION, ALWAYS, stock=ALWAYS),
    }
)


def default_param(name: str, key: str) -> object:
    """Lever ``name``'s declared default for parameter ``key``.

    Raises:
        KeyError: when ``name`` declares no parameter ``key``.
    """
    rule = LEVERS.get(name)
    if rule is None:
        raise KeyError(f"lever {name} declares no parameter {key}")
    return rule.param(key).default


def knob_name(name: str, key: Optional[str] = None) -> str:
    """The environment variable of lever ``name`` (or of its parameter ``key``)."""
    return PREFIX + name.upper() + ("" if key is None else SEP + key)


# --------------------------------------------------------------------------
# The environment
# --------------------------------------------------------------------------
def snapshot_of(environ: Mapping[str, str]) -> Dict[str, str]:
    """The variables of ``environ`` the resolver reads: lever knobs, retired names, facts.

    A lever knob set to an empty value is left out (unset); a retired name is kept
    whatever its value.
    """
    snapshot: Dict[str, str] = {}
    for name, value in environ.items():
        if name.startswith(PREFIX):
            if value.strip():
                snapshot[name] = value
        elif retired_replacement(name) is not None:
            snapshot[name] = value
        elif name in (KV_DEFER_CLOSE_ENV, UCX_RNDV_THRESH_ENV):
            snapshot[name] = value
    return snapshot


_SNAPSHOT: Optional[Mapping[str, str]] = None


def environment() -> Mapping[str, str]:
    """The snapshot of this process's environment, read on the first call and never again."""
    global _SNAPSHOT
    if _SNAPSHOT is None:
        _SNAPSHOT = types.MappingProxyType(snapshot_of(os.environ))
    return _SNAPSHOT


def facts_of(snapshot: Mapping[str, str]) -> Facts:
    """The two shared-path facts of ``snapshot``."""
    return Facts(
        kv_defer_close=snapshot.get(KV_DEFER_CLOSE_ENV) == "1",
        ucx_rndv_thresh_set=bool(snapshot.get(UCX_RNDV_THRESH_ENV, "").strip()),
    )


#: The prefix of every item ``tagged`` lists for the TP agreement.
AGREEMENT_TAG = "lever environment: "


def tagged(snapshot: Mapping[str, str]) -> List[str]:
    """``snapshot`` as items for ``dispatch.agreed_union``: every rank must list the same.

    The knobs and the retired names with their raw values, the facts as resolved.
    """
    facts = facts_of(snapshot)
    items = [
        f"{AGREEMENT_TAG}{name}={value}"
        for name, value in sorted(snapshot.items())
        if name not in (KV_DEFER_CLOSE_ENV, UCX_RNDV_THRESH_ENV)
    ]
    items.append(f"{AGREEMENT_TAG}{facts.describe()}")
    return items


@dataclasses.dataclass(frozen=True)
class Knobs:
    """The lever knobs of a snapshot, parsed and validated."""

    #: Lever name -> ``auto`` | ``on`` | ``off``.
    modes: Mapping[str, str]
    #: (lever name, parameter key) -> parsed value.
    params: Mapping[Tuple[str, str], object]
    default: str
    role: str
    #: The knobs as set (names under ``PREFIX``).
    raw: Mapping[str, str]


def parse_knobs(snapshot: Mapping[str, str], names: Iterable[str]) -> Knobs:
    """The lever knobs of ``snapshot`` for a registry of lever ``names``.

    Raises:
        ValueError: naming every retired switch that is set, every unknown name under
            ``PREFIX``, every value outside its set and every parameter outside its range
            or on a lever that declares no such parameter.
    """
    names = frozenset(names)
    errors: List[str] = []
    for name in sorted(snapshot):
        replacement = retired_replacement(name)
        if replacement is not None:
            errors.append(f"{name} is retired (use {replacement})")
    modes: Dict[str, str] = {}
    params: Dict[Tuple[str, str], object] = {}
    default, role = ON, AUTO
    raw: Dict[str, str] = {}
    for name in sorted(snapshot):
        if not name.startswith(PREFIX):
            continue
        value = snapshot[name].strip().lower()
        raw[name] = snapshot[name].strip()
        rest = name[len(PREFIX) :]
        if rest in ("_BITWISE", "_NUMERICS"):
            continue  # retired: reported above
        if rest == "_DEFAULT":
            if value not in DEFAULT_VALUES:
                errors.append(f"{name}={value!r}: expected one of {list(DEFAULT_VALUES)}")
            default = value
            continue
        if rest == "_ROLE":
            allowed = [AUTO] + [r.value for r in Role if r is not Role.UNKNOWN]
            if value not in allowed:
                errors.append(f"{name}={value!r}: expected one of {allowed}")
            role = value
            continue
        lever_part, sep, key = rest.partition(SEP)
        lever = lever_part.lower()
        if rest.startswith("_") or lever not in names or lever.upper() != lever_part:
            close = difflib.get_close_matches(lever, sorted(names), n=1)
            hint = f" (did you mean {knob_name(close[0])}?)" if close else ""
            errors.append(f"{name}: no lever of this target is named {lever!r}{hint}")
            continue
        if not sep:
            if value not in LEVER_VALUES:
                errors.append(f"{name}={value!r}: expected one of {list(LEVER_VALUES)}")
            else:
                modes[lever] = value
            continue
        rule = LEVERS.get(lever)
        declared = [param.key for param in rule.params] if rule is not None else []
        if key not in declared:
            errors.append(
                f"{name}: lever {lever} declares "
                + (f"the parameters {declared}" if declared else "no parameter")
            )
            continue
        try:
            params[(lever, key)] = rule.param(key).parse(snapshot[name])
        except ValueError as error:
            errors.append(f"{name}: {error}")
    if errors:
        raise ValueError("; ".join(errors))
    return Knobs(modes, params, default, role, raw)


def no_composition_notes(snapshot: Mapping[str, str], names: Iterable[str]) -> List[str]:
    """For a deployment without a composition: validate ``snapshot``, then say what it sets.

    Raises:
        ValueError: as ``parse_knobs``.
    """
    knobs = parse_knobs(snapshot, names)
    return [
        f"{name}={value} has no effect without a composition" for name, value in knobs.raw.items()
    ]


# --------------------------------------------------------------------------
# The deployment
# --------------------------------------------------------------------------
def arena_of(spec_worker, draft_model, paged_backends: Sequence[str]) -> Arena:
    """The drafter context arena the DFlash worker will keep, by its own predicate.

    ``paged`` when the drafter attention backend is one of ``paged_backends``
    (``speculative.dflash._PAGED_ATTENTION_BACKENDS``) or the drafter asks for pages
    (``_paged_ctx_cache``); else ``windowed`` when the worker windows its arena; else
    ``dense``. An ``AUTO`` worker backend is the drafter's, as the worker resolves it at
    the drafter's first forward.
    """
    backend = getattr(spec_worker, "_dflash_attention_backend", None)
    if backend == "AUTO":
        backend = getattr(draft_model, "dflash_attention_backend", backend)
    if backend in tuple(paged_backends) or getattr(draft_model, "_paged_ctx_cache", False):
        return Arena.PAGED
    if getattr(spec_worker, "windowed_ctx_arena", False):
        return Arena.WINDOWED
    return Arena.DENSE


def worker_arena(spec_worker) -> Optional[Arena]:
    """The arena the DFlash worker built, once it has (None before its first drafter step)."""
    if not getattr(spec_worker, "_ctx_buf_inited", False):
        return None
    if getattr(spec_worker, "_ctx_paged", False):
        return Arena.PAGED
    if getattr(spec_worker, "_ctx_base", None) is not None:
        return Arena.WINDOWED
    return Arena.DENSE


#: The role source of a server with a cache transceiver and no role hint.
DISAGGREGATED_UNHINTED = "no hint on a disaggregated server"


def role_hint(snapshot: Mapping[str, str]) -> str:
    """The role hint ``ROLE_KNOB`` of ``snapshot`` as set (lower-cased), ``auto`` when unset.

    Not validated here: ``parse_knobs`` names a bad value with every other grammar error.
    """
    return snapshot.get(ROLE_KNOB, AUTO).strip().lower() or AUTO


def role_of(snapshot: Mapping[str, str], disaggregated: Optional[bool]) -> Tuple[Role, str]:
    """The server's role and where it came from.

    The hint ``ROLE_KNOB`` wins when set. Otherwise ``agg`` without a cache transceiver,
    and ``unknown`` with one or without engine facts.

    Raises:
        ValueError: for a hint the server contradicts: ``agg`` with a cache transceiver,
            ``ctx`` or ``gen`` without one.
    """
    hint = role_hint(snapshot)
    if hint != AUTO:
        role = Role(hint)
        if role is Role.AGG and disaggregated is True:
            raise ValueError(f"{ROLE_KNOB}=agg on a server with a cache transceiver")
        if role in (Role.CTX, Role.GEN) and disaggregated is False:
            raise ValueError(f"{ROLE_KNOB}={hint} on a server without a cache transceiver")
        return role, "hint"
    if disaggregated is None:
        return Role.UNKNOWN, "no engine facts"
    if disaggregated:
        return Role.UNKNOWN, DISAGGREGATED_UNHINTED
    return Role.AGG, "no cache transceiver"


def draft_logit_transform_of(draft_model) -> Optional[bool]:
    """Whether ``draft_model`` declares a logit transform, or None without a drafter.

    ``DFlashForCausalLM`` resolves the transform its config declares into
    ``logit_multiplier`` (1.0: none) and ``logit_softcap`` (None: none) at construction.
    A drafter without those attributes declares none.
    """
    if draft_model is None:
        return None
    multiplier = getattr(draft_model, "logit_multiplier", 1.0)
    softcap = getattr(draft_model, "logit_softcap", None)
    return bool(float(multiplier) != 1.0 or softcap is not None)


def deployment_of(
    model_config,
    compute_capability,
    arena: Arena,
    engine_facts,
    snapshot: Mapping[str, str],
    draft_logit_transform: Optional[bool] = None,
    rejection_sampling: Optional[bool] = None,
) -> Deployment:
    """The ``Deployment`` of a model: its config's mapping, the device, the arena and the engine.

    ``engine_facts`` carries ``max_batch_size``, ``max_num_tokens`` and ``disaggregated``
    (``_router_index.DeploymentFacts``), or is None when the target class was built without
    them. ``draft_logit_transform`` is ``draft_logit_transform_of`` the drafter;
    ``rejection_sampling`` whether the speculative config asks for rejection sampling once
    forced acceptance has applied (``dispatch.rejection_sampling_of``).

    Raises:
        ValueError: as ``role_of``.
    """
    major, minor = compute_capability
    role, source = role_of(snapshot, getattr(engine_facts, "disaggregated", None))
    return Deployment(
        sm=10 * int(major) + int(minor),
        tp=int(model_config.mapping.tp_size),
        max_batch_size=getattr(engine_facts, "max_batch_size", None),
        max_num_tokens=getattr(engine_facts, "max_num_tokens", None),
        role=role,
        role_source=source,
        arena=arena,
        facts=facts_of(snapshot),
        draft_logit_transform=draft_logit_transform,
        rejection_sampling=rejection_sampling,
    )


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LeverTable:
    """Every served lever's decision, the deployment it was resolved for, and the knobs set."""

    deployment: Deployment
    #: Served lever name -> its decision, in registry order.
    decisions: Mapping[str, Decision]
    #: The rule-table levers the composition does not serve.
    unserved: FrozenSet[str]
    #: The knobs as set (names under ``PREFIX``).
    knobs: Mapping[str, str]
    #: ``__DEFAULT``: ``on`` or ``off``.
    default: str = ON
    #: Things the log should say as warnings (an unknown role, ``on`` outside a gate, ...).
    warnings: Tuple[str, ...] = ()

    def on(self, name: str) -> bool:
        decision = self.decisions.get(name)
        return decision is not None and decision.on

    def param(self, name: str, key: str) -> object:
        """``name``'s resolved parameter ``key``, or its default when ``name`` is not served.

        Raises:
            KeyError: when ``name`` declares no parameter ``key``.
        """
        decision = self.decisions.get(name)
        if decision is not None and key in decision.params:
            return decision.params[key]
        return default_param(name, key)

    def paths(self) -> FrozenSet[str]:
        """The levers on: closed under ``depends_on`` and ``Lever.requires`` by construction."""
        return frozenset(name for name, decision in self.decisions.items() if decision.on)

    def describe(self) -> Tuple[str, ...]:
        """The log lines of the table (without the log tag)."""
        knobs = ", ".join(f"{name}={value}" for name, value in self.knobs.items()) or "none"
        lines = [
            f"levers: {self.deployment.describe()}; default {self.default}; facts: "
            f"{self.deployment.facts.describe()}; knobs set: {knobs}"
        ]
        for name, decision in self.decisions.items():
            lever = LEVERS.get(name)
            kind = lever.kind.value if lever is not None else "composition"
            lines.append(
                f"lever {name} {ON if decision.on else OFF} ({decision.source}: {decision.why})"
                + (f" {dict(decision.params)}" if decision.params else "")
                + f" [{kind}]"
            )
        if self.unserved:
            lines.append(
                f"levers not served at sm{self.deployment.sm} tep{self.deployment.tp}: "
                + ", ".join(name for name in LEVERS if name in self.unserved)
            )
        lines.append(f"lever table json (resolved): {self.as_json()}")
        return tuple(lines)

    def as_json(self, installed: Optional[Mapping[str, bool]] = None) -> str:
        """The table as one JSON line; with ``installed``, each lever's install outcome too.

        A lever without a reported outcome counts as installed when it resolved on.
        """
        levers = {}
        params = {}
        for name, decision in self.decisions.items():
            entry = {"resolved": ON if decision.on else OFF, "source": decision.source}
            if installed is not None:
                entry["installed"] = bool(installed.get(name, decision.on))
            levers[name] = entry
            if decision.params:
                params[name] = {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in decision.params.items()
                }
        table = {
            "deployment": self.deployment.as_dict(),
            "facts": dataclasses.asdict(self.deployment.facts),
            "default": self.default,
            "knobs": dict(self.knobs),
            "params": params,
            "levers": levers,
        }
        return json.dumps(table, sort_keys=True, separators=(",", ":"))


def _order(
    served: FrozenSet[str],
    depends_on: Mapping[str, Sequence[str]],
    requires: Mapping[str, Sequence[str]],
) -> List[str]:
    """``served`` in an order where every lever follows what it builds on, requires or reads.

    Ties keep the registry order of ``depends_on``.
    """
    registry = [name for name in depends_on if name in served]
    edges = {
        name: {
            other
            for other in (
                *depends_on[name],
                *requires.get(name, ()),
                *(LEVERS[name].requires if name in LEVERS else ()),
                *(LEVERS[name].after if name in LEVERS else ()),
            )
            if other in served and other != name
        }
        for name in registry
    }
    ordered: List[str] = []
    done: set = set()
    while len(ordered) < len(registry):
        ready = [name for name in registry if name not in done and edges[name] <= done]
        if not ready:
            raise RuntimeError(
                f"lever dependencies form a cycle among {sorted(set(registry) - done)}"
            )
        ordered.append(ready[0])
        done.add(ready[0])
    return ordered


def resolve(
    deployment: Deployment,
    served: Iterable[str],
    structural: Mapping[str, str],
    snapshot: Mapping[str, str],
    depends_on: Mapping[str, Sequence[str]],
    requires: Optional[Mapping[str, Sequence[str]]] = None,
) -> LeverTable:
    """The lever table of ``deployment`` for the composition's ``served`` levers.

    ``structural`` maps each lever a structural rule excludes to the reason;
    ``depends_on`` is the registry (``dispatch.DEPENDS_ON``: every lever and what it
    builds on); ``requires`` adds the deployment's own requirements
    (``dispatch.structural_rules``). See the module docstring for the order.

    Raises:
        ValueError: on every error of the module docstring, all named in one message.
    """
    knobs = parse_knobs(snapshot, depends_on)
    served = frozenset(served)
    decisions, errors, warnings = _resolve_pass(
        deployment, served, structural, knobs, depends_on, dict(requires or {})
    )
    if errors:
        raise ValueError("; ".join(errors))
    ordered = {name: decisions[name] for name in depends_on if name in decisions}
    return LeverTable(
        deployment=deployment,
        decisions=types.MappingProxyType(ordered),
        unserved=frozenset(LEVERS) - served,
        knobs=types.MappingProxyType(dict(knobs.raw)),
        default=knobs.default,
        warnings=tuple(warnings),
    )


def _resolve_pass(
    deployment: Deployment,
    served: FrozenSet[str],
    structural: Mapping[str, str],
    knobs: Knobs,
    depends_on: Mapping[str, Sequence[str]],
    requires: Mapping[str, Sequence[str]],
) -> Tuple[Dict[str, Decision], List[str], List[str]]:
    """Every served lever's decision, the errors and the warnings."""
    requires = dict(requires or {})
    served = frozenset(served)
    knob_on = frozenset(name for name, mode in knobs.modes.items() if mode == ON)
    needed = _built_on(knob_on, depends_on, requires)
    d = deployment
    errors: List[str] = []
    warnings: List[str] = []
    transfers_kv = d.role in (Role.GEN, Role.CTX) or (
        d.role is Role.UNKNOWN and d.role_source == DISAGGREGATED_UNHINTED
    )
    if d.facts.ucx_rndv_thresh_set and transfers_kv:
        errors.append(
            f"{UCX_RNDV_THRESH_ENV} is set on a {d.role.value} server with a cache transceiver: "
            "it would move this process's KV-transfer rendezvous too"
        )
    if d.role is Role.UNKNOWN:
        warnings.append(
            f"role unknown ({d.role_source}): counted as a decoding server; set {ROLE_KNOB}"
        )
    decisions: Dict[str, Decision] = {}
    t = Resolving(served, decisions)
    for name in _order(served, depends_on, requires):
        lever = LEVERS.get(name)
        mode = knobs.modes.get(name)
        params = {
            param.key: knobs.params.get((name, param.key), param.default_of(d))
            for param in (lever.params if lever is not None else ())
        }
        t.pending = (name, params)
        decision = _decide(
            name,
            lever,
            mode,
            knobs.default,
            params,
            d,
            t,
            structural,
            depends_on,
            requires,
            needed,
            knob_on,
        )
        if mode == ON and not decision.on:
            errors.append(f"{knob_name(name)}=on, but {decision.why}")
        if (
            mode == ON
            and decision.on
            and lever is not None
            and lever.kind is Kind.NUMERIC
            and not any(gate.holds(d) for gate in lever.gates)
        ):
            warnings.append(
                f"{knob_name(name)}=on outside its accuracy gates ("
                + "; ".join(gate.scope for gate in lever.gates)
                + ")"
            )
        decisions[name] = decision
    for name, decision in decisions.items():
        rule = LEVERS.get(name)
        if not decision.on or rule is None:
            continue
        for other in rule.conflicts:
            if other in decisions and decisions[other].on:
                errors.append(
                    f"{name} ({decision.source}) and {other} ({decisions[other].source}) cannot "
                    f"both be on; set {knob_name(other)}=off or {knob_name(name)}=off"
                )
    # The MXFP8 copy takes any call of MIN_ROWS rows or more, context rows or not.
    fp8 = decisions.get("context_fp8_dense")
    if fp8 is not None and fp8.on and d.decodes() and d.reaches(int(fp8.params["MIN_ROWS"])):
        errors.append(
            f"{knob_name('context_fp8_dense')}=on, but decode steps of {fp8.params['MIN_ROWS']} "
            f"rows or more are reachable at max_batch_size {_count(d.max_batch_size)}, and the "
            f"MXFP8 copy would take them too; set {knob_name('context_fp8_dense', 'MIN_ROWS')} "
            f"above max_batch_size * {d.rows_per_request}"
        )
    for name, mode in knobs.modes.items():
        if name in served:
            continue
        if mode == ON:
            errors.append(
                f"{knob_name(name)}=on, but the composition at sm{d.sm} tep{d.tp} does not "
                f"serve {name}"
            )
        else:
            warnings.append(f"{knob_name(name)}={mode} has no effect: {name} is not served here")
    for name, key in knobs.params:
        if name not in served:
            warnings.append(f"{knob_name(name, key)} has no effect: {name} is not served here")
    return decisions, errors, warnings


def _built_on(
    names: Iterable[str],
    depends_on: Mapping[str, Sequence[str]],
    requires: Mapping[str, Sequence[str]],
) -> FrozenSet[str]:
    """Every lever ``names`` build on or require, transitively (``names`` themselves excluded)."""
    found: set = set()
    stack = list(names)
    while stack:
        name = stack.pop()
        for other in (
            *depends_on.get(name, ()),
            *requires.get(name, ()),
            *(LEVERS[name].requires if name in LEVERS else ()),
        ):
            if other not in found:
                found.add(other)
                stack.append(other)
    return frozenset(found)


def _decide(
    name: str,
    lever: Optional[Lever],
    mode: Optional[str],
    default: str,
    params: Mapping[str, object],
    d: Deployment,
    t: Resolving,
    structural: Mapping[str, str],
    depends_on: Mapping[str, Sequence[str]],
    requires: Mapping[str, Sequence[str]],
    needed: FrozenSet[str] = frozenset(),
    knob_on: FrozenSet[str] = frozenset(),
) -> Decision:
    """One lever's decision, from the decisions already made (``resolve``'s order).

    ``needed`` are the levers a lever turned on by its knob builds on or requires: they come on
    with it (where they can run). A lever ``knob_on`` turns on that it conflicts with (either
    row's ``conflicts``) stays off by default: the knob's lever takes those steps.
    """
    if name in structural:
        return Decision(name, False, "structural", structural[name], params)
    for other in depends_on[name]:
        if not t.on(other):
            return Decision(
                name, False, "dependency", f"it builds on {other}, which is off", params
            )
    for other in (*requires.get(name, ()), *(lever.requires if lever is not None else ())):
        if not t.on(other):
            return Decision(name, False, "dependency", f"it requires {other}, which is off", params)
    if lever is not None and lever.excluded is not None:
        reason = lever.excluded(d, t)
        if reason is not None:
            return Decision(name, False, "structural", reason, params)
    if mode == OFF:
        return Decision(name, False, "knob", f"{knob_name(name)}=off", params)
    if mode == ON:
        return Decision(name, True, "knob", f"{knob_name(name)}=on", params)
    if name in needed:
        return Decision(name, True, "knob", "a lever turned on by its knob needs it", params)
    if lever is None:
        raise RuntimeError(f"{name} is served but has no row in LEVERS")
    if default == OFF:
        if lever.stock is None:
            return Decision(name, False, "stock", f"{DEFAULT_KNOB}=off", params)
        on, why = lever.stock(d, t, params)
        return Decision(name, on, "stock", f"{DEFAULT_KNOB}=off, its upstream rule: {why}", params)
    for other in sorted(_conflict_partners(name) & knob_on):
        return Decision(
            name, False, "default", f"steps aside for {other}, turned on by its knob", params
        )
    on, why = lever.on_by_default(d, t, params)
    return Decision(name, on, "default", why, params)


def _conflict_partners(name: str) -> FrozenSet[str]:
    """The levers ``name`` may not be on with: its row's ``conflicts`` and the rows naming it."""
    row = LEVERS.get(name)
    mine = set(row.conflicts) if row is not None else set()
    return frozenset(mine | {other for other, lever in LEVERS.items() if name in lever.conflicts})
