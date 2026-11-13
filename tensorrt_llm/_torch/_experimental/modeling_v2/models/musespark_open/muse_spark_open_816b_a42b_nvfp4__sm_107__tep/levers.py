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

    MUSESPARK_OPEN_LEVER_<LEVER>=auto|on|off          one per lever; unset or empty: the default
    MUSESPARK_OPEN_LEVER_<LEVER>__<PARAM>=<value>      a parameter the lever declares (Rule.params)
    MUSESPARK_OPEN_LEVER__DEFAULT=auto|baseline        what an unset lever knob means
    MUSESPARK_OPEN_LEVER__ROLE=auto|agg|ctx|gen        the server's role; unset or auto: derived

``<LEVER>`` is the lever's name upper-cased; ``__`` introduces a parameter or a resolver
setting and never occurs inside a lever name. Values are case-insensitive and
whitespace-stripped. Two shared-path variables are read as facts and never set or retired
here: ``TRTLLM_KV_DEFER_CLOSE`` (on iff exactly ``1``, the KV cache manager's own test) and
``UCX_RNDV_THRESH`` (set or not).

Semantics, in the order a lever is resolved:

* structural: a fact the lever's code needs (an arena kind, a fact variable, the drafter's
  logit transform, rejection sampling; ``dispatch.structural_rules``) excludes it; nothing
  overrides that;
* dependency: a lever it builds on (``dispatch.FastPath.depends_on``) or requires
  (``Rule.requires``) is off;
* knob: ``off`` removes it, and with it every lever built on it or requiring it; ``on``
  keeps it whatever its deployment rule and gates say;
* band: under ``auto``, a lever of the other band's selection (below) resolves as under
  ``baseline``, unless a lever turned ``on`` by its knob builds on it or requires it;
* default: ``auto`` runs the lever's deployment rule (``Rule.auto``), and a lossless-numeric
  lever also needs one of its accuracy gates (``Rule.gates``); ``baseline`` turns every
  lever off that is not the upstream target's, and gives the upstream ones their upstream
  rule (``Rule.baseline``).

Bands. Every deployment belongs to exactly one band, by ``band_of``: an aggregated server
(no cache transceiver, or the ``agg`` role hint), at any TEP size, ``max_batch_size`` and
arena, to ``Band.AGGREGATED_SM107``; every other deployment (a context, generation or
unhinted disaggregated server) to ``Band.DEFAULT``.
Each band owns a selection of levers (``BAND_LEVERS``): the aggregated sm107 band the
high-concurrency TEP4, the 2..4-request TEP4 and the single-request TEP8 levers; the default
band the low-latency levers of the sm107 compositions and the upstream rules they changed. A
lever in neither selection is both bands' (``full_attention_fold_multi``'s rule names the band
itself, ``_full_attention_fold_multi``). Under ``auto`` a lever of the
other band's selection resolves as under ``baseline`` (off, or its upstream rule), so each
band's table is its own selection; its knob still turns it on where its structural and
dependency rules allow, and the levers it builds on or requires then resolve by their own
rules.

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
#: What an unset lever knob means: ``auto`` (the default) or ``baseline``.
DEFAULT_KNOB = PREFIX + "_DEFAULT"
#: The server's role: ``auto`` (derived, the default), ``agg``, ``ctx`` or ``gen``.
ROLE_KNOB = PREFIX + "_ROLE"
#: The shared-path facts the resolver reads.
KV_DEFER_CLOSE_ENV = "TRTLLM_KV_DEFER_CLOSE"
UCX_RNDV_THRESH_ENV = "UCX_RNDV_THRESH"

AUTO = "auto"
ON = "on"
OFF = "off"
BASELINE = "baseline"
LEVER_VALUES = (AUTO, ON, OFF)
DEFAULT_VALUES = (AUTO, BASELINE)

#: The decision sources, in the order a lever is resolved.
SOURCES = ("structural", "dependency", "knob", "baseline", "band", "auto", "gate")

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
        "MUSESPARK_OPEN_SWA_PREP_SPLIT": PREFIX + "SWA_PREP_SPLIT",
        "MUSESPARK_OPEN_SWA_ONE2": PREFIX + "SWA_EIGHT_REQUESTS",
        "MUSESPARK_OPEN_FA_PREP_FOLD_MULTI": PREFIX + "FULL_ATTENTION_FOLD_MULTI",
        "MUSESPARK_OPEN_FA_STAGES": PREFIX + "SPLIT_KV_DEEP_RINGS",
        "MUSESPARK_OPEN_MR_STAGES": PREFIX + "SPLIT_KV_DEEP_RINGS",
        "MUSESPARK_OPEN_BF16_HALF_WINDOW": PREFIX + "BF16_HALF_WINDOW",
        "MUSESPARK_OPEN_PREROUTED_META": PREFIX + "PREROUTED_EXPERTS",
        "MUSESPARK_OPEN_TEP8_PREROUTED": PREFIX + "PREROUTED_EXPERTS",
        "MUSESPARK_OPEN_PREROUTED_SO": "nothing: the experts run trtllm::fp4_block_scale_moe_prerouted",
        "MUSESPARK_OPEN_ROUTER_ROWS_TILING": "nothing: the sm107 compositions tile the per-row router shard",
        "MUSESPARK_OPEN_TEP8_RING_BUDDY_MAX_ROWS": PREFIX + "TEP8_RING_BUDDY_WIDE",
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
        "MUSESPARK_OPEN_TEP8_FINALIZE_TRANSPORT_TOKENS": PREFIX + "TEP8_FINALIZE_TRANSPORT_64",
        "MUSESPARK_OPEN_HOP_ENTRY_TRIGGER": PREFIX + "HOP_ENTRY_TRIGGER",
        "MUSESPARK_OPEN_TEP8_HOP_DIRECT": PREFIX + "TEP8_DIRECT_HOP",
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

    An integer in ``low..high``; with ``choices``, a non-empty subset of ``choices``
    (comma-separated, kept in ``choices`` order); with ``ranges``, a non-empty list of
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

    def default_of(self, deployment: "Deployment") -> object:
        return self.default if self.default_for is None else self.default_for(deployment)

    def parse(self, raw: str) -> object:
        """The value ``raw`` gives this parameter.

        Raises:
            ValueError: for a value outside the parameter's range or choices.
        """
        text = raw.strip().lower()
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
class Rule:
    """How one lever resolves when its rule is not the default (served, on).

    ``origin`` is ``upstream`` for the upstream target's own paths and their upstream
    behaviour, else the serving region the lever targets (``HC`` high concurrency,
    ``midc`` mid concurrency, ``LL`` low latency): under ``__DEFAULT=baseline`` a
    non-upstream lever is off unless its knob is set, and an upstream one runs
    ``baseline`` (or ``auto`` when it has none). ``after`` names the decisions ``auto`` /
    ``baseline`` read; ``requires`` levers that must be on;
    ``conflicts`` levers that may not be on with it; ``excluded`` a structural reason it
    cannot be on, from the decisions so far.
    """

    origin: str
    kind: Kind
    auto: RuleFn
    baseline: Optional[RuleFn] = None
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
    """The decisions made so far, as a rule reads them (``Rule.after``)."""

    def __init__(self, served: FrozenSet[str], decisions: Mapping[str, Decision]) -> None:
        self._served = served
        self._decisions = decisions

    def on(self, name: str) -> bool:
        """Whether ``name`` resolved on; False for a lever the composition does not serve.

        Raises:
            RuntimeError: for a served lever not decided yet (a rule that reads it must
                name it in ``Rule.after``).
        """
        if name not in self._served:
            return False
        decision = self._decisions.get(name)
        if decision is None:
            raise RuntimeError(f"lever rule reads {name}, which is not decided yet")
        return decision.on

    def param(self, name: str, key: str) -> object:
        """``name``'s parameter ``key`` as resolved, or its default when ``name`` is not served."""
        decision = self._decisions.get(name)
        if decision is not None and key in decision.params:
            return decision.params[key]
        return RULES[name].param(key).default


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


def _ll_params() -> Tuple[Param, ...]:
    return (Param("MAX_BATCH", LL_MAX_BATCH, 1, 64),)


def _ll_gate() -> Gate:
    """Where the numeric low-latency levers passed their accuracy gates."""

    def holds(d: Deployment) -> bool:
        return (
            d.tp in (4, 8)
            and d.role in (Role.GEN, Role.AGG)
            and d.max_batch_size is not None
            and d.max_batch_size <= LL_MAX_BATCH
        )

    return Gate(
        holds,
        f"sm107 tep4 and tep8 servers of max_batch_size up to {LL_MAX_BATCH}: greedy drift, "
        "GSM8K and SPEED-Bench acceptance",
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


def _ll(kind: Kind, yields_to: Tuple[str, ...] = (), exclusive: bool = False) -> Rule:
    """A low-latency lever of decode steps: on while ``max_batch_size`` is at most MAX_BATCH.

    With ``yields_to`` it is off while one of those levers is on (``_yielding``); with
    ``exclusive`` it also may not be on with one of them (it would take their steps), so
    turning it on by its knob next to one of them raises.
    """
    gates = (_ll_gate(),) if kind is Kind.NUMERIC else ()
    rule = _at_most("MAX_BATCH")
    if yields_to:
        rule = _yielding(yields_to, rule)
    return Rule(
        "LL",
        kind,
        rule,
        params=_ll_params(),
        gates=gates,
        after=yields_to,
        conflicts=yields_to if exclusive else (),
    )


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


def _tep4_path_rule() -> Rule:
    return Rule(
        "upstream",
        Kind.COMPOSITION,
        _always("its rows, wherever served"),
        baseline=_not_on_tep8(
            "the upstream sm107 tep8 composition does not serve it", "its rows, wherever served"
        ),
    )


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


def _decode_qkv_excluded(d: Deployment, t: Resolving) -> Optional[str]:
    if "qkv" not in t.param("context_fp8_dense", "PROJECTIONS"):
        return "it reads context_fp8_dense's QKV copies, which context_fp8_dense__PROJECTIONS leaves out"
    return None


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
RULES: Mapping[str, Rule] = types.MappingProxyType(
    {
        "packed_prep_append": Rule(
            "HC",
            Kind.BITWISE,
            _packed_prep_append,
            params=(Param("MIN_ROWS", 33, 8, 1024),),
        ),
        "windowed_commit_parallel": Rule(
            "HC",
            Kind.BITWISE,
            _windowed_commit_parallel,
            params=(
                Param("MIN_REQUESTS", 5, 1, 64, default_for=lambda d: 1 if _hc_selection(d) else 5),
            ),
        ),
        "context_fp8_dense": Rule(
            "HC",
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
        "context_fp8_attn_gather": Rule(
            "HC", Kind.LOSSY, _lossy, after=("context_fp8_dense",), excluded=_attn_gather_excluded
        ),
        "wide_direct_transports": Rule("midc", Kind.BITWISE, _off_at_hc(_wide_steps)),
        "wide_router_per_row": Rule(
            "midc",
            Kind.NUMERIC,
            _wide_steps,
            gates=(_router_gate(Role.GEN, "generation servers at tep4 with max_batch_size 2..4"),),
        ),
        "wide_oproj_publish": Rule("midc", Kind.BITWISE, _off_at_hc(_wide_steps)),
        "wide_finalize_transport": Rule(
            "midc",
            Kind.BITWISE,
            _off_at_hc(_always("17..32-token finalize calls on a 32-token transport")),
        ),
        "lm_head_inbox": Rule(
            "upstream",
            Kind.COMPOSITION,
            _never("the sm107 compositions gather the 8-row head without the inbox"),
            baseline=_always("8-row heads"),
        ),
        "graph_start_qkv_prefetch": Rule(
            "upstream",
            Kind.COMPOSITION,
            _graph_start_prefetch,
            baseline=_always("graph starts of split-K row counts"),
        ),
        "step_loop_on_execution_stream": Rule(
            "LL",
            Kind.BITWISE,
            _at_most("MAX_BATCH"),
            params=(Param("MAX_BATCH", 4, 1, 64),),
        ),
        "incremental_block_offsets": Rule(
            "LL",
            Kind.BITWISE,
            _at_most("MAX_BATCH"),
            params=(Param("MAX_BATCH", 8, 1, 64),),
        ),
        "drafter_kv_side_chain": Rule(
            "LL",
            Kind.BITWISE,
            _always("decode steps of 1..64 requests over a dense arena or pages"),
        ),
        "drafter_fc_early_fork": Rule("LL", Kind.BITWISE, _always("forked side-chain steps")),
        "drafter_preamble_from_commit": Rule(
            "LL", Kind.BITWISE, _always("folded steps over a dense arena or pages")
        ),
        "head_sampler_coop_fold": Rule(
            "LL",
            Kind.BITWISE,
            _off_at_hc(_coop_fold),
            params=(Param("MAX_ROWS", 16, 1, 128),),
        ),
        "head_multimem_gather": Rule(
            "LL",
            Kind.FLAGGED,
            _flagged,
            requires=("step_loop_on_execution_stream",),
            conflicts=("lm_head_inbox",),
        ),
        "half_window_experts": Rule(
            "upstream",
            Kind.COMPOSITION,
            _at_most("MAX_BATCH"),
            baseline=_at_most("MAX_BATCH", requires_decode=False),
            params=(Param("MAX_BATCH", 8, 1, 64),),
        ),
        "hot_replicas": Rule(
            "upstream",
            Kind.COMPOSITION,
            _hot_replicas(requires_decode=True),
            baseline=_hot_replicas(requires_decode=False),
            after=("half_window_experts",),
            conflicts=("half_window_experts",),
        ),
        "prefill_cutedsl_experts": Rule(
            "upstream", Kind.COMPOSITION, _always("expert calls of 4096..8192 rows")
        ),
        "vocab_parallel_head": Rule(
            "upstream",
            Kind.COMPOSITION,
            _vocab_parallel_head,
            baseline=_never("the upstream sm107 compositions sample on the split sampler"),
            params=(
                Param("MAX_BATCH", LL_MAX_BATCH, 1, 64),
                Param("MIN_ROWS", 8, 8, 16),
            ),
            after=("distributed_head_sampler",),
            conflicts=("distributed_head_sampler",),
        ),
        **{name: _tep4_path_rule() for name in SM107_TEP8_TEP4_PATHS},
        # The Vera Rubin low-latency levers.
        "splitk_pool_split8": _ll(Kind.NUMERIC),
        "splitk_frontier_tables": _ll(Kind.NUMERIC, yields_to=("wide_qkv_prefetch",)),
        "tep8_wide64_splitk": _ll(Kind.NUMERIC),
        "tep8_publish_splitk": _ll(Kind.NUMERIC),
        "tep8_drafter_wide_gemms": _ll(Kind.NUMERIC),
        "tep8_fc_shard_wide": _ll(Kind.NUMERIC),
        "weight_prefetch_after_join": _ll(
            Kind.BITWISE, yields_to=("wide_qkv_prefetch", "layer0_qkv_table")
        ),
        "next_qkv_prefetch_skip": _ll(Kind.BITWISE),
        "swa_one_cluster_block": _ll(
            Kind.NUMERIC, yields_to=("swa_blocks_per_program",), exclusive=True
        ),
        "swa_prep_split": _ll(Kind.BITWISE, yields_to=("swa_blocks_per_program",), exclusive=True),
        "swa_eight_requests": _ll(
            Kind.NUMERIC, yields_to=("swa_blocks_per_program",), exclusive=True
        ),
        "full_attention_fold_multi": Rule(
            "LL",
            Kind.BITWISE,
            _full_attention_fold_multi,
            params=_ll_params(),
            after=("wide_full_attention",),
        ),
        "split_kv_deep_rings": _ll(Kind.BITWISE),
        "bf16_half_window": _ll(Kind.NUMERIC),
        "prerouted_experts": _ll(Kind.BITWISE),
        "tep8_ring_buddy_wide": _ll(Kind.NUMERIC),
        "tep8_ring_buddy_pdl": _ll(Kind.BITWISE),
        "tep8_tile_placement": _ll(Kind.NUMERIC),
        "tep8_router_push": _ll(Kind.NUMERIC),
        "tep8_router_prewait": _ll(Kind.BITWISE),
        "finalize_entry_trigger": _ll(Kind.BITWISE),
        "tep8_finalize_transport_64": _ll(Kind.BITWISE),
        "hop_entry_trigger": _ll(Kind.BITWISE),
        "tep8_early_multicast_hop": _ll(Kind.BITWISE),
        "tep8_direct_hop": _ll(Kind.BITWISE),
        "tep8_publish_join": _ll(Kind.BITWISE),
        "tep8_pool_programmatic_fork": _ll(Kind.BITWISE),
        "tep8_wide_projection_gathers": _ll(Kind.NUMERIC),
        "step_commit_v2": _ll(Kind.BITWISE, yields_to=("step_loop_inline_commits",)),
        "wide_norm_gate_pdl": _ll(Kind.BITWISE),
        "drafter_kv_glue": Rule(
            "LL",
            Kind.BITWISE,
            _drafter_kv_glue,
            params=_ll_params(),
            after=("drafter_kv_side_chain", "context_kv_late_join"),
            conflicts=("drafter_kv_side_chain",),
        ),
        "drafter_kv_fork_before_head": _ll(
            Kind.BITWISE, yields_to=("context_kv_late_join", "context_kv_fork_last_layer")
        ),
        "context_sp_extra_parts": Rule(
            "LL",
            Kind.BITWISE,
            _context_workers,
            params=(
                Param("MAX_BATCH", LL_MAX_BATCH, 1, 64),
                Param(
                    "PARTS",
                    ("router_input", "ce_gather"),
                    choices=("shard_routing", "router_input", "ce_gather"),
                ),
            ),
        ),
        "step_window_compact": Rule(
            "HC", Kind.BITWISE, _on_agg_tep4("the windowed arena compaction of mixed steps")
        ),
        "fmk_context_fmha": Rule(
            "HC",
            Kind.NUMERIC,
            _on_agg_tep4("the context rows of mixed steps"),
            gates=(_AGG_TEP4_GATE,),
        ),
        "context_kv_late_join": Rule(
            "HC", Kind.BITWISE, _on_agg_tep4("windowed block decodes after a forked commit")
        ),
        "hop_cluster_split": Rule(
            "HC",
            Kind.BITWISE,
            _on_agg_tep4("wide post-attention hops of the ROWS row counts"),
            params=(
                Param("CLUSTER", 2, 1, 2),
                Param("ROWS", ((192, 208), (248, 264)), 9, 1024, ranges=True),
            ),
        ),
        "drafter_fc_inbox_wide": Rule(
            "HC",
            Kind.BITWISE,
            _on_agg_tep4("drafter fc gathers of up to MAX_ROWS rows", min_rows=129),
            params=(Param("MAX_ROWS", 384, 128, 512), Param("PROGRAMS", 128, 1, 256)),
        ),
        "l061_tile32_pin": Rule(
            "HC",
            Kind.NUMERIC,
            _on_agg_tep4("layer 0 / 61 owner-window expert calls of 256..384 rows"),
            gates=(_AGG_TEP4_GATE,),
        ),
        "router_staged_epilogue": Rule(
            "HC", Kind.BITWISE, _on_agg_tep4("router GEMM calls of 9..512 rows")
        ),
        "split_kv_deep_ring": Rule(
            "HC", Kind.BITWISE, _on_agg_tep4("a 20-page ring in the E4M3-query split-KV decode")
        ),
        "context_prep_rows_vec": Rule(
            "HC",
            Kind.BITWISE,
            _on_agg_tep4("context steps' prep on the vectorized rows kernel"),
        ),
        "context_embedding_replicated": Rule(
            "HC",
            Kind.BITWISE,
            _on_agg_tep4("context steps' token embedding from the replicated table"),
        ),
        "context_sp_router_input": Rule(
            "HC",
            Kind.BITWISE,
            _on_agg_tep4("the router's FP32 input from the sequence-parallel residual join"),
        ),
        "decode_qkv_mxfp8": Rule(
            "HC", Kind.LOSSY, _lossy, after=("context_fp8_dense",), excluded=_decode_qkv_excluded
        ),
        "swa_blocks_per_program": Rule(
            "midc",
            Kind.BITWISE,
            _two_to_four_request_steps("sliding-window split programs over 2..4 key blocks"),
        ),
        "wide_qkv_prefetch": Rule(
            "midc",
            Kind.BITWISE,
            _two_to_four_request_steps("the 9..32-row next-QKV prefetch after the experts"),
        ),
        "wide_capture_fold": Rule(
            "midc",
            Kind.BITWISE,
            _two_to_four_request_steps("the wide direct boundary's DFlash capture"),
        ),
        "megakernel": Rule(
            "LL",
            Kind.BITWISE,
            _on_single_request("one-sequence 8-row decode steps"),
            requires=MEGAKERNEL_NUMERICS,
        ),
        "megakernel_one_exchange_boundaries": Rule(
            "LL",
            Kind.NUMERIC,
            _on_single_request("the megakernel's MoE boundaries"),
            gates=(_SINGLE_REQUEST_GATE,),
        ),
        "megakernel_two_term_pool": Rule(
            "LL",
            Kind.NUMERIC,
            _on_single_request("the megakernel's pool and router GEMVs"),
            gates=(_SINGLE_REQUEST_GATE,),
            requires=("megakernel_one_exchange_boundaries",),
        ),
        "megakernel_router_by_linearity": Rule(
            "LL",
            Kind.NUMERIC,
            _on_single_request("the megakernel's router logits"),
            gates=(_SINGLE_REQUEST_GATE,),
        ),
        "megakernel_zero_exchange_boundary": Rule(
            "LL",
            Kind.BITWISE,
            _always("the megakernel's post-attention boundaries"),
            requires=("megakernel_one_exchange_boundaries",),
            excluded=_megakernel_variant_excluded,
        ),
        "layer0_qkv_table": Rule(
            "LL", Kind.BITWISE, _on_single_request("8-row decode steps' layer-0 QKV rows")
        ),
        "edge_bf16_moe_front": Rule(
            "LL", Kind.BITWISE, _on_single_request("the BF16 expert layers' 8-row MoE fronts")
        ),
        "last_layer_boundary": Rule(
            "LL", Kind.BITWISE, _on_single_request("the last layer's 1..8-row boundaries")
        ),
        "context_kv_fork_last_layer": Rule(
            "LL", Kind.BITWISE, _on_single_request("forked side-chain steps")
        ),
        # Its LM-head GEMM runs at one fixed split-K tactic on every rank, which can differ from
        # the autotuner's per-rank choice for the stock 8-row head it replaces.
        "distributed_head_sampler": Rule(
            "LL",
            Kind.NUMERIC,
            _on_single_request("8-row verify heads"),
            gates=(_SINGLE_REQUEST_GATE,),
            requires=("step_loop_on_execution_stream",),
            conflicts=("lm_head_inbox",),
        ),
        "drafter_preamble_late_join": Rule(
            "LL", Kind.BITWISE, _on_single_request("folded steps with a commit-written preamble")
        ),
        "drafter_folds": Rule(
            "LL", Kind.BITWISE, _on_single_request("one-request 8-row drafter block decodes")
        ),
        "draft_head_tail": Rule(
            "LL", Kind.BITWISE, _on_single_request("one-request 7-position draft steps")
        ),
        "step_loop_inline_commits": Rule(
            "LL", Kind.BITWISE, _on_single_request("the one-stream step loop's commits")
        ),
        "rejection_sampler": Rule(
            "LL",
            Kind.COMPOSITION,
            _on_single_request("rejection-sampled one-request decode steps"),
        ),
    }
)


#: Each band's lever selection (module docstring): under ``auto`` a lever listed for one
#: band resolves as under ``baseline`` in the other. ``full_attention_fold_multi`` is in
#: neither: its rule serves each band's steps (``_full_attention_fold_multi``).
BAND_LEVERS: Mapping[Band, FrozenSet[str]] = types.MappingProxyType(
    {
        Band.AGGREGATED_SM107: frozenset(
            {
                # High-concurrency TEP4.
                "step_window_compact",
                "fmk_context_fmha",
                "context_kv_late_join",
                "hop_cluster_split",
                "drafter_fc_inbox_wide",
                "l061_tile32_pin",
                "router_staged_epilogue",
                "split_kv_deep_ring",
                "context_prep_rows_vec",
                "context_embedding_replicated",
                "context_sp_router_input",
                "decode_qkv_mxfp8",
                # 2..4-request TEP4 decode steps.
                "swa_blocks_per_program",
                "wide_qkv_prefetch",
                "wide_capture_fold",
                # The single-request TEP8 decode schedule.
                "megakernel",
                "megakernel_one_exchange_boundaries",
                "megakernel_two_term_pool",
                "megakernel_router_by_linearity",
                "megakernel_zero_exchange_boundary",
                "layer0_qkv_table",
                "edge_bf16_moe_front",
                "last_layer_boundary",
                "context_kv_fork_last_layer",
                "distributed_head_sampler",
                "drafter_preamble_late_join",
                "drafter_folds",
                "draft_head_tail",
                "step_loop_inline_commits",
                "rejection_sampler",
            }
        ),
        Band.DEFAULT: frozenset(
            {
                # The upstream rules the sm107 compositions changed: under baseline (and in
                # the aggregated sm107 band) their upstream sm107 rule.
                "vocab_parallel_head",
                *SM107_TEP8_TEP4_PATHS,
                # The Vera Rubin low-latency levers.
                "splitk_pool_split8",
                "splitk_frontier_tables",
                "tep8_wide64_splitk",
                "tep8_publish_splitk",
                "tep8_drafter_wide_gemms",
                "tep8_fc_shard_wide",
                "weight_prefetch_after_join",
                "next_qkv_prefetch_skip",
                "swa_one_cluster_block",
                "swa_prep_split",
                "swa_eight_requests",
                "split_kv_deep_rings",
                "bf16_half_window",
                "prerouted_experts",
                "tep8_ring_buddy_wide",
                "tep8_ring_buddy_pdl",
                "tep8_tile_placement",
                "tep8_router_push",
                "tep8_router_prewait",
                "finalize_entry_trigger",
                "tep8_finalize_transport_64",
                "hop_entry_trigger",
                "tep8_early_multicast_hop",
                "tep8_direct_hop",
                "tep8_publish_join",
                "tep8_pool_programmatic_fork",
                "tep8_wide_projection_gathers",
                "step_commit_v2",
                "wide_norm_gate_pdl",
                "drafter_kv_glue",
                "drafter_kv_fork_before_head",
                "context_sp_extra_parts",
            }
        ),
    }
)


def lever_band(name: str) -> Optional[Band]:
    """The band whose selection lever ``name`` belongs to, or None for a lever of both."""
    for band, names in BAND_LEVERS.items():
        if name in names:
            return band
    return None


def default_param(name: str, key: str) -> object:
    """Lever ``name``'s declared default for parameter ``key``.

    Raises:
        KeyError: when ``name`` declares no parameter ``key``.
    """
    rule = RULES.get(name)
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
    default, role = AUTO, AUTO
    raw: Dict[str, str] = {}
    for name in sorted(snapshot):
        if not name.startswith(PREFIX):
            continue
        value = snapshot[name].strip().lower()
        raw[name] = snapshot[name].strip()
        rest = name[len(PREFIX) :]
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
        rule = RULES.get(lever)
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


def role_of(snapshot: Mapping[str, str], disaggregated: Optional[bool]) -> Tuple[Role, str]:
    """The server's role and where it came from.

    The hint ``ROLE_KNOB`` wins when set. Otherwise ``agg`` without a cache transceiver,
    and ``unknown`` with one or without engine facts.

    Raises:
        ValueError: for a hint the server contradicts: ``agg`` with a cache transceiver,
            ``ctx`` or ``gen`` without one.
    """
    hint = snapshot.get(ROLE_KNOB, AUTO).strip().lower() or AUTO
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
    default: str = AUTO
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
        """The levers on: closed under ``depends_on`` and ``Rule.requires`` by construction."""
        return frozenset(name for name, decision in self.decisions.items() if decision.on)

    def describe(self) -> Tuple[str, ...]:
        """The log lines of the table (without the log tag)."""
        knobs = ", ".join(f"{name}={value}" for name, value in self.knobs.items()) or "none"
        lines = [
            f"levers: {self.deployment.describe()}; default {self.default}; facts: "
            f"{self.deployment.facts.describe()}; knobs set: {knobs}"
        ]
        for name, decision in self.decisions.items():
            rule = RULES.get(name)
            if rule is None and decision.on:
                continue
            kind, origin = (rule.kind.value, rule.origin) if rule else ("composition", "upstream")
            lines.append(
                f"lever {name} {ON if decision.on else OFF} ({decision.source}: {decision.why})"
                + (f" {dict(decision.params)}" if decision.params else "")
                + f" [{kind}, {origin}]"
            )
        if self.unserved:
            lines.append(
                f"levers not served at sm{self.deployment.sm} tep{self.deployment.tp}: "
                + ", ".join(name for name in RULES if name in self.unserved)
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
                *(RULES[name].requires if name in RULES else ()),
                *(RULES[name].after if name in RULES else ()),
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
    requires = dict(requires or {})
    served = frozenset(served)
    needed = _built_on(
        {name for name, mode in knobs.modes.items() if mode == ON}, depends_on, requires
    )
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
        rule = RULES.get(name)
        mode = knobs.modes.get(name)
        params = {
            param.key: knobs.params.get((name, param.key), param.default_of(d))
            for param in (rule.params if rule is not None else ())
        }
        decision = _decide(
            name, rule, mode, knobs.default, params, d, t, structural, depends_on, requires, needed
        )
        if mode == ON and not decision.on:
            errors.append(f"{knob_name(name)}=on, but {decision.why}")
        if (
            mode == ON
            and decision.on
            and rule is not None
            and rule.kind is Kind.NUMERIC
            and not any(gate.holds(d) for gate in rule.gates)
        ):
            warnings.append(
                f"{knob_name(name)}=on outside its accuracy gates ("
                + "; ".join(gate.scope for gate in rule.gates)
                + ")"
            )
        decisions[name] = decision
    for name, decision in decisions.items():
        rule = RULES.get(name)
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
    if errors:
        raise ValueError("; ".join(errors))
    ordered = {name: decisions[name] for name in depends_on if name in decisions}
    return LeverTable(
        deployment=d,
        decisions=types.MappingProxyType(ordered),
        unserved=frozenset(RULES) - served,
        knobs=types.MappingProxyType(dict(knobs.raw)),
        default=knobs.default,
        warnings=tuple(warnings),
    )


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
            *(RULES[name].requires if name in RULES else ()),
        ):
            if other not in found:
                found.add(other)
                stack.append(other)
    return frozenset(found)


def _decide(
    name: str,
    rule: Optional[Rule],
    mode: Optional[str],
    default: str,
    params: Mapping[str, object],
    d: Deployment,
    t: Resolving,
    structural: Mapping[str, str],
    depends_on: Mapping[str, Sequence[str]],
    requires: Mapping[str, Sequence[str]],
    needed: FrozenSet[str] = frozenset(),
) -> Decision:
    """One lever's decision, from the decisions already made (``resolve``'s order).

    ``needed`` are the levers a lever turned on by its knob builds on or requires: they
    resolve by their own rule in either band, so the band never blocks such a knob.
    """
    if name in structural:
        return Decision(name, False, "structural", structural[name], params)
    for other in depends_on[name]:
        if not t.on(other):
            return Decision(
                name, False, "dependency", f"it builds on {other}, which is off", params
            )
    for other in (*requires.get(name, ()), *(rule.requires if rule is not None else ())):
        if not t.on(other):
            return Decision(name, False, "dependency", f"it requires {other}, which is off", params)
    if rule is not None and rule.excluded is not None:
        reason = rule.excluded(d, t)
        if reason is not None:
            return Decision(name, False, "structural", reason, params)
    if mode == OFF:
        return Decision(name, False, "knob", f"{knob_name(name)}=off", params)
    if mode == ON:
        return Decision(name, True, "knob", f"{knob_name(name)}=on", params)
    baseline = mode is None and default == BASELINE
    if rule is None:
        return Decision(name, True, BASELINE if baseline else AUTO, "served", params)
    owner, band = lever_band(name), band_of(d)
    outside = not baseline and owner is not None and owner is not band and name not in needed
    if baseline or outside:
        if rule.baseline is not None:
            on, why = rule.baseline(d, t, params)
        elif rule.origin == "upstream":
            on, why = rule.auto(d, t, params)
        else:
            on, why = False, f"not an upstream lever ({rule.origin}): off unless its knob is set"
        if outside:
            why = (
                f"a lever of the {owner.value} band on a {band.value} band deployment, as under "
                f"baseline: {why}"
            )
            return Decision(name, on, "band", why, params)
        return Decision(name, on, BASELINE, why, params)
    on, why = rule.auto(d, t, params)
    if on and rule.kind is Kind.NUMERIC:
        held = [gate for gate in rule.gates if gate.holds(d)]
        if not held:
            scopes = "; ".join(gate.scope for gate in rule.gates)
            return Decision(
                name, False, "gate", f"{why}, but no accuracy gate holds ({scopes})", params
            )
        why = f"{why}; gate: {held[0].scope}"
    return Decision(name, on, AUTO, why, params)
