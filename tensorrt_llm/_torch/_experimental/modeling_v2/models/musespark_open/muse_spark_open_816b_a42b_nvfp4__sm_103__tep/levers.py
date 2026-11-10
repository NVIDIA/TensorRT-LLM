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

* structural: a fact the lever's code needs (an arena kind, an architecture, a fact
  variable; ``dispatch.structural_rules``) excludes it; nothing overrides that;
* dependency: a lever it builds on (``dispatch.FastPath.depends_on``) or requires
  (``Rule.requires``) is off;
* knob: ``off`` removes it, and with it every lever built on it or requiring it; ``on``
  keeps it whatever its deployment rule and gates say;
* default: ``auto`` runs the lever's deployment rule (``Rule.auto``), and a lossless-numeric
  lever also needs one of its accuracy gates (``Rule.gates``); ``baseline`` turns every
  lever off that is not the upstream target's, and gives the upstream ones their upstream
  rule (``Rule.baseline``).

``on`` for a lever the composition does not serve, that a structural rule excludes, whose
dependency or requirement is off, or that conflicts with another lever that is on, raises.
So does every grammar error and every retired switch (``RETIRED``); ``UCX_RNDV_THRESH`` on
a server with a cache transceiver (a ``ctx`` or ``gen`` role, or an unhinted one). The resolver
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
SOURCES = ("structural", "dependency", "knob", "baseline", "auto", "gate")

#: Retired switches and what replaced them. Setting one, to any value, raises.
#:
#: Every switch of the sm103 decode stack became a lever; a deployment that still exports one
#: of these names is told which knob to set instead, rather than running with the switch
#: silently ignored. A row-bound or mode switch names its lever's parameter knob, which has
#: no effect on its own: set the lever too.
RETIRED: Mapping[str, str] = types.MappingProxyType(
    {
        "TRTLLM_MUSESPARK_TEP8_FC_SHARD_MAX_ROWS": (
            PREFIX + "TEP8_FC_SHARD_WIDE=on, with its MAX_ROWS"
        ),
        "MUSESPARK_OPEN_TEP8_PERSISTENT_EXPERTS_MAX_ROWS": (
            PREFIX + "TEP8_PERSISTENT_BATCH_EXPERTS=on, with its MAX_ROWS"
        ),
        "TRTLLM_MUSESPARK_TEP8_EXPERTS_EARLY_START": PREFIX + "TEP8_EXPERTS_EARLY_START=on",
        "MUSESPARK_OPEN_TEP8_ROUTER_SHARD": (
            PREFIX + "TEP8_ROUTER_SHARD=on (and " + PREFIX + "TEP8_ROUTER_SHARD_WIDE=on for wide)"
        ),
        "MUSESPARK_OPEN_TEP8_WIDE_DIRECT_TRANSPORTS": (
            PREFIX + "TEP8_WIDE_DIRECT_TRANSPORTS=on, with its SITES"
        ),
        "MUSESPARK_OPEN_TEP8_WIDE_DIRECT_POOL": PREFIX + "TEP8_WIDE_DIRECT_POOL=on",
        "MUSESPARK_OPEN_DRAFTER_WIDE_HOPS": (
            PREFIX + "WIDE_DRAFTER_HOPS=on, with its TP8_MAX_ROWS"
        ),
        "TRTLLM_MUSESPARK_TEP8_FINALIZE_TWO_SHOT": PREFIX + "FINALIZE_TWO_SHOT=on",
        "TRTLLM_MUSESPARK_TEP8_FINALIZE_TWO_SHOT_MAX_ROWS": (
            PREFIX + "FINALIZE_TWO_SHOT=on, with its TP8_MAX_ROWS"
        ),
        "TRTLLM_MUSESPARK_TEP8_FINALIZE_TRANSPORT_MAX_TOKENS": (
            PREFIX + "TEP8_FINALIZE_TRANSPORT_WIDE=on, with its MAX_TOKENS"
        ),
        "TRTLLM_MUSESPARK_FINALIZE_TRANSPORT_TRIGGER": PREFIX + "FINALIZE_ENTRY_TRIGGER=on",
        "TRTLLM_MUSESPARK_TEP8_BATCH_SPLITK": PREFIX + "TEP8_BATCH_SPLITK=on",
        "TRTLLM_MUSESPARK_MULTI_REQUEST_COMBINE_GATE": (PREFIX + "MULTI_REQUEST_COMBINE_GATE=on"),
        "TRTLLM_MUSESPARK_ROUTER_STAGED_GEMM": (PREFIX + "ROUTER_STAGED_GEMM=on, with its BUILD"),
        "MUSESPARK_OPEN_DRAFTER_FUSED_PREP": PREFIX + "DRAFTER_FUSED_PREP=on",
        "MUSESPARK_OPEN_DRAFTER_FUSED_PREP_MIN_REQUESTS": (
            PREFIX + "DRAFTER_FUSED_PREP=on, with its MIN_REQUESTS"
        ),
        "MUSESPARK_OPEN_RING_BUDDY_MAX_ROWS": (
            PREFIX + "TEP8_RING_BUDDY_WIDE=on, with its MAX_ROWS"
        ),
        "TRTLLM_MUSESPARK_TEP8_WIDE_BALANCE_ROWS": (
            PREFIX + "TEP8_WIDE_BALANCE=on, with its MAX_ROWS"
        ),
        "MUSESPARK_OPEN_RING_BUDDY_STATS": (
            PREFIX + "RING_BUDDY_STATS=on, with its INTERVAL_S (whole seconds)"
        ),
        "TRTLLM_MUSESPARK_TEP8_EXPERTS_IN_KERNEL_PLACEMENT": (
            PREFIX + "TEP8_EXPERTS_IN_KERNEL_PLACEMENT=on"
        ),
        "TRTLLM_MUSESPARK_TEP8_EXPERTS_INTERLEAVED_FETCH": (
            PREFIX + "TEP8_EXPERTS_INTERLEAVED_FETCH=on"
        ),
        "TRTLLM_MUSESPARK_DRAFTER_CONV_EPILOGUE": PREFIX + "DRAFTER_CONV_EPILOGUE=on",
        "TRTLLM_MUSESPARK_DRAFTER_FINISH_NORM": PREFIX + "DRAFTER_FINISH_NORM=on",
        "TRTLLM_MUSESPARK_DRAFTER_SILU_EPILOGUE": PREFIX + "DRAFTER_SILU_EPILOGUE=on",
        "TRTLLM_MUSESPARK_TEP8_ATTN_BLOCK_SPLITK": PREFIX + "TEP8_ATTN_BLOCK_SPLITK=on",
        "TRTLLM_MUSESPARK_SWA_BLOCK_SINGLE_EARLY_LOADS": (
            PREFIX + "SWA_BLOCK_SINGLE_EARLY_LOADS=on"
        ),
        "TRTLLM_MUSESPARK_DRAFTER_MULTI_REQUEST_CELLS": (PREFIX + "DRAFTER_MULTI_REQUEST_CELLS=on"),
    }
)
#: Name prefixes only retired switches carry; any other name under them raises too.
RETIRED_PREFIXES: Mapping[str, str] = types.MappingProxyType({})


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

    def decodes(self) -> bool:
        """Whether this server runs decode steps (an unknown role counts as one that does)."""
        return self.role is not Role.CTX

    def reaches(self, rows: int) -> bool:
        """Whether some decode step can have at least ``rows`` rows (true for an unknown
        ``max_batch_size``)."""
        return self.max_batch_size is None or self.max_batch_size * self.rows_per_request >= rows

    def describe(self) -> str:
        return (
            f"sm{self.sm} tep{self.tp}, {self.role.value} ({self.role_source}), max_batch_size "
            f"{_count(self.max_batch_size)}, max_num_tokens {_count(self.max_num_tokens)}, "
            f"{self.arena.value} arena"
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
        return out


def _count(value: Optional[int]) -> str:
    return "unknown" if value is None else str(value)


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


def _always(why: str) -> RuleFn:
    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        return True, why

    return rule


def _knob_only(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    """Off under ``auto``: on only by its knob, until this target gates it."""
    return False, "flagged: on only by its knob, until it is gated on this target"


def _tep8_only(what: str) -> RuleFn:
    """On at TP8, the size this path was written and measured for; off elsewhere by its knob only.

    The ported decode stack was built for the sm103 TP8 composition over two trays. At TP4 these
    paths are either unserved by the composition or untuned, so they stay knob-only there.

    What carries this default is the acceptance evidence: on the SPEED-Bench 80-row coding set
    (chat-templated, rejection sampling, DFlash d7, concurrency 1) every lever listed below runs
    **token-identical** to the stack with all of them off -- 4.6303 AL, the same 3623/6986
    accepted/drafted, on GB300 TEP8. Every arm there repeats bitwise, so that is an exact result,
    not one inside a noise band; and because the emitted tokens are identical, it does not depend
    on the baseline's absolute level. A lever is default-on here only if it was in a measured arm.

    The decode-step speedup measured for this stack (12.23 -> 11.04 ms at bs4) was taken with
    ``TRTLLM_MULTI_INSTANCE_CONTEXT_FMHA=1`` also set. That switch is **not** a lever and is not
    enabled here: it is the one component that moves acceptance (-3.79 % AL on the same gate, all
    of the loss). So the levers' own share of that speedup is not yet separated, and this
    docstring does not claim it.
    """

    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if d.tp == 8:
            return True, what
        return False, f"tep{d.tp}: on only by its knob (written for the tep8 composition)"

    return rule


def _tep4_only(what: str) -> RuleFn:
    """On at TP4, where the composition carries the path; off at any other size by its knob only.

    The two paths that read it (``wide_drafter_hops``, ``finalize_two_shot``) are the sm103 TP4
    composition's own, served there under ``auto`` as before; the wider composition serves them
    too, but leaves them to their knob.
    """

    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if d.tp == 4:
            return True, what
        return False, f"tep{d.tp}: on only by its knob (the tep4 composition's path)"

    return rule


def _ported_tep8(kind: Kind, what: str, params: Tuple[Param, ...] = ()) -> Rule:
    """A ported switch that is ON by default at TP8 -- the parallelism it was written for."""
    return Rule("LL", kind, _tep8_only(what), params=params)


def _ported(kind: Kind, params: Tuple[Param, ...] = ()) -> Rule:
    """A switch of the sm103 decode stack, ported from an environment variable: default off.

    Every one of them gates a decode step of at most a few hundred rows, so the serving region
    is ``LL``; it is off under ``auto`` and ``baseline`` (``_knob_only``, and a non-upstream
    origin), so a deployment that sets no knob runs exactly the paths it ran before.
    """
    return Rule("LL", kind, _knob_only, params=params)


def _tep8_decode(what: str) -> RuleFn:
    """On for the decode steps of a tep8 server; off elsewhere, and on its knob only.

    The levers that use it were written for the sm103 tep8 decode step and are bitwise the
    path they replace, checked by running the model: each reproduced the served arm's decode
    step structure exactly, which a lever that moved a single sampled token could not.
    """

    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if not d.decodes():
            return False, _NO_DECODE
        if d.tp != 8:
            return False, f"tep{d.tp}: on only by its knob (the tep8 composition's path)"
        return True, what

    return rule


def _ported_on(kind: Kind, what: str, params: Tuple[Param, ...] = ()) -> Rule:
    """A ported switch this target now serves by default on a tep8 decode server.

    ``baseline`` still turns it off (the origin is not ``upstream``), so a deployment can ask
    for the pre-port paths with one knob.
    """
    return Rule("LL", kind, _tep8_decode(what), params=params)


def _hot_replicas(requires_decode: bool) -> RuleFn:
    def rule(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
        if requires_decode and not d.decodes():
            return False, _NO_DECODE
        if t.on("half_window_experts"):
            return False, "yields to half_window_experts, which places each step's experts"
        return True, "33..512-row decode steps"

    return rule


LL_MAX_BATCH = 8


#: The TP4 paths: on wherever served under ``auto``. ``wide_drafter_hops`` was one until the
#: wider composition began to serve it too; it has its own rule now (``_tep4_only``).
TEP4_PATHS = (
    "multi_request_prep_append",
    "multi_request_split_kv",
    "multi_request_post_attention_hop",
    "router_gemm",
    "projection_publish",
)

# --------------------------------------------------------------------------
# The parameter defaults of the ported switches
# --------------------------------------------------------------------------
# Each is the value its environment variable named, or the bound the module it configures
# already serves. They are literals because this resolver imports no module of the target.
#
#: ``wide_drafter_hops__TP8_MAX_ROWS``: the rows the sm103 tep4 composition serves
#: (``ops/allreduce/drafter_hop.WIDE_SERVED_ROWS_BY_SM[(10, 3)]``), the drafter blocks of
#: 2..7 requests at K = 7.
WIDE_DRAFTER_HOPS_TP8_ROWS = 56
#: ``finalize_two_shot__TP8_MAX_ROWS``: the last row of the tep8 band as
#: ``kernels/finalize_two_shot/install.ROWS_BY_SM_TP[(103, 8)]`` declares it (19 requests).
FINALIZE_TWO_SHOT_TP8_ROWS = 152
#: ``tep8_fc_shard_wide__MAX_ROWS``: the five-request decode step (40 rows at K = 7) the
#: sharded ``fc``'s note names; its capacity is 128 (``spec_worker.FC_SHARD_TP8_ROW_CAPACITY``).
FC_SHARD_TP8_ROWS = 40
#: ``tep8_persistent_batch_experts__MAX_ROWS``: the largest stacked call the persistent
#: kernel's batch builds serve (``ops/moe/persistent_experts.BATCH_MAX_ROWS``).
PERSISTENT_BATCH_ROWS = 32
#: ``tep8_wide_direct_transports__SITES``: the sites the path can publish
#: (``kernels/wide_direct_transports/transports.SITES``); both, as its ``1`` meant.
WIDE_DIRECT_SITES = ("pool", "unpool")
#: ``tep8_finalize_transport_wide__MAX_TOKENS``: the widest TP8 layout
#: (``ops/moe/finalize_allreduce_transport.TP8_WIDE_MAX_TOKENS``), five requests at K = 7.
FINALIZE_TRANSPORT_TOKENS = 40
#: ``router_staged_gemm__BUILD``: the staged GEMM's builds and the one its ``1`` selected
#: (``ops/moe/router_gemm.STAGED_BUILDS``, ``STAGED_DEFAULT``). Exactly one is named.
ROUTER_STAGED_BUILDS = ("bulk", "v4", "bulk-prewait", "v4-prewait")
ROUTER_STAGED_BUILD = "bulk-prewait"
#: ``drafter_fused_prep__MIN_REQUESTS``: ``ops/spec/drafter_step_prep.DEFAULT_MIN_REQUESTS``.
DRAFTER_PREP_MIN_REQUESTS = 2
#: ``tep8_ring_buddy_wide__MAX_ROWS``: the stacked path's row cap on the tep8 ring, replacing
#: ``kernels/replicated_experts/ring_buddy.MAX_ROWS`` (16). 128 is ``8 * B`` for a 16-request
#: verify step at K = 7; the ceiling is one ``ring_buddy_assign`` program's rows
#: (``ops/moe/ring_buddy_assign.MAX_SLOTS // TOP_K``), which ``install`` refuses to exceed.
RING_BUDDY_MAX_ROWS = 128
RING_BUDDY_ROW_CEILING = 256
#: ``tep8_wide_balance__MAX_ROWS``: the band the balanced NVFP4 placement adds above the
#: stacked cap, from ``ring_buddy_assign.WIDE_BALANCE_MIN_ROWS`` (129) to one wide
#: ``ring_buddy_assign`` program's rows (``WIDE_MAX_SLOTS // TOP_K``). The default stays at the
#: narrow program's ceiling; the band above it is reached only by setting the parameter, and
#: the wide program places every call of more than 64 rows.
WIDE_BALANCE_ROWS = 256
WIDE_BALANCE_MIN_ROWS = 129
WIDE_BALANCE_ROW_CEILING = 512
#: ``ring_buddy_stats__INTERVAL_S``: seconds between the placement-census reports TP rank 0
#: logs. Whole seconds only (the knob grammar has no fractional parameter).
RING_BUDDY_STATS_INTERVAL_S = 60


def _tep4_path_rule() -> Rule:
    return Rule(
        "upstream",
        Kind.COMPOSITION,
        _always("its rows, wherever served"),
    )


def _vocab_parallel_head(d: Deployment, t: Resolving, p: Mapping[str, object]) -> Tuple[bool, str]:
    return True, "verify heads of 16..1024 rows"


#: Every lever whose rule is not the default; every other served lever is on under ``auto``
#: and ``baseline``.
RULES: Mapping[str, Rule] = types.MappingProxyType(
    {
        "lm_head_inbox": Rule(
            "upstream",
            Kind.COMPOSITION,
            _always("8-row heads"),
        ),
        "graph_start_qkv_prefetch": Rule(
            "upstream",
            Kind.COMPOSITION,
            _always("graph starts of split-K row counts"),
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
        "vocab_parallel_head": Rule(
            "upstream",
            Kind.COMPOSITION,
            _vocab_parallel_head,
            params=(
                Param("MAX_BATCH", LL_MAX_BATCH, 1, 64),
                Param("MIN_ROWS", 16, 8, 16),
            ),
        ),
        # The two tep4 composition paths the wider composition also serves, on its knob only.
        # ``TP8_MAX_ROWS`` is read only where the tep8 workspace is built; at tep4 each path
        # keeps the rows its own composition serves.
        "wide_drafter_hops": Rule(
            "upstream",
            Kind.COMPOSITION,
            _tep4_only("9..56-row block decodes"),
            params=(Param("TP8_MAX_ROWS", WIDE_DRAFTER_HOPS_TP8_ROWS, 9, 64),),
        ),
        "finalize_two_shot": Rule(
            "upstream",
            Kind.COMPOSITION,
            _tep4_only("72..512-row decode steps"),
            params=(Param("TP8_MAX_ROWS", FINALIZE_TWO_SHOT_TP8_ROWS, 24, 152),),
        ),
        # The sm103 decode-stack switches, ported from environment variables: default off.
        "tep8_fc_shard_wide": _ported_tep8(
            Kind.FLAGGED,
            "the TP8 sharded drafter fc over the tray's NVLink",
            (Param("MAX_ROWS", FC_SHARD_TP8_ROWS, 9, 128),),
        ),
        "tep8_persistent_batch_experts": _ported_tep8(
            Kind.FLAGGED,
            "the persistent expert kernel's batch builds at TP8",
            (Param("MAX_ROWS", PERSISTENT_BATCH_ROWS, 9, 32),),
        ),
        "tep8_experts_early_start": _ported_tep8(
            Kind.FLAGGED, "the stacked expert call launched with its placement's early trigger"
        ),
        "tep8_router_shard": _ported_tep8(Kind.FLAGGED, "the TP8 wide-prologue router shard"),
        "tep8_router_shard_wide": _ported_tep8(
            Kind.FLAGGED, "router_shard_wide's built-in bits on the TP8 wide prologue"
        ),
        "tep8_wide_direct_pool": _ported_tep8(
            Kind.FLAGGED, "pool_into_route's TP8 wide direct sites"
        ),
        "tep8_wide_direct_transports": _ported(
            Kind.FLAGGED, (Param("SITES", WIDE_DIRECT_SITES, choices=WIDE_DIRECT_SITES),)
        ),
        "tep8_batch_splitk": _ported_tep8(Kind.BITWISE, "the TP8 batch split-K projection kernels"),
        "finalize_entry_trigger": _ported_tep8(
            Kind.BITWISE, "the finalize transport's entry trigger"
        ),
        "tep8_finalize_transport_wide": _ported_tep8(
            Kind.FLAGGED,
            "the widened TP8 finalize transport",
            (Param("MAX_TOKENS", FINALIZE_TRANSPORT_TOKENS, 17, 40),),
        ),
        "multi_request_combine_gate": _ported(Kind.BITWISE),
        "router_staged_gemm": _ported(
            Kind.FLAGGED,
            (Param("BUILD", (ROUTER_STAGED_BUILD,), choices=ROUTER_STAGED_BUILDS),),
        ),
        # NOT default-on: it resolved off (dependency) in BOTH measured arms, so the
        # bitwise-clean SPEED-Bench result does not cover it.  Knob-only until it does.
        "drafter_fused_prep": _ported(
            Kind.FLAGGED, (Param("MIN_REQUESTS", DRAFTER_PREP_MIN_REQUESTS, 1, 64),)
        ),
        "tep8_ring_buddy_wide": _ported_tep8(
            Kind.FLAGGED,
            "the TP8 ring-buddy placement's wide row cap",
            (Param("MAX_ROWS", RING_BUDDY_MAX_ROWS, 1, RING_BUDDY_ROW_CEILING),),
        ),
        "tep8_wide_balance": _ported_tep8(
            Kind.FLAGGED,
            "the 256-row wide balance",
            (
                Param(
                    "MAX_ROWS",
                    WIDE_BALANCE_ROWS,
                    WIDE_BALANCE_MIN_ROWS,
                    WIDE_BALANCE_ROW_CEILING,
                ),
            ),
        ),
        # Diagnostics: one extra assign launch per placement, and the in-kernel placement
        # stands aside for it (``ring_buddy``'s ``_stats is None``), so it is never on by
        # itself under ``auto``.
        "ring_buddy_stats": _ported(
            Kind.FLAGGED,
            (Param("INTERVAL_S", RING_BUDDY_STATS_INTERVAL_S, 1, 86400),),
        ),
        "tep8_experts_in_kernel_placement": _ported_tep8(
            Kind.FLAGGED, "placement computed inside the expert kernel"
        ),
        "tep8_experts_interleaved_fetch": _ported_tep8(
            Kind.FLAGGED, "the expert kernel's interleaved weight fetch"
        ),
        "drafter_conv_epilogue": _ported_on(
            Kind.BITWISE, "9..32-row drafter blocks on a tep8 decode server"
        ),
        "drafter_finish_norm": _ported_on(
            Kind.BITWISE, "drafter blocks the drafter hops do not take, tep8 decode"
        ),
        # +560 MiB per generation rank for the interleaved gate_up copy, which comes out of
        # the KV cache at a fixed free_gpu_memory_fraction.
        "drafter_silu_epilogue": _ported_on(
            Kind.BITWISE, "9..32-row drafter blocks on a tep8 decode server"
        ),
        # The o_proj half is bitwise: split_k 1 reproduces the built-in cuBLAS call, checked
        # on this target. The QKV half is not, at TEP8. It replaces the 9..32-row call with a
        # split_k 4 cell on the premise that the served call there is already that cell -- true
        # only where a wide table is armed, and the path that arms one
        # (``wide_projection_gathers``) is TEP4-only on sm103. At TEP8 the served call is
        # cuBLAS, so the summation order changes and the decoded text moves with it, which a
        # batch-4 run makes plain. Lossless-numeric, and it needs a gate before it is on.
        "tep8_attn_block_splitk": _ported(Kind.NUMERIC),
        "swa_block_single_early_loads": _ported_on(
            Kind.BITWISE, "the one-request sliding-window block on a tep8 decode server"
        ),
        # The qkv / gate_up / coefficient cells keep their row's split_k and are bitwise the
        # table's row; the o_proj and down_proj cells reproduced cuBLAS's bits on every checked
        # shape, which cannot be proven against a closed library.
        "drafter_multi_request_cells": _ported(Kind.FLAGGED),
        **{name: _tep4_path_rule() for name in TEP4_PATHS},
    }
)


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


def deployment_of(
    model_config,
    compute_capability,
    arena: Arena,
    engine_facts,
    snapshot: Mapping[str, str],
) -> Deployment:
    """The ``Deployment`` of a model: its config's mapping, the device, the arena and the engine.

    ``engine_facts`` carries ``max_batch_size``, ``max_num_tokens`` and ``disaggregated``
    (``_router_index.DeploymentFacts``), or is None when the target class was built without
    them.

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
            name, rule, mode, knobs.default, params, d, t, structural, depends_on, requires
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
) -> Decision:
    """One lever's decision, from the decisions already made (``resolve``'s order)."""
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
    if baseline:
        if rule.baseline is not None:
            on, why = rule.baseline(d, t, params)
        elif rule.origin == "upstream":
            on, why = rule.auto(d, t, params)
        else:
            on, why = False, f"not an upstream lever ({rule.origin}): off unless its knob is set"
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
