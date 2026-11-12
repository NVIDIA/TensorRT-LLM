# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode fast paths of the glm_moe_dsa sm_103 tp8 target, which of them a process runs, and what happens
when one cannot.

Composition. ``FAST_PATHS`` names every fast path of the family once, with the
paths it extends (``depends_on``): a path that extends another has no effect
without it. Each target serves some of them (``TargetPaths``), and every path
it serves is on by default. ``TRTLLM_STAIRCASE_GLM_LL_DISABLE`` takes a
comma-separated list of path names and marks those off in the plan together
with every path that depends on them, transitively; a name that is not a path
the target serves raises. ``resolve`` reads it once, at construction, into a
``Plan``. The fast paths read the installed plan through ``is_enabled``;
``DISABLE`` turns a path and its dependents off.

Rank uniformity. Several fast paths are collectives across the model's ranks: a
rank that runs one while a peer runs the generic path waits forever. So every
rank must read the same knob values. ``check_rank_uniform`` allgathers them,
with the shared switches the kernels read and the modeling_v2 mode
(``rank_uniform_inputs``), once, at construction, before they are parsed and
before any fast path builds a collective workspace, and raises on every rank
with the per-rank table on a mismatch. The plan is a function of the knobs
alone, so equal knobs give every rank the same plan, and a value no rank can
parse fails on every rank alike. Once the model is built,
``check_rank_uniform_adopted`` compares the fast-path state every rank
attached to its modules the same way.

Phases. ``step`` runs a model step under the ``Phase`` of the target that
takes it (``PREFILL`` or ``DECODE``); ``phase`` reads it, and is None outside
a step.

Two kinds of not taking a fast path, kept apart:

* **uncovered**: the step is outside the shapes the path takes (more rows than
  its kernel holds, a KV length outside its tier, a prefill step). ``uncovered``
  logs the reason once per (path, reason); the caller runs the next tier or the
  generic path in every mode.
* **missed**: the step is inside the shapes the path takes, but a precondition
  is absent (a workspace, a compiled kernel, multi-node NVLink memory, a device
  check). Under ``modeling_v2='require'`` ``missed`` raises
  ``FastPathUnavailable`` naming the path and the precondition; otherwise it
  logs once per (path, reason), counts the miss, and the caller falls back. A
  miss decided per call never raises: it is counted and logged once.

The kernels report their decisions through ``log_capability``, which also
classifies every decline (``classify_decline``) by where it is decided
(``DECLINE_KINDS``), the layers each path serves (``LAYER_CELLS``) and the
reason: a path turned off is neither kind, a decline decided once inside the
path's layers is a miss, and a decline decided per call is a miss only when
its reason names a missing resource.

Selection. A kernel or a shared hook asks ``is_enabled(path)`` whether a fast
path runs, looks up its entry points with ``get_impl(path, name)`` (the kernel
package of the path registers them with ``register`` when it is imported) and
logs its per-rank decisions with ``log_capability``. ``is_enabled`` answers
from the installed plan (``install``); without one, which is the case of every
process that does not build a target of the family, no fast path runs. ``install`` logs a
record line (``feature_record``): every enabled path with its numerical variant
(``NUMERICAL_VARIANTS``), every set knob of the family and the shared
TensorRT-LLM switches the kernels read.

Validation. ``TRTLLM_MODELING_V2_VALIDATE`` (read once, in ``resolve``) turns on
the diagnostics of the fast paths that have some (``validate_mode``): the
step-input stager's guard and the draft loop's in-pipeline native oracle.

Knobs. The knobs of the family are ``KNOBS``: ``DISABLE``, the numerical variant
of ``routed_moe_into_hop`` and the cross-rank top-k wait timeout (read by the
C++ kernels). A target refuses a knob that only a path it does not serve reads
(``KNOB_PATHS``). The kernels run at fixed launch parameters; ``resolve`` refuses
every other ``TRTLLM_STAIRCASE_GLM_LL_*`` variable that is set.
"""

from __future__ import annotations

import enum
import functools
import hashlib
import importlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    MODELING_V2_VALIDATE_ENV,
    ModelingV2Mode,
    step_contract_enabled,
)
from tensorrt_llm.logger import logger

#: Prefix of every knob of this target; every set variable with it is part of the rank-uniform
#: check.
KNOB_PREFIX = "TRTLLM_STAIRCASE_GLM_LL_"
#: Comma-separated fast-path names to turn off, with every path that depends on them.
DISABLE_ENV = KNOB_PREFIX + "DISABLE"
#: The numerical variant of ``routed_moe_into_hop`` (``NUMERICAL_VARIANTS``).
ACCUMULATION_ENV = KNOB_PREFIX + "ROUTED_MOE_INTO_HOP_ACCUMULATION"
#: How long the cross-rank top-k kernels wait for their peers before they trap, in ms (C++).
CROSS_RANK_TOPK_TIMEOUT_ENV = KNOB_PREFIX + "CROSS_RANK_TOPK_TIMEOUT_MS"
#: The knobs of this target; ``resolve`` refuses any other set ``KNOB_PREFIX`` variable.
KNOBS: FrozenSet[str] = frozenset({DISABLE_ENV, ACCUMULATION_ENV, CROSS_RANK_TOPK_TIMEOUT_ENV})
#: A fast-path selector this target does not read; ``resolve`` refuses it.
FEATURE_ENV = KNOB_PREFIX + "FEATURES"
#: A narrowing of ``deferred_input_prep`` to some of its parts; ``resolve`` refuses it.
PARTS_ENV = KNOB_PREFIX + "DEFERRED_INPUT_PREP_PARTS"

_VALIDATE_HINT = f"set {MODELING_V2_VALIDATE_ENV}=1 instead"
#: Knobs this target does not read, each with what to use instead; ``resolve`` refuses them.
REMOVED_KNOBS: Mapping[str, str] = {
    FEATURE_ENV: f"every fast path of this target is on; {KNOB_PREFIX}DISABLE turns paths off",
    PARTS_ENV: (
        "each part is a fast path of its own (prep = deferred_input_prep, store = "
        f"sampler_store, stream = execution_stream_loop); list the parts to drop in "
        f"{KNOB_PREFIX}DISABLE"
    ),
    KNOB_PREFIX + "DEFERRED_INPUT_PREP_DEBUG": _VALIDATE_HINT,
    KNOB_PREFIX + "DRAFT_LOOP_VERIFY": _VALIDATE_HINT,
    KNOB_PREFIX + "CROSS_LAYER_EXPERT_PREFETCH_STATS": "the hit counters are not reported",
    KNOB_PREFIX + "CROSS_LAYER_EXPERT_PREFETCH_REPORT_EVERY": "the hit counters are not reported",
    KNOB_PREFIX + "SHARED_INDEXER_ABSORBED_QUERY_FORCE_NATIVE": (
        f"list shared_indexer_absorbed_query in {KNOB_PREFIX}DISABLE to run absorbed_query on the "
        "shared-indexer layers"
    ),
}
#: What ``resolve`` says about any other set variable with the prefix that is not in ``KNOBS``.
_FIXED_HINT = "the fast paths of this target run at fixed launch parameters"

_LOG_TAG = "[glm_moe_dsa]"


@dataclass(frozen=True)
class FastPath:
    """One decode fast path: its name and the paths it extends."""

    name: str
    depends_on: Tuple[str, ...] = ()


# The attention query front, the indexer and top-k selection, sparse attention,
# the attention output and its cross-rank hop, the MoE region and its hop, the
# dense layers, the draft (MTP) layer and loop, and the engine-side step work.
FAST_PATHS: Tuple[FastPath, ...] = (
    # The target layers' query front: the query projections and the absorbed query.
    FastPath("absorbed_query"),
    # The same query front on the draft layer.
    FastPath("draft_absorbed_query"),
    # The query front of layers that reuse an earlier layer's indexer selection.
    FastPath("shared_indexer_absorbed_query", depends_on=("absorbed_query",)),
    # One projection launch for the latent KV and the indexer key and weights.
    FastPath("fused_kv_indexer_projection", depends_on=("absorbed_query",)),
    # The query front of layers that run the full indexer, with its rope and
    # top-k exchange started early.
    FastPath(
        "indexer_layer_front",
        depends_on=(
            "absorbed_query",
            "draft_absorbed_query",
            "indexer_topk",
            "rope_kv_append",
            "cross_rank_topk",
        ),
    ),
    # The next layer's RMSNorm deferred across the post-MoE hop into its projection.
    FastPath("deferred_norm", depends_on=("absorbed_query", "post_moe_hop")),
    # The deferred-norm hop and projection chain with cache-eviction hints.
    FastPath("deferred_norm_evict_hints", depends_on=("deferred_norm",)),
    # Rope, FP8 query, latent-KV append, indexer-K scatter and the top-k to
    # KV-index bridge in one launch per layer.
    FastPath("rope_kv_append"),
    # Indexer logits and top-k selection on decode steps.
    FastPath("indexer_topk"),
    # Top-k over the KV pages each rank owns, merged across the ranks.
    FastPath("cross_rank_topk"),
    # The cross-rank top-k with a cluster merge of the per-rank candidates.
    FastPath("cross_rank_topk_cluster_merge", depends_on=("cross_rank_topk",)),
    # The cluster merge also emits the top-k to KV-index bridge.
    FastPath(
        "cross_rank_topk_global_emit",
        depends_on=("rope_kv_append", "cross_rank_topk_cluster_merge"),
    ),
    # Sparse MLA attention for the decode rows.
    FastPath("sparse_mla_decode"),
    # The attention output projection and its local tail.
    FastPath("attention_output_tail"),
    # The attention tail's prefetch schedule: the next weights are fetched while the current
    # attention runs.
    FastPath("attention_output_tail_prefetch", depends_on=("attention_output_tail",)),
    # The attention output published straight into the post-attention hop.
    FastPath(
        "attention_output_into_hop", depends_on=("attention_output_tail", "post_attention_hop")
    ),
    # Output-projection weight staging inside an already-waiting kernel of the attention window.
    FastPath("oproj_weight_staging", depends_on=("attention_output_into_hop", "sparse_mla_decode")),
    # Cross-rank sum, residual add, RMSNorm and router after attention, in one hop.
    FastPath("post_attention_hop"),
    # The MoE routing (top-8) computed inside the post-attention hop.
    FastPath("hop_routing", depends_on=("attention_output_into_hop", "routed_moe")),
    # The router GEMM at the pre-MoE boundary.
    FastPath("router_gemm"),
    # The routed NVFP4 experts in one launch.
    FastPath("routed_moe"),
    # The routed-expert result written straight into the post-MoE hop.
    FastPath("routed_moe_into_hop", depends_on=("post_moe_hop", "routed_moe")),
    # The shared expert's weights prefetched into L2 by the routed-expert launch.
    FastPath("shared_expert_prefetch", depends_on=("routed_moe_into_hop", "hop_routing")),
    # Cross-rank finalize, reduction and RMSNorm after the MoE, in one hop.
    FastPath("post_moe_hop"),
    # The next MoE layer's predicted experts prefetched into L2.
    FastPath("cross_layer_expert_prefetch", depends_on=("routed_moe",)),
    # The dense layers' small-row MLP.
    FastPath("dense_mlp"),
    # The draft layer's input normalization and packing.
    FastPath("draft_front"),
    # The draft layer's bf16 routed experts.
    FastPath("draft_routed_moe"),
    # Only the accepted rows of the first draft step through the post-attention hop.
    FastPath("draft_live_rows", depends_on=("post_attention_hop", "draft_loop")),
    # Draft-token selection at the end of each draft step.
    FastPath("draft_tail"),
    # The draft loop's embedding, bookkeeping and sparse-attention metadata.
    FastPath("draft_loop"),
    # The decode step's metadata prologue and the target embedding in one launch.
    FastPath("decode_prologue", depends_on=("draft_loop",)),
    # The decode step's input preparation recorded and issued as one launch.
    FastPath("deferred_input_prep"),
    # The speculative sampler's store update and host copy in one launch.
    FastPath("sampler_store"),
    # The overlap executor loop on the execution stream.
    FastPath("execution_stream_loop"),
    # Freed host pages returned to the operating system lazily on decode ranks.
    FastPath("host_page_retention", depends_on=("post_attention_hop",)),
)

FAST_PATH_NAMES: Tuple[str, ...] = tuple(p.name for p in FAST_PATHS)
_BY_NAME: Mapping[str, FastPath] = {p.name: p for p in FAST_PATHS}


def _dependents() -> Dict[str, FrozenSet[str]]:
    """Every path that depends on each path, directly or transitively."""
    direct: Dict[str, List[str]] = {name: [] for name in FAST_PATH_NAMES}
    for path in FAST_PATHS:
        for base in path.depends_on:
            direct[base].append(path.name)
    closure = {}
    for name in FAST_PATH_NAMES:
        seen, todo = set(), list(direct[name])
        while todo:
            dependent = todo.pop()
            if dependent not in seen:
                seen.add(dependent)
                todo.extend(direct[dependent])
        closure[name] = frozenset(seen)
    return closure


_DEPENDENTS = _dependents()


def parse_disable(raw: Optional[str], served: Iterable[str] = FAST_PATH_NAMES) -> FrozenSet[str]:
    """The fast-path names a ``DISABLE`` value lists (case and blanks ignored).

    ``served`` is the fast paths of the target that reads the value (every fast path by default).

    Raises:
        ValueError: naming every entry that is not one of ``served``.
    """
    if not raw:
        return frozenset()
    names = {tok.strip().lower() for tok in raw.split(",") if tok.strip()}
    served = frozenset(served)
    unknown = sorted(names - served)
    if unknown:
        raise ValueError(
            f"{DISABLE_ENV} names {unknown}, which are not fast paths of this target; "
            f"the fast paths are {[name for name in FAST_PATH_NAMES if name in served]}"
        )
    return frozenset(names)


def disable_closure(names: FrozenSet[str]) -> FrozenSet[str]:
    """``names`` and every path that depends on one of them."""
    closed = set(names)
    for name in names:
        closed |= _DEPENDENTS[name]
    return frozenset(closed)


#: Every knob of this target set in an environment, as (name, value) sorted by name.
Knobs = Tuple[Tuple[str, str], ...]


def read_knobs(environ: Optional[Mapping[str, str]] = None) -> Knobs:
    """Every knob of the family set in ``environ`` (the process environment by default)."""
    env = os.environ if environ is None else environ
    return tuple(sorted((k, v) for k, v in env.items() if k.startswith(KNOB_PREFIX)))


def knob_digest(knobs: Knobs) -> str:
    """A short digest of ``knobs`` for the log line."""
    return hashlib.sha256(repr(knobs).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Plan:
    """The composition this process resolved: which fast paths run, and under which mode."""

    mode: ModelingV2Mode
    # Paths the DISABLE knob names.
    disabled: FrozenSet[str]
    # Paths turned off only because a path they depend on is disabled.
    disabled_dependents: FrozenSet[str]
    # Every knob of this target set in the environment, as (name, value).
    knobs: Knobs
    # Whether the fast paths run their diagnostics (TRTLLM_MODELING_V2_VALIDATE).
    validate: bool = False
    # What the target that resolved the plan serves (``TargetPaths``); None serves every path.
    target: Optional["TargetPaths"] = None

    @property
    def require(self) -> bool:
        return self.mode is ModelingV2Mode.REQUIRE

    @property
    def served(self) -> FrozenSet[str]:
        """The fast paths the target serves; the others are off in every plan of the target."""
        return frozenset(FAST_PATH_NAMES) if self.target is None else self.target.served

    def enabled(self, name: str) -> bool:
        """Whether fast path ``name`` runs in this process.

        Raises:
            KeyError: if ``name`` is not a fast path, so a misspelled call site fails.
        """
        if name not in _BY_NAME:
            raise KeyError(f"{name!r} is not a fast path of this target")
        return (
            name in self.served
            and name not in self.disabled
            and name not in self.disabled_dependents
        )

    def describe(self) -> str:
        """One line naming the mode, the number of fast paths that run and what is off."""
        on = sum(self.enabled(name) for name in FAST_PATH_NAMES)
        words = f"modeling_v2={self.mode.value}, {on}/{len(self.served)} fast paths"
        if self.disabled:
            words += f", disabled {sorted(self.disabled)}"
        if self.disabled_dependents:
            words += f" and their dependents {sorted(self.disabled_dependents)}"
        return words + f", knobs {knob_digest(self.knobs)}"


#: The knobs that only one fast path reads; a target that does not serve the path refuses them.
KNOB_PATHS: Mapping[str, str] = {ACCUMULATION_ENV: "routed_moe_into_hop"}


def _refuse_removed_knobs(knobs: Knobs, served: Iterable[str] = FAST_PATH_NAMES) -> None:
    """Raise if a variable with the knob prefix that is not in ``KNOBS`` is set, or a knob of a
    fast path outside ``served``, naming each with what to use instead."""
    served = frozenset(served)
    refused = [
        f"{name} ({REMOVED_KNOBS.get(name, _FIXED_HINT)})" for name, _ in knobs if name not in KNOBS
    ]
    refused += [
        f"{name} (this target does not serve {KNOB_PATHS[name]})"
        for name, _ in knobs
        if name in KNOB_PATHS and KNOB_PATHS[name] not in served
    ]
    if refused:
        raise ValueError(f"this target does not read {'; '.join(refused)}; unset them")


def resolve(
    mode: ModelingV2Mode,
    environ: Optional[Mapping[str, str]] = None,
    target: Optional["TargetPaths"] = None,
) -> Plan:
    """Read the knobs once and resolve the composition of ``target``; arms ``missed`` for ``mode``.

    ``target`` is what the target serves (every fast path, with the default tables, when None).

    Raises:
        ValueError: if ``DISABLE`` names something that is not a fast path the target serves, or
            if a variable with the knob prefix that is not in ``KNOBS`` (or a knob of a path the
            target does not serve) is set.
    """
    knobs = read_knobs(environ)
    served = FAST_PATH_NAMES if target is None else target.served
    _refuse_removed_knobs(knobs, served)
    disabled = parse_disable(dict(knobs).get(DISABLE_ENV), served)
    plan = Plan(
        mode=mode,
        disabled=disabled,
        disabled_dependents=disable_closure(disabled) - disabled,
        knobs=knobs,
        validate=step_contract_enabled(),
        target=target,
    )
    set_require(plan.require)
    logger.info(f"{_LOG_TAG} dispatch: {plan.describe()}")
    return plan


# --------------------------------------------------------------------------------------------------
# Rank uniformity
# --------------------------------------------------------------------------------------------------

#: Gathers one rank's value from every rank of a ``world_size``-rank group, or returns None
#: (having logged why) when no such group is available.
Allgather = Callable[[Any, int], Optional[List[Any]]]

#: The key under which ``rank_uniform_inputs`` records the modeling_v2 mode.
MODE_KEY = "modeling_v2"


def rank_uniform_inputs(
    knobs: Knobs, mode: ModelingV2Mode, environ: Optional[Mapping[str, str]] = None
) -> Knobs:
    """What every rank must agree on before the model is built, as (name, value) sorted by name.

    ``knobs``, the shared switches the kernels read to choose a collective or a protocol
    (``SHARED_ENV``) and the modeling_v2 validation switch, where set in ``environ`` (the
    process environment by default), and ``mode``.
    """
    env = os.environ if environ is None else environ
    shared = [(name, env[name]) for name in SHARED_ENV + (MODELING_V2_VALIDATE_ENV,) if name in env]
    return tuple(sorted(list(knobs) + shared + [(MODE_KEY, mode.value)]))


def _world_allgather(value: Any, world_size: int) -> Optional[List[Any]]:
    """``value`` from every rank of the session's MPI communicator.

    Returns None, with a warning, without MPI or when the communicator is not
    the ``world_size`` ranks of this model: gathering over any other group
    would either prove nothing or wait on ranks that never build this model.
    """
    from tensorrt_llm._utils import mpi_comm, mpi_disabled

    if mpi_disabled():
        logger.warning(
            f"{_LOG_TAG} dispatch: no MPI communicator, so the ranks are not checked for agreement"
        )
        return None
    comm = mpi_comm()
    if comm.Get_size() != world_size:
        logger.warning(
            f"{_LOG_TAG} dispatch: the MPI communicator has {comm.Get_size()} ranks, not the "
            f"{world_size} of this model, so the ranks are not checked for agreement"
        )
        return None
    return comm.allgather(value)


def _rank_table(gathered: Sequence[Knobs]) -> str:
    knob_names = sorted({name for knobs in gathered for name, _ in knobs})
    lines = []
    for rank, knobs in enumerate(gathered):
        values = dict(knobs)
        knob_words = ", ".join(f"{name}={values.get(name, '<unset>')!r}" for name in knob_names)
        lines.append(f"  rank {rank}: knobs {knob_digest(knobs)} ({knob_words or 'none set'})")
    return "\n".join(lines)


def check_rank_uniform(
    knobs: Knobs, world_size: int, allgather: Optional[Allgather] = None
) -> None:
    """Raise on every rank unless every rank read the same knob values.

    Called once, at construction, on every rank, before the knobs are parsed:
    the allgather is itself a collective, so a rank must not raise before it.
    ``knobs`` is ``rank_uniform_inputs`` of the knobs. ``allgather`` defaults to
    the session's MPI communicator.

    Raises:
        RuntimeError: with one line per rank, when the ranks disagree.
    """
    gathered = (allgather or _world_allgather)(knobs, world_size)
    if gathered is None or all(peer == knobs for peer in gathered):
        return
    raise RuntimeError(
        f"{_LOG_TAG} the ranks read different knob values, so they would resolve different "
        "fast paths; a collective fast path would wait forever on the ranks that skip it. "
        f"Set the {KNOB_PREFIX}* variables, {', '.join(SHARED_ENV)}, "
        f"{MODELING_V2_VALIDATE_ENV} and {MODE_KEY} identically on every rank (before the "
        f"ranks start):\n{_rank_table(gathered)}"
    )


#: The fast-path state of a built model, module by module in construction order: per module,
#: its path and, per fast-path object attached to it, the attribute, the object's class and its
#: rank-independent flags.
AdoptedState = Tuple[Tuple[str, Tuple[Tuple[str, ...], ...]], ...]


def check_rank_uniform_adopted(
    state: AdoptedState, world_size: int, allgather: Optional[Allgather] = None
) -> None:
    """Raise on every rank unless every rank attached the same fast-path state to its modules.

    Called once, on every rank, after the fast paths are attached; a fast path that engaged on
    some ranks only would run its collective on those ranks alone.

    Raises:
        RuntimeError: naming the first module whose state differs, with one line per rank.
    """
    gathered = (allgather or _world_allgather)(state, world_size)
    if gathered is None or all(peer == state for peer in gathered):
        return
    depth = max(len(peer) for peer in gathered)
    first = next(
        i
        for i in range(depth)
        if len({peer[i] if i < len(peer) else None for peer in gathered}) > 1
    )
    lines = []
    for rank, peer in enumerate(gathered):
        lines.append(f"  rank {rank}: {peer[first] if first < len(peer) else '<no module>'}")
    raise RuntimeError(
        f"{_LOG_TAG} the ranks attached different fast-path state to their modules; a "
        "collective fast path would wait forever on the ranks that skip it. First module that "
        "differs:\n" + "\n".join(lines)
    )


# --------------------------------------------------------------------------------------------------
# Fallback policy
# --------------------------------------------------------------------------------------------------


class FastPathUnavailable(RuntimeError):
    """A step inside a fast path's shapes could not take it under ``modeling_v2='require'``."""


_lock = threading.Lock()
_require = [False]
_logged: set = set()
_missed: Dict[str, int] = {}


def set_require(require: bool) -> None:
    """Whether ``missed`` raises (``modeling_v2='require'``); set by ``resolve``."""
    _require[0] = bool(require)


def require_mode() -> bool:
    return _require[0]


def _log_once(level: Callable[[str], None], path: str, reason: str, message: str) -> None:
    key = (path, reason)
    with _lock:
        if key in _logged:
            return
        _logged.add(key)
    level(message)


def missed(path: str, rows: Optional[int], reason: str, *, per_call: bool = False) -> None:
    """``path`` takes this step's shape but cannot run: raise under require, else log once and count.

    A miss decided per call (``per_call``) never raises: a step that is already being served
    falls back, and the miss is counted and logged once. The caller falls back after it returns.
    """
    at = "" if rows is None else f" at {rows} rows"
    if _require[0] and not per_call:
        raise FastPathUnavailable(
            f"modeling_v2='require': the GLM-5.2 decode fast path {path!r} takes this step{at} "
            f"but cannot run: {reason}"
        )
    with _lock:
        _missed[path] = _missed.get(path, 0) + 1
    _log_once(
        logger.warning,
        path,
        reason,
        f"{_LOG_TAG} {path}/missed{at}: {reason}; falling back (logged once per reason)",
    )


def uncovered(path: str, reason: str) -> None:
    """The step is outside ``path``'s shapes: log once per (path, reason); the next tier runs."""
    _log_once(logger.info, path, reason, f"{_LOG_TAG} {path}/uncovered: {reason} (logged once)")


def missed_counts() -> Dict[str, int]:
    """Misses per fast path in this process: per-call misses in every mode, the others under
    ``auto`` (``require`` raises on them instead)."""
    with _lock:
        return dict(_missed)


def reset_fallback_state() -> None:
    """Forget the logged reasons, the miss counts and the installed plan."""
    with _lock:
        _logged.clear()
        _missed.clear()
    _INSTALLED[0] = None
    _VALIDATE[0] = False
    _TARGET[0] = DEFAULT_TARGET
    classify_decline.cache_clear()


# --------------------------------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------------------------------


class Phase(enum.Enum):
    """Which target takes a model step."""

    PREFILL = "prefill"
    DECODE = "decode"


_PHASE: List[Optional[Phase]] = [None]


class _Step:
    """The scope of one model step under a phase; restores the enclosing phase on exit."""

    __slots__ = ("_phase", "_previous")

    def __init__(self, phase: Phase) -> None:
        self._phase = phase
        self._previous: Optional[Phase] = None

    def __enter__(self) -> None:
        self._previous = _PHASE[0]
        _PHASE[0] = self._phase

    def __exit__(self, *exc_info: Any) -> None:
        _PHASE[0] = self._previous


def step(phase: Phase) -> _Step:
    """A context manager that runs the enclosed model step under ``phase``; scopes nest."""
    return _Step(phase)


def phase() -> Optional[Phase]:
    """The phase of the model step being run, or None outside a step."""
    return _PHASE[0]


# --------------------------------------------------------------------------------------------------
# Decline classes
# --------------------------------------------------------------------------------------------------


class Decline(enum.Enum):
    """What a fast path's decline means."""

    #: The path is turned off: neither a miss nor outside its shapes.
    DISABLED = "disabled"
    #: Outside the path's shapes (a layer it does not serve, a row count, a step kind).
    UNCOVERED = "uncovered"
    #: Decided once, at construction or in the warm-up, inside the path's layers, with a
    #: precondition absent: ``missed``, which raises under ``require``.
    MISSED = "missed"
    #: Decided per call, with a resource absent: counted and logged once, never raised.
    MISSED_PER_CALL = "missed per call"


_TARGET_LAYERS = frozenset(range(78))
_DRAFT_LAYERS = frozenset({78})
# The decoder layers that run the full indexer: the three dense layers and every fourth layer
# from layer 6. The other target layers reuse an earlier layer's top-k selection.
_FULL_INDEXER_LAYERS = frozenset((0, 1, 2) + tuple(range(6, 78, 4)))
_SHARED_INDEXER_LAYERS = _TARGET_LAYERS - _FULL_INDEXER_LAYERS

#: Per fast path and per layer-keyed capability key family (the key without its layer index),
#: the layers on which the path engages on every rank when this target serves its checkpoint.
#: A decline on a layer outside its target's cells is outside the path's shapes.
LAYER_CELLS: Mapping[str, Mapping[str, FrozenSet[int]]] = {
    "absorbed_query": {"layer": _TARGET_LAYERS},
    "draft_absorbed_query": {"layer": _DRAFT_LAYERS},
    "shared_indexer_absorbed_query": {"layer": _SHARED_INDEXER_LAYERS},
    # Only the shared-indexer layer that the last dense layer feeds.
    "fused_kv_indexer_projection": {"layer": frozenset({3})},
    "indexer_layer_front": {"target_layer": _FULL_INDEXER_LAYERS, "draft_layer": _DRAFT_LAYERS},
    # Boundaries into the shared-indexer layers that a MoE layer feeds.
    "deferred_norm": {"scope[shared]_into_layer": _SHARED_INDEXER_LAYERS - {3}},
    "deferred_norm_evict_hints": {"projection_layer": _SHARED_INDEXER_LAYERS - {3}},
    "sparse_mla_decode": {
        "layer": _TARGET_LAYERS | _DRAFT_LAYERS,
        "call:layer": _TARGET_LAYERS | _DRAFT_LAYERS,
    },
    "attention_output_tail": {"layer": _TARGET_LAYERS},
    "oproj_weight_staging": {"layer": _TARGET_LAYERS},
}

#: Reason fragments that turn a path off rather than decline it.
DISABLED_REASONS: Tuple[str, ...] = ("DISABLE",)
#: Reason fragments that name a layer kind or a composition the path does not serve.
LAYER_KIND_REASONS: Tuple[str, ...] = (
    "target layers only",
    "draft (MTP) layers only",
    "full indexer layer",
    "shared-indexer layer",
    "layers not selected (",
    "draft or unindexed layer",
    "not a target layer",
    "is a full-indexer layer",
    "not a DSA layer",
    "not MLA",
    "needs the post_attention_hop hop",
    "no post_attention_hop hop",
    "post_attention_hop without sparse_mla_decode",
)
#: Reason fragments (any case) of a per-call decline that name a missing resource.
RESOURCE_REASONS: Tuple[str, ...] = (
    "workspace",
    "compile failed",
    "not compiled",
    "not loaded",
    "not built",
    "failed",
    "unavailable",
    "not warmed up",
    "before CUDA graph capture",
    "packet does not match",
    "does not forward",
    "another rank",
    "MNNVL",
    "fabric",
)


def _any_of(fragments: Iterable[str], flags: int = 0) -> "re.Pattern[str]":
    return re.compile("|".join(map(re.escape, fragments)), flags)


_DISABLED_RE = _any_of(DISABLED_REASONS)
_LAYER_KIND_RE = _any_of(LAYER_KIND_REASONS)
_RESOURCE_RE = _any_of(RESOURCE_REASONS, re.IGNORECASE)


@dataclass(frozen=True)
class DeclineSite:
    """Capability keys of a fast path that are decided alike.

    ``key`` matches a whole key; a key that names a layer captures its index as ``layer``, and
    ``cells`` names the ``LAYER_CELLS`` family of those layers. A ``per_call`` site decides on
    every call; the others decide once, at construction or in the warm-up. ``uncovered``
    matches reasons that are outside the path's shapes at this site.
    """

    path: str
    key: "re.Pattern[str]"
    per_call: bool = False
    cells: Optional[str] = None
    uncovered: Optional["re.Pattern[str]"] = None


def _site(
    path: str,
    key: str,
    per_call: bool = False,
    cells: Optional[str] = None,
    uncovered: Optional[str] = None,
) -> DeclineSite:
    return DeclineSite(
        path,
        re.compile(key),
        per_call,
        cells,
        None if uncovered is None else re.compile(re.escape(uncovered)),
    )


_LAYER = r"(?P<layer>\d+)"
_EXPERT_PREFETCH = r"expert_prefetch\[.*\]"

#: Every capability key the fast paths log, by where and when it is decided.
DECLINE_KINDS: Tuple[DeclineSite, ...] = (
    _site("absorbed_query", "form"),
    _site("absorbed_query", f"layer{_LAYER}", cells="layer"),
    _site("draft_absorbed_query", f"layer{_LAYER}", cells="layer"),
    _site("shared_indexer_absorbed_query", f"layer{_LAYER}", cells="layer"),
    _site("fused_kv_indexer_projection", f"layer{_LAYER}", cells="layer"),
    _site("indexer_layer_front", f"target_layer{_LAYER}", cells="target_layer"),
    _site("indexer_layer_front", f"draft_layer{_LAYER}", cells="draft_layer"),
    _site("indexer_layer_front", "rope"),
    _site("indexer_layer_front", r"cp_indexer_decode_.+", per_call=True),
    _site("deferred_norm", "deferred_hop"),
    _site(
        "deferred_norm", rf"scope\[shared\]_into_layer{_LAYER}", cells="scope[shared]_into_layer"
    ),
    _site("deferred_norm", r"deferred_hop_routed\[.*\]"),
    _site("deferred_norm", f"projection_layer{_LAYER}", cells="projection_layer"),
    _site("deferred_norm_evict_hints", "deferred_hop"),
    _site("deferred_norm_evict_hints", r"deferred_hop_routed\[.*\]"),
    _site("deferred_norm_evict_hints", f"projection_layer{_LAYER}", cells="projection_layer"),
    _site("rope_kv_append", "rope|scatter_defer|bridge", per_call=True),
    _site("indexer_topk", "indexer_decode", per_call=True),
    _site("cross_rank_topk", "cp_indexer_decode", per_call=True),
    _site("cross_rank_topk_cluster_merge", "cp_indexer_decode", per_call=True),
    _site("cross_rank_topk_global_emit", "cp_indexer_decode", per_call=True),
    _site("sparse_mla_decode", f"layer{_LAYER}", cells="layer"),
    _site("sparse_mla_decode", r"request:.*", per_call=True),
    _site("sparse_mla_decode", f"call:layer{_LAYER}", per_call=True, cells="call:layer"),
    _site("attention_output_tail", f"layer{_LAYER}", cells="layer"),
    _site("attention_output_tail_prefetch", f"layer{_LAYER}", cells="layer"),
    _site("attention_output_tail_prefetch", "schedule"),
    _site("attention_output_into_hop", f"layer{_LAYER}", cells="layer"),
    _site("attention_output_into_hop", "into_hop_transport"),
    _site("oproj_weight_staging", f"layer{_LAYER}", cells="layer"),
    _site("post_attention_hop", r"post_attn_hop(:.*)?"),
    _site("hop_routing", "routing_hop"),
    _site("hop_routing", r"routed_moe_fork rows=.*", per_call=True),
    _site("hop_routing", _EXPERT_PREFETCH),
    _site("router_gemm", r"router:.*"),
    _site("routed_moe", r"rows=.*", per_call=True),
    _site("routed_moe", _EXPERT_PREFETCH),
    _site("routed_moe_into_hop", r"rows=.*", per_call=True),
    _site("routed_moe_into_hop", r"post_moe_routed_input.*"),
    _site("shared_expert_prefetch", r"routed rows=.*", per_call=True),
    _site("shared_expert_prefetch", _EXPERT_PREFETCH),
    _site("post_moe_hop", r"cluster_kernel(_compile)?|post_moe_hop\[.*\]"),
    _site("cross_layer_expert_prefetch", r"expert_prefetch|fused_predict|xc_prefetch|link\[.*\]"),
    _site("cross_layer_expert_prefetch", r"spec_experts(_declined)?", per_call=True),
    _site("dense_mlp", r"dense_mlp:.*", per_call=True),
    _site("draft_front", "draft_front"),
    _site("draft_front", r"draft_front_call(_fallback)?", per_call=True),
    _site("draft_routed_moe", r"rows=.*", per_call=True),
    _site("draft_live_rows", r"live_row_hop(:.*)?"),
    _site("draft_live_rows", r"live_row_hop_call(:.*)?|step0:.*", per_call=True),
    _site("draft_tail", r"draft_tail\[tp\d+\]"),
    _site("draft_tail", r"draft_tail\[(step: .*|graph|eager)\]", per_call=True),
    _site("draft_loop", r"(rows|draft_step|finalize|draft_loop):.*", per_call=True),
    _site("draft_loop", "draft_embedding"),
    _site("decode_prologue", r"target_(prologue|embedding)(\[.*\])?|take_embedding", per_call=True),
    _site("deferred_input_prep", r"prep(\[.*\])?", per_call=True),
    # Before the overlap executor loop declares its domain every call declines; the decision
    # is final only once the loop has declared it.
    _site(
        "deferred_input_prep",
        "loop_domain",
        uncovered="no overlap executor loop declared its domain",
    ),
    _site("deferred_input_prep", "engaged"),
    _site("sampler_store", "engaged"),
    _site("execution_stream_loop", "engaged"),
    _site("host_page_retention", "host_policy"),
)

_SITES_BY_PATH: Mapping[str, Tuple[DeclineSite, ...]] = {
    name: tuple(s for s in DECLINE_KINDS if s.path == name) for name in FAST_PATH_NAMES
}


@dataclass(frozen=True)
class TargetPaths:
    """What one target of the family serves of the fast paths, and where their code lives.

    ``served`` is the fast paths the target runs: a path outside it is off in every plan of the
    target, ``DISABLE`` may not name it and a knob only it reads is refused. ``layer_cells`` is
    ``LAYER_CELLS`` for the target's layers and shapes; ``uncovered`` lists, per path, reason
    fragments of declines that are outside the target's shapes at every site of the path.
    ``kernel_modules`` names the module of each path whose kernel package is the target's own;
    every other path's package is the target's (``KERNELS_PACKAGE``).

    Raises:
        ValueError: if a name is not a fast path, or a served path extends one the target does not
            serve.
    """

    served: FrozenSet[str] = frozenset(FAST_PATH_NAMES)
    layer_cells: Mapping[str, Mapping[str, FrozenSet[int]]] = field(
        default_factory=lambda: LAYER_CELLS
    )
    uncovered: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    kernel_modules: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        named = set(self.served) | set(self.layer_cells) | set(self.uncovered)
        unknown = sorted((named | set(self.kernel_modules)) - set(FAST_PATH_NAMES))
        if unknown:
            raise ValueError(f"{unknown} are not fast paths of the family")
        lacking = sorted(
            f"{name} extends {base}"
            for name in self.served
            for base in _BY_NAME[name].depends_on
            if base not in self.served
        )
        if lacking:
            raise ValueError(f"a served fast path extends one the target does not serve: {lacking}")


#: Every fast path, with the default tables.
DEFAULT_TARGET = TargetPaths()
# The tables of the installed plan's target.
_TARGET: List[TargetPaths] = [DEFAULT_TARGET]


def _decline_site(path: str, key: str) -> Optional[Tuple[DeclineSite, "re.Match[str]"]]:
    """The site of ``path`` whose key pattern matches all of ``key``, with the match."""
    for site in _SITES_BY_PATH.get(path, ()):
        match = site.key.fullmatch(key)
        if match is not None:
            return site, match
    return None


@functools.lru_cache(maxsize=4096)
def classify_decline(path: str, key: str, reason: str) -> Decline:
    """What a decline of fast path ``path`` at capability ``key`` for ``reason`` means.

    In order: a reason that turns the path off is ``DISABLED``; a key no site of the path
    matches is ``UNCOVERED``; a per-call decline is ``MISSED_PER_CALL`` when its reason names a
    missing resource and ``UNCOVERED`` otherwise; a decline decided once is ``UNCOVERED`` when
    its reason names a layer kind or a composition the path does not serve, or its layer is
    outside the path's cells, and ``MISSED`` otherwise. The cells and the target's uncovered
    reasons are the installed plan's target's (``TargetPaths``). Memoized: a repeated decline
    costs one lookup.
    """
    if _DISABLED_RE.search(reason):
        return Decline.DISABLED
    found = _decline_site(path, key)
    if found is None:
        return Decline.UNCOVERED
    site, match = found
    if site.uncovered is not None and site.uncovered.search(reason):
        return Decline.UNCOVERED
    if any(fragment in reason for fragment in _TARGET[0].uncovered.get(path, ())):
        return Decline.UNCOVERED
    if site.per_call:
        return Decline.MISSED_PER_CALL if _RESOURCE_RE.search(reason) else Decline.UNCOVERED
    if _LAYER_KIND_RE.search(reason):
        return Decline.UNCOVERED
    if site.cells is not None:
        cells = _TARGET[0].layer_cells.get(path, {}).get(site.cells, frozenset())
        if int(match["layer"]) not in cells:
            return Decline.UNCOVERED
    return Decline.MISSED


def _record_decline(path: str, key: str, reason: str) -> None:
    kind = classify_decline(path, key, reason)
    if kind is Decline.MISSED:
        missed(path, None, f"{key}: {reason}" if reason else key)
    elif kind is Decline.MISSED_PER_CALL:
        missed(path, None, reason or key, per_call=True)


# --------------------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------------------

#: The parts of the decode step's input preparation, each a fast path of its own: the step-input
#: stager, the sampler's store and the overlap loop on the execution stream.
PART_PATHS: Mapping[str, str] = {
    "prep": "deferred_input_prep",
    "store": "sampler_store",
    "stream": "execution_stream_loop",
}
_PART_ONLY_PATHS = frozenset(PART_PATHS.values()) - {PART_PATHS["prep"]}

#: Every fast path except the parts that only ``deferred_input_prep`` brings along.
SELECTOR_PATHS: Tuple[str, ...] = tuple(n for n in FAST_PATH_NAMES if n not in _PART_ONLY_PATHS)

#: The family's kernel packages, one directory per fast path under ``kernels/``.
KERNELS_PACKAGE = f"{__package__}.kernels"
#: The fast paths whose code lives in the kernel directory of the path they extend.
KERNEL_DIRECTORY: Mapping[str, str] = {
    "draft_absorbed_query": "absorbed_query",
    "shared_indexer_absorbed_query": "absorbed_query",
    "cross_rank_topk": "indexer_topk",
    "attention_output_tail_prefetch": "attention_output_tail",
    "attention_output_into_hop": "attention_output_tail",
    "routed_moe_into_hop": "routed_moe",
    "sampler_store": "deferred_input_prep",
    "execution_stream_loop": "deferred_input_prep",
}


def _bases() -> Dict[str, FrozenSet[str]]:
    """Every path each path extends, directly or transitively."""
    closure: Dict[str, FrozenSet[str]] = {}

    def visit(name: str) -> FrozenSet[str]:
        if name not in closure:
            bases = set()
            for base in _BY_NAME[name].depends_on:
                bases |= {base} | visit(base)
            closure[name] = frozenset(bases)
        return closure[name]

    for name in FAST_PATH_NAMES:
        visit(name)
    return closure


#: The paths each path has no effect without (the transitive closure of ``depends_on``).
REQUIRED_FEATURES: Mapping[str, FrozenSet[str]] = {
    name: bases for name, bases in _bases().items() if bases
}

# Numerical variants (a variant has its own name). Per fast path with variants: the knob that
# selects one (None: a single variant) and the variant names, default first. A path without an
# entry has one numerical form, equal to the generic path's or held to a frozen error budget.
NUMERICAL_VARIANTS: Dict[str, Tuple[Optional[str], Tuple[str, ...]]] = {
    # Exact top-k over each page owner's indexer K, identical on every rank.
    "cross_rank_topk": (None, ("rank-consistent",)),
    # The cluster merge's rows are byte-identical to the cross-rank top-k's (same set and order).
    "cross_rank_topk_cluster_merge": (None, ("rank-consistent",)),
    # Its id rows are the cluster merge's; its pool and copy rows are the values of the
    # rope/KV-append top-k bridge.
    "cross_rank_topk_global_emit": (None, ("rank-consistent",)),
    # What the routed experts write into the post-MoE hop: fp32 sums the top-k contributions in
    # another order (nondeterministic from run to run); identity is bit-exact.
    "routed_moe_into_hop": (ACCUMULATION_ENV, ("fp32", "identity")),
    # The next layer's RMSNorm deferred into its projection (changes rounding), only on boundaries
    # into shared-indexer layers: full-indexer layers keep the post-MoE hop's normalization.
    "deferred_norm": (None, ("shared",)),
    # One projection launch for the fused-A GEMM and the indexer key/weights GEMM: kva keeps the
    # fused-A order (bit-identical); kw (full-indexer and draft layers) takes the same mma.sync
    # order instead of the cuBLAS split-K GEMM (rounds differently). This target runs it on the
    # dense-fed shared-indexer layer, which computes kva only.
    "fused_kv_indexer_projection": (None, ("kw-fused-a-order",)),
}

#: Prefixes of the set variables the record line lists (``side_selectors``).
SIDE_SELECTOR_PREFIXES = (KNOB_PREFIX,)
#: Shared TensorRT-LLM switches that the kernels also read.
SHARED_ENV = ("TLLM_TEST_MNNVL", "TRTLLM_ENABLE_PDL", "TRTLLM_FORCE_MNNVL_AR")


def feature_variants(
    paths: Iterable[str], environ: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """The numerical variant of every path in ``paths`` that has some.

    Raises:
        ValueError: if a knob names an unknown variant of a path in ``paths``.
    """
    env = os.environ if environ is None else environ
    variants = {}
    for path in sorted(paths):
        if path not in NUMERICAL_VARIANTS:
            continue
        name, choices = NUMERICAL_VARIANTS[path]
        raw = env.get(name) if name else None
        value = (raw or "").strip().lower() or choices[0]
        if value not in choices:
            raise ValueError(
                f"{name}={raw!r} is not a {path} variant; known variants are {list(choices)} "
                f"(unset selects {choices[0]!r})"
            )
        variants[path] = value
    return variants


def side_selectors(environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Every set knob of the family."""
    env = os.environ if environ is None else environ
    return {name: env[name] for name in sorted(env) if name.startswith(SIDE_SELECTOR_PREFIXES)}


def shared_env(environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The set shared TensorRT-LLM switches that the kernels read."""
    env = os.environ if environ is None else environ
    return {name: env[name] for name in SHARED_ENV if name in env}


def feature_record(paths: Iterable[str], environ: Optional[Mapping[str, str]] = None) -> str:
    """The one-line record of a selection.

    ``paths=`` lists every path as ``path`` or ``path[variant]``; ``side_selectors=`` and
    ``shared_env=`` are JSON objects of the set variables.

    Raises:
        ValueError: as ``feature_variants``.
    """
    variants = feature_variants(paths, environ)
    listed = ",".join(f"{p}[{variants[p]}]" if p in variants else p for p in sorted(paths))

    def _json(values: Dict[str, str]) -> str:
        return json.dumps(values, sort_keys=True, separators=(",", ":"))

    return (
        f"paths={listed or '<none>'} "
        f"side_selectors={_json(side_selectors(environ))} "
        f"shared_env={_json(shared_env(environ))}"
    )


def _log_info(message: str) -> None:
    logger.info(message)


# The fast paths of the installed plan; None while no plan is installed.
_INSTALLED: List[Optional[FrozenSet[str]]] = [None]
# Whether the installed plan runs the fast paths' diagnostics.
_VALIDATE: List[bool] = [False]


def install(plan: Plan, environ: Optional[Mapping[str, str]] = None) -> None:
    """Answer ``is_enabled`` from ``plan`` from now on, and log its record line once.

    Raises:
        ValueError: as ``feature_variants``, for a knob that names an unknown variant of an
            enabled path.
    """
    paths = frozenset(n for n in FAST_PATH_NAMES if plan.enabled(n))
    record = feature_record(paths, environ)
    _INSTALLED[0] = paths
    _VALIDATE[0] = plan.validate
    _TARGET[0] = DEFAULT_TARGET if plan.target is None else plan.target
    classify_decline.cache_clear()
    _log_info(f"{_LOG_TAG} dispatch: record {record}")


def installed() -> Optional[FrozenSet[str]]:
    """The fast paths of the installed plan, or None."""
    return _INSTALLED[0]


def enabled_features() -> FrozenSet[str]:
    """The fast paths that run in this process: the installed plan's, else none."""
    plan_paths = _INSTALLED[0]
    return plan_paths if plan_paths is not None else frozenset()


def validate_mode() -> bool:
    """Whether the installed plan runs the fast paths' diagnostics (False without a plan)."""
    return _VALIDATE[0]


def is_enabled(path: str) -> bool:
    """Whether fast path ``path`` runs in this process.

    Raises:
        ValueError: if ``path`` is not a fast path of the family.
    """
    if path not in _BY_NAME:
        raise ValueError(f"{path!r} is not a fast path of this target")
    return path in enabled_features()


# Entry points per fast path. The kernel package of a path registers them when it is imported;
# a call site looks one up with ``get_impl`` only after ``is_enabled`` returned True.
_REGISTRY: Dict[str, Dict[str, Callable]] = {}

# Per-rank capability decisions, logged once per (path, key).
_CAPABILITY_LOGGED: Dict[Tuple[str, str], bool] = {}


def register(path: str, name: str, fn: Callable) -> Callable:
    """Register ``fn`` as entry point ``name`` of fast path ``path``.

    Raises:
        ValueError: if ``path`` is not a fast path of the family.
    """
    if path not in _BY_NAME:
        raise ValueError(f"{path!r} is not a fast path of this target")
    _REGISTRY.setdefault(path, {})[name] = fn
    return fn


def get_impl(path: str, name: str) -> Callable:
    """Import the kernel package of ``path`` on first use and return its entry ``name``.

    The package is the installed plan's target's own (``TargetPaths.kernel_modules``) or the
    target's.

    Raises:
        KeyError: if the package does not register ``name`` for ``path``.
    """
    if path not in _REGISTRY:
        module = _TARGET[0].kernel_modules.get(path)
        importlib.import_module(module or f"{KERNELS_PACKAGE}.{KERNEL_DIRECTORY.get(path, path)}")
    entries = _REGISTRY.get(path, {})
    if name not in entries:
        raise KeyError(
            f"fast path {path!r} does not register {name!r}; registered: {sorted(entries)}"
        )
    return entries[name]


def log_capability(path: str, key: str, eligible: bool, reason: str = "") -> bool:
    """Log a per-rank capability decision once per (path, key) and return ``eligible``.

    Every decline, logged or not, is also classified (``classify_decline``): a miss decided
    once goes to ``missed``, which raises under ``require``; a miss decided per call is counted
    and logged once; a decline outside the path's shapes or of a path turned off adds nothing
    to the capability line.
    """
    if (path, key) not in _CAPABILITY_LOGGED:
        _CAPABILITY_LOGGED[(path, key)] = eligible
        state = "engaged" if eligible else "native fallback"
        suffix = f" ({reason})" if reason else ""
        _log_info(f"{_LOG_TAG} {path}/{key}: {state}{suffix}")
    if not eligible:
        _record_decline(path, key, reason)
    return eligible
