# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which fast paths a pipeline stage of this target is built with, decided once at construction.

``FAST_PATHS`` names every lever of the target: the context-step paths a stage at TP1 runs.
A composition (``TargetPaths``) is the set served at one ``CompositionKey`` (the GPU
architecture and the tensor-parallel size), with the pipeline topology it is written for
(``StageTopology``: four stages on the four GPUs of one node). It applies to one deployment
only: a disaggregated context server (``MUSESPARK_OPEN_LEVER__ROLE=ctx``) on an NVFP4
checkpoint, whose split puts the same number of full-attention layers on every stage
(``stages.partition_refusal``), either without speculative decoding or with a DFlash
drafter whose context attention is the paged TRTLLM backend (``drafter_refusal``). Every
other deployment the routing sends here (an aggregated or generation server, another
split, another GPU, another speculative decoding) has no composition: it runs the built-in
stage forward under ``modeling_v2="auto"`` (``NO_PATHS``) and raises under ``require``. No
decode, sequence-parallel, hop, transport or publish path exists in this target, so none
can be planned.

With a DFlash drafter (``DrafterContext``) the context server writes the drafter's prompt
context and never drafts: the captures of the drafter's target layers are carried along
the pipeline to the last stage, which projects them and appends the drafter K/V into the
draft layers of the shared paged KV cache (``kernels/dflash_context``). That is part of
the composition, not a lever: no knob turns it off, and the drafter arena is
``levers.Arena.PAGED``.

The lever table is the TEP target's resolver (``levers.resolve``) over this target's
registry: the same knob grammar (``MUSESPARK_OPEN_LEVER_<LEVER>``), the same rules for the
levers both targets serve (``levers.LEVERS``), and every TEP fast path name known to the
grammar, so a TEP-only lever knob set ``on`` raises ("does not serve") and ``off`` or
``auto`` warns. The drafter arena is ``levers.Arena.NONE`` without a drafter and
``levers.Arena.PAGED`` with one. Structural rules (``structural_rules``) leave out what a
stage's head count cannot run.

Agreement over the pipeline stages (the PP group; the routing serves ``world_size ==
pp_size``, tp 1). ``agreed_composition`` agrees the composition decision itself first, in one
collective every stage enters whatever its own decision, so a stage whose role environment,
device or drafter differed cannot leave the others waiting in a collective it never
reaches: the stages then run the built-in stage alike (raise alike under ``require``).
``compose`` then agrees the plan, so every stage runs the same lever set or every stage
raises the same error. Installers check and build without changing a layer, agree the
outcome (``agreed_install``), and only then change the stage's layers, so another stage's
refusal leaves this stage's model as built.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple, TypeVar

from tensorrt_llm._torch._experimental.modeling_v2._router_index import ModelingV2Mode
from tensorrt_llm.logger import logger

from . import stages as _stages
from ._tep import levers as _levers
from ._tep import module as _tep_module

#: The log tag of this target.
TAG = "[musespark_open pp]"

#: GPUs per node: one compute tray of an NVL72 rack.
GPUS_PER_NODE = 4

#: The per-rank (query, KV) head counts of a stage (all of the model's heads at TP1), which the
#: stage's multi-instance context FMHA install serves (``kernels/fmk_pure_context_fmha``).
FMK_STAGE_HEADS = (128, 16)


@dataclasses.dataclass(frozen=True)
class FastPath:
    """One lever of this target: what it changes, the levers it builds on, and when it installs.

    ``hook`` is ``construct`` (the shell's constructor, on the built model) or ``load``
    (after the stage's weights load).
    """

    name: str
    summary: str
    hook: str
    depends_on: Tuple[str, ...] = ()


#: Every fast path of this target. The names are the TEP target's, for the same ops at
#: the stage's full (unsliced) shapes, so one knob names one op on both targets; the two
#: stage MoE swaps (``context_router_input``, ``context_finalize_bf16``) run kernels the TEP
#: target runs only inside its sequence-parallel forward (``context_sp_router_input`` and
#: ``context_sequence_parallel``), so they have names of their own.
FAST_PATHS: Tuple[FastPath, ...] = (
    FastPath(
        "context_prep",
        "context steps' pre-attention prep in one launch per layer at the stage's 128 query / 16 "
        "KV heads, padded graph rows zeroed",
        "construct",
    ),
    FastPath(
        "context_prep_rows_vec",
        "context steps' pre-attention prep on the vectorized rows kernel",
        "construct",
        ("context_prep",),
    ),
    FastPath(
        "context_fp8_dense",
        "the stage's dense projections of context calls of MIN_ROWS rows or more on the Rubin "
        "MXFP8 GEMM, at TP1 tactics (lossy)",
        "load",
    ),
    FastPath(
        "prefill_cutedsl_experts",
        "the stage's NVFP4 experts of 4096..8192-row calls on the Rubin CuTe DSL GEMMs over all "
        "256 experts",
        "load",
    ),
    FastPath(
        "fmk_pure_context_fmha",
        "the full-attention context FMHA of pure context steps on the multi-instance paged "
        "context FMHA",
        "construct",
    ),
    FastPath(
        "context_router_input",
        "context calls' router FP32 input written by the attention-side residual join, in "
        "place of the router's BF16 -> FP32 copy (the TEP sequence-parallel join, TP1)",
        "construct",
    ),
    FastPath(
        "context_finalize_bf16",
        "context calls' MoE finalize storing BF16 from its sparse-slot kernel, in place of an "
        "FP32 store and a cast (the TEP sequence-parallel finalize, TP1)",
        "construct",
    ),
)

#: The name of every fast path.
FAST_PATH_NAMES: FrozenSet[str] = frozenset(path.name for path in FAST_PATHS)

_BY_NAME: Dict[str, FastPath] = {path.name: path for path in FAST_PATHS}

#: Every lever of this target and the levers it builds on, in ``FAST_PATHS`` order.
DEPENDS_ON: Mapping[str, Tuple[str, ...]] = {path.name: path.depends_on for path in FAST_PATHS}


def closure(names: Iterable[str]) -> FrozenSet[str]:
    """``names`` and every fast path they build on, transitively.

    Raises:
        KeyError: for a name that is not a fast path of this target.
    """
    seen = set()
    stack = list(names)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(_BY_NAME[name].depends_on)
    return frozenset(seen)


@functools.lru_cache(maxsize=None)
def known_levers() -> Mapping[str, Tuple[str, ...]]:
    """The registry the knob grammar reads: this target's levers (``DEPENDS_ON``), then every
    other TEP fast path name, served nowhere here."""
    tep_names = sorted(_tep_module("dispatch").FAST_PATH_NAMES - FAST_PATH_NAMES)
    return {**DEPENDS_ON, **{name: () for name in tep_names}}


def stage_snapshot(snapshot: Mapping[str, str], served: Iterable[str]) -> Dict[str, str]:
    """``snapshot`` with a knob for each served lever of this target that has no row in the TEP
    target's table (``levers.LEVERS``).

    Such a lever is on wherever the stage serves it, and off under
    ``MUSESPARK_OPEN_LEVER__DEFAULT=off``; its own knob, when set, wins.
    """
    value = (
        _levers.OFF
        if snapshot.get(_levers.DEFAULT_KNOB, "").strip().lower() == _levers.OFF
        else _levers.ON
    )
    resolved = dict(snapshot)
    for name in sorted(set(served) - set(_levers.LEVERS)):
        resolved.setdefault(_levers.knob_name(name), value)
    return resolved


@dataclasses.dataclass(frozen=True)
class StageTopology:
    """Where the stages sit: ``pp_size`` stages over ``nodes`` nodes of four GPUs, tp 1."""

    description: str
    pp_size: int
    nodes: int

    def holds(self, mapping: Any) -> bool:
        """Whether ``mapping`` is ``pp_size`` one-rank stages over ``nodes`` nodes of four."""
        if getattr(mapping, "gpus_per_node", None) != GPUS_PER_NODE:
            return False
        if int(getattr(mapping, "pp_size", 1)) != self.pp_size:
            return False
        if int(getattr(mapping, "tp_size", 1)) != 1:
            return False
        group = list(getattr(mapping, "pp_group", ()) or ())
        return (
            len(group) == self.pp_size
            and len({rank // GPUS_PER_NODE for rank in group}) == self.nodes
        )


#: Four stages on the four GPUs of one node.
ONE_NODE_PP4 = StageTopology("four pipeline stages on one node of 4", 4, 1)


def describe_topology(mapping: Any) -> str:
    """The topology facts ``StageTopology.holds`` reads, in words."""
    if mapping is None:
        return "no mapping"
    group = list(getattr(mapping, "pp_group", ()) or ())
    nodes = len({rank // GPUS_PER_NODE for rank in group}) if group else 0
    return (
        f"{getattr(mapping, 'gpus_per_node', None)} GPUs per node, pp {mapping.pp_size} over "
        f"{nodes} node(s), tp {mapping.tp_size}, pp_partition "
        f"{getattr(mapping, 'pp_partition', None)}"
    )


#: What a composition is keyed by: the GPU architecture as an SM number and the
#: tensor-parallel size.
CompositionKey = Tuple[int, int]


def sm_of(compute_capability: Optional[Tuple[int, int]]) -> Optional[int]:
    """The SM number of ``compute_capability`` (``(10, 7)`` -> 107), or None without one."""
    if compute_capability is None:
        return None
    major, minor = compute_capability
    return 10 * int(major) + int(minor)


@dataclasses.dataclass(frozen=True)
class Precondition:
    """A public setting the fast paths are written against, and how a built stage shows it.

    ``holds(text_model, model_config)`` reads the built text model, its config, or the public
    helper the built-in model reads the setting through. ``needed_by`` names the fast paths
    written against it; empty means every composition.
    """

    requirement: str
    holds: Callable[[Any, Any], bool]
    needed_by: Tuple[str, ...] = ()

    def needed(self, paths: FrozenSet[str]) -> bool:
        """Whether a stage built with ``paths`` needs this precondition."""
        return not self.needed_by or bool(set(self.needed_by) & set(paths))


@dataclasses.dataclass(frozen=True)
class TargetPaths:
    """A composition: the fast paths served at one ``CompositionKey``, closed under
    ``depends_on``, the topology they are written for, and the composition's own
    preconditions.

    Raises:
        ValueError: at construction, for a name that is not a fast path or a fast path
            whose dependency is not served.
    """

    served: FrozenSet[str]
    topology: StageTopology
    preconditions: Tuple[Precondition, ...] = ()

    def __post_init__(self) -> None:
        unknown = sorted(set(self.served) - FAST_PATH_NAMES)
        if unknown:
            raise ValueError(f"{TAG} not fast paths: {unknown}")
        missing = sorted(closure(self.served) - set(self.served))
        if missing:
            raise ValueError(f"{TAG} a served fast path builds on unserved {missing}")


@dataclasses.dataclass(frozen=True)
class Plan:
    """The levers a stage is built with: a whole composition's table, or none (``NO_PATHS``).

    ``levers`` is the table ``compose`` resolved (None for ``NO_PATHS`` and for a plan built
    from names); ``stage`` is this rank's pipeline stage, ``rank0`` whether it logs the
    per-deployment lines.

    Raises:
        ValueError: at construction, for a name that is not a fast path or a fast path
            whose dependency is not in the plan.
    """

    paths: FrozenSet[str] = frozenset()
    levers: Optional[Any] = dataclasses.field(default=None, compare=False)
    stage: int = dataclasses.field(default=0, compare=False)
    rank0: bool = dataclasses.field(default=True, compare=False)
    #: Lever -> (installed, why not), as the installers reported (``installed``).
    reports: Dict[str, Tuple[bool, str]] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        unknown = sorted(set(self.paths) - FAST_PATH_NAMES)
        if unknown:
            raise ValueError(f"{TAG} not fast paths: {unknown}")
        missing = sorted(closure(self.paths) - set(self.paths))
        if missing:
            raise ValueError(f"{TAG} a fast path of the plan builds on {missing}, not in it")

    def enabled(self, path: str) -> bool:
        """Whether ``path`` is one of the plan's fast paths.

        Raises:
            KeyError: for a name that is not a fast path of this target.
        """
        if path not in FAST_PATH_NAMES:
            raise KeyError(f"{TAG} not a fast path: {path!r}")
        return path in self.paths

    def param(self, name: str, key: str) -> Any:
        """Lever ``name``'s parameter ``key``: as resolved, else its declared default.

        Raises:
            KeyError: for a name that is not a fast path, or a parameter ``name`` does not
                declare.
        """
        if name not in FAST_PATH_NAMES:
            raise KeyError(f"{TAG} not a fast path: {name!r}")
        if self.levers is not None:
            return self.levers.param(name, key)
        return _levers.default_param(name, key)

    def installed(self, name: str, ok: bool, why: str = "") -> None:
        """An installer's report on lever ``name``, agreed over the stages.

        A refusal of a lever its knob set ``on`` raises; any other refusal is a warning.

        Raises:
            ValueError: for a refusal of a lever set ``on`` by its knob.
        """
        self.reports[name] = (bool(ok), why)
        if ok:
            return
        decision = self.levers.decisions.get(name) if self.levers is not None else None
        message = f"{TAG} lever {name} is not installed: {why}"
        if decision is not None and decision.source == "knob":
            raise ValueError(f"{message} ({_levers.knob_name(name)}=on)")
        if self.rank0:
            logger.warning(f"{message}; the stage runs without it")

    def report(self) -> str:
        """The installed lever table as one JSON line (``levers.LeverTable.as_json``)."""
        if self.levers is None:
            return "{}"
        return self.levers.as_json({name: ok for name, (ok, _) in self.reports.items()})

    def __bool__(self) -> bool:
        return bool(self.paths)

    def describe(self) -> str:
        """The plan's fast paths in ``FAST_PATHS`` order, for the log line."""
        names = [path.name for path in FAST_PATHS if path.name in self.paths]
        return f"{len(names)} fast paths: {', '.join(names)}" if names else "no fast path"


#: The plan of a deployment with no composition.
NO_PATHS = Plan()


def plan_of(*names: str) -> Plan:
    """The plan of ``names`` and every fast path they build on."""
    return Plan(closure(names))


@dataclasses.dataclass(frozen=True)
class StageDeployment(_levers.Deployment):
    """The ``levers.Deployment`` of a pipeline stage: also its pipeline size and the
    (query, KV) heads every rank runs, which the structural rules read."""

    pp: int = 1
    heads: Tuple[int, int] = (0, 0)

    def describe(self) -> str:
        return (
            f"sm{self.sm} pp{self.pp} tp{self.tp}, {self.role.value} ({self.role_source}), "
            f"max_batch_size {_count(self.max_batch_size)}, max_num_tokens "
            f"{_count(self.max_num_tokens)}, {self.heads[0]} / {self.heads[1]} heads per rank, "
            f"{self.arena.value} arena"
        )

    def as_dict(self) -> dict:
        out = super().as_dict()
        out.update(pp=self.pp, heads=list(self.heads))
        return out


def _count(value: Optional[int]) -> str:
    return "unknown" if value is None else str(value)


def structural_rules(served: FrozenSet[str], deployment: StageDeployment) -> Mapping[str, str]:
    """What a stage's shape excludes: each served lever left out, and why. Pure.

    * ``fmk_pure_context_fmha``'s install on a stage serves ``FMK_STAGE_HEADS`` per rank
      (the kernel is built and its work planned at those head counts); a stage that ran
      other head counts would not get it.
    """
    exclusions: Dict[str, str] = {}
    if "fmk_pure_context_fmha" in served and tuple(deployment.heads) != FMK_STAGE_HEADS:
        exclusions["fmk_pure_context_fmha"] = (
            f"its stage install serves {FMK_STAGE_HEADS[0]} query / {FMK_STAGE_HEADS[1]} KV "
            f"heads per rank, and this stage runs {deployment.heads[0]} / {deployment.heads[1]}"
        )
    return exclusions


def nvfp4_checkpoint(model_config: Any) -> bool:
    """Whether the checkpoint is NVFP4 (``hf_quant_config.json`` ``quant_algo``)."""
    from tensorrt_llm.quantization.mode import QuantAlgo

    quant_config = getattr(model_config, "quant_config", None)
    return getattr(quant_config, "quant_algo", None) == QuantAlgo.NVFP4


def _layers_of(model_config: Any) -> Tuple[int, int]:
    pretrained = model_config.pretrained_config
    return int(pretrained.num_hidden_layers), int(pretrained.every_n_layers_nope)


def _split_refusal(model_config: Any) -> Optional[str]:
    mapping = model_config.mapping
    num_layers, every_n = _layers_of(model_config)
    return _stages.partition_refusal(
        num_layers, every_n, int(mapping.pp_size), getattr(mapping, "pp_partition", None)
    )


#: The speculative decoding a context server's stages serve (``decoding_type``).
DRAFTER_DECODING_TYPE = "DFlash"
#: The drafter context attention backend they serve: the paged one, whose pages are draft
#: layers of the target's KV cache manager, which the cache transceiver sends with the
#: target layers.
DRAFTER_ATTENTION_BACKEND = "TRTLLM"


@dataclasses.dataclass(frozen=True)
class DrafterContext:
    """The DFlash drafter context the stages of a context server write, and never draft from.

    ``target_layers`` are the target layers whose residual the drafter reads
    (``target_layer_ids``, ascending); ``draft_len`` the drafter's ``max_draft_len``.
    The context lives in pages of the draft layers (``levers.Arena.PAGED``), on the last
    stage.
    """

    target_layers: Tuple[int, ...]
    draft_len: int

    @property
    def arena(self) -> Any:
        """The drafter context arena: ``levers.Arena.PAGED``."""
        return _levers.Arena.PAGED

    def key(self) -> str:
        """The drafter as the stages compare it (``agreed_composition``)."""
        return (
            f"{DRAFTER_DECODING_TYPE} {DRAFTER_ATTENTION_BACKEND} {self.arena.value} arena, "
            f"target layers {list(self.target_layers)}, draft length {self.draft_len}"
        )


def drafter_refusal(model_config: Any) -> Optional[str]:
    """Why this target does not serve the deployment's speculative decoding, or None. Pure.

    None without speculative decoding, and for a DFlash drafter on the
    ``DRAFTER_ATTENTION_BACKEND`` with at least one draft token and resolved target layers
    of the target model; the reason otherwise.
    """
    spec = getattr(model_config, "spec_config", None)
    if spec is None:
        return None
    decoding = getattr(spec, "decoding_type", None)
    if decoding != DRAFTER_DECODING_TYPE:
        return (
            f"speculative decoding {decoding or type(spec).__name__} (a stage serves "
            f"{DRAFTER_DECODING_TYPE} only)"
        )
    backend = getattr(spec, "attention_backend", None)
    if backend != DRAFTER_ATTENTION_BACKEND:
        return (
            f"the {DRAFTER_DECODING_TYPE} drafter attention backend is {backend}; a stage writes "
            f"the drafter context into paged draft KV, the {DRAFTER_ATTENTION_BACKEND} backend's"
        )
    draft_len = getattr(spec, "max_draft_len", None)
    if not isinstance(draft_len, int) or draft_len < 1:
        return f"the {DRAFTER_DECODING_TYPE} max_draft_len is {draft_len}; the context store needs 1 or more"
    layers = getattr(spec, "target_layer_ids", None)
    if not layers:
        return f"the {DRAFTER_DECODING_TYPE} target_layer_ids are not resolved"
    num_layers, _ = _layers_of(model_config)
    ids = [int(layer) for layer in layers]
    if len(set(ids)) != len(ids) or any(not 0 <= layer < num_layers for layer in ids):
        return (
            f"the {DRAFTER_DECODING_TYPE} target_layer_ids {ids} are not distinct layers of the "
            f"{num_layers}-layer decoder"
        )
    return None


def drafter_of(model_config: Any) -> Optional[DrafterContext]:
    """The ``DrafterContext`` of a deployment ``drafter_refusal`` accepts; None without a drafter.

    Raises:
        ValueError: for speculative decoding ``drafter_refusal`` refuses.
    """
    spec = getattr(model_config, "spec_config", None)
    if spec is None:
        return None
    reason = drafter_refusal(model_config)
    if reason is not None:
        raise ValueError(f"{TAG} {reason}")
    return DrafterContext(
        target_layers=tuple(sorted(int(layer) for layer in spec.target_layer_ids)),
        draft_len=int(spec.max_draft_len),
    )


def composition(
    model_config: Any,
    compositions: Mapping[CompositionKey, TargetPaths],
    compute_capability: Optional[Tuple[int, int]],
    snapshot: Mapping[str, str],
) -> Optional[TargetPaths]:
    """The composition this stage is served with, or None. Pure: reads its arguments only.

    A disaggregated context server (the role hint ``levers.ROLE_KNOB`` is ``ctx`` in
    ``snapshot``) without speculative decoding or with a drafter ``drafter_refusal``
    accepts, on an NVFP4 checkpoint, at a key and topology of ``compositions``, whose split
    ``stages.partition_refusal`` accepts.
    """
    if _levers.role_hint(snapshot) != _levers.Role.CTX.value:
        return None
    if drafter_refusal(model_config) is not None:
        return None
    mapping = getattr(model_config, "mapping", None)
    if mapping is None or not nvfp4_checkpoint(model_config):
        return None
    served = compositions.get((sm_of(compute_capability), int(mapping.tp_size)))
    if served is None or not served.topology.holds(mapping):
        return None
    if _split_refusal(model_config) is not None:
        return None
    return served


def served_compositions(compositions: Mapping[CompositionKey, TargetPaths]) -> str:
    """The compositions this target serves, in words."""
    keys = (
        ", ".join(
            f"sm{sm} tp {tp} as {compositions[sm, tp].topology.description}"
            for sm, tp in sorted(compositions)
        )
        or "none"
    )
    return (
        f"{keys}, on a disaggregated context server ({_levers.ROLE_KNOB}=ctx) without "
        f"speculative decoding or with a {DRAFTER_DECODING_TYPE} drafter on the "
        f"{DRAFTER_ATTENTION_BACKEND} attention backend, an NVFP4 checkpoint, and a split "
        "with as many full-attention layers on every stage"
    )


def unserved_reason(
    model_config: Any,
    compositions: Mapping[CompositionKey, TargetPaths],
    compute_capability: Optional[Tuple[int, int]],
    snapshot: Mapping[str, str],
) -> str:
    """Why ``composition`` found none for this deployment, naming every fact it read."""
    mapping = getattr(model_config, "mapping", None)
    spec = getattr(model_config, "spec_config", None)
    arch = "no CUDA device" if compute_capability is None else f"sm{sm_of(compute_capability)}"
    split = _split_refusal(model_config) if mapping is not None else "no mapping"
    if spec is None:
        speculative = "off"
    else:
        refused = drafter_refusal(model_config)
        speculative = type(spec).__name__ + (f" ({refused})" if refused else "")
    return (
        f"{arch}, {describe_topology(mapping)}, role hint {_levers.role_hint(snapshot)}, "
        f"speculative decoding {speculative}, NVFP4 "
        f"checkpoint {nvfp4_checkpoint(model_config)}, split {split or 'served'}; the fast "
        "paths are served at " + served_compositions(compositions)
    )


@dataclasses.dataclass(frozen=True)
class Decision:
    """The composition the stages are served with, as they agreed (``agreed_composition``).

    ``served`` is None when no stage has a composition, or when the stages disagree
    (``split``); ``reason`` says why none: this stage's unserved reason, or each stage's
    decision. ``drafter`` is the DFlash drafter context the stages write with the
    composition, None without speculative decoding (and without a composition).
    """

    served: Optional[TargetPaths]
    reason: str = ""
    split: bool = False
    drafter: Optional[DrafterContext] = None


def _decision_key(
    model_config: Any,
    compute_capability: Optional[Tuple[int, int]],
    served,
    drafter: Optional[DrafterContext] = None,
) -> str:
    """One stage's composition decision, as the stages compare it."""
    if served is None:
        return "none"
    key = f"sm{sm_of(compute_capability)} tp{int(model_config.mapping.tp_size)}"
    return key if drafter is None else f"{key}, {drafter.key()}"


def agreed_composition(
    model_config: Any,
    compositions: Mapping[CompositionKey, TargetPaths],
    compute_capability: Optional[Tuple[int, int]],
    snapshot: Mapping[str, str],
) -> Decision:
    """The composition decision every stage agrees on.

    Collective over the PP group when it has more than one stage, whatever this stage's own
    decision: every stage calls it once, at the same point (the shell's constructor, before
    the built-in constructor). Stages that disagree (another role environment, device or
    drafter on some stage) all get ``Decision(None, split=True)``, naming each stage's
    decision; the drafter and its arena are part of what they agree on.
    """
    served = composition(model_config, compositions, compute_capability, snapshot)
    drafter = drafter_of(model_config) if served is not None else None
    mapping = getattr(model_config, "mapping", None)
    if group_size(mapping) > 1:
        everyone = _allgather(
            mapping, _decision_key(model_config, compute_capability, served, drafter)
        )
        if len(set(everyone)) > 1:
            stages = ", ".join(f"stage {rank}: {key}" for rank, key in enumerate(everyone))
            return Decision(None, f"the stages disagree on the composition ({stages})", True)
    if served is None:
        return Decision(
            None, unserved_reason(model_config, compositions, compute_capability, snapshot)
        )
    return Decision(served, drafter=drafter)


def lever_environment() -> Mapping[str, str]:
    """The lever environment (``MUSESPARK_OPEN_LEVER_*``) as this rank sees it now."""
    return _levers.environment()


def require_served(
    model_config: Any,
    mode: ModelingV2Mode,
    compositions: Mapping[CompositionKey, TargetPaths],
    compute_capability: Optional[Tuple[int, int]],
    snapshot: Optional[Mapping[str, str]] = None,
    decision: Optional[Decision] = None,
) -> None:
    """Raise under ``require`` when the deployment has no composition; before any construction.

    ``decision`` is the agreed one (``agreed_composition``); without it, it is agreed here
    under ``require`` (a collective).

    Raises:
        ValueError: under ``modeling_v2="require"`` when the agreed decision has no
            composition (on every stage alike).
    """
    if mode is not ModelingV2Mode.REQUIRE:
        return
    snapshot = _levers.environment() if snapshot is None else snapshot
    if decision is None:
        decision = agreed_composition(model_config, compositions, compute_capability, snapshot)
    if decision.served is None:
        raise ValueError(
            f"modeling_v2='require': {TAG} serves no fast path for this deployment: "
            + decision.reason
        )


def _pdl_enabled(text_model, model_config) -> bool:
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    return bool(get_env_enable_pdl())


def _builtin_switch(name: str, on: bool) -> Callable[[Any, Any], bool]:
    def holds(text_model, model_config) -> bool:
        from tensorrt_llm._torch.models import modeling_musespark_open as builtin

        return bool(getattr(builtin, name)()) == on

    return holds


def _builtin_prep_kernels(text_model, model_config) -> bool:
    return _tep_module("attention")._builtin_prep_kernel_mismatch() is None


def owned_layers(text_model) -> List[Any]:
    """The decoder layers this stage runs (``MuseSparkOpenModel._pp_stage``), in order.

    Raises:
        ValueError: for a text model that is not a pipeline stage.
    """
    stage = getattr(text_model, "_pp_stage", None)
    if stage is None:
        raise ValueError(f"{TAG} the text model is not a pipeline stage (no _pp_stage)")
    first, end = stage
    return [text_model.layers[index] for index in range(first, end)]


def _default_qkv_weight_layout(text_model, model_config) -> bool:
    """The built-in attention keeps the checkpoint's QKV weight layout on every stage layer."""
    return all(
        not getattr(layer.self_attn, "_qkv_weight_layout", False)
        and getattr(layer.self_attn, "_qkv_layout_inverse", None) is None
        for layer in owned_layers(text_model)
    )


_PREP = ("context_prep",)

#: Every public precondition of a fast path of this target, in the order they are reported.
PRECONDITIONS: Tuple[Precondition, ...] = (
    Precondition("programmatic dependent launch (TRTLLM_ENABLE_PDL=1)", _pdl_enabled),
    Precondition(
        "the unfused pre-attention prep (MUSESPARK_OPEN_FUSED_PREATTN off, its default)",
        _builtin_switch("_fused_preattention_enabled", False),
        _PREP,
    ),
    Precondition(
        "the built-in prep on flashinfer's CuTe-DSL RMSNorm and cos/sin-cache RoPE "
        "(flashinfer available, FLASHINFER_USE_CUDA_NORM unset)",
        _builtin_prep_kernels,
        _PREP,
    ),
    Precondition(
        "the checkpoint's QKV weight layout (MUSESPARK_OPEN_LL_QKV_WEIGHT_LAYOUT off, its default)",
        _default_qkv_weight_layout,
        _PREP,
    ),
)


def preconditions(paths: FrozenSet[str], served: TargetPaths) -> Tuple[Precondition, ...]:
    """The preconditions of a stage built with ``paths`` from composition ``served``."""
    return tuple(p for p in PRECONDITIONS if p.needed(paths)) + tuple(served.preconditions)


def missing_preconditions(text_model: Any, model_config: Any, required) -> List[str]:
    """The requirements of ``required`` this rank's built stage does not meet."""
    return [p.requirement for p in required if not p.holds(text_model, model_config)]


def deployment_of(
    model_config: Any,
    compute_capability,
    engine_facts: Any,
    snapshot: Mapping[str, str],
    drafter: Optional[DrafterContext] = None,
) -> StageDeployment:
    """The ``StageDeployment`` of a stage: its mapping and shape, the device, the engine and
    the drafter arena (``levers.Arena.NONE`` without ``drafter``).

    Raises:
        ValueError: as ``levers.role_of``.
    """
    major, minor = compute_capability
    role, source = _levers.role_of(snapshot, getattr(engine_facts, "disaggregated", None))
    mapping = model_config.mapping
    pretrained = model_config.pretrained_config
    tp = int(mapping.tp_size)
    return StageDeployment(
        sm=10 * int(major) + int(minor),
        tp=tp,
        max_batch_size=getattr(engine_facts, "max_batch_size", None),
        max_num_tokens=getattr(engine_facts, "max_num_tokens", None),
        role=role,
        role_source=source,
        arena=_levers.Arena.NONE if drafter is None else drafter.arena,
        facts=_levers.facts_of(snapshot),
        pp=int(mapping.pp_size),
        heads=(
            int(pretrained.num_attention_heads) // tp,
            int(pretrained.num_key_value_heads) // tp,
        ),
    )


def table_for(
    compositions: Mapping[CompositionKey, TargetPaths],
    sm: int,
    tp: int,
    pp: int,
    heads: Tuple[int, int],
    max_batch_size: Optional[int],
    role: str,
    knobs: Optional[Mapping[str, str]] = None,
    max_num_tokens: Optional[int] = None,
    facts: Optional[Any] = None,
    arena: str = "none",
) -> Any:
    """The lever table ``compose`` resolves for a stage, without a model or a device.

    For tests and for reviewing a deployment's levers. ``role`` is a ``levers.Role`` value;
    ``knobs`` the lever environment (names under ``levers.PREFIX``); ``arena`` a
    ``levers.Arena`` value (``paged`` with a DFlash drafter).

    Raises:
        KeyError: for a key no composition serves.
        ValueError: as ``levers.resolve``.
    """
    served = compositions[(sm, tp)].served
    role = _levers.Role(role)
    deployment = StageDeployment(
        sm=sm,
        tp=tp,
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        role=role,
        role_source=(
            "hint" if role in (_levers.Role.CTX, _levers.Role.GEN) else "no cache transceiver"
        ),
        arena=_levers.Arena(arena),
        facts=facts or _levers.Facts(),
        pp=pp,
        heads=tuple(heads),
    )
    return _levers.resolve(
        deployment,
        served,
        structural_rules(served, deployment),
        stage_snapshot(dict(knobs or {}), served),
        known_levers(),
    )


# --------------------------------------------------------------------------
# Agreement over the stages
# --------------------------------------------------------------------------
_T = TypeVar("_T")

#: The prefix of a rank's lever error carried through the agreement.
_LEVER_ERROR = "lever table: "


def group_size(mapping) -> int:
    """The pipeline stages of ``mapping`` (``pp_size``; 1 for None)."""
    return int(getattr(mapping, "pp_size", 1) or 1) if mapping is not None else 1


def _allgather(mapping, local):
    """``local`` of every stage, in stage order: a collective over the PP group only."""
    from tensorrt_llm._torch.distributed.communicator import Distributed

    return Distributed.get(mapping).pp_allgather(local)


def agreed_union(mapping, local: List[str]) -> List[str]:
    """Every item some stage lists, the same list on every stage.

    Collective over the PP group when it has more than one stage: every stage must call it at
    the same point. An item only some stages list names those stages.
    """
    if group_size(mapping) <= 1:
        return list(dict.fromkeys(local))
    everyone = _allgather(mapping, list(local))
    ranks: dict = {}
    for rank, items in enumerate(everyone):
        for item in items:
            ranks.setdefault(item, []).append(rank)
    return [
        item if len(seen) == len(everyone) else f"{item} (on stages {seen} of {len(everyone)})"
        for item, seen in ranks.items()
    ]


def agreed_refusal(mapping, local: Optional[str]) -> Optional[str]:
    """``local`` (this stage's reason to refuse, or None) as every stage agrees; collective."""
    if group_size(mapping) <= 1:
        return local
    everyone = _allgather(mapping, local)
    refused = [(rank, why) for rank, why in enumerate(everyone) if why is not None]
    if not refused:
        return None
    reasons = sorted({why for _, why in refused})
    if len(refused) == len(everyone) and len(reasons) == 1:
        return reasons[0]
    return f"{'; '.join(reasons)} (on stages {[rank for rank, _ in refused]} of {len(everyone)})"


def agreed_install(mapping, install: Callable[[], _T], refusals=(ValueError,)) -> _T:
    """``install()``'s result, or its refusal raised on every stage alike (collective).

    ``install`` checks and builds without changing a layer; the installers return what
    changes the layers and call it after the agreement.
    """
    result = None
    held: Optional[Exception] = None
    try:
        result = install()
    except refusals as error:
        held = error
    reason = agreed_refusal(mapping, None if held is None else (str(held) or type(held).__name__))
    if reason is not None:
        raise ValueError(reason) from held
    return result


def compose(
    text_model: Any,
    model_config: Any,
    mode: ModelingV2Mode,
    compositions: Mapping[CompositionKey, TargetPaths],
    compute_capability: Optional[Tuple[int, int]],
    *,
    facts: Any = None,
    snapshot: Optional[Mapping[str, str]] = None,
    decision: Optional[Decision] = None,
) -> Plan:
    """The plan this stage is built with: its composition's lever table, or ``NO_PATHS``.

    Call after the built-in constructor and before anything is installed. ``decision`` is
    the agreed composition decision (``agreed_composition``, agreed here when not given).
    Without a composition: ``NO_PATHS`` under ``auto``, a raise under ``require`` (the lever
    environment is still validated). Otherwise the lever table (``levers.resolve`` under
    ``structural_rules``), then the preconditions of its levers, the lever environment and
    any stage's lever error, agreed over the stages in one collective.

    Raises:
        ValueError: under ``require`` without a composition; under any mode for a lever
            error, a precondition missing on some stage, or stages that disagree on the lever
            environment (on every stage alike).
    """
    snapshot = _levers.environment() if snapshot is None else snapshot
    if decision is None:
        decision = agreed_composition(model_config, compositions, compute_capability, snapshot)
    served = decision.served
    mapping = getattr(model_config, "mapping", None)
    if served is None:
        require_served(
            model_config, mode, compositions, compute_capability, snapshot, decision=decision
        )
        notes = _levers.no_composition_notes(snapshot, known_levers())
        if decision.split:
            logger.warning(f"{TAG} {decision.reason}; every stage runs the built-in stage")
        logger.info(
            f"{TAG} no fast path for this stage ({decision.reason}); running the built-in stage"
        )
        for note in notes:
            logger.info(f"{TAG} {note}")
        return NO_PATHS
    table = None
    local: List[str] = []
    try:
        _levers.parse_knobs(snapshot, known_levers())
        deployment = deployment_of(
            model_config, compute_capability, facts, snapshot, decision.drafter
        )
        table = _levers.resolve(
            deployment,
            served.served,
            structural_rules(served.served, deployment),
            stage_snapshot(snapshot, served.served),
            known_levers(),
        )
    except ValueError as error:
        local.append(f"{_LEVER_ERROR}{error}")
    if table is not None:
        local += missing_preconditions(
            text_model, model_config, preconditions(table.paths(), served)
        )
    tags = list(_levers.tagged(snapshot))
    agreed = agreed_union(mapping, local + tags)
    own = set(tags)
    split = [item for item in agreed if item.startswith(_levers.AGREEMENT_TAG) and item not in own]
    errors = [item[len(_LEVER_ERROR) :] for item in agreed if item.startswith(_LEVER_ERROR)]
    missing = [
        item for item in agreed if not item.startswith((_levers.AGREEMENT_TAG, _LEVER_ERROR))
    ]
    stage = int(getattr(mapping, "pp_rank", 0) or 0)
    composed = (
        f"{TAG} sm{sm_of(compute_capability)} pp{mapping.pp_size} tp{mapping.tp_size} stage "
        f"{stage} on {served.topology.description}"
    )
    if split:
        raise ValueError(
            f"{composed}: the stages disagree on the lever environment: " + "; ".join(split)
        )
    if errors:
        raise ValueError(f"{composed}: " + "; ".join(errors))
    if missing:
        raise ValueError(f"{composed}: the fast paths need " + "; ".join(missing))
    plan = Plan(table.paths(), levers=table, stage=stage, rank0=stage == 0)
    drafter = decision.drafter
    with_drafter = (
        "without a drafter"
        if drafter is None
        else f"writing the {drafter.key()} context on the last stage"
    )
    logger.info(f"{composed} {with_drafter}: {plan.describe()}")
    if plan.rank0:
        for line in table.describe():
            logger.info(f"{TAG} {line}")
        for warning in table.warnings:
            logger.warning(f"{TAG} lever warning: {warning}")
    return plan
