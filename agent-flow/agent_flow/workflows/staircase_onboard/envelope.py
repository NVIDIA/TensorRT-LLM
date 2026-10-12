# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a target is certified FOR: the serving envelope and the feature matrix.

A gate launches an engine at one configuration tuple -- a batch limit, a
token budget per step, a sequence length, a CUDA-graph capture list -- and
measures that tuple. The tempting reading of a green run is that the tuples
the gates launched ARE the certified set. That reading hides every defect that
only appears outside them: a run whose gates all used ``max_num_tokens`` 8192
certifies nothing about 16384, and a target-side fixed-size buffer sized 8192,
or a kernel that overflows a 32-bit offset past 2**25 rows, passes every gate
and crashes the first deployment that asks for more.

So the spec declares two things up front, and this module derives everything
else from them:

``serving_envelope``
    What the target is claimed to serve: ``max_batch_size``,
    ``max_num_tokens``, ``max_seq_len``, the CUDA-graph policy (an explicit
    batch list, or the padding policy the engine generates one from) and the
    MEMBERS -- every parallel configuration claimed, including multi-node and
    pipeline-parallel ones a single allocation cannot run. Catalog receipts,
    capacity constants and integer-width limits are judged against this box,
    never against the gate tuples.

``feature_matrix``
    The feature combinations that must have ENGINE evidence: which features
    on, at which members, which kind of evidence (boot, accuracy, witness,
    acceptance), at which point of the envelope (the gate tuple, the ceiling,
    or a named point).

From those, :func:`derive_certification` produces a
:class:`CertificationPlan`:

- the **gate tuples** the run's existing gates launch, each checked to lie
  inside the envelope (a gate measuring outside the envelope measures a
  configuration nobody claimed);
- the **envelope probes** the run must launch in addition -- a boot at the
  envelope's ceiling, per unit, so that warm-up at the real ``max_num_tokens``
  and capture at every claimed graph batch happen in the run, not in the first
  deployment;
- the **matrix cells**, each classified as gated (an existing gate covers
  it), probe (a derived probe covers it) or external (this allocation cannot
  produce it, so delivery requires a recorded evidence file);
- the **certified set** every smith, reviewer and auditor certifies against.

A spec with no ``serving_envelope`` is still accepted, as the migration path:
the envelope is then DERIVED from the gate tuples (the old semantics, stated
as such everywhere it is rendered) and ``python -m
agent_flow.workflows.staircase_onboard.envelope migrate <task.yaml>`` prints a
declared block to start from.
"""

from __future__ import annotations

import argparse
import copy
import math
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from .task_schema import (
    ACCURACY_FIELD,
    FEATURE_MATRIX_FIELD,
    FEATURES_FIELD,
    MAX_WITNESS_PROMPT_WORDS,
    SERVING_ENVELOPE_FIELD,
    TaskSchemaError,
    assembly_units,
    members_for,
    parse_parallel_segment,
    reject_unknown_keys,
)

# ------------------------------------------------------------------- schema

#: The three engine limits an envelope bounds, by their engine-argument names.
#: Spelled as the engine spells them on purpose: a capacity constant is tied
#: to one of these by name (see :mod:`.capacity_audit`), and a probe passes
#: them as ``--<axis>=<value>``.
ENVELOPE_AXES: tuple[str, ...] = ("max_batch_size", "max_num_tokens", "max_seq_len")

ENVELOPE_KEYS: tuple[str, ...] = (
    *ENVELOPE_AXES,
    "cuda_graph",
    "members",
    "max_draft_len",
    "gpus_per_node",
    "probe_prompt_tokens",
)
ENVELOPE_REQUIRED: tuple[str, ...] = (*ENVELOPE_AXES, "cuda_graph", "members")
CUDA_GRAPH_KEYS: tuple[str, ...] = ("batch_sizes", "padding", "max_batch_size")

MATRIX_CELL_KEYS: tuple[str, ...] = (
    "name",
    "features",
    "members",
    "evidence",
    "at",
    "prompt_tokens",
    "note",
)
EVIDENCE_BOOT = "boot"
EVIDENCE_ACCURACY = "accuracy"
EVIDENCE_WITNESS = "witness"
EVIDENCE_ACCEPTANCE = "acceptance"
EVIDENCE_KINDS: tuple[str, ...] = (
    EVIDENCE_BOOT,
    EVIDENCE_ACCURACY,
    EVIDENCE_WITNESS,
    EVIDENCE_ACCEPTANCE,
)
AT_GATE = "gate"
AT_CEILING = "ceiling"

#: How a matrix cell gets its evidence.
CELL_GATED = "gated"
CELL_PROBE = "probe"
CELL_EXTERNAL = "external"

#: Where an external cell's evidence is recorded, under the workspace.
EVIDENCE_DIRNAME = "evidence"

#: The name a derived probe gate is recorded under.
PROBE_GATE = "envelope"

#: ``tp4``, ``tep4``, ``dep8``, ``tep4pp2``. The gated member grammar
#: (:data:`~.task_schema.PARALLEL_PATTERN`) has no pipeline axis because one
#: allocation serves every gated member; an envelope member is a CLAIM and may
#: name a configuration this allocation cannot run.
ENVELOPE_MEMBER_PATTERN = re.compile(r"^(tp|tep|dep)(\d+)(?:pp(\d+))?$")

#: Engine defaults where a gate command does not state a limit. The same for
#: ``quickstart_advanced.py`` and ``trtllm-eval``: both take them from the
#: LLM-API's own field defaults. ``max_seq_len`` defaults to the checkpoint's
#: ``max_position_embeddings``, which is why it is ``None`` here.
ENGINE_DEFAULTS: Mapping[str, int | None] = {
    "max_batch_size": 2048,
    "max_num_tokens": 8192,
    "max_seq_len": None,
}

#: CudaGraphConfig's own default bound when neither a list nor a maximum is
#: given.
DEFAULT_GRAPH_MAX_BATCH_SIZE = 128

_FLAG_AXES = {f"--{axis}": axis for axis in ENVELOPE_AXES}
_GRAPH_LIST_FLAG = "--cuda_graph_batch_sizes"
_GRAPH_PADDING_FLAG = "--cuda_graph_padding_enabled"


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


# ------------------------------------------------------------------ members


@dataclass(frozen=True)
class EnvelopeMember:
    """One parallel configuration the target is claimed to serve.

    ``size`` is the width of one pipeline stage (the trailing number of the
    gated grammar); ``pp`` multiplies it. ``tep4pp2`` is eight devices, two
    stages of four.
    """

    name: str
    kind: str
    size: int
    pp: int = 1

    @property
    def world_size(self) -> int:
        return self.size * self.pp

    @property
    def attention_dp(self) -> bool:
        """Whether ops after the attention-DP gather see every rank's rows."""
        return self.kind == "dep"

    def nodes(self, gpus_per_node: int | None) -> int:
        if not gpus_per_node:
            return 1
        return max(1, math.ceil(self.world_size / gpus_per_node))

    def describe(self, gpus_per_node: int | None) -> str:
        parts = [f"{self.world_size} GPU(s)"]
        if self.pp > 1:
            parts.insert(0, f"PP{self.pp}")
        nodes = self.nodes(gpus_per_node)
        if nodes > 1:
            parts.append(f"{nodes} nodes")
        return f"{self.name} ({', '.join(parts)})"


def parse_envelope_member(text: str) -> EnvelopeMember:
    """Parse ``tep4`` / ``dep8`` / ``tp4pp2``. Raises :class:`ValueError`."""
    match = ENVELOPE_MEMBER_PATTERN.match(str(text))
    if match is None:
        raise ValueError(
            f"{text!r} is not an envelope member; expected tp<N>, tep<N> or dep<N>, "
            f"optionally followed by pp<M> (e.g. 'tep4pp2')"
        )
    size = int(match.group(2))
    pp = int(match.group(3) or 1)
    if size < 1 or pp < 1:
        raise ValueError(f"{text!r} declares a zero-width member")
    return EnvelopeMember(name=str(text), kind=match.group(1), size=size, pp=pp)


# -------------------------------------------------------------- graph policy


def generate_graph_batch_sizes(max_batch_size: int, padding: bool) -> list[int]:
    """The capture list the engine generates from a bound and a padding policy.

    Mirrors ``CudaGraphConfig._generate_cuda_graph_batch_sizes`` in
    ``tensorrt_llm/llmapi/llm_args.py``: with padding, ``1, 2, 4`` then
    multiples of 8 to 128 then steps of 64; without, every size to 31, then
    32, 64, 128 and powers of two. Restated rather than imported so this
    module stays importable on a login node with no engine.
    """
    if max_batch_size < 1:
        return []
    if padding:
        sizes = [1, 2, 4] + [i * 8 for i in range(1, 17)]
        while sizes[-1] + 64 <= max_batch_size:
            sizes.append(sizes[-1] + 64)
    else:
        sizes = list(range(1, 32)) + [32, 64, 128]
        if max_batch_size > 1:
            sizes += [2**i for i in range(8, math.ceil(math.log(max_batch_size, 2)))]
    sizes = sorted({size for size in sizes if size <= max_batch_size})
    if not sizes or sizes[-1] != max_batch_size:
        sizes.append(max_batch_size)
    return sizes


@dataclass(frozen=True)
class GraphPolicy:
    """Which batch sizes are captured. Empty ``batch_sizes`` means eager."""

    batch_sizes: tuple[int, ...] = ()
    padding: bool = False
    source: str = "eager"

    @property
    def enabled(self) -> bool:
        return bool(self.batch_sizes)

    def clipped(self, max_batch_size: int | None) -> tuple[int, ...]:
        """The sizes an engine with this batch limit actually captures."""
        if max_batch_size is None:
            return self.batch_sizes
        return tuple(size for size in self.batch_sizes if size <= max_batch_size)


def resolve_graph_policy(block: Any, max_batch_size: int | None) -> GraphPolicy:
    """Read ``serving_envelope.cuda_graph`` (already validated)."""
    if block is False or block is None:
        return GraphPolicy()
    block = dict(block)
    explicit = block.get("batch_sizes")
    padding = bool(block.get("padding", False))
    if explicit:
        return GraphPolicy(
            batch_sizes=tuple(sorted({int(v) for v in explicit})),
            padding=padding,
            source="explicit list",
        )
    bound = int(block.get("max_batch_size") or max_batch_size or DEFAULT_GRAPH_MAX_BATCH_SIZE)
    return GraphPolicy(
        batch_sizes=tuple(generate_graph_batch_sizes(bound, padding)),
        padding=padding,
        source=f"{'padding' if padding else 'no-padding'} policy up to {bound}",
    )


# ------------------------------------------------------------------ envelope


@dataclass(frozen=True)
class ServingEnvelope:
    """The box every certification is judged against.

    ``declared`` is False when the spec stated no envelope and this one was
    derived from the gate tuples. That is the legacy reading -- the certified
    set IS the gate tuples -- and every rendering says so.
    """

    max_batch_size: int | None
    max_num_tokens: int | None
    max_seq_len: int | None
    graph: GraphPolicy = field(default_factory=GraphPolicy)
    members: tuple[EnvelopeMember, ...] = ()
    max_draft_len: int = 0
    gpus_per_node: int | None = None
    probe_prompt_tokens: int | None = None
    declared: bool = True

    def limit(self, axis: str) -> int | None:
        return getattr(self, axis)

    @property
    def capture_batches(self) -> tuple[int, ...]:
        return self.graph.clipped(self.max_batch_size)

    def member(self, name: str) -> EnvelopeMember | None:
        return next((m for m in self.members if m.name == name), None)

    @property
    def max_world_size(self) -> int:
        return max((m.world_size for m in self.members), default=1)

    @property
    def max_gather_world_size(self) -> int:
        """The widest attention-DP gather any member performs, or 1."""
        return max((m.world_size for m in self.members if m.attention_dp), default=1)


def envelope_from_spec(
    block: Mapping[str, Any],
    *,
    gpus_per_node: int | None = None,
) -> ServingEnvelope:
    """Build the declared envelope from a validated ``serving_envelope`` block."""
    max_batch_size = int(block["max_batch_size"])
    return ServingEnvelope(
        max_batch_size=max_batch_size,
        max_num_tokens=int(block["max_num_tokens"]),
        max_seq_len=int(block["max_seq_len"]),
        graph=resolve_graph_policy(block.get("cuda_graph"), max_batch_size),
        members=tuple(parse_envelope_member(m) for m in block.get("members") or ()),
        max_draft_len=int(block.get("max_draft_len") or 0),
        gpus_per_node=int(block.get("gpus_per_node") or gpus_per_node or 0) or None,
        probe_prompt_tokens=(
            int(block["probe_prompt_tokens"]) if block.get("probe_prompt_tokens") else None
        ),
        declared=True,
    )


# ------------------------------------------------------------- engine tuples


@dataclass(frozen=True)
class EngineTuple:
    """One configuration an engine is launched at, and who launches it."""

    gate: str
    unit: str
    member: str
    max_batch_size: int | None
    max_num_tokens: int | None
    max_seq_len: int | None
    graph_batch_sizes: tuple[int, ...] = ()
    source: str = ""
    #: A generated prompt of at least this many words, sent beside the boot
    #: prompts. Probes only.
    prompt_tokens: int | None = None
    #: Whether batches are padded up to the next captured size. Probes only:
    #: a probe serves the envelope's own padding policy.
    graph_padding: bool = False

    def limit(self, axis: str) -> int | None:
        return getattr(self, axis)

    def point(self) -> str:
        """``B=8 T=2048 L=16384 G=1..8``: the tuple, compactly."""

        def show(value: int | None) -> str:
            return "default" if value is None else str(value)

        graphs = compact_values(self.graph_batch_sizes) if self.graph_batch_sizes else "eager"
        text = (
            f"B={show(self.max_batch_size)} T={show(self.max_num_tokens)} "
            f"L={show(self.max_seq_len)} G={graphs}"
        )
        if self.prompt_tokens:
            text += f" prompt>={self.prompt_tokens}"
        return text

    def label(self) -> str:
        return f"{self.gate} {self.unit}@{self.member} ({self.point()})"

    def outside(self, envelope: ServingEnvelope, *, checkpoint_max: int | None) -> list[str]:
        """Every way this tuple exceeds the envelope, as sentences."""
        problems: list[str] = []
        for axis in ENVELOPE_AXES:
            bound = envelope.limit(axis)
            value = self.limit(axis)
            if bound is None:
                continue
            if value is None and axis == "max_seq_len":
                # The engine then sizes for the checkpoint's own maximum.
                value = checkpoint_max
                if value is None:
                    continue
            if value is None:
                continue
            if value > bound:
                problems.append(f"{axis}={value} > serving_envelope.{axis}={bound}")
        return problems


def compact_values(values: Sequence[int]) -> str:
    """``1,2,3,4,8,16`` -> ``1..4,8,16``: a capture list on one line."""
    ordered = sorted(set(int(v) for v in values))
    runs: list[str] = []
    start = prev = None
    for value in ordered:
        if start is None:
            start = prev = value
            continue
        if value == prev + 1:
            prev = value
            continue
        runs.append(str(start) if start == prev else f"{start}..{prev}")
        start = prev = value
    if start is not None:
        runs.append(str(start) if start == prev else f"{start}..{prev}")
    return ",".join(runs)


def _flag_values(flags: Iterable[str]) -> dict[str, list[str]]:
    """``{"--max_batch_size": ["8"], ...}`` from runtime ``extra_flags``.

    Three spellings survive in real specs -- ``--flag=v``, ``--flag v`` as one
    string, and the flag and its value(s) as separate elements -- and a
    ``nargs='+'`` flag (the graph list) takes every following bare value.
    """
    words: list[str] = []
    for flag in flags:
        words.extend(str(flag).split())
    found: dict[str, list[str]] = {}
    current: str | None = None
    for word in words:
        if word.startswith("--"):
            name, _, value = word.partition("=")
            found[name] = [value] if value else []
            current = None if value else name
            continue
        if current is not None:
            found[current].append(word)
    return found


def _int_or_none(values: Sequence[str] | None) -> int | None:
    if not values:
        return None
    try:
        return int(values[-1])
    except ValueError:
        return None


def runtime_tuple(
    runtime: Mapping[str, Any],
    *,
    gate: str,
    unit: str,
    member: str,
    max_batch_size: int | None = None,
) -> EngineTuple:
    """The tuple a quickstart gate (boot, witness, acceptance) launches.

    ``max_batch_size`` is a witness's own override (both legs at one limit).
    """
    flags = _flag_values(runtime.get("extra_flags") or ())
    values: dict[str, int | None] = {}
    for spelled, axis in _FLAG_AXES.items():
        values[axis] = _int_or_none(flags.get(spelled))
        if values[axis] is None:
            values[axis] = ENGINE_DEFAULTS[axis]
    if max_batch_size is not None:
        values["max_batch_size"] = int(max_batch_size)
    graphs: tuple[int, ...] = ()
    if runtime.get("cuda_graph"):
        explicit = [int(v) for v in flags.get(_GRAPH_LIST_FLAG, []) if v.isdigit()]
        if explicit:
            graphs = tuple(sorted(set(explicit)))
        else:
            graphs = tuple(
                generate_graph_batch_sizes(
                    DEFAULT_GRAPH_MAX_BATCH_SIZE, _GRAPH_PADDING_FLAG in flags
                )
            )
        graphs = tuple(
            g for g in graphs if values["max_batch_size"] is None or g <= values["max_batch_size"]
        )
    return EngineTuple(
        gate=gate,
        unit=unit,
        member=member,
        graph_batch_sizes=graphs,
        source="runtime block",
        **values,
    )


def options_tuple(
    accuracy: Mapping[str, Any],
    options_path: str | None,
    *,
    unit: str,
    member: str,
) -> EngineTuple:
    """The tuple ``trtllm-eval`` launches for the accuracy gate.

    The options YAML is read first because ``trtllm-eval`` lets it override
    its own flags; ``accuracy.max_seq_len`` / ``max_batch_size`` fill what the
    YAML leaves out; the engine defaults fill the rest.
    """
    options: dict[str, Any] = {}
    if options_path and Path(options_path).is_file():
        loaded = yaml.safe_load(Path(options_path).read_text(encoding="utf-8")) or {}
        options = loaded if isinstance(loaded, dict) else {}
    values: dict[str, int | None] = {}
    for axis in ENVELOPE_AXES:
        value = options.get(axis)
        if value is None:
            value = accuracy.get(axis)
        values[axis] = int(value) if value is not None else ENGINE_DEFAULTS[axis]
    graphs: tuple[int, ...] = ()
    graph_config = options.get("cuda_graph_config")
    if isinstance(graph_config, dict):
        policy = resolve_graph_policy(
            {
                "batch_sizes": graph_config.get("batch_sizes"),
                "padding": bool(graph_config.get("enable_padding", False)),
                "max_batch_size": graph_config.get("max_batch_size")
                or DEFAULT_GRAPH_MAX_BATCH_SIZE,
            },
            values["max_batch_size"],
        )
        graphs = policy.clipped(values["max_batch_size"])
    source = f"options {Path(options_path).name}" if options_path else "trtllm-eval defaults"
    return EngineTuple(
        gate=EVIDENCE_ACCURACY,
        unit=unit,
        member=member,
        graph_batch_sizes=graphs,
        source=source,
        **values,
    )


# -------------------------------------------------------------- unit gates


@dataclass(frozen=True)
class UnitGates:
    """What the run's own gates measure for one unit."""

    unit: str
    features: tuple[str, ...]
    members: tuple[str, ...]
    kinds: frozenset[str]
    #: The witness runs at one member only; see the workflow's call site.
    witness_member: str | None = None


def features_through(task_data: Mapping[str, Any], unit: str) -> tuple[str, ...]:
    """The features a unit's target implements: every feature up to it."""
    features = [str(f) for f in task_data.get(FEATURES_FIELD) or ()]
    if unit not in features:
        return ()
    return tuple(features[: features.index(unit) + 1])


def default_unit_kinds(task_data: Mapping[str, Any], unit: str) -> frozenset[str]:
    """The gate kinds a unit runs, read from what the spec declares for it.

    The orchestrator passes its own exact answer; this is the reading a
    caller without it (preflight in a test, the migration command) uses: a
    witness where ``accuracy.witness`` has a block for the unit, an
    acceptance gate where ``accuracy.acceptance`` names it.
    """
    accuracy = task_data.get(ACCURACY_FIELD) or {}
    kinds = {EVIDENCE_BOOT, EVIDENCE_ACCURACY}
    witness = accuracy.get("witness") if isinstance(accuracy, dict) else None
    if isinstance(witness, dict) and unit in witness:
        kinds.add(EVIDENCE_WITNESS)
    acceptance = accuracy.get("acceptance") if isinstance(accuracy, dict) else None
    if isinstance(acceptance, dict) and unit in acceptance:
        kinds.add(EVIDENCE_ACCEPTANCE)
    return frozenset(kinds)


def resolve_unit_gates(
    task_data: Mapping[str, Any],
    world_size: int,
    *,
    unit_kinds: Callable[[str], Iterable[str]] | None = None,
) -> list[UnitGates]:
    """Every unit's gated members and gate kinds, cheapest member first."""
    from .gates import order_members_by_cost

    gates: list[UnitGates] = []
    for unit in assembly_units(task_data):
        members = tuple(order_members_by_cost(members_for(task_data, unit, world_size)))
        kinds = frozenset(
            unit_kinds(unit) if unit_kinds is not None else default_unit_kinds(task_data, unit)
        )
        gates.append(
            UnitGates(
                unit=unit,
                features=features_through(task_data, unit),
                members=members,
                kinds=kinds,
                witness_member=members[0] if EVIDENCE_WITNESS in kinds and members else None,
            )
        )
    return gates


def gate_tuples(task_data: Mapping[str, Any], gates: Sequence[UnitGates]) -> list[EngineTuple]:
    """Every tuple the run's existing gates launch."""
    from .gates import resolve_llm_api_options, resolve_runtime

    accuracy = task_data.get(ACCURACY_FIELD) or {}
    witness_blocks = accuracy.get("witness") if isinstance(accuracy, dict) else None
    witness_blocks = witness_blocks if isinstance(witness_blocks, dict) else {}
    tuples: list[EngineTuple] = []
    for unit_gates in gates:
        unit = unit_gates.unit
        runtime = resolve_runtime(task_data, unit)
        options = resolve_llm_api_options(accuracy, unit) if isinstance(accuracy, dict) else None
        for member in unit_gates.members:
            if EVIDENCE_BOOT in unit_gates.kinds:
                tuples.append(runtime_tuple(runtime, gate=EVIDENCE_BOOT, unit=unit, member=member))
            if EVIDENCE_ACCURACY in unit_gates.kinds:
                tuples.append(options_tuple(accuracy, options, unit=unit, member=member))
            if EVIDENCE_ACCEPTANCE in unit_gates.kinds:
                tuples.append(
                    runtime_tuple(runtime, gate=EVIDENCE_ACCEPTANCE, unit=unit, member=member)
                )
        if unit_gates.witness_member is not None:
            block = witness_blocks.get(unit) or {}
            batch = block.get("max_batch_size") if isinstance(block, dict) else None
            tuples.append(
                runtime_tuple(
                    runtime,
                    gate=EVIDENCE_WITNESS,
                    unit=unit,
                    member=unit_gates.witness_member,
                    max_batch_size=int(batch) if batch else None,
                )
            )
    return tuples


def derived_envelope(tuples: Sequence[EngineTuple], members: Sequence[str]) -> ServingEnvelope:
    """The legacy envelope: the hull of the gate tuples.

    This is what a run without ``serving_envelope`` has always certified --
    the gate tuples and nothing more -- made explicit so it can be rendered
    as exactly that.
    """

    def hull(axis: str) -> int | None:
        known = [t.limit(axis) for t in tuples if t.limit(axis) is not None]
        return max(known) if known else None

    graphs = sorted({g for t in tuples for g in t.graph_batch_sizes})
    parsed: list[EnvelopeMember] = []
    for name in members:
        try:
            parsed.append(parse_envelope_member(name))
        except ValueError:
            continue
    return ServingEnvelope(
        max_batch_size=hull("max_batch_size"),
        max_num_tokens=hull("max_num_tokens"),
        max_seq_len=hull("max_seq_len"),
        graph=GraphPolicy(
            batch_sizes=tuple(graphs),
            source="union of the gates' capture lists" if graphs else "eager",
        ),
        members=tuple(parsed),
        declared=False,
    )


# ------------------------------------------------------------------- matrix


@dataclass(frozen=True)
class MatrixCell:
    """One required piece of engine evidence, and how it will be obtained."""

    cell: str
    features: tuple[str, ...]
    member: str
    evidence: str
    at: str
    status: str
    reason: str
    unit: str | None = None

    @property
    def slug(self) -> str:
        """A file-name-safe identity: the evidence file for an external cell.

        Keyed by the cell's NAME rather than its feature list, which for a
        late unit is every feature of the run; the schema keeps names unique.
        """
        raw = f"{self.cell}__{self.member}__{self.evidence}__{self.at}"
        return re.sub(r"[^A-Za-z0-9_.+-]+", "-", raw).strip("-")

    def render(self) -> str:
        feats = "+".join(self.features) or "core"
        where = f" via unit `{self.unit}`" if self.unit else ""
        return (
            f"[{self.status:<8}] {feats} @ {self.member}: {self.evidence} at {self.at}"
            f"{where} -- {self.reason}"
        )


def _at_label(at: Any) -> str:
    if isinstance(at, dict):
        return ",".join(f"{k}={v}" for k, v in sorted(at.items()))
    return str(at or AT_GATE)


def _point_tuple(
    at: Any,
    envelope: ServingEnvelope,
    base: EngineTuple | None,
    *,
    unit: str,
    member: str,
    prompt_tokens: int | None,
) -> EngineTuple:
    """The engine tuple a probe at ``at`` launches."""
    if at == AT_CEILING:
        values = {axis: envelope.limit(axis) for axis in ENVELOPE_AXES}
        label = AT_CEILING
    else:
        values = {axis: (base.limit(axis) if base else None) for axis in ENVELOPE_AXES}
        values.update({axis: int(v) for axis, v in dict(at).items()})
        label = _at_label(at)
    graphs = envelope.graph.clipped(values["max_batch_size"])
    return EngineTuple(
        gate=PROBE_GATE,
        unit=unit,
        member=member,
        graph_batch_sizes=graphs,
        source=f"serving envelope ({label})",
        prompt_tokens=prompt_tokens or envelope.probe_prompt_tokens,
        graph_padding=envelope.graph.padding and bool(graphs),
        **values,
    )


def _reaches(tuple_: EngineTuple, target: EngineTuple) -> bool:
    """Whether an existing gate tuple already measures ``target``'s point."""
    for axis in ENVELOPE_AXES:
        want = target.limit(axis)
        have = tuple_.limit(axis)
        if want is None:
            continue
        if have is None or have < want:
            return False
    return set(target.graph_batch_sizes) <= set(tuple_.graph_batch_sizes)


# --------------------------------------------------------------------- plan


@dataclass(frozen=True)
class CertificationPlan:
    """Everything derived from the envelope and the matrix, for one run."""

    envelope: ServingEnvelope
    tuples: tuple[EngineTuple, ...] = ()
    probes: tuple[EngineTuple, ...] = ()
    cells: tuple[MatrixCell, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    #: The world size of the run's own allocation.
    world_size: int = 1

    @property
    def declared(self) -> bool:
        return self.envelope.declared

    def probes_for(self, unit: str) -> list[EngineTuple]:
        return [probe for probe in self.probes if probe.unit == unit]

    @property
    def external_cells(self) -> list[MatrixCell]:
        return [cell for cell in self.cells if cell.status == CELL_EXTERNAL]

    @property
    def certified_capture_batches(self) -> tuple[int, ...]:
        """Every batch size a graph is captured at: the envelope's and the gates'.

        The union, because a gate that captures a size the envelope does not
        list still runs the catalog at that size, and an uncertified value in
        a gate is an uncertified value however it got there.
        """
        sizes = set(self.envelope.capture_batches)
        for tuple_ in (*self.tuples, *self.probes):
            sizes.update(tuple_.graph_batch_sizes)
        return tuple(sorted(sizes))

    def standard(self) -> list[str]:
        """The certification standard, one line per axis."""
        env = self.envelope
        gather = env.max_gather_world_size
        lines = [
            f"tokens per step per rank: T <= {env.max_num_tokens}"
            + (
                f" (x{gather} = {env.max_num_tokens * gather} after an attention-DP gather)"
                if gather > 1 and env.max_num_tokens
                else ""
            ),
            f"requests per step: <= {env.max_batch_size}",
            f"per-request length: <= {env.max_seq_len}",
            "CUDA-graph capture batches G: "
            + (compact_values(self.certified_capture_batches) or "none (eager only)")
            + "; a call site captured at row multiple m certifies m*G for every G",
            "pools: any size up to the largest the GPU allocates at the envelope's "
            "settings, with rows addressed at the far end",
        ]
        if env.max_draft_len:
            lines.append(
                f"speculation: up to {env.max_draft_len} draft token(s) per request per step "
                f"({env.max_draft_len + 1} rows per request at the verify step)"
            )
        return lines

    def notes(self) -> list[str]:
        """Preflight's notes: what is certified, what is probed, what is owed."""
        env = self.envelope
        notes: list[str] = []
        if env.declared:
            members = ", ".join(m.describe(env.gpus_per_node) for m in env.members)
            notes.append(
                f"serving envelope: B<={env.max_batch_size} T<={env.max_num_tokens} "
                f"L<={env.max_seq_len} G={compact_values(env.capture_batches) or 'eager'}; "
                f"members {members}"
            )
        else:
            notes.append(
                "no serving_envelope declared: the certified set is the hull of the gate "
                f"tuples (B<={env.max_batch_size} T<={env.max_num_tokens} "
                f"L<={env.max_seq_len}). Nothing outside those tuples is certified. "
                "`python -m agent_flow.workflows.staircase_onboard.envelope migrate "
                "<task.yaml>` prints a block to declare."
            )
        if self.probes:
            notes.append(
                f"{len(self.probes)} envelope probe(s): "
                + "; ".join(probe.label() for probe in self.probes)
            )
        external = self.external_cells
        if external:
            notes.append(
                f"{len(external)} feature-matrix cell(s) need evidence this allocation cannot "
                f"produce; delivery requires `{EVIDENCE_DIRNAME}/<cell>.yaml` for each: "
                + ", ".join(cell.slug for cell in external)
            )
        notes.extend(self.warnings)
        return notes

    def render(self) -> str:
        """``certification.md``: the whole plan, for every role to read."""
        env = self.envelope
        head = (
            "the DECLARED serving envelope"
            if env.declared
            else "a DERIVED envelope -- the hull of the gate tuples, because the spec "
            "declares no `serving_envelope`. Nothing outside these tuples is certified"
        )
        lines = [
            "# Certification plan",
            "",
            f"The certified set is {head}. Catalog receipts, capacity constants and "
            "integer-width limits are judged against it. **A gate tuple is evidence "
            "inside the set, not the set.**",
            "",
            "## Certification standard",
            "",
            *(f"- {line}" for line in self.standard()),
            "",
            "## Members",
            "",
        ]
        gated = {t.member for t in self.tuples}
        for member in env.members:
            where = (
                "gated by this run" if member.name in gated else "not runnable in this allocation"
            )
            lines.append(f"- {member.describe(env.gpus_per_node)}: {where}")
        lines += ["", "## Gate tuples (what the run's gates launch)", ""]
        lines += [f"- {t.label()} [{t.source}]" for t in self.tuples] or ["- (none)"]
        lines += ["", "## Envelope probes (launched in addition, per unit)", ""]
        lines += [f"- {p.label()}" for p in self.probes] or ["- (none)"]
        lines += ["", "## Feature matrix", ""]
        lines += [f"- {cell.render()}" for cell in self.cells] or ["- (none)"]
        external = self.external_cells
        if external:
            lines += [
                "",
                "## Evidence owed from outside this allocation",
                "",
                f"Delivery fails until each of these has `{EVIDENCE_DIRNAME}/<cell>.yaml` "
                "in the workspace, with `verdict: pass`, the `tree` it was measured on and "
                "the `log` that shows it:",
                "",
                *(f"- `{cell.slug}`" for cell in external),
            ]
        if self.warnings:
            lines += ["", "## Warnings", "", *(f"- {w}" for w in self.warnings)]
        return "\n".join(lines) + "\n"

    def prompt_block(self, path: Path | str | None = None) -> str:
        """The block a role's prompt carries: the standard, and where the rest is."""
        env = self.envelope
        kind = "declared" if env.declared else "DERIVED from the gate tuples (legacy)"
        where = f" The full plan is `{path}`." if path else ""
        return (
            f"## Serving envelope ({kind})\n\n"
            "Certify against the envelope, never against the tuples a gate happened to "
            f"run.{where}\n\n" + "\n".join(f"- {line}" for line in self.standard()) + "\n"
        )


def _matrix_cells_from_spec(task_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    block = task_data.get(FEATURE_MATRIX_FIELD)
    return [dict(cell) for cell in block] if isinstance(block, list) else []


def _unit_for(features: Sequence[str], gates: Sequence[UnitGates]) -> UnitGates | None:
    """The earliest unit whose target implements every feature in ``features``."""
    wanted = set(features)
    return next((g for g in gates if wanted <= set(g.features)), None)


def derive_certification(
    task_data: Mapping[str, Any],
    *,
    world_size: int,
    gpus_per_node: int | None = None,
    checkpoint_config: Mapping[str, Any] | None = None,
    unit_kinds: Callable[[str], Iterable[str]] | None = None,
) -> CertificationPlan:
    """Derive the gate tuples, probes, matrix cells and certified set for a run.

    Pure apart from reading the accuracy gate's options YAMLs. ``errors`` are
    inconsistencies the run must not start with; preflight raises on them.
    """
    gates = resolve_unit_gates(task_data, world_size, unit_kinds=unit_kinds)
    tuples = gate_tuples(task_data, gates)
    gated_members = sorted({m for g in gates for m in g.members})
    checkpoint_max = None
    if checkpoint_config:
        text_config = checkpoint_config.get("text_config")
        source = text_config if isinstance(text_config, dict) else checkpoint_config
        raw = source.get("max_position_embeddings") or checkpoint_config.get(
            "max_position_embeddings"
        )
        checkpoint_max = int(raw) if isinstance(raw, int) else None

    block = task_data.get(SERVING_ENVELOPE_FIELD)
    environment = task_data.get("environment") or {}
    node_width = gpus_per_node or environment.get("gpus_per_node")
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(block, dict):
        envelope = derived_envelope(tuples, gated_members)
        cells = [
            MatrixCell(
                cell=f"{g.unit}-gates",
                features=g.features,
                member=member,
                evidence=kind,
                at=AT_GATE,
                status=CELL_GATED,
                reason="the unit's own gate",
                unit=g.unit,
            )
            for g in gates
            for member in g.members
            for kind in sorted(g.kinds)
            if kind != EVIDENCE_WITNESS or member == g.witness_member
        ]
        return CertificationPlan(
            envelope=envelope,
            tuples=tuple(tuples),
            cells=tuple(cells),
            world_size=world_size,
        )

    envelope = envelope_from_spec(block, gpus_per_node=node_width)
    claimed = {m.name for m in envelope.members}

    for member in gated_members:
        if member not in claimed:
            errors.append(
                f"member {member} is gated by this run but is not in "
                f"serving_envelope.members {sorted(claimed)}: a gate would measure a "
                f"configuration the target does not claim"
            )
    for tuple_ in tuples:
        problems = tuple_.outside(envelope, checkpoint_max=checkpoint_max)
        if problems:
            hint = (
                " (unset max_seq_len means the checkpoint's max_position_embeddings; pass "
                "--max_seq_len)"
                if any(p.startswith("max_seq_len") for p in problems) and tuple_.max_seq_len is None
                else ""
            )
            errors.append(
                f"the {tuple_.label()} gate launches outside the envelope: "
                f"{'; '.join(problems)}{hint}"
            )
    if (
        checkpoint_max is not None
        and envelope.max_seq_len
        and envelope.max_seq_len > checkpoint_max
    ):
        errors.append(
            f"serving_envelope.max_seq_len={envelope.max_seq_len} exceeds the checkpoint's "
            f"max_position_embeddings={checkpoint_max}"
        )
    if (
        envelope.max_batch_size
        and envelope.max_num_tokens
        and envelope.max_batch_size * (1 + envelope.max_draft_len) > envelope.max_num_tokens
    ):
        warnings.append(
            f"a full batch of decode steps needs {envelope.max_batch_size} x "
            f"{1 + envelope.max_draft_len} = "
            f"{envelope.max_batch_size * (1 + envelope.max_draft_len)} tokens, over "
            f"max_num_tokens={envelope.max_num_tokens}: the scheduler can never fill the "
            f"batch limit in one step"
        )
    if node_width:
        for member in envelope.members:
            if member.nodes(int(node_width)) > 1:
                warnings.append(
                    f"{member.describe(int(node_width))} spans nodes; its evidence comes "
                    f"from a multi-node session, not from this run's gates"
                )

    cells: list[MatrixCell] = []
    probes: dict[tuple[str, str, str], EngineTuple] = {}

    def gate_base(unit: str, member: str, kind: str = EVIDENCE_BOOT) -> EngineTuple | None:
        """The tuple ``kind``'s gate launches for (unit, member): a point's other axes."""
        return next(
            (t for t in tuples if t.unit == unit and t.member == member and t.gate == kind),
            None,
        )

    def add_probe(probe: EngineTuple) -> None:
        key = (probe.unit, probe.member, probe.point())
        probes.setdefault(key, probe)

    # The gates the run already runs are cells by construction.
    for g in gates:
        for member in g.members:
            for kind in sorted(g.kinds):
                if kind == EVIDENCE_WITNESS and member != g.witness_member:
                    continue
                cells.append(
                    MatrixCell(
                        cell=f"{g.unit}-gates",
                        features=g.features,
                        member=member,
                        evidence=kind,
                        at=AT_GATE,
                        status=CELL_GATED,
                        reason="the unit's own gate",
                        unit=g.unit,
                    )
                )
        # The minimum envelope evidence, per unit: one boot at the ceiling, at
        # the cheapest member. Warm-up at the real token budget and capture at
        # every claimed batch size are what expose a fixed-size buffer.
        if g.members:
            probe = _point_tuple(
                AT_CEILING,
                envelope,
                gate_base(g.unit, g.members[0]),
                unit=g.unit,
                member=g.members[0],
                prompt_tokens=None,
            )
            add_probe(probe)
            cells.append(
                MatrixCell(
                    cell=f"{g.unit}-ceiling",
                    features=g.features,
                    member=g.members[0],
                    evidence=EVIDENCE_BOOT,
                    at=AT_CEILING,
                    status=CELL_PROBE,
                    reason=probe.label(),
                    unit=g.unit,
                )
            )

    all_features = tuple(str(f) for f in task_data.get(FEATURES_FIELD) or ())
    for member in envelope.members:
        if member.name in gated_members:
            continue
        cells.append(
            MatrixCell(
                cell="envelope-member",
                features=all_features,
                member=member.name,
                evidence=EVIDENCE_BOOT,
                at=AT_CEILING,
                status=CELL_EXTERNAL,
                reason=(
                    f"{member.describe(envelope.gpus_per_node)} is claimed by the envelope "
                    f"and gated by no unit of this run"
                ),
            )
        )

    for index, raw in enumerate(_matrix_cells_from_spec(task_data)):
        name = str(raw.get("name") or f"cell{index}")
        features = tuple(str(f) for f in raw.get("features") or ())
        evidence = raw.get("evidence") or [EVIDENCE_BOOT, EVIDENCE_ACCURACY]
        evidence = [evidence] if isinstance(evidence, str) else list(evidence)
        at = raw.get("at") or AT_GATE
        members = raw.get("members") or [m.name for m in envelope.members]
        unit_gates = _unit_for(features, gates)
        for member in members:
            for kind in evidence:
                status, reason, unit = _classify(
                    unit_gates,
                    member=str(member),
                    kind=str(kind),
                    at=at,
                    envelope=envelope,
                    tuples=tuples,
                    prompt_tokens=raw.get("prompt_tokens"),
                    gate_base=gate_base,
                    add_probe=add_probe,
                )
                cells.append(
                    MatrixCell(
                        cell=name,
                        features=features,
                        member=str(member),
                        evidence=str(kind),
                        at=_at_label(at),
                        status=status,
                        reason=reason,
                        unit=unit,
                    )
                )

    return CertificationPlan(
        envelope=envelope,
        tuples=tuple(tuples),
        probes=tuple(probes.values()),
        cells=tuple(_dedupe_cells(cells)),
        errors=tuple(errors),
        warnings=tuple(warnings),
        world_size=world_size,
    )


def _classify(
    unit_gates: UnitGates | None,
    *,
    member: str,
    kind: str,
    at: Any,
    envelope: ServingEnvelope,
    tuples: Sequence[EngineTuple],
    prompt_tokens: Any,
    gate_base: Callable[..., EngineTuple | None],
    add_probe: Callable[[EngineTuple], None],
) -> tuple[str, str, str | None]:
    """Classify one declared cell as gated, probe or external."""
    if unit_gates is None:
        return CELL_EXTERNAL, "no unit of this run implements this feature set", None
    unit = unit_gates.unit
    if member not in unit_gates.members:
        return (
            CELL_EXTERNAL,
            f"{member} is not gated by the `{unit}` unit (gated: "
            f"{', '.join(unit_gates.members) or 'none'})",
            unit,
        )
    if kind not in unit_gates.kinds:
        return CELL_EXTERNAL, f"the `{unit}` unit runs no {kind} gate", unit
    if kind == EVIDENCE_WITNESS and member != unit_gates.witness_member:
        return (
            CELL_EXTERNAL,
            f"the `{unit}` witness runs at {unit_gates.witness_member} only",
            unit,
        )
    if at == AT_GATE:
        return CELL_GATED, f"the `{unit}` unit's {kind} gate", unit
    # The point moves only the axes it names; the rest stay at what this
    # kind's own gate launches. A long prompt is a boot probe's workload.
    target = _point_tuple(
        at,
        envelope,
        gate_base(unit, member, kind),
        unit=unit,
        member=member,
        prompt_tokens=int(prompt_tokens) if prompt_tokens else None,
    )
    if kind != EVIDENCE_BOOT:
        target = replace(target, prompt_tokens=None)
    existing = [t for t in tuples if t.unit == unit and t.member == member and t.gate == kind]
    if any(_reaches(t, target) for t in existing):
        return CELL_GATED, f"the `{unit}` {kind} gate already runs at {target.point()}", unit
    if kind == EVIDENCE_BOOT:
        add_probe(target)
        return CELL_PROBE, target.label(), unit
    return (
        CELL_EXTERNAL,
        f"the `{unit}` {kind} gate runs below {target.point()}; record engine evidence "
        f"at that point",
        unit,
    )


def _dedupe_cells(cells: Iterable[MatrixCell]) -> list[MatrixCell]:
    seen: dict[tuple[str, str, str, str], MatrixCell] = {}
    rank = {CELL_GATED: 0, CELL_PROBE: 1, CELL_EXTERNAL: 2}
    for cell in cells:
        key = (",".join(cell.features), cell.member, cell.evidence, cell.at)
        held = seen.get(key)
        # The cheapest way to get the evidence wins: a declared cell that an
        # existing gate already covers is not owed twice. On a tie the later
        # cell -- the declared one, over the derived default -- keeps its name.
        if held is None or rank[cell.status] <= rank[held.status]:
            seen[key] = cell
    return list(seen.values())


# ----------------------------------------------------------------- probes


def probe_flags(probe: EngineTuple) -> list[str]:
    """The engine flags that put a quickstart boot at ``probe``'s point."""
    flags = [f"--{axis}={probe.limit(axis)}" for axis in ENVELOPE_AXES if probe.limit(axis)]
    if probe.graph_batch_sizes:
        flags.append(f"{_GRAPH_LIST_FLAG} " + " ".join(str(g) for g in probe.graph_batch_sizes))
    if probe.graph_padding:
        flags.append(_GRAPH_PADDING_FLAG)
    return flags


def _without_axis_flags(flags: Iterable[str]) -> list[str]:
    """``flags`` with every envelope-axis and graph-list spelling removed."""
    owned = (*_FLAG_AXES, _GRAPH_LIST_FLAG, _GRAPH_PADDING_FLAG)
    kept: list[str] = []
    skip_values = False
    for flag in flags:
        word = str(flag)
        head = word.split()[0] if word.split() else word
        name = head.partition("=")[0]
        if name in owned:
            # A bare flag's value(s) may follow as separate elements.
            skip_values = "=" not in head and len(word.split()) == 1
            continue
        if skip_values and not word.startswith("--"):
            continue
        skip_values = False
        kept.append(word)
    return kept


def probe_task_data(
    task_data: Mapping[str, Any],
    unit: str,
    probe: EngineTuple,
) -> dict[str, Any]:
    """A copy of the spec whose runtime block puts ``unit``'s boot at the probe point.

    The probe IS the boot gate -- same launcher, same prompts, same keywords,
    same runtime switches -- with only the envelope's limits and capture list
    changed. A probe built any other way would measure two variables at once.
    """
    from .gates import resolve_runtime

    data = copy.deepcopy(dict(task_data))
    runtime = dict(resolve_runtime(task_data, unit))
    runtime["extra_flags"] = [
        *_without_axis_flags(runtime.get("extra_flags") or ()),
        *probe_flags(probe),
    ]
    runtime["cuda_graph"] = bool(probe.graph_batch_sizes)
    data["runtime"] = {"default": runtime}
    return data


def probe_prompts(base: Sequence[str], probe: EngineTuple) -> tuple[str, ...]:
    """The boot prompts, plus one generated long prompt when the probe asks for one."""
    if not probe.prompt_tokens:
        return tuple(base)
    from .witness import filler_words

    words = min(int(probe.prompt_tokens), MAX_WITNESS_PROMPT_WORDS)
    return (*base, filler_words(words))


def run_envelope_probe(
    environment: Any,
    *,
    task_data: Mapping[str, Any],
    probe: EngineTuple,
    spec: Any,
    runner: Any,
    features: Any,
) -> Any:
    """Boot the unit at ``probe``'s point. Returns a ``GateResult``.

    Recorded as the ``envelope`` gate with the point in its detail, so the
    gate history reads as a third kind of evidence rather than as a second
    boot that happened to pass.
    """
    from .gates import BootSpec, run_boot_gate

    probe_spec = BootSpec(
        prompts=probe_prompts(spec.prompts, probe),
        keywords=tuple(spec.keywords),
        max_tokens=int(spec.max_tokens),
    )
    result = run_boot_gate(
        environment,
        task_data=probe_task_data(task_data, probe.unit, probe),
        spec=probe_spec,
        runner=runner,
        features=features,
        unit=probe.unit,
        member=probe.member,
    )
    return replace(result, name=PROBE_GATE, detail=f"at {probe.point()}: {result.detail}")


# ----------------------------------------------------------------- evidence


def check_external_evidence(plan: CertificationPlan, workspace: Path | str) -> list[Any]:
    """Delivery checks: every external matrix cell has a recorded verdict.

    An evidence file is YAML with ``verdict: pass``, the ``tree`` it was
    measured on and the ``log`` that shows it. A missing or failing one is a
    failed check, because the envelope claims that cell and nothing in this
    run measured it.
    """
    from .delivery import Check

    root = Path(workspace) / EVIDENCE_DIRNAME
    checks: list[Check] = []
    for cell in plan.external_cells:
        path = root / f"{cell.slug}.yaml"
        name = f"matrix:{cell.slug}"
        if not path.is_file():
            checks.append(
                Check(
                    name=name,
                    passed=False,
                    detail=f"no evidence at {path}; {cell.reason}",
                )
            )
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            checks.append(Check(name=name, passed=False, detail=f"{path} is not YAML: {exc}"))
            continue
        if not isinstance(data, dict):
            checks.append(Check(name=name, passed=False, detail=f"{path} is not a mapping"))
            continue
        missing = [key for key in ("verdict", "tree", "log") if not data.get(key)]
        verdict = str(data.get("verdict", "")).strip().lower()
        if missing:
            detail = f"{path} lacks {', '.join(missing)}"
        elif verdict != "pass":
            detail = f"{path} records verdict {verdict!r}"
        else:
            detail = f"{path}: pass on {data['tree']} ({data['log']})"
        checks.append(Check(name=name, passed=not missing and verdict == "pass", detail=detail))
    return checks


# --------------------------------------------------------------- validation


def validate_serving_envelope(block: Any, errors: list[str]) -> None:
    """Schema-validate ``serving_envelope``. Appends to ``errors``."""
    where = SERVING_ENVELOPE_FIELD
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping, got {type(block).__name__}")
        return
    reject_unknown_keys(block, ENVELOPE_KEYS, errors, where=where)
    for key in ENVELOPE_REQUIRED:
        if block.get(key) is None:
            errors.append(
                f"'{where}.{key}' is required: the envelope is what every certification is "
                f"judged against, so none of its limits may be left to an engine default"
            )
    for key in ENVELOPE_AXES:
        value = block.get(key)
        if value is not None and not _positive_int(value):
            errors.append(f"'{where}.{key}' must be an integer >= 1, got {value!r}")
    for key in ("gpus_per_node", "probe_prompt_tokens"):
        value = block.get(key)
        if value is not None and not _positive_int(value):
            errors.append(f"'{where}.{key}' must be an integer >= 1, got {value!r}")
    probe_words = block.get("probe_prompt_tokens")
    if _positive_int(probe_words) and probe_words > MAX_WITNESS_PROMPT_WORDS:
        errors.append(
            f"'{where}.probe_prompt_tokens' is {probe_words}; the probe prompt travels as one "
            f"argv string, which Linux caps at 128 KiB, so the limit is "
            f"{MAX_WITNESS_PROMPT_WORDS}"
        )
    draft = block.get("max_draft_len")
    if draft is not None and not _non_negative_int(draft):
        errors.append(f"'{where}.max_draft_len' must be an integer >= 0, got {draft!r}")

    _validate_graph_policy(block.get("cuda_graph"), block.get("max_batch_size"), errors)

    members = block.get("members")
    if members is not None:
        if not isinstance(members, list) or not members:
            errors.append(f"'{where}.members' must be a non-empty list of parallel members")
        else:
            seen: set[str] = set()
            for index, member in enumerate(members):
                try:
                    parse_envelope_member(member)
                except ValueError as exc:
                    errors.append(f"'{where}.members[{index}]': {exc}")
                    continue
                if member in seen:
                    errors.append(f"'{where}.members' lists {member!r} twice")
                seen.add(member)


def _validate_graph_policy(block: Any, max_batch_size: Any, errors: list[str]) -> None:
    where = f"{SERVING_ENVELOPE_FIELD}.cuda_graph"
    if block is None or block is False:
        return
    if not isinstance(block, dict):
        errors.append(
            f"'{where}' must be false (eager only) or a mapping with 'batch_sizes' or "
            f"'padding'; got {block!r}"
        )
        return
    reject_unknown_keys(block, CUDA_GRAPH_KEYS, errors, where=where)
    sizes = block.get("batch_sizes")
    if sizes is None and "padding" not in block:
        errors.append(
            f"'{where}' needs 'batch_sizes' (the capture list) or 'padding' (the policy the "
            f"engine generates one from): the capture list decides which row counts every "
            f"captured call site is certified at"
        )
    if sizes is not None:
        if not isinstance(sizes, list) or not sizes or not all(_positive_int(s) for s in sizes):
            errors.append(f"'{where}.batch_sizes' must be a non-empty list of integers >= 1")
        elif _positive_int(max_batch_size) and max(sizes) > max_batch_size:
            errors.append(
                f"'{where}.batch_sizes' captures {max(sizes)}, over "
                f"{SERVING_ENVELOPE_FIELD}.max_batch_size={max_batch_size}"
            )
    padding = block.get("padding")
    if padding is not None and not isinstance(padding, bool):
        errors.append(f"'{where}.padding' must be true or false")
    bound = block.get("max_batch_size")
    if bound is not None:
        if not _positive_int(bound):
            errors.append(f"'{where}.max_batch_size' must be an integer >= 1")
        elif _positive_int(max_batch_size) and bound > max_batch_size:
            errors.append(
                f"'{where}.max_batch_size'={bound} exceeds "
                f"{SERVING_ENVELOPE_FIELD}.max_batch_size={max_batch_size}"
            )


def validate_feature_matrix(data: Mapping[str, Any], errors: list[str]) -> None:
    """Schema-validate ``feature_matrix`` against the spec's features and envelope."""
    where = FEATURE_MATRIX_FIELD
    block = data.get(FEATURE_MATRIX_FIELD)
    envelope = data.get(SERVING_ENVELOPE_FIELD)
    if not isinstance(envelope, dict):
        errors.append(
            f"'{where}' needs '{SERVING_ENVELOPE_FIELD}': a matrix cell names members and "
            f"points of the envelope, so there has to be one"
        )
        return
    if not isinstance(block, list):
        errors.append(f"'{where}' must be a list of cells, got {type(block).__name__}")
        return
    features = {str(f) for f in data.get(FEATURES_FIELD) or ()}
    claimed = {str(m) for m in envelope.get("members") or ()}
    names: set[str] = set()
    for index, cell in enumerate(block):
        cell_where = f"{where}[{index}]"
        if not isinstance(cell, dict):
            errors.append(f"'{cell_where}' must be a mapping")
            continue
        reject_unknown_keys(cell, MATRIX_CELL_KEYS, errors, where=cell_where)
        name = cell.get("name")
        if name is not None:
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", name):
                errors.append(
                    f"'{cell_where}.name' must be a file-name-safe string (letters, digits, "
                    f"'_', '.', '+', '-'): it names the cell's evidence file"
                )
            elif name in names:
                errors.append(f"'{cell_where}.name' {name!r} is used by another cell")
            names.add(str(name))
        cell_features = cell.get("features") or []
        if not isinstance(cell_features, list) or not all(
            isinstance(f, str) for f in cell_features
        ):
            errors.append(f"'{cell_where}.features' must be a list of feature names ([] = core)")
        else:
            unknown = sorted(set(cell_features) - features)
            if unknown:
                errors.append(
                    f"'{cell_where}.features' names {unknown}, which this run does not build "
                    f"(features: {sorted(features)})"
                )
        members = cell.get("members")
        if members is not None:
            if not isinstance(members, list) or not members:
                errors.append(f"'{cell_where}.members' must be a non-empty list")
            else:
                stray = sorted(str(m) for m in members if str(m) not in claimed)
                if stray:
                    errors.append(
                        f"'{cell_where}.members' names {stray}, which "
                        f"'{SERVING_ENVELOPE_FIELD}.members' does not claim"
                    )
        evidence = cell.get("evidence")
        kinds = [evidence] if isinstance(evidence, str) else evidence
        if kinds is not None and (
            not isinstance(kinds, list) or not kinds or any(k not in EVIDENCE_KINDS for k in kinds)
        ):
            errors.append(
                f"'{cell_where}.evidence' must be one of {list(EVIDENCE_KINDS)} or a list of "
                f"them, got {evidence!r}"
            )
        _validate_point(cell.get("at"), envelope, errors, f"{cell_where}.at")
        prompt_tokens = cell.get("prompt_tokens")
        if prompt_tokens is not None and (
            not _positive_int(prompt_tokens) or prompt_tokens > MAX_WITNESS_PROMPT_WORDS
        ):
            errors.append(
                f"'{cell_where}.prompt_tokens' must be an integer in 1..{MAX_WITNESS_PROMPT_WORDS}"
            )


def _validate_point(at: Any, envelope: Mapping[str, Any], errors: list[str], where: str) -> None:
    if at is None or at in (AT_GATE, AT_CEILING):
        return
    if not isinstance(at, dict) or not at:
        errors.append(
            f"'{where}' must be '{AT_GATE}', '{AT_CEILING}' or a mapping of "
            f"{list(ENVELOPE_AXES)} to values; got {at!r}"
        )
        return
    reject_unknown_keys(at, ENVELOPE_AXES, errors, where=where)
    for axis, value in at.items():
        if axis not in ENVELOPE_AXES:
            continue
        if not _positive_int(value):
            errors.append(f"'{where}.{axis}' must be an integer >= 1")
            continue
        bound = envelope.get(axis)
        if _positive_int(bound) and value > bound:
            errors.append(
                f"'{where}.{axis}'={value} is outside "
                f"{SERVING_ENVELOPE_FIELD}.{axis}={bound}; a point outside the envelope "
                f"is a plan change, not a matrix cell"
            )


def validate_certification_blocks(data: Mapping[str, Any], errors: list[str]) -> None:
    """The schema's entry point for both blocks."""
    if data.get(SERVING_ENVELOPE_FIELD) is not None:
        validate_serving_envelope(data[SERVING_ENVELOPE_FIELD], errors)
    if data.get(FEATURE_MATRIX_FIELD) is not None:
        validate_feature_matrix(data, errors)


# ---------------------------------------------------------------- migration


def migration_block(
    task_data: Mapping[str, Any],
    *,
    world_size: int,
    unit_kinds: Callable[[str], Iterable[str]] | None = None,
) -> str:
    """A ``serving_envelope`` + ``feature_matrix`` block for a spec that has none.

    Derived from the gate tuples, so adopting it unchanged keeps the old
    certified set -- now stated rather than implied. The comments say what to
    widen: the point of declaring an envelope is to claim what the target is
    deployed at, which is usually more than the gates ran.
    """
    gates = resolve_unit_gates(task_data, world_size, unit_kinds=unit_kinds)
    tuples = gate_tuples(task_data, gates)
    members = sorted({m for g in gates for m in g.members})
    hull = derived_envelope(tuples, members)
    graphs = list(hull.graph.batch_sizes)
    envelope: dict[str, Any] = {
        "max_batch_size": hull.max_batch_size or ENGINE_DEFAULTS["max_batch_size"],
        "max_num_tokens": hull.max_num_tokens or ENGINE_DEFAULTS["max_num_tokens"],
        "max_seq_len": hull.max_seq_len or 0,
        "cuda_graph": {"batch_sizes": graphs} if graphs else False,
        "members": members,
    }
    matrix: list[dict[str, Any]] = []
    for g in gates:
        matrix.append(
            {
                "name": f"{g.unit}-gates",
                "features": list(g.features),
                "members": list(g.members),
                "evidence": sorted(g.kinds - {EVIDENCE_WITNESS}),
                "at": AT_GATE,
            }
        )
        # The witness runs at one member; a cell claiming it at the others
        # would be owed as external evidence by the block meant to change
        # nothing.
        if g.witness_member is not None:
            matrix.append(
                {
                    "name": f"{g.unit}-witness",
                    "features": list(g.features),
                    "members": [g.witness_member],
                    "evidence": [EVIDENCE_WITNESS],
                    "at": AT_GATE,
                }
            )
    body = yaml.safe_dump(
        {SERVING_ENVELOPE_FIELD: envelope, FEATURE_MATRIX_FIELD: matrix},
        sort_keys=False,
        default_flow_style=None,
    )
    header = (
        "# ---------------------------------------------------- serving envelope\n"
        "# Generated from this spec's gate tuples: adopting it unchanged certifies\n"
        "# exactly what the gates ran -- the legacy certified set, now written down.\n"
        "# WIDEN IT to what the target is deployed at (max_num_tokens, max_seq_len,\n"
        "# max_batch_size, the capture list, every member including multi-node and\n"
        "# PP ones). Every limit raised here is probed at the ceiling by the run.\n"
    )
    if not hull.max_seq_len:
        header += (
            "# max_seq_len: no gate states one, so the engine used the checkpoint's\n"
            "# max_position_embeddings. Fill it in before using this block.\n"
        )
    return header + body


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent_flow.workflows.staircase_onboard.envelope",
        description="Serving-envelope tools for staircase-onboard task specs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    migrate = sub.add_parser(
        "migrate",
        help="print a serving_envelope + feature_matrix block derived from the gate tuples",
    )
    migrate.add_argument("task", help="path to task.yaml")
    migrate.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="world size of a unified run (default: from the member set)",
    )
    migrate.add_argument(
        "--write",
        action="store_true",
        help="append the block to the task file instead of printing it",
    )
    plan = sub.add_parser("plan", help="print the certification plan for a task.yaml")
    plan.add_argument("task", help="path to task.yaml")
    plan.add_argument("--world-size", type=int, default=None)
    args = parser.parse_args(argv)

    from .task_schema import load_and_validate_task_yaml

    try:
        data = load_and_validate_task_yaml(args.task)
    except TaskSchemaError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    world_size = args.world_size
    if world_size is None:
        segment = str(data.get("parallel"))
        _, size = parse_parallel_segment(segment)
        if size is None:
            _, size = parse_parallel_segment(members_for(data, None, 4)[0])
        world_size = size or 4

    if args.command == "plan":
        result = derive_certification(data, world_size=world_size)
        sys.stdout.write(result.render())
        for error in result.errors:
            sys.stderr.write(f"ERROR: {error}\n")
        return 1 if result.errors else 0

    if data.get(SERVING_ENVELOPE_FIELD) is not None:
        sys.stderr.write(f"{args.task} already declares '{SERVING_ENVELOPE_FIELD}'\n")
        return 1
    block = migration_block(data, world_size=world_size)
    if args.write:
        path = Path(args.task)
        text = path.read_text(encoding="utf-8")
        path.write_text(text.rstrip("\n") + "\n\n" + block, encoding="utf-8")
        sys.stdout.write(f"appended a serving_envelope block to {path}\n")
    else:
        sys.stdout.write(block)
    return 0


__all__ = [
    "AT_CEILING",
    "AT_GATE",
    "CELL_EXTERNAL",
    "CELL_GATED",
    "CELL_PROBE",
    "CUDA_GRAPH_KEYS",
    "ENGINE_DEFAULTS",
    "ENVELOPE_AXES",
    "ENVELOPE_KEYS",
    "EVIDENCE_DIRNAME",
    "EVIDENCE_KINDS",
    "CertificationPlan",
    "EngineTuple",
    "EnvelopeMember",
    "GraphPolicy",
    "MatrixCell",
    "PROBE_GATE",
    "ServingEnvelope",
    "UnitGates",
    "check_external_evidence",
    "compact_values",
    "default_unit_kinds",
    "derive_certification",
    "derived_envelope",
    "envelope_from_spec",
    "features_through",
    "gate_tuples",
    "generate_graph_batch_sizes",
    "migration_block",
    "options_tuple",
    "parse_envelope_member",
    "probe_flags",
    "probe_prompts",
    "probe_task_data",
    "resolve_graph_policy",
    "resolve_unit_gates",
    "run_envelope_probe",
    "runtime_tuple",
    "validate_certification_blocks",
    "validate_feature_matrix",
    "validate_serving_envelope",
]


if __name__ == "__main__":
    sys.exit(_main())
