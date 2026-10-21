# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract: every check a run must pass, compiled from the resolved task spec.

The contract is the run's definition of "done". It names each check by unit,
gate and member, with a tier that says when it runs:

- ``inner``: after every change to the target (boot, accuracy, and the serve
  and logit gates where the spec turns them on or can derive them);
- ``proof``: what proves a feature is on (a witness, acceptance), and the
  checks of each configuration variant (``spec/variants.py``);
- ``delivery``: once, on the tree being delivered (the envelope boot, and the
  serve gate at one rank).

Verdicts are appended to ``contract_status.jsonl`` as gates return, keyed by
unit. Delivery evaluates the contract: every required check's latest verdict
must have passed, on the target as it is now. A verdict recorded against an
earlier state of the target (its ``marker``), or with other ``inputs``
(:func:`check_inputs`), is stale and counts as missing. Checks that carry one
``equivalence`` key (:func:`equivalence_key`) share a pass on the target as it
is now.

The first state of the target on which the first unit's boot and accuracy
hold at its narrowest member is where the bring-up was delivered
(:func:`bringup_checks`). The first on which every required check holds is
where correctness was delivered. Both are appended to ``correctness.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from ..checks import envelope_boot, logit_agreement, serve_gate
from ..checks.gates import (
    GreedySetting,
    load_llm_api_options,
    order_members_by_cost,
    resolve_boot_spec,
    resolve_discriminators,
    resolve_llm_api_options,
    resolve_max_draft_len,
    resolve_per_unit,
    resolve_runtime,
    speculation_flags,
)
from ..spec import variants as variants_mod
from ..spec.task_schema import (
    BENCHMARK_MEMBERS_EVERY,
    PROFILE_BRINGUP,
    TaskSchemaError,
    anchor_pending,
    assembly_units,
    members_for,
    parse_parallel_segment,
    profile_of,
    witness_enabled,
)
from .stages.common import ACCEPTANCE_GATED_UNITS, WITNESS_GATED_UNITS

TIER_INNER = "inner"
TIER_PROOF = "proof"
TIER_DELIVERY = "delivery"

CONTRACT_FILENAME = "contract.yaml"
STATUS_FILENAME = "contract_status.jsonl"
#: One line per milestone: each state of the target the contract first held on,
#: and the bring-up; the last of each tier is where it stands.
MILESTONE_FILENAME = "correctness.jsonl"
#: A milestone's tier: the first unit's boot and accuracy at its narrowest
#: member, or every required check.
MILESTONE_BRINGUP = "bringup"
MILESTONE_CORRECTNESS = "correctness"

#: The verdict name each witness kind returns (see witness.py).
_WITNESS_GATE = {
    "residency": "residency",
    "equivalence": "equivalence",
    "kv_quant": "kv_quant",
}


@dataclass(frozen=True)
class Check:
    """One check: a gate, at one member, for one unit."""

    unit: str
    gate: str
    member: str
    tier: str
    required: bool = True
    #: Checks with the same non-empty key measure the same engine on the same
    #: workload against the same criterion; see :func:`equivalence_key`.
    equivalence: str = field(default="", compare=False)
    #: What the check reads from outside the checkout, as a digest; see :func:`check_inputs`.
    inputs: str = field(default="", compare=False)

    @property
    def key(self) -> str:
        return f"{self.unit}:{self.gate}@{self.member}"


@dataclass(frozen=True)
class Contract:
    checks: tuple[Check, ...]
    #: Why a unit's accuracy benchmark is owed where it is and nowhere else,
    #: one line per unit whose benchmark is sampled (:func:`benchmark_members`).
    notes: tuple[str, ...] = ()

    def required(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.required)

    def by_tier(self, tier: str) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.tier == tier)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "checks": [
                {k: v for k, v in asdict(c).items() if k not in ("equivalence", "inputs") or v}
                for c in self.checks
            ]
        }
        if self.notes:
            out["notes"] = list(self.notes)
        return out

    def render(self) -> str:
        """A markdown table, one row per check."""
        lines = ["| unit | gate | member | tier | required |", "|---|---|---|---|---|"]
        for c in self.checks:
            lines.append(
                f"| {c.unit} | {c.gate} | {c.member} | {c.tier} | {'yes' if c.required else 'no'} |"
            )
        if self.notes:
            lines += ["", *(f"- {note}" for note in self.notes)]
        return "\n".join(lines)


def _witness_gate(unit: str) -> str:
    kind = WITNESS_GATED_UNITS[unit]
    return f"replay[{unit}]" if kind == "replay" else _WITNESS_GATE[kind]


#: Accuracy-block keys only other gates read. Every other key, one added
#: later included, is part of an accuracy check's equivalence.
_NOT_ACCURACY_INPUTS = frozenset(
    {"boot", "acceptance", "witness", "envelope", "serve", "logits", "benchmark_members"}
)


def _per_unit(key: str) -> Callable[[Mapping[str, Any], str], Any]:
    return lambda accuracy, unit: resolve_per_unit(accuracy, key, unit)


def _options(accuracy: Mapping[str, Any], unit: str) -> Any:
    """The unit's LLM-API options as their content; the path when the file cannot be read."""
    path = resolve_llm_api_options(accuracy, unit)
    if not path:
        return None
    if not Path(path).is_file():
        return {"missing": str(path)}
    try:
        return load_llm_api_options(path)
    except (OSError, ValueError, yaml.YAMLError):
        return {"unreadable": str(path)}


#: The accuracy-block keys the accuracy gate reads through a resolver, and
#: that resolver; the rest are compared as written (see `_as_written`).
_ACCURACY_RESOLVERS: dict[str, Callable[[Mapping[str, Any], str], Any]] = {
    "llm_api_options": _options,
    "discriminators": resolve_discriminators,
    **{
        key: _per_unit(key)
        for key in ("benchmark", "benchmark_args", "reference_yaml", "model_key", "samples")
    },
}


def _as_written(value: Any, unit: str) -> list[Any]:
    """``value`` as it applies to ``unit``, without assuming how it falls back.

    A mapping's own entry for ``unit`` when it has one; otherwise the value
    itself, which every unit without an entry shares whatever the fallback is.
    """
    if isinstance(value, Mapping) and unit in value:
        return ["unit", value[unit]]
    return ["run", value]


def equivalence_key(
    task_data: Mapping[str, Any], unit: str, gate: str, member: str, features: Sequence[str]
) -> str:
    """What decides a boot, accuracy or acceptance check's verdict, as one key; ``""`` for any other gate.

    Two checks with one key run one command line on the same tree and judge
    its output by one criterion, so a pass of either is a pass of both. Built
    from the spec and the content of the options file, never its path, and
    conservatively: where this cannot see how a value falls back, the keys
    differ rather than agree.
    """
    accuracy = task_data.get("accuracy") or {}
    if gate == "boot":
        try:
            workload = asdict(resolve_boot_spec(task_data, unit, features))
        except (OSError, TypeError, ValueError):
            return ""
        # Where the workload was declared is named in the verdict, not judged by it.
        workload.pop("source", None)
        inputs = {
            "runtime": resolve_runtime(task_data, unit),
            "speculation": speculation_flags(features),
            "llm_api_options": _options(accuracy, unit),
            "workload": workload,
        }
    elif gate == "acceptance":
        inputs = {
            "runtime": resolve_runtime(task_data, unit),
            "speculation": speculation_flags(
                features, max_draft_len=resolve_max_draft_len(accuracy, unit)
            ),
            "llm_api_options": _options(accuracy, unit),
            "acceptance": _as_written(accuracy.get("acceptance"), unit),
        }
    elif gate.startswith("accuracy["):
        inputs = {
            key: (
                _ACCURACY_RESOLVERS[key](accuracy, unit)
                if key in _ACCURACY_RESOLVERS
                else _as_written(accuracy[key], unit)
            )
            for key in sorted(set(accuracy) - _NOT_ACCURACY_INPUTS)
        }
    else:
        return ""
    payload = json.dumps({"gate": gate, "member": member, **inputs}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


#: File digests by ``(path, size, mtime)``: an input is read once per change.
_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_digest(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def _with_contents(value: Any, checkout: Path | None) -> Any:
    """``value`` with every path of a file outside ``checkout`` paired with its content's digest.

    A file in the checkout is part of the marker, its anchor tables by the
    entries the gates read; repeating it here would make an edit to another
    model's entry stale a verdict the marker keeps.
    """
    if isinstance(value, Mapping):
        return {str(k): _with_contents(v, checkout) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_with_contents(v, checkout) for v in value]
    if not isinstance(value, str) or not value or "\n" in value or len(value) > 4096:
        return value
    path = Path(value)
    try:
        if not path.is_absolute() or not path.is_file():
            return value
        resolved = path.resolve()
        if checkout is not None and resolved.is_relative_to(checkout):
            return value
        return {"path": value, "sha1": _file_digest(resolved)}
    except (OSError, ValueError):
        return value


def _for_unit(value: Any, unit: str, others: set[str]) -> list[Any]:
    """``value`` as it applies to ``unit``: its own entry, else the value less other units' entries."""
    if isinstance(value, Mapping) and unit in value:
        return ["unit", value[unit]]
    if isinstance(value, Mapping):
        return ["run", {k: v for k, v in value.items() if k not in others}]
    return ["run", value]


def check_inputs(
    task_data: Mapping[str, Any], unit: str, gate: str, member: str, features: Sequence[str]
) -> str:
    """What a check reads from outside the checkout, as one digest.

    Every file outside the checkout the spec names is read by content.
    Conservative: a change anywhere in what applies to the unit makes its
    checks stale; one another unit's entry alone does not.
    """
    accuracy = task_data.get("accuracy") or {}
    repo = str(task_data.get("trtllm_repo_path") or "")
    try:
        checkout = Path(repo).resolve() if repo else None
    except OSError:
        checkout = None
    others = (set(assembly_units(task_data)) | set(variants_mod.variants(task_data))) - {unit}
    entries = {
        key: (
            _ACCURACY_RESOLVERS[key](accuracy, unit)
            if key in _ACCURACY_RESOLVERS
            else _for_unit(accuracy[key], unit, others)
        )
        for key in sorted(accuracy)
    }
    # A block holding other units' entries alone says nothing to this one.
    entries = {k: v for k, v in entries.items() if v != ["run", {}]}
    environment = task_data.get("environment") or {}
    payload = {
        "gate": gate,
        "member": member,
        "features": sorted(str(f) for f in features),
        "checkpoint": task_data.get("checkpoint_path"),
        "parallel": task_data.get("parallel"),
        "runtime": resolve_runtime(task_data, unit),
        "accuracy": entries,
        "env": environment.get("env") if isinstance(environment, Mapping) else None,
        "variant": variants_mod.variants(task_data).get(unit),
    }
    text = json.dumps(_with_contents(payload, checkout), sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _width(member: str) -> int:
    """How many ranks ``member`` serves on; 0 when it does not say."""
    try:
        return parse_parallel_segment(member)[1] or 0
    except TaskSchemaError:
        return 0


def benchmark_members(
    task_data: Mapping[str, Any], unit: str, members: Sequence[str], features: Sequence[str]
) -> tuple[list[str], str]:
    """Where ``unit``'s accuracy benchmark runs, of ``members`` (in their order), and why.

    Every member, unless the logit gate is owed and
    ``accuracy.benchmark_members`` is not ``every``. Then: the narrowest
    member, the widest (the cheapest to start among equals), and every member
    whose served engine the logit gate does not run whole
    (:func:`~..checks.logit_agreement.unseen`); the speculator does not count
    as unseen where the unit's boot compares the HF reference's greedy tokens.
    The second value is a line for the contract's notes; ``""`` where every
    member runs it.
    """
    accuracy = task_data.get("accuracy") or {}
    every = list(members)
    if (
        len(every) <= 2
        or accuracy.get("benchmark_members") == BENCHMARK_MEMBERS_EVERY
        or not logit_agreement.LogitsSpec.from_task(task_data).enabled
    ):
        return every, ""
    ranked = order_members_by_cost(every)
    why: dict[str, str] = {
        min(ranked, key=_width): "the narrowest",
    }
    why.setdefault(max(ranked, key=_width), "the widest")
    greedy = GreedySetting.from_task(task_data, unit) is not None
    for member in every:
        unseen = logit_agreement.unseen(member, features)
        if greedy:
            unseen.pop(logit_agreement.UNSEEN_SPECULATION, None)
        if unseen and member not in why:
            why[member] = "; ".join(unseen.values())
    kept = [m for m in every if m in why]
    if len(kept) == len(every):
        return every, ""
    proven = [m for m in every if m not in why]
    note = (
        f"{unit}: accuracy at {', '.join(f'{m} ({why[m]})' for m in kept)}; "
        f"{', '.join(proven)} proven equivalent by the logit gate"
    )
    return kept, note


def compile_contract(task_data: Mapping[str, Any], world_size: int) -> Contract:
    """Every check the run's units and delivery owe, in the order they run.

    A unit's boot, accuracy and acceptance checks carry their
    :func:`equivalence_key`; a configuration variant's do not, so a variant
    never shares a verdict. Every check carries its :func:`check_inputs`.
    A storage or scheduling variant (one with no acceptance gate) boots at
    every member of its unit; its accuracy stays at the members it names.
    The ``bringup`` profile owes each unit's checks at its cheapest member
    only, and no variant's and no delivery check (``task_schema.PROFILES``).
    """
    accuracy = task_data.get("accuracy") or {}
    serving = serve_gate.ServeSpec.from_task(task_data).enabled
    logits = logit_agreement.LogitsSpec.from_task(task_data).enabled
    units = assembly_units(task_data)
    bringup = profile_of(task_data) == PROFILE_BRINGUP
    checks: list[Check] = []
    notes: list[str] = []
    for index, unit in enumerate(units):
        members = order_members_by_cost(members_for(task_data, unit, world_size))
        if bringup and len(members) > 1:
            notes.append(
                f"{unit}: the bringup profile gates it at {members[0]} only, not at "
                f"{', '.join(members[1:])}"
            )
            members = members[:1]
        features = units[1 : index + 1]
        implemented = set(features)
        benchmark = resolve_per_unit(accuracy, "benchmark", unit) or "accuracy"
        measured, note = benchmark_members(task_data, unit, members, features)
        if note:
            notes.append(note)

        def add(gate: str, tier: str, at=members, unit=unit, features=features) -> None:
            for m in at:
                same = equivalence_key(task_data, unit, gate, m, features)
                inputs = check_inputs(task_data, unit, gate, m, features)
                checks.append(Check(unit, gate, m, tier, equivalence=same, inputs=inputs))

        add("boot", TIER_INNER)
        if anchor_pending(accuracy, unit):
            notes.append(
                f"{unit}: accuracy anchor pending; the logit gate is its correctness oracle "
                "and the benchmark is not a check"
            )
        else:
            add(f"accuracy[{benchmark}]", TIER_INNER, at=measured)
        if serving:
            add(serve_gate.GATE_NAME, TIER_INNER, at=[serve_gate.serve_member(members)])
        if logits:
            add(logit_agreement.GATE_NAME, TIER_INNER)
        if unit in WITNESS_GATED_UNITS and witness_enabled(task_data):
            add(_witness_gate(unit), TIER_PROOF)
        if ACCEPTANCE_GATED_UNITS & implemented:
            add("acceptance", TIER_PROOF)
    for name in () if bringup else variants_mod.variants(task_data):
        spec = variants_mod.variant_spec(task_data, name, world_size=world_size)
        members = order_members_by_cost(members_for(spec, name, world_size))
        benchmark = resolve_per_unit(spec.get("accuracy") or {}, "benchmark", name) or "accuracy"
        gate_names = {
            "boot": "boot",
            "accuracy": f"accuracy[{benchmark}]",
            "acceptance": "acceptance",
        }
        gates = variants_mod.variant_gates(task_data, name)
        base = variants_mod.base_unit(task_data, name)
        if variants_mod.variant_anchor_pending(task_data, spec, name):
            gates = tuple(gate for gate in gates if gate != "accuracy")
        everywhere = order_members_by_cost(members_for(task_data, base, world_size))
        built = units[1 : units.index(base) + 1] if base in units else []
        for gate in gates:
            at = members
            if gate == "boot" and "acceptance" not in gates:
                at = [*members, *(m for m in everywhere if m not in members)]
            checks.extend(
                Check(
                    name,
                    gate_names[gate],
                    m,
                    TIER_PROOF,
                    inputs=check_inputs(spec, name, gate_names[gate], m, built),
                )
                for m in at
            )
    if bringup and variants_mod.variants(task_data):
        notes.append(
            "the bringup profile gates no configuration variant: "
            + ", ".join(variants_mod.variants(task_data))
        )
    last_features = units[1:]
    if units and not bringup and envelope_boot.EnvelopeSpec.from_task(task_data).enabled:
        last = units[-1]
        members = order_members_by_cost(members_for(task_data, last, world_size))
        checks.extend(
            Check(
                last,
                envelope_boot.GATE_NAME,
                m,
                TIER_DELIVERY,
                inputs=check_inputs(task_data, last, envelope_boot.GATE_NAME, m, last_features),
            )
            for m in members
        )
    if units and serving and not bringup:
        last = units[-1]
        single = serve_gate.delivery_member(members_for(task_data, last, world_size))
        if single is not None:
            inputs = check_inputs(task_data, last, serve_gate.GATE_NAME, single, last_features)
            checks.append(Check(last, serve_gate.GATE_NAME, single, TIER_DELIVERY, inputs=inputs))
    return Contract(checks=tuple(checks), notes=tuple(notes))


def write_contract(workspace: Path, contract: Contract) -> Path:
    path = Path(workspace) / CONTRACT_FILENAME
    path.write_text(yaml.safe_dump(contract.as_dict(), sort_keys=False), encoding="utf-8")
    return path


#: How a gate's warning that its engine left little memory free begins (``gates.MemoryMargin``).
LOW_MARGIN = "LOW MARGIN"


@dataclass
class Verdict:
    unit: str
    gate: str
    member: str
    passed: bool
    detail: str = ""
    tree_state: str = ""
    #: The target's identity when the check ran (``workflow._target_marker``).
    marker: str = ""
    #: The least device memory, in GiB, any engine of the gate left free once
    #: up; ``None`` when the gate read none.
    min_free_gib: float | None = None
    #: The gate warned that an engine left less free than its unit's margin.
    low_margin: bool = False
    #: The commit each command of the check imported the package under test
    #: from (``GateResult.imported``): one per leg or engine job that said.
    imported: list[str] = field(default_factory=list)
    #: Why the verdict says nothing about one tree (``GateResult.void``); such a
    #: verdict never passed.
    void: str = ""
    #: The check's :attr:`Check.inputs` when it ran.
    inputs: str = ""
    #: What the check read of the tree: the id of its list of Python files
    #: (``workflow._narrowed``), and the target's marker narrowed to them on
    #: the commit it measured. With both, the verdict holds on any later state
    #: of the target whose narrowed marker is the same (:func:`evaluate`).
    reads: str = ""
    reads_marker: str = ""
    #: How long the check's gate ran, in seconds; 0 when it did not say.
    duration_s: float = 0.0


#: Fields written only when they say something.
_UNSAID: dict[str, Any] = {
    "min_free_gib": None,
    "low_margin": False,
    "imported": [],
    "void": "",
    "inputs": "",
    "reads": "",
    "reads_marker": "",
    "duration_s": 0.0,
}
_VERDICT_FIELDS = frozenset(f.name for f in fields(Verdict))


def _seconds(value: Any) -> float:
    """A duration a gate reported, as seconds; 0 for one it did not."""
    try:
        return round(float(value or 0.0), 3)
    except (TypeError, ValueError):
        return 0.0


def engine_margin(result: Any) -> tuple[float | None, bool]:
    """The least free device memory in ``result``'s engine records, and whether it warned so."""
    free = [
        float(device["free_gib"])
        for record in getattr(result, "engine", ()) or ()
        for device in (record.get("memory") or {}).get("devices") or ()
        if device.get("free_gib") is not None
    ]
    warned = any(str(w).startswith(LOW_MARGIN) for w in getattr(result, "warnings", ()) or ())
    return (min(free) if free else None), warned


@dataclass
class ContractStatus:
    """The verdicts recorded against a contract, append-only on disk."""

    path: Path
    verdicts: list[Verdict] = field(default_factory=list)

    @classmethod
    def load(cls, workspace: Path) -> "ContractStatus":
        path = Path(workspace) / STATUS_FILENAME
        verdicts: list[Verdict] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    data = json.loads(line)
                    verdicts.append(
                        Verdict(**{k: v for k, v in data.items() if k in _VERDICT_FIELDS})
                    )
        return cls(path=path, verdicts=verdicts)

    def record(
        self,
        unit: str,
        result: Any,
        marker: str = "",
        inputs: str = "",
        *,
        reads: tuple[str, str] = ("", ""),
    ) -> Verdict:
        """Append one gate result, as returned by any gate, for ``unit``.

        ``marker`` is the target's state when the check was started; a result
        measured on a snapshot names the commit's own (``GateResult.marker``),
        and is recorded under that. ``inputs`` is the check's
        (:attr:`Check.inputs`). ``reads`` is what it read of the tree, as
        :attr:`Verdict.reads` and :attr:`Verdict.reads_marker`.
        """
        min_free, warned = engine_margin(result)
        void = str(getattr(result, "void", "") or "")
        verdict = Verdict(
            unit=unit,
            gate=str(result.name),
            member=str(result.member or ""),
            passed=bool(result.passed) and not void,
            detail=str(result.detail)[:500],
            tree_state=str(getattr(result, "tree_state", "") or ""),
            marker=str(getattr(result, "marker", "") or "") or marker,
            min_free_gib=min_free,
            low_margin=warned,
            imported=[str(i) for i in getattr(result, "imported", ()) or ()],
            void=void,
            inputs=inputs,
            reads=reads[0],
            reads_marker=reads[1],
            duration_s=_seconds(getattr(result, "duration_s", 0.0)),
        )
        self.verdicts.append(verdict)
        line = {k: v for k, v in asdict(verdict).items() if k not in _UNSAID or v != _UNSAID[k]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        return verdict

    def latest(self) -> dict[str, Verdict]:
        """The last verdict per check key."""
        last: dict[str, Verdict] = {}
        for v in self.verdicts:
            last[f"{v.unit}:{v.gate}@{v.member}"] = v
        return last

    def margins(self) -> dict[str, Verdict]:
        """Per check key, its last verdict whose gate read its engines' free memory or warned.

        Whatever the target it was measured on: a check whose last run failed
        before its engine was up keeps the margin its run before that left.
        """
        last: dict[str, Verdict] = {}
        for v in self.verdicts:
            if v.min_free_gib is not None or v.low_margin:
                last[f"{v.unit}:{v.gate}@{v.member}"] = v
        return last


@dataclass(frozen=True)
class Evaluation:
    passed: tuple[Check, ...]
    failed: tuple[tuple[Check, Verdict], ...]
    missing: tuple[Check, ...]
    #: Passed checks with no verdict of their own: each, and the equivalent
    #: check whose pass it holds by.
    shared: tuple[tuple[Check, Check], ...] = ()

    @property
    def holds(self) -> bool:
        return not self.failed and not self.missing

    def render(self) -> str:
        lines = [
            f"{len(self.passed)} passed, {len(self.failed)} failed, {len(self.missing)} missing"
        ]
        lines += [f"- {'VOID' if v.void else 'FAILED'} {c.key}: {v.detail}" for c, v in self.failed]
        lines += [f"- MISSING {c.key}" for c in self.missing]
        lines += [f"- SHARED {c.key}: holds by the pass of {s.key}" for c, s in self.shared]
        return "\n".join(lines)


def evaluate(
    contract: Contract,
    verdicts: Iterable[Verdict] | ContractStatus,
    marker: str | None = None,
    holds: Callable[[Verdict], bool] | None = None,
) -> Evaluation:
    """Each required check against its latest verdict.

    With ``marker``, a verdict recorded against another state of the target is
    stale and the check counts as missing, unless ``holds`` says it holds there
    still (what it read of the tree is unchanged: :attr:`Verdict.reads`); so is
    one whose :attr:`Verdict.inputs` are not the check's (one without inputs
    counts as the check's). With an established ``marker``, a check with no
    verdict of its own passes when an equivalent check (the same
    :attr:`Check.equivalence`) passed on this state of the target. A check's own
    verdict always wins over an equivalent's, so a check that failed here stays
    failed, and two verdicts recorded where the marker was not established are
    never taken to be about one tree.
    """
    latest = (
        verdicts.latest()
        if isinstance(verdicts, ContractStatus)
        else {f"{v.unit}:{v.gate}@{v.member}": v for v in verdicts}
    )
    if marker is not None:
        latest = {
            k: v
            for k, v in latest.items()
            if v.marker == marker
            or (marker and holds is not None and v.reads and v.reads_marker and holds(v))
        }
    checks = {c.key: c for c in contract.checks}
    latest = {
        k: v
        for k, v in latest.items()
        if not v.inputs or k not in checks or not checks[k].inputs or v.inputs == checks[k].inputs
    }
    holders: dict[str, Check] = {}
    if marker:
        for check in contract.checks:
            verdict = latest.get(check.key)
            if check.equivalence and verdict is not None and verdict.passed:
                holders.setdefault(check.equivalence, check)
    passed, failed, missing, shared = [], [], [], []
    for check in contract.required():
        verdict = latest.get(check.key)
        holder = holders.get(check.equivalence) if check.equivalence else None
        if verdict is None and holder is not None:
            passed.append(check)
            shared.append((check, holder))
        elif verdict is None:
            missing.append(check)
        elif verdict.passed:
            passed.append(check)
        else:
            failed.append((check, verdict))
    return Evaluation(tuple(passed), tuple(failed), tuple(missing), tuple(shared))


def measured_commit(
    status: ContractStatus, evaluation: Evaluation, checks: Iterable[Check], marker: str
) -> str | None:
    """The one commit every check of ``checks`` that passed on ``marker`` imported; ``None``.

    Each passed check counts by its own verdict, or by its equivalent's where
    it holds by that (:attr:`Evaluation.shared`). ``None`` when one of them
    names no commit, or two name different ones: a milestone names a commit
    only where its checks say which.
    """
    latest = status.latest()
    holder = {check.key: by.key for check, by in evaluation.shared}
    commits: set[str] = set()
    for check in checks:
        verdict = latest.get(holder.get(check.key, check.key))
        if (
            verdict is None
            or not verdict.passed
            or verdict.marker != marker
            or not verdict.imported
        ):
            return None
        commits.update(verdict.imported)
    return commits.pop() if len(commits) == 1 else None


@dataclass(frozen=True)
class Milestone:
    """A tier delivered on one commit of the target.

    ``correctness``: every required check held there. ``bringup``: the first
    unit's boot and accuracy held at its narrowest member (:func:`bringup_checks`).
    ``marker`` is the state of the target the checks measured, and ``sha`` the
    commit that holds exactly that state.
    """

    sha: str
    marker: str
    #: The branch that names ``sha`` (``<target branch>-correct``, ``-bringup``);
    #: ``""`` when none was written.
    ref: str = ""
    #: When it was recorded, and how long after the run opened (``None``: not known).
    at: float = 0.0
    elapsed_s: float | None = None
    #: How many required checks held.
    checks: int = 0
    #: :data:`MILESTONE_CORRECTNESS` or :data:`MILESTONE_BRINGUP`.
    tier: str = MILESTONE_CORRECTNESS

    def describe(self, marker: str | None = None, evaluation: Evaluation | None = None) -> str:
        """One sentence: where the tier was delivered and, given the target now, whether it still is."""
        what = "Bring-up" if self.tier == MILESTONE_BRINGUP else "Correctness"
        line = f"{what} delivered at {self.sha[:10]}"
        if self.ref:
            line += f" (`{self.ref}`)"
        if self.elapsed_s is not None:
            line += f", {self.elapsed_s / 3600:.1f} h after the run started"
        if marker is None or self.tier == MILESTONE_BRINGUP:
            return line + "."
        if marker == self.marker:
            return line + "; the target is still there."
        line += "; the target has moved since"
        if evaluation is not None and not evaluation.holds:
            line += (
                f", and on it {len(evaluation.failed)} required checks failed and "
                f"{len(evaluation.missing)} are stale or missing"
            )
        return line + ". It moves forward once the contract holds on a later commit."


def milestones(workspace: Path, tier: str | None = MILESTONE_CORRECTNESS) -> list[Milestone]:
    """Every milestone of ``tier`` the run recorded, in order; every one with ``tier=None``."""
    path = Path(workspace) / MILESTONE_FILENAME
    if not path.is_file():
        return []
    known = {f.name for f in fields(Milestone)}
    recorded = [
        Milestone(**{k: v for k, v in json.loads(line).items() if k in known})
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [m for m in recorded if tier is None or m.tier == tier]


def latest_milestone(workspace: Path) -> Milestone | None:
    """Where correctness stands delivered, or None before the contract first held."""
    recorded = milestones(workspace)
    return recorded[-1] if recorded else None


def bringup_milestone(workspace: Path) -> Milestone | None:
    """Where the bring-up was delivered, or None before it was."""
    recorded = milestones(workspace, MILESTONE_BRINGUP)
    return recorded[0] if recorded else None


def record_milestone(workspace: Path, milestone: Milestone) -> None:
    """Append ``milestone`` to ``correctness.jsonl``."""
    path = Path(workspace) / MILESTONE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(milestone)) + "\n")


def standing(evaluation: Evaluation, workspace: Path, marker: str | None = None) -> str:
    """``evaluation`` rendered, then where the bring-up and correctness stand delivered once they have been.

    ``marker`` is the state of the target ``evaluation`` was read on; ``""`` or ``None``
    when it is not established, and then nothing is said of whether the target moved.
    """
    rendered = evaluation.render()
    bringup = bringup_milestone(workspace)
    if bringup is not None:
        rendered = f"{rendered}\n\n{bringup.describe()}"
    milestone = latest_milestone(workspace)
    if milestone is None:
        return rendered
    return f"{rendered}\n\n{milestone.describe(marker or None, evaluation)}"


# ------------------------------------------------------------------- the bring-up


def bringup_checks(contract: Contract) -> tuple[Check, ...]:
    """The checks whose passes deliver the bring-up: the first unit's boot and accuracy at its narrowest member.

    Empty for a contract with no such checks.
    """
    if not contract.checks:
        return ()
    first = contract.checks[0].unit
    own = [
        c
        for c in contract.required()
        if c.unit == first
        and c.tier == TIER_INNER
        and (c.gate == "boot" or c.gate.startswith("accuracy["))
    ]
    measured = {c.member for c in own if c.gate.startswith("accuracy[")}
    if not measured:
        return ()
    narrowest = min(order_members_by_cost(sorted(measured)), key=_width)
    return tuple(c for c in own if c.member == narrowest)


def bringup_holds(contract: Contract, evaluation: Evaluation) -> bool:
    """Whether every bring-up check passed in ``evaluation`` (its own verdict or an equivalent's)."""
    owed = bringup_checks(contract)
    passed = {c.key for c in evaluation.passed}
    return bool(owed) and all(c.key in passed for c in owed)


# ------------------------------------------------------------------- the cells

#: Where each cell's latest holding commit is kept, and the commit holding the most at once.
CELLS_FILENAME = "cells.json"


def cell_name(cell: tuple[str, str]) -> str:
    """``core@tp1`` for ``("core", "tp1")``."""
    return f"{cell[0]}@{cell[1]}"


def cells(contract: Contract) -> dict[tuple[str, str], tuple[Check, ...]]:
    """The required checks by cell: a unit or a variant at one member, in the contract's order."""
    out: dict[tuple[str, str], list[Check]] = {}
    for check in contract.required():
        out.setdefault((check.unit, check.member), []).append(check)
    return {cell: tuple(checks) for cell, checks in out.items()}


def cells_holding(contract: Contract, evaluation: Evaluation) -> list[tuple[str, str]]:
    """The cells every one of whose checks passed in ``evaluation``, in the contract's order."""
    passed = {c.key for c in evaluation.passed}
    return [cell for cell, owed in cells(contract).items() if all(c.key in passed for c in owed)]


def load_cells(workspace: Path) -> dict[str, Any]:
    """``cells.json``: ``{"cells": {name: {sha, at}}, "best": {sha, cells, ref, at} | None}``."""
    path = Path(workspace) / CELLS_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {"cells": dict(data.get("cells") or {}), "best": data.get("best")}


def save_cells(workspace: Path, record: Mapping[str, Any]) -> None:
    path = Path(workspace) / CELLS_FILENAME
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(partial, path)


# ---------------------------------------------------------------- memory re-checks


def tightest_checks(contract: Contract, facts: Mapping[str, Any]) -> tuple[str, ...]:
    """The required checks of the configuration whose engine left the least memory free.

    ``facts`` is ``runtime_facts.yaml`` read: its envelope names the
    ``<unit>@<member>`` that left the least free. The checks of the gates that
    measured it there come first, the one whose engine left the least first;
    then that configuration's other inner-tier checks. Empty when the facts
    name no such configuration, or the contract owes nothing there.
    """
    envelope = facts.get("envelope") if isinstance(facts, Mapping) else None
    least = envelope.get("free_gib") if isinstance(envelope, Mapping) else None
    at = str(least.get("at") or "") if isinstance(least, Mapping) else ""
    unit, _, member = at.partition("@")
    if not unit or not member:
        return ()
    owed = [c for c in contract.required() if c.unit == unit and c.member == member]
    members = facts.get("members") if isinstance(facts.get("members"), Mapping) else {}
    entry = members.get(at) if isinstance(members, Mapping) else None
    engines = entry.get("engines") if isinstance(entry, Mapping) else None
    free: dict[str, float] = {}
    for engine in engines if isinstance(engines, list) else ():
        if isinstance(engine, Mapping) and isinstance(engine.get("free_gib"), (int, float)):
            gate = str(engine.get("gate") or "")
            free[gate] = min(free.get(gate, float("inf")), float(engine["free_gib"]))
    measured = sorted((c for c in owed if c.gate in free), key=lambda c: free[c.gate])
    rest = [c for c in owed if c.gate not in free and c.tier == TIER_INNER]
    return tuple(c.key for c in (*measured, *rest))


__all__ = [
    "CELLS_FILENAME",
    "CONTRACT_FILENAME",
    "Check",
    "Contract",
    "ContractStatus",
    "Evaluation",
    "LOW_MARGIN",
    "MILESTONE_BRINGUP",
    "MILESTONE_CORRECTNESS",
    "MILESTONE_FILENAME",
    "Milestone",
    "STATUS_FILENAME",
    "TIER_DELIVERY",
    "TIER_INNER",
    "TIER_PROOF",
    "Verdict",
    "benchmark_members",
    "bringup_checks",
    "bringup_holds",
    "bringup_milestone",
    "cell_name",
    "cells",
    "cells_holding",
    "check_inputs",
    "compile_contract",
    "engine_margin",
    "equivalence_key",
    "evaluate",
    "latest_milestone",
    "load_cells",
    "measured_commit",
    "milestones",
    "record_milestone",
    "save_cells",
    "standing",
    "tightest_checks",
    "write_contract",
]
