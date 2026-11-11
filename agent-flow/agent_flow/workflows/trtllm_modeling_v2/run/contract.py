# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract: every check a run must pass, compiled from the resolved task spec.

The contract is the run's definition of "done". It names each check by unit,
gate and member. Nothing runs it while the workers build: a worker runs the
checks of its own unit by hand (``trtllm-modeling-v2 gate``) and its conclusion
is trusted. The contract is run once, whole, on the final commit
(``run/certify.py``), and a run is delivered only when every required check
passed on that one commit.

Verdicts are appended to ``contract_status.jsonl``, each with the tree it
measured (:func:`tree_id`: the commit and the native libraries built into the
checkout). A verdict about any other tree does not count (:func:`evaluate`).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ..checks import check_config, disagg, serve
from ..checks.check_config import order_members_by_cost
from ..spec import variants as variants_mod
from ..spec.task_schema import (
    DISAGG_FEATURE,
    PROFILE_BRINGUP,
    PROFILE_SMOKE,
    SPECULATORS,
    TaskSchemaError,
    assembly_units,
    members_for,
    parse_parallel_segment,
    profile_of,
    resolve_per_unit,
)

CONTRACT_FILENAME = "contract.yaml"
STATUS_FILENAME = "contract_status.jsonl"
#: Where the native libraries an engine loads are built: inside the checkout, ignored by git.
NATIVE_LIBRARY_DIRS: tuple[str, ...] = ("tensorrt_llm", "tensorrt_llm/libs")
NATIVE_LIBRARY_PATTERNS: tuple[str, ...] = ("*.so", "*.so.*")


@dataclass(frozen=True)
class Check:
    """One check: a gate, at one member, for one unit or configuration variant."""

    unit: str
    gate: str
    member: str
    required: bool = True

    @property
    def key(self) -> str:
        return f"{self.unit}:{self.gate}@{self.member}"


@dataclass(frozen=True)
class Contract:
    checks: tuple[Check, ...]
    #: Why a unit or variant owes less than every check at every member, one line each.
    notes: tuple[str, ...] = ()

    def required(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.required)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"checks": [asdict(c) for c in self.checks]}
        if self.notes:
            out["notes"] = list(self.notes)
        return out

    def render(self) -> str:
        """A markdown table, one row per check."""
        lines = ["| unit | gate | member | required |", "|---|---|---|---|"]
        for c in self.checks:
            lines.append(f"| {c.unit} | {c.gate} | {c.member} | {'yes' if c.required else 'no'} |")
        if self.notes:
            lines += ["", *(f"- {note}" for note in self.notes)]
        return "\n".join(lines)


def _width(member: str) -> int:
    """How many ranks ``member`` serves on; 0 when it does not say."""
    try:
        return parse_parallel_segment(member)[1] or 0
    except TaskSchemaError:
        return 0


def benchmark_members(task_data: Mapping[str, Any], unit: str, members: Sequence[str]) -> list[str]:
    """Where ``unit``'s accuracy benchmark runs: every member, or the one ``benchmark_members`` names."""
    named = (task_data.get("accuracy") or {}).get("benchmark_members")
    return [str(named)] if named in members else list(members)


def benchmark_member(task_data: Mapping[str, Any], members: Sequence[str]) -> tuple[str, str]:
    """The one member of ``members`` a configuration variant's benchmark runs at, and why.

    The member ``accuracy.benchmark_members`` names; else an attention-DP member,
    whose served engine takes the most paths (the router, dummy requests, per-rank
    caches); else the widest, the cheapest to start among equals.
    """
    named = (task_data.get("accuracy") or {}).get("benchmark_members")
    if named in members:
        return str(named), "named in accuracy.benchmark_members"
    ranked = order_members_by_cost(list(members))
    for member in ranked:
        if parse_parallel_segment(member)[0] == "dep":
            return member, "attention DP serves through the most paths"
    return max(ranked, key=_width), "the widest"


def compile_contract(task_data: Mapping[str, Any], world_size: int) -> Contract:
    """Every check the run's units and variants owe, units first, each key once.

    Each unit owes what its profile (``task_schema.profile_of``) asks:
    ``smoke`` its boot alone, ``bringup`` its checks at its cheapest member,
    ``certified`` what ``standard`` owes and the serve gate. ``disagg`` owes its
    disaggregated-serving test alone, once (``checks/disagg.py``): its boot and
    accuracy would run one aggregated engine, which the units before it prove.
    A storage or scheduling variant (one with no acceptance gate) boots at
    every member of its unit and runs its benchmark at one
    (:func:`benchmark_member`) unless it names its members; a variant of
    ``disagg`` owes nothing of its own, its axis being proven on core. A
    ``smoke`` or ``bringup`` run owes no variant and no one-rank serve check.
    """
    accuracy = task_data.get("accuracy") or {}
    serving = serve.ServeSpec.from_task(task_data).enabled
    units = assembly_units(task_data)
    lean = profile_of(task_data) in (PROFILE_SMOKE, PROFILE_BRINGUP)
    checks: dict[str, Check] = {}
    notes: list[str] = []

    def add(unit: str, gate: str, at: Iterable[str]) -> None:
        for member in at:
            check = Check(unit, gate, member)
            checks.setdefault(check.key, check)

    for index, unit in enumerate(units):
        level = profile_of(task_data, unit)
        members = order_members_by_cost(members_for(task_data, unit, world_size))
        if level == PROFILE_BRINGUP and len(members) > 1:
            notes.append(
                f"{unit}: the bringup profile gates it at {members[0]} only, not at "
                f"{', '.join(members[1:])}"
            )
            members = members[:1]
        implemented = set(units[1 : index + 1])
        if unit == DISAGG_FEATURE:
            # The test sets its own engines' layout on one node: it runs once, at the
            # member the request validates fully (benchmark_member).
            member, why = benchmark_member(task_data, members)
            add(unit, disagg.GATE_NAME, [member])
            notes.append(
                f"{unit}: its disaggregated-serving test alone proves it, once at {member} "
                f"({why}); its boot and accuracy would run one aggregated engine"
            )
            continue
        add(unit, "boot", members)
        if level == PROFILE_SMOKE:
            notes.append(f"{unit}: the smoke profile owes its boot alone")
            continue
        benchmark = resolve_per_unit(accuracy, "benchmark", unit) or "accuracy"
        add(unit, f"accuracy[{benchmark}]", benchmark_members(task_data, unit, members))
        if serving:
            add(unit, serve.GATE_NAME, [serve.serve_member(members)])
        # Speculative decoding preserves the output distribution by construction, so a
        # draft path that computes the wrong thing passes boot and accuracy and shows
        # only as speed: per-prompt acceptance is the detector.
        if SPECULATORS & implemented:
            add(unit, "acceptance", members)
    for name in () if lean else variants_mod.variants(task_data):
        level = profile_of(task_data, name)
        if level == PROFILE_BRINGUP:
            notes.append(f"{name}: the bringup profile gates no configuration variant")
            continue
        spec = check_config.variant_spec(task_data, name, world_size=world_size)
        members = order_members_by_cost(members_for(spec, name, world_size))
        base = variants_mod.base_unit(task_data, name)
        if base == DISAGG_FEATURE:
            notes.append(f"{name}: a variant of {base} owes no check; its axis is proven on core")
            continue
        benchmark = resolve_per_unit(spec.get("accuracy") or {}, "benchmark", name) or "accuracy"
        gate_names = {
            "boot": "boot",
            "accuracy": f"accuracy[{benchmark}]",
            "acceptance": "acceptance",
        }
        gates = variants_mod.variant_gates(task_data, name)
        if level == PROFILE_SMOKE:
            gates = ("boot",)
        everywhere = order_members_by_cost(members_for(task_data, base, world_size))
        named = variants_mod.variants(task_data)[name].get("members")
        if (
            variants_mod.anchor_state(task_data, name) == variants_mod.ANCHOR_RELATIVE
            and "accuracy" in gates
        ):
            notes.append(f"{name}: its anchor is {base}'s own score at the same member and commit")
        for gate in gates:
            at = members
            if gate == "boot" and "acceptance" not in gates:
                at = [*members, *(m for m in everywhere if m not in members)]
            elif gate == "accuracy" and not named and len(at) > 1:
                member, why = benchmark_member(spec, members)
                at = [member]
                notes.append(f"{name}: accuracy at {member} ({why}); its other members boot")
            add(name, gate_names[gate], at)
    if lean and variants_mod.variants(task_data):
        notes.append(
            f"the {profile_of(task_data)} profile gates no configuration variant: "
            + ", ".join(variants_mod.variants(task_data))
        )
    if units and serving and not lean:
        # The served target at one rank, where a user starts.
        single = serve.delivery_member(members_for(task_data, units[-1], world_size))
        if single is not None:
            add(units[-1], serve.GATE_NAME, [single])
    return Contract(checks=tuple(checks.values()), notes=tuple(notes))


def write_contract(workspace: Path, contract: Contract) -> Path:
    path = Path(workspace) / CONTRACT_FILENAME
    path.write_text(yaml.safe_dump(contract.as_dict(), sort_keys=False), encoding="utf-8")
    return path


# ------------------------------------------------------------------ the tree a verdict is about


def native_libraries(repo: Path) -> list[str]:
    """Each native library built into the checkout: ``<path> <size> <mtime_ns>``, sorted."""
    found: set[str] = set()
    for directory in NATIVE_LIBRARY_DIRS:
        for pattern in NATIVE_LIBRARY_PATTERNS:
            for path in (repo / directory).glob(pattern):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                found.add(f"{path.relative_to(repo).as_posix()} {stat.st_size} {stat.st_mtime_ns}")
    return sorted(found)


def tree_id(repo: str | Path, commit: str) -> str:
    """What a verdict is about: ``commit``, and the native libraries built into ``repo``.

    The libraries are outside git, so a rebuild changes what a check measures
    without a commit; their sizes and modification times are part of the id.
    ``""`` without a commit: nothing is established, and no verdict counts.
    """
    if not commit:
        return ""
    digest = hashlib.sha1()
    for library in native_libraries(Path(repo)):
        digest.update(os.fsencode(f"{library}\n"))
    return f"{commit}+{digest.hexdigest()[:12]}"


# ------------------------------------------------------------------------------ verdicts


@dataclass
class Verdict:
    unit: str
    gate: str
    member: str
    passed: bool
    detail: str = ""
    #: The tree the check measured (:func:`tree_id`).
    tree: str = ""
    #: The certification round that ran it.
    round: int = 0
    #: How long the check's gate ran, in seconds; 0 when it did not say.
    duration_s: float = 0.0
    #: The score a benchmark measured (``GateResult.measured``), pass or fail, the reference
    #: it was held to and its standard error: what the cross-member shift check reads.
    measured: float | None = None
    reference: float | None = None
    se: float | None = None
    #: The gate's one-line summary, with its numbers.
    summary: str = ""

    @property
    def key(self) -> str:
        return f"{self.unit}:{self.gate}@{self.member}"


_VERDICT_FIELDS = frozenset(f.name for f in fields(Verdict))


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


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
                try:
                    data = json.loads(line)
                except ValueError:
                    continue  # a torn line: the check it was for runs again
                if isinstance(data, dict):
                    verdicts.append(
                        Verdict(**{k: v for k, v in data.items() if k in _VERDICT_FIELDS})
                    )
        return cls(path=path, verdicts=verdicts)

    def record(self, check: Check, result: Any, *, tree: str, round: int = 0) -> Verdict:
        """Append one gate result for ``check``, measured on ``tree``.

        A result that says the infrastructure ended it measured nothing: it is
        recorded as not passed, never as a pass.
        """
        infrastructure = str(getattr(result, "infrastructure", "") or "")
        verdict = Verdict(
            unit=check.unit,
            gate=check.gate,
            member=check.member,
            passed=bool(result.passed) and not infrastructure,
            detail=str(result.detail)[:500],
            tree=tree,
            round=round,
            duration_s=round_seconds(getattr(result, "duration_s", 0.0)),
            measured=_number(getattr(result, "measured", None)),
            reference=_number(getattr(result, "reference", None)),
            se=_number(getattr(result, "se", None)),
            summary=str(result.summary())[:1000] if hasattr(result, "summary") else "",
        )
        self.verdicts.append(verdict)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(verdict)) + "\n")
        return verdict

    def on(self, tree: str, round: int | None = None) -> dict[str, Verdict]:
        """The last verdict per check key measured on ``tree`` (in ``round``, when given)."""
        last: dict[str, Verdict] = {}
        for v in self.verdicts:
            if tree and v.tree == tree and (round is None or v.round == round):
                last[v.key] = v
        return last


def round_seconds(value: Any) -> float:
    number = _number(value)
    return round(number, 3) if number is not None else 0.0


@dataclass(frozen=True)
class Evaluation:
    passed: tuple[Check, ...]
    failed: tuple[tuple[Check, Verdict], ...]
    missing: tuple[Check, ...]

    @property
    def holds(self) -> bool:
        return not self.failed and not self.missing

    def render(self) -> str:
        lines = [
            f"{len(self.passed)} passed, {len(self.failed)} failed, {len(self.missing)} missing"
        ]
        lines += [f"- FAILED {c.key}: {v.detail}" for c, v in self.failed]
        lines += [f"- MISSING {c.key}" for c in self.missing]
        return "\n".join(lines)


def evaluate(contract: Contract, verdicts: Mapping[str, Verdict]) -> Evaluation:
    """Each required check against its verdict in ``verdicts`` (:meth:`ContractStatus.on` a tree).

    A check with no verdict there is missing. Nothing is shared between checks
    and nothing is carried over from another tree.
    """
    passed, failed, missing = [], [], []
    for check in contract.required():
        verdict = verdicts.get(check.key)
        if verdict is None:
            missing.append(check)
        elif verdict.passed:
            passed.append(check)
        else:
            failed.append((check, verdict))
    return Evaluation(tuple(passed), tuple(failed), tuple(missing))


__all__ = [
    "CONTRACT_FILENAME",
    "STATUS_FILENAME",
    "Check",
    "Contract",
    "ContractStatus",
    "Evaluation",
    "Verdict",
    "benchmark_member",
    "benchmark_members",
    "compile_contract",
    "evaluate",
    "native_libraries",
    "tree_id",
    "write_contract",
]
