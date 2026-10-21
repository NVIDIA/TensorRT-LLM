# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The keep gate: the contract's accuracy gate, on the tree a keep would make the best.

A candidate the e2e judge calls a WIN still changes numerics (a fusion, a rounding, a
GEMM tactic), so before it is kept the campaign measures the stacked tree the way the
contract measures the target. Nothing about that measurement is configured under
``perf:``: the benchmark, the anchor row (``reference_yaml`` + ``model_key`` +
``discriminators``), the floor rule and its sample count, the unit's LLM-API options
file, its runtime and ``benchmark_args`` are the contract's, read by the contract's own
code (:func:`..checks.gates.run_accuracy_gate`). ``perf.campaign.accuracy`` only names
the unit (default ``core``) and the members (default ``tp1`` and the unit's widest
member), or declares ``gate: off``.

A gate is a few spool jobs on the dev sessions, side by side where their devices allow:
one ``trtllm-eval`` per member, and the target's own test suite with the modeling_v2
contract tests (``tests``). A member's verdict is ``run_accuracy_gate``'s own. The gate
calls it twice: once with a runner that only takes the command it would run (the job's
script), and once the job has finished with a runner that hands back what the job
printed. The command, the score it parses, the floor and the verdict are therefore the
contract's code, and nothing here restates them.

Under ``<campaign>/gates/<key>/``: ``<part>/result.json`` per part (a member, or
``tests``), ``<part>/attempt-<n>.json`` for each run that reached no verdict, and
``keep.json``, the gate's verdict: every member's score against the anchor and the
floor, the unit, the members and the tree. Only a gate whose every part passed keeps.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..checks import gates as gates_mod
from ..infra.environment import RunEnvironment, resolve_environment
from ..infra.execution.commands import TIMEOUT_RETURNCODE, CommandResult
from ..infra.execution.spool import SESSION_LOSS_REASONS
from ..infra.substrate.scheduler import RC_MIXED_CHECKOUT, RC_REFUSED, RC_WALL_LIMIT
from ..infra.vcs import git_output
from ..spec import variants as variants_mod
from ..spec.task_schema import (
    TaskSchemaError,
    assembly_units,
    members_for,
    parse_parallel_segment,
    resolve_per_unit,
    widest_world_size,
)
from .measure import INFRA_RCS
from .schema import keep_gate_off, validate_campaign_accuracy

DEFAULT_UNIT = "core"
TESTS_PART = "tests"
KEEP_FILE = "keep.json"
PART_FILE = "result.json"
#: The suite's wall limit; a member's is the contract's own (``ACCURACY_TIMEOUT_S``).
TESTS_WALL = "1h"
PASS, FAIL, ERROR = "PASS", "FAIL", "ERROR"
#: Receipts of a job that measured nothing about the tree: the substrate ended it, the
#: session refused it, or it imported another checkout than the one under test.
UNMEASURED_RCS = INFRA_RCS | {RC_REFUSED, RC_MIXED_CHECKOUT}
#: Why a member's command was only taken, not run: the job runs it.
_QUEUED = "queued as a spool job; judged when it finishes"


class KeepGateError(ValueError):
    """The keep gate cannot be defined from the contract, and is not declared off."""


@dataclass(frozen=True)
class GateJob:
    """One part of a keep gate as a spool job: the script, and what it asks of a session."""

    part: str
    script: str
    gpus: int
    wall: str


def _width(member: str) -> int:
    return parse_parallel_segment(member)[1] or 0


#: The widest members are told apart by how much of the forward they run differently
#: from tp1: attention data parallelism and expert parallelism first.
_UNLIKE_TP1 = {"dep": 0, "tep": 1, "tp": 2}


def default_members(members: Sequence[str]) -> list[str]:
    """``tp1`` (else the narrowest member) and the widest member, the least like tp1 among equals."""
    first = "tp1" if "tp1" in members else min(members, key=lambda m: (_width(m), m))
    wide = max(_width(m) for m in members)
    last = min(
        (m for m in members if _width(m) == wide),
        key=lambda m: (_UNLIKE_TP1.get(parse_parallel_segment(m)[0], len(_UNLIKE_TP1)), m),
    )
    return list(dict.fromkeys([first, last]))


class _Capture:
    """A runner that takes the command a gate would run and runs nothing."""

    def __init__(self) -> None:
        self.script: str | None = None
        self.timeout = gates_mod.ACCURACY_TIMEOUT_S

    def __call__(
        self, command: str, env: Mapping[str, str] | None = None, timeout: int = 0
    ) -> CommandResult:
        exports = "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in (env or {}).items())
        self.script, self.timeout = exports + command, timeout or self.timeout
        return CommandResult(TIMEOUT_RETURNCODE, timed_out=True, infrastructure=_QUEUED)


@dataclass(frozen=True)
class KeepGate:
    """What a keep must pass on the tree it would keep; see the module docstring."""

    #: The spec the unit's checks run under (a variant's overlay applied).
    spec: Mapping[str, Any]
    environment: RunEnvironment
    unit: str = DEFAULT_UNIT
    #: The members the accuracy gate runs at; none when ``perf.campaign.accuracy.gate: off``.
    members: tuple[str, ...] = ()
    #: The contract's anchor for ``unit``.
    reference: float | None = None
    benchmark: str = ""
    #: The suite run before a keep, repo-relative (``perf.campaign.gate_tests``).
    tests: tuple[str, ...] = ()
    #: The floor the anchor gets before any output is read, for the record.
    planned_floor: Mapping[str, Any] = field(default_factory=dict)

    @property
    def suite_devices(self) -> int:
        """The devices the suite's job holds: the widest member's.

        The target's suite carries multi-rank tests (a collective's op matrix
        needs as many devices as the widest member); on fewer devices such a
        test fails for the job's size, not for the candidate.
        """
        return max((_width(m) or 1 for m in self.members), default=1)

    @property
    def ungated(self) -> bool:
        """No accuracy gate: a keep passes the suite alone and is recorded as ungated."""
        return not self.members

    def parts(self) -> list[str]:
        """The widest member first (it needs the most devices free), then the rest, then the suite."""
        members = sorted(self.members, key=lambda m: (-_width(m), m))
        return [*members, *([TESTS_PART] if self.tests else [])]

    def describe(self) -> str:
        """One line for the journal: what a keep must pass."""
        suite = f"the target's suite ({len(self.tests)} paths)" if self.tests else "no suite"
        if self.ungated:
            return (
                f"no accuracy gate (perf.campaign.accuracy.gate: off), {suite}; every keep "
                "is recorded as ungated"
            )
        floor = self.planned_floor.get("rule") or "floor resolved from the eval's output"
        return (
            f"the contract's {self.benchmark} gate at {self.unit}@{','.join(self.members)}, "
            f"anchor {self.reference:.4f} ({floor}), then {suite}"
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "members": list(self.members),
            "benchmark": self.benchmark,
            "anchor": self.reference,
            "floor": self.planned_floor.get("value"),
            "floor_rule": self.planned_floor.get("rule", ""),
            "tests": list(self.tests),
            "ungated": self.ungated,
        }

    # ------------------------------------------------------------------ the jobs
    def needs(self, part: str) -> tuple[int, str]:
        """The devices and the wall limit the job of ``part`` asks a session for."""
        if part == TESTS_PART:
            return self.suite_devices, TESTS_WALL
        return _width(part) or 1, str(gates_mod.ACCURACY_TIMEOUT_S)

    def prepare(self, part: str, tree: Path, out: Path) -> GateJob | dict:
        """The job that runs ``part`` on ``tree``, or the part's verdict when it needs no job.

        A member whose gate refuses before running anything (the target's path is dirty
        in ``tree``) has its verdict at once.
        """
        out.mkdir(parents=True, exist_ok=True)
        if part == TESTS_PART:
            return self._tests_job()
        capture = _Capture()
        result = self._accuracy(part, tree, out, capture)
        if capture.script is None:
            return self._member_verdict(part, result, rc=None)
        (out / "command.sh").write_text(capture.script + "\n")
        return GateJob(part, capture.script, _width(part) or 1, str(int(capture.timeout)))

    def judge(
        self, part: str, tree: Path, out: Path, *, rc: int, output: str, record: Mapping[str, Any]
    ) -> dict:
        """The verdict of ``part`` from its job: return code, log and receipt record."""
        if part == TESTS_PART:
            return self._tests_verdict(rc, output)
        if rc in UNMEASURED_RCS:
            return {
                "verdict": ERROR,
                "part": part,
                "rc": rc,
                "infrastructure": rc in INFRA_RCS,
                "detail": f"the job measured nothing (rc {rc}, {record.get('reason') or 'no reason'})",
            }
        submitted = out / "command.sh"
        queued = submitted.read_text().rstrip("\n") if submitted.exists() else None
        lost = record.get("reason") in SESSION_LOSS_REASONS
        timed_out = bool(record.get("timed_out")) or rc == RC_WALL_LIMIT

        def replay(command: str, env: Mapping[str, str] | None = None, timeout: int = 0):
            capture = _Capture()
            capture(command, env, timeout)
            if queued is not None and capture.script != queued:
                return CommandResult(
                    rc,
                    timed_out=True,
                    infrastructure="the gate's command changed after its job was queued",
                )
            why = f"the session that ran it was lost ({record.get('reason')})" if lost else ""
            return CommandResult(rc, stdout=output, timed_out=timed_out or lost, infrastructure=why)

        result = self._accuracy(part, tree, out, replay)
        result = replace(result, duration_s=float(record.get("duration_s") or 0.0))
        return self._member_verdict(part, result, rc=rc)

    def verdict(self, parts: Mapping[str, Mapping[str, Any]], tree: Path) -> dict:
        """The keep verdict: PASS only when every part passed."""
        failed = [p for p, v in parts.items() if v.get("verdict") == FAIL]
        broken = [p for p, v in parts.items() if v.get("verdict") not in (PASS, FAIL)]
        missing = [p for p in self.parts() if p not in parts]
        verdict = FAIL if failed else (ERROR if broken or missing else PASS)
        scores = {m: parts[m].get("score") for m in self.members if m in parts}
        floors = [
            parts[m]["floor"] for m in self.members if parts.get(m, {}).get("floor") is not None
        ]
        known = [s for s in scores.values() if s is not None]
        rule = next(
            (
                parts[m].get("floor_rule")
                for m in self.members
                if parts.get(m, {}).get("floor_rule")
            ),
            self.planned_floor.get("rule", ""),
        )
        if verdict == PASS:
            reason = self._pass_reason(scores, max(floors) if floors else None, parts)
        else:
            reason = "; ".join(
                [
                    f"{p}: {parts[p].get('detail') or parts[p].get('summary')}"
                    for p in failed + broken
                ]
                + ([f"not run: {', '.join(missing)}"] if missing else [])
            )
        return {
            "verdict": verdict,
            "score": min(known) if known else None,
            "anchor": self.reference,
            "floor": max(floors) if floors else self.planned_floor.get("value"),
            "floor_rule": rule,
            "scores": scores,
            "unit": self.unit,
            "members": list(self.members),
            "benchmark": self.benchmark,
            "tests": parts.get(TESTS_PART, {}).get("summary") if self.tests else None,
            "ungated": self.ungated,
            "tree": str(tree),
            "commit": (git_output(tree, "rev-parse", "HEAD") or "").strip(),
            "reason": reason,
            "parts": {p: v.get("verdict") for p, v in parts.items()},
        }

    # ------------------------------------------------------------------ members
    def _accuracy(self, member: str, tree: Path, out: Path, runner) -> gates_mod.GateResult:
        """The contract's accuracy gate of ``member`` on ``tree``, through ``runner``."""
        return gates_mod.run_accuracy_gate(
            replace(self.environment, world_size=_width(member) or 1),
            task_data={**self.spec, "trtllm_repo_path": str(tree)},
            reference=float(self.reference or 0.0),
            runner=runner,
            unit=self.unit,
            member=member,
            workdir=out,
        )

    def _member_verdict(self, member: str, result: gates_mod.GateResult, *, rc: int | None) -> dict:
        verdict = PASS if result.passed else (ERROR if result.infrastructure else FAIL)
        return {
            **json.loads(json.dumps(asdict(result), default=str)),
            "verdict": verdict,
            "part": member,
            "score": result.measured,
            "anchor": result.reference,
            "rc": rc,
            "summary": result.summary(),
        }

    def _pass_reason(self, scores: Mapping[str, Any], floor: float | None, parts: Mapping) -> str:
        bits = []
        if scores:
            measured = ", ".join(
                f"{m} {s:.2f}" if s is not None else f"{m} -" for m, s in scores.items()
            )
            bits.append(
                f"{self.benchmark} at {self.unit}: {measured} >= floor "
                + (f"{floor:.2f}" if floor is not None else "-")
                + f" (anchor {self.reference:.2f})"
            )
        else:
            bits.append("no accuracy gate (perf.campaign.accuracy.gate: off)")
        if self.tests:
            bits.append(f"tests: {parts.get(TESTS_PART, {}).get('summary', '')}")
        return "; ".join(bits)

    # ------------------------------------------------------------------ the suite
    def _tests_job(self) -> GateJob:
        """``pytest`` from ``tests/unittest`` of the job's tree (its working directory)."""
        paths = [
            str(Path(t).relative_to("tests/unittest")) if t.startswith("tests/unittest/") else t
            for t in self.tests
        ]
        pytest = replace(self.environment, world_size=1).python_command(
            ["-m", "pytest", "-q", "-p", "no:cacheprovider", *paths], single_process=True
        )
        return GateJob(TESTS_PART, f"cd tests/unittest && {pytest}", self.suite_devices, TESTS_WALL)

    def _tests_verdict(self, rc: int, output: str) -> dict:
        """The suite's verdict from pytest's return code and output.

        rc 1 (tests failed) is the candidate's; 2-5 (interrupted, internal or usage error,
        nothing collected) and the substrate's own codes are the tree's environment: ERROR.
        """
        lines = [
            line
            for line in output.splitlines()
            if line.strip() and not (line.startswith("=== ") and line.endswith(" ==="))
        ]
        summary = lines[-1] if lines else ""
        failed = [line.split(" ", 1)[1] for line in lines if line.startswith("FAILED ")][:50]
        verdict = PASS if rc == 0 else (FAIL if rc == 1 else ERROR)
        detail = {
            PASS: f"target tests passed: {summary}",
            FAIL: f"target tests failed: {summary}",
            ERROR: f"target tests could not run (rc {rc}): {summary}",
        }[verdict]
        return {
            "verdict": verdict,
            "part": TESTS_PART,
            "rc": rc,
            "infrastructure": rc in INFRA_RCS,
            "summary": summary,
            "detail": detail,
            "failed": failed,
            "paths": list(self.tests),
        }


# --------------------------------------------------------------------- resolution
def _unit_spec(task: Mapping[str, Any], unit: str, workspace: Path) -> dict:
    """The spec ``unit``'s checks run under: the task's, or a variant's overlay of it."""
    if variants_mod.is_variant(task, unit):
        return variants_mod.variant_spec(
            task,
            unit,
            world_size=widest_world_size(task, 1),
            options_dir=Path(workspace) / variants_mod.OPTIONS_DIRNAME,
        )
    if unit not in assembly_units(task):
        known = [*assembly_units(task), *sorted(variants_mod.variants(task))]
        raise KeepGateError(f"perf.campaign.accuracy.unit: {unit!r} is not a unit of {known}")
    return dict(task)


def _anchor(spec: Mapping[str, Any], unit: str) -> float:
    """The contract's anchor for ``unit``: the row the contract's own accuracy gate reads."""
    from ..run.preflight import resolve_model_key

    accuracy = dict(spec.get("accuracy") or {})
    if not resolve_per_unit(accuracy, "reference_yaml", unit):
        raise KeepGateError(f"the contract names no accuracy.reference_yaml for {unit}")
    try:
        return gates_mod.resolve_reference(spec, model_key=resolve_model_key(spec, unit), unit=unit)
    except (KeyError, OSError, ValueError, yaml.YAMLError) as exc:
        raise KeepGateError(f"the contract's anchor for {unit} does not resolve: {exc}") from exc


def resolve_keep_gate(
    task: Mapping[str, Any],
    campaign: Mapping[str, Any],
    *,
    workspace: Path,
    tests: Sequence[str] = (),
) -> KeepGate:
    """The keep gate the contract defines for this campaign.

    Raises :class:`KeepGateError` when ``perf.campaign.accuracy`` is not valid, names a
    unit or member the contract does not have, or the contract gives the unit no anchor
    -- unless ``perf.campaign.accuracy: {gate: off}`` declares that keeps go ungated.
    """
    block = dict((campaign or {}).get("accuracy") or {})
    errors = validate_campaign_accuracy(block)
    if errors:
        raise KeepGateError("; ".join(errors))
    environment = resolve_environment(task, devices=(), world_size=1)
    # The gate's options file selects the target (`modeling_v2: <mode>`) from the run's
    # switch; a run that leaves it unset would have the keep gate measure the stock model.
    environment = replace(
        environment,
        env={gates_mod.MODELING_V2_ENV: gates_mod.MODE_REQUIRE, **environment.env},
    )
    if keep_gate_off(block):
        return KeepGate(spec=dict(task), environment=environment, tests=tuple(tests))
    unit = str(block.get("unit") or DEFAULT_UNIT)
    try:
        spec = _unit_spec(task, unit, workspace)
        allowed = members_for(spec, unit, widest_world_size(spec, 1))
        for member in allowed:
            parse_parallel_segment(member)
    except TaskSchemaError as exc:
        raise KeepGateError(f"the contract's members of {unit} do not resolve: {exc}") from exc
    named = [str(m) for m in block.get("members") or ()]
    unknown = [m for m in named if m not in allowed]
    if unknown:
        raise KeepGateError(
            f"perf.campaign.accuracy.members: {unknown} are not members of {unit} ({allowed})"
        )
    for key in ("checkpoint_path", "parallel"):
        if not spec.get(key):
            raise KeepGateError(f"the contract has no {key}, which its accuracy gate runs at")
    reference = _anchor(spec, unit)
    accuracy = dict(spec.get("accuracy") or {})
    benchmark = str(resolve_per_unit(accuracy, "benchmark", unit) or "gsm8k")
    floor, _ = gates_mod.resolve_floor(accuracy, unit, benchmark, reference)
    return KeepGate(
        spec=spec,
        environment=environment,
        unit=unit,
        members=tuple(named or default_members(allowed)),
        reference=reference,
        benchmark=benchmark,
        tests=tuple(tests),
        planned_floor={"value": floor.value, "rule": floor.describe()} if floor else {},
    )


__all__ = [
    "DEFAULT_UNIT",
    "ERROR",
    "FAIL",
    "KEEP_FILE",
    "PART_FILE",
    "PASS",
    "TESTS_PART",
    "GateJob",
    "KeepGate",
    "KeepGateError",
    "default_members",
    "resolve_keep_gate",
]
