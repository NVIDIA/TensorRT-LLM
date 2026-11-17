# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check verdicts, measured-tree provenance and journal rendering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..infra.compute.commands import CommandResult
from ..infra.git import git_output
from .runtime_evidence import describe_engine

#: What a gate records when it could not ask git anything — no repository, no
#: ``git`` on the host, no commits yet. Never an exception: a gate adjudicates
#: a model, not a toolchain.
UNKNOWN_TREE_STATE = "unknown"


def describe_tree_state(repo_root: str | Path) -> str:
    """The commit a gate is about to measure, as one token: ``git describe --always``.

    Not ``--dirty``, which refreshes the whole worktree (a minute on a large
    networked checkout).
    """
    described = git_output(repo_root, "describe", "--always")
    if described is None:
        return UNKNOWN_TREE_STATE
    return described.strip() or UNKNOWN_TREE_STATE


def inspect_tree(task_data: Mapping[str, Any]) -> str:
    """The tree a gate is about to measure: :func:`describe_tree_state` of ``trtllm_repo_path``.

    A contract check measures a read-only snapshot of one commit
    (``run/workspace/snapshots.py``), which stamps its own description over this one.
    """
    repo_root = task_data.get("trtllm_repo_path")
    return describe_tree_state(repo_root) if repo_root else UNKNOWN_TREE_STATE


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict, with everything needed to defend it later."""

    name: str
    passed: bool
    detail: str
    #: The parallel configuration this verdict was measured at; ``None`` when
    #: the caller passes no member.
    member: str | None = None
    command: str = ""
    measured: float | None = None
    reference: float | None = None
    tolerance: float | None = None
    duration_s: float = 0.0
    stdout_tail: str = ""
    #: The repository state this verdict was measured on —
    #: ``git describe --always`` under ``trtllm_repo_path``, or
    #: :data:`UNKNOWN_TREE_STATE` where git could not answer.
    tree_state: str = UNKNOWN_TREE_STATE
    #: The target's marker for the commit this verdict measured, when the check
    #: ran on a snapshot of one (``run/workspace/snapshots.py``): the contract records the
    #: verdict under it, whatever the checkout became while the check ran.
    marker: str = ""
    #: What each command of the gate imported the package under test from, in
    #: order, as its processes reported it: the commit of the snapshot it ran
    #: in, or the path of a checkout no commit pins. Empty where no command
    #: said, and for a gate that ran on no snapshot.
    imported: tuple[str, ...] = ()
    #: Why this verdict says nothing about one tree, when it does not: its
    #: commands imported more than one tree, or a tree no commit pins, or no
    #: snapshot could be made. A void verdict never passes.
    void: str = ""
    #: The tree's Python files the gate's processes imported, relative to it;
    #: empty where a command did not say (``run/workspace/snapshots.py:CheckTree.read``).
    reads: tuple[str, ...] = ()
    #: The per-prompt numbers behind ``measured``, where a gate has them.
    breakdown: str = ""
    #: The accuracy floor the measurement was held to, and the binomial
    #: standard error and sample count it was derived from.
    floor: float | None = None
    se: float | None = None
    samples: int | None = None
    #: How ``floor`` was derived, verbatim (see :meth:`.accuracy_policy.Floor.describe`).
    floor_rule: str = ""
    #: Things the verdict wants a reader to see that do not change it -- a
    #: floor not backed by a measurement, a comparison that had no baseline.
    warnings: tuple[str, ...] = ()
    #: Whether the summary line carries ``detail`` even on a pass: true for a
    #: verdict that rests on more than its headline numbers.
    explained: bool = False
    #: Set when the gate failed because the session under it was lost, not
    #: because of anything it measured; says why. Such a result is not a
    #: verdict about the target: the workflow re-runs the gate once a session
    #: is back instead of recording it or handing it to the assembler.
    infrastructure: str = ""
    #: One record per engine the gate built (:meth:`EngineAttestation.record`):
    #: the configuration it ran against the declared one, the pools it sized
    #: and the device memory it left free. Empty for a gate that built none.
    #: Not hashed: the records are mappings.
    engine: tuple[Mapping[str, Any], ...] = field(default=(), hash=False)

    def summary(self) -> str:
        """One line for the journal, carrying the numbers verbatim."""
        verdict = "VOID" if self.void else "PASS" if self.passed else "FAIL"
        label = f"{self.name}@{self.member}" if self.member else self.name
        tree = f" [tree {self.tree_state}]"
        breakdown = f" [{self.breakdown}]" if self.breakdown else ""
        breakdown += describe_engine(self.engine)
        if self.warnings:
            breakdown += "".join(f" [WARNING: {w}]" for w in self.warnings)
        why = f" -- {self.detail}" if self.detail and (self.explained or not self.passed) else ""
        if self.measured is not None and self.floor is not None:
            return (
                f"[{verdict}] {label}: measured {self.measured:.4f} vs floor "
                f"{self.floor:.4f} [{self.floor_rule}] (margin "
                f"{self.measured - self.floor:+.4f}){why}{breakdown} in "
                f"{self.duration_s:.0f}s{tree}"
            )
        if self.measured is not None and self.reference is not None:
            margin = self.measured - (self.reference - (self.tolerance or 0.0))
            return (
                f"[{verdict}] {label}: measured {self.measured:.4f} vs "
                f"reference {self.reference:.4f} - tol {self.tolerance or 0.0:.2f} "
                f"(margin {margin:+.4f}){why}{breakdown} in {self.duration_s:.0f}s{tree}"
            )
        return f"[{verdict}] {label}: {self.detail}{breakdown} ({self.duration_s:.0f}s){tree}"


def verdict_factory(tree_state: str) -> Callable[..., GateResult]:
    """A :class:`GateResult` constructor that stamps ``tree_state`` onto every verdict."""
    return partial(GateResult, tree_state=tree_state)


def lost_session(result: CommandResult) -> dict[str, Any]:
    """The :class:`GateResult` fields of a gate whose session was lost under it."""
    return {
        "passed": False,
        "detail": f"did not run to a verdict: {result.infrastructure}",
        "infrastructure": result.infrastructure,
    }


def measured_on(
    result: GateResult,
    *,
    tree_state: str,
    marker: str = "",
    imported: Sequence[str] = (),
    void: str = "",
    warnings: Sequence[str] = (),
    reads: Sequence[str] = (),
) -> GateResult:
    """``result`` as measured on one pinned tree: what it is, its marker, and what was imported.

    A void result (``void`` says why) does not pass, whatever it measured, and
    its detail leads with why.
    """
    fields: dict[str, Any] = {
        "tree_state": tree_state,
        "marker": marker,
        "imported": tuple(imported),
        "warnings": (*result.warnings, *warnings),
        "reads": tuple(reads),
    }
    if void:
        detail = f"{void}; what it measured: {result.detail}" if result.detail else void
        fields.update(passed=False, void=void, explained=True, detail=detail)
    return replace(result, **fields)


def tree_commit(repo_root: str | Path | None) -> str | None:
    """The full commit ``HEAD`` names in ``repo_root``, or ``None`` if git cannot say."""
    if not repo_root:
        return None
    answer = git_output(repo_root, "rev-parse", "HEAD")
    return answer.strip() if answer and answer.strip() else None


@dataclass
class GateLog:
    """Every gate a run has fired, in order. The journal's raw material."""

    results: list[GateResult] = field(default_factory=list)

    def record(self, result: GateResult) -> GateResult:
        self.results.append(result)
        return result

    def render(self) -> str:
        return "\n".join(result.summary() for result in self.results)


__all__ = [
    "GateLog",
    "GateResult",
    "UNKNOWN_TREE_STATE",
    "describe_tree_state",
    "inspect_tree",
    "lost_session",
    "measured_on",
    "tree_commit",
    "verdict_factory",
]
