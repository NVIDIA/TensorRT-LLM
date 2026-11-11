# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check verdicts, measured-tree provenance and journal rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping

from ..infra.compute.commands import CommandResult
from ..infra.git import git_output
from .attest import describe as describe_engine

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
    """The tree a gate is about to measure: :func:`describe_tree_state` of ``trtllm_repo_path``."""
    repo_root = task_data.get("trtllm_repo_path")
    return describe_tree_state(repo_root) if repo_root else UNKNOWN_TREE_STATE


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict, with everything needed to defend it later."""

    name: str
    passed: bool
    detail: str
    #: The deployment this verdict was measured on; ``None`` for a check of the tree alone.
    deployment: str | None = None
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
    #: The per-prompt numbers behind ``measured``, where a gate has them.
    breakdown: str = ""
    #: The accuracy floor the measurement was held to, and the binomial
    #: standard error and sample count it was derived from.
    floor: float | None = None
    se: float | None = None
    samples: int | None = None
    #: How ``floor`` was derived, verbatim (see :meth:`.anchors.Floor.describe`).
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
    #: A score held to nothing (no anchor, no sample count, a relative anchor whose
    #: deployment has no score): a report line, neither a pass nor a failure. ``passed``
    #: is true with it, so nothing takes the score for a failed check.
    unjudged: bool = False
    #: Which path judged: ``in process`` or ``through the endpoint``; empty for a check
    #: that has one path only.
    path: str = ""
    #: One record per server the check attested (:meth:`.attest.Attestation.record`):
    #: its options against what every rank's engine ran. Empty for a check that
    #: attested none. Not hashed: the records are mappings.
    engine: tuple[Mapping[str, Any], ...] = field(default=(), hash=False)

    def summary(self) -> str:
        """One line for the journal, carrying the numbers verbatim."""
        verdict = "NO VERDICT" if self.unjudged else "PASS" if self.passed else "FAIL"
        label = f"{self.deployment}:{self.name}" if self.deployment else self.name
        tree = f" [tree {self.tree_state}]" + (f" [{self.path}]" if self.path else "")
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
