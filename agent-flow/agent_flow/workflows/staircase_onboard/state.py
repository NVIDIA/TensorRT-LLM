# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint state for the staircase-onboard workflow.

Persisted to ``<workspace>/.staircase_onboard_state.json`` so a run
interrupted by Ctrl-C / crash / reboot can be continued by re-running the
workflow against the same workspace (wipe with ``--clean`` to start over).

Staircase's own doctrine is that there is no resume *mode* — re-running the
same invocation is how a run resumes, because every step decides for itself
whether its work is already done. That still holds for the work products
(a catalog entry that exists is not rebuilt; a gate that passed is not
re-run). The checkpoint here is a different thing: it records **where the
orchestrator was**, so a resumed run does not re-pay the agent turns that
already produced those products. The two are complementary — the checkpoint
saves tokens, the idempotent steps save correctness.

Stages, in order:

- ``preflight``  — pure Python. Checkpoint parses, arch matches the host,
                   parallel segment is well formed, the device set is
                   claimed, TMPDIR is sane, multi-rank probes pass. No
                   agent runs here.
- ``plan``       — PlanDrafter <-> PlanReviewer, capped at
                   ``max_plan_rounds``. Products: ``plan.md`` and
                   ``vocabulary.yaml`` (the typed dispatch queue).
- ``smith``      — one smith per queue item, dispatched concurrently
                   within a dependency batch. Products: catalog entries
                   and their GPU tests, committed per entry.
- ``assemble``   — one assembler per unit, serially (core first, then one
                   increment per feature), each followed by the gates.
- ``deliver``    — Python verification across the in-tree trees, one
                   stateless auditor turn, one lint pass, then the merge.

``stage`` always names the phase currently in progress (or pending); on
resume the workflow jumps directly to it instead of rerunning earlier
phases.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATE_FILENAME = ".staircase_onboard_state.json"
SCHEMA_VERSION = 1

STAGE_PREFLIGHT = "preflight"
STAGE_PLAN = "plan"
STAGE_SMITH = "smith"
STAGE_ASSEMBLE = "assemble"
STAGE_DELIVER = "deliver"
# The performance stage: after delivery when `perf.enabled`, or the entry stage
# of `staircase-onboard tune` on an already-onboarded target. Perf is measured,
# never a release gate, so it never sits between assemble and deliver.
STAGE_TUNE = "tune"

_VALID_STAGES = (
    STAGE_PREFLIGHT,
    STAGE_PLAN,
    STAGE_SMITH,
    STAGE_ASSEMBLE,
    STAGE_DELIVER,
    STAGE_TUNE,
)

# Queue-item lifecycle. ``running`` is persisted rather than kept in
# memory so a crash mid-batch is distinguishable from a crash before it:
# a resumed run re-dispatches ``running`` items (the agent's work may have
# been lost) but leaves ``done`` ones alone.
#
# ``deferred`` is a failure only a later unit pays for: the row's entry is
# needed by a feature increment and by nothing core uses, so the smith phase
# records it and carries on instead of closing the run. The row also carries
# ``deferred_to`` (the unit that retries it before its first assembler turn)
# and ``deferred_why``. Persisted, not recomputed, because the retry is owed
# by a later process as often as by this one, and a resume that forgot a
# deferral would assemble that unit without the entry.
ITEM_PENDING = "pending"
ITEM_RUNNING = "running"
ITEM_DONE = "done"
ITEM_FAILED = "failed"
ITEM_DEFERRED = "deferred"
_VALID_ITEM_STATUSES = (ITEM_PENDING, ITEM_RUNNING, ITEM_DONE, ITEM_FAILED, ITEM_DEFERRED)


#: Verdicts a re-run should re-enter rather than refuse.
#:
#: The distinction is not "how bad was it" but **who repairs it and where**.
#: Each of these closes a run whose agent work is finished and whose failure
#: names a concrete edit the operator makes in the repo or in ``task.yaml`` --
#: a missing CI test file, an entry needing certification, a witness block
#: that was never configured. The run cannot observe that repair except by
#: running its own check again, and every one of those checks is cheap and
#: side-effect-free.
#:
#: Everything else stays refused, and deliberately. ``ceiling``,
#: ``plan-failed``, ``smith-blocked``, ``assembly-exhausted`` and
#: ``assembler-blocked`` are budgets spent or judgments made by agents; a
#: re-run re-dispatches paid turns with nothing having changed, which is how a
#: workflow burns a night discovering the same thing four times. Those need
#: ``--clean`` and a human deciding what to do differently. The one narrow
#: exception is a smith-phase ``smith-blocked`` whose every failed row is now
#: tagged for a later unit: re-entering it dispatches none of them, so it is
#: not a re-run of the same thing (``workflow._smith_block_is_deferrable``).
#:
#: Prefix match, because ``witness-unconfigured`` carries its unit.
REMEDIABLE_VERDICTS: tuple[str, ...] = (
    "delivery-failed",
    "audit-uncovered",
    "witness-unconfigured",
    # The three ways the bounded remediation cycle gives up. All of them
    # close a run whose remaining work is a human judgement made in the repo
    # -- a receipt that did not land, a coverage evaluator disagreeing with a
    # contract, a budget that was set too low for this target. The queue is
    # in the checkpoint, so re-running resumes the cycle rather than
    # restarting it, which is the whole reason it is persisted.
    "audit-no-progress",
    "audit-remediation-budget",
    "audit-verification-failed",
    # An assembler that stops on `knowledge_gap` or `environment` names work
    # only the operator can supply (a verify round, a ruling, an
    # infrastructure repair); once it lands, re-entering resumes the unit
    # instead of `--clean` re-measuring every gate. Measured on
    # DeepSeek-V4.1: the core assembler's envelope sweep stopped the run on
    # four catalog verifies. `vocabulary_gap` stays out: the run onboards
    # those itself, so re-entering on one would repeat the same turn.
    "assembler-blocked:knowledge_gap",
    "assembler-blocked:environment",
)


def is_remediable(verdict: str) -> bool:
    """Whether a closed run's verdict describes a repair made outside the run."""
    return any(verdict == known or verdict.startswith(known + ":") for known in REMEDIABLE_VERDICTS)


@dataclass
class WorkflowState:
    """One run's resume point.

    Everything here is orchestrator bookkeeping. The run's *evidence* —
    what each agent did and what each gate measured — lives in
    ``journal.md`` and ``progress.yaml`` beside it, because those are
    append-only and this is not.
    """

    task_path: str
    stage: str = STAGE_PREFLIGHT
    done: bool = False

    # Why the run closed, as ``_close_run`` recorded it. Persisted for one
    # reason: some verdicts name a repair the operator makes *outside* the
    # run, and re-running is the only way to confirm it. A closed run that
    # cannot say why it closed has to be started over from scratch, which
    # throws away every gate it had already passed. See ``REMEDIABLE_VERDICTS``.
    verdict: str = ""

    # Identity, resolved by preflight and stable for the run's life.
    slug: str = ""
    branch: str = ""
    devices: list[int] = field(default_factory=list)

    # Plan phase. ``plan_round`` counts *completed* drafter->reviewer
    # rounds, so it is also the number of re-drafts already spent.
    plan_round: int = 0
    plan_approved: bool = False

    # Smith phase. One entry per dispatch-queue item, carrying the queue
    # row plus its ``status``. Stored in full rather than as a pointer
    # into vocabulary.yaml so a queue item's outcome survives an edit to
    # that file. A ``deferred`` row is still owed by the assembly phase:
    # see ``ITEM_DEFERRED``.
    queue: list[dict[str, Any]] = field(default_factory=list)

    # Assembly phase. ``unit_index`` 0 is the core assembly; 1..N are the
    # feature increments in the order task.yaml lists them.
    #
    # ``unit_turns`` counts assembler **dispatches**, not entries into the
    # loop: a restart that finds the unit's work already on disk hands the
    # turn back rather than spending it. See
    # ``workflow.StaircaseOnboardWorkflow._assemble_one_unit``.
    unit_index: int = 0
    unit_turns: int = 0
    units_done: list[str] = field(default_factory=list)

    # Per unit, the identity of the work its last ``UNIT_READY`` was written
    # about -- see ``workflow.StaircaseOnboardWorkflow._target_marker``.
    # Persisted because it only answers a question a *different process*
    # asks: "is what is on disk still what the previous process declared
    # ready?" A resume that cannot answer it re-dispatches, which is the
    # safe direction.
    unit_marker: dict[str, str] = field(default_factory=dict)

    # Delivery phase.
    audit_clean: bool = False

    # The UNCOVERED remediation cycle: the verify queue, which round it is
    # on, and the signature of the still-open set. Persisted because the
    # convergence rules are *across* rounds -- "did this shrink?" cannot be
    # answered by a process that forgot the previous answer, and a resume
    # that forgets it re-dispatches work it already did. See
    # ``remediation.RemediationState``.
    remediation: dict[str, Any] = field(default_factory=dict)
    # The tune stage's checkpoint: entry, baseline/reference labels, ladder and
    # campaign progress (see tune/stage.py). Empty until the stage runs.
    tune: dict[str, Any] = field(default_factory=dict)

    # The metric staircase exists to drive down: how many times the run
    # stopped for a human decision. Carried in the checkpoint so it
    # survives a resume and can be reported honestly at the end.
    human_stops: int = 0


def _coerce_queue(raw: Any, path: Path) -> list[dict[str, Any]]:
    """Validate the persisted queue, rejecting anything the workflow cannot walk.

    A malformed queue is worth failing on rather than repairing: the
    orchestrator dispatches from it, so a silently-dropped row is a
    catalog entry that never gets built and a target that fails its
    vocabulary audit much later, with nothing pointing back here.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: `queue` must be a list, got {type(raw).__name__}")
    queue: list[dict[str, Any]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: queue[{index}] must be a mapping, got {type(row).__name__}")
        status = row.get("status", ITEM_PENDING)
        if status not in _VALID_ITEM_STATUSES:
            raise ValueError(
                f"{path}: queue[{index}] has status {status!r}; expected one of "
                f"{_VALID_ITEM_STATUSES}. Delete the file to start fresh."
            )
        queue.append(dict(row))
    return queue


def load_state(path: Path) -> WorkflowState:
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {version!r} in {path}; "
            f"expected {SCHEMA_VERSION}. Delete the file to start fresh."
        )
    stage = data.get("stage")
    if stage not in _VALID_STAGES:
        raise ValueError(
            f"Unsupported stage {stage!r} in {path}; expected one of "
            f"{_VALID_STAGES}. Delete the file to start fresh."
        )
    return WorkflowState(
        task_path=str(data["task_path"]),
        stage=stage,
        done=bool(data.get("done", False)),
        verdict=str(data.get("verdict", "")),
        slug=str(data.get("slug", "")),
        branch=str(data.get("branch", "")),
        devices=[int(d) for d in data.get("devices", [])],
        plan_round=int(data.get("plan_round", 0)),
        plan_approved=bool(data.get("plan_approved", False)),
        queue=_coerce_queue(data.get("queue"), path),
        unit_index=int(data.get("unit_index", 0)),
        unit_turns=int(data.get("unit_turns", 0)),
        # Absent in checkpoints written before the resume check existed. An
        # empty map reads as "no unit's work has a recorded identity", which
        # makes every such resume re-dispatch exactly as it used to.
        unit_marker={str(k): str(v) for k, v in (data.get("unit_marker") or {}).items()},
        units_done=[str(u) for u in data.get("units_done", [])],
        audit_clean=bool(data.get("audit_clean", False)),
        human_stops=int(data.get("human_stops", 0)),
        # Absent in checkpoints written before the remediation cycle existed.
        # An empty queue is the correct reading of those: no round has run.
        remediation=dict(data.get("remediation") or {}),
        # Absent before the tune stage existed: no campaign has run.
        tune=dict(data.get("tune") or {}),
    )


def save_state(path: Path, state: WorkflowState) -> None:
    """Write the checkpoint atomically.

    Same fsync + ``os.replace`` dance as the other workflows: the
    orchestrator checkpoints between every stage, so a torn write here
    would strand a run that was otherwise fine.
    """
    payload = {"version": SCHEMA_VERSION, **state.__dict__}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=STATE_FILENAME + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "REMEDIABLE_VERDICTS",
    "is_remediable",
    "ITEM_DEFERRED",
    "ITEM_DONE",
    "ITEM_FAILED",
    "ITEM_PENDING",
    "ITEM_RUNNING",
    "SCHEMA_VERSION",
    "STAGE_ASSEMBLE",
    "STAGE_DELIVER",
    "STAGE_PLAN",
    "STAGE_PREFLIGHT",
    "STAGE_SMITH",
    "STATE_FILENAME",
    "WorkflowState",
    "load_state",
    "save_state",
]
