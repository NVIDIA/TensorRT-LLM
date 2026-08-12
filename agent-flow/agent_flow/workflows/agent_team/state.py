"""Checkpoint state for the agent-team workflow.

Persisted to ``<workspace>/.agent_team_state.json`` so a run interrupted by
Ctrl-C / crash / reboot can be continued by re-running the workflow
against the same workspace (wipe with ``--clean`` to start over).
Per-task isolation is handled by the workflow placing each task in its
own workspace subdirectory.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

STATE_FILENAME = ".agent_team_state.json"
SCHEMA_VERSION = 5

# Stages in the workflow.
#
# Plan phase (cycles indefinitely until the human approves; resume index
# tracked by ``plan_next_iteration_index``):
#   - ``plan_drafter``  — PlanDrafter writes/refines plan.md.
#   - ``plan_reviewer`` — PlanReviewer (AI) emits APPROVE/REJECT.
#   - ``plan_human``    — PlanDrafter asks the human for approval via
#                         ``ask_human`` and polishes plan.md based on the
#                         reply until the human approves.
#
# Build phase (cycles, governed by ``next_iteration_index`` and
# ``num_iterations``):
#   - ``coder``    — Coder implements / refines.
#   - ``reviewer`` — Reviewer (AI) emits APPROVE/REJECT after running tests.
#   - ``qa``       — QA emits APPROVE/REJECT after independent validation
#                    (only runs when the reviewer APPROVEs).
#
# Replan sub-cycle (only used when the workflow is run with
# ``replan_on_qa=True``; otherwise these stages never appear in a
# checkpoint). After every QA turn, the PlanDrafter is re-invoked in
# ``replan`` mode to revise ``plan.md`` and ``acceptance-criteria.md``
# based on the latest coder/reviewer/qa findings, and decides whether
# the workflow is ``DONE``, needs a quick polish (back to coder), or a
# major rewrite that needs another PlanReviewer / human check:
#   - ``replan``          — PlanDrafter runs in replan mode.
#   - ``replan_reviewer`` — PlanReviewer re-reviews the revised plan
#                           (only entered when the PlanDrafter said the
#                           replan is a major one).
#   - ``replan_human``    — Human signs off on a major replan via
#                           ``ask_human`` (only entered when
#                           ``plan_human_review_enabled`` is set).
#
# ``stage`` always names the agent currently in progress (or pending) for
# the active phase iteration; on resume the workflow jumps directly to
# this stage instead of rerunning earlier stages.
STAGE_PLAN_DRAFTER = "plan_drafter"
STAGE_PLAN_REVIEWER = "plan_reviewer"
STAGE_PLAN_HUMAN = "plan_human"
STAGE_CODER = "coder"
STAGE_REVIEWER = "reviewer"
STAGE_QA = "qa"
STAGE_REPLAN = "replan"
STAGE_REPLAN_REVIEWER = "replan_reviewer"
STAGE_REPLAN_HUMAN = "replan_human"
_VALID_STAGES = (
    STAGE_PLAN_DRAFTER,
    STAGE_PLAN_REVIEWER,
    STAGE_PLAN_HUMAN,
    STAGE_CODER,
    STAGE_REVIEWER,
    STAGE_QA,
    STAGE_REPLAN,
    STAGE_REPLAN_REVIEWER,
    STAGE_REPLAN_HUMAN,
)
# v4 predates the replan sub-cycle: same fields, same semantics, minus
# the three replan stages. A v4 checkpoint therefore loads unchanged
# under v5 (the next ``save_state`` re-stamps it as v5); it just must
# not claim a replan stage it could never have reached.
_V4_STAGES = (
    STAGE_PLAN_DRAFTER,
    STAGE_PLAN_REVIEWER,
    STAGE_PLAN_HUMAN,
    STAGE_CODER,
    STAGE_REVIEWER,
    STAGE_QA,
)


@dataclass
class WorkflowState:
    task_path: str
    num_iterations: int
    next_iteration_index: int = 0
    plan_next_iteration_index: int = 0
    done: bool = False
    stage: str = STAGE_PLAN_DRAFTER


def load_state(path: Path) -> WorkflowState:
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version")
    if version not in (SCHEMA_VERSION, 4):
        raise ValueError(
            f"Unsupported checkpoint version {version!r} in {path}; "
            f"expected {SCHEMA_VERSION}. Delete the file to start fresh."
        )
    valid_stages = _VALID_STAGES if version == SCHEMA_VERSION else _V4_STAGES
    stage = data.get("stage")
    if stage not in valid_stages:
        raise ValueError(
            f"Unsupported stage {stage!r} in {path}; expected one of "
            f"{valid_stages}. Delete the file to start fresh."
        )
    return WorkflowState(
        task_path=str(data["task_path"]),
        num_iterations=int(data.get("num_iterations", 0)),
        next_iteration_index=int(data.get("next_iteration_index", 0)),
        plan_next_iteration_index=int(data.get("plan_next_iteration_index", 0)),
        done=bool(data.get("done", False)),
        stage=stage,
    )


def save_state(path: Path, state: WorkflowState) -> None:
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
