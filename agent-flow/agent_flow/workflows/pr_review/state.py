"""Checkpoint state for the pr-review workflow.

Persisted to ``<workspace>/.pr_review_state.json`` so a run interrupted by
Ctrl-C / crash / reboot can be continued by re-running the workflow against
the same workspace (wipe with ``--clean`` to start over). Per-PR/MR isolation
is handled by the workflow placing each PR/MR in its own workspace
subdirectory.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

STATE_FILENAME = ".pr_review_state.json"
SCHEMA_VERSION = 1

# Stages in the workflow.
#
# The workflow runs two review stages with the reviewer/coder roles swapped
# between the two backends. Each stage cycles rounds of reviewer ⇄ coder until
# they converge (reviewer APPROVE + coder AGREE) or the coder stands firm on a
# push-back (coder wins), then advances to the next stage:
#
#   - ``s1_reviewer`` — Stage 1 reviewer (Claude Code) reviews the diff.
#   - ``s1_coder``    — Stage 1 coder (Codex) addresses the review / pushes back.
#   - ``s2_reviewer`` — Stage 2 reviewer (Codex) reviews the diff.
#   - ``s2_coder``    — Stage 2 coder (Claude Code) addresses the review / pushes back.
#
# Each stage's round counter is tracked separately by
# ``stage1_next_round_index`` / ``stage2_next_round_index`` so resume jumps to
# the right round of the right stage. ``stage`` always names the agent
# currently in progress (or pending); on resume the workflow jumps directly to
# this stage instead of rerunning earlier ones.
STAGE_S1_REVIEWER = "s1_reviewer"
STAGE_S1_CODER = "s1_coder"
STAGE_S2_REVIEWER = "s2_reviewer"
STAGE_S2_CODER = "s2_coder"
_VALID_STAGES = (
    STAGE_S1_REVIEWER,
    STAGE_S1_CODER,
    STAGE_S2_REVIEWER,
    STAGE_S2_CODER,
)

# The two stages, in order, paired with the (reviewer, coder) sub-stages that
# make up each one. Used by the orchestrator to drive the stage machine and to
# map a stage back to its progress.yaml list.
STAGE1 = "stage1"
STAGE2 = "stage2"


@dataclass
class WorkflowState:
    pr_context_path: str
    num_rounds: int
    next_round_index: int = 0
    stage: str = STAGE_S1_REVIEWER
    done: bool = False


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
        pr_context_path=str(data["pr_context_path"]),
        num_rounds=int(data.get("num_rounds", 0)),
        next_round_index=int(data.get("next_round_index", 0)),
        stage=stage,
        done=bool(data.get("done", False)),
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
