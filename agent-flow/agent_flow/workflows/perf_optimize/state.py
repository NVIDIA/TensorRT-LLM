"""Checkpoint state for the perf-optimize workflow.

Persisted to ``<workspace>/.perf_optimize_state.json`` so a run
interrupted by Ctrl-C / crash / reboot can be continued by re-running
the workflow against the same workspace (wipe with ``--clean`` to start
over). Per-task isolation is handled by the user placing each task in
its own workspace subdirectory.

Unlike perf-analyze's linear ladder, perf-optimize loops: an outer
*round* loop selects up to ``max_items_per_round`` roadmap items. Their
nested attempt loops (optimizer ⇄ evaluator) run serially with direct
acceptance or in parallel before an Integrator combines them, so the
state carries the checkpointed batch in addition to the loop counters and
records whether the standing runtime profile is still current. The loop
runs the full round budget unless a deterministic break fires (an analyzer
turn leaves the roadmap with no actionable pending item / optional
improvement target met); ``qa`` then runs **once** as the campaign's final
verification. ``stage`` always names the agent currently in progress (or
pending); on resume the workflow jumps directly to this stage with the
recorded round/item/attempt indices.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATE_FILENAME = ".perf_optimize_state.json"
# Version 2: fixed-round loop (no per-round QA gate), three-way evaluator
# decisions, one-shot final-verification QA. Version-1 checkpoints were
# written under per-round CONTINUE/APPROVE semantics and must not resume
# silently into the new loop.
#
# Adding the projector stage/field was purely additive (old v2
# checkpoints load with ``projector_done=False`` and every old stage
# string is still valid), so the version stays at 2.
# Version 3 replaces the shared scalar optimizer/evaluator cursor with a
# checkpointed batch.  V2 checkpoints cannot be resumed safely because a
# single checkout and a single current_item_id do not identify the work done
# by concurrent item workers.
SCHEMA_VERSION = 3
ITEM_EXECUTIONS = ("serial", "parallel")

# Stages in the workflow:
#
#   - ``benchmarker`` — one-shot: launches ``trtllm-serve``, measures the
#                       baseline operating point, writes
#                       ``baseline/benchmark_results.md``.
#   - ``projector``   — one-shot, conditional: runs only when task.yaml
#                       carries a ``sol`` block, between the baseline
#                       benchmark and round 1. Derives the analytical
#                       speed-of-light (SOL) ceiling per the
#                       internal-perf-sol-analysis skill and writes
#                       ``sol_projection.md``; skipped (never marked
#                       done) otherwise.
#   - ``analyzer``    — start of each round: profiles the current build and
#                       writes/updates ``roadmap.yaml`` (items ordered by
#                       expected perf benefit). Opens **replan-only** (no
#                       server, no profiler) when the standing runtime
#                       profile is known to remain current.
#   - ``optimizer_evaluator`` — batch of isolated per-item attempt loops,
#                       sequential or parallel per ``item_execution``.
#   - ``integrator``  — combines candidate-ready code/config and measures the
#                       authoritative accepted round state.
#   - ``evaluator``   — inner loop: reviews the change (code quality,
#                       functionality, measured perf) and decides
#                       APPROVE (accept), REJECT (fail the item, move
#                       on), or PUSH_BACK (retry the optimizer with
#                       feedback). On APPROVE its turn also captures the
#                       accept-evidence nsys profile of the accepted
#                       state (when nsys is configured).
#   - ``qa``          — one-shot after the round loop: the campaign's
#                       final verification — independent benchmark,
#                       sanity completions, and the optional accuracy
#                       eval. Skipped when no item was accepted.
#   - ``reporter``    — one-shot: synthesizes ``optimization_report.md`` /
#                       ``.html`` from every role's artifacts.
STAGE_BENCHMARKER = "benchmarker"
STAGE_PROJECTOR = "projector"
STAGE_ANALYZER = "analyzer"
# Public stage while the per-item optimizer/evaluator attempt loops run.
STAGE_OPTIMIZER_EVALUATOR = "optimizer_evaluator"
# Public stage that combines candidate-ready items and measures the result.
STAGE_INTEGRATOR = "integrator"
# Internal worker phases.  They are deliberately not valid global stages.
STAGE_OPTIMIZER = "optimizer"
STAGE_EVALUATOR = "evaluator"
STAGE_QA = "qa"
STAGE_REPORTER = "reporter"
_VALID_STAGES = (
    STAGE_BENCHMARKER,
    STAGE_PROJECTOR,
    STAGE_ANALYZER,
    STAGE_OPTIMIZER_EVALUATOR,
    STAGE_INTEGRATOR,
    STAGE_QA,
    STAGE_REPORTER,
)

# The stages that make up one optimization round, in order. ``qa`` is
# not one of them: it runs once, after the round loop concludes.
ROUND_STAGES = (
    STAGE_ANALYZER,
    STAGE_OPTIMIZER_EVALUATOR,
    STAGE_INTEGRATOR,
)

# Stages that only make sense while a roadmap item is being worked on —
# they require ``current_item_id`` to be set.
_BATCH_STAGES = (STAGE_OPTIMIZER_EVALUATOR, STAGE_INTEGRATOR)


@dataclass
class WorkflowState:
    task_path: str
    # Loop bounds resolved from task.yaml (+ CLI override) at init and
    # frozen into the checkpoint so a resume keeps the original budget.
    # ``max_items_per_round`` defaults to 1 here (not the task-schema
    # default) so checkpoints written before the field existed resume
    # with the behavior they were started under.
    max_rounds: int = 3
    max_attempts_per_item: int = 3
    max_items_per_round: int = 1
    # Frozen from task.yaml so a resumed batch cannot silently switch its
    # scheduling/finalization semantics. Missing v3 fields load as parallel.
    item_execution: str = "parallel"
    # Loop position. ``round_index`` / ``item_index`` / ``attempt_index``
    # are 0-based; round N writes under ``rounds/round_<N+1>/``, item J
    # under ``item_<J+1>_<id>/``, and attempt K under ``attempt_<K+1>/``.
    # ``item_index`` counts items that reached a terminal status
    # (accepted/failed) this round; while an item is in the optimizer ⇄
    # evaluator loop it names that item's 0-based position in the round.
    round_index: int = 0
    item_index: int = 0
    attempt_index: int = 0
    # Whether the next analyzer turn must profile rather than re-plan from
    # ``last_profiled_analysis_dir``. True initially and before an accepted
    # candidate or integrated batch is promoted into the campaign.
    profile_required: bool = True
    # The analysis directory holding the newest evidence of the current
    # campaign's build — the round directory of the last analyzer turn
    # that actually produced a profile. Imported evidence belongs to a
    # different run and never advances it; neither do replan-only rounds.
    # Empty until this campaign's first real profile lands.
    last_profiled_analysis_dir: str = ""
    # The roadmap item id currently in the optimizer ⇄ evaluator loop
    # ("" outside it).
    current_item_id: str = ""
    # Why the orchestrator auto-rejected the previous attempt for using a
    # disallowed ``optimize.approaches`` value ("" when the previous
    # attempt reached the evaluator normally). Feeds the retry
    # instructions, so it must survive a crash between the auto-reject
    # and the retry.
    approach_violation: str = ""
    # The most recent nsys capture of the system as currently accepted:
    # the round's ``analysis/`` directory after each analyzer profile, or
    # an accepted attempt's ``profile/`` directory after an
    # accept-evidence capture. The evaluator compares its own capture
    # against it; "" until the first capture lands (or when nsys is not
    # configured).
    last_nsys_dir: str = ""
    # The ``--reuse-analysis`` source this workspace was seeded from ("" on
    # a normal run), and whether round 1's analyzer still owes its
    # plan-only turn over the imported artifacts (cleared once it has run,
    # so a resume never re-plans from scratch).
    reuse_analysis_dir: str = ""
    reuse_pending: bool = False
    # The dedicated campaign branch in ``trtllm_repo_path`` and the
    # HEAD it was created from (the reporter diffs ``base..HEAD``).
    campaign_git_branch: str = ""
    campaign_git_base_commit: str = ""

    # Optimizer/evaluator batch item fields.  The scalar fields
    # above are populated only on a worker-local WorkflowState copy so the
    # existing prompt/path helpers can be reused.  ``item_batch`` is the
    # checkpointed global authority.
    item_worktree_path: str = ""
    item_branch: str = ""
    item_base_commit: str = ""
    item_batch: list[dict[str, Any]] = field(default_factory=list)
    batch_started: bool = False
    batch_completed: bool = False

    integration_worktree_path: str = ""
    integration_branch: str = ""
    benchmarker_done: bool = False
    projector_done: bool = False
    reporter_done: bool = False
    done: bool = False
    stage: str = STAGE_BENCHMARKER


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
    item_execution = str(data.get("item_execution", "parallel"))
    if item_execution not in ITEM_EXECUTIONS:
        raise ValueError(
            f"Unsupported item_execution {item_execution!r} in {path}; expected one of "
            f"{ITEM_EXECUTIONS}. Delete the file to start fresh."
        )
    current_item_id = str(data.get("current_item_id", "") or "")
    item_batch = data.get("item_batch", [])
    if not isinstance(item_batch, list):
        raise ValueError(f"Checkpoint {path} has a non-list item_batch")
    if stage in _BATCH_STAGES and not item_batch:
        raise ValueError(
            f"Checkpoint {path} is at stage {stage!r} but records no "
            f"item_batch — the checkpoint is inconsistent. Delete the "
            f"file to start fresh."
        )
    return WorkflowState(
        task_path=str(data["task_path"]),
        max_rounds=int(data.get("max_rounds", 3)),
        max_attempts_per_item=int(data.get("max_attempts_per_item", 3)),
        max_items_per_round=int(data.get("max_items_per_round", 1)),
        item_execution=item_execution,
        round_index=int(data.get("round_index", 0)),
        item_index=int(data.get("item_index", 0)),
        attempt_index=int(data.get("attempt_index", 0)),
        profile_required=(
            data["profile_required"] if isinstance(data.get("profile_required"), bool) else True
        ),
        last_profiled_analysis_dir=str(data.get("last_profiled_analysis_dir", "") or ""),
        current_item_id=current_item_id,
        approach_violation=str(data.get("approach_violation", "") or ""),
        last_nsys_dir=str(data.get("last_nsys_dir", "") or ""),
        reuse_analysis_dir=str(data.get("reuse_analysis_dir", "") or ""),
        reuse_pending=bool(data.get("reuse_pending", False)),
        campaign_git_branch=str(data.get("campaign_git_branch", "") or ""),
        campaign_git_base_commit=str(data.get("campaign_git_base_commit", "") or ""),
        item_worktree_path=str(data.get("item_worktree_path", "") or ""),
        item_branch=str(data.get("item_branch", "") or ""),
        item_base_commit=str(data.get("item_base_commit", "") or ""),
        item_batch=item_batch,
        batch_started=bool(data.get("batch_started", False)),
        batch_completed=bool(data.get("batch_completed", False)),
        integration_worktree_path=str(data.get("integration_worktree_path", "") or ""),
        integration_branch=str(data.get("integration_branch", "") or ""),
        benchmarker_done=bool(data.get("benchmarker_done", False)),
        projector_done=bool(data.get("projector_done", False)),
        reporter_done=bool(data.get("reporter_done", False)),
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
