"""Checkpoint state for the perf-optimize workflow.

Persisted to ``<workspace>/.perf_optimize_state.json`` so a run
interrupted by Ctrl-C / crash / reboot can be continued by re-running
the workflow against the same workspace (wipe with ``--clean`` to start
over). Per-task isolation is handled by the user placing each task in
its own workspace subdirectory.

Unlike perf-analyze's linear ladder, perf-optimize loops: an outer
*round* loop (analyzer → items) nests an *item* loop (up to
``max_items_per_round`` roadmap items applied one at a time per round)
which nests an *attempt* loop (optimizer ⇄ evaluator) per item, so the
state carries the loop counters, the roadmap item under optimization,
the accepts-since-the-last-analysis tally, and whether that analysis is
still safe to reuse. A rejected config attempt is hard-reverted and the
next round may re-plan from the standing profile; an accept, an older
checkpoint with unknown history, or a reverted code attempt whose
gitignored build output may survive requires a fresh profile instead.
The loop runs the full round budget unless a deterministic break fires
(an analyzer turn leaves the roadmap with no actionable pending item /
optional improvement target met); ``qa`` then runs **once** as the
campaign's final verification. ``stage`` always names the agent currently
in progress (or pending); on resume the workflow jumps directly to this
stage with the recorded round/item/attempt indices.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

STATE_FILENAME = ".perf_optimize_state.json"
# Version 2: fixed-round loop (no per-round QA gate), three-way evaluator
# decisions, one-shot final-verification QA. Version-1 checkpoints were
# written under per-round CONTINUE/APPROVE semantics and must not resume
# silently into the new loop.
#
# Later fields are additive: old v2 checkpoints load with
# ``projector_done=False``, and a missing ``profile_required`` buys one
# conservative profile. Every old stage string remains valid, so the
# version stays at 2.
SCHEMA_VERSION = 2

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
#                       server, no profiler) only when the standing
#                       runtime profile is known to remain current.
#   - ``optimizer``   — inner loop: applies the top pending roadmap item
#                       (up to ``max_items_per_round`` items per round,
#                       one at a time).
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
STAGE_OPTIMIZER = "optimizer"
STAGE_EVALUATOR = "evaluator"
STAGE_QA = "qa"
STAGE_REPORTER = "reporter"
_VALID_STAGES = (
    STAGE_BENCHMARKER,
    STAGE_PROJECTOR,
    STAGE_ANALYZER,
    STAGE_OPTIMIZER,
    STAGE_EVALUATOR,
    STAGE_QA,
    STAGE_REPORTER,
)

# The stages that make up one optimization round, in order. ``qa`` is
# not one of them: it runs once, after the round loop concludes.
ROUND_STAGES = (
    STAGE_ANALYZER,
    STAGE_OPTIMIZER,
    STAGE_EVALUATOR,
)

# Stages that only make sense while a roadmap item is being worked on —
# they require ``current_item_id`` to be set.
_ITEM_STAGES = (STAGE_OPTIMIZER, STAGE_EVALUATOR)


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
    # Loop position. ``round_index`` / ``item_index`` / ``attempt_index``
    # are 0-based; round N writes under ``rounds/round_<N+1>/``, item J
    # under ``item_<J+1>_<id>/``, and attempt K under ``attempt_<K+1>/``.
    # ``item_index`` counts items that reached a terminal status
    # (accepted/failed) this round; while an item is in the optimizer ⇄
    # evaluator loop it names that item's 0-based position in the round.
    round_index: int = 0
    item_index: int = 0
    attempt_index: int = 0
    # Measured items accepted since the analyzer last ran. This is prompt
    # context, not the conservative re-profile gate: an older checkpoint
    # can have unknown history, and a rejected code attempt can leave
    # gitignored build output behind without incrementing this count.
    accepts_since_analysis: int = 0
    # The item most recently included in that count. The accept path
    # checkpoints this id before committing, so resuming the same
    # evaluator stage after a crash does not count one approval twice.
    # Cleared with the tally after an analyzer turn.
    last_counted_accept_id: str = ""
    # Whether the next analyzer turn must profile rather than re-plan from
    # ``last_profiled_analysis_dir``. True initially, after every accept,
    # and after any reverted code attempt that may have rebuilt a
    # gitignored binary/cache. ``load_state`` also defaults this to True
    # for checkpoints written before the field existed: one conservative
    # profile is safer than asserting an unknown runtime is unchanged.
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
    # The dedicated optimization branch in ``trtllm_repo_path`` and the
    # HEAD it was created from (the reporter diffs ``base..HEAD``).
    git_branch: str = ""
    git_base_commit: str = ""
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
    current_item_id = str(data.get("current_item_id", "") or "")
    if stage in _ITEM_STAGES and not current_item_id:
        raise ValueError(
            f"Checkpoint {path} is at stage {stage!r} but records no "
            f"current_item_id — the checkpoint is inconsistent. Delete the "
            f"file to start fresh."
        )
    return WorkflowState(
        task_path=str(data["task_path"]),
        max_rounds=int(data.get("max_rounds", 3)),
        max_attempts_per_item=int(data.get("max_attempts_per_item", 3)),
        max_items_per_round=int(data.get("max_items_per_round", 1)),
        round_index=int(data.get("round_index", 0)),
        item_index=int(data.get("item_index", 0)),
        attempt_index=int(data.get("attempt_index", 0)),
        # Keep the accepted-item count honest. Checkpoints written before
        # the field existed have unknown history, but that uncertainty is
        # represented by ``profile_required`` rather than by inventing one
        # accepted item and interpolating it into user/agent-facing text.
        accepts_since_analysis=int(data.get("accepts_since_analysis", 0)),
        last_counted_accept_id=str(data.get("last_counted_accept_id", "") or ""),
        # Additive v2 migration: old checkpoints carry no explicit proof
        # that their standing profile is current (and may predate the
        # rejected-code/build-artifact guard), so buy one conservative
        # profile. Once saved again this becomes an ordinary boolean.
        profile_required=(
            data["profile_required"] if isinstance(data.get("profile_required"), bool) else True
        ),
        last_profiled_analysis_dir=str(data.get("last_profiled_analysis_dir", "") or ""),
        current_item_id=current_item_id,
        approach_violation=str(data.get("approach_violation", "") or ""),
        last_nsys_dir=str(data.get("last_nsys_dir", "") or ""),
        reuse_analysis_dir=str(data.get("reuse_analysis_dir", "") or ""),
        reuse_pending=bool(data.get("reuse_pending", False)),
        git_branch=str(data.get("git_branch", "") or ""),
        git_base_commit=str(data.get("git_base_commit", "") or ""),
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
