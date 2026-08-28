"""Checkpoint state for the perf-analyze workflow.

Persisted to ``<workspace>/.perf_analyze_state.json`` so a run
interrupted by Ctrl-C / crash / reboot can be continued by re-running
the workflow against the same workspace (wipe with ``--clean`` to start
over). Per-task isolation is handled by the user placing each task in
its own workspace subdirectory.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

STATE_FILENAME = ".perf_analyze_state.json"
# Adding the projector stage/field was purely additive (old checkpoints
# load with ``projector_done=False`` and every old stage string is still
# valid), so the version stays at 1 — bumping it would hard-fail every
# in-flight workspace for no safety gain. The profiler→analyzer rename is
# handled the same way: ``load_state`` maps the legacy ``"profiler"``
# stage string and ``profiler_done`` key onto their analyzer equivalents.
SCHEMA_VERSION = 1

# Stages in the workflow — a linear, one-shot pipeline. Each stage runs
# exactly once; there is no review/iteration loop:
#
#   - ``benchmarker`` — launches ``trtllm-serve``, drives the single
#                       operating point with ``benchmark_serving.py``,
#                       writes ``benchmark_results.md`` + raw JSON.
#   - ``projector``   — derives an analytical speed-of-light (SOL)
#                       ceiling for the served model, following the
#                       ``internal-perf-sol-analysis`` skill,
#                       and writes ``sol_projection.md`` (plus the
#                       machine-readable ``sol_work/peaks.json`` the
#                       analyzer's correlation joins against). The stage
#                       is conditional: it runs only when ``task.yaml``
#                       carries a ``sol`` block, and is skipped (never
#                       marked done) otherwise.
#   - ``analyzer``    — relaunches the server under nsys and the torch
#                       profiler, replays the same load, writes
#                       ``profile_findings.md`` + trace artifacts (and,
#                       with a ``sol`` block, the measured↔SOL
#                       correlation). Named like perf-optimize's
#                       diagnosis stage — perf-optimize's analyzer is
#                       this role plus roadmap authoring.
#   - ``reporter``    — synthesizes the benchmark + projection + profile
#                       evidence into ``performance_report.md`` / ``.html``
#                       with the main-bottleneck verdict.
#
# ``stage`` always names the agent currently in progress (or pending);
# on resume the workflow jumps directly to this stage instead of
# rerunning earlier stages.
STAGE_BENCHMARKER = "benchmarker"
STAGE_PROJECTOR = "projector"
STAGE_ANALYZER = "analyzer"
STAGE_REPORTER = "reporter"
_VALID_STAGES = (
    STAGE_BENCHMARKER,
    STAGE_PROJECTOR,
    STAGE_ANALYZER,
    STAGE_REPORTER,
)
# The analyzer stage was called "profiler" before the perf-optimize
# alignment; checkpoints written by that version keep loading.
_LEGACY_ANALYZER_STAGE = "profiler"


@dataclass
class WorkflowState:
    task_path: str
    benchmarker_done: bool = False
    projector_done: bool = False
    analyzer_done: bool = False
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
    if stage == _LEGACY_ANALYZER_STAGE:
        stage = STAGE_ANALYZER
    if stage not in _VALID_STAGES:
        raise ValueError(
            f"Unsupported stage {stage!r} in {path}; expected one of "
            f"{_VALID_STAGES}. Delete the file to start fresh."
        )
    return WorkflowState(
        task_path=str(data["task_path"]),
        benchmarker_done=bool(data.get("benchmarker_done", False)),
        projector_done=bool(data.get("projector_done", False)),
        analyzer_done=bool(data.get("analyzer_done", data.get("profiler_done", False))),
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
