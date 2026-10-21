# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint state for the trtllm-modeling-v2 workflow.

Persisted to ``<workspace>/.trtllm_modeling_v2_state.json`` so a run
interrupted by Ctrl-C, a crash or a reboot continues when the same command is
run again against the same workspace (``--clean`` starts over).

The checkpoint records only **where the orchestrator was**. The run's
progress lives in files the run keeps anyway: the task graph
(``graph.yaml``), the contract's verdicts (``contract_status.jsonl``) and the
decision files.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATE_FILENAME = ".trtllm_modeling_v2_state.json"
SCHEMA_VERSION = 2

STAGE_PREFLIGHT = "preflight"
STAGE_PLANNED = "planned"
STAGE_TUNE = "tune"

_VALID_STAGES = (STAGE_PREFLIGHT, STAGE_PLANNED, STAGE_TUNE)


#: Verdicts a re-run should re-enter rather than refuse: each names a repair the
#: operator makes outside the run (the repository, ``task.yaml``, the cluster).
#: Prefix match, because ``witness-unconfigured`` and ``session-lost`` carry
#: what they were about.
REMEDIABLE_VERDICTS: tuple[str, ...] = (
    "delivery-failed",
    "witness-unconfigured",
    "session-lost",
    "escalated",
    "planner-stalled",
    "planner-failed",
)


def is_remediable(verdict: str) -> bool:
    """Whether a closed run's verdict describes a repair made outside the run."""
    return any(verdict == known or verdict.startswith(known + ":") for known in REMEDIABLE_VERDICTS)


@dataclass
class WorkflowState:
    """One run's resume point; the run's evidence lives in the files beside it."""

    task_path: str
    stage: str = STAGE_PREFLIGHT
    done: bool = False

    # Why the run closed; see ``REMEDIABLE_VERDICTS``.
    verdict: str = ""

    # Identity, resolved by preflight and stable for the run's life.
    slug: str = ""
    branch: str = ""
    devices: list[int] = field(default_factory=list)

    # The tune stage's checkpoint (see tune/stage.py). Empty until it runs.
    tune: dict[str, Any] = field(default_factory=dict)

    # How many times the run stopped for a human decision.
    human_stops: int = 0


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
        human_stops=int(data.get("human_stops", 0)),
        tune=dict(data.get("tune") or {}),
    )


def save_state(path: Path, state: WorkflowState) -> None:
    """Write the checkpoint atomically (fsync, then ``os.replace``)."""
    payload = {"version": SCHEMA_VERSION, **state.__dict__}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=STATE_FILENAME + ".", suffix=".tmp", dir=path.parent)
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
    "SCHEMA_VERSION",
    "STAGE_PLANNED",
    "STAGE_PREFLIGHT",
    "STAGE_TUNE",
    "STATE_FILENAME",
    "WorkflowState",
    "is_remediable",
    "load_state",
    "save_state",
]
