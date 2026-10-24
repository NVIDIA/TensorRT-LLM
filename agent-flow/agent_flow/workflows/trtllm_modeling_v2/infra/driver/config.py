# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a driver works against, resolved once from the run's workspace.

Shared by the operator loop (:mod:`.core`) and the external transport's turn
hand-out (``infra/external/driver.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..dispatch import DISPATCH_DIRNAME
from ..substrate import sessions as sessions_mod

#: Where claims live, under the dispatch directory.
CLAIMS_DIRNAME = "claims"

#: Worker turns a driver runs at once: the orchestrator's default task concurrency,
#: so a full set of tasks is never held back by the driver. Capped: many more
#: concurrent turns on one account stall the API stream.
DEFAULT_MAX_INFLIGHT = 8


class DriverError(RuntimeError):
    """A driver action that cannot be carried out."""


@dataclass
class DriverConfig:
    """What a driver works against, resolved once from the workspace."""

    workspace: Path
    root: Path
    task: Mapping[str, Any] = field(default_factory=dict)
    max_inflight: int = DEFAULT_MAX_INFLIGHT
    #: ``(spool, job name)`` of the orchestrator when it runs as a spool job.
    orchestrator_job: tuple[Path, str] | None = None

    @property
    def dispatch(self) -> Path:
        return self.workspace / DISPATCH_DIRNAME

    @property
    def claims(self) -> Path:
        return self.dispatch / CLAIMS_DIRNAME

    @property
    def skills(self) -> Path:
        return self.workspace / ".claude" / "skills"

    @property
    def sessions(self) -> tuple[sessions_mod.SessionSpec, ...]:
        return sessions_mod.sessions_from_task(self.task)

    @property
    def command(self) -> str:
        return f"trtllm-modeling-v2 driver --workspace {self.workspace}"


def load_config(
    workspace: str | Path,
    *,
    root: str | Path | None = None,
    max_inflight: int = DEFAULT_MAX_INFLIGHT,
    orchestrator_job: str | None = None,
) -> DriverConfig:
    """Resolve a driver's configuration from ``<workspace>/task.yaml``.

    ``orchestrator_job`` is ``<spool>:<job name>`` when the orchestrator runs
    as a spool job; its receipt then also counts as the orchestrator exiting.
    """
    workspace = Path(workspace).resolve()
    if not (workspace / DISPATCH_DIRNAME).is_dir() and not (workspace / "task.yaml").is_file():
        raise DriverError(
            f"{workspace} is not a trtllm_modeling_v2 workspace (no task.yaml and no {DISPATCH_DIRNAME}/)"
        )
    try:
        loaded = yaml.safe_load((workspace / "task.yaml").read_text(encoding="utf-8"))
        task = loaded if isinstance(loaded, dict) else {}
    except (OSError, yaml.YAMLError):
        task = {}
    resolved_root = Path(root or task.get("trtllm_repo_path") or Path.cwd()).resolve()
    job = None
    if orchestrator_job:
        spool, sep, name = orchestrator_job.rpartition(":")
        if not sep or not spool or not name:
            raise DriverError("--orchestrator-job must be <spool>:<job name>")
        job = (Path(spool), name)
    return DriverConfig(
        workspace=workspace,
        root=resolved_root,
        task=task,
        max_inflight=max_inflight,
        orchestrator_job=job,
    )


__all__ = [
    "CLAIMS_DIRNAME",
    "DEFAULT_MAX_INFLIGHT",
    "DriverConfig",
    "DriverError",
    "load_config",
]
