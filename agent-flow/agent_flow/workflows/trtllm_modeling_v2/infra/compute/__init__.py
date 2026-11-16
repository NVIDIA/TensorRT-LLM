# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The GPU side of a run: the pool, its one spool, and how a command reaches it.

Everything that runs ON the compute node (:mod:`.scheduler`) or is called from
a job's shell (:mod:`.spoolctl`, ``scripts/``) is standard-library only and
runs by path, because the interpreter inside the container is the engine's,
not the orchestrator's. The orchestrator's side is :mod:`.executor` (a command
as a job in the spool), :mod:`.pool` (when a holder is needed, and its end),
:mod:`.slurm` (how a holder and a step are built on Slurm) and :mod:`.sessions`
(the watchers' entry scripts).
"""

from pathlib import Path

#: The directory this package lives in; the scheduler and the spool client are resolved from here.
GPU_DIR = Path(__file__).resolve().parent

#: The scheduler, as a file a container's ``python3`` can run.
SCHEDULER_PATH = GPU_DIR / "scheduler.py"

#: The submit/wait/status/cancel client, as a file a shell's ``python3`` can run.
SPOOLCTL_PATH = GPU_DIR / "spoolctl.py"

__all__ = ["GPU_DIR", "SCHEDULER_PATH", "SPOOLCTL_PATH"]
