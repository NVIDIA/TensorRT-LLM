# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The GPU side of a run: compute sessions, their spools, and how a command reaches one.

Everything that runs ON the compute node (:mod:`.scheduler`) or is called from
a job's shell (:mod:`.spoolctl`, ``scripts/``) is standard-library only and
runs by path, because the interpreter inside the container is the engine's,
not the orchestrator's. The orchestrator's side is :mod:`.executor` (a command,
run here or through a spool), :mod:`.placement` (which spool can take it) and
:mod:`.sessions` (declaring, rendering and handing over sessions).
"""

from pathlib import Path

#: The directory this package lives in; the scheduler and the spool client are resolved from here.
GPU_DIR = Path(__file__).resolve().parent

#: The scheduler, as a file a container's ``python3`` can run.
SCHEDULER_PATH = GPU_DIR / "scheduler.py"

#: The submit/wait/status/cancel client, as a file a shell's ``python3`` can run.
SPOOLCTL_PATH = GPU_DIR / "spoolctl.py"

__all__ = ["GPU_DIR", "SCHEDULER_PATH", "SPOOLCTL_PATH"]
