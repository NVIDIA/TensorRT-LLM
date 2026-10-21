# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The execution substrate a workflow run stands on: compute sessions and their spools.

Everything that runs ON the compute node (:mod:`.scheduler`) or is called from
a job's shell (:mod:`.spoolctl`) is standard-library only and runs by path,
because the interpreter inside the container is the engine's, not the
orchestrator's.
"""

from pathlib import Path

#: The directory this package lives in; scripts and the scheduler are resolved from here.
SUBSTRATE_DIR = Path(__file__).resolve().parent

#: The scheduler, as a file a container's ``python3`` can run.
SCHEDULER_PATH = SUBSTRATE_DIR / "scheduler.py"

#: The submit/wait/status/cancel client, as a file a shell's ``python3`` can run.
SPOOLCTL_PATH = SUBSTRATE_DIR / "spoolctl.py"

#: Shell programs a job runs: the slot wrapper, the multi-node launch, the native rebuild.
SCRIPTS_DIR = SUBSTRATE_DIR.parent / "scripts"

__all__ = ["SCHEDULER_PATH", "SCRIPTS_DIR", "SPOOLCTL_PATH", "SUBSTRATE_DIR"]
