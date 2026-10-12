# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a command as a Slurm step.

**The step is still a local subprocess**, and saying so is the point. An
``srun`` line is a shell line: this host runs ``srun``, ``srun`` talks to the
controller, and the exit code and output come back down the same pipe as any
other child process. A separate executor class that secretly did the same
thing as the local one would be decoration.

What it is *not* decoration for is the capability declaration and the
provenance tag. ``distributed=True`` and ``executor="slurm"`` are the two
facts a caller and an auditor respectively need, and neither is derivable
from the command string without parsing it.

The ``srun`` line itself — partition, ntasks, ``--gres``, the MPI plugin, the
launcher wrapper — is **not** built here. Those rules belong to whatever
knows the engine being launched, and getting them wrong produces
``MPI_ERR_SPAWN`` rather than an error this layer could diagnose. This
executor receives a line that is already correct.
"""

from __future__ import annotations

from typing import ClassVar

from .base import ExecutionCapabilities
from .local import LocalCommandExecutor


class SlurmStepExecutor(LocalCommandExecutor):
    """A rendered ``srun`` line, executed here, reported as a Slurm step."""

    name: ClassVar[str] = "slurm"
    capabilities = ExecutionCapabilities(gpu=True, distributed=True)

    def __init__(self, *, max_world_size: int | None = None) -> None:
        if max_world_size is not None:
            self.capabilities = ExecutionCapabilities(
                gpu=True, distributed=True, max_world_size=max_world_size
            )


__all__ = ["SlurmStepExecutor"]
