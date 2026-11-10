# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who runs a command: a subprocess on this host.

``environment.py`` builds the command line; :func:`select_command_executor` is the one place
that maps an ``environment.kind`` onto who runs it. The workflow ships one kind, ``local``
(:class:`~.commands.LocalCommandExecutor`). :class:`~.commands.CommandExecutor` is the seam:
whoever wants commands to run somewhere else (a batch scheduler, a queue of compute sessions)
wraps this one function in their own package; nothing else in the workflow knows where a
command lands.
"""

from __future__ import annotations

from typing import Any, Mapping

from .commands import CommandExecutor, CommandResult, CommandSpec, LocalCommandExecutor
from .environment import RunEnvironment


def select_command_executor(environment: RunEnvironment) -> CommandExecutor:
    """The executor this run's commands go through."""
    if environment.kind != "local":
        raise ValueError(
            f"environment.kind {environment.kind!r} has no executor here: the workflow runs "
            "commands on the orchestrator's own host (`local`)"
        )
    return LocalCommandExecutor()


def shell_runner_for(executor: CommandExecutor, *, default_timeout: int = 1800) -> Any:
    """``executor`` as the ``(command, env, timeout) -> CommandResult`` callable checks take."""

    def run(
        command: str,
        env: Mapping[str, str] | None = None,
        timeout: int = default_timeout,
    ) -> CommandResult:
        return executor.execute(CommandSpec(argv=command, env=env, timeout_s=timeout))

    return run


__all__ = ["select_command_executor", "shell_runner_for"]
