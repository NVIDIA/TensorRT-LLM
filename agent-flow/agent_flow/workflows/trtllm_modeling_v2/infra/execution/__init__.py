# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where this run's commands and agent turns go, decided once from the resolved environment.

`environment.py` answers *what the command line is*: the environment, the
launcher wrapper, the MPI scrub. This package answers *who runs it*, and it is
the only place in the workflow that maps an ``environment.kind`` onto a
mechanism:

- ``local``: a subprocess here;
- ``session_spool``: a file in a spool a long-lived allocation drains
  (:mod:`.spool`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..environment import RunEnvironment
from ..substrate import placement
from .capacity import CheckAdmission
from .commands import CommandExecutor, CommandResult, CommandSpec, LocalCommandExecutor
from .spool import SpoolCommandExecutor
from .turns import (
    TURNS_DIRNAME,
    InProcessTurnExecutor,
    LayerProvider,
    TurnExecutor,
    TurnOutcome,
    TurnRequest,
    TurnTimeout,
)


def select_command_executor(
    environment: RunEnvironment,
    *,
    name_prefix: str = "gate",
    tree: Path | str | None = None,
    gpus: str = "all",
    priority: int = 0,
) -> CommandExecutor:
    """The executor this run's commands go through.

    ``tree`` is the checkout under test: on a spool every command then runs with
    it as its working directory and first ``PYTHONPATH`` entry. ``gpus`` is what
    each spool job asks the scheduler for: a device count, or ``all`` for the
    whole node. The spools of every session declared with ``role: gates`` form
    a pool beside ``environment.spool``, and each command goes to the least
    loaded live one. ``priority`` is the class each spool job queues at
    (``priority.py``).
    """
    if environment.kind == "session_spool" and environment.spool:
        pool = [spool for _, spool, role in environment.sessions if role == "gates"]
        # Another session's idle devices take a gate no gate session can start now.
        overflow = [spool for _, spool, role in environment.sessions if role != "gates"]
        return SpoolCommandExecutor(
            environment.spool,
            name_prefix=name_prefix,
            tree=tree,
            gpus=gpus,
            priority=priority,
            pool=pool,
            overflow=overflow,
            # The tune campaign's dev sessions, only while nothing runs or waits there.
            last_resort=environment.dev_spools,
        )
    return LocalCommandExecutor()


def startable_checks(
    environment: RunEnvironment,
    gpus: str | None = None,
    wall_s: float | None = None,
    *,
    now: float | None = None,
) -> int | None:
    """How many more checks would start at once on this run's sessions.

    A check asks for ``gpus`` devices (``all`` for a whole node; by default
    ``environment.world_size``, its member's width) with ``wall_s`` to run
    (the session's default when ``None``), and goes where the executor
    :func:`select_command_executor` builds sends it
    (:meth:`SpoolCommandExecutor.startable`). ``None`` when no session has a
    live watcher, and under ``local``: nothing there says what is free.
    """
    want = str(environment.world_size if gpus is None else gpus)
    executor = select_command_executor(environment, gpus=want)
    if not isinstance(executor, SpoolCommandExecutor):
        return None
    job = placement.JobNeeds(
        name="check",
        gpus=None if want == "all" else max(1, int(want)),
        wall_s=wall_s,
        holds_gpus=True,
        owner=executor.name_prefix,
    )
    return executor.startable(job, now)


def wait_for_session(executor: CommandExecutor | None, timeout_s: float) -> bool:
    """Wait up to ``timeout_s`` for the session ``executor`` submits to to heart-beat again.

    False at once for an executor with no session behind it: nothing there can come back.
    """
    if isinstance(executor, SpoolCommandExecutor):
        return executor.wait_for_session(timeout_s)
    return False


def shell_runner_for(executor: CommandExecutor, *, default_timeout: int = 1800) -> Any:
    """``executor`` as the ``(command, env, timeout) -> CommandResult`` callable gates take."""

    def run(
        command: str,
        env: Mapping[str, str] | None = None,
        timeout: int = default_timeout,
    ) -> CommandResult:
        return executor.execute(CommandSpec(argv=command, env=env, timeout_s=timeout))

    return run


def select_turn_executor(
    agent_transport: TurnExecutor | None,
    layer_provider: LayerProvider,
    *,
    timeout_s: float | None = None,
) -> TurnExecutor:
    """The transport this run's role turns go through: in process, unless one was given.

    ``timeout_s`` bounds every in-process turn alike (each role has its own
    bound without it).
    """
    return agent_transport or InProcessTurnExecutor(layer_provider, timeout_s=timeout_s)


__all__ = [
    "TURNS_DIRNAME",
    "CheckAdmission",
    "CommandExecutor",
    "CommandResult",
    "CommandSpec",
    "InProcessTurnExecutor",
    "LayerProvider",
    "LocalCommandExecutor",
    "SpoolCommandExecutor",
    "TurnExecutor",
    "TurnOutcome",
    "TurnRequest",
    "TurnTimeout",
    "select_command_executor",
    "select_turn_executor",
    "shell_runner_for",
    "startable_checks",
    "wait_for_session",
]
