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

Agent turns run in process (:class:`InProcessTurnExecutor`) unless the run was
given another transport (`transport.ExternalAgentTransport`). Either bounds a
turn by its role's timeout; one in process leaves its prompt, streamed activity
and answer under ``<workspace>/turns/`` (:mod:`.turns`).

How many checks run at once follows the devices those spools have free
(:func:`check_capacity`, :class:`CheckAdmission`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..environment import RunEnvironment
from ..substrate import placement
from .capacity import (
    DEFAULT_CHECK_CAP,
    DEFAULT_CHECK_FALLBACK,
    DEFAULT_CHECK_FLOOR,
    CheckAdmission,
    more_checks,
)
from .commands import (
    NOT_FOUND_RETURNCODE,
    TIMEOUT_RETURNCODE,
    CommandExecutor,
    CommandResult,
    CommandSpec,
    LocalCommandExecutor,
)
from .spool import SpoolCommandExecutor
from .turns import (
    DEFAULT_ROLE_TURN_TIMEOUTS,
    DEFAULT_TURN_TIMEOUT_S,
    TURNS_DIRNAME,
    InProcessTurnExecutor,
    LayerProvider,
    TurnExecutionError,
    TurnExecutor,
    TurnOutcome,
    TurnRequest,
    TurnTimeout,
    turns_in_flight,
)

KIND_LOCAL = "local"
KIND_SESSION_SPOOL = "session_spool"


def select_command_executor(
    environment: RunEnvironment,
    *,
    name_prefix: str = "gate",
    tree: Path | str | None = None,
    gpus: str = "all",
) -> CommandExecutor:
    """The executor this run's commands go through.

    ``tree`` is the checkout under test: on a spool every command then runs with
    it as its working directory and first ``PYTHONPATH`` entry. ``gpus`` is what
    each spool job asks the scheduler for: a device count, or ``all`` for the
    whole node. The spools of every session declared with ``role: gates`` form
    a pool beside ``environment.spool``, and each command goes to the least
    loaded live one.
    """
    kind = environment.kind
    if kind == KIND_SESSION_SPOOL and environment.spool:
        # Every gate session's spool is a place a gate may run.
        pool = [spool for _, spool, role in environment.sessions if role == "gates"]
        # Another session's idle devices take a gate no gate session can start now.
        overflow = [spool for _, spool, role in environment.sessions if role != "gates"]
        return SpoolCommandExecutor(
            environment.spool,
            name_prefix=name_prefix,
            tree=tree,
            gpus=gpus,
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


def check_capacity(
    environment: RunEnvironment,
    now: float | None = None,
    *,
    gpus: str | None = None,
    wall_s: float | None = None,
    in_flight: int = 0,
    starting: int = 0,
    floor: int = DEFAULT_CHECK_FLOOR,
    cap: int = DEFAULT_CHECK_CAP,
    fallback: int = DEFAULT_CHECK_FALLBACK,
) -> int:
    """How many more checks may start now, with ``in_flight`` already running.

    :func:`startable_checks` read through :func:`more_checks`: never fewer
    than ``floor`` in flight, never more than ``cap``, and ``fallback`` when
    no session can be read. File reads only.
    """
    return more_checks(
        startable_checks(environment, gpus, wall_s, now=now),
        in_flight=in_flight,
        starting=starting,
        floor=floor,
        cap=cap,
        fallback=fallback,
    )


def wait_for_session(executor: CommandExecutor | None, timeout_s: float) -> bool:
    """Wait up to ``timeout_s`` for the session ``executor`` submits to to heart-beat again.

    What a result whose ``infrastructure`` is set asks of its caller before the
    command is worth running again. False at once for an executor with no
    session behind it: nothing there can come back.
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
    records: Path | str | None = None,
) -> TurnExecutor:
    """The transport this run's role turns go through: in process, unless one was given.

    ``timeout_s`` bounds every in-process turn alike (each role has its own
    bound without it), and ``records`` is where each one leaves its files; a
    given transport keeps its own.
    """
    return agent_transport or InProcessTurnExecutor(
        layer_provider, timeout_s=timeout_s, records=records
    )


__all__ = [
    "DEFAULT_CHECK_CAP",
    "DEFAULT_CHECK_FALLBACK",
    "DEFAULT_CHECK_FLOOR",
    "DEFAULT_ROLE_TURN_TIMEOUTS",
    "DEFAULT_TURN_TIMEOUT_S",
    "KIND_LOCAL",
    "KIND_SESSION_SPOOL",
    "NOT_FOUND_RETURNCODE",
    "TIMEOUT_RETURNCODE",
    "TURNS_DIRNAME",
    "CheckAdmission",
    "CommandExecutor",
    "CommandResult",
    "CommandSpec",
    "InProcessTurnExecutor",
    "LayerProvider",
    "LocalCommandExecutor",
    "SpoolCommandExecutor",
    "TurnExecutionError",
    "TurnExecutor",
    "TurnOutcome",
    "TurnRequest",
    "TurnTimeout",
    "check_capacity",
    "more_checks",
    "select_command_executor",
    "select_turn_executor",
    "shell_runner_for",
    "startable_checks",
    "turns_in_flight",
    "wait_for_session",
]
