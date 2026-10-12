# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where this run's commands land, decided once, from the resolved environment.

`environment.py` answers *what the command line is* — the ``srun`` prefix,
the launcher wrapper, the MPI scrub. This module answers the other half,
*who runs it*, and it is the only place in the workflow that maps a
``task.yaml`` environment kind onto a mechanism.

That split is the point. The orchestrator used to carry the branch itself::

    if result.environment.kind == "session_spool":
        self._shell_runner = gates_mod.spool_runner(result.environment.spool)

which reads as a small thing and is not: every mechanism added since — the
broker, the spool — arrived as another line beside a stage transition, in
the file that also owns resume and gate sequencing. Deleting a mechanism
should not require reading the state machine.

The two ``*_runner_for`` adapters exist because the call sites this replaces
took plain callables, and there are a lot of them: three gates, two
witnesses, the inventory probe, six preflight probes. Converting them all at
once would have put a mechanical rename in the same change as a structural
one. They are a migration surface, not a permanent one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from agent_flow.runtime import (
    CommandExecutor,
    CommandSpec,
    DelegatedCommandExecutor,
    ExecutionRecord,
    InProcessTurnExecutor,
    LocalCommandExecutor,
    SlurmStepExecutor,
    SpoolCommandExecutor,
    TurnExecutor,
    TurnOutcome,
    TurnRequest,
)
from agent_flow.runtime.turns import LayerProvider

from .environment import RunEnvironment

#: Environment kinds the workflow knows how to resolve. Kept here rather than
#: imported from the schema because this is the list that has *executors*
#: behind it, which is not the same question as which kinds validate.
KIND_LOCAL = "local"
KIND_SLURM = "slurm"
KIND_SLURM_BROKER = "slurm_broker"
KIND_SESSION_SPOOL = "session_spool"

#: Where a delegated run leaves its scripts, logs and receipts. Under the
#: workspace because that is on the shared filesystem both sides can see.
BROKER_DIRNAME = "broker"


def select_command_executor(
    environment: RunEnvironment,
    *,
    name_prefix: str = "gate",
    turns: TurnExecutor | None = None,
    workspace: Path | str | None = None,
    tree: Path | str | None = None,
) -> CommandExecutor:
    """The executor this run's commands go through.

    ``slurm_broker`` needs the two optional arguments and is the reason they
    exist: that cluster's scheduler is reachable only through a tool the
    *agent* holds, so submission is delegated to a turn while the verdict
    stays here — the job writes its log and exit code to ``workspace/broker``
    and the orchestrator reads both itself. See
    :class:`~agent_flow.runtime.DelegatedCommandExecutor`.

    Without them it falls back to the local executor, which is what the
    broker path did before it was wired. That fallback is for callers with no
    turn executor to hand over (a bare `--dry-run`, a test), not a mode
    anybody should gate on.

    ``tree`` is the checkout under test. On a spool every command then runs
    with that checkout as its working directory and first ``PYTHONPATH``
    entry, and the session refuses a result whose processes imported the
    engine from anywhere else.
    """
    kind = environment.kind
    if kind == KIND_SESSION_SPOOL and environment.spool:
        return SpoolCommandExecutor(environment.spool, name_prefix=name_prefix, tree=tree)
    if kind == KIND_SLURM:
        return SlurmStepExecutor(max_world_size=environment.world_size or None)
    if kind == KIND_SLURM_BROKER and turns is not None and workspace is not None:
        return DelegatedCommandExecutor(
            turns,
            root=Path(workspace) / BROKER_DIRNAME,
            render_submission=lambda script: environment.broker_submit(script, script=True),
            name_prefix=name_prefix,
        )
    return LocalCommandExecutor()


def shell_runner_for(executor: CommandExecutor, *, default_timeout: int = 1800) -> Any:
    """Adapt ``executor`` to the ``(command, env, timeout) -> ShellResult`` shape.

    The shape `gates.ShellRunner` names, and what `witness.py` and
    `survey.py` call. ``ShellResult`` is imported lazily to keep this module
    importable from `gates` without a cycle.
    """
    from .gates import ShellResult

    def run(
        command: str,
        env: Mapping[str, str] | None = None,
        timeout: int = default_timeout,
    ) -> ShellResult:
        record = executor.execute(
            CommandSpec(argv=command, env=env, timeout_s=timeout, label="shell")
        )
        return _as_shell_result(record, ShellResult)

    return run


def command_runner_for(executor: CommandExecutor, *, default_timeout: int = 120) -> Any:
    """Adapt ``executor`` to the ``(argv, env, timeout) -> CommandResult`` shape.

    The shape `preflight.CommandRunner` names. A sequence rather than a
    shell line, and that difference is preserved all the way down: a probe
    must not be word-split.
    """
    from .preflight import CommandResult

    def run(
        argv: Any,
        env: Mapping[str, str] | None = None,
        timeout: int = default_timeout,
    ) -> CommandResult:
        record = executor.execute(
            CommandSpec(argv=tuple(str(part) for part in argv), env=env, timeout_s=timeout)
        )
        return CommandResult(
            returncode=record.returncode,
            stdout=record.stdout,
            stderr=record.stderr,
            timed_out=record.timed_out,
        )

    return run


def _as_shell_result(record: ExecutionRecord, shell_result_cls: Any) -> Any:
    return shell_result_cls(
        returncode=record.returncode,
        stdout=record.stdout,
        stderr=record.stderr,
        timed_out=record.timed_out,
    )


# --------------------------------------------------------------------- turns


class _CallableTurnExecutor(TurnExecutor):
    """Wrap a pre-`TurnRequest` transport callable.

    The external transport was wired as ``transport(role=..., what=...,
    prompt=..., system_prompt=..., decision_file=...)`` before the turn
    types existed, and the workflow's constructor still accepts anything
    with that shape. Keeping the door open costs one adapter; closing it
    would break a caller that has no way to know it was supposed to change.
    """

    name = "callable"

    def __init__(self, call: Any) -> None:
        self._call = call

    def run(self, request: TurnRequest) -> TurnOutcome:
        text = self._call(
            role=request.role,
            what=request.what,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            decision_file=request.decision_file,
        )
        return TurnOutcome(
            role=request.role,
            text=text if isinstance(text, str) else str(text),
            transport=getattr(self._call, "name", self.name),
            decision_ref=request.decision_file,
        )


def select_turn_executor(
    agent_transport: Any,
    layer_provider: LayerProvider,
) -> TurnExecutor:
    """The executor this run's role turns go through.

    ``None`` means the ordinary path: this process builds a layer per role
    and calls it. Anything else means the host could not be trusted to
    authenticate one, and the turn — **and only the turn** — has been moved
    somewhere that can.
    """
    if agent_transport is None:
        return InProcessTurnExecutor(layer_provider)
    if isinstance(agent_transport, TurnExecutor):
        return agent_transport
    return _CallableTurnExecutor(agent_transport)


__all__ = [
    "BROKER_DIRNAME",
    "KIND_LOCAL",
    "KIND_SESSION_SPOOL",
    "KIND_SLURM",
    "KIND_SLURM_BROKER",
    "command_runner_for",
    "select_command_executor",
    "select_turn_executor",
    "shell_runner_for",
]
