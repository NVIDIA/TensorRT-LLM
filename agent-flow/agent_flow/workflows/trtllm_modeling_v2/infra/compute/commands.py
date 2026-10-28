# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One command, what it may take, and what came back.

**A timeout is not an error**: ``timed_out`` is apart from the exit code because a gate that
timed out has said nothing about what it was gating, while one that exited non-zero has.
**Neither is a lost session**: ``infrastructure`` says the substrate, not the command, ended the
wait (the allocation stopped heart-beating, none ever watched, or the job was closed out as
orphaned); a caller should wait for the substrate to come back rather than re-measure.
"""

from __future__ import annotations

import abc
import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence

#: What a command that never finished reports: ``timeout(1)``'s code.
TIMEOUT_RETURNCODE = 124

#: What a command whose executable does not exist reports: the shell's "command not found".
#: The probe could not run, rather than ran and disagreed.
NOT_FOUND_RETURNCODE = 127


@dataclass(frozen=True)
class CommandSpec:
    """What to run, and the bound it runs under.

    ``argv`` is a **string** run as a shell line (a launch prefix is shell syntax) or a
    **sequence** run with no shell, safe from word splitting.
    """

    argv: str | Sequence[str]
    env: Mapping[str, str] | None = None
    timeout_s: int = 600
    #: Working directory, where the executor supports one.
    cwd: str | None = None

    @property
    def rendered(self) -> str:
        """``argv`` as a single line, for logs and for shell execution."""
        if isinstance(self.argv, str):
            return self.argv
        return " ".join(shlex.quote(str(part)) for part in self.argv)

    @property
    def is_shell_line(self) -> bool:
        return isinstance(self.argv, str)


@dataclass(frozen=True)
class CommandResult:
    """What a command did: its exit code, its output, and whether it ran out of time."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    #: Why the substrate, not the command, ended it; empty when the command ran to its own
    #: outcome. A result that carries it also has ``timed_out`` set.
    infrastructure: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class CommandExecutor(abc.ABC):
    """Run one command somewhere and report what happened."""

    @abc.abstractmethod
    def execute(self, spec: CommandSpec) -> CommandResult:
        """Run ``spec`` to completion or to its deadline."""


#: How long a timed-out command's processes get to exit on ``SIGTERM``, then on ``SIGKILL``.
TERMINATE_GRACE_S = 10.0


class LocalCommandExecutor(CommandExecutor):
    """A subprocess on this host.

    The environment is the parent's with ``spec.env`` on top: replacing it would drop what the
    host already set up (a venv's ``PATH``, a container's library paths, ``TMPDIR``). The
    command runs in a process group of its own, and a timeout ends the group: a launcher's ranks
    or a test's workers outlive their shell, and would hold their devices for the next command.
    """

    def execute(self, spec: CommandSpec) -> CommandResult:
        try:
            process = subprocess.Popen(
                spec.rendered if spec.is_shell_line else list(spec.argv),
                shell=spec.is_shell_line,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, **(spec.env or {})},
                cwd=spec.cwd,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            # Only reachable on the argv path: a shell reports this itself.
            return CommandResult(returncode=NOT_FOUND_RETURNCODE, stderr=str(exc))
        try:
            stdout, stderr = process.communicate(timeout=spec.timeout_s)
        except subprocess.TimeoutExpired:
            stdout, stderr = end_process_group(process)
            return CommandResult(
                returncode=TIMEOUT_RETURNCODE, stdout=stdout, stderr=stderr, timed_out=True
            )
        except BaseException:
            # A group of its own does not get the terminal's interrupt: it ends here.
            end_process_group(process)
            raise
        return CommandResult(
            returncode=process.returncode, stdout=stdout or "", stderr=stderr or ""
        )


def end_process_group(
    process: subprocess.Popen, grace_s: float = TERMINATE_GRACE_S
) -> tuple[str, str]:
    """End ``process`` and its group: ``SIGTERM``, then ``SIGKILL`` after ``grace_s``.

    Returns what it wrote; ``("", "")`` when a process that left the group keeps the pipes open.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=grace_s)
        except subprocess.TimeoutExpired:
            continue
        return stdout or "", stderr or ""
    return "", ""


__all__ = [
    "NOT_FOUND_RETURNCODE",
    "TIMEOUT_RETURNCODE",
    "CommandExecutor",
    "CommandResult",
    "CommandSpec",
    "LocalCommandExecutor",
    "TERMINATE_GRACE_S",
    "end_process_group",
]
