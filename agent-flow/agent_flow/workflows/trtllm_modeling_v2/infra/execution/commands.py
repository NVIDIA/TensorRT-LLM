# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One command, what it may take, and what came back.

**A timeout is not an error here**: ``timed_out`` is separate from the exit
code because a gate that timed out has said nothing about what it was gating,
while a gate that exited non-zero has.

**Neither is a lost session.** ``infrastructure`` names the third outcome: the
substrate, not the command, ended the wait (the allocation that would run it
stopped heart-beating, none ever watched, or it was closed out as orphaned). A
caller that would re-measure on a timeout should instead wait for the substrate
to come back.
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

#: What a command whose executable does not exist reports: the shell's own
#: "command not found". Distinct from a non-zero exit: the probe could not run,
#: rather than ran and disagreed.
NOT_FOUND_RETURNCODE = 127


@dataclass(frozen=True)
class CommandSpec:
    """What to run, and the bound it runs under.

    ``argv`` is either a **string**, run as a shell line, or a **sequence**,
    executed directly with no shell between. A resolved launch prefix is shell
    syntax; a probe like ``nvidia-smi --query-gpu=...`` has no reason to be
    exposed to word splitting.
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
    #: Why the substrate ended this command rather than the command itself;
    #: empty when the command ran to its own outcome. A result that carries it
    #: also has ``timed_out`` set.
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

    The environment is the parent's with ``spec.env`` layered on top:
    everything here runs inside an environment somebody already set up (a
    venv's ``PATH``, a container's library paths, ``TMPDIR``), and replacing it
    turns a working host into a broken one.

    The command runs in a process group of its own, and a timeout ends the
    group: a launcher's ranks or a test's workers outlive the shell that
    started them, and would hold their devices for the next command.
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

    Returns what it wrote before it ended; ``("", "")`` when a process that
    left the group keeps its pipes open past the second grace.
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
