# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One command, and where it ran.

A workflow that measures something needs to say *what to run* and *how long
it may take*. Where that lands — this host, an ``srun`` step, a file in a
spool a long-lived allocation drains — is a property of the deployment, and
every time it leaked into the workflow it arrived as a new branch beside a
stage transition.

So: the caller builds a :class:`CommandSpec`, an executor returns an
:class:`ExecutionRecord`, and the record carries enough provenance that the
verdict drawn from it can be defended later without knowing which executor
produced it.

**A timeout is not an error here.** ``timed_out`` is a separate field from
the exit code because the two need different handling — a gate that timed out
has said nothing about the thing it was gating, while a gate that exited
non-zero has.
"""

from __future__ import annotations

import abc
import shlex
from dataclasses import dataclass, field
from typing import ClassVar, Mapping, Sequence


@dataclass(frozen=True)
class ExecutionCapabilities:
    """What an executor can actually run, in machine-readable form.

    Stated rather than inferred, so a caller can refuse an impossible
    request before paying for it instead of discovering the limit from a
    failure three hours in.
    """

    gpu: bool = False
    distributed: bool = False
    #: True when the executor runs inside an allocation acquired once and
    #: reused, rather than queueing per command. That is the difference
    #: between paying a scheduler wait (and a container import) per gate and
    #: paying it once for the run.
    persistent_allocation: bool = False
    max_world_size: int | None = None
    architectures: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandSpec:
    """What to run, and the bounds it runs under.

    ``argv`` is either a **string**, which is a shell line and is run as
    one, or a **sequence**, which is executed directly with no shell between.
    Both forms are real and neither is a convenience: a resolved launch
    prefix (environment assignments, or an ``srun`` line) *is* shell syntax
    and re-parsing it into an argv would mean reimplementing the shell, while
    a probe like ``nvidia-smi --query-gpu=...`` has no reason to be exposed
    to word splitting at all.
    """

    argv: str | Sequence[str]
    env: Mapping[str, str] | None = None
    timeout_s: int = 600
    #: A short name for this command in logs and artifacts. Not an id.
    label: str = ""
    #: How many ranks the command expects. Carried so an executor can refuse
    #: a request its allocation cannot satisfy; never used to rewrite the
    #: command, which is the caller's business.
    world_size: int = 1
    gpu_arch: str | None = None
    #: Where to run it, when the executor supports a working directory.
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
class ExecutionRecord:
    """What happened, with the provenance a verdict rests on."""

    command: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_s: float = 0.0
    #: Which executor ran it — ``local``, ``slurm``, ``spool``.
    executor: str = ""
    label: str = ""
    #: Where the full output lives, when the executor wrote one. The tail a
    #: record carries is for reading; this is for auditing.
    log_path: str | None = None
    started_at: float = 0.0
    artifacts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def combined(self) -> str:
        return self.stdout + self.stderr

    def tail(self, lines: int = 40) -> str:
        return "\n".join(self.combined.splitlines()[-lines:])


class CommandExecutor(abc.ABC):
    """Run one command somewhere and report what happened."""

    #: Recorded in :attr:`ExecutionRecord.executor`.
    name: ClassVar[str] = "command-executor"

    #: Overridden per executor. A class attribute rather than a property so
    #: a caller can ask without constructing one.
    capabilities: ExecutionCapabilities = ExecutionCapabilities()

    @abc.abstractmethod
    def execute(self, spec: CommandSpec) -> ExecutionRecord:
        """Run ``spec`` to completion or to its deadline."""


__all__ = [
    "CommandExecutor",
    "CommandSpec",
    "ExecutionCapabilities",
    "ExecutionRecord",
]
