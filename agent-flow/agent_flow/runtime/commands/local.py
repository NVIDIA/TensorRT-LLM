# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a command in a subprocess of this process."""

from __future__ import annotations

import os
import subprocess
import time
from typing import ClassVar

from .base import CommandExecutor, CommandSpec, ExecutionCapabilities, ExecutionRecord

#: What a command that never finished reports. 124 is what ``timeout(1)``
#: uses, so a caller reading exit codes by hand sees a familiar number.
TIMEOUT_RETURNCODE = 124

#: What a command whose executable does not exist reports. 127 is the
#: shell's own "command not found", and it matters that it is distinct from
#: a non-zero exit: the first says the probe could not run, the second says
#: it ran and disagreed.
NOT_FOUND_RETURNCODE = 127


class LocalCommandExecutor(CommandExecutor):
    """A subprocess on this host.

    Covers both shapes a caller needs. A shell line is run through the shell
    because a resolved launch prefix *is* shell syntax; an argv sequence is
    executed directly, because a probe has no reason to be exposed to word
    splitting. :attr:`CommandSpec.argv`'s type decides which, so neither
    caller has to pass a flag.

    The inherited environment is the parent's, with ``spec.env`` layered on
    top. Replacing it outright would be the stricter choice and the wrong
    one: everything here runs inside an environment somebody else already
    set up — a venv's ``PATH``, a container's library paths, ``TMPDIR`` —
    and dropping it turns a working host into a broken one.
    """

    name: ClassVar[str] = "local"
    capabilities = ExecutionCapabilities(gpu=True, distributed=False, max_world_size=None)

    def execute(self, spec: CommandSpec) -> ExecutionRecord:
        merged = {**os.environ, **(spec.env or {})}
        command = spec.rendered
        started_at = time.time()
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command if spec.is_shell_line else list(spec.argv),
                shell=spec.is_shell_line,
                capture_output=True,
                text=True,
                timeout=spec.timeout_s,
                env=merged,
                cwd=spec.cwd,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return self._record(
                spec,
                command,
                started,
                started_at,
                returncode=TIMEOUT_RETURNCODE,
                timed_out=True,
            )
        except FileNotFoundError as exc:
            # Only reachable on the argv path -- a shell reports this itself.
            return self._record(
                spec,
                command,
                started,
                started_at,
                returncode=NOT_FOUND_RETURNCODE,
                stderr=str(exc),
            )
        return self._record(
            spec,
            command,
            started,
            started_at,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    def _record(
        self,
        spec: CommandSpec,
        command: str,
        started: float,
        started_at: float,
        *,
        returncode: int,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
    ) -> ExecutionRecord:
        return ExecutionRecord(
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_s=time.monotonic() - started,
            executor=self.name,
            label=spec.label,
            started_at=started_at,
        )


__all__ = ["NOT_FOUND_RETURNCODE", "TIMEOUT_RETURNCODE", "LocalCommandExecutor"]
