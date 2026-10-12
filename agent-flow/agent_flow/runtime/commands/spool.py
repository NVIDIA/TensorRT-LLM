# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a command by writing a file a long-lived allocation is watching.

The session is an allocation acquired once, whose own main process drains
``<spool>/pending``. Work is a **file**, which is the property that matters:
anything able to write a file can run GPU work, including a plain Python
process on a host with no GPU and no reachable scheduler.

That is what hands command execution back to the orchestrator. The
alternative on such a host is to route every command through an agent
holding a broker tool, and then the verdict rests on that agent's account of
what a number was. Here it rests on a file.

It also removes the per-command scheduler tax: one queue wait and one
container import for the run, instead of one of each per gate.

Completion is the appearance of ``done/<name>.rc``, which the watcher writes
last and renames into place, so a reader that sees it can trust the log
beside it is complete. Same ordering rule as the dispatch mailbox's
``.done``, for the same reason.
"""

from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import ClassVar

from .base import CommandExecutor, CommandSpec, ExecutionCapabilities, ExecutionRecord
from .local import TIMEOUT_RETURNCODE


class SpoolCommandExecutor(CommandExecutor):
    """Submit through a spool directory and block on the receipt."""

    name: ClassVar[str] = "spool"
    capabilities = ExecutionCapabilities(
        gpu=True,
        distributed=True,
        persistent_allocation=True,
    )

    def __init__(
        self,
        spool: str | Path,
        *,
        poll_s: float = 3.0,
        name_prefix: str = "gate",
    ) -> None:
        self.root = Path(spool)
        self.poll_s = poll_s
        self.name_prefix = name_prefix

    def _name(self) -> str:
        return f"{self.name_prefix}-{int(time.time() * 1000)}"

    def execute(self, spec: CommandSpec) -> ExecutionRecord:
        name = self._name()
        command = spec.rendered
        exports = "".join(
            f"export {k}={shlex.quote(str(v))}\n" for k, v in sorted((spec.env or {}).items())
        )
        script = f"#!/bin/bash\nset -u\n{exports}{command}\n"

        (self.root / "pending").mkdir(parents=True, exist_ok=True)
        # Write beside the spool and rename in, so the watcher cannot claim
        # a half-written script.
        staging = self.root / f".{name}.sh.tmp"
        staging.write_text(script, encoding="utf-8")
        staging.rename(self.root / "pending" / f"{name}.sh")

        receipt = self.root / "done" / f"{name}.rc"
        log = self.root / "logs" / f"{name}.log"
        started_at = time.time()
        started = time.monotonic()
        deadline = started + spec.timeout_s
        while not receipt.exists():
            if time.monotonic() > deadline:
                tail = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
                return ExecutionRecord(
                    command=command,
                    returncode=TIMEOUT_RETURNCODE,
                    stdout=tail,
                    timed_out=True,
                    duration_s=time.monotonic() - started,
                    executor=self.name,
                    label=spec.label,
                    log_path=str(log),
                    started_at=started_at,
                )
            time.sleep(self.poll_s)

        returncode = int((receipt.read_text(encoding="utf-8").strip() or "1"))
        output = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        return ExecutionRecord(
            command=command,
            returncode=returncode,
            stdout=output,
            duration_s=time.monotonic() - started,
            executor=self.name,
            label=spec.label,
            log_path=str(log),
            started_at=started_at,
            artifacts=(str(log), str(receipt)),
        )


__all__ = ["SpoolCommandExecutor"]
