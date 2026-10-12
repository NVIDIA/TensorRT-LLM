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

Each script carries a header line the session's scheduler reads::

    # spool: gpus=all wall=<timeout_s> owner=<name_prefix> token=<uuid>

``gpus`` says how many devices the command holds (``all`` is the whole node,
exclusively), ``wall`` is the deadline the scheduler enforces from the job's
start, and ``token`` ties the receipt to this submission: a receipt whose
``done/<name>.json`` carries another token belongs to someone else. With a
``tree`` the scheduler runs the command in that checkout -- working directory
and first ``PYTHONPATH`` entry -- and refuses the result if any process
imported the package under test from somewhere else.

**The deadline counts from the job's start, not from its submission.** Time
spent queued behind other work says nothing about the command, and charging
it to the command failed gates that never ran. While the job is queued the
wait is bounded by the session's liveness instead: a spool whose watchers
stop heart-beating fails the command after ``stale_s``, so a dead session
cannot hang its caller. A spool that has never had a watcher keeps the
original bound (``timeout_s`` from submission).
"""

from __future__ import annotations

import itertools
import json
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import ClassVar

from .base import CommandExecutor, CommandSpec, ExecutionCapabilities, ExecutionRecord
from .local import TIMEOUT_RETURNCODE

#: How long a receipt may exist without an exit code in it before it is read as a failure.
RECEIPT_GRACE_S = 30.0


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
        gpus: str = "all",
        tree: str | Path | None = None,
        queue_timeout_s: float | None = None,
        stale_s: float = 180.0,
        grace_s: float = 180.0,
    ) -> None:
        self.root = Path(spool)
        #: Bound on time spent queued; ``None`` waits as long as the session is alive.
        self.queue_timeout_s = queue_timeout_s
        #: A session whose watchers have not heart-beat for this long is gone.
        self.stale_s = stale_s
        #: Allowance past the job's own deadline for the session to stop it and write the receipt.
        self.grace_s = grace_s
        self.poll_s = poll_s
        self.name_prefix = name_prefix
        self.gpus = gpus
        self.tree = str(Path(tree).resolve()) if tree else None
        self._seq = itertools.count(1)

    def _name(self) -> str:
        """Unique across processes and threads: two gates in one millisecond must not collide."""
        return f"{self.name_prefix}-{int(time.time() * 1000)}-{os.getpid()}-{next(self._seq)}"

    def _header(self, spec: CommandSpec, token: str) -> str:
        fields = [f"gpus={self.gpus}", f"owner={self.name_prefix}", f"token={token}"]
        if spec.timeout_s and spec.timeout_s > 0:
            fields.insert(1, f"wall={int(spec.timeout_s)}")
        if self.tree and not spec.cwd:
            fields.append(f"tree={self.tree}")
        return "#spool: " + " ".join(fields) + "\n"

    def _record(self, name: str) -> dict:
        """The scheduler's verdict record for ``name``; empty when it wrote none."""
        try:
            record = json.loads((self.root / "done" / f"{name}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return record if isinstance(record, dict) else {}

    def _watcher_age(self, now: float) -> float | None:
        """Seconds since any watcher on this spool heart-beat; ``None`` if none ever did."""
        ages = []
        for path in (self.root / "watchers").glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                ages.append(now - float(record["alive_epoch"]))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        beat = self.root / "watcher.heartbeat"
        if beat.exists():
            try:
                ages.append(now - beat.stat().st_mtime)
            except OSError:
                pass
        return min(ages) if ages else None

    def _started_epoch(self, name: str) -> float | None:
        """When the session started ``name``, once it has."""
        try:
            record = json.loads(
                (self.root / "running" / f"{name}.json").read_text(encoding="utf-8")
            )
            return float(record["started_epoch"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        record = self._record(name)
        if record.get("started_epoch"):
            return float(record["started_epoch"])
        return None

    @staticmethod
    def _exit_code(receipt: Path) -> int | None:
        """The exit code a receipt holds, or ``None`` while it holds none yet.

        A receipt that exists is not yet a receipt: a writer that does not
        rename into place creates the file empty and fills it a moment later,
        and reading it in between turned a passing command into exit code 1.
        """
        try:
            text = receipt.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _receipt_is_ours(self, name: str, token: str) -> bool:
        """A receipt with another token in its record is a previous job of the same name."""
        theirs = self._record(name).get("token")
        return not theirs or theirs == token

    def execute(self, spec: CommandSpec) -> ExecutionRecord:
        name = self._name()
        token = uuid.uuid4().hex
        command = spec.rendered
        exports = "".join(
            f"export {k}={shlex.quote(str(v))}\n" for k, v in sorted((spec.env or {}).items())
        )
        cwd = f"cd {shlex.quote(spec.cwd)} || exit 1\n" if spec.cwd else ""
        script = f"#!/bin/bash\nset -u\n{self._header(spec, token)}{cwd}{exports}{command}\n"

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
        job_started: float | None = None
        unreadable_since: float | None = None
        while True:
            if receipt.exists() and self._receipt_is_ours(name, token):
                returncode = self._exit_code(receipt)
                if returncode is not None:
                    break
                unreadable_since = unreadable_since or time.monotonic()
                if time.monotonic() - unreadable_since > RECEIPT_GRACE_S:
                    # Present but never completed: its writer died mid-write.
                    returncode = 1
                    break
            now = time.time()
            if job_started is None:
                job_started = self._started_epoch(name)
                if job_started is None and (self.root / "running" / f"{name}.sh").exists():
                    # Claimed by a watcher that writes no start record (the
                    # legacy flock watcher): count from when it was seen claimed.
                    job_started = now
            age = self._watcher_age(now)
            reason = None
            if job_started is not None:
                if now > job_started + spec.timeout_s + self.grace_s:
                    reason = (
                        f"ran past its {spec.timeout_s}s deadline (counted from its start) "
                        f"and the session did not close it"
                    )
            elif age is None:
                if now > started_at + spec.timeout_s:
                    reason = f"nothing started it within {spec.timeout_s}s and no session watches {self.root}"
            elif self.queue_timeout_s is not None and now > started_at + self.queue_timeout_s:
                reason = f"queued for more than {self.queue_timeout_s}s without starting"
            if reason is None and age is not None and age > self.stale_s:
                reason = f"the session watching {self.root} stopped heart-beating {age:.0f}s ago"
            if reason is not None:
                tail = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
                return ExecutionRecord(
                    command=command,
                    returncode=TIMEOUT_RETURNCODE,
                    stdout=f"{tail}\n=== spool executor: {reason} ===\n",
                    timed_out=True,
                    duration_s=time.monotonic() - started,
                    executor=self.name,
                    label=spec.label,
                    log_path=str(log),
                    started_at=started_at,
                )
            time.sleep(self.poll_s)

        output = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        return ExecutionRecord(
            command=command,
            returncode=returncode,
            stdout=output,
            # The scheduler stops a job at its wall limit, or when it stops
            # making progress; either way it said nothing about what it ran.
            timed_out=bool(self._record(name).get("timed_out")),
            duration_s=time.monotonic() - started,
            executor=self.name,
            label=spec.label,
            log_path=str(log),
            started_at=started_at,
            artifacts=(str(log), str(receipt)),
        )


__all__ = ["SpoolCommandExecutor"]
