# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who runs a command: a subprocess here, or a job in a compute session's spool.

``environment.py`` builds the command line; :func:`select_command_executor` is the one place that
maps an ``environment.kind`` onto who runs it: ``local`` a subprocess here (:mod:`.commands`),
``session_spool`` a job in a spool (``spoolctl.py``'s protocol). :class:`SpoolCommandExecutor` is
the gates' policy over the spool:

- **The deadline counts from the job's start**: time queued behind other work says nothing about
  the command. A queued wait is bounded by the session's liveness instead, so a dead session
  cannot hang its caller.
- **A command given up on is withdrawn** (``cancel/<name>``), so a session that comes back does
  not run work nobody waits for. A spool whose heartbeat is already stale gets nothing queued.
- **A lost session is not a failed command**: the result says so in ``infrastructure``, and
  :meth:`SpoolCommandExecutor.wait_for_session` waits for one to come back before a re-run.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import placement, spoolctl
from .commands import (
    TIMEOUT_RETURNCODE,
    CommandExecutor,
    CommandResult,
    CommandSpec,
    LocalCommandExecutor,
)
from .environment import RunEnvironment
from .scheduler import SESSION_LOSS_REASONS, WATCHER_STALE_S, SpoolPaths, watcher_age

#: The classes a job queues at; the spool scheduler starts a waiting job of a
#: higher one first. Checks on the way to the first unit's accuracy: its boot and benchmark.
ACCURACY = 2
#: Every other check: the feature units, the variants, the proofs, delivery.
FEATURE = 1
#: The tune campaign's measurements and candidates.
PERF = 0

_local = threading.local()


@contextlib.contextmanager
def job_priority(priority: int) -> Iterator[None]:
    """Jobs submitted in this thread, inside the block, queue at ``priority``.

    Set where the job's purpose is known, read where it is submitted (:func:`current_job_priority`),
    so the code between needs no argument for it.
    """
    previous = getattr(_local, "priority", None)
    _local.priority = priority
    try:
        yield
    finally:
        _local.priority = previous


def current_job_priority() -> int:
    """The class this thread's jobs queue at; :data:`PERF` (the scheduler's 0) by default."""
    priority = getattr(_local, "priority", None)
    return PERF if priority is None else int(priority)


class SpoolCommandExecutor(CommandExecutor):
    """Submit through a spool directory and block on the receipt."""

    def __init__(
        self,
        spool: str | Path,
        *,
        poll_s: float = 3.0,
        name_prefix: str = "gate",
        gpus: str = "all",
        nodes: int = 1,
        priority: int = 0,
        tree: str | Path | None = None,
        pool: Sequence[str | Path] = (),
    ) -> None:
        self.root = Path(spool)
        #: Every spool a command may go to, ``spool`` first.
        self.pool = [self.root, *(Path(p) for p in pool if Path(p) != self.root)]
        self._pick_lock = threading.Lock()
        self.poll_s = poll_s
        self.name_prefix = name_prefix
        # Gang execution reserves every device of each participating node.
        self.gpus = "all" if nodes > 1 else gpus
        if nodes < 1:
            raise ValueError("nodes must be positive")
        self.nodes = nodes
        #: The class each job queues at (:data:`ACCURACY`, :data:`FEATURE`, :data:`PERF`).
        self.priority = priority
        #: The checkout a job runs against (its working directory and first ``PYTHONPATH``
        #: entry); the session refuses a result that imported the package from anywhere else.
        self.tree = str(Path(tree).resolve()) if tree else None
        #: Allowance past a job's own deadline for the session to stop it.
        self.grace_s = 180.0
        self._seq = itertools.count(1)

    def _name(self) -> str:
        """Unique across processes and threads: two gates in one millisecond must not collide.

        The pid alone does not separate processes: each sandboxed command has its own pid
        namespace, where two gates started together share a small pid.
        """
        return (
            f"{self.name_prefix}-{int(time.time() * 1000)}-{os.getpid()}-"
            f"{uuid.uuid4().hex[:6]}-{next(self._seq)}"
        )

    def _alive(self, root: Path, now: float | None = None) -> bool:
        age = watcher_age(root, now)
        return age is not None and age <= WATCHER_STALE_S

    def session_alive(self) -> bool:
        """Whether a watcher on any spool of the pool has heart-beat within ``WATCHER_STALE_S``."""
        return any(self._alive(root) for root in self.pool)

    def _pick(self, job: placement.JobNeeds, now: float | None = None) -> Path:
        """The live spool of the pool to send a command to; the pool's first when none is live.

        Only the live spools that can take ``job`` now compete
        (:attr:`placement.SpoolView.pickable`): a session in hand-over, or one ending before the
        job's wall limit, would park it for a successor that may be hours away. One that starts
        it at once comes first, then the least loaded; failing all, the least loaded live one.
        """
        if len(self.pool) == 1:
            return self.root
        now = time.time() if now is None else now
        live: list[tuple[int, Path]] = []
        ready: list[tuple[bool, int, Path]] = []
        for root in self.pool:
            if not self._alive(root, now):
                continue
            paths = SpoolPaths(root)
            load = sum(1 for _ in paths.pending.glob("*.sh")) + sum(
                1 for _ in paths.running.glob("*.sh")
            )
            live.append((load, root))
            view = placement.assess(root, job, stale_s=WATCHER_STALE_S, now=now)
            if view.pickable:
                # A job already waiting is seen at once; the watcher's free devices
                # only on its next tick.
                waiting = any(paths.pending.glob("*.sh"))
                ready.append((not (view.starts_now and not waiting), load, root))
        if ready:
            return min(ready, key=lambda choice: choice[:2])[2]
        return min(live, key=lambda choice: choice[0])[1] if live else self.root

    def wait_for_session(self, timeout_s: float, *, poll_s: float | None = None) -> bool:
        """Wait at most ``timeout_s`` for a live watcher in the session pool."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        interval = self.poll_s if poll_s is None else poll_s
        while not self.session_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(interval, remaining))
        return True

    def _withdraw(self, root: Path, name: str) -> str:
        try:
            flag = spoolctl.cancel_job(root, name)
        except OSError as exc:
            return f"; withdrawing it failed ({exc}), so a session draining {root} may still run it"
        return f"; withdrawn through {flag}"

    def _log(self, root: Path, name: str) -> str:
        log = SpoolPaths(root).log(name)
        return log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""

    def execute(self, spec: CommandSpec) -> CommandResult:
        name = self._name()
        token = uuid.uuid4().hex
        submitted = time.time()
        header = {
            "gpus": self.gpus,
            "nodes": self.nodes if self.nodes > 1 else None,
            "wall": int(spec.timeout_s) if spec.timeout_s and spec.timeout_s > 0 else None,
            "owner": self.name_prefix,
            "token": token,
            "tree": self.tree if self.tree and not spec.cwd else None,
            "priority": self.priority or None,
        }
        script = spoolctl.render_job(spec.rendered, header=header, cwd=spec.cwd, env=spec.env)
        with self._pick_lock:
            root = self._pick(placement.JobNeeds.from_script(name, script), submitted)
            age = watcher_age(root, submitted)
            if age is None or age <= WATCHER_STALE_S:
                root.mkdir(parents=True, exist_ok=True)
                spoolctl.queue_job(
                    root, name, script, owner=self.name_prefix, require_watcher=False
                )
        if age is not None and age > WATCHER_STALE_S:
            reason = (
                f"the session watching {root} stopped heart-beating {age:.0f}s ago; "
                f"nothing was queued"
            )
            return CommandResult(
                returncode=TIMEOUT_RETURNCODE,
                stdout=f"=== spool executor: {reason} ===\n",
                timed_out=True,
                infrastructure=reason,
            )
        try:
            record = spoolctl.wait_for_job(
                root,
                name,
                token=token,
                poll_s=self.poll_s,
                timeout_s=spec.timeout_s,
                grace_s=self.grace_s,
                unwatched_s=spec.timeout_s,
                submitted_at=submitted,
            )
        except spoolctl.SpoolError as exc:
            reason = str(exc) + self._withdraw(root, name)
            lost = exc.exit_code == spoolctl.EXIT_NO_SESSION
            return CommandResult(
                returncode=TIMEOUT_RETURNCODE,
                stdout=f"{self._log(root, name)}\n=== spool executor: {reason} ===\n",
                timed_out=True,
                infrastructure=reason if lost else "",
            )
        lost = ""
        if record.get("reason") in SESSION_LOSS_REASONS:
            lost = (
                f"the session that ran it stopped heart-beating and the job was closed "
                f"out as {record['reason']}: {record.get('detail') or 'no detail recorded'}"
            )
        return CommandResult(
            returncode=int(record["rc"]),
            stdout=self._log(root, name),
            # The scheduler stops a job at its wall limit or when it stops making
            # progress, and a job whose session died under it measured nothing.
            timed_out=bool(record.get("timed_out")) or bool(lost),
            infrastructure=lost,
        )


def select_command_executor(
    environment: RunEnvironment,
    *,
    name_prefix: str = "gate",
    tree: Path | str | None = None,
    gpus: str = "all",
    nodes: int = 1,
    priority: int = 0,
) -> CommandExecutor:
    """The executor this run's commands go through.

    On a spool, ``tree`` (the checkout under test) is every job's working directory and first
    ``PYTHONPATH`` entry, and ``gpus`` (a count, or ``all``) what each job asks for. The spools of
    every ``role: gates`` session form a pool beside ``environment.spool``, and each command goes
    to the live one that would start it soonest.
    """
    if environment.kind == "session_spool" and environment.spool:
        pool = [session.spool for session in environment.sessions if session.role == "gates"]
        return SpoolCommandExecutor(
            environment.spool,
            name_prefix=name_prefix,
            tree=tree,
            gpus=gpus,
            nodes=nodes,
            priority=priority,
            pool=pool,
        )
    if nodes > 1:
        raise ValueError("multi-node commands require session_spool gang execution")
    return LocalCommandExecutor()


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


__all__ = [
    "ACCURACY",
    "FEATURE",
    "PERF",
    "SESSION_LOSS_REASONS",
    "SpoolCommandExecutor",
    "current_job_priority",
    "job_priority",
    "select_command_executor",
    "shell_runner_for",
    "wait_for_session",
]
