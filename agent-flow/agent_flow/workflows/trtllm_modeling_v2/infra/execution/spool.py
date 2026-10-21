# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a command through a compute-session spool and block on its receipt.

The session is an allocation acquired once whose main process drains
``<spool>/pending`` (`infra/substrate/scheduler.py`): one queue wait and one
container import for the run instead of one of each per gate, and work any
process able to write a file can submit. The protocol is `infra/substrate/spoolctl.py`'s;
this executor is the gate's policy over it:

- **The deadline counts from the job's start**, not from its submission: time
  queued behind other work says nothing about the command. While it is queued,
  the wait is bounded by the session's liveness instead, so a dead session
  cannot hang its caller.
- **A command given up on is withdrawn** (``cancel/<name>``), so a session that
  comes back does not run work nobody is waiting for. A spool whose heartbeat is
  already stale gets nothing queued at all.
- **A lost session is not a failed command**: when the session, not the command,
  ended the wait, the result says so in ``infrastructure``, and
  :meth:`SpoolCommandExecutor.wait_for_session` is how a caller waits for one to
  come back before running the command again.
- **A pool spreads the work**: with several spools (one per gate session), each
  command goes to the live spool with the least work queued or running among
  those that can take it now (``placement``: not in hand-over, and with the
  command's wall limit left before the allocation ends); to the least loaded
  live one when none can.
- **Idle devices elsewhere take the overflow**: a command no spool of the pool
  would start at once goes to another session's spool that would (its devices
  free, nothing queued); failing that, to a last-resort spool (the tune
  campaign's dev sessions) while nothing of the campaign's runs or waits there.
- **A queued command moves to where it starts**: while it waits in one queue,
  a spool that would start it now takes it (:meth:`SpoolCommandExecutor._relocate`),
  so devices freed elsewhere, or a session that died, do not keep it waiting.
  :meth:`SpoolCommandExecutor.startable` counts by the same rules how many
  more commands would start at once.
"""

from __future__ import annotations

import itertools
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Sequence

from ..substrate import placement, spoolctl
from ..substrate.scheduler import SESSION_LOSS_REASONS, SpoolPaths, job_spec_from_script
from .commands import TIMEOUT_RETURNCODE, CommandExecutor, CommandResult, CommandSpec

#: Where a command may go, in the order :meth:`SpoolCommandExecutor._pick` tries.
_POOL, _OVERFLOW, _LAST_RESORT = "pool", "overflow", "last_resort"


class SpoolCommandExecutor(CommandExecutor):
    """Submit through a spool directory and block on the receipt."""

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
        pool: Sequence[str | Path] = (),
        overflow: Sequence[str | Path] = (),
        last_resort: Sequence[str | Path] = (),
    ) -> None:
        self.root = Path(spool)
        #: Every spool a command may go to, ``spool`` first.
        self.pool = [self.root, *(Path(p) for p in pool if Path(p) != self.root)]
        #: Spools of other sessions a command goes to only when it starts there
        #: at once and no spool of the pool can start it now.
        self.overflow = [Path(p) for p in overflow if Path(p) not in self.pool]
        #: Spools kept for other work, lent only when neither the pool nor the
        #: overflow would start a command at once, and only while nothing waits
        #: there and nothing but commands it lent already runs there.
        self.last_resort = [
            Path(p) for p in last_resort if Path(p) not in (*self.pool, *self.overflow)
        ]
        self._pick_lock = threading.Lock()
        self.poll_s = poll_s
        self.name_prefix = name_prefix
        self.gpus = gpus
        #: The checkout a job runs against: its working directory and first
        #: ``PYTHONPATH`` entry; the session refuses a result that imported the
        #: package under test from anywhere else.
        self.tree = str(Path(tree).resolve()) if tree else None
        #: Bound on time spent queued; ``None`` waits as long as the session is alive.
        self.queue_timeout_s = queue_timeout_s
        #: A session whose watchers have not heart-beat for this long is gone.
        self.stale_s = stale_s
        #: Allowance past a job's own deadline for the session to stop it.
        self.grace_s = grace_s
        self._seq = itertools.count(1)

    def _name(self) -> str:
        """Unique across processes and threads: two gates in one millisecond must not collide.

        The pid alone does not separate processes: each sandboxed command runs in
        its own pid namespace, where two gates started together share a small pid.
        """
        return (
            f"{self.name_prefix}-{int(time.time() * 1000)}-{os.getpid()}-"
            f"{uuid.uuid4().hex[:6]}-{next(self._seq)}"
        )

    def _alive(self, root: Path) -> bool:
        age = spoolctl.watcher_age(root)
        return age is not None and age <= self.stale_s

    def session_alive(self) -> bool:
        """Whether a watcher on any spool of the pool has heart-beat within ``stale_s``."""
        return any(self._alive(root) for root in self.pool)

    def _pick(self, job: placement.JobNeeds | None = None, now: float | None = None) -> Path:
        """The live spool to send a command to; the first of the pool when none is live.

        With ``job``, only the live spools that can take it now compete
        (:attr:`placement.SpoolView.pickable`): a session in hand-over, or
        one whose allocation ends before the job's wall limit, would park it
        for a successor that may be hours away. One that would start it at
        once comes first, then the least loaded. When no spool can take it,
        the least loaded live one is still the best place to queue it.

        When no spool of the pool would start ``job`` at once, an
        :attr:`overflow` spool that would (its devices free, nothing queued,
        out of hand-over) takes it: the gates wait behind each other while
        another session's devices stand idle. When none would either, a
        :attr:`last_resort` spool that would start it takes it, while nothing
        waits there and nothing of its own runs (:meth:`_lendable`).
        """
        now = time.time() if now is None else now
        if len(self.pool) == 1 and not (job is not None and (self.overflow or self.last_resort)):
            return self.root
        live: list[tuple[int, Path]] = []
        ready: list[tuple[bool, int, Path]] = []
        starts_now = False
        for root in self.pool:
            if not self._alive(root):
                continue
            paths = SpoolPaths(root)
            load = sum(1 for _ in paths.pending.glob("*.sh")) + sum(
                1 for _ in paths.running.glob("*.sh")
            )
            live.append((load, root))
            if job is not None:
                view = placement.assess(root, job, stale_s=self.stale_s, now=now)
                if view.pickable:
                    starts = view.starts_now and _nothing_queued(root)
                    ready.append((not starts, load, root))
                    starts_now = starts_now or starts
        if job is not None and not starts_now:
            for spools, free_enough in (
                (self.overflow, _nothing_queued),
                (self.last_resort, self._lendable),
            ):
                idle = [
                    view
                    for view in (
                        placement.assess(root, job, stale_s=self.stale_s, now=now)
                        for root in spools
                        if self._alive(root)
                    )
                    if view.pickable and view.starts_now and free_enough(view.path)
                ]
                if idle:
                    return min(idle, key=placement.rank).path
        if ready:
            # One that starts the job at once first, then the least loaded.
            return min(ready, key=lambda choice: choice[:2])[2]
        return min(live, key=lambda choice: choice[0])[1] if live else self.root

    def startable(self, job: placement.JobNeeds, now: float | None = None) -> int | None:
        """How many more jobs like ``job`` would start at once where :meth:`_pick` sends them.

        Summed over every spool with a live watcher (:meth:`_offer`); ``None``
        when there is none, so there is nothing to count. File reads only: each
        spool's watcher records and its queue on disk.
        """
        now = time.time() if now is None else now
        offers = [
            self._offer(root, job, now, tier)
            for tier, spools in (
                (_POOL, self.pool),
                (_OVERFLOW, self.overflow),
                (_LAST_RESORT, self.last_resort),
            )
            for root in spools
        ]
        live = [offer for offer in offers if offer is not None]
        return sum(live) if live else None

    def _offer(self, root: Path, job: placement.JobNeeds, now: float, tier: str) -> int | None:
        """How many jobs like ``job`` ``root`` would start at once; ``None`` with no live watcher.

        Each live watcher that can take the job (:attr:`placement.SpoolView.pickable`)
        fits as many as its free devices hold. A spool of the pool offers those
        fits less what the jobs queued there on disk will take first; an
        overflow spool offers them only with nothing queued, and a last-resort
        spool only while it is :meth:`_lendable`: what :meth:`_pick` would send
        to each.
        """
        views = placement.watcher_views(root, job, stale_s=self.stale_s, now=now)
        if not views:
            return None
        fits = sum(_fits(job, view) for view in views if view.pickable)
        if tier == _OVERFLOW:
            return fits if _nothing_queued(root) else 0
        if tier == _LAST_RESORT:
            return fits if self._lendable(root) else 0
        node = max(view.devices for view in views)
        taken = sum(_fits_taken(job, queued, node) for queued in placement.queued_jobs(root))
        return max(0, fits - taken)

    def wait_for_session(self, timeout_s: float, *, poll_s: float | None = None) -> bool:
        """Block until a watcher heart-beats on this spool again, for at most ``timeout_s``."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        interval = self.poll_s if poll_s is None else poll_s
        while not self.session_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(interval, remaining))
        return True

    def _lendable(self, root: Path) -> bool:
        """Whether a last-resort spool lends its free devices: nothing waits, and nothing of its own runs.

        Its own work (a campaign's profile, say) must not share the node with
        a gate; gates this executor lent it already may.
        """
        if not _nothing_queued(root):
            return False
        for script in SpoolPaths(root).running.glob("*.sh"):
            try:
                text = script.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return False
            if job_spec_from_script(script.stem, text).owner != self.name_prefix:
                return False
        return True

    def _elsewhere(self) -> bool:
        """Whether a command could go to more than one spool."""
        return len(self.pool) + len(self.overflow) + len(self.last_resort) > 1

    def _relocate(self, job: placement.JobNeeds, current: Path) -> Path | None:
        """Move ``job``, still queued in ``current``, to a spool that would start it now; that spool.

        A job is placed when it is submitted, and a session that frees
        devices later, or a queue whose session died, would otherwise keep it
        waiting while devices elsewhere stand idle. The script is renamed into
        the other spool's queue: a watcher that claimed it first has renamed it
        into its own ``running/``, and the rename fails, so a job never runs
        twice. ``None`` when it stays.
        """
        now = time.time()
        with self._pick_lock:
            target = self._pick(job, now)
            if target == current or not self._alive(target):
                return None
            view = placement.assess(target, job, stale_s=self.stale_s, now=now)
            if not (view.pickable and view.starts_now and _nothing_queued(target)):
                return None
            source = SpoolPaths(current).pending / f"{job.name}.sh"
            destination = SpoolPaths(target).pending / f"{job.name}.sh"
            if destination.exists():
                return None
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.rename(source, destination)
            except OSError:
                return None
        return target

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
            "wall": int(spec.timeout_s) if spec.timeout_s and spec.timeout_s > 0 else None,
            "owner": self.name_prefix,
            "token": token,
            "tree": self.tree if self.tree and not spec.cwd else None,
        }
        script = spoolctl.render_job(spec.rendered, header=header, cwd=spec.cwd, env=spec.env)
        with self._pick_lock:
            root = self._pick(placement.JobNeeds.from_script(name, script), submitted)
            age = spoolctl.watcher_age(root, submitted)
            if age is None or age <= self.stale_s:
                root.mkdir(parents=True, exist_ok=True)
                spoolctl.queue_job(
                    root, name, script, owner=self.name_prefix, require_watcher=False
                )
        if age is not None and age > self.stale_s:
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
        job = placement.JobNeeds.from_script(name, script)
        where = {"root": root}

        def relocate(current: Path) -> Path | None:
            moved = self._relocate(job, current)
            if moved is not None:
                where["root"] = moved
            return moved

        try:
            record = spoolctl.wait_for_job(
                root,
                name,
                token=token,
                poll_s=self.poll_s,
                timeout_s=spec.timeout_s,
                stale_s=self.stale_s,
                grace_s=self.grace_s,
                queue_timeout_s=self.queue_timeout_s,
                unwatched_s=spec.timeout_s,
                submitted_at=submitted,
                relocate=relocate if self._elsewhere() else None,
            )
        except spoolctl.SpoolError as exc:
            root = where["root"]
            reason = str(exc) + self._withdraw(root, name)
            lost = exc.exit_code == spoolctl.EXIT_NO_SESSION
            return CommandResult(
                returncode=TIMEOUT_RETURNCODE,
                stdout=f"{self._log(root, name)}\n=== spool executor: {reason} ===\n",
                timed_out=True,
                infrastructure=reason if lost else "",
            )
        root = where["root"]
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


def _nothing_queued(root: Path) -> bool:
    """No job waits in ``root``'s queue on disk.

    The watcher reports its free devices on its next tick, so a burst of
    submissions would all read the same free devices and pile onto one
    session; a job already waiting in the queue is seen at once.
    """
    return not any(SpoolPaths(root).pending.glob("*.sh"))


def _width(gpus: int | None, node: int) -> int:
    """Devices a job asking for ``gpus`` (``None``: the whole node) holds on a node of ``node``."""
    return max(1, node if gpus is None else gpus)


def _fits(job: placement.JobNeeds, view: placement.SpoolView) -> int:
    """How many jobs like ``job`` the free devices of ``view``'s watcher hold."""
    return view.free // _width(job.gpus, view.devices) if view.devices else 0


def _fits_taken(job: placement.JobNeeds, queued: placement.JobNeeds, node: int) -> int:
    """How many fits of ``job`` the queued job ``queued`` takes on nodes of ``node`` devices."""
    if not queued.holds_gpus:
        return 0
    held, each = _width(queued.gpus, node), _width(job.gpus, node)
    return -(-held // each)


__all__ = ["SESSION_LOSS_REASONS", "SpoolCommandExecutor"]
