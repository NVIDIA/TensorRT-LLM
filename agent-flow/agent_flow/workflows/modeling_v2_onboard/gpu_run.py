# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run a command on GPUs from the run's shared pool.

    python -m agent_flow.workflows.modeling_v2_onboard.gpu_run --pool <dir> \
        --task <task id> [--gpus N] [--minutes M] <worktree> <command...>

Waits for N free GPUs on one holder job whose allocation has at least M more
minutes, locks them, and runs the command as a step of that job, inside the
pool's container, in ``<worktree>``, with ``CUDA_VISIBLE_DEVICES`` set to the
locked GPUs. The GPUs are free again the moment this process exits;
SIGTERM/SIGINT stop the step.

Waiting commands are served by their task's priority class, read from the
run's ``gpu_priorities.json`` on every poll: ``accuracy`` before ``feature``
before ``perf``, first come first served within a class. Each waiter ahead of
a command is placed first, node by node: on the fullest node it fits, or, when
none has room, it claims the node with the most free GPUs, to start there once
that node drains. A command takes the GPUs left over only, and ``perf`` work
never takes the last ``reserve`` free GPUs.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import getpass
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Waiter", "free_gpus", "main", "place", "tickets"]

RANKS = {"accuracy": 0, "feature": 1, "perf": 2}
DEFAULT_CLASS = "feature"
POLL_SECONDS = 15
# Margin kept between a command's expected end and its holder job's end.
MARGIN_MINUTES = 10


@dataclass(frozen=True)
class Waiter:
    """A command waiting for GPUs."""

    pid: int
    rank: int
    since: float
    gpus: int


def place(
    me: Waiter,
    waiters: list[Waiter],
    free: dict[str, int],
    *,
    perf: bool,
    reserve: int,
    usable: set[str] | None = None,
) -> str | None:
    """The node ``me`` may take its GPUs on now, or ``None`` if it waits.

    Args:
        me: The command asking.
        waiters: The other commands waiting.
        free: Free GPUs on each node of the pool.
        perf: ``me`` is ``perf`` work, which leaves ``reserve`` GPUs free.
        reserve: Free GPUs ``perf`` work may not take.
        usable: The nodes ``me`` may run on (enough time left); all if ``None``.
    """
    left = dict(free)
    ahead = [w for w in waiters if (w.rank, w.since) < (me.rank, me.since)]
    for waiter in sorted(ahead, key=lambda w: (w.rank, w.since)):
        if not left:
            break
        fits = [node for node, n in left.items() if n >= waiter.gpus]
        if fits:
            node = min(fits, key=lambda k: left[k])
            left[node] -= waiter.gpus
        else:
            # The node it will start on once that node drains: no one else takes its GPUs.
            left[max(left, key=lambda k: left[k])] = 0
    if perf and sum(left.values()) - me.gpus < reserve:
        return None
    fits = [node for node, n in left.items() if n >= me.gpus and (usable is None or node in usable)]
    return min(fits, key=lambda k: left[k]) if fits else None


def _task_class(priorities: Path, task: str) -> str:
    try:
        classes = json.loads(priorities.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        classes = {}
    klass = classes.get(task, DEFAULT_CLASS)
    return klass if klass in RANKS else DEFAULT_CLASS


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def tickets(queue: Path) -> list[dict]:
    """The tickets of the commands waiting for GPUs; dead waiters' are removed."""
    live = []
    for path in queue.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            Waiter(data["pid"], data["rank"], data["since"], data["gpus"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if _alive(data["pid"]):
            live.append(data)
        else:
            path.unlink(missing_ok=True)
    return live


def _waiters(queue: Path) -> list[Waiter]:
    return [Waiter(t["pid"], t["rank"], t["since"], t["gpus"]) for t in tickets(queue)]


def _seconds(left: str) -> int:
    """Seconds in a ``squeue %L`` time-left value."""
    if left in ("UNLIMITED", "NOT_SET", "INVALID"):
        return 10**9
    days, _, rest = left.rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return int(days or 0) * 86400 + hours * 3600 + minutes * 60 + seconds


def _holder_jobs(job_name: str) -> list[tuple[str, int]]:
    """Running holder jobs and their seconds left, longest first."""
    out = subprocess.run(
        ["squeue", "-h", "-u", getpass.getuser(), "-n", job_name, "-t", "R", "-o", "%i %L"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    jobs = [
        (line.split()[0], _seconds(line.split()[1])) for line in out.splitlines() if line.strip()
    ]
    return sorted(jobs, key=lambda job: -job[1])


def _lock(locks: Path, job: str, gpus_per_node: int, n: int) -> list[tuple[int, int]]:
    """Lock ``n`` free GPUs of ``job``: ``(gpu, lock fd)``, or none if fewer are free."""
    held: list[tuple[int, int]] = []
    for gpu in range(gpus_per_node):
        fd = os.open(locks / f"{job}.{gpu}", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        held.append((gpu, fd))
        if len(held) == n:
            return held
    for _, fd in held:
        os.close(fd)
    return []


def free_gpus(locks: Path, job: str, gpus_per_node: int) -> int:
    """How many GPUs of ``job`` no command holds."""
    count = 0
    for gpu in range(gpus_per_node):
        fd = os.open(locks / f"{job}.{gpu}", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            count += 1
        except BlockingIOError:
            pass
        finally:
            os.close(fd)
    return count


def _acquire(pool: dict, directory: Path, task: str, n: int, minutes: int):
    """Wait for this command's turn and lock ``n`` GPUs on one holder job."""
    locks, queue = directory / "locks", directory / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    locks.mkdir(parents=True, exist_ok=True)
    ticket = queue / f"{os.getpid()}.json"
    atexit.register(ticket.unlink, missing_ok=True)
    since, waited, last_note = time.time(), 0, -1
    priorities = Path(pool["priorities"])
    while True:
        # Re-read every poll: the planner may re-prioritize the task.
        klass = _task_class(priorities, task)
        me = Waiter(os.getpid(), RANKS[klass], since, n)
        ticket.write_text(json.dumps({**me.__dict__, "task": task, "class": klass}))
        guard = os.open(queue / ".lock", os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(guard, fcntl.LOCK_EX)
        try:
            jobs = _holder_jobs(pool["job_name"])
            free = {job: free_gpus(locks, job, pool["gpus_per_node"]) for job, _ in jobs}
            others = [w for w in _waiters(queue) if w.pid != me.pid]
            total = sum(free.values())
            usable = {job for job, left in jobs if left >= (minutes + MARGIN_MINUTES) * 60}
            node = place(
                me,
                others,
                free,
                perf=klass == "perf",
                reserve=pool["reserve"],
                usable=usable,
            )
            held = _lock(locks, node, pool["gpus_per_node"], n) if node else []
            if held:
                ticket.unlink(missing_ok=True)
                return node, held
        finally:
            fcntl.flock(guard, fcntl.LOCK_UN)
            os.close(guard)
        if waited // 600 != last_note:
            last_note = waited // 600
            print(
                f"[gpu_run] {task} ({klass}) waiting for {n} pool GPU(s): {total} free on "
                f"{len(jobs)} node(s), {len(others)} other waiter(s)",
                flush=True,
            )
        time.sleep(POLL_SECONDS)
        waited += POLL_SECONDS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pool", required=True, type=Path, help="the run's gpu_pool directory")
    parser.add_argument("--task", required=True, help="your task id (sets your priority)")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--minutes", type=int, default=60, help="how long the command may run")
    parser.add_argument("worktree", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    pool = json.loads((args.pool / "pool.json").read_text(encoding="utf-8"))
    if not args.command:
        parser.error("no command")
    if not 1 <= args.gpus <= pool["gpus_per_node"]:
        parser.error(f"--gpus must be between 1 and {pool['gpus_per_node']}")
    worktree = args.worktree.resolve()

    job, held = _acquire(pool, args.pool, args.task, args.gpus, args.minutes)
    gpus = ",".join(str(gpu) for gpu, _ in held)
    note = (
        f"pid={os.getpid()} task={args.task} start={time.strftime('%FT%T')} "
        f"cmd={' '.join(args.command)[:300]}\n"
    )
    for _, fd in held:
        os.ftruncate(fd, 0)
        os.write(fd, note.encode())
    print(f"[gpu_run] job {job} GPUs {gpus}", flush=True)
    mounts = ",".join(f"{path}:{path}" for path in pool["container_mounts"])
    inner = ["bash", "-c", 'cd "$1" && shift && exec "$@"', "_", str(worktree), *args.command]
    srun = [
        "srun",
        f"--jobid={job}",
        "--overlap",
        "-N1",
        "-n1",
        "--mpi=none",
        f"--container-image={pool['container_image']}",
        f"--container-mounts={mounts}",
        f"--container-workdir={worktree}",
        "--no-container-mount-home",
        "env",
        f"CUDA_VISIBLE_DEVICES={gpus}",
        *inner,
    ]
    proc = subprocess.Popen(srun)

    def stop(signum, _frame):
        proc.send_signal(signal.SIGTERM if signum == signal.SIGTERM else signal.SIGINT)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        return proc.wait()
    finally:
        for _, fd in held:
            os.close(fd)


if __name__ == "__main__":
    sys.exit(main())
