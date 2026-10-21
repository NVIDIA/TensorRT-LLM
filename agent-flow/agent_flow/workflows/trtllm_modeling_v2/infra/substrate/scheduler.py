#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The compute half of ``environment.kind: session_spool``: a scheduler that drains a spool.

A compute session is an allocation whose main process is this scheduler. Work
arrives as files on a shared filesystem, so anything that can write a file --
the orchestrator, an agent, an operator -- can run GPU work without a
scheduler client of its own.

This module is **standard-library only** and runs by path
(``python3 scheduler.py watch <spool>``), because the interpreter inside the
container is the engine's, not the orchestrator's.

PROTOCOL
========

::

    <spool>/pending/<name>.sh     a job, renamed into place (never written in place)
    <spool>/running/<name>.sh     claimed by a watcher (atomic rename; a job runs once)
    <spool>/running/<name>.json   who runs it: watcher, devices, pid, start time
    <spool>/logs/<name>.log       stdout+stderr, framed by "=== start" / "=== end" lines
    <spool>/done/<name>.json      the verdict: rc, reason, devices, times, owner, token
    <spool>/done/<name>.rc        the exit code, renamed into place LAST
    <spool>/cancel/<name>         ask the scheduler to stop a job (the sanctioned kill)
    <spool>/requeued/<name>.json  the runs of a job lost with their allocation, while it runs again
    <spool>/STOP                  finish running work, admit nothing, exit

A reader that sees ``done/<name>.rc`` may trust the log and the JSON beside it.

LOST ALLOCATIONS
================

A job in ``running/`` whose watcher stopped heart-beating ran on an
allocation that is gone: Slurm requeued it onto another node, or it ended or
failed. The scheduler that finds it (the requeued allocation's next start, or
any other watcher of the spool) runs it again: the script goes back to
``pending/`` with its submission time, so its place in the queue, and
``requeued/<name>.json`` records the run that was lost. The receipt of the
run that finishes lists every run, one per allocation, in ``attempts``. An
earlier start of this same allocation is known lost at once (its
``SLURM_RESTART_COUNT`` is lower); any other run once its watcher has been
silent ``orphan_after_s``.

A job is run again at most ``requeue`` times: its header's ``requeue=N``,
else the watcher's ``--requeue`` (1). ``requeue=0`` opts out. A job that
opted out, used its re-runs, or whose cancel was requested gets
``reason=node_lost`` (``RC_NODE_LOST``) instead. A job whose processes still
run on this host outlived only its watcher; it is closed out as ``orphaned``
(``RC_ORPHANED``) and not run again beside itself.

HAND-OVER
=========

::

    <spool>/handover/request.json           the ending allocation asks for a successor
    <spool>/handover/acks/<allocation>.json the job a driver submitted to follow <allocation>
    <spool>/handover/archive/               requests and acks no live allocation needs

A request names its allocation (``slurm_job``) and the watcher that wrote it,
and stands only while that watcher heart-beats. An ack is keyed by the
allocation it follows, so a later ack for the next allocation never replaces
it. A watcher that starts on a spool sets aside the request of the
allocation it succeeds and the acks that answered it.

A job declares what it needs in header comment lines anywhere in its first
lines::

    # spool: gpus=2 wall=2h priority=0 mem_gb=200 owner=r014 token=<uuid> requeue=1

``gpus`` is a device count or ``all`` (the whole node, exclusively). Without
it a job is classified by its name and body: a ``free-`` name holds nothing,
any other name holds the whole node, and a body that calls ``gpu_slot.sh N``
holds N devices.

A submitter also says when it queued a job, to the microsecond, in a line of
its own after the shebang, ``#spool-queued: <epoch seconds>``
(:func:`stamp_queued`). It is not a header key, so a watcher that does not
know it reads past it.

ADMISSION
=========

One priority queue per watcher, ordered by effective priority (``priority``
plus one per ``aging_s`` waited), then whole-node jobs before device-slot
jobs, then submission order: the pending file's modification time, and
within one second of it the time the job's ``#spool-queued:`` line states.
A shared filesystem may keep modification times to the second only, and
jobs queued in one second then still run in the order they were queued, not
by name. A GPU job that cannot start blocks every GPU job ranked below it,
so a waiting whole-node or multi-device job is never starved by a stream of
single-device jobs; jobs that hold no device always start.

A GPU job starts only on devices whose memory has come back. An engine sizes
its KV-cache and recurrent pools from the device memory it finds free, so one
started beside memory a cancelled job (or a rank or launcher child of it) has
not given back reserves less than the same engine elsewhere, or fails. When a
GPU job is about to start, one ``nvidia-smi`` reading of the free devices
hands out those with at most ``device_max_used_mib`` in use first, and the
job waits for any other it gets; a device still in use
``device_release_wait_s`` after a reading first found it so is quarantined
until a reading finds it released. Without ``nvidia-smi`` nothing waits.

LIMITS AND THE SAFE-KILL RULE
=============================

Every job runs in its own session with core dumps disabled and with the
allocation's MPI context removed from its environment (``PMI*``, ``PMIX*``,
``OMPI*``, ``SLURM*``, ``PRTE*``, ``PALS*``). A process that inherits that
context is part of the allocation's MPI job, and when it dies abnormally
Slurm's MPI error handler kills the **whole allocation**. Because the
scheduler removes the context, it may stop its own jobs: on a wall-clock limit,
a memory limit or a cancel request it sends SIGTERM and, ``kill_grace_s``
later, SIGKILL. **It never signals a process whose environment still carries
the allocation's context** (see :func:`is_mpi_attached`); such a process is
left alone, the job is marked failed, and its devices stay quarantined until
it exits by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import resource
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

try:
    from . import provenance as _provenance
    from . import rebuild as _rebuild
except ImportError:  # run by path: the sibling modules are on sys.path[0]
    import provenance as _provenance  # type: ignore[no-redef]
    import rebuild as _rebuild  # type: ignore[no-redef]

# --------------------------------------------------------------------------- protocol

#: Prefix of a job header line.
HEADER_TAG = "#spool:"

#: Prefix of the line that says when a job was queued (:func:`stamp_queued`).
QUEUED_TAG = "#spool-queued:"

#: Header lines are only looked for this far into a script.
HEADER_SCAN_LINES = 64

#: A job whose name starts with this and declares no ``gpus`` holds no device.
FREE_PREFIX = "free-"

#: The ``gpus`` value that asks for the whole node, exclusively.
ALL_GPUS = "all"

#: Names a job may have. A name is a file name on a shared filesystem.
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: errno values of a full filesystem or quota: transient for a long-lived watcher.
FULL_FILESYSTEM_ERRNOS = frozenset({28, 122})  # ENOSPC, EDQUOT

#: Where, under a spool, the hand-over files live.
HANDOVER_DIRNAME = "handover"

#: The single ack file of a flow before acks were keyed by allocation; still read.
LEGACY_ACK_FILENAME = "ack.json"
RC_WALL_LIMIT = 124
RC_STALLED = 210
RC_MEMORY_LIMIT = 211
RC_ORPHANED = 212
RC_REFUSED = 213
RC_HOST_MEMORY = 214
RC_LAUNCH_FAILED = 215
RC_CANCELLED = 216
RC_MIXED_CHECKOUT = 217
RC_NODE_LOST = 218

#: Verdict reasons of a job its session ended, not the job: nothing in them is a verdict.
SESSION_LOSS_REASONS = frozenset({"orphaned", "node_lost"})

#: How many times a job lost with its allocation runs again, unless its header says.
DEFAULT_REQUEUE = 1

#: Put first on every job's PYTHONPATH: records which checkout each process imported.
RESOLVE_HOOK_DIR = Path(__file__).resolve().parent / "resolvehook"

#: Modules the resolve hook watches unless a job says otherwise.
DEFAULT_RESOLVE_MODULES = "tensorrt_llm"

#: Verdict reasons after which the job has said nothing about what it ran.
TIMEOUT_REASONS = frozenset({"wall_limit", "stalled"})

#: Environment-variable prefixes that tie a process to the allocation's MPI job.
SCRUB_PATTERN = re.compile(r"^(PMI|PMIX|OMPI|SLURM|PRTE|PALS|MPI_LOCALRANKID|MPI_LOCALNRANKS)")

#: Log lines after which a job that stops making progress is presumed hung: a
#: rank that died while its peers wait for it forever in a collective.
DEFAULT_FAILURE_PATTERNS: tuple[str, ...] = (
    r"Traceback \(most recent call last\)",
    r"\bMPI_ABORT\b",
    r"Segmentation fault",
    r"terminate called after throwing",
    r"CUDA error: ",
    r"\bNCCL error\b",
)

#: How much new log a pass reads looking for failure lines.
_LOG_SCAN_BYTES = 1 << 20

#: A job is admitted near the allocation's end only if its wall limit leaves this much spare.
HANDOVER_MARGIN_S = 120.0

#: A job's expected run time (:func:`expected_s`): the spool's finished jobs like it
#: (same owner, wall limit and device ask), at least this many, give this multiple of
#: their 90th-percentile run time, never under the floor nor over the wall limit.
EXPECT_MIN_SAMPLES = 5
EXPECT_FACTOR = 2.0
EXPECT_FLOOR_S = 300.0

_EXPECT_CACHE: dict[str, tuple[tuple[int, int], dict[tuple[str, float, str], list[float]]]] = {}


def _finished_runs(done: Path) -> dict[tuple[str, float, str], list[float]]:
    """The run times of a spool's finished jobs, by (owner, wall limit, device ask).

    Read again only when the directory changed (its mtime or its count of
    records). A job stopped at its limit or
    lost with its allocation says nothing about how long such a job needs.
    """
    try:
        records = sorted(done.glob("*.json"))
        stamp = (done.stat().st_mtime_ns, len(records))
    except OSError:
        return {}
    cached = _EXPECT_CACHE.get(str(done))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    runs: dict[tuple[str, float, str], list[float]] = {}
    for path in records:
        record = read_json(path)
        if not record or record.get("timed_out") or record.get("reason") != "exit":
            continue
        try:
            key = (
                str(record.get("owner") or ""),
                float(record.get("wall_s") or 0),
                str(record.get("gpus", "")),
            )
            runs.setdefault(key, []).append(float(record["duration_s"]))
        except (KeyError, TypeError, ValueError):
            continue
    _EXPECT_CACHE[str(done)] = (stamp, runs)
    return runs


def expected_s(done: Path, owner: str, wall_s: float, gpus: str) -> float | None:
    """How long a job like this one runs, from the spool's history; ``None`` without enough of it.

    A gate's wall limit is its worst case (hours for an accuracy bundle) while
    such gates finish in minutes; judged by the limit, a session near its end
    takes none of them. Never more than ``wall_s``.
    """
    if not owner or wall_s <= 0:
        return None
    durations = sorted(_finished_runs(done).get((owner, float(wall_s), str(gpus)), ()))
    if len(durations) < EXPECT_MIN_SAMPLES:
        return None
    p90 = durations[int(0.9 * (len(durations) - 1))]
    return min(float(wall_s), max(EXPECT_FLOOR_S, EXPECT_FACTOR * p90))


#: After a memory-pressure victim is gone, how long to let the host reclaim its memory.
PRESSURE_SETTLE_S = 15.0

#: Device memory in use (MiB) above which a device still holds what a finished job left.
DEFAULT_DEVICE_MAX_USED_MIB = 1024.0

#: How long a device may hold memory no job of the watcher accounts for before it is quarantined.
DEFAULT_DEVICE_RELEASE_WAIT_S = 300.0

#: How often a quarantined device is read again while no job is waiting to start.
DEVICE_RECHECK_S = 60.0

_GPU_SLOT_CALL = re.compile(r"gpu_slot\.sh['\"]?\s+(\d+|all)\b")
_GB = 1024.0**3


def utc_iso(epoch: float | None = None) -> str:
    """``2026-10-11T20:35:41Z``: the spelling every spool file uses."""
    moment = datetime.fromtimestamp(time.time() if epoch is None else epoch, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_duration(text: str) -> float:
    """Seconds from ``7200``, ``90s``, ``45m``, ``2h``, ``1d`` or ``HH:MM:SS``. ``0`` is none."""
    value = str(text).strip().lower()
    if not value:
        raise ValueError("empty duration")
    if ":" in value:
        parts = [float(p) for p in value.split(":")]
        if len(parts) > 3 or any(p < 0 for p in parts):
            raise ValueError(f"not HH:MM:SS: {text!r}")
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        return seconds
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(value[-1])
    number = float(value[:-1] if scale else value)
    if number < 0 or math.isnan(number):
        raise ValueError(f"negative duration: {text!r}")
    return number * (scale or 1)


def _parse_gpus(text: str) -> int | None:
    if text.strip().lower() == ALL_GPUS:
        return None
    value = int(text)
    if value < 0:
        raise ValueError("must be >= 0 or 'all'")
    return value


def _parse_count(text: str) -> int:
    value = int(text)
    if value < 0:
        raise ValueError("must be >= 0")
    return value


def _parse_gb(text: str) -> float:
    value = float(text)
    if value < 0 or math.isnan(value):
        raise ValueError("must be >= 0")
    return value


def _parse_tree(text: str) -> str:
    if not text.startswith("/"):
        raise ValueError("must be an absolute path")
    return os.path.normpath(text)


def _parse_modules(text: str) -> str:
    if text != "off" and not re.fullmatch(r"[A-Za-z_][\w.]*(,[A-Za-z_][\w.]*)*", text):
        raise ValueError("must be 'off' or a comma list of module names")
    return text


def _parse_word(text: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._@:+/-]{1,200}", text):
        raise ValueError("must be one word of [A-Za-z0-9._@:+/-]")
    return text


#: Header key -> (JobSpec attribute, converter). Unknown keys refuse the job:
#: a misspelt limit that is silently ignored is a limit nobody has.
HEADER_KEYS: dict[str, tuple[str, Callable[[str], object]]] = {
    "gpus": ("gpus", _parse_gpus),
    "priority": ("priority", int),
    "wall": ("wall_s", parse_duration),
    "mem_gb": ("mem_gb", _parse_gb),
    "owner": ("owner", _parse_word),
    "token": ("token", _parse_word),
    "label": ("label", _parse_word),
    "tree": ("tree", _parse_tree),
    "resolve": ("resolve", _parse_modules),
    "stall": ("stall_s", parse_duration),
    "failure_stall": ("failure_stall_s", parse_duration),
    "host_mem_gb": ("host_mem_gb", _parse_gb),
    "shm_gb": ("shm_gb", _parse_gb),
    "requeue": ("requeue", _parse_count),
}


@dataclass
class JobSpec:
    """What a job asked for, read from its header."""

    name: str
    #: Device count, or ``None`` for the whole node.
    gpus: int | None = 0
    priority: int = 0
    #: Wall-clock limit from START, in seconds. ``None`` takes the class default; 0 is none.
    wall_s: float | None = None
    #: Resident-memory limit for the job's processes, in GiB. 0 is none.
    mem_gb: float = 0.0
    owner: str = ""
    token: str = ""
    label: str = ""
    #: The checkout the job runs against: its working directory and first PYTHONPATH entry.
    tree: str = ""
    #: Modules whose import origin is recorded; ``off`` disables the hook.
    resolve: str = ""
    #: Seconds without log growth after which the job is presumed hung. 0 is none.
    stall_s: float | None = None
    #: Seconds without log growth, after a failure line, after which the job is presumed hung.
    failure_stall_s: float | None = None
    #: Host memory (GiB) that must be available before the job starts.
    host_mem_gb: float | None = None
    #: Free ``/dev/shm`` (GiB) that must be available before the job starts.
    shm_gb: float = 0.0
    #: Times the job runs again when its allocation is lost under it. ``None`` takes the
    #: watcher's default; 0 never does.
    requeue: int | None = None
    #: When the job was submitted (the pending file's mtime).
    submitted: float = 0.0
    #: When the submitter queued it, to the microsecond (its ``#spool-queued:``
    #: line); 0 when the script does not say.
    queued: float = 0.0
    #: Header keys the job set explicitly.
    explicit: frozenset[str] = frozenset()
    #: Why the job cannot run at all; a non-empty list refuses it.
    errors: list[str] = field(default_factory=list)

    @property
    def whole(self) -> bool:
        """Whether the job holds the whole node exclusively."""
        return self.gpus is None

    @property
    def holds_gpus(self) -> bool:
        """Whether the job takes part in device accounting."""
        return self.gpus is None or self.gpus > 0

    def describe_gpus(self) -> str:
        return ALL_GPUS if self.gpus is None else str(self.gpus)


def parse_header(text: str) -> tuple[dict[str, str], list[str]]:
    """Return the ``#spool:`` key/value pairs of a script, and any syntax errors."""
    values: dict[str, str] = {}
    errors: list[str] = []
    for line in text.splitlines()[:HEADER_SCAN_LINES]:
        stripped = line.strip()
        if not stripped.startswith(HEADER_TAG):
            continue
        for token in stripped[len(HEADER_TAG) :].split():
            key, sep, value = token.partition("=")
            if not sep or not key:
                errors.append(f"header token {token!r} is not key=value")
                continue
            values[key] = value
    return values, errors


def _is_queued_line(line: str) -> bool:
    return line.strip().startswith(QUEUED_TAG)


def stamp_queued(script: str, now: float | None = None) -> str:
    """``script`` with the line that says it was queued ``now``, after its shebang.

    A line an earlier submission of the same script left is replaced.
    """
    now = time.time() if now is None else now
    lines = [line for line in script.splitlines(keepends=True) if not _is_queued_line(line)]
    head = 1 if lines and lines[0].startswith("#!") else 0
    lines.insert(head, f"{QUEUED_TAG} {now:.6f}\n")
    return "".join(lines)


def queued_at(text: str) -> float:
    """When ``text``'s ``#spool-queued:`` line says the job was queued; 0 without one."""
    for line in text.splitlines()[:HEADER_SCAN_LINES]:
        if _is_queued_line(line):
            try:
                value = float(line.strip()[len(QUEUED_TAG) :].strip())
            except ValueError:
                return 0.0
            return value if math.isfinite(value) and value > 0 else 0.0
    return 0.0


def job_spec_from_script(name: str, text: str, submitted: float = 0.0) -> JobSpec:
    """Classify a job from its name, header and body."""
    values, errors = parse_header(text)
    spec = JobSpec(
        name=name, submitted=submitted, queued=queued_at(text), explicit=frozenset(values)
    )
    if not NAME_PATTERN.match(name):
        errors.append(f"job name {name!r} must match {NAME_PATTERN.pattern}")
    for key, raw in values.items():
        entry = HEADER_KEYS.get(key)
        if entry is None:
            errors.append(f"unknown header key {key!r}; known: {', '.join(sorted(HEADER_KEYS))}")
            continue
        attr, convert = entry
        try:
            setattr(spec, attr, convert(raw))
        except ValueError as exc:
            errors.append(f"{key}={raw!r}: {exc}")
    if "gpus" not in values:
        body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        match = _GPU_SLOT_CALL.search(body)
        if match:
            spec.gpus = _parse_gpus(match.group(1))
        elif name.startswith(FREE_PREFIX):
            spec.gpus = 0
        else:
            spec.gpus = None
    spec.errors = errors
    return spec


def scrub_environment(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` without the variables that attach a process to the allocation's MPI job."""
    return {k: v for k, v in env.items() if not SCRUB_PATTERN.match(k)}


def is_mpi_attached(environ: Mapping[str, str]) -> bool:
    """Whether a process carries the allocation's Slurm/PMI context: the safe-kill rule.

    True for any ``SLURM_*`` variable (the step's environment), ``PMI_FD``
    (Slurm's PMI-2 wire-up) or a ``PMIX_NAMESPACE`` Slurm created. An
    ``mpirun`` started inside a scrubbed job has its own PMIx namespace and is
    not attached; signalling it or its ranks cannot reach the allocation.
    """
    for key, value in environ.items():
        if key.startswith("SLURM_") or key == "PMI_FD":
            return True
        if key == "PMIX_NAMESPACE" and value.startswith("slurm"):
            return True
    return False


def restart_count(environ: Mapping[str, str]) -> int | None:
    """Times Slurm has requeued this allocation (``SLURM_RESTART_COUNT``); ``None`` outside Slurm.

    Slurm sets it only once a job has been requeued, so a job that has an id
    and no count is on its first start.
    """
    value = environ.get("SLURM_RESTART_COUNT", "").strip()
    if value.isdigit():
        return int(value)
    return 0 if environ.get("SLURM_JOB_ID") else None


# --------------------------------------------------------------------------- /proc


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    ppid: int
    session: int
    state: str


def read_proc_table(proc_root: Path = Path("/proc")) -> dict[int, ProcInfo]:
    """Every visible process's parent, session and state."""
    table: dict[int, ProcInfo] = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return table
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # `pid (comm) state ppid pgrp session ...`; comm may hold spaces and parens.
        tail = raw[raw.rfind(")") + 2 :].split()
        if len(tail) < 4:
            continue
        table[int(entry.name)] = ProcInfo(
            pid=int(entry.name), ppid=int(tail[1]), session=int(tail[3]), state=tail[0]
        )
    return table


def read_environ(pid: int, proc_root: Path = Path("/proc")) -> dict[str, str] | None:
    """A process's initial environment, or ``None`` when it cannot be read."""
    try:
        raw = (proc_root / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    env: dict[str, str] = {}
    for item in raw.split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep:
            env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env


def rss_bytes(pid: int, proc_root: Path = Path("/proc")) -> int:
    """Resident set size of one process; 0 when it is gone."""
    try:
        fields = (proc_root / str(pid) / "statm").read_text(encoding="utf-8").split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return 0


def job_processes(leader: int, table: Mapping[int, ProcInfo]) -> list[int]:
    """Live processes that belong to a job: its session, and every descendant of its leader."""
    children: dict[int, list[int]] = {}
    for info in table.values():
        children.setdefault(info.ppid, []).append(info.pid)
    found = {info.pid for info in table.values() if info.session == leader}
    stack = [leader]
    while stack:
        pid = stack.pop()
        if pid in table:
            found.add(pid)
        for child in children.get(pid, ()):
            if child not in found:
                stack.append(child)
    found.discard(os.getpid())
    return sorted(pid for pid in found if pid in table and table[pid].state != "Z")


def processes_with_marker(
    table: Mapping[int, ProcInfo],
    spool: str,
    job: str,
    watcher: str,
    proc_root: Path = Path("/proc"),
) -> list[int]:
    """Live processes whose environment names this job of this watcher on this spool.

    A process that calls ``setsid`` and is re-parented to init once the job's
    leader exits (``nsys profile`` and the application it launches do exactly
    that) is neither in the leader's session nor its descendant any more, so
    :func:`job_processes` stops seeing it. Its environment still carries the
    markers every job is started with, and that is what this finds. Job names
    and watcher ids are unique only within a spool, so the spool is part of
    the match: without it a namesake job of another spool on this host matches.
    """
    want = {
        f"MV2FLOW_SPOOL={spool}".encode(),
        f"MV2FLOW_SPOOL_JOB={job}".encode(),
        f"MV2FLOW_SPOOL_WATCHER={watcher}".encode(),
    }
    found = []
    for pid, info in table.items():
        if info.state == "Z" or pid == os.getpid():
            continue
        try:
            parts = set((proc_root / str(pid) / "environ").read_bytes().split(b"\0"))
        except OSError:
            continue
        if want <= parts:
            found.append(pid)
    return sorted(found)


#: ``prctl`` option that reads whether the calling process is a child subreaper.
_PR_GET_CHILD_SUBREAPER = 37


def adopts_orphans() -> bool:
    """Whether orphans are re-parented to this process: PID 1 of its namespace, or a subreaper."""
    if os.getpid() == 1:
        return True
    try:
        import ctypes

        flag = ctypes.c_int(0)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(flag), 0, 0, 0) != 0:
            return False
    except (ImportError, OSError, AttributeError):
        return False
    return flag.value != 0


def host_memory_available_gb(
    meminfo: Path = Path("/proc/meminfo"), cgroup_root: Path = Path("/sys/fs/cgroup")
) -> float | None:
    """Memory a new job can still use, in GiB: the smaller of the host's and the cgroup's."""
    available: float | None = None
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024 / _GB
                break
    except (OSError, ValueError, IndexError):
        available = None
    for limit_file, usage_file in (
        ("memory.max", "memory.current"),
        ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes"),
    ):
        try:
            limit_text = (cgroup_root / limit_file).read_text(encoding="utf-8").strip()
            usage = int((cgroup_root / usage_file).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if limit_text == "max" or not limit_text.isdigit() or int(limit_text) >= 1 << 60:
            break
        headroom = (int(limit_text) - usage) / _GB
        available = headroom if available is None else min(available, headroom)
        break
    return available


def shm_available_gb(path: str = "/dev/shm") -> float | None:
    """Free space in ``/dev/shm``, in GiB."""
    try:
        stat = os.statvfs(path)
    except OSError:
        return None
    return stat.f_bavail * stat.f_frsize / _GB


def read_device_memory_mib(timeout_s: float = 30.0) -> dict[str, int] | None:
    """Memory in use on every device of the node, in MiB, keyed by index and by UUID.

    ``nvidia-smi`` ignores ``CUDA_VISIBLE_DEVICES`` and reports the whole node,
    so a watcher's device id, which is a ``CUDA_VISIBLE_DEVICES`` entry, is
    found under whichever of the two it spells. ``None`` when ``nvidia-smi``
    is missing or reads no device.
    """
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    used: dict[str, int] = {}
    for line in done.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 3 and fields[2].isdigit():
            used[fields[0]] = used[fields[1]] = int(fields[2])
    return used or None


# --------------------------------------------------------------------------- files


def write_text_atomic(path: Path, text: str) -> None:
    """Write beside ``path`` and rename into place, so no reader sees a partial file."""
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def write_json_atomic(path: Path, data: Mapping[str, object]) -> None:
    write_text_atomic(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class SpoolPaths:
    """Where each part of the protocol lives under one spool."""

    SUBDIRS = ("pending", "running", "done", "logs", "history", "watchers", "cancel", "requeued")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.pending = self.root / "pending"
        self.running = self.root / "running"
        self.done = self.root / "done"
        self.logs = self.root / "logs"
        self.history = self.root / "history"
        self.watchers = self.root / "watchers"
        self.cancel = self.root / "cancel"
        self.requeued = self.root / "requeued"
        self.heartbeat = self.root / "watcher.heartbeat"
        self.jobid = self.root / "watcher.jobid"
        self.status = self.root / "watcher.status"
        self.snapshot = self.root / "status.json"
        self.stop = self.root / "STOP"
        self.handover = self.root / HANDOVER_DIRNAME
        self.handover_request = self.handover / "request.json"
        self.acks = self.handover / "acks"
        self.handover_archive = self.handover / "archive"

    def ensure(self) -> None:
        for name in self.SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def log(self, name: str) -> Path:
        return self.logs / f"{name}.log"

    def receipt(self, name: str) -> Path:
        return self.done / f"{name}.rc"

    def record(self, name: str) -> Path:
        return self.done / f"{name}.json"

    def requeue_record(self, name: str) -> Path:
        """The runs of ``name`` lost with their allocation, kept while it runs again."""
        return self.requeued / f"{name}.json"

    def earlier_runs(self, name: str, token: str) -> dict | None:
        """``name``'s requeue record when it is this submission's (its ``token``), else ``None``."""
        record = read_json(self.requeue_record(name))
        if record is None or str(record.get("token") or "") != (token or ""):
            return None
        return record

    def result_files(self, name: str) -> list[Path]:
        """Every file a previous run of ``name`` left behind."""
        return [
            self.done / f"{name}.rc",
            self.done / f"{name}.json",
            self.done / f"{name}.sh",
            self.logs / f"{name}.log",
            self.resolve_log(name),
        ]

    def resolve_log(self, name: str) -> Path:
        return self.logs / f"{name}.resolve.jsonl"

    def archive(self, name: str, stamp: str | None = None) -> list[Path]:
        """Move a previous run's results under ``history/`` instead of overwriting them."""
        stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        moved = []
        for path in self.result_files(name):
            if path.exists():
                target = self.history / f"{path.stem}.{stamp}{path.suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, target)
                moved.append(target)
        return moved

    def live_watchers(self, stale_s: float, now: float | None = None) -> list[dict]:
        """Status records of every watcher that heart-beat within ``stale_s``."""
        now = time.time() if now is None else now
        out = []
        for path in sorted(self.watchers.glob("*.json")):
            record = read_json(path)
            if record and now - float(record.get("alive_epoch", 0)) <= stale_s:
                out.append(record)
        return out

    def ack_path(self, allocation: str, job_id: str = "") -> Path:
        """Where the ack of a successor to ``allocation`` lives (by its job, if it names none)."""
        key = allocation if allocation not in ("", "unknown") else f"job-{job_id or 'unknown'}"
        return self.acks / f"{_FILE_KEY.sub('_', key)}.json"

    def read_acks(self) -> list[tuple[Path, dict]]:
        """Every ack on this spool, one per allocation it follows, and a legacy ``ack.json``."""
        paths = [*sorted(self.acks.glob("*.json")), self.handover / LEGACY_ACK_FILENAME]
        return [(path, record) for path, record in ((p, read_json(p)) for p in paths) if record]

    def archive_handover(self, path: Path, label: str, *, why: str, by: str) -> Path | None:
        """Move a hand-over file under ``handover/archive/``, noting why; ``None`` if it is gone.

        The rename is the claim: of two readers setting aside one file, one
        moves it and the other finds it gone.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = self.handover_archive / f"{_FILE_KEY.sub('_', label)}.{stamp}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(path, target)
        except FileNotFoundError:
            return None
        record = read_json(target)
        if record is not None:
            record["archived"] = {"at": utc_iso(), "by": by, "why": why, "was": path.name}
            write_json_atomic(target, record)
        return target


#: Characters a hand-over file name keeps; anything else becomes ``_``.
_FILE_KEY = re.compile(r"[^A-Za-z0-9._-]")


def answering_ack(
    acks: Iterable[tuple[Path, Mapping[str, object]]], request: Mapping[str, object] | None
) -> Mapping[str, object] | None:
    """The ack among ``acks`` that answers ``request`` (see :func:`ack_answers`), or ``None``."""
    return next((ack for _, ack in acks if ack_answers(ack, request)), None)


def following_ack(
    acks: Iterable[tuple[Path, Mapping[str, object]]], allocation: str, session: str = ""
) -> Mapping[str, object] | None:
    """The ack among ``acks`` of a successor to ``allocation`` (:func:`ack_follows`), or ``None``."""
    return next((ack for _, ack in acks if ack_follows(ack, allocation, session)), None)


def _ends_after(record: Mapping[str, Any], end: Any) -> bool:
    """Whether a watcher's allocation outlasts ``end``; an unknown end on either side does."""
    own = record.get("ends_at_epoch")
    if own is None or end is None:
        return True
    try:
        return float(own) > float(end)
    except (TypeError, ValueError):
        return True


def stale_request(request: Mapping[str, object], live: Iterable[Mapping[str, object]]) -> str:
    """Why ``request`` no longer asks for anything, or ``""`` while it does.

    A request stands while the watcher that wrote it heart-beats (``live`` are
    the spool's live watcher records) and no successor to it is up. One whose
    writer is gone is an ended allocation's, or an earlier start of a
    requeued one, which asks again when it needs to.
    """
    live = list(live)
    job = str(request.get("slurm_job") or "")
    writer = str(request.get("watcher") or "")
    jobs = {str(w.get("slurm_job") or "") for w in live}
    if writer and writer not in {str(w.get("watcher") or "") for w in live}:
        if job not in ("", "unknown") and job in jobs:
            return (
                f"watcher {writer}, which wrote it, stopped heart-beating; allocation {job} "
                f"runs on under a new watcher (requeued or restarted), which asks when it needs to"
            )
        return f"allocation {job or '?'} has no live watcher: it ended"
    if not writer and (job in ("", "unknown") or job not in jobs):
        return f"allocation {job or '?'} has no live watcher: it ended"
    for peer in live:
        mine = peer.get("watcher") if writer else peer.get("slurm_job")
        own = str(mine or "") == (writer or job)
        if not own and _ends_after(peer, request.get("ends_at_epoch")):
            return f"its successor {peer.get('watcher')} (allocation {peer.get('slurm_job')}) is up"
    return ""


def ack_follows(ack: Mapping[str, object] | None, allocation: str, session: str = "") -> bool:
    """Whether ``ack`` names a successor to ``allocation``, the allocation of ``session``.

    ``<spool>/handover/acks/<allocation>.json`` records the job submitted to
    follow an allocation (``succeeds``), so it holds whether it was written
    before or after that allocation's request. An allocation outside Slurm
    (``unknown``) names no one allocation; an ack for ``allocation``'s own
    job is the ack of the hand-over it came from, not one that follows it;
    and an ack that names another session follows none of this one's.
    """
    if not ack or allocation in ("", "unknown") or str(ack.get("job_id") or "") == allocation:
        return False
    named = str(ack.get("session") or "")
    if named and session and named != session:
        return False
    return str(ack.get("succeeds") or "") == allocation


def ack_answers(ack: Mapping[str, object] | None, request: Mapping[str, object] | None) -> bool:
    """Whether ``ack`` answers ``request``, both in one spool's ``handover/``.

    It does when it was written for that request (its ``request_id``), or
    when it already names a successor to the allocation the request is for
    (:func:`ack_follows`): a driver may submit the successor before the
    scheduler asks for one.
    """
    if not ack or not request:
        return False
    request_id = request.get("request_id")
    if request_id and ack.get("request_id") == request_id:
        return True
    return ack_follows(ack, str(request.get("slurm_job") or ""), str(request.get("session") or ""))


# --------------------------------------------------------------------------- scheduler


@dataclass
class SchedulerConfig:
    """How one watcher runs its jobs."""

    #: Device ids this watcher hands out, as ``CUDA_VISIBLE_DEVICES`` spells them.
    devices: tuple[str, ...] = ()
    poll_s: float = 3.0
    #: Wall limit for a GPU job that declares none. Device-free jobs default to none.
    default_wall_s: float = 7200.0
    #: Seconds of waiting that raise a pending job's priority by one. 0 disables aging.
    aging_s: float = 3600.0
    #: SIGTERM to SIGKILL.
    kill_grace_s: float = 60.0
    #: How long processes a finished job left behind may outlive it.
    leftover_grace_s: float = 10.0
    #: A running job whose watcher has not heart-beat for this long was lost with it.
    orphan_after_s: float = 180.0
    #: Times a job lost with its allocation runs again when its header does not say.
    requeue: int = DEFAULT_REQUEUE
    #: Below this much available host memory, the largest job is stopped. 0 disables it.
    pressure_floor_gb: float = 0.0
    #: The session this watcher belongs to, for status and successor requests.
    session: str = ""
    #: When the allocation ends (epoch seconds); ``None`` when unknown.
    ends_at: float | None = None
    #: How long before ``ends_at`` to request a successor and stop admitting long jobs.
    handover_lead_s: float = 3600.0
    #: The rendered entry script, named in the successor request.
    entry: str = ""
    #: Default for ``stall``: log-idle seconds before any job is presumed hung. 0 is off.
    stall_s: float = 0.0
    #: Default for ``failure_stall`` on GPU jobs: log-idle seconds after a failure line.
    failure_stall_s: float = 600.0
    failure_patterns: tuple[str, ...] = DEFAULT_FAILURE_PATTERNS
    #: Host memory a whole-node job needs before it starts, unless it says otherwise. 0 is off.
    engine_host_mem_gb: float = 0.0
    #: How long a job may wait for host memory before it is refused.
    host_mem_wait_s: float = 900.0
    #: A GPU job starts on a device only while at most this much of its memory (MiB) is in use.
    #: 0 is off.
    device_max_used_mib: float = DEFAULT_DEVICE_MAX_USED_MIB
    #: How long a free device may stay over that before it is quarantined.
    device_release_wait_s: float = DEFAULT_DEVICE_RELEASE_WAIT_S


@dataclass
class Hold:
    """Why a job that could otherwise start must wait."""

    reason: str
    #: Whether jobs ranked below it must wait too (it is waiting for a resource they share).
    blocks_queue: bool = True


@dataclass
class Verdict:
    rc: int
    reason: str
    detail: str = ""


@dataclass
class RunningJob:
    """A job this watcher started, from launch until its last process is gone."""

    spec: JobSpec
    proc: subprocess.Popen
    devices: tuple[str, ...]
    started: float
    log_path: Path
    verdict: Verdict | None = None
    exited_at: float | None = None
    receipt_written: bool = False
    term_sent_at: float | None = None
    kill_sent_at: float | None = None
    unsafe_pids: tuple[int, ...] = ()
    rss_gb: float = 0.0
    peak_rss_gb: float = 0.0
    pids: tuple[int, ...] = ()
    log_size: int = 0
    last_progress: float = 0.0
    failure_line: str = ""
    #: Every pid this scheduler has sent a signal to.
    signalled: set[int] = field(default_factory=set)
    #: The earlier runs of this job that were lost with their allocation, oldest first.
    attempts: list[dict] = field(default_factory=list)

    @property
    def wall_s(self) -> float:
        return float(self.spec.wall_s or 0.0)


def _attempt_of(record: Mapping[str, Any] | None, token: str) -> dict:
    """``{"attempt": n}`` for a job with earlier runs in ``record`` (its requeue record), else ``{}``."""
    if record is None or str(record.get("token") or "") != (token or ""):
        return {}
    attempts = record.get("attempts") or []
    return {"attempt": len(attempts) + 1} if attempts else {}


class Scheduler:
    """One watcher over one spool: admission, device slots, limits and receipts.

    Args:
        spool: The spool directory.
        config: Limits and device pool.
        clock: Wall-clock source; injectable so limits can be tested without waiting.
        proc_root: ``/proc``, injectable for tests.
        environ: The environment jobs inherit before scrubbing.
        watcher_id: This watcher's identity in the spool; ``<host>-<pid>`` by default.
    """

    def __init__(
        self,
        spool: str | Path,
        config: SchedulerConfig | None = None,
        *,
        clock: Callable[[], float] = time.time,
        proc_root: Path = Path("/proc"),
        environ: Mapping[str, str] | None = None,
        watcher_id: str | None = None,
    ) -> None:
        self.paths = SpoolPaths(spool)
        self.config = config or SchedulerConfig()
        self.clock = clock
        self.proc_root = proc_root
        self.environ = dict(os.environ if environ is None else environ)
        self.host = socket.gethostname()
        self.slurm_job = self.environ.get("SLURM_JOB_ID", "unknown")
        #: Which start of a requeued allocation this is; ``None`` outside Slurm.
        self.restart = restart_count(self.environ)
        self.watcher_id = watcher_id or f"{self.host}-{os.getpid()}"
        self.started = self.clock()
        self.jobs: dict[str, RunningJob] = {}
        self.pending_reasons: dict[str, str] = {}
        self.counters: dict[str, int] = {}
        self._spec_cache: dict[str, tuple[float, JobSpec]] = {}
        self._last_pending: list[JobSpec] = []
        self._pressure_settle_until = 0.0
        self.handover: dict = {}
        #: The successor request this watcher wrote, kept to write it again if it goes.
        self._request: dict = {}
        #: Whether this watcher has set aside its predecessor's hand-over files.
        self._taken_over = False
        self._memory_wait_since: dict[str, float] = {}
        self._memory_committed_gb = 0.0
        #: Free device -> when a reading first found it over ``device_max_used_mib``.
        self._device_held_since: dict[str, float] = {}
        #: Device -> ``used_mib`` and ``since`` (first found over) for memory never released.
        self._device_quarantine: dict[str, dict] = {}
        #: This pass's device reading, taken at most once a pass (see _read_device_memory).
        self._device_reading: Mapping[str, int] | None = None
        self._device_read = False
        self._device_read_epoch = 0.0
        #: This pass's native locations by job tree (see _admission_hold).
        self._native_seen: dict[str, list[str]] = {}
        self._failure_re = (
            re.compile("|".join(self.config.failure_patterns))
            if self.config.failure_patterns
            else None
        )
        self._adopts_orphans = adopts_orphans()
        self.paths.ensure()

    # ------------------------------------------------------------- lifecycle

    def announce(self, banner: Iterable[str] = ()) -> None:
        """Write the startup banner and the first heartbeat."""
        lines = [
            f"host={self.host}",
            f"pid={os.getpid()}",
            f"watcher={self.watcher_id}",
            f"started={utc_iso(self.started)}",
            f"spool={self.paths.root}",
            f"session={self.config.session or '-'}",
            f"devices={','.join(self.config.devices) or '-'}",
            f"CUDA_VISIBLE_DEVICES={self.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
            *banner,
            f"watcher: ready at {utc_iso()}",
        ]
        write_text_atomic(self.paths.status, "\n".join(lines) + "\n")
        self.heartbeat()

    def stopping(self) -> bool:
        return self.paths.stop.exists()

    def serve(self) -> int:
        """Run until ``STOP`` is present and every job this watcher started is gone."""
        while True:
            try:
                self.tick()
            except OSError as exc:
                if exc.errno not in FULL_FILESYSTEM_ERRNOS:
                    raise
                # A full filesystem (a shared project quota other users fill) fails
                # our small writes for minutes at a time. The jobs keep running;
                # dying here would orphan every one of them and end the session.
                print(
                    f"watcher: {exc.strerror} at {utc_iso()}; retrying next tick",
                    file=sys.stderr,
                    flush=True,
                )
            if self.stopping() and not self.jobs:
                self._append_status(f"watcher: STOP honoured, exiting at {utc_iso()}")
                self._retire()
                return 0
            if self.handover.get("successor") and not self.jobs:
                self._append_status(
                    f"watcher: handed over to {self.handover['successor']}, exiting at {utc_iso()}"
                )
                self._retire()
                return 0
            time.sleep(self.config.poll_s)

    def _retire(self) -> None:
        for path in (self.paths.watchers / f"{self.watcher_id}.json",):
            path.unlink(missing_ok=True)
        if not self.paths.live_watchers(self.config.orphan_after_s, self.clock()):
            self.paths.heartbeat.unlink(missing_ok=True)
            self.paths.jobid.unlink(missing_ok=True)

    def _append_status(self, line: str) -> None:
        with self.paths.status.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    # ------------------------------------------------------------------ tick

    def tick(self) -> None:
        """One scheduling pass: reap, enforce limits, admit, report."""
        now = self.clock()
        table = read_proc_table(self.proc_root)
        self._reap_zombie_children(table)
        for job in list(self.jobs.values()):
            self._advance(job, now, table)
        self._enforce_memory_pressure(now, table)
        self._recover_orphans(now, table)
        if not self._taken_over:
            self._take_over_handover(now)
            self._taken_over = True
        self._plan_handover(now)
        if not self.stopping():
            self._admit(now)
        else:
            self.pending_reasons = {spec.name: "watcher is stopping" for spec in self._last_pending}
        self.heartbeat()

    def heartbeat(self) -> None:
        """Write the liveness files and this watcher's status snapshot."""
        now = self.clock()
        write_text_atomic(self.paths.heartbeat, f"alive={utc_iso(now)}\n")
        write_text_atomic(self.paths.jobid, f"{self.slurm_job}\n")
        snapshot = self.snapshot(now)
        write_json_atomic(self.paths.watchers / f"{self.watcher_id}.json", snapshot)
        write_json_atomic(self.paths.snapshot, snapshot)

    def snapshot(self, now: float) -> dict:
        """What this watcher is doing, for status readers and the watchdog."""
        allocated = self._allocated()
        quarantined = self._quarantined()
        requeued = self._requeued()
        return {
            "watcher": self.watcher_id,
            "host": self.host,
            "pid": os.getpid(),
            "slurm_job": self.slurm_job,
            "restart": self.restart,
            "session": self.config.session,
            "spool": str(self.paths.root),
            "started": utc_iso(self.started),
            "alive": utc_iso(now),
            "alive_epoch": now,
            "stopping": self.stopping(),
            "ends_at": utc_iso(self.config.ends_at) if self.config.ends_at else None,
            "default_wall_s": self.config.default_wall_s,
            "ends_at_epoch": self.config.ends_at,
            "remaining_s": round(self.config.ends_at - now, 1) if self.config.ends_at else None,
            "handover": dict(self.handover) or None,
            "devices": list(self.config.devices),
            "free": [d for d in self.config.devices if d not in allocated | quarantined],
            "quarantined": sorted(quarantined),
            "quarantined_memory": {
                device: {"used_mib": entry["used_mib"], "since": utc_iso(entry["since"])}
                for device, entry in sorted(self._device_quarantine.items())
            },
            "running": [self._describe_running(job, now) for job in self.jobs.values()],
            "pending": [
                {
                    "name": spec.name,
                    "gpus": spec.describe_gpus(),
                    "priority": spec.priority,
                    "effective_priority": self._effective_priority(spec, now),
                    "waiting_s": round(now - spec.submitted, 1),
                    "owner": spec.owner,
                    "reason": self.pending_reasons.get(spec.name, ""),
                    **_attempt_of(requeued.get(spec.name), spec.token),
                }
                for spec in self._last_pending
            ],
            "counters": dict(self.counters),
        }

    def _requeued(self) -> dict[str, dict]:
        """Every requeue record on this spool, by job name."""
        records = {}
        for path in sorted(self.paths.requeued.glob("*.json")):
            record = read_json(path)
            if record is not None:
                records[path.name[: -len(".json")]] = record
        return records

    def _describe_running(self, job: RunningJob, now: float) -> dict:
        state = "running"
        if job.verdict is not None:
            state = "quarantined" if job.unsafe_pids else "finishing"
        return {
            "name": job.spec.name,
            "state": state,
            "gpus": job.spec.describe_gpus(),
            "devices": list(job.devices),
            "pid": job.proc.pid,
            "started": utc_iso(job.started),
            "elapsed_s": round(now - job.started, 1),
            "wall_s": job.wall_s,
            "rss_gb": round(job.rss_gb, 2),
            "peak_rss_gb": round(job.peak_rss_gb, 2),
            "owner": job.spec.owner,
            "verdict": None if job.verdict is None else job.verdict.reason,
            "idle_s": round(now - max(job.last_progress, job.started), 1),
            "failure_line": job.failure_line,
            "unsafe_pids": list(job.unsafe_pids),
            **({"attempt": len(job.attempts) + 1} if job.attempts else {}),
        }

    # ------------------------------------------------------------- admission

    def _effective_priority(self, spec: JobSpec, now: float) -> int:
        aging = self.config.aging_s
        bonus = int(max(0.0, now - spec.submitted) // aging) if aging > 0 else 0
        return spec.priority + bonus

    def _rank_key(self, spec: JobSpec, now: float) -> tuple:
        return (
            -self._effective_priority(spec, now),
            0 if spec.whole else 1,
            spec.submitted,
            spec.queued or spec.submitted,
            spec.name,
        )

    def _allocated(self) -> set[str]:
        return {d for job in self.jobs.values() for d in job.devices}

    def _quarantined(self) -> set[str]:
        held = {d for job in self.jobs.values() if job.unsafe_pids for d in job.devices}
        return held | set(self._device_quarantine)

    def _free_devices(self) -> list[str]:
        """Devices no job of this watcher holds and no quarantine keeps, in the order handed out."""
        taken = self._allocated() | set(self._device_quarantine)
        return [d for d in self.config.devices if d not in taken]

    def _whole_running(self) -> bool:
        return any(job.spec.whole for job in self.jobs.values())

    def _scan_pending(self) -> list[JobSpec]:
        specs = []
        seen = set()
        for path in sorted(self.paths.pending.glob("*.sh")):
            name = path.name[: -len(".sh")]
            seen.add(name)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            cached = self._spec_cache.get(name)
            if cached and cached[0] == mtime:
                specs.append(cached[1])
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            spec = self._classify(job_spec_from_script(name, text, submitted=mtime))
            self._spec_cache[name] = (mtime, spec)
            specs.append(spec)
        for stale in set(self._spec_cache) - seen:
            del self._spec_cache[stale]
        return specs

    def _classify(self, spec: JobSpec) -> JobSpec:
        """Apply this watcher's defaults and refuse what it can never run."""
        if spec.wall_s is None:
            spec.wall_s = self.config.default_wall_s if spec.holds_gpus else 0.0
        if spec.stall_s is None:
            spec.stall_s = self.config.stall_s
        if spec.failure_stall_s is None:
            spec.failure_stall_s = self.config.failure_stall_s if spec.holds_gpus else 0.0
        if spec.host_mem_gb is None:
            spec.host_mem_gb = self.config.engine_host_mem_gb if spec.whole else 0.0
        if spec.tree and not Path(spec.tree).is_dir():
            spec.errors.append(f"tree {spec.tree} is not a directory this watcher can see")
        if spec.gpus is not None and spec.gpus > len(self.config.devices):
            spec.errors.append(
                f"asks for {spec.gpus} GPU(s); this session hands out "
                f"{len(self.config.devices)} ({','.join(self.config.devices) or 'none'})"
            )
        return spec

    def _admit(self, now: float) -> None:
        pending = self._scan_pending()
        self._last_pending = []
        self.pending_reasons = {}
        self._memory_committed_gb = 0.0
        self._device_reading, self._device_read = None, False
        self._native_seen = {}
        if self._device_quarantine and now - self._device_read_epoch >= DEVICE_RECHECK_S:
            self._read_device_memory(now)
        blocked_by: str | None = None
        for spec in sorted(pending, key=lambda s: self._rank_key(s, now)):
            if spec.errors:
                self._refuse(spec, "; ".join(spec.errors))
                continue
            if (self.paths.cancel / spec.name).exists():
                self._refuse(
                    spec, "cancelled before it started", rc=RC_CANCELLED, reason="cancelled"
                )
                continue
            if (self.paths.running / f"{spec.name}.sh").exists():
                self._wait(spec, "a job of the same name is still running")
                continue
            hold = self._admission_hold(spec, now)
            if hold is not None:
                self._wait(spec, hold.reason)
                continue
            if spec.holds_gpus:
                if blocked_by is not None:
                    self._wait(spec, f"queued behind {blocked_by}")
                    continue
                placed = self._place(spec, now)
                if isinstance(placed, Hold):
                    self._wait(spec, placed.reason)
                    blocked_by = spec.name
                    continue
                devices = placed
            else:
                devices = ()
            # Memory is checked only once the devices are there, so a job's
            # wait for memory is not charged while it is queued for devices.
            memory = self._memory_hold(spec, now)
            if memory is not None:
                if (self.paths.pending / f"{spec.name}.sh").exists():
                    self._wait(spec, memory.reason)
                    if spec.holds_gpus and blocked_by is None:
                        blocked_by = spec.name
                continue
            self._start(spec, devices, now)

    def _devices_for(self, spec: JobSpec, free: list[str]) -> tuple[str, ...] | None:
        """The devices ``spec`` would get now, or ``None`` if it cannot start yet."""
        if spec.whole:
            busy = any(job.spec.holds_gpus for job in self.jobs.values())
            return tuple(free) if len(free) == len(self.config.devices) and not busy else None
        need = int(spec.gpus or 0)
        if len(free) >= need and not self._whole_running():
            return tuple(free[:need])
        return None

    def _place(self, spec: JobSpec, now: float) -> "tuple[str, ...] | Hold":
        """The devices a GPU job starts on now, or why it waits for devices.

        Once the job has devices, the free ones are read, and those with at
        most ``device_max_used_mib`` in use are handed out first; the job
        waits for any it still gets that are over it. Only free devices are
        read against it: the memory of this watcher's running jobs is on the
        devices they hold. A device still over it ``device_release_wait_s``
        after a reading first found it so is quarantined, and the job is
        placed without it. Only a reading returns a quarantined device, so a
        job that only a quarantined device would place takes a reading first.
        """
        free = self._free_devices()
        devices = self._devices_for(spec, free)
        if devices is None and self._device_quarantine:
            quarantined = [d for d in self.config.devices if d in self._device_quarantine]
            if self._devices_for(spec, free + quarantined) is not None:
                self._read_device_memory(now)
                free = self._free_devices()
                devices = self._devices_for(spec, free)
        if devices is None:
            return Hold(self._device_wait_reason(spec, free))
        reading = self._read_device_memory(now) if devices else None
        if reading is None:
            return devices
        free = self._free_devices()
        limit = self.config.device_max_used_mib
        held = {d: reading[d] for d in free if reading.get(d, 0) > limit}
        for device in set(self._device_held_since) - set(held):
            del self._device_held_since[device]
        for device in held:
            self._device_held_since.setdefault(device, now)
        overdue = {
            device: used
            for device, used in held.items()
            if now - self._device_held_since[device] >= self.config.device_release_wait_s
        }
        if overdue:
            self._quarantine_devices(overdue, now)
            free = [d for d in free if d not in overdue]
        devices = self._devices_for(
            spec, [d for d in free if d not in held] + [d for d in free if d in held]
        )
        if devices is None:
            return Hold(self._device_wait_reason(spec, free))
        waiting = [d for d in devices if d in held]
        if not waiting:
            return devices
        since = min(self._device_held_since[d] for d in waiting)
        return Hold(
            "device memory in use: "
            + ", ".join(f"{d} ({held[d]} MiB)" for d in waiting)
            + f" > {limit:.0f} MiB (waiting {now - since:.0f}s of "
            f"{self.config.device_release_wait_s:.0f}s for it to be released)"
        )

    def _device_wait_reason(self, spec: JobSpec, free: list[str]) -> str:
        reason = (
            "waiting for the whole node to drain"
            if spec.whole
            else f"waiting for {spec.gpus} GPU(s); {len(free)} free"
        )
        if self._device_quarantine:
            reason += "; quarantined, memory not released: " + ", ".join(
                f"{device} ({entry['used_mib']} MiB)"
                for device, entry in sorted(self._device_quarantine.items())
            )
        return reason

    def device_memory_mib(self) -> Mapping[str, int] | None:
        """Memory in use on each device (MiB) by device id; ``None`` when it cannot be read."""
        return read_device_memory_mib()

    def _read_device_memory(self, now: float) -> Mapping[str, int] | None:
        """This pass's device reading: ``nvidia-smi`` runs once a pass at most.

        A reading returns every quarantined device it finds released.
        """
        if self.config.device_max_used_mib <= 0:
            return None
        if self._device_read:
            return self._device_reading
        self._device_read, self._device_read_epoch = True, now
        self._device_reading = self.device_memory_mib()
        if self._device_reading is None:
            return None
        for device, entry in list(self._device_quarantine.items()):
            used = self._device_reading.get(device)
            if used is None:
                continue
            if used > self.config.device_max_used_mib:
                entry["used_mib"] = used
                continue
            del self._device_quarantine[device]
            self._append_status(
                f"watcher: device {device} released its memory ({used} MiB in use); "
                f"back in service at {utc_iso(now)}"
            )
        return self._device_reading

    def _quarantine_devices(self, overdue: Mapping[str, int], now: float) -> None:
        """Take devices out of service whose memory no job of this watcher accounts for."""
        for device, used in overdue.items():
            since = self._device_held_since.pop(device, now)
            self._device_quarantine[device] = {"used_mib": used, "since": since}
            self.counters["device_quarantined"] = self.counters.get("device_quarantined", 0) + 1
            self._append_status(
                f"watcher: device {device} quarantined at {utc_iso(now)}: {used} MiB in use "
                f"{now - since:.0f}s after a reading first found it so, and no job here holds it"
            )

    def _admission_hold(self, spec: JobSpec, now: float) -> "Hold | None":
        """Why this job must wait whatever the devices say; ``None`` if it may start.

        These holds never block the jobs ranked below: they are about the job
        itself (the session is ending, the native build it loads is being
        rebuilt), not about a resource it shares with them.
        """
        hold = self._handover_hold(spec, now)
        if hold is not None:
            return hold
        if spec.holds_gpus:
            for marker in _rebuild.active_announcements(self.paths.root, now):
                # A job that names no tree imports whatever the session's
                # interpreter resolves, which is most often the tree being
                # rebuilt, so it waits too.
                if spec.tree and not _rebuild.loads_rebuilt(
                    spec.tree, marker, self._native_locations(spec.tree)
                ):
                    continue
                linked = (
                    f"; {spec.tree} shares its native build"
                    if spec.tree and not _rebuild.same_tree(spec.tree, marker.get("tree"))
                    else ""
                )
                return Hold(
                    f"native rebuild of {marker.get('tree')} in progress since "
                    f"{utc_iso(float(marker.get('started', now)))} on {marker.get('host')}"
                    + linked,
                    blocks_queue=False,
                )
        return None

    def _native_locations(self, tree: str) -> list[str]:
        """Where ``tree``'s native build really is, resolved once a pass."""
        if tree not in self._native_seen:
            self._native_seen[tree] = _rebuild.native_locations(tree)
        return self._native_seen[tree]

    def _memory_hold(self, spec: JobSpec, now: float) -> "Hold | None":
        """Wait for host memory and /dev/shm a job needs; refuse it if they never come back.

        An engine that has just exited releases its host allocations lazily:
        VMM-backed tables can take 1-6 minutes to come back, and an engine
        started into that gap fails at creation or hangs. So a job that
        declares ``host_mem_gb`` (whole-node jobs inherit the session's
        ``engine_host_mem_gb``) waits for it here, before it is started.
        """
        need = float(spec.host_mem_gb or 0.0)
        shm_need = float(spec.shm_gb or 0.0)
        if need <= 0 and shm_need <= 0:
            return None
        available = self.available_memory_gb()
        shm = shm_available_gb() if shm_need > 0 else None
        short = []
        if need > 0 and available is not None and available - self._memory_committed_gb < need:
            short.append(
                f"host memory {available - self._memory_committed_gb:.0f} GiB available, {need:.0f} needed"
            )
        if shm_need > 0 and shm is not None and shm < shm_need:
            short.append(f"/dev/shm {shm:.0f} GiB free, {shm_need:.0f} needed")
        if not short:
            self._memory_wait_since.pop(spec.name, None)
            self._memory_committed_gb += need
            return None
        since = self._memory_wait_since.setdefault(spec.name, now)
        if now - since >= self.config.host_mem_wait_s:
            self._memory_wait_since.pop(spec.name, None)
            holders = sorted(self.jobs.values(), key=lambda job: job.rss_gb, reverse=True)[:3]
            detail = (
                "; ".join(short)
                + f" after waiting {now - since:.0f}s"
                + (
                    " (largest jobs here: "
                    + ", ".join(f"{job.spec.name} {job.rss_gb:.0f} GiB" for job in holders)
                    + ")"
                    if holders
                    else " (no job of this session holds it)"
                )
            )
            self._refuse(spec, detail, rc=RC_HOST_MEMORY, reason="host_memory")
            return Hold("refused: host memory never came back")
        return Hold("; ".join(short) + f" (waiting {now - since:.0f}s for it to be released)")

    # -------------------------------------------------------------- hand-over

    def _plan_handover(self, now: float) -> None:
        """Near the allocation's end: ask for a successor once, and notice when one is up.

        A successor already up needs no request, and one written before it
        came up is set aside. A request that left the disk while this
        watcher still needs a successor (a reader judged it stale through a
        heartbeat gap) is written again.
        """
        ends_at = self.config.ends_at
        if ends_at is None or ends_at - now > self.config.handover_lead_s:
            return
        for peer in self.paths.live_watchers(self.config.orphan_after_s, now):
            if peer.get("watcher") == self.watcher_id or not _ends_after(peer, ends_at):
                continue
            if self.handover.get("successor") != peer.get("watcher"):
                self._append_status(f"watcher: successor {peer.get('watcher')} is up")
            self.handover["successor"] = peer.get("watcher")
            on_disk = read_json(self.paths.handover_request) or {}
            if self._request and on_disk.get("request_id") == self._request["request_id"]:
                self._set_aside(
                    self.paths.handover_request,
                    f"request-{self.slurm_job}",
                    f"its successor {peer.get('watcher')} is up",
                )
            return
        if "requested_at" not in self.handover:
            request_id = f"{self.config.session or 'session'}-{self.slurm_job}-{int(now)}"
            self._request = {
                "request_id": request_id,
                "session": self.config.session,
                "spool": str(self.paths.root),
                "watcher": self.watcher_id,
                "slurm_job": self.slurm_job,
                "ends_at": utc_iso(ends_at),
                "ends_at_epoch": ends_at,
                "remaining_s": round(ends_at - now, 1),
                "entry": self.config.entry,
                "requested_at": utc_iso(now),
            }
            self._put_request(supersede=True)
            self.handover.update({"request_id": request_id, "requested_at": utc_iso(now)})
            self._append_status(
                f"watcher: allocation ends {utc_iso(ends_at)}; successor requested ({request_id})"
            )
        elif self._request and not self.paths.handover_request.exists():
            self._put_request(supersede=False)
            self._append_status(
                f"watcher: successor request {self._request['request_id']} was no longer on "
                f"disk; written again at {utc_iso(now)}"
            )

    def _put_request(self, *, supersede: bool) -> None:
        """Write this watcher's request; with ``supersede``, set aside another one first."""
        on_disk = read_json(self.paths.handover_request)
        if supersede and on_disk and on_disk.get("request_id") != self._request["request_id"]:
            self._set_aside(
                self.paths.handover_request,
                f"request-{on_disk.get('slurm_job') or 'unknown'}",
                f"superseded by {self._request['request_id']} of watcher {self.watcher_id}",
            )
        self.paths.handover.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.paths.handover_request, self._request)

    def _set_aside(self, path: Path, label: str, why: str) -> bool:
        """Archive a hand-over file; whether it moved. Bookkeeping never stops the watcher."""
        try:
            return self.paths.archive_handover(path, label, why=why, by=self.watcher_id) is not None
        except OSError as exc:
            print(f"watcher: could not set aside {path}: {exc}", file=sys.stderr, flush=True)
            return False

    def _take_over_handover(self, now: float) -> None:
        """On start: set aside the request of the allocation this one succeeds, and its acks.

        That request is answered by this watcher being up, and left on disk it
        names an allocation that has ended. An ack is kept while it follows
        this allocation (an early ack for its successor, or one written
        before a requeue of it) or a live allocation this one does not
        succeed; every other one recorded a hand-over that is over.
        """
        live = [
            peer
            for peer in self.paths.live_watchers(self.config.orphan_after_s, now)
            if peer.get("watcher") != self.watcher_id
        ]
        took = []
        request = read_json(self.paths.handover_request)
        if (
            request
            and request.get("watcher") != self.watcher_id
            and _ends_after({"ends_at_epoch": self.config.ends_at}, request.get("ends_at_epoch"))
        ):
            label = f"request-{request.get('slurm_job') or 'unknown'}"
            why = (
                f"successor {self.watcher_id} (allocation {self.slurm_job}) started at "
                f"{utc_iso(now)}"
            )
            if self._set_aside(self.paths.handover_request, label, why):
                took.append(str(request.get("request_id") or label))
        keep = {self.slurm_job} | {
            str(peer.get("slurm_job") or "")
            for peer in live
            if _ends_after(peer, self.config.ends_at)
        }
        for path, ack in self.paths.read_acks():
            follows = str(ack.get("succeeds") or "")
            if str(ack.get("job_id") or "") != self.slurm_job and follows in keep - {""}:
                continue
            label = f"ack-{follows or 'job-' + str(ack.get('job_id') or 'unknown')}"
            why = (
                f"allocation {self.slurm_job} is up"
                if str(ack.get("job_id") or "") == self.slurm_job
                else f"successor {self.watcher_id} (allocation {self.slurm_job}) started"
            )
            if self._set_aside(path, label, why):
                took.append(f"ack of job {ack.get('job_id')}")
        if took:
            self._append_status(f"watcher: set aside the hand-over it succeeds: {', '.join(took)}")

    def _handover_hold(self, spec: JobSpec, now: float) -> "Hold | None":
        ends_at = self.config.ends_at
        if ends_at is None or ends_at - now > self.config.handover_lead_s:
            return None
        successor = self.handover.get("successor")
        if successor:
            return Hold(f"this session is handing over to {successor}", blocks_queue=False)
        remaining = ends_at - now
        wall = float(spec.wall_s or 0)
        # How long such a job runs, not its worst case: one that overruns is lost with
        # the allocation and runs again on the successor (LOST ALLOCATIONS).
        wall = expected_s(self.paths.done, spec.owner, wall, spec.describe_gpus()) or wall
        if spec.holds_gpus and (wall <= 0 or wall + HANDOVER_MARGIN_S > remaining):
            return Hold(
                f"the allocation ends in {remaining:.0f}s, too soon for a job allowed "
                f"{'unlimited time' if wall <= 0 else f'{wall:.0f}s'}; it waits for the "
                f"successor session, which drains this spool",
                blocks_queue=False,
            )
        return None

    def _wait(self, spec: JobSpec, reason: str) -> None:
        self.pending_reasons[spec.name] = reason
        self._last_pending.append(spec)

    # ---------------------------------------------------------------- launch

    def _job_environment(self, spec: JobSpec, devices: tuple[str, ...]) -> dict[str, str]:
        env = scrub_environment(self.environ)
        if spec.holds_gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        env.update(
            {
                "MV2FLOW_SPOOL": str(self.paths.root),
                "MV2FLOW_SPOOL_JOB": spec.name,
                "MV2FLOW_SPOOL_WATCHER": self.watcher_id,
                "MV2FLOW_SPOOL_DEVICES": ",".join(devices),
                "MV2FLOW_SPOOL_SESSION_JOBID": self.slurm_job,
                "GPU_SLOT_SPOOL": str(self.paths.root),
                # Under `srun` every node inherits the batch host's HOSTNAME.
                "HOSTNAME": self.host,
            }
        )
        if spec.owner:
            env["MV2FLOW_JOB_PREFIX"] = spec.owner
        python_path = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        if spec.tree:
            # The working directory comes first on an MPI worker's sys.path,
            # ahead of PYTHONPATH, so both have to name the tree.
            env["MV2FLOW_TREE"] = spec.tree
            python_path = [spec.tree, *[p for p in python_path if p != spec.tree]]
        if spec.resolve != "off" and RESOLVE_HOOK_DIR.is_dir():
            python_path = [str(RESOLVE_HOOK_DIR), *python_path]
            env["MV2FLOW_RESOLVE_LOG"] = str(self.paths.resolve_log(spec.name))
            env["MV2FLOW_RESOLVE_MODULES"] = spec.resolve or DEFAULT_RESOLVE_MODULES
        if python_path:
            env["PYTHONPATH"] = os.pathsep.join(python_path)
        return env

    def _job_cwd(self, spec: JobSpec) -> str | None:
        return spec.tree or None

    def _start(self, spec: JobSpec, devices: tuple[str, ...], now: float) -> bool:
        pending = self.paths.pending / f"{spec.name}.sh"
        running = self.paths.running / f"{spec.name}.sh"
        try:
            os.rename(pending, running)
        except OSError:
            # Another watcher on this spool claimed it first.
            return False
        self._spec_cache.pop(spec.name, None)
        attempts = self._earlier_attempts(spec)
        archived = self.paths.archive(spec.name)
        if attempts:
            self._note_lost_log(spec, attempts, archived)
        log_path = self.paths.log(spec.name)
        env = self._job_environment(spec, devices)
        cwd = self._job_cwd(spec)
        header = (
            f"=== start {utc_iso(now)} on {self.host} job={spec.name} "
            f"gpus={spec.describe_gpus()} devices={','.join(devices) or '-'} "
            f"wall={int(spec.wall_s or 0)}s watcher={self.watcher_id} ===\n"
        )
        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write(header + self._extra_log_header(spec) + "\n")
            log = log_path.open("ab")
            try:
                proc = subprocess.Popen(
                    ["bash", str(running)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                    env=env,
                    start_new_session=True,
                    preexec_fn=_disable_core_dumps,
                )
            finally:
                log.close()
        except OSError as exc:
            self.counters["launch_failed"] = self.counters.get("launch_failed", 0) + 1
            self._finalize_unstarted(spec, RC_LAUNCH_FAILED, "launch_failed", str(exc))
            return False
        job = RunningJob(
            spec=spec,
            proc=proc,
            devices=devices,
            started=now,
            log_path=log_path,
            attempts=attempts,
        )
        # The scheduler's own header is not the job's progress.
        job.log_size = len(header.encode("utf-8")) + len(self._extra_log_header(spec).encode()) + 1
        job.last_progress = now
        self.jobs[spec.name] = job
        write_json_atomic(
            self.paths.running / f"{spec.name}.json",
            {
                "name": spec.name,
                "watcher": self.watcher_id,
                "host": self.host,
                "pid": proc.pid,
                "devices": list(devices),
                "gpus": spec.describe_gpus(),
                "started": utc_iso(now),
                "started_epoch": now,
                "wall_s": job.wall_s,
                "owner": spec.owner,
                "token": spec.token,
                "slurm_job": self.slurm_job,
                "restart": self.restart,
                "session": self.config.session,
                **({"attempt": len(attempts) + 1} if attempts else {}),
            },
        )
        self.counters["started"] = self.counters.get("started", 0) + 1
        return True

    def _earlier_attempts(self, spec: JobSpec) -> list[dict]:
        """The runs of ``spec`` lost with their allocation before this one, oldest first.

        A requeue record another submission of the name left behind is not
        this job's history, and is dropped.
        """
        path = self.paths.requeue_record(spec.name)
        if not path.exists():
            return []
        record = self.paths.earlier_runs(spec.name, spec.token)
        if record is None:
            path.unlink(missing_ok=True)
            return []
        return [dict(run) for run in record.get("attempts") or [] if isinstance(run, dict)]

    def _note_lost_log(self, spec: JobSpec, attempts: list[dict], archived: list[Path]) -> None:
        """Name, in the last lost run, where its log was archived as this run started."""
        log = next((path for path in archived if path.suffix == ".log"), None)
        if log is None or attempts[-1].get("log"):
            return
        attempts[-1]["log"] = str(log)
        record = self.paths.earlier_runs(spec.name, spec.token)
        if record is not None:
            record["attempts"] = attempts
            write_json_atomic(self.paths.requeue_record(spec.name), record)

    def _extra_log_header(self, spec: JobSpec) -> str:
        if not spec.tree:
            return ""
        return f"=== tree {spec.tree} (working directory and first PYTHONPATH entry) ===\n"

    # ----------------------------------------------------------------- limits

    def _advance(self, job: RunningJob, now: float, table: Mapping[int, ProcInfo]) -> None:
        leader_rc = job.proc.poll()
        pids = job_processes(job.proc.pid, table) if table else []
        if leader_rc is None and job.proc.pid not in pids:
            pids.append(job.proc.pid)
        job.pids = tuple(pids)
        job.rss_gb = sum(rss_bytes(pid, self.proc_root) for pid in pids) / _GB
        job.peak_rss_gb = max(job.peak_rss_gb, job.rss_gb)

        self._track_progress(job, now)
        if job.verdict is None:
            if leader_rc is not None:
                job.verdict = Verdict(leader_rc, "exit")
                job.exited_at = now
                self._write_receipt(job, now)
            else:
                verdict = self._limit_verdict(job, now)
                if verdict is not None:
                    job.verdict = verdict
                    self._signal(job, pids, signal.SIGTERM, now)
                    job.term_sent_at = now

        if job.verdict is None:
            return

        if table:
            # Once the job is over, also find what escaped its session (see
            # processes_with_marker): a stranded profiler or server keeps the
            # job's devices busy for every job admitted after it.
            escaped = processes_with_marker(
                table, str(self.paths.root), job.spec.name, self.watcher_id, self.proc_root
            )
            pids = sorted(set(pids) | set(escaped))
            job.pids = tuple(pids)

        live = [pid for pid in pids if pid != job.proc.pid or leader_rc is None]
        safe = [pid for pid in live if pid not in job.unsafe_pids]

        if job.term_sent_at is None and safe and job.exited_at is not None:
            if now - job.exited_at >= self.config.leftover_grace_s:
                self._log(job, f"reaping {len(safe)} process(es) the job left behind")
                self._signal(job, safe, signal.SIGTERM, now)
                job.term_sent_at = now
        if job.term_sent_at is not None and job.kill_sent_at is None:
            # A process forked after the SIGTERM went out was not in its list.
            late = [pid for pid in safe if pid not in job.signalled]
            if late:
                self._signal(job, late, signal.SIGTERM, now)
        if (
            job.term_sent_at is not None
            and job.kill_sent_at is None
            and safe
            and now - job.term_sent_at >= self.config.kill_grace_s
        ):
            self._signal(job, safe, signal.SIGKILL, now)
            job.kill_sent_at = now

        if not job.receipt_written and not safe:
            self._write_receipt(job, now)

        if not live:
            self._release(job)

    def _track_progress(self, job: RunningJob, now: float) -> None:
        """Note log growth, and the first failure line in it."""
        try:
            size = job.log_path.stat().st_size
        except OSError:
            return
        if size == job.log_size:
            return
        if self._failure_re is not None and not job.failure_line:
            try:
                with job.log_path.open("rb") as handle:
                    handle.seek(max(job.log_size, size - _LOG_SCAN_BYTES))
                    chunk = handle.read(_LOG_SCAN_BYTES).decode("utf-8", errors="replace")
            except OSError:
                chunk = ""
            for line in chunk.splitlines():
                if self._failure_re.search(line):
                    job.failure_line = line.strip()[:300]
                    break
        job.log_size = size
        job.last_progress = now

    def _limit_verdict(self, job: RunningJob, now: float) -> Verdict | None:
        """Why a still-running job must stop now, if it must."""
        if (self.paths.cancel / job.spec.name).exists():
            return Verdict(RC_CANCELLED, "cancelled", "cancel requested through the spool")
        if job.wall_s and now - job.started > job.wall_s:
            return Verdict(
                RC_WALL_LIMIT,
                "wall_limit",
                f"ran {now - job.started:.0f}s against a wall limit of {job.wall_s:.0f}s "
                f"(measured from start, not from submission)",
            )
        idle = now - max(job.last_progress, job.started)
        if job.failure_line and job.spec.failure_stall_s and idle > job.spec.failure_stall_s:
            return Verdict(
                RC_STALLED,
                "stalled",
                f"no progress for {idle:.0f}s after a failure line ({job.failure_line!r}); "
                f"the other processes are presumed to be waiting on one that died",
            )
        if job.spec.stall_s and idle > job.spec.stall_s:
            return Verdict(RC_STALLED, "stalled", f"no log output for {idle:.0f}s")
        if job.spec.mem_gb and job.rss_gb > job.spec.mem_gb:
            return Verdict(
                RC_MEMORY_LIMIT,
                "memory_limit",
                f"resident memory {job.rss_gb:.1f} GiB exceeded the job's limit of "
                f"{job.spec.mem_gb:.1f} GiB",
            )
        return None

    def _enforce_memory_pressure(self, now: float, table: Mapping[int, ProcInfo]) -> None:
        floor = self.config.pressure_floor_gb
        if floor <= 0:
            return
        if now < self._pressure_settle_until or any(
            job.verdict is not None and job.verdict.reason == "memory_pressure"
            for job in self.jobs.values()
        ):
            # One victim at a time, and its memory takes a moment to come back.
            return
        available = self.available_memory_gb()
        if available is None or available >= floor:
            return
        candidates = [job for job in self.jobs.values() if job.verdict is None]
        if not candidates:
            return
        victim = max(candidates, key=lambda job: job.rss_gb)
        victim.verdict = Verdict(
            RC_MEMORY_LIMIT,
            "memory_pressure",
            f"host memory available fell to {available:.1f} GiB (floor {floor:.1f} GiB); "
            f"this job held the most ({victim.rss_gb:.1f} GiB resident)",
        )
        pids = job_processes(victim.proc.pid, table) or [victim.proc.pid]
        self._signal(victim, pids, signal.SIGTERM, now)
        victim.term_sent_at = now

    def available_memory_gb(self) -> float | None:
        """Host memory a new job could still use, in GiB (cgroup-aware)."""
        return host_memory_available_gb()

    def _signal(self, job: RunningJob, pids: Iterable[int], sig: int, now: float) -> None:
        """Signal a job's processes, honouring the safe-kill rule."""
        sent, refused = [], []
        for pid in pids:
            environ = read_environ(pid, self.proc_root)
            if environ is None:
                continue
            if is_mpi_attached(environ):
                refused.append(pid)
                continue
            try:
                os.kill(pid, sig)
                sent.append(pid)
                job.signalled.add(pid)
            except ProcessLookupError:
                continue
            except PermissionError:
                refused.append(pid)
        if refused:
            job.unsafe_pids = tuple(sorted(set(job.unsafe_pids) | set(refused)))
        name = signal.Signals(sig).name
        detail = job.verdict.detail if job.verdict else ""
        self._log(
            job,
            f"{name} to {len(sent)} process(es) at {utc_iso(now)}"
            + (f": {detail}" if detail and sig == signal.SIGTERM else "")
            + (
                f"; left {len(refused)} MPI-attached process(es) alone "
                f"({', '.join(map(str, refused))}), devices quarantined until they exit"
                if refused
                else ""
            ),
        )

    def _log(self, job: RunningJob, message: str) -> None:
        try:
            with job.log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"=== scheduler: {message} ===\n")
        except OSError:
            pass

    # --------------------------------------------------------------- receipts

    def _record(self, spec: JobSpec, verdict: Verdict, **extra: object) -> dict:
        return {
            "name": spec.name,
            "rc": verdict.rc,
            "reason": verdict.reason,
            "detail": verdict.detail,
            "timed_out": verdict.reason in TIMEOUT_REASONS,
            "owner": spec.owner,
            "token": spec.token,
            "label": spec.label,
            "tree": spec.tree,
            "gpus": spec.describe_gpus(),
            "watcher": self.watcher_id,
            "host": self.host,
            "session": self.config.session,
            "slurm_job": self.slurm_job,
            "submitted": utc_iso(spec.submitted) if spec.submitted else None,
            **extra,
        }

    def _write_receipt(self, job: RunningJob, now: float) -> None:
        if not self._holds(job.spec.name):
            # This watcher fell silent long enough for another to find the run
            # lost; what it found (a re-run, or a receipt) stands.
            self._log(
                job,
                f"another watcher took this job over while this one was silent; the result "
                f"of this run, ended {utc_iso(now)}, is not recorded",
            )
            job.receipt_written = True
            return
        verdict = job.verdict or Verdict(RC_LAUNCH_FAILED, "launch_failed")
        if verdict.reason == "exit":
            verdict = self._judge_provenance(job, verdict)
            job.verdict = verdict
        runs = {}
        if job.attempts:
            this = self._run_entry(len(job.attempts) + 1, job.devices, utc_iso(job.started))
            this.update(ended=utc_iso(now), outcome=verdict.reason, rc=verdict.rc)
            runs = {"attempt": len(job.attempts) + 1, "attempts": [*job.attempts, this]}
        record = self._record(
            job.spec,
            verdict,
            devices=list(job.devices),
            started=utc_iso(job.started),
            started_epoch=job.started,
            ended=utc_iso(now),
            duration_s=round(now - job.started, 3),
            wall_s=job.wall_s,
            peak_rss_gb=round(job.peak_rss_gb, 3),
            unsafe_pids=list(job.unsafe_pids),
            **runs,
        )
        self._finish_files(job.spec.name, record, job.log_path, now)
        if job.attempts:
            self.paths.requeue_record(job.spec.name).unlink(missing_ok=True)
        job.receipt_written = True
        key = f"verdict_{verdict.reason}"
        self.counters[key] = self.counters.get(key, 0) + 1

    def _holds(self, name: str) -> bool:
        """Whether the run of ``name`` in ``running/`` is still this watcher's to close.

        It is not once another watcher found it lost while this one was
        silent: that watcher removed it from ``running/`` (and re-queued it or
        closed it out), or a re-run of it there names another watcher.
        """
        meta = read_json(self.paths.running / f"{name}.json")
        if meta is not None:
            return meta.get("watcher") == self.watcher_id
        return (self.paths.running / f"{name}.sh").exists()

    def _run_entry(self, attempt: int, devices: Iterable[str], started: str) -> dict:
        """One run of a job on this watcher's allocation, as ``attempts`` lists it."""
        return {
            "attempt": attempt,
            "watcher": self.watcher_id,
            "host": self.host,
            "slurm_job": self.slurm_job,
            "restart": self.restart,
            "session": self.config.session,
            "devices": list(devices),
            "started": started,
        }

    def _judge_provenance(self, job: RunningJob, verdict: Verdict) -> Verdict:
        """Refuse a result whose processes imported from different checkouts."""
        records = _provenance.read_records(self.paths.resolve_log(job.spec.name))
        if not records:
            return verdict
        judged = _provenance.check(records, tree=job.spec.tree or None)
        if judged.ok:
            return verdict
        self._log(job, f"mixed checkout: {judged.message}")
        return Verdict(
            verdict.rc if verdict.rc != 0 else RC_MIXED_CHECKOUT,
            "mixed_checkout",
            f"{judged.message}. The processes of this job did not all run the code "
            f"under test, so its result describes no single tree.",
        )

    def _finish_files(self, name: str, record: dict, log_path: Path, now: float) -> None:
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"=== end rc={record['rc']} reason={record['reason']} {utc_iso(now)} ===\n"
                )
        except OSError:
            pass
        write_json_atomic(self.paths.record(name), record)
        running = self.paths.running / f"{name}.sh"
        if running.exists():
            os.replace(running, self.paths.done / f"{name}.sh")
        (self.paths.running / f"{name}.json").unlink(missing_ok=True)
        write_text_atomic(self.paths.receipt(name), f"{record['rc']}\n")

    def _finalize_unstarted(
        self, spec: JobSpec, rc: int, reason: str, detail: str, **extra: object
    ) -> None:
        """Close out a job this watcher did not run: refused, never launched, or lost.

        A job re-queued after its allocation was lost keeps its earlier runs
        in the receipt (``attempts``) unless ``extra`` lists them.
        """
        now = self.clock()
        if "attempts" not in extra:
            attempts = self._earlier_attempts(spec)
            if attempts:
                extra.update(attempt=len(attempts), attempts=attempts)
        record = self._record(spec, Verdict(rc, reason, detail), ended=utc_iso(now), **extra)
        log_path = self.paths.log(spec.name)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"=== scheduler: {reason}: {detail} ===\n")
        self._finish_files(spec.name, record, log_path, now)
        if extra.get("attempts"):
            self.paths.requeue_record(spec.name).unlink(missing_ok=True)

    def _refuse(
        self, spec: JobSpec, detail: str, *, rc: int = RC_REFUSED, reason: str = "refused"
    ) -> None:
        pending = self.paths.pending / f"{spec.name}.sh"
        running = self.paths.running / f"{spec.name}.sh"
        if (self.paths.running / f"{spec.name}.sh").exists():
            return
        try:
            os.rename(pending, running)
        except OSError:
            return
        self._spec_cache.pop(spec.name, None)
        self.paths.archive(spec.name)
        self.counters[reason] = self.counters.get(reason, 0) + 1
        self._finalize_unstarted(spec, rc, reason, detail)
        (self.paths.cancel / spec.name).unlink(missing_ok=True)

    def _release(self, job: RunningJob) -> None:
        now = self.clock()
        if not job.receipt_written:
            self._write_receipt(job, now)
        if job.verdict is not None and job.verdict.reason == "memory_pressure":
            self._pressure_settle_until = now + PRESSURE_SETTLE_S
        self.jobs.pop(job.spec.name, None)
        (self.paths.cancel / job.spec.name).unlink(missing_ok=True)

    # ---------------------------------------------------------------- orphans

    def _recover_orphans(self, now: float, table: Mapping[int, ProcInfo]) -> None:
        """Run again, or close out, the jobs a watcher that is gone left in ``running/``.

        A job whose watcher stopped heart-beating, or ran under an earlier
        start of this allocation or of this watcher (:meth:`_earlier_start`),
        was lost with that allocation: :meth:`_lost` queues it again or closes
        it out. One whose processes still run on this host (``table``) lost
        only its watcher, and one no watcher recorded starting has nothing to
        say where it ran; both are closed out as ``orphaned``.
        """
        fresh = {
            record.get("watcher")
            for record in self.paths.live_watchers(self.config.orphan_after_s, now)
        }
        fresh.add(self.watcher_id)
        for script in sorted(self.paths.running.glob("*.sh")):
            name = script.name[: -len(".sh")]
            if name in self.jobs:
                continue
            meta = read_json(self.paths.running / f"{name}.json")
            if meta is None:
                try:
                    idle = now - max(
                        script.stat().st_mtime,
                        self.paths.log(name).stat().st_mtime
                        if self.paths.log(name).exists()
                        else 0,
                    )
                except OSError:
                    continue
                if idle < self.config.orphan_after_s or len(fresh) > 1:
                    continue
                detail = f"no watcher claims it and its log has been idle for {idle:.0f}s"
                self._orphan(name, script, None, detail)
                continue
            owner = str(meta.get("watcher") or "")
            requeued = self._earlier_start(meta)
            if owner in fresh and not requeued:
                continue
            closed = read_json(self.paths.record(name))
            if closed and closed.get("watcher") == owner and self.paths.receipt(name).exists():
                # The watcher wrote the receipt and died before tidying up.
                os.replace(script, self.paths.done / f"{name}.sh")
                (self.paths.running / f"{name}.json").unlink(missing_ok=True)
                continue
            alive = (
                processes_with_marker(table, str(self.paths.root), name, owner, self.proc_root)
                if owner
                else []
            )
            if alive:
                detail = (
                    f"watcher {owner} stopped heart-beating; the job it started at "
                    f"{meta.get('started')} still runs here, unattended "
                    f"(pid {', '.join(map(str, alive))})"
                )
                self._orphan(name, script, meta, detail)
                continue
            self._lost(name, script, meta, requeued or self._gone(meta), now)

    def _earlier_start(self, meta: Mapping[str, Any]) -> str:
        """Why the run ``meta`` records was on an earlier start of this allocation, or ``""``.

        Slurm ends a start of a job, every process of it included, before it
        starts the job again, so a run recorded under a lower
        ``SLURM_RESTART_COUNT`` of this job is gone even while its watcher's
        last heartbeat is still fresh. A run that names this watcher yet is
        not one of its jobs was started before it, by a watcher that had its
        name (the same host and pid, after a reboot).
        """
        if meta.get("watcher") == self.watcher_id:
            return (
                f"watcher {self.watcher_id}, which started the job at {meta.get('started')}, "
                f"ended before the watcher of that name now on this spool started, and the job "
                f"with it"
            )
        job = str(meta.get("slurm_job") or "")
        theirs = meta.get("restart")
        if (
            job in ("", "unknown")
            or job != self.slurm_job
            or self.restart is None
            or not isinstance(theirs, int)
            or theirs >= self.restart
        ):
            return ""
        return (
            f"allocation {job} was requeued: its start {theirs} on {meta.get('host')} "
            f"(watcher {meta.get('watcher')}), which started the job at "
            f"{meta.get('started')}, ended, and the job with it"
        )

    def _gone(self, meta: Mapping[str, Any]) -> str:
        """Why the run ``meta`` records, whose watcher stopped heart-beating, is gone."""
        job = str(meta.get("slurm_job") or "")
        where = (
            f"watcher {meta.get('watcher')} on {meta.get('host')}, which started the job at "
            f"{meta.get('started')}"
        )
        if job not in ("", "unknown") and job == self.slurm_job:
            return (
                f"allocation {job} was requeued or restarted: {where}, is gone, and the job with it"
            )
        return (
            f"{where}, stopped heart-beating: its allocation {job or '?'} is gone "
            f"(ended, failed or requeued), and the job with it"
        )

    def _spec_of(self, name: str, script: Path, meta: Mapping[str, Any] | None) -> JobSpec:
        """What a job left in ``running/`` asked for: its header, else what its run recorded."""
        try:
            text = script.read_text(encoding="utf-8", errors="replace")
            submitted = script.stat().st_mtime
        except OSError:
            text, submitted = "", 0.0
        spec = job_spec_from_script(name, text, submitted=submitted)
        spec.owner = spec.owner or str((meta or {}).get("owner") or "")
        spec.token = spec.token or str((meta or {}).get("token") or "")
        return spec

    def _orphan(self, name: str, script: Path, meta: Mapping[str, Any] | None, detail: str) -> None:
        """Close out a job whose run is not known lost with its allocation (``orphaned``)."""
        spec = self._spec_of(name, script, meta)
        ran = (
            {"started": meta.get("started"), "devices": list(meta.get("devices") or [])}
            if meta
            else {}
        )
        self.counters["orphaned"] = self.counters.get("orphaned", 0) + 1
        self._finalize_unstarted(spec, RC_ORPHANED, "orphaned", detail, **ran)

    def _lost(self, name: str, script: Path, meta: Mapping[str, Any], why: str, now: float) -> None:
        """A run lost with its allocation: queue the job again, or close it out ``node_lost``.

        It goes back to ``pending/`` (keeping its submission time, so its place
        in the queue) while it has re-runs left, its cancel was not requested
        and no job of its name waits there. ``requeued/<name>.json`` records
        the lost run first, so a reader that finds the job queued again finds
        why.
        """
        spec = self._spec_of(name, script, meta)
        earlier = self._earlier_attempts(spec)
        run = {
            "attempt": len(earlier) + 1,
            "watcher": meta.get("watcher"),
            "host": meta.get("host"),
            "slurm_job": meta.get("slurm_job"),
            "restart": meta.get("restart"),
            "session": meta.get("session"),
            "devices": list(meta.get("devices") or []),
            "started": meta.get("started"),
            "lost": utc_iso(now),
            "found_by": self.watcher_id,
            "detail": why,
        }
        attempts = [*earlier, run]
        allowance = self.config.requeue if spec.requeue is None else spec.requeue
        if (self.paths.cancel / name).exists():
            stop = "its cancel was requested"
        elif allowance <= 0:
            stop = "it is not to be run again (requeue=0)"
        elif len(earlier) >= allowance:
            stop = f"it already ran again {len(earlier)} time(s), all it may (requeue={allowance})"
        elif (self.paths.pending / f"{name}.sh").exists():
            stop = "a job of the same name is queued"
        else:
            stop = ""
        if stop:
            run["outcome"] = "node_lost"
            self.counters["node_lost"] = self.counters.get("node_lost", 0) + 1
            self._finalize_unstarted(
                spec,
                RC_NODE_LOST,
                "node_lost",
                f"{why}; not run again: {stop}",
                started=meta.get("started"),
                devices=list(meta.get("devices") or []),
                attempt=len(attempts),
                attempts=attempts,
            )
            (self.paths.cancel / name).unlink(missing_ok=True)
            self._append_status(f"watcher: {name} closed out node_lost at {utc_iso(now)}: {stop}")
            return
        run["outcome"] = "requeued"
        write_json_atomic(
            self.paths.requeue_record(name),
            {
                "name": name,
                "token": spec.token,
                "owner": spec.owner,
                "requeue": allowance,
                "attempts": attempts,
                "requeued": utc_iso(now),
                "requeued_epoch": now,
                "by": self.watcher_id,
            },
        )
        try:
            os.rename(script, self.paths.pending / f"{name}.sh")
        except FileNotFoundError:
            return  # another watcher of the spool queued it again first
        (self.paths.running / f"{name}.json").unlink(missing_ok=True)
        self._spec_cache.pop(name, None)
        self.counters["requeued"] = self.counters.get("requeued", 0) + 1
        note = f"{why}; queued again for run {len(attempts) + 1} of at most {allowance + 1}"
        try:
            with self.paths.log(name).open("a", encoding="utf-8") as log:
                log.write(f"=== scheduler: {note} ===\n")
        except OSError:
            pass
        self._append_status(f"watcher: {name} lost at {utc_iso(now)}: {note}")

    def _reap_zombie_children(self, table: Mapping[int, ProcInfo]) -> None:
        """Collect exited orphans re-parented to this watcher (when it is PID 1 or a subreaper).

        Any other exited child of this process was started by code that waits
        for it: collecting it here takes its exit status, and ``subprocess``
        then reports 0 for it.
        """
        if not self._adopts_orphans:
            return
        mine = {job.proc.pid for job in self.jobs.values()}
        me = os.getpid()
        for info in table.values():
            if info.ppid == me and info.state == "Z" and info.pid not in mine:
                try:
                    os.waitpid(info.pid, os.WNOHANG)
                except ChildProcessError:
                    continue


def _disable_core_dumps() -> None:
    """A core of a process with large pinned mappings can push the allocation over its limit."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


# --------------------------------------------------------------------------- CLI


def discover_devices(requested: str, environ: Mapping[str, str]) -> tuple[str, ...]:
    """The device ids to hand out: ``auto``, a count, or an explicit comma list."""
    visible = [d.strip() for d in environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if requested and requested != "auto":
        if "," in requested or not requested.isdigit():
            return tuple(d.strip() for d in requested.split(",") if d.strip())
        count = int(requested)
        return tuple(visible[:count]) if len(visible) >= count else tuple(map(str, range(count)))
    if visible:
        return tuple(visible)
    try:
        listed = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    count = sum(1 for line in listed.stdout.splitlines() if line.startswith("GPU "))
    return tuple(map(str, range(count)))


def _banner() -> list[str]:
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return smi.stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return ["nvidia-smi: not available"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scheduler.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    watch = sub.add_parser(
        "watch", help="become the spool's scheduler (the allocation's main process)"
    )
    watch.add_argument("spool", type=Path)
    watch.add_argument("--poll", type=float, default=3.0, help="seconds between passes")
    watch.add_argument(
        "--gpus", default="auto", help="'auto', a device count, or an explicit comma list of ids"
    )
    watch.add_argument("--session", default="", help="session name, for status and hand-over")
    watch.add_argument("--default-wall", default="2h", help="wall limit for GPU jobs that set none")
    watch.add_argument("--aging", default="1h", help="waiting time that raises priority by one")
    watch.add_argument("--kill-grace", default="60s", help="SIGTERM to SIGKILL")
    watch.add_argument(
        "--orphan-after", default="180s", help="heartbeat age after which a watcher's jobs are lost"
    )
    watch.add_argument(
        "--requeue",
        type=int,
        default=DEFAULT_REQUEUE,
        help="times a job lost with its allocation runs again, unless its header says "
        "(0 = never: it is closed out node_lost)",
    )
    watch.add_argument(
        "--time-limit",
        help="the allocation's time limit; with SLURM_JOB_START_TIME it fixes the end",
    )
    watch.add_argument("--ends-at", type=float, help="the allocation's end, epoch seconds")
    watch.add_argument("--handover-lead", default="1h", help="request a successor this early")
    watch.add_argument("--entry", default="", help="this session's entry script, for successors")
    watch.add_argument(
        "--stall", default="0", help="log-idle time that marks any job hung (0 = off)"
    )
    watch.add_argument(
        "--failure-stall",
        default="10m",
        help="log-idle time after a failure line that marks a GPU job hung",
    )
    watch.add_argument(
        "--engine-host-mem-gb",
        type=float,
        default=0.0,
        help="host memory a whole-node job needs available before it starts (0 = off)",
    )
    watch.add_argument("--host-mem-wait", default="15m", help="how long to wait for that memory")
    watch.add_argument(
        "--device-max-used-mib",
        type=float,
        default=DEFAULT_DEVICE_MAX_USED_MIB,
        help="start a GPU job on a device only with at most this much memory in use (0 = off)",
    )
    watch.add_argument(
        "--device-release-wait",
        default=f"{DEFAULT_DEVICE_RELEASE_WAIT_S:g}s",
        help="how long a free device may stay over that before it is quarantined",
    )
    watch.add_argument(
        "--pressure-floor-gb",
        type=float,
        default=0.0,
        help="stop the largest job when available host memory falls below this (0 = off)",
    )
    return parser


def allocation_end(
    args: argparse.Namespace, environ: Mapping[str, str], now: float | None = None
) -> float | None:
    """When this allocation ends: Slurm's own figure, else the declared limit from its start."""
    now = time.time() if now is None else now
    value = environ.get("SLURM_JOB_END_TIME", "")
    if value.isdigit() and int(value) > 0:
        return float(value)
    if getattr(args, "ends_at", None):
        return float(args.ends_at)
    if getattr(args, "time_limit", None):
        start = environ.get("SLURM_JOB_START_TIME", "")
        begun = float(start) if start.isdigit() else now
        return begun + parse_duration(args.time_limit)
    return None


def config_from_args(args: argparse.Namespace, environ: Mapping[str, str]) -> SchedulerConfig:
    return SchedulerConfig(
        devices=discover_devices(args.gpus, environ),
        poll_s=args.poll,
        default_wall_s=parse_duration(args.default_wall),
        aging_s=parse_duration(args.aging),
        kill_grace_s=parse_duration(args.kill_grace),
        orphan_after_s=parse_duration(args.orphan_after),
        requeue=max(0, getattr(args, "requeue", DEFAULT_REQUEUE)),
        pressure_floor_gb=args.pressure_floor_gb,
        session=args.session,
        ends_at=allocation_end(args, environ),
        stall_s=parse_duration(args.stall),
        failure_stall_s=parse_duration(args.failure_stall),
        engine_host_mem_gb=args.engine_host_mem_gb,
        host_mem_wait_s=parse_duration(args.host_mem_wait),
        device_max_used_mib=args.device_max_used_mib,
        device_release_wait_s=parse_duration(args.device_release_wait),
        handover_lead_s=parse_duration(args.handover_lead),
        entry=args.entry,
    )


def make_scheduler(args: argparse.Namespace) -> Scheduler:
    return Scheduler(args.spool, config_from_args(args, os.environ))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scheduler = make_scheduler(args)

    def _terminated(signum: int, frame: object) -> None:
        del frame
        scheduler._append_status(f"watcher: signal {signum} at {utc_iso()}, exiting")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _terminated)
    scheduler.announce(_banner())
    return scheduler.serve()


__all__ = [
    "ALL_GPUS",
    "DEFAULT_REQUEUE",
    "FREE_PREFIX",
    "HANDOVER_DIRNAME",
    "HEADER_KEYS",
    "HEADER_TAG",
    "JobSpec",
    "LEGACY_ACK_FILENAME",
    "NAME_PATTERN",
    "QUEUED_TAG",
    "RC_CANCELLED",
    "RC_HOST_MEMORY",
    "RC_LAUNCH_FAILED",
    "RC_MEMORY_LIMIT",
    "RC_MIXED_CHECKOUT",
    "RC_NODE_LOST",
    "RC_ORPHANED",
    "RC_REFUSED",
    "RC_STALLED",
    "RC_WALL_LIMIT",
    "RESOLVE_HOOK_DIR",
    "DEFAULT_FAILURE_PATTERNS",
    "shm_available_gb",
    "RunningJob",
    "SESSION_LOSS_REASONS",
    "Scheduler",
    "SchedulerConfig",
    "SpoolPaths",
    "TIMEOUT_REASONS",
    "Verdict",
    "ack_answers",
    "ack_follows",
    "adopts_orphans",
    "answering_ack",
    "following_ack",
    "host_memory_available_gb",
    "is_mpi_attached",
    "job_processes",
    "job_spec_from_script",
    "parse_duration",
    "parse_header",
    "queued_at",
    "read_device_memory_mib",
    "read_environ",
    "read_json",
    "read_proc_table",
    "restart_count",
    "scrub_environment",
    "stale_request",
    "stamp_queued",
    "utc_iso",
    "write_json_atomic",
    "write_text_atomic",
]


if __name__ == "__main__":
    sys.exit(main())
