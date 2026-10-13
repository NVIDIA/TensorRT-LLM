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
    <spool>/STOP                  finish running work, admit nothing, exit

A reader that sees ``done/<name>.rc`` may trust the log and the JSON beside it.

A job declares what it needs in header comment lines anywhere in its first
lines::

    # spool: gpus=2 wall=2h priority=0 mem_gb=200 owner=r014 token=<uuid>

``gpus`` is a device count or ``all`` (the whole node, exclusively). Without
it a job is classified the way the flock-based watcher did: a ``free-`` name
holds nothing, any other name holds the whole node, and a body that calls
``gpu_slot.sh N`` holds N devices.

ADMISSION
=========

One priority queue per watcher, ordered by effective priority (``priority``
plus one per ``aging_s`` waited), then whole-node jobs before device-slot
jobs, then submission time. A GPU job that cannot start blocks every GPU job
ranked below it, so a waiting whole-node or multi-device job is never starved
by a stream of single-device jobs; jobs that hold no device always start.

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
from typing import Callable, Iterable, Mapping

try:
    from . import provenance as _provenance
    from . import rebuild as _rebuild
except ImportError:  # run by path: the sibling modules are on sys.path[0]
    import provenance as _provenance  # type: ignore[no-redef]
    import rebuild as _rebuild  # type: ignore[no-redef]

# --------------------------------------------------------------------------- protocol

#: Prefix of a job header line.
HEADER_TAG = "#spool:"

#: Header lines are only looked for this far into a script.
HEADER_SCAN_LINES = 64

#: A job whose name starts with this and declares no ``gpus`` holds no device.
FREE_PREFIX = "free-"

#: The ``gpus`` value that asks for the whole node, exclusively.
ALL_GPUS = "all"

#: Names a job may have. A name is a file name on a shared filesystem.
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

RC_WALL_LIMIT = 124
RC_STALLED = 210
RC_MEMORY_LIMIT = 211
RC_ORPHANED = 212
RC_REFUSED = 213
RC_HOST_MEMORY = 214
RC_LAUNCH_FAILED = 215
RC_CANCELLED = 216
RC_MIXED_CHECKOUT = 217

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

#: After a memory-pressure victim is gone, how long to let the host reclaim its memory.
PRESSURE_SETTLE_S = 15.0

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
    #: When the job was submitted (the pending file's mtime).
    submitted: float = 0.0
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


def job_spec_from_script(name: str, text: str, submitted: float = 0.0) -> JobSpec:
    """Classify a job from its name, header and body."""
    values, errors = parse_header(text)
    spec = JobSpec(name=name, submitted=submitted, explicit=frozenset(values))
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
    table: Mapping[int, ProcInfo], job: str, watcher: str, proc_root: Path = Path("/proc")
) -> list[int]:
    """Live processes whose environment names this job of this watcher.

    A process that calls ``setsid`` and is re-parented to init once the job's
    leader exits (``nsys profile`` and the application it launches do exactly
    that) is neither in the leader's session nor its descendant any more, so
    :func:`job_processes` stops seeing it. Its environment still carries the
    markers every job is started with, and that is what this finds."""
    want = {f"STAIRCASE_SPOOL_JOB={job}".encode(), f"STAIRCASE_SPOOL_WATCHER={watcher}".encode()}
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

    SUBDIRS = ("pending", "running", "done", "logs", "history", "watchers", "cancel")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.pending = self.root / "pending"
        self.running = self.root / "running"
        self.done = self.root / "done"
        self.logs = self.root / "logs"
        self.history = self.root / "history"
        self.watchers = self.root / "watchers"
        self.cancel = self.root / "cancel"
        self.heartbeat = self.root / "watcher.heartbeat"
        self.jobid = self.root / "watcher.jobid"
        self.status = self.root / "watcher.status"
        self.snapshot = self.root / "status.json"
        self.stop = self.root / "STOP"

    def ensure(self) -> None:
        for name in self.SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def log(self, name: str) -> Path:
        return self.logs / f"{name}.log"

    def receipt(self, name: str) -> Path:
        return self.done / f"{name}.rc"

    def record(self, name: str) -> Path:
        return self.done / f"{name}.json"

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
    #: A running job whose watcher has not heart-beat for this long is orphaned.
    orphan_after_s: float = 180.0
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

    @property
    def wall_s(self) -> float:
        return float(self.spec.wall_s or 0.0)


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
        self.watcher_id = watcher_id or f"{self.host}-{os.getpid()}"
        self.started = self.clock()
        self.jobs: dict[str, RunningJob] = {}
        self.pending_reasons: dict[str, str] = {}
        self.counters: dict[str, int] = {}
        self._spec_cache: dict[str, tuple[float, JobSpec]] = {}
        self._last_pending: list[JobSpec] = []
        self._pressure_settle_until = 0.0
        self.handover: dict = {}
        self._memory_wait_since: dict[str, float] = {}
        self._memory_committed_gb = 0.0
        self._failure_re = (
            re.compile("|".join(self.config.failure_patterns))
            if self.config.failure_patterns
            else None
        )
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
            self.tick()
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
        self._recover_orphans(now)
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
        return {
            "watcher": self.watcher_id,
            "host": self.host,
            "pid": os.getpid(),
            "slurm_job": self.slurm_job,
            "session": self.config.session,
            "spool": str(self.paths.root),
            "started": utc_iso(self.started),
            "alive": utc_iso(now),
            "alive_epoch": now,
            "stopping": self.stopping(),
            "ends_at": utc_iso(self.config.ends_at) if self.config.ends_at else None,
            "ends_at_epoch": self.config.ends_at,
            "remaining_s": round(self.config.ends_at - now, 1) if self.config.ends_at else None,
            "handover": dict(self.handover) or None,
            "devices": list(self.config.devices),
            "free": [d for d in self.config.devices if d not in allocated],
            "quarantined": sorted(quarantined),
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
                }
                for spec in self._last_pending
            ],
            "counters": dict(self.counters),
        }

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
            spec.name,
        )

    def _allocated(self) -> set[str]:
        return {d for job in self.jobs.values() for d in job.devices}

    def _quarantined(self) -> set[str]:
        return {d for job in self.jobs.values() if job.unsafe_pids for d in job.devices}

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
        free = [d for d in self.config.devices if d not in self._allocated()]
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
                devices = self._devices_for(spec, free)
                if devices is None:
                    self._wait(
                        spec,
                        "waiting for the whole node to drain"
                        if spec.whole
                        else f"waiting for {spec.gpus} GPU(s); {len(free)} free",
                    )
                    blocked_by = spec.name
                    continue
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
            if self._start(spec, devices, now):
                free = [d for d in free if d not in devices]

    def _devices_for(self, spec: JobSpec, free: list[str]) -> tuple[str, ...] | None:
        """The devices ``spec`` would get now, or ``None`` if it cannot start yet."""
        if spec.whole:
            busy = any(job.spec.holds_gpus for job in self.jobs.values())
            return tuple(free) if len(free) == len(self.config.devices) and not busy else None
        need = int(spec.gpus or 0)
        if len(free) >= need and not self._whole_running():
            return tuple(free[:need])
        return None

    def _admission_hold(self, spec: JobSpec, now: float) -> "Hold | None":
        """Why this job must wait whatever the devices say; ``None`` if it may start.

        These holds never block the jobs ranked below: they are about the job
        itself (the session is ending, its tree is being rebuilt), not about a
        resource it shares with them.
        """
        hold = self._handover_hold(spec, now)
        if hold is not None:
            return hold
        if spec.holds_gpus:
            for marker in _rebuild.active_announcements(self.paths.root, now):
                # A job that names no tree imports whatever the session's
                # interpreter resolves, which is most often the tree being
                # rebuilt, so it waits too.
                if not spec.tree or _rebuild.same_tree(spec.tree, marker.get("tree")):
                    return Hold(
                        f"native rebuild of {marker.get('tree')} in progress since "
                        f"{utc_iso(float(marker.get('started', now)))} on {marker.get('host')}",
                        blocks_queue=False,
                    )
        return None

    def _memory_hold(self, spec: JobSpec, now: float) -> "Hold | None":
        """Wait for host memory and /dev/shm a job needs; refuse it if they never come back.

        An engine that has just exited releases its host allocations lazily:
        VMM-backed tables were measured coming back 1-6 minutes later, and an
        engine started into that gap fails at creation or hangs. So a job that
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
        """Near the allocation's end: ask for a successor once, and notice when one is up."""
        ends_at = self.config.ends_at
        if ends_at is None or ends_at - now > self.config.handover_lead_s:
            return
        directory = self.paths.root / "handover"
        if "requested_at" not in self.handover:
            request_id = f"{self.config.session or 'session'}-{self.slurm_job}-{int(now)}"
            directory.mkdir(parents=True, exist_ok=True)
            write_json_atomic(
                directory / "request.json",
                {
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
                },
            )
            self.handover.update({"request_id": request_id, "requested_at": utc_iso(now)})
            self._append_status(
                f"watcher: allocation ends {utc_iso(ends_at)}; successor requested ({request_id})"
            )
        for peer in self.paths.live_watchers(self.config.orphan_after_s, now):
            peer_end = peer.get("ends_at_epoch")
            if peer.get("watcher") == self.watcher_id:
                continue
            if peer_end is None or float(peer_end) > ends_at:
                if self.handover.get("successor") != peer.get("watcher"):
                    self._append_status(f"watcher: successor {peer.get('watcher')} is up")
                self.handover["successor"] = peer.get("watcher")
                break

    def _handover_hold(self, spec: JobSpec, now: float) -> "Hold | None":
        ends_at = self.config.ends_at
        if ends_at is None or ends_at - now > self.config.handover_lead_s:
            return None
        successor = self.handover.get("successor")
        if successor:
            return Hold(f"this session is handing over to {successor}", blocks_queue=False)
        remaining = ends_at - now
        wall = float(spec.wall_s or 0)
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
                "STAIRCASE_SPOOL": str(self.paths.root),
                "STAIRCASE_SPOOL_JOB": spec.name,
                "STAIRCASE_SPOOL_WATCHER": self.watcher_id,
                "STAIRCASE_SPOOL_DEVICES": ",".join(devices),
                "STAIRCASE_SPOOL_SESSION_JOBID": self.slurm_job,
                "GPU_SLOT_SPOOL": str(self.paths.root),
                # Under `srun` every node inherits the batch host's HOSTNAME.
                "HOSTNAME": self.host,
            }
        )
        if spec.owner:
            env["STAIRCASE_JOB_PREFIX"] = spec.owner
        python_path = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        if spec.tree:
            # The working directory comes first on an MPI worker's sys.path,
            # ahead of PYTHONPATH, so both have to name the tree.
            env["STAIRCASE_TREE"] = spec.tree
            python_path = [spec.tree, *[p for p in python_path if p != spec.tree]]
        if spec.resolve != "off" and RESOLVE_HOOK_DIR.is_dir():
            python_path = [str(RESOLVE_HOOK_DIR), *python_path]
            env["STAIRCASE_RESOLVE_LOG"] = str(self.paths.resolve_log(spec.name))
            env["STAIRCASE_RESOLVE_MODULES"] = spec.resolve or DEFAULT_RESOLVE_MODULES
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
        self.paths.archive(spec.name)
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
        job = RunningJob(spec=spec, proc=proc, devices=devices, started=now, log_path=log_path)
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
            },
        )
        self.counters["started"] = self.counters.get("started", 0) + 1
        return True

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
            escaped = processes_with_marker(table, job.spec.name, self.watcher_id, self.proc_root)
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
        verdict = job.verdict or Verdict(RC_LAUNCH_FAILED, "launch_failed")
        if verdict.reason == "exit":
            verdict = self._judge_provenance(job, verdict)
            job.verdict = verdict
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
        )
        self._finish_files(job.spec.name, record, job.log_path, now)
        job.receipt_written = True
        key = f"verdict_{verdict.reason}"
        self.counters[key] = self.counters.get(key, 0) + 1

    def _judge_provenance(self, job: RunningJob, verdict: Verdict) -> Verdict:
        """Refuse a result whose processes imported from different checkouts (F044)."""
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

    def _finalize_unstarted(self, spec: JobSpec, rc: int, reason: str, detail: str) -> None:
        now = self.clock()
        record = self._record(spec, Verdict(rc, reason, detail), ended=utc_iso(now))
        log_path = self.paths.log(spec.name)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"=== scheduler: {reason}: {detail} ===\n")
        self._finish_files(spec.name, record, log_path, now)

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

    def _recover_orphans(self, now: float) -> None:
        """Close out jobs a dead watcher left in ``running/``."""
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
            if meta is not None:
                owner = meta.get("watcher")
                if owner in fresh:
                    continue
                closed = read_json(self.paths.record(name))
                if closed and closed.get("watcher") == owner and self.paths.receipt(name).exists():
                    # The watcher wrote the receipt and died before tidying up.
                    os.replace(script, self.paths.done / f"{name}.sh")
                    (self.paths.running / f"{name}.json").unlink(missing_ok=True)
                    continue
                detail = (
                    f"watcher {owner} stopped heart-beating; the job started at "
                    f"{meta.get('started')} on {meta.get('host')} did not finish"
                )
            else:
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
            spec = JobSpec(name=name, owner=str((meta or {}).get("owner", "")))
            spec.token = str((meta or {}).get("token", ""))
            self.counters["orphaned"] = self.counters.get("orphaned", 0) + 1
            self._finalize_unstarted(spec, RC_ORPHANED, "orphaned", detail)

    def _reap_zombie_children(self, table: Mapping[int, ProcInfo]) -> None:
        """Collect exited processes reparented to this watcher (it may be PID 1)."""
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
    watch.add_argument("--orphan-after", default="180s", help="heartbeat age that orphans a job")
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
    for name in ("SLURM_JOB_END_TIME",):
        value = environ.get(name, "")
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
        pressure_floor_gb=args.pressure_floor_gb,
        session=args.session,
        ends_at=allocation_end(args, environ),
        stall_s=parse_duration(args.stall),
        failure_stall_s=parse_duration(args.failure_stall),
        engine_host_mem_gb=args.engine_host_mem_gb,
        host_mem_wait_s=parse_duration(args.host_mem_wait),
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
    "FREE_PREFIX",
    "HEADER_KEYS",
    "HEADER_TAG",
    "JobSpec",
    "NAME_PATTERN",
    "RC_CANCELLED",
    "RC_HOST_MEMORY",
    "RC_LAUNCH_FAILED",
    "RC_MEMORY_LIMIT",
    "RC_MIXED_CHECKOUT",
    "RC_ORPHANED",
    "RC_REFUSED",
    "RC_STALLED",
    "RC_WALL_LIMIT",
    "RESOLVE_HOOK_DIR",
    "DEFAULT_FAILURE_PATTERNS",
    "shm_available_gb",
    "RunningJob",
    "Scheduler",
    "SchedulerConfig",
    "SpoolPaths",
    "TIMEOUT_REASONS",
    "Verdict",
    "host_memory_available_gb",
    "is_mpi_attached",
    "job_processes",
    "job_spec_from_script",
    "parse_duration",
    "parse_header",
    "read_environ",
    "read_json",
    "read_proc_table",
    "scrub_environment",
    "utc_iso",
    "write_json_atomic",
    "write_text_atomic",
]


if __name__ == "__main__":
    sys.exit(main())
