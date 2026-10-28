#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The compute half of ``environment.kind: session_spool``: a scheduler that drains a spool.

A compute session is an allocation whose main process is this scheduler. This module is
**standard-library only** and runs by path (``python3 scheduler.py watch <spool>``): the
interpreter inside the container is the engine's, not the orchestrator's.

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

A reader that sees ``done/<name>.rc`` may trust the log and the JSON beside it. A job declares
what it needs in header lines among its first :data:`HEADER_SCAN_LINES` lines, for example
``#spool: gpus=2 wall=2h priority=0 owner=r014 token=<uuid> tree=<checkout>``.

``gpus`` is a device count or ``all`` (the whole node, exclusively). Without it a ``free-`` name
holds nothing and any other name holds the whole node.

LOST ALLOCATIONS
================

A job in ``running/`` whose watcher stopped heart-beating ran on an allocation that is gone
(Slurm requeued it onto another node, or it ended or failed). The scheduler that finds it (the
requeued allocation's next start, or any other watcher of the spool) runs it again, at most
:data:`MAX_RERUNS` times: the script goes back to ``pending/`` with its submission time, so its
place in the queue, and a ``#spool-rerun: <n>`` line after its shebang counts its re-runs (the
receipt's ``reruns``). A run on an earlier start of this same allocation (a lower
``SLURM_RESTART_COUNT``) is known lost at once; any other once its watcher has been silent
:data:`WATCHER_STALE_S`.

A job that used its re-runs, whose cancel was requested or whose name is already queued gets
``reason=node_lost`` (``RC_NODE_LOST``) instead. A job whose processes still run on this host
outlived only its watcher: it is closed out as ``orphaned`` (``RC_ORPHANED``), never run again
beside itself.

HAND-OVER
=========

::

    <spool>/handover/acks/<allocation>.json   the job a driver submitted to follow <allocation>

``handover_lead`` before its allocation ends a watcher is in hand-over (its status record's
``handover``): it holds each GPU job that would not finish in time, and the operator loop asks
for a successor until an ack names one. An ack is keyed by the allocation it follows, so a later
ack for the next allocation never replaces it. Once a successor's watcher is up, the ending
watcher admits nothing more and exits when its jobs are done.

ADMISSION
=========

One priority queue per watcher, ordered by effective priority (``priority`` plus one per
:data:`AGING_S` waited), then whole-node jobs before device-slot jobs, then submission order (the
pending file's mtime), then name. A GPU job waiting for devices blocks every GPU job ranked below
it, so a stream of single-device jobs never starves a whole-node or multi-device one; a job that
holds no device never waits for one.

A GPU job starts only on devices whose memory has come back. An engine sizes its KV-cache and
recurrent pools from the device memory it finds free, so one started beside memory a cancelled
job (or a rank or launcher child of it) has not given back reserves less than the same engine
elsewhere, or fails. As a GPU job is about to start, one ``nvidia-smi`` reading hands out first
the free devices with at most :data:`DEVICE_MAX_USED_MIB` in use, and the job waits for any other
it gets; a device still in use :data:`DEVICE_RELEASE_WAIT_S` after a reading first found it so is
quarantined until a reading finds it released. Without ``nvidia-smi`` nothing waits.

LIMITS AND THE SAFE-KILL RULE
=============================

Every job runs in its own session with core dumps disabled and without the allocation's MPI
context in its environment (``PMI*``, ``PMIX*``, ``OMPI*``, ``SLURM*``, ``PRTE*``, ``PALS*``). A
process that inherits that context is part of the allocation's MPI job, and when it dies
abnormally Slurm's MPI error handler kills the **whole allocation**. Having removed it, the
scheduler may stop its own jobs: on a wall-clock limit, a stall after a failure line or a cancel
request it sends SIGTERM and, :data:`KILL_GRACE_S` later, SIGKILL. **It never signals a process
whose environment still carries the allocation's context** (:func:`is_mpi_attached`); that
process is left alone, the job is marked failed, and its devices stay quarantined until it exits.
"""

from __future__ import annotations

import argparse
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
    from ..files import read_json, write_json_atomic, write_text_atomic
    from . import provenance as _provenance
    from . import rebuild as _rebuild
except ImportError:  # run by path: the sibling modules are on sys.path[0]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import provenance as _provenance  # type: ignore[no-redef]
    import rebuild as _rebuild  # type: ignore[no-redef]
    from files import read_json, write_json_atomic, write_text_atomic

# --------------------------------------------------------------------------- protocol

HEADER_TAG = "#spool:"

#: Prefix of the line that counts a job's re-runs after lost allocations (:func:`stamp_rerun`).
RERUN_TAG = "#spool-rerun:"

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

HANDOVER_DIRNAME = "handover"

RC_WALL_LIMIT = 124
RC_STALLED = 210
RC_ORPHANED = 212
RC_REFUSED = 213
RC_LAUNCH_FAILED = 215
RC_CANCELLED = 216
RC_MIXED_CHECKOUT = 217
RC_NODE_LOST = 218

#: Verdict reasons of a job its session ended, not the job: nothing in them is a verdict.
SESSION_LOSS_REASONS = frozenset({"orphaned", "node_lost"})

#: A watcher that has not heart-beat for this long is gone, and its allocation with it.
WATCHER_STALE_S = 180.0

#: How many times a job lost with its allocation runs again.
MAX_RERUNS = 1

#: Seconds between scheduling passes.
POLL_S = 3.0

#: Wall limit of a GPU job that declares none (``--default-wall``). Device-free jobs have none.
DEFAULT_WALL_S = 7200.0

#: Seconds of waiting that raise a pending job's priority by one.
AGING_S = 3600.0

#: SIGTERM to SIGKILL.
KILL_GRACE_S = 60.0

#: Seconds without log growth, after a failure line, after which a GPU job is presumed hung.
FAILURE_STALL_S = 600.0

#: Put first on every job's PYTHONPATH: records which checkout each process imported.
RESOLVE_HOOK_DIR = Path(__file__).resolve().parent / "resolvehook"

RESOLVE_MODULES = "tensorrt_llm"

#: Verdict reasons after which the job has said nothing about what it ran.
TIMEOUT_REASONS = frozenset({"wall_limit", "stalled"})

#: Environment-variable prefixes that tie a process to the allocation's MPI job.
SCRUB_PATTERN = re.compile(r"^(PMI|PMIX|OMPI|SLURM|PRTE|PALS|MPI_LOCALRANKID|MPI_LOCALNRANKS)")

#: Log lines after which a job that stops making progress is presumed hung: a
#: rank that died while its peers wait for it forever in a collective.
_FAILURE_LINE = re.compile(
    r"Traceback \(most recent call last\)"
    r"|\bMPI_ABORT\b"
    r"|Segmentation fault"
    r"|terminate called after throwing"
    r"|CUDA error: "
    r"|\bNCCL error\b"
)

#: How much new log a pass reads looking for failure lines.
_LOG_SCAN_BYTES = 1 << 20

#: A job is admitted near the allocation's end only if its wall limit leaves this much spare.
HANDOVER_MARGIN_S = 120.0

#: Device memory in use (MiB) above which a device still holds what a finished job left.
DEVICE_MAX_USED_MIB = 1024.0

#: How long a device may hold memory no job of the watcher accounts for before it is quarantined.
DEVICE_RELEASE_WAIT_S = 300.0

#: How often a quarantined device is read again while no job is waiting to start.
DEVICE_RECHECK_S = 60.0

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


def _parse_tree(text: str) -> str:
    if not text.startswith("/"):
        raise ValueError("must be an absolute path")
    return os.path.normpath(text)


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
    "owner": ("owner", _parse_word),
    "token": ("token", _parse_word),
    "tree": ("tree", _parse_tree),
}


@dataclass
class JobSpec:
    """What a job asked for, read from its header."""

    name: str
    #: Device count, or ``None`` for the whole node.
    gpus: int | None = 0
    priority: int = 0
    #: Wall-clock limit from START, in seconds. ``None`` takes the watcher's default; 0 is none.
    wall_s: float | None = None
    owner: str = ""
    token: str = ""
    #: The checkout the job runs against: its working directory and first PYTHONPATH entry.
    tree: str = ""
    #: When the job was submitted (the pending file's mtime).
    submitted: float = 0.0
    #: Times the job has run again after lost allocations (its ``#spool-rerun:`` line).
    reruns: int = 0
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


def stamp_rerun(script: str, reruns: int) -> str:
    """``script`` with the line that counts its re-runs set to ``reruns``, after its shebang."""
    lines = [line for line in script.splitlines(keepends=True) if not line.startswith(RERUN_TAG)]
    head = 1 if lines and lines[0].startswith("#!") else 0
    lines.insert(head, f"{RERUN_TAG} {reruns}\n")
    return "".join(lines)


def reruns_of(text: str) -> int:
    """How many times the job in ``text`` has run again (its ``#spool-rerun:`` line), else 0."""
    for line in text.splitlines()[:HEADER_SCAN_LINES]:
        if line.startswith(RERUN_TAG):
            value = line[len(RERUN_TAG) :].strip()
            return int(value) if value.isdigit() else 0
    return 0


def job_spec_from_script(name: str, text: str, submitted: float = 0.0) -> JobSpec:
    """Classify a job from its name and header."""
    values, errors = parse_header(text)
    spec = JobSpec(name=name, submitted=submitted, reruns=reruns_of(text))
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
        spec.gpus = 0 if name.startswith(FREE_PREFIX) else None
    spec.errors = errors
    return spec


def scrub_environment(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` without the variables that attach a process to the allocation's MPI job."""
    return {k: v for k, v in env.items() if not SCRUB_PATTERN.match(k)}


def is_mpi_attached(environ: Mapping[str, str]) -> bool:
    """Whether a process carries the allocation's Slurm/PMI context: the safe-kill rule.

    ``PMI_FD`` is Slurm's PMI-2 wire-up. An ``mpirun`` started inside a scrubbed job has its own
    PMIx namespace and is not attached: signalling it or its ranks cannot reach the allocation.
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


def read_proc_table() -> dict[int, ProcInfo]:
    """Every visible process's parent, session and state."""
    table: dict[int, ProcInfo] = {}
    try:
        entries = list(Path("/proc").iterdir())
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


def read_environ(pid: int) -> dict[str, str] | None:
    """A process's initial environment, or ``None`` when it cannot be read."""
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    env: dict[str, str] = {}
    for item in raw.split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep:
            env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env


def rss_bytes(pid: int) -> int:
    """Resident set size of one process; 0 when it is gone."""
    try:
        fields = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()
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
    table: Mapping[int, ProcInfo], spool: str, job: str, watcher: str
) -> list[int]:
    """Live processes whose environment names this job of this watcher on this spool.

    A process that calls ``setsid`` and is re-parented to init once the leader exits escapes
    :func:`job_processes` (``nsys profile`` and its application do) but keeps the job's markers.
    Job names and watcher ids are unique only within a spool, so the spool is matched too.
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
            parts = set(Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"))
        except OSError:
            continue
        if want <= parts:
            found.append(pid)
    return sorted(found)


def read_device_memory_mib() -> dict[str, int] | None:
    """Memory in use on every device of the node, in MiB, keyed by index and by UUID.

    ``nvidia-smi`` ignores ``CUDA_VISIBLE_DEVICES``, so a watcher's device id (an entry of it) is
    found under whichever key it spells. ``None`` when ``nvidia-smi`` is missing or reads no device.
    """
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
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
        self.status = self.root / "watcher.status"
        self.snapshot = self.root / "status.json"
        self.handover = self.root / HANDOVER_DIRNAME
        self.acks = self.handover / "acks"

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

    def archive(self, name: str) -> list[Path]:
        """Move a previous run's results under ``history/`` instead of overwriting them."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
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
        """Every ack on this spool, one per allocation it follows."""
        paths = sorted(self.acks.glob("*.json"))
        return [(path, record) for path, record in ((p, read_json(p)) for p in paths) if record]


#: Characters a hand-over file name keeps; anything else becomes ``_``.
_FILE_KEY = re.compile(r"[^A-Za-z0-9._-]")


def following_ack(
    acks: Iterable[tuple[Path, Mapping[str, object]]], allocation: str, session: str = ""
) -> Mapping[str, object] | None:
    """The ack among ``acks`` of a successor to ``allocation`` (:func:`ack_follows`), if any."""
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


def watcher_records(spool: str | Path) -> list[dict]:
    """Each watcher's latest record on ``spool``, from ``watchers/*.json`` and ``status.json``.

    Every reader outside the scheduler tells a live session from a gone one by this and
    :func:`watcher_age`.
    """
    paths = SpoolPaths(spool)
    found: dict[str, dict] = {}
    for record in (read_json(p) for p in [*sorted(paths.watchers.glob("*.json")), paths.snapshot]):
        if not record or "alive_epoch" not in record:
            continue
        key = str(record.get("watcher") or "")
        held = found.get(key)
        if held is None or float(record["alive_epoch"]) > float(held["alive_epoch"]):
            found[key] = record
    return list(found.values())


def latest_watcher(spool: str | Path) -> dict:
    """The newest watcher record, including a successor not yet in status.json."""
    return max(watcher_records(spool), key=lambda row: float(row["alive_epoch"]), default={})


def watcher_age(spool: str | Path, now: float | None = None) -> float | None:
    """Seconds since any watcher on ``spool`` last heart-beat; ``None`` if none ever did."""
    now = time.time() if now is None else now
    beats = [float(record["alive_epoch"]) for record in watcher_records(spool)]
    return now - max(beats) if beats else None


def ack_follows(ack: Mapping[str, object] | None, allocation: str, session: str = "") -> bool:
    """Whether ``ack`` names a successor to ``allocation``, the allocation of ``session``.

    An allocation outside Slurm (``unknown``) names no one allocation; an ack for
    ``allocation``'s own job is the ack of the hand-over it came from; an ack that names another
    session follows none of this one's.
    """
    if not ack or allocation in ("", "unknown") or str(ack.get("job_id") or "") == allocation:
        return False
    named = str(ack.get("session") or "")
    if named and session and named != session:
        return False
    return str(ack.get("succeeds") or "") == allocation


# --------------------------------------------------------------------------- scheduler


@dataclass
class SchedulerConfig:
    """How one watcher runs its jobs."""

    #: Device ids this watcher hands out, as ``CUDA_VISIBLE_DEVICES`` spells them.
    devices: tuple[str, ...] = ()
    #: Wall limit for a GPU job that declares none. Device-free jobs default to none.
    default_wall_s: float = DEFAULT_WALL_S
    #: How long processes a finished job left behind may outlive it.
    leftover_grace_s: float = 10.0
    #: The session this watcher belongs to, for status and the operator loop.
    session: str = ""
    #: When the allocation ends (epoch seconds); ``None`` when unknown.
    ends_at: float | None = None
    #: How long before ``ends_at`` the watcher is in hand-over and stops admitting long jobs.
    handover_lead_s: float = 3600.0


@dataclass
class Hold:
    """Why a job that could otherwise start must wait."""

    reason: str


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

    ``clock`` is injectable so limits can be tested without waiting; ``environ`` is what jobs
    inherit, before scrubbing.
    """

    def __init__(
        self,
        spool: str | Path,
        config: SchedulerConfig | None = None,
        *,
        clock: Callable[[], float] = time.time,
        environ: Mapping[str, str] | None = None,
        watcher_id: str | None = None,
    ) -> None:
        self.paths = SpoolPaths(spool)
        self.config = config or SchedulerConfig()
        self.clock = clock
        self.environ = dict(os.environ if environ is None else environ)
        self.host = socket.gethostname()
        self.slurm_job = self.environ.get("SLURM_JOB_ID", "unknown")
        #: Which start of a requeued allocation this is; ``None`` outside Slurm.
        self.restart = restart_count(self.environ)
        self.watcher_id = watcher_id or f"{self.host}-{os.getpid()}"
        self.started = self.clock()
        self.jobs: dict[str, RunningJob] = {}
        self.pending_reasons: dict[str, str] = {}
        self._spec_cache: dict[str, tuple[float, JobSpec]] = {}
        self._last_pending: list[JobSpec] = []
        #: Set once the allocation is within ``handover_lead_s`` of its end: since when,
        #: and the successor's watcher once one is up.
        self.handover: dict = {}
        #: Free device -> when a reading first found it over ``DEVICE_MAX_USED_MIB``.
        self._device_held_since: dict[str, float] = {}
        #: Device -> ``used_mib`` and ``since`` (first found over) for memory never released.
        self._device_quarantine: dict[str, dict] = {}
        #: This pass's device reading, taken at most once a pass (see _read_device_memory).
        self._device_reading: Mapping[str, int] | None = None
        self._device_read = False
        self._device_read_epoch = 0.0
        #: This pass's native locations by job tree (see _admission_hold).
        self._native_seen: dict[str, list[str]] = {}
        self.paths.ensure()

    # ------------------------------------------------------------- lifecycle

    def announce(self) -> None:
        """Write the startup lines of the status file and the first heartbeat."""
        lines = [
            f"host={self.host}",
            f"pid={os.getpid()}",
            f"watcher={self.watcher_id}",
            f"started={utc_iso(self.started)}",
            f"spool={self.paths.root}",
            f"session={self.config.session or '-'}",
            f"devices={','.join(self.config.devices) or '-'}",
            f"CUDA_VISIBLE_DEVICES={self.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
            f"watcher: ready at {utc_iso()}",
        ]
        write_text_atomic(self.paths.status, "\n".join(lines) + "\n")
        self.heartbeat()

    def serve(self) -> int:
        """Run until a successor is up and every job this watcher started is gone."""
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
            if self.handover.get("successor") and not self.jobs:
                self._append_status(
                    f"watcher: handed over to {self.handover['successor']}, exiting at {utc_iso()}"
                )
                (self.paths.watchers / f"{self.watcher_id}.json").unlink(missing_ok=True)
                return 0
            time.sleep(POLL_S)

    def _append_status(self, line: str) -> None:
        with self.paths.status.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    # ------------------------------------------------------------------ tick

    def tick(self) -> None:
        """One scheduling pass: reap, enforce limits, admit, report."""
        now = self.clock()
        table = read_proc_table()
        for job in list(self.jobs.values()):
            self._advance(job, now, table)
        self._recover_orphans(now, table)
        self._plan_handover(now)
        self._admit(now)
        self.heartbeat()

    def heartbeat(self) -> None:
        """Write this watcher's status snapshot: its record, and the spool's ``status.json``."""
        snapshot = self.snapshot(self.clock())
        write_json_atomic(self.paths.watchers / f"{self.watcher_id}.json", snapshot)
        write_json_atomic(self.paths.snapshot, snapshot)

    def snapshot(self, now: float) -> dict:
        """What this watcher is doing, for status readers and the operator loop."""
        allocated = self._allocated()
        quarantined = self._quarantined()
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
                    "reruns": spec.reruns,
                }
                for spec in self._last_pending
            ],
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
            "owner": job.spec.owner,
            "verdict": None if job.verdict is None else job.verdict.reason,
            "idle_s": round(now - max(job.last_progress, job.started), 1),
            "failure_line": job.failure_line,
            "unsafe_pids": list(job.unsafe_pids),
            "reruns": job.spec.reruns,
        }

    # ------------------------------------------------------------- admission

    def _effective_priority(self, spec: JobSpec, now: float) -> int:
        return spec.priority + int(max(0.0, now - spec.submitted) // AGING_S)

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
        held = {d for job in self.jobs.values() if job.unsafe_pids for d in job.devices}
        return held | set(self._device_quarantine)

    def _free_devices(self) -> list[str]:
        """Devices no job of this watcher holds and no quarantine keeps, in the order handed out."""
        taken = self._allocated() | set(self._device_quarantine)
        return [d for d in self.config.devices if d not in taken]

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
            self._start(spec, devices, now)

    def _devices_for(self, spec: JobSpec, free: list[str]) -> tuple[str, ...] | None:
        """The devices ``spec`` would get now, or ``None`` if it cannot start yet."""
        if spec.whole:
            busy = any(job.spec.holds_gpus for job in self.jobs.values())
            return tuple(free) if len(free) == len(self.config.devices) and not busy else None
        need = int(spec.gpus or 0)
        if len(free) >= need and not any(job.spec.whole for job in self.jobs.values()):
            return tuple(free[:need])
        return None

    def _place(self, spec: JobSpec, now: float) -> "tuple[str, ...] | Hold":
        """The devices a GPU job starts on now, or why it waits for devices.

        Only free devices are read against :data:`DEVICE_MAX_USED_MIB` (running jobs' memory is
        on the devices they hold). Only a reading returns a quarantined device, so a job that only
        a quarantined device would place takes a reading first.
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
        held = {d: reading[d] for d in free if reading.get(d, 0) > DEVICE_MAX_USED_MIB}
        for device in set(self._device_held_since) - set(held):
            del self._device_held_since[device]
        for device in held:
            self._device_held_since.setdefault(device, now)
        overdue = {
            device: used
            for device, used in held.items()
            if now - self._device_held_since[device] >= DEVICE_RELEASE_WAIT_S
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
            + f" > {DEVICE_MAX_USED_MIB:.0f} MiB (waiting {now - since:.0f}s of "
            f"{DEVICE_RELEASE_WAIT_S:.0f}s for it to be released)"
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
            if used > DEVICE_MAX_USED_MIB:
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
                    f"{utc_iso(float(marker.get('started', now)))} on {marker.get('host')}" + linked
                )
        return None

    def _native_locations(self, tree: str) -> list[str]:
        """Where ``tree``'s native build really is, resolved once a pass."""
        if tree not in self._native_seen:
            self._native_seen[tree] = _rebuild.native_locations(tree)
        return self._native_seen[tree]

    # -------------------------------------------------------------- hand-over

    def _plan_handover(self, now: float) -> None:
        """Within ``handover_lead_s`` of the allocation's end: say so, and notice a successor.

        A successor is a live watcher of this spool whose allocation outlasts this one.
        """
        ends_at = self.config.ends_at
        if ends_at is None or ends_at - now > self.config.handover_lead_s:
            return
        if not self.handover:
            self.handover["since"] = utc_iso(now)
            self._append_status(f"watcher: allocation ends {utc_iso(ends_at)}; in hand-over")
        for peer in self.paths.live_watchers(WATCHER_STALE_S, now):
            if peer.get("watcher") == self.watcher_id or not _ends_after(peer, ends_at):
                continue
            if self.handover.get("successor") != peer.get("watcher"):
                self._append_status(f"watcher: successor {peer.get('watcher')} is up")
            self.handover["successor"] = peer.get("watcher")
            return

    def _handover_hold(self, spec: JobSpec, now: float) -> "Hold | None":
        ends_at = self.config.ends_at
        if ends_at is None or ends_at - now > self.config.handover_lead_s:
            return None
        successor = self.handover.get("successor")
        if successor:
            return Hold(f"this session is handing over to {successor}")
        remaining = ends_at - now
        wall = float(spec.wall_s or 0)
        if spec.holds_gpus and (wall <= 0 or wall + HANDOVER_MARGIN_S > remaining):
            return Hold(
                f"the allocation ends in {remaining:.0f}s, too soon for a job allowed "
                f"{'unlimited time' if wall <= 0 else f'{wall:.0f}s'}; it waits for the "
                f"successor session, which drains this spool"
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
                "MV2FLOW_SPOOL_SESSION_JOBID": self.slurm_job,
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
        if RESOLVE_HOOK_DIR.is_dir():
            python_path = [str(RESOLVE_HOOK_DIR), *python_path]
            env["MV2FLOW_RESOLVE_LOG"] = str(self.paths.resolve_log(spec.name))
            env["MV2FLOW_RESOLVE_MODULES"] = RESOLVE_MODULES
        if python_path:
            env["PYTHONPATH"] = os.pathsep.join(python_path)
        return env

    def _start(self, spec: JobSpec, devices: tuple[str, ...], now: float) -> None:
        pending = self.paths.pending / f"{spec.name}.sh"
        running = self.paths.running / f"{spec.name}.sh"
        try:
            os.rename(pending, running)
        except OSError:
            # Another watcher on this spool claimed it first.
            return
        self._spec_cache.pop(spec.name, None)
        self.paths.archive(spec.name)
        log_path = self.paths.log(spec.name)
        env = self._job_environment(spec, devices)
        header = (
            f"=== start {utc_iso(now)} on {self.host} job={spec.name} "
            f"gpus={spec.describe_gpus()} devices={','.join(devices) or '-'} "
            f"wall={int(spec.wall_s or 0)}s watcher={self.watcher_id} ===\n"
        )
        if spec.tree:
            header += f"=== tree {spec.tree} (working directory and first PYTHONPATH entry) ===\n"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write(header + "\n")
            log = log_path.open("ab")
            try:
                proc = subprocess.Popen(
                    ["bash", str(running)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=spec.tree or None,
                    env=env,
                    start_new_session=True,
                    preexec_fn=_disable_core_dumps,
                )
            finally:
                log.close()
        except OSError as exc:
            self._finalize_unstarted(spec, RC_LAUNCH_FAILED, "launch_failed", str(exc))
            return
        job = RunningJob(spec=spec, proc=proc, devices=devices, started=now, log_path=log_path)
        # The scheduler's own header is not the job's progress.
        job.log_size = len(header.encode("utf-8")) + 1
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
            },
        )

    # ----------------------------------------------------------------- limits

    def _advance(self, job: RunningJob, now: float, table: Mapping[int, ProcInfo]) -> None:
        leader_rc = job.proc.poll()
        pids = job_processes(job.proc.pid, table) if table else []
        if leader_rc is None and job.proc.pid not in pids:
            pids.append(job.proc.pid)
        job.pids = tuple(pids)
        job.rss_gb = sum(rss_bytes(pid) for pid in pids) / _GB

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
                table, str(self.paths.root), job.spec.name, self.watcher_id
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
            and now - job.term_sent_at >= KILL_GRACE_S
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
        if not job.failure_line:
            try:
                with job.log_path.open("rb") as handle:
                    handle.seek(max(job.log_size, size - _LOG_SCAN_BYTES))
                    chunk = handle.read(_LOG_SCAN_BYTES).decode("utf-8", errors="replace")
            except OSError:
                chunk = ""
            for line in chunk.splitlines():
                if _FAILURE_LINE.search(line):
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
        if job.failure_line and job.spec.holds_gpus and idle > FAILURE_STALL_S:
            return Verdict(
                RC_STALLED,
                "stalled",
                f"no progress for {idle:.0f}s after a failure line ({job.failure_line!r}); "
                f"the other processes are presumed to be waiting on one that died",
            )
        return None

    def _signal(self, job: RunningJob, pids: Iterable[int], sig: int, now: float) -> None:
        """Signal a job's processes, honouring the safe-kill rule."""
        sent, refused = [], []
        for pid in pids:
            environ = read_environ(pid)
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
            "tree": spec.tree,
            "gpus": spec.describe_gpus(),
            "watcher": self.watcher_id,
            "host": self.host,
            "session": self.config.session,
            "slurm_job": self.slurm_job,
            "submitted": utc_iso(spec.submitted) if spec.submitted else None,
            "reruns": spec.reruns,
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
        record = self._record(
            job.spec,
            verdict,
            devices=list(job.devices),
            started=utc_iso(job.started),
            started_epoch=job.started,
            ended=utc_iso(now),
            duration_s=round(now - job.started, 3),
            wall_s=job.wall_s,
            unsafe_pids=list(job.unsafe_pids),
        )
        self._finish_files(job.spec.name, record, job.log_path, now)
        job.receipt_written = True

    def _holds(self, name: str) -> bool:
        """Whether the run of ``name`` in ``running/`` is still this watcher's to close.

        Not once another watcher found it lost while this one was silent: that watcher took it
        out of ``running/`` (to re-queue or close it), or a re-run of it there names another.
        """
        meta = read_json(self.paths.running / f"{name}.json")
        if meta is not None:
            return meta.get("watcher") == self.watcher_id
        return (self.paths.running / f"{name}.sh").exists()

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
        """Close out a job this watcher did not run: refused, never launched, or lost."""
        now = self.clock()
        record = self._record(spec, Verdict(rc, reason, detail), ended=utc_iso(now), **extra)
        log_path = self.paths.log(spec.name)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"=== scheduler: {reason}: {detail} ===\n")
        self._finish_files(spec.name, record, log_path, now)

    def _refuse(
        self, spec: JobSpec, detail: str, *, rc: int = RC_REFUSED, reason: str = "refused"
    ) -> None:
        pending = self.paths.pending / f"{spec.name}.sh"
        running = self.paths.running / f"{spec.name}.sh"
        if running.exists():
            return
        try:
            os.rename(pending, running)
        except OSError:
            return
        self._spec_cache.pop(spec.name, None)
        self.paths.archive(spec.name)
        self._finalize_unstarted(spec, rc, reason, detail)
        (self.paths.cancel / spec.name).unlink(missing_ok=True)

    def _release(self, job: RunningJob) -> None:
        now = self.clock()
        if not job.receipt_written:
            self._write_receipt(job, now)
        self.jobs.pop(job.spec.name, None)
        (self.paths.cancel / job.spec.name).unlink(missing_ok=True)

    # ---------------------------------------------------------------- orphans

    def _recover_orphans(self, now: float, table: Mapping[int, ProcInfo]) -> None:
        """Run again, or close out, the jobs a watcher that is gone left in ``running/``.

        One whose processes still run on this host (``table``) lost only its
        watcher, and one no watcher recorded starting has nothing to say where
        it ran; both are closed out as ``orphaned``.
        """
        fresh = {record.get("watcher") for record in self.paths.live_watchers(WATCHER_STALE_S, now)}
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
                if idle < WATCHER_STALE_S or len(fresh) > 1:
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
            alive = processes_with_marker(table, str(self.paths.root), name, owner) if owner else []
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

        Slurm ends every process of a job's start before starting it again, so a run recorded
        under a lower ``SLURM_RESTART_COUNT`` of this job is gone even while its watcher's last
        heartbeat is fresh. A run that names this watcher yet is not one of its jobs was started
        by an earlier watcher of that name (the same host and pid, after a reboot).
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
        self._finalize_unstarted(spec, RC_ORPHANED, "orphaned", detail, **ran)

    def _lost(self, name: str, script: Path, meta: Mapping[str, Any], why: str, now: float) -> None:
        """A run lost with its allocation: queue the job again, or close it out ``node_lost``."""
        spec = self._spec_of(name, script, meta)
        if (self.paths.cancel / name).exists():
            stop = "its cancel was requested"
        elif spec.reruns >= MAX_RERUNS:
            stop = f"it already ran again {spec.reruns} time(s), all it may"
        elif (self.paths.pending / f"{name}.sh").exists():
            stop = "a job of the same name is queued"
        else:
            stop = ""
        if stop:
            self._finalize_unstarted(
                spec,
                RC_NODE_LOST,
                "node_lost",
                f"{why}; not run again: {stop}",
                started=meta.get("started"),
                devices=list(meta.get("devices") or []),
            )
            (self.paths.cancel / name).unlink(missing_ok=True)
            self._append_status(f"watcher: {name} closed out node_lost at {utc_iso(now)}: {stop}")
            return
        # The rename is the claim: another watcher of the spool finds the script gone.
        claimed = self.paths.running / f".{name}.{self.watcher_id}.lost"
        try:
            os.rename(script, claimed)
        except FileNotFoundError:
            return
        submitted = claimed.stat()
        text = claimed.read_text(encoding="utf-8", errors="replace")
        claimed.write_text(stamp_rerun(text, spec.reruns + 1), encoding="utf-8")
        os.utime(claimed, ns=(submitted.st_atime_ns, submitted.st_mtime_ns))
        os.rename(claimed, self.paths.pending / f"{name}.sh")
        (self.paths.running / f"{name}.json").unlink(missing_ok=True)
        self._spec_cache.pop(name, None)
        note = f"{why}; queued again (re-run {spec.reruns + 1} of at most {MAX_RERUNS})"
        try:
            with self.paths.log(name).open("a", encoding="utf-8") as log:
                log.write(f"=== scheduler: {note} ===\n")
        except OSError:
            pass
        self._append_status(f"watcher: {name} lost at {utc_iso(now)}: {note}")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scheduler.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    watch = sub.add_parser(
        "watch", help="become the spool's scheduler (the allocation's main process)"
    )
    watch.add_argument("spool", type=Path)
    watch.add_argument(
        "--gpus", default="auto", help="'auto', a device count, or an explicit comma list of ids"
    )
    watch.add_argument("--session", default="", help="session name, for status and hand-over")
    watch.add_argument("--default-wall", default="2h", help="wall limit for GPU jobs that set none")
    watch.add_argument(
        "--time-limit",
        help="the allocation's time limit; with SLURM_JOB_START_TIME it fixes the end",
    )
    watch.add_argument("--handover-lead", default="1h", help="how long before the end to hand over")
    return parser


def allocation_end(args: argparse.Namespace, environ: Mapping[str, str]) -> float | None:
    """When this allocation ends: Slurm's own figure, else the declared limit from its start."""
    value = environ.get("SLURM_JOB_END_TIME", "")
    if value.isdigit() and int(value) > 0:
        return float(value)
    if args.time_limit:
        start = environ.get("SLURM_JOB_START_TIME", "")
        begun = float(start) if start.isdigit() else time.time()
        return begun + parse_duration(args.time_limit)
    return None


def config_from_args(args: argparse.Namespace, environ: Mapping[str, str]) -> SchedulerConfig:
    return SchedulerConfig(
        devices=discover_devices(args.gpus, environ),
        default_wall_s=parse_duration(args.default_wall),
        session=args.session,
        ends_at=allocation_end(args, environ),
        handover_lead_s=parse_duration(args.handover_lead),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scheduler = Scheduler(args.spool, config_from_args(args, os.environ))

    def _terminated(signum: int, frame: object) -> None:
        del frame
        scheduler._append_status(f"watcher: signal {signum} at {utc_iso()}, exiting")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _terminated)
    scheduler.announce()
    return scheduler.serve()


__all__ = [
    "ALL_GPUS",
    "AGING_S",
    "DEFAULT_WALL_S",
    "DEVICE_MAX_USED_MIB",
    "DEVICE_RELEASE_WAIT_S",
    "FAILURE_STALL_S",
    "FREE_PREFIX",
    "HANDOVER_DIRNAME",
    "HANDOVER_MARGIN_S",
    "HEADER_KEYS",
    "HEADER_TAG",
    "JobSpec",
    "KILL_GRACE_S",
    "MAX_RERUNS",
    "NAME_PATTERN",
    "POLL_S",
    "RC_CANCELLED",
    "RC_LAUNCH_FAILED",
    "RC_MIXED_CHECKOUT",
    "RC_NODE_LOST",
    "RC_ORPHANED",
    "RC_REFUSED",
    "RC_STALLED",
    "RC_WALL_LIMIT",
    "RERUN_TAG",
    "RESOLVE_HOOK_DIR",
    "RunningJob",
    "SESSION_LOSS_REASONS",
    "Scheduler",
    "SchedulerConfig",
    "SpoolPaths",
    "TIMEOUT_REASONS",
    "Verdict",
    "WATCHER_STALE_S",
    "ack_follows",
    "following_ack",
    "is_mpi_attached",
    "job_processes",
    "job_spec_from_script",
    "parse_duration",
    "parse_header",
    "read_device_memory_mib",
    "read_environ",
    "read_proc_table",
    "reruns_of",
    "restart_count",
    "scrub_environment",
    "stamp_rerun",
    "utc_iso",
    "watcher_age",
    "watcher_records",
]


if __name__ == "__main__":
    sys.exit(main())
