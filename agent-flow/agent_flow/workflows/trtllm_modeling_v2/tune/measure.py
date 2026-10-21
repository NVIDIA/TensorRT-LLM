# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurements: ``bench.py`` runs in whole-node jobs on spool sessions, queued and collected.

A measurement starts only on an idle node: nothing runs there, whoever's job it
would be, and the last job there ended at least :data:`IDLE_SETTLE_S` ago (a
server's teardown and the host's load outlast the job before it).

A measurement over several perf configs is one :class:`Run` per config. A
candidate's runs share one job, so all of them are judged on its host; the
stage's own measurements (ground truth, the wrap-up) spread their runs over the
nodes (:meth:`BenchSpec.runs_command`), and :func:`write_index` joins them into
the label once all are on disk.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..infra.substrate import placement, spoolctl
from ..infra.substrate.scheduler import (
    RC_HOST_MEMORY,
    RC_LAUNCH_FAILED,
    RC_NODE_LOST,
    RC_ORPHANED,
    RC_STALLED,
    parse_duration,
)
from .records import PARTS_FILE, Measurement


def load_measurement(data: Path, label: str) -> Measurement | None:
    """The finished measurement under ``data/label``, or None while it is incomplete."""
    path = data / label
    if (path / "summary.json").exists() and (path / "meta.json").exists():
        return Measurement.load(path)
    return None


def spool_host(spool: str) -> str:
    """The host a session spool's scheduler last reported, or ``""``."""
    try:
        return str(json.loads((Path(spool) / "status.json").read_text()).get("host") or "")
    except (OSError, ValueError):
        return ""


#: Return codes the substrate gives a job that it, not the job, ended.
INFRA_RCS = frozenset({RC_STALLED, RC_ORPHANED, RC_HOST_MEMORY, RC_LAUNCH_FAILED, RC_NODE_LOST})
#: How long a node must have run nothing before a measurement starts there (seconds).
IDLE_SETTLE_S = 120.0


def node_busy(spool: str, *, settle_s: float, now: float) -> str:
    """Why ``spool``'s node cannot start a measurement now; ``""`` when it is idle.

    Read from the spool's status: a job a watcher claimed (``running/<name>.sh``), a job
    a live watcher's record (:func:`placement.watcher_records`) says runs, or a receipt
    written less than ``settle_s`` ago -- whoever's job it was.
    """
    root = Path(spool)
    running = {p.name[: -len(".sh")] for p in (root / "running").glob("*.sh")}
    for record in placement.watcher_records(root):
        if now - float(record["alive_epoch"]) > spoolctl.WATCHER_STALE_S:
            continue
        jobs: list[Any] = record.get("running") or []
        running.update(str(j.get("name")) for j in jobs if isinstance(j, Mapping) and j.get("name"))
    if running:
        return "running " + ", ".join(sorted(running))
    if settle_s > 0:
        ended = 0.0
        for receipt in (root / "done").glob("*.rc"):
            try:
                ended = max(ended, receipt.stat().st_mtime)
            except OSError:  # set aside meanwhile
                continue
        if now - ended < settle_s:
            return f"a job ended {max(now - ended, 0.0):.0f}s ago (idle from {settle_s:.0f}s)"
    return ""


def receipt_rc(path: Path) -> int | None:
    """The return code a job's receipt records, or None when it has none (or no number yet)."""
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class BenchPart:
    """One perf config of a campaign over several: its workload and its own base configs."""

    name: str
    isl: int
    osl: int
    concurrencies: Sequence[int]
    configs: Sequence[Path]
    constraint: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Run:
    """One server session of a measurement: ``label`` at its perf config ``part``.

    ``part`` is empty for a measurement of one config: its one run is the label.
    """

    label: str
    part: str = ""

    @property
    def name(self) -> str:
        """``t-baseline.tp1``; the label itself for a measurement of one config."""
        return f"{self.label}.{self.part}" if self.part else self.label


@dataclass
class BenchSpec:
    """How to run ``bench.py`` inside a session: the same for every measurement.

    With ``parts``, one measurement is one ``bench.py`` run per part, each at
    the part's workload and base configs followed by the configs the caller
    passes (the overlays), in one job, under ``<label>/<part>``; ``parts.json``
    names them and their constraints (see ``records.Measurement``).
    """

    bench_py: str
    model: str
    env_script: str
    isl: int
    osl: int
    concurrencies: Sequence[int]
    tree: str
    extra_args: Sequence[str] = ()
    wall: str = "2h"
    parts: Sequence[BenchPart] = ()

    def command(self, label: str, out_root: Path, configs: Sequence[Path]) -> str:
        if not self.parts:
            return self._one(label, out_root, configs, self.isl, self.osl, self.concurrencies)
        root = Path(out_root) / label
        steps = [
            self._one(part.name, root, [*part.configs, *configs], part.isl, part.osl,
                      part.concurrencies, source=False)
            for part in self.parts
        ]  # fmt: skip
        index = {
            "parts": [part.name for part in self.parts],
            "constraints": {part.name: dict(part.constraint) for part in self.parts},
        }
        # The label's meta is its first part's (host, tree, modeling_v2), relabelled:
        # a control is matched to a measurement by the host it ran on.
        merge = (
            "import json,sys; r=sys.argv[1]; m=json.load(open(r+'/'+sys.argv[2]+'/meta.json')); "
            "m.update(label=sys.argv[3], parts=json.loads(sys.argv[4])['parts']); "
            "json.dump(m, open(r+'/meta.json','w')); open(r+'/summary.json','w').write('[]'); "
            f"open(r+'/{PARTS_FILE}','w').write(sys.argv[4])"
        )
        steps.append(
            " ".join(
                shlex.quote(a)
                for a in [
                    "python",
                    "-c",
                    merge,
                    str(root),
                    self.parts[0].name,
                    label,
                    json.dumps(index),
                ]
            )
        )
        return f"source {shlex.quote(self.env_script)}\n" + " && \\\n".join(steps)

    def runs(self, label: str) -> list[Run]:
        """``label``'s runs: one per perf config, or the label itself over one config."""
        return [Run(label, part.name) for part in self.parts] or [Run(label)]

    def runs_command(
        self, runs: Sequence[Run], out_root: Path, configs: Mapping[str, Sequence[Path]]
    ) -> str:
        """One job that measures ``runs`` one after another, each where :meth:`command` puts it.

        A run of a perf config serves that config's workload and base configs followed by
        its label's ``configs[label]``, under ``<label>/<part>``; a run of one config is
        :meth:`command`'s. The label's own files are not written here (:func:`write_index`).
        """
        parts = {part.name: part for part in self.parts}
        steps = []
        for run in runs:
            if not run.part:
                steps.append(self._one(run.label, out_root, configs[run.label], self.isl,
                                       self.osl, self.concurrencies, source=False))  # fmt: skip
                continue
            part = parts[run.part]
            steps.append(self._one(part.name, Path(out_root) / run.label,
                                   [*part.configs, *configs[run.label]], part.isl, part.osl,
                                   part.concurrencies, source=False))  # fmt: skip
        return f"source {shlex.quote(self.env_script)}\n" + " && \\\n".join(steps)

    def _one(
        self,
        label: str,
        out_root: Path,
        configs: Sequence[Path],
        isl: int,
        osl: int,
        concurrencies: Sequence[int],
        *,
        source: bool = True,
    ) -> str:
        argv = [
            "python",
            self.bench_py,
            "--model",
            self.model,
            "--label",
            label,
            "--out-root",
            str(out_root),
        ]
        for cfg in configs:
            argv += ["--config", str(cfg)]
        argv += ["--isl", str(isl), "--osl", str(osl)]
        argv += [
            "--concurrency",
            ",".join(str(c) for c in concurrencies),
            "--tree",
            self.tree,
            "--force",
        ]
        argv += list(self.extra_args)
        line = " ".join(shlex.quote(a) for a in argv)
        return f"source {shlex.quote(self.env_script)}\n{line}" if source else line


def write_index(out_root: Path, label: str, parts: Sequence[BenchPart]) -> None:
    """Join ``label``'s runs, each on disk, into one measurement, as :meth:`BenchSpec.command` does.

    The label's meta is its first part's, relabelled, with the host each part ran on
    (``hosts``). Its ``hostname`` is theirs when they share one and empty otherwise: a
    label measured across nodes is no host's control.
    """
    root = Path(out_root) / label
    hosts = {
        part.name: str(
            json.loads((root / part.name / "meta.json").read_text()).get("hostname") or ""
        )
        for part in parts
    }
    meta = json.loads((root / parts[0].name / "meta.json").read_text())
    shared = set(hosts.values())
    meta.update(label=label, parts=list(hosts), hosts=hosts,
                hostname=shared.pop() if len(shared) == 1 else "")  # fmt: skip
    index = {"parts": list(hosts), "constraints": {p.name: dict(p.constraint) for p in parts}}
    (root / PARTS_FILE).write_text(json.dumps(index))
    (root / "summary.json").write_text("[]")
    # Last: a label dir with its meta.json is a finished measurement (:func:`load_measurement`).
    (root / "meta.json").write_text(json.dumps(meta))


#: The label of a same-host control joined from several labels' runs: ``t-ground-<host>-<n>``.
GROUND_CONTROL = "t-ground"


def host_controls(out_root: Path, labels: Sequence[str], parts: Sequence[str]) -> list[Path]:
    """The same-host controls ``labels`` make: per host, one for each time it ran every part.

    ``labels`` measured one thing (one tree, one config), each run on whichever node was
    idle. A host's ``n``-th control takes, for each part, the ``n``-th of these labels that
    ran it there. A control whose parts all come from one label is that label; any other
    is a label dir of links to the runs it takes, ``t-ground-<host>-<n>``. A host that
    never ran some part makes none: its first control is the campaign's to measure.
    Without parts, each label on disk is its host's control.
    """
    root = Path(out_root)
    done = [label for label in labels if (root / label / "meta.json").exists()]
    if not parts:
        return [root / label for label in done]
    ran: dict[str, dict[str, list[str]]] = {}
    for label in done:
        for part in parts:
            try:
                meta = json.loads((root / label / part / "meta.json").read_text())
            except (OSError, ValueError):
                continue
            host = str(meta.get("hostname") or "")
            ran.setdefault(host, {}).setdefault(part, []).append(label)
    controls = []
    for host, by_part in ran.items():
        for n in range(min(len(by_part.get(part, ())) for part in parts)):
            taken = {part: by_part[part][n] for part in parts}
            if len(set(taken.values())) == 1:
                controls.append(root / taken[parts[0]])
            else:
                tag = re.sub(r"[^A-Za-z0-9._-]+", "_", host) or "host"
                controls.append(_join(root, f"{GROUND_CONTROL}-{tag}-{n + 1}", taken, host))
    return controls


def _join(root: Path, name: str, taken: Mapping[str, str], host: str) -> Path:
    """A label dir ``name`` whose part ``p`` is a link to ``<taken[p]>/<p>``: one host's runs."""
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    for part, label in taken.items():
        link = path / part
        if link.is_symlink():
            link.unlink()
        link.symlink_to(Path("..") / label / part, target_is_directory=True)
    first = next(iter(taken))
    index = json.loads((root / taken[first] / PARTS_FILE).read_text())
    index["parts"] = list(taken)
    meta = json.loads((path / first / "meta.json").read_text())
    meta.update(label=name, parts=list(taken), hostname=host, hosts={p: host for p in taken},
                runs={p: f"{label}/{p}" for p, label in taken.items()})  # fmt: skip
    (path / PARTS_FILE).write_text(json.dumps(index))
    (path / "summary.json").write_text("[]")
    (path / "meta.json").write_text(json.dumps(meta))
    return path


class SpoolRunner:
    """Queue jobs across sessions and collect them.

    A whole-node measurement goes to a session with nothing of ours in flight on an
    idle node (:meth:`free_spools`, :func:`node_busy` with ``settle_s``); the keep
    gate's jobs share sessions by device count (:meth:`place`). The substrate's own
    client (``spoolctl.queue_job``) does the queuing, so names, receipts and archiving
    follow the same protocol the gates use. ``earlier`` names the prefixes earlier
    attempts of the same work queued under.
    """

    def __init__(
        self, spools: Sequence[str], prefix: str = "tune", earlier: Sequence[str] = ()
    ) -> None:
        self.spools = list(spools)
        self.prefix = prefix
        self.earlier = list(earlier)
        self.settle_s = IDLE_SETTLE_S
        #: Why each live session was last passed over by :meth:`free_spools` (spool -> reason).
        self.not_idle: dict[str, str] = {}
        #: Each session's last :func:`node_busy` reading: spool -> (when, why).
        self._idle_read: dict[str, tuple[float, str]] = {}
        self.inflight: dict[str, tuple[str, str]] = {}  # job -> (spool, label)
        #: The devices each job in flight asked for (job -> count), for :meth:`place`.
        self.held: dict[str, int] = {}
        #: The last finished job of each label: label -> (spool, job), for :meth:`output`.
        self.finished: dict[str, tuple[str, str]] = {}

    def queued(self, label: str) -> bool:
        """A job of ``label`` is in flight: queued by this process, or adopted."""
        return any(job_label == label for _, job_label in self.inflight.values())

    def alive(self, spool: str) -> bool:
        """Whether ``spool``'s watcher has heart-beat recently."""
        age = spoolctl.watcher_age(spool)
        return age is not None and age <= spoolctl.WATCHER_STALE_S

    #: How long one reading of a node's status stands: one campaign step asks many times.
    IDLE_READ_S = 10.0

    def free_spools(self) -> list[str]:
        """Sessions with a live watcher, nothing of ours in flight, and an idle node."""
        busy = {spool for spool, _ in self.inflight.values()}
        free = []
        now = time.time()
        for spool in self.spools:
            if spool in busy or not self.alive(spool):
                continue
            when, why = self._idle_read.get(spool, (0.0, ""))
            if now - when > self.IDLE_READ_S:
                why = node_busy(spool, settle_s=self.settle_s, now=now)
                self._idle_read[spool] = (now, why)
            if why:
                self.not_idle[spool] = why
                continue
            self.not_idle.pop(spool, None)
            free.append(spool)
        return free

    def _views(self, gpus: int, wall: str) -> list[tuple[str, placement.SpoolView]]:
        job = placement.JobNeeds(
            name=self.prefix, gpus=gpus, wall_s=parse_duration(wall), holds_gpus=True
        )
        now = time.time()
        return [
            (spool, placement.assess(spool, job, stale_s=spoolctl.WATCHER_STALE_S, now=now))
            for spool in self.spools
        ]

    def place(self, gpus: int, wall: str = "1h") -> str | None:
        """The session to queue a ``gpus``-device job on now, or None when none can take it.

        Unlike :meth:`free_spools` (one whole-node measurement per session), jobs of this
        runner share a session. Of the sessions that can take the job (``placement``), one
        where this runner's own jobs leave room for it comes first, then the one running
        fewest of them, then the order ``spoolctl launch --pick`` ranks them in.
        """
        held: dict[str, int] = {}
        for name, (spool, _) in self.inflight.items():
            held[spool] = held.get(spool, 0) + self.held.get(name, 1)
        able = [(spool, view) for spool, view in self._views(gpus, wall) if view.pickable]
        if not able:
            return None
        spool, _ = min(
            able,
            key=lambda sv: (
                held.get(sv[0], 0) + gpus > sv[1].devices,
                held.get(sv[0], 0),
                placement.rank(sv[1]),
            ),
        )
        return spool

    def widest(self) -> int:
        """The most devices a live session of this runner hands out; 0 when none says."""
        return max((view.devices for _, view in self._views(1, "1h")), default=0)

    def submit(
        self,
        spool: str,
        label: str,
        command: str,
        *,
        wall: str = "2h",
        tree: str | None = None,
        gpus: str = "all",
    ) -> str:
        name = f"{self.prefix}-{label}"
        header = {"gpus": gpus, "wall": wall}
        if tree:
            header["tree"] = tree
        script = spoolctl.render_job(command, header=header)
        receipt = Path(spool) / "done" / f"{name}.rc"
        if receipt.exists():
            # A re-run of a label whose earlier run failed: its old receipt would
            # read as this run finishing at once.
            receipt.rename(receipt.with_name(f"{receipt.name}.superseded-{int(time.time())}"))
        self._idle_read.pop(spool, None)
        try:
            spoolctl.queue_job(spool, name, script)
        except spoolctl.SpoolError as exc:
            # Queued by an earlier process of this campaign: adopt it, never twice.
            if "already queued or running" not in str(exc) or not self.adopt(label):
                raise
            return name
        self.inflight[name] = (spool, label)
        self.held[name] = int(gpus) if str(gpus).isdigit() else 1 << 16
        return name

    def adopt(self, label: str) -> bool:
        """Re-register a job queued by an earlier process (resume after a restart).

        Only work that can still be this process's result: a job pending or running (the
        same work in flight) or a receipt of 0 (its result on disk). A failed receipt is
        never adopted: the label is owed a fresh run with retries of its own (:meth:`submit`
        sets the old receipt aside). An earlier attempt's job is adopted only while it is in
        flight (so the same measurement never runs twice); its receipts are that attempt's.
        """
        for prefix in (self.prefix, *self.earlier):
            name = f"{prefix}-{label}"
            for spool in self.spools:
                root = Path(spool)
                in_flight = any(
                    (root / sub / f"{name}.sh").exists() for sub in ("pending", "running")
                )
                if in_flight or (
                    prefix == self.prefix and receipt_rc(root / "done" / f"{name}.rc") == 0
                ):
                    self.inflight[name] = (spool, label)
                    return True
        return False

    def cancel(self, label: str) -> bool:
        """Ask the session to stop a job of ours (the scheduler's sanctioned kill) and forget it."""
        for name, (spool, job_label) in list(self.inflight.items()):
            if job_label == label:
                del self.inflight[name]
                self.held.pop(name, None)
                cancel_dir = Path(spool) / "cancel"
                cancel_dir.mkdir(parents=True, exist_ok=True)
                (cancel_dir / name).touch()
                return True
        return False

    def output(self, label: str) -> tuple[str, dict]:
        """What the last finished job of ``label`` printed (its log), and its receipt record."""
        spool, name = self.finished.get(label, ("", ""))
        if not spool:
            return "", {}
        log = Path(spool) / "logs" / f"{name}.log"
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        try:
            record = json.loads((Path(spool) / "done" / f"{name}.json").read_text())
        except (OSError, ValueError):
            record = {}
        return text, record if isinstance(record, dict) else {}

    #: A session silent this long is gone (its Slurm job ended): its jobs died with it.
    DEAD_SESSION_S = 600.0

    def poll(self) -> list[tuple[str, str, int]]:
        done = []
        for name, (spool, label) in list(self.inflight.items()):
            receipt = Path(spool) / "done" / f"{name}.rc"
            age = spoolctl.watcher_age(spool)
            if not receipt.exists() and age is not None and age > self.DEAD_SESSION_S:
                # Nobody will write this receipt until a new session adopts the spool;
                # set the job aside so that session does not run it again, and report
                # it lost so the caller re-queues it on a live session.
                for sub in ("pending", "running"):
                    script = Path(spool) / sub / f"{name}.sh"
                    if script.exists():
                        script.rename(script.with_name(f"{script.name}.lost-{int(time.time())}"))
                del self.inflight[name]
                self.held.pop(name, None)
                done.append((label, spool, RC_ORPHANED))
                continue
            if receipt.exists():
                try:
                    rc = int(receipt.read_text().strip() or "1")
                except ValueError:
                    continue
                del self.inflight[name]
                self.held.pop(name, None)
                self.finished[label] = (spool, name)
                done.append((label, spool, rc))
        return done


__all__ = [
    "GROUND_CONTROL",
    "IDLE_SETTLE_S",
    "INFRA_RCS",
    "BenchPart",
    "BenchSpec",
    "Run",
    "SpoolRunner",
    "host_controls",
    "load_measurement",
    "node_busy",
    "receipt_rc",
    "spool_host",
    "write_index",
]
