# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurements: ``bench.py`` runs in whole-node jobs on spool sessions, queued and collected.

A measurement starts only on an idle node: nothing runs there, whoever's job it would be, and the
last job there ended at least :data:`IDLE_SETTLE_S` ago (a server's teardown and the host's load
outlast the job before it). A measurement over several perf configs is one :class:`Run` per
config. A candidate's runs share one job, so all are judged on its host; the stage's own
measurements (ground truth, the wrap-up) spread their runs over the nodes, and
:func:`index_writer.write_index` joins them once all are on disk.
"""

from __future__ import annotations

import json
import math
import re
import shlex
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from ..infra.compute import placement, spoolctl
from ..infra.compute.scheduler import (
    RC_LAUNCH_FAILED,
    RC_NODE_LOST,
    RC_ORPHANED,
    RC_STALLED,
    latest_watcher,
    parse_duration,
    watcher_age,
    watcher_records,
)
from ..infra.git import git_checked
from .harness import index_writer
from .harness.bench import resolve_config
from .records import PARTS_FILE, Measurement, PerfLedger


def load_measurement(data: Path, label: str) -> Measurement | None:
    """The finished measurement under ``data/label``, or None while it is incomplete."""
    path = data / label
    if (path / "summary.json").exists() and (path / "meta.json").exists():
        try:
            return Measurement.load(path)
        except (KeyError, OSError, TypeError, ValueError):
            return None
    return None


def validate_measurement(
    m: Measurement, bench: "BenchSpec", configs: Sequence[Path], commit: str
) -> None:
    """Require complete serving metrics and matching revision/workload/config provenance."""
    expected = bench.workloads(configs)
    actual = m.parts or {"": m}
    if actual.keys() != expected.keys():
        raise ValueError("serving measurement has missing or extra perf configs")
    for name, part in actual.items():
        isl, osl, concurrencies, paths = expected[name]
        meta, tree = part.meta, part.meta.get("tree") or {}
        config = json.loads(json.dumps(resolve_config(list(paths), []), default=str))
        if tree.get("commit") != commit or tree.get("dirty_files"):
            raise ValueError("serving measurement is not of the expected clean revision")
        if (meta.get("isl"), meta.get("osl"), meta.get("concurrencies"), meta.get("config")) != (
            isl,
            osl,
            list(concurrencies),
            config,
        ):
            raise ValueError("serving measurement workload/config does not match")
        if not meta.get("hostname") or set(part.points) != set(concurrencies):
            raise ValueError("serving measurement lacks host or concurrency coverage")
        for row in part.points.values():
            for key in ("tok_s_user", "output_tok_s_gpu", "mean_tpot_ms", "mean_ttft_ms"):
                value = row.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"serving measurement has invalid {key}")


def spool_host(spool: str) -> str:
    """The host a session spool's scheduler last reported, or ``""``."""
    return str(latest_watcher(spool).get("host") or "")


#: Return codes the substrate gives a job that it, not the job, ended.
INFRA_RCS = frozenset({RC_STALLED, RC_ORPHANED, RC_LAUNCH_FAILED, RC_NODE_LOST})
#: How long a node must have run nothing before a measurement starts there (seconds).
IDLE_SETTLE_S = 120.0


def node_busy(spool: str, *, settle_s: float, now: float) -> str:
    """Why ``spool``'s node cannot start a measurement now; ``""`` when it is idle.

    Busy is a job in ``running/`` or in a live watcher's record, or a receipt written less than
    ``settle_s`` ago, whoever's job it was.
    """
    root = Path(spool)
    running = {p.name[: -len(".sh")] for p in (root / "running").glob("*.sh")}
    for record in watcher_records(root):
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

    With ``parts``, one measurement is one ``bench.py`` run per part, at the part's workload and
    base configs plus the caller's overlays, under ``<label>/<part>``; ``parts.json`` names them
    and their constraints (``records.Measurement``).
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

    def workloads(self, configs: Sequence[Path]) -> dict[str, tuple]:
        """Each run's workload and ordered config overlays, shared by launch and validation."""
        return {
            part.name: (part.isl, part.osl, part.concurrencies, [*part.configs, *configs])
            for part in self.parts
        } or {"": (self.isl, self.osl, self.concurrencies, list(configs))}

    def command(self, label: str, out_root: Path, configs: Sequence[Path]) -> str:
        command = self.runs_command(self.runs(label), out_root, {label: configs})
        if not self.parts:
            return command
        index = {
            "parts": [part.name for part in self.parts],
            "constraints": {part.name: dict(part.constraint) for part in self.parts},
        }
        merge_command = shlex.join(
            [
                "python",
                str(Path(index_writer.__file__).resolve()),
                "--out-root",
                str(out_root),
                "--label",
                label,
                "--index",
                json.dumps(index),
            ]
        )
        return command + " && " + merge_command

    def runs(self, label: str) -> list[Run]:
        """``label``'s runs: one per perf config, or the label itself over one config."""
        return [Run(label, part.name) for part in self.parts] or [Run(label)]

    def runs_command(
        self, runs: Sequence[Run], out_root: Path, configs: Mapping[str, Sequence[Path]]
    ) -> str:
        """One job that measures ``runs`` one after another, each where :meth:`command` puts it.

        A run's configs are its part's then ``configs[label]``. The label's own files are
        :func:`index_writer.write_index`'s.
        """
        steps = []
        for run in runs:
            isl, osl, concurrencies, paths = self.workloads(configs[run.label])[run.part]
            steps.append(
                self._one(
                    run.part or run.label,
                    Path(out_root) / run.label if run.part else out_root,
                    paths,
                    isl,
                    osl,
                    concurrencies,
                )
            )
        return f"source {shlex.quote(self.env_script)}\n" + " && \\\n".join(steps)

    def _one(
        self,
        label: str,
        out_root: Path,
        configs: Sequence[Path],
        isl: int,
        osl: int,
        concurrencies: Sequence[int],
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
        return line


#: The label of a same-host control joined from several labels' runs: ``t-ground-<host>-<n>``.
GROUND_CONTROL = "t-ground"


def host_controls(out_root: Path, labels: Sequence[str], parts: Sequence[str]) -> list[Path]:
    """The same-host controls ``labels`` make: per host, one for each time it ran every part.

    ``labels`` measured one tree at one config, each run on whichever node was idle. A host's
    ``n``-th control takes, per part, the ``n``-th label that ran it there: that label if it
    gave every part, else a label dir of links to the runs (``t-ground-<host>-<n>``). A host that
    never ran some part makes none. Without parts, each label on disk is its host's control.
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
    """Queue jobs across sessions, through ``spoolctl.queue_job``, and collect them.

    A whole-node measurement goes to a session with nothing of ours in flight on an idle node
    (:meth:`free_spools`); the keep gate's jobs share sessions by device count (:meth:`place`).
    ``earlier`` names the prefixes earlier attempts of the same work queued under.
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
        age = watcher_age(spool)
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
        job = placement.JobNeeds(gpus=gpus, wall_s=parse_duration(wall), holds_gpus=True)
        now = time.time()
        return [
            (spool, placement.assess(spool, job, stale_s=spoolctl.WATCHER_STALE_S, now=now))
            for spool in self.spools
        ]

    def place(self, gpus: int, wall: str = "1h") -> str | None:
        """The session to queue a ``gpus``-device job on now, or None when none can take it.

        These jobs share a session. Of those that can take the job, one where this runner's jobs
        leave room comes first, then the one running fewest, then :func:`placement.rank`'s order.
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

        Only a job pending or running, or a receipt of 0. A failed receipt is never adopted: the
        label is owed a fresh run with retries of its own. An earlier attempt's job is adopted
        only while in flight, so a measurement never runs twice; its receipts are that attempt's.
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
        """Ask the session to stop a job of ours (the sanctioned kill), and forget it."""
        for name, (spool, job_label) in list(self.inflight.items()):
            if job_label == label:
                del self.inflight[name]
                self.held.pop(name, None)
                spoolctl.cancel_job(spool, name)
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

    def poll(self) -> list[tuple[str, str, int]]:
        done = []
        for name, (spool, label) in list(self.inflight.items()):
            receipt = Path(spool) / "done" / f"{name}.rc"
            age = watcher_age(spool)
            if not receipt.exists() and age is not None and age > spoolctl.WATCHER_STALE_S:
                # Nobody writes this receipt until a new session adopts the spool: set the job
                # aside so that session does not run it again, and report it lost for a re-queue.
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


def _differs(path: Path, bench: BenchSpec, configs: Sequence[Path], tree: str | None) -> str:
    """How the measurement under ``path`` differs from what ``bench`` measures now, or ``""``.

    Compares each part's ``meta.json``: the workload, the merged LLM-API config, and the commit
    of ``tree``. A field not recorded, or configs that no longer read, are not held against it.
    """
    commit = git_checked(Path(tree), "rev-parse", "HEAD", check=False) if tree else ""
    for name, (isl, osl, concurrencies, paths) in bench.workloads(configs).items():
        where = path / name
        try:
            meta = json.loads((where / "meta.json").read_text())
        except (OSError, ValueError):
            return f"has no readable {where.name}/meta.json"
        for key, want in (("isl", isl), ("osl", osl), ("concurrencies", list(concurrencies))):
            if key in meta and meta[key] != want:
                return f"was measured at {key} {meta[key]}, not {want}"
        if "config" in meta:
            try:
                # The same round trip bench.py's meta.json took.
                config = json.loads(json.dumps(resolve_config(paths, []), default=str))
            except (OSError, yaml.YAMLError):
                config = meta["config"]
            if meta["config"] != config:
                return f"was measured at another config than {[p.name for p in paths]}"
        measured = (meta.get("tree") or {}).get("commit")
        if commit and measured and measured != commit:
            return f"was measured at {str(measured)[:10]}, not {commit[:10]}"
    return ""


def _key(run: Run, labels: Mapping[str, Sequence[Path]]) -> tuple[tuple[str, ...], str]:
    """What ``run`` measures: its label's configs and its member. Repeats share it."""
    return tuple(map(str, labels[run.label])), run.part


class MeasurementRunner:
    """Measure labels and resume their jobs using explicitly supplied campaign state."""

    def __init__(
        self,
        *,
        bench: BenchSpec,
        data: Path,
        runner: SpoolRunner,
        state: dict,
        failed: dict[str, tuple[str, int, str]],
        ledger: PerfLedger,
        checkpoint: Callable[[], None],
        log: Callable[[str], None],
        poll_s: float,
    ) -> None:
        self.bench, self.data, self.perf = bench, data, runner
        self.t, self.failed, self.ledger = state, failed, ledger
        self.checkpoint, self.log, self.poll_s = checkpoint, log, poll_s

    def measure(
        self,
        labels: Mapping[str, Sequence[Path]],
        tree: str | None = None,
        *,
        bench: BenchSpec | None = None,
        phase: str = "",
    ) -> dict[str, Measurement | None]:
        """Measure every label not yet on disk, its runs spread over the idle nodes.

        A label is one run per perf config (:meth:`BenchSpec.runs`); each idle node takes its
        share of the runs left as one job (:meth:`_share`). Failed runs are recorded,
        never automatically repeated.
        """
        bench = bench or self.bench
        todo: list[Run] = []
        for label, configs in labels.items():
            if not self._reusable(label, configs, tree, bench):
                todo += [
                    r for r in bench.runs(label) if not self._run_reusable(r, configs, tree, bench)
                ]
        queued = self._adopt_runs(todo, labels)
        while todo:
            held = {run for runs in queued.values() for run in runs}
            waiting = [run for run in todo if run not in held]
            free = self.perf.free_spools() if waiting else []
            while waiting and free:
                # The node most of the waiting members are new to chooses first.
                ran = {sp: self._ran(sp, labels, bench, queued) for sp in free}
                spool = max(
                    free, key=lambda sp: sum(not ran[sp].get(_key(r, labels)) for r in waiting)
                )
                free.remove(spool)
                runs = self._share(waiting, ran[spool], labels, bench, queued)
                job = runs[0].name
                self.perf.submit(spool, job, bench.runs_command(runs, self.data, labels),
                                 wall=bench.wall, tree=tree)  # fmt: skip
                queued[job] = runs
                waiting = [run for run in waiting if run not in runs]
                self.log(f"[tune] measuring {', '.join(r.name for r in runs)} on {spool}")
            self._note_queued(queued)
            for job, spool, rc in self.perf.poll():
                runs = queued.pop(job, [])
                missing = [run for run in runs if not self._measured(run)]
                if rc != 0:
                    for run in runs:
                        self.failed[run.label] = (str(spool), rc, job)
                    todo = [run for run in todo if run.label not in {r.label for r in runs}]
                if missing:
                    # The job's runs go in order: its first missing one is the run that failed
                    # (a session that died, a full filesystem -- or an adopted receipt whose
                    # results are gone, rc 0), and the runs after it never started.
                    failed = missing[0]
                    self.log(f"[tune] {failed.name} failed rc={rc}")
                    self.failed[failed.label] = (str(spool), rc, job)
                    todo = [run for run in todo if run.label != failed.label]
                todo = [run for run in todo if run not in runs or run in missing]
            if todo:
                time.sleep(self.poll_s)
        self._note_queued({})
        out = {}
        for label in labels:
            path = self.data / label
            runs = bench.runs(label)
            if bench.parts and not (path / "meta.json").exists() and all(map(self._measured, runs)):
                index_writer.write_index(
                    self.data,
                    label,
                    [p.name for p in bench.parts],
                    {p.name: dict(p.constraint) for p in bench.parts},
                )
            out[label] = None if label in self.failed else load_measurement(self.data, label)
            if out[label]:
                self.failed.pop(label, None)
                hosts = out[label].meta.get("hosts")
                self.ledger.measurement(out[label], phase=phase or self.t["phase"],
                                        **({"hosts": hosts} if hosts else {}))  # fmt: skip
        return out

    def _measured(self, run: Run) -> bool:
        return (self.data / run.label / run.part / "meta.json").exists()

    def _ran(
        self,
        spool: str,
        labels: Mapping[str, Sequence[Path]],
        bench: BenchSpec,
        queued: Mapping[str, Sequence[Run]],
    ) -> dict[tuple[tuple[str, ...], str], int]:
        """How often ``spool``'s host ran (or runs) each member at each of ``labels``' configs."""
        jobs = {job: sp for sp, job in self.perf.inflight.values()}
        where = {run: jobs[job] for job, runs in queued.items() if job in jobs for run in runs}
        host = spool_host(spool) or spool
        ran: dict[tuple[tuple[str, ...], str], int] = {}
        for label in labels:
            for run in bench.runs(label):
                if run in where:
                    on = spool_host(where[run]) or where[run]
                elif self._measured(run):
                    try:
                        meta = json.loads(
                            (self.data / run.label / run.part / "meta.json").read_text()
                        )
                    except (OSError, ValueError):
                        continue
                    on = str(meta.get("hostname") or "")
                else:
                    continue
                if on == host:
                    ran[_key(run, labels)] = ran.get(_key(run, labels), 0) + 1
        return ran

    def _share(
        self,
        waiting: Sequence[Run],
        ran: Mapping[tuple[tuple[str, ...], str], int],
        labels: Mapping[str, Sequence[Path]],
        bench: BenchSpec,
        queued: Mapping[str, Sequence[Run]],
    ) -> list[Run]:
        """The runs a free node measures next, in one job; ``ran`` is what its host has run.

        Its fair share of the runs left over the live nodes, or, while the repeats have members
        its host has not run, one of each (a whole set on one host is its control,
        :func:`host_controls`). Never more runs than one label has: a job stays inside a label's
        wall. Fewest-run members first, so a member's repeats spread over the hosts.
        """
        same = [_key(Run(label), labels)[0] for label in labels]
        repeated = {r for r in waiting if same.count(_key(r, labels)[0]) > 1}
        new = {r.part for r in repeated if not ran.get(_key(r, labels))}
        live = sum(1 for sp in self.perf.spools if self.perf.alive(sp)) or 1
        load = len(waiting) + sum(len(runs) for runs in queued.values())
        size = min(len(bench.parts) or 1, max(-(-load // live), len(new)))
        order = {label: n for n, label in enumerate(labels)}
        members = {part.name: n for n, part in enumerate(bench.parts)}
        counts, pool, chosen = dict(ran), list(waiting), []
        while pool and len(chosen) < size:
            run = min(pool, key=lambda r: (counts.get(_key(r, labels), 0), r not in repeated,
                                           members.get(r.part, 0), order[r.label]))  # fmt: skip
            pool.remove(run)
            chosen.append(run)
            counts[_key(run, labels)] = counts.get(_key(run, labels), 0) + 1
        return chosen

    #: ``state.tune`` key: the jobs a measurement has in flight, by job, with their runs.
    QUEUED_RUNS = "queued_runs"

    def _note_queued(self, queued: Mapping[str, Sequence[Run]]) -> None:
        """Checkpoint the jobs in flight and their runs, for a restart to adopt."""
        note = {job: [[run.label, run.part] for run in runs] for job, runs in queued.items()}
        if note != (self.t.get(self.QUEUED_RUNS) or {}):
            self.t[self.QUEUED_RUNS] = note
            self.checkpoint()

    def _adopt_runs(
        self, todo: Sequence[Run], labels: Mapping[str, Sequence[Path]]
    ) -> dict[str, list[Run]]:
        """The jobs an earlier process left in flight for ``todo``, adopted: never run twice.

        Those it checkpointed (:meth:`_note_queued`), and a label's whole job queued before
        its runs were spread.
        """
        queued: dict[str, list[Run]] = {}
        for job, names in (self.t.get(self.QUEUED_RUNS) or {}).items():
            runs = [run for run in todo if [run.label, run.part] in names]
            if runs and self.perf.adopt(job):
                queued[job] = runs
        held = {run for runs in queued.values() for run in runs}
        for label in labels:
            runs = [run for run in todo if run.label == label and run not in held]
            if runs and label not in queued and self.perf.adopt(label):
                queued[label] = runs
        return queued

    def _reusable(
        self, label: str, configs: Sequence[Path], tree: str | None, bench: BenchSpec
    ) -> bool:
        """``label`` is on disk and measured the workload, configs and commit ``bench`` would.

        A measurement of anything else (the spec changed between attempts, the best tree moved)
        is set aside under ``<label>.stale-<time>`` and measured again.
        """
        path = self.data / label
        if not (path / "meta.json").exists():
            return False
        why = _differs(path, bench, configs, tree)
        if not why:
            return True
        aside = path.with_name(f"{label}.stale-{int(time.time())}")
        path.rename(aside)
        self.ledger.append("stale_measurement", label=label, reason=why, moved_to=str(aside))
        self.log(f"[tune] {label} on disk {why}; set aside as {aside.name} and measured again")
        return False

    def _run_reusable(
        self, run: Run, configs: Sequence[Path], tree: str | None, bench: BenchSpec
    ) -> bool:
        """:meth:`_reusable` for one run of a label not yet joined (``<part>.stale-<time>``)."""
        path = self.data / run.label
        if not run.part or not (path / run.part / "meta.json").exists():
            return False
        part = next(p for p in bench.parts if p.name == run.part)
        why = _differs(path, replace(bench, parts=(part,)), configs, tree)
        if not why:
            return True
        aside = path / f"{run.part}.stale-{int(time.time())}"
        (path / run.part).rename(aside)
        self.ledger.append("stale_measurement", label=run.name, reason=why, moved_to=str(aside))
        self.log(f"[tune] {run.name} on disk {why}; set aside as {aside.name} and measured again")
        return False


__all__ = [
    "MeasurementRunner",
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
    "validate_measurement",
]
