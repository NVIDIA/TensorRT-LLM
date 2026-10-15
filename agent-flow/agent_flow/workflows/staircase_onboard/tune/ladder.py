# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The config ladder: serving-config rungs measured and kept by rule, not by opinion.

Staircase's tuning campaign moved one LLM-API knob per iteration and kept it
when it won by a threshold somewhere and lost beyond the noise nowhere. That
part of the campaign needs no judgement once the rungs are written down, so
here it is Python -- the same reason preflight is Python: a broken host or an
unmeasurable knob should cost a job, not an agent turn.

A **rung** is one overlay on the serving config (an LLM-API yaml fragment
and/or ``env_overrides``). Rungs in one ``group`` are alternatives (at most one
is kept). The ladder runs in **waves**:

1. *control*: the current best config, measured once per session, which is
   also the noise floor (cross-node spread of identical configs).
2. *probe wave*: every pending rung measured against the control, one rung
   per whole-node job, as many in parallel as there are sessions.
3. *combine*: the best winner of each group stacked onto the control and
   measured; if the stack holds every single winner's gain within the noise it
   is the new control, otherwise winners are added greedily in order of gain.
4. repeat with the rungs that were neutral, until a wave keeps nothing.

The accuracy gate runs once on the final ladder config (and per rung marked
``regate`` before that rung may be kept). A failure bisects the kept stack.

What is **not** a rung: a knob that selects kernels inside the stock modules
(``moe_config.backend``, ``attn_backend``, ``use_cute_dsl_*``,
``nvfp4_gemm_config``). A modeling_v2 target pins its kernels through the
catalog, so those knobs never reach it; they belong to the *stock-tuned*
reference, never to the target's ladder (``scope: module``).
"""

from __future__ import annotations

import copy
import json
import shlex
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from .records import Comparison, Measurement, NoiseFloor, PerfLedger, compare


@dataclass(frozen=True)
class Rung:
    name: str
    overlay: Mapping[str, Any]
    why: str
    group: str = ""
    scope: str = "runtime"  # runtime | module | target-env
    helps: str = "both"  # latency | throughput | both
    regate: bool = False
    # Fill-ins resolved against the workload: "{max_con}", "{isl}", "{isl2}".
    templated: bool = False

    def resolve(self, workload: Mapping[str, int]) -> dict:
        if not self.templated:
            return copy.deepcopy(dict(self.overlay))
        text = json.dumps(self.overlay)
        for key, value in workload.items():
            text = text.replace(f'"{{{key}}}"', str(value))
        return json.loads(text)


def default_rungs() -> list[Rung]:
    """The ladder for a single-GPU aggregated target.

    Order is the prior: cheapest and most-often-winning first. Every observation here
    is a starting point, re-measured per target -- a rung that won elsewhere can lose
    here (padding lost 11% on one tp1 MLA target).
    """
    return [
        Rung(
            "cg-padding",
            {"cuda_graph_config": {"enable_padding": True, "max_batch_size": "{max_con}"}},
            "Default graph sizes are 1..32,64,128 without padding: a decode batch of 33..127 "
            "(except 64) runs eager. Padding rounds up to the next captured size.",
            group="cuda-graph",
            helps="throughput",
            templated=True,
        ),
        Rung(
            "cg-dense",
            {"cuda_graph_config": {"batch_sizes": "{dense_sizes}"}},
            "Capture every decode batch size up to the max concurrency: exact hits, no "
            "padding waste, more capture time and graph memory.",
            group="cuda-graph",
            helps="throughput",
            templated=True,
        ),
        Rung(
            "mnt-2x",
            {"max_num_tokens": "{isl2}"},
            "Per-iteration token budget 2x ISL: fewer prefill iterations, larger prefill "
            "GEMMs; decode steps that share an iteration with a prefill get slower.",
            group="max-num-tokens",
            helps="throughput",
            templated=True,
        ),
        Rung(
            "mnt-half",
            {"max_num_tokens": "{isl_half}"},
            "Per-iteration budget ISL/2: chunked prefill interleaves decode more finely "
            "(TPOT at high concurrency), costs TTFT.",
            group="max-num-tokens",
            helps="throughput",
            templated=True,
        ),
        Rung(
            "in-graph-sampling",
            {"enable_in_graph_sampling": True},
            "Capture part of sampling into the decode graph: fewer eager launches per step.",
            helps="latency",
        ),
        Rung(
            "low-latency-dispatch",
            {"enable_low_latency_host_dispatch": True},
            "Spin-wait host-task dispatch (cudaLaunchHostFunc_v2): shorter callback latency "
            "per step at the cost of a spinning core.",
            helps="latency",
        ),
        Rung(
            "early-first-token",
            {"enable_early_first_token_response": True},
            "Emit the first token ahead of the next sample step under the overlap scheduler (TTFT).",
            helps="latency",
        ),
        Rung(
            "stream-interval-4",
            {"stream_interval": 4},
            "Emit responses every 4 iterations: less per-step response-path host work at high "
            "concurrency. Changes streaming granularity (a serving-policy choice, report it).",
            group="stream-interval",
            helps="throughput",
        ),
        Rung(
            "sched-max-util",
            {"scheduler_config": {"capacity_scheduler_policy": "MAX_UTILIZATION"}},
            "Admit by current rather than reserved-to-completion KV: packs more requests, may "
            "pause/evict.",
            helps="throughput",
        ),
        Rung(
            "kv-frac-0.85",
            {"kv_cache_config": {"free_gpu_memory_fraction": 0.85}},
            "Larger KV pool; inert unless capacity binds at the max concurrency.",
            helps="throughput",
        ),
        Rung(
            "gc-gen0-100k",
            {"garbage_collection_gen0_threshold": 100000},
            "Fewer Python gen-0 collections on the executor loop (host jitter).",
            helps="latency",
        ),
    ]


MODULE_KNOBS = (
    "moe_config",
    "attn_backend",
    "use_cute_dsl_blockscaling_mm",
    "use_cute_dsl_blockscaling_bmm",
    "use_cute_dsl_bf16_bmm",
    "use_cute_dsl_bf16_gemm",
    "nvfp4_gemm_config",
)


def rungs_from_spec(spec: Iterable[Mapping[str, Any]]) -> list[Rung]:
    """Extra rungs a task declares -- typically a target's own switches (``env_overrides``)."""
    out = []
    for item in spec:
        out.append(
            Rung(
                name=item["name"],
                overlay=item.get("overlay") or {},
                why=item.get("why", ""),
                group=item.get("group", ""),
                scope=item.get("scope", "runtime"),
                helps=item.get("helps", "both"),
                regate=bool(item.get("regate", False)),
                templated=bool(item.get("templated", False)),
            )
        )
    return out


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(dict(base))
    for key, value in (over or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def workload_fillins(isl: int, osl: int, concurrencies: Sequence[int]) -> dict[str, Any]:
    max_con = max(concurrencies)
    return {
        "max_con": max_con,
        "isl": isl,
        "isl2": 2 * isl,
        "isl_half": max(256, isl // 2),
        "dense_sizes": json.dumps(list(range(1, max_con + 1))),
    }


# --------------------------------------------------------------------------------------
# Execution: one measurement = one whole-node job in a spool session.
# --------------------------------------------------------------------------------------


@dataclass
class BenchSpec:
    """How to run ``bench.py`` inside a session: the same for every rung."""

    bench_py: str
    model: str
    env_script: str
    isl: int
    osl: int
    concurrencies: Sequence[int]
    tree: str
    extra_args: Sequence[str] = ()
    wall: str = "2h"

    def command(self, label: str, out_root: Path, configs: Sequence[Path]) -> str:
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
        argv += ["--isl", str(self.isl), "--osl", str(self.osl)]
        argv += [
            "--concurrency",
            ",".join(str(c) for c in self.concurrencies),
            "--tree",
            self.tree,
            "--force",
        ]
        argv += list(self.extra_args)
        return f"source {shlex.quote(self.env_script)}\n" + " ".join(shlex.quote(a) for a in argv)


class SpoolRunner:
    """Queue measurements across sessions, one in flight per session, and collect them.

    ``submit`` returns immediately; ``poll`` returns the jobs that finished. The
    substrate's own client (``spoolctl.queue_job``) does the queuing, so names,
    receipts and archiving follow the same protocol the gates use.
    """

    def __init__(self, spools: Sequence[str], prefix: str = "tune") -> None:
        from ..substrate import spoolctl

        self._spoolctl = spoolctl
        self.spools = list(spools)
        self.prefix = prefix
        self.inflight: dict[str, tuple[str, str]] = {}  # job -> (spool, label)

    def free_spools(self) -> list[str]:
        """Sessions with a live watcher and nothing of ours in flight."""
        busy = {spool for spool, _ in self.inflight.values()}
        live = []
        for spool in self.spools:
            if spool in busy:
                continue
            age = self._spoolctl.watcher_age(spool)
            if age is not None and age <= self._spoolctl.WATCHER_STALE_S:
                live.append(spool)
        return live

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
        script = self._spoolctl.render_job(command, header=header)
        receipt = Path(spool) / "done" / f"{name}.rc"
        if receipt.exists():
            # A re-run of a label whose earlier run failed: its old receipt would
            # read as this run finishing at once.
            receipt.rename(receipt.with_name(f"{receipt.name}.superseded-{int(time.time())}"))
        try:
            self._spoolctl.queue_job(spool, name, script)
        except self._spoolctl.SpoolError as exc:
            # Queued by an earlier process of this campaign: adopt it, never twice.
            if "already queued or running" not in str(exc) or not self.adopt(label):
                raise
            return name
        self.inflight[name] = (spool, label)
        return name

    def adopt(self, label: str) -> bool:
        """Re-register a job queued by an earlier process (resume after a restart)."""
        name = f"{self.prefix}-{label}"
        for spool in self.spools:
            root = Path(spool)
            if any(
                (root / sub / f"{name}{ext}").exists()
                for sub, ext in (("pending", ".sh"), ("running", ".sh"), ("done", ".rc"))
            ):
                self.inflight[name] = (spool, label)
                return True
        return False

    def cancel(self, label: str) -> bool:
        """Ask the session to stop a job of ours (the scheduler's sanctioned kill) and forget it."""
        name = f"{self.prefix}-{label}"
        entry = self.inflight.pop(name, None)
        if entry is None:
            return False
        cancel_dir = Path(entry[0]) / "cancel"
        cancel_dir.mkdir(parents=True, exist_ok=True)
        (cancel_dir / name).touch()
        return True

    #: A session silent this long is gone (its Slurm job ended): its jobs died with it.
    DEAD_SESSION_S = 600.0
    RC_SESSION_LOST = 212  # the scheduler's RC_ORPHANED

    def poll(self) -> list[tuple[str, str, int]]:
        done = []
        for name, (spool, label) in list(self.inflight.items()):
            receipt = Path(spool) / "done" / f"{name}.rc"
            age = self._spoolctl.watcher_age(spool)
            if not receipt.exists() and age is not None and age > self.DEAD_SESSION_S:
                # Nobody will write this receipt until a new session adopts the spool;
                # set the job aside so that session does not run it again, and report
                # it lost so the caller re-queues it on a live session.
                for sub in ("pending", "running"):
                    script = Path(spool) / sub / f"{name}.sh"
                    if script.exists():
                        script.rename(script.with_name(f"{script.name}.lost-{int(time.time())}"))
                del self.inflight[name]
                done.append((label, spool, self.RC_SESSION_LOST))
                continue
            if receipt.exists():
                try:
                    rc = int(receipt.read_text().strip() or "1")
                except ValueError:
                    continue
                del self.inflight[name]
                done.append((label, spool, rc))
        return done


# --------------------------------------------------------------------------------------
# The ladder state machine.
# --------------------------------------------------------------------------------------


@dataclass
class RungState:
    name: str
    status: str = "pending"  # pending | running | measured | kept | rejected | failed | skipped
    label: str = ""
    verdict: str = ""
    reason: str = ""
    deltas: dict = field(default_factory=dict)
    wave: int = 0


@dataclass
class LadderState:
    control_overlay: dict = field(default_factory=dict)
    control_labels: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    rungs: dict[str, RungState] = field(default_factory=dict)
    wave: int = 0
    phase: str = "control"  # control | probe | combine | done
    combine_label: str = ""
    history: list[dict] = field(default_factory=list)
    # The noise floor is a property of the host and the workload, measured once
    # in wave 0 from identical-config repeats and reused by every later wave.
    noise_spread: dict = field(default_factory=dict)
    noise_samples: int = 0

    @classmethod
    def load(cls, path: Path) -> "LadderState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        state = cls(**{k: v for k, v in data.items() if k != "rungs"})
        state.rungs = {k: RungState(**v) for k, v in data.get("rungs", {}).items()}
        return state

    def save(self, path: Path) -> None:
        data = asdict(self)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
        tmp.replace(path)


class Ladder:
    """Drive the ladder to completion (or one step at a time with ``step``)."""

    def __init__(
        self,
        *,
        workdir: Path,
        base_configs: Sequence[Path],
        rungs: Sequence[Rung],
        bench: BenchSpec,
        runner: SpoolRunner,
        min_gain: float = 0.03,
        noise_factor: float = 2.0,
        target_mode: bool = True,
        reprobe_min: float = 0.005,
        log: Callable[[str], None] = print,
    ) -> None:
        self.workdir = workdir
        self.reprobe_min = reprobe_min
        self.base_configs = list(base_configs)
        self.bench = bench
        self.runner = runner
        self.min_gain = min_gain
        self.noise_factor = noise_factor
        self.log = log
        self.fill = workload_fillins(bench.isl, bench.osl, bench.concurrencies)
        self.rungs = {}
        for rung in rungs:
            if target_mode and any(key in MODULE_KNOBS for key in rung.overlay):
                self.log(f"[ladder] skip {rung.name}: module knob, inert for a modeling_v2 target")
                continue
            self.rungs[rung.name] = rung
        self.state_path = workdir / "ladder_state.json"
        self.ledger = PerfLedger(workdir / "perf_ledger.jsonl")
        self.data = workdir / "data"
        self.cfgdir = workdir / "configs"
        for d in (self.data, self.cfgdir):
            d.mkdir(parents=True, exist_ok=True)
        self.state = LadderState.load(self.state_path)
        for name in self.rungs:
            self.state.rungs.setdefault(name, RungState(name))
        self._resume()

    def seed_controls(self, dirs: Sequence[Path]) -> None:
        """Count existing measurements of the identity config as wave-0 control repeats.

        They must have been taken with the same base configs and workload; the
        caller vouches for that (the ladder records which dirs it adopted).
        """
        if self.state.wave != 0 or self.state.control_labels:
            return
        for src in dirs:
            link = self.data / src.name
            if not link.exists():
                link.symlink_to(src.resolve(), target_is_directory=True)
            self.state.control_labels.append(src.name)
        self.ledger.append(
            "ladder_seed_controls", labels=[d.name for d in dirs], dirs=[str(d) for d in dirs]
        )
        self.state.save(self.state_path)

    def _resume(self) -> None:
        st = self.state
        labels = list(st.control_labels) + [
            rs.label for rs in st.rungs.values() if rs.status == "running"
        ]
        if st.combine_label:
            labels.append(st.combine_label)
        for rs in st.rungs.values():
            # Finished while no process watched it: its measurement is on disk.
            if rs.status == "running" and rs.label and self._measurement(rs.label):
                rs.status = "measured"
        for label in labels:
            if label and not self._measurement(label) and not self.runner.adopt(label):
                self.log(
                    f"[ladder] resume: {label} has no measurement and no job; it will be re-queued"
                )
                for rs in st.rungs.values():
                    if rs.label == label and rs.status == "running":
                        rs.status, rs.label = "pending", ""
                if label in st.control_labels:
                    st.control_labels.remove(label)
                if label == st.combine_label:
                    st.combine_label = ""

    # -- config files --------------------------------------------------------------
    def _write_overlay(self, name: str, overlay: Mapping[str, Any]) -> Path:
        path = self.cfgdir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(dict(overlay), sort_keys=False))
        return path

    def _configs(self, extra: Mapping[str, Any] | None = None, name: str = "") -> list[Path]:
        configs = list(self.base_configs)
        if self.state.control_overlay:
            configs.append(self._write_overlay("control", self.state.control_overlay))
        if extra:
            configs.append(self._write_overlay(name, extra))
        return configs

    def _stack(self, names: Iterable[str]) -> dict:
        overlay: dict = {}
        for name in names:
            overlay = deep_merge(overlay, self.rungs[name].resolve(self.fill))
        return overlay

    # -- measurement bookkeeping ---------------------------------------------------
    def _measurement(self, label: str) -> Measurement | None:
        path = self.data / label
        if (path / "summary.json").exists() and (path / "meta.json").exists():
            return Measurement.load(path)
        return None

    def _controls(self) -> list[Measurement]:
        return [m for m in (self._measurement(label) for label in self.state.control_labels) if m]

    def noise(self) -> NoiseFloor:
        """Same-host repeat spread, the worst host's.

        Hosts differ by more than knobs do: on GB300 nodes of one cluster the
        latency end of identical configs differed by up to 4% between nodes while
        repeats on one node agreed within 0.1% (2026-10-12). So a probe is
        judged against a control from its own host (:meth:`control`), and the
        floor is the same-host spread, never the cross-host one.
        """
        if self.state.noise_spread:
            return NoiseFloor(
                spread=dict(self.state.noise_spread), samples=self.state.noise_samples
            )
        by_host: dict[str, list[Measurement]] = {}
        for m in self._controls():
            by_host.setdefault(str(m.meta.get("hostname")), []).append(m)
        floors = [NoiseFloor.from_measurements(ms) for ms in by_host.values() if len(ms) >= 2]
        if not floors:
            return NoiseFloor.from_measurements(self._controls())
        spread: dict[str, float] = {}
        for f in floors:
            for k, v in f.spread.items():
                spread[k] = max(spread.get(k, 0.0), v)
        return NoiseFloor(spread=spread, samples=sum(f.samples for f in floors))

    def control(self, host: str | None = None) -> Measurement | None:
        """The control on ``host`` (median throughput end among its repeats), else any host's."""
        reps = self._controls()
        same = [m for m in reps if host and m.meta.get("hostname") == host]
        reps = same or reps
        if not reps:
            return None
        key = lambda m: sorted(m.headline().items())[-1][1]  # noqa: E731
        return sorted(reps, key=key)[len(reps) // 2]

    @staticmethod
    def spool_host(spool: str) -> str:
        try:
            return str(json.loads((Path(spool) / "status.json").read_text()).get("host") or "")
        except (OSError, ValueError):
            return ""

    def _spools_without_control(self) -> list[str]:
        """Live sessions whose host has no control yet.

        A session that died (preempted, its Slurm job requeued) refuses jobs; it gets a
        control when it is back.
        """
        hosts = {m.meta.get("hostname") for m in self._controls()}
        live = set(self.runner.free_spools()) | {
            spool for spool, _ in self.runner.inflight.values()
        }
        return [s for s in self.runner.spools if s in live and self.spool_host(s) not in hosts]

    def _submit(self, label: str, configs: Sequence[Path]) -> bool:
        free = self.runner.free_spools()
        if not free:
            return False
        cmd = self.bench.command(label, self.data, configs)
        self.runner.submit(free[0], label, cmd, wall=self.bench.wall)
        self.log(f"[ladder] queued {label} on {free[0]}")
        return True

    # -- the waves --------------------------------------------------------------------
    def step(self) -> bool:
        """Advance as far as possible without blocking. Returns False once done."""
        st = self.state
        for label, spool, rc in self.runner.poll():
            self.log(f"[ladder] {label} finished rc={rc} on {spool}")
            m = self._measurement(label)
            if m:
                self.ledger.measurement(m, ladder_wave=st.wave, spool=spool)
            for rs in st.rungs.values():
                if rs.label == label and rs.status == "running":
                    rs.status = "measured" if m else "failed"
                    if not m:
                        rs.reason = (
                            f"measurement failed rc={rc}; see {self.data / (label + '.partial')}"
                        )
        if st.phase == "control":
            # One control per session host: a probe is judged against its own host.
            inflight_spools = {spool for spool, _ in self.runner.inflight.values()}
            for spool in self._spools_without_control():
                if spool in inflight_spools:
                    continue
                label = f"w{st.wave:02d}-control-{Path(spool).name}"
                if label in st.control_labels:
                    continue
                cmd = self.bench.command(label, self.data, self._configs())
                self.runner.submit(spool, label, cmd, wall=self.bench.wall)
                st.control_labels.append(label)
                self.log(f"[ladder] queued control {label} on {spool}")
            if self.runner.inflight:
                self.state.save(self.state_path)
                return True
            st.phase = "probe"
            noise = self.noise()
            if not st.noise_spread and noise.samples >= 2:
                st.noise_spread, st.noise_samples = dict(noise.spread), noise.samples
            self.ledger.append(
                "noise_floor", wave=st.wave, spread=noise.spread, samples=noise.samples
            )
            self.log(f"[ladder] noise floor wave {st.wave}: {noise.spread}")
        if st.phase == "probe":
            pending = [rs for rs in st.rungs.values() if rs.status == "pending"]
            for rs in pending:
                if not self.runner.free_spools():
                    break
                label = f"w{st.wave:02d}-{rs.name}"
                overlay = self.rungs[rs.name].resolve(self.fill)
                if self._submit(label, self._configs(overlay, rs.name)):
                    rs.status, rs.label, rs.wave = "running", label, st.wave
            if (
                any(rs.status in ("pending", "running") for rs in st.rungs.values())
                or self.runner.inflight
            ):
                self.state.save(self.state_path)
                return True
            self._decide_wave()
        if st.phase == "combine":
            if not st.combine_label:
                winners = self._winners()
                if len(winners) <= 1:
                    self._adopt(winners)
                    return self._next_wave()
                st.combine_label = f"w{st.wave:02d}-combine"
                overlay = self._stack(winners)
                if not self._submit(st.combine_label, self._configs(overlay, "combine")):
                    st.combine_label = ""
                self.state.save(self.state_path)
                return True
            if self.runner.inflight:
                self.state.save(self.state_path)
                return True
            combo = self._measurement(st.combine_label)
            winners = self._winners()
            ctrl = self.control(combo.meta.get("hostname") if combo else None)
            if combo and ctrl:
                cmp = compare(
                    combo,
                    ctrl,
                    self.noise(),
                    min_gain=self.min_gain,
                    noise_factor=self.noise_factor,
                )
                self.ledger.append("ladder_combine", wave=st.wave, rungs=winners, **cmp.as_dict())
                if cmp.verdict == "WIN":
                    self._adopt(winners, label=st.combine_label)
                else:
                    best = winners[:1]
                    self.log(
                        f"[ladder] stack did not hold ({cmp.reason}); keeping best single {best}"
                    )
                    self._adopt(best)
            return self._next_wave()
        return st.phase != "done"

    def _winners(self) -> list[str]:
        """Best WIN per group, ordered by its best delta."""
        best: dict[str, tuple[float, str]] = {}
        for rs in self.state.rungs.values():
            if rs.wave != self.state.wave or rs.verdict != "WIN":
                continue
            gain = max(rs.deltas.values()) if rs.deltas else 0.0
            group = self.rungs[rs.name].group or rs.name
            if group not in best or gain > best[group][0]:
                best[group] = (gain, rs.name)
        return [name for _, name in sorted(best.values(), reverse=True)]

    def _decide_wave(self) -> None:
        st = self.state
        noise = self.noise()
        for rs in st.rungs.values():
            if rs.wave != st.wave or rs.status != "measured":
                continue
            cand = self._measurement(rs.label)
            ctrl = self.control(cand.meta.get("hostname") if cand else None)
            if not (cand and ctrl):
                rs.status, rs.reason = "failed", "no measurement"
                continue
            cmp: Comparison = compare(
                cand, ctrl, noise, min_gain=self.min_gain, noise_factor=self.noise_factor
            )
            rs.verdict, rs.reason, rs.deltas = cmp.verdict, cmp.reason, dict(cmp.deltas)
            rs.status = "measured"
            self.ledger.append(
                "ladder_probe",
                wave=st.wave,
                rung=rs.name,
                regate=self.rungs[rs.name].regate,
                **cmp.as_dict(),
            )
            self.log(f"[ladder] {rs.name}: {cmp.verdict} {cmp.reason}")
        st.phase = "combine"

    def _adopt(self, names: Sequence[str], label: str = "") -> None:
        st = self.state
        for rs in st.rungs.values():
            if rs.wave != st.wave or rs.status != "measured":
                continue
            if rs.name in names:
                rs.status = "kept"
            elif rs.verdict == "WIN":
                rs.status = "rejected"
                rs.reason += " | lost to a stronger member of its group or did not stack"
            elif rs.verdict == "LOSS":
                rs.status = "rejected"
            else:
                rs.status = "neutral"
        if names:
            st.kept.extend(names)
            st.control_overlay = deep_merge(st.control_overlay, self._stack(names))
        if not label and names:
            label = next((rs.label for rs in st.rungs.values() if rs.name == names[0]), "")
        st.history.append(
            {
                "wave": st.wave,
                "kept": list(names),
                "label": label,
                "control_overlay": st.control_overlay,
            }
        )
        self.ledger.append(
            "ladder_adopt", wave=st.wave, kept=list(names), control_overlay=st.control_overlay
        )
        self.log(f"[ladder] wave {st.wave} kept {list(names) or 'nothing'}")

    def _next_wave(self) -> bool:
        st = self.state
        kept_now = st.history[-1]["kept"] if st.history else []
        if not kept_now:
            st.phase = "done"
            self.state.save(self.state_path)
            self._write_final()
            return False
        # A neutral rung that came close (best delta >= reprobe_min) gets one more
        # chance against the new control; a flat one does not: with same-host
        # noise at 0.1-0.4%, a rung flat at wave 0 rarely flips, and re-probing
        # every neutral rung costs a whole wave. Rejected and failed rungs never
        # re-run, nor does anything from a kept group.
        kept_groups = {self.rungs[n].group or n for n in st.kept}
        adoption = st.history[-1].get("label") if st.history else ""
        st.wave += 1
        st.phase, st.combine_label = "control", ""
        st.control_labels = [adoption] if adoption else []
        for rs in st.rungs.values():
            group = self.rungs[rs.name].group or rs.name
            if (
                rs.status == "neutral"
                and group not in kept_groups
                and max(rs.deltas.values() or [0.0]) >= self.reprobe_min
            ):
                rs.status, rs.label, rs.verdict, rs.reason, rs.deltas = "pending", "", "", "", {}
        if not any(rs.status == "pending" for rs in st.rungs.values()):
            st.phase = "done"
            self.state.save(self.state_path)
            self._write_final()
            return False
        self.state.save(self.state_path)
        return True

    def _write_final(self) -> None:
        final = deep_merge({}, self.state.control_overlay)
        (self.workdir / "ladder_final.yaml").write_text(
            "# The config ladder's result: overlay on the identity config.\n"
            + yaml.safe_dump(final, sort_keys=False)
        )
        self.ledger.append("ladder_done", kept=self.state.kept, overlay=final)
        self.log(
            f"[ladder] done; kept {self.state.kept}; overlay -> {self.workdir / 'ladder_final.yaml'}"
        )

    def run(self, poll_s: float = 20.0) -> dict:
        while self.step():
            time.sleep(poll_s)
        return {"kept": self.state.kept, "overlay": self.state.control_overlay}

    def summary(self) -> str:
        st = self.state
        lines = [
            f"phase={st.phase} wave={st.wave} kept={st.kept}",
            f"control labels={st.control_labels}",
        ]
        for rs in st.rungs.values():
            d = ", ".join(f"{k} {v:+.1%}" for k, v in rs.deltas.items())
            lines.append(
                f"  {rs.name:22s} {rs.status:9s} w{rs.wave} {rs.verdict:7s} {d} {rs.reason[:90]}"
            )
        return "\n".join(lines)
