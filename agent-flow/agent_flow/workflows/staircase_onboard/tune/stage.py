# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tune stage: ground truth -> config ladder -> modeling campaign -> wrap-up.

Entered after delivery when ``perf.enabled`` (chained), or directly by
``staircase-onboard tune`` on an already-onboarded target (standalone). Every
phase checkpoints into ``state.tune`` and is idempotent on resume: a label
whose measurement completed is never re-measured, a candidate whose decision
is on disk is never re-dispatched.

The workflow object is used through four attributes only (``workspace``,
``task_data``, ``_checkpoint`` and a ``dispatch`` callable), so the stage runs
under a fake in the CPU tests exactly as it runs under the orchestrator.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from ..substrate.spoolctl import SpoolError
from .campaign import INFRA_RCS, Campaign, gate_command, git, read_decision
from .ladder import BenchSpec, Ladder, SpoolRunner, default_rungs, rungs_from_spec
from .records import Measurement, NoiseFloor, PerfLedger, format_table
from .requests import CampaignContext, render_rebase_request, render_request, turn_prompt
from .worktree import open_candidate

PACKAGE = Path(__file__).resolve().parent.parent

PERF_KEYS = {
    "enabled", "model", "target", "identity_configs", "workload", "reference", "spools", "dev_spools",
    "env_script", "noise_repeats", "ladder", "campaign", "source_embargo", "sister_engines", "wall",
}  # fmt: skip


def validate_perf_block(data: Mapping[str, Any]) -> list[str]:
    """Closed keys, batched errors -- the task schema's style."""
    perf = data.get("perf")
    if perf is None:
        return []
    if not isinstance(perf, Mapping):
        return ["perf: must be a mapping"]
    errors = [f"perf.{k}: unknown key" for k in perf if k not in PERF_KEYS]
    workload = perf.get("workload") or {}
    for key in ("isl", "osl"):
        if key in workload and (not isinstance(workload[key], int) or workload[key] <= 0):
            errors.append(f"perf.workload.{key}: a positive integer")
    con = workload.get("concurrency", [1])
    if not (isinstance(con, list) and con and all(isinstance(c, int) and c > 0 for c in con)):
        errors.append("perf.workload.concurrency: a non-empty list of positive integers")
    if perf.get("reference", "stock") not in ("stock", "none"):
        errors.append("perf.reference: stock | none")
    for key in ("identity_configs", "spools"):
        if key in perf and not isinstance(perf[key], list):
            errors.append(f"perf.{key}: a list")
    campaign = perf.get("campaign") or {}
    if campaign and not isinstance(campaign.get("accuracy", {}), Mapping):
        errors.append("perf.campaign.accuracy: a mapping {task, anchor}")
    return errors


@dataclass
class TuneSpec:
    workdir: Path
    repo: Path
    model: str
    target: str
    identity_configs: list[Path]
    isl: int
    osl: int
    concurrency: list[int]
    reference: str
    spools: list[str]
    dev_spools: list[str]
    env_script: Path
    noise_repeats: int = 2
    ladder: dict = field(default_factory=dict)
    campaign: dict = field(default_factory=dict)
    embargo: list[str] = field(default_factory=list)
    sister_engines: list[str] = field(default_factory=list)
    wall: str = "2h"

    @classmethod
    def from_task(cls, task: Mapping[str, Any], workspace: Path, *, target: str = "") -> "TuneSpec":
        perf = task.get("perf") or {}
        workload = perf.get("workload") or {}
        env = task.get("environment") or {}
        session_spools = [s.get("spool") for s in env.get("sessions") or [] if s.get("spool")]
        spools = [str(s) for s in perf.get("spools") or session_spools]
        return cls(
            # Absolute: every path in a job script is read on a compute node from
            # the job's own cwd (the tree measured), not from where the CLI ran.
            workdir=Path(workspace).resolve() / "tune",
            repo=Path(task.get("trtllm_repo_path") or ".").resolve(),
            model=str(perf.get("model") or task.get("checkpoint_path")),
            target=str(perf.get("target") or target),
            identity_configs=[Path(p).resolve() for p in perf.get("identity_configs") or []],
            isl=int(workload.get("isl", 1024)),
            osl=int(workload.get("osl", 1024)),
            concurrency=[int(c) for c in workload.get("concurrency", [1, 128])],
            reference=str(perf.get("reference", "stock")),
            spools=[str(Path(s).resolve()) for s in spools],
            dev_spools=[str(Path(s).resolve()) for s in perf.get("dev_spools") or spools],
            env_script=Path(perf.get("env_script") or Path(workspace) / "tune" / "env.sh").resolve(),
            noise_repeats=int(perf.get("noise_repeats", 2)),
            ladder=dict(perf.get("ladder") or {}),
            campaign=dict(perf.get("campaign") or {}),
            embargo=[str(e) for e in (task.get("source_embargo") or []) + (perf.get("source_embargo") or [])],
            sister_engines=[str(s) for s in perf.get("sister_engines") or []],
            wall=str(perf.get("wall", "2h")),
        )

    def bench(self) -> BenchSpec:
        return BenchSpec(
            bench_py=str(PACKAGE / "tune" / "bench.py"), model=self.model, env_script=str(self.env_script),
            isl=self.isl, osl=self.osl, concurrencies=self.concurrency, tree=str(self.repo), wall=self.wall,
        )  # fmt: skip


def _overlay_file(workdir: Path, name: str, data: Mapping[str, Any]) -> Path:
    path = workdir / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(data), sort_keys=False))
    return path


class TuneStage:
    """One campaign. ``dispatch(role, what, prompt)`` runs one agent turn and returns its text."""

    def __init__(
        self,
        spec: TuneSpec,
        tune_state: dict,
        *,
        dispatch: Callable[[str, str, str], str],
        checkpoint: Callable[[], None],
        runner_factory: Callable[[Sequence[str], str], Any] | None = None,
        poll_s: float = 30.0,
        log: Callable[[str], None] = print,
    ) -> None:
        self.spec = spec
        self.t = tune_state
        self.dispatch = dispatch
        self.checkpoint = checkpoint
        self.poll_s = poll_s
        self.log = log
        spec.workdir.mkdir(parents=True, exist_ok=True)
        self.data = spec.workdir / "data"
        self.data.mkdir(exist_ok=True)
        self.ledger = PerfLedger(spec.workdir / "perf_ledger.jsonl")
        factory = runner_factory or (lambda spools, prefix: SpoolRunner(spools, prefix=prefix))
        # Job names carry the workspace: spools outlive a run, and a second run's
        # "tune-t-baseline" must never adopt the first run's receipt.
        run = hashlib.sha1(str(spec.workdir.resolve()).encode()).hexdigest()[:6]
        self.runners = lambda spools, prefix: factory(spools, f"{prefix}{run}")
        self.branch_prefix = f"tune/{run}"
        self.perf = self.runners(spec.spools, "tune")
        self.t.setdefault("phase", "ground")

    # ------------------------------------------------------------------ helpers
    def _measure(self, labels: Mapping[str, Sequence[Path]], tree: str | None = None) -> dict[str, Measurement | None]:
        """Measure every label not yet on disk, as many at once as there are sessions."""
        bench = self.spec.bench()
        todo = {l: c for l, c in labels.items() if not (self.data / l / "meta.json").exists()}
        for label in todo:
            self.perf.adopt(label)
        queued = {name.split("-", 1)[1] for name in self.perf.inflight}
        retries: dict[str, int] = {}
        while todo:
            for label, configs in list(todo.items()):
                if label in queued:
                    continue
                free = self.perf.free_spools()
                if not free:
                    break
                self.perf.submit(free[0], label, bench.command(label, self.data, configs), wall=bench.wall, tree=tree)
                queued.add(label)
                self.log(f"[tune] measuring {label} on {free[0]}")
            for label, _, rc in self.perf.poll():
                if rc != 0 and not (self.data / label / "meta.json").exists():
                    # A session that died, a full filesystem: measure again before giving up.
                    retries[label] = retries.get(label, 0) + 1
                    if retries[label] <= (2 if rc in INFRA_RCS else 1):
                        self.log(f"[tune] {label} failed rc={rc}; re-queued")
                        queued.discard(label)
                        continue
                    self.log(f"[tune] {label} failed rc={rc}")
                todo.pop(label, None)
            if todo:
                time.sleep(self.poll_s)
        out = {}
        for label in labels:
            path = self.data / label
            out[label] = Measurement.load(path) if (path / "meta.json").exists() else None
            if out[label]:
                self.ledger.measurement(out[label], phase=self.t["phase"])
        return out

    # ------------------------------------------------------------------- phases
    def ground(self) -> bool:
        s = self.spec
        stock = _overlay_file(s.workdir, "stock", {"modeling_v2": "off"})
        labels: dict[str, Sequence[Path]] = {"t-baseline": s.identity_configs}
        labels.update({f"t-baseline-r{i}": s.identity_configs for i in range(1, s.noise_repeats + 1)})
        if s.reference == "stock":
            labels["t-reference"] = [*s.identity_configs, stock]
        got = self._measure(labels)
        if got.get("t-baseline") is None:
            self.log("[tune] the baseline could not be measured; the campaign cannot start")
            return False
        repeats = [m for k, m in got.items() if k.startswith("t-baseline") and m]
        noise = NoiseFloor.from_measurements(repeats)
        self.t.update(baseline="t-baseline", reference="t-reference" if s.reference == "stock" else "",
                      noise_spread=noise.spread, noise_samples=noise.samples, phase="ladder")  # fmt: skip
        self.ledger.append("noise_floor", phase="ground", spread=noise.spread, samples=noise.samples)
        self.checkpoint()
        return True

    def ladder(self) -> None:
        s, cfg = self.spec, self.spec.ladder
        if cfg.get("enabled", True):
            rungs = [r for r in default_rungs() + rungs_from_spec(cfg.get("extra_rungs", [])) if r.name not in set(cfg.get("skip", []))]
            if cfg.get("only"):
                rungs = [r for r in rungs if r.name in set(cfg["only"])]
            lad = Ladder(
                workdir=s.workdir / "ladder", base_configs=s.identity_configs, rungs=rungs, bench=s.bench(),
                runner=self.runners(s.spools, "tunel"), min_gain=float(cfg.get("min_gain", 0.01)),
                noise_factor=float(cfg.get("noise_factor", 3.0)), reprobe_min=float(cfg.get("reprobe_min", 0.005)),
                log=self.log,
            )  # fmt: skip
            seeds = [self.data / l for l in ["t-baseline", *[f"t-baseline-r{i}" for i in range(1, s.noise_repeats + 1)]]]
            lad.seed_controls([p for p in seeds if (p / "meta.json").exists()])
            while True:
                try:
                    if not lad.step():
                        break
                except SpoolError as exc:
                    # A session died between choosing it and queuing on it (preempted).
                    self.log(f"[ladder] {exc}; retrying on the next poll")
                time.sleep(self.poll_s)
            self.t["ladder_overlay"] = lad.state.control_overlay
            self.t["ladder_kept"] = lad.state.kept
        overlay = self.t.get("ladder_overlay") or {}
        self.t["config_overlays"] = [str(_overlay_file(s.workdir, "ladder-final", overlay))] if overlay else []
        self.t["phase"] = "campaign"
        self.checkpoint()

    def _context(self, standings: str) -> CampaignContext:
        s = self.spec
        return CampaignContext(
            repo=s.repo, package=PACKAGE, campaign_dir=s.workdir, perf_root=s.workdir, target_dir=s.target,
            target_tests=s.target.replace("tensorrt_llm/_torch/_experimental/", "tests/unittest/_torch/"),
            model=s.model, dev_spools=[Path(p) for p in s.dev_spools], env_script=s.env_script, embargo=s.embargo,
            workload=f"ISL {s.isl} / OSL {s.osl}, concurrency {s.concurrency}, CUDA graphs on.",
            standings_table=standings, sister_engines=s.sister_engines, notes=self._sources_note(),
        )  # fmt: skip

    def _sources_note(self) -> str:
        """Where a tuner looks before writing a kernel (the skill's 'Where hypotheses come from')."""
        s = self.spec
        surface = s.workdir.parent / "surface.md"
        lines = ["## Sources to check before writing a kernel", ""]
        if surface.exists():
            lines.append(f"- FlashInfer **as installed**: `{surface}` section 'FlashInfer API' lists the installed version's "
                         "public callables the engine does not call yet -- a demand one answers is a tier-3 entry "
                         "(`from flashinfer import ...` in the entry), not a new kernel.")
        lines += [
            f"- Earlier choices for this target: `{s.workdir / '08-kernel-sources' / 'kernel_choices.json'}` (chosen + "
            "alternatives with timings); extend it, do not re-race what is recorded.",
            f"- Prior art in this repository: `git -C {s.repo} log --oneline -i --grep='perf' -- "
            "tensorrt_llm/_torch/modules/<similar module> tensorrt_llm/_torch/models/<similar model>` (strategies "
            "for the same op shapes in other models -- never the embargoed in-tree implementation of this model).",
            "- Profiles of the current best tree: the latest `roadmap*.yaml` evidence in the campaign workdir.",
        ]
        return "\n".join(lines)

    def _standings(self) -> str:
        ms = [Measurement.load(self.data / l) for l in (self.t.get("baseline"), self.t.get("reference")) if l and (self.data / l / "meta.json").exists()]
        return format_table(ms, ms[0] if ms else None) if ms else "(no measurement yet)"

    def _gate_tests(self) -> list[str]:
        """The target's own suite and the modeling_v2 contract tests: run on every tree gated."""
        if not self.spec.campaign.get("gate_tests", True):
            return []
        shared = "tests/unittest/_torch/modeling_v2/"
        return [self.spec.target.replace("tensorrt_llm/_torch/_experimental/", "tests/unittest/_torch/"),
                *(shared + f"test_modeling_v2_{n}.py" for n in ("claims", "no_stale_claims", "target_contract", "routing"))]

    def _roadmap(self, n: int = 1, tried: Sequence[str] = ()) -> list[dict]:
        """Roadmap ``n``: the first from the baseline profile, later ones re-profiled on the moved best tree."""
        what = "roadmap" if n == 1 else f"roadmap-{n}"
        path = self.spec.workdir / f"{what}.yaml"
        if not path.exists():
            request = self.spec.workdir / f"{what}-request.md"
            again = ""
            if tried:
                again = (
                    "\n\nThis is roadmap " + str(n) + ": re-profile the best tree (`" + str(self.spec.workdir / "worktrees" / "best")
                    + "`) -- the kept candidates moved the profile. Already attempted (do not repeat; their decision files under `"
                    + str(self.spec.workdir / "candidates") + "/<slug>/decision.yaml` list measured follow-ups worth ranking): "
                    + ", ".join(f"`{t}`" for t in tried) + "."
                )
            request.write_text(
                f"# Tuner request: {what}\n\nProfile the target at the campaign config (nsys through the substrate, "
                f"see the skill's references/nsys.md), read the forward, and write `{path}`: a YAML list of "
                "`{slug, kind, hypothesis, evidence, expected, modeling_edits}` ranked by the exclusive time each "
                "can remove. Then write the decision file `" + str(self.spec.workdir / f"{what}-decision.yaml")
                + "` with `decision: CANDIDATE` (roadmap ready) or `STOP`." + again
                + "\n\n## Standings\n\n" + self._standings() + "\n"
            )
            self.dispatch("tuner", what, turn_prompt(PACKAGE, request))
        if not path.exists():
            return []
        return [dict(r) for r in yaml.safe_load(path.read_text()) or []]

    def campaign(self) -> None:
        s, cfg = self.spec, self.spec.campaign
        if not cfg.get("enabled", True):
            self.t["phase"] = "wrap"
            self.checkpoint()
            return
        acc = cfg.get("accuracy") or {}
        camp = Campaign(
            workdir=s.workdir, repo=s.repo, bench=s.bench(), base_configs=s.identity_configs,
            perf_runner=self.runners(s.spools, "tunec"), gate_runner=self.runners(s.dev_spools, "tuneg"),
            gate_cmd=gate_command(PACKAGE, s.env_script, s.model, float(acc.get("anchor", 0.0)), acc.get("task", "gsm8k"),
                                  tests=self._gate_tests()),
            noise=NoiseFloor(spread=dict(self.t.get("noise_spread") or {}), samples=int(self.t.get("noise_samples", 0))),
            min_gain=float(cfg.get("min_gain", 0.01)), noise_factor=float(cfg.get("noise_factor", 3.0)),
            branch_prefix=self.branch_prefix, log=self.log,
        )  # fmt: skip
        worktrees = s.workdir / "worktrees"
        if not camp.state.best_tree:
            camp.state.best_tree = str(open_candidate(s.repo, worktrees, "best", "HEAD", prefix=self.branch_prefix))
            camp.state.best_commit = git(Path(camp.state.best_tree), "rev-parse", "HEAD")
            camp.state.config_overlays = list(self.t.get("config_overlays") or [])
        if not camp.state.control_label:
            # The ladder's last wave measured the best tree at the final overlay on
            # each of its hosts: those are the campaign's first same-host controls.
            # Other hosts get theirs lazily, behind the first measurement that needs
            # one -- the tuner turns start now instead of after a control run.
            lad = s.workdir / "ladder"
            labels = json.loads((lad / "ladder_state.json").read_text()).get("control_labels") or [] if (lad / "ladder_state.json").exists() else []
            for label in labels:
                if (lad / "data" / label / "meta.json").exists():
                    camp.set_control(lad / "data" / label)
        n_road = int(self.t.get("roadmaps", 1))
        roadmap = [r for r in self._roadmap(n_road, tried=list(camp.state.candidates) if n_road > 1 else ())
                   if r.get("slug") not in camp.state.candidates]
        max_roadmaps = int(cfg.get("max_roadmaps", 2))
        max_par = int(cfg.get("max_parallel_tuners", 4))
        misses, budget = int(self.t.get("misses", 0)), int(cfg.get("max_iterations", 12))
        max_misses = int(cfg.get("max_consecutive_misses", 3))
        futures: dict[str, Future] = {}
        with ThreadPoolExecutor(max_workers=max(1, max_par), thread_name_prefix="tuner") as pool:
            while True:
                while roadmap and len(futures) < max_par and len(camp.state.candidates) + len(futures) < budget:
                    item = roadmap.pop(0)
                    slug = str(item["slug"])
                    done = s.workdir / "candidates" / slug / "decision.yaml"
                    if done.exists():
                        # The turn finished while no orchestrator watched it (a restart):
                        # take its decision instead of asking the same question again.
                        try:
                            read_decision(done)
                            camp.submit_candidate(slug, done)
                            self.log(f"[tune] {slug}: decision found on resume; not re-dispatched")
                            continue
                        except (OSError, ValueError) as exc:
                            self.log(f"[tune] {slug}: unreadable decision on resume ({exc}); dispatching")
                    base = camp.state.best_commit or "HEAD"
                    wt = open_candidate(s.repo, worktrees, slug, base, prefix=self.branch_prefix)
                    req = render_request(
                        self._context(self._standings()), slug=slug, worktree=wt, base=base,
                        branch=f"{self.branch_prefix}/{slug}",
                        hypothesis=str(item.get("hypothesis", "")), job_prefix=slug[:12],
                        deliverables=str(item.get("deliverables", "Commits, entry tests, micro-bench, decision.yaml.")),
                        modeling_edits=bool(item.get("modeling_edits", True)),
                    )  # fmt: skip
                    futures[slug] = pool.submit(self.dispatch, "tuner", slug, turn_prompt(PACKAGE, req))
                for slug, fut in list(futures.items()):
                    if fut.done():
                        del futures[slug]
                        decision = s.workdir / "candidates" / slug / "decision.yaml"
                        try:
                            read_decision(decision)
                            if slug in camp.state.candidates and camp.state.candidates[slug].status not in ("conflict", "blocked"):
                                continue
                            camp.submit_candidate(slug, decision)
                        except (OSError, ValueError) as exc:
                            self.log(f"[tune] {slug}: no valid decision ({exc}); counted as a miss")
                            misses += 1
                busy = camp.step()
                for st in list(camp.state.candidates.values()):
                    # One rebase per tree it conflicted with: a winner left out of a batch is
                    # rebased onto the batch (pinned to its commit) while the batch is measured.
                    key = f"rebase-{st.slug}-{st.rebase_ref[:10]}"
                    if st.status == "conflict" and not self.t.get(key) and st.slug not in futures and st.worktree:
                        # A textual conflict with an earlier keep is not a verdict: rebase and retry.
                        self.t[key] = True
                        onto = st.rebase_ref if st.rebase_onto == "tune/eval-batch" else f"{self.branch_prefix}/best"
                        req = render_rebase_request(self._context(self._standings()), slug=st.slug,
                                                    worktree=Path(st.worktree), best_branch=onto, reason=st.reason)
                        futures[st.slug] = pool.submit(self.dispatch, "tuner", f"rebase-{st.slug}", turn_prompt(PACKAGE, req))
                for st in camp.state.candidates.values():
                    if st.status in ("kept",) and not self.t.get(f"counted-{st.slug}"):
                        self.t[f"counted-{st.slug}"] = True
                        misses = 0
                    elif st.status in ("reverted", "dropped", "blocked") and not self.t.get(f"counted-{st.slug}"):
                        self.t[f"counted-{st.slug}"] = True
                        misses += 1
                self.t["misses"] = misses
                self.checkpoint()
                n = int(self.t.get("roadmaps", 1))
                if (not futures and not busy and not roadmap and misses < max_misses and len(camp.state.candidates) < budget
                        and n < max_roadmaps):
                    # The roadmap is spent with budget left: profile the moved best tree for the next one.
                    self.t["roadmaps"] = n + 1
                    self.checkpoint()
                    roadmap = [r for r in self._roadmap(n + 1, tried=list(camp.state.candidates)) if r.get("slug") not in camp.state.candidates]
                    continue
                if not futures and not busy and (not roadmap or misses >= max_misses or len(camp.state.candidates) >= budget):
                    break
                time.sleep(self.poll_s)
        self.t["best_tree"], self.t["best_commit"] = camp.state.best_tree, camp.state.best_commit
        self.t["kept"] = list(camp.state.kept)
        self.t["phase"] = "wrap"
        self.checkpoint()

    def wrap(self) -> str:
        s = self.spec
        overlays = [Path(p) for p in self.t.get("config_overlays") or []]
        labels: dict[str, Sequence[Path]] = {"t-final": [*s.identity_configs, *overlays]}
        got = self._measure(labels, tree=self.t.get("best_tree") or None)
        if s.reference == "stock" and overlays:
            stock = _overlay_file(s.workdir, "stock", {"modeling_v2": "off"})
            got.update(self._measure({"t-reference-tuned": [*s.identity_configs, *overlays, stock]}))
        rows = [Measurement.load(self.data / l) for l in ("t-baseline", "t-reference", "t-reference-tuned", "t-control", "t-final")
                if (self.data / l / "meta.json").exists()]  # fmt: skip
        summary = s.workdir / "summary.md"
        summary.write_text(
            "# Tune stage summary\n\n"
            + format_table(rows, rows[0] if rows else None)
            + f"\n\nLadder kept: {self.t.get('ladder_kept', [])}\n\nModeling kept: {self.t.get('kept', [])}\n"
            + f"\nBest tree: {self.t.get('best_tree', '')} @ {str(self.t.get('best_commit', ''))[:10]}\n"
        )
        self.t["phase"] = "done"
        self.checkpoint()
        return "tuned"

    def run(self) -> str | None:
        if self.t["phase"] == "ground" and not self.ground():
            return "tune-failed:baseline"
        if self.t["phase"] == "ladder":
            self.ladder()
        if self.t["phase"] == "campaign":
            self.campaign()
        if self.t["phase"] == "wrap":
            return self.wrap()
        return "tuned" if self.t["phase"] == "done" else None


def run_tune_phase(wf: Any, state: Any, log: Any) -> str | None:
    """The workflow's entry: build the spec from task.yaml and run the stage."""
    errors = validate_perf_block(wf.task_data)
    if errors or not wf.task_data.get("perf"):
        wf.journal.event("tune not started", "; ".join(errors) or "task.yaml has no perf block")
        return "tune-failed:spec"
    target = getattr(getattr(wf, "_preflight", None), "target_rel", "") or ""
    spec = TuneSpec.from_task(wf.task_data, wf.workspace, target=str(target))

    def dispatch(role: str, what: str, prompt: str) -> str:
        return wf._run_turn(role, what, prompt, unit="tune")

    stage = TuneStage(spec, state.tune, dispatch=dispatch, checkpoint=lambda: wf._checkpoint(state),
                      log=lambda m: wf.journal.event("tune", m))  # fmt: skip
    return stage.run()
