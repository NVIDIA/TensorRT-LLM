# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``staircase-onboard tune ...``: the perf stage's command line.

    tune ladder run    --spec tune.yaml      drive the config ladder to completion
    tune ladder status --spec tune.yaml      where the ladder stands
    tune table         --spec tune.yaml [DIR ...]   markdown curve table of label dirs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .campaign import Campaign, gate_command
from .ladder import BenchSpec, Ladder, SpoolRunner, default_rungs, rungs_from_spec
from .records import Measurement, NoiseFloor, format_table
from .worktree import open_candidate


def load_spec(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text()) or {}
    perf = spec.get("perf", spec)  # a full task.yaml carries it under `perf:`
    for key in ("workdir", "model", "identity_configs", "spools", "env_script", "tree"):
        if key not in perf:
            raise SystemExit(f"{path}: perf spec is missing {key!r}")
    return perf


def build_ladder(perf: dict, log=print) -> Ladder:
    workdir = Path(perf["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    pkg = Path(__file__).resolve().parent
    bench = BenchSpec(
        bench_py=str(pkg / "bench.py"),
        model=perf["model"],
        env_script=perf["env_script"],
        isl=int(perf.get("isl", 8192)),
        osl=int(perf.get("osl", 1024)),
        concurrencies=[int(c) for c in perf.get("concurrency", [1, 128])],
        tree=perf["tree"],
        extra_args=perf.get("bench_args", []),
        wall=perf.get("wall", "2h"),
    )
    ladder_spec = perf.get("ladder", {}) or {}
    skip = set(ladder_spec.get("skip", []))
    only = set(ladder_spec.get("only", []))
    rungs = [r for r in default_rungs() + rungs_from_spec(ladder_spec.get("extra_rungs", [])) if r.name not in skip]
    if only:
        rungs = [r for r in rungs if r.name in only]
    runner = SpoolRunner(perf["spools"], prefix=ladder_spec.get("job_prefix", "tune"))
    ladder = Ladder(
        workdir=workdir,
        base_configs=[Path(p) for p in perf["identity_configs"]],
        rungs=rungs,
        bench=bench,
        runner=runner,
        min_gain=float(ladder_spec.get("min_gain", 0.03)),
        noise_factor=float(ladder_spec.get("noise_factor", 2.0)),
        target_mode=perf.get("modeling_v2", "require") == "require",
        log=log,
    )
    seeds = [Path(p) for p in ladder_spec.get("seed_controls", [])]
    if seeds:
        ladder.seed_controls(seeds)
    return ladder


def bench_spec(perf: dict) -> BenchSpec:
    pkg = Path(__file__).resolve().parent
    return BenchSpec(
        bench_py=str(pkg / "bench.py"),
        model=perf["model"],
        env_script=perf["env_script"],
        isl=int(perf.get("isl", 8192)),
        osl=int(perf.get("osl", 1024)),
        concurrencies=[int(c) for c in perf.get("concurrency", [1, 128])],
        tree=perf["tree"],
        extra_args=perf.get("bench_args", []),
        wall=perf.get("wall", "2h"),
    )


def build_campaign(perf: dict, log=print) -> Campaign:
    c = perf.get("campaign", {}) or {}
    workdir = Path(c.get("workdir") or Path(perf["workdir"]).parent / "campaign")
    workdir.mkdir(parents=True, exist_ok=True)
    repo = Path(perf["tree"])
    prefix = str(c.get("branch_prefix", "tune"))  # a second campaign in one repository needs its own
    best = open_candidate(repo, Path(c.get("worktrees", workdir / "worktrees")), "best", c.get("base", "HEAD"), prefix=prefix)
    noise_spread = c.get("noise_spread") or {}
    camp = Campaign(
        workdir=workdir,
        repo=repo,
        bench=bench_spec(perf),
        base_configs=[Path(p) for p in perf["identity_configs"]],
        perf_runner=SpoolRunner(c.get("perf_spools", perf["spools"]), prefix="camp"),
        gate_runner=SpoolRunner(c.get("gate_spools", perf["spools"]), prefix="camp"),
        gate_cmd=gate_command(Path(__file__).resolve().parent.parent, Path(perf["env_script"]), perf["model"],
                              float(c.get("anchor", 0.0)), c.get("task", "gsm8k"), tests=c.get("gate_tests") or ()),
        noise=NoiseFloor(spread=noise_spread, samples=int(c.get("noise_samples", 0))),
        min_gain=float(c.get("min_gain", 0.01)),
        noise_factor=float(c.get("noise_factor", 3.0)),
        branch_prefix=prefix,
        log=log,
    )
    if not camp.state.best_tree:
        camp.state.best_tree = str(best)
        camp.state.best_commit = subprocess.run(["git", "-C", str(best), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if c.get("config_overlays") and not camp.state.config_overlays:
        camp.state.config_overlays = [str(p) for p in c["config_overlays"]]
    if c.get("control") and not camp.state.control_label:
        camp.set_control(Path(c["control"]))
    camp.state.save(camp.state_path)
    return camp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="staircase-onboard tune")
    sub = parser.add_subparsers(dest="cmd", required=True)
    lad = sub.add_parser("ladder")
    lad.add_argument("action", choices=["run", "status", "step"])
    lad.add_argument("--spec", required=True, type=Path)
    lad.add_argument("--poll", type=float, default=20.0)
    camp = sub.add_parser("campaign")
    camp.add_argument("action", choices=["status", "submit", "run", "step"])
    camp.add_argument("--spec", required=True, type=Path)
    camp.add_argument("--slug", default="")
    camp.add_argument("--decision", type=Path, default=None)
    camp.add_argument("--poll", type=float, default=30.0)
    camp.add_argument("--idle-exit", type=float, default=0.0, help="seconds idle before `run` exits; 0 = never")
    tab = sub.add_parser("table")
    tab.add_argument("--baseline", type=Path, default=None)
    tab.add_argument("dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "ladder":
        perf = load_spec(args.spec)
        log = lambda msg: print(msg, flush=True)  # noqa: E731
        ladder = build_ladder(perf, log=log)
        if args.action == "status":
            print(ladder.summary())
        elif args.action == "step":
            ladder.step()
            print(ladder.summary())
        else:
            result = ladder.run(poll_s=args.poll)
            print(ladder.summary())
            print(f"[ladder] result: {result}")
        return 0
    if args.cmd == "campaign":
        perf = load_spec(args.spec)
        log = lambda msg: print(msg, flush=True)  # noqa: E731
        c = build_campaign(perf, log=log)
        if args.action == "submit":
            # The running loop ingests it; two processes never write the state file.
            print(f"[campaign] queued {c.enqueue(args.slug, args.decision)}")
        elif args.action == "step":
            c.step()
        elif args.action == "run":
            # A campaign loop stays up while candidates may still arrive (--idle-exit 0).
            idle = 0.0
            while True:
                busy = c.step()
                inbox = list((c.workdir / "inbox").glob("*.json"))
                if busy or inbox:
                    idle = 0.0
                elif args.idle_exit and idle >= args.idle_exit:
                    break
                time.sleep(args.poll)
                idle += args.poll
        print(c.summary())
        return 0
    if args.cmd == "table":
        ms = [Measurement.load(d) for d in args.dirs if (d / "summary.json").exists()]
        base = Measurement.load(args.baseline) if args.baseline else (ms[0] if ms else None)
        print(format_table(ms, base))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
