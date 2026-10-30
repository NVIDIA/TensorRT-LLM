# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``trtllm-modeling-v2 tune ...``: the perf stage's command line.

tune campaign run|status|step|submit --spec tune.yaml   drive or inspect a modeling campaign
tune table         --spec tune.yaml [DIR ...]   markdown curve table of label dirs
tune ab --a A --b B --a A2 [...]                judge an A/B the way the campaign does
        [--changed-members M,..] [--path-marker T [--marker-log LOG ...]]

``run`` and ``step`` keep candidates, so they need the keep gate: the spec is then a whole
task.yaml whose contract gives the anchor (``keep_gate.py``), or its
``perf.campaign.accuracy`` says ``{gate: off}``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from .campaign import Campaign
from .inert import inert_members
from .keep_gate import KeepGate, KeepGateError, resolve_keep_gate
from .measure import BenchSpec, SpoolRunner
from .records import Measurement, NoiseFloor, compare, format_table, remeasure_reason
from .schema import CONTRACT_ACCURACY_KEYS
from .worktree import open_candidate


def load_spec(path: Path) -> tuple[dict, dict]:
    """The task a spec file is (empty for a bare perf spec) and its perf block."""
    spec = yaml.safe_load(path.read_text()) or {}
    perf = spec.get("perf", spec)  # a full task.yaml carries it under `perf:`
    for key in ("workdir", "model", "identity_configs", "spools", "env_script", "tree"):
        if key not in perf:
            raise SystemExit(f"{path}: perf spec is missing {key!r}")
    return (spec if "perf" in spec else {}), perf


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


def campaign_dir(perf: dict) -> Path:
    c = perf.get("campaign", {}) or {}
    return Path(c.get("workdir") or Path(perf["workdir"]).parent / "campaign")


def keep_gate_for(task: dict, perf: dict) -> KeepGate:
    """The keep gate a campaign run from this spec keeps through; SystemExit when there is none."""
    c = perf.get("campaign", {}) or {}
    stale = sorted(set(c) & set(CONTRACT_ACCURACY_KEYS))
    if stale:
        raise SystemExit(
            f"campaign.{stale}: the keep gate is the contract's accuracy gate and reads "
            + "; ".join(CONTRACT_ACCURACY_KEYS[k] for k in stale)
        )
    try:
        return resolve_keep_gate(
            task, c, workspace=campaign_dir(perf), tests=c.get("gate_tests") or ()
        )
    except KeepGateError as exc:
        raise SystemExit(
            f"the campaign cannot keep anything: {exc}. Give the spec a contract anchor (a "
            "whole task.yaml) or `campaign: {accuracy: {gate: off}}`"
        ) from exc


def build_campaign(perf: dict, log=print, keep_gate: KeepGate | None = None) -> Campaign:
    c = perf.get("campaign", {}) or {}
    workdir = campaign_dir(perf)
    workdir.mkdir(parents=True, exist_ok=True)
    repo = Path(perf["tree"])
    prefix = str(
        c.get("branch_prefix", "tune")
    )  # a second campaign in one repository needs its own
    best = open_candidate(
        repo,
        Path(c.get("worktrees", workdir / "worktrees")),
        "best",
        c.get("base", "HEAD"),
        prefix=prefix,
    )
    noise_spread = c.get("noise_spread") or {}
    # A declared floor counts on the tree `noise_commit` names only; without one, on every tree.
    noise_commit = c.get("noise_commit")
    camp = Campaign(
        workdir=workdir,
        repo=repo,
        bench=bench_spec(perf),
        base_configs=[Path(p) for p in perf["identity_configs"]],
        perf_runner=SpoolRunner(c.get("perf_spools", perf["spools"]), prefix="camp"),
        gate_runner=SpoolRunner(c.get("gate_spools", perf["spools"]), prefix="camp"),
        keep_gate=keep_gate,
        noise=NoiseFloor(spread=noise_spread, samples=int(c.get("noise_samples", 0))),
        noise_commit=str(noise_commit) if noise_commit else None,
        min_gain=float(c.get("min_gain", 0.01)),
        noise_factor=float(c.get("noise_factor", 3.0)),
        branch_prefix=prefix,
        log=log,
        # No tuner turn follows a split here: it is journaled with its mode and reverted.
        split_variants=False,
    )
    if not camp.state.best_tree:
        camp.state.best_tree = str(best)
        camp.state.best_commit = subprocess.run(
            ["git", "-C", str(best), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    if c.get("config_overlays") and not camp.state.config_overlays:
        camp.state.config_overlays = [str(p) for p in c["config_overlays"]]
    if c.get("control") and not camp.state.control_label:
        camp.set_control(Path(c["control"]))
    camp.state.save(camp.state_path)
    return camp


def read_marker_logs(paths: Sequence[Path], marker: str) -> dict[str, str]:
    """``{path: text}`` of each ``--marker-log``; SystemExit for one that does not show ``marker``.

    Such a log says nothing about where the path runs, and taking it would make the marker's
    absence elsewhere read as a measurement.
    """
    if paths and not marker:
        raise SystemExit("--marker-log needs --path-marker: the line to find in it")
    logs = {}
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SystemExit(f"--marker-log {path}: {exc}") from exc
        if marker not in text:
            raise SystemExit(f"--marker-log {path}: `{marker}` is not in it")
        logs[str(path)] = text
    return logs


def judge_ab(
    controls: list[Measurement],
    candidates: list[Measurement],
    *,
    min_gain: float,
    noise_factor: float,
    changed: Sequence[str] = (),
    marker: str = "",
    marker_logs: Mapping[str, str] | None = None,
) -> str:
    """The campaign's verdict on B against A and A' of one node, as lines to print.

    The floor is the controls' own spread, per site; the members the candidate provably
    does not change (``changed``, ``marker``, ``marker_logs``: :mod:`.inert`) are judged
    nowhere; and a verdict one disturbed run explains says to measure B and A once more,
    interleaved.
    """
    hosts = sorted({str(m.meta.get("hostname") or "?") for m in [*controls, *candidates]})
    lines = [f"controls: {', '.join(m.label for m in controls)}",
             f"candidate: {', '.join(m.label for m in candidates)}", f"hosts: {', '.join(hosts)}"]  # fmt: skip
    if len(hosts) > 1:
        lines.append("warning: not one node; hosts differ by more than most candidates move")
    if len(controls) < 2:
        lines.append("warning: one control measures no floor; measure A' after B on the same node")
    noise = NoiseFloor.from_measurements(controls)
    inert = inert_members(
        candidates, controls, changed=changed, marker=marker, marker_logs=marker_logs
    )
    cmp = compare(Measurement.mean(candidates), Measurement.mean(controls), noise,
                  min_gain=min_gain, noise_factor=noise_factor, inert=inert)  # fmt: skip
    lines += [f"inert {member}: {why}" for member, why in inert.items()]
    lines.append(f"verdict: {cmp.verdict}: {cmp.reason}")
    why = remeasure_reason(cmp, noise, min_gain=min_gain, noise_factor=noise_factor)
    if why and len(candidates) == 1:
        lines.append(
            f"re-measure before it counts ({why}): B again, then A again, on this node, each "
            "started on an idle node; then judge with every A and both B"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trtllm-modeling-v2 perf")
    sub = parser.add_subparsers(dest="cmd", required=True)
    camp = sub.add_parser("campaign")
    camp.add_argument("action", choices=["status", "submit", "run", "step"])
    camp.add_argument("--spec", required=True, type=Path)
    camp.add_argument("--slug", default="")
    camp.add_argument("--decision", type=Path, default=None)
    camp.add_argument("--poll", type=float, default=30.0)
    camp.add_argument(
        "--idle-exit", type=float, default=0.0, help="seconds idle before `run` exits; 0 = never"
    )
    tab = sub.add_parser("table")
    tab.add_argument("--baseline", type=Path, default=None)
    tab.add_argument("dirs", nargs="+", type=Path)
    ab = sub.add_parser("ab", help="judge B against A and A' of one node, as the campaign does")
    ab.add_argument("--a", action="append", type=Path, required=True, help="a control; repeat")
    ab.add_argument("--b", action="append", type=Path, required=True, help="the candidate; repeat")
    ab.add_argument(
        "--changed-members",
        default="",
        help="comma-separated perf configs it runs at; in a round of none of them, a member "
        "whose engine reads as the control's is not judged",
    )
    ab.add_argument("--path-marker", default="", help="a line the change prints on its path")
    ab.add_argument(
        "--marker-log",
        action="append",
        type=Path,
        default=[],
        help="a log outside these measurements where the path marker printed; repeat",
    )
    ab.add_argument("--min-gain", type=float, default=0.01)
    ab.add_argument("--noise-factor", type=float, default=3.0)
    args = parser.parse_args(argv)

    if args.cmd == "campaign":
        task, perf = load_spec(args.spec)
        log = lambda msg: print(msg, flush=True)  # noqa: E731
        gate = None
        if args.action in ("run", "step"):
            gate = keep_gate_for(task, perf)
            log(f"[campaign] keep gate: {gate.describe()}")
        c = build_campaign(perf, log=log, keep_gate=gate)
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
    if args.cmd == "ab":
        print(
            judge_ab(
                [Measurement.load(d) for d in args.a],
                [Measurement.load(d) for d in args.b],
                min_gain=args.min_gain,
                noise_factor=args.noise_factor,
                changed=[m.strip() for m in args.changed_members.split(",") if m.strip()],
                marker=args.path_marker,
                marker_logs=read_marker_logs(args.marker_log, args.path_marker),
            )
        )
        return 0
    if args.cmd == "table":
        ms = [Measurement.load(d) for d in args.dirs if (d / "summary.json").exists()]
        base = Measurement.load(args.baseline) if args.baseline else (ms[0] if ms else None)
        print(format_table(ms, base))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
