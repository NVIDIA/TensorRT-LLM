# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare confidence scheduling with fixed draft lengths at equal per-user speed.

Reads the output directories of run_benchmark.sh, each with the config it ran. Runs of the same
dataset, batch size, method, draft length and re-ranking setting are averaged; confidence settings
with and without device re-ranking both count as confidence scheduling, so each point uses
whichever is faster (--no-rerank leaves out the re-ranking runs). Runs with tensor-parallel
attention (no attention DP) count as a dataset of their own, "<dataset> TP attention". For each
dataset and method, the points that no other point beats on both per-user speed (tok/s/user) and
throughput per GPU (tok/s/GPU) form a line; neighbouring points are joined linearly. The two lines
are compared where both exist, without extrapolation, and the largest gain of confidence scheduling
over fixed draft lengths is reported. The ratio of two linear pieces peaks at an end, so the maximum
is found at a measured point of either line.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import yaml

GPUS = 8


def round_half_up(x: float, digits: int = 0) -> float:
    scale = 10**digits
    return math.floor(x * scale + 0.5) / scale


def load_runs(run_dirs: list[Path]) -> list[dict]:
    runs = []
    for run_dir in sorted(run_dirs):
        if not (run_dir / "report.json").exists():
            continue
        meta = json.loads((run_dir / "run.json").read_text())
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        spec = config["speculative_config"]
        confidence = bool(spec.get("enable_confidence_scheduling"))
        if confidence and "enable_fused_confidence_scheduler" not in spec:
            sys.exit(
                f"{run_dir}/config.yaml does not set enable_fused_confidence_scheduler, so its "
                "re-ranking setting is unknown; write configs with make_configs.py"
            )
        perf = json.loads((run_dir / "report.json").read_text())["performance"]
        runs.append(
            {
                "dataset": meta["dataset"]
                + ("" if config.get("enable_attention_dp") else " TP attention"),
                "bs": meta["batch_size"],
                "method": "confidence" if confidence else "fixed",
                "k": spec["max_draft_len"],
                "rerank": confidence and spec["enable_fused_confidence_scheduler"],
                "tps_gpu": perf["system_output_throughput_tok_s"] / GPUS,
                "tps_user": perf["output_throughput_per_user_tok_s"],
            }
        )
    return runs


def average(runs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in runs:
        groups[(r["dataset"], r["bs"], r["method"], r["k"], r["rerank"])].append(r)
    return [
        {
            "dataset": d,
            "bs": bs,
            "method": m,
            "k": k,
            "rerank": rerank,
            "n": len(rs),
            "tps_gpu": round_half_up(sum(r["tps_gpu"] for r in rs) / len(rs)),
            "tps_user": round_half_up(sum(r["tps_user"] for r in rs) / len(rs), 1),
        }
        for (d, bs, m, k, rerank), rs in groups.items()
    ]


def setting(point: dict) -> str:
    return f"K{point['k']}{' re-rank' if point['rerank'] else ''}"


def frontier(points: list[dict]) -> list[dict]:
    best = [
        p
        for i, p in enumerate(points)
        if not any(
            q is not p
            and q["tps_user"] >= p["tps_user"]
            and q["tps_gpu"] >= p["tps_gpu"]
            and (q["tps_user"] > p["tps_user"] or q["tps_gpu"] > p["tps_gpu"] or j < i)
            for j, q in enumerate(points)
        )
    ]
    return sorted(best, key=lambda p: p["tps_user"])


def value_at(line: list[dict], x: float) -> float:
    for a, b in zip(line, line[1:]):
        if x <= b["tps_user"]:
            return a["tps_gpu"] + (x - a["tps_user"]) / (b["tps_user"] - a["tps_user"]) * (
                b["tps_gpu"] - a["tps_gpu"]
            )
    return line[-1]["tps_gpu"]


def iso_gain(fixed: list[dict], conf: list[dict]) -> list[tuple[float, float, float, float]]:
    """(tok/s/user, fixed and confidence tok/s/GPU, gain %) at the points within both lines' range."""
    if len(fixed) < 2 or len(conf) < 2:
        return []
    lo = max(fixed[0]["tps_user"], conf[0]["tps_user"])
    hi = min(fixed[-1]["tps_user"], conf[-1]["tps_user"])
    if not lo < hi:
        return []
    xs = {lo, hi} | {p["tps_user"] for p in fixed + conf if lo <= p["tps_user"] <= hi}
    out = []
    for x in sorted(xs):
        f, c = value_at(fixed, x), value_at(conf, x)
        out.append((x, f, c, round_half_up((c / f - 1) * 100, 1)))
    return out


def describe(point: dict | None) -> str:
    if point is None:
        return "-"
    return f"{setting(point)} {point['tps_gpu']:.0f}, {point['tps_user']:.1f} (n={point['n']})"


def plot(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(2, len(results))
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4.8 * rows), squeeze=False)
    for ax in axes.flat[len(results) :]:
        ax.set_visible(False)
    colors = {"confidence": "tab:blue", "fixed": "tab:gray"}
    for ax, (dataset, r) in zip(axes.flat, results.items()):
        for method in ("fixed", "confidence"):
            pts = [p for p in r["points"] if p["method"] == method]
            ax.scatter(
                [p["tps_user"] for p in pts],
                [p["tps_gpu"] for p in pts],
                s=14,
                color=colors[method],
                alpha=0.4,
            )
            line = r[method]
            ax.plot(
                [p["tps_user"] for p in line],
                [p["tps_gpu"] for p in line],
                color=colors[method],
                marker="o",
                ms=4,
                label="confidence scheduling" if method == "confidence" else "fixed draft length",
            )
            for p in line:
                ax.annotate(
                    f"{p['bs']} {setting(p)}",
                    (p["tps_user"], p["tps_gpu"]),
                    fontsize=7,
                    textcoords="offset points",
                    xytext=(0, 6 if method == "confidence" else -11),
                    ha="center",
                    color=colors[method],
                )
        if r["best"]:
            x, f, c, g = r["best"]
            ax.annotate(
                "", xy=(x, c), xytext=(x, f), arrowprops={"arrowstyle": "->", "color": "tab:green"}
            )
            ax.text(x, (f + c) / 2, f"  {g:+.1f}%", color="tab:green", va="center")
        ax.set_title(dataset)
        ax.set_xlabel("output tok/s per user")
        ax.set_ylabel("output tok/s per GPU")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "run_dirs", type=Path, nargs="+", help="output directories of run_benchmark.sh"
    )
    parser.add_argument("--plot", type=Path, help="write a PNG of the per-dataset lines")
    parser.add_argument(
        "--no-rerank", action="store_true", help="leave out runs with device re-ranking"
    )
    args = parser.parse_args()

    runs = load_runs(args.run_dirs)
    if args.no_rerank:
        runs = [r for r in runs if not r["rerank"]]
    points = average(runs)
    results = {}
    for dataset in sorted({p["dataset"] for p in points}):
        pts = [p for p in points if p["dataset"] == dataset]
        r = {"points": pts}
        for method in ("fixed", "confidence"):
            r[method] = frontier([p for p in pts if p["method"] == method])
        curve = iso_gain(r["fixed"], r["confidence"])
        r["best"] = max(curve, key=lambda row: row[3]) if curve else None
        results[dataset] = r

        print(f"\n== {dataset}")
        print(
            "batch size | best fixed (tok/s/GPU, tok/s/user) | best confidence | gain at same batch size"
        )
        for bs in sorted({p["bs"] for p in pts}):
            top = {
                m: max(
                    (p for p in pts if p["bs"] == bs and p["method"] == m),
                    key=lambda p: p["tps_gpu"],
                    default=None,
                )
                for m in ("fixed", "confidence")
            }
            f, c = top["fixed"], top["confidence"]
            gain = f"{(c['tps_gpu'] / f['tps_gpu'] - 1) * 100:+.1f}%" if f and c else "-"
            print(f"{bs:>10} | {describe(f):>34} | {describe(c):>30} | {gain}")
        if r["best"]:
            x, f, c, g = r["best"]
            print(
                f"Largest gain at equal per-user speed: {g:+.1f}% at {x:.1f} tok/s/user "
                f"({f:.0f} -> {c:.0f} tok/s/GPU)"
            )
        else:
            print("Equal-speed comparison needs both methods at two or more batch sizes.")
    if args.plot:
        plot(results, args.plot)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
