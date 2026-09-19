# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the blog's figures and statistics from the bundled timing data.

Requires matplotlib and numpy. Run from any directory; outputs stay beside this file.
"""

import csv
import gzip
import json
from pathlib import Path
from statistics import geometric_mean, mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
COPYRIGHT = (
    "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. SPDX-License-Identifier: Apache-2.0"
)
ARMS = ["radix_cuda"]
MODELS = {
    "flash": "DeepSeek-V4 Flash · K=512",
    "pro": "DeepSeek-V4 Pro · K=1024",
    "v32": "DeepSeek-V3.2 · K=2048",
}
LABELS = {
    "gvr_v2": "GVR V2",
    "temporal_tiered": "GVR V1",
    "radix_cuda": "TensorRT-LLM radix CUDA",
}
COLORS = {
    "gvr_v2": "#579600",
    "temporal_tiered": "#386781",
    "radix_cuda": "#64748b",
}
REACHABLE_BW = 6.912116
TEMPORAL = ["temporal_tiered"]
CROSS_CAMPAIGN = set(TEMPORAL + ARMS)
COMPARISON_ARMS = ["gvr_v2", *TEMPORAL, *ARMS]


def _load() -> list[dict]:
    rows = []
    for path in sorted(ROOT.glob("*_timings.csv.gz")):
        with gzip.open(path, "rt") as handle:
            for row in csv.DictReader(line for line in handle if not line.startswith("#")):
                for name in ("batch", "n", "k", "layer"):
                    row[name] = int(row[name])
                for name in list(row):
                    if name.endswith("_us"):
                        row[name] = float(row[name]) if row[name] else None
                rows.append(row)
    if len(rows) != 9746 or len({(r["cell"], r["batch"]) for r in rows}) != 9746:
        raise ValueError("Expected exactly 9,746 unique cases")
    with gzip.open(ROOT / "temporal_comparison.csv.gz", "rt") as handle:
        historical = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    lookup = {(r["cell"], int(r["batch"])): r for r in historical}
    if len(lookup) != len(rows) or len(historical) != len(rows):
        raise ValueError("Temporal observations must cover the same 9,746 unique cases")
    for row in rows:
        old = lookup[(row["cell"], row["batch"])]
        for arm in TEMPORAL:
            row[arm + "_us"] = float(old[arm + "_us"])
    return rows


def _stats(rows: list[dict], arm: str, reference: str = "gvr_v2") -> dict:
    valid = [r for r in rows if r[arm + "_us"] is not None]
    ratios = [r[arm + "_us"] / r[reference + "_us"] for r in valid]
    if not valid:
        return {"cases": 0}
    return {
        "cases": len(valid),
        "geomean": geometric_mean(ratios),
        "minimum": min(ratios),
        "p5": float(np.percentile(ratios, 5)),
        "p95": float(np.percentile(ratios, 95)),
        "wins": sum(x > 1 for x in ratios),
        "win_percent": 100 * mean(x > 1 for x in ratios),
        "baseline_median_us": median(r[arm + "_us"] for r in valid),
        "gvr_median_us": median(r[reference + "_us"] for r in valid),
    }


def _save(fig: plt.Figure, name: str) -> None:
    path = ROOT / (name + ".svg")
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={"Date": None, "Description": COPYRIGHT},
    )
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def _comparison(rows: list[dict]) -> dict:
    """Normalize every bar to V2 over one common case set per model."""
    result = {}
    for model in MODELS:
        matched = _matching(rows, model)
        result[model] = {
            "cases": len(matched),
            "layers": len({r["layer"] for r in matched}),
            "latency_relative_to_v2": {
                arm: _stats(matched, arm)["geomean"] for arm in COMPARISON_ARMS
            },
        }
    return result


def _overview(rows: list[dict]) -> None:
    data = _comparison(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), sharey=True)
    fig.subplots_adjust(left=0.185, right=0.97, bottom=0.25, top=0.70, wspace=0.14)
    labels = [
        "GVR V2",
        "GVR V1",
        "TRT-LLM radix CUDA",
    ]
    positions = [2.8, 1.6, 0.4]
    for ax, (model, title) in zip(axes, MODELS.items()):
        panel = data[model]
        ax.axhspan(2.3, 3.3, color="#edf5df", zorder=0)
        ax.axvline(1, color="#579600", alpha=0.55, linewidth=1, linestyle=(0, (2, 3)))
        for y, arm in zip(positions, COMPARISON_ARMS):
            value = panel["latency_relative_to_v2"][arm]
            ax.barh(y, value, height=0.63, color=COLORS[arm], zorder=3)
            ax.text(
                value + 0.10,
                y,
                f"{value:.2f}×",
                va="center",
                fontsize=10.5,
                weight="bold" if arm == "gvr_v2" else "normal",
                color="#447a00" if arm == "gvr_v2" else "#334155",
            )
        name, kval = title.split(" · ")
        ax.set_title(name, loc="left", fontsize=12.5, weight="bold", pad=28)
        ax.text(
            0,
            1.025,
            kval,
            transform=ax.transAxes,
            fontsize=9.5,
            color="#52616f",
        )
        ax.set(xlim=(0, 5.95), ylim=(-0.15, 3.35), xticks=[0, 1, 2, 3, 4, 5])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
        ax.set_yticks(positions, labels, fontsize=10.5)
        ax.tick_params(axis="both", length=0, pad=8)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#d5dce3")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e9edf1", linewidth=0.7)
    axes[0].get_yticklabels()[0].set(color="#447a00", weight="bold")
    fig.text(
        0.035,
        0.935,
        "The GVR evolution: V1, V2, and radix CUDA",
        fontsize=20,
        weight="bold",
        color="#17202b",
    )
    fig.text(
        0.035,
        0.875,
        "Geometric-mean kernel time relative to GVR V2  ·  shorter is faster  ·  B200 / FP32",
        fontsize=11,
        color="#52616f",
    )
    fig.text(
        0.185,
        0.13,
        "Same workloads within each panel. GVR V2 (PR #19076) = 1.00×.",
        fontsize=10,
        color="#334155",
    )
    fig.text(
        0.035,
        0.067,
        "GVR V1: temporal hint calibration. GVR V2: current-row self-sampling.",
        fontsize=9,
        color="#52616f",
    )
    _save(fig, "speedup")


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    color: str = "#edf5df",
    fontsize: float = 11,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            *size,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=color,
            edgecolor="#ccd5dd",
            linewidth=0.8,
        )
    )
    ax.text(
        xy[0] + size[0] / 2,
        xy[1] + size[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#17202b",
        linespacing=1.5,
    )


def _evolution(rows: list[dict]) -> None:
    fig = plt.figure(figsize=(14, 5.6))
    ax = fig.add_axes((0.02, 0.16, 0.61, 0.69))
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")
    ax.text(
        0.2,
        5.55,
        "GVR V1 (temporal hint): calibration + multi-thresholding",
        fontsize=11,
        weight="bold",
        color=COLORS["temporal_tiered"],
    )
    for x, label in [
        (0.2, "Previous indices\n→ current-score gather"),
        (3.6, "Hint-derived thresholds\n→ multiple exact counts"),
        (7.0, "Collect candidates\n→ exact refinement"),
    ]:
        _box(ax, (x, 3.8), (2.8, 1.2), label, "#eef3f7", 10)
    ax.text(
        0.2,
        3.25,
        "Biased toward previous winners; overlap varies across layers and steps.\n"
        "Weak hints add verification and recovery work.",
        fontsize=9.5,
        color="#475569",
    )
    ax.text(
        0.2,
        2.55,
        "GVR V2 streaming: self-sampling + multi-thresholding",
        fontsize=11,
        weight="bold",
        color="#447a00",
    )
    for x, label in [
        (0.2, "Coalesced current-row\nsample → tail bracket"),
        (3.6, "Full-row classification\n→ many exact counts"),
        (7.0, "Emit certain winners\n→ refine crossing bin"),
    ]:
        _box(ax, (x, 0.8), (2.8, 1.2), label, fontsize=10)
    for y in (4.4, 1.4):
        for x in (3.02, 6.42):
            ax.annotate(
                "",
                (x + 0.5, y),
                (x, y),
                arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.5},
            )
    ax.text(
        0.2,
        0.23,
        "No temporal-overlap dependency; one calibration rule for both phases.",
        fontsize=9.5,
        color="#447a00",
    )
    bars = fig.add_axes((0.76, 0.30, 0.21, 0.4))
    arms = [*TEMPORAL, "gvr_v2"]
    for i, arm in enumerate(arms):
        ratios = [r["radix_cuda_us"] / r[arm + "_us"] for r in rows]
        value = geometric_mean(ratios)
        bars.barh(i, value, height=0.52, color=COLORS[arm])
        bars.text(value + 0.1, i, f"{value:.2f}×", va="center", weight="bold", fontsize=12)
    bars.set_yticks(
        range(len(arms)),
        [LABELS[arm] for arm in arms],
        fontsize=10,
    )
    bars.invert_yaxis()
    bars.set_xlim(0, 6)
    bars.set_xticks([0, 1, 2, 3, 4, 5])
    bars.set_xlabel("Speedup over radix CUDA", fontsize=10)
    bars.set_title("Measured evolution", fontsize=12, loc="left", weight="bold", pad=18)
    bars.grid(axis="x", alpha=0.12)
    bars.set_axisbelow(True)
    fig.suptitle(
        "From temporal prediction to current-row calibration",
        x=0.03,
        ha="left",
        fontsize=20,
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.035,
        0.06,
        "Design goals: improve the practical performance floor and average latency; "
        "remove the temporal prior's framework lifecycle.",
        fontsize=10,
        color="#475569",
    )
    fig.text(
        0.035,
        0.005,
        "Flows show streaming paths; bars compare complete GVR V1 and GVR V2 implementations.",
        fontsize=9,
        color="#475569",
    )
    _save(fig, "evolution")


def _candidate_work() -> None:
    """Illustrate tail counts and candidate amplification without measured data."""
    fig = plt.figure(figsize=(16.7, 7.5), facecolor="white")
    ink, muted = "#17202b", "#52616f"
    green, orange, rose = "#447a00", "#b56b0b", "#ad4b70"
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set(xlim=(0, 1), ylim=(0, 1))
    canvas.axis("off")
    for x, width in ((0.025, 0.47), (0.52, 0.455)):
        canvas.add_patch(
            FancyBboxPatch(
                (x, 0.205),
                width,
                0.65,
                boxstyle="round,pad=0.008,rounding_size=0.014",
                facecolor="#f8fafc",
                edgecolor="#dce3e9",
                linewidth=0.9,
            )
        )
    fig.text(
        0.04, 0.947, "Threshold quality sets candidate work", fontsize=23, weight="bold", color=ink
    )
    fig.text(
        0.04,
        0.90,
        "A tighter threshold reduces extra candidates; exact counts keep admission safe.",
        fontsize=12.5,
        color=muted,
    )
    fig.text(
        0.044, 0.80, "A   Choose an admission threshold", fontsize=15.5, weight="bold", color=ink
    )
    fig.text(
        0.54, 0.80, "B   Explain the admitted population", fontsize=15.5, weight="bold", color=ink
    )

    # One finite row defines both panels, including the left-continuous tie jump.
    scores = np.array([1.4, 3.0, 5.8, 7.5, 9.0])
    multiplicities = np.array([70, 80, 70, 70, 60])
    row = np.repeat(scores, multiplicities)
    k, q, tau = 100, 4.4, 7.5
    admitted = int(np.count_nonzero(row >= q))
    at_boundary = int(np.count_nonzero(row >= tau))
    above_boundary = int(np.count_nonzero(row > tau))
    excess = at_boundary - k
    shell = admitted - at_boundary

    ax = fig.add_axes((0.089, 0.32, 0.335, 0.425), facecolor="#f8fafc")
    ax.set(xlim=(0, 10), ylim=(0, 3.9))
    ax.axhspan(1, 2.8, color="#edf4e5", zorder=0)
    for level in (1, 2.8):
        ax.axhline(level, color="#adc096", linewidth=1, linestyle=(0, (4, 4)))
    thresholds = np.r_[0, scores, 10]
    counts = np.array([np.count_nonzero(row >= t) / k for t in thresholds])
    ax.step(thresholds, counts, where="pre", color=ink, linewidth=2.3, zorder=3)
    ax.text(9.7, 3.28, "Over capacity", color=muted, fontsize=11.5, ha="right")
    ax.text(0.25, 0.25, "Too few candidates", color=muted, fontsize=11.5)
    ax.vlines(q, 0, admitted / k, color=green, linewidth=1.2, linestyles=(0, (3, 3)))
    ax.scatter([q], [admitted / k], s=110, color=green, edgecolor="white", linewidth=1.4, zorder=5)
    ax.text(
        4.6,
        2.31,
        r"$C_p=C(q)$",
        color=green,
        fontsize=15,
        ha="center",
    )
    ax.vlines(tau, 0, at_boundary / k, color=rose, linewidth=1.1, linestyles=(0, (3, 3)))
    ax.scatter([tau], [at_boundary / k], s=42, color=rose, zorder=6)
    ax.scatter([tau], [above_boundary / k], s=35, facecolor="white", edgecolor=rose, zorder=6)
    ax.text(
        7.6,
        1.56,
        r"$C(\tau)$",
        ha="center",
        fontsize=12.5,
        color=rose,
    )
    # Brackets split C(q), not the entire tie jump: K cuts through that jump.
    for start, stop, color, symbol in (
        (0, 1, green, r"$K$"),
        (1, at_boundary / k, rose, r"$E$"),
        (at_boundary / k, admitted / k, orange, r"$D$"),
    ):
        ax.plot(
            [10.14, 10.38, 10.38, 10.14],
            [start, start, stop, stop],
            color=color,
            linewidth=1.8,
            clip_on=False,
        )
        ax.text(
            10.72,
            (start + stop) / 2,
            symbol,
            color=color,
            fontsize=16,
            va="center",
            clip_on=False,
        )
    ax.hlines(
        [at_boundary / k, admitted / k],
        [tau, q],
        [10.14, 10.14],
        colors=[rose, green],
        linewidth=0.9,
        linestyles=(0, (3, 3)),
        clip_on=False,
    )
    ax.set_yticks([0, 1, 2.8], ["0", r"$K$", r"$B_r$"])
    ax.set_xticks([q, tau], [r"$q$", r"$\tau$"])
    for tick, color in zip(ax.get_xticklabels(), (green, rose)):
        tick.set_color(color)
    ax.set_ylabel(r"Candidates $C(t)$", fontsize=12.5, labelpad=10)
    ax.set_xlabel("Higher threshold →", fontsize=12, labelpad=7)
    ax.tick_params(length=0, pad=7, labelsize=13)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    fig.text(
        0.089,
        0.223,
        r"Safe admission:  $K\leq C(q)\leq B_r$",
        fontsize=12,
        color=green,
    )

    fig.text(0.55, 0.732, "CANDIDATE AMPLIFICATION", fontsize=10.5, weight="bold", color=muted)
    fig.text(0.55, 0.666, r"$C_p\,/\,K$", fontsize=26, color=green)
    fig.text(0.665, 0.678, "Candidates per required winner", fontsize=12.5, color=ink)
    right = fig.add_axes((0.55, 0.564, 0.392, 0.072))
    right.set(xlim=(0, admitted), ylim=(0, 1))
    right.axis("off")
    segments = [
        (k, "#deedc8", green, r"$K$", "Required output"),
        (excess, "#f3dce5", rose, r"$E$", r"Excess boundary ties: $C(\tau)-K$"),
        (shell, "#fae6c7", orange, r"$D$", r"Boundary shell: $q\leq x<\tau$"),
    ]
    left = 0
    for i, (width, fill, color, symbol, label) in enumerate(segments):
        right.add_patch(
            Rectangle((left, 0), width, 1, facecolor=fill, edgecolor="white", linewidth=2)
        )
        right.text(
            left + width / 2, 0.5, symbol, fontsize=21, ha="center", va="center", color=color
        )
        y = 0.499 - i * 0.079
        fig.text(0.555, y, symbol, fontsize=18, color=color, va="center")
        fig.text(0.589, y, label, fontsize=13, color=ink, va="center")
        canvas.plot([0.55, 0.943], [y - 0.037, y - 0.037], color="#e1e7ec", lw=0.8)
        left += width
    fig.text(0.746, 0.24, r"$C_p=C(q)=K+E+D(q,\tau)$", fontsize=19, ha="center", color=ink)

    canvas.add_patch(
        FancyBboxPatch(
            (0.027, 0.07),
            0.941,
            0.096,
            boxstyle="round,pad=0.008,rounding_size=0.013",
            facecolor="#edf4e5",
            edgecolor="none",
        )
    )
    fig.text(0.045, 0.113, "V2", fontsize=17, weight="bold", color=green)
    fig.text(0.11, 0.112, r"$C(q)$  Candidate handling", fontsize=14, color=ink)
    canvas.annotate(
        "",
        (0.624, 0.108),
        (0.413, 0.108),
        arrowprops={"arrowstyle": "->", "color": green, "lw": 1.4},
    )
    fig.text(0.515, 0.132, "Exact bin counts", fontsize=10.5, color=green, ha="center")
    fig.text(0.65, 0.112, r"$m$  Crossing-bin refinement", fontsize=14, color=green)
    fig.text(
        0.04,
        0.025,
        "Schematic finite-score example · population counts only",
        fontsize=10.5,
        color=muted,
    )
    _save(fig, "candidate_work")


def _algorithm() -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set(xlim=(0, 14), ylim=(0, 7))
    ax.axis("off")
    headings = [
        (0.2, "1  SELF-SAMPLE", "Place the bracket near the current tail"),
        (4.9, "2  MULTI-THRESHOLD", "Get many exact counts from one classification"),
        (9.65, "3  REFINE", "Finish only the uncertain boundary"),
    ]
    for x, heading, subtitle in headings:
        ax.text(x, 6.5, heading, fontsize=14, weight="bold", color="#447a00")
        ax.text(x, 6.04, subtitle, fontsize=9.3, color="#475569")
    ax.text(0.2, 5.5, "Current row: sparse, regularly spaced vector loads", fontsize=9)
    for i in range(32):
        sampled = i % 8 < 2
        ax.add_patch(
            Rectangle(
                (0.2 + i * 0.126, 4.85), 0.112, 0.36, facecolor="#76b900" if sampled else "#dfe5eb"
            )
        )
    ax.text(0.2, 4.46, "Sample histogram ≈ current score distribution", fontsize=9)
    sample = [1, 3, 5, 8, 11, 14, 12, 9, 6, 3, 2, 1]
    for i, h in enumerate(sample):
        ax.add_patch(Rectangle((0.25 + i * 0.32, 2.7), 0.28, h * 0.085, facecolor="#a7c976"))
    for x, label, y in [(2.37, "T_floor", 2.16), (2.69, "T", 2.45), (3.33, "T_K", 2.16)]:
        ax.plot([x, x], [2.66, 3.9], color="#7557a6", linestyle="--", linewidth=1)
        ax.text(x, y, label, ha="center", fontsize=10, color="#7557a6")
    ax.text(
        0.2,
        1.45,
        "Ranks ≈ 2AS/N, AS/N, KS/N\n→ safety floor, admission threshold, upper anchor",
        fontsize=9.3,
        linespacing=1.5,
    )
    ax.annotate(
        "", (4.7, 3.6), (4.2, 3.6), arrowprops={"arrowstyle": "->", "lw": 2, "color": "#64748b"}
    )
    _box(ax, (4.95, 4.8), (4.05, 0.65), "Every valid score is examined", "#e9eff5", 11)
    ax.text(4.9, 4.25, "Exact histogram → suffix counts at all bin boundaries", fontsize=9)
    counts = [1000, 700, 400, 250, 150, 100, 80, 73, 380, 330, 270]
    for i, count in enumerate(counts):
        color = "#f5b642" if i == 7 else ("#579600" if i > 7 else "#cbd5e1")
        ax.add_patch(Rectangle((5.0 + i * 0.36, 2.7), 0.31, 0.18 + count / 800, facecolor=color))
    ax.text(5.0, 2.35, "T", fontsize=10)
    ax.text(8.9, 2.35, "H", fontsize=10)
    ax.annotate(
        "73 in crossing bin",
        (7.7, 3.02),
        (6.0, 3.65),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#8c661c"},
        color="#8c661c",
    )
    ax.text(8.43, 3.82, "980\nabove", fontsize=10, ha="center", color="#447a00")
    ax.text(
        4.9,
        1.45,
        "256 verification bins (shown schematically)\nOne bin assignment per survivor, then an on-chip scan",
        fontsize=9.3,
        linespacing=1.5,
    )
    ax.annotate(
        "", (9.5, 3.6), (9.05, 3.6), arrowprops={"arrowstyle": "->", "lw": 2, "color": "#64748b"}
    )
    _box(ax, (9.75, 4.45), (3.75, 0.85), "980 certain winners", fontsize=13)
    _box(ax, (9.75, 3.0), (3.75, 0.85), "Select 44 of 73 boundary candidates", "#fff3d9", 10.5)
    _box(ax, (9.75, 1.55), (3.75, 0.85), "1,024 exact output indices", fontsize=12)
    for y in (4.05, 2.6):
        ax.annotate(
            "",
            (11.6, y - 0.1),
            (11.6, y + 0.25),
            arrowprops={"arrowstyle": "->", "color": "#64748b"},
        )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.27, top=0.98)
    guard = fig.add_axes((0.025, 0.03, 0.95, 0.26))
    guard.set(xlim=(0, 14), ylim=(0, 3))
    guard.axis("off")
    guard.text(
        0.05, 2.8, "EXACTNESS CHECKS BEFORE OUTPUT", fontsize=12, weight="bold", color="#334155"
    )
    _box(
        guard,
        (3.2, 1.96),
        (7.6, 0.44),
        "Full-row coverage · valid bracket · complete candidates",
        "#f1f4f7",
        11,
    )
    paths = [
        (0.15, "Enough survivors\nRefine the crossing → exact Top-K", "#edf5df"),
        (4.85, "Too few survivors\nLower admission and verify again", "#fff3d9"),
        (9.55, "Overflow or unusable bracket\nExact complete-set / whole-row recovery", "#e9eff5"),
    ]
    for x, label, color in paths:
        _box(guard, (x, 0.24), (4.15, 0.98), label, color, 10)
        guard.annotate(
            "",
            (x + 2.075, 1.34),
            (7, 1.86),
            arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.4},
        )
    _save(fig, "algorithm")


def _gpu_sampling() -> None:
    """Show sample-window ownership and the on-chip calibration pipeline."""
    fig, ax = plt.subplots(figsize=(16.7, 7.0))
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.03, top=0.97)
    ax.set(xlim=(0, 16), ylim=(0, 7))
    ax.axis("off")
    ink, muted = "#17202b", "#52616f"
    blue, purple = "#386781", "#7557a6"
    ax.text(
        0.1, 6.68, "Spread short sample windows across the current row", fontsize=21, weight="bold"
    )
    windows = (0.7, 5.5, 10.3)
    for j, x in enumerate(windows):
        ax.add_patch(Rectangle((x, 5.65), 2.4, 0.4, facecolor="#e3efcd", edgecolor="#a7c976"))
        ax.text(
            x + 1.2, 5.85, f"Window {j}", ha="center", va="center", fontsize=15, color="#447a00"
        )
        if j < 2:
            ax.plot([x + 2.55, x + 4.65], [5.85, 5.85], color="#b6c0c9", linestyle=(0, (2, 3)))
    ax.text(13.05, 5.83, "…", fontsize=22, color=muted)
    ax.text(
        0.7,
        5.2,
        "Window starts use a regular stride; no previous-index lookup or random-number generation.",
        fontsize=15,
        color=muted,
    )

    panels = [
        (0.7, "MAIN · 8 scores / 32 bytes", 8, 0.56),
        (8.3, "CLUS · 16 scores / 64 bytes", 16, 0.38),
    ]
    for x, heading, scores, width in panels:
        ax.text(x, 4.57, heading, fontsize=19, weight="bold", color=ink)
        for i in range(scores):
            fill = "#dcebf3" if i < 8 else "#e9dff4"
            ax.add_patch(
                Rectangle(
                    (x + i * width, 3.38),
                    width,
                    0.56,
                    facecolor=fill,
                    edgecolor="white",
                    linewidth=1,
                )
            )
        for vector in range(scores // 4):
            left = x + vector * 4 * width
            ax.add_patch(
                Rectangle(
                    (left, 3.38), 4 * width, 0.56, fill=False, edgecolor="#8e9ba7", linewidth=1
                )
            )
            ax.text(left + 2 * width, 4.1, "float4 · 16 B", ha="center", fontsize=13, color=muted)
        for worker in range(scores // 8):
            start = x + worker * 8 * width
            color = blue if worker == 0 else purple
            ax.plot(
                [start, start, start + 8 * width, start + 8 * width],
                [3.21, 3.08, 3.08, 3.21],
                color=color,
                linewidth=1.2,
            )
            label = "Work item j" if worker == 0 else "Work item j + P"
            ax.text(start + 4 * width, 2.78, label, fontsize=15, ha="center", color=color)
        ax.text(
            x,
            2.3,
            "Two vector loads per work item; eight retained sample scores.",
            fontsize=14,
            color=muted,
        )

    stages = [
        (0.2, "1  REGISTER VALUES", "Warp min/max reductions"),
        (5.6, "2  SHARED HISTOGRAM", "256 bins · atomic increments"),
        (11.0, "3  WARP-0 SCAN", "Rank crossings → anchors"),
    ]
    for x, heading, description in stages:
        _box(ax, (x, 0.63), (4.75, 1.05), heading + "\n" + description, "#f1f5f8", 15)
        if x < 11:
            ax.annotate(
                "",
                (x + 5.25, 1.15),
                (x + 4.85, 1.15),
                arrowprops={"arrowstyle": "->", "color": muted, "lw": 1.5},
            )
    ax.text(
        0.2,
        0.11,
        "P = number of sample windows. CTA threads process work items in strides; a window is not a CUDA block.",
        fontsize=13,
        color=muted,
    )
    _save(fig, "gpu_sampling")


def _integration() -> None:
    fig, ax = plt.subplots(figsize=(14, 8.4))
    ax.set(xlim=(0, 14), ylim=(0, 9))
    ax.axis("off")
    ax.text(
        0.3,
        8.65,
        "One selection core, two row interfaces",
        fontsize=21,
        weight="bold",
        color="#17202b",
    )
    _box(
        ax,
        (0.75, 7.35),
        (12.5, 0.82),
        "Sparse-attention indexer → TopK dispatcher\nOne self-sampling configuration for both phases",
        "#e9eff5",
        12,
    )
    adapters = [
        (
            0.75,
            "DECODE · run_varlen\nKV lengths + MTP offset + compression → valid prefix\n"
            "Routing: streaming, register, or cluster families",
            "#edf5df",
        ),
        (
            7.45,
            "PREFILL · run_prefill\nCompressed-column window [start, end)\n"
            "Local indices · one thread block per row",
            "#e9eff5",
        ),
    ]
    for x, label, color in adapters:
        _box(ax, (x, 5.1), (5.8, 1.38), label, color, 11.5)
        ax.annotate(
            "",
            (x + 2.9, 6.6),
            (7, 7.23),
            arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.5},
        )
        ax.annotate(
            "",
            (x + 2.9, 4.34),
            (x + 2.9, 4.98),
            arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.5},
        )
    ax.text(
        3.65,
        4.62,
        "streaming route",
        ha="center",
        fontsize=9,
        color="#447a00",
        backgroundcolor="white",
    )
    ax.text(
        10.35,
        4.62,
        "compile-time window mode",
        ha="center",
        fontsize=9,
        color="#52616f",
        backgroundcolor="white",
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.75, 2.03),
            12.5,
            2.15,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor="#f5f9ee",
            edgecolor="#99bb6c",
            linewidth=1.3,
        )
    )
    ax.text(
        7,
        3.73,
        "Shared streaming implementation · GvrMainKernel",
        ha="center",
        fontsize=14,
        weight="bold",
        color="#447a00",
    )
    stages = ["Self-sample", "Multi-threshold\nexact counts", "Collect + refine", "Exact indices"]
    for i, label in enumerate(stages):
        x = 1.03 + i * 3.08
        _box(ax, (x, 2.49), (2.55, 0.7), label, "#ffffff", 11)
        if i < 3:
            ax.annotate(
                "",
                (x + 2.95, 2.84),
                (x + 2.68, 2.84),
                arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.4},
            )
    ax.text(
        7,
        2.14,
        "Prefill specializes addressing, masks, and index origin; selection logic is shared.",
        ha="center",
        fontsize=10,
        color="#52616f",
    )
    _box(
        ax,
        (0.75, 0.72),
        (12.5, 0.66),
        "Caller-owned INT32 output · no temporal-prior seed, handoff, or write-back",
        "#edf5df",
        12,
    )
    ax.annotate(
        "", (7, 1.5), (7, 1.91), arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.5}
    )
    ax.text(
        0.75,
        0.1,
        "Runtime support: layout gates · precompiled launchers · exact native fallback",
        fontsize=10.5,
        color="#52616f",
    )
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.99)
    _save(fig, "integration")


def _matching(rows: list[dict], model: str) -> list[dict]:
    required = COMPARISON_ARMS
    return [
        r for r in rows if r["model"] == model and all(r[a + "_us"] is not None for a in required)
    ]


def _line_data(rows: list[dict], arm: str, batch: int) -> tuple[list[float], list[float]]:
    selected = [r for r in rows if r["batch"] == batch]
    buckets = sorted({r["isl_bucket"] for r in selected}, key=lambda x: int(x[:-1]))
    groups = [[r for r in selected if r["isl_bucket"] == bucket] for bucket in buckets]
    return (
        [median(r["n"] for r in group) for group in groups],
        [mean(r[arm + "_us"] for r in group) for group in groups],
    )


def _legend(fig: plt.Figure) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[a],
            linewidth=2.5,
            linestyle="--" if a in CROSS_CAMPAIGN else "-",
            label=LABELS[a],
        )
        for a in COMPARISON_ARMS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=10,
    )


def _latency(rows: list[dict]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for i, (model, title) in enumerate(MODELS.items()):
        matched = _matching(rows, model)
        for j, batch in enumerate((1, 1024)):
            ax = axes[i, j]
            for arm in COMPARISON_ARMS:
                x, y = _line_data(matched, arm, batch)
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=3,
                    color=COLORS[arm],
                    linewidth=2.3 if arm == "gvr_v2" else 1.6,
                    linestyle="--" if arm in CROSS_CAMPAIGN else "-",
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(
                [2**n for n in ((10, 12, 14, 16, 18) if model != "v32" else (12, 14, 16, 18))]
            )
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1024:g}K"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.set_title(f"{title}  |  B={batch}", fontsize=12, loc="left")
            ax.set_ylabel("Mean kernel time (µs)")
            ax.set_xlabel("Valid indexer row length N")
            ax.grid(which="major", alpha=0.18)
    fig.suptitle("Latency across row lengths and batch sizes", fontsize=18, weight="bold", y=0.995)
    fig.subplots_adjust(hspace=0.52, wspace=0.2, bottom=0.14, top=0.94)
    _legend(fig)
    _save(fig, "latency")


def _roofline_reachable_rates(rows: list[dict]) -> dict:
    calibration = json.loads((ROOT / "provenance.json").read_text())["roofline_model"]
    by_model = {}
    for model in MODELS:
        matched = _matching(rows, model)
        k = matched[0]["k"]
        by_model[model] = {}
        for arm in COMPARISON_ARMS:
            widths, times = _line_data(matched, arm, 1024)
            rates = []
            for n, us in zip(widths, times):
                intensity = n / (4 * (n + k))
                roof = min(
                    calibration["measured_compare_t_s"],
                    calibration["measured_bandwidth_tb_s"] * intensity,
                )
                rates.append(100 * (1024 * n / (us * 1e6)) / roof)
            by_model[model][arm] = {
                "points": len(rates),
                "average_percent": mean(rates),
                "peak_percent": max(rates),
            }
    return {
        "batch": 1024,
        "reference": "Calibrated roof at each plotted intensity",
        "aggregation": "Arithmetic mean and maximum of plotted-point reachable rates",
        "by_model": by_model,
    }


def _roofline(rows: list[dict]) -> None:
    model_data = json.loads((ROOT / "provenance.json").read_text())["roofline_model"]
    bw = model_data["measured_bandwidth_tb_s"]
    compare = model_data["measured_compare_t_s"]
    fig = plt.figure(figsize=(14, 9.4), facecolor="white")
    top = fig.add_axes((0.075, 0.61, 0.875, 0.245))
    intensity = np.geomspace(0.05, 24, 400)
    top.axvspan(0.125, 0.25, color="#edf5df", zorder=0)
    top.plot(
        intensity,
        np.minimum(
            model_data["theoretical_bandwidth_tb_s"] * intensity,
            model_data["theoretical_compare_t_s"],
        ),
        color="#94a3b8",
        linestyle=(0, (5, 3)),
        linewidth=1.8,
    )
    top.plot(intensity, np.minimum(bw * intensity, compare), color="#273746", linewidth=2.4)
    top.plot([0.125, 0.25], [bw * 0.125, bw * 0.25], color=COLORS["gvr_v2"], linewidth=5)
    top.annotate(
        "Ideal Top-K band\n0.125–0.25 compare/byte",
        xy=(0.18, bw * 0.18),
        xytext=(0.055, 12),
        fontsize=10.5,
        color="#447a00",
        arrowprops={"arrowstyle": "->", "color": "#579600", "connectionstyle": "arc3,rad=.15"},
    )
    top.text(0.85, 2.0, "Bandwidth slope\n6.912 TB/s × intensity", fontsize=10.5, color="#334155")
    top.annotate(
        "Compare ceiling\n37.047 Tcompare/s",
        xy=(12, compare),
        xytext=(7, 4.5),
        fontsize=10.5,
        color="#334155",
        arrowprops={"arrowstyle": "->", "color": "#64748b"},
    )
    top.plot(compare / bw, compare, "o", color="#273746", markersize=5)
    top.text(5.0, 63, "Knee: 5.36", fontsize=10, ha="center", color="#52616f")
    top.set(xscale="log", yscale="log", xlim=(0.05, 24), ylim=(0.2, 100))
    top.set_xticks([0.125, 0.25, 1, 4, 16], ["0.125", "0.25", "1", "4", "16"])
    top.set_yticks([1, 10, 100], ["1", "10", "100"])
    top.minorticks_off()
    top.set_xlabel("Operational intensity  ·  compare/byte", fontsize=10.5, labelpad=8)
    top.set_ylabel("Tcompare/s", fontsize=10.5)
    top.grid(axis="y", color="#e9edf1", linewidth=0.7)
    top.tick_params(length=0, pad=7)
    for spine in ("left", "bottom"):
        top.spines[spine].set_color("#d5dce3")
    top.legend(
        handles=[
            Line2D([0], [0], color="#273746", lw=2.4, label="Measured calibration"),
            Line2D(
                [0],
                [0],
                color="#94a3b8",
                lw=1.8,
                linestyle=(0, (5, 3)),
                label="Theoretical reference",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.015, 1.21),
        ncol=2,
        fontsize=9.5,
        frameon=False,
    )
    for i, (model, title) in enumerate(MODELS.items()):
        ax = fig.add_axes((0.075 + i * 0.305, 0.205, 0.26, 0.23))
        matched = _matching(rows, model)
        k = matched[0]["k"]
        xroof = np.linspace(0.125, 0.25, 100)
        ax.fill_between(xroof, xroof * bw, 1.95, color="#f2f5f7", zorder=0)
        ax.plot(xroof, xroof * bw, color="#273746", linestyle=(0, (2, 2)), linewidth=1.5)
        for arm in [*TEMPORAL, *ARMS, "gvr_v2"]:
            widths, times = _line_data(matched, arm, 1024)
            x = [n / (4 * (n + k)) for n in widths]
            y = [1024 * n / (us * 1e6) for n, us in zip(widths, times)]
            ax.plot(
                x,
                y,
                color=COLORS[arm],
                linewidth=2.8 if arm == "gvr_v2" else 1.5,
                linestyle="--" if arm in CROSS_CAMPAIGN else "-",
                marker="o" if arm == "gvr_v2" else ".",
                markersize=4.5,
                alpha=1 if arm == "gvr_v2" else 0.8,
                zorder=5 if arm == "gvr_v2" else 3,
            )
        ax.set(xlim=(0.123, 0.253), ylim=(0, 1.95))
        ax.set_xticks([0.125, 0.175, 0.225, 0.25], [".125", ".175", ".225", ".250"])
        ax.set_yticks([0, 0.5, 1, 1.5], ["0", "0.5", "1.0", "1.5"])
        ax.set_xlabel("Intensity (compare/byte)", fontsize=10, labelpad=8)
        ax.set_title(title, fontsize=11, loc="left", pad=12, weight="bold")
        if i == 0:
            ax.set_ylabel("Work throughput (Tcompare/s)", fontsize=10)
            ax.text(0.133, 1.73, "Calibrated roof", fontsize=9, color="#52616f")
        ax.grid(axis="y", color="#e9edf1", linewidth=0.7)
        ax.tick_params(length=0, labelsize=9, pad=7)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#d5dce3")
    fig.text(
        0.035,
        0.96,
        "Top-K lives on the bandwidth slope",
        fontsize=21,
        weight="bold",
        color="#17202b",
    )
    fig.text(
        0.035,
        0.92,
        "A. The full B200 roofline  ·  Top-K intensity stays far below the compute knee",
        fontsize=11.5,
        color="#52616f",
    )
    fig.text(
        0.075,
        0.515,
        "B. Pareto curves across intensities",
        fontsize=13,
        weight="bold",
        color="#17202b",
    )
    fig.text(
        0.075,
        0.482,
        "B = 1024  ·  identical layers per model  ·  higher is faster",
        fontsize=10.5,
        color="#52616f",
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[a],
            linewidth=2.5,
            linestyle="--" if a in CROSS_CAMPAIGN else "-",
            label=LABELS[a],
        )
        for a in COMPARISON_ARMS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.53, 0.077),
        fontsize=10,
    )
    fig.text(
        0.075,
        0.046,
        "Shared ideal work: BN comparisons. Minimum traffic: 4B(N + K) bytes. "
        "Extra passes and output work remain in measured time.",
        fontsize=9,
        color="#52616f",
    )
    fig.text(
        0.075,
        0.018,
        "Line styles distinguish benchmark runs. Work throughput uses the same logical task for every kernel.",
        fontsize=9,
        color="#52616f",
    )
    _save(fig, "roofline")


def _speedup_map(rows: list[dict], arm: str, label: str, scope: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    batches = sorted({r["batch"] for r in rows})
    is_v1 = arm == "temporal_tiered"
    norm = Normalize(vmin=1, vmax=3 if is_v1 else 21)
    cmap = plt.get_cmap("YlGnBu")
    for ax, (model, title) in zip(axes, MODELS.items()):
        selected = [r for r in rows if r["model"] == model and r[arm + "_us"] is not None]
        buckets = sorted({r["isl_bucket"] for r in selected}, key=lambda s: int(s[:-1]))
        values = []
        widths = []
        for bucket in buckets:
            group = [r for r in selected if r["isl_bucket"] == bucket]
            widths.append(median(r["n"] for r in group) / 1024)
            values.append(
                [
                    geometric_mean(
                        r[arm + "_us"] / r["gvr_v2_us"] for r in group if r["batch"] == b
                    )
                    for b in batches
                ]
            )
        data = np.asarray(values)
        graphic = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
        for y in range(len(buckets)):
            for x in range(len(batches)):
                if is_v1 and any(
                    r[arm + "_us"] < r["gvr_v2_us"]
                    for r in selected
                    if r["isl_bucket"] == buckets[y] and r["batch"] == batches[x]
                ):
                    ax.add_patch(
                        Polygon(
                            [(x + 0.21, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y - 0.21)],
                            facecolor="#d46d24",
                            edgecolor="white",
                            linewidth=0.3,
                        )
                    )
                red, green, blue, _ = cmap(norm(data[y, x]))
                brightness = 0.299 * red + 0.587 * green + 0.114 * blue
                ax.text(
                    x,
                    y,
                    f"{data[y, x]:.2f}" if is_v1 else f"{data[y, x]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.9 if is_v1 else 7.1,
                    color="white" if brightness < 0.5 else "#17202b",
                )
        ax.set_xticks(range(len(batches)), batches, rotation=60, fontsize=9)
        ax.set_yticks(range(len(widths)), [f"{n:.0f}K" for n in widths], fontsize=9)
        ax.set_xlabel("Batch size B")
        ax.set_ylabel("Valid row length N (rounded)")
        ax.set_title(title, loc="left", fontsize=11, pad=12)
    fig.suptitle(
        f"GVR V2 vs {label}: gains across the full length–batch grid",
        fontsize=18,
        x=0.035,
        ha="left",
        weight="bold",
        y=1.02,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.87, bottom=0.34, wspace=0.27)
    cax = fig.add_axes((0.34, 0.055 if is_v1 else 0.09, 0.32, 0.026))
    ticks = [1, 1.5, 2, 2.5, 3] if is_v1 else [1, 5, 10, 15, 21]
    bar = fig.colorbar(graphic, cax=cax, orientation="horizontal", ticks=ticks)
    if is_v1:
        bar.ax.set_xticklabels(["1× · parity", "1.5×", "2×", "2.5×", "3×"])
    bar.set_label(f"{label} time / GVR V2 time · geometric mean across layers", fontsize=9)
    fig.text(
        0.06,
        0.19 if is_v1 else 0.17,
        f"{scope} · 1.0× is parity · shared color scale across all three models.",
        fontsize=9,
    )
    if is_v1:
        fig.add_artist(
            Polygon(
                [(0.06, 0.155), (0.07, 0.155), (0.07, 0.133)],
                transform=fig.transFigure,
                facecolor="#d46d24",
                edgecolor="none",
            )
        )
        fig.text(
            0.08,
            0.138,
            "Orange corner: at least one layer is slower in V2. Layer averages do not show every case.",
            fontsize=9,
            color="#52616f",
        )
    _save(fig, "gvr_v1_map" if is_v1 else arm + "_map")


def main() -> None:
    """Validate the frozen dataset, then regenerate statistics and ten figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "gvr-v2-blog",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    rows = _load()
    summary = {
        "copyright": COPYRIGHT,
        "reference": json.loads((ROOT / "provenance.json").read_text())["reference"],
        "overall": {a: _stats(rows, a) for a in ARMS},
        "comparison_common_cases": _comparison(rows),
        "by_model": {
            m: {a: _stats([r for r in rows if r["model"] == m], a) for a in ARMS} for m in MODELS
        },
        "temporal_vs_v2": {a: _stats(rows, a) for a in TEMPORAL},
        "roofline_reachable_rate": _roofline_reachable_rates(rows),
        "evolution_vs_radix": {
            a: {
                "geomean": geometric_mean(r["radix_cuda_us"] / r[a + "_us"] for r in rows),
                "wins_percent": 100 * mean(r["radix_cuda_us"] > r[a + "_us"] for r in rows),
            }
            for a in [*TEMPORAL, "gvr_v2"]
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, result in summary["overall"].items():
        print(
            name, f"{result['geomean']:.6f}×", result["cases"], f"wins {result['win_percent']:.3f}%"
        )
    _overview(rows)
    _evolution(rows)
    _candidate_work()
    _algorithm()
    _gpu_sampling()
    _speedup_map(rows, "radix_cuda", "radix CUDA", "TensorRT-LLM production dispatcher")
    _speedup_map(rows, "temporal_tiered", "GVR V1", "GVR V1 (temporal hint)")
    _latency(rows)
    _roofline(rows)
    _integration()


if __name__ == "__main__":
    main()
