# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Agent concurrency Pareto PNG for trace_replay_pareto_frontier.v4 JSON.

Loaded dynamically by examples/scaffolding/trace_replay/pareto/trace_replay_pareto_aggregate.py
(via importlib on a resolved sibling path), so this module is intentionally
standalone — no relative imports, no package init required.

Plots concurrent agent sessions (``concurrency``) vs per-user generation speed
(``median_tps_per_user``) and highlights the Pareto frontier: the points where
you can't increase concurrency without sacrificing per-agent throughput (or
vice versa). Aggregate system throughput is annotated next to each marker so
the plot also conveys total tokens/s served at each load point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PathLike = Union[str, Path]


def _pareto_frontier_upper_right(
    points: Sequence[Tuple[float, float]],
) -> List[int]:
    n = len(points)
    order = sorted(range(n), key=lambda i: (-points[i][0], -points[i][1]))
    frontier: List[int] = []
    max_y = float("-inf")
    for i in order:
        y = points[i][1]
        if y > max_y:
            frontier.append(i)
            max_y = y
    frontier.sort(key=lambda i: points[i][0])
    return frontier


def _successful_runs(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in record.get("runs", []) if r.get("status") == "success"]


def _title_suffix(record: Dict[str, Any]) -> str:
    naming = record.get("artifact_naming", {}) or {}
    model = naming.get("model_name") or "model"
    tp = naming.get("tensor_parallel_size")
    ep = naming.get("moe_expert_parallel_size")
    tp_ep = ""
    if tp is not None or ep is not None:
        tp_ep = f" (TP{tp}/EP{ep})"
    return f"{model}{tp_ep}"


def _resolve_trace_label(record: Dict[str, Any]) -> str:
    """Human-readable trace name(s) for the figure caption.

    Prefers ``trace_meta.trace_dir`` (single trace) or ``trace_meta.traces[]``
    (mix) basename; falls back to ``aggregator_args.trace_dir`` or
    ``trace_meta.trace_id`` if neither is set.
    """
    tm = record.get("trace_meta") or {}
    traces = tm.get("traces")
    if isinstance(traces, list) and traces:
        parts = []
        for t in traces:
            td = t.get("trace_dir") or ""
            parts.append(Path(td).name if td else (t.get("trace_id") or "?"))
        return " + ".join(parts)
    td = tm.get("trace_dir")
    if td:
        return Path(td).name
    aa = record.get("aggregator_args") or {}
    dirs = aa.get("trace_dir") or []
    if dirs:
        return " + ".join(Path(d).name for d in dirs)
    return tm.get("trace_id") or "(unknown trace)"


def _format_point_label(r: Dict[str, Any]) -> str:
    """Pareto point label using the abbreviations documented in
    :data:`POINT_LABEL_LEGEND` (rendered as the figure caption).
    """
    b = r.get("max_batch_size")
    n = r.get("total_sessions")
    c = r.get("concurrency")
    real_max = r.get("real_cache_hit_max")
    real_avg = r.get("real_cache_hit_avg")
    opt = r.get("optimal_cache_hit")
    rmx = f"{real_max:.3f}" if isinstance(real_max, (int, float)) else "—"
    rav = f"{real_avg:.3f}" if isinstance(real_avg, (int, float)) else "—"
    ops = f"{opt:.3f}" if isinstance(opt, (int, float)) else "—"
    return f"B={b} N={n} C={c}\nR_max={rmx}  R_avg={rav}  O={ops}"


POINT_LABEL_LEGEND = (
    "B = max_batch_size    N = total_sessions    C = concurrency\n"
    "R_max / R_avg = real_cache_hit per session (max / mean across N)    "
    "O = optimal_cache_hit (offline upper bound)    "
    "A = output_tps_aggregate")


def write_agent_pareto_png_from_json_file(
    output_json: PathLike,
    *,
    figure_caption: Optional[str] = None,
    png_path: Optional[PathLike] = None,
) -> Optional[Path]:
    output_json = Path(output_json).expanduser().resolve()
    with output_json.open("r", encoding="utf-8") as f:
        record = json.load(f)

    runs = _successful_runs(record)
    if not runs:
        print(f"WARNING: no successful runs in {output_json}; skipping agent Pareto PNG.")
        return None

    xs = [float(r["concurrency"]) for r in runs]
    ys = [float(r["median_tps_per_user"]) for r in runs]
    aggs = [float(r.get("output_tps_aggregate", 0.0)) for r in runs]
    labels = [
        f"{_format_point_label(r)}  A={agg:.1f}"
        for r, agg in zip(runs, aggs)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, color="tab:green", s=64, zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, annotation_clip=False)

    # Normally concurrency vs per-agent throughput has a real tradeoff and
    # every ladder point lands on the upper-right frontier. If the strict
    # frontier collapses to one point (degenerate sweep), fall back to the
    # x-sorted trajectory so the plot is still informative.
    frontier = _pareto_frontier_upper_right(list(zip(xs, ys)))
    if len(frontier) >= 2:
        line_idx = frontier
        line_label = "Agent Pareto Frontier"
    else:
        line_idx = sorted(range(len(xs)), key=lambda i: xs[i])
        line_label = "Ladder trajectory"
    if len(line_idx) >= 2:
        ax.plot(
            [xs[i] for i in line_idx],
            [ys[i] for i in line_idx],
            color="tab:red",
            linewidth=2,
            marker="o",
            label=line_label,
            zorder=2,
        )
        ax.legend(loc="best")

    ax.set_title(f"Agent Concurrency Pareto — {_title_suffix(record)}")
    ax.set_xlabel("Concurrent agent sessions (concurrency)")
    ax.set_ylabel("Per-agent throughput: median_tps_per_user (tokens/s)")
    ax.grid(True, alpha=0.3)

    # Caption directly below the plot: per-point label legend + the trace
    # name in one text block so the caption area stays compact.
    fig.subplots_adjust(bottom=0.18)
    caption = (
        f"{POINT_LABEL_LEGEND}\n"
        f"Trace: {_resolve_trace_label(record)}")
    if figure_caption:
        caption = f"{caption}\n{figure_caption}"
    fig.text(0.5, 0.04, caption, ha="center", fontsize=8,
             style="italic", wrap=True)

    if png_path is None:
        png_path = output_json.with_name(output_json.stem + "_agent_pareto.png")
    else:
        png_path = Path(png_path)

    # fig.tight_layout() removed: would override explicit subplots_adjust
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path
