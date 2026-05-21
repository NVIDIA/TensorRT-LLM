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

r"""Job-throughput (agent-sessions/h) Pareto PNG for v4 JSON.

Loaded dynamically by examples/scaffolding/trace_replay/pareto/trace_replay_pareto_aggregate.py
(via importlib on a resolved sibling path), so this module is intentionally
standalone — no relative imports, no package init required.

"Job" here means an agent session (one trace replay), not a token. Axes
mirror the InferenceMAX convention with sessions instead of tokens — y is
aggregate per-GPU rate, x is the per-user observed completion rate (1/SD,
the session analog of 1/TPOT):

    x = 3600 / SD                        # sessions/h/user (single-task speed)
    y = 3600 * N / (WC * TP)             # sessions/h/gpu (aggregate)

SD ≠ WC × constant across ladder steps, so y/x is non-constant and the plot
shows a real Pareto curve.
"""

from __future__ import annotations

import json
from collections import defaultdict
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


def _per_session_block_hit_rates(detail):
    """Engine-measured per-session block hit rate, derived on demand."""
    if not detail:
        return []
    per_session: Dict[Tuple[int, int], List[int]] = defaultdict(lambda: [0, 0])
    for entry in detail:
        si = entry.get("session_index")
        if si is None:
            continue
        ti = int(entry.get("trace_index") or 0)
        per_session[(ti, int(si))][0] += int(entry.get("num_reused_blocks") or 0)
        per_session[(ti, int(si))][1] += int(entry.get("num_missed_blocks") or 0)
    out = []
    for (_ti, _si), (reused, missed) in per_session.items():
        total = reused + missed
        if total > 0:
            out.append(reused / total)
    return out


def _format_point_label(r: Dict[str, Any]) -> str:
    """Pareto point label using the abbreviations documented in
    :data:`POINT_LABEL_LEGEND` (rendered as the figure caption).
    """
    b = r.get("max_batch_size")
    n = r.get("total_sessions")
    c = r.get("concurrency")
    rates = _per_session_block_hit_rates(r.get("replay_assistant_generations_detail"))
    real_max = max(rates) if rates else None
    real_avg = (sum(rates) / len(rates)) if rates else None
    opt = r.get("optimal_cache_hit")
    rmx = f"{real_max:.3f}" if isinstance(real_max, (int, float)) else "—"
    rav = f"{real_avg:.3f}" if isinstance(real_avg, (int, float)) else "—"
    ops = f"{opt:.3f}" if isinstance(opt, (int, float)) else "—"
    return f"B={b} N={n} C={c}\nR_max={rmx}  R_avg={rav}  O={ops}"


POINT_LABEL_LEGEND = (
    "B = max_batch_size    N = total_sessions    C = concurrency\n"
    "R_max / R_avg = engine-measured block hit rate per session (max / mean across N)    "
    "O = optimal_cache_hit (offline upper bound)")


def _tp_size(record: Dict[str, Any]) -> Optional[int]:
    naming = record.get("artifact_naming", {}) or {}
    tp = naming.get("tensor_parallel_size")
    if tp:
        return int(tp)
    cli = record.get("cli_args", {}) or {}
    tp = cli.get("tensor_parallel_size")
    return int(tp) if tp else None


def write_job_pareto_png_from_json_file(
    output_json: PathLike,
    *,
    curve_label: str = "Pareto Frontier",
    figure_caption: Optional[str] = None,
    png_path: Optional[PathLike] = None,
) -> Optional[Path]:
    output_json = Path(output_json).expanduser().resolve()
    with output_json.open("r", encoding="utf-8") as f:
        record = json.load(f)

    tp = _tp_size(record)
    if not tp:
        print(
            f"WARNING: tensor_parallel_size missing from {output_json}; "
            "skipping job-throughput Pareto PNG."
        )
        return None

    usable = []
    for r in _successful_runs(record):
        n = r.get("total_sessions")
        wc = r.get("wall_clock_s")
        sd = r.get("session_duration_mean_s")
        if not n or not wc or not sd:
            continue
        x = 3600.0 / float(sd)
        y = 3600.0 * float(n) / (float(wc) * tp)
        usable.append((r, x, y))
    if not usable:
        print(
            f"WARNING: no runs in {output_json} have "
            "total_sessions/wall_clock_s/session_duration_mean_s; "
            "skipping job-throughput Pareto PNG."
        )
        return None

    xs = [x for _, x, _ in usable]
    ys = [y for _, _, y in usable]
    labels = [_format_point_label(r) for r, _, _ in usable]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, color="tab:purple", s=64, zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, annotation_clip=False)

    frontier = _pareto_frontier_upper_right(list(zip(xs, ys)))
    if len(frontier) >= 2:
        line_idx = frontier
        line_label = curve_label
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

    ax.set_title(f"Job-level Throughput Pareto — {_title_suffix(record)}")
    ax.set_xlabel("sessions/h/user = 3600 / SD")
    ax.set_ylabel("sessions/h/gpu = 3600 * N / (WC * TP)")
    ax.grid(True, alpha=0.3)
    # axes auto-scale: don't force zero so small ladders fill the panel

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
        png_path = output_json.with_name(output_json.stem + "_job_pareto.png")
    else:
        png_path = Path(png_path)

    # fig.tight_layout() removed: would override explicit subplots_adjust
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path
