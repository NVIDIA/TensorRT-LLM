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

r"""Per-session KV-cache block hit rate vs session start time.

Loaded dynamically by examples/scaffolding/trace_replay/pareto/trace_replay_pareto_aggregate.py.
Designed for "single ladder step, very large N, big arrival_jitter" runs
where session start times spread across many minutes and the user wants
to see the temporal evolution of cache hit rate (early-arrivers cold,
late-arrivers warm, or eviction kicking in mid-run, etc.).

Axes:
    x = session start time (seconds, relative to the first session's arrival)
    y = per-session block hit rate

Each session is one scatter point at (start_time, hit_rate); points are
also connected in time order with a thin line so the temporal trajectory
is visible. The offline upper bound is drawn as a horizontal dashed line.

For multi-step ladder runs the plotter emits one PNG per step
(``*_session_hit_vs_time_B<B>.png``) so each step is self-contained.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PathLike = Union[str, Path]


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


def _session_timeline(detail: List[Dict[str, Any]]) -> List[Tuple[float, float, float, Tuple[int, int]]]:
    """Return [(start_time, end_time, hit_rate, (trace_index, session_index)), ...].

    start_time/end_time are ``time.monotonic()`` seconds (comparable only
    within the run that produced this detail list). hit_rate is the
    session's block-weighted block hit rate.
    """
    by_session: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for e in detail or []:
        si = e.get("session_index")
        if si is None:
            continue
        ti = int(e.get("trace_index") or 0)
        by_session[(ti, int(si))].append(e)

    out: List[Tuple[float, float, float, Tuple[int, int]]] = []
    for sid, calls in by_session.items():
        if not calls:
            continue
        starts = [c.get("arrival_time") or c.get("server_arrival_time")
                  for c in calls]
        starts = [s for s in starts if isinstance(s, (int, float))]
        ends = [c.get("last_token_time") for c in calls]
        ends = [s for s in ends if isinstance(s, (int, float))]
        if not starts or not ends:
            continue
        start = float(min(starts))
        end = float(max(ends))
        reused = sum(int(c.get("num_reused_blocks") or 0) for c in calls)
        missed = sum(int(c.get("num_missed_blocks") or 0) for c in calls)
        total = reused + missed
        if total <= 0:
            continue
        out.append((start, end, reused / total, sid))
    out.sort(key=lambda x: x[0])
    return out


def _render_one(
    *,
    record: Dict[str, Any],
    run: Dict[str, Any],
    output_json: Path,
    figure_caption: Optional[str],
    png_path: Optional[Path],
) -> Optional[Path]:
    detail = run.get("replay_assistant_generations_detail")
    if not detail:
        return None
    timeline = _session_timeline(detail)
    if not timeline:
        return None

    t0 = timeline[0][0]
    xs = [s[0] - t0 for s in timeline]
    ys = [s[2] for s in timeline]
    end_xs = [s[1] - t0 for s in timeline]

    fig, ax = plt.subplots(figsize=(11, 6))

    # Per-session lifetime as a thin horizontal segment (start -> end).
    for x_s, x_e, y in zip(xs, end_xs, ys):
        ax.hlines(y, x_s, x_e, colors="tab:blue", linewidth=0.8, alpha=0.25,
                  zorder=1)

    # Scatter at the start time + connect in time order.
    ax.plot(xs, ys, color="tab:blue", linewidth=0.6, alpha=0.5, zorder=2,
            label="time-ordered connection")
    ax.scatter(xs, ys, color="tab:blue", s=18, zorder=3,
               label="session start (x) vs block hit rate (y)")

    opt = run.get("optimal_cache_hit")
    if isinstance(opt, (int, float)):
        ax.axhline(opt, color="tab:red", linestyle="--", linewidth=1.5,
                   label=f"optimal_cache_hit = {opt:.3f}", zorder=4)

    b = run.get("max_batch_size")
    c = run.get("concurrency")
    n = run.get("total_sessions")
    ax.set_xlabel(
        "session start time (s, relative to first session's arrival)")
    ax.set_ylabel("session block hit rate "
                  "(sum num_reused / (sum num_reused + sum num_missed))")
    ax.set_title(
        f"Session KV hit rate vs start time — {_title_suffix(record)}\n"
        f"B={b}  N={n}  C={c}  (one point per session, "
        f"line = lifetime, scatter = start time)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.subplots_adjust(bottom=0.18)
    legend = (
        "Each point = one session. x = session arrival time "
        "(time.monotonic, rebased to 0). "
        "y = block-weighted hit rate over that session's calls.\n"
        "Thin blue segment = session lifetime (start → end). "
        "Dashed red = optimal_cache_hit (offline upper bound).")
    caption = f"{legend}\nTrace: {_resolve_trace_label(record)}"
    if figure_caption:
        caption = f"{caption}\n{figure_caption}"
    fig.text(0.5, 0.04, caption, ha="center", fontsize=8, style="italic",
             wrap=True)

    if png_path is None:
        suffix = f"_session_hit_vs_time_B{b}.png" if b is not None else "_session_hit_vs_time.png"
        png_path = output_json.with_name(output_json.stem + suffix)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def write_session_hit_vs_time_png_from_json_file(
    output_json: PathLike,
    *,
    figure_caption: Optional[str] = None,
    png_path: Optional[PathLike] = None,
) -> Optional[Path]:
    """Emit one ``*_session_hit_vs_time_B<B>.png`` per successful run."""
    output_json = Path(output_json).expanduser().resolve()
    with output_json.open("r", encoding="utf-8") as f:
        record = json.load(f)

    runs = _successful_runs(record)
    if not runs:
        print(f"WARNING: no successful runs in {output_json}; "
              "skipping session-hit-vs-time PNG.")
        return None

    last_png: Optional[Path] = None
    explicit = Path(png_path) if png_path is not None else None
    for run in runs:
        target = explicit if (explicit and len(runs) == 1) else None
        png = _render_one(
            record=record,
            run=run,
            output_json=output_json,
            figure_caption=figure_caption,
            png_path=target,
        )
        if png is not None:
            last_png = png
    return last_png
