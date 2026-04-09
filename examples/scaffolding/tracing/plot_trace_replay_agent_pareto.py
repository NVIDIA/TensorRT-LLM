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

r"""Plot agent session rate vs per-GPU wall throughput from trace replay JSON (offline).

Reads the same schema written by ``run_trace_replay_pareto_frontier.py`` (and
compatible v2/v3 reports). For each successful run:

* **x** (axis label ``task/user/hours``) = ``3600 / session_duration_mean_s`` —
  tasks per user-hour implied by mean session latency (see
  :func:`_mean_trace_duration_s`).
* **y** (axis label ``task/gpu/hours``) = ``n_sessions * 3600 / (gpu_count *
  wall_clock_s)`` — tasks per GPU-hour over wall time; ``gpu_count`` matches
  ``output_tps_per_gpu`` (``tensor_parallel_size``, then ``cli_args``, then
  ``host.cuda_device_count``; see :func:`_gpu_count_for_run`).

Points are annotated with ``ladder_step``.

Example::

    python examples/scaffolding/tracing/plot_trace_replay_agent_pareto.py \\
        examples/scaffolding/tracing/traces/sympy__sympy-21847/sympy__sympy-21847_20260414_020107.json

Writes ``<stem>_agent_pareto.png`` next to each JSON unless ``--output`` is set
(single input only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

Point3 = Tuple[float, float, Union[int, str]]


def _step_from_run(r: Dict[str, Any]) -> Union[int, str]:
    step = r.get("ladder_step")
    if step is None:
        step = r.get("max_batch_size")
    return step if step is not None else "?"


def _gpu_count_for_run(r: Dict[str, Any], data: Dict[str, Any]) -> Optional[int]:
    """GPU count aligned with ``output_tps_per_gpu`` (aggregate / TP size)."""
    cfg = r.get("llm_effective_config") or {}
    tp = cfg.get("tensor_parallel_size")
    if tp is not None:
        try:
            n = int(tp)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    cli = data.get("cli_args") or {}
    tp = cli.get("tensor_parallel_size")
    if tp is not None:
        try:
            n = int(tp)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    host = data.get("host") or {}
    n = host.get("cuda_device_count")
    if n is not None:
        try:
            c = int(n)
            if c > 0:
                return c
        except (TypeError, ValueError):
            pass
    return None


def _mean_trace_duration_s(r: Dict[str, Any]) -> Optional[float]:
    """Mean wall time per trace (seconds), from ``session_duration_mean_s`` or sum/n."""
    m = r.get("session_duration_mean_s")
    if m is not None:
        try:
            mf = float(m)
            if mf >= 0:
                return mf
        except (TypeError, ValueError):
            pass
    sm = r.get("session_duration_sum_s")
    if sm is None:
        sm = r.get("aggregate_latency_person_s")
    n = r.get("n_sessions")
    if sm is None or n is None:
        return None
    try:
        sf = float(sm)
        nf = float(n)
    except (TypeError, ValueError):
        return None
    if nf <= 0:
        return None
    return sf / nf


def collect_agent_pareto_points(
    runs: List[Dict[str, Any]],
    data: Dict[str, Any],
) -> List[Point3]:
    """Return ``(task/user/h, task/gpu/h, ladder_step)``."""
    pts: List[Point3] = []
    for r in runs:
        if r.get("status") != "success":
            continue
        mean_s = _mean_trace_duration_s(r)
        if mean_s is None or mean_s <= 0:
            continue
        x = 3600.0 / mean_s
        n_raw = r.get("n_sessions")
        wall = r.get("wall_clock_s")
        g = _gpu_count_for_run(r, data)
        if n_raw is None or wall is None or g is None or g <= 0:
            continue
        try:
            n_sessions = float(n_raw)
            wall_s = float(wall)
        except (TypeError, ValueError):
            continue
        if wall_s <= 0:
            continue
        y = n_sessions * 3600.0 / (float(g) * wall_s)
        pts.append((x, y, _step_from_run(r)))
    pts.sort(key=lambda t: t[0])
    return pts


def write_agent_pareto_png_from_json_file(
    json_path: Path,
    *,
    figure_caption: Optional[str] = None,
    png_path: Optional[Path] = None,
) -> Optional[Path]:
    """Load a trace replay JSON report and save an agent Pareto plot as PNG."""
    json_path = json_path.expanduser().resolve()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs: List[Dict[str, Any]] = data.get("runs") or []
    pts = collect_agent_pareto_points(runs, data)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(
            f"WARNING: matplotlib not available ({exc!r}); skip agent Pareto PNG.",
            file=sys.stderr,
        )
        return None

    if png_path is not None:
        out = png_path.expanduser().resolve()
    else:
        out = json_path.with_name(f"{json_path.stem}_agent_pareto.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    if not pts:
        ax.text(
            0.5,
            0.5,
            "No successful runs with mean latency, n_sessions, wall_clock_s, GPU count",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("task/user/hours vs task/gpu/hours")
    else:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs,
            ys,
            marker="o",
            color="tab:green",
            linewidth=1.5,
            label="Pareto ladder",
        )
        for xf, yf, step in pts:
            ax.annotate(
                str(step),
                xy=(xf, yf),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color="tab:green",
                ha="left",
                va="bottom",
            )
        ax.set_xlabel("tasks/user/hours")
        ax.set_ylabel("tasks/gpu/hours")
        ax.set_title("tasks/user/hours vs tasks/gpu/hours")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    if figure_caption:
        fig.text(0.5, 0.02, figure_caption, ha="center", fontsize=9)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Plot task/user/hours vs task/gpu/hours from trace replay JSON."),
    )
    p.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="One or more trace replay report JSON files.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="PNG path (only valid with a single input JSON). "
        "Default: <json_stem>_agent_pareto.png",
    )
    p.add_argument(
        "--figure-caption",
        type=str,
        default=None,
        help="Optional caption below the figure.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.output is not None and len(args.json_files) > 1:
        print(
            "error: --output may only be used with a single JSON file",
            file=sys.stderr,
        )
        return 2
    for jp in args.json_files:
        png = args.output if len(args.json_files) == 1 else None
        written = write_agent_pareto_png_from_json_file(
            jp,
            figure_caption=args.figure_caption,
            png_path=png,
        )
        if written is not None:
            print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
