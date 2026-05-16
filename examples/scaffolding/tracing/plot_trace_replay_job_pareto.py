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

Loaded dynamically by examples/scaffolding/pareto/trace_replay_pareto_aggregate.py
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
    labels = [
        (f"step={r.get('ladder_step')} B={r.get('max_batch_size')} C={r.get('concurrency')}")
        for r, _, _ in usable
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, color="tab:purple", s=64, zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

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
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if figure_caption:
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.5, 0.02, figure_caption, ha="center", fontsize=8, wrap=True)

    if png_path is None:
        png_path = output_json.with_name(output_json.stem + "_job_pareto.png")
    else:
        png_path = Path(png_path)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path
