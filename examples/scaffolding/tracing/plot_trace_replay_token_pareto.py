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

r"""Token-level throughput Pareto PNG for trace_replay_pareto_frontier.v4 JSON.

Loaded dynamically by examples/scaffolding/pareto/trace_replay_pareto_aggregate.py
(via importlib on a resolved sibling path), so this module is intentionally
standalone — no relative imports, no package init required.

Axes follow the InferenceMAX (SemiAnalysis) convention:

    y = (total_input + total_output) / WC / TP    # tokens/s/gpu
                                                  # input AND output, all GPUs
    x = 1000 / median_TPOT_ms                     # tokens/s/user (intvty)

x comes from per-LLM-request TPOT measured by the streaming OpenAI worker
(``(latency_s - ttft_s) / (output_len - 1)``, skipping ``output_len<=1``),
matching SemiAnalysis's ``benchmark_serving.py``. The numerator on y is
reconstructed from trace_meta: per-session prompt + completion totals ×
total_sessions for that step.

Required combined-JSON fields:
    artifact_naming.tensor_parallel_size
    trace_meta.prompt_tokens_assistant_sum   (per-task input tokens)
    trace_meta.completion_tokens_sum         (per-task output tokens)
    runs[i].total_sessions, .wall_clock_s, .median_intvty
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
    # Return indices of points that are non-dominated when both axes are
    # maximized (no other point has x>=mine AND y>=mine with at least one
    # strict inequality). Order the result by x ascending for line plotting.
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


def write_token_pareto_png_from_json_file(
    output_json: PathLike,
    *,
    curve_label: str = "Pareto Frontier",
    figure_caption: Optional[str] = None,
    png_path: Optional[PathLike] = None,
) -> Optional[Path]:
    output_json = Path(output_json).expanduser().resolve()
    with output_json.open("r", encoding="utf-8") as f:
        record = json.load(f)

    runs = _successful_runs(record)
    if not runs:
        print(f"WARNING: no successful runs in {output_json}; skipping throughput Pareto PNG.")
        return None

    naming = record.get("artifact_naming") or {}
    n_gpu = naming.get("tensor_parallel_size")
    if not n_gpu:
        print(
            f"WARNING: artifact_naming.tensor_parallel_size missing in "
            f"{output_json}; skipping throughput Pareto PNG."
        )
        return None
    n_gpu = int(n_gpu)

    tm = record.get("trace_meta") or {}
    prompt_per_task = tm.get("prompt_tokens_assistant_sum")
    comp_per_task = tm.get("completion_tokens_sum")
    if comp_per_task is None:
        comp_per_task = tm.get("assistant_output_tokens_sum")
    if prompt_per_task is None or comp_per_task is None:
        print(
            f"WARNING: {output_json} is missing "
            "trace_meta.prompt_tokens_assistant_sum or "
            "trace_meta.completion_tokens_sum; "
            "skipping throughput Pareto PNG."
        )
        return None
    total_per_task = float(prompt_per_task) + float(comp_per_task)

    usable = []
    for run in runs:
        x = run.get("median_intvty")
        n = run.get("total_sessions")
        wc = run.get("wall_clock_s")
        if x is None or not n or not wc:
            continue
        y = float(n) * total_per_task / (float(wc) * n_gpu)
        usable.append((run, float(x), y))
    if not usable:
        print(
            f"WARNING: no runs in {output_json} have "
            "median_intvty / total_sessions / wall_clock_s; "
            "skipping throughput Pareto PNG."
        )
        return None

    xs = [x for _, x, _ in usable]
    ys = [y for _, _, y in usable]
    labels = [
        (f"step={r.get('ladder_step')} B={r.get('max_batch_size')} C={r.get('concurrency')}")
        for r, _, _ in usable
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, color="tab:blue", s=64, zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

    # Normally the sweep has a real tradeoff and the upper-right Pareto
    # frontier has ≥ 2 points. For degenerate sweeps (e.g. a small ladder
    # where both axes improve monotonically with the same load knob) the
    # strict frontier collapses to one point; fall back to connecting all
    # points in x-sorted order so the ladder trajectory is still visible.
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

    ax.set_title(
        f"Token-level Throughput Pareto — {_title_suffix(record)}\n"
        f"per-task tokens: prompt={int(prompt_per_task)}, "
        f"completion={int(comp_per_task)}"
    )
    ax.set_xlabel("tokens/s/user (intvty) = 1000 / median_TPOT_ms")
    ax.set_ylabel("tokens/s/gpu = N * (PROMPT+COMPLETION) / WC / TP")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if figure_caption:
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.5, 0.02, figure_caption, ha="center", fontsize=8, wrap=True)

    if png_path is None:
        png_path = output_json.with_name(output_json.stem + "_throughput_pareto.png")
    else:
        png_path = Path(png_path)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path
