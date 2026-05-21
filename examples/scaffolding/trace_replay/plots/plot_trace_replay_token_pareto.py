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

Loaded dynamically by examples/scaffolding/trace_replay/pareto/trace_replay_pareto_aggregate.py
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

    ``optimal_cache_hit`` is read from the run row (stamped by the
    aggregator from <trace_dir>/*.cachehit.json). The engine-measured
    per-session max/avg are computed on demand from
    ``replay_assistant_generations_detail`` — there is only one canonical
    cache-hit dataset (the offline UB); engine measurements come straight
    off the run's perf-metrics output.
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


# Caption rendered at the bottom of every PNG so each point's 5-tuple
# label (B / N / C / R / O) is self-documenting without the reader having
# to dig into the aggregator schema.
POINT_LABEL_LEGEND = (
    "B = max_batch_size    N = total_sessions    C = concurrency\n"
    "R_max / R_avg = engine-measured block hit rate per session (max / mean across N)    "
    "O = optimal_cache_hit (offline upper bound)")


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
    labels = [_format_point_label(r) for r, _, _ in usable]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, color="tab:blue", s=64, zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, annotation_clip=False)

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
        png_path = output_json.with_name(output_json.stem + "_throughput_pareto.png")
    else:
        png_path = Path(png_path)

    # fig.tight_layout() removed: would override explicit subplots_adjust
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path
