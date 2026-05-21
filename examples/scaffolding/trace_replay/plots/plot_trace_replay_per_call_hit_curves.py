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

r"""Per-LLM-call KV cache hit-rate trajectory, sampled across sessions.

Loaded dynamically by examples/scaffolding/trace_replay/pareto/trace_replay_pareto_aggregate.py.

x = LLM call index along the trace (0..n_calls-1, **trace event order**).
    Each session's i-th measured call is the entry whose
    ``(branch_path, hop_within_branch)`` matches the i-th assistant event
    in the trace.cachehit.json. This aligns every session's i-th sample
    to the same trace event regardless of how the engine scheduler
    interleaved concurrent sub-agent branches at runtime.
y = per-call ``kv_cache_hit_rate`` (engine-measured block hit rate for the
    prefill of that call)

For a high-N single-step run we sample every ``stride`` sessions (default
50) after sorting sessions by start time, so the legend shows arrival
spread across the run. Each sampled session contributes one line.

For multi-step ladder runs the plotter emits one PNG per step.
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

DEFAULT_STRIDE = 40


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


def _resolve_cachehit_path(record: Dict[str, Any]) -> Optional[Path]:
    """Locate the ``*.trace.cachehit.json`` file that pairs with the trace
    consumed by this record. Prefer ``trace_meta.trace_file`` (single-trace
    case); fall back to scanning ``trace_meta.trace_dir`` /
    ``trace_meta.traces[0]`` for a file matching ``*.trace.cachehit.json``."""
    tm = record.get("trace_meta") or {}
    candidates: List[Path] = []
    tf = tm.get("trace_file")
    if isinstance(tf, str) and tf:
        tfp = Path(tf)
        if tfp.name.endswith(".trace.json"):
            candidates.append(
                tfp.with_name(tfp.name[: -len(".trace.json")]
                              + ".trace.cachehit.json"))
        else:
            candidates.append(tfp.with_suffix(".cachehit.json"))
    traces = tm.get("traces")
    if isinstance(traces, list) and traces:
        first = traces[0] or {}
        ftf = first.get("trace_file")
        if isinstance(ftf, str) and ftf:
            ftp = Path(ftf)
            if ftp.name.endswith(".trace.json"):
                candidates.append(
                    ftp.with_name(ftp.name[: -len(".trace.json")]
                                  + ".trace.cachehit.json"))
        ftd = first.get("trace_dir")
        if isinstance(ftd, str) and ftd:
            candidates.extend(Path(ftd).glob("*.trace.cachehit.json"))
    td = tm.get("trace_dir")
    if isinstance(td, str) and td:
        candidates.extend(Path(td).glob("*.trace.cachehit.json"))

    for c in candidates:
        if c.exists():
            return c
    return None


def _trace_optimal_and_branches(
    record: Dict[str, Any],
) -> Tuple[List[float], List[Tuple[int, ...]]]:
    """Return (optimal_hit_rates, trace_branches) walking the paired
    cachehit JSON's events in trace order.

    ``optimal_hit_rates[i]`` = ``optimal_cache_block_hit_rate`` at the
    i-th assistant event. ``trace_branches[i]`` = that event's
    ``branch_path`` as a tuple. The two lists are aligned 1:1.
    Empty tuples on both sides if the file is missing.
    """
    p = _resolve_cachehit_path(record)
    if p is None:
        return [], []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARNING: could not read cachehit json {p}: {e}")
        return [], []
    rates: List[float] = []
    branches: List[Tuple[int, ...]] = []
    for ev in data.get("events") or []:
        if ev.get("event_type") != "message":
            continue
        if ev.get("role") != "assistant":
            continue
        r = ev.get("optimal_cache_block_hit_rate")
        if not isinstance(r, (int, float)):
            continue
        rates.append(float(r))
        branches.append(tuple(ev.get("branch_path") or []))
    return rates, branches


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


def _sessions_with_call_curves(
    detail: List[Dict[str, Any]],
    trace_branches: List[Tuple[int, ...]],
) -> List[Tuple[float, Tuple[int, int], List[float]]]:
    """Return [(start_time, (trace_idx, session_idx), [per_call_hit_rate, ...]), ...]
    sorted by start_time ascending.

    Within each session the per-call list is in **trace event order**,
    not first_iter order. Each chart x-axis slot maps to exactly the
    same trace event across all sessions: position ``i`` always shows
    the engine measurement for the i-th assistant event in
    ``trace_branches`` (= the trace.cachehit.json sequence).

    Mapping: group a session's detail entries by ``branch_path`` and
    sort each group by ``first_iter`` (= hop order within that branch,
    which is necessarily trace-respecting because hop k+1 depends on
    hop k's decode). Then walk ``trace_branches`` in order, popping the
    head of the corresponding branch group for each step.
    """
    by_session: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for e in detail or []:
        si = e.get("session_index")
        if si is None:
            continue
        ti = int(e.get("trace_index") or 0)
        by_session[(ti, int(si))].append(e)

    out: List[Tuple[float, Tuple[int, int], List[float]]] = []
    for sid, calls in by_session.items():
        if not calls:
            continue
        # Bucket by branch_path; sort each bucket by first_iter to get
        # the within-branch hop order.
        by_branch: Dict[Tuple[int, ...], List[Dict[str, Any]]] = defaultdict(list)
        for c in calls:
            bp = tuple(c.get("branch_path") or [])
            by_branch[bp].append(c)
        for bp in by_branch:
            by_branch[bp].sort(key=lambda c: c.get("first_iter") or 0)

        # Walk trace_branches in order; for each step, take the next
        # unused entry from this session's by_branch bucket. This yields
        # the session's detail entries in trace event order.
        if trace_branches:
            consumed: Dict[Tuple[int, ...], int] = {bp: 0 for bp in by_branch}
            ordered: List[Dict[str, Any]] = []
            for tbp in trace_branches:
                bucket = by_branch.get(tbp)
                if bucket is None:
                    continue
                k = consumed[tbp]
                if k < len(bucket):
                    ordered.append(bucket[k])
                    consumed[tbp] = k + 1
        else:
            # No trace_branches available; fall back to first_iter
            # ordering (degenerate but keeps the plot usable).
            ordered = sorted(calls, key=lambda c: c.get("first_iter") or 0)

        starts = [c.get("arrival_time") or c.get("server_arrival_time")
                  for c in ordered]
        starts = [s for s in starts if isinstance(s, (int, float))]
        if not starts:
            continue
        start = float(min(starts))
        hit_rates: List[float] = []
        for c in ordered:
            r = c.get("kv_cache_hit_rate")
            if isinstance(r, (int, float)):
                hit_rates.append(float(r))
        if not hit_rates:
            continue
        out.append((start, sid, hit_rates))
    out.sort(key=lambda x: x[0])
    return out


def _render_one(
    *,
    record: Dict[str, Any],
    run: Dict[str, Any],
    output_json: Path,
    figure_caption: Optional[str],
    png_path: Optional[Path],
    stride: int,
) -> Optional[Path]:
    detail = run.get("replay_assistant_generations_detail")
    if not detail:
        return None
    opt_rates, trace_branches = _trace_optimal_and_branches(record)
    sessions = _sessions_with_call_curves(detail, trace_branches)
    if not sessions:
        return None

    t0 = sessions[0][0]
    # Sample every `stride` sessions (in start-time order); always include
    # the first and last so the temporal span is visible.
    n = len(sessions)
    idxs = list(range(0, n, max(stride, 1)))
    if (n - 1) not in idxs:
        idxs.append(n - 1)

    cmap = plt.get_cmap("viridis")
    color_steps = max(len(idxs) - 1, 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    for plot_i, sess_i in enumerate(idxs):
        start, (ti, si), hit_rates = sessions[sess_i]
        rel_t = start - t0
        color = cmap(plot_i / color_steps)
        xs = list(range(len(hit_rates)))
        ax.plot(xs, hit_rates, color=color, linewidth=1.2, marker=".",
                markersize=4, alpha=0.85,
                label=f"session #{si}  t={rel_t:6.1f}s")

    if opt_rates:
        ox = list(range(len(opt_rates)))
        ax.plot(ox, opt_rates, color="tab:red", linestyle="--", linewidth=1.4,
                marker="x", markersize=4, alpha=0.85,
                label="optimal per-call cache block hit rate", zorder=4)

    b = run.get("max_batch_size")
    c = run.get("concurrency")
    n_sessions = run.get("total_sessions")
    ax.set_xlabel("LLM call index along the trace (0 = first assistant turn)")
    ax.set_ylabel(
        "per-call kv_cache_hit_rate "
        "(num_reused_blocks / (num_reused + num_missed) for that prefill)")
    ax.set_title(
        f"Per-call KV hit-rate trajectories — {_title_suffix(record)}\n"
        f"B={b}  N={n_sessions}  C={c}  "
        f"(showing {len(idxs)} of {n} sessions, sampled every {stride} in start-time order)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)

    fig.subplots_adjust(bottom=0.18)
    legend = (
        f"x = LLM call index (0..{max(len(s[2]) for s in sessions) - 1} "
        "for this trace, **trace event order**; each session's i-th sample "
        "is aligned to the same trace event via (branch_path, hop)).    "
        "y = engine kv_cache_hit_rate per prefill.\n"
        f"Curves: every {stride}th session by start time. "
        "Dashed red = optimal per-call cache block hit rate "
        "(from trace cachehit.json, offline upper bound per LLM call).")
    caption = f"{legend}\nTrace: {_resolve_trace_label(record)}"
    if figure_caption:
        caption = f"{caption}\n{figure_caption}"
    fig.text(0.5, 0.04, caption, ha="center", fontsize=8, style="italic",
             wrap=True)

    if png_path is None:
        suffix = (f"_per_call_hit_curves_B{b}.png" if b is not None
                  else "_per_call_hit_curves.png")
        png_path = output_json.with_name(output_json.stem + suffix)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def write_per_call_hit_curves_png_from_json_file(
    output_json: PathLike,
    *,
    figure_caption: Optional[str] = None,
    png_path: Optional[PathLike] = None,
    stride: int = DEFAULT_STRIDE,
) -> Optional[Path]:
    """Emit one ``*_per_call_hit_curves_B<B>.png`` per successful run."""
    output_json = Path(output_json).expanduser().resolve()
    with output_json.open("r", encoding="utf-8") as f:
        record = json.load(f)

    runs = _successful_runs(record)
    if not runs:
        print(f"WARNING: no successful runs in {output_json}; "
              "skipping per-call hit-curves PNG.")
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
            stride=stride,
        )
        if png is not None:
            last_png = png
    return last_png
