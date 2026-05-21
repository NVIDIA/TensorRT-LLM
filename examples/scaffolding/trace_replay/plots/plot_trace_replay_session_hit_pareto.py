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

r"""Per-session KV-cache block hit-rate curve for v4 combined JSON.

Loaded dynamically by examples/scaffolding/trace_replay/pareto/trace_replay_pareto_aggregate.py
(via importlib on a resolved sibling path), so this module is intentionally
standalone -- no relative imports, no package init required.

Axes:
    x = max_batch_size (B) -- one value per ladder step
    y = block hit rate    -- one value per session within each step

For each ladder step the plot shows:
    - a marker at the per-session **mean** rate (R_avg)
    - a translucent band spanning the per-session **min .. max** rate
      (i.e. the fluctuation envelope across the N sessions of that step)

The optimal upper bound (offline simulator) is drawn as a horizontal
dashed line for reference.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _per_session_block_hit_rates(detail):
    """Engine-measured per-session block hit rate, computed from a run
    row's ``replay_assistant_generations_detail`` on demand (no cached
    copy in the aggregate JSON)."""
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
    rates: List[float] = []
    for (_ti, _si), (reused, missed) in per_session.items():
        total = reused + missed
        if total > 0:
            rates.append(reused / total)
    return rates

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


CAPTION_LEGEND = (
    "Marker = mean across sessions (R_avg).    "
    "Band = min..max envelope across the step's sessions.\n"
    "Dashed line = optimal_cache_hit (offline upper bound from "
    "<trace_dir>/*.cachehit.json).")


def write_session_hit_pareto_png_from_json_file(
    output_json: PathLike,
    *,
    figure_caption: Optional[str] = None,
    png_path: Optional[PathLike] = None,
) -> Optional[Path]:
    """Render the per-session block hit-rate vs B plot.

    Reads:
      - ``max_batch_size``                          (x value)
      - ``total_sessions``                          (N, annotated)
      - ``replay_assistant_generations_detail``     (engine measurements,
        per-session rates derived on demand)
      - ``optimal_cache_hit``                       (horizontal reference,
        offline upper bound)
    """
    output_json = Path(output_json).expanduser().resolve()
    with output_json.open("r", encoding="utf-8") as f:
        record = json.load(f)

    runs = _successful_runs(record)
    usable = []
    for r in runs:
        b = r.get("max_batch_size")
        rates = _per_session_block_hit_rates(
            r.get("replay_assistant_generations_detail")
        )
        if b is None or not rates:
            continue
        usable.append((int(b), r, list(rates)))
    if not usable:
        print(f"WARNING: no runs in {output_json} have max_batch_size + "
              "per-session detail; skipping session-hit PNG.")
        return None
    usable.sort(key=lambda t: t[0])

    bs = [b for b, _, _ in usable]
    means = [sum(rs) / len(rs) for _, _, rs in usable]
    mins = [min(rs) for _, _, rs in usable]
    maxs = [max(rs) for _, _, rs in usable]
    ns = [r.get("total_sessions") for _, r, _ in usable]

    fig, ax = plt.subplots(figsize=(10, 6))
    # Fluctuation envelope (min..max across the step's sessions).
    ax.fill_between(bs, mins, maxs, alpha=0.20, color="tab:blue",
                    label="min..max across sessions")
    # Mean curve.
    ax.plot(bs, means, marker="o", color="tab:blue", linewidth=2,
            label="mean (R_avg)", zorder=3)

    # Per-point N annotation.
    for b, mean, n in zip(bs, means, ns):
        ax.annotate(f"N={n}", (b, mean), xytext=(6, 6),
                    textcoords="offset points", fontsize=8,
                    annotation_clip=False)

    # Optimal upper bound (constant across steps) -> horizontal dashed line.
    opt = None
    for _, r, _ in usable:
        if isinstance(r.get("optimal_cache_hit"), (int, float)):
            opt = float(r["optimal_cache_hit"])
            break
    if opt is not None:
        ax.axhline(opt, color="tab:red", linestyle="--", linewidth=1.5,
                   label=f"optimal_cache_hit = {opt:.3f}", zorder=2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(bs)
    ax.set_xticklabels([str(b) for b in bs])
    ax.set_xlabel("max_batch_size (B)")
    ax.set_ylabel("per-session block hit rate "
                  "(num_reused_blocks / (num_reused + num_missed))")
    ax.set_title(f"Per-session KV cache hit rate vs B — {_title_suffix(record)}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    # y auto-scale; cap at [0, 1] only if needed.
    ax.set_ylim(top=min(1.02, max(maxs + ([opt] if opt is not None else [])) * 1.05))

    fig.subplots_adjust(bottom=0.18)
    caption = (
        f"{CAPTION_LEGEND}\n"
        f"Trace: {_resolve_trace_label(record)}")
    if figure_caption:
        caption = f"{caption}\n{figure_caption}"
    fig.text(0.5, 0.04, caption, ha="center", fontsize=8, style="italic",
             wrap=True)

    if png_path is None:
        png_path = output_json.with_name(output_json.stem +
                                         "_session_hit_vs_B.png")
    else:
        png_path = Path(png_path)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path
