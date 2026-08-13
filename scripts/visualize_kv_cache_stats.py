#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Visualize per-iteration KV cache statistics from a JSON file.

Produced by ``test_auto_dtype_vswa_reuse_kv_cache_stats`` (or any test that
writes the same schema).

Usage:
    python scripts/visualize_kv_cache_stats.py kv_cache_stats_output/kv_cache_stats_*.json
    python scripts/visualize_kv_cache_stats.py stats.json --output charts.png
    python scripts/visualize_kv_cache_stats.py stats.json --per-window
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def load_stats(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_kv_entries(payload: dict, per_window: bool = False):
    """Return a list of dicts, one per iteration with kvCacheIterationStats.

    Each dict has the stat fields plus ``_iter`` (global iteration index),
    ``_phase``, and ``_collectedAt``.

    When *per_window* is False the fields are aggregated (summed for
    counters, averaged for rates, maxed for gauges) across window sizes.
    When True a ``_windowSize`` key is added and each window size
    produces its own row.
    """
    rows = []
    for entry in payload.get("stats", []):
        kv = entry.get("kvCacheIterationStats")
        if not kv:
            continue
        iteration = entry.get("iter", len(rows))
        phase = entry.get("_phase", "")
        collected_at = entry.get("_collectedAt", 0)

        if per_window:
            for ws, fields in kv.items():
                row = dict(fields)
                row["_iter"] = iteration
                row["_phase"] = phase
                row["_collectedAt"] = collected_at
                row["_windowSize"] = int(ws)
                rows.append(row)
        else:
            # Aggregate across window sizes
            agg = _aggregate_windows(kv)
            agg["_iter"] = iteration
            agg["_phase"] = phase
            agg["_collectedAt"] = collected_at
            rows.append(agg)
    return rows


_GAUGE_FIELDS = {
    "primaryMaxNumBlocks",
    "primaryFreeNumBlocks",
    "primaryUsedNumBlocks",
    "secondaryMaxNumBlocks",
    "secondaryFreeNumBlocks",
    "secondaryUsedNumBlocks",
}
_RATE_FIELDS = {"iterCacheHitRate"}


def _aggregate_windows(kv: dict) -> dict:
    """Aggregate stats across window sizes."""
    agg = {}
    n = len(kv)
    if n == 0:
        return agg
    for ws, fields in kv.items():
        for k, v in fields.items():
            if k in _GAUGE_FIELDS:
                agg[k] = max(agg.get(k, 0), v)
            elif k in _RATE_FIELDS:
                agg[k] = agg.get(k, 0) + v
            else:
                agg[k] = agg.get(k, 0) + v
    # Average the rate fields
    for k in _RATE_FIELDS:
        if k in agg:
            agg[k] /= n
    return agg


def _field_series(rows, field):
    return np.array([r.get(field, 0) for r in rows], dtype=np.float64)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all(rows, title_prefix: str = "", per_window: bool = False):
    """Create a multi-panel figure with KV cache diagnostics."""
    if not rows:
        print("No kvCacheIterationStats entries found.", file=sys.stderr)
        sys.exit(1)

    iters = np.arange(len(rows))

    # Detect phase boundaries for vertical lines
    phases = [r["_phase"] for r in rows]
    phase_boundaries = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            phase_boundaries.append((i, phases[i]))

    fig, axes = plt.subplots(8, 1, figsize=(14, 32), sharex=True)
    fig.suptitle(f"{title_prefix}KV Cache Iteration Statistics", fontsize=14, y=0.98)

    def _add_phase_markers(ax):
        for idx, label in phase_boundaries:
            ax.axvline(idx, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(
                idx, ax.get_ylim()[1], f" {label}", fontsize=7, va="top", ha="left", color="grey"
            )

    # --- Panel 1: GPU Pool Utilization ---
    ax = axes[0]
    max_blocks = _field_series(rows, "primaryMaxNumBlocks")
    used_blocks = _field_series(rows, "primaryUsedNumBlocks")
    free_blocks = _field_series(rows, "primaryFreeNumBlocks")
    utilization = np.where(max_blocks > 0, used_blocks / max_blocks, 0)

    ax.fill_between(iters, utilization, alpha=0.3, color="tab:blue", label="GPU utilization")
    ax.plot(iters, utilization, color="tab:blue", linewidth=1)
    ax.set_ylabel("GPU Pool Utilization")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Primary (GPU) Pool Utilization", fontsize=10)
    _add_phase_markers(ax)

    # Annotate block counts on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(
        iters,
        used_blocks,
        color="tab:orange",
        linewidth=0.8,
        linestyle=":",
        alpha=0.7,
        label="used blocks",
    )
    ax2.plot(
        iters,
        free_blocks,
        color="tab:green",
        linewidth=0.8,
        linestyle=":",
        alpha=0.7,
        label="free blocks",
    )
    ax2.set_ylabel("Block Count")
    ax2.legend(loc="upper left", fontsize=7)

    # --- Panel 2: Block Reuse Breakdown ---
    ax = axes[1]
    full_reuse = _field_series(rows, "iterFullReusedBlocks")
    partial_reuse = _field_series(rows, "iterPartialReusedBlocks")
    missed = _field_series(rows, "iterMissedBlocks")

    ax.bar(iters, full_reuse, label="Full Reuse", color="tab:green", alpha=0.8, width=1.0)
    ax.bar(
        iters,
        partial_reuse,
        bottom=full_reuse,
        label="Partial Reuse",
        color="tab:orange",
        alpha=0.8,
        width=1.0,
    )
    ax.bar(
        iters,
        missed,
        bottom=full_reuse + partial_reuse,
        label="Missed",
        color="tab:red",
        alpha=0.8,
        width=1.0,
    )
    ax.set_ylabel("Blocks")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Block Reuse Breakdown (per iteration)", fontsize=10)
    _add_phase_markers(ax)

    # --- Panel 3: Cache Hit Rate ---
    ax = axes[2]
    hit_rate = _field_series(rows, "iterCacheHitRate")
    ax.plot(iters, hit_rate, color="tab:purple", linewidth=1)
    ax.fill_between(iters, hit_rate, alpha=0.2, color="tab:purple")
    ax.set_ylabel("Cache Hit Rate")
    ax.set_ylim(0, max(1.05, hit_rate.max() * 1.1) if hit_rate.max() > 0 else 1.05)
    ax.set_title("Per-Iteration Cache Hit Rate", fontsize=10)
    _add_phase_markers(ax)

    # --- Panel 4: Context-Phase Allocation ---
    ax = axes[3]
    alloc_total = _field_series(rows, "iterAllocTotalBlocks")
    ax.plot(iters, alloc_total, label="AllocTotal", color="tab:blue", linewidth=1)
    ax.fill_between(iters, alloc_total, alpha=0.2, color="tab:blue")
    ax.set_ylabel("Blocks")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Context-Phase Allocation (per iteration)", fontsize=10)
    _add_phase_markers(ax)

    # --- Panel 5: Generation-Phase Allocation ---
    ax = axes[4]
    alloc_new = _field_series(rows, "iterAllocNewBlocks")
    gen_alloc = _field_series(rows, "iterGenAllocBlocks")
    ax.plot(iters, gen_alloc, label="GenAlloc", color="tab:brown", linewidth=1)
    ax.plot(iters, alloc_new, label="AllocNew", color="tab:cyan", linewidth=1)
    ax.set_ylabel("Blocks")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Generation-Phase Allocation (per iteration)", fontsize=10)
    _add_phase_markers(ax)

    # --- Panel 6: Onboard Traffic (Host → GPU) ---
    ax = axes[5]
    onboard_bytes = _field_series(rows, "iterOnboardBytes")
    onboard_mib = onboard_bytes / (1024 * 1024)
    ax.plot(iters, onboard_mib, label="Onboard (Host→GPU)", color="tab:blue", linewidth=1)
    ax.fill_between(iters, onboard_mib, alpha=0.2, color="tab:blue")
    ax.set_ylabel("MiB")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Onboard Traffic — Host → GPU (per iteration)", fontsize=10)
    _add_phase_markers(ax)
    if onboard_bytes.sum() == 0:
        ax.text(
            0.5,
            0.5,
            "No onboard transfers (secondary pool inactive)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="grey",
            style="italic",
        )

    # --- Panel 7: Offload Traffic (GPU → Host) ---
    ax = axes[6]
    offload_bytes = _field_series(rows, "iterOffloadBytes")
    offload_mib = offload_bytes / (1024 * 1024)
    ax.plot(iters, offload_mib, label="Offload (GPU→Host)", color="tab:red", linewidth=1)
    ax.fill_between(iters, offload_mib, alpha=0.2, color="tab:red")
    ax.set_ylabel("MiB")
    ax.set_xlabel("Iteration Index")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Offload Traffic — GPU → Host (per iteration)", fontsize=10)
    _add_phase_markers(ax)
    if offload_bytes.sum() == 0:
        ax.text(
            0.5,
            0.5,
            "No offload transfers (secondary pool inactive)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="grey",
            style="italic",
        )

    # --- Panel 8: Intra-Device Copy Traffic (GPU → GPU) ---
    ax = axes[7]
    intra_copy_bytes = _field_series(rows, "iterIntraDeviceCopyBytes")
    intra_copy_mib = intra_copy_bytes / (1024 * 1024)
    ax.plot(
        iters, intra_copy_mib, label="Intra-Device Copy (GPU→GPU)", color="tab:olive", linewidth=1
    )
    ax.fill_between(iters, intra_copy_mib, alpha=0.2, color="tab:olive")
    ax.set_ylabel("MiB")
    ax.set_xlabel("Iteration Index")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Intra-Device Copy — GPU → GPU (per iteration)", fontsize=10)
    _add_phase_markers(ax)
    if intra_copy_bytes.sum() == 0:
        ax.text(
            0.5,
            0.5,
            "No intra-device copies",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="grey",
            style="italic",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_per_window(rows, title_prefix: str = ""):
    """One figure per window size with the same 5-panel layout."""
    window_sizes = sorted({r["_windowSize"] for r in rows})
    figs = {}
    for ws in window_sizes:
        ws_rows = [r for r in rows if r["_windowSize"] == ws]
        fig = plot_all(ws_rows, title_prefix=f"{title_prefix}[window={ws}] ")
        figs[ws] = fig
    return figs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize KV cache iteration statistics from JSON"
    )
    parser.add_argument("json_file", help="Path to the stats JSON file")
    parser.add_argument("--output", "-o", help="Output image path (default: display interactively)")
    parser.add_argument(
        "--per-window", action="store_true", help="Generate separate charts per window size"
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved images (default: 150)")
    args = parser.parse_args()

    payload = load_stats(args.json_file)
    model = payload.get("model", "unknown")
    num_entries = payload.get("num_entries", "?")
    print(f"Model: {model}")
    print(f"Total stats entries: {num_entries}")

    if args.per_window:
        rows = extract_kv_entries(payload, per_window=True)
        window_sizes = sorted({r["_windowSize"] for r in rows})
        print(f"Window sizes: {window_sizes}")
        kv_count = len(rows)
        print(f"Entries with kvCacheIterationStats: {kv_count}")
        if kv_count == 0:
            print("No kvCacheIterationStats found in any entry.", file=sys.stderr)
            sys.exit(1)

        figs = plot_per_window(rows, title_prefix=f"{model} — ")
        if args.output:
            out = Path(args.output)
            for ws, fig in figs.items():
                p = out.with_stem(f"{out.stem}_window{ws}")
                fig.savefig(p, dpi=args.dpi, bbox_inches="tight")
                print(f"Saved: {p}")
            plt.close("all")
        else:
            plt.show()
    else:
        rows = extract_kv_entries(payload, per_window=False)
        kv_count = len(rows)
        print(f"Entries with kvCacheIterationStats: {kv_count}")
        if kv_count == 0:
            print("No kvCacheIterationStats found in any entry.", file=sys.stderr)
            sys.exit(1)

        fig = plot_all(rows, title_prefix=f"{model} — ")
        if args.output:
            fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved: {args.output}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
