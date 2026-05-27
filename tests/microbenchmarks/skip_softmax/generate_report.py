#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the kernel microbench CSVs into a single markdown report.

Reads ``pass1_sparsity.csv`` (skipped/total per shape x threshold) and
``pass2_speedup.csv`` (median latency per shape x threshold), joins them,
and emits a markdown file with one summary table per config plus a
sparsity-vs-speedup scatter (matplotlib PNG when available, otherwise a
text table).

The report is intentionally not committed; store output under
``work/skip-softmax-stat/<date>/`` per the user's instructions.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _join_rows(p1: List[Dict[str, str]],
               p2: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Group joined rows by config name."""
    by_cfg: Dict[str, Dict[float, Dict[str, str]]] = defaultdict(dict)
    for r in p1:
        cfg = r["config"]
        thr = float(r["threshold_scale_factor"])
        by_cfg[cfg][thr] = dict(r)
    for r in p2:
        cfg = r["config"]
        thr = float(r["threshold_scale_factor"])
        by_cfg[cfg].setdefault(thr, {"config": cfg, "threshold_scale_factor": r["threshold_scale_factor"]})
        by_cfg[cfg][thr]["elapsed_us_median"] = r.get("elapsed_us_median", "")
        by_cfg[cfg][thr]["speedup"] = r.get("speedup", "")
    out: Dict[str, List[Dict[str, str]]] = {}
    for cfg, by_thr in by_cfg.items():
        out[cfg] = [by_thr[t] for t in sorted(by_thr.keys())]
    return out


def _plot_speedup_vs_sparsity(joined: Dict[str, List[Dict[str, str]]],
                              out_png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(8, 6))
    for cfg, rows in sorted(joined.items()):
        xs: List[float] = []
        ys: List[float] = []
        for r in rows:
            sp = r.get("achieved_sparsity")
            spd = r.get("speedup")
            if sp in (None, "", "None") or spd in (None, "", "None"):
                continue
            try:
                xs.append(float(sp) * 100.0)
                ys.append(float(spd))
            except ValueError:
                continue
        if xs:
            ax.plot(xs, ys, "o-", label=cfg)
    ax.set_xlabel("Achieved sparsity (%)")
    ax.set_ylabel("Kernel speedup vs threshold=0")
    ax.set_title("FMHA skip-softmax: speedup vs achieved sparsity (sm90)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def render_markdown(joined: Dict[str, List[Dict[str, str]]], out_md: Path,
                    plot_png: Path) -> None:
    lines: List[str] = []
    lines.append("# Skip-softmax FMHA kernel microbench (sm90)")
    lines.append("")
    lines.append(
        "Two-pass driver: pass 1 collects achieved sparsity using the "
        "`_skipSoftmaxStat` cubin family; pass 2 measures latency on the "
        "production `_skipSoftmax` family. Inputs are random tensors so the "
        "achieved sparsity does not match the calibrated curves in blog 16 "
        "or the wan22 calibration doc - the relationship between achieved "
        "sparsity and kernel speedup is what we are characterising.")
    lines.append("")
    if plot_png.exists():
        lines.append(f"![speedup vs sparsity]({plot_png.name})")
        lines.append("")
    for cfg, rows in sorted(joined.items()):
        lines.append(f"## {cfg}")
        lines.append("")
        lines.append(
            "| threshold | skipped / total | achieved sparsity | latency (us) | speedup |"
        )
        lines.append(
            "|----------:|------------------|------------------:|-------------:|--------:|"
        )
        for r in rows:
            thr = r.get("threshold_scale_factor", "")
            skipped = r.get("skipped_blocks", "") or ""
            total = r.get("total_blocks", "") or ""
            sp = r.get("achieved_sparsity", "") or ""
            try:
                sp_pct = f"{float(sp) * 100:.2f} %"
            except (ValueError, TypeError):
                sp_pct = "-"
            lat = r.get("elapsed_us_median", "") or ""
            spd = r.get("speedup", "") or ""
            try:
                lat_str = f"{float(lat):.2f}"
            except (ValueError, TypeError):
                lat_str = "-"
            try:
                spd_str = f"{float(spd):.3f}x"
            except (ValueError, TypeError):
                spd_str = "-"
            sk_total = f"{skipped} / {total}" if total else "-"
            lines.append(
                f"| {thr} | {sk_total} | {sp_pct} | {lat_str} | {spd_str} |")
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    p1 = _read_csv(args.out_dir / "pass1_sparsity.csv")
    p2 = _read_csv(args.out_dir / "pass2_speedup.csv")
    if not p1 and not p2:
        print(f"no CSVs found under {args.out_dir}", file=sys.stderr)
        return 1
    joined = _join_rows(p1, p2)
    plot_png = args.out_dir / "speedup_vs_sparsity.png"
    _plot_speedup_vs_sparsity(joined, plot_png)
    render_markdown(joined, args.out_dir / "report.md", plot_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
