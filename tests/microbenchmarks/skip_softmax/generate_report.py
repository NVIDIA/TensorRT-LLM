#!/usr/bin/env python3
#
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
"""Render skip-softmax microbench CSVs into markdown reports."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PASS1_CSV = "pass1_sparsity.csv"
PASS2_CSV = "pass2_speedup.csv"
TRANSITION_MIN_SPARSITY = 0.05
TRANSITION_MAX_SPARSITY = 0.95
SPEEDUP_REL_TOLERANCE = 0.05


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _join_rows(
    p1: list[dict[str, str]], p2: list[dict[str, str]]
) -> dict[str, list[dict[str, str]]]:
    """Group joined rows by config name."""
    by_cfg: dict[str, dict[float, dict[str, str]]] = defaultdict(dict)
    for row in p1:
        cfg = row["config"]
        threshold = float(row["threshold_scale_factor"])
        by_cfg[cfg][threshold] = dict(row)
    for row in p2:
        cfg = row["config"]
        threshold = float(row["threshold_scale_factor"])
        by_cfg[cfg].setdefault(
            threshold,
            {
                "config": cfg,
                "threshold_scale_factor": row["threshold_scale_factor"],
            },
        )
        by_cfg[cfg][threshold]["elapsed_us_median"] = row.get("elapsed_us_median", "")
        by_cfg[cfg][threshold]["speedup"] = row.get("speedup", "")

    joined: dict[str, list[dict[str, str]]] = {}
    for cfg, by_threshold in by_cfg.items():
        joined[cfg] = [by_threshold[threshold] for threshold in sorted(by_threshold.keys())]
    return joined


def _joined_from_dir(out_dir: Path) -> dict[str, list[dict[str, str]]]:
    return _join_rows(_read_csv(out_dir / PASS1_CSV), _read_csv(out_dir / PASS2_CSV))


def _float_value(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_float(value: str | None, precision: int = 3) -> str:
    parsed = _float_value(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.{precision}f}"


def _format_pct(value: str | None) -> str:
    parsed = _float_value(value)
    if parsed is None:
        return "N/A"
    return f"{parsed * 100.0:.2f} %"


def _row_sparsity(row: dict[str, str]) -> float | None:
    sparsity = _float_value(row.get("achieved_sparsity"))
    if sparsity is not None:
        return sparsity
    threshold = _float_value(row.get("threshold_scale_factor"))
    if threshold == 0:
        return 0.0
    return None


def _format_row_sparsity(row: dict[str, str]) -> str:
    sparsity = _row_sparsity(row)
    if sparsity is None:
        return "N/A"
    return f"{sparsity * 100.0:.2f} %"


def _format_speedup(value: str | None) -> str:
    parsed = _float_value(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.3f}x"


def _config_title(cfg: str, rows: list[dict[str, str]]) -> str:
    if not rows:
        return cfg
    dtype = rows[0].get("dtype", "")
    seq_len = rows[0].get("seq_len_kv", "")
    mask = rows[0].get("mask", "")
    num_heads_q = rows[0].get("num_heads_q", "")
    num_heads_kv = rows[0].get("num_heads_kv", "")
    if cfg.startswith("wan22_a14b_720p"):
        return f"Wan2.2 A14B 720p {dtype} non-causal s{seq_len}"
    if mask == "bidirectional":
        return f"Diffusion {dtype} non-causal s{seq_len}"
    if dtype and seq_len:
        gqa = ""
        if num_heads_q and num_heads_kv and num_heads_q != num_heads_kv:
            gqa = f" GQA {num_heads_q}/{num_heads_kv}"
        return f"LLM prefill {dtype} {int(seq_len) // 1024}k{gqa}"
    return cfg


def _row_by_threshold(rows: list[dict[str, str]]) -> dict[float, dict[str, str]]:
    return {float(row["threshold_scale_factor"]): row for row in rows}


def _transition_count(rows: list[dict[str, str]]) -> int:
    count = 0
    for row in rows:
        sparsity = _row_sparsity(row)
        if (
            sparsity is not None
            and TRANSITION_MIN_SPARSITY < sparsity
            and sparsity < TRANSITION_MAX_SPARSITY
        ):
            count += 1
    return count


def _relative_delta(lhs: str | None, rhs: str | None) -> float | None:
    lhs_value = _float_value(lhs)
    rhs_value = _float_value(rhs)
    if lhs_value is None or rhs_value is None:
        return None
    denominator = (abs(lhs_value) + abs(rhs_value)) / 2.0
    if denominator == 0:
        return 0.0
    return abs(lhs_value - rhs_value) / denominator


def _plot_speedup_vs_sparsity(joined: dict[str, list[dict[str, str]]], out_png: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    for cfg, rows in sorted(joined.items()):
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            sparsity = _row_sparsity(row)
            speedup = _float_value(row.get("speedup"))
            if sparsity is None or speedup is None:
                continue
            xs.append(sparsity * 100.0)
            ys.append(speedup)
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


def _plot_backend_overlay(
    backend_rows: dict[str, dict[str, list[dict[str, str]]]], cfg: str, out_png: Path
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False
    title_rows: list[dict[str, str]] = []
    for backend, joined in sorted(backend_rows.items()):
        rows = joined.get(cfg, [])
        title_rows = title_rows or rows
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            sparsity = _row_sparsity(row)
            speedup = _float_value(row.get("speedup"))
            if sparsity is None or speedup is None:
                continue
            xs.append(sparsity * 100.0)
            ys.append(speedup)
        if xs:
            ax.plot(xs, ys, "o-", label=backend)
            plotted = True
    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Achieved sparsity (%)")
    ax.set_ylabel("Kernel speedup vs threshold=0")
    ax.set_title(_config_title(cfg, title_rows))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return True


def _plot_name(cfg: str) -> str:
    return f"speedup_vs_sparsity_{cfg}.png"


def render_markdown(joined: dict[str, list[dict[str, str]]], out_md: Path, plot_png: Path) -> None:
    lines: list[str] = []
    lines.append("# Skip-softmax FMHA kernel microbench (sm90)")
    lines.append("")
    lines.append(
        "Two-pass driver: pass 1 collects achieved sparsity using the "
        "`_skipSoftmaxStat` cubin family; pass 2 measures latency on the "
        "production `_skipSoftmax` family. Inputs are random tensors so the "
        "achieved sparsity does not match the calibrated curves in blog 16 "
        "or the wan22 calibration doc; the relationship between achieved "
        "sparsity and kernel speedup is what we are characterising."
    )
    lines.append("")
    if plot_png.exists():
        lines.append(f"![speedup vs sparsity]({plot_png.name})")
        lines.append("")
    for cfg, rows in sorted(joined.items()):
        lines.append(f"## {cfg}")
        lines.append("")
        lines.append("| threshold | skipped / total | achieved sparsity | latency (us) | speedup |")
        lines.append(
            "|----------:|------------------|------------------:|-------------:|--------:|"
        )
        for row in rows:
            threshold = row.get("threshold_scale_factor", "")
            skipped = row.get("skipped_blocks", "") or ""
            total = row.get("total_blocks", "") or ""
            skipped_total = f"{skipped} / {total}" if total else "N/A"
            lines.append(
                f"| {threshold} | {skipped_total} | "
                f"{_format_row_sparsity(row)} | "
                f"{_format_float(row.get('elapsed_us_median'), 2)} | "
                f"{_format_speedup(row.get('speedup'))} |"
            )
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")


def render_backend_overlay_markdown(
    backend_rows: dict[str, dict[str, list[dict[str, str]]]], out_md: Path
) -> None:
    cfgs = sorted({cfg for rows in backend_rows.values() for cfg in rows})
    lines: list[str] = []
    lines.append("# Skip-softmax seqlen-relative sweep (H200 sm90)")
    lines.append("")
    lines.append(
        "Thresholds are per-shape `threshold_scale_factor` values derived by "
        "multiplying each config's `seq_len_kv`. Each plot overlays the "
        "`fmha-exe` standalone path and the `torch-op` Python path."
    )
    lines.append("")

    for cfg in cfgs:
        first_rows = next((rows[cfg] for rows in backend_rows.values() if cfg in rows), [])
        plot_png = out_md.parent / _plot_name(cfg)
        if _plot_backend_overlay(backend_rows, cfg, plot_png):
            lines.append(f"![{cfg}]({plot_png.name})")
            lines.append("")

        lines.append(f"## {_config_title(cfg, first_rows)}")
        lines.append("")
        for backend, joined in sorted(backend_rows.items()):
            count = _transition_count(joined.get(cfg, []))
            lines.append(f"- `{backend}` transition-zone points (5-95% sparsity): {count}")
        lines.append("")

        backend_by_threshold = {
            backend: _row_by_threshold(joined.get(cfg, []))
            for backend, joined in backend_rows.items()
        }
        thresholds = sorted(
            {threshold for rows in backend_by_threshold.values() for threshold in rows}
        )
        seq_len = _float_value(first_rows[0].get("seq_len_kv") if first_rows else None)

        lines.append(
            "| multiplier | threshold | fmha-exe sparsity | fmha-exe latency (us) | "
            "fmha-exe speedup | torch-op sparsity | torch-op latency (us) | "
            "torch-op speedup | speedup delta |"
        )
        lines.append(
            "|-----------:|----------:|------------------:|----------------------:|"
            "-----------------:|-----------------:|---------------------:|"
            "----------------:|--------------:|"
        )
        for threshold in thresholds:
            fmha = backend_by_threshold.get("fmha-exe", {}).get(threshold, {})
            torch_op = backend_by_threshold.get("torch-op", {}).get(threshold, {})
            delta = _relative_delta(fmha.get("speedup"), torch_op.get("speedup"))
            if delta is None:
                delta_text = "N/A"
            else:
                delta_text = f"{delta * 100.0:.2f} %"
                if delta > SPEEDUP_REL_TOLERANCE:
                    delta_text += " (FLAG)"
            multiplier = "N/A"
            if seq_len:
                multiplier = f"{threshold / seq_len:.2f}"
            lines.append(
                f"| {multiplier} | {threshold:g} | "
                f"{_format_row_sparsity(fmha)} | "
                f"{_format_float(fmha.get('elapsed_us_median'), 2)} | "
                f"{_format_speedup(fmha.get('speedup'))} | "
                f"{_format_row_sparsity(torch_op)} | "
                f"{_format_float(torch_op.get('elapsed_us_median'), 2)} | "
                f"{_format_speedup(torch_op.get('speedup'))} | "
                f"{delta_text} |"
            )
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")


def _parse_backend_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--backend-dir must have the form backend=/path/to/csv_dir"
        )
    backend, path = value.split("=", 1)
    if not backend:
        raise argparse.ArgumentTypeError("backend name must not be empty")
    return backend, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--backend-dir",
        action="append",
        default=[],
        type=_parse_backend_dir,
        help="backend=/path/to/csv_dir. Repeat to build an overlay report.",
    )
    parser.add_argument("--report-name", default="report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.backend_dir:
        backend_rows = {backend: _joined_from_dir(path) for backend, path in args.backend_dir}
        if not any(backend_rows.values()):
            print("no CSVs found under any --backend-dir", file=sys.stderr)
            return 1
        render_backend_overlay_markdown(backend_rows, args.out_dir / args.report_name)
        return 0

    p1 = _read_csv(args.out_dir / PASS1_CSV)
    p2 = _read_csv(args.out_dir / PASS2_CSV)
    if not p1 and not p2:
        print(f"no CSVs found under {args.out_dir}", file=sys.stderr)
        return 1
    joined = _join_rows(p1, p2)
    plot_png = args.out_dir / "speedup_vs_sparsity.png"
    _plot_speedup_vs_sparsity(joined, plot_png)
    render_markdown(joined, args.out_dir / args.report_name, plot_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
