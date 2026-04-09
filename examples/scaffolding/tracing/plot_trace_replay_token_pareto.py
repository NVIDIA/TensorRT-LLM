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

r"""Plot median tok/s per user vs aggregate tok/s per GPU from trace replay JSON.

Reads the same schema written by ``run_trace_replay_pareto_frontier.py`` (and
compatible v2/v3 reports). For each successful run:

* **x** = ``median_tps_per_user`` (or ``pareto_x_median_tps_per_user``).
* **y** = ``output_tps_per_gpu`` (or ``pareto_y_output_tps_per_gpu``).

Points are annotated with ``ladder_step`` (fallback: ``max_batch_size``).

Example::

    python examples/scaffolding/tracing/plot_trace_replay_token_pareto.py \\
        examples/scaffolding/tracing/traces/sympy__sympy-21847/run_folder/sympy__sympy-21847_model_tp4_ep4_20260414_020107.json

Writes ``<stem>_throughput_pareto.png`` next to each JSON unless ``--output`` is set
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


def collect_token_pareto_points(runs: List[Dict[str, Any]]) -> List[Point3]:
    """Return ``(median_tps_per_user, output_tps_per_gpu, ladder_step)``."""
    pts: List[Point3] = []
    for r in runs:
        if r.get("status") != "success":
            continue
        x = r.get("median_tps_per_user")
        if x is None:
            x = r.get("pareto_x_median_tps_per_user")
        y = r.get("output_tps_per_gpu")
        if y is None:
            y = r.get("pareto_y_output_tps_per_gpu")
        if x is None or y is None:
            continue
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            continue
        pts.append((xf, yf, _step_from_run(r)))
    pts.sort(key=lambda t: t[0])
    return pts


def write_token_pareto_png_from_json_file(
    json_path: Path,
    *,
    curve_label: str = "Pareto Frontier",
    figure_caption: Optional[str] = None,
    png_path: Optional[Path] = None,
) -> Optional[Path]:
    """Load a trace replay JSON report and save a throughput Pareto plot as PNG."""
    json_path = json_path.expanduser().resolve()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs: List[Dict[str, Any]] = data.get("runs") or []
    points = collect_token_pareto_points(runs)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(
            f"WARNING: matplotlib not available ({exc!r}); skip throughput Pareto PNG.",
            file=sys.stderr,
        )
        return None

    if png_path is not None:
        out = png_path.expanduser().resolve()
    else:
        out = json_path.with_name(f"{json_path.stem}_throughput_pareto.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    if not points:
        ax.text(
            0.5,
            0.5,
            "No successful runs with median_tps_per_user and output_tps_per_gpu",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Throughput per User vs Output Throughput per GPU")
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            marker="o",
            color="black",
            linewidth=1.5,
            label=curve_label,
        )
        for xf, yf, step in points:
            ax.annotate(
                str(step),
                xy=(xf, yf),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color="black",
                ha="left",
                va="bottom",
            )
        ax.set_xlabel("median_tps_per_user (tok/s/user)")
        ax.set_ylabel("output_tps_per_gpu (tok/s/gpu)")
        ax.set_title("Throughput per User vs Output Throughput per GPU")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    if figure_caption:
        fig.text(0.5, 0.02, figure_caption, ha="center", fontsize=9)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Plot median_tps_per_user vs output_tps_per_gpu from trace replay JSON."),
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
        "Default: <json_stem>_throughput_pareto.png",
    )
    p.add_argument(
        "--pareto-curve-label",
        type=str,
        default="Pareto Frontier",
        help="Legend label for the Pareto curve.",
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
        written = write_token_pareto_png_from_json_file(
            jp,
            curve_label=args.pareto_curve_label,
            figure_caption=args.figure_caption,
            png_path=png,
        )
        if written is not None:
            print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
