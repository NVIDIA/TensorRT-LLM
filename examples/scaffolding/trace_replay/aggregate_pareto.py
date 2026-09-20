r"""Aggregate a directory of Pareto run JSONs into one CSV, one row per run.

Each row is one Pareto point. The two columns to plot are
``job_x_jobs_per_h_per_user`` (X) and ``job_y_jobs_per_h_per_gpu`` (Y): that
scatter is the job-level Pareto curve this harness exists to produce. The
``token_*`` columns are the token-level curve, kept for comparison.

Runs without a steady-state window are skipped rather than replaced by a
wall-clock average, which would silently average in the ramp-up and the
drain.

Example::

    python examples/scaffolding/trace_replay/aggregate_pareto.py results/ \
        --output-csv results/pareto.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# job_x / job_y lead the metric columns because they are the headline result.
COLUMNS = [
    "run_json",
    "max_batch_size",
    "concurrency",
    "total_sessions",
    "tensor_parallel_size",
    "job_x_jobs_per_h_per_user",
    "job_y_jobs_per_h_per_gpu",
    "token_x_tps_per_user",
    "token_y_tps_per_gpu",
    "token_y_tps_per_gpu_output_only",
    "jobs_completed_in_window",
    "window_duration_s",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path, help="Directory of run JSON files.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <results_dir>/pareto.csv",
    )
    args = parser.parse_args()

    output_csv = args.output_csv or args.results_dir / "pareto.csv"
    rows = []
    skipped = []
    for path in sorted(args.results_dir.glob("*.json")):
        if path == output_csv:
            continue
        report = json.loads(path.read_text())
        if report.get("schema") != "trace_replay_pareto_run/v1":
            continue
        pareto = report.get("pareto") or {}
        if not pareto.get("valid"):
            skipped.append((path.name, pareto.get("reason")))
            continue
        config = report.get("config") or {}
        row = {"run_json": path.name}
        row.update({k: config.get(k) for k in COLUMNS if k in config})
        row.update({k: pareto.get(k) for k in COLUMNS if k in pareto})
        rows.append(row)

    rows.sort(key=lambda r: (r["max_batch_size"] or 0, r["concurrency"] or 0))
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output_csv} ({len(rows)} points)")
    print("job-level Pareto curve (the result: X = jobs/h/user, Y = jobs/h/GPU):")
    for row in rows:
        print(
            f"  B={row['max_batch_size']:<5} C={row['concurrency']:<5} "
            f"{row['job_x_jobs_per_h_per_user']:8.1f} jobs/h/user   "
            f"{row['job_y_jobs_per_h_per_gpu']:8.1f} jobs/h/gpu"
        )
    for name, reason in skipped:
        print(f"skipped {name}: {reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
