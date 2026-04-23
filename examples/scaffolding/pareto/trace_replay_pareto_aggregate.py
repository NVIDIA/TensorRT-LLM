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

r"""Merge per-step Pareto JSONs into a combined v4 report + write two PNGs.

Consumes the single-step JSONs produced by ``trace_replay_client.py``
(``schema == "trace_replay_pareto_frontier.step.v4"``) and assembles a
combined ``trace_replay_pareto_frontier.v4`` record compatible with the
existing plot helpers in ``../tracing/``. Does no network I/O and starts
no GPU runtime — purely offline post-processing.

Typical usage (one job, one ladder sweep)::

    python examples/scaffolding/pareto/trace_replay_pareto_aggregate.py \
        --step_jsons out/step8.json out/step16.json out/step32.json \
        --trace_dir .../traces/swebench/django__django-14787 \
        --output_json out/django__django-14787_Qwen3-235B-A22B_tp4_ep4.json

If ``--step_jsons`` is omitted, ``--step_glob`` is used instead. The
combined JSON and two PNG siblings
(``<stem>_throughput_pareto.png`` and ``<stem>_agent_pareto.png``) are
written under the combined JSON's parent directory.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tensorrt_llm.scaffolding.execution_trace import ExecutionTrace

from _common import (
    atomic_write_json,
    count_assistant_completion_tokens,
    count_parallel_regions,
    collect_trace_file_stats,
    find_compact_trace_file,
    pareto_config_filename_suffix,
    read_json,
    summarize_trace_events,
)

COMBINED_SCHEMA = "trace_replay_pareto_frontier.v4"
EXPECTED_STEP_SCHEMA = "trace_replay_pareto_frontier.step.v4"


# ---------------------------------------------------------------------------
# Plot helper loader: aggregator lives in .../scaffolding/pareto/ and the
# plot helpers live in the sibling .../scaffolding/tracing/. Load them
# dynamically so we don't have to turn tracing/ into a package or copy
# code.
# ---------------------------------------------------------------------------


def _load_plot_helper(module_stem: str, symbol_name: str) -> Optional[Callable]:
    module_path = (
        Path(__file__).resolve().parent.parent / "tracing" / f"{module_stem}.py"
    )
    if not module_path.exists():
        print(
            f"WARNING: plot helper not found at {module_path}; "
            f"skipping {symbol_name}.",
            file=sys.stderr,
        )
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_stem, str(module_path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        symbol = getattr(module, symbol_name, None)
        return symbol if callable(symbol) else None
    except Exception as exc:
        print(
            f"WARNING: failed to load {module_stem}.{symbol_name}: {exc!r}",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Step-JSON ingestion
# ---------------------------------------------------------------------------


def _resolve_step_paths(args: argparse.Namespace) -> List[Path]:
    if args.step_jsons:
        paths = [Path(p).expanduser().resolve() for p in args.step_jsons]
    elif args.step_glob:
        matches = sorted(glob.glob(args.step_glob))
        if not matches:
            raise FileNotFoundError(
                f"No step JSONs matched --step_glob={args.step_glob!r}"
            )
        paths = [Path(p).expanduser().resolve() for p in matches]
    else:
        raise ValueError("Must pass --step_jsons or --step_glob")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing step JSONs: {missing}")
    return paths


def _load_step_records(paths: List[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for p in paths:
        record = read_json(p)
        schema = record.get("schema")
        if schema != EXPECTED_STEP_SCHEMA:
            print(
                f"WARNING: {p} has schema={schema!r}, expected "
                f"{EXPECTED_STEP_SCHEMA!r}; continuing anyway.",
                file=sys.stderr,
            )
        if "run_row" not in record:
            raise ValueError(f"{p} is missing the required 'run_row' field")
        records.append(record)
    # Deterministic ordering: ladder_index first, fall back to ladder_step.
    def _sort_key(rec: Dict[str, Any]) -> tuple:
        row = rec.get("run_row", {})
        return (int(row.get("ladder_index", 0)), int(row.get("ladder_step", 0)))

    records.sort(key=_sort_key)
    return records


def _first_nonempty(records: List[Dict[str, Any]], field: str) -> Optional[Any]:
    for rec in records:
        val = rec.get(field)
        if val:
            return val
    return None


# ---------------------------------------------------------------------------
# Trace metadata (fresh read of the trace file, so we do not inherit stale
# summaries from step JSONs — cheap enough since it is metadata only).
# ---------------------------------------------------------------------------


def _trace_meta(trace_dir: Path) -> Dict[str, Any]:
    trace_path = find_compact_trace_file(trace_dir)
    trace = ExecutionTrace.load(str(trace_path))
    return {
        "trace_dir": str(trace_dir),
        "trace_file": str(trace_path),
        "trace_id": trace.trace_id,
        "num_events": len(trace.events),
        "parallel_region_counts": count_parallel_regions(trace.events),
        "assistant_output_tokens_sum": count_assistant_completion_tokens(trace.events),
        **summarize_trace_events(trace.events),
        **collect_trace_file_stats(trace_path),
    }


# ---------------------------------------------------------------------------
# Combined v4 record assembly
# ---------------------------------------------------------------------------


def _assemble_combined_record(
    *,
    args: argparse.Namespace,
    step_paths: List[Path],
    step_records: List[Dict[str, Any]],
    trace_meta: Dict[str, Any],
    output_json: Path,
) -> Dict[str, Any]:
    first = step_records[0]

    # Detect OOM: first row with error_kind=="out_of_memory" marks the
    # stop point; subsequent rows that lack a run should be tagged
    # skipped_after_prior_oom so plot helpers ignore them cleanly.
    runs: List[Dict[str, Any]] = []
    oom_at_index: Optional[int] = None
    oom_at_step: Optional[int] = None
    oom_error: Optional[str] = None

    for rec in step_records:
        row = dict(rec["run_row"])
        runs.append(row)
        if (
            oom_at_index is None
            and row.get("error_kind") == "out_of_memory"
        ):
            oom_at_index = row.get("ladder_index")
            oom_at_step = row.get("ladder_step")
            oom_error = row.get("error")

    if oom_at_index is not None:
        # Mark every run strictly after the first OOM as skipped_after_prior_oom
        # (unless the client already recorded success there — unlikely but
        # preserve that signal if present).
        for row in runs:
            idx = row.get("ladder_index")
            if (
                isinstance(idx, int)
                and idx > oom_at_index
                and row.get("status") != "success"
            ):
                row["status"] = "skipped_after_prior_oom"
                row.setdefault("error_kind", "out_of_memory_inherited")
                row["prior_oom_ladder_index"] = oom_at_index
                row["prior_oom_ladder_step"] = oom_at_step
                row["skipped_reason"] = (
                    "Larger ladder values were not run after out-of-memory at "
                    f"ladder_index={oom_at_index}, ladder_step={oom_at_step}."
                )

    ladder = [int(r.get("ladder_step")) for r in runs if r.get("ladder_step") is not None]

    # Prefer metadata from the first step record that actually has it —
    # synthetic "skipped" / "failed" step JSONs carry only minimal fields,
    # which would otherwise poison the combined report's top-level metadata.
    metadata_cli_args = _first_nonempty(step_records, "cli_args") or {}
    metadata_host = _first_nonempty(step_records, "host") or {}
    metadata_base_url = _first_nonempty(step_records, "base_url")
    metadata_model = _first_nonempty(step_records, "model")

    model_name = (
        args.model_name
        or (Path(metadata_model).name if metadata_model else None)
        or "model"
    )
    filename_suffix = pareto_config_filename_suffix(
        args.model_name or metadata_model,
        args.tensor_parallel_size,
        args.moe_expert_parallel_size,
    )

    record: Dict[str, Any] = {
        "schema": COMBINED_SCHEMA,
        "run_started_at_utc": first.get("run_started_at_utc"),
        "run_finished_at_utc": step_records[-1].get(
            "run_finished_at_utc",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        ),
        "aggregated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "aggregator_argv": sys.argv,
        "aggregator_args": {
            "step_jsons": [str(p) for p in step_paths],
            "trace_dir": str(args.trace_dir) if args.trace_dir else None,
            "model_name": args.model_name,
            "tensor_parallel_size": args.tensor_parallel_size,
            "moe_expert_parallel_size": args.moe_expert_parallel_size,
            "output_json": str(output_json),
            "pareto_curve_label": args.pareto_curve_label,
            "no_pareto_png": args.no_pareto_png,
        },
        "backend": "trtllm-serve",
        "artifact_naming": {
            "model_name": model_name,
            "tensor_parallel_size": args.tensor_parallel_size,
            "moe_expert_parallel_size": args.moe_expert_parallel_size,
            "enable_attention_dp": False,
            "filename_suffix": filename_suffix,
        },
        # Plot helpers use cli_args.tensor_parallel_size and
        # host.cuda_device_count as fallbacks for per-GPU normalization;
        # inject canonical values on top of whatever the client recorded.
        "cli_args": {
            **metadata_cli_args,
            "tensor_parallel_size": args.tensor_parallel_size,
            "moe_expert_parallel_size": args.moe_expert_parallel_size,
        },
        "host": metadata_host,
        "trace_meta": trace_meta,
        "llm_fixed_config": {
            "backend": "trtllm-serve",
            "tensor_parallel_size": args.tensor_parallel_size,
            "moe_expert_parallel_size": args.moe_expert_parallel_size,
        },
        "trtllm_serve_reference": {
            "base_url": metadata_base_url,
            "model": metadata_model,
        },
        "experiment": {
            "ladder": ladder,
            "ladder_len": len(ladder),
            "design": (
                "one ladder step per run; each row records its own "
                "(ladder_step, max_batch_size, total_sessions, concurrency) tuple; "
                "trtllm-serve is restarted between steps with the row's max_batch_size"
            ),
        },
        "metrics_notes": {
            "throughput_basis": "output_tps_* / median_tps_per_user use "
            "total_output_tokens_replay_actual / wall_clock_s (per step)",
            "output_tps_aggregate": "total_output_tokens_replay_actual / wall_clock_s",
            "output_tps_per_gpu": "output_tps_aggregate / tensor_parallel_size",
            "mean_tps_per_user_session_time": "total_output_tokens_replay_actual / sum(session durations)",
        },
        "step_artifacts": [str(p) for p in step_paths],
        "runs": runs,
    }

    if oom_at_index is not None:
        record["pareto_ladder_oom"] = {
            "stopped_after_oom": True,
            "first_oom_ladder_index": oom_at_index,
            "first_oom_ladder_step": oom_at_step,
            "error": oom_error,
        }

    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate per-step trace replay JSONs into a combined Pareto "
            "frontier report and write throughput + agent Pareto PNGs."
        )
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--step_jsons",
        type=str,
        nargs="+",
        help="Explicit list of per-step JSON files (will be sorted by ladder_index).",
    )
    group.add_argument(
        "--step_glob",
        type=str,
        default=None,
        help="Glob pattern to collect per-step JSON files (e.g. 'out/step*.json').",
    )
    p.add_argument(
        "--trace_dir",
        type=Path,
        required=True,
        help=(
            "Trace directory (same as the one passed to the client); trace_meta "
            "is recomputed from the *.trace.json here for a clean report."
        ),
    )
    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "Model name / checkpoint path for artifact_naming. Default: the "
            "model recorded in the first step JSON."
        ),
    )
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size the server was run with (for per-GPU normalization).",
    )
    p.add_argument(
        "--moe_expert_parallel_size",
        type=int,
        default=4,
        help="MoE expert parallel size the server was run with.",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Path for the combined trace_replay_pareto_frontier.v4 JSON report.",
    )
    p.add_argument(
        "--pareto_curve_label",
        type=str,
        default="Pareto Frontier",
        help="Legend label for the throughput Pareto PNG (successful runs only).",
    )
    p.add_argument(
        "--no_pareto_png",
        action="store_true",
        default=False,
        help="Skip writing the two Pareto PNGs (JSON still written).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    step_paths = _resolve_step_paths(args)
    step_records = _load_step_records(step_paths)
    if not step_records:
        print("error: no step records loaded", file=sys.stderr)
        return 2

    trace_dir = args.trace_dir.expanduser().resolve()
    trace_meta = _trace_meta(trace_dir)

    output_json = args.output_json.expanduser().resolve()
    record = _assemble_combined_record(
        args=args,
        step_paths=step_paths,
        step_records=step_records,
        trace_meta=trace_meta,
        output_json=output_json,
    )
    atomic_write_json(output_json, record)
    print(f"Wrote {output_json}")

    if args.no_pareto_png:
        return 0

    write_token = _load_plot_helper(
        "plot_trace_replay_token_pareto",
        "write_token_pareto_png_from_json_file",
    )
    if write_token is not None:
        try:
            png = write_token(
                output_json,
                curve_label=args.pareto_curve_label,
                figure_caption=None,
                png_path=None,
            )
            if png is not None:
                print(f"Wrote {png}")
        except Exception as exc:
            print(f"WARNING: throughput Pareto PNG failed: {exc!r}", file=sys.stderr)

    write_agent = _load_plot_helper(
        "plot_trace_replay_agent_pareto",
        "write_agent_pareto_png_from_json_file",
    )
    if write_agent is not None:
        try:
            png = write_agent(
                output_json,
                figure_caption=None,
                png_path=None,
            )
            if png is not None:
                print(f"Wrote {png}")
        except Exception as exc:
            print(f"WARNING: agent Pareto PNG failed: {exc!r}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
