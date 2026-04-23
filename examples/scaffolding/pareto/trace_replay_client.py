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

r"""Single-ladder-step trace replay against an external ``trtllm-serve``.

Pure OpenAI client. Expects ``trtllm-serve`` (or any OpenAI-compatible
completions endpoint that honors token-id prompts and ``ignore_eos``) to be
already running at ``--base_url`` with the desired ``max_batch_size``, TP,
EP, etc. Submits ``--total_sessions`` :class:`ReplayEngine` tasks with at most
``--concurrency`` of them in flight at any time (the remaining sessions
wait on an :class:`asyncio.Semaphore`), measures per-session and aggregate
throughput, and writes a single **step JSON** summarizing this ladder point.

The three load knobs are fully decoupled, matching the pattern used by
``examples/scaffolding/benchmarks/`` (see ``__main__.py`` /
``chat_benchmark.py``):

* ``--max_batch_size``: the server's ``trtllm-serve --max_batch_size`` for
  this step (metadata only, the actual cap is enforced server-side).
* ``--total_sessions``: total number of trace replays to execute (total work).
* ``--concurrency``: maximum number of replays in flight at any moment
  (load shaping, equivalent to ``chat_concurrency`` in the benchmarks).

Server lifecycle (start trtllm-serve with ``--max_batch_size``, poll
``/health``, kill after the client exits) lives in the Slurm driver script,
not here.

Example::

    python examples/scaffolding/pareto/trace_replay_client.py \
        --base_url http://127.0.0.1:8000/v1 \
        --model /path/to/Qwen3-235B-A22B \
        --trace_dir .../traces/swebench/django__django-14787 \
        --total_sessions 32 --concurrency 16 --max_batch_size 16 \
        --ladder_index 1 --ladder_step 16 \
        --tensor_parallel_size 4 --moe_expert_parallel_size 4 \
        --output_json .../step16.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import socket
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import TRTOpenaiWorker
from tensorrt_llm.scaffolding.execution_trace import ExecutionTrace
from tensorrt_llm.scaffolding.replay import ReplayEngine, ReplayGenerationStats

from _common import (
    TeeTextIO,
    args_to_dict,
    atomic_write_json,
    collect_host_info,
    collect_trace_file_stats,
    count_assistant_completion_tokens,
    count_parallel_regions,
    find_compact_trace_file,
    is_oom_exception,
    pareto_config_filename_suffix,
    percentile,
    summarize_trace_events,
)

# Schema version for the single-step JSON written by this client. The
# aggregator expects this identifier and concatenates ``runs[0]`` from N
# step JSONs into the combined ``trace_replay_pareto_frontier.v4`` record.
STEP_SCHEMA = "trace_replay_pareto_frontier.step.v4"


# ---------------------------------------------------------------------------
# One concurrent session (one full trace replay)
# ---------------------------------------------------------------------------


async def _one_session(
    worker: TRTOpenaiWorker,
    trace: ExecutionTrace,
    *,
    semaphore: asyncio.Semaphore,
    session_index: int,
    total_sessions: int,
    concurrency: int,
    max_batch_size: int,
    ladder_step: int,
) -> Tuple[float, ReplayGenerationStats]:
    label = (
        f"[step={ladder_step} N={total_sessions} C={concurrency} B={max_batch_size}] "
        f"session {session_index + 1}/{total_sessions}"
    )
    async with semaphore:
        print(f"{label}: replay start", flush=True)
        stats = ReplayGenerationStats()
        t0 = time.perf_counter()
        await ReplayEngine(worker, generation_stats=stats).launch_trace(trace)
        elapsed = time.perf_counter() - t0
        print(f"{label}: replay done in {elapsed:.3f}s", flush=True)
        return elapsed, stats


# ---------------------------------------------------------------------------
# Throughput / latency accounting (backend-agnostic)
# ---------------------------------------------------------------------------


def compute_run_row(
    *,
    trace: ExecutionTrace,
    results: List[Tuple[float, ReplayGenerationStats]],
    wall_s: float,
    total_sessions: int,
    concurrency: int,
    max_batch_size: int,
    ladder_index: int,
    ladder_step: int,
    tensor_parallel_size: int,
) -> Dict[str, Any]:
    """Build one ``runs[]`` row from successful per-session results.

    The three load dimensions (``total_sessions``, ``concurrency``,
    ``max_batch_size``) are recorded independently so plot helpers can
    distinguish a step that stacked more total work at the same in-flight
    concurrency from one that raised the concurrency itself. Field names
    and formulas otherwise match ``trace_replay_pareto_frontier.v3``
    (produced by the older in-process script), so existing plot helpers
    keep working unchanged.
    """
    durations = [r[0] for r in results]
    stats_list = [r[1] for r in results]

    tokens_per_trace_trace_metadata = count_assistant_completion_tokens(trace.events)
    total_out_tokens_trace_metadata = float(total_sessions * tokens_per_trace_trace_metadata)

    per_session_replay_output = [s.sum_replay_output_tokens() for s in stats_list]
    per_session_trace_completion = [s.sum_trace_completion_tokens() for s in stats_list]
    total_out_tokens_replay_actual = float(sum(per_session_replay_output))

    detail_session0 = stats_list[0].entries if stats_list else []

    tp_sizes = [
        per_session_replay_output[i] / durations[i]
        for i in range(len(durations))
        if durations[i] > 0
    ]

    row: Dict[str, Any] = {
        "ladder_index": ladder_index,
        "ladder_step": ladder_step,
        "total_sessions": total_sessions,
        "concurrency": concurrency,
        "max_batch_size": max_batch_size,
        "status": "success",
        "error": None,
        "error_traceback": None,
        "wall_clock_s": wall_s,
        # From trace file ``completion_tokens`` (original recording).
        "assistant_output_tokens_per_trace": tokens_per_trace_trace_metadata,
        "total_output_tokens_trace_metadata": total_out_tokens_trace_metadata,
        # Backward-compatible alias.
        "total_output_tokens_estimated": total_out_tokens_trace_metadata,
        # Measured during this replay (decoder output token ids per session).
        "per_session_replay_output_token_sum": per_session_replay_output,
        "per_session_total_output_tokens": list(per_session_replay_output),
        "per_session_trace_completion_token_sum": per_session_trace_completion,
        "assistant_output_tokens_per_trace_replay_actual_mean": (
            statistics.mean(per_session_replay_output) if per_session_replay_output else None
        ),
        "total_output_tokens_replay_actual": total_out_tokens_replay_actual,
        "replay_assistant_generations_detail_session0": detail_session0,
        "session_duration_s": durations,
        "session_duration_min_s": min(durations) if durations else None,
        "session_duration_max_s": max(durations) if durations else None,
        "session_duration_sum_s": sum(durations) if durations else None,
        "session_duration_stdev_s": (
            statistics.stdev(durations) if len(durations) > 1 else 0.0
        ),
        "session_duration_p50_s": statistics.median(durations) if durations else None,
        "session_duration_p90_s": percentile(durations, 0.9) if durations else None,
        "session_duration_p99_s": percentile(durations, 0.99) if durations else None,
        "session_duration_mean_s": statistics.mean(durations) if durations else None,
        "session_duration_cv": (
            (statistics.stdev(durations) / statistics.mean(durations))
            if len(durations) > 1 and statistics.mean(durations) > 0
            else None
        ),
        "aggregate_latency_person_s": sum(durations) if durations else None,
        "median_tps_per_user": statistics.median(tp_sizes) if tp_sizes else None,
        "mean_tps_per_user": statistics.mean(tp_sizes) if tp_sizes else None,
        "min_tps_per_user": min(tp_sizes) if tp_sizes else None,
        "max_tps_per_user": max(tp_sizes) if tp_sizes else None,
        "output_tps_aggregate": (total_out_tokens_replay_actual / wall_s) if wall_s > 0 else None,
        "output_tokens_per_wall_s_per_session_mean": (
            (total_out_tokens_replay_actual / wall_s / total_sessions)
            if wall_s > 0 and total_sessions
            else None
        ),
        "mean_tps_per_user_session_time": (
            (total_out_tokens_replay_actual / sum(durations))
            if durations and sum(durations) > 0
            else None
        ),
    }

    tp = int(tensor_parallel_size or 0)
    agg = row["output_tps_aggregate"]
    row["output_tps_per_gpu"] = (agg / tp) if (agg is not None and tp > 0) else None
    row["pareto_x_median_tps_per_user"] = row.get("median_tps_per_user")
    row["pareto_y_output_tps_per_gpu"] = row.get("output_tps_per_gpu")
    row["output_tps_per_aggregate_1gpu_equiv"] = agg

    try:
        import torch

        if torch.cuda.is_available():
            ng = torch.cuda.device_count()
            row["output_tps_per_cuda_device_count"] = (
                (agg / ng) if (agg is not None and ng > 0) else None
            )
    except Exception:
        pass

    return row


def _failed_run_row(
    *,
    exc: BaseException,
    total_sessions: int,
    concurrency: int,
    max_batch_size: int,
    ladder_index: int,
    ladder_step: int,
    wall_s: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "ladder_index": ladder_index,
        "ladder_step": ladder_step,
        "total_sessions": total_sessions,
        "concurrency": concurrency,
        "max_batch_size": max_batch_size,
        "status": "failed",
        "error": repr(exc),
        "error_traceback": traceback.format_exception_only(type(exc), exc),
        "wall_clock_s": wall_s,
    }
    if is_oom_exception(exc):
        row["error_kind"] = "out_of_memory"
    return row


# ---------------------------------------------------------------------------
# Banner / startup diagnostics
# ---------------------------------------------------------------------------


def _print_banner(
    *,
    args: argparse.Namespace,
    trace_dir: Path,
    trace_path: Path,
    output_json: Path,
    output_log: Path,
) -> None:
    width = 78
    bar = "#" * width
    rule = "=" * width
    cli = args_to_dict(args)
    print("", flush=True)
    print(bar, flush=True)
    print(
        "#" + " TRACE REPLAY CLIENT (single ladder step) ".center(width - 2) + "#",
        flush=True,
    )
    print(bar, flush=True)
    print(f"  hostname          : {socket.gethostname()}", flush=True)
    print(f"  cwd               : {os.getcwd()}", flush=True)
    print(f"  command           : {shlex.join(sys.argv)}", flush=True)
    print(rule, flush=True)
    print(f"  base_url          : {args.base_url}", flush=True)
    print(f"  model             : {args.model}", flush=True)
    print(f"  trace_dir         : {trace_dir}", flush=True)
    print(f"  trace_file        : {trace_path}", flush=True)
    print(f"  output_json       : {output_json}", flush=True)
    print(f"  output_log        : {output_log}", flush=True)
    print(f"  ladder_index      : {args.ladder_index}", flush=True)
    print(f"  ladder_step       : {args.ladder_step}", flush=True)
    print(f"  total_sessions (N): {args.total_sessions}   # total trace replays", flush=True)
    print(f"  concurrency    (C): {args.concurrency}   # max in-flight replays", flush=True)
    print(f"  max_batch_size (B): {args.max_batch_size}   # server scheduler cap", flush=True)
    print(f"  tensor_parallel   : {args.tensor_parallel_size}", flush=True)
    print(f"  moe_expert_parallel: {args.moe_expert_parallel_size}", flush=True)
    print(rule, flush=True)
    for key in sorted(cli.keys()):
        print(f"  {key:28s} = {cli[key]!r}", flush=True)
    print(bar + "\n", flush=True)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


async def run_client(args: argparse.Namespace) -> int:
    trace_dir = args.trace_dir.expanduser().resolve()
    trace_path = find_compact_trace_file(trace_dir)
    trace = ExecutionTrace.load(str(trace_path))

    client = AsyncOpenAI(
        api_key=args.openai_api_key,
        base_url=args.base_url,
        timeout=args.request_timeout_s,
    )
    worker = TRTOpenaiWorker(client, args.model, kv_cache_hint_enabled=False)

    total_sessions = int(args.total_sessions)
    concurrency = int(args.concurrency)
    max_batch_size = int(args.max_batch_size)
    ladder_index = int(args.ladder_index)
    ladder_step = int(args.ladder_step)
    if total_sessions <= 0:
        raise ValueError(f"--total_sessions must be > 0 (got {total_sessions})")
    if concurrency <= 0:
        raise ValueError(f"--concurrency must be > 0 (got {concurrency})")
    if max_batch_size <= 0:
        raise ValueError(f"--max_batch_size must be > 0 (got {max_batch_size})")

    semaphore = asyncio.Semaphore(concurrency)
    wall_t0 = time.perf_counter()
    try:
        results = await asyncio.gather(
            *[
                _one_session(
                    worker,
                    trace,
                    semaphore=semaphore,
                    session_index=i,
                    total_sessions=total_sessions,
                    concurrency=concurrency,
                    max_batch_size=max_batch_size,
                    ladder_step=ladder_step,
                )
                for i in range(total_sessions)
            ]
        )
        wall_s = time.perf_counter() - wall_t0
        run_row = compute_run_row(
            trace=trace,
            results=results,
            wall_s=wall_s,
            total_sessions=total_sessions,
            concurrency=concurrency,
            max_batch_size=max_batch_size,
            ladder_index=ladder_index,
            ladder_step=ladder_step,
            tensor_parallel_size=int(args.tensor_parallel_size),
        )
        exit_code = 0
    except Exception as exc:
        wall_s = time.perf_counter() - wall_t0
        print(f"Replay failed: {exc!r}", file=sys.stderr)
        traceback.print_exc()
        run_row = _failed_run_row(
            exc=exc,
            total_sessions=total_sessions,
            concurrency=concurrency,
            max_batch_size=max_batch_size,
            ladder_index=ladder_index,
            ladder_step=ladder_step,
            wall_s=wall_s,
        )
        exit_code = 1

    run_row["llm_effective_config"] = {
        "backend": "trtllm-serve",
        "base_url": args.base_url,
        "model": args.model,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "moe_expert_parallel_size": int(args.moe_expert_parallel_size),
        "max_batch_size": max_batch_size,
        "total_sessions": total_sessions,
        "concurrency": concurrency,
        "ladder_index": ladder_index,
        "ladder_step": ladder_step,
    }

    # Self-describing step JSON: embeds enough metadata that the aggregator
    # can reconstruct a combined v4 report without re-reading the trace.
    step_record: Dict[str, Any] = {
        "schema": STEP_SCHEMA,
        "run_started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cli_argv": sys.argv,
        "cli_args": args_to_dict(args),
        "base_url": args.base_url,
        "model": args.model,
        "artifact_naming": {
            "model_name": Path(os.path.expanduser(args.model or "")).name or "model",
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "moe_expert_parallel_size": int(args.moe_expert_parallel_size),
            "enable_attention_dp": False,
            "filename_suffix": pareto_config_filename_suffix(
                args.model,
                int(args.tensor_parallel_size),
                int(args.moe_expert_parallel_size),
            ),
        },
        "trace_dir": str(trace_dir),
        "trace_file": str(trace_path),
        "host": collect_host_info(),
        "trace_meta": {
            "trace_id": trace.trace_id,
            "num_events": len(trace.events),
            "parallel_region_counts": count_parallel_regions(trace.events),
            "assistant_output_tokens_sum": count_assistant_completion_tokens(trace.events),
            **summarize_trace_events(trace.events),
            **collect_trace_file_stats(trace_path),
        },
        "run_row": run_row,
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    output_json = args.output_json.expanduser().resolve()
    atomic_write_json(output_json, step_record)
    print(f"Wrote {output_json}", flush=True)
    return exit_code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pareto trace replay against an external trtllm-serve (single "
            "ladder step)."
        )
    )
    p.add_argument(
        "--base_url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL exposed by trtllm-serve.",
    )
    p.add_argument(
        "--openai_api_key",
        type=str,
        default="tensorrt_llm",
        help="Placeholder API key (trtllm-serve does not authenticate).",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model identifier as exposed by trtllm-serve (usually the "
            "--model argument that was passed to trtllm-serve, i.e. a "
            "checkpoint directory path)."
        ),
    )
    p.add_argument(
        "--trace_dir",
        type=Path,
        required=True,
        help="Directory containing a *.trace.json (compact execution trace).",
    )
    p.add_argument(
        "--total_sessions",
        type=int,
        required=True,
        help=(
            "Total number of trace replays to execute at this ladder step "
            "(each session replays the full compact trace exactly once)."
        ),
    )
    p.add_argument(
        "--concurrency",
        type=int,
        required=True,
        help=(
            "Maximum number of trace replays in flight at any moment "
            "(an asyncio.Semaphore gates admission). When concurrency == "
            "total_sessions every session starts at t=0; when concurrency < "
            "total_sessions the remaining sessions wait their turn."
        ),
    )
    p.add_argument(
        "--max_batch_size",
        type=int,
        required=True,
        help=(
            "trtllm-serve --max_batch_size this step was started with; "
            "recorded in the step JSON as metadata. The actual capacity is "
            "enforced server-side and is independent of --concurrency and "
            "--total_sessions."
        ),
    )
    p.add_argument(
        "--ladder_index",
        type=int,
        default=0,
        help="Zero-based index of this ladder step within the full ladder.",
    )
    p.add_argument(
        "--ladder_step",
        type=int,
        required=True,
        help=(
            "Ladder identifier for this step (recorded verbatim and used for "
            "sorting/annotation by the aggregator and plot helpers). Not "
            "tied to any of --total_sessions / --concurrency / --max_batch_size."
        ),
    )
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="TP size the server is running with (metadata, used for per-GPU normalization).",
    )
    p.add_argument(
        "--moe_expert_parallel_size",
        type=int,
        default=4,
        help="MoE expert-parallel size the server is running with (metadata).",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Path for the single-step JSON output.",
    )
    p.add_argument(
        "--request_timeout_s",
        type=float,
        default=3600.0,
        help="Per-HTTP-request timeout for the OpenAI client.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json.expanduser().resolve()
    output_log = output_json.with_suffix(".log")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(output_log, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeTextIO(original_stdout, log_fp)
    sys.stderr = TeeTextIO(original_stderr, log_fp)
    try:
        trace_dir = args.trace_dir.expanduser().resolve()
        trace_path = find_compact_trace_file(trace_dir)
        _print_banner(
            args=args,
            trace_dir=trace_dir,
            trace_path=trace_path,
            output_json=output_json,
            output_log=output_log,
        )
        exit_code = asyncio.run(run_client(args))
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fp.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
