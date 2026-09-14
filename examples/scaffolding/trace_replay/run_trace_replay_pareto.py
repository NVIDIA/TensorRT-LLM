r"""Replay one trace concurrently and report steady-state job-level Pareto metrics.

Where ``run_trace_replay.py`` replays a single session, this driver replays
``--total-sessions`` copies of the same trace with at most ``--concurrency`` of
them in flight, which is what a job-level Pareto point measures: many agent
sessions served at once. One run yields one point; sweeping ``--concurrency``
(and the server's ``--max_batch_size``) traces the curve.

The result of a run is the pair ``pareto.job_x_jobs_per_h_per_user`` (agent
tasks one user completes per hour) and ``pareto.job_y_jobs_per_h_per_gpu``
(tasks a GPU serves per hour). Those two are the axes of the job-level Pareto
curve. The ``token_*`` fields alongside them are the conventional token-level
view, reported for comparison, and everything else in the JSON is the raw
material they are derived from.

Example::

    python examples/scaffolding/trace_replay/run_trace_replay_pareto.py \
        /path/to/some.trace.json \
        --model your_model_name \
        --openai-base-url http://127.0.0.1:8000/v1 \
        --total-sessions 200 --concurrency 64 --max-batch-size 64 \
        --tensor-parallel-size 4 --arrival-jitter-s 60

Then aggregate a directory of run JSONs into one CSV with
``aggregate_pareto.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import openai

from tensorrt_llm.scaffolding.trace_replay.execution_trace import ExecutionTrace
from tensorrt_llm.scaffolding.trace_replay.replay import ReplayEngine, ReplayGenerationStats
from tensorrt_llm.scaffolding.worker import TRTOpenaiWorker

LOGGER = logging.getLogger(__name__)

try:
    from .metrics import (
        compute_steady_state_pareto,
        compute_steady_state_window,
        steady_state_excl_count,
        summarize_trace_events,
    )
except ImportError:
    from metrics import (  # type: ignore[no-redef]
        compute_steady_state_pareto,
        compute_steady_state_window,
        steady_state_excl_count,
        summarize_trace_events,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one trace at concurrency and write job-level Pareto metrics JSON.",
    )
    parser.add_argument("trace_json", type=Path, help="Input .trace.json file.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name exposed by the trtllm-serve OpenAI endpoint.",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible endpoint base URL for trtllm-serve.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="OpenAI API key for the endpoint.",
    )
    parser.add_argument(
        "--total-sessions",
        type=int,
        required=True,
        help="Number of sessions replayed in this run. Must exceed 2x the concurrency "
        "for a steady-state window to exist.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        required=True,
        help="Maximum number of sessions in flight at any moment.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        required=True,
        help="The server's --max_batch_size. Recorded as metadata and used to size "
        "the steady-state window.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="TP size the server runs with, used as the GPU count for per-GPU rates.",
    )
    parser.add_argument(
        "--attention-dp-size",
        type=int,
        default=1,
        help="Attention-DP size the server runs with. Above 1, the warmup pins one "
        "request per rank so every rank caches the system prompt.",
    )
    parser.add_argument(
        "--arrival-jitter-s",
        type=float,
        default=0.0,
        help="Stagger session arrivals by U[0, jitter) seconds. Without it, identical "
        "copies of one trace stay phase-aligned and the server sees artificial waves "
        "of prefill and decode. A value near one mean turn duration works well.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the warmup that preloads the trace's system prompt, leaving the "
        "first call of every session to miss the prefix cache.",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=3600.0,
        help="Per-request timeout for the OpenAI client.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Default: <trace_base>_pareto_B<B>_C<C>_<timestamp>.json",
    )
    return parser.parse_args()


def _normalize_base_url(raw: str) -> str:
    url = raw.strip().rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


def _default_output_json(trace_file: Path, args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (
        trace_file.name[: -len(".trace.json")]
        if trace_file.name.endswith(".trace.json")
        else trace_file.stem
    )
    name = f"{base}_pareto_B{args.max_batch_size}_C{args.concurrency}_{stamp}.json"
    return trace_file.parent / name


def _system_prefixes(trace: ExecutionTrace) -> List[Tuple[str, int]]:
    """Return ``(cache_key, token_count)`` for every system message in the trace.

    The cache key mirrors the one :class:`QueueExecutor` uses, so token ids
    pre-generated here are the ones every session sends.
    """
    prefixes: List[Tuple[str, int]] = []
    for event in trace.events:
        if event.event_type != "message" or event.role != "system" or not event.tokens:
            continue
        key = event.system_prompt_id or f"conv:{int(event.conversation_id or 0)}"
        prefixes.append((key, int(event.tokens)))
    return prefixes


async def _warmup(
    client: openai.AsyncOpenAI,
    model: str,
    *,
    system_token_cache: Dict[str, List[int]],
    attention_dp_size: int,
    request_timeout_s: float,
) -> float:
    """Prefill every system prefix once so sessions start on a cache hit.

    Under attention DP the router places requests per rank, so one request is
    pinned to each rank; a single shared prefill would only warm one of them.
    """
    t0 = time.perf_counter()
    for token_ids in system_token_cache.values():
        for rank in range(max(1, attention_dp_size)):
            extra_body: Dict[str, Any] = {"ignore_eos": True}
            if attention_dp_size > 1:
                extra_body["attention_dp_rank"] = rank
                extra_body["attention_dp_relax"] = False
            stream = await client.completions.create(
                model=model,
                prompt=token_ids,
                max_tokens=1,
                stream=True,
                timeout=request_timeout_s,
                extra_body=extra_body,
            )
            async for _ in stream:
                pass
    return time.perf_counter() - t0


async def _one_session(
    worker: TRTOpenaiWorker,
    trace: ExecutionTrace,
    *,
    semaphore: asyncio.Semaphore,
    session_index: int,
    arrival_delay_s: float,
    system_token_cache: Dict[str, List[int]],
) -> Tuple[ReplayGenerationStats, float, float]:
    """Replay the trace once, returning its stats and admission/completion times.

    The jitter sleep happens before the semaphore so a waiting session does not
    occupy a concurrency slot while idling.
    """
    if arrival_delay_s > 0:
        await asyncio.sleep(arrival_delay_s)
    async with semaphore:
        stats = ReplayGenerationStats(session_index=session_index)
        start_s = time.perf_counter()
        await ReplayEngine(
            worker,
            generation_stats=stats,
            system_token_cache=system_token_cache,
        ).launch_trace(trace)
        end_s = time.perf_counter()
        LOGGER.info("session %d done in %.1fs", session_index, end_s - start_s)
        return stats, start_s, end_s


async def _run(args: argparse.Namespace, trace: ExecutionTrace) -> Dict[str, Any]:
    client = openai.AsyncOpenAI(
        base_url=_normalize_base_url(args.openai_base_url),
        api_key=args.openai_api_key,
        timeout=args.request_timeout_s,
    )
    worker = TRTOpenaiWorker(client, model=args.model)

    # Pre-generate the system prefixes so warmup and every session send the
    # same token ids, which is what makes the shared prefix a cache hit.
    rng = random.Random(0)
    system_token_cache: Dict[str, List[int]] = {
        key: [rng.randint(100, 30000) for _ in range(tokens)]
        for key, tokens in _system_prefixes(trace)
    }

    jitter_rng = random.Random(0)
    delays = [
        jitter_rng.uniform(0.0, args.arrival_jitter_s) if args.arrival_jitter_s > 0 else 0.0
        for _ in range(args.total_sessions)
    ]

    try:
        warmup_s = 0.0
        if not args.no_warmup and system_token_cache:
            warmup_s = await _warmup(
                client,
                args.model,
                system_token_cache=system_token_cache,
                attention_dp_size=args.attention_dp_size,
                request_timeout_s=args.request_timeout_s,
            )
            LOGGER.info("warmup done in %.1fs", warmup_s)

        semaphore = asyncio.Semaphore(args.concurrency)
        t0 = time.perf_counter()
        # A run is minutes of work per session, so one failed request must not
        # discard every other session's measurements: collect failures instead
        # and let the surviving sessions produce the metrics.
        settled = await asyncio.gather(
            *[
                _one_session(
                    worker,
                    trace,
                    semaphore=semaphore,
                    session_index=i,
                    arrival_delay_s=delays[i],
                    system_token_cache=system_token_cache,
                )
                for i in range(args.total_sessions)
            ],
            return_exceptions=True,
        )
        wall_clock_s = time.perf_counter() - t0
    finally:
        worker.shutdown()
        await client.close()

    results = [r for r in settled if not isinstance(r, BaseException)]
    failures = [repr(r) for r in settled if isinstance(r, BaseException)]
    if failures:
        LOGGER.error("%d of %d sessions failed", len(failures), args.total_sessions)
        for failure in dict.fromkeys(failures):
            LOGGER.error("  %s", failure)
    if not results:
        raise RuntimeError(f"all {args.total_sessions} sessions failed; first: {failures[0]}")

    # Offsets are relative to the first admission, so every timestamp in the
    # report is comparable across runs.
    session_start_offset_s = [start - t0 for _, start, _ in results]
    session_end_offset_s = [end - t0 for _, _, end in results]
    session_duration_s = [end - start for _, start, end in results]

    replay_detail: List[Dict[str, Any]] = []
    for stats, _, _ in results:
        for entry in stats.entries:
            entry = dict(entry)
            for src, dst in (
                ("client_request_start_s", "client_request_start_offset_s"),
                ("client_request_end_s", "client_request_end_offset_s"),
            ):
                value = entry.pop(src, None)
                entry[dst] = (value - t0) if value is not None else None
            replay_detail.append(entry)

    window = compute_steady_state_window(
        session_start_offset_s=session_start_offset_s,
        session_end_offset_s=session_end_offset_s,
        excl_count=steady_state_excl_count(
            args.concurrency, args.max_batch_size, args.attention_dp_size
        ),
    )
    pareto = compute_steady_state_pareto(
        window=window,
        session_end_offset_s=session_end_offset_s,
        session_duration_mean_s=statistics.mean(session_duration_s),
        replay_detail=replay_detail,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    return {
        "wall_clock_s": wall_clock_s,
        "warmup_wall_s": warmup_s,
        "sessions_completed": len(results),
        "sessions_failed": len(failures),
        "session_failures": sorted(set(failures)),
        "session_start_offset_s": session_start_offset_s,
        "session_end_offset_s": session_end_offset_s,
        "session_duration_s": session_duration_s,
        "session_duration_mean_s": statistics.mean(session_duration_s),
        "replay_assistant_generations_detail": replay_detail,
        "steady_state_window": window,
        "pareto": pareto,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    args = parse_args()
    trace_file = args.trace_json.expanduser().resolve()
    if not trace_file.name.endswith(".trace.json"):
        print(f"error: only .trace.json input is supported, got: {trace_file}", file=sys.stderr)
        raise SystemExit(2)
    # The window excludes excl_count sessions at each end, and excl_count is
    # capped by the server's capacity, so this is the real sizing condition.
    excl_count = steady_state_excl_count(
        args.concurrency, args.max_batch_size, args.attention_dp_size
    )
    if args.total_sessions <= 2 * excl_count:
        LOGGER.warning(
            "total_sessions (%d) <= 2 * excl_count (%d): the run will have no steady-state window",
            args.total_sessions,
            excl_count,
        )
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else _default_output_json(trace_file, args)
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    trace = ExecutionTrace.load(str(trace_file))
    LOGGER.info(
        "replaying trace_id=%s (%d events) with N=%d C=%d B=%d",
        trace.trace_id,
        len(trace.events),
        args.total_sessions,
        args.concurrency,
        args.max_batch_size,
    )
    run = asyncio.run(_run(args, trace))

    report = {
        "schema": "trace_replay_pareto_run/v1",
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "trace_file": str(trace_file),
        "trace_id": trace.trace_id,
        "trace_summary": summarize_trace_events(trace.events),
        "config": {
            "model": args.model,
            "base_url": _normalize_base_url(args.openai_base_url),
            "total_sessions": args.total_sessions,
            "concurrency": args.concurrency,
            "max_batch_size": args.max_batch_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "attention_dp_size": args.attention_dp_size,
            "arrival_jitter_s": args.arrival_jitter_s,
            "warmup": not args.no_warmup,
        },
        **run,
    }
    output_json.write_text(json.dumps(report, indent=2))
    LOGGER.info("wrote %s", output_json)

    pareto = run["pareto"]
    if pareto.get("valid"):
        LOGGER.info(
            "RESULT job-level Pareto point: %.1f jobs/h/user (X), %.1f jobs/h/gpu (Y)",
            pareto["job_x_jobs_per_h_per_user"],
            pareto["job_y_jobs_per_h_per_gpu"],
        )
        LOGGER.info(
            "  token-level, for comparison: %.1f tokens/s/user, %.0f tokens/s/gpu",
            pareto["token_x_tps_per_user"],
            pareto["token_y_tps_per_gpu"],
        )
    else:
        LOGGER.warning("no steady-state window, so no Pareto point: %s", pareto.get("reason"))


if __name__ == "__main__":
    main()
