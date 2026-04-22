r"""Replay one ``.trace.json`` file via TRTOpenaiWorker and write replay metrics JSON.

Example::

    python examples/scaffolding/trace_replay/run_trace_replay.py \
        /path/to/some.trace.json \
        --model your_model_name \
        --openai-base-url http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import openai

from tensorrt_llm.scaffolding.execution_trace import ExecutionTrace
from tensorrt_llm.scaffolding.replay import ReplayEngine, ReplayGenerationStats
from tensorrt_llm.scaffolding.worker import TRTOpenaiWorker

LOGGER = logging.getLogger(__name__)

try:
    from .metrics import (
        collect_trace_file_stats,
        compute_replay_run_metrics,
        count_assistant_completion_tokens,
        count_parallel_regions,
        summarize_trace_events,
    )
except ImportError:
    from metrics import (  # type: ignore[no-redef]
        collect_trace_file_stats,
        compute_replay_run_metrics,
        count_assistant_completion_tokens,
        count_parallel_regions,
        summarize_trace_events,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one trace replay through TRTOpenaiWorker and save metrics JSON.",
    )
    parser.add_argument(
        "trace_json",
        type=Path,
        help="Input .trace.json file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name exposed by trtllm-serve OpenAI endpoint.",
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
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="TP size for per-GPU throughput normalization in output metrics.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Default: <trace_base>_<model>_replay_statistics_<timestamp>.json",
    )
    return parser.parse_args()


def _normalize_base_url(raw: str) -> str:
    url = raw.strip().rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def _default_output_json(trace_file: Path, model: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model).strip("_")
    safe_model = safe_model or "model"
    trace_base = (
        trace_file.name[: -len(".trace.json")]
        if trace_file.name.endswith(".trace.json")
        else trace_file.stem
    )
    return trace_file.parent / f"{trace_base}_{safe_model}_replay_statistics_{stamp}.json"


async def _run_one_replay(
    worker: TRTOpenaiWorker,
    trace: ExecutionTrace,
) -> Dict[str, Any]:
    stats = ReplayGenerationStats()
    LOGGER.info("Starting replay for trace_id=%s with %d events", trace.trace_id, len(trace.events))
    t0 = time.perf_counter()
    await ReplayEngine(worker, generation_stats=stats).launch_trace(trace)
    elapsed = time.perf_counter() - t0
    LOGGER.info("Replay finished in %.3fs", elapsed)
    return {
        "elapsed_s": elapsed,
        "replay_output_token_sum": stats.sum_replay_output_tokens(),
        "trace_completion_token_sum": stats.sum_trace_completion_tokens(),
        "per_assistant_generation_detail": stats.entries,
    }


async def _run_replay_with_client(
    trace: ExecutionTrace,
    model: str,
    base_url: str,
    api_key: str,
) -> Dict[str, Any]:
    client = openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    worker = TRTOpenaiWorker(client, model=model)
    try:
        return await _run_one_replay(worker, trace)
    finally:
        worker.shutdown()
        await _safe_close_client(client)


async def _safe_close_client(client: openai.AsyncOpenAI) -> None:
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        maybe_coro = close_fn()
        if asyncio.iscoroutine(maybe_coro):
            await maybe_coro


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    trace_file = args.trace_json.expanduser().resolve()
    if not trace_file.name.endswith(".trace.json"):
        print(
            f"error: only .trace.json input is supported, got: {trace_file}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else _default_output_json(trace_file, args.model).resolve()
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading trace file: %s", trace_file)
    trace = ExecutionTrace.load(str(trace_file))
    LOGGER.info("Loaded trace_id=%s, events=%d", trace.trace_id, len(trace.events))
    base_url = _normalize_base_url(args.openai_base_url)
    LOGGER.info("Using OpenAI endpoint: %s", base_url)
    LOGGER.info("Replay model: %s", args.model)
    replay_result = asyncio.run(
        _run_replay_with_client(
            trace=trace,
            model=args.model,
            base_url=base_url,
            api_key=args.openai_api_key,
        )
    )

    run_metrics = compute_replay_run_metrics(
        trace=trace,
        n_sessions=1,
        wall_clock_s=float(replay_result["elapsed_s"]),
        session_duration_s=[float(replay_result["elapsed_s"])],
        replay_output_token_sum_by_session=[int(replay_result["replay_output_token_sum"])],
        trace_completion_token_sum_by_session=[int(replay_result["trace_completion_token_sum"])],
        replay_detail_session0=replay_result["per_assistant_generation_detail"],
        tensor_parallel_size=args.tensor_parallel_size,
    )

    record: Dict[str, Any] = {
        "schema": "trace_replay.single.v1",
        "run_started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "host": {
            "hostname": socket.gethostname(),
            "cwd": os.getcwd(),
            "python_version": sys.version,
            "pid": os.getpid(),
        },
        "cli_argv": sys.argv,
        "trace_file": str(trace_file),
        "openai": {
            "base_url": base_url,
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "trace_meta": {
            "trace_id": trace.trace_id,
            "num_events": len(trace.events),
            "parallel_region_counts": count_parallel_regions(trace.events),
            "assistant_output_tokens_sum": count_assistant_completion_tokens(trace.events),
            **summarize_trace_events(trace.events),
            **collect_trace_file_stats(trace_file),
        },
        "run": run_metrics,
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(record, fp, indent=2, ensure_ascii=False, default=str)

    LOGGER.info("Replay statistics written to: %s", output_json)
    print(output_json)


if __name__ == "__main__":
    main()
