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

r"""Run one Tree-of-Thought research prompt.

Start the MCP servers and an OpenAI-compatible TensorRT-LLM endpoint, then run::

    python examples/scaffolding/contrib/tree_of_thought_research/run_tot_research.py \
        --config examples/scaffolding/contrib/iter_research/config.yaml --enable_tracing
"""

from __future__ import annotations

import argparse
import asyncio
import io
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import MCPWorker, TaskMetricsCollector, TRTOpenaiWorker
from tensorrt_llm.scaffolding.contrib.tree_of_thought_research import (
    create_tot_research_scaffolding_llm,
)

_MCP_SERVICE_ORDER = (
    ("tavily_search", 8087),
    ("fetch_webpage", 8085),
    ("python_interpreter", 8086),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tree-of-Thought research on one prompt.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--num_thoughts_per_step", type=int, default=None)
    parser.add_argument("--branch_factor", type=int, default=None)
    parser.add_argument("--complete_score_threshold", type=float, default=None)
    parser.add_argument("--max_tavily_search_chars", type=int, default=None)
    parser.add_argument("--max_parallel_requests", type=int, default=None)
    parser.add_argument("--enable_statistics", action="store_true")
    parser.add_argument("--enable_tracing", action="store_true")
    parser.add_argument("--trace_output_dir", type=str, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Use current sources to compare TensorRT-LLM and vLLM for serving "
            "large language models. Include concrete differences supported by "
            "tool observations."
        ),
        help="Research prompt to run.",
    )
    return parser.parse_args()


def _load_config(config_path: str | None) -> dict:
    if config_path and Path(config_path).is_file():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _mcp_sse_urls(cfg: dict) -> list[str]:
    tools = cfg.get("mcp_tools") or {}
    client_host = str(cfg.get("mcp_client_host") or "127.0.0.1")
    urls: list[str] = []
    for name, default_port in _MCP_SERVICE_ORDER:
        tool_cfg = tools.get(name) or {}
        host = tool_cfg.get("client_host") or client_host
        port = int(tool_cfg.get("port", default_port))
        urls.append(f"http://{host}:{port}/sse")
    return urls


def _dump_task_metrics_summary(output_dir: Path) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        TaskMetricsCollector.print_summary()
    summary = buf.getvalue()
    print(summary, end="")

    summary_path = output_dir / "tot_research.metrics.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Task metrics summary saved to {summary_path}")


async def main() -> None:
    args = parse_arguments()
    cfg = _load_config(args.config)

    openai_api_key = args.openai_api_key or cfg.get("openai_api_key", "tensorrt_llm")
    base_url = args.base_url or cfg.get("base_url", "http://localhost:8000/v1")
    model = args.model or cfg.get("model", "Qwen3/Qwen3-30B-A3B")
    max_tokens = (
        args.max_tokens if args.max_tokens is not None else int(cfg.get("max_tokens", 16384))
    )
    max_depth = args.max_depth if args.max_depth is not None else int(cfg.get("max_depth", 3))
    num_thoughts = (
        args.num_thoughts_per_step
        if args.num_thoughts_per_step is not None
        else int(cfg.get("num_thoughts_per_step", 3))
    )
    branch_factor = (
        args.branch_factor if args.branch_factor is not None else int(cfg.get("branch_factor", 2))
    )
    complete_score_threshold = (
        args.complete_score_threshold
        if args.complete_score_threshold is not None
        else float(cfg.get("complete_score_threshold", 8.0))
    )
    max_tavily_search_chars = (
        args.max_tavily_search_chars
        if args.max_tavily_search_chars is not None
        else int(cfg.get("max_tavily_search_chars", 6000))
    )
    max_parallel_requests = (
        args.max_parallel_requests
        if args.max_parallel_requests is not None
        else int(cfg.get("max_parallel_requests", 1024))
    )

    trace_output_dir: Path | None = None
    if args.enable_tracing or args.enable_statistics:
        trace_output_dir_str = args.trace_output_dir or cfg.get("trace_output_dir")
        if trace_output_dir_str:
            trace_output_dir = Path(trace_output_dir_str)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_output_dir = Path(f"tot_research_trace_{timestamp}")
        trace_output_dir.mkdir(parents=True, exist_ok=True)

    mcp_urls = _mcp_sse_urls(cfg)
    client = AsyncOpenAI(api_key=openai_api_key, base_url=base_url)
    generation_worker = TRTOpenaiWorker(client, model)
    mcp_worker = MCPWorker.init_with_urls(mcp_urls)
    await mcp_worker.init_in_asyncio_event_loop()

    print(f"MCP SSE URLs ({len(mcp_urls)}): {mcp_urls}")
    print(f"Prompt: {args.prompt}")
    llm = create_tot_research_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=max_tokens,
        max_depth=max_depth,
        num_thoughts_per_step=num_thoughts,
        branch_factor=branch_factor,
        complete_score_threshold=complete_score_threshold,
        max_tavily_search_chars=max_tavily_search_chars,
        max_parallel_requests=max_parallel_requests,
        enable_statistics=args.enable_statistics,
        enable_tracing=args.enable_tracing,
    )

    try:
        future = llm.generate_async(args.prompt)
        result = await future.aresult()
        text = result.outputs[0].text if result.outputs else ""
        print("\n" + "=" * 80)
        print("FINAL ANSWER:")
        print("=" * 80)
        print(text)

        if args.enable_tracing and result.task_collections:
            tracer = result.task_collections.get("execution_tracer")
            if tracer:
                trace = tracer.export_trace()
                assert trace_output_dir is not None
                trace_path = trace_output_dir / "tot_research.trace.json"
                full_trace_path = trace_output_dir / "tot_research.full.trace.json"
                trace.save(str(trace_path))
                trace.save(str(full_trace_path), full=True)
                print(f"Execution trace saved to {trace_path}")
                print(f"Full execution trace saved to {full_trace_path}")
    finally:
        if args.enable_statistics:
            assert trace_output_dir is not None
            _dump_task_metrics_summary(trace_output_dir)
        llm.shutdown()
        generation_worker.shutdown()
        mcp_worker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
