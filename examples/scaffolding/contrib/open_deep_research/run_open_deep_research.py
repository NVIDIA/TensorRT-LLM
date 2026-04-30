r"""Open Deep Research runner (TensorRT-LLM scaffolding).

Start Apiary (if using python_interpreter), then MCP servers and LLM, then::

    python examples/scaffolding/contrib/open_deep_research/run_open_deep_research.py \\
        --config examples/scaffolding/contrib/open_deep_research/config.yaml --enable_tracing
"""

import argparse
import asyncio
import io
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import (
    MCPWorker,
    QueryCollector,
    TaskMetricsCollector,
    TRTOpenaiWorker,
)
from tensorrt_llm.scaffolding.contrib.open_deep_research import (
    create_open_deep_research_scaffolding_llm,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Open Deep Research with TensorRT-LLM scaffolding")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (same layout as iter_research/config.yaml).",
    )
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--max_tavily_search_chars", type=int, default=None)
    parser.add_argument("--max_parallel_requests", type=int, default=None)
    parser.add_argument(
        "--enable_statistics",
        action="store_true",
        help="Print live task metrics and a final summary. Does not write metrics JSON.",
    )
    parser.add_argument("--enable_query_collector", action="store_true")
    parser.add_argument("--enable_tracing", action="store_true")
    parser.add_argument(
        "--trace_output_dir",
        type=str,
        default=None,
        help="Directory for trace outputs when --enable_tracing is set",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="From 2020 to 2050, how many elderly people will there be in Japan? "
        "What is their consumption potential across various aspects such as clothing, "
        "food, housing, and transportation? Based on population projections, elderly "
        "consumer willingness, and potential changes in their consumption habits, "
        "please produce a market size analysis report for the elderly demographic.",
        help="Research prompt to use.",
    )
    return parser.parse_args()


def _load_config(config_path: str | None) -> dict:
    if config_path and Path(config_path).is_file():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_MCP_SERVICE_ORDER = (
    ("tavily_search", 8087),
    # ("google_scholar", 8084),
    ("fetch_webpage", 8085),
    ("python_interpreter", 8086),
)


def _mcp_sse_urls(cfg: dict) -> list[str]:
    tools = cfg.get("mcp_tools") or {}
    ch = str(cfg.get("mcp_client_host") or "127.0.0.1")
    out: list[str] = []
    for name, default_port in _MCP_SERVICE_ORDER:
        t = tools.get(name) or {}
        out.append(f"http://{t.get('client_host') or ch}:{int(t.get('port', default_port))}/sse")
    return out


def _dump_task_metrics_summary(output_dir: Path) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        TaskMetricsCollector.print_summary()
    summary = buf.getvalue()
    print(summary, end="")

    summary_path = output_dir / "open_deep_research.metrics.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Task metrics summary saved to {summary_path}")


async def main():
    args = parse_arguments()
    cfg = _load_config(args.config)

    openai_api_key = args.openai_api_key or cfg.get("openai_api_key", "tensorrt_llm")
    base_url = args.base_url or cfg.get("base_url", "http://localhost:8000/v1")
    model = args.model or cfg.get("model", "Qwen3/Qwen3-30B-A3B")
    max_tokens = (
        args.max_tokens if args.max_tokens is not None else int(cfg.get("max_tokens", 16384))
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
            trace_output_dir = Path(f"open_deep_research_trace_{timestamp}")
        trace_output_dir.mkdir(parents=True, exist_ok=True)

    mcp_urls = _mcp_sse_urls(cfg)
    client = AsyncOpenAI(api_key=openai_api_key, base_url=base_url)

    generation_worker = TRTOpenaiWorker(client, model)

    print(f"MCP SSE URLs ({len(mcp_urls)}): {mcp_urls}")
    if args.enable_statistics:
        TaskMetricsCollector.reset()
        print(
            "Statistics enabled: "
            f"model={model}, base_url={base_url}, max_tokens={max_tokens}, "
            f"max_tavily_search_chars={max_tavily_search_chars}, "
            f"max_parallel_requests={max_parallel_requests}, prompt_chars={len(args.prompt)}"
        )
    mcp_worker = MCPWorker.init_with_urls(mcp_urls)
    await mcp_worker.init_in_asyncio_event_loop()

    llm = create_open_deep_research_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=max_tokens,
        max_tavily_search_chars=max_tavily_search_chars,
        max_parallel_requests=max_parallel_requests,
        enable_statistics=args.enable_statistics,
        enable_query_collector=args.enable_query_collector,
        enable_tracing=args.enable_tracing,
    )

    try:
        future = llm.generate_async(args.prompt)
        result = await future.aresult()

        assert result.outputs[0].text is not None
        print("final output:" + result.outputs[0].text)

        if args.enable_query_collector:
            QueryCollector.get_global_info()
            print("Query info dumped to query_result.json!")

        if args.enable_tracing and result.task_collections:
            tracer = result.task_collections.get("execution_tracer")
            if tracer:
                trace = tracer.export_trace()
                assert trace_output_dir is not None
                trace_path = trace_output_dir / "open_deep_research.trace.json"
                full_trace_path = trace_output_dir / "open_deep_research.full.trace.json"
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
    return


if __name__ == "__main__":
    asyncio.run(main())
