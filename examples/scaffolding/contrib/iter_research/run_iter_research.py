r"""IterResearch runner (TensorRT-LLM scaffolding).

Start Apiary, then the MCP servers, then the LLM, then::

    python examples/scaffolding/contrib/iter_research/run_iter_research.py \\
        --config examples/scaffolding/contrib/iter_research/config.yaml
"""

import argparse
import asyncio
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import (
    ChatTokenCounter,
    MCPWorker,
    TaskMetricsCollector,
    TaskTimer,
    TRTOpenaiWorker,
)
from tensorrt_llm.scaffolding.contrib.iter_research import create_iter_research_scaffolding_llm


def parse_arguments():
    parser = argparse.ArgumentParser(description="IterResearch with TensorRT-LLM scaffolding")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max_turn", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--question", type=str, default=None, help="Research question to ask")
    parser.add_argument("--enable_statistics", action="store_true")
    parser.add_argument("--enable_tracing", action="store_true")
    return parser.parse_args()


def _load_config(config_path: str | None) -> dict:
    if config_path and Path(config_path).is_file():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_MCP_SERVICE_ORDER = (
    # ("google_search", 8083),
    ("tavily_search", 8087),
    ("google_scholar", 8084),
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


async def main():
    args = parse_arguments()
    cfg = _load_config(args.config)

    openai_api_key = args.openai_api_key or cfg.get("openai_api_key", "tensorrt_llm")
    base_url = args.base_url or cfg.get("base_url", "http://localhost:8000/v1")
    model = args.model or cfg.get("model", "Qwen/Qwen2.5-72B-Instruct")
    mcp_urls = _mcp_sse_urls(cfg)
    max_turn = args.max_turn or cfg.get("max_turn", 25)
    max_tokens = args.max_tokens or cfg.get("max_tokens", 16384)

    client = AsyncOpenAI(api_key=openai_api_key, base_url=base_url)
    generation_worker = TRTOpenaiWorker(client, model)

    print(f"MCP SSE URLs ({len(mcp_urls)}): {mcp_urls}")
    mcp_worker = MCPWorker.init_with_urls(mcp_urls)
    await mcp_worker.init_in_asyncio_event_loop()

    llm = create_iter_research_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=max_tokens,
        max_turn=max_turn,
        enable_statistics=args.enable_statistics,
        enable_tracing=args.enable_tracing,
    )

    question = args.question or (
        "What are the latest advances in large language model inference "
        "optimization? Compare the approaches of vLLM, TensorRT-LLM, and "
        "SGLang in terms of performance, features, and architecture."
    )
    # question = args.question or (
    #     "Fetch the live HTML/markdown from https://www.timeanddate.com/worldclock/ "
    #     "and tell me the exact current time shown for New York City on that page right now.")
    # question = args.question or (
    #     "In the sandbox, run Python to simulate 200,000 rolls of two fair six-sided dice, "
    #     "count how often the sum is 7, and report the empirical probability"
    #     " and the code’s stdout (no hand calculation)."
    # )

    print(f"Question: {question}")
    print(f"Max turns: {max_turn}")
    print("Starting IterResearch...")

    future = llm.generate_async(question)
    result = await future.aresult()

    assert result.outputs[0].text is not None
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(result.outputs[0].text)

    if args.enable_statistics:
        token_info = ChatTokenCounter.get_global_info()
        print(f"\nToken counting info: {token_info}")
        timer_info = TaskTimer.get_global_info()
        print(f"Timer info: {timer_info}")
        TaskMetricsCollector.export_to_json("iter_research_metrics.json")

    if args.enable_tracing and result.task_collections:
        tracer = result.task_collections.get("execution_tracer")
        if tracer:
            trace = tracer.export_trace()
            trace.save("iter_research_trace.json")
            print("Execution trace saved to iter_research_trace.json")

    llm.shutdown()
    generation_worker.shutdown()
    mcp_worker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
