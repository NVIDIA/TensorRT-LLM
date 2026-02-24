import argparse
import asyncio

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import (
    ChatTokenCounter,
    MCPWorker,
    QueryCollector,
    TaskMetricsCollector,
    TaskTimer,
    TRTOpenaiWorker,
)
from tensorrt_llm.scaffolding.contrib.Coder import create_coder_scaffolding_llm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the Coder agent for code generation and modification tasks."
    )
    parser.add_argument("--openai_api_key", type=str, default="tensorrt_llm")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen3/Qwen3-30B-A3B")
    parser.add_argument(
        "--mcp_url",
        type=str,
        default="http://0.0.0.0:8083/sse",
        help="URL for the CoderMCP server",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for the Coder agent. If not provided, a default example is used.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16384,
        help="Maximum tokens for generation",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=50,
        help="Maximum tool-calling iterations",
    )
    parser.add_argument("--enable_statistics", action="store_true")
    parser.add_argument("--enable_query_collector", action="store_true")
    return parser.parse_args()


async def main():
    args = parse_arguments()
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)

    generation_worker = TRTOpenaiWorker(client, args.model)

    mcp_worker = MCPWorker.init_with_urls([args.mcp_url])
    await mcp_worker.init_in_asyncio_event_loop()

    llm = create_coder_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        enable_statistics=args.enable_statistics,
    )

    # Use custom prompt or default example
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = "Implement a collective communication library in C++"

    print(f"Running Coder agent with prompt:\n{prompt}\n")
    print("-" * 50)

    future = llm.generate_async(prompt)
    result = await future.aresult()

    assert result.outputs[0].text is not None
    print("\nFinal output:\n" + result.outputs[0].text)

    if args.enable_statistics:
        token_counting_info = ChatTokenCounter.get_global_info()
        print("\nToken counting info: " + str(token_counting_info))
        timer_info = TaskTimer.get_global_info()
        print("Timer info: " + str(timer_info))
        TaskMetricsCollector.export_to_json("coder_task_metrics.json")

    if args.enable_query_collector:
        QueryCollector.get_global_info()
        print("Query info dumped to query_result.json!")

    llm.shutdown()
    generation_worker.shutdown()
    mcp_worker.shutdown()
    return


if __name__ == "__main__":
    asyncio.run(main())
