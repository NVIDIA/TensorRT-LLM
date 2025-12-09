import argparse
import asyncio
import sys
import threading

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import (
    ChatTokenCounter,
    MCPWorker,
    NativeGenerationController,
    ScaffoldingLlm,
    TaskTimer,
    TRTOpenaiWorker,
)
from tensorrt_llm.scaffolding.benchmark import ScaffoldingBenchRequest, async_scaffolding_benchmark
from tensorrt_llm.scaffolding.contrib.DeepResearch import create_open_deep_research_scaffolding_llm
from tensorrt_llm.scaffolding.load_generation_strategy import ConcurrentStrategy

# Global lock for thread-safe printing
print_lock = threading.Lock()


def print_benchmark_results(
    benchmark_type, results, requests_start_time, requests_execution_time, total_time
):
    avg_all = sum(requests_execution_time) / len(requests_execution_time)

    with print_lock:
        print("\n" + "=" * 60)
        print(f"{benchmark_type} Benchmark Results:")
        print("=" * 60)

        for i, (start_time, execution_time) in enumerate(
            zip(requests_start_time, requests_execution_time)
        ):
            print(
                f"{benchmark_type} request {i}: start time = {start_time:.3f}s, execution time = {execution_time:.3f}s"
            )

        print(f"\n{benchmark_type} total requests number: {len(results)}")
        print(f"{benchmark_type} total execution time: {total_time:.3f}s")
        print(f"{benchmark_type} average execution time (all): {avg_all:.3f}s")

        print("=" * 60 + "\n")
        sys.stdout.flush()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_api_key", type=str, default="tensorrt_llm")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="gpt-oss-20b")

    # Benchmark mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="agent_chat",
        choices=["agent_only", "chat_only", "agent_chat"],
        help="Benchmark mode: 'agent_only', 'chat_only', or 'agent_chat' (default: agent_chat)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--agent_prompt_num",
        type=int,
        default=128,
        help="Number of prompts to send for agent benchmark (default: 128)",
    )
    parser.add_argument(
        "--chat_prompt_num",
        type=int,
        default=20,
        help="Number of prompts to send for chat benchmark (default: 20)",
    )
    parser.add_argument(
        "--times", type=int, default=1, help="Number of times to run the benchmark (default: 1)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=40,
        help="Number of concurrent requests for Concurrent strategy (default: 40)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16 * 1024,
        help="Maximum number of tokens to generate (default: 8192)",
    )
    parser.add_argument(
        "--max_parallel_requests",
        type=int,
        default=1024,
        help="Maximum number of parallel requests (default: 1024)",
    )
    parser.add_argument("--enable_statistics", action="store_true")

    return parser.parse_args()


async def async_agent_benchmark(args):
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)

    generation_worker = TRTOpenaiWorker(client, args.model)

    mcp_worker = MCPWorker.init_with_urls(["http://0.0.0.0:8082/sse"])
    await mcp_worker.init_in_asyncio_event_loop()

    llm = create_open_deep_research_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=args.max_tokens,
        max_parallel_requests=args.max_parallel_requests,
        enable_statistics=args.enable_statistics,
    )

    prompt = """
        From 2020 to 2050, how many elderly people will there be in China, \
        the United States, South Korea, and Japan? \
        What is their consumption potential across various aspects such as \
        clothing, food, housing, and transportation? \
        Based on population projections, elderly consumer willingness, \
        and potential changes in their consumption habits, \
        please produce a market size analysis report for the elderly \
        demographic in these countries.
    """

    task_collection_types = {}
    requests = [
        ScaffoldingBenchRequest(prompt=str(i) + ". " + prompt) for i in range(args.agent_prompt_num)
    ]
    strategy = ConcurrentStrategy(concurrency=args.concurrency)

    for i in range(args.times):
        (
            results,
            requests_start_time,
            requests_execution_time,
            total_time,
        ) = await async_scaffolding_benchmark(
            llm, task_collection_types, requests, strategy=strategy
        )

        print_benchmark_results(
            "Agent", results, requests_start_time, requests_execution_time, total_time
        )

        with print_lock:
            if args.enable_statistics:
                token_counting_info = ChatTokenCounter.get_global_info()
                print("token counting info: " + str(token_counting_info))
                timer_info = TaskTimer.get_global_info()
                print("timer info: " + str(timer_info))

    # Graceful shutdown
    await mcp_worker.async_shutdown()
    llm.shutdown()
    generation_worker.shutdown()

    # Wait for LLM's internal event loop to fully stop
    if not llm.own_loop:
        await llm.main_loop_stop_event.wait()

    return


async def async_chat_benchmark(args):
    """Chat benchmark using simple generation without agent capabilities."""
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    generation_worker = TRTOpenaiWorker(client, args.model)

    chat_controller = NativeGenerationController(
        sampling_params={
            "temperature": 0.9,
            "max_tokens": args.max_tokens,
        }
    )

    chat_llm = ScaffoldingLlm(
        chat_controller,
        {NativeGenerationController.WorkerTag.GENERATION: generation_worker},
        max_parallel_requests=args.max_parallel_requests,
    )

    chat_prompt = (
        "Natalia sold clips to 48 of her friends in April, "
        "and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?"
    )

    task_collection_types = {}
    requests = [
        ScaffoldingBenchRequest(prompt=f"{i}. {chat_prompt}") for i in range(args.chat_prompt_num)
    ]
    strategy = ConcurrentStrategy(concurrency=args.concurrency)

    (
        results,
        requests_start_time,
        requests_execution_time,
        total_time,
    ) = await async_scaffolding_benchmark(
        chat_llm, task_collection_types, requests, strategy=strategy
    )

    print_benchmark_results(
        "Chat", results, requests_start_time, requests_execution_time, total_time
    )

    # Graceful shutdown
    chat_llm.shutdown()
    generation_worker.shutdown()

    # Wait for LLM's internal event loop to fully stop
    if not chat_llm.own_loop:
        await chat_llm.main_loop_stop_event.wait()

    return


def run_async_in_thread(coro):
    """Run async coroutine in a separate thread with proper cleanup."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())

            # Check for any remaining tasks (should be none after proper shutdown)
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if pending:
                print(f"Warning: {len(pending)} tasks still pending after shutdown")
                # Cancel as last resort for abnormal cases
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as e:
            print(f"Warning: Error during event loop cleanup: {e}")
        finally:
            loop.close()


def run_benchmark_in_thread(async_func, args, name):
    """Create a thread to run async benchmark function."""
    thread = threading.Thread(target=run_async_in_thread, args=(async_func(args),), name=name)
    return thread


if __name__ == "__main__":
    args = parse_arguments()

    # Select benchmarks based on mode
    benchmarks = []
    if args.mode in ["agent_only", "agent_chat"]:
        benchmarks.append((async_agent_benchmark, "Agent-Benchmark"))
    if args.mode in ["chat_only", "agent_chat"]:
        benchmarks.append((async_chat_benchmark, "Chat-Benchmark"))

    # Create and start all benchmark threads
    threads = []
    print(f"Starting benchmarks in mode: {args.mode}")
    print("=" * 60)
    for async_func, name in benchmarks:
        thread = run_benchmark_in_thread(async_func, args, name)
        threads.append(thread)
        print(f"- {name} thread")
    print()

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print("\n" + "=" * 60)
    print("All benchmarks completed!")
    print("=" * 60)
    sys.stdout.flush()
