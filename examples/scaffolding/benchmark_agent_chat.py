import argparse
import asyncio
import json
import os
import random
import sys
import threading
from enum import Enum
from pathlib import Path
from typing import List

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from tensorrt_llm.scaffolding import (
    Worker,
    MCPWorker,
    Controller,
    NativeChatController,
    QueryCollector,
    ScaffoldingLlm,
    TaskMetricsCollector,
    TRTOpenaiWorker,
)
from tensorrt_llm.scaffolding.benchmark import ScaffoldingBenchRequest, async_scaffolding_benchmark
from tensorrt_llm.scaffolding.contrib.DeepResearch import create_open_deep_research_scaffolding_llm
from tensorrt_llm.scaffolding.load_generation_strategy import ConcurrentStrategy
from tensorrt_llm.scaffolding.task import GenerationTask, TaskStatus

# Global lock for thread-safe printing
print_lock = threading.Lock()


def load_prompts_from_json(num_prompts: int) -> List[str]:
    """Load prompts from all JSON files in the data directory.

    Args:
        num_prompts: Number of prompts to return

    Returns:
        List[str]: List of prompts with index prefixes
    """
    script_dir = Path(__file__).parent
    data_dir = script_dir / "contrib" / "DeepResearch" / "data"

    prompts = []

    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist")
    else:
        try:
            json_files = ["open_deepresearch_bench.json"]

            if not json_files:
                print(f"Warning: No JSON files found in {data_dir}")

            # Sort to ensure consistent order across runs
            for json_file in sorted(json_files):
                file_path = os.path.join(data_dir, json_file)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "prompt" in item:
                                prompts.append(item["prompt"])
                    elif isinstance(data, dict) and "prompt" in data:
                        prompts.append(data["prompt"])

                    print(
                        f"Loaded {
                            len(
                                [
                                    item
                                    for item in data
                                    if isinstance(item, dict) and 'prompt' in item
                                ]
                            )
                            if isinstance(data, list)
                            else (1 if 'prompt' in data else 0)
                        } prompts from {json_file}"
                    )

                except Exception as e:
                    print(f"Warning: Failed to read {file_path}: {e}")
                    continue

        except Exception as e:
            print(f"Warning: Failed to read directory {data_dir}: {e}")

    # Fallback to default prompt if no prompts loaded
    if not prompts:
        print("Warning: No prompts loaded from data files, using default prompt")
        default_prompt = (
            "Please analyze the global economic trends and provide a comprehensive report "
            "on the key factors influencing market dynamics in the coming years."
        )
        prompts = [default_prompt]

    if len(prompts) < num_prompts:
        original_prompts = prompts.copy()
        repeat_times = (num_prompts + len(original_prompts) - 1) // len(original_prompts)
        prompts = []
        for i in range(repeat_times):
            for p in original_prompts:
                if i == 0:
                    prompts.append(p)
                else:
                    prompts.append(f"[{i}].{p}")
    return prompts[:num_prompts]


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
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help="Path to the directory containing the generation model")

    # Benchmark enable flags
    parser.add_argument(
        "--enable_normal_agent",
        action="store_true",
        help="Enable normal agent benchmark",
    )
    parser.add_argument(
        "--enable_chatbot",
        action="store_true",
        help="Enable chatbot benchmark",
    )
    parser.add_argument(
        "--enable_multiround_chatbot",
        action="store_true",
        help="Enable multiround chatbot benchmark",
    )

    # Benchmark parameters
    parser.add_argument(
        "--agent_prompt_num",
        type=int,
        default=100,
        help="Number of prompts to send for agent benchmark (default: 10)",
    )
    parser.add_argument(
        "--chat_prompt_num",
        type=int,
        default=20,
        help="Number of prompts to send for chat benchmark (default: 20)",
    )
    parser.add_argument(
        "--chat_multiround_rounds",
        type=int,
        default=3,
        help="Number of rounds for multiround chat benchmark (default: 3)",
    )
    parser.add_argument(
        "--times", type=int, default=1, help="Number of times to run the benchmark (default: 1)"
    )
    parser.add_argument(
        "--normal_agent_concurrency",
        type=int,
        default=32,
        help="Concurrency for normal agent benchmark (default: 32)",
    )
    parser.add_argument(
        "--chat_concurrency",
        type=int,
        default=32,
        help="Concurrency for chatbot benchmark (default: 32)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8 * 1024,
        help="Maximum number of tokens to generate (default: 16384)",
    )
    parser.add_argument(
        "--max_tokens_chat",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate for chat (default: 1024)",
    )
    parser.add_argument(
        "--max_parallel_requests",
        type=int,
        default=1024,
        help="Maximum number of parallel requests (default: 1024)",
    )
    parser.add_argument("--enable_statistics", action="store_true")

    parser.add_argument("--enable_query_collector", action="store_true")

    # Burst agent parameters
    parser.add_argument(
        "--enable_burst_agent",
        action="store_true",
        help="Enable burst agent benchmark (simulates sudden traffic spike)",
    )
    parser.add_argument(
        "--burst_delay",
        type=float,
        default=240,
        help="Delay in seconds before burst traffic starts (default: 240)",
    )
    parser.add_argument(
        "--burst_prompt_num",
        type=int,
        default=32,
        help="Number of prompts for burst traffic (default: 32)",
    )
    parser.add_argument(
        "--burst_agent_concurrency",
        type=int,
        default=32,
        help="Concurrency for burst agent benchmark (default: 32)",
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        #TODO: fit the model real vocab size
        default=1000,
        help="Vocabulary size for multiround chat benchmark (default: 10000)",
    )

    return parser.parse_args()


async def run_agent_benchmark_core(llm, prompts, concurrency, benchmark_name, args, times=1):
    """Core agent benchmark logic that can be reused.

    Args:
        llm: The ScaffoldingLlm instance to use
        prompts: List of prompts to benchmark
        concurrency: Number of concurrent requests
        benchmark_name: Name for the benchmark (used in output)
        args: Command line arguments
        times: Number of times to run the benchmark
    """
    task_collection_types = {}
    requests = [ScaffoldingBenchRequest(prompt=prompt) for prompt in prompts]
    strategy = ConcurrentStrategy(concurrency=concurrency)

    for i in range(times):
        (
            results,
            requests_start_time,
            requests_execution_time,
            total_time,
        ) = await async_scaffolding_benchmark(
            llm, task_collection_types, requests, strategy=strategy
        )

        print_benchmark_results(
            benchmark_name, results, requests_start_time, requests_execution_time, total_time
        )

        with print_lock:
            if args.enable_statistics:
                print(f"Task metrics summary: {benchmark_name}")
                TaskMetricsCollector.print_summary()

        if args.enable_query_collector:
            QueryCollector.get_global_info()
            print(f"Query info dumped to query_result.json! ({benchmark_name})")

    return results


async def create_agent_resources(args):
    """Create isolated resources for an agent benchmark.

    Returns:
        Tuple of (llm, mcp_worker, generation_worker) for cleanup
    """
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
        enable_query_collector=args.enable_query_collector,
    )

    return llm, mcp_worker, generation_worker


async def cleanup_agent_resources(llm, mcp_worker, generation_worker):
    """Cleanup agent resources."""
    # Shutdown MCP worker and wait for background tasks to complete
    await mcp_worker.shutdown()

    llm.shutdown()
    generation_worker.shutdown()

    if not llm.own_loop:
        await llm.main_loop_stop_event.wait()


async def async_agent_benchmark(args):
    """Normal agent benchmark."""
    llm, mcp_worker, generation_worker = await create_agent_resources(args)
    prompts = load_prompts_from_json(args.agent_prompt_num)
    try:
        await run_agent_benchmark_core(
            llm, prompts, args.normal_agent_concurrency, "Agent-Normal", args, times=args.times
        )
    finally:
        await cleanup_agent_resources(llm, mcp_worker, generation_worker)


async def async_burst_agent_benchmark(args):
    """Burst agent benchmark that simulates sudden traffic spike."""
    with print_lock:
        print(f"\n[Burst] Waiting {args.burst_delay}s before starting burst traffic...")
        sys.stdout.flush()

    await asyncio.sleep(args.burst_delay)

    llm, mcp_worker, generation_worker = await create_agent_resources(args)

    with print_lock:
        print(
            f"\n[Burst] Starting burst traffic with "
            f"{args.burst_prompt_num} prompts, "
            f"concurrency={args.burst_agent_concurrency}"
        )
        sys.stdout.flush()

    try:
        prompts = load_prompts_from_json(args.burst_prompt_num)
        await run_agent_benchmark_core(
            llm, prompts, args.burst_agent_concurrency, "Agent-Burst", args, times=1
        )
    finally:
        await cleanup_agent_resources(llm, mcp_worker, generation_worker)


async def async_chat_benchmark(args):
    """Chat benchmark using simple generation without agent capabilities."""
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    chat_worker = TRTOpenaiWorker(client, args.model)

    chat_controller = NativeChatController(
        sampling_params={
            "temperature": 0.9,
            "max_tokens": args.max_tokens_chat,
        }
    )

    chat_llm = ScaffoldingLlm(
        chat_controller,
        {NativeChatController.WorkerTag.GENERATION: chat_worker},
        max_parallel_requests=args.max_parallel_requests,
    )

    prompts = load_prompts_from_json(args.chat_prompt_num)

    task_collection_types = {}
    requests = [ScaffoldingBenchRequest(prompt=prompt) for prompt in prompts]
    strategy = ConcurrentStrategy(concurrency=args.chat_concurrency)

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
    chat_worker.shutdown()

    # Wait for LLM's internal event loop to fully stop
    if not chat_llm.own_loop:
        await chat_llm.main_loop_stop_event.wait()

    return


class FakeMultiroundWorker(Worker):
    """Worker that simulates a user in multi-round conversations.
    
    Processes GenerationTask by combining previous input/output as history
    and adding random tokens for the next round's input.
    """
    
    rounds: int = 4
    input_tokens: List[int] = [1000, 100, 100, 100]
    output_tokens: List[int] = [500, 500, 500, 500]
    user_response_times: List[float] = [1.0, 1.0, 1.0, 1.0]
    
    def __init__(self, model_dir: str, vocab_size: int):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=False,
            use_fast=True,
        )
        self.vocab_size = vocab_size
    
    def _generate_random_token_ids(self, num_tokens: int) -> List[int]:
        if num_tokens <= 0:
            return []
        return [random.randint(0, self.vocab_size - 1) for _ in range(num_tokens)]
    
    async def multiround_generation_handler(self, task:GenerationTask) -> TaskStatus:
        """Process tasks to generate the next round's input."""
        if task.customized_result_fields is None:
            task.customized_result_fields = {}
        
        current_round = task.customized_result_fields.get('round_index', 0)
        if current_round >= self.rounds:
            task.customized_result_fields['is_conversation_end'] = True
            return TaskStatus.SUCCESS
        
        await asyncio.sleep(self.user_response_times[current_round])
        
        # Build new input: previous input + output + random tokens
        new_input_tokens = []
        if task.input_str:
            new_input_tokens.extend(self.tokenizer.encode(task.input_str))
        if task.output_str:
            new_input_tokens.extend(self.tokenizer.encode(task.output_str))
        new_input_tokens.extend(self._generate_random_token_ids(self.input_tokens[current_round]))
        
        # Update task for next round
        task.input_str = self.tokenizer.decode(new_input_tokens)
        task.ignore_eos = True
        task.output_tokens = None
        task.max_tokens = len(new_input_tokens) + self.output_tokens[current_round]
        task.output_str = None
        task.customized_result_fields['round_index'] = current_round + 1
        task.customized_result_fields['is_conversation_end'] = False
        
        return TaskStatus.SUCCESS
    
    task_handlers = {GenerationTask: multiround_generation_handler}


class FakeMultiroundChatController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation"
        FAKE_MULTIROUND = "fake_multiround"

    def __init__(self, rounds: int):
        super().__init__()
        self.rounds = rounds

    def generate(self, prompt: str):
        task = GenerationTask()
        yield from self.process([task])
        return task.create_scaffolding_output()

    def process(self, tasks) -> TaskStatus:
        task = tasks[0]
        for _ in range(self.rounds):
            task.worker_tag = self.WorkerTag.FAKE_MULTIROUND
            yield [task]
            task.worker_tag = self.WorkerTag.GENERATION
            yield [task]
    
        return TaskStatus.SUCCESS

async def async_multiround_chat_benchmark(args):
    """Multiround chat benchmark."""
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    generation_worker = TRTOpenaiWorker(client, args.model)
    multiround_worker = FakeMultiroundWorker(model_dir=args.model_dir, vocab_size=args.vocab_size)
    
    chat_controller = FakeMultiroundChatController(rounds=args.chat_multiround_rounds)
    chat_llm = ScaffoldingLlm(
        chat_controller,
        {FakeMultiroundChatController.WorkerTag.GENERATION: generation_worker, FakeMultiroundChatController.WorkerTag.FAKE_MULTIROUND: multiround_worker},
    )

    task_collection_types = {}
    requests = [ScaffoldingBenchRequest(prompt="Just a placeholder prompt") for _ in range(args.chat_prompt_num)]
    strategy = ConcurrentStrategy(concurrency=args.chat_concurrency)

    (
        results,
        requests_start_time,
        requests_execution_time,
        total_time,
    ) = await async_scaffolding_benchmark(
        chat_llm, task_collection_types, requests, strategy=strategy
    )

    print("chat finished")

    # Graceful shutdown
    chat_llm.shutdown(shutdown_workers=True)

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

    # Select benchmarks based on enable flags
    benchmarks = []
    if args.enable_normal_agent:
        benchmarks.append((async_agent_benchmark, "Agent-Benchmark"))
    if args.enable_burst_agent:
        benchmarks.append((async_burst_agent_benchmark, "Burst-Agent-Benchmark"))
    if args.enable_chatbot:
        benchmarks.append((async_chat_benchmark, "Chat-Benchmark"))
    if args.enable_multiround_chatbot:
        benchmarks.append((async_multiround_chat_benchmark, "Multiround-Chat-Benchmark"))

    if not benchmarks:
        print(
            "No benchmark enabled. Use --enable_normal_agent, --enable_burst_agent, or --enable_chatbot"
        )
        sys.exit(1)

    # Create and start all benchmark threads
    threads = []
    enabled_flags = []
    if args.enable_normal_agent:
        enabled_flags.append("normal_agent")
    if args.enable_burst_agent:
        enabled_flags.append("burst_agent")
    if args.enable_chatbot:
        enabled_flags.append("chatbot")
    print(f"Starting benchmarks: {', '.join(enabled_flags)}")
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
