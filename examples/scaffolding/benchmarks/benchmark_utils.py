"""Common utilities for scaffolding benchmarks."""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import List

from tensorrt_llm.scaffolding import ScaffoldingLlm

# Global lock for thread-safe printing
print_lock = threading.Lock()


async def shutdown_llm(llm: ScaffoldingLlm, shutdown_workers: bool = True) -> None:
    """Gracefully shutdown a ScaffoldingLlm instance.

    Args:
        llm: The ScaffoldingLlm instance to shutdown.
        shutdown_workers: Whether to shutdown workers as well.
    """
    llm.shutdown(shutdown_workers=shutdown_workers)

    # Wait for LLM's internal event loop to fully stop
    if not llm.own_loop:
        await llm.main_loop_stop_event.wait()


def load_prompts_from_json(num_prompts: int) -> List[str]:
    """Load prompts from all JSON files in the data directory.

    Args:
        num_prompts: Number of prompts to return

    Returns:
        List[str]: List of prompts with index prefixes
    """
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "contrib" / "open_deep_research" / "data"

    prompts = []

    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist")
    else:
        try:
            json_files = ["open_deep_research_bench.json"]

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
