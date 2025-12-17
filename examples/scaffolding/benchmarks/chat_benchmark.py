"""Chat benchmark implementation for scaffolding benchmarks."""

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import NativeChatController, ScaffoldingLlm, TRTOpenaiWorker
from tensorrt_llm.scaffolding.benchmark import ScaffoldingBenchRequest, async_scaffolding_benchmark
from tensorrt_llm.scaffolding.load_generation_strategy import ConcurrentStrategy

from .benchmark_utils import load_prompts_from_json, print_benchmark_results


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
