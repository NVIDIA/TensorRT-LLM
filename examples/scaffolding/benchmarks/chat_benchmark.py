"""Chat benchmark implementation for scaffolding benchmarks."""

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import NativeChatController, ScaffoldingLlm, TRTOpenaiWorker
from tensorrt_llm.scaffolding.benchmark import ScaffoldingBenchRequest, async_scaffolding_benchmark
from tensorrt_llm.scaffolding.load_generation_strategy import ConcurrentStrategy
from tensorrt_llm.scaffolding.task import ChatTask
from tensorrt_llm.scaffolding.task_collection import TaskMetricsCollector, with_task_collection

from .benchmark_utils import load_prompts_from_json, print_benchmark_results, shutdown_llm


async def async_chat_benchmark(args):
    """Chat benchmark using simple generation without agent capabilities.

    Returns:
        Tuple of (results, requests_start_time, requests_execution_time, total_time)
    """
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    chat_worker = TRTOpenaiWorker(client, args.model, getattr(args, "kv_cache_hint_enabled", False))

    # Optionally wrap controller with task metrics collection
    controller_type = NativeChatController
    if getattr(args, "enable_statistics", False):
        controller_type = with_task_collection(
            "ChatTaskCollection",
            TaskMetricsCollector,
            controller_name="Chat",
            task_types=[ChatTask],
            enable_print=False,
        )(NativeChatController)

    chat_controller = controller_type(
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

    await shutdown_llm(chat_llm)

    return results, requests_start_time, requests_execution_time, total_time
