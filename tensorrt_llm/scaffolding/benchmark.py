import asyncio
import time
from typing import List, Mapping, Optional, Tuple, Type

from pydantic import BaseModel

from tensorrt_llm.scaffolding.load_generation_strategy import \
    LoadGenerationStrategy
from tensorrt_llm.scaffolding.scaffolding_llm import (ScaffoldingLlm,
                                                      ScaffoldingResult)
from tensorrt_llm.scaffolding.task_collection import (TaskCollection,
                                                      with_task_collection)


class ScaffoldingBenchRequest(BaseModel):
    prompt: str
    scope_params: dict[str, str | list[str]] | None = None


def _apply_scope_params(
    scaffolding_llm: ScaffoldingLlm,
    request: ScaffoldingBenchRequest,
    result: ScaffoldingResult,
) -> None:
    if request.scope_params is None:
        return

    seen_workers: set[int] = set()
    for worker in scaffolding_llm.workers.values():
        worker_id = id(worker)
        if worker_id in seen_workers:
            continue
        seen_workers.add(worker_id)

        setter = getattr(worker, "set_scope_params", None)
        if callable(setter):
            setter(result.id, **request.scope_params)


async def enqueue_requests(input_queue, requests):
    for request in requests:
        await input_queue.put(request)

    await input_queue.put(None)


async def process_request(scaffolding_llm: ScaffoldingLlm, request,
                          output_queue, target_time, semaphore):
    async with semaphore:
        wait_time = target_time - time.time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        request_start_time = time.time()
        result = scaffolding_llm.generate_async(request.prompt)
        _apply_scope_params(scaffolding_llm, request, result)
        await result.aresult()
        request_execution_time = time.time() - request_start_time
        await output_queue.put(
            (result, request_start_time, request_execution_time))


async def run_scaffolding_llm(scaffolding_llm, input_queue, output_queue,
                              strategy: LoadGenerationStrategy):
    semaphore = strategy.get_semaphore()
    time_generator = strategy.request_times()

    tasks = set()

    while True:
        request = await input_queue.get()
        if request is None:
            break
        target_time = await time_generator.__anext__()

        task = asyncio.create_task(
            process_request(scaffolding_llm, request, output_queue, target_time,
                            semaphore))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    await asyncio.gather(*tasks)
    await output_queue.put(None)


def wrapper_prototype_controller_with_task_collection(scaffolding_llm,
                                                      task_collection_types):
    prototype_controller_type = type(scaffolding_llm.prototype_controller)
    controller_type_with_task_collection = prototype_controller_type

    for name, task_collection_type in task_collection_types.items():
        scaffolding_llm.prototype_controller.task_collections[
            name] = task_collection_type()
        controller_type_with_task_collection = with_task_collection(
            name, task_collection_type)(controller_type_with_task_collection)

    scaffolding_llm.enable_output_task_collection()


async def async_scaffolding_benchmark(
    scaffolding_llm: ScaffoldingLlm,
    task_collection_types: Mapping[str, Type[TaskCollection]],
    requests: List[ScaffoldingBenchRequest],
    concurrency: Optional[int] = None,
    strategy: Optional[LoadGenerationStrategy] = None
) -> Tuple[List[ScaffoldingResult], List[float], List[float], float]:
    if strategy is None:
        if concurrency is None:
            raise ValueError("Must provide either 'strategy' or 'concurrency'")
        from tensorrt_llm.scaffolding.load_generation_strategy import \
            ConcurrentStrategy
        strategy = ConcurrentStrategy(concurrency=concurrency)

    wrapper_prototype_controller_with_task_collection(scaffolding_llm,
                                                      task_collection_types)

    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    start_time = time.time()
    results = []
    requests_execution_time = []
    requests_start_time = []

    enqueue_task = asyncio.create_task(enqueue_requests(input_queue, requests))

    run_scaffolding_llm_task = asyncio.create_task(
        run_scaffolding_llm(scaffolding_llm, input_queue, output_queue,
                            strategy))

    while True:
        try:
            item = await asyncio.wait_for(output_queue.get(), timeout=1.0)
            if item is None:
                break
            result, request_start_time, request_execution_time = item
            results.append(result)
            requests_execution_time.append(request_execution_time)
            requests_start_time.append(request_start_time)
        except asyncio.TimeoutError:
            continue

    total_time = time.time() - start_time

    enqueue_task.result()
    run_scaffolding_llm_task.result()

    return results, requests_start_time, requests_execution_time, total_time
