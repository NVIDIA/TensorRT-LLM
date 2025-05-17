import asyncio
import time
from typing import List, Mapping, Tuple, Type

from pydantic import BaseModel

from tensorrt_llm.scaffolding.scaffolding_llm import (ScaffoldingLlm,
                                                      ScaffoldingResult)
from tensorrt_llm.scaffolding.task_collection import (TaskCollection,
                                                      with_task_collection)


class ScaffoldingBenchRequest(BaseModel):
    prompt: str


async def enqueue_requests(input_queue, requests):
    for request in requests:
        await input_queue.put(request)

    await input_queue.put(None)


async def process_request(scaffolding_llm, request, output_queue, semaphore):
    async with semaphore:
        request_start_time = time.time()
        result = scaffolding_llm.generate_async(request.prompt)
        await result.aresult()
        request_execution_time = time.time() - request_start_time
        await output_queue.put((result, request_execution_time))


async def run_scaffolding_llm(scaffolding_llm, input_queue, output_queue,
                              concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = set()

    while True:
        request = await input_queue.get()
        if request is None:
            break
        task = asyncio.create_task(
            process_request(scaffolding_llm, request, output_queue, semaphore))
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
        concurrency: int) -> Tuple[List[ScaffoldingResult], List[float], float]:
    wrapper_prototype_controller_with_task_collection(scaffolding_llm,
                                                      task_collection_types)

    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    start_time = time.time()
    results = []
    requests_execution_time = []
    enqueue_task = asyncio.create_task(enqueue_requests(input_queue, requests))

    run_scaffolding_llm_task = asyncio.create_task(
        run_scaffolding_llm(scaffolding_llm, input_queue, output_queue,
                            concurrency))

    while True:
        try:
            item = await asyncio.wait_for(output_queue.get(), timeout=1.0)
            if item is None:
                break
            result, request_execution_time = item
            results.append(result)
            requests_execution_time.append(request_execution_time)
        except asyncio.TimeoutError:
            continue

    total_time = time.time() - start_time

    enqueue_task.result()
    run_scaffolding_llm_task.result()

    return results, requests_execution_time, total_time
