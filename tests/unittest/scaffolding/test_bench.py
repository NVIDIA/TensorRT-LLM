import asyncio
from typing import List

from tensorrt_llm.scaffolding import (GenerationTask,
                                      NativeGenerationController,
                                      ScaffoldingBenchRequest, ScaffoldingLlm,
                                      Task, TaskCollection, TaskStatus, Worker,
                                      async_scaffolding_benchmark)

OUTPUT_STR = "Yes."


class DummyWorker(Worker):

    async def dummy_generation_handler(self, task: GenerationTask):
        task.result = OUTPUT_STR
        return TaskStatus.SUCCESS

    task_handlers = {GenerationTask: dummy_generation_handler}


class DummyTaskCollection(TaskCollection):

    def __init__(self):
        super().__init__()
        self.output_len = 0

    def before_yield(self, tasks: List[Task]):
        pass

    def after_yield(self, tasks: List[Task]):
        self.output_len = len(tasks[0].result)


def test_scaffolding_benchmark():
    task_collection_types = {"bench_dummy_collection": DummyTaskCollection}

    prototype_controller = NativeGenerationController()
    dummy_worker = DummyWorker()
    workers = {NativeGenerationController.WorkerTag.GENERATION: dummy_worker}
    scaffolding_llm = ScaffoldingLlm(prototype_controller, workers)

    requests_num = 100
    requests = [
        ScaffoldingBenchRequest(prompt="Is today a nice day?")
        for _ in range(requests_num)
    ]

    concurrency = 10

    results, requests_execution_time, total_time = asyncio.run(
        async_scaffolding_benchmark(scaffolding_llm, task_collection_types,
                                    requests, concurrency))

    scaffolding_llm.shutdown()

    assert len(results) == requests_num
    assert len(requests_execution_time) == requests_num
    assert results[0].cur_output == OUTPUT_STR
    assert results[0].task_collections[
        "bench_dummy_collection"].output_len == len(OUTPUT_STR)
