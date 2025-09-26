import asyncio
import copy
import time
from enum import Enum
from typing import List

from tensorrt_llm.scaffolding import (Controller, ParallelProcess,
                                      ScaffoldingLlm, Task, TaskStatus, Worker)


class DummyTask(Task):

    def __init__(self, turn: int):
        self.turn = turn
        self.numbers = []

    @staticmethod
    def create_from_prompt(prompt: str) -> "DummyTask":
        task = DummyTask(2)
        return task

    def create_scaffolding_output(self):
        self.verify()
        return None

    def verify(self):
        for i in range(len(self.numbers)):
            assert self.numbers[i] == i, "task.numbers[i] should be i"


class DummyControllerBase(Controller):

    def generate(self, prompt: str, **kwargs):
        task = DummyTask.create_from_prompt(prompt)
        yield from self.process([task], **kwargs)
        return task.create_scaffolding_output()


# Controller that yields task.turn times for each task
class DummyController(DummyControllerBase):

    class WorkerTag(Enum):
        DUMMY = "dummy"

    def process(self, tasks: List[Task], **kwargs):
        yield_tasks = tasks
        while len(yield_tasks) > 0:
            new_tasks = []
            for task in yield_tasks:
                if len(task.numbers) < task.turn:
                    task.worker_tag = self.WorkerTag.DUMMY
                    new_tasks.append(task)
            yield_tasks = new_tasks

            if len(yield_tasks) > 0:
                yield yield_tasks


# The flag to enable parallel process
# We can use this flag to compare the performance of parallel process
# and sequence process
ENABLE_PARALLEL_PROCESS = True


class DummyParallelController(DummyControllerBase):

    def __init__(self, controllers):
        self.controllers = controllers

    def process(self, tasks: List[Task], **kwargs):
        global ENABLE_PARALLEL_PROCESS
        if ENABLE_PARALLEL_PROCESS:
            tasks_list = [
                copy.deepcopy(tasks) for _ in range(len(self.controllers))
            ]

            kwargs_list = [kwargs for _ in range(len(self.controllers))]

            yield ParallelProcess(self.controllers, tasks_list, kwargs_list)

            tasks = tasks_list[0]
        else:
            original_tasks = copy.deepcopy(tasks)
            for controller in self.controllers:
                tasks = copy.deepcopy(original_tasks)
                yield from controller.process(tasks, **kwargs)


class DummyWorker(Worker):

    async def dummy_handler(self, task: DummyTask):
        await asyncio.sleep(1)
        task.numbers.append(len(task.numbers))
        return TaskStatus.SUCCESS

    task_handlers = {DummyTask: dummy_handler}


def parallel_process_helper_run_and_verify(controllers):
    # Obtain the generator from parallel_process_helper.
    parallel_controller = DummyParallelController(controllers)
    worker = DummyWorker()
    llm = ScaffoldingLlm(parallel_controller,
                         {DummyController.WorkerTag.DUMMY: worker})

    global ENABLE_PARALLEL_PROCESS
    ENABLE_PARALLEL_PROCESS = True
    start_time = time.time()
    llm.generate("")
    end_time = time.time()
    print('Parallel process time:', end_time - start_time)

    ENABLE_PARALLEL_PROCESS = False
    start_time = time.time()
    llm.generate("")
    end_time = time.time()
    print('Sequence process time:', end_time - start_time)

    llm.shutdown()


def test_parallel_process_helper():
    NUM_CONTROLLERS = 3
    controllers = []

    for _ in range(NUM_CONTROLLERS):
        controller = DummyController()
        controllers.append(controller)

    parallel_process_helper_run_and_verify(controllers)


def test_parallel_process_helper_with_two_level():
    NUM_CONTROLLERS_LEVEL_1 = 2
    NUM_CONTROLLERS_LEVEL_2 = 2

    controllers_level_1 = []

    for _ in range(NUM_CONTROLLERS_LEVEL_1):
        controller = DummyController()
        controllers_level_1.append(controller)

    parallel_controller = DummyParallelController(controllers_level_1)
    controllers_level_2 = [parallel_controller]

    for _ in range(NUM_CONTROLLERS_LEVEL_2):
        controller = DummyController()
        controllers_level_2.append(controller)

    parallel_process_helper_run_and_verify(controllers_level_2)
