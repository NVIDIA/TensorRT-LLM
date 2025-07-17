import copy
from enum import Enum
from typing import List

from tensorrt_llm.scaffolding import (Controller, ParallelProcess,
                                      ScaffoldingLlm, Task, TaskCollection,
                                      TaskStatus, Worker, with_task_collection)


class DummyTask(Task):

    def __init__(self):
        self.before_flag = False
        self.after_flag = False

    @staticmethod
    def create_from_prompt(prompt: str) -> "DummyTask":
        task = DummyTask()
        return task

    def create_scaffolding_output(self):
        return None

    def verify(self):
        assert self.before_flag, "task.before_flag has not been set to True"
        assert self.after_flag, "task.after_flag has not been set to True"


class DummyTaskCollection(TaskCollection):

    def __init__(self):
        super().__init__()
        self.task_count = 0

    def before_yield(self, tasks: List[Task]):
        for task in tasks:
            task.before_flag = True

    def after_yield(self, tasks: List[Task]):
        for task in tasks:
            task.after_flag = True
            self.task_count += 1


class DummyControllerBase(Controller):

    class WorkerTag(Enum):
        DUMMY = "dummy"

    def __init__(self, expected_task_count: int):
        super().__init__()
        self.expected_task_count = expected_task_count

    def generate(self, prompt: str, **kwargs):
        task = DummyTask.create_from_prompt(prompt)
        yield from self.process([task], **kwargs)
        return task.create_scaffolding_output()

    def verify(self):
        assert self.task_collections[
            "dummy"].task_count == self.expected_task_count, (
                "task count is not as expected")


@with_task_collection("dummy", DummyTaskCollection)
# verify that we can have multiple task collections and they works separately
@with_task_collection("dummy2", DummyTaskCollection)
class DummyController(DummyControllerBase):

    def process(self, tasks: List[Task], **kwargs):
        for task in tasks:
            task.worker_tag = self.WorkerTag.DUMMY

        yield tasks

        for task in tasks:
            task.verify()

        self.verify()


@with_task_collection("dummy", DummyTaskCollection)
class DummyParallelController(DummyControllerBase):

    def __init__(self, controllers):
        expected_task_count = sum(
            [controller.expected_task_count for controller in controllers])
        super().__init__(expected_task_count)
        self.controllers = controllers

    def process(self, tasks: List[Task], **kwargs):
        tasks_list = [
            copy.deepcopy(tasks) for _ in range(len(self.controllers))
        ]

        kwargs_list = [kwargs for _ in range(len(self.controllers))]

        yield ParallelProcess(self.controllers, tasks_list, kwargs_list)

        for tasks in tasks_list:
            for task in tasks:
                task.verify()

        tasks = tasks_list[0]
        self.verify()


class DummyWorker(Worker):

    async def dummy_handler(self, task: DummyTask):
        return TaskStatus.SUCCESS

    task_handlers = {DummyTask: dummy_handler}


def run(controller, expected_task_count):
    worker = DummyWorker()
    llm = ScaffoldingLlm(controller, {DummyController.WorkerTag.DUMMY: worker})

    llm.generate("")
    llm.shutdown()


def test_dummy_task_collection():
    controller = DummyController(1)
    run(controller, 1)

    controller = DummyParallelController(
        [DummyController(1), DummyController(1)])
    run(controller, 2)
