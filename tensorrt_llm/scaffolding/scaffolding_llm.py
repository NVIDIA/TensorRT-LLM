import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

from .controller import Controller, ScaffoldingOutput
from .worker import Worker


@dataclass(frozen=True)
class ScaffoldingRequest:
    prompt: str
    kwargs: Mapping[str, Any]
    controller: Controller
    result: "ScaffoldingResult"


class ScaffoldingResult:

    def __init__(self, ):
        self._done = False
        self.aqueue = asyncio.Queue()
        self.output = None

    def set_output(self, output: ScaffoldingOutput):
        self.aqueue.put_nowait(output)

    async def aresult_step(self):
        # TODO: error handling?
        self.output = await self.aqueue.get()
        self._done = True

    def result(self) -> "ScaffoldingResult":
        if not self._done:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(self.aresult_step(), loop).result()
        return self

    async def aresult(self) -> "ScaffoldingResult":
        if not self._done:
            await self.aresult_step()
        return self

    def __await__(self):
        return self.aresult().__await__()


class ScaffoldingLlm:

    def __init__(
            self,
            controller_cls: type,  # type of Controller
            controller_args: dict,  # args for init a Crontroller instance
            workers: Mapping[
                str, Worker],  # map of role of Crontroller to a worker instance
    ):
        self.controller_cls = controller_cls
        self.controller_args = controller_args
        self.workers = workers

        self.loop = self._get_loop()
        asyncio.set_event_loop(self.loop)
        self.task_queue = asyncio.Queue()
        if self.own_loop:
            self._run_main_loop_thread()
        else:
            self._run_main_loop_coroutine()

        # For top scheduler
        self.running_req_count = 0
        self.max_parallel_requests = 64
        self.pending_queue = deque()

    def __enter__(self):
        return self

    def __exit__(self):
        self.shutdown()

    def _get_loop(self):
        try:
            self.own_loop = False
            return asyncio.get_running_loop()
        except RuntimeError:
            self.own_loop = True
            return asyncio.new_event_loop()
        return None

    async def _main_loop_async_func(self):

        async def handle_single_request(request: ScaffoldingRequest):
            gen = request.controller.generate(request.prompt, **request.kwargs)
            try:
                while True:
                    task_list = next(gen)
                    async_tasks = []
                    for task in task_list:
                        task_role = task.role
                        assert task_role in self.workers.keys()
                        worker = self.workers[task_role]
                        async_tasks.append(
                            asyncio.create_task(worker.run_task(task)))
                    await asyncio.gather(*async_tasks)
                    # TODO: if we need use results?
            except StopIteration as e:
                scaffolding_output = e.value
                request.result.set_output(scaffolding_output)
            self.running_req_count -= 1
            maybe_schedule()

        def schedule_request(request: ScaffoldingRequest):
            asyncio.create_task(handle_single_request(request))
            self.running_req_count += 1

        def maybe_schedule(request: ScaffoldingRequest = None):
            if request is not None:
                self.pending_queue.append(request)
            while self.running_req_count < self.max_parallel_requests and len(
                    self.pending_queue) > 0:
                first_request = self.pending_queue.popleft()
                schedule_request(first_request)

        async def handle_event():
            while True:
                item = await self.task_queue.get()
                if item is None:
                    return
                elif isinstance(item, ScaffoldingRequest):
                    maybe_schedule(item)
                else:
                    raise ValueError(
                        f'type of task_queue item ({type(item)}) is not supported'
                    )

        handle_event_task = asyncio.create_task(handle_event())
        await handle_event_task

    def _run_main_loop_coroutine(self):
        asyncio.run_coroutine_threadsafe(self._main_loop_async_func(),
                                         self.loop)

    def _run_main_loop_thread(self):

        def main_loop_thread():
            self.loop.run_until_complete(self._main_loop_async_func())

        main_loop_thread = threading.Thread(target=main_loop_thread)
        main_loop_thread.start()

    def generate_async(self, prompt: str) -> ScaffoldingResult:

        result = ScaffoldingResult()

        async def put_request():
            request = ScaffoldingRequest(
                prompt=prompt,
                kwargs={},
                result=result,
                controller=self.controller_cls(**(self.controller_args)))

            await self.task_queue.put(request)

        asyncio.run_coroutine_threadsafe(put_request(), self.loop)

        return result

    def generate(self, prompt: str) -> ScaffoldingResult:
        scaffolding_result = self.generate_async(prompt)
        scaffolding_result.result()  # wait done
        return scaffolding_result

    def shutdown(self):
        # Let the merge thread break
        async def put_shutdown_task():
            await self.task_queue.put(None)

        asyncio.run_coroutine_threadsafe(put_shutdown_task(), self.loop)
