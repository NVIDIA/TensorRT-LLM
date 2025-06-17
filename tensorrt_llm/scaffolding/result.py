import asyncio
from dataclasses import dataclass
from typing import Mapping, Optional

from tensorrt_llm.executor.result import GenerationResult


@dataclass(slots=True)
class ScaffoldingOutput:

    def __init__(self):
        self.output_str = None


class ScaffoldingResult:

    def __init__(self, streaming_event: Optional[asyncio.Event] = None):
        super().__init__()
        self.aqueue = asyncio.Queue()
        self.output = None
        self._done = False
        self.task_collections = None
        self.streaming_event = streaming_event

    def set_output(self, output: GenerationResult):
        print("[set_output] called")
        self.aqueue.put_nowait(output)
        self._done = True
        print("[set_output] put")

    async def set_output_async(self, output: GenerationResult):
        print("[set_output_async] called")
        await self.aqueue.put(output)
        print("[set_output_async] put")

    def set_task_collections(self, task_collections: Mapping[str,
                                                             "TaskCollection"]):
        self.task_collections = task_collections

    @property
    def finished(self) -> bool:
        return self.output is not None and self.output.finished

    async def _aresult_step(self):
        print("[_aresult_step] waiting for response")
        # TODO: error handling or raise exception?
        response = await self.aqueue.get()
        print("[_aresult_step] response received")
        if response is None:
            raise Exception("ScaffoldingLlm execution failed")
        self._handle_response(response)

    def result(self, timeout: Optional[float] = None) -> "ScaffoldingResult":
        if not self.finished:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(self.aresult(), loop).result()
        return self

    async def aresult(self) -> "ScaffoldingResult":
        while not self.finished:
            await self._aresult_step()
        return self

    def __await__(self):
        return self.aresult().__await__()

    def __iter__(self):
        return self

    def __next__(self):
        if self._done and self.finished:
            raise StopIteration

        self._result_step()
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.finished:
            print("[_aresult_step] streaming_event set")
            self.streaming_event.set() if self.streaming_event else None
        if self._done and self.finished:
            raise StopAsyncIteration

        await self._aresult_step()
        return self

    def _handle_response(self, response: GenerationResult):
        self.output = response  # .outputs[0].text
