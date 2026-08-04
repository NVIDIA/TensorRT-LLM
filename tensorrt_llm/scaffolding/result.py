import asyncio
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Union


@dataclass
class ScaffoldingOutput:
    text: str
    token_ids: List[int]


class ScaffoldingResult:

    def __init__(self):
        super().__init__()
        self.aqueue = asyncio.Queue()
        #self.cur_output: GenerationResult = None
        self.outputs = []
        # only support one output for now, so use an empty obj to init
        self.outputs.append(ScaffoldingOutput("", []))
        self._done = False
        self.task_collections = None

    def set_output(self, output: Union[ScaffoldingOutput, Any]):
        if isinstance(output, ScaffoldingOutput):
            self.set_output_streaming(output)
        # terminate
        self.set_output_streaming(None)

    def set_output_streaming(self, output: Union[ScaffoldingOutput, Any]):
        self.aqueue.put_nowait(output)

    def set_task_collections(self, task_collections: Mapping[str,
                                                             "TaskCollection"]):
        self.task_collections = task_collections

    async def _aresult_step(self):
        # TODO: error handling or raise exception?
        obj = await self.aqueue.get()
        if obj is None:
            self._done = True
        else:  # obj is ScaffoldingOutput
            self.outputs[0] = obj

    def result(self, timeout: Optional[float] = None) -> "ScaffoldingResult":
        if not self._done:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(self.aresult(), loop).result()
        return self

    async def aresult(self) -> "ScaffoldingResult":
        while not self._done:
            await self._aresult_step()
        return self

    def __await__(self):
        return self.aresult().__await__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration

        await self._aresult_step()
        return self
