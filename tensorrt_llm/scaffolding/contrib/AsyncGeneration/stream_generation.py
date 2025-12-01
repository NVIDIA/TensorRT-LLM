import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from tensorrt_llm.scaffolding.task import GenerationTask, TaskStatus


@dataclass
class StreamGenerationTask(GenerationTask):
    # input field
    # if the flag is set to True, the worker will cancel the generation work
    cancel_flag: Optional[bool] = field(default=False)
    # the task will be returned to the controller with at least new streaming_step tokens
    # if the streaming_step is set to 0,
    # the task will be returned to the controller immediately with
    # new tokens that have already been generated.
    streaming_step: Optional[int] = field(default=1)

    #result field
    # worker set this field and identify the same task by this field
    request_handle: Any = field(default=None)
    # worker set this field to True when the generation is finished
    end_flag: bool = field(default=False)

    @staticmethod
    def create_from_generation_task(
            task: GenerationTask,
            streaming_step: int) -> "StreamGenerationTask":
        stream_task = StreamGenerationTask()
        stream_task.__dict__ = copy.deepcopy(task.__dict__)
        stream_task.streaming_step = streaming_step
        return stream_task


async def stream_generation_handler(worker,
                                    task: StreamGenerationTask) -> TaskStatus:

    async def get_step_or_more_tokens(task: StreamGenerationTask):
        if task.cancel_flag:
            task.end_flag = True
            task.request_handle.abort()
            return TaskStatus.SUCCESS

        for _ in range(task.streaming_step):
            await task.request_handle._aresult_step()
            if task.request_handle._done:
                break
        while not task.request_handle._done:
            async_task = asyncio.create_task(
                task.request_handle._aresult_step())
            if not async_task.done():
                async_task.cancel()
                break

        task.output_str = task.request_handle.outputs[0].text
        task.output_tokens = task.request_handle.outputs[0].token_ids
        task.cumulative_logprob = task.request_handle.outputs[
            0].cumulative_logprob
        task.logprobs = task.request_handle.outputs[0].logprobs
        if task.request_handle._done:
            task.end_flag = True

    sampling_params = worker.convert_task_params(task)
    if task.request_handle is None:
        task.request_handle = worker.llm.generate_async(
            task.input_str, sampling_params=sampling_params, streaming=True)
    await get_step_or_more_tokens(task)

    # TODO: error handle
    return TaskStatus.SUCCESS
