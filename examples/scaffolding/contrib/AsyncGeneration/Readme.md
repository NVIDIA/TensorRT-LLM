## Overview

`StreamGenerationTask` is an extension of `GenerationTask` designed for token-level streaming generation in asynchronous LLM workflows using TensorRT-LLM. It enables the controller to receive partial results during generation, which is critical for real-time or latency-sensitive applications such as chatbots, speech generation, or UI-interactive systems.

---

## Features

- ✅ Supports **streamed token delivery** by step (e.g., `streaming_step=1`).
- ✅ Supports **cancellation** of generation using a flag (`cancel_flag=True`).
- ✅ Tracks **stream completion status** (`end_flag=True` when done).
- ✅ Integrated with a streaming-capable LLM interface (`generate_async`).

---

## Fields in `StreamGenerationTask`

| Field | Description |
|-------|-------------|
| `cancel_flag` | If `True`, the generation will be cancelled on the worker side. |
| `streaming_step` | Number of new tokens required before returning control to the controller. If set to `0`, the task is returned immediately if any new tokens are available. |
| `request_handle` | Internal handle for the streaming generation (used by the worker). |
| `end_flag` | Indicates whether generation is finished. |
| `output_str` / `output_tokens` / `logprobs` | Outputs after each generation step. |

---

## Usage in Worker

Here's how a Worker would handler `StreamGenerationTask`
You can see more details in stream_generation_controller.py and stream_generation_task.py

```python
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

    sampling_params = worker.combine_sampling_params_with_generation_task(task)
    if task.request_handle == None:
        task.request_handle = worker.llm.generate_async(
            task.input_str, sampling_params=sampling_params, streaming=True)
    await get_step_or_more_tokens(task)

    # TODO: error handle
    return TaskStatus.SUCCESS
```
## Notes
Ensure the `worker.llm.generate_async(...)` method supports streaming=True.

The controller is responsible for repeatedly calling the `stream_generation_handler` until `task.end_flag` is set to True.

## TODO

- Add error handling for failed `request_handle`
- Support retry or backoff mechanism if generation stalls
