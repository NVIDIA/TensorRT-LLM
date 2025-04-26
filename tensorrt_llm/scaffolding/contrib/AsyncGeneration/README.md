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

## Usage in Controller/Worker

The Controller can utilize `StreamGenerationTask` to enable efficient streaming-based generation workflows:
- It sends tasks to the worker, which returns them when the number of newly generated tokens reaches the specified `streaming_step`.
- It can cancel long-running tasks by setting `task.cancel_flag = True` when the number of generated tokens exceeds a predefined threshold.

To support this behavior on the worker side, we have implemented    `stream_generation_handler` and you need to register it with the worker in your project. This handler should process `StreamGenerationTask` instances step-by-step and update relevant fields such as `output_tokens`, `output_str`.

This design allows the controller and worker to coordinate generation in a token-efficient and responsive manner, ideal for real-time applications.

You can see more details in `stream_generation_controller.py` and `stream_generation_task.py` from `examples/scaffolding/contrib/AsyncGeneration`.

## Notes
Remember to register the `stream_generation_handler` with the `TRTLLMWorker`.

## TODO

- Add error handling for failed `request_handle`.
- Support retry or backoff mechanism if generation stalls.
