# Scheduler

TensorRT LLM PyTorch backend employs inflight batching, a mechanism where batching and scheduling occur dynamically at each LLM step.
The scheduler is invoked to determine which requests are scheduled at the current step.

## Scheduler Introduction

There are two kinds of schedulers:

- `CapacityScheduler`: This scheduler decides if resources should be allocated for each active request.
It considers the KV cache capacity and other resources, if applicable.
The input to `CapacityScheduler` includes all active requests that need processing.
The primary output is `fitting_requests`, representing the requests for which resources are reserved at the current step.
Another output is `paused_requests`, which supports request pausing in the C++ runtime.
- `MicroBatchScheduler`: This scheduler selects some requests from `fitting_requests` chosen by `CapacityScheduler`.
Another input is `inflight_request_ids`, the IDs of requests whose earlier work has not finished yet; the scheduler skips these requests so they are not scheduled again before their new tokens or encoder outputs exist.
With pipeline parallelism (`pp_size > 1`), `PyExecutor` runs its pipeline-parallel executor loop: when a micro-batch is queued, it adds the IDs of the generation requests and of the context requests that finish their last context chunk, and it removes them after the results of that micro-batch are processed.
Non-final context chunks are not added, so later chunks of a request can be scheduled while earlier chunks are still in the pipeline.
For encoder-decoder models, encoder request IDs stay in the set from encoder submission until the encoder output is ready.
With disaggregated serving, pipeline parallelism, and KV cache reuse, requests whose recompute pause is waiting for all pipeline-parallel ranks to agree are also held in the set.
Overlap scheduling does not add generation requests to the set; the next step reads the tokens sampled by the previous step directly on the device.
Without pipeline parallelism or encoder work, `inflight_request_ids` is therefore typically empty.
The outputs are `context_requests` and `generation_requests`, which are the scheduled context and generation requests.
Requests not in these lists are not selected for the model forward pass.

`SimpleScheduler` combines these two schedulers, first using `CapacityScheduler` and then `MicroBatchScheduler`, to get the final schedule result.
The inputs to `SimpleScheduler` include `active_requests` and `inflight_request_ids`, and the outputs are `context_requests`, `generation_requests`, and `paused_requests`.

## Customize Your Own Scheduler

To customize the scheduler or batching mechanism, implement your own `CapacityScheduler` and `MicroBatchScheduler` by inheriting their respective classes.
If two-step scheduling is unnecessary, inherit `RequestScheduler` and implement `schedule_request` directly.

An example of a `CapacityScheduler` implementation is the `GuaranteedNoEvictScheduler` class, found in [scheduler.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py).
This class was used before the C++ binding of `CapacityScheduler` and initially employed a Python-based scheduler.
It inherits `CapacityScheduler` and implements its own `schedule_request` method.
This method processes all `active_requests` and tries to schedule more requests that can fit in the KV cache.
Resource estimation should align with resource allocation and deallocation in `kv_cache_manager`.

Here is the code snippet:

```python
class GuaranteedNoEvictScheduler(CapacityScheduler):
    # only schedule requests has no_schedule_until_state <= state < no_schedule_after_state
    no_schedule_until_state = LlmRequestState.CONTEXT_INIT
    no_schedule_after_state = LlmRequestState.GENERATION_COMPLETE

    def __init__(self, max_num_requests: int, kv_cache_manager):
        super(GuaranteedNoEvictScheduler, self).__init__()
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        scheduled_requests = []
        pending_requests = []
        reserved_blocks = 0
        max_blocks = self.kv_cache_manager.get_max_resource_count()
        for request in active_requests:
            req_state = request.state
            # if request cannot be scheduled yet or request should no longer be scheduled, skip
            if req_state.value < self.no_schedule_until_state.value or req_state.value >= self.no_schedule_after_state.value:
                continue

            if len(scheduled_requests
                   ) >= self.max_num_requests or reserved_blocks >= max_blocks:
                break
            elif req_state == LlmRequestState.GENERATION_IN_PROGRESS or req_state == LlmRequestState.GENERATION_TO_COMPLETE:
                scheduled_requests.append(request)
                reserved_blocks += self.kv_cache_manager.get_needed_resource_to_completion(
                    request)
            else:
                pending_requests.append(request)

        available_blocks = max_blocks - reserved_blocks
        for request in pending_requests:
            req_state = request.state
            if len(scheduled_requests) >= self.max_num_requests:
                break
            elif req_state == LlmRequestState.CONTEXT_INIT:
                needed_blocks = self.kv_cache_manager.get_needed_resource_to_completion(
                    request)
                if needed_blocks <= available_blocks:
                    scheduled_requests.append(request)
                    available_blocks -= needed_blocks
                elif needed_blocks > available_blocks:
                    # If one requests fails to be scheduled, break
                    break

        assert len(scheduled_requests) > 0, (
            "no pending request can get enough resource to complete, "
            "please increase KV cache pool size.")
        return scheduled_requests, []
```

After implementing your own scheduler, integrate it into the PyExecutor.
For the PyTorch backend, the code is in [py_executor_creator.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/py_executor_creator.py).
In the `create_py_executor` function, there are two lines creating `CapacityScheduler`:

```python
    capacity_scheduler = BindCapacityScheduler(max_num_requests,
                                               kv_cache_manager.impl)
```

Similar adjustments can be made for `MicroBatchScheduler`. This allows the `PyExecutor` to execute with your customized scheduling logic.
