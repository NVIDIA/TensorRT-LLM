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
Another input is `inflight_request_ids`, which supports pipeline parallelism or overlapped execution in the C++ runtime.
Since PyTorch Flow does not support pipeline parallelism, `inflight_request_ids` is an empty set.
The outputs are `context_requests` and `generation_requests`, which are the scheduled context and generation requests.
Requests not in these lists are not selected for the model forward pass.

`SimpleScheduler` combines these two schedulers, first using `CapacityScheduler` and then `MicroBatchScheduler`, to get the final schedule result.
The inputs to `SimpleScheduler` include `active_requests` and `inflight_request_ids`, and the outputs are `context_requests`, `generation_requests`, and `paused_requests`.

## Customize Your Own Scheduler

To customize the scheduler or batching mechanism, implement your own `CapacityScheduler` and `MicroBatchScheduler` by inheriting their respective classes.
If two-step scheduling is unnecessary, inherit `RequestScheduler` and implement `schedule_request` directly.

The live `CapacityScheduler` implementation is `BindCapacityScheduler` in [scheduler/scheduler.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py).
It wraps the C++ capacity scheduler and selects a policy such as `CapacitySchedulerPolicy.GUARANTEED_NO_EVICT`.
The Python-side policy helper is `GuaranteedNoEvictPolicy` in the same module (used by `PyCapacityScheduler`); there is no longer a `GuaranteedNoEvictScheduler` `CapacityScheduler` subclass.
When implementing a custom scheduler, resource estimation should align with resource allocation and deallocation in `kv_cache_manager`.

After implementing your own scheduler, integrate it into the PyExecutor.
For the PyTorch backend, construction lives in `create_py_executor_instance` in [_util.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/_util.py):

```python
    capacity_scheduler = BindCapacityScheduler(
        scheduler_capacity,
        kv_cache_manager.impl if kv_cache_manager is not None else None,
        peft_cache_manager.impl if peft_cache_manager is not None else None,
        scheduler_config.capacity_scheduler_policy,
        ...
    )
    mb_scheduler = BindMicroBatchScheduler(
        max_batch_size,
        max_num_tokens,
        ctx_chunk_config,
        no_schedule_until_state=no_schedule_until_state,
    )
```

Similar adjustments can be made for `MicroBatchScheduler`. This allows the `PyExecutor` to execute with your customized scheduling logic.