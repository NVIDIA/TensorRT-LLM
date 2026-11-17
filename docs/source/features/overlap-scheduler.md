# Overlap Scheduler

To maximize GPU utilization, the scheduler overlaps CPU tasks (e.g., checking sampling stop criteria, updating responses, scheduling the next batch) with GPU computation.

## How It Works

At step *n*, the system launches GPU computation for step *n+1* without waiting for CPU tasks (e.g., stop criteria checks) from step *n* to complete. This allows:

- CPU work (step *n*) and GPU computation (step *n+1*) to run concurrently.
- Better GPU occupancy by reducing idle time.

This concurrent execution pipeline is illustrated in the `PyExecutor`'s logic:

```python
# Schedule and launch GPU work for the current step (n)
scheduled_batch, _, _ = self._schedule()
batch_outputs = self._forward_step(scheduled_batch, previous_tensors_device)
sample_state = self._sample_async(scheduled_batch, batch_outputs)

# While the GPU is busy, process the CPU-bound results from the previous step (n-1)
if self.previous_batch is not None:
    self._process_previous_batch()
```

## Tradeoff

The optimization introduces one extra decoding step but significantly improves throughput.

## Usage

Enabled by default. To disable, set `disable_overlap_scheduler=True` in the configuration.

### Client I/O During GPU Waits

With `enable_client_io_during_gpu_wait: true` (prototype, off by default), the
overlap loop publishes each iteration's responses, and the request intake parses
newly received requests, while the executor waits on the GPU, so this work does
not compete with the executor's own host work. Responses can reach clients up to
one iteration later, and a new request can be scheduled up to one iteration
later. The option has no effect when the overlap scheduler is disabled.


## References

- [NanoFlow: Towards Optimal Large Language Model Serving Throughput](https://arxiv.org/abs/2408.12757)
- https://lmsys.org/blog/2024-12-04-sglang-v0-4/#zero-overhead-batch-scheduler
