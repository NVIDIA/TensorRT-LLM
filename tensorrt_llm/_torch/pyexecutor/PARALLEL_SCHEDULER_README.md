# ParallelStreamScheduler

The `ParallelStreamScheduler` is a new scheduler implementation that enables running two batches in parallel on different CUDA streams, maximizing GPU utilization and improving throughput for LLM inference.

## Overview

The parallel scheduler extends the existing TensorRT-LLM scheduler architecture by:

1. **Parallel Execution**: Running two batches simultaneously on separate CUDA streams
2. **Smart Load Balancing**: Distributing requests between streams based on workload characteristics
3. **Flexible Configuration**: Configurable execution strategies and synchronization options
4. **Backward Compatibility**: Maintains compatibility with existing scheduler interfaces

## Key Features

### 1. Parallel Stream Execution
- Executes context and generation phases on separate CUDA streams
- Reduces GPU idle time by overlapping computation
- Supports both synchronous and asynchronous execution modes

### 2. Load Balancing Strategies
- **Round-robin**: Simple alternating assignment of requests
- **Smart**: Separates context and generation requests, then distributes them
- **Balanced**: Uses workload estimation for optimal distribution

### 3. Configuration Options
- Enable/disable parallel execution
- Minimum requests threshold for parallel execution
- Stream priority settings
- Context-generation fusion options
- Stream synchronization control

## Usage

### Basic Setup

```python
from tensorrt_llm._torch.pyexecutor.scheduler import (
    ParallelStreamScheduler,
    ParallelExecutionConfig,
    BindCapacityScheduler,
    BindMicroBatchScheduler
)

# 1. Create configuration
config = ParallelExecutionConfig(
    enable_parallel_execution=True,
    load_balancing_strategy="smart",
    min_requests_for_parallel=2,
    stream_priority=0,
    enable_stream_synchronization=True,
    context_generation_fusion=False
)

# 2. Create underlying schedulers
capacity_scheduler = BindCapacityScheduler(
    max_num_requests=10,
    kv_cache_manager=your_kv_cache_manager,
    two_step_lookahead=False
)

micro_batch_scheduler = BindMicroBatchScheduler(
    max_batch_size=8,
    max_num_tokens=2048
)

# 3. Create parallel scheduler
parallel_scheduler = ParallelStreamScheduler(
    capacity_scheduler=capacity_scheduler,
    micro_batch_scheduler=micro_batch_scheduler,
    config=config
)
```

### Scheduling Requests

```python
# Schedule requests for parallel execution
scheduler_output = parallel_scheduler.schedule_request(
    active_requests=your_active_requests,
    inflight_request_ids=your_inflight_ids
)

# Or get stream-specific results
parallel_output = parallel_scheduler.schedule_request_parallel(
    active_requests=your_active_requests,
    inflight_request_ids=your_inflight_ids
)

# Access stream-specific requests
stream_0_context = parallel_output.stream_0_context_requests
stream_0_generation = parallel_output.stream_0_generation_requests
stream_1_context = parallel_output.stream_1_context_requests
stream_1_generation = parallel_output.stream_1_generation_requests
```

### Executing Batches

```python
# Synchronous execution
parallel_scheduler.execute_parallel_batches(
    stream_0_context=stream_0_context,
    stream_0_generation=stream_0_generation,
    stream_1_context=stream_1_context,
    stream_1_generation=stream_1_generation,
    context_executor=your_context_executor,
    generation_executor=your_generation_executor
)

# Asynchronous execution
stream_0_event, stream_1_event = parallel_scheduler.execute_parallel_batches_async(
    stream_0_context=stream_0_context,
    stream_0_generation=stream_0_generation,
    stream_1_context=stream_1_context,
    stream_1_generation=stream_1_generation,
    context_executor=your_context_executor,
    generation_executor=your_generation_executor
)

# Wait for completion when needed
stream_0_event.wait()
stream_1_event.wait()
```

## Configuration Options

### ParallelExecutionConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_parallel_execution` | bool | True | Enable parallel execution |
| `load_balancing_strategy` | str | "smart" | Load balancing strategy ("round_robin", "smart", "balanced") |
| `min_requests_for_parallel` | int | 2 | Minimum requests to enable parallel execution |
| `stream_priority` | int | 0 | Priority for CUDA streams |
| `enable_stream_synchronization` | bool | True | Enable stream synchronization |
| `context_generation_fusion` | bool | False | Fuse context and generation on same stream |

### Load Balancing Strategies

#### Round-robin
Simple alternating assignment of requests to streams:
```
Request 1 → Stream 0
Request 2 → Stream 1
Request 3 → Stream 0
Request 4 → Stream 1
...
```

#### Smart (Default)
Separates context and generation requests, then distributes them:
```
Context requests: [C1, C2, C3, C4] → Stream 0: [C1, C3], Stream 1: [C2, C4]
Generation requests: [G1, G2, G3] → Stream 0: [G1, G3], Stream 1: [G2]
```

#### Balanced
Uses workload estimation for optimal distribution:
```
Request 1 (workload=5.0) → Stream 0 (total=5.0)
Request 2 (workload=3.0) → Stream 1 (total=3.0)
Request 3 (workload=4.0) → Stream 1 (total=7.0)
Request 4 (workload=2.0) → Stream 0 (total=7.0)
```

## Integration with Existing Code

The `ParallelStreamScheduler` is designed to be a drop-in replacement for existing schedulers:

```python
# Replace SimpleScheduler with ParallelStreamScheduler
# old_scheduler = SimpleScheduler(capacity_scheduler, micro_batch_scheduler)
new_scheduler = ParallelStreamScheduler(capacity_scheduler, micro_batch_scheduler, config)

# Use the same interface
scheduler_output = new_scheduler.schedule_request(active_requests, inflight_ids)
```

## Performance Considerations

### When to Use Parallel Execution

- **High throughput scenarios**: Multiple requests with varying workloads
- **Mixed workload**: Combination of context and generation requests
- **GPU utilization**: When single-stream execution leaves GPU underutilized

### When to Use Single Stream

- **Low latency scenarios**: Single request or very few requests
- **Memory constraints**: When parallel execution exceeds memory limits
- **Simple workloads**: Uniform request characteristics

### Optimization Tips

1. **Load Balancing**: Use "smart" or "balanced" strategies for heterogeneous workloads
2. **Stream Priority**: Set appropriate priorities based on your use case
3. **Synchronization**: Disable stream synchronization for maximum parallelism when possible
4. **Context-Generation Fusion**: Enable fusion for simpler workloads

## Example Integration

See `parallel_scheduler_example.py` for a complete example of how to integrate the `ParallelStreamScheduler` into your inference pipeline.

## Limitations

1. **Memory Usage**: Parallel execution may require more GPU memory
2. **Complexity**: Adds complexity to the scheduling logic
3. **Synchronization**: Requires careful handling of stream synchronization
4. **Request Dependencies**: May not be suitable for requests with complex dependencies

## Future Enhancements

- Support for more than two streams
- Dynamic load balancing based on runtime metrics
- Integration with CUDA graphs for better performance
- Advanced workload prediction algorithms
