# Parallel Scheduler Benchmark Tests

This directory contains comprehensive benchmark tests that compare the performance of the new `ParallelStreamScheduler` against the existing `SimpleScheduler` in TensorRT-LLM.

## Overview

The benchmark tests measure:
- **Throughput**: Requests processed per second
- **Latency**: Average, P95, and P99 response times
- **GPU Utilization**: How efficiently the GPU is used
- **Memory Usage**: GPU memory consumption
- **Stream Utilization**: For parallel execution (additional metric)

## Test Files

### 1. `test_parallel_scheduler_benchmark.py`
Basic benchmark test that simulates parallel execution by running the same workload with different configurations.

**Features:**
- Compares SimpleScheduler vs ParallelScheduler
- Tests different load balancing strategies
- Multiple workload sizes (basic, medium, large)
- Statistical analysis with multiple runs

### 2. `test_parallel_scheduler_integration.py`
Integration test that actually uses the ParallelStreamScheduler with a custom executor wrapper.

**Features:**
- Real integration with the executor
- Custom ParallelExecutorWrapper class
- More realistic performance measurements
- Stream utilization tracking

## Running the Tests

### Prerequisites

1. **GPU Requirements**: At least 1 GPU (tests will be skipped if no GPU is available)
2. **Model**: A TensorRT-LLM model engine (tests use the existing test fixtures)
3. **Dependencies**: All TensorRT-LLM dependencies must be installed

### Basic Test Execution

```bash
# Run basic benchmark tests
pytest tests/unittest/bindings/test_parallel_scheduler_benchmark.py -v

# Run integration tests
pytest tests/unittest/bindings/test_parallel_scheduler_integration.py -v

# Run all benchmark tests
pytest tests/unittest/bindings/test_parallel_scheduler_*.py -v
```

### Specific Test Execution

```bash
# Run only basic benchmark
pytest tests/unittest/bindings/test_parallel_scheduler_benchmark.py::test_scheduler_benchmark_basic -v

# Run only medium workload test
pytest tests/unittest/bindings/test_parallel_scheduler_benchmark.py::test_scheduler_benchmark_medium -v

# Run configuration comparison test
pytest tests/unittest/bindings/test_parallel_scheduler_benchmark.py::test_parallel_scheduler_configurations -v
```

### Running with Custom Parameters

You can modify the test parameters in the test files:

```python
# In the test functions, you can adjust:
num_requests = 20        # Number of requests to process
max_prompt_len = 20      # Maximum prompt length
max_max_tokens = 20      # Maximum tokens to generate
num_runs = 3            # Number of runs for averaging
```

## Understanding the Results

### Sample Output

```
Running benchmark with 20 requests, 3 runs each
Model path: /path/to/model

Running with SimpleScheduler...
  Run 1/3
  Run 2/3
  Run 3/3

Running with ParallelScheduler...
  Run 1/3
  Run 2/3
  Run 3/3

============================================================
SCHEDULER PERFORMANCE COMPARISON
============================================================

SimpleScheduler (Average) Results:
  Total Time: 2.456s
  Throughput: 8.14 req/s
  GPU Memory: 1.23 GB
  GPU Utilization: 85.2%
  Avg Latency: 0.123s
  P95 Latency: 0.145s
  P99 Latency: 0.167s

ParallelScheduler (Average) Results:
  Total Time: 1.987s
  Throughput: 10.07 req/s
  GPU Memory: 1.45 GB
  GPU Utilization: 92.1%
  Stream Utilization: 88.5%
  Avg Latency: 0.099s
  P95 Latency: 0.118s
  P99 Latency: 0.134s

PERFORMANCE COMPARISON:
  Throughput Improvement: +23.7%
  Latency Improvement: +19.5%
  Memory Overhead: +17.9%
  GPU Utilization Improvement: +8.1%
  Stream Utilization: 88.5%
  ✅ ParallelScheduler shows 23.7% better throughput
  ✅ ParallelScheduler shows 8.1% better GPU utilization
============================================================
```

### Key Metrics Explained

1. **Throughput (req/s)**: Number of requests processed per second
   - Higher is better
   - Shows overall system performance

2. **Latency (s)**: Time to complete individual requests
   - Lower is better
   - P95/P99 show worst-case performance

3. **GPU Memory (GB)**: Memory usage during execution
   - Parallel execution may use more memory
   - Should be reasonable (<50% overhead)

4. **GPU Utilization (%)**: How efficiently the GPU is used
   - Higher is generally better
   - Parallel execution should improve this

5. **Stream Utilization (%)**: For parallel execution only
   - Shows how well the parallel streams are utilized
   - Should be close to GPU utilization

### Performance Expectations

#### When Parallel Scheduler Should Excel:
- **High throughput scenarios**: Multiple requests with varying workloads
- **Mixed workloads**: Combination of context and generation requests
- **GPU underutilization**: When single-stream execution leaves GPU idle

#### When Simple Scheduler Might Be Better:
- **Low latency scenarios**: Single request or very few requests
- **Memory constraints**: When parallel execution exceeds memory limits
- **Simple workloads**: Uniform request characteristics

## Load Balancing Strategies

The tests compare three load balancing strategies:

### 1. Round-robin
- Simple alternating assignment of requests to streams
- Good for uniform workloads
- Minimal overhead

### 2. Smart (Default)
- Separates context and generation requests, then distributes them
- Better for mixed workloads
- Considers request characteristics

### 3. Balanced
- Uses workload estimation for optimal distribution
- Best for heterogeneous workloads
- More computational overhead

## Troubleshooting

### Common Issues

1. **Test Skipped - No GPU**
   ```
   SKIPPED [1] test_parallel_scheduler_benchmark.py::test_scheduler_benchmark_basic: Requires at least 1 GPU
   ```
   - Ensure CUDA is available
   - Check `torch.cuda.device_count()`

2. **Memory Errors**
   ```
   RuntimeError: CUDA out of memory
   ```
   - Reduce `num_requests` or `max_prompt_len`
   - Increase `free_gpu_memory_fraction` in executor config

3. **Timeout Errors**
   ```
   assert i < max_wait_ms
   ```
   - Increase `max_wait_ms` in the test
   - Check if model is loading correctly

4. **Import Errors**
   ```
   ImportError: cannot import name 'ParallelStreamScheduler'
   ```
   - Ensure the scheduler module is properly installed
   - Check Python path

### Debug Mode

Run tests with verbose output for debugging:

```bash
pytest tests/unittest/bindings/test_parallel_scheduler_benchmark.py -v -s
```

The `-s` flag shows print statements, which include detailed progress information.

## Customization

### Adding New Metrics

To add new performance metrics, modify the `SchedulerBenchmarkResult` class:

```python
class SchedulerBenchmarkResult:
    def __init__(self, scheduler_name: str):
        # ... existing metrics ...
        self.new_metric = 0.0  # Add your metric here
```

### Testing Different Configurations

Create custom configurations for testing:

```python
custom_config = ParallelExecutionConfig(
    enable_parallel_execution=True,
    load_balancing_strategy="balanced",
    min_requests_for_parallel=5,
    stream_priority=1,
    enable_stream_synchronization=False,
    context_generation_fusion=True
)
```

### Benchmarking Different Workloads

Modify the request generation in the test functions:

```python
# For different workload patterns
for i in range(num_requests):
    # Vary prompt lengths more dramatically
    prompt_len = random.randint(5, max_prompt_len * 2)

    # Vary token generation more dramatically
    max_tokens = random.randint(1, max_max_tokens * 3)

    # Add more complex sampling configurations
    num_return_sequences = random.choice([1, 2, 4])
```

## Contributing

When adding new benchmark tests:

1. **Follow the naming convention**: `test_*_benchmark_*.py`
2. **Include comprehensive metrics**: Throughput, latency, memory, utilization
3. **Add proper assertions**: Verify both schedulers complete successfully
4. **Document new features**: Update this README with new capabilities
5. **Include error handling**: Graceful handling of edge cases

## Future Enhancements

Potential improvements to the benchmark tests:

1. **Multi-GPU testing**: Support for tensor/pipeline parallelism
2. **Real-time monitoring**: Integration with nvidia-ml-py for accurate GPU metrics
3. **Workload profiling**: Detailed analysis of request patterns
4. **Automated reporting**: Generate performance reports in various formats
5. **Continuous benchmarking**: Integration with CI/CD for performance regression testing
