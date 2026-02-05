# Benchmarking with trtllm-bench

AutoDeploy is integrated with the `trtllm-bench` performance benchmarking utility, enabling you to measure comprehensive performance metrics such as token throughput, request throughput, and latency for your AutoDeploy-optimized models.

## Getting Started

Before benchmarking with AutoDeploy, review the [TensorRT LLM benchmarking guide](../../performance/perf-benchmarking.md#running-with-the-pytorch-workflow) to familiarize yourself with the standard trtllm-bench workflow and best practices.

## Basic Usage

Invoke the AutoDeploy backend by specifying `--backend _autodeploy` in your `trtllm-bench` command:

```bash
trtllm-bench \
  --model meta-llama/Llama-3.1-8B \
  throughput \
  --dataset /tmp/synthetic_128_128.txt \
  --backend _autodeploy
```

```{note}
As in the PyTorch workflow, AutoDeploy does not require a separate `trtllm-bench build` step. The model is automatically optimized during benchmark initialization.
```

## Advanced Configuration

For more granular control over AutoDeploy's behavior during benchmarking, use the `--extra_llm_api_options` flag with a YAML configuration file:

```bash
trtllm-bench \
  --model meta-llama/Llama-3.1-8B \
  throughput \
  --dataset /tmp/synthetic_128_128.txt \
  --backend _autodeploy \
  --extra_llm_api_options autodeploy_config.yaml
```

### Configuration Examples

#### Basic Performance Configuration (`autodeploy_config.yaml`)

```yaml
# runtime engine
runtime: trtllm

# model loading
skip_loading_weights: false

# Sequence configuration
max_batch_size: 256

# transform options
transforms:
  insert_cached_attention:
    # attention backend
    backend: flashinfer
  resize_kv_cache:
    # fraction of free memory to use for kv-caches
    free_mem_ratio: 0.8
  compile_model:
    # compilation backend
    backend: torch-opt
    # CUDA Graph optimization
    cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256]
```

Enable multi-GPU execution by specifying `--tp n`, where `n` is the number of GPUs.

## Configuration Options Reference

### Core Performance Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compile_backend` | `torch-compile` | Compilation backend: `torch-simple`, `torch-compile`, `torch-cudagraph`, `torch-opt` |
| `runtime` | `trtllm` | Runtime engine: `trtllm`, `demollm` |
| `free_mem_ratio` | `0.0` | Fraction of available GPU memory for KV cache (0.0-1.0) |
| `skip_loading_weights` | `false` | Skip weight loading for architecture-only benchmarks |

### CUDA Graph Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cuda_graph_batch_sizes` | `null` | List of batch sizes for CUDA graph creation |

```{tip}
For optimal CUDA graph performance, specify batch sizes that match your expected workload patterns. For example: `[1, 2, 4, 8, 16, 32, 64, 128]`
```

## Performance Optimization Tips

1. **Memory Management**: Set `free_mem_ratio` to 0.8-0.9 for optimal KV cache utilization
1. **Compilation Backend**: Use `torch-opt` for production workloads
1. **Attention Backend**: `flashinfer` generally provides the best performance for most models
1. **CUDA Graphs**: Enable CUDA graphs for batch sizes that match your production traffic patterns.
