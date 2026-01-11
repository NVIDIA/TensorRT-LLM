### :title Runtime Configuration Examples
### :order 6
### :section Customization
'''
This script demonstrates various runtime configuration options in TensorRT-LLM,
including KV cache management and CUDA graph optimizations.

**KV Cache Configuration:**

The KV cache (key-value cache) stores attention keys and values during inference,
which is crucial for efficient autoregressive generation. Proper KV cache configuration helps with:

1. **Memory Management**: Control GPU memory allocation for the key-value cache through
   `free_gpu_memory_fraction`, balancing memory between model weights and cache storage.

2. **Block Reuse Optimization**: Enable `enable_block_reuse` to optimize memory usage
   for shared prefixes across multiple requests, improving throughput for common prompts.

3. **Performance Tuning**: Configure cache block sizes and total capacity to match
   your workload characteristics (batch size, sequence length, and request patterns).

Please refer to the `KvCacheConfig` API reference for more details.

**CUDA Graph Configuration:**

CUDA graphs help reduce kernel launch overhead and improve GPU utilization by capturing
and replaying GPU operations. Benefits include:

- Reduced kernel launch overhead for repeated operations
- Better GPU utilization through optimized execution
- Improved throughput for inference workloads

Please refer to the `CudaGraphConfig` API reference for more details.

**How to Run:**

Run all examples:
```bash
python llm_runtime.py
```

Run specific example:
```bash
python llm_runtime.py --example kv_cache
python llm_runtime.py --example cuda_graph
```
'''

import argparse

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig


def example_cuda_graph_config():
    """
    Example demonstrating CUDA graph configuration for performance optimization.

    CUDA graphs help with:
    - Reduced kernel launch overhead
    - Better GPU utilization
    - Improved throughput for repeated operations
    """
    print("\n=== CUDA Graph Configuration Example ===")

    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[1, 2, 4],
        enable_padding=True,
    )

    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        cuda_graph_config=cuda_graph_config,  # Enable CUDA graphs
        max_batch_size=4,
        max_seq_len=512,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5))

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(max_tokens=50, temperature=0.8, top_p=0.95)

    # This should benefit from CUDA graphs
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print()


def example_kv_cache_config():
    """Example demonstrating KV cache configuration for memory management and performance."""
    print("\n=== KV Cache Configuration Example ===")
    print("\n1. KV Cache Configuration:")

    llm_advanced = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_batch_size=8,
        max_seq_len=1024,
        kv_cache_config=KvCacheConfig(
            # free_gpu_memory_fraction: the fraction of free GPU memory to allocate to the KV cache
            free_gpu_memory_fraction=0.5,
            # enable_block_reuse: whether to enable block reuse
            enable_block_reuse=True))

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    outputs = llm_advanced.generate(prompts)
    for i, output in enumerate(outputs):
        print(f"Query {i+1}: {output.prompt}")
        print(f"Answer: {output.outputs[0].text[:100]}...")
        print()


def main():
    """
    Main function to run all runtime configuration examples.
    """
    parser = argparse.ArgumentParser(
        description="Runtime Configuration Examples")
    parser.add_argument("--example",
                        type=str,
                        choices=["kv_cache", "cuda_graph", "all"],
                        default="all",
                        help="Which example to run")

    args = parser.parse_args()

    if args.example == "kv_cache" or args.example == "all":
        example_kv_cache_config()

    if args.example == "cuda_graph" or args.example == "all":
        example_cuda_graph_config()


if __name__ == "__main__":
    main()
