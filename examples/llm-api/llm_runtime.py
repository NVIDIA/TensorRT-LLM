### :title Runtime Configuration Examples
### :order 6
### :section Customization

import argparse
import datetime

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (CudaGraphConfig, KvCacheConfig,
                                 KvCacheRetentionConfig)


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
    print("\n=== KV Cache Configuration Example ===")
    print("\n1. KV Cache Configuration:")

    llm_advanced = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       max_batch_size=8,
                       max_seq_len=1024,
                       kv_cache_config=KvCacheConfig(
                           free_gpu_memory_fraction=0.5,
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


def example_kv_cache_retention_config():
    """
    Example demonstrating KV cache retention configuration for advanced KV cache management via LLMAPI.

    KV cache retention config helps with:
    - Enables users to influence how blocks are selected for eviction.
    - Users can specify two attributes that guide block eviction: priority and duration.
    - The priority value sets the relative retention priority (how important it is to retain that block in the cache),
    - and the duration value sets how long this priority level should apply for.
    """
    print("\n=== Basic KV Cache Retention Configuration Example ===")

    # Create a retention configuration that the first 2 tokens
    # with priority 90 and 30-second retention period and 80 decode retention priority
    kv_cache_retention_config = KvCacheRetentionConfig([
        KvCacheRetentionConfig.TokenRangeRetentionConfig(
            0, 2, 90, datetime.timedelta(seconds=30))
    ], 80)

    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              max_batch_size=4,
              max_seq_len=512,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.8,
                                            enable_block_reuse=True))

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(max_tokens=50, temperature=0.8, top_p=0.95)

    # Generate with KV cache retention configuration
    outputs = llm.generate(prompts,
                           sampling_params,
                           kv_cache_retention_config=kv_cache_retention_config)

    for i, output in enumerate(outputs):
        print(f"Query {i+1}: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print()

    # Example with multiple token ranges and different priorities
    print("\n=== Advanced KV Cache Retention Configuration Example ===")

    advanced_prompts = [
        "System: You are a helpful assistant. User: What is 2+2? Assistant:",
        "System: You are a math tutor. User: Solve 15*3. Assistant:",
    ]

    advanced_retention_config = KvCacheRetentionConfig(
        [
            # High priority for system prompt (tokens 0-5)
            KvCacheRetentionConfig.TokenRangeRetentionConfig(0, 5, 90, None),
            # Medium priority for user input (tokens 5-10) with 60-second retention
            KvCacheRetentionConfig.TokenRangeRetentionConfig(
                5, 10, 60, datetime.timedelta(seconds=60)),
            # Lower priority to 30 for generated content (tokens 10+) with 30-second retention
            KvCacheRetentionConfig.TokenRangeRetentionConfig(
                10, None, 30, datetime.timedelta(seconds=30))
        ],
        70)  # Decode priority set to 70

    print(
        "Using advanced retention configuration with multiple token ranges...")
    outputs_advanced = llm.generate(
        advanced_prompts,
        sampling_params,
        kv_cache_retention_config=advanced_retention_config)

    for i, output in enumerate(outputs_advanced):
        print(f"Query {i+1}: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print()


def main():
    """
    Main function to run all runtime configuration examples.
    """
    parser = argparse.ArgumentParser(
        description="Runtime Configuration Examples")
    parser.add_argument(
        "--example",
        type=str,
        choices=["kv_cache", "kv_cache_retention", "cuda_graph", "all"],
        default="all",
        help="Which example to run")

    args = parser.parse_args()

    if args.example == "kv_cache" or args.example == "all":
        example_kv_cache_config()

    if args.example == "cuda_graph" or args.example == "all":
        example_cuda_graph_config()

    if args.example == "kv_cache_retention" or args.example == "all":
        example_kv_cache_retention_config()


if __name__ == "__main__":
    main()
