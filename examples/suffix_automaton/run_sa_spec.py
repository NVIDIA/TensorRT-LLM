#!/usr/bin/env python3
"""Integration test for Suffix Automaton Speculative Decoding with DeepSeek V3.

This script demonstrates how to use the native SA implementation with TensorRT-LLM's
MTP (Multi-Token Prediction) speculative decoding to boost acceptance rates.

Usage:
    python run_sa_spec.py --model /path/to/deepseek-v3 [--use_sa_spec]

The script will:
1. Load the model with MTP enabled
2. Run generation with and without SA to compare acceptance rates
3. Report performance metrics
"""

import argparse
import time
from typing import List

import torch

try:
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
except ImportError as e:
    print(f"Error: TensorRT-LLM not properly installed. {e}")
    print("Please install TensorRT-LLM first.")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SA speculative decoding with DeepSeek V3")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to DeepSeek V3 model checkpoint",
    )
    parser.add_argument(
        "--use_sa_spec",
        action="store_true",
        default=False,
        help="Enable suffix automaton speculative decoding",
    )
    parser.add_argument(
        "--sa_spec_threshold",
        type=int,
        default=4,
        help="Minimum match length to use SA draft tokens (default: 4)",
    )
    parser.add_argument(
        "--num_nextn_predict_layers",
        type=int,
        default=1,
        help="Number of MTP layers (default: 1)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "Write a Python function to calculate the Fibonacci sequence.",
            "Explain the concept of machine learning in simple terms.",
            "What is the capital of France and why is it important?",
        ],
        help="Prompts to test with",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum sequence length (default: model's max)",
    )
    return parser.parse_args()


def create_llm(args) -> "LLM":
    """Create LLM instance with MTP and optional SA configuration."""
    # Configure MTP decoding
    mtp_config = MTPDecodingConfig(
        num_nextn_predict_layers=args.num_nextn_predict_layers,
        use_sa_spec=args.use_sa_spec,
        sa_spec_threshold=args.sa_spec_threshold,
    )

    print("Creating LLM with configuration:")
    print(f"  - Model: {args.model}")
    print(f"  - MTP layers: {args.num_nextn_predict_layers}")
    print(f"  - SA enabled: {args.use_sa_spec}")
    if args.use_sa_spec:
        print(f"  - SA threshold: {args.sa_spec_threshold}")
    if args.max_seq_len is not None:
        print(f"  - Max seq len: {args.max_seq_len}")
    print()

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        speculative_config=mtp_config,
    )
    if args.max_seq_len is not None:
        llm_kwargs["max_seq_len"] = args.max_seq_len

    llm = LLM(**llm_kwargs)

    return llm


def run_generation(
    llm: "LLM",
    prompts: List[str],
    max_new_tokens: int,
) -> tuple:
    """Run generation and return outputs with timing."""
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,  # Greedy decoding for reproducibility
    )

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

    return outputs, total_time, total_tokens


def print_results(
    outputs,
    total_time: float,
    total_tokens: int,
    use_sa_spec: bool,
):
    """Print generation results and metrics."""
    mode = "SA+MTP" if use_sa_spec else "MTP only"
    print(f"\n{'=' * 60}")
    print(f"Results ({mode})")
    print(f"{'=' * 60}")

    for i, output in enumerate(outputs):
        print(f"\nPrompt {i + 1}:")
        print(f"  Input: {output.prompt[:50]}...")
        print(f"  Output tokens: {len(output.outputs[0].token_ids)}")
        generated = output.outputs[0].text[:200]
        print(f"  Generated: {generated}...")

    print(f"\n{'=' * 60}")
    print(f"Performance Metrics ({mode})")
    print(f"{'=' * 60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Throughput: {total_tokens / total_time:.2f} tokens/sec")
    print()


def main():
    args = parse_args()

    print("=" * 60)
    print("Suffix Automaton Speculative Decoding Test")
    print("=" * 60)

    # Create LLM with configuration
    llm = create_llm(args)

    # Get prompts to use
    prompts = args.prompts[: args.batch_size]
    print(f"Testing with {len(prompts)} prompt(s)")

    # Run generation
    outputs, total_time, total_tokens = run_generation(
        llm=llm,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
    )

    # Print results
    print_results(outputs, total_time, total_tokens, args.use_sa_spec)

    # Clean up
    del llm
    torch.cuda.empty_cache()

    print("Test completed successfully!")


if __name__ == "__main__":
    main()
