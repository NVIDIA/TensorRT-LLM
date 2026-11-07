# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
### :title Measure startup time
### :order 4
### :section Basics
"""Run an LLM model on random requests, then print the startup time metrics.

The script reports two times measured in this process:
- ``import_seconds``: time to import ``tensorrt_llm``.
- ``time_to_ready_seconds``: time from the script start until the ``LLM``
  instance is created and ready to accept requests.

It also prints the startup metrics that worker rank 0 records while it loads
the model weights, sets up the KV cache and warms up the model.

Example:
    python3 llm_startup_time.py --model meta-llama/Llama-3.1-8B-Instruct \
        --num_requests 32 --input_len 512 --output_len 128
"""

import argparse
import random
import time
from typing import Any

# Record the start time before importing TensorRT-LLM to include import time.
script_start_time = time.perf_counter()

from tensorrt_llm import LLM, SamplingParams  # noqa: E402

import_end_time = time.perf_counter()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model name or path to a local HF checkpoint.",
    )
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--num_requests", type=int, default=32)
    parser.add_argument(
        "--input_len", type=int, default=512, help="Number of random prompt tokens per request."
    )
    parser.add_argument(
        "--output_len", type=int, default=128, help="Number of tokens to generate per request."
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_random_prompts(
    num_requests: int, input_len: int, vocab_size: int, seed: int
) -> list[list[int]]:
    """Make prompts of random token IDs within ``[0, vocab_size)``."""
    rng = random.Random(seed)
    return [[rng.randrange(vocab_size) for _ in range(input_len)] for _ in range(num_requests)]


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested metrics into dotted names.

    Example: ``{"model_loader": {"total_model_loading_seconds": 1.5}}``
    becomes ``{"model_loader.total_model_loading_seconds": 1.5}``.
    """
    metric_name_to_value = {}
    for name, value in metrics.items():
        metric_name = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            metric_name_to_value.update(flatten_metrics(value, metric_name))
        else:
            metric_name_to_value[metric_name] = value
    return metric_name_to_value


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print("=" * 60)
    print(f"= {title}")
    print("=" * 60)
    for name, value in flatten_metrics(metrics).items():
        if isinstance(value, float):
            value = f"{value:.3f}"
        print(f"{name}: {value}")


def main() -> None:
    args = parse_arguments()

    # The LLM constructor returns once the model is loaded and warmed up and
    # the executor accepts requests.
    llm = LLM(model=args.model, tensor_parallel_size=args.tp_size)
    ready_time = time.perf_counter()

    # The TRT-LLM tokenizer wrapper does not expose vocab_size, so read it
    # from the wrapped HF tokenizer.
    vocab_size = llm.tokenizer.tokenizer.vocab_size
    prompts = make_random_prompts(args.num_requests, args.input_len, vocab_size, args.seed)
    # Ignore EOS so that every request generates exactly output_len tokens.
    sampling_params = SamplingParams(max_tokens=args.output_len, ignore_eos=True)
    generate_start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    generate_seconds = time.perf_counter() - generate_start_time
    num_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

    print_metrics("STARTUP METRICS (worker rank 0)", llm.startup_metrics)
    print_metrics(
        "RUN SUMMARY",
        {
            "num_requests": len(outputs),
            "num_output_tokens": num_output_tokens,
            "generate_seconds": generate_seconds,
            "import_seconds": import_end_time - script_start_time,
            "llm_init_seconds": ready_time - import_end_time,
            "time_to_ready_seconds": ready_time - script_start_time,
        },
    )

    llm.shutdown()


if __name__ == "__main__":
    main()
