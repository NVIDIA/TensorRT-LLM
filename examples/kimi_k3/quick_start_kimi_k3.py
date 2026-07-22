# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Run Kimi K3 with the TensorRT-LLM LLM API.

Example:
    trtllm-llmapi-launch python3 examples/kimi_k3/quick_start_kimi_k3.py \
        --model /path/to/kimi-k3-checkpoint --tp-size 16
"""

import argparse

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

SAMPLES = [
    ("The capital of France is", "Paris"),
    ("1 + 1 = 2, 2 + 2 = 4, 4 + 4 =", "8"),
    ("Water is made of hydrogen and", "oxygen"),
    (
        "Question: Natalia sold clips to 48 of her friends in April, and "
        "then she sold half as many clips in May. How many clips did "
        "Natalia sell altogether in April and May?\nAnswer:",
        "#### 72",
    ),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a Kimi K3 checkpoint (the Slurm launcher validates "
        "and bind-mounts it, so remote Hugging Face IDs are not supported "
        "through that flow).",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=16,
        help="Number of GPUs used for expert parallelism (default: 16).",
    )
    parser.add_argument(
        "--enable-block-reuse",
        action="store_true",
        help="Enable KV-cache block reuse (unified-pool hybrid cache "
        "manager with KDA recurrent-state snapshots; prefix-cache hits "
        "skip recomputing shared prompt prefixes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    llm = LLM(
        model=args.model,
        # DEP deployment: attention runs data-parallel and the experts
        # shard across the expert-parallel group (EP width == tp).
        tensor_parallel_size=args.tp_size,
        enable_attention_dp=True,
        moe_expert_parallel_size=args.tp_size,
        trust_remote_code=True,
        max_batch_size=8,
        max_seq_len=4096,
        max_num_tokens=4096,
        enable_chunked_prefill=True,
        disable_overlap_scheduler=False,
        # CUDA graphs require the graph-safe MLA latent-cache append (write
        # positions derived from device tensors, attention_backend/utils.py);
        # verified at GSM8K parity with the eager path.
        cuda_graph_config=CudaGraphConfig(enable_padding=True,
                                          max_batch_size=8),
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=args.enable_block_reuse,
            free_gpu_memory_fraction=0.25,
            tokens_per_block=64,
        ),
    )

    sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
    prompts = [prompt for prompt, _ in SAMPLES]
    try:
        for output, (_, expected) in zip(llm.generate(prompts, sampling_params), SAMPLES):
            generated_text = output.outputs[0].text
            print(f"Prompt: {output.prompt!r}")
            print(f"Generated text: {generated_text!r}")
            print(f"Contains expected text {expected!r}: {expected in generated_text}\n")
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
