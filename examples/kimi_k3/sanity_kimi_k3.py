# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 ("golden prairie") end-to-end sanity test via the LLM API.

Runs a few greedy prompts through the standard TRT-LLM PyTorch backend and
asserts the outputs contain expected content. Two modes:

* full model (default): needs 16 GPUs (4x GB300 trays), asserts output
  quality (coherent completions, correct GSM8K-style answer).
* truncated debug (``KIMI_K3_NUM_LAYERS_OVERRIDE=<N>``, e.g. 4 on a single
  4-GPU tray): only asserts the pipeline runs e2e (load -> prefill ->
  decode -> shutdown); output text is NOT checked (a 4/93-layer model
  produces gibberish by construction).

Launch with ``sanity_kimi_k3.sbatch`` (same directory), or manually inside
the run container on every rank:

    trtllm-llmapi-launch python examples/kimi_k3/sanity_kimi_k3.py

Environment (the sbatch launcher sets everything up):
  KIMI_K3_CKPT                 model dir (required)
  KIMI_K3_TP                   tensor_parallel_size == EP width (default 16;
                               896 % TP must be 0)
  KIMI_K3_FUSED_MOE            native (default) = in-tree TRTLLM-Gen SiTU
                               fused MoE (mxe4m3_mxe2m1_block_scale_moe_runner,
                               no external cubin env needed);
                               0 = slow reference dequant loop
  KIMI_K3_NUM_LAYERS_OVERRIDE  truncate to first N layers (debug; skips
                               output-quality assertions)
Exit code 0 = PASS, 1 = FAIL.
"""

import os
import sys

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig

PROMPTS_AND_CHECKS = [
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


def main() -> int:
    ckpt = os.environ["KIMI_K3_CKPT"]  # exported by the sbatch launcher
    tp = int(os.environ.get("KIMI_K3_TP", "16"))
    truncated = os.environ.get("KIMI_K3_NUM_LAYERS_OVERRIDE") is not None
    max_tokens = int(os.environ.get("KIMI_K3_MAX_TOKENS", "64"))

    print(f"[sanity] ckpt={ckpt} tp(EP)={tp} truncated={truncated} "
          f"fused_moe={os.environ.get('KIMI_K3_FUSED_MOE', '0')}")

    llm = LLM(
        model=ckpt,
        tensor_parallel_size=tp,
        trust_remote_code=True,  # tiktoken tokenizer ships with the ckpt
        max_batch_size=8,
        max_seq_len=4096,
        max_num_tokens=4096,
        enable_chunked_prefill=False,
        disable_overlap_scheduler=True,
        cuda_graph_config=None,  # CUDA graphs unsupported for Kimi K3
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            free_gpu_memory_fraction=float(
                os.environ.get("KIMI_K3_FREE_GPU_FRACTION", "0.25")),
            # tokens_per_block=64 keeps the MLA (576, 512) generation path
            # on the flashinfer trtllm-gen kernel (32 falls back to a C++
            # path requiring num_heads % 64 == 0; K3 has 96 query heads).
            tokens_per_block=64,
        ),
    )

    prompts = [p for p, _ in PROMPTS_AND_CHECKS]
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    outputs = llm.generate(prompts, sampling)

    failures = []
    for out, (prompt, expected) in zip(outputs, PROMPTS_AND_CHECKS):
        text = out.outputs[0].text
        print("=" * 80)
        print(f"PROMPT:   {prompt!r}")
        print(f"OUTPUT:   {text!r}")
        if not truncated and expected not in text:
            failures.append(f"expected {expected!r} in output of {prompt!r}")
    llm.shutdown()

    if failures:
        print("[sanity] FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[sanity] PASS" + (" (pipeline only, truncated model)"
                             if truncated else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
