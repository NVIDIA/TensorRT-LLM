# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send one GSM8K request through MiniMax-M3 and dump per-layer activations.

Run this twice, once per KV-cache dtype, then diff the two dump directories
with compare_layer_dumps.py to find the first layer that diverges:

    export LLM_MODELS_ROOT=/path/to/models
    python examples/minimax_m3/dump_layers_one_request.py \
        --kv-dtype fp8   --tokens-per-block 128 --dump-dir /tmp/m3_fp8
    python examples/minimax_m3/dump_layers_one_request.py \
        --kv-dtype nvfp4 --tokens-per-block 128 --dump-dir /tmp/m3_nvfp4
    python examples/minimax_m3/compare_layer_dumps.py /tmp/m3_fp8 /tmp/m3_nvfp4

CUDA graphs are always disabled here: the dump hooks live in Python and a graph
replay would skip them, which would silently drop every decode step. Both runs
use greedy sampling and the same prompt, so with a correct KV cache the two
dumps should differ only by the KV quantization error.

Needs 4 GPUs (>=140 GB each) and the MSA kernels (fmha_sm100) available.
"""

from __future__ import annotations

import argparse
import os


def _gsm8k_prompt(index: int) -> str:
    from datasets import load_dataset

    dataset_dir = f"{os.environ['LLM_MODELS_ROOT']}/datasets/openai/gsm8k"
    question = load_dataset(dataset_dir, "main", split="test")[index]["question"]
    return f"Question: {question}\nAnswer:"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-dtype", required=True, choices=["auto", "fp8", "nvfp4"])
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--tokens-per-block", type=int, default=128)
    parser.add_argument("--model-path", default=None,
                        help="defaults to $LLM_MODELS_ROOT/MiniMax-M3-NVFP4")
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16,
                        help="decode steps to dump; keep small, one file per "
                             "step/layer/tag is written")
    parser.add_argument("--triton", action="store_true",
                        help="use the Triton sparse path instead of MSA")
    args = parser.parse_args()

    models_root = os.environ.get("LLM_MODELS_ROOT")
    if models_root is None:
        parser.error("LLM_MODELS_ROOT must be set")
    model_path = args.model_path or f"{models_root}/MiniMax-M3-NVFP4"

    prompt = _gsm8k_prompt(args.question_index)

    # Engine warmup issues a configuration-dependent number of dummy forwards
    # before the request, so dumping has to arm on the request's own prefill
    # rather than on a count from process start. Tokenize here because the env
    # is read at import time in the workers, before llm.tokenizer exists.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_tokens = len(tokenizer(prompt)["input_ids"])
    print(f"prompt is {prompt_tokens} tokens; arming the dump on that prefill")

    # Read at import time by the modeling and attention modules.
    os.environ["TRTLLM_M3_DUMP_DIR"] = args.dump_dir
    os.environ["TRTLLM_M3_DUMP_ARM_TOKENS"] = str(prompt_tokens)
    os.environ.setdefault("TRTLLM_M3_DUMP_MAX_STEPS", str(args.max_tokens + 1))
    os.environ.setdefault("TRTLLM_M3_MSA_DEBUG", "1")

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig, MiniMaxM3SparseAttentionConfig, MoeConfig

    use_msa = not args.triton
    llm = LLM(
        model_path,
        tensor_parallel_size=4,
        moe_expert_parallel_size=4,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.6,
            enable_block_reuse=False,
            dtype=args.kv_dtype,
            use_kv_cache_manager_v2=use_msa,
            tokens_per_block=args.tokens_per_block,
        ),
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation="msa" if use_msa else "triton",
            indexer_kv_dtype="fp8" if use_msa else "bf16",
            fuse_qkv_index_projection=use_msa,
        ),
        moe_config=MoeConfig(backend="CUTLASS"),
        cuda_graph_config=None,
        max_seq_len=4096,
        max_num_tokens=8192,
        max_batch_size=1,
        trust_remote_code=True,
    )
    with llm:
        output = llm.generate([prompt], SamplingParams(max_tokens=args.max_tokens,
                                                       temperature=0))[0]
        print(f"kv_dtype={args.kv_dtype} P={args.tokens_per_block} "
              f"kv_cache_quant_algo={llm.args.quant_config.kv_cache_quant_algo}")
        print(f"prompt:\n{prompt}")
        print(f"completion:\n{output.outputs[0].text!r}")
        print(f"token ids: {list(output.outputs[0].token_ids)}")
    print(f"\ndumps written to {args.dump_dir}")


if __name__ == "__main__":
    main()
