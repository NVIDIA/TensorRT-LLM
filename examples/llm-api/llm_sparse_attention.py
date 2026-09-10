# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
### :title Sparse Attention
### :order 5
### :section Customization
r"""
This example demonstrates how to use sparse attention with TensorRT-LLM.

Supported sparse attention algorithms:
- RocketKV
- DSA

Usage:
```bash
python llm_sparse_attention.py \
    --model_path nvidia/Llama-3.1-8B-Instruct-FP8 \
    --algo ROCKETKV \
    --attention_backend TRTLLM \
    --window_size 32 \
    --kernel_size 63 \
    --prompt_budget 2048
```

When ``--input_file`` is omitted, the example uses a built-in
needle-in-a-haystack prompt that exceeds the default ``prompt_budget`` and
exercises the sparse attention path.
"""
import argparse
import json

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (CudaGraphConfig, DeepSeekSparseAttentionConfig,
                                 KvCacheConfig, MoeConfig,
                                 RocketSparseAttentionConfig)

# The built-in prompt follows a self-contained needle-in-a-haystack layout:
# 1. Cycle routine expedition log templates to form a deterministic haystack.
# 2. Replace one numbered log entry with a unique access-code needle.
# 3. Append a question that asks the model to retrieve the needle.
# With the default model, 128 entries produce about 2.8K tokens, exceeding the
# default 2,048-token prompt budget so that sparse attention is exercised.
_DEFAULT_LOG_TEMPLATES = (
    "The survey team checked the northern weather station and recorded normal "
    "temperature and pressure readings.",
    "Technicians inspected backup batteries, radio transmitters, and emergency "
    "lighting; all systems passed routine checks.",
    "Researchers cataloged soil samples, labeled storage containers, and "
    "updated the expedition inventory.",
    "The navigation group reviewed trail maps, satellite images, and the next "
    "day's travel schedule.",
    "At sunset, the field crew secured scientific instruments and uploaded the "
    "day's measurements.",
    "The medical officer reviewed first-aid supplies and confirmed the "
    "evacuation plan with base camp.",
    "Engineers calibrated wind sensors and verified timestamps against the "
    "observatory master clock.",
    "The logistics team counted food, water, fuel, and spare parts before "
    "closing the storage area.",
)
_DEFAULT_NUM_LOG_ENTRIES = 128
_DEFAULT_NEEDLE_INDEX = 87
_DEFAULT_ACCESS_CODE = "314159"


def _build_default_prompt():
    log_entries = []
    for index in range(_DEFAULT_NUM_LOG_ENTRIES):
        if index == _DEFAULT_NEEDLE_INDEX:
            message = (f"The expedition access code is {_DEFAULT_ACCESS_CODE}. "
                       "Remember this code for the final question.")
        else:
            message = _DEFAULT_LOG_TEMPLATES[index %
                                             len(_DEFAULT_LOG_TEMPLATES)]
        log_entries.append(f"Log entry {index}: {message}")

    return (
        "Read the following expedition field log carefully. One entry contains "
        "an access code that you will need to recall.\n\n" +
        "\n".join(log_entries) + "\n\nWhat is the expedition access code?")


DEFAULT_PROMPTS = [_build_default_prompt()]


def read_input(input_file):
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            ret = json.loads(line)
            results.append(ret)
    return results


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        default="nvidia/Llama-3.1-8B-Instruct-FP8",
                        help="The local path or Hugging Face ID of the model.")
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help="Optional path to a JSONL input file. The built-in "
        "long prompt is used when omitted.")

    # Build config
    parser.add_argument('--algo',
                        type=str,
                        default='ROCKETKV',
                        choices=['ROCKETKV', 'DSA'])
    parser.add_argument('--attention_backend',
                        type=str,
                        default='TRTLLM',
                        choices=['VANILLA', 'TRTLLM'])

    # RocketKV config
    parser.add_argument('--window_size',
                        type=int,
                        default=32,
                        help="The window size for RocketKV.")
    parser.add_argument('--kernel_size',
                        type=int,
                        default=63,
                        help="The kernel size for RocketKV.")
    parser.add_argument('--prompt_budget',
                        type=int,
                        default=2048,
                        help="The prompt budget for RocketKV.")
    parser.add_argument('--topk',
                        type=int,
                        default=64,
                        help='Top-k for RocketKV')
    parser.add_argument('--kt_cache_dtype',
                        type=str,
                        default='float8_e5m2',
                        choices=['bfloat16', 'float8_e5m2'])
    parser.add_argument('--index_max_chunk_size',
                        type=int,
                        default=32768,
                        help="The maximum chunk size for the indexer.")
    parser.add_argument("--max_seq_len",
                        type=int,
                        default=10240,
                        help="The maximum sequence length.")
    parser.add_argument("--max_batch_size",
                        type=int,
                        default=256,
                        help="The maximum batch size.")
    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=128,
                        help="The maximum new tokens.")
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=81920,
        help=
        "The maximum total tokens (context + generation) across all sequences in a batch."
    )

    # Parallelism
    parser.add_argument('--moe_backend',
                        type=str,
                        default='CUTLASS',
                        choices=[
                            'CUTLASS', 'TRTLLM', 'VANILLA', 'DEEPGEMM',
                            'CUTEDSL', 'TRITON'
                        ])
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--moe_ep_size', type=int, default=-1)
    parser.add_argument('--enable_attention_dp',
                        default=False,
                        action='store_true')

    # KV cache
    parser.add_argument('--kv_cache_dtype', type=str, default='auto')
    parser.add_argument("--kv_cache_fraction", type=float, default=0.7)
    parser.add_argument('--tokens_per_block', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=10)

    # Runtime
    parser.add_argument('--print_iter_log',
                        default=False,
                        action='store_true',
                        help='Print iteration logs during execution')
    parser.add_argument('--use_cuda_graph', default=False, action='store_true')
    parser.add_argument('--cuda_graph_padding_enabled',
                        default=False,
                        action='store_true')
    parser.add_argument('--cuda_graph_batch_sizes',
                        nargs='+',
                        type=int,
                        default=None)
    parser.add_argument('--enable_chunked_prefill',
                        default=False,
                        action='store_true',
                        help='Enable chunked prefill')
    args = parser.parse_args()
    return args


def run_llm(args, sparse_attention_config):
    if args.input_file is None:
        prompts = DEFAULT_PROMPTS
        reference = [None] * len(prompts)
    else:
        data = read_input(args.input_file)
        num_samples = args.num_samples if args.num_samples is not None else len(
            data)
        data = data[:num_samples]
        prompts = [{
            'prompt': sample['input_context'] + sample['input_query']
        } for sample in data]
        reference = [sample['outputs'] for sample in data]

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=
        False,  # sparse attention does not support kv cache reuse now
        free_gpu_memory_fraction=args.kv_cache_fraction,
        tokens_per_block=args.tokens_per_block,
        dtype=args.kv_cache_dtype,
    )

    cuda_graph_config = CudaGraphConfig(
        batch_sizes=args.cuda_graph_batch_sizes,
        enable_padding=args.cuda_graph_padding_enabled,
    ) if args.use_cuda_graph else None

    llm = LLM(
        model=args.model_path,
        backend='pytorch',
        kv_cache_config=kv_cache_config,
        attn_backend=args.attention_backend,
        sparse_attention_config=sparse_attention_config,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        max_num_tokens=args.max_num_tokens,
        tensor_parallel_size=args.tp_size,
        moe_expert_parallel_size=args.moe_ep_size,
        enable_attention_dp=args.enable_attention_dp,
        cuda_graph_config=cuda_graph_config,
        print_iter_log=args.print_iter_log,
        enable_iter_perf_stats=args.print_iter_log,
        moe_config=MoeConfig(backend=args.moe_backend),
        enable_chunked_prefill=args.enable_chunked_prefill,
    )

    sampling_params = SamplingParams(add_special_tokens=False,
                                     max_tokens=args.max_new_tokens,
                                     temperature=0.8,
                                     top_p=0.95)

    outputs = llm.generate(prompts, sampling_params)
    for idx, output in enumerate(outputs):
        result = f'Generated text: {output.outputs[0].text!r}'
        if reference[idx] is not None:
            result += f', ref: {reference[idx]}'
        print(result)


def run_RocketKV(args):
    sparse_attention_config = RocketSparseAttentionConfig(
        window_size=args.window_size,
        kernel_size=args.kernel_size,
        prompt_budget=args.prompt_budget,
        topk=args.topk,
        kt_cache_dtype=args.kt_cache_dtype,
    )
    run_llm(args, sparse_attention_config)


def run_DSA(args):
    sparse_attention_config = DeepSeekSparseAttentionConfig(
        indexer_max_chunk_size=args.index_max_chunk_size, )
    run_llm(args, sparse_attention_config)


def main():
    args = parse_arguments()
    if args.algo == 'ROCKETKV':
        run_RocketKV(args)
    elif args.algo == 'DSA':
        run_DSA(args)
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")


if __name__ == "__main__":
    main()
