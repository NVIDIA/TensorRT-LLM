import argparse
import json

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, RocketSparseAttentionConfig


def read_input(input_file):
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            ret = json.loads(line)
            results.append(ret)
    return results


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=
        "/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default="tests/unittest/_torch/multi_gpu/test_star_attention_input.jsonl"
    )
    # Build config
    parser.add_argument('--algo',
                        type=str,
                        default='ROCKETKV',
                        choices=['ROCKETKV'])
    parser.add_argument('--attention_backend',
                        type=str,
                        default='TRTLLM',
                        choices=['VANILLA', 'TRTLLM'])
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
    parser.add_argument("--max_seq_len",
                        type=int,
                        default=8192,
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
        default=8192,
        help=
        "The maximum total tokens (context + generation) across all sequences in a batch."
    )
    parser.add_argument('--tensor_parallel_size', type=int, default=1)

    # KV cache
    parser.add_argument('--kv_cache_dtype', type=str, default='auto')
    parser.add_argument("--kv_cache_fraction", type=float, default=None)
    parser.add_argument('--num_samples', type=int, default=10)

    args = parser.parse_args()
    return args


def run_RocketKV(args):
    data = read_input(args.input_file)
    num_samples = args.num_samples if args.num_samples is not None else len(
        data)
    data = data[:num_samples]

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=
        False,  # sparse attention does not support kv cache reuse now
        free_gpu_memory_fraction=args.kv_cache_fraction,
        dtype=args.kv_cache_dtype,
    )
    sparse_attention_config = RocketSparseAttentionConfig(
        window_size=args.window_size,
        kernel_size=args.kernel_size,
        prompt_budget=args.prompt_budget,
    )

    llm = LLM(
        model=args.model_path,
        backend='pytorch',
        kv_cache_config=kv_cache_config,
        attn_backend=args.attention_backend,
        sparse_attention_config=sparse_attention_config,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        max_num_tokens=args.max_num_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        cuda_graph_config=
        None,  # sparse attention does not support cuda graph now
    )

    prompts = []
    reference = []
    for sample in data:
        prompts.append(
            {'prompt': sample['input_context'] + sample['input_query']})
        reference.append(sample['outputs'])

    sampling_params = SamplingParams(add_special_tokens=False,
                                     max_tokens=args.max_new_tokens,
                                     temperature=0.8,
                                     top_p=0.95)

    outputs = llm.generate(prompts, sampling_params)
    for idx, output in enumerate(outputs):
        print(
            f'Generated text: {output.outputs[0].text!r}, ref: {reference[idx]}'
        )


def main():
    args = parse_arguments()
    if args.algo == 'ROCKETKV':
        run_RocketKV(args)
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")


if __name__ == "__main__":
    main()
