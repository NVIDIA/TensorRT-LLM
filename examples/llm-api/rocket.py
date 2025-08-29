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
    parser.add_argument('--attention_backend',
                        type=str,
                        default='TRTLLM',
                        choices=[
                            'VANILLA', 'TRTLLM', 'FLASHINFER',
                            'FLASHINFER_STAR_ATTENTION'
                        ])
    parser.add_argument("--max_seq_len",
                        type=int,
                        default=8192,
                        help="The maximum sequence length.")
    parser.add_argument("--max_batch_size",
                        type=int,
                        default=4,
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

    parser.add_argument("--kv_cache_fraction", type=float, default=0.7)

    parser.add_argument('--num_samples', type=int, default=1)

    args = parser.parse_args()
    return args


example_prompts = [
    "John writes 20 pages a day. How long will it take him to write 3 books that are 400 pages each?",
    "John orders food for a massive restaurant. He orders 1000 pounds of beef for $8 per pound. He also orders twice that much chicken at $3 per pound. How much did everything cost?",
    "Sally and Bob have made plans to go on a trip at the end of the year. They both decide to work as babysitters and save half of what they've earned for their trip. If Sally makes $6 per day and Bob makes $4 per day, how much money will they both have saved for their trip after a year?",
]


def main():
    args = parse_arguments()
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
        window_size=32,
        kernel_size=63,
        prompt_budget=2048,
    )
    # Model could accept HF model name, a path to local HF model,
    # or TensorRT Model Optimizer's quantized checkpoints like nvidia/Llama-3.1-8B-Instruct-FP8 on HF.
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
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
    )

    # Sample prompts.
    # prompts = [
    #     "Hello, my name is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]

    prompts = []
    reference = []
    for sample in data:
        prompts.append(
            {'prompt': sample['input_context'] + sample['input_query']})
        reference.append(sample['outputs'])

    # Create a sampling params.
    sampling_params = SamplingParams(add_special_tokens=False,
                                     max_tokens=args.max_new_tokens,
                                     temperature=0.8,
                                     top_p=0.95)

    # prompts = example_prompts
    outputs = llm.generate(prompts, sampling_params)
    for idx, output in enumerate(outputs):
        print(
            # f'Prompt: {prompts[idx]}, Generated text: {output.outputs[0].text!r}'
            f'Generated text: {output.outputs[0].text!r}, ref: {reference[idx]}'
        )
    # Got output like
    # Prompt: 'Hello, my name is', Generated text: '\n\nJane Smith. I am a student pursuing my degree in Computer Science at [university]. I enjoy learning new things, especially technology and programming'
    # Prompt: 'The president of the United States is', Generated text: 'likely to nominate a new Supreme Court justice to fill the seat vacated by the death of Antonin Scalia. The Senate should vote to confirm the'
    # Prompt: 'The capital of France is', Generated text: 'Paris.'
    # Prompt: 'The future of AI is', Generated text: 'an exciting time for us. We are constantly researching, developing, and improving our platform to create the most advanced and efficient model available. We are'


if __name__ == '__main__':
    main()
