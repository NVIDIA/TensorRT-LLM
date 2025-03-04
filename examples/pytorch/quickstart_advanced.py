import argparse

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.llmapi import MTPDecodingConfig

example_prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def add_llm_args(parser):
    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        help="Model checkpoint directory.")
    parser.add_argument("--prompt",
                        type=str,
                        nargs="+",
                        help="A single or a list of text prompts.")

    # Parallelism
    parser.add_argument('--attention_backend',
                        type=str,
                        default='TRTLLM',
                        choices=[
                            'VANILLA', 'TRTLLM', 'FLASHINFER',
                            'FLASHINFER_STAR_ATTENTION'
                        ])
    parser.add_argument('--enable_attention_dp',
                        default=False,
                        action='store_true')
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--moe_ep_size', type=int, default=-1)
    parser.add_argument('--moe_tp_size', type=int, default=-1)

    # KV cache
    parser.add_argument('--kv_cache_dtype', type=str, default='auto')
    parser.add_argument('--kv_cache_enable_block_reuse',
                        default=True,
                        action='store_false')
    parser.add_argument("--kv_cache_fraction", type=float, default=None)

    # Runtime
    parser.add_argument('--enable_overlap_scheduler',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_chunked_prefill',
                        default=False,
                        action='store_true')
    parser.add_argument('--use_cuda_graph', default=False, action='store_true')
    parser.add_argument('--print_iter_log',
                        default=False,
                        action='store_true',
                        help='Print iteration logs during execution')

    # Sampling
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument('--load_format', type=str, default='auto')

    # Speculative decoding
    parser.add_argument('--mtp_nextn', type=int, default=0)
    return parser


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LLM models with the PyTorch workflow.")
    parser = add_llm_args(parser)
    args = parser.parse_args()
    return args


def setup_llm(args):
    pytorch_config = PyTorchConfig(
        enable_overlap_scheduler=args.enable_overlap_scheduler,
        kv_cache_dtype=args.kv_cache_dtype,
        attn_backend=args.attention_backend,
        use_cuda_graph=args.use_cuda_graph,
        load_format=args.load_format,
        print_iter_log=args.print_iter_log,
    )

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=args.kv_cache_enable_block_reuse,
        free_gpu_memory_fraction=args.kv_cache_fraction,
    )

    mtp_config = MTPDecodingConfig(
        num_nextn_predict_layers=args.mtp_nextn) if args.mtp_nextn > 0 else None

    llm = LLM(
        model=args.model_dir,
        pytorch_backend_config=pytorch_config,
        kv_cache_config=kv_cache_config,
        tensor_parallel_size=args.tp_size,
        enable_attention_dp=args.enable_attention_dp,
        moe_expert_parallel_size=args.moe_ep_size,
        moe_tensor_parallel_size=args.moe_tp_size,
        enable_chunked_prefill=args.enable_chunked_prefill,
        speculative_config=mtp_config,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    return llm, sampling_params


def main():
    args = parse_arguments()
    prompts = args.prompt if args.prompt else example_prompts

    llm, sampling_params = setup_llm(args)
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
