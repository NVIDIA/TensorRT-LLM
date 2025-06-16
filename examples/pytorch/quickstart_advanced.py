import argparse

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.llmapi import (DraftTargetDecodingConfig, EagleDecodingConfig,
                                 KvCacheConfig, MTPDecodingConfig,
                                 NGramDecodingConfig, TorchCompileConfig)

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
    # Build config
    parser.add_argument("--max_seq_len",
                        type=int,
                        default=None,
                        help="The maximum sequence length.")
    parser.add_argument("--max_batch_size",
                        type=int,
                        default=2048,
                        help="The maximum batch size.")
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=8192,
        help=
        "The maximum total tokens (context + generation) across all sequences in a batch."
    )

    # Parallelism
    parser.add_argument('--attention_backend',
                        type=str,
                        default='TRTLLM',
                        choices=[
                            'VANILLA', 'TRTLLM', 'FLASHINFER',
                            'FLASHINFER_STAR_ATTENTION'
                        ])
    parser.add_argument('--moe_backend',
                        type=str,
                        default='CUTLASS',
                        choices=['CUTLASS', 'TRTLLM', 'VANILLA'])
    parser.add_argument('--enable_attention_dp',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_trtllm_sampler',
                        default=False,
                        action='store_true')
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--moe_ep_size', type=int, default=-1)
    parser.add_argument('--moe_tp_size', type=int, default=-1)
    parser.add_argument('--moe_cluster_size', type=int, default=-1)

    # KV cache
    parser.add_argument('--kv_cache_dtype', type=str, default='auto')
    parser.add_argument('--disable_kv_cache_reuse',
                        default=False,
                        action='store_true')
    parser.add_argument("--kv_cache_fraction", type=float, default=None)

    # Runtime
    parser.add_argument('--disable_overlap_scheduler',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_chunked_prefill',
                        default=False,
                        action='store_true')
    parser.add_argument('--use_cuda_graph', default=False, action='store_true')
    parser.add_argument('--cuda_graph_padding_enabled',
                        default=False,
                        action='store_true')
    parser.add_argument('--cuda_graph_batch_sizes',
                        nargs='+',
                        type=int,
                        default=None)
    parser.add_argument('--print_iter_log',
                        default=False,
                        action='store_true',
                        help='Print iteration logs during execution')
    parser.add_argument('--use_torch_compile',
                        default=False,
                        action='store_true',
                        help='Use torch.compile to optimize the model')
    parser.add_argument('--use_piecewise_cuda_graph',
                        default=False,
                        action='store_true',
                        help='Use piecewise CUDA graph to optimize the model')

    # Sampling
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument('--load_format', type=str, default='auto')

    # Speculative decoding
    parser.add_argument('--spec_decode_algo', type=str, default=None)
    parser.add_argument('--spec_decode_nextn', type=int, default=1)
    parser.add_argument('--draft_model_dir',
                        '--eagle_model_dir',
                        type=str,
                        default=None)
    parser.add_argument('--max_matching_ngram_size', type=int, default=5)
    parser.add_argument('--use_one_model', default=False, action='store_true')

    # Relaxed acceptance
    parser.add_argument('--use_relaxed_acceptance_for_thinking',
                        default=False,
                        action='store_true')
    parser.add_argument('--relaxed_topk', type=int, default=1)
    parser.add_argument('--relaxed_delta', type=float, default=0.)

    # HF
    parser.add_argument('--trust_remote_code',
                        default=False,
                        action='store_true')
    parser.add_argument('--return_context_logits',
                        default=False,
                        action='store_true')
    parser.add_argument('--return_generation_logits',
                        default=False,
                        action='store_true')
    parser.add_argument('--logprobs', default=False, action='store_true')
    return parser


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LLM models with the PyTorch workflow.")
    parser = add_llm_args(parser)
    args = parser.parse_args()
    return args


def setup_llm(args):
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=not args.disable_kv_cache_reuse,
        free_gpu_memory_fraction=args.kv_cache_fraction,
    )

    spec_decode_algo = args.spec_decode_algo.upper(
    ) if args.spec_decode_algo is not None else None

    if spec_decode_algo == 'MTP':
        if not args.use_one_model:
            print(
                "MTP only supports one model style spec decode; ignoring default use_one_model=False"
            )

        spec_config = MTPDecodingConfig(
            num_nextn_predict_layers=args.spec_decode_nextn,
            use_relaxed_acceptance_for_thinking=args.
            use_relaxed_acceptance_for_thinking,
            relaxed_topk=args.relaxed_topk,
            relaxed_delta=args.relaxed_delta)
    elif spec_decode_algo == "EAGLE3":
        spec_config = EagleDecodingConfig(
            max_draft_len=args.spec_decode_nextn,
            pytorch_weights_path=args.draft_model_dir,
            eagle3_one_model=args.use_one_model)
    elif spec_decode_algo == "DRAFT_TARGET":
        spec_config = DraftTargetDecodingConfig(
            max_draft_len=args.spec_decode_nextn,
            pytorch_weights_path=args.draft_model_dir)
    elif spec_decode_algo == "NGRAM":
        spec_config = NGramDecodingConfig(
            prompt_lookup_num_tokens=args.spec_decode_nextn,
            max_matching_ngram_size=args.max_matching_ngram_size,
            is_keep_all=True,
            is_use_oldest=True,
            is_public_pool=True,
        )
    else:
        spec_config = None

    llm = LLM(
        model=args.model_dir,
        backend='pytorch',
        disable_overlap_scheduler=args.disable_overlap_scheduler,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_cache_config=kv_cache_config,
        attn_backend=args.attention_backend,
        use_cuda_graph=args.use_cuda_graph,
        cuda_graph_padding_enabled=args.cuda_graph_padding_enabled,
        cuda_graph_batch_sizes=args.cuda_graph_batch_sizes,
        load_format=args.load_format,
        print_iter_log=args.print_iter_log,
        enable_iter_perf_stats=args.print_iter_log,
        torch_compile_config=TorchCompileConfig(
            torch_compile_fullgraph=args.use_torch_compile,
            torch_compile_inductor_enabled=args.use_torch_compile,
            torch_compile_piecewise_cuda_graph=args.use_piecewise_cuda_graph)
        if args.use_torch_compile else None,
        moe_backend=args.moe_backend,
        enable_trtllm_sampler=args.enable_trtllm_sampler,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        enable_attention_dp=args.enable_attention_dp,
        tensor_parallel_size=args.tp_size,
        pipeline_parallel_size=args.pp_size,
        moe_expert_parallel_size=args.moe_ep_size,
        moe_tensor_parallel_size=args.moe_tp_size,
        moe_cluster_parallel_size=args.moe_cluster_size,
        enable_chunked_prefill=args.enable_chunked_prefill,
        speculative_config=spec_config,
        trust_remote_code=args.trust_remote_code,
        gather_generation_logits=args.return_generation_logits)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_context_logits=args.return_context_logits,
        return_generation_logits=args.return_generation_logits,
        logprobs=args.logprobs)
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

        if args.return_context_logits:
            print(f"[{i}] Context logits: {output.context_logits}")
        if args.return_generation_logits:
            print(
                f"[{i}] Generation logits: {output.outputs[0].generation_logits}"
            )
        if args.logprobs:
            print(f"[{i}] Logprobs: {output.outputs[0].logprobs}")


if __name__ == '__main__':
    main()
