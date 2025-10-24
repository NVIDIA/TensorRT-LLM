import argparse

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (AttentionDpConfig, AutoDecodingConfig,
                                 CudaGraphConfig, DraftTargetDecodingConfig,
                                 EagleDecodingConfig, KvCacheConfig, MoeConfig,
                                 MTPDecodingConfig, NGramDecodingConfig,
                                 TorchCompileConfig)

example_prompts = [
    "Hello, my name is",
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
                        choices=[
                            'CUTLASS', 'TRTLLM', 'VANILLA', 'WIDEEP',
                            'DEEPGEMM', 'CUTEDSL', 'TRITON'
                        ])
    parser.add_argument('--enable_attention_dp',
                        default=False,
                        action='store_true')
    parser.add_argument('--attention_dp_enable_balance',
                        default=False,
                        action='store_true')
    parser.add_argument('--attention_dp_time_out_iters', type=int, default=0)
    parser.add_argument('--attention_dp_batching_wait_iters',
                        type=int,
                        default=0)
    parser.add_argument('--sampler_type',
                        default="auto",
                        choices=["auto", "TorchSampler", "TRTLLMSampler"])
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--moe_ep_size', type=int, default=-1)
    parser.add_argument('--moe_tp_size', type=int, default=-1)
    parser.add_argument('--moe_cluster_size', type=int, default=-1)
    parser.add_argument(
        '--use_low_precision_moe_combine',
        default=False,
        action='store_true',
        help='Use low precision combine in MoE (only for NVFP4 quantization)')

    # KV cache
    parser.add_argument('--kv_cache_dtype', type=str, default='auto')
    parser.add_argument('--disable_kv_cache_reuse',
                        default=False,
                        action='store_true')

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
    parser.add_argument('--apply_chat_template',
                        default=False,
                        action='store_true')

    # Sampling
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument('--load_format', type=str, default='auto')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--best_of', type=int, default=None)
    parser.add_argument('--max_beam_width', type=int, default=1)

    # Speculative decoding
    parser.add_argument('--spec_decode_algo', type=str, default=None)
    parser.add_argument('--spec_decode_max_draft_len', type=int, default=1)
    parser.add_argument('--draft_model_dir', type=str, default=None)
    parser.add_argument('--max_matching_ngram_size', type=int, default=5)
    parser.add_argument('--use_one_model', default=False, action='store_true')
    parser.add_argument('--eagle_choices', type=str, default=None)
    parser.add_argument('--use_dynamic_tree',
                        default=False,
                        action='store_true')
    parser.add_argument('--dynamic_tree_max_topK', type=int, default=None)

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

    parser.add_argument('--additional_model_outputs',
                        type=str,
                        default=None,
                        nargs='+')

    return parser


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LLM models with the PyTorch workflow.")
    parser = add_llm_args(parser)
    parser.add_argument("--kv_cache_fraction", type=float, default=0.9)
    args = parser.parse_args()
    return args


def setup_llm(args, **kwargs):
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=not args.disable_kv_cache_reuse,
        free_gpu_memory_fraction=args.kv_cache_fraction,
        dtype=args.kv_cache_dtype,
    )

    spec_decode_algo = args.spec_decode_algo.upper(
    ) if args.spec_decode_algo is not None else None

    if spec_decode_algo == 'MTP':
        if not args.use_one_model:
            print("Running MTP eagle with two model style.")
        spec_config = MTPDecodingConfig(
            num_nextn_predict_layers=args.spec_decode_max_draft_len,
            use_relaxed_acceptance_for_thinking=args.
            use_relaxed_acceptance_for_thinking,
            relaxed_topk=args.relaxed_topk,
            relaxed_delta=args.relaxed_delta,
            mtp_eagle_one_model=args.use_one_model,
            speculative_model_dir=args.model_dir)
    elif spec_decode_algo == "EAGLE3":
        spec_config = EagleDecodingConfig(
            max_draft_len=args.spec_decode_max_draft_len,
            speculative_model_dir=args.draft_model_dir,
            eagle3_one_model=args.use_one_model,
            eagle_choices=args.eagle_choices,
            use_dynamic_tree=args.use_dynamic_tree,
            dynamic_tree_max_topK=args.dynamic_tree_max_topK)
    elif spec_decode_algo == "DRAFT_TARGET":
        spec_config = DraftTargetDecodingConfig(
            max_draft_len=args.spec_decode_max_draft_len,
            speculative_model_dir=args.draft_model_dir)
    elif spec_decode_algo == "NGRAM":
        spec_config = NGramDecodingConfig(
            max_draft_len=args.spec_decode_max_draft_len,
            max_matching_ngram_size=args.max_matching_ngram_size,
            is_keep_all=True,
            is_use_oldest=True,
            is_public_pool=True,
        )
    elif spec_decode_algo == "AUTO":
        spec_config = AutoDecodingConfig()
    else:
        spec_config = None

    cuda_graph_config = CudaGraphConfig(
        batch_sizes=args.cuda_graph_batch_sizes,
        enable_padding=args.cuda_graph_padding_enabled,
    ) if args.use_cuda_graph else None

    attention_dp_config = AttentionDpConfig(
        enable_balance=args.attention_dp_enable_balance,
        timeout_iters=args.attention_dp_time_out_iters,
        batching_wait_iters=args.attention_dp_batching_wait_iters,
    )

    llm = LLM(
        model=args.model_dir,
        backend='pytorch',
        disable_overlap_scheduler=args.disable_overlap_scheduler,
        kv_cache_config=kv_cache_config,
        attn_backend=args.attention_backend,
        cuda_graph_config=cuda_graph_config,
        load_format=args.load_format,
        print_iter_log=args.print_iter_log,
        enable_iter_perf_stats=args.print_iter_log,
        torch_compile_config=TorchCompileConfig(
            enable_fullgraph=args.use_torch_compile,
            enable_inductor=args.use_torch_compile,
            enable_piecewise_cuda_graph= \
                args.use_piecewise_cuda_graph)
        if args.use_torch_compile else None,
        moe_config=MoeConfig(backend=args.moe_backend, use_low_precision_moe_combine=args.use_low_precision_moe_combine),
        sampler_type=args.sampler_type,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        enable_attention_dp=args.enable_attention_dp,
        attention_dp_config=attention_dp_config,
        tensor_parallel_size=args.tp_size,
        pipeline_parallel_size=args.pp_size,
        moe_expert_parallel_size=args.moe_ep_size,
        moe_tensor_parallel_size=args.moe_tp_size,
        moe_cluster_parallel_size=args.moe_cluster_size,
        enable_chunked_prefill=args.enable_chunked_prefill,
        speculative_config=spec_config,
        trust_remote_code=args.trust_remote_code,
        gather_generation_logits=args.return_generation_logits,
        max_beam_width=args.max_beam_width,
        **kwargs)

    use_beam_search = args.max_beam_width > 1
    best_of = args.best_of or args.n
    if use_beam_search:
        if args.n == 1 and args.best_of is None:
            args.n = args.max_beam_width
        assert best_of <= args.max_beam_width, f"beam width: {best_of}, should be less or equal to max_beam_width: {args.max_beam_width}"

    assert best_of >= args.n, f"In sampling mode best_of value: {best_of} should be less or equal to n: {args.n}"

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_context_logits=args.return_context_logits,
        return_generation_logits=args.return_generation_logits,
        logprobs=args.logprobs,
        n=args.n,
        best_of=best_of,
        use_beam_search=use_beam_search,
        additional_model_outputs=args.additional_model_outputs)
    return llm, sampling_params


def main():
    args = parse_arguments()
    prompts = args.prompt if args.prompt else example_prompts

    llm, sampling_params = setup_llm(args)
    new_prompts = []
    if args.apply_chat_template:
        for prompt in prompts:
            messages = [{"role": "user", "content": f"{prompt}"}]
            new_prompts.append(
                llm.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True))
        prompts = new_prompts
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        for sequence_idx, sequence in enumerate(output.outputs):
            generated_text = sequence.text
            # Skip printing the beam_idx if no beam search was used
            sequence_id_text = f"[{sequence_idx}]" if args.max_beam_width > 1 or args.n > 1 else ""
            print(
                f"[{i}]{sequence_id_text} Prompt: {prompt!r}, Generated text: {generated_text!r}"
            )
            if args.return_context_logits:
                print(
                    f"[{i}]{sequence_id_text} Context logits: {output.context_logits}"
                )
            if args.return_generation_logits:
                print(
                    f"[{i}]{sequence_id_text} Generation logits: {sequence.generation_logits}"
                )
            if args.logprobs:
                print(f"[{i}]{sequence_id_text} Logprobs: {sequence.logprobs}")

            if args.additional_model_outputs:
                for output_name in args.additional_model_outputs:
                    if sequence.additional_context_outputs:
                        print(
                            f"[{i}]{sequence_id_text} Context {output_name}: {sequence.additional_context_outputs[output_name]}"
                        )
                    print(
                        f"[{i}]{sequence_id_text} Generation {output_name}: {sequence.additional_generation_outputs[output_name]}"
                    )


if __name__ == '__main__':
    main()
