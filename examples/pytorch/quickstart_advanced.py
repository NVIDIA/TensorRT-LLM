import argparse

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.llmapi import (EagleDecodingConfig, KvCacheConfig,
                                 MTPDecodingConfig, NGramDecodingConfig)

example_prompts = [
    """Question: At a birthday party, 30% of the guests are married, 50% are single, and the rest are children. If there are 1000 guests, how many more married people are there than children?
    Answer: There are 1000 x 30/100 = <<1000*30/100=300>>300 people who are married.
    There are 1000 x 50/100 = <<1000*50/100=500>>500 people who are single.
    So, there are a total of 300 + 500 = <<300+500=800>>800 that are either married or single.
    This means, 1000 - 800 = <<1000-800=200>>200 are children.
    Therefore, there are 300 - 200 = <<300-200=100>>100 more married people than children.
    #### 100

    Question: Dulce's father has eight orange trees on his farm. If each tree has 200 fruits and Dulce picks 2/5 of the oranges from each tree, calculate the total number of fruits remaining in all the trees.
    Answer: The total number of oranges in all the trees before Dulce picked any is 8 trees * 200 oranges/tree = <<8*200=1600>>1600 oranges.
    If each orange tree has 200 oranges, and Dulce picks 2/5 of each tree's oranges, she picks 2/5 * 200 oranges = <<2/5*200=80>>80 oranges.
    Since the total number of orange trees is eight, Dulce picked 8 trees * 80 oranges/tree = <<8*80=640>>640 oranges from all the trees.
    After picking 640 oranges from the trees, the total number of oranges remaining became 1600 oranges - 640 oranges = 960 oranges.
    #### 960

    Question: Jorge has an equal number of baseball cards as Matias, who has 6 fewer cards than Carlos. If Carlos has 20 baseball cards, what is the total number of baseball cards the three have?
    Answer: If Carlos has 20 cards, Matias has 20-6 = <<20-6=14>>14 baseball cards.
    Since Matias and Jorge have the same amount of baseball cards, the total number of cards the three have is 14+14+20 =<<14+14+20=48>>48 cards.
    #### 48

    Question: At the end of the first quarter, the winning team had double the points of the losing team. At the end of the second quarter, the winning team had 10 more points than it started with. At the end of the third quarter, the winning team had 20 more points than the number it had in the second quarter. If the total points the winning team scored in the game was 80, and the losing team had 10 points in the first quarter, calculate the total number of points the team scored in the fourth quarter.
    Answer: At the end of the first quarter, the winning team had double the points of the losing team, meaning the winning team had already scored 10*2=<<10*2=20>>20 points.
    At the end of the second quarter, the winning team had 10 more points than it started with, a total of 20+10=30 points.
    At the end of the third quarter, the winning team had 20 more points than the number it had in the second quarter, a total of 20+30=<<20+30=50>>50 points.
    If the total points the winning team scored in the game was 80, they scored 80-50=<<80-50=30>>30 points in the fourth quarter.
    #### 30

    Question: The number of short students in a class is 2/5 of the total number of students. If there are 90 tall students, and the class has 400 students, calculate the total number of students with average height.
    Answer: The number of short students in the class is 2/5*400 = <<2/5*400=160>>160
    The total number of short and tall students is 160+90 = <<160+90=250>>250
    Since there are 400 students in the class, the number of students with average height is 400-250 = <<400-250=150>>150
    #### 150

    """ * 16 +
    """Question: Sophia and Rose went together to the market to buy onions and potatoes. Rose bought 4 times the number of onions and potatoes Sophia bought. If Rose bought 12 onions and 4 potatoes, how many onions and potatoes in total did Sophia buy at the market?
    Answer:
    """
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
                        choices=['CUTLASS', 'TRTLLM'])
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
    parser.add_argument('--eagle_model_dir', type=str, default=None)
    parser.add_argument('--max_matching_ngram_size', type=int, default=5)

    # Relaxed acceptance
    parser.add_argument('--use_relaxed_acceptance_for_thinking',
                        default=False,
                        action='store_true')
    parser.add_argument('--relaxed_topk', type=int, default=1)
    parser.add_argument('--relaxed_delta', type=float, default=0.)

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
        spec_config = MTPDecodingConfig(
            num_nextn_predict_layers=args.spec_decode_nextn,
            use_relaxed_acceptance_for_thinking=args.
            use_relaxed_acceptance_for_thinking,
            relaxed_topk=args.relaxed_topk,
            relaxed_delta=args.relaxed_delta)
    elif spec_decode_algo == "EAGLE3":
        spec_config = EagleDecodingConfig(
            max_draft_len=args.spec_decode_nextn,
            pytorch_eagle_weights_path=args.eagle_model_dir)
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

    llm = LLM(model=args.model_dir,
              backend='pytorch',
              disable_overlap_scheduler=args.disable_overlap_scheduler,
              kv_cache_dtype=args.kv_cache_dtype,
              kv_cache_config=kv_cache_config,
              attn_backend=args.attention_backend,
              use_cuda_graph=args.use_cuda_graph,
              load_format=args.load_format,
              print_iter_log=args.print_iter_log,
              enable_iter_perf_stats=args.print_iter_log,
              torch_compile_enabled=args.use_torch_compile,
              torch_compile_piecewise_cuda_graph=args.use_piecewise_cuda_graph,
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
              speculative_config=spec_config)

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
