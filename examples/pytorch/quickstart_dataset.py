import argparse
import json

from datasets import load_dataset

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import (EagleDecodingConfig, KvCacheConfig,
                                 MTPDecodingConfig, NGramDecodingConfig)

example_prompts = [
    "system: You are a friendly chatbot who always responds in the style of a pirate\nuser: Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.\nassistant:"
]


class DatasetBase:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_dataset(self, size):
        pass

    def convert_to_chat_format(self, messages):
        formatted_text = ""
        for message in self.system_prompt + messages:
            role = message.get('role', '')
            content = message.get('content', '')
            formatted_text += f"{role}:{content}\n"
        formatted_text += "assistant:"
        return formatted_text


class MTBenchDataset(DatasetBase):

    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.system_prompt = [{
            "role":
            "system",
            "content":
            "You are a friendly chatbot who always responds in the style of a pirate",
        }]
        self.max_turn = 2

    def load_dataset(self, size):
        source = "lmsys/mt_bench_human_judgments"

        unique = []
        dataset = load_dataset(source, split='human')
        full_size = len(dataset)
        if size is None:
            size = full_size
        if size > full_size:
            size = full_size

        # Filter out the
        user_messages = []
        for i in range(len(dataset)):
            if dataset[i]['conversation_a'][0]['content'] not in unique:
                unique.append(dataset[i]['conversation_a'][0]['content'])
                user_messages.append([
                    data for data in dataset[i]['conversation_a']
                    if data['role'] == 'user'
                ])
        return user_messages[:size]

    def convert_to_chat_format(self, messages):
        formatted_text = ""
        for message in self.system_prompt + messages:
            role = message.get('role', '')
            content = message.get('content', '')
            if role is not None and content is not None:
                formatted_text += f"{role}:{content}\n"
        formatted_text += "assistant:"
        return formatted_text


class AADataset(DatasetBase):

    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.max_turn = 1
        self.system_prompt = [{
            "role": "system",
            "content": "You are a helpful assistant.",
        }]

    def load_dataset(self, size):
        file = "/home/scratch.tjohnsen_ent/datasets/AA_dataset.txt"
        with open(file, 'r') as f:
            data = f.readlines()

        data = [[{
            "role": "user",
            "content": json.loads(item).get('user_prompt')
        }] for item in data]
        return data[:size]

    def convert_to_chat_format(self, messages):
        formatted_text = ""
        for message in self.system_prompt + messages:
            role = message.get('role', '')
            content = message.get('content', '')
            formatted_text += f"{role}:{content}\n"
        formatted_text += "assistant:"
        return formatted_text


class MagpieDataset(DatasetBase):

    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.max_turn = 1
        self.system_prompt = [dict()]

    def load_dataset(self, size):
        source = "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered"
        dataset = load_dataset(source, split='train')
        full_size = len(dataset)
        turns = []
        conversations = []
        for data in dataset:
            original_conversations = data.get('conversations', [])
            processed_conversations = []
            turn = 0
            for conv in original_conversations:
                match conv.get('from'):
                    case 'human':
                        processed_conversations.append({
                            "role":
                            "user",
                            "content":
                            conv.get('value')
                        })
                        turn += 1
                    case 'gpt':
                        processed_conversations.append({
                            "role":
                            "assistant",
                            "content":
                            conv.get('value')
                        })
                    case _:
                        raise ValueError(f"Unknown role: {conv}")

            conversations.append(processed_conversations)
            turns.append(turn)

        # import pdb; pdb.set_trace()

        if size is None:
            size = full_size
        if size > full_size:
            size = full_size

        self.max_turn = max(turns[:size])

        print(f"Max turn: {self.max_turn}")
        return conversations[:size]


def add_llm_args(parser):
    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        help="Model checkpoint directory.")
    parser.add_argument("--prompt",
                        type=str,
                        nargs="+",
                        help="A single or a list of text prompts.")
    parser.add_argument("--dataset",
                        type=str,
                        choices=['mtbench', 'aa', 'magpie'],
                        help="The name of the dataset to use.")
    parser.add_argument("--dataset_size",
                        type=int,
                        default=None,
                        help="The size of the dataset to use.")
    parser.add_argument("--max_turn",
                        type=int,
                        default=None,
                        help="The maximum number of turns to use.")
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
    parser.add_argument('--enable_trtllm_decoder',
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
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument('--load_format', type=str, default='auto')
    parser.add_argument('--return_generation_logits',
                        default=False,
                        action='store_true')
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
    pytorch_config = PyTorchConfig(
        disable_overlap_scheduler=args.disable_overlap_scheduler,
        kv_cache_dtype=args.kv_cache_dtype,
        attn_backend=args.attention_backend,
        use_cuda_graph=args.use_cuda_graph,
        load_format=args.load_format,
        print_iter_log=args.print_iter_log,
        enable_iter_perf_stats=args.print_iter_log,
        torch_compile_enabled=args.use_torch_compile,
        torch_compile_piecewise_cuda_graph=args.use_piecewise_cuda_graph,
        moe_backend=args.moe_backend,
        enable_trtllm_decoder=args.enable_trtllm_decoder)

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
              max_seq_len=args.max_seq_len,
              max_batch_size=args.max_batch_size,
              max_num_tokens=args.max_num_tokens,
              pytorch_backend_config=pytorch_config,
              kv_cache_config=kv_cache_config,
              tensor_parallel_size=args.tp_size,
              pipeline_parallel_size=args.pp_size,
              enable_attention_dp=args.enable_attention_dp,
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
        exclude_input_from_output=True,
        return_perf_metrics=True,
    )
    return llm, sampling_params


def main():
    args = parse_arguments()
    llm, sampling_params = setup_llm(args)

    if not args.dataset:
        print("No dataset provided, using example prompts")
        model_prompts = example_prompts
        outputs = llm.generate(model_prompts, sampling_params)

        for i, output in enumerate(outputs):
            print(
                f"[{i}] Prompt: {output.prompt!r}\nGenerated text: {output.outputs[0].text!r}"
            )
        return

    dataset = None
    match args.dataset:
        case 'mtbench':
            dataset = MTBenchDataset(args.dataset)
        case 'aa':
            dataset = AADataset(args.dataset)
        case 'magpie':
            dataset = MagpieDataset(args.dataset)

    # data_items = batch x conversation
    # conversation = turn x message
    data_items = dataset.load_dataset(args.dataset_size)
    current_turn = 1
    assistant_outputs = [[] for _ in range(len(data_items))]
    model_prompts = []

    while current_turn <= dataset.max_turn and (args.max_turn is None or
                                                current_turn <= args.max_turn):
        for idx, conversation in enumerate(data_items):
            assemble_messages = []
            for i in range(current_turn):
                assemble_messages.append(conversation[i])
                if i < (current_turn - 1):
                    assemble_messages.append({
                        'role':
                        'assistant',
                        'content':
                        assistant_outputs[idx][i]
                    })
            tokenized_prompts = dataset.convert_to_chat_format(
                assemble_messages)

            model_prompts.append(tokenized_prompts)
        outputs = llm.generate(model_prompts, sampling_params)
        model_prompts = []
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(
                f"Turn {current_turn} - [{i}] Prompt: {prompt!r}\nGenerated text: {generated_text!r}"
            )
            assistant_outputs[i].append(generated_text)

        current_turn += 1

    if args.print_iter_log:
        stats = llm.get_stats()
        for stat in stats:
            print(stat)


if __name__ == '__main__':
    main()
