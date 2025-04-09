import argparse
import json
import os
import time
from difflib import SequenceMatcher

import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def dump_jsonl(data, fname):
    dname = os.path.dirname(fname)
    if not os.path.exists(dname):
        os.makedirs(dname)

    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        default="./Llama-3-8B-Instruct-Gradient-1048k")
    parser.add_argument('--input_file',
                        type=str,
                        default="./niah_single_2_seq16384_sample20.jsonl")
    parser.add_argument('--num_procs', type=int, default=1)
    parser.add_argument('--sa_block_size', type=int, default=32768)
    parser.add_argument('--sa_anchor_size', type=int, default=32768)
    parser.add_argument('--output_file',
                        type=str,
                        default="./outputs/niah_single_2.jsonl")
    parser.add_argument('--tensor_parallel_size', type=int, default=1)

    parser.add_argument('--max_input_len', type=int, default=512 * 1024)
    parser.add_argument('--max_seq_len', type=int, default=(512 + 8) * 1024)
    parser.add_argument('--max_batch_size', type=int, default=1)
    parser.add_argument('--max_num_tokens', type=int, default=(256 + 8) * 1024)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--num_kv_cache_max_tokens', type=int, default=270336)
    parser.add_argument('--num_samples', type=int, default=None)

    args = parser.parse_args()
    return args


def similarity_score(a, b):
    "similar compare a and b "
    return SequenceMatcher(None, a, b).ratio()


# Generate the outputs using either TRT or PyTorch (based on the use_pytorch argument). It’s the same function for both workflows.
def generate_llm_outputs(args,
                         prompts,
                         backend=None,
                         fp8=False,
                         fp8_kv_cache=False):
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                               kv_cache_quant_algo=QuantAlgo.FP8 if fp8_kv_cache
                               else None) if fp8 else QuantConfig()
    cp_config = {
        "cp_type": "star_attention",
        "cp_anchor_size": args.sa_anchor_size,
        "block_size": args.sa_block_size
    }

    pytorch_backend_config = PyTorchConfig(
        attn_backend='FLASHINFER_STAR_ATTENTION')
    llm = LLM(model=args.model_path,
              max_batch_size=args.max_batch_size,
              max_input_len=args.max_input_len,
              max_seq_len=args.max_seq_len,
              max_num_tokens=args.max_num_tokens,
              quant_config=quant_config,
              tensor_parallel_size=1,
              context_parallel_size=args.num_procs,
              cp_config=cp_config,
              pytorch_backend_config=pytorch_backend_config)

    sampling_params = SamplingParams(add_special_tokens=False,
                                     max_tokens=args.max_new_tokens)
    for prompt in [prompts[0]]:
        context = prompt['input_context']
        query = prompt['input_query']
        output = llm.generate(context,
                              queries=query,
                              use_tqdm=False,
                              sampling_params=sampling_params)
    print(f'[StarAttention] LLM warmup done')

    results, contexts, queries = [], [], []

    num_samples = args.num_samples if args.num_samples is not None else len(
        prompts)
    prompts = prompts[:num_samples]

    for prompt in prompts:
        contexts.append(prompt['input_context'])
        queries.append(prompt['input_query'])

    t0 = time.time()
    outputs = llm.generate(contexts,
                           queries=queries,
                           use_tqdm=True,
                           sampling_params=sampling_params)
    t1 = time.time()
    eg_count = 0
    for prompt, output in zip(prompts, outputs):
        eg = prompt
        ret = {
            'index': eg.get('index', -1),
            'pred': output.outputs[0].text,
            'input_context': eg['input_context'],
            'input_query': eg['input_query'],
            'outputs': (eg['outputs'] if 'outputs' in eg else [eg['output']]),
            'others': eg.get('others', {}),
            'truncation': eg.get('truncation', -1),
            'length': eg.get('length', -1),
        }
        results.append(ret)

        ctx_str = eg['input_context']
        pred = eg['outputs'][0]
        pred_index = ctx_str.index(pred)
        pred_pos = len(llm.tokenizer.encode(ctx_str[:pred_index]))
        print('------------------------')
        print(f'eg id = {eg_count}')
        print(f'magic_number_pos = {pred_pos} / ctx_len = {len(contexts)}')
        print(f'output = {output.outputs[0].text}')
        print(f'refernce = {pred}')
        eg_count += 1

    return results, t1 - t0


def read_input(input_file):
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            ret = json.loads(line)
            results.append(ret)
    return results


def main():
    args = parse_arguments()
    prompts = read_input(args.input_file)
    print('read data done')
    # Generate outputs using Pytorch.
    results, elapsed_time = generate_llm_outputs(args,
                                                 prompts,
                                                 backend='pytorch')
    torch.cuda.empty_cache()
    num_samples = args.num_samples if args.num_samples is not None else len(
        prompts)
    print(
        f'[StarAttention] Generate done, input files = {args.input_file}, samples = {num_samples}, total latency = {elapsed_time}s, seq average latency = {elapsed_time / num_samples}s'
    )
    print(
        f'StarAttention] Results file saved at {args.output_file}, please use ruler evaluator to summarize it'
    )
    dump_jsonl(results, args.output_file)


if __name__ == '__main__':
    main()
