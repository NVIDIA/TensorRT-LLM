import argparse
import json
import os
import time
from difflib import SequenceMatcher

import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import CpType


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
    parser.add_argument('--max_batch_size', type=int, default=20)
    parser.add_argument('--max_num_tokens', type=int, default=(256 + 8) * 1024)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--num_kv_cache_max_tokens', type=int, default=270336)
    parser.add_argument('--num_samples', type=int, default=None)

    args = parser.parse_args()
    return args


def similarity_score(a, b):
    "similar compare a and b "
    return SequenceMatcher(None, a, b).ratio()


def generate_llm_outputs(args, data, fp8=False, fp8_kv_cache=False):
    kv_cache_config = KvCacheConfig(dtype="fp8" if fp8_kv_cache else "auto")
    cp_config = {
        "cp_type": CpType.STAR,
        "cp_anchor_size": args.sa_anchor_size,
        "block_size": args.sa_block_size
    }

    llm = LLM(model=args.model_path,
              max_batch_size=args.max_batch_size,
              max_input_len=args.max_input_len,
              max_seq_len=args.max_seq_len,
              max_num_tokens=args.max_num_tokens,
              kv_cache_config=kv_cache_config,
              tensor_parallel_size=1,
              context_parallel_size=args.num_procs,
              cp_config=cp_config,
              attn_backend='FLASHINFER_STAR_ATTENTION')

    sampling_params = SamplingParams(add_special_tokens=False,
                                     max_tokens=args.max_new_tokens)
    for sample in data[:1]:
        inputs = {
            'prompt': sample['input_context'],
            'query': sample['input_query']
        }
        output = llm.generate(inputs,
                              use_tqdm=False,
                              sampling_params=sampling_params)
    print(f'[StarAttention] LLM warmup done')

    results, inputs = [], []

    num_samples = args.num_samples if args.num_samples is not None else len(
        data)
    data = data[:num_samples]

    for sample in data:
        inputs.append({
            'prompt': sample['input_context'],
            'query': sample['input_query']
        })

    t0 = time.time()
    outputs = llm.generate(inputs,
                           use_tqdm=True,
                           sampling_params=sampling_params)
    t1 = time.time()
    eg_count = 0
    for eg, output in zip(data, outputs):
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
        ctx_len = len(llm.tokenizer.encode(ctx_str))
        pred = eg['outputs'][0]
        pred_index = ctx_str.index(pred)
        pred_pos = len(llm.tokenizer.encode(ctx_str[:pred_index]))
        print('------------------------')
        print(f'eg id = {eg_count}')
        print(f'magic_number_pos = {pred_pos} / ctx_len = {ctx_len}')
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
    data = read_input(args.input_file)
    print('read data done')
    # Generate outputs using Pytorch.
    results, elapsed_time = generate_llm_outputs(args, data)
    torch.cuda.empty_cache()
    num_samples = args.num_samples if args.num_samples is not None else len(
        data)
    print(
        f'[StarAttention] Generate done, input files = {args.input_file}, samples = {num_samples}, total latency = {elapsed_time}s, seq average latency = {elapsed_time / num_samples}s'
    )
    print(
        f'StarAttention] Results file saved at {args.output_file}, please use ruler evaluator to summarize it'
    )
    dump_jsonl(results, args.output_file)


if __name__ == '__main__':
    main()
