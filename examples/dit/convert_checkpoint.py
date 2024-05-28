import argparse
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import safetensors.torch
import torch

import tensorrt_llm
from tensorrt_llm import str_dtype_to_torch
from tensorrt_llm.mapping import Mapping


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timm_ckpt',
                        type=str,
                        default="./DiT-XL-2-512x512.pt")
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument('--input_size',
                        type=int,
                        default=64,
                        help='The input latent size')
    parser.add_argument('--patch_size',
                        type=int,
                        default=2,
                        help='The patch size for patchify')
    parser.add_argument('--in_channels',
                        type=int,
                        default=4,
                        help='The channels of input latent')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=1152,
                        help='The hidden size of DiT')
    parser.add_argument('--depth',
                        type=int,
                        default=28,
                        help='The number of DiTBlock layers')
    parser.add_argument('--num_heads',
                        type=int,
                        default=16,
                        help='The number of heads of attention module')
    parser.add_argument(
        '--mlp_ratio',
        type=float,
        default=4.0,
        help=
        'The ratio of hidden size compared to input hidden size in MLP layer')
    parser.add_argument(
        '--class_dropout_prob',
        type=float,
        default=0.1,
        help='The probability to drop class token when training')
    parser.add_argument('--num_classes',
                        type=int,
                        default=1000,
                        help='The number of classes for conditional control')
    parser.add_argument('--learn_sigma',
                        type=bool,
                        default=True,
                        help='Whether the model learn sigma')
    parser.add_argument('--cfg_scale', type=float, default=4.0)
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    args = parser.parse_args()
    return args


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return torch.chunk(v, tp_size)[idx].contiguous()
    else:
        return torch.chunk(v, tp_size, dim=dim)[idx].clone()


def split_qkv_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    v = v.reshape(3, n_hidden, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel), n_hidden)
    return split_v.clone()


def split_qkv_bias_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV bias according to tensor parallelism
    """
    v = v.reshape(3, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel))
    return split_v.clone()


def split_matrix_tp(v, tensor_parallel, rank, dim):
    return split(v, tensor_parallel, rank, dim=dim)


def convert_timm_dit(args, mapping, dtype='float32'):

    weights = {}
    tik = time.time()
    torch_dtype = str_dtype_to_torch(dtype)
    tensor_parallel = mapping.tp_size
    model_params = dict(torch.load(args.timm_ckpt))
    timm_to_trtllm_name = {
        't_embedder.mlp.0.weight': 't_embedder.mlp1.weight',
        't_embedder.mlp.0.bias': 't_embedder.mlp1.bias',
        't_embedder.mlp.2.weight': 't_embedder.mlp2.weight',
        't_embedder.mlp.2.bias': 't_embedder.mlp2.bias',
        'blocks.(\d+).mlp.fc1.weight': 'blocks.*.mlp.fc.weight',
        'blocks.(\d+).mlp.fc1.bias': 'blocks.*.mlp.fc.bias',
        'blocks.(\d+).mlp.fc2.weight': 'blocks.*.mlp.proj.weight',
        'blocks.(\d+).mlp.fc2.bias': 'blocks.*.mlp.proj.bias',
        'blocks.(\d+).attn.proj.weight': 'blocks.*.attn.dense.weight',
        'blocks.(\d+).attn.proj.bias': 'blocks.*.attn.dense.bias',
        'blocks.(\d+).adaLN_modulation.1.weight':
        'blocks.*.adaLN_modulation.weight',
        'blocks.(\d+).adaLN_modulation.1.bias':
        'blocks.*.adaLN_modulation.bias',
        'final_layer.adaLN_modulation.1.weight':
        'final_layer.adaLN_modulation.weight',
        'final_layer.adaLN_modulation.1.bias':
        'final_layer.adaLN_modulation.bias'
    }

    def get_trtllm_name(timm_name):
        for k, v in timm_to_trtllm_name.items():
            m = re.match(k, timm_name)
            if m is not None:
                if "*" in v:
                    v = v.replace("*", m.groups()[0])
                return v
        return timm_name

    weights = dict()
    for name, param in model_params.items():
        weights[get_trtllm_name(name)] = param.contiguous().to(torch_dtype)

    assert len(weights) == len(model_params)

    for k, v in weights.items():
        if re.match('blocks.*.attn.qkv.weight', k):
            weights[k] = split_qkv_tp(v, args.num_heads, args.hidden_size,
                                      tensor_parallel, mapping.tp_rank)
        elif re.match('blocks.*.attn.qkv.bias', k):
            weights[k] = split_qkv_bias_tp(v, args.num_heads, args.hidden_size,
                                           tensor_parallel, mapping.tp_rank)
        elif re.match('blocks.*.attn.dense.weight', k):
            weights[k] = split_matrix_tp(v,
                                         tensor_parallel,
                                         mapping.tp_rank,
                                         dim=1)
        elif re.match('blocks.*.mlp.fc.weight', k):
            weights[k] = split_matrix_tp(v,
                                         tensor_parallel,
                                         mapping.tp_rank,
                                         dim=0)
        elif re.match('blocks.*.mlp.fc.bias', k):
            weights[k] = split(v, tensor_parallel, mapping.tp_rank)
        elif re.match('blocks.*.mlp.proj.weight', k):
            weights[k] = split_matrix_tp(v,
                                         tensor_parallel,
                                         mapping.tp_rank,
                                         dim=1)
        elif re.match(r'.*adaLN_modulation.weight', k):
            weights[k] = split_matrix_tp(v,
                                         tensor_parallel,
                                         mapping.tp_rank,
                                         dim=0)
        elif re.match(r'.*adaLN_modulation.bias', k):
            weights[k] = split(v, tensor_parallel, mapping.tp_rank)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def save_config(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config = {
        'architecture': "DiT",
        'dtype': args.dtype,
        'input_size': args.input_size,
        'patch_size': args.patch_size,
        'in_channels': args.in_channels,
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.depth,
        'num_attention_heads': args.num_heads,
        'mlp_ratio': args.mlp_ratio,
        'class_dropout_prob': args.class_dropout_prob,
        'num_classes': args.num_classes,
        'learn_sigma': args.learn_sigma,
        'cfg_scale': args.cfg_scale,
        'mapping': {
            'world_size': args.tp_size * args.pp_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        }
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


def covert_and_save(args, rank):
    if rank == 0:
        save_config(args)

    mapping = Mapping(world_size=args.tp_size * args.pp_size,
                      rank=rank,
                      tp_size=args.tp_size,
                      pp_size=args.pp_size)

    weights = convert_timm_dit(args, mapping, dtype=args.dtype)

    safetensors.torch.save_file(
        weights, os.path.join(args.output_dir, f'rank{rank}.safetensors'))


def execute(workers, func, args):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, args, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(
                exceptions
            ) == 0, "Checkpoint conversion failed, please check error log."


def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size

    assert args.pp_size == 1, "PP is not supported yet."

    tik = time.time()

    if args.timm_ckpt is None:
        return

    execute(args.workers, [covert_and_save] * world_size, args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
