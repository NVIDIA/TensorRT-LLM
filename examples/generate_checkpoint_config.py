import argparse
import json
import os

from tensorrt_llm.quantization import KV_CACHE_QUANT_ALGO_LIST, QUANT_ALGO_LIST


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_path',
        type=str,
        default='config.json',
        help='The path to save the TensorRT LLM checkpoint config.json file')
    parser.add_argument('--architecture', type=str, default='GPTForCausalLM')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--max_position_embeddings', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--intermediate_size', type=int, default=None)
    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--num_key_value_heads', type=int, default=None)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--norm_epsilon', type=float, default=1e-5)
    parser.add_argument('--position_embedding_type',
                        type=str,
                        default='learned_absolute')
    parser.add_argument(
        '--use_parallel_embedding',
        action='store_true',
        default=False,
        help=
        'By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled'
    )
    parser.add_argument(
        '--embedding_sharding_dim',
        type=int,
        default=0,
        choices=[0, 1],
        help=
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharing is only enabled when embedding_sharding_dim = 0'
    )

    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')

    parser.add_argument('--quant_algo',
                        type=str,
                        default=None,
                        choices=[None] + QUANT_ALGO_LIST)
    parser.add_argument('--kv_cache_quant_algo',
                        type=str,
                        default=None,
                        choices=[None] + KV_CACHE_QUANT_ALGO_LIST)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--smoothquant_val', type=float, default=None)
    parser.add_argument('--has_zero_point', default=False, action='store_true')
    parser.add_argument('--pre_quant_scale', default=False, action='store_true')
    parser.add_argument('--exclude_modules', nargs='+', default=None)

    parser.add_argument('--bias', default=False, action='store_true')
    parser.add_argument('--apply_query_key_layer_scaling',
                        default=False,
                        action='store_true')
    parser.add_argument('--rotary_pct', type=float, default=1.0)
    parser.add_argument('--rotary_base', type=float, default=10000.0)
    parser.add_argument('--rotary_scaling', nargs=2, type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size

    assert args.output_path.endswith('.json')
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = {
        'architecture': args.architecture,
        'dtype': args.dtype,
        'vocab_size': args.vocab_size,
        'max_position_embeddings': args.max_position_embeddings,
        'hidden_size': args.hidden_size,
        'intermediate_size': args.intermediate_size,
        'num_hidden_layers': args.num_hidden_layers,
        'num_attention_heads': args.num_attention_heads,
        'num_key_value_heads': args.num_key_value_heads,
        'hidden_act': args.hidden_act,
        'norm_epsilon': args.norm_epsilon,
        'position_embedding_type': args.position_embedding_type,
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'quantization': {
            'quant_algo': args.quant_algo,
            'kv_cache_quant_algo': args.kv_cache_quant_algo,
            'exclude_modules': args.exclude_modules,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'bias': args.bias,
        'apply_query_key_layer_scaling': args.apply_query_key_layer_scaling,
        'rotary_pct': args.rotary_pct,
        'rotary_base': args.rotary_base,
        'rotary_scaling': args.rotary_scaling,
    }

    if args.intermediate_size is None:
        config['intermediate_size'] = args.hidden_size * 4
    if args.num_key_value_heads is None:
        config['num_key_value_heads'] = args.num_attention_heads

    if args.quant_algo is not None:
        if 'AWQ' in args.quant_algo or 'GPTQ' in args.quant_algo:
            config['quantization'].update({
                'group_size':
                args.group_size,
                'has_zero_point':
                args.has_zero_point,
                'pre_quant_scale':
                args.pre_quant_scale,
            })
        if 'SQ' in args.quant_algo:
            config['quantization'].update({
                'smoothquant_val':
                args.smoothquant_val,
            })

    with open(args.output_path, 'w') as f:
        json.dump(config, f, indent=4)
