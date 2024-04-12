import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import QWenForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.models.qwen.convert import from_hugging_face, quantize
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
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
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--n_kv_head', type=int, default=None)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--inter_size', type=int, default=22016)
    parser.add_argument('--rms_norm_eps', type=float, default=1e-06)

    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--disable_weight_only_quant_plugin',
        default=False,
        action="store_true",
        help=
        'By default, using plugin implementation for weight quantization. Enabling disable_weight_only_quant_plugin flag will use ootb implementation instead of plugin.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4', 'int4_gptq'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        "--smoothquant",
        "-sq",
        type=float,
        default=None,
        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
        " to Smoothquant the model, and output int8 weights."
        " A good first try is 0.5. Must be in [0, 1]")
    parser.add_argument(
        '--per_channel',
        action="store_true",
        default=False,
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--per_token',
        action="store_true",
        default=False,
        help=
        'By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--per_group',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale weights in the int4 range. '
        'per_group chooses at run time, and for each group, a custom scaling factor. '
        'The flag is built for GPTQ/AWQ quantization.')

    parser.add_argument('--hidden_act', type=str, default='silu')

    parser.add_argument('--rotary_base', type=float, default=10000.0)
    parser.add_argument('--rotary_scaling', nargs=2, type=str, default=None)

    parser.add_argument('--group_size',
                        type=int,
                        default=128,
                        help='Group size used in GPTQ quantization.')

    parser.add_argument("--dataset-cache-dir",
                        type=str,
                        default=None,
                        help="cache dir to load the hugging face dataset")
    parser.add_argument("--load_model_on_cpu", action="store_true")

    parser.add_argument(
        '--use_parallel_embedding',
        action="store_true",
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
    parser.add_argument(
        '--use_embedding_sharing',
        action="store_true",
        default=False,
        help=
        'Try to reduce the engine size by sharing the embedding lookup table between two layers.'
        'Note: the flag might not take effect when the criteria are not met.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument(
        '--save_config_only',
        action="store_true",
        default=False,
        help=
        'Only save the model config w/o read and converting weights, be careful, this is for debug only'
    )

    args = parser.parse_args()
    return args


def args_to_quantization(args: argparse.Namespace) -> QuantConfig:
    '''return config dict with quantization info based on the command line args
    '''
    quant_config = QuantConfig()
    quant_config.exclude_modules = ['lm_head']
    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            quant_config.quant_algo = QuantAlgo.W8A16
        elif args.weight_only_precision == 'int4':
            quant_config.quant_algo = QuantAlgo.W4A16
    elif args.smoothquant:
        if args.per_channel:
            if args.per_token:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
            else:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
        else:
            if args.per_token:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
            else:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN

    if args.int8_kv_cache:
        quant_config.kv_cache_quant_algo = QuantAlgo.INT8

    if args.weight_only_precision == 'int4_gptq':
        quant_config.group_size = args.group_size
        quant_config.has_zero_point = True
        quant_config.pre_quant_scale = False
        quant_config.quant_algo = QuantAlgo.W4A16_GPTQ

    return quant_config


def has_any_quant(args):
    quant_config = args_to_quantization(args)
    return quant_config.quant_algo is not None or quant_config.kv_cache_quant_algo is not None


def args_to_build_options(args):
    return {
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'share_embedding_table': args.use_embedding_sharing,
        'disable_weight_only_quant_plugin':
        args.disable_weight_only_quant_plugin
    }


def from_cli_args(args):
    n_kv_head = args.n_kv_head if args.n_kv_head is not None else args.n_head  # default to MHA
    config = {
        'architecture': "QWenForCausalLM",
        'dtype': args.dtype,
        'logits_dtype': 'float32',
        'num_hidden_layers': args.n_layer,
        'num_attention_heads': args.n_head,
        'hidden_size': args.n_embd,
        'intermediate_size': args.inter_size,
        'num_key_value_heads': n_kv_head,
        'vocab_size': args.vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': args.n_positions,
        'hidden_act': args.hidden_act,
        'rotary_base': args.rotary_base,
        'norm_epsilon': args.rms_norm_eps,
        'mapping': {
            'world_size': args.tp_size * args.pp_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size
        },
        'quantization': args_to_quantization(args).asdict()
    }
    config.update(args_to_build_options(args))
    return config


def preload_model(args, model_dir, load_model_on_cpu):
    from transformers import AutoModelForCausalLM
    if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto" if not load_model_on_cpu else 'cpu',
            torch_dtype='auto',
            trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map='auto' if not load_model_on_cpu else 'cpu',
            torch_dtype='auto',
            trust_remote_code=True).half()

    return model


def convert_and_save_hf(args):
    model_dir = args.model_dir
    load_model_on_cpu = args.load_model_on_cpu
    world_size = args.tp_size * args.pp_size
    # Need to convert the cli args to the kay-value pairs and override them in the generate config dict.
    # Ideally these fields will be moved out of the config and pass them into build API, keep them here for compatibility purpose for now,
    # before the refactor is done.
    override_fields = {}
    quantization = args_to_quantization(args)
    override_fields.update(args_to_build_options(args))

    if args.smoothquant is not None or args.int8_kv_cache:
        assert not args.load_model_on_cpu, "When using quantization, TRT-LLM needs to load the model to GPU"
        mapping = Mapping(
            world_size=world_size,
            rank=-1,  #intentinoally make -1 to avoid mistake
            tp_size=args.tp_size,
            pp_size=args.pp_size)
        #TODO: change to QWenForCausalLM.quantize later
        quantize(args.dtype,
                 args.model_dir,
                 args.output_dir,
                 mapping=mapping,
                 quantization=quantization,
                 override_fields=override_fields,
                 dataset_cache_dir=args.dataset_cache_dir,
                 smoothquant_val=args.smoothquant,
                 int8_kv_cache=args.int8_kv_cache)
    else:
        hf_model = preload_model(args, model_dir, load_model_on_cpu)

        def convert_and_save_rank(args, rank):
            mapping = Mapping(world_size=world_size,
                              rank=rank,
                              tp_size=args.tp_size,
                              pp_size=args.pp_size)
            #TODO: change to QWenForCausalLM.from_hugging_face later
            qwen = from_hugging_face(
                QWenForCausalLM,
                hf_model,
                model_dir,
                args.dtype,
                mapping=mapping,
                quantization=quantization,
                from_hf_gptq=(args.use_weight_only
                              and args.weight_only_precision == 'int4_gptq'),
                override_fields=override_fields)
            qwen.save_checkpoint(args.output_dir, save_config=(rank == 0))
            del qwen
            release_gc()

        execute(args.workers, [convert_and_save_rank] * world_size, args)


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

    args.tp_size * args.pp_size
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.model_dir is None:
        config = from_cli_args(args)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    else:
        assert args.model_dir is not None
        convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
