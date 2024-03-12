import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import safetensors

import tensorrt_llm
from tensorrt_llm.layers import MoeConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.models.llama.convert import (create_config_from_hugging_face,
                                               from_hugging_face, quantize)
from tensorrt_llm.models.llama.weight import load_from_gptq_llama


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--meta_ckpt_dir', type=str, default=None)

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
    parser.add_argument('--inter_size', type=int, default=11008)
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
        '--ammo_quant_ckpt_path',
        type=str,
        default=None,
        help='Path of a quantized model checkpoint in .npz format')

    parser.add_argument(
        '--per_group',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale weights in the int4 range. '
        'per_group chooses at run time, and for each group, a custom scaling factor. '
        'The flag is built for GPTQ/AWQ quantization.')

    parser.add_argument('--load_by_shard',
                        action='store_true',
                        help='Load a pretrained model shard-by-shard.')
    parser.add_argument('--hidden_act', type=str, default='silu')

    parser.add_argument('--rotary_base', type=float, default=10000.0)

    parser.add_argument('--group_size',
                        type=int,
                        default=128,
                        help='Group size used in GPTQ quantization.'
                        )  # AWQ is only supported by quantize.py script

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
    parser.add_argument('--use_prompt_tuning',
                        action="store_true",
                        default=False)
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
        '--moe_num_experts',
        default=0,
        type=int,
        help='Specify the number of experts to use for MOE layers')
    parser.add_argument(
        '--moe_top_k',
        default=0,
        type=int,
        help=
        'Specify the top_k value to use for MOE layers. Default to 1 if --moe_num_experts is set'
    )
    parser.add_argument(
        '--moe_tp_mode',
        default=MoeConfig.ParallelismMode.TENSOR_PARALLEL,
        type=int,
        help=
        'Controls how to distribute experts in TP. Check layers/moe.py for accepted values',
    )
    parser.add_argument(
        '--moe_renorm_mode',
        default=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
        type=int,
        help=
        'Controls renormalization after gate logits. Check layers/moe.py for accepted values',
    )

    parser.add_argument('--hf_lora_dir', type=str, default=None)
    parser.add_argument(
        '--lora_target_modules',
        nargs='+',
        default=None,
        choices=[
            "attn_qkv",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_dense",
            "mlp_h_to_4h",
            "mlp_gate",
            "mlp_4h_to_h",
        ],
        help=
        "Add lora in which modules. Only be activated when use_lora_plugin is enabled."
    )
    parser.add_argument(
        '--max_lora_rank',
        type=int,
        default=64,
        help='maximum lora rank for different lora modules. '
        'It is used to compute the workspace size of lora plugin.')
    parser.add_argument(
        '--save_config_only',
        action="store_true",
        default=False,
        help=
        'Only save the model config w/o read and converting weights, be careful, this is for debug only'
    )

    args = parser.parse_args()
    return args


def args_to_quantization(args: argparse.Namespace):
    '''return config dict with quantization info based on the command line args
    '''
    config = {
        'quantization': {
            'quant_algo': None,
            'kv_cache_quant_algo': None,
            'exclude_modules': ['lm_head'],
        }
    }

    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            config['quantization']['quant_algo'] = 'W8A16'
        elif args.weight_only_precision == 'int4':
            config['quantization']['quant_algo'] = 'W4A16'
    elif args.smoothquant:
        if args.per_channel:
            if args.per_token:
                config['quantization'][
                    'quant_algo'] = 'W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN'
            else:
                config['quantization'][
                    'quant_algo'] = 'W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN'
        else:
            if args.per_token:
                config['quantization'][
                    'quant_algo'] = 'W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN'
            else:
                config['quantization'][
                    'quant_algo'] = 'W8A8_SQ_PER_TENSOR_PLUGIN'

    if args.int8_kv_cache:
        config['quantization']['kv_cache_quant_algo'] = 'INT8'

    if args.weight_only_precision == 'int4_gptq':
        config['quantization'].update({
            "group_size": args.group_size,
            "has_zero_point": True,
            "pre_quant_scale": False,
            'quant_algo': 'W4A16_GPTQ'
        })
    return config


def has_any_quant(args):
    config = args_to_quantization(args)
    return config['quantization']['quant_algo'] is not None or config[
        'quantization']['kv_cache_quant_algo'] is not None


def create_config_from_args(args: argparse.Namespace):
    config = {}
    mapping = Mapping(world_size=args.tp_size * args.pp_size,
                      tp_size=args.tp_size,
                      pp_size=args.pp_size)

    # Need to convert the cli args to the kay-value pairs and override them in the generate config dict.
    # Ideally these fields will be moved out of the config and pass them into build API, keep them here for compatibility purpose for now,
    # before the refactor is done.
    override_fields = {'moe_tp_mode': args.moe_tp_mode}
    override_fields.update(args_to_quantization(args))
    override_fields.update(args_to_build_options(args))

    assert args.model_dir is not None
    kwargs = {
        'hf_lora_dir': args.hf_lora_dir,
        'lora_target_modules': args.lora_target_modules,
        'max_lora_rank': args.max_lora_rank,
    }
    config = create_config_from_hugging_face(args.model_dir,
                                             args.dtype,
                                             mapping,
                                             override_fields=override_fields,
                                             **kwargs)
    return config


def convert_and_save_meta(args, rank):
    mapping = Mapping(world_size=args.tp_size * args.pp_size,
                      tp_size=args.tp_size,
                      pp_size=args.pp_size,
                      rank=rank)
    override_fields = {'moe_tp_mode': args.moe_tp_mode}
    override_fields.update(args_to_quantization(args))
    override_fields.update(args_to_build_options(args))

    assert not has_any_quant(
        args
    ), "quantization from meta checkpoint or empty model were never supported"
    assert not args.hf_lora_dir, "lora is only supported when loading from hf model dir for now"
    kwargs = {}
    assert args.meta_ckpt_dir is not None
    llama = LLaMAForCausalLM.from_meta_ckpt(args.meta_ckpt_dir,
                                            args.dtype,
                                            mapping,
                                            override_fileds=override_fields,
                                            **kwargs)
    llama.save_checkpoint(args.output_dir, save_config=(rank == 0))


def args_to_build_options(args):
    return {
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'share_embedding_table': args.use_embedding_sharing,
        'use_prompt_tuning': args.use_prompt_tuning,
        'disable_weight_only_quant_plugin':
        args.disable_weight_only_quant_plugin
    }


def from_cli_args(args):
    config = {}
    mapping = Mapping(world_size=args.tp_size * args.pp_size,
                      tp_size=args.tp_size,
                      pp_size=args.pp_size)
    architecture = "LlamaForCausalLM"
    n_layer = args.n_layer
    n_head = args.n_head
    n_embd = args.n_embd
    inter_size = args.inter_size
    n_kv_head = args.n_kv_head if args.n_kv_head is not None else n_head  # default to MHA
    vocab_size = args.vocab_size
    n_positions = args.n_positions
    hidden_act = args.hidden_act
    rotary_base = args.rotary_base
    rms_norm_eps = args.rms_norm_eps
    moe_num_experts = args.moe_num_experts
    moe_top_k = args.moe_top_k
    moe_tp_mode = args.moe_tp_mode
    config['moe_normalization_mode'] = args.moe_renorm_mode
    # config values from reading model config
    config.update({
        'architecture': architecture,
        'dtype': args.dtype,
        'logits_dtype': 'float32',
        'num_hidden_layers': n_layer,
        'num_attention_heads': n_head,
        'hidden_size': n_embd,
        'intermediate_size': inter_size,
        'num_key_value_heads': n_kv_head,
        'vocab_size': vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': n_positions,
        'hidden_act': hidden_act,
        'rotary_base': rotary_base,
        'norm_epsilon': rms_norm_eps,
        'moe_num_experts': moe_num_experts,
        'moe_top_k': moe_top_k,
        'moe_tp_mode': moe_tp_mode,
        'mapping': {
            'world_size': mapping.tp_size * mapping.pp_size,
            'tp_size': mapping.tp_size,
            'pp_size': mapping.pp_size
        }
    })
    config.update(args_to_build_options(args))
    return config


def convert_and_save_hf(args):
    model_dir = args.model_dir
    load_model_on_cpu = args.load_model_on_cpu
    load_by_shard = args.load_by_shard
    world_size = args.tp_size * args.pp_size
    # Need to convert the cli args to the kay-value pairs and override them in the generate config dict.
    # Ideally these fields will be moved out of the config and pass them into build API, keep them here for compatibility purpose for now,
    # before the refactor is done.
    override_fields = {'moe_tp_mode': args.moe_tp_mode}
    override_fields.update(args_to_quantization(args))
    override_fields.update(args_to_build_options(args))
    assert model_dir is not None

    if args.smoothquant is not None or args.int8_kv_cache:
        assert not args.load_by_shard, "When using quantization, TRT-LLM needs to load the whole HF model, thus load by shard not supported"
        assert not args.load_model_on_cpu, "When using quantization, TRT-LLM needs to load the model to GPU"
        mapping = Mapping(
            world_size=world_size,
            rank=-1,  #intentinoally make -1 to avoid mistake
            tp_size=args.tp_size,
            pp_size=args.pp_size)
        quantize(args.dtype,
                 args.model_dir,
                 args.output_dir,
                 mapping,
                 override_fields=override_fields,
                 dataset_cache_dir=args.dataset_cache_dir,
                 smoothquant_val=args.smoothquant,
                 int8_kv_cache=args.int8_kv_cache,
                 hf_lora_dir=args.hf_lora_dir,
                 lora_target_modules=args.lora_target_modules,
                 max_lora_rank=args.max_lora_rank)
    else:
        for rank in range(world_size):
            mapping = Mapping(world_size=world_size,
                              rank=rank,
                              tp_size=args.tp_size,
                              pp_size=args.pp_size)
            #TODO: change to LLaMAForCausalLM.from_hugging_face after refactor is done
            llama = from_hugging_face(
                LLaMAForCausalLM,
                model_dir,
                args.dtype,
                mapping=mapping,
                load_by_shard=load_by_shard,
                load_model_on_cpu=load_model_on_cpu,
                override_fields=override_fields,
                hf_lora_dir=args.hf_lora_dir,
                lora_target_modules=args.lora_target_modules,
                max_lora_rank=args.max_lora_rank)
            llama.save_checkpoint(args.output_dir, save_config=(rank == 0))


def convert_and_save_gptq(args, rank):
    config = create_config_from_args(args)
    if rank == 0:
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    mapping = Mapping(world_size=config['mapping']['tp_size'] *
                      config['mapping']['pp_size'],
                      rank=rank,
                      tp_size=config['mapping']['tp_size'],
                      pp_size=config['mapping']['pp_size'])
    weights = load_from_gptq_llama(args.ammo_quant_ckpt_path,
                                   config['num_hidden_layers'],
                                   config['vocab_size'],
                                   mapping,
                                   dtype=config['dtype'])
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

    # changing the default to be consistent as the cli help said.
    if args.moe_num_experts and args.moe_top_k == 0:
        args.moe_top_k = 1
    world_size = args.tp_size * args.pp_size
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ####### save config
    if (args.model_dir is None and args.meta_ckpt_dir is None):
        config = from_cli_args(args)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        return
    elif args.meta_ckpt_dir is not None:
        execute(args.workers, [convert_and_save_meta] * world_size, args)
    elif args.weight_only_precision == 'int4_gptq':
        assert args.model_dir is not None
        assert args.ammo_quant_ckpt_path is not None
        execute(args.workers, [convert_and_save_gptq] * world_size, args)
    else:  # all other non-gptq paths from hf model
        assert args.model_dir is not None
        assert args.ammo_quant_ckpt_path is None, "only gptq weights only needs this option"
        convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
