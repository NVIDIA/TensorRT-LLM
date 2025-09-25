import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoConfig

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.layers import MoeConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import MLLaMAForCausalLM
from tensorrt_llm.models.convert_utils import infer_dtype
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


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
    parser.add_argument(
        '--moe_tp_size',
        type=int,
        default=-1,
        help=
        'N-way tensor parallelism size for MOE, default is tp_size, which will do tp-only for MoE'
    )
    parser.add_argument(
        '--moe_ep_size',
        type=int,
        default=-1,
        help=
        'N-way expert parallelism size for MOE, default is 1, which will do tp-only for MoE'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help=
        "The data type for the model weights and activations if not quantized. "
        "If 'auto', the data type is automatically inferred from the source model; "
        "however, if the source dtype is float32, it is converted to float16.")
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--n_kv_head', type=int, default=None)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--inter_size', type=int, default=11008)
    parser.add_argument('--multiple_of', type=int, default=None)
    parser.add_argument('--ffn_dim_multiplier', type=float, default=None)
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
        '--calib_dataset',
        type=str,
        default='ccdv/cnn_dailymail',
        help=
        "The huggingface dataset name or the local directory of the dataset for calibration."
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
        '--fp8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. fp8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--quant_ckpt_path',
        type=str,
        default=None,
        help='Path of a quantized model checkpoint in .safetensors format')
    parser.add_argument("--use_fp8",
                        action="store_true",
                        default=False,
                        help="Enable FP8 per-tensor quantization")
    parser.add_argument("--use_fp8_rowwise",
                        action="store_true",
                        default=False,
                        help="Enable Fp8 per-token per-channel quantization")

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
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT LLM checkpoint')
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
        '--moe_renorm_mode',
        default=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
        type=int,
        help=
        'Controls renormalization after gate logits. Check layers/moe.py for accepted values',
    )
    parser.add_argument(
        '--save_config_only',
        action="store_true",
        default=False,
        help=
        'Only save the model config w/o read and converting weights, be careful, this is for debug only'
    )
    parser.add_argument('--log_level', type=str, default='info')

    args = parser.parse_args()
    # changing the default to be consistent as the cli help said.
    if args.moe_num_experts and args.moe_top_k == 0:
        args.moe_top_k = 1
    return args


def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
    '''return config dict with quantization info based on the command line args
    '''
    quant_config = QuantConfig()
    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            quant_config.quant_algo = QuantAlgo.W8A16
        elif args.weight_only_precision == 'int4':
            quant_config.quant_algo = QuantAlgo.W4A16
    elif args.use_fp8:
        quant_config.quant_algo = QuantAlgo.FP8
    elif args.smoothquant:
        quant_config.smoothquant_val = args.smoothquant
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
    elif args.use_fp8_rowwise:
        quant_config.quant_algo = QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
        # this will be overwritten if specified in the hf config.
        quant_config.clamp_val = [-1200.0, 1200.0]

    if args.int8_kv_cache:
        quant_config.kv_cache_quant_algo = QuantAlgo.INT8

    if args.fp8_kv_cache:
        quant_config.kv_cache_quant_algo = QuantAlgo.FP8

    if args.weight_only_precision == 'int4_gptq':
        quant_config.group_size = args.group_size
        quant_config.has_zero_point = True
        quant_config.pre_quant_scale = False
        quant_config.quant_algo = QuantAlgo.W4A16_GPTQ

    return quant_config


def update_quant_config_from_hf(quant_config, hf_config) -> QuantConfig:
    hf_config_dict = hf_config.to_dict()
    if hf_config_dict.get('quantization_config'):
        # update the quant_algo, and clamp_val.
        if hf_config_dict['quantization_config'].get(
                'quant_method') == 'fbgemm_fp8':
            logger.info(
                "Load quantization configs from huggingface model_config.")
            quant_config.quant_algo = QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
            activation_scale_ub = hf_config_dict['quantization_config'].get(
                'activation_scale_ub', 1200.0)
            quant_config.clamp_val = [-activation_scale_ub, activation_scale_ub]
    return quant_config


# def convert_and_save_meta(args, rank):
#     mapping = Mapping(world_size=args.tp_size * args.pp_size,
#                       tp_size=args.tp_size,
#                       pp_size=args.pp_size,
#                       moe_tp_size=args.moe_tp_size,
#                       moe_ep_size=args.moe_ep_size,
#                       rank=rank)
#     llama = MLLaMAForCausalLM.from_meta_ckpt(
#         args.meta_ckpt_dir,
#         args.dtype,
#         quant_config=args_to_quant_config(args),
#         mapping=mapping,
#         use_parallel_embedding=args.use_parallel_embedding,
#         embedding_sharding_dim=args.embedding_sharding_dim)
#     llama.save_checkpoint(args.output_dir, save_config=(rank == 0))


def args_to_build_options(args):
    return {
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'disable_weight_only_quant_plugin':
        args.disable_weight_only_quant_plugin,
        'quant_ckpt_path': args.quant_ckpt_path,
        'load_model_on_cpu': args.load_model_on_cpu,
    }


def from_cli_args(args):
    n_kv_head = args.n_kv_head if args.n_kv_head is not None else args.n_head
    config = {
        'architecture': "MLLaMAForCausalLM",
        'dtype': infer_dtype(args.dtype),
        'logits_dtype': 'float32',
        'num_hidden_layers': args.n_layer,
        'num_attention_heads': args.n_head,
        'hidden_size': args.n_embd,
        'intermediate_size': args.inter_size,
        'ffn_dim_multiplier': args.ffn_dim_multiplier,
        'multiple_of': args.multiple_of,
        'num_key_value_heads': n_kv_head,
        'vocab_size': args.vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': args.n_positions,
        'hidden_act': args.hidden_act,
        'rotary_base': args.rotary_base,
        'norm_epsilon': args.rms_norm_eps,
        'moe': {
            'num_experts': args.moe_num_experts,
            'top_k': args.moe_top_k,
            'normalization_mode': args.moe_renorm_mode,
        },
        'mapping': {
            'world_size': args.tp_size * args.pp_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
            'moe_tp_size': args.moe_tp_size,
            'moe_ep_size': args.moe_ep_size,
        },
        'quantization': args_to_quant_config(args).to_dict()
    }
    config.update(args_to_build_options(args))
    return config


def convert_and_save_hf(args):
    model_dir = args.model_dir
    load_by_shard = args.load_by_shard
    world_size = args.tp_size * args.pp_size
    # Need to convert the cli args to the key-value pairs and override them in the generate config dict.
    # Ideally these fields will be moved out of the config and pass them into build API, keep them here for compatibility purpose for now,
    # before the refactor is done.
    override_fields = {}
    override_fields.update(args_to_build_options(args))

    quant_config = args_to_quant_config(args)

    try:
        hf_config = AutoConfig.from_pretrained(model_dir,
                                               trust_remote_code=True)
        quant_config = update_quant_config_from_hf(quant_config, hf_config)
    except:
        # llava_llama needs its own defined config.
        logger.warning("AutoConfig cannot load the huggingface config.")

    if args.smoothquant is not None or args.int8_kv_cache:
        assert not args.load_by_shard, "When using quantization, TRT-LLM needs to load the whole HF model, thus load by shard not supported"
        mapping = Mapping(world_size=world_size,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size,
                          moe_tp_size=args.moe_tp_size,
                          moe_ep_size=args.moe_ep_size)
        # TODO: support moe quantization for tp + ep
        MLLaMAForCausalLM.quantize(
            args.model_dir,
            args.output_dir,
            dtype=args.dtype,
            mapping=mapping,
            quant_config=quant_config,
            device='cpu' if args.load_model_on_cpu else 'cuda',
            calib_dataset=args.calib_dataset,
            **override_fields)
    else:
        # When not loading by shard, preload one complete model and then slice per rank weights from this
        # this saves the disk reloading time
        def convert_and_save_rank(args, rank):
            mapping = Mapping(world_size=world_size,
                              rank=rank,
                              tp_size=args.tp_size,
                              pp_size=args.pp_size,
                              moe_tp_size=args.moe_tp_size,
                              moe_ep_size=args.moe_ep_size)
            tik = time.time()
            llama = MLLaMAForCausalLM.from_hugging_face(
                model_dir,
                args.dtype,
                mapping=mapping,
                quant_config=quant_config,
                load_by_shard=load_by_shard,
                **override_fields,
            )
            print(f'Total time of reading and converting {time.time()-tik} s')
            tik = time.time()
            llama.save_checkpoint(args.output_dir, save_config=(rank == 0))
            del llama
            print(f'Total time of saving checkpoint {time.time()-tik} s')

        execute(args.workers, [convert_and_save_rank] * world_size, args)
        release_gc()


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
    logger.set_level(args.log_level)

    args.tp_size * args.pp_size
    if (args.moe_tp_size == -1 and args.moe_ep_size == -1):
        # moe default to tp-only
        args.moe_tp_size = args.tp_size
        args.moe_ep_size = 1
    elif (args.moe_tp_size == -1):
        args.moe_tp_size = args.tp_size // args.moe_ep_size
    elif (args.moe_ep_size == -1):
        args.moe_ep_size = args.tp_size // args.moe_tp_size
    assert (args.moe_tp_size * args.moe_ep_size == args.tp_size
            ), "moe_tp_size * moe_ep_size must equal to tp_size"
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if (args.model_dir is None
            and args.meta_ckpt_dir is None):  # generate fake config.json
        config = from_cli_args(args)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    elif args.meta_ckpt_dir is not None:
        assert False, "Not support this case now"
        # assert args.model_dir is None, "Shall not specify both meta checkpoint dir and hugging face dir"
        # execute(args.workers, [convert_and_save_meta] * world_size, args)
    else:  # all other paths from hf model
        assert args.model_dir is not None
        assert (
            args.quant_ckpt_path is not None
            and args.weight_only_precision == 'int4_gptq'
        ) or args.quant_ckpt_path is None, "only gptq weights only needs this option"
        convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
