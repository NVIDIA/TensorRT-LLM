import argparse
import copy
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import safetensors
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import PretrainedConfig
from tensorrt_llm.models.cogvlm.convert import convert_hf_cogvlm
from tensorrt_llm.models.convert_utils import load_calib_dataset
from tensorrt_llm.models.llama.convert import (capture_activation_range,
                                               load_weights_from_gptq,
                                               load_weights_from_hf_by_shard,
                                               load_weights_from_meta_ckpt,
                                               smooth_llama_model)

try:
    from transformers import LlavaConfig, LlavaForConditionalGeneration
except ImportError:
    pass


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
        '--quant_ckpt_path',
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

    parser.add_argument(
        '--enable_fp8',
        default=False,
        action='store_true',
        help='Use FP8 Linear layer for Attention QKV/Dense and MLP.')
    parser.add_argument(
        '--fp8_kv_cache',
        default=False,
        action="store_true",
        help='By default, we use dtype for KV cache. fp8_kv_cache chooses int8 '
        'quantization for KV')
    parser.add_argument('--load_by_shard',
                        action='store_true',
                        help='Load a pretrained model shard-by-shard.')
    parser.add_argument('--hidden_act', type=str, default='silu')

    parser.add_argument('--rotary_base', type=float, default=10000.0)
    parser.add_argument('--rotary_scaling', nargs=2, type=str, default=None)

    parser.add_argument('--group_size',
                        type=int,
                        default=128,
                        help='Group size used in GPTQ/AWQ quantization.')

    parser.add_argument("--storage-type",
                        "-t",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16"])
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
    parser.add_argument('--use_prompt_tuning',
                        action="store_true",
                        default=False)
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument('--enable_pos_shift',
                        default=False,
                        action='store_true',
                        help='Enable position shift for streamingllm method')
    parser.add_argument(
        '--dense_context_fmha',
        default=False,
        action='store_true',
        help=
        'Enable dense fmha in context phase, otherwise sliding window attention.'
        'If dense_context_fmha=False, the sliding window size is the max attention window size.'
    )
    args = parser.parse_args()
    return args


def update_quantization_from_args(config: dict, args: argparse.Namespace):
    '''update the given config dict in-place based on the command line args
    '''
    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            config['quantization']['quant_algo'] = 'W8A16'
        elif args.weight_only_precision == 'int4':
            config['quantization']['quant_algo'] = 'W4A16'
    elif args.smoothquant:
        config['quantization']['sq_use_plugin'] = True
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


def create_config_from_args(args: argparse.Namespace):
    config = {
        'architecture': args.architecture,
        'dtype': args.dtype,
        'logits_dtype': 'float32',
        'num_hidden_layers': args.n_layer,
        'num_attention_heads': args.n_head,
        'hidden_size': args.n_embd,
        'intermediate_size': args.inter_size,
        'num_key_value_heads': args.n_kv_head,
        'vocab_size': args.vocab_size,
        'position_embedding_type': 'learned_absolute',
        'max_position_embeddings': args.n_positions,
        'hidden_act': args.hidden_act,
        'rotary_base': args.rotary_base,
        'rotary_scaling': args.rotary_scaling,
        'norm_epsilon': args.rms_norm_eps,
        'quantization': {
            'quant_algo': None,
            'kv_cache_quant_algo': None,
        },
        'mapping': {
            'world_size': args.tp_size * args.pp_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'use_prompt_tuning': args.use_prompt_tuning,
        'enable_pos_shift': args.enable_pos_shift,
        'dense_context_fmha': args.dense_context_fmha,
    }
    update_quantization_from_args(config, args)
    return config


def smooth_quant(model, args):
    assert model is not None
    act_range = {}
    llama_qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    llama_smoother = {}

    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
        "TOKENIZERS_PARALLELISM", "false")
    if args.load_model_on_cpu:
        logger.warning(
            "Note that running capture_activation_range on cpu would be very slow."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,
                                              trust_remote_code=True,
                                              use_fast=False,
                                              padding_side='left')
    dataset = load_calib_dataset(args.calib_dataset,
                                 cache_dir=args.dataset_cache_dir)

    act_range = capture_activation_range(model, tokenizer, dataset)
    if args.smoothquant is not None:
        smooth_llama_model(model, act_range, args.smoothquant, llama_qkv_para,
                           llama_smoother)
    return act_range, llama_qkv_para, llama_smoother


def main():
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    logger.info(tensorrt_llm.__version__)
    args = parse_arguments()
    if args.model_dir is None and args.meta_ckpt_dir is None:
        raise AssertionError(
            "One of the model_dir or meta_ckpt_dir must be specified to generate the checkpoint"
        )

    world_size = args.tp_size * args.pp_size

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    hf_config = None
    if args.model_dir is not None:
        hf_config = AutoConfig.from_pretrained(args.model_dir,
                                               trust_remote_code=True)
        if hf_config.model_type == "llava":
            # LLaVA = Vision model + Llama LLM
            # We load a llava config and use its' text config as llama config
            hf_config = LlavaConfig.from_pretrained(args.model_dir).text_config
            hf_config.model_type = "llava"  # Replace llama with llava

        if hf_config.architectures[0] == "CogVLMForCausalLM":
            hf_config.model_type = 'cogvlm'
        args.model_type = hf_config.model_type
        args.n_head = hf_config.num_attention_heads
        args.inter_size = hf_config.intermediate_size
        args.n_layer = hf_config.num_hidden_layers
        args.n_embd = hf_config.hidden_size
        if hasattr(hf_config, "num_key_value_heads"):
            args.n_kv_head = hf_config.num_key_value_heads
        if args.n_kv_head is None:
            args.n_kv_head = args.n_head
        args.rms_norm_eps = hf_config.rms_norm_eps
        args.vocab_size = hf_config.vocab_size
        args.n_positions = hf_config.max_position_embeddings

        args.architecture = hf_config.architectures[0]

    elif args.meta_ckpt_dir is not None:
        with open(Path(args.meta_ckpt_dir, "params.json")) as fp:
            meta_config: dict = json.load(fp)
        args.n_embd = meta_config["dim"]
        args.n_head = meta_config["n_heads"]
        args.n_layer = meta_config["n_layers"]
        args.n_kv_head = meta_config.get("n_kv_heads", args.n_head)

        if "hidden_dim" in meta_config:
            args.inter_size = meta_config["hidden_dim"]
        else:
            args.multiple_of = meta_config.get("multiple_of", 1)
            n_embd = int(4 * args.n_embd * 2 / 3)
            args.ffn_dim_multiplier = meta_config.get("ffn_dim_multiplier", 1)
            args.inter_size = args.multiple_of * (
                (int(n_embd * args.ffn_dim_multiplier) + args.multiple_of - 1)
                // args.multiple_of)
        args.rms_norm_eps = meta_config["norm_eps"]
        args.architecture = "LlamaForCausalLM"
    else:
        args.n_kv_head = args.n_kv_head or args.n_head
        args.architecture = "LlamaForCausalLM"

    if args.rotary_scaling is not None:
        # assert args.use_gpt_attention_plugin, "RoPE scaling is only supported through GPT attention plugin."
        rotary_scaling = {
            "type": args.rotary_scaling[0],
            "factor": float(args.rotary_scaling[1])
        }
        assert rotary_scaling["type"] in ["linear", "dynamic"]
        assert rotary_scaling["factor"] > 1.0
        args.rotary_scaling = rotary_scaling

    config = create_config_from_args(args)

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    act_range = {}
    llama_qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    llama_smoother = {}
    model = None
    if args.model_dir is not None:

        if args.model_type == "llava":
            hf_llava = LlavaForConditionalGeneration.from_pretrained(
                args.model_dir, dtype="auto")
            model = hf_llava.language_model
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_dir,
                device_map='auto' if not args.load_model_on_cpu else 'cpu',
                dtype='auto' if not args.smoothquant else torch.float16,
                trust_remote_code=True,
            )
        if args.smoothquant is not None or args.int8_kv_cache:
            act_range, llama_qkv_para, llama_smoother = smooth_quant(
                model, args)

    def covert_and_save(rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            weights = load_weights_from_gptq(
                args.quant_ckpt_path,
                PretrainedConfig.from_dict(copy.deepcopy(config)),
            )

        elif args.meta_ckpt_dir is not None:
            weights = load_weights_from_meta_ckpt(
                args.meta_ckpt_dir,
                PretrainedConfig.from_dict(copy.deepcopy(config)),
            )

        else:
            if args.load_by_shard:
                weights = load_weights_from_hf_by_shard(
                    args.model_dir,
                    PretrainedConfig.from_dict(copy.deepcopy(config)),
                )

            else:
                if args.weight_only_precision == 'int8':
                    plugin_weight_only_quant_type = torch.int8
                elif args.weight_only_precision == 'int4':
                    plugin_weight_only_quant_type = torch.quint4x2
                weights = convert_hf_cogvlm(
                    model,
                    mapping,
                    vocab_size=args.vocab_size,
                    dtype=args.dtype,
                    use_weight_only=args.use_weight_only,
                    use_gemm_woq_plugin=not args.
                    disable_weight_only_quant_plugin,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type,
                    use_parallel_embedding=args.use_parallel_embedding,
                    sharding_dim=args.embedding_sharding_dim,
                    use_smooth_quant=args.smoothquant,
                    per_channel=args.per_channel,
                    per_token=args.per_token,
                    int8_kv_cache=args.int8_kv_cache,
                    act_range=act_range,
                    qkv_para=llama_qkv_para,
                    smoother=llama_smoother)

        safetensors.torch.save_file(
            weights, os.path.join(args.output_dir, f'rank{rank}.safetensors'))

    if args.workers == 1:

        for rank in range(world_size):
            covert_and_save(rank)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as p:
            futures = [
                p.submit(covert_and_save, rank) for rank in range(world_size)
            ]
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

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
