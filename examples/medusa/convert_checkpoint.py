import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import safetensors
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, LlamaTokenizer,
                          Qwen2Config)

import tensorrt_llm
from tensorrt_llm._utils import numpy_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import (LLaMAForCausalLM, PretrainedConfig,
                                 QWenForCausalLM)
from tensorrt_llm.models.convert_utils import load_calib_dataset
from tensorrt_llm.models.llama.convert import load_weights_from_hf_by_shard
from tensorrt_llm.models.medusa.weight import (capture_activation_range,
                                               convert_hf_llama, load_medusa_hf)
from tensorrt_llm.quantization import QuantAlgo

try:
    from transformers import MixtralForCausalLM
except ImportError:
    MixtralForCausalLM = None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        help="base model or Medusa-enhanced quantized model from ModelOpt")
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

    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
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
    parser.add_argument("--load-model-on-cpu", action="store_true")
    parser.add_argument("--convert-model-on-cpu", action="store_true")
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

    parser.add_argument('--num_medusa_heads', type=int, default=4)
    parser.add_argument('--num_medusa_layers', type=int, default=1)
    parser.add_argument('--max_medusa_token_len', type=int, default=63)
    parser.add_argument('--medusa_hidden_act', type=str, default="silu")
    parser.add_argument('--medusa_model_dir', type=str, default=None)
    parser.add_argument('--model_type', type=str, default="llama")
    args = parser.parse_args()
    return args


def main():
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    assert args.model_type in ["llama", "mixtral",
                               "qwen2"], "Invalid model type"
    world_size = args.tp_size * args.pp_size

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.model_dir is not None:
        config_cls = Qwen2Config if args.model_type == "qwen2" else LlamaConfig
        hf_config = config_cls.from_pretrained(args.model_dir)

        args.model_type = hf_config.model_type
        args.n_head = hf_config.num_attention_heads
        args.inter_size = hf_config.intermediate_size
        args.n_layer = hf_config.num_hidden_layers
        args.n_embd = hf_config.hidden_size
        args.n_kv_head = hf_config.num_key_value_heads
        args.rms_norm_eps = hf_config.rms_norm_eps
        args.vocab_size = hf_config.vocab_size
        args.n_positions = hf_config.max_position_embeddings
        args.rotary_base = hf_config.rope_theta
        args.rotary_scaling = hf_config.rope_scaling

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

    if args.rotary_scaling is not None:
        # assert args.use_gpt_attention_plugin, "RoPE scaling is only supported through GPT attention plugin."
        rotary_scaling = {
            "type": args.rotary_scaling["rope_type"],
        }
        args.rotary_scaling = rotary_scaling

    # ModelOpt quantized ckpt
    quant_config_file = Path(args.model_dir) / "hf_quant_config.json"

    if quant_config_file.exists():
        with open(quant_config_file, 'r') as f:
            quant_config = json.load(f)

        is_modelopt_ckpt = quant_config.get("producer",
                                            {}).get("name") == "modelopt"
        if is_modelopt_ckpt:  # quantized ckpt
            args.medusa_model_dir = args.model_dir
            args.num_medusa_heads = hf_config.medusa["num_medusa_heads"]
            args.num_medusa_layers = hf_config.medusa["num_medusa_layers"]
            if args.smoothquant or args.calib_dataset:
                logger.warning(
                    "Checkpoint is already quantized. All quantization-related flags will be ignored."
                )
    else:
        is_modelopt_ckpt = False

    config = {
        'architecture': 'MedusaForCausalLM',
        'dtype': args.dtype,
        'logits_dtype': 'float32',
        'num_hidden_layers': args.n_layer,
        'num_attention_heads': args.n_head,
        'hidden_size': args.n_embd,
        'intermediate_size': args.inter_size,
        'num_key_value_heads': args.n_kv_head,
        'vocab_size': args.vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
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
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'max_draft_len': args.max_medusa_token_len,
        'num_medusa_heads': args.num_medusa_heads,
        'num_medusa_layers': args.num_medusa_layers,
        'model_type': args.model_type,
    }
    if args.model_type == "qwen2":
        config['qwen_type'] = args.model_type

    if is_modelopt_ckpt:
        with open(quant_config_file, "r") as f:
            hf_quant_config = json.load(f)
            for key, value in hf_quant_config.get("quantization", {}).items():
                config['quantization'][key] = value
    elif args.use_weight_only:
        if args.weight_only_precision == 'int8':
            config['quantization']['quant_algo'] = QuantAlgo.W8A16
        elif args.weight_only_precision == 'int4':
            config['quantization']['quant_algo'] = QuantAlgo.W4A16
    elif args.smoothquant:
        if args.per_channel:
            if args.per_token:
                config['quantization'][
                    'quant_algo'] = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
            else:
                config['quantization'][
                    'quant_algo'] = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
        else:
            if args.per_token:
                config['quantization'][
                    'quant_algo'] = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
            else:
                config['quantization'][
                    'quant_algo'] = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN

    if args.int8_kv_cache:
        config['quantization']['kv_cache_quant_algo'] = QuantAlgo.INT8

    if args.weight_only_precision == 'int4_gptq':
        config['quantization'].update({
            "group_size": args.group_size,
            "has_zero_point": True,
            "pre_quant_scale": False,
            'quant_algo': QuantAlgo.W4A16_GPTQ
        })

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    if args.weight_only_precision == 'int8':
        plugin_weight_only_quant_type = torch.int8
    elif args.weight_only_precision == 'int4':
        plugin_weight_only_quant_type = torch.quint4x2

    act_range = {}
    llama_qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    llama_smoother = {}
    model = None
    if args.model_dir is not None:
        if args.model_type == "qwen2":
            model = QWenForCausalLM.from_hugging_face(args.model_dir,
                                                      args.dtype)
        elif not is_modelopt_ckpt:
            hf_model = LlamaForCausalLM if args.model_type != "mixtral" else MixtralForCausalLM
            model = hf_model.from_pretrained(
                args.model_dir,
                dtype='auto',
                device_map='auto' if not args.load_model_on_cpu else 'cpu',
                trust_remote_code=True)

        if args.smoothquant is not None or args.int8_kv_cache:
            os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
                "TOKENIZERS_PARALLELISM", "false")
            if args.load_model_on_cpu:
                logger.warning(
                    "Note that running capture_activation_range on cpu would be very slow."
                )
            tokenizer = LlamaTokenizer.from_pretrained(args.model_dir,
                                                       padding_side='left')
            dataset = load_calib_dataset(args.calib_dataset,
                                         cache_dir=args.dataset_cache_dir)

            act_range = capture_activation_range(model, tokenizer, dataset)
            if args.smoothquant is not None:
                smooth_llama_model(model, act_range, args.smoothquant,
                                   llama_qkv_para, llama_smoother)
    convert_args = {
        'hf_model': model,
        'act_range': act_range,
        'llama_qkv_para': llama_qkv_para,
        'llama_smoother': llama_smoother,
        'config': config,
        'is_modelopt_ckpt': is_modelopt_ckpt
    }

    def covert_and_save(rank, convert_args):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        if convert_args["is_modelopt_ckpt"]:
            convert_args["hf_model"] = LLaMAForCausalLM.from_hugging_face(
                args.model_dir,
                args.dtype,
                mapping=mapping,
                quant_config=convert_args["config"]["quantization"],
                load_by_shard=args.load_by_shard)

        if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            assert False, "Never supported"
        else:
            if args.load_by_shard:
                weights = load_weights_from_hf_by_shard(
                    args.model_dir, PretrainedConfig.from_dict(config))

            else:
                if args.model_type == "qwen2" or convert_args[
                        "is_modelopt_ckpt"]:
                    weights = {
                        name: numpy_to_torch(param.raw_value)
                        for name, param in
                        convert_args['hf_model'].named_parameters()
                    }
                else:
                    weights = convert_hf_llama(
                        convert_args['hf_model'],
                        mapping,
                        rank,
                        dtype=args.dtype,
                        use_weight_only=args.use_weight_only,
                        plugin_weight_only_quant_type=
                        plugin_weight_only_quant_type,
                        use_parallel_embedding=args.use_parallel_embedding,
                        sharding_dim=args.embedding_sharding_dim,
                        use_smooth_quant=args.smoothquant,
                        per_channel=args.per_channel,
                        per_token=args.per_token,
                        int8_kv_cache=args.int8_kv_cache,
                        act_range=convert_args['act_range'],
                        qkv_para=convert_args['llama_qkv_para'],
                        smoother=convert_args['llama_smoother'])

                if args.medusa_model_dir is not None:
                    config_file = Path(args.medusa_model_dir) / "config.json"
                    with open(config_file) as fp:
                        config = json.load(fp)
                    if not convert_args["is_modelopt_ckpt"]:
                        num_medusa_heads_from_config = config.get(
                            'medusa_num_heads', args.num_medusa_heads)
                        args.num_medusa_layers = config.get(
                            'medusa_num_layers', args.num_medusa_layers)
                    if args.num_medusa_heads is None:
                        args.num_medusa_heads = num_medusa_heads_from_config

                    assert args.max_medusa_token_len > 0, "should have max_medusa_token_len > 0"

                    medusa_weights = load_medusa_hf(
                        medusa_path=args.medusa_model_dir,
                        num_medusa_heads=args.num_medusa_heads,
                        num_medusa_layers=args.num_medusa_layers,
                        mapping=mapping,
                        dtype=args.dtype,
                        use_weight_only=args.use_weight_only,
                        plugin_weight_only_quant_type=
                        plugin_weight_only_quant_type,
                        is_modelopt_ckpt=convert_args["is_modelopt_ckpt"])
                    weights.update(medusa_weights)

        safetensors.torch.save_file(
            weights, os.path.join(args.output_dir, f'rank{rank}.safetensors'))

    if args.workers == 1:
        for rank in range(world_size):
            covert_and_save(rank, convert_args)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as p:
            futures = [
                p.submit(covert_and_save, rank, convert_args)
                for rank in range(world_size)
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
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
