import argparse
import json
import os
import time
from pathlib import Path

from tqdm import tqdm
from transformers import LlamaConfig

import tensorrt_llm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.eagle.config import EagleConfig
from tensorrt_llm.models.eagle.model import EagleForCausalLM
from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
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
    parser.add_argument('--dtype',
                        type=str,
                        default='auto',
                        choices=['auto', 'float16', 'bfloat16', 'float32'])
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

    parser.add_argument('--eagle_model_dir', type=str, default=None)
    parser.add_argument('--max_draft_len', type=int, default=63)
    parser.add_argument(
        '--num_eagle_layers',
        type=int,
        default=4,
        help=
        'Maximum depth of the EAGLE choices tree, i.e. maximum number of accepted draft tokens.'
    )
    parser.add_argument(
        '--max_non_leaves_per_layer',
        type=int,
        default=10,
        help='Maximum number of non-leaf nodes in the EAGLE choice tree.')
    args = parser.parse_args()
    return args


def convert_and_save_hf(config, args):
    world_size = args.tp_size * args.pp_size
    tllm_config = EagleConfig.from_dict(config)
    for rank in range(world_size):
        tllm_config.mapping = Mapping(world_size=world_size,
                                      rank=rank,
                                      cp_size=1,
                                      tp_size=args.tp_size,
                                      pp_size=args.pp_size)

        model = EagleForCausalLM(tllm_config)

        def check_and_update(module, dict):
            if hasattr(module, 'tllm_to_externel_key_dict'):
                module.tllm_to_externel_key_dict.update(dict)
            else:
                module.tllm_to_externel_key_dict = dict

        def copy(tensors):
            if isinstance(tensors, list):
                if None in tensors:
                    return tensors
                else:
                    return [tensor.clone() for tensor in tensors]
            elif tensors is None:
                return tensors
            else:
                return tensors.clone()

        shared_weight_prefixs = []
        tllm_weights = {}
        customized_dict = {"drafter": ""}
        if args.eagle_model_dir is None:
            # Single checkpoint for ModelOpt
            for idx, eagle_net in enumerate(model.eagle_nets):
                check_and_update(eagle_net.drafter.fc, {"fc": "fc"})
                check_and_update(eagle_net.drafter.vocab_embedding,
                                 {f"eagle_nets.{idx}": "model"})
                check_and_update(eagle_net.lm_head, {f"eagle_nets.{idx}": ""})
                shared_weight_prefixs.append(f"eagle_nets.{idx}")
                customized_dict[f'eagle_nets.{idx}'] = 'eagle_module'
            loader = ModelWeightsLoader(eagle_model_dir, customized_dict)
            loader.update_key_mapping(model)
            for tllm_key, _ in tqdm(model.named_parameters()):
                if any([
                        tllm_key.startswith(prefix)
                        for prefix in shared_weight_prefixs
                ]):
                    tllm_weights.update(loader.load(tllm_key, preprocess=copy))
                else:
                    tllm_weights.update(loader.load(tllm_key))
            loader.fill(tllm_weights)
        else:
            # Double checkpoint for HF
            for idx, eagle_net in enumerate(model.eagle_nets):
                check_and_update(eagle_net.drafter.fc, {"fc": "fc"})
                check_and_update(eagle_net.drafter.vocab_embedding,
                                 {f"eagle_nets.{idx}": ""})
                check_and_update(eagle_net.lm_head, {f"eagle_nets.{idx}": ""})
                shared_weight_prefixs.append(f"eagle_nets.{idx}")
                customized_dict[f'eagle_nets.{idx}'] = ''

            # Load base model
            base_loader = ModelWeightsLoader(args.model_dir)
            base_loader.update_key_mapping(model)
            for tllm_key, _ in tqdm(model.transformer.named_parameters()):
                tllm_weights.update(base_loader.load("transformer." + tllm_key))
            tllm_weights.update(base_loader.load("lm_head.weight"))
            for idx in range(args.num_eagle_layers):
                tllm_weights.update(
                    base_loader.load(f"eagle_nets.{idx}.lm_head.weight",
                                     preprocess=copy))

            # Load eagle model
            eagle_loader = ModelWeightsLoader(eagle_model_dir, customized_dict)
            eagle_loader.update_key_mapping(model)
            for tllm_key, _ in tqdm(model.eagle_nets.named_parameters()):
                if not tllm_key.endswith("lm_head.weight"):
                    if any([
                            tllm_key.startswith(prefix)
                            for prefix in shared_weight_prefixs
                    ]):
                        tllm_weights.update(
                            eagle_loader.load("eagle_nets." + tllm_key,
                                              preprocess=copy))
                    else:
                        tllm_weights.update(
                            eagle_loader.load("eagle_nets." + tllm_key))
            base_loader.fill(tllm_weights)
        model.save_checkpoint(args.output_dir, save_config=(rank == 0))


if __name__ == '__main__':
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size

    assert args.pp_size == 1, "Pipeline parallelism is not supported in EAGLE yet."

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    hf_config = None
    eagle_model_dir = args.model_dir if args.eagle_model_dir is None else args.eagle_model_dir
    if args.model_dir is not None:
        hf_config = LlamaConfig.from_pretrained(args.model_dir)

        args.model_type = hf_config.model_type
        args.n_head = hf_config.num_attention_heads
        args.inter_size = hf_config.intermediate_size
        args.n_layer = hf_config.num_hidden_layers
        args.n_embd = hf_config.hidden_size
        args.n_kv_head = hf_config.num_key_value_heads
        args.rms_norm_eps = hf_config.rms_norm_eps
        args.vocab_size = hf_config.vocab_size
        args.rotary_scaling = hf_config.rope_scaling
        args.rotary_base = hf_config.rope_theta
        args.n_positions = hf_config.max_position_embeddings
        args.dtype = str(
            hf_config.torch_dtype)[6:] if args.dtype == 'auto' else args.dtype
        if 'head_dim' in hf_config:
            args.head_dim = hf_config.head_dim
        else:
            args.head_dim = args.n_embd // args.n_head
        if 'head_size' in hf_config:
            args.head_size = hf_config.head_size
        else:
            args.head_size = args.head_dim

        if args.eagle_model_dir is None:
            hf_config_eagle = hf_config.eagle
            args.n_head_eagle = hf_config_eagle['num_attention_heads']
            args.inter_size_eagle = hf_config_eagle['intermediate_size']
            args.n_layer_eagle = hf_config_eagle['num_hidden_layers']
            args.n_embd_eagle = hf_config_eagle['hidden_size']
            args.n_kv_head_eagle = hf_config_eagle['num_key_value_heads']
            args.rms_norm_eps_eagle = hf_config_eagle['rms_norm_eps']
            args.n_positions_eagle = hf_config_eagle['max_position_embeddings']
            if 'head_dim' in hf_config_eagle:
                args.head_dim_eagle = hf_config_eagle['head_dim']
            else:
                args.head_dim_eagle = args.n_embd_eagle // args.n_head_eagle
            if 'head_size' in hf_config_eagle:
                args.head_size_eagle = hf_config_eagle['head_size']
            else:
                args.head_size_eagle = args.head_dim_eagle
        else:
            hf_config_eagle = LlamaConfig.from_pretrained(args.eagle_model_dir)
            args.n_head_eagle = hf_config_eagle.num_attention_heads
            args.inter_size_eagle = hf_config_eagle.intermediate_size
            args.n_layer_eagle = hf_config_eagle.num_hidden_layers
            args.n_embd_eagle = hf_config_eagle.hidden_size
            args.n_kv_head_eagle = hf_config_eagle.num_key_value_heads
            args.rms_norm_eps_eagle = hf_config_eagle.rms_norm_eps
            args.n_positions_eagle = hf_config_eagle.max_position_embeddings
            if 'head_dim' in hf_config_eagle:
                args.head_dim_eagle = hf_config_eagle.head_dim
            else:
                args.head_dim_eagle = args.n_embd_eagle // args.n_head_eagle
            if 'head_size' in hf_config_eagle:
                args.head_size_eagle = hf_config_eagle.head_size
            else:
                args.head_size_eagle = args.head_dim_eagle

    elif args.meta_ckpt_dir is not None:
        assert False, "meta ckpt is not supported yet"

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

    eagle_net_config = {
        'architecture': "LlamaForCausalLM",
        'dtype': args.dtype,
        'logits_dtype': 'float32',
        'num_hidden_layers': args.n_layer_eagle,
        'num_attention_heads': args.n_head_eagle,
        'hidden_size': args.n_embd_eagle,
        'intermediate_size': args.inter_size_eagle,
        'num_key_value_heads': args.n_kv_head_eagle,
        'vocab_size': args.vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': args.n_positions_eagle,
        'hidden_act': args.hidden_act,
        'rotary_base': args.rotary_base,
        'rotary_scaling': args.rotary_scaling,
        'norm_epsilon': args.rms_norm_eps_eagle,
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
        'head_dim': args.head_dim_eagle,
        'head_size': args.head_size_eagle
    }

    config = {
        'architecture': 'EagleForCausalLM',
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
        'max_draft_len': args.max_draft_len,
        'num_eagle_layers': args.num_eagle_layers,
        'max_non_leaves_per_layer': args.max_non_leaves_per_layer,
        'eagle_net_config': eagle_net_config,
        'head_dim': args.head_dim,
        'head_size': args.head_size
    }

    assert args.max_draft_len <= 256, "args.max_draft_len > 256 is not supported"

    if args.use_weight_only:
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

    # Update quant config if hf_quant_config.json exists
    quant_config = {}
    try:
        with open(eagle_model_dir + '/' + 'hf_quant_config.json') as f:
            quant_config = json.load(f)
            if "lm_head" in quant_config['quantization']['exclude_modules']:
                quant_config['quantization']['exclude_modules'] += [
                    f"eagle_nets.{i}.lm_head"
                    for i in range(args.num_eagle_layers)
                ]
            config['quantization'].update(quant_config['quantization'])
            config['eagle_net_config']['quantization'].update(
                quant_config['quantization'])
    except IOError:
        pass

    convert_and_save_hf(config, args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')
