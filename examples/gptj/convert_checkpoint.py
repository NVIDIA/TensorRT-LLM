import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import safetensors
import torch
from transformers import AutoModelForCausalLM, GPTJConfig, GPTJForCausalLM

import tensorrt_llm
from tensorrt_llm.mapping import Mapping
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
    parser.add_argument('--vocab_size', type=int, default=50400)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=28)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--norm_eps', type=float, default=1e-05)
    parser.add_argument('--rotary_dim', type=int, default=64)
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
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    args = parser.parse_args()

    return args


def load_gptj_config(model_dir: str) -> GPTJConfig:
    """ Helper utility to load GPTJConfig.

    A pretrained checkpoint from modeling_RW.py has a different structure
    and is not compatible with `transformers.GPTJConfig` and
    `transformers.GPTJModel`. We need to manually set the config values.
    """

    config = GPTJConfig.from_pretrained(model_dir)
    return config


def split(weight: torch.Tensor,
          tp_size: int,
          rank: int = 0,
          dim: int = 0) -> torch.Tensor:
    if tp_size == 1:
        return weight
    elif weight.ndim == 1:
        return torch.chunk(weight, tp_size)[rank].contiguous()
    else:
        return torch.chunk(weight, tp_size, dim=dim)[rank].contiguous()


def split_matrix(weight: torch.Tensor, tp_size: int, rank: int,
                 dim: int) -> torch.Tensor:
    return split(weight, tp_size, rank, dim=dim)


def get_weight(params: Dict[str, torch.Tensor], prefix: str,
               dtype: torch.dtype) -> torch.Tensor:
    if f'{prefix}.weight' not in params:
        return None
    return params[f'{prefix}.weight'].to(dtype).detach().cpu()


def get_bias(params: Dict[str, torch.Tensor], prefix: str,
             dtype: torch.dtype) -> torch.Tensor:
    if f'{prefix}.bias' not in params:
        return None
    return params[f'{prefix}.bias'].to(dtype).detach().cpu()


def get_weight_and_bias(params: Dict[str, torch.Tensor], prefix: str,
                        dtype: torch.dtype) -> Tuple[torch.Tensor]:
    return get_weight(params, prefix, dtype), get_bias(params, prefix, dtype)


def get_tllm_linear_weight(
    weight: torch.Tensor,
    prefix: str,
    bias: Optional[torch.Tensor] = None,
    use_weight_only: bool = False,
    plugin_weight_only_quant_type: torch.dtype = torch.int8
) -> Dict[str, torch.Tensor]:
    results = {}
    if use_weight_only:
        v = weight.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v, plugin_weight_only_quant_type)
        results[f'{prefix}.weight'] = processed_torch_weights
        results[f'{prefix}.per_channel_scale'] = torch_weight_scales
    else:
        results[f'{prefix}.weight'] = weight.contiguous()

    if bias is not None:
        results[f'{prefix}.bias'] = bias

    return results


def get_tllm_param(
    param: torch.Tensor,
    name: str,
    use_weight_only: bool = False,
    plugin_weight_only_quant_type: torch.dtype = torch.int8
) -> Dict[str, torch.Tensor]:
    results = {}
    if name.endswith('.weight') and use_weight_only:
        v = param.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v, plugin_weight_only_quant_type)
        results[name] = processed_torch_weights
        results[name.replace('weight',
                             'per_channel_scale')] = torch_weight_scales
    else:
        results[name] = param

    return results


def convert_hf_gptj(hf_model: GPTJForCausalLM,
                    hf_config: GPTJConfig,
                    mapping: Mapping,
                    dtype: str = 'float32',
                    use_weight_only: bool = False,
                    plugin_weight_only_quant_type: torch.dtype = torch.int8):

    weights = {}
    tik = time.time()

    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_hidden_layers = hf_config.num_hidden_layers

    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        prefix = f'transformer.h.{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'
        # Attention QKV (no bias)
        q_weight = get_weight(model_params, f'{prefix}.attn.q_proj', dtype)
        k_weight = get_weight(model_params, f'{prefix}.attn.k_proj', dtype)
        v_weight = get_weight(model_params, f'{prefix}.attn.v_proj', dtype)
        q_w = split_matrix(q_weight, mapping.tp_size, mapping.tp_rank, dim=0)
        k_w = split_matrix(k_weight, mapping.tp_size, mapping.tp_rank, dim=0)
        v_w = split_matrix(v_weight, mapping.tp_size, mapping.tp_rank, dim=0)
        qkv_w = torch.concatenate([q_w, k_w, v_w], dim=0)
        weights.update(
            get_tllm_linear_weight(qkv_w, f'{tllm_prex}.attention.qkv', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))
        # Attention dense (not bias)
        attn_dense_weight = get_weight(model_params, f'{prefix}.attn.out_proj',
                                       dtype)
        attn_dense_w = split_matrix(attn_dense_weight,
                                    mapping.tp_size,
                                    mapping.tp_rank,
                                    dim=1)
        weights.update(
            get_tllm_linear_weight(attn_dense_w, f'{tllm_prex}.attention.dense',
                                   None, use_weight_only,
                                   plugin_weight_only_quant_type))
        # MLP fc_in (with bias)
        mlp_fc_weight, mlp_fc_bias = get_weight_and_bias(
            model_params, f'{prefix}.mlp.fc_in', dtype)
        mlp_fc_w = split_matrix(mlp_fc_weight,
                                mapping.tp_size,
                                mapping.tp_rank,
                                dim=0)
        mlp_fc_b = split_matrix(mlp_fc_bias,
                                mapping.tp_size,
                                mapping.tp_rank,
                                dim=0)
        weights.update(
            get_tllm_linear_weight(mlp_fc_w, f'{tllm_prex}.mlp.fc', mlp_fc_b,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))
        # MLP fc_out (with bias)
        mlp_proj_weight, mlp_proj_bias = get_weight_and_bias(
            model_params, f'{prefix}.mlp.fc_out', dtype)
        mlp_proj_w = split_matrix(mlp_proj_weight,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  dim=1)
        # Only rank0 will get bias
        if mapping.tp_size > 1 and mapping.tp_rank > 0:
            mlp_proj_bias = torch.zeros(mlp_proj_weight.shape[0],
                                        dtype=mlp_proj_weight.dtype)
        weights.update(
            get_tllm_linear_weight(mlp_proj_w, f'{tllm_prex}.mlp.proj',
                                   mlp_proj_bias, use_weight_only,
                                   plugin_weight_only_quant_type))

        input_ln_weight, input_ln_bias = get_weight_and_bias(
            model_params, f'{prefix}.ln_1', dtype)
        weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_weight
        weights[f'{tllm_prex}.input_layernorm.bias'] = input_ln_bias

    if mapping.is_first_pp_rank():
        # Embedding
        embed_w = get_weight(model_params, 'transformer.wte', dtype)
        weights['transformer.vocab_embedding.weight'] = embed_w
    if mapping.is_last_pp_rank():
        # lm_head weight and bias
        lm_head_w, ln_head_bias = get_weight_and_bias(model_params, 'lm_head',
                                                      dtype)
        weights['lm_head.weight'] = split_matrix(lm_head_w,
                                                 mapping.tp_size,
                                                 mapping.tp_rank,
                                                 dim=0)
        weights['lm_head.bias'] = split_matrix(ln_head_bias,
                                               mapping.tp_size,
                                               mapping.tp_rank,
                                               dim=0)
        ln_f_w, ln_f_b = get_weight_and_bias(model_params, 'transformer.ln_f',
                                             dtype)
        # ln_f weight and bias
        weights['transformer.ln_f.weight'] = ln_f_w
        if ln_f_b is not None:
            weights['transformer.ln_f.bias'] = ln_f_b

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def main():
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    quant_algo = None
    plugin_weight_only_quant_type = None
    if args.use_weight_only and args.weight_only_precision == 'int8':
        plugin_weight_only_quant_type = torch.int8
        quant_algo = QuantAlgo.W8A16
    elif args.use_weight_only and args.weight_only_precision == 'int4':
        plugin_weight_only_quant_type = torch.quint4x2
        quant_algo = QuantAlgo.W4A16

    if args.model_dir is not None:
        hf_config = load_gptj_config(args.model_dir)
        architecture = hf_config.architectures[0]
        args.vocab_size = hf_config.vocab_size
        args.n_positions = hf_config.max_position_embeddings
        args.n_layer = hf_config.num_hidden_layers
        args.n_head = hf_config.num_attention_heads
        args.n_embd = hf_config.hidden_size
        args.norm_eps = hf_config.layer_norm_epsilon
        args.rotary_dim = hf_config.rotary_dim
    else:
        architecture = "GPTJForCausalLM"

    config = {
        'architecture': architecture,
        'dtype': args.dtype,
        'num_hidden_layers': args.n_layer,
        'num_attention_heads': args.n_head,
        'hidden_size': args.n_embd,
        'norm_epsilon': args.norm_eps,
        'vocab_size': args.vocab_size,
        'position_embedding_type': 'rope_gptj',
        'max_position_embeddings': args.n_positions,
        'hidden_act': 'gelu',
        'quantization': {
            'quant_algo': quant_algo
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'rotary_dim': args.rotary_dim,
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    if args.model_dir is None:
        return

    hf_model = AutoModelForCausalLM.from_pretrained(args.model_dir,
                                                    trust_remote_code=True,
                                                    torch_dtype="auto")

    def covert_and_save(rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        weights = convert_hf_gptj(
            hf_model,
            hf_config,
            mapping,
            dtype=args.dtype,
            use_weight_only=args.use_weight_only,
            plugin_weight_only_quant_type=plugin_weight_only_quant_type)

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

    del hf_model
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
