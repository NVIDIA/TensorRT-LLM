import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Dict, List, Optional, Tuple

import numpy as np
import safetensors
import torch
from transformers import AutoModelForCausalLM, FalconConfig, FalconForCausalLM

import tensorrt_llm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.llama.utils import (  # TODO: move the utils to common dir shared by models
    iterate_shard_files, load_state_dict, retrieved_layer_index_from_name)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float32',
                        choices=['float16', 'float32'])
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
    parser.add_argument('--group_size',
                        type=int,
                        default=128,
                        help='Group size used in GPTQ/AWQ quantization.')
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
        choices=['int8', 'int4', 'int4_awq'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        '--quantize_lm_head',
        default=False,
        action="store_true",
        help='Quantize lm_head weights as well when using int4_awq.')

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


def load_falcon_config(model_dir: str) -> FalconConfig:
    """ Helper utility to load FalconConfig.

    A pretrained checkpoint from modeling_RW.py has a different structure
    and is not compatible with `transformers.FalconConfig` and
    `transformers.FalconModel`. We need to manually set the config values.
    """

    config = FalconConfig.from_pretrained(model_dir)
    config.architectures = ["FalconForCausalLM"]

    if config.model_type not in ['RefinedWebModel', 'RefinedWeb']:
        return config

    if config.model_type == 'RefinedWeb':
        # Case 1. Falcon-40B / Falcon-40B-instruct
        # https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json
        config.num_hidden_layers = config.n_layer
        config.num_attention_heads = config.n_head
        config.num_kv_heads = config.n_head_kv
        config.new_decoder_architecture = True
    elif config.model_type == 'RefinedWebModel':
        # Case 2. Falcon-7B / Falcon-7B-instruct
        # https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
        config.num_hidden_layers = config.n_layer
        config.num_attention_heads = config.n_head
        config.num_kv_heads = 1 if config.multi_query else config.n_head
        config.new_decoder_architecture = False
    else:
        raise ValueError("Shouldn't reach here.")
    config.model_type = 'falcon'

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


def reorder_qkv_weight_or_bias(weight: torch.Tensor,
                               head_dim: int,
                               num_heads: int,
                               num_kv_heads: Optional[int] = None,
                               tp_size: int = 1,
                               is_bias: bool = False) -> torch.Tensor:
    """ Reorder the qkv weight for TRT-LLM use.

    The shape of the fused QKV weights in HF is different from the shape that
    TRT-LLM requires. In particular, the weight of HF consists of interleaved
    q, k, v head weights, while that of TRT-LLM is contiguous.
        HF     : [q1, k1, v1, ..., qh, kh, vh]
        TRT-LLM: [q1, ..., qh, k1, ..., kh, v1, vh]
    where qi, vi, ki are weight vectors corresponding to attention head i.
    It's similar to multi/grouped query attention cases.

    We reorder and split the weight of an attention layer to fit into TRT-LLM.
    The reordered weight and bias will be
        weight: (T, Qh * D + 2 * KVh * D, H)
        bias  : (T, Qh * D + 2 * KVh * D)
    where T=tp_size, Qh=local_num_q_heads, KVh=local_num_kv_heads, D=head_dim,
    H=hidden_dim. In the multi/grouped query attention, the number of K/V
    attention heads are less than that of Q attention, so that K/V attention
    heads may be shared across different ranks if necessary.

    For tensor parallelism, we use the first dimension to select the
    corresponding weights.
    """

    # Query types and expected kv heads.
    #  - Conventional MHA: num_heads = num_kv_heads
    #  - Multi-Query Attention: num_kv_heads = 1
    #  - Grouped-Query Attention: num_heads % num_kv_heads = 0
    num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    assert num_heads % num_kv_heads == 0, \
        f'num_heads({num_heads}) must be divisible by '\
        f'num_kv_heads({num_kv_heads})).'

    # The number of attention heads per group: N q head + 1 k head + 1 v head.
    num_group_heads = num_heads // num_kv_heads + 2
    assert weight.shape[0] == num_kv_heads * num_group_heads * head_dim, \
        f'{weight.shape[0]} != {num_kv_heads} * {num_group_heads} * {head_dim}'

    qkv_in = num_heads * head_dim if not is_bias else 1

    # Split Q/K/V weights
    weight = weight.reshape(num_kv_heads, num_heads // num_kv_heads + 2,
                            head_dim, qkv_in)
    q_w = weight[:, :-2, ...]  # (nKV, num_heads // nKV, head_dim, qkv_in)
    k_w = weight[:, -2:-1, ...]  # (nKV, 1, head_dim, qkv_in)
    v_w = weight[:, -1:, ...]  # (nKV, 1, head_dim, qkv_in)

    if num_kv_heads < num_heads and num_kv_heads < tp_size:
        # Duplicate K/V heads to make sure that each rank has at least one
        # K/V heads. For instance, num_heads=8, num_kv_heads=2, tp_size=4,
        # we will make the qkv weight as below.
        #   Orig: [q0 q1 q2 q3 k0 v0 q4 q5 q6 q7 k1 v0 v1]
        #   >>>>  [[q0 q1 k0 v0], [q2 q3 k0 v0], [q4 q5 k1 v1], [q6 q7 k1 v1]]
        assert tp_size % num_kv_heads == 0
        num_dups = tp_size // num_kv_heads

        # k_w and v_w have the same shape.
        new_shape = (num_kv_heads, num_dups) + k_w.shape[2:]
        k_w = torch.broadcast_to(k_w, size=new_shape)
        v_w = torch.broadcast_to(v_w, size=new_shape)

        # Update the number of kv heads.
        num_kv_heads = tp_size

    reordered = torch.concat(
        [
            q_w.reshape(tp_size, num_heads // tp_size, head_dim, qkv_in),
            k_w.reshape(tp_size, num_kv_heads // tp_size, head_dim, qkv_in),
            v_w.reshape(tp_size, num_kv_heads // tp_size, head_dim, qkv_in),
        ],
        dim=1,
    )

    qkv_out = (num_heads + 2 * num_kv_heads) // tp_size * head_dim
    return reordered.reshape((tp_size, qkv_out, -1))


def split_qkv_weight(weight: torch.Tensor,
                     hidden_size: int,
                     num_heads: int,
                     tp_size: int,
                     rank: int,
                     is_bias: bool,
                     num_kv_heads: Optional[int] = None) -> torch.Tensor:
    """ Splits the QKV matrix according to tensor parallelism """
    head_dim = hidden_size // num_heads
    weight = reorder_qkv_weight_or_bias(weight,
                                        head_dim=head_dim,
                                        num_heads=num_heads,
                                        num_kv_heads=num_kv_heads,
                                        tp_size=tp_size,
                                        is_bias=is_bias)

    # Copy a sliced tensor to prevent memory leak. A sliced tensor shares the
    # memory buffer of the original tensor. So, returning without copying makes
    # the buffer of a loaded "qkv" be referenced, resulting GC can't release
    # those weights until the whole process ends.
    if not is_bias:
        return weight[rank, ...].clone().contiguous()
    else:
        return weight[rank, ...].ravel().clone().contiguous()


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


def convert_hf_falcon(hf_model: FalconForCausalLM,
                      hf_config: FalconConfig,
                      mapping: Mapping,
                      dtype: str = 'float32',
                      use_parallel_embedding: bool = False,
                      sharding_dim: int = 0,
                      share_embedding_table: bool = False,
                      use_weight_only: bool = False,
                      plugin_weight_only_quant_type: torch.dtype = torch.int8):
    weights = {}
    tik = time.time()

    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_attention_heads = hf_config.num_attention_heads
    hidden_size = hf_config.hidden_size
    vocab_size = hf_config.vocab_size
    num_kv_heads = getattr(hf_config, 'num_kv_heads', num_attention_heads)
    num_hidden_layers = hf_config.num_hidden_layers
    parallel_attention = hf_config.parallel_attn
    new_decoder_architecture = hf_config.new_decoder_architecture

    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        prefix = f'transformer.h.{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'
        qkv_weight, qkv_bias = get_weight_and_bias(
            model_params, f'{prefix}.self_attention.query_key_value', dtype)
        qkv_w = split_qkv_weight(qkv_weight,
                                 hidden_size,
                                 num_attention_heads,
                                 mapping.tp_size,
                                 mapping.tp_rank,
                                 is_bias=False,
                                 num_kv_heads=num_kv_heads)
        if qkv_bias is None:
            qkv_b = None
        else:
            qkv_b = split_qkv_weight(qkv_bias,
                                     hidden_size,
                                     num_attention_heads,
                                     mapping.tp_size,
                                     mapping.tp_rank,
                                     is_bias=True,
                                     num_kv_heads=num_kv_heads)
        weights.update(
            get_tllm_linear_weight(qkv_w, f'{tllm_prex}.attention.qkv', qkv_b,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))

        attn_dense_weight, attn_dense_bias = get_weight_and_bias(
            model_params, f'{prefix}.self_attention.dense', dtype)
        attn_dense_w = split_matrix(attn_dense_weight,
                                    mapping.tp_size,
                                    mapping.tp_rank,
                                    dim=1)
        weights.update(
            get_tllm_linear_weight(attn_dense_w, f'{tllm_prex}.attention.dense',
                                   attn_dense_bias, use_weight_only,
                                   plugin_weight_only_quant_type))

        mlp_fc_weight, mlp_fc_bias = get_weight_and_bias(
            model_params, f'{prefix}.mlp.dense_h_to_4h', dtype)
        mlp_fc_w = split_matrix(mlp_fc_weight,
                                mapping.tp_size,
                                mapping.tp_rank,
                                dim=0)
        if mlp_fc_bias is None:
            mlp_fc_b = None
        else:
            mlp_fc_b = split_matrix(mlp_fc_bias,
                                    mapping.tp_size,
                                    mapping.tp_rank,
                                    dim=0)
        weights.update(
            get_tllm_linear_weight(mlp_fc_w, f'{tllm_prex}.mlp.fc', mlp_fc_b,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))

        mlp_proj_weight, mlp_proj_bias = get_weight_and_bias(
            model_params, f'{prefix}.mlp.dense_4h_to_h', dtype)
        mlp_proj_w = split_matrix(mlp_proj_weight,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  dim=1)
        weights.update(
            get_tllm_linear_weight(mlp_proj_w, f'{tllm_prex}.mlp.proj',
                                   mlp_proj_bias, use_weight_only,
                                   plugin_weight_only_quant_type))

        if new_decoder_architecture:
            input_ln_weight, input_ln_bias = get_weight_and_bias(
                model_params, f'{prefix}.ln_attn', dtype)
            weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_weight
            if input_ln_bias is not None:
                weights[f'{tllm_prex}.input_layernorm.bias'] = input_ln_bias

            mlp_ln_weight, mlp_ln_bias = get_weight_and_bias(
                model_params, f'{prefix}.ln_mlp', dtype)
            weights[f'{tllm_prex}.mlp_layernorm.weight'] = mlp_ln_weight
            if mlp_ln_bias is not None:
                weights[f'{tllm_prex}.mlp_layernorm.bias'] = mlp_ln_bias
        else:
            input_ln_weight, input_ln_bias = get_weight_and_bias(
                model_params, f'{prefix}.input_layernorm', dtype)
            weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_weight
            if input_ln_bias is not None:
                weights[f'{tllm_prex}.input_layernorm.bias'] = input_ln_bias

            if not parallel_attention:
                post_ln_weight, post_ln_bias = get_weight_and_bias(
                    model_params, f'{prefix}.post_attention_layernorm', dtype)
                if post_ln_weight is not None:
                    weights[
                        f'{tllm_prex}.post_layernorm.weight'] = post_ln_weight
                if post_ln_bias is not None:
                    weights[f'{tllm_prex}.post_layernorm.bias'] = post_ln_bias

    embed_w = get_weight(model_params, 'transformer.word_embeddings', dtype)
    if mapping.is_first_pp_rank():
        if not use_parallel_embedding:
            weights['transformer.vocab_embedding.weight'] = embed_w
        else:
            if sharding_dim == 0:
                assert vocab_size % mapping.tp_size == 0
            else:
                assert hidden_size % mapping.tp_size == 0
            weights['transformer.vocab_embedding.weight'] = split_matrix(
                embed_w, mapping.tp_size, mapping.tp_rank, sharding_dim)

    if mapping.is_last_pp_rank():
        if not share_embedding_table:
            weights['lm_head.weight'] = split_matrix(embed_w.clone(),
                                                     mapping.tp_size,
                                                     mapping.tp_rank,
                                                     dim=0)
        ln_f_w, ln_f_b = get_weight_and_bias(model_params, 'transformer.ln_f',
                                             dtype)
        weights['transformer.ln_f.weight'] = ln_f_w
        if ln_f_b is not None:
            weights['transformer.ln_f.bias'] = ln_f_b

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def load_from_hf_falcon_checkpoint(
        hf_model_dir: str,
        hf_config: FalconConfig,
        mapping: Mapping,
        dtype: str = 'float32',
        use_parallel_embedding: bool = False,
        sharding_dim: int = 0,
        share_embedding_table: bool = False,
        use_weight_only: bool = False,
        plugin_weight_only_quant_type: torch.dtype = torch.int8):

    weights = {}
    tik = time.time()

    dtype = getattr(torch, dtype)
    num_attention_heads = hf_config.num_attention_heads
    hidden_size = hf_config.hidden_size
    vocab_size = hf_config.vocab_size
    num_kv_heads = getattr(hf_config, 'num_kv_heads', num_attention_heads)
    num_hidden_layers = hf_config.num_hidden_layers

    layers_range = mapping.pp_layers(num_hidden_layers)
    for model_file in iterate_shard_files(hf_model_dir, mapping.tp_rank):
        state_dict = load_state_dict(model_file, dtype)
        for name, param in state_dict.items():
            param = param.detach().cpu()
            l = retrieved_layer_index_from_name(name)
            if l is not None:
                if l not in layers_range:
                    continue
                tllm_prex = f'transformer.layers.{l-layers_range[0]}'

                if 'self_attention.query_key_value' in name:
                    if name.endswith('weight'):
                        qkv_w = split_qkv_weight(param,
                                                 hidden_size,
                                                 num_attention_heads,
                                                 mapping.tp_size,
                                                 mapping.tp_rank,
                                                 is_bias=False,
                                                 num_kv_heads=num_kv_heads)
                        weights.update(
                            get_tllm_param(qkv_w,
                                           f'{tllm_prex}.attention.qkv.weight',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))
                    else:
                        qkv_b = split_qkv_weight(param,
                                                 hidden_size,
                                                 num_attention_heads,
                                                 mapping.tp_size,
                                                 mapping.tp_rank,
                                                 is_bias=True,
                                                 num_kv_heads=num_kv_heads)
                        weights.update(
                            get_tllm_param(qkv_b,
                                           f'{tllm_prex}.attention.qkv.bias',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))

                elif 'self_attention.dense' in name:
                    if name.endswith('weight'):
                        attn_dense_w = split_matrix(param,
                                                    mapping.tp_size,
                                                    mapping.tp_rank,
                                                    dim=1)
                        weights.update(
                            get_tllm_param(
                                attn_dense_w,
                                f'{tllm_prex}.attention.dense.weight',
                                use_weight_only, plugin_weight_only_quant_type))
                    else:
                        weights.update(
                            get_tllm_param(param,
                                           f'{tllm_prex}.attention.dense.bias',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))

                elif 'mlp.dense_h_to_4h' in name:
                    if name.endswith('weight'):
                        mlp_fc_w = split_matrix(param,
                                                mapping.tp_size,
                                                mapping.tp_rank,
                                                dim=0)
                        weights.update(
                            get_tllm_param(mlp_fc_w,
                                           f'{tllm_prex}.mlp.fc.weight',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))
                    else:
                        mlp_fc_b = split_matrix(param,
                                                mapping.tp_size,
                                                mapping.tp_rank,
                                                dim=0)
                        weights.update(
                            get_tllm_param(mlp_fc_b, f'{tllm_prex}.mlp.fc.bias',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))

                elif 'mlp.dense_4h_to_h' in name:
                    if name.endswith('weight'):
                        mlp_proj_w = split_matrix(param,
                                                  mapping.tp_size,
                                                  mapping.tp_rank,
                                                  dim=1)
                        weights.update(
                            get_tllm_param(mlp_proj_w,
                                           f'{tllm_prex}.mlp.proj.weight',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))
                    else:
                        weights.update(
                            get_tllm_param(param, f'{tllm_prex}.mlp.proj.bias',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))

                elif 'ln_attn' in name or 'input_layernorm' in name:
                    if name.endswith('weight'):
                        weights[f'{tllm_prex}.input_layernorm.weight'] = param
                    else:
                        weights[f'{tllm_prex}.input_layernorm.bias'] = param
                elif 'ln_mlp' in name:
                    if name.endswith('weight'):
                        weights[f'{tllm_prex}.mlp_layernorm.weight'] = param
                    else:
                        weights[f'{tllm_prex}.mlp_layernorm.bias'] = param
                elif 'post_attention_layernorm' in name:
                    if name.endswith('weight'):
                        weights[f'{tllm_prex}.post_layernorm.weight'] = param
                    else:
                        weights[f'{tllm_prex}.post_layernorm.bias'] = param
            else:
                if 'word_embeddings' in name:
                    if mapping.is_first_pp_rank():
                        if not use_parallel_embedding:
                            weights[
                                'transformer.vocab_embedding.weight'] = param
                        else:
                            if sharding_dim == 0:
                                assert vocab_size % mapping.tp_size == 0
                            else:
                                assert hidden_size % mapping.tp_size == 0
                            weights[
                                'transformer.vocab_embedding.weight'] = split_matrix(
                                    param, mapping.tp_size, mapping.tp_rank,
                                    sharding_dim)

                    if mapping.is_last_pp_rank() and not share_embedding_table:
                        weights['lm_head.weight'] = split_matrix(
                            param.clone(),
                            mapping.tp_size,
                            mapping.tp_rank,
                            dim=0)
                elif 'ln_f' in name:
                    if mapping.is_last_pp_rank():
                        if name.endswith('weight'):
                            weights['transformer.ln_f.weight'] = param
                        else:
                            weights['transformer.ln_f.bias'] = param
        del state_dict

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def load_from_awq_falcon(quant_ckpt_path: str,
                         hf_config: FalconConfig,
                         mapping: Mapping,
                         use_parallel_embedding: bool = False,
                         sharding_dim: int = 0,
                         share_embedding_table: bool = False,
                         quantize_lm_head: bool = False,
                         dtype: str = "float16"):
    weights = {}
    tik = time.time()
    hidden_size = hf_config.hidden_size
    vocab_size = hf_config.vocab_size
    num_hidden_layers = hf_config.num_hidden_layers
    parallel_attention = hf_config.parallel_attn
    new_decoder_architecture = hf_config.new_decoder_architecture

    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
    torch_dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)

    if not quant_ckpt_path.endswith(".npz"):
        raise ValueError("Unsupported AWQ quantized checkpoint format")

    awq_falcon = np.load(quant_ckpt_path)
    awq_prefix = "_np:"
    awq_suffix_list = [
        ":weight",
        ":weights_scaling_factor",
        ":prequant_scaling_factor",
    ]
    awq_key_list = [
        "vocab_embedding:weight",  # embedding
        "lm_head",  # lm_head
        "final_layernorm",  # ln_f
        "attention:qkv:",  # attention.qkv
        "attention:dense",  # attention.dense
        "mlp:proj",  # mlp.proj
        "mlp:fc",  # mlp.fc
        "input_layernorm",  # input_layernorm.weight
        "mlp_layernorm",  # mlp_layernorm.weight
    ]
    split_sym = ":"
    AMMO_WEIGHT_SCALING_FACTOR_COEFF = 7

    def load(key):
        if awq_prefix + key not in awq_falcon:
            return None
        v = torch.from_numpy(awq_falcon[awq_prefix + key]).to(torch_dtype)
        if "weights_scaling_factor" in key:
            v *= AMMO_WEIGHT_SCALING_FACTOR_COEFF  # For AMMO *.npz checkpoints
        return v

    group_size = load("layers:0:attention:dense:weight").numel() // load(
        "layers:0:attention:dense:weights_scaling_factor").numel()

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            raise ValueError("Invalid TP size")
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def AWQ_quantize_pack_preprocess(weight, scale):
        weight /= scale.repeat_interleave(group_size, dim=0)
        qweight_int8 = torch.clamp(torch.round(weight.cuda()).char(), -8, 7)
        int4_weight = preprocessor(packer(qweight_int8.cpu()), torch.quint4x2)
        return int4_weight.view(torch.float16)

    def get_tllm_weight_from_awq(v: List[torch.Tensor],
                                 tllm_prex: str,
                                 tp_dim: int = 0):
        weight = v[0].T.contiguous()
        [k, n] = weight.shape
        weight = torch_split(weight, tp_dim)
        amax = v[1].reshape((n, k // group_size)).T.contiguous()
        amax = torch_split(amax, tp_dim)
        pre_quant_scale = v[2].reshape((1, k))
        if tp_dim == 0:
            pre_quant_scale = torch_split(pre_quant_scale, 1)
        scale = amax / 8.0
        results = {
            f'{tllm_prex}.weight': AWQ_quantize_pack_preprocess(weight, scale),
            f'{tllm_prex}.weights_scaling_factor': scale.to(torch_dtype),
            f'{tllm_prex}.prequant_scaling_factor':
            pre_quant_scale.to(torch_dtype),
        }
        return results

    def get_scale(weight):
        [k, n] = weight.shape
        weight_t = weight.T.contiguous()
        weight_t = weight_t.reshape(n, k // group_size, group_size)
        weight_t = torch.abs(weight_t.reshape(-1, group_size))
        amax, idx = weight_t.max(1)
        amax = amax.reshape(n, k // group_size).T.contiguous()
        scale = amax / 8
        return scale

    def get_tllm_qkv_weight_from_awq(prefix, tllm_prex: str):
        q_weight = load(prefix + "q" + awq_suffix_list[0]).T.contiguous()
        k_weight = load(prefix + "k" + awq_suffix_list[0]).T.contiguous()
        v_weight = load(prefix + "v" + awq_suffix_list[0]).T.contiguous()
        dim_k = q_weight.shape[0]
        q_weight = torch_split(q_weight, 1)
        k_weight = torch_split(k_weight, 1)
        v_weight = torch_split(v_weight, 1)
        qkv_pre_quant_scale = load(prefix + "q" + awq_suffix_list[2]).reshape(
            (1, dim_k))
        qkv_weights = torch.cat((q_weight, k_weight, v_weight), dim=1)
        qkv_scale = get_scale(qkv_weights)

        results = {
            f'{tllm_prex}.prequant_scaling_factor':
            qkv_pre_quant_scale.to(torch_dtype),
            f'{tllm_prex}.weight':
            AWQ_quantize_pack_preprocess(qkv_weights, qkv_scale),
            f'{tllm_prex}.weights_scaling_factor':
            qkv_scale.to(torch_dtype),
        }
        return results

    # Load weights from AWQ checkpoint into TRT-LLM module
    # 1. embedding
    v = load(awq_key_list[0])
    # TRT-LLM requires vocab_size to be multiple of 64 for successful GEMM
    if quantize_lm_head and v.shape[0] % 64 != 0:
        v = torch.nn.functional.pad(v, [0, 0, 0, 64 - v.shape[0] % 64])
    if mapping.is_first_pp_rank():
        if not use_parallel_embedding:
            weights['transformer.vocab_embedding.weight'] = v.to(torch_dtype)
        else:
            if sharding_dim == 0:
                assert vocab_size % mapping.tp_size == 0
            else:
                assert hidden_size % mapping.tp_size == 0
            weights['transformer.vocab_embedding.weight'] = split_matrix(
                v.to(torch_dtype), mapping.tp_size, mapping.tp_rank,
                sharding_dim)

    # 2. lm_head
    if mapping.is_last_pp_rank():
        if quantize_lm_head:
            assert not share_embedding_table
            v = [load(awq_key_list[1] + suf) for suf in awq_suffix_list]
            if v[0].shape[0] % 64 != 0:
                v[0] = torch.nn.functional.pad(
                    v[0], [0, 0, 0, 64 - v[0].shape[0] % 64])
                v[1] = torch.nn.functional.pad(
                    v[1], [0, 0, 0, 64 - v[1].shape[0] % 64], value=1)
            weights.update(get_tllm_weight_from_awq(v, 'lm_head', 1))
        elif not share_embedding_table:
            v = load(awq_key_list[1] + awq_suffix_list[0])
            weights['lm_head.weight'] = torch_split(v.to(torch_dtype), 0)

    # 3. ln_f
    v_weight = load(awq_key_list[2] + split_sym + "weight")
    v_bias = load(awq_key_list[2] + split_sym + "bias")
    if mapping.is_last_pp_rank():
        weights['transformer.ln_f.weight'] = v_weight.to(torch_dtype)
        weights['transformer.ln_f.bias'] = v_bias.to(torch_dtype)

    # 4. Weights inside each layer
    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        prefix = f'layers{split_sym}{l}{split_sym}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'

        # 4.1 attention.qkv
        weights.update(
            get_tllm_qkv_weight_from_awq(prefix + awq_key_list[3],
                                         f'{tllm_prex}.attention.qkv'))
        q_b = load(prefix + awq_key_list[3] + 'q:bias')
        k_b = load(prefix + awq_key_list[3] + 'k:bias')
        v_b = load(prefix + awq_key_list[3] + 'v:bias')
        if q_b is not None:
            q_b = torch_split(q_b, dim=0)
            k_b = torch_split(k_b, dim=0)
            v_b = torch_split(v_b, dim=0)
            qkv_b = torch.cat((q_b, k_b, v_b), dim=0)
            weights[f'{tllm_prex}.attention.qkv.bias'] = qkv_b

        # 4.2 attention.dense
        v = [load(prefix + awq_key_list[4] + suf) for suf in awq_suffix_list]
        weights.update(
            get_tllm_weight_from_awq(v,
                                     f'{tllm_prex}.attention.dense',
                                     tp_dim=0))
        b = load(prefix + awq_key_list[4] + ':bias')
        if b is not None:
            if mapping.tp_rank > 0:
                b = torch.zeros_like(b)
            weights[f'{tllm_prex}.attention.dense.bias'] = b

        # 4.4 mlp.fc
        v = [load(prefix + awq_key_list[6] + suf) for suf in awq_suffix_list]
        weights.update(
            get_tllm_weight_from_awq(v, f'{tllm_prex}.mlp.fc', tp_dim=1))
        b = load(prefix + awq_key_list[6] + ':bias')
        if b is not None:
            b = torch_split(b, dim=0)
            weights[f'{tllm_prex}.mlp.fc.bias'] = b

        # 4.3 mlp.proj
        v = [load(prefix + awq_key_list[5] + suf) for suf in awq_suffix_list]
        weights.update(
            get_tllm_weight_from_awq(v, f'{tllm_prex}.mlp.proj', tp_dim=0))
        b = load(prefix + awq_key_list[5] + ':bias')
        if b is not None:
            if mapping.tp_rank > 0:
                b = torch.zeros_like(b)
            weights[f'{tllm_prex}.mlp.proj.bias'] = b

        if new_decoder_architecture:
            # 4.5 input_layernorm
            v = load(prefix + awq_key_list[7] + split_sym + "weight")
            weights[f'{tllm_prex}.input_layernorm.weight'] = v.to(torch_dtype)
            v = load(prefix + awq_key_list[7] + split_sym + "bias")
            weights[f'{tllm_prex}.input_layernorm.bias'] = v.to(torch_dtype)

            # 4.6 mlp_layernorm
            v = load(prefix + awq_key_list[8] + split_sym + "weight")
            weights[f'{tllm_prex}.mlp_layernorm.weight'] = v.to(torch_dtype)
            v = load(prefix + awq_key_list[8] + split_sym + "bias")
            weights[f'{tllm_prex}.mlp_layernorm.bias'] = v.to(torch_dtype)
        else:
            # 4.5 input_layernorm
            v = load(prefix + awq_key_list[7] + split_sym + "weight")
            weights[f'{tllm_prex}.input_layernorm.weight'] = v.to(torch_dtype)
            v = load(prefix + awq_key_list[7] + split_sym + "bias")
            weights[f'{tllm_prex}.input_layernorm.bias'] = v.to(torch_dtype)

            if not parallel_attention:
                # 4.6 post_layernorm
                v = load(prefix + 'post_layernorm' + split_sym + "weight")
                weights[f'{tllm_prex}.post_layernorm.weight'] = v.to(
                    torch_dtype)
                v = load(prefix + 'post_layernorm' + split_sym + "bias")
                weights[f'{tllm_prex}.post_layernorm.bias'] = v.to(torch_dtype)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Elapsed time: {t}')
    return weights


def load_from_fp8_falcon(quant_ckpt_path: str, hf_config: FalconConfig,
                         mapping: Mapping, fp8_kv_cache: bool):
    """
    Get the fp8 scaling factors for Falcon model.
    """
    fake_fp8_sf_dt = torch.float32

    fp8_falcon = np.load(quant_ckpt_path)
    weights = {}
    num_hidden_layers = hf_config.num_hidden_layers

    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        prefix = f'_np:layers:{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'

        weights[f'{tllm_prex}.attention.qkv.activation_scaling_factor'] = torch.tensor(
            [
                max(
                    fp8_falcon[
                        f'{prefix}:attention:qkv:q:activation_scaling_factor'].
                    item(), fp8_falcon[
                        f'{prefix}:attention:qkv:k:activation_scaling_factor'].
                    item(), fp8_falcon[
                        f'{prefix}:attention:qkv:v:activation_scaling_factor'].
                    item())
            ],
            dtype=fake_fp8_sf_dt)
        weights[
            f'{tllm_prex}.attention.qkv.weights_scaling_factor'] = torch.tensor(
                [
                    max(
                        fp8_falcon[
                            f'{prefix}:attention:qkv:q:weights_scaling_factor'].
                        item(), fp8_falcon[
                            f'{prefix}:attention:qkv:k:weights_scaling_factor'].
                        item(), fp8_falcon[
                            f'{prefix}:attention:qkv:v:weights_scaling_factor'].
                        item())
                ],
                dtype=fake_fp8_sf_dt)
        weights[
            f'{tllm_prex}.attention.dense.activation_scaling_factor'] = torch.tensor(
                [
                    fp8_falcon[
                        f'{prefix}:attention:dense:activation_scaling_factor'].
                    item()
                ],
                dtype=fake_fp8_sf_dt)
        weights[
            f'{tllm_prex}.attention.dense.weights_scaling_factor'] = torch.tensor(
                [
                    fp8_falcon[
                        f'{prefix}:attention:dense:weights_scaling_factor'].
                    item()
                ],
                dtype=fake_fp8_sf_dt)

        weights[f'{tllm_prex}.mlp.fc.activation_scaling_factor'] = torch.tensor(
            [fp8_falcon[f'{prefix}:mlp:fc:activation_scaling_factor'].item()],
            dtype=fake_fp8_sf_dt)
        weights[f'{tllm_prex}.mlp.fc.weights_scaling_factor'] = torch.tensor(
            [fp8_falcon[f'{prefix}:mlp:fc:weights_scaling_factor'].item()],
            dtype=fake_fp8_sf_dt)
        weights[
            f'{tllm_prex}.mlp.proj.activation_scaling_factor'] = torch.tensor(
                [
                    fp8_falcon[f'{prefix}:mlp:proj:activation_scaling_factor'].
                    item()
                ],
                dtype=fake_fp8_sf_dt)
        weights[f'{tllm_prex}.mlp.proj.weights_scaling_factor'] = torch.tensor(
            [fp8_falcon[f'{prefix}:mlp:proj:weights_scaling_factor'].item()],
            dtype=fake_fp8_sf_dt)

        if fp8_kv_cache:
            # Not calibrarting KV cache.
            scaling_factor = 1.0
            weights[
                f'{tllm_prex}.attention.kv_cache_scaling_factor'] = torch.tensor(
                    [scaling_factor], dtype=fake_fp8_sf_dt)

    return weights


if __name__ == '__main__':
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    assert args.tp_size * args.pp_size == args.world_size

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    hf_config = load_falcon_config(args.model_dir)
    config = {
        'architecture': hf_config.architectures[0],
        'dtype': args.dtype,
        'logits_dtype': args.logits_dtype,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_attention_heads,
        'num_key_value_heads': hf_config.num_kv_heads,
        'hidden_size': hf_config.hidden_size,
        'norm_epsilon': hf_config.layer_norm_epsilon,
        'vocab_size': hf_config.vocab_size,
        'position_embedding_type':
        'alibi_with_scale' if hf_config.alibi else 'rope_gpt_neox',
        'max_position_embeddings': hf_config.max_position_embeddings,
        'hidden_act': 'gelu',
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'share_embedding_table': args.use_embedding_sharing,
        'quantization': {
            'use_weight_only': args.use_weight_only,
            'weight_only_precision': args.weight_only_precision,
            'per_group': args.per_group,
            'group_size': args.group_size,
            'enable_fp8': args.enable_fp8,
            'fp8_kv_cache': args.fp8_kv_cache,
        },
        'mapping': {
            'world_size': args.world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'bias': hf_config.bias,
        'parallel_attention': hf_config.parallel_attn,
        'new_decoder_architecture': hf_config.new_decoder_architecture,
    }
    if args.weight_only_precision == 'int4_awq':
        exclude_modules = ['lm_head'] if not args.quantize_lm_head else []
        config['quantization'].update({
            'zero': False,
            'pre_quant_scale': True,
            'exclude_modules': exclude_modules,
        })
        if args.quantize_lm_head and config['vocab_size'] % 64 != 0:
            config['vocab_size'] = int((config['vocab_size'] + 63) / 64) * 64

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    if args.weight_only_precision == 'int8':
        plugin_weight_only_quant_type = torch.int8
    elif args.weight_only_precision == 'int4':
        plugin_weight_only_quant_type = torch.quint4x2

    def covert_and_save(rank):
        mapping = Mapping(world_size=args.tp_size * args.pp_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        if args.use_weight_only and args.weight_only_precision == 'int4_awq':
            weights = load_from_awq_falcon(
                args.ammo_quant_ckpt_path,
                hf_config,
                mapping,
                use_parallel_embedding=args.use_parallel_embedding,
                sharding_dim=args.embedding_sharding_dim,
                share_embedding_table=args.use_embedding_sharing,
                quantize_lm_head=args.quantize_lm_head,
                dtype=args.dtype)
        else:
            if args.load_by_shard:
                weights = load_from_hf_falcon_checkpoint(
                    args.model_dir,
                    hf_config,
                    mapping,
                    dtype=args.dtype,
                    use_parallel_embedding=args.use_parallel_embedding,
                    sharding_dim=args.embedding_sharding_dim,
                    share_embedding_table=args.use_embedding_sharing,
                    use_weight_only=args.use_weight_only,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            else:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    args.model_dir, trust_remote_code=True, torch_dtype="auto")
                weights = convert_hf_falcon(
                    hf_model,
                    hf_config,
                    mapping,
                    dtype=args.dtype,
                    use_parallel_embedding=args.use_parallel_embedding,
                    sharding_dim=args.embedding_sharding_dim,
                    share_embedding_table=args.use_embedding_sharing,
                    use_weight_only=args.use_weight_only,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type)
                del hf_model

            if args.enable_fp8 or args.fp8_kv_cache:
                scales = load_from_fp8_falcon(args.ammo_quant_ckpt_path,
                                              hf_config, mapping,
                                              args.fp8_kv_cache)
                weights.update(scales)

        safetensors.torch.save_file(
            weights, os.path.join(args.output_dir, f'rank{rank}.safetensors'))

    if args.workers == 1:
        for rank in range(args.world_size):
            covert_and_save(rank)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as p:
            futures = [
                p.submit(covert_and_save, rank)
                for rank in range(args.world_size)
            ]
            wait(futures)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')
