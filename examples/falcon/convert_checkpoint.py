import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import safetensors
import torch
from transformers import AutoModelForCausalLM, FalconConfig, FalconForCausalLM

import tensorrt_llm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import (iterate_shard_files,
                                               load_state_dict,
                                               retrieved_layer_index_from_name)
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
    parser.add_argument('--log_level', type=str, default='info')
    args = parser.parse_args()

    tensorrt_llm.logger.set_level(args.log_level)
    return args


def load_falcon_config(model_dir: str) -> FalconConfig:
    """ Helper utility to load FalconConfig.

    A pretrained checkpoint from modeling_RW.py has a different structure
    and is not compatible with `transformers.FalconConfig` and
    `transformers.FalconModel`. We need to manually set the config values.
    """

    config = FalconConfig.from_pretrained(model_dir)
    config.architectures = ["FalconForCausalLM"]
    # Falcon-7B config may not have num_kv_heads or n_head_kv.
    # Although Falcon-180B uses GQA (num_kv_heads=8), its config
    # has multi_query=True.
    if getattr(config, 'multi_query', False) and \
            not getattr(config, 'new_decoder_architecture', False):
        config.num_kv_heads = 1

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
        return torch.chunk(weight, tp_size)[rank].clone()
    else:
        return torch.chunk(weight, tp_size, dim=dim)[rank].clone()


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
        return weight[rank, ...].clone()
    else:
        return weight[rank, ...].ravel().clone()


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
        results[f'{prefix}.weight'] = weight

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
            l = retrieved_layer_index_from_name(name)
            if l is not None:
                if l not in layers_range:
                    continue
                prefix = f'transformer.layers.{l-layers_range[0]}'

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
                                           f'{prefix}.attention.qkv.weight',
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
                                           f'{prefix}.attention.qkv.bias',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))

                elif 'self_attention.dense' in name:
                    if name.endswith('weight'):
                        attn_dense_w = split_matrix(param,
                                                    mapping.tp_size,
                                                    mapping.tp_rank,
                                                    dim=1)
                        weights.update(
                            get_tllm_param(attn_dense_w,
                                           f'{prefix}.attention.dense.weight',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))
                    else:
                        weights.update(
                            get_tllm_param(param,
                                           f'{prefix}.attention.dense.bias',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))

                elif 'mlp.dense_h_to_4h' in name:
                    if name.endswith('weight'):
                        mlp_fc_w = split_matrix(param,
                                                mapping.tp_size,
                                                mapping.tp_rank,
                                                dim=0)
                        weights.update(
                            get_tllm_param(mlp_fc_w, f'{prefix}.mlp.fc.weight',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))
                    else:
                        mlp_fc_b = split_matrix(param,
                                                mapping.tp_size,
                                                mapping.tp_rank,
                                                dim=0)
                        weights.update(
                            get_tllm_param(mlp_fc_b, f'{prefix}.mlp.fc.bias',
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
                                           f'{prefix}.mlp.proj.weight',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))
                    else:
                        weights.update(
                            get_tllm_param(param, f'{prefix}.mlp.proj.bias',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))

                elif 'ln_attn' in name or 'input_layernorm' in name:
                    if name.endswith('weight'):
                        weights[f'{prefix}.input_layernorm.weight'] = param
                    else:
                        weights[f'{prefix}.input_layernorm.bias'] = param
                elif 'ln_mlp' in name:
                    if name.endswith('weight'):
                        weights[f'{prefix}.mlp_layernorm.weight'] = param
                    else:
                        weights[f'{prefix}.mlp_layernorm.bias'] = param
                elif 'post_attention_layernorm' in name:
                    if name.endswith('weight'):
                        weights[f'{prefix}.post_layernorm.weight'] = param
                    else:
                        weights[f'{prefix}.post_layernorm.bias'] = param
            elif 'word_embeddings' in name:
                if mapping.is_first_pp_rank():
                    if not use_parallel_embedding:
                        weights['transformer.vocab_embedding.weight'] = param
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
                    weights['lm_head.weight'] = split_matrix(param,
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


if __name__ == '__main__':
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

    hf_config = load_falcon_config(args.model_dir)
    config = {
        'architecture': hf_config.architectures[0],
        'dtype': args.dtype,
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
            'quant_algo': quant_algo,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'bias': hf_config.bias,
        'parallel_attention': hf_config.parallel_attn,
        'new_decoder_architecture': hf_config.new_decoder_architecture,
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    def covert_and_save(rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

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
    print(f'Total time of converting checkpoints: {t}')
