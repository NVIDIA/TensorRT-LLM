import time
from typing import Dict, Optional

import torch

from tensorrt_llm.quantization import QuantAlgo

from ..convert_utils import get_weight, get_weight_and_bias, split_matrix_tp
from .config import GPTJConfig


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


def load_weights_from_hf_model(hf_model, config: GPTJConfig):
    quant_algo = config.quantization.quant_algo
    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    if quant_algo == QuantAlgo.W8A16:
        plugin_weight_only_quant_type = torch.int8
    elif quant_algo == QuantAlgo.W4A16:
        plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None

    weights = {}
    tik = time.time()

    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, config.dtype)
    num_hidden_layers = config.num_hidden_layers
    mapping = config.mapping

    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        prefix = f'transformer.h.{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'
        # Attention QKV (no bias)
        q_weight = get_weight(model_params, f'{prefix}.attn.q_proj', dtype)
        k_weight = get_weight(model_params, f'{prefix}.attn.k_proj', dtype)
        v_weight = get_weight(model_params, f'{prefix}.attn.v_proj', dtype)
        q_w = split_matrix_tp(q_weight, mapping.tp_size, mapping.tp_rank, dim=0)
        k_w = split_matrix_tp(k_weight, mapping.tp_size, mapping.tp_rank, dim=0)
        v_w = split_matrix_tp(v_weight, mapping.tp_size, mapping.tp_rank, dim=0)
        qkv_w = torch.concatenate([q_w, k_w, v_w], dim=0)
        weights.update(
            get_tllm_linear_weight(qkv_w, f'{tllm_prex}.attention.qkv', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))
        # Attention dense (not bias)
        attn_dense_weight = get_weight(model_params, f'{prefix}.attn.out_proj',
                                       dtype)
        attn_dense_w = split_matrix_tp(attn_dense_weight,
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
        mlp_fc_w = split_matrix_tp(mlp_fc_weight,
                                   mapping.tp_size,
                                   mapping.tp_rank,
                                   dim=0)
        mlp_fc_b = split_matrix_tp(mlp_fc_bias,
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
        mlp_proj_w = split_matrix_tp(mlp_proj_weight,
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
        if config.use_parallel_embedding:
            embed_w = split_matrix_tp(embed_w,
                                      mapping.tp_size,
                                      mapping.tp_rank,
                                      dim=0)
        weights['transformer.vocab_embedding.weight'] = embed_w

    if mapping.is_last_pp_rank():
        # lm_head weight and bias
        lm_head_w, ln_head_bias = get_weight_and_bias(model_params, 'lm_head',
                                                      dtype)
        weights['lm_head.weight'] = split_matrix_tp(lm_head_w,
                                                    mapping.tp_size,
                                                    mapping.tp_rank,
                                                    dim=0)
        weights['lm_head.bias'] = split_matrix_tp(ln_head_bias,
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
