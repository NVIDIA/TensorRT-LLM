import copy
import functools
import os
import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import safetensors
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.models import ChatGLMConfig
from tensorrt_llm.models.convert_utils import (generate_int8, get_weight,
                                               get_weight_and_bias,
                                               load_calib_dataset, smooth_gemm)
from tensorrt_llm.quantization import QuantAlgo

from .config import GLM_ARCH1_VERSIONS, GLM_ARCH2_VERSIONS


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


def tile_kv_weight_bias(v: torch.Tensor, kv_num_head: int, tp_size: int):
    head_size = v.shape[0] // kv_num_head
    reps = tp_size // kv_num_head
    if v.ndim == 1:
        v = v.reshape(kv_num_head, head_size)[:, None, :]
        v = v.expand(kv_num_head, reps, head_size).reshape(-1).clone()
    else:
        hidden_size = v.shape[1]
        v = v.reshape(kv_num_head, head_size, hidden_size)[:, None, :, :]
        v = v.expand(kv_num_head, reps, head_size,
                     hidden_size).reshape(-1, hidden_size).clone()
    return v


def split_qkv(v: torch.Tensor, tp_size: int, rank: int, hidden_size: int,
              num_heads: int, num_kv_heads: int):
    head_size = hidden_size // num_heads
    if tp_size == 1:
        return v

    assert v.shape[0] == hidden_size + head_size * num_kv_heads * 2
    query = v[:hidden_size]
    key = v[hidden_size:hidden_size + head_size * num_kv_heads]
    value = v[hidden_size + head_size * num_kv_heads:hidden_size +
              head_size * num_kv_heads * 2]

    if num_kv_heads < tp_size:
        key = tile_kv_weight_bias(key, num_kv_heads, tp_size)
        value = tile_kv_weight_bias(value, num_kv_heads, tp_size)
    assert (key.shape[0] % (tp_size * head_size)) == 0
    assert (value.shape[0] % (tp_size * head_size)) == 0

    q_tmp = torch.chunk(query, tp_size, dim=0)[rank]
    k_tmp = torch.chunk(key, tp_size, dim=0)[rank]
    v_tmp = torch.chunk(value, tp_size, dim=0)[rank]
    return torch.concatenate([q_tmp, k_tmp, v_tmp], dim=0).contiguous()


def split_embedding(
    param: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    use_parallel_embedding: bool = False,
    sharding_dim: int = 0,
) -> torch.Tensor:
    if param is None:
        return None
    if not use_parallel_embedding:
        return param

    vocab_size, hidden_size = param.size()
    if sharding_dim == 0:
        if vocab_size % tp_size != 0:
            vocab_size_padded = pad_vocab_size(vocab_size, tp_size)
            pad_width = vocab_size_padded - vocab_size
            param = torch.nn.functional.pad(param, (0, 0, 0, pad_width),
                                            value=0)
        else:
            assert hidden_size % tp_size == 0
    return split(param, tp_size, tp_rank, dim=sharding_dim)


def swap_and_split_mlp(weight: torch.Tensor,
                       tp_size: int,
                       tp_rank: int,
                       dim: int = 0) -> torch.Tensor:
    """Swap the positions of gate and fc weights, and split weights for tensor parallel.
    """
    gate_weight, fc_weight = torch.chunk(weight, 2, dim=dim)
    fc_w = split(fc_weight, tp_size, tp_rank, dim=dim)
    gate_w = split(gate_weight, tp_size, tp_rank, dim=dim)
    return torch.cat([fc_w, gate_w], dim=dim).contiguous()


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


@torch.no_grad()
def capture_activation_range(
    model,
    tokenizer,
    dataset,
    num_samples=64,
    seq_len=512,
):

    model.eval()
    device = next(model.parameters()).device
    scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    def stat_tensor(name, tensor, key):
        tensor = tensor.view(-1, tensor.shape[-1]).detach()
        comming_max = tensor.abs().max(dim=0)[0].float()
        if scales[name][key] is None:
            scales[name][key] = comming_max
        else:
            scales[name][key] = torch.max(scales[name][key], comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, "x")
        stat_tensor(name, y, "y")
        # TODO: we don't need to do it every forward because inference does not change weight
        if scales[name]["w"] is None:
            scales[name]["w"] = m.weight.abs().clip(1e-8, None).max(dim=1)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="Calibration"):
        input_ids = tokenizer(
            dataset[i],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
        )
        model(input_ids.input_ids.to(device))

    for h in hooks:
        h.remove()

    return scales


@torch.no_grad()
def smooth_chatglm_model(
    model,
    act_range,
    alpha,
    model_smoother,
):
    for name, module in model.named_modules():
        if not module._get_name() == "GLMBlock":
            continue

        # QKV multiplication weight
        layer_name = name + '.self_attention.query_key_value'
        print(f'Smoothing module: {layer_name}')
        weight = module.self_attention.query_key_value.weight
        smoother = smooth_gemm(
            weight,
            act_range[layer_name]["x"],
            module.input_layernorm.weight,
            None,
            alpha,
        )
        act_range[layer_name]["x"] = act_range[layer_name]["x"] / smoother
        act_range[layer_name]["w"] = weight.abs().max(dim=1)[0]

        # Dense multiplication weight
        layer_name = name + ".self_attention.dense"
        print(f'Smoothing module: {layer_name}')
        weight = module.self_attention.dense.weight
        smoother = smooth_gemm(
            weight,
            act_range[layer_name]["x"],
            None,
            None,
            alpha,
        )
        model_smoother[layer_name] = smoother.float()
        act_range[layer_name]["x"] = act_range[layer_name]["x"] / smoother
        act_range[layer_name]["w"] = weight.abs().max(dim=1)[0]

        # Multilayer perceptron h -> 4h weight
        layer_name = name + ".mlp.dense_h_to_4h"
        print(f'Smoothing module: {layer_name}')
        weight = module.mlp.dense_h_to_4h.weight
        smoother = smooth_gemm(
            weight,
            act_range[layer_name]["x"],
            module.post_attention_layernorm.weight,
            None,
            alpha,
        )
        act_range[layer_name]["x"] = act_range[layer_name]["x"] / smoother
        act_range[layer_name]["w"] = weight.abs().max(dim=1)[0]

        # Multilayer perceptron 4h -> h weight
        layer_name = name + ".mlp.dense_4h_to_h"
        print(f'Smoothing module: {layer_name}')
        weight = module.mlp.dense_4h_to_h.weight
        smoother = smooth_gemm(
            weight,
            act_range[layer_name]["x"],
            None,
            None,
            alpha,
        )
        model_smoother[layer_name] = smoother.float()
        act_range[layer_name]["x"] = act_range[layer_name]["x"] / smoother
        act_range[layer_name]["w"] = weight.abs().max(dim=1)[0]


def get_tllm_linear_sq_weight(vals,
                              prefix,
                              shape,
                              is_qkv=False,
                              per_token=False,
                              per_channel=False,
                              last_prefix=None,
                              smoother_value=None,
                              smoother_shape=None):
    results = {}
    col_shape = shape if (is_qkv or per_channel) else [1, 1]

    if per_token:
        if per_channel:
            original_weights = np.array(vals["weight.int8.col"])
        else:
            original_weights = np.array(vals["weight.int8"])

        cur_weights = original_weights
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix +
                'weight'] = torch.from_numpy(cur_weights).t().contiguous()
        if smoother_value is None:
            results[last_prefix] = torch.from_numpy(
                np.array([1.0], dtype=np.float16))

        if per_channel:
            cur_per_channel_value = vals["scale_w_quant_orig.col"]
        else:
            cur_per_channel_value = vals["scale_w_quant_orig"]

        results[prefix +
                'per_channel_scale'] = cur_per_channel_value.reshape(col_shape)
    else:
        if per_channel:
            original_weights = np.array(vals["weight.int8.col"])
        else:
            original_weights = np.array(vals["weight.int8"])
        cur_weights = original_weights

        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix +
                'weight'] = torch.from_numpy(cur_weights).t().contiguous()

        if per_channel:
            cur_per_channel_value = vals["scale_y_accum_quant.col"]
        else:
            cur_per_channel_value = vals["scale_y_accum_quant"]

        results[prefix + 'per_channel_scale'] = torch.from_numpy(
            np.array([cur_per_channel_value],
                     dtype=np.float32).reshape(col_shape)).contiguous()

        results[last_prefix] = torch.from_numpy(
            np.array([vals['scale_x_orig_quant']],
                     dtype=np.float32)).contiguous()

        results[prefix + 'act_scale'] = torch.from_numpy(
            np.array([[vals["scale_y_quant_orig"]]],
                     dtype=np.float32)).contiguous()

    if smoother_value is not None:
        results[prefix + 'smoother'] = smoother_value.reshape(
            smoother_shape).contiguous().to(torch.float32)

    return results


def load_weights_from_hf_model(hf_model: AutoModel,
                               config: ChatGLMConfig,
                               act_range: Optional[dict] = None,
                               smoother: Optional[dict] = None):
    weights = {}
    tik = time.time()

    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, config.dtype)
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    num_kv_heads = config.num_key_value_heads
    num_hidden_layers = config.num_hidden_layers

    chatglm_version = config.chatglm_version
    mapping = config.mapping
    use_parallel_embedding = config.use_parallel_embedding
    sharding_dim = config.embedding_sharding_dim

    quant_algo = config.quantization.quant_algo
    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    use_smooth_quant = config.quantization._use_plugin_sq
    per_channel = use_smooth_quant and 'PER_CHANNEL' in quant_algo
    per_token = use_smooth_quant and 'PER_TOKEN' in quant_algo
    int8_kv_cache = config.quantization.kv_cache_quant_algo == QuantAlgo.INT8

    if use_weight_only:
        if quant_algo == QuantAlgo.W8A16:
            plugin_weight_only_quant_type = torch.int8
        elif quant_algo == QuantAlgo.W4A16:
            plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None

    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        if chatglm_version in GLM_ARCH1_VERSIONS:
            prefix = f'transformer.layers.{l}'
        elif chatglm_version in GLM_ARCH2_VERSIONS:
            prefix = f'transformer.encoder.layers.{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'

        # Attention QKV
        attention_attr_name = ''
        if chatglm_version in GLM_ARCH1_VERSIONS:
            attention_attr_name = 'attention'
        elif chatglm_version in GLM_ARCH2_VERSIONS:
            attention_attr_name = 'self_attention'
        qkv_weight, qkv_bias = get_weight_and_bias(
            model_params, f'{prefix}.{attention_attr_name}.query_key_value',
            dtype)

        if use_smooth_quant:
            qkv_act_range = act_range.get(
                f'{prefix}.{attention_attr_name}.query_key_value')
            qkv_vals_int8 = generate_int8(qkv_weight.t(),
                                          qkv_act_range,
                                          is_qkv=True,
                                          multi_query_mode=True)
            weights.update(
                get_tllm_linear_sq_weight(
                    vals=qkv_vals_int8,
                    prefix=f'{tllm_prex}.attention.qkv.',
                    shape=[1, qkv_weight.size(0)],
                    is_qkv=True,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=f'{tllm_prex}.input_layernorm.scale_to_int',
                    smoother_value=None,
                    smoother_shape=None))
            if qkv_bias is not None:
                qkv_b = split_qkv(qkv_bias,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  hidden_size,
                                  num_attention_heads,
                                  num_kv_heads=num_kv_heads)
                weights[f'{tllm_prex}.attention.qkv.bias'] = qkv_b
        else:
            qkv_w = split_qkv(qkv_weight,
                              mapping.tp_size,
                              mapping.tp_rank,
                              hidden_size,
                              num_attention_heads,
                              num_kv_heads=num_kv_heads)
            if qkv_bias is None:
                qkv_b = None
            else:
                qkv_b = split_qkv(qkv_bias,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  hidden_size,
                                  num_attention_heads,
                                  num_kv_heads=num_kv_heads)

            weights.update(
                get_tllm_linear_weight(qkv_w, f'{tllm_prex}.attention.qkv',
                                       qkv_b, use_weight_only,
                                       plugin_weight_only_quant_type))

        if int8_kv_cache:
            qkv_act_range = act_range.get(
                f'{prefix}.{attention_attr_name}.query_key_value')
            qkv_vals_int8 = generate_int8(qkv_weight.t(),
                                          qkv_act_range,
                                          is_qkv=True,
                                          multi_query_mode=True)
            weights[
                f'{tllm_prex}.attention.kv_cache_scaling_factor'] = qkv_vals_int8[
                    'scale_y_quant_orig'].contiguous()

        # Attention dense
        attn_dense_weight, attn_dense_bias = get_weight_and_bias(
            model_params, f'{prefix}.{attention_attr_name}.dense', dtype)

        if use_smooth_quant:
            dense_act_range = act_range.get(
                f'{prefix}.{attention_attr_name}.dense')
            dense_smoother = smoother.get(
                f'{prefix}.{attention_attr_name}.dense')
            dense_vals_int8 = generate_int8(attn_dense_weight.t(),
                                            dense_act_range,
                                            is_qkv=False,
                                            multi_query_mode=True)
            weights.update(
                get_tllm_linear_sq_weight(
                    vals=dense_vals_int8,
                    prefix=f'{tllm_prex}.attention.dense.',
                    shape=[1, hidden_size],
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=
                    f'{tllm_prex}.attention.quantization_scaling_factor',
                    smoother_value=dense_smoother,
                    smoother_shape=[1, hidden_size]))
            if attn_dense_bias is not None:
                weights[f'{tllm_prex}.attention.dense.bias'] = attn_dense_bias
        else:
            attn_dense_w = split(attn_dense_weight,
                                 mapping.tp_size,
                                 mapping.tp_rank,
                                 dim=1)
            weights.update(
                get_tllm_linear_weight(attn_dense_w,
                                       f'{tllm_prex}.attention.dense',
                                       attn_dense_bias, use_weight_only,
                                       plugin_weight_only_quant_type))

        # MLP FC
        mlp_fc_weight, mlp_fc_bias = get_weight_and_bias(
            model_params, f'{prefix}.mlp.dense_h_to_4h', dtype)

        if use_smooth_quant:
            fc_act_range = act_range.get(f'{prefix}.mlp.dense_h_to_4h')
            fc_vals_int8 = generate_int8(mlp_fc_weight.t(),
                                         fc_act_range,
                                         is_qkv=False,
                                         multi_query_mode=True)
            cur_weights = get_tllm_linear_sq_weight(
                vals=fc_vals_int8,
                prefix=f'{tllm_prex}.mlp.fc.',
                shape=[1, mlp_fc_weight.size(0)],
                is_qkv=False,
                per_token=per_token,
                per_channel=per_channel,
                last_prefix=f'{tllm_prex}.post_layernorm.scale_to_int',
                smoother_value=None,
                smoother_shape=None,
            )
            cur_weights[f'{tllm_prex}.mlp.fc.weight'] = swap_and_split_mlp(
                cur_weights[f'{tllm_prex}.mlp.fc.weight'],
                mapping.tp_size,
                mapping.tp_rank,
                dim=0,
            )
            if per_channel:
                cur_weights[
                    f'{tllm_prex}.mlp.fc.per_channel_scale'] = swap_and_split_mlp(
                        cur_weights[f'{tllm_prex}.mlp.fc.per_channel_scale'],
                        mapping.tp_size,
                        mapping.tp_rank,
                        dim=1,
                    )
            weights.update(cur_weights)
            if chatglm_version in GLM_ARCH1_VERSIONS:
                if mlp_fc_bias is not None:
                    mlp_fc_b = split(mlp_fc_bias,
                                     mapping.tp_size,
                                     mapping.tp_rank,
                                     dim=0)
                    weights[f'{tllm_prex}.mlp.fc.bias'] = mlp_fc_b
            elif chatglm_version in GLM_ARCH2_VERSIONS:
                if mlp_fc_bias is not None:
                    mlp_fc_b = swap_and_split_mlp(mlp_fc_bias, mapping.tp_size,
                                                  mapping.tp_rank)
                    weights[f'{tllm_prex}.mlp.fc.bias'] = mlp_fc_b
        else:
            if chatglm_version in GLM_ARCH1_VERSIONS:
                mlp_fc_w = split(mlp_fc_weight,
                                 mapping.tp_size,
                                 mapping.tp_rank,
                                 dim=0)
                if mlp_fc_bias is None:
                    mlp_fc_b = None
                else:
                    mlp_fc_b = split(mlp_fc_bias,
                                     mapping.tp_size,
                                     mapping.tp_rank,
                                     dim=0)
            elif chatglm_version in GLM_ARCH2_VERSIONS:
                mlp_fc_w = swap_and_split_mlp(mlp_fc_weight, mapping.tp_size,
                                              mapping.tp_rank)
                if mlp_fc_bias is None:
                    mlp_fc_b = None
                else:
                    mlp_fc_b = swap_and_split_mlp(mlp_fc_bias, mapping.tp_size,
                                                  mapping.tp_rank)
            weights.update(
                get_tllm_linear_weight(mlp_fc_w, f'{tllm_prex}.mlp.fc',
                                       mlp_fc_b, use_weight_only,
                                       plugin_weight_only_quant_type))

        # MLP Proj
        mlp_proj_weight, mlp_proj_bias = get_weight_and_bias(
            model_params, f'{prefix}.mlp.dense_4h_to_h', dtype)

        if use_smooth_quant:
            proj_act_range = act_range.get(f'{prefix}.mlp.dense_4h_to_h')
            proj_smoother = smoother.get(f'{prefix}.mlp.dense_4h_to_h')
            proj_vals_int8 = generate_int8(mlp_proj_weight.t(),
                                           proj_act_range,
                                           is_qkv=False,
                                           multi_query_mode=True)
            weights.update(
                get_tllm_linear_sq_weight(
                    vals=proj_vals_int8,
                    prefix=f'{tllm_prex}.mlp.proj.',
                    shape=[1, hidden_size],
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=f'{tllm_prex}.mlp.quantization_scaling_factor',
                    smoother_value=proj_smoother,
                    smoother_shape=[1, config.intermediate_size]))
            if mlp_proj_bias is not None:
                weights[f'{tllm_prex}.mlp.proj.bias'] = mlp_proj_bias
        else:
            mlp_proj_w = split(mlp_proj_weight,
                               mapping.tp_size,
                               mapping.tp_rank,
                               dim=1)
            weights.update(
                get_tllm_linear_weight(mlp_proj_w, f'{tllm_prex}.mlp.proj',
                                       mlp_proj_bias, use_weight_only,
                                       plugin_weight_only_quant_type))

        input_ln_weight, input_ln_bias = get_weight_and_bias(
            model_params, f'{prefix}.input_layernorm', dtype)
        weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_weight
        if input_ln_bias is not None:
            weights[f'{tllm_prex}.input_layernorm.bias'] = input_ln_bias

        post_ln_weight, post_ln_bias = get_weight_and_bias(
            model_params, f'{prefix}.post_attention_layernorm', dtype)
        weights[f'{tllm_prex}.post_layernorm.weight'] = post_ln_weight
        if post_ln_bias is not None:
            weights[f'{tllm_prex}.post_layernorm.bias'] = post_ln_bias

    if mapping.is_first_pp_rank():
        if chatglm_version == 'glm':
            embed_w = get_weight(model_params, 'word_embeddings', dtype)
            pos_embed_w = get_weight(model_params,
                                     'transformer.position_embeddings', dtype)
            weights['transformer.position_embedding.weight'] = split_embedding(
                pos_embed_w,
                tp_size=mapping.tp_size,
                tp_rank=mapping.tp_rank,
                use_parallel_embedding=use_parallel_embedding,
                sharding_dim=sharding_dim)
            block_embed_w = get_weight(model_params,
                                       'transformer.block_position_embeddings',
                                       dtype)
            weights['transformer.block_embedding.weight'] = split_embedding(
                block_embed_w,
                tp_size=mapping.tp_size,
                tp_rank=mapping.tp_rank,
                use_parallel_embedding=use_parallel_embedding,
                sharding_dim=sharding_dim)
        elif chatglm_version == 'chatglm':
            embed_w = get_weight(model_params, 'transformer.word_embeddings',
                                 dtype)
        elif chatglm_version in GLM_ARCH2_VERSIONS:
            embed_w = get_weight(model_params,
                                 'transformer.embedding.word_embeddings', dtype)

        weights['transformer.vocab_embedding.weight'] = split_embedding(
            embed_w,
            tp_size=mapping.tp_size,
            tp_rank=mapping.tp_rank,
            use_parallel_embedding=use_parallel_embedding,
            sharding_dim=sharding_dim)

    if mapping.is_last_pp_rank():
        if chatglm_version == 'glm':
            lm_head_weight = get_weight(model_params, 'word_embeddings',
                                        dtype).clone()
        elif chatglm_version == 'chatglm':
            lm_head_weight = get_weight(model_params,
                                        'transformer.word_embeddings',
                                        dtype).clone()
        elif chatglm_version in GLM_ARCH2_VERSIONS:
            lm_head_weight = get_weight(model_params,
                                        'transformer.output_layer', dtype)

        weights['lm_head.weight'] = split(lm_head_weight,
                                          mapping.tp_size,
                                          mapping.tp_rank,
                                          dim=0)

        if chatglm_version in GLM_ARCH1_VERSIONS:
            ln_f_w, ln_f_b = get_weight_and_bias(model_params,
                                                 'transformer.final_layernorm',
                                                 dtype)
        elif chatglm_version in GLM_ARCH2_VERSIONS:
            ln_f_w, ln_f_b = get_weight_and_bias(
                model_params, 'transformer.encoder.final_layernorm', dtype)
        weights['transformer.ln_f.weight'] = ln_f_w
        if ln_f_b is not None:
            weights['transformer.ln_f.bias'] = ln_f_b

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def quantize(hf_model_dir: str,
             output_dir: str,
             config: ChatGLMConfig,
             calib_dataset: str = 'cnn_dailymail',
             device: str = 'auto',
             trust_remote_code: bool = True):
    '''
        Quantize the save the model as TRT-LLM checkpoint to output_dir
    '''
    os.makedirs(output_dir, exist_ok=True)
    config.to_json_file(os.path.join(output_dir, 'config.json'))

    mapping = config.mapping
    assert mapping.rank == 0, "quantize should be called at rank 0 only"

    quant_config = config.quantization
    use_smooth_quant = quant_config._use_plugin_sq
    int8_kv_cache = quant_config.kv_cache_quant_algo == QuantAlgo.INT8

    assert use_smooth_quant or int8_kv_cache, "Call from_hugging_face when there is no quantization"
    assert hf_model_dir is not None
    ## only load and call smooth quant routine once for all ranks
    if config.chatglm_version == 'glm':
        device_map = 'cuda' if device != "cpu" else 'cpu'
    else:
        device_map = 'auto' if device != "cpu" else 'cpu'
    hf_model = AutoModel.from_pretrained(
        hf_model_dir,
        trust_remote_code=trust_remote_code,
        dtype='auto' if config.chatglm_version != 'glm' else getattr(
            torch, config.dtype),
        device_map=device_map)

    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
        "TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_dir,
        trust_remote_code=trust_remote_code,
    )
    dataset = load_calib_dataset(calib_dataset)

    act_range = capture_activation_range(hf_model,
                                         tokenizer,
                                         dataset,
                                         num_samples=64)
    smoother = {}
    if use_smooth_quant:
        smooth_chatglm_model(hf_model, act_range, quant_config.smoothquant_val,
                             smoother)

    for rank in range(mapping.world_size):
        # To avoid changing the mapping arg in-place, also the given mapping from caller is rank agnostic, since quantize is called from only one rank
        config = copy.deepcopy(config)
        config.set_rank(rank)
        weights = load_weights_from_hf_model(
            hf_model,
            config=config,
            act_range=act_range,
            smoother=smoother,
        )
        safetensors.torch.save_file(
            weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
        del weights
