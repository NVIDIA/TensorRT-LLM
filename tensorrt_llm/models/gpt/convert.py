# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import functools
import os
import shutil
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import safetensors
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoModelForVision2Seq,
                          AutoTokenizer)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.pytorch_utils import Conv1D

from ..._utils import pad_vocab_size, str_dtype_to_torch
from ...logger import logger
from ...quantization import QuantAlgo
from ..convert_utils import (generate_int8, get_weight, get_weight_and_bias,
                             load_calib_dataset,
                             retrieved_layer_index_from_name, smooth_gemm)
from .config import GPTConfig


def rename_keys(model_state, layer_rename_config: Dict[str, str]):
    if not layer_rename_config:
        return model_state

    new_state_dict = {}
    for key, value in model_state.items():
        for old, new in layer_rename_config.items():
            key = key.replace(old, new)
        assert key not in new_state_dict, f"Key already exists: {key}"
        new_state_dict[key] = value

    return new_state_dict


def get_needed_padding(value: int, multiple: int) -> int:
    return (multiple - value % multiple) % multiple


def pad_array_up_to(v: torch.Tensor, axis: int, multiple: int) -> torch.Tensor:
    a = [0 for i in range(len(v.shape) * 2)]
    a[axis * 2 - 1] = get_needed_padding(v.shape[axis], multiple)
    return torch.nn.functional.pad(v, a)


def split(param: torch.Tensor,
          tp_rank: int,
          tp_size: int,
          is_column: bool = True) -> torch.Tensor:
    """Split linear layer's weight, bias or scaling factors for tensor parallelism."""
    if param is None:
        return None
    assert param.ndim in [1, 2]
    if tp_size == 1:
        return param
    if param.numel() == 1:
        return param
    if param.ndim == 1 and not is_column:
        return param
    split_dim = 0 if (param.ndim == 1 or is_column) else 1
    return torch.chunk(param, tp_size, dim=split_dim)[tp_rank].contiguous()


def split_qkv(
    param: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
) -> torch.Tensor:
    """Split qkv layer's weight, bias or scaling factors for tensor parallelism.

    param: (num_heads*head_dim + 2*num_kv_heads*head_dim, in_dim)
    """
    if param is None:
        return None
    assert hidden_size % num_heads == 0
    head_dim = hidden_size // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    assert num_heads % num_kv_heads == 0
    assert num_heads % tp_size == 0

    q_param, k_param, v_param = torch.split(
        param, [hidden_size, num_kv_heads * head_dim, num_kv_heads * head_dim],
        dim=0)

    if num_kv_heads < tp_size:
        assert tp_size % num_kv_heads == 0
        num_dups = tp_size // num_kv_heads
        remain_shape = k_param.shape[1:]
        k_param = k_param.view(
            num_kv_heads, head_dim,
            *remain_shape).repeat_interleave(num_dups, dim=0).view(
                num_kv_heads * head_dim * num_dups, *remain_shape)
        v_param = v_param.view(
            num_kv_heads, head_dim,
            *remain_shape).repeat_interleave(num_dups, dim=0).view(
                num_kv_heads * head_dim * num_dups, *remain_shape)
    else:
        assert num_kv_heads % tp_size == 0

    q_param = split(q_param, tp_rank, tp_size, is_column=True)
    k_param = split(k_param, tp_rank, tp_size, is_column=True)
    v_param = split(v_param, tp_rank, tp_size, is_column=True)
    return torch.cat([q_param, k_param, v_param], dim=0)


def split_embedding(
    param: torch.Tensor,
    tp_rank: int,
    tp_size: int,
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
    return split(param, tp_rank, tp_size, is_column=(sharding_dim == 0))


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


@torch.no_grad()
def capture_activation_range(model,
                             tokenizer,
                             dataset,
                             num_samples=512,
                             seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    def stat_tensor(name, tensor, act_scales, key):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key],
                                              comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, act_scales, "x")
        stat_tensor(name, y, act_scales, "y")

        if act_scales[name]["w"] is None:
            act_scales[name]["w"] = m.weight.abs().clip(1e-8,
                                                        None).max(dim=0)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="calibrating model"):
        input_ids = tokenizer(dataset[i],
                              return_tensors="pt",
                              max_length=seq_len,
                              truncation=True).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def smooth_gpt_model(model, scales, alpha):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(module, GPT2Block):
            continue

        # qkv_proj
        layer_name = name + ".attn.c_attn"
        smoother = smooth_gemm(module.attn.c_attn.weight.T,
                               scales[layer_name]["x"], module.ln_1.weight,
                               module.ln_1.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_attn.weight.abs().max(dim=0)[0]

        # fc1
        layer_name = name + ".mlp.c_fc"
        smoother = smooth_gemm(module.mlp.c_fc.weight.T,
                               scales[layer_name]["x"], module.ln_2.weight,
                               module.ln_2.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.c_fc.weight.abs().max(dim=0)[0]


def get_tllm_linear_sq_weight(vals,
                              prefix,
                              shape,
                              tensor_parallel,
                              is_qkv=False,
                              per_token=False,
                              per_channel=False,
                              last_prefix=None,
                              bias=None,
                              smoother_value=None,
                              smoother_shape=None,
                              rank=0,
                              cat_dim=0,
                              multi_query_mode=False):
    results = {}

    def multi_query_split(data, local_dim, head_size, tp_size, cur_rank):
        q, k, v = torch.split(data, [local_dim, head_size, head_size], dim=-1)
        q_split = torch.split(q, q.shape[-1] // tp_size, dim=-1)
        k_split = torch.split(k, q.shape[-1] // tp_size, dim=-1)
        v_split = torch.split(v, q.shape[-1] // tp_size, dim=-1)
        return [
            torch.concat((q_split[ii], k_split[ii], v_split[ii]), dim=-1)
            for ii in range(tp_size)
        ][cur_rank]

    col_shape = shape if (is_qkv or per_channel) else [1, 1]

    if per_token:
        if per_channel:
            original_weights = vals["weight.int8.col"]
        else:
            original_weights = vals["weight.int8"]
        local_dim = original_weights.shape[0]
        head_size = (original_weights.shape[1] - local_dim) // 2

        if multi_query_mode:
            cur_weights = multi_query_split(original_weights, local_dim,
                                            head_size, tensor_parallel, rank)
        else:
            cur_weights = np.split(original_weights,
                                   tensor_parallel,
                                   axis=cat_dim)[rank]
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix + 'weight'] = cur_weights.t().contiguous()
        if smoother_value is None:
            results[last_prefix] = torch.from_numpy(
                np.array([1.0], dtype=np.float32))

        if per_channel:
            cur_per_channel_value = vals["scale_w_quant_orig.col"]
            if smoother_value is None:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_w_quant_orig.col"], local_dim, head_size,
                        tensor_parallel, rank)
                else:
                    cur_per_channel_value = np.split(
                        vals["scale_w_quant_orig.col"],
                        tensor_parallel,
                        axis=cat_dim)[rank]
        else:
            cur_per_channel_value = vals["scale_w_quant_orig"]
            if is_qkv:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_w_quant_orig"], local_dim, head_size,
                        tensor_parallel, rank)
                else:
                    cur_per_channel_value = np.split(vals["scale_w_quant_orig"],
                                                     tensor_parallel,
                                                     axis=cat_dim)[rank]

        results[prefix + 'per_channel_scale'] = cur_per_channel_value.reshape(
            col_shape).contiguous()
    else:
        if per_channel:
            original_weights = np.array(vals["weight.int8.col"])
        else:
            original_weights = np.array(vals["weight.int8"])
        local_dim = original_weights.shape[0]
        head_size = (original_weights.shape[1] - local_dim) // 2

        if multi_query_mode:
            cur_weights = multi_query_split(original_weights, local_dim,
                                            head_size, tensor_parallel, rank)
        else:
            cur_weights = np.split(original_weights,
                                   tensor_parallel,
                                   axis=cat_dim)[rank]
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix +
                'weight'] = torch.from_numpy(cur_weights).t().contiguous()

        if per_channel:
            cur_per_channel_value = vals["scale_y_accum_quant.col"]
            if smoother_value is None:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_y_accum_quant.col"], local_dim, head_size,
                        tensor_parallel, rank)
                else:
                    cur_per_channel_value = np.split(
                        vals["scale_y_accum_quant.col"],
                        tensor_parallel,
                        axis=cat_dim)[rank]
        else:
            cur_per_channel_value = vals["scale_y_accum_quant"]
            # QKV is always per_channel
            if is_qkv:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_y_accum_quant"], local_dim, head_size,
                        tensor_parallel, rank)
                else:
                    cur_per_channel_value = np.split(
                        vals["scale_y_accum_quant"],
                        tensor_parallel,
                        axis=cat_dim)[rank]

        results[prefix + 'per_channel_scale'] = cur_per_channel_value.reshape(
            col_shape).contiguous()

        results[last_prefix] = vals['scale_x_orig_quant'].contiguous()

        results[prefix + 'act_scale'] = vals["scale_y_quant_orig"].contiguous()

    if smoother_value is not None:
        cur_smoother_value = np.split(smoother_value,
                                      tensor_parallel,
                                      axis=cat_dim)[rank]
        results[prefix + 'smoother'] = cur_smoother_value.reshape(
            smoother_shape).contiguous().to(torch.float32)

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def load_weights_from_hf_model(hf_model,
                               config: GPTConfig,
                               act_range: Optional[dict] = None):
    quant_algo = config.quantization.quant_algo
    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    if quant_algo == QuantAlgo.W8A16:
        plugin_weight_only_quant_type = torch.int8
    elif quant_algo == QuantAlgo.W4A16:
        plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None

    use_smooth_quant = config.quantization._use_plugin_sq
    per_channel = use_smooth_quant and 'PER_CHANNEL' in quant_algo
    per_token = use_smooth_quant and 'PER_TOKEN' in quant_algo
    int8_kv_cache = config.quantization.kv_cache_quant_algo == QuantAlgo.INT8
    if use_smooth_quant or int8_kv_cache:
        assert act_range is not None

    weights = {}
    tik = time.time()

    hf_config = hf_model.config
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, config.dtype)
    gpt_variant = config.gpt_variant
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_kv_heads = config.num_key_value_heads
    num_hidden_layers = config.num_hidden_layers
    multi_query_mode = (num_kv_heads != num_attention_heads)

    mapping = config.mapping
    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        if gpt_variant in ['starcoder2', 'nemotron']:
            prefix = f'model.layers.{l}'
        elif gpt_variant == 'persimmon':
            is_fuyu = f'language_model.model.embed_tokens.weight' in model_params
            prefix = f'language_model.model.layers.{l}' if is_fuyu else f'model.layers.{l}'
        elif gpt_variant == 'kosmos-2':
            prefix = f'text_model.model.layers.{l}'
        else:
            prefix = f'transformer.h.{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'

        # (1) Attention QKV Linear
        if gpt_variant == 'santacoder':
            q_w, q_b = get_weight_and_bias(model_params,
                                           f'{prefix}.attn.q_attn', dtype)
            kv_w, kv_b = get_weight_and_bias(model_params,
                                             f'{prefix}.attn.kv_attn', dtype)
            qkv_w = torch.cat([q_w, kv_w], dim=-1)
            qkv_b = torch.cat([q_b, kv_b], dim=-1)
        elif gpt_variant in ['starcoder2', 'nemotron', 'kosmos-2']:
            q_w, q_b = get_weight_and_bias(model_params,
                                           f'{prefix}.self_attn.q_proj', dtype)
            k_w, k_b = get_weight_and_bias(model_params,
                                           f'{prefix}.self_attn.k_proj', dtype)
            v_w, v_b = get_weight_and_bias(model_params,
                                           f'{prefix}.self_attn.v_proj', dtype)
            qkv_w = torch.cat([q_w.cuda(), k_w.cuda(), v_w.cuda()], dim=0)
            qkv_b = torch.cat([q_b.cuda(), k_b.cuda(),
                               v_b.cuda()], dim=0) if q_b is not None else None
        elif gpt_variant == 'persimmon':
            qkv_w, qkv_b = get_weight_and_bias(
                model_params, f'{prefix}.self_attn.query_key_value', dtype)
        else:
            qkv_w, qkv_b = get_weight_and_bias(model_params,
                                               f'{prefix}.attn.c_attn', dtype)
        if gpt_variant in ['gpt2', 'santacoder', 'jais']:
            qkv_w = qkv_w.t().contiguous()  # transpose for Conv1D

        if use_smooth_quant:
            qkv_out_dim = qkv_w.shape[0]
            qkv_w_t = qkv_w.t()
            if not multi_query_mode:
                qkv_w_t = qkv_w_t.reshape(hidden_size, 3, hidden_size)
            int8_weights = generate_int8(qkv_w_t,
                                         act_range.get(f'{prefix}.attn.c_attn'),
                                         is_qkv=True,
                                         multi_query_mode=multi_query_mode)
            qkv_b = split_qkv(qkv_b, mapping.tp_rank, mapping.tp_size,
                              hidden_size, num_attention_heads, num_kv_heads)
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    f'{tllm_prex}.attention.qkv.',
                    [1, qkv_out_dim // mapping.tp_size],
                    mapping.tp_size,
                    is_qkv=True,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=f'{tllm_prex}.input_layernorm.scale_to_int',
                    bias=qkv_b,
                    smoother_value=None,
                    smoother_shape=None,
                    rank=mapping.tp_rank,
                    cat_dim=-1,
                    multi_query_mode=multi_query_mode))
        else:
            if gpt_variant == 'persimmon':
                qkv_w = split(qkv_w,
                              mapping.tp_rank,
                              mapping.tp_size,
                              is_column=True)

                qkv_b = split(qkv_b,
                              mapping.tp_rank,
                              mapping.tp_size,
                              is_column=True)
            else:
                qkv_w = split_qkv(qkv_w, mapping.tp_rank, mapping.tp_size,
                                  hidden_size, num_attention_heads,
                                  num_kv_heads)
                qkv_b = split_qkv(qkv_b, mapping.tp_rank, mapping.tp_size,
                                  hidden_size, num_attention_heads,
                                  num_kv_heads)
            weights.update(
                get_tllm_linear_weight(qkv_w, f'{tllm_prex}.attention.qkv',
                                       qkv_b, use_weight_only,
                                       plugin_weight_only_quant_type))

        if int8_kv_cache:
            qkv_w_t = qkv_w.t()
            if not multi_query_mode:
                qkv_w_t = qkv_w_t.reshape(hidden_size, 3, hidden_size)
            int8_weights = generate_int8(qkv_w_t,
                                         act_range.get(f'{prefix}.attn.c_attn'),
                                         is_qkv=True,
                                         multi_query_mode=multi_query_mode)
            weights[
                f'{tllm_prex}.attention.kv_cache_scaling_factor'] = int8_weights[
                    'scale_y_quant_orig'].contiguous()

        # (2) Attention Dense Linear
        if gpt_variant in ['starcoder2', 'nemotron']:
            attn_dense_w, attn_dense_b = get_weight_and_bias(
                model_params, f'{prefix}.self_attn.o_proj', dtype)
        elif gpt_variant == 'persimmon':
            attn_dense_w, attn_dense_b = get_weight_and_bias(
                model_params, f'{prefix}.self_attn.dense', dtype)
        elif gpt_variant == 'kosmos-2':
            attn_dense_w, attn_dense_b = get_weight_and_bias(
                model_params, f'{prefix}.self_attn.out_proj', dtype)
        else:
            attn_dense_w, attn_dense_b = get_weight_and_bias(
                model_params, f'{prefix}.attn.c_proj', dtype)
        if gpt_variant in ['gpt2', 'santacoder', 'jais']:
            attn_dense_w = attn_dense_w.t().contiguous()  # transpose for Conv1D

        if use_smooth_quant:
            attn_dense_w_t = attn_dense_w.t()
            int8_weights = generate_int8(attn_dense_w_t,
                                         act_range.get(f'{prefix}.attn.c_proj'))
            # change it to the real smoother if dense layer is applied smooth quant
            fake_smoother_value = torch.ones([1, hidden_size],
                                             dtype=torch.float32)
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    f'{tllm_prex}.attention.dense.', [1, hidden_size],
                    mapping.tp_size,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=
                    f'{tllm_prex}.attention.quantization_scaling_factor',
                    bias=attn_dense_b,
                    smoother_value=fake_smoother_value,
                    smoother_shape=[1, hidden_size // mapping.tp_size],
                    rank=mapping.tp_rank,
                    cat_dim=0))
        else:
            attn_dense_w = split(attn_dense_w,
                                 mapping.tp_rank,
                                 mapping.tp_size,
                                 is_column=False)
            weights.update(
                get_tllm_linear_weight(attn_dense_w,
                                       f'{tllm_prex}.attention.dense',
                                       attn_dense_b, use_weight_only,
                                       plugin_weight_only_quant_type))

        # (3) MLP FC Linear
        if gpt_variant == 'persimmon':
            suffix = "mlp.dense_h_to_4h"
        elif gpt_variant == 'kosmos-2':
            suffix = "ffn.fc1"
        elif gpt_variant == 'nemotron':
            suffix = "mlp.up_proj"
        else:
            suffix = "mlp.c_fc"
        mlp_fc_w, mlp_fc_b = get_weight_and_bias(model_params,
                                                 f'{prefix}.{suffix}', dtype)
        if gpt_variant in ['gpt2', 'santacoder', 'jais']:
            mlp_fc_w = mlp_fc_w.t().contiguous()  # transpose for Conv1D
        if gpt_variant in ['jais']:
            mlp_fc_w = pad_array_up_to(mlp_fc_w, 0, mapping.tp_size)
            mlp_fc_b = pad_array_up_to(mlp_fc_b, 0, mapping.tp_size)
        if use_smooth_quant:
            mlp_fc_w_t = mlp_fc_w.t()
            int8_weights = generate_int8(mlp_fc_w_t,
                                         act_range.get(f'{prefix}.mlp.c_fc'))
            mlp_fc_b = split(mlp_fc_b,
                             mapping.tp_rank,
                             mapping.tp_size,
                             is_column=True)
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    f'{tllm_prex}.mlp.fc.',
                    [1, 4 * hidden_size // mapping.tp_size],
                    mapping.tp_size,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=f'{tllm_prex}.post_layernorm.scale_to_int',
                    bias=mlp_fc_b,
                    smoother_value=None,
                    smoother_shape=None,
                    rank=mapping.tp_rank,
                    cat_dim=-1))
        else:
            mlp_fc_w = split(mlp_fc_w,
                             mapping.tp_rank,
                             mapping.tp_size,
                             is_column=True)
            mlp_fc_b = split(mlp_fc_b,
                             mapping.tp_rank,
                             mapping.tp_size,
                             is_column=True)
            if gpt_variant in ['jais']:
                weights.update(
                    get_tllm_linear_weight(mlp_fc_w, f'{tllm_prex}.mlp.gate',
                                           mlp_fc_b, use_weight_only,
                                           plugin_weight_only_quant_type))
            else:
                weights.update(
                    get_tllm_linear_weight(mlp_fc_w, f'{tllm_prex}.mlp.fc',
                                           mlp_fc_b, use_weight_only,
                                           plugin_weight_only_quant_type))
            if gpt_variant in ['jais']:
                mlp_fc2_w, mlp_fc2_b = get_weight_and_bias(
                    model_params, f'{prefix}.mlp.c_fc2', dtype)
                mlp_fc2_w = mlp_fc2_w.t().contiguous()
                mlp_fc2_w = pad_array_up_to(mlp_fc2_w, 0, mapping.tp_size)
                mlp_fc2_b = pad_array_up_to(mlp_fc2_b, 0, mapping.tp_size)
                mlp_fc2_w = split(mlp_fc2_w,
                                  mapping.tp_rank,
                                  mapping.tp_size,
                                  is_column=True)
                mlp_fc2_b = split(mlp_fc2_b,
                                  mapping.tp_rank,
                                  mapping.tp_size,
                                  is_column=True)
                weights.update(
                    get_tllm_linear_weight(mlp_fc2_w, f'{tllm_prex}.mlp.fc',
                                           mlp_fc2_b, use_weight_only,
                                           plugin_weight_only_quant_type))

        # (4) MLP Proj Layer
        if gpt_variant == 'persimmon':
            suffix = "mlp.dense_4h_to_h"
        elif gpt_variant == 'kosmos-2':
            suffix = "ffn.fc2"
        elif gpt_variant == 'nemotron':
            suffix = "mlp.down_proj"
        else:
            suffix = "mlp.c_proj"
        mlp_proj_w, mlp_proj_b = get_weight_and_bias(model_params,
                                                     f'{prefix}.{suffix}',
                                                     dtype)
        if gpt_variant in ['gpt2', 'santacoder', 'jais']:
            mlp_proj_w = mlp_proj_w.t().contiguous()  # transpose for Conv1D
        if gpt_variant in ['jais']:
            mlp_proj_w = pad_array_up_to(mlp_proj_w, 1, mapping.tp_size)
        if use_smooth_quant:
            mlp_proj_w_t = mlp_proj_w.t()
            int8_weights = generate_int8(mlp_proj_w_t,
                                         act_range.get(f'{prefix}.mlp.c_proj'))
            # change it to the real smoother if proj layer is applied smooth quant
            fake_smoother_value = torch.ones([1, 4 * hidden_size],
                                             dtype=torch.float32)
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    f'{tllm_prex}.mlp.proj.', [1, hidden_size],
                    mapping.tp_size,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=f'{tllm_prex}.mlp.quantization_scaling_factor',
                    bias=mlp_proj_b,
                    smoother_value=fake_smoother_value,
                    smoother_shape=[1, 4 * hidden_size // mapping.tp_size],
                    rank=mapping.tp_rank,
                    cat_dim=0))
        else:
            mlp_proj_w = split(mlp_proj_w,
                               mapping.tp_rank,
                               mapping.tp_size,
                               is_column=False)
            weights.update(
                get_tllm_linear_weight(mlp_proj_w, f'{tllm_prex}.mlp.proj',
                                       mlp_proj_b, use_weight_only,
                                       plugin_weight_only_quant_type))

        # (5) Input layernorm
        apply_layernorm_1p = gpt_variant == 'nemotron'
        if gpt_variant in ['starcoder2', 'nemotron', 'persimmon']:
            input_ln_w, input_ln_b = get_weight_and_bias(
                model_params, f'{prefix}.input_layernorm', dtype)
        elif gpt_variant == 'kosmos-2':
            input_ln_w, input_ln_b = get_weight_and_bias(
                model_params, f'{prefix}.self_attn_layer_norm', dtype)
        else:
            input_ln_w, input_ln_b = get_weight_and_bias(
                model_params, f'{prefix}.ln_1', dtype)
        if apply_layernorm_1p:
            input_ln_w += 1.0
        weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_w
        if input_ln_b is not None:
            weights[f'{tllm_prex}.input_layernorm.bias'] = input_ln_b

        # (6) Post layernorm
        if gpt_variant in ['starcoder2', 'nemotron', 'persimmon']:
            post_ln_w, post_ln_b = get_weight_and_bias(
                model_params, f'{prefix}.post_attention_layernorm', dtype)
        elif gpt_variant == 'kosmos-2':
            post_ln_w, post_ln_b = get_weight_and_bias(
                model_params, f'{prefix}.final_layer_norm', dtype)
        else:
            post_ln_w, post_ln_b = get_weight_and_bias(model_params,
                                                       f'{prefix}.ln_2', dtype)
        if apply_layernorm_1p:
            post_ln_w += 1.0
        weights[f'{tllm_prex}.post_layernorm.weight'] = post_ln_w
        if post_ln_b is not None:
            weights[f'{tllm_prex}.post_layernorm.bias'] = post_ln_b

        if gpt_variant == 'persimmon':
            q_layernorm_w, q_layernorm_b = get_weight_and_bias(
                model_params, f'{prefix}.self_attn.q_layernorm', dtype)

            weights[f'{tllm_prex}.attention.q_layernorm.weight'] = q_layernorm_w
            weights[f'{tllm_prex}.attention.q_layernorm.bias'] = q_layernorm_b

            k_layernorm_w, k_layernorm_b = get_weight_and_bias(
                model_params, f'{prefix}.self_attn.k_layernorm', dtype)

            weights[f'{tllm_prex}.attention.k_layernorm.weight'] = k_layernorm_w
            weights[f'{tllm_prex}.attention.k_layernorm.bias'] = k_layernorm_b

        if gpt_variant == 'kosmos-2':
            q_layernorm_w, q_layernorm_b = get_weight_and_bias(
                model_params, f'{prefix}.self_attn.inner_attn_ln', dtype)

            weights[
                f'{tllm_prex}.attention.inner_layernorm.weight'] = q_layernorm_w
            weights[
                f'{tllm_prex}.attention.inner_layernorm.bias'] = q_layernorm_b

            k_layernorm_w, k_layernorm_b = get_weight_and_bias(
                model_params, f'{prefix}.ffn.ffn_layernorm', dtype)

            weights[f'{tllm_prex}.mlp.inner_layernorm.weight'] = k_layernorm_w
            weights[f'{tllm_prex}.mlp.inner_layernorm.bias'] = k_layernorm_b

    if mapping.is_first_pp_rank():
        if gpt_variant in ['starcoder2', 'nemotron']:
            embed_w = get_weight(model_params, 'model.embed_tokens', dtype)
        elif gpt_variant == 'kosmos-2':
            embed_w = get_weight(model_params, 'text_model.model.embed_tokens',
                                 dtype)
        elif gpt_variant == 'persimmon':
            embed_w = get_weight(model_params,
                                 ('language_model.' if is_fuyu else '') +
                                 'model.embed_tokens', dtype)
        else:
            embed_w = get_weight(model_params, 'transformer.wte', dtype)
        weights['transformer.vocab_embedding.weight'] = split_embedding(
            embed_w,
            mapping.tp_rank,
            mapping.tp_size,
            use_parallel_embedding=config.use_parallel_embedding,
            sharding_dim=config.embedding_sharding_dim)

        if gpt_variant == 'kosmos-2':
            padding_idx = hf_config.text_config.pad_token_id
            sin_pos_embedding = hf_model.text_model.model.embed_positions.get_embedding(
                padding_idx + 1 + hf_config.text_config.max_position_embeddings,
                hf_config.text_config.embed_dim,
                padding_idx=padding_idx)  # [2 + num_embeddings, embed_dim]
            pos_embed_w = sin_pos_embedding[2:].to(dtype).detach().cpu()
        else:
            pos_embed_w = get_weight(model_params, 'transformer.wpe', dtype)
        if pos_embed_w is not None:
            weights['transformer.position_embedding.weight'] = split_embedding(
                pos_embed_w,
                mapping.tp_rank,
                mapping.tp_size,
                use_parallel_embedding=config.use_parallel_embedding,
                sharding_dim=config.embedding_sharding_dim)

    if mapping.is_last_pp_rank():
        if gpt_variant in ['starcoder2', 'nemotron']:
            embed_w = get_weight(model_params, 'lm_head', dtype)
            if embed_w is None:
                embed_w = get_weight(model_params, 'model.embed_tokens', dtype)
        elif gpt_variant == 'persimmon':
            embed_w = get_weight(model_params,
                                 ('language_model.' if is_fuyu else '') +
                                 'lm_head', dtype)
        elif gpt_variant == 'kosmos-2':
            embed_w = get_weight(model_params, 'text_model.model.embed_tokens',
                                 dtype)
        else:
            embed_w = get_weight(model_params, 'transformer.wte', dtype)

        if vocab_size % mapping.tp_size != 0:
            vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
            pad_width = vocab_size_padded - vocab_size
            embed_w = torch.nn.functional.pad(embed_w, (0, 0, 0, pad_width),
                                              value=0)
        if hasattr(hf_config, 'logits_scale'):
            embed_w *= hf_config.logits_scale
        weights['lm_head.weight'] = split(embed_w.clone(),
                                          mapping.tp_rank,
                                          mapping.tp_size,
                                          is_column=True)

        if gpt_variant in ['starcoder2', 'nemotron']:
            ln_f_w, ln_f_b = get_weight_and_bias(model_params, 'model.norm',
                                                 dtype)
        elif gpt_variant == 'persimmon':
            ln_f_w, ln_f_b = get_weight_and_bias(
                model_params, ('language_model.' if is_fuyu else '') +
                'model.final_layernorm', dtype)
        elif gpt_variant == 'kosmos-2':
            ln_f_w, ln_f_b = get_weight_and_bias(model_params,
                                                 'text_model.model.layer_norm',
                                                 dtype)
        else:
            ln_f_w, ln_f_b = get_weight_and_bias(model_params,
                                                 'transformer.ln_f', dtype)
        if apply_layernorm_1p:
            ln_f_w += 1.0
        weights['transformer.ln_f.weight'] = ln_f_w
        if ln_f_b is not None:
            weights['transformer.ln_f.bias'] = ln_f_b

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def quantize(hf_model_dir: str,
             output_dir: str,
             config: GPTConfig,
             device: str = 'cuda',
             calib_dataset: str = 'cnn_dailymail',
             trust_remote_code: bool = True):
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
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        device_map='auto' if device != 'cpu' else 'cpu',
        dtype='auto' if not use_smooth_quant else torch.float16,
        trust_remote_code=trust_remote_code)

    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
        "TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_dir,
        trust_remote_code=trust_remote_code,
        use_fast=False,
        padding_side='left')

    dataset = load_calib_dataset(calib_dataset)
    act_range = capture_activation_range(hf_model, tokenizer, dataset)
    if use_smooth_quant:
        smooth_gpt_model(hf_model, act_range, quant_config.smoothquant_val)

    for rank in range(mapping.world_size):
        # To avoid changing the mapping arg in-place, also the given mapping from caller is rank agnostic, since quantize is called from only one rank
        config = copy.deepcopy(config)
        config.set_rank(rank)
        weights = load_weights_from_hf_model(
            hf_model,
            config=config,
            act_range=act_range,
        )
        safetensors.torch.save_file(
            weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
        del weights


def load_hf_gpt(model_dir: str, load_model_on_cpu: bool = False):
    if 'kosmos-2' in model_dir:
        hf_model = AutoModelForVision2Seq.from_pretrained(
            model_dir, trust_remote_code=True)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map='auto' if not load_model_on_cpu else 'cpu',
            dtype='auto',
            trust_remote_code=True,
        )
    return hf_model


def cpu_map_location(storage, loc):
    return storage.cpu()


def gpu_map_location(storage, loc):
    if loc.startswith("cuda"):
        training_gpu_idx = int(loc.split(":")[1])
        inference_gpu_idx = training_gpu_idx % torch.cuda.device_count()
        return storage.cuda(inference_gpu_idx)
    elif loc.startswith("cpu"):
        return storage.cpu()
    else:
        raise ValueError(f"Not handled {loc}")


def copy_tokenizer_files(config, out_dir):
    basenames = {
        "model": "tokenizer",
        "vocab_file": "vocab",
        "merge_file": "merges",
    }

    for key in basenames.keys():
        if config[key] is None:
            continue
        path = Path(config[key])
        if not path.exists():
            logger.debug(f"Tokenizer {key}: {path} file not found")
            continue

        dst_path = out_dir / f"{basenames[key]}{path.suffix}"
        logger.debug(f"Copy tokenizer {key}: {path}->{dst_path}")
        shutil.copy(path.as_posix(), dst_path.as_posix())


def update_tokenizer_paths(tokenizer_config: Dict,
                           tokenizer_file_paths: Dict[str, Optional[str]]):
    for key, new_path in tokenizer_file_paths.items():
        old_path = tokenizer_config[key]
        if old_path is None:
            continue
        old_path = Path(old_path)
        if new_path:
            logger.debug(f"Update tokenizer {key} {old_path} -> {new_path}")
            tokenizer_config[key] = new_path.as_posix()
        elif not old_path.exists():
            logger.warning(
                f"Tokenizer {key}'s path {old_path} does not exists: set it to None"
            )
            tokenizer_config[key] = None
    return tokenizer_config


def unpack_nemo_ckpt(nemo_archive_path: Union[str, Path],
                     out_dir_path: Union[str, Path]):
    nemo_archive_path = Path(nemo_archive_path)
    if not nemo_archive_path.exists():
        raise FileNotFoundError(f"{nemo_archive_path} does not exist")

    for tar_mode in ["r:", "r:gz"]:
        try:
            with tarfile.open(nemo_archive_path, mode=tar_mode) as tar_file:

                def is_within_directory(directory, target):

                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)

                    prefix = os.path.commonprefix([abs_directory, abs_target])

                    return prefix == abs_directory

                def safe_members(tar_file):
                    members = []
                    for member in tar_file.getmembers():
                        member_path = os.path.join(out_dir_path, member.name)
                        if not is_within_directory(out_dir_path, member_path):
                            raise Exception(
                                "Attempted Path Traversal in Tar File")
                        members.append(member)
                    return members

                for member in safe_members(tar_file):
                    tar_file.extract(member,
                                     path=out_dir_path,
                                     numeric_owner=False,
                                     filter=tarfile.data_filter)

            return out_dir_path
        except tarfile.ReadError:
            pass

    raise RuntimeError(f"Could not unpack {nemo_archive_path}")


def extract_layers_with_prefix(model_, prefix):
    length_to_trim = len(prefix)
    model_state = model_.get("state_dict", model_)
    return {
        key[length_to_trim:]: model_state[key]
        for key in model_state.keys() if prefix in key
    }


class UnpackedNemoCheckpointDir:

    def __init__(self,
                 checkpoints_dir: Union[str, Path],
                 load_checkpoints_to_cpu: bool = False):
        self._checkpoints_dir = Path(checkpoints_dir)
        self._load_checkpoints_to_cpu = load_checkpoints_to_cpu

    @property
    @functools.lru_cache
    def model_config(self):
        model_config = None

        model_config_filename = "model_config.yaml"
        model_configs_paths = list(
            self._checkpoints_dir.rglob(model_config_filename))
        if model_configs_paths:
            if len(model_configs_paths) > 1:
                raise RuntimeError(
                    f"There are more than single {model_config_filename} "
                    f"in {self._checkpoints_dir}: {', '.join(map(lambda p: p.as_posix(), model_configs_paths))}"
                )
            model_config_path = model_configs_paths[0]
            logger.debug(f"Loading model config from {model_config_path}")
            with model_config_path.open("r") as model_config_file:
                model_config = yaml.load(model_config_file,
                                         Loader=yaml.SafeLoader)
        else:
            logger.debug("Searching model config in checkpoints")
            # try to obtain from checkpoint
            checkpoint_name = self.checkpoint_name
            checkpoints_paths = sorted(
                self._checkpoints_dir.rglob(checkpoint_name))
            if checkpoints_paths:
                # assume that parallel ranks 0 checkpoint should have model config embedded
                checkpoint_path = checkpoints_paths[0]

                map_location_fn = cpu_map_location if self._load_checkpoints_to_cpu else gpu_map_location

                model_00 = torch.load(checkpoint_path,
                                      map_location=map_location_fn)
                if "hyper_parameters" in model_00 and "cfg" in model_00[
                        "hyper_parameters"]:
                    model_config = model_00["hyper_parameters"]["cfg"]
                    logger.debug(
                        f"Loaded model config from checkpoint {checkpoint_path}"
                    )
                else:
                    logger.debug(
                        f"Could not find model config in checkpoint {checkpoint_path}"
                    )
                del model_00

        if model_config is None:
            logger.warning(
                f"Could not find checkpoint with NeMo model config in {self._checkpoints_dir}"
            )

        logger.debug(f"Loaded model config {model_config}")

        return model_config

    @property
    def checkpoints_dir(self):
        return self._checkpoints_dir

    def get_checkpoints_paths(self,
                              tensor_model_parallel_size=1,
                              pipeline_model_parallel_size=1):
        """
        Injects tensor/pipeline model parallel ranks into the filepath.
        Does nothing if not using model parallelism.
        """

        checkpoint_path_without_rank = self.checkpoints_dir / self.checkpoint_name

        def _inject_parallel_ranks(tp_rank, pp_rank):
            if tensor_model_parallel_size > 1 or pipeline_model_parallel_size > 1:
                if pipeline_model_parallel_size is None or pipeline_model_parallel_size == 1:
                    checkpoint_path = (checkpoint_path_without_rank.parent /
                                       f"mp_rank_{tp_rank:02d}" /
                                       checkpoint_path_without_rank.name)
                else:
                    checkpoint_path = (
                        checkpoint_path_without_rank.parent /
                        f"tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}" /
                        checkpoint_path_without_rank.name)
                return checkpoint_path
            else:
                return checkpoint_path_without_rank

        return [[
            _inject_parallel_ranks(tp_rank=tp_rank, pp_rank=pp_rank)
            for pp_rank in range(pipeline_model_parallel_size)
        ] for tp_rank in range(tensor_model_parallel_size)]

    @property
    @functools.lru_cache
    def checkpoint_name(self):
        patterns = [
            "model_weights.ckpt",  # older megatron checkpoints
            "*last.ckpt",  # newer format of checkpoints
        ]
        for pattern in patterns:
            model_files = sorted(list(self._checkpoints_dir.rglob(pattern)))
            if model_files:
                return model_files[0].name

        raise ValueError(
            f"Could not find checkpoint files in {self._checkpoints_dir}")

    @functools.lru_cache
    def get_tokenizer_file_path(self, tokenizer_key, file_key,
                                default_filename_pattern):
        model_config = self.model_config
        file_property = None
        if tokenizer_key in model_config and file_key in model_config[
                tokenizer_key]:
            file_property = model_config[tokenizer_key][file_key]
        elif file_key in model_config:
            file_property = model_config[file_key]

        logger.debug(
            f"model_config[{tokenizer_key}][{file_key}]={file_property}")

        if file_property and file_property.startswith("nemo:"):
            filename = file_property.split("nemo:")[1]
            filename_pattern = f"*{filename}"
        elif file_property and file_property.startswith("/artifacts/"):
            filename = Path(file_property).name
            filename_pattern = f"*{filename}"
        elif file_property is None or file_property == "None":
            filename_pattern = None
        else:
            filename_pattern = default_filename_pattern
            logger.warning(
                f"Tokenizer file from config: {tokenizer_key}.{file_key}={file_property} "
                f"looks like unsupported path. Pattern {filename_pattern} will be used."
            )

        file_path = None
        if filename_pattern is not None:
            files_paths = list(self._checkpoints_dir.glob(filename_pattern))
            if files_paths:
                assert len(files_paths) == 1
                file_path = files_paths[0]

        return file_path

    @functools.lru_cache
    def get_all_tokenizer_file_paths(self):
        return {
            "model":
            self.get_tokenizer_file_path("tokenizer", "model", "*.model"),
            "vocab_file":
            self.get_tokenizer_file_path("tokenizer", "vocab_file", "*vocab*"),
            "merge_file":
            self.get_tokenizer_file_path("tokenizer", "merge_file",
                                         "*merge*.txt"),
        }


@torch.no_grad()
def load_torch_checkpoints(checkpoints_paths,
                           merge_factor,
                           tp_rank,
                           pp_rank,
                           map_location_fn,
                           handle_model_level_weights,
                           layer_rename_config: Dict[str, str] = {}):
    models = []
    for k in range(merge_factor):
        rank_weights = checkpoints_paths[tp_rank * merge_factor + k][pp_rank]
        model = torch.load(rank_weights, map_location=map_location_fn)
        model = rename_keys(model, layer_rename_config)
        handle_model_level_weights(model, tp_rank * merge_factor + k, pp_rank)
        layers = extract_layers_with_prefix(model,
                                            "model.language_model.encoder.")
        models.append(layers)
    return models


@torch.no_grad()
def load_weights_from_nemo(nemo_ckpt_dir: str, config: GPTConfig, **kwargs):
    assert config.mapping.pp_size == 1, \
        "Pipeline parallelism is not supported."
    assert not config.quantization.quant_mode.has_any_quant(), \
        "Quantization is not supported."

    load_model_on_cpu = kwargs.pop('load_model_on_cpu', False)
    nemo_rename_key = kwargs.pop('nemo_rename_key', [])
    layer_rename_config = {
        pattern.split(':')[0]: pattern.split(':')[1]
        for pattern in nemo_rename_key
    }

    unpacked_checkpoints_dir = UnpackedNemoCheckpointDir(
        nemo_ckpt_dir, load_checkpoints_to_cpu=load_model_on_cpu)
    nemo_model_config = unpacked_checkpoints_dir.model_config

    checkpoints_paths = unpacked_checkpoints_dir.get_checkpoints_paths(
        nemo_model_config.get("tensor_model_parallel_size", 1),
        nemo_model_config.get("pipeline_model_parallel_size", 1),
    )

    if unpacked_checkpoints_dir._load_checkpoints_to_cpu:
        map_location_fn = cpu_map_location
    else:
        map_location_fn = gpu_map_location
    dtype = str_dtype_to_torch(config.dtype)

    # load position_embedding from rank 0
    model_00 = torch.load(checkpoints_paths[0][0], map_location=map_location_fn)
    model_00 = model_00.get("state_dict", model_00)
    model_00 = rename_keys(model_00, layer_rename_config)
    has_position_embedding = "model.language_model.embedding.position_embeddings.weight" in model_00
    has_lm_head = "model.language_model.output_layer.weight" in model_00
    del model_00

    num_layers = nemo_model_config["num_layers"]
    training_tp_size = nemo_model_config.get("tensor_model_parallel_size", 1)
    training_pp_size = nemo_model_config.get("pipeline_model_parallel_size", 1)
    inference_tp_size = config.mapping.tp_size
    inference_tp_rank = config.mapping.tp_rank

    apply_layernorm_1p = (nemo_model_config.get('normalization',
                                                '') == "layernorm1p")
    split_gated_activation = ("swiglu"
                              in nemo_model_config.get('activation', "gelu"))
    num_attention_heads = nemo_model_config["num_attention_heads"]
    # use_attention_nemo_shape = True
    transpose_weights = True
    # multi_query_mode = False
    local_dim = None

    # merge_factor: how many TP training nodes are merged into an inference TP node
    # split_factor: in how many parts a TP training node is split
    gcd = np.gcd(training_tp_size, inference_tp_size)
    merge_factor = training_tp_size // gcd
    split_factor = inference_tp_size // gcd

    model_level_weights = defaultdict(list)

    def handle_model_level_weights(model, tp_idx: int, pp_idx: int):
        if tp_idx == 0 and pp_idx == 0:
            if has_position_embedding:
                val = model[
                    "model.language_model.embedding.position_embeddings.weight"].detach(
                    ).cpu()
                model_level_weights[
                    "transformer.position_embedding.weight"].append(val)
        if pp_idx == 0:
            val = model.get(
                "state_dict", model
            )["model.language_model.embedding.word_embeddings.weight"].detach(
            ).cpu()
            model_level_weights["transformer.vocab_embedding.weight"].append(
                val)
        if has_lm_head and pp_idx == training_pp_size - 1:
            val = model.get(
                "state_dict",
                model)["model.language_model.output_layer.weight"].detach().cpu(
                )
            model_level_weights["lm_head.weight"].append(val)

    weights = {}
    tik = time.time()
    tp_rank = inference_tp_rank // split_factor
    # for tp_rank in range(training_tp_size // merge_factor):
    for pp_rank in range(training_pp_size):
        models = load_torch_checkpoints(checkpoints_paths, merge_factor,
                                        tp_rank, pp_rank, map_location_fn,
                                        handle_model_level_weights,
                                        layer_rename_config)
        for name in list(models[0].keys()):
            params = [model[name].detach().cpu() for model in models]
            if transpose_weights and params[0].ndim == 2:
                params = [p.T for p in params]
            if "layernorm.weight" in name and apply_layernorm_1p:
                params = [p + 1.0 for p in params]

            l = retrieved_layer_index_from_name(name)
            if l is not None:
                new_l = l + pp_rank * num_layers // training_pp_size
                prefix = f'transformer.layers.{new_l}'

                if 'attention.query_key_value' in name:
                    if name.endswith('weight'):
                        hidden_dim = params[0].shape[0]
                        if local_dim is None:
                            local_dim = params[0].shape[-1] // 3

                        # multi_query_mode = False; use_attention_nemo_shape = True
                        head_num = num_attention_heads // training_tp_size
                        size_per_head = hidden_dim // num_attention_heads
                        params = [
                            param.reshape(hidden_dim, head_num, 3,
                                          size_per_head) for param in params
                        ]
                        params = [param.permute(0, 2, 1, 3) for param in params]
                        params = [
                            param.reshape(hidden_dim, 3, local_dim)
                            for param in params
                        ]
                        cat_dim = -1
                        param = torch.concat(params, dim=cat_dim)
                        param = torch.chunk(param, split_factor,
                                            dim=cat_dim)[inference_tp_rank %
                                                         split_factor]
                        weights[
                            f'{prefix}.attention.qkv.weight'] = param.reshape(
                                hidden_dim, -1).t()
                    else:
                        if local_dim is None:
                            local_dim = params[0].shape[-1] // 3

                        # multi_query_mode = False; use_attention_nemo_shape = True
                        head_num = num_attention_heads // training_tp_size
                        size_per_head = local_dim // head_num
                        params = [
                            param.reshape(head_num, 3, size_per_head)
                            for param in params
                        ]
                        params = [param.permute(1, 0, 2) for param in params]
                        params = [
                            param.reshape(3, local_dim) for param in params
                        ]
                        cat_dim = -1
                        param = torch.concat(params, dim=cat_dim)
                        param = torch.chunk(param, split_factor,
                                            dim=cat_dim)[inference_tp_rank %
                                                         split_factor]
                        weights[f'{prefix}.attention.qkv.bias'] = param.reshape(
                            -1)

                elif 'attention.dense' in name:
                    if name.endswith('weight'):
                        cat_dim = 0
                        param = torch.concat(params, dim=cat_dim)
                        param = torch.chunk(param, split_factor,
                                            dim=cat_dim)[inference_tp_rank %
                                                         split_factor]
                        weights[f'{prefix}.attention.dense.weight'] = param.t()
                    else:
                        weights[f'{prefix}.attention.dense.bias'] = params[0]

                elif 'mlp.dense_h_to_4h' in name:
                    if name.endswith('weight'):
                        if split_gated_activation:
                            params = [torch.chunk(p, 2, dim=-1) for p in params]
                            params, gate_params = list(zip(*params))
                        cat_dim = -1
                        param = torch.concat(params, dim=cat_dim)
                        param = torch.chunk(param, split_factor,
                                            dim=cat_dim)[inference_tp_rank %
                                                         split_factor]
                        weights[f'{prefix}.mlp.fc.weight'] = param.t()
                        if split_gated_activation:
                            gate_param = torch.concat(gate_params, dim=cat_dim)
                            gate_param = torch.chunk(
                                gate_param, split_factor,
                                dim=cat_dim)[inference_tp_rank % split_factor]
                            weights[f'{prefix}.mlp.gate.weight'] = gate_param.t(
                            )
                    else:
                        if split_gated_activation:
                            params = [torch.chunk(p, 2, dim=-1) for p in params]
                            params, gate_params = list(zip(*params))
                        cat_dim = -1
                        param = torch.concat(params, dim=cat_dim)
                        param = torch.chunk(param, split_factor,
                                            dim=cat_dim)[inference_tp_rank %
                                                         split_factor]
                        weights[f'{prefix}.mlp.fc.bias'] = param
                        if split_gated_activation:
                            gate_param = torch.concat(gate_params, dim=cat_dim)
                            gate_param = torch.chunk(
                                gate_param, split_factor,
                                dim=cat_dim)[inference_tp_rank % split_factor]
                            weights[f'{prefix}.mlp.gate.bias'] = gate_param

                elif 'mlp.dense_4h_to_h' in name:
                    if name.endswith('weight'):
                        cat_dim = 0
                        param = torch.concat(params, dim=cat_dim)
                        param = torch.chunk(param, split_factor,
                                            dim=cat_dim)[inference_tp_rank %
                                                         split_factor]
                        weights[f'{prefix}.mlp.proj.weight'] = param.t()
                    else:
                        weights[f'{prefix}.mlp.proj.bias'] = params[0]

                elif 'input_layernorm' in name:
                    if name.endswith('weight'):
                        weights[f'{prefix}.input_layernorm.weight'] = params[0]
                    else:
                        weights[f'{prefix}.input_layernorm.bias'] = params[0]
                elif 'post_attention_layernorm' in name:
                    if name.endswith('weight'):
                        weights[f'{prefix}.post_layernorm.weight'] = params[0]
                    else:
                        weights[f'{prefix}.post_layernorm.bias'] = params[0]

            elif 'final_layernorm' in name:
                if name.endswith('weight'):
                    weights['transformer.ln_f.weight'] = params[0]
                else:
                    weights['transformer.ln_f.bias'] = params[0]
            for model in models:
                del model[name]
        del models

    for key in list(model_level_weights.keys()):
        weights[key] = torch.concat(model_level_weights[key], dim=0)
        weights[key] = torch.chunk(weights[key], split_factor,
                                   dim=0)[inference_tp_rank % split_factor]
        del model_level_weights[key]
    for key, param in weights.items():
        weights[key] = weights[key].to(dtype).contiguous()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights
