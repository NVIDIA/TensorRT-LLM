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
# limitations under the License.import functools

import copy
import functools
import json
import os
import time
from collections import defaultdict
from typing import List, Optional

import numpy as np
import safetensors
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.pytorch_utils import Conv1D

from ..._utils import pad_vocab_size, str_dtype_to_torch
from ...logger import logger
from ...mapping import Mapping
from ...quantization import QuantAlgo
from ..convert_utils import (dup_kv_bias, dup_kv_weight, generate_int8,
                             get_weight, get_weight_and_bias,
                             load_calib_dataset, smooth_gemm,
                             smooth_gemm_fc1_gate, split, split_matrix_tp,
                             split_qkv_bias_tp, split_qkv_tp)
from .config import QWenConfig
from .utils import get_qwen_key_list, make_context


@torch.no_grad()
def smooth_qwen_model(model, scales, alpha, qwen_qkv_para, qwen_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not module._get_name() == "QWenBlock":
            continue
        # qkv_proj
        layer_name = name + ".attn.c_attn"
        smoother = smooth_gemm(module.attn.c_attn.weight,
                               scales[layer_name]["x"], module.ln_1.weight,
                               None, alpha)

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_attn.weight.abs().max(dim=1)[0]

        # see transpose_weights function
        qwen_qkv_para[layer_name] = module.attn.c_attn.weight.transpose(
            0, 1).contiguous()

        # =================================================================
        layer_name = name + ".attn.c_proj"
        smoother = smooth_gemm(
            module.attn.c_proj.weight,
            scales[layer_name]["x"],
            None,
            None,
            alpha=alpha,
        )
        qwen_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_proj.weight.abs().max(dim=1)[0]
        # ==================================================================
        fc1_layer_name = name + ".mlp.w1"
        gate_layer_name = name + ".mlp.w2"

        smoother = smooth_gemm_fc1_gate(module.mlp.w1.weight,
                                        module.mlp.w2.weight,
                                        scales[fc1_layer_name]["x"],
                                        module.ln_2.weight, None, alpha)

        scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
        scales[fc1_layer_name]["w"] = module.mlp.w1.weight.abs().max(dim=1)[0]

        scales[gate_layer_name]["x"] = scales[gate_layer_name]["x"] / smoother
        scales[gate_layer_name]["w"] = module.mlp.w2.weight.abs().max(dim=1)[0]

        # ==================================================================
        layer_name = name + ".mlp.c_proj"
        smoother = smooth_gemm(module.mlp.c_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        qwen_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.c_proj.weight.abs().max(dim=1)[0]


@torch.no_grad()
def smooth_qwen2_model(model, scales, alpha, qwen_qkv_para, qwen_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        from transformers.models.qwen2_vl.modeling_qwen2_vl import \
            Qwen2VLDecoderLayer
        if not isinstance(module, Qwen2DecoderLayer) and not isinstance(
                module, Qwen2VLDecoderLayer):
            continue
        # qkv_proj
        layer_name_q = name + ".self_attn.q_proj"
        layer_name_k = name + ".self_attn.k_proj"
        layer_name_v = name + ".self_attn.v_proj"
        layer_name_qkv = name + ".self_attn.qkv_proj"

        weight = torch.cat([
            module.self_attn.q_proj.weight, module.self_attn.k_proj.weight,
            module.self_attn.v_proj.weight
        ],
                           dim=0)

        smoother = smooth_gemm(weight, scales[layer_name_q]["x"],
                               module.input_layernorm.weight, None, alpha)

        scales[layer_name_qkv]["x"] = scales[layer_name_q]["x"] / smoother
        scales[layer_name_qkv]["w"] = weight.abs().max(dim=1)[0]
        scales[layer_name_qkv]["y"] = torch.cat([
            scales[layer_name_q]["y"], scales[layer_name_k]["y"],
            scales[layer_name_v]["y"]
        ],
                                                dim=0)

        # see transpose_weights function
        qwen_qkv_para[layer_name_qkv] = weight.transpose(0, 1).contiguous()

        # =================================================================
        layer_name = name + ".self_attn.o_proj"
        smoother = smooth_gemm(module.self_attn.o_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        qwen_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.self_attn.o_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        fc1_layer_name = name + ".mlp.gate_proj"
        gate_layer_name = name + ".mlp.up_proj"

        smoother = smooth_gemm_fc1_gate(module.mlp.gate_proj.weight,
                                        module.mlp.up_proj.weight,
                                        scales[fc1_layer_name]["x"],
                                        module.post_attention_layernorm.weight,
                                        None, alpha)

        scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
        scales[fc1_layer_name]["w"] = module.mlp.gate_proj.weight.abs().max(
            dim=1)[0]

        scales[gate_layer_name]["x"] = scales[gate_layer_name]["x"] / smoother
        scales[gate_layer_name]["w"] = module.mlp.up_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        layer_name = name + ".mlp.down_proj"
        smoother = smooth_gemm(module.mlp.down_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        qwen_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.down_proj.weight.abs().max(
            dim=1)[0]

    scales_keys_to_rename = [
        key for key in scales.keys() if 'language_model.' in key
    ]

    qwen_qkv_para_keys_to_rename = [
        key for key in qwen_qkv_para.keys() if 'language_model.' in key
    ]

    qwen_smoother_keys_to_rename = [
        key for key in qwen_smoother.keys() if 'language_model.' in key
    ]

    for key in scales_keys_to_rename:
        scales[key.replace('language_model.', '')] = scales[key]
        del scales[key]

    for key in qwen_qkv_para_keys_to_rename:
        qwen_qkv_para[key.replace('language_model.', '')] = qwen_qkv_para[key]
        del qwen_qkv_para[key]

    for key in qwen_smoother_keys_to_rename:
        qwen_smoother[key.replace('language_model.', '')] = qwen_smoother[key]
        del qwen_smoother[key]


@torch.no_grad()
def capture_activation_range(model,
                             qwen_type,
                             tokenizer,
                             dataset,
                             system_prompt,
                             chat_format,
                             num_samples=512,
                             seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    if qwen_type == 'qwen':
        tokenizer.pad_token_id = tokenizer.im_end_id
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
                                                        None).max(dim=1)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="calibrating model"):
        line = dataset[i]
        line = line + ' TL;DR: '
        line = line.strip()
        line = line.replace(" n't", "n't")
        if qwen_type == 'qwen':
            _, input_id_list = make_context(tokenizer=tokenizer,
                                            query=line,
                                            history=[],
                                            system=system_prompt,
                                            chat_format=chat_format,
                                            max_input_length=seq_len)
            line_encoded = torch.from_numpy(
                np.array(input_id_list,
                         dtype=np.int32)).type(torch.int32).unsqueeze(0)
            line_encoded = line_encoded.to(device)
        else:
            line_encoded = tokenizer(line,
                                     return_tensors="pt",
                                     max_length=seq_len,
                                     padding=True,
                                     truncation=True).input_ids.to(device)
        model(line_encoded)
    for h in hooks:
        h.remove()
    return act_scales


def get_tllm_linear_weight(weight,
                           prefix,
                           bias=None,
                           use_weight_only=False,
                           plugin_weight_only_quant_type=torch.int8,
                           dtype='float32',
                           use_gemm_woq_plugin=True,
                           postfix='weight',
                           quant_scale_name=None):
    results = {}
    if use_weight_only:
        if weight.dim() > 2:
            v = weight.transpose(1, 2).contiguous().clone()
        else:
            v = weight.t().contiguous().clone()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v.cpu(), plugin_weight_only_quant_type)
        if not use_gemm_woq_plugin:
            results[prefix + postfix] = v.to(dtype)
        else:
            results[prefix + postfix] = processed_torch_weights
        if quant_scale_name is not None:
            results[quant_scale_name] = torch_weight_scales
        else:
            results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + postfix] = weight.clone()

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


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
        k_split = torch.split(k, k.shape[-1] // tp_size, dim=-1)
        v_split = torch.split(v, v.shape[-1] // tp_size, dim=-1)
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
            cur_weights = torch.chunk(original_weights,
                                      tensor_parallel,
                                      dim=cat_dim)[rank]
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
                    cur_per_channel_value = torch.split(
                        vals["scale_w_quant_orig"],
                        tensor_parallel,
                        axis=cat_dim)[rank]

        results[prefix +
                'per_channel_scale'] = cur_per_channel_value.reshape(col_shape)
    else:
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
            cur_weights = torch.chunk(original_weights,
                                      tensor_parallel,
                                      dim=cat_dim)[rank]
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix + 'weight'] = cur_weights.t().contiguous()

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
        cur_smoother_value = torch.split(smoother_value,
                                         smoother_value.shape[-1] //
                                         tensor_parallel,
                                         dim=cat_dim)[rank]

        results[prefix + 'smoother'] = cur_smoother_value.reshape(
            smoother_shape).contiguous().to(torch.float32)

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def load_hf_qwen(model_dir: str, load_model_on_cpu: bool = False):
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    if config['architectures'] == ['Qwen2ForSequenceClassification']:
        from transformers import Qwen2ForSequenceClassification as model_cls
    elif config['architectures'] == ['Qwen2VLForConditionalGeneration']:
        from transformers import Qwen2VLForConditionalGeneration as model_cls
    else:
        from transformers import AutoModelForCausalLM as model_cls

    model = model_cls.from_pretrained(
        model_dir,
        device_map='auto' if not load_model_on_cpu else 'cpu',
        dtype='auto',
        trust_remote_code=True)
    return model


def convert_hf_qwen(hf_model,
                    qwen_type,
                    mapping: Mapping,
                    vocab_size=32000,
                    dtype='float32',
                    use_parallel_embedding=False,
                    sharding_dim=0,
                    use_weight_only=False,
                    use_gemm_woq_plugin=False,
                    plugin_weight_only_quant_type=torch.int8,
                    use_smooth_quant=False,
                    per_channel=False,
                    per_token=False,
                    int8_kv_cache=False,
                    act_range=[],
                    qkv_para=[],
                    smoother=[],
                    moe_config=None):
    weights = {}
    tik = time.time()
    tensor_parallel = mapping.tp_size
    model_params = dict(hf_model.named_parameters())

    dtype = getattr(torch, dtype)
    hf_config = hf_model.config
    if hasattr(hf_config, 'llm_config'):
        hf_config = hf_config.llm_config

    #This is for InternVL2 - 1B
    keys_to_rename = [
        key for key in model_params.keys() if 'language_model.' in key
    ]
    keys_to_delete = [
        key for key in model_params.keys() if 'vision_model.' in key
    ]
    for key in keys_to_rename:
        keys_rename = key.replace('language_model.', '')
        model_params[keys_rename] = model_params[key]
        del model_params[key]
    for key in keys_to_delete:
        del model_params[key]

    num_attention_heads = hf_config.num_attention_heads
    hidden_size = hf_config.hidden_size
    head_size = hidden_size // num_attention_heads
    if qwen_type == 'qwen':
        intermediate_size = hf_config.intermediate_size // 2  # Qwen version 1 has actual intermediate_size one half of what's in hf_config
    else:
        intermediate_size = hf_config.intermediate_size
    num_key_value_heads = hf_config.num_key_value_heads if hasattr(
        hf_config, "num_key_value_heads") else num_attention_heads
    mha_mode = (num_key_value_heads == num_attention_heads)
    layers_range = mapping.pp_layers(hf_config.num_hidden_layers)

    layer_prefix = "transformer.h." if qwen_type == 'qwen' else "model.layers."
    key_list = get_qwen_key_list(qwen_type)

    for l in layers_range:
        prefix = layer_prefix + f'{l}.'
        tllm_prex = f'transformer.layers.{l - layers_range[0]}.'
        if qwen_type == 'qwen':
            qkv_weight, qkv_bias = get_weight_and_bias(model_params,
                                                       prefix + key_list[0],
                                                       dtype)
            qkv_w = split_qkv_tp(qkv_weight, num_attention_heads, hidden_size,
                                 tensor_parallel, mapping.tp_rank)
            qkv_b = split_qkv_bias_tp(qkv_bias, num_attention_heads,
                                      hidden_size, tensor_parallel,
                                      mapping.tp_rank)
        else:
            q_weight, q_bias = get_weight_and_bias(
                model_params, prefix + key_list[0] + 'q_proj', dtype)
            k_weight, k_bias = get_weight_and_bias(
                model_params, prefix + key_list[0] + 'k_proj', dtype)
            v_weight, v_bias = get_weight_and_bias(
                model_params, prefix + key_list[0] + 'v_proj', dtype)
            if not mha_mode:
                if num_key_value_heads < tensor_parallel:
                    # duplicate the KV heads up to tensor_parallel
                    k_weight = dup_kv_weight(k_weight, num_key_value_heads,
                                             tensor_parallel)
                    v_weight = dup_kv_weight(v_weight, num_key_value_heads,
                                             tensor_parallel)
                    k_bias = dup_kv_bias(k_bias, num_key_value_heads,
                                         tensor_parallel)
                    v_bias = dup_kv_bias(v_bias, num_key_value_heads,
                                         tensor_parallel)
                assert (k_weight.shape[0] % (mapping.tp_size * head_size)) == 0
                assert (v_weight.shape[0] % (mapping.tp_size * head_size)) == 0

                if k_bias is not None and v_bias is not None:
                    assert (k_bias.shape[0] %
                            (mapping.tp_size * head_size)) == 0
                    assert (v_bias.shape[0] %
                            (mapping.tp_size * head_size)) == 0

                wq = split(q_weight, mapping.tp_size, mapping.tp_rank)
                wk = split(k_weight, mapping.tp_size, mapping.tp_rank)
                wv = split(v_weight, mapping.tp_size, mapping.tp_rank)

                qkv_w = torch.concat((wq, wk, wv))

                if q_bias is not None and k_bias is not None and v_bias is not None:
                    bq = split(q_bias, mapping.tp_size, mapping.tp_rank)
                    bk = split(k_bias, mapping.tp_size, mapping.tp_rank)
                    bv = split(v_bias, mapping.tp_size, mapping.tp_rank)
                    qkv_b = torch.concat((bq, bk, bv))
                else:
                    qkv_b = None
            else:
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

                qkv_w = split_qkv_tp(qkv_weight, num_attention_heads,
                                     hidden_size, tensor_parallel,
                                     mapping.tp_rank)
                qkv_b = split_qkv_bias_tp(qkv_bias, num_attention_heads,
                                          hidden_size, tensor_parallel,
                                          mapping.tp_rank)

        if use_smooth_quant:
            qkv_proj_key = key_list[
                0] if qwen_type == 'qwen' else 'self_attn.qkv_proj'
            qkv_weight = qkv_para[prefix + qkv_proj_key]
            qkv_out_dim = qkv_weight.shape[1]

            if not mha_mode:
                local_dim = qkv_weight.shape[0]
                kv_hidden_size = (qkv_weight.shape[-1] - local_dim) // 2
                qkv_weight = qkv_weight.reshape(local_dim,
                                                local_dim + 2 * kv_hidden_size)
            else:
                qkv_weight = qkv_weight.reshape(hidden_size, 3, hidden_size)

            int8_weights = generate_int8(qkv_weight,
                                         act_range.get(prefix + qkv_proj_key),
                                         is_qkv=True,
                                         multi_query_mode=bool(not mha_mode))

            weights.update(
                get_tllm_linear_sq_weight(int8_weights,
                                          tllm_prex + 'attention.qkv.',
                                          [1, qkv_out_dim // tensor_parallel],
                                          tensor_parallel,
                                          is_qkv=True,
                                          per_token=per_token,
                                          per_channel=per_channel,
                                          last_prefix=tllm_prex +
                                          'input_layernorm.scale_to_int',
                                          bias=qkv_b,
                                          smoother_value=None,
                                          smoother_shape=None,
                                          rank=mapping.tp_rank,
                                          cat_dim=-1,
                                          multi_query_mode=bool(not mha_mode)))
        else:
            weights.update(
                get_tllm_linear_weight(qkv_w, tllm_prex + 'attention.qkv.',
                                       qkv_b, use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))
        if int8_kv_cache:
            if qwen_type == 'qwen':
                qkv_y = act_range.get(prefix + key_list[0])["y"]
            else:
                qkv_y = torch.cat([
                    act_range.get(prefix + key_list[0] + 'q_proj')["y"],
                    act_range.get(prefix + key_list[0] + 'k_proj')["y"],
                    act_range.get(prefix + key_list[0] + 'v_proj')["y"]
                ],
                                  dim=0)

            int8_kv_scales = qkv_y.max() / 127.

            kv_cache_weights = {}

            kv_cache_weights[
                tllm_prex +
                'attention.kv_cache_scaling_factor'] = int8_kv_scales.reshape(
                    [1])

            weights.update(kv_cache_weights)

        attn_dense_weight = get_weight(model_params, prefix + key_list[1],
                                       dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=1)
        if use_smooth_quant:
            attn_dense_weight = attn_dense_weight.t()
            int8_weights = generate_int8(attn_dense_weight,
                                         act_range.get(prefix + key_list[1]))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'attention.dense.', [1, hidden_size],
                    tensor_parallel,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex +
                    'attention.quantization_scaling_factor',
                    smoother_value=smoother[(prefix + key_list[1])],
                    smoother_shape=[1, hidden_size // tensor_parallel],
                    rank=mapping.tp_rank,
                    cat_dim=0))
        else:
            weights.update(
                get_tllm_linear_weight(split_v, tllm_prex + 'attention.dense.',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

        # Qwen3: Add q_norm and k_norm weight conversion
        if qwen_type in ('qwen3', 'qwen3_moe'):
            # Process q_norm.weight
            q_norm_weight = get_weight(model_params,
                                       prefix + key_list[0] + 'q_norm', dtype)
            weights.update(
                get_tllm_linear_weight(
                    q_norm_weight,
                    tllm_prex + 'attention.q_layernorm.',
                    None,
                    False,  # LayerNorm should not be quantized
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin))

            # Process k_norm.weight
            k_norm_weight = get_weight(model_params,
                                       prefix + key_list[0] + 'k_norm', dtype)
            weights.update(
                get_tllm_linear_weight(
                    k_norm_weight,
                    tllm_prex + 'attention.k_layernorm.',
                    None,
                    False,  # LayerNorm should not be quantized
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin))

        if moe_config and moe_config.has_moe():
            if qwen_type == "qwen2_moe":
                # shared_expert for qwen2_moe
                shared_expert_up_proj = model_params[
                    f'model.layers.{l}.mlp.shared_expert.up_proj.weight']
                shared_expert_down_proj = model_params[
                    f'model.layers.{l}.mlp.shared_expert.down_proj.weight']
                shared_expert_gate = model_params[
                    f'model.layers.{l}.mlp.shared_expert.gate_proj.weight']
                shared_expert_up_proj = split(shared_expert_up_proj,
                                              mapping.tp_size,
                                              mapping.tp_rank,
                                              dim=0)
                shared_expert_down_proj = split(shared_expert_down_proj,
                                                mapping.tp_size,
                                                mapping.tp_rank,
                                                dim=1)
                shared_expert_gate = split(shared_expert_gate,
                                           mapping.tp_size,
                                           mapping.tp_rank,
                                           dim=0)
                shared_expert_gate_up_proj = torch.concat(
                    [shared_expert_up_proj, shared_expert_gate],
                    dim=-2).to(dtype)

                ## mlp.shared_expert.gate_up_proj.weight
                weights.update(
                    get_tllm_linear_weight(shared_expert_gate_up_proj,
                                           tllm_prex + 'mlp.shared_expert.fc.',
                                           None, use_weight_only,
                                           plugin_weight_only_quant_type, dtype,
                                           use_gemm_woq_plugin))

                ## mlp.shared_expert.down_proj.weight
                weights.update(
                    get_tllm_linear_weight(
                        shared_expert_down_proj.to(dtype),
                        tllm_prex + 'mlp.shared_expert.proj.', None,
                        use_weight_only, plugin_weight_only_quant_type, dtype,
                        use_gemm_woq_plugin))

                moe_shared_expert_gate_weights = get_weight(
                    model_params, prefix + 'mlp.shared_expert_gate', dtype)
                weights.update(
                    get_tllm_linear_weight(
                        moe_shared_expert_gate_weights,
                        tllm_prex + 'mlp.shared_expert_gate.',
                        None,
                        False,  # Router should never be quantized
                        plugin_weight_only_quant_type,
                        dtype,
                        use_gemm_woq_plugin))

            ## fine-grained experts
            rank_experts = list(range(moe_config.num_experts))
            if mapping.has_moe_ep():
                rank_experts = mapping.ep_experts(moe_config.num_experts)
            for suffix in ["gate_proj", "down_proj", "up_proj"]:
                model_params[f'model.layers.{l}.mlp.experts.{suffix}.weight'] = \
                            torch.stack([model_params[f'model.layers.{l}.mlp.experts.{expert}.{suffix}.weight'].detach()
                                        for expert in rank_experts])
            w3 = model_params[f'model.layers.{l}.mlp.experts.up_proj.weight']
            w2 = model_params[f'model.layers.{l}.mlp.experts.down_proj.weight']
            w1 = model_params[f'model.layers.{l}.mlp.experts.gate_proj.weight']
            if mapping.has_moe_tp():
                w3 = split(w3, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)
                w2 = split(w2, mapping.moe_tp_size, mapping.moe_tp_rank, dim=2)
                w1 = split(w1, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)

            moe_experts_w3w1_weights = torch.concat([w3, w1], dim=-2).to(dtype)

            ## mlp.experts.w2.weight
            weights.update(
                get_tllm_linear_weight(w2.to(dtype), tllm_prex + 'mlp.proj.',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

            ## mlp.experts.w3w1.weight
            weights.update(
                get_tllm_linear_weight(moe_experts_w3w1_weights,
                                       tllm_prex + 'mlp.fc.', None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

            moe_experts_gate_weights = get_weight(model_params,
                                                  prefix + 'mlp.gate',
                                                  torch.float32)
            weights.update(
                get_tllm_linear_weight(
                    moe_experts_gate_weights,
                    tllm_prex + 'mlp.router.',
                    None,
                    False,  # Router should never be quantized
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin))

        else:
            mlp_gate_weight = get_weight(model_params, prefix + key_list[2],
                                         dtype)
            split_v = split_matrix_tp(mlp_gate_weight,
                                      tensor_parallel,
                                      mapping.tp_rank,
                                      dim=0)
            if use_smooth_quant:
                mlp_gate_weight = mlp_gate_weight.t()
                int8_weights = generate_int8(
                    mlp_gate_weight, act_range.get(prefix + key_list[2]))

                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.gate.',
                        [1, intermediate_size // tensor_parallel],
                        tensor_parallel,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                        smoother_value=None,
                        smoother_shape=None,
                        rank=mapping.tp_rank,
                        cat_dim=-1))
            else:
                weights.update(
                    get_tllm_linear_weight(split_v, tllm_prex + 'mlp.gate.',
                                           None, use_weight_only,
                                           plugin_weight_only_quant_type, dtype,
                                           use_gemm_woq_plugin))

            mlp_fc_weight = get_weight(model_params, prefix + key_list[3],
                                       dtype)
            split_v = split_matrix_tp(mlp_fc_weight,
                                      tensor_parallel,
                                      mapping.tp_rank,
                                      dim=0)

            if use_smooth_quant:
                mlp_fc_weight = mlp_fc_weight.t()  #verified
                int8_weights = generate_int8(
                    mlp_fc_weight, act_range.get(prefix + key_list[3]))

                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.fc.',
                        [1, intermediate_size // tensor_parallel],
                        tensor_parallel,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                        smoother_value=None,
                        smoother_shape=None,
                        rank=mapping.tp_rank,
                        cat_dim=-1))
            else:
                weights.update(
                    get_tllm_linear_weight(split_v, tllm_prex + 'mlp.fc.', None,
                                           use_weight_only,
                                           plugin_weight_only_quant_type, dtype,
                                           use_gemm_woq_plugin))

            mlp_proj_weight = get_weight(model_params, prefix + key_list[4],
                                         dtype)
            split_v = split_matrix_tp(mlp_proj_weight,
                                      tensor_parallel,
                                      mapping.tp_rank,
                                      dim=1)

            if use_smooth_quant:
                mlp_proj_weight = mlp_proj_weight.t()
                int8_weights = generate_int8(
                    mlp_proj_weight, act_range.get(prefix + key_list[4]))

                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.proj.', [1, hidden_size],
                        tensor_parallel,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex +
                        'mlp.quantization_scaling_factor',
                        smoother_value=smoother[prefix + key_list[4]],
                        smoother_shape=[
                            1, intermediate_size // tensor_parallel
                        ],
                        rank=mapping.tp_rank,
                        cat_dim=0))
            else:
                weights.update(
                    get_tllm_linear_weight(split_v, tllm_prex + 'mlp.proj.',
                                           None, use_weight_only,
                                           plugin_weight_only_quant_type, dtype,
                                           use_gemm_woq_plugin))

        # Layer norms do not use tensor parallelism
        input_ln_weight = get_weight(model_params, prefix + key_list[5], dtype)
        weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight

        post_ln_weight = get_weight(model_params, prefix + key_list[6], dtype)
        weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight

    v = get_weight(model_params, key_list[7], dtype)

    if mapping.is_last_pp_rank():
        if hf_config.tie_word_embeddings:
            # lm_head.weight has the same weights as embedding
            lm_head_weights = v.clone()
        else:
            lm_head_weights = get_weight(model_params, 'lm_head', dtype)

        if vocab_size % mapping.tp_size != 0:
            # padding
            vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
            pad_width = vocab_size_padded - vocab_size

            lm_head_weights = torch.from_numpy(
                np.pad(lm_head_weights.detach().cpu().numpy(),
                       ((0, pad_width), (0, 0)),
                       'constant',
                       constant_values=0))
        weights['lm_head.weight'] = split_matrix_tp(lm_head_weights,
                                                    tensor_parallel,
                                                    mapping.tp_rank,
                                                    dim=0)

    if use_parallel_embedding:
        v = split_matrix_tp(v,
                            mapping.tp_size,
                            mapping.tp_rank,
                            dim=sharding_dim)

    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v

    if mapping.is_last_pp_rank():
        ln_f_w = get_weight(model_params, key_list[8], dtype)
        weights['transformer.ln_f.weight'] = ln_f_w

    if hasattr(hf_model, 'score'):
        score = get_weight(model_params, 'score', dtype)
        weights['lm_head.weight'] = score

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def quantize(hf_model_dir: str,
             output_dir: str,
             config: QWenConfig,
             calib_dataset='cnn_dailymail'):
    '''
        Quantize the save the model as TRT-LLM checkpoint to output_dir
    '''
    os.makedirs(output_dir, exist_ok=True)
    config.to_json_file(os.path.join(output_dir, 'config.json'))

    mapping = config.mapping
    assert mapping.rank == 0, "quantize should be called at rank 0 only"

    quant_config = config.quantization
    use_smooth_quant = quant_config._use_plugin_sq
    int8_kv_cache = quant_config.kv_cache_quant_algo == "INT8"

    assert use_smooth_quant or int8_kv_cache, "Call from_hugging_face when there is no quantization"
    assert hf_model_dir is not None
    ## only load and call smooth quant routine once for all ranks
    hf_config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    if hf_config.architectures == ['Qwen2VLForConditionalGeneration']:
        from transformers import Qwen2VLForConditionalGeneration as model_cls
    else:
        from transformers import AutoModelForCausalLM as model_cls
    hf_model = model_cls.from_pretrained(
        hf_model_dir,
        device_map='auto',
        dtype='auto' if not use_smooth_quant else torch.float16,
        trust_remote_code=True).half()

    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
        "TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir,
                                              trust_remote_code=True,
                                              use_fast=False,
                                              padding_side='left')
    dataset = load_calib_dataset(calib_dataset)

    system_prompt = "You are a useful assistant, please directly output the corresponding summary according to the article entered by the user."
    gen_config_path = os.path.join(hf_model_dir, 'generation_config.json')
    with open(gen_config_path, 'r') as f:
        gen_config = json.load(f)
    chat_format = getattr(gen_config, 'chat_format', 'chatml')
    act_range = capture_activation_range(hf_model, config.qwen_type, tokenizer,
                                         dataset, system_prompt, chat_format)
    qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    smoother = {}
    if use_smooth_quant:
        if config.qwen_type == 'qwen':
            smooth_qwen_model(hf_model, act_range, quant_config.smoothquant_val,
                              qkv_para, smoother)
        else:
            smooth_qwen2_model(hf_model, act_range,
                               quant_config.smoothquant_val, qkv_para, smoother)

    for rank in range(mapping.world_size):
        # To avoid changing the mapping arg in-place, also the given mapping from caller is rank agnostic, since quantize is called from only one rank
        config = copy.deepcopy(config)
        config.set_rank(rank)
        weights = load_weights_from_hf_model(hf_model,
                                             config=config,
                                             act_range=act_range,
                                             qkv_para=qkv_para,
                                             smoother=smoother)
        safetensors.torch.save_file(
            weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
        del weights


def load_weights_from_hf_model(hf_model,
                               config: QWenConfig,
                               act_range: Optional[dict] = None,
                               qkv_para: Optional[dict] = None,
                               smoother: Optional[dict] = None):
    #TODO: simplify the parameters here

    assert hf_model is not None
    plugin_weight_only_quant_type = None  # the value does not matter when use_weight_only is False
    quant_algo = config.quantization.quant_algo
    if quant_algo == QuantAlgo.W8A16:
        plugin_weight_only_quant_type = torch.int8
    elif quant_algo == QuantAlgo.W4A16:
        plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None
    use_gemm_woq_plugin = (not config.disable_weight_only_quant_plugin)

    mapping = config.mapping
    moe_config = config.moe

    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    use_smooth_quant = config.quantization._use_plugin_sq
    per_channel = use_smooth_quant and 'PER_CHANNEL' in quant_algo
    per_token = use_smooth_quant and 'PER_TOKEN' in quant_algo
    int8_kv_cache = config.quantization.kv_cache_quant_algo == QuantAlgo.INT8
    qwen_type = config.qwen_type
    weights = convert_hf_qwen(
        hf_model,
        qwen_type,
        mapping,
        vocab_size=config.vocab_size,
        dtype=config.dtype,
        use_weight_only=use_weight_only,
        use_gemm_woq_plugin=use_gemm_woq_plugin,
        plugin_weight_only_quant_type=plugin_weight_only_quant_type,
        use_parallel_embedding=config.use_parallel_embedding,
        sharding_dim=config.embedding_sharding_dim,
        use_smooth_quant=use_smooth_quant,
        per_channel=per_channel,
        per_token=per_token,
        int8_kv_cache=int8_kv_cache,
        act_range=act_range,
        qkv_para=qkv_para,
        smoother=smoother,
        moe_config=moe_config)
    return weights


def load_weights_from_hf_gptq_model(hf_model, config: QWenConfig):
    logger.info("loading weights from groupwise GPTQ QWen safetensors...")
    weights = {}
    tik = time.time()

    qwen_type = config.qwen_type
    num_hidden_layers = config.num_hidden_layers
    mapping = config.mapping
    dtype = config.dtype

    model_params = {k: v for k, v in hf_model.state_dict().items()}
    torch.cuda.empty_cache()
    valid_types = ('qwen', 'qwen2', 'qwen2_vl', 'qwen3', 'qwen3_moe')
    assert qwen_type in valid_types, f"Unsupported Qwen type: {qwen_type}, only {valid_types} are supported for GPTQ."
    layer_prefix = "transformer.h." if qwen_type == 'qwen' else "model.layers."
    key_list = get_qwen_key_list(qwen_type)

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def unpack_int32_into_int8(w_packed):
        # unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                                 w_packed_int4x2.shape[1] * 2,
                                 dtype=torch.int8)
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.contiguous()

    def process_and_assign_weight(v: List[torch.Tensor],
                                  tllm_prex: str,
                                  tp_dim: int = -1):
        if tp_dim == -1:
            qweight_int32, qzeros_int32, scales_fp16 = [
                item.cpu() for item in v
            ]
        else:
            qweight_int32, qzeros_int32, scales_fp16 = [
                torch_split(item, tp_dim).cpu() for item in v
            ]

        USE_UINT4_INPUT = 1  # Set to true if checkpoint store UINT4 weights
        USE_GPTQ_FOR_QWEN = 1  # GPTQ-for-QWEN added 1 to zeros

        qweight_unpacked_int8 = unpack_int32_into_int8(
            qweight_int32.T).T.contiguous() - 8
        qweight_interleaved = preprocessor(packer(qweight_unpacked_int8),
                                           torch.quint4x2,
                                           torch.float16).view(torch.float16)
        # zeros = zeros * scales
        qzeros_unpacked_int32 = unpack_int32_into_int8(qzeros_int32)
        if not USE_UINT4_INPUT:
            # Correcting UINT4 values back to INT4 order
            mask_negative = qzeros_unpacked_int32[qzeros_unpacked_int32 < 0]
            mask_positive = qzeros_unpacked_int32[qzeros_unpacked_int32 >= 0]
            qzeros_unpacked_int32 = qzeros_unpacked_int32 + 16 * mask_negative - 16 * mask_positive
        zeros_x_scales_fp16 = (-qzeros_unpacked_int32 + 8 * USE_UINT4_INPUT -
                               USE_GPTQ_FOR_QWEN) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        results = {
            f'{tllm_prex}.weight': qweight_interleaved,
            f'{tllm_prex}.weights_scaling_factor': scales_fp16,
            f'{tllm_prex}.zero': zeros_x_scales_fp16,
        }
        return results

    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    # Load weights from GPTQ checkpoint into TRT-LLM module
    # 1. vocab_embedding
    v = model_params[key_list[7] + '.weight']
    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v.to(torch_dtype)

    # 2. ln_f
    v = model_params[key_list[8] + '.weight']
    if mapping.is_last_pp_rank():
        weights['transformer.ln_f.weight'] = v.to(torch_dtype)

    # 3. lm_head
    v = model_params['lm_head.weight']
    if mapping.is_last_pp_rank():
        weights['lm_head.weight'] = torch_split(v, 0).to(torch_dtype)

    # 4. Weights inside each layer
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))
    suffixs = [".qweight", ".qzeros", ".scales"]

    for l in tqdm(layers_range, desc="loading weight in each layer..."):
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = layer_prefix + str(layer_idx) + "."
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'
        # 4.1 attention.qkv
        qkv_weight_list = []
        if qwen_type == 'qwen':
            for suf in suffixs:
                qkv_part = model_params[prefix + key_list[0] + suf]
                q_emb = qkv_part.shape[1] // 3
                model_emb = qkv_part.shape[0]
                qkv_part = qkv_part.reshape(model_emb, 3, q_emb)
                qkv_part = torch_split(qkv_part, 2)
                qkv_part = qkv_part.reshape(model_emb,
                                            3 * (q_emb // mapping.tp_size))
                qkv_weight_list.append(qkv_part)
        else:
            for suf in suffixs:
                qkv_list = []
                for comp in ["q_proj", "k_proj", "v_proj"]:
                    comp_part = model_params[prefix + key_list[0] + comp + suf]
                    comp_part = torch_split(comp_part, 1)
                    qkv_list.append(comp_part)
                qkv_weight_list.append(torch.cat(qkv_list, dim=1))
        weights.update(
            process_and_assign_weight(qkv_weight_list,
                                      f'{tllm_prex}.attention.qkv'))
        # 4.2 attention.bias
        suf = ".bias"
        if qwen_type == 'qwen':
            qkv_bias = model_params[prefix + key_list[0] +
                                    suf].to(torch_dtype).cpu().contiguous()
            q_emb = qkv_bias.shape[0] // 3
            qkv_bias = qkv_bias.reshape(3, q_emb)
            split_v = split(qkv_bias, mapping.tp_size, mapping.rank, dim=1)
            qkv_bias = split_v.reshape(3 * (q_emb // mapping.tp_size))
        else:
            qkv_bias_list = []
            for comp in ["q_proj", "k_proj", "v_proj"]:
                comp_part = model_params[prefix + key_list[0] + comp + suf].to(
                    torch_dtype).cpu().contiguous()
                comp_part = torch_split(comp_part, dim=0)
                qkv_bias_list.append(comp_part)
            qkv_bias = torch.cat(qkv_bias_list, dim=0)
        weights[tllm_prex + ".attention.qkv.bias"] = qkv_bias
        # 4.3 attention.dense
        qkv_dense_list = []
        for suf in suffixs:
            qkv_dense_part = model_params[prefix + key_list[1] + suf]
            qkv_dense_list.append(qkv_dense_part)
        weights.update(
            process_and_assign_weight(qkv_dense_list,
                                      f'{tllm_prex}.attention.dense',
                                      tp_dim=0))
        # 4.4 mlp.gate
        mlp_gate_list = []
        for suf in suffixs:
            mlp_gate_part = model_params[prefix + key_list[2] + suf]
            mlp_gate_list.append(mlp_gate_part)
        weights.update(
            process_and_assign_weight(mlp_gate_list,
                                      f'{tllm_prex}.mlp.gate',
                                      tp_dim=1))
        # 4.5 mlp.fc
        mlp_fc_list = []
        for suf in suffixs:
            mlp_fc_part = model_params[prefix + key_list[3] + suf]
            mlp_fc_list.append(mlp_fc_part)
        weights.update(
            process_and_assign_weight(mlp_fc_list,
                                      f'{tllm_prex}.mlp.fc',
                                      tp_dim=1))
        # 4.6 mlp.proj
        mlp_proj_list = []
        for suf in suffixs:
            mlp_proj_part = model_params[prefix + key_list[4] + suf]
            mlp_proj_list.append(mlp_proj_part)
        weights.update(
            process_and_assign_weight(mlp_proj_list,
                                      f'{tllm_prex}.mlp.proj',
                                      tp_dim=0))
        # 4.7 input_layernorm
        v = model_params[prefix + key_list[5] + '.weight']
        weights[f'{tllm_prex}.input_layernorm.weight'] = v.to(torch_dtype)
        # 4.8 post_layernorm
        v = model_params[prefix + key_list[6] + '.weight']
        weights[f'{tllm_prex}.post_layernorm.weight'] = v.to(torch_dtype)

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"weights loaded. total time: {t}")

    return weights
