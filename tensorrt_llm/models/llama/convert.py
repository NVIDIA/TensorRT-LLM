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
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import safetensors
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.pytorch_utils import Conv1D

from ..._utils import pad_vocab_size, release_gc, str_dtype_to_torch
from ...logger import logger
from ...quantization import QuantAlgo
from ..convert_utils import (dup_kv_weight, generate_int8,
                             get_tllm_linear_weight, iterate_shard_files,
                             load_calib_dataset, load_state_dict,
                             retrieved_layer_index_from_name, smooth_gemm,
                             smooth_gemm_fc1_gate, split, split_matrix_tp,
                             split_qkv_bias_tp, split_qkv_tp)
from ..modeling_utils import PretrainedConfig
from .config import LLaMAConfig


@torch.no_grad()
def smooth_llama_model(model, scales, alpha, llama_qkv_para, llama_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(
                module,
                LlamaDecoderLayer) and not module.__class__.__name__ in [
                    "InternLMDecoderLayer", "MistralDecoderLayer"
                ]:
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
        llama_qkv_para[layer_name_qkv] = weight.transpose(0, 1)

        # =================================================================
        layer_name = name + ".self_attn.o_proj"
        smoother = smooth_gemm(module.self_attn.o_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        llama_smoother[layer_name] = smoother.float()

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
        llama_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.down_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        if hasattr(module, 'residual_mlp'):
            fc1_layer_name = name + ".residual_mlp.w1"
            gate_layer_name = name + ".residual_mlp.w3"

            smoother = smooth_gemm_fc1_gate(module.residual_mlp.w1.weight,
                                            module.residual_mlp.w3.weight,
                                            scales[fc1_layer_name]["x"],
                                            module.residual_layernorm.weight,
                                            None, alpha)

            scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
            scales[fc1_layer_name]["w"] = module.residual_mlp.w1.weight.abs(
            ).max(dim=1)[0]

            scales[gate_layer_name][
                "x"] = scales[gate_layer_name]["x"] / smoother
            scales[gate_layer_name]["w"] = module.residual_mlp.w3.weight.abs(
            ).max(dim=1)[0]

            # ==================================================================
            layer_name = name + ".residual_mlp.w2"
            smoother = smooth_gemm(module.residual_mlp.w2.weight,
                                   scales[layer_name]["x"], None, None, alpha)
            llama_smoother[layer_name] = smoother.float()
            scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
            scales[layer_name]["w"] = module.residual_mlp.w2.weight.abs().max(
                dim=1)[0]


@torch.no_grad()
def capture_activation_range(model,
                             tokenizer,
                             dataset,
                             num_samples=512,
                             seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    tokenizer.pad_token = tokenizer.eos_token

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
        datapoint = dataset[i:i + 1]
        line = copy.copy(datapoint)
        line[0] = line[0] + ' TL;DR: '
        line[0] = line[0].strip()
        line[0] = line[0].replace(" n't", "n't")
        input_ids = tokenizer(line,
                              return_tensors="pt",
                              max_length=seq_len,
                              padding=True,
                              truncation=True).input_ids.to(device)
        model(input_ids)
    for h in hooks:
        h.remove()
    return act_scales


def get_weight(named_params, prefix, dtype):
    if named_params[prefix + '.weight'].dtype != dtype:
        named_params[prefix +
                     '.weight'].data = named_params[prefix +
                                                    '.weight'].to(dtype)
    return named_params[prefix + '.weight'].detach()


def get_weight_and_scale(named_params,
                         prefix,
                         dtype,
                         mapping=None,
                         split_scale=False):
    if prefix + '.weight_scale' not in named_params:
        return get_weight(named_params, prefix, dtype), None
    else:
        assert named_params[prefix + '.weight'].dtype == torch.float8_e4m3fn
        assert named_params[prefix + '.weight_scale'].dtype == torch.float32
        weight_scale = named_params[prefix + '.weight_scale'].detach()
        if split_scale:
            weight_scale = split(weight_scale,
                                 mapping.tp_size,
                                 mapping.tp_rank,
                                 dim=0)
        return named_params[prefix +
                            '.weight'].detach(), weight_scale.reshape(-1)


def get_bias(named_params, prefix, dtype):
    if named_params[prefix + '.bias'].dtype != dtype:
        named_params[prefix + '.bias'].data = named_params[prefix +
                                                           '.bias'].to(dtype)
    return named_params[prefix + '.bias'].detach()


def get_weight_and_bias(named_params, prefix, dtype):
    return get_weight(named_params, prefix,
                      dtype), get_bias(named_params, prefix, dtype)


def fp8_per_channel_quant_weight_gpu(weight, clamp_val, rank=0):
    weight = weight.to("cuda:" + str(rank))
    # activation range bound.
    x = weight.to(torch.float32).clamp(clamp_val[0], clamp_val[1])
    xmax = x.abs().max(-1, keepdim=True).values
    # minimum scaling factor.
    torch_weight_scales = (xmax / 448.0).clamp(min=1.0 / (448.0 * 512.0))
    out = x / torch_weight_scales
    torch_weight_scales = torch_weight_scales.reshape(-1)
    out = torch.clamp(out, -448, 448)
    processed_torch_weights = out.to(torch.float8_e4m3fn)

    processed_torch_weights = processed_torch_weights.to(
        torch.float8_e4m3fn).cpu()
    torch_weight_scales = torch_weight_scales.cpu()

    return processed_torch_weights, torch_weight_scales


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
        q, k, v = np.split(data, [local_dim, local_dim + head_size], axis=-1)
        q_split = np.split(q, tp_size, axis=-1)
        k_split = np.split(k, tp_size, axis=-1)
        v_split = np.split(v, tp_size, axis=-1)
        return [
            np.concatenate((q_split[ii], k_split[ii], v_split[ii]), axis=-1)
            for ii in range(tp_size)
        ][cur_rank]

    col_shape = shape if (is_qkv or per_channel) else [1, 1]

    if per_token:
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

        results[prefix + 'per_channel_scale'] = torch.from_numpy(
            np.array(cur_per_channel_value,
                     dtype=np.float32).reshape(col_shape)).contiguous()
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
        cur_smoother_value = np.split(smoother_value,
                                      tensor_parallel,
                                      axis=cat_dim)[rank]
        results[prefix + 'smoother'] = cur_smoother_value.reshape(
            smoother_shape).contiguous().to(torch.float32)

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def get_prefix_and_param_name_map(architecture, use_safetensors=False):

    key_postfix = ""
    if use_safetensors:
        key_postfix = ".weight"

    architecture = architecture.lower()
    if "exaone" in architecture:
        model_prefix = "transformer"
        param_name_map = {
            "vocab_embedding": "wte" + key_postfix,  # vocab_embedding
            "lm_head": "lm_head" + key_postfix,  # lm_head
            "ln_f": "ln_f" + key_postfix,  # ln_f
            "attention.qkv": "attn.attention",  # attention.qkv
            "qkv_suffix": "_proj" + key_postfix,  # qkv_suffix
            "attention.dense":
            "attn.attention.out_proj" + key_postfix,  # attention.dense
            "mlp.gate": "mlp.c_fc_1" + key_postfix,  # mlp.gate
            "mlp.proj": "mlp.c_proj" + key_postfix,  # mlp.proj
            "mlp.fc": "mlp.c_fc_0" + key_postfix,  # mlp.fc
            "input_layernorm": "ln_1" + key_postfix,  # input_layernorm
            "post_layernorm": "ln_2" + key_postfix,  # post_layernorm
        }
        layer_prefix = 'h'
    else:  # LLaMA
        model_prefix = "model"
        param_name_map = {
            "vocab_embedding": "embed_tokens" + key_postfix,  # vocab_embedding
            "lm_head": "lm_head" + key_postfix,  # lm_head
            "ln_f": "norm" + key_postfix,  # ln_f
            "attention.qkv": "self_attn",  # attention.qkv
            "qkv_suffix": "_proj" + key_postfix,  # qkv suffix
            "attention.dense":
            "self_attn.o_proj" + key_postfix,  # attention.dense
            "mlp.gate": "mlp.up_proj" + key_postfix,  # mlp.gate
            "mlp.proj": "mlp.down_proj" + key_postfix,  # mlp.proj
            "mlp.fc": "mlp.gate_proj" + key_postfix,  # mlp.fc
            "input_layernorm":
            "input_layernorm" + key_postfix,  # input_layernorm
            "post_layernorm":
            "post_attention_layernorm" + key_postfix,  # post_layernorm
        }
        layer_prefix = 'layers'

    return model_prefix, layer_prefix, param_name_map


def load_hf_llama(model_dir: str, load_model_on_cpu: bool = False):
    if "vila" in model_dir:
        sys.path.append(model_dir + "/../VILA")
        from llava.model import LlavaLlamaConfig, LlavaLlamaModel  # noqa
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_dir,
            device_map='auto',
            trust_remote_code=True,
        )
        return model.llm

    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_cls = AutoModelForCausalLM
    if hf_config.model_type == "llava":
        from transformers import LlavaForConditionalGeneration
        model_cls = LlavaForConditionalGeneration
    if hf_config.model_type == "llava_next":
        from transformers import LlavaNextForConditionalGeneration
        model_cls = LlavaNextForConditionalGeneration
    model = model_cls.from_pretrained(
        model_dir,
        device_map='auto' if not load_model_on_cpu else 'cpu',
        torch_dtype='auto',
        trust_remote_code=True,
    )
    if hf_config.model_type in ["llava", "llava_next"]:
        model = model.language_model
    return model


def load_weights_from_hf_model(hf_model,
                               config: LLaMAConfig,
                               act_range: Optional[dict] = None,
                               qkv_para: Optional[dict] = None,
                               smoother: Optional[dict] = None):
    quant_algo = config.quantization.quant_algo
    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    if quant_algo == QuantAlgo.W8A16:
        plugin_weight_only_quant_type = torch.int8
    elif quant_algo == QuantAlgo.W4A16:
        plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None
    use_gemm_woq_plugin = (not config.disable_weight_only_quant_plugin)
    use_fp8_rowwise = quant_algo in [QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN]

    use_smooth_quant = config.quantization.use_plugin_sq
    per_channel = use_smooth_quant and 'PER_CHANNEL' in quant_algo
    per_token = use_smooth_quant and 'PER_TOKEN' in quant_algo
    int8_kv_cache = config.quantization.kv_cache_quant_algo == QuantAlgo.INT8
    fp8_kv_cache = config.quantization.kv_cache_quant_algo == QuantAlgo.FP8
    if use_smooth_quant or int8_kv_cache:
        assert act_range is not None
        assert qkv_para is not None
        assert smoother is not None

    weights = {}
    tik = time.time()
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, config.dtype)

    mapping = config.mapping
    moe_config = config.moe
    mha_mode = (config.num_key_value_heads == config.num_attention_heads)
    layers_range = config.mapping.pp_layers(config.num_hidden_layers)
    exclude_layers_id = [0, config.num_hidden_layers - 1]

    model_prefix, layer_prefix, param_name_map = get_prefix_and_param_name_map(
        config.architecture)

    def convert_layer(l):
        prefix = f'{model_prefix}.{layer_prefix}.{l}.'
        tllm_prex = f'transformer.layers.{l - layers_range[0]}.'
        q_weight = get_weight(
            model_params, prefix + f'{param_name_map["attention.qkv"]}.q_proj',
            dtype)
        k_weight = get_weight(
            model_params, prefix + f'{param_name_map["attention.qkv"]}.k_proj',
            dtype)
        v_weight = get_weight(
            model_params, prefix + f'{param_name_map["attention.qkv"]}.v_proj',
            dtype)

        # Meta's recipe of not using fp8 rowwise for the first and last layer.
        use_fp8_rowwise_in_layer = use_fp8_rowwise and (
            l not in exclude_layers_id)

        if not mha_mode:
            if config.num_key_value_heads < mapping.tp_size:
                # duplicate the KV heads up to tensor_parallel
                k_weight = dup_kv_weight(k_weight, config.num_key_value_heads,
                                         mapping.tp_size)
                v_weight = dup_kv_weight(v_weight, config.num_key_value_heads,
                                         mapping.tp_size)
            assert (k_weight.shape[0] %
                    (mapping.tp_size * config.head_size)) == 0
            assert (v_weight.shape[0] %
                    (mapping.tp_size * config.head_size)) == 0

            wq = split(q_weight, mapping.tp_size, mapping.tp_rank)
            wk = split(k_weight, mapping.tp_size, mapping.tp_rank)
            wv = split(v_weight, mapping.tp_size, mapping.tp_rank)

            split_v = torch.concat((wq, wk, wv))

        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

            split_v = split_qkv_tp(qkv_weight, config.num_attention_heads,
                                   config.hidden_size, mapping.tp_size,
                                   mapping.tp_rank)

        if prefix + f'{param_name_map["attention.qkv"]}.q_proj.bias' in model_params:
            # only used in Internlm 7B models
            q_bias = get_bias(
                model_params,
                prefix + f'{param_name_map["attention.qkv"]}.q_proj', dtype)
            k_bias = get_bias(
                model_params,
                prefix + f'{param_name_map["attention.qkv"]}.k_proj', dtype)
            v_bias = get_bias(
                model_params,
                prefix + f'{param_name_map["attention.qkv"]}.v_proj', dtype)
            qkv_bias = torch.cat((q_bias, k_bias, v_bias))
            split_bias_v = split_qkv_bias_tp(qkv_bias,
                                             config.num_attention_heads,
                                             config.hidden_size,
                                             mapping.tp_size, mapping.tp_rank)
        else:
            split_bias_v = None

        if use_smooth_quant:
            qkv_weight = qkv_para[prefix +
                                  f'{param_name_map["attention.qkv"]}.qkv_proj']
            qkv_out_dim = qkv_weight.shape[1]

            if not mha_mode:
                local_dim = qkv_weight.shape[0]
                kv_hidden_size = (qkv_weight.shape[-1] - local_dim) // 2
                qkv_weight = qkv_weight.reshape(local_dim,
                                                local_dim + 2 * kv_hidden_size)
            else:
                qkv_weight = qkv_weight.reshape(config.hidden_size, 3,
                                                config.hidden_size)

            int8_weights = generate_int8(
                qkv_weight,
                act_range.get(prefix +
                              f'{param_name_map["attention.qkv"]}.qkv_proj'),
                is_qkv=True,
                multi_query_mode=bool(not mha_mode))

            weights.update(
                get_tllm_linear_sq_weight(int8_weights,
                                          tllm_prex + 'attention.qkv.',
                                          [1, qkv_out_dim // mapping.tp_size],
                                          mapping.tp_size,
                                          is_qkv=True,
                                          bias=split_bias_v,
                                          per_token=per_token,
                                          per_channel=per_channel,
                                          last_prefix=tllm_prex +
                                          'input_layernorm.scale_to_int',
                                          smoother_value=None,
                                          smoother_shape=None,
                                          rank=mapping.tp_rank,
                                          cat_dim=-1,
                                          multi_query_mode=bool(not mha_mode)))
        else:
            weights.update(
                get_tllm_linear_weight(split_v,
                                       tllm_prex + 'attention.qkv.',
                                       split_bias_v,
                                       use_weight_only,
                                       plugin_weight_only_quant_type,
                                       dtype,
                                       use_gemm_woq_plugin,
                                       use_fp8_rowwise=False))

        if int8_kv_cache:
            qkv_y = torch.cat([
                act_range.get(prefix +
                              f'{param_name_map["attention.qkv"]}.q_proj')["y"],
                act_range.get(prefix +
                              f'{param_name_map["attention.qkv"]}.k_proj')["y"],
                act_range.get(prefix +
                              f'{param_name_map["attention.qkv"]}.v_proj')["y"]
            ],
                              dim=0)

            int8_kv_scales = qkv_y.max() / 127.

            kv_cache_weights = {}

            kv_cache_weights[
                tllm_prex +
                'attention.kv_cache_scaling_factor'] = int8_kv_scales.reshape(
                    [1])

            weights.update(kv_cache_weights)
        elif fp8_kv_cache:
            # FIXME: set it to 1.0f for fp8 kv cache.
            weights[tllm_prex +
                    'attention.kv_cache_scaling_factor'] = torch.tensor(
                        [1.0], dtype=torch.float32)

        attn_dense_weight = get_weight(
            model_params, prefix + param_name_map["attention.dense"], dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  dim=1)

        if prefix + f'{param_name_map["attention.dense"]}.bias' in model_params:
            attn_dense_bias = get_bias(
                model_params, prefix + param_name_map["attention.dense"], dtype)
        else:
            attn_dense_bias = None
        if use_smooth_quant:
            attn_dense_weight = attn_dense_weight.t()
            proj_out_dim = attn_dense_weight.shape[0]

            int8_weights = generate_int8(
                attn_dense_weight,
                act_range.get(prefix + param_name_map["attention.dense"]))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'attention.dense.', [1, config.hidden_size],
                    mapping.tp_size,
                    is_qkv=False,
                    bias=attn_dense_bias,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex +
                    'attention.quantization_scaling_factor',
                    smoother_value=smoother[(
                        prefix + param_name_map["attention.dense"])],
                    smoother_shape=[1, proj_out_dim // mapping.tp_size],
                    rank=mapping.tp_rank,
                    cat_dim=0))
        else:
            weights.update(
                get_tllm_linear_weight(split_v,
                                       tllm_prex + 'attention.dense.',
                                       attn_dense_bias,
                                       use_weight_only,
                                       plugin_weight_only_quant_type,
                                       dtype,
                                       use_gemm_woq_plugin,
                                       use_fp8_rowwise=False))

        if moe_config.has_moe():
            rank_experts = list(range(moe_config.num_experts))
            if mapping.has_moe_ep():
                rank_experts = mapping.ep_experts(moe_config.num_experts)
            for suffix in ["w1", "w2", "w3"]:
                model_params[f'model.layers.{l}.block_sparse_moe.experts.{suffix}.weight'] = \
                            torch.stack([model_params[f'model.layers.{l}.block_sparse_moe.experts.{expert}.{suffix}.weight'].detach()
                                        for expert in rank_experts])
            w3 = model_params[
                f'model.layers.{l}.block_sparse_moe.experts.w3.weight']
            w2 = model_params[
                f'model.layers.{l}.block_sparse_moe.experts.w2.weight']
            w1 = model_params[
                f'model.layers.{l}.block_sparse_moe.experts.w1.weight']
            if mapping.has_moe_tp():
                w3 = split(w3, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)
                w2 = split(w2, mapping.moe_tp_size, mapping.moe_tp_rank, dim=2)
                w1 = split(w1, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)

            model_params[
                f'model.layers.{l}.block_sparse_moe.experts.w3w1.weight'] = torch.concat(
                    [w3, w1], dim=-2)

            model_params[
                f'model.layers.{l}.block_sparse_moe.experts.w2.weight'] = w2

            ## block_sparse_moe.experts.w2.weight
            moe_experts_w2_weights = get_weight(
                model_params, prefix + 'block_sparse_moe.experts.w2', dtype)
            weights.update(
                get_tllm_linear_weight(moe_experts_w2_weights,
                                       tllm_prex + 'mlp.proj.', None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))
            ##block_sparse_moe.experts.w3w1.weight
            moe_experts_w3w1_weights = get_weight(
                model_params, prefix + 'block_sparse_moe.experts.w3w1', dtype)
            weights.update(
                get_tllm_linear_weight(moe_experts_w3w1_weights,
                                       tllm_prex + 'mlp.fc.', None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

            if config.residual_mlp:
                residual_mlp_gate_weights = get_weight(
                    model_params, prefix + 'residual_mlp.w3', dtype)
                if use_smooth_quant:
                    residual_mlp_gate_weights = residual_mlp_gate_weights.t()
                    int8_weights = generate_int8(
                        residual_mlp_gate_weights,
                        act_range.get(prefix + 'residual_mlp.w3'))
                    weights.update(
                        get_tllm_linear_sq_weight(
                            int8_weights,
                            tllm_prex + 'residual_mlp.gate.',
                            [1, config.hidden_size // mapping.tp_size],
                            mapping.tp_size,
                            is_qkv=False,
                            per_token=per_token,
                            per_channel=per_channel,
                            last_prefix=tllm_prex +
                            'post_layernorm.scale_to_int',
                            smoother_value=None,
                            smoother_shape=None,
                            rank=mapping.tp_rank,
                            cat_dim=-1))
                else:
                    split_v = split_matrix_tp(residual_mlp_gate_weights,
                                              mapping.tp_size,
                                              mapping.tp_rank,
                                              dim=0)
                    weights.update(
                        get_tllm_linear_weight(split_v,
                                               tllm_prex + 'residual_mlp.gate.',
                                               None, use_weight_only,
                                               plugin_weight_only_quant_type,
                                               dtype, use_gemm_woq_plugin))

                residual_mlp_fc_weight = get_weight(model_params,
                                                    prefix + 'residual_mlp.w1',
                                                    dtype)
                if use_smooth_quant:
                    residual_mlp_fc_weight = residual_mlp_fc_weight.t(
                    )  #verified
                    int8_weights = generate_int8(
                        residual_mlp_fc_weight,
                        act_range.get(prefix + 'residual_mlp.w1'))
                    weights.update(
                        get_tllm_linear_sq_weight(
                            int8_weights,
                            tllm_prex + 'residual_mlp.fc.',
                            [1, config.hidden_size // mapping.tp_size],
                            mapping.tp_size,
                            is_qkv=False,
                            per_token=per_token,
                            per_channel=per_channel,
                            last_prefix=tllm_prex +
                            'post_layernorm.scale_to_int',
                            smoother_value=None,
                            smoother_shape=None,
                            rank=mapping.tp_rank,
                            cat_dim=-1))
                else:
                    split_v = split_matrix_tp(residual_mlp_fc_weight,
                                              mapping.tp_size,
                                              mapping.tp_rank,
                                              dim=0)
                    weights.update(
                        get_tllm_linear_weight(split_v,
                                               tllm_prex + 'residual_mlp.fc.',
                                               None, use_weight_only,
                                               plugin_weight_only_quant_type,
                                               dtype, use_gemm_woq_plugin))

                residual_mlp_proj_weight = get_weight(
                    model_params, prefix + 'residual_mlp.w2', dtype)

                if use_smooth_quant:
                    residual_mlp_proj_weight = residual_mlp_proj_weight.t()
                    int8_weights = generate_int8(
                        residual_mlp_proj_weight,
                        act_range.get(prefix + 'residual_mlp.w2'))
                    weights.update(
                        get_tllm_linear_sq_weight(
                            int8_weights,
                            tllm_prex + 'residual_mlp.proj.',
                            [1, config.hidden_size],
                            mapping.tp_size,
                            is_qkv=False,
                            per_token=per_token,
                            per_channel=per_channel,
                            last_prefix=tllm_prex +
                            'residual_mlp.quantization_scaling_factor',
                            smoother_value=smoother[prefix + 'residual_mlp.w2'],
                            smoother_shape=[
                                1, config.hidden_size // mapping.tp_size
                            ],
                            rank=mapping.tp_rank,
                            cat_dim=0))
                else:
                    split_v = split_matrix_tp(residual_mlp_proj_weight,
                                              mapping.tp_size,
                                              mapping.tp_rank,
                                              dim=1)
                    weights.update(
                        get_tllm_linear_weight(split_v,
                                               tllm_prex + 'residual_mlp.proj.',
                                               None, use_weight_only,
                                               plugin_weight_only_quant_type,
                                               dtype, use_gemm_woq_plugin))

            moe_experts_gate_weights = get_weight(
                model_params, prefix + 'block_sparse_moe.gate', torch.float32)
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
            mlp_gate_weight, mlp_gate_weight_scale = get_weight_and_scale(
                model_params, prefix + param_name_map["mlp.gate"], dtype,
                mapping, True)
            split_v = split_matrix_tp(mlp_gate_weight,
                                      mapping.tp_size,
                                      mapping.tp_rank,
                                      dim=0)
            if use_smooth_quant:

                mlp_gate_weight = mlp_gate_weight.t()
                int8_weights = generate_int8(
                    # mlp_gate_weight, act_range.get(prefix + 'mlp.up_proj'))
                    mlp_gate_weight,
                    act_range.get(prefix + param_name_map["mlp.gate"]))

                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.gate.',
                        [1, config.intermediate_size // mapping.tp_size],
                        mapping.tp_size,
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
                    get_tllm_linear_weight(
                        split_v,
                        tllm_prex + 'mlp.gate.',
                        None,
                        use_weight_only,
                        plugin_weight_only_quant_type,
                        dtype,
                        use_gemm_woq_plugin,
                        use_fp8_rowwise_in_layer,
                        weight_scale=mlp_gate_weight_scale,
                        clamp_value=config.quantization.clamp_val))

            mlp_fc_weight, mlp_fc_weight_scale = get_weight_and_scale(
                model_params, prefix + param_name_map["mlp.fc"], dtype, mapping,
                True)
            split_v = split_matrix_tp(mlp_fc_weight,
                                      mapping.tp_size,
                                      mapping.tp_rank,
                                      dim=0)

            if use_smooth_quant:
                mlp_fc_weight = mlp_fc_weight.t()  #verified
                int8_weights = generate_int8(
                    mlp_fc_weight,
                    act_range.get(prefix + param_name_map["mlp.fc"]))
                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.fc.',
                        [1, config.intermediate_size // mapping.tp_size],
                        mapping.tp_size,
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
                    get_tllm_linear_weight(
                        split_v,
                        tllm_prex + 'mlp.fc.',
                        None,
                        use_weight_only,
                        plugin_weight_only_quant_type,
                        dtype,
                        use_gemm_woq_plugin,
                        use_fp8_rowwise_in_layer,
                        weight_scale=mlp_fc_weight_scale,
                        clamp_value=config.quantization.clamp_val))

            mlp_proj_weight, mlp_proj_weight_scale = get_weight_and_scale(
                model_params, prefix + param_name_map["mlp.proj"], dtype)
            split_v = split_matrix_tp(mlp_proj_weight,
                                      mapping.tp_size,
                                      mapping.tp_rank,
                                      dim=1)

            if use_smooth_quant:
                mlp_proj_weight = mlp_proj_weight.t()
                int8_weights = generate_int8(
                    mlp_proj_weight,
                    act_range.get(prefix + param_name_map["mlp.proj"]))
                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.proj.', [1, config.hidden_size],
                        mapping.tp_size,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex +
                        'mlp.quantization_scaling_factor',
                        smoother_value=smoother[prefix +
                                                param_name_map["mlp.proj"]],
                        smoother_shape=[
                            1, config.intermediate_size // mapping.tp_size
                        ],
                        rank=mapping.tp_rank,
                        cat_dim=0))
            else:
                weights.update(
                    get_tllm_linear_weight(
                        split_v,
                        tllm_prex + 'mlp.proj.',
                        None,
                        use_weight_only,
                        plugin_weight_only_quant_type,
                        dtype,
                        use_gemm_woq_plugin,
                        use_fp8_rowwise_in_layer,
                        weight_scale=mlp_proj_weight_scale,
                        clamp_value=config.quantization.clamp_val))

        # Layer norms do not use tensor parallelism
        input_ln_weight = get_weight(model_params,
                                     prefix + param_name_map["input_layernorm"],
                                     dtype)
        weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight

        post_ln_weight = get_weight(model_params,
                                    prefix + param_name_map["post_layernorm"],
                                    dtype)
        weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight

        if config.residual_mlp:
            residual_ln_weight = get_weight(model_params,
                                            prefix + 'residual_layernorm',
                                            dtype)
            weights[tllm_prex +
                    'residual_layernorm.weight'] = residual_ln_weight

        cur_block_weights = [
            weight_name for weight_name in model_params
            if weight_name.find(prefix) != -1
        ]
        for weight_name in cur_block_weights:
            model_params[weight_name] = None

    for l in layers_range:
        convert_layer(l)
        release_gc()

    v = get_weight(model_params,
                   f'{model_prefix}.{param_name_map["vocab_embedding"]}', dtype)
    if hf_model.config.tie_word_embeddings:
        # lm_head.weight has the same weights as embedding
        if mapping.is_last_pp_rank():
            if config.vocab_size % mapping.tp_size != 0:
                # padding
                vocab_size_padded = pad_vocab_size(config.vocab_size,
                                                   mapping.tp_size)
                pad_width = vocab_size_padded - config.vocab_size

                v = torch.nn.functional.pad(v, (0, 0, 0, pad_width), 'constant',
                                            0)
            weights['lm_head.weight'] = split(v, mapping.tp_size,
                                              mapping.tp_rank)

    if config.use_parallel_embedding:
        v = split_matrix_tp(v,
                            mapping.tp_size,
                            mapping.tp_rank,
                            dim=config.embedding_sharding_dim)

    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v

    lm_head_weights = get_weight(model_params, param_name_map["lm_head"], dtype)

    if mapping.is_last_pp_rank():
        if config.vocab_size % mapping.tp_size != 0:
            # padding
            vocab_size_padded = pad_vocab_size(config.vocab_size,
                                               mapping.tp_size)
            pad_width = vocab_size_padded - config.vocab_size

            lm_head_weights = torch.nn.functional.pad(lm_head_weights,
                                                      (0, 0, 0, pad_width),
                                                      'constant',
                                                      value=0)
        weights['lm_head.weight'] = split_matrix_tp(lm_head_weights,
                                                    mapping.tp_size,
                                                    mapping.tp_rank,
                                                    dim=0)
        ln_f_w = get_weight(model_params,
                            f'{model_prefix}.{param_name_map["ln_f"]}', dtype)
        weights['transformer.ln_f.weight'] = ln_f_w

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def quantize(hf_model_dir: str,
             output_dir: str,
             config: LLaMAConfig,
             device: str = 'cuda',
             calib_dataset: str = 'cnn_dailymail'):
    '''
        Quantize the save the model as TRT-LLM checkpoint to output_dir
    '''
    os.makedirs(output_dir, exist_ok=True)
    config.to_json_file(os.path.join(output_dir, 'config.json'))

    mapping = config.mapping
    assert mapping.rank == 0, "quantize should be called at rank 0 only"

    quant_config = config.quantization
    use_smooth_quant = quant_config.use_plugin_sq
    int8_kv_cache = quant_config.kv_cache_quant_algo == QuantAlgo.INT8

    assert use_smooth_quant or int8_kv_cache, "Call from_hugging_face when there is no quantization"
    assert hf_model_dir is not None
    ## only load and call smooth quant routine once for all ranks
    hf_config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    assert "llava" not in hf_config.model_type, "Smooth quant llava/vila/llava_next is not supported yet"
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        device_map='auto' if device != 'cpu' else 'cpu',
        torch_dtype='auto' if not use_smooth_quant else torch.float16,
        trust_remote_code=True)

    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
        "TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir,
                                              trust_remote_code=True,
                                              use_fast=False,
                                              padding_side='left')

    dataset = load_calib_dataset(calib_dataset)

    act_range = capture_activation_range(hf_model, tokenizer, dataset)
    qkv_para, smoother = {}, {}
    if use_smooth_quant:
        smooth_llama_model(hf_model, act_range, quant_config.smoothquant_val,
                           qkv_para, smoother)

    for rank in range(mapping.world_size):
        # To avoid changing the mapping arg in-place, also the given mapping from caller is rank agnostic, since quantize is called from only one rank
        config = copy.deepcopy(config)
        config.set_rank(rank)
        weights = load_weights_from_hf_model(
            hf_model,
            config=config,
            act_range=act_range,
            qkv_para=qkv_para,
            smoother=smoother,
        )
        safetensors.torch.save_file(
            weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
        del weights


class QkvWeightHelper:
    """ A helper utility for loading QKV weights from sharded files. """

    def __init__(self, config: PretrainedConfig):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.tp_size = config.mapping.tp_size
        self.tp_rank = config.mapping.tp_rank
        self.is_mha = self.num_heads == self.num_kv_heads
        self.head_size = None if not hasattr(config,
                                             "head_size") else config.head_size
        self._qkv_weights = {}
        self.remove_duplicated_kv_heads = getattr(config,
                                                  'remove_duplicated_kv_heads',
                                                  False)

    @staticmethod
    def is_qkv_weight(name):
        for k in ['q_proj', 'k_proj', 'v_proj']:
            if 'self_attn' in name and k in name:
                return True
        return False

    def add_weight(self, i: int, name: str, weight: torch.Tensor):
        if 'q_proj' in name:
            tag = 'q'
        elif 'k_proj' in name:
            tag = 'k'
        elif 'v_proj' in name:
            tag = 'v'
        else:
            raise ValueError(f'Got an unexpected parameter of name {name}')
        if i not in self._qkv_weights:
            self._qkv_weights[i] = {}
        self._qkv_weights[i][tag] = weight

    def is_qkv_prepared(self, layer_idx):
        if layer_idx not in self._qkv_weights:
            return False
        weights = self._qkv_weights[layer_idx]
        return 'q' in weights and 'k' in weights and 'v' in weights

    def split_qkv_weights(self, layer_idx):
        if not self.is_qkv_prepared(layer_idx):
            return None
        weights = self._qkv_weights.pop(layer_idx)  # to prevent memory leak.
        q, k, v = (torch.tensor(weights[t]) for t in ['q', 'k', 'v'])

        if self.remove_duplicated_kv_heads:
            head_size = self.hidden_size // self.num_heads if self.head_size is None else self.head_size
            k = k.reshape(
                [k.shape[0] // head_size // 2, 2, head_size, self.hidden_size])
            v = v.reshape(
                [v.shape[0] // head_size // 2, 2, head_size, self.hidden_size])
            assert (k[:, 0] == k[:, 1]).all()
            assert (v[:, 0] == v[:, 1]).all()
            k = k[:, 0].reshape([-1, self.hidden_size])
            v = v[:, 0].reshape([-1, self.hidden_size])

        if not self.is_mha:
            head_size = self.hidden_size // self.num_heads if self.head_size is None else self.head_size
            if self.num_kv_heads < self.tp_size:
                # duplicate the KV heads up to tensor_parallel
                k = dup_kv_weight(k, self.num_kv_heads, self.tp_size)
                v = dup_kv_weight(v, self.num_kv_heads, self.tp_size)
            assert k.shape[0] % (self.tp_size * head_size) == 0
            assert v.shape[0] % (self.tp_size * head_size) == 0
            wq = split(q, self.tp_size, self.tp_rank)
            wk = split(k, self.tp_size, self.tp_rank)
            wv = split(v, self.tp_size, self.tp_rank)
            fused_qkv = torch.cat((wq, wk, wv), dim=0)
        else:
            qkv = torch.cat([q, k, v], dim=0)
            qkv = qkv.reshape(3, q.shape[0], q.shape[1])
            fused_qkv = split(qkv, self.tp_size, self.tp_rank, dim=1)
            fused_qkv = fused_qkv.reshape(3 * (q.shape[0] // self.tp_size),
                                          q.shape[1])
        return fused_qkv


def load_weights_from_hf_by_shard(model_dir: str, config: LLaMAConfig):
    '''Weights-only quantization is the only supported quantization recipe here.'''
    logger.info('Loading weights from HF LLaMA...')
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
    dtype = getattr(torch, config.dtype)

    mapping = config.mapping
    moe_config = config.moe
    assert not moe_config.has_moe(), "MoE does not support sharded load"
    assert "Exaone" not in config.architecture, "EXAONE model currently not support sharded load"

    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_dir)

    quant_mode = config.quant_mode
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    elif config.quant_mode.has_fp8_rowwise():
        plugin_weight_only_quant_type = torch.float8_e4m3fn
    else:
        plugin_weight_only_quant_type = None
    use_weight_only = quant_mode.is_weight_only()
    use_fp8_rowwise = quant_mode.has_fp8_rowwise()
    # Meta's recipe of not using fp8 rowwise for the first and last layer.
    exclude_layers_id = [0, config.num_hidden_layers - 1]

    layers_range = mapping.pp_layers(config.num_hidden_layers)

    qkv_weight_helper = QkvWeightHelper(config)

    def convert_to_dtype(name, param, model_params, dtype):
        # fp8 rowwise weights will only load fp8 weights and scales for the mlp layer.
        if ('weight_scale' in name or name.replace('weight', 'weight_scale') in model_params) \
           and use_fp8_rowwise:
            assert 'mlp' in name, "only MLP layers support fp8 rowwise currently."
            return param
        else:
            return param.to(dtype)

    def fp8_rowwise_quantization(name,
                                 param,
                                 model_params,
                                 clamp_value,
                                 split_scale=False):
        # check if weights are already quantized.
        loaded_weight_scale = model_params.get(
            name.replace('weight', 'weight_scale'))
        if loaded_weight_scale is not None:
            assert param.dtype == torch.float8_e4m3fn, "weight data type must be torch.float8_e4m3fn"
            if split_scale:
                assert mapping is not None
                loaded_weight_scale = split(loaded_weight_scale,
                                            mapping.tp_size,
                                            mapping.tp_rank,
                                            dim=0)

            return param, loaded_weight_scale.reshape(-1)
        else:
            return fp8_per_channel_quant_weight_gpu(param, clamp_value)

    for model_file in iterate_shard_files(model_dir,
                                          rank=mapping.tp_rank,
                                          progress_bar=False):
        logger.debug(f'Loading file {str(model_file)}...')
        model_params = load_state_dict(model_file)
        for name, param in model_params.items():
            logger.debug(f'Converting weight {name}...')
            layer_idx = retrieved_layer_index_from_name(name)
            tllm_prex = f'transformer.layers.{layer_idx}.'

            param = convert_to_dtype(name, param, model_params, dtype)

            if layer_idx is None:
                layer = None
            else:
                if layer_idx not in layers_range:
                    continue
                else:
                    tllm_prex = f'transformer.layers.{layer_idx - layers_range[0]}.'

            # Meta's recipe of not using fp8 rowwise for the first and last layer.
            use_fp8_rowwise_in_layer = use_fp8_rowwise and (
                layer_idx not in exclude_layers_id)

            if 'model.embed_tokens.weight' in name:
                if hf_config.tie_word_embeddings:
                    # lm_head.weight has the same weights as embedding
                    if mapping.is_last_pp_rank():

                        if config.vocab_size % mapping.tp_size != 0:
                            # padding
                            vocab_size_padded = pad_vocab_size(
                                config.vocab_size, mapping.tp_size)
                            pad_width = vocab_size_padded - config.vocab_size
                            param = torch.from_numpy(
                                np.pad(param.detach().cpu().numpy(),
                                       ((0, pad_width), (0, 0)),
                                       'constant',
                                       constant_values=0))
                        weights['lm_head.weight'] = split(
                            param, mapping.tp_size, mapping.tp_rank)
                if config.use_parallel_embedding:
                    param = split(param, mapping.tp_size, mapping.tp_rank,
                                  config.embedding_sharding_dim)
                if mapping.is_first_pp_rank():
                    weights['transformer.vocab_embedding.weight'] = param
            elif 'model.norm.weight' in name:
                if mapping.is_last_pp_rank():
                    weights['transformer.ln_f.weight'] = param
            elif 'lm_head.weight' in name:
                if mapping.is_last_pp_rank():
                    if config.vocab_size % mapping.tp_size != 0:
                        # padding
                        vocab_size_padded = pad_vocab_size(
                            config.vocab_size, mapping.tp_size)
                        pad_width = vocab_size_padded - config.vocab_size
                        param = torch.from_numpy(
                            np.pad(param.detach().cpu().numpy(),
                                   ((0, pad_width), (0, 0)),
                                   'constant',
                                   constant_values=0))
                    weights['lm_head.weight'] = split(param, mapping.tp_size,
                                                      mapping.tp_rank)
            elif 'input_layernorm.weight' in name:
                weights[tllm_prex + 'input_layernorm.weight'] = param
            elif 'post_attention_layernorm.weight' in name:
                weights[tllm_prex + 'post_layernorm.weight'] = param
            elif qkv_weight_helper.is_qkv_weight(name):
                qkv_weight_helper.add_weight(layer_idx, name, param)
                if not qkv_weight_helper.is_qkv_prepared(layer_idx):
                    continue
                split_v = qkv_weight_helper.split_qkv_weights(layer_idx)
                if use_weight_only:
                    param = split_v.transpose()
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            param, plugin_weight_only_quant_type)
                    weights[tllm_prex +
                            'attention.qkv.weight'] = processed_torch_weights
                    weights[
                        tllm_prex +
                        'attention.qkv.per_channel_scale'] = torch_weight_scales
                else:
                    weights[tllm_prex + 'attention.qkv.weight'] = split_v
            elif 'self_attn.o_proj.weight' in name:
                split_v = split(param, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            split_v.transpose(), plugin_weight_only_quant_type)
                    weights[tllm_prex +
                            'attention.dense.weight'] = processed_torch_weights
                    weights[
                        tllm_prex +
                        'attention.dense.per_channel_scale'] = torch_weight_scales
                else:
                    weights[tllm_prex + 'attention.dense.weight'] = split_v
            elif name.endswith('mlp.up_proj.weight'):
                split_v = split(param, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            split_v.transpose(), plugin_weight_only_quant_type)
                    weights[tllm_prex +
                            'mlp.gate.weight'] = processed_torch_weights
                    weights[tllm_prex +
                            'mlp.gate.per_channel_scale'] = torch_weight_scales
                elif use_fp8_rowwise_in_layer:
                    processed_torch_weights, torch_weight_scales = fp8_rowwise_quantization(
                        name, split_v, model_params,
                        config.quantization.clamp_val, True)
                    weights[tllm_prex +
                            'mlp.gate.weight'] = processed_torch_weights.view(
                                plugin_weight_only_quant_type)
                    weights[
                        tllm_prex +
                        'mlp.gate.per_channel_scale'] = torch_weight_scales.to(
                            torch.float32)
                else:
                    weights[tllm_prex + 'mlp.gate.weight'] = split_v
            elif name.endswith('mlp.down_proj.weight'):
                split_v = split(param, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            split_v.transpose(), plugin_weight_only_quant_type)
                    weights[tllm_prex +
                            'mlp.proj.weight'] = processed_torch_weights
                    weights[tllm_prex +
                            'mlp.proj.per_channel_scale'] = torch_weight_scales
                elif use_fp8_rowwise_in_layer:
                    processed_torch_weights, torch_weight_scales = fp8_rowwise_quantization(
                        name, split_v, model_params,
                        config.quantization.clamp_val)
                    weights[tllm_prex +
                            'mlp.proj.weight'] = processed_torch_weights.view(
                                plugin_weight_only_quant_type)
                    weights[
                        tllm_prex +
                        'mlp.proj.per_channel_scale'] = torch_weight_scales.to(
                            torch.float32)
                else:
                    weights[tllm_prex + 'mlp.proj.weight'] = split_v

            elif name.endswith('mlp.gate_proj.weight'):
                split_v = split(param, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            split_v.transpose(), plugin_weight_only_quant_type)
                    layer.mlp.fc.weight.value = processed_torch_weights
                    layer.mlp.fc.per_channel_scale.value = torch_weight_scales
                    weights[tllm_prex +
                            'mlp.fc.weight'] = processed_torch_weights
                    weights[tllm_prex +
                            'mlp.fc.per_channel_scale'] = torch_weight_scales
                elif use_fp8_rowwise_in_layer:
                    processed_torch_weights, torch_weight_scales = fp8_rowwise_quantization(
                        name, split_v, model_params,
                        config.quantization.clamp_val, True)
                    weights[tllm_prex +
                            'mlp.fc.weight'] = processed_torch_weights.view(
                                plugin_weight_only_quant_type)
                    weights[
                        tllm_prex +
                        'mlp.fc.per_channel_scale'] = torch_weight_scales.to(
                            torch.float32)
                else:
                    weights[tllm_prex + 'mlp.fc.weight'] = split_v

        del model_params

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')
    return weights


def load_weights_from_hf_safetensors(model_dir: str, config: LLaMAConfig):
    logger.info('Loading weights from Huggingface {} safetensors...'.format(
        config.architecture.split('ForCausalLM')[0]))
    tik = time.time()
    import json
    import os

    import safetensors
    weights = {}

    model_dir = model_dir if model_dir.endswith("/") else model_dir + "/"
    safetensors_map = {}
    has_safetensor_index_json = True
    try:
        with open(model_dir + "model.safetensors.index.json", 'r') as fr:
            sharding_map = json.load(fr)
        for k, v in sharding_map['weight_map'].items():
            safetensors_map[k] = int(v[6:11]) - 1
    except FileNotFoundError:
        has_safetensor_index_json = False

    shard_files = []
    for name in os.listdir(model_dir):
        if name.endswith(".safetensors"):
            if has_safetensor_index_json and name not in sharding_map[
                    'weight_map'].values():
                continue
            shard_files.append(name)
    shard_files.sort()
    safetensors_ptrs = [
        safetensors.safe_open(model_dir + shard_file,
                              framework="pt",
                              device="cpu") for shard_file in shard_files
    ]

    mapping = config.mapping
    num_hidden_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    pad_vocab = vocab_size % mapping.tp_size != 0
    vocab_size_padded = pad_vocab_size(config.vocab_size, mapping.tp_size)
    dtype = config.dtype

    moe_config = config.moe

    kv_tp_size = None
    kv_tp_rank = None
    if config.num_key_value_heads < mapping.tp_size:
        kv_tp_size = config.num_key_value_heads
        kv_tp_rank = mapping.tp_rank * kv_tp_size // mapping.tp_size

    model_prefix, layer_prefix, param_name_map = get_prefix_and_param_name_map(
        config.architecture, use_safetensors=True)

    torch_dtype = str_dtype_to_torch(dtype)

    def load(key,
             tp_dim=-1,
             no_prefix=0,
             is_expert_weights=False,
             tp_size=None,
             tp_rank=None):
        if not no_prefix:
            key = f'{model_prefix}.' + key
        ptr_idx = safetensors_map[key] if key in safetensors_map else 0

        if key not in safetensors_ptrs[ptr_idx].keys():
            return None

        tensor_slice = safetensors_ptrs[ptr_idx].get_slice(key)
        tensor_shape = tensor_slice.get_shape()
        if tp_dim == -1:
            res = tensor_slice[:]
        elif tp_dim >= 0 and tp_dim < len(tensor_shape):
            if is_expert_weights:
                tp_size = tp_size or mapping.moe_tp_size
                tp_rank = tp_rank or mapping.moe_tp_rank
            else:
                tp_size = tp_size or mapping.tp_size
                tp_rank = tp_rank or mapping.tp_rank
            dim_size = tensor_shape[tp_dim]
            if dim_size % tp_size != 0:
                logger.error(
                    f"Current weight {key}'s shape {tensor_shape} is invalid at dimension {tp_dim} for TP size {tp_size}"
                )
            indices = [slice(None)] * len(tensor_shape)
            indices[tp_dim] = slice(dim_size * tp_rank // tp_size,
                                    dim_size * (tp_rank + 1) // tp_size)
            res = tensor_slice[indices]
        else:
            raise ValueError(
                f"Invalid TP dim {tp_dim} for weight {key}'s shape {tensor_shape}"
            )
        return res.to(torch_dtype).contiguous(
        ) if "block_sparse_moe.gate" not in key else res.to(torch.float32)

    def load_and_set(target,
                     key,
                     tp_dim=-1,
                     no_prefix=0,
                     is_expert_weights=False):
        res = load(key, tp_dim, no_prefix, is_expert_weights)
        weights[target] = res
        if "weight" in key:
            bias = load(key.replace("weight", "bias"), -1, no_prefix,
                        is_expert_weights)
            if bias is not None:
                weights[target.replace("weight", "bias")] = bias

    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = load(
            param_name_map["vocab_embedding"], config.embedding_sharding_dim
            if config.use_parallel_embedding else -1)  # vocab_embedding

    if mapping.is_last_pp_rank():
        v = load(param_name_map["lm_head"], -1, 1) if pad_vocab else load(
            param_name_map["lm_head"], 0, 1)  # lm_head
        if pad_vocab:
            v = torch.nn.functional.pad(
                v, (0, 0, 0, vocab_size_padded - vocab_size), 'constant', 0)
            v = split(v, mapping.tp_size, mapping.tp_rank)
        weights['lm_head.weight'] = v
        weights['transformer.ln_f.weight'] = load(
            param_name_map["ln_f"])  # ln_f

    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        layer_idx = l - layers_range[0]
        prefix = f'{layer_prefix}.{l}.'
        tllm_prex = f'transformer.layers.{layer_idx}'

        # Attention
        qkv_list = []
        for comp in ["q", "k", "v"]:
            tp_size = kv_tp_size if comp != "q" else None
            tp_rank = kv_tp_rank if comp != "q" else None
            weight_part = load(prefix + f'{param_name_map["attention.qkv"]}.' +
                               comp + param_name_map["qkv_suffix"],
                               0,
                               tp_size=tp_size,
                               tp_rank=tp_rank)
            qkv_list.append(weight_part)
            bias_part = load(
                (prefix + f'{param_name_map["attention.qkv"]}.' + comp +
                 param_name_map["qkv_suffix"]).replace("weight", "bias"),
                0,
                tp_size=tp_size,
                tp_rank=tp_rank)
            if bias_part is not None:
                qkv_list.append(bias_part)
        if len(qkv_list) == 3:
            # No bias
            weights[f'{tllm_prex}.attention.qkv.weight'] = torch.cat(
                qkv_list, 0)
        else:
            weights[f'{tllm_prex}.attention.qkv.weight'] = torch.cat(
                qkv_list[::2], 0)
            weights[f'{tllm_prex}.attention.qkv.bias'] = torch.cat(
                qkv_list[1::2], 0)
        load_and_set(f'{tllm_prex}.attention.dense.weight',
                     prefix + param_name_map["attention.dense"],
                     1)  # attention.dense

        # MLP
        if not moe_config.has_moe():
            load_and_set(f'{tllm_prex}.mlp.gate.weight',
                         prefix + param_name_map["mlp.gate"], 0)  # mlp.gate
            load_and_set(f'{tllm_prex}.mlp.proj.weight',
                         prefix + param_name_map["mlp.proj"], 1)  # mlp.proj
            load_and_set(f'{tllm_prex}.mlp.fc.weight',
                         prefix + param_name_map["mlp.fc"], 0)  # mlp.fc

        else:
            weights[f'{tllm_prex}.mlp.router.weight'] = load(
                prefix + 'block_sparse_moe.gate.weight')
            rank_experts = list(range(moe_config.num_experts))
            if mapping.has_moe_ep():
                rank_experts = mapping.ep_experts(moe_config.num_experts)

            expert_weight_list = []
            for suffix in range(3):
                tp_dim = -1
                if mapping.has_moe_tp():
                    tp_dim = 1 if suffix == 1 else 0
                expert_weight_list.append(
                    torch.stack(
                        list(
                            load(
                                prefix +
                                f'block_sparse_moe.experts.{expert}.w{suffix + 1}.weight',
                                tp_dim=tp_dim,
                                is_expert_weights=True)
                            for expert in rank_experts)))

            w1 = expert_weight_list[0]
            w2 = expert_weight_list[1]
            w3 = expert_weight_list[2]

            weights[f'{tllm_prex}.mlp.fc.weight'] = \
                torch.concat([w3, w1], dim=-2).contiguous()
            weights[f'{tllm_prex}.mlp.proj.weight'] = w2.contiguous()

        load_and_set(f'{tllm_prex}.input_layernorm.weight', prefix +
                     param_name_map["input_layernorm"])  # input_layernorm
        load_and_set(f'{tllm_prex}.post_layernorm.weight', prefix +
                     param_name_map["post_layernorm"])  # post_layernorm

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')

    return weights


def load_weights_from_gptq(quant_ckpt_path: str, config: LLaMAConfig):
    logger.info('Loading weights from groupwise GPTQ LLaMA safetensors...')
    weights = {}
    tik = time.time()

    num_hidden_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    dtype = config.dtype
    mapping = config.mapping

    gptq_llama = safetensors.safe_open(quant_ckpt_path,
                                       framework="pt",
                                       device=0)
    gptq_prefix = "model."
    gptq_suffix_list = [".qweight", ".qzeros", ".scales"]
    gptq_key_list = [
        "embed_tokens.weight",  # vocab_embedding
        "lm_head.weight",  # lm_head
        "norm.weight",  # ln_f
        "self_attn.",  # attention.qkv
        "_proj",  # qkv suffix
        "self_attn.o_proj",  # attention.dense
        "mlp.up_proj",  # mlp.gate
        "mlp.down_proj",  # mlp.proj
        "mlp.gate_proj",  # mlp.fc
        "input_layernorm.weight",  # input_layernorm
        "post_attention_layernorm.weight",  # post_layernorm
    ]
    split_sym = "."

    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    def load(key, no_prefix=0):
        if no_prefix:
            return gptq_llama.get_tensor(key)
        else:
            return gptq_llama.get_tensor(gptq_prefix + key)

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def unpack_int32_into_int8(w_packed):
        # Unpack inputs packed in int32/float32 into uint4 and store them in int8 format
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
        USE_GPTQ_FOR_LLAMA = 1  # GPTQ-for-LLaMA added 1 to zeros

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
                               USE_GPTQ_FOR_LLAMA) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        results = {
            f'{tllm_prex}.weight': qweight_interleaved,
            f'{tllm_prex}.weights_scaling_factor': scales_fp16,
            f'{tllm_prex}.zero': zeros_x_scales_fp16,
        }
        return results

    # Load weights from GPTQ checkpoint into TRT-LLM module
    # 1. vocab_embedding
    v = load(gptq_key_list[0])
    if mapping.is_first_pp_rank():
        # tensorrt_llm_llama.vocab_embedding.weight.value = v.to(
        #     torch_dtype).cpu().numpy()
        weights['transformer.vocab_embedding.weight'] = v.to(torch_dtype)
    # 2. lm_head
    v = load(gptq_key_list[1], "no_prefix")
    if mapping.is_last_pp_rank():
        # tensorrt_llm_llama.lm_head.weight.value = torch_split(
        #     v, 0).to(torch_dtype).cpu().numpy()
        if vocab_size % mapping.tp_size != 0:
            # padding
            vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
            pad_width = vocab_size_padded - vocab_size
            v = torch.from_numpy(
                np.pad(v.detach().cpu().numpy(), ((0, pad_width), (0, 0)),
                       'constant',
                       constant_values=0))
        weights['lm_head.weight'] = torch_split(v, 0).to(torch_dtype)

    # 3. ln_f
    v = load(gptq_key_list[2])
    if mapping.is_last_pp_rank():
        # tensorrt_llm_llama.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()
        weights['transformer.ln_f.weight'] = v.to(torch_dtype)
    # 4. Weights inside each layer
    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        layer_idx = l - layers_range[0]
        prefix = "layers" + split_sym + str(layer_idx) + split_sym
        logger.info(f'Process weights in layer: {layer_idx}')
        # layer = tensorrt_llm_llama.layers[layer_idx]
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'
        # 4.1 attention.qkv
        qkv_weight_list = []
        for suf in gptq_suffix_list:
            qkv_list = []
            for comp in ["q", "k", "v"]:
                comp_part = load(prefix + gptq_key_list[3] + comp +
                                 gptq_key_list[4] + suf)
                comp_part = torch_split(comp_part, 1)
                qkv_list.append(comp_part)
            qkv_weight_list.append(torch.cat(qkv_list, dim=1))

        # process_and_assign_weight(layer.attention.qkv, qkv_weight_list)
        weights.update(
            process_and_assign_weight(qkv_weight_list,
                                      f'{tllm_prex}.attention.qkv'))
        # 4.2 attention.dense
        v = [load(prefix + gptq_key_list[5] + suf) for suf in gptq_suffix_list]
        # process_and_assign_weight(layer.attention.dense, v, 0)
        weights.update(
            process_and_assign_weight(v,
                                      f'{tllm_prex}.attention.dense',
                                      tp_dim=0))
        # 4.3 mlp.gate
        v = [load(prefix + gptq_key_list[6] + suf) for suf in gptq_suffix_list]
        # process_and_assign_weight(layer.mlp.gate, v, 1)
        weights.update(
            process_and_assign_weight(v, f'{tllm_prex}.mlp.gate', tp_dim=1))
        # 4.4 mlp.proj
        v = [load(prefix + gptq_key_list[7] + suf) for suf in gptq_suffix_list]
        # process_and_assign_weight(layer.mlp.proj, v, 0)
        weights.update(
            process_and_assign_weight(v, f'{tllm_prex}.mlp.proj', tp_dim=0))
        # 4.5 mlp.fc
        v = [load(prefix + gptq_key_list[8] + suf) for suf in gptq_suffix_list]
        # process_and_assign_weight(layer.mlp.fc, v, 1)
        weights.update(
            process_and_assign_weight(v, f'{tllm_prex}.mlp.fc', tp_dim=1))
        # 4.6 input_layernorm
        v = load(prefix + gptq_key_list[9])
        # layer.input_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()
        weights[f'{tllm_prex}.input_layernorm.weight'] = v.to(torch_dtype)

        # 4.7 post_layernorm
        v = load(prefix + gptq_key_list[10])
        # layer.post_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()
        weights[f'{tllm_prex}.post_layernorm.weight'] = v.to(torch_dtype)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')

    return weights


def load_torch_meta_ckpt(meta_ckpt_path: Path):
    '''
        meta_ckpt_path: The format of meta_ckpt_path is like <xxx>/consolidated.xx There are two possible cases:
            1. A file like <xxx>/consolidated.xx.pth, loading it by torch.load directly
            2. A folder like <xxx>/consolidated.xx/, need to load all weights in the folder.
    '''
    file_path = meta_ckpt_path.parent / (meta_ckpt_path.name + ".pth")
    if file_path.exists() and file_path.is_file():
        return torch.load(file_path, map_location="cpu")
    else:
        folder_path = meta_ckpt_path
        assert folder_path.exists() and folder_path.is_dir()

        ckpts = list(Path(folder_path).glob("consolidated-*.pth"))

        all_weights = {}
        for ckpt in ckpts:
            _weight = torch.load(ckpt, map_location="cpu")
            all_weights = all_weights | _weight
            del _weight
        return all_weights


def load_weights_from_meta_ckpt(meta_ckpt_dir: str, config: LLaMAConfig):
    torch_dtype = str_dtype_to_torch(config.dtype)
    mapping = config.mapping
    use_fp8_rowwise = config.quant_mode.has_fp8_rowwise()
    if config.quant_mode.has_any_quant() and not use_fp8_rowwise:
        logger.error(
            "Meta ckpts only support fp8_rowwise quantization currently.")
    weights = {}
    # Meta's recipe of not using fp8 rowwise for the first and last layer.
    exclude_layers_id = [0, config.num_hidden_layers - 1]

    def gather_ckpts(ckpts):
        gathered = {}
        for k in ckpts[0]:
            d = 0
            # TODO(bhsueh) not sure should we consider tok here.
            if any([n in k for n in ["wo", "w2"]]):
                d = 1
            if "norm" in k or "rope" in k:  # no TP
                gathered[k] = ckpts[0][k].clone()
            else:
                gathered[k] = torch.cat([pt[k] for pt in ckpts], dim=d).clone()
        return gathered

    def split_ckpt(ckpt, ranks_per_ckpt, ckpt_rank):
        split_ckpt = {}
        for k, v in ckpt.items():
            d = 0
            if any(n in k for n in
                   ["wo", "feed_forward.w2", "tok", "feed_forward.gate"]):
                d = 1
            if "norm" in k or "rope" in k:  # no TP
                split_ckpt[k] = v.clone()
            elif config.num_key_value_heads < mapping.tp_size and any(
                    n in k for n in ["wk", "wv"]):
                assert mapping.tp_size % config.num_key_value_heads == 0
                # special case: we need to duplicate KV head
                tmp = dup_kv_weight(v, config.num_key_value_heads,
                                    mapping.tp_size)
                split_ckpt[k] = torch.split(tmp,
                                            tmp.shape[d] // ranks_per_ckpt,
                                            dim=d)[ckpt_rank].clone()
            else:
                split_ckpt[k] = torch.split(v,
                                            v.shape[d] // ranks_per_ckpt,
                                            dim=d)[ckpt_rank].clone()
        return split_ckpt

    def get_current_weights(num_ckpts):
        if num_ckpts > mapping.tp_size:
            # combine ckpts
            assert (num_ckpts % mapping.tp_size) == 0
            nf = num_ckpts // mapping.tp_size
            fs = nf * mapping.tp_rank
            file_ids = list(range(fs, fs + nf))
            ckpts = []
            for f in file_ids:
                ckpt = load_torch_meta_ckpt(
                    Path(meta_ckpt_dir, f"consolidated.{f:02d}"))
                ckpts.append(ckpt)
            return gather_ckpts(ckpts)
        elif num_ckpts < mapping.tp_size:
            # split ckpt
            assert (mapping.tp_size % num_ckpts) == 0
            ranks_per_ckpt = mapping.tp_size // num_ckpts
            ckpt_fid = mapping.tp_rank // ranks_per_ckpt
            ckpt_rank = mapping.tp_rank % ranks_per_ckpt
            nH_per_ckpt = config.num_attention_heads // num_ckpts
            assert (nH_per_ckpt % ranks_per_ckpt) == 0
            ckpt = load_torch_meta_ckpt(
                Path(meta_ckpt_dir, f"consolidated.{ckpt_fid:02d}"))
            return split_ckpt(ckpt, ranks_per_ckpt, ckpt_rank)

        # num_ckpts == tensor_parallel, 1:1 mapping from files to TP
        return load_torch_meta_ckpt(
            Path(meta_ckpt_dir, f"consolidated.{mapping.tp_rank:02d}"))

    def permute(w, nH, d, dH):
        # due to MQA's wk, nH*dH != d could be true
        return w.view(nH, dH // 2, 2, d).transpose(1, 2).reshape(nH * dH, d)

    def extract_layer_idx(name):
        ss = name.split('.')
        for s in ss:
            if s.isdigit():
                return s
        return None

    if not hasattr(load_weights_from_meta_ckpt, "saved_embed"):
        load_weights_from_meta_ckpt.saved_embed = None

    def combine_embeddings(embeds, num_ckpts):
        if len(embeds) == 1:
            return embeds[0]
        assert [
            embeds[i].shape == embeds[i + 1].shape
            for i in range(len(embeds) - 1)
        ]
        if embeds[0].shape[0] == config.vocab_size // num_ckpts:
            merge_dim = 0
        elif embeds[0].shape[1] == config.hidden_size // num_ckpts:
            merge_dim = 1
        else:
            logger.error("Unable to infer embedding split dimension")
            assert False, "Unable to infer embedding split dimension"
        return torch.cat(embeds, dim=merge_dim)

    def gather_embedding(cur_embed, name: str, num_ckpts):
        if mapping.tp_size == 1:
            # even if num_ckpts > 1, get_current_weights will already have it gathered
            return cur_embed
        if load_weights_from_meta_ckpt.saved_embed is None:
            embeds = [None] * num_ckpts
            for i in range(num_ckpts):
                ckpt = load_torch_meta_ckpt(
                    Path(meta_ckpt_dir, f"consolidated.{i:02d}"))
                embeds[i] = ckpt[name]
            embed = combine_embeddings(embeds, num_ckpts).to(torch_dtype)
            load_weights_from_meta_ckpt.saved_embed = embed

        return load_weights_from_meta_ckpt.saved_embed

    logger.info('Loading weights from Meta LLaMA checkpoints ...')
    tik = time.time()

    num_kv_heads = config.num_key_value_heads
    mha_mode = (num_kv_heads == config.num_attention_heads)

    ckpts = list(Path(meta_ckpt_dir).glob("consolidated.*"))
    num_ckpts = len(ckpts)
    # llama/llama2 doesn't have MQA. So, simplifying loader logic by not worrying about it.
    assert num_kv_heads > 1 or num_kv_heads >= num_ckpts, \
        f"We don't know how the {num_kv_heads} KV heads are distributed among {num_ckpts} checkpoints."

    tik = time.time()
    ckpt = get_current_weights(num_ckpts)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'[{mapping.rank}] get_current_weights. Total time: {t}')

    head_size = config.hidden_size // config.num_attention_heads
    layers_range = mapping.pp_layers(config.num_hidden_layers)

    for l in layers_range:
        prefix = f'layers.{l}.attention.'
        q_weight = permute(ckpt[prefix + 'wq.weight'].clone(),
                           nH=(config.num_attention_heads // mapping.tp_size),
                           d=config.hidden_size,
                           dH=head_size)
        if num_kv_heads < mapping.tp_size and num_ckpts >= mapping.tp_size:
            assert mapping.tp_size % num_kv_heads == 0
            assert False, "Not supported yet"
        k_weight = permute(ckpt[prefix + 'wk.weight'].clone(),
                           nH=((num_kv_heads + mapping.tp_size - 1) //
                               mapping.tp_size),
                           d=config.hidden_size,
                           dH=head_size)
        v_weight = ckpt[prefix + 'wv.weight'].clone()

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        ckpt[prefix + 'qkv.weight'] = qkv_weight

    for k, v in tqdm(ckpt.items()):
        dtype = torch_dtype if 'feed_forward.gate' not in k else torch.float32

        v = v.to(dtype)
        if "tok_embeddings" in k:
            if not config.use_parallel_embedding:
                v = gather_embedding(v, k, num_ckpts)
            elif config.embedding_sharding_dim == 0:
                # this needs a gather and then resplit along different dims
                v = gather_embedding(v, k, num_ckpts)
                v = split(v, mapping.tp_size, mapping.tp_rank, 0)
            if mapping.is_first_pp_rank():
                weights['transformer.vocab_embedding.weight'] = v
        elif "output" in k:
            if mapping.is_last_pp_rank():
                if config.vocab_size % mapping.tp_size != 0:
                    # padding
                    vocab_size_padded = pad_vocab_size(config.vocab_size,
                                                       mapping.tp_size)
                    pad_width = vocab_size_padded - config.vocab_size
                    v = torch.from_numpy(
                        np.pad(v.detach().cpu().numpy(),
                               ((0, pad_width), (0, 0)),
                               'constant',
                               constant_values=0))
                weights['lm_head.weight'] = v
        elif k == "norm.weight":
            if mapping.is_last_pp_rank():
                weights['transformer.ln_f.weight'] = v
        else:
            # layer specific weights
            layer_idx = extract_layer_idx(k)
            if layer_idx is None or int(layer_idx) not in layers_range:
                continue

            # Meta's recipe of not using fp8 rowwise for the first and last layer.
            use_fp8_rowwise_in_layer = use_fp8_rowwise and (
                int(layer_idx) not in exclude_layers_id)
            idx = int(layer_idx) - layers_range[0]
            tllm_prex = f'transformer.layers.{idx}.'

            if 'attention_norm.weight' in k:
                weights[tllm_prex + 'input_layernorm.weight'] = v
            elif 'ffn_norm.weight' in k:
                weights[tllm_prex + 'post_layernorm.weight'] = v
            elif 'feed_forward.w3.weight' in k:
                if use_fp8_rowwise_in_layer:
                    processed_torch_weights, torch_weight_scales = fp8_per_channel_quant_weight_gpu(
                        v, config.quantization.clamp_val)
                    weights[tllm_prex +
                            'mlp.gate.weight'] = processed_torch_weights
                    weights[tllm_prex +
                            'mlp.gate.per_channel_scale'] = torch_weight_scales
                else:
                    weights[tllm_prex + 'mlp.gate.weight'] = v
            elif 'feed_forward.w2.weight' in k:
                if use_fp8_rowwise_in_layer:
                    processed_torch_weights, torch_weight_scales = fp8_per_channel_quant_weight_gpu(
                        v, config.quantization.clamp_val)
                    weights[tllm_prex +
                            'mlp.proj.weight'] = processed_torch_weights
                    weights[tllm_prex +
                            'mlp.proj.per_channel_scale'] = torch_weight_scales
                else:
                    weights[tllm_prex + 'mlp.proj.weight'] = v
            elif 'feed_forward.w1.weight' in k:
                if use_fp8_rowwise_in_layer:
                    processed_torch_weights, torch_weight_scales = fp8_per_channel_quant_weight_gpu(
                        v, config.quantization.clamp_val)
                    weights[tllm_prex +
                            'mlp.fc.weight'] = processed_torch_weights
                    weights[tllm_prex +
                            'mlp.fc.per_channel_scale'] = torch_weight_scales
                else:
                    weights[tllm_prex + 'mlp.fc.weight'] = v
            elif 'attention.wo.weight' in k:
                weights[tllm_prex + 'attention.dense.weight'] = v
            elif 'attention.qkv.weight' in k:
                weights[tllm_prex + 'attention.qkv.weight'] = v
            elif 'feed_forward.gate' in k:
                weights[tllm_prex + 'mlp.router.weight'] = v

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')
    return weights
