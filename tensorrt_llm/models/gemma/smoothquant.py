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
import math
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      LlamaDecoderLayer,
                                                      LlamaRotaryEmbedding,
                                                      apply_rotary_pos_emb,
                                                      repeat_kv)
from transformers.pytorch_utils import Conv1D

from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import (dup_kv_weight, generate_int8,
                                               get_weight, smooth_gemm,
                                               smooth_gemm_fc1_gate, split,
                                               split_matrix_tp)

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, Cache

    # transformers included  ⬆️ `Cache` in https://github.com/huggingface/transformers/commit/633215ba58fe5114d8c8d32e415a04600e010701 - transformers 4.33, which is installed in the tests, is before this.


@torch.no_grad()
def capture_activation_range(model,
                             tokenizer,
                             dataset,
                             num_samples=1,
                             seq_len=512):
    model.cuda().eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    # tokenizer.pad_token = tokenizer.eos_token

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
            act_scales[name]["w"] = m.weight.abs().clip(
                1e-8, None).max(dim=1)[0].float()

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
        # input_ids = tokenizer(line,
        #                       return_tensors="pt",
        #                       max_length=seq_len,
        #                       padding=True,
        #                       truncation=True).input_ids.to(device)
        inputs = tokenizer.EncodeAsIds(line[0])
        inputs = np.array([[tokenizer.bos_id()] + inputs], dtype=np.int32)
        input_ids = torch.tensor(inputs, dtype=torch.int32).to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def smooth_model(model, scales, alpha: Optional[float], qkv_para,
                 smoother_dict):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(module, LlamaDecoderLayer):
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
        qkv_para[layer_name_qkv] = weight.transpose(0, 1)

        # =================================================================
        layer_name = name + ".self_attn.o_proj"
        smoother = smooth_gemm(module.self_attn.o_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        smoother_dict[layer_name] = smoother.float()

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
        smoother_dict[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.down_proj.weight.abs().max(
            dim=1)[0]


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
            cur_weights = torch.split(original_weights,
                                      original_weights.shape[-1] //
                                      tensor_parallel,
                                      dim=-1)[rank]
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
                        axis=-1)[rank]
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
                        vals["scale_w_quant_orig"].shape[-1] // tensor_parallel,
                        dim=-1)[rank]

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
            cur_weights = torch.split(original_weights,
                                      original_weights.shape[-1] //
                                      tensor_parallel,
                                      dim=-1)[rank]
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


def split_qkv_tp(qkv, n_head, n_kv_heads, head_size, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    kv_head_size = n_kv_heads * head_size
    q, k, v = torch.split(qkv, [n_head * head_size, kv_head_size, kv_head_size],
                          dim=0)
    q = split(q, tensor_parallel, rank, dim=0)
    k = split(k, tensor_parallel, rank, dim=0)
    v = split(v, tensor_parallel, rank, dim=0)
    return torch.concatenate([q, k, v], dim=0).contiguous()


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
        results[f'{prefix}weight'] = processed_torch_weights
        results[f'{prefix}per_channel_scale'] = torch_weight_scales
    else:
        results[f'{prefix}weight'] = weight.contiguous()

    if bias is not None:
        results[f'{prefix}bias'] = bias

    return results


class LlamaAttentionExtend(LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_dim = self.config.head_size
        self.q_proj = nn.Linear(self.config.hidden_size,
                                self.config.num_attention_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(self.config.hidden_size,
                                self.config.num_key_value_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(self.config.hidden_size,
                                self.config.num_key_value_heads * self.head_dim,
                                bias=False)
        self.o_proj = nn.Linear(self.config.num_attention_heads * self.head_dim,
                                self.config.hidden_size,
                                bias=False)
        self.config.head_dim = self.head_dim
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        past_key_value: "Optional[Cache]" = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.config.num_key_value_heads *
                                 self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.config.num_attention_heads * self.head_dim) //
                self.config.pretraining_tp,
                dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len,
                                         self.config.num_attention_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len,
                                     self.config.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len,
                                         self.config.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states,
                                                        key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, :key_states.
                                             shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights,
                                             p=self.attention_dropout,
                                             training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.config.num_attention_heads, q_len,
                                  self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()

        # Here is what we extend.
        attn_output = attn_output.reshape(
            bsz, q_len, self.config.num_attention_heads * self.head_dim)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size //
                                            self.config.pretraining_tp,
                                            dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size //
                                                     self.config.pretraining_tp,
                                                     dim=1)
            attn_output = sum([
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


def create_model_from_config(trt_llm_config, weights):
    model_config = LlamaConfig()
    model_config.vocab_size = trt_llm_config.vocab_size
    model_config.dtype = trt_llm_config.dtype
    model_config.max_position_embeddings = trt_llm_config.max_position_embeddings
    model_config.hidden_size = trt_llm_config.hidden_size
    model_config.num_hidden_layers = trt_llm_config.num_hidden_layers
    model_config.num_attention_heads = trt_llm_config.num_attention_heads
    model_config.num_key_value_heads = trt_llm_config.num_key_value_heads
    model_config.hidden_act = trt_llm_config.hidden_act
    model_config.head_size = trt_llm_config.head_size
    model_config.intermediate_size = trt_llm_config.intermediate_size
    model = LlamaForCausalLM(model_config)
    # Hack attention module since head_dim * num_heads > hidden_size for 7B.
    for i in range(model_config.num_hidden_layers):
        module = model.model.layers[i].self_attn
        model.model.layers[i].self_attn = LlamaAttentionExtend(
            module.config, module.layer_idx)
    # Copy wegiht to LLAMA model.
    replace_name_dict = {
        'attention.dense': 'self_attn.o_proj',
        'mlp.proj': 'mlp.down_proj',
        'mlp.gate': 'mlp.up_proj',
        'mlp.fc': 'mlp.gate_proj',
        'ln_f': 'norm',
        'post_layernorm': 'post_attention_layernorm',
        'vocab_embedding': 'embed_tokens',
    }
    for name in list(weights):
        param = weights[name]
        weights.pop(name)
        new_name = name.replace('transformer', 'model')
        for _name in replace_name_dict:
            if _name in new_name:
                new_name = new_name.replace(_name, replace_name_dict[_name])
        if 'attention.qkv' in name:
            qw, kw, vw = torch.split(param, [
                model_config.num_attention_heads * model_config.head_size,
                model_config.num_key_value_heads * model_config.head_size,
                model_config.num_key_value_heads * model_config.head_size,
            ],
                                     dim=0)
            weights[new_name.replace('attention.qkv', 'self_attn.q_proj')] = qw
            weights[new_name.replace('attention.qkv', 'self_attn.k_proj')] = kw
            weights[new_name.replace('attention.qkv', 'self_attn.v_proj')] = vw
        else:
            weights[new_name] = param

    if "lm_head.weight" not in weights:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"].clone()
    model.load_state_dict(weights)
    return model


def convert_hf_model(*, hf_model: "AutoModelForCausalLM", mapping: Mapping,
                     vocab_size: int, dtype: str, use_parallel_embedding: bool,
                     sharding_dim: int, use_weight_only: bool,
                     plugin_weight_only_quant_type: torch.dtype,
                     use_smooth_quant: bool, per_channel: bool, per_token: bool,
                     int8_kv_cache: bool,
                     act_range: "defaultdict[Any, dict[str, None]]",
                     qkv_para: Dict, smoother: Dict):

    weights = {}
    tik = time.time()
    tensor_parallel = mapping.tp_size
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_attention_heads = hf_model.config.num_attention_heads
    hidden_size = hf_model.config.hidden_size
    intermediate_size = hf_model.config.intermediate_size
    head_size = hf_model.config.head_size
    num_key_value_heads = hf_model.config.num_key_value_heads
    mha_mode = (num_key_value_heads == num_attention_heads)

    num_hidden_layers = hf_model.config.num_hidden_layers
    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        print("Processing layer", l)
        prefix = f'model.layers.{l}.'
        layer_idx = int(l) - layers_range[0]
        tllm_prex = f'transformer.layers.{layer_idx}.'

        if use_smooth_quant:
            qkv_weight = qkv_para[prefix + 'self_attn.qkv_proj']
            qkv_out_dim = qkv_weight.shape[1]

            if not mha_mode:
                hidden_size = qkv_weight.shape[0]
                local_dim = hidden_size
                head_size = (qkv_weight.shape[-1] - local_dim) // 2
                qkv_weight = qkv_weight.reshape(hidden_size,
                                                local_dim + 2 * head_size)
            else:
                qkv_weight = qkv_weight.reshape(hidden_size, 3,
                                                head_size * num_attention_heads)

            int8_weights = generate_int8(qkv_weight,
                                         act_range.get(prefix +
                                                       'self_attn.qkv_proj'),
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
                                          smoother_value=None,
                                          smoother_shape=None,
                                          rank=mapping.tp_rank,
                                          cat_dim=-1,
                                          multi_query_mode=bool(not mha_mode)))
        else:
            q_weight = get_weight(model_params, prefix + 'self_attn.q_proj',
                                  dtype)
            k_weight = get_weight(model_params, prefix + 'self_attn.k_proj',
                                  dtype)
            v_weight = get_weight(model_params, prefix + 'self_attn.v_proj',
                                  dtype)
            if not mha_mode:
                if num_key_value_heads < tensor_parallel:
                    # duplicate the KV heads up to tensor_parallel
                    k_weight = dup_kv_weight(k_weight, num_key_value_heads,
                                             tensor_parallel)
                    v_weight = dup_kv_weight(v_weight, num_key_value_heads,
                                             tensor_parallel)
                assert (k_weight.shape[0] % (mapping.tp_size * head_size)) == 0
                assert (v_weight.shape[0] % (mapping.tp_size * head_size)) == 0

                wq = split(q_weight, mapping.tp_size, mapping.tp_rank)
                wk = split(k_weight, mapping.tp_size, mapping.tp_rank)
                wv = split(v_weight, mapping.tp_size, mapping.tp_rank)

                split_v = torch.concat((wq, wk, wv))

            else:
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

                split_v = split_qkv_tp(qkv_weight, num_attention_heads,
                                       num_key_value_heads, head_size,
                                       tensor_parallel, mapping.tp_rank)
            weights.update(
                get_tllm_linear_weight(split_v, tllm_prex + 'attention.qkv.',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type))

        if int8_kv_cache:
            qkv_y = torch.cat([
                act_range.get(prefix + 'self_attn.q_proj')["y"],
                act_range.get(prefix + 'self_attn.k_proj')["y"],
                act_range.get(prefix + 'self_attn.v_proj')["y"]
            ],
                              dim=0)
            int8_kv_scales = qkv_y.max() / 127.
            kv_cache_weights = {}
            kv_cache_weights[
                tllm_prex +
                'attention.kv_cache_scaling_factor'] = int8_kv_scales.reshape(
                    [1])

            weights.update(kv_cache_weights)

        # Attention dense.
        attn_dense_weight = get_weight(model_params,
                                       prefix + 'self_attn.o_proj', dtype)
        if use_smooth_quant:
            attn_dense_weight = attn_dense_weight.t()
            int8_weights = generate_int8(
                attn_dense_weight, act_range.get(prefix + 'self_attn.o_proj'))
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
                    smoother_value=smoother[(prefix + 'self_attn.o_proj')],
                    smoother_shape=[
                        1, head_size * num_attention_heads // tensor_parallel
                    ],
                    rank=mapping.tp_rank,
                    cat_dim=0))
        else:
            attn_dense_weight = split_matrix_tp(attn_dense_weight,
                                                tensor_parallel,
                                                mapping.tp_rank,
                                                dim=1)
            weights.update(
                get_tllm_linear_weight(attn_dense_weight,
                                       tllm_prex + 'attention.dense.', None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type))
        # MLP hf up to trt gate
        mlp_up_weight = get_weight(model_params, prefix + 'mlp.up_proj', dtype)
        if use_smooth_quant:
            mlp_up_weight = mlp_up_weight.t()
            int8_weights = generate_int8(mlp_up_weight,
                                         act_range.get(prefix + 'mlp.up_proj'))
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
            mlp_up_weight = split_matrix_tp(mlp_up_weight,
                                            tensor_parallel,
                                            mapping.tp_rank,
                                            dim=0)
            weights.update(
                get_tllm_linear_weight(mlp_up_weight, tllm_prex + 'mlp.gate.',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type))

        # MLP trt Gate to mlp fc
        mlp_gate_weight = get_weight(model_params, prefix + 'mlp.gate_proj',
                                     dtype)
        if use_smooth_quant:
            mlp_gate_weight = mlp_gate_weight.t()
            int8_weights = generate_int8(
                mlp_gate_weight, act_range.get(prefix + 'mlp.gate_proj'))
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
            mlp_gate_weight = split_matrix_tp(mlp_gate_weight,
                                              tensor_parallel,
                                              mapping.tp_rank,
                                              dim=0)
            weights.update(
                get_tllm_linear_weight(mlp_gate_weight, tllm_prex + 'mlp.fc.',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type))

        # MLP down
        mlp_proj_weight = get_weight(model_params, prefix + 'mlp.down_proj',
                                     dtype)
        if use_smooth_quant:
            mlp_proj_weight = mlp_proj_weight.t()
            int8_weights = generate_int8(
                mlp_proj_weight, act_range.get(prefix + 'mlp.down_proj'))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'mlp.proj.', [1, hidden_size],
                    tensor_parallel,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + 'mlp.quantization_scaling_factor',
                    smoother_value=smoother[prefix + 'mlp.down_proj'],
                    smoother_shape=[1, intermediate_size // tensor_parallel],
                    rank=mapping.tp_rank,
                    cat_dim=-1))
        else:
            mlp_proj_weight = split_matrix_tp(mlp_proj_weight,
                                              tensor_parallel,
                                              mapping.tp_rank,
                                              dim=1)
            weights.update(
                get_tllm_linear_weight(mlp_proj_weight, tllm_prex + 'mlp.proj.',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type))

        # Layer norms do not use tensor parallelism
        input_ln_weight = get_weight(model_params, prefix + 'input_layernorm',
                                     dtype)
        weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight

        post_ln_weight = get_weight(model_params,
                                    prefix + 'post_attention_layernorm', dtype)
        weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight

    v = get_weight(model_params, 'model.embed_tokens', dtype)

    if use_parallel_embedding:
        v = split_matrix_tp(v,
                            mapping.tp_size,
                            mapping.tp_rank,
                            dim=sharding_dim)

    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v

    lm_head_weights = get_weight(model_params, 'lm_head', dtype)

    if mapping.is_last_pp_rank():

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
        ln_f_w = get_weight(model_params, 'model.norm', dtype)
        weights['transformer.ln_f.weight'] = ln_f_w

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights
