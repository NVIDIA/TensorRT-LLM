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
import configparser
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from safetensors import safe_open

from ..._utils import (numpy_to_torch, pad_vocab_size, str_dtype_to_torch,
                       torch_to_numpy)
from ...layers import MoeConfig
from ...logger import logger
from ...mapping import Mapping
from ...quantization import QuantMode
from ..convert_utils import (iterate_shard_files, load_state_dict,
                             retrieved_layer_index_from_name)
from ..modeling_utils import PretrainedConfig


def gen_suffix(rank, use_smooth_quant, quant_per_channel):
    suffix = f"{rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix
    return suffix


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v: Union[np.ndarray, torch.Tensor],
          tp_size: int,
          tp_rank: int,
          dim=0):
    if tp_size == 1:
        return v
    assert len(v.shape) > 1 or dim == 0
    if isinstance(v, np.ndarray):
        return np.ascontiguousarray(
            np.split(v, tp_size, axis=dim)[tp_rank].copy())
    else:
        assert v.shape[dim] % tp_size == 0, \
            'Unable to split: shape={v.shape} (dim={dim}) tp_size={tp_size}.'
        split_size = v.shape[dim] // tp_size
        return v.split(split_size, dim=dim)[tp_rank].clone().detach()


def dup_kv_weight(v, num_head, tp_size):
    assert tp_size % num_head == 0
    reps = tp_size // num_head
    head_size = v.shape[0] // num_head
    v = v.reshape(num_head, head_size,
                  -1)[:, None, :, :].expand(num_head, reps, head_size,
                                            v.shape[1])
    return v.reshape(num_head * reps * head_size, -1).clone().detach()


def parse_bin_config(ini_file):
    gpt_config = configparser.ConfigParser()
    gpt_config.read(ini_file)

    n_embd = gpt_config.getint('llama', 'hidden_size')
    n_head = gpt_config.getint('llama', 'num_attention_heads')
    n_layer = gpt_config.getint('llama', 'num_hidden_layers')
    n_positions = gpt_config.getint('llama', 'max_position_embeddings')
    vocab_size = gpt_config.getint('llama', 'vocab_size')
    hidden_act = gpt_config.get('llama', 'hidden_act')
    inter_size = gpt_config.getint('llama', 'intermediate_size', fallback=None)
    n_kv_head = gpt_config.getint('llama', 'num_key_value_heads', fallback=None)

    if inter_size is None:
        inter_size = 4 * n_embd

    return n_embd, n_head, n_layer, n_positions, vocab_size, hidden_act, inter_size, n_kv_head


class QkvWeightHelper:
    """ A helper utility for loading QKV weights from sharded files. """

    def __init__(self, config: PretrainedConfig):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.tp_size = config.mapping.tp_size
        self.tp_rank = config.mapping.tp_rank
        self.is_mha = self.num_heads == self.num_kv_heads
        self._qkv_weights = {}

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

        if not self.is_mha:
            head_size = self.hidden_size // self.num_heads
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


def load_from_hf_checkpoint(model_dir, mapping=Mapping(), config=None):
    '''Weights-only quantization is the only supported quantization recipe here.'''
    logger.info('Loading weights from HF LLaMA...')
    tik = time.time()
    weights = {}
    dtype = config.dtype
    if isinstance(dtype, str):
        dtype = str_dtype_to_torch(dtype)

    moe_config = MoeConfig(config.moe_num_experts, config.moe_top_k,
                           config.moe_tp_mode, config.moe_normalization_mode)
    assert not moe_config.has_moe(), "MoE does not support sharded load"

    model_dir = Path(model_dir)

    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_dir)

    quant_mode = config.quant_mode
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    layers_range = mapping.pp_layers(config.num_hidden_layers)

    qkv_weight_helper = QkvWeightHelper(config)

    for model_file in iterate_shard_files(model_dir,
                                          rank=mapping.tp_rank,
                                          progress_bar=False):
        logger.debug(f'Loading file {str(model_file)}...')
        model_params = load_state_dict(model_file, dtype=dtype)
        for name, param in model_params.items():
            logger.debug(f'Converting weight {name}...')
            layer_idx = retrieved_layer_index_from_name(name)
            if layer_idx is None:
                layer = None
            else:
                if layer_idx not in layers_range:
                    continue
            tllm_prex = f'transformer.layers.{layer_idx}.'

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
            elif 'mlp.up_proj.weight' in name:
                split_v = split(param, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            split_v.transpose(), plugin_weight_only_quant_type)
                    weights[tllm_prex +
                            'mlp.gate.weight'] = processed_torch_weights
                    weights[tllm_prex +
                            'mlp.gate.per_channel_scale'] = torch_weight_scales
                else:
                    weights[tllm_prex + 'mlp.gate.weight'] = split_v
            elif 'mlp.down_proj.weight' in name:
                split_v = split(param, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            split_v.transpose(), plugin_weight_only_quant_type)
                    weights[tllm_prex +
                            'mlp.proj.weight'] = processed_torch_weights
                    weights[tllm_prex +
                            'mlp.proj.per_channel_scale'] = torch_weight_scales
                else:
                    weights[tllm_prex + 'mlp.proj.weight'] = split_v

            elif 'mlp.gate_proj.weight' in name:
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
                else:
                    weights[tllm_prex + 'mlp.fc.weight'] = split_v

        del model_params
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')
    return weights


def load_from_hf_llama(tensorrt_llm_llama: 'LLaMAForCausalLM',
                       hf_llama,
                       mapping=Mapping(),
                       dtype='float32',
                       use_gemm_woq_plugin=True):
    logger.info('Loading weights from HF LLaMA...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_llama, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()
    num_kv_heads = tensorrt_llm_llama.config.num_key_value_heads
    mha_mode = (num_kv_heads == tensorrt_llm_llama.config.num_attention_heads)

    model_params = dict(hf_llama.named_parameters())
    # concatenate, duplicate and reshape q, k, v -> qkv
    for l in range(hf_llama.config.num_hidden_layers):
        prefix = f'model.layers.{l}.self_attn.'
        q_weight = model_params[prefix + 'q_proj.weight']
        k_weight = model_params[prefix + 'k_proj.weight']
        v_weight = model_params[prefix + 'v_proj.weight']
        if not mha_mode:
            head_size = tensorrt_llm_llama.config.hidden_size // tensorrt_llm_llama.config.num_attention_heads
            if num_kv_heads < mapping.tp_size:
                # duplicate the KV heads up to tensor_parallel
                k_weight = dup_kv_weight(k_weight, num_kv_heads,
                                         mapping.tp_size)
                v_weight = dup_kv_weight(v_weight, num_kv_heads,
                                         mapping.tp_size)
            assert (k_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            assert (v_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            qkv_weight = [q_weight, k_weight, v_weight]
        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        model_params[prefix + 'qkv_proj.weight'] = qkv_weight

    moe_config = MoeConfig(tensorrt_llm_llama.config.moe_num_experts,
                           tensorrt_llm_llama.config.moe_top_k,
                           tensorrt_llm_llama.config.moe_tp_mode,
                           tensorrt_llm_llama.config.moe_normalization_mode)
    # concatenate MoE gated activations & stack experts
    for l in range(hf_llama.config.num_hidden_layers):

        if not moe_config.has_moe():
            continue

        rank_experts = list(range(moe_config.num_experts))
        if moe_config.tp_mode == moe_config.ParallelismMode.EXPERT_PARALLEL:
            rank_experts = mapping.ep_experts(moe_config.num_experts)
        for suffix in ["w1", "w2", "w3"]:
            model_params[f'model.layers.{l}.block_sparse_moe.experts.{suffix}.weight'] = \
                torch.stack(list(model_params[f'model.layers.{l}.block_sparse_moe.experts.{expert}.{suffix}.weight']
                            for expert in rank_experts))

        w3 = model_params[
            f'model.layers.{l}.block_sparse_moe.experts.w3.weight']
        w2 = model_params[
            f'model.layers.{l}.block_sparse_moe.experts.w2.weight']
        w1 = model_params[
            f'model.layers.{l}.block_sparse_moe.experts.w1.weight']
        if moe_config.tp_mode == moe_config.ParallelismMode.TENSOR_PARALLEL:
            w3 = split(w3, mapping.tp_size, mapping.tp_rank, dim=1)
            w2 = split(w2, mapping.tp_size, mapping.tp_rank, dim=2)
            w1 = split(w1, mapping.tp_size, mapping.tp_rank, dim=1)
        # concat w3 and w1 for gated expert
        model_params[f'model.layers.{l}.block_sparse_moe.experts.w3w1.weight'] = \
            torch.concat([w3, w1], dim=-2)
        model_params[
            f'model.layers.{l}.block_sparse_moe.experts.w2.weight'] = w2

    torch_dtype = str_dtype_to_torch(dtype)
    layers_range = mapping.pp_layers(hf_llama.config.num_hidden_layers)

    vocab_size = hf_llama.config.vocab_size
    weights = {}
    for k, v in model_params.items():
        t_dtype = torch_dtype if "block_sparse_moe.gate" not in k else torch.float32
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(t_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(t_dtype).detach().cpu())
        if 'model.embed_tokens.weight' in k:
            if hf_llama.config.tie_word_embeddings:
                # lm_head.weight has the same weights as embedding
                if mapping.is_last_pp_rank():
                    if vocab_size % mapping.tp_size != 0:
                        # padding
                        vocab_size_padded = pad_vocab_size(
                            vocab_size, mapping.tp_size)
                        pad_width = vocab_size_padded - vocab_size
                        v = torch.from_numpy(
                            np.pad(v.detach().cpu().numpy(),
                                   ((0, pad_width), (0, 0)),
                                   'constant',
                                   constant_values=0))
                    weights['lm_head.weight'] = split(v, mapping.tp_size,
                                                      mapping.tp_rank)

            if tensorrt_llm_llama.config.use_parallel_embedding:
                v = split(v, mapping.tp_size, mapping.tp_rank,
                          tensorrt_llm_llama.config.embedding_sharding_dim)
            if mapping.is_first_pp_rank():
                weights['transformer.vocab_embedding.weight'] = v
        elif 'model.norm.weight' in k:
            if mapping.is_last_pp_rank():
                weights['transformer.ln_f.weight'] = v

        elif 'lm_head.weight' in k:
            if mapping.is_last_pp_rank():
                if vocab_size % mapping.tp_size != 0:
                    # padding
                    vocab_size_padded = tensorrt_llm_llama.lm_head.out_features * mapping.tp_size
                    pad_width = vocab_size_padded - vocab_size
                    v = np.pad(v, ((0, pad_width), (0, 0)),
                               'constant',
                               constant_values=0)

                weights['lm_head.weight'] = split(v, mapping.tp_size,
                                                  mapping.tp_rank)
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None or int(layer_idx) not in layers_range:
                continue
            idx = int(layer_idx) - layers_range[0]
            if 'input_layernorm.weight' in k:
                weights['transformer.layers.{}.input_layernorm.weight'.format(
                    idx)] = v
            elif 'post_attention_layernorm.weight' in k:
                weights['transformer.layers.{}.post_layernorm.weight'.format(
                    idx)] = v

            elif 'self_attn.qkv_proj.weight' in k:
                if not mha_mode:
                    assert isinstance(v, list) and len(v) == 3
                    wq = split(v[0], mapping.tp_size, mapping.tp_rank)
                    wk = split(v[1], mapping.tp_size, mapping.tp_rank)
                    wv = split(v[2], mapping.tp_size, mapping.tp_rank)
                    split_v = np.concatenate((wq, wk, wv))
                else:
                    q_emb = v.shape[0] // 3
                    model_emb = v.shape[1]
                    v = v.reshape(3, q_emb, model_emb)
                    split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // mapping.tp_size),
                                              model_emb)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        weights['transformer.layers.{}.attention.qkv.weight'.
                                format(idx)] = v
                    else:
                        weights['transformer.layers.{}.attention.qkv.weight'.
                                format(idx)] = processed_torch_weights

                    weights[
                        'transformer.layers.{}.attention.qkv.per_channel_scale'.
                        format(idx)] = torch_weight_scales
                else:
                    weights['transformer.layers.{}.attention.qkv.weight'.format(
                        idx)] = split_v

            elif 'self_attn.o_proj.weight' in k:
                # dst = tensorrt_llm_llama.layers[idx].attention.dense.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        weights['transformer.layers.{}.attention.dense.weight'.
                                format(idx)] = v
                    else:
                        weights['transformer.layers.{}.attention.dense.weight'.
                                format(idx)] = processed_torch_weights

                    weights[
                        'transformer.layers.{}.attention.dense.per_channel_scale'
                        .format(idx)] = torch_weight_scales

                else:
                    weights['transformer.layers.{}.attention.dense.weight'.
                            format(idx)] = split_v

            elif 'mlp.up_proj.weight' in k:
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)

                    if not use_gemm_woq_plugin:
                        weights['transformer.layers.{}.mlp.gate.weight'.format(
                            idx)] = v
                    else:
                        weights['transformer.layers.{}.mlp.gate.weight'.format(
                            idx)] = processed_torch_weights

                    weights['transformer.layers.{}.mlp.gate.per_channel_scale'.
                            format(idx)] = torch_weight_scales
                else:
                    weights['transformer.layers.{}.mlp.gate.weight'.format(
                        idx)] = split_v

            elif 'mlp.down_proj.weight' in k:
                # dst = tensorrt_llm_llama.layers[idx].mlp.proj.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        weights['transformer.layers.{}.mlp.proj.weight'.format(
                            idx)] = v
                    else:
                        weights['transformer.layers.{}.mlp.proj.weight'.format(
                            idx)] = processed_torch_weights

                    weights['transformer.layers.{}.mlp.proj.per_channel_scale'.
                            format(idx)] = torch_weight_scales
                else:
                    weights['transformer.layers.{}.mlp.proj.weight'.format(
                        idx)] = split_v
            elif 'mlp.gate_proj.weight' in k:
                # dst = tensorrt_llm_llama.layers[idx].mlp.fc.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)

                    if not use_gemm_woq_plugin:
                        weights['transformer.layers.{}.mlp.fc.weight'.format(
                            idx)] = v
                    else:
                        weights['transformer.layers.{}.mlp.fc.weight'.format(
                            idx)] = processed_torch_weights

                    weights['transformer.layers.{}.mlp.fc.per_channel_scale'.
                            format(idx)] = torch_weight_scales
                else:
                    # dst.value = np.ascontiguousarray(split_v)
                    weights['transformer.layers.{}.mlp.fc.weight'.format(
                        idx)] = split_v
            elif 'experts.w2.weight' in k:
                # Note: no need for splitting, it's already been done above
                split_v = v
                if use_weight_only:
                    v = np.ascontiguousarray(
                        np.transpose(split_v, axes=(0, 2, 1)))
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)
                    weights['transformer.layers.{}.mlp.proj.weight'.format(
                        idx)] = processed_torch_weights
                    weights['transformer.layers.{}.mlp.proj.per_channel_scale'.
                            format(idx)] = torch_weight_scales

                else:
                    weights['transformer.layers.{}.mlp.proj.weight'.format(
                        idx)] = v
            elif 'experts.w3w1.weight' in k:
                # Note: no need for splitting, it's already been done above
                split_v = v
                if use_weight_only:
                    v = np.ascontiguousarray(
                        np.transpose(split_v, axes=(0, 2, 1)))
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)
                    weights['transformer.layers.{}.mlp.fc.weight'.format(
                        idx)] = processed_torch_weights
                    weights['transformer.layers.{}.mlp.fc.per_channel_scale'.
                            format(idx)] = torch_weight_scales

                else:
                    weights['transformer.layers.{}.mlp.fc.weight'.format(
                        idx)] = v

            elif 'block_sparse_moe.gate' in k:
                v = split(v, mapping.tp_size, mapping.tp_rank, dim=-1)
                weights['transformer.layers.{}.mlp.router.weight'.format(
                    idx)] = v

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')
    return weights


def load_from_gptq_llama(config: PretrainedConfig, quant_ckpt_path):
    logger.info('Loading weights from groupwise GPTQ LLaMA safetensors...')
    weights = {}
    tik = time.time()

    num_hidden_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    dtype = config.dtype
    mapping = config.mapping

    gptq_llama = safe_open(quant_ckpt_path, framework="pt", device=0)
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


def load_from_meta_llama(meta_ckpt_dir, mapping, config):
    torch_dtype = str_dtype_to_torch(config.dtype)
    weights = {}

    def gather_ckpts(ckpts):
        gathered = {}
        for k in ckpts[0]:
            d = 0
            if any([n in k for n in ["wo", "w2", "tok"]]):
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
                ckpt = torch.load(Path(meta_ckpt_dir,
                                       f"consolidated.{f:02d}.pth"),
                                  map_location="cpu")
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
            ckpt = torch.load(Path(meta_ckpt_dir,
                                   f"consolidated.{ckpt_fid:02d}.pth"),
                              map_location="cpu")
            return split_ckpt(ckpt, ranks_per_ckpt, ckpt_rank)

        # num_ckpts == tensor_parallel, 1:1 mapping from files to TP
        return torch.load(Path(meta_ckpt_dir,
                               f"consolidated.{mapping.tp_rank:02d}.pth"),
                          map_location="cpu")

    def permute(w, nH, d, dH):
        # due to MQA's wk, nH*dH != d could be true
        return w.view(nH, dH // 2, 2, d).transpose(1, 2).reshape(nH * dH, d)

    def extract_layer_idx(name):
        ss = name.split('.')
        for s in ss:
            if s.isdigit():
                return s
        return None

    if not hasattr(load_from_meta_llama, "saved_embed"):
        load_from_meta_llama.saved_embed = None

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
        if load_from_meta_llama.saved_embed is None:
            embeds = [None] * num_ckpts
            for i in range(num_ckpts):
                ckpt = torch.load(Path(meta_ckpt_dir,
                                       f"consolidated.{i:02d}.pth"),
                                  map_location="cpu")
                embeds[i] = ckpt[name]
            embed = combine_embeddings(embeds, num_ckpts).to(torch_dtype)
            load_from_meta_llama.saved_embed = embed

        return load_from_meta_llama.saved_embed

    logger.info('Loading weights from Meta LLaMA checkpoints ...')
    tik = time.time()

    num_kv_heads = config.num_key_value_heads
    mha_mode = (num_kv_heads == config.num_attention_heads)

    ckpts = list(Path(meta_ckpt_dir).glob("consolidated.*.pth"))
    num_ckpts = len(ckpts)
    # llama/llama2 doesn't have MQA. So, simplifying loader logic by not worrying about it.
    assert num_kv_heads > 1 or num_kv_heads >= num_ckpts, \
        f"We don't know how the {num_kv_heads} KV heads are distributed among {num_ckpts} checkpoints."

    head_size = config.hidden_size // config.num_attention_heads
    ckpt = get_current_weights(num_ckpts)
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

    for k, v in ckpt.items():
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
            idx = int(layer_idx) - layers_range[0]
            tllm_prex = f'transformer.layers.{idx}.'

            if 'attention_norm.weight' in k:
                weights[tllm_prex + 'input_layernorm.weight'] = v
            elif 'ffn_norm.weight' in k:
                weights[tllm_prex + 'post_layernorm.weight'] = v
            elif 'feed_forward.w3.weight' in k:
                weights[tllm_prex + 'mlp.gate.weight'] = v
            elif 'feed_forward.w2.weight' in k:
                weights[tllm_prex + 'mlp.proj.weight'] = v
            elif 'feed_forward.w1.weight' in k:
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


def load_from_hf_safetensors(model_dir: str, config: PretrainedConfig, mapping):
    logger.info('Loading weights from Huggingface LLaMA safetensors...')
    tik = time.time()
    import json
    import os

    import safetensors
    weights = {}

    model_dir = model_dir if model_dir.endswith("/") else model_dir + "/"
    safetensors_map = {}
    try:
        with open(model_dir + "model.safetensors.index.json", 'r') as fr:
            sharding_map = json.load(fr)
        for k, v in sharding_map['weight_map'].items():
            safetensors_map[k] = int(v[6:11]) - 1
    except FileNotFoundError:
        pass
    shard_files = []
    for name in os.listdir(model_dir):
        if name.endswith(".safetensors"):
            shard_files.append(name)
    shard_files.sort()
    safetensors_ptrs = [
        safetensors.safe_open(model_dir + shard_file,
                              framework="pt",
                              device="cpu") for shard_file in shard_files
    ]

    num_hidden_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    pad_vocab = vocab_size % mapping.tp_size != 0
    vocab_size_padded = pad_vocab_size(config.vocab_size, mapping.tp_size)
    dtype = config.dtype

    moe_config = MoeConfig(config.moe_num_experts, config.moe_top_k,
                           config.moe_tp_mode, config.moe_normalization_mode)

    model_prefix = "model."
    key_list = [
        "embed_tokens.weight",  # vocab_embedding
        "lm_head.weight",  # lm_head
        "norm.weight",  # ln_f
        "self_attn.",  # attention.qkv
        "_proj.weight",  # qkv suffix
        "self_attn.o_proj.weight",  # attention.dense
        "mlp.up_proj.weight",  # mlp.gate
        "mlp.down_proj.weight",  # mlp.proj
        "mlp.gate_proj.weight",  # mlp.fc
        "input_layernorm.weight",  # input_layernorm
        "post_attention_layernorm.weight",  # post_layernorm
    ]

    torch_dtype = str_dtype_to_torch(dtype)

    def load(key, tp_dim=-1, no_prefix=0):
        if not no_prefix:
            key = model_prefix + key
        ptr_idx = safetensors_map[key] if key in safetensors_map else 0
        if tp_dim == -1:
            res = safetensors_ptrs[ptr_idx].get_tensor(key)
        else:
            tensor_slice = safetensors_ptrs[ptr_idx].get_slice(key)
            tensor_shape = tensor_slice.get_shape()
            if tensor_shape[tp_dim] % mapping.tp_size != 0:
                logger.error(
                    "Current weight shape is invalid for mapping.tp_size=" +
                    str(mapping.tp_size))
            slice_width = tensor_shape[tp_dim] // mapping.tp_size
            if tp_dim == 0:
                res = tensor_slice[slice_width * mapping.tp_rank:slice_width *
                                   (mapping.tp_rank + 1), :]
            elif tp_dim == 1:
                res = tensor_slice[:,
                                   slice_width * mapping.tp_rank:slice_width *
                                   (mapping.tp_rank + 1)]
            else:
                assert False, "Invalid TP dim"
        return res.to(torch_dtype).contiguous(
        ) if "block_sparse_moe.gate" not in key else res.to(torch.float32)

    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = load(
            key_list[0], config.embedding_sharding_dim
            if config.use_parallel_embedding else -1)  # vocab_embedding

    if mapping.is_last_pp_rank():
        v = load(key_list[1], -1, 1) if pad_vocab else load(key_list[1], 0,
                                                            1)  # lm_head
        if pad_vocab:
            v = torch.nn.functional.pad(
                v, (0, 0, 0, vocab_size_padded - vocab_size), 'constant', 0)
            v = split(v, mapping.tp_size, mapping.tp_rank)
        weights['lm_head.weight'] = v
        weights['transformer.ln_f.weight'] = load(key_list[2])  # ln_f

    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        layer_idx = l - layers_range[0]
        prefix = f'layers.{l}.'
        tllm_prex = f'transformer.layers.{layer_idx}'

        # Attention
        qkv_list = []
        for comp in ["q", "k", "v"]:
            comp_part = load(prefix + key_list[3] + comp + key_list[4], 0)
            qkv_list.append(comp_part)
        weights[f'{tllm_prex}.attention.qkv.weight'] = torch.cat(qkv_list, 0)
        weights[f'{tllm_prex}.attention.dense.weight'] = load(
            prefix + key_list[5], 1)  # attention.dense

        # MLP
        if not moe_config.has_moe():
            weights[f'{tllm_prex}.mlp.gate.weight'] = load(
                prefix + key_list[6], 0)  # mlp.gate
            weights[f'{tllm_prex}.mlp.proj.weight'] = load(
                prefix + key_list[7], 1)  # mlp.proj
            weights[f'{tllm_prex}.mlp.fc.weight'] = load(
                prefix + key_list[8], 0)  # mlp.fc

        else:
            weights[f'{tllm_prex}.mlp.router.weight'] = load(
                prefix + 'block_sparse_moe.gate.weight')
            rank_experts = list(range(moe_config.num_experts))
            if moe_config.tp_mode == moe_config.ParallelismMode.EXPERT_PARALLEL:
                rank_experts = mapping.ep_experts(moe_config.num_experts)

            expert_weight_list = []
            for suffix in range(3):
                tp_dim = -1
                if moe_config.tp_mode == moe_config.ParallelismMode.TENSOR_PARALLEL:
                    tp_dim = 1 if suffix == 1 else 0
                expert_weight_list.append(
                    torch.stack(
                        list(
                            load(
                                prefix +
                                f'block_sparse_moe.experts.{expert}.w{suffix + 1}.weight',
                                tp_dim=tp_dim) for expert in rank_experts)))

            w1 = expert_weight_list[0]
            w2 = expert_weight_list[1]
            w3 = expert_weight_list[2]

            weights[f'{tllm_prex}.mlp.fc.weight'] = \
                torch.concat([w3, w1], dim=-2).contiguous()
            weights[f'{tllm_prex}.mlp.proj.weight'] = w2.contiguous()

        weights[f'{tllm_prex}.input_layernorm.weight'] = load(
            prefix + key_list[9])  # input_layernorm
        weights[f'{tllm_prex}.post_layernorm.weight'] = load(
            prefix + key_list[10])  # post_layernorm

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')

    return weights
