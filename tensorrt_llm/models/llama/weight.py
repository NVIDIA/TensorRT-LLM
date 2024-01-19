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
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from safetensors import safe_open

import tensorrt_llm
from tensorrt_llm._utils import (numpy_to_torch, str_dtype_to_np,
                                 str_dtype_to_torch, torch_to_numpy)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.quantized.quant import get_dummy_quant_scales
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime.lora_manager import LoraConfig

from .utils import (iterate_shard_files, load_state_dict,
                    retrieved_layer_index_from_name)


def get_scaling_factors(
    model_path: Union[str, Path],
    num_layers: int,
    quant_mode: Optional[QuantMode] = None,
) -> Optional[Dict[str, List[int]]]:
    """ Get the scaling factors for LLaMA model

    Returns a dictionary of scaling factors for the selected layers of the
    LLaMA model.

    Args:
        model_path (str): Path to the quantized LLaMA model
        layers (list): List of layers to get the scaling factors for. If None,
            all layers are selected.

    Returns:
        dict: Dictionary of scaling factors for the selected layers of the
        LLaMA model.

        example:

        {
            'qkv_act': qkv_act_scale,
            'qkv_weights': qkv_weights_scale,
            'qkv_output' : qkv_outputs_scale,
            'dense_act': dense_act_scale,
            'dense_weights': dense_weights_scale,
            'fc_act': fc_act_scale,
            'fc_weights': fc_weights_scale,
            'gate_act': gate_act_scale,
            'gate_weights': gate_weights_scale,
            'proj_act': proj_act_scale,
            'proj_weights': proj_weights_scale,
        }
    """

    if model_path is None:
        tensorrt_llm.logger.warning(
            f"--quantized_fp8_model_path not specified. "
            f"Initialize quantization scales automatically.")
        return get_dummy_quant_scales(num_layers)
    weight_dict = np.load(model_path)
    # yapf: disable
    scaling_factor = {
        'qkv_act': [],
        'qkv_weights': [],
        'dense_act': [],
        'dense_weights': [],
        'fc_act': [],
        'fc_weights': [],
        'gate_act': [],
        'gate_weights': [],
        'proj_act': [],
        'proj_weights': [],
    }

    if quant_mode is not None and quant_mode.has_fp8_kv_cache():
        scaling_factor['qkv_output'] = []

    for layer in range(num_layers):
        scaling_factor['qkv_act'].append(max(
            weight_dict[f'_np:layers:{layer}:attention:qkv:q:activation_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:k:activation_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:v:activation_scaling_factor'].item()
        ))
        scaling_factor['qkv_weights'].append(max(
            weight_dict[f'_np:layers:{layer}:attention:qkv:q:weights_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:k:weights_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:v:weights_scaling_factor'].item()
        ))
        if quant_mode is not None and quant_mode.has_fp8_kv_cache():
            # Not calibrarting KV cache.
            scaling_factor['qkv_output'].append(1.0)
        scaling_factor['dense_act'].append(
            weight_dict[f'_np:layers:{layer}:attention:dense:activation_scaling_factor'].item())
        scaling_factor['dense_weights'].append(
            weight_dict[f'_np:layers:{layer}:attention:dense:weights_scaling_factor'].item())
        scaling_factor['fc_act'].append(weight_dict[f'_np:layers:{layer}:mlp:fc:activation_scaling_factor'].item())
        scaling_factor['fc_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:fc:weights_scaling_factor'].item())
        scaling_factor['gate_act'].append(weight_dict[f'_np:layers:{layer}:mlp:gate:activation_scaling_factor'].item())
        scaling_factor['gate_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:gate:weights_scaling_factor'].item())
        scaling_factor['proj_act'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:activation_scaling_factor'].item())
        scaling_factor['proj_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:weights_scaling_factor'].item())
    # yapf: enable
    for k, v in scaling_factor.items():
        assert len(v) == num_layers, \
            f'Expect scaling factor {k} of length {num_layers}, got {len(v)}'

    return scaling_factor


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


def load_from_hf_llama(tensorrt_llm_llama: 'LLaMAForCausalLM',
                       hf_llama,
                       mapping=Mapping(),
                       dtype='float32',
                       use_gemm_woq_plugin=True,
                       lora_config=LoraConfig()):
    tensorrt_llm.logger.info('Loading weights from HF LLaMA...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_llama, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()
    num_kv_heads = tensorrt_llm_llama.num_kv_heads
    mha_mode = (num_kv_heads == tensorrt_llm_llama.num_heads)

    model_params = dict(hf_llama.named_parameters())
    # concatenate, duplicate and reshape q, k, v -> qkv
    for l in range(hf_llama.config.num_hidden_layers):
        prefix = f'model.layers.{l}.self_attn.'
        q_weight = model_params[prefix + 'q_proj.weight']
        k_weight = model_params[prefix + 'k_proj.weight']
        v_weight = model_params[prefix + 'v_proj.weight']
        if not mha_mode:
            head_size = tensorrt_llm_llama.hidden_size // tensorrt_llm_llama.num_heads
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

    # concatenate MoE gated activations & stack experts
    for l in range(hf_llama.config.num_hidden_layers):
        moe_config = tensorrt_llm_llama.moe_config
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
    layers_per_pipeline_stage = hf_llama.config.num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    vocab_size = hf_llama.config.vocab_size
    for k, v in model_params.items():
        t_dtype = torch_dtype if "block_sparse_moe.gate" not in k else torch.float32
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(t_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(t_dtype).detach().cpu())
        if 'model.embed_tokens.weight' in k:
            if lora_config.is_valid and lora_config.embedding_weight is not None:
                v = torch_to_numpy(
                    lora_config.embedding_weight.to(torch_dtype).detach().cpu())
            if hf_llama.config.tie_word_embeddings:
                # lm_head.weight has the same weights as embedding
                if mapping.is_last_pp_rank():
                    tensorrt_llm_llama.lm_head.weight.value = np.ascontiguousarray(
                        split(v, mapping.tp_size, mapping.tp_rank))
            if tensorrt_llm_llama.use_parallel_embedding:
                v = split(v, mapping.tp_size, mapping.tp_rank,
                          tensorrt_llm_llama.embedding_sharding_dim)
            if mapping.is_first_pp_rank():
                tensorrt_llm_llama.vocab_embedding.weight.value = v
        elif 'model.norm.weight' in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_llama.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            if mapping.is_last_pp_rank():
                if lora_config.is_valid and lora_config.lm_head_weight is not None:
                    v = torch_to_numpy(
                        lora_config.lm_head_weight.to(
                            torch_dtype).detach().cpu())
                    vocab_size = v.shape[0]
                if vocab_size % mapping.tp_size != 0:
                    # padding
                    vocab_size_padded = tensorrt_llm_llama.lm_head.out_features * mapping.tp_size
                    pad_width = vocab_size_padded - vocab_size
                    v = np.pad(v, ((0, pad_width), (0, 0)),
                               'constant',
                               constant_values=0)
                tensorrt_llm_llama.lm_head.weight.value = np.ascontiguousarray(
                    split(v, mapping.tp_size, mapping.tp_rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None or int(layer_idx) not in layers_range:
                continue
            idx = int(layer_idx) - mapping.pp_rank * layers_per_pipeline_stage
            if 'input_layernorm.weight' in k:
                tensorrt_llm_llama.layers[idx].input_layernorm.weight.value = v
            elif 'post_attention_layernorm.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].post_layernorm.weight
                dst.value = v
            elif 'self_attn.qkv_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].attention.qkv.weight
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
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = numpy_to_torch(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = torch_to_numpy(processed_torch_weights)
                    scales = tensorrt_llm_llama.layers[
                        idx].attention.qkv.per_channel_scale
                    scales.value = torch_to_numpy(torch_weight_scales)
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'self_attn.o_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].attention.dense.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = numpy_to_torch(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = torch_to_numpy(processed_torch_weights)
                    scales = tensorrt_llm_llama.layers[
                        idx].attention.dense.per_channel_scale
                    scales.value = torch_to_numpy(torch_weight_scales)
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.up_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].mlp.gate.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)

                    if not use_gemm_woq_plugin:
                        dst.value = numpy_to_torch(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = torch_to_numpy(processed_torch_weights)

                    scales = tensorrt_llm_llama.layers[
                        idx].mlp.gate.per_channel_scale
                    scales.value = torch_to_numpy(torch_weight_scales)
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.down_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].mlp.proj.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = numpy_to_torch(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = torch_to_numpy(processed_torch_weights)
                    scales = tensorrt_llm_llama.layers[
                        idx].mlp.proj.per_channel_scale
                    scales.value = torch_to_numpy(torch_weight_scales)
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.gate_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].mlp.fc.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)

                    if not use_gemm_woq_plugin:
                        dst.value = numpy_to_torch(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = torch_to_numpy(processed_torch_weights)
                    scales = tensorrt_llm_llama.layers[
                        idx].mlp.fc.per_channel_scale
                    scales.value = torch_to_numpy(torch_weight_scales)
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'experts.w2.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].mlp.experts_weight_2
                # Note: no need for splitting, it's already been done above
                split_v = v
                if use_weight_only:
                    v = np.ascontiguousarray(
                        np.transpose(split_v, axes=(0, 2, 1)))
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)
                    dst.value = processed_torch_weights
                    tensorrt_llm_llama.layers[
                        idx].mlp.experts_scale_2.value = torch_weight_scales
                else:
                    dst.value = np.ascontiguousarray(v)
            elif 'experts.w3w1.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].mlp.experts_weight_1
                # Note: no need for splitting, it's already been done above
                split_v = v
                if use_weight_only:
                    v = np.ascontiguousarray(
                        np.transpose(split_v, axes=(0, 2, 1)))
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)
                    dst.value = processed_torch_weights
                    tensorrt_llm_llama.layers[
                        idx].mlp.experts_scale_1.value = torch_weight_scales
                else:
                    dst.value = np.ascontiguousarray(v)
            elif 'block_sparse_moe.gate' in k:
                v = split(v, mapping.tp_size, mapping.tp_rank, dim=-1)
                tensorrt_llm_llama.layers[
                    idx].mlp.router.weight.value = np.ascontiguousarray(v)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


class QkvWeightHelper:
    """ A helper utility for loading QKV weights from sharded files. """

    def __init__(self, model: 'LLaMAForCausalLM'):
        self.hidden_size = model.hidden_size
        self.num_heads = model.num_heads
        self.num_kv_heads = model.num_kv_heads
        self.tp_size = model.mapping.tp_size
        self.tp_rank = model.mapping.tp_rank
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

    def is_qkv_prepared(self, layer_id):
        if layer_id not in self._qkv_weights:
            return False
        weights = self._qkv_weights[layer_id]
        return 'q' in weights and 'k' in weights and 'v' in weights

    def split_qkv_weights(self, layer_id):
        if not self.is_qkv_prepared(layer_id):
            return None
        weights = self._qkv_weights.pop(layer_id)  # to prevent memory leak.
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


def load_from_hf_checkpoint(
        tensorrt_llm_llama: 'LLaMAForCausalLM',
        model_dir: Union[str, Path],
        mapping=Mapping(),
        dtype: Union[str, torch.dtype] = torch.float32,
        lora_config=LoraConfig(),
):
    tensorrt_llm.logger.info('Loading weights from HF LLaMA...')
    tik = time.time()
    if isinstance(dtype, str):
        dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)

    assert not tensorrt_llm_llama.moe_config.has_moe(
    ), "MoE does not support sharded load"

    model_dir = Path(model_dir)

    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_dir)

    quant_mode = getattr(tensorrt_llm_llama, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    layers_range = mapping.pp_layers(tensorrt_llm_llama.num_layers)

    qkv_weight_helper = QkvWeightHelper(tensorrt_llm_llama)

    for model_file in iterate_shard_files(model_dir,
                                          rank=mapping.tp_rank,
                                          progress_bar=False):
        tensorrt_llm.logger.debug(f'Loading file {str(model_file)}...')
        model_params = load_state_dict(model_file, dtype=dtype)
        for name, param in model_params.items():
            tensorrt_llm.logger.debug(f'Converting weight {name}...')
            i = retrieved_layer_index_from_name(name)
            if i is None:
                layer = None
            else:
                if i not in layers_range:
                    continue
                layer = tensorrt_llm_llama.layers[i - layers_range[0]]

            if 'model.embed_tokens.weight' in name:
                if lora_config.is_valid and lora_config.embedding_weight is not None:
                    param = lora_config.embedding_weight.to(dtype)
                if hf_config.tie_word_embeddings:
                    # lm_head.weight has the same weights as embedding
                    if mapping.is_last_pp_rank():
                        tensorrt_llm_llama.lm_head.weight.value = split(
                            param, mapping.tp_size, mapping.tp_rank)
                if tensorrt_llm_llama.use_parallel_embedding:
                    param = split(param, mapping.tp_size, mapping.tp_rank,
                                  tensorrt_llm_llama.embedding_sharding_dim)
                if mapping.is_first_pp_rank():
                    tensorrt_llm_llama.vocab_embedding.weight.value = param
            elif 'model.norm.weight' in name:
                if mapping.is_last_pp_rank():
                    tensorrt_llm_llama.ln_f.weight.value = param
            elif 'lm_head.weight' in name:
                if lora_config.is_valid and lora_config.lm_head_weight is not None:
                    param = lora_config.lm_head_weight.to(dtype)
                if mapping.is_last_pp_rank():
                    tensorrt_llm_llama.lm_head.weight.value = split(
                        param, mapping.tp_size, mapping.tp_rank)
            elif 'input_layernorm.weight' in name:
                layer.input_layernorm.weight.value = param
            elif 'post_attention_layernorm.weight' in name:
                layer.post_layernorm.weight.value = param
            elif qkv_weight_helper.is_qkv_weight(name):
                qkv_weight_helper.add_weight(i, name, param)
                if not qkv_weight_helper.is_qkv_prepared(i):
                    continue
                split_v = qkv_weight_helper.split_qkv_weights(i)
                if use_weight_only:
                    param = split_v.transpose()
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            param, plugin_weight_only_quant_type)
                    layer.attention.qkv.weight.value = processed_torch_weights
                    layer.attention.qkv.per_channel_scale.value = torch_weight_scales
                else:
                    layer.attention.qkv.weight.value = split_v
            elif 'self_attn.o_proj.weight' in name:
                split_v = split(param, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            split_v.transpose(), plugin_weight_only_quant_type)
                    layer.attention.dense.weight.value = processed_torch_weights
                    layer.attention.dense.per_channel_scale.value = torch_weight_scales
                else:
                    layer.attention.dense.weight.value = split_v
            elif 'mlp.up_proj.weight' in name:
                split_v = split(param, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            split_v.transpose(), plugin_weight_only_quant_type)
                    layer.mlp.gate.weight.value = processed_torch_weights
                    layer.mlp.gate.per_channel_scale.value = torch_weight_scales
                else:
                    layer.mlp.gate.weight.value = split_v
            elif 'mlp.down_proj.weight' in name:
                split_v = split(param, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            split_v.transpose(), plugin_weight_only_quant_type)
                    layer.mlp.proj.weight.value = processed_torch_weights
                    layer.mlp.proj.per_channel_scale.value = torch_weight_scales
                else:
                    layer.mlp.proj.weight.value = split_v
            elif 'mlp.gate_proj.weight' in name:
                split_v = split(param, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                            split_v.transpose(), plugin_weight_only_quant_type)
                    layer.mlp.fc.weight.value = processed_torch_weights
                    layer.mlp.fc.per_channel_scale.value = torch_weight_scales
                else:
                    layer.mlp.fc.weight.value = split_v
        del model_params
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


def load_from_meta_llama(tensorrt_llm_llama: 'LLaMAForCausalLM',
                         meta_ckpt_dir,
                         mapping=Mapping(),
                         dtype="float32"):
    torch_dtype = str_dtype_to_torch(dtype)

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
        moe_config = tensorrt_llm_llama.moe_config
        split_ckpt = {}
        for k, v in ckpt.items():
            d = 0
            if any(n in k for n in
                   ["wo", "feed_forward.w2", "tok", "feed_forward.gate"]):
                d = 1
            if "feed_forward.experts" in k and ("w2" in k) == (
                    not quant_mode.is_weight_only()):
                d = 1

            if "norm" in k or "rope" in k:  # no TP
                split_ckpt[k] = v.clone()
            elif tensorrt_llm_llama.num_kv_heads < mapping.tp_size and any(
                    n in k for n in ["wk", "wv"]):
                assert mapping.tp_size % tensorrt_llm_llama.num_kv_heads == 0
                # special case: we need to duplicate KV head
                tmp = dup_kv_weight(v, tensorrt_llm_llama.num_kv_heads,
                                    mapping.tp_size)
                split_ckpt[k] = torch.split(tmp,
                                            tmp.shape[d] // ranks_per_ckpt,
                                            dim=d)[ckpt_rank].clone()
            elif "experts" in k and moe_config.tp_mode == moe_config.ParallelismMode.EXPERT_PARALLEL:
                rank_experts = mapping.ep_experts(moe_config.num_experts)
                expert_id = int(k[k.find("experts"):].split(".")[1])
                if expert_id in rank_experts:
                    split_ckpt[k] = v.clone()
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
            nH_per_ckpt = tensorrt_llm_llama.num_heads // num_ckpts
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

    if not hasattr(load_from_meta_llama, "saved_embed"):
        load_from_meta_llama.saved_embed = None

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
            embed = torch.cat(embeds, dim=1).to(torch_dtype)
            load_from_meta_llama.saved_embed = torch_to_numpy(
                embed)  # cache the embedding, not needed if no refit
        return load_from_meta_llama.saved_embed

    tensorrt_llm.logger.info('Loading weights from Meta LLaMA checkpoints ...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_llama, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        torch.int8
    elif quant_mode.is_int4_weight_only():
        torch.quint4x2
    quant_mode.is_weight_only()
    num_kv_heads = tensorrt_llm_llama.num_kv_heads
    mha_mode = (num_kv_heads == tensorrt_llm_llama.num_heads)

    ckpts = list(Path(meta_ckpt_dir).glob("consolidated.*.pth"))
    num_ckpts = len(ckpts)
    # llama/llama2 doesn't have MQA. So, simplifying loader logic by not worrying about it.
    assert num_kv_heads > 1 or num_kv_heads >= num_ckpts, \
        f"We don't know how the {num_kv_heads} KV heads are distributed among {num_ckpts} checkpoints."

    head_size = tensorrt_llm_llama.hidden_size // tensorrt_llm_llama.num_heads
    ckpt = get_current_weights(num_ckpts)
    layers_per_pipeline_stage = tensorrt_llm_llama.num_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for l in layers_range:
        prefix = f'layers.{l}.attention.'
        q_weight = permute(ckpt[prefix + 'wq.weight'].clone(),
                           nH=(tensorrt_llm_llama.num_heads // mapping.tp_size),
                           d=tensorrt_llm_llama.hidden_size,
                           dH=head_size)
        if num_kv_heads < mapping.tp_size and num_ckpts >= mapping.tp_size:
            assert mapping.tp_size % num_kv_heads == 0
            assert False, "Not supported yet"
        k_weight = permute(ckpt[prefix + 'wk.weight'].clone(),
                           nH=((num_kv_heads + mapping.tp_size - 1) //
                               mapping.tp_size),
                           d=tensorrt_llm_llama.hidden_size,
                           dH=head_size)
        v_weight = ckpt[prefix + 'wv.weight'].clone()

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        ckpt[prefix + 'qkv.weight'] = qkv_weight

    for l in layers_range:
        moe_config = tensorrt_llm_llama.moe_config
        if not moe_config.has_moe():
            continue

        rank_experts = list(range(moe_config.num_experts))
        if moe_config.tp_mode == moe_config.ParallelismMode.EXPERT_PARALLEL:
            rank_experts = mapping.ep_experts(moe_config.num_experts)
        for suffix in ["w1", "w2", "w3"]:
            ckpt[f'layers.{l}.feed_forward.experts.{suffix}.weight'] = \
                torch.stack(list(ckpt[f'layers.{l}.feed_forward.experts.{expert}.{suffix}.weight']
                            for expert in rank_experts))

        # concat w3 and w1 for gated expert
        ckpt[f'layers.{l}.feed_forward.experts.w3w1.weight'] = \
            torch.concat([ckpt[f'layers.{l}.feed_forward.experts.w3.weight'],
                          ckpt[f'layers.{l}.feed_forward.experts.w1.weight']], dim=-2)

    for k, v in ckpt.items():
        dtype = torch_dtype if 'feed_forward.gate' not in k else torch.float32
        v = torch_to_numpy(v.to(dtype).detach().cpu())
        if "tok_embeddings" in k:
            if not tensorrt_llm_llama.use_parallel_embedding:
                v = gather_embedding(v, k, num_ckpts)
            elif tensorrt_llm_llama.embedding_sharding_dim == 0:
                # this needs a gather and then resplit along different dims
                v = gather_embedding(v, k, num_ckpts)
                v = split(v, mapping.tp_size, mapping.tp_rank, 0)
            if mapping.is_first_pp_rank():
                tensorrt_llm_llama.vocab_embedding.weight.value = v
        elif "output" in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_llama.lm_head.weight.value = v
        elif k == "norm.weight":
            if mapping.is_last_pp_rank():
                tensorrt_llm_llama.ln_f.weight.value = v
        else:
            # layer specific weights
            layer_idx = extract_layer_idx(k)
            if layer_idx is None or int(layer_idx) not in layers_range:
                continue
            idx = int(layer_idx) - mapping.pp_rank * layers_per_pipeline_stage
            if 'attention_norm.weight' in k:
                tensorrt_llm_llama.layers[idx].input_layernorm.weight.value = v
            elif 'ffn_norm.weight' in k:
                tensorrt_llm_llama.layers[idx].post_layernorm.weight.value = v
            elif 'feed_forward.w3.weight' in k:
                tensorrt_llm_llama.layers[idx].mlp.gate.weight.value = v
            elif 'feed_forward.w2.weight' in k:
                tensorrt_llm_llama.layers[idx].mlp.proj.weight.value = v
            elif 'feed_forward.w1.weight' in k:
                tensorrt_llm_llama.layers[idx].mlp.fc.weight.value = v
            elif 'attention.wo.weight' in k:
                tensorrt_llm_llama.layers[idx].attention.dense.weight.value = v
            elif 'attention.qkv.weight' in k:
                tensorrt_llm_llama.layers[idx].attention.qkv.weight.value = v
            elif 'experts.w2.weight' in k:
                tensorrt_llm_llama.layers[idx].mlp.experts_weight_2.value = v
            elif 'experts.w3w1.weight' in k:
                tensorrt_llm_llama.layers[idx].mlp.experts_weight_1.value = v
            elif 'feed_forward.gate' in k:
                tensorrt_llm_llama.layers[idx].mlp.router.weight.value = v

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return


def load_from_binary(tensorrt_llm_llama: 'LLaMAForCausalLM',
                     dir_path,
                     mapping=Mapping(),
                     fp16=False,
                     multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from binary...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_llama, 'quant_mode', QuantMode(0))

    n_embd, n_head, n_layer, n_positions, vocab_size, hidden_act, inter_size, n_kv_head = parse_bin_config(
        Path(dir_path) / 'config.ini')
    np_dtype = np.float16 if fp16 else np.float32

    def fromfile(dir_path, name, shape=None, dtype=None):
        dtype = np_dtype if dtype is None else dtype
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    def set_smoothquant_scale_factors(module,
                                      pre_scale_weight,
                                      dir_path,
                                      basename,
                                      shape,
                                      per_tok_dyn,
                                      per_channel,
                                      is_qkv=False,
                                      rank=None):
        suffix = "bin"
        if per_channel:
            if rank is not None:
                suffix = f"{rank}." + suffix
            suffix = "col." + suffix

        col_shape = shape if (per_channel or is_qkv) else [1, 1]

        if per_tok_dyn:
            if pre_scale_weight is not None:
                pre_scale_weight.value = np.array([1.0], dtype=np.float32)
            if is_qkv and not per_channel:
                t = fromfile(dir_path,
                             f"{basename}scale_w_quant_orig.{rank}.{suffix}",
                             col_shape, np.float32)
            else:
                t = fromfile(dir_path, f"{basename}scale_w_quant_orig.{suffix}",
                             col_shape, np.float32)
            module.per_channel_scale.value = t
        else:
            t = fromfile(dir_path, f"{basename}scale_x_orig_quant.bin", [1],
                         np.float32)
            pre_scale_weight.value = t
            if is_qkv:
                t = fromfile(dir_path,
                             f"{basename}scale_y_accum_quant.{rank}.{suffix}",
                             col_shape, np.float32)
            else:
                t = fromfile(dir_path,
                             f"{basename}scale_y_accum_quant.{suffix}",
                             col_shape, np.float32)
            module.per_channel_scale.value = t
            t = fromfile(dir_path, f"{basename}scale_y_quant_orig.bin", [1, 1],
                         np.float32)
            module.act_scale.value = t

    def set_smoother(module, dir_path, base_name, shape, rank):
        suffix = f"{rank}.bin"
        t = fromfile(dir_path, f"{base_name}.smoother.{suffix}", shape,
                     np.float32)
        module.smoother.value = t

    # Determine the quantization mode.
    quant_mode = getattr(tensorrt_llm_llama, "quant_mode", QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    # Do we use SmoothQuant?
    use_smooth_quant = quant_mode.has_act_and_weight_quant()
    # Do we use quantization per token?
    quant_per_token_dyn = quant_mode.has_per_token_dynamic_scaling()
    # Do we use quantization per channel?
    quant_per_channel = quant_mode.has_per_channel_scaling()

    # Do we use INT4/INT8 weight-only?
    use_weight_only = quant_mode.is_weight_only()

    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    # Debug
    suffix = gen_suffix(mapping.tp_rank, use_smooth_quant, quant_per_channel)
    # The type of weights.
    w_type = np_dtype if not use_smooth_quant else np.int8

    if mapping.is_first_pp_rank():
        tensorrt_llm_llama.vocab_embedding.weight.value = (fromfile(
            dir_path, 'vocab_embedding.weight.bin', [vocab_size, n_embd]))

    if mapping.is_last_pp_rank():
        tensorrt_llm_llama.ln_f.weight.value = (fromfile(
            dir_path, 'ln_f.weight.bin'))
    # share input embedding
    lm_head_weight = fromfile(dir_path, 'lm_head.weight.bin',
                              [vocab_size, n_embd])

    if vocab_size % mapping.tp_size != 0:
        # padding
        vocab_size_padded = tensorrt_llm_llama.lm_head.out_features * mapping.tp_size
        pad_width = vocab_size_padded - vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                'constant',
                                constant_values=0)
    if mapping.is_last_pp_rank():
        tensorrt_llm_llama.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, mapping.tp_size, mapping.tp_rank))

    layers_per_pipeline_stage = tensorrt_llm_llama.num_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for i in layers_range:
        n_groups = n_head // n_kv_head
        c_attn_out_dim = (
            3 * n_embd // mapping.tp_size) if not multi_query_mode else (
                n_embd // mapping.tp_size +
                (n_embd // n_head * n_groups) // mapping.tp_size * 2)
        idx = i - mapping.pp_rank * layers_per_pipeline_stage
        tensorrt_llm_llama.layers[idx].input_layernorm.weight.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.weight.bin'))
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.weight.' + suffix,
            [n_embd, c_attn_out_dim], w_type)
        if t is not None:
            dst = tensorrt_llm_llama.layers[idx].attention.qkv.weight
            if use_smooth_quant:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
                set_smoothquant_scale_factors(
                    tensorrt_llm_llama.layers[idx].attention.qkv,
                    tensorrt_llm_llama.layers[idx].input_layernorm.scale_to_int,
                    dir_path,
                    'model.layers.' + str(i) + '.attention.query_key_value.',
                    [1, c_attn_out_dim],
                    quant_per_token_dyn,
                    quant_per_channel,
                    rank=mapping.tp_rank,
                    is_qkv=True)
            elif use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(t), plugin_weight_only_quant_type)
                dst.value = processed_torch_weights.numpy()
                scales = tensorrt_llm_llama.layers[
                    idx].attention.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_llama.layers[idx].attention.dense.weight
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attention.dense.weight.' + suffix,
            [n_embd // mapping.tp_size, n_embd], w_type)
        if use_smooth_quant:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
            dense_scale = getattr(tensorrt_llm_llama.layers[idx].attention,
                                  "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_llama.layers[idx].attention.dense, dense_scale,
                dir_path, 'model.layers.' + str(i) + '.attention.dense.',
                [1, n_embd], quant_per_token_dyn, quant_per_channel)
            set_smoother(tensorrt_llm_llama.layers[idx].attention.dense,
                         dir_path,
                         'model.layers.' + str(i) + '.attention.dense',
                         [1, n_embd // mapping.tp_size], mapping.tp_rank)
        elif use_weight_only:
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_llama.layers[
                idx].attention.dense.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_llama.layers[idx].post_layernorm.weight
        dst.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.post_layernorm.weight.bin')

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.fc.weight.' + suffix,
                     [n_embd, inter_size // mapping.tp_size], w_type)

        if use_smooth_quant:
            tensorrt_llm_llama.layers[
                idx].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            set_smoothquant_scale_factors(
                tensorrt_llm_llama.layers[idx].mlp.fc,
                tensorrt_llm_llama.layers[idx].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.fc.',
                [1, inter_size // mapping.tp_size],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_llama.layers[idx].mlp.fc.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)

            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_llama.layers[idx].mlp.fc.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_llama.layers[
                idx].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.gate.weight.' + suffix,
                     [n_embd, inter_size // mapping.tp_size], w_type)
        if use_smooth_quant:
            tensorrt_llm_llama.layers[
                idx].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            set_smoothquant_scale_factors(
                tensorrt_llm_llama.layers[idx].mlp.gate,
                tensorrt_llm_llama.layers[idx].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.gate.',
                [1, inter_size // mapping.tp_size],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_llama.layers[idx].mlp.gate.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_llama.layers[idx].mlp.gate.per_channel_scale

            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_llama.layers[
                idx].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.proj.weight.' + suffix,
                     [inter_size // mapping.tp_size, n_embd], w_type)
        if use_smooth_quant:
            tensorrt_llm_llama.layers[
                idx].mlp.proj.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            proj_scale = getattr(tensorrt_llm_llama.layers[idx].mlp,
                                 "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_llama.layers[idx].mlp.proj, proj_scale, dir_path,
                'model.layers.' + str(i) + '.mlp.proj.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
            set_smoother(tensorrt_llm_llama.layers[idx].mlp.proj, dir_path,
                         'model.layers.' + str(i) + '.mlp.proj',
                         [1, inter_size // mapping.tp_size], mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_llama.layers[idx].mlp.proj.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)

            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_llama.layers[idx].mlp.proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_llama.layers[idx].mlp.proj.weight.value = (
                np.ascontiguousarray(np.transpose(t, [1, 0])))

        if use_int8_kv_cache:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            tensorrt_llm_llama.layers[
                idx].attention.kv_cache_scaling_factor.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


def load_from_gptq_llama(tensorrt_llm_llama,
                         quant_ckpt_path,
                         mapping=Mapping(),
                         dtype="float16",
                         bin_model_dir=None):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise GPTQ LLaMA safetensors...')
    tik = time.time()

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

    packer = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    def load(key, no_prefix=0):
        if no_prefix:
            return gptq_llama.get_tensor(key)
        else:
            return gptq_llama.get_tensor(gptq_prefix + key)

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
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

    def process_and_assign_weight(mOp, v, tp_dim=-1):
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
                                           torch.quint4x2).view(torch.float16)
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

        # return processed interleaved weight, original scales and zeros * scales
        mOp.weight.value = qweight_interleaved.cpu().numpy()
        mOp.weights_scaling_factor.value = scales_fp16.cpu().numpy()
        mOp.zero.value = zeros_x_scales_fp16.cpu().numpy()

    # Load weights from GPTQ checkpoint into TRT-LLM module
    # 1. vocab_embedding
    v = load(gptq_key_list[0])
    if mapping.is_first_pp_rank():
        tensorrt_llm_llama.vocab_embedding.weight.value = v.to(
            torch_dtype).cpu().numpy()

    # 2. lm_head
    v = load(gptq_key_list[1], "no_prefix")
    if mapping.is_last_pp_rank():
        tensorrt_llm_llama.lm_head.weight.value = torch_split(
            v, 0).to(torch_dtype).cpu().numpy()

    # 3. ln_f
    v = load(gptq_key_list[2])
    if mapping.is_last_pp_rank():
        tensorrt_llm_llama.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()

    # 4. Weights inside each layer
    num_hidden_layers = tensorrt_llm_llama.num_layers
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for l in layers_range:
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = "layers" + split_sym + str(layer_idx) + split_sym
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        layer = tensorrt_llm_llama.layers[layer_idx]

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

        process_and_assign_weight(layer.attention.qkv, qkv_weight_list)

        # 4.2 attention.dense
        v = [load(prefix + gptq_key_list[5] + suf) for suf in gptq_suffix_list]
        process_and_assign_weight(layer.attention.dense, v, 0)

        # 4.3 mlp.gate
        v = [load(prefix + gptq_key_list[6] + suf) for suf in gptq_suffix_list]
        process_and_assign_weight(layer.mlp.gate, v, 1)

        # 4.4 mlp.proj
        v = [load(prefix + gptq_key_list[7] + suf) for suf in gptq_suffix_list]
        process_and_assign_weight(layer.mlp.proj, v, 0)

        # 4.5 mlp.fc
        v = [load(prefix + gptq_key_list[8] + suf) for suf in gptq_suffix_list]
        process_and_assign_weight(layer.mlp.fc, v, 1)

        # 4.6 input_layernorm
        v = load(prefix + gptq_key_list[9])
        layer.input_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

        # 4.7 post_layernorm
        v = load(prefix + gptq_key_list[10])
        layer.post_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return


def load_from_awq_llama(tensorrt_llm_llama: 'LLaMAForCausalLM',
                        quant_ckpt_path,
                        quantize_lm_head=False,
                        mapping=Mapping(),
                        dtype="float16",
                        bin_model_dir=None):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise AWQ LLaMA checkpoint...')
    tik = time.time()

    if quant_ckpt_path.endswith(".pt"):
        awq_llama = torch.load(quant_ckpt_path)
        awq_prefix = "model."
        awq_suffix_list = [
            ".weight",
            ".weight_quantizer._amax",
            ".input_quantizer._pre_quant_scale",
        ]
        awq_key_list = [
            "embed_tokens.weight",  # vocab_embedding
            "lm_head",  # lm_head
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

        def load(key):
            if "lm_head" in key:
                v = awq_llama[key]
            else:
                v = awq_llama[awq_prefix + key]
            return v

        group_size = load("layers.0.self_attn.o_proj.weight").numel() // load(
            "layers.0.self_attn.o_proj.weight_quantizer._amax").numel()
    elif quant_ckpt_path.endswith(".npz"):
        awq_llama = np.load(quant_ckpt_path)
        awq_prefix = "_np:"
        awq_suffix_list = [
            ":weight",
            ":weights_scaling_factor",
            ":prequant_scaling_factor",
        ]
        awq_key_list = [
            "vocab_embedding:weight",  # vocab_embedding
            "lm_head",  # lm_head
            "final_layernorm:weight",  # ln_f
            "attention:qkv:",  # attention.qkv
            "",  # qkv suffix
            "attention:dense",  # attention.dense
            "mlp:gate",  # mlp.gate
            "mlp:proj",  # mlp.proj
            "mlp:fc",  # mlp.fc
            "input_layernorm:weight",  # input_layernorm
            "post_layernorm:weight",  # post_layernorm
        ]
        split_sym = ":"

        def load(key):
            v = torch.from_numpy(awq_llama[awq_prefix + key])
            if "weights_scaling_factor" in key:
                v *= 7  # For AMMO *.npz checkpoints
            return v

        group_size = load("layers:0:attention:dense:weight").numel() // load(
            "layers:0:attention:dense:weights_scaling_factor").numel()
    else:
        assert False, "Unsupported AWQ quantized checkpoint format"

    quant_mode = getattr(tensorrt_llm_llama, 'quant_mode', QuantMode(0))
    # INT8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()
    # FP8 KV cache
    use_fp8_kv_cache = quant_mode.has_fp8_kv_cache()

    packer = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    def fromfile(dir_path, name, shape=None, dtype=None):
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def AWQ_quantize_pack_preprocess(weight, scale):
        weight /= scale.repeat_interleave(group_size, dim=0)
        qweight_int8 = torch.clamp(torch.round(weight.cuda()).char(), -8, 7)
        int4_weight = preprocessor(packer(qweight_int8.cpu()), torch.quint4x2)
        return int4_weight.view(torch.float16).cpu().numpy()

    def process_and_assign_weight(mOp, v, tp_dim=0):
        weight = v[0].T.contiguous()
        [k, n] = weight.shape
        weight = torch_split(weight, tp_dim)
        amax = v[1].reshape((n, k // group_size)).T.contiguous()
        amax = torch_split(amax, tp_dim)
        pre_quant_scale = v[2].reshape((1, k))
        if tp_dim == 0:
            pre_quant_scale = torch_split(pre_quant_scale, 1)
        scale = amax / 8.0
        mOp.weight.value = AWQ_quantize_pack_preprocess(weight, scale)
        mOp.weights_scaling_factor.value = scale.to(torch_dtype).cpu().numpy()
        mOp.prequant_scaling_factor.value = pre_quant_scale.to(
            torch_dtype).cpu().numpy()

    def reSmooth_and_get_scale(weight, pre_quant_scale, avg_pre_quant_scale):
        # deSmooth and reSmooth
        [k, n] = weight.shape
        if quant_ckpt_path.endswith("pt"):
            # NPZ files are already re-smoothed
            weight *= pre_quant_scale.repeat((n, 1)).transpose(1,
                                                               0).contiguous()
            weight /= avg_pre_quant_scale.repeat(
                (n, 1)).transpose(1, 0).contiguous()

        # Get scale
        weight_t = weight.T.contiguous()
        weight_t = weight_t.reshape(n, k // group_size, group_size)
        weight_t = torch.abs(weight_t.reshape(-1, group_size))
        amax, idx = weight_t.max(1)
        amax = amax.reshape(n, k // group_size).T.contiguous()
        scale = amax / 8
        return weight, scale

    def process_and_assign_qkv_weight(prefix, mOp):
        q_weight = load(prefix + "q" + awq_key_list[4] +
                        awq_suffix_list[0]).T.contiguous()
        k_weight = load(prefix + "k" + awq_key_list[4] +
                        awq_suffix_list[0]).T.contiguous()
        v_weight = load(prefix + "v" + awq_key_list[4] +
                        awq_suffix_list[0]).T.contiguous()
        dim_k = q_weight.shape[0]
        q_weight = torch_split(q_weight, 1)
        k_weight = torch_split(k_weight, 1)
        v_weight = torch_split(v_weight, 1)
        q_pre_quant_scale = load(prefix + "q" + awq_key_list[4] +
                                 awq_suffix_list[2]).reshape((1, dim_k))
        k_pre_quant_scale = load(prefix + "k" + awq_key_list[4] +
                                 awq_suffix_list[2]).reshape((1, dim_k))
        v_pre_quant_scale = load(prefix + "v" + awq_key_list[4] +
                                 awq_suffix_list[2]).reshape((1, dim_k))
        qkv_pre_quant_scale = (q_pre_quant_scale + k_pre_quant_scale +
                               v_pre_quant_scale) / 3.0
        q_weight, q_scale = reSmooth_and_get_scale(q_weight, q_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        k_weight, k_scale = reSmooth_and_get_scale(k_weight, k_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        v_weight, v_scale = reSmooth_and_get_scale(v_weight, v_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        qkv_weights = torch.cat((q_weight, k_weight, v_weight), dim=1)
        qkv_scale = torch.cat((q_scale, k_scale, v_scale), dim=1)

        mOp.prequant_scaling_factor.value = qkv_pre_quant_scale.to(
            torch_dtype).cpu().numpy()
        mOp.weight.value = AWQ_quantize_pack_preprocess(qkv_weights, qkv_scale)
        mOp.weights_scaling_factor.value = qkv_scale.to(
            torch_dtype).cpu().numpy()

    # Load weights from AWQ checkpoint into TRT-LLM module
    # 1. vocab_embedding
    v = load(awq_key_list[0])
    # TRT-LLM requires vocab_size to be multiple of 64 for successful GEMM
    if quantize_lm_head and v.shape[0] % 64 != 0:
        v = torch.nn.functional.pad(v, [0, 0, 0, 64 - v.shape[0] % 64])
    if mapping.is_first_pp_rank():
        tensorrt_llm_llama.vocab_embedding.weight.value = v.to(
            torch_dtype).cpu().numpy()

    # 2. lm_head
    if quantize_lm_head:
        v = [load(awq_key_list[1] + suf) for suf in awq_suffix_list]
        if v[0].shape[0] % 64 != 0:
            v[0] = torch.nn.functional.pad(v[0],
                                           [0, 0, 0, 64 - v[0].shape[0] % 64])
            scale_align = 64 * (v[0].shape[1] // group_size)
            v[1] = v[1].reshape(-1)
            v[1] = torch.nn.functional.pad(
                v[1], [0, scale_align - v[1].shape[0] % scale_align], value=1)
        if mapping.is_last_pp_rank():
            process_and_assign_weight(tensorrt_llm_llama.lm_head, v, 1)
    else:
        v = load(awq_key_list[1] + awq_suffix_list[0])
        if mapping.is_last_pp_rank():
            tensorrt_llm_llama.lm_head.weight.value = torch_split(
                v, 0).to(torch_dtype).cpu().numpy()

    # 3. ln_f
    v = load(awq_key_list[2])
    if mapping.is_last_pp_rank():
        tensorrt_llm_llama.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()

    # 4. Weights inside each layer
    num_hidden_layers = tensorrt_llm_llama.num_layers
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for l in layers_range:
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = "layers" + split_sym + str(layer_idx) + split_sym
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        layer = tensorrt_llm_llama.layers[layer_idx]

        # 4.1 attention.qkv
        process_and_assign_qkv_weight(prefix + awq_key_list[3],
                                      layer.attention.qkv)

        # 4.2 attention.dense
        v = [load(prefix + awq_key_list[5] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.attention.dense, v, 0)

        # 4.3 mlp.gate
        v = [load(prefix + awq_key_list[6] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.gate, v, 1)

        # 4.4 mlp.proj
        v = [load(prefix + awq_key_list[7] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.proj, v, 0)

        # 4.5 mlp.fc
        v = [load(prefix + awq_key_list[8] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.fc, v, 1)

        # 4.6 input_layernorm
        v = load(prefix + awq_key_list[9])
        layer.input_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

        # 4.7 post_layernorm
        v = load(prefix + awq_key_list[10])
        layer.post_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

        # 4.8 attention.kv_cache_scaling_factor for INT8 KV cache
        if use_int8_kv_cache:
            assert bin_model_dir, "You must pass --bin_model_dir to tell TRT-LLM where to look for scales of INT8 kv cache."
            t = fromfile(
                bin_model_dir, 'model.layers.' + str(layer_idx) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            assert t is not None, f"{bin_model_dir} does not contain model.layers.{layer_idx}.attention.query_key_value.scale_y_quant_orig.bin"
            layer.attention.kv_cache_scaling_factor.value = t

        # 4.9 attention.kv_cache_scaling_factor for FP8 KV cache
        if use_fp8_kv_cache:
            t = torch.Tensor([1.0])
            layer.attention.kv_cache_scaling_factor.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
