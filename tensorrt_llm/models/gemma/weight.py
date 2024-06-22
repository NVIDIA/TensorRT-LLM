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
import os
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch

from ..._utils import (numpy_to_torch, pad_vocab_size, str_dtype_to_torch,
                       torch_to_numpy)
from ...logger import logger
from ...mapping import Mapping
from ...quantization import QuantMode


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
    model_config = configparser.ConfigParser()
    model_config.read(ini_file)

    n_embd = model_config.getint('gemma', 'hidden_size')
    n_head = model_config.getint('gemma', 'num_attention_heads')
    n_head_size = model_config.getint('gemma',
                                      'head_size',
                                      fallback=n_embd // n_head)
    n_layer = model_config.getint('gemma', 'num_hidden_layers')
    n_positions = model_config.getint('gemma', 'max_position_embeddings')
    vocab_size = model_config.getint('gemma', 'vocab_size')
    hidden_act = model_config.get('gemma', 'hidden_act')
    inter_size = model_config.getint('gemma',
                                     'intermediate_size',
                                     fallback=None)
    n_kv_head = model_config.getint('gemma',
                                    'num_key_value_heads',
                                    fallback=None)

    if inter_size is None:
        inter_size = 4 * n_embd

    return n_embd, n_head, n_layer, n_positions, vocab_size, hidden_act, inter_size, n_kv_head, n_head_size


def load_from_binary(tensorrt_llm_gemma,
                     dir_path,
                     mapping=Mapping(),
                     fp16=False,
                     multi_query_mode=False):
    logger.info('Loading weights from binary...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_gemma, 'quant_mode', QuantMode(0))

    n_embd, n_head, n_layer, n_positions, vocab_size, hidden_act, inter_size, n_kv_head, n_head_size = parse_bin_config(
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
    quant_mode = getattr(tensorrt_llm_gemma, "quant_mode", QuantMode(0))
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
        tensorrt_llm_gemma.vocab_embedding.weight.value = (fromfile(
            dir_path, 'vocab_embedding.weight.bin', [vocab_size, n_embd]))

    if mapping.is_last_pp_rank():
        tensorrt_llm_gemma.ln_f.weight.value = (fromfile(
            dir_path, 'ln_f.weight.bin'))
    # share input embedding
    lm_head_weight = fromfile(dir_path, 'lm_head.weight.bin',
                              [vocab_size, n_embd])

    if vocab_size % mapping.tp_size != 0:
        # padding
        vocab_size_padded = tensorrt_llm_gemma.lm_head.out_features * mapping.tp_size
        pad_width = vocab_size_padded - vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                'constant',
                                constant_values=0)
    if mapping.is_last_pp_rank():
        tensorrt_llm_gemma.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, mapping.tp_size, mapping.tp_rank))

    num_hidden_layers = tensorrt_llm_gemma.num_layers
    layers_range = mapping.pp_layers(num_hidden_layers)

    # This code does not support the case where the number of ranks is greater than the number of K/V heads for GQA.
    assert (n_kv_head % mapping.tp_size == 0) or (n_kv_head == 1)

    # Compute the number of K/V heads per rank. It's 1 for MQA.
    kv_heads_per_rank = min(1, n_kv_head // mapping.tp_size)
    # The N-dimension for each rank of the QKV matrix is number of columns for Q + 2 * number of columns for K/V.
    if multi_query_mode:
        c_attn_out_dim = n_head * n_head_size // mapping.tp_size + 2 * kv_heads_per_rank * n_head_size
    else:
        c_attn_out_dim = 3 * (n_head * n_head_size) // mapping.tp_size

    for i in layers_range:
        idx = i - layers_range[0]
        tensorrt_llm_gemma.layers[idx].input_layernorm.weight.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.weight.bin'))
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.weight.' + suffix,
            [n_embd, c_attn_out_dim], w_type)
        if t is not None:
            dst = tensorrt_llm_gemma.layers[idx].attention.qkv.weight
            if use_smooth_quant:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
                set_smoothquant_scale_factors(
                    tensorrt_llm_gemma.layers[idx].attention.qkv,
                    tensorrt_llm_gemma.layers[idx].input_layernorm.scale_to_int,
                    dir_path,
                    'model.layers.' + str(i) + '.attention.query_key_value.',
                    [1, c_attn_out_dim],
                    quant_per_token_dyn,
                    quant_per_channel,
                    rank=mapping.tp_rank,
                    is_qkv=True)
            elif use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(t), plugin_weight_only_quant_type)
                dst.value = processed_torch_weights.numpy()
                scales = tensorrt_llm_gemma.layers[
                    idx].attention.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_gemma.layers[idx].attention.dense.weight
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attention.dense.weight.' + suffix,
            [(n_head * n_head_size) // mapping.tp_size, n_embd], w_type)
        if use_smooth_quant:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
            dense_scale = getattr(tensorrt_llm_gemma.layers[idx].attention,
                                  "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_gemma.layers[idx].attention.dense, dense_scale,
                dir_path, 'model.layers.' + str(i) + '.attention.dense.',
                [1, n_embd], quant_per_token_dyn, quant_per_channel)
            set_smoother(tensorrt_llm_gemma.layers[idx].attention.dense,
                         dir_path,
                         'model.layers.' + str(i) + '.attention.dense',
                         [1, n_embd // mapping.tp_size], mapping.tp_rank)
        elif use_weight_only:
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_gemma.layers[
                idx].attention.dense.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_gemma.layers[idx].post_layernorm.weight
        dst.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.post_layernorm.weight.bin')

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.fc.weight.' + suffix,
                     [n_embd, inter_size // mapping.tp_size], w_type)

        if use_smooth_quant:
            tensorrt_llm_gemma.layers[
                idx].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            set_smoothquant_scale_factors(
                tensorrt_llm_gemma.layers[idx].mlp.fc,
                tensorrt_llm_gemma.layers[idx].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.fc.',
                [1, inter_size // mapping.tp_size],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_gemma.layers[idx].mlp.fc.weight
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)

            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_gemma.layers[idx].mlp.fc.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_gemma.layers[
                idx].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.gate.weight.' + suffix,
                     [n_embd, inter_size // mapping.tp_size], w_type)
        if use_smooth_quant:
            tensorrt_llm_gemma.layers[
                idx].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            set_smoothquant_scale_factors(
                tensorrt_llm_gemma.layers[idx].mlp.gate,
                tensorrt_llm_gemma.layers[idx].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.gate.',
                [1, inter_size // mapping.tp_size],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_gemma.layers[idx].mlp.gate.weight
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_gemma.layers[idx].mlp.gate.per_channel_scale

            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_gemma.layers[
                idx].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.proj.weight.' + suffix,
                     [inter_size // mapping.tp_size, n_embd], w_type)
        if use_smooth_quant:
            tensorrt_llm_gemma.layers[
                idx].mlp.proj.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            proj_scale = getattr(tensorrt_llm_gemma.layers[idx].mlp,
                                 "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_gemma.layers[idx].mlp.proj, proj_scale, dir_path,
                'model.layers.' + str(i) + '.mlp.proj.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
            set_smoother(tensorrt_llm_gemma.layers[idx].mlp.proj, dir_path,
                         'model.layers.' + str(i) + '.mlp.proj',
                         [1, inter_size // mapping.tp_size], mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_gemma.layers[idx].mlp.proj.weight
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)

            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_gemma.layers[idx].mlp.proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_gemma.layers[idx].mlp.proj.weight.value = (
                np.ascontiguousarray(np.transpose(t, [1, 0])))

        if use_int8_kv_cache:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            tensorrt_llm_gemma.layers[
                idx].attention.kv_cache_scaling_factor.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')


def load_from_hf_gemma(tensorrt_llm_gemma: 'GemmaForCausalLM',
                       hf_gemma,
                       mapping=Mapping(),
                       dtype='float32',
                       use_gemm_woq_plugin=True):
    logger.info('Loading weights from HF Gemma...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_gemma, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()
    num_kv_heads = tensorrt_llm_gemma.config.num_key_value_heads
    mha_mode = (num_kv_heads == tensorrt_llm_gemma.config.num_attention_heads)

    model_params = dict(hf_gemma.named_parameters())
    # concatenate, duplicate and reshape q, k, v -> qkv
    for l in range(hf_gemma.config.num_hidden_layers):
        prefix = f'model.layers.{l}.self_attn.'
        q_weight = model_params[prefix + 'q_proj.weight']
        k_weight = model_params[prefix + 'k_proj.weight']
        v_weight = model_params[prefix + 'v_proj.weight']
        if not mha_mode:
            head_size = tensorrt_llm_gemma.config.hidden_size // tensorrt_llm_gemma.config.num_attention_heads
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

    torch_dtype = str_dtype_to_torch(dtype)
    layers_range = mapping.pp_layers(hf_gemma.config.num_hidden_layers)

    vocab_size = hf_gemma.config.vocab_size
    weights = {}
    for k, v in model_params.items():
        t_dtype = torch_dtype
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(t_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(t_dtype).detach().cpu())
        if 'model.embed_tokens.weight' in k:
            if hf_gemma.config.tie_word_embeddings:
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

            if tensorrt_llm_gemma.config.use_parallel_embedding:
                v = split(v, mapping.tp_size, mapping.tp_rank,
                          tensorrt_llm_gemma.config.embedding_sharding_dim)
            if mapping.is_first_pp_rank():
                weights['transformer.vocab_embedding.weight'] = torch_to_numpy(
                    numpy_to_torch(v).to(torch.float32) *
                    np.sqrt(tensorrt_llm_gemma.config.hidden_size))
        elif 'model.norm.weight' in k:
            if mapping.is_last_pp_rank():
                weights['transformer.ln_f.weight'] = torch_to_numpy(
                    numpy_to_torch(v) + 1.0)

        elif 'lm_head.weight' in k:
            if mapping.is_last_pp_rank():
                if vocab_size % mapping.tp_size != 0:
                    # padding
                    vocab_size_padded = tensorrt_llm_gemma.lm_head.out_features * mapping.tp_size
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
                    idx)] = torch_to_numpy(numpy_to_torch(v) + 1.0)
            elif 'post_attention_layernorm.weight' in k:
                weights['transformer.layers.{}.post_layernorm.weight'.format(
                    idx)] = torch_to_numpy(numpy_to_torch(v) + 1.0)

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
                # dst = tensorrt_llm_gemma.layers[idx].attention.dense.weight
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
                # dst = tensorrt_llm_gemma.layers[idx].mlp.proj.weight
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
                # dst = tensorrt_llm_gemma.layers[idx].mlp.fc.weight
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

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')
    return weights


def quantize_fp8_weights(weights, num_layers, mapping):

    def get_scaling_factor(weight):
        amax = weight.max()
        scale = 448.0 / amax
        return scale

    layers_range = mapping.pp_layers(num_layers)
    scaling_factors = {}
    scaled_weights = {}
    trt_llm_prefix = "transformer.layers"
    for l in layers_range:
        # attention.qkv.weight
        for name in [
                "attention.qkv", "attention.dense", "mlp.fc", "mlp.gate",
                "mlp.proj"
        ]:
            trt_llm_name = ".".join((trt_llm_prefix, str(l), name, "weight"))
            scale_name = ".".join(
                (trt_llm_prefix, str(l), name, "weights_scaling_factor"))
            weight = weights[trt_llm_name].float()
            dtype = weights[trt_llm_name].dtype
            scale = get_scaling_factor(weight)
            scaled_weights[trt_llm_name] = (weight *
                                            scale).to(dtype).contiguous()
            scaling_factors[scale_name] = numpy_to_torch(
                np.asarray([1 / scale]).astype(np.float32))
    return scaling_factors


def load_from_fp8_gemma(quant_ckpt_path: str, num_layers: int, mapping: Mapping,
                        fp8_kv_cache: bool, weight_scales: dict):
    """
    Get the fp8 scaling factors.
    """
    fake_fp8_sf_dt = torch.float32

    if quant_ckpt_path is not None and os.path.isfile(quant_ckpt_path):
        fp8_gemma = np.load(quant_ckpt_path)
    else:
        fp8_gemma = None
        logger.info(
            f"There is not quantized checkpoint, use dummy fp8 scaling factors instead."
        )
    weights = {}

    def get_fp8_gemma(name):
        if fp8_gemma is not None:
            return fp8_gemma[name]
        else:
            return torch.tensor([1.0], dtype=fake_fp8_sf_dt).numpy()

    layers_range = mapping.pp_layers(num_layers)
    for l in layers_range:
        prefix = f'_np:layers:{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'

        weights[f'{tllm_prex}.attention.qkv.activation_scaling_factor'] = max(
            get_fp8_gemma(
                f'{prefix}:attention:qkv:q:activation_scaling_factor'),
            get_fp8_gemma(
                f'{prefix}:attention:qkv:k:activation_scaling_factor'),
            get_fp8_gemma(
                f'{prefix}:attention:qkv:v:activation_scaling_factor'))
        weights[f'{tllm_prex}.attention.qkv.weights_scaling_factor'] = max(
            get_fp8_gemma(f'{prefix}:attention:qkv:q:weights_scaling_factor'),
            get_fp8_gemma(f'{prefix}:attention:qkv:k:weights_scaling_factor'),
            get_fp8_gemma(f'{prefix}:attention:qkv:v:weights_scaling_factor'))
        weights[
            f'{tllm_prex}.attention.dense.activation_scaling_factor'] = get_fp8_gemma(
                f'{prefix}:attention:dense:activation_scaling_factor')
        weights[
            f'{tllm_prex}.attention.dense.weights_scaling_factor'] = get_fp8_gemma(
                f'{prefix}:attention:dense:weights_scaling_factor')

        weights[
            f'{tllm_prex}.mlp.fc.activation_scaling_factor'] = get_fp8_gemma(
                f'{prefix}:mlp:fc:activation_scaling_factor')
        weights[f'{tllm_prex}.mlp.fc.weights_scaling_factor'] = get_fp8_gemma(
            f'{prefix}:mlp:fc:weights_scaling_factor')

        weights[
            f'{tllm_prex}.mlp.gate.activation_scaling_factor'] = get_fp8_gemma(
                f'{prefix}:mlp:gate:activation_scaling_factor')
        weights[f'{tllm_prex}.mlp.gate.weights_scaling_factor'] = get_fp8_gemma(
            f'{prefix}:mlp:gate:weights_scaling_factor')

        weights[
            f'{tllm_prex}.mlp.proj.activation_scaling_factor'] = get_fp8_gemma(
                f'{prefix}:mlp:proj:activation_scaling_factor')
        weights[f'{tllm_prex}.mlp.proj.weights_scaling_factor'] = get_fp8_gemma(
            f'{prefix}:mlp:proj:weights_scaling_factor')

        if fp8_kv_cache:
            # Not calibrating KV cache.
            scaling_factor = 1.0
            weights[
                f'{tllm_prex}.attention.kv_cache_scaling_factor'] = torch.tensor(
                    [scaling_factor], dtype=fake_fp8_sf_dt).numpy()
            if fp8_gemma is None:
                weights.update(weight_scales)

    for key in weights:
        if isinstance(weights[key], np.ndarray):
            weights[key] = numpy_to_torch(weights[key])
    return weights


def dummy_scaling_factor_sq(weights):
    for name in list(weights):
        if any([
                _name in name for _name in [
                    'mlp.proj.weight', 'mlp.gate.weight', 'mlp.fc.weight',
                    'attention.qkv.weight', 'attention.dense.weight'
                ]
        ]):
            print("Processing:", name)
            weight = weights[name]
            out_dim, in_dim = weight.shape
            weights_scaling_factor = (np.abs(weight).max(1, keepdims=True) /
                                      127.)
            prequant_scaling_factor = np.ones([in_dim], dtype=weight.dtype)
            activation_scaling_factor = np.array([0.1], dtype=np.float32)
            int_weight = (weight / weights_scaling_factor).round().astype(
                np.int8)
            weights[name.replace(
                'weight', 'prequant_scaling_factor')] = prequant_scaling_factor
            weights[name.replace(
                'weight',
                'weights_scaling_factor')] = weights_scaling_factor.astype(
                    np.float32).squeeze(1)
            weights[name.replace(
                'weight',
                'activation_scaling_factor')] = activation_scaling_factor
            weights[name] = int_weight
    return weights


def dummy_scaling_factor_kv_cache(weights):
    for name in list(weights):
        if 'attention.qkv.weight' in name:
            kv_cache_scaling_factor = np.array([0.1], dtype=np.float32)
            weights[name.replace(
                'qkv.weight',
                'kv_cache_scaling_factor')] = kv_cache_scaling_factor


def dummy_weights_awq(weights, precision, trt_llm_config, group_size):
    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    use_fp8_kv_cache = trt_llm_config.quant_mode.has_fp8_kv_cache()
    use_int8_kv_cache = trt_llm_config.quant_mode.has_int8_kv_cache()
    num_layers = trt_llm_config.num_hidden_layers
    for name in list(weights):
        if any([
                _name in name for _name in [
                    'mlp.proj.weight', 'mlp.gate.weight', 'mlp.fc.weight',
                    'attention.qkv.weight', 'attention.dense.weight'
                ]
        ]):
            print("Processing:", name)
            weight = np.ascontiguousarray(weights[name].T)
            in_dim, out_dim = weight.shape
            scale = np.amax(weight) / 7
            weights_scaling_factor = np.ones([out_dim, in_dim // group_size
                                              ]) * scale.astype(np.float32)
            weight_smoothed = (weight.astype(np.float32) / scale).astype(
                np.int8)
            weight_smoothed[weight_smoothed < -8] = -8
            weight_smoothed[weight_smoothed > 7] = 7
            prequant_scaling_factor = np.ones([in_dim], dtype=weight.dtype)
            weights[name] = packer(
                torch.from_numpy(weight_smoothed)).T.contiguous().numpy()
            weights[name.replace(
                'weight', 'prequant_scaling_factor')] = prequant_scaling_factor
            weights[name.replace(
                'weight',
                'weights_scaling_factor')] = weights_scaling_factor.astype(
                    weight.dtype)
            if precision == "w4a8_awq":
                alpha = np.array([1], dtype=np.float32)
                weights[name.replace('weight', 'alpha')] = alpha
    if use_fp8_kv_cache or use_int8_kv_cache:
        for l in range(num_layers):
            t = np.array([1], dtype=np.float32)
            weights[
                f"transformer.layers.{l}.attention.kv_cache_scaling_factor"] = t

    return weights
