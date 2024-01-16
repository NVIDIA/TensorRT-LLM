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
import logging
import time
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import (numpy_to_torch, pad_vocab_size,
                                 str_dtype_to_np, str_dtype_to_torch,
                                 torch_to_numpy)
from tensorrt_llm.functional import is_gated_activation
from tensorrt_llm.models import GPTLMHeadModel
from tensorrt_llm.quantization import QuantMode

LOGGER = logging.getLogger(__name__)


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


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    elif len(v.shape) == 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])
    return None


def parse_ft_config(ini_file):
    gpt_config = configparser.ConfigParser()
    gpt_config.read(ini_file)

    n_embd = gpt_config.getint('gpt', 'n_embd')
    n_head = gpt_config.getint('gpt', 'n_head')
    n_layer = gpt_config.getint('gpt', 'n_layer')
    n_positions = gpt_config.getint('gpt', 'n_positions')
    vocab_size = gpt_config.getint('gpt', 'vocab_size')
    do_layer_norm_before = gpt_config.getboolean('gpt',
                                                 'do_layer_norm_before',
                                                 fallback=True)
    rotary_base = gpt_config.getfloat('gpt', 'rotary_base', fallback=None)
    rotary_scaling_type = gpt_config.get('gpt',
                                         'rotary_scaling_type',
                                         fallback=None)
    rotary_scaling_factor = gpt_config.get('gpt',
                                           'rotary_scaling_factor',
                                           fallback=None)
    if rotary_scaling_type is None:
        if rotary_scaling_factor is not None:
            raise ValueError(
                f"'rotary_scaling_factor={rotary_scaling_factor}' is found in ini "
                f"config file {ini_file}, whereas 'rotary_scaling_type' is missing "
                f"in the config. The 'rotary_scaling_factor' will be ignored and "
                f"rotary scaling will not be used.")
        rotary_scaling = None
    else:
        if rotary_scaling_factor is None:
            raise ValueError(
                f"'rotary_scaling_factor={rotary_scaling_factor}' was not found "
                f"in ini config file {ini_file}, whereas 'rotary_scaling_type' is "
                f"provided  and equals {repr(rotary_scaling_type)}.")
        rotary_scaling = [rotary_scaling_type, rotary_scaling_factor]
    rotary_pct = gpt_config.getfloat('gpt', 'rotary_pct', fallback=None)
    hidden_act = gpt_config.get('gpt', 'activation_function')
    bias = gpt_config.getboolean('gpt', 'bias', fallback=True)
    inter_size = gpt_config.getint('gpt', 'intermediate_size', fallback=None)
    dtype = gpt_config.get('gpt', 'storage_dtype', fallback='float32')

    if inter_size is None:
        inter_size = 4 * n_embd

    multi_query_mode = gpt_config.getboolean('gpt',
                                             'multi_query_mode',
                                             fallback=False)
    prompt_num_tasks = gpt_config.getint('gpt', 'prompt_num_tasks', fallback=0)
    prompt_max_vocab_size = gpt_config.getint('gpt',
                                              'prompt_max_vocab_size',
                                              fallback=0)
    return {
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "n_positions": n_positions,
        "vocab_size": vocab_size,
        "do_layer_norm_before": do_layer_norm_before,
        "hidden_act": hidden_act,
        "rotary_pct": rotary_pct,
        "rotary_base": rotary_base,
        "rotary_scaling": rotary_scaling,
        "bias": bias,
        "inter_size": inter_size,
        "multi_query_mode": multi_query_mode,
        "dtype": dtype,
        "prompt_num_tasks": prompt_num_tasks,
        "prompt_max_vocab_size": prompt_max_vocab_size
    }


def check_embedding_share(dir_path):
    share_embedding_table = False
    lm_file = dir_path + '/' + 'model.lm_head.weight.bin'
    if not Path(lm_file).exists():
        share_embedding_table = True
    return share_embedding_table


def load_from_ft(tensorrt_llm_gpt: GPTLMHeadModel,
                 dir_path,
                 rank=0,
                 tensor_parallel=1,
                 dtype='float32',
                 use_parallel_embedding=False,
                 sharding_dim=0,
                 share_embedding_table=False,
                 scaling_factors=None):
    tensorrt_llm.logger.info('Loading weights from FT...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_gpt, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    _parsed_params = parse_ft_config(Path(dir_path) / 'config.ini')
    n_embd = _parsed_params["n_embd"]
    n_head = _parsed_params["n_head"]
    n_layer = _parsed_params["n_layer"]
    n_positions = _parsed_params["n_positions"]
    vocab_size = _parsed_params["vocab_size"]
    do_layer_norm_before = _parsed_params["do_layer_norm_before"]
    hidden_act = _parsed_params["hidden_act"]
    bias = _parsed_params["bias"]
    inter_size = _parsed_params["inter_size"]
    multi_query_mode = _parsed_params["multi_query_mode"]

    np_dtype = str_dtype_to_np(dtype)

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
            t = fromfile(dir_path, f"{basename}scale_w_quant_orig.{suffix}",
                         col_shape, np.float32)
            module.per_channel_scale.value = t
        else:
            t = fromfile(dir_path, f"{basename}scale_x_orig_quant.bin", [1],
                         np.float32)
            pre_scale_weight.value = t
            t = fromfile(dir_path, f"{basename}scale_y_accum_quant.{suffix}",
                         col_shape, np.float32)
            module.per_channel_scale.value = t
            t = fromfile(dir_path, f"{basename}scale_y_quant_orig.bin", [1, 1],
                         np.float32)
            module.act_scale.value = t

    # Determine the quantization mode.
    quant_mode = getattr(tensorrt_llm_gpt, "quant_mode", QuantMode(0))
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

    #Enable FP8 Gemm
    enable_fp8_qdq = quant_mode.has_fp8_qdq()

    # Debug
    suffix = gen_suffix(rank, use_smooth_quant, quant_per_channel)
    # The type of weights.
    w_type = np_dtype if not use_smooth_quant else np.int8

    pe = fromfile(dir_path, 'model.wpe.bin', [n_positions, n_embd])
    if pe is not None:
        tensorrt_llm_gpt.position_embedding.weight.value = (pe)

    vocab_embedding_weight = fromfile(dir_path, 'model.wte.bin',
                                      [vocab_size, n_embd])
    if not use_parallel_embedding:
        tensorrt_llm_gpt.vocab_embedding.weight.value = vocab_embedding_weight
    else:
        if sharding_dim == 0:
            if vocab_size % tensor_parallel != 0:
                # padding
                vocab_size_padded = pad_vocab_size(
                    tensorrt_llm_gpt.vocab_embedding.num_embeddings,
                    tensor_parallel)
                pad_width = vocab_size_padded - vocab_size
                vocab_embedding_weight = np.pad(vocab_embedding_weight,
                                                ((0, pad_width), (0, 0)),
                                                'constant',
                                                constant_values=0)
        tensorrt_llm_gpt.vocab_embedding.weight.value = np.ascontiguousarray(
            split(vocab_embedding_weight,
                  tensor_parallel,
                  rank,
                  dim=sharding_dim))

    if do_layer_norm_before:
        tensorrt_llm_gpt.ln_f.bias.value = (fromfile(
            dir_path, 'model.final_layernorm.bias.bin'))
        tensorrt_llm_gpt.ln_f.weight.value = (fromfile(
            dir_path, 'model.final_layernorm.weight.bin'))

    # share input embedding
    if not share_embedding_table:
        lm_head_weight = fromfile(dir_path, 'model.lm_head.weight.bin',
                                  [vocab_size, n_embd])
        if lm_head_weight is None:
            lm_head_weight = fromfile(dir_path, 'model.wte.bin',
                                      [vocab_size, n_embd])
        if vocab_size % tensor_parallel != 0:
            # padding
            vocab_size_padded = tensorrt_llm_gpt.lm_head.out_features * tensor_parallel
            pad_width = vocab_size_padded - vocab_size
            lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                    'constant',
                                    constant_values=0)
        tensorrt_llm_gpt.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, tensor_parallel, rank))
    fake_fp8_sf_dt = np.float32
    for i in range(n_layer):
        c_attn_out_dim = (3 * n_embd //
                          tensor_parallel) if not multi_query_mode else (
                              n_embd // tensor_parallel +
                              (n_embd // n_head) * 2)
        gpt_layer = tensorrt_llm_gpt.layers[i]
        gpt_layer.input_layernorm.weight.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.weight.bin'))
        gpt_layer.input_layernorm.bias.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.bias.bin'))
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.weight.' + suffix,
            [n_embd, c_attn_out_dim], w_type)
        if t is not None:
            dst = gpt_layer.attention.qkv.weight
            if use_smooth_quant:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
                set_smoothquant_scale_factors(
                    gpt_layer.attention.qkv,
                    gpt_layer.input_layernorm.scale_to_int,
                    dir_path,
                    'model.layers.' + str(i) + '.attention.query_key_value.',
                    [1, c_attn_out_dim],
                    quant_per_token_dyn,
                    quant_per_channel,
                    rank=rank,
                    is_qkv=True)
            elif use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                    numpy_to_torch(t), plugin_weight_only_quant_type)
                dst.value = torch_to_numpy(processed_torch_weights)
                scales = tensorrt_llm_gpt.layers[
                    i].attention.qkv.per_channel_scale
                scales.value = torch_to_numpy(torch_weight_scales)
            else:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
        if bias:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.bias.' + str(rank) + '.bin')
            if t is not None:
                dst = gpt_layer.attention.qkv.bias
                dst.value = np.ascontiguousarray(t)
        if enable_fp8_qdq:
            tensorrt_llm_gpt.layers[
                i].attention.qkv.activation_scaling_factor.value = np.array(
                    [scaling_factors['qkv_act'][i]], dtype=fake_fp8_sf_dt)
            tensorrt_llm_gpt.layers[
                i].attention.qkv.weights_scaling_factor.value = np.array(
                    [scaling_factors['qkv_weights'][i]], dtype=fake_fp8_sf_dt)
            tensorrt_llm_gpt.layers[
                i].attention.kv_cache_scaling_factor.value = np.array(
                    [scaling_factors['qkv_output'][i]], dtype=np.float32)

        dst = gpt_layer.attention.dense.weight
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attention.dense.weight.' + suffix,
            [n_embd // tensor_parallel, n_embd], w_type)
        if use_smooth_quant:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
            dense_scale = getattr(gpt_layer.attention,
                                  "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                gpt_layer.attention.dense, dense_scale, dir_path,
                'model.layers.' + str(i) + '.attention.dense.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
            # change it to the real smoother if dense layer is applied smooth quant
            gpt_layer.attention.dense.smoother.value = np.ones(
                [1, n_embd // tensor_parallel], dtype=np.float32)
        elif use_weight_only:
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                numpy_to_torch(t), plugin_weight_only_quant_type)
            dst.value = torch_to_numpy(processed_torch_weights)
            scales = tensorrt_llm_gpt.layers[
                i].attention.dense.per_channel_scale
            scales.value = torch_to_numpy(torch_weight_scales)
        else:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        if bias:
            dst = gpt_layer.attention.dense.bias
            dst.value = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.attention.dense.bias.bin')
        if enable_fp8_qdq:
            tensorrt_llm_gpt.layers[
                i].attention.dense.activation_scaling_factor.value = np.array(
                    [scaling_factors['dense_act'][i]], dtype=fake_fp8_sf_dt)
            tensorrt_llm_gpt.layers[
                i].attention.dense.weights_scaling_factor.value = np.array(
                    [scaling_factors['dense_weights'][i]], dtype=fake_fp8_sf_dt)

        dst = gpt_layer.post_layernorm.weight
        dst.value = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.post_attention_layernorm.weight.bin')

        dst = gpt_layer.post_layernorm.bias
        dst.value = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.post_attention_layernorm.bias.bin')
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.dense_h_to_4h.weight.' + suffix,
            [n_embd, inter_size // tensor_parallel], w_type)
        if use_smooth_quant:
            tensorrt_llm_gpt.layers[
                i].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            set_smoothquant_scale_factors(gpt_layer.mlp.fc,
                                          gpt_layer.post_layernorm.scale_to_int,
                                          dir_path,
                                          'model.layers.' + str(i) +
                                          '.mlp.dense_h_to_4h.',
                                          [1, inter_size // tensor_parallel],
                                          quant_per_token_dyn,
                                          quant_per_channel,
                                          rank=rank)
        elif use_weight_only:
            dst = gpt_layer.mlp.fc.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                numpy_to_torch(t), plugin_weight_only_quant_type)
            dst.value = torch_to_numpy(processed_torch_weights)
            scales = gpt_layer.mlp.fc.per_channel_scale
            scales.value = torch_to_numpy(torch_weight_scales)
        else:
            tensorrt_llm_gpt.layers[
                i].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
        if bias:
            gpt_layer.mlp.fc.bias.value = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.mlp.dense_h_to_4h.bias.' + str(rank) + '.bin')
        if is_gated_activation(hidden_act):
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.mlp.dense_h_to_4h.gate.weight.' + str(rank) + '.bin',
                [n_embd, inter_size // tensor_parallel])
            tensorrt_llm_gpt.layers[
                i].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
        if enable_fp8_qdq:
            tensorrt_llm_gpt.layers[
                i].mlp.fc.activation_scaling_factor.value = np.array(
                    [scaling_factors['fc_act'][i]], dtype=fake_fp8_sf_dt)
            tensorrt_llm_gpt.layers[
                i].mlp.fc.weights_scaling_factor.value = np.array(
                    [scaling_factors['fc_weights'][i]], dtype=fake_fp8_sf_dt)

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.dense_4h_to_h.weight.' + suffix,
            [inter_size // tensor_parallel, n_embd], w_type)
        if use_smooth_quant:
            tensorrt_llm_gpt.layers[
                i].mlp.proj.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            proj_scale = getattr(gpt_layer.mlp, "quantization_scaling_factor",
                                 None)
            set_smoothquant_scale_factors(
                gpt_layer.mlp.proj, proj_scale, dir_path,
                'model.layers.' + str(i) + '.mlp.dense_4h_to_h.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
            # change it to the real smoother if proj layer is applied smooth quant
            gpt_layer.mlp.proj.smoother.value = np.ones(
                [1, inter_size // tensor_parallel], dtype=np.float32)
        elif use_weight_only:
            dst = gpt_layer.mlp.proj.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                numpy_to_torch(t), plugin_weight_only_quant_type)
            dst.value = torch_to_numpy(processed_torch_weights)
            scales = gpt_layer.mlp.proj.per_channel_scale
            scales.value = torch_to_numpy(torch_weight_scales)
        else:
            gpt_layer.mlp.proj.weight.value = (np.ascontiguousarray(
                np.transpose(t, [1, 0])))
        if bias:
            gpt_layer.mlp.proj.bias.value = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.mlp.dense_4h_to_h.bias.bin')

        if use_int8_kv_cache:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            gpt_layer.attention.kv_cache_scaling_factor.value = t

        if enable_fp8_qdq:
            tensorrt_llm_gpt.layers[
                i].mlp.proj.activation_scaling_factor.value = np.array(
                    [scaling_factors['proj_act'][i]], dtype=fake_fp8_sf_dt)
            tensorrt_llm_gpt.layers[
                i].mlp.proj.weights_scaling_factor.value = np.array(
                    [scaling_factors['proj_weights'][i]], dtype=fake_fp8_sf_dt)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


def load_from_hf_gpt(tensorrt_llm_gpt: GPTLMHeadModel,
                     hf_gpt,
                     rank=0,
                     tensor_parallel=1,
                     dtype='float32',
                     multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from HF GPT...')
    tik = time.time()

    valid_lm_head_weight = False
    hidden_size = tensorrt_llm_gpt._hidden_size
    head_size = tensorrt_llm_gpt._num_heads // hidden_size
    for k, v in hf_gpt.state_dict().items():
        torch_dtype = str_dtype_to_torch(dtype)
        v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'wte.weight' in k:
            tensorrt_llm_gpt.vocab_embedding.weight.value = v
        elif 'wpe.weight' in k:
            tensorrt_llm_gpt.position_embedding.weight.value = v
        elif 'ln_f.weight' in k:
            tensorrt_llm_gpt.ln_f.weight.value = v
        elif 'ln_f.bias' in k:
            tensorrt_llm_gpt.ln_f.bias.value = v
        elif 'lm_head.weight' in k:
            tensorrt_llm_gpt.lm_head.weight.value = np.ascontiguousarray(
                split(v, tensor_parallel, rank))
            valid_lm_head_weight = True
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if 'ln_1.weight' in k:
                tensorrt_llm_gpt.layers[idx].input_layernorm.weight.value = v
            elif 'ln_1.bias' in k:
                tensorrt_llm_gpt.layers[idx].input_layernorm.bias.value = v
            elif 'attn.c_attn.weight' in k:
                if multi_query_mode:
                    # HF-StarCoder uses torch.nn.Linear
                    w_qkv = v.reshape(hidden_size + 2 * head_size, 3,
                                      hidden_size)
                    w_q, w_kv = np.split(w_qkv, [hidden_size, 2 * head_size])
                    w_q = split(w_q, tensor_parallel, rank)
                    dst = tensorrt_llm_gpt.layers[idx].attention.qkv.weight
                    dst.value = np.ascontiguousarray(np.concatenate(w_q, w_kv))
                else:
                    # HF-GPT uses Conv1D instead of Linear
                    v = v.transpose()
                    dst = tensorrt_llm_gpt.layers[idx].attention.qkv.weight
                    dst.value = np.ascontiguousarray(
                        split(v, tensor_parallel, rank))
            elif 'attn.c_attn.bias' in k:
                if multi_query_mode:
                    v.reshape(hidden_size + 2 * head_size, 3)
                    bias_q, bias_kv = np.split(w_qkv,
                                               [hidden_size, 2 * head_size])
                    bias_q = split(bias_q, tensor_parallel, rank)
                    dst = tensorrt_llm_gpt.layers[idx].attention.qkv.bias
                    dst.value = np.ascontiguousarray(
                        np.concatenate(bias_q, bias_kv))
                else:
                    dst = tensorrt_llm_gpt.layers[idx].attention.qkv.bias
                    dst.value = np.ascontiguousarray(
                        split(v, tensor_parallel, rank))
            elif 'attn.q_attn.weight' in k:
                # Get the corresponding kv_atten.weight:
                # ex: transformer.h.23.attn.kv_attn.weight
                u = hf_gpt.state_dict()[k.replace('q_attn', 'kv_attn')]
                u = u.to(torch_dtype).cpu().numpy(force=True)
                # HF-SantaCoder uses transformer.Conv1D so we transpose to match shape
                # In addition, kv_head must be broadcasted to all ranks so split is not applied
                v = split(v.transpose(), tensor_parallel, rank)  # W_q
                u = u.transpose()  # W_kv
                dst = tensorrt_llm_gpt.layers[idx].attention.qkv.weight
                dst.value = np.ascontiguousarray(np.concatenate((v, u)))
            elif 'attn.q_attn.bias' in k:
                # Get the corresponding kv_atten.bias:
                # ex: transformer.h.23.attn.kv_attn.bias
                u = hf_gpt.state_dict()[k.replace('q_attn', 'kv_attn')]
                u = u.to(torch_dtype).cpu().numpy(force=True)
                v = split(v, tensor_parallel, rank)
                dst = tensorrt_llm_gpt.layers[idx].attention.qkv.bias
                dst.value = np.ascontiguousarray(np.concatenate((v, u)))
            elif 'attn.c_proj.weight' in k:
                v = v.transpose()
                dst = tensorrt_llm_gpt.layers[idx].attention.dense.weight
                dst.value = np.ascontiguousarray(
                    split(v, tensor_parallel, rank, dim=1))
            elif 'attn.c_proj.bias' in k:
                dst = tensorrt_llm_gpt.layers[idx].attention.dense.bias
                dst.value = v
            elif 'ln_2.weight' in k:
                dst = tensorrt_llm_gpt.layers[idx].post_layernorm.weight
                dst.value = v
            elif 'ln_2.bias' in k:
                dst = tensorrt_llm_gpt.layers[idx].post_layernorm.bias
                dst.value = v
            elif 'mlp.c_fc.weight' in k:
                v = v.transpose()
                tensorrt_llm_gpt.layers[
                    idx].mlp.fc.weight.value = np.ascontiguousarray(
                        split(v, tensor_parallel, rank))
            elif 'mlp.c_fc.bias' in k:
                tensorrt_llm_gpt.layers[
                    idx].mlp.fc.bias.value = np.ascontiguousarray(
                        split(v, tensor_parallel, rank))
            elif 'mlp.c_proj.weight' in k:
                v = v.transpose()
                tensorrt_llm_gpt.layers[
                    idx].mlp.proj.weight.value = np.ascontiguousarray(
                        split(v, tensor_parallel, rank, dim=1))
            elif 'mlp.c_proj.bias' in k:
                tensorrt_llm_gpt.layers[idx].mlp.proj.bias.value = v

    if not valid_lm_head_weight:
        # Use wte as lm_head weight to match the load_from_ft implementation.
        lm_head_weight = tensorrt_llm_gpt.vocab_embedding.weight.raw_value
        vocab_size = hf_gpt.config.vocab_size
        if vocab_size % tensor_parallel != 0:
            # padding
            vocab_size_padded = tensorrt_llm_gpt.lm_head.out_features * tensor_parallel
            pad_width = vocab_size_padded - vocab_size
            lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                    'constant',
                                    constant_values=0)
        tensorrt_llm_gpt.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, tensor_parallel, rank))

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
