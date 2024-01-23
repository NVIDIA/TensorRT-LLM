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
from operator import attrgetter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import tensorrt_llm
from tensorrt_llm._utils import (str_dtype_to_np, str_dtype_to_torch,
                                 torch_to_numpy)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import QWenForCausalLM
from tensorrt_llm.quantization import QuantMode


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
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def parse_ft_config(ini_file):
    qwen_config = configparser.ConfigParser()
    qwen_config.read(ini_file)

    vocab_size = qwen_config.getint('qwen', 'vocab_size')
    hidden_size = qwen_config.getint('qwen', 'hidden_size')
    inter_size = qwen_config.getint('qwen', 'intermediate_size', fallback=None)
    num_hidden_layers = qwen_config.getint(
        "qwen",
        "num_hidden_layers",
        fallback=32,
    )
    max_position_embeddings = qwen_config.getint("qwen",
                                                 "max_position_embeddings",
                                                 fallback=8192)
    kv_channels = qwen_config.getint('qwen', 'kv_channels', fallback=128)
    rotary_pct = qwen_config.getfloat('qwen', 'rotary_pct', fallback=0.0)
    rotary_emb_base = qwen_config.getint('qwen',
                                         'rotary_emb_base',
                                         fallback=10000)
    multi_query_mode = qwen_config.getboolean('qwen',
                                              'multi_query_mode',
                                              fallback=False)
    return (vocab_size, hidden_size, inter_size, num_hidden_layers, kv_channels,
            rotary_pct, rotary_emb_base, multi_query_mode,
            max_position_embeddings)


def load_from_ft(tensorrt_llm_qwen: QWenForCausalLM,
                 dir_path,
                 mapping=Mapping(),
                 dtype='float16',
                 share_embedding_table=False,
                 parallel_embedding_table=False,
                 multi_query_mode=False,
                 use_gemm_woq_plugin=True):
    tensorrt_llm.logger.info('Loading weights from FT...')
    tik = time.time()
    quant_mode = getattr(tensorrt_llm_qwen, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    (vocab_size, hidden_size, inter_size, num_hidden_layers, kv_channels,
     rotary_pct, rotary_emb_base, multi_query_mode,
     max_position_embeddings) = parse_ft_config(Path(dir_path) / 'config.ini')
    np_dtype = str_dtype_to_np(dtype)

    def fromfile(dir_path, name, shape=None, dtype=np.float16):
        dtype = np_dtype if dtype is None else dtype
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        else:
            print(f"Warning: {p} not found.")
        return None

    def set_smoothquant_scale_factors(
        module,
        pre_scale_weight,
        dir_path,
        basename,
        shape,
        per_tok_dyn,
        per_channel,
        is_qkv=False,
        rank=None,
    ):
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

    def set_smoother(module, dir_path, base_name, shape, rank):
        suffix = f"{rank}.bin"
        t = fromfile(dir_path, f"{base_name}.smoother.{suffix}", shape,
                     np.float32)
        module.smoother.value = t

    # Determine the quantization mode.
    quant_mode = getattr(tensorrt_llm_qwen, "quant_mode", QuantMode(0))
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
        tensorrt_llm_qwen.embedding.vocab_embedding.weight.value = (fromfile(
            dir_path, 'vocab_embedding.weight.bin', [vocab_size, hidden_size]))

    if mapping.is_last_pp_rank():
        tensorrt_llm_qwen.ln_f.weight.value = (fromfile(dir_path,
                                                        'ln_f.weight.bin'))

    lm_head_weight = fromfile(dir_path, 'lm_head.weight.bin',
                              [vocab_size, hidden_size])

    if vocab_size % mapping.tp_size != 0:
        # padding
        vocab_size_padded = tensorrt_llm_qwen.lm_head.out_features * mapping.tp_size
        pad_width = vocab_size_padded - vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                'constant',
                                constant_values=0)
    if mapping.is_last_pp_rank():
        tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, mapping.tp_size, mapping.tp_rank))

    layers_range = list(
        range(mapping.pp_rank * tensorrt_llm_qwen.num_layers,
              (mapping.pp_rank + 1) * tensorrt_llm_qwen.num_layers, 1))

    for i in layers_range:
        c_attn_out_dim = (3 * hidden_size //
                          mapping.tp_size) if not multi_query_mode else (
                              hidden_size // mapping.tp_size +
                              (hidden_size // num_hidden_layers) * 2)

        tensorrt_llm_qwen.layers[i].ln_1.weight.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.ln_1.weight.bin')

        dst = tensorrt_llm_qwen.layers[i].ln_2.weight
        dst.value = fromfile(dir_path,
                             'model.layers.' + str(i) + '.ln_2.weight.bin')

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attention.qkv.weight.' + suffix,
            [hidden_size, c_attn_out_dim], w_type)
        if t is not None:
            dst = tensorrt_llm_qwen.layers[i].attention.qkv.weight
            if use_smooth_quant:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
                set_smoothquant_scale_factors(
                    tensorrt_llm_qwen.layers[i].attention.qkv,
                    tensorrt_llm_qwen.layers[i].ln_1.scale_to_int,
                    dir_path,
                    'model.layers.' + str(i) + '.attention.qkv.',
                    [1, c_attn_out_dim],
                    quant_per_token_dyn,
                    quant_per_channel,
                    rank=mapping.tp_rank,
                    is_qkv=True)
            elif use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(t), plugin_weight_only_quant_type)
                if not use_gemm_woq_plugin:
                    dst.value = torch.tensor(t).numpy().astype(
                        str_dtype_to_np(dtype))
                else:
                    dst.value = processed_torch_weights.numpy()
                scales = tensorrt_llm_qwen.layers[
                    i].attention.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_qwen.layers[i].attention.qkv.bias
        t = fromfile(
            dir_path, 'model.layers.' + str(i) + '.attention.qkv.bias.' +
            str(mapping.tp_rank) + '.bin', [c_attn_out_dim])
        dst.value = np.ascontiguousarray(t)

        dst = tensorrt_llm_qwen.layers[i].attention.dense.weight
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attention.dense.weight.' + suffix,
            [hidden_size // mapping.tp_size, hidden_size], w_type)
        if use_smooth_quant:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
            dense_scale = getattr(tensorrt_llm_qwen.layers[i].attention,
                                  "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].attention.dense,
                dense_scale,
                dir_path,
                'model.layers.' + str(i) + '.attention.dense.',
                [1, hidden_size],
                quant_per_token_dyn,
                quant_per_channel,
            )
            set_smoother(tensorrt_llm_qwen.layers[i].attention.dense, dir_path,
                         'model.layers.' + str(i) + '.attention.dense',
                         [1, hidden_size // mapping.tp_size], mapping.tp_rank)

        elif use_weight_only:
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            if not use_gemm_woq_plugin:
                dst.value = torch.tensor(t).numpy().astype(
                    str_dtype_to_np(dtype))
            else:
                dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_qwen.layers[
                i].attention.dense.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.w1.weight.' + suffix,
                     [hidden_size, inter_size // mapping.tp_size // 2], w_type)
        if use_smooth_quant:
            tensorrt_llm_qwen.layers[
                i].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].mlp.gate,
                tensorrt_llm_qwen.layers[i].ln_2.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.w1.',
                [1, inter_size // mapping.tp_size // 2],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_qwen.layers[i].mlp.gate.weight
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            if not use_gemm_woq_plugin:
                dst.value = torch.tensor(t).numpy().astype(
                    str_dtype_to_np(dtype))
            else:
                dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_qwen.layers[i].mlp.gate.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_qwen.layers[
                i].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.w2.weight.' + suffix,
                     [hidden_size, inter_size // mapping.tp_size // 2], w_type)
        if use_smooth_quant:
            tensorrt_llm_qwen.layers[
                i].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].mlp.fc,
                tensorrt_llm_qwen.layers[i].ln_2.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.w2.',
                [1, inter_size // mapping.tp_size // 2],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_qwen.layers[i].mlp.fc.weight
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            if not use_gemm_woq_plugin:
                dst.value = torch.tensor(t).numpy().astype(
                    str_dtype_to_np(dtype))
            else:
                dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_qwen.layers[i].mlp.fc.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_qwen.layers[
                i].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.c_proj.weight.' + suffix,
                     [inter_size // mapping.tp_size // 2, hidden_size], w_type)
        if use_smooth_quant:
            tensorrt_llm_qwen.layers[
                i].mlp.proj.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            proj_scale = getattr(tensorrt_llm_qwen.layers[i].mlp,
                                 "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].mlp.proj, proj_scale, dir_path,
                'model.layers.' + str(i) + '.mlp.c_proj.', [1, hidden_size],
                quant_per_token_dyn, quant_per_channel)
            set_smoother(tensorrt_llm_qwen.layers[i].mlp.proj, dir_path,
                         'model.layers.' + str(i) + '.mlp.c_proj',
                         [1, inter_size // mapping.tp_size // 2],
                         mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_qwen.layers[i].mlp.proj.weight
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            if not use_gemm_woq_plugin:
                dst.value = torch.tensor(t).numpy().astype(
                    str_dtype_to_np(dtype))
            else:
                dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_qwen.layers[i].mlp.proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_qwen.layers[
                i].mlp.proj.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        if use_int8_kv_cache:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.qkv.scale_y_quant_orig.bin', [1], np.float32)
            tensorrt_llm_qwen.layers[
                i].attention.kv_cache_scaling_factor.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


def load_from_hf_qwen(tensorrt_llm_qwen: tensorrt_llm.models.QWenForCausalLM,
                      hf_qwen,
                      mapping=Mapping(),
                      max_position_embeddings=8192,
                      rotary_emb_base=10000,
                      kv_channels=128,
                      dtype="float32",
                      multi_query_mode=False,
                      use_gemm_woq_plugin=True):
    tensorrt_llm.logger.info('Loading weights from HF QWen...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_qwen, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    model_params = dict(hf_qwen.named_parameters())
    torch_dtype = str_dtype_to_torch(dtype)
    for k, v in tqdm(model_params.items(),
                     total=len(model_params),
                     ncols=80,
                     desc="Converting..."):
        if 'visual' in k:
            continue
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'transformer.wte.weight' in k:
            tensorrt_llm_qwen.embedding.vocab_embedding.weight.value = v
        elif 'transformer.ln_f.weight' in k:
            tensorrt_llm_qwen.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
                split(v, mapping.tp_size, mapping.tp_rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx >= tensorrt_llm_qwen.num_layers:
                continue
            if 'ln_1.weight' in k:
                tensorrt_llm_qwen.layers[idx].ln_1.weight.value = v
            elif 'ln_2.weight' in k:
                tensorrt_llm_qwen.layers[idx].ln_2.weight.value = v
            elif 'attn.c_attn.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].attention.qkv.weight
                if multi_query_mode:
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
                    processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].attention.qkv.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'attn.c_attn.bias' in k:
                dst = tensorrt_llm_qwen.layers[idx].attention.qkv.bias
                if multi_query_mode:
                    assert isinstance(v, list) and len(v) == 3
                    wq = split(v[0], mapping.tp_size, mapping.tp_rank)
                    wk = split(v[1], mapping.tp_size, mapping.tp_rank)
                    wv = split(v[2], mapping.tp_size, mapping.tp_rank)
                    split_v = np.concatenate((wq, wk, wv))
                else:
                    q_emb = v.shape[0] // 3
                    v = v.reshape(3, q_emb)
                    split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // mapping.tp_size))
                dst.value = np.ascontiguousarray(split_v)
            elif 'attn.c_proj.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].attention.dense.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].attention.dense.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.w1.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.gate.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].mlp.gate.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.w2.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.fc.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].mlp.fc.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.c_proj.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.proj.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].mlp.proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            else:
                print("unknown key: ", k)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return


def load_from_gptq_qwen(
        tensorrt_llm_qwen: QWenForCausalLM,
        quant_ckpt_path,
        mapping=Mapping(),
        dtype="float16",
):
    tensorrt_llm.logger.info(
        "loading weights from groupwise gptq qwen safetensors...")
    tik = time.time()

    if quant_ckpt_path.endswith(".safetensors"):
        groupwise_qweight_safetensors = safe_open(quant_ckpt_path,
                                                  framework="pt",
                                                  device='cpu')
        model_params = {
            key: groupwise_qweight_safetensors.get_tensor(key)
            for key in groupwise_qweight_safetensors.keys()
        }
    elif quant_ckpt_path.endswith(".pt"):
        model_params = torch.load(quant_ckpt_path,
                                  map_location=torch.device("cpu"))
    else:
        if Path(quant_ckpt_path).is_dir():
            model = AutoModelForCausalLM.from_pretrained(
                quant_ckpt_path, device_map="auto",
                trust_remote_code=True).eval().cpu()
            model_params = {k: v for k, v in model.state_dict().items()}
            torch.cuda.empty_cache()
            del model
        else:
            raise ValueError("quantized checkpoint format not supported!")

    def unpack_int32_into_int8(w_packed):
        # unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                                 w_packed_int4x2.shape[1] * 2,
                                 dtype=torch.int8)
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.contiguous()

    def preprocess_groupwise_weight_params(
        weight_name,
        qweight_int32=None,
        qzeros_int32=None,
        scales_fp16=None,
    ):
        if weight_name is not None:
            qweight_int32 = model_params[weight_name].cpu()
            qzeros_int32 = model_params[weight_name[:-7] + "qzeros"].cpu()
            scales_fp16 = model_params[weight_name[:-7] + "scales"].cpu()

        UINT4_TO_INT4_FLAG = 1
        GPTQ_FLAG = 1
        packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
        preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm

        qweight_unpacked_int8 = (
            unpack_int32_into_int8(qweight_int32.T).T.contiguous() - 8)
        qweight_interleaved = preprocessor(packer(qweight_unpacked_int8),
                                           torch.quint4x2).view(torch.float16)
        # zeros = zeros * scales
        qzeros_unpacked_int32 = unpack_int32_into_int8(qzeros_int32)

        zeros_x_scales_fp16 = (-qzeros_unpacked_int32 + 8 * UINT4_TO_INT4_FLAG -
                               GPTQ_FLAG) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        # return processed interleaved weight, original scales and zeros * scales
        return (
            qweight_interleaved.contiguous(),  # dtype: int8
            zeros_x_scales_fp16.contiguous(),  # dtype: float16
            scales_fp16.contiguous(),  # dtype: float16
        )

    layer_ids = [
        extract_layer_idx(key) for key in model_params.keys()
        if 'visual' not in key
    ]  #exclude 'visual' for Qwen-VL case
    layer_ids = [
        int(layer_idx) for layer_idx in layer_ids if layer_idx is not None
    ]
    num_hidden_layers = max(layer_ids) + 1
    suffixs = ["qweight", "qzeros", "scales"]

    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(
            mapping.pp_rank * layers_per_pipeline_stage,
            (mapping.pp_rank + 1) * layers_per_pipeline_stage,
            1,
        ))
    torch_dtype = str_dtype_to_torch(dtype)
    for layer in tqdm(layers_range,
                      ncols=80,
                      desc="loading attention weight..."):
        prefix = f"transformer.h.{layer}.attn."
        split_qkv_suf = []
        for suf in suffixs:
            qkv_part = model_params[prefix + "c_attn." + suf].cpu()
            q_emb = qkv_part.shape[1] // 3
            model_emb = qkv_part.shape[0]
            qkv_part = qkv_part.reshape(model_emb, 3, q_emb)
            split_qkv = split(qkv_part, mapping.tp_size, mapping.rank, dim=2)
            split_qkv = split_qkv.reshape(model_emb,
                                          3 * (q_emb // mapping.tp_size))
            if isinstance(split_qkv, np.ndarray):
                split_qkv = torch.from_numpy(split_qkv)
            # dype: int32, int32, float16
            split_qkv_suf.append(split_qkv)

        idx = layer - mapping.pp_rank * layers_per_pipeline_stage
        th_bias = model_params[prefix + "c_attn.bias"].to(
            torch_dtype).cpu().contiguous()

        q_emb = th_bias.shape[0] // 3
        th_bias = th_bias.reshape(3, q_emb)
        split_v = split(th_bias, mapping.tp_size, mapping.rank, dim=1)
        split_v = split_v.reshape(3 * (q_emb // mapping.tp_size))

        tensorrt_llm_qwen.layers[
            idx].attention.qkv.bias.value = np.ascontiguousarray(split_v)

        th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
            None,
            split_qkv_suf[0],
            split_qkv_suf[1],
            split_qkv_suf[2],
        )
        tensorrt_llm_qwen.layers[
            idx].attention.qkv.weight.value = th_qweight.numpy()
        tensorrt_llm_qwen.layers[idx].attention.qkv.zero.value = th_zero.to(
            torch_dtype).numpy()
        tensorrt_llm_qwen.layers[
            idx].attention.qkv.weights_scaling_factor.value = th_scale.to(
                torch_dtype).numpy()

    for k, v in tqdm(model_params.items(),
                     ncols=80,
                     desc="loading other weight..."):
        if 'visual' in k:
            continue
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())

        if "transformer.wte.weight" in k:
            if mapping.is_first_pp_rank():
                tensorrt_llm.logger.info(f"converting: {k}")
                tensorrt_llm_qwen.embedding.vocab_embedding.weight.value = v
        elif "transformer.ln_f.weight" in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_qwen.ln_f.weight.value = v
        elif "lm_head.weight" in k:
            tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
                split(v, mapping.tp_size, mapping.rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx not in layers_range:
                continue
            idx = idx - mapping.pp_rank * layers_per_pipeline_stage

            if "ln_1.weight" in k:
                tensorrt_llm_qwen.layers[idx].ln_1.weight.value = v
            elif "ln_2.weight" in k:
                tensorrt_llm_qwen.layers[idx].ln_2.weight.value = v
            elif 'post_attention_layernorm.weight' in k:
                tensorrt_llm_qwen.layers[idx].post_layernorm.weight.value = v
            elif "attn.c_proj.qweight" in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[0] // mapping.tp_size,
                                      dim=0)[mapping.tp_rank]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_qwen.layers[
                    idx].attention.dense.weight.value = th_qweight.numpy()
                tensorrt_llm_qwen.layers[
                    idx].attention.dense.zero.value = th_zero.to(
                        torch_dtype).numpy()
                tensorrt_llm_qwen.layers[
                    idx].attention.dense.weights_scaling_factor.value = th_scale.to(
                        torch_dtype).numpy()
            elif "mlp.w1.qweight" in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[1] // mapping.tp_size,
                                      dim=1)[mapping.tp_rank]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_qwen.layers[
                    idx].mlp.gate.weight.value = th_qweight.numpy()
                tensorrt_llm_qwen.layers[idx].mlp.gate.zero.value = th_zero.to(
                    torch_dtype).numpy()
                tensorrt_llm_qwen.layers[
                    idx].mlp.gate.weights_scaling_factor.value = th_scale.to(
                        torch_dtype).numpy()
            elif "mlp.c_proj.qweight" in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[0] // mapping.tp_size,
                                      dim=0)[mapping.tp_rank]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_qwen.layers[
                    idx].mlp.proj.weight.value = th_qweight.numpy()
                tensorrt_llm_qwen.layers[idx].mlp.proj.zero.value = th_zero.to(
                    torch_dtype).numpy()
                tensorrt_llm_qwen.layers[
                    idx].mlp.proj.weights_scaling_factor.value = th_scale.to(
                        torch_dtype).numpy()
            elif "mlp.w2.qweight" in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[1] // mapping.tp_size,
                                      dim=1)[mapping.tp_rank]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_qwen.layers[
                    idx].mlp.fc.weight.value = th_qweight.numpy()
                tensorrt_llm_qwen.layers[idx].mlp.fc.zero.value = th_zero.to(
                    torch_dtype).numpy()
                tensorrt_llm_qwen.layers[
                    idx].mlp.fc.weights_scaling_factor.value = th_scale.to(
                        torch_dtype).numpy()
            elif 'attn.c_attn.bias' in k:
                dst = tensorrt_llm_qwen.layers[idx].attention.qkv.bias
                q_emb = v.shape[0] // 3
                v = v.reshape(3, q_emb)
                split_v = split(v, mapping.tp_size, mapping.rank, dim=1)
                split_v = split_v.reshape(3 * (q_emb // mapping.tp_size))
                dst.value = np.ascontiguousarray(split_v)

    tok = time.time()
    t = time.strftime("%h:%m:%s", time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f"weights loaded. total time: {t}")


def load_from_awq_qwen(tensorrt_llm_qwen: QWenForCausalLM,
                       quant_ckpt_path,
                       quantize_lm_head=False,
                       mapping=Mapping(),
                       dtype="float16"):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise AWQ Qwen safetensors...')
    tik = time.time()

    if quant_ckpt_path.endswith(".safetensors"):
        groupwise_qweight_safetensors = safe_open(quant_ckpt_path,
                                                  framework="pt",
                                                  device=0)
        model_params = {
            key: groupwise_qweight_safetensors.get_tensor(key)
            for key in groupwise_qweight_safetensors.keys()
        }
    elif quant_ckpt_path.endswith(".pt"):
        model_params = torch.load(quant_ckpt_path,
                                  map_location=torch.device('cpu'))
    else:
        assert False, "Quantized checkpoint format not supported!"

    group_size = model_params["transformer.h.0.attn.c_proj.weight"].numel(
    ) // model_params[
        "transformer.h.0.attn.c_proj.weight_quantizer._amax"].numel()

    awq_block_names = [
        "ln_1.weight",
        "ln_2.weight",
    ]

    tensorrt_llm_block_names = [
        "ln_1.weight",
        "ln_2.weight",
    ]

    getattr(tensorrt_llm_qwen, 'quant_mode', QuantMode(0))

    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def AWQ_quantize_pack_preprocess(weight, scale):
        scale = scale.repeat_interleave(group_size, dim=0)
        weight = weight / scale  # fp16 -> int8
        qweight_int8 = torch.clamp(torch.round(weight.cuda()).char(), -8, 7)
        int4_weight = packer(qweight_int8.cpu())
        int4_weight = preprocessor(int4_weight,
                                   torch.quint4x2)  # int8 save as uint4
        return int4_weight.view(torch.float16).cpu().numpy()

    def process_and_assign_attn_weight(model_params, mPrefix, mOp, tp_dim=0):
        weight = model_params[mPrefix + ".weight"].to(torch_dtype)
        q_emb = weight.shape[0] // 3
        model_emb = weight.shape[1]
        weight = weight.reshape(3, q_emb, model_emb)
        # [k, n] = weight.shape
        split_v = split(weight, mapping.tp_size, mapping.rank, dim=tp_dim)
        split_v = split_v.reshape(3 * (q_emb // mapping.tp_size), model_emb)
        amax = model_params[mPrefix + ".weight_quantizer._amax"].reshape(
            (q_emb * 3, int(model_emb / group_size))).to(torch_dtype)
        amax = amax.reshape(3, q_emb, model_emb // group_size)
        split_amax = split(amax, mapping.tp_size, mapping.rank, dim=tp_dim)
        split_amax = split_amax.reshape(3 * (q_emb // mapping.tp_size),
                                        model_emb // group_size)
        if isinstance(split_v, np.ndarray):
            split_v = torch.from_numpy(split_v)
        if isinstance(split_amax, np.ndarray):
            split_amax = torch.from_numpy(split_amax)
        split_v = split_v.T.contiguous()
        split_amax = split_amax.T.contiguous()
        pre_quant_scale = model_params[
            mPrefix + ".input_quantizer._pre_quant_scale"].reshape(
                (1, model_emb)).to(torch_dtype)
        split_scale = split_amax / 8.0
        mOp.weight.value = AWQ_quantize_pack_preprocess(split_v, split_scale)
        mOp.weights_scaling_factor.value = split_scale.cpu().numpy()
        mOp.prequant_scaling_factor.value = pre_quant_scale.cpu().numpy()

    def process_and_assign_weight(model_params, mPrefix, mOp, tp_dim=0):
        weight = model_params[mPrefix + ".weight"].T.contiguous()
        [k, n] = weight.shape
        weight = torch_split(weight, tp_dim)
        amax = model_params[mPrefix + ".weight_quantizer._amax"].reshape(
            (n, int(k / group_size))).T.contiguous()
        amax = torch_split(amax, tp_dim)
        pre_quant_scale = model_params[
            mPrefix + ".input_quantizer._pre_quant_scale"].reshape((1, k))
        if tp_dim == 0:
            pre_quant_scale = torch_split(pre_quant_scale, 1)
        scale = amax / 8.0
        mOp.weight.value = AWQ_quantize_pack_preprocess(weight, scale)
        mOp.weights_scaling_factor.value = scale.to(torch_dtype).cpu().numpy()
        mOp.prequant_scaling_factor.value = pre_quant_scale.to(
            torch_dtype).cpu().numpy()

    # Check if we need to pad vocab
    v = model_params.get('transformer.wte.weight')
    [vocab_size, k] = v.shape
    pad_vocab = False
    pad_vocab_size1 = vocab_size
    if quantize_lm_head and vocab_size % 64 != 0:
        pad_vocab = True
        pad_vocab_size1 = int((vocab_size + 63) / 64) * 64
    if pad_vocab:
        new_v = torch.zeros([pad_vocab_size1, k])
        new_v[:vocab_size, :] = v
        v = new_v
    if mapping.is_first_pp_rank():
        tensorrt_llm_qwen.embedding.vocab_embedding.weight.value = v.to(
            torch_dtype).cpu().numpy()

    layer_ids = [extract_layer_idx(key) for key in model_params.keys()]
    layer_ids = [
        int(layer_idx) for layer_idx in layer_ids if layer_idx is not None
    ]

    num_hidden_layers = max(layer_ids) + 1
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for layer_idx in tqdm(layers_range, "Loading weights..."):
        prefix = "transformer.h." + str(layer_idx) + "."
        for idx, awq_attr in enumerate(awq_block_names):
            v = model_params[prefix + awq_attr]
            layer = attrgetter(tensorrt_llm_block_names[idx])(
                tensorrt_llm_qwen.layers[layer_idx])
            setattr(layer, 'value', v.to(torch_dtype).cpu().numpy())

        mPrefix = prefix + "attn.c_attn"
        mOp = tensorrt_llm_qwen.layers[layer_idx].attention.qkv
        process_and_assign_attn_weight(model_params, mPrefix, mOp, 1)

        # Attention QKV Liner Bias
        th_bias = model_params[prefix + "attn.c_attn.bias"].cpu().to(
            torch_dtype).contiguous()
        q_emb = th_bias.shape[0] // 3
        th_bias = th_bias.reshape(3, q_emb)
        split_v = split(th_bias, mapping.tp_size, mapping.rank, dim=1)
        split_v = split_v.reshape(3 * (q_emb // mapping.tp_size))
        tensorrt_llm_qwen.layers[
            layer_idx].attention.qkv.bias.value = np.ascontiguousarray(split_v)

        # Attention Dense (out_proj) Linear
        mPrefix = prefix + "attn.c_proj"
        mOp = tensorrt_llm_qwen.layers[layer_idx].attention.dense
        process_and_assign_weight(model_params, mPrefix, mOp, 0)

        # MLP down_proj (mlp.gate) Linear
        mPrefix = prefix + "mlp.w1"
        mOp = tensorrt_llm_qwen.layers[layer_idx].mlp.gate
        process_and_assign_weight(model_params, mPrefix, mOp, 1)

        # MLP up_proj (mlp.fc) Linear
        mPrefix = prefix + "mlp.w2"
        mOp = tensorrt_llm_qwen.layers[layer_idx].mlp.fc
        process_and_assign_weight(model_params, mPrefix, mOp, 1)

        # MLP gate_proj (mlp.proj) Linear
        mPrefix = prefix + "mlp.c_proj"
        mOp = tensorrt_llm_qwen.layers[layer_idx].mlp.proj
        process_and_assign_weight(model_params, mPrefix, mOp, 0)

    v = model_params['transformer.ln_f.weight']
    if mapping.is_last_pp_rank():
        tensorrt_llm_qwen.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()

    # lm_head
    if pad_vocab:
        weight = model_params['lm_head.weight']
        [vocab_size, k] = weight.shape
        new_weight = torch.zeros([pad_vocab_size1, k])
        new_weight[:vocab_size, :] = weight
        new_weight = new_weight.T.contiguous()
        amax = model_params['lm_head.weight_quantizer._amax'].reshape(
            [vocab_size, k // group_size])
        new_amax = torch.ones([pad_vocab_size1, k // group_size])
        new_amax[:vocab_size, :] = amax
        new_amax = new_amax.T.contiguous()
        new_scale = new_amax / 8
        tensorrt_llm_qwen.lm_head.weight.value = AWQ_quantize_pack_preprocess(
            new_weight, new_scale)
        tensorrt_llm_qwen.lm_head.weights_scaling_factor.value = new_scale.to(
            torch_dtype).cpu().numpy()
        tensorrt_llm_qwen.lm_head.prequant_scaling_factor.value = model_params[
            'lm_head.input_quantizer._pre_quant_scale'].to(
                torch_dtype).cpu().numpy()
    elif quantize_lm_head:
        mPrefix = "lm_head"
        mOp = tensorrt_llm_qwen.lm_head
        if mapping.is_last_pp_rank():
            process_and_assign_weight(model_params, mPrefix, mOp, 1)
    else:
        tensorrt_llm_qwen.lm_head.weight.value = torch_split(
            model_params['lm_head.weight'], 0).to(torch_dtype).cpu().numpy()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    del groupwise_qweight_safetensors
    del model_params
    import gc
    gc.collect()
    return
