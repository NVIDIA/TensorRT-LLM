# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import torch
from tqdm import tqdm

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
                 multi_query_mode=False):
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
        tensorrt_llm_qwen.vocab_embedding.weight.value = (fromfile(
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
                processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(t), plugin_weight_only_quant_type)
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
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
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
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
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
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
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
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
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
                i].attention.kv_orig_quant_scale.value = 1.0 / t
            tensorrt_llm_qwen.layers[i].attention.kv_quant_orig_scale.value = t

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
                      multi_query_mode=False):
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
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'transformer.wte.weight' in k:
            tensorrt_llm_qwen.vocab_embedding.weight.value = v
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
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
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
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
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
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
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
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
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
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
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
