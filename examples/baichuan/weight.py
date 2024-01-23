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
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_np, str_dtype_to_torch,
                                 torch_to_numpy)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import BaichuanForCausalLM
from tensorrt_llm.models.quantized.quant import get_dummy_quant_scales
from tensorrt_llm.quantization import QuantMode


def get_scaling_factors(
    model_path: Union[str, Path],
    num_layers: int,
    quant_mode: Optional[QuantMode] = None,
) -> Optional[Dict[str, List[int]]]:
    """ Get the scaling factors for Baichuan model

    Returns a dictionary of scaling factors for the selected layers of the
    Baichuan model.

    Args:
        model_path (str): Path to the quantized Baichuan model
        layers (list): List of layers to get the scaling factors for. If None,
            all layers are selected.

    Returns:
        dict: Dictionary of scaling factors for the selected layers of the
        Baichuan model.

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
        logger.warning(f"--quantized_fp8_model_path not specified. "
                       f"Initialize quantization scales automatically.")
        return get_dummy_quant_scales(num_layers)
    weight_dict = np.load(model_path)

    # yapf: disable
    scaling_factor = {
        'qkv_act': [],
        'qkv_weights': [],
        'qkv_output': [],
        'dense_act': [],
        'dense_weights': [],
        'fc_act': [],
        'fc_weights': [],
        'gate_act': [],
        'gate_weights': [],
        'proj_act': [],
        'proj_weights': [],
    }

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
        scaling_factor['dense_act'].append(weight_dict[f'_np:layers:{layer}:attention:dense:activation_scaling_factor'].item())
        scaling_factor['dense_weights'].append(weight_dict[f'_np:layers:{layer}:attention:dense:weights_scaling_factor'].item())
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
        return np.ascontiguousarray(np.split(v, tp_size)[idx].copy())
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx].copy())


def load_from_hf_baichuan(tensorrt_llm_baichuan,
                          hf_baichuan,
                          model_version,
                          rank=0,
                          tensor_parallel=1,
                          dtype="float32",
                          use_gemm_woq_plugin=True):
    assert model_version is not None
    tensorrt_llm.logger.info(
        f'Loading weights from HF Baichuan {model_version}...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_baichuan, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    model_params = dict(hf_baichuan.named_parameters())
    for k, v in model_params.items():
        torch_dtype = str_dtype_to_torch(dtype)
        v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'model.embed_tokens.weight' in k:
            tensorrt_llm_baichuan.vocab_embedding.weight.value = v
        elif 'model.norm.weight' in k:
            tensorrt_llm_baichuan.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            if model_version.startswith('v2'):
                # baichuan v2 models use NormHead
                tensorrt_llm.logger.info(
                    f'Normalizing lm_head.weight for {model_version}')
                original_v = model_params[k]
                v = torch_to_numpy(
                    torch.nn.functional.normalize(original_v).to(
                        torch_dtype).detach().cpu())
            tensorrt_llm_baichuan.lm_head.weight.value = np.ascontiguousarray(
                split(v, tensor_parallel, rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx >= tensorrt_llm_baichuan.num_layers:
                continue
            if 'input_layernorm.weight' in k:
                tensorrt_llm_baichuan.layers[
                    idx].input_layernorm.weight.value = v
            elif 'post_attention_layernorm.weight' in k:
                dst = tensorrt_llm_baichuan.layers[idx].post_layernorm.weight
                dst.value = v
            elif 'self_attn.W_pack.weight' in k:
                dst = tensorrt_llm_baichuan.layers[idx].attention.qkv.weight
                q_emb = v.shape[0] // 3
                model_emb = v.shape[1]
                v = v.reshape(3, q_emb, model_emb)
                split_v = split(v, tensor_parallel, rank, dim=1)
                split_v = split_v.reshape(3 * (q_emb // tensor_parallel),
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
                    scales = tensorrt_llm_baichuan.layers[
                        idx].attention.qkv.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'self_attn.o_proj.weight' in k:
                dst = tensorrt_llm_baichuan.layers[idx].attention.dense.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_baichuan.layers[
                        idx].attention.dense.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.up_proj.weight' in k:
                dst = tensorrt_llm_baichuan.layers[idx].mlp.gate.weight
                split_v = split(v, tensor_parallel, rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_baichuan.layers[
                        idx].mlp.gate.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.down_proj.weight' in k:
                dst = tensorrt_llm_baichuan.layers[idx].mlp.proj.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_baichuan.layers[
                        idx].mlp.proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.gate_proj.weight' in k:
                dst = tensorrt_llm_baichuan.layers[idx].mlp.fc.weight
                split_v = split(v, tensor_parallel, rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    if not use_gemm_woq_plugin:
                        dst.value = torch.tensor(v).numpy().astype(
                            str_dtype_to_np(dtype))
                    else:
                        dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_baichuan.layers[
                        idx].mlp.fc.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


def parse_bin_config(ini_file):
    baichuan_config = configparser.ConfigParser()
    baichuan_config.read(ini_file)

    n_embd = baichuan_config.getint('baichuan', 'hidden_size')
    n_head = baichuan_config.getint('baichuan', 'num_attention_heads')
    n_kv_head = n_head
    n_layer = baichuan_config.getint('baichuan', 'num_hidden_layers')
    if baichuan_config.has_option('baichuan', 'max_position_embeddings'):
        n_positions = baichuan_config.getint('baichuan',
                                             'max_position_embeddings')
    else:
        n_positions = baichuan_config.getint('baichuan', 'model_max_length')
    vocab_size = baichuan_config.getint('baichuan', 'vocab_size')
    hidden_act = baichuan_config.get('baichuan', 'hidden_act')
    inter_size = baichuan_config.getint('baichuan',
                                        'intermediate_size',
                                        fallback=None)

    if inter_size is None:
        inter_size = 4 * n_embd

    return n_embd, n_head, n_layer, n_positions, vocab_size, hidden_act, inter_size, n_kv_head


def gen_suffix(rank, use_smooth_quant, quant_per_channel):
    suffix = f"{rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix
    return suffix


def load_from_binary(tensorrt_llm_baichuan: BaichuanForCausalLM,
                     dir_path,
                     model_version,
                     mapping=Mapping(),
                     fp16=False,
                     multi_query_mode=False,
                     dtype="float32",
                     use_gemm_woq_plugin=True):
    tensorrt_llm.logger.info('Loading weights from binary...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_baichuan, 'quant_mode', QuantMode(0))

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
    quant_mode = getattr(tensorrt_llm_baichuan, "quant_mode", QuantMode(0))
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
        tensorrt_llm_baichuan.vocab_embedding.weight.value = (fromfile(
            dir_path, 'vocab_embedding.weight.bin', [vocab_size, n_embd]))

    if mapping.is_last_pp_rank():
        tensorrt_llm_baichuan.ln_f.weight.value = (fromfile(
            dir_path, 'ln_f.weight.bin'))
    # share input embedding
    lm_head_weight = fromfile(dir_path, 'lm_head.weight.bin',
                              [vocab_size, n_embd])
    if model_version.startswith('v2'):
        # baichuan v2 models use NormHead
        tensorrt_llm.logger.info(
            f'Normalizing lm_head.weight for {model_version}')
        lm_head_weight = lm_head_weight / np.linalg.norm(
            lm_head_weight, axis=1, keepdims=True)

    if vocab_size % mapping.tp_size != 0:
        # padding
        vocab_size_padded = tensorrt_llm_baichuan.lm_head.out_features * mapping.tp_size
        pad_width = vocab_size_padded - vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                'constant',
                                constant_values=0)
    if mapping.is_last_pp_rank():
        tensorrt_llm_baichuan.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, mapping.tp_size, mapping.tp_rank))

    layers_range = list(
        range(mapping.pp_rank * tensorrt_llm_baichuan.num_layers,
              (mapping.pp_rank + 1) * tensorrt_llm_baichuan.num_layers, 1))

    for i in layers_range:
        n_groups = n_head // n_kv_head
        c_attn_out_dim = (
            3 * n_embd // mapping.tp_size) if not multi_query_mode else (
                n_embd // mapping.tp_size +
                (n_embd // n_head * n_groups) // mapping.tp_size * 2)
        idx = i - mapping.pp_rank * tensorrt_llm_baichuan.num_layers
        tensorrt_llm_baichuan.layers[idx].input_layernorm.weight.value = (
            fromfile(dir_path,
                     'model.layers.' + str(i) + '.input_layernorm.weight.bin'))
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.weight.' + suffix,
            [n_embd, c_attn_out_dim], w_type)
        if t is not None:
            dst = tensorrt_llm_baichuan.layers[idx].attention.qkv.weight
            if use_smooth_quant:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
                set_smoothquant_scale_factors(
                    tensorrt_llm_baichuan.layers[idx].attention.qkv,
                    tensorrt_llm_baichuan.layers[idx].input_layernorm.
                    scale_to_int,
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
                if not use_gemm_woq_plugin:
                    dst.value = torch.tensor(t).numpy().astype(
                        str_dtype_to_np(dtype))
                else:
                    dst.value = processed_torch_weights.numpy()
                scales = tensorrt_llm_baichuan.layers[
                    i].attention.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_baichuan.layers[idx].attention.dense.weight
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attention.dense.weight.' + suffix,
            [n_embd // mapping.tp_size, n_embd], w_type)
        if use_smooth_quant:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
            dense_scale = getattr(tensorrt_llm_baichuan.layers[idx].attention,
                                  "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_baichuan.layers[idx].attention.dense, dense_scale,
                dir_path, 'model.layers.' + str(i) + '.attention.dense.',
                [1, n_embd], quant_per_token_dyn, quant_per_channel)
            set_smoother(tensorrt_llm_baichuan.layers[idx].attention.dense,
                         dir_path,
                         'model.layers.' + str(i) + '.attention.dense',
                         [1, n_embd // mapping.tp_size], mapping.tp_rank)
        elif use_weight_only:
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            if not use_gemm_woq_plugin:
                dst.value = torch.tensor(t).numpy().astype(
                    str_dtype_to_np(dtype))
            else:
                dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_baichuan.layers[
                i].attention.dense.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_baichuan.layers[idx].post_layernorm.weight
        dst.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.post_layernorm.weight.bin')

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.fc.weight.' + suffix,
                     [n_embd, inter_size // mapping.tp_size], w_type)

        if use_smooth_quant:
            tensorrt_llm_baichuan.layers[
                idx].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            set_smoothquant_scale_factors(
                tensorrt_llm_baichuan.layers[idx].mlp.fc,
                tensorrt_llm_baichuan.layers[idx].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.fc.',
                [1, inter_size // mapping.tp_size],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_baichuan.layers[i].mlp.fc.weight
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            if not use_gemm_woq_plugin:
                dst.value = torch.tensor(t).numpy().astype(
                    str_dtype_to_np(dtype))
            else:
                dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_baichuan.layers[i].mlp.fc.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_baichuan.layers[
                idx].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.gate.weight.' + suffix,
                     [n_embd, inter_size // mapping.tp_size], w_type)
        if use_smooth_quant:
            tensorrt_llm_baichuan.layers[
                idx].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            set_smoothquant_scale_factors(
                tensorrt_llm_baichuan.layers[idx].mlp.gate,
                tensorrt_llm_baichuan.layers[idx].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.gate.',
                [1, inter_size // mapping.tp_size],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_baichuan.layers[i].mlp.gate.weight
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            if not use_gemm_woq_plugin:
                dst.value = torch.tensor(t).numpy().astype(
                    str_dtype_to_np(dtype))
            else:
                dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_baichuan.layers[i].mlp.gate.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_baichuan.layers[
                idx].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.proj.weight.' + suffix,
                     [inter_size // mapping.tp_size, n_embd], w_type)
        if use_smooth_quant:
            tensorrt_llm_baichuan.layers[
                idx].mlp.proj.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            proj_scale = getattr(tensorrt_llm_baichuan.layers[idx].mlp,
                                 "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_baichuan.layers[idx].mlp.proj, proj_scale,
                dir_path, 'model.layers.' + str(i) + '.mlp.proj.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
            set_smoother(tensorrt_llm_baichuan.layers[idx].mlp.proj, dir_path,
                         'model.layers.' + str(i) + '.mlp.proj',
                         [1, inter_size // mapping.tp_size], mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_baichuan.layers[i].mlp.proj.weight
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            if not use_gemm_woq_plugin:
                dst.value = torch.tensor(t).numpy().astype(
                    str_dtype_to_np(dtype))
            else:
                dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_baichuan.layers[i].mlp.proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_baichuan.layers[idx].mlp.proj.weight.value = (
                np.ascontiguousarray(np.transpose(t, [1, 0])))

        if use_int8_kv_cache:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            tensorrt_llm_baichuan.layers[
                idx].attention.kv_cache_scaling_factor.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


def load_from_awq_baichuan(tensorrt_llm_baichuan: BaichuanForCausalLM,
                           quant_ckpt_path,
                           model_version,
                           quantize_lm_head,
                           mapping=Mapping(),
                           dtype="float16",
                           bin_model_dir=None):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise AWQ Baichuan checkpoint...')
    tik = time.time()

    if quant_ckpt_path.endswith(".npz"):
        awq_baichuan = np.load(quant_ckpt_path)
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
            v = torch.from_numpy(awq_baichuan[awq_prefix + key])
            if "weights_scaling_factor" in key:
                v *= 7  # For AMMO *.npz checkpoints
            return v

        group_size = load("layers:0:attention:dense:weight").numel() // load(
            "layers:0:attention:dense:weights_scaling_factor").numel()
    else:
        assert False, "Unsupported AWQ quantized checkpoint format"

    quant_mode = getattr(tensorrt_llm_baichuan, 'quant_mode', QuantMode(0))
    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
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
    if v.shape[0] % 64 != 0:
        v = torch.nn.functional.pad(v, [0, 0, 0, 64 - v.shape[0] % 64])
    if mapping.is_first_pp_rank():
        tensorrt_llm_baichuan.vocab_embedding.weight.value = v.to(
            torch_dtype).cpu().numpy()

    # 2. lm_head
    if not quantize_lm_head:
        assert (
            awq_key_list[1] + awq_suffix_list[1]
        ) not in awq_baichuan, "lm_head is quantized in checkpoint, please re-generate it"
        original_v = load(awq_key_list[1] + awq_suffix_list[0])
        if model_version.startswith('v2'):
            # baichuan v2 models use NormHead which is not quantized
            tensorrt_llm.logger.info(
                f'Normalizing lm_head.weight for {model_version}')
            v = torch_split(torch.nn.functional.normalize(original_v), 0)
        else:
            v = torch_split(original_v, 0)
        if mapping.is_last_pp_rank():
            tensorrt_llm_baichuan.lm_head.weight.value = v.to(
                torch_dtype).cpu().numpy()
    else:
        assert not model_version.startswith(
            'v2'), f"Do not support quantizing lm_head for {model_version}"
        v = [load(awq_key_list[1] + suf) for suf in awq_suffix_list]
        if v[0].shape[0] % 64 != 0:
            v[0] = torch.nn.functional.pad(v[0],
                                           [0, 0, 0, 64 - v[0].shape[0] % 64])
            scale_align = 64 * (v[0].shape[1] // group_size)
            v[1] = v[1].reshape(-1)
            v[1] = torch.nn.functional.pad(
                v[1], [0, scale_align - v[1].shape[0] % scale_align], value=1)
        if mapping.is_last_pp_rank():
            process_and_assign_weight(tensorrt_llm_baichuan.lm_head, v, 1)

    # 3. ln_f
    v = load(awq_key_list[2])
    if mapping.is_last_pp_rank():
        tensorrt_llm_baichuan.ln_f.weight.value = v.to(
            torch_dtype).cpu().numpy()

    # 4. Weights inside each layer
    num_hidden_layers = tensorrt_llm_baichuan.num_layers
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for l in layers_range:
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = "layers" + split_sym + str(layer_idx) + split_sym
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        layer = tensorrt_llm_baichuan.layers[layer_idx]

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

        # 4.8 attention.kv_cache_scaling_factor
        if use_int8_kv_cache:
            assert bin_model_dir, "You must pass --bin_model_dir to tell TRT-LLM where to look for scales of INT8 kv cache."
            t = fromfile(
                bin_model_dir, 'model.layers.' + str(layer_idx) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            assert t is not None, f"{bin_model_dir} does not contain model.layers.{layer_idx}.attention.query_key_value.scale_y_quant_orig.bin"
            layer.attention.kv_cache_scaling_factor.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


def load_from_gptq_baichuan(tensorrt_llm_baichuan,
                            quant_ckpt_path,
                            model_version,
                            mapping=Mapping(),
                            dtype="float16",
                            bin_model_dir=None):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise GPTQ Baichuan safetensors...')
    tik = time.time()

    gptq_baichuan = safe_open(quant_ckpt_path, framework="pt", device=0)
    gptq_prefix = "model."
    gptq_suffix_list = [".qweight", ".qzeros", ".scales"]
    gptq_key_list = [
        "embed_tokens.weight",  # vocab_embedding
        "lm_head.weight",  # lm_head
        "norm.weight",  # ln_f
        "self_attn.W_pack",  # attention.qkv
        "_proj",  #
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
            return gptq_baichuan.get_tensor(key)
        else:
            return gptq_baichuan.get_tensor(gptq_prefix + key)

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
        tensorrt_llm_baichuan.vocab_embedding.weight.value = v.to(
            torch_dtype).cpu().numpy()

    # 2. lm_head
    original_v = load(gptq_key_list[1], "no_prefix")
    if model_version.startswith('v2'):
        # baichuan v2 models use NormHead
        tensorrt_llm.logger.info(
            f'Normalizing lm_head.weight for {model_version}')
        v = torch_split(torch.nn.functional.normalize(original_v), 0)
    else:
        v = torch_split(original_v, 0)
    if mapping.is_last_pp_rank():
        tensorrt_llm_baichuan.lm_head.weight.value = v.to(
            torch_dtype).cpu().numpy()

    # 3. ln_f
    v = load(gptq_key_list[2])
    if mapping.is_last_pp_rank():
        tensorrt_llm_baichuan.ln_f.weight.value = v.to(
            torch_dtype).cpu().numpy()

    # 4. Weights inside each layer
    num_hidden_layers = tensorrt_llm_baichuan.num_layers
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))
    for l in layers_range:
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = "layers" + split_sym + str(layer_idx) + split_sym
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        layer = tensorrt_llm_baichuan.layers[layer_idx]

        # 4.1 attention.qkv
        qkv_weight_list = []
        for suf in gptq_suffix_list:
            qkv_list = []
            comp_part = load(prefix + gptq_key_list[3] + suf)
            qkv = torch.chunk(comp_part, 3, 1)
            for i in range(3):
                comp_part = qkv[i]
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

        # 4.7 pst_layernorm
        v = load(prefix + gptq_key_list[10])
        layer.post_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return
