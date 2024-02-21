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
from typing import Dict, List, Optional, Union

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.quantized.quant import get_dummy_quant_scales
from tensorrt_llm.quantization import QuantMode


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
    tensorrt_llm.logger.info('Loading weights from binary...')
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

    layers_per_pipeline_stage = tensorrt_llm_gemma.num_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

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
        idx = i - mapping.pp_rank * layers_per_pipeline_stage
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
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


def load_from_hf_llama():
    # leave for preventing import issue
    pass


def quantize_fp8_weigths(weights, num_layers, mapping):

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
            weight = weights[trt_llm_name]
            dtype = weights[trt_llm_name].dtype
            scale = get_scaling_factor(weight)
            scaled_weights[trt_llm_name] = np.ascontiguousarray(
                (weight * scale).astype(dtype))
            scaling_factors[scale_name] = np.asarray([1 / scale
                                                      ]).astype(np.float32)
    return scaling_factors


def load_from_fp8_llama(quant_ckpt_path: str, num_layers: int, mapping: Mapping,
                        fp8_kv_cache: bool, weight_scales: dict):
    """
    Get the fp8 scaling factors.
    """
    fake_fp8_sf_dt = torch.float32

    if quant_ckpt_path is not None and os.path.isfile(quant_ckpt_path):
        fp8_llama = np.load(quant_ckpt_path)
    else:
        fp8_llama = None
        tensorrt_llm.logger.info(
            f"There is not quantized checkpoint, use dummy fp8 scaling factors instead."
        )
    weights = {}

    def get_fp8_llama(name):
        if fp8_llama is not None:
            return fp8_llama[name]
        else:
            return torch.tensor([1.0], dtype=fake_fp8_sf_dt).numpy()

    layers_range = mapping.pp_layers(num_layers)
    for l in layers_range:
        prefix = f'_np:layers:{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'

        weights[f'{tllm_prex}.attention.qkv.activation_scaling_factor'] = max(
            get_fp8_llama(
                f'{prefix}:attention:qkv:q:activation_scaling_factor'),
            get_fp8_llama(
                f'{prefix}:attention:qkv:k:activation_scaling_factor'),
            get_fp8_llama(
                f'{prefix}:attention:qkv:v:activation_scaling_factor'))
        weights[f'{tllm_prex}.attention.qkv.weights_scaling_factor'] = max(
            get_fp8_llama(f'{prefix}:attention:qkv:q:weights_scaling_factor'),
            get_fp8_llama(f'{prefix}:attention:qkv:k:weights_scaling_factor'),
            get_fp8_llama(f'{prefix}:attention:qkv:v:weights_scaling_factor'))
        weights[
            f'{tllm_prex}.attention.dense.activation_scaling_factor'] = get_fp8_llama(
                f'{prefix}:attention:dense:activation_scaling_factor')
        weights[
            f'{tllm_prex}.attention.dense.weights_scaling_factor'] = get_fp8_llama(
                f'{prefix}:attention:dense:weights_scaling_factor')

        weights[
            f'{tllm_prex}.mlp.fc.activation_scaling_factor'] = get_fp8_llama(
                f'{prefix}:mlp:fc:activation_scaling_factor')
        weights[f'{tllm_prex}.mlp.fc.weights_scaling_factor'] = get_fp8_llama(
            f'{prefix}:mlp:fc:weights_scaling_factor')

        weights[
            f'{tllm_prex}.mlp.gate.activation_scaling_factor'] = get_fp8_llama(
                f'{prefix}:mlp:gate:activation_scaling_factor')
        weights[f'{tllm_prex}.mlp.gate.weights_scaling_factor'] = get_fp8_llama(
            f'{prefix}:mlp:gate:weights_scaling_factor')

        weights[
            f'{tllm_prex}.mlp.proj.activation_scaling_factor'] = get_fp8_llama(
                f'{prefix}:mlp:proj:activation_scaling_factor')
        weights[f'{tllm_prex}.mlp.proj.weights_scaling_factor'] = get_fp8_llama(
            f'{prefix}:mlp:proj:weights_scaling_factor')

        if fp8_kv_cache:
            # Not calibrarting KV cache.
            scaling_factor = 1.0
            weights[
                f'{tllm_prex}.attention.kv_cache_scaling_factor'] = torch.tensor(
                    [scaling_factor], dtype=fake_fp8_sf_dt).numpy()
            if fp8_llama is None:
                weights.update(weight_scales)

    return weights


def dummy_scaling_facotr_sq(weights):
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


def dummy_scaling_facotr_kv_cache(weights):
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
