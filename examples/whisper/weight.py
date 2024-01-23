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
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_np
from tensorrt_llm.quantization import QuantMode


def fromfile(dir_path, name, shape=None, dtype=None):
    p = dir_path + '/' + name
    if Path(p).exists():
        t = np.fromfile(p, dtype=dtype)
        if shape is not None:
            t = t.reshape(shape)
        return t
    return None


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment *
                               torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[
        np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def trans_weight(weight):
    return np.ascontiguousarray(weight)


def load_encoder_weight(tensorrt_llm_whisper, model_metadata: dict,
                        model_params: dict, n_layer: int):
    tensorrt_llm.logger.info('Loading encoder weights from PT...')

    quant_mode = getattr(tensorrt_llm_whisper, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2

    use_weight_only = quant_mode.is_weight_only()

    tensorrt_llm_whisper.positional_embedding.value = sinusoids(
        model_metadata['n_audio_ctx'], model_metadata['n_audio_state']).numpy()

    tensorrt_llm_whisper.conv1.weight.value = torch.unsqueeze(
        model_params['encoder.conv1.weight'], -1).numpy()
    tensorrt_llm_whisper.conv1.bias.value = model_params[
        'encoder.conv1.bias'].numpy()
    tensorrt_llm_whisper.conv2.weight.value = torch.unsqueeze(
        model_params['encoder.conv2.weight'], -1).numpy()
    tensorrt_llm_whisper.conv2.bias.value = model_params[
        'encoder.conv2.bias'].numpy()

    for i in range(n_layer):
        tensorrt_llm_whisper.encoder_layers[
            i].attention_layernorm.weight.value = model_params[
                'encoder.blocks.' + str(i) + '.attn_ln.weight'].numpy()
        tensorrt_llm_whisper.encoder_layers[
            i].attention_layernorm.bias.value = model_params[
                'encoder.blocks.' + str(i) + '.attn_ln.bias'].numpy()

        t = torch.cat([
            model_params['encoder.blocks.' + str(i) + '.attn.query.weight'],
            model_params['encoder.blocks.' + str(i) + '.attn.key.weight'],
            model_params['encoder.blocks.' + str(i) + '.attn.value.weight']
        ],
                      dim=0).numpy()

        if t is not None:
            dst = tensorrt_llm_whisper.encoder_layers[i].attention.qkv.weight
            if use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(np.ascontiguousarray(t.transpose(1, 0))),
                    plugin_weight_only_quant_type)
                dst.value = processed_torch_weights.numpy()
                scales = tensorrt_llm_whisper.encoder_layers[
                    i].attention.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = t

        bias_shape = model_params['encoder.blocks.' + str(i) +
                                  '.attn.query.bias'].shape
        dtype = model_params['encoder.blocks.' + str(i) +
                             '.attn.query.bias'].dtype
        fused_bias = torch.cat([
            model_params['encoder.blocks.' + str(i) + '.attn.query.bias'],
            torch.zeros([*bias_shape], dtype=dtype),
            model_params['encoder.blocks.' + str(i) + '.attn.value.bias']
        ],
                               dim=0).numpy()
        tensorrt_llm_whisper.encoder_layers[
            i].attention.qkv.bias.value = fused_bias

        t = trans_weight(model_params['encoder.blocks.' + str(i) +
                                      '.attn.out.weight'].numpy())
        if t is not None:
            dst = tensorrt_llm_whisper.encoder_layers[i].attention.dense.weight
            if use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(np.ascontiguousarray(t.transpose(1, 0))),
                    plugin_weight_only_quant_type)
                dst.value = processed_torch_weights.numpy()
                scales = tensorrt_llm_whisper.encoder_layers[
                    i].attention.dense.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = t
        tensorrt_llm_whisper.encoder_layers[
            i].attention.dense.bias.value = trans_weight(
                model_params['encoder.blocks.' + str(i) +
                             '.attn.out.bias'].numpy())

        tensorrt_llm_whisper.encoder_layers[
            i].mlp_layernorm.weight.value = model_params[
                'encoder.blocks.' + str(i) + '.mlp_ln.weight'].numpy()
        tensorrt_llm_whisper.encoder_layers[
            i].mlp_layernorm.bias.value = model_params['encoder.blocks.' +
                                                       str(i) +
                                                       '.mlp_ln.bias'].numpy()

        t = trans_weight(model_params['encoder.blocks.' + str(i) +
                                      '.mlp.0.weight'].numpy())
        if t is not None:
            dst = tensorrt_llm_whisper.encoder_layers[i].mlp.fc.weight
            if use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(np.ascontiguousarray(t.transpose(1, 0))),
                    plugin_weight_only_quant_type)
                dst.value = processed_torch_weights.numpy()
                scales = tensorrt_llm_whisper.encoder_layers[
                    i].mlp.fc.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = t
        tensorrt_llm_whisper.encoder_layers[i].mlp.fc.bias.value = trans_weight(
            model_params['encoder.blocks.' + str(i) + '.mlp.0.bias'].numpy())

        t = trans_weight(model_params['encoder.blocks.' + str(i) +
                                      '.mlp.2.weight'].numpy())
        if t is not None:
            dst = tensorrt_llm_whisper.encoder_layers[i].mlp.proj.weight
            if use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(np.ascontiguousarray(t.transpose(1, 0))),
                    plugin_weight_only_quant_type)
                dst.value = processed_torch_weights.numpy()
                scales = tensorrt_llm_whisper.encoder_layers[
                    i].mlp.proj.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = t
        tensorrt_llm_whisper.encoder_layers[
            i].mlp.proj.bias.value = trans_weight(
                model_params['encoder.blocks.' + str(i) +
                             '.mlp.2.bias'].numpy())

    tensorrt_llm_whisper.ln_post.weight.value = model_params[
        'encoder.ln_post.weight'].numpy()
    tensorrt_llm_whisper.ln_post.bias.value = model_params[
        'encoder.ln_post.bias'].numpy()


def fuse_qkv(q, k, v):
    qkv_weight = np.concatenate((q, k, v))
    return qkv_weight


def load_decoder_weight(
    tllm_model,
    model_params: dict,
):
    tensorrt_llm.logger.info('Loading decoder weights from PT...')

    quant_mode = getattr(tllm_model, 'quant_mode', QuantMode(0))
    param_dtype = 'float16'

    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    tllm_model.embedding.vocab_embedding.weight.value = trans_weight(
        model_params['decoder.token_embedding.weight'].numpy())
    tllm_model.lm_head.weight.value = trans_weight(
        model_params['decoder.token_embedding.weight'].numpy())
    if tllm_model.embedding.position_embedding:
        tllm_model.embedding.position_embedding.weight.value = trans_weight(
            model_params['decoder.positional_embedding'].numpy())

    for i in range(tllm_model.num_layers):
        layer = tllm_model.decoder_layers[i]

        t = torch.cat([
            model_params['decoder.blocks.' + str(i) + '.attn.query.weight'],
            model_params['decoder.blocks.' + str(i) + '.attn.key.weight'],
            model_params['decoder.blocks.' + str(i) + '.attn.value.weight']
        ],
                      dim=0).numpy()

        if t is not None:
            dst = layer.self_attention.qkv.weight
            if use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(np.ascontiguousarray(t.transpose(1, 0))),
                    plugin_weight_only_quant_type)
                dst.value = torch.tensor(np.ascontiguousarray(t.transpose(
                    1, 0))).numpy().astype(str_dtype_to_np(param_dtype))
                scales = layer.self_attention.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = t

        t = trans_weight(model_params['decoder.blocks.' + str(i) +
                                      '.attn.out.weight'].numpy())

        if t is not None:
            dst = layer.self_attention.dense.weight
            if use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(np.ascontiguousarray(t.transpose(1, 0))),
                    plugin_weight_only_quant_type)
                dst.value = torch.tensor(np.ascontiguousarray(t.transpose(
                    1, 0))).numpy().astype(str_dtype_to_np(param_dtype))
                scales = layer.self_attention.dense.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = t

        if tllm_model.has_attention_qkvo_bias:
            bias_shape = model_params['decoder.blocks.' + str(i) +
                                      '.attn.query.bias'].shape
            dtype = model_params['decoder.blocks.' + str(i) +
                                 '.attn.query.bias'].dtype
            layer.self_attention.qkv.bias.value = fuse_qkv(
                trans_weight(model_params['decoder.blocks.' + str(i) +
                                          '.attn.query.bias'].numpy()),
                torch.zeros([*bias_shape], dtype=dtype).numpy(),
                trans_weight(model_params['decoder.blocks.' + str(i) +
                                          '.attn.value.bias'].numpy()))
            layer.self_attention.dense.bias.value = trans_weight(
                model_params['decoder.blocks.' + str(i) +
                             '.attn.out.bias'].numpy())

        layer.self_attention_layernorm.weight.value = trans_weight(
            model_params['decoder.blocks.' + str(i) +
                         '.attn_ln.weight'].numpy())
        layer.self_attention_layernorm.bias.value = trans_weight(
            model_params['decoder.blocks.' + str(i) + '.attn_ln.bias'].numpy())

        t = torch.cat([
            model_params['decoder.blocks.' + str(i) +
                         '.cross_attn.query.weight'],
            model_params['decoder.blocks.' + str(i) + '.cross_attn.key.weight'],
            model_params['decoder.blocks.' + str(i) +
                         '.cross_attn.value.weight']
        ],
                      dim=0).numpy()

        if t is not None:
            dst = layer.cross_attention.qkv.weight
            if use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(np.ascontiguousarray(t.transpose(1, 0))),
                    plugin_weight_only_quant_type)
                dst.value = torch.tensor(np.ascontiguousarray(t.transpose(
                    1, 0))).numpy().astype(str_dtype_to_np(param_dtype))
                scales = layer.cross_attention.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = t

        layer.cross_attention.dense.weight.value = trans_weight(
            model_params['decoder.blocks.' + str(i) +
                         '.cross_attn.out.weight'].numpy())

        t = trans_weight(model_params['decoder.blocks.' + str(i) +
                                      '.cross_attn.out.weight'].numpy())

        if t is not None:
            dst = layer.cross_attention.dense.weight
            if use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(np.ascontiguousarray(t.transpose(1, 0))),
                    plugin_weight_only_quant_type)
                dst.value = torch.tensor(np.ascontiguousarray(t.transpose(
                    1, 0))).numpy().astype(str_dtype_to_np(param_dtype))
                scales = layer.cross_attention.dense.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = t

        if tllm_model.has_attention_qkvo_bias:
            bias_shape = model_params['decoder.blocks.' + str(i) +
                                      '.cross_attn.query.bias'].shape
            dtype = model_params['decoder.blocks.' + str(i) +
                                 '.cross_attn.query.bias'].dtype
            cross_attn_qkv_bias = fuse_qkv(
                trans_weight(model_params['decoder.blocks.' + str(i) +
                                          '.cross_attn.query.bias'].numpy()),
                torch.zeros([*bias_shape], dtype=dtype).numpy(),
                trans_weight(model_params['decoder.blocks.' + str(i) +
                                          '.cross_attn.value.bias'].numpy()))

            layer.cross_attention.qkv.bias.value = cross_attn_qkv_bias

            layer.cross_attention.dense.bias.value = trans_weight(
                model_params['decoder.blocks.' + str(i) +
                             '.cross_attn.out.bias'].numpy())

        layer.cross_attention_layernorm.weight.value = trans_weight(
            model_params['decoder.blocks.' + str(i) +
                         '.cross_attn_ln.weight'].numpy())
        layer.cross_attention_layernorm.bias.value = trans_weight(
            model_params['decoder.blocks.' + str(i) +
                         '.cross_attn_ln.bias'].numpy())

        t = trans_weight(model_params['decoder.blocks.' + str(i) +
                                      '.mlp.0.weight'].numpy())

        if t is not None:
            dst = layer.mlp.fc.weight
            if use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(np.ascontiguousarray(t.transpose(1, 0))),
                    plugin_weight_only_quant_type)
                dst.value = torch.tensor(np.ascontiguousarray(t.transpose(
                    1, 0))).numpy().astype(str_dtype_to_np(param_dtype))
                scales = layer.mlp.fc.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = t

        t = trans_weight(model_params['decoder.blocks.' + str(i) +
                                      '.mlp.2.weight'].numpy())

        if t is not None:
            dst = layer.mlp.proj.weight
            if use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(np.ascontiguousarray(t.transpose(1, 0))),
                    plugin_weight_only_quant_type)
                dst.value = torch.tensor(np.ascontiguousarray(t.transpose(
                    1, 0))).numpy().astype(str_dtype_to_np(param_dtype))
                scales = layer.mlp.proj.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = t

        if tllm_model.has_mlp_bias:
            layer.mlp.fc.bias.value = trans_weight(
                model_params['decoder.blocks.' + str(i) +
                             '.mlp.0.bias'].numpy())
            layer.mlp.proj.bias.value = trans_weight(
                model_params['decoder.blocks.' + str(i) +
                             '.mlp.2.bias'].numpy())

        layer.mlp_layernorm.weight.value = trans_weight(
            model_params['decoder.blocks.' + str(i) + '.mlp_ln.weight'].numpy())
        layer.mlp_layernorm.bias.value = trans_weight(
            model_params['decoder.blocks.' + str(i) + '.mlp_ln.bias'].numpy())

    if tllm_model.final_layernorm:
        tllm_model.final_layernorm.weight.value = trans_weight(
            model_params['decoder.ln.weight'].numpy())
        tllm_model.final_layernorm.bias.value = trans_weight(
            model_params['decoder.ln.bias'].numpy())
