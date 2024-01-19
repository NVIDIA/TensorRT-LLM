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
from typing import Any

import numpy as np

from ...layers import ColumnLinear, RowLinear
from ...module import Module
from ...quantization import QuantMode
from ...quantization.layers import FP8Linear, FP8RowLinear
from ...quantization.quantize import weight_only_quantize

# isort: off
from ...quantization.layers import (SmoothQuantAttention, SmoothQuantGatedMLP,
                                    SmoothQuantLayerNorm, SmoothQuantMLP,
                                    SmoothQuantRmsNorm,
                                    WeightOnlyGroupwiseQuantColumnLinear,
                                    WeightOnlyGroupwiseQuantRowLinear)
# isort: on


def _smooth_quantize_gpt(model, quant_mode):
    assert quant_mode.has_act_and_weight_quant()
    for layer in model.layers:
        assert hasattr(layer,
                       "input_layernorm"), "The layer has no input_layernorm"
        layer.input_layernorm = SmoothQuantLayerNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)
        assert hasattr(layer, "attention"), "The layer has no attention"
        layer.attention = SmoothQuantAttention(
            layer.hidden_size,
            num_attention_heads=layer.num_attention_heads,
            num_kv_heads=layer.attention.num_attention_kv_heads * layer.tp_size,
            max_position_embeddings=layer.max_position_embeddings,
            num_layers=layer.num_layers,
            apply_query_key_layer_scaling=layer.apply_query_key_layer_scaling,
            dtype=layer.dtype,
            attention_mask_type=layer.attention_mask_type,
            bias=(layer.attention.dense.bias != None),
            qkv_bias_only=(layer.attention.qkv.bias != None
                           and layer.attention.dense.bias == None),
            position_embedding_type=layer.position_embedding_type,
            tp_group=layer.tp_group,
            tp_size=layer.tp_size,
            tp_rank=layer.attention.tp_rank,
            quant_mode=quant_mode)
        assert hasattr(layer, "mlp"), "The layer has no mlp"
        layer.mlp = SmoothQuantMLP(hidden_size=layer.hidden_size,
                                   ffn_hidden_size=layer.hidden_size * 4,
                                   hidden_act=layer.hidden_act,
                                   bias=(layer.mlp.fc.bias != None),
                                   dtype=layer.dtype,
                                   tp_group=layer.tp_group,
                                   tp_size=layer.tp_size,
                                   quant_mode=quant_mode)
        assert hasattr(layer,
                       "post_layernorm"), "The layer has no post_layernorm"
        layer.post_layernorm = SmoothQuantLayerNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)

    return model


def _smooth_quantize_llama(model, quant_mode):
    assert quant_mode.has_act_and_weight_quant()
    for layer in model.layers:
        assert hasattr(layer,
                       "input_layernorm"), "The layer has no input_layernorm"
        layer.input_layernorm = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)
        assert hasattr(layer, "attention"), "The layer has no attention"
        layer.attention = SmoothQuantAttention(
            layer.hidden_size,
            num_attention_heads=layer.num_attention_heads,
            num_kv_heads=layer.num_kv_heads,
            max_position_embeddings=layer.max_position_embeddings,
            num_layers=model.num_layers,
            dtype=layer.dtype,
            attention_mask_type=layer.attention_mask_type,
            position_embedding_type=layer.position_embedding_type,
            rotary_embedding_base=layer.attention.rotary_embedding_base,
            tp_group=layer.tp_group,
            tp_size=layer.tp_size,
            quant_mode=quant_mode,
            bias=layer.attention.qkv.bias is not None)

        assert hasattr(layer, "mlp"), "The layer has no mlp"
        if hasattr(model, "moe_config"):
            assert not model.moe_config.has_moe(
            ), "MOE does not support smooth quant"
        layer.mlp = SmoothQuantGatedMLP(hidden_size=model.hidden_size,
                                        ffn_hidden_size=layer.mlp_hidden_size,
                                        hidden_act=layer.hidden_act,
                                        dtype=layer.dtype,
                                        tp_group=layer.tp_group,
                                        tp_size=layer.tp_size,
                                        quant_mode=quant_mode,
                                        bias=layer.mlp.fc.bias is not None)
        assert hasattr(
            layer,
            "post_layernorm"), "The layer has no post_rmspost_layernormnorm"
        layer.post_layernorm = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)

    return model


def _smooth_quantize_bloom(model, quant_mode):
    assert quant_mode.has_act_and_weight_quant()
    for layer in model.layers:
        assert hasattr(layer,
                       "input_layernorm"), "The layer has no input_layernorm"
        layer.input_layernorm = SmoothQuantLayerNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)
        assert hasattr(layer, "attention"), "The layer has no attention"
        layer.attention = SmoothQuantAttention(
            layer.hidden_size,
            num_attention_heads=layer.num_attention_heads,
            max_position_embeddings=layer.max_position_embeddings,
            num_layers=layer.num_layers,
            dtype=layer.dtype,
            attention_mask_type=layer.attention_mask_type,
            position_embedding_type=layer.position_embedding_type,
            tp_group=layer.tp_group,
            tp_size=layer.tp_size,
            tp_rank=layer.tp_rank,
            quant_mode=quant_mode)

        assert hasattr(layer, "mlp"), "The layer has no mlp"
        layer.mlp = SmoothQuantMLP(hidden_size=layer.hidden_size,
                                   ffn_hidden_size=layer.hidden_size * 4,
                                   hidden_act=layer.hidden_act,
                                   dtype=layer.dtype,
                                   tp_group=layer.tp_group,
                                   tp_size=layer.tp_size,
                                   quant_mode=quant_mode)
        assert hasattr(
            layer,
            "post_layernorm"), "The layer has no post_rmspost_layernormnorm"
        layer.post_layernorm = SmoothQuantLayerNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)

    setattr(model, 'quant_mode', quant_mode)
    return model


def _smooth_quantize_baichuan(model, quant_mode):
    assert quant_mode.has_act_and_weight_quant()
    for layer in model.layers:
        assert hasattr(layer,
                       "input_layernorm"), "The layer has no input_layernorm"
        layer.input_layernorm = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)
        assert hasattr(layer, "attention"), "The layer has no attention"
        layer.attention = SmoothQuantAttention(
            layer.hidden_size,
            num_attention_heads=layer.num_attention_heads,
            num_kv_heads=layer.num_kv_heads,
            max_position_embeddings=layer.max_position_embeddings,
            num_layers=model.num_layers,
            dtype=layer.dtype,
            attention_mask_type=layer.attention_mask_type,
            position_embedding_type=layer.position_embedding_type,
            rotary_embedding_base=layer.attention.rotary_embedding_base,
            tp_group=layer.tp_group,
            tp_size=layer.tp_size,
            tp_rank=layer.tp_rank,
            quant_mode=quant_mode,
            bias=layer.attention.qkv.bias is not None)

        assert hasattr(layer, "mlp"), "The layer has no mlp"
        if hasattr(model, "moe_config"):
            assert not model.moe_config.has_moe(
            ), "MOE does not support smooth quant"
        layer.mlp = SmoothQuantGatedMLP(hidden_size=model.hidden_size,
                                        ffn_hidden_size=layer.mlp_hidden_size,
                                        hidden_act=layer.hidden_act,
                                        dtype=layer.dtype,
                                        tp_group=layer.tp_group,
                                        tp_size=layer.tp_size,
                                        quant_mode=quant_mode,
                                        bias=layer.mlp.fc.bias is not None)
        assert hasattr(
            layer,
            "post_layernorm"), "The layer has no post_rmspost_layernormnorm"
        layer.post_layernorm = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)

    return model


def _smooth_quantize_internlm(model, quant_mode):
    assert quant_mode.has_act_and_weight_quant()
    for layer in model.layers:
        assert hasattr(layer,
                       "input_layernorm"), "The layer has no input_layernorm"
        layer.input_layernorm = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)
        assert hasattr(layer, "attention"), "The layer has no attention"
        layer.attention = SmoothQuantAttention(
            layer.hidden_size,
            num_attention_heads=layer.num_attention_heads,
            num_kv_heads=layer.num_kv_heads,
            max_position_embeddings=layer.max_position_embeddings,
            num_layers=model.num_layers,
            dtype=layer.dtype,
            attention_mask_type=layer.attention_mask_type,
            position_embedding_type=layer.position_embedding_type,
            tp_group=layer.tp_group,
            tp_size=layer.tp_size,
            quant_mode=quant_mode,
            bias=model.attn_bias)

        assert hasattr(layer, "mlp"), "The layer has no mlp"
        layer.mlp = SmoothQuantGatedMLP(hidden_size=model.hidden_size,
                                        ffn_hidden_size=layer.mlp_hidden_size,
                                        hidden_act=layer.hidden_act,
                                        dtype=layer.dtype,
                                        tp_group=layer.tp_group,
                                        tp_size=layer.tp_size,
                                        quant_mode=quant_mode,
                                        bias=False)
        assert hasattr(
            layer,
            "post_layernorm"), "The layer has no post_rmspost_layernormnorm"
        layer.post_layernorm = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)

    setattr(model, 'quant_mode', quant_mode)
    return model


def _smooth_quantize_qwen(model, quant_mode):
    assert quant_mode.has_act_and_weight_quant()
    for layer in model.layers:
        assert hasattr(layer, "ln_1"), "The layer has no ln_1"
        layer.ln_1 = SmoothQuantRmsNorm(normalized_shape=layer.hidden_size,
                                        dtype=layer.dtype,
                                        quant_mode=quant_mode)
        assert hasattr(layer, "attention"), "The layer has no attention"
        layer.attention = SmoothQuantAttention(
            layer.hidden_size,
            layer.num_attention_heads,
            max_position_embeddings=layer.max_position_embeddings,
            num_layers=layer.num_layers,
            apply_query_key_layer_scaling=layer.apply_query_key_layer_scaling,
            attention_mask_type=layer.attention_mask_type,
            bias=layer.bias,
            qkv_bias_only=True,
            dtype=layer.dtype,
            position_embedding_type=layer.position_embedding_type,
            tp_group=layer.tp_group,
            tp_size=layer.tp_size,
            quant_mode=quant_mode)
        assert hasattr(layer, "mlp"), "The layer has no mlp"
        layer.mlp = SmoothQuantGatedMLP(hidden_size=layer.hidden_size,
                                        ffn_hidden_size=layer.mlp_hidden_size //
                                        2,
                                        hidden_act=layer.hidden_act,
                                        dtype=layer.dtype,
                                        bias=layer.bias,
                                        tp_group=layer.tp_group,
                                        tp_size=layer.tp_size,
                                        quant_mode=quant_mode)
        assert hasattr(layer, "ln_2"), "The layer has no ln_2"
        layer.ln_2 = SmoothQuantRmsNorm(normalized_shape=layer.hidden_size,
                                        dtype=layer.dtype,
                                        quant_mode=quant_mode)

    setattr(model, 'quant_mode', quant_mode)
    return model


def _smooth_quantize_chatglm(model, quant_mode):
    assert quant_mode.has_act_and_weight_quant()
    for layer in model.layers:
        assert hasattr(layer, "pre_norm"), "The layer has no pre_norm"
        layer.pre_norm = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode,
        )
        assert hasattr(layer, "attention"), "The layer has no attention"
        layer.attention = SmoothQuantAttention(
            hidden_size=layer.hidden_size,
            num_attention_heads=layer.num_heads,
            num_kv_heads=layer.num_kv_heads,
            max_position_embeddings=layer.max_seq_length,
            num_layers=layer.num_layers,
            apply_query_key_layer_scaling=layer.apply_query_key_layer_scaling,
            dtype=layer.dtype,
            attention_mask_type=layer.attention_mask_type,
            position_embedding_type=layer.position_embedding_type,
            rotary_embedding_base=layer.rotary_embedding_base,
            tp_group=layer.tp_group,
            tp_size=layer.tp_size,
            quant_mode=quant_mode,
            bias=layer.dense_bias,
            qkv_bias_only=layer.bias and not layer.dense_bias,
        )
        assert hasattr(layer, "mlp"), "The layer has no mlp"
        layer.mlp = SmoothQuantMLP(
            hidden_size=layer.hidden_size,
            ffn_hidden_size=layer.ffn_hidden_size,
            hidden_act=layer.hidden_act,
            dtype=layer.dtype,
            tp_group=layer.tp_group,
            tp_size=layer.tp_size,
            quant_mode=quant_mode,
            bias=layer.dense_bias,
        )
        assert hasattr(layer, "post_norm"), "The layer has no post_norm"
        layer.post_norm = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode,
        )
    return model


def _smooth_quantize(model, quant_mode):
    from ...models import (BaichuanForCausalLM, BloomForCausalLM,
                           ChatGLMHeadModel, GPTLMHeadModel, LLaMAForCausalLM,
                           QWenForCausalLM)
    assert isinstance(model, GPTLMHeadModel) or isinstance(model, LLaMAForCausalLM) \
            or isinstance(model, BloomForCausalLM) or isinstance(model, BaichuanForCausalLM) \
            or isinstance(model, QWenForCausalLM) or isinstance(model, ChatGLMHeadModel), \
            "Only GPTLMHeadModel, LLaMAForCausalLM BloomForCausalLM and BaichuanForCausalLM are well tested now"
    if isinstance(model, GPTLMHeadModel):
        return _smooth_quantize_gpt(model, quant_mode)
    elif isinstance(model, LLaMAForCausalLM):
        return _smooth_quantize_llama(model, quant_mode)
    elif isinstance(model, BloomForCausalLM):
        return _smooth_quantize_bloom(model, quant_mode)
    elif isinstance(model, BaichuanForCausalLM):
        return _smooth_quantize_baichuan(model, quant_mode)
    elif isinstance(model, QWenForCausalLM):
        return _smooth_quantize_qwen(model, quant_mode)
    elif isinstance(model, ChatGLMHeadModel):
        return _smooth_quantize_chatglm(model, quant_mode)
    else:
        assert False, f"Model {type(model).__name__} is not supported by SmoothQuant yet"


def _weight_only_groupwise_quantize(model,
                                    quant_mode,
                                    group_size=128,
                                    pre_quant_scale=False,
                                    zero=False,
                                    exclude_modules=None,
                                    current_key_name=None):
    exclude_modules = ['lm_head'
                       ] if exclude_modules is None else exclude_modules

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if len(list(module.children())) > 0:
            _weight_only_groupwise_quantize(module, quant_mode, group_size,
                                            pre_quant_scale, zero,
                                            exclude_modules, current_key_name)

        if isinstance(module, ColumnLinear) and name not in exclude_modules:
            if not any(key in '.'.join(current_key_name)
                       for key in exclude_modules):
                model._modules[name] = WeightOnlyGroupwiseQuantColumnLinear(
                    in_features=module.in_features,
                    out_features=module.out_features * module.tp_size,
                    group_size=group_size,
                    pre_quant_scale=pre_quant_scale,
                    zero=zero,
                    bias=module.bias is not None,
                    dtype=module.dtype,
                    tp_group=module.tp_group,
                    tp_size=module.tp_size,
                    gather_output=module.gather_output)
        elif isinstance(module, RowLinear) and name not in exclude_modules:
            if not any(key in '.'.join(current_key_name)
                       for key in exclude_modules):
                model._modules[name] = WeightOnlyGroupwiseQuantRowLinear(
                    in_features=module.in_features * module.tp_size,
                    out_features=module.out_features,
                    group_size=group_size,
                    pre_quant_scale=pre_quant_scale,
                    zero=zero,
                    bias=module.bias is not None,
                    dtype=module.dtype,
                    tp_group=module.tp_group,
                    tp_size=module.tp_size)

        current_key_name.pop(-1)

    return model


def quantize_model(model: Module, quant_mode: QuantMode, **kwargs: Any):
    if quant_mode.is_weight_only():
        if quant_mode.has_per_group_scaling():
            model = _weight_only_groupwise_quantize(model, quant_mode, **kwargs)
        else:
            model = weight_only_quantize(model, quant_mode, **kwargs)
    elif quant_mode.has_fp8_qdq() or quant_mode.has_fp8_kv_cache():
        model = _fp8_quantize(model, quant_mode, **kwargs)
    elif quant_mode.has_act_and_weight_quant():
        model = _smooth_quantize(model, quant_mode)

    setattr(model, "quant_mode", quant_mode)
    return model


def get_dummy_quant_scales(num_layers):
    return {
        'lm_head_act': 0.99,
        'lm_head_weights': 0.99,
        'fc_act': [0.99 for _ in range(num_layers)],
        'fc_weights': [0.99 for _ in range(num_layers)],
        'gate_act': [0.99 for _ in range(num_layers)],
        'gate_weights': [0.99 for _ in range(num_layers)],
        'proj_act': [0.99 for _ in range(num_layers)],
        'proj_weights': [0.99 for _ in range(num_layers)],
        'qkv_act': [0.99 for _ in range(num_layers)],
        'qkv_weights': [0.99 for _ in range(num_layers)],
        'qkv_output': [5.0 for _ in range(num_layers)],
        'dense_act': [0.99 for _ in range(num_layers)],
        'dense_weights': [0.99 for _ in range(num_layers)],
    }


def _quantize_layer(layer, layer_idx, quant_mode, quant_scales):
    assert hasattr(layer, "mlp"), "The layer has no mlp"
    fake_fp8_sf_dt = np.float32

    assert isinstance(layer.mlp.fc, (FP8Linear, FP8RowLinear))
    assert isinstance(layer.mlp.proj, (FP8Linear, FP8RowLinear))
    layer.mlp.fc.activation_scaling_factor.value = np.array(
        [quant_scales['fc_act'][layer_idx]], dtype=fake_fp8_sf_dt)
    layer.mlp.fc.weights_scaling_factor.value = np.array(
        [quant_scales['fc_weights'][layer_idx]], dtype=fake_fp8_sf_dt)
    layer.mlp.proj.activation_scaling_factor.value = np.array(
        [quant_scales['proj_act'][layer_idx]], dtype=fake_fp8_sf_dt)
    layer.mlp.proj.weights_scaling_factor.value = np.array(
        [quant_scales['proj_weights'][layer_idx]], dtype=fake_fp8_sf_dt)
    if hasattr(layer.mlp, 'gate'):
        assert isinstance(layer.mlp.gate, (FP8Linear, FP8RowLinear))
        layer.mlp.gate.activation_scaling_factor.value = np.array(
            [quant_scales['gate_act'][layer_idx]], dtype=fake_fp8_sf_dt)
        layer.mlp.gate.weights_scaling_factor.value = np.array(
            [quant_scales['gate_weights'][layer_idx]], dtype=fake_fp8_sf_dt)

    assert hasattr(layer, "attention"), "The layer has no attention"
    assert isinstance(layer.attention.qkv, (FP8Linear, FP8RowLinear))
    assert isinstance(layer.attention.dense, (FP8Linear, FP8RowLinear))
    layer.attention.qkv.activation_scaling_factor.value = np.array(
        [quant_scales['qkv_act'][layer_idx]], dtype=fake_fp8_sf_dt)
    layer.attention.qkv.weights_scaling_factor.value = np.array(
        [quant_scales['qkv_weights'][layer_idx]], dtype=fake_fp8_sf_dt)
    if quant_mode.has_fp8_kv_cache():
        layer.attention.kv_cache_scaling_factor.value = np.array(
            [quant_scales['qkv_output'][layer_idx]], dtype=fake_fp8_sf_dt)
    layer.attention.dense.activation_scaling_factor.value = np.array(
        [quant_scales['dense_act'][layer_idx]], dtype=fake_fp8_sf_dt)
    layer.attention.dense.weights_scaling_factor.value = np.array(
        [quant_scales['dense_weights'][layer_idx]], dtype=fake_fp8_sf_dt)

    return layer


def _default_fp8_quantize(model,
                          quant_mode: QuantMode,
                          quant_scales: dict = None):
    """
    Quantize all linear layers (i.e., MLP, Attention QKV/Dense) and KV cache IO with dummy scales
    This is used by benchmark script and therefore is intentionally decoupled from AMMO toolkit
    """
    if quant_scales is None:
        num_layers = getattr(model, '_num_layers',
                             getattr(model, 'num_layers', None))
        assert num_layers is not None
        quant_scales = get_dummy_quant_scales(num_layers)

    assert model.quant_mode == quant_mode, "Quant setting not consistent with model init setting"

    use_fp8_qdq = quant_mode.has_fp8_qdq()
    assert use_fp8_qdq

    for layer_idx, layer in enumerate(model.layers):
        layer = _quantize_layer(layer, layer_idx, quant_mode, quant_scales)

    # TODO: add lm_head

    return model


def _fp8_quantize(model, quant_mode: QuantMode, quant_scales: dict = None):
    from ...models import (BaichuanForCausalLM, FalconForCausalLM,
                           GPTJForCausalLM, GPTLMHeadModel, LLaMAForCausalLM)
    if isinstance(model, (FalconForCausalLM, GPTJForCausalLM, GPTLMHeadModel,
                          LLaMAForCausalLM, BaichuanForCausalLM)):
        return _default_fp8_quantize(model, quant_mode, quant_scales)
    raise NotImplementedError(
        f"Model {model} is not implemented by fp8_quantize yet")
