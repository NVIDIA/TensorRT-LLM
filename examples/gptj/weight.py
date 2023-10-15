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
import time
from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm.models import GPTJForCausalLM
from tensorrt_llm.models.quantized.quant import get_dummy_quant_scales
from tensorrt_llm.quantization import QuantMode


def get_scaling_factors(
    model_path: Union[str, Path],
    num_layers: int,
    quant_mode: Optional[QuantMode] = None,
) -> Optional[Dict[str, List[int]]]:
    """ Get the scaling factors for GPT-J model

    Returns a dictionary of scaling factors for the selected layers of the
    GPT-J model.

    Args:
        model_path (str): Path to the quantized GPT-J model
        layers (list): List of layers to get the scaling factors for. If None,
            all layers are selected.

    Returns:
        dict: Dictionary of scaling factors for the selected layers of the
        GPT-J model.

        example:

        {
            'qkv_act': qkv_act_scale,
            'qkv_weights': qkv_weights_scale,
            'qkv_output' : qkv_outputs_scale,
            'dense_act': dense_act_scale,
            'dense_weights': dense_weights_scale,
            'fc_act': fc_act_scale,
            'fc_weights': fc_weights_scale,
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
        scaling_factor['proj_act'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:activation_scaling_factor'].item())
        scaling_factor['proj_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:weights_scaling_factor'].item())
    # yapf: enable
    for k, v in scaling_factor.items():
        assert len(v) == num_layers, \
        f'Expect scaling factor {k} of length {num_layers}, got {len(v)}'

    return scaling_factor


def load_from_hf_gpt_j(tensorrt_llm_gpt_j: GPTJForCausalLM,
                       hf_gpt_j,
                       fp16=False,
                       scaling_factors=None):

    hf_model_gptj_block_names = [
        "ln_1.weight",
        "ln_1.bias",
        "mlp.fc_in.weight",
        "mlp.fc_in.bias",
        "mlp.fc_out.weight",
        "mlp.fc_out.bias",
    ]

    tensorrt_llm_model_gptj_block_names = [
        "input_layernorm.weight",
        "input_layernorm.bias",
        "mlp.fc.weight",
        "mlp.fc.bias",
        "mlp.proj.weight",
        "mlp.proj.bias",
    ]

    quant_mode = getattr(tensorrt_llm_gpt_j, 'quant_mode', QuantMode(0))

    tensorrt_llm.logger.info('Loading weights from HF GPT-J...')
    tik = time.time()

    torch_dtype = torch.float16 if fp16 else torch.float32
    hf_gpt_j_state_dict = hf_gpt_j.state_dict()

    v = hf_gpt_j_state_dict.get('transformer.wte.weight')
    tensorrt_llm_gpt_j.embedding.weight.value = v.to(torch_dtype).cpu().numpy()

    n_layer = hf_gpt_j.config.n_layer

    for layer_idx in range(n_layer):
        prefix = "transformer.h." + str(layer_idx) + "."
        for idx, hf_attr in enumerate(hf_model_gptj_block_names):
            v = hf_gpt_j_state_dict.get(prefix + hf_attr)
            layer = attrgetter(tensorrt_llm_model_gptj_block_names[idx])(
                tensorrt_llm_gpt_j.layers[layer_idx])
            if idx == 2 and scaling_factors:
                tensorrt_llm_gpt_j.layers[
                    layer_idx].mlp.fc.activation_scaling_factor.value = np.array(
                        [scaling_factors['fc_act'][layer_idx]],
                        dtype=np.float32)

                tensorrt_llm_gpt_j.layers[
                    layer_idx].mlp.fc.weights_scaling_factor.value = np.array(
                        [scaling_factors['fc_weights'][layer_idx]],
                        dtype=np.float32)

            elif idx == 4 and scaling_factors:
                tensorrt_llm_gpt_j.layers[
                    layer_idx].mlp.proj.activation_scaling_factor.value = np.array(
                        [scaling_factors['proj_act'][layer_idx]],
                        dtype=np.float32)

                tensorrt_llm_gpt_j.layers[
                    layer_idx].mlp.proj.weights_scaling_factor.value = np.array(
                        [scaling_factors['proj_weights'][layer_idx]],
                        dtype=np.float32)
            setattr(layer, 'value', v.to(torch_dtype).cpu().numpy())

        # Attention QKV Linear
        # concatenate the Q, K, V layers weights.
        q_weights = hf_gpt_j_state_dict.get(prefix + "attn.q_proj.weight")
        k_weights = hf_gpt_j_state_dict.get(prefix + "attn.k_proj.weight")
        v_weights = hf_gpt_j_state_dict.get(prefix + "attn.v_proj.weight")
        qkv_weights = torch.cat((q_weights, k_weights, v_weights))
        layer = attrgetter("attention.qkv.weight")(
            tensorrt_llm_gpt_j.layers[layer_idx])
        setattr(layer, "value", qkv_weights.to(torch_dtype).cpu().numpy())
        if scaling_factors:
            tensorrt_llm_gpt_j.layers[
                layer_idx].attention.qkv.activation_scaling_factor.value = np.array(
                    [scaling_factors['qkv_act'][layer_idx]], dtype=np.float32)
            tensorrt_llm_gpt_j.layers[
                layer_idx].attention.qkv.weights_scaling_factor.value = np.array(
                    [scaling_factors['qkv_weights'][layer_idx]],
                    dtype=np.float32)

        if quant_mode.has_fp8_kv_cache():
            if scaling_factors:
                tensorrt_llm_gpt_j.layers[
                    layer_idx].attention.kv_orig_quant_scale.value = np.array(
                        [scaling_factors['qkv_output'][layer_idx]],
                        dtype=np.float32)
                tensorrt_llm_gpt_j.layers[
                    layer_idx].attention.kv_quant_orig_scale.value = np.array(
                        [1.0 / scaling_factors['qkv_output'][layer_idx]],
                        dtype=np.float32)

        # Attention Dense (out_proj) Linear
        v = hf_gpt_j_state_dict.get(prefix + "attn.out_proj.weight")
        layer = attrgetter("attention.dense.weight")(
            tensorrt_llm_gpt_j.layers[layer_idx])
        setattr(layer, "value", v.to(torch_dtype).cpu().numpy())
        if scaling_factors:
            tensorrt_llm_gpt_j.layers[
                layer_idx].attention.dense.activation_scaling_factor.value = np.array(
                    [scaling_factors['dense_act'][layer_idx]], dtype=np.float32)
            tensorrt_llm_gpt_j.layers[
                layer_idx].attention.dense.weights_scaling_factor.value = np.array(
                    [scaling_factors['dense_weights'][layer_idx]],
                    dtype=np.float32)

    v = hf_gpt_j_state_dict.get('transformer.ln_f.weight')
    tensorrt_llm_gpt_j.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()

    v = hf_gpt_j_state_dict.get('transformer.ln_f.bias')
    tensorrt_llm_gpt_j.ln_f.bias.value = v.to(torch_dtype).cpu().numpy()

    v = hf_gpt_j_state_dict.get('lm_head.weight')
    tensorrt_llm_gpt_j.lm_head.weight.value = v.to(torch_dtype).cpu().numpy()

    v = hf_gpt_j_state_dict.get('lm_head.bias')
    tensorrt_llm_gpt_j.lm_head.bias.value = v.to(torch_dtype).cpu().numpy()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


def AWQ_quantize_pack_preprocess(weight, scale, group_size, packer,
                                 preprocessor):
    scale = scale.repeat_interleave(group_size, dim=0)
    weight = weight / scale
    weight = torch.round(weight).char()
    weight = torch.where(weight > 7, 7, weight)
    qweight_int8 = torch.where(weight < -8, -8, weight)
    int4_weight = packer(qweight_int8.cpu())
    int4_weight = preprocessor(int4_weight, torch.quint4x2)
    return int4_weight.view(torch.float32).cpu().numpy()


def process_and_assign_weight(awq_gpt_j, mPrefix, mOp, group_size, packer,
                              preprocessor, torch_dtype):
    weight = awq_gpt_j[mPrefix + ".weight"].T.contiguous()
    [k, n] = weight.shape
    amax = awq_gpt_j[mPrefix + ".weight_quantizer._amax"].reshape(
        (n, int(k / group_size))).T.contiguous()
    pre_quant_scale = awq_gpt_j[mPrefix +
                                ".input_quantizer._pre_quant_scale"].reshape(
                                    (1, k))
    scale = amax / 8.0
    mOp.qweight.value = AWQ_quantize_pack_preprocess(weight, scale, group_size,
                                                     packer, preprocessor)
    mOp.scale.value = scale.to(torch_dtype).cpu().numpy()
    mOp.pre_quant_scale.value = pre_quant_scale.to(torch_dtype).cpu().numpy()


def deSmooth(weight, pre_quant_scale):
    [k, n] = weight.shape
    pre_quant_scale = pre_quant_scale.repeat((n, 1)).transpose(1,
                                                               0).contiguous()
    weight = weight * pre_quant_scale
    return weight


def reSmooth(weight, pre_quant_scale):
    [k, n] = weight.shape
    pre_quant_scale = pre_quant_scale.repeat((n, 1)).transpose(1,
                                                               0).contiguous()
    weight = weight / pre_quant_scale
    return weight


def get_scale(weight, group_size):
    weight = weight.T.contiguous()
    [n, k] = weight.shape
    weight = weight.reshape(n, int(k / group_size), group_size)
    weight = torch.abs(weight.reshape(-1, group_size))
    amax, idx = weight.max(1)
    amax = amax.reshape(n, int(k / group_size)).T.contiguous()
    return amax / 8


def reSmooth_and_get_scale(weight, pre_quant_scale, avg_pre_quant_scale,
                           group_size):
    weight = deSmooth(weight, pre_quant_scale)
    weight = reSmooth(weight, avg_pre_quant_scale)
    scale = get_scale(weight, group_size)
    return weight, scale


def process_and_assign_qkv_weight(awq_gpt_j, prefix, mOp, group_size, packer,
                                  preprocessor, torch_dtype):
    q_weight = awq_gpt_j[prefix + "attn.q_proj.weight"].T.contiguous()
    k_weight = awq_gpt_j[prefix + "attn.k_proj.weight"].T.contiguous()
    v_weight = awq_gpt_j[prefix + "attn.v_proj.weight"].T.contiguous()
    [k, n] = q_weight.shape

    q_pre_quant_scale = awq_gpt_j[
        prefix + "attn.q_proj.input_quantizer._pre_quant_scale"].reshape((1, k))
    k_pre_quant_scale = awq_gpt_j[
        prefix + "attn.k_proj.input_quantizer._pre_quant_scale"].reshape((1, k))
    v_pre_quant_scale = awq_gpt_j[
        prefix + "attn.v_proj.input_quantizer._pre_quant_scale"].reshape((1, k))

    qkv_pre_quant_scale = (q_pre_quant_scale + k_pre_quant_scale +
                           v_pre_quant_scale) / 3.0
    q_weight, q_scale = reSmooth_and_get_scale(q_weight, q_pre_quant_scale,
                                               qkv_pre_quant_scale, group_size)
    k_weight, k_scale = reSmooth_and_get_scale(k_weight, k_pre_quant_scale,
                                               qkv_pre_quant_scale, group_size)
    v_weight, v_scale = reSmooth_and_get_scale(v_weight, v_pre_quant_scale,
                                               qkv_pre_quant_scale, group_size)

    qkv_weights = torch.cat((q_weight, k_weight, v_weight), dim=1)
    qkv_scale = torch.cat((q_scale, k_scale, v_scale), dim=1)
    mOp.pre_quant_scale.value = qkv_pre_quant_scale.to(
        torch_dtype).cpu().numpy()
    mOp.qweight.value = AWQ_quantize_pack_preprocess(qkv_weights, qkv_scale,
                                                     group_size, packer,
                                                     preprocessor)
    mOp.scale.value = qkv_scale.to(torch_dtype).cpu().numpy()


def load_from_awq_gpt_j(tensorrt_llm_gpt_j: GPTJForCausalLM,
                        awq_gpt_j,
                        config,
                        fp16=False,
                        group_size=128):

    awq_gptj_block_names = [
        "ln_1.weight",
        "ln_1.bias",
        "mlp.fc_in.bias",
        "mlp.fc_out.bias",
    ]

    tensorrt_llm_model_gptj_block_names = [
        "input_layernorm.weight",
        "input_layernorm.bias",
        "mlp.fc.bias",
        "mlp.proj.bias",
    ]

    getattr(tensorrt_llm_gpt_j, 'quant_mode', QuantMode(0))

    packer = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm

    tensorrt_llm.logger.info('Loading weights from AWQ GPT-J...')
    tik = time.time()

    torch_dtype = torch.float16 if fp16 else torch.float32

    #check if we need to pad vocab
    v = awq_gpt_j.get('transformer.wte.weight')
    [vocab_size, k] = v.shape
    pad_vocab = False
    pad_vocab_size = vocab_size
    if vocab_size % 64 != 0:
        pad_vocab = True
        pad_vocab_size = int((vocab_size + 63) / 64) * 64
    if pad_vocab:
        new_v = torch.zeros([pad_vocab_size, k])
        new_v[:vocab_size, :] = v
        v = new_v
    tensorrt_llm_gpt_j.embedding.weight.value = v.to(torch_dtype).cpu().numpy()

    n_layer = config["n_layer"]

    for layer_idx in range(n_layer):
        prefix = "transformer.h." + str(layer_idx) + "."
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        for idx, awq_attr in enumerate(awq_gptj_block_names):
            v = awq_gpt_j[prefix + awq_attr]
            layer = attrgetter(tensorrt_llm_model_gptj_block_names[idx])(
                tensorrt_llm_gpt_j.layers[layer_idx])
            setattr(layer, 'value', v.to(torch_dtype).cpu().numpy())

        # Attention QKV Linear
        # concatenate the Q, K, V layers weights.
        process_and_assign_qkv_weight(
            awq_gpt_j, prefix,
            tensorrt_llm_gpt_j.layers[layer_idx].attention.qkv, group_size,
            packer, preprocessor, torch_dtype)

        # Attention Dense (out_proj) Linear
        mPrefix = prefix + "attn.out_proj"
        mOp = tensorrt_llm_gpt_j.layers[layer_idx].attention.dense
        process_and_assign_weight(awq_gpt_j, mPrefix, mOp, group_size, packer,
                                  preprocessor, torch_dtype)

        # MLP Dense (mlp.fc) Linear
        mPrefix = prefix + "mlp.fc_in"
        mOp = tensorrt_llm_gpt_j.layers[layer_idx].mlp.fc
        process_and_assign_weight(awq_gpt_j, mPrefix, mOp, group_size, packer,
                                  preprocessor, torch_dtype)

        # MLP Desne (mlp.proj) Linear
        mPrefix = prefix + "mlp.fc_out"
        mOp = tensorrt_llm_gpt_j.layers[layer_idx].mlp.proj
        process_and_assign_weight(awq_gpt_j, mPrefix, mOp, group_size, packer,
                                  preprocessor, torch_dtype)

    v = awq_gpt_j['transformer.ln_f.weight']
    tensorrt_llm_gpt_j.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()

    v = awq_gpt_j['transformer.ln_f.bias']
    tensorrt_llm_gpt_j.ln_f.bias.value = v.to(torch_dtype).cpu().numpy()

    #lm_head
    if pad_vocab:
        weight = awq_gpt_j['lm_head.weight']
        [vocab_size, k] = weight.shape
        new_weight = torch.zeros([pad_vocab_size, k])
        new_weight[:vocab_size, :] = weight
        new_weight = new_weight.T.contiguous()
        amax = awq_gpt_j['lm_head.weight_quantizer._amax'].reshape(
            [vocab_size, int(k / group_size)])
        new_amax = torch.ones([pad_vocab_size, int(k / group_size)])
        new_amax[:vocab_size, :] = amax
        new_amax = new_amax.T.contiguous()
        new_scale = new_amax / 8
        tensorrt_llm_gpt_j.lm_head.qweight.value = AWQ_quantize_pack_preprocess(
            new_weight, new_scale, group_size, packer, preprocessor)
        tensorrt_llm_gpt_j.lm_head.scale.value = new_scale.to(
            torch_dtype).cpu().numpy()
        tensorrt_llm_gpt_j.lm_head.pre_quant_scale.value = awq_gpt_j[
            'lm_head.input_quantizer._pre_quant_scale'].to(
                torch_dtype).cpu().numpy()

        bias = awq_gpt_j['lm_head.bias']
        new_bias = torch.zeros([pad_vocab_size])
        new_bias[:vocab_size] = bias
        tensorrt_llm_gpt_j.lm_head.bias.value = new_bias.to(
            torch_dtype).cpu().numpy()
    else:
        mPrefix = "lm_head"
        mOp = tensorrt_llm_gpt_j.lm_head
        process_and_assign_weight(awq_gpt_j, mPrefix, mOp, group_size, packer,
                                  preprocessor, torch_dtype)

        v = awq_gpt_j['lm_head.bias']
        tensorrt_llm_gpt_j.lm_head.bias.value = v.to(torch_dtype).cpu().numpy()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
