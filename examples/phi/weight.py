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
import time
from operator import attrgetter

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.models import PhiForCausalLM


def numpy_split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def torch_split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    else:
        return (torch.split(v, v.shape[dim] // tp_size,
                            dim=dim)[idx]).contiguous()


def load_from_hf_phi(tensorrt_llm_phi: PhiForCausalLM,
                     hf_phi,
                     dtype,
                     rank=0,
                     tp_size=1):

    hf_model_phi_block_names = [
        "input_layernorm.weight",
        "input_layernorm.bias",
        "self_attn.dense.weight",
        "self_attn.dense.bias",
        "mlp.fc1.weight",
        "mlp.fc1.bias",
        "mlp.fc2.weight",
        "mlp.fc2.bias",
    ]

    tensorrt_llm_model_phi_block_names = [
        "input_layernorm.weight",
        "input_layernorm.bias",
        "attention.dense.weight",
        "attention.dense.bias",
        "mlp.fc.weight",
        "mlp.fc.bias",
        "mlp.proj.weight",
        "mlp.proj.bias",
    ]

    tensorrt_llm.logger.info('Loading weights from HF Phi...')

    tik = time.time()

    torch_dtype = str_dtype_to_torch(dtype)
    hf_phi_state_dict = hf_phi.state_dict()

    # [vocab_size, hidden_size]
    v = torch_to_numpy(
        hf_phi_state_dict.get('model.embed_tokens.weight').to(
            torch_dtype).cpu())
    if tensorrt_llm_phi._use_parallel_embedding:
        v = numpy_split(v, tp_size, rank,
                        tensorrt_llm_phi._embedding_sharding_dim)
    tensorrt_llm_phi.vocab_embedding.weight.value = v

    n_layer = hf_phi.config.num_hidden_layers

    for layer_idx in range(n_layer):
        prefix = "model.layers." + str(layer_idx) + "."
        for idx, hf_attr in enumerate(hf_model_phi_block_names):
            v = torch_to_numpy(
                hf_phi_state_dict.get(prefix + hf_attr).to(torch_dtype).cpu())

            layer = attrgetter(tensorrt_llm_model_phi_block_names[idx])(
                tensorrt_llm_phi.layers[layer_idx])

            if tp_size > 1:
                if 'self_attn.dense.weight' in hf_attr:
                    # [n=hidden_size, k=hidden_size] ->
                    # [n=hidden_size, k=hidden_size // tp_size]
                    split_v = numpy_split(v, tp_size, rank, dim=1)
                elif 'mlp.fc1.weight' in hf_attr:
                    # [hidden_size * 4, hidden_size] ->
                    # [hidden_size * 4 // tp_size, hidden_size]
                    split_v = numpy_split(v, tp_size, rank, dim=0)
                elif 'mlp.fc1.bias' in hf_attr:
                    # [hidden_size * 4] -> [hidden_size * 4 // tp_size]
                    split_v = numpy_split(v, tp_size, rank, dim=0)
                elif 'mlp.fc2.weight' in hf_attr:
                    # [hidden_size, hidden_size * 4] ->
                    # [hidden_size, hidden_size * 4 // tp_size]
                    split_v = numpy_split(v, tp_size, rank, dim=1)
                else:
                    split_v = v
                setattr(layer, 'value', split_v)
            else:
                setattr(layer, 'value', v)

        num_heads = hf_phi.config.num_attention_heads
        hidden_size = hf_phi.config.hidden_size
        hidden_size // num_heads

        # Attention QKV Linear
        # [(num_heads x q)|(num_heads x k)|(num_heads x v), hidden_size]
        q_weights = hf_phi_state_dict.get(prefix + "self_attn.q_proj.weight")
        k_weights = hf_phi_state_dict.get(prefix + "self_attn.k_proj.weight")
        v_weights = hf_phi_state_dict.get(prefix + "self_attn.v_proj.weight")
        q_bias = hf_phi_state_dict.get(prefix + "self_attn.q_proj.bias")
        k_bias = hf_phi_state_dict.get(prefix + "self_attn.k_proj.bias")
        v_bias = hf_phi_state_dict.get(prefix + "self_attn.v_proj.bias")
        qkv_weights = torch.cat((q_weights, k_weights, v_weights), dim=0)
        qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)

        qkv_weights = qkv_weights.reshape([hidden_size * 3, hidden_size])
        qkv_bias = qkv_bias.reshape([hidden_size * 3])

        if tp_size > 1:
            qkv_weights = torch_to_numpy(
                qkv_weights.reshape(3, hidden_size,
                                    hidden_size).to(torch_dtype).cpu())
            split_qkv_weights = numpy_split(qkv_weights, tp_size, rank,
                                            dim=1).reshape(
                                                3 * (hidden_size // tp_size),
                                                hidden_size)
            tensorrt_llm_phi.layers[layer_idx].attention.qkv.weight.value = \
                np.ascontiguousarray(split_qkv_weights)

            qkv_bias = torch_to_numpy(
                qkv_bias.reshape(3, hidden_size).to(torch_dtype).cpu())
            split_qkv_bias = numpy_split(qkv_bias, tp_size, rank,
                                         dim=1).reshape(
                                             3 * (hidden_size // tp_size))
            tensorrt_llm_phi.layers[layer_idx].attention.qkv.bias.value = \
                np.ascontiguousarray(split_qkv_bias)
        else:
            tensorrt_llm_phi.layers[layer_idx].attention.qkv.weight.value = \
                torch_to_numpy(qkv_weights.to(torch_dtype).cpu())
            tensorrt_llm_phi.layers[layer_idx].attention.qkv.bias.value = \
                torch_to_numpy(qkv_bias.to(torch_dtype).cpu())

    v = hf_phi_state_dict.get('model.final_layernorm.weight')
    tensorrt_llm_phi.ln_f.weight.value = torch_to_numpy(v.to(torch_dtype).cpu())

    v = hf_phi_state_dict.get('model.final_layernorm.bias')
    tensorrt_llm_phi.ln_f.bias.value = torch_to_numpy(v.to(torch_dtype).cpu())

    v = torch_to_numpy(
        hf_phi_state_dict.get('lm_head.weight').to(torch_dtype).cpu())
    if tp_size > 1:
        # [vocab_size, hidden_size] ->
        # [vocab_size // tp_size, hidden_size]
        if v.shape[0] % tp_size != 0:
            # padding
            vocab_size_padded = pad_vocab_size(v.shape[0], tp_size)
            pad_width = vocab_size_padded - v.shape[0]
            v = np.pad(v, ((0, pad_width), (0, 0)),
                       'constant',
                       constant_values=0)

        split_v = numpy_split(v, tp_size, rank, dim=0)
        tensorrt_llm_phi.lm_head.weight.value = split_v
    else:
        tensorrt_llm_phi.lm_head.weight.value = v

    v = torch_to_numpy(
        hf_phi_state_dict.get('lm_head.bias').to(torch_dtype).cpu())
    split_v = numpy_split(v, tp_size, rank, dim=0)

    tensorrt_llm_phi.lm_head.bias.value = split_v

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
