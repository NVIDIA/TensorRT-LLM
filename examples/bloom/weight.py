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
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def reorder_qkv_weight_or_bias(v, n_head, n_hidden, is_bias=False):
    """ Reorder the qkv weight.

    Note that the shape of the fused QKV weights in HF is different from the
    shape that TRT-LLM requires.
       HF: (num_heads x 3 x head_dim, hidden_size)
       TRT-LLM: (3 x num_heads x head_dim, hidden_size)
    This is unlike to the other models in HF e.g. GPT where they have the
    same shape with TRT-LLM, i.e., (3 x num_heads x head_dim, hidden_size). Also,
    to split across attention heads in tensor parallel, we reshape the qkv
        weight: (3, num_heads x head_dim, hidden).
        bias  : (3, num_heads x head_dim).
    """

    head_dim = n_hidden // n_head

    # (3 x hidden, ...) view as (num_heads, 3, head_dim, ...)
    v = v.reshape(n_head, 3, head_dim, -1)
    # permute to (3, num_heads, head_dim, ...)
    v = v.transpose((1, 0, 2, 3))
    # final shape: weight=(3, hidden, hidden) or bias=(3, hidden)
    if is_bias:
        return v.reshape(3, n_hidden)
    return v.reshape(3, n_hidden, n_hidden)


def split_qkv_tp(tensorrt_llm_bloom, v, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    n_heads = tensorrt_llm_bloom._num_heads
    hidden_size = tensorrt_llm_bloom._hidden_size
    v = reorder_qkv_weight_or_bias(v, n_heads, hidden_size, is_bias=False)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (hidden_size // tensor_parallel), hidden_size)

    return np.ascontiguousarray(split_v)


def split_qkv_bias_tp(tensorrt_llm_bloom, v, tensor_parallel, rank):
    """
    Splits the QKV bias according to tensor parallelism
    """
    layer = tensorrt_llm_bloom.layers[0]
    n_heads = layer.num_attention_heads
    hidden_size = layer.hidden_size
    v = reorder_qkv_weight_or_bias(v, n_heads, hidden_size, is_bias=True)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (hidden_size // tensor_parallel))
    return np.ascontiguousarray(split_v)


def split_matrix_tp(v, tensor_parallel, rank, dim):
    return np.ascontiguousarray(split(v, tensor_parallel, rank, dim=dim))


def get_weight(config, prefix, dtype):
    return config[prefix + '.weight'].to(dtype).detach().cpu().numpy()


def get_bias(config, prefix, dtype):
    return config[prefix + '.bias'].to(dtype).detach().cpu().numpy()


def get_weight_and_bias(config, prefix, dtype):
    return get_weight(config, prefix, dtype), get_bias(config, prefix, dtype)


def check_embedding_share(dir_path):
    share_embedding_table = False
    if Path(dir_path).exists():
        share_embedding_table = True
    return share_embedding_table


def load_from_hf_bloom(tensorrt_llm_bloom,
                       hf_bloom,
                       rank=0,
                       tensor_parallel=1,
                       fp16=False,
                       use_parallel_embedding=False,
                       sharding_dim=0,
                       share_embedding_table=False):
    tensorrt_llm.logger.info('Loading weights from HF BLOOM...')
    tik = time.time()

    model_params = dict(hf_bloom.named_parameters())
    dtype = torch.float16 if fp16 else torch.float32
    for l in range(hf_bloom.config.num_hidden_layers):
        prefix = f'transformer.h.{l}.'

        qkv_weight, qkv_bias = get_weight_and_bias(
            model_params, prefix + 'self_attention.query_key_value', dtype)
        tensorrt_llm_bloom.layers[l].attention.qkv.weight.value = split_qkv_tp(
            tensorrt_llm_bloom, qkv_weight, tensor_parallel, rank)
        tensorrt_llm_bloom.layers[
            l].attention.qkv.bias.value = split_qkv_bias_tp(
                tensorrt_llm_bloom, qkv_bias, tensor_parallel, rank)

        attn_dense_weight, attn_dense_bias = get_weight_and_bias(
            model_params, prefix + 'self_attention.dense', dtype)
        tensorrt_llm_bloom.layers[
            l].attention.dense.weight.value = split_matrix_tp(attn_dense_weight,
                                                              tensor_parallel,
                                                              rank,
                                                              dim=1)
        tensorrt_llm_bloom.layers[
            l].attention.dense.bias.value = attn_dense_bias

        mlp_fc_weight, mlp_fc_bias = get_weight_and_bias(
            model_params, prefix + 'mlp.dense_h_to_4h', dtype)
        tensorrt_llm_bloom.layers[l].mlp.fc.weight.value = split_matrix_tp(
            mlp_fc_weight, tensor_parallel, rank, dim=0)
        tensorrt_llm_bloom.layers[l].mlp.fc.bias.value = split_matrix_tp(
            mlp_fc_bias, tensor_parallel, rank, dim=0)

        mlp_proj_weight, mlp_proj_bias = get_weight_and_bias(
            model_params, prefix + 'mlp.dense_4h_to_h', dtype)
        tensorrt_llm_bloom.layers[l].mlp.proj.weight.value = split_matrix_tp(
            mlp_proj_weight, tensor_parallel, rank, dim=1)
        tensorrt_llm_bloom.layers[l].mlp.proj.bias.value = mlp_proj_bias

        # Layer norms do not use tensor parallelism
        input_ln_weight, input_ln_bias = get_weight_and_bias(
            model_params, prefix + 'input_layernorm', dtype)
        tensorrt_llm_bloom.layers[
            l].input_layernorm.weight.value = input_ln_weight
        tensorrt_llm_bloom.layers[l].input_layernorm.bias.value = input_ln_bias

        post_ln_weight, post_ln_bias = get_weight_and_bias(
            model_params, prefix + 'post_attention_layernorm', dtype)
        tensorrt_llm_bloom.layers[
            l].post_layernorm.weight.value = post_ln_weight
        tensorrt_llm_bloom.layers[l].post_layernorm.bias.value = post_ln_bias

    embed_w = get_weight(model_params, 'transformer.word_embeddings', dtype)
    if not share_embedding_table:
        tensorrt_llm_bloom.lm_head.weight.value = split_matrix_tp(
            embed_w.copy(), tensor_parallel, rank, dim=0)

    if not use_parallel_embedding:
        tensorrt_llm_bloom.embedding.weight.value = embed_w
    else:
        assert hf_bloom.config.vocab_size % tensor_parallel == 0
        tensorrt_llm_bloom.embedding.weight.value = split_matrix_tp(
            embed_w, tensor_parallel, rank, dim=sharding_dim)

    embed_f_w, embed_f_b = get_weight_and_bias(
        model_params, 'transformer.word_embeddings_layernorm', dtype)
    tensorrt_llm_bloom.ln_embed.weight.value = embed_f_w
    tensorrt_llm_bloom.ln_embed.bias.value = embed_f_b

    ln_f_w, ln_f_b = get_weight_and_bias(model_params, 'transformer.ln_f',
                                         dtype)
    tensorrt_llm_bloom.ln_f.weight.value = ln_f_w
    tensorrt_llm_bloom.ln_f.bias.value = ln_f_b

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
