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

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_np
from tensorrt_llm.models import BloomForCausalLM
from tensorrt_llm.quantization import QuantMode


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


def set_layer_weight(layer, val, quant_mode):
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    if use_weight_only:
        v = np.ascontiguousarray(val.transpose())
        processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
            torch.tensor(v), plugin_weight_only_quant_type)
        # workaround for trt not supporting int8 inputs in plugins currently
        layer.weight.value = processed_torch_weights.view(
            dtype=torch.float32).numpy()
        layer.per_channel_scale.value = torch_weight_scales.numpy()
    else:
        layer.weight.value = np.ascontiguousarray(val)


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

    quant_mode = getattr(tensorrt_llm_bloom, 'quant_mode', QuantMode(0))

    model_params = dict(hf_bloom.named_parameters())
    dtype = torch.float16 if fp16 else torch.float32
    for l in range(hf_bloom.config.num_hidden_layers):
        prefix = f'transformer.h.{l}.'

        qkv_weight, qkv_bias = get_weight_and_bias(
            model_params, prefix + 'self_attention.query_key_value', dtype)
        split_v = split_qkv_tp(tensorrt_llm_bloom, qkv_weight, tensor_parallel,
                               rank)
        set_layer_weight(tensorrt_llm_bloom.layers[l].attention.qkv, split_v,
                         quant_mode)
        tensorrt_llm_bloom.layers[
            l].attention.qkv.bias.value = split_qkv_bias_tp(
                tensorrt_llm_bloom, qkv_bias, tensor_parallel, rank)

        attn_dense_weight, attn_dense_bias = get_weight_and_bias(
            model_params, prefix + 'self_attention.dense', dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  tensor_parallel,
                                  rank,
                                  dim=1)
        set_layer_weight(tensorrt_llm_bloom.layers[l].attention.dense, split_v,
                         quant_mode)
        tensorrt_llm_bloom.layers[
            l].attention.dense.bias.value = attn_dense_bias

        mlp_fc_weight, mlp_fc_bias = get_weight_and_bias(
            model_params, prefix + 'mlp.dense_h_to_4h', dtype)
        split_v = split_matrix_tp(mlp_fc_weight, tensor_parallel, rank, dim=0)
        set_layer_weight(tensorrt_llm_bloom.layers[l].mlp.fc, split_v,
                         quant_mode)
        tensorrt_llm_bloom.layers[l].mlp.fc.bias.value = split_matrix_tp(
            mlp_fc_bias, tensor_parallel, rank, dim=0)

        mlp_proj_weight, mlp_proj_bias = get_weight_and_bias(
            model_params, prefix + 'mlp.dense_4h_to_h', dtype)
        split_v = split_matrix_tp(mlp_proj_weight, tensor_parallel, rank, dim=1)
        set_layer_weight(tensorrt_llm_bloom.layers[l].mlp.proj, split_v,
                         quant_mode)
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


def parse_config(ini_file):
    bloom_config = configparser.ConfigParser()
    bloom_config.read(ini_file)

    n_embd = bloom_config.getint('bloom', 'hidden_size')
    n_head = bloom_config.getint('bloom', 'n_head')
    n_layer = bloom_config.getint('bloom', 'n_layer')
    vocab_size = bloom_config.getint('bloom', 'vocab_size')
    do_layer_norm_before = bloom_config.getboolean('bloom',
                                                   'do_layer_norm_before',
                                                   fallback=True)
    rotary_pct = bloom_config.getfloat('bloom', 'rotary_pct', fallback=0.0)
    bias = bloom_config.getboolean('bloom', 'bias', fallback=True)
    inter_size = bloom_config.getint('bloom',
                                     'intermediate_size',
                                     fallback=None)
    dtype = bloom_config.get('bloom', 'storage_dtype', fallback='float32')

    if inter_size is None:
        inter_size = 4 * n_embd

    multi_query_mode = bloom_config.getboolean('bloom',
                                               'multi_query_mode',
                                               fallback=False)
    prompt_num_tasks = bloom_config.getint('bloom',
                                           'prompt_num_tasks',
                                           fallback=0)
    prompt_max_vocab_size = bloom_config.getint('bloom',
                                                'prompt_max_vocab_size',
                                                fallback=0)
    return n_embd, n_head, n_layer, vocab_size, do_layer_norm_before, rotary_pct, bias, inter_size, multi_query_mode, dtype, prompt_num_tasks, prompt_max_vocab_size


def load_from_bin(tensorrt_llm_bloom: BloomForCausalLM,
                  dir_path,
                  rank=0,
                  tensor_parallel=1,
                  dtype='float32',
                  use_parallel_embedding=False,
                  sharding_dim=0,
                  share_embedding_table=False):
    tensorrt_llm.logger.info('Loading weights from bin...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_bloom, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        torch.int8
    elif quant_mode.is_int4_weight_only():
        torch.quint4x2
    n_embd, n_head, n_layer, vocab_size, do_layer_norm_before, rotary_pct, bias, inter_size, multi_query_mode, *_ = parse_config(
        Path(dir_path) / 'config.ini')
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

    def set_smoother(module, dir_path, base_name, shape, rank):
        suffix = f"{rank}.bin"
        t = fromfile(dir_path, f"{base_name}.smoother.{suffix}", shape,
                     np.float32)
        module.smoother.value = t

    # Determine the quantization mode.
    quant_mode = getattr(tensorrt_llm_bloom, "quant_mode", QuantMode(0))
    # Do we use SmoothQuant?
    use_smooth_quant = quant_mode.has_act_and_weight_quant()
    # Do we use quantization per token?
    quant_per_token_dyn = quant_mode.has_per_token_dynamic_scaling()
    # Do we use quantization per channel?
    quant_per_channel = quant_mode.has_per_channel_scaling()

    # Do we use INT4/INT8 weight-only?
    quant_mode.is_weight_only()

    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    def sq_trick(x):
        return x.view(np.float32) if use_smooth_quant else x

    # Debug
    suffix = gen_suffix(rank, use_smooth_quant, quant_per_channel)
    # The type of weights.
    w_type = np_dtype if not use_smooth_quant else np.int8

    vocab_embedding_weight = (fromfile(dir_path, 'model.wpe.bin',
                                       [vocab_size, n_embd]))
    embed_w = np.ascontiguousarray(
        split(vocab_embedding_weight.copy(), tensor_parallel, rank))
    if not share_embedding_table:
        tensorrt_llm_bloom.lm_head.weight.value = embed_w

    if not use_parallel_embedding:
        tensorrt_llm_bloom.embedding.weight.value = np.ascontiguousarray(
            vocab_embedding_weight)
    else:
        assert vocab_size % tensor_parallel == 0
        tensorrt_llm_bloom.embedding.weight.value = np.ascontiguousarray(
            split(vocab_embedding_weight,
                  tensor_parallel,
                  rank,
                  dim=sharding_dim))

    tensorrt_llm_bloom.ln_embed.bias.value = (fromfile(
        dir_path, 'model.word_embeddings_layernorm.bias.bin'))
    tensorrt_llm_bloom.ln_embed.weight.value = (fromfile(
        dir_path, 'model.word_embeddings_layernorm.weight.bin'))

    tensorrt_llm_bloom.ln_f.bias.value = (fromfile(
        dir_path, 'model.final_layernorm.bias.bin'))
    tensorrt_llm_bloom.ln_f.weight.value = (fromfile(
        dir_path, 'model.final_layernorm.weight.bin'))

    for i in range(n_layer):
        c_attn_out_dim = (3 * n_embd //
                          tensor_parallel) if not multi_query_mode else (
                              n_embd // tensor_parallel +
                              (n_embd // n_head) * 2)
        tensorrt_llm_bloom.layers[i].input_layernorm.weight.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.weight.bin'))
        tensorrt_llm_bloom.layers[i].input_layernorm.bias.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.bias.bin'))

        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.weight.' + suffix,
            [n_embd, c_attn_out_dim], w_type)
        if t is not None:
            layer = tensorrt_llm_bloom.layers[i].attention.qkv
            if use_smooth_quant:
                layer.weight.value = sq_trick(
                    np.ascontiguousarray(np.transpose(t, [1, 0])))
                set_smoothquant_scale_factors(
                    layer,
                    tensorrt_llm_bloom.layers[i].input_layernorm.scale_to_int,
                    dir_path,
                    'model.layers.' + str(i) + '.attention.query_key_value.',
                    [1, c_attn_out_dim],
                    quant_per_token_dyn,
                    quant_per_channel,
                    rank=rank,
                    is_qkv=True)
            else:
                set_layer_weight(layer, np.transpose(t, [1, 0]), quant_mode)
        if bias:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.bias.' + str(rank) + '.bin')
            if t is not None:
                layer.bias.value = np.ascontiguousarray(t)

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attention.dense.weight.' + suffix,
            [n_embd // tensor_parallel, n_embd], w_type)
        layer = tensorrt_llm_bloom.layers[i].attention.dense
        if use_smooth_quant:
            layer.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            dense_scale = getattr(tensorrt_llm_bloom.layers[i].attention,
                                  "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                layer, dense_scale, dir_path,
                'model.layers.' + str(i) + '.attention.dense.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
            # set it to ones if dense layer is not applied smooth quant
            # layer.smoother.value = np.ones(
            #     [1, n_embd // tensor_parallel], dtype=np.float32)
            # set it to the real smoother if dense layer is applied smooth quant
            set_smoother(layer, dir_path,
                         'model.layers.' + str(i) + '.attention.dense',
                         [1, n_embd // tensor_parallel], rank)
        else:
            set_layer_weight(layer, np.transpose(t, [1, 0]), quant_mode)
        if bias:
            layer.bias.value = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.attention.dense.bias.bin')

        dst = tensorrt_llm_bloom.layers[i].post_layernorm.weight
        dst.value = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.post_attention_layernorm.weight.bin')
        dst = tensorrt_llm_bloom.layers[i].post_layernorm.bias
        dst.value = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.post_attention_layernorm.bias.bin')

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.dense_h_to_4h.weight.' + suffix,
            [n_embd, inter_size // tensor_parallel], w_type)
        layer = tensorrt_llm_bloom.layers[i].mlp.fc
        if use_smooth_quant:
            layer.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            set_smoothquant_scale_factors(
                layer,
                tensorrt_llm_bloom.layers[i].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.dense_h_to_4h.',
                [1, inter_size // tensor_parallel],
                quant_per_token_dyn,
                quant_per_channel,
                rank=rank)
        else:
            set_layer_weight(layer, np.transpose(t, [1, 0]), quant_mode)
        if bias:
            layer.bias.value = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.mlp.dense_h_to_4h.bias.' + str(rank) + '.bin')

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.dense_4h_to_h.weight.' + suffix,
            [inter_size // tensor_parallel, n_embd], w_type)
        layer = tensorrt_llm_bloom.layers[i].mlp.proj
        if use_smooth_quant:
            layer.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            proj_scale = getattr(tensorrt_llm_bloom.layers[i].mlp,
                                 "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                layer, proj_scale, dir_path,
                'model.layers.' + str(i) + '.mlp.dense_4h_to_h.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
            # set it to ones if proj layer is not applied smooth quant
            # layer.smoother.value = np.ones(
            #     [1, inter_size // tensor_parallel], dtype=np.float32)
            # set it to the real smoother if proj layer is applied smooth quant
            set_smoother(layer, dir_path,
                         'model.layers.' + str(i) + '.mlp.dense_4h_to_h',
                         [1, inter_size // tensor_parallel], rank)
        else:
            set_layer_weight(layer, np.transpose(t, [1, 0]), quant_mode)
        if bias:
            layer.bias.value = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.mlp.dense_4h_to_h.bias.bin')

        if use_int8_kv_cache:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            tensorrt_llm_bloom.layers[
                i].attention.kv_orig_quant_scale.value = 1.0 / t
            tensorrt_llm_bloom.layers[i].attention.kv_quant_orig_scale.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
