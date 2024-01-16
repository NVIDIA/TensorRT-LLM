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
from typing import Dict, List, Optional, Union

import numpy as np
import torch

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_np
from tensorrt_llm.mapping import Mapping
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
    elif len(v.shape) == 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])
    return None


def parse_config(ini_file):
    gpt_config = configparser.ConfigParser()
    gpt_config.read(ini_file)

    n_embd = gpt_config.getint('gpt', 'n_embd')
    n_head = gpt_config.getint('gpt', 'n_head')
    n_layer = gpt_config.getint('gpt', 'n_layer')
    n_positions = gpt_config.getint('gpt', 'n_positions')
    vocab_size = gpt_config.getint('gpt', 'vocab_size')
    do_layer_norm_before = gpt_config.getboolean('gpt',
                                                 'do_layer_norm_before',
                                                 fallback=True)
    rotary_pct = gpt_config.getfloat('gpt', 'rotary_pct', fallback=0.0)
    hidden_act = gpt_config.get('gpt', 'activation_function')
    bias = gpt_config.getboolean('gpt', 'bias', fallback=True)
    inter_size = gpt_config.getint('gpt', 'intermediate_size', fallback=None)
    dtype = gpt_config.get('gpt', 'storage_dtype', fallback='float32')

    if inter_size is None:
        inter_size = 4 * n_embd

    multi_query_mode = gpt_config.getboolean('gpt',
                                             'multi_query_mode',
                                             fallback=False)
    prompt_num_tasks = gpt_config.getint('gpt', 'prompt_num_tasks', fallback=0)
    prompt_max_vocab_size = gpt_config.getint('gpt',
                                              'prompt_max_vocab_size',
                                              fallback=0)
    return n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before, hidden_act, rotary_pct, bias, inter_size, multi_query_mode, dtype, prompt_num_tasks, prompt_max_vocab_size


def load_from_bin_gpt_j(tensorrt_llm_gpt_j: GPTJForCausalLM,
                        dir_path,
                        rank=0,
                        tensor_parallel=1,
                        dtype='float32',
                        use_parallel_embedding=False,
                        sharding_dim=0,
                        share_embedding_table=False,
                        scaling_factors=None):
    tensorrt_llm.logger.info('Loading weights from bin...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_gpt_j, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before, hidden_act, rotary_pct, bias, inter_size, multi_query_mode, *_ = parse_config(
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

    #Enable FP8 Gemm
    enable_fp8_qdq = quant_mode.has_fp8_qdq()

    def sq_trick(x):
        return x.view(np.float32) if use_smooth_quant else x

    # Debug
    suffix = gen_suffix(rank, use_smooth_quant, quant_per_channel)
    # The type of weights.
    w_type = np_dtype if not use_smooth_quant else np.int8

    # pe = fromfile(dir_path, 'model.wpe.bin', [n_positions, n_embd])
    # if pe is not None:
    #     tensorrt_llm_gpt_j.embedding.position_embedding.weight.value = (pe)

    vocab_embedding_weight = fromfile(dir_path, 'model.wte.bin',
                                      [vocab_size, n_embd])
    if not use_parallel_embedding:
        tensorrt_llm_gpt_j.vocab_embedding.weight.value = vocab_embedding_weight
    else:
        if sharding_dim == 0:
            if vocab_size % tensor_parallel != 0:
                # padding
                vocab_size_padded = pad_vocab_size(
                    tensorrt_llm_gpt_j.vocab_embedding.num_embeddings,
                    tensor_parallel)
                pad_width = vocab_size_padded - vocab_size
                vocab_embedding_weight = np.pad(vocab_embedding_weight,
                                                ((0, pad_width), (0, 0)),
                                                'constant',
                                                constant_values=0)
        tensorrt_llm_gpt_j.vocab_embedding.weight.value = np.ascontiguousarray(
            split(vocab_embedding_weight,
                  tensor_parallel,
                  rank,
                  dim=sharding_dim))

    if do_layer_norm_before:
        tensorrt_llm_gpt_j.ln_f.bias.value = (fromfile(
            dir_path, 'model.final_layernorm.bias.bin'))
        tensorrt_llm_gpt_j.ln_f.weight.value = (fromfile(
            dir_path, 'model.final_layernorm.weight.bin'))

    # share input embedding
    if not share_embedding_table:
        lm_head_weight = fromfile(dir_path, 'model.lm_head.weight.bin',
                                  [vocab_size, n_embd])
        lm_head_bias = fromfile(dir_path, 'model.lm_head.bias.bin',
                                [vocab_size])
        if lm_head_weight is None:
            lm_head_weight = fromfile(dir_path, 'model.wte.bin',
                                      [vocab_size, n_embd])
        if vocab_size % tensor_parallel != 0:
            # padding
            vocab_size_padded = tensorrt_llm_gpt_j.lm_head.out_features * tensor_parallel
            pad_width = vocab_size_padded - vocab_size
            lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                    'constant',
                                    constant_values=0)
        tensorrt_llm_gpt_j.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, tensor_parallel, rank))
        tensorrt_llm_gpt_j.lm_head.bias.value = np.ascontiguousarray(
            split(lm_head_bias, tensor_parallel, rank))
    fake_fp8_sf_dt = np.float32
    for i in range(n_layer):
        c_attn_out_dim = (3 * n_embd //
                          tensor_parallel) if not multi_query_mode else (
                              n_embd // tensor_parallel +
                              (n_embd // n_head) * 2)
        tensorrt_llm_gpt_j.layers[i].input_layernorm.weight.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.weight.bin'))
        tensorrt_llm_gpt_j.layers[i].input_layernorm.bias.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.bias.bin'))
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.weight.' + suffix,
            [n_embd, c_attn_out_dim], w_type)
        if t is not None:
            dst = tensorrt_llm_gpt_j.layers[i].attention.qkv.weight
            if use_smooth_quant:
                dst.value = sq_trick(
                    np.ascontiguousarray(np.transpose(t, [1, 0])))
                set_smoothquant_scale_factors(
                    tensorrt_llm_gpt_j.layers[i].attention.qkv,
                    tensorrt_llm_gpt_j.layers[i].input_layernorm.scale_to_int,
                    dir_path,
                    'model.layers.' + str(i) + '.attention.query_key_value.',
                    [1, c_attn_out_dim],
                    quant_per_token_dyn,
                    quant_per_channel,
                    rank=rank,
                    is_qkv=True)
            elif use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(t), plugin_weight_only_quant_type)
                dst.value = processed_torch_weights.numpy()
                scales = tensorrt_llm_gpt_j.layers[
                    i].attention.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        if enable_fp8_qdq:
            tensorrt_llm_gpt_j.layers[
                i].attention.qkv.activation_scaling_factor.value = np.array(
                    [scaling_factors['qkv_act'][i]], dtype=fake_fp8_sf_dt)
            tensorrt_llm_gpt_j.layers[
                i].attention.qkv.weights_scaling_factor.value = np.array(
                    [scaling_factors['qkv_weights'][i]], dtype=fake_fp8_sf_dt)
            tensorrt_llm_gpt_j.layers[
                i].attention.kv_cache_scaling_factor.value = np.array(
                    [scaling_factors['qkv_output'][i]], dtype=np.float32)

        dst = tensorrt_llm_gpt_j.layers[i].attention.dense.weight
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attention.dense.weight.' + suffix,
            [n_embd // tensor_parallel, n_embd], w_type)
        if use_smooth_quant:
            dst.value = sq_trick(np.ascontiguousarray(np.transpose(t, [1, 0])))
            dense_scale = getattr(tensorrt_llm_gpt_j.layers[i].attention,
                                  "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_gpt_j.layers[i].attention.dense, dense_scale,
                dir_path, 'model.layers.' + str(i) + '.attention.dense.',
                [1, n_embd], quant_per_token_dyn, quant_per_channel)
            # change it to the real smoother if dense layer is applied smooth quant
            tensorrt_llm_gpt_j.layers[
                i].attention.dense.smoother.value = np.ones(
                    [1, n_embd // tensor_parallel], dtype=np.float32)
        elif use_weight_only:
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_gpt_j.layers[
                i].attention.dense.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        if enable_fp8_qdq:
            tensorrt_llm_gpt_j.layers[
                i].attention.dense.activation_scaling_factor.value = np.array(
                    [scaling_factors['dense_act'][i]], dtype=fake_fp8_sf_dt)
            tensorrt_llm_gpt_j.layers[
                i].attention.dense.weights_scaling_factor.value = np.array(
                    [scaling_factors['dense_weights'][i]], dtype=fake_fp8_sf_dt)

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.dense_h_to_4h.weight.' + suffix,
            [n_embd, inter_size // tensor_parallel], w_type)
        if use_smooth_quant:
            tensorrt_llm_gpt_j.layers[i].mlp.fc.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            set_smoothquant_scale_factors(
                tensorrt_llm_gpt_j.layers[i].mlp.fc,
                tensorrt_llm_gpt_j.layers[i].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.dense_h_to_4h.',
                [1, inter_size // tensor_parallel],
                quant_per_token_dyn,
                quant_per_channel,
                rank=rank)
        elif use_weight_only:
            dst = tensorrt_llm_gpt_j.layers[i].mlp.fc.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_gpt_j.layers[i].mlp.fc.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_gpt_j.layers[
                i].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
        if bias:
            tensorrt_llm_gpt_j.layers[i].mlp.fc.bias.value = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.mlp.dense_h_to_4h.bias.' + str(rank) + '.bin')
        if enable_fp8_qdq:
            tensorrt_llm_gpt_j.layers[
                i].mlp.fc.activation_scaling_factor.value = np.array(
                    [scaling_factors['fc_act'][i]], dtype=fake_fp8_sf_dt)
            tensorrt_llm_gpt_j.layers[
                i].mlp.fc.weights_scaling_factor.value = np.array(
                    [scaling_factors['fc_weights'][i]], dtype=fake_fp8_sf_dt)

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.dense_4h_to_h.weight.' + suffix,
            [inter_size // tensor_parallel, n_embd], w_type)
        if use_smooth_quant:
            tensorrt_llm_gpt_j.layers[i].mlp.proj.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            proj_scale = getattr(tensorrt_llm_gpt_j.layers[i].mlp,
                                 "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_gpt_j.layers[i].mlp.proj, proj_scale, dir_path,
                'model.layers.' + str(i) + '.mlp.dense_4h_to_h.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
            # change it to the real smoother if proj layer is applied smooth quant
            tensorrt_llm_gpt_j.layers[i].mlp.proj.smoother.value = np.ones(
                [1, inter_size // tensor_parallel], dtype=np.float32)
        elif use_weight_only:
            dst = tensorrt_llm_gpt_j.layers[i].mlp.proj.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_gpt_j.layers[i].mlp.proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_gpt_j.layers[i].mlp.proj.weight.value = (
                np.ascontiguousarray(np.transpose(t, [1, 0])))
        if bias:
            tensorrt_llm_gpt_j.layers[i].mlp.proj.bias.value = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.mlp.dense_4h_to_h.bias.bin')

        if use_int8_kv_cache:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            tensorrt_llm_gpt_j.layers[
                i].attention.kv_cache_scaling_factor.value = t

        if enable_fp8_qdq:
            tensorrt_llm_gpt_j.layers[
                i].mlp.proj.activation_scaling_factor.value = np.array(
                    [scaling_factors['proj_act'][i]], dtype=fake_fp8_sf_dt)
            tensorrt_llm_gpt_j.layers[
                i].mlp.proj.weights_scaling_factor.value = np.array(
                    [scaling_factors['proj_weights'][i]], dtype=fake_fp8_sf_dt)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


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
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2

    # Do we use INT4/INT8 weight-only?
    use_weight_only = quant_mode.is_weight_only()

    tensorrt_llm.logger.info('Loading weights from HF GPT-J...')
    tik = time.time()

    torch_dtype = torch.float16 if fp16 else torch.float32
    hf_gpt_j_state_dict = hf_gpt_j.state_dict()

    v = hf_gpt_j_state_dict.get('transformer.wte.weight')
    tensorrt_llm_gpt_j.vocab_embedding.weight.value = v.to(
        torch_dtype).cpu().numpy()

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
            if use_weight_only and (idx == 2 or idx == 4):
                processed_torch_weights, torch_weight_scales = \
                    torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        v.transpose(0, 1).contiguous(), plugin_weight_only_quant_type
                    )
                layer.value = processed_torch_weights.numpy()
                if idx == 2:
                    scales = tensorrt_llm_gpt_j.layers[
                        layer_idx].mlp.fc.per_channel_scale
                elif idx == 4:
                    scales = tensorrt_llm_gpt_j.layers[
                        layer_idx].mlp.proj.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                setattr(layer, 'value', v.to(torch_dtype).cpu().numpy())

        # Attention QKV Linear
        # concatenate the Q, K, V layers weights.
        q_weights = hf_gpt_j_state_dict.get(prefix + "attn.q_proj.weight")
        k_weights = hf_gpt_j_state_dict.get(prefix + "attn.k_proj.weight")
        v_weights = hf_gpt_j_state_dict.get(prefix + "attn.v_proj.weight")
        qkv_weights = torch.cat((q_weights, k_weights, v_weights))
        layer = attrgetter("attention.qkv.weight")(
            tensorrt_llm_gpt_j.layers[layer_idx])
        if use_weight_only:
            processed_torch_weights, torch_weight_scales = \
                torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                qkv_weights.transpose(0, 1).contiguous(), plugin_weight_only_quant_type)
            layer.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_gpt_j.layers[
                layer_idx].attention.qkv.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
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
                    layer_idx].attention.kv_cache_scaling_factor.value = np.array(
                        [scaling_factors['qkv_output'][layer_idx]],
                        dtype=np.float32)

        # Attention Dense (out_proj) Linear
        v = hf_gpt_j_state_dict.get(prefix + "attn.out_proj.weight")
        layer = attrgetter("attention.dense.weight")(
            tensorrt_llm_gpt_j.layers[layer_idx])
        if use_weight_only:
            processed_torch_weights, torch_weight_scales = \
                torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                v.transpose(0, 1).contiguous(), plugin_weight_only_quant_type)
            layer.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_gpt_j.layers[
                layer_idx].attention.dense.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
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


def load_from_awq_gpt_j(tensorrt_llm_gpt_j: GPTJForCausalLM,
                        quant_ckpt_path: str,
                        quantize_lm_head=False,
                        mapping=Mapping(),
                        fp16=False,
                        ft_model_dir=None):

    awq_gptj_block_names = [
        "input_layernorm:weight",
        "input_layernorm:bias",
        "mlp:fc:bias",
        "mlp:proj:bias",
    ]

    tensorrt_llm_model_gptj_block_names = [
        "input_layernorm.weight",
        "input_layernorm.bias",
        "mlp.fc.bias",
        "mlp.proj.bias",
    ]

    awq_gpt_j = np.load(quant_ckpt_path)
    awq_prefix = "_np:"
    AMMO_WEIGHT_SCALING_FACTOR_COEFF = 7

    def load(key):
        v = torch.from_numpy(awq_gpt_j[awq_prefix + key])
        if "weights_scaling_factor" in key:
            v *= AMMO_WEIGHT_SCALING_FACTOR_COEFF  # For AMMO *.npz checkpoints
        return v

    group_size = load("layers:0:attention:dense:weight").numel() // load(
        "layers:0:attention:dense:weights_scaling_factor").numel()

    quant_mode = getattr(tensorrt_llm_gpt_j, 'quant_mode', QuantMode(0))
    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    packer = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm
    torch_dtype = torch.float16 if fp16 else torch.float32

    tensorrt_llm.logger.info('Loading weights from AWQ GPT-J...')
    tik = time.time()

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
        scale = scale.repeat_interleave(group_size, dim=0)
        weight = weight / scale
        qweight_int8 = torch.clamp(torch.round(weight.cuda()).char(), -8, 7)
        int4_weight = packer(qweight_int8.cpu())
        int4_weight = preprocessor(int4_weight, torch.quint4x2)
        return int4_weight.view(torch.float16).cpu().numpy()

    def process_and_assign_weight(mPrefix, mOp, tp_dim=0):
        weight = load(mPrefix + ":weight").T.contiguous()
        [k, n] = weight.shape
        weight = weight.split(weight.shape[tp_dim] // mapping.tp_size,
                              dim=tp_dim)[mapping.tp_rank]
        amax = load(mPrefix + ":weights_scaling_factor").reshape(
            (n, int(k / group_size))).T.contiguous()
        amax = amax.split(amax.shape[tp_dim] // mapping.tp_size,
                          dim=tp_dim)[mapping.tp_rank]
        pre_quant_scale = load(mPrefix + ":prequant_scaling_factor").reshape(
            (1, k))
        if tp_dim == 0:
            pre_quant_scale = pre_quant_scale.split(k // mapping.tp_size,
                                                    dim=1)[mapping.tp_rank]
        scale = amax / 8.0
        mOp.weight.value = AWQ_quantize_pack_preprocess(weight, scale)
        mOp.weights_scaling_factor.value = scale.to(torch_dtype).cpu().numpy()
        mOp.prequant_scaling_factor.value = pre_quant_scale.to(
            torch_dtype).cpu().numpy()

    def deSmooth(weight, pre_quant_scale):
        [k, n] = weight.shape
        pre_quant_scale = pre_quant_scale.repeat(
            (n, 1)).transpose(1, 0).contiguous()
        weight = weight * pre_quant_scale
        return weight

    def reSmooth(weight, pre_quant_scale):
        [k, n] = weight.shape
        pre_quant_scale = pre_quant_scale.repeat(
            (n, 1)).transpose(1, 0).contiguous()
        weight = weight / pre_quant_scale
        return weight

    def get_scale(weight):
        weight = weight.T.contiguous()
        [n, k] = weight.shape
        weight = weight.reshape(n, int(k / group_size), group_size)
        weight = torch.abs(weight.reshape(-1, group_size))
        amax, idx = weight.max(1)
        amax = amax.reshape(n, int(k / group_size)).T.contiguous()
        return amax / 8

    def reSmooth_and_get_scale(weight, pre_quant_scale, avg_pre_quant_scale):
        if quant_ckpt_path.endswith("pt"):
            # NPZ files are already re-smoothed
            weight = deSmooth(weight, pre_quant_scale)
            weight = reSmooth(weight, avg_pre_quant_scale)
        scale = get_scale(weight)
        return weight, scale

    def process_and_assign_qkv_weight(prefix, mOp):
        q_weight = load(prefix + "attention:qkv:q:weight").T.contiguous()
        k_weight = load(prefix + "attention:qkv:k:weight").T.contiguous()
        v_weight = load(prefix + "attention:qkv:v:weight").T.contiguous()
        k = q_weight.shape[0]

        q_weight = q_weight.split(q_weight.shape[1] // mapping.tp_size,
                                  dim=1)[mapping.tp_rank]
        k_weight = k_weight.split(k_weight.shape[1] // mapping.tp_size,
                                  dim=1)[mapping.tp_rank]
        v_weight = v_weight.split(v_weight.shape[1] // mapping.tp_size,
                                  dim=1)[mapping.tp_rank]

        q_pre_quant_scale = load(
            prefix + "attention:qkv:q:prequant_scaling_factor").reshape((1, k))
        k_pre_quant_scale = load(
            prefix + "attention:qkv:k:prequant_scaling_factor").reshape((1, k))
        v_pre_quant_scale = load(
            prefix + "attention:qkv:v:prequant_scaling_factor").reshape((1, k))

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

    # check if we need to pad vocab
    v = load('vocab_embedding:weight')
    [vocab_size, k] = v.shape
    pad_vocab = False
    pad_vocab_size = vocab_size
    if quantize_lm_head and vocab_size % 64 != 0:
        pad_vocab = True
        pad_vocab_size = int((vocab_size + 63) / 64) * 64
    if pad_vocab:
        new_v = torch.zeros([pad_vocab_size, k])
        new_v[:vocab_size, :] = v
        v = new_v
    if mapping.is_first_pp_rank():
        tensorrt_llm_gpt_j.embedding.weight.value = v.to(
            torch_dtype).cpu().numpy()

    n_layer = len(tensorrt_llm_gpt_j.layers)

    for layer_idx in range(n_layer):
        prefix = "layers:" + str(layer_idx) + ":"
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        for idx, awq_attr in enumerate(awq_gptj_block_names):
            v = load(prefix + awq_attr)
            if awq_attr == "mlp:fc:bias":
                v = v.split(v.shape[0] // mapping.tp_size, dim=0)[mapping.rank]
            elif awq_attr == "mlp:proj:bias":
                v = torch.zeros_like(v) if mapping.rank != 0 else v
            layer = attrgetter(tensorrt_llm_model_gptj_block_names[idx])(
                tensorrt_llm_gpt_j.layers[layer_idx])
            setattr(layer, 'value', v.to(torch_dtype).cpu().numpy())

        # Attention QKV Linear
        # concatenate the Q, K, V layers weights.
        process_and_assign_qkv_weight(
            prefix, tensorrt_llm_gpt_j.layers[layer_idx].attention.qkv)

        # Attention Dense Linear
        mPrefix = prefix + "attention:dense"
        mOp = tensorrt_llm_gpt_j.layers[layer_idx].attention.dense
        process_and_assign_weight(mPrefix, mOp, 0)

        # MLP Dense (mlp.fc) Linear
        mPrefix = prefix + "mlp:fc"
        mOp = tensorrt_llm_gpt_j.layers[layer_idx].mlp.fc
        process_and_assign_weight(mPrefix, mOp, 1)

        # MLP Dense (mlp.proj) Linear
        mPrefix = prefix + "mlp:proj"
        mOp = tensorrt_llm_gpt_j.layers[layer_idx].mlp.proj
        process_and_assign_weight(mPrefix, mOp, 0)

        if use_int8_kv_cache:
            assert ft_model_dir, "You must pass --ft_model_dir to tell TRT-LLM where to look for scales of INT8 kv cache."
            t = fromfile(
                ft_model_dir, 'model.layers.' + str(layer_idx) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            assert t is not None, f"{ft_model_dir} does not contain model.layers.{layer_idx}.attention.query_key_value.scale_y_quant_orig.bin"
            tensorrt_llm_gpt_j.layers[
                layer_idx].attention.kv_cache_scaling_factor.value = t

    v = load('final_layernorm:weight')
    tensorrt_llm_gpt_j.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()

    v = load('final_layernorm:bias')
    tensorrt_llm_gpt_j.ln_f.bias.value = v.to(torch_dtype).cpu().numpy()

    # lm_head
    if pad_vocab:
        weight = load('lm_head:weight')
        [vocab_size, k] = weight.shape
        new_weight = torch.zeros([pad_vocab_size, k])
        new_weight[:vocab_size, :] = weight
        new_weight = new_weight.T.contiguous()
        new_weight = new_weight.split(new_weight.shape[1] // mapping.tp_size,
                                      dim=1)[mapping.tp_rank]
        amax = load('lm_head:weights_scaling_factor').reshape(
            [vocab_size, int(k / group_size)])
        new_amax = torch.ones([pad_vocab_size, int(k / group_size)])
        new_amax[:vocab_size, :] = amax
        new_amax = new_amax.T.contiguous()
        new_amax = new_amax.split(new_amax.shape[1] // mapping.tp_size,
                                  dim=1)[mapping.tp_rank]
        new_scale = new_amax / 8
        tensorrt_llm_gpt_j.lm_head.weight.value = AWQ_quantize_pack_preprocess(
            new_weight, new_scale)
        tensorrt_llm_gpt_j.lm_head.weights_scaling_factor.value = new_scale.to(
            torch_dtype).cpu().numpy()
        tensorrt_llm_gpt_j.lm_head.prequant_scaling_factor.value = load(
            'lm_head:prequant_scaling_factor').to(torch_dtype).cpu().numpy()

        bias = load('lm_head:bias')
        new_bias = torch.zeros([pad_vocab_size])
        new_bias[:vocab_size] = bias
        new_bias = new_bias.split(pad_vocab_size // mapping.tp_size,
                                  dim=0)[mapping.tp_rank]
        tensorrt_llm_gpt_j.lm_head.bias.value = new_bias.to(
            torch_dtype).cpu().numpy()
    elif quantize_lm_head:
        mPrefix = "lm_head"
        mOp = tensorrt_llm_gpt_j.lm_head
        process_and_assign_weight(mPrefix, mOp, 1)
        v = load('lm_head:bias')
        tensorrt_llm_gpt_j.lm_head.bias.value = torch_split(
            v, 0).to(torch_dtype).cpu().numpy()
    else:
        weight = load('lm_head:weight')
        tensorrt_llm_gpt_j.lm_head.weight.value = torch_split(
            weight, 0).to(torch_dtype).cpu().numpy()
        bias = load('lm_head:bias')
        tensorrt_llm_gpt_j.lm_head.bias.value = torch_split(
            bias, 0).to(torch_dtype).cpu().numpy()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
