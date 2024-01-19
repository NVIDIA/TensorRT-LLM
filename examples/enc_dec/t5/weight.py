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
from os import path
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from tensorrt_llm import logger
from tensorrt_llm._utils import (numpy_to_dtype, str_dtype_to_np,
                                 str_dtype_to_torch, torch_to_numpy)
from tensorrt_llm.functional import (LayerNormPositionType, LayerNormType,
                                     MLPType)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import (  # TODO: probably need to change model name to distinguish from other models
    DecoderModel, EncoderModel)

layernorm_type_map = {i.name: i.value for i in LayerNormType}
layernorm_position_map = {i.name: i.value for i in LayerNormPositionType}
mlp_type_map = {i.name: i.value for i in MLPType}


def parse_t5_config(config, component, args):
    if component == 'encoder':
        args.n_layer = config.getint(component, 'num_layers')
        args.n_head = config.getint(component, 'num_heads')
        args.head_size = config.getint(component, 'd_kv')
        args.hidden_size = config.getint(component, 'd_model')
        args.ffn_hidden_size = config.getint(component, 'd_ff')
        args.vocab_size = config.getint(component, 'vocab_size')
        args.n_positions = config.getint(component, 'n_positions', fallback=512)
        args.has_position_embedding = config.getboolean(
            component, 'has_position_embedding',
            fallback=False)  # TODO: hardcoded here
        args.has_token_type_embedding = config.getboolean(
            component, 'has_token_type_embedding', fallback=False)
        args.has_embedding_layernorm = config.getboolean(
            component, 'has_embedding_layernorm', fallback=False)
        args.has_embedding_scale = config.getboolean(component,
                                                     'has_embedding_scale',
                                                     fallback=False)
        args.q_scaling = config.getfloat(component, 'q_scaling', fallback=1.0)
        args.has_attention_qkvo_bias = config.getboolean(
            component, 'has_attention_qkvo_bias',
            fallback=False)  # TODO: hardcoded here
        args.has_mlp_bias = config.getboolean(component,
                                              'has_mlp_bias',
                                              fallback=False)
        args.has_model_final_layernorm = config.getboolean(
            component, 'has_model_final_layernorm', fallback=True)
        args.layernorm_eps = config.getfloat(component, 'layer_norm_epsilon')
        args.layernorm_position = layernorm_position_map[config.get(
            component, 'layernorm_position',
            fallback='pre_layernorm')]  # TODO: hardcoded here
        args.layernorm_type = layernorm_type_map[config.get(
            component, 'layernorm_type',
            fallback='RmsNorm')]  # TODO: hardcoded here
        args.hidden_act = config.get(component, 'dense_act_fn')
        args.gated_act = config.getboolean(component, 'is_gated_act')
        args.mlp_type = mlp_type_map['GatedMLP' if args.gated_act else 'MLP']
        args.relative_attention = config.getboolean(component,
                                                    'relative_attention',
                                                    fallback=True)
        args.num_buckets = config.getint(component,
                                         'relative_attention_num_buckets')
        args.max_distance = config.getint(component,
                                          'relative_attention_max_distance')
        args.ckpt_weight_dtype = config.get(component, 'weight_data_type')

    elif component == 'decoder':
        args.n_layer = config.getint(component, 'num_decoder_layers')
        args.n_head = config.getint(component, 'num_heads')
        args.head_size = config.getint(component, 'd_kv')
        args.hidden_size = config.getint(component, 'd_model')
        args.ffn_hidden_size = config.getint(component, 'd_ff')
        args.vocab_size = config.getint(component, 'vocab_size')
        args.n_positions = config.getint(component, 'n_positions', fallback=512)
        args.has_position_embedding = config.getboolean(
            component, 'has_position_embedding',
            fallback=False)  # TODO: hardcoded here
        args.has_token_type_embedding = config.getboolean(
            component, 'has_token_type_embedding', fallback=False)
        args.has_embedding_layernorm = config.getboolean(
            component, 'has_embedding_layernorm', fallback=False)
        args.has_embedding_scale = config.getboolean(component,
                                                     'has_embedding_scale',
                                                     fallback=False)
        args.q_scaling = config.getfloat(component, 'q_scaling', fallback=1.0)
        args.has_attention_qkvo_bias = config.getboolean(
            component, 'has_attention_qkvo_bias', fallback=False)
        args.has_mlp_bias = config.getboolean(component,
                                              'has_mlp_bias',
                                              fallback=False)
        args.has_model_final_layernorm = config.getboolean(
            component, 'has_model_final_layernorm', fallback=True)
        args.layernorm_eps = config.getfloat(component, 'layer_norm_epsilon')
        args.layernorm_position = layernorm_position_map[config.get(
            component, 'layernorm_position',
            fallback='pre_layernorm')]  # TODO: hardcoded here
        args.layernorm_type = layernorm_type_map[config.get(component,
                                                            'layernorm_type',
                                                            fallback='RmsNorm')]
        args.hidden_act = config.get(component, 'dense_act_fn')
        args.gated_act = config.getboolean(component, 'is_gated_act')
        args.mlp_type = mlp_type_map['GatedMLP' if args.gated_act else 'MLP']
        args.has_lm_head_bias = config.getboolean(
            component,  # TODO: T5 with bias
            'has_lm_head_bias',
            fallback=False)
        args.relative_attention = config.getboolean(component,
                                                    'relative_attention',
                                                    fallback=True)
        args.num_buckets = config.getint(component,
                                         'relative_attention_num_buckets')
        args.max_distance = config.getint(component,
                                          'relative_attention_max_distance')
        args.logits_dtype = config.get(component,
                                       'logits_dtype',
                                       fallback='float32')
        args.rescale_before_lm_head = config.getboolean(
            component, 'tie_word_embeddings'
        )  # default is True (for T5), but False for Flan-T5
        args.encoder_hidden_size = config.getint('encoder', 'd_model')
        args.encoder_num_heads = config.getint('encoder', 'num_heads')
        args.encoder_head_size = config.getint('encoder', 'd_kv')
        args.ckpt_weight_dtype = config.get(component, 'weight_data_type')

    else:
        assert False, 'Unsupported component!'

    return args


def fuse_qkv(q, k, v):
    qkv_weight = np.concatenate((q, k, v))
    return qkv_weight


def load_from_hf_t5(tllm_model, pytorch_ckpt_path, component, dtype="float32"):
    torch_dtype = str_dtype_to_torch(dtype)

    pytorch_ckpt = torch.load(path.join(pytorch_ckpt_path, 'pytorch_model.bin'))
    pytorch_model = {
        key: torch_to_numpy(value.to(torch_dtype))
        for key, value in pytorch_ckpt.items()
    }

    if component == "encoder":
        tllm_model.embedding.vocab_embedding.weight.value = np.ascontiguousarray(
            pytorch_model['shared.weight'])
        if tllm_model.embedding.position_embedding:
            tllm_model.embedding.position_embedding.weight.value = pytorch_model[
                'encoder.embed_positions.weight']
        if tllm_model.embedding.token_type_embedding:
            tllm_model.embedding.token_type_embedding.weight.value = pytorch_model[
                'encoder.embed_token_type.weight']

        # all layers use 1st layer's attn table
        # transpose from [num_buckets, num_heads] -> [num_heads, num_buckets]
        # ascontiguousarray is very important! otherwise TRT always receives the original layout
        relative_attention_table = np.ascontiguousarray(pytorch_model[
            f'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
                                                        .T)

        for i in range(tllm_model.num_layers):
            layer = tllm_model.encoder_layers[i]
            layer_prefix = f'encoder.block.{i}.'

            # attention table for all layers
            layer.attention.rel_attn_table.value = relative_attention_table

            layer.attention.qkv.weight.value = fuse_qkv(
                pytorch_model[f'{layer_prefix}layer.0.SelfAttention.q.weight'],
                pytorch_model[f'{layer_prefix}layer.0.SelfAttention.k.weight'],
                pytorch_model[f'{layer_prefix}layer.0.SelfAttention.v.weight'])
            layer.attention.dense.weight.value = pytorch_model[
                f'{layer_prefix}layer.0.SelfAttention.o.weight']

            if tllm_model.has_attention_qkvo_bias:
                layer.attention.qkv.bias.value = fuse_qkv(
                    pytorch_model[
                        f'{layer_prefix}layer.0.SelfAttention.q.bias'],
                    pytorch_model[
                        f'{layer_prefix}layer.0.SelfAttention.k.bias'],
                    pytorch_model[f'{layer_prefix}layer.0.SelfAttention.v.bias']
                )
                layer.attention.dense.bias.value = pytorch_model[
                    f'{layer_prefix}layer.0.SelfAttention.o.bias']

            layer.attention_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}layer.0.layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.attention_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}layer.0.layer_norm.bias']

            layer.mlp.fc.weight.value = pytorch_model[
                f'{layer_prefix}layer.1.DenseReluDense.wi.weight']
            layer.mlp.proj.weight.value = pytorch_model[
                f'{layer_prefix}layer.1.DenseReluDense.wo.weight']

            if tllm_model.has_mlp_bias:
                layer.mlp.fc.bias.value = pytorch_model[
                    f'{layer_prefix}layer.1.DenseReluDense.wi.bias']
                layer.mlp.proj.bias.value = pytorch_model[
                    f'{layer_prefix}layer.1.DenseReluDense.wo.bias']

            layer.mlp_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}layer.1.layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.mlp_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}layer.1.layer_norm.bias']

        if tllm_model.final_layernorm:
            tllm_model.final_layernorm.weight.value = pytorch_model[
                'encoder.final_layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                tllm_model.final_layernorm.bias.value = pytorch_model[
                    'encoder.final_layer_norm.bias']

    if component == "decoder":
        tllm_model.embedding.vocab_embedding.weight.value = pytorch_model[
            'shared.weight']
        if tllm_model.embedding.position_embedding:
            tllm_model.embedding.position_embedding.weight.value = pytorch_model[
                'decoder.embed_positions.weight']
        if tllm_model.embedding.token_type_embedding:
            tllm_model.embedding.token_type_embedding.weight.value = pytorch_model[
                'decoder.embed_token_type.weight']

        # all layers use 1st layer's attn table
        # transpose from [num_buckets, num_heads] --> [num_heads, num_buckets]
        # ascontiguousarray is very important! otherwise TRT always receives the original layout
        relative_attention_table = np.ascontiguousarray(pytorch_model[
            f'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
                                                        .T)

        for i in range(tllm_model.num_layers):
            layer = tllm_model.decoder_layers[i]
            layer_prefix = f'decoder.block.{i}.'

            # attention table for all layers
            layer.self_attention.rel_attn_table.value = relative_attention_table

            # self attn
            layer.self_attention.qkv.weight.value = fuse_qkv(
                pytorch_model[f'{layer_prefix}layer.0.SelfAttention.q.weight'],
                pytorch_model[f'{layer_prefix}layer.0.SelfAttention.k.weight'],
                pytorch_model[f'{layer_prefix}layer.0.SelfAttention.v.weight'])
            layer.self_attention.dense.weight.value = pytorch_model[
                f'{layer_prefix}layer.0.SelfAttention.o.weight']

            if tllm_model.has_attention_qkvo_bias:
                layer.self_attention.qkv.bias.value = fuse_qkv(
                    pytorch_model[
                        f'{layer_prefix}layer.0.SelfAttention.q.bias'],
                    pytorch_model[
                        f'{layer_prefix}layer.0.SelfAttention.k.bias'],
                    pytorch_model[f'{layer_prefix}layer.0.SelfAttention.v.bias']
                )
                layer.self_attention.dense.bias.value = pytorch_model[
                    f'{layer_prefix}layer.0.SelfAttention.o.bias']

            layer.self_attention_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}layer.0.layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.self_attention_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}layer.0.layer_norm.bias']

            # cross attn
            layer.cross_attention.qkv.weight.value = fuse_qkv(
                pytorch_model[
                    f'{layer_prefix}layer.1.EncDecAttention.q.weight'],
                pytorch_model[
                    f'{layer_prefix}layer.1.EncDecAttention.k.weight'],
                pytorch_model[f'{layer_prefix}layer.1.EncDecAttention.v.weight']
            )
            layer.cross_attention.dense.weight.value = pytorch_model[
                f'{layer_prefix}layer.1.EncDecAttention.o.weight']

            if tllm_model.has_attention_qkvo_bias:
                layer.cross_attention.qkv.bias.value = fuse_qkv(
                    pytorch_model[
                        f'{layer_prefix}layer.1.EncDecAttention.q.bias'],
                    pytorch_model[
                        f'{layer_prefix}layer.1.EncDecAttention.k.bias'],
                    pytorch_model[
                        f'{layer_prefix}layer.1.EncDecAttention.v.bias'])
                layer.cross_attention.dense.bias.value = pytorch_model[
                    f'{layer_prefix}layer.1.EncDecAttention.o.bias']

            layer.cross_attention_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}layer.1.layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.cross_attention_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}layer.1.layer_norm.bias']

            layer.mlp.fc.weight.value = pytorch_model[
                f'{layer_prefix}layer.2.DenseReluDense.wi.weight']
            layer.mlp.proj.weight.value = pytorch_model[
                f'{layer_prefix}layer.2.DenseReluDense.wo.weight']

            if tllm_model.has_mlp_bias:
                layer.mlp.fc.bias.value = pytorch_model[
                    f'{layer_prefix}layer.2.DenseReluDense.wi.bias']
                layer.mlp.proj.bias.value = pytorch_model[
                    f'{layer_prefix}layer.2.DenseReluDense.wo.bias']

            layer.mlp_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}layer.2.layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.mlp_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}layer.2.layer_norm.bias']

        if tllm_model.final_layernorm:
            tllm_model.final_layernorm.weight.value = pytorch_model[
                'decoder.final_layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                tllm_model.final_layernorm.bias.value = pytorch_model[
                    'decoder.final_layer_norm.bias']

        tllm_model.lm_head.weight.value = pytorch_model['lm_head.weight']


# TODO: only support t5, biases are not loaded
def load_from_binary_t5(tllm_model: Union[EncoderModel, DecoderModel],
                        dir_path,
                        args,
                        mapping: Optional[Mapping] = None,
                        dtype='float32',
                        use_parallel_embedding=False,
                        sharding_dim=0,
                        share_embedding_table=False,
                        scaling_factors=None):
    logger.info('Loading weights from binary...')
    tik = time.time()

    if mapping is None:
        mapping = Mapping()

    ckpt_np_dtype = str_dtype_to_np(args.ckpt_weight_dtype)

    def fromfile(name, split=True, shape=None) -> Optional[np.ndarray]:
        p = path.join(
            dir_path,
            f'{name}.{str(mapping.tp_rank)}.bin' if split else f'{name}.bin')
        if Path(p).exists():
            # load from original dtype and cast to inference dtype
            t = np.fromfile(p, dtype=ckpt_np_dtype)
            t = numpy_to_dtype(t, dtype)
            if shape is not None:
                t = t.reshape(shape)
            t = np.ascontiguousarray(t)
            return t
        return None

    component = 'encoder' if isinstance(tllm_model, EncoderModel) else 'decoder'

    if mapping.is_first_pp_rank():
        wte = fromfile('shared.weight',
                       shape=[args.vocab_size, -1],
                       split=False)
        tllm_model.embedding.vocab_embedding.weight.value = wte

    # T5 special: all layers use 1st layer's attn table
    relative_attention_table = fromfile(
        f'{component}.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
        shape=[args.n_head // mapping.tp_size, args.num_buckets])

    # TP is by loading different split weights. PP is by loading different layer weights
    # TODO: fix llama's wrong def of .num_layers field in PP. enc_dec is the correct way
    layers_range = list(
        range(mapping.pp_rank * tllm_model.num_layers,
              (mapping.pp_rank + 1) * tllm_model.num_layers, 1))

    for layer_idx in layers_range:
        pp_offset_layer_idx = layer_idx - mapping.pp_rank * tllm_model.num_layers
        layer = getattr(tllm_model, f'{component}_layers')[pp_offset_layer_idx]
        layer_prefix = f'{component}.block.{layer_idx}'

        self_attention_layer = getattr(
            layer, 'attention' if component == 'encoder' else 'self_attention')

        # attention table for all layers
        self_attention_layer.rel_attn_table.value = relative_attention_table

        # self attention
        attention_hidden_size = args.n_head * args.head_size  # head size * num_heads not necessarily equals hidden_dim, such as Flan-T5
        self_attention_layer.qkv.weight.value = fromfile(
            f'{layer_prefix}.layer.0.SelfAttention.qkv.weight',
            shape=[
                3 * attention_hidden_size // mapping.tp_size, args.hidden_size
            ])
        self_attention_layer.dense.weight.value = fromfile(
            f'{layer_prefix}.layer.0.SelfAttention.o.weight',
            shape=[args.hidden_size, attention_hidden_size // mapping.tp_size])
        self_attention_layernorm = getattr(
            layer, 'self_attention_layernorm'
            if component == 'decoder' else 'attention_layernorm')
        self_attention_layernorm.weight.value = fromfile(
            f'{layer_prefix}.layer.0.layer_norm.weight', split=False)

        # cross attention
        if component == 'decoder':
            attention_hidden_size = args.n_head * args.head_size
            layer.cross_attention.qkv.weight.value = fromfile(
                f'{layer_prefix}.layer.1.EncDecAttention.qkv.weight',
                shape=[
                    3 * attention_hidden_size // mapping.tp_size,
                    args.hidden_size
                ])
            layer.cross_attention.dense.weight.value = fromfile(
                f'{layer_prefix}.layer.1.EncDecAttention.o.weight',
                shape=[
                    args.hidden_size, attention_hidden_size // mapping.tp_size
                ])
            layer.cross_attention_layernorm.weight.value = fromfile(
                f'{layer_prefix}.layer.1.layer_norm.weight', split=False)

        # MLP
        hf_component_idx = 1 if component == 'encoder' else 2
        layer.mlp.fc.weight.value = fromfile(
            f'{layer_prefix}.layer.{hf_component_idx}.DenseReluDense.wi.weight',
            shape=[args.ffn_hidden_size // mapping.tp_size, args.hidden_size])
        if args.gated_act:
            layer.mlp.gate.weight.value = fromfile(
                f'{layer_prefix}.layer.{hf_component_idx}.DenseReluDense.wi2.weight',
                shape=[
                    args.ffn_hidden_size // mapping.tp_size, args.hidden_size
                ])
        layer.mlp.proj.weight.value = fromfile(
            f'{layer_prefix}.layer.{hf_component_idx}.DenseReluDense.wo.weight',
            shape=[args.hidden_size, args.ffn_hidden_size // mapping.tp_size])
        layer.mlp_layernorm.weight.value = fromfile(
            f'{layer_prefix}.layer.{hf_component_idx}.layer_norm.weight',
            split=False)

    if mapping.is_last_pp_rank():
        if tllm_model.has_model_final_layernorm:
            tllm_model.final_layernorm.weight.value = fromfile(
                f'{component}.final_layer_norm.weight', split=False)

        if component == 'decoder':
            tllm_model.lm_head.weight.value = fromfile(
                'lm_head.weight',
                shape=[args.vocab_size // mapping.tp_size, args.hidden_size])

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')
