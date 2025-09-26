# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import builtins
import collections
import functools
import json
import math
import os
import re
from collections import OrderedDict
from typing import Optional

import numpy as np
import tensorrt as trt
import torch
from tqdm import tqdm

import tensorrt_llm
from tensorrt_llm._common import default_net
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_str
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.functional import (ACT2FN, AttentionMaskType, LayerNormType,
                                     PositionEmbeddingType, Tensor,
                                     constant_to_tensor_)
from tensorrt_llm.layers import (ColumnLinear, Conv3d, LayerNorm, Linear,
                                 RowLinear)
from tensorrt_llm.layers.attention import (Attention, AttentionParams,
                                           BertAttention, KeyValueCacheParams,
                                           bert_attention, layernorm_map)
from tensorrt_llm.layers.normalization import RmsNorm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.models.model_weights_loader import (ModelWeightsFormat,
                                                      ModelWeightsLoader)
from tensorrt_llm.models.modeling_utils import PretrainedConfig, PretrainedModel
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.plugin import current_all_reduce_helper
from tensorrt_llm.quantization import QuantMode

from ...functional import (allgather, arange, cast, chunk, concat, constant,
                           cos, div, einsum, exp, expand, expand_dims,
                           expand_mask, masked_select, matmul, meshgrid2d, pad,
                           permute, pow, rearrange, repeat, repeat_interleave,
                           rms_norm, shape, sin, slice, softmax, split, squeeze,
                           stack, sum, unsqueeze, where)
from .config import STDiTModelConfig

# [TODO] For now, we only support static shape, which might contains `-1` when inputs are with dynamic shape.
USE_STATIC_SHAPE = True


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple([x] * n)

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


# [TODO] make constant `1` compatible with `scale`
def t2i_modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


class ModuleSequential(ModuleList):

    def __init__(self, modules) -> None:
        super(ModuleSequential, self).__init__(modules=modules)

    def forward(self, *args, **kwargs):
        module = self.__getitem__(0)
        outputs = module(*args, **kwargs)
        for idx in range(1, len(self._modules)):
            module = self.__getitem__(idx)
            outputs = module(outputs)
        return outputs


class Activation(Module):

    def __init__(self, act_fn='silu'):
        super().__init__()
        self.act_fn = act_fn

    def forward(self, input: Tensor):
        return ACT2FN[self.act_fn](input)


class RotaryEmbedder(Module):

    def __init__(self,
                 dim,
                 theta=10000,
                 interpolate_factor=1.,
                 theta_rescale_factor=1.,
                 seq_before_head_dim=False,
                 cache_if_possible=True,
                 use_xpos=False,
                 dtype=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        super().__init__()
        theta *= theta_rescale_factor**(dim / (dim - 2))
        freqs = 1. / (theta
                      **(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        self.freqs = Parameter(freqs, dtype=dtype)
        self.cached_freqs = None
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2
        self.scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        self.cache_if_possible = cache_if_possible
        self.use_xpos = use_xpos
        self.dtype = dtype
        self.mapping = mapping
        self.quant_mode = quant_mode

    def get_freqs(self,
                  t: Tensor,
                  seq_len: Optional[int] = None,
                  offset: int = 0):
        should_cache = self.cache_if_possible and seq_len is not None
        if should_cache and isinstance(self.cached_freqs, Tensor):
            if (offset + seq_len) <= self.cached_freqs.shape[0]:
                return slice(self.cached_freqs,
                             starts=[offset] +
                             [0] * len(self.cached_freqs.shape[1:]),
                             sizes=[seq_len, *self.cached_freqs.shape[1:]])
        freqs = self.freqs.value
        freqs = unsqueeze(t, axis=-1) * unsqueeze(freqs, axis=0)
        freqs = repeat_interleave(freqs, repeats=2, dim=(freqs.ndim() - 1))
        if should_cache:
            self.cached_freqs = freqs
        return freqs

    def get_seq_pos(self, seq_len: int, dtype: trt.DataType, offset: int = 0):
        return (arange(start=0, end=seq_len, dtype=trt_dtype_to_str(dtype)) +
                offset) / self.interpolate_factor

    def rotate_half(self, x: Tensor):
        x = x.view([*x.shape[:-1], x.shape[-1] // 2, 2])
        x1, x2 = x.unbind(x.ndim() - 1)
        x = stack([-1 * x2, x1], dim=-1)
        x = x.view([*x.shape[:-2], x.shape[-2] * x.shape[-1]])
        return x

    def apply_rotary_emb(self,
                         freqs: Tensor,
                         t: Tensor,
                         start_index: int = 0,
                         scale: int = 1.,
                         seq_dim: int = -2):
        if t.ndim() == 3:
            seq_len = t.shape[seq_dim]
            # freqs = freqs[-seq_len:]
            freqs = slice(starts=[freqs.shape[0] - seq_len], sizes=[seq_len])
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size ' + \
                                        'to rotate in all the positions {rot_dim}'
        t_left = slice(t,
                       starts=[0] * t.ndim(),
                       sizes=[*t.shape[:-1], start_index])
        t_right = slice(t,
                        starts=[0] * (t.ndim() - 1) + [end_index],
                        sizes=[*t.shape[:-1], t.shape[-1] - end_index])

        t = (t * cos(freqs) * scale) + (self.rotate_half(t) * sin(freqs) *
                                        scale)
        return concat([t_left, t, t_right], dim=-1)

    def rotate_queries_or_keys(self,
                               t: Tensor,
                               seq_dim: Optional[int] = None,
                               offset: int = 0,
                               freq_seq_len: Optional[int] = None):
        seq_dim = self.default_seq_dim if seq_dim is None else seq_dim
        assert not self.use_xpos, 'you must use `.rotate_queries_and_keys` method ' + \
                                  'instead and pass in both queries and keys, for ' + \
                                  'length extrapolatable rotary embeddings'

        seq_len = t.shape[seq_dim]
        if freq_seq_len is not None:
            assert freq_seq_len >= seq_len
            seq_len = freq_seq_len

        freqs = self.get_freqs(self.get_seq_pos(seq_len,
                                                dtype=t.dtype,
                                                offset=offset),
                               seq_len=seq_len,
                               offset=offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
        rope_output = self.apply_rotary_emb(freqs, t, seq_dim=seq_dim)
        return rope_output


class STDiTRmsNorm(RmsNorm):

    def __init__(self,
                 normalized_shape,
                 num_groups=1,
                 eps=1e-06,
                 elementwise_affine=True,
                 dtype=None):
        super().__init__(normalized_shape, num_groups, eps, elementwise_affine,
                         dtype)

    def forward(self, hidden_states):
        weight = None if self.weight is None else self.weight.value
        return rms_norm(input=hidden_states,
                        normalized_shape=self.normalized_shape,
                        num_groups=self.num_groups,
                        weight=weight,
                        eps=self.eps)


class STDiTAttention(BertAttention):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 qk_layernorm=True,
                 layernorm_type=LayerNormType.RmsNorm,
                 layernorm_eps=1e-06,
                 bias=True,
                 rotary_embedding_func=None,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0,
                 cp_group=None,
                 cp_size=1,
                 quant_mode: QuantMode = QuantMode(0)):
        assert hidden_size % num_attention_heads == 0, "hidden_size should be divisible by num_attention_heads"
        super().__init__(hidden_size=hidden_size,
                         num_attention_heads=num_attention_heads,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         tp_rank=tp_rank,
                         cp_group=cp_group,
                         cp_size=cp_size,
                         quant_mode=quant_mode)

        self.qk_layernorm = qk_layernorm
        if self.qk_layernorm:
            ln_type = layernorm_map[layernorm_type]
            self.q_layernorm = ln_type(self.attention_head_size,
                                       eps=layernorm_eps,
                                       dtype=dtype)
            self.k_layernorm = ln_type(self.attention_head_size,
                                       eps=layernorm_eps,
                                       dtype=dtype)
        self.rotary_embedding_func = rotary_embedding_func

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                input_lengths=None,
                max_input_length: int = None):

        assert isinstance(hidden_states, Tensor)

        B = shape(hidden_states, 0)
        N = shape(hidden_states, 1)
        C = shape(hidden_states, 2)
        input_lengths = expand(unsqueeze(N, 0).cast('int32'), unsqueeze(B, 0))

        assert (self.qkv is not None)
        qkv = self.qkv(hidden_states)

        kv_size = self.attention_head_size * self.num_attention_kv_heads
        query, key, value = split(
            qkv, [self.attention_hidden_size, kv_size, kv_size], dim=2)
        query = query.view(
            concat([B, N, self.num_attention_heads,
                    self.attention_head_size])).permute(dims=[0, 2, 1, 3])
        key = key.view(
            concat(
                [B, N, self.num_attention_kv_heads,
                 self.attention_head_size])).permute(dims=[0, 2, 1, 3])

        if self.qk_layernorm:
            query = self.q_layernorm(query)
            key = self.k_layernorm(key)

        if self.rotary_embedding_func is not None:
            query = self.rotary_embedding_func(query)
            key = self.rotary_embedding_func(key)

        # TODO deal with qkv
        query = query.permute(dims=[0, 2, 1, 3]).view(
            concat([B, N, self.attention_hidden_size]))
        key = key.permute(dims=[0, 2, 1, 3]).view(concat([B, N, kv_size]))
        qkv = concat([query, key, value], dim=2)

        if default_net().plugin_config.bert_attention_plugin:
            # TRT plugin mode
            assert input_lengths is not None
            assert self.cp_size == 1
            if default_net().plugin_config.remove_input_padding:
                qkv = qkv.view(
                    concat([-1, self.attention_hidden_size + 2 * kv_size]))
                max_input_length = constant(
                    np.zeros([
                        max_input_length,
                    ], dtype=np.int32))
            context = bert_attention(qkv,
                                     input_lengths,
                                     self.num_attention_heads,
                                     self.attention_head_size,
                                     q_scaling=self.q_scaling,
                                     max_distance=self.max_distance,
                                     max_input_length=max_input_length)
        else:
            # plain TRT mode
            def transpose_for_scores(x):
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), self.num_attention_heads,
                    self.attention_head_size
                ])
                return x.view(new_x_shape).permute([0, 2, 1, 3])

            kv_size = self.attention_head_size * self.num_attention_kv_heads
            query, key, value = split(
                qkv, [self.attention_hidden_size, kv_size, kv_size], dim=2)
            if self.cp_size > 1 and self.cp_group is not None:
                key = allgather(key, self.cp_group, gather_dim=1)
                value = allgather(value, self.cp_group, gather_dim=1)
            query = transpose_for_scores(query)
            key = transpose_for_scores(key)
            value = transpose_for_scores(value)

            key = key.permute([0, 1, 3, 2])
            attention_scores = matmul(query, key, use_fp32_acc=False)
            attention_scores = attention_scores / (self.q_scaling *
                                                   self.norm_factor)

            if attention_mask is not None:
                attention_mask = expand_mask(attention_mask, shape(query, 2))
                attention_mask = cast(attention_mask, attention_scores.dtype)
                attention_scores = attention_scores + attention_mask

            attention_probs = softmax(attention_scores, dim=-1)

            context = matmul(attention_probs, value,
                             use_fp32_acc=False).permute([0, 2, 1, 3])
            context = context.view(
                concat([
                    shape(context, 0),
                    shape(context, 1), self.attention_hidden_size
                ]))

        context = self.dense(context)
        context = context.view(concat([B, N, C]))
        return context


class STDiTCrossAttention(Attention):

    def __init__(self,
                 *,
                 local_layer_idx,
                 hidden_size,
                 num_attention_heads,
                 attention_mask_type=AttentionMaskType.causal,
                 qkv_bias=True,
                 dense_bias=True,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0,
                 cp_group=[0],
                 cp_size=1,
                 cp_rank=0,
                 quant_mode: QuantMode = QuantMode(0)):
        assert hidden_size % num_attention_heads == 0, "hidden_size should be divisible by num_attention_heads"
        super().__init__(local_layer_idx=local_layer_idx,
                         hidden_size=hidden_size,
                         num_attention_heads=num_attention_heads,
                         attention_mask_type=attention_mask_type,
                         bias=qkv_bias,
                         dense_bias=dense_bias,
                         cross_attention=True,
                         position_embedding_type=position_embedding_type,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         tp_rank=tp_rank,
                         cp_group=cp_group,
                         cp_size=cp_size,
                         cp_rank=cp_rank,
                         quant_mode=quant_mode)

    def forward(self,
                hidden_states: Tensor,
                encoder_output: Tensor,
                use_cache=False,
                attention_params: Optional[AttentionParams] = None,
                kv_cache_params: Optional[KeyValueCacheParams] = None):
        bs = shape(encoder_output, 0)
        encoder_input_length = shape(encoder_output, 1)
        encoder_hidden_size = shape(encoder_output, 2)
        encoder_output = encoder_output.view(
            concat([bs * 2, encoder_input_length // 2, encoder_hidden_size]))

        if default_net().plugin_config.remove_input_padding:
            B = shape(hidden_states, 0)
            N = shape(hidden_states, 1)
            C = shape(hidden_states, 2)
            hidden_states = hidden_states.view(concat([B * N, C]))
            encoder_output = encoder_output.view(
                concat([-1, encoder_hidden_size]))

        context = super().forward(hidden_states=hidden_states,
                                  encoder_output=encoder_output,
                                  use_cache=use_cache,
                                  attention_params=attention_params,
                                  kv_cache_params=kv_cache_params)
        context = context.view(concat([B, -1, C]))
        return context


class T2IFinalLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_patch,
                 out_channels,
                 d_t=None,
                 d_s=None,
                 dtype=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size,
                                    elementwise_affine=False,
                                    eps=1e-6,
                                    dtype=dtype)
        self.linear = Linear(hidden_size,
                             num_patch * out_channels,
                             bias=True,
                             dtype=dtype)
        self.scale_shift_table = Parameter(torch.randn(2, hidden_size) /
                                           hidden_size**0.5,
                                           dtype=dtype)
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s
        self.dtype = dtype
        self.mapping = mapping
        self.quant_mode = quant_mode

    def t_mask_select(self, x_mask, x, masked_x, T: int, S: int):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = where(expand_dims(x_mask,
                              [x_mask.ndim(), x_mask.ndim() + 1]), x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self,
                x,
                t,
                x_mask=None,
                t0=None,
                T: Optional[int] = None,
                S: Optional[int] = None):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s
        shift, scale = chunk(expand_dims(self.scale_shift_table.value, 0) +
                             expand_dims(t, 1),
                             chunks=2,
                             dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        if x_mask is not None:
            shift_zero, scale_zero = chunk(
                expand_dims(self.scale_shift_table.value, 0) +
                expand_dims(t0, 1),
                chunks=2,
                dim=1)
            x_zero = t2i_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)
        x = self.linear(x)
        self.register_network_output('output', x)
        return x


class PositionEmbedding2D(Module):

    def __init__(self,
                 dim: int,
                 dtype=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4"
        half_dim = dim // 2
        self.inv_freq = Parameter(
            1.0 / (10000**(torch.arange(0, half_dim, 2).float() / half_dim)),
            is_buffer=True,
            dtype=dtype)
        self.dtype = dtype
        self.mapping = mapping
        self.quant_mode = quant_mode

    def _get_sin_cos_emb(self, t):
        out = einsum("i,d->id", [t, self.inv_freq.value])
        emb_cos = cos(out)
        emb_sin = sin(out)
        return concat([emb_sin, emb_cos], dim=-1)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
        self,
        dtype,
        h: int,
        w: int,
        scale: Tensor,
        base_size: Optional[int] = None,
    ):
        grid_h = div(arange(0, h, 'float32'), scale.cast('float32'))
        grid_w = div(arange(0, w, 'float32'), scale.cast('float32'))
        if base_size is not None:
            grid_h *= float(base_size) / h
            grid_w *= float(base_size) / w
        grid_h, grid_w = meshgrid2d(grid_w, grid_h)  # here w goes first
        grid_h = permute(grid_h, [1, 0]).flatten()
        grid_w = permute(grid_w, [1, 0]).flatten()
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return unsqueeze(concat([emb_h, emb_w], dim=-1), 0).cast(dtype)

    def forward(self, x, h: int, w: int, scale: Tensor, base_size=None):
        pos_embedding = self._get_cached_emb(x.dtype, h, w, scale, base_size)
        self.register_network_output('output', pos_embedding)
        return pos_embedding


class TimestepEmbedder(Module):

    def __init__(self,
                 hidden_size,
                 frequency_embedding_size=256,
                 dtype=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.mlp = ModuleSequential([
            Linear(frequency_embedding_size,
                   hidden_size,
                   bias=True,
                   dtype=dtype),
            Activation('silu'),
            Linear(hidden_size, hidden_size, bias=True, dtype=dtype)
        ])
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype
        self.mapping = mapping
        self.quant_mode = quant_mode

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, dtype=None):
        half = dim // 2
        freqs = exp(
            -math.log(max_period) *
            arange(start=0, end=half, dtype=trt_dtype_to_str(trt.float32)) /
            constant(np.array([half], dtype=np.float32)))
        args = unsqueeze(t, -1).cast(trt.float32) * unsqueeze(freqs, 0)
        embedding = concat([cos(args), sin(args)], dim=-1)
        if dtype is not None:
            embedding = embedding.cast(dtype)
        if dim % 2:
            embedding = pad(embedding, (0, 0, 0, 1))
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(
            t, self.frequency_embedding_size).cast(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class SizeEmbedder(TimestepEmbedder):

    def __init__(self,
                 hidden_size,
                 frequency_embedding_size=256,
                 dtype=str_dtype_to_trt("float16"),
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        super().__init__(hidden_size=hidden_size,
                         frequency_embedding_size=frequency_embedding_size,
                         dtype=dtype,
                         mapping=mapping,
                         quant_mode=quant_mode)
        self.mlp = ModuleSequential([
            Linear(frequency_embedding_size,
                   hidden_size,
                   bias=True,
                   dtype=dtype),
            Activation('silu'),
            Linear(hidden_size, hidden_size, bias=True, dtype=dtype)
        ])
        self.outdim = hidden_size

    def forward(self, s, bs: int):
        if not USE_STATIC_SHAPE:
            raise NotImplementedError('Only static shape is supported')
        if s.ndim() == 1:
            s = unsqueeze(s, 1)
        assert s.ndim() == 2
        if s.shape[0] != bs:
            s = repeat(s, [bs // s.shape[0], 1])
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).cast(
            self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb,
                          "(b d) d2 -> b (d d2)",
                          b=b,
                          d=dims,
                          d2=self.outdim)
        self.register_network_output('output', s_emb)
        return s_emb


class CaptionMLP(Module):

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer="gelu",
            bias=True,
            dtype=None,
            mapping=Mapping(),
            quant_mode=QuantMode(0),
            inner_layernorm=False,
            eps=1e-05,
    ):
        super().__init__()
        hidden_act = act_layer
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        fc_output_size = 2 * hidden_features if hidden_act in [
            'swiglu', 'gegelu'
        ] else hidden_features
        self.inner_layernorm = LayerNorm(hidden_features, dtype=dtype,
                                         eps=eps) if inner_layernorm else None

        self.fc1 = ColumnLinear(in_features,
                                fc_output_size,
                                bias=bias[0],
                                dtype=dtype,
                                tp_group=mapping.tp_group,
                                tp_size=mapping.tp_size,
                                gather_output=False)
        self.fc2 = RowLinear(hidden_features,
                             out_features,
                             bias=bias[1],
                             dtype=dtype,
                             tp_group=mapping.tp_group,
                             tp_size=mapping.tp_size)

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.hidden_act = hidden_act
        self.dtype = dtype
        self.bias = bias
        self.mapping = mapping
        self.quant_mode = quant_mode
        self.eps = eps

    def forward(self, hidden_states, gegelu_limit=None):
        inter = self.fc1(hidden_states)
        if self.hidden_act == 'gegelu':
            inter = ACT2FN[self.hidden_act](inter, gegelu_limit)
        else:
            inter = ACT2FN[self.hidden_act](inter)
        if self.inner_layernorm is not None:
            inter = self.inner_layernorm(inter)
        output = self.fc2(inter)
        return output


class CaptionEmbedder(Module):

    def __init__(self,
                 in_channels,
                 hidden_size,
                 uncond_prob,
                 act_layer='gelu',
                 token_num=120,
                 dtype=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.y_proj = CaptionMLP(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            mapping=mapping,
            dtype=dtype,
        )
        self.y_embedding = Parameter(torch.randn(token_num, in_channels) /
                                     in_channels**0.5,
                                     dtype=dtype)
        self.uncond_prob = uncond_prob
        self.dtype = dtype
        self.mapping = mapping
        self.quant_mode = quant_mode

    def token_drop(self, caption, force_drop_ids=None):
        if not USE_STATIC_SHAPE:
            raise NotImplementedError('Only static shape is supported')
        assert (isinstance(force_drop_ids, torch.Tensor)
                or isinstance(force_drop_ids, np.array))
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = torch.Tensor(force_drop_ids) == 1
        drop_ids = constant(drop_ids.cpu().numpy())
        caption = where(expand_dims(drop_ids, [1, 2, 3]),
                        self.y_embedding.value, caption)
        return caption

    def forward(self, caption, force_drop_ids=None):
        if force_drop_ids is not None:
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        self.register_network_output('output', caption)
        return caption


class PatchEmbed3D(Module):

    def __init__(self,
                 patch_size=(2, 4, 4),
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None,
                 flatten=True,
                 dtype=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv3d(in_chans,
                           embed_dim,
                           kernel_size=patch_size,
                           stride=patch_size,
                           dtype=dtype)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        self.dtype = dtype
        self.mapping = mapping
        self.quant_mode = quant_mode

    def forward(self, x):
        if not USE_STATIC_SHAPE:
            raise NotImplementedError('Only static shape is supported')
        _, _, D, H, W = x.shape
        if W % self.patch_size[2] != 0:
            x = pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = pad(
                x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D = shape(x, 2)
            Wh = shape(x, 3)
            Ww = shape(x, 4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view([-1, self.embed_dim, D, Wh, Ww])
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        self.register_network_output('output', x)
        return x


class STDiT3Block(Module):

    def __init__(self,
                 hidden_size,
                 num_heads,
                 mlp_ratio=4.0,
                 rope=None,
                 qk_norm=False,
                 temporal=False,
                 dtype=None,
                 local_layer_idx=0,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.temporal = temporal
        self.hidden_size = hidden_size

        attn_cls = STDiTAttention
        mha_cls = STDiTCrossAttention

        self.norm1 = LayerNorm(hidden_size,
                               eps=1e-6,
                               elementwise_affine=False,
                               dtype=dtype)
        self.attn = attn_cls(hidden_size=hidden_size,
                             num_attention_heads=num_heads,
                             qk_layernorm=qk_norm,
                             bias=True,
                             rotary_embedding_func=rope,
                             dtype=dtype,
                             tp_group=mapping.tp_group,
                             tp_size=mapping.tp_size,
                             tp_rank=mapping.tp_rank,
                             cp_group=mapping.cp_group,
                             cp_size=mapping.cp_size,
                             quant_mode=quant_mode)
        self.cross_attn = mha_cls(
            local_layer_idx=local_layer_idx,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            attention_mask_type=tensorrt_llm.layers.AttentionMaskType.causal,
            qkv_bias=True,
            dense_bias=True,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            tp_rank=mapping.tp_rank,
            cp_group=mapping.cp_group,
            cp_size=mapping.cp_size,
            quant_mode=quant_mode)
        self.norm2 = LayerNorm(hidden_size,
                               eps=1e-6,
                               elementwise_affine=False,
                               dtype=dtype)
        self.mlp = CaptionMLP(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer='gelu',
            mapping=mapping,
            dtype=dtype,
        )
        self.scale_shift_table = Parameter(torch.randn(6, hidden_size) /
                                           hidden_size**0.5,
                                           dtype=dtype)
        self.dtype = dtype
        self.mapping = mapping
        self.quant_mode = quant_mode

    def t_mask_select(self, x_mask, x, masked_x, T: int, S: int):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = where(expand_dims(x_mask, [2, 3]), x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(
        self,
        x,
        y,
        t,
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T: Optional[int] = None,  # number of frames
        S: Optional[int] = None,  # number of pixel patches
        attention_params: Optional[AttentionParams] = None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
    ):
        if not USE_STATIC_SHAPE:
            raise NotImplementedError('Only static shape is supported')
        # prepare modulate parameters
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(
            expand_dims(self.scale_shift_table.value, 0) + t.view([B, 6, -1]),
            chunks=6,
            dim=1)
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = chunk(
                expand_dims(self.scale_shift_table.value, 0) +
                t0.view([B, 6, -1]),
                chunks=6,
                dim=1)

        # modulate (attention)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero,
                                    scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # attention
        if self.temporal:
            x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
            x_m = self.attn(
                x_m, max_input_length=attention_params.max_context_length)
            x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        else:
            x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
            x_m = self.attn(
                x_m, max_input_length=attention_params.max_context_length)
            x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

        # modulate (attention)
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + x_m_s

        # cross attention
        cattn = self.cross_attn(
            hidden_states=x,
            encoder_output=y,
            attention_params=AttentionParams(
                sequence_length=attention_params.sequence_length,
                context_lengths=attention_params.context_lengths,
                host_context_lengths=attention_params.host_context_lengths,
                max_context_length=attention_params.max_context_length,
                host_request_types=attention_params.host_request_types,
                encoder_input_lengths=attention_params.encoder_input_lengths,
                encoder_max_input_length=attention_params.
                encoder_max_input_length,
                host_runtime_perf_knobs=attention_params.
                host_runtime_perf_knobs,
                host_context_progress=attention_params.host_context_progress,
            ),
            kv_cache_params=KeyValueCacheParams(
                past_key_value=kv_cache_params.past_key_value,
                host_past_key_value_lengths=kv_cache_params.
                host_past_key_value_lengths,
                host_max_attention_window_sizes=kv_cache_params.
                host_max_attention_window_sizes,
                host_sink_token_length=kv_cache_params.host_sink_token_length,
                cache_indirection=kv_cache_params.cache_indirection,
                kv_cache_block_offsets=None,
                host_kv_cache_block_offsets=None,
                host_kv_cache_pool_pointers=None,
                host_kv_cache_pool_mapping=None,
                cross_kv_cache_block_offsets=None,
                host_cross_kv_cache_block_offsets=None,
                host_cross_kv_cache_pool_pointers=None,
                host_cross_kv_cache_pool_mapping=None,
            ))
        x = x + cattn

        # modulate (MLP)
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero,
                                    scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + x_m_s

        return x


class STDiT3Model(PretrainedModel):

    def __init__(self, config: STDiTModelConfig):
        self.check_config(config)
        super().__init__(config)
        self.learn_sigma = config.learn_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.learn_sigma else config.in_channels
        self.caption_channels = config.caption_channels
        self.depth = config.num_hidden_layers
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.model_max_length = config.model_max_length
        self.latent_size = config.latent_size
        self.input_sq_size = config.input_sq_size
        self.patch_size = config.stdit_patch_size
        self.class_dropout_prob = config.class_dropout_prob
        self.qk_norm = config.qk_norm
        self.dtype = config.dtype
        self.mapping = config.mapping

        self.pos_embed = PositionEmbedding2D(self.hidden_size, dtype=self.dtype)
        self.rope = RotaryEmbedder(dim=self.hidden_size // self.num_heads,
                                   dtype=self.dtype)
        self.x_embedder = PatchEmbed3D(self.patch_size,
                                       self.in_channels,
                                       self.hidden_size,
                                       dtype=self.dtype)
        self.t_embedder = TimestepEmbedder(self.hidden_size, dtype=self.dtype)
        self.fps_embedder = SizeEmbedder(self.hidden_size, dtype=self.dtype)
        self.t_block = ModuleSequential([
            Activation('silu'),
            Linear(self.hidden_size,
                   6 * self.hidden_size,
                   bias=True,
                   dtype=self.dtype)
        ])
        self.y_embedder = CaptionEmbedder(in_channels=self.caption_channels,
                                          hidden_size=self.hidden_size,
                                          uncond_prob=self.class_dropout_prob,
                                          act_layer='gelu',
                                          token_num=self.model_max_length,
                                          dtype=self.dtype)
        self.spatial_blocks = ModuleList([
            STDiT3Block(hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        qk_norm=self.qk_norm,
                        dtype=self.dtype,
                        local_layer_idx=idx,
                        mapping=self.mapping) for idx in range(self.depth)
        ])
        self.temporal_blocks = ModuleList([
            STDiT3Block(hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        qk_norm=self.qk_norm,
                        temporal=True,
                        rope=self.rope.rotate_queries_or_keys,
                        dtype=self.dtype,
                        local_layer_idx=idx,
                        mapping=self.mapping) for idx in range(self.depth)
        ])
        self.final_layer = T2IFinalLayer(self.hidden_size,
                                         np.prod(self.patch_size),
                                         self.out_channels,
                                         dtype=self.dtype,
                                         mapping=self.mapping)

    def check_config(self, config: PretrainedConfig):
        config.set_if_not_exist('caption_channels', 4096)
        config.set_if_not_exist('num_hidden_layers', 28)
        config.set_if_not_exist('latent_size', [30, 45, 80])
        config.set_if_not_exist('hidden_size', 1152)
        config.set_if_not_exist('stdit_patch_size', [1, 2, 2])
        config.set_if_not_exist('in_channels', 4)
        config.set_if_not_exist('input_sq_size', 512)
        config.set_if_not_exist('num_attention_heads', 16)
        config.set_if_not_exist('mlp_ratio', 4.0)
        config.set_if_not_exist('class_dropout_prob', 0.1)
        config.set_if_not_exist('model_max_length', 300)
        config.set_if_not_exist('learn_sigma', True)
        config.set_if_not_exist('dtype', None)
        config.set_if_not_exist('qk_norm', True)
        config.set_if_not_exist('skip_y_embedder', False)

    def __post_init__(self):
        return

    def get_dynamic_size(self, x: Tensor):
        if not USE_STATIC_SHAPE:
            raise NotImplementedError('Only static shape is supported')
        _, _, T, H, W = x.shape
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def encode_text(self, y: Tensor, mask: Optional[Tensor] = None):
        y = self.y_embedder(y)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = repeat(mask, sizes=(y.shape[0] // mask.shape[0], 1))
            mask = squeeze(squeeze(mask, 1), 1)
            y = masked_select(
                squeeze(y, 1),
                where(
                    unsqueeze(mask, -1).__eq__(
                        constant_to_tensor_(0, dtype=mask.dtype)),
                    constant_to_tensor_(False),
                    constant_to_tensor_(True))).view((1, -1, self.hidden_size))
            # [TODO] how to convert y_lens to list?
            # y_lens = mask.sum(dim=1).tolist()
            y_lens = sum(mask, dim=1)
        else:
            y_lens = constant(
                np.array([y.shape[2]] * y.shape[0], dtype=np.int64))
            y = squeeze(y, 1).view((1, -1, self.hidden_size))
        self.register_network_output('encode_text.output.y', y)
        self.register_network_output('encode_text.output.y_lens', y_lens)
        return y, y_lens

    def unpatchify(self, x: Tensor, N_t: int, N_h: int, N_w: int, R_t: int,
                   R_h: int, R_w: int):
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = slice(x,
                  starts=[0] * x.ndim(),
                  sizes=concat([
                      shape(x, 0),
                      shape(x, 1),
                      constant(np.array([R_t, R_h, R_w]).astype(np.int64))
                  ]))
        self.register_network_output('unpatchify.output', x)
        return x

    def forward(self,
                x: Tensor,
                timestep: Tensor,
                y: Tensor,
                fps: Tensor,
                height: Tensor,
                width: Tensor,
                mask: Optional[Tensor] = None,
                x_mask: Optional[Tensor] = None,
                attention_params: Optional[AttentionParams] = None,
                kv_cache_params: Optional[KeyValueCacheParams] = None,
                **kwargs):
        if not USE_STATIC_SHAPE:
            raise NotImplementedError('Only static shape is supported')
        assert tuple(x.shape[2:]) == tuple(
            self.latent_size), "For now only static shape is supported."
        B = x.shape[0]
        x = x.cast(self.dtype)
        timestep = timestep.cast(self.dtype)
        y = y.cast(self.dtype)
        fps = fps.cast(self.dtype)

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.shape
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        base_size = round(S**0.5)
        resolution_sq = pow(
            height.cast('float32') * width.cast('float32'),
            constant_to_tensor_(0.5, dtype='float32'))
        scale = (resolution_sq / self.input_sq_size).cast(self.dtype)
        pos_emb = self.pos_embed(x, h=H, w=W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(unsqueeze(fps, 1), bs=B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = constant(
                np.zeros(shape=timestep.shape).astype(np.float32)).cast(x.dtype)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        # === get y embed ===
        if self.config.skip_y_embedder:
            y_lens = mask
            if isinstance(y_lens, Tensor):
                y_lens = y_lens.cast('int64')
        else:
            y, y_lens = self.encode_text(y, mask)
        y_lens = None  #[11, 11]

        # === get x embed ===
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb
        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === blocks ===
        cnt = 0
        for spatial_block, temporal_block in zip(self.spatial_blocks,
                                                 self.temporal_blocks):
            x = spatial_block(
                x,
                y,
                t_mlp,
                x_mask=x_mask,
                t0=t0_mlp,
                T=T,
                S=S,
                attention_params=attention_params,
                kv_cache_params=kv_cache_params,
            )
            x = temporal_block(
                x,
                y,
                t_mlp,
                x_mask=x_mask,
                t0=t0_mlp,
                T=T,
                S=S,
                attention_params=attention_params,
                kv_cache_params=kv_cache_params,
            )
            cnt += 1

        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # cast to float32 for better accuracy
        output = x.cast('float32')
        output.mark_output('output', 'float32')
        return output

    def prepare_inputs(self, max_batch_size, **kwargs):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        mapping = self.config.mapping
        if mapping.tp_size > 1:
            current_all_reduce_helper().set_workspace_tensor(mapping, 1)

        def stdit_default_batch_range(max_batch_size):
            return [max_batch_size, max_batch_size, max_batch_size]

        default_range = stdit_default_batch_range
        # [NOTE] For now only static batch size is supported, so we run the model with max_batch_size.
        batch_size = max_batch_size

        x = Tensor(name='x',
                   dtype=self.dtype,
                   shape=[batch_size, self.in_channels, *self.latent_size],
                   dim_range=OrderedDict([
                       ('batch_size', [default_range(max_batch_size)]),
                       ('in_channels', [[self.in_channels] * 3]),
                       ('latent_frames', [[self.latent_size[0]] * 3]),
                       ('latent_height', [[self.latent_size[1]] * 3]),
                       ('latent_width', [[self.latent_size[2]] * 3]),
                   ]))
        timestep = Tensor(name='timestep',
                          dtype=self.dtype,
                          shape=[batch_size],
                          dim_range=OrderedDict([
                              ('batch_size', [default_range(max_batch_size)]),
                          ]))
        y = Tensor(
            name='y',
            dtype=self.dtype,
            shape=[batch_size, 1, self.model_max_length, self.caption_channels],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
                ('mask_batch_size', [[1, 1, 1]]),
                ('num_tokens', [[self.model_max_length] * 3]),
                ('caption_channels', [[self.caption_channels] * 3]),
            ]))
        mask = Tensor(name='mask',
                      dtype=trt.int32,
                      shape=[1, self.model_max_length],
                      dim_range=OrderedDict([
                          ('mask_batch_size', [[1, 1, 1]]),
                          ('num_tokens', [[self.model_max_length] * 3]),
                      ]))
        x_mask = Tensor(name='x_mask',
                        dtype=tensorrt_llm.str_dtype_to_trt('bool'),
                        shape=[batch_size, self.latent_size[0]],
                        dim_range=OrderedDict([
                            ('batch_size', [default_range(max_batch_size)]),
                            ('latent_frames', [[self.latent_size[0]] * 3]),
                        ]))
        fps = Tensor(name='fps', dtype=self.dtype, shape=[
            1,
        ])
        height = Tensor(name='height', dtype=self.dtype, shape=[
            1,
        ])
        width = Tensor(name='width', dtype=self.dtype, shape=[
            1,
        ])

        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        remove_input_padding = default_net().plugin_config.remove_input_padding

        cross_attn_batch_size = batch_size
        max_cattn_seq_len = int(
            np.prod([
                np.ceil(d / p)
                for d, p in zip(self.latent_size, self.patch_size)
            ]))
        max_cattn_enc_len = self.model_max_length
        attn_inputs = GenerationMixin().prepare_attention_inputs(
            max_batch_size=cross_attn_batch_size,
            opt_batch_size=cross_attn_batch_size,
            max_beam_width=1,
            max_input_len=max_cattn_seq_len,
            max_seq_len=max_cattn_seq_len,
            num_kv_heads=self.num_heads,
            head_size=self.hidden_size // self.num_heads,
            num_layers=self.depth,
            kv_dtype=self.dtype,
            kv_cache_type=KVCacheType.DISABLED,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            enable_ctx_gen_opt_profiles=False,
            mapping=self.mapping,
        )

        sequence_length = attn_inputs['sequence_length']
        host_context_lengths = attn_inputs['host_context_lengths']
        host_max_attention_window_sizes = attn_inputs[
            'host_max_attention_window_sizes']
        host_sink_token_length = attn_inputs['host_sink_token_length']
        context_lengths = attn_inputs['context_lengths']
        host_request_types = attn_inputs['host_request_types']

        host_past_key_value_lengths = attn_inputs['host_past_key_value_lengths']
        past_key_value = attn_inputs['past_key_value']
        if past_key_value:
            past_key_value = past_key_value[0]
        cache_indirection = attn_inputs['cache_indirection']
        host_runtime_perf_knobs_tensor = attn_inputs['host_runtime_perf_knobs']
        host_context_progress = attn_inputs['host_context_progress']
        cross_encoder_input_lengths = Tensor(name='encoder_input_lengths',
                                             shape=(cross_attn_batch_size, ),
                                             dtype=str_dtype_to_trt('int32'))
        cross_max_encoder_seq_len = Tensor(
            name='encoder_max_input_length',
            shape=[-1],
            dim_range=OrderedDict([
                ("encoder_max_input_length",
                 [[1, (max_cattn_enc_len + 1) // 2, max_cattn_enc_len]])
            ]),
            dtype=str_dtype_to_trt('int32'))

        attention_params = AttentionParams(
            sequence_length=sequence_length,
            context_lengths=context_lengths,
            host_context_lengths=host_context_lengths,
            max_context_length=max_cattn_seq_len,
            host_request_types=host_request_types,
            encoder_input_lengths=cross_encoder_input_lengths,
            encoder_max_input_length=cross_max_encoder_seq_len,
            host_runtime_perf_knobs=host_runtime_perf_knobs_tensor,
            host_context_progress=host_context_progress)

        kv_cache_params = KeyValueCacheParams(
            past_key_value=past_key_value,
            host_past_key_value_lengths=host_past_key_value_lengths,
            host_max_attention_window_sizes=host_max_attention_window_sizes,
            host_sink_token_length=host_sink_token_length,
            cache_indirection=cache_indirection,
            kv_cache_block_offsets=None,
            host_kv_cache_block_offsets=None,
            host_kv_cache_pool_pointers=None,
            host_kv_cache_pool_mapping=None,
            cross_kv_cache_block_offsets=None,
            host_cross_kv_cache_block_offsets=None,
            host_cross_kv_cache_pool_pointers=None,
            host_cross_kv_cache_pool_mapping=None,
        )

        return {
            'x': x,
            'timestep': timestep,
            'y': y,
            'mask': mask,
            'x_mask': x_mask,
            'fps': fps,
            'height': height,
            'width': width,
            'attention_params': attention_params,
            'kv_cache_params': kv_cache_params,
        }

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_dir: str,
                        dtype='float16',
                        mapping=Mapping(),
                        **kwargs):

        quant_ckpt_path = kwargs.pop('quant_ckpt_path', None)
        assert os.path.exists(f"{pretrained_model_dir}/config.json")
        with open(f"{pretrained_model_dir}/config.json", 'r') as f:
            hf_config = json.load(f)

        hf_tllm_config_remapping = {
            'model_type': 'architecture',
            'depth': 'num_hidden_layers',
            'num_heads': 'num_attention_heads',
            'patch_size': 'stdit_patch_size',
            'pred_sigma': 'learn_sigma',
        }
        for hf_key, tllm_key in hf_tllm_config_remapping.items():
            hf_config[tllm_key] = hf_config.pop(hf_key)
        hf_config.update(kwargs)

        model_config = STDiTModelConfig.from_input_config(hf_config,
                                                          dtype=dtype,
                                                          mapping=mapping)

        model_dir = pretrained_model_dir
        custom_dict = {}
        if quant_ckpt_path is not None:
            model_dir = quant_ckpt_path

        loader = STDiT3ModelWeightsLoader(model_dir, custom_dict)
        model = cls(model_config)
        loader.generate_tllm_weights(model)
        return model


class STDiT3ModelWeightsLoader(ModelWeightsLoader):

    def translate_to_external_key(self, tllm_key: str,
                                  tllm_to_externel_key_dict: dict):
        """Convert and load external checkpoint into a TensorRT LLM model.
        """
        trtllm_to_hf_name = {
            r"spatial_blocks.(\d+).attn.q_layernorm.weight":
            "spatial_blocks.*.attn.q_norm.weight",
            r"spatial_blocks.(\d+).attn.k_layernorm.weight":
            "spatial_blocks.*.attn.k_norm.weight",
            r"spatial_blocks.(\d+).attn.dense.weight":
            "spatial_blocks.*.attn.proj.weight",
            r"spatial_blocks.(\d+).attn.dense.bias":
            "spatial_blocks.*.attn.proj.bias",
            r"temporal_blocks.(\d+).attn.q_layernorm.weight":
            "temporal_blocks.*.attn.q_norm.weight",
            r"temporal_blocks.(\d+).attn.k_layernorm.weight":
            "temporal_blocks.*.attn.k_norm.weight",
            r"temporal_blocks.(\d+).attn.dense.weight":
            "temporal_blocks.*.attn.proj.weight",
            r"temporal_blocks.(\d+).attn.dense.bias":
            "temporal_blocks.*.attn.proj.bias",
            r"spatial_blocks.(\d+).cross_attn.dense.weight":
            "spatial_blocks.*.cross_attn.proj.weight",
            r"spatial_blocks.(\d+).cross_attn.dense.bias":
            "spatial_blocks.*.cross_attn.proj.bias",
            r"temporal_blocks.(\d+).cross_attn.dense.weight":
            "temporal_blocks.*.cross_attn.proj.weight",
            r"temporal_blocks.(\d+).cross_attn.dense.bias":
            "temporal_blocks.*.cross_attn.proj.bias",
        }

        for k, v in trtllm_to_hf_name.items():
            m = re.match(k, tllm_key)
            if m is not None:
                matched_pos = m.groups()
                placeholders = v.count('*')
                assert len(matched_pos) == placeholders
                for i in range(len(matched_pos)):
                    v = v.replace('*', matched_pos[i], 1)
                return v
        return tllm_key

    def load_tensor(self, key, tp_size=1, tp_dim=-1, tp_rank=0):
        hidden_size = self.model.config.hidden_size

        if "attn.qkv" in key:
            is_cross_attn = "cross_attn.qkv" in key
            if is_cross_attn:
                # process for cross attention
                process_qkv_names = [
                    'q_linear'.join(key.split('qkv')),
                    'kv_linear'.join(key.split('qkv'))
                ]
            else:
                process_qkv_names = [key]
            qkv_tensors = []
            for qkv_key in process_qkv_names:
                # Retrieve shard index
                assert qkv_key in self.shard_map
                ptr_idx = self.shard_map[qkv_key]
                if self.format == ModelWeightsFormat.SAFETENSORS:
                    # Force to load Pytorch tensor
                    tensor = self.shards[ptr_idx].get_tensor(qkv_key)
                else:
                    tensor = self.shards[ptr_idx][qkv_key]
                qkv_tensors.append(tensor)
            if is_cross_attn:
                tensor = torch.concat(qkv_tensors, dim=0)
            else:
                tensor = qkv_tensors[0]
            # Post-process weight and bias if tp_size > 1
            if tp_size > 1:
                if "weight" in key:
                    tensor = tensor.reshape(3, hidden_size, hidden_size)
                elif "bias" in key:
                    tensor = tensor.reshape(3, hidden_size)
                tp_dim = 1
            tensor_shape = tensor.shape
        else:
            # Retrieve shard index
            if key in self.shard_map:
                ptr_idx = self.shard_map[key]
            else:
                return None

            if self.format == ModelWeightsFormat.SAFETENSORS:
                tensor = self.shards[ptr_idx].get_slice(key)
                tensor_shape = tensor.get_shape()
                if tensor_shape == []:
                    tensor = self.shards[ptr_idx].get_tensor(key).unsqueeze(0)
                    tensor_shape = tensor.shape
            else:
                tensor = self.shards[ptr_idx][key]
                tensor_shape = tensor.shape

        if tp_size <= 1 or tp_dim < 0:
            return tensor[:]
        else:
            if len(tensor_shape) == 1 and (tp_dim > 0 or tensor_shape[0] == 1):
                return tensor[:]
            else:
                width = tensor_shape[tp_dim]
                if width == 1:
                    return tensor[:]
                slice_width = math.ceil(width / tp_size)
                slice_start = tp_rank * slice_width
                slice_end = builtins.min((tp_rank + 1) * slice_width, width)
                slice_obj = [builtins.slice(None)] * len(tensor_shape)
                slice_obj[tp_dim] = builtins.slice(slice_start, slice_end)
                res = tensor[tuple(slice_obj)]
                if "qkv.weight" in key:
                    res = res.reshape(3 * (hidden_size // tp_size), hidden_size)
                elif "qkv.bias" in key:
                    res = res.reshape(3 * (hidden_size // tp_size))
                return res

    def generate_tllm_weights(self,
                              model,
                              custom_postprocess_kwargs: dict = {}):
        self.update_key_mapping(model)
        tp_module_patterns = [
            r'.*_blocks.*.attn.qkv.weight$',
            r'.*_blocks.*.attn.qkv.bias$',
            r'.*_blocks.*.attn.dense.weight$',
            r'.*_blocks.*.cross_attn.qkv.weight$',
            r'.*_blocks.*.cross_attn.qkv.bias$',
            r'.*_blocks.*.cross_attn.dense.weight$',
            r'.*_blocks.*.mlp.fc1.weight$',
            r'.*_blocks.*.mlp.fc1.bias$',
            r'.*_blocks.*.mlp.fc2.weight$',
        ]
        tllm_weights = {}
        for tllm_key, _ in tqdm(model.named_parameters()):
            skip_tp = not any([
                re.match(pattern, tllm_key) is not None
                for pattern in tp_module_patterns
            ])
            tllm_weights.update(
                self.load(tllm_key,
                          custom_postprocess_kwargs=custom_postprocess_kwargs,
                          skip_tp=skip_tp))
        self.fill(tllm_weights)
