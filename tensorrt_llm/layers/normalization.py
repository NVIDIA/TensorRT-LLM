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
from typing import Optional

from ..functional import (ACT2FN, Tensor, chunk, group_norm, layer_norm,
                          rms_norm, unsqueeze)
from ..mapping import Mapping
from ..module import Module
from ..parameter import Parameter
from .embedding import CombinedTimestepLabelEmbeddings, Embedding
from .linear import Linear


class LayerNorm(Module):

    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 elementwise_affine=True,
                 bias=True,
                 dtype=None,
                 tp_size=1,
                 tp_dim=-1):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
            if bias:
                self.bias = Parameter(shape=self.normalized_shape, dtype=dtype)
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.eps = eps
        self.dtype = dtype
        self.tp_size = tp_size
        self.tp_dim = tp_dim

    def forward(self, x, normalized_shape=None):
        weight = 1. if self.weight is None else self.weight.value
        bias = 0. if self.bias is None else self.bias.value
        if normalized_shape is None:
            normalized_shape = self.normalized_shape
        return layer_norm(x, normalized_shape, weight, bias, self.eps)


class RmsNorm(Module):

    def __init__(self,
                 normalized_shape,
                 num_groups=1,
                 eps=1e-06,
                 elementwise_affine=True,
                 dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        self.num_groups = num_groups
        num_channels = normalized_shape[-1]
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)

        self.eps = eps
        self.dtype = dtype

    def forward(self, x, normalized_shape=None):
        weight = None if self.weight is None else self.weight.value
        if normalized_shape is None:
            normalized_shape = self.normalized_shape
        return rms_norm(x, normalized_shape, self.num_groups, weight, self.eps)


class GroupNorm(Module):

    def __init__(self,
                 num_groups,
                 num_channels,
                 eps=1e-05,
                 affine=True,
                 dtype=None):
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.affine = affine

        if self.affine:
            self.weight = Parameter(shape=(self.num_channels, ), dtype=dtype)
            self.bias = Parameter(shape=(self.num_channels, ), dtype=dtype)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.eps = eps

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        bias = None if self.bias is None else self.bias.value
        return group_norm(x, self.num_groups, weight, bias, self.eps)


class AdaLayerNorm(Module):

    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 norm_elementwise_affine: bool = False,
                 norm_eps: float = 1e-5,
                 chunk_dim: int = 0,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2
        if num_embeddings is not None:
            self.emb = Embedding(num_embeddings, embedding_dim, dtype=dtype)
        else:
            self.emb = None
        self.silu = ACT2FN['silu']
        self.linear = Linear(embedding_dim,
                             output_dim,
                             tp_group=mapping.tp_group,
                             tp_size=mapping.tp_size,
                             dtype=dtype)
        self.norm = LayerNorm(output_dim // 2,
                              eps=norm_eps,
                              elementwise_affine=norm_elementwise_affine,
                              dtype=dtype)

    def forward(self,
                x: Tensor,
                timestep: Optional[Tensor] = None,
                temb: Optional[Tensor] = None):
        assert timestep is not None or temb is not None
        if self.emb is not None and timestep is not None:
            temb = self.emb(timestep)
        temb = self.linear(self.silu(temb))
        if self.chunk_dim == 1:
            shift, scale = chunk(temb, 2, dim=1)
            shift = unsqueeze(shift, 1)
            scale = unsqueeze(scale, 1)
        else:
            scale, shift = chunk(temb, 2, dim=0)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(Module):

    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: Optional[int] = None,
                 norm_type: str = "layer_norm",
                 bias: bool = True,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings,
                                                       embedding_dim,
                                                       dtype=dtype)
        else:
            self.emb = None

        self.silu = ACT2FN['silu']
        self.linear = Linear(embedding_dim,
                             6 * embedding_dim,
                             bias=bias,
                             tp_group=mapping.tp_group,
                             tp_size=mapping.tp_size,
                             dtype=dtype)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim,
                                  elementwise_affine=False,
                                  eps=1e-6,
                                  dtype=dtype)
        elif norm_type == "fp32_layer_norm":
            self.norm = LayerNorm(embedding_dim,
                                  elementwise_affine=False,
                                  bias=False,
                                  dtype=dtype)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(self,
                x: Tensor,
                timestep: Optional[Tensor] = None,
                class_labels: Optional[Tensor] = None,
                hidden_dtype: str = None,
                emb: Optional[Tensor] = None):
        assert emb is not None or self.emb is not None
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(
            emb, 6, dim=1)
        x = self.norm(x) * (1 + unsqueeze(scale_msa, 1)) + unsqueeze(
            shift_msa, 1)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(Module):

    def __init__(self,
                 embedding_dim: int,
                 norm_type: str = "layer_norm",
                 bias: bool = True,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()

        self.silu = ACT2FN['silu']
        self.linear = Linear(embedding_dim,
                             3 * embedding_dim,
                             bias=bias,
                             tp_group=mapping.tp_group,
                             tp_size=mapping.tp_size,
                             dtype=dtype)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim,
                                  elementwise_affine=False,
                                  eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(self, x: Tensor, emb: Optional[Tensor] = None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = chunk(emb, 3, dim=1)
        x = self.norm(x) * (1 + unsqueeze(scale_msa, 1)) + unsqueeze(
            shift_msa, 1)
        return x, gate_msa


class AdaLayerNormContinuous(Module):

    def __init__(self,
                 embedding_dim: int,
                 conditioning_embedding_dim: int,
                 elementwise_affine: bool = True,
                 eps: float = 1e-5,
                 bias: bool = True,
                 norm_type: str = "layer_norm",
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        self.silu = ACT2FN['silu']
        self.linear = Linear(conditioning_embedding_dim,
                             embedding_dim * 2,
                             bias=bias,
                             tp_group=mapping.tp_group,
                             tp_size=mapping.tp_size,
                             dtype=dtype)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim,
                                  eps=eps,
                                  elementwise_affine=elementwise_affine,
                                  bias=bias,
                                  dtype=dtype)
        elif norm_type == "rms_norm":
            self.norm = RmsNorm(embedding_dim,
                                eps=eps,
                                elementwise_affine=elementwise_affine,
                                dtype=dtype)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: Tensor, conditioning_embedding: Tensor):
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(self.silu(conditioning_embedding).cast(x.dtype))
        scale, shift = chunk(emb, 2, dim=1)
        x = self.norm(x) * unsqueeze((1 + scale), 1) + unsqueeze(shift, 1)
        return x


class SD35AdaLayerNormZeroX(Module):

    def __init__(self,
                 embedding_dim: int,
                 norm_type: str = "layer_norm",
                 bias: bool = True,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        self.silu = ACT2FN['silu']
        self.linear = Linear(embedding_dim,
                             9 * embedding_dim,
                             bias=bias,
                             tp_group=mapping.tp_group,
                             tp_size=mapping.tp_size,
                             dtype=dtype)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim,
                                  elementwise_affine=False,
                                  eps=1e-6,
                                  dtype=dtype)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'."
            )

    def forward(self, hidden_states: Tensor, emb: Tensor):
        emb = self.linear(self.silu(emb).cast(hidden_states.dtype))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = chunk(
            emb, 9, dim=1)
        norm_hidden_states = self.norm(hidden_states)
        hidden_states = norm_hidden_states * (
            1 + unsqueeze(scale_msa, 1)) + unsqueeze(shift_msa, 1)
        norm_hidden_states2 = norm_hidden_states * (
            1 + unsqueeze(scale_msa2, 1)) + unsqueeze(shift_msa2, 1)
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2
