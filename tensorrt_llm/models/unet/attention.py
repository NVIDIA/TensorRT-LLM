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
import math
from typing import Optional

from ..._common import precision
from ...functional import geglu, matmul, softmax, split
from ...layers import Conv2d, GroupNorm, LayerNorm, Linear
from ...module import Module, ModuleList


class AttentionBlock(Module):

    def __init__(self,
                 channels: int,
                 num_head_channels: Optional[int] = None,
                 num_groups: int = 32,
                 rescale_output_factor: float = 1.0,
                 eps: float = 1e-5):
        super().__init__()
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.group_norm = GroupNorm(num_channels=channels,
                                    num_groups=num_groups,
                                    eps=eps,
                                    affine=True)

        self.qkv = Linear(channels, channels * 3)
        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = Linear(channels, channels, 1)

    def transpose_for_scores(self, projection):
        new_projection_shape = projection.size()[:-1] + (self.num_heads,
                                                         self.num_head_size)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(
            [0, 2, 1, 3])
        return new_projection

    def forward(self, hidden_states):
        assert not hidden_states.is_dynamic()

        residual = hidden_states
        batch, channel, height, width = hidden_states.size()

        # norm
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view([batch, channel,
                                            height * width]).transpose(1, 2)

        # proj to q, k, v
        qkv_proj = self.qkv(hidden_states)

        query_proj, key_proj, value_proj = split(qkv_proj, channel, dim=2)

        # transpose
        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        # get scores
        with precision('float32'):
            attention_scores = matmul(query_states,
                                      (key_states).transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(
                self.channels / self.num_heads)
            attention_probs = softmax(attention_scores, dim=-1)

        # compute attention output
        hidden_states = matmul(attention_probs, value_states)
        hidden_states = hidden_states.permute([0, 2, 1, 3])

        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels, )
        hidden_states = hidden_states.view(new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).view(
            [batch, channel, height, width])

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


def _transpose_for_scores(tensor, heads):
    batch_size, seq_len, dim = tensor.size()
    tensor = tensor.view([batch_size, seq_len, heads, dim // heads])
    tensor = tensor.permute([0, 2, 1, 3])
    return tensor


def _attention(query, key, value, scale):
    # Multiply scale first to avoid overflow
    # Do not use use_fp32_acc or it will be very slow
    attention_scores = matmul(query * math.sqrt(scale),
                              key.transpose(-1, -2) * math.sqrt(scale),
                              use_fp32_acc=False)
    attention_probs = softmax(attention_scores, dim=-1)
    hidden_states = matmul(attention_probs, value, use_fp32_acc=False)
    hidden_states = hidden_states.permute([0, 2, 1, 3])
    return hidden_states


class SelfAttention(Module):

    def __init__(self,
                 query_dim: int,
                 heads: int = 8,
                 dim_head: int = 64,
                 dtype=None):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads
        self._slice_size = None

        self.to_qkv = Linear(query_dim,
                             3 * self.inner_dim,
                             bias=False,
                             dtype=dtype)
        self.to_out = Linear(self.inner_dim, query_dim, dtype=dtype)

    def forward(self, hidden_states, mask=None):
        assert not hidden_states.is_dynamic()

        qkv = self.to_qkv(hidden_states)

        query, key, value = split(qkv, self.inner_dim, dim=2)
        query = _transpose_for_scores(query, self.heads)
        key = _transpose_for_scores(key, self.heads)
        value = _transpose_for_scores(value, self.heads)
        hidden_states = _attention(query, key, value, self.scale)

        batch_size, seq_len, head_size, head_dim = hidden_states.size()
        hidden_states = hidden_states.view(
            [batch_size, seq_len, head_size * head_dim])
        return self.to_out(hidden_states)


class CrossAttention(Module):

    def __init__(self,
                 query_dim: int,
                 context_dim: Optional[int] = None,
                 heads: int = 8,
                 dim_head: int = 64,
                 dtype=None):
        super().__init__()
        self.inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head**-0.5
        self.heads = heads
        self._slice_size = None

        self.to_q = Linear(query_dim, self.inner_dim, bias=False, dtype=dtype)
        self.to_kv = Linear(context_dim,
                            2 * self.inner_dim,
                            bias=False,
                            dtype=dtype)
        self.to_out = Linear(self.inner_dim, query_dim, dtype=dtype)

    def forward(self, hidden_states, context=None, mask=None):
        assert not hidden_states.is_dynamic()
        query = self.to_q(hidden_states)
        is_cross_attn = context is not None
        context = context if is_cross_attn else hidden_states
        assert not context.is_dynamic()
        kv = self.to_kv(context)

        query = _transpose_for_scores(query, self.heads)
        key, value = split(kv, self.inner_dim, dim=2)
        key = _transpose_for_scores(key, self.heads)
        value = _transpose_for_scores(value, self.heads)
        hidden_states = _attention(query, key, value, self.scale)

        batch_size, seq_len, head_size, head_dim = hidden_states.size()
        hidden_states = hidden_states.view(
            [batch_size, seq_len, head_size * head_dim])
        return self.to_out(hidden_states)


class FeedForward(Module):

    def __init__(self,
                 dim: int,
                 dim_out: Optional[int] = None,
                 mult: int = 4,
                 dtype=None):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.proj_in = Linear(dim, inner_dim * 2, dtype=dtype)
        self.proj_out = Linear(inner_dim, dim_out, dtype=dtype)

    def forward(self, hidden_states):
        x = self.proj_in(hidden_states)
        x = geglu(x)
        return self.proj_out(x)


class BasicTransformerBlock(Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: Optional[int] = None,
        dtype=None,
    ):
        super().__init__()
        self.attn1 = SelfAttention(query_dim=dim,
                                   heads=n_heads,
                                   dim_head=d_head,
                                   dtype=dtype)  # is a self-attention
        self.ff = FeedForward(dim, dtype=dtype)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dtype=dtype)  # is self-attn if context is none
        self.norm1 = LayerNorm(dim, dtype=dtype)
        self.norm2 = LayerNorm(dim, dtype=dtype)
        self.norm3 = LayerNorm(dim, dtype=dtype)

    def forward(self, hidden_states, context=None):
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states),
                                   context=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class Transformer2DModel(Module):

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        use_linear_projection: bool = False,
        dtype=None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = GroupNorm(num_groups=norm_num_groups,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True,
                              dtype=dtype)

        if use_linear_projection:
            self.proj_in = Linear(in_channels, inner_dim, dtype=dtype)
        else:
            self.proj_in = Conv2d(in_channels,
                                  inner_dim,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=(0, 0),
                                  dtype=dtype)

        self.transformer_blocks = ModuleList([
            BasicTransformerBlock(inner_dim,
                                  num_attention_heads,
                                  attention_head_dim,
                                  context_dim=cross_attention_dim,
                                  dtype=dtype) for d in range(num_layers)
        ])

        if use_linear_projection:
            self.proj_out = Linear(inner_dim, in_channels, dtype=dtype)
        else:
            self.proj_out = Conv2d(inner_dim,
                                   in_channels,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=(0, 0),
                                   dtype=dtype)

    def forward(self, hidden_states, context=None):
        assert not hidden_states.is_dynamic()
        batch, _, height, weight = hidden_states.size()
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.size()[1]
            hidden_states = hidden_states.permute([0, 2, 3, 1]).view(
                [batch, height * weight, inner_dim])
        else:
            inner_dim = hidden_states.size()[1]
            hidden_states = hidden_states.permute([0, 2, 3, 1]).view(
                [batch, height * weight, inner_dim])
            hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=context)

        if not self.use_linear_projection:
            hidden_states = hidden_states.view(
                [batch, height, weight, inner_dim]).permute([0, 3, 1, 2])
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.view(
                [batch, height, weight, inner_dim]).permute([0, 3, 1, 2])

        return hidden_states + residual
