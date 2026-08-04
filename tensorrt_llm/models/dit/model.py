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
from collections import OrderedDict

import numpy as np
import tensorrt as trt

from ..._utils import str_dtype_to_trt, trt_dtype_to_str
from ...functional import (Tensor, allgather, arange, chunk, concat, constant,
                           cos, exp, expand, shape, silu, sin, slice, split,
                           unsqueeze)
from ...layers import MLP, BertAttention, Conv2d, Embedding, LayerNorm, Linear
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...plugin import current_all_reduce_helper
from ...quantization import QuantMode
from ..modeling_utils import PretrainedConfig, PretrainedModel


def modulate(x, shift, scale, dtype):
    ones = 1.0
    if dtype is not None:
        ones = constant(np.ones(1, dtype=np.float32)).cast(dtype)
    return x * (ones + unsqueeze(scale, 1)) + unsqueeze(shift, 1)


class TimestepEmbedder(Module):

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.mlp1 = Linear(frequency_embedding_size,
                           hidden_size,
                           bias=True,
                           dtype=dtype)
        self.mlp2 = Linear(hidden_size, hidden_size, bias=True, dtype=dtype)
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = exp(
            -math.log(max_period) *
            arange(start=0, end=half, dtype=trt_dtype_to_str(trt.float32)) /
            constant(np.array([half], dtype=np.float32)))
        args = unsqueeze(t, -1).cast(trt.float32) * unsqueeze(freqs, 0)
        embedding = concat([cos(args), sin(args)], dim=-1)
        if self.dtype is not None: embedding = embedding.cast(self.dtype)
        assert dim % 2 == 0
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp2(silu(self.mlp1(t_freq)))

        return t_emb


class LabelEmbedder(Module):

    def __init__(self, num_classes, hidden_size, dropout_prob, dtype=None):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = Embedding(num_classes + use_cfg_embedding,
                                         hidden_size,
                                         dtype=dtype)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, force_drop_ids=None):
        assert force_drop_ids is None
        embeddings = self.embedding_table(labels)
        return embeddings


class PatchEmbed(Module):

    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 input_c: int,
                 output_c: int,
                 bias: bool = True,
                 dtype: trt.DataType = None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size)**2
        self.proj = Conv2d(input_c,
                           output_c,
                           kernel_size=(patch_size, patch_size),
                           stride=(patch_size, patch_size),
                           bias=bias,
                           dtype=dtype)

    def forward(self, x):
        assert x.shape[2] == self.img_size
        assert x.shape[3] == self.img_size
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x


class DiTBlock(Module):

    def __init__(self,
                 hidden_size,
                 num_heads,
                 mapping=Mapping(),
                 mlp_ratio=4.0,
                 dtype=None,
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.dtype = dtype
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = BertAttention(hidden_size,
                                  num_heads,
                                  tp_group=mapping.tp_group,
                                  tp_size=mapping.tp_size,
                                  tp_rank=mapping.tp_rank,
                                  cp_group=mapping.cp_group,
                                  cp_size=mapping.cp_size,
                                  cp_rank=mapping.cp_rank,
                                  dtype=dtype,
                                  quant_mode=quant_mode)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=int(hidden_size * mlp_ratio),
                       hidden_act='gelu',
                       tp_group=mapping.tp_group,
                       tp_size=mapping.tp_size,
                       dtype=dtype,
                       quant_mode=quant_mode)
        self.adaLN_modulation = Linear(hidden_size,
                                       6 * hidden_size,
                                       tp_group=mapping.tp_group,
                                       tp_size=mapping.tp_size,
                                       bias=True,
                                       dtype=dtype)

    def forward(self, x, c, input_lengths):
        c = self.adaLN_modulation(silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(
            c, 6, dim=1)

        x = x + unsqueeze(gate_msa, 1) * self.attn(modulate(
            self.norm1(x), shift_msa, scale_msa, self.dtype),
                                                   input_lengths=input_lengths)
        x = x + unsqueeze(gate_mlp, 1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp, self.dtype))
        return x


class FinalLayer(Module):

    def __init__(self,
                 hidden_size,
                 patch_size,
                 out_channels,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        self.dtype = dtype
        self.norm_final = LayerNorm(hidden_size,
                                    elementwise_affine=False,
                                    eps=1e-6)
        self.linear = Linear(hidden_size,
                             patch_size * patch_size * out_channels,
                             bias=True,
                             dtype=dtype)
        self.adaLN_modulation = Linear(hidden_size,
                                       2 * hidden_size,
                                       tp_group=mapping.tp_group,
                                       tp_size=mapping.tp_size,
                                       bias=True,
                                       dtype=dtype)

    def forward(self, x, c):
        shift, scale = chunk(self.adaLN_modulation(silu(c)), 2, dim=1)

        x = modulate(self.norm_final(x), shift, scale, self.dtype)
        x = self.linear(x)

        return x


class DiT(PretrainedModel):

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        super().__init__(config)
        self.learn_sigma = config.learn_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.learn_sigma else config.in_channels
        self.input_size = config.input_size
        self.patch_size = config.patch_size
        self.num_heads = config.num_attention_heads
        self.dtype = str_dtype_to_trt(config.dtype)
        self.cfg_scale = config.cfg_scale
        self.mapping = config.mapping

        self.x_embedder = PatchEmbed(config.input_size,
                                     config.patch_size,
                                     config.in_channels,
                                     config.hidden_size,
                                     bias=True,
                                     dtype=self.dtype)
        self.t_embedder = TimestepEmbedder(config.hidden_size, dtype=self.dtype)
        self.y_embedder = LabelEmbedder(config.num_classes,
                                        config.hidden_size,
                                        config.class_dropout_prob,
                                        dtype=self.dtype)
        num_patches = self.x_embedder.num_patches

        self.pos_embed = Parameter(shape=(1, num_patches, config.hidden_size),
                                   dtype=self.dtype)
        self.blocks = ModuleList([
            DiTBlock(config.hidden_size,
                     config.num_attention_heads,
                     mlp_ratio=config.mlp_ratio,
                     mapping=config.mapping,
                     dtype=self.dtype,
                     quant_mode=config.quant_mode)
            for _ in range(config.num_hidden_layers)
        ])
        self.final_layer = FinalLayer(config.hidden_size,
                                      config.patch_size,
                                      self.out_channels,
                                      mapping=config.mapping,
                                      dtype=self.dtype)

    # We need to invoke default `__post_init__()` for quantized layers.
    #  def __post_init__(self):
    #     return

    def check_config(self, config: PretrainedConfig):
        config.set_if_not_exist('input_size', 32)
        config.set_if_not_exist('patch_size', 2)
        config.set_if_not_exist('in_channels', 4)
        config.set_if_not_exist('mlp_ratio', 4.0)
        config.set_if_not_exist('class_dropout_prob', 0.1)
        config.set_if_not_exist('num_classes', 1000)
        config.set_if_not_exist('learn_sigma', True)
        config.set_if_not_exist('dtype', None)
        config.set_if_not_exist('cfg_scale', None)

    def unpatchify(self, x: Tensor):
        c = self.out_channels
        p = self.x_embedder.patch_size
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]

        x = x.view(shape=(x.shape[0], h, w, p, p, c))
        x = x.permute((0, 5, 1, 3, 2, 4))
        imgs = x.view(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, latent, timestep, label):
        """
        Forward pass of DiT.
        latent: (N, C, H, W)
        timestep: (N,)
        label: (N,)
        """
        if self.cfg_scale is not None:
            output = self.forward_with_cfg(latent, timestep, label)
        else:
            output = self.forward_without_cfg(latent, timestep, label)
        output.mark_output('output', self.dtype)
        return output

    def forward_without_cfg(self, x, t, y):
        """
        Forward pass without classifier-free guidance.
        """
        x = self.x_embedder(x) + self.pos_embed.value
        t = self.t_embedder(t)
        y = self.y_embedder(y)
        self.register_network_output('t_embedder', t)
        self.register_network_output('x_embedder', x)
        self.register_network_output('y_embedder', y)
        c = t + y
        input_length = constant(np.array([x.shape[1]], dtype=np.int32))
        input_lengths = expand(input_length, unsqueeze(shape(x, 0), 0))
        # Split squeence for CP here
        if self.mapping.cp_size > 1:
            assert x.shape[1] % self.mapping.cp_size == 0
            x = chunk(x, self.mapping.cp_size, dim=1)[self.mapping.cp_rank]
            input_lengths = input_lengths // self.mapping.cp_size
        for block in self.blocks:
            x = block(x, c, input_lengths)  # (N, T, D)
        self.register_network_output('before_final_layer', x)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        self.register_network_output('final_layer', x)

        # All gather after CP
        if self.mapping.cp_size > 1:
            x = allgather(x, self.mapping.cp_group, gather_dim=1)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        self.register_network_output('unpatchify', x)
        return x

    def forward_with_cfg(self, x, t, y):
        """
        Forward pass with classifier-free guidance.
        """
        batch_size = shape(x, 0)
        half = slice(
            x, [0, 0, 0, 0],
            concat([batch_size / 2, x.shape[1], x.shape[2], x.shape[3]]))
        combined = concat([half, half], dim=0)
        self.register_network_output('combined', combined)
        model_out = self.forward_without_cfg(combined, t, y)

        _, d, h, w = model_out.shape
        eps, rest = split(model_out, [3, d - 3], dim=1)
        cond_eps = slice(eps, [0, 0, 0, 0], concat([batch_size / 2, 3, h, w]))
        uncond_eps = slice(eps, concat([batch_size / 2, 0, 0, 0]),
                           concat([batch_size / 2, 3, h, w]))
        self.register_network_output('cond_eps', cond_eps)
        self.register_network_output('uncond_eps', uncond_eps)

        half_eps = uncond_eps + self.cfg_scale * (cond_eps - uncond_eps)
        eps = concat([half_eps, half_eps], dim=0)
        self.register_network_output('eps', eps)

        return concat([eps, rest], dim=1)

    def prepare_inputs(self, max_batch_size, **kwargs):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''
        mapping = self.config.mapping
        if mapping.tp_size > 1:
            current_all_reduce_helper().set_workspace_tensor(mapping, 1)

        def dit_default_range(max_batch_size):
            return [2, max(2, (max_batch_size + 1) // 2), max_batch_size]

        default_range = dit_default_range
        if self.cfg_scale is not None:
            max_batch_size *= 2

        latent = Tensor(
            name='latent',
            dtype=self.dtype,
            shape=[-1, self.in_channels, self.input_size, self.input_size],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
                ('in_channels', [[self.in_channels] * 3]),
                ('latent_height', [[self.input_size] * 3]),
                ('latent_width', [[self.input_size] * 3]),
            ]))
        timestep = Tensor(name='timestep',
                          dtype=trt.int32,
                          shape=[-1],
                          dim_range=OrderedDict([
                              ('batch_size', [default_range(max_batch_size)]),
                          ]))
        label = Tensor(name='label',
                       dtype=trt.int32,
                       shape=[-1],
                       dim_range=OrderedDict([
                           ('batch_size', [default_range(max_batch_size)]),
                       ]))
        return {'latent': latent, 'timestep': timestep, 'label': label}
