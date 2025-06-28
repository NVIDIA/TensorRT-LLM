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
from typing import Optional

import numpy as np
import tensorrt as trt

from ..._common import default_net
from ...functional import (Tensor, allgather, bert_attention, chunk, clip,
                           concat, expand, matmul, shape, slice, softmax, stack)
from ...layers import MLP, ColumnLinear, LayerNorm, Linear, RmsNorm, RowLinear
from ...layers.activation import GELU
from ...layers.embedding import (CombinedTimestepGuidanceTextProjEmbeddings,
                                 CombinedTimestepTextProjEmbeddings,
                                 FluxPosEmbed)
from ...layers.normalization import (AdaLayerNormContinuous, AdaLayerNormZero,
                                     AdaLayerNormZeroSingle)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..model_weights_loader import ModelWeightsLoader
from ..modeling_utils import PretrainedModel
from .config import FluxConfig


class FluxAttention(Module):

    def __init__(self,
                 query_dim: int,
                 cross_attention_dim: Optional[int] = None,
                 added_kv_proj_dim: Optional[int] = None,
                 dim_head: int = 64,
                 heads: int = 8,
                 out_dim: int = None,
                 context_pre_only=None,
                 bias: bool = False,
                 qk_norm: Optional[str] = None,
                 eps: float = 1e-5,
                 pre_only=False,
                 mapping=None,
                 dtype=None):
        super().__init__()

        self.cp_size = mapping.cp_size
        self.cp_group = mapping.cp_group
        self.cp_rank = mapping.cp_rank
        self.tp_group = mapping.tp_group
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.heads = self.heads // self.tp_size
        self.dim_head = dim_head
        # default attn settings
        self.norm_factor = math.sqrt(dim_head)
        self.q_scaling = 1.0
        self.max_distance = 0

        self.added_kv_proj_dim = added_kv_proj_dim

        self.group_norm = None
        self.spatial_norm = None

        added_proj_bias = True
        out_bias = True

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = LayerNorm(dim_head, eps=eps)
            self.norm_k = LayerNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm":
            self.norm_q = RmsNorm(dim_head, eps=eps, dtype=dtype)
            self.norm_k = RmsNorm(dim_head, eps=eps, dtype=dtype)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}.")

        self.norm_cross = None
        self.dtype = dtype

        self.to_q = ColumnLinear(query_dim,
                                 self.inner_dim,
                                 bias=bias,
                                 tp_group=self.tp_group,
                                 tp_size=self.tp_size,
                                 gather_output=False,
                                 dtype=dtype)
        self.to_k = ColumnLinear(self.cross_attention_dim,
                                 self.inner_kv_dim,
                                 bias=bias,
                                 tp_group=self.tp_group,
                                 tp_size=self.tp_size,
                                 gather_output=False,
                                 dtype=dtype)
        self.to_v = ColumnLinear(self.cross_attention_dim,
                                 self.inner_kv_dim,
                                 bias=bias,
                                 tp_group=self.tp_group,
                                 tp_size=self.tp_size,
                                 gather_output=False,
                                 dtype=dtype)

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = ColumnLinear(added_kv_proj_dim,
                                           self.inner_kv_dim,
                                           bias=added_proj_bias,
                                           tp_group=self.tp_group,
                                           tp_size=self.tp_size,
                                           gather_output=False,
                                           dtype=dtype)
            self.add_v_proj = ColumnLinear(added_kv_proj_dim,
                                           self.inner_kv_dim,
                                           bias=added_proj_bias,
                                           tp_group=self.tp_group,
                                           tp_size=self.tp_size,
                                           gather_output=False,
                                           dtype=dtype)
            if self.context_pre_only is not None:
                self.add_q_proj = ColumnLinear(added_kv_proj_dim,
                                               self.inner_dim,
                                               bias=added_proj_bias,
                                               tp_group=self.tp_group,
                                               tp_size=self.tp_size,
                                               gather_output=False,
                                               dtype=dtype)

        if not self.pre_only:
            self.to_out = ModuleList([
                RowLinear(self.inner_dim,
                          self.out_dim,
                          bias=out_bias,
                          tp_group=self.tp_group,
                          tp_size=self.tp_size,
                          dtype=dtype)
            ])

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = RowLinear(self.inner_dim,
                                        self.out_dim,
                                        bias=out_bias,
                                        tp_group=self.tp_group,
                                        tp_size=self.tp_size,
                                        dtype=dtype)

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "rms_norm":
                self.norm_added_q = RmsNorm(dim_head, eps=eps, dtype=dtype)
                self.norm_added_k = RmsNorm(dim_head, eps=eps, dtype=dtype)
            else:
                raise ValueError(f"unknown qk_norm: {qk_norm}.")
        else:
            self.norm_added_q = None
            self.norm_added_k = None

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tensor] = None,
        max_input_length: Optional[Tensor] = None,
    ):
        shape(hidden_states, 0) if encoder_hidden_states is None else shape(
            encoder_hidden_states, 0)
        if attention_mask is not None:
            raise NotImplementedError()

        # `sample` projections.
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        head_dim = self.dim_head
        inner_dim = head_dim * self.heads

        def bsd_to_bsnh(x):
            # inner_dim -> heads, head_dim
            bs = shape(x, 0)
            seq_len = shape(x, 1)
            hid_dim = shape(x, 2)
            x = x.view(concat([bs, seq_len, hid_dim // head_dim, head_dim]))
            return x

        def bsnh_to_bsd(x):
            bs = shape(x, 0)
            seq_len = shape(x, 1)
            num_heads = shape(x, 2)
            x = x.view(concat([bs, seq_len, num_heads * head_dim]))
            return x

        if self.norm_q is not None:
            query = bsd_to_bsnh(query)
            query = self.norm_q(query)
            query = bsnh_to_bsd(query)
        if self.norm_k is not None:
            key = bsd_to_bsnh(key)
            key = self.norm_k(key)
            key = bsnh_to_bsd(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = self.add_q_proj(
                encoder_hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(
                encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(
                encoder_hidden_states)

            if self.norm_added_q is not None:
                encoder_hidden_states_query_proj = bsd_to_bsnh(
                    encoder_hidden_states_query_proj)
                encoder_hidden_states_query_proj = self.norm_added_q(
                    encoder_hidden_states_query_proj)
                encoder_hidden_states_query_proj = bsnh_to_bsd(
                    encoder_hidden_states_query_proj)
            if self.norm_added_k is not None:
                encoder_hidden_states_key_proj = bsd_to_bsnh(
                    encoder_hidden_states_key_proj)
                encoder_hidden_states_key_proj = self.norm_added_k(
                    encoder_hidden_states_key_proj)
                encoder_hidden_states_key_proj = bsnh_to_bsd(
                    encoder_hidden_states_key_proj)

            # attention
            query = concat([encoder_hidden_states_query_proj, query], dim=1)
            key = concat([encoder_hidden_states_key_proj, key], dim=1)
            value = concat([encoder_hidden_states_value_proj, value], dim=1)

        if image_rotary_emb is not None:

            def apply_rotary_emb(x, position_embedding, mark_prefix='query_'):
                x = bsd_to_bsnh(x).permute([0, 2, 1, 3])
                cos, sin = position_embedding
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)
                b, s, h, d = shape(x, 0), shape(x, 1), shape(x, 2), shape(x, 3)
                x_real, x_imag = x.view(concat([b, s, h, d / 2, 2])).unbind(-1)
                x_rotated = stack([0 - x_imag, x_real], dim=-1).flatten(3)
                out = (x.cast(cos.dtype) * cos +
                       x_rotated.cast(sin.dtype) * sin).cast(x.dtype)
                out = out.permute([0, 2, 1, 3])
                return out

            query = bsnh_to_bsd(
                apply_rotary_emb(query, image_rotary_emb, mark_prefix='query_'))
            key = bsnh_to_bsd(
                apply_rotary_emb(key, image_rotary_emb, mark_prefix='key_'))

        if default_net().plugin_config.bert_attention_plugin:
            # TRT plugin mode
            seq_len = shape(query, 1)
            qkv = concat([query, key, value], dim=-1)
            input_lengths = expand(
                shape(qkv, 1).unsqueeze(0),
                shape(qkv, 0).unsqueeze(0)).cast("int32")

            hidden_states = bert_attention(qkv,
                                           input_lengths,
                                           self.heads,
                                           head_dim,
                                           q_scaling=self.q_scaling,
                                           relative_attention=False,
                                           max_distance=self.max_distance,
                                           relative_attention_bias=None,
                                           max_input_length=max_input_length,
                                           cp_group=self.cp_group,
                                           cp_rank=self.cp_rank,
                                           cp_size=self.cp_size)
        else:
            # plain TRT mode
            def transpose_for_scores(x):
                new_x_shape = concat(
                    [shape(x, 0),
                     shape(x, 1), self.heads, head_dim])
                return x.view(new_x_shape).permute([0, 2, 1, 3])

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

            attention_probs = softmax(attention_scores, dim=-1)

            context = matmul(attention_probs, value,
                             use_fp32_acc=False).permute([0, 2, 1, 3])
            hidden_states = context.view(
                concat([shape(context, 0),
                        shape(context, 1), inner_dim]))

        if encoder_hidden_states is not None:
            bs, seq_len = shape(hidden_states, 0), shape(hidden_states, 1)
            enc_seq_len = shape(encoder_hidden_states, 1)
            encoder_hidden_states = slice(hidden_states,
                                          starts=[0, 0, 0],
                                          sizes=concat(
                                              [bs, enc_seq_len, inner_dim]))
            hidden_states = slice(hidden_states,
                                  starts=concat([0, enc_seq_len, 0]),
                                  sizes=concat(
                                      [bs, seq_len - enc_seq_len, inner_dim]))

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            if self.tp_size > 1:
                hidden_states = bsd_to_bsnh(hidden_states)
                # gather across 'num_heads' dim
                hidden_states = allgather(hidden_states,
                                          group=self.tp_group,
                                          gather_dim=2)
                hidden_states = bsnh_to_bsd(hidden_states)
            return hidden_states


class FluxTransformerBlock(Module):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/66eef9a6dc8a97815a69fdf97aa20c8ece63d3f6/src/diffusers/models/transformers/transformer_flux.py#L107

    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
    """

    def __init__(self,
                 dim,
                 num_attention_heads,
                 attention_head_dim,
                 qk_norm="rms_norm",
                 eps=1e-6,
                 mapping=None,
                 dtype=None):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, mapping=mapping, dtype=dtype)

        self.norm1_context = AdaLayerNormZero(dim, mapping=mapping, dtype=dtype)

        self.attn = FluxAttention(query_dim=dim,
                                  cross_attention_dim=None,
                                  added_kv_proj_dim=dim,
                                  dim_head=attention_head_dim,
                                  heads=num_attention_heads,
                                  out_dim=dim,
                                  context_pre_only=False,
                                  bias=True,
                                  qk_norm=qk_norm,
                                  eps=eps,
                                  mapping=mapping,
                                  dtype=dtype)

        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = MLP(hidden_size=dim,
                      ffn_hidden_size=4 * dim,
                      hidden_act='gelu',
                      tp_group=mapping.tp_group,
                      tp_size=mapping.tp_size,
                      dtype=dtype)

        self.norm2_context = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = MLP(hidden_size=dim,
                              ffn_hidden_size=4 * dim,
                              hidden_act='gelu',
                              tp_group=mapping.tp_group,
                              tp_size=mapping.tp_size,
                              dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb)

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (
            1 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1)

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(
            1) * context_ff_output
        if encoder_hidden_states.dtype == trt.DataType.HALF:
            encoder_hidden_states = clip(encoder_hidden_states, -65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(Module):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/66eef9a6dc8a97815a69fdf97aa20c8ece63d3f6/src/diffusers/models/transformers/transformer_flux.py#L44
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
    """

    def __init__(self,
                 dim,
                 num_attention_heads,
                 attention_head_dim,
                 mlp_ratio=4.0,
                 mapping=None,
                 dtype=None):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.cp_size = mapping.cp_size
        self.cp_group = mapping.cp_group
        self.cp_rank = mapping.cp_rank
        self.tp_group = mapping.tp_group
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank

        self.norm = AdaLayerNormZeroSingle(dim, mapping=mapping, dtype=dtype)
        self.proj_mlp = Linear(dim,
                               self.mlp_hidden_dim,
                               tp_group=self.tp_group,
                               tp_size=self.tp_size,
                               dtype=dtype)
        self.act_mlp = GELU(approximate="tanh")
        self.proj_out = Linear(dim + self.mlp_hidden_dim,
                               dim,
                               tp_group=self.tp_group,
                               tp_size=self.tp_size,
                               dtype=dtype)

        self.attn = FluxAttention(query_dim=dim,
                                  cross_attention_dim=None,
                                  dim_head=attention_head_dim,
                                  heads=num_attention_heads,
                                  out_dim=dim,
                                  bias=True,
                                  qk_norm="rms_norm",
                                  eps=1e-6,
                                  pre_only=True,
                                  mapping=mapping,
                                  dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = concat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == trt.DataType.HALF:
            hidden_states = clip(hidden_states, -65504, 65504)

        return hidden_states


class FluxTransformer2DModel(PretrainedModel):
    """
    Adapted from: https://github.com/huggingface/diffusers/blob/66eef9a6dc8a97815a69fdf97aa20c8ece63d3f6/src/diffusers/models/transformers/transformer_flux.py#L206
    """
    config_class = FluxConfig

    def __init__(self, config: FluxConfig):
        super().__init__(config)
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        self.dtype = config.dtype

        self.in_channels = config.in_channels
        self.out_channels = self.in_channels

        self.inner_dim = config.num_attention_heads * config.attention_head_dim

        # TODO: remove hard code
        axes_dims_rope = (16, 56, 56)
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (CombinedTimestepGuidanceTextProjEmbeddings
                                  if config.guidance_embeds else
                                  CombinedTimestepTextProjEmbeddings)
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=config.pooled_projection_dim,
            mapping=self.mapping,
            dtype=self.dtype)

        self.context_embedder = Linear(config.joint_attention_dim,
                                       self.inner_dim,
                                       tp_group=self.mapping.tp_group,
                                       tp_size=self.mapping.tp_size,
                                       dtype=self.dtype)
        self.x_embedder = Linear(config.in_channels,
                                 self.inner_dim,
                                 tp_group=self.mapping.tp_group,
                                 tp_size=self.mapping.tp_size,
                                 dtype=self.dtype)

        self.transformer_blocks = ModuleList([
            FluxTransformerBlock(dim=self.inner_dim,
                                 num_attention_heads=config.num_attention_heads,
                                 attention_head_dim=config.attention_head_dim,
                                 mapping=self.mapping,
                                 dtype=self.dtype)
            for i in range(config.num_layers)
        ])

        self.single_transformer_blocks = ModuleList([
            FluxSingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                mapping=self.mapping,
                dtype=self.dtype) for i in range(self.config.num_single_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(self.inner_dim,
                                               self.inner_dim,
                                               elementwise_affine=False,
                                               eps=1e-6,
                                               mapping=self.mapping,
                                               dtype=self.dtype)
        self.proj_out = Linear(self.inner_dim,
                               config.patch_size * config.patch_size *
                               self.out_channels,
                               bias=True,
                               tp_group=self.mapping.tp_group,
                               tp_size=self.mapping.tp_size,
                               dtype=self.dtype)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        pooled_projections: Tensor = None,
        timestep: Tensor = None,
        img_ids: Tensor = None,
        txt_ids: Tensor = None,
        guidance: Tensor = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
    ) -> Tensor:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`FloatTensor` of shape `(batch size, img_seq, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if self.mapping.cp_size > 1:
            hidden_states = chunk(hidden_states,
                                  chunks=self.mapping.cp_size,
                                  dim=1)[self.mapping.cp_rank]
            encoder_hidden_states = chunk(encoder_hidden_states,
                                          chunks=self.mapping.cp_size,
                                          dim=1)[self.mapping.cp_rank]

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.cast(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.cast(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (self.time_text_embed(timestep, pooled_projections)
                if guidance is None else self.time_text_embed(
                    timestep, guidance, pooled_projections))
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim() == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim() == 3:
            img_ids = img_ids[0]

        if self.mapping.cp_size > 1:
            txt_ids = chunk(txt_ids, chunks=self.mapping.cp_size,
                            dim=0)[self.mapping.cp_rank]
            img_ids = chunk(img_ids, chunks=self.mapping.cp_size,
                            dim=0)[self.mapping.cp_rank]

        ids = concat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(
                    self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                control_feature = controlnet_block_samples[index_block //
                                                           interval_control]
                if self.mapping.cp_size > 1:
                    control_feature = chunk(control_feature,
                                            chunks=self.mapping.cp_size,
                                            dim=1)[self.mapping.cp_rank]
                hidden_states = hidden_states + control_feature

        hidden_states = concat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(
                    controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                control_feature = controlnet_single_block_samples[
                    index_block // interval_control]
                if self.mapping.cp_size > 1:
                    control_feature = chunk(control_feature,
                                            chunks=self.mapping.cp_size,
                                            dim=1)[self.mapping.cp_rank]
                bs, seq_len = shape(hidden_states, 0), shape(hidden_states, 1)
                enc_seq_len = shape(encoder_hidden_states, 1)
                encoder_hidden_states = slice(
                    hidden_states,
                    starts=[0, 0, 0],
                    sizes=concat([bs, enc_seq_len, self.inner_dim]))
                hidden_states = slice(
                    hidden_states,
                    starts=concat([0, enc_seq_len, 0]),
                    sizes=concat([bs, seq_len - enc_seq_len, self.inner_dim]))
                hidden_states = hidden_states + control_feature
                hidden_states = concat([encoder_hidden_states, hidden_states],
                                       dim=1)

        bs, seq_len = shape(hidden_states, 0), shape(hidden_states, 1)
        enc_seq_len = shape(encoder_hidden_states, 1)
        hidden_states = slice(hidden_states,
                              starts=concat([0, enc_seq_len, 0]),
                              sizes=concat(
                                  [bs, seq_len - enc_seq_len, self.inner_dim]))

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        if self.mapping.cp_size > 1:
            hidden_states = allgather(hidden_states,
                                      group=self.mapping.cp_group,
                                      gather_dim=1)
        hidden_states.mark_output("output", hidden_states.dtype)

        return hidden_states

    def prepare_inputs(self, max_batch_size, **kwargs):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the ranges of the dimensions of when using TRT dynamic shapes.
           @return: a list contains values which can be fed into the self.forward()
        '''

        def flux_default_range(max_batch_size):
            return [1, max(1, (max_batch_size + 1) // 2), max_batch_size]

        default_range = flux_default_range
        # TODO: customized input h/w
        input_h, input_w = 1024, 1024
        H = input_h / 8
        W = input_w / 8
        img_seq_len = int(H * W / 4)
        txt_len = 512

        hidden_states = Tensor(name='hidden_states',
                               dtype=self.dtype,
                               shape=[-1, img_seq_len, self.config.in_channels],
                               dim_range=OrderedDict([
                                   ('batch_size',
                                    [default_range(max_batch_size)]),
                                   ('img_seq_len', [[img_seq_len] * 3]),
                                   ('in_channels',
                                    [[self.config.in_channels] * 3]),
                               ]))
        encoder_hidden_states = Tensor(
            name='encoder_hidden_states',
            dtype=self.dtype,
            shape=[-1, txt_len, 4096],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
                ('txt_len', [[txt_len] * 3]),
                ('joint_attention_dim', [[self.config.joint_attention_dim] * 3
                                         ]),
            ]))
        pooled_projections = Tensor(
            name='pooled_projections',
            dtype=self.dtype,
            shape=[-1, self.config.pooled_projection_dim],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
                ('pooled_projection_dim',
                 [[self.config.pooled_projection_dim] * 3]),
            ]))
        timestep = Tensor(name='timestep',
                          dtype=self.dtype,
                          shape=[-1],
                          dim_range=OrderedDict([
                              ('batch_size', [default_range(max_batch_size)]),
                          ]))
        img_ids = Tensor(name='img_ids',
                         dtype=self.dtype,
                         shape=[img_seq_len, 3],
                         dim_range=OrderedDict([
                             ('img_seq_len', [[img_seq_len] * 3]),
                             ('channels', [[3] * 3]),
                         ]))
        txt_ids = Tensor(name='txt_ids',
                         dtype=self.dtype,
                         shape=[txt_len, 3],
                         dim_range=OrderedDict([
                             ('txt_len', [[txt_len] * 3]),
                             ('channels', [[3] * 3]),
                         ]))
        guidance = Tensor(name='guidance',
                          dtype=trt.float32,
                          shape=[-1],
                          dim_range=OrderedDict([
                              ('batch_size', [default_range(max_batch_size)]),
                          ]))
        return {
            'hidden_states': hidden_states,
            'encoder_hidden_states': encoder_hidden_states,
            'pooled_projections': pooled_projections,
            'timestep': timestep,
            'img_ids': img_ids,
            'txt_ids': txt_ids,
            'guidance': guidance
        }

    @classmethod
    def from_hugging_face(cls,
                          hf_model_or_dir: str,
                          dtype: str = 'auto',
                          mapping: Optional[Mapping] = None,
                          **kwargs):
        ''' Create a LLaMAForCausalLM object from give parameters
        '''
        quant_ckpt_path = kwargs.pop('quant_ckpt_path', None)

        hf_model_dir = hf_model_or_dir
        hf_config_or_dir = hf_model_or_dir

        config = FluxConfig.from_hugging_face(hf_config_or_dir,
                                              dtype=dtype,
                                              mapping=mapping,
                                              **kwargs)

        custom_dict = {}
        if quant_ckpt_path is not None:
            hf_model_dir = quant_ckpt_path

        loader = FluxModelWeightsLoader(hf_model_dir, custom_dict)
        model = cls(config)
        loader.generate_tllm_weights(model)
        return model


class FluxModelWeightsLoader(ModelWeightsLoader):

    def translate_to_external_key(self, tllm_key: str,
                                  tllm_to_externel_key_dict: dict):
        """Translate TRT-LLM key into HF key or HF key list (e.g. QKV/MoE/GPTQ)

        Base class mapping methods:
        tllm_key : "transformer.layers.0.attention.  qkv .weight"
                          |        |   |     |        |     |
        translated: ["  model  .layers.0.self_attn.q_proj.weight,
                     "  model  .layers.0.self_attn.k_proj.weight,
                     "  model  .layers.0.self_attn.v_proj.weight]

        However, Flux's HF model and TRT-LLM model don't have same hierarchical structure.
        E.g.,
        trtllm_key: "transformer_blocks.0.ff.fc.weight"
        HF_key:     "transformer_blocks.0.ff.net.0.proj.weight"

        So we rewrite ModelWeightsLoader here.

        Args:
            tllm_key (str): Input TRT-LLM key.
            tllm_to_externel_key_dict (dict): User specified dict with higher priority. \
            Generated from layer attributes automatically.

        Returns:
            hf_keys (str | list[str]) : Translated HF key(s).
        """
        trtllm_to_hf_name = {
            'transformer_blocks.(\d+).(\w+).fc.weight':
            'transformer_blocks.*.*.net.0.proj.weight',
            'transformer_blocks.(\d+).(\w+).fc.bias':
            'transformer_blocks.*.*.net.0.proj.bias',
            'transformer_blocks.(\d+).(\w+).proj.weight':
            'transformer_blocks.*.*.net.2.weight',
            'transformer_blocks.(\d+).(\w+).proj.bias':
            'transformer_blocks.*.*.net.2.bias',
        }
        import re
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
