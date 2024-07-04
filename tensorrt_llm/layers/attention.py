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
from typing import List, Optional

import numpy as np
import tensorrt as trt

from .._common import default_net, precision
from .._utils import (fp32_array, int32_array, is_same_dtype, trt_dtype_to_np,
                      trt_dtype_to_str, trt_gte_10)
from ..functional import (ACT2FN, AllReduceFusionParams, AttentionMaskType,
                          Conditional, PositionEmbeddingType,
                          RopeEmbeddingUtils, RotaryScalingType, Tensor, arange,
                          bert_attention, cast, clip, concat, constant,
                          embedding, expand, expand_dims, expand_mask,
                          generate_alibi_biases, generate_alibi_slopes,
                          gpt_attention, matmul)
from ..functional import max as fmax
from ..functional import (minimum, repeat_interleave, shape, slice, softmax,
                          split, unsqueeze, where)
from ..module import Module
from ..parameter import Parameter
from ..quantization import QuantMode
from ..quantization.functional import dequantize, quantize
from .linear import ColumnLinear, QKVColumnLinear, RowLinear
from .lora import LoraRuntimeParams
from .normalization import LayerNorm

from ..functional import maximum  # isort:skip


def make_causal_mask(bsz, tgt_len, past_key_values_length, dtype):
    _range = arange(start=constant(int32_array(0)),
                    end=tgt_len,
                    dtype=trt_dtype_to_str(dtype))
    mask = repeat_interleave(_range, tgt_len, 0).view(concat([tgt_len,
                                                              tgt_len]))
    mask = where(mask < mask.transpose(-1, -2), 1.0, 0.0)

    zero = constant(fp32_array(0))
    zero = expand_dims(zero, [0, 1])
    zero = expand(zero, concat([tgt_len, past_key_values_length]))
    mask = concat([zero, mask], dim=1)
    mask *= np.finfo(trt_dtype_to_np(dtype)).min.item()
    mask = mask.view(concat([1, 1, tgt_len, tgt_len + past_key_values_length]))
    mask = expand(mask,
                  concat([bsz, 1, tgt_len, tgt_len + past_key_values_length]))
    return mask


def compute_relative_bias(query_length,
                          key_length,
                          num_buckets,
                          max_distance,
                          bidirectional,
                          rel_attn_table,
                          tp_size=1,
                          tp_group=None,
                          tp_rank=None):

    def make_relative_position_bucket(relative_position, bidirectional,
                                      num_buckets, max_distance):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += where(relative_position > 0, num_buckets, 0)
            relative_position = relative_position.abs()
        else:
            relative_position = 0 - minimum(relative_position, 0)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        max_exact_fp = constant(fp32_array(max_exact))
        tmp = cast(relative_position, "float32") / max_exact_fp
        tmp = tmp.log()
        const1 = math.log(max_distance / max_exact)
        const2 = constant(fp32_array(num_buckets - max_exact))
        relative_position_if_large = tmp / const1 * const2
        relative_position_if_large = cast(relative_position_if_large, "int32")
        relative_position_if_large = max_exact + relative_position_if_large
        relative_position_if_large = minimum(relative_position_if_large,
                                             num_buckets - 1)

        relative_buckets += where(is_small, relative_position,
                                  relative_position_if_large)
        return relative_buckets

    context_position = arange(start=constant(int32_array(0)),
                              end=query_length,
                              dtype=trt_dtype_to_str(trt.int32))
    context_position = unsqueeze(context_position, -1)
    memory_position = arange(start=constant(int32_array(0)),
                             end=key_length,
                             dtype=trt_dtype_to_str(trt.int32))
    memory_position = unsqueeze(memory_position, 0)
    relative_position = memory_position - context_position
    relative_position_bucket = make_relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional,
        num_buckets,
        max_distance,
    )
    # shape (query_length, key_length, num_heads)
    values = embedding(relative_position_bucket,
                       rel_attn_table,
                       tp_size=tp_size,
                       tp_group=tp_group,
                       tp_rank=tp_rank)
    # shape (1, num_heads, query_length, key_length)
    values = unsqueeze(values.permute([2, 0, 1]), 0)
    return values


class AttentionParams(object):

    def __init__(self,
                 sequence_length: Tensor = None,
                 context_lengths: Tensor = None,
                 host_context_lengths: Tensor = None,
                 max_context_length: int = None,
                 host_request_types: Tensor = None,
                 encoder_input_lengths: Tensor = None,
                 encoder_max_input_length: Tensor = None):
        self.sequence_length = sequence_length
        self.context_lengths = context_lengths
        self.host_context_lengths = host_context_lengths
        # max allowed context length. Required to
        # compute scratch memory size.
        self.max_context_length = max_context_length
        self.host_request_types = host_request_types

        self.encoder_input_lengths = encoder_input_lengths
        self.encoder_max_input_length = encoder_max_input_length

    def is_valid_cross_attn(self, do_cross_attention):
        if do_cross_attention:
            if self.encoder_input_lengths is None:
                return False
            if self.encoder_max_input_length is None:
                return False
        return True

    def is_valid(self, gpt_attention_plugin, remove_input_padding):
        if gpt_attention_plugin:
            if self.sequence_length is None:
                return False
            if self.context_lengths is None:
                return False
            if self.host_request_types is None:
                return False
            if self.max_context_length is None:
                return False

        if remove_input_padding:
            if self.host_context_lengths is None:
                return False
            if not gpt_attention_plugin:
                return False

        return True


class SpecDecodingParams:

    def __init__(self,
                 spec_decoding_generation_lengths: Tensor = None,
                 spec_decoding_position_offsets: Tensor = None,
                 spec_decoding_packed_mask: Tensor = None):

        self.spec_decoding_generation_lengths = spec_decoding_generation_lengths
        self.spec_decoding_position_offsets = spec_decoding_position_offsets
        self.spec_decoding_packed_mask = spec_decoding_packed_mask


class KeyValueCacheParams:

    def __init__(self,
                 past_key_value: List[Tensor] = None,
                 host_past_key_value_lengths: Tensor = None,
                 host_max_attention_window_sizes: Tensor = None,
                 host_sink_token_length: Tensor = None,
                 kv_cache_block_offsets: Tensor = None,
                 host_kv_cache_block_offsets: Tensor = None,
                 host_kv_cache_pool_pointers: Tensor = None,
                 cache_indirection: Tensor = None,
                 past_key_value_length: Tensor = None,
                 cross_kv_cache_block_offsets: Tensor = None,
                 host_cross_kv_cache_block_offsets: Tensor = None,
                 host_cross_kv_cache_pool_pointers: Tensor = None):
        self.past_key_value = past_key_value
        self.host_past_key_value_lengths = host_past_key_value_lengths
        self.host_max_attention_window_sizes = host_max_attention_window_sizes
        self.host_sink_token_length = host_sink_token_length
        self.kv_cache_block_offsets = kv_cache_block_offsets
        self.host_kv_cache_block_offsets = host_kv_cache_block_offsets
        self.host_kv_cache_pool_pointers = host_kv_cache_pool_pointers
        self.cross_kv_cache_block_offsets = cross_kv_cache_block_offsets
        self.host_cross_kv_cache_block_offsets = host_cross_kv_cache_block_offsets
        self.host_cross_kv_cache_pool_pointers = host_cross_kv_cache_pool_pointers
        self.cache_indirection = cache_indirection
        # self.past_key_value_length = past_key_value_length

    def get_first_past_key_value(self):
        if self.past_key_value is None:
            return None
        return self.past_key_value[0]

    def fill_none_tensor_list(self, list_size):
        if self.past_key_value is None:
            self.past_key_value = tuple([None] * list_size)

    def is_valid(self, gpt_attention_plugin):
        if gpt_attention_plugin:
            if self.host_past_key_value_lengths is None:
                return False
            if self.host_max_attention_window_sizes is None:
                return False
            if self.host_sink_token_length is None:
                return False
            if self.cache_indirection is None:
                return False

        return True


class BlockSparseAttnParams:

    def __init__(self,
                 block_size: int = 64,
                 homo_head_pattern: bool = False,
                 num_local_blocks: int = 16,
                 vertical_stride: int = 8):
        self.block_size = block_size
        self.homo_head_pattern = homo_head_pattern
        self.num_local_blocks = num_local_blocks
        self.vertical_stride = vertical_stride


class Attention(Module):

    def __init__(self,
                 *,
                 local_layer_idx,
                 hidden_size,
                 num_attention_heads,
                 num_kv_heads=None,
                 max_position_embeddings=1024,
                 num_layers=1,
                 apply_query_key_layer_scaling=False,
                 attention_head_size=None,
                 qk_layernorm=False,
                 inner_layernorm=False,
                 eps=1e-05,
                 attention_mask_type=AttentionMaskType.padding,
                 bias=True,
                 dtype=None,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 rotary_embedding_base=10000.0,
                 rotary_embedding_scaling=None,
                 rotary_embedding_percentage=1.0,
                 rope_scaling_short_factors=None,
                 rope_scaling_long_factors=None,
                 rope_scaling_short_mscale=None,
                 rope_scaling_long_mscale=None,
                 original_max_position_embeddings=1024,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0,
                 quant_mode: QuantMode = QuantMode(0),
                 q_scaling=1.0,
                 cross_attention=False,
                 relative_attention=False,
                 max_distance=0,
                 num_buckets=0,
                 dense_bias=None,
                 clip_qkv=None,
                 alibi_bias_max=8,
                 skip_cross_qkv=False,
                 max_attn_value=0.0,
                 block_sparse_params=None,
                 use_implicit_relative_attention=False):
        super().__init__()

        self.local_layer_idx = local_layer_idx
        self.cross_attention = cross_attention
        self.attention_mask_type = attention_mask_type
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.num_kv_heads = num_kv_heads
        assert num_attention_heads % tp_size == 0, \
        "num_attention_heads must be divisible by tp_size"
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = (
            num_kv_heads + tp_size - 1
        ) // tp_size if num_kv_heads is not None else self.num_attention_heads
        self.hidden_size = hidden_size
        self.attention_hidden_size = self.attention_head_size * self.num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.bias = bias
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype
        self.dense_bias = dense_bias
        if dense_bias is None:
            self.dense_bias = bias

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = q_scaling
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers
        # Whether to scale ALiBi bias. Mathematically, it's equivalent to
        # normalizing QK after adding bias.
        #   - False, inv_sqrt_Dh * Q*K^T + alibi_bias
        #   - True,  inv_sqrt_Dh * Q*K^T + inv_sqrt_Dh * alibi_bias
        self.scale_alibi_bias = position_embedding_type == PositionEmbeddingType.alibi_with_scale
        self.alibi_bias_max = alibi_bias_max
        self.position_embedding_type = position_embedding_type

        self.relative_attention = relative_attention
        self.max_distance = max_distance
        self.num_buckets = num_buckets
        self.rotary_embedding_base = rotary_embedding_base
        self.rotary_embedding_scaling = rotary_embedding_scaling
        self.rotary_embedding_scale_type = RotaryScalingType.none
        self.rotary_embedding_scale = 1.0
        self.rotary_embedding_percentage = rotary_embedding_percentage
        self.use_implicit_relative_attention = self.relative_attention and use_implicit_relative_attention
        if rotary_embedding_scaling is not None:
            assert rotary_embedding_scaling["type"] in ["linear", "dynamic"]
            self.rotary_embedding_scale_type = RotaryScalingType.linear if rotary_embedding_scaling[
                "type"] == "linear" else RotaryScalingType.dynamic
            self.rotary_embedding_scale = rotary_embedding_scaling["factor"]

        self.rotary_embedding_dim = 0
        if self.position_embedding_type.is_rope():
            self.rotary_embedding_dim = int(self.attention_head_size *
                                            rotary_embedding_percentage)

            if self.position_embedding_type == PositionEmbeddingType.long_rope:
                embed_positions_short_factors, embed_positions_long_factors, \
                embed_positions_short_factors_for_attention_plugin, \
                embed_positions_long_factors_for_attention_plugin, mscale \
                    = RopeEmbeddingUtils.create_sinusoidal_positions_long_rope(
                    self.max_position_embeddings,
                    original_max_position_embeddings, self.rotary_embedding_dim,
                    self.rotary_embedding_base, rope_scaling_short_factors,
                    rope_scaling_long_factors, rope_scaling_short_mscale, rope_scaling_long_mscale)

                if rope_scaling_short_mscale is not None:
                    assert rope_scaling_long_mscale is not None
                    short_mscale = rope_scaling_short_mscale
                    long_mscale = rope_scaling_long_mscale
                else:
                    short_mscale = long_mscale = mscale

                rope_scaling_short_factors = np.array(
                    rope_scaling_short_factors).reshape(1, -1)
                rope_scaling_long_factors = np.array(
                    rope_scaling_long_factors).reshape(1, -1)

                self.register_parameter(
                    'embed_positions_short_factors',
                    Parameter(embed_positions_short_factors,
                              dtype='float32',
                              is_buffer=True))
                self.register_parameter(
                    'embed_positions_long_factors',
                    Parameter(embed_positions_long_factors,
                              dtype='float32',
                              is_buffer=True))
                self.register_parameter(
                    'embed_positions_short_factors_for_attention_plugin',
                    Parameter(
                        embed_positions_short_factors_for_attention_plugin,
                        dtype='float32',
                        is_buffer=True))
                self.register_parameter(
                    'embed_positions_long_factors_for_attention_plugin',
                    Parameter(embed_positions_long_factors_for_attention_plugin,
                              dtype='float32',
                              is_buffer=True))
                self.short_mscale = short_mscale
                self.long_mscale = long_mscale
                self.register_parameter(
                    'rope_scaling_short_factors',
                    Parameter(rope_scaling_short_factors,
                              dtype='float32',
                              is_buffer=True))
                self.register_parameter(
                    'rope_scaling_long_factors',
                    Parameter(rope_scaling_long_factors,
                              dtype='float32',
                              is_buffer=True))
            else:
                # Rotary cos/sin cache.
                embed_positions = RopeEmbeddingUtils.create_sinusoidal_positions(
                    self.max_position_embeddings,
                    self.rotary_embedding_dim,
                )
                self.register_parameter(
                    'embed_positions',
                    Parameter(embed_positions, dtype='float32', is_buffer=True))
                embed_positions_for_gpt_attention = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                    self.max_position_embeddings, self.rotary_embedding_dim,
                    self.rotary_embedding_base, self.rotary_embedding_scale,
                    self.rotary_embedding_scale_type)
                self.register_parameter(
                    'embed_positions_for_gpt_attention',
                    Parameter(embed_positions_for_gpt_attention,
                              dtype='float32',
                              is_buffer=True))

        elif self.position_embedding_type.is_alibi():
            alibi_scale = 1. / self.norm_factor if self.scale_alibi_bias else 1.
            alibi_slopes = generate_alibi_slopes(
                self.num_attention_heads * self.tp_size,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                alibi_scale=alibi_scale,
                alibi_bias_max=self.alibi_bias_max)
            self.register_parameter(
                'alibi_slopes',
                Parameter(alibi_slopes, dtype='float32', is_buffer=True))

        self.quant_mode = quant_mode
        self.max_attn_value = max_attn_value
        self.register_parameter('kv_cache_scaling_factor', None)
        self.register_parameter('attention_output_orig_quant_scale', None)

        self.block_sparse_params = block_sparse_params if block_sparse_params is not None else BlockSparseAttnParams(
        )

        # The output feature size is therefore (h/tp + 2*kvh/tp) * d, where h is num_heads,
        # d is head_size, kvh is the num_kv_heads and tp is tensor_parallel_size.
        # In ColumnLinear op, the output dim is calculated by (h + 2*kvh) * d / tp,
        # which matches the desired output size (h/tp + 2*kvh/tp) * d after splitting

        # out dim is not necessarily hidden_size + kv specific size (in MQA/GQA), but num_heads * heads_size
        # example: d_model != num_heads * head_size in Flan-T5/ByT5/Gemma
        self.qkv = QKVColumnLinear(
            hidden_size,
            tp_size * self.num_attention_heads * self.attention_head_size +
            (2 * tp_size * self.num_attention_kv_heads *
             self.attention_head_size),
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=False)
        self.dense = RowLinear(tp_size * self.num_attention_heads *
                               self.attention_head_size,
                               hidden_size,
                               bias=self.dense_bias,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

        # see optimize_model's add_lora for LoRA initialization
        self.qkv_lora = None

        # per-layer relative attention table
        if self.use_implicit_relative_attention:
            self.rel_attn_table = Parameter(shape=(num_attention_heads //
                                                   tp_size, num_buckets),
                                            dtype=dtype)
        self.qk_layernorm = qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = LayerNorm(self.attention_head_size, dtype=dtype)
            self.k_layernorm = LayerNorm(self.attention_head_size, dtype=dtype)
        self.inner_layernorm = LayerNorm(self.hidden_size, dtype=dtype,
                                         eps=eps) if inner_layernorm else None
        if clip_qkv is not None:
            self.clip_qkv = fp32_array([clip_qkv])
        else:
            self.clip_qkv = None

        self.skip_cross_qkv = skip_cross_qkv

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None,
                encoder_output: Optional[Tensor] = None,
                position_embedding=None,
                norm_before_bmm1=False,
                lora_layer_params=None,
                cross_kv_cache_gen: Optional[Tensor] = None,
                cross_qkv_reuse: Optional[Tensor] = None,
                reduce_fusion_params: Optional[AllReduceFusionParams] = None):

        assert isinstance(hidden_states, Tensor)

        spec_decoding_params = SpecDecodingParams(
        ) if spec_decoding_params is None else spec_decoding_params

        alibi_slopes = None
        if self.position_embedding_type.is_alibi():
            alibi_slopes = self.alibi_slopes.value
            if default_net().plugin_config.gpt_attention_plugin:
                alibi_slopes = cast(alibi_slopes, hidden_states.dtype)

        qkv_lora_params = None
        if lora_layer_params is not None:
            if not self.cross_attention:
                qkv_lora_params = lora_layer_params.get_runtime_params(
                    0, "attn_qkv")
            else:
                qkv_lora_params = lora_layer_params.get_runtime_params(
                    0, "cross_attn_qkv")

        unfuse_qkv_gemm = self.qkv is None
        if unfuse_qkv_gemm:
            qkv_gemm = [self.q, self.k, self.v]
            qkv = [gemm(hidden_states) for gemm in qkv_gemm]
            if default_net(
            ).plugin_config.lora_plugin and qkv_lora_params is not None:
                lora = self.qkv.lora(hidden_states, qkv_lora_params)
                kv_size = self.attention_head_size * self.num_attention_kv_heads
                qkv_lora = split(lora,
                                 [self.attention_hidden_size, kv_size, kv_size],
                                 dim=1)
                qkv = [tensor + lora for tensor, lora in zip(qkv, qkv_lora)]
        else:
            qkv = self.qkv(hidden_states, qkv_lora_params)

        if self.clip_qkv is not None:
            qkv = clip(qkv, -self.clip_qkv, self.clip_qkv)

        if default_net().plugin_config.remove_input_padding:
            if unfuse_qkv_gemm:
                for tensor in qkv:
                    assert tensor.ndim() == 2
            else:
                assert qkv.ndim() == 2

        if default_net(
        ).plugin_config.lora_plugin and qkv_lora_params is None and lora_layer_params is not None:
            if not self.cross_attention:
                q_lora_params = lora_layer_params.get_runtime_params(
                    0, "attn_q")
                k_lora_params = lora_layer_params.get_runtime_params(
                    0, "attn_k")
                v_lora_params = lora_layer_params.get_runtime_params(
                    0, "attn_v")
            else:
                q_lora_params = lora_layer_params.get_runtime_params(
                    0, "cross_attn_q")
                k_lora_params = lora_layer_params.get_runtime_params(
                    0, "cross_attn_k")
                v_lora_params = lora_layer_params.get_runtime_params(
                    0, "cross_attn_v")

            assert (q_lora_params is not None and k_lora_params is not None and v_lora_params is not None) or \
                (q_lora_params is None and k_lora_params is None and v_lora_params is None), "q_lora_params, k_lora_params and v_lora_params should be all enabled or all disabled at the same time."

            if q_lora_params is not None and k_lora_params is not None and v_lora_params is not None:
                qkv_lora_runtime_params = LoraRuntimeParams(
                    lora_ranks=[
                        q_lora_params.lora_ranks[0],
                        k_lora_params.lora_ranks[0],
                        v_lora_params.lora_ranks[0],
                    ],
                    lora_weights_pointers=[
                        q_lora_params.lora_weights_pointers[0],
                        k_lora_params.lora_weights_pointers[0],
                        v_lora_params.lora_weights_pointers[0],
                    ],
                    host_request_types=q_lora_params.host_request_types,
                    host_context_lengths=q_lora_params.host_context_lengths,
                    max_context_length=q_lora_params.max_context_length,
                    max_encoder_context_length=q_lora_params.
                    max_encoder_context_length,
                    host_encoder_input_lengths=q_lora_params.
                    host_encoder_input_lengths,
                )

                q_lora, k_lora, v_lora = self.qkv_lora(hidden_states,
                                                       qkv_lora_runtime_params)
                qkv_lora = concat([q_lora, k_lora, v_lora],
                                  dim=q_lora.rank() - 1)
                qkv = qkv + qkv_lora
        if self.qk_layernorm:
            base_shape = shape(qkv, 0) if qkv.ndim() == 2 else concat(
                [shape(qkv, 0), shape(qkv, 1)])
            # here we assume that q, k and v have the same number of attention heads
            # TODO: allow different number of attention heads for q, k and v.
            qkv = qkv.view(
                concat([
                    base_shape, self.num_attention_heads, 3,
                    self.attention_head_size
                ]))

            query, key, value = split(qkv, 1, dim=qkv.ndim() - 2)
            q_shape = concat([
                base_shape, self.num_attention_heads, self.attention_head_size
            ])
            query = query.view(q_shape)
            key = key.view(q_shape)
            value = value.view(q_shape)

            query = self.q_layernorm(query)
            key = self.k_layernorm(key)
            qkv = concat([query, key, value], dim=query.ndim() - 2)
            qkv = qkv.view(concat([base_shape, self.attention_hidden_size * 3]))
        if self.position_embedding_type == PositionEmbeddingType.chatglm:
            qkv = RopeEmbeddingUtils.apply_rotary_pos_emb_chatglm(
                qkv,
                position_embedding,
                self.num_attention_heads,
                self.attention_head_size,
                self.max_position_embeddings,
                self.rotary_embedding_scale,
                default_net().plugin_config.remove_input_padding,
            )
            self.rotary_embedding_scale_type = RotaryScalingType.none
            self.rotary_embedding_scale = 1.0

        paged_kv_cache = default_net().plugin_config.paged_kv_cache

        assert attention_params is None or attention_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin,
            default_net().plugin_config.remove_input_padding)
        assert kv_cache_params is None or kv_cache_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin)

        past_key_value = None if kv_cache_params is None else kv_cache_params.get_first_past_key_value(
        )

        # if cross attention, cross QKV only needs to be calculated once in the
        # 1st decoding step --> write to cross KV cache --> remains constant
        # during the entire decoding steps.
        # 1st and >1st steps are distinguished by a boolean tensor `cross_kv_cache_gen` passed at runtime
        # also, cross KV cache max length is set from encoder output seqlen,
        # this maps to the max context length concept in decoder-only models
        cross_qkv = None
        if self.cross_attention and encoder_output:
            assert isinstance(encoder_output, Tensor)

            def compute_cross_qkv(encoder_output):
                cross_qkv = self.qkv(encoder_output, qkv_lora_params)

                if default_net(
                ).plugin_config.lora_plugin and qkv_lora_params is None and lora_layer_params is not None:
                    cross_q_lora, cross_k_lora, cross_v_lora = self.qkv_lora(
                        encoder_output,
                        qkv_lora_runtime_params,
                        is_cross_attention=True)
                    cross_qkv_lora = concat(
                        [cross_q_lora, cross_k_lora, cross_v_lora],
                        dim=cross_q_lora.rank() - 1)
                    cross_qkv = cross_qkv + cross_qkv_lora

                return cross_qkv

            if self.skip_cross_qkv:
                conditional = Conditional(cross_kv_cache_gen)
                cond_in1 = conditional.add_input(encoder_output)
                cond_in2 = conditional.add_input(cross_qkv_reuse)

                ## True branch: context phase, compute cross qkv
                cross_qkv_true = compute_cross_qkv(cond_in1)

                ## False branch: generation phase, no compute but need to obey shape constraints
                # because TRT's IfConditional requires the output shape of two subgraphs to be identical
                # our 1st attempt was to stack encoder_output [B, S, H] or [N, H] --> cross qkv [B, S, 3*H] or [N, 3*H],
                # but it still introduces unnecessary concat. A better solution is to create a dummy torch tensor `cross_qkv_resue`
                # with the correct shape and reuse it in every generation step
                cross_qkv_false = cond_in2
                cross_qkv = conditional.add_output(cross_qkv_true,
                                                   cross_qkv_false)
            else:
                cross_qkv = compute_cross_qkv(encoder_output)

        if default_net().plugin_config.gpt_attention_plugin:
            if self.cross_attention and (past_key_value is not None):
                past_key_value = kv_cache_params.past_key_value[1]
            assert self.attention_mask_type in [
                AttentionMaskType.causal, AttentionMaskType.bidirectional,
                AttentionMaskType.bidirectionalglm,
                AttentionMaskType.blocksparse
            ], 'Plugin only support masked MHA.'

            # KV cache scales.
            if self.kv_cache_scaling_factor is not None:
                kv_orig_quant_scale = constant(fp32_array(
                    [1.0])) / self.kv_cache_scaling_factor.value
                kv_quant_orig_scale = self.kv_cache_scaling_factor.value
            else:
                kv_orig_quant_scale = None
                kv_quant_orig_scale = None

            # Attention output scales
            assert (
                not default_net().plugin_config.use_fp8_context_fmha
            ) or self.quant_mode.has_fp8_qdq(
            ), "FP8 Context FMHA must be used together with the fp8 quantization workflow."

            attention_output_orig_quant_scale = self.attention_output_orig_quant_scale.value if self.attention_output_orig_quant_scale is not None else None

            if self.position_embedding_type == PositionEmbeddingType.long_rope:
                short = slice(
                    self.embed_positions_short_factors_for_attention_plugin.
                    value, concat([0, 0, 0]),
                    concat([
                        max(attention_params.sequence_length,
                            self.original_max_position_embeddings),
                        self.rotary_embedding_dim // 2, 2
                    ]))
                long = slice(
                    self.embed_positions_long_factors_for_attention_plugin.
                    value, concat([0, 0, 0]),
                    concat([
                        max(attention_params.sequence_length,
                            self.original_max_position_embeddings),
                        self.rotary_embedding_dim // 2, 2
                    ]))
                short = short.view((1, -1))
                long = long.view((1, -1))
                embed_positions = concat([short, long], dim=0)
                select = where(
                    fmax(attention_params.sequence_length, dim=0) <=
                    self.original_max_position_embeddings, 0, 1)
                rotary_cos_sin = slice(embed_positions,
                                       concat([select, 0]),
                                       sizes=concat([1, shape(long, 1)]))
                short_factors = self.rope_scaling_short_factors.value
                long_factors = self.rope_scaling_long_factors.value
                scale_factors = concat([short_factors, long_factors], dim=0)
                rope_scaling_factors = slice(scale_factors,
                                             concat([select, 0]),
                                             sizes=concat(
                                                 [1, shape(long_factors, 1)]))
                rope_scaling_factors = rope_scaling_factors.view((-1, ))
            else:
                # Rotary cos/sin cache.
                rotary_cos_sin = self.embed_positions_for_gpt_attention.value if self.position_embedding_type.is_rope(
                ) else None
                rope_scaling_factors = None

            if self.position_embedding_type == PositionEmbeddingType.long_rope:
                short_mscale, long_mscale = self.short_mscale, self.long_mscale
            else:
                short_mscale, long_mscale = None, None

            context, past_key_value = gpt_attention(
                qkv=qkv,
                past_key_value=past_key_value,
                sequence_length=attention_params.sequence_length,
                host_past_key_value_lengths=kv_cache_params.
                host_past_key_value_lengths,
                host_max_attention_window_sizes=kv_cache_params.
                host_max_attention_window_sizes,
                host_sink_token_length=kv_cache_params.host_sink_token_length,
                context_lengths=attention_params.context_lengths,
                cache_indirection=kv_cache_params.cache_indirection,
                host_request_types=attention_params.host_request_types,
                layer_idx=self.local_layer_idx,
                num_heads=self.num_attention_heads,
                num_kv_heads=self.num_attention_kv_heads,
                hidden_size_per_head=self.attention_head_size,
                q_scaling=self.q_scaling,
                rotary_embedding_dim=self.rotary_embedding_dim,
                rotary_embedding_base=self.rotary_embedding_base,
                rotary_embedding_scale_type=self.rotary_embedding_scale_type,
                rotary_embedding_scaling_factors=rope_scaling_factors,
                rotary_embedding_short_m_scale=short_mscale,
                rotary_embedding_long_m_scale=long_mscale,
                rotary_embedding_scale=self.rotary_embedding_scale,
                rotary_embedding_max_positions=self.max_position_embeddings,
                rotary_embedding_original_max_positions=self.
                original_max_position_embeddings,
                position_embedding_type=self.position_embedding_type,
                rotary_cos_sin=rotary_cos_sin,
                kv_orig_quant_scale=kv_orig_quant_scale,
                kv_quant_orig_scale=kv_quant_orig_scale,
                attention_output_orig_quant_scale=
                attention_output_orig_quant_scale,
                kv_cache_quant_mode=self.quant_mode,
                max_context_length=attention_params.max_context_length,
                mask_type=self.attention_mask_type,
                block_sparse_block_size=self.block_sparse_params.block_size,
                block_sparse_homo_head_pattern=self.block_sparse_params.
                homo_head_pattern,
                block_sparse_num_local_blocks=self.block_sparse_params.
                num_local_blocks,
                block_sparse_vertical_stride=self.block_sparse_params.
                vertical_stride,
                alibi_slopes=alibi_slopes,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                kv_cache_block_offsets=kv_cache_params.kv_cache_block_offsets
                if not self.cross_attention else
                kv_cache_params.cross_kv_cache_block_offsets,
                host_kv_cache_block_offsets=kv_cache_params.
                host_kv_cache_block_offsets if not self.cross_attention else
                kv_cache_params.host_cross_kv_cache_block_offsets,
                host_kv_cache_pool_pointers=kv_cache_params.
                host_kv_cache_pool_pointers if not self.cross_attention else
                kv_cache_params.host_cross_kv_cache_pool_pointers,
                do_cross_attention=self.cross_attention,
                cross_qkv=cross_qkv,
                cross_qkv_length=attention_params.encoder_max_input_length,
                encoder_input_lengths=attention_params.encoder_input_lengths,
                relative_attention_bias=self.rel_attn_table.value
                if self.relative_attention else None,
                max_distance=self.max_distance,
                host_context_lengths=attention_params.host_context_lengths,
                use_cache=use_cache,
                spec_decoding_generation_lengths=spec_decoding_params.
                spec_decoding_generation_lengths,
                spec_decoding_position_offsets=spec_decoding_params.
                spec_decoding_position_offsets,
                spec_decoding_packed_mask=spec_decoding_params.
                spec_decoding_packed_mask,
                qk_tanh_scale=self.max_attn_value)

        else:
            # plain TensorRT mode
            assert paged_kv_cache == False

            def transpose_for_scores(x,
                                     rotary: bool = False,
                                     is_kv: bool = False):
                _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), _num_attention_heads, self.attention_head_size
                ])
                if rotary:
                    return x.view(new_x_shape)
                else:
                    return x.view(new_x_shape).permute([0, 2, 1, 3])

            # qkv after projection is of shape
            #   [bs, seqlen, (num_attention_heads + 2 * num_attention_kv_heads), attention_head_size].
            # The projected and split qkv after transpose_for_scores():
            #   Q[bs, num_attention_heads, seqlen, attention_head_size]
            #   K[bs, num_attention_kv_heads, seqlen, attention_head_size]
            #   V[bs, num_attention_kv_heads, seqlen, attention_head_size]
            kv_size = self.attention_head_size * self.num_attention_kv_heads
            if unfuse_qkv_gemm:
                query, key, value = qkv[0], qkv[1], qkv[2]
            else:
                query, key, value = split(
                    qkv, [self.attention_hidden_size, kv_size, kv_size], dim=2)

            # in cross attention mode, replace kv by encoder_output
            if self.cross_attention and encoder_output is not None:
                encoder_qkv = self.qkv(encoder_output)
                _, key, value = split(
                    encoder_qkv, [self.attention_hidden_size, kv_size, kv_size],
                    dim=2)

            query = transpose_for_scores(
                query, rotary=self.position_embedding_type.is_rope())
            key = transpose_for_scores(
                key, is_kv=True, rotary=self.position_embedding_type.is_rope())
            value = transpose_for_scores(value, is_kv=True)

            if self.position_embedding_type.is_rope():
                if self.position_embedding_type == PositionEmbeddingType.long_rope:
                    sequence_length = shape(hidden_states, 1)
                    short = slice(
                        self.embed_positions_short_factors.value,
                        concat([0, 0, 0]),
                        concat([
                            1,
                            max(sequence_length,
                                self.original_max_position_embeddings),
                            self.rotary_embedding_dim
                        ]))
                    long = slice(
                        self.embed_positions_long_factors.value,
                        concat([0, 0, 0]),
                        concat([
                            1,
                            max(sequence_length,
                                self.original_max_position_embeddings),
                            self.rotary_embedding_dim
                        ]))
                    embed_positions = concat([short, long], dim=0)
                    select = where(
                        sequence_length <=
                        self.original_max_position_embeddings, 0, 1)
                    embed_positions = slice(embed_positions,
                                            concat([select, 0, 0]),
                                            sizes=shape(short))
                    embed_positions = cast(embed_positions, self.dtype)
                elif is_same_dtype(self.dtype, trt.bfloat16):
                    embed_positions = cast(self.embed_positions.value,
                                           trt.bfloat16)
                else:
                    embed_positions = cast(self.embed_positions.value,
                                           query.dtype)

                if self.rotary_embedding_dim is not None:
                    # When shape(hidden_states, 1) > 1(Context phase), the embedding start from 0,
                    # otherwise (Generation phase) move start to position
                    start = where(
                        shape(hidden_states, 1) > 1, 0,
                        shape(past_key_value, 3))
                    size = where(
                        shape(hidden_states, 1) > 1, shape(hidden_states, 1), 1)
                    sincos = slice(embed_positions, concat([0, start, 0]),
                                   concat([1, size, self.rotary_embedding_dim]))
                    sin, cos = split(sincos,
                                     self.rotary_embedding_dim // 2,
                                     dim=-1)

                    key_rot_size = concat([
                        shape(key, 0),
                        shape(key, 1),
                        shape(key, 2), self.rotary_embedding_dim
                    ])
                    query_rot_size = concat([
                        shape(query, 0),
                        shape(query, 1),
                        shape(query, 2), self.rotary_embedding_dim
                    ])
                    remaining = shape(key, 3) - self.rotary_embedding_dim
                    key_pass_size = concat([
                        shape(key, 0),
                        shape(key, 1),
                        shape(key, 2), remaining
                    ])
                    query_pass_size = concat([
                        shape(query, 0),
                        shape(query, 1),
                        shape(query, 2), remaining
                    ])
                    k_rot = slice(key, [0, 0, 0, 0], key_rot_size)
                    k_pass = slice(key, [0, 0, 0, self.rotary_embedding_dim],
                                   key_pass_size)

                    q_rot = slice(query, [0, 0, 0, 0], query_rot_size)
                    q_pass = slice(query, [0, 0, 0, self.rotary_embedding_dim],
                                   query_pass_size)

                    k_rot = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        k_rot, [cos, sin], self.position_embedding_type)
                    q_rot = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        q_rot, [cos, sin], self.position_embedding_type)

                    key = concat([k_rot, k_pass], dim=3)
                    query = concat([q_rot, q_pass], dim=3)
                else:
                    key = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        key, [cos, sin], self.position_embedding_type)
                    query = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        query, [cos, sin], self.position_embedding_type)

                key = key.permute([0, 2, 1, 3])
                query = query.permute([0, 2, 1, 3])

            if past_key_value is not None and not self.cross_attention:
                if self.kv_cache_scaling_factor is not None:
                    past_key_value = dequantize(
                        past_key_value, self.kv_cache_scaling_factor.value)

                # past_key_value [bs, 2, num_heads, max_seq_len, head_dim]
                past_key, past_value = split(past_key_value, 1, dim=1)

                key_shape = concat([
                    shape(past_key, 0),
                    shape(past_key, 2),
                    shape(past_key, 3),
                    shape(past_key, 4)
                ])
                past_key = past_key.view(key_shape, zero_is_placeholder=False)
                past_value = past_value.view(key_shape,
                                             zero_is_placeholder=False)

                key = concat([past_key, key], dim=2)
                value = concat([past_value, value], dim=2)

            if use_cache:
                key_inflated_shape = concat([
                    shape(key, 0), 1,
                    shape(key, 1),
                    shape(key, 2),
                    shape(key, 3)
                ])
                inflated_key = key.view(key_inflated_shape,
                                        zero_is_placeholder=False)
                inflated_value = value.view(key_inflated_shape,
                                            zero_is_placeholder=False)
                past_key_value = concat([inflated_key, inflated_value], dim=1)

                # TRT quantizes the tensor value by doing `cast(clip(fp_value / scale))` while
                # the plugin quantizes it by doing `cast(clip(fp_value * scale))`.
                if self.kv_cache_scaling_factor is not None:
                    past_key_value = quantize(
                        past_key_value,
                        self.kv_cache_scaling_factor.value,
                        dtype='fp8'
                        if self.quant_mode.has_fp8_kv_cache() else 'int8')

            # MQA broadcast
            if self.num_attention_heads // self.num_attention_kv_heads > 1:
                key = repeat_interleave(
                    key,
                    self.num_attention_heads // self.num_attention_kv_heads, 1)
                value = repeat_interleave(
                    value,
                    self.num_attention_heads // self.num_attention_kv_heads, 1)

            key_length = shape(key, 2)

            # The following code creates a 2D tensor with 0s in the lower triangular (including the diagonal) and
            # +INF in the upper triangular parts. This bias tensor will be added to the output of the Q*K^T matrix
            # multiplication (BMM1). The +INF elements will be transformed to 0s by the Softmax operator that
            # follows. The elements that corresponds to 0s in the bias are unaffected by the bias tensor.
            #
            # Note that when we added to another bias tensor B (for example, with AliBi), the values in the lower-
            # triangular part of the B tensor are not affected and the upper-triangular ones are set to +INF.
            if self.attention_mask_type == AttentionMaskType.causal and not self.cross_attention:
                if self.position_embedding_type.is_alibi():
                    query_length = shape(query, 2)
                    # bsz, tatget_length, past_key_value_length
                    buffer = make_causal_mask(shape(query, 0), query_length,
                                              key_length - query_length,
                                              trt.float32)
                    starts = concat([0, 0, 0, 0])
                    sizes = concat([1, 1, query_length, key_length])
                    generated_mask = slice(buffer, starts, sizes)

                else:
                    query_length = shape(query, 2)
                    starts = concat([0, 0, key_length - query_length, 0])
                    sizes = concat([1, 1, query_length, key_length])
                    if self.position_embedding_type == PositionEmbeddingType.long_rope:
                        buf_shape = (self.original_max_position_embeddings,
                                     self.original_max_position_embeddings)
                    else:
                        buf_shape = (self.max_position_embeddings,
                                     self.max_position_embeddings)
                    select_buf = np.expand_dims(
                        np.tril(np.ones(buf_shape)).astype(bool), (0, 1))

                    select_buf = np.logical_not(select_buf)
                    mask_buf = np.zeros_like(select_buf, np.float32)
                    mask_buf[select_buf] = float('-inf')
                    buffer = constant(mask_buf)
                    generated_mask = slice(buffer, starts, sizes)

            elif self.attention_mask_type == AttentionMaskType.bidirectional and not self.cross_attention:
                query_length = shape(query, 2)
                zero_buf = np.expand_dims(
                    np.zeros((self.max_position_embeddings,
                              self.max_position_embeddings),
                             dtype=np.float32), (0, 1))

                zero_buf[:, :, :-1, -1] = 1
                zero_buf *= -10000

                mask = constant(zero_buf)

                # context phase, query_length
                mask_size = where(query_length > 1, query_length, 1)
                mask_start = where(query_length > 1,
                                   self.max_position_embeddings - mask_size, 1)
                start = concat([0, 0, mask_start, mask_start])
                size = concat([1, 1, mask_size, mask_size])
                generated_mask = slice(mask, start, size)

            if attention_mask is not None:
                if self.cross_attention:
                    batch_size = shape(attention_mask, 0)
                    query_len = shape(attention_mask, 1)
                    encoder_input_len = shape(attention_mask, 2)
                    attention_mask = attention_mask.view(
                        concat([batch_size, 1, query_len, encoder_input_len]))
                    attention_mask = where(attention_mask == 0, float('-inf'),
                                           0.0)
                else:
                    attention_mask = expand_mask(attention_mask,
                                                 shape(query, 2))
            bias = attention_mask
            if self.position_embedding_type.is_alibi():
                alibi_biases = generate_alibi_biases(alibi_slopes, key_length)
                bias = alibi_biases if bias is None else bias + alibi_biases

            if self.relative_attention:
                query_length = shape(query, 2)
                if self.use_implicit_relative_attention:
                    relative_bias = compute_relative_bias(
                        query_length + key_length - 1,
                        key_length,
                        self.num_buckets,
                        self.max_distance,
                        False,  # bidirectional
                        self.rel_attn_table.value.transpose(1, 0),
                        tp_size=self.tp_size,
                        tp_group=self.tp_group,
                        tp_rank=self.tp_rank)
                else:
                    relative_bias = unsqueeze(self.rel_attn_table.value, 0)
                start = concat([0, 0, query_length + key_length - 2, 0])
                size = concat([
                    shape(relative_bias, 0),
                    shape(relative_bias, 1), 1, key_length
                ])
                relative_bias = slice(relative_bias, start, size)

            key = key.permute([0, 1, 3, 2])
            with precision('float32'):
                if norm_before_bmm1:
                    # Apply norm on query earlier to prevent matmul fp16 overflow.
                    query /= (self.q_scaling * self.norm_factor)
                if trt_gte_10() or self.position_embedding_type.is_alibi():
                    attention_scores = matmul(query, key)
                else:
                    # For TRT 9.x, OOTB need this WAR to fuse mha.
                    attention_scores = matmul(cast(query, 'float32'),
                                              cast(key, 'float32'))
                if not norm_before_bmm1:
                    attention_scores = attention_scores / (self.q_scaling *
                                                           self.norm_factor)
                if self.max_attn_value > 0:
                    attention_scores = self.max_attn_value * ACT2FN['tanh'](
                        attention_scores / self.max_attn_value)

                if self.attention_mask_type in [
                        AttentionMaskType.causal,
                        AttentionMaskType.bidirectional
                ] and not self.cross_attention:

                    bias = generated_mask if bias is None else bias + generated_mask

                if bias is not None:
                    bias = cast(bias, attention_scores.dtype)
                    attention_scores = attention_scores + bias

                if self.relative_attention:
                    attention_scores = attention_scores + relative_bias

            attention_probs = softmax(attention_scores, dim=-1)

            if trt_gte_10() or self.position_embedding_type.is_alibi():
                # For trt_version() == 9.x and pos_embed == alibi, TRT has gpu buffer management issues. Need this WAR to avoid peak gpu mem regression.
                # A dummy reshape WAR for mha fusion for 10.0
                attention_probs = attention_probs.view(
                    concat([
                        shape(attention_probs, 0),
                        shape(attention_probs, 1),
                        shape(attention_probs, 2),
                        shape(value, 2)
                    ]))
                context = matmul(attention_probs, value,
                                 use_fp32_acc=False).permute([0, 2, 1, 3])
            else:
                # For TRT 9.x, need this WAR to fuse mha.
                context = matmul(attention_probs,
                                 cast(value, 'float32')).permute([0, 2, 1, 3])
                if context.dtype != value.dtype:
                    context = cast(context, value.dtype)
            context = context.view(
                concat([
                    shape(context, 0),
                    shape(context, 1), self.attention_hidden_size
                ]))

        dense_lora_params = None
        if lora_layer_params is not None:
            dense_lora_params = lora_layer_params.get_runtime_params(
                0, "attn_dense")

        if self.inner_layernorm is not None:
            context = self.inner_layernorm(context)
        context = self.dense(context,
                             lora_runtime_params=dense_lora_params,
                             reduce_fusion_params=reduce_fusion_params)

        if use_cache:
            return (context, past_key_value)
        else:
            return context

    def set_rel_attn_table(self, max_seq_len, precomputed_relative_attention):
        self.rel_attn_table = Parameter(shape=(self.num_attention_heads,
                                               max_seq_len + 1,
                                               max_seq_len + 1),
                                        dtype=self.dtype)
        self.rel_attn_table.value = precomputed_relative_attention


class BertAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings=1024,
                 num_layers=1,
                 attention_head_size=None,
                 num_kv_heads=None,
                 q_scaling=1.0,
                 apply_query_key_layer_scaling=False,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0,
                 relative_attention=False,
                 max_distance=0,
                 num_buckets=0):
        super().__init__()

        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = (
            num_kv_heads + tp_size - 1
        ) // tp_size if num_kv_heads is not None else self.num_attention_heads
        self.hidden_size = hidden_size
        self.attention_hidden_size = self.attention_head_size * self.num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = q_scaling
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers

        self.dtype = dtype

        self.relative_attention = relative_attention
        self.max_distance = max_distance
        self.num_buckets = num_buckets

        # out dim is not necessarily hidden_size + kv specific size (in MQA/GQA), but num_heads * heads_size
        # example: d_model != num_heads * head_size in Flan-T5
        self.qkv = ColumnLinear(hidden_size,
                                tp_size * self.attention_hidden_size +
                                (2 * tp_size * self.num_attention_kv_heads *
                                 self.attention_head_size),
                                bias=bias,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size,
                                gather_output=False)
        self.dense = RowLinear(tp_size * self.num_attention_heads *
                               self.attention_head_size,
                               hidden_size,
                               bias=bias,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

        # see optimize_model's add_lora for LoRA initialization
        self.qkv_lora = None

        # per-layer relative attention table
        if relative_attention:
            self.rel_attn_table = Parameter(shape=(num_attention_heads //
                                                   tp_size, num_buckets),
                                            dtype=dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                input_lengths=None,
                max_input_length=None,
                lora_layer_params=None):
        assert isinstance(hidden_states, Tensor)

        qkv_lora_params = None
        if lora_layer_params is not None:
            qkv_lora_params = lora_layer_params.get_runtime_params(
                0, "attn_qkv")

        qkv = self.qkv(hidden_states, qkv_lora_params)

        if default_net().plugin_config.remove_input_padding:
            assert qkv.ndim() == 2

        if default_net(
        ).plugin_config.lora_plugin and qkv_lora_params is None and lora_layer_params is not None:
            q_lora_params = lora_layer_params.get_runtime_params(0, "attn_q")
            k_lora_params = lora_layer_params.get_runtime_params(0, "attn_k")
            v_lora_params = lora_layer_params.get_runtime_params(0, "attn_v")

            assert (q_lora_params is not None and k_lora_params is not None and v_lora_params is not None) or \
                (q_lora_params is None and k_lora_params is None and v_lora_params is None), "q_lora_params, k_lora_params and v_lora_params should be all enabled or all disabled at the same time."

            if q_lora_params is not None and k_lora_params is not None and v_lora_params is not None:
                qkv_lora_params = LoraRuntimeParams(
                    lora_ranks=[
                        q_lora_params.lora_ranks[0],
                        k_lora_params.lora_ranks[0],
                        v_lora_params.lora_ranks[0],
                    ],
                    lora_weights_pointers=[
                        q_lora_params.lora_weights_pointers[0],
                        k_lora_params.lora_weights_pointers[0],
                        v_lora_params.lora_weights_pointers[0],
                    ],
                    host_request_types=q_lora_params.host_request_types,
                    host_context_lengths=q_lora_params.host_context_lengths,
                    max_context_length=q_lora_params.max_context_length)

                q_lora, k_lora, v_lora = self.qkv_lora(hidden_states,
                                                       qkv_lora_params)
                qkv_lora = concat([q_lora, k_lora, v_lora],
                                  dim=q_lora.rank() - 1)
                qkv = qkv + qkv_lora

        if default_net().plugin_config.bert_attention_plugin:
            # TRT plugin mode
            assert input_lengths is not None
            context = bert_attention(
                qkv,
                input_lengths,
                self.num_attention_heads,
                self.attention_head_size,
                q_scaling=self.q_scaling,
                relative_attention=self.relative_attention,
                max_distance=self.max_distance,
                relative_attention_bias=self.rel_attn_table.value
                if self.relative_attention else None,
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
            query = transpose_for_scores(query)
            key = transpose_for_scores(key)
            value = transpose_for_scores(value)

            key = key.permute([0, 1, 3, 2])
            attention_scores = matmul(query, key, use_fp32_acc=False)
            attention_scores = attention_scores / (self.q_scaling *
                                                   self.norm_factor)

            if self.relative_attention:
                query_len = shape(attention_scores, 2)
                key_len = shape(attention_scores, 3)
                bias = compute_relative_bias(
                    query_len,
                    key_len,
                    self.num_buckets,
                    self.max_distance,
                    True,  # bidirectional
                    self.rel_attn_table.value.transpose(1, 0),
                    tp_size=self.tp_size,
                    tp_group=self.tp_group,
                    tp_rank=self.tp_rank)
                attention_scores = attention_scores + bias

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

        dense_lora_params = None
        if lora_layer_params is not None:
            dense_lora_params = lora_layer_params.get_runtime_params(
                0, "attn_dense")
        context = self.dense(context, lora_runtime_params=dense_lora_params)

        return context


class CogVLMAttention(Attention):

    def __init__(
            self,
            *,
            local_layer_idx,
            hidden_size,
            num_attention_heads,
            num_kv_heads=None,
            max_position_embeddings=1024,
            attention_mask_type=AttentionMaskType.causal,
            bias=True,
            dtype=None,
            rotary_embedding_base=10000.0,
            rotary_embedding_scaling=None,
            tp_group=None,
            tp_size=1,
            tp_rank=0,
            vision_start=1,
            vision_length=1225,
            quant_mode: QuantMode = QuantMode(0),
            dense_bias=None,
    ):
        super().__init__(
            local_layer_idx=local_layer_idx,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            attention_mask_type=attention_mask_type,
            bias=bias,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=rotary_embedding_base,
            rotary_embedding_scaling=rotary_embedding_scaling,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_mode=quant_mode)

        self.vision_length = vision_length
        self.vision_start = vision_start

        self.vis_qkv = QKVColumnLinear(
            hidden_size,
            tp_size * self.num_attention_heads * self.attention_head_size +
            (2 * tp_size * self.num_attention_kv_heads *
             self.attention_head_size),
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=False)
        self.vis_dense = RowLinear(tp_size * self.num_attention_heads *
                                   self.attention_head_size,
                                   hidden_size,
                                   bias=self.dense_bias,
                                   dtype=dtype,
                                   tp_group=tp_group,
                                   tp_size=tp_size)
        embed_positions_for_gpt_attention = RopeEmbeddingUtils.create_sinusoidal_positions_for_cogvlm_attention_plugin(
            self.max_position_embeddings, self.rotary_embedding_dim,
            self.rotary_embedding_base, self.rotary_embedding_scale,
            self.rotary_embedding_scale_type, self.vision_start,
            self.vision_length)
        self.register_parameter(
            'embed_positions_for_gpt_attention',
            Parameter(embed_positions_for_gpt_attention, dtype='float32'))

    def forward(self,
                hidden_states: Tensor,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):

        assert isinstance(hidden_states, Tensor)
        assert (not default_net().plugin_config.remove_input_padding)
        assert (default_net().plugin_config.gpt_attention_plugin)

        bs = shape(hidden_states, 0)
        seq_length = shape(hidden_states, 1)
        bos = slice(hidden_states, [0, 0, 0],
                    concat([bs, self.vision_start, self.hidden_size]))
        vis_seq_length = minimum(self.vision_length + 1, seq_length - 1)
        vision_hidden_states = slice(
            hidden_states, [0, self.vision_start, 0],
            concat([bs, vis_seq_length, self.hidden_size]))
        text_seq_length = maximum(
            0, seq_length - (self.vision_length + 1 + self.vision_start))
        language_hidden_states = slice(
            hidden_states, [0, self.vision_length + 1 + self.vision_start, 0],
            concat([bs, text_seq_length, self.hidden_size]))
        bos_qkv = self.qkv(bos)
        language_qkv = self.qkv(language_hidden_states)
        vision_qkv = self.vis_qkv(vision_hidden_states)
        qkv = concat([bos_qkv, vision_qkv, language_qkv], dim=1)

        assert attention_params is None or attention_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin,
            default_net().plugin_config.remove_input_padding)
        assert kv_cache_params is None or kv_cache_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin)

        past_key_value = None if kv_cache_params is None else kv_cache_params.get_first_past_key_value(
        )

        if default_net().plugin_config.gpt_attention_plugin:
            if self.cross_attention and (past_key_value is not None):
                past_key_value = kv_cache_params.past_key_value[1]
            assert self.attention_mask_type in [
                AttentionMaskType.causal, AttentionMaskType.bidirectional,
                AttentionMaskType.bidirectionalglm
            ], 'Plugin only support masked MHA.'

            # KV cache scales.
            kv_orig_quant_scale = constant(
                fp32_array([1.0])
            ) / self.kv_cache_scaling_factor.value if self.quant_mode.has_kv_cache_quant(
            ) else None
            kv_quant_orig_scale = self.kv_cache_scaling_factor.value if self.quant_mode.has_kv_cache_quant(
            ) else None

            # Attention output scales
            assert (
                not default_net().plugin_config.use_fp8_context_fmha
            ) or self.quant_mode.has_fp8_qdq(
            ), "FP8 Context FMHA must be used together with the fp8 quantization workflow."

            rotary_cos_sin = self.embed_positions_for_gpt_attention.value
            attention_output_orig_quant_scale = self.attention_output_orig_quant_scale.value if self.attention_output_orig_quant_scale is not None else None
            context, past_key_value = gpt_attention(
                qkv=qkv,
                past_key_value=past_key_value,
                sequence_length=attention_params.sequence_length,
                host_past_key_value_lengths=kv_cache_params.
                host_past_key_value_lengths,
                host_max_attention_window_sizes=kv_cache_params.
                host_max_attention_window_sizes,
                host_sink_token_length=kv_cache_params.host_sink_token_length,
                context_lengths=attention_params.context_lengths,
                cache_indirection=kv_cache_params.cache_indirection,
                host_request_types=attention_params.host_request_types,
                layer_idx=self.local_layer_idx,
                num_heads=self.num_attention_heads,
                num_kv_heads=self.num_attention_kv_heads,
                hidden_size_per_head=self.attention_head_size,
                q_scaling=self.q_scaling,
                rotary_embedding_dim=self.rotary_embedding_dim,
                rotary_embedding_base=self.rotary_embedding_base,
                rotary_embedding_scale_type=self.rotary_embedding_scale_type,
                rotary_embedding_scale=self.rotary_embedding_scale,
                rotary_embedding_max_positions=self.max_position_embeddings,
                position_embedding_type=self.position_embedding_type,
                rotary_cos_sin=rotary_cos_sin,
                kv_orig_quant_scale=kv_orig_quant_scale,
                kv_quant_orig_scale=kv_quant_orig_scale,
                attention_output_orig_quant_scale=
                attention_output_orig_quant_scale,
                kv_cache_quant_mode=self.quant_mode,
                max_context_length=attention_params.max_context_length,
                mask_type=self.attention_mask_type,
                alibi_slopes=None,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                vision_start=self.vision_start,
                vision_length=self.vision_length,
                kv_cache_block_offsets=kv_cache_params.kv_cache_block_offsets,
                host_kv_cache_block_offsets=kv_cache_params.
                host_kv_cache_block_offsets,
                host_kv_cache_pool_pointers=kv_cache_params.
                host_kv_cache_pool_pointers,
                do_cross_attention=self.cross_attention,
                cross_qkv=None,
                cross_qkv_length=attention_params.encoder_max_input_length,
                encoder_input_lengths=attention_params.encoder_input_lengths,
                relative_attention_bias=self.rel_attn_table.value
                if self.relative_attention else None,
                max_distance=self.max_distance,
                host_context_lengths=attention_params.host_context_lengths,
                use_cache=use_cache,
                spec_decoding_position_offsets=None,
                spec_decoding_packed_mask=None,
            )

        bs = shape(context, 0)
        seq_length = shape(context, 1)
        bos = slice(context, [0, 0, 0],
                    concat([bs, self.vision_start, self.hidden_size]))
        vis_seq_length = minimum(self.vision_length + 1, seq_length - 1)
        vision_hidden_states = slice(
            context, [0, self.vision_start, 0],
            concat([bs, vis_seq_length, self.hidden_size]))
        text_seq_length = maximum(
            0, seq_length - (self.vision_length + 1 + self.vision_start))
        language_hidden_states = slice(
            context, [0, self.vision_length + 1 + self.vision_start, 0],
            concat([bs, text_seq_length, self.hidden_size]))

        bos_dense = self.dense(bos)
        language_dense = self.dense(language_hidden_states)
        vision_dense = self.vis_dense(vision_hidden_states)
        context = concat([bos_dense, vision_dense, language_dense], dim=1)

        if use_cache:
            return (context, past_key_value)
        else:
            return context
