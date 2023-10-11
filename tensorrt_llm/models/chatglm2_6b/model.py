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
import numpy as np
import tensorrt as trt
import torch

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (PositionEmbeddingType, Tensor, concat, constant,
                           expand, expand_dims, gather_last_token_logits,
                           gpt_attention, index_select, select, shape, slice,
                           split)
from ...layers import (MLP, AttentionMaskType, AttentionParams, ColumnLinear,
                       Embedding, KeyValueCacheParams, RmsNorm, RowLinear)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...quantization import QuantMode
from ..generation_mixin import GenerationMixin


def apply_rotary_pos_emb_trt(x: Tensor, rope_cache: Tensor) -> Tensor:
    # x-> [seq, batch, num_heads, 2]
    x = x.permute((1, 0, 2, 3))
    # sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    sq = shape(x, 0)
    b = shape(x, 1)
    nh = shape(x, 2)
    shape(x, 3)
    # rope_cache shape: seq,batch,heads,2 rot_dim = 2* numheads
    #rope_cache: seq,batch,num_states/4,2
    rot_dim = shape(rope_cache, 2) * constant(np.array(2, dtype=np.int32))
    starts = concat([0, 0, 0, 0])
    sizes = concat([sq, b, nh, rot_dim])
    # first half
    x_rot = slice(x, starts, sizes)
    starts = concat([0, 0, 0, rot_dim])
    # second half
    x_pass = slice(x, starts, sizes)
    # truncate to support variable sizes
    rope_cache = slice(rope_cache, (0, 0, 0, 0), (concat(
        [sq,
         shape(rope_cache, 1),
         shape(rope_cache, 2),
         shape(rope_cache, 3)])))
    xshaped = x_rot.view(concat([sq, b, nh, rot_dim / 2, 2]))
    rope_cache = rope_cache.view(concat([sq, b, 1, shape(xshaped, 3), 2]))
    # first half
    xshape0 = select(xshaped, 4, 0)
    # second half
    xshape1 = select(xshaped, 4, 1)
    # first half
    rope_cache0 = select(rope_cache, 4, 0)
    # second half
    rope_cache1 = select(rope_cache, 4, 1)
    out0 = xshape0 * rope_cache0 - xshape1 * rope_cache1
    out1 = xshape1 * rope_cache0 + xshape0 * rope_cache1
    out0 = expand_dims(out0, 4)
    out1 = expand_dims(out1, 4)
    x_out2_v1 = concat([out0, out1], 4)
    x_out2 = x_out2_v1.view(
        concat([sq, b, nh, shape(x_out2_v1, 3) * shape(x_out2_v1, 4)]))
    output = concat([x_out2, x_pass], dim=3)
    # to batch,seq,num_group,head_states
    output = output.permute((1, 0, 2, 3))
    return output


class RotaryEmbedding(Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, seq_len: int):
        theta = 1.0 / (10000**(torch.arange(0, self.dim, 2) / self.dim))
        seq_idx = torch.arange(seq_len)
        idx_theta = torch.outer(seq_idx, theta).float()
        cache = torch.stack(
            [torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        cache = cache.half()
        # create rope embeddings and make it constant
        cache = constant(cache.numpy())
        return cache


class ChatGLM2Attention(Module):

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        layer_number,
        kv_channels=128,
        multi_query_group_num=2,
        apply_query_key_layer_scaling=False,
        attention_mask_type=AttentionMaskType.causal,
        qkv_bias=True,
        linear_bias=False,
        dtype='float16',
        use_int8_kv_cache=False,
        tp_group=None,
        tp_size=1,
    ):
        super().__init__()

        self.attention_mask_type = attention_mask_type
        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_multi_query_groups_per_partition = multi_query_group_num
        self.num_attention_kv_heads = self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.projection_size = num_attention_heads * kv_channels
        self.hidden_size_per_attention_head = kv_channels
        self.layer_number = layer_number
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.q_scaling = 1
        if apply_query_key_layer_scaling:
            self.q_scaling *= self.layer_number
        self.position_embedding_type = PositionEmbeddingType.learned_absolute
        self.multi_block_mode = False
        self.multi_query_mode = False

        self.rotary_embedding_dim = 0

        self.dtype = dtype

        self.use_int8_kv_cache = use_int8_kv_cache
        if self.use_int8_kv_cache:
            self.kv_orig_quant_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_quant_orig_scale = Parameter(shape=(1, ), dtype='float32')
        else:
            self.register_parameter('kv_orig_quant_scale', None)
            self.register_parameter('kv_quant_orig_scale', None)

        # Note: in multi_query_mode, only query heads are split between multiple GPUs,
        # while key/value head are not split as there is only one head per key/value.
        # The output feature size is therefore (h/tp + 2) * d, where h is num_heads,
        # d is head_size, and tp is tensor_parallel_size.
        # In ColumnLinear op, the output dim is calculated by (h + 2*tp) * d / tp,
        # which matches the desired output size (h/tp + 2) * d after splitting
        self.qkv_hidden_size = (self.projection_size +
                                2 * self.hidden_size_per_attention_head * 2)
        self.qkv = ColumnLinear(hidden_size,
                                self.qkv_hidden_size,
                                bias=qkv_bias,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size,
                                gather_output=False)
        self.dense = RowLinear(hidden_size,
                               hidden_size,
                               bias=linear_bias,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

    def forward(self,
                hidden_states: Tensor,
                rotary_pos_emb,
                use_cache=True,
                kv_cache_params=None,
                attention_params=None):
        if not default_net().plugin_config.gpt_attention_plugin:
            raise ValueError(
                'ChatGLM2 is only supported with GPTAttention plugin,pleas build it with --use_gpt_attention_plugin argument.'
            )
        assert isinstance(hidden_states, Tensor)
        qkv = self.qkv(hidden_states)
        query, key, value = split(qkv, [
            self.num_attention_heads * self.hidden_size_per_attention_head,
            self.num_multi_query_groups_per_partition *
            self.hidden_size_per_attention_head,
            self.num_multi_query_groups_per_partition *
            self.hidden_size_per_attention_head,
        ],
                                  dim=-1)
        query = query.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1), self.num_attention_heads,
                self.attention_head_size
            ]))
        key = key.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1), self.num_multi_query_groups_per_partition,
                self.attention_head_size
            ]))
        value = value.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1), self.num_multi_query_groups_per_partition,
                self.attention_head_size
            ]))

        if rotary_pos_emb is not None:
            query = apply_rotary_pos_emb_trt(query, rotary_pos_emb)
            key = apply_rotary_pos_emb_trt(key, rotary_pos_emb)
        # batch,seq,num_group,1,head_states
        key = expand_dims(key, 3)
        #expand 16x
        expand_rate = self.num_attention_heads // self.num_multi_query_groups_per_partition
        key = expand(
            key,
            concat([
                shape(key, 0),
                shape(key, 1),
                shape(key, 2), expand_rate,
                shape(key, 4)
            ]))
        # batch,seq,num_heads,head_states
        key = key.view(
            concat([
                shape(key, 0),
                shape(key, 1),
                shape(key, 2) * shape(key, 3),
                shape(key, 4)
            ]))
        value = expand_dims(value, 3)
        value = expand(
            value,
            concat([
                shape(value, 0),
                shape(value, 1),
                shape(value, 2), expand_rate,
                shape(value, 4)
            ]))
        value = value.view(
            concat([
                shape(value, 0),
                shape(value, 1),
                shape(value, 2) * shape(value, 3),
                shape(value, 4)
            ]))
        qkv = concat([query, key, value], dim=2)
        qkv = qkv.view(
            concat([shape(qkv, 0),
                    shape(qkv, 1), self.hidden_size * 3]))
        assert attention_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin,
            default_net().plugin_config.remove_input_padding)
        assert kv_cache_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin)
        kv_orig_quant_scale = self.kv_orig_quant_scale.value if self.use_int8_kv_cache else None
        kv_quant_orig_scale = self.kv_quant_orig_scale.value if self.use_int8_kv_cache else None
        context, past_key_value = gpt_attention(
            tensor=qkv,
            past_key_value=kv_cache_params.get_first_past_key_value(),
            sequence_length=attention_params.sequence_length,
            host_past_key_value_lengths=kv_cache_params.
            host_past_key_value_lengths,
            context_lengths=attention_params.context_lengths,
            cache_indirection=kv_cache_params.cache_indirection,
            host_request_types=attention_params.host_request_types,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.
            num_attention_heads,  # since self.multi_query_mode is set to False
            hidden_size_per_head=self.attention_head_size,
            q_scaling=self.q_scaling,
            rotary_embedding_dim=self.rotary_embedding_dim,
            position_embedding_type=self.position_embedding_type,
            multi_block_mode=self.multi_block_mode,
            kv_orig_quant_scale=kv_orig_quant_scale,
            kv_quant_orig_scale=kv_quant_orig_scale,
            kv_cache_quant_mode=QuantMode.INT8_KV_CACHE
            if self.use_int8_kv_cache else QuantMode(0),
            kv_cache_block_pointers=kv_cache_params.
            get_first_kv_cache_block_pointers(),
            max_context_length=attention_params.max_context_length,
            host_context_lengths=attention_params.host_context_lengths)
        # dense layer after self-attention
        context = self.dense(context)
        if use_cache:
            return (context, past_key_value)
        else:
            return context


class ChatGLM2Block(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 kv_channels=128,
                 multi_query_group_num=2,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.causal,
                 qkv_bias=True,
                 linear_bias=False,
                 use_int8_kv_cache=False,
                 tp_group=None,
                 tp_size=1,
                 ffn_hiden_size=13696,
                 layer_number=1,
                 eps=1e-5,
                 act_func='swiglu',
                 dtype=trt.float16,
                 quant_mode=QuantMode(0)):
        super(ChatGLM2Block, self).__init__()
        self.layer_number = layer_number
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dtype = dtype
        self.ffn_hiden_size = ffn_hiden_size
        self.apply_residual_connection_post_layernorm = False
        self.fp32_residual_connection = False

        LayerNormFunc = RmsNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(self.hidden_size,
                                             eps=eps,
                                             dtype=dtype)

        # Self attention.
        self.self_attention = ChatGLM2Attention(
            hidden_size, num_attention_heads, layer_number, kv_channels,
            multi_query_group_num, apply_query_key_layer_scaling,
            attention_mask_type, qkv_bias, linear_bias, dtype,
            use_int8_kv_cache, tp_group, tp_size)
        self.hidden_dropout = 0.0

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(self.hidden_size,
                                                      eps=eps,
                                                      dtype=dtype)

        self.mlp = MLP(self.hidden_size, ffn_hiden_size, act_func, linear_bias,
                       dtype)

    def forward(self,
                hidden_states,
                rotary_pos_emb,
                use_cache=True,
                kv_cache_params=None,
                attention_params=None):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.

        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            rotary_pos_emb,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = hidden_states + attention_output

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = residual + mlp_output

        return output, kv_cache


class ChatGLM2Transformer(Module):
    """Transformer class."""

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 kv_channels=128,
                 multi_query_group_num=2,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.causal,
                 qkv_bias=True,
                 linear_bias=False,
                 use_int8_kv_cache=False,
                 tp_group=None,
                 tp_size=1,
                 ffn_hiden_size=13696,
                 num_layers=28,
                 eps=1e-5,
                 act_func='swiglu',
                 dtype=trt.float16,
                 quant_mode=QuantMode(0)):
        super(ChatGLM2Transformer, self).__init__()

        self.fp32_residual_connection = False
        self.post_layer_norm = True

        # Number of layers.
        self.num_layers = num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return ChatGLM2Block(hidden_size, num_attention_heads, kv_channels,
                                 multi_query_group_num,
                                 apply_query_key_layer_scaling,
                                 attention_mask_type, qkv_bias, linear_bias,
                                 use_int8_kv_cache, tp_group, tp_size,
                                 ffn_hiden_size, layer_number, eps, act_func,
                                 dtype, quant_mode)

        self.layers = ModuleList(
            build_layer(i + 1) for i in range(self.num_layers))

        if self.post_layer_norm:
            self.final_layernorm = RmsNorm(hidden_size, eps=eps, dtype=dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(self,
                hidden_states,
                rotary_pos_emb,
                use_cache=True,
                kv_cache_params=None,
                attention_params=None):

        presents = []
        for index in range(self.num_layers):
            layer = self._get_layer(index)
            hidden_states, kv_cache = layer(
                hidden_states,
                rotary_pos_emb,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[kv_cache_params.past_key_value[index]],
                    kv_cache_block_pointers=[
                        kv_cache_params.kv_cache_block_pointers[index]
                    ],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params)
            presents.append(kv_cache)

        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents


class ChatGLM2Model(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 kv_channels=128,
                 multi_query_group_num=2,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.causal,
                 qkv_bias=True,
                 linear_bias=False,
                 use_int8_kv_cache=False,
                 mapping=Mapping(),
                 ffn_hiden_size=13696,
                 num_layers=28,
                 eps=1e-5,
                 act_func='swiglu',
                 dtype=trt.float16,
                 quant_mode=QuantMode(0),
                 max_seq_length=32768,
                 vocab_size=65024):
        super(ChatGLM2Model, self).__init__()

        self.dtype = dtype
        self.embedding = Embedding(vocab_size, hidden_size, dtype=dtype)
        self.num_layers = num_layers
        self.multi_query_group_num = multi_query_group_num
        self.kv_channels = kv_channels

        # Rotary positional embeddings
        self.max_seq_length = max_seq_length
        rotary_dim = kv_channels
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, )
        self.encoder = ChatGLM2Transformer(
            hidden_size, num_attention_heads, kv_channels,
            multi_query_group_num, apply_query_key_layer_scaling,
            attention_mask_type, qkv_bias, linear_bias, use_int8_kv_cache,
            mapping.tp_group, mapping.tp_size, ffn_hiden_size, num_layers, eps,
            act_func, dtype, quant_mode)

    def forward(
        self,
        input_ids: Tensor,
        position_ids,
        use_cache=True,
        kv_cache_params=None,
        attention_params=None,
    ):

        inputs_embeds = self.embedding(input_ids)
        # Rotary positional embeddings
        # generate 32768 pos embeddings
        # max_seq_length,head_dim/4,2
        rotary_pos_emb = self.rotary_pos_emb(self.max_seq_length)
        flat_position = position_ids.view(
            concat([shape(position_ids, 0) * shape(position_ids, 1)]))
        selected_pos_emb = index_select(rotary_pos_emb, 0, flat_position)
        # selected batch,seq from rotary_pos_emb
        selected_pos_emb = selected_pos_emb.view(
            concat([
                shape(position_ids, 0),
                shape(position_ids, 1),
                shape(rotary_pos_emb, 1),
                shape(rotary_pos_emb, 2)
            ]))
        # seq,batch
        selected_pos_emb = selected_pos_emb.permute((1, 0, 2, 3))
        # return inputs_embeds,selected_pos_emb
        # Run encoder.

        hidden_states, presents = self.encoder(
            inputs_embeds,
            selected_pos_emb,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        return hidden_states, presents


class ChatGLM2HeadModel(ChatGLM2Model, GenerationMixin):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 kv_channels=128,
                 multi_query_group_num=2,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.causal,
                 qkv_bias=True,
                 linear_bias=False,
                 use_int8_kv_cache=False,
                 mapping=Mapping(),
                 ffn_hiden_size=13696,
                 num_layers=28,
                 eps=1e-5,
                 act_func='swiglu',
                 dtype=trt.float16,
                 quant_mode=QuantMode(0),
                 max_seq_length=32768,
                 vocab_size=65024,
                 use_cache=True,
                 kv_cache_block_pointers=None):
        if isinstance(dtype, str):
            self._kv_dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._kv_dtype = dtype
        self._dtype = self._kv_dtype
        if quant_mode.has_int8_kv_cache():
            self._kv_dtype = str_dtype_to_trt('int8')
        elif quant_mode.has_fp8_kv_cache():
            self._kv_dtype = str_dtype_to_trt('fp8')
        self.use_cache = use_cache
        self.kv_cache_block_pointers = kv_cache_block_pointers
        self.quant_mode = quant_mode
        self._num_layers = num_layers
        self._num_heads = num_attention_heads
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._tp_size = mapping.tp_size
        super().__init__(hidden_size, num_attention_heads, kv_channels,
                         multi_query_group_num, apply_query_key_layer_scaling,
                         attention_mask_type, qkv_bias, linear_bias,
                         use_int8_kv_cache, mapping, ffn_hiden_size, num_layers,
                         eps, act_func, dtype, quant_mode, max_seq_length,
                         vocab_size)
        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        self.lm_head = ColumnLinear(hidden_size,
                                    vocab_size_padded,
                                    bias=False,
                                    dtype=dtype,
                                    tp_group=mapping.tp_group,
                                    tp_size=mapping.tp_size,
                                    gather_output=True)

    def forward(self,
                input_ids=None,
                position_ids=None,
                last_token_ids=None,
                kv_cache_params=None,
                attention_params=None):

        hidden_states = super().forward(input_ids, position_ids, self.use_cache,
                                        kv_cache_params, attention_params)

        if self.use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self._dtype)

        if default_net().plugin_config.paged_kv_cache == False:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self._kv_dtype)
            return (lm_logits, presents)
        return lm_logits

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       max_new_tokens,
                       use_cache,
                       max_beam_width: int = 1):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''
        # Prepare inputs
        head_size = self._hidden_size // self._num_heads
        num_heads = self._num_heads // self._tp_size
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin

        model_inputs = self.prepare_basic_inputs(
            max_batch_size,
            max_beam_width,
            max_input_len,
            max_new_tokens,
            num_heads,
            head_size,
            self.num_layers,
            self._kv_dtype,
            remove_input_padding,
            use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin)

        return (model_inputs['input_ids'], model_inputs['position_ids'],
                model_inputs['last_token_ids'],
                KeyValueCacheParams(
                    past_key_value=model_inputs['past_key_value'],
                    host_past_key_value_lengths=model_inputs[
                        'host_past_key_value_lengths'],
                    kv_cache_block_pointers=model_inputs[
                        'kv_cache_block_pointers_list'],
                    cache_indirection=model_inputs['cache_indirection'],
                ),
                AttentionParams(
                    sequence_length=model_inputs['sequence_length'],
                    context_lengths=model_inputs['context_lengths'],
                    host_context_lengths=model_inputs['host_context_lengths'],
                    max_context_length=max_input_len,
                    host_request_types=model_inputs['host_request_types']))
