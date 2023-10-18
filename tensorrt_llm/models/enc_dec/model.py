import math
from collections import OrderedDict
from typing import Optional

import numpy as np
import tensorrt as trt

from tensorrt_llm._common import default_net
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import (LayerNormPositionType, LayerNormType,
                                     PositionEmbeddingType, Tensor, assertion,
                                     concat, constant, expand, expand_mask,
                                     gather_last_token_logits, shape, slice)
from tensorrt_llm.layers import (MLP, Attention, AttentionMaskType,
                                 AttentionParams, BertAttention, ColumnLinear,
                                 Embedding, GroupNorm, KeyValueCacheParams,
                                 LayerNorm, RmsNorm)
from tensorrt_llm.module import Module, ModuleList

layernorm_map = {
    LayerNormType.LayerNorm: LayerNorm,
    LayerNormType.RmsNorm: RmsNorm,
    LayerNormType.GroupNorm: GroupNorm,
}


class EncDecEmbedding(Module):

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings=None,
        has_position_embedding=False,
        type_vocab_size=None,
        has_embedding_layernorm=False,
        has_embedding_scale=False,
        layernorm_eps=1e-5,
        layernorm_type=LayerNormType.LayerNorm,
        dtype=None,
    ):
        super().__init__()

        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        self.vocab_embedding = Embedding(vocab_size, hidden_size, dtype=dtype)
        self.position_embedding = None
        self.max_position_embeddings = max_position_embeddings
        if has_position_embedding:
            self.position_embedding = Embedding(max_position_embeddings,
                                                hidden_size,
                                                dtype=dtype)

        self.token_type_embedding = None
        if type_vocab_size:
            self.token_type_embedding = Embedding(type_vocab_size,
                                                  hidden_size,
                                                  dtype=dtype)

        # e.g. BART true, T5 false
        self.embedding_layernorm = None
        if has_embedding_layernorm:
            self.embedding_layernorm = ln_type(normalized_shape=hidden_size,
                                               eps=layernorm_eps,
                                               dtype=dtype)

        # e.g. BART true, T5 false
        self.embedding_scale = 1.0
        if has_embedding_scale:
            self.embedding_scale = math.sqrt(hidden_size)

        # Note: embedding offset in BART is not considered as a standard. For the specific case,
        # we just need to shrink its position embedding table by [offset:] during weight loading

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        seq_len_2d = concat([1, shape(input_ids, 1)])

        if self.position_embedding:
            position_ids_buffer = constant(
                np.expand_dims(
                    np.arange(self.max_position_embeddings).astype(np.int32),
                    0))
            if position_ids is None:
                # slice
                position_ids = slice(position_ids_buffer,
                                     starts=[0, 0],
                                     sizes=seq_len_2d)
                position_ids = expand(position_ids, shape(input_ids))

        if self.token_type_embedding:
            token_type_ids_buffer = constant(
                np.expand_dims(
                    np.zeros(self.max_position_embeddings).astype(np.int32), 0))
            if token_type_ids is None:
                # slice
                token_type_ids = slice(token_type_ids_buffer,
                                       starts=[0, 0],
                                       sizes=seq_len_2d)
                token_type_ids = expand(token_type_ids, shape(input_ids))

        x = self.vocab_embedding(input_ids) * self.embedding_scale
        if self.position_embedding:
            x = x + self.position_embedding(position_ids)
        if self.token_type_embedding:
            x = x + self.token_type_embedding(token_type_ids)
        if self.embedding_layernorm:
            x = self.embedding_layernorm(x)

        return x


class EncoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 num_attention_heads,
                 num_kv_heads,
                 max_position_embeddings=None,
                 q_scaling=1.0,
                 has_attention_qkvo_bias=False,
                 has_mlp_bias=False,
                 layernorm_position=LayerNormPositionType.pre_layernorm,
                 layernorm_type=LayerNormType.LayerNorm,
                 layernorm_eps=1e-5,
                 hidden_act="relu",
                 tp_group=None,
                 tp_size=1,
                 dtype=None,
                 residual_scaling=1.0,
                 relative_attention=False,
                 max_distance=0,
                 num_buckets=0):
        super().__init__()

        # e.g. BART regular, T5 RMS
        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        # e.g. BART post, T5 pre
        self.layernorm_position = layernorm_position

        # Note: q_scaling convention in TRT-LLM plugin is 1.f / (q_scaling * sqrt(head_size))
        # if don't want q_scaling, use 1/sqrt(head_size) to cancel
        # e.g. BART false, T5 false
        self.attention = BertAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            q_scaling=q_scaling,
            bias=has_attention_qkvo_bias,
            tp_group=tp_group,
            tp_size=tp_size,
            dtype=dtype,
            relative_attention=relative_attention,
            max_distance=max_distance,
            num_buckets=num_buckets)

        self.attention_layernorm = ln_type(normalized_shape=hidden_size,
                                           eps=layernorm_eps,
                                           dtype=dtype)

        self.mlp = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            hidden_act=hidden_act,
            bias=has_mlp_bias,
            tp_group=tp_group,
            tp_size=tp_size,
            dtype=dtype,
        )
        self.mlp_layernorm = ln_type(normalized_shape=hidden_size,
                                     eps=layernorm_eps,
                                     dtype=dtype)

        self.residual_scaling = residual_scaling

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                input_lengths=None):
        assert isinstance(hidden_states, Tensor)

        # self attention
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.attention_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
        )

        hidden_states = residual + attention_output

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.attention_layernorm(hidden_states)

        # MLP
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        return hidden_states


class DecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 num_attention_heads,
                 num_kv_heads,
                 max_position_embeddings=None,
                 q_scaling=1.0,
                 has_attention_qkvo_bias=False,
                 has_mlp_bias=False,
                 layernorm_position=LayerNormPositionType.pre_layernorm,
                 layernorm_type=LayerNormType.LayerNorm,
                 layernorm_eps=1e-5,
                 hidden_act="relu",
                 tp_group=None,
                 tp_size=1,
                 dtype=None,
                 residual_scaling=1.0,
                 relative_attention=False,
                 max_distance=0,
                 num_buckets=0):
        super().__init__()

        # e.g. BART regular, T5 RMS
        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        # e.g. BART post, T5 pre
        self.layernorm_position = layernorm_position

        # Note: q_scaling convention in TRT-LLM plugin is 1.f / (q_scaling * sqrt(head_size))
        # if don't want q_scaling, use 1/sqrt(head_size) to cancel
        # e.g. BART false, T5 false
        self.self_attention = Attention(
            hidden_size,
            num_attention_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            q_scaling=q_scaling,
            bias=has_attention_qkvo_bias,
            attention_mask_type=AttentionMaskType.causal,
            tp_group=tp_group,
            tp_size=tp_size,
            dtype=dtype,
            cross_attention=False,
            relative_attention=relative_attention,
            max_distance=max_distance,
            num_buckets=num_buckets,
            position_embedding_type=PositionEmbeddingType.relative
            if relative_attention else PositionEmbeddingType.learned_absolute)

        self.self_attention_layernorm = ln_type(normalized_shape=hidden_size,
                                                eps=layernorm_eps,
                                                dtype=dtype)

        self.cross_attention = Attention(
            hidden_size,
            num_attention_heads,
            num_kv_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            q_scaling=q_scaling,
            bias=has_attention_qkvo_bias,
            attention_mask_type=AttentionMaskType.causal,
            tp_group=tp_group,
            tp_size=tp_size,
            dtype=dtype,
            cross_attention=True,
            relative_attention=
            False,  # Cross attention has no relative attention bias
            max_distance=max_distance,
            num_buckets=num_buckets,
            position_embedding_type=PositionEmbeddingType.learned_absolute)

        self.cross_attention_layernorm = ln_type(normalized_shape=hidden_size,
                                                 eps=layernorm_eps,
                                                 dtype=dtype)

        self.mlp = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            hidden_act=hidden_act,
            bias=has_mlp_bias,
            tp_group=tp_group,
            tp_size=tp_size,
            dtype=dtype,
        )

        self.mlp_layernorm = ln_type(normalized_shape=hidden_size,
                                     eps=layernorm_eps,
                                     dtype=dtype)

        self.residual_scaling = residual_scaling

    def forward(
        self,
        hidden_states: Tensor,
        encoder_output: Optional[Tensor] = None,
        attention_mask=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
    ):
        assert isinstance(hidden_states, Tensor)

        if encoder_output:
            assert isinstance(encoder_output, Tensor)

        # self-attention
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.self_attention_layernorm(hidden_states)

        attention_output = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )

        if use_cache:
            attention_output, presents_self = attention_output

        hidden_states = residual + attention_output

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.self_attention_layernorm(hidden_states)

        # cross attention
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.cross_attention_layernorm(hidden_states)

        attention_output = self.cross_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_output=encoder_output,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )

        if use_cache:
            attention_output, presents_cross = attention_output

        hidden_states = residual + attention_output

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.cross_attention_layernorm(hidden_states)

        # MLP
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        if use_cache:
            return (hidden_states, presents_self, presents_cross)
        return hidden_states


class EncoderModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 ffn_hidden_size,
                 vocab_size,
                 dtype,
                 num_kv_heads=None,
                 max_position_embeddings=None,
                 has_position_embedding=False,
                 relative_attention=False,
                 max_distance=None,
                 num_buckets=None,
                 type_vocab_size=None,
                 has_embedding_layernorm=False,
                 has_embedding_scale=False,
                 q_scaling=1.0,
                 has_attention_qkvo_bias=False,
                 has_mlp_bias=False,
                 has_model_final_layernorm=False,
                 layernorm_eps=1e-5,
                 layernorm_position=LayerNormPositionType.pre_layernorm,
                 layernorm_type=LayerNormType.LayerNorm,
                 hidden_act="relu",
                 tp_group=None,
                 tp_size=1,
                 residual_scaling=1.0):
        super().__init__()

        self.has_position_embedding = has_position_embedding
        self.has_token_type_embedding = type_vocab_size is not None

        # e.g. BART regular, T5 RMS
        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        # e.g. BART true, T5 false
        self.has_attention_qkvo_bias = has_attention_qkvo_bias
        self.has_mlp_bias = has_mlp_bias

        if isinstance(dtype, str):
            self._dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._dtype = dtype

        self.num_layers = num_layers

        self.embedding = EncDecEmbedding(
            vocab_size,
            hidden_size,
            max_position_embeddings=max_position_embeddings,
            has_position_embedding=has_position_embedding,
            type_vocab_size=type_vocab_size,
            has_embedding_layernorm=has_embedding_layernorm,
            has_embedding_scale=has_embedding_scale,
            layernorm_eps=layernorm_eps,
            layernorm_type=layernorm_type,
            dtype=dtype)

        self.encoder_layers = ModuleList([
            EncoderLayer(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                num_attention_heads=num_heads,
                num_kv_heads=num_kv_heads if num_kv_heads else num_heads,
                max_position_embeddings=max_position_embeddings,
                q_scaling=q_scaling,
                has_attention_qkvo_bias=has_attention_qkvo_bias,
                has_mlp_bias=has_mlp_bias,
                layernorm_position=layernorm_position,
                layernorm_eps=layernorm_eps,
                layernorm_type=layernorm_type,
                hidden_act=hidden_act,
                tp_group=tp_group,
                tp_size=tp_size,
                dtype=dtype,
                residual_scaling=residual_scaling,
                relative_attention=relative_attention,
                max_distance=max_distance,
                num_buckets=num_buckets) for _ in range(num_layers)
        ])

        # e.g. BART false, T5 true
        if has_model_final_layernorm:
            self.final_layernorm = ln_type(normalized_shape=hidden_size,
                                           eps=layernorm_eps,
                                           dtype=dtype)

    def forward(self,
                input_ids: Tensor,
                input_lengths=None,
                position_ids=None,
                token_type_ids=None):
        hidden_states = self.embedding(input_ids, position_ids, token_type_ids)
        for layer_idx, encoder_layer in enumerate(self.encoder_layers):
            hidden_states = encoder_layer(hidden_states=hidden_states,
                                          input_lengths=input_lengths)

        if self.final_layernorm:
            hidden_states = self.final_layernorm(hidden_states)

        hidden_states.mark_output('encoder_output', self._dtype)

        return hidden_states

    def prepare_inputs(self, max_batch_size, max_input_len):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        inlen_range = [1, (max_input_len + 1) // 2, max_input_len]
        num_tokens_range = [
            1,
            (max_input_len * max_batch_size + 1) // 2,
            max_input_len * max_batch_size,
        ]

        position_ids, token_type_ids = None, None
        remove_input_padding = default_net().plugin_config.remove_input_padding
        if remove_input_padding:
            input_ids = Tensor(
                name="input_ids",
                dtype=trt.int32,
                shape=[1, -1],
                dim_range=OrderedDict([("batch_size_fake", [1]),
                                       ("num_tokens", [num_tokens_range])]),
            )
            if self.has_position_embedding:
                position_ids = Tensor(
                    name='position_ids',
                    dtype=trt.int32,
                    shape=[1, -1],
                    dim_range=OrderedDict([('batch_size_fake', [1]),
                                           ('num_tokens', [num_tokens_range])]),
                )
            if self.has_token_type_embedding:
                token_type_ids = Tensor(
                    name='token_type_ids',
                    dtype=trt.int32,
                    shape=[1, -1],
                    dim_range=OrderedDict([('batch_size_fake', [1]),
                                           ('num_tokens', [num_tokens_range])]),
                )
        else:
            input_ids = Tensor(
                name="input_ids",
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([("batch_size", [bs_range]),
                                       ("input_len", [inlen_range])]),
            )
            if self.has_position_embedding:
                position_ids = Tensor(
                    name='position_ids',
                    dtype=trt.int32,
                    shape=[-1, -1],
                    dim_range=OrderedDict([('batch_size', [bs_range]),
                                           ('input_len', [inlen_range])]),
                )
            if self.has_token_type_embedding:
                token_type_ids = Tensor(
                    name='token_type_ids',
                    dtype=trt.int32,
                    shape=[-1, -1],
                    dim_range=OrderedDict([('batch_size', [bs_range]),
                                           ('input_len', [inlen_range])]),
                )

        input_lengths = Tensor(
            name="input_lengths",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size", [bs_range])]),
        )

        return (input_ids, input_lengths, position_ids, token_type_ids)


class DecoderModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 ffn_hidden_size,
                 encoder_num_heads,
                 encoder_hidden_size,
                 vocab_size,
                 dtype,
                 logits_dtype='float32',
                 num_kv_heads=None,
                 max_position_embeddings=None,
                 has_position_embedding=False,
                 relative_attention=False,
                 max_distance=None,
                 num_buckets=None,
                 type_vocab_size=None,
                 has_embedding_layernorm=False,
                 has_embedding_scale=False,
                 q_scaling=1.0,
                 has_attention_qkvo_bias=False,
                 has_mlp_bias=False,
                 has_model_final_layernorm=False,
                 layernorm_eps=1e-5,
                 layernorm_position=LayerNormPositionType.pre_layernorm,
                 layernorm_type=LayerNormType.LayerNorm,
                 hidden_act="relu",
                 has_lm_head_bias=False,
                 tp_group=None,
                 tp_size=1,
                 residual_scaling=1.0):
        super().__init__()

        self.has_position_embedding = has_position_embedding
        self.has_token_type_embedding = type_vocab_size is not None

        # e.g. BART regular, T5 RMS
        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        # e.g. BART true, T5 false
        self.has_attention_qkvo_bias = has_attention_qkvo_bias
        self.has_mlp_bias = has_mlp_bias

        if isinstance(dtype, str):
            self._dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._dtype = dtype

        # no quantization considered for now
        self._kv_dtype = self._dtype

        if isinstance(logits_dtype, str):
            self._logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self._logits_dtype = logits_dtype

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_heads = encoder_num_heads
        self.tp_size = tp_size

        self.has_position_embedding = has_position_embedding
        self.has_token_type_embedding = type_vocab_size is not None

        self.embedding = EncDecEmbedding(
            vocab_size,
            hidden_size,
            max_position_embeddings=max_position_embeddings,
            has_position_embedding=has_position_embedding,
            type_vocab_size=type_vocab_size,
            has_embedding_layernorm=has_embedding_layernorm,
            has_embedding_scale=has_embedding_scale,
            layernorm_eps=layernorm_eps,
            layernorm_type=layernorm_type,
            dtype=dtype)

        self.decoder_layers = ModuleList([
            DecoderLayer(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                num_attention_heads=num_heads,
                num_kv_heads=num_kv_heads if num_kv_heads else num_heads,
                max_position_embeddings=max_position_embeddings,
                q_scaling=q_scaling,
                has_attention_qkvo_bias=has_attention_qkvo_bias,
                has_mlp_bias=has_mlp_bias,
                layernorm_position=layernorm_position,
                layernorm_eps=layernorm_eps,
                layernorm_type=layernorm_type,
                hidden_act=hidden_act,
                tp_group=tp_group,
                tp_size=tp_size,
                dtype=dtype,
                residual_scaling=residual_scaling,
                relative_attention=relative_attention,
                max_distance=max_distance,
                num_buckets=num_buckets) for _ in range(num_layers)
        ])

        # e.g. BART false, T5 true
        if has_model_final_layernorm:
            self.final_layernorm = ln_type(normalized_shape=hidden_size,
                                           eps=layernorm_eps,
                                           dtype=dtype)

        self.lm_head = ColumnLinear(
            hidden_size,
            vocab_size,
            bias=has_lm_head_bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=True,
        )

    def forward(
        self,
        decoder_input_ids: Tensor,
        encoder_output: Tensor,
        position_ids=None,
        token_type_ids=None,
        use_cache=False,
        attention_mask=None,
        last_token_ids=None,
        kv_cache_params=None,
        attention_params=None,
    ):
        assert last_token_ids is not None, "Expecting last token ids to be not None"
        assert isinstance(decoder_input_ids, Tensor)

        hidden_states = self.embedding(decoder_input_ids, position_ids,
                                       token_type_ids)

        past_key_value = kv_cache_params.past_key_value
        if past_key_value is None:
            past_key_value = tuple([None] * len(self.decoder_layers))

        if use_cache:
            presents = []

        if attention_mask is not None:
            attention_mask = expand_mask(attention_mask,
                                         shape(decoder_input_ids, -1))

        for decoder_layer, past in zip(self.decoder_layers,
                                       kv_cache_params.past_key_value):
            hidden_states = decoder_layer(
                hidden_states,
                encoder_output=encoder_output,
                attention_mask=attention_mask,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=past,
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params,
            )

            if use_cache:
                presents_self, presents_cross = hidden_states[1], hidden_states[
                    2]
                presents.append((presents_self, presents_cross))
                hidden_states = hidden_states[0]

        if self.final_layernorm:
            hidden_states = self.final_layernorm(hidden_states)

        # [bs, seq, hidden_size] -> [bs, hidden_size]
        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        # Rescale output before projecting on vocab (for T5)
        # See https://github.com/huggingface/transformers/blob/0b192de1f353b0e04dad4813e02e2c672de077be/src/transformers/models/t5/modeling_t5.py#L1769-L1772
        # Note: this is specific for T5, to make it more generic, one can pass in a config:
        #   self.config.tie_word_embeddings - default to be True for T5
        hidden_states = hidden_states * (self.hidden_size**-0.5)

        # [bs, hidden_size] -> [bs, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self._logits_dtype)

        if use_cache:
            for i, present in enumerate(presents):
                present[0].mark_output(f'present_key_value_{i}', self._kv_dtype)
                present[1].mark_output(f'cross_present_key_value_{i}',
                                       self._kv_dtype)
            return (lm_logits, tuple(presents))

        return lm_logits

    def prepare_inputs(
        self,
        num_layers,
        max_batch_size,
        max_beam_width,
        max_input_len,
        max_new_tokens,
        max_encoder_input_len,
    ):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        max_output_len = max_input_len + max_new_tokens

        head_size = self.hidden_size // self.num_heads
        num_heads = self.num_heads // self.tp_size
        encoder_head_size = self.encoder_hidden_size // self.encoder_num_heads
        bb_range = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range = [1, 1, max_input_len
                       ]  # context phase >= 1, generation phase = 1
        encoder_inlen_range = [
            1, (max_encoder_input_len + 1) // 2, max_encoder_input_len
        ]
        mask_len_range = [1, (max_output_len + 1) // 2 + 1, max_output_len + 1]
        max_output_len_range = [0, (max_output_len + 1) // 2, max_output_len]

        num_tokens_range = [
            1,
            max_batch_size * max_beam_width,
            max(max_input_len * max_batch_size,
                max_beam_width * max_batch_size),
        ]

        # No enable_two_optimization_profiles support yet

        encoder_input_len_range = [
            0, (max_encoder_input_len + 1) // 2, max_encoder_input_len
        ]
        past_key_value = []
        sequence_length = None
        host_past_key_value_lengths = None
        attention_mask = None
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        remove_input_padding = default_net().plugin_config.remove_input_padding

        position_ids = None
        token_type_ids = None
        if remove_input_padding:
            input_ids = Tensor(name='input_ids',
                               dtype=trt.int32,
                               shape=[1, -1],
                               dim_range=OrderedDict([
                                   ('batch_size_fake', [1]),
                                   ('num_tokens', [num_tokens_range]),
                               ]))
            if self.has_position_embedding:
                position_ids = Tensor(name='position_ids',
                                      dtype=trt.int32,
                                      shape=[1, -1],
                                      dim_range=OrderedDict([
                                          ('batch_size_fake', [1]),
                                          ('num_tokens', [num_tokens_range]),
                                      ]))
            if self.has_token_type_embedding:
                token_type_ids = Tensor(
                    name='token_type_ids',
                    dtype=trt.int32,
                    shape=[1, -1],
                    dim_range=OrderedDict([('batch_size_fake', [1]),
                                           ('num_tokens', [num_tokens_range])]),
                )
        else:
            input_ids = Tensor(name='input_ids',
                               dtype=trt.int32,
                               shape=[-1, -1],
                               dim_range=OrderedDict([
                                   ('batch_size_beam_width', [bb_range]),
                                   ('input_len', [inlen_range]),
                               ]))
            if self.has_position_embedding:
                position_ids = Tensor(name='position_ids',
                                      dtype=trt.int32,
                                      shape=[-1, -1],
                                      dim_range=OrderedDict([
                                          ('batch_size_beam_width', [bb_range]),
                                          ('input_len', [inlen_range]),
                                      ]))
            if self.has_token_type_embedding:
                token_type_ids = Tensor(
                    name='token_type_ids',
                    dtype=trt.int32,
                    shape=[-1, -1],
                    dim_range=OrderedDict([('batch_size_beam_width', [bb_range
                                                                      ]),
                                           ('input_len', [inlen_range])]),
                )

        if use_gpt_attention_plugin:
            host_past_key_value_lengths = Tensor(
                name='host_past_key_value_lengths',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', [bb_range])]),
            )

        context_lengths = None
        host_context_lengths = None
        host_request_types = None
        if use_gpt_attention_plugin and remove_input_padding:
            host_context_lengths = Tensor(name='host_context_lengths',
                                          dtype=trt.int32,
                                          shape=[-1],
                                          dim_range=OrderedDict([
                                              ('batch_size_beam_width',
                                               [bb_range])
                                          ]))

        if use_gpt_attention_plugin:
            sequence_length = Tensor(
                name='sequence_length',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', [bb_range])]),
            )

            context_lengths = Tensor(name='context_lengths',
                                     dtype=trt.int32,
                                     shape=[-1],
                                     dim_range=OrderedDict([
                                         ('batch_size_beam_width', [bb_range])
                                     ]))
            host_request_types = Tensor(name='host_request_types',
                                        dtype=trt.int32,
                                        shape=[-1],
                                        dim_range=OrderedDict([
                                            ('batch_size_beam_width',
                                             [bb_range])
                                        ]))

        encoder_input_lengths = Tensor(
            name="encoder_input_lengths",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size", [bs_range])]),
        )
        encoder_max_input_length = Tensor(
            name="encoder_max_input_length",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("encoder_max_input_length",
                                    [encoder_inlen_range])]),
        )
        last_token_ids = Tensor(
            name="last_token_ids",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size_last_token_ids", [bb_range])]),
        )
        if not use_gpt_attention_plugin:
            attention_mask = Tensor(
                name='attention_mask',
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([
                    ('batch_size_beam_width', [bb_range]),
                    ('mask_len', [mask_len_range]),
                ]),
            )

        cache_indirection = Tensor(
            name='cache_indirection',
            dtype=trt.int32,
            shape=[-1, -1, -1],
            dim_range=OrderedDict([
                ('batch_size_cache', [bs_range]),
                ('beam_width', [beam_width_range]),
                ('max_seq_len', [max_output_len_range]),
            ]),
        )

        encoder_output = Tensor(
            name="encoder_output",
            dtype=self._dtype,
            shape=[-1, -1, self.encoder_hidden_size],
            dim_range=OrderedDict([
                ("batch_size", [bs_range]),
                ("encoder_input_len", [encoder_input_len_range]),
                ("encoder_hidden_size", [self.encoder_hidden_size]),
            ]),
        )

        for i in range(num_layers):
            kv_dim_range = OrderedDict([
                ('batch_size_beam_width', [bb_range]),
                ('kv', [2]),
                ('num_heads', [num_heads]),
                ('past_key_len', [max_output_len_range]),
                ('head_size', [head_size]),
            ])
            kv = Tensor(name=f'past_key_value_{i}',
                        dtype=self._kv_dtype,
                        shape=[-1, 2, num_heads, -1, head_size],
                        dim_range=kv_dim_range)

            cross_kv_dim_range = OrderedDict([
                ('batch_size_beam_width', [bb_range]),
                ('kv', [2]),
                ('cross_num_heads', [self.encoder_num_heads]),
                ('cross_past_key_len', [encoder_input_len_range]),
                ('cross_head_size', [encoder_head_size]),
            ])
            cross_kv = Tensor(
                name=f'cross_past_key_value_{i}',
                dtype=self._kv_dtype,
                shape=[-1, 2, self.encoder_num_heads, -1, encoder_head_size],
                dim_range=cross_kv_dim_range)
            past_key_value.append((kv, cross_kv))

            # TODO: Remove this when TRT fix the named dimension
            if not remove_input_padding:
                assertion(shape(input_ids, 0) == shape(kv, 0), 'batch size')

            kv_cache_params = KeyValueCacheParams(
                past_key_value=past_key_value,
                host_past_key_value_lengths=host_past_key_value_lengths,
                cache_indirection=cache_indirection)

            attention_params = AttentionParams(
                sequence_length=sequence_length,
                context_lengths=context_lengths,
                host_context_lengths=host_context_lengths,
                max_context_length=max_input_len,
                host_request_types=host_request_types,
                encoder_input_lengths=encoder_input_lengths,
                encoder_max_input_length=encoder_max_input_length,
            )

        return (input_ids, encoder_output, position_ids, token_type_ids, True,
                attention_mask, last_token_ids, kv_cache_params,
                attention_params)
