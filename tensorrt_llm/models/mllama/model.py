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
from typing import List, Optional, Union

import tensorrt as trt
import torch

from tensorrt_llm._common import default_net
from tensorrt_llm._utils import numpy_to_torch, str_dtype_to_torch
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.functional import (Conditional, LayerNormPositionType,
                                     LayerNormType, MLPType,
                                     PositionEmbeddingType, Tensor, assertion,
                                     gather_last_token_logits, maximum, minimum,
                                     recv, reduce, send, shape, tanh)
from tensorrt_llm.layers import (MLP, Attention, AttentionMaskParams,
                                 AttentionMaskType, AttentionParams,
                                 ColumnLinear, Embedding, FusedGatedMLP,
                                 GatedMLP, GroupNorm, KeyValueCacheParams,
                                 LayerNorm, LoraParams, RmsNorm)
from tensorrt_llm.lora_helper import (LoraConfig,
                                      get_default_trtllm_modules_to_hf_modules,
                                      use_lora)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
from tensorrt_llm.models.modeling_utils import PretrainedModel, QuantConfig
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.quantization import QuantMode

from .config import MLLaMAConfig

layernorm_map = {
    LayerNormType.LayerNorm: LayerNorm,
    LayerNormType.RmsNorm: RmsNorm,
    LayerNormType.GroupNorm: GroupNorm,
}

mlp_map = {
    MLPType.MLP: MLP,
    MLPType.GatedMLP: GatedMLP,
    MLPType.FusedGatedMLP: FusedGatedMLP,
}

ADD_DEBUG_TENSOR = False


class CrossAttentionTransformerBlock(Module):

    def __init__(
            self,
            *,
            local_layer_idx,
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            num_kv_heads,
            head_size,
            max_position_embeddings=None,
            q_scaling=1.0,
            has_attention_qkvo_bias=False,
            has_mlp_bias=False,
            layernorm_position=LayerNormPositionType.pre_layernorm,
            layernorm_type=LayerNormType.RmsNorm,
            layernorm_eps=1e-5,
            hidden_act="gated-silu",
            mlp_type=MLPType.GatedMLP,
            mapping=Mapping(),
            dtype=None,
            residual_scaling=1.0,
            relative_attention=False,
            max_distance=0,
            num_buckets=0,
            fp16_clamping=False,
            skip_cross_kv=False,
            use_implicit_relative_attention=False,
            rotary_embedding_base=None,
            rotary_embedding_scaling=None,
            quant_mode=QuantMode(0),
    ):
        super().__init__()
        self.local_layer_idx = local_layer_idx
        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        self.layernorm_position = layernorm_position
        assert self.layernorm_position == LayerNormPositionType.pre_layernorm

        self.cross_attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_size=head_size,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            q_scaling=q_scaling,
            bias=has_attention_qkvo_bias,
            attention_mask_type=AttentionMaskType.causal,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            tp_rank=mapping.tp_rank,
            dtype=dtype,
            cross_attention=True,
            relative_attention=
            False,  # Cross attention has no relative attention bias
            max_distance=max_distance,
            num_buckets=num_buckets,
            position_embedding_type=PositionEmbeddingType.
            learned_absolute,  # we don't use rope for cross attn
            skip_cross_kv=skip_cross_kv,
            qk_layernorm=True,
            layernorm_type=layernorm_type,
            quant_mode=quant_mode,
        )

        self.input_layernorm = ln_type(normalized_shape=hidden_size,
                                       eps=layernorm_eps,
                                       dtype=dtype)
        self.gate_attn = Parameter(shape=tuple((1, )), dtype=dtype)

        self.mlp_type = mlp_type
        mlp_f = mlp_map[mlp_type]

        self.mlp = mlp_f(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            hidden_act=hidden_act,
            bias=has_mlp_bias,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            dtype=dtype,
        )

        self.post_layernorm = ln_type(normalized_shape=hidden_size,
                                      eps=layernorm_eps,
                                      dtype=dtype)
        self.gate_ffwd = Parameter(shape=tuple((1, )), dtype=dtype)

        self.residual_scaling = residual_scaling

        self.fp16_clamping = fp16_clamping
        self.no_ffn = False

    def forward(self,
                hidden_states: Tensor,
                encoder_output: Optional[Tensor] = None,
                attention_mask_params=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                lora_layer_params=None,
                cross_kv_cache_gen: Optional[Tensor] = None,
                cross_kv_reuse: Optional[Tensor] = None,
                full_text_row_masked_out_mask: Tensor = None,
                skip_cross_attn_blocks: Tensor = None):

        assert isinstance(hidden_states, Tensor)

        if encoder_output:
            assert isinstance(encoder_output, Tensor)

        if ADD_DEBUG_TENSOR:
            hidden_states.mark_output(
                f'{self.local_layer_idx:2d}/1.0: hidden_states',
                hidden_states.dtype)
        # cross attention
        residual = hidden_states * self.residual_scaling

        # skip input_layernorm
        if skip_cross_attn_blocks is not None:
            input_ln_conditional = Conditional(skip_cross_attn_blocks)
            skip_result = input_ln_conditional.add_input(hidden_states)
            hidden_states = input_ln_conditional.add_input(hidden_states)
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = input_ln_conditional.add_output(
                skip_result, hidden_states)
        else:
            hidden_states = self.input_layernorm(hidden_states)

        if ADD_DEBUG_TENSOR:
            hidden_states.mark_output(
                f'{self.local_layer_idx:2d}/2.1: normed_input',
                hidden_states.dtype)
        # pass full_text_row_masked_out_mask and xattn_mask
        attention_output = self.cross_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask_params.cross_attention_mask,
            attention_packed_mask=attention_mask_params.
            cross_attention_packed_mask,
            encoder_output=encoder_output,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params,
            cross_kv_cache_gen=cross_kv_cache_gen,
            cross_kv_reuse=cross_kv_reuse,
            skip_attn=skip_cross_attn_blocks,
        )

        if use_cache:
            attention_output, presents_cross = attention_output
        if ADD_DEBUG_TENSOR:
            attention_output.mark_output(
                f'{self.local_layer_idx:2d}/3.1: cross_attention_output',
                attention_output.dtype)

        attn_residual_scale = tanh(self.gate_attn.value.cast(trt.float32)).cast(
            attention_output.dtype)

        attention_input = hidden_states
        hidden_states = residual + attn_residual_scale * attention_output

        # use to skip attention_output with residual
        # Since conditional does not work for gpt_attention_plugin, we replace the
        # attention_output by hidden_states (input of attention) now.
        if skip_cross_attn_blocks is not None:
            attn_conditional = Conditional(skip_cross_attn_blocks)
            skip_result = attn_conditional.add_input(attention_input)
            hidden_states = attn_conditional.add_input(hidden_states)
            hidden_states = attn_conditional.add_output(skip_result,
                                                        hidden_states)

        if ADD_DEBUG_TENSOR:
            hidden_states.mark_output(
                f'{self.local_layer_idx:2d}/3.2: cross_attn_output_with_residual',
                hidden_states.dtype)

        if self.fp16_clamping:
            hidden_states = maximum(-64000.0, hidden_states)
            hidden_states = minimum(64000.0, hidden_states)

        # MLP
        # skip post_layernorm and mlp
        if skip_cross_attn_blocks is not None:
            mlp_conditional = Conditional(skip_cross_attn_blocks)
            skip_case = mlp_conditional.add_input(hidden_states)
            hidden_states = mlp_conditional.add_input(hidden_states)

        attention_output = attention_output * full_text_row_masked_out_mask  # TODO should move this mask into attention?

        residual = hidden_states * self.residual_scaling

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)
        if ADD_DEBUG_TENSOR:
            hidden_states.mark_output(
                f'{self.local_layer_idx:2d}/4.1: mlp_output',
                hidden_states.dtype)

        hidden_states = hidden_states * full_text_row_masked_out_mask
        if ADD_DEBUG_TENSOR:
            hidden_states.mark_output(
                f'{self.local_layer_idx:2d}/4.2: masked_mlp_output',
                hidden_states.dtype)
        ffn_residual_scale = tanh(self.gate_ffwd.value.cast(trt.float32)).cast(
            hidden_states.dtype)
        hidden_states = residual + ffn_residual_scale * hidden_states * float(
            not self.no_ffn)

        if skip_cross_attn_blocks is not None:
            hidden_states = mlp_conditional.add_output(skip_case, hidden_states)

        if self.fp16_clamping:
            hidden_states = maximum(-64000.0, hidden_states)
            hidden_states = minimum(64000.0, hidden_states)

        if ADD_DEBUG_TENSOR:
            hidden_states.mark_output(
                f'{self.local_layer_idx:2d}/4.4: transformer_out',
                hidden_states.dtype)

        if use_cache:
            return (hidden_states, presents_cross)
        return hidden_states


class TransformerBlock(Module):

    def __init__(
            self,
            *,
            local_layer_idx,
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            num_kv_heads,
            head_size,
            max_position_embeddings=None,
            q_scaling=1.0,
            has_attention_qkvo_bias=False,
            has_mlp_bias=False,
            layernorm_position=LayerNormPositionType.pre_layernorm,
            layernorm_type=LayerNormType.RmsNorm,
            layernorm_eps=1e-5,
            hidden_act="gated-silu",
            mlp_type=MLPType.GatedMLP,
            mapping=Mapping(),
            dtype=None,
            residual_scaling=1.0,
            relative_attention=False,
            max_distance=0,
            num_buckets=0,
            fp16_clamping=False,
            skip_cross_kv=False,
            use_implicit_relative_attention=False,
            rotary_embedding_base=None,
            rotary_embedding_scaling=None,
            quant_mode=QuantMode(0),
    ):
        super().__init__()
        self.local_layer_idx = local_layer_idx
        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        self.layernorm_position = layernorm_position
        assert self.layernorm_position == LayerNormPositionType.pre_layernorm

        self.self_attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_size=head_size,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            q_scaling=q_scaling,
            bias=has_attention_qkvo_bias,
            attention_mask_type=AttentionMaskType.causal,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            tp_rank=mapping.tp_rank,
            dtype=dtype,
            cross_attention=False,
            relative_attention=relative_attention,
            max_distance=max_distance if use_implicit_relative_attention else 0,
            num_buckets=num_buckets,
            position_embedding_type=PositionEmbeddingType.relative
            if relative_attention else PositionEmbeddingType.rope_gpt_neox,
            use_implicit_relative_attention=use_implicit_relative_attention,
            rotary_embedding_base=rotary_embedding_base,
            rotary_embedding_scaling=rotary_embedding_scaling,
            quant_mode=quant_mode,
        )

        self.input_layernorm = ln_type(normalized_shape=hidden_size,
                                       eps=layernorm_eps,
                                       dtype=dtype)

        self.mlp_type = mlp_type
        mlp_f = mlp_map[mlp_type]
        self.mlp = mlp_f(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            hidden_act=hidden_act,
            bias=has_mlp_bias,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            dtype=dtype,
        )

        self.post_layernorm = ln_type(normalized_shape=hidden_size,
                                      eps=layernorm_eps,
                                      dtype=dtype)

        self.residual_scaling = residual_scaling

        self.fp16_clamping = fp16_clamping

    def forward(
        self,
        hidden_states: Tensor,
        encoder_output: Optional[Tensor] = None,  # not used
        attention_mask_params=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
        lora_layer_params=None,
        cross_kv_cache_gen: Optional[Tensor] = None,
        cross_kv_reuse: Optional[Tensor] = None,
        full_text_row_masked_out_mask: Tensor = None,  # not used
        skip_cross_attn_blocks=None,
    ):
        assert isinstance(hidden_states, Tensor)

        # self-attention
        residual = hidden_states * self.residual_scaling
        if ADD_DEBUG_TENSOR:
            hidden_states.mark_output(
                f'{self.local_layer_idx:2d}/1.0: hidden_states',
                hidden_states.dtype)

        hidden_states = self.input_layernorm(hidden_states)
        if ADD_DEBUG_TENSOR:
            hidden_states.mark_output(
                f'{self.local_layer_idx:2d}/2.1: normed attn_input',
                hidden_states.dtype)

        attention_output = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask_params.self_attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params)

        if use_cache:
            attention_output, presents_self = attention_output

        if ADD_DEBUG_TENSOR:
            attention_output.mark_output(
                f'{self.local_layer_idx:2d}/3.1: self_attention_output',
                attention_output.dtype)

        hidden_states = residual + attention_output
        if ADD_DEBUG_TENSOR:
            hidden_states.mark_output(
                f'{self.local_layer_idx:2d}/3.1: attention_output_with_residual',
                hidden_states.dtype)

        if self.fp16_clamping:
            hidden_states = maximum(-64000.0, hidden_states)
            hidden_states = minimum(64000.0, hidden_states)

        # MLP
        residual = hidden_states * self.residual_scaling

        hidden_states = self.post_layernorm(hidden_states)
        if ADD_DEBUG_TENSOR:
            hidden_states.mark_output(
                f'{self.local_layer_idx:2d}/3.2: normed_mlp_input',
                hidden_states.dtype)

        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)

        if ADD_DEBUG_TENSOR:
            hidden_states.mark_output(
                f'{self.local_layer_idx:2d}/4.1: mlp_output',
                hidden_states.dtype)

        hidden_states = residual + hidden_states
        if ADD_DEBUG_TENSOR:
            hidden_states.mark_output(
                f'{self.local_layer_idx:2d}/4.2: mlp_output_residual',
                hidden_states.dtype)

        if self.fp16_clamping:
            hidden_states = maximum(-64000.0, hidden_states)
            hidden_states = minimum(64000.0, hidden_states)

        if use_cache:
            return (hidden_states, presents_self)
        return hidden_states


class MLLaMAModel(Module):

    def __init__(self, config: MLLaMAConfig) -> None:
        super().__init__()
        self.config = config
        self.position_embedding_type = config.position_embedding_type

        self.mapping = self.config.mapping

        self.layernorm_type = self.config.layernorm_type
        ln_type = layernorm_map[self.layernorm_type]

        self.has_attention_qkvo_bias = self.config.has_attention_qkvo_bias
        self.has_mlp_bias = self.config.has_mlp_bias

        self.has_model_final_layernorm = self.config.has_model_final_layernorm
        self._dtype = self.config.dtype
        # no quantization considered for now
        self._kv_dtype = self._dtype
        self._logits_dtype = self.config.logits_dtype

        self.total_num_layers = self.config.num_hidden_layers
        self.num_layers = self.config.num_hidden_layers // self.mapping.pp_size

        self.hidden_size = self.config.hidden_size
        self.encoder_hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        # num_kv_heads = self.num_heads
        num_kv_heads = self.config.num_key_value_heads
        if num_kv_heads is None or num_kv_heads <= 0:
            num_kv_heads = self.num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = self.hidden_size // self.num_heads if self.config.head_size is None else self.config.head_size

        self.fp16_clamping = False

        self.skip_cross_kv = self.config.skip_cross_kv
        self.mlp_type = MLPType.MLP if not hasattr(
            self.config, "mlp_type") else self.config.mlp_type
        self.use_implicit_relative_attention = self.config.use_implicit_relative_attention if hasattr(
            self.config, "use_implicit_relative_attention") else False

        self.cross_attention_layers = self.config.cross_attention_layers

        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(
                self.config.embed_vocab_size,
                self.config.hidden_size,
                dtype=self._dtype,
                tp_size=self.mapping.tp_size
                if self.config.use_parallel_embedding else 1,
                tp_group=self.mapping.tp_group
                if self.config.use_parallel_embedding else None,
                sharding_dim=self.config.embedding_sharding_dim,
                tp_rank=self.mapping.tp_rank)

        layers_range = self.mapping.pp_layers(self.total_num_layers)
        _layers = []
        for layer_idx in layers_range:
            local_layer_idx = layer_idx - layers_range[0]
            args = {
                "local_layer_idx": local_layer_idx,
                "hidden_size": self.config.hidden_size,
                "ffn_hidden_size": self.config.intermediate_size,
                "num_attention_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_size": self.head_size,
                "max_position_embeddings": self.config.max_position_embeddings,
                "layernorm_position": self.config.layernorm_position,
                "layernorm_eps": self.config.norm_epsilon,
                "layernorm_type": self.config.layernorm_type,
                "hidden_act": self.config.hidden_act,
                "mlp_type": self.mlp_type,
                "mapping": self.mapping,
                "dtype": self._dtype,
                "residual_scaling": self.config.residual_scaling,
                "max_distance": self.config.max_distance,
                "num_buckets": self.config.num_buckets,
                "fp16_clamping": self.fp16_clamping,
                "skip_cross_kv": self.skip_cross_kv,
                "rotary_embedding_base": self.config.rotary_base,
                "rotary_embedding_scaling": self.config.rotary_scaling,
                "quant_mode": self.config.quant_mode,
            }
            if layer_idx in self.cross_attention_layers:
                assert layers_range[0] == 0, "not support PP now"
                _layers.append(CrossAttentionTransformerBlock(**args))
            else:
                _layers.append(TransformerBlock(**args))

        self.layers = ModuleList(_layers)

        if self.mapping.is_last_pp_rank():
            self.ln_f = None
            if self.has_model_final_layernorm:
                self.ln_f = ln_type(normalized_shape=self.config.hidden_size,
                                    eps=self.config.norm_epsilon,
                                    dtype=self.config.dtype)

        if self.config.relative_attention and not self.use_implicit_relative_attention:
            self.rel_attn_table = Parameter(
                shape=(self.config.num_attention_heads // self.mapping.tp_size,
                       self.config.num_buckets),
                dtype=self._dtype)

    def forward(
        self,
        decoder_input_ids: Tensor,
        encoder_output: Tensor,
        use_cache=False,
        attention_mask_params=None,
        last_token_ids=None,
        kv_cache_params=None,
        attention_params=None,
        hidden_states=None,
        lora_params: LoraParams = None,
        cross_kv_cache_gen: Optional[Tensor] = None,
        cross_kv_reuse: Optional[Tensor] = None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
        skip_cross_attn_blocks: Optional[Tensor] = None,
    ):
        if self.mapping.is_first_pp_rank():
            assert isinstance(decoder_input_ids, Tensor)
        else:
            assert isinstance(hidden_states, Tensor)

        # In PP, layer 0 has ids as inputs, all other layers have hidden_states as inputs
        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(decoder_input_ids)
            self.register_network_output('embedding_layer_output',
                                         hidden_states)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        full_text_row_masked_out_mask = reduce(
            (attention_mask_params.cross_attention_mask).cast(
                hidden_states.dtype),
            trt.ReduceOperation.MAX,
            dim=-1,
            keepdim=True)
        if ADD_DEBUG_TENSOR:
            full_text_row_masked_out_mask.mark_output(
                "full_text_row_masked_out_mask",
                full_text_row_masked_out_mask.dtype)

        cross_attention_mask_type = attention_mask_params.cross_attention_mask.dtype
        attention_mask_params.cross_attention_mask = (
            attention_mask_params.cross_attention_mask.cast(
                full_text_row_masked_out_mask.dtype) *
            full_text_row_masked_out_mask).cast(cross_attention_mask_type)

        invert_mask = 1.0 - attention_mask_params.cross_attention_mask.cast(
            hidden_states.dtype)
        invert_full_text_row_masked_out_mask = 1.0 - full_text_row_masked_out_mask
        final_mask = invert_mask - invert_full_text_row_masked_out_mask
        attention_mask_params.cross_attention_mask = final_mask.cast(
            cross_attention_mask_type)
        if ADD_DEBUG_TENSOR:
            attention_mask_params.cross_attention_mask.mark_output(
                "attention_mask_params.cross_attention_mask",
                attention_mask_params.cross_attention_mask.dtype)

        if use_cache:
            presents = []
        for i, (decoder_layer, past) in enumerate(
                zip(self.layers, kv_cache_params.past_key_value)):

            lora_layer_params = None
            if lora_params is not None and lora_params.lora_ranks is not None:
                lora_layer_params = lora_params.get_layer_params(i)
            hidden_states = decoder_layer(
                hidden_states,
                encoder_output=encoder_output,
                attention_mask_params=attention_mask_params,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=past,
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=kv_cache_params.
                    host_max_attention_window_sizes,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    cache_indirection=kv_cache_params.cache_indirection,
                    kv_cache_block_offsets=kv_cache_params.
                    kv_cache_block_offsets,
                    host_kv_cache_block_offsets=kv_cache_params.
                    host_cross_kv_cache_block_offsets,
                    host_kv_cache_pool_pointers=kv_cache_params.
                    host_kv_cache_pool_pointers,
                    host_kv_cache_pool_mapping=kv_cache_params.
                    host_kv_cache_pool_mapping,
                    cross_kv_cache_block_offsets=kv_cache_params.
                    cross_kv_cache_block_offsets,
                    host_cross_kv_cache_block_offsets=kv_cache_params.
                    host_cross_kv_cache_block_offsets,
                    host_cross_kv_cache_pool_pointers=kv_cache_params.
                    host_cross_kv_cache_pool_pointers,
                    host_cross_kv_cache_pool_mapping=kv_cache_params.
                    host_cross_kv_cache_pool_mapping,
                ),
                skip_cross_attn_blocks=skip_cross_attn_blocks if isinstance(
                    decoder_layer, CrossAttentionTransformerBlock) else None,
                attention_params=attention_params,
                lora_layer_params=lora_layer_params,
                cross_kv_cache_gen=cross_kv_cache_gen,
                cross_kv_reuse=cross_kv_reuse,
                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            )

            if use_cache:
                present = hidden_states[1]
                presents.append((present))
                hidden_states = hidden_states[0]

        if self.mapping.is_last_pp_rank():
            if self.ln_f:
                hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states

    def precompute_relative_attention_bias(self, build_config):
        if self.config.relative_attention and not self.use_implicit_relative_attention:
            relative_attention_bias_builder = torch.ops.tensorrt_llm.relative_attention_bias
            rel_attn_precomputed = torch.zeros(
                (self.config.num_attention_heads // self.mapping.tp_size,
                 build_config.max_seq_len + 1, build_config.max_seq_len + 1),
                dtype=str_dtype_to_torch(self.config.dtype),
                device='cuda')
            rel_attn_table = numpy_to_torch(
                self.rel_attn_table.raw_value).to('cuda')
            relative_attention_bias_builder(
                rel_attn_precomputed,
                rel_attn_table,
                self.config.num_attention_heads // self.mapping.tp_size,
                build_config.max_seq_len,
                self.config.num_buckets,
                False,
                self.config.max_distance,
            )
            for layer_idx in range(self.num_layers):
                self.layers[layer_idx].self_attention.set_rel_attn_table(
                    build_config.max_seq_len, rel_attn_precomputed)


# TODO try to inherit the DecoderModelForCausalLM
class MLLaMAForCausalLM(PretrainedModel):
    config_class = MLLaMAConfig

    def __init__(self, config: MLLaMAConfig):
        super().__init__(config)
        Attention.create_attention_const_params(self, config)
        self.position_embedding_type = config.position_embedding_type
        self.transformer = MLLaMAModel(config)

        self.mapping = self.config.mapping

        self.has_model_final_layernorm = self.config.has_model_final_layernorm
        self._dtype = self.config.dtype
        self._kv_dtype = self._dtype
        self._logits_dtype = self.config.logits_dtype

        if self.mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=False if not hasattr(self.config, "has_lm_head_bias") else
                self.config.has_lm_head_bias,
                dtype=self.config.dtype,
                tp_group=self.config.mapping.tp_group,
                tp_size=self.config.mapping.tp_size,
                gather_output=True,
            )

        self.trtllm_modules_to_hf_modules = {
            **get_default_trtllm_modules_to_hf_modules(),
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_dense": "self_attn.o_proj",
            "cross_attn_q": "encoder_attn.q_proj",
            "cross_attn_k": "encoder_attn.k_proj",
            "cross_attn_v": "encoder_attn.v_proj",
            "cross_attn_dense": "encoder_attn.o_proj",
        }

    def forward(
        self,
        decoder_input_ids: Tensor,
        encoder_output: Tensor,
        use_cache=False,
        attention_mask_params=None,
        last_token_ids=None,
        kv_cache_params=None,
        attention_params=None,
        hidden_states=None,
        lora_params: LoraParams = None,
        cross_kv_cache_gen: Optional[Tensor] = None,
        cross_kv_reuse: Optional[Tensor] = None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
        skip_cross_attn_blocks: Optional[Tensor] = None,
    ):
        if self.mapping.is_first_pp_rank():
            assert isinstance(decoder_input_ids, Tensor)
        else:
            assert isinstance(hidden_states, Tensor)
        attention_params = Attention.fill_attention_params(
            self, attention_params)
        hidden_states = self.transformer(
            decoder_input_ids=decoder_input_ids,
            encoder_output=encoder_output,
            use_cache=use_cache,
            attention_mask_params=attention_mask_params,
            last_token_ids=last_token_ids,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            hidden_states=hidden_states,
            lora_params=lora_params,
            cross_kv_cache_gen=cross_kv_cache_gen,
            cross_kv_reuse=cross_kv_reuse,
            prompt_embedding_table=prompt_embedding_table,
            prompt_tasks=prompt_tasks,
            prompt_vocab_size=prompt_vocab_size,
            skip_cross_attn_blocks=skip_cross_attn_blocks,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            # [bs, seq, hidden_size] or [num_tokens, hidden_size] -> [bs, hidden_size]
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids,
                default_net().plugin_config.remove_input_padding)

            # [bs, hidden_size] -> [bs, vocab_size]
            lm_logits = self.lm_head(hidden_states)
            lm_logits.mark_output(f'logits', self._logits_dtype)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())
            hidden_states.mark_output(f'hidden_states_output', self._dtype)

        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in zip(self.mapping.pp_layers(self.total_num_layers),
                                  presents):
                present[0].mark_output(f'present_key_value_{i}', self._kv_dtype)
                if default_net().plugin_config.gpt_attention_plugin:
                    present[1].mark_output(f'cross_present_key_value_{i}',
                                           self._kv_dtype)
            if self.mapping.is_last_pp_rank():
                return (lm_logits, tuple(presents))
            return (hidden_states, tuple(presents))
        else:
            if self.mapping.is_last_pp_rank():
                return lm_logits
            return hidden_states

    def prepare_inputs(self,
                       max_batch_size,
                       max_beam_width,
                       max_decoder_input_len,
                       max_seq_len,
                       max_encoder_input_len,
                       gather_context_logits: bool = False,
                       gather_generation_logits: bool = False,
                       lora_target_modules: List[str] = None,
                       prompt_embedding_table_size: int = 0,
                       use_cache=True,
                       *args,
                       **kwargs):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        max_output_len = max_decoder_input_len + max_seq_len

        head_size = self.transformer.head_size
        num_kv_heads = (self.transformer.num_kv_heads + self.mapping.tp_size -
                        1) // self.mapping.tp_size

        encoder_head_size = head_size
        encoder_num_kv_heads = num_kv_heads

        bb_range = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range = [
            1, 1, max_decoder_input_len
        ]  # context phase >= 1 (if forced_input_ids), generation phase = 1
        encoder_inlen_range = [
            1, (max_encoder_input_len + 1) // 2, max_encoder_input_len
        ]
        mask_len_range = [1, (max_output_len + 1) // 2 + 1, max_output_len + 1]
        max_output_len_range = [0, (max_output_len + 1) // 2, max_output_len]

        encoder_num_tokens_range = [
            0,  # 0 for generation phase, >0 for context phase
            (max_encoder_input_len * max_batch_size + 1) // 2,
            max_encoder_input_len * max_batch_size,
        ]
        decoder_num_tokens_range = [
            1,
            max_batch_size * max_beam_width,
            max(max_decoder_input_len * max_batch_size,
                max_beam_width * max_batch_size),
        ]

        # No enable_two_optimization_profiles support yet
        encoder_input_len_range = [
            0,  # 0 for generation phase, >0 for context phase
            (max_encoder_input_len + 1) // 2,
            max_encoder_input_len
        ]
        # pack masks into bits (store as int32).
        max_cross_packed_mask_dim0 = max_batch_size * (
            (max_decoder_input_len + 128 - 1) // 128) * 128
        max_cross_packed_mask_dim1 = (
            (max_encoder_input_len + 256 - 1) // 256) * 256 // 32
        cross_packed_mask_dim0_range = [
            1, (max_cross_packed_mask_dim0 + 1) // 2, max_cross_packed_mask_dim0
        ]
        cross_packed_mask_dim1_range = [
            0,  # 0 for generation phase, >0 for context phase
            (max_cross_packed_mask_dim1 + 1) // 2,
            max_cross_packed_mask_dim1
        ]
        past_key_value = []
        sequence_length = None
        host_past_key_value_lengths = None
        attention_mask_params = AttentionMaskParams()
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        remove_input_padding = default_net().plugin_config.remove_input_padding
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        use_lora_plugin = default_net().plugin_config.lora_plugin
        kv_cache_type = None
        if not use_cache:
            kv_cache_type = KVCacheType.DISABLED
        else:
            if paged_kv_cache:
                kv_cache_type = KVCacheType.PAGED
            else:
                kv_cache_type = KVCacheType.CONTINUOUS

        input_ids, hidden_states = None, None
        if remove_input_padding:
            if self.mapping.is_first_pp_rank():
                input_ids = Tensor(name='input_ids',
                                   dtype=trt.int32,
                                   shape=[-1],
                                   dim_range=OrderedDict([
                                       ('decoder_num_tokens',
                                        [decoder_num_tokens_range]),
                                   ]))
            else:
                hidden_states = Tensor(name='hidden_states_input',
                                       dtype=self._dtype,
                                       shape=[-1, self.hidden_size],
                                       dim_range=OrderedDict([
                                           ('decoder_num_tokens',
                                            [decoder_num_tokens_range]),
                                           ('hidden_size', [self.hidden_size]),
                                       ]))
        else:
            if self.mapping.is_first_pp_rank():
                input_ids = Tensor(name='input_ids',
                                   dtype=trt.int32,
                                   shape=[-1, -1],
                                   dim_range=OrderedDict([
                                       ('batch_size_beam_width', [bb_range]),
                                       ('input_len', [inlen_range]),
                                   ]))
            else:
                hidden_states = Tensor(name='hidden_states_input',
                                       dtype=self._dtype,
                                       shape=[-1, -1, self.hidden_size],
                                       dim_range=OrderedDict([
                                           ('batch_size_beam_width', [bb_range
                                                                      ]),
                                           ('input_len', [inlen_range]),
                                           ('hidden_size', [self.hidden_size]),
                                       ]))

        encoder_input_lengths = Tensor(
            name="encoder_input_lengths",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size_beam_width", [bb_range])]),
        )
        encoder_max_input_length = Tensor(
            name="encoder_max_input_length",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("encoder_max_input_length",
                                    [encoder_inlen_range])]),
        )
        if remove_input_padding:
            encoder_output = Tensor(
                name="encoder_output",
                dtype=self._dtype,
                shape=[-1, self.config.hidden_size],
                dim_range=OrderedDict([
                    ("encoder_num_tokens", [encoder_num_tokens_range]),
                    ("hidden_size", [self.config.hidden_size]),
                ]),
            )
        else:
            encoder_output = Tensor(
                name="encoder_output",
                dtype=self._dtype,
                shape=[-1, -1, self.config.hidden_size],
                dim_range=OrderedDict([
                    ("batch_size_beam_width_encoder", [bb_range]),
                    ("encoder_input_len", [encoder_input_len_range]),
                    ("hidden_size", [self.config.hidden_size]),
                ]),
            )

        context_lengths = None
        host_context_lengths = None
        host_request_types = None
        host_runtime_perf_knobs = None
        host_context_progress = None
        if use_gpt_attention_plugin and remove_input_padding:
            host_context_lengths = Tensor(name='host_context_lengths',
                                          dtype=trt.int32,
                                          shape=[-1],
                                          dim_range=OrderedDict([
                                              ('batch_size_beam_width',
                                               [bb_range])
                                          ]))

        if use_gpt_attention_plugin:
            if kv_cache_type != KVCacheType.DISABLED:
                sequence_length = Tensor(
                    name='sequence_length',
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict([('batch_size_beam_width', [bb_range])
                                           ]),
                )
                host_past_key_value_lengths = Tensor(
                    name='host_past_key_value_lengths',
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict([('batch_size_beam_width', [bb_range])
                                           ]),
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

            host_runtime_perf_knobs = Tensor(name='host_runtime_perf_knobs',
                                             dtype=trt.int64,
                                             shape=[16],
                                             dim_range=OrderedDict([
                                                 ('perf_knob_size', [16])
                                             ]))

            host_context_progress = Tensor(name='host_context_progress',
                                           dtype=trt.int64,
                                           shape=[1],
                                           dim_range=OrderedDict([
                                               ('context_progress_size', [1])
                                           ]))

        last_token_ids = None
        if self.mapping.is_last_pp_rank() and not gather_context_logits:
            last_token_ids = Tensor(
                name="last_token_ids",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([("batch_size_last_token_ids", [bb_range])
                                       ]),
            )

        attention_mask = None
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
            assert False, "not support non-attention-plugin case now"

        cross_attention_mask = Tensor(
            name='cross_attention_mask',
            dtype=trt.bool,
            shape=[-1, -1],
            dim_range=OrderedDict([
                ('decoder_num_tokens_2',
                 [decoder_num_tokens_range
                  ]),  # TODO should use same name as input_ids
                ('encoder_input_len_2', [encoder_input_len_range]),
            ]),
        )

        cross_attention_packed_mask = Tensor(
            name='cross_attention_packed_mask',
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict([
                ('cross_packed_mask_dim0', [cross_packed_mask_dim0_range]),
                ('cross_packed_mask_dim1', [cross_packed_mask_dim1_range]),
            ]),
        )

        # create the attention_mask_params.
        attention_mask_params = AttentionMaskParams(
            attention_mask, None, cross_attention_mask,
            cross_attention_packed_mask)

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

        layers_range = self.mapping.pp_layers(self.transformer.total_num_layers)
        num_pp_layers = len(layers_range)

        host_max_attention_window_sizes = None
        host_sink_token_length = None
        if use_gpt_attention_plugin:
            host_max_attention_window_sizes = Tensor(
                name=f'host_max_attention_window_sizes',
                dtype=trt.int32,
                shape=[num_pp_layers],
                dim_range=OrderedDict([('num_layers', [num_pp_layers])]))
            host_sink_token_length = Tensor(name='host_sink_token_length',
                                            dtype=trt.int32,
                                            shape=[1],
                                            dim_range=OrderedDict([('scalar',
                                                                    [1])]))
        # TODO LoRA for mllama is not verified.
        lora_weights_pointers = None
        lora_ranks = None
        lora_params = None
        if use_lora_plugin:
            lora_weights_pointers = []
            lora_ranks = []
            missing_qkv_modules = []
            if any(x in lora_target_modules
                   for x in ["attn_q", "attn_k", "attn_v"]):
                for lora_module in [
                        "attn_q",
                        "attn_k",
                        "attn_v",
                ]:
                    if lora_module not in lora_target_modules:
                        missing_qkv_modules.append(lora_module)
            if any(x in lora_target_modules
                   for x in ["cross_attn_q", "cross_attn_k", "cross_attn_v"]):
                for lora_module in [
                        "cross_attn_q", "cross_attn_k", "cross_attn_v"
                ]:
                    if lora_module not in lora_target_modules:
                        missing_qkv_modules.append(lora_module)

            # For LoRA
            for i in layers_range:
                lora_weight_pointer_dict = {}
                lora_rank_dict = {}
                for lora_module in (lora_target_modules + missing_qkv_modules):
                    lora_weight_pointer = Tensor(
                        name=f'{lora_module}_lora_weights_pointers_{i}',
                        dtype=trt.int64,
                        shape=[-1, 3],
                        dim_range=OrderedDict([('batch_size_beam_width',
                                                [bb_range]),
                                               ('in_out_scales', [3])]))
                    lora_weight_pointer_dict.update({
                        f'{lora_module}_lora_weights_pointers':
                        lora_weight_pointer
                    })

                    lora_rank = Tensor(name=f'{lora_module}_lora_ranks_{i}',
                                       dtype=trt.int32,
                                       shape=[-1],
                                       dim_range=OrderedDict([
                                           ('batch_size_beam_width', [bb_range])
                                       ]))
                    lora_rank_dict.update(
                        {f'{lora_module}_lora_ranks': lora_rank})

                lora_weights_pointers.append(lora_weight_pointer_dict)
                lora_ranks.append(lora_rank_dict)

            # For cross attention, we need to use encoder_input_lengths (in CPU) to pass
            # as the host_context_lengths to the lora_plugin. But for self attention, we
            # should keep using the original host_context_lengths. Therefore, we keep both
            # of them in the lora_params.
            host_encoder_input_lengths = None
            if remove_input_padding:
                host_encoder_input_lengths = Tensor(
                    name="host_encoder_input_lengths",
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict([("batch_size_beam_width", [bb_range])
                                           ]),
                )

            lora_params = LoraParams(
                lora_ranks=lora_ranks,
                lora_weights_pointers=lora_weights_pointers,
                host_context_lengths=host_context_lengths,
                max_context_length=max_decoder_input_len,
                max_encoder_context_length=max_encoder_input_len,
                host_request_types=host_request_types,
                host_encoder_input_lengths=host_encoder_input_lengths,
            )

        kv_cache_block_offsets = None
        host_kv_cache_block_offsets = None
        host_kv_cache_pool_pointers = None
        host_kv_cache_pool_mapping = None

        cross_kv_cache_block_offsets = None
        host_cross_kv_cache_block_offsets = None
        host_cross_kv_cache_pool_pointers = None
        host_cross_kv_cache_pool_mapping = None

        if use_cache:
            if not paged_kv_cache:
                for i in layers_range:
                    kv_dim_range = OrderedDict([
                        ('batch_size_beam_width', [bb_range]),
                        ('kv', [2]),
                        ('num_heads', [num_kv_heads]),
                        ('past_key_len', [max_output_len_range]),
                        ('head_size', [head_size]),
                    ])
                    kv = Tensor(name=f'past_key_value_{i}',
                                dtype=self._kv_dtype,
                                shape=[-1, 2, num_kv_heads, -1, head_size],
                                dim_range=kv_dim_range)

                    past_key_value.append(kv)

                if i in self.transformer.cross_attention_layers:
                    xa_layer_id = self.transformer.cross_attention_layers.index(
                        i) + layers_range[-1]
                    cross_kv_dim_range = OrderedDict([
                        ('batch_size_beam_width', [bb_range]),
                        ('kv', [2]),
                        ('cross_num_heads', [encoder_num_kv_heads]),
                        ('cross_past_key_len', [encoder_input_len_range]),
                        ('cross_head_size', [encoder_head_size]),
                    ])
                    cross_kv = Tensor(
                        name=f'cross_past_key_value_{xa_layer_id}',
                        dtype=self._kv_dtype,
                        shape=[
                            -1, 2, encoder_num_kv_heads, -1, encoder_head_size
                        ],
                        dim_range=cross_kv_dim_range)
                    past_key_value.append(kv)

                # TODO: Remove this when TRT fix the named dimension
                if not remove_input_padding:
                    assertion(
                        shape(
                            input_ids if self.mapping.is_first_pp_rank() else
                            hidden_states, 0) == shape(kv, 0), 'batch size')

            else:  # paged_kv_cache == True
                # PagedKV setup for KV cache of self-attention
                max_blocks_per_seq_range = [[
                    math.ceil(max_output_len_range[0] / tokens_per_block),
                    math.ceil(max_output_len_range[1] / tokens_per_block),
                    math.ceil(max_output_len_range[2] / tokens_per_block)
                ]]
                max_blocks_per_seq_range = [[
                    x for x in max_blocks_per_seq_range[0]
                ]]

                # PagedKV setup for KV cache of cross-attention
                max_cross_blocks_per_seq_range = [[
                    math.ceil(encoder_input_len_range[0] / tokens_per_block),
                    math.ceil(encoder_input_len_range[1] / tokens_per_block),
                    math.ceil(encoder_input_len_range[2] / tokens_per_block)
                ]]
                max_cross_blocks_per_seq_range = [[
                    x for x in max_cross_blocks_per_seq_range[0]
                ]]

                num_kv_cache_pools = 2

                kv_cache_block_offsets = Tensor(
                    name=f'kv_cache_block_offsets',
                    dtype=trt.int32,
                    shape=[num_kv_cache_pools, -1, 2, -1],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', [num_kv_cache_pools]),
                        ('batch_size_beam_width', [bb_range]),
                        ('kv', [2]),
                        ('max_blocks_per_seq', max_blocks_per_seq_range),
                    ]))
                host_kv_cache_block_offsets = Tensor(
                    name=f'host_kv_cache_block_offsets',
                    dtype=trt.int32,
                    shape=[num_kv_cache_pools, -1, 2, -1],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', [num_kv_cache_pools]),
                        ('batch_size_beam_width', [bb_range]),
                        ('kv', [2]),
                        ('max_blocks_per_seq', max_blocks_per_seq_range),
                    ]))
                host_kv_cache_pool_pointers = Tensor(
                    name=f'host_kv_cache_pool_pointers',
                    dtype=trt.int64,
                    shape=[num_kv_cache_pools, 2],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', [num_kv_cache_pools]),
                        ('num_pools', [2]),
                    ]))
                host_kv_cache_pool_mapping = Tensor(
                    name=f"host_kv_cache_pool_mapping",
                    dtype=trt.int32,
                    # 2: (Index of pool, Index of layer within pool)
                    shape=[num_pp_layers, 2],
                    dim_range=OrderedDict([
                        ('pools_mapping', [num_pp_layers]),
                        ('layer_cache_pool_locator', [2]),
                    ]))

                # paged blocks for cross kv
                cross_kv_cache_block_offsets = Tensor(
                    name=f'cross_kv_cache_block_offsets',
                    dtype=trt.int32,
                    shape=[num_kv_cache_pools, -1, 2, -1],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', [num_kv_cache_pools]),
                        ('batch_size_beam_width', [bb_range]),
                        ('kv', [2]),
                        ('max_cross_blocks_per_seq',
                         max_cross_blocks_per_seq_range),
                    ]))
                host_cross_kv_cache_block_offsets = Tensor(
                    name=f'host_cross_kv_cache_block_offsets',
                    dtype=trt.int32,
                    shape=[num_kv_cache_pools, -1, 2, -1],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', [num_kv_cache_pools]),
                        ('batch_size_beam_width', [bb_range]),
                        ('kv', [2]),
                        ('max_cross_blocks_per_seq',
                         max_cross_blocks_per_seq_range),
                    ]))
                host_cross_kv_cache_pool_pointers = Tensor(
                    name=f'host_cross_kv_cache_pool_pointers',
                    dtype=trt.int64,
                    shape=[num_kv_cache_pools, 2],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', [num_kv_cache_pools]),
                        ('num_pools', [2]),
                    ]))
                host_cross_kv_cache_pool_mapping = Tensor(
                    name=f"host_cross_kv_cache_pool_mapping",
                    dtype=trt.int32,
                    # 2: (Index of pool, Index of layer within pool)
                    shape=[num_pp_layers, 2],
                    dim_range=OrderedDict([
                        ('pools_mapping', [num_pp_layers]),
                        ('layer_cache_pool_locator', [2]),
                    ]))

                for i in layers_range:
                    past_key_value.append(None)

            kv_cache_params = KeyValueCacheParams(
                past_key_value=past_key_value,
                host_past_key_value_lengths=host_past_key_value_lengths,
                host_max_attention_window_sizes=host_max_attention_window_sizes,
                host_sink_token_length=host_sink_token_length,
                cache_indirection=cache_indirection,
                kv_cache_block_offsets=kv_cache_block_offsets,
                host_kv_cache_block_offsets=host_kv_cache_block_offsets,
                host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
                cross_kv_cache_block_offsets=cross_kv_cache_block_offsets,
                host_cross_kv_cache_block_offsets=
                host_cross_kv_cache_block_offsets,
                host_cross_kv_cache_pool_pointers=
                host_cross_kv_cache_pool_pointers,
                host_cross_kv_cache_pool_mapping=
                host_cross_kv_cache_pool_mapping,
            )

            attention_params = AttentionParams(
                sequence_length=sequence_length,
                context_lengths=context_lengths,
                host_context_lengths=host_context_lengths,
                max_context_length=max_decoder_input_len,
                host_request_types=host_request_types,
                host_runtime_perf_knobs=host_runtime_perf_knobs,
                host_context_progress=host_context_progress,
                encoder_input_lengths=encoder_input_lengths,
                encoder_max_input_length=encoder_max_input_length,
            )

        cross_kv_cache_gen = Tensor(name='cross_kv_cache_gen',
                                    dtype=trt.bool,
                                    shape=[1],
                                    dim_range=OrderedDict([
                                        ('boolean', [1]),
                                    ]))
        cross_kv_reuse = None
        num_heads = (self.transformer.num_heads + self.mapping.tp_size -
                     1) // self.mapping.tp_size
        cross_kv_out_dim = 2 * num_kv_heads * self.transformer.head_size
        if self.transformer.skip_cross_kv:
            if remove_input_padding:
                cross_kv_reuse = Tensor(
                    name="cross_kv_reuse",
                    dtype=self._dtype,
                    shape=[-1, cross_kv_out_dim],
                    dim_range=OrderedDict([
                        ("encoder_num_tokens", [encoder_num_tokens_range]),
                        ("encoder_kv_size", [cross_kv_out_dim]),
                    ]),
                )
            else:
                cross_kv_reuse = Tensor(
                    name="cross_kv_reuse",
                    dtype=self._dtype,
                    shape=[-1, -1, cross_kv_out_dim],
                    dim_range=OrderedDict([
                        ("batch_size_beam_width_encoder", [bb_range]),
                        ("encoder_input_len", [encoder_input_len_range]),
                        ("encoder_kv_size", [cross_kv_out_dim]),
                    ]),
                )

        skip_cross_attn_blocks = None
        if self.config.skip_cross_attn_blocks:
            skip_cross_attn_blocks = Tensor(name='skip_cross_attn_blocks',
                                            dtype=trt.bool,
                                            shape=[1],
                                            dim_range=OrderedDict([
                                                ('boolean', [1]),
                                            ]))

        prompt_embedding_table = None
        tasks = None
        prompt_vocab_size = None

        if self.mapping.is_first_pp_rank() and prompt_embedding_table_size > 0:
            p_embedding_range = [[
                1, prompt_embedding_table_size // 2, prompt_embedding_table_size
            ]]

            prompt_embedding_table = Tensor(
                name='prompt_embedding_table',
                dtype=self._dtype,
                shape=[-1, self.transformer.hidden_size],
                dim_range=OrderedDict([
                    ('prompt_embedding_table_size', p_embedding_range),
                    ('hidden_size', [self.transformer.hidden_size]),
                ]))
            if remove_input_padding:
                num_tokens_range = [
                    1,
                    (max_decoder_input_len * max_batch_size + 1) // 2,
                    max_decoder_input_len * max_batch_size,
                ]
                tasks = Tensor(name='tasks',
                               dtype=trt.int32,
                               shape=[-1],
                               dim_range=OrderedDict([
                                   ('decoder_num_tokens',
                                    [decoder_num_tokens_range]),
                               ]))
            else:
                tasks = Tensor(name='tasks',
                               dtype=trt.int32,
                               shape=[-1, 1],
                               dim_range=OrderedDict([
                                   ('batch_size', bs_range),
                                   ('broadcast_dim', [1]),
                               ]))
            prompt_vocab_size = Tensor(name='prompt_vocab_size',
                                       dtype=trt.int32,
                                       shape=[1],
                                       dim_range=OrderedDict([('size', [1])]))

        result = {
            'decoder_input_ids': input_ids,
            'encoder_output': encoder_output,
            'use_cache': True,
            'attention_mask_params': attention_mask_params,
            'last_token_ids': last_token_ids,
            'kv_cache_params': kv_cache_params,
            'attention_params': attention_params,
            'hidden_states': hidden_states,
            'lora_params': lora_params,
            'cross_kv_cache_gen': cross_kv_cache_gen,
            'cross_kv_reuse': cross_kv_reuse,
            'prompt_embedding_table': prompt_embedding_table,
            'prompt_tasks': tasks,
            'prompt_vocab_size': prompt_vocab_size,
            'skip_cross_attn_blocks': skip_cross_attn_blocks,
        }

        return result

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config, self.trtllm_modules_to_hf_modules)

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        ''' Create a MLLaMAForCausalLM object from give parameters
        '''
        import transformers

        kwargs.pop('load_by_shard', False)
        kwargs.pop('load_model_on_cpu', False)
        quant_ckpt_path = kwargs.pop('quant_ckpt_path', None)

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        config = MLLaMAConfig.from_hugging_face(hf_config_or_dir,
                                                dtype=dtype,
                                                mapping=mapping,
                                                quant_config=quant_config,
                                                **kwargs)

        custom_dict = {
            "lm_head": "language_model.lm_head",
            "transformer.ln_f": "language_model.model.norm",
            "transformer": "language_model.model",
            "self_attention": "self_attn",
            "cross_attention": "cross_attn",
            "vocab_embedding": "embed_tokens",
            "gate_attn": "cross_attn_attn_gate",
            "gate_ffwd": "cross_attn_mlp_gate",
            "q_layernorm": "q_norm",
            "k_layernorm": "k_norm",
        }

        if quant_ckpt_path is not None:
            hf_model_dir = quant_ckpt_path

        loader = ModelWeightsLoader(hf_model_dir, custom_dict)
        model = cls(config)
        loader.generate_tllm_weights(model)

        return model
