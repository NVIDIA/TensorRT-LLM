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
from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import gather_last_token_logits, recv, send
from ...layers import (Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, GatedMLP, KeyValueCacheParams,
                       PositionEmbeddingType, RmsNorm)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..generation_mixin import GenerationMixin
from tensorrt import DataType


class YiDecoderLayer(Module):
    
    def __init__(self,
                 *,
                 hidden_size,
                 num_attention_heads,
                 num_key_value_heads,
                 max_position_embeddings,
                 dtype,
                 hidden_act,
                 position_embedding_type,
                 rope_theta,
                 rope_scaling,
                 intermediate_size,
                 rms_norm_eps,
                 tp_group,
                 tp_size,
                 layer_id):

        super().__init__()
        self.layer_id = layer_id
        
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            position_embedding_type=position_embedding_type,
            rotary_embedding_base=rope_theta,
            rotary_embedding_scaling=rope_scaling,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            instance_id=2*layer_id)
        
        self.mlp = GatedMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=intermediate_size,
            hidden_act=hidden_act,
            bias=False,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            instance_id=2*layer_id+1)

        self.ln1 = RmsNorm(normalized_shape=hidden_size,
                           eps=rms_norm_eps,
                           dtype=dtype)
        
        self.ln2 = RmsNorm(normalized_shape=hidden_size,
                           eps=rms_norm_eps,
                           dtype=dtype)

    def forward(self,
                *,
                hidden_states,
                attention_mask,
                use_cache,
                kv_cache_params,
                attention_params,
                all_reduce_workspace):
        
        residual = hidden_states

        hidden_states = self.ln1(hidden_states)
            
        # Self Attention
        hidden_states = self.self_attn(hidden_states,
                                attention_mask=attention_mask,
                                use_cache=use_cache,
                                kv_cache_params=kv_cache_params,
                                attention_params=attention_params,
                                workspace=all_reduce_workspace)
            
        if use_cache:
            hidden_states, presents = hidden_states
            
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return (hidden_states, presents) if use_cache else hidden_states


class YiModel(Module):

    def __init__(self,
                 *,
                 num_hidden_layers,
                 num_attention_heads,
                 num_key_value_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype,
                 intermediate_size,
                 position_embedding_type,
                 rope_theta,
                 rope_scaling,
                 mapping,
                 use_parallel_embedding,
                 embedding_sharding_dim,
                 rms_norm_eps):

        super().__init__()
        self.mapping = mapping
        
        if mapping.is_first_pp_rank():
            self.embed_tokens = Embedding(
                num_embeddings=vocab_size,
                embedding_dim=hidden_size,
                dtype=dtype,
                tp_size=mapping.tp_size if use_parallel_embedding else 1,
                tp_group=mapping.tp_group if use_parallel_embedding else None,
                sharding_dim=embedding_sharding_dim,
                tp_rank=mapping.tp_rank)
        
        self.layers = ModuleList([
            YiDecoderLayer(
                        hidden_size=hidden_size,
                        num_attention_heads=num_attention_heads,
                        num_key_value_heads=num_key_value_heads,
                        max_position_embeddings=max_position_embeddings,
                        dtype=dtype,
                        hidden_act=hidden_act,
                        position_embedding_type=position_embedding_type,
                        rope_theta=rope_theta,
                        rope_scaling=rope_scaling,
                        intermediate_size=intermediate_size,
                        rms_norm_eps=rms_norm_eps,
                        tp_group=mapping.tp_group,
                        tp_size=mapping.tp_size,
                        layer_id=i)
            for i in self.get_transformer_layers(self.mapping, num_hidden_layers)
        ])
        
        if self.mapping.is_last_pp_rank():
            self.norm = RmsNorm(normalized_shape=hidden_size,
                                eps=rms_norm_eps,
                                dtype=dtype)
            
    def forward(self,
                *,
                input_ids,
                position_ids,
                use_cache,
                attention_mask,
                kv_cache_params,
                attention_params,
                hidden_states,
                all_reduce_workspace):

        if kv_cache_params.past_key_value is None:
            kv_cache_params.past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        if self.mapping.is_first_pp_rank():
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())
        
        for layer, past, pointer in zip(
                self.layers, kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers):
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    kv_cache_block_pointers=[pointer],
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params,
                all_reduce_workspace=all_reduce_workspace)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        if self.mapping.is_last_pp_rank():
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())
        
        return (hidden_states, tuple(presents)) if use_cache else hidden_states
    

class YiForCausalLM(YiModel, GenerationMixin):

    def __init__(self,
                 *,
                 num_hidden_layers,
                 num_attention_heads,
                 num_key_value_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype = 'bfloat16',
                 intermediate_size=11008,
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 rope_theta=5000000.0,
                 rope_scaling=None,
                 mapping=Mapping(),
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0,
                 rms_norm_eps=1e-5):
        
        if isinstance(dtype, str):
            self.dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, DataType)
            self.dtype = dtype

        super().__init__(
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            intermediate_size=intermediate_size,
            position_embedding_type=position_embedding_type,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            mapping=mapping,
            use_parallel_embedding=use_parallel_embedding,
            embedding_sharding_dim=embedding_sharding_dim,
            rms_norm_eps=rms_norm_eps)
        
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.use_parallel_embedding = use_parallel_embedding
        self.embedding_sharding_dim = embedding_sharding_dim
        self.kv_dtype = self.dtype

        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        if self.mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(hidden_size,
                                        vocab_size_padded,
                                        bias=False,
                                        dtype=dtype,
                                        tp_group=mapping.tp_group,
                                        tp_size=mapping.tp_size,
                                        gather_output=True)

    def forward(self,
                input_ids,
                position_ids=None, #TODO : unused arugment?
                use_cache=False,
                last_token_ids=None,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                all_reduce_workspace=None):

        hidden_states = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            hidden_states=hidden_states,
            all_reduce_workspace=all_reduce_workspace)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids,
                default_net().plugin_config.remove_input_padding)

            # [batch_size, hidden_size] -> [batch_size, vocab_size]
            lm_logits = self.lm_head(hidden_states)

            # GenerationSession does not support bfloat16 logits now, will fallback to float32 automatically
            lm_logits.mark_output('logits', self.dtype)
        else:
            hidden_states.mark_output('hidden_states_output', self.dtype)
            
        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in zip(
                    self.get_transformer_layers(self.mapping, self.num_hidden_layers), presents):
                present.mark_output(f'present_key_value_{i}', self.kv_dtype)
            if self.mapping.is_last_pp_rank():
                return (lm_logits, presents)
            return (hidden_states, presents)
        else:
            if self.mapping.is_last_pp_rank():
                return lm_logits
            return hidden_states

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       max_new_tokens,
                       use_cache,
                       max_beam_width,
                       max_num_tokens: int = None):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self.hidden_size // self.num_attention_heads
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net().plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        use_custom_all_reduce = default_net().plugin_config.use_custom_all_reduce

        model_inputs = self.prepare_basic_inputs(
            max_batch_size,
            max_beam_width,
            max_input_len,
            max_new_tokens,
            self.num_key_value_heads,
            head_size,
            self.num_hidden_layers,
            self.kv_dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            use_custom_all_reduce=use_custom_all_reduce,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            dtype=self.dtype,
            num_heads=self.num_attention_heads,
            mapping=self.mapping,
            max_num_tokens=max_num_tokens)

        return (model_inputs['input_ids'],
                model_inputs['position_ids'],
                True,
                model_inputs['last_token_ids'],
                model_inputs['attention_mask'],
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
                    host_request_types=model_inputs['host_request_types']),
                model_inputs['hidden_states_input'],
                model_inputs['all_reduce_workspace'])
