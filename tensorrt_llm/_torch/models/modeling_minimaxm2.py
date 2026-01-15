# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..models.modeling_utils import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import MiniMaxM2MoeRoutingMethod, create_moe
from ..modules.linear import Linear
from ..modules.rms_norm import RMSNorm
from ..utils import AuxStreamType
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model


# MiniMax M2/M2.1 requires the implementation of the following two additional components:
#  1. MoE routing method: Currently, TRT-LLM does not support
#     the following routing method: sigmoid -> add bias -> topk -> renorm.
#  2. QK layer normalization needs to be performed across the head_num * head_size dimension,
#     which conflicts with the current TP-mode attention logic.
# For the better performance, we suggest to enable attention DP when using MiniMax M2/M2.1 model.
class MiniMaxM2MoE(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.enable_attention_dp = model_config.mapping.enable_attention_dp

        # moe gate (linear layer) only runs in half/full precision for now
        self.gate = Linear(
            self.hidden_dim, self.num_experts, bias=False, dtype=torch.float32, quant_config=None
        )

        self.e_score_correction_bias = nn.Parameter(
            torch.empty((self.num_experts), dtype=torch.float32), requires_grad=False
        )

        reduce_results = True
        self.experts = create_moe(
            routing_method=MiniMaxM2MoeRoutingMethod(
                top_k=self.top_k,
                num_experts=self.num_experts,
                callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
            ),
            num_experts=self.num_experts,
            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
            reduce_results=reduce_results,
            model_config=model_config,
            layer_idx=layer_idx,
        )

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1

        self.e_score_correction_bias.copy_(
            weights[0]["e_score_correction_bias"][:].to(self.e_score_correction_bias.dtype)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        hidden_states_f32 = hidden_states.to(torch.float32)
        router_logits = self.gate(hidden_states_f32)
        final_hidden_states = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=False,
        )
        return final_hidden_states


# It's a little bit tricky to implement special qk norm
# because rms dim is hidden_size * num_heads, not hidden_size, after qkv linear,
# the result size is hidden_size * num_heads / tp_size.
# Actually, we have two strategies to implement qk norm attention:
# 1. the first linear layer is not col parallel, then we can use the normal rms layer norm. each attention use full qkv
# 2. we use col parallel linear layer, then we use allgather to gather qkv from all gpus,
#    then we use rms norm on q and k. Finally, we split qkv to each gpus and continue.
# for better performance, we choose the second strategy here.
# Most adaptions are from QKNormRoPEAttention.
class MiniMaxM2Attention(Attention):
    def __init__(
        self,
        *,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        self.pretrained_config = config

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            rope_fusion=True,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

        self.q_norm = RMSNorm(
            hidden_size=self.q_size * self.tp_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.k_norm = RMSNorm(
            hidden_size=self.kv_size * self.tp_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def apply_qk_norm(self, q, k):
        if self.qkv_proj.mapping.tp_size > 1:
            # collect q and k from all gpus
            from ..distributed import allgather

            temp_q = allgather(q, self.qkv_proj.mapping)
            temp_k = allgather(k, self.qkv_proj.mapping)
            temp_q = self.q_norm(temp_q)
            temp_k = self.k_norm(temp_k)
            q = temp_q.reshape(-1, self.tp_size, self.q_size)[:, self.tp_rank, :].reshape(
                -1, self.q_size
            )
            k = temp_k.reshape(-1, self.tp_size, self.kv_size)[:, self.tp_rank, :].reshape(
                -1, self.kv_size
            )
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)

        return q, k

    def apply_rope(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ):
        """
        The apply_rope method is called in the forward method of the Attention class.
        The apply_rope method is overridden in this class to apply QK norm and RoPE to the input tensor.
        """
        # Apply QK norm before RoPE.
        q, k, v = self.split_qkv(q, k, v)
        q, k = self.apply_qk_norm(q, k)
        return super().apply_rope(q, k, v, position_ids)


class MiniMaxM2DecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_size = config.hidden_size

        self.self_attn = MiniMaxM2Attention(model_config=model_config, layer_idx=layer_idx)

        self.block_sparse_moe = MiniMaxM2MoE(
            model_config=model_config, aux_stream=aux_stream, layer_idx=layer_idx
        )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.mapping = model_config.mapping
        self.layer_idx = layer_idx

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states, attn_metadata)
        return hidden_states, residual


class MiniMaxM2Model(DecoderModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        # add this for kv cache initialization (if we use bf16 for kv cache)
        quant_config = model_config.quant_config
        if quant_config is None or (
            (not quant_config.quant_mode.has_fp8_kv_cache())
            and (not quant_config.quant_mode.has_fp4_kv_cache())
        ):
            model_config.pretrained_config.torch_dtype = torch.bfloat16
        config = model_config.pretrained_config
        self.vocab_size = config.vocab_size
        self.aux_stream = torch.cuda.Stream()

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            enable_torch_compile_for_embedding=model_config.enable_torch_compile_for_embedding,
        )

        self.layers = nn.ModuleList(
            [
                MiniMaxM2DecoderLayer(model_config, layer_idx, self.aux_stream)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("MiniMaxM2ForCausalLM")
class MiniMaxM2ForCausalLM(DecoderModelForCausalLM[MiniMaxM2Model, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(
            MiniMaxM2Model(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )
