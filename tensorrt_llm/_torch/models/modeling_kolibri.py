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

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.configs.kolibri import Kolibri1Config
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, register_auto_model
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.moe.fused_moe import BaseMoeRoutingMethod, RoutingMethodType, create_moe
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.functional import PositionEmbeddingType


class Kolibri1RoutingMethod(BaseMoeRoutingMethod):
    """Routing method for Kolibri 1:
    - Computes Top-k over (logits + e_score_correction_bias).
    - Gathers weights from sigmoid(logits).
    """

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        norm_topk_prob: bool = False,
        output_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.e_score_correction_bias = e_score_correction_bias
        self.norm_topk_prob = norm_topk_prob
        self.output_dtype = output_dtype

    @property
    def requires_separated_routing(self) -> bool:
        # Runs router scoring in Python / @torch.compile
        return True

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.SigmoidRenorm

    def apply(
        self,
        router_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = router_logits.float()
        scores = (
            logits + self.e_score_correction_bias
            if self.e_score_correction_bias is not None
            else logits
        )
        _, topk_ids = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        topk_weights = torch.sigmoid(logits.gather(dim=-1, index=topk_ids))
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_ids.to(torch.int32), topk_weights.to(self.output_dtype)


class Kolibri1Attention(Attention):
    """Kolibri 1 Attention Layer:

    - Alternates SWA and Full Attention based on config.layer_types.
    - SWA layers use RoPE + sliding window.
    - Full-attention layers use RNoPE (No Positional Encoding, global window).
    - Applies per-head QK RMSNorm.
    """

    def __init__(
        self,
        model_config: ModelConfig[Kolibri1Config],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        layer_types = getattr(config, "layer_types", [])
        self.is_full_attention = (
            layer_idx is not None
            and layer_idx < len(layer_types)
            and layer_types[layer_idx] == "full_attention"
        )
        self.attention_window_size = (
            None if self.is_full_attention else getattr(config, "sliding_window", None)
        )

        pos_embd_params = None
        if not self.is_full_attention:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )

        self.q_norm = RMSNorm(
            hidden_size=config.head_dim,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.k_norm = RMSNorm(
            hidden_size=config.head_dim,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            skip_rope=self.is_full_attention,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply head-dim RMSNorm independently to Q and K."""
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)
        return q, k

    def apply_rope(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 1. Unpack fused QKV into separate Q, K, V tensors
        q, k, v = self.split_qkv(q, k, v)
        assert k is not None and v is not None, "k and v must be present after split_qkv"

        # 2. Normalize Q and K heads
        q, k = self.apply_qk_norm(q, k)

        # 3. Skip RoPE on full-attention layers (RNoPE)
        if self.is_full_attention:
            return q, k, v

        # 4. Apply RoPE on SWA layers
        return super().apply_rope(q, k, v, position_ids)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_window_size=self.attention_window_size,
            **kwargs,
        )


class Kolibri1MoE(nn.Module):
    """Kolibri 1 MoE Block:
    - 384 fine-grained routed experts (intermediate_size=512).
    - 1 shared expert (GatedMLP, intermediate_size=512).
    - Gate linear layer with e_score_correction_bias parameter.
    """

    def __init__(
        self,
        model_config: ModelConfig[Kolibri1Config],
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        # 1. Router Gate: model.layers.{i}.mlp.gate
        self.gate = Linear(
            self.hidden_dim,
            self.num_experts,
            bias=False,
            dtype=config.torch_dtype,
        )
        # model.layers.{i}.mlp.gate.e_score_correction_bias
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32),
            requires_grad=False,
        )

        # 2. Routing Strategy
        self.routing_method = Kolibri1RoutingMethod(
            top_k=self.top_k,
            num_experts=self.num_experts,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            norm_topk_prob=config.norm_topk_prob,  # False from config.json!
        )

        # 3. 384 Routed Experts
        self.experts = create_moe(
            num_experts=self.num_experts,
            routing_method=self.routing_method,
            hidden_size=self.hidden_dim,
            intermediate_size=config.moe_intermediate_size,  # 512
            aux_stream_dict=aux_stream_dict,
            dtype=config.torch_dtype,
            reduce_results=False,
            model_config=model_config,
            layer_idx=layer_idx,
        )

        # 4. 1 Shared Expert (intermediate_size=512)
        from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

        self.shared_experts = GatedMLP(
            hidden_size=self.hidden_dim,
            intermediate_size=config.shared_expert_intermediate_size,  # 512
            bias=False,
            dtype=config.torch_dtype,
            config=model_config,
            is_shared_expert=True,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)

        # 1. Compute router logits [num_tokens, 384]
        router_logits = self.gate(hidden_states)

        # 2. Run 384 routed experts (top-6 activated)
        routed_out = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
            use_dp_padding=False,
        )

        # 3. Run shared expert
        shared_out = self.shared_experts(hidden_states)

        # 4. Sum outputs and restore shape
        return (routed_out + shared_out).view(orig_shape)


class Kolibri1DecoderLayer(DecoderLayer):
    """Kolibri 1 Decoder Layer with 4-Norm Sandwich Structure:

    - input_layernorm -> self_attn -> post_attn_norm -> residual add
    - post_attention_layernorm -> mlp -> post_ffn_norm -> residual add
    """

    def __init__(
        self,
        model_config: ModelConfig[Kolibri1Config],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        # Attention sublayer
        self.self_attn = Kolibri1Attention(model_config, layer_idx=layer_idx)

        # MoE sublayer
        self.mlp = Kolibri1MoE(model_config, aux_stream_dict, layer_idx=layer_idx)

        # Sandwich Normalizations
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attn_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_ffn_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # 1. Attention sublayer with post-attention sandwich norm
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        hidden_states = self.post_attn_norm(hidden_states)

        # 2. Add attention output to residual and compute pre-FFN norm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # 3. MoE sublayer with post-FFN sandwich norm
        hidden_states = self.mlp(
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        hidden_states = self.post_ffn_norm(hidden_states)

        return hidden_states, residual


class Kolibri1Model(DecoderModel):
    """Kolibri 1 Backbone Model."""

    def __init__(self, model_config: ModelConfig[Kolibri1Config]):
        super().__init__(model_config)
        config = self.model_config

        self.aux_stream_dict = (
            {
                AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(),
                AuxStreamType.MoeBalancer: torch.cuda.Stream(),
                AuxStreamType.MoeOutputMemset: torch.cuda.Stream(),
            }
            if torch.cuda.is_available()
            else {}
        )

        self.preload_weight_modules = []
        if config.moe_backend == "TRTLLM":
            self.preload_weight_modules = [
                "experts",
                "routing_method",
                "all_reduce",
            ]

        if model_config.mapping.enable_attention_dp:
            self.embed_tokens = Embedding(
                config.pretrained_config.vocab_size,
                config.pretrained_config.hidden_size,
                dtype=config.pretrained_config.torch_dtype,
            )
        else:
            self.embed_tokens = Embedding(
                config.pretrained_config.vocab_size,
                config.pretrained_config.hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )

        self.layers = nn.ModuleList(
            [
                Kolibri1DecoderLayer(
                    model_config,
                    layer_idx=layer_idx,
                    aux_stream_dict=self.aux_stream_dict,
                )
                for layer_idx in range(config.pretrained_config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            hidden_size=config.pretrained_config.hidden_size,
            eps=config.pretrained_config.rms_norm_eps,
            dtype=config.pretrained_config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
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
                spec_metadata=spec_metadata,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("Kolibri1ForCausalLM")
class Kolibri1ForCausalLM(SpecDecOneEngineForCausalLM[Kolibri1Model, Kolibri1Config]):
    """Kolibri 1 Causal Language Model."""

    def __init__(self, model_config: ModelConfig[Kolibri1Config]):
        super().__init__(
            Kolibri1Model(model_config),
            model_config,
        )
        self.preload_weight_modules = self.model.preload_weight_modules
