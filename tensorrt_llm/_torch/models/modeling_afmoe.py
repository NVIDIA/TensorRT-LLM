# SPDX-License-Identifier: Apache-2.0
"""Inference-only AFMoE (Arcee Foundation MoE) for TensorRT-LLM.

Follows the HF implementation of AfmoeForCausalLM.

Key architectural features:
  - Per-layer attention type (sliding_attention vs global)
  - Q/K RMSNorm in attention
  - Gated attention output (sigmoid gate)
  - RoPE only on local (sliding-window) attention layers
  - Dense MLP for early layers, MoE with shared experts for later layers
  - 4 layer norms per decoder block (pre/post attention, pre/post MLP)
  - Optional muP embedding scaling
"""

from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ...logger import logger
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from ..distributed import AllReduce
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import DeepSeekV3MoeRoutingMethod, create_moe
from ..modules.fused_moe.routing import Deepseekv3RoutingImpl
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..utils import AuxStreamType
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model


class AfmoeConfig(PretrainedConfig):
    model_type = "afmoe"


logger.warning_once(
    "transformers does not natively support 'AfmoeConfig'. "
    "Registering AfmoeConfig so AutoConfig can load AFMoE checkpoints.",
    key="AFMOE_REGISTER_WARNING",
)
AutoConfig.register(AfmoeConfig.model_type, AfmoeConfig)


def _validate_routing_config(config: PretrainedConfig) -> None:
    """Validate that the routing config matches our Deepseekv3RoutingImpl assumptions."""
    score_func = getattr(config, "scoring_func", getattr(config, "score_func", "sigmoid"))
    if score_func != "sigmoid":
        raise ValueError(
            f"AFMoE implementation uses sigmoid scoring via "
            f"Deepseekv3RoutingImpl, but config has "
            f"scoring_func={score_func!r}. Only 'sigmoid' is supported."
        )

    norm_topk = getattr(config, "norm_topk_prob", getattr(config, "route_norm", True))
    if not norm_topk:
        raise ValueError(
            "AFMoE implementation assumes normalized top-k probabilities "
            "(norm_topk_prob=True / route_norm=True), but config disables it."
        )


class AfmoeGate(nn.Module):
    """Router gate for AFMoE, following the DeepSeekV3 grouped top-k pattern."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        route_scale: float,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=dtype),
            requires_grad=False,
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.routing_impl = Deepseekv3RoutingImpl(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=route_scale,
            is_fused=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.ops.trtllm.dsv3_router_gemm_op(
            hidden_states,
            self.weight.t(),
            bias=None,
            out_dtype=torch.float32,
        )
        return logits

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        self.weight.copy_(weights[0]["weight"][:])
        self.e_score_correction_bias.copy_(
            weights[0]["e_score_correction_bias"][:].to(self.e_score_correction_bias.dtype)
        )

    @property
    def routing_method(self) -> DeepSeekV3MoeRoutingMethod:
        return DeepSeekV3MoeRoutingMethod(
            top_k=self.routing_impl.top_k,
            n_group=self.routing_impl.n_group,
            topk_group=self.routing_impl.topk_group,
            routed_scaling_factor=self.routing_impl.routed_scaling_factor,
            is_fused=self.routing_impl.is_fused,
            callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
        )


class AfmoeMoE(nn.Module):
    """MoE layer with shared experts for AFMoE.

    Both routed experts and shared experts produce TP-partial results
    (reduce_results=False / reduce_output=False).  After summing them
    we perform a single AllReduce so that each rank holds the full
    hidden-state, matching the DeepSeekV3 MoE pattern.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config

        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.num_shared_experts = getattr(config, "num_shared_experts", 0)
        self.enable_attention_dp = model_config.mapping.enable_attention_dp

        self.gate = AfmoeGate(
            hidden_size=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_group=config.n_group,
            topk_group=config.topk_group,
            route_scale=getattr(config, "route_scale", 1.0),
            dtype=config.torch_dtype,
        )

        self.experts = create_moe(
            num_experts=self.num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=self.hidden_dim,
            intermediate_size=config.moe_intermediate_size,
            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
            dtype=config.torch_dtype,
            reduce_results=False,
            model_config=model_config,
            layer_idx=layer_idx,
        )

        if self.num_shared_experts > 0:
            shared_intermediate = config.moe_intermediate_size * self.num_shared_experts
            self.shared_experts = GatedMLP(
                hidden_size=self.hidden_dim,
                intermediate_size=shared_intermediate,
                bias=False,
                dtype=config.torch_dtype,
                config=model_config,
                reduce_output=False,
                layer_idx=layer_idx,
            )
        else:
            self.shared_experts = None

        self.mapping = model_config.mapping

        self.allreduce = None
        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        router_logits = self.gate(hidden_states)

        routed_output = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=False,
        )

        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            final_output = shared_output.add_(routed_output)
        else:
            final_output = routed_output

        if self.allreduce is not None:
            final_output = self.allreduce(final_output)

        return final_output


class AfmoeAttention(Attention):
    """Attention with Q/K norm, per-layer sliding window, gated output.

    Uses a separate gate_proj linear (not fused into QKV) to gate the
    attention output with sigmoid, matching the HF checkpoint layout.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        layer_types = getattr(config, "layer_types", [])
        self.is_local_attention = (
            layer_idx is not None
            and layer_idx < len(layer_types)
            and layer_types[layer_idx] == "sliding_attention"
        )
        self._attention_window_size = config.sliding_window if self.is_local_attention else None

        rope_params = RopeParams.from_config(config) if self.is_local_attention else None
        pos_embd_params = (
            PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=rope_params,
            )
            if self.is_local_attention
            else None
        )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=getattr(config, "max_position_embeddings", 131072),
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

        self.q_norm = RMSNorm(
            hidden_size=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.k_norm = RMSNorm(
            hidden_size=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        self.gate_proj = Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            quant_config=model_config.get_quant_config(),
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=model_config.use_cute_dsl_blockscaling_mm,
        )

    def apply_rope(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ):
        q, k, v = self.split_qkv(q, k, v)

        q_shape = q.shape
        k_shape = k.shape
        q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(q_shape)
        k = self.k_norm(k.reshape(-1, self.num_key_value_heads, self.head_dim)).reshape(k_shape)

        if self.is_local_attention and not self.rope_fusion and position_ids is not None:
            q, k = self.rotary_emb(position_ids, [q, k])

        return q, k, v

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv, None, None

        q, k, v = self.apply_rope(q, k, v, position_ids)
        q, k, v = self.convert_qkv(q, k, v)

        attn_output = self.forward_impl(
            q,
            k,
            v,
            attn_metadata,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_window_size=self._attention_window_size,
            attention_mask_data=None,
            mrope_config=None,
        )

        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output


class AfmoeDecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = AfmoeAttention(model_config, layer_idx=layer_idx)

        num_dense_layers = getattr(config, "num_dense_layers", 0)
        self.moe_enabled = layer_idx >= num_dense_layers
        if self.moe_enabled:
            self.mlp = AfmoeMoE(model_config, aux_stream, layer_idx=layer_idx)
        else:
            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                bias=False,
                dtype=config.torch_dtype,
                config=model_config,
                layer_idx=layer_idx,
            )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.pre_mlp_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_mlp_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

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

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_mlp_layernorm(hidden_states, residual)

        if self.moe_enabled:
            hidden_states = self.mlp(hidden_states, attn_metadata)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_mlp_layernorm(hidden_states)

        return hidden_states, residual


class AfmoeModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        _validate_routing_config(config)

        self.vocab_size = config.vocab_size
        self.mup_enabled = getattr(config, "mup_enabled", False)
        self.hidden_size = config.hidden_size
        self.aux_stream = torch.cuda.Stream()

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            enable_torch_compile_for_embedding=model_config.enable_torch_compile_for_embedding,
        )

        self.layers = nn.ModuleList(
            [
                AfmoeDecoderLayer(model_config, layer_idx, self.aux_stream)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
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
                "You cannot specify both input_ids and inputs_embeds at "
                "the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.mup_enabled:
            inputs_embeds = inputs_embeds * (self.hidden_size**0.5)

        hidden_states = inputs_embeds

        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("AfmoeForCausalLM")
class AfmoeForCausalLM(DecoderModelForCausalLM[AfmoeModel, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(
            AfmoeModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights: Dict, weight_mapper, **kwargs):
        weights = weight_mapper.preprocess_weights(weights)
        super().load_weights(weights=weights, weight_mapper=weight_mapper, **kwargs)
