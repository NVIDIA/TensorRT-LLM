# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ChatGLM2/3 ``_torch`` decoder model."""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from .modeling_utils import (
    DecoderModel,
    DecoderModelForCausalLM,
    duplicate_kv_weight,
    register_auto_model,
)

_IGNORABLE_WEIGHT_SUFFIXES = ("rotary_pos_emb.inv_freq",)
_IGNORABLE_WEIGHT_SUBSTRINGS = ("prefix_encoder",)


def normalize_chatglm_config(config: PretrainedConfig) -> PretrainedConfig:
    """Fill missing canonical HF aliases from ChatGLM-native fields."""

    def _set_missing(name: str, value) -> None:
        if getattr(config, name, None) is None:
            setattr(config, name, value)

    _set_missing("num_hidden_layers", getattr(config, "num_layers", None))
    _set_missing("vocab_size", getattr(config, "padded_vocab_size", None))
    _set_missing("intermediate_size", getattr(config, "ffn_hidden_size", None))

    _set_missing("head_dim", getattr(config, "kv_channels", None))
    if getattr(config, "multi_query_attention", False):
        _set_missing("num_key_value_heads", getattr(config, "multi_query_group_num", None))
    else:
        _set_missing("num_key_value_heads", getattr(config, "num_attention_heads", None))

    _set_missing("max_position_embeddings", getattr(config, "seq_length", None))
    _set_missing("rms_norm_eps", getattr(config, "layernorm_epsilon", None))
    # ChatGLM rotates the first half of each head.
    _set_missing("partial_rotary_factor", 0.5)
    # ChatGLM derives the RoPE base as 10000 * rope_ratio (rope_ratio defaults to 1).
    # RopeParams.from_config reads only rope_theta, so fill it here or long-context
    # variants (e.g. chatglm3-6b-32k, rope_ratio>1) would silently use base 10000.
    _set_missing("rope_theta", 10000.0 * (getattr(config, "rope_ratio", None) or 1.0))

    _set_missing(
        "attention_bias",
        bool(getattr(config, "add_qkv_bias", False) or getattr(config, "add_bias_linear", False)),
    )
    _set_missing("mlp_bias", bool(getattr(config, "add_bias_linear", False)))

    return config


class ChatGLMAttention(Attention):
    """ChatGLM multi-query attention with partial GPT-J RoPE."""

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gptj,
            rope=RopeParams.from_config(config),
            is_neox=False,
        )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=pos_embd_params,
            # ChatGLM's partial GPT-J RoPE is applied before the attention op.
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=config.mlp_bias,
            config=model_config,
            head_dim=config.head_dim,
        )


class ChatGLMDecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config

        self.self_attn = ChatGLMAttention(model_config, layer_idx=layer_idx)

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            dtype=config.torch_dtype,
            config=model_config,
        )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, **kwargs)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
        return hidden_states, residual


class ChatGLMModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList(
            [
                ChatGLMDecoderLayer(model_config, layer_idx)
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
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the "
                "same time, and must specify either one"
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


@register_auto_model("ChatGLMModel")
@register_auto_model("ChatGLMForConditionalGeneration")
@register_auto_model("ChatGLMForCausalLM")
class ChatGLMForCausalLM(DecoderModelForCausalLM[ChatGLMModel, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        normalize_chatglm_config(model_config.pretrained_config)
        super().__init__(
            ChatGLMModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights: Dict, **kwargs) -> None:
        """Load ChatGLM checkpoint weights with strict accounting."""
        config = self.config
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        ffn = config.intermediate_size
        tp_size = (
            1
            if self.model_config.mapping.enable_attention_dp
            else self.model_config.mapping.tp_size
        )

        consumed = set()

        def _take(key: str) -> torch.Tensor:
            if key not in weights:
                raise KeyError(f"ChatGLM checkpoint is missing required weight: {key}")
            consumed.add(key)
            return weights[key][:]

        def _load_vanilla(module: nn.Module, key: str) -> None:
            tensor = _take(key)
            if hasattr(module, "load_weights"):
                module.load_weights(weights=[{"weight": tensor}])
            else:
                module.weight.data.copy_(tensor)

        _load_vanilla(self.model.embed_tokens, "transformer.embedding.word_embeddings.weight")
        _load_vanilla(self.model.norm, "transformer.encoder.final_layernorm.weight")
        _load_vanilla(self.lm_head, "transformer.output_layer.weight")

        for i, layer in enumerate(self.model.layers):
            p = f"transformer.encoder.layers.{i}."
            _load_vanilla(layer.input_layernorm, p + "input_layernorm.weight")
            _load_vanilla(layer.post_attention_layernorm, p + "post_attention_layernorm.weight")
            _load_vanilla(layer.self_attn.o_proj, p + "self_attention.dense.weight")
            _load_vanilla(layer.mlp.down_proj, p + "mlp.dense_4h_to_h.weight")

            # ChatGLM stores fused QKV as [Q, K, V].
            qkv_w = _take(p + "self_attention.query_key_value.weight")
            qkv_b = _take(p + "self_attention.query_key_value.bias")
            q_w, k_w, v_w = torch.split(qkv_w, [q_size, kv_size, kv_size], dim=0)
            q_b, k_b, v_b = torch.split(qkv_b, [q_size, kv_size, kv_size], dim=0)
            # Duplicate compact KV heads only when TP needs replicated KV.
            k_w = duplicate_kv_weight(k_w, num_kv_heads, tp_size)
            v_w = duplicate_kv_weight(v_w, num_kv_heads, tp_size)
            k_b = duplicate_kv_weight(k_b, num_kv_heads, tp_size)
            v_b = duplicate_kv_weight(v_b, num_kv_heads, tp_size)
            layer.self_attn.qkv_proj.load_weights(
                weights=[
                    {"weight": q_w, "bias": q_b},
                    {"weight": k_w, "bias": k_b},
                    {"weight": v_w, "bias": v_b},
                ]
            )

            # Fused gate/up: ChatGLM SwiGLU is silu(first_half) * second_half.
            h4h = _take(p + "mlp.dense_h_to_4h.weight")
            gate_w, up_w = torch.split(h4h, [ffn, ffn], dim=0)
            layer.mlp.gate_up_proj.load_weights(
                weights=[
                    {"weight": gate_w},
                    {"weight": up_w},
                ]
            )

        def _is_ignorable(key: str) -> bool:
            return any(key.endswith(s) for s in _IGNORABLE_WEIGHT_SUFFIXES) or any(
                s in key for s in _IGNORABLE_WEIGHT_SUBSTRINGS
            )

        unexpected = sorted(k for k in weights if k not in consumed and not _is_ignorable(k))
        if unexpected:
            raise ValueError(
                "Unexpected/unconsumed ChatGLM checkpoint weights "
                f"({len(unexpected)}): {unexpected[:8]}"
                f"{' ...' if len(unexpected) > 8 else ''}"
            )
