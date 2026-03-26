# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Eagle model implementation for AutoDeploy.

Eagle is a speculative decoding draft model that predicts next tokens based on
hidden states from a target model (e.g., Llama-3.1-8B-Instruct).

This file contains:
- Generic Eagle infrastructure (EagleModel, EagleDrafterForCausalLM, EagleWrapper)
- Llama-specific Eagle layer implementation (LlamaEagleLayer)
- Layer dispatch functions for model-specific layer construction

Model-specific layers for other architectures (e.g., NemotronH) are defined in their
respective model files and registered via get_eagle_layers().
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from ....pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from ...shim.interface import CachedSequenceInterface
from ...utils._config import deep_merge_dicts
from ...utils.logger import ad_logger
from .modeling_nemotron_h import build_nemotron_eagle_layers

# =============================================================================
# Layer Dispatch Functions
# =============================================================================


def get_eagle_layers(config, model_type: str) -> Union[nn.ModuleList, nn.Module]:
    """Build Eagle layers for the given model type.

    This function dispatches to model-specific layer builders based on model_type.
    Each builder returns layers that implement the unified forward signature:
    forward(hidden_states, inputs_embeds, position_ids) -> Tensor

    For backward compatibility with checkpoints:
    - Single layer: returns layer directly (not wrapped in ModuleList)
    - Multiple layers: returns nn.ModuleList of layer instances

    Args:
        config: Model configuration (e.g., EagleConfig for Llama)
        model_type: The base model type (e.g., "llama", "nemotron_h")

    Returns:
        nn.ModuleList of layers for the Eagle model, or single layer if there is only one layer
    """
    layers: list[nn.Module]
    match model_type:
        case "llama":
            layers = build_llama_eagle_layers(config)
        case "nemotron_h":
            layers = build_nemotron_eagle_layers(config)
        case _:
            raise ValueError(
                f"Model type '{model_type}' not supported for Eagle drafter. "
                f"Supported types: llama, nemotron_h"
            )

    if len(layers) == 1:
        return layers[0]
    return nn.ModuleList(layers)


def build_llama_eagle_layers(config) -> list[nn.Module]:
    """Build Llama-style Eagle decoder layers.

    Each layer handles RoPE internally, making the EagleModel fully model-agnostic.
    """
    return [LlamaEagleLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]


class EagleConfig(PretrainedConfig):
    """Config for Eagle3 drafter models.

    Extends PretrainedConfig with Eagle-specific parameters while preserving
    all base model config values.

    Args:
        config: Base config for the draft model from its config.json.
        model_type: The base model type (e.g., "llama", "nemotron_h") used to look up defaults.
    """

    # Map model_type -> default Eagle config values
    # Includes _checkpoint_conversion_mapping for model-specific weight key transformations
    _drafter_defaults: Dict[str, Dict[str, Any]] = {
        "llama": {
            "load_embedding_from_target": True,
            "load_lm_head_from_target": False,
            "num_capture_layers": 3,
            "normalize_target_hidden_state": False,
            # Whether the final norm (pre-lm_head) is handled inside the layers.
            # If False, the wrapper applies self.norm after the layers.
            # If True, layers have their own final_layernorm and wrapper skips self.norm.
            "layers_handle_final_norm": False,
            # Llama Eagle checkpoint: fc.*, midlayer.* -> model.fc.*, model.layers.*
            "_checkpoint_conversion_mapping": {
                "^(?!lm_head|norm)": "model.",
                "midlayer": "layers",
            },
        },
        "nemotron_h": {
            "load_embedding_from_target": True,
            "load_lm_head_from_target": True,
            "num_capture_layers": 1,
            "normalize_target_hidden_state": True,
            "mtp_hybrid_override_pattern": "*E",
            # NemotronH MTP layers have final_layernorm on the last layer,
            # so the wrapper should NOT apply an additional norm.
            "layers_handle_final_norm": True,
            # NemotronH MTP checkpoint: mtp.* -> model.*
            "_checkpoint_conversion_mapping": {
                r"^mtp\.": "model.",
            },
        },
    }
    # Some custom HF config classes expose backward-compatibility fields as properties instead of
    # storing them directly in __dict__. Those values do not survive config.to_dict(), so carry
    # them over explicitly before rebuilding a generic EagleConfig.
    _preserved_config_attrs: Dict[str, tuple[str, ...]] = {
        "nemotron_h": ("mtp_hybrid_override_pattern",),
    }

    def __init__(self, config: PretrainedConfig, model_type: str):
        if model_type not in self._drafter_defaults:
            raise ValueError(
                f"Unsupported model_type '{model_type}' for EagleConfig. "
                f"Supported types: {list(self._drafter_defaults.keys())}"
            )

        defaults = self._drafter_defaults[model_type]
        config_dict = config.to_dict()
        for key in self._preserved_config_attrs.get(model_type, ()):
            if key not in config_dict and hasattr(config, key):
                config_dict[key] = getattr(config, key)

        # Log when config overrides a default
        for key, value in defaults.items():
            if key in config_dict and config_dict[key] != value:
                ad_logger.info(
                    f"EagleConfig: config has '{key}={config_dict[key]}', "
                    f"overriding default '{value}'"
                )

        merged = deep_merge_dicts(defaults, config_dict)
        super().__init__(**merged)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, dim, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.base = getattr(config, "rope_theta", 10000.0)
        self.config = config

        self.factor = 2

        max_position_embeddings = self.config.max_position_embeddings

        if (
            not hasattr(config, "rope_type")
            or config.rope_type is None
            or config.rope_type == "default"
        ):
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
            )
            self.max_seq_len_cached = max_position_embeddings

        elif config.rope_type == "ntk":
            assert self.config.orig_max_position_embeddings is not None
            orig_max_position_embeddings = self.config.orig_max_position_embeddings

            self.base = self.base * (
                (self.factor * max_position_embeddings / orig_max_position_embeddings)
                - (self.factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
            )

            self.max_seq_len_cached = orig_max_position_embeddings
        else:
            raise ValueError(f"Not support rope_type: {config.rope_type}")

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)

    else:
        q_embed = None

    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed


class EagleRMSNorm(nn.Module):
    """RMSNorm implementation that uses the torch_rmsnorm custom op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        result = torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )
        return result


class EagleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        dtype = config.torch_dtype
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias, dtype=dtype
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias, dtype=dtype
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias, dtype=dtype
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Eagle3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.is_causal = True
        dtype = config.torch_dtype

        # Note: Eagle3Attention expects 2 * hidden_size input, which is the concatenation of the hidden states
        # and the input embeddings.

        self.q_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.k_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.v_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        cos, sin = position_embeddings

        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [Batch, Seq, Heads, Dim]
        query_states = query_states.view(bsz, q_len, -1, self.head_dim)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )

        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=self.is_causal,
            layout="bsnd",
        )

        attn_output = attn_output.view(bsz, q_len, self.num_attention_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output


# =============================================================================
# Llama-Specific Eagle Layer
# =============================================================================


class LlamaEagleLayer(nn.Module):
    """Eagle decoder layer for Llama-family models.

    Architecture:
    - Normalize embeds and hidden states, concatenate to 2*hidden_size
    - Self-attention with RoPE (computed internally from position_ids)
    - Add residual
    - Normalize, gated MLP (SwiGLU), add residual
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype

        # Normalization layers
        self.hidden_norm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention (expects 2*hidden_size input from concat)
        self.self_attn = Eagle3Attention(config, layer_idx=layer_idx)

        # MLP (gated SwiGLU style)
        self.mlp = EagleMLP(config)

        # RoPE
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            config=config, dim=self.head_dim, device=torch.device("cuda")
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Forward pass with unified interface.

        Args:
            hidden_states: Hidden states from target model [batch, seq, hidden_size]
            inputs_embeds: Token embeddings [batch, seq, hidden_size]
            position_ids: Position IDs for RoPE [batch, seq]

        Returns:
            Updated hidden states [batch, seq, hidden_size]
        """
        # Compute RoPE internally
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        # Normalize and concatenate embeds + hidden states
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        embeds = self.input_layernorm(inputs_embeds)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # Self-attention with RoPE
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class EagleModel(nn.Module):
    """Generic Eagle model architecture.

    This model is model-agnostic - it accepts layers from the factory and passes
    position_ids through to them. Layers handle model-specific logic (e.g., RoPE)
    internally.

    Args:
        config: Model configuration
        layers: nn.ModuleList of layers (for multi-layer) or single nn.Module (for single layer).
                Each layer implements the unified forward signature:
                forward(hidden_states, inputs_embeds, position_ids) -> Tensor
    """

    def __init__(self, config, layers: Union[nn.ModuleList, nn.Module]):
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype

        load_embedding_from_target = getattr(config, "load_embedding_from_target", False)
        self.embed_tokens = (
            None
            if load_embedding_from_target
            else nn.Embedding(config.vocab_size, config.hidden_size)
        )

        # Vocab mapping for draft -> target token conversion
        draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        if draft_vocab_size != config.vocab_size:
            self.d2t = nn.Parameter(
                torch.empty((draft_vocab_size,), dtype=torch.int32),
                requires_grad=False,
            )

        # Hidden size compression for target hidden states (multi-layer capture)
        num_capture_layers = getattr(config, "num_capture_layers", 1)
        self.fc = (
            nn.Linear(
                config.hidden_size * num_capture_layers,
                config.hidden_size,
                bias=getattr(config, "bias", False),
                dtype=self.dtype,
            )
            if num_capture_layers > 1
            else None
        )

        # Layers (injected by factory - model-specific)
        # Can be ModuleList (multi-layer) or single Module (single layer) for checkpoint compat
        # No rotary_emb here - layers handle RoPE internally if needed
        self.layers = layers

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the Eagle model.

        Args:
            inputs_embeds: Token embeddings [batch, seq, hidden_size]
            position_ids: Position IDs [batch, seq] - passed to layers
            hidden_states: Hidden states from target model [batch, seq, hidden_size]

        Returns:
            Updated hidden states [batch, seq, hidden_size]
        """
        # Pass position_ids through to layers - they decide what to do with it
        # (e.g., Llama layers compute RoPE, NemotronH layers ignore it)
        if isinstance(self.layers, nn.ModuleList):
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                )
        elif isinstance(self.layers, nn.Module):
            hidden_states = self.layers(
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
            )
        else:
            raise TypeError(
                f"Expected self.layers to be nn.ModuleList or nn.Module, got {type(self.layers).__name__}"
            )

        return hidden_states


@dataclass
class Eagle3DraftOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    norm_hidden_state: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class EagleDrafterForCausalLM(PreTrainedModel):
    """HuggingFace-compatible wrapper for EagleModel.

    This wrapper makes EagleModel compatible with AutoDeploy's model loading
    and inference pipeline. It accepts layers from the factory to enable
    model-specific layer implementations.

    Args:
        config: Model configuration (should be EagleConfig with model-type specific defaults)
        layers: Layers to use in EagleModel. Can be nn.ModuleList (multi-layer) or a single
                nn.Module (single-layer). If None, builds based on model_type.
    """

    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LlamaEagleLayer", "NemotronHEagleLayer"]

    def __init__(self, config, layers: Optional[Union[nn.ModuleList, nn.Module]] = None):
        super().__init__(config)

        # Read checkpoint conversion mapping from config (set by EagleConfig based on model_type)
        self._checkpoint_conversion_mapping = getattr(
            config, "_checkpoint_conversion_mapping", None
        )

        self.load_embedding_from_target = getattr(config, "load_embedding_from_target", False)
        self.load_lm_head_from_target = getattr(config, "load_lm_head_from_target", False)
        # Whether layers handle the final norm (pre-lm_head) internally.
        # If True, layers have their own final_layernorm and we skip self.norm in forward.
        # If False (default), we apply self.norm after the layers.
        self._layers_handle_final_norm = getattr(config, "layers_handle_final_norm", False)

        # If layers not provided, build based on model_type
        if layers is None:
            layers = get_eagle_layers(config, config.model_type)

        self.model = EagleModel(config, layers)

        # Only create norm if layers don't handle final normalization internally.
        if not self._layers_handle_final_norm:
            # Use fallback chain for eps: rms_norm_eps (Llama) -> layer_norm_epsilon (NemotronH) -> default
            norm_eps = getattr(config, "rms_norm_eps", getattr(config, "layer_norm_epsilon", 1e-6))
            self.norm = EagleRMSNorm(config.hidden_size, eps=norm_eps)
        else:
            self.norm = None
        # draft_vocab_size defaults to vocab_size if not specified
        draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.lm_head = (
            None
            if self.load_lm_head_from_target
            else nn.Linear(
                config.hidden_size, draft_vocab_size, bias=False, dtype=config.torch_dtype
            )
        )

        eagle_config = getattr(config, "eagle_config", {})
        self._return_hidden_post_norm = eagle_config.get("return_hidden_post_norm", False)

    def forward(
        self,
        inputs_embeds: torch.LongTensor,
        position_ids: torch.LongTensor,
        **kwargs,
    ) -> Eagle3DraftOutput:
        """Forward pass for Eagle drafter.

        Args:
            inputs_embeds: Input token embeddings [batch, seq, hidden_size]
            position_ids: Position IDs [batch, seq]. Required.
            **kwargs: Must contain 'hidden_states' from the target model.

        Returns:
            Eagle3DraftOutput with norm_hidden_state and last_hidden_state.

        Raises:
            ValueError: If hidden_states or position_ids is not provided.
        """
        if position_ids is None:
            raise ValueError("position_ids must be provided.")
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            raise ValueError("hidden_states must be provided.")

        hidden_states = self.model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, hidden_states=hidden_states
        )

        # Apply final norm only if layers don't handle it internally.
        # For Llama: layers don't normalize, so we apply self.norm here.
        # For NemotronH: layers have final_layernorm, so hidden_states are already normalized.
        if self.norm is not None:
            norm_hidden_state = self.norm(hidden_states)
        else:
            norm_hidden_state = hidden_states  # already normalized by layer

        last_hidden_state = norm_hidden_state if self._return_hidden_post_norm else hidden_states

        return Eagle3DraftOutput(
            norm_hidden_state=norm_hidden_state,
            last_hidden_state=last_hidden_state,
        )

    def get_input_embeddings(self):
        if self.model.embed_tokens is not None:
            return self.model.embed_tokens
        else:
            raise NotImplementedError(
                "EagleDrafterForCausalLM does not have an input embedding layer."
            )

    def get_output_embeddings(self):
        if self.lm_head is not None:
            return self.lm_head
        else:
            raise NotImplementedError(
                "EagleDrafterForCausalLM does not have an output embedding layer."
            )


@dataclass
class EagleWrapperOutput(ModelOutput):
    """Output format compatible with Eagle3OneModelSampler/MTPSampler.

    This output format allows the one-model speculative decoding flow to bypass
    logits-based sampling in the sampler. The EagleWrapper performs all sampling
    and verification internally, returning pre-computed tokens.
    """

    # logits: [batch_size, 1, vocab_size].  Used for compatibility.
    logits: Optional[torch.Tensor] = None

    # new_tokens: [batch_size, max_draft_len + 1]. Accepted tokens from verification.
    # This is a 2D tensor where each row contains the accepted tokens for a request,
    # padded if fewer tokens were accepted.
    new_tokens: Optional[torch.Tensor] = None

    # new_tokens_lens: [batch_size]. Number of newly accepted tokens per request in this iteration.
    new_tokens_lens: Optional[torch.Tensor] = None

    # next_draft_tokens: [batch_size, max_draft_len]. Draft tokens for the next iteration.
    # These are the tokens predicted by the draft model, already converted via d2t.
    next_draft_tokens: Optional[torch.Tensor] = None

    # next_new_tokens: [batch_size, max_draft_len + 1]. Input tokens for the next iteration.
    # Format: [last_accepted_token, draft_token_0, draft_token_1, ...]
    next_new_tokens: Optional[torch.Tensor] = None


@dataclass
class EagleWrapperConfig:
    max_draft_len: int
    load_embedding_from_target: bool
    load_lm_head_from_target: bool
    normalize_target_hidden_state: bool = False
    sync_before_hidden_state_capture: bool = False


class EagleWrapper(nn.Module):
    """Combined target + draft model for one-model Eagle speculative decoding.

    Dual-purpose:
    1. Prefill-only mode (export time): runs both models without KV cache for graph capture.
    2. KV-cache mode (inference): runs graph-captured models with cached attention and
       in-place metadata updates. Hidden states flow through layer-prefixed
       `*_hidden_states_cache` kwargs.
    """

    _requires_csi: bool = True

    def __init__(self, config: EagleWrapperConfig, target_model: nn.Module, draft_model: nn.Module):
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.max_draft_len = config.max_draft_len
        self.load_embedding_from_target = config.load_embedding_from_target
        self.load_lm_head_from_target = config.load_lm_head_from_target
        self.normalize_target_hidden_state = config.normalize_target_hidden_state
        self.sync_before_hidden_state_capture = config.sync_before_hidden_state_capture
        self._draft_hidden_size = draft_model.config.hidden_size
        self._draft_num_capture_layers = getattr(draft_model.config, "num_capture_layers", 1)
        self._buffers_initialized = False
        self._buf_new_tokens_2d: Optional[torch.Tensor] = None
        self._buf_next_new_tokens: Optional[torch.Tensor] = None
        self._buf_new_tokens_lens: Optional[torch.Tensor] = None
        self._buf_c_offset_ones: Optional[torch.Tensor] = None
        self._buf_hidden_states: Optional[torch.Tensor] = None

    @property
    def _draft_inner_model(self):
        """Get the inner model submodule of the draft model.

        Before export: self.draft_model.model (EagleModel inside EagleDrafterForCausalLM).
        After export: self.draft_model.model (preserved by DraftModelExportInfo.post_process).
        """
        return self.draft_model.model

    @property
    def _draft_dtype(self):
        """Get the dtype of the draft model (works before and after export)."""
        return getattr(self._draft_inner_model, "dtype", None) or torch.bfloat16

    def normalize_target_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the target model's final normalization to hidden states.

        MTP hidden states are captured at the residual add (pre-norm_f), but the
        MTP head expects post-norm_f input. The target model must expose
        get_final_normalization() for this to work.
        """
        norm_fn = getattr(self.target_model, "get_final_normalization", None)
        if norm_fn is None:
            raise RuntimeError(
                "MTP requires the target model to expose get_final_normalization(), "
                f"but {type(self.target_model).__name__} does not implement it."
            )
        return norm_fn()(hidden_states)

    def apply_eagle3_fc(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the fc layer that fuses hidden states from multiple target layers."""
        hidden_states = hidden_states.to(self._draft_dtype)

        fc = getattr(self._draft_inner_model, "fc", None)
        if fc is not None:
            hidden_states = fc(hidden_states)
        return hidden_states

    # TODO: go through this logic, not sure if it is correct at the moment
    def apply_d2t(self, draft_output_ids: torch.Tensor) -> torch.Tensor:
        """Apply draft-to-target token mapping if available."""
        d2t = getattr(self._draft_inner_model, "d2t", None)
        if d2t is not None:
            draft_output_ids = d2t[draft_output_ids] + draft_output_ids
        return draft_output_ids

    def apply_draft_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply embedding to input_ids for the draft model."""
        if self.load_embedding_from_target:
            embeds = self.target_model.get_input_embeddings()(input_ids)
            return embeds.to(self._draft_dtype)
        else:
            return self.draft_model.get_input_embeddings()(input_ids)

    def apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply lm_head to get logits from hidden states."""
        if self.load_lm_head_from_target:
            lm_head_weights = self.target_model.get_output_embeddings()(hidden_states)
            return lm_head_weights.to(self._draft_dtype)
        else:
            return self.draft_model.get_output_embeddings()(hidden_states)

    def sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        ret = torch.argmax(logits, dim=-1)
        return ret

    def forward(self, cache_seq_interface: Optional[CachedSequenceInterface] = None, **kwargs):
        """Dispatch to appropriate forward implementation.

        - If seq_info is provided: inference mode (after graph transforms + cache init).
        - Otherwise: prefill-only mode (export time, before caches are inserted).
        """
        if cache_seq_interface is not None:
            return self._forward_with_kv_cache(cache_seq_interface)
        else:
            return self._forward_prefill_only(**kwargs)

    def _forward_prefill_only(self, input_ids: torch.Tensor, position_ids: torch.Tensor, **kwargs):
        """Forward pass without KV cache (prefill-only mode, used at export time).

        Runs each submodule once so that export capture hooks see the correct kwargs.
        Mirrors the simplest real inference case: a pure context batch where all input
        tokens are accepted, the target model samples a bonus token, and the draft model
        produces one draft token.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # --- Phase 1: Target model forward ---
        input_embeds = self.target_model.get_input_embeddings()(input_ids)
        target_logits = self.target_model(
            inputs_embeds=input_embeds, position_ids=position_ids
        ).logits

        # Context case: sample bonus token from last position logits per batch
        bonus_token = self.sample_greedy(target_logits[:, -1, :])  # [batch_size]

        # --- Phase 2: Draft model forward (single iteration) ---
        # Construct draft input: input_ids shifted left by 1 with bonus token at end,
        draft_input_ids = input_ids.roll(-1, dims=1)
        draft_input_ids[:, -1] = bonus_token

        draft_config = self.draft_model.config
        hidden_size = draft_config.hidden_size
        num_capture_layers = getattr(draft_config, "num_capture_layers", 1)
        hidden_states = torch.zeros(
            batch_size * seq_len,
            hidden_size * num_capture_layers,
            device=device,
            dtype=input_embeds.dtype,
        )
        hidden_states = self.apply_eagle3_fc(hidden_states)
        hidden_states = hidden_states.view(batch_size, seq_len, -1)

        draft_embeds = self.apply_draft_embedding(draft_input_ids)
        draft_output = self.draft_model(
            inputs_embeds=draft_embeds,
            position_ids=position_ids,
            hidden_states=hidden_states,
        )

        # Sample one draft token from last position
        draft_logits = self.apply_lm_head(draft_output.norm_hidden_state)
        draft_token = self.sample_greedy(draft_logits[:, -1, :])  # [batch_size]
        draft_token = self.apply_d2t(draft_token)

        # --- Package output: [bonus_token, draft_token] per batch ---
        new_tokens = torch.stack([bonus_token, draft_token], dim=1)  # [batch_size, 2]

        return EagleWrapperOutput(
            new_tokens=new_tokens,
            new_tokens_lens=torch.ones(batch_size, dtype=torch.long, device=device),
        )

    # ================================================================== #
    #  KV-cache forward (inference after graph transforms)               #
    # ================================================================== #

    @staticmethod
    def _filter_kwargs_for_submodule(kwargs: dict, submodule: nn.Module) -> dict:
        """Filter kwargs to only include those accepted by submodule's forward (GraphModule)."""
        expected_names = {node.name for node in submodule.graph.nodes if node.op == "placeholder"}
        return {k: v for k, v in kwargs.items() if k in expected_names}

    def _ensure_buffers(self, csi: CachedSequenceInterface) -> None:
        """Lazily initialize stable output buffers for the KV-cache forward path."""
        if self._buffers_initialized:
            return

        max_batch_size = csi.info.max_batch_size
        max_num_tokens = csi.info.max_num_tokens
        device = csi.info.device
        ids_dtype = csi.get_arg("input_ids").dtype

        hidden_state_buffers = [
            tensor
            for name, tensor in csi.named_args.items()
            if name.endswith("hidden_states_cache")
        ]
        hidden_states_dtype = (
            hidden_state_buffers[0].dtype if hidden_state_buffers else self._draft_dtype
        )
        hidden_states_width = self._draft_hidden_size * self._draft_num_capture_layers

        self._buf_new_tokens_2d = torch.empty(
            max_batch_size,
            self.max_draft_len + 1,
            dtype=ids_dtype,
            device=device,
        )
        self._buf_next_new_tokens = torch.empty(
            max_batch_size,
            self.max_draft_len + 1,
            dtype=ids_dtype,
            device=device,
        )
        self._buf_new_tokens_lens = torch.ones(max_batch_size, dtype=torch.int32, device=device)
        self._buf_c_offset_ones = torch.ones(max_batch_size, dtype=torch.int32, device=device)
        self._buf_hidden_states = torch.empty(
            max_num_tokens,
            hidden_states_width,
            dtype=hidden_states_dtype,
            device=device,
        )
        self._buffers_initialized = True

    def _collect_hidden_states(self, kwargs: dict, num_tokens: int) -> torch.Tensor:
        """Read layer-prefixed ``*_hidden_states_cache`` buffers into a pre-allocated concat buffer.

        Returns:
            Tensor of shape [num_tokens, hidden_size * num_capture_layers].
        """
        assert self._buf_hidden_states is not None, "_ensure_buffers must run before collecting."
        buffers = sorted(
            [
                (name, tensor)
                for name, tensor in kwargs.items()
                if name.endswith("hidden_states_cache")
            ],
            key=lambda x: x[0],
        )
        if not buffers:
            raise ValueError("No *_hidden_states_cache buffers found in kwargs.")
        offset = 0
        for _, buf in buffers:
            width = buf.shape[1]
            self._buf_hidden_states[:num_tokens, offset : offset + width].copy_(buf[:num_tokens])
            offset += width
        return self._buf_hidden_states[:num_tokens, :offset]

    def _forward_with_kv_cache(self, csi: CachedSequenceInterface):
        """Forward pass with KV cache (inference after graph transforms).

        Phases: target forward -> collect hidden states -> gather + sample + verify ->
                draft loop with in-place metadata updates -> package output.

        Expected kwargs that are accessed directly are described in the required_kwargs property.
        Additional kwargs are forwarded to target/draft submodules (kv caches, etc.).
        """
        self._ensure_buffers(csi)
        assert self._buf_new_tokens_2d is not None
        assert self._buf_next_new_tokens is not None
        assert self._buf_new_tokens_lens is not None
        assert self._buf_c_offset_ones is not None

        # ---- Phase 0: Check batch information ----
        # determine batch information from batch_info_host
        batch_info = csi.info.batch_info
        num_prefill, num_extend, num_decode = batch_info.get_num_sequences()
        num_prefill_tokens, num_extend_tokens, num_decode_tokens = batch_info.get_num_tokens()
        num_sequences = num_prefill + num_extend + num_decode
        num_total_tokens = num_prefill_tokens + num_extend_tokens + num_decode_tokens

        # some sanity checks on the batch
        assert num_decode == 0, "decode without drafting is not supported inside the eagle wrapper"
        if num_extend > 0:
            assert num_extend_tokens // num_extend == 1 + self.max_draft_len, "Unexpected draft len"

        # ---- Phase 1: Target model forward ----
        out = self.target_model(
            inputs_embeds=self.target_model.get_input_embeddings()(csi.get_arg("input_ids")),
            **self._filter_kwargs_for_submodule(csi.named_args, self.target_model),
        )
        # NOTE: we assume gather_context_logits is False so that gathering here works!
        target_logits = csi.info.maybe_gather_and_squeeze(out.logits)

        # ---- Phase 2: Collect hidden states from cache buffers ----
        if self.sync_before_hidden_state_capture:
            # FlashInfer speculative decoding still relies on the target-written
            # hidden_states_cache buffers being visible before we concatenate them
            # for verification. TRTLLM attention does not require this sync, and
            # speculative FlashInfer is currently kept on torch-simple rather than
            # cudagraph replay.
            torch.cuda.synchronize()
        hidden_states = self._collect_hidden_states(csi.named_args, num_total_tokens)
        if self.normalize_target_hidden_state:
            # MTP: hidden states are captured at the residual add (pre-normalization).
            # Apply the target model's final normalization to match the PyTorch backend
            # which passes normalized hidden_states to MTPEagleWorker.
            hidden_states = self.normalize_target_hidden_states(hidden_states)
            # Cast to draft model dtype (e.g. target may be FP8, draft BF16).
            hidden_states = hidden_states.to(self._draft_dtype)
        else:
            # Eagle3: compress hidden states from multiple captured layers via fc.
            # apply_eagle3_fc also handles the target->draft dtype cast.
            hidden_states = self.apply_eagle3_fc(hidden_states)

        # ---- Phase 3: Sample ----
        # check dtype/device
        ids_dtype = csi.get_arg("input_ids").dtype

        # sample tokens from gathered logits
        sampled_tokens = self.sample_greedy(target_logits).to(ids_dtype)

        # store the new tokens sampled with the target model in 2d grid as expected by runtime. This
        # includes:
        # 1. idx=0: golden/bonus token for prefill+extend sequences --> guaranteed new token
        # 2. idx>1: target-sampled+accepted/declined tokens from previous draft iteration
        new_tokens_2d_extend = sampled_tokens[num_prefill:].view(num_extend, 1 + self.max_draft_len)
        new_tokens_2d = self._buf_new_tokens_2d[:num_sequences]
        if num_prefill > 0:
            new_tokens_2d.zero_()
            new_tokens_2d[:num_prefill, 0] = sampled_tokens[:num_prefill]
        if num_extend > 0:
            new_tokens_2d[num_prefill:] = new_tokens_2d_extend

        # ---- Phase 4: Verify ----
        # get original input ids
        input_ids_flat = csi.get_arg("input_ids", unflatten=False, truncate=True)

        # build output ids (and input ids for initial draft iteration). Those consist of
        # - prefill input ids rolled left by 1 with the bonus token at the end
        # - all sampled tokens from the extend requests
        # Either way the total number of tokens is the same as in the target model call just before!
        if num_prefill > 0:
            output_ids_target = input_ids_flat.roll(-1, dims=0)  # [total_tokens]
            lgi_prefill = csi.get_arg("token_gather_indices")[:num_prefill]
            output_ids_target[lgi_prefill] = new_tokens_2d[:num_prefill, 0]
            output_ids_target[num_prefill_tokens:] = new_tokens_2d[num_prefill:].flatten()
        else:
            output_ids_target = new_tokens_2d.flatten()  # [total_tokens]

        # build new_tokens_lens
        if num_extend > 0:
            input_ids_extend = input_ids_flat[num_prefill_tokens:].view(num_extend, -1)
            mask_same = new_tokens_2d_extend[:, :-1] == input_ids_extend[:, 1:]
            # + 1 since it's the bonus token this is not counted in the cumprod. Note that
            # 1 <= new_tokens_lens_extend <= max_draft_len + 1
            new_tokens_lens_extend = mask_same.cumprod(dim=1).sum(dim=1, dtype=torch.int32) + 1

        if num_prefill == 0:
            new_tokens_lens = new_tokens_lens_extend
        else:
            new_tokens_lens = self._buf_new_tokens_lens[:num_sequences]
            new_tokens_lens.fill_(1)
            if num_extend > 0:
                new_tokens_lens[num_prefill:] = new_tokens_lens_extend

        # MTP state promotion: commit accepted intermediate mamba states to base state
        # immediately after verification, before cache offset computation and draft loop.
        # Must happen inside model forward (not in ad_executor) for correct timing —
        # update_mamba_states reads .num_seqs and .num_contexts from attn_metadata.
        kv_cache_manager = csi.kv_cache_manager
        if num_extend > 0 and isinstance(kv_cache_manager, MambaHybridCacheManager):
            if kv_cache_manager.is_speculative():
                _ctx = SimpleNamespace(num_seqs=num_sequences, num_contexts=num_prefill)
                kv_cache_manager.update_mamba_states(
                    attn_metadata=_ctx,
                    num_accepted_tokens=new_tokens_lens,
                )

        # compute the cache and position offset based on the number of new tokens compared to the
        # maximum draft length. NOTE: cache is currently at the position corresponding to the last
        # draft token. Hence the following constraint is true:
        # c_offset[:num_prefill] == 1
        # -(max_draft_len-1) <= c_offset <= 1
        c_offset = new_tokens_lens - self.max_draft_len
        if num_prefill > 0:
            c_offset[:num_prefill].fill_(1)

        # updated token_gather_indices and info based on c_offset for retrieval of
        # last accepted tokens for both output and first draft iteration. It's computed as follows:
        # last_token_index = cu_seqlen[1:] - 1
        # num_draft_tokens_rejected = 1 - c_offset
        # last_accepted_tokens = last_token_index - num_draft_tokens_rejected
        # last_accepted_tokens = cu_seqlen[1:] + c_offset - 2
        csi.info.batch_info.update_tokens_gather_info(num_sequences, True)
        last_accepted_tokens = csi.get_arg("cu_seqlen", truncate=True)[1:] + c_offset - 2
        csi.info.copy_("token_gather_indices", last_accepted_tokens, strict=False)

        # ---- Phase 4: Prepare for draft loop and next_new_tokens tensor ----
        ids_dtype = output_ids_target.dtype

        # store current collected output ids asinput ids for the first iteration of the draft loop
        csi.info.copy_("input_ids", output_ids_target)

        # a 2D grid of latest verified + new draft tokens for each sequence. This includes:
        # 1. idx=0: latest verified token or bonus token for prefill+extend sequences
        # 2. idx>1: new draft tokens that will be drafted below
        next_new_tokens = self._buf_next_new_tokens[:num_sequences]

        # we can already fill in next_new_tokens for idx=0 which corresponds to last accepted tokens
        # which we already stored in the input_ids for the first draft iteration.
        next_new_tokens[:, 0] = csi.info.maybe_gather_and_squeeze(csi.get_arg("input_ids"))

        # ---- Phase 5: Draft loop ----
        for draft_idx in range(self.max_draft_len):
            # run forward pass on the draft model in shape [num_sequences, 1]
            draft_output = self.draft_model(
                inputs_embeds=self.apply_draft_embedding(csi.get_arg("input_ids")),
                hidden_states=csi.info.unflatten(hidden_states),
                **self._filter_kwargs_for_submodule(csi.named_args, self.draft_model),
            )
            draft_output_logits = self.apply_lm_head(draft_output.norm_hidden_state)

            # extract gathered logits and hidden state if not yet done as part of draft forward pass
            latest_logits = csi.info.maybe_gather_and_squeeze(draft_output_logits)
            hidden_states = csi.info.maybe_gather_and_squeeze(draft_output.last_hidden_state)

            # sample from logits which is output from this draft iteration and input for the next
            draft_tokens = next_new_tokens[:, draft_idx + 1]
            draft_tokens[:] = self.sample_greedy(latest_logits)
            draft_tokens[:] = self.apply_d2t(draft_tokens)

            # update cache offset for the next draft iteration
            # for idx=0 --> we use pre-computed c_offset from the verification step
            # for idx>1 --> we offset uniformly by 1
            if draft_idx == 1:
                c_offset = self._buf_c_offset_ones[:num_sequences]

            # switch to generate (if not done already), store new tokens, and offset cache
            # can be skipped for last iteration since after we return metadata will be reset
            if draft_idx < self.max_draft_len - 1:
                csi.info.switch_to_generate_()
                csi.info.copy_("input_ids", draft_tokens)
                csi.info.offset_pos_and_cache_(c_offset)

        # ---- Phase 6: Package output ----
        return EagleWrapperOutput(
            logits=target_logits,
            new_tokens=new_tokens_2d,
            new_tokens_lens=new_tokens_lens,
            next_draft_tokens=next_new_tokens[:, 1:],
            next_new_tokens=next_new_tokens,
        )
