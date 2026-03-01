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

"""Eagle3 model implementation for AutoDeploy.

Eagle3 is a speculative decoding draft model that predicts next tokens based on
hidden states from a target model (e.g., Llama-3.1-8B-Instruct).

This file contains model definitions used for executing Eagle3 speculative decoding in AutoDeploy.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from ...shim.interface import CachedSequenceInterface
from ...utils._config import deep_merge_dicts
from ...utils.logger import ad_logger


class EagleConfig(PretrainedConfig):
    """Config for Eagle3 drafter models.

    Extends PretrainedConfig with Eagle-specific parameters while preserving
    all base model config values.

    Args:
        config: Base config for the draft model from its config.json.
        model_type: The base model type (e.g., "llama") used to look up defaults.
    """

    # Map model_type -> default Eagle config values
    _drafter_defaults: Dict[str, Dict[str, Any]] = {
        "llama": {
            "load_embedding_from_target": True,
            "load_lm_head_from_target": False,
            "num_capture_layers": 3,
        },
    }

    def __init__(self, config: PretrainedConfig, model_type: str):
        if model_type not in self._drafter_defaults:
            raise ValueError(
                f"Unsupported model_type '{model_type}' for EagleConfig. "
                f"Supported types: {list(self._drafter_defaults.keys())}"
            )

        defaults = self._drafter_defaults[model_type]
        config_dict = config.to_dict()

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
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
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

        # Note: Eagle3Attention expects 2 * hidden_size input, which is the concatenation of the hidden states
        # and the input embeddings.

        self.q_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
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


class Eagle3DecoderLayer(nn.Module):
    """Eagle decoder layer with modified attention and hidden state normalization."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.dtype = config.torch_dtype
        self.self_attn = Eagle3Attention(config, layer_idx=layer_idx)
        self.hidden_norm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = EagleMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embeds: torch.Tensor,
        position_embeds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)

        embeds = self.input_layernorm(embeds)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeds,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Eagle3Model(nn.Module):
    """Core Eagle model architecture."""

    def __init__(self, config):
        super().__init__()

        self.dtype = config.torch_dtype

        load_embedding_from_target = getattr(config, "load_embedding_from_target", False)
        self.embed_tokens = (
            None
            if load_embedding_from_target
            else nn.Embedding(config.vocab_size, config.hidden_size)
        )

        if config.draft_vocab_size is not None and config.draft_vocab_size != config.vocab_size:
            # Vocab mappings for draft <-> target token conversion
            # Needed to convert draft outputs to target inputs for Eagle3.
            # Since we reuse the target model's embedding in the drafter, we need
            # to do this conversion after every draft iteration.
            self.d2t = nn.Parameter(
                torch.empty((config.draft_vocab_size,), dtype=torch.int32),
                requires_grad=False,
            )

        # Hidden size compression for target hidden states.
        # Assumption: No feedforward fusion needed if we have just one capture layer (valid for MTPEagle)
        self.fc = (
            nn.Linear(
                config.hidden_size * config.num_capture_layers,
                config.hidden_size,
                bias=getattr(config, "bias", False),
                dtype=self.dtype,
            )
            if config.num_capture_layers > 1
            else None
        )

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        self.rotary_emb = LlamaRotaryEmbedding(
            config=config, dim=self.head_dim, device=torch.device("cuda")
        )

        if config.num_hidden_layers > 1:
            self.midlayer = nn.ModuleList(
                [Eagle3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
            )
        else:
            self.midlayer = Eagle3DecoderLayer(config, layer_idx=0)

        self.num_hidden_layers = config.num_hidden_layers

    # Assumption: The hidden states are already fused if necessary
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeds = (cos, sin)

        if self.num_hidden_layers > 1:
            for layer in self.midlayer:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    embeds=inputs_embeds,
                    position_embeds=position_embeds,
                )
        else:
            hidden_states = self.midlayer(
                hidden_states=hidden_states,
                embeds=inputs_embeds,
                position_embeds=position_embeds,
            )

        return hidden_states


@dataclass
class Eagle3DraftOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    norm_hidden_state: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class Eagle3DrafterForCausalLM(PreTrainedModel):
    """HuggingFace-compatible wrapper for EagleModel.

    This wrapper makes EagleModel compatible with AutoDeploy's model loading
    and inference pipeline.
    """

    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Eagle3DecoderLayer"]

    # Checkpoint conversion mapping: Eagle checkpoints have keys like "fc.weight"
    # but the wrapper model expects "model.fc.weight" (due to self.model = Eagle3Model).
    # This mapping tells the factory to add "model." prefix when loading weights.
    # Used by AutoModelForCausalLMFactory._remap_param_names_load_hook()

    _checkpoint_conversion_mapping = {
        "^(?!lm_head|norm)": "model.",  # Prepend "model." to all keys EXCEPT lm_head and norm
    }

    def __init__(self, config):
        super().__init__(config)

        self.load_embedding_from_target = getattr(config, "load_embedding_from_target", False)
        self.load_lm_head_from_target = getattr(config, "load_lm_head_from_target", False)

        self.model = Eagle3Model(config)
        self.norm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = (
            None
            if self.load_lm_head_from_target
            else nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
        )

        eagle_config = getattr(config, "eagle_config", {})
        self._return_hidden_post_norm = eagle_config.get("return_hidden_post_norm", False)

    def forward(
        self,
        inputs_embeds: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Eagle3DraftOutput:
        """
        Kwargs:
            hidden_states: Hidden states from the target model. Required.

        Raises:
            ValueError: If hidden_states is not provided in kwargs.
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            raise ValueError("hidden_states must be provided.")

        hidden_states = self.model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, hidden_states=hidden_states
        )

        norm_hidden_state = self.norm(hidden_states)

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
                "Eagle3DrafterForCausalLM does not have an input embedding layer."
            )

    def get_output_embeddings(self):
        if self.lm_head is not None:
            return self.lm_head
        else:
            raise NotImplementedError(
                "Eagle3DrafterForCausalLM does not have an output embedding layer."
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


class EagleWrapper(nn.Module):
    """Combined target + draft model for one-model Eagle speculative decoding.

    Dual-purpose:
    1. Prefill-only mode (export time): runs both models without KV cache for graph capture.
    2. KV-cache mode (inference): runs graph-captured models with cached attention and
       in-place metadata updates. Hidden states flow through hidden_states_cache_* kwargs.
    """

    def __init__(self, config: EagleWrapperConfig, target_model: nn.Module, draft_model: nn.Module):
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.max_draft_len = config.max_draft_len
        self.load_embedding_from_target = config.load_embedding_from_target
        self.load_lm_head_from_target = config.load_lm_head_from_target

    @property
    def _draft_inner_model(self):
        """Get the inner model submodule of the draft model.

        Before export: self.draft_model.model (Eagle3Model inside Eagle3DrafterForCausalLM).
        After export: self.draft_model.model (preserved by DraftModelExportInfo.post_process).
        """
        return self.draft_model.model

    @property
    def _draft_dtype(self):
        """Get the dtype of the draft model (works before and after export)."""
        return getattr(self._draft_inner_model, "dtype", None) or torch.bfloat16

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

    @staticmethod
    def _collect_hidden_states(kwargs: dict, num_tokens: int) -> torch.Tensor:
        """Read hidden_states_cache_* buffers from kwargs, concatenate, and slice.

        Returns:
            Tensor of shape [num_tokens, hidden_size * num_capture_layers].
        """
        # TODO: we should eventually use the resource manager to have them be concatenated before
        # and we just write into a view and hence can just take the whole tensor here.
        buffers = sorted(
            [
                (name, tensor)
                for name, tensor in kwargs.items()
                if name.startswith("hidden_states_cache")
            ],
            key=lambda x: x[0],
        )
        if not buffers:
            raise ValueError("No hidden_states_cache_* buffers found in kwargs.")
        hidden_states = torch.cat([buf[:num_tokens] for _, buf in buffers], dim=1)
        return hidden_states

    @property
    def required_kwargs(self) -> List[str]:
        return [
            "input_ids",
            "position_ids",
            "cu_seqlen",
            "batch_info_host",
            "tokens_gather_info_host",
            "token_gather_indices",
            # NOTE: below kwargs accessed in offset_pos_and_cache_
            # TODO: see if there is a better way to handle this... this property here should only
            # return kwargs explicit requested by the forward pass i/f.
            "seq_len",
            "extra_page_per_seq",
        ]

    def _forward_with_kv_cache(self, csi: CachedSequenceInterface):
        """Forward pass with KV cache (inference after graph transforms).

        Phases: target forward -> collect hidden states -> gather + sample + verify ->
                draft loop with in-place metadata updates -> package output.

        Expected kwargs that are accessed directly are described in the required_kwargs property.
        Additional kwargs are forwarded to target/draft submodules (kv caches, etc.).
        """
        # ---- Phase 0: Check batch information ----
        # determine batch information from batch_info_host
        batch_info_list = csi.get_arg("batch_info_host").tolist()
        num_prefill, num_extend, num_decode = batch_info_list[::2]
        num_prefill_tokens, num_extend_tokens, num_decode_tokens = batch_info_list[1::2]
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
        hidden_states = self._collect_hidden_states(csi.named_args, num_total_tokens)
        hidden_states = self.apply_eagle3_fc(hidden_states)

        # ---- Phase 3: Sample ----
        # check dtype/device
        device = csi.info.device
        ids_dtype = csi.get_arg("input_ids").dtype

        # sample tokens from gathered logits
        sampled_tokens = self.sample_greedy(target_logits).to(ids_dtype)

        # store the new tokens sampled with the target model in 2d grid as expected by runtime. This
        # includes:
        # 1. idx=0: golden/bonus token for prefill+extend sequences --> guaranteed new token
        # 2. idx>1: target-sampled+accepted/declined tokens from previous draft iteration
        new_tokens_2d_extend = sampled_tokens[num_prefill:].view(num_extend, 1 + self.max_draft_len)
        if num_prefill > 0:
            new_tokens_2d = torch.zeros(
                num_sequences, self.max_draft_len + 1, dtype=ids_dtype, device=device
            )
            new_tokens_2d[:num_prefill, 0] = sampled_tokens[:num_prefill]
            new_tokens_2d[num_prefill:] = new_tokens_2d_extend
        else:
            new_tokens_2d = new_tokens_2d_extend

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
            new_tokens_lens = torch.ones(num_sequences, dtype=torch.int32, device=device)
            if num_extend > 0:
                new_tokens_lens[num_prefill:] = new_tokens_lens_extend

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
        tokens_gather_info_host = csi.get_arg("tokens_gather_info_host")
        tokens_gather_info_host[0] = num_sequences
        tokens_gather_info_host[1] = True
        last_accepted_tokens = csi.get_arg("cu_seqlen", truncate=True)[1:] + c_offset - 2
        csi.info.copy_("token_gather_indices", last_accepted_tokens, strict=False)

        # ---- Phase 4: Prepare for draft loop and next_new_tokens tensor ----
        device = csi.info.device
        ids_dtype = output_ids_target.dtype

        # store current collected output ids asinput ids for the first iteration of the draft loop
        csi.info.copy_("input_ids", output_ids_target)

        # a 2D grid of latest verified + new draft tokens for each sequence. This includes:
        # 1. idx=0: latest verified token or bonus token for prefill+extend sequences
        # 2. idx>1: new draft tokens that will be drafted below
        next_new_tokens = torch.empty(
            num_sequences, self.max_draft_len + 1, dtype=ids_dtype, device=device
        )

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
                c_offset = torch.ones(num_sequences, dtype=torch.int32, device=device)

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
