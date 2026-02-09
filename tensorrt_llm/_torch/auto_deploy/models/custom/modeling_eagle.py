# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

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
    def __init__(self, config, target_model, draft_model, resource_manager):
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.resource_manager = resource_manager
        self.max_draft_len = config.max_draft_len
        self.load_embedding_from_target = config.load_embedding_from_target
        self.load_lm_head_from_target = config.load_lm_head_from_target

    def apply_eagle3_fc(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the fc layer that fuses hidden states from multiple target layers."""
        draft_model = self.draft_model.model
        hidden_states = hidden_states.to(draft_model.dtype)

        fc = getattr(draft_model, "fc", None)
        if fc is not None:
            hidden_states = fc(hidden_states)
        return hidden_states

    def apply_d2t(self, draft_output_ids: torch.Tensor) -> torch.Tensor:
        """Apply draft-to-target token mapping if available."""
        d2t = getattr(self.draft_model.model, "d2t", None)
        if d2t is not None:
            draft_output_ids = d2t[draft_output_ids] + draft_output_ids
        return draft_output_ids

    def apply_draft_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply embedding to input_ids for the draft model."""
        if self.load_embedding_from_target:
            embeds = self.target_model.get_input_embeddings()(input_ids)
            return embeds.to(self.draft_model.dtype)
        else:
            return self.draft_model.get_input_embeddings()(input_ids)

    def apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply lm_head to get logits from hidden states."""
        if self.load_lm_head_from_target:
            lm_head_weights = self.target_model.get_output_embeddings()(hidden_states)
            return lm_head_weights.to(self.draft_model.dtype)
        else:
            return self.draft_model.get_output_embeddings()(hidden_states)

    def sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        ret = torch.argmax(logits, dim=-1)
        return ret

    def sample_and_verify(
        self, input_ids, target_logits: torch.Tensor, num_previously_accepted: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            target_logits: [batch_size, seq_len, vocab_size]
            num_previously_accepted: [batch_size]. Number of input tokens accepted so far for each batch.

        Returns:
            output_ids: [batch_size, seq_len] (result of greedy sampling on input ids)
            num_newly_accepted_tokens: [batch_size]. Number of newly accepted tokens in each batch.
            num_accepted_tokens: [batch_size]. Number of tokens accepted in each batch, including previously accepted.
                So num_accepted_tokens[i] = num_previously_accepted + num_newly_accepted_tokens.
            last_logits_3d: [batch_size, 1, vocab_size]. The logit used to sample the bonus token.

        How it works:
        - Get input ids that were not previously accepted.
        - Get the corresponding target logits to these input ids (target_logit[j-1] corresponds to input_ids[j])
        - Sample a token from the logits for each batch and compare to input_ids to get the newly accepted tokens.
        - The output_ids consist of the previously accepted tokens, the newly accepted tokens,
        and a newly sampled token after the last accepted token.
        """

        batch_size, seq_len = input_ids.shape

        # First, check that num_previously_accepted is <= seq_len for each batch
        # Additionally, num_previously_accepted should be >= 1 for each batch,
        # which corresponds to having some context tokens (context tokens are always accepted).
        assert (num_previously_accepted >= 1).all(), (
            "num_previously_accepted must be >= 1. Please provide non-empty context in each batch."
        )
        assert (num_previously_accepted <= seq_len).all(), (
            "num_previously_accepted must be <= seq_len for each batch"
        )

        # We get input tokens that were not yet accepted.
        unchecked_input_ids = [
            input_ids[i, num_previously_accepted[i] : seq_len].unsqueeze(0)
            for i in range(batch_size)
        ]

        # We get the corresponding target logits for the unchecked input tokens.
        # logit j-1 corresponds to input j.
        # Note that because of our check that num_previously_accepted is >= 1
        # We also get the last output token for each batch, because we may need to append it
        # at the end.
        unchecked_target_logits = [
            target_logits[i, (num_previously_accepted[i] - 1) : seq_len, :].unsqueeze(0)
            for i in range(batch_size)
        ]

        unchecked_output_ids = [self.sample_greedy(x) for x in unchecked_target_logits]

        # corresponding_output_ids: [batch_size, seq_len - 1]. The output ids that correspond to the unchecked input ids
        # Omits the last index because that corresponds to a freshly sampled output model.
        corresponding_output_ids = [output_id[:, :-1] for output_id in unchecked_output_ids]

        # After sample_greedy, corresponding_output_ids should have same shape as unchecked_input_ids
        assert [x.shape for x in unchecked_input_ids] == [
            x.shape for x in corresponding_output_ids
        ], "unchecked_input_ids and corresponding_output_ids must have the same shape"

        matches = [
            (corresponding_output_ids[i] == unchecked_input_ids[i]).int() for i in range(batch_size)
        ]

        # Compute num_newly_accepted_tokens per batch (handles different sizes across batches)
        num_newly_accepted_tokens = []
        for i in range(batch_size):
            if matches[i].numel() == 0:
                # No unchecked tokens for this batch (num_previously_accepted == seq_len)
                num_newly_accepted_tokens.append(
                    torch.tensor(0, dtype=torch.long, device=input_ids.device)
                )
            else:
                # prefix_matches[j] is 1 if first j+1 tokens all matched
                prefix_matches = matches[i].cumprod(dim=-1)
                num_newly_accepted_tokens.append(prefix_matches.sum().long())
        num_newly_accepted_tokens = torch.stack(num_newly_accepted_tokens)

        # num_accepted_tokens: [batch_size]. The total number of accepted tokens in each batch,
        # including previously accepted tokens.
        num_accepted_tokens = num_previously_accepted + num_newly_accepted_tokens

        assert (num_accepted_tokens <= seq_len).all(), (
            "num_accepted_tokens must be <= seq_len for each batch"
        )

        # Construct draft_input_ids for the draft model
        # For each sequence:
        # 1. Take previously accepted tokens (skipping the first one)
        # 2. Append newly accepted tokens directly from input_ids.
        # 3. Append the sampled token for last accepted position: unchecked_output_ids[0][num_newly_accepted]
        # 4. Fill the rest with zeros (padding)
        # Total real tokens: (num_previously_accepted - 1) + num_newly_accepted + 1 = num_accepted_tokens

        draft_input_ids = torch.zeros(
            (batch_size, seq_len), dtype=input_ids.dtype, device=input_ids.device
        )

        for i in range(batch_size):
            # 1. Previously accepted tokens (skip the first one in keeping with Eagle convention)
            # Note that this potentially includes context tokens, but is structured this way because we
            # want the output to contain the entire prefix of accepted tokens because the drafters have no KV cache.
            prev_accepted = input_ids[i, 1 : num_previously_accepted[i]]

            # 2. Newly accepted input tokens
            newly_accepted = input_ids[
                i,
                num_previously_accepted[i] : num_previously_accepted[i]
                + num_newly_accepted_tokens[i],
            ]

            # 3. The sampled output token for the last accepted position
            # unchecked_output_ids[i][j] is the sampled token for position (num_previously_accepted + j)
            # We want the token for position num_accepted_tokens, which is index num_newly_accepted_tokens
            next_token = unchecked_output_ids[i][0][num_newly_accepted_tokens[i]].unsqueeze(0)

            # Concatenate all parts
            draft_prefix = torch.cat([prev_accepted, newly_accepted, next_token])

            # Sanity check: draft_prefix length should equal num_accepted_tokens
            assert draft_prefix.shape[0] == num_accepted_tokens[i], (
                f"draft_prefix length {draft_prefix.shape[0]} != num_accepted_tokens {num_accepted_tokens[i]}"
            )

            # Fill into draft_input_ids (rest remains zeros as padding)
            draft_input_ids[i, : num_accepted_tokens[i]] = draft_prefix

        # Construct last_logits_3d: [batch_size, 1, vocab_size]
        # This is the logit used to sample the bonus token for each sequence.
        # The bonus token is sampled from unchecked_target_logits[i][0][num_newly_accepted_tokens[i]]
        last_logits_list = []
        for i in range(batch_size):
            # unchecked_target_logits[i] has shape [1, num_unchecked + 1, vocab_size]
            # Index num_newly_accepted_tokens[i] gives the logit for the bonus token
            bonus_logit = unchecked_target_logits[i][0, num_newly_accepted_tokens[i], :].unsqueeze(
                0
            )
            last_logits_list.append(bonus_logit)
        last_logits_3d = torch.stack(last_logits_list, dim=0)  # [batch_size, 1, vocab_size]

        return draft_input_ids, num_newly_accepted_tokens, num_accepted_tokens, last_logits_3d

    def forward(self, input_ids, position_ids, **kwargs):
        """Dispatch to appropriate forward implementation based on kwargs.

        If num_previously_accepted is provided, use the prefill-only (no KV cache) implementation.
        Otherwise, this is the KV cache case which is not yet implemented.
        """
        num_previously_accepted = kwargs.get("num_previously_accepted", None)

        if num_previously_accepted is not None:
            return self._forward_prefill_only(input_ids, position_ids, **kwargs)
        else:
            # KV cached case - not implemented yet
            raise NotImplementedError(
                "EagleWrapper forward with KV cache is not implemented. "
                "This code path is reached when num_previously_accepted is not provided in kwargs."
            )

    def _forward_prefill_only(self, input_ids, position_ids, **kwargs):
        """Forward pass without KV cache (prefill-only mode).

        This is the original implementation that recomputes all attention
        from scratch on every forward call.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        num_previously_accepted = kwargs.get("num_previously_accepted", None)
        if num_previously_accepted is None:
            raise ValueError("num_previously_accepted must be provided for prefill-only mode.")

        # Compute embeddings using the target embedding layer
        input_embeds = self.target_model.get_input_embeddings()(input_ids)

        # target_logits: [batch_size, seq_len, vocab_size]
        # Pass embeddings to target model instead of input_ids
        target_logits = self.target_model(
            inputs_embeds=input_embeds, position_ids=position_ids
        ).logits

        # output_ids: [batch_size, seq_len]. Contains a prefix of accepted tokens from the target model,
        # a generated token from the target model, and some padding to fill out the tensor.
        # num_accepted_tokens: [batch_size]. The number of accepted tokens in each batch.
        # num_newly_accepted_tokens: [batch_size]. The number of newly accepted tokens in each batch.

        output_ids, num_newly_accepted_tokens, num_accepted_tokens, _ = self.sample_and_verify(
            input_ids, target_logits, num_previously_accepted
        )

        # Get hidden states from the resource manager
        # resource_manager.hidden_states is [max_tokens, hidden_size * num_capture_layers] (flattened)
        # We slice to get [batch_size * seq_len, hidden_size * num_capture_layers]
        hidden_states = self.resource_manager.hidden_states[: (batch_size * seq_len), :]

        # Apply eagle3 fc to reduce hidden size.
        # Note: Since we are in prefill-only mode, this is extremely wasteful - we will apply the eagle3 fc layer
        # to hidden states that we have applied it to previously. But, this is generally the case in prefill-only mode.
        # Input: [batch_size * seq_len, hidden_size * num_capture_layers]
        # Output: [batch_size * seq_len, hidden_size]
        hidden_states = self.apply_eagle3_fc(hidden_states)

        # Reshape from [batch_size * seq_len, hidden_size] to [batch_size, seq_len, hidden_size]
        hidden_size = hidden_states.shape[-1]
        hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)

        # Create a working buffer for the drafting loop in [batch, seq + draft_len, hidden] format.
        # This is separate from resource_manager.hidden_states which remains in flattened format.
        all_hidden_states = torch.zeros(
            (batch_size, seq_len + self.max_draft_len, hidden_size),
            device=device,
            dtype=hidden_states.dtype,
        )
        # Copy the initial hidden states from target model
        all_hidden_states[:, :seq_len, :] = hidden_states

        # Construct our inputs for the drafting loop.
        # We want tensors that will be able to hold all the tokens we draft.

        dummy_input_ids = torch.zeros(
            (batch_size, self.max_draft_len), device=device, dtype=output_ids.dtype
        )

        # draft_input_ids: [batch_size, seq_len + self.max_draft_len]
        draft_input_ids = torch.cat((output_ids, dummy_input_ids), dim=1)

        draft_position_ids = 1 + torch.arange(
            self.max_draft_len, device=device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)

        draft_position_ids = draft_position_ids + position_ids[:, -1:].expand(
            -1, self.max_draft_len
        )

        # draft_position_ids: [batch_size, seq_len + self.max_draft_len]
        # These position ids will work throughout the drafting loop.
        draft_position_ids = torch.cat((position_ids, draft_position_ids), dim=1)

        # The number of tokens currently in the draft input ids. Possibly includes padding.
        curr_num_tokens = seq_len

        # [batch_size]
        # The number of valid tokens currently in the draft input ids (does not include padding).
        curr_valid_tokens = num_accepted_tokens.clone()

        batch_indices = torch.arange(batch_size, device=device)

        for _ in range(self.max_draft_len):
            # Get the input ids, position ids, and hidden states for the current tokens.
            # size of tensor is constant for the current iteration and constant across dimensions (curr_num_tokens)
            # These tensors may correspond to padding tokens, but due to the causality of the draft model,
            # we can extract the draft tokens and hidden states corresponding to the valid tokens.

            input_ids = draft_input_ids[:, :curr_num_tokens]
            position_ids = draft_position_ids[:, :curr_num_tokens]
            hidden_states = all_hidden_states[:, :curr_num_tokens, :]

            inputs_embeds = self.apply_draft_embedding(input_ids)
            draft_output = self.draft_model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                hidden_states=hidden_states,
            )

            draft_output_logits = self.apply_lm_head(draft_output.norm_hidden_state)

            # get the output logits for the latest valid token in each batch
            # It is at curr_valid_tokens-1 due to 0-indexing.
            latest_draft_logits = draft_output_logits[batch_indices, curr_valid_tokens - 1, :]

            # draft_output_tokens: [batch_size, 1]
            draft_output_tokens = self.sample_greedy(latest_draft_logits)

            # if the lm_head outputs tokens from the draft vocab, we need to convert them to tokens
            # from the target vocab before the next iteration.
            draft_output_tokens = self.apply_d2t(draft_output_tokens)

            # insert the draft output tokens into the draft input ids.
            draft_input_ids[batch_indices, curr_valid_tokens] = draft_output_tokens

            # Similarly, we want the hidden state for the latest drafted token in each batch.
            # This is a draft hidden state for the token that was just created from the latest valid token.

            # [batch_size, seq_len + self.max_draft_len, hidden_size]
            all_hidden_states[batch_indices, curr_valid_tokens, :] = draft_output.last_hidden_state[
                batch_indices, curr_valid_tokens - 1, :
            ]

            curr_valid_tokens = curr_valid_tokens + 1
            curr_num_tokens = curr_num_tokens + 1

        # Return the full draft_input_ids tensor for each batch element.
        # The valid prefix within each tensor has length:
        # num_previously_accepted[i] + num_newly_accepted_tokens[i] + max_draft_len
        # Callers should use this to slice out the valid tokens if needed.
        new_tokens = [draft_input_ids[i] for i in range(batch_size)]

        return EagleWrapperOutput(
            new_tokens=new_tokens,
            new_tokens_lens=num_newly_accepted_tokens,
        )
