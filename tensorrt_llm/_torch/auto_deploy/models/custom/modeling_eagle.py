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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from tensorrt_llm._torch.speculative.eagle3 import Eagle3ResourceManager
from tensorrt_llm.llmapi.llm_args import EagleDecodingConfig

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
            return embeds.to(self.draft_model.model.dtype)
        else:
            return self.draft_model.get_input_embeddings()(input_ids)

    def apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply lm_head to get logits from hidden states."""
        if self.load_lm_head_from_target:
            lm_head_weights = self.target_model.get_output_embeddings()(hidden_states)
            return lm_head_weights.to(self.draft_model.model.dtype)
        else:
            return self.draft_model.get_output_embeddings()(hidden_states)

    def _filter_kwargs_for_submodule(self, kwargs: dict, submodule: nn.Module) -> dict:
        """Filter kwargs to only include those accepted by submodule's forward."""
        # Get input names from GraphModule's placeholder nodes
        expected_names = {node.name for node in submodule.graph.nodes if node.op == "placeholder"}
        return {k: v for k, v in kwargs.items() if k in expected_names}

    def _recompute_metadata_from_position_ids(
        self,
        kwargs: dict,
        packed_position_ids: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> dict:
        """Recompute all position-related metadata from packed position_ids and seq_lens.

        This is the shared helper used by _prepare_next_draft_metadata to derive
        all metadata from position_ids.

        Args:
            kwargs: Current kwargs dict (already filtered for submodule). Used to get
                page_size from k_cache and to copy non-metadata keys.
            packed_position_ids: Position IDs in packed format [1, total_tokens].
            seq_lens: Number of tokens per sequence [num_seq]. Can be a tensor or list.

        Returns:
            New kwargs dict with all metadata recomputed from position_ids.
        """
        device = packed_position_ids.device

        # Convert seq_lens to tensor if needed
        if not isinstance(seq_lens, torch.Tensor):
            seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        else:
            seq_lens_tensor = seq_lens.int().to(device)

        num_seq = len(seq_lens_tensor)

        # Get page_size from kv_cache shape
        # HND layout (default): [..., num_kv_heads, page_size, head_dim]
        # page_size is always at index -2 (second to last)
        page_size = None
        for key, val in kwargs.items():
            if key.startswith("kv_cache") and isinstance(val, torch.Tensor):
                page_size = val.shape[-2]
                break

        if page_size is None:
            raise ValueError(
                f"Could not determine page_size from kv_cache in kwargs. Keys: {list(kwargs.keys())}"
            )

        # Compute cu_seqlen: cumulative sequence lengths [0, seq_len[0], seq_len[0]+seq_len[1], ...]
        cu_seqlen = torch.zeros(num_seq + 1, dtype=torch.int32, device=device)
        cu_seqlen[1:] = torch.cumsum(seq_lens_tensor, dim=0)
        cu_seqlen_host = cu_seqlen.cpu()

        # Compute batch_info: [num_prefill, num_prefill_tokens, num_decode]
        # seq_len > 1 is "prefill" (multi-token), seq_len == 1 is "decode"
        num_prefill = (seq_lens_tensor > 1).sum().item()
        num_prefill_tokens = seq_lens_tensor[seq_lens_tensor > 1].sum().item()
        num_decode = (seq_lens_tensor <= 1).sum().item()
        batch_info_host = torch.tensor(
            [num_prefill, num_prefill_tokens, num_decode], dtype=torch.int32, device="cpu"
        )

        # Compute seq_len_with_cache for each sequence
        # seq_len_with_cache = last_position + 1 = position_ids[last_token_of_seq] + 1
        # Last token of sequence i is at packed index cu_seqlen[i+1] - 1
        last_token_indices = cu_seqlen[1:] - 1  # [num_seq]
        last_positions = packed_position_ids[0, last_token_indices]  # [num_seq]
        seq_len_with_cache = (last_positions + 1).int()
        seq_len_with_cache_host = seq_len_with_cache.cpu()

        # Compute last_page_len: (seq_len_with_cache - 1) % page_size + 1
        last_page_len = ((seq_len_with_cache - 1) % page_size + 1).int()
        last_page_len_host = last_page_len.cpu()

        # Compute pages_per_seq: ceil(seq_len_with_cache / page_size) for each sequence
        # Using integer math: (x + page_size - 1) // page_size = ceil(x / page_size)
        pages_per_seq = ((seq_len_with_cache + page_size - 1) // page_size).int()
        pages_per_seq_host = pages_per_seq.cpu()

        # Compute cu_num_pages: cumulative sum [0, p0, p0+p1, p0+p1+p2, ...]
        cu_num_pages = torch.zeros(num_seq + 1, dtype=torch.int32, device=device)
        cu_num_pages[1:] = torch.cumsum(pages_per_seq, dim=0)
        cu_num_pages_host = cu_num_pages.cpu()

        # Build new kwargs with recomputed metadata
        metadata_updates = {
            "batch_info_host": batch_info_host,
            "cu_seqlen_host": cu_seqlen_host,
            "cu_seqlen": cu_seqlen,
            "seq_len_with_cache": seq_len_with_cache,
            "seq_len_with_cache_host": seq_len_with_cache_host,
            "last_page_len": last_page_len,
            "last_page_len_host": last_page_len_host,
            "pages_per_seq": pages_per_seq,
            "pages_per_seq_host": pages_per_seq_host,
            "cu_num_pages": cu_num_pages,
            "cu_num_pages_host": cu_num_pages_host,
        }

        new_kwargs = {}
        for key, val in kwargs.items():
            if key in metadata_updates:
                new_kwargs[key] = metadata_updates[key]
            else:
                # Pass through unchanged (caches, cache_loc, etc.)
                new_kwargs[key] = val

        return new_kwargs

    def _prepare_next_draft_metadata(
        self,
        kwargs: dict,
        curr_position_ids: torch.Tensor,
        curr_cu_seq_len: torch.Tensor,
        num_sequences: int,
        num_accepted: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Prepare position_ids and metadata for the next draft iteration.

        Extracts positions from each sequence at the last accepted token and increments
        by 1, then recomputes all metadata including cu_num_pages for page boundary handling.

        Args:
            kwargs: Current kwargs dict
            curr_position_ids: [1, total_tokens] - current position IDs
            curr_cu_seq_len: [num_seq + 1] - cumulative sequence lengths (may be pre-allocated)
            num_sequences: Actual number of sequences to process
            num_accepted: [num_seq] - number of accepted tokens per sequence.
                Extracts at cu_seqlen[i] + num_accepted[i] - 1 for each sequence.

        Returns:
            Tuple of:
            - new_position_ids: [1, num_seq] - next position for each sequence
            - new_kwargs: dict with recomputed metadata
        """
        # Compute extraction indices: cu_seqlen[i] + num_accepted[i] - 1
        extraction_indices = (curr_cu_seq_len[:num_sequences] + num_accepted - 1).long()

        # Get positions at extraction indices and increment by 1
        extracted_positions = curr_position_ids[0, extraction_indices]
        new_position_ids = (extracted_positions + 1).unsqueeze(0)  # [1, num_sequences]

        # Each draft iteration is a standard decode step (1 token per sequence)
        seq_lens = torch.ones(
            new_position_ids.shape[1], dtype=torch.int32, device=curr_position_ids.device
        )

        # Recompute all metadata from the new packed position_ids
        new_kwargs = self._recompute_metadata_from_position_ids(kwargs, new_position_ids, seq_lens)

        return new_position_ids, new_kwargs

    def _extract_draft_iteration_outputs(
        self,
        draft_idx: int,
        draft_output: Any,
        draft_output_logits: torch.Tensor,
        draft_kwargs: dict,
        num_sequences: int,
        num_accepted: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract logits and hidden state from draft model output.

        This helper encapsulates the iteration-dependent indexing logic:
        - iter 0 (prefill-like): Extract at num_accepted-1 indices (leveraging causal attention)
        - iter 1+ (decode-like): Each sequence has exactly 1 token, simple indexing

        Args:
            draft_idx: Current draft iteration index (0 = first/prefill-like, 1+ = decode-like)
            draft_output: Output from draft model forward
            draft_output_logits: Logits from lm_head, packed [1, total_tokens, vocab_size]
            draft_kwargs: kwargs used for this iteration (contains cu_seqlen)
            num_sequences: Actual number of sequences to process
            num_accepted: [num_seq] - number of accepted tokens per sequence (required for iter 0)

        Returns:
            Tuple of:
            - latest_draft_logits: [num_seq, vocab_size] - logits at extraction position
            - last_draft_hidden_state: [num_seq, hidden_size] - hidden state at extraction position
        """
        if draft_idx == 0:
            # First call: extract at num_accepted-1 indices (leveraging causal attention)
            # With untruncated input, we ran full sequences through the draft model
            # The output at position (num_accepted - 1) is equivalent to running truncated input
            # cu_seqlen[i] gives the start of sequence i in the packed tensor
            # We extract at cu_seqlen[i] + num_accepted[i] - 1
            if num_accepted is None:
                raise ValueError("num_accepted must be provided for draft_idx == 0")

            cu_seqlen = draft_kwargs["cu_seqlen"]
            # Compute extraction indices: cu_seqlen[i] + num_accepted[i] - 1 for each sequence
            extraction_indices = cu_seqlen[:num_sequences] + num_accepted - 1  # [num_sequences]
            extraction_indices = extraction_indices.long()

            latest_draft_logits = draft_output_logits[0, extraction_indices, :]  # [num_seq, vocab]
            last_draft_hidden_state = draft_output.last_hidden_state[
                0, extraction_indices, :
            ]  # [num_seq, hidden]
        else:
            # Subsequent calls: decode_cu_seqlen = [0, 1, 2, ..., num_sequences]
            # Last token of seq i is at position i (since each seq has 1 token)
            # Simplifies to taking all tokens: output[0, :, :] = [num_seq, ...]
            latest_draft_logits = draft_output_logits[0, :, :]  # [num_seq, vocab]
            last_draft_hidden_state = draft_output.last_hidden_state[0, :, :]  # [num_seq, hidden]

        return latest_draft_logits, last_draft_hidden_state

    def sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        ret = torch.argmax(logits, dim=-1)
        return ret

    def prepare_draft_context(
        self,
        packed_input_ids: torch.Tensor,
        packed_logits: torch.Tensor,
        cu_seqlen: torch.Tensor,
        num_context: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Sample new tokens for prefill sequences (no verification needed).

        For prefill, all input tokens are the prompt (already "accepted").
        We just sample a new token from the last logit position.

        Args:
            packed_input_ids: [1, total_tokens] - packed input token ids (full batch)
            packed_logits: [1, total_tokens, vocab_size] - packed logits from target model (full batch)
            cu_seqlen: [num_sequences + 1] - cumulative sequence lengths (full batch)
            num_context: number of context (prefill) sequences

        Returns:
            packed_output_ids: [1, ctx_tokens] - shifted input + sampled token per context sequence,
                or None if num_context == 0
            num_accepted: [num_context] - always seq_len (all prompt tokens accepted),
                or None if num_context == 0
        """
        if num_context == 0:
            return None, None

        # Slice to get only context sequences
        ctx_cu_seqlen = cu_seqlen[: num_context + 1]
        ctx_end_token = ctx_cu_seqlen[-1]
        ctx_input_ids = packed_input_ids[:, :ctx_end_token]  # [1, ctx_tokens]
        ctx_logits = packed_logits[:, :ctx_end_token, :]  # [1, ctx_tokens, vocab_size]

        seq_lens = ctx_cu_seqlen[1:] - ctx_cu_seqlen[:-1]

        # Squeeze batch dimension for processing
        flat_input_ids = ctx_input_ids.squeeze(0)  # [ctx_tokens]
        flat_logits = ctx_logits.squeeze(0)  # [ctx_tokens, vocab_size]

        # Sample from last logit position of each sequence
        last_positions = ctx_cu_seqlen[1:] - 1  # [num_context]
        last_logits = flat_logits[last_positions]
        sampled_tokens = self.sample_greedy(last_logits)  # [num_context]

        # Construct output: for each sequence, output = input[1:] + sampled_token
        # Vectorized: roll shifts all tokens left by 1, then scatter sampled tokens at sequence ends
        flat_output_ids = flat_input_ids.roll(-1)  # output[i] = input[i+1] (with wraparound)
        flat_output_ids[last_positions] = sampled_tokens.to(flat_output_ids.dtype)

        packed_output_ids = flat_output_ids.unsqueeze(0)  # [1, ctx_tokens]
        num_accepted = seq_lens.long()

        return packed_output_ids, num_accepted

    def _count_context_sequences(
        self,
        position_ids: torch.Tensor,
        seq_starts: torch.Tensor,
        num_sequences: int,
    ) -> int:
        """Count context (prefill) sequences and validate batch ordering.

        Context sequences have first_position_id == 0 (start of a new sequence).
        Spec_dec sequences have first_position_id > 0 (continuation).

        Asserts that all context sequences come before all spec_dec sequences
        (i.e., no interleaving).

        TODO: This implementation inspects position_ids tensor values to classify sequences.
        This should be refactored to receive num_context directly via batch_info_host / batch_info
        from the runtime, avoiding tensor value inspection entirely.

        Args:
            position_ids: [1, total_tokens] packed position IDs
            seq_starts: [num_sequences] tensor of start positions for each sequence in packed tensor
            num_sequences: Total number of sequences in batch

        Returns:
            num_context: Count of context sequences (first num_context are context,
                         remaining num_sequences - num_context are spec_dec)
        """
        if num_sequences == 0:
            return 0

        # Gather first position_id for each sequence (seq_starts is already on device)
        first_positions = position_ids[0, seq_starts.long()]  # [num_sequences]
        first_positions_cpu = first_positions.tolist()

        # Count context sequences (first_position == 0) and validate ordering
        # Context sequences must come first, then spec_dec sequences
        num_context = 0
        seen_spec_dec = False
        for i in range(num_sequences):
            is_context = first_positions_cpu[i] == 0
            if is_context:
                assert not seen_spec_dec, (
                    f"Batch ordering violation: context sequence at index {i} found after "
                    f"spec_dec sequence. Context sequences must come before spec_dec sequences."
                )
                num_context += 1
            else:
                seen_spec_dec = True

        return num_context

    def _store_target_hidden_states(
        self,
        kwargs: dict,
        cu_seqlen: torch.Tensor,
        num_sequences: int,
    ) -> int:
        """Store hidden states from target model forward pass.

        Collects hidden_states_cache_* buffers from kwargs and stores
        them in resource_manager for use by the draft model.

        Args:
            kwargs: Forward kwargs containing hidden_states_cache_* buffers
            cu_seqlen: [num_sequences + 1] cumulative sequence lengths tensor (on device)
            num_sequences: Total number of sequences in batch

        Returns:
            num_tokens: Total token count stored
        """
        # Collect hidden_states_cache buffers from kwargs (e.g., hidden_states_cache_0, _1, _2)
        # Sort by name to ensure correct order: hidden_states_cache_0, _1, _2, etc.
        hidden_state_items = sorted(
            [
                (name, tensor)
                for name, tensor in kwargs.items()
                if name.startswith("hidden_states_cache")
            ],
            key=lambda x: x[0],
        )

        hidden_state_buffers = [tensor for _, tensor in hidden_state_items]

        # Compute num_tokens from cu_seqlen
        # cu_seqlen[num_sequences] gives total tokens across all sequences
        num_tokens = int(cu_seqlen[num_sequences].item())
        self.resource_manager.store_hidden_states(hidden_state_buffers, num_tokens)

        return num_tokens

    def sample_and_verify_prefill_only(
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

        # During export/tracing with meta tensors, we cannot perform actual verification
        # because operations like .item(), .all(), and data-dependent indexing fail on meta tensors.
        # Return simulated values with correct shapes instead.
        if input_ids.device.type == "meta":
            # Simulate accepting all tokens (optimistic case for shape tracing)
            draft_input_ids = input_ids.clone()
            num_newly_accepted_tokens = (
                torch.full((batch_size,), seq_len, dtype=torch.long, device=input_ids.device)
                - num_previously_accepted
            )
            num_accepted_tokens = torch.full(
                (batch_size,), seq_len, dtype=torch.long, device=input_ids.device
            )
            last_logits_3d = target_logits[:, -1:, :]  # [batch_size, 1, vocab_size]
            return draft_input_ids, num_newly_accepted_tokens, num_accepted_tokens, last_logits_3d

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

    def sample_and_verify_decode(
        self,
        packed_input_ids: torch.Tensor,
        packed_target_logits: torch.Tensor,
        cu_seqlen: torch.Tensor,
        num_context: int,
        num_spec_dec: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Verify draft tokens for decode (spec_dec) sequences.

        Uses packed representation with batch dimension 1 for both input and output.
        Assumes all sequences have uniform length (1 + max_draft_len).

        Args:
            packed_input_ids: [1, total_tokens] - packed input token ids (full batch)
            packed_target_logits: [1, total_tokens, vocab_size] - packed logits from target model (full batch)
            cu_seqlen: [num_sequences + 1] - cumulative sequence lengths (full batch)
            num_context: number of context (prefill) sequences
            num_spec_dec: number of spec_dec (verification) sequences

        Returns:
            packed_output_ids: [1, sd_tokens] - greedy samples at ALL spec_dec positions,
                or None if num_spec_dec == 0
            num_accepted: [num_spec_dec] - total accepted tokens (verified + 1 bonus),
                or None if num_spec_dec == 0
        """
        if num_spec_dec == 0:
            return None, None

        # Slice to get only spec_dec sequences
        sd_cu_seqlen_raw = cu_seqlen[num_context : num_context + num_spec_dec + 1]
        sd_start_token = sd_cu_seqlen_raw[0]
        sd_end_token = sd_cu_seqlen_raw[-1]
        sd_input_ids = packed_input_ids[:, sd_start_token:sd_end_token]  # [1, sd_tokens]
        sd_logits = packed_target_logits[
            :, sd_start_token:sd_end_token, :
        ]  # [1, sd_tokens, vocab_size]

        total_tokens = sd_input_ids.shape[1]
        seq_len = total_tokens // num_spec_dec  # All spec_dec sequences have uniform length

        # Squeeze batch dimension and greedy sample all positions
        flat_input_ids = sd_input_ids.squeeze(0)  # [sd_tokens]
        flat_logits = sd_logits.squeeze(0)  # [sd_tokens, vocab_size]
        sampled_all = self.sample_greedy(flat_logits)  # [sd_tokens]

        # Reshape to [num_spec_dec, seq_len] for per-sequence verification
        flat_input_2d = flat_input_ids.view(num_spec_dec, seq_len)
        sampled_2d = sampled_all.view(num_spec_dec, seq_len)

        # Verify: compare draft tokens (input[:, 1:]) with what target would have sampled (sampled[:, :-1])
        draft_tokens = flat_input_2d[:, 1:]  # [num_spec_dec, max_draft_len]
        sampled_for_verification = sampled_2d[:, :-1]  # [num_spec_dec, max_draft_len]
        matches = (sampled_for_verification == draft_tokens).int()

        # Count consecutive matches from start using cumulative product
        prefix_matches = matches.cumprod(dim=1)
        num_accepted = prefix_matches.sum(dim=1) + 1  # +1 for bonus token

        packed_output_ids = sampled_all.unsqueeze(0)  # [1, sd_tokens]
        return packed_output_ids, num_accepted

    def forward(self, input_ids, position_ids, **kwargs):
        """Dispatch to appropriate forward implementation based on kwargs.

        If num_previously_accepted is provided, use the prefill-only (no KV cache) implementation.
        Otherwise, use the KV cache implementation (for generation with cached attention).
        """
        num_previously_accepted = kwargs.get("num_previously_accepted", None)

        if num_previously_accepted is not None:
            return self._forward_prefill_only(input_ids, position_ids, **kwargs)
        else:
            return self._forward_with_kv_cache(input_ids, position_ids, **kwargs)

    def _forward_with_kv_cache(self, input_ids, position_ids, **kwargs):
        """Forward pass with KV cache support (for generation with cached attention).

        TODO: We use input positions to distinguish prefill from spec-dec sequences (both are multi-token requests).
        We should use a way that does not require us to inspect input positions.
        """
        # Extract batch_info from kwargs: [num_multi_token, num_multi_token_total, num_single_token]
        batch_info_host = kwargs.get("batch_info_host", None)
        if batch_info_host is None:
            raise ValueError("batch_info_host must be provided in kwargs. ")

        num_multi_token, _num_multi_token_total, num_single_token = batch_info_host.tolist()

        # With spec dec enabled, all requests are multi-token (context or spec dec verification)
        # Single-token decode doesn't happen in one-model spec dec flow
        assert num_single_token == 0, (
            f"Single-token requests not expected in spec dec flow. Got num_single_token={num_single_token}"
        )

        # Total number of sequences = num_multi_token (since num_single_token == 0)
        num_sequences = num_multi_token

        cu_seqlen = kwargs.get("cu_seqlen", None)
        if cu_seqlen is None:
            raise ValueError("cu_seqlen must be provided in kwargs for packed sequence format.")

        seq_starts = cu_seqlen[:num_sequences]  # [num_sequences]

        # Count context vs spec_dec sequences
        # With ordered batches, context sequences are [:num_context], spec_dec are [num_context:]
        num_context = self._count_context_sequences(position_ids, seq_starts, num_sequences)
        num_spec_dec = num_sequences - num_context

        inputs_embeds = self.target_model.get_input_embeddings()(input_ids)
        target_kwargs = self._filter_kwargs_for_submodule(kwargs, self.target_model)

        # target_logits: [batch_size, seq_len, vocab_size]
        target_logits = self.target_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, **target_kwargs
        ).logits

        # Capture and store hidden states from target model forward pass
        num_tokens = self._store_target_hidden_states(kwargs, cu_seqlen, num_sequences)
        hidden_states = self.resource_manager.hidden_states[:num_tokens, :]
        hidden_states = self.apply_eagle3_fc(hidden_states)

        # Get device from input_ids
        device = input_ids.device

        num_accepted_tokens = torch.zeros(num_sequences, dtype=torch.long, device=device)

        # Process context and spec_dec sequences (methods return None if count is 0)
        packed_output_chunks = []

        ctx_output, ctx_accepted = self.prepare_draft_context(
            input_ids, target_logits, cu_seqlen, num_context
        )
        if ctx_output is not None:
            packed_output_chunks.append(ctx_output)
            num_accepted_tokens[:num_context] = ctx_accepted

        sd_output, sd_accepted = self.sample_and_verify_decode(
            input_ids, target_logits, cu_seqlen, num_context, num_spec_dec
        )
        if sd_output is not None:
            packed_output_chunks.append(sd_output)
            num_accepted_tokens[num_context:] = sd_accepted

        packed_output_ids = (
            torch.cat(packed_output_chunks, dim=1) if packed_output_chunks else input_ids
        )

        # Filter kwargs for draft_model (exclude target_model caches)
        draft_kwargs = self._filter_kwargs_for_submodule(kwargs, self.draft_model)

        # We pass EVERYTHING to the first draft:
        # - packed_output_ids: [1, total_tokens] - greedy samples at all positions
        # - position_ids: [1, total_tokens] - full input position_ids (no truncation)
        # - hidden_states: [total_tokens, hidden_size] - full hidden states (no truncation)
        # After first draft, we extract at num_accepted-1 indices (leveraging causal attention)

        draft_input_ids = packed_output_ids  # [1, total_tokens]
        draft_position_ids = position_ids  # [1, total_tokens]
        draft_hidden_states = hidden_states.unsqueeze(0)  # [1, total_tokens, hidden_size]

        draft_tokens: list[torch.Tensor] = []  # List of tensors, each [num_sequences]

        for draft_idx in range(self.max_draft_len):
            inputs_embeds = self.apply_draft_embedding(draft_input_ids)
            draft_output = self.draft_model(
                inputs_embeds=inputs_embeds,
                position_ids=draft_position_ids,
                hidden_states=draft_hidden_states,
                **draft_kwargs,
            )

            # draft_output.norm_hidden_state is packed 3D: [1, total_tokens, hidden_size]
            draft_output_logits = self.apply_lm_head(draft_output.norm_hidden_state)

            # Extract logits and hidden state at appropriate positions:
            # - iter 0: extract at num_accepted-1 indices (leveraging causal attention)
            # - iter 1+: extract at last position (each seq has 1 token)
            latest_draft_logits, last_draft_hidden_state = self._extract_draft_iteration_outputs(
                draft_idx,
                draft_output,
                draft_output_logits,
                draft_kwargs,
                num_sequences,
                num_accepted_tokens,
            )

            # Sample draft token from logits
            draft_token = self.sample_greedy(latest_draft_logits)
            draft_token = self.apply_d2t(draft_token)

            # Store the draft token tensor for this iteration
            # draft_token has shape [num_sequences]
            draft_tokens.append(draft_token)

            if draft_idx < self.max_draft_len - 1:
                # For iter 0: use actual num_accepted from verification
                # For iter 1+: we drafted 1 token per sequence, so num_accepted=1
                num_accepted = (
                    num_accepted_tokens if draft_idx == 0 else torch.ones_like(num_accepted_tokens)
                )
                draft_position_ids, draft_kwargs = self._prepare_next_draft_metadata(
                    draft_kwargs,
                    curr_position_ids=draft_position_ids,
                    curr_cu_seq_len=draft_kwargs["cu_seqlen"],
                    num_sequences=num_sequences,
                    num_accepted=num_accepted,
                )

                # Prepare tensor inputs for next iteration
                # last_draft_token: [num_sequences] -> [1, num_sequences]
                draft_input_ids = draft_token.unsqueeze(0)
                # last_draft_hidden_state: [num_sequences, hidden_size] -> [1, num_sequences, hidden_size]
                draft_hidden_states = last_draft_hidden_state.unsqueeze(0)

        # ============================================================
        # End of Drafting Loop
        # ============================================================

        flat_output_ids = packed_output_ids.squeeze(0)  # [total_tokens]

        new_tokens_2d = torch.zeros(
            (num_sequences, self.max_draft_len + 1),
            dtype=torch.int32,
            device=device,
        )
        new_tokens_lens = torch.zeros(num_sequences, dtype=torch.int32, device=device)

        # Convert draft_tokens list to tensor: [batch_size, max_draft_len]
        # draft_tokens is a list of max_draft_len tensors, each of shape [num_sequences]
        assert len(draft_tokens) == self.max_draft_len, (
            f"Expected {self.max_draft_len} draft tokens, got {len(draft_tokens)}"
        )
        assert all(t.shape == (num_sequences,) for t in draft_tokens), (
            f"All draft tokens should have shape [{num_sequences}]"
        )
        # Stack along dim=1 to get [num_sequences, max_draft_len]
        next_draft_tokens = torch.stack(draft_tokens, dim=1).to(dtype=torch.int32, device=device)

        # Prepare next_new_tokens: [batch_size, max_draft_len + 1]
        # Format: [last_accepted_token, draft_token_0, draft_token_1, ...]
        next_new_tokens = torch.zeros(
            (num_sequences, self.max_draft_len + 1),
            dtype=torch.int32,
            device=device,
        )

        # Process CONTEXT (prefill) requests - first num_context sequences (vectorized)
        # prepare_draft_context() returns output_ids = [input_ids[1:], sampled_token]
        # The sampled token is at the last position of each sequence
        # Note: all operations are no-ops when num_context == 0 (empty slices)
        ctx_last_positions = cu_seqlen[1 : num_context + 1] - 1  # [num_context]
        ctx_sampled_tokens = flat_output_ids[ctx_last_positions].to(torch.int32)

        new_tokens_2d[:num_context, 0] = ctx_sampled_tokens
        new_tokens_lens[:num_context] = 1
        next_new_tokens[:num_context, 0] = ctx_sampled_tokens
        next_new_tokens[:num_context, 1:] = next_draft_tokens[:num_context]

        # Process SPEC_DEC (verification) requests - sequences after context (vectorized)
        # packed_output_ids contains greedy samples at ALL positions (same length as input)
        # num_accepted_tokens: N + 1 (verified drafts + golden)
        # Note: all operations are no-ops when num_spec_dec == 0 (empty slices)

        # All spec_dec sequences have uniform length: 1 (accepted from prev round) + max_draft_len
        sd_seq_len = 1 + self.max_draft_len
        sd_start = cu_seqlen[num_context]
        sd_flat = flat_output_ids[sd_start : cu_seqlen[num_sequences]]

        # Copy to new_tokens_2d: reshape to [num_spec_dec, sd_seq_len] for block copy
        new_tokens_2d[num_context:, :sd_seq_len] = sd_flat.view(-1, sd_seq_len).to(torch.int32)

        # Set lengths from num_accepted_tokens
        new_tokens_lens[num_context:] = num_accepted_tokens[num_context:].int()

        # Get golden token (last accepted) for each sequence directly from flat format
        # Golden token for sequence i is at flat index: sd_start + i * sd_seq_len + (num_accepted[i] - 1)
        sd_num_accepted = num_accepted_tokens[num_context:]
        seq_offsets = torch.arange(num_spec_dec, device=device) * sd_seq_len
        golden_flat_indices = sd_start + seq_offsets + (sd_num_accepted - 1)
        golden_tokens = flat_output_ids[golden_flat_indices].to(torch.int32)

        next_new_tokens[num_context:, 0] = golden_tokens
        next_new_tokens[num_context:, 1:] = next_draft_tokens[num_context:]

        return EagleWrapperOutput(
            logits=target_logits,
            new_tokens=new_tokens_2d,
            new_tokens_lens=new_tokens_lens,  # Use corrected value, not num_newly_accepted_tokens
            next_draft_tokens=next_draft_tokens,
            next_new_tokens=next_new_tokens,
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

        output_ids, num_newly_accepted_tokens, num_accepted_tokens, _ = (
            self.sample_and_verify_prefill_only(input_ids, target_logits, num_previously_accepted)
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


class ADHiddenStateManager(Eagle3ResourceManager):
    """AutoDeploy-specific hidden state manager for Eagle3 speculative decoding.

    Stores hidden states for use by the draft model in EagleWrapper.forward().
    This class extends Eagle3ResourceManager with functionality tailored to
    how AutoDeploy captures hidden states.
    """

    def __init__(
        self,
        config: EagleDecodingConfig,
        dtype: torch.dtype,
        hidden_size: int,
        max_num_requests: int,
        max_seq_len: int,
        max_num_tokens: int,
    ):
        super().__init__(config, dtype, hidden_size, max_num_requests, max_seq_len, max_num_tokens)

        self.hidden_state_write_indices: torch.Tensor = torch.empty(
            max_num_tokens, dtype=torch.long, device="cuda"
        )

    @classmethod
    def build_from_target_engine(
        cls,
        engine,
        config: EagleDecodingConfig,
        max_num_requests: int,
    ) -> "ADHiddenStateManager":
        hidden_state_buffer = cls._get_hidden_state_buffers(engine.cache_seq_interface)[0]
        dtype = hidden_state_buffer.dtype
        hidden_size = hidden_state_buffer.shape[1]

        return cls(
            config=config,
            dtype=dtype,
            hidden_size=hidden_size,
            max_num_requests=max_num_requests,
            max_seq_len=engine.llm_args.max_seq_len,
            max_num_tokens=engine.llm_args.max_num_tokens,
        )

    @classmethod
    def build_from_target_factory(
        cls,
        target_factory,
        config: EagleDecodingConfig,
        max_num_requests: int,
        max_num_tokens: int,
    ) -> "ADHiddenStateManager":
        hidden_size = target_factory.hidden_size
        if hidden_size is None:
            raise ValueError(
                "Cannot determine hidden_size from target_factory. "
                "Ensure the factory implements the hidden_size property."
            )

        dtype = target_factory.dtype
        if dtype is None:
            raise ValueError("dtype must be available in target factory.")

        return cls(
            config=config,
            dtype=dtype,
            hidden_size=hidden_size,
            max_num_requests=max_num_requests,
            max_seq_len=target_factory.max_seq_len,
            max_num_tokens=max_num_tokens,
        )

    @staticmethod
    def _get_hidden_state_buffers(
        cache_seq_interface,
    ) -> List[torch.Tensor]:
        hidden_state_buffers = []
        for name, tensor in cache_seq_interface.named_args.items():
            if "hidden_states_cache" in name:
                hidden_state_buffers.append(tensor)

        if not hidden_state_buffers:
            raise ValueError(
                "No hidden_state_buffers found in cache_seq_interface. Check if we are actually running Eagle3."
            )
        return hidden_state_buffers

    def prepare_hidden_states_capture(self, ordered_requests, cache_seq_interface) -> None:
        """Prepare the hidden states for capture by establishing indices that the hidden states will be written to."""
        seq_lens = cache_seq_interface.info.seq_len
        num_tokens = sum(seq_lens)

        start_idx = 0
        hidden_states_write_indices = []
        for request, seq_len in zip(ordered_requests, seq_lens):
            request_id = request.request_id
            slot_id = self.slot_manager.get_slot(request_id)
            self.start_indices[slot_id] = start_idx
            hidden_states_write_indices.extend(range(start_idx, start_idx + seq_len))
            start_idx += max(seq_len, self.max_total_draft_tokens + 1)
            assert start_idx < self.hidden_states.shape[0], (
                f"start_idx {start_idx} exceeds hidden_states capacity {self.hidden_states.shape[0]}"
            )

        if len(hidden_states_write_indices) != num_tokens:
            raise ValueError(
                f"len(hidden_state_write_indices) ({len(hidden_states_write_indices)}) != num_tokens \
                ({num_tokens}). Check whether ordered_requests matches up with seq_lens."
            )

        hidden_state_write_indices_host = torch.tensor(
            hidden_states_write_indices, dtype=torch.long
        )

        self.hidden_state_write_indices[:num_tokens].copy_(
            hidden_state_write_indices_host, non_blocking=True
        )

    def store_hidden_states(
        self, hidden_state_buffers: List[torch.Tensor], num_tokens: int
    ) -> None:
        """Store hidden states from buffers into self.hidden_states.

        This method takes a list of hidden state tensors (one per captured layer)
        and copies them into the resource manager's hidden_states buffer using
        the write indices set up by prepare_hidden_states_capture().

        Args:
            hidden_state_buffers: List of tensors, each of shape [max_num_tokens, hidden_size].
                One tensor per captured layer.
            num_tokens: Number of tokens to copy from each buffer.
        """
        if not hidden_state_buffers:
            return

        hidden_states = [buffer[:num_tokens] for buffer in hidden_state_buffers]
        hidden_states = torch.cat(hidden_states, dim=1)
        hidden_states = hidden_states.to(dtype=self.dtype)

        # Use write indices to copy to the correct locations in self.hidden_states
        token_idx = self.hidden_state_write_indices[:num_tokens]
        self.hidden_states[:, : hidden_states.shape[1]].index_copy_(0, token_idx, hidden_states)

    def capture_hidden_states(self, cache_seq_interface) -> None:
        full_hidden_states = self._get_hidden_state_buffers(cache_seq_interface)
        if not full_hidden_states:
            return

        num_tokens = sum(cache_seq_interface.info.seq_len)
        self.store_hidden_states(full_hidden_states, num_tokens)
