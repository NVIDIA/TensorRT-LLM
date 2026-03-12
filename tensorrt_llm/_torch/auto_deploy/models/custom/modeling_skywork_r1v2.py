# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Slimmed down PyTorch Skywork-R1V2 model implementation for auto_deploy export.

Source:
https://huggingface.co/Skywork/Skywork-R1V2-38B

Skywork-R1V2-38B is a multimodal VLM (InternVL-based) with a Qwen2 LLM backbone.
For AutoDeploy, only the LLM (text) path is exported; the vision tower stays in eager.

This implementation differs from the original HuggingFace version in the following ways:
* Only the LLM backbone is instantiated (vision model is not needed for AD export)
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)

The HF checkpoint config class (SkyworkChatConfig, model_type="skywork_chat") is NOT
imported here — it uses trust_remote_code and is only available when the checkpoint is
present in the local HF cache.  AutoDeploy loads it via AutoConfig.from_pretrained at
runtime and dispatches to SkyworkR1V2ForConditionalGeneration via the "SkyworkChatConfig"
key registered below.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2Config
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

# ---------------------------------------------------------------------------
# Defaults from HuggingFace source (configuration_skywork_chat.py /
# configuration_skywork_vit.py in the checkpoint's remote code)
# ---------------------------------------------------------------------------
_HF_DEFAULT_SELECT_LAYER: int = -1  # use last ViT layer output
_HF_DEFAULT_DOWNSAMPLE_RATIO: float = 0.5  # pixel-shuffle spatial compression
_HF_DEFAULT_PS_VERSION: str = "v1"  # pixel-shuffle version
_HF_DEFAULT_NORM_TYPE: str = "rms_norm"  # vision encoder norm (SkyworkVisionConfig.norm_type)
_HF_DEFAULT_INITIALIZER_FACTOR: float = (
    0.1  # layer-scale init (SkyworkVisionConfig.initializer_factor)
)

# ---------------------------------------------------------------------------
# Vision tower components (eager PyTorch, never exported)
# ---------------------------------------------------------------------------


class VisionRMSNorm(nn.Module):
    """Plain-PyTorch RMSNorm for the vision encoder (runs in eager mode only)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VisionEmbeddings(nn.Module):
    """Patch + class + positional embeddings for the ViT encoder."""

    def __init__(self, vision_config):
        super().__init__()
        self.embed_dim = vision_config.hidden_size
        self.image_size = vision_config.image_size
        self.patch_size = vision_config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed: torch.Tensor, H: int, W: int) -> torch.Tensor:
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat(
            [
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            dim=1,
        )
        return embeddings + position_embedding.to(target_dtype)


class VisionAttention(nn.Module):
    """Multi-head self-attention for the ViT (naive SDPA, no flash attention)."""

    def __init__(self, vision_config):
        super().__init__()
        self.embed_dim = vision_config.hidden_size
        self.num_heads = vision_config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=vision_config.qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.qk_normalization = vision_config.qk_normalization
        if self.qk_normalization:
            self.q_norm = VisionRMSNorm(self.embed_dim, eps=vision_config.layer_norm_eps)
            self.k_norm = VisionRMSNorm(self.embed_dim, eps=vision_config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, C = hidden_states.shape
        qkv = (
            self.qkv(hidden_states)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # [B, H, N, D]

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class VisionMLP(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.fc1 = nn.Linear(vision_config.hidden_size, vision_config.intermediate_size)
        self.fc2 = nn.Linear(vision_config.intermediate_size, vision_config.hidden_size)
        self.act = ACT2FN[vision_config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.embed_dim = vision_config.hidden_size
        norm_type = getattr(vision_config, "norm_type", _HF_DEFAULT_NORM_TYPE)
        eps = vision_config.layer_norm_eps
        if norm_type == "rms_norm":
            self.norm1 = VisionRMSNorm(self.embed_dim, eps=eps)
            self.norm2 = VisionRMSNorm(self.embed_dim, eps=eps)
        else:
            self.norm1 = nn.LayerNorm(self.embed_dim, eps=eps)
            self.norm2 = nn.LayerNorm(self.embed_dim, eps=eps)

        self.attn = VisionAttention(vision_config)
        self.mlp = VisionMLP(vision_config)

        initializer_factor = getattr(
            vision_config, "initializer_factor", _HF_DEFAULT_INITIALIZER_FACTOR
        )
        self.ls1 = nn.Parameter(initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(initializer_factor * torch.ones(self.embed_dim))
        # DropPath is for training only; at inference it is always identity.
        self.drop_path1 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.drop_path1(
            self.attn(self.norm1(hidden_states).to(hidden_states.dtype)) * self.ls1
        )
        hidden_states = hidden_states + self.drop_path2(
            self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * self.ls2
        )
        return hidden_states


class VisionEncoder(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.layers = nn.ModuleList(
            [VisionEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)]
        )

    def forward(
        self, hidden_states: torch.Tensor, output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        all_hidden_states = [] if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states


class VisionModel(nn.Module):
    """Skywork ViT vision encoder — eager only, never torch.export-ed."""

    def __init__(self, vision_config):
        super().__init__()
        self.embeddings = VisionEmbeddings(vision_config)
        self.encoder = VisionEncoder(vision_config)

    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        hidden_states = self.embeddings(pixel_values)
        return self.encoder(hidden_states, output_hidden_states=output_hidden_states)


# ---------------------------------------------------------------------------
# Model components (Qwen2-based LLM backbone)
# ---------------------------------------------------------------------------


class SkyworkR1V2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class SkyworkR1V2RotaryEmbedding(nn.Module):
    """Standard RoPE with precomputed cos/sin cache."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._ad_cos_cached.to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached.to(dtype=x.dtype, device=x.device)
        # Return full cached tables; slicing by position_ids happens downstream in attention.
        return cos, sin


class SkyworkR1V2MLP(nn.Module):
    """SwiGLU MLP (identical to Qwen2)."""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SkyworkR1V2Attention(nn.Module):
    """Grouped Query Attention for Qwen2 backbone.

    Qwen2 uses bias on Q/K/V projections but no bias on O projection.
    No per-head Q/K normalization (unlike Qwen3).
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** (-0.5)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Slice full RoPE tables by position_ids here (D3 convention).
        cos, sin = position_embeddings
        cos = cos[position_ids]  # [B, S, head_dim]
        sin = sin[position_ids]  # [B, S, head_dim]

        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q,
            k,
            cos,
            sin,
            2,
        )

        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            self.scaling,
            None,  # sinks
            None,  # sliding_window
            None,  # logit_cap
            "bsnd",
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class SkyworkR1V2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SkyworkR1V2Attention(config, layer_idx=layer_idx)
        self.mlp = SkyworkR1V2MLP(config)
        self.input_layernorm = SkyworkR1V2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SkyworkR1V2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Dataclass outputs
# ---------------------------------------------------------------------------


@dataclass
class SkyworkR1V2Output(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class SkyworkR1V2CausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Transformer model (language_model.model.*)
# ---------------------------------------------------------------------------


class SkyworkR1V2TransformerModel(nn.Module):
    """Qwen2 transformer body (maps to language_model.model.*)."""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                SkyworkR1V2DecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = SkyworkR1V2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = SkyworkR1V2RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> SkyworkR1V2Output:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings, position_ids)

        hidden_states = self.norm(hidden_states)
        return SkyworkR1V2Output(last_hidden_state=hidden_states)


class SkyworkR1V2LanguageModel(nn.Module):
    """Wraps transformer + lm_head (maps to language_model.*)."""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.model = SkyworkR1V2TransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> SkyworkR1V2CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        return SkyworkR1V2CausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Top-level model (registered with AD)
# ---------------------------------------------------------------------------


class SkyworkR1V2ForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """Skywork-R1V2 model for AutoDeploy.

    Always constructed with a SkyworkChatConfig (loaded via trust_remote_code by AD),
    which includes a nested llm_config (Qwen2Config) for the LLM backbone and a
    vision_config for the vision tower.

    Weight hierarchy matches the HF checkpoint:
      vision_model.embeddings.*
      vision_model.encoder.layers.X.*
      mlp1.{0,1,3}.{weight,bias}
      language_model.model.embed_tokens.weight
      language_model.model.layers.X.self_attn.{q,k,v}_proj.{weight,bias}
      language_model.model.layers.X.self_attn.o_proj.weight
      language_model.model.layers.X.mlp.{gate_proj,up_proj,down_proj}.weight
      language_model.model.layers.X.{input_layernorm,post_attention_layernorm}.weight
      language_model.model.norm.weight
      language_model.lm_head.weight

    AD exports only the LLM forward path (input_ids → logits).  The vision tower
    runs in eager PyTorch and is not included in the torch.export graph.
    """

    config_class = None  # SkyworkChatConfig uses trust_remote_code; not imported here
    base_model_prefix = "language_model"
    _no_split_modules = ["SkyworkR1V2DecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def __init__(self, config, **kwargs):
        super().__init__(config)
        # config is a SkyworkChatConfig with a nested llm_config (Qwen2Config).
        llm_config = getattr(config, "llm_config", config)
        self.language_model = SkyworkR1V2LanguageModel(llm_config)

        # Vision tower — only present when config includes a vision_config.
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            self.vision_model = VisionModel(vision_config)
            self._select_layer = getattr(config, "select_layer", _HF_DEFAULT_SELECT_LAYER)
            self._downsample_ratio = getattr(
                config, "downsample_ratio", _HF_DEFAULT_DOWNSAMPLE_RATIO
            )
            self._ps_version = getattr(config, "ps_version", _HF_DEFAULT_PS_VERSION)
            scale = int(1 / self._downsample_ratio)
            vit_hidden = vision_config.hidden_size
            llm_hidden = llm_config.hidden_size
            self.mlp1 = nn.Sequential(
                nn.LayerNorm(vit_hidden * scale**2),
                nn.Linear(vit_hidden * scale**2, llm_hidden),
                nn.GELU(),
                nn.Linear(llm_hidden, llm_hidden),
            )

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.language_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.language_model.lm_head = new_embeddings

    def get_decoder(self):
        return self.language_model.model

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self._ps_version != "v1":
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the vision tower + MLP projector; returns embeddings in LLM hidden space."""
        if self._select_layer == -1:
            vit_embeds, _ = self.vision_model(pixel_values)
        else:
            _, all_hidden = self.vision_model(pixel_values, output_hidden_states=True)
            vit_embeds = all_hidden[self._select_layer]
        vit_embeds = vit_embeds[:, 1:, :]  # strip CLS token
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self._downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return self.mlp1(vit_embeds)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> SkyworkR1V2CausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


# Register with AutoModelForCausalLMFactory.
# The key must match type(config).__name__ for the config AD loads at runtime
# ("SkyworkChatConfig").  Without this, AD falls through to
# AutoModelForCausalLM.from_config which loads the full VLM — the HF vision
# component imports 'timm' (not installed) and crashes before any export begins.
AutoModelForCausalLMFactory.register_custom_model_cls(
    "SkyworkChatConfig", SkyworkR1V2ForConditionalGeneration
)
