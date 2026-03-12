# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slimmed down Phi-4-reasoning-vision-15B implementation for auto_deploy export.

Source:
https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B

This implementation differs from the Hugging Face version in the following ways:
* The top-level config is flattened to match the checkpoint config.json, but a nested
  ``text_config`` is created internally so AutoDeploy can export the text submodule.
* The text decoder is prefill-only (no KV cache / decode-time runtime args).
* The text attention uses AutoDeploy reference ops (``torch_attention`` and
  ``torch_rope_with_explicit_cos_sin``).
* The SigLIP2 vision tower stays in eager mode and is used only by the wrapper path.
* Vision placeholders are expanded into embeddings in plain PyTorch before calling the
  exported text submodule.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from torch import nn
from torch._prims_common import DeviceLikeType
from transformers import AutoConfig, Siglip2VisionConfig
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.models.hf import (
    AutoModelForCausalLMFactory,
    AutoModelForImageTextToTextFactory,
)
from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger
from tensorrt_llm.tokenizer.tokenizer import TransformersTokenizer

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200


def _infer_siglip_patch_size(mm_vision_tower: Optional[str], vision_config: Dict) -> int:
    patch_size = vision_config.get("patch_size", None)
    if patch_size is not None:
        return patch_size
    tower_name = (mm_vision_tower or "").lower()
    if "patch14" in tower_name:
        return 14
    return 16


def build_vision_projector(config: "Phi4VisionRConfig") -> nn.Module:
    """Build the MLP projector used by the HF checkpoint."""
    projector_type = getattr(config, "mm_projector_type", "mlp2x_gelu")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.text_config.hidden_size)

    if projector_type.startswith("mlp") and projector_type.endswith("x_gelu"):
        depth = int(projector_type[len("mlp") : projector_type.index("x_gelu")])
        modules: List[nn.Module] = [
            nn.Linear(config.mm_hidden_size, config.text_config.hidden_size)
        ]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(
                nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)
            )
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return nn.Identity()

    raise ValueError(f"Unsupported mm_projector_type: {projector_type}")


class Phi4VisionRConfig(PretrainedConfig):
    """Flattened config for Phi-4-reasoning-vision-15B with nested text_config."""

    model_type = "phi4-siglip"

    def __init__(
        self,
        text_config: Optional[Union[Phi3Config, Dict]] = None,
        vision_config: Optional[Union[Siglip2VisionConfig, Dict]] = None,
        mm_vision_tower: Optional[str] = None,
        mm_projector_type: str = "mlp2x_gelu",
        mm_hidden_size: int = 1152,
        min_num_patches: int = 256,
        max_num_patches: int = 3600,
        tokenizer_model_max_length: int = 16384,
        tokenizer_padding_side: str = "right",
        use_mm_proj: bool = True,
        image_aspect_ratio: str = "square",
        freeze_mm_mlp_adapter: bool = False,
        tune_mm_mlp_adapter: bool = False,
        unfreeze_vision_tower: bool = True,
        use_s2: bool = False,
        mm_projector_lr: Optional[float] = None,
        **kwargs,
    ):
        if text_config is None:
            text_kwargs = dict(kwargs)
            text_config = Phi3Config(**text_kwargs)
        elif isinstance(text_config, dict):
            text_config = Phi3Config(**text_config)

        vision_config_dict: Dict
        if vision_config is None:
            vision_config_dict = {}
        elif isinstance(vision_config, Siglip2VisionConfig):
            vision_config_dict = vision_config.to_dict()
        else:
            vision_config_dict = dict(vision_config)

        vision_config_dict.setdefault(
            "patch_size", _infer_siglip_patch_size(mm_vision_tower, vision_config_dict)
        )
        self.vision_config = Siglip2VisionConfig(**vision_config_dict)
        self.text_config = text_config

        self.mm_vision_tower = mm_vision_tower
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.tokenizer_model_max_length = tokenizer_model_max_length
        self.tokenizer_padding_side = tokenizer_padding_side
        self.use_mm_proj = use_mm_proj
        self.image_aspect_ratio = image_aspect_ratio
        self.freeze_mm_mlp_adapter = freeze_mm_mlp_adapter
        self.tune_mm_mlp_adapter = tune_mm_mlp_adapter
        self.unfreeze_vision_tower = unfreeze_vision_tower
        self.use_s2 = use_s2
        self.mm_projector_lr = mm_projector_lr

        # Mirror common text attributes on the top-level config for compatibility.
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.intermediate_size = self.text_config.intermediate_size
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_key_value_heads = self.text_config.num_key_value_heads
        self.hidden_act = self.text_config.hidden_act
        self.max_position_embeddings = self.text_config.max_position_embeddings
        self.original_max_position_embeddings = getattr(
            self.text_config,
            "original_max_position_embeddings",
            self.text_config.max_position_embeddings,
        )
        self.rms_norm_eps = self.text_config.rms_norm_eps
        self.rope_theta = self.text_config.rope_theta
        self.partial_rotary_factor = self.text_config.partial_rotary_factor
        self.initializer_range = self.text_config.initializer_range
        self.attention_bias = self.text_config.attention_bias
        self.attention_dropout = self.text_config.attention_dropout
        self.pad_token_id = self.text_config.pad_token_id
        self.bos_token_id = self.text_config.bos_token_id
        self.eos_token_id = self.text_config.eos_token_id
        self.tie_word_embeddings = self.text_config.tie_word_embeddings

        super().__init__(
            pad_token_id=self.text_config.pad_token_id,
            bos_token_id=self.text_config.bos_token_id,
            eos_token_id=self.text_config.eos_token_id,
            tie_word_embeddings=self.text_config.tie_word_embeddings,
        )


def _coerce_phi4_visionr_config(config: PretrainedConfig) -> Phi4VisionRConfig:
    if isinstance(config, Phi4VisionRConfig):
        return config

    if hasattr(config, "text_config"):
        return Phi4VisionRConfig(
            text_config=getattr(config, "text_config"),
            vision_config=getattr(config, "vision_config", None),
            mm_vision_tower=getattr(config, "mm_vision_tower", None),
            mm_projector_type=getattr(config, "mm_projector_type", "mlp2x_gelu"),
            mm_hidden_size=getattr(config, "mm_hidden_size", 1152),
            min_num_patches=getattr(config, "min_num_patches", 256),
            max_num_patches=getattr(config, "max_num_patches", 3600),
            tokenizer_model_max_length=getattr(config, "tokenizer_model_max_length", 16384),
            tokenizer_padding_side=getattr(config, "tokenizer_padding_side", "right"),
            use_mm_proj=getattr(config, "use_mm_proj", True),
            image_aspect_ratio=getattr(config, "image_aspect_ratio", "square"),
            freeze_mm_mlp_adapter=getattr(config, "freeze_mm_mlp_adapter", False),
            tune_mm_mlp_adapter=getattr(config, "tune_mm_mlp_adapter", False),
            unfreeze_vision_tower=getattr(config, "unfreeze_vision_tower", True),
            use_s2=getattr(config, "use_s2", False),
            mm_projector_lr=getattr(config, "mm_projector_lr", None),
        )

    text_config_kwargs = {}
    for key in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_act",
        "max_position_embeddings",
        "original_max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
        "rope_scaling",
        "partial_rotary_factor",
        "attention_dropout",
        "attention_bias",
        "resid_pdrop",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "tie_word_embeddings",
        "initializer_range",
        "head_dim",
    ):
        if hasattr(config, key):
            text_config_kwargs[key] = getattr(config, key)

    return Phi4VisionRConfig(
        text_config=Phi3Config(**text_config_kwargs),
        vision_config=getattr(config, "vision_config", None),
        mm_vision_tower=getattr(config, "mm_vision_tower", None),
        mm_projector_type=getattr(config, "mm_projector_type", "mlp2x_gelu"),
        mm_hidden_size=getattr(config, "mm_hidden_size", 1152),
        min_num_patches=getattr(config, "min_num_patches", 256),
        max_num_patches=getattr(config, "max_num_patches", 3600),
        tokenizer_model_max_length=getattr(config, "tokenizer_model_max_length", 16384),
        tokenizer_padding_side=getattr(config, "tokenizer_padding_side", "right"),
        use_mm_proj=getattr(config, "use_mm_proj", True),
        image_aspect_ratio=getattr(config, "image_aspect_ratio", "square"),
        freeze_mm_mlp_adapter=getattr(config, "freeze_mm_mlp_adapter", False),
        tune_mm_mlp_adapter=getattr(config, "tune_mm_mlp_adapter", False),
        unfreeze_vision_tower=getattr(config, "unfreeze_vision_tower", True),
        use_s2=getattr(config, "use_s2", False),
        mm_projector_lr=getattr(config, "mm_projector_lr", None),
    )


class Phi4VisionRMSNorm(nn.Module):
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


class Phi4VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("_ad_inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self._ad_inv_freq.dtype)
        freqs = torch.outer(t, self._ad_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


class Phi4VisionMLP(nn.Module):
    def __init__(self, config: Phi3Config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)


class Phi4VisionAttention(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** (-0.5)

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.qkv_proj = nn.Linear(config.hidden_size, op_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        kv_pos = self.num_key_value_heads * self.head_dim

        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + kv_pos]
        value_states = qkv[..., query_pos + kv_pos :]

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )

        cos, sin = position_embeddings
        cos = cos[position_ids]
        sin = sin[position_ids]

        query_states, key_states = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            query_states, key_states, cos, sin, 2
        )

        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            scale=self.scaling,
            layout="bsnd",
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class Phi4VisionDecoderLayer(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Phi4VisionAttention(config, layer_idx=layer_idx)
        self.mlp = Phi4VisionMLP(config)
        self.input_layernorm = Phi4VisionRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Phi4VisionRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@dataclass
class Phi4VisionTextOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Phi4VisionCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


def _prepare_4d_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mask = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - mask) * torch.finfo(dtype).min


def _eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
) -> torch.Tensor:
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(attn_weights, value).transpose(1, 2).contiguous()


class Phi4VisionSiglip2Embeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Linear(
            config.num_channels * self.patch_size * self.patch_size,
            self.embed_dim,
        )
        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype
        resized_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for idx in range(batch_size):
            height, width = spatial_shapes[idx]
            resized = F.interpolate(
                positional_embeddings,
                size=(int(height), int(width)),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized = resized.reshape(embed_dim, int(height) * int(width)).transpose(0, 1)
            resized = resized.to(source_dtype)
            resized_embeddings[idx, : int(height) * int(width)] = resized
            resized_embeddings[idx, int(height) * int(width) :] = resized[0]
        return resized_embeddings

    def forward(
        self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor
    ) -> torch.Tensor:
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=self.patch_embedding.weight.dtype)
        )
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings,
            spatial_shapes,
            max_length=pixel_values.shape[1],
        )
        return patch_embeds + resized_positional_embeddings


class Phi4VisionSiglip2Attention(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length, embed_dim = hidden_states.shape
        query_states = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        attn_output = _eager_attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            scaling=self.scale,
        )
        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim)
        return self.out_proj(attn_output), None


class Phi4VisionSiglip2MLP(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Phi4VisionSiglip2EncoderLayer(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Phi4VisionSiglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Phi4VisionSiglip2MLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Phi4VisionSiglip2Encoder(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [Phi4VisionSiglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        all_hidden_states = [hidden_states] if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


class Phi4VisionSiglip2MultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Phi4VisionSiglip2MLP(config)
        self.num_heads = config.num_attention_heads

    def forward(
        self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        if attention_mask is not None:
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attn_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype)
            attn_mask = attn_mask.repeat(1, self.num_heads, target_len, 1)
            attn_mask = attn_mask.reshape(-1, target_len, source_len)
        else:
            attn_mask = None
        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attn_mask)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class Phi4VisionSiglip2VisionTransformer(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.embeddings = Phi4VisionSiglip2Embeddings(config)
        self.encoder = Phi4VisionSiglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = Phi4VisionSiglip2MultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_hidden_states: bool = False,
    ) -> BaseModelOutputWithPooling:
        hidden_states = self.embeddings(pixel_values, spatial_shapes)
        encoder_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attention_mask,
            output_hidden_states=output_hidden_states,
        )
        last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
        pooler_output = self.head(last_hidden_state, attention_mask)
        hidden_states_out = None
        if output_hidden_states and encoder_outputs.hidden_states is not None:
            hidden_states_out = tuple(
                self.post_layernorm(hs) if idx == len(encoder_outputs.hidden_states) - 1 else hs
                for idx, hs in enumerate(encoder_outputs.hidden_states)
            )
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=hidden_states_out,
        )


class Phi4VisionSiglip2VisionModel(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.vision_model = Phi4VisionSiglip2VisionTransformer(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_hidden_states: bool = False,
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_hidden_states=output_hidden_states,
        )


class Phi4VisionTower(nn.Module):
    """Vision tower wrapper with the checkpoint-compatible nested ``vision_tower`` path."""

    def __init__(self, config: Phi4VisionRConfig):
        super().__init__()
        self.vision_tower = Phi4VisionSiglip2VisionModel(config.vision_config)
        self.select_layer = -2

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
    ) -> List[torch.Tensor]:
        outputs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[self.select_layer]
        return [
            hidden_states[idx][pixel_attention_mask[idx].bool()]
            for idx in range(hidden_states.shape[0])
        ]


class Phi4VisionRModel(nn.Module):
    """Multimodal wrapper that keeps vision eager and calls the exported text model."""

    def __init__(self, config: Phi4VisionRConfig):
        super().__init__()
        config = _coerce_phi4_visionr_config(config)
        self.full_config = config
        self.config = config.text_config
        self.padding_idx = self.config.pad_token_id
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Phi4VisionDecoderLayer(self.config, layer_idx=idx)
                for idx in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = Phi4VisionRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        head_dim = getattr(
            self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads
        )
        rotary_dim = int(head_dim * self.config.partial_rotary_factor)
        self.rotary_emb = Phi4VisionRotaryEmbedding(
            rotary_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )
        self.vision_tower = Phi4VisionTower(config)
        self.mm_projector = build_vision_projector(config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _forward_text(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Phi4VisionTextOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            raise ValueError("position_ids must be provided")

        position_embeddings = self.rotary_emb(inputs_embeds)
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids, position_embeddings)
        hidden_states = self.norm(hidden_states)
        return Phi4VisionTextOutput(last_hidden_state=hidden_states)

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
    ) -> List[torch.Tensor]:
        features = self.vision_tower(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        return [self.mm_projector(feat) for feat in features]

    def _normalize_vision_inputs(
        self,
        pixel_values: Union[torch.Tensor, Sequence[torch.Tensor]],
        pixel_attention_mask: Union[torch.Tensor, Sequence[torch.Tensor]],
        spatial_shapes: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.cat(list(pixel_values), dim=0)
        if isinstance(pixel_attention_mask, (list, tuple)):
            pixel_attention_mask = torch.cat(list(pixel_attention_mask), dim=0)
        if isinstance(spatial_shapes, (list, tuple)):
            spatial_shapes = torch.cat(list(spatial_shapes), dim=0)
        return pixel_values, pixel_attention_mask, spatial_shapes

    def _merge_image_embeddings(
        self,
        input_ids: torch.Tensor,
        image_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        embeds_per_sample: List[torch.Tensor] = []
        image_idx = 0
        embedding = self.get_input_embeddings()

        for sample_ids in input_ids:
            image_positions = torch.where(sample_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            if not image_positions:
                embeds_per_sample.append(embedding(sample_ids))
                continue

            pieces: List[torch.Tensor] = []
            cursor = 0
            for pos in image_positions:
                if cursor < pos:
                    pieces.append(embedding(sample_ids[cursor:pos]))
                pieces.append(image_features[image_idx])
                image_idx += 1
                cursor = pos + 1
            if cursor < sample_ids.numel():
                pieces.append(embedding(sample_ids[cursor:]))
            embeds_per_sample.append(torch.cat(pieces, dim=0))

        if image_idx != len(image_features):
            raise ValueError(
                f"Consumed {image_idx} image feature groups, but received {len(image_features)}."
            )
        return embeds_per_sample

    def _forward_with_images(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
    ) -> Phi4VisionTextOutput:
        pixel_values, pixel_attention_mask, spatial_shapes = self._normalize_vision_inputs(
            pixel_values,
            pixel_attention_mask,
            spatial_shapes,
        )
        image_features = self.encode_images(pixel_values, pixel_attention_mask, spatial_shapes)
        inputs_embeds_per_sample = self._merge_image_embeddings(input_ids, image_features)

        outputs = []
        max_seq_len = max(embed.shape[0] for embed in inputs_embeds_per_sample)
        for embeds in inputs_embeds_per_sample:
            seq_len = embeds.shape[0]
            position_ids = torch.arange(seq_len, device=embeds.device, dtype=torch.long).unsqueeze(
                0
            )
            hidden_states = self._forward_text(
                inputs_embeds=embeds.unsqueeze(0),
                position_ids=position_ids,
            ).last_hidden_state.squeeze(0)
            if seq_len < max_seq_len:
                hidden_states = torch.cat(
                    [
                        hidden_states,
                        hidden_states.new_zeros(max_seq_len - seq_len, hidden_states.shape[-1]),
                    ],
                    dim=0,
                )
            outputs.append(hidden_states)

        return Phi4VisionTextOutput(last_hidden_state=torch.stack(outputs, dim=0))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        pixel_attention_mask: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        spatial_shapes: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        **kwargs,
    ) -> Phi4VisionTextOutput:
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("input_ids must be provided when pixel_values are present")
            if pixel_attention_mask is None or spatial_shapes is None:
                raise ValueError(
                    "pixel_attention_mask and spatial_shapes must be provided with pixel_values"
                )
            return self._forward_with_images(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )

        return self._forward_text(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )


class Phi4VisionRForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """Top-level Phi-4-reasoning-vision-15B model for AutoDeploy."""

    config_class = Phi4VisionRConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["Phi4VisionDecoderLayer"]

    def __init__(self, config: Phi4VisionRConfig, **kwargs):
        config = _coerce_phi4_visionr_config(config)
        super().__init__(config)
        self.model = Phi4VisionRModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()

    def _init_weights(self, module):
        std = self.config.text_config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        pixel_attention_mask: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        spatial_shapes: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        **kwargs,
    ) -> Phi4VisionCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        return Phi4VisionCausalLMOutput(logits=logits)


class Phi4VisionRTokenizer(TransformersTokenizer):
    """Tokenizer wrapper that applies the default chat template for raw text prompts."""

    def encode(self, text: str, *args, **kwargs) -> List[int]:
        if (
            getattr(self.tokenizer, "chat_template", None) is not None
            and "<|im_start|>" not in text
        ):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return super().encode(text, *args, **kwargs)


class Phi4VisionRProcessorWrapper:
    """Processor wrapper that uses the tokenizer chat template for text-only messages."""

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = Phi4VisionRTokenizer(processor.tokenizer)

    def __call__(self, *args, **kwargs):
        return self.processor(*args, **kwargs)

    def apply_chat_template(self, conversation, *args, **kwargs):
        return self.tokenizer.apply_chat_template(conversation, *args, **kwargs)


@ModelFactoryRegistry.register("Phi4VisionRAutoModelForImageTextToText")
class Phi4VisionRAutoModelForImageTextToTextFactory(AutoModelForImageTextToTextFactory):
    """Model-specific ImageTextToText factory that keeps Phi-4 prompt templating out of llm.py."""

    def _build_model(self, device: DeviceLikeType) -> nn.Module:
        model_config, unused_kwargs = self._get_model_config()
        with (init_empty_weights if device == "meta" else nullcontext)():
            ad_logger.info(
                f"Using custom model implementation {Phi4VisionRForConditionalGeneration}"
            )
            model = Phi4VisionRForConditionalGeneration._from_config(model_config, **unused_kwargs)

        if device == "meta":
            if hasattr(model, "post_init"):
                model.post_init()
        else:
            model.to(device)

        self._set_sharding_config(model.config)
        self._checkpoint_conversion_mapping = getattr(model, "_checkpoint_conversion_mapping", None)
        model.eval()
        return model

    def init_processor(self) -> Optional[Phi4VisionRProcessorWrapper]:
        processor = super().init_processor()
        if processor is None:
            return None
        return Phi4VisionRProcessorWrapper(processor)

    def init_tokenizer(self) -> Optional[Phi4VisionRTokenizer]:
        processor = self.init_processor()
        if processor is None:
            return None
        return processor.tokenizer


try:
    AutoConfig.register("phi4-siglip", Phi4VisionRConfig, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register("phi4-siglip", Phi4VisionRConfig)
    except ValueError:
        pass

AutoModelForCausalLMFactory.register_custom_model_cls(
    "Phi4VisionRConfig", Phi4VisionRForConditionalGeneration
)
AutoModelForCausalLMFactory.register_custom_model_cls(
    "Phi4VisionR", Phi4VisionRForConditionalGeneration
)
AutoModelForImageTextToTextFactory.register_custom_model_cls(
    "Phi4VisionRConfig", Phi4VisionRForConditionalGeneration
)
AutoModelForImageTextToTextFactory.register_custom_model_cls(
    "Phi4VisionR", Phi4VisionRForConditionalGeneration
)
