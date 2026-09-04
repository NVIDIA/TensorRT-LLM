# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

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

"""MiniMax-H3 packed audio-video diffusion transformer.

The architecture follows the Diffusers implementations while using TRT-LLM's
linear, attention, normalization, and dynamic-weight-loading modules
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from tensorrt_llm._torch.models.hf_parameter_utils import get_parameter_device
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm.quantization.mode import QuantAlgo

from .modeling_utils import MiniMaxH3TimestepEmbedding, MiniMaxH3Timesteps

MINIMAX_H3_MODALITY_NUM = 3
MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2


@dataclass
class MiniMaxH3TransformerOutput:
    """Velocity predictions for packed video and audio rows."""

    sample: torch.Tensor
    audio_sample: torch.Tensor


@dataclass
class MiniMaxH3StaticContext:
    """Lossless request-static transformer inputs reusable across denoise steps."""

    text_embeds: torch.Tensor
    rotary_emb: tuple[torch.Tensor, torch.Tensor]

    @property
    def sequence_length(self) -> int:
        """Return the packed sequence length represented by the cached RoPE."""

        return self.rotary_emb[0].shape[0]


def apply_minimax_h3_rotary_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply MiniMax-H3's partial split-half RoPE to ``[B, S, H, D]``."""

    rotary_dim = cos.shape[-1]
    if rotary_dim > hidden_states.shape[-1] or rotary_dim % 2:
        raise ValueError(
            f"Invalid rotary dimension {rotary_dim} for head dimension {hidden_states.shape[-1]}."
        )

    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    x1, x2 = hidden_states_rotary.chunk(2, dim=-1)
    hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()


class MiniMaxH3RotaryPosEmbed(nn.Module):
    """Three-axis RoPE over packed ``(time, height, width)`` coordinates."""

    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                "position_ids must have shape [sequence_length, 3], got "
                f"{list(position_ids.shape)}."
            )
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        freqs = torch.cat(freqs.unbind(dim=1), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


class MiniMaxH3Attention(Attention):
    """Full non-causal attention with per-head RMSNorm and partial 3D RoPE."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm_eps: float,
        model_config: DiffusionModelConfig,
        layer_idx: int,
        module_name: str,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=attention_head_dim,
            qkv_mode=QKVMode.FUSE_QKV,
            qk_norm=True,
            qk_norm_mode="per_head",
            eps=qk_norm_eps,
            bias=False,
            # H3 rotates a leading split-half region and passes the tail through.
            # The shared fused kernel currently implements a different layout.
            fuse_qk_norm_rope=False,
            interleave=False,
            config=model_config,
            layer_idx=layer_idx,
            module_name=module_name,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length = hidden_states.shape[:2]
        query, key, value = self.get_qkv(hidden_states)
        query = query.view(
            batch_size, sequence_length, self.local_num_attention_heads, self.head_dim
        )
        key = key.view(batch_size, sequence_length, self.local_num_key_value_heads, self.head_dim)
        query, key = self.apply_qk_norm(query, key)

        if rotary_emb is not None:
            query = apply_minimax_h3_rotary_emb(query, *rotary_emb)
            key = apply_minimax_h3_rotary_emb(key, *rotary_emb)

        query = query.flatten(2)
        key = key.flatten(2)
        hidden_states = self._attn_impl(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            timestep=timestep,
        )
        return self.to_out[0](hidden_states)


def _rms_norm(hidden_size: int, eps: float, dtype: torch.dtype) -> RMSNorm:
    return RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype, has_weights=True)


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    """Produce six AdaLN vectors for every ``(timestep, modality)`` pair."""

    def __init__(
        self,
        time_embed_dim: int,
        hidden_size: int,
        model_config: DiffusionModelConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = Linear(
            time_embed_dim,
            6 * hidden_size * MINIMAX_H3_MODALITY_NUM,
            bias=True,
            dtype=model_config.torch_dtype,
            mapping=model_config.mapping,
            quant_config=model_config.quant_config,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            reduce_output=False,
        )

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temb = self.linear(F.silu(temb).to(self.linear.dtype))
        return temb.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormOut(nn.Module):
    """Final per-timestep shift/scale RMSNorm."""

    def __init__(
        self,
        hidden_size: int,
        time_embed_dim: int,
        eps: float,
        model_config: DiffusionModelConfig,
    ) -> None:
        super().__init__()
        self.norm = _rms_norm(hidden_size, eps, model_config.torch_dtype)
        self.linear = Linear(
            time_embed_dim,
            2 * hidden_size,
            bias=True,
            dtype=model_config.torch_dtype,
            mapping=model_config.mapping,
            quant_config=model_config.quant_config,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            reduce_output=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.linear(F.silu(temb).to(self.linear.dtype)).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        model_config: DiffusionModelConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.norm1 = _rms_norm(hidden_size, norm_eps, model_config.torch_dtype)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
            model_config=model_config,
            layer_idx=layer_idx,
            module_name=f"token_refiner.refiner_blocks.{layer_idx}.attn",
        )
        self.norm2 = _rms_norm(hidden_size, norm_eps, model_config.torch_dtype)
        self.ff = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=ffn_dim,
            bias=False,
            dtype=model_config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ff(hidden_states.reshape(-1, hidden_states.shape[-1])).reshape_as(
            hidden_states
        )
        return residual + hidden_states


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
        model_config: DiffusionModelConfig,
    ) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    model_config=model_config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.final_norm = _rms_norm(hidden_size, final_norm_eps, model_config.torch_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        model_config: DiffusionModelConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.norm1 = _rms_norm(hidden_size, norm_eps, model_config.torch_dtype)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
            model_config=model_config,
            layer_idx=layer_idx,
            module_name=f"transformer_blocks.{layer_idx}.attn",
        )
        self.norm2 = _rms_norm(hidden_size, norm_eps, model_config.torch_dtype)
        self.ff = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=ffn_dim,
            bias=False,
            dtype=model_config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(
            time_embed_dim=time_embed_dim,
            hidden_size=hidden_size,
            model_config=model_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

        residual = hidden_states
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_msa.index_select(0, adaln_indices)
        ) + shift_msa.index_select(0, adaln_indices)
        hidden_states = residual + gate_msa.index_select(0, adaln_indices) * self.attn(
            norm_hidden_states,
            rotary_emb,
            key_padding_mask,
            timestep,
        )

        residual = hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_mlp.index_select(0, adaln_indices)
        ) + shift_mlp.index_select(0, adaln_indices)
        ff_output = self.ff(
            norm_hidden_states.reshape(-1, norm_hidden_states.shape[-1])
        ).reshape_as(norm_hidden_states)
        return residual + gate_mlp.index_select(0, adaln_indices) * ff_output


class MiniMaxH3Transformer3DModel(BaseDiffusionModel):
    """MiniMax-H3's dense single-stream joint audio-video transformer.

    Args:
        model_config: VisualGen configuration containing the checkpoint
            architecture, dtype, distributed mapping, and loading settings.
    """

    def __init__(self, model_config: DiffusionModelConfig) -> None:
        super().__init__(model_config)
        if model_config.mapping.tp_size != 1:
            raise NotImplementedError(
                "MiniMax-H3 initial support is TP=1 only. AdaLN tensor-parallel "
                "sharding requires a model-specific gather contract."
            )
        quant_algo = model_config.quant_config.quant_algo
        if quant_algo is not None and not (
            quant_algo == QuantAlgo.FP8 and model_config.dynamic_weight_quant
        ):
            raise NotImplementedError(
                "MiniMax-H3 initial support allows BF16 or dynamic per-tensor FP8 weights only."
            )

        cfg = model_config.pretrained_config
        num_attention_heads = cfg.num_attention_heads
        attention_head_dim = cfg.attention_head_dim
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_layers
        num_refiner_layers = cfg.num_refiner_layers
        ffn_dim = cfg.ffn_dim
        in_channels = cfg.in_channels
        audio_in_channels = cfg.audio_in_channels
        patch_size = tuple(cfg.patch_size)
        text_dim = cfg.text_dim
        freq_dim = cfg.freq_dim
        time_embed_hidden_dim = cfg.time_embed_hidden_dim
        time_embed_dim = cfg.time_embed_dim
        rope_freq_dim = cfg.rope_freq_dim
        rope_theta = float(cfg.rope_theta)
        norm_eps = float(cfg.norm_eps)
        qk_norm_eps = float(cfg.qk_norm_eps)
        final_norm_eps = float(cfg.final_norm_eps)

        video_patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.config = SimpleNamespace(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )

        fp32_linear_kwargs = {
            "dtype": torch.float32,
            "mapping": model_config.mapping,
            "quant_config": model_config.quant_config,
            "skip_create_weights_in_init": model_config.skip_create_weights_in_init,
            "force_dynamic_quantization": model_config.force_dynamic_quantization,
            "reduce_output": False,
        }
        bf16_linear_kwargs = {
            "dtype": model_config.torch_dtype,
            "mapping": model_config.mapping,
            "quant_config": model_config.quant_config,
            "skip_create_weights_in_init": model_config.skip_create_weights_in_init,
            "force_dynamic_quantization": model_config.force_dynamic_quantization,
            "reduce_output": False,
        }

        self.proj_in = Linear(video_patch_dim, hidden_size, bias=True, **fp32_linear_kwargs)
        self.audio_proj_in = Linear(audio_in_channels, hidden_size, bias=True, **fp32_linear_kwargs)
        self.context_embedder = Linear(text_dim, hidden_size, bias=True, **bf16_linear_kwargs)

        self.time_proj = MiniMaxH3Timesteps(
            num_channels=freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = MiniMaxH3TimestepEmbedding(
            in_channels=freq_dim,
            time_embed_dim=time_embed_hidden_dim,
            out_dim=time_embed_dim,
        )
        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim, rope_theta)

        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_dim=ffn_dim,
            num_layers=num_refiner_layers,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
            model_config=model_config,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    model_config=model_config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size,
            time_embed_dim,
            final_norm_eps,
            model_config,
        )
        self.proj_out = Linear(hidden_size, video_patch_dim, bias=True, **fp32_linear_kwargs)
        self.audio_proj_out = Linear(
            hidden_size, audio_in_channels, bias=True, **fp32_linear_kwargs
        )

    @property
    def device(self) -> torch.device:
        return get_parameter_device(self)

    def prepare_static_context(
        self,
        encoder_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> MiniMaxH3StaticContext:
        """Project/refine text and build RoPE once for an entire request."""

        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.dtype))
        text_embeds = self.token_refiner(text_embeds)
        return MiniMaxH3StaticContext(
            text_embeds=text_embeds,
            rotary_emb=self.rope(position_ids),
        )

    def _validate_layout(
        self,
        position_ids: torch.Tensor,
        token_tags: torch.Tensor,
        timestep_indices: torch.Tensor,
        timestep: torch.Tensor,
    ) -> None:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"position_ids must have shape [sequence_length, 3], got {list(position_ids.shape)}."
            )
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "token_tags and timestep_indices must match the packed sequence length."
            )
        if token_tags.numel() and (
            token_tags.min().item() < -1 or token_tags.max().item() >= MINIMAX_H3_MODALITY_NUM
        ):
            raise ValueError("token_tags values must be one of -1, 0, 1, or 2.")
        if timestep_indices.numel() and (
            timestep_indices.min().item() < 0 or timestep_indices.max().item() >= timestep.shape[0]
        ):
            raise ValueError("timestep_indices contains an index outside the timestep table.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: Optional[dict[str, object]] = None,
        return_dict: bool = True,
        static_context: Optional[MiniMaxH3StaticContext] = None,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        del attention_kwargs
        if position_ids is None:
            if static_context is None:
                raise ValueError("position_ids is required when static_context is not provided.")
            sequence_length = static_context.sequence_length
        else:
            sequence_length = position_ids.shape[0]
            self._validate_layout(position_ids, token_tags, timestep_indices, timestep)

        if static_context is None:
            if encoder_hidden_states is None or position_ids is None:
                raise ValueError(
                    "encoder_hidden_states and position_ids are required to prepare static context."
                )
            static_context = self.prepare_static_context(
                encoder_hidden_states,
                position_ids,
            )
        elif token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "token_tags and timestep_indices must match the cached packed sequence length."
            )

        if static_context.sequence_length != sequence_length:
            raise ValueError("static_context RoPE length must match the packed sequence length.")

        if static_context.text_embeds.shape[1] != text_indices.numel():
            raise ValueError("text_indices must contain one entry for every cached text embedding.")

        video_embeds = self.proj_in(hidden_states.to(self.proj_in.dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.dtype))
        text_embeds = static_context.text_embeds
        packed_hidden_states = text_embeds.new_zeros(
            (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
        )
        packed_hidden_states = packed_hidden_states.index_copy(1, text_indices, text_embeds)
        packed_hidden_states = packed_hidden_states.index_copy(
            1, video_indices, video_embeds.to(text_embeds.dtype)
        )
        packed_hidden_states = packed_hidden_states.index_copy(
            1, audio_indices, audio_embeds.to(text_embeds.dtype)
        )

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)
        key_padding_mask = None
        if bool((token_tags < 0).any()):
            key_padding_mask = (
                (token_tags >= 0).unsqueeze(0).expand(packed_hidden_states.shape[0], -1)
            )

        for block in self.transformer_blocks:
            packed_hidden_states = block(
                packed_hidden_states,
                temb,
                adaln_indices,
                static_context.rotary_emb,
                key_padding_mask,
                timestep,
            )

        packed_hidden_states = self.norm_out(
            packed_hidden_states,
            temb,
            timestep_indices,
        )
        video_output = self.proj_out(packed_hidden_states.to(self.proj_out.dtype)).index_select(
            1, video_indices
        )
        audio_output = self.audio_proj_out(
            packed_hidden_states.to(self.audio_proj_out.dtype)
        ).index_select(1, audio_indices)

        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(
            sample=video_output,
            audio_sample=audio_output,
        )

    @staticmethod
    def _remap_feed_forward_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        remapped = {}
        for key, value in weights.items():
            if ".ff.net.0.proj." in key:
                prefix, suffix = key.split("net.0.proj.", maxsplit=1)
                # Diffusers' SwiGLU stores ``[up, gate]`` while TRT-LLM's
                # fused GatedMLP consumes ``[gate, up]``.
                up, gate = value.chunk(2, dim=0)
                remapped[f"{prefix}gate.{suffix}"] = gate
                remapped[f"{prefix}up.{suffix}"] = up
            else:
                remapped[key.replace(".ff.net.2.", ".ff.down_proj.")] = value
        return remapped

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load the official converted Diffusers checkpoint layout."""

        weights = self._remap_feed_forward_weights(weights)
        loader = DynamicLinearWeightLoader(
            self.model_config,
            params_map={
                "qkv_proj": ["to_q", "to_k", "to_v"],
                "gate_up_proj": ["gate", "up"],
            },
        )
        device = self.device
        missing = []
        for name, module in tqdm(
            self.named_modules(),
            desc="Loading MiniMax-H3 weights",
        ):
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()
                module.to(device)
            if len(module._parameters) == 0:
                continue

            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)
                required_parameters = ["weight"]
                if module.bias is not None:
                    required_parameters.append("bias")
                missing_parameters = [
                    parameter_name
                    for parameter_name in required_parameters
                    if not weight_dicts
                    or any(parameter_name not in entry for entry in weight_dicts)
                ]
                if missing_parameters:
                    missing.extend(
                        f"{name}.{parameter_name}" for parameter_name in missing_parameters
                    )
                    continue
                loader.load_linear_weights(module, name, weight_dicts)
                continue

            module_weights = loader.filter_weights(name, weights)
            for param_name, param in module._parameters.items():
                if param is None:
                    continue
                if param_name not in module_weights:
                    missing.append(f"{name}.{param_name}")
                    continue
                param.data.copy_(module_weights[param_name].to(param.dtype))

        if missing:
            preview = ", ".join(missing[:8])
            suffix = " ..." if len(missing) > 8 else ""
            raise ValueError(
                f"MiniMax-H3 checkpoint is missing {len(missing)} required tensors: "
                f"{preview}{suffix}"
            )

    def post_load_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, Linear):
                module.post_load_weights()
        # The released BF16 checkpoint deliberately keeps these modules in FP32.
        for module in (
            self.proj_in,
            self.audio_proj_in,
            self.proj_out,
            self.audio_proj_out,
        ):
            if not module.has_any_quant:
                module.to(torch.float32)
        self.time_embedder.to(torch.float32)
