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

"""MiniMax-H3 packed audio-video diffusion transformer."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm.quantization.mode import QuantAlgo

from ..kernels.fused_adaln import h3_adaln_supported
from ..ops.adaln import gate_res_norm_mod as h3_gate_res_norm_mod
from ..ops.adaln import norm_mod as h3_norm_mod
from ..ops.vsa import H3VSAMetadata
from ..parallel.sequence import H3SequenceSharder, shard_dit_inputs, shard_refiner_inputs
from ..weights import load_transformer_weights
from .attention import (
    MiniMaxH3Attention,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3VSAAttention,
    _dense_attention_model_config,
)
from .linear import Linear
from .mlp import GatedMLP
from .norm import RMSNorm

MINIMAX_H3_MODALITY_NUM = 3


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


def _norm_2d(norm: RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
    """Flatten packed rows to select the CTA-per-row RMSNorm kernel."""
    shape = hidden_states.shape
    return norm(hidden_states.reshape(-1, shape[-1])).view(shape)


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
            tensor_parallel_mode=None,
            reduce_output=False,
        )

    def table(self, temb: torch.Tensor) -> torch.Tensor:
        """Return the modulation table ``[n_t * 3, 6 * hidden_size]``.

        One row per ``(timestep, modality)`` pair; the six column blocks are
        shift, scale and gate for attention, then shift, scale and gate for the MLP.
        """
        temb = self.linear(F.silu(temb).to(self.linear.dtype))
        return temb.view(-1, 6 * self.hidden_size)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.table(temb).chunk(6, dim=-1)


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
        self.norm = RMSNorm(
            hidden_size=hidden_size, eps=eps, dtype=model_config.torch_dtype, has_weights=True
        )
        self.linear = Linear(
            time_embed_dim,
            2 * hidden_size,
            bias=True,
            dtype=model_config.torch_dtype,
            mapping=model_config.mapping,
            quant_config=model_config.quant_config,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            tensor_parallel_mode=None,
            reduce_output=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.linear(F.silu(temb).to(self.linear.dtype)).chunk(2, dim=-1)
        hidden_states = _norm_2d(self.norm, hidden_states)
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
        fuse_bf16_gate_up_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(
            hidden_size=hidden_size, eps=norm_eps, dtype=model_config.torch_dtype, has_weights=True
        )
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
            model_config=_dense_attention_model_config(model_config),
            layer_idx=layer_idx,
            module_name=f"token_refiner.refiner_blocks.{layer_idx}.attn",
        )
        self.norm2 = RMSNorm(
            hidden_size=hidden_size, eps=norm_eps, dtype=model_config.torch_dtype, has_weights=True
        )
        self.ff = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=ffn_dim,
            bias=False,
            dtype=model_config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
            reduce_output=model_config.mapping.tp_size > 1,
            fuse_bf16_gate_up_swiglu=fuse_bf16_gate_up_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        valid_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            _norm_2d(self.norm1, hidden_states),
            key_padding_mask=key_padding_mask,
            valid_seq_len=valid_seq_len,
        )
        residual = hidden_states
        hidden_states = _norm_2d(self.norm2, hidden_states)
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
        fuse_bf16_gate_up_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self._attn_backend = model_config.attention.backend.upper()
        self._supports_key_padding_mask = self._attn_backend in (
            "VANILLA",
            "FA4",
            "TRTLLM",
            "FLASHINFER",
        )
        self.sharder = H3SequenceSharder(
            model_config.visual_gen_mapping,
            num_attention_heads=num_attention_heads,
        )
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
                    fuse_bf16_gate_up_swiglu=fuse_bf16_gate_up_swiglu,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(
            hidden_size=hidden_size,
            eps=final_norm_eps,
            dtype=model_config.torch_dtype,
            has_weights=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        hidden_states, key_padding_mask, valid_seq_len = shard_refiner_inputs(
            hidden_states, self.sharder, self._attn_backend, self._supports_key_padding_mask
        )
        for block in self.refiner_blocks:
            hidden_states = block(
                hidden_states,
                key_padding_mask=key_padding_mask,
                valid_seq_len=valid_seq_len,
            )
        hidden_states = _norm_2d(self.final_norm, hidden_states)
        return self.sharder.gather(hidden_states, dim=1, unpad_to=seq_len)


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
        attention_cls: type[MiniMaxH3Attention] | None = None,
        fuse_bf16_gate_up_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(
            hidden_size=hidden_size, eps=norm_eps, dtype=model_config.torch_dtype, has_weights=True
        )
        if attention_cls is None:
            sparse = model_config.attention.sparse_attention_config
            attention_cls = (
                MiniMaxH3VSAAttention
                if sparse is not None and sparse.algorithm == "vsa"
                else MiniMaxH3Attention
            )
        self.attn = attention_cls(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
            model_config=model_config,
            layer_idx=layer_idx,
            module_name=f"transformer_blocks.{layer_idx}.attn",
        )
        self.norm2 = RMSNorm(
            hidden_size=hidden_size, eps=norm_eps, dtype=model_config.torch_dtype, has_weights=True
        )
        self.ff = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=ffn_dim,
            bias=False,
            dtype=model_config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
            reduce_output=model_config.mapping.tp_size > 1,
            fuse_bf16_gate_up_swiglu=fuse_bf16_gate_up_swiglu,
        )
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(
            time_embed_dim=time_embed_dim,
            hidden_size=hidden_size,
            model_config=model_config,
        )
        self._fused_adaln = h3_adaln_supported(hidden_size, model_config.torch_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        valid_seq_len: Optional[int] = None,
        vsa_metadata: H3VSAMetadata | None = None,
    ) -> torch.Tensor:
        if self._fused_adaln and hidden_states.dtype == torch.bfloat16:
            return self._forward_fused_adaln(
                hidden_states,
                temb,
                adaln_indices,
                rotary_emb,
                key_padding_mask,
                timestep,
                valid_seq_len,
                vsa_metadata,
            )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

        residual = hidden_states
        norm_hidden_states = _norm_2d(self.norm1, hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_msa.index_select(0, adaln_indices)
        ) + shift_msa.index_select(0, adaln_indices)
        hidden_states = residual + gate_msa.index_select(0, adaln_indices) * self.attn(
            norm_hidden_states,
            rotary_emb,
            key_padding_mask,
            timestep,
            valid_seq_len,
            vsa_metadata=vsa_metadata,
        )

        residual = hidden_states
        norm_hidden_states = _norm_2d(self.norm2, hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_mlp.index_select(0, adaln_indices)
        ) + shift_mlp.index_select(0, adaln_indices)
        ff_output = self.ff(
            norm_hidden_states.reshape(-1, norm_hidden_states.shape[-1])
        ).reshape_as(norm_hidden_states)
        return residual + gate_mlp.index_select(0, adaln_indices) * ff_output

    def _forward_fused_adaln(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        timestep: Optional[torch.Tensor],
        valid_seq_len: Optional[int],
        vsa_metadata: H3VSAMetadata | None,
    ) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        mod = self.adaln_proj.table(temb)
        norm_hidden_states = h3_norm_mod(
            hidden_states, self.norm1.weight, mod, adaln_indices, self.norm1.variance_epsilon, 1, 0
        )
        attn_output = self.attn(
            norm_hidden_states,
            rotary_emb,
            key_padding_mask,
            timestep,
            valid_seq_len,
            vsa_metadata=vsa_metadata,
        )
        hidden_states, norm_hidden_states = h3_gate_res_norm_mod(
            hidden_states,
            attn_output,
            self.norm2.weight,
            mod,
            adaln_indices,
            self.norm2.variance_epsilon,
            2,
            4,
            3,
        )
        ff_output = self.ff(
            norm_hidden_states.reshape(-1, norm_hidden_states.shape[-1])
        ).reshape_as(norm_hidden_states)
        gate_mlp = mod[:, 5 * hidden_size : 6 * hidden_size]
        return hidden_states + gate_mlp.index_select(0, adaln_indices) * ff_output


class MiniMaxH3Transformer3DModel(BaseDiffusionModel):
    """MiniMax-H3's packed single-stream joint audio-video transformer.

    Args:
        model_config: VisualGen configuration containing the checkpoint
            architecture, dtype, distributed mapping, and loading settings.
    """

    def __init__(
        self,
        model_config: DiffusionModelConfig,
        *,
        attention_cls: type[MiniMaxH3Attention] | None = None,
    ) -> None:
        super().__init__(model_config)
        self._attn_backend = getattr(model_config.attention, "backend", "VANILLA")
        self._supports_key_padding_mask = self._attn_backend in (
            "VANILLA",
            "FA4",
            "TRTLLM",
            "FLASHINFER",
        )
        quant_algo = model_config.quant_config.quant_algo
        if quant_algo is not None and not (
            quant_algo in (QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.NVFP4)
            and model_config.dynamic_weight_quant
        ):
            raise NotImplementedError(
                "MiniMax-H3 supports BF16, or dynamically quantized FP8 / FP8_BLOCK_SCALES / NVFP4 weights only."
            )
        # The NVFP4 quantizer consumes BF16 media activations.
        self._projection_act_dtype = (
            model_config.torch_dtype if quant_algo == QuantAlgo.NVFP4 else torch.float32
        )
        # The dynamic loader does not initialize NVFP4's static activation scale.
        if quant_algo == QuantAlgo.NVFP4 and not model_config.force_dynamic_quantization:
            raise NotImplementedError(
                f"MiniMax-H3 {quant_algo.name} requires force_dynamic_quantization=True; "
                "the dynamic loader does not calibrate Linear.input_scale, so the "
                "static activation-scale path would read an uninitialized scale."
            )

        # The packed sequence is long enough to fuse gate/up and SwiGLU.
        fuse_bf16_gate_up_swiglu = quant_algo is None

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
        self.sharder = H3SequenceSharder(
            model_config.visual_gen_mapping,
            num_attention_heads=num_attention_heads,
        )

        video_patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        # Block-scaled GEMMs require aligned dimensions. Keep the checkpoint's
        # small FP32 media projections unquantized, including their output heads.
        if quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
            excluded = list(model_config.quant_config.exclude_modules or [])
            for name, input_dim, output_dim in (
                ("proj_in", video_patch_dim, hidden_size),
                ("audio_proj_in", audio_in_channels, hidden_size),
                ("proj_out", hidden_size, video_patch_dim),
                ("audio_proj_out", hidden_size, audio_in_channels),
            ):
                if input_dim % 128 or output_dim % 128:
                    if name not in excluded:
                        excluded.append(name)
            model_config.quant_config.exclude_modules = excluded
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
            "tensor_parallel_mode": None,
            "reduce_output": False,
        }
        bf16_linear_kwargs = {
            "dtype": model_config.torch_dtype,
            "mapping": model_config.mapping,
            "quant_config": model_config.quant_config,
            "skip_create_weights_in_init": model_config.skip_create_weights_in_init,
            "force_dynamic_quantization": model_config.force_dynamic_quantization,
            "tensor_parallel_mode": None,
            "reduce_output": False,
        }

        self.proj_in = Linear(video_patch_dim, hidden_size, bias=True, **fp32_linear_kwargs)
        self.audio_proj_in = Linear(audio_in_channels, hidden_size, bias=True, **fp32_linear_kwargs)
        self.context_embedder = Linear(text_dim, hidden_size, bias=True, **bf16_linear_kwargs)

        self.time_proj = Timesteps(
            num_channels=freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = TimestepEmbedding(
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
            fuse_bf16_gate_up_swiglu=fuse_bf16_gate_up_swiglu,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    time_embed_dim=time_embed_dim,
                    attention_cls=attention_cls,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    model_config=model_config,
                    layer_idx=layer_idx,
                    fuse_bf16_gate_up_swiglu=fuse_bf16_gate_up_swiglu,
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

        self.apply_quant_config_exclude_modules()

    def apply_quant_config_exclude_modules(self) -> None:
        quant_config = self.model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return

        for name, module in self.named_modules():
            if isinstance(module, Linear) and quant_config.is_module_excluded_from_quantization(
                name
            ):
                module.disable_quantization()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

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
        sequence_length: int,
        token_tags: torch.Tensor,
        timestep_indices: torch.Tensor,
        timestep: torch.Tensor,
    ) -> None:
        """Range-check the packed layout; the fused AdaLN kernels gather ``mod[idx]`` without bounds checks."""
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
        conditioning_timesteps: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: Optional[dict[str, object]] = None,
        return_dict: bool = True,
        static_context: Optional[MiniMaxH3StaticContext] = None,
        vsa_metadata: H3VSAMetadata | None = None,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        """Predict packed velocities using native conditioning and normalized attention time.

        ``conditioning_timesteps`` is the unique H3 time table for AdaLN;
        ``timestep`` is descending noise in [0, 1] for attention/graph scheduling.
        """
        del attention_kwargs
        if position_ids is None:
            if static_context is None:
                raise ValueError("position_ids is required when static_context is not provided.")
            sequence_length = static_context.sequence_length
        else:
            if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
                raise ValueError(
                    f"position_ids must have shape [sequence_length, 3], got {list(position_ids.shape)}."
                )
            sequence_length = position_ids.shape[0]
        self._validate_layout(sequence_length, token_tags, timestep_indices, conditioning_timesteps)

        if static_context is None:
            if encoder_hidden_states is None or position_ids is None:
                raise ValueError(
                    "encoder_hidden_states and position_ids are required to prepare static context."
                )
            static_context = self.prepare_static_context(
                encoder_hidden_states,
                position_ids,
            )

        if static_context.sequence_length != sequence_length:
            raise ValueError("static_context RoPE length must match the packed sequence length.")

        if static_context.text_embeds.shape[1] != text_indices.numel():
            raise ValueError("text_indices must contain one entry for every cached text embedding.")

        video_embeds = self.proj_in(hidden_states.to(self._projection_act_dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self._projection_act_dtype))
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

        temb = self.time_proj(conditioning_timesteps)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)
        rotary_emb = static_context.rotary_emb
        (
            packed_hidden_states,
            adaln_indices,
            timestep_indices_for_norm,
            rotary_emb,
            key_padding_mask,
            valid_seq_len,
        ) = shard_dit_inputs(
            packed_hidden_states,
            adaln_indices,
            timestep_indices,
            token_tags,
            rotary_emb,
            self.sharder,
            self._attn_backend,
            self._supports_key_padding_mask,
        )

        sparse_inputs = {} if vsa_metadata is None else {"vsa_metadata": vsa_metadata}
        for block in self.transformer_blocks:
            packed_hidden_states = block(
                packed_hidden_states,
                temb,
                adaln_indices,
                rotary_emb,
                key_padding_mask,
                timestep,
                valid_seq_len=valid_seq_len,
                **sparse_inputs,
            )

        packed_hidden_states = self.norm_out(
            packed_hidden_states,
            temb,
            timestep_indices_for_norm,
        )
        packed_hidden_states = self.sharder.gather(
            packed_hidden_states,
            dim=1,
            unpad_to=sequence_length,
        )
        video_output = self.proj_out(
            packed_hidden_states.to(self._projection_act_dtype)
        ).index_select(1, video_indices)
        audio_output = self.audio_proj_out(
            packed_hidden_states.to(self._projection_act_dtype)
        ).index_select(1, audio_indices)

        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(
            sample=video_output,
            audio_sample=audio_output,
        )

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        load_transformer_weights(self, weights)

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
