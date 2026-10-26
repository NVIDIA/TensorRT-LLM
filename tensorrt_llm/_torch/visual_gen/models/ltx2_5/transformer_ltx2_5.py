# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2x
"""LTX-2.5 transformer extensions over the LTX-2.5 architecture."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modules.mlp import MLP

from ..ltx2.ltx2_core.modality import Modality
from ..ltx2.ltx2_core.transformer_args import TransformerArgs, TransformerArgsPreprocessor
from ..ltx2.transformer_ltx2 import TransformerConfig
from .transformer_base import _LTX25ModelBase, _LTX25TransformerBlockBase


class _LTX25ArgsPreprocessor(TransformerArgsPreprocessor):
    """Preserve reference latent strides at the BF16 patch projection boundary."""

    def __init__(self, *args, keyframes_embedding_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyframes_embedding_provider = keyframes_embedding_provider

    def prepare(
        self,
        modality: Modality,
        static_context: torch.Tensor,
        static_mask: torch.Tensor | None,
        static_pe: tuple[torch.Tensor, torch.Tensor],
        static_cross_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> TransformerArgs:
        # Refinement latents are channel-major views. Calling contiguous() here
        # changes the BF16 GEMM rounding across refinement steps.
        x = self.patchify_proj(modality.latent)
        if modality.keyframes_mask is not None and self.keyframes_embedding_provider is not None:
            embedding = self.keyframes_embedding_provider()
            if embedding is not None:
                x = x + (modality.keyframes_mask > 0).to(x.dtype) * embedding.to(x.dtype)
        timestep, embedded = self._prepare_timestep(
            modality.timesteps, x.shape[0], modality.latent.dtype
        )
        return TransformerArgs(
            x=x,
            context=static_context,
            context_mask=static_mask,
            timesteps=timestep,
            embedded_timestep=embedded,
            positional_embeddings=static_pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
        )


class LTX25TransformerBlock(_LTX25TransformerBlockBase):
    """LTX-2.5 block with LTX-2.5's bias-free video feed-forward network."""

    def __init__(self, *args, video=None, config=None, **kwargs) -> None:
        super().__init__(*args, video=video, config=config, **kwargs)
        if video is not None:
            self.ff = MLP(
                hidden_size=video.dim,
                intermediate_size=video.dim * 4,
                bias=False,
                activation=torch.nn.GELU(approximate="tanh"),
                dtype=config.torch_dtype if config else None,
                config=config,
                layer_idx=self.idx,
            )


class LTX25Model(_LTX25ModelBase):
    """LTX-2.5 model with keyframe markers and bias-free video FFNs."""

    def __init__(
        self,
        *args,
        use_keyframes_abs_pos_embedding: bool = True,
        **kwargs,
    ) -> None:
        self.use_keyframes_abs_pos_embedding = use_keyframes_abs_pos_embedding
        super().__init__(*args, **kwargs)

    def _init_video(self, in_channels, out_channels, caption_channels, norm_eps):
        super()._init_video(in_channels, out_channels, caption_channels, norm_eps)
        self.keyframes_abs_pos_embedding = (
            torch.nn.Parameter(torch.zeros(1, self.inner_dim))
            if self.use_keyframes_abs_pos_embedding
            else None
        )

    def _init_transformer_blocks(
        self,
        num_layers,
        attention_head_dim,
        cross_attention_dim,
        audio_attention_head_dim,
        audio_cross_attention_dim,
        norm_eps,
        apply_gated_attention,
    ):
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                LTX25TransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    config=self.model_config,
                )
                for idx in range(num_layers)
            ]
        )

    def _keyframes_embedding(self) -> torch.Tensor | None:
        return self.keyframes_abs_pos_embedding

    def _init_preprocessors(self, cross_pe_max_pos: int) -> None:
        super()._init_preprocessors(cross_pe_max_pos)
        for name in ("video_args_preprocessor", "audio_args_preprocessor"):
            preprocessor = getattr(self, name, None)
            if preprocessor is None:
                continue
            simple = getattr(preprocessor, "simple_preprocessor", preprocessor)
            replacement = _LTX25ArgsPreprocessor(
                patchify_proj=simple.patchify_proj,
                adaln=simple.adaln,
                caption_projection=simple.caption_projection,
                inner_dim=simple.inner_dim,
                max_pos=simple.max_pos,
                num_attention_heads=simple.num_attention_heads,
                use_middle_indices_grid=simple.use_middle_indices_grid,
                timestep_scale_multiplier=simple.timestep_scale_multiplier,
                double_precision_rope=simple.double_precision_rope,
                positional_embedding_theta=simple.positional_embedding_theta,
                rope_type=simple.rope_type,
                keyframes_embedding_provider=self._keyframes_embedding
                if name == "video_args_preprocessor"
                else None,
            )
            if simple is preprocessor:
                setattr(self, name, replacement)
            else:
                preprocessor.simple_preprocessor = replacement
