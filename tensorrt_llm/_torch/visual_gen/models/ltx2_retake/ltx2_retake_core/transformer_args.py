# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""Retake extensions to the shared LTX-2 transformer argument preprocessing."""

from dataclasses import dataclass, replace

import torch

from ...ltx2.ltx2_core.adaln import AdaLayerNormSingle
from ...ltx2.ltx2_core.rope import LTXRopeType
from ...ltx2.ltx2_core.text_projection import PixArtAlphaTextProjection
from ...ltx2.ltx2_core.transformer_args import (
    MultiModalTransformerArgsPreprocessor as LTX2MultiModalTransformerArgsPreprocessor,
)
from ...ltx2.ltx2_core.transformer_args import TransformerArgs as LTX2TransformerArgs
from ...ltx2.ltx2_core.transformer_args import (
    TransformerArgsPreprocessor as LTX2TransformerArgsPreprocessor,
)
from .modality import Modality


@dataclass(frozen=True)
class TransformerArgs(LTX2TransformerArgs):
    """Shared transformer inputs plus LTX-2.3 text-cross-attention modulation."""

    prompt_timestep: torch.Tensor | None = None


class TransformerArgsPreprocessor(LTX2TransformerArgsPreprocessor):
    def __init__(
        self,
        patchify_proj: torch.nn.Module,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        prompt_adaln: AdaLayerNormSingle | None = None,
    ) -> None:
        super().__init__(
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
        )
        self.prompt_adaln = prompt_adaln

    def prepare(
        self,
        modality: Modality,
        static_context: torch.Tensor,
        static_mask: torch.Tensor | None,
        static_pe: tuple[torch.Tensor, torch.Tensor],
        static_cross_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> TransformerArgs:
        args = super().prepare(
            modality,
            static_context=static_context,
            static_mask=static_mask,
            static_pe=static_pe,
            static_cross_pe=static_cross_pe,
        )
        prompt_timestep = None
        if self.prompt_adaln is not None:
            if modality.sigma is None:
                raise ValueError("cross-attention AdaLN requires a per-batch modality sigma")
            prompt_timestep, _ = self.prompt_adaln(
                modality.sigma * self.timestep_scale_multiplier,
                hidden_dtype=modality.latent.dtype,
            )
            prompt_timestep = prompt_timestep.unsqueeze(1)
        return TransformerArgs(**vars(args), prompt_timestep=prompt_timestep)


class MultiModalTransformerArgsPreprocessor(LTX2MultiModalTransformerArgsPreprocessor):
    def __init__(
        self,
        patchify_proj: torch.nn.Module,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
        prompt_adaln: AdaLayerNormSingle | None = None,
    ) -> None:
        super().__init__(
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            cross_scale_shift_adaln=cross_scale_shift_adaln,
            cross_gate_adaln=cross_gate_adaln,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            cross_pe_max_pos=cross_pe_max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            audio_cross_attention_dim=audio_cross_attention_dim,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
        )
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            prompt_adaln=prompt_adaln,
        )

    def prepare(
        self,
        modality: Modality,
        static_context: torch.Tensor,
        static_mask: torch.Tensor | None,
        static_pe: tuple[torch.Tensor, torch.Tensor],
        static_cross_pe: tuple[torch.Tensor, torch.Tensor],
    ) -> TransformerArgs:
        args = self.simple_preprocessor.prepare(
            modality,
            static_context=static_context,
            static_mask=static_mask,
            static_pe=static_pe,
        )
        cross_sigma = modality.cross_modality_sigma
        if cross_sigma is None:
            return args
        batch_size = args.x.shape[0]
        if cross_sigma.ndim > 1 or cross_sigma.numel() != batch_size:
            raise ValueError(
                "cross modality sigma must be scalar per batch: "
                f"got shape {tuple(cross_sigma.shape)} for batch size {batch_size}"
            )
        scale_shift, gate = self._prepare_cross_attention_timestep(
            modality_timesteps=modality.timesteps,
            cross_modality_sigma=cross_sigma,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=batch_size,
            hidden_dtype=modality.latent.dtype,
        )
        return replace(
            args,
            cross_positional_embeddings=static_cross_pe,
            cross_scale_shift_timestep=scale_shift,
            cross_gate_timestep=gate,
        )

    def _prepare_cross_attention_timestep(
        self,
        modality_timesteps: torch.Tensor,
        cross_modality_sigma: torch.Tensor,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modality_timesteps = modality_timesteps * timestep_scale_multiplier
        scale_shift, _ = self.cross_scale_shift_adaln(
            modality_timesteps.flatten(), hidden_dtype=hidden_dtype
        )
        scale_shift = scale_shift.view(batch_size, -1, scale_shift.shape[-1])

        gate_scale = self.av_ca_timestep_scale_multiplier
        gate, _ = self.cross_gate_adaln(
            (cross_modality_sigma * gate_scale).flatten(), hidden_dtype=hidden_dtype
        )
        gate = gate.view(batch_size, -1, gate.shape[-1])
        if gate.shape[1] != scale_shift.shape[1]:
            gate = gate.expand(-1, scale_shift.shape[1], -1).contiguous()
        return scale_shift, gate
