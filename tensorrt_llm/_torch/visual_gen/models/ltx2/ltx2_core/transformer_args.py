# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

# Orchestration logic that wires patchify, AdaLN, caption projection, and RoPE
# generation into a unified TransformerArgs dataclass for the transformer blocks.

from dataclasses import dataclass, replace

import torch

from .adaln import AdaLayerNormSingle
from .modality import Modality
from .rope import (
    LTXRopeType,
    _generate_freq_grid_np,
    _generate_freq_grid_pytorch,
    precompute_freqs_cis,
)
from .text_projection import PixArtAlphaTextProjection


@dataclass(frozen=True)
class TransformerArgs:
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: tuple[torch.Tensor, torch.Tensor]
    cross_positional_embeddings: tuple[torch.Tensor, torch.Tensor] | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    enabled: bool


class TransformerArgsPreprocessor:
    """Converts a Modality into TransformerArgs for transformer blocks.

    Handles: patchify projection, AdaLN timestep embedding,
    caption projection, attention mask preparation, and RoPE generation.
    """

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
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type

        # Cache for context/mask/PE — these depend only on text context and
        # positions which are constant across denoise steps.  Keyed by
        # id(modality.context) which is stable within a single generate() call.
        self._cached_context: torch.Tensor | None = None
        self._cached_mask: torch.Tensor | None = None
        self._cached_pe: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cache_key: int | None = None

    def _prepare_timestep(
        self,
        timestep: torch.Tensor,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = timestep * self.timestep_scale_multiplier
        timestep, embedded_timestep = self.adaln(timestep.flatten(), hidden_dtype=hidden_dtype)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        return timestep, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = x.shape[0]
        context = self.caption_projection(context.contiguous())
        context = context.view(batch_size, -1, x.shape[-1])
        return context, attention_mask

    def _prepare_attention_mask(
        self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype
    ) -> torch.Tensor | None:
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask
        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freq_grid_generator = (
            _generate_freq_grid_np if self.double_precision_rope else _generate_freq_grid_pytorch
        )
        return precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )

    def prepare(self, modality: Modality) -> TransformerArgs:
        x = self.patchify_proj(modality.latent.contiguous())
        timestep, embedded_timestep = self._prepare_timestep(
            modality.timesteps, x.shape[0], modality.latent.dtype
        )

        # Cache context projection, attention mask, and RoPE.
        # modality.context is the same tensor object across denoise steps
        # (pipeline passes the same video_embeds/audio_embeds every step).
        ctx_key = modality.context.data_ptr()
        if self._cache_key != ctx_key:
            context, attention_mask = self._prepare_context(
                modality.context, x, modality.context_mask
            )
            attention_mask = self._prepare_attention_mask(attention_mask, modality.latent.dtype)
            pe = self._prepare_positional_embeddings(
                positions=modality.positions,
                inner_dim=self.inner_dim,
                max_pos=self.max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                num_attention_heads=self.num_attention_heads,
                x_dtype=modality.latent.dtype,
            )
            self._cached_context = context
            self._cached_mask = attention_mask
            self._cached_pe = pe
            self._cache_key = ctx_key

        return TransformerArgs(
            x=x,
            context=self._cached_context,
            context_mask=self._cached_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=self._cached_pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
        )


class MultiModalTransformerArgsPreprocessor:
    """Extends TransformerArgsPreprocessor with cross-modal (AV) attention args."""

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
    ) -> None:
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
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
        self._cached_cross_pe: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cross_pe_key: int | None = None

    def prepare(self, modality: Modality) -> TransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality)

        pos_key = modality.positions.data_ptr()
        if self._cross_pe_key != pos_key:
            self._cached_cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
                positions=modality.positions[:, 0:1, :],
                inner_dim=self.audio_cross_attention_dim,
                max_pos=[self.cross_pe_max_pos],
                use_middle_indices_grid=True,
                num_attention_heads=self.simple_preprocessor.num_attention_heads,
                x_dtype=modality.latent.dtype,
            )
            self._cross_pe_key = pos_key

        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            timestep=modality.timesteps,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=modality.latent.dtype,
        )
        return replace(
            transformer_args,
            cross_positional_embeddings=self._cached_cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = timestep * timestep_scale_multiplier
        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.flatten(), hidden_dtype=hidden_dtype
        )
        scale_shift_timestep = scale_shift_timestep.view(
            batch_size, -1, scale_shift_timestep.shape[-1]
        )

        gate_noise_timestep, _ = self.cross_gate_adaln(
            timestep.flatten() * av_ca_factor, hidden_dtype=hidden_dtype
        )
        gate_noise_timestep = gate_noise_timestep.view(
            batch_size, -1, gate_noise_timestep.shape[-1]
        )

        return scale_shift_timestep, gate_noise_timestep
