# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

"""Retake-specific extensions to the native LTX-2 transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from ..ltx2.ltx2_core.adaln import AdaLayerNormSingle
from ..ltx2.ltx2_core.rope import LTXRopeType
from ..ltx2.transformer_ltx2 import BasicAVTransformerBlock as LTX2BasicAVTransformerBlock
from ..ltx2.transformer_ltx2 import LTX2Attention as LTX2AttentionBase
from ..ltx2.transformer_ltx2 import LTX2CacheDiTPattern0BlockWrapper, LTXModelType
from ..ltx2.transformer_ltx2 import LTXModel as LTX2ModelBase
from ..ltx2.transformer_ltx2 import TransformerConfig as LTX2TransformerConfig
from .ltx2_retake_core.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


@dataclass
class TransformerConfig(LTX2TransformerConfig):
    cross_attention_adaln: bool = False


class BasicAVTransformerBlock(LTX2BasicAVTransformerBlock):
    """LTX-2.3 block extended with timestep-conditioned text cross-attention AdaLN."""

    def _init_video_modules(
        self,
        cfg: TransformerConfig,
        rope_type: LTXRopeType,
        eps: float,
        model_config: "DiffusionModelConfig | None",
        idx: int,
    ) -> None:
        super()._init_video_modules(cfg, rope_type, eps, model_config, idx)
        if cfg.cross_attention_adaln:
            self.scale_shift_table = nn.Parameter(torch.empty(9, cfg.dim))
            self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, cfg.dim))

    def _init_audio_modules(
        self,
        cfg: TransformerConfig,
        rope_type: LTXRopeType,
        eps: float,
        model_config: "DiffusionModelConfig",
        idx: int,
    ) -> None:
        super()._init_audio_modules(cfg, rope_type, eps, model_config, idx)
        if cfg.cross_attention_adaln:
            self.audio_scale_shift_table = nn.Parameter(torch.empty(9, cfg.dim))
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty(2, cfg.dim))

    def _run_text_cross_adaln(
        self,
        query: torch.Tensor,
        args: TransformerArgs,
        attention: LTX2AttentionBase,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor,
    ) -> torch.Tensor:
        """Apply timestep-conditioned query/context modulation and gate text cross-attention."""
        if args.prompt_timestep is None:
            raise ValueError("Text cross-attention AdaLN requires prompt_timestep")
        shift_pair, scale_pair, gate_pair = self._get_ada_table_ts_pairs(
            scale_shift_table, query.shape[0], args.timesteps, slice(6, 9)
        )
        query_dtype = query.dtype
        shift_q = (shift_pair[0] + shift_pair[1]).to(query_dtype)
        scale_q = (scale_pair[0] + scale_pair[1]).to(query_dtype)
        gate = (gate_pair[0] + gate_pair[1]).to(query_dtype)
        query = query * (1 + scale_q) + shift_q

        batch_size = query.shape[0]
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None].to(dtype=query_dtype)
            + args.prompt_timestep.reshape(batch_size, args.prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)
        context_dtype = args.context.dtype
        context = args.context * (1 + scale_kv.to(context_dtype)) + shift_kv.to(context_dtype)
        text_kv = attention.project_kv(context, pe=None)
        return (
            attention(
                query,
                context=context,
                pre_projected_kv=text_kv,
                timestep=args.timesteps,
            )
            * gate
        )

    def _run_text_cross_attention(
        self,
        query: torch.Tensor,
        args: TransformerArgs,
        attention: LTX2AttentionBase,
        text_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if args.prompt_timestep is None:
            return super()._run_text_cross_attention(query, args, attention, text_kv)
        if attention is self.attn2:
            scale_shift_table = self.scale_shift_table
            prompt_scale_shift_table = self.prompt_scale_shift_table
        elif attention is self.audio_attn2:
            scale_shift_table = self.audio_scale_shift_table
            prompt_scale_shift_table = self.audio_prompt_scale_shift_table
        else:
            raise ValueError("Cross-attention AdaLN received an unsupported attention module")
        return self._run_text_cross_adaln(
            query,
            args,
            attention,
            scale_shift_table,
            prompt_scale_shift_table,
        )


class LTXModel(LTX2ModelBase):
    """Native LTX-2 transformer with the retake checkpoint extensions."""

    def __init__(
        self,
        *,
        cross_attention_adaln: bool = False,
        **kwargs: Any,
    ) -> None:
        self.cross_attention_adaln = cross_attention_adaln
        super().__init__(
            model_type=LTXModelType.AudioVideo,
            **kwargs,
        )

    def _init_video(
        self, in_channels: int, out_channels: int, caption_channels: int, norm_eps: float
    ) -> None:
        super()._init_video(in_channels, out_channels, caption_channels, norm_eps)
        self.prompt_adaln_single = (
            AdaLayerNormSingle(
                self.inner_dim,
                embedding_coefficient=2,
                make_linear=self._make_linear,
            )
            if self.cross_attention_adaln
            else None
        )
        if self.cross_attention_adaln:
            self.adaln_single = AdaLayerNormSingle(
                self.inner_dim,
                embedding_coefficient=9,
                make_linear=self._make_linear,
            )

    def _init_audio(
        self, in_channels: int, out_channels: int, caption_channels: int, norm_eps: float
    ) -> None:
        super()._init_audio(in_channels, out_channels, caption_channels, norm_eps)
        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(
                self.audio_inner_dim,
                embedding_coefficient=2,
                make_linear=self._make_linear,
            )
            if self.cross_attention_adaln
            else None
        )
        if self.cross_attention_adaln:
            self.audio_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim,
                embedding_coefficient=9,
                make_linear=self._make_linear,
            )

    def _init_preprocessors(self, cross_pe_max_pos: int | None) -> None:
        if cross_pe_max_pos is None:
            raise ValueError("LTX-2 retake requires audio-video cross-attention positions.")
        self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
            patchify_proj=self.patchify_proj,
            adaln=self.adaln_single,
            caption_projection=self.caption_projection,
            cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
            cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
            inner_dim=self.inner_dim,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            cross_pe_max_pos=cross_pe_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            audio_cross_attention_dim=self.audio_cross_attention_dim,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            double_precision_rope=self.double_precision_rope,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
            av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            prompt_adaln=self.prompt_adaln_single,
        )
        self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
            patchify_proj=self.audio_patchify_proj,
            adaln=self.audio_adaln_single,
            caption_projection=self.audio_caption_projection,
            cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
            cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
            inner_dim=self.audio_inner_dim,
            max_pos=self.audio_positional_embedding_max_pos,
            num_attention_heads=self.audio_num_attention_heads,
            cross_pe_max_pos=cross_pe_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            audio_cross_attention_dim=self.audio_cross_attention_dim,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            double_precision_rope=self.double_precision_rope,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
            av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            prompt_adaln=self.audio_prompt_adaln_single,
        )

    def _prepare_text_kv_cache(
        self, context: torch.Tensor, *, audio: bool
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        if self.cross_attention_adaln:
            return None
        return super()._prepare_text_kv_cache(context, audio=audio)

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        apply_gated_attention: bool,
    ) -> None:
        video_config = TransformerConfig(
            dim=self.inner_dim,
            heads=self.num_attention_heads,
            d_head=attention_head_dim,
            context_dim=cross_attention_dim,
            apply_gated_attention=apply_gated_attention,
            cross_attention_adaln=self.cross_attention_adaln,
        )
        audio_config = TransformerConfig(
            dim=self.audio_inner_dim,
            heads=self.audio_num_attention_heads,
            d_head=audio_attention_head_dim,
            context_dim=audio_cross_attention_dim,
            apply_gated_attention=apply_gated_attention,
            cross_attention_adaln=self.cross_attention_adaln,
        )
        blocks: list[nn.Module] = [
            BasicAVTransformerBlock(
                idx=idx,
                video=video_config,
                audio=audio_config,
                rope_type=self.rope_type,
                norm_eps=norm_eps,
                config=self.model_config,
                stage2_ulysses_group=(
                    self._stage2_groups.ulysses_group if self._has_stage2 else None
                ),
                stage2_sharder=self._sharder_s2 if self._has_stage2 else None,
            )
            for idx in range(num_layers)
        ]
        if self._uses_cache_dit():
            blocks = [
                LTX2CacheDiTPattern0BlockWrapper(block, parent=self)
                for block in blocks  # type: ignore[misc]
            ]
        self.transformer_blocks = nn.ModuleList(blocks)
