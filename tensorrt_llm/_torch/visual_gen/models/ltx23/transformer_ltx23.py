# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3 transformer using LTX-2 attention, MLP, preprocessing, and fused AdaLN ops.

Each stream has nine AdaLN slots: self-attention [0:3], FFN [3:6], and text
cross-attention [6:9]. Text K/V is modulated by sigma and projected each step.
Caption projection is handled by the feature extractor before the connectors.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.utils import SequenceSharder

from ..ltx2.ltx2_core.adaln import AdaLayerNormSingle
from ..ltx2.ltx2_core.utils_ltx2 import (
    apply_fused_gate_resid,
    apply_fused_gate_resid_rmsnorm_shift_scale,
    apply_fused_rmsnorm_shift_scale,
    apply_shift_scale,
    is_fused_adaln_supported_dim,
)
from ..ltx2.text_cache import TextCache
from ..ltx2.transformer_ltx2 import (
    AudioShardMode,
    BasicAVTransformerBlock,
    LTX2Attention,
    LTXModel,
    TransformerConfig,
)
from .ltx23_core.modality import LTX23Modality

_get_ada_table_ts_pairs = BasicAVTransformerBlock._get_ada_table_ts_pairs
_get_av_ca_ada_table_ts_pairs = BasicAVTransformerBlock._get_av_ca_ada_table_ts_pairs


class LTX23TransformerBlock(nn.Module):
    """Dual-stream LTX-2.3 block with sigma-dependent text cross-attention."""

    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type=None,
        norm_eps: float = 1e-6,
        config=None,
        stage2_ulysses_group=None,
        stage2_sharder=None,
    ):
        super().__init__()
        if stage2_ulysses_group is not None or stage2_sharder is not None:
            raise NotImplementedError("Two-stage topology is not supported for LTX-2.3.")
        self.idx = idx
        self.norm_eps = norm_eps
        vgm = config.visual_gen_mapping if config is not None else None
        self._sharder = SequenceSharder.from_vgm(vgm)
        self._fuse_adaln = (video is None or is_fused_adaln_supported_dim(video.dim)) and (
            audio is None or is_fused_adaln_supported_dim(audio.dim)
        )
        dense_config = config
        if config is not None and config.attention.quant_attention_config is not None:
            attention = config.attention.model_copy(update={"quant_attention_config": None})
            dense_config = config.model_copy(update={"attention": attention})

        def attn(
            name, query_dim, context_dim, kv, gated, model_config=dense_config, *, parallel=False
        ):
            return LTX2Attention(
                query_dim=query_dim,
                context_dim=context_dim,
                heads=kv.heads,
                dim_head=kv.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=gated,
                config=model_config,
                layer_idx=idx,
                module_name=f"transformer_blocks.{idx}.{name}",
                enable_sequence_parallel=parallel,
            )

        if video is not None:
            gated_v = video.apply_gated_attention
            # Quantized attention applies only to video self-attention; audio
            # head dimensions and cross-attention require the dense backend.
            self.attn1 = attn("attn1", video.dim, None, video, gated_v, config, parallel=True)
            self.attn2 = attn("attn2", video.dim, video.context_dim, video, gated_v)
            self.ff = BasicAVTransformerBlock._make_mlp(video, config, idx)
            self.scale_shift_table = nn.Parameter(torch.empty(9, video.dim))
            self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, video.dim))

        if audio is not None:
            gated_a = audio.apply_gated_attention
            self.audio_attn1 = attn("audio_attn1", audio.dim, None, audio, gated_a)
            self.audio_attn2 = attn("audio_attn2", audio.dim, audio.context_dim, audio, gated_a)
            self.audio_ff = BasicAVTransformerBlock._make_mlp(audio, config, idx)
            self.audio_scale_shift_table = nn.Parameter(torch.empty(9, audio.dim))
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty(2, audio.dim))

        if audio is not None and video is not None:
            self.audio_to_video_attn = attn(
                "audio_to_video_attn", video.dim, audio.dim, audio, video.apply_gated_attention
            )
            self.video_to_audio_attn = attn(
                "video_to_audio_attn",
                audio.dim,
                video.dim,
                audio,
                audio.apply_gated_attention,
                parallel=True,
            )
            self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.empty(5, video.dim))

    def _text_cross_attention(
        self,
        attn_input: torch.Tensor,
        context: torch.Tensor,
        attn: LTX2Attention,
        prompt_scale_shift_table: torch.Tensor,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Query is pre-modulated by the caller; K/V are shifted from prompt_timestep."""
        batch_size = attn_input.shape[0]
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None].to(
                device=attn_input.device, dtype=attn_input.dtype
            )
            + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)
        enc = apply_shift_scale(context, scale_kv, shift_kv)
        k, v = attn.project_kv(enc)
        return attn(attn_input, context=enc, pre_projected_kv=(k, v), timestep=timestep)

    def _self_attn_and_text_ca(
        self,
        x,
        stream,
        attn1,
        attn2,
        scale_shift_table,
        prompt_table,
        prompt_timestep,
    ):
        shift_msa, scale_msa, gate_msa = _get_ada_table_ts_pairs(
            scale_shift_table, x.shape[0], stream.timesteps, slice(0, 3)
        )
        shift_ca, scale_ca, gate_ca = _get_ada_table_ts_pairs(
            scale_shift_table, x.shape[0], stream.timesteps, slice(6, 9)
        )
        norm_x = apply_fused_rmsnorm_shift_scale(
            x,
            scale_msa[0],
            scale_msa[1],
            shift_msa[0],
            shift_msa[1],
            self.norm_eps,
            self._fuse_adaln,
        )
        msa = attn1(
            norm_x,
            pe=stream.positional_embeddings,
            timestep=stream.timesteps,
            key_padding_mask=stream.audio_padding_mask,
        )
        x, q_input = apply_fused_gate_resid_rmsnorm_shift_scale(
            x,
            msa,
            gate_msa[0],
            gate_msa[1],
            scale_ca[0],
            scale_ca[1],
            shift_ca[0],
            shift_ca[1],
            self.norm_eps,
            self._fuse_adaln,
        )
        ca = self._text_cross_attention(
            q_input, stream.context, attn2, prompt_table, stream.timesteps, prompt_timestep
        )
        return apply_fused_gate_resid(x, ca, gate_ca[0], gate_ca[1], self._fuse_adaln)

    def _ffn(self, x, stream, ff, scale_shift_table):
        shift, scale, gate = _get_ada_table_ts_pairs(
            scale_shift_table, x.shape[0], stream.timesteps, slice(3, 6)
        )
        scaled = apply_fused_rmsnorm_shift_scale(
            x,
            scale[0],
            scale[1],
            shift[0],
            shift[1],
            self.norm_eps,
            self._fuse_adaln,
        )
        return apply_fused_gate_resid(x, ff(scaled), gate[0], gate[1], self._fuse_adaln)

    def _apply_video_audio_cross_attention(
        self,
        query_x,
        query_pre,
        kv_pre,
        query_stream,
        kv_stream,
        query_table,
        kv_table,
        attn,
        is_a2v: bool,
    ):
        q_scale_a2v, q_shift_a2v, q_scale_v2a, q_shift_v2a, gate = _get_av_ca_ada_table_ts_pairs(
            query_table,
            query_x.shape[0],
            query_stream.cross_scale_shift_timestep,
            query_stream.cross_gate_timestep,
        )
        kv_scale_a2v, kv_shift_a2v, kv_scale_v2a, kv_shift_v2a, _ = _get_av_ca_ada_table_ts_pairs(
            kv_table,
            kv_pre.shape[0],
            kv_stream.cross_scale_shift_timestep,
            kv_stream.cross_gate_timestep,
        )
        if is_a2v:
            scale_q, shift_q = q_scale_a2v, q_shift_a2v
            scale_kv, shift_kv = kv_scale_a2v, kv_shift_a2v
        else:
            scale_q, shift_q = q_scale_v2a, q_shift_v2a
            scale_kv, shift_kv = kv_scale_v2a, kv_shift_v2a

        q_norm = apply_fused_rmsnorm_shift_scale(
            query_pre,
            scale_q[0],
            scale_q[1],
            shift_q[0],
            shift_q[1],
            self.norm_eps,
            self._fuse_adaln,
        )
        kv_norm = apply_fused_rmsnorm_shift_scale(
            kv_pre,
            scale_kv[0],
            scale_kv[1],
            shift_kv[0],
            shift_kv[1],
            self.norm_eps,
            self._fuse_adaln,
        )
        k, v = attn.project_kv(kv_norm, pe=kv_stream.cross_positional_embeddings)
        query_pe = query_stream.cross_positional_embeddings
        query_timestep = query_stream.timesteps
        if not is_a2v and self._sharder.is_active:
            # Audio stays replicated, as in LTX-2's conditional audio path.
            # Only the v2a query is sharded to match the video K/V Ulysses layout.
            query_length = q_norm.shape[1]
            q_norm = self._sharder.shard(q_norm, dim=1)
            query_timestep = self._sharder.shard(
                query_timestep, dim=1, expected_seq_len=query_length
            )
            if query_pe is not None:
                pe_dim = 0 if query_pe[0].ndim == 2 else 1
                query_pe = tuple(self._sharder.shard(pe, dim=pe_dim) for pe in query_pe)
            if not attn.is_ulysses:
                k = self._sharder.gather(k, dim=1)
                v = self._sharder.gather(v, dim=1)
        out = attn(
            q_norm,
            pre_projected_kv=(k, v),
            pe=query_pe,
            timestep=query_timestep,
            key_padding_mask=kv_stream.audio_padding_mask if is_a2v else None,
        )
        if not is_a2v and self._sharder.is_active:
            out = self._sharder.gather(out, dim=1)
        return apply_fused_gate_resid(query_x, out, gate[0], gate[1], self._fuse_adaln)

    def forward(
        self,
        video,
        audio,
        video_prompt_timestep: torch.Tensor | None = None,
        audio_prompt_timestep: torch.Tensor | None = None,
    ):
        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None
        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0
        run_a2v = run_vx and audio is not None and ax is not None and ax.numel() > 0
        run_v2a = run_ax and video is not None and vx is not None and vx.numel() > 0

        if run_vx:
            vx = self._self_attn_and_text_ca(
                vx,
                video,
                self.attn1,
                self.attn2,
                self.scale_shift_table,
                self.prompt_scale_shift_table,
                video_prompt_timestep,
            )
        if run_ax:
            ax = self._self_attn_and_text_ca(
                ax,
                audio,
                self.audio_attn1,
                self.audio_attn2,
                self.audio_scale_shift_table,
                self.audio_prompt_scale_shift_table,
                audio_prompt_timestep,
            )

        if run_a2v or run_v2a:
            vx_pre, ax_pre = vx, ax
            if run_a2v:
                vx = self._apply_video_audio_cross_attention(
                    vx,
                    vx_pre,
                    ax_pre,
                    video,
                    audio,
                    self.scale_shift_table_a2v_ca_video,
                    self.scale_shift_table_a2v_ca_audio,
                    self.audio_to_video_attn,
                    is_a2v=True,
                )
            if run_v2a:
                ax = self._apply_video_audio_cross_attention(
                    ax,
                    ax_pre,
                    vx_pre,
                    audio,
                    video,
                    self.scale_shift_table_a2v_ca_audio,
                    self.scale_shift_table_a2v_ca_video,
                    self.video_to_audio_attn,
                    is_a2v=False,
                )

        if run_vx:
            vx = self._ffn(vx, video, self.ff, self.scale_shift_table)
        if run_ax:
            ax = self._ffn(ax, audio, self.audio_ff, self.audio_scale_shift_table)

        return (
            replace(video, x=vx) if video is not None else None,
            replace(audio, x=ax) if audio is not None else None,
        )


class LTX23Model(LTXModel):
    """LTX-2.3 transformer: 9-slot AdaLN, prompt AdaLN, Identity caption projection."""

    _transformer_block_class = LTX23TransformerBlock
    _cache_text_kv = False

    @property
    def _audio_shard_mode(self) -> AudioShardMode:
        """Replicate the short audio stream and shard its v2a attention queries."""
        return AudioShardMode.CONDITIONAL if self._active_sharder.is_active else AudioShardMode.NONE

    def _init_stream(self, dim, in_channels, out_channels, norm_eps, prefix=""):
        setattr(self, f"{prefix}patchify_proj", self._make_linear(in_channels, dim))
        setattr(
            self,
            f"{prefix}adaln_single",
            AdaLayerNormSingle(dim, embedding_coefficient=9, make_linear=self._make_linear),
        )
        setattr(
            self,
            f"{prefix}prompt_adaln_single",
            AdaLayerNormSingle(dim, embedding_coefficient=2, make_linear=self._make_linear),
        )
        setattr(self, f"{prefix}caption_projection", nn.Identity())
        setattr(self, f"{prefix}scale_shift_table", nn.Parameter(torch.empty(2, dim)))
        setattr(
            self,
            f"{prefix}norm_out",
            nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps),
        )
        setattr(self, f"{prefix}proj_out", self._make_linear(dim, out_channels))

    def _init_video(self, in_channels, out_channels, caption_channels, norm_eps):
        self._init_stream(self.inner_dim, in_channels, out_channels, norm_eps)

    def _init_audio(self, in_channels, out_channels, caption_channels, norm_eps):
        self._init_stream(
            self.audio_inner_dim, in_channels, out_channels, norm_eps, prefix="audio_"
        )

    def _uses_cache_dit(self) -> bool:
        # The LTX-2 Cache-DiT wrapper does not accept prompt timestep inputs.
        return False

    def _compute_prompt_timestep(self, adaln, sigma, batch_size, dtype):
        """prompt_timestep from the global sigma (drives context K/V modulation)."""
        sigma_scaled = sigma * self.timestep_scale_multiplier
        prompt_ts, _ = adaln(sigma_scaled.flatten(), hidden_dtype=dtype)
        return prompt_ts.view(batch_size, -1, prompt_ts.shape[-1])

    def _prepare_stream(self, modality, text_cache, is_audio: bool):
        if modality is None:
            return None, None
        if is_audio:
            preprocessor = self.audio_args_preprocessor
            context, mask = text_cache.audio_context, text_cache.audio_mask
            pe, cross_pe = text_cache.audio_pe, text_cache.audio_cross_pe
            adaln = self.audio_prompt_adaln_single
        else:
            preprocessor = self.video_args_preprocessor
            context, mask = text_cache.video_context, text_cache.video_mask
            pe, cross_pe = text_cache.video_pe, text_cache.video_cross_pe
            adaln = self.prompt_adaln_single
        args = preprocessor.prepare(modality, context, mask, pe, cross_pe)
        cross_sigma = modality.cross_modality_sigma
        if cross_sigma is not None:
            batch_size = args.x.shape[0]
            if cross_sigma.ndim > 1 or cross_sigma.numel() != batch_size:
                raise ValueError(
                    "cross modality sigma must be scalar per batch: "
                    f"got shape {tuple(cross_sigma.shape)} for batch size {batch_size}"
                )
            gate, _ = preprocessor.cross_gate_adaln(
                (cross_sigma * preprocessor.av_ca_timestep_scale_multiplier).flatten(),
                hidden_dtype=modality.latent.dtype,
            )
            gate = gate.view(batch_size, -1, gate.shape[-1])
            if gate.shape[1] != args.cross_scale_shift_timestep.shape[1]:
                gate = gate.expand(-1, args.cross_scale_shift_timestep.shape[1], -1).contiguous()
            args = replace(args, cross_gate_timestep=gate)
        prompt_ts = self._compute_prompt_timestep(
            adaln, modality.sigma, modality.latent.shape[0], modality.latent.dtype
        )
        return args, prompt_ts

    def _stream_output(self, args, is_audio: bool):
        if args is None:
            return None
        if is_audio:
            table, norm, proj = (
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
            )
        else:
            table, norm, proj = self.scale_shift_table, self.norm_out, self.proj_out
        return self._process_output(table, norm, proj, args.x, args.embedded_timestep)

    def forward(
        self,
        video: LTX23Modality | None,
        audio: LTX23Modality | None,
        *,
        text_cache: TextCache,
        timestep: torch.Tensor | None = None,
        step_index=None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run the complete modality batch without splitting CFG branches."""
        audio_length = None
        audio_padding_mask = None
        if audio is not None and self._audio_pad > 0:
            audio_length = audio.latent.shape[1]
            audio = self._pad_modality_audio(audio, self._audio_pad)
            audio_padding_mask = torch.arange(audio.latent.shape[1], device=audio.latent.device)
            audio_padding_mask = (audio_padding_mask < audio_length).expand(
                audio.latent.shape[0], -1
            )
        video_args, video_prompt_ts = self._prepare_stream(video, text_cache, is_audio=False)
        audio_args, audio_prompt_ts = self._prepare_stream(audio, text_cache, is_audio=True)
        if video_args is not None and self._active_sharder.is_active:
            video_args = self._shard_transformer_args(video_args)
        if audio_args is not None and audio_padding_mask is not None:
            audio_args = replace(audio_args, audio_padding_mask=audio_padding_mask)

        for block in self.transformer_blocks:
            video_args, audio_args = block(
                video_args,
                audio_args,
                video_prompt_timestep=video_prompt_ts,
                audio_prompt_timestep=audio_prompt_ts,
            )

        video_output = self._stream_output(video_args, is_audio=False)
        audio_output = self._stream_output(audio_args, is_audio=True)
        if video_output is not None and self._active_sharder.is_active:
            video_output = self._gather_sequence(video_output)
        if audio_output is not None and audio_length is not None:
            audio_output = audio_output[:, :audio_length]
        return video_output, audio_output
