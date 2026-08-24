# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3 transformer.

Ported from LTX-2 (LTXModel / BasicAVTransformerBlock) with these changes:

- 9-slot per-block AdaLN (MSA [0:3], MLP [3:6], text-cross-attn [6:9]) plus a
  per-block prompt_scale_shift_table [2, dim] that modulates the text context
  K/V from a sigma-derived prompt_timestep.
- adaln_single embedding_coefficient 6 -> 9, and a new prompt_adaln_single
  (coeff 2) that produces prompt_timestep from sigma.
- caption_projection becomes nn.Identity, since LTX-2.3 projects to inner_dim in
  the text_embedding_projection feature extractor before the connector.
- Text K/V is projected per denoise step from the prompt-modulated context,
  rather than cached per block as in LTX-2.
- Audio/video cross-attention is unchanged from LTX-2 and reused as-is.

Conditioning runs through the same fused AdaLN ops as LTX-2: the text
cross-attention query shift/scale (slots 6, 7) and output gate (slot 8) fold into
the surrounding gate-residual kernels instead of taking a standalone pass.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from ..ltx2.ltx2_core.adaln import AdaLayerNormSingle
from ..ltx2.ltx2_core.utils_ltx2 import (
    apply_fused_gate_resid,
    apply_fused_gate_resid_rmsnorm_shift_scale,
    apply_fused_rmsnorm_shift_scale,
    apply_shift_scale,
    is_fused_adaln_supported_dim,
)
from ..ltx2.transformer_ltx2 import (
    BasicAVTransformerBlock,
    LTX2Attention,
    LTXModel,
    TransformerConfig,
)
from .text_conditioning_ltx23 import LTX23TextConditioning

_get_ada_table_ts_pairs = BasicAVTransformerBlock._get_ada_table_ts_pairs
_get_av_ca_ada_table_ts_pairs = BasicAVTransformerBlock._get_av_ca_ada_table_ts_pairs
_make_mlp = BasicAVTransformerBlock._make_mlp


class LTX23TransformerBlock(nn.Module):
    """Dual-stream (audio/video) LTX-2.3 block, eager reference-parity forward."""

    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type=None,
        norm_eps: float = 1e-6,
        config=None,
    ):
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps
        self._fuse_adaln = (video is None or is_fused_adaln_supported_dim(video.dim)) and (
            audio is None or is_fused_adaln_supported_dim(audio.dim)
        )

        def attn(name, query_dim, context_dim, kv, gated):
            return LTX2Attention(
                query_dim=query_dim,
                context_dim=context_dim,
                heads=kv.heads,
                dim_head=kv.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=gated,
                config=config,
                layer_idx=idx,
                module_name=f"transformer_blocks.{idx}.{name}",
                enable_sequence_parallel=False,
            )

        if video is not None:
            gated_v = video.apply_gated_attention
            self.attn1 = attn("attn1", video.dim, None, video, gated_v)
            self.attn2 = attn("attn2", video.dim, video.context_dim, video, gated_v)
            self.ff = _make_mlp(video, config, idx)
            self.scale_shift_table = nn.Parameter(torch.empty(9, video.dim))
            self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, video.dim))

        if audio is not None:
            gated_a = audio.apply_gated_attention
            self.audio_attn1 = attn("audio_attn1", audio.dim, None, audio, gated_a)
            self.audio_attn2 = attn("audio_attn2", audio.dim, audio.context_dim, audio, gated_a)
            self.audio_ff = _make_mlp(audio, config, idx)
            self.audio_scale_shift_table = nn.Parameter(torch.empty(9, audio.dim))
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty(2, audio.dim))

        if audio is not None and video is not None:
            self.audio_to_video_attn = attn(
                "audio_to_video_attn", video.dim, audio.dim, audio, video.apply_gated_attention
            )
            self.video_to_audio_attn = attn(
                "video_to_audio_attn", audio.dim, video.dim, audio, audio.apply_gated_attention
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
        msa = attn1(norm_x, pe=stream.positional_embeddings, timestep=stream.timesteps)
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
                scale_v_a2v, shift_v_a2v, _, _, gate_a2v = _get_av_ca_ada_table_ts_pairs(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                )
                a2v_vx = apply_fused_rmsnorm_shift_scale(
                    vx_pre,
                    scale_v_a2v[0],
                    scale_v_a2v[1],
                    shift_v_a2v[0],
                    shift_v_a2v[1],
                    self.norm_eps,
                    self._fuse_adaln,
                )
                scale_a_a2v, shift_a_a2v, _, _, _ = _get_av_ca_ada_table_ts_pairs(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                )
                a2v_ax = apply_fused_rmsnorm_shift_scale(
                    ax_pre,
                    scale_a_a2v[0],
                    scale_a_a2v[1],
                    shift_a_a2v[0],
                    shift_a_a2v[1],
                    self.norm_eps,
                    self._fuse_adaln,
                )
                k_a2v, v_a2v = self.audio_to_video_attn.project_kv(
                    a2v_ax, pe=audio.cross_positional_embeddings
                )
                a2v_out = self.audio_to_video_attn(
                    a2v_vx,
                    pre_projected_kv=(k_a2v, v_a2v),
                    pe=video.cross_positional_embeddings,
                    timestep=video.timesteps,
                )
                vx = apply_fused_gate_resid(vx, a2v_out, gate_a2v[0], gate_a2v[1], self._fuse_adaln)

            if run_v2a:
                _, _, scale_a_v2a, shift_a_v2a, gate_v2a = _get_av_ca_ada_table_ts_pairs(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                )
                v2a_ax = apply_fused_rmsnorm_shift_scale(
                    ax_pre,
                    scale_a_v2a[0],
                    scale_a_v2a[1],
                    shift_a_v2a[0],
                    shift_a_v2a[1],
                    self.norm_eps,
                    self._fuse_adaln,
                )
                _, _, scale_v_v2a, shift_v_v2a, _ = _get_av_ca_ada_table_ts_pairs(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                )
                v2a_vx = apply_fused_rmsnorm_shift_scale(
                    vx_pre,
                    scale_v_v2a[0],
                    scale_v_v2a[1],
                    shift_v_v2a[0],
                    shift_v_v2a[1],
                    self.norm_eps,
                    self._fuse_adaln,
                )
                k_v2a, v_v2a = self.video_to_audio_attn.project_kv(
                    v2a_vx, pe=video.cross_positional_embeddings
                )
                v2a_out = self.video_to_audio_attn(
                    v2a_ax,
                    pre_projected_kv=(k_v2a, v_v2a),
                    pe=audio.cross_positional_embeddings,
                    timestep=audio.timesteps,
                )
                ax = apply_fused_gate_resid(ax, v2a_out, gate_v2a[0], gate_v2a[1], self._fuse_adaln)

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

    def _init_video(self, in_channels, out_channels, caption_channels, norm_eps):
        self.patchify_proj = self._make_linear(in_channels, self.inner_dim)
        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, embedding_coefficient=9, make_linear=self._make_linear
        )
        self.prompt_adaln_single = AdaLayerNormSingle(
            self.inner_dim, embedding_coefficient=2, make_linear=self._make_linear
        )
        self.caption_projection = nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = self._make_linear(self.inner_dim, out_channels)

    def _init_audio(self, in_channels, out_channels, caption_channels, norm_eps):
        self.audio_patchify_proj = self._make_linear(in_channels, self.audio_inner_dim)
        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim, embedding_coefficient=9, make_linear=self._make_linear
        )
        self.audio_prompt_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim, embedding_coefficient=2, make_linear=self._make_linear
        )
        self.audio_caption_projection = nn.Identity()
        self.audio_scale_shift_table = nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = nn.LayerNorm(
            self.audio_inner_dim, elementwise_affine=False, eps=norm_eps
        )
        self.audio_proj_out = self._make_linear(self.audio_inner_dim, out_channels)

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
        self.transformer_blocks = nn.ModuleList(
            [
                LTX23TransformerBlock(
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

    def _compute_prompt_timestep(self, adaln, sigma, batch_size, dtype):
        """prompt_timestep from the global sigma (drives context K/V modulation)."""
        sigma_scaled = sigma * self.timestep_scale_multiplier
        prompt_ts, _ = adaln(sigma_scaled.flatten(), hidden_dtype=dtype)
        return prompt_ts.view(batch_size, -1, prompt_ts.shape[-1])

    def prepare_text_cache(
        self,
        *,
        video_context: torch.Tensor | None = None,
        video_context_mask: torch.Tensor | None = None,
        video_positions: torch.Tensor | None = None,
        audio_context: torch.Tensor | None = None,
        audio_context_mask: torch.Tensor | None = None,
        audio_positions: torch.Tensor | None = None,
        dtype: torch.dtype,
    ) -> LTX23TextConditioning:
        """Step-invariant text conditioning; K/V is not cached per block."""
        out = LTX23TextConditioning()

        if video_context is not None:
            v_ctx, v_mask, v_pe, v_cross_pe = self.video_args_preprocessor.prepare_text_cache(
                video_context, video_context_mask, video_positions, dtype
            )
            out.video_context = v_ctx
            out.video_mask = v_mask

        if audio_context is not None:
            a_ctx, a_mask, a_pe, a_cross_pe = self.audio_args_preprocessor.prepare_text_cache(
                audio_context, audio_context_mask, audio_positions, dtype
            )
            if self._audio_pad > 0:
                a_pe = self._pad_pe(a_pe, self._audio_pad, seq_dim=1)
                a_cross_pe = self._pad_pe(a_cross_pe, self._audio_pad, seq_dim=1)
            out.audio_context = a_ctx
            out.audio_mask = a_mask
        else:
            a_pe = a_cross_pe = None

        # One-time PE reshape/shard so the denoise loop has no PE work.
        fuse_video = self.transformer_blocks[0].attn1.fuse_qk_norm_rope
        fuse_audio = (
            self.transformer_blocks[0].audio_attn1.fuse_qk_norm_rope
            if hasattr(self.transformer_blocks[0], "audio_attn1")
            else True
        )
        if video_context is not None:
            out.video_pe = self._make_pe_local(v_pe, is_audio=False, fuse=fuse_video)
            out.video_cross_pe = self._make_pe_local(v_cross_pe, is_audio=False, fuse=fuse_video)
        if audio_context is not None:
            out.audio_pe = self._make_pe_local(a_pe, is_audio=True, fuse=fuse_audio)
            out.audio_cross_pe = self._make_pe_local(a_cross_pe, is_audio=True, fuse=fuse_audio)
        return out

    def forward(
        self,
        video,
        audio,
        *,
        text_cache: LTX23TextConditioning,
        timestep: torch.Tensor | None = None,
        step_index=None,
    ):
        video_args = (
            self.video_args_preprocessor.prepare(
                video,
                text_cache.video_context,
                text_cache.video_mask,
                text_cache.video_pe,
                text_cache.video_cross_pe,
            )
            if video is not None
            else None
        )
        audio_args = (
            self.audio_args_preprocessor.prepare(
                audio,
                text_cache.audio_context,
                text_cache.audio_mask,
                text_cache.audio_pe,
                text_cache.audio_cross_pe,
            )
            if audio is not None
            else None
        )

        video_prompt_ts = (
            self._compute_prompt_timestep(
                self.prompt_adaln_single, video.sigma, video.latent.shape[0], video.latent.dtype
            )
            if video is not None
            else None
        )
        audio_prompt_ts = (
            self._compute_prompt_timestep(
                self.audio_prompt_adaln_single,
                audio.sigma,
                audio.latent.shape[0],
                audio.latent.dtype,
            )
            if audio is not None
            else None
        )

        for block in self.transformer_blocks:
            video_args, audio_args = block(
                video_args,
                audio_args,
                video_prompt_timestep=video_prompt_ts,
                audio_prompt_timestep=audio_prompt_ts,
            )

        vx = (
            self._process_output(
                self.scale_shift_table,
                self.norm_out,
                self.proj_out,
                video_args.x,
                video_args.embedded_timestep,
            )
            if video_args is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_args.x,
                audio_args.embedded_timestep,
            )
            if audio_args is not None
            else None
        )
        return vx, ax
