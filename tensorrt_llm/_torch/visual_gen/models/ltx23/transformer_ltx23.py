# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3 ("V2") transformer.

Ported from LTX-2 (``LTXModel`` / ``BasicAVTransformerBlock``) with the
version-specific changes verified against the LTX-2.3 checkpoint:

* 9-slot per-block AdaLN (MSA [0:3], MLP [3:6], **text-cross-attn [6:9]**) plus a
  per-block ``prompt_scale_shift_table [2, dim]`` that modulates the text-context
  (K/V) using a sigma-derived ``prompt_timestep`` (video + audio).
* ``adaln_single`` embedding_coefficient 6 -> 9; new ``prompt_adaln_single``
  (coeff 2) produces ``prompt_timestep`` from sigma (video + audio).
* ``caption_projection`` -> ``nn.Identity`` (LTX-2.3 projects to inner_dim in the
  ``text_embedding_projection`` feature extractor *before* the connector).
* Text K/V is projected **per denoise step** from the prompt-modulated context
  (no static per-block K/V cache like LTX-2).
* Audio<->video cross-attention is byte-for-byte identical to LTX-2 (5-slot
  ``scale_shift_table_a2v_ca_*`` + the ``av_ca_*`` AdaLN singles) and is reused.

Perf (Phase-0b): the video+audio self-attention, FFN, and audio<->video
cross-attention conditioning now use the same fused AdaLN C++ ops as LTX-2
(``apply_fused_rmsnorm_shift_scale`` / ``apply_fused_gate_resid_rmsnorm`` /
``apply_fused_gate_resid``), which collapse the RMSNorm + shift/scale + gate +
residual elementwise chains into single kernel launches. This is gated by
``self._fuse_adaln`` (only when the inner dims are supported, and overridable via
``LTX23_DISABLE_FUSE=1``); on unsupported dims / when disabled the helpers fall
back to the eager math, so the result is unchanged. The only remaining eager
conditioning is the text cross-attention query shift/scale + gate and the
sigma-driven prompt K/V modulation (both LTX-2.3-specific; no norm/residual to
fuse). CUDA graph / Ulysses / fp8 remain out of scope here.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import replace

import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.utils import gelu_tanh

from ..ltx2.ltx2_core.adaln import AdaLayerNormSingle
from ..ltx2.ltx2_core.utils_ltx2 import (
    apply_fused_gate_resid,
    apply_fused_gate_resid_rmsnorm,
    apply_fused_rmsnorm_shift_scale,
    apply_shift_scale,
    is_fused_adaln_supported_dim,
)
from ..ltx2.transformer_ltx2 import (
    LTX2Attention,
    LTXModel,
    TransformerConfig,
)
from .text_conditioning_ltx23 import LTX23TextConditioning

# ---------------------------------------------------------------------------
# Perf-only instrumentation (env-guarded; NO behavior change when unset).
#
# The defining LTX-2.3 cost is that each denoise step re-projects the text
# cross-attention K/V from the sigma-modulated prompt context, whereas LTX-2
# projects it once and caches it. These flags exist to *measure* that overhead:
#
#   LTX23_FREEZE_TEXT_KV=1  Compute each block's text K/V once (first step) and
#                           reuse it, mimicking LTX-2's static cache. This makes
#                           the output NUMERICALLY WRONG -- it is a perf A/B knob
#                           only, to isolate the recompute cost via wall-clock.
#   LTX23_NVTX=1            Wrap the text-K/V recompute in an NVTX range so nsys
#                           can attribute GPU time to it directly.
# ---------------------------------------------------------------------------
_FREEZE_TEXT_KV = os.environ.get("LTX23_FREEZE_TEXT_KV", "0") == "1"
_NVTX = os.environ.get("LTX23_NVTX", "0") == "1"

# Perf A/B: force the eager (pre-fusion) AdaLN path even when the fused kernels
# are supported. LTX23_DISABLE_FUSE=1 reproduces the original unfused behavior so
# the fused-vs-eager speedup can be measured in the SAME build (one env var
# apart). Numerically identical either way -- this only changes kernel fusion.
_DISABLE_FUSE = os.environ.get("LTX23_DISABLE_FUSE", "0") == "1"


@contextlib.contextmanager
def _nvtx_range(name: str):
    """Env-guarded NVTX range (no-op unless LTX23_NVTX=1). Used to attribute
    per-step GPU time to each transformer sub-block under nsys."""
    if _NVTX:
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def _get_ada_values(
    scale_shift_table: torch.Tensor,
    batch_size: int,
    timestep: torch.Tensor,
    indices: slice,
) -> tuple[torch.Tensor, ...]:
    """Per-slot AdaLN modulators for ``indices``: one [B, T, D] tensor per slot.

    Broadcast-adds ``scale_shift_table[indices]`` to the reshaped timestep
    embedding, matching the LTX reference ``get_ada_values``.
    """
    num_ada_params = scale_shift_table.shape[0]
    return (
        scale_shift_table[indices]
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=timestep.device, dtype=timestep.dtype)
        + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
    ).unbind(dim=2)


def _get_ada_table_ts_pairs(
    scale_shift_table: torch.Tensor,
    batch_size: int,
    timestep: torch.Tensor,
    indices: slice,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Pair-form companion to ``_get_ada_values`` (matches LTX-2).

    Returns one ``(table_slice, ts_slice)`` pair per slot in ``indices`` *without*
    materializing the broadcast-add. The fused AdaLN kernels fold ``table + ts``
    (and the cast) internally, so passing the pair directly avoids the loose
    add/cast elementwise kernels the combined form would emit.

    Each pair is ``(table_slice [D], ts_slice [B, T, D])``.
    """
    num_ada_params = scale_shift_table.shape[0]
    ts_reshaped = timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)
    return tuple(
        (scale_shift_table[i], ts_reshaped[:, :, i, :])
        for i in range(*indices.indices(num_ada_params))
    )


def _get_av_ca_ada_values(
    scale_shift_table: torch.Tensor,
    batch_size: int,
    scale_shift_timestep: torch.Tensor,
    gate_timestep: torch.Tensor,
    num_scale_shift_values: int = 4,
) -> tuple[torch.Tensor, ...]:
    """AV cross-attention modulators from a [5, D] table.

    Returns (scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate) — identical
    layout to LTX-2's ``_get_av_ca_ada_values``.
    """
    ss_table = scale_shift_table[:num_scale_shift_values, :]
    gate_table = scale_shift_table[num_scale_shift_values:, :]
    num_gate = scale_shift_table.shape[0] - num_scale_shift_values

    ss_vals = (
        ss_table.unsqueeze(0).unsqueeze(0).to(
            device=scale_shift_timestep.device, dtype=scale_shift_timestep.dtype
        )
        + scale_shift_timestep.reshape(
            batch_size, scale_shift_timestep.shape[1], num_scale_shift_values, -1
        )
    ).unbind(dim=2)
    gate_vals = (
        gate_table.unsqueeze(0).unsqueeze(0).to(
            device=gate_timestep.device, dtype=gate_timestep.dtype
        )
        + gate_timestep.reshape(batch_size, gate_timestep.shape[1], num_gate, -1)
    ).unbind(dim=2)
    return (*ss_vals, gate_vals[0])


def _get_av_ca_ada_table_ts_pairs(
    scale_shift_table: torch.Tensor,
    batch_size: int,
    scale_shift_timestep: torch.Tensor,
    gate_timestep: torch.Tensor,
    num_scale_shift_values: int = 4,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Pair-form companion to ``_get_av_ca_ada_values`` (matches LTX-2).

    Returns ``(scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate)`` where each entry
    is a ``(table_slice [D], ts_slice [B, T, D])`` pair. The first
    ``num_scale_shift_values`` slots use ``scale_shift_timestep``; the gate slot
    uses ``gate_timestep``. The fused AdaLN kernels fold ``table + ts`` internally.
    """
    num_ada_params = scale_shift_table.shape[0]
    num_gate_values = num_ada_params - num_scale_shift_values
    ss_table = scale_shift_table[:num_scale_shift_values, :]
    gate_table = scale_shift_table[num_scale_shift_values:, :]
    ss_ts = scale_shift_timestep.reshape(
        batch_size, scale_shift_timestep.shape[1], num_scale_shift_values, -1
    )
    gate_ts = gate_timestep.reshape(batch_size, gate_timestep.shape[1], num_gate_values, -1)
    ss_pairs = tuple((ss_table[i], ss_ts[:, :, i, :]) for i in range(num_scale_shift_values))
    gate_pairs = tuple((gate_table[i], gate_ts[:, :, i, :]) for i in range(num_gate_values))
    return (*ss_pairs, *gate_pairs)


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
        dtype = config.torch_dtype if config is not None else None

        # Fused AdaLN is only valid when the (per-modality) inner dims are
        # supported by the fused kernels; otherwise the helpers fall back to the
        # eager path (numerically identical). Matches LTX-2's per-block gate.
        video_supports_fused_adaln = video is None or is_fused_adaln_supported_dim(video.dim)
        audio_supports_fused_adaln = audio is None or is_fused_adaln_supported_dim(audio.dim)
        self._fuse_adaln = (
            video_supports_fused_adaln and audio_supports_fused_adaln and not _DISABLE_FUSE
        )

        if video is not None:
            self.attn1 = LTX2Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
                config=config,
                layer_idx=idx,
                module_name=f"transformer_blocks.{idx}.attn1",
                enable_sequence_parallel=False,
            )
            self.attn2 = LTX2Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
                config=config,
                layer_idx=idx,
                module_name=f"transformer_blocks.{idx}.attn2",
                enable_sequence_parallel=False,
            )
            self.ff = MLP(
                hidden_size=video.dim,
                intermediate_size=video.dim * 4,
                bias=True,
                activation=gelu_tanh,
                dtype=dtype,
                config=config,
                layer_idx=idx,
            )
            self.scale_shift_table = nn.Parameter(torch.empty(9, video.dim))
            self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, video.dim))

        if audio is not None:
            self.audio_attn1 = LTX2Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
                config=config,
                layer_idx=idx,
                module_name=f"transformer_blocks.{idx}.audio_attn1",
                enable_sequence_parallel=False,
            )
            self.audio_attn2 = LTX2Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
                config=config,
                layer_idx=idx,
                module_name=f"transformer_blocks.{idx}.audio_attn2",
                enable_sequence_parallel=False,
            )
            self.audio_ff = MLP(
                hidden_size=audio.dim,
                intermediate_size=audio.dim * 4,
                bias=True,
                activation=gelu_tanh,
                dtype=dtype,
                config=config,
                layer_idx=idx,
            )
            self.audio_scale_shift_table = nn.Parameter(torch.empty(9, audio.dim))
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty(2, audio.dim))

        if audio is not None and video is not None:
            self.audio_to_video_attn = LTX2Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
                config=config,
                layer_idx=idx,
                module_name=f"transformer_blocks.{idx}.audio_to_video_attn",
                enable_sequence_parallel=False,
            )
            self.video_to_audio_attn = LTX2Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
                config=config,
                layer_idx=idx,
                module_name=f"transformer_blocks.{idx}.video_to_audio_attn",
                enable_sequence_parallel=False,
            )
            self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.empty(5, video.dim))

    def _text_cross_attention(
        self,
        x_normed: torch.Tensor,
        context: torch.Tensor,
        attn: LTX2Attention,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """LTX-2.3 text cross-attention with query + sigma-driven K/V modulation.

        Mirrors ltx-core ``apply_cross_attention_adaln``:
          attn_input = x_normed * (1 + scale_q) + shift_q          # slots [6:9]
          enc        = context  * (1 + scale_kv) + shift_kv        # prompt table
          out        = attn(attn_input, K/V from enc) * gate
        """
        batch_size = x_normed.shape[0]
        shift_q, scale_q, gate = _get_ada_values(
            scale_shift_table, batch_size, timestep, slice(6, 9)
        )
        attn_input = apply_shift_scale(x_normed, scale_q, shift_q)

        # Perf-only A/B path: reuse the first step's text K/V instead of
        # recomputing it (see LTX23_FREEZE_TEXT_KV). Keyed by query batch so CFG
        # cond/uncond (same shape) reuse without a shape crash. Numerically wrong
        # by design -- used solely to time the recompute we skip here.
        if _FREEZE_TEXT_KV:
            cache = getattr(attn, "_frozen_text_kv", None)
            if cache is None:
                cache = {}
                attn._frozen_text_kv = cache
            hit = cache.get(batch_size)
            if hit is not None:
                out = attn(attn_input, pre_projected_kv=hit, timestep=timestep)
                return out * gate

        if _NVTX:
            torch.cuda.nvtx.range_push("ltx23_text_kv_recompute")
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None].to(
                device=x_normed.device, dtype=x_normed.dtype
            )
            + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)
        enc = apply_shift_scale(context, scale_kv, shift_kv)
        # Project K/V from the *modulated* context this step (no static cache).
        k, v = attn.project_kv(enc)
        if _NVTX:
            torch.cuda.nvtx.range_pop()
        if _FREEZE_TEXT_KV:
            attn._frozen_text_kv[batch_size] = (k, v)
        out = attn(attn_input, context=enc, pre_projected_kv=(k, v), timestep=timestep)
        return out * gate

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

        # --- Video self-attention + text cross-attention ---
        if run_vx:
            # Pair-form MSA modulators (slot 0=shift, 1=scale, 2=gate); the fused
            # kernels fold table+ts internally.
            vshift_msa, vscale_msa, vgate_msa = _get_ada_table_ts_pairs(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            norm_vx = apply_fused_rmsnorm_shift_scale(
                vx, vscale_msa[0], vscale_msa[1], vshift_msa[0], vshift_msa[1],
                self.norm_eps, self._fuse_adaln,
            )
            with _nvtx_range("ltx23_video_self_attn"):
                v_msa = self.attn1(norm_vx, pe=video.positional_embeddings, timestep=video.timesteps)
            # Fused gate-residual + RMSNorm: vx <- vx + v_msa*gate_msa; then rms_norm
            # for the text cross-attn query (its slot-[6:9] shift/scale is applied
            # inside _text_cross_attention).
            vx, attn2_q_input = apply_fused_gate_resid_rmsnorm(
                vx, v_msa, vgate_msa[0], vgate_msa[1], self.norm_eps, self._fuse_adaln,
            )
            with _nvtx_range("ltx23_video_text_cross_attn"):
                vx = vx + self._text_cross_attention(
                    attn2_q_input,
                    video.context,
                    self.attn2,
                    self.scale_shift_table,
                    self.prompt_scale_shift_table,
                    video.timesteps,
                    video_prompt_timestep,
                )

        # --- Audio self-attention + text cross-attention ---
        if run_ax:
            ashift_msa, ascale_msa, agate_msa = _get_ada_table_ts_pairs(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )
            norm_ax = apply_fused_rmsnorm_shift_scale(
                ax, ascale_msa[0], ascale_msa[1], ashift_msa[0], ashift_msa[1],
                self.norm_eps, self._fuse_adaln,
            )
            with _nvtx_range("ltx23_audio_self_attn"):
                a_msa = self.audio_attn1(norm_ax, pe=audio.positional_embeddings, timestep=audio.timesteps)
            ax, audio_attn2_q_input = apply_fused_gate_resid_rmsnorm(
                ax, a_msa, agate_msa[0], agate_msa[1], self.norm_eps, self._fuse_adaln,
            )
            with _nvtx_range("ltx23_audio_text_cross_attn"):
                ax = ax + self._text_cross_attention(
                    audio_attn2_q_input,
                    audio.context,
                    self.audio_attn2,
                    self.audio_scale_shift_table,
                    self.audio_prompt_scale_shift_table,
                    audio.timesteps,
                    audio_prompt_timestep,
                )

        # --- Bidirectional audio <-> video cross-attention (identical to LTX-2) ---
        if run_a2v or run_v2a:
            if _NVTX:
                torch.cuda.nvtx.range_push("ltx23_av_cross_attn")
            vx_pre, ax_pre = vx, ax
            if run_a2v:
                scale_v_a2v, shift_v_a2v, _, _, gate_a2v = _get_av_ca_ada_table_ts_pairs(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                )
                a2v_vx = apply_fused_rmsnorm_shift_scale(
                    vx_pre, scale_v_a2v[0], scale_v_a2v[1], shift_v_a2v[0], shift_v_a2v[1],
                    self.norm_eps, self._fuse_adaln,
                )
                scale_a_a2v, shift_a_a2v, _, _, _ = _get_av_ca_ada_table_ts_pairs(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                )
                a2v_ax = apply_fused_rmsnorm_shift_scale(
                    ax_pre, scale_a_a2v[0], scale_a_a2v[1], shift_a_a2v[0], shift_a_a2v[1],
                    self.norm_eps, self._fuse_adaln,
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
                    ax_pre, scale_a_v2a[0], scale_a_v2a[1], shift_a_v2a[0], shift_a_v2a[1],
                    self.norm_eps, self._fuse_adaln,
                )
                _, _, scale_v_v2a, shift_v_v2a, _ = _get_av_ca_ada_table_ts_pairs(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                )
                v2a_vx = apply_fused_rmsnorm_shift_scale(
                    vx_pre, scale_v_v2a[0], scale_v_v2a[1], shift_v_v2a[0], shift_v_v2a[1],
                    self.norm_eps, self._fuse_adaln,
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

            if _NVTX:
                torch.cuda.nvtx.range_pop()

        # --- Video FFN ---
        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = _get_ada_table_ts_pairs(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = apply_fused_rmsnorm_shift_scale(
                vx, vscale_mlp[0], vscale_mlp[1], vshift_mlp[0], vshift_mlp[1],
                self.norm_eps, self._fuse_adaln,
            )
            with _nvtx_range("ltx23_video_ffn"):
                vx = apply_fused_gate_resid(
                    vx, self.ff(vx_scaled), vgate_mlp[0], vgate_mlp[1], self._fuse_adaln
                )

        # --- Audio FFN ---
        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = _get_ada_table_ts_pairs(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = apply_fused_rmsnorm_shift_scale(
                ax, ascale_mlp[0], ascale_mlp[1], ashift_mlp[0], ashift_mlp[1],
                self.norm_eps, self._fuse_adaln,
            )
            with _nvtx_range("ltx23_audio_ffn"):
                ax = apply_fused_gate_resid(
                    ax, self.audio_ff(ax_scaled), agate_mlp[0], agate_mlp[1], self._fuse_adaln
                )

        return (
            replace(video, x=vx) if video is not None else None,
            replace(audio, x=ax) if audio is not None else None,
        )


class LTX23Model(LTXModel):
    """LTX-2.3 transformer. Reuses LTXModel machinery; overrides V2-specific parts.

    The embeddings connectors and split feature extractor are pipeline-level
    components (matching LTX-2, where the transformer weight load excludes the
    ``*_embeddings_connector.`` prefixes). This model therefore owns only the
    diffusion transformer itself and receives already-connector-processed text
    context in ``prepare_text_cache``.
    """

    # -- Init overrides ------------------------------------------------------

    def _init_video(self, in_channels, out_channels, caption_channels, norm_eps):
        self.patchify_proj = self._make_linear(in_channels, self.inner_dim)
        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, embedding_coefficient=9, make_linear=self._make_linear
        )
        self.prompt_adaln_single = AdaLayerNormSingle(
            self.inner_dim, embedding_coefficient=2, make_linear=self._make_linear
        )
        # caption projection happens before the connector (text_embedding_projection),
        # so the in-transformer projection is bypassed.
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

    def reset_text_kv_cache(self) -> None:
        """Clear the perf-only frozen text-K/V cache (see LTX23_FREEZE_TEXT_KV).

        No-op unless a prior forward ran under ``LTX23_FREEZE_TEXT_KV=1``. Call at
        the start of each generation so the cache re-primes on step 0.
        """
        for block in self.transformer_blocks:
            for name in ("attn2", "audio_attn2"):
                attn = getattr(block, name, None)
                if attn is not None and hasattr(attn, "_frozen_text_kv"):
                    attn._frozen_text_kv = {}

    # -- Text conditioning (no static K/V) -----------------------------------

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
        """Compute step-invariant text conditioning. No per-block K/V (step-varying).

        ``*_context`` are the connector outputs (the pipeline runs the split
        feature extractor + the video/audio embeddings connectors before calling
        this). caption_projection is Identity, so the preprocessor passes the
        connector output through unchanged.
        """
        out = LTX23TextConditioning()

        if video_context is not None:
            v_ctx, v_mask, v_pe, v_cross_pe = self.video_args_preprocessor.prepare_text_cache(
                video_context, video_context_mask, video_positions, dtype
            )
            # NOTE (LTX-2.3): unlike LTX-2 we deliberately do NOT pre-project a
            # static per-block text K/V here. The K/V is modulated by the
            # sigma-derived prompt_timestep and is therefore step-varying, so it
            # is (re)projected inside each denoise step (see block forward).
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

        # Build sharded-local PE in the form the attention consumer expects
        # (identical to LTX-2; one-time reshape/shard so the denoise loop has no
        # PE reshape work). fuse flags are read off the constructed attentions.
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

    # -- Forward -------------------------------------------------------------

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
