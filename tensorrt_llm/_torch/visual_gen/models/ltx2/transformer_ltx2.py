# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

# Architecture ported from LTX-2,
# with compute-heavy components replaced by TRT-LLM optimized modules:
#   - Linear projections  → tensorrt_llm._torch.modules.linear.Linear
#   - RMSNorm (QK norm)   → tensorrt_llm._torch.modules.rms_norm.RMSNorm
#   - FeedForward (MLP)    → tensorrt_llm._torch.modules.mlp.MLP
#   - Attention backend    → tensorrt_llm._torch.visual_gen.attention_backend
#
# Architecture-specific components (RoPE, AdaLN, timestep/text embeddings,
# modality dataclass, transformer args) are ported from LTX-2 and live
# in the ltx2_core/ subpackage.

# TODO: replace torch rms_norm with TRT-LLM RMSNorm (no weights)

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from tensorrt_llm._torch.modules.linear import Linear, WeightMode
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.parallelism import setup_sequence_parallelism
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from .ltx2_core.adaln import AdaLayerNormSingle
from .ltx2_core.modality import Modality
from .ltx2_core.perturbations import BatchedPerturbationConfig, PerturbationType
from .ltx2_core.rope import LTXRopeType, apply_rotary_emb
from .ltx2_core.text_projection import PixArtAlphaTextProjection
from .ltx2_core.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from .ltx2_core.utils_ltx2 import rms_norm

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


# ---------------------------------------------------------------------------
# LTX2Attention: TRT-LLM Linear + RMSNorm + attention backend + LTX-2 RoPE
# ---------------------------------------------------------------------------


class LTX2Attention(Attention):
    """LTX-2 attention: extends base Attention with LTX-specific RoPE, gated
    attention, and separate K-RoPE for audio-video cross-attention.

    Inherits from base Attention:
    - Q/K/V Linear creation with quant_config propagation
    - QK RMSNorm (norm_q / norm_k)
    - Backend dispatch with automatic HND/NHD layout handling (_attn_impl)
    - Output projection (to_out)

    Adds LTX-2 specifics:
    - LTX 3D RoPE (INTERLEAVED / SPLIT) with separate k_pe support
    - Gated attention (to_gate_logits)
    - Cross-attention with different context_dim for K/V input
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: int = 0,
        use_ulysses: bool = False,
    ):
        from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

        config = config or DiffusionModelConfig()

        # Store before super().__init__() — _init_qkv_proj needs _context_dim
        self._context_dim = context_dim if context_dim is not None else query_dim
        self.rope_type = rope_type
        self._is_cross_attn = context_dim is not None

        # Self-attention: FUSE_QKV enables the optimized backend + auto Ulysses
        # wrapping from the base class.
        # Cross-attention: SEPARATE_QKV since K/V come from a different source.
        qkv_mode = QKVMode.SEPARATE_QKV if self._is_cross_attn else QKVMode.FUSE_QKV

        super().__init__(
            hidden_size=query_dim,
            num_attention_heads=heads,
            head_dim=dim_head,
            qkv_mode=qkv_mode,
            qk_norm=True,
            qk_norm_mode="full",
            eps=norm_eps,
            bias=True,
            config=config,
            layer_idx=layer_idx,
        )

        # For audio self-attention that may need a runtime Ulysses toggle
        # (sequence length not always divisible by ulysses_size), create a
        # plain backend as fallback.  The base class already set self.attn
        # to UlyssesAttention(inner_backend=sharded_backend).
        self._has_dual_attn = False
        ulysses_size = config.parallel.dit_ulysses_size
        if use_ulysses and not self._is_cross_attn and ulysses_size > 1:
            self._ulysses_attn = self.attn
            self._plain_attn = create_attention(
                backend=self.attn_backend,
                layer_idx=self.layer_idx,
                num_heads=self.num_attention_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.num_key_value_heads,
                quant_config=self.quant_config,
                dtype=self.dtype,
            )
            self._has_dual_attn = True

        if apply_gated_attention:
            self.to_gate_logits = Linear(
                query_dim,
                heads,
                bias=True,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
            )
        else:
            self.to_gate_logits = None

    def set_ulysses_active(self, active: bool):
        """Toggle between UlyssesAttention and plain attention at runtime.

        Only effective for modules created with ``use_ulysses=True``.
        """
        if not self._has_dual_attn:
            return
        self._modules.pop("attn", None)
        self.attn = self._ulysses_attn if active else self._plain_attn

    def _init_qkv_proj(self):
        """Override for cross-attention: use _context_dim for K/V input.

        Self-attention delegates to the base class which creates a fused
        qkv_proj (FUSE_QKV).
        """
        if not self._is_cross_attn:
            super()._init_qkv_proj()
            return
        self.to_q = Linear(
            self.hidden_size,
            self.q_dim,
            bias=self.bias,
            dtype=self.dtype,
            mapping=self.mapping,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
        )
        self.to_k = Linear(
            self._context_dim,
            self.kv_dim,
            bias=self.bias,
            dtype=self.dtype,
            mapping=self.mapping,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
        )
        self.to_v = Linear(
            self._context_dim,
            self.kv_dim,
            bias=self.bias,
            dtype=self.dtype,
            mapping=self.mapping,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
        )

    def project_kv(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and normalize K/V from context.

        Used by the project-before-gather pattern in AV cross-attention:
        project K/V on sharded data, then all-gather the smaller projected
        tensors instead of all-gathering the full context first.
        """
        k = self.to_k(context)
        v = self.to_v(context)
        if self.qk_norm:
            k = self.norm_k(k)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        pre_projected_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Query input [B, T, D].
            context: Key/value input [B, S, C]. None → self-attention.
            pe: (cos, sin) RoPE embeddings for Q (and K when k_pe is None).
            k_pe: Separate (cos, sin) RoPE embeddings for K (for AV cross-attn).
            pre_projected_kv: Pre-projected (k, v) tuple from project_kv().
                When provided, skips K/V projection and K-norm (already done).
        """
        if pre_projected_kv is not None:
            k, v = pre_projected_kv
            q = self.to_q(x)
            if self.qk_norm:
                q = self.norm_q(q)
        else:
            q, k, v = self.get_qkv(x, context)
            q, k = self.apply_qk_norm(q, k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        out = self._attn_impl(q, k, v)

        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, self.num_attention_heads, self.head_dim)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.num_attention_heads * self.head_dim)

        return self.to_out[0](out)


# ---------------------------------------------------------------------------
# TransformerConfig + BasicAVTransformerBlock
# ---------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False


class BasicAVTransformerBlock(nn.Module):
    """Dual-stream (Audio/Video) transformer block using TRT-LLM primitives.

    Each block contains per-modality self-attention, cross-attention (text),
    bidirectional AV cross-attention, and FFN — all with AdaLN modulation.
    """

    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        config: Optional["DiffusionModelConfig"] = None,
    ):
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps

        self._use_ulysses = False
        self._audio_is_sharded = False
        if config is not None and config.parallel.dit_ulysses_size > 1:
            self._use_ulysses = True
            self._ulysses_size = config.parallel.dit_ulysses_size
            self._ulysses_pg = getattr(config, "ulysses_process_group", None)

        if video is not None:
            self._init_video_modules(video, rope_type, norm_eps, config, idx)

        if audio is not None:
            self._init_audio_modules(audio, rope_type, norm_eps, config, idx)

        if audio is not None and video is not None:
            self._init_av_cross_modules(video, audio, rope_type, norm_eps, config, idx)

    @staticmethod
    def _make_mlp(cfg, model_config, idx):
        dtype = model_config.torch_dtype if model_config else None
        return MLP(
            hidden_size=cfg.dim,
            intermediate_size=cfg.dim * 4,
            bias=True,
            activation=lambda x: F.gelu(x, approximate="tanh"),
            dtype=dtype,
            config=model_config,
            layer_idx=idx,
        )

    def _init_video_modules(self, cfg, rope_type, eps, model_config, idx):
        self.attn1 = LTX2Attention(
            query_dim=cfg.dim,
            heads=cfg.heads,
            dim_head=cfg.d_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
            use_ulysses=True,
        )
        self.attn2 = LTX2Attention(
            query_dim=cfg.dim,
            context_dim=cfg.context_dim,
            heads=cfg.heads,
            dim_head=cfg.d_head,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
        )
        self.ff = self._make_mlp(cfg, model_config, idx)
        self.scale_shift_table = nn.Parameter(torch.empty(6, cfg.dim))

    def _init_audio_modules(self, cfg, rope_type, eps, model_config, idx):
        self.audio_attn1 = LTX2Attention(
            query_dim=cfg.dim,
            heads=cfg.heads,
            dim_head=cfg.d_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
            use_ulysses=True,
        )
        self.audio_attn2 = LTX2Attention(
            query_dim=cfg.dim,
            context_dim=cfg.context_dim,
            heads=cfg.heads,
            dim_head=cfg.d_head,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
        )
        self.audio_ff = self._make_mlp(cfg, model_config, idx)
        self.audio_scale_shift_table = nn.Parameter(torch.empty(6, cfg.dim))

    def _init_av_cross_modules(self, v_cfg, a_cfg, rope_type, eps, model_config, idx):
        self.audio_to_video_attn = LTX2Attention(
            query_dim=v_cfg.dim,
            context_dim=a_cfg.dim,
            heads=a_cfg.heads,
            dim_head=a_cfg.d_head,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=v_cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
        )
        self.video_to_audio_attn = LTX2Attention(
            query_dim=a_cfg.dim,
            context_dim=v_cfg.dim,
            heads=a_cfg.heads,
            dim_head=a_cfg.d_head,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=a_cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
        )
        self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.empty(5, a_cfg.dim))
        self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.empty(5, v_cfg.dim))

    # -- AdaLN helpers -------------------------------------------------------

    @staticmethod
    def _get_ada_values(
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        ada_values = (
            scale_shift_table[indices]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    @staticmethod
    def _get_av_ca_ada_values(
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        ss_table = scale_shift_table[:num_scale_shift_values, :]
        gate_table = scale_shift_table[num_scale_shift_values:, :]

        ss_vals = (
            ss_table.unsqueeze(0)
            .unsqueeze(0)
            .to(device=scale_shift_timestep.device, dtype=scale_shift_timestep.dtype)
            + scale_shift_timestep.reshape(
                batch_size, scale_shift_timestep.shape[1], num_scale_shift_values, -1
            )
        ).unbind(dim=2)

        gate_vals = (
            gate_table.unsqueeze(0)
            .unsqueeze(0)
            .to(device=gate_timestep.device, dtype=gate_timestep.dtype)
            + gate_timestep.reshape(
                batch_size, gate_timestep.shape[1], num_ada_params - num_scale_shift_values, -1
            )
        ).unbind(dim=2)

        ss_chunks = [t.squeeze(2) for t in ss_vals]
        gate_chunks = [t.squeeze(2) for t in gate_vals]
        return (*ss_chunks, *gate_chunks)

    # -- Sequence-parallel helpers for AV cross-attention ----------------------

    def _sp_all_gather(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """All-gather *x* along *dim* across sequence-parallel ranks."""
        x = x.contiguous()
        gathered = [torch.empty_like(x) for _ in range(self._ulysses_size)]
        dist.all_gather(gathered, x, group=self._ulysses_pg)
        return torch.cat(gathered, dim=dim)

    def _sp_gather_pe(self, pe):
        """All-gather RoPE (cos, sin) tuple along the sequence dim."""
        if pe is None:
            return None
        cos, sin = pe
        # Split RoPE: [B, H, S, D] — sequence at dim 2
        # Interleaved RoPE: [B, S, D] — sequence at dim 1
        seq_dim = 2 if cos.ndim == 4 else 1
        return (self._sp_all_gather(cos, dim=seq_dim), self._sp_all_gather(sin, dim=seq_dim))

    # -- Forward -------------------------------------------------------------

    def forward(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations=None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """Forward with optional perturbation masking for STG.

        Args:
            perturbations: Optional ``BatchedPerturbationConfig`` that masks
                attention outputs for selected blocks/modalities.
        """
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        has_perturbations = perturbations is not None and isinstance(
            perturbations, BatchedPerturbationConfig
        )

        # --- Video self-attention + text cross-attention ---
        if run_vx:
            skip_v_self = has_perturbations and perturbations.all_in_batch(
                PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx
            )
            vshift_msa, vscale_msa, vgate_msa = self._get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            if not skip_v_self:
                norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
                v_self_out = self.attn1(norm_vx, pe=video.positional_embeddings) * vgate_msa
                if has_perturbations and perturbations.any_in_batch(
                    PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx
                ):
                    v_self_out = v_self_out * perturbations.mask_like(
                        PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, v_self_out
                    )
                vx = vx + v_self_out
            vx = vx + self.attn2(
                rms_norm(vx, eps=self.norm_eps),
                context=video.context,
            )
            del vshift_msa, vscale_msa, vgate_msa

        # --- Audio self-attention + text cross-attention ---
        if run_ax:
            skip_a_self = has_perturbations and perturbations.all_in_batch(
                PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx
            )
            ashift_msa, ascale_msa, agate_msa = self._get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )
            if not skip_a_self:
                norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
                a_self_out = self.audio_attn1(norm_ax, pe=audio.positional_embeddings) * agate_msa
                if has_perturbations and perturbations.any_in_batch(
                    PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx
                ):
                    a_self_out = a_self_out * perturbations.mask_like(
                        PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, a_self_out
                    )
                ax = ax + a_self_out
            ax = ax + self.audio_attn2(
                rms_norm(ax, eps=self.norm_eps),
                context=audio.context,
            )
            del ashift_msa, ascale_msa, agate_msa

        # --- Bidirectional audio ↔ video cross-attention ---
        if run_a2v or run_v2a:
            skip_a2v = has_perturbations and perturbations.all_in_batch(
                PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx
            )
            skip_v2a = has_perturbations and perturbations.all_in_batch(
                PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx
            )

            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            (
                scale_ca_audio_a2v,
                shift_ca_audio_a2v,
                scale_ca_audio_v2a,
                shift_ca_audio_v2a,
                gate_out_v2a,
            ) = self._get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            (
                scale_ca_video_a2v,
                shift_ca_video_a2v,
                scale_ca_video_v2a,
                shift_ca_video_v2a,
                gate_out_a2v,
            ) = self._get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            if run_a2v and not skip_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v

                # Project-before-gather: K/V projections run on sharded data
                # so they benefit from Ulysses scaling.  Only the smaller
                # projected tensors are all-gathered.
                k_a2v, v_a2v = self.audio_to_video_attn.project_kv(ax_scaled)
                if self._audio_is_sharded:
                    k_a2v = self._sp_all_gather(k_a2v)
                    v_a2v = self._sp_all_gather(v_a2v)
                    k_pe_a2v = self._sp_gather_pe(audio.cross_positional_embeddings)
                else:
                    k_pe_a2v = audio.cross_positional_embeddings

                a2v_out = (
                    self.audio_to_video_attn(
                        vx_scaled,
                        pre_projected_kv=(k_a2v, v_a2v),
                        pe=video.cross_positional_embeddings,
                        k_pe=k_pe_a2v,
                    )
                    * gate_out_a2v
                )
                if has_perturbations and perturbations.any_in_batch(
                    PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx
                ):
                    a2v_out = a2v_out * perturbations.mask_like(
                        PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, a2v_out
                    )
                vx = vx + a2v_out

            if run_v2a and not skip_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a

                # Project-before-gather (video → audio direction).
                k_v2a, v_v2a = self.video_to_audio_attn.project_kv(vx_scaled)
                if self._use_ulysses:
                    k_v2a = self._sp_all_gather(k_v2a)
                    v_v2a = self._sp_all_gather(v_v2a)
                    k_pe_v2a = self._sp_gather_pe(video.cross_positional_embeddings)
                else:
                    k_pe_v2a = video.cross_positional_embeddings

                v2a_out = (
                    self.video_to_audio_attn(
                        ax_scaled,
                        pre_projected_kv=(k_v2a, v_v2a),
                        pe=audio.cross_positional_embeddings,
                        k_pe=k_pe_v2a,
                    )
                    * gate_out_v2a
                )
                if has_perturbations and perturbations.any_in_batch(
                    PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx
                ):
                    v2a_out = v2a_out * perturbations.mask_like(
                        PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, v2a_out
                    )
                ax = ax + v2a_out

        # --- Video FFN ---
        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self._get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, None)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

        # --- Audio FFN ---
        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self._get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, None)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        return (
            replace(video, x=vx) if video is not None else None,
            replace(audio, x=ax) if audio is not None else None,
        )


class LTX2CacheDiTPattern0BlockWrapper(nn.Module):
    """Pattern_0: (video x, audio x) in/out; the 2nd slot is cache-dit's encoder_hidden_states name only.

    Caption/RoPE/masks live on the parent's _cache_dit_*_args; Pattern_0 only threads the two latent streams.
    """

    def __init__(self, inner: BasicAVTransformerBlock, parent: "LTXModel"):
        super().__init__()
        self.inner = inner
        # Same module as inner; use a direct ref so torch.compile/CachedBlocks cannot break delegation.
        object.__setattr__(self, "_inner_module", inner)
        # Must not register parent as a submodule (would cycle the module graph).
        object.__setattr__(self, "_ltx_parent", parent)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        perturbations=None,
        **kwargs: Any,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        p = object.__getattribute__(self, "_ltx_parent")
        inner_mod = object.__getattribute__(self, "_inner_module")
        va_src = p._cache_dit_video_args
        aa_src = p._cache_dit_audio_args
        va = replace(va_src, x=hidden_states) if va_src is not None else None
        if aa_src is not None and encoder_hidden_states is not None:
            aa: Optional[TransformerArgs] = replace(aa_src, x=encoder_hidden_states)
        else:
            aa = aa_src
        out_v, out_a = inner_mod(video=va, audio=aa, perturbations=perturbations, **kwargs)
        return (
            out_v.x if out_v is not None else None,
            out_a.x if out_a is not None else None,
        )

    def __getattr__(self, name: str):
        if name in ("_inner_module", "_ltx_parent"):
            return object.__getattribute__(self, name)
        inner_mod = object.__getattribute__(self, "_inner_module")
        return getattr(inner_mod, name)


# ---------------------------------------------------------------------------
# LTXModelType + LTXModel (top-level)
# ---------------------------------------------------------------------------


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(nn.Module):
    """LTX-2 transformer built from TRT-LLM primitives.

    Native implementation using optimized TRT-LLM Linear, RMSNorm, MLP, and
    attention backends for all compute-heavy operations.

    The architecture-specific wiring (RoPE, AdaLN, dual-stream blocks, etc.)
    follows the Lightricks reference implementation.
    """

    def __init__(
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        model_config: Optional["DiffusionModelConfig"] = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.model_type = model_type
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta

        cross_pe_max_pos = None

        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(in_channels, out_channels, caption_channels, norm_eps)

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(audio_in_channels, audio_out_channels, caption_channels, norm_eps)

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(
                self.positional_embedding_max_pos[0],
                self.audio_positional_embedding_max_pos[0],
            )
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)

        # Ulysses sequence parallelism — must run before block/attention init
        # so that model_config.ulysses_process_group is available.
        primary_heads = (
            num_attention_heads if model_type.is_video_enabled() else audio_num_attention_heads
        )
        (self.use_ulysses, self.ulysses_size, self.ulysses_pg, self.ulysses_rank) = (
            setup_sequence_parallelism(
                model_config=model_config,
                num_attention_heads=primary_heads,
            )
        )
        # Audio is sharded by Ulysses only when its sequence length is
        # divisible by ulysses_size (checked at runtime in forward).
        # Head divisibility is validated here since the attention backend
        # is created at init with sharded head counts.
        if self.use_ulysses and model_type.is_audio_enabled():
            if audio_num_attention_heads % self.ulysses_size != 0:
                raise ValueError(
                    f"audio_num_attention_heads ({audio_num_attention_heads}) "
                    f"must be divisible by ulysses_size ({self.ulysses_size})"
                )

        self._audio_is_sharded = False
        self._cache_dit_video_args: Optional[TransformerArgs] = None
        self._cache_dit_audio_args: Optional[TransformerArgs] = None

        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=(
                audio_attention_head_dim if model_type.is_audio_enabled() else 0
            ),
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            apply_gated_attention=apply_gated_attention,
        )

        self.__post_init__()

    @property
    def device(self):
        return next(self.parameters()).device

    def __post_init__(self):
        """Apply quant exclusions then materialize deferred Linear weights."""
        self._apply_quant_config_exclude_modules()
        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    # ==================== FP8 static checkpoint workaround ====================
    # Pre-quantized FP8 checkpoints (HuggingFace _quantization_metadata format)
    # embed layer names using the original checkpoint convention, which diverges
    # from TRT-LLM model names after QKV fusion and FF remapping.
    #
    # _remap_exclude_modules translates those names so that non-quantized layers
    # are correctly excluded from FP8 quantization.
    #
    # TODO: Remove this block once checkpoint tooling emits model-convention
    # names directly (i.e. qkv_proj, up_proj, down_proj instead of
    # to_q/to_k/to_v, ff.net.0.proj, ff.net.2).
    # ========================================================================

    @staticmethod
    def _remap_exclude_modules(exclude_modules: list[str]) -> list[str]:
        """Translate checkpoint-convention exclude names to model-convention names.

        The checkpoint uses naming conventions that differ from the TRT-LLM
        model after QKV fusion and FF remapping:
          - Self-attention QKV: ``to_q / to_k / to_v`` → fused ``qkv_proj``
          - FeedForward:        ``ff.net.0.proj / ff.net.2`` → ``ff.up_proj / ff.down_proj``

        Returns a combined list containing both original and remapped patterns
        so that ``fnmatch`` can match either convention.
        """
        remapped: set[str] = set()
        for entry in exclude_modules:
            for qkv_suffix in (".to_q", ".to_k", ".to_v"):
                if entry.endswith(qkv_suffix):
                    remapped.add(entry[: -len(qkv_suffix)] + ".qkv_proj")
            for ff_prefix in (".ff.", ".audio_ff."):
                old_up = ff_prefix + "net.0.proj"
                old_down = ff_prefix + "net.2"
                if old_up in entry:
                    remapped.add(entry.replace(old_up, ff_prefix + "up_proj"))
                elif old_down in entry:
                    remapped.add(entry.replace(old_down, ff_prefix + "down_proj"))
        return list(exclude_modules) + sorted(remapped)

    # ==================== End FP8 static checkpoint workaround ===============

    def _apply_quant_config_exclude_modules(self):
        if self.model_config is None:
            return
        quant_config = self.model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return

        kv_cache_quant_algo = quant_config.kv_cache_quant_algo if quant_config else None
        no_quant_config = QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo)

        needs_remap = quant_config.quant_algo in (QuantAlgo.FP8,)
        if needs_remap:
            # FP8 static checkpoint: remap exclude names (see above)
            all_patterns = self._remap_exclude_modules(quant_config.exclude_modules)
        else:
            all_patterns = list(quant_config.exclude_modules)

        for name, module in self.named_modules():
            if isinstance(module, Linear):
                is_excluded = any(fnmatch.fnmatchcase(name, pat) for pat in all_patterns)
                if is_excluded and getattr(module, "quant_config", None) is not None:
                    module.quant_config = no_quant_config

    # -- Initialization helpers ----------------------------------------------

    def _make_linear(self, in_features: int, out_features: int, bias: bool = True) -> nn.Module:
        """Create a Linear layer using the TRT-LLM backend."""
        dtype = self.model_config.torch_dtype if self.model_config else None
        quant_config = self.model_config.quant_config if self.model_config else None
        skip_create = self.model_config.skip_create_weights_in_init if self.model_config else False
        force_dq = self.model_config.force_dynamic_quantization if self.model_config else False
        mapping = getattr(self.model_config, "mapping", None) if self.model_config else None
        return Linear(
            in_features,
            out_features,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create,
            force_dynamic_quantization=force_dq,
        )

    def _init_video(self, in_channels, out_channels, caption_channels, norm_eps):
        self.patchify_proj = self._make_linear(in_channels, self.inner_dim)
        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            make_linear=self._make_linear,
        )
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.inner_dim,
            make_linear=self._make_linear,
        )
        self.scale_shift_table = nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = self._make_linear(self.inner_dim, out_channels)

    def _init_audio(self, in_channels, out_channels, caption_channels, norm_eps):
        self.audio_patchify_proj = self._make_linear(in_channels, self.audio_inner_dim)
        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            make_linear=self._make_linear,
        )
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.audio_inner_dim,
            make_linear=self._make_linear,
        )
        self.audio_scale_shift_table = nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = nn.LayerNorm(
            self.audio_inner_dim, elementwise_affine=False, eps=norm_eps
        )
        self.audio_proj_out = self._make_linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(self, num_scale_shift_values):
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
            make_linear=self._make_linear,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
            make_linear=self._make_linear,
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
            make_linear=self._make_linear,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
            make_linear=self._make_linear,
        )

    def _init_preprocessors(self, cross_pe_max_pos):
        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
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
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )

    def _uses_cache_dit(self) -> bool:
        mc = getattr(self, "model_config", None)
        return mc is not None and getattr(mc, "cache_backend", None) == "cache_dit"

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
        blocks: list[nn.Module] = [
            BasicAVTransformerBlock(
                idx=idx,
                video=video_config,
                audio=audio_config,
                rope_type=self.rope_type,
                norm_eps=norm_eps,
                config=self.model_config,
            )
            for idx in range(num_layers)
        ]
        if self._uses_cache_dit():
            blocks = [
                LTX2CacheDiTPattern0BlockWrapper(b, parent=self)
                for b in blocks  # type: ignore[misc]
            ]
        self.transformer_blocks = nn.ModuleList(blocks)

    # -- Ulysses sequence sharding / gathering --------------------------------

    def _shard_transformer_args(self, args: TransformerArgs) -> TransformerArgs:
        """Shard sequence-dependent fields of *args* for Ulysses."""
        seq_len = args.x.shape[1]
        chunk = seq_len // self.ulysses_size
        s = self.ulysses_rank * chunk
        e = s + chunk

        def _shard(t):
            if t is None or t.ndim < 2 or t.shape[1] != seq_len:
                return t
            return t[:, s:e]

        def _shard_pe(pe):
            if pe is None:
                return None
            cos, sin = pe
            if cos.ndim == 4 and cos.shape[2] == seq_len:
                # Split RoPE: [B, H, S, D] — sequence dim at index 2
                return (cos[:, :, s:e], sin[:, :, s:e])
            elif cos.ndim == 3 and cos.shape[1] == seq_len:
                # Interleaved RoPE: [B, S, D] — sequence dim at index 1
                return (cos[:, s:e], sin[:, s:e])
            return pe

        return replace(
            args,
            x=args.x[:, s:e],
            timesteps=_shard(args.timesteps),
            embedded_timestep=_shard(args.embedded_timestep),
            positional_embeddings=_shard_pe(args.positional_embeddings),
            cross_positional_embeddings=_shard_pe(args.cross_positional_embeddings),
            cross_scale_shift_timestep=_shard(args.cross_scale_shift_timestep),
            cross_gate_timestep=_shard(args.cross_gate_timestep),
        )

    def _gather_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather hidden states along the sequence dim."""
        x = x.contiguous()
        gathered = [torch.empty_like(x) for _ in range(self.ulysses_size)]
        dist.all_gather(gathered, x, group=self.ulysses_pg)
        return torch.cat(gathered, dim=1)

    def configure_audio_ulysses(self, audio_seq_len: int) -> None:
        """Configure whether audio uses Ulysses based on sequence length.

        Call once before the denoising loop when the audio token count is
        known.  The decision is cached — ``forward()`` uses it without
        re-checking.
        """
        if not self.use_ulysses:
            self._audio_is_sharded = False
            return

        self._audio_is_sharded = audio_seq_len % self.ulysses_size == 0
        for block in self.transformer_blocks:
            target = block.inner if isinstance(block, LTX2CacheDiTPattern0BlockWrapper) else block
            target._audio_is_sharded = self._audio_is_sharded
            if hasattr(target, "audio_attn1"):
                target.audio_attn1.set_ulysses_active(self._audio_is_sharded)

    # -- Output processing ---------------------------------------------------

    @staticmethod
    def _process_output(
        scale_shift_table: nn.Parameter,
        norm_out: nn.LayerNorm,
        proj_out: nn.Module,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        x = norm_out(x)
        x = x * (1 + scale) + shift
        return proj_out(x)

    # -- Forward -------------------------------------------------------------

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations=None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass through the LTX-2 transformer.

        Args:
            video: Video modality input (or None).
            audio: Audio modality input (or None).
            perturbations: Optional ``BatchedPerturbationConfig`` for STG.

        Returns:
            Tuple of (video_output, audio_output) velocity predictions.
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio) if audio is not None else None

        # Shard sequences for Ulysses parallelism.
        # Video is always sharded.  Audio sharding is decided once by
        # configure_audio_ulysses() and cached in self._audio_is_sharded.
        if self.use_ulysses:
            if video_args is not None:
                video_args = self._shard_transformer_args(video_args)
            if self._audio_is_sharded and audio_args is not None:
                audio_args = self._shard_transformer_args(audio_args)

        if self._uses_cache_dit():
            vx = video_args.x if video_args is not None else None
            ax = audio_args.x if audio_args is not None else None
            for block in self.transformer_blocks:
                self._cache_dit_video_args = video_args
                self._cache_dit_audio_args = audio_args
                vx, ax = block(vx, ax, perturbations=perturbations)
                if video_args is not None and vx is not None:
                    video_args = replace(video_args, x=vx)
                if audio_args is not None and ax is not None:
                    audio_args = replace(audio_args, x=ax)
        else:
            for block in self.transformer_blocks:
                video_args, audio_args = block(
                    video=video_args,
                    audio=audio_args,
                    perturbations=perturbations,
                )

        # Gather sequences back to full length for output processing.
        # Only gather embedded_timestep if it was actually sharded (dim-1
        # matches x); scalar timestep embeddings [B, 1, D] are
        # broadcast-compatible and must not be gathered.
        if self.use_ulysses:
            if video_args is not None:
                gathered_vx = self._gather_sequence(video_args.x)
                v_et = video_args.embedded_timestep
                if v_et.shape[1] == video_args.x.shape[1]:
                    v_et = self._gather_sequence(v_et)
                video_args = replace(
                    video_args,
                    x=gathered_vx,
                    embedded_timestep=v_et,
                )
            if self._audio_is_sharded and audio_args is not None:
                gathered_ax = self._gather_sequence(audio_args.x)
                a_et = audio_args.embedded_timestep
                if a_et.shape[1] == audio_args.x.shape[1]:
                    a_et = self._gather_sequence(a_et)
                audio_args = replace(
                    audio_args,
                    x=gathered_ax,
                    embedded_timestep=a_et,
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

    @staticmethod
    def _remap_transformer_block_keys_for_cache_dit_wrapper(weights: dict) -> dict:
        """Map checkpoint keys for wrapped blocks: insert inner. after transformer_blocks.<layer_idx>."""
        prefix = "transformer_blocks."
        out: dict = {}
        for key, value in weights.items():
            if not key.startswith(prefix):
                out[key] = value
                continue
            rest = key[len(prefix) :]
            first_dot = rest.find(".")
            if first_dot == -1:
                out[key] = value
                continue
            layer_idx_str = rest[:first_dot]
            if not layer_idx_str.isdigit():
                out[key] = value
                continue
            tail = rest[first_dot + 1 :]
            out[f"{prefix}{layer_idx_str}.inner.{tail}"] = value
        return out

    # -- Weight loading (from a single LTX-2 .safetensors checkpoint) -------------------------

    def load_weights(self, weights: dict) -> None:
        """Load checkpoint weights with key remapping.

        Handles naming differences between checkpoint and model:
          FFN:    ``ff.net.0.proj.*`` / ``ff.net.2.*`` → ``ff.up_proj.*`` / ``ff.down_proj.*``
          QKNorm: ``*.q_norm.*`` / ``*.k_norm.*``      → ``*.norm_q.*``   / ``*.norm_k.*``
        """
        remapped = {}
        for key, value in weights.items():
            new_key = key
            for ff_prefix in (".ff.", ".audio_ff."):
                if ff_prefix + "net.0.proj." in new_key:
                    new_key = new_key.replace(ff_prefix + "net.0.proj.", ff_prefix + "up_proj.")
                elif ff_prefix + "net.2." in new_key:
                    new_key = new_key.replace(ff_prefix + "net.2.", ff_prefix + "down_proj.")
            new_key = new_key.replace(".q_norm.", ".norm_q.")
            new_key = new_key.replace(".k_norm.", ".norm_k.")
            remapped[new_key] = value
        weights = remapped

        if self._uses_cache_dit():
            weights = self._remap_transformer_block_keys_for_cache_dit_wrapper(weights)

        target_dtype = self.model_config.torch_dtype if self.model_config else torch.bfloat16

        model_keys = {
            (name + "." + pname) if name else pname
            for name, mod in self.named_modules()
            for pname, p in mod._parameters.items()
            if p is not None
        }
        checkpoint_keys = set(weights.keys())

        # FUSE_QKV self-attention: model has qkv_proj, checkpoint has
        # to_q/to_k/to_v.  The weight loader fuses them via params_map.
        # Exclude these from mismatch warnings.
        fused_model_params = set()
        fused_ckpt_params = set()
        for name, mod in self.named_modules():
            if isinstance(mod, Linear):
                wlc = getattr(mod, "weights_loading_config", None)
                if wlc and getattr(wlc, "weight_mode", None) == WeightMode.FUSED_QKV_LINEAR:
                    parent = ".".join(name.split(".")[:-1])
                    for pname, p in mod._parameters.items():
                        if p is not None:
                            fused_model_params.add(f"{name}.{pname}")
                            for src in ("to_q", "to_k", "to_v"):
                                fused_ckpt_params.add(f"{parent}.{src}.{pname}")

        missing = (model_keys - checkpoint_keys) - fused_model_params
        unexpected = (checkpoint_keys - model_keys) - fused_ckpt_params
        quantized = (
            self.model_config is not None and self.model_config.quant_config.quant_algo is not None
        )
        dynamic_weight_quant = (
            self.model_config is not None and self.model_config.dynamic_weight_quant
        )
        if missing:
            logger.warning(
                f"LTXModel: {len(missing)} model params NOT in checkpoint: "
                f"{sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}"
            )
        if unexpected:
            logger.warning(
                f"LTXModel: {len(unexpected)} checkpoint keys NOT in model: "
                f"{sorted(unexpected)[:20]}{'...' if len(unexpected) > 20 else ''}"
            )
        loaded = model_keys & checkpoint_keys
        logger.info(
            f"LTXModel weight check: {len(loaded)} matched, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
        if quantized and missing:
            if dynamic_weight_quant:
                logger.info(
                    "Dynamic quantization is enabled -- missing scale parameters "
                    "(e.g. weight_scale, input_scale) are expected and will be "
                    "computed by DynamicLinearWeightLoader during weight loading."
                )
            else:
                logger.info(
                    "Pre-quantized checkpoint -- missing parameters "
                    "(e.g. alpha, inv_input_scale, kv_scales) are derived from "
                    "checkpoint scales during Linear.load_weights()."
                )

        for param_name, param in self._parameters.items():
            if param is not None and param_name in weights:
                param.data.copy_(weights[param_name].to(target_dtype))

        self._load_weights_trtllm(weights, target_dtype)

    def _load_weights_trtllm(self, weights: dict, target_dtype: torch.dtype) -> None:
        """TRT-LLM weight loading with dynamic quantization support."""
        params_map = {
            "qkv_proj": ["to_q", "to_k", "to_v"],
        }
        loader = DynamicLinearWeightLoader(self.model_config, params_map=params_map)

        for name, module in tqdm(self.named_modules(), desc="Loading LTXModel weights"):
            if len(module._parameters) == 0:
                continue

            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)
                if weight_dicts:
                    loader.load_linear_weights(module, name, weight_dicts)
            else:
                module_weights = loader.filter_weights(name, weights)
                for param_name, param in module._parameters.items():
                    if param is not None and param_name in module_weights:
                        param.data.copy_(module_weights[param_name].to(target_dtype))

    def post_load_weights(self) -> None:
        """Post-load hooks: finalize quantized Linear layers."""
        for _, module in self.named_modules():
            if isinstance(module, Linear) and hasattr(module, "post_load_weights"):
                module.post_load_weights()
