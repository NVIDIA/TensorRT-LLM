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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from tensorrt_llm._torch.modules.linear import Linear, WeightMode
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm._torch.visual_gen.utils import SequenceSharder
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
from .ltx2_core.utils_ltx2 import (
    apply_fused_adaln_modulate,
    apply_fused_gate_resid_rms_modulate,
    apply_fused_resid_gate_rms_quant,
    apply_fused_resid_rms_dual_shift_scale,
    apply_shift_scale,
    get_nvfp4_input_scale,
    rms_norm,
)
from .text_cache import TextCache

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
        enable_sequence_parallel: bool = False,
        use_ulysses: bool = False,
        async_ulysses: bool = False,
    ):
        from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

        config = config or DiffusionModelConfig()
        vgm = config.visual_gen_mapping

        # Store before super().__init__() — _init_qkv_proj needs _context_dim
        self._context_dim = context_dim if context_dim is not None else query_dim
        self.rope_type = rope_type
        self._is_cross_attn = context_dim is not None

        # Async ulysses opt-in: V/Q/K GEMMs interleave with the all-to-all on a
        # side stream. Forces SEPARATE_QKV so the 3 projections can issue
        # independently.
        self._use_async_ulysses = bool(
            use_ulysses
            and not self._is_cross_attn
            and async_ulysses
            and vgm is not None
            and vgm.ulysses_size > 1
        )

        # Self-attention: FUSE_QKV enables the optimized backend + auto Ulysses
        # wrapping from the base class.
        # Cross-attention or async ulysses: SEPARATE_QKV.
        if self._is_cross_attn or self._use_async_ulysses:
            qkv_mode = QKVMode.SEPARATE_QKV
        else:
            qkv_mode = QKVMode.FUSE_QKV

        # Caller opts in via enable_sequence_parallel. Cross-attn supports
        # Ulysses-only (SEPARATE_QKV + ring/attn2d is rejected in Attention);
        # when ring/attn2d CP is active we disable wrappers and fall back to
        # the plain backend + all-gather in the AV cross-attn forward path.
        ulysses_size = vgm.ulysses_size if vgm is not None else 1
        cp_size = vgm.cp_size if vgm is not None else 1
        if self._is_cross_attn:
            enable_sp = enable_sequence_parallel and cp_size == 1
        else:
            enable_sp = enable_sequence_parallel

        # Map LTX RoPE type to the fused-kernel INTERLEAVE template parameter:
        #   INTERLEAVED → pair (2i, 2i+1) pattern   → kernel INTERLEAVE=true
        #   SPLIT       → rotate-half pattern        → kernel INTERLEAVE=false
        # (cos/sin are stored block-duplicated for SPLIT; see _split_freqs_cis.)
        super().__init__(
            hidden_size=query_dim,
            num_attention_heads=heads,
            head_dim=dim_head,
            qkv_mode=qkv_mode,
            qk_norm=True,
            qk_norm_mode="full",
            eps=norm_eps,
            bias=True,
            interleave=(rope_type == LTXRopeType.INTERLEAVED),
            fuse_qk_norm_rope=True,
            config=config,
            layer_idx=layer_idx,
            enable_sequence_parallel=enable_sp,
            enable_ulysses=use_ulysses,
            async_ulysses=self._use_async_ulysses,
        )

        # Validate Ulysses head divisibility (from main).
        self._has_dual_attn = False
        if enable_sp and ulysses_size > 1:
            U = ulysses_size
            H = self.num_attention_heads
            H_kv = self.num_key_value_heads
            if H % U != 0 or H_kv % U != 0:
                raise ValueError(
                    f"Ulysses requires num_attention_heads ({H}) and "
                    f"num_key_value_heads ({H_kv}) divisible by ulysses_size ({U})"
                )
            # Base class already built `self.attn` as the Ulysses-wrapped path
            # (sharded inner backend + UlyssesAttention) for both self-attn and
            # cross-attn paths.

        # For audio self-attention that may need a runtime Ulysses toggle
        # (sequence length not always divisible by ulysses_size), create a
        # plain backend as fallback.  The base class already set self.attn
        # to UlyssesAttention(inner_backend=sharded_backend).
        if use_ulysses and not self._is_cross_attn and ulysses_size > 1:
            self._ulysses_attn = self.attn
            self._plain_attn = create_attention(
                backend=self.attn_backend,
                layer_idx=self.layer_idx,
                num_heads=H,
                head_dim=self.head_dim,
                num_kv_heads=H_kv,
                quant_config=self.quant_config,
                dtype=self.dtype,
                attention_config=config.attention,
                attention_metadata_state=config.attention_metadata_state,
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
        """Toggle between Ulysses-wrapped and plain attention at runtime.

        Effective for modules created with ``enable_sequence_parallel=True``
        (works for both self-attn and cross-attn). No-op otherwise.
        """
        if self._has_dual_attn:
            self._modules.pop("attn", None)
            self.attn = self._ulysses_attn if active else self._plain_attn

    def is_ulysses_active(self) -> bool:
        """Whether ``self.attn`` is currently the Ulysses-wrapped path.

        Symmetric with ``set_ulysses_active``. Returns False when no Ulysses
        pair was built (e.g. Attention2D mode, ulysses_size==1, or cross-attn
        without pure Ulysses), so callers can use it to decide whether to pass
        seq-sharded K/V (wrapper handles a2a) or to all-gather K/V into full
        sequence first (plain backend).
        """
        return self._has_dual_attn and self.attn is self._ulysses_attn

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
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project K/V from context, optionally apply RMSNorm + RoPE on K.

        Used by the project-before-gather pattern in AV cross-attention.
        When *pe* is given, RoPE is applied on the LOCAL K shard (Ulysses)
        before all-gather. RoPE is per-token element-wise so it commutes with
        seq-dim concat — bit-identical to the post-gather rope while saving
        the cos/sin all-gather collective and reducing K-rope compute by U×.
        The forward() consumer should pass ``k_pe=None`` to signal that K is
        already rotated.
        """
        k = self.to_k(context)
        v = self.to_v(context)

        # All cross-attn K-norm paths (with or without RoPE) go through the
        # split-fuse kernels. Fallback only kicks in for unsupported head_dim
        # — the fused kernel template covers {64, 128}; mini-config tests use
        # head_dim=32 and must take the eager branch.
        if self.qk_norm and self.head_dim in (64, 128):
            self.apply_split_norm_or_norm_rope(k, self.norm_k.weight, self.num_key_value_heads, pe)
        else:
            if self.qk_norm:
                k = self.norm_k(k)
            if pe is not None:
                k = apply_rotary_emb(k, pe, self.rope_type)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        pre_projected_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Caller contract:
          - FUSE_QKV (self-attn): pe must be set; k_pe and pre_projected_kv unused.
          - SEPARATE_QKV (cross-attn): cached path requires pre_projected_kv;
            uncached path uses ``context`` (may be None when the async-Ulysses
            inner backend was swapped to a non-async one — falls back to
            self-attn via kv_source=x). pe optional (None = norm-only).
            k_pe overrides pe for K (e.g. AV cross-attn) when provided.

        Args:
            key_padding_mask: Optional ``[B, S_kv]`` bool tensor; True = valid,
                False = pad. Forwarded through ``_attn_impl`` to the backend.
                Honored by ``VanillaAttention`` and ``FlashAttn4Attention``
                (and ``UlyssesAttention`` wrapping either); the TRTLLM backend
                silently ignores it. ``LTX2Attention`` constructs ``audio_attn1``
                with a VANILLA backend whenever Ulysses is active under a
                TRTLLM backend config (see ``_init_audio_modules``).

        Routing:
          1. Async-Ulysses self-attn → ``forward_async`` (V/Q/K rolling A2A).
          2. FUSE_QKV self-attn → packed fused kernel (or naive mini-config).
          3. SEPARATE_QKV cross-attn → split fused kernel (or naive mini-config).
        """
        # Async-Ulysses self-attn dispatch. ``hasattr`` guard: audio_attn1 may
        # have ``set_ulysses_active(False)`` swap ``self.attn`` to a plain
        # backend that lacks ``forward_async`` — fall through to the sync
        # uncached SEPARATE_QKV branch, which handles context=None via
        # kv_source=x.
        if (
            self.qkv_mode == QKVMode.SEPARATE_QKV
            and self._use_async_ulysses
            and context is None
            and pre_projected_kv is None
            and hasattr(self.attn, "forward_async")
        ):
            return self.forward_async(x, freqs=pe)

        # Fused gate: prod uses fused kernels (head_dim ∈ {64, 128}); mini-config
        # tests (head_dim=32) fall to naive ops.
        use_fused = self.fuse_qk_norm_rope and self.head_dim in (64, 128) and self.qk_norm

        if self.qkv_mode == QKVMode.FUSE_QKV:
            # ─── sync self-attn ───
            if use_fused and pe is not None:
                # Fused packed kernel: norm + RoPE on QKV in-place.
                qkv = self.qkv_proj(x)
                cos, sin = pe
                self.apply_packed_qk_norm_rope(qkv, cos, sin)
                q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
            else:
                # Naive (mini-config head_dim ∉ {64, 128}).
                q, k, v = self.get_qkv(x)
                if self.qk_norm:
                    q = self.norm_q(q)
                    k = self.norm_k(k)
                if pe is not None:
                    q = apply_rotary_emb(q, pe, self.rope_type)
                    k = apply_rotary_emb(k, pe, self.rope_type)

        elif self.qkv_mode == QKVMode.SEPARATE_QKV:
            if pre_projected_kv is not None:
                # ─── cached cross-attn (text + AV cross-attn) ───
                # K/V cached by caller; we only norm+RoPE Q here.
                k, v = pre_projected_kv
                q = self.to_q(x)
                if use_fused:
                    self.apply_split_norm_or_norm_rope(
                        q, self.norm_q.weight, self.num_attention_heads, pe
                    )
                else:
                    if self.qk_norm:
                        q = self.norm_q(q)
                    if pe is not None:
                        q = apply_rotary_emb(q, pe, self.rope_type)
            else:
                # ─── uncached cross-attn / async self-attn fallback ───
                # LTX-2 prod doesn't use uncached cross-attn (always pre-projects
                # K/V). This branch also catches async self-attn when the inner
                # backend lacks forward_async (audio Ulysses-inactive swap):
                # context=None then, fall back to self-attn via kv_source=x.
                kv_source = context if context is not None else x
                q = self.to_q(x)
                k = self.to_k(kv_source)
                v = self.to_v(kv_source)
                if use_fused:
                    self.apply_split_norm_or_norm_rope(
                        q, self.norm_q.weight, self.num_attention_heads, pe
                    )
                    self.apply_split_norm_or_norm_rope(
                        k,
                        self.norm_k.weight,
                        self.num_key_value_heads,
                        k_pe if k_pe is not None else pe,
                    )
                else:
                    if self.qk_norm:
                        q = self.norm_q(q)
                        k = self.norm_k(k)
                    if pe is not None:
                        q = apply_rotary_emb(q, pe, self.rope_type)
                    k_pe_use = k_pe if k_pe is not None else pe
                    if k_pe_use is not None:
                        k = apply_rotary_emb(k, k_pe_use, self.rope_type)

        attn_kwargs = {}
        if key_padding_mask is not None:
            attn_kwargs["key_padding_mask"] = key_padding_mask
        out = self._attn_impl(q, k, v, **attn_kwargs)

        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, self.num_attention_heads, self.head_dim)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.num_attention_heads * self.head_dim)

        return self.to_out[0](out)

    def forward_async(
        self,
        x: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """LTX-2 async-Ulysses self-attn driver. Structurally mirrors base
        ``Attention.forward_async`` (single function, fused/unfused branches)
        but uses LTX-2's ``apply_rotary_emb`` (with ``rope_type``) on the
        unfused fallback and injects gated-attention scaling in 4D between
        the attn output and ``to_out``.

        Precondition: caller in ``LTX2Attention.forward`` gates on
        ``_use_async_ulysses`` + ``hasattr(self.attn, "forward_async")``.

        Returns 3D ``[B, S, H*D]`` matching ``forward``'s output contract.
        """
        B, S, _ = x.shape
        H = self.num_attention_heads
        KV = self.num_key_value_heads
        D = self.head_dim
        # Mirrors LTX2Attention.forward's fused gate; qkv_mode is implicitly
        # SEPARATE_QKV under async (caller-enforced). head_dim check matches
        # the fused split kernel's HEAD_DIM template instantiations {64, 128}.
        use_fused = (
            self.fuse_qk_norm_rope
            and self.head_dim in (64, 128)
            and freqs is not None
            and self.qk_norm
        )

        # SEPARATE_QKV self-attn 3x fp4_quantize dedup; see Attention.forward_async.
        if self._maybe_share_qkv_quantize and getattr(self.to_q, "input_scale", None) is not None:
            x_2d = x.reshape(-1, x.shape[-1])
            fp4, sf = torch.ops.trtllm.tunable_fp4_quantize(
                x_2d, self.to_q.input_scale, self.to_q.scaling_vector_size, False
            )
            qkv_input = Fp4QuantizedTensor(fp4, sf, is_sf_swizzled=False)
        else:
            qkv_input = x

        def compute_q():
            q = self.to_q(qkv_input)
            if q.dim() == 2:
                q = q.view(B, S, -1)
            if use_fused:
                self.apply_split_norm_rope(q, self.norm_q.weight, H, freqs[0], freqs[1])
                return q.view(B, S, H, D)
            # Unfused fallback (mini-config); LTX-2 RoPE with rope_type.
            if self.qk_norm:
                q = self.norm_q(q)
            q = q.view(B, S, H, D)
            if freqs is not None:
                q = apply_rotary_emb(q, freqs, self.rope_type)
            return q

        def compute_k():
            k = self.to_k(qkv_input)
            if k.dim() == 2:
                k = k.view(B, S, -1)
            if use_fused:
                self.apply_split_norm_rope(k, self.norm_k.weight, KV, freqs[0], freqs[1])
                return k.view(B, S, KV, D)
            if self.qk_norm:
                k = self.norm_k(k)
            k = k.view(B, S, KV, D)
            if freqs is not None:
                k = apply_rotary_emb(k, freqs, self.rope_type)
            return k

        def compute_v():
            return self.to_v(qkv_input).view(B, S, KV, D)

        out_4d = self.attn.forward_async(compute_q, compute_k, compute_v)

        # LTX-2 gated-attention scaling in 4D before to_out.
        if self.to_gate_logits is not None:
            gates = 2.0 * torch.sigmoid(self.to_gate_logits(x))
            out_4d = out_4d * gates.unsqueeze(-1)

        b, t = out_4d.shape[:2]
        return self.to_out[0](out_4d.reshape(b, t, H * D))


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

        # Per-block sharder: mirrors the root sharder's topology so the block
        # can run cross-attention all-gathers independently.  Head-divisibility
        # is checked once at the root model — skip num_heads here.
        vgm = config.visual_gen_mapping if config is not None else None
        self._sharder = SequenceSharder.from_vgm(vgm)
        self._audio_is_sharded = False

        # Whether to dispatch AdaLN modulation to the fused CUDA kernels. Resolved
        # once at construction; call sites just consult the flag. The kernels are
        # bf16 + hidden_dim in {2048, 4096}; non-matching cases raise at the C++
        # boundary -- this flag is the only Python-side guard.
        from .ltx2_core.utils_ltx2 import is_fused_adaln_supported_dim

        v_ok = video is None or is_fused_adaln_supported_dim(video.dim)
        a_ok = audio is None or is_fused_adaln_supported_dim(audio.dim)
        self._fuse_adaln = v_ok and a_ok
        # BENCH HOOK (revert before commit): allow forcing eager fallback via env var
        # for A/B e2e timing comparison.
        import os as _os_bench

        if _os_bench.environ.get("TLLM_DISABLE_FUSE_ADALN") == "1":
            self._fuse_adaln = False

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
        _async_ulysses = model_config.parallel.async_ulysses if model_config is not None else False
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
            enable_sequence_parallel=True,
            use_ulysses=True,
            async_ulysses=_async_ulysses,
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
            enable_sequence_parallel=False,
        )
        self.ff = self._make_mlp(cfg, model_config, idx)
        self.scale_shift_table = nn.Parameter(torch.empty(6, cfg.dim))

    def _init_audio_modules(self, cfg, rope_type, eps, model_config, idx):
        # Audio under Ulysses needs key_padding_mask support on audio_attn1
        # (audio is padded to be divisible by ulysses_size; mask zeros pad
        # slots). TRTLLM self-attn silently drops key_padding_mask, so downgrade
        # to VANILLA whenever Ulysses is active under a TRTLLM backend config.
        # Mirrors the existing cross-attn TRTLLM→VANILLA fallback
        # (modules/attention.py); audio is small (T_a ~ 126) so the downgrade
        # is negligible.
        audio_self_config = model_config
        vgm = model_config.visual_gen_mapping
        ulysses_size = vgm.ulysses_size if vgm is not None else 1
        if ulysses_size > 1 and model_config.attention.backend == "TRTLLM":
            audio_self_config = model_config.model_copy(
                update={
                    "attention": model_config.attention.model_copy(update={"backend": "VANILLA"})
                }
            )
        self.audio_attn1 = LTX2Attention(
            query_dim=cfg.dim,
            heads=cfg.heads,
            dim_head=cfg.d_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=cfg.apply_gated_attention,
            config=audio_self_config,
            layer_idx=idx,
            enable_sequence_parallel=True,
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
            enable_sequence_parallel=False,
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
            enable_sequence_parallel=False,
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
            enable_sequence_parallel=True,
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
        """Combined-form AdaLN values for the slots in ``indices``. Returns one bf16
        ``[batch_size, T, D]`` tensor per slot (broadcast-add of ``scale_shift_table``
        cast to bf16 + ``timestep`` reshaped to per-slot views, then unbound on the
        slot axis).
        """
        num_ada_params = scale_shift_table.shape[0]
        return (
            scale_shift_table[indices]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)

    @staticmethod
    def _get_ada_table_ts_pairs(
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Pair-form companion to ``_get_ada_values``: returns one ``(table_slice, ts_slice)``
        pair per slot in ``indices`` without materializing the broadcast-add. Consumers that
        fuse the add into a downstream kernel (Phase 0b of the fused C++ ops) accept the pair
        directly; the broadcast-add then becomes dead code and is DCE'd by Inductor when no
        combined-form slot from the same range is consumed elsewhere.

        Returns a tuple of (table_slice fp32 [D], ts_slice bf16 [B, T, D]) per slot.
        """
        num_ada_params = scale_shift_table.shape[0]
        ts_reshaped = timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)
        return tuple(
            (scale_shift_table[i], ts_reshaped[:, :, i, :])
            for i in range(*indices.indices(num_ada_params))
        )

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
        return self._sharder.gather(x, dim=dim)

    # -- Forward -------------------------------------------------------------

    def forward(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations=None,
        text_kv_video: tuple[torch.Tensor, torch.Tensor] | None = None,
        text_kv_audio: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """Forward with optional perturbation masking for STG.

        Args:
            perturbations: Optional ``BatchedPerturbationConfig`` that masks
                attention outputs for selected blocks/modalities.
            text_kv_video: Pre-projected (K, V) for video text cross-attention.
                Falls back to inline computation if ``None``.
            text_kv_audio: Pre-projected (K, V) for audio text cross-attention.
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

        # When the AV cross-attn block will run AND fused AdaLN is active, we
        # defer the text-cross-attn residual on V/A into the fused KB kernel
        # (residual + RMSNorm + dual shift_scale). Otherwise the residual is
        # added in place immediately as before.
        defer_text_residual = (run_a2v or run_v2a) and self._fuse_adaln
        text_v_attn_raw = None
        text_a_attn_raw = None
        # Raw cross-attn outputs deferred into KA at Sites 7/8. Set inside the
        # AV cross-attn block when defer_*_into_ka is True; consumed by the FFN
        # block. Init here so they're in scope even when the AV block is skipped.
        a2v_attn_raw = None
        v2a_attn_raw = None

        # --- Video self-attention + text cross-attention ---
        if run_vx:
            skip_v_self = has_perturbations and perturbations.all_in_batch(
                PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx
            )
            if not skip_v_self:
                # MSA modulators in pair form: slot 0 = shift_msa, 1 = scale_msa, 2 = gate_msa.
                (
                    (vshift_msa_table, vshift_msa_ts),
                    (vscale_msa_table, vscale_msa_ts),
                    (vgate_msa_table, vgate_msa_ts),
                ) = self._get_ada_table_ts_pairs(
                    self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
                )
                norm_vx = apply_fused_adaln_modulate(
                    vx,
                    vscale_msa_table,
                    vscale_msa_ts,
                    vshift_msa_table,
                    vshift_msa_ts,
                    self.norm_eps,
                    self._fuse_adaln,
                    fp4_input_scale=get_nvfp4_input_scale(self.attn1.qkv_proj),
                )
                v_attn_raw = self.attn1(norm_vx, pe=video.positional_embeddings)
                if has_perturbations and perturbations.any_in_batch(
                    PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx
                ):
                    # Mask commutes with the gate mul: (attn * mask) * gate == attn * (gate * mask).
                    v_attn_raw = v_attn_raw * perturbations.mask_like(
                        PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, v_attn_raw
                    )
                # KD fuses vx <- vx + v_attn_raw * gate_msa; rms_norm; optional FP4 quant.
                vx, attn2_q_input = apply_fused_resid_gate_rms_quant(
                    vx,
                    v_attn_raw,
                    vgate_msa_table,
                    vgate_msa_ts,
                    self.norm_eps,
                    self._fuse_adaln,
                    fp4_input_scale=get_nvfp4_input_scale(self.attn2.to_q),
                )
            else:
                attn2_q_input = rms_norm(vx, eps=self.norm_eps)
            _v_attn2_out = self.attn2(
                attn2_q_input,
                context=video.context,
                pre_projected_kv=text_kv_video,
            )
            if defer_text_residual:
                text_v_attn_raw = _v_attn2_out
            else:
                vx = vx + _v_attn2_out

        # --- Audio self-attention + text cross-attention ---
        if run_ax:
            skip_a_self = has_perturbations and perturbations.all_in_batch(
                PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx
            )
            if not skip_a_self:
                # MSA modulators in pair form: slot 0 = shift_msa, 1 = scale_msa, 2 = gate_msa.
                (
                    (ashift_msa_table, ashift_msa_ts),
                    (ascale_msa_table, ascale_msa_ts),
                    (agate_msa_table, agate_msa_ts),
                ) = self._get_ada_table_ts_pairs(
                    self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
                )
                norm_ax = apply_fused_adaln_modulate(
                    ax,
                    ascale_msa_table,
                    ascale_msa_ts,
                    ashift_msa_table,
                    ashift_msa_ts,
                    self.norm_eps,
                    self._fuse_adaln,
                    fp4_input_scale=get_nvfp4_input_scale(self.audio_attn1.qkv_proj),
                )
                a_attn_raw = self.audio_attn1(
                    norm_ax,
                    pe=audio.positional_embeddings,
                    key_padding_mask=audio.audio_padding_mask,
                )
                if has_perturbations and perturbations.any_in_batch(
                    PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx
                ):
                    a_attn_raw = a_attn_raw * perturbations.mask_like(
                        PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, a_attn_raw
                    )
                # KD: fused ax <- ax + a_attn_raw * gate_msa; rms_norm; optional FP4 quant.
                ax, audio_attn2_q_input = apply_fused_resid_gate_rms_quant(
                    ax,
                    a_attn_raw,
                    agate_msa_table,
                    agate_msa_ts,
                    self.norm_eps,
                    self._fuse_adaln,
                    fp4_input_scale=get_nvfp4_input_scale(self.audio_attn2.to_q),
                )
            else:
                audio_attn2_q_input = rms_norm(ax, eps=self.norm_eps)
            _a_attn2_out = self.audio_attn2(
                audio_attn2_q_input,
                context=audio.context,
                pre_projected_kv=text_kv_audio,
            )
            if defer_text_residual:
                text_a_attn_raw = _a_attn2_out
            else:
                ax = ax + _a_attn2_out

        # --- Bidirectional audio ↔ video cross-attention ---
        if run_a2v or run_v2a:
            skip_a2v = has_perturbations and perturbations.all_in_batch(
                PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx
            )
            skip_v2a = has_perturbations and perturbations.all_in_batch(
                PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx
            )

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

            # KB fuses (text_attn residual + RMSNorm + dual shift_scale) per modality.
            # Fallback when residual was already added (text_*_attn_raw is None):
            # do RMSNorm + two individual shift_scale calls.
            if text_v_attn_raw is not None:
                # KB fuses residual + RMS + dual shift_scale. Pass NVFP4 input_scales
                # to also emit packed FP4 + 128x4 SWIZZLED SF for the downstream
                # cross-attn Q/K projections (the wrapper handles None vs not-None
                # uniformly and falls back gracefully when fuse=False).
                # Pair form for the 4 ss modulators: AV CA video table[:4]
                # paired with cross_scale_shift_timestep (4 slots in last dim).
                v_ss_table = self.scale_shift_table_a2v_ca_video[:4]
                (
                    (v_scale_a2v_table, v_scale_a2v_ts),
                    (v_shift_a2v_table, v_shift_a2v_ts),
                    (v_scale_v2a_table, v_scale_v2a_ts),
                    (v_shift_v2a_table, v_shift_v2a_ts),
                ) = self._get_ada_table_ts_pairs(
                    v_ss_table, vx.shape[0], video.cross_scale_shift_timestep, slice(0, 4)
                )
                vx, vx_scaled_a2v, vx_scaled_v2a = apply_fused_resid_rms_dual_shift_scale(
                    vx,
                    text_v_attn_raw,
                    v_scale_a2v_table,
                    v_scale_a2v_ts,
                    v_shift_a2v_table,
                    v_shift_a2v_ts,
                    v_scale_v2a_table,
                    v_scale_v2a_ts,
                    v_shift_v2a_table,
                    v_shift_v2a_ts,
                    self.norm_eps,
                    fuse=self._fuse_adaln,
                    fp4_input_scale1=get_nvfp4_input_scale(self.audio_to_video_attn.to_q),
                    fp4_input_scale2=get_nvfp4_input_scale(self.video_to_audio_attn.to_k),
                )
                text_v_attn_raw = None
            else:
                vx_norm3 = rms_norm(vx, eps=self.norm_eps)
                vx_scaled_a2v = apply_shift_scale(
                    vx_norm3, scale_ca_video_a2v, shift_ca_video_a2v, self._fuse_adaln
                )
                vx_scaled_v2a = apply_shift_scale(
                    vx_norm3, scale_ca_video_v2a, shift_ca_video_v2a, self._fuse_adaln
                )

            if text_a_attn_raw is not None:
                # KB dual for audio side (a2v consumes ax K side, v2a consumes ax Q side).
                # Pair form for the 4 ss modulators: AV CA audio table[:4]
                # paired with cross_scale_shift_timestep.
                a_ss_table = self.scale_shift_table_a2v_ca_audio[:4]
                (
                    (a_scale_a2v_table, a_scale_a2v_ts),
                    (a_shift_a2v_table, a_shift_a2v_ts),
                    (a_scale_v2a_table, a_scale_v2a_ts),
                    (a_shift_v2a_table, a_shift_v2a_ts),
                ) = self._get_ada_table_ts_pairs(
                    a_ss_table, ax.shape[0], audio.cross_scale_shift_timestep, slice(0, 4)
                )
                ax, ax_scaled_a2v, ax_scaled_v2a = apply_fused_resid_rms_dual_shift_scale(
                    ax,
                    text_a_attn_raw,
                    a_scale_a2v_table,
                    a_scale_a2v_ts,
                    a_shift_a2v_table,
                    a_shift_a2v_ts,
                    a_scale_v2a_table,
                    a_scale_v2a_ts,
                    a_shift_v2a_table,
                    a_shift_v2a_ts,
                    self.norm_eps,
                    fuse=self._fuse_adaln,
                    fp4_input_scale1=get_nvfp4_input_scale(self.audio_to_video_attn.to_k),
                    fp4_input_scale2=get_nvfp4_input_scale(self.video_to_audio_attn.to_q),
                )
                text_a_attn_raw = None
            else:
                ax_norm3 = rms_norm(ax, eps=self.norm_eps)
                ax_scaled_a2v = apply_shift_scale(
                    ax_norm3, scale_ca_audio_a2v, shift_ca_audio_a2v, self._fuse_adaln
                )
                ax_scaled_v2a = apply_shift_scale(
                    ax_norm3, scale_ca_audio_v2a, shift_ca_audio_v2a, self._fuse_adaln
                )

            # KA fuses the LAST cross-attn residual (a2v on V, v2a on A) with the
            # downstream FFN pre-norm + modulate. Eligible only when fused AdaLN is
            # on, that cross-attn direction actually runs (and is not fully skipped),
            # and no per-batch perturbation mask is applied between attn*gate and
            # residual (mask path requires an extra elementwise mul that KA can't fuse).
            defer_a2v_into_ka = (
                run_a2v
                and not skip_a2v
                and self._fuse_adaln
                and not (
                    has_perturbations
                    and perturbations.any_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx)
                )
            )
            defer_v2a_into_ka = (
                run_v2a
                and not skip_v2a
                and self._fuse_adaln
                and not (
                    has_perturbations
                    and perturbations.any_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx)
                )
            )
            if run_a2v and not skip_a2v:
                # Project-before-gather: K/V projections run on sharded data
                # so they benefit from Ulysses scaling.  RoPE is applied to K
                # inside project_kv on the sharded shard (RoPE commutes with
                # seq-dim concat), so the cos/sin all-gather is unneeded and K
                # rope work is U× cheaper. a2v keeps the all-gather path
                # (Q=video huge, K/V=audio small — AG of the small audio is
                # far cheaper than full-video a2a collectives).
                # key_padding_mask zeros attention on the audio pad slots that
                # configure_audio_ulysses appended to make T_a divisible by U.
                k_a2v, v_a2v = self.audio_to_video_attn.project_kv(
                    ax_scaled_a2v, pe=audio.cross_positional_embeddings
                )
                if self._audio_is_sharded:
                    k_a2v = self._sp_all_gather(k_a2v)
                    v_a2v = self._sp_all_gather(v_a2v)

                _a2v_attn = self.audio_to_video_attn(
                    vx_scaled_a2v,
                    pre_projected_kv=(k_a2v, v_a2v),
                    pe=video.cross_positional_embeddings,
                    k_pe=None,  # K already rotated in project_kv
                    key_padding_mask=audio.audio_padding_mask,
                )
                if defer_a2v_into_ka:
                    # Defer gate*attn + residual into KA at Site 7.
                    a2v_attn_raw = _a2v_attn
                else:
                    a2v_out = _a2v_attn * gate_out_a2v
                    if has_perturbations and perturbations.any_in_batch(
                        PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx
                    ):
                        a2v_out = a2v_out * perturbations.mask_like(
                            PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, a2v_out
                        )
                    vx = vx + a2v_out

            if run_v2a and not skip_v2a:
                # v2a: when the Ulysses wrapper is active, K/V (video, large)
                # stay seq-sharded and the wrapper handles Q + K|V + output
                # a2a internally. RoPE is applied to K in project_kv on the
                # local shard (commutes with a2a along the seq dim, so
                # rotate-before-gather is value-preserving). No
                # key_padding_mask — video K/V is unpadded; padded audio Q is
                # stripped on exit by LTXModel.forward. When inactive (no
                # wrapper built, Stage 2 disable, or audio not sharded), fall
                # back to AG so the plain backend sees full K/V. Gate on
                # is_ulysses_active() — _audio_is_sharded can be true under
                # Attention2D where no wrapper was built.
                k_v2a, v_v2a = self.video_to_audio_attn.project_kv(
                    vx_scaled_v2a, pe=video.cross_positional_embeddings
                )
                if not self.video_to_audio_attn.is_ulysses_active() and self._sharder.is_active:
                    # Fallback: wrapper inactive → all-gather sharded video
                    # K/V to full so plain backend can run.
                    k_v2a = self._sp_all_gather(k_v2a)
                    v_v2a = self._sp_all_gather(v_v2a)

                _v2a_attn = self.video_to_audio_attn(
                    ax_scaled_v2a,
                    pre_projected_kv=(k_v2a, v_v2a),
                    pe=audio.cross_positional_embeddings,
                    k_pe=None,  # K already rotated in project_kv
                )
                if defer_v2a_into_ka:
                    v2a_attn_raw = _v2a_attn
                else:
                    v2a_out = _v2a_attn * gate_out_v2a
                    if has_perturbations and perturbations.any_in_batch(
                        PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx
                    ):
                        v2a_out = v2a_out * perturbations.mask_like(
                            PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, v2a_out
                        )
                    ax = ax + v2a_out

        # --- Video FFN ---
        if run_vx:
            # MLP modulators: slot 3 (shift_mlp), 4 (scale_mlp) feed the fused kernel as
            # pair form; slot 5 (gate_mlp) is consumed as a bf16 tensor by the elementwise
            # gate-mul below.
            (
                (vshift_mlp_table, vshift_mlp_ts),
                (vscale_mlp_table, vscale_mlp_ts),
            ) = self._get_ada_table_ts_pairs(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 5)
            )
            (vgate_mlp,) = self._get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(5, 6)
            )
            if a2v_attn_raw is not None:
                # KC: vx <- vx + a2v_attn_raw*gate; rms_norm; (1+scale)*normed+shift.
                # Gate comes from AV CA table[4]+cross_gate_timestep.
                v_gate_a2v_table = self.scale_shift_table_a2v_ca_video[4]
                v_gate_a2v_ts = video.cross_gate_timestep
                vx, vx_scaled = apply_fused_gate_resid_rms_modulate(
                    vx,
                    a2v_attn_raw,
                    v_gate_a2v_table,
                    v_gate_a2v_ts,
                    vscale_mlp_table,
                    vscale_mlp_ts,
                    vshift_mlp_table,
                    vshift_mlp_ts,
                    self.norm_eps,
                    fuse=self._fuse_adaln,
                    fp4_input_scale=get_nvfp4_input_scale(self.ff.up_proj),
                )
                a2v_attn_raw = None
            else:
                # No media cross-attn residual to consume: pure RMSNorm + modulate via KA.
                vx_scaled = apply_fused_adaln_modulate(
                    vx,
                    vscale_mlp_table,
                    vscale_mlp_ts,
                    vshift_mlp_table,
                    vshift_mlp_ts,
                    self.norm_eps,
                    self._fuse_adaln,
                    fp4_input_scale=get_nvfp4_input_scale(self.ff.up_proj),
                )
            vx = vx + self.ff(vx_scaled) * vgate_mlp

        # --- Audio FFN ---
        if run_ax:
            (
                (ashift_mlp_table, ashift_mlp_ts),
                (ascale_mlp_table, ascale_mlp_ts),
            ) = self._get_ada_table_ts_pairs(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 5)
            )
            (agate_mlp,) = self._get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(5, 6)
            )
            if v2a_attn_raw is not None:
                # KC: ax <- ax + v2a_attn_raw*gate; rms_norm; (1+scale)*normed+shift.
                # Gate comes from AV CA audio table[4]+cross_gate_timestep.
                a_gate_v2a_table = self.scale_shift_table_a2v_ca_audio[4]
                a_gate_v2a_ts = audio.cross_gate_timestep
                ax, ax_scaled = apply_fused_gate_resid_rms_modulate(
                    ax,
                    v2a_attn_raw,
                    a_gate_v2a_table,
                    a_gate_v2a_ts,
                    ascale_mlp_table,
                    ascale_mlp_ts,
                    ashift_mlp_table,
                    ashift_mlp_ts,
                    self.norm_eps,
                    fuse=self._fuse_adaln,
                    fp4_input_scale=get_nvfp4_input_scale(self.audio_ff.up_proj),
                )
                v2a_attn_raw = None
            else:
                ax_scaled = apply_fused_adaln_modulate(
                    ax,
                    ascale_mlp_table,
                    ascale_mlp_ts,
                    ashift_mlp_table,
                    ashift_mlp_ts,
                    self.norm_eps,
                    self._fuse_adaln,
                    fp4_input_scale=get_nvfp4_input_scale(self.audio_ff.up_proj),
                )
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
        # Cache-DiT's CachedBlocks iterates all inner blocks with the same kwargs,
        # so looking up per-block text_kv via inner.idx is required — passing
        # text_kv via the outer loop would feed every block block-0's KV.
        idx = inner_mod.idx
        v_kv_list = getattr(p, "_cache_dit_v_kv", None)
        a_kv_list = getattr(p, "_cache_dit_a_kv", None)
        kwargs.setdefault("text_kv_video", v_kv_list[idx] if v_kv_list else None)
        kwargs.setdefault("text_kv_audio", a_kv_list[idx] if a_kv_list else None)
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


class LTXModel(BaseDiffusionModel):
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
        from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

        model_config = model_config or DiffusionModelConfig()
        super().__init__(model_config)
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

        # Fused AdaLN modulation flag: True iff every inner_dim in this model
        # matches the kernel's supported set. Threaded into output-head paths.
        from .ltx2_core.utils_ltx2 import is_fused_adaln_supported_dim

        v_ok = (not model_type.is_video_enabled()) or is_fused_adaln_supported_dim(self.inner_dim)
        a_ok = (not model_type.is_audio_enabled()) or is_fused_adaln_supported_dim(
            self.audio_inner_dim
        )
        self._fuse_adaln = v_ok and a_ok
        # BENCH HOOK (revert before commit): see BasicAVTransformerBlock.__init__.
        import os as _os_bench

        if _os_bench.environ.get("TLLM_DISABLE_FUSE_ADALN") == "1":
            self._fuse_adaln = False

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(
                self.positional_embedding_max_pos[0],
                self.audio_positional_embedding_max_pos[0],
            )
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)

        vgm = model_config.visual_gen_mapping
        # Validate video-head divisibility through the factory; audio heads
        # are checked separately because the factory only accepts one count.
        self._sharder = SequenceSharder.from_vgm(
            vgm,
            num_attention_heads=num_attention_heads if model_type.is_video_enabled() else None,
        )
        self._cp_size = vgm.cp_size if vgm is not None else 1
        self._ulysses_size = vgm.ulysses_size if vgm is not None else 1
        if (
            self._sharder.is_active
            and vgm is not None
            and vgm.ulysses_size > 1
            and model_type.is_audio_enabled()
            and audio_num_attention_heads % vgm.ulysses_size != 0
        ):
            raise ValueError(
                f"audio_num_attention_heads ({audio_num_attention_heads}) "
                f"must be divisible by ulysses_size ({vgm.ulysses_size})"
            )

        if self.model_config.mapping.tp_size > 1:
            raise ValueError("LTX2 does not currently support TP.")

        self._audio_is_sharded = False
        self._audio_pad = 0  # set by configure_audio_ulysses
        self._cache_dit_video_args: Optional[TransformerArgs] = None
        self._cache_dit_audio_args: Optional[TransformerArgs] = None
        # Per-block text cross-attn KV lists, looked up by inner.idx in the
        # Pattern_0 wrapper. Set once per forward in the Cache-DiT path.
        self._cache_dit_v_kv: Optional[list] = None
        self._cache_dit_a_kv: Optional[list] = None

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

    # -- Sequence sharding / gathering ----------------------------------------

    def _shard_transformer_args(self, args: TransformerArgs) -> TransformerArgs:
        """Shard step-dependent fields of *args* across sequence-parallel ranks.

        PE (``positional_embeddings`` / ``cross_positional_embeddings``) is
        already sharded-local in ``TextCache`` (one-time in
        ``prepare_text_cache``) so we leave it untouched. Only step-varying
        fields (``x``, timesteps, etc.) need slicing each step.
        """
        seq_len = args.x.shape[1]
        sh = self._sharder
        return replace(
            args,
            x=sh.shard(args.x, dim=1),
            timesteps=sh.shard(args.timesteps, dim=1, expected_seq_len=seq_len),
            embedded_timestep=sh.shard(args.embedded_timestep, dim=1, expected_seq_len=seq_len),
            cross_scale_shift_timestep=sh.shard(
                args.cross_scale_shift_timestep, dim=1, expected_seq_len=seq_len
            ),
            cross_gate_timestep=sh.shard(args.cross_gate_timestep, dim=1, expected_seq_len=seq_len),
        )

    def _make_pe_local(
        self,
        pe: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        is_audio: bool,
        fuse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sharded-local PE for the attention consumer.

        Slices the source 4D PE along seq dim by Ulysses rank (one-time, in
        ``prepare_text_cache``), then either reshapes to 2D ``[T_local, H*D]``
        for the fused kernel or keeps 4D for the eager apply_rotary_emb path.
        LTX-2 SPLIT rope produces 4D PE; INTERLEAVED is not used in prod.

        ``_audio_is_sharded`` (set in ``configure_audio_ulysses``) already
        encodes whether audio_seq_len is divisible by ulysses_size, so we
        gate sharding on that flag alone — no second divisibility check.
        """
        if pe is None:
            return None
        cos, sin = pe
        sh = self._sharder
        if sh.is_active and (not is_audio or self._audio_is_sharded):
            chunk = cos.shape[1] // sh.size
            s = sh.rank * chunk
            e = s + chunk
            cos = cos[:, s:e]
            sin = sin[:, s:e]
        cos = cos.contiguous()
        sin = sin.contiguous()
        if fuse:
            # [B, T_local, H, D] -> [B*T_local, H*D]. PE source from
            # precompute_freqs_cis has B=1 so this collapses to [T_local, H*D];
            # the fused kernel broadcasts cos over B internally.
            cos = cos.reshape(cos.shape[0] * cos.shape[1], -1)
            sin = sin.reshape(sin.shape[0] * sin.shape[1], -1)
        return (cos, sin)

    def _gather_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather hidden states along the sequence dim."""
        return self._sharder.gather(x, dim=1)

    @staticmethod
    def _pad_pe(
        pe: tuple[torch.Tensor, torch.Tensor] | None,
        pad: int,
        seq_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Repeat-last pad a (cos, sin) PE tuple by ``pad`` slots on ``seq_dim``.

        Used by ``prepare_text_cache`` to extend audio PE when audio padding
        is active, so the rope helper sees consistent shapes between padded
        audio input and PE. Repeat-last keeps RoPE values at pad slots equal
        to the last valid token (no OOB positional index).

        Caller passes ``seq_dim`` explicitly (depends on rope type and PE
        layout): SPLIT rope = ``[B, H, T, D]`` so 2; INTERLEAVED = ``[B, T, D]``
        so 1.
        """
        if pe is None or pad <= 0:
            return pe
        cos, sin = pe
        if not (0 <= seq_dim < cos.ndim):
            raise ValueError(f"_pad_pe: seq_dim={seq_dim} out of range for cos.ndim={cos.ndim}")

        def _ext(t: torch.Tensor) -> torch.Tensor:
            idx = [slice(None)] * t.ndim
            idx[seq_dim] = slice(t.shape[seq_dim] - 1, t.shape[seq_dim])
            last = t[tuple(idx)]
            tail_shape = list(t.shape)
            tail_shape[seq_dim] = pad
            tail = last.expand(*tail_shape).contiguous()
            return torch.cat([t, tail], dim=seq_dim)

        return (_ext(cos), _ext(sin))

    @staticmethod
    def _pad_modality_audio(audio: Modality, pad: int) -> Modality:
        """Pad ``audio`` on the token axis by ``pad`` slots.

        Used by ``forward`` when Ulysses is active and ``T_a % ulysses_size
        != 0`` to make audio shardable.

        - ``latent``: zero-pad on ``dim=1``. Attention zeros out pad rows via
          ``audio_padding_mask``; norm/MLP on zero rows is harmless because
          pad rows are stripped before output processing.
        - ``positions``: repeat-last on ``dim=2`` (works for both 3D
          ``(B, n_dims, T)`` and 4D ``(B, n_dims, T, 2)`` shapes). RoPE
          cos/sin at pad slots equals the last valid token's; keeps RoPE
          indices inside the model's trained range regardless of
          ``positional_embedding_max_pos``.
        - ``timesteps``: repeat-last on ``dim=1`` only when per-token
          ``(B, T)``; scalar ``(B,)`` timesteps are not padded.
        - ``context`` / ``context_mask``: untouched (text-side, audio-token-
          count-independent).
        """
        if pad <= 0:
            return audio
        # Latent: zero-pad. F.pad pads last dim first; (0,0,0,pad) pads dim=-2.
        latent = F.pad(audio.latent, (0, 0, 0, pad))
        # Positions: repeat-last on dim=2.
        pos = audio.positions
        last = pos[:, :, -1:, ...]
        repeat_shape = list(pos.shape)
        repeat_shape[2] = pad
        tail = last.expand(*repeat_shape).contiguous()
        positions = torch.cat([pos, tail], dim=2)
        # Timesteps: repeat-last on dim=1 only when per-token.
        if audio.timesteps.ndim >= 2 and audio.timesteps.shape[1] == pos.shape[2]:
            ts = audio.timesteps
            last_ts = ts[:, -1:, ...].expand(ts.shape[0], pad, *ts.shape[2:]).contiguous()
            timesteps = torch.cat([ts, last_ts], dim=1)
        else:
            timesteps = audio.timesteps
        return replace(audio, latent=latent, positions=positions, timesteps=timesteps)

    def configure_audio_ulysses(self, audio_seq_len: int) -> None:
        """Configure audio sharding + padding for Ulysses.

        Call once before the denoising loop when the audio token count is
        known. The decision is cached — ``forward()`` uses it without
        re-checking.

        When sequence parallelism is active, audio is always padded to
        ``(U - T_a % U) % U`` slots so ``T_a`` becomes divisible by ``U``
        and a ``[B, T_a_padded]`` validity mask is attached so attention
        zeros out pad positions. ``forward`` strips the pad tail on exit.
        """
        if not self._sharder.is_active:
            self._audio_is_sharded = False
            self._audio_pad = 0
            return

        U = self._sharder.size
        self._audio_pad = (U - audio_seq_len % U) % U
        self._audio_is_sharded = True

        for block in self.transformer_blocks:
            target = block.inner if isinstance(block, LTX2CacheDiTPattern0BlockWrapper) else block
            target._audio_is_sharded = self._audio_is_sharded
            if hasattr(target, "audio_attn1"):
                target.audio_attn1.set_ulysses_active(self._audio_is_sharded)
            # v2a cross-attn requires sharded audio Q — gate on
            # _audio_is_sharded (video K/V is always sharded under Ulysses).
            if hasattr(target, "video_to_audio_attn"):
                target.video_to_audio_attn.set_ulysses_active(self._audio_is_sharded)

    def set_ulysses_enabled(self, enabled: bool) -> None:
        """Enable or disable Ulysses parallelism at runtime.

        Call with ``False`` before running the transformer on a single
        rank (e.g. Stage 2 of the two-stage pipeline where non-primary
        workers have already exited).  Call with ``True`` to restore
        multi-rank operation; audio sharding will be reconfigured by
        the next :meth:`configure_audio_ulysses` call.
        """
        if self._sharder.size <= 1:
            return

        if enabled:
            self._sharder.enable()
        else:
            self._sharder.disable()
            self._audio_is_sharded = False

        for block in self.transformer_blocks:
            target = block.inner if isinstance(block, LTX2CacheDiTPattern0BlockWrapper) else block
            if enabled:
                target._sharder.enable()
            else:
                target._sharder.disable()
                target._audio_is_sharded = False
            if hasattr(target, "attn1"):
                target.attn1.set_ulysses_active(enabled)
            if hasattr(target, "audio_attn1") and not enabled:
                target.audio_attn1.set_ulysses_active(False)
            if hasattr(target, "video_to_audio_attn") and not enabled:
                target.video_to_audio_attn.set_ulysses_active(False)

    # -- Output processing ---------------------------------------------------

    @staticmethod
    def _process_output(
        scale_shift_table: nn.Parameter,
        norm_out: nn.LayerNorm,
        proj_out: nn.Module,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
        fuse_adaln: bool,
    ) -> torch.Tensor:
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        x = norm_out(x)
        x = apply_shift_scale(x, scale, shift, fuse_adaln)
        return proj_out(x)

    # -- Forward -------------------------------------------------------------

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
    ) -> TextCache:
        """Compute step-invariant preprocessor outputs and text KV projections.

        Called once before the denoise loop.  The returned ``TextCache``
        is passed to ``forward()`` on every step.  Does not require latent
        data — only text context, positions, and dtype are needed.
        """
        v_ctx = v_mask = v_pe = v_cross_pe = v_kv = None
        a_ctx = a_mask = a_pe = a_cross_pe = a_kv = None

        if video_context is not None:
            v_ctx, v_mask, v_pe, v_cross_pe = self.video_args_preprocessor.prepare_text_cache(
                video_context, video_context_mask, video_positions, dtype
            )
            v_kv = [block.attn2.project_kv(v_ctx) for block in self.transformer_blocks]

        if audio_context is not None:
            a_ctx, a_mask, a_pe, a_cross_pe = self.audio_args_preprocessor.prepare_text_cache(
                audio_context, audio_context_mask, audio_positions, dtype
            )
            a_kv = [block.audio_attn2.project_kv(a_ctx) for block in self.transformer_blocks]
            # cos/sin are token-major: token axis is dim 1 for both SPLIT and
            # INTERLEAVED rope, matching `_make_pe_local`'s `cos[:, s:e]` shard.
            if self._audio_pad > 0:
                a_pe = self._pad_pe(a_pe, self._audio_pad, seq_dim=1)
                a_cross_pe = self._pad_pe(a_cross_pe, self._audio_pad, seq_dim=1)

        # Build sharded-local PE in the form the attention consumer expects.
        # fuse_qk_norm_rope=True (LTX-2 default) -> 2D [T_local, H*D] contiguous,
        # ready for the fused kernel; False -> 4D [B, T_local, H, D] for the
        # naive apply_rotary_emb path. Done one-time here, so the inner loop
        # has no reshape/contiguous/shard work on PE.
        # Inspect any LTX2Attention to learn whether fusion is on (per-modality
        # attentions are constructed with the same flag in this codepath).
        fuse_video = self.transformer_blocks[0].attn1.fuse_qk_norm_rope
        fuse_audio = (
            self.transformer_blocks[0].audio_attn1.fuse_qk_norm_rope
            if hasattr(self.transformer_blocks[0], "audio_attn1")
            else True
        )
        v_pe = self._make_pe_local(v_pe, is_audio=False, fuse=fuse_video)
        v_cross_pe = self._make_pe_local(v_cross_pe, is_audio=False, fuse=fuse_video)
        a_pe = self._make_pe_local(a_pe, is_audio=True, fuse=fuse_audio)
        a_cross_pe = self._make_pe_local(a_cross_pe, is_audio=True, fuse=fuse_audio)

        return TextCache(
            video_context=v_ctx,
            video_mask=v_mask,
            video_pe=v_pe,
            video_cross_pe=v_cross_pe,
            video_kv=v_kv,
            audio_context=a_ctx,
            audio_mask=a_mask,
            audio_pe=a_pe,
            audio_cross_pe=a_cross_pe,
            audio_kv=a_kv,
        )

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations=None,
        *,
        text_cache: TextCache,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass through the LTX-2 transformer.

        Args:
            video: Video modality input (or None).
            audio: Audio modality input (or None).
            perturbations: Optional ``BatchedPerturbationConfig`` for STG.
            text_cache: Pre-computed step-invariant outputs from ``prepare_text_cache()``.
                Always required — callers must invoke ``prepare_text_cache()`` first.

        Returns:
            Tuple of (video_output, audio_output) velocity predictions.
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        # Audio padding for Ulysses: when self._audio_pad > 0 (set once by
        # configure_audio_ulysses to make T_a divisible by ulysses_size), pad
        # audio on entry to make it shardable. Build a [B, T_a_padded] bool mask
        # (True=valid, False=pad) that travels through TransformerArgs to
        # audio_attn1 + a2v. Strip the padded tail on output below.
        # text_cache.audio_{pe,cross_pe} are already padded in prepare_text_cache.
        audio_padding_mask = None
        s_real_audio = None
        if audio is not None and self._audio_pad > 0:
            s_real_audio = audio.latent.shape[1]
            audio = self._pad_modality_audio(audio, self._audio_pad)
            s_full_audio = audio.latent.shape[1]
            audio_padding_mask = torch.ones(
                (audio.latent.shape[0], s_full_audio),
                dtype=torch.bool,
                device=audio.latent.device,
            )
            audio_padding_mask[:, s_real_audio:] = False

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

        # Attach the full-seq audio padding mask (identical across Ulysses
        # ranks; _shard_transformer_args passes it through unchanged).
        if audio_args is not None and audio_padding_mask is not None:
            audio_args = replace(audio_args, audio_padding_mask=audio_padding_mask)

        # Shard sequences for parallelism (Ulysses head-sharding, ring CP, or Attention2D).
        # Video is always sharded.  Audio sharding is decided once by
        # configure_audio_ulysses() and cached in self._audio_is_sharded.
        if self._sharder.is_active:
            if video_args is not None:
                video_args = self._shard_transformer_args(video_args)
            if self._audio_is_sharded and audio_args is not None:
                audio_args = self._shard_transformer_args(audio_args)

        v_kv = text_cache.video_kv
        a_kv = text_cache.audio_kv

        if self._uses_cache_dit():
            # Cache-DiT path: wrapper consumes (vx, ax) positional args and
            # reads full TransformerArgs from self._cache_dit_*_args. text_kv
            # must be looked up by inner.idx inside the wrapper — Cache-DiT's
            # CachedBlocks iterates inner blocks with a single shared kwargs
            # dict, so passing text_kv[i] through the outer loop would feed
            # every inner block block-0's KV.
            self._cache_dit_video_args = video_args
            self._cache_dit_audio_args = audio_args
            self._cache_dit_v_kv = v_kv
            self._cache_dit_a_kv = a_kv
            vx = video_args.x if video_args is not None else None
            ax = audio_args.x if audio_args is not None else None
            for block in self.transformer_blocks:
                vx, ax = block(vx, ax, perturbations=perturbations)
                if video_args is not None and vx is not None:
                    video_args = replace(video_args, x=vx)
                if audio_args is not None and ax is not None:
                    audio_args = replace(audio_args, x=ax)
        else:
            for i, block in enumerate(self.transformer_blocks):
                video_args, audio_args = block(
                    video=video_args,
                    audio=audio_args,
                    perturbations=perturbations,
                    text_kv_video=v_kv[i] if v_kv else None,
                    text_kv_audio=a_kv[i] if a_kv else None,
                )

        # Gather sequences back to full length for output processing.
        # Only gather embedded_timestep if it was actually sharded (dim-1
        # matches x); scalar timestep embeddings [B, 1, D] are
        # broadcast-compatible and must not be gathered.
        if self._sharder.is_active:
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
                self._fuse_adaln,
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
                self._fuse_adaln,
            )
            if audio_args is not None
            else None
        )
        # Strip the padded tail when audio was padded on entry.
        if ax is not None and s_real_audio is not None:
            ax = ax[:, :s_real_audio]
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
