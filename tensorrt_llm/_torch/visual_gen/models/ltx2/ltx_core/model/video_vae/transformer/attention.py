"""3D Neighborhood Attention via NATTEN + absolute RoPE prelude.
Parameter shell shared by det ``NABlock`` and both diff-attn roles.
Diffusion AdaLN residuals live in pathway packages (each owns its RoPE);
det stages use ``det_attn_rope`` from :meth:`NeighborhoodAttention3D.forward`.
``attention_function`` selects the NA backend (NATTEN, Triton/eager fallback, or CuTe DSL).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.det_attn_rope import (
    det_project_qkv_eager,
    det_project_qkv_lean,
    det_qkv_rope,
    det_qkv_rope_at_times,
    det_rope_qk_,
    linear_into_phase_buffer,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.det_rms_norm import DetRMSNorm
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.qkv import QKVProjections
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.rope_math import (
    DEFAULT_ABS_ROPE_NUM_TILES,
    AbsRopeTableCache,
    default_rope_dim_split,
    rope_inv_freqs,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.keyframes import KeyframePlanes, KeyframeStream

try:
    import natten

    _NATTEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    natten = None  # type: ignore[assignment]
    _NATTEN_AVAILABLE = False


def natten_available() -> bool:
    return _NATTEN_AVAILABLE


class NAAttentionCallable(Protocol):
    """A windowed 3D neighborhood-attention backend.
    Q/K/V arrive as ``(B, T, H, W, NH, HD)``, already normed, scaled and RoPE'd;
    the return is ``(B, T, H, W, NH*HD)`` or anything reshapeable to it. The owning
    module is passed so a backend can read configuration (``kernel_size``, softmax
    bound, …). Backend-specific settings (NATTEN's kernel pin) live on the callable.
    """

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor: ...


class JointNAAttentionCallable(Protocol):
    """A windowed 3D NA backend that also carries a keyframe plane stack.
    Same conventions as :class:`NAAttentionCallable` -- Q/K/V already normed, scaled and
    absolutely RoPE'd -- with a second stream shaped ``(B, P, H, W, NH, HD)`` whose plane
    axis sits in video's temporal slot. Both streams' RoPE must share one origin, so
    ``keyframe_planes.times`` are tile-local. Backends that build plane tables on the host read the
    planes' host tensors; the others read the device ones. Returns one output per stream.
    NATTEN and the CuTe DSL kernel cannot express a joint window, so this is a separate
    slot from ``attention_function`` rather than a widening of it: it keeps the shipping
    keyframe-less hot path untouched, and it is immune to the install-order hazard that
    ``configure_natten_backend`` creates by overwriting ``attention_function`` wholesale.
    """

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keyframe_q: torch.Tensor,
        keyframe_k: torch.Tensor,
        keyframe_v: torch.Tensor,
        keyframe_planes: KeyframePlanes,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class NattenAttention(NAAttentionCallable):
    """``natten.na3d``, the default backend.
    ``backend`` pins ``na3d``'s own kernel choice (e.g. ``"cutlass-fna"``); ``None``
    leaves NATTEN's auto-pick (hopper-fna on H100, etc.).
    """

    def __init__(self, backend: str | None = None) -> None:
        self._backend = backend

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if not _NATTEN_AVAILABLE:
            raise ImportError(
                "natten is required for NeighborhoodAttention3D. "
                "Install with: uv sync --package ltx-core --extra natten "
                '(or: uv pip install "natten==0.21.7+torch2130cu132" -f https://whl.natten.org; '
                "requires torch==2.13.0+cu132)"
            )
        # scale=1.0: callers already applied ``attn.scale`` to Q.
        # RMSNorm under bf16 autocast can leave Q/K in float32 while V stays bf16;
        # natten requires a uniform dtype (same cast pattern as flash-attn paths).
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        return natten.na3d(q, k, v, kernel_size=attn.kernel_size, scale=1.0, backend=self._backend)


class NeighborhoodAttention3D(nn.Module):
    """3D Neighborhood Attention with absolute RoPE + pluggable NA backend.
    Q/K receive absolute RoPE; attention is ``attention_function`` (NATTEN by
    default; Triton or eager SDPA when natten is missing; CuTe DSL via
    DiffVAE BLACKWELL_DSL install). Relative gather-based NA is not used as a
    production gather path on this branch.
    NATTEN shifts its window inward at grid boundaries instead of
    clamp-and-mask; interior positions match the gather reference closely,
    boundary positions may differ slightly.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        rope_dim_split: tuple[int, int, int] | None = None,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, f"dim={dim} not divisible by head_dim={head_dim}"
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5

        if rope_dim_split is None:
            rope_dim_split = default_rope_dim_split(head_dim)
        assert sum(rope_dim_split) == head_dim, f"rope_dim_split={rope_dim_split} must sum to head_dim={head_dim}"
        self.rope_dim_split = rope_dim_split
        self.rope_base = rope_base
        self.rope_num_tiles = DEFAULT_ABS_ROPE_NUM_TILES
        self.rope_compute_dtype = torch.float32
        # The decode scope's abs-RoPE table cache while one is open (see ``abs_rope_table_cache``).
        self.rope_table_cache: AbsRopeTableCache | None = None
        # Kept for the chunked opaque residual (string arg); callable is the swap surface.
        self.natten_backend: str | None = None
        self.attention_function: NAAttentionCallable = NattenAttention()
        # Separate slot, installed for every mode; never NATTEN/DSL. Only the keyframe
        # decode path reads it, so keyframe-less decode keeps NATTEN when it is installed.
        self.joint_attention_function: JointNAAttentionCallable | None = None

        self.register_buffer("rope_inv_t", rope_inv_freqs(rope_dim_split[0], rope_base), persistent=False)
        self.register_buffer("rope_inv_h", rope_inv_freqs(rope_dim_split[1], rope_base), persistent=False)
        self.register_buffer("rope_inv_w", rope_inv_freqs(rope_dim_split[2], rope_base), persistent=False)

        self.qkv = QKVProjections(dim)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = DetRMSNorm(head_dim, eps=1e-6)
        self.k_norm = DetRMSNorm(head_dim, eps=1e-6)

        # W-chunking configuration (consumed by ``chunked.attn``).
        self.w_chunks = 1  # 1 = no chunking

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K/V as owned contiguous ``(B,T,H,W,NH,HD)`` tensors."""
        batch, t, h, w, _ = x.shape
        q, k, v = self.qkv(x)
        shape = (batch, t, h, w, self.num_heads, self.head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Det-stage NA: opaque abs-RoPE via ``det_attn_rope`` + ``attention_function``.
        ``x``/output: (B, T, H, W, C) — channels-last. RoPE positions are local
        0-based (see ``det_attn_rope`` module docstring for why that is
        equivalent under tiled decode).
        """
        batch, t, h, w, _ = x.shape
        kt, kh, kw = self.kernel_size
        if t < kt or h < kh or w < kw:
            raise ValueError(
                f"3D neighborhood attention requires spatial dims >= kernel_size; "
                f"got (T,H,W)=({t},{h},{w}) vs kernel={self.kernel_size}"
            )

        if torch.compiler.is_compiling():
            q, k, v = det_qkv_rope(self, x)
        else:
            # Eager: norms (and Q's scale) and RoPE in place on the fresh projections; same values.
            q, k, v = det_project_qkv_eager(self, x)
            q, k = det_rope_qk_(self, q.contiguous(), k.contiguous())
        # natten's CUTLASS kernel silently produces wrong output if inputs are non-contiguous.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = self.attention_function(self, q, k, v)
        out = out.reshape(batch, t, h, w, self.dim)
        return self.proj(out)

    def forward_with_keyframes(
        self,
        x: torch.Tensor,
        keyframes: KeyframeStream,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        """Dual-stream det NA: one joint softmax over video and keyframe planes.
        Unlike :meth:`forward` there is no ``dims >= kernel_size`` floor: the joint window
        is clamp-and-mask, so an undersized axis simply masks its out-of-range offsets.
        """
        if self.joint_attention_function is None:
            raise RuntimeError(
                "keyframe decode needs joint_attention_function installed; build the decoder "
                "through apply_diffvae_config / apply_diffvae_mode"
            )
        batch, t, h, w, _ = x.shape
        planes = keyframes.x.shape[1]

        eager = not torch.compiler.is_compiling()
        if eager:
            # Projections in workspace phase buffers, norms, scale and RoPE in place; same values.
            q, k, v = det_project_qkv_lean(self, x)
            del x
            q, k = det_rope_qk_(self, q, k)
        else:
            q, k, v = det_qkv_rope(self, x)
        keyframe_q, keyframe_k, keyframe_v = det_qkv_rope_at_times(
            self, keyframes.x, keyframes.planes.times.device_tensor
        )
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        keyframe_q = keyframe_q.contiguous()
        keyframe_k = keyframe_k.contiguous()
        keyframe_v = keyframe_v.contiguous()

        out, keyframe_out = self.joint_attention_function(
            self,
            q,
            k,
            v,
            keyframe_q,
            keyframe_k,
            keyframe_v,
            keyframes.planes,
        )
        # Q/K/V are dead once the attention has run; three volumes need not stay live across the projection.
        del q, k, v, keyframe_q, keyframe_k, keyframe_v
        out = out.reshape(batch, t, h, w, self.dim)
        out = linear_into_phase_buffer(self.proj, out) if eager else self.proj(out)
        keyframe_out = self.proj(keyframe_out.reshape(batch, planes, h, w, self.dim))
        return out, dataclasses.replace(keyframes, x=keyframe_out)


def configure_w_chunks(module_root: nn.Module, w_chunks: int = 1) -> None:
    """Set W-chunking on ``NeighborhoodAttention3D`` under ``module_root``.
    When ``w_chunks > 1``, also sets ``rope_num_tiles=1`` so ``chunked.attn``
    owns the W axis (RoPE W-tiling would double-split). Pass only the diffusion
    residual subtree — det-stage attention must keep its default RoPE tiling.
    """
    for module in module_root.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.w_chunks = w_chunks
            if w_chunks > 1:
                module.rope_num_tiles = 1
