"""Combined* diffusion AdaLN residual attention (full-volume NA + nested RoPE).
Owns nested full-volume abs-RoPE for the Combined / ``w_chunks==1`` path.
Does not share a residual body with ``chunked`` — only the NA module weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.ops import apply_rms_norm_uint32_safe
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.rope_math import (
    h_positions,
    rot_abs_axis_impl,
    t_positions,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.keyframes import KeyframePlanes

# Transparent path: innermost per-axis rotation is a nested compile region.
_rot_abs_axis = torch.compiler.nested_compile_region(rot_abs_axis_impl)


def _apply_nested_abs_rope_slab(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    w_pos: torch.Tensor,
    compute_dtype: torch.dtype,
    t_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate one W-extent with nested per-axis abs-RoPE.
    ``t_pos`` overrides the integer ``arange`` on the first axis; the keyframe stream passes
    its fractional plane times there so both streams' RoPE shares one origin.
    """
    d_t, d_h, _ = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    t = x.shape[1]
    h = x.shape[2]
    positions_t = t_positions(t, x.device) if t_pos is None else t_pos
    xt = _rot_abs_axis(x[..., :d_t], positions_t, inv_t, axis=1, compute_dtype=compute_dtype)
    xh = _rot_abs_axis(
        x[..., d_t : d_t + d_h],
        h_positions(h, x.device),
        inv_h,
        axis=2,
        compute_dtype=compute_dtype,
    )
    xw = _rot_abs_axis(x[..., d_t + d_h :], w_pos, inv_w, axis=3, compute_dtype=compute_dtype)
    return torch.cat([xt, xh, xw], dim=-1)


def _apply_nested_full_volume_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_tiles: int,
    compute_dtype: torch.dtype,
    t_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fixed-``num_tiles`` W split + nested per-slab rotation (Dynamo-safe).
    Both streams share the same W extent at stage 5, so the same ``num_tiles`` gives
    identical slab boundaries and ``w_pos`` -- required for their W phases to be comparable
    inside the joint softmax.
    """
    slabs = torch.chunk(x, num_tiles, dim=3)
    w_off = 0
    parts: list[torch.Tensor] = []
    for slab in slabs:
        w_slab = slab.shape[3]
        w_pos = torch.arange(w_slab, dtype=torch.float32, device=x.device) + w_off
        parts.append(
            _apply_nested_abs_rope_slab(
                slab,
                rope_split,
                inv_freqs,
                w_pos=w_pos,
                compute_dtype=compute_dtype,
                t_pos=t_pos,
            )
        )
        w_off = w_off + w_slab
    return torch.cat(parts, dim=3)


def _qkv_nested_rope(
    attn: NeighborhoodAttention3D,
    x: torch.Tensor,
    t_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K/V proj + norm/scale + nested abs-RoPE.
    ``t_pos`` overrides the integer first-axis positions; the keyframe stream passes
    its (possibly fractional) plane times there.
    """
    q, k, v = attn.project_qkv(x)
    q = apply_rms_norm_uint32_safe(attn.q_norm, q) * attn.scale
    k = apply_rms_norm_uint32_safe(attn.k_norm, k)
    inv_freqs = (
        attn.rope_inv_t.to(device=x.device),
        attn.rope_inv_h.to(device=x.device),
        attn.rope_inv_w.to(device=x.device),
    )
    positions = None if t_pos is None else t_pos.to(device=x.device, dtype=torch.float32)
    q = _apply_nested_full_volume_rope(
        q,
        attn.rope_dim_split,
        inv_freqs,
        num_tiles=attn.rope_num_tiles,
        compute_dtype=attn.rope_compute_dtype,
        t_pos=positions,
    )
    k = _apply_nested_full_volume_rope(
        k,
        attn.rope_dim_split,
        inv_freqs,
        num_tiles=attn.rope_num_tiles,
        compute_dtype=attn.rope_compute_dtype,
        t_pos=positions,
    )
    return q, k, v


def full_with_keyframes(
    x: torch.Tensor,
    keyframe_x: torch.Tensor,
    attn: NeighborhoodAttention3D,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
    keyframe_planes: KeyframePlanes,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual-stream AdaLN residual attention: one joint softmax, shared weights.
    The planes' metadata rather than a ``KeyframeStream`` here: this is the stage-5 hot path,
    and rebuilding a dataclass per block inside a compiled region buys nothing.
    Both streams take the *same* ``scale``/``shift``. Upstream computes a separate keyframe
    modulation, but the two are identical unless per-frame timestep conditioning supplies a
    conditioning mask, which we deliberately do not implement yet.
    No ``dims >= kernel_size`` floor, unlike :func:`full`: the joint window is
    clamp-and-mask, so undersized axes simply mask their out-of-range offsets.
    """
    if attn.joint_attention_function is None:
        raise RuntimeError(
            "keyframe decode needs joint_attention_function installed; build the decoder "
            "through apply_diffvae_config / apply_diffvae_mode"
        )
    batch, t, h, w, _ = x.shape
    planes = keyframe_x.shape[1]

    y = apply_rms_norm_uint32_safe(norm, x) * (1.0 + scale) + shift
    keyframe_y = apply_rms_norm_uint32_safe(norm, keyframe_x) * (1.0 + scale) + shift

    q, k, v = _qkv_nested_rope(attn, y)
    keyframe_q, keyframe_k, keyframe_v = _qkv_nested_rope(attn, keyframe_y, keyframe_planes.times.device_tensor)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    keyframe_q = keyframe_q.contiguous()
    keyframe_k = keyframe_k.contiguous()
    keyframe_v = keyframe_v.contiguous()

    out, keyframe_out = attn.joint_attention_function(
        attn,
        q,
        k,
        v,
        keyframe_q,
        keyframe_k,
        keyframe_v,
        keyframe_planes,
    )
    x = x + attn.proj(out.reshape(batch, t, h, w, attn.dim))
    keyframe_x = keyframe_x + attn.proj(keyframe_out.reshape(batch, planes, h, w, attn.dim))
    return x, keyframe_x


def full(
    x: torch.Tensor,
    attn: NeighborhoodAttention3D,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """``x + NA(modulate(norm(x)))`` with nested full-volume abs-RoPE."""
    y = apply_rms_norm_uint32_safe(norm, x) * (1.0 + scale) + shift
    batch, t, h, w, _ = y.shape
    kt, kh, kw = attn.kernel_size
    if t < kt or h < kh or w < kw:
        raise ValueError(
            f"3D neighborhood attention requires spatial dims >= kernel_size; "
            f"got (T,H,W)=({t},{h},{w}) vs kernel={attn.kernel_size}"
        )

    q, k, v = _qkv_nested_rope(attn, y)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = attn.attention_function(attn, q, k, v)
    out = out.reshape(batch, t, h, w, attn.dim)
    return x + attn.proj(out)
