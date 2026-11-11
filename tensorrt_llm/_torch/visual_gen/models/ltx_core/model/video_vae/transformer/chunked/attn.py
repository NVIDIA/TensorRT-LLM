"""Chunked* diffusion AdaLN residual attention (W-loop + halo + in-slab RoPE).
Owns the fixed-extent W-chunk residual and its in-slab abs-RoPE (``w_pos``).
Does **not** call ``combined.attn``; shares only NA module weights (+ ``rope_math``).
NA goes through ``attn.attention_function`` (same pluggable backend as combined / det).
The W-chunk loop stays behind opaque ``ltx_core::na_residual_w_chunked`` so Dynamo
does not unroll it; the callable is bound via a contextvar around the op call
(custom_op schemas cannot take Python callables as args).
The keyframe (dual-stream) variant lives here too, behind its *own* op
``ltx_core::na_residual_w_chunked_joint`` rather than a widened
``mutates_args`` on the video-only op -- same reasoning as the separate
``joint_attention_function`` slot: the shipping hot path's schema stays byte-identical
and the keyframe arm remains trivially separable. Both streams are cut into the *same*
W slabs with the same ``halo`` and the same ``w_pos``, which is what keeps their W RoPE
phases comparable inside the joint softmax.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.keyframes import KeyframePlanes
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.ops import rms_norm_uint32_safe
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.attention import (
    JointNAAttentionCallable,
    NAAttentionCallable,
    NattenAttention,
    NeighborhoodAttention3D,
    natten_available,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.rope_math import (
    h_positions,
    rot_abs_axis_impl,
    t_positions,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.types import HostDeviceTensorPair

# Bound by ``chunked()`` for the duration of the opaque custom_op. Dynamo only sees
# the op call site; the Python body reads these at runtime.
_CHUNKED_ATTENTION_FUNCTION: ContextVar[NAAttentionCallable | None] = ContextVar(
    "ltx_chunked_attention_function",
    default=None,
)
_CHUNKED_ATTN_MODULE: ContextVar[NeighborhoodAttention3D | None] = ContextVar(
    "ltx_chunked_attn_module",
    default=None,
)


@contextmanager
def _bound_chunked_attention(
    attention_function: NAAttentionCallable,
    attn: NeighborhoodAttention3D,
) -> Iterator[None]:
    t_fn = _CHUNKED_ATTENTION_FUNCTION.set(attention_function)
    t_mod = _CHUNKED_ATTN_MODULE.set(attn)
    try:
        yield
    finally:
        _CHUNKED_ATTENTION_FUNCTION.reset(t_fn)
        _CHUNKED_ATTN_MODULE.reset(t_mod)


def _resolve_chunked_attention() -> tuple[NAAttentionCallable, NeighborhoodAttention3D]:
    fn = _CHUNKED_ATTENTION_FUNCTION.get()
    mod = _CHUNKED_ATTN_MODULE.get()
    if fn is None or mod is None:
        raise RuntimeError(
            "ltx_core::na_residual_w_chunked requires attention_function binding; "
            "call via chunked(), not the custom_op directly"
        )
    return fn, mod


def _resolve_chunked_joint_attention() -> tuple[JointNAAttentionCallable, NeighborhoodAttention3D]:
    """The joint backend for the keyframe arm, read off the bound NA module.
    Deliberately not a third contextvar: the joint callable lives in its own slot on the
    module, so binding the module is enough and there is one place it can come from.
    """
    mod = _CHUNKED_ATTN_MODULE.get()
    if mod is None:
        raise RuntimeError(
            "ltx_core::na_residual_w_chunked_joint requires NA module binding; "
            "call via chunked_with_keyframes(), not the custom_op directly"
        )
    if mod.joint_attention_function is None:
        raise RuntimeError(
            "keyframe decode needs joint_attention_function installed; build the decoder "
            "through apply_diffvae_config / apply_diffvae_mode"
        )
    return mod.joint_attention_function, mod


def _apply_in_slab_abs_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    w_pos: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
    t_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-extent abs-RoPE via raw rot (no RoPE tile loop, no nested region).
    Used inside the W-chunk residual body: the chunk is already one W slab.
    Nested-in-nested breaks FakeTensor mode matching under Inductor.
    ``w_pos`` must be length ``x.shape[3]`` (global W coords for this slab).
    ``t_pos`` overrides the integer ``arange`` on the first axis; the keyframe stream
    passes its fractional plane times there so both streams' RoPE shares one origin.
    """
    d_t, d_h, _ = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    t = x.shape[1]
    h = x.shape[2]
    positions_t = t_positions(t, x.device) if t_pos is None else t_pos
    xt = rot_abs_axis_impl(x[..., :d_t], positions_t, inv_t, axis=1, compute_dtype=compute_dtype)
    xh = rot_abs_axis_impl(
        x[..., d_t : d_t + d_h],
        h_positions(h, x.device),
        inv_h,
        axis=2,
        compute_dtype=compute_dtype,
    )
    xw = rot_abs_axis_impl(x[..., d_t + d_h :], w_pos, inv_w, axis=3, compute_dtype=compute_dtype)
    return torch.cat([xt, xh, xw], dim=-1)


def _slab_qkv(  # noqa: PLR0913
    y: torch.Tensor,
    w_pos: torch.Tensor,
    q_w: torch.Tensor,
    q_b: torch.Tensor,
    k_w: torch.Tensor,
    k_b: torch.Tensor,
    v_w: torch.Tensor,
    v_b: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_heads: int,
    head_dim: int,
    attn_scale: float,
    rope_dtype: torch.dtype,
    *,
    t_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K/V proj + norm/scale + in-slab abs-RoPE for one already-modulated W slab.
    Shared by the video-only and joint bodies so the two cannot drift apart in the
    numerics that feed the softmax.
    """
    batch, t, h, ext_w, _ = y.shape
    eps = 1e-6
    head_shape = (batch, t, h, ext_w, num_heads, head_dim)
    q = F.linear(y, q_w, q_b).view(head_shape)
    q = rms_norm_uint32_safe(q, (head_dim,), q_norm_w, eps) * attn_scale
    q = _apply_in_slab_abs_rope(q, rope_split, inv_freqs, w_pos, compute_dtype=rope_dtype, t_pos=t_pos)
    k = F.linear(y, k_w, k_b).view(head_shape)
    k = rms_norm_uint32_safe(k, (head_dim,), k_norm_w, eps)
    k = _apply_in_slab_abs_rope(k, rope_split, inv_freqs, w_pos, compute_dtype=rope_dtype, t_pos=t_pos)
    v = F.linear(y, v_w, v_b).view(head_shape)
    return q, k, v


def _attn_on_chunk_impl(  # noqa: PLR0913
    attention_function: NAAttentionCallable,
    attn: NeighborhoodAttention3D | SimpleNamespace,
    x_chunk: torch.Tensor,
    w_pos: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_weight: torch.Tensor,
    q_w: torch.Tensor,
    q_b: torch.Tensor,
    k_w: torch.Tensor,
    k_b: torch.Tensor,
    v_w: torch.Tensor,
    v_b: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    proj_w: torch.Tensor,
    proj_b: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    kt: int,
    kh: int,
    kw: int,
    num_heads: int,
    head_dim: int,
    dim: int,
    attn_scale: float,
    rope_num_tiles: int,
    rope_compute_bf16: bool,
) -> torch.Tensor:
    """One W-slab: rms_norm + modulate + QKV / in-slab RoPE / ``attention_function`` / proj."""
    del rope_num_tiles, kt, kh, kw  # chunk is already one W slab; kernel_size lives on ``attn``
    if not natten_available() and type(attention_function) is NattenAttention:
        raise ImportError("natten is required for W-chunked residual attention")

    batch, t, h, ext_w, _ = x_chunk.shape
    rope_dtype = torch.bfloat16 if rope_compute_bf16 else torch.float32
    inv_freqs = (inv_t, inv_h, inv_w)
    rope_split = (d_t, d_h, d_w)

    y = rms_norm_uint32_safe(x_chunk, (dim,), norm_weight, 1e-6)
    y = y * (1.0 + scale) + shift

    q, k, v = _slab_qkv(
        y,
        w_pos,
        q_w,
        q_b,
        k_w,
        k_b,
        v_w,
        v_b,
        q_norm_w,
        k_norm_w,
        rope_split,
        inv_freqs,
        num_heads,
        head_dim,
        attn_scale,
        rope_dtype,
    )
    del y

    out = attention_function(attn, q.contiguous(), k.contiguous(), v.contiguous())  # type: ignore[arg-type]
    del q, k, v
    return F.linear(out.reshape(batch, t, h, ext_w, dim), proj_w, proj_b)


def _joint_attn_on_chunk_impl(  # noqa: PLR0913
    joint_attention_function: JointNAAttentionCallable,
    attn: NeighborhoodAttention3D | SimpleNamespace,
    x_chunk: torch.Tensor,
    keyframe_chunk: torch.Tensor,
    w_pos: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    keyframe_times_host: torch.Tensor,
    keyframe_valid_host: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_weight: torch.Tensor,
    q_w: torch.Tensor,
    q_b: torch.Tensor,
    k_w: torch.Tensor,
    k_b: torch.Tensor,
    v_w: torch.Tensor,
    v_b: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    proj_w: torch.Tensor,
    proj_b: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    kt: int,
    kh: int,
    kw: int,
    num_heads: int,
    head_dim: int,
    dim: int,
    attn_scale: float,
    rope_num_tiles: int,
    rope_compute_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One W-slab of both streams through a single joint softmax; shared weights.
    Both slabs get the same ``scale``/``shift`` and the same ``w_pos``; only the first RoPE
    axis differs, where the keyframe stream substitutes its fractional plane times.
    """
    del rope_num_tiles, kt, kh, kw  # chunk is already one W slab; kernel_size lives on ``attn``
    batch, t, h, ext_w, _ = x_chunk.shape
    planes = keyframe_chunk.shape[1]
    rope_dtype = torch.bfloat16 if rope_compute_bf16 else torch.float32
    inv_freqs = (inv_t, inv_h, inv_w)
    rope_split = (d_t, d_h, d_w)
    weights = (q_w, q_b, k_w, k_b, v_w, v_b, q_norm_w, k_norm_w)
    meta = (rope_split, inv_freqs, num_heads, head_dim, attn_scale, rope_dtype)

    y = rms_norm_uint32_safe(x_chunk, (dim,), norm_weight, 1e-6)
    y = y * (1.0 + scale) + shift
    keyframe_y = rms_norm_uint32_safe(keyframe_chunk, (dim,), norm_weight, 1e-6)
    keyframe_y = keyframe_y * (1.0 + scale) + shift

    q, k, v = _slab_qkv(y, w_pos, *weights, *meta)
    del y
    times = keyframe_times.to(device=keyframe_chunk.device, dtype=torch.float32)
    keyframe_q, keyframe_k, keyframe_v = _slab_qkv(keyframe_y, w_pos, *weights, *meta, t_pos=times)
    del keyframe_y

    out, keyframe_out = joint_attention_function(
        attn,  # type: ignore[arg-type]
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        keyframe_q.contiguous(),
        keyframe_k.contiguous(),
        keyframe_v.contiguous(),
        KeyframePlanes(
            HostDeviceTensorPair(keyframe_times_host.to(dtype=torch.float32), times),
            HostDeviceTensorPair(keyframe_valid_host, keyframe_valid),
        ),
    )
    del q, k, v, keyframe_q, keyframe_k, keyframe_v
    return (
        F.linear(out.reshape(batch, t, h, ext_w, dim), proj_w, proj_b),
        F.linear(keyframe_out.reshape(batch, planes, h, ext_w, dim), proj_w, proj_b),
    )


@torch.compiler.disable
def _pack_residual_weights(attn: NeighborhoodAttention3D, norm: nn.RMSNorm, x: torch.Tensor) -> tuple:
    """Pack NA + norm weights / RoPE meta for the opaque chunked residual op."""
    d_t, d_h, d_w = attn.rope_dim_split
    kt, kh, kw = attn.kernel_size
    inv_t = attn.rope_inv_t.to(device=x.device)
    inv_h = attn.rope_inv_h.to(device=x.device)
    inv_w = attn.rope_inv_w.to(device=x.device)
    return (
        norm.weight,
        attn.qkv.to_q.weight,
        attn.qkv.to_q.bias,
        attn.qkv.to_k.weight,
        attn.qkv.to_k.bias,
        attn.qkv.to_v.weight,
        attn.qkv.to_v.bias,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.proj.weight,
        attn.proj.bias,
        inv_t,
        inv_h,
        inv_w,
        d_t,
        d_h,
        d_w,
        kt,
        kh,
        kw,
        attn.num_heads,
        attn.head_dim,
        attn.dim,
        float(attn.scale),
        attn.rope_num_tiles,
        attn.rope_compute_dtype == torch.bfloat16,
    )


def _cut_slab(
    x: torch.Tensor,
    core_start: int,
    core_end: int,
    halo: int,
    extent: int,
    left_halo: torch.Tensor | None,
    *,
    has_right_neighbor: bool,
) -> torch.Tensor:
    """One fixed-``extent`` W slab: core + neighbour halos + edge-replicated volume edges.
    ``left_halo`` is ``None`` exactly on the first slab (the true left W boundary); it is
    passed in rather than re-read from ``x`` because the caller mutates ``x`` in place as it
    goes, so the left halo has to be captured before the previous slab's ``add_``.
    Missing true-boundary halo slots are edge-replicated -- same spirit as trailing-frame
    replicate -- so attention never sees zero-padded W boundaries. This is a deliberate
    approximation of the untiled window at the two volume edges, and it is shared verbatim
    by the video-only and keyframe arms so both streams' slabs stay phase-comparable.
    """
    w = x.shape[3]
    c = x.shape[4]
    buf = x.new_zeros(*x.shape[:3], extent, c)
    core_len = core_end - core_start
    if left_halo is not None:
        lh = left_halo.shape[3]
        buf[:, :, :, halo - lh : halo, :] = left_halo
    buf[:, :, :, halo : halo + core_len, :] = x[:, :, :, core_start:core_end, :]
    right_filled = 0
    if has_right_neighbor:
        right_end = min(w, core_end + halo)
        right = x[:, :, :, core_end:right_end, :]
        right_filled = right.shape[3]
        buf[:, :, :, halo + core_len : halo + core_len + right_filled, :] = right

    # True W-boundary: replicate-pad missing halo slots (do not overwrite neighbors).
    if left_halo is None and halo > 0 and core_len > 0:
        edge_l = buf[:, :, :, halo : halo + 1, :]
        buf[:, :, :, :halo, :] = edge_l.expand(*x.shape[:3], halo, c)
    missing_right = extent - (halo + core_len + right_filled)
    if missing_right > 0 and core_len > 0:
        edge_r = buf[:, :, :, halo + core_len - 1 : halo + core_len, :]
        lo_r = halo + core_len + right_filled
        buf[:, :, :, lo_r:extent, :] = edge_r.expand(*x.shape[:3], missing_right, c)
    return buf


def _slab_geometry(w: int, w_chunks: int, halo: int) -> tuple[int, int]:
    """``(chunk_w, extent)`` for a W-chunked pass."""
    chunk_w = (w + w_chunks - 1) // w_chunks
    return chunk_w, chunk_w + 2 * halo


def _run_w_chunked_residual(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    packed: tuple,
    w_chunks: int,
    halo: int,
    attn_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """In-place ``x += attn(...)`` with fixed-extent W slabs.
    Neighbor-sourced halos are copied when available. Missing true-boundary
    left/right halo slots (volume edges) are edge-replicated — same spirit as
    trailing-frame replicate — so NA never sees zero-padded W boundaries.
    """
    w = x.shape[3]
    chunk_w, extent = _slab_geometry(w, w_chunks, halo)
    left_halo: torch.Tensor | None = None

    for i in range(w_chunks):
        core_start = i * chunk_w
        core_end = min(w, (i + 1) * chunk_w)
        core_len = core_end - core_start
        has_right = i + 1 < w_chunks

        buf = _cut_slab(x, core_start, core_end, halo, extent, left_halo, has_right_neighbor=has_right)
        if has_right:
            take = min(halo, core_len)
            left_halo = x[:, :, :, core_end - take : core_end, :].clone()

        w_pos = torch.arange(extent, device=x.device, dtype=torch.float32) + (core_start - halo)
        out = attn_fn(buf, w_pos, scale, shift, *packed)
        x[:, :, :, core_start:core_end, :].add_(out[:, :, :, halo : halo + core_len, :])

    return x


def _run_w_chunked_joint_residual(
    x: torch.Tensor,
    keyframe_x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    packed: tuple,
    w_chunks: int,
    halo: int,
    attn_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place dual-stream ``x += attn(...)`` / ``kf_x += attn(...)`` over shared W slabs.
    The two streams are cut with one geometry and one ``w_pos``, so a video core query and
    the keyframe keys it needs land at the same buffer offsets. That is why the keyframe
    stream must already share the video stream's ``H``/``W`` -- it does at stage 5, where
    keyframe upsampling is spatial-only, and it is checked rather than assumed.
    """
    if x.shape[2:4] != keyframe_x.shape[2:4]:
        raise ValueError(
            f"joint W-chunked residual needs matching (H, W); video {tuple(x.shape[2:4])} vs "
            f"keyframes {tuple(keyframe_x.shape[2:4])}"
        )
    w = x.shape[3]
    chunk_w, extent = _slab_geometry(w, w_chunks, halo)
    left_halo: torch.Tensor | None = None
    keyframe_left_halo: torch.Tensor | None = None

    for i in range(w_chunks):
        core_start = i * chunk_w
        core_end = min(w, (i + 1) * chunk_w)
        core_len = core_end - core_start
        has_right = i + 1 < w_chunks

        buf = _cut_slab(x, core_start, core_end, halo, extent, left_halo, has_right_neighbor=has_right)
        keyframe_buf = _cut_slab(
            keyframe_x,
            core_start,
            core_end,
            halo,
            extent,
            keyframe_left_halo,
            has_right_neighbor=has_right,
        )
        if has_right:
            take = min(halo, core_len)
            left_halo = x[:, :, :, core_end - take : core_end, :].clone()
            keyframe_left_halo = keyframe_x[:, :, :, core_end - take : core_end, :].clone()

        w_pos = torch.arange(extent, device=x.device, dtype=torch.float32) + (core_start - halo)
        out, keyframe_out = attn_fn(
            buf,
            keyframe_buf,
            w_pos,
            keyframe_times,
            keyframe_valid,
            scale,
            shift,
            *packed,
        )
        x[:, :, :, core_start:core_end, :].add_(out[:, :, :, halo : halo + core_len, :])
        keyframe_x[:, :, :, core_start:core_end, :].add_(keyframe_out[:, :, :, halo : halo + core_len, :])

    return x, keyframe_x


@torch.library.custom_op("ltx_core::na_residual_w_chunked", mutates_args=("x",))
def _na_residual_w_chunked_op(  # noqa: PLR0913
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_weight: torch.Tensor,
    q_w: torch.Tensor,
    q_b: torch.Tensor,
    k_w: torch.Tensor,
    k_b: torch.Tensor,
    v_w: torch.Tensor,
    v_b: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    proj_w: torch.Tensor,
    proj_b: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    kt: int,
    kh: int,
    kw: int,
    num_heads: int,
    head_dim: int,
    dim: int,
    attn_scale: float,
    rope_num_tiles: int,
    rope_compute_bf16: bool,
    w_chunks: int,
) -> None:
    """Opaque in-place W-chunked residual — Dynamo does not unroll the chunk loop.
    Reads ``attention_function`` / NA module from the contextvars set by ``chunked()``.
    """
    attention_function, attn = _resolve_chunked_attention()
    halo = kw // 2
    packed = (
        norm_weight,
        q_w,
        q_b,
        k_w,
        k_b,
        v_w,
        v_b,
        q_norm_w,
        k_norm_w,
        proj_w,
        proj_b,
        inv_t,
        inv_h,
        inv_w,
        d_t,
        d_h,
        d_w,
        kt,
        kh,
        kw,
        num_heads,
        head_dim,
        dim,
        attn_scale,
        rope_num_tiles,
        rope_compute_bf16,
    )

    def _slab(
        x_chunk: torch.Tensor,
        w_pos: torch.Tensor,
        scale_: torch.Tensor,
        shift_: torch.Tensor,
        *rest: object,
    ) -> torch.Tensor:
        return _attn_on_chunk_impl(attention_function, attn, x_chunk, w_pos, scale_, shift_, *rest)  # type: ignore[arg-type]

    _run_w_chunked_residual(x, scale, shift, packed, w_chunks, halo, _slab)


@_na_residual_w_chunked_op.register_fake
def _na_residual_w_chunked_fake(  # noqa: PLR0913
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_weight: torch.Tensor,
    q_w: torch.Tensor,
    q_b: torch.Tensor,
    k_w: torch.Tensor,
    k_b: torch.Tensor,
    v_w: torch.Tensor,
    v_b: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    proj_w: torch.Tensor,
    proj_b: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    kt: int,
    kh: int,
    kw: int,
    num_heads: int,
    head_dim: int,
    dim: int,
    attn_scale: float,
    rope_num_tiles: int,
    rope_compute_bf16: bool,
    w_chunks: int,
) -> None:
    del (
        x,
        scale,
        shift,
        norm_weight,
        q_w,
        q_b,
        k_w,
        k_b,
        v_w,
        v_b,
        q_norm_w,
        k_norm_w,
        proj_w,
        proj_b,
        inv_t,
        inv_h,
        inv_w,
        d_t,
        d_h,
        d_w,
        kt,
        kh,
        kw,
        num_heads,
        head_dim,
        dim,
        attn_scale,
        rope_num_tiles,
        rope_compute_bf16,
        w_chunks,
    )


@torch.library.custom_op("ltx_core::na_residual_w_chunked_joint", mutates_args=("x", "keyframe_x"))
def _na_residual_w_chunked_joint_op(  # noqa: PLR0913
    x: torch.Tensor,
    keyframe_x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    keyframe_times_host: torch.Tensor,
    keyframe_valid_host: torch.Tensor,
    norm_weight: torch.Tensor,
    q_w: torch.Tensor,
    q_b: torch.Tensor,
    k_w: torch.Tensor,
    k_b: torch.Tensor,
    v_w: torch.Tensor,
    v_b: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    proj_w: torch.Tensor,
    proj_b: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    kt: int,
    kh: int,
    kw: int,
    num_heads: int,
    head_dim: int,
    dim: int,
    attn_scale: float,
    rope_num_tiles: int,
    rope_compute_bf16: bool,
    w_chunks: int,
) -> None:
    """Opaque in-place dual-stream W-chunked residual (own op, video-only schema untouched).
    Reads the joint backend / NA module from the contextvars set by
    ``chunked_with_keyframes``. Slot tables are built inside the joint backend, so their
    T-dependent extents never reach Dynamo.
    """
    joint_attention_function, attn = _resolve_chunked_joint_attention()
    halo = kw // 2
    packed = (
        norm_weight,
        q_w,
        q_b,
        k_w,
        k_b,
        v_w,
        v_b,
        q_norm_w,
        k_norm_w,
        proj_w,
        proj_b,
        inv_t,
        inv_h,
        inv_w,
        d_t,
        d_h,
        d_w,
        kt,
        kh,
        kw,
        num_heads,
        head_dim,
        dim,
        attn_scale,
        rope_num_tiles,
        rope_compute_bf16,
    )

    def _slab(
        x_chunk: torch.Tensor,
        keyframe_chunk: torch.Tensor,
        w_pos: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        scale_: torch.Tensor,
        shift_: torch.Tensor,
        *rest: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _joint_attn_on_chunk_impl(
            joint_attention_function,
            attn,
            x_chunk,
            keyframe_chunk,
            w_pos,
            times,
            valid,
            keyframe_times_host,
            keyframe_valid_host,
            scale_,
            shift_,
            *rest,  # type: ignore[arg-type]
        )

    _run_w_chunked_joint_residual(
        x,
        keyframe_x,
        scale,
        shift,
        keyframe_times,
        keyframe_valid,
        packed,
        w_chunks,
        halo,
        _slab,
    )


@_na_residual_w_chunked_joint_op.register_fake
def _na_residual_w_chunked_joint_fake(  # noqa: PLR0913
    x: torch.Tensor,
    keyframe_x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    keyframe_times_host: torch.Tensor,
    keyframe_valid_host: torch.Tensor,
    norm_weight: torch.Tensor,
    q_w: torch.Tensor,
    q_b: torch.Tensor,
    k_w: torch.Tensor,
    k_b: torch.Tensor,
    v_w: torch.Tensor,
    v_b: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    proj_w: torch.Tensor,
    proj_b: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    kt: int,
    kh: int,
    kw: int,
    num_heads: int,
    head_dim: int,
    dim: int,
    attn_scale: float,
    rope_num_tiles: int,
    rope_compute_bf16: bool,
    w_chunks: int,
) -> None:
    del (
        x,
        keyframe_x,
        scale,
        shift,
        keyframe_times,
        keyframe_valid,
        keyframe_times_host,
        keyframe_valid_host,
        norm_weight,
        q_w,
        q_b,
        k_w,
        k_b,
        v_w,
        v_b,
        q_norm_w,
        k_norm_w,
        proj_w,
        proj_b,
        inv_t,
        inv_h,
        inv_w,
        d_t,
        d_h,
        d_w,
        kt,
        kh,
        kw,
        num_heads,
        head_dim,
        dim,
        attn_scale,
        rope_num_tiles,
        rope_compute_bf16,
        w_chunks,
    )


def chunked_with_keyframes(
    x: torch.Tensor,
    keyframe_x: torch.Tensor,
    attn: NeighborhoodAttention3D,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
    keyframe_planes: KeyframePlanes,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place dual-stream W-chunked residual through one joint softmax per slab.
    Keyframe counterpart of :func:`chunked`, using ``attn.joint_attention_function``
    (never natten) instead of ``attn.attention_function``.
    """
    if attn.w_chunks < 1:
        raise ValueError(f"w_chunks must be >= 1, got {attn.w_chunks}")
    packed = _pack_residual_weights(attn, norm, x)
    with _bound_chunked_attention(attn.attention_function, attn):
        _na_residual_w_chunked_joint_op(
            x,
            keyframe_x,
            scale,
            shift,
            keyframe_planes.times.device_tensor,
            keyframe_planes.valid.device_tensor,
            keyframe_planes.times.host_tensor,
            keyframe_planes.valid.host_tensor,
            *packed,
            attn.w_chunks,
        )
    return x, keyframe_x


def chunked(
    x: torch.Tensor,
    attn: NeighborhoodAttention3D,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """In-place W-chunked ``x += NA(modulate(norm(x)))`` with in-slab RoPE.
    Opaque to Dynamo via ``ltx_core::na_residual_w_chunked``; NA uses
    ``attn.attention_function`` (bound for the op body via contextvar).
    """
    if attn.w_chunks < 1:
        raise ValueError(f"w_chunks must be >= 1, got {attn.w_chunks}")
    packed = _pack_residual_weights(attn, norm, x)
    with _bound_chunked_attention(attn.attention_function, attn):
        _na_residual_w_chunked_op(x, scale, shift, *packed, attn.w_chunks)
    return x
