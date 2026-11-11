"""Opaque full-volume abs-RoPE for deterministic (pre-diffusion) NA.
Owns QKV + opaque ``custom_op`` packaging for det ``NA.forward``. Det stages
have differently shaped T/H/W; the opaque op keeps Dynamo from specializing
on each stage shape. Diffusion paths use ``diff_attn/`` RoPE — not this file.
No T/H/W origin/offset is threaded for tiled decode, and none is needed:
every attention call here is ``natten.na3d``, a local window with no
cross-tile tokens. Absolute-vs-local RoPE differs by a global phase that
cancels inside the attention softmax over that window, so the attention
output is unchanged. Since every tiled-decode call processes exactly one
tile in isolation, using each tile's local 0-based positions is identical
to using its true absolute origin. Absolute origin still matters for
whether a tile contains the latent's first frame (``drop_leading_frame``),
which is handled outside RoPE in the decoder stage / per-tile decode.
"""

from __future__ import annotations

import torch

try:
    from ltx_kernels.vae.rope_dsl import abs_rope_supported, run_abs_rope
except ImportError:  # ltx-kernels is optional; the slab path below covers every device
    abs_rope_supported = None
    run_abs_rope = None

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae import decode_workspace
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.ops import apply_rms_norm_uint32_safe
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.rope_math import (
    DEFAULT_ABS_ROPE_NUM_TILES,
    AbsRopeTableCache,
    RopeAxis,
    RopePositions,
    cached_rope_tables,
    h_positions,
    rot_abs_axis_impl,
    t_positions,
)


def _apply_opaque_rope_slab(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    w_pos: torch.Tensor,
    compute_dtype: torch.dtype,
    t_pos: torch.Tensor | None = None,
    w_pos_key: RopePositions | None = None,
    cache: AbsRopeTableCache | None = None,
) -> torch.Tensor:
    """Rotate one W-extent with raw abs-RoPE (runs inside the opaque op).
    ``t_pos`` overrides the default integer ``arange`` on the first axis; the keyframe
    stream passes its (possibly fractional) plane times there. ``w_pos_key`` names ``w_pos``
    by value for ``cache``.
    """
    d_t, d_h, _ = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    t = x.shape[1]
    h = x.shape[2]
    positions_t = t_positions(t, x.device) if t_pos is None else t_pos
    xt = rot_abs_axis_impl(
        x[..., :d_t],
        positions_t,
        inv_t,
        axis=1,
        compute_dtype=compute_dtype,
        pos_key=RopePositions(RopeAxis.T, t) if t_pos is None else None,
        cache=cache,
    )
    xh = rot_abs_axis_impl(
        x[..., d_t : d_t + d_h],
        h_positions(h, x.device),
        inv_h,
        axis=2,
        compute_dtype=compute_dtype,
        pos_key=RopePositions(RopeAxis.H, h),
        cache=cache,
    )
    xw = rot_abs_axis_impl(
        x[..., d_t + d_h :], w_pos, inv_w, axis=3, compute_dtype=compute_dtype, pos_key=w_pos_key, cache=cache
    )
    return torch.cat([xt, xh, xw], dim=-1)


def _apply_opaque_tiled_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_tiles: int,
    compute_dtype: torch.dtype,
    t_pos: torch.Tensor | None = None,
    cache: AbsRopeTableCache | None = None,
) -> torch.Tensor:
    """Fixed-``num_tiles`` W split + per-slab rotation (body of the opaque op); tables from ``cache`` if given.
    With ``ltx-kernels`` installed, a CUDA bf16 volume at the fp32 compute dtype is rotated in
    one launch instead (:func:`ltx_kernels.vae.rope_dsl.run_abs_rope`); the split only bounds
    the slab path's fp32 temporaries, so it does not change the angles.
    The keyframe stream shares the video stream's W extent at every stage, so passing the
    same ``num_tiles`` yields identical slab boundaries and identical ``w_pos`` -- which
    is what keeps the two streams' W phases comparable inside the joint softmax.
    """
    if compute_dtype == torch.float32 and abs_rope_supported is not None and abs_rope_supported(x, rope_split):
        return run_abs_rope(x, rope_split, _rope_tables(x, inv_freqs, t_pos, cache))
    slabs = torch.chunk(x, num_tiles, dim=3)
    w_off = 0
    parts: list[torch.Tensor] = []
    for slab in slabs:
        w_slab = slab.shape[3]
        w_pos = torch.arange(w_slab, dtype=torch.float32, device=x.device) + w_off
        parts.append(
            _apply_opaque_rope_slab(
                slab,
                rope_split,
                inv_freqs,
                w_pos=w_pos,
                compute_dtype=compute_dtype,
                t_pos=t_pos,
                w_pos_key=RopePositions(RopeAxis.W, w_slab, w_off),
                cache=cache,
            )
        )
        w_off = w_off + w_slab
    return torch.cat(parts, dim=3)


def _rope_tables(
    x: torch.Tensor,
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    t_pos: torch.Tensor | None,
    cache: AbsRopeTableCache | None,
) -> tuple[torch.Tensor, ...]:
    """``(cos_t, sin_t, cos_h, sin_h, cos_w, sin_w)`` over ``x``'s T/H/W, as the slab path computes them.
    torch's own fp32 ``cos`` / ``sin`` of ``pos * inv``: the slab offsets make the W positions
    the plain ``arange(W)``, so these are exactly the angles :func:`rot_abs_axis_impl` uses.
    """
    _, t, h, w, _, _ = x.shape

    def build() -> tuple[torch.Tensor, ...]:
        positions = (
            t_positions(t, x.device) if t_pos is None else t_pos.to(device=x.device, dtype=torch.float32),
            h_positions(h, x.device),
            torch.arange(w, dtype=torch.float32, device=x.device),
        )
        tables: list[torch.Tensor] = []
        for pos, inv in zip(positions, inv_freqs, strict=True):
            angles = pos[:, None] * inv[None, :]
            tables += [angles.cos(), angles.sin()]
        return tuple(tables)

    # Integer positions are named by their extents; explicit plane times are not cached.
    key = (
        None
        if t_pos is not None
        else (
            RopePositions(RopeAxis.T, t),
            RopePositions(RopeAxis.H, h),
            RopePositions(RopeAxis.W, w),
            *(inv.data_ptr() for inv in inv_freqs),
            x.device,
        )
    )
    return cached_rope_tables(cache, key, build)


@torch.library.custom_op("ltx_core::abs_rope", mutates_args=())
def _abs_rope_op(
    x: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    num_tiles: int,
    compute_dtype_is_bf16: bool,
) -> torch.Tensor:
    """Opaque out-of-place abs-RoPE: Dynamo sees one node."""
    compute_dtype = torch.bfloat16 if compute_dtype_is_bf16 else torch.float32
    return _apply_opaque_tiled_rope(
        x,
        (d_t, d_h, d_w),
        (inv_t, inv_h, inv_w),
        num_tiles=num_tiles,
        compute_dtype=compute_dtype,
    )


@_abs_rope_op.register_fake
def _abs_rope_fake(
    x: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    num_tiles: int,
    compute_dtype_is_bf16: bool,
) -> torch.Tensor:
    del inv_t, inv_h, inv_w, d_t, d_h, d_w, num_tiles, compute_dtype_is_bf16
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def _apply_opaque_abs_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_tiles: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"compute_dtype must be float32 or bfloat16, got {compute_dtype}")
    d_t, d_h, d_w = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    return _abs_rope_op(
        x,
        inv_t,
        inv_h,
        inv_w,
        d_t,
        d_h,
        d_w,
        num_tiles,
        compute_dtype == torch.bfloat16,
    )


@torch.library.custom_op("ltx_core::abs_rope_at_t", mutates_args=())
def _abs_rope_at_t_op(
    x: torch.Tensor,
    t_pos: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    num_tiles: int,
    compute_dtype_is_bf16: bool,
) -> torch.Tensor:
    """Opaque abs-RoPE with caller-supplied (possibly fractional) first-axis positions."""
    compute_dtype = torch.bfloat16 if compute_dtype_is_bf16 else torch.float32
    return _apply_opaque_tiled_rope(
        x,
        (d_t, d_h, d_w),
        (inv_t, inv_h, inv_w),
        num_tiles=num_tiles,
        compute_dtype=compute_dtype,
        t_pos=t_pos,
    )


@_abs_rope_at_t_op.register_fake
def _abs_rope_at_t_fake(
    x: torch.Tensor,
    t_pos: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    num_tiles: int,
    compute_dtype_is_bf16: bool,
) -> torch.Tensor:
    del t_pos, inv_t, inv_h, inv_w, d_t, d_h, d_w, num_tiles, compute_dtype_is_bf16
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def _rope_config(
    attn: object, x: torch.Tensor
) -> tuple[tuple[torch.Tensor, ...], int, torch.dtype, AbsRopeTableCache | None]:
    """``(inv_freqs, num_tiles, compute_dtype, cache)`` read off an attention module.
    ``cache`` is the :class:`AbsRopeTableCache` the module holds for an enclosing
    :func:`~ltx_core.model.video_vae.transformer.rope_math.abs_rope_table_cache` scope, if this thread may
    read it (never while tracing or capturing). With one, the inverse frequencies are the cache's tensors of
    the module's ``rope_dim_split`` and ``rope_base``: the module's ``rope_inv_*`` buffers are
    :func:`rope_inv_freqs` of those (non-persistent, so never loaded), cast with the module, so the values
    are the same and every block of one configuration shares one set of cached tables.
    """
    buffers = (attn.rope_inv_t, attn.rope_inv_h, attn.rope_inv_w)  # type: ignore[attr-defined]
    # Checked before the slot is read, so a traced graph never guards on the scope's cache.
    cache = None if torch.compiler.is_compiling() else getattr(attn, "rope_table_cache", None)
    if cache is not None and not cache.usable():
        cache = None
    if cache is None:
        inv_freqs = tuple(inv.to(device=x.device) for inv in buffers)
    else:
        base = float(attn.rope_base)  # type: ignore[attr-defined]
        inv_freqs = tuple(
            cache.inv_freqs(dim, base, inv.dtype, x.device)
            for dim, inv in zip(attn.rope_dim_split, buffers, strict=True)  # type: ignore[attr-defined]
        )
    num_tiles = getattr(attn, "rope_num_tiles", DEFAULT_ABS_ROPE_NUM_TILES)
    compute_dtype = getattr(attn, "rope_compute_dtype", torch.float32)
    return inv_freqs, num_tiles, compute_dtype, cache


def _det_project_qkv(attn: object, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared Q/K/V proj + norm + scale for :func:`det_qkv_rope` and ``_at_times``."""
    q, k, v = attn.project_qkv(x)  # type: ignore[attr-defined]
    q = apply_rms_norm_uint32_safe(attn.q_norm, q)  # type: ignore[attr-defined]
    k = apply_rms_norm_uint32_safe(attn.k_norm, k)  # type: ignore[attr-defined]
    q = q * attn.scale  # type: ignore[attr-defined]
    return q, k, v


def det_qkv_rope_at_times(
    attn: object,
    x: torch.Tensor,
    t_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K/V proj + norm/scale + abs-RoPE with explicit first-axis positions.
    The keyframe-stream counterpart of :func:`det_qkv_rope`. ``t_pos`` is ``(P,)`` float
    stage times *in the same origin* the video stream's RoPE uses, so the joint softmax
    sees true relative offsets: absolute-vs-local RoPE only cancels as a global phase
    when every token in the softmax shares one origin, which no longer holds once
    keyframe tokens join a video window.
    """
    if t_pos.ndim != 1 or t_pos.shape[0] != x.shape[1]:
        raise ValueError(f"t_pos must be ({x.shape[1]},) to match the plane axis, got {tuple(t_pos.shape)}")
    q, k, v = _det_project_qkv(attn, x) if torch.compiler.is_compiling() else det_project_qkv_eager(attn, x)

    inv_freqs, num_tiles, compute_dtype, cache = _rope_config(attn, x)
    d_t, d_h, d_w = attn.rope_dim_split  # type: ignore[attr-defined]
    positions = t_pos.to(device=x.device, dtype=torch.float32)
    if cache is not None:
        # Eager under a table scope: the op's body itself, so the H/W tables come from the scope's cache.
        rotated = [
            _apply_opaque_tiled_rope(
                tensor,
                (d_t, d_h, d_w),
                inv_freqs,
                num_tiles=num_tiles,
                compute_dtype=compute_dtype,
                t_pos=positions,
                cache=cache,
            )
            for tensor in (q, k)
        ]
        return rotated[0], rotated[1], v
    rotated = [
        _abs_rope_at_t_op(
            tensor,
            positions,
            *inv_freqs,
            d_t,
            d_h,
            d_w,
            num_tiles,
            compute_dtype == torch.bfloat16,
        )
        for tensor in (q, k)
    ]
    return rotated[0], rotated[1], v


def det_qkv_rope(attn: object, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K/V proj + norm/scale + opaque full-volume abs-RoPE."""
    q, k, v = _det_project_qkv(attn, x)
    inv_freqs, num_tiles, compute_dtype, cache = _rope_config(attn, x)
    rope_split = attn.rope_dim_split  # type: ignore[attr-defined]
    if cache is not None:
        # Eager under a table scope: the op's body itself, with the scope's cached tables.
        q, k = (
            _apply_opaque_tiled_rope(
                t, rope_split, inv_freqs, num_tiles=num_tiles, compute_dtype=compute_dtype, cache=cache
            )
            for t in (q, k)
        )
        return q, k, v
    q = _apply_opaque_abs_rope(q, rope_split, inv_freqs, num_tiles=num_tiles, compute_dtype=compute_dtype)
    k = _apply_opaque_abs_rope(k, rope_split, inv_freqs, num_tiles=num_tiles, compute_dtype=compute_dtype)
    return q, k, v


def linear_into_phase_buffer(linear: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """``linear(x)`` for a contiguous channels-last ``x``, into a workspace phase buffer when one is active.
    Exactly ``addmm(bias, x.view(-1, K), W.t())`` with an ``out=``, so the values are those of ``nn.Linear``.
    """
    if not x.is_contiguous() or linear.bias is None:
        return linear(x)
    rows = x.view(-1, linear.in_features)
    out = decode_workspace.phase_empty((rows.shape[0], linear.out_features), dtype=x.dtype, device=x.device)
    torch.addmm(linear.bias, rows, linear.weight.t(), out=out)
    return out.view(*x.shape[:-1], linear.out_features)


def det_project_qkv_lean(attn: object, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`_det_project_qkv` for the eager dual-stream det blocks: projections into workspace phase
    buffers, norms (and Q's scale) in place. Same values."""
    batch, t, h, w, _ = x.shape
    qkv = attn.qkv  # type: ignore[attr-defined]
    shape = (batch, t, h, w, attn.num_heads, attn.head_dim)  # type: ignore[attr-defined]
    q = linear_into_phase_buffer(qkv.to_q, x).view(shape)
    k = linear_into_phase_buffer(qkv.to_k, x).view(shape)
    v = linear_into_phase_buffer(qkv.to_v, x).view(shape)
    q = attn.q_norm.forward_(q, scale=attn.scale)  # type: ignore[attr-defined]
    k = attn.k_norm.forward_(k)  # type: ignore[attr-defined]
    return q, k, v


def det_project_qkv_eager(attn: object, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`_det_project_qkv` on fresh projections with the norms (and Q's scale) in place. Same values."""
    q, k, v = attn.project_qkv(x)  # type: ignore[attr-defined]
    q = attn.q_norm.forward_(q, scale=attn.scale)  # type: ignore[attr-defined]
    k = attn.k_norm.forward_(k)  # type: ignore[attr-defined]
    return q, k, v


def det_rope_qk_(attn: object, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The RoPE half of :func:`det_qkv_rope`, rotating contiguous ``q`` and ``k`` in place.
    One :func:`~ltx_kernels.vae.rope_dsl.run_abs_rope` launch per tensor where it applies, else W slab by
    W slab, each slab read in full before it is overwritten; both equal the out-of-place rotation bit for
    bit. Eager callers only: the compiled det path keeps the pure ``ltx_core::abs_rope`` op.
    """
    inv_freqs, num_tiles, compute_dtype, cache = _rope_config(attn, q)
    rope_split = attn.rope_dim_split  # type: ignore[attr-defined]
    for x in (q, k):
        if not x.is_contiguous():
            raise ValueError("det_rope_qk_ needs contiguous Q/K (fresh projections)")
        if compute_dtype == torch.float32 and abs_rope_supported is not None and abs_rope_supported(x, rope_split):
            run_abs_rope(x, rope_split, _rope_tables(x, inv_freqs, None, cache), out=x)
            continue
        w_off = 0
        for slab in torch.chunk(x, num_tiles, dim=3):
            w_slab = slab.shape[3]
            w_pos = torch.arange(w_slab, dtype=torch.float32, device=x.device) + w_off
            slab.copy_(
                _apply_opaque_rope_slab(
                    slab,
                    rope_split,
                    inv_freqs,
                    w_pos=w_pos,
                    compute_dtype=compute_dtype,
                    w_pos_key=RopePositions(RopeAxis.W, w_slab, w_off),
                    cache=cache,
                )
            )
            w_off = w_off + w_slab
    return q, k
