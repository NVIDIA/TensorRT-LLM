# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel wrappers for WinoVae: the CuTe DSL norm kernel (:mod:`jit_norm`, :mod:`kernels.norm`),
the upsample gather and the portable spatial Winograd conv. Each wrapper states
its layout, dtype and shape preconditions; choosing a fallback when they do not hold is the
caller's job (``can_*`` / ``*_available`` predicates), not the wrapper's. The frame operations
(:mod:`kernels.frame_ops`) and the Blackwell spatial Winograd (:mod:`kernels.blackwell.spatial`)
are called directly."""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch.visual_gen.models.wan import jit_norm
from tensorrt_llm._torch.visual_gen.models.wan.jit_norm import FrameGeometry, rms_norm_silu_rows

from .wino_tables import SPATIAL_WINO


def kernels_available() -> bool:
    """The norm kernel (:mod:`jit_norm`, every variant) can run."""
    return jit_norm_available()


def rms_norm_silu_supported(x: torch.Tensor) -> bool:
    return (
        kernels_available()
        and x.is_cuda
        and x.dtype is torch.bfloat16
        and x.shape[1] % 8 == 0
        and _channel_rows(x) is not None
    )


def _channel_rows(x: torch.Tensor) -> torch.Tensor | None:
    """View a channels-last NC(T)HW tensor as [positions, C] without copying."""
    if x.dim() == 5:
        if not x.is_contiguous(memory_format=torch.channels_last_3d):
            return None
        return x.permute(0, 2, 3, 4, 1).reshape(-1, x.shape[1])
    if x.dim() == 4:
        if not x.is_contiguous(memory_format=torch.channels_last):
            return None
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    return None


def rms_norm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    *,
    reference_rounding: bool = False,
    in_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    silu: bool = True,
    out_geometry: FrameGeometry | None = None,
) -> torch.Tensor:
    """The norm kernel (:mod:`jit_norm` in row mode): RMSNorm over channels (+ SiLU) of a
    channels-last bf16 CUDA tensor whose channel count is a multiple of 8
    (``rms_norm_silu_supported``); fp32 arithmetic, bf16 result of ``x``'s shape.
    ``silu=False`` returns the RMSNorm (with ``bias``) without the SiLU.

    ``in_bias`` is a deferred convolution bias, applied to ``x`` first as a bf16 add.
    ``out`` is an optional channels-last destination of ``x``'s shape (a frame-buffer
    window); the result is written there and returned. With ``out_geometry`` the destination
    holds frames with halo lines and only their logical positions are written.
    """
    rows = _channel_rows(x)
    if rows is None:
        raise ValueError("rms_norm_silu needs a channels-last CUDA tensor")
    dst = None
    if out is not None:
        dst = _channel_rows(out)
        if dst is None:
            raise ValueError("out must be a channels-last CUDA tensor")
    _, out_rows = rms_norm_silu_rows(
        rows,
        gamma.reshape(-1),
        bias.reshape(-1) if bias is not None else None,
        float(scale),
        reference_rounding=reference_rounding,
        in_bias=in_bias.reshape(-1) if in_bias is not None else None,
        out=dst,
        silu=silu,
        frames=_frame_count(x),
        out_geometry=out_geometry,
        return_sum=False,
    )
    return out if out is not None else _rows_to_like(out_rows, x)


def _frame_count(x: torch.Tensor) -> int:
    """Frames of a channels-last ``[N, C, T, H, W]`` or ``[N, C, H, W]`` tensor."""
    return x.shape[0] * x.shape[2] if x.dim() == 5 else x.shape[0]


def _rows_to_like(rows: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """View a [rows, C] result back as the channels-last 4D/5D tensor it came from."""
    if like.dim() == 5:
        n, c, t, h, w = like.shape
        return rows.view(n, t, h, w, c).permute(0, 4, 1, 2, 3)
    n, c, h, w = like.shape
    return rows.view(n, h, w, c).permute(0, 3, 1, 2)


def add_rms_norm_silu(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    *,
    reference_rounding: bool = False,
    in_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_geometry: FrameGeometry | None = None,
    return_sum: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """``s = (x + in_bias) + residual`` (bf16-rounded) and ``SiLU(RMSNorm(s))``; returns
    ``(s, act)``, with ``s`` None (not stored) unless ``return_sum``.
    ``out``: optional channels-last destination for the activation (with ``out_geometry``:
    frames with halo lines, logical positions written)."""
    rows = _channel_rows(x)
    rrows = _channel_rows(residual)
    if rows is None or rrows is None:
        raise ValueError("add_rms_norm_silu needs channels-last CUDA tensors")
    dst = _channel_rows(out) if out is not None else None
    if out is not None and dst is None:
        raise ValueError("out must be a channels-last CUDA tensor")
    s_rows, a_rows = rms_norm_silu_rows(
        rows,
        gamma.reshape(-1),
        bias.reshape(-1) if bias is not None else None,
        float(scale),
        reference_rounding=reference_rounding,
        in_bias=in_bias.reshape(-1) if in_bias is not None else None,
        residual=rrows,
        out=dst,
        frames=_frame_count(x),
        out_geometry=out_geometry,
        return_sum=return_sum,
    )
    s = _rows_to_like(s_rows, x) if s_rows is not None else None
    return s, (out if out is not None else _rows_to_like(a_rows, x))


def apply_in_bias(x: torch.Tensor, in_bias: torch.Tensor | None) -> torch.Tensor:
    """Eager form of the deferred convolution bias: ``x + bias`` broadcast over channels."""
    if in_bias is None:
        return x
    return x + in_bias.reshape(1, -1, *([1] * (x.dim() - 2)))


# ---------------------------------------------------------------------------
# Temporal Winograd F(order, 3), order 2 or 4: norm kernels that emit / consume the
# transformed frames
# ---------------------------------------------------------------------------


# Temporal Winograd F(order, 3) transforms (the norm kernel reads BT and AT from here). Order 2 is the
# classic F(2,3); order 4 uses the points (0, 1, -1, 2, -1/2, inf), whose output transform AT
# holds only powers of two (exact products) and whose input transform BT holds multiples of
# 1/2 (exact products of 16-bit values in fp32).
TEMPORAL_WINO: dict[int, dict[str, list[list[float]]]] = {
    2: dict(
        BT=[[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]],
        G=[[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]],
        AT=[[1, 1, 1, 0], [0, 1, -1, -1]],
    ),
    4: dict(
        BT=[
            [1, 1.5, -2, -1.5, 1, 0],
            [0, -1, -2.5, -0.5, 1, 0],
            [0, 1, 0.5, -2.5, 1, 0],
            [0, -0.5, -1, 0.5, 1, 0],
            [0, 2, -1, -2, 1, 0],
            [0, 1, 1.5, -2, -1.5, 1],
        ],
        G=[
            [1, 0, 0],
            [-1 / 3, -1 / 3, -1 / 3],
            [1 / 3, -1 / 3, 1 / 3],
            [1 / 15, 2 / 15, 4 / 15],
            [-16 / 15, 8 / 15, -4 / 15],
            [0, 0, 1],
        ],
        AT=[
            [1, 1, 1, 1, 1, 0],
            [0, 1, -1, 2, -0.5, 0],
            [0, 1, 1, 4, 0.25, 0],
            [0, 1, -1, 8, -0.125, 1],
        ],
    ),
}


def winograd_weights(
    weight: torch.Tensor, order: int = 2, dtype: torch.dtype | None = None
) -> list[torch.Tensor]:
    """``[Cout, Cin, 3, kh, kw]`` -> ``order + 2`` channels-last filters ``[Cout, Cin, kh, kw]``,
    ``G_k = sum_i G[k][i] g_i`` over the temporal taps (fp32 math, rounded once to ``dtype``,
    the weight's dtype by default). Order 2: ``g0, (g0+g1+g2)/2, (g0-g1+g2)/2, g2``."""
    taps = weight.detach().float()  # [Cout, Cin, 3, kh, kw]
    gm = torch.tensor(TEMPORAL_WINO[order]["G"], dtype=torch.float32, device=weight.device)
    gs = torch.einsum("kt,oitab->koiab", gm, taps)  # [comps, Cout, Cin, kh, kw]
    out_dtype = weight.dtype if dtype is None else dtype
    return [g.to(out_dtype).contiguous(memory_format=torch.channels_last) for g in gs]


# Spatial filter transforms G of F(2x2, 3x3) and F(4x4, 3x3) (:mod:`wino_tables`, shared with
# the sm100 path), applied in fp64 and rounded once. The matching B^T and A^T live in
# kernels/spatial_winograd.py.
_WINO_G = {tile: torch.tensor(SPATIAL_WINO[tile]["G"], dtype=torch.float64) for tile in (2, 4)}


def winograd_spatial_weights(weight: torch.Tensor, tile: int = 2, order: int = 2) -> torch.Tensor:
    """``[Cout, Cin, 3, 3, 3]`` -> ``U [(order+2) * (tile+2)^2, Cin, Cout]`` fp16: the temporal
    components of :func:`winograd_weights`, each transformed as ``G g G^T`` for spatial
    F(tile x tile, 3x3), ordered ``component * (tile+2)^2 + row * (tile+2) + col`` to match
    :func:`kernels.spatial_winograd.wino_spatial_in`. The transformed data is fp16 end to end (V, U, M)."""
    taps = weight.detach().double()  # [Cout, Cin, 3, 3, 3]
    gt = torch.tensor(TEMPORAL_WINO[order]["G"], dtype=torch.float64, device=weight.device)
    comps = torch.einsum("kt,oitab->koiab", gt, taps)  # [comps, Cout, Cin, 3, 3]
    gm = _WINO_G[tile].to(weight.device)
    u = torch.einsum("xa,noiab,yb->nxyoi", gm, comps, gm)  # [comps, a, a, Cout, Cin]
    cout, cin = weight.shape[:2]
    coords = (tile + 2) ** 2
    return u.reshape((order + 2) * coords, cout, cin).transpose(1, 2).contiguous().to(torch.float16)


def winograd_spatial_taps_weights(weight: torch.Tensor, tile: int, taps: int) -> torch.Tensor:
    """Filters for :func:`wino_spatial_conv_taps`: ``[Cout, Cin, 3, 3, 3]`` ->
    ``U [(tile+2)^2, taps * Cin, Cout]`` fp16. ``taps = 1`` keeps the current-frame tap
    (zero history: the first frame of a clip, every image); ``taps = 3`` stacks the taps
    for frames t-2, t-1, t along K, in that order, so one GEMM sums the causal 3x3x3 conv."""
    g = weight.detach().double()[:, :, 3 - taps :]  # [Cout, Cin, taps, 3, 3]
    gm = _WINO_G[tile].to(weight.device)
    u = torch.einsum("xa,oitab,yb->xytio", gm, g, gm)  # [a, a, taps, Cin, Cout]
    cout, cin = weight.shape[:2]
    return u.reshape((tile + 2) ** 2, taps * cin, cout).contiguous().to(torch.float16)


def wino_spatial_conv_taps(
    taps: list[torch.Tensor],
    zero_front: int,
    u: torch.Tensor,
    height: int,
    width: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """One output frame of a causal 3x3x3 conv (padding 1) by spatial Winograd alone, from
    input frames in separate buffers: ``zero_front`` zero frames, then the frames of each
    tensor in ``taps`` (``[n, H*W, Cin]`` bf16, oldest first; the causal history and the new
    frame, no concatenation). ``u`` from :func:`winograd_spatial_taps_weights`; the frames are
    stacked along the GEMM's K axis, so one batched GEMM sums the conv. Returns ``[H*W, Cout]``
    bf16, the bias added by the output transform."""
    tile = spatial_tile_of(u, 1)
    tile_rows = -(-height // tile)
    cout = u.shape[2]
    from .kernels.spatial_winograd import wino_spatial_in_taps, wino_spatial_out

    v = wino_spatial_in_taps(
        [t.view(t.shape[0], height, width, t.shape[-1]) for t in taps],
        zero_front,
        0,
        tile_rows,
        tile,
    )
    out = torch.empty((1, height, width, cout), dtype=taps[0].dtype, device=taps[0].device)
    m = torch.bmm(v, u)
    wino_spatial_out(m, out, 1, 0, tile_rows, tile, bias)
    return out.view(height * width, cout)


def spatial_tile_of(u: torch.Tensor, components: int = 4) -> int:
    """The tile size a :func:`winograd_spatial_weights` tensor was built for."""
    return int(round((u.shape[0] // components) ** 0.5)) - 2


@functools.lru_cache(maxsize=None)
def jit_norm_available() -> bool:
    """cuda-python and the CuTe DSL (``nvidia-cutlass-dsl``) for the norm kernel, and a GPU."""
    try:
        import cutlass.cute  # noqa: F401
        from cuda.bindings import driver  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def wino_spatial_conv_cublas(
    d: torch.Tensor, u: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """3x3 conv (padding 1) of each of the ``order + 2`` Winograd inputs (4 or 6) by its filter as
    spatial Winograd: ``D [comps, pairs, H*W, Cin]`` (bf16 or fp16) -> ``M [comps, pairs, H*W,
    Cout]`` in ``D``'s dtype (bias-free, fp32 accumulate).
    Input and output transforms are CuTe DSL kernels (:mod:`kernels.spatial_winograd`); the batched GEMMs
    are one cuBLAS call. The tile size (2 or 4) follows the filters ``u``."""
    comps, pairs, positions, cin = d.shape
    tile = spatial_tile_of(u, comps)
    frames = d.view(comps * pairs, height, width, cin)
    tile_rows = -(-height // tile)
    from .kernels.spatial_winograd import wino_spatial_in, wino_spatial_out

    v = wino_spatial_in(frames, comps, 0, tile_rows, tile)
    m = torch.bmm(v, u)
    out = torch.empty((comps * pairs, height, width, u.shape[2]), dtype=d.dtype, device=d.device)
    wino_spatial_out(m, out, comps, 0, tile_rows, tile)
    return out.view(comps, pairs, positions, u.shape[2])


def frames_view(x: torch.Tensor) -> torch.Tensor:
    """Batch-1 channels-last ``[1, C, T, H, W]`` -> ``[T, H*W, C]`` view (no copy)."""
    _, c, t, h, w = x.shape
    rows = _channel_rows(x)
    if rows is None or x.shape[0] != 1:
        raise ValueError("frames_view needs a batch-1 channels-last CUDA tensor")
    return rows.view(t, h * w, c)


def frames_to_video(f: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """``[T, H*W, C]`` -> channels-last ``[1, C, T, H, W]`` view."""
    t, _, c = f.shape
    return f.view(1, t, h, w, c).permute(0, 4, 1, 2, 3)


def pairs_to_images(d: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """``[pairs, H*W, C]`` -> channels-last ``[pairs, C, H, W]`` view for conv2d."""
    p, _, c = d.shape
    return d.view(p, h, w, c).permute(0, 3, 1, 2)


def images_to_pairs(m: torch.Tensor) -> torch.Tensor:
    """channels-last ``[pairs, C, H, W]`` -> ``[pairs, H*W, C]`` view."""
    rows = _channel_rows(m)
    if rows is None:
        raise ValueError("images_to_pairs needs a channels-last CUDA tensor")
    return rows.view(m.shape[0], m.shape[2] * m.shape[3], m.shape[1])


def add_bias_and_residual(
    x: torch.Tensor | None,
    m: list[torch.Tensor] | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    frames: int,
    height: int,
    width: int,
    pad_h: int = 0,
    pad_w: int = 0,
    products_geometry: FrameGeometry | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``frames + bias + residual`` in frame layout ``[T, P, C]``, eager rounding order; the
    frames are ``x``, or the temporal Winograd products ``m`` after their output transform
    (:func:`jit_norm.add_bias_and_residual`).

    With ``pad_h``/``pad_w`` the result is ``[T, (H+pad_h)*(W+pad_w), C]`` with zero rows/columns
    appended (the downsampler's ``ZeroPad2d((0, 1, 0, 1))``). ``products_geometry``: the
    products carry halo rows / columns (spatially sharded decode). ``out``: a contiguous
    destination of the result's shape (e.g. a conv's frame-buffer window), written and
    returned."""
    return jit_norm.add_bias_and_residual(
        x=x,
        m=m,
        bias=bias.reshape(-1).contiguous() if bias is not None else None,
        residual=residual,
        frames=frames,
        height=height,
        width=width,
        pad_h=pad_h,
        pad_w=pad_w,
        products_geometry=products_geometry,
        out=out,
    )


def wino_norm_silu(
    x: torch.Tensor | None,
    m: list[torch.Tensor] | None,
    residual: torch.Tensor | None,
    in_bias: torch.Tensor | None,
    cache: torch.Tensor | None,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    frames: int,
    *,
    return_sum: bool,
    reference_rounding: bool = False,
    order: int = 2,
    fp16_inputs: bool = False,
    products_geometry: FrameGeometry | None = None,
    inputs_geometry: FrameGeometry | None = None,
    inputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """The temporal Winograd norm (:func:`jit_norm.wino_norm_silu`); per-channel tensors may
    have any shape with ``C`` elements."""
    return jit_norm.wino_norm_silu(
        x=x,
        m=m,
        residual=residual,
        in_bias=in_bias.reshape(-1) if in_bias is not None else None,
        cache=cache,
        gamma=gamma.reshape(-1),
        bias=bias.reshape(-1) if bias is not None else None,
        scale=float(scale),
        frames=frames,
        return_sum=return_sum,
        order=order,
        fp16_inputs=fp16_inputs,
        reference_rounding=reference_rounding,
        products_geometry=products_geometry,
        inputs_geometry=inputs_geometry,
        inputs=inputs,
    )


# ---------------------------------------------------------------------------
# Nearest-2x upsample + 3x3 conv as one 2x2 conv on the low-res grid + phase gather
# ---------------------------------------------------------------------------


def upsample_gather_supported(cout: int, factor_t: int, repeats: int) -> bool:
    """The gather kernel runs and covers these shapes (:func:`kernels.upsample_gather.can_upsample_gather`)."""
    if not kernels_available():
        return False
    from .kernels.upsample_gather import can_upsample_gather

    return can_upsample_gather(cout, factor_t, repeats)


def pack_upsample_weights(
    weight: torch.Tensor, bias: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fold nearest-2x upsampling into a 3x3 conv's weights.

    ``weight`` is ``[Cout, Cin, 3, 3]``. Returns ``[4*Cout, Cin, 2, 2]`` with output
    phase ``2p + q`` in channel block ``p*2+q``: with the conv run at ``padding=1``,
    temp position ``j`` covers low-res rows ``(j-1, j)`` and phase ``p`` of output ``i``
    reads temp row ``i + p``, so original tap ``dy`` lands on ``ky = (p+dy)//2 - p + 1``.
    Composed in fp32, cast once; the bias is replicated per phase.
    """
    cout, cin = weight.shape[:2]
    w = weight.detach().float()
    packed = w.new_zeros(4 * cout, cin, 2, 2)
    for p in (0, 1):
        for q in (0, 1):
            block = packed[(2 * p + q) * cout : (2 * p + q + 1) * cout]
            for dy in (-1, 0, 1):
                ky = (p + dy) // 2 - p + 1
                for dx in (-1, 0, 1):
                    kx = (q + dx) // 2 - q + 1
                    block[:, :, ky, kx] += w[:, :, dy + 1, dx + 1]
    packed = packed.to(weight.dtype).contiguous(memory_format=torch.channels_last)
    packed_bias = bias.detach().repeat(4).contiguous() if bias is not None else None
    return packed, packed_bias


def upsample_gather(
    temp: torch.Tensor,
    shortcut: torch.Tensor | None,
    batch: int,
    frames: int,
    factor_t: int = 1,
    repeats: int = 1,
    crop: int = 0,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Phase gather (:mod:`kernels.upsample_gather`) with the conv bias and the DupUp3D shortcut add
    folded in."""
    from .kernels.upsample_gather import upsample_gather as kernel

    return kernel(temp, shortcut, batch, frames, factor_t, repeats, crop, bias)
