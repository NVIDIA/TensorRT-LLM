# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused kernels for WinoVae. Triton prototypes; each has an exact PyTorch fallback."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm._torch.visual_gen.models.wan.jit_norm import (
    FrameGeometry,
    UpsampleSource,
    combine,
    rms_norm_silu_rows,
    wino_norm_silu,
)

_NORM_EPS = 1e-12  # F.normalize default
_MAX_TILE_ELEMENTS = 4096


def cuda_op_available() -> bool:
    """The runtime-compiled norm / Winograd norm / combine kernel (:mod:`jit_norm`) can run."""
    return jit_norm_available()


def can_use_cuda_rms_norm_silu(x: torch.Tensor) -> bool:
    return (
        cuda_op_available()
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


def can_fuse_rms_norm_silu(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.dim() in (4, 5)
        and x.shape[1] <= _MAX_TILE_ELEMENTS
        and _channel_rows(x) is not None
    )


@triton.jit
def _rms_norm_silu_kernel(
    x_ptr,
    y_ptr,
    gamma_ptr,
    bias_ptr,
    n_rows,
    scale,
    eps,
    C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    REFERENCE_ROUNDING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # One program normalizes BLOCK_M positions; channels are contiguous per position.
    row0 = tl.program_id(0).to(tl.int64) * BLOCK_M
    rows = row0 + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    row_ok = rows < n_rows
    col_ok = cols < C
    mask = row_ok[:, None] & col_ok[None, :]
    offs = rows[:, None] * C + cols[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    sumsq = tl.sum(xf * xf, axis=1)
    denom = tl.maximum(tl.sqrt(sumsq), eps)
    n = xf / denom[:, None]
    gamma = tl.load(gamma_ptr + cols, mask=col_ok, other=0.0)
    if REFERENCE_ROUNDING:
        # Today's eager sequence: round to the activation dtype after normalize,
        # after each multiply and after the bias add; SiLU in fp32 on that value.
        n = n.to(x.dtype).to(tl.float32)
        r = (n * scale).to(x.dtype).to(tl.float32)
        r = (r * gamma.to(tl.float32)).to(x.dtype).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(bias_ptr + cols, mask=col_ok, other=0.0)
            r = (r + b.to(tl.float32)).to(x.dtype).to(tl.float32)
    else:
        r = n * scale * gamma.to(tl.float32)
        if HAS_BIAS:
            b = tl.load(bias_ptr + cols, mask=col_ok, other=0.0)
            r = r + b.to(tl.float32)
    y = r * tl.sigmoid(r)
    tl.store(y_ptr + offs, y.to(x.dtype), mask=mask)


def rms_norm_silu_cuda(
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
    """The CUDA kernel (:mod:`jit_norm` in row mode); same contract as ``rms_norm_silu``.
    ``silu=False`` returns the RMSNorm alone (plain form only: no bias, in_bias or out).

    ``in_bias`` is a deferred convolution bias, applied to ``x`` first as a bf16 add.
    ``out`` is an optional channels-last destination of ``x``'s shape (a frame-buffer
    window); the result is written there and returned. With ``out_geometry`` the destination
    holds frames with halo lines and only their logical positions are written.
    """
    rows = _channel_rows(x)
    if rows is None:
        raise ValueError("rms_norm_silu_cuda needs a channels-last CUDA tensor")
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


def add_rms_norm_silu_cuda(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """``s = (x + in_bias) + residual`` (bf16-rounded) and ``SiLU(RMSNorm(s))``; returns ``(s, act)``.
    ``out``: optional channels-last destination for the activation (with ``out_geometry``:
    frames with halo lines, logical positions written)."""
    rows = _channel_rows(x)
    rrows = _channel_rows(residual)
    if rows is None or rrows is None:
        raise ValueError("add_rms_norm_silu_cuda needs channels-last CUDA tensors")
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
    )
    return _rows_to_like(s_rows, x), (out if out is not None else _rows_to_like(a_rows, x))


def upsample_source(
    temp: torch.Tensor,
    shortcut: torch.Tensor | None,
    bias: torch.Tensor | None,
    factor_t: int,
    repeats: int,
    crop: int,
) -> UpsampleSource:
    """The kernel-side description of a fused upsample input from the model's tensors: ``temp``
    ``[T, 4C, H+1, W+1]`` channels-last (the packed 2x2 conv output), ``shortcut`` ``[1, Cin,
    Tr, H, W]`` channels-last or None (the DupUp3D input), ``bias`` ``[C]`` or None."""
    rows = _channel_rows(temp)
    if rows is None:
        raise ValueError("temp must be a channels-last CUDA tensor")
    t, c4, hp1, wp1 = temp.shape
    sc = None
    if shortcut is not None:
        sc = frames_view(_channels_last_3d_if_needed_rows(shortcut))
    if sc is None:
        factor_t, repeats, crop = 1, 1, 0
    return UpsampleSource(
        rows.view(t, hp1 * wp1, c4),
        sc,
        bias.reshape(-1) if bias is not None else None,
        factor_t,
        repeats,
        crop,
        hp1 - 1,
        wp1 - 1,
    )


def _channels_last_3d_if_needed_rows(x: torch.Tensor) -> torch.Tensor:
    return (
        x
        if x.is_contiguous(memory_format=torch.channels_last_3d)
        else x.contiguous(memory_format=torch.channels_last_3d)
    )


def upsample_rms_norm_silu_cuda(
    up: UpsampleSource,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    *,
    reference_rounding: bool = False,
    out: torch.Tensor | None = None,
    out_geometry: FrameGeometry | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The upsample gather fused into the row-mode norm: returns ``(x, SiLU(RMSNorm(x)))`` with
    ``x`` the gathered frames as channels-last ``[1, C, T, 2H, 2W]``; ``out`` is an optional
    channels-last destination for the activation (with ``out_geometry``: frames with halo
    lines, logical positions written)."""
    dst = _channel_rows(out) if out is not None else None
    if out is not None and dst is None:
        raise ValueError("out must be a channels-last CUDA tensor")
    s_rows, a_rows = rms_norm_silu_rows(
        None,
        gamma.reshape(-1),
        bias.reshape(-1) if bias is not None else None,
        float(scale),
        reference_rounding=reference_rounding,
        out=dst,
        out_geometry=out_geometry,
        upsample=up,
    )
    h, w = 2 * up.height, 2 * up.width
    x = frames_to_video(s_rows.view(up.frames, h * w, up.channels), h, w)
    return x, (
        out
        if out is not None
        else frames_to_video(a_rows.view(up.frames, h * w, up.channels), h, w)
    )


def apply_in_bias(x: torch.Tensor, in_bias: torch.Tensor | None) -> torch.Tensor:
    """Eager form of the deferred convolution bias: ``x + bias`` broadcast over channels."""
    if in_bias is None:
        return x
    return x + in_bias.reshape(1, -1, *([1] * (x.dim() - 2)))


def add_rms_norm_silu_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | float,
    scale: float,
    in_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    s = apply_in_bias(x, in_bias) + residual
    return s, rms_norm_silu_reference(s, gamma, bias, scale)


def rms_norm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    *,
    reference_rounding: bool = False,
    block_m: int | None = None,
    num_warps: int | None = None,
    enable_fp_fusion: bool | None = None,
) -> torch.Tensor:
    """SiLU(RMSNorm(x)) for a channels-last tensor, one memory pass (Triton prototype).

    ``gamma``/``bias`` are the module parameters in their stored dtype with any
    broadcast dims; ``scale`` is ``sqrt(C)``. ``reference_rounding`` reproduces
    the eager path's intermediate roundings; otherwise everything stays in fp32
    until the single store.
    """
    rows = _channel_rows(x)
    if rows is None:
        raise ValueError("rms_norm_silu needs a channels-last CUDA tensor")
    n_rows, C = rows.shape
    out = torch.empty_like(x)
    out_rows = _channel_rows(out)
    gamma_flat = gamma.reshape(-1)
    bias_flat = bias.reshape(-1) if bias is not None else gamma_flat  # dummy pointer when unused
    block_c = triton.next_power_of_2(C)
    block_m = block_m or max(1, min(64, _MAX_TILE_ELEMENTS // block_c))
    grid = (triton.cdiv(n_rows, block_m),)
    _rms_norm_silu_kernel[grid](
        rows,
        out_rows,
        gamma_flat,
        bias_flat,
        n_rows,
        float(scale),
        _NORM_EPS,
        C=C,
        HAS_BIAS=bias is not None,
        REFERENCE_ROUNDING=reference_rounding,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=num_warps or (4 if block_c <= 512 else 8),
        # FMA contraction would merge `r * gamma + b` across the intermediate rounding
        # that reference mode exists to reproduce.
        enable_fp_fusion=(not reference_rounding) if enable_fp_fusion is None else enable_fp_fusion,
    )
    return out


def rms_norm_reference(x: torch.Tensor, gamma: torch.Tensor, scale: float) -> torch.Tensor:
    """The eager WanRMSNorm.forward alone (no activation, no bias)."""
    needs_fp32 = x.dtype in (torch.float16, torch.bfloat16)
    normalized = F.normalize(x.float() if needs_fp32 else x, dim=1).to(x.dtype)
    return normalized * scale * gamma


def rms_norm_silu_reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | float,
    scale: float,
    in_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """The eager path, verbatim (WanRMSNorm.forward followed by F.silu)."""
    x = apply_in_bias(x, in_bias)
    needs_fp32 = x.dtype in (torch.float16, torch.bfloat16)
    normalized = F.normalize(x.float() if needs_fp32 else x, dim=1).to(x.dtype)
    return F.silu(normalized * scale * gamma + bias)


# ---------------------------------------------------------------------------
# Temporal Winograd F(2,3): norm kernels that emit / consume the transformed frames
# ---------------------------------------------------------------------------


# Temporal Winograd F(order, 3) transforms, mirrored in cuda/vae_wino_tables.cuh. Order 2 is the
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


# Spatial filter transforms G of F(2x2, 3x3) (Lavin & Gray, points 0, +-1, inf) and F(4x4, 3x3)
# (points 0, +-3/2, +-5/8, inf, shared with the sm100 path: the real point set with the
# lowest fp16 rounding sensitivity, 1.7e-3 per conv against 2.5e-3 for (0, +-1, 2, -1/2, inf)),
# applied in fp64 and rounded once. The matching B^T and A^T live in
# cpp/tensorrt_llm/kernels/vaeWinoSpatialKernel.cu.
_WINO_G = {
    2: torch.tensor([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], dtype=torch.float64),
    4: torch.tensor(
        [
            [256 / 225, 0, 0],
            [128 / 1071, 64 / 357, 32 / 119],
            [128 / 1071, -64 / 357, 32 / 119],
            [-2048 / 2975, -256 / 595, -32 / 119],
            [-2048 / 2975, 256 / 595, -32 / 119],
            [0, 0, 1],
        ],
        dtype=torch.float64,
    ),
}


def winograd_spatial_weights(weight: torch.Tensor, tile: int = 2, order: int = 2) -> torch.Tensor:
    """``[Cout, Cin, 3, 3, 3]`` -> ``U [(order+2) * (tile+2)^2, Cin, Cout]`` fp16: the temporal
    components of :func:`winograd_weights`, each transformed as ``G g G^T`` for spatial
    F(tile x tile, 3x3), ordered ``component * (tile+2)^2 + row * (tile+2) + col`` to match
    ``trtllm::vae_wino_spatial_in``. The transformed data is fp16 end to end (V, U, M)."""
    taps = weight.detach().double()  # [Cout, Cin, 3, 3, 3]
    gt = torch.tensor(TEMPORAL_WINO[order]["G"], dtype=torch.float64, device=weight.device)
    comps = torch.einsum("kt,oitab->koiab", gt, taps)  # [comps, Cout, Cin, 3, 3]
    gm = _WINO_G[tile].to(weight.device)
    u = torch.einsum("xa,noiab,yb->nxyoi", gm, comps, gm)  # [comps, a, a, Cout, Cin]
    cout, cin = weight.shape[:2]
    coords = (tile + 2) ** 2
    return u.reshape((order + 2) * coords, cout, cin).transpose(1, 2).contiguous().to(torch.float16)


def winograd_spatial_taps_weights(weight: torch.Tensor, tile: int, taps: int) -> torch.Tensor:
    """Filters for :func:`wino_spatial_conv_frames`: ``[Cout, Cin, 3, 3, 3]`` ->
    ``U [(tile+2)^2, taps * Cin, Cout]`` fp16. ``taps = 1`` keeps the current-frame tap
    (zero history: the first frame of a clip, every image); ``taps = 3`` stacks the taps
    for frames t-2, t-1, t along K, in that order, so one GEMM sums the causal 3x3x3 conv."""
    g = weight.detach().double()[:, :, 3 - taps :]  # [Cout, Cin, taps, 3, 3]
    gm = _WINO_G[tile].to(weight.device)
    u = torch.einsum("xa,oitab,yb->xytio", gm, g, gm)  # [a, a, taps, Cin, Cout]
    cout, cin = weight.shape[:2]
    return u.reshape((tile + 2) ** 2, taps * cin, cout).contiguous().to(torch.float16)


def wino_spatial_conv_frames(
    frames: torch.Tensor,
    u: torch.Tensor,
    height: int,
    width: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """One output frame of a causal 3x3x3 conv (padding 1) by spatial Winograd alone:
    ``frames [taps, H*W, Cin]`` bf16 (the taps' input frames, oldest first) with ``u`` from
    :func:`winograd_spatial_taps_weights` -> ``[H*W, Cout]`` bf16, bias added by the output
    transform. The taps are summed inside the GEMM (K = taps * Cin)."""
    taps, _, cin = frames.shape
    tile = spatial_tile_of(u, 1)
    tile_rows = -(-height // tile)
    v = torch.ops.trtllm.vae_wino_spatial_in(
        frames.view(taps, height, width, cin), 1, 0, tile_rows, tile, taps > 1
    )
    m = torch.bmm(v, u)
    out = torch.empty((1, height, width, u.shape[2]), dtype=frames.dtype, device=frames.device)
    torch.ops.trtllm.vae_wino_spatial_out(m, out, 1, 0, tile_rows, tile, bias)
    return out.view(height * width, u.shape[2])


def wino_spatial_conv_taps(
    taps: list[torch.Tensor],
    zero_front: int,
    u: torch.Tensor,
    height: int,
    width: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`wino_spatial_conv_frames` with the taps' input frames in separate buffers:
    ``zero_front`` zero frames, then the frames of each tensor in ``taps`` (``[n, H*W, C]``
    bf16, oldest first; the causal history and the new frame, no concatenation). ``u`` from
    :func:`winograd_spatial_taps_weights`; one batched GEMM and the output transform kernel."""
    tile = spatial_tile_of(u, 1)
    tile_rows = -(-height // tile)
    cout = u.shape[2]
    v = torch.ops.trtllm.vae_wino_spatial_in_taps(
        [t.view(t.shape[0], height, width, t.shape[-1]) for t in taps],
        zero_front,
        0,
        tile_rows,
        tile,
    )
    out = torch.empty((1, height, width, cout), dtype=taps[0].dtype, device=taps[0].device)
    m = torch.bmm(v, u)
    torch.ops.trtllm.vae_wino_spatial_out(m, out, 1, 0, tile_rows, tile, bias)
    return out.view(height * width, cout)


def spatial_tile_of(u: torch.Tensor, components: int = 4) -> int:
    """The tile size a :func:`winograd_spatial_weights` tensor was built for."""
    return int(round((u.shape[0] // components) ** 0.5)) - 2


def cuda_wino_spatial_available() -> bool:
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "vae_wino_spatial_in")


@functools.lru_cache(maxsize=None)
def jit_norm_available() -> bool:
    """cuda-python and a CUDA toolkit (nvcc) for the runtime-compiled norm kernel."""
    try:
        from cuda.bindings import driver  # noqa: F401
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError:
        return False
    import os

    return (
        CUDA_HOME is not None
        and os.path.exists(os.path.join(CUDA_HOME, "bin", "nvcc"))
        and torch.cuda.is_available()
    )


@functools.lru_cache(maxsize=None)
def sm100_wino_available() -> bool:
    """The Blackwell spatial-Winograd kernels: sm100 and ``nvidia-cutlass-dsl`` installed."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100.spatial import sm100_wino_available

    return sm100_wino_available()


def sm100_spatial_eligible(cin: int, cout: int, width: int) -> bool:
    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100.spatial import spatial_eligible

    return spatial_eligible(cin, cout) and width % 4 == 0


def sm100_spatial_filters(g: list[torch.Tensor]) -> torch.Tensor:
    """Transformed filters for :func:`wino_spatial_conv_sm100` from the component filters
    ``g[k] [Cout, Cin, 3, 3]`` (:func:`winograd_weights`)."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100.spatial import spatial_filters

    return spatial_filters(g)


def sm100_fused_eligible(cin: int, cout: int, height: int, width: int) -> bool:
    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100.spatial import fused_eligible

    return fused_eligible(cin, cout, height, width)


def sm100_fused_filters(g: list[torch.Tensor]) -> list[torch.Tensor]:
    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100.spatial import fused_filters

    return fused_filters(g)


def wino_spatial_conv_fused(
    d: torch.Tensor, filters: list[torch.Tensor], height: int, width: int
) -> torch.Tensor:
    """Same contract as :func:`wino_spatial_conv_cuda` for the 256-output-channel convs: the
    fused F(2,3)-horizontal / direct-vertical kernel (CuTe DSL, sm100), filters from
    :func:`sm100_fused_filters`."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100.spatial import (
        wino_spatial_conv_fused as conv,
    )

    return conv(d, filters, height, width)


def wino_spatial_conv_sm100(
    d: torch.Tensor, u: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Same contract as :func:`wino_spatial_conv_cuda` with ``u`` from
    :func:`sm100_spatial_filters`: the Blackwell F(4x4, 3x3) pipeline (CuTe DSL, sm100)."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100.spatial import (
        wino_spatial_conv as conv,
    )

    return conv(d, u, height, width)


def wino_spatial_conv_cuda(
    d: torch.Tensor, u: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """3x3 conv (padding 1) of the four slabs by their four filters as spatial Winograd:
    ``D [4, pairs, H*W, Cin]`` -> ``M [4, pairs, H*W, Cout]`` (bias-free, fp32 accumulate).
    Input transform and output transform are CUDA kernels; the batched GEMMs are one cuBLAS
    call. The tile size (2 or 4) follows the filters ``u``."""
    comps, pairs, positions, cin = d.shape
    tile = spatial_tile_of(u, comps)
    frames = d.view(comps * pairs, height, width, cin)
    tile_rows = -(-height // tile)
    v = torch.ops.trtllm.vae_wino_spatial_in(frames, comps, 0, tile_rows, tile)
    m = torch.bmm(v, u)
    out = torch.empty((comps * pairs, height, width, u.shape[2]), dtype=d.dtype, device=d.device)
    torch.ops.trtllm.vae_wino_spatial_out(m, out, comps, 0, tile_rows, tile)
    return out.view(comps, pairs, positions, u.shape[2])


def unpatchify_clamp_available() -> bool:
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "vae_unpatchify_clamp")


def unpatchify_clamp_into(chunk: torch.Tensor, out: torch.Tensor, frame0: int, patch: int) -> None:
    """Write ``clamp(unpatchify(chunk), -1, 1)`` into ``out[:, :, frame0:frame0 + t]``: ``chunk``
    ``[1, C*patch*patch, t, H, W]`` and ``out`` ``[1, C, T, H*patch, W*patch]``, both channels-last
    bf16. Same element mapping as :func:`wan_vae.unpatchify`."""
    torch.ops.trtllm.vae_unpatchify_clamp(chunk, out, frame0, patch)


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


def cuda_frame_ops_available() -> bool:
    """The prebuilt frame ops (time interleave, AvgDown shortcut add)."""
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "vae_interleave_bias")


def wino_combine_cuda(
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
) -> torch.Tensor:
    """``(combine(m) | x) + bias + residual`` in frame layout ``[T, P, C]``, eager rounding order.

    With ``pad_h``/``pad_w`` the result is ``[T, (H+pad_h)*(W+pad_w), C]`` with zero rows/columns
    appended (the downsampler's ``ZeroPad2d((0, 1, 0, 1))``). ``products_geometry``: the
    products carry halo rows / columns (spatially sharded decode)."""
    return combine(
        x,
        m,
        bias.reshape(-1).contiguous() if bias is not None else None,
        residual,
        frames,
        height,
        width,
        pad_h,
        pad_w,
        products_geometry=products_geometry,
    )


def avg_down_add_cuda(
    x: torch.Tensor,
    r: torch.Tensor,
    out_height: int,
    out_width: int,
    in_width: int,
    factor_t: int,
    factor_s: int,
    group_size: int,
    pad_t: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x (+ bias) + AvgDown3D(r)`` in frame layout: ``x [T', P', Cout]``, ``r [T, P, Cin]``;
    ``bias`` is the downsample conv's bias when that conv ran without it."""
    return torch.ops.trtllm.vae_avg_down_add(
        x,
        r,
        out_height,
        out_width,
        in_width,
        factor_t,
        factor_s,
        group_size,
        pad_t,
        bias.reshape(-1).contiguous() if bias is not None else None,
    )


def interleave_bias_cuda(u: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """``u [T, P, 2C]`` -> ``[2T, P, C]``: even frames from channels ``0:C``, odd from ``C:2C``, plus the bias."""
    return torch.ops.trtllm.vae_interleave_bias(
        u, bias.reshape(-1).contiguous() if bias is not None else None
    )


def interleave_bias_reference(u: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    t, p, c2 = u.shape
    c = c2 // 2
    ub = u + bias.reshape(1, 1, -1) if bias is not None else u
    return torch.stack([ub[:, :, :c], ub[:, :, c:]], dim=1).reshape(2 * t, p, c)


def wino_norm_silu_cuda(
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
    reference_rounding: bool = False,
    order: int = 2,
    half_slabs: bool = False,
    products_geometry: FrameGeometry | None = None,
    slab_geometry: FrameGeometry | None = None,
    upsample: UpsampleSource | None = None,
    slabs: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """All tensors in frame layout ``[T, P, C]`` (``m``/``D`` as ``[T/order, P, C]``, cache
    ``[0..2, P, C]``). Returns ``(sum or None, y_tail [2, P, C], D [order+2, T/order, P, C])``;
    ``D[k]`` is slab k, fp16 when ``half_slabs`` (required for order 4) else bf16. The
    geometries describe products / slabs that carry halo rows or columns (sharded decode);
    ``slabs`` is an optional preallocated ``D``. ``upsample`` replaces ``x``: the frames are
    formed from the packed upsample conv output while reading (:func:`upsample_source`) and
    returned as the sum."""
    return wino_norm_silu(
        x,
        m,
        residual,
        in_bias.reshape(-1) if in_bias is not None else None,
        cache,
        gamma.reshape(-1),
        bias.reshape(-1) if bias is not None else None,
        float(scale),
        frames,
        order=order,
        half_slabs=half_slabs,
        reference_rounding=reference_rounding,
        products_geometry=products_geometry,
        slab_geometry=slab_geometry,
        upsample=upsample,
        slabs=slabs,
    )


def wino_combine_reference(m: list[torch.Tensor]) -> torch.Tensor:
    """``m[k] [groups, P, C]`` (``order + 2`` of them) -> bf16 frames ``[order*groups, P, C]``:
    frame j of a group is ``sum_k AT[j][k] m_k`` in index order with bf16 rounding after every
    add (the products are exact: powers of two)."""
    order = len(m) - 2
    at = TEMPORAL_WINO[order]["AT"]
    frames = []
    for row in at:
        terms = [(c, m[k]) for k, c in enumerate(row) if c != 0]
        acc = (terms[0][0] * terms[0][1].float()).to(torch.bfloat16)
        for c, mk in terms[1:]:
            acc = (acc.float() + c * mk.float()).to(torch.bfloat16)
        frames.append(acc)
    return torch.stack(frames, dim=1).reshape(order * m[0].shape[0], *m[0].shape[1:])


def wino_norm_silu_reference(
    x: torch.Tensor | None,
    m: list[torch.Tensor] | None,
    residual: torch.Tensor | None,
    in_bias: torch.Tensor | None,
    cache: torch.Tensor | None,
    gamma: torch.Tensor,
    bias: torch.Tensor | float,
    scale: float,
    order: int = 2,
    slab_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor | None, torch.Tensor, list[torch.Tensor]]:
    """Eager form of the kernel, same rounding points (bf16 after every add of the combine,
    the slabs formed in fp32 and rounded once to ``slab_dtype``)."""
    if x is None:
        assert m is not None
        x = wino_combine_reference(m)
    s = None
    if in_bias is not None:
        x = s = x + in_bias.reshape(1, 1, -1)
    if residual is not None:
        x = s = x + residual
    if m is not None and s is None:
        s = x
    t, p, c = x.shape
    y = rms_norm_silu_reference(x.permute(0, 2, 1), gamma.reshape(1, -1, 1), bias, scale).permute(
        0, 2, 1
    )
    cache_frames = cache.shape[0] if cache is not None else 0
    pad = torch.zeros(2 - cache_frames, p, c, dtype=x.dtype, device=x.device)
    hist = torch.cat([pad, cache], 0) if cache is not None and cache_frames else pad
    f = torch.cat([hist, y], 0).float()  # frames -2 .. T-1
    bt = torch.tensor(TEMPORAL_WINO[order]["BT"], dtype=torch.float32, device=x.device)
    groups = f.unfold(0, order + 2, order)  # [groups, P, C, order+2]
    d = torch.einsum("ki,gpci->kgpc", bt, groups).to(slab_dtype)
    return s, y[-2:].contiguous(), [d[k] for k in range(order + 2)]


# ---------------------------------------------------------------------------
# Nearest-2x upsample + 3x3 conv as one 2x2 conv on the low-res grid + phase gather
# ---------------------------------------------------------------------------


def cuda_upsample_available() -> bool:
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "vae_upsample_gather")


def can_use_cuda_upsample_gather(cout: int, factor_t: int, repeats: int) -> bool:
    """Mirror of the kernel's instantiation table."""
    return (
        cuda_upsample_available()
        and cout % 8 == 0
        and factor_t in (1, 2)
        and 1 <= repeats <= 32
        and repeats & (repeats - 1) == 0
    )


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


def upsample_gather_reference(
    temp: torch.Tensor,
    batch: int,
    frames: int,
    shortcut_expanded: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager phase gather: ``out[.., 2i+p, 2j+q] = temp[.., phase, i+p, j+q]`` (+ bias, + shortcut)."""
    bt, c4, hp1, wp1 = temp.shape
    cout, h, w = c4 // 4, hp1 - 1, wp1 - 1
    t4 = temp.view(bt, 4, cout, hp1, wp1)
    out = temp.new_empty(bt, cout, 2 * h, 2 * w)
    for p in (0, 1):
        for q in (0, 1):
            out[:, :, p::2, q::2] = t4[:, 2 * p + q, :, p : p + h, q : q + w]
    out = out.view(batch, frames, cout, 2 * h, 2 * w).permute(0, 2, 1, 3, 4)
    out = apply_in_bias(out, bias)
    if shortcut_expanded is not None:
        out = out + shortcut_expanded
    return out.contiguous(memory_format=torch.channels_last_3d)


def upsample_gather_cuda(
    temp: torch.Tensor,
    shortcut: torch.Tensor | None,
    batch: int,
    frames: int,
    factor_t: int = 1,
    repeats: int = 1,
    crop: int = 0,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """CUDA phase gather with the conv bias and the DupUp3D shortcut add folded in."""
    return torch.ops.trtllm.vae_upsample_gather(
        temp,
        shortcut,
        batch,
        frames,
        factor_t,
        repeats,
        crop,
        bias.contiguous() if bias is not None else None,
    )
