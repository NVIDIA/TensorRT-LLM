# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Wan VAE position-walk kernel (:mod:`.kernels.norm`): call-site interface.

One kernel covers the fused RMSNorm + SiLU, the temporal Winograd norm (Winograd input producer
and product consumer, order 2 or 4) and the bias / residual add; its parameters (dtypes, order,
lane layout, mode flags) are all known at the call site. Each combination is compiled with the
CuTe DSL on first use (cached on disk across processes when flashinfer's CuTe DSL kernel cache
is installed) and launched on the current torch
stream, so it is CUDA-graph safe.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import torch

BLOCK_THREADS = 256  # threads per block of the norm kernel
_WARPS = BLOCK_THREADS // 32
_VEC = 8  # bf16 / fp16 channels per 16-byte vector

MID_NONE, MID_NORM, MID_NORM_SILU = 0, 1, 2
OUT_WINOGRAD, OUT_ROWS = 0, 1


@dataclass(frozen=True)
class HaloPush:
    """Where a kernel that fills a frame buffer also stores the first and last logical line of
    the split axis: into the neighbouring ranks' buffers, which hold them as halo lines.

    ``axis`` 0: the frames are split along their height (rows travel), 1: along their width
    (columns travel). ``left`` / ``right`` (the lower / higher rank; None without a neighbour)
    are views of the neighbour's buffer over the destination line: ``[comps, groups, n, C]``
    with ``n`` the line's positions and unit stride over the channels, so the view's data
    pointer and strides are what the kernel needs and a host copy into the view is the same
    transfer."""

    axis: int
    left: torch.Tensor | None
    right: torch.Tensor | None

    def __post_init__(self) -> None:
        for side in (self.left, self.right):
            if side is not None and (side.dim() != 4 or side.stride(3) != 1):
                raise ValueError(
                    "a halo destination is a [comps, groups, n, C] view with unit channel stride"
                )


@dataclass(frozen=True)
class FrameGeometry:
    """Physical layout of a frame that carries halo rows / columns around its ``height`` x
    ``width`` logical positions: logical (i, j) sits at ``(i + row0) * cols + j + col0`` inside
    ``rows * cols`` physical positions (spatially sharded models exchange the halos).

    ``push``: where the producer of such frames also stores their border lines (the
    neighbouring ranks' halo lines); None when the halo lines are filled afterwards."""

    height: int
    width: int
    rows: int
    cols: int
    row0: int = 0
    col0: int = 0
    push: HaloPush | None = field(default=None, compare=False)

    @property
    def frame(self) -> int:
        return self.rows * self.cols

    @property
    def offset(self) -> int:
        return self.row0 * self.cols + self.col0

    @property
    def identity(self) -> bool:
        return (
            self.rows == self.height
            and self.cols == self.width
            and self.offset == 0
            and self.push is None
        )

    def __post_init__(self) -> None:
        if self.row0 + self.height > self.rows or self.col0 + self.width > self.cols:
            raise ValueError("the logical frame must fit inside the physical one")


def _map_args(geo: FrameGeometry | None) -> tuple[int, int, int]:
    return (geo.frame, geo.cols, geo.offset) if geo is not None else (0, 0, 0)


@functools.cache
def _kernel_api():
    """The kernel module and the stream accessor, imported on first use (this module imports
    without the CuTe DSL)."""
    from .kernels import norm as norm_kernel
    from .kernels.common import current_stream

    return norm_kernel, current_stream


@functools.lru_cache(maxsize=None)
def _sm_count(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def row_layout(channels: int, fold_single: bool = True) -> tuple[int, int]:
    """The lane layout of one channel row: ``(lanes, k)`` with ``lanes`` a power of two in
    1..32 and ``k`` 16-byte vectors per lane, ``lanes * k * 8 == channels``. ``channels`` must
    be a multiple of 8. With ``fold_single`` a one-vector-per-lane result becomes two vectors
    on half the lanes (``k == 1`` only occurs for ``lanes == 1``)."""
    # Two vectors per lane give the two-slice pipeline with cross-frame prefetch four rows per
    # warp in flight; at one vector the per-row overhead amortizes over too few elements.
    nvec = channels // _VEC
    lanes = 32
    while nvec % lanes:
        lanes //= 2
    k = nvec // lanes
    if fold_single and k == 1 and lanes > 1:
        lanes //= 2
        k = 2
    return lanes, k


@dataclass(frozen=True, kw_only=True)
class Variant:
    """Compile-time parameters of one norm kernel build.

    ``order``: temporal Winograd order (1 for plain rows). ``lanes`` x ``k``: lanes per position
    and 16-byte vectors per lane (:func:`row_layout`). ``in_wino``: the input is the Winograd
    products rather than x. ``mid``: one of the ``MID_*`` stages; ``out``: ``OUT_WINOGRAD`` or
    ``OUT_ROWS``. ``padded``: the row output appends zero rows and columns. ``in_map`` /
    ``out_map``: the products / outputs sit in frames with halo lines (:class:`FrameGeometry`);
    ``halo``: the output's border lines are pushed to the neighbours as well."""

    fp16_winograd: bool
    order: int
    lanes: int
    k: int
    reference: bool
    has_bias: bool
    has_in_bias: bool
    has_residual: bool
    in_wino: bool
    mid: int
    out: int
    padded: bool
    in_map: bool = False
    out_map: bool = False
    halo: bool = False

    @property
    def positions_per_block(self) -> int:
        return (32 // self.lanes) * _WARPS

    @property
    def ring_slots(self) -> int:
        """Normalized frames each position keeps in shared memory for the Winograd output."""
        return self.order + 1

    def shared_bytes(self, channels: int) -> int:
        """Shared memory of one block: three fp32 per-channel tables, plus the bf16 frame ring
        of every position for the Winograd output."""
        ring = (
            self.ring_slots * self.positions_per_block * channels * 2
            if self.out == OUT_WINOGRAD
            else 0
        )
        return 3 * channels * 4 + ring


def _addr(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


def _halo_args(push: HaloPush | None, channels: int) -> dict:
    """The kernel's halo-push arguments: per side (``_l`` / ``_r``) the destination's address,
    positions per frame, vectors between components, positions between line positions and
    whether the side is present; then the axis. All zero without a push or a neighbour on that
    side."""
    sides = (None, None) if push is None else (push.left, push.right)
    args = {"halo_axis": 0 if push is None else push.axis}
    for name, side in zip(("l", "r"), sides):
        ptr = frame = comp = step = 0
        if side is not None:
            if side.shape[3] != channels:
                raise ValueError("the halo destination does not match the channel count")
            for d in (0, 1, 2):
                if side.stride(d) % channels:
                    raise ValueError("the halo destination strides must be whole positions")
            ptr = side.data_ptr()
            frame = side.stride(1) // channels
            comp = side.stride(0) // _VEC
            step = side.stride(2) // channels
        args[f"halo_{name}"] = ptr
        args[f"halo_frame_{name}"] = frame
        args[f"halo_comp_{name}"] = comp
        args[f"halo_step_{name}"] = step
        args[f"has_{name}"] = int(ptr != 0)
    return args


def _launch(
    variant: Variant,
    device: int,
    *,
    x,
    m_ptr: int,
    d,
    comp_stride: int,
    in_comp_stride: int,
    residual,
    in_bias,
    cache,
    cache_frames: int,
    gamma,
    bias,
    out_sum,
    y_tail,
    y,
    positions: int,
    channels: int,
    frames: int,
    height: int,
    width: int,
    pad_h: int,
    pad_w: int,
    scale: float,
    in_geo: FrameGeometry | None = None,
    out_geo: FrameGeometry | None = None,
) -> None:
    norm_kernel, current_stream = _kernel_api()
    out_positions = (height + pad_h) * (width + pad_w) if variant.padded else positions
    if out_positions <= 0:
        return
    halo = _halo_args(out_geo.push if out_geo is not None else None, channels)
    in_map, out_map = _map_args(in_geo), _map_args(out_geo)
    norm_kernel.launch(
        variant,
        device,
        out_sum is not None,
        -(-out_positions // variant.positions_per_block),
        _sm_count(device),
        dict(
            x=_addr(x),
            m=m_ptr,
            d=_addr(d),
            residual=_addr(residual),
            in_bias=_addr(in_bias),
            cache=_addr(cache) if cache_frames else 0,
            gamma=_addr(gamma),
            bias=_addr(bias),
            sum=_addr(out_sum),
            y_tail=_addr(y_tail),
            y=_addr(y),
            **halo,
            comp_vecs=comp_stride // _VEC,
            in_comp_vecs=in_comp_stride // _VEC,
            in_frame=in_map[0],
            in_w=in_map[1],
            in_off=in_map[2],
            out_frame=out_map[0],
            out_w=out_map[1],
            out_off=out_map[2],
            positions=positions,
            frames=frames,
            cache_frames=cache_frames,
            height=height,
            width=width,
            pad_h=pad_h,
            pad_w=pad_w,
            scale=float(scale),
            eps=1e-12,
        ),
        current_stream(device),
    )


def _stacked_products(m: list[torch.Tensor], winograd_dtype: torch.dtype) -> tuple[int, int]:
    """(base pointer, component stride in elements) of the temporal Winograd conv products,
    which the kernel reads as one stacked buffer [comps, groups, P, C]. In the model they are
    consecutive slices of the one buffer the conv wrote; anything else is stacked here: a copy,
    and the caller's list ``m`` is modified in place to hold views of it."""
    comps = len(m)
    numel = m[0].numel()
    base = m[0].data_ptr()
    if any(mk.dtype != winograd_dtype or not mk.is_contiguous() for mk in m):
        raise ValueError("m must be contiguous tensors in the Winograd dtype")
    if any(m[k].data_ptr() != base + k * numel * 2 for k in range(comps)):
        stacked = torch.stack(m)
        m[:] = list(stacked.unbind(0))
        base = m[0].data_ptr()
    return base, numel


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
    order: int = 2,
    fp16_inputs: bool = False,
    reference_rounding: bool = False,
    products_geometry: FrameGeometry | None = None,
    inputs_geometry: FrameGeometry | None = None,
    inputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Temporal Winograd norm: RMSNorm + SiLU of ``T`` frames, written as the order-``order``
    temporal Winograd inputs of the conv that follows. Every tensor is bf16 frames ``[.., P, C]``
    of ``P`` logical positions, except the products and the Winograd inputs (fp16 when
    ``fp16_inputs``).

    Input, exactly one of: ``x`` ``[T, P, C]``, the frames; or ``m``, the ``order + 2``
    temporal Winograd conv products, each ``[T/order, P, C]``, turned into frames by the output
    transform first. ``in_bias`` ``[C]`` is a conv bias not yet applied to them, ``residual``
    ``[T, P, C]`` the block shortcut, both added before the norm. ``cache`` ``[0..2, P, C]``:
    the previous chunk's last normalized frames (none: zero history). ``gamma`` / ``bias``
    ``[C]``: the norm's scale and shift.

    Returns ``(sum, y_tail, D)``: ``sum`` ``[T, P, C]``, the pre-norm frames, when
    ``return_sum`` is set and there is a residual, an input bias or product input (else None:
    the kernel does not store it); ``y_tail`` ``[2, P, C]``, the last
    two normalized frames (the next chunk's cache); ``D`` ``[order + 2, T/order, input
    positions, C]``, the Winograd inputs.

    ``products_geometry`` / ``inputs_geometry``: the products are read from, and the Winograd inputs
    written to, frames with halo rows or columns (``[T/order, rows * cols, C]``); the kernel
    touches the logical positions only (and, with ``inputs_geometry.push``, stores the border
    lines into the neighbours' buffers as well). ``inputs``: a preallocated destination for
    ``D`` (``[order + 2, T/order, input positions, C]``, contiguous) instead of a fresh one."""
    _check_wino_operands(x, m, residual, in_bias, cache, gamma, bias, frames, order, fp16_inputs)
    comps = order + 2
    winograd_dtype = torch.float16 if fp16_inputs else torch.bfloat16
    products_geometry, inputs_geometry, positions, height, width = _wino_layout(
        x, m, products_geometry, inputs_geometry
    )
    src = x if x is not None else m[0]
    channels = src.shape[-1]
    _check_frames(frames, positions, channels, order, x, residual, cache, m)
    device = src.device.index
    opts = dict(device=src.device, dtype=torch.bfloat16)
    want_sum = return_sum and (residual is not None or in_bias is not None or x is None)
    out_sum = torch.empty((frames, positions, channels), **opts) if want_sum else None
    y_tail = torch.empty((2, positions, channels), **opts)
    input_positions = inputs_geometry.frame if inputs_geometry is not None else positions
    d_shape = (comps, frames // order, input_positions, channels)
    d = _winograd_inputs_buffer(inputs, d_shape, winograd_dtype, src.device)
    cache_frames = cache.shape[0] if cache is not None and cache.numel() else 0
    m_ptr, m_stride = 0, 0
    if m is not None:
        if len(m) != comps:
            raise ValueError("m must hold order + 2 tensors")
        m_ptr, m_stride = _stacked_products(m, winograd_dtype)
    lanes, k = row_layout(channels, fold_single=True)
    variant = Variant(
        fp16_winograd=fp16_inputs,
        order=order,
        lanes=lanes,
        k=k,
        reference=reference_rounding,
        has_bias=bias is not None,
        has_in_bias=in_bias is not None,
        has_residual=residual is not None,
        in_wino=x is None,
        mid=MID_NORM_SILU,
        out=OUT_WINOGRAD,
        padded=False,
        in_map=products_geometry is not None,
        out_map=inputs_geometry is not None,
        halo=inputs_geometry is not None and inputs_geometry.push is not None,
    )
    _launch(
        variant,
        device,
        x=x,
        m_ptr=m_ptr,
        d=d,
        comp_stride=d[0].numel(),
        in_comp_stride=m_stride,
        residual=residual,
        in_bias=in_bias,
        cache=cache,
        cache_frames=cache_frames,
        gamma=gamma,
        bias=bias,
        out_sum=out_sum,
        y_tail=y_tail,
        y=None,
        positions=positions,
        channels=channels,
        frames=frames,
        height=height,
        width=width,
        pad_h=0,
        pad_w=0,
        scale=scale,
        in_geo=products_geometry,
        out_geo=inputs_geometry,
    )
    return out_sum, y_tail, d


def _check_wino_operands(
    x: torch.Tensor | None,
    m: list[torch.Tensor] | None,
    residual: torch.Tensor | None,
    in_bias: torch.Tensor | None,
    cache: torch.Tensor | None,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    frames: int,
    order: int,
    fp16_inputs: bool,
) -> None:
    """The operand rules of :func:`wino_norm_silu`: exactly one of ``x`` / ``m``, bf16 rows,
    order 2 or 4 (4 with fp16 Winograd inputs), whole groups of ``order`` frames."""
    if (x is None) == (m is None):
        raise ValueError("exactly one of x and m")
    for name, t in (
        ("x", x),
        ("residual", residual),
        ("cache", cache),
        ("gamma", gamma),
        ("bias", bias),
        ("in_bias", in_bias),
    ):
        if t is not None and t.dtype is not torch.bfloat16:
            raise TypeError(f"{name} must be bf16 (the kernel reads bf16 rows), got {t.dtype}")
    if order not in (2, 4) or (order == 4 and not fp16_inputs):
        raise ValueError("order must be 2 or 4; order 4 needs fp16 Winograd inputs")
    if frames < order or frames % order:
        raise ValueError("frames must be a positive multiple of order")


def _check_frames(
    frames: int,
    positions: int,
    channels: int,
    order: int,
    x: torch.Tensor | None,
    residual: torch.Tensor | None,
    cache: torch.Tensor | None,
    m: list[torch.Tensor] | None,
) -> None:
    """The shapes of the operands the kernel reads by address: ``x`` and ``residual`` contiguous
    ``[frames, positions, C]``, ``cache`` contiguous ``[0..2, positions, C]``, each product
    ``[frames / order, .., C]``."""
    want = (frames, positions, channels)
    for name, t in (("x", x), ("residual", residual)):
        if t is not None and (tuple(t.shape) != want or not t.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous {list(want)} tensor")
    if (
        cache is not None
        and cache.numel()
        and (
            cache.dim() != 3
            or cache.shape[0] > 2
            or tuple(cache.shape[1:]) != (positions, channels)
            or not cache.is_contiguous()
        )
    ):
        raise ValueError(f"cache must be a contiguous [0..2, {positions}, {channels}] tensor")
    if m is not None and any(
        mk.dim() != 3 or mk.shape[0] != frames // order or mk.shape[2] != channels for mk in m
    ):
        raise ValueError(f"each product must be [{frames // order}, positions, {channels}]")


def _wino_layout(
    x: torch.Tensor | None,
    m: list[torch.Tensor] | None,
    products_geometry: FrameGeometry | None,
    inputs_geometry: FrameGeometry | None,
) -> tuple[FrameGeometry | None, FrameGeometry | None, int, int, int]:
    """The frame layout of :func:`wino_norm_silu`: the geometries (identity ones dropped), the
    logical positions per frame and the logical ``(height, width)`` (zeros without a
    geometry)."""
    if products_geometry is not None and products_geometry.identity:
        products_geometry = None
    if inputs_geometry is not None and inputs_geometry.identity:
        inputs_geometry = None
    if products_geometry is not None and x is not None:
        raise ValueError("products_geometry applies to the products")
    src = x if x is not None else m[0]
    if products_geometry is not None:
        positions = products_geometry.height * products_geometry.width
        if m[0].shape[1] != products_geometry.frame:
            raise ValueError("the products do not match products_geometry")
    else:
        positions = src.shape[1]
    height, width = (
        (inputs_geometry.height, inputs_geometry.width)
        if inputs_geometry is not None
        else (products_geometry.height, products_geometry.width)
        if products_geometry is not None
        else (0, 0)
    )
    if height and width and height * width != positions:
        raise ValueError("the geometries must agree on the logical frame")
    if src.shape[-1] % _VEC:
        raise ValueError("channels must be a multiple of 8")
    return products_geometry, inputs_geometry, positions, height, width


def _winograd_inputs_buffer(
    inputs: torch.Tensor | None, d_shape: tuple, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """The Winograd-input destination: ``inputs`` when given (it must be exactly
    ``d_shape``, ``dtype`` and contiguous), else a new tensor."""
    if inputs is None:
        return torch.empty(d_shape, device=device, dtype=dtype)
    if tuple(inputs.shape) != d_shape or inputs.dtype != dtype or not inputs.is_contiguous():
        raise ValueError(f"inputs must be a contiguous {dtype} tensor of shape {d_shape}")
    return inputs


def rms_norm_silu_rows(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    *,
    reference_rounding: bool = False,
    in_bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    silu: bool = True,
    frames: int = 1,
    out_geometry: FrameGeometry | None = None,
    return_sum: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """RMSNorm (+ SiLU, + residual, + deferred bias) on rows ``[N, C]`` bf16: returns
    ``(sum or None, y)``; ``sum = round(x (+ in_bias) + residual)`` is written when
    ``return_sum`` is set and a residual is given; ``out`` is an optional destination for
    ``y``.

    With ``out_geometry`` the rows are ``frames`` frames of ``height * width`` positions and
    ``out`` holds frames with halo rows / columns (``[frames * rows * cols, C]``); only the
    logical positions are written."""
    _check_rows(x=x, residual=residual, gamma=gamma, bias=bias, in_bias=in_bias, out=out)
    rows, channels = x.shape
    if residual is not None and (
        tuple(residual.shape) != (rows, channels) or not residual.is_contiguous()
    ):
        raise ValueError(f"residual must be a contiguous [{rows}, {channels}] tensor")
    for name, t in (("gamma", gamma), ("bias", bias), ("in_bias", in_bias)):
        if t is not None and (t.numel() != channels or not t.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous [{channels}] tensor")
    out_geometry, frames = _rows_geometry(out_geometry, out, rows, frames)
    height, width = (out_geometry.height, out_geometry.width) if out_geometry else (0, 0)
    out_rows = frames * out_geometry.frame if out_geometry is not None else rows
    y = _destination(out, (out_rows, channels), x.device)
    want_sum = return_sum and residual is not None
    out_sum = (
        torch.empty((rows, channels), device=x.device, dtype=torch.bfloat16) if want_sum else None
    )
    lanes, k = row_layout(channels, fold_single=True)
    variant = Variant(
        fp16_winograd=False,
        order=1,
        lanes=lanes,
        k=k,
        reference=reference_rounding,
        has_bias=bias is not None,
        has_in_bias=in_bias is not None,
        has_residual=residual is not None,
        in_wino=False,
        mid=MID_NORM_SILU if silu else MID_NORM,
        out=OUT_ROWS,
        padded=False,
        in_map=False,
        out_map=out_geometry is not None,
        halo=out_geometry is not None and out_geometry.push is not None,
    )
    _launch(
        variant,
        x.device.index,
        x=x,
        m_ptr=0,
        d=None,
        comp_stride=0,
        in_comp_stride=0,
        residual=residual,
        in_bias=in_bias,
        cache=None,
        cache_frames=0,
        gamma=gamma,
        bias=bias,
        out_sum=out_sum,
        y_tail=None,
        y=y,
        positions=rows // frames,
        channels=channels,
        frames=frames,
        height=height,
        width=width,
        pad_h=0,
        pad_w=0,
        scale=scale,
        out_geo=out_geometry,
    )
    return out_sum, y


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
    """``frames + bias + residual`` as bf16 frames ``[T, (H+pad_h)*(W+pad_w), C]`` with zero
    rows/columns appended, every add rounded like the eager chain. The frames are ``x``, or the
    temporal Winograd products ``m`` turned into frames first (the output transform A^T along
    time: frame ``g*order + j`` is ``sum_k AT[j][k] m_k[g]``). ``products_geometry``: the
    products carry halo rows / columns around their ``height`` x ``width`` logical positions.
    ``out``: a contiguous bf16 destination of the result's shape on the input's device,
    written and returned."""
    products_geometry = _add_input_geometry(x, m, products_geometry, height, width)
    for name, t in (("x", x), ("residual", residual), ("bias", bias)):
        if t is not None and t.dtype is not torch.bfloat16:
            raise TypeError(f"{name} must be bf16 (the kernel reads bf16 rows), got {t.dtype}")
    if m is not None and m[0].dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"m must be bf16 or fp16, got {m[0].dtype}")
    src = x if x is not None else m[0]
    channels = src.shape[2]
    order, m_ptr, stride = _products_layout(m, frames) if m is not None else (1, 0, 0)
    _check_frames(frames, height * width, channels, order, x, residual, None, m)
    fp16_winograd = m is not None and m[0].dtype == torch.float16
    padded = pad_h > 0 or pad_w > 0
    y = _destination(out, (frames, (height + pad_h) * (width + pad_w), channels), src.device)
    lanes, k = row_layout(channels, fold_single=True)
    variant = Variant(
        fp16_winograd=fp16_winograd,
        order=order,
        lanes=lanes,
        k=k,
        reference=True,
        has_bias=False,
        has_in_bias=bias is not None,
        has_residual=residual is not None,
        in_wino=m is not None,
        mid=MID_NONE,
        out=OUT_ROWS,
        padded=padded,
        in_map=products_geometry is not None,
    )
    _launch(
        variant,
        src.device.index,
        x=x,
        m_ptr=m_ptr,
        d=None,
        comp_stride=0,
        in_comp_stride=stride,
        residual=residual,
        in_bias=bias,
        cache=None,
        cache_frames=0,
        gamma=None,
        bias=None,
        out_sum=None,
        y_tail=None,
        y=y,
        positions=height * width,
        channels=channels,
        frames=frames,
        height=height,
        width=width,
        pad_h=pad_h,
        pad_w=pad_w,
        scale=1.0,
        in_geo=products_geometry,
    )
    return y


def _check_rows(**tensors: torch.Tensor | None) -> None:
    """The row kernel's operands are bf16 (absent ones None) with a channel count that is a
    multiple of 8 (``x`` is ``[N, C]``)."""
    for name, t in tensors.items():
        if t is not None and t.dtype is not torch.bfloat16:
            raise TypeError(f"{name} must be bf16 (the kernel reads bf16 rows), got {t.dtype}")
    if tensors["x"].shape[1] % _VEC:
        raise ValueError("channels must be a multiple of 8")


def _rows_geometry(
    geometry: FrameGeometry | None, out: torch.Tensor | None, rows: int, frames: int
) -> tuple[FrameGeometry | None, int]:
    """``(geometry, frames)`` of a row call: no geometry (and one frame) for plain rows or an
    identity layout, else checked against the destination and the row count."""
    if geometry is None or geometry.identity:
        return None, 1
    if out is None or rows != frames * geometry.height * geometry.width:
        raise ValueError("out_geometry needs a destination and frames * height * width rows")
    if out.shape[0] != frames * geometry.frame:
        raise ValueError("out does not match out_geometry")
    return geometry, frames


def _add_input_geometry(
    x: torch.Tensor | None,
    m: list[torch.Tensor] | None,
    geometry: FrameGeometry | None,
    height: int,
    width: int,
) -> FrameGeometry | None:
    """The products' halo layout of an add call (None for compact input), after checking
    that exactly one of ``x`` / ``m`` is given and that the input matches ``height x width``."""
    if (x is None) == (m is None):
        raise ValueError("exactly one of x and m")
    if geometry is not None and geometry.identity:
        geometry = None
    if geometry is not None and (
        x is not None
        or (geometry.height, geometry.width) != (height, width)
        or m[0].shape[1] != geometry.frame
    ):
        raise ValueError("products_geometry must describe the products' frames")
    src = x if x is not None else m[0]
    if geometry is None and src.shape[1] != height * width:
        raise ValueError("height * width must equal the position count")
    return geometry


def _products_layout(m: list[torch.Tensor], frames: int) -> tuple[int, int, int]:
    """``(order, base address, component stride)`` of the temporal Winograd products."""
    order = len(m) - 2
    if order not in (2, 4) or frames % order:
        raise ValueError(
            "m must hold order + 2 tensors (order 2 or 4) and T must be divisible by the order"
        )
    m_ptr, stride = _stacked_products(m, m[0].dtype)
    return order, m_ptr, stride


def _destination(
    out: torch.Tensor | None, shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    """``out`` after checking it is a contiguous bf16 ``shape`` tensor on ``device``, or a new
    one."""
    if out is None:
        return torch.empty(shape, device=device, dtype=torch.bfloat16)
    if (
        tuple(out.shape) != tuple(shape)
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or out.device != device
    ):
        raise ValueError(f"out must be a contiguous bf16 {list(shape)} tensor on {device}")
    return out
