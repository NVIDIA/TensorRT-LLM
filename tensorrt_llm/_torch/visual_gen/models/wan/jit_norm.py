# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime compilation of the Wan VAE position-walk kernel (``cuda/vae_norm_kernel.cuh``).

One kernel template covers the fused RMSNorm + SiLU, the temporal Winograd norm (slab
producer and product consumer, order 2 or 4) and the Winograd combine; its parameters (dtypes,
order, lane layout, mode flags, stages) are all known at the call site. Instead of
instantiating every combination ahead of time, this module compiles the one a call needs on
first use (nvcc from the toolkit, a few seconds), caches the cubin in memory and on disk, and
launches it through the driver API on the current torch stream (CUDA-graph safe).
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import pathlib
import subprocess
import tempfile
from dataclasses import dataclass, field

import torch

_SRC_DIR = pathlib.Path(__file__).parent / "cuda"
_HEADERS = (
    "packed_vector.cuh",
    "vae_norm_common.cuh",
    "vae_wino_tables.cuh",
    "vae_norm_kernel.cuh",
)
_THREADS = 256
_WARPS = _THREADS // 32
_VEC = 8  # 16-byte vectors of 16-bit elements

MID_NONE, MID_NORM, MID_NORM_SILU = 0, 1, 2
OUT_SLABS, OUT_ROWS = 0, 1


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


@dataclass(frozen=True)
class UpsampleSource:
    """Frames defined by the phase gather of a packed 2x2 upsample conv output, formed by the
    kernel while reading: output pixel ``(2i+p, 2j+q)`` of frame ``t`` is
    ``round(temp[t, phase 2p+q, i+p, j+q] + bias)`` plus, when a shortcut is given, the DupUp3D
    element ``shortcut[(t + crop) // factor_t, i, j, (c * factor + S) // repeats]`` with
    ``factor = 4 * factor_t`` and ``S = (((t + crop) % factor_t) * 2 + p) * 2 + q``, rounded to
    bf16 after each add (the arithmetic of the standalone gather kernel). ``temp`` is
    ``[T, (height+1) * (width+1), 4C]`` rows, ``shortcut`` ``[framesR, height * width, Cin]``
    rows, ``bias`` ``[C]``; ``height`` and ``width`` are the low-resolution frame size."""

    temp: torch.Tensor
    shortcut: torch.Tensor | None
    bias: torch.Tensor | None
    factor_t: int
    repeats: int
    crop: int
    height: int
    width: int

    @property
    def frames(self) -> int:
        return self.temp.shape[0]

    @property
    def channels(self) -> int:
        return self.temp.shape[-1] // 4

    @property
    def positions(self) -> int:
        return 4 * self.height * self.width

    @property
    def ratio(self) -> int:
        """Shortcut elements per output channel (factor / repeats)."""
        return 4 * self.factor_t // self.repeats if self.shortcut is not None else 1

    def check(self) -> None:
        if self.temp.dim() != 3 or self.temp.shape[1] != (self.height + 1) * (self.width + 1):
            raise ValueError("temp must be [T, (height+1)*(width+1), 4C] rows")
        if self.temp.shape[-1] % (4 * _VEC):
            raise ValueError("temp channels must be 4 * (a multiple of 8)")
        if self.shortcut is not None:
            factor = 4 * self.factor_t
            if self.factor_t not in (1, 2) or factor % self.repeats or self.ratio not in (1, 2):
                raise ValueError("the shortcut mapping must give 1 or 2 elements per channel")
            if self.shortcut.dim() != 3 or self.shortcut.shape[1] != self.height * self.width:
                raise ValueError("shortcut must be [framesR, height*width, Cin] rows")
            if self.shortcut.shape[0] * self.factor_t != self.frames + self.crop:
                raise ValueError("shortcut frames do not match the output frames")
            if self.shortcut.shape[-1] * self.repeats != self.channels * factor:
                raise ValueError("shortcut channel repeat mismatch")
        if self.bias is not None and self.bias.numel() != self.channels:
            raise ValueError("bias must be [C]")


def _up_variant(up: UpsampleSource | None) -> tuple[int, int]:
    """The kernel's (UP_FT, UP_RATIO) template arguments: 0 without an upsample input, else the
    shortcut's temporal factor (1 when there is no shortcut) and the elements per channel."""
    if up is None:
        return 0, 1
    return (up.factor_t if up.shortcut is not None else 1), up.ratio


def _map_args(geo: FrameGeometry | None) -> tuple[int, int, int]:
    return (geo.frame, geo.cols, geo.offset) if geo is not None else (0, 0, 0)


def _check(result, what: str):
    err, *values = result
    if int(err) != 0:
        raise RuntimeError(f"{what} failed: {err}")
    return values[0] if len(values) == 1 else values


@functools.lru_cache(maxsize=None)
def _cuda_home() -> str:
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is needed to compile the VAE norm kernel at run time")
    return CUDA_HOME


@functools.lru_cache(maxsize=None)
def _arch(device: int) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


@functools.lru_cache(maxsize=None)
def _sm_count(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.lru_cache(maxsize=None)
def _sources() -> dict[str, str]:
    return {name: (_SRC_DIR / name).read_text() for name in _HEADERS}


def _extra_defines() -> list[str]:
    """Experiment knobs for the kernel source (``-D`` flags), from TRTLLM_VAE_JIT_DEFINES."""
    raw = os.environ.get("TRTLLM_VAE_JIT_DEFINES", "")
    return [f"-D{d}" for d in raw.split(",") if d]


def _cache_dir() -> pathlib.Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return pathlib.Path(base) / "tensorrt_llm" / "vae_jit"


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


def blocks_per_sm(k: int, in_wino: bool, order: int, out: int) -> int:
    """Resident blocks per SM the kernel is compiled for (mirrors ``blocksPerSm`` in the
    kernel source, including the ``VAEWINO_IN_BLOCKS`` experiment knob)."""
    if out == OUT_ROWS:
        return 4 if k <= 2 else 3
    if in_wino and order == 4:
        for d in _extra_defines():
            if d.startswith("-DVAEWINO_IN_BLOCKS="):
                knob = int(d.split("=", 1)[1])
                if knob:
                    return knob
    if k >= 4 or order == 4:
        return 2
    return 4 if (not in_wino and k <= 2) else 3


@dataclass(frozen=True, kw_only=True)
class Variant:
    slab_half: bool
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
    up_ft: int = 0
    up_ratio: int = 1
    halo: bool = False

    def name(self) -> str:
        def b(v: bool) -> str:
            return "true" if v else "false"

        td = "__half" if self.slab_half else "__nv_bfloat16"
        return (
            f"vaewino::vaeNormKernel<__nv_bfloat16, {td}, {self.order}, {_VEC}, {self.lanes}, {self.k}, "
            f"{b(self.reference)}, {b(self.has_bias)}, {b(self.has_in_bias)}, {b(self.has_residual)}, "
            f"{b(self.in_wino)}, {self.mid}, {self.out}, {b(self.padded)}, {b(self.in_map)}, "
            f"{b(self.out_map)}, {self.up_ft}, {self.up_ratio}, {b(self.halo)}>"
        )

    def signature(self) -> str:
        t, td = "__nv_bfloat16", "__half" if self.slab_half else "__nv_bfloat16"
        return (
            f"({t} const*, {td} const*, {td}*, long long, long long, {t} const*, {t} const*, {t} const*, "
            f"{t} const*, "
            f"{t} const*, {t}*, {t}*, {t}*, long long, int, int, int, int, int, int, int, long long, int, int, "
            f"long long, int, int, int, int, int, int, float, float, "
            f"void*, void*, long long, long long, long long, long long, int, int, int)"
        )

    @property
    def positions_per_block(self) -> int:
        return (32 // self.lanes) * _WARPS

    def shared_bytes(self, channels: int) -> int:
        ring = (
            (self.order + 1) * self.positions_per_block * channels * 2
            if self.out == OUT_SLABS
            else 0
        )
        return 3 * channels * 4 + ring


class _Kernel:
    def __init__(self, function, module, variant: Variant) -> None:
        self.function = function
        self.module = module
        self.variant = variant
        self.smem_configured = 0


def _compile_cubin(variant: Variant, arch: str) -> tuple[bytes, str]:
    """nvcc compiles one explicit instantiation; the mangled name comes from the cubin."""
    cuda_home = _cuda_home()
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.exists(nvcc):
        raise RuntimeError(
            f"nvcc not found at {nvcc}; the runtime-compiled VAE norm kernel needs the CUDA toolkit"
        )
    name = variant.name()
    src = (
        '#include "vae_norm_kernel.cuh"'
        + "\n"
        + f"template __global__ void {name}{variant.signature()};"
        + "\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        cu = os.path.join(tmp, "vae_norm_kernel_inst.cu")
        cubin = os.path.join(tmp, "k.cubin")
        pathlib.Path(cu).write_text(src)
        cmd = [
            nvcc,
            "-cubin",
            f"-arch={arch}",
            "--std=c++17",
            "-O3",
            *_extra_defines(),
            f"-I{_SRC_DIR}",
            f"-I{os.path.join(cuda_home, 'include', 'cccl')}",
            "-o",
            cubin,
            cu,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"nvcc failed for {name}: {res.stderr[-4000:]}")
        data = pathlib.Path(cubin).read_bytes()
        dump = subprocess.run(
            [os.path.join(cuda_home, "bin", "cuobjdump"), "--dump-resource-usage", cubin],
            capture_output=True,
            text=True,
        ).stdout
    lowered = None
    for line in dump.splitlines():
        line = line.strip()
        if line.startswith("Function ") and "vaeNormKernel" in line:
            lowered = line[len("Function ") :].rstrip(":")
            break
    if lowered is None:
        raise RuntimeError(f"could not find the kernel symbol in the cubin for {name}")
    return data, lowered


@functools.lru_cache(maxsize=None)
def _kernel(device: int, variant: Variant) -> _Kernel:
    from cuda.bindings import driver as cuda

    arch = _arch(device)
    name = variant.name()
    digest = hashlib.sha256()
    for h in _HEADERS:
        digest.update(_sources()[h].encode())
    digest.update(f"{name}|{arch}|{'|'.join(_extra_defines())}|v3".encode())
    cache = _cache_dir() / f"{digest.hexdigest()[:32]}.cubin"
    lowered_path = cache.with_suffix(".name")
    if cache.exists() and lowered_path.exists():
        cubin = cache.read_bytes()
        lowered = lowered_path.read_text()
    else:
        cubin, lowered = _compile_cubin(variant, arch)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Several processes (the ranks of a sharded run) may compile the same variant at once:
        # each writes its own temporary and the rename is atomic, so the cache stays consistent.
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(cubin)
        os.replace(tmp, cache)
        tmp_name = lowered_path.with_suffix(f".{os.getpid()}.tmp")
        tmp_name.write_text(lowered)
        os.replace(tmp_name, lowered_path)
    torch.cuda.set_device(device)
    torch.cuda.current_stream(device)  # make sure the primary context exists
    module = _check(cuda.cuModuleLoadData(cubin), "cuModuleLoadData")
    function = _check(cuda.cuModuleGetFunction(module, lowered.encode()), "cuModuleGetFunction")
    return _Kernel(function, module, variant)


_ARG_TYPES = (
    *(ctypes.c_void_p,) * 3,
    ctypes.c_longlong,
    ctypes.c_longlong,
    *(ctypes.c_void_p,) * 8,
    ctypes.c_longlong,
    *(ctypes.c_int,) * 7,
    ctypes.c_longlong,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_longlong,
    *(ctypes.c_int,) * 6,
    ctypes.c_float,
    ctypes.c_float,
    *(ctypes.c_void_p,) * 2,
    *(ctypes.c_longlong,) * 4,
    *(ctypes.c_int,) * 3,
)


def _addr(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


def _halo_args(push: HaloPush | None, channels: int) -> tuple:
    """The kernel's halo-push arguments: per side the destination's address, positions per
    frame, positions between line positions and elements between components, then the axis."""
    if push is None:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)
    ptrs, frames, comps, steps = [], [], [], []
    for side in (push.left, push.right):
        if side is None:
            ptrs.append(0)
            frames.append(0)
            comps.append(0)
            steps.append(0)
            continue
        if side.shape[3] != channels:
            raise ValueError("the halo destination does not match the channel count")
        for d in (0, 1, 2):
            if side.stride(d) % channels:
                raise ValueError("the halo destination strides must be whole positions")
        ptrs.append(side.data_ptr())
        frames.append(side.stride(1) // channels)
        comps.append(side.stride(0))
        steps.append(side.stride(2) // channels)
    return (*ptrs, *frames, *comps, *steps, push.axis)


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
    up: UpsampleSource | None = None,
) -> None:
    from cuda.bindings import driver as cuda

    kernel = _kernel(device, variant)
    smem = variant.shared_bytes(channels)
    if smem > kernel.smem_configured:
        _check(
            cuda.cuFuncSetAttribute(
                kernel.function,
                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem,
            ),
            "cuFuncSetAttribute",
        )
        kernel.smem_configured = smem
    out_positions = (height + pad_h) * (width + pad_w) if variant.padded else positions
    if out_positions <= 0:
        return
    steps = -(-out_positions // variant.positions_per_block)
    persistent = _sm_count(device) * blocks_per_sm(
        variant.k, variant.in_wino, variant.order, variant.out
    )
    blocks = int(min(steps, persistent))
    args = (
        _addr(x),
        m_ptr,
        _addr(d),
        comp_stride,
        in_comp_stride,
        _addr(residual),
        _addr(in_bias),
        _addr(cache) if cache_frames else 0,
        _addr(gamma),
        _addr(bias),
        _addr(out_sum),
        _addr(y_tail),
        _addr(y),
        positions,
        channels,
        frames,
        cache_frames,
        height,
        width,
        pad_h,
        pad_w,
        *_map_args(in_geo),
        *_map_args(out_geo),
        *(
            (up.height, up.width, up.crop, up.shortcut.shape[-1] if up.shortcut is not None else 0)
            if up is not None
            else (0, 0, 0, 0)
        ),
        float(scale),
        1e-12,
        *_halo_args(out_geo.push if out_geo is not None else None, channels),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    _check(
        cuda.cuLaunchKernel(
            kernel.function, blocks, 1, 1, _THREADS, 1, 1, smem, stream, (args, _ARG_TYPES), 0
        ),
        "cuLaunchKernel",
    )


def _stacked_products(m: list[torch.Tensor], slab_dtype: torch.dtype) -> tuple[int, int]:
    """(base pointer, component stride in elements) of the products, which the kernel reads as
    one stacked buffer [comps, groups, P, C]: consecutive slices of the previous norm's slab
    buffer in the model; anything else is stacked here (a copy)."""
    comps = len(m)
    numel = m[0].numel()
    base = m[0].data_ptr()
    if any(mk.dtype != slab_dtype or not mk.is_contiguous() for mk in m):
        raise ValueError("m must be contiguous tensors in the slab dtype")
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
    order: int = 2,
    half_slabs: bool = False,
    reference_rounding: bool = False,
    products_geometry: FrameGeometry | None = None,
    slab_geometry: FrameGeometry | None = None,
    upsample: UpsampleSource | None = None,
    slabs: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Temporal Winograd norm: frame layout ``[T, P, C]`` (``m``/``D`` as
    ``[T/order, P, C]``, cache ``[0..2, P, C]``). Returns ``(sum or None, y_tail, D)``.

    ``products_geometry`` / ``slab_geometry``: the products are read from, and the slabs
    written to, frames with halo rows or columns (``[T/order, rows * cols, C]``); the kernel
    touches the logical positions only (and, with ``slab_geometry.push``, stores the border
    lines into the neighbours' buffers as well). ``slabs``: a preallocated destination for
    ``D`` (``[order + 2, T/order, slab positions, C]``, contiguous) instead of a fresh one.
    With ``upsample`` (instead of ``x`` or ``m``) the frames are formed from the packed
    upsample conv output while reading and written as the sum."""
    if upsample is not None:
        if x is not None or m is not None or residual is not None or in_bias is not None:
            raise ValueError("upsample replaces x, m, residual and in_bias")
        upsample.check()
        x, residual, in_bias = upsample.temp, upsample.shortcut, upsample.bias
    elif (x is None) == (m is None):
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
    if order not in (2, 4) or (order == 4 and not half_slabs):
        raise ValueError("order must be 2 or 4; order 4 needs fp16 slabs")
    if frames < order or frames % order:
        raise ValueError("frames must be a positive multiple of order")
    comps = order + 2
    slab_dtype = torch.float16 if half_slabs else torch.bfloat16
    if products_geometry is not None and products_geometry.identity:
        products_geometry = None
    if slab_geometry is not None and slab_geometry.identity:
        slab_geometry = None
    if products_geometry is not None and x is not None:
        raise ValueError("products_geometry applies to the products")
    src = x if x is not None else m[0]
    channels = upsample.channels if upsample is not None else src.shape[-1]
    if products_geometry is not None:
        positions = products_geometry.height * products_geometry.width
        if m[0].shape[1] != products_geometry.frame:
            raise ValueError("the products do not match products_geometry")
    elif upsample is not None:
        positions = upsample.positions
    else:
        positions = src.shape[1]
    height, width = (
        (slab_geometry.height, slab_geometry.width)
        if slab_geometry is not None
        else (products_geometry.height, products_geometry.width)
        if products_geometry is not None
        else (0, 0)
    )
    if height and width and height * width != positions:
        raise ValueError("the geometries must agree on the logical frame")
    if channels % _VEC:
        raise ValueError("channels must be a multiple of 8")
    device = src.device.index
    opts = dict(device=src.device, dtype=torch.bfloat16)
    want_sum = residual is not None or in_bias is not None or x is None or upsample is not None
    out_sum = torch.empty((frames, positions, channels), **opts) if want_sum else None
    y_tail = torch.empty((2, positions, channels), **opts)
    slab_positions = slab_geometry.frame if slab_geometry is not None else positions
    d_shape = (comps, frames // order, slab_positions, channels)
    if slabs is None:
        d = torch.empty(d_shape, device=src.device, dtype=slab_dtype)
    else:
        if tuple(slabs.shape) != d_shape or slabs.dtype != slab_dtype or not slabs.is_contiguous():
            raise ValueError(f"slabs must be a contiguous {slab_dtype} tensor of shape {d_shape}")
        d = slabs
    cache_frames = cache.shape[0] if cache is not None and cache.numel() else 0
    m_ptr, m_stride = 0, 0
    if m is not None:
        if len(m) != comps:
            raise ValueError("m must hold order + 2 tensors")
        m_ptr, m_stride = _stacked_products(m, slab_dtype)
    lanes, k = row_layout(channels, fold_single=True)
    up_ft, up_ratio = _up_variant(upsample)
    variant = Variant(
        slab_half=half_slabs,
        order=order,
        lanes=lanes,
        k=k,
        reference=reference_rounding,
        has_bias=bias is not None,
        has_in_bias=in_bias is not None,
        has_residual=residual is not None,
        in_wino=x is None,
        mid=MID_NORM_SILU,
        out=OUT_SLABS,
        padded=False,
        in_map=products_geometry is not None,
        out_map=slab_geometry is not None,
        up_ft=up_ft,
        up_ratio=up_ratio,
        halo=slab_geometry is not None and slab_geometry.push is not None,
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
        out_geo=slab_geometry,
        up=upsample,
    )
    return out_sum, y_tail, d


def rms_norm_silu_rows(
    x: torch.Tensor | None,
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
    upsample: UpsampleSource | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """RMSNorm (+ SiLU, + residual, + deferred bias) on rows ``[N, C]`` bf16: returns
    ``(sum or None, y)``; ``sum = round(x (+ in_bias) + residual)`` is written when a residual
    is given; ``out`` is an optional destination for ``y``.

    With ``out_geometry`` the rows are ``frames`` frames of ``height * width`` positions and
    ``out`` holds frames with halo rows / columns (``[frames * rows * cols, C]``); only the
    logical positions are written.

    With ``upsample`` (instead of ``x``) the rows are its ``frames`` frames of ``4 * height *
    width`` positions, formed from the packed upsample conv output while reading; the sum is
    always returned then."""
    if upsample is not None:
        if x is not None or residual is not None or in_bias is not None:
            raise ValueError("upsample replaces x, residual and in_bias")
        upsample.check()
        x, residual, in_bias = upsample.temp, upsample.shortcut, upsample.bias
        frames, channels = upsample.frames, upsample.channels
        rows = frames * upsample.positions
    else:
        rows, channels = x.shape
    for name, t in (
        ("x", x),
        ("residual", residual),
        ("gamma", gamma),
        ("bias", bias),
        ("in_bias", in_bias),
        ("out", out),
    ):
        if t is not None and t.dtype is not torch.bfloat16:
            raise TypeError(f"{name} must be bf16 (the kernel reads bf16 rows), got {t.dtype}")
    if channels % _VEC:
        raise ValueError("channels must be a multiple of 8")
    if out_geometry is not None and out_geometry.identity:
        out_geometry = None
    height = width = 0
    if out_geometry is not None:
        height, width = out_geometry.height, out_geometry.width
        if out is None or rows != frames * height * width:
            raise ValueError("out_geometry needs a destination and frames * height * width rows")
        if out.shape[0] != frames * out_geometry.frame:
            raise ValueError("out does not match out_geometry")
    elif upsample is None:
        frames = 1
    device = x.device.index
    opts = dict(device=x.device, dtype=torch.bfloat16)
    y = out if out is not None else torch.empty((rows, channels), **opts)
    out_sum = (
        torch.empty((rows, channels), **opts)
        if residual is not None or upsample is not None
        else None
    )
    lanes, k = row_layout(channels, fold_single=True)
    up_ft, up_ratio = _up_variant(upsample)
    variant = Variant(
        slab_half=False,
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
        up_ft=up_ft,
        up_ratio=up_ratio,
        halo=out_geometry is not None and out_geometry.push is not None,
    )
    _launch(
        variant,
        device,
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
        up=upsample,
    )
    return out_sum, y


def combine(
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
    """``(combine(m) | x) + bias + residual`` as
    bf16 frames ``[T, (H+pad_h)*(W+pad_w), C]`` with zero rows/columns appended, every add
    rounded like the eager chain. ``products_geometry``: the products carry halo rows /
    columns around their ``height`` x ``width`` logical positions."""
    if (x is None) == (m is None):
        raise ValueError("exactly one of x and m")
    if products_geometry is not None and products_geometry.identity:
        products_geometry = None
    if products_geometry is not None and (
        x is not None
        or (products_geometry.height, products_geometry.width) != (height, width)
        or m[0].shape[1] != products_geometry.frame
    ):
        raise ValueError("products_geometry must describe the products' frames")
    src = x if x is not None else m[0]
    channels = src.shape[2]
    positions = height * width
    if products_geometry is None and src.shape[1] != positions:
        raise ValueError("height * width must equal the position count")
    device = src.device.index
    order, m_ptr, stride = 1, 0, 0
    slab_half = False
    if m is not None:
        order = len(m) - 2
        if order not in (2, 4) or frames % order:
            raise ValueError(
                "m must hold order + 2 tensors (order 2 or 4) and T must be divisible by the order"
            )
        slab_half = m[0].dtype == torch.float16
        m_ptr, stride = _stacked_products(m, m[0].dtype)
    padded = pad_h > 0 or pad_w > 0
    y = torch.empty(
        (frames, (height + pad_h) * (width + pad_w), channels),
        device=src.device,
        dtype=torch.bfloat16,
    )
    lanes, k = row_layout(channels, fold_single=True)
    variant = Variant(
        slab_half=slab_half,
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
        device,
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
        positions=positions,
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
