# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial Winograd F(m x m, 3x3) transform kernels in CuTe DSL (m = 2 or 4): the input and
output transforms of the portable spatial Winograd path around cuBLAS batched GEMMs.

Input transform: ``V = B^T d B`` per ``(m+2) x (m+2)`` window of frames ``[N, H, W, C]`` (bf16
or fp16; zero outside the frame), fp16 out, laid out ``[components * (m+2)^2, pairs * tiles,
C]`` (frame ``n`` is pair ``n % pairs`` of component ``n // pairs``) or, with stacked frames,
``[(m+2)^2, tiles, N * C]`` (the frames side by side along the GEMM's K axis). Stacked frames
may come from up to three buffers (segments: zero frames, the causal history, the new frame).
Output transform: ``y = A^T M A (+ bias)`` per tile, written into frames ``[N, H, W, C]``.

The rows of B^T and A^T are :func:`wino_tables.bt_row` / :func:`wino_tables.at_row` (F(2x2, 3x3)
on the points (0, +-1, inf), F(4x4, 3x3) on (0, +-3/2, +-5/8, inf)). Both transforms are
separable: rows of B^T (or A^T) along one axis of the window, then along the other.

The input transform with ``C % 64 == 0`` (every production layer) stages a block's window rows
in shared memory: one tile row x 8 tiles x 64 channels per block, 16-byte loads, then one
thread per (tile, channel pair). Other channel counts and the output transform run one thread
per tile and group of ``lanes`` channels (4 for m = 2, 2 for m = 4) and keep the window in
registers.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32, Int32, Int64
from cutlass.base_dsl.typing import Vector

from .. import wino_tables
from ..wino_tables import at_row, bt_row
from .common import (
    CHANNELS_PER_VECTOR,
    THREADS,
    check_aligned,
    check_bias,
    compile_cached,
    current_stream,
    flat_view,
    pack2,
    pack_words,
)

# The transform coefficients are traced into the kernels, so their source keys the cache.
_KEY_FILES = [__file__, wino_tables.__file__]

STAGED_TILES = 8
STAGED_CHANNELS = 64


class _Pair:
    """The fp32 values of two channels in one 64-bit register pair. ``+`` and ``-`` are single
    packed instructions (FADD2) for both channels; ``k * pair`` with a Python constant ``k``
    stays pending (:class:`_Scaled`) until it meets a ``+`` or ``-``, which then emits one
    packed fma (FFMA2). :func:`bt_row` and :func:`at_row` are written with plain operators and run
    on pairs unchanged."""

    def __init__(self, v: Vector) -> None:
        self.v = v

    @staticmethod
    def of(lo, hi) -> "_Pair":
        return _Pair(Vector.from_elements((Float32(lo), Float32(hi)), Float32))

    @property
    def lo(self):
        return Float32(self.v[0])

    @property
    def hi(self):
        return Float32(self.v[1])

    def __rmul__(self, k: float) -> "_Scaled":
        return _Scaled(k, self)

    def __add__(self, o):
        if isinstance(o, _Scaled):
            return o.fma(self)
        return _Pair(cute.arch.add_packed_f32x2(self.v, o.v))

    def __sub__(self, o):
        if isinstance(o, _Scaled):
            return _Scaled(-o.k, o.x).fma(self)
        return _Pair(cute.arch.sub_packed_f32x2(self.v, o.v))


class _Scaled:
    """``k * x`` for a Python constant ``k`` and a :class:`_Pair` ``x``, not yet emitted."""

    def __init__(self, k: float, x: _Pair) -> None:
        self.k = k
        self.x = x

    def _const(self, k: float) -> Vector:
        return Vector.from_elements((Float32(k), Float32(k)), Float32)

    def fma(self, c: _Pair) -> _Pair:
        """``k * x + c`` in one packed fma."""
        return _Pair(cute.arch.fma_packed_f32x2(self._const(self.k), self.x.v, c.v))

    def value(self) -> _Pair:
        return _Pair(cute.arch.mul_packed_f32x2(self._const(self.k), self.x.v))

    def __add__(self, o):
        return self.fma(o.value() if isinstance(o, _Scaled) else o)

    def __sub__(self, o):
        if isinstance(o, _Scaled):
            return self.fma(_Scaled(-o.k, o.x).value())
        raise NotImplementedError("k * x - pair is not used by the transforms")


def _udiv(a, b):
    """``a // b`` for non-negative 32-bit values: an unsigned division (the signed ``//`` adds
    the floor-division sign fix-ups)."""
    return Int32(cutlass.Uint32(a) // cutlass.Uint32(b))


def _separable_transform(fn, m: int, rows: int, window: list) -> list:
    """The separable transform of a square window of values (``window[a][b]``) by the first
    ``rows`` rows of ``fn`` (``bt_row`` with ``rows = m + 2``, ``at_row`` with ``rows = m``): first
    along a (``T[x][b]``), then along b."""
    a_len = len(window)
    t = [
        [fn(m, x, [window[a][b] for a in range(a_len)]) for b in range(a_len)] for x in range(rows)
    ]
    return [[fn(m, y, t[x]) for y in range(rows)] for x in range(rows)]


def _lanes(m: int) -> int:
    """Channels per thread of the register-window kernels: 4 for m = 2 and 2 for m = 4 (the 6x6
    window of fp32 values has to stay in registers)."""
    return 4 if m == 2 else 2


class _SpatialTransformKernel:
    """What the input and output transform kernels share: the tile size ``m``, the window size
    ``m + 2``, the frame dtype, the register-window channel count and its grid width."""

    def __init__(self, m: int, fp16_frames: bool) -> None:
        self.m = m
        self.window_size = m + 2
        self.dtype = Float16 if fp16_frames else BFloat16
        self.lanes = _lanes(m)

    def _tile_cols(self, width):
        return (width + self.m - 1) // self.m

    def _grid_x(self, width, channels):
        """Blocks along x of a register-window launch: tile columns times channel groups."""
        return self._tile_cols(width) * ((channels // self.lanes + THREADS - 1) // THREADS)

    def _frame_pair(self, n, frames, components):
        """``(pairs, component, pair)`` of frame ``n``: frame ``n`` is pair ``n % pairs`` of
        component ``n // pairs``."""
        pairs = _udiv(frames, components)
        comp = _udiv(n, pairs)
        pair = n - comp * pairs
        return pairs, comp, pair

    def _component_row(self, comp, pair, pairs, tile, tiles):
        """Row of ``tile`` at coordinate 0 in the ``[components * (m+2)^2, pairs * tiles, C]``
        layout."""
        size = self.window_size
        return (comp * (size * size) * pairs + pair) * tiles + tile


class InputTransformKernel(_SpatialTransformKernel):
    def __init__(self, m: int, fp16_frames: bool, stacked: bool, staged: bool) -> None:
        super().__init__(m, fp16_frames)
        self.stacked = stacked
        self.staged = staged
        self.window_cols = m * STAGED_TILES + 2
        # The staged m = 4 window (26 KB) fits 4 blocks per SM; ask ptxas for a register count
        # that fits 4 as well (64 registers), so shared memory stays the only occupancy limit.
        self.min_blocks = 4 if staged and m == 4 else 0

    @cute.jit
    def wrapper(
        self,
        v_ptr: cute.Pointer,
        seg0: Int64,
        seg1: Int64,
        seg2: Int64,
        n0: Int32,
        n1: Int32,
        frames: Int32,
        height: Int32,
        width: Int32,
        channels: Int32,
        components: Int32,
        row0: Int32,
        row1: Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.staged):
            tw = self._tile_cols(width)
            grid_x = (tw + STAGED_TILES - 1) // STAGED_TILES * (channels // STAGED_CHANNELS)
        else:
            grid_x = self._grid_x(width, channels)
        smem = self.window_size * self.window_cols * STAGED_CHANNELS * 2 if self.staged else 0
        self.kernel(
            v_ptr, seg0, seg1, seg2, n0, n1, frames, height, width, channels, components, row0, row1
        ).launch(
            grid=[grid_x, row1 - row0, frames],
            block=[THREADS, 1, 1],
            smem=smem,
            stream=stream,
            min_blocks_per_mp=self.min_blocks,
        )

    @cute.kernel
    def kernel(
        self,
        v_ptr: cute.Pointer,
        seg0: Int64,
        seg1: Int64,
        seg2: Int64,
        n0: Int32,
        n1: Int32,
        frames: Int32,
        height: Int32,
        width: Int32,
        channels: Int32,
        components: Int32,
        row0: Int32,
        row1: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bx, by, bz = cute.arch.block_idx()
        tid, bx, n = Int32(tid), Int32(bx), Int32(bz)
        tw = _udiv(width + self.m - 1, self.m)
        tiles = Int64(row1 - row0) * tw
        i = row0 + Int32(by)
        addr, local = self._frame_buffer(n, seg0, seg1, seg2, n0, n1)
        # First pixel of the frame, in pixels of its own buffer.
        frame_pixel = Int64(local) * (height * width)
        present = addr != 0  # address 0: zero frames

        half_ch = channels >> 1
        if cutlass.const_expr(self.stacked):
            coord_words = tiles * frames * half_ch
        else:
            coord_words = tiles * _udiv(frames, components) * half_ch
        v_out = flat_view(cute.recast_ptr(v_ptr, dtype=Int32), 1)

        if cutlass.const_expr(self.staged):
            groups = _udiv(tw + STAGED_TILES - 1, STAGED_TILES)
            chunk = _udiv(bx, groups)
            j0 = (bx - chunk * groups) * STAGED_TILES
            win = self._stage_window(
                addr, present, frame_pixel, i, height, width, tid, j0, chunk, channels
            )
            # Transform: thread -> (tile within the group, channel pair).
            tile_in_group = tid // (STAGED_CHANNELS // 2)
            channel_pair = tid - tile_in_group * (STAGED_CHANNELS // 2)
            j = j0 + tile_in_group
            if j < tw:
                vals = self._staged_window_pair(win, tile_in_group, channel_pair)
                tile = Int64(i - row0) * tw + j
                c0 = chunk * STAGED_CHANNELS + 2 * channel_pair
                base = self._v_word(tile, c0, n, frames, components, half_ch, tiles)
                self._transform_store(v_out, base, coord_words, vals, 1)
        else:
            cvecs = _udiv(channels, self.lanes)
            blk = _udiv(bx, tw)
            j = bx - blk * tw
            cv = blk * THREADS + tid
            if cv < cvecs:
                vals = self._register_window(
                    addr, present, frame_pixel, i, height, width, j, cv, cvecs
                )
                tile = Int64(i - row0) * tw + j
                base = self._v_word(tile, cv * self.lanes, n, frames, components, half_ch, tiles)
                self._transform_store(v_out, base, coord_words, vals, self.lanes // 2)

    @cute.jit
    def _frame_buffer(self, n, seg0, seg1, seg2, n0, n1):
        """The buffer holding frame ``n`` (block-uniform) and the frame's index inside it: the
        first ``n0`` frames are in ``seg0``, the next ``n1`` in ``seg1``, the rest in ``seg2``."""
        addr = seg0
        local = n
        if n >= n0:
            addr = seg1
            local = n - n0
            if n >= n0 + n1:
                addr = seg2
                local = n - n0 - n1
        return addr, local

    @cute.jit
    def _stage_window(self, addr, present, frame_pixel, i, height, width, tid, j0, chunk, channels):
        """Copies the block's ``(m+2) x window_cols`` window (one tile row, ``STAGED_TILES``
        tiles, 64 channels) into shared memory as 16-byte vectors, zero outside the frame; all
        global loads are issued before the first shared-memory store. Returns the window viewed
        as channel pairs."""
        m, size, cols, ch = self.m, self.window_size, self.window_cols, STAGED_CHANNELS
        src = flat_view(
            cute.make_ptr(self.dtype, addr, cute.AddressSpace.gmem, assumed_align=16),
            CHANNELS_PER_VECTOR,
        )
        smem = cute.arch.get_dyn_smem(self.dtype, alignment=16)
        win = flat_view(smem, CHANNELS_PER_VECTOR)
        per_pixel = ch // CHANNELS_PER_VECTOR
        cvecs = _udiv(channels, CHANNELS_PER_VECTOR)
        total = size * cols * per_pixel
        bufs = []
        for step in cutlass.range_constexpr((total + THREADS - 1) // THREADS):
            slot = tid + step * THREADS
            vec_in_pixel = slot % per_pixel
            pix = slot // per_pixel
            a = pix // cols
            b = pix - a * cols
            row = m * i - 1 + a
            col = m * j0 - 1 + b
            inside = (
                present and slot < total and row >= 0 and row < height and col >= 0 and col < width
            )
            src_vec = (frame_pixel + (row * width + col)) * cvecs + chunk * per_pixel + vec_in_pixel
            buf = cute.make_rmem_tensor((CHANNELS_PER_VECTOR,), self.dtype)
            buf.fill(0.0)
            if inside:
                buf.store(src[(None, src_vec)].load())
            bufs.append(buf)
        for step in cutlass.range_constexpr((total + THREADS - 1) // THREADS):
            slot = tid + step * THREADS
            if slot < total:
                win[(None, slot)].store(bufs[step].load())
        cute.arch.sync_threads()
        return flat_view(smem, 2)

    def _staged_window_pair(self, win, tile_in_group, channel_pair) -> list:
        """The ``(m+2) x (m+2)`` window of tile ``tile_in_group`` of the staged group at
        ``channel_pair``, as fp32 pairs."""
        m, size, cols, ch = self.m, self.window_size, self.window_cols, STAGED_CHANNELS
        return [
            [
                win[(None, ((a * cols + m * tile_in_group + b) * ch + 2 * channel_pair) // 2)]
                .load()
                .to(Float32)
                for b in range(size)
            ]
            for a in range(size)
        ]

    @cute.jit
    def _register_window(self, addr, present, frame_pixel, i, height, width, j, cv, cvecs) -> list:
        """The ``(m+2) x (m+2)`` window of tile ``j`` at channel group ``cv`` (``lanes``
        channels) read straight from global memory, zero outside the frame, as fp32 vectors."""
        m, size, lanes = self.m, self.window_size, self.lanes
        src = flat_view(
            cute.make_ptr(self.dtype, addr, cute.AddressSpace.gmem, assumed_align=2 * lanes), lanes
        )
        window = []
        for a in cutlass.range_constexpr(size):
            row = m * i - 1 + a
            for b in cutlass.range_constexpr(size):
                col = m * j - 1 + b
                inside = present and row >= 0 and row < height and col >= 0 and col < width
                src_vec = (frame_pixel + (row * width + col)) * cvecs + cv
                buf = cute.make_rmem_tensor((lanes,), self.dtype)
                buf.fill(0.0)
                if inside:
                    buf.store(src[(None, src_vec)].load())
                window.append(buf)
        return [[window[a * size + b].load().to(Float32) for b in range(size)] for a in range(size)]

    def _transform_store(self, v_out, base, coord_words, vals, npairs: int) -> None:
        """``B^T d B`` of each channel of ``vals`` (``(m+2) x (m+2)`` fp32 vectors of ``2 *
        npairs`` channels), stored as fp16 pairs: coordinate ``(x, y)`` of pair ``e2`` at word
        ``base + e2 + (x * (m+2) + y) * coord_words``."""
        m, size = self.m, self.window_size
        for e2 in range(npairs):
            lo, hi = 2 * e2, 2 * e2 + 1
            if m == 4:
                # The 6x6 transform on packed pairs: half the fp32 instructions (FFMA2, FADD2).
                out = _separable_transform(
                    bt_row,
                    m,
                    size,
                    [
                        [_Pair.of(vals[a][b][lo], vals[a][b][hi]) for b in range(size)]
                        for a in range(size)
                    ],
                )
                halves = [[(o.lo, o.hi) for o in row] for row in out]
            else:
                # F(2x2) is a few adds per value; scalar code keeps fewer registers live and
                # measures faster.
                out_lo = _separable_transform(
                    bt_row, m, size, [[vals[a][b][lo] for b in range(size)] for a in range(size)]
                )
                out_hi = _separable_transform(
                    bt_row, m, size, [[vals[a][b][hi] for b in range(size)] for a in range(size)]
                )
                halves = [[(out_lo[x][y], out_hi[x][y]) for y in range(size)] for x in range(size)]
            for x in range(size):
                for y in range(size):
                    v_out[base + e2 + (x * size + y) * coord_words] = pack2(*halves[x][y], True)

    def _v_word(self, tile, c0, n, frames, components, half_ch, tiles):
        """Word index (fp16 pairs) of channel ``c0`` (even) of coordinate 0 of ``tile``."""
        if cutlass.const_expr(self.stacked):
            row = tile * frames + n
        else:
            pairs, comp, pair = self._frame_pair(n, frames, components)
            row = self._component_row(comp, pair, pairs, tile, tiles)
        return row * half_ch + (c0 >> 1)


class OutputTransformKernel(_SpatialTransformKernel):
    """One thread per tile and group of ``lanes`` channels: the ``(m+2)^2`` coordinates of its
    channels in, ``m x m`` output pixels out (cut at the frame edge), the bias added after the
    transform."""

    def __init__(self, m: int, fp16_frames: bool, has_bias: bool) -> None:
        super().__init__(m, fp16_frames)
        self.fp16 = fp16_frames
        self.has_bias = has_bias

    @cute.jit
    def wrapper(
        self,
        m_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        frames: Int32,
        height: Int32,
        width: Int32,
        channels: Int32,
        components: Int32,
        row0: Int32,
        row1: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            m_ptr, y_ptr, bias_ptr, frames, height, width, channels, components, row0, row1
        ).launch(
            grid=[self._grid_x(width, channels), row1 - row0, frames],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        m_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        frames: Int32,
        height: Int32,
        width: Int32,
        channels: Int32,
        components: Int32,
        row0: Int32,
        row1: Int32,
    ):
        m, lanes = self.m, self.lanes
        tid, _, _ = cute.arch.thread_idx()
        bx, by, bz = cute.arch.block_idx()
        tid, bx, n = Int32(tid), Int32(bx), Int32(bz)
        tw = _udiv(width + m - 1, m)
        cvec = _udiv(channels, lanes)
        blk = _udiv(bx, tw)
        j = bx - blk * tw
        cv = blk * THREADS + tid
        if cv < cvec:
            i = row0 + Int32(by)
            tiles = Int64(row1 - row0) * tw
            pairs, comp, pair = self._frame_pair(n, frames, components)
            gm = flat_view(m_ptr, lanes)
            gb = flat_view(bias_ptr, lanes)
            # Output pixels as vectors of lanes / 2 packed pairs (one 8- or 4-byte store each).
            gy = flat_view(cute.recast_ptr(y_ptr, dtype=Int32), lanes // 2)
            # The bias first: its load does not wait for the coordinate addresses.
            bias = None
            if cutlass.const_expr(self.has_bias):
                bias = gb[(None, cv)].load().to(Float32)
            tile = Int64(i - row0) * tw + j
            row = self._component_row(comp, pair, pairs, tile, tiles)
            first = row * cvec + cv
            coord_vecs = tiles * pairs * cvec
            coords = self._load_coords(gm, first, coord_vecs)
            frame_pixel = Int64(n) * height * width
            outs = self._transform_channels(coords, bias)
            self._store_tile(gy, outs, frame_pixel, i, j, cv, cvec, height, width)

    def _load_coords(self, gm, first, coord_vecs) -> list:
        """The ``(m+2)^2`` coordinates of the thread's channels, starting at vector ``first`` of
        ``gm`` with ``coord_vecs`` vectors between coordinates, as fp32 vectors."""
        size, lanes = self.window_size, self.lanes
        # The coordinates are read-only for the kernel's lifetime: non-coherent loads.
        nc = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(), Float16, num_bits_per_copy=16 * lanes, invariant=True
        )
        coords = []
        for k in range(size * size):
            frag = cute.make_rmem_tensor((lanes,), Float16)
            cute.copy(nc, gm[(None, first + k * coord_vecs)], frag)
            coords.append(frag.load().to(Float32))
        return coords

    def _transform_channels(self, coords: list, bias) -> list:
        """``A^T M A (+ bias)`` per channel: ``outs[channel][r][s]`` for output pixel ``(r,
        s)`` of the tile."""
        m, size = self.m, self.window_size
        outs = []
        for ch in range(self.lanes):
            y = _separable_transform(
                at_row, m, m, [[coords[x * size + b][ch] for b in range(size)] for x in range(size)]
            )
            if self.has_bias:
                y = [[y[r][s] + bias[ch] for s in range(m)] for r in range(m)]
            outs.append(y)
        return outs

    @cute.jit
    def _store_tile(self, gy, outs, frame_pixel, i, j, cv, cvec, height, width):
        """Store the tile's ``m x m`` output pixels that lie inside the frame."""
        m, lanes = self.m, self.lanes
        # Output vector index of pixel (m*i, m*j) and the steps to the next pixel and row; the
        # frame-edge tests once per row and per column.
        col_step = Int64(cvec)
        row_step = Int64(width) * col_step
        row_base = (frame_pixel + Int64(m * i) * width + m * j) * col_step + cv
        row_ok = [m * i + r < height for r in range(m)]
        col_ok = [m * j + s < width for s in range(m)]
        for r in cutlass.range_constexpr(m):
            pix = row_base
            for s in cutlass.range_constexpr(m):
                if row_ok[r] and col_ok[s]:
                    words = cute.make_rmem_tensor((lanes // 2,), Int32)
                    packed = pack_words([outs[ch][r][s] for ch in range(lanes)], self.fp16)
                    for e2 in cutlass.range_constexpr(lanes // 2):
                        words[e2] = packed[e2]
                    gy[(None, pix)].store(words.load())
                pix = pix + col_step
            row_base = row_base + row_step


@functools.lru_cache(maxsize=None)
def _input_kernel(device: int, m: int, fp16: bool, stacked: bool, staged: bool):
    from cutlass.cute.runtime import make_ptr

    kernel = InputTransformKernel(m, fp16, stacked, staged)

    def compile_fn():
        return cute.compile(
            kernel.wrapper,
            make_ptr(Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
            Int64(16),
            Int64(16),
            Int64(16),
            *([Int32(1)] * 9),
            current_stream(device),
            options="--enable-tvm-ffi",
        )

    name = f"spatial_in[{m},{int(fp16)},{int(stacked)},{int(staged)}]"
    return compile_cached("trtllm_wan_vae_spatial", name, compile_fn, _KEY_FILES)


@functools.lru_cache(maxsize=None)
def _output_kernel(device: int, m: int, fp16: bool, has_bias: bool):
    from cutlass.cute.runtime import make_ptr

    kernel = OutputTransformKernel(m, fp16, has_bias)

    def compile_fn():
        align = 2 * kernel.lanes
        return cute.compile(
            kernel.wrapper,
            make_ptr(Float16, 16, cute.AddressSpace.gmem, assumed_align=align),
            make_ptr(kernel.dtype, 16, cute.AddressSpace.gmem, assumed_align=align),
            make_ptr(kernel.dtype, 16, cute.AddressSpace.gmem, assumed_align=align),
            *([Int32(1)] * 7),
            current_stream(device),
            options="--enable-tvm-ffi",
        )

    name = f"spatial_out[{m},{int(fp16)},{int(has_bias)}]"
    return compile_cached("trtllm_wan_vae_spatial", name, compile_fn, _KEY_FILES)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _tile_counts(row0: int, row1: int, width: int, tile: int) -> tuple[int, int]:
    """``(coordinates per tile, tiles)`` of tile rows ``[row0, row1)`` of a frame ``width``
    pixels wide."""
    return (tile + 2) ** 2, (row1 - row0) * _ceil_div(width, tile)


def _check_grid(frames: int, row0: int, row1: int) -> None:
    """The launches put frames on grid z and tile rows on grid y, both capped at 65535."""
    if frames > 65535 or row1 - row0 > 65535:
        raise ValueError("too many frames or tile rows for the grid")


def _check_frames(frames: torch.Tensor, tile: int, row0: int, row1: int) -> None:
    if frames.dim() != 4 or frames.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("frames must be [N, H, W, C] bf16 or fp16")
    if not frames.is_contiguous():
        raise ValueError("frames must be contiguous")
    check_aligned(16, frames=frames)
    if tile not in (2, 4) or frames.shape[3] % CHANNELS_PER_VECTOR:
        raise ValueError("tile must be 2 or 4 and the channel count a multiple of 8")
    if not 0 <= row0 < row1 <= _ceil_div(frames.shape[1], tile):
        raise ValueError("bad tile-row range")


def _launch_input(
    v: torch.Tensor,
    segments: list[tuple[int, int]],
    frames: int,
    height: int,
    width: int,
    channels: int,
    components: int,
    row0: int,
    row1: int,
    tile: int,
    fp16: bool,
    stacked: bool,
) -> torch.Tensor:
    _check_grid(frames, row0, row1)
    addrs = [addr for addr, _ in segments] + [0] * (3 - len(segments))
    # Frame counts of the first two segments; a count of ``frames`` stands for "no later
    # segment", so the kernel never moves on to it.
    counts = [count for _, count in segments[:-1]] + [frames] * (3 - len(segments))
    staged = channels % STAGED_CHANNELS == 0
    device = v.device.index
    _input_kernel(device, tile, fp16, stacked, staged)(
        v.data_ptr(),
        addrs[0],
        addrs[1],
        addrs[2],
        counts[0],
        counts[1],
        frames,
        height,
        width,
        channels,
        components,
        row0,
        row1,
        current_stream(device),
    )
    return v


def wino_spatial_in(
    frames: torch.Tensor,
    components: int,
    row0: int,
    row1: int,
    tile: int = 2,
    stack_frames: bool = False,
) -> torch.Tensor:
    """``V = B^T d B`` for tile rows ``[row0, row1)``: frames ``[N, H, W, C]`` bf16 or fp16
    (contiguous, 16-byte aligned) -> fp16 ``[components * (tile+2)^2, (N / components) * tiles,
    C]``, or with ``stack_frames`` (``components == 1``) ``[(tile+2)^2, tiles, N * C]``."""
    _check_frames(frames, tile, row0, row1)
    n, h, w, c = frames.shape
    if components < 1 or n % components or (stack_frames and components != 1):
        raise ValueError("frames must split into components (one when stacking frames)")
    coords, tiles = _tile_counts(row0, row1, w, tile)
    shape = (
        (coords, tiles, n * c)
        if stack_frames
        else (components * coords, n // components * tiles, c)
    )
    v = torch.empty(shape, dtype=torch.float16, device=frames.device)
    return _launch_input(
        v,
        [(frames.data_ptr(), n)],
        n,
        h,
        w,
        c,
        components,
        row0,
        row1,
        tile,
        frames.dtype == torch.float16,
        stack_frames,
    )


def wino_spatial_in_taps(
    taps: list[torch.Tensor], zero_front: int, row0: int, row1: int, tile: int = 2
) -> torch.Tensor:
    """The stacked-frame input transform over frames in separate buffers: ``zero_front`` zero
    frames, then the frames of each tensor in ``taps`` (``[n_i, H, W, C]``, same shape and dtype,
    contiguous, 16-byte aligned), side by side along K: ``V [(tile+2)^2, tiles, (zero_front + sum
    n_i) * C]`` fp16."""
    if not taps or zero_front < 0:
        raise ValueError("taps must hold at least one tensor and zero_front must be >= 0")
    for t in taps:
        _check_frames(t, tile, row0, row1)
        if t.shape[1:] != taps[0].shape[1:] or t.dtype != taps[0].dtype:
            raise ValueError("every tap must be [n, H, W, C] of the same shape and dtype")
    segments = ([(0, zero_front)] if zero_front else []) + [
        (t.data_ptr(), t.shape[0]) for t in taps
    ]
    if len(segments) > 3:
        raise ValueError("at most three frame segments (zero frames, history, new frame)")
    _, h, w, c = taps[0].shape
    n = sum(s[1] for s in segments)
    coords, tiles = _tile_counts(row0, row1, w, tile)
    v = torch.empty((coords, tiles, n * c), dtype=torch.float16, device=taps[0].device)
    return _launch_input(
        v, segments, n, h, w, c, 1, row0, row1, tile, taps[0].dtype == torch.float16, True
    )


def wino_spatial_out(
    m: torch.Tensor,
    frames: torch.Tensor,
    components: int,
    row0: int,
    row1: int,
    tile: int = 2,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = A^T M A (+ bias)`` for tile rows ``[row0, row1)`` written into ``frames``
    (``[N, H, W, C]`` bf16 or fp16, contiguous, 16-byte aligned); ``m`` is fp16 ``[components *
    (tile+2)^2, (N / components) * tiles, C]`` (contiguous, 16-byte aligned), ``bias`` a
    contiguous ``[C]`` tensor of the frames' dtype. Returns ``frames``."""
    _check_frames(frames, tile, row0, row1)
    n, h, w, c = frames.shape
    coords, tiles = _tile_counts(row0, row1, w, tile)
    if components < 1 or n % components:
        raise ValueError("frames must split into components")
    if (
        m.dtype != torch.float16
        or not m.is_contiguous()
        or tuple(m.shape) != (components * coords, n // components * tiles, c)
    ):
        raise ValueError(
            "m must be contiguous fp16 [components*(tile+2)^2, (N/components)*tiles, C]"
        )
    check_bias(bias, c, frames.dtype, "C")
    _check_grid(n, row0, row1)
    check_aligned(16, m=m)
    check_aligned(2 * _lanes(tile), bias=bias)
    device = frames.device.index
    _output_kernel(device, tile, frames.dtype == torch.float16, bias is not None)(
        m.data_ptr(),
        frames.data_ptr(),
        bias.data_ptr() if bias is not None else 0,
        n,
        h,
        w,
        c,
        components,
        row0,
        row1,
        current_stream(device),
    )
    return frames
