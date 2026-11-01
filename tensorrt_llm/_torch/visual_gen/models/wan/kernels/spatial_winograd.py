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

F(2x2, 3x3) runs on the points (0, +-1, inf), F(4x4, 3x3) on (0, +-P, +-Q, inf); see the
comment above ``P`` and ``Q`` for the matrices. Both transforms are separable: rows of B^T (or
A^T) along one axis of the window, then along the other.

The input transform with ``C % 64 == 0`` (every production layer) stages a block's window rows
in shared memory: one tile row x 8 tiles x 64 channels per block, 16-byte loads, then one
thread per (tile, channel pair). Other channel counts and the output transform run one thread
per (tile, channel vector) and keep the window in registers.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32, Int32, Int64
from cutlass.base_dsl.typing import Vector

from .common import CHANNELS_PER_VECTOR, compile_cached, flat_view, pack2

THREADS = 256
STAGED_TILES = 8
STAGED_CHANNELS = 64

# F(4x4, 3x3) interpolation points, besides 0 and infinity: +-P and +-Q. Chosen for low fp16
# rounding error; both are dyadic (3/2, 5/8), so every product of them used below is exact in
# fp32 and folds into a single constant at trace time.
#
# Where the coefficients come from (Toom-Cook on the points 0, -P, +P, -Q, +Q, inf):
#
#   Each row of B^T holds the coefficients (of x^0 .. x^5, applied to d0 .. d5) of the
#   polynomial that vanishes on all finite points but one:
#     row 0 drops 0:   (x^2 - P^2)(x^2 - Q^2) = P^2 Q^2 - (P^2 + Q^2) x^2 + x^4
#                      -> [P^2 Q^2, 0, -(P^2 + Q^2), 0, 1, 0]
#     rows 1, 2 drop one of +-P:  x (x^2 - Q^2)(x +- P) -> [0, -+P Q^2, -Q^2, +-P, 1, 0]
#     rows 3, 4 drop one of +-Q:  the same with P and Q swapped
#     row 5 (infinity): row 0 times x, i.e. shifted by one.
#   Rows 1 and 2 share an even part (d4 - Q^2 d2) and an odd part (P d3 - P Q^2 d1) and take
#   their sum and difference; rows 3 and 4 likewise.
#
#   A^T row r evaluates the m outputs at the points: [0^r, P^r, (-P)^r, Q^r, (-Q)^r, inf],
#   where 0^0 = 1 and the infinity column is 1 only in the last row r = 3:
#     row 0: t0 + t1 + t2 + t3 + t4
#     row r: P^r (t1 +- t2) + Q^r (t3 +- t4)  (+ for even r, - for odd r), plus t5 for r = 3.
#
# The filter transform G (scaled so that A^T [(G g) * (B^T d)] is the convolution) and these
# same B^T / A^T are tabulated in ``wino_tables.SPATIAL_WINO``; the kernel tests check the two
# agree.
P = 3 / 2
Q = 5 / 8


class _Pair:
    """The fp32 values of two channels in one 64-bit register pair. ``+`` and ``-`` are single
    packed instructions (FADD2) for both channels; ``k * pair`` with a Python constant ``k``
    stays pending (:class:`_Scaled`) until it meets a ``+`` or ``-``, which then emits one
    packed fma (FFMA2). :func:`_bt` and :func:`_at` are written with plain operators and run
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


def _bt(m: int, x: int, d: list):
    """Row ``x`` of B^T applied to the ``m + 2`` values ``d``."""
    if m == 2:
        return (d[0] - d[2], d[1] + d[2], d[2] - d[1], d[1] - d[3])[x]
    if x == 0:
        return (P * Q) ** 2 * d[0] - (P * P + Q * Q) * d[2] + d[4]
    if x == 5:
        return (P * Q) ** 2 * d[1] - (P * P + Q * Q) * d[3] + d[5]
    a, b = (P, Q) if x in (1, 2) else (Q, P)
    even = d[4] - b * b * d[2]
    odd = a * d[3] - a * b * b * d[1]
    return even + odd if x in (1, 3) else even - odd


def _at(m: int, r: int, t: list):
    """Row ``r`` of A^T applied to the ``m + 2`` values ``t``."""
    if m == 2:
        return t[0] + t[1] + t[2] if r == 0 else t[1] - t[2] - t[3]
    if r == 0:
        return t[0] + t[1] + t[2] + t[3] + t[4]
    if r % 2 == 0:
        y = P**r * (t[1] + t[2]) + Q**r * (t[3] + t[4])
    else:
        y = P**r * (t[1] - t[2]) + Q**r * (t[3] - t[4])
    return y + t[5] if r == 3 else y


def _udiv(a, b):
    """``a // b`` for non-negative 32-bit values: an unsigned division (the signed ``//`` adds
    the floor-division sign fix-ups)."""
    return Int32(cutlass.Uint32(a) // cutlass.Uint32(b))


def _transform2(m: int, fn, window: list) -> list:
    """The separable transform of an ``A x A`` window of values (``window[a][b]``) by the rows
    of ``fn`` (``_bt`` or ``_at``): first along a (``T[x][b]``), then along b."""
    a_len = len(window)
    rows = m + 2 if fn is _bt else m
    t = [
        [fn(m, x, [window[a][b] for a in range(a_len)]) for b in range(a_len)] for x in range(rows)
    ]
    return [[fn(m, y, t[x]) for y in range(rows)] for x in range(rows)]


class InputTransformKernel:
    def __init__(self, m: int, fp16_frames: bool, stacked: bool, staged: bool) -> None:
        self.m = m
        self.A = m + 2
        self.dtype = Float16 if fp16_frames else BFloat16
        self.stacked = stacked
        self.staged = staged
        # Unstaged kernel: channels per thread, 4 for m = 2 and 2 for m = 4 (its 6x6 window of
        # fp32 values has to stay in registers).
        self.lanes = 4 if m == 2 else 2
        self.cols = m * STAGED_TILES + 2
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
        tw = (width + self.m - 1) // self.m
        if cutlass.const_expr(self.staged):
            grid_x = (tw + STAGED_TILES - 1) // STAGED_TILES * (channels // STAGED_CHANNELS)
        else:
            grid_x = tw * ((channels // self.lanes + THREADS - 1) // THREADS)
        smem = self.A * self.cols * STAGED_CHANNELS * 2 if self.staged else 0
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
        # The frame's buffer and its index inside it (block-uniform); address 0: zero frames.
        addr = seg0
        local = n
        if n >= n0:
            addr = seg1
            local = n - n0
            if n >= n0 + n1:
                addr = seg2
                local = n - n0 - n1
        # First pixel of the frame, in pixels of its own buffer.
        frame_pixel = Int64(local) * (height * width)
        geom = (addr != 0, frame_pixel, i, height, width)

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
            win = self._stage_window(addr, geom, tid, j0, chunk, channels)
            # Transform: thread -> (tile within the group, channel pair).
            tt = tid // (STAGED_CHANNELS // 2)
            pp = tid - tt * (STAGED_CHANNELS // 2)
            j = j0 + tt
            if j < tw:
                vals = self._staged_window_pair(win, tt, pp)
                tile = Int64(i - row0) * tw + j
                c0 = chunk * STAGED_CHANNELS + 2 * pp
                base = self._v_word(tile, c0, n, frames, components, half_ch, tiles)
                self._transform_store(v_out, base, coord_words, vals, 1)
        else:
            cvecs = _udiv(channels, self.lanes)
            blk = _udiv(bx, tw)
            j = bx - blk * tw
            cv = blk * THREADS + tid
            if cv < cvecs:
                vals = self._register_window(addr, geom, j, cv, cvecs)
                tile = Int64(i - row0) * tw + j
                base = self._v_word(tile, cv * self.lanes, n, frames, components, half_ch, tiles)
                self._transform_store(v_out, base, coord_words, vals, self.lanes // 2)

    @cute.jit
    def _stage_window(self, addr, geom, tid, j0, chunk, channels):
        """Copies the block's ``A x cols`` window (one tile row, ``STAGED_TILES`` tiles, 64
        channels) into shared memory as 16-byte vectors, zero outside the frame; all global
        loads are issued before the first shared-memory store. Returns the window viewed as
        channel pairs."""
        m, A, cols, ch = self.m, self.A, self.cols, STAGED_CHANNELS
        present, frame_pixel, i, height, width = geom
        src = flat_view(
            cute.make_ptr(self.dtype, addr, cute.AddressSpace.gmem, assumed_align=16),
            CHANNELS_PER_VECTOR,
        )
        smem = cute.arch.get_dyn_smem(self.dtype, alignment=16)
        win = flat_view(smem, CHANNELS_PER_VECTOR)
        per_pixel = ch // CHANNELS_PER_VECTOR
        cvecs = _udiv(channels, CHANNELS_PER_VECTOR)
        total = A * cols * per_pixel
        bufs = []
        for it in cutlass.range_constexpr((total + THREADS - 1) // THREADS):
            q = tid + it * THREADS
            cc = q % per_pixel
            pix = q // per_pixel
            a = pix // cols
            b = pix - a * cols
            row = m * i - 1 + a
            col = m * j0 - 1 + b
            inside = (
                present and q < total and row >= 0 and row < height and col >= 0 and col < width
            )
            g = (frame_pixel + (row * width + col)) * cvecs + chunk * per_pixel + cc
            buf = cute.make_rmem_tensor((CHANNELS_PER_VECTOR,), self.dtype)
            buf.fill(0.0)
            if inside:
                buf.store(src[(None, g)].load())
            bufs.append(buf)
        for it in cutlass.range_constexpr((total + THREADS - 1) // THREADS):
            q = tid + it * THREADS
            if q < total:
                win[(None, q)].store(bufs[it].load())
        cute.arch.sync_threads()
        return flat_view(smem, 2)

    def _staged_window_pair(self, win, tt, pp) -> list:
        """The ``A x A`` window of tile ``tt`` of the staged group at channel pair ``pp``, as
        fp32 pairs."""
        m, A, cols, ch = self.m, self.A, self.cols, STAGED_CHANNELS
        return [
            [
                win[(None, ((a * cols + m * tt + b) * ch + 2 * pp) // 2)].load().to(Float32)
                for b in range(A)
            ]
            for a in range(A)
        ]

    @cute.jit
    def _register_window(self, addr, geom, j, cv, cvecs) -> list:
        """The ``A x A`` window of tile ``j`` at channel vector ``cv`` (``lanes`` channels) read
        straight from global memory, zero outside the frame, as fp32 vectors."""
        m, A, lanes = self.m, self.A, self.lanes
        present, frame_pixel, i, height, width = geom
        src = flat_view(
            cute.make_ptr(self.dtype, addr, cute.AddressSpace.gmem, assumed_align=2 * lanes), lanes
        )
        window = []
        for a in cutlass.range_constexpr(A):
            row = m * i - 1 + a
            for b in cutlass.range_constexpr(A):
                col = m * j - 1 + b
                inside = present and row >= 0 and row < height and col >= 0 and col < width
                g = (frame_pixel + (row * width + col)) * cvecs + cv
                buf = cute.make_rmem_tensor((lanes,), self.dtype)
                buf.fill(0.0)
                if inside:
                    buf.store(src[(None, g)].load())
                window.append(buf)
        return [[window[a * A + b].load().to(Float32) for b in range(A)] for a in range(A)]

    def _transform_store(self, v_out, base, coord_words, vals, npairs: int) -> None:
        """``B^T d B`` of each channel of ``vals`` (``A x A`` fp32 vectors of ``2 * npairs``
        channels), stored as fp16 pairs: coordinate ``(x, y)`` of pair ``e2`` at word
        ``base + e2 + (x * A + y) * coord_words``."""
        m, A = self.m, self.A
        for e2 in range(npairs):
            lo, hi = 2 * e2, 2 * e2 + 1
            if m == 4:
                # The 6x6 transform on packed pairs: half the fp32 instructions (FFMA2, FADD2).
                out = _transform2(
                    m,
                    _bt,
                    [
                        [_Pair.of(vals[a][b][lo], vals[a][b][hi]) for b in range(A)]
                        for a in range(A)
                    ],
                )
                halves = [[(o.lo, o.hi) for o in row] for row in out]
            else:
                # F(2x2) is a few adds per value; scalar code keeps fewer registers live and
                # measures faster.
                out_lo = _transform2(m, _bt, [[vals[a][b][lo] for b in range(A)] for a in range(A)])
                out_hi = _transform2(m, _bt, [[vals[a][b][hi] for b in range(A)] for a in range(A)])
                halves = [[(out_lo[x][y], out_hi[x][y]) for y in range(A)] for x in range(A)]
            for x in range(A):
                for y in range(A):
                    v_out[base + e2 + (x * A + y) * coord_words] = pack2(*halves[x][y], True)

    def _v_word(self, tile, c0, n, frames, components, half_ch, tiles):
        """Word index (fp16 pairs) of channel ``c0`` (even) of coordinate 0 of ``tile``."""
        if cutlass.const_expr(self.stacked):
            row = tile * frames + n
        else:
            pairs = _udiv(frames, components)
            comp = _udiv(n, pairs)
            pair = n - comp * pairs
            row = (comp * (self.A * self.A) * pairs + pair) * tiles + tile
        return row * half_ch + (c0 >> 1)


class OutputTransformKernel:
    """One thread per (tile, channel vector): the ``(m+2)^2`` coordinates of its channels in,
    ``m x m`` output pixels out (cut at the frame edge), the bias added after the transform."""

    def __init__(self, m: int, fp16_frames: bool, has_bias: bool) -> None:
        self.m = m
        self.A = m + 2
        self.dtype = Float16 if fp16_frames else BFloat16
        self.fp16 = fp16_frames
        self.has_bias = has_bias
        self.lanes = 4 if m == 2 else 2

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
        tw = (width + self.m - 1) // self.m
        grid_x = tw * ((channels // self.lanes + THREADS - 1) // THREADS)
        self.kernel(
            m_ptr, y_ptr, bias_ptr, frames, height, width, channels, components, row0, row1
        ).launch(grid=[grid_x, row1 - row0, frames], block=[THREADS, 1, 1], stream=stream)

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
        m, A, lanes = self.m, self.A, self.lanes
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
            pairs = _udiv(frames, components)
            comp = _udiv(n, pairs)
            pair = n - comp * pairs
            gm = flat_view(m_ptr, lanes)
            gb = flat_view(bias_ptr, lanes)
            # Output pixels as vectors of lanes / 2 packed pairs (one 8- or 4-byte store each).
            half = lanes // 2
            gy = flat_view(cute.recast_ptr(y_ptr, dtype=Int32), half)
            # The bias first: its load does not wait for the coordinate addresses.
            bias = None
            if cutlass.const_expr(self.has_bias):
                bias = gb[(None, cv)].load().to(Float32)
            tile = Int64(i - row0) * tw + j
            row = (comp * (A * A) * pairs + pair) * tiles + tile
            first = row * cvec + cv
            coord_vecs = tiles * pairs * cvec
            # The coordinates are read-only for the kernel's lifetime: non-coherent loads.
            nc = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(), Float16, num_bits_per_copy=16 * lanes, invariant=True
            )
            coords = []
            for k in cutlass.range_constexpr(A * A):
                frag = cute.make_rmem_tensor((lanes,), Float16)
                cute.copy(nc, gm[(None, first + k * coord_vecs)], frag)
                coords.append(frag.load().to(Float32))
            frame0 = Int64(n) * height * width
            outs = []
            for ch in cutlass.range_constexpr(lanes):
                y = _transform2(
                    m, _at, [[coords[x * A + b][ch] for b in range(A)] for x in range(A)]
                )
                if cutlass.const_expr(self.has_bias):
                    y = [[y[r][s] + bias[ch] for s in range(m)] for r in range(m)]
                outs.append(y)
            # Output vector index of pixel (m*i, m*j) and the steps to the next pixel and row; the
            # frame-edge tests once per row and per column.
            col_step = Int64(cvec)
            row_step = Int64(width) * col_step
            row_base = (frame0 + Int64(m * i) * width + m * j) * col_step + cv
            row_ok = [m * i + r < height for r in range(m)]
            col_ok = [m * j + s < width for s in range(m)]
            for r in cutlass.range_constexpr(m):
                pix = row_base
                for s in cutlass.range_constexpr(m):
                    if row_ok[r] and col_ok[s]:
                        words = cute.make_rmem_tensor((half,), Int32)
                        for e2 in cutlass.range_constexpr(half):
                            words[e2] = pack2(outs[2 * e2][r][s], outs[2 * e2 + 1][r][s], self.fp16)
                        gy[(None, pix)].store(words.load())
                    pix = pix + col_step
                row_base = row_base + row_step


@functools.lru_cache(maxsize=None)
def _input_kernel(device: int, m: int, fp16: bool, stacked: bool, staged: bool):
    from cutlass.cute.runtime import make_ptr

    kernel = InputTransformKernel(m, fp16, stacked, staged)

    def compile_fn():
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        return cute.compile(
            kernel.wrapper,
            make_ptr(Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
            Int64(16),
            Int64(16),
            Int64(16),
            *([Int32(1)] * 9),
            stream,
            options="--enable-tvm-ffi",
        )

    name = f"spatial_in[{m},{int(fp16)},{int(stacked)},{int(staged)}]"
    return compile_cached("trtllm_wan_vae_spatial", name, compile_fn, [__file__])


@functools.lru_cache(maxsize=None)
def _output_kernel(device: int, m: int, fp16: bool, has_bias: bool):
    from cutlass.cute.runtime import make_ptr

    kernel = OutputTransformKernel(m, fp16, has_bias)
    dtype = Float16 if fp16 else BFloat16

    def compile_fn():
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        lanes = kernel.lanes
        return cute.compile(
            kernel.wrapper,
            make_ptr(Float16, 16, cute.AddressSpace.gmem, assumed_align=2 * lanes),
            make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=2 * lanes),
            make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=2 * lanes),
            *([Int32(1)] * 7),
            stream,
            options="--enable-tvm-ffi",
        )

    name = f"spatial_out[{m},{int(fp16)},{int(has_bias)}]"
    return compile_cached("trtllm_wan_vae_spatial", name, compile_fn, [__file__])


def _stream(device: int):
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _check_frames(frames: torch.Tensor, tile: int, row0: int, row1: int) -> None:
    if frames.dim() != 4 or frames.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("frames must be [N, H, W, C] bf16 or fp16")
    if not frames.is_contiguous():
        raise ValueError("frames must be contiguous")
    if tile not in (2, 4) or frames.shape[3] % 8:
        raise ValueError("tile must be 2 or 4 and the channel count a multiple of 8")
    if not 0 <= row0 < row1 <= -(-frames.shape[1] // tile):
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
    if frames > 65535 or row1 - row0 > 65535:
        raise ValueError("too many frames or tile rows for the grid")
    segs = list(segments) + [(0, 0)] * (3 - len(segments))
    staged = channels % STAGED_CHANNELS == 0
    device = v.device.index
    _input_kernel(device, tile, fp16, stacked, staged)(
        v.data_ptr(),
        segs[0][0],
        segs[1][0],
        segs[2][0],
        segs[0][1] if len(segments) > 1 else frames,
        segs[1][1] if len(segments) > 2 else frames,
        frames,
        height,
        width,
        channels,
        components,
        row0,
        row1,
        _stream(device),
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
    """``V = B^T d B`` for tile rows ``[row0, row1)``: frames ``[N, H, W, C]`` bf16 or fp16 ->
    fp16 ``[components * (tile+2)^2, (N / components) * tiles, C]``, or with ``stack_frames``
    (``components == 1``) ``[(tile+2)^2, tiles, N * C]``."""
    _check_frames(frames, tile, row0, row1)
    n, h, w, c = frames.shape
    if components < 1 or n % components or (stack_frames and components != 1):
        raise ValueError("frames must split into components (one when stacking frames)")
    coords = (tile + 2) ** 2
    tiles = (row1 - row0) * -(-w // tile)
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
    frames, then the frames of each tensor in ``taps`` (``[n_i, H, W, C]``, same shape and dtype),
    side by side along K: ``V [(tile+2)^2, tiles, (zero_front + sum n_i) * C]`` fp16."""
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
    coords = (tile + 2) ** 2
    tiles = (row1 - row0) * -(-w // tile)
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
    (``[N, H, W, C]`` bf16 or fp16); ``m`` is fp16 ``[components * (tile+2)^2, (N /
    components) * tiles, C]``. Returns ``frames``."""
    _check_frames(frames, tile, row0, row1)
    n, h, w, c = frames.shape
    coords = (tile + 2) ** 2
    tiles = (row1 - row0) * -(-w // tile)
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
    if bias is not None and (
        bias.dtype != frames.dtype or bias.shape != (c,) or not bias.is_contiguous()
    ):
        raise ValueError("bias must be [C] of the frames' dtype")
    if n > 65535 or row1 - row0 > 65535:
        raise ValueError("too many frames or tile rows for the grid")
    if frames.data_ptr() % 16 or m.data_ptr() % 16:
        raise ValueError("frames and m must be 16-byte aligned")
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
        _stream(device),
    )
    return frames
