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

F(2x2, 3x3): Lavin & Gray, points (0, +-1, inf). F(4x4, 3x3): points (0, +-3/2, +-5/8, inf);
B^T and A^T entries are dyadic. The arithmetic (separable passes, rows of B^T and A^T as written
in :func:`_bt` / :func:`_at`, the products fused into an fma) is the one the outputs are defined
by; every other add and multiply rounds on its own.

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
from cutlass._mlir_helpers.math import RoundingMode

from .common import VEC, compile_cached, pack2

THREADS = 256
STAGED_TILES = 8
STAGED_CHANNELS = 64
_RN = RoundingMode.NEAREST_EVEN
_FAR = 1 << 40  # extent of the flat views (indexing is unchecked)


def _add(a, b):
    return cute.math.add(a, b, rounding=_RN)


def _sub(a, b):
    return cute.math.add(a, -b, rounding=_RN)


def _mul(k: float, a):
    return cute.math.mul(Float32(k), a, rounding=_RN)


def _fma(k: float, a, c):
    return cute.math.fma(Float32(k), a, c)


def _bt(m: int, x: int, d: list):
    """Row ``x`` of B^T applied to the ``m + 2`` values ``d``."""
    if m == 2:
        return (_sub(d[0], d[2]), _add(d[1], d[2]), _sub(d[2], d[1]), _sub(d[1], d[3]))[x]
    if x == 0:
        return _add(_fma(0.87890625, d[0], _mul(-2.640625, d[2])), d[4])
    if x == 5:
        return _add(_fma(0.87890625, d[1], _mul(-2.640625, d[3])), d[5])
    if x in (1, 2):
        e = _fma(-0.390625, d[2], d[4])
        o = _fma(1.5, d[3], _mul(-0.5859375, d[1]))
    else:
        e = _fma(-2.25, d[2], d[4])
        o = _fma(0.625, d[3], _mul(-1.40625, d[1]))
    return _add(e, o) if x in (1, 3) else _sub(e, o)


def _at(m: int, r: int, t: list):
    """Row ``r`` of A^T applied to the ``m + 2`` values ``t``."""
    if m == 2:
        return _add(_add(t[0], t[1]), t[2]) if r == 0 else _sub(_sub(t[1], t[2]), t[3])
    if r == 0:
        return _add(_add(_add(_add(t[0], t[1]), t[2]), t[3]), t[4])
    if r == 1:
        return _fma(1.5, _sub(t[1], t[2]), _mul(0.625, _sub(t[3], t[4])))
    if r == 2:
        return _fma(2.25, _add(t[1], t[2]), _mul(0.390625, _add(t[3], t[4])))
    return _add(_fma(3.375, _sub(t[1], t[2]), _mul(0.244140625, _sub(t[3], t[4]))), t[5])


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
            grid=[grid_x, row1 - row0, frames], block=[THREADS, 1, 1], smem=smem, stream=stream
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
        m, A, cols = self.m, self.A, self.cols
        tid, _, _ = cute.arch.thread_idx()
        bx, by, bz = cute.arch.block_idx()
        tid, bx, n = Int32(tid), Int32(bx), Int32(bz)
        tw = _udiv(width + m - 1, m)
        tiles = Int64(row1 - row0) * tw
        pairs = _udiv(frames, components)
        comp = _udiv(n, pairs)
        pair = n - comp * pairs
        # The frame's buffer and its index inside it (block-uniform); address 0: zero frames.
        addr = seg0
        local = n
        if n >= n0:
            addr = seg1
            local = n - n0
            if n >= n0 + n1:
                addr = seg2
                local = n - n0 - n1
        present = addr != 0
        frame_elems = Int64(height) * width * channels
        i = row0 + Int32(by)
        if cutlass.const_expr(self.stacked):
            coord_words = tiles * frames * channels // 2
        else:
            coord_words = tiles * pairs * channels // 2
        v_out = cute.make_tensor(cute.recast_ptr(v_ptr, dtype=Int32), cute.make_layout((_FAR,)))

        if cutlass.const_expr(self.staged):
            ch = STAGED_CHANNELS
            groups = _udiv(tw + STAGED_TILES - 1, STAGED_TILES)
            chunk = _udiv(bx, groups)
            j0 = (bx - chunk * groups) * STAGED_TILES
            src = cute.make_tensor(
                cute.make_ptr(self.dtype, addr, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((VEC, _FAR), stride=(1, VEC)),
            )
            smem = cute.arch.get_dyn_smem(self.dtype, alignment=16)
            win = cute.make_tensor(
                smem, cute.make_layout((VEC, A * cols * ch // VEC), stride=(1, VEC))
            )
            win2 = cute.make_tensor(smem, cute.make_layout((2, A * cols * ch // 2), stride=(1, 2)))
            # Copy phase: 16-byte chunks over (window row, window column, 8-channel chunk), all
            # loads issued before the first shared-memory store; zero outside the frame.
            per_pixel = ch // VEC
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
                g = (
                    Int64(local) * frame_elems
                    + (Int64(row) * width + col) * channels
                    + chunk * ch
                    + cc * VEC
                ) // VEC
                buf = cute.make_rmem_tensor((VEC,), self.dtype)
                buf.store(cute.full((VEC,), 0.0, Float32).to(self.dtype))
                if inside:
                    buf.store(src[(None, g)].load())
                bufs.append(buf)
            for it in cutlass.range_constexpr((total + THREADS - 1) // THREADS):
                q = tid + it * THREADS
                if q < total:
                    win[(None, q)].store(bufs[it].load())
            cute.arch.sync_threads()

            # Transform phase: thread -> (tile within the group, channel pair).
            tt = tid // (ch // 2)
            pp = tid - tt * (ch // 2)
            j = j0 + tt
            if j < tw:
                window = []
                for a in cutlass.range_constexpr(A):
                    row_vals = []
                    for b in cutlass.range_constexpr(A):
                        pv = win2[(None, ((a * cols + m * tt + b) * ch + 2 * pp) // 2)].load()
                        row_vals.append(pv.to(Float32))
                    window.append(row_vals)
                tile = Int64(i - row0) * tw + j
                base = self._v_word(
                    tile, chunk * ch + 2 * pp, n, comp, pair, pairs, frames, channels, tiles
                )
                out = [
                    _transform2(m, _bt, [[window[a][b][e] for b in range(A)] for a in range(A)])
                    for e in range(2)
                ]
                for x in cutlass.range_constexpr(A):
                    for y in cutlass.range_constexpr(A):
                        v_out[base + (x * A + y) * coord_words] = pack2(
                            out[0][x][y], out[1][x][y], True
                        )
        else:
            lanes = self.lanes
            cvec = _udiv(channels, lanes)
            blk = _udiv(bx, tw)
            j = bx - blk * tw
            cv = blk * THREADS + tid
            if cv < cvec:
                c0 = cv * lanes
                src = cute.make_tensor(
                    cute.make_ptr(
                        self.dtype, addr, cute.AddressSpace.gmem, assumed_align=2 * lanes
                    ),
                    cute.make_layout((lanes, _FAR), stride=(1, lanes)),
                )
                window = []
                for a in cutlass.range_constexpr(A):
                    row = m * i - 1 + a
                    row_vals = []
                    for b in cutlass.range_constexpr(A):
                        col = m * j - 1 + b
                        inside = present and row >= 0 and row < height and col >= 0 and col < width
                        g = (
                            Int64(local) * frame_elems + (Int64(row) * width + col) * channels + c0
                        ) // lanes
                        buf = cute.make_rmem_tensor((lanes,), self.dtype)
                        buf.store(cute.full((lanes,), 0.0, Float32).to(self.dtype))
                        if inside:
                            buf.store(src[(None, g)].load())
                        row_vals.append(buf)
                    window.append(row_vals)
                tile = Int64(i - row0) * tw + j
                base = self._v_word(tile, c0, n, comp, pair, pairs, frames, channels, tiles)
                vals = [[window[a][b].load().to(Float32) for b in range(A)] for a in range(A)]
                for e2 in cutlass.range_constexpr(lanes // 2):
                    out = [
                        _transform2(
                            m, _bt, [[vals[a][b][2 * e2 + e] for b in range(A)] for a in range(A)]
                        )
                        for e in range(2)
                    ]
                    for x in cutlass.range_constexpr(A):
                        for y in cutlass.range_constexpr(A):
                            v_out[base + e2 + (x * A + y) * coord_words] = pack2(
                                out[0][x][y], out[1][x][y], True
                            )

    def _v_word(self, tile, c0, n, comp, pair, pairs, frames, channels, tiles):
        """Word index (fp16 pairs) of channel ``c0`` (even) of coordinate 0 of ``tile``."""
        if self.stacked:
            off = tile * (Int64(frames) * channels) + Int64(n) * channels + c0
        else:
            row = Int64(comp) * (self.A * self.A) * (tiles * pairs) + Int64(pair) * tiles + tile
            off = row * channels + c0
        return off // 2


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
        self.lane_shift = self.lanes.bit_length() - 1

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
            c0 = cv * lanes
            i = row0 + Int32(by)
            tiles = Int64(row1 - row0) * tw
            pairs = _udiv(frames, components)
            comp = _udiv(n, pairs)
            pair = n - comp * pairs
            gm = cute.make_tensor(m_ptr, cute.make_layout((lanes, _FAR), stride=(1, lanes)))
            gb = cute.make_tensor(bias_ptr, cute.make_layout((lanes, _FAR), stride=(1, lanes)))
            # Output pixels as vectors of lanes / 2 packed pairs (one 8- or 4-byte store each).
            half = lanes // 2
            gy = cute.make_tensor(
                cute.recast_ptr(y_ptr, dtype=Int32),
                cute.make_layout((half, _FAR), stride=(1, half)),
            )
            tile = Int64(i - row0) * tw + j
            row = Int64(comp) * (A * A) * (tiles * pairs) + Int64(pair) * tiles + tile
            first = (row * channels + c0) >> self.lane_shift
            coord_vecs = tiles * pairs * cvec
            # The coordinates are read-only for the kernel's lifetime: non-coherent loads.
            nc = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(), Float16, num_bits_per_copy=16 * lanes, invariant=True
            )
            coords = []
            idx = first
            for k in cutlass.range_constexpr(A * A):
                frag = cute.make_rmem_tensor((lanes,), Float16)
                cute.copy(nc, gm[(None, idx)], frag)
                coords.append(frag.load().to(Float32))
                idx = idx + coord_vecs
            bias = None
            if cutlass.const_expr(self.has_bias):
                bias = gb[(None, cv)].load().to(Float32)
            frame0 = Int64(n) * height * width
            outs = []
            for ch in cutlass.range_constexpr(lanes):
                y = _transform2(
                    m, _at, [[coords[x * A + b][ch] for b in range(A)] for x in range(A)]
                )
                if cutlass.const_expr(self.has_bias):
                    y = [[_add(y[r][s], bias[ch]) for s in range(m)] for r in range(m)]
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
