# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input and output transforms of the spatial Winograd F(4x4, 3x3) channels-last 3x3 conv on
Blackwell, around the batched GEMM of :mod:`.gemm`:

  U[xi, co, ci] = (G g G^T)[xi]       fp16, xi = i*6 + j       (filter transform, on the host)
  V[xi, t,  ci] = (B^T d B)[xi]       fp16, t = 4x4 output tile (input transform)
  M[xi, t,  co] = sum_ci V[xi,t,ci] U[xi,co,ci]                (batched tcgen05 GEMMs)
  y[tile t]     = A^T M[:, :, t] A                             (output transform)

B^T and A^T are the F(4,3) matrices of ``wino_tables.SPATIAL_WINO`` (points 0, +-3/2, +-5/8,
inf), well-conditioned enough to keep the fp16 error inside tolerance. Frame and tile
coordinates are 64-bit: a 32-bit frame index times the frame stride overflows above 2^31
elements (1080x1920 x 512 channels from the third frame on).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float16, Float32, Int32, Int64

from ...wino_tables import at_row, bt_row

# Output tile edge and input window edge.
TILE = 4
WINDOW = TILE + 2
THREADS = 128  # per CTA
# Channels per thread: the input transform holds a WINDOW x WINDOW window of IN_VECTOR-channel
# vectors in registers, the output transform the same window of OUT_VECTOR-channel vectors.
IN_VECTOR = 2
OUT_VECTOR = 4


def _bt4(d: list) -> list:
    """All six rows of the F(4, 3) B^T applied to ``d``."""
    return [bt_row(TILE, x, d) for x in range(WINDOW)]


def _at4(m: list) -> list:
    """All four rows of the F(4, 3) A^T applied to ``m``."""
    return [at_row(TILE, r, m) for r in range(TILE)]


def _vector_view(tensor: cute.Tensor, tile: tuple, coord: tuple) -> cute.Tensor:
    """Contiguous rank-1 view (last tile mode) of ``tensor`` at vector coordinate ``coord``."""
    v = cute.local_tile(tensor, tile, coord)
    return v[tuple(0 for _ in range(len(tile) - 1)) + (None,)]


def _tile_grid(height: int, width: int) -> tuple[int, int]:
    """Tile rows and columns of a frame (the last tile row may be partial; ``width`` is a
    multiple of :data:`TILE`)."""
    return (height + TILE - 1) // TILE, width // TILE


def num_tiles(frames: int, height: int, width: int) -> int:
    """Output tiles of ``frames`` frames of ``height x width``."""
    tile_rows, tile_cols = _tile_grid(height, width)
    return frames * tile_rows * tile_cols


def _tile_coords(lin, groups: int, height: int, width: int):
    """Work item ``lin`` (channel group fastest, then tile column, tile row, frame) as ``(frame,
    tile row, tile column, tile index, channel group)``; frame and tile index in 64 bits."""
    tile_rows, tile_cols = _tile_grid(height, width)
    g = lin % groups
    t = lin // groups
    tw = t % tile_cols
    r = t // tile_cols
    th = r % tile_rows
    n = Int64(r // tile_rows)
    t64 = Int64(t)
    return n, th, tw, t64, g


# ---------------------------------------------------------------- input transform
@cute.jit
def _in_load_xform_w(
    mX: cute.Tensor,
    n,
    th,
    tw,
    g,
    H: cutlass.Constexpr,
    W: cutlass.Constexpr,
):
    """Load the input window of tile ``(th, tw)`` (zero outside the frame) and apply B^T along W
    to each of its rows: ``rows[a][j]``."""
    rows = []
    for a in cutlass.range_constexpr(WINDOW):
        h = TILE * th - 1 + a
        h_ok = (h >= 0) and (h < H)
        h_in = h if h_ok else Int32(0)
        d = []
        for b in cutlass.range_constexpr(WINDOW):
            w = TILE * tw - 1 + b
            w_ok = (w >= 0) and (w < W)
            w_in = w if w_ok else Int32(0)
            mask = Float32(1.0) if (h_ok and w_ok) else Float32(0.0)
            frag = cute.make_rmem_tensor((IN_VECTOR,), mX.element_type)
            cute.autovec_copy(_vector_view(mX, (1, 1, 1, IN_VECTOR), (n, h_in, w_in, g)), frag)
            d.append(frag.load().to(Float32) * mask)
        rows.append(_bt4(d))
    return rows


def _in_xform_h_store(mV: cute.Tensor, rows: list, t64, g) -> None:
    """Apply B^T along H to the W-transformed rows and store ``V[i * WINDOW + j, t, channels]``."""
    for j in range(WINDOW):
        v = _bt4([rows[a][j] for a in range(WINDOW)])
        for i in range(WINDOW):
            out = cute.make_rmem_tensor((IN_VECTOR,), Float16)
            out.store(v[i].to(Float16))
            cute.autovec_copy(out, _vector_view(mV, (1, 1, IN_VECTOR), (i * WINDOW + j, t64, g)))


@cute.kernel
def _in_xform_kernel(
    mX: cute.Tensor,
    mV: cute.Tensor,
    N: cutlass.Constexpr,
    H: cutlass.Constexpr,
    W: cutlass.Constexpr,
    cin: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    groups = cin // IN_VECTOR
    lin = bidx * THREADS + tidx
    if lin < num_tiles(N, H, W) * groups:
        n, th, tw, t64, g = _tile_coords(lin, groups, H, W)
        rows = _in_load_xform_w(mX, n, th, tw, g, H, W)
        _in_xform_h_store(mV, rows, t64, g)


@cute.jit
def in_xform(
    mX: cute.Tensor,
    mV: cute.Tensor,
    N: cutlass.Constexpr,
    H: cutlass.Constexpr,
    W: cutlass.Constexpr,
    cin: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """Input transform: ``X [N, H, W, cin]`` (fp16 or bf16, ``W`` a multiple of 4) ->
    ``V [36, num_tiles(N, H, W), cin]`` fp16, tiles frame by frame in row-major order."""
    nblk = (num_tiles(N, H, W) * (cin // IN_VECTOR) + THREADS - 1) // THREADS
    _in_xform_kernel(mX, mV, N, H, W, cin).launch(
        grid=[nblk, 1, 1], block=[THREADS, 1, 1], stream=stream
    )


# ---------------------------------------------------------------- output transform
def _out_load_xform_w(mM: cute.Tensor, t64, g) -> list:
    """Load the GEMM products of tile ``t`` and apply A^T along W to each row: ``rows[i][q]``."""
    rows = []
    for i in range(WINDOW):
        m = []
        for j in range(WINDOW):
            frag = cute.make_rmem_tensor((OUT_VECTOR,), mM.element_type)
            cute.autovec_copy(_vector_view(mM, (1, 1, OUT_VECTOR), (i * WINDOW + j, t64, g)), frag)
            m.append(frag.load().to(Float32))
        rows.append(_at4(m))
    return rows


@cute.jit
def _out_xform_h_store(
    mY: cute.Tensor,
    rows: list,
    n,
    th,
    tw,
    g,
    H: cutlass.Constexpr,
):
    """Apply A^T along H to the W-transformed rows and store the output tile, dropping the rows
    below the frame."""
    for q in cutlass.range_constexpr(TILE):
        y_col = _at4([rows[i][q] for i in range(WINDOW)])
        for p in cutlass.range_constexpr(TILE):
            out = cute.make_rmem_tensor((OUT_VECTOR,), mY.element_type)
            out.store(y_col[p].to(mY.element_type))
            h = TILE * th + p
            if cutlass.const_expr(H % TILE == 0 or p == 0):
                cute.autovec_copy(
                    out, _vector_view(mY, (1, 1, 1, OUT_VECTOR), (n, h, TILE * tw + q, g))
                )
            else:
                if h < H:
                    cute.autovec_copy(
                        out, _vector_view(mY, (1, 1, 1, OUT_VECTOR), (n, h, TILE * tw + q, g))
                    )


@cute.kernel
def _out_xform_kernel(
    mM: cute.Tensor,
    mY: cute.Tensor,
    N: cutlass.Constexpr,
    H: cutlass.Constexpr,
    W: cutlass.Constexpr,
    cout: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    groups = cout // OUT_VECTOR
    lin = bidx * THREADS + tidx
    if lin < num_tiles(N, H, W) * groups:
        n, th, tw, t64, g = _tile_coords(lin, groups, H, W)
        rows = _out_load_xform_w(mM, t64, g)
        _out_xform_h_store(mY, rows, n, th, tw, g, H)


@cute.jit
def out_xform(
    mM: cute.Tensor,
    mY: cute.Tensor,
    N: cutlass.Constexpr,
    H: cutlass.Constexpr,
    W: cutlass.Constexpr,
    cout: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """Output transform: ``M [36, num_tiles(N, H, W), cout]`` (the GEMM products) -> ``Y [N, H,
    W, cout]`` in ``Y``'s dtype, ``W`` a multiple of 4."""
    nblk = (num_tiles(N, H, W) * (cout // OUT_VECTOR) + THREADS - 1) // THREADS
    _out_xform_kernel(mM, mY, N, H, W, cout).launch(
        grid=[nblk, 1, 1], block=[THREADS, 1, 1], stream=stream
    )
