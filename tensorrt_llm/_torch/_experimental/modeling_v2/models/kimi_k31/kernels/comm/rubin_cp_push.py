# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-sided MNNVL push of row blocks over a group of ranks, with an epoch barrier (CuTe DSL).

Every rank of the group maps one segment per rank (``peers[d]`` is the address of rank ``d``'s
segment on this rank). A push copies, for every destination ``d`` and every block ``b``, rows
``r < n_d`` of ``row_bytes_b`` bytes

    src_b + d * src_dest_b + (s0_d + r) * src_row_b          (this rank's memory)

to

    peers[d] + dst_off_b + me * dst_src_b + (d0_d + r) * dst_row_b      (rank d's segment)

with ``(s0_d, d0_d, n_d)`` row ``d`` of an int32 ``[world, 3]`` table shared by the blocks. No
staging copy on either side: the sources and the destinations are strided row blocks.

Completion: every thread fences its stores at system scope, the CTA counts itself on this GPU's
counter (GPU scope), and the last CTA to count stores ``epoch`` into flag ``me`` of every rank's flag area, then waits
until flag ``s`` of its own area holds ``epoch`` for every source ``s``, and resets the counter.
When the kernel ends, this rank's segment holds what every source pushed to it, and every rank of
the group has started the same push, so the stream work each rank issued before it is complete.

The last CTA spins on flags other ranks write: the kernel must not share the GPU with another
kernel that waits on the other ranks (issue it on the stream of the forward, between layers).
"""

from __future__ import annotations

import functools
from typing import NamedTuple, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: Threads per CTA.
THREADS = 256
#: Row blocks one push carries.
MAX_BLOCKS = 5
#: Bytes of a vector a thread moves, largest first; a block uses the largest that divides every
#: address and stride of it.
VECTOR_BYTES = (16, 8, 4)
#: Bytes of the flag area of a segment: one int32 per source rank.
FLAG_BYTES = 4096
#: CTAs per SM of a push's grid (fewer when it has fewer tiles).
_CTAS_PER_SM = 4
_OPTIONS = "--opt-level 2 --enable-tvm-ffi"


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _ld_nc_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_nc_v2(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _ld_nc_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_nc_u64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.u64 $0, [$1];",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_v4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_v2(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_b32(addr, v0, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(v0, loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_sys(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.sys;", "", loc=loc, ip=ip)


@dsl_user_op
def _atom_add_acq_rel_gpu(addr, value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_release_sys(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.release.sys.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_acquire_sys(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_relaxed_gpu(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.relaxed.gpu.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@cute.jit
def _load_vec(src: Int64, vb: int):
    """One ``vb``-byte vector at ``src`` (memory no kernel writes during the push)."""
    if cutlass.const_expr(vb == 16):
        return _ld_nc_v4(src)
    elif cutlass.const_expr(vb == 8):
        return _ld_nc_v2(src)
    else:
        return (_ld_nc_b32(src),)


@cute.jit
def _store_vec(dst: Int64, words: tuple, vb: int):
    if cutlass.const_expr(vb == 16):
        _st_v4(dst, *words)
    elif cutlass.const_expr(vb == 8):
        _st_v2(dst, *words)
    else:
        _st_b32(dst, words[0])


class CpPushKernel:
    """The push of ``blocks`` over ``world`` ranks.

    ``blocks``: per block ``(vector bytes, vectors per row, src_row, src_dest, dst_row, dst_src)``,
    the row geometry in bytes, fixed per call site; a call passes each block's source address,
    destination offset and row tiles. Grid ``nctas`` CTAs of ``THREADS`` threads; a CTA strides
    over the tiles of every block, a tile is ``rows_per_tile`` rows of one destination,
    destinations varying fastest from ``me`` on (the CTAs in flight write to every peer at once).
    """

    def __init__(self, world: int, blocks: tuple):
        if not 1 <= len(blocks) <= MAX_BLOCKS:
            raise ValueError(f"{len(blocks)} blocks, at most {MAX_BLOCKS}")
        self.P = world
        self.blocks = tuple(blocks)
        self.NB = len(blocks)
        self.rows_per_tile = tuple(rows_per_tile(b[1]) for b in blocks)
        # Vectors a thread moves per tile: all loads first, then the stores.
        self.per_thread = tuple(
            -(-(br * b[1]) // THREADS) for br, b in zip(self.rows_per_tile, blocks)
        )

    @cute.jit
    def __call__(
        self,
        peers: Int64,
        table: cute.Tensor,
        counter: Int64,
        flags_off: Int64,
        epoch: Int32,
        me: Int32,
        nctas: Int32,
        src0: Int64,
        dst0: Int64,
        tiles0: Int32,
        src1: Int64,
        dst1: Int64,
        tiles1: Int32,
        src2: Int64,
        dst2: Int64,
        tiles2: Int32,
        src3: Int64,
        dst3: Int64,
        tiles3: Int32,
        src4: Int64,
        dst4: Int64,
        tiles4: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            peers,
            table,
            counter,
            flags_off,
            epoch,
            me,
            nctas,
            src0,
            dst0,
            tiles0,
            src1,
            dst1,
            tiles1,
            src2,
            dst2,
            tiles2,
            src3,
            dst3,
            tiles3,
            src4,
            dst4,
            tiles4,
        ).launch(grid=(nctas, 1, 1), block=(THREADS, 1, 1), stream=stream)

    @cute.jit
    def copy_block(
        self,
        b: int,
        src: Int64,
        dst_off: Int64,
        tiles: Int32,
        peers: Int64,
        table: cute.Tensor,
        me: Int32,
        nctas: Int32,
        bidx: Int32,
        tidx: Int32,
    ):
        vb, nv, src_row, src_dest, dst_row, dst_src = self.blocks[b]
        br = self.rows_per_tile[b]
        per = self.per_thread[b]
        for t in cutlass.range(bidx, tiles * self.P, nctas, unroll=1):
            d = (t % self.P + me) % self.P
            r0 = (t // self.P) * br
            s0 = table[d * 3]
            d0 = table[d * 3 + 1]
            n = table[d * 3 + 2]
            if r0 < n:
                s_base = src + Int64(d) * src_dest + Int64(s0 + r0) * src_row
                d_base = (
                    _ld_nc_u64(peers + Int64(d * 8))
                    + dst_off
                    + Int64(me) * dst_src
                    + Int64(d0 + r0) * dst_row
                )
                # This thread's vectors of the tile: rows past n (or past the tile) read the
                # tile's first vector and store nothing.
                live = n - r0
                offs_s = []
                offs_d = []
                oks = []
                for k in cutlass.range_constexpr(per):
                    i = tidx + k * THREADS
                    r = i // nv
                    v = i - r * nv
                    ok = (r < live) & (i < br * nv)
                    keep = Int32(ok)
                    offs_s.append((Int64(r) * src_row + Int64(v * vb)) * Int64(keep))
                    offs_d.append(Int64(r) * dst_row + Int64(v * vb))
                    oks.append(ok)
                vals = [_load_vec(s_base + offs_s[k], vb) for k in range(per)]
                for k in cutlass.range_constexpr(per):
                    if oks[k]:
                        _store_vec(d_base + offs_d[k], vals[k], vb)

    @cute.kernel
    def kernel(
        self,
        peers: Int64,
        table: cute.Tensor,
        counter: Int64,
        flags_off: Int64,
        epoch: Int32,
        me: Int32,
        nctas: Int32,
        src0: Int64,
        dst0: Int64,
        tiles0: Int32,
        src1: Int64,
        dst1: Int64,
        tiles1: Int32,
        src2: Int64,
        dst2: Int64,
        tiles2: Int32,
        src3: Int64,
        dst3: Int64,
        tiles3: Int32,
        src4: Int64,
        dst4: Int64,
        tiles4: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        srcs = (src0, src1, src2, src3, src4)
        dsts = (dst0, dst1, dst2, dst3, dst4)
        tiles = (tiles0, tiles1, tiles2, tiles3, tiles4)
        for b in cutlass.range_constexpr(self.NB):
            self.copy_block(b, srcs[b], dsts[b], tiles[b], peers, table, me, nctas, bidx, tidx)
        # Completion: this CTA's stores are fenced at system scope before it counts (the counter
        # is this GPU's: GPU scope); the last CTA acquires every count, signals every rank at
        # system scope (release, cumulative over the CTAs' stores), then waits for every source.
        _fence_sys()
        cute.arch.barrier()
        if tidx == 0:
            old = _atom_add_acq_rel_gpu(counter, Int32(1))
            if old == nctas - 1:
                for p in cutlass.range_constexpr(self.P):
                    peer = _ld_nc_u64(peers + Int64(p * 8))
                    _st_release_sys(peer + flags_off + Int64(me) * 4, epoch)
                own = _ld_nc_u64(peers + Int64(me) * 8) + flags_off
                for p in cutlass.range_constexpr(self.P):
                    f = _ld_acquire_sys(own + Int64(p * 4))
                    while f < epoch:
                        f = _ld_acquire_sys(own + Int64(p * 4))
                _st_relaxed_gpu(counter, Int32(0))


def rows_per_tile(vectors_per_row: int) -> int:
    """Rows of one tile: about four vectors per thread, at most 64 rows."""
    return max(1, min(64, (THREADS * 4) // vectors_per_row))


class PushArgs(NamedTuple):
    """One block of a push, in bytes (addresses and strides; see the module docstring)."""

    src: int
    src_row: int
    src_dest: int
    row_bytes: int
    dst_off: int
    dst_row: int
    dst_src: int
    max_rows: int  # the most rows of any destination (sizes the grid)


def vector_bytes(block: PushArgs) -> int:
    """The largest vector width that divides every address, stride and the row of ``block``."""
    terms = (
        block.src,
        block.src_row,
        block.src_dest,
        block.row_bytes,
        block.dst_off,
        block.dst_row,
        block.dst_src,
    )
    for vb in VECTOR_BYTES:
        if not any(x % vb for x in terms):
            return vb
    raise ValueError(f"a push block needs 4-byte aligned rows and strides: {block}")


@functools.cache
def _compiled(world: int, blocks: tuple):
    kernel = CpPushKernel(world, blocks)
    a, i = Int64(1 << 20), Int32(1)
    table = cute.runtime.make_fake_tensor(cutlass.Int32, (3 * world,), stride=(1,))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel, a, table, a, a, i, Int32(0), i, *((a, a, i) * MAX_BLOCKS), stream, options=_OPTIONS
    )


@functools.cache
def _sm_count(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def run_cp_push(
    peers: int,
    table: torch.Tensor,
    counter: int,
    flags_off: int,
    epoch: int,
    me: int,
    world: int,
    blocks: Sequence[PushArgs],
) -> None:
    """Launch the push of ``blocks`` (device addresses as integers) on the current stream.

    ``peers``: int64 ``[world]`` device array of the segments; ``table``: contiguous int32
    ``[3 * world]`` device tensor of ``(s0, d0, n)`` per destination; ``counter``: a zeroed int32
    this rank's pushes share; ``flags_off``: byte offset of the flag area in every segment; ``epoch``: larger than every
    earlier push's on this workspace.
    """
    shapes, args, tiles_total = [], [], 0
    for blk in blocks:
        vb = vector_bytes(blk)
        nv = blk.row_bytes // vb
        tiles = -(-blk.max_rows // rows_per_tile(nv)) if blk.max_rows > 0 else 0
        tiles_total += tiles * world
        shapes.append((vb, nv, blk.src_row, blk.src_dest, blk.dst_row, blk.dst_src))
        args += (blk.src, blk.dst_off, tiles)
    args += (0, 0, 0) * (MAX_BLOCKS - len(blocks))
    nctas = max(1, min(tiles_total, _CTAS_PER_SM * _sm_count(torch.cuda.current_device())))
    _compiled(world, tuple(shapes))(peers, table, counter, flags_off, epoch, me, nctas, *args)


__all__ = [
    "CpPushKernel",
    "FLAG_BYTES",
    "MAX_BLOCKS",
    "PushArgs",
    "rows_per_tile",
    "run_cp_push",
    "vector_bytes",
]
