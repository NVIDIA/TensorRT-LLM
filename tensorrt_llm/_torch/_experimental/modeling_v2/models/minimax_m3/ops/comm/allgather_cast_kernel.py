# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# Vocab-dim all-gather of bf16 logit shards into fp32 full-vocabulary rows -- CTM (prims/cute) kernel
# =============================================================================
#
# Every rank r holds shard_r, bf16 [rows, S] (its vocabulary columns [r S, r S + S)). Every rank writes
#   out[i, r S + j] = float(shard_r[i, j])          for every rank r, row i < rows, column j < S
# i.e. torch.cat(all_gather(shard), -1).float(): the gather, the concat and the cast in one launch. The cast is a bit
# shift (bf16 is the high half of fp32), so out holds every rank's bits exactly: -0.0, NaN payloads and infinities
# included.
#
# Geometry: G CTAs of 256 threads. The shard is N = rows S / 8 16-byte vectors (S a multiple of 8); CTA c owns
# vectors [c N / G, (c + 1) N / G) of every rank's shard. CTA c pushes its vectors of this rank's shard into slot
# [buffer][rank] of every rank's workspace (multimem stores through the multicast mapping), then raises its ready
# flag [rank][c] in every rank's workspace (one multimem max, release, system scope). It converts its vectors of this
# rank's own shard straight from the input meanwhile, then polls the flags [r][c] of the other ranks in its own copy
# (acquire, system scope) and converts their vectors from its own buffer.
#
# fp32 mode (MC-ALLGATHER, the draft-logits gather): the shard is fp32 [rows, S] (S a multiple of 4), the 16-byte
# vectors hold 4 fp32, and every vector is stored as is (no widening): out = torch.cat(all_gather(shard), -1), bit for
# bit. A slot then holds max_rows / 2 rows.
#
# Flags carry the call's epoch, not the data: there is no empty-word sentinel, so no bit pattern of the logits is
# reserved. A flag only grows (multimem max of the epoch), so it needs no re-arming, and calls with different grids
# share it: a reader waits until flag >= its epoch.
#
# ORDERING: every call must start after the previous call on its rank completed. The op launches the kernel without
# programmatic dependent launch, so a launch waits for all prior work on the stream, whatever kernels run between two
# calls (a PDL launch would rely on every kernel in between executing griddepcontrol.wait). The kernel still releases
# its own dependents early.
#
# Buffers: 3 rotating buffers, call n using buffer n % 3. Rank r's call n + 3 reuses the buffer that rank q reads in
# call n only after r's call n + 2 completed, which waited for q's pushes of call n + 2, which q issues after its call
# n + 1 (and so n) completed.
#
# flags (rank-local int32): [0] epoch of this call, [1] CTAs arrived, [4] debug code, [5] debug detail. The last CTA to
# arrive advances the epoch. A flag poll that waits longer than POLL_TIMEOUT_NS (a peer that never raises the column,
# e.g. a grid that differs between ranks) writes the debug words and traps instead of hanging.
# =============================================================================
"""CTM vocab-dim all-gather + concat + fp32 cast of bf16 logit shards over a multicast NVLink workspace, in one
launch."""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

THREADS = 256
BUFFERS = 3
FLAG_WORDS = 8
POLL_TIMEOUT_NS = 30_000_000_000  # 30 s: far past any straggler, short of a hung job
DEBUG_POLL = 2
VEC_ELEMS = 8  # bf16 per 16-byte vector
VEC_ELEMS_FP32 = 4  # fp32 per 16-byte vector (fp32 mode)
HI_MASK = -65536  # 0xFFFF0000


def vectors(rows: int, shard: int) -> int:
    """16-byte vectors of one rank's [rows, shard] bf16 slice."""
    return rows * shard // VEC_ELEMS


def slot_words(max_rows: int, shard: int) -> int:
    """Int32 words of one rank's slot in one buffer."""
    return max_rows * shard // 2


def workspace_words(max_rows: int, shard: int, world: int, max_grid: int) -> int:
    """Int32 words of the multicast workspace: the buffers, then the [world][max_grid] ready flags."""
    return BUFFERS * world * slot_words(max_rows, shard) + world * max_grid


def flag_base(max_rows: int, shard: int, world: int) -> int:
    return BUFFERS * world * slot_words(max_rows, shard)


@dsl_user_op
def _atomic_add_acq_rel(addr_i64, val, *, loc=None, ip=None):
    """atom.acq_rel.gpu.global.add.u32, returning the old value."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;", "=r,l,r", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _multimem_st_v4(addr_i64, a, b, c, d, *, loc=None, ip=None):
    """multimem.st.global.v4.f32: one 16-byte store into every rank's copy (the bits, as stored)."""
    _llvm.inline_asm(
        None,
        [addr_i64.ir_value(loc=loc, ip=ip), a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip),
         c.ir_value(loc=loc, ip=ip), d.ir_value(loc=loc, ip=ip)],
        "multimem.st.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _multimem_max_release_sys(addr_i64, val, *, loc=None, ip=None):
    """multimem.red.release.sys.global.max.u32: raise every rank's copy of a flag to val, ordered after the CTA's
    stores (the caller's barrier makes them the releasing thread's)."""
    _llvm.inline_asm(
        None,
        [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "multimem.red.release.sys.global.max.u32 [$0], $1;", "l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _ld_acquire_sys(addr_i64, *, loc=None, ip=None):
    """ld.acquire.sys.global.u32."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip)],
            "ld.acquire.sys.global.u32 $0, [$1];", "=r,l", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _globaltimer(*, loc=None, ip=None):
    """%globaltimer (ns)."""
    return cutlass.Int64(
        _llvm.inline_asm(
            _T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _trap(*, loc=None, ip=None):
    """fence.sc.sys (the debug words reach memory), then trap."""
    _llvm.inline_asm(
        None, [], "fence.sc.sys;\n\ttrap;", "", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


def _store_fp32x8(out, word, got):
    """The 8 bf16 of a 16-byte vector, widened to fp32 bit for bit, as two 16-byte stores at out word `word`."""
    lo = []
    for q in range(4):
        w = cutlass.Int32(got[q])
        lo.append(cutlass.Int32(w << cutlass.Int32(16)))
        lo.append(cutlass.Int32(w & cutlass.Int32(HI_MASK)))
    out.store((lo[0], lo[1], lo[2], lo[3]), idx=word, alignment=16)
    out.store((lo[4], lo[5], lo[6], lo[7]), idx=word + cutlass.Int32(4), alignment=16)


def _store_out(out, word, got, fp32: bool):
    """One 16-byte input vector at out word `word`: 4 fp32 copied as is (fp32 mode), else 8 bf16 widened."""
    if cutlass.const_expr(fp32):
        out.store(
            (
                cutlass.Int32(got[0]),
                cutlass.Int32(got[1]),
                cutlass.Int32(got[2]),
                cutlass.Int32(got[3]),
            ),
            idx=word,
            alignment=16,
        )
    else:
        _store_fp32x8(out, word, got)


@cute.kernel
def allgather_cast_kernel(
    shard: cutlass.Array,  # int32 words of bf16 [rows, S]: this rank's shard
    out: cutlass.Array,  # int32 bits of fp32 [rows, world * S], out
    buf_uc: cutlass.Array,  # int32 words: this rank's workspace
    buf_mc: cutlass.Array,  # int32 words: its multicast mapping
    flags: cutlass.Array,  # int32 [FLAG_WORDS], rank-local
    rank: cutlass.Int32,
    grid: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
    shard_cols: cutlass.Constexpr[int],  # S
    world: cutlass.Constexpr[int],
    slot: cutlass.Constexpr[int],  # int32 words per (buffer, rank) slot
    flag_off: cutlass.Constexpr[int],  # first flag word of the workspace
    max_grid: cutlass.Constexpr[int],  # flag columns per rank
    fp32: cutlass.Constexpr[bool],  # fp32 shard copied as is (else bf16 widened to fp32)
):
    ve = VEC_ELEMS_FP32 if fp32 else VEC_ELEMS
    n_vec = rows * shard_cols // ve
    row_vecs = shard_cols // ve
    out_row = world * shard_cols

    tx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    s_epoch = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)

    # The shard is the previous kernel's output and the epoch the previous call's: both after the grid wait.
    prims.griddepcontrol(prims.GridDepAction.WAIT)
    if tx == 0:
        s_epoch.store(flags.load(idx=0, is_volatile=True), idx=0)
    prims.barrier_cta_sync(0)
    epoch = s_epoch.load(idx=0)
    buf_base = (epoch % cutlass.Int32(BUFFERS)) * cutlass.Int32(world * slot)

    lo = cta * cutlass.Int32(n_vec) // cutlass.Int32(grid)
    hi = (cta + cutlass.Int32(1)) * cutlass.Int32(n_vec) // cutlass.Int32(grid)

    # Push this CTA's vectors of this rank's shard into every rank's slot [buffer][rank].
    mine = buf_base + rank * cutlass.Int32(slot)
    v = lo + tx
    while v < hi:
        got = shard.load(idx=v * cutlass.Int32(4), vector_size=4, alignment=16)
        _multimem_st_v4(
            buf_mc.data_ptr(mine + v * cutlass.Int32(4)).toint(),
            cutlass.Int32(got[0]), cutlass.Int32(got[1]), cutlass.Int32(got[2]), cutlass.Int32(got[3]),
        )  # fmt: skip
        v += cutlass.Int32(THREADS)
    prims.barrier_cta_sync(0)
    target = epoch + cutlass.Int32(1)
    if tx == 0:
        _multimem_max_release_sys(
            buf_mc.data_ptr(cutlass.Int32(flag_off) + rank * cutlass.Int32(max_grid) + cta).toint(),
            target,
        )
    prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)

    # This rank's own columns, straight from the input while the peers' pushes are in flight.
    v = lo + tx
    while v < hi:
        got = shard.load(idx=v * cutlass.Int32(4), vector_size=4, alignment=16)
        row = v // cutlass.Int32(row_vecs)
        col = v - row * cutlass.Int32(row_vecs)
        _store_out(
            out,
            row * cutlass.Int32(out_row)
            + rank * cutlass.Int32(shard_cols)
            + col * cutlass.Int32(ve),
            got,
            fp32,
        )
        v += cutlass.Int32(THREADS)

    # Thread r waits for rank r's flag [r][cta]; the barrier hands its acquire to the whole CTA.
    if tx < cutlass.Int32(world):
        if tx != rank:
            addr = buf_uc.data_ptr(
                cutlass.Int32(flag_off) + tx * cutlass.Int32(max_grid) + cta
            ).toint()
            start = _globaltimer()
            while _ld_acquire_sys(addr) - target < cutlass.Int32(0):
                if _globaltimer() - start > cutlass.Int64(POLL_TIMEOUT_NS):
                    flags.store(cutlass.Int32(DEBUG_POLL), idx=4)
                    flags.store(tx * cutlass.Int32(65536) + cta, idx=5)
                    _trap()
    prims.barrier_cta_sync(0)

    for r in cutlass.range_constexpr(world):
        if cutlass.Int32(r) != rank:
            src = buf_base + cutlass.Int32(r * slot)
            v = lo + tx
            while v < hi:
                got = buf_uc.load(
                    idx=src + v * cutlass.Int32(4), vector_size=4, alignment=16, is_volatile=True
                )
                row = v // cutlass.Int32(row_vecs)
                col = v - row * cutlass.Int32(row_vecs)
                _store_out(
                    out, row * cutlass.Int32(out_row) + cutlass.Int32(r * shard_cols) + col * cutlass.Int32(ve),
                    got, fp32,
                )  # fmt: skip
                v += cutlass.Int32(THREADS)

    # Every CTA read the epoch before arriving; the last one advances it for the next call.
    if tx == 0:
        arrived = _atomic_add_acq_rel(flags.data_ptr(1).toint(), cutlass.Int32(1))
        if arrived == cutlass.Int32(grid - 1):
            flags.store(cutlass.Int32(0), idx=1)
            flags.store(target, idx=0)


@cute.jit
def allgather_cast(
    shard: cute.Tensor,
    out: cute.Tensor,
    buf_uc: cute.Tensor,
    buf_mc: cute.Tensor,
    flags: cute.Tensor,
    rank: cutlass.Int32,
    grid: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
    shard_cols: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    slot: cutlass.Constexpr[int],
    flag_off: cutlass.Constexpr[int],
    max_grid: cutlass.Constexpr[int],
    fp32: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    allgather_cast_kernel(
        shard, out, buf_uc, buf_mc, flags, rank, grid, rows, shard_cols, world, slot, flag_off, max_grid, fp32,
    ).launch(
        grid=[grid, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
        use_pdl=use_pdl,
    )  # fmt: skip
