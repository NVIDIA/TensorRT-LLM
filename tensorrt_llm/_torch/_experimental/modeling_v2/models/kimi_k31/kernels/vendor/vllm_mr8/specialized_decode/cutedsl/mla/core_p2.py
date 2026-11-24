# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modified by NVIDIA for TensorRT-LLM (2026): environment-variable switches replaced by constants (lines 41-57).
"""DS-T2P: the persistent DS-T2 on mla.wlist's work list (the B8-B32 MLA decode core).

DS-T2 (mla.core_d2): 2-CTA clusters of two 128-row tiles of one request and value half,
cta_group::2 M256 FP8 MMAs with Q and P in TMEM, each CTA loading 26
KB of each 64-key page. Here each cluster runs a sequence of work-list items (q tile
pair, value half, request, KV split) and the pipeline runs on across items, so an item's
fixed work (Q intake, ring ramp, pipeline fill) overlaps the previous item's tail; only
the read of O out of TMEM sits between an item's last PV and the next item's first.

An odd q-tile count (DL3 q = 4: 3 tiles per request) ends each (request, split)
with an odd item: both CTAs take the last tile (identical Q, so identical S and P)
and the page's PV runs as two lane-masked M256 MMAs (disable-output-lane): rank 0's
lanes against value columns 0-255, rank 1's against 256-511, each CTA loading two
128-column value blocks per page. No padded tile.

Numerics: DS-T2's (the same partitions as core_q's work list: mla.dcp.long_per).
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64

from ..common import tensormap
from ..common.primitives import (
    asm,
    fence_tmem,
    store_words,
)
from .dcp import long_per, lpc_chunk, lpc_table
from .qvalid import expect_valid, init_valid, tma_valid3, wait_valid

ROWS = 128  # rows per tile (TMEM lanes)
PAGE = 64  # keys per KV page = per S stage
NRD = 8  # QK-half ring slots after the Q staging
NVD = 12  # value-block ring slots after them
NWG = 3  # softmax warp groups (global pages jj % NWG); groups 0-1 also run the epilogue
# the Q staging buffer (72 KB) joins the rings (2 QK halves + 4 value blocks) once Q is
# in TMEM
NR, NV = NRD + 2, NVD + 4
HALF = 32  # keys per CTA in a page's QK (N64 split across the pair)
# diagnostics only (DS_DEBUG, not read here: none): noq,noepi (timing probes; wrong results),
# pstamp / prog (clocks, per-CTA progress and item store times), smemchk (SMEM size registers
# and a 4 KB top-of-allocation canary; with prog)
DEBUG = frozenset()
# the epilogue's partial stores: "shfl" (an 8 x 8 butterfly transpose of 16 B chunks per
# 8 lanes: each store writes 4 full 128 B row segments; 1 GPU, AgentX mix8-32: -0.7 to
# -5.6% vs "rows") or "rows" (each thread its row's 32 B pieces); the same values
EPI = "shfl"
# load pages that precede this step's tokens before the PDL wait (A/B switch)
PRE = 1
KBLOCK = PAGE * 128  # one 128-column latent block of a page (8 KB)
HBLOCK = HALF * 128  # one 128-column latent block of half a page's keys (4 KB)
RSLOT = 4 * HBLOCK + HALF * 64  # this CTA's 32 keys: latent + RoPE (18 KB)
VSLOT = PAGE * 128  # this CTA's 128 value columns of the page's 64 keys (8 KB)
QBLOCK = ROWS * 128  # one 128-column latent block of the Q tile (16 KB)
QSTAGE = 4 * QBLOCK + ROWS * 64  # the Q tile's staging image (72 KB)
DATA = QSTAGE + NRD * RSLOT + NVD * VSLOT
# TMEM columns
T_S, T_O, T_P, T_Q = 0, 128, 384, 432  # S 2 x 64, O 256, P 3 x 16, Q 144
NP = 3  # P stages
TMEM_COLS = 576
# instruction descriptors: FP32 D, E4M3 A and B, M = 128, bit 29 = Rubin dense FP8
# (K = 64); N at bits 17-22, B MN-major at bit 16
QK_INSTR = 1 << 4 | (PAGE >> 3) << 17 | (ROWS >> 4) << 24 | 1 << 29
PV_INSTR = 1 << 4 | 1 << 16 | (256 >> 3) << 17 | (ROWS >> 4) << 24 | 1 << 29
# cta_group::2: M = 256 (each CTA's 128 rows)
QK2_INSTR = 1 << 4 | (PAGE >> 3) << 17 | (256 >> 4) << 24 | 1 << 29
PV2_INSTR = 1 << 4 | 1 << 16 | (256 >> 3) << 17 | (256 >> 4) << 24 | 1 << 29
SW128, SW64 = 2, 4  # descriptor layout types
MASKED = -1.0e6  # tokenspeed's finite mask score
LOG2_E = 1.4426950408889634
SKIP = 8.0  # tokenspeed's skip_correction_threshold
X_SENT = 0xFFFFFFFF
NT = 32 * (
    4 * NWG + 4
)  # softmax groups, QK issuer, Q + QK-half loader, value loader, PV
W_MMA, W_K0, W_PV = 4 * NWG, 4 * NWG + 1, 4 * NWG + 3
NBARS = 2 * NRD + 2 * NVD + 2 * NP + NWG + 24
WLIST_EMPTY_WORD = 0x7FFF << 16  # mla.wlist.WLIST_EMPTY

# Rubin TP16 DCP16 rows-mode shapes: (rows, gathered heads) -> (q, short split,
# launched split, long tiles), as mla.core_lc.ROWS (mla.dcp.long_per partition).
ROWS_TABLE = {(8, 96): (8, 6, 17, 48), (8, 64): (7, 6, 26, 40)}


@asm()
def tma4(dst, descriptor, c0, c1, c2, c3, barrier):
    code = "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
    code += ".mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4, $5}], [$6];"
    operands = (dst, Int64(descriptor), Int32(c0), Int32(c1), Int32(c2), Int32(c3))
    return code, "r,l,r,r,r,r,r,~{memory}", (*operands, barrier)


@asm()
def tma3(dst, descriptor, c0, c1, c2, barrier):
    code = "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
    code += ".mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4}], [$5];"
    operands = (dst, Int64(descriptor), Int32(c0), Int32(c1), Int32(c2))
    return code, "r,l,r,r,r,r,~{memory}", (*operands, barrier)


@asm()
def mbar_init(barrier, count):
    return f"mbarrier.init.shared::cta.b64 [$0], {count};", "r,~{memory}", (barrier,)


@asm()
def mbar_arrive(barrier):
    code = "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];"
    return code, "r,~{memory}", (barrier,)


@asm()
def mma_tmem_a(d, a, b, instruction, accumulate):
    """One E4M3 MMA with A (K-major, 4 per column) from TMEM."""
    code = "{ .reg .pred p; setp.ne.b32 p,$4,0; "
    code += "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0],[$1],$2,$3,p; }"
    operands = (Uint32(d), Uint32(a), Uint64(b), Uint32(instruction))
    return code, "r,r,l,r,r", (*operands, Uint32(accumulate))


@asm()
def ld_tmem32(address):
    """32 FP32 columns of this lane, waiting for the load."""
    names = ",".join(f"${i}" for i in range(32))
    code = f"tcgen05.ld.sync.aligned.32x32b.x32.b32 {{{names}}},[$32]; "
    code += "tcgen05.wait::ld.sync.aligned;"
    constraints = ",".join(["=f"] * 32 + ["r", "~{memory}"])
    return code, constraints, (Uint32(address),), (Float32,) * 32


@asm()
def ld_red32(address):
    """32 FP32 columns of this lane and their max (tcgen05.ld.red), waiting."""
    names = ",".join(f"${i}" for i in range(32))
    code = f"tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 {{{names}}}, $32, [$33]; "
    code += "tcgen05.wait::ld.sync.aligned;"
    constraints = ",".join(["=f"] * 33 + ["r", "~{memory}"])
    return code, constraints, (Uint32(address),), (Float32,) * 33


@asm((Float32, Float32), pure=True)
def ffma2(a0, a1, b, c):
    """fma.rn.f32x2 of (a0, a1) * (b, b) + (c, c) (IEEE FP32 per lane)."""
    code = "{ .reg .b64 x, y, z, w; mov.b64 x, {$2, $3}; mov.b64 y, {$4, $4}; "
    code += "mov.b64 z, {$5, $5}; fma.rn.f32x2 w, x, y, z; mov.b64 {$0, $1}, w; }"
    return code, "=f,=f,f,f,f,f", (Float32(a0), Float32(a1), Float32(b), Float32(c))


@asm()
def qk_block(d, a, blo, bhi, brope, instruction, bar, bar2):
    """One elected lane: the page's QK into TMEM ``d`` -- 8 latent K64 MMAs (A = Q in
    TMEM columns a + 16 k, B = the 128B-swizzled image of latent dims 0-255 ``blo`` (k <
    4) or 256-511 ``bhi``, advanced (k % 4 // 2) * 8 KB + (k % 2) * 64 B), the RoPE MMA
    (A = a + 128, B = ``brope``), then commits to ``bar`` and ``bar2``."""
    mma = "tcgen05.mma.cta_group::1.kind::f8f6f4"
    commit = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = "{ .reg .pred e, pf, pt; .reg .b32 ta; .reg .b64 tb; "
    code += "elect.sync _|e, 0xffffffff; setp.ne.u32 pf, 0, 0; setp.eq.u32 pt, 0, 0; "
    for k in range(8):
        off = (k % 4 // 2) * (KBLOCK >> 4) + (k % 2) * 4
        code += (
            f"add.u32 ta, $1, {16 * k}; add.s64 tb, {'$2' if k < 4 else '$3'}, {off}; "
        )
        code += f"@e {mma} [$0], [ta], tb, $5, {'pf' if k == 0 else 'pt'}; "
    code += f"add.u32 ta, $1, 128; @e {mma} [$0], [ta], $4, $5, pt; "
    code += f"@e {commit} [$6]; @e {commit} [$7]; }}"
    operands = (
        Uint32(d),
        Uint32(a),
        Uint64(blo),
        Uint64(bhi),
        Uint64(brope),
        Uint32(instruction),
    )
    return code, "r,r,l,l,l,r,r,r,~{memory}", (*operands, Uint32(bar), Uint32(bar2))


@asm()
def pv_block(d, a, b, instruction, accumulate, odone, empty, final, last):
    """One elected lane: the page's PV MMA, then commits to ``odone`` and ``empty`` (and
    ``final`` when ``last`` is nonzero)."""
    commit = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = "{ .reg .pred e, pa, pl; elect.sync _|e, 0xffffffff; "
    code += "setp.ne.u32 pa, $4, 0; setp.ne.u32 pl, $8, 0; and.pred pl, pl, e; "
    code += "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], [$1], $2, $3, pa; "
    code += f"@e {commit} [$5]; @e {commit} [$6]; @pl {commit} [$7]; }}"
    operands = (
        Uint32(d),
        Uint32(a),
        Uint64(b),
        Uint32(instruction),
        Uint32(accumulate),
    )
    return (
        code,
        "r,r,l,r,r,r,r,r,r,~{memory}",
        (*operands, Uint32(odone), Uint32(empty), Uint32(final), Uint32(last)),
    )


@asm()
def st_tmem(address, words):
    """``words`` (Uint32) into this lane's columns address.., waiting."""
    n = len(words)
    names = ",".join(f"${i + 1}" for i in range(n))
    code = f"tcgen05.st.sync.aligned.32x32b.x{n}.b32 [$0],{{{names}}}; "
    code += "tcgen05.wait::st.sync.aligned;"
    return code, "r" + ",r" * n + ",~{memory}", (Uint32(address), *words)


@asm()
def st_tmem_f(address, values):
    n = len(values)
    names = ",".join(f"${i + 1}" for i in range(n))
    code = f"tcgen05.st.sync.aligned.32x32b.x{n}.b32 [$0],{{{names}}}; "
    code += "tcgen05.wait::st.sync.aligned;"
    return code, "r" + ",f" * n + ",~{memory}", (Uint32(address), *values)


@asm((Uint32,) * 4)
def ld_shared4(address):
    code = "ld.shared.v4.b32 {$0,$1,$2,$3}, [$4];"
    return code, "=r,=r,=r,=r,r,~{memory}", (Uint32(address),)


@asm()
def st_shared4f(address, values):
    code = "st.shared.v4.f32 [$0], {$1, $2, $3, $4};"
    return code, "r,f,f,f,f,~{memory}", (Uint32(address), *map(Float32, values))


@asm()
def st_global4(pointer, words):
    code = "st.global.v4.b32 [$0], {$1, $2, $3, $4};"
    return code, "l,r,r,r,r,~{memory}", (Int64(pointer), *map(Uint32, words))


@asm(Float32)
def ld_shared_f(address):
    return "ld.shared.f32 $0, [$1];", "=f,r,~{memory}", (Uint32(address),)


@asm()
def st_shared_f(address, value):
    return "st.shared.f32 [$0], $1;", "r,f,~{memory}", (Uint32(address), Float32(value))


@asm(Uint32, pure=True)
def e4m3x4(a, b, c, d):
    """Four FP32 -> E4M3 (rn, satfinite), a in the low byte."""
    code = "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; "
    code += "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi}; }"
    return code, "=r,f,f,f,f", (Float32(a), Float32(b), Float32(c), Float32(d))


@asm((Float32, Float32), pure=True)
def add2(a, b):
    """add.rn.f32x2 of (a0, a1) + (b0, b1)."""
    code = "{ .reg .b64 x, y, z; mov.b64 x, {$2, $3}; mov.b64 y, {$4, $5}; "
    code += "add.rn.f32x2 z, x, y; mov.b64 {$0, $1}, z; }"
    return code, "=f,=f,f,f,f,f", (*map(Float32, a), *map(Float32, b))


@asm()
def dealloc_tmem(address, columns):
    """``tcgen05.dealloc`` with the column count in a register (576 on sm_107)."""
    code = "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;"
    return code, "r,r,~{memory}", (Uint32(address), Uint32(columns))


@cute.jit
def wait_bar(address, phase):
    """Mbarrier parity wait through the NVVM intrinsic (retry loop with a suspend-time
    hint: waiting warps sleep instead of spinning)."""
    ptr = cute.make_ptr(Int64, Int64(address), cute.AddressSpace.smem, assumed_align=8)
    cute.arch.mbarrier_wait(ptr, phase)


@asm()
def bar_arrive(barrier_id, count):
    """``bar.arrive`` with a register barrier id."""
    return f"bar.arrive $0, {count};", "r,~{memory}", (Int32(barrier_id),)


@asm()
def bar_sync(barrier_id, count):
    """``bar.sync`` with a register barrier id."""
    return f"bar.sync $0, {count};", "r,~{memory}", (Int32(barrier_id),)


@asm(Float32, pure=True)
def fmax(a, b):
    return "max.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def ffma(a, b, c):
    return "fma.rn.f32 $0,$1,$2,$3;", "=f,f,f,f", (Float32(a), Float32(b), Float32(c))


@asm(Float32, pure=True)
def ex2(x):
    return "ex2.approx.ftz.f32 $0,$1;", "=f,f", (Float32(x),)


@asm()
def store_f32x8(pointer, values):
    code = "st.global.v8.f32 [$0],{$1,$2,$3,$4,$5,$6,$7,$8};"
    return code, "l" + ",f" * 8 + ",~{memory}", (Int64(pointer), *values)


@asm()
def store_f32(pointer, value):
    return "st.global.f32 [$0],$1;", "l,f,~{memory}", (Int64(pointer), Float32(value))


@asm(Int64)
def globaltimer():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@asm(Uint32)
def smem_reg(which):
    """DS_DEBUG=smemchk: %total_smem_size (0) or %dynamic_smem_size (1)."""
    name = ("%total_smem_size", "%dynamic_smem_size")[which]
    return f"mov.u32 $0, {name};", "=r", ()


@asm()
def st_shared_u32(address, value):
    return "st.shared.u32 [$0], $1;", "r,r,~{memory}", (Uint32(address), Uint32(value))


@asm(Uint32)
def ld_shared_u32(address):
    return "ld.volatile.shared.u32 $0, [$1];", "=r,r,~{memory}", (Uint32(address),)


@asm(Int32)
def smid():
    return "mov.u32 $0, %smid;", "=r", ()


@asm(Int64)
def clock64():
    return "mov.u64 $0, %clock64;", "=l,~{memory}", ()


@cute.jit
def stamp_base(words, cta):
    """DS_DEBUG=pstamp: CTA 0-1's stamp rows ([2, 20, 64] int64 at words[6:8])."""
    base = Int64(0)
    if cutlass.const_expr("pstamp" in DEBUG):  # noqa: SIM102 (a trace-time condition)
        if cta < 2:
            address = Int64(Uint32(words[6])) | Int64(Uint32(words[7])) << 32
            base = address + Int64(cta * 20 * 64 * 8)
    return base


@cute.jit
def progress(words, cta, slot, value, lane):
    """DS_DEBUG=stamp / prog / pstamp: per-CTA progress words (after the stamp rows):
    64 slots per CTA: 0 QK page, 1 PV page, 2+g softmax group g page (prog), 16-23
    phase globaltimers, 24-25 clocks, 26 SM id, 32+w warp w at the end."""
    if cutlass.const_expr(("stamp" in DEBUG) | ("prog" in DEBUG)):  # noqa: SIM102
        if lane == 0:
            address = Int64(Uint32(words[6])) | Int64(Uint32(words[7])) << 32
            address = address + Int64(2 * 20 * 64 * 8) + Int64(cta * 64 + slot) * 8
            ptr = cute.make_ptr(Int64, address, cute.AddressSpace.gmem, assumed_align=8)
            cute.make_tensor(ptr, cute.make_layout(1))[0] = Int64(value) + Int64(1)


@cute.jit
def stamp(base, event, index, lane):
    """DS_DEBUG=pstamp: %clock64 of ``event`` (page ``index``) by lane 0 into the stamp
    rows of CTAs 0-1."""
    if cutlass.const_expr("pstamp" in DEBUG):  # noqa: SIM102 (a trace-time condition)
        if (base != 0) & (lane == 0):
            slot = event * 64 + cutlass.min(index, Int32(63))
            ptr = cute.make_ptr(
                Int64, base + Int64(slot) * 8, cute.AddressSpace.gmem, assumed_align=8
            )
            cute.make_tensor(ptr, cute.make_layout(1))[0] = clock64()


@cute.jit
def sdesc(
    address,
    lbo: cutlass.Constexpr[int],
    sbo: cutlass.Constexpr[int],
    layout: cutlass.Constexpr[int],
):
    """Shared memory matrix descriptor (sm100 format; the start address field is 15
    bits on sm_107 (bit 14 is reserved on sm_100): SMEM past 256 KB)."""
    value = Uint64((address >> 4) & 32767) | Uint64(1) << 46
    value |= Uint64(lbo >> 4) << 16 | Uint64(sbo >> 4) << 32 | Uint64(layout) << 61
    return value


@asm()
def tma4_2(dst, descriptor, c0, c1, c2, c3, barrier):
    """4-D TMA box into this CTA's ``dst``, completing on the pair leader's ``barrier``
    (a shared::cluster address; ``.cta_group::2``)."""
    code = "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
    code += (
        ".mbarrier::complete_tx::bytes.cta_group::2 [$0], [$1, {$2, $3, $4, $5}], [$6];"
    )
    operands = (dst, Int64(descriptor), Int32(c0), Int32(c1), Int32(c2), Int32(c3))
    return code, "r,l,r,r,r,r,r,~{memory}", (*operands, barrier)


@asm()
def tma3_2(dst, descriptor, c0, c1, c2, barrier):
    code = "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
    code += ".mbarrier::complete_tx::bytes.cta_group::2 [$0], [$1, {$2, $3, $4}], [$5];"
    operands = (dst, Int64(descriptor), Int32(c0), Int32(c1), Int32(c2))
    return code, "r,l,r,r,r,r,~{memory}", (*operands, barrier)


@asm(Uint32)
def mapa(address, rank):
    """This CTA's shared ``address`` in cluster CTA ``rank``'s window."""
    return (
        "mapa.shared::cluster.u32 $0, $1, $2;",
        "=r,r,r",
        (Uint32(address), Int32(rank)),
    )


@asm()
def leader_arrive(address):
    """One arrival on the pair leader's (rank 0) copy of the barrier at this CTA's
    offset ``address`` (default semantics, as CUTLASS's 2-SM pipelines: the TMEM hazards
    are ordered by the tcgen05 fences around the barrier)."""
    code = "{ .reg .b32 ra; mapa.shared::cluster.u32 ra, $0, 0; "
    code += "mbarrier.arrive.shared::cluster.b64 _, [ra]; }"
    return code, "r,~{memory}", (Uint32(address),)


@asm()
def arm_tx(barrier, nbytes):
    """One arrival on ``barrier`` expecting ``nbytes`` of complete_tx."""
    code = "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$0], $1;"
    return code, "r,r,~{memory}", (Uint32(barrier), Int32(nbytes))


@asm()
def qk_pair(d, a, blat, brope, instruction, bar, bar2):
    """The pair leader's elected lane: the page's cta_group::2 QK (M256: each CTA's Q
    rows in its TMEM; N64: each CTA's 32 keys at the same shared memory offset) -- 8
    latent K64 MMAs (B advanced (k // 2) * 4 KB + (k % 2) * 64 B) and the RoPE MMA --
    then commits multicast onto ``bar`` and ``bar2`` in both CTAs."""
    mma = "tcgen05.mma.cta_group::2.kind::f8f6f4"
    commit = "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
    commit += ".multicast::cluster.b64"
    code = "{ .reg .pred e, pf, pt; .reg .b32 ta; .reg .b64 tb; .reg .b16 m; "
    code += "mov.b16 m, 3; "
    code += "elect.sync _|e, 0xffffffff; setp.ne.u32 pf, 0, 0; setp.eq.u32 pt, 0, 0; "
    for k in range(8):
        off = (k // 2) * (HBLOCK >> 4) + (k % 2) * 4
        code += f"add.u32 ta, $1, {16 * k}; add.s64 tb, $2, {off}; "
        code += f"@e {mma} [$0], [ta], tb, $4, {'pf' if k == 0 else 'pt'}; "
    code += f"add.u32 ta, $1, 128; @e {mma} [$0], [ta], $3, $4, pt; "
    code += f"@e {commit} [$5], m; @e {commit} [$6], m; }}"
    operands = (Uint32(d), Uint32(a), Uint64(blat), Uint64(brope), Uint32(instruction))
    return code, "r,r,l,l,r,r,r,~{memory}", (*operands, Uint32(bar), Uint32(bar2))


@asm()
def pv_page(
    d, a, b, b2, instruction, accumulate, odone, empty, empty2, final, last, odd
):
    """The pair leader's elected lane: the page's cta_group::2 PV (as pv_pair), or for
    an odd item (``odd`` nonzero; both CTAs hold the same P) two lane-masked MMAs --
    rank 0's lanes against ``b`` (value columns 0-255), rank 1's against ``b2``
    (256-511) -- then commits multicast onto ``odone``, ``empty`` (``empty2`` when
    odd, ``final`` when ``last``) in both CTAs."""
    commit = "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
    commit += ".multicast::cluster.b64"
    mma = "tcgen05.mma.cta_group::2.kind::f8f6f4"
    code = "{ .reg .pred e, pa, pl, po, pn; .reg .b16 m; .reg .b32 z, f; mov.b16 m, 3; "
    code += "mov.b32 z, 0; mov.b32 f, 0xFFFFFFFF; elect.sync _|e, 0xffffffff; "
    code += "setp.ne.u32 pa, $5, 0; setp.ne.u32 pl, $10, 0; and.pred pl, pl, e; "
    code += "setp.ne.u32 po, $11, 0; and.pred pn, e, !po; and.pred po, po, e; "
    code += f"@pn {mma} [$0], [$1], $2, $4, pa; "
    code += f"@po {mma} [$0], [$1], $2, $4, {{z, z, z, z, f, f, f, f}}, pa; "
    code += f"@po {mma} [$0], [$1], $3, $4, {{f, f, f, f, z, z, z, z}}, pa; "
    code += f"@e {commit} [$6], m; @e {commit} [$7], m; @po {commit} [$8], m; "
    code += f"@pl {commit} [$9], m; }}"
    operands = (Uint32(d), Uint32(a), Uint64(b), Uint64(b2), Uint32(instruction))
    barriers = (Uint32(odone), Uint32(empty), Uint32(empty2), Uint32(final))
    return (
        code,
        "r,r,l,l,r,r,r,r,r,r,r,r,~{memory}",
        (*operands, Uint32(accumulate), *barriers, Uint32(last), Uint32(odd)),
    )


@asm()
def pv_pair(d, a, b, instruction, accumulate, odone, empty, final, last):
    """The pair leader's elected lane: the page's cta_group::2 PV (M256 N256: each CTA's
    P rows in its TMEM, each CTA's 128 value columns of the 64 keys), then commits
    multicast onto ``odone``, ``empty`` (and ``final`` when ``last``) in both CTAs."""
    commit = "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
    commit += ".multicast::cluster.b64"
    code = "{ .reg .pred e, pa, pl; .reg .b16 m; mov.b16 m, 3; "
    code += "elect.sync _|e, 0xffffffff; "
    code += "setp.ne.u32 pa, $4, 0; setp.ne.u32 pl, $8, 0; and.pred pl, pl, e; "
    code += "@e tcgen05.mma.cta_group::2.kind::f8f6f4 [$0], [$1], $2, $3, pa; "
    code += f"@e {commit} [$5], m; @e {commit} [$6], m; @pl {commit} [$7], m; }}"
    operands = (
        Uint32(d),
        Uint32(a),
        Uint64(b),
        Uint32(instruction),
        Uint32(accumulate),
    )
    return (
        code,
        "r,r,l,r,r,r,r,r,r,~{memory}",
        (*operands, Uint32(odone), Uint32(empty), Uint32(final), Uint32(last)),
    )


@asm()
def dealloc_tmem2(address, columns):
    """``tcgen05.dealloc.cta_group::2`` with the column count in a register."""
    code = "tcgen05.dealloc.cta_group::2.sync.aligned.b32 $0, $1;"
    return code, "r,r,~{memory}", (Uint32(address), Uint32(columns))


@cute.jit
def quarter_of(warp):
    return warp % 4


@cute.jit
def bx_of(rank, c):
    return rank + 2 * c


@asm()
def cp_q2(t0, d0):
    """The pair leader's elected lane: each CTA's Q tile, staging image -> TMEM (A
    layout) with 18 cta_group::2 128x256b copies: latent block kb's 32 B chunk h (SW128,
    SBO 1 KB) to columns t0 + 32 kb + 8 h, the RoPE's chunk h (SW64, SBO 512 B) to t0 +
    128 + 8 h. ``d0``: the descriptor of the latent block 0 at chunk 0 (start address
    field in 16 B units; the RoPE image starts 4 x 16 KB on)."""
    code = "{ .reg .pred e; .reg .b32 t; .reg .b64 d, r; elect.sync _|e, 0xffffffff; "
    for kb in range(4):
        for h in range(4):
            off = kb * (QBLOCK >> 4) + h * 2
            code += f"add.u32 t, $0, {32 * kb + 8 * h}; add.s64 d, $1, {off}; "
            code += "@e tcgen05.cp.cta_group::2.128x256b [t], d; "
    # the RoPE image: SW64 layout (2 at bit 61 -> 4) and SBO 512 B (bits 32-45: 1024 ->
    # 512)
    code += f"add.s64 r, $1, {4 * (QBLOCK >> 4)}; "
    code += "and.b64 r, r, 0x1FFFC000FFFFFFFF; or.b64 r, r, 0x8000002000000000; "
    for h in range(2):
        code += f"add.u32 t, $0, {128 + 8 * h}; add.s64 d, r, {h * 2}; "
        code += "@e tcgen05.cp.cta_group::2.128x256b [t], d; "
    code += "}"
    return code, "r,l,~{memory}", (Uint32(t0), Uint64(d0))


@asm()
def commit_pair(bar):
    """The pair leader's elected lane: commit (its prior tcgen05 ops) multicast onto
    ``bar`` in both CTAs."""
    code = "{ .reg .pred e; .reg .b16 m; mov.b16 m, 3; elect.sync _|e, 0xffffffff; "
    code += "@e tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
    code += ".multicast::cluster.b64 [$0], m; }"
    return code, "r,~{memory}", (Uint32(bar),)


@asm()
def store_f32x4_v(pointer, values):
    code = "st.global.v4.f32 [$0],{$1,$2,$3,$4};"
    return code, "l" + ",f" * 4 + ",~{memory}", (Int64(pointer), *map(Float32, values))


@asm()
def store_f32x8_v(pointer, values):
    code = "st.global.v8.f32 [$0],{$1,$2,$3,$4,$5,$6,$7,$8};"
    return code, "l" + ",f" * 8 + ",~{memory}", (Int64(pointer), *map(Float32, values))


class P2Core:
    """DS-T2 (mla.core_d2) as a persistent kernel on mla.wlist's work list: cluster c
    (of P = sms / 2) runs the list's words c, c + P, .. (q tile t = tile pair t // 2,
    value half t % 2 | request << 8 | split << 16). The rings, S / P stages and barrier
    phases run on across items: the next item's K pages load and its QK runs while the
    current item's softmax, PV and epilogue finish. Per item: Q staging -> TMEM by the
    leader's tcgen05.cp after the previous item's last QK; the first PV waits until both
    CTAs' epilogues have read the previous item's O out of TMEM."""

    def __init__(
        self,
        heads,
        q,
        split,
        cp,
        causal,
        x_rows,
        long,
        slots,
        pairs,
        x_columns=1536,
        reqs=1,
        lpc=False,
    ):
        self.H, self.q, self.split, self.cp = heads, q, split, tuple(cp)
        self.causal, self.x_rows, self.long = causal, x_rows, tuple(long)
        self.x_columns = x_columns
        self.R = q * heads
        self.tiles = -(-self.R // ROWS)
        # items per (request, split) (the work list's ``tiles``): tile pairs x 2 value
        # halves, then one odd item for an odd tile count
        self.even, self.odd = self.tiles // 2, self.tiles % 2
        self.items_per = 2 * self.even + self.odd
        self.reqs, self.slots, self.pairs = reqs, slots, pairs
        self.rounds = -(-slots // pairs)
        self.grid = (2 * pairs, 1, 1)
        # LPC (dcp.lpc_table): groups of items_per clusters, one item stream each; a
        # group runs at most reqs chunks
        self.lpc = lpc
        self.groups = pairs // self.items_per
        if lpc:
            self.rounds = reqs

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,
        kv: cute.Tensor,
        blocks: cute.Tensor,
        lens: cute.Tensor,
        totals: cute.Tensor,
        workspace: cute.Tensor,
        words: cute.Tensor,
        wl: cute.Tensor,
        scale: Float32,
        stream,
    ):
        import cuda.bindings.driver as cuda

        u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        none = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE
        sw = cuda.CUtensorMapSwizzle
        qa = query.iterator.toint()
        rows, requests = query.shape[1] * query.shape[2], query.shape[0]
        qdims, qstrides = (576, rows, requests), (576, rows * 576)
        qlat = tensormap.encode_map(
            qa, u8, qdims, qstrides, (128, ROWS, 1), sw.CU_TENSOR_MAP_SWIZZLE_128B, none
        )
        qrope = tensormap.encode_map(
            qa, u8, qdims, qstrides, (64, ROWS, 1), sw.CU_TENSOR_MAP_SWIZZLE_64B, none
        )
        ka = kv.iterator.toint()
        pages, prow, ppage = kv.shape[0], kv.stride[1], kv.stride[0]
        khalf = tensormap.encode_map(
            ka,
            u8,
            (128, PAGE, 4, pages),
            (prow, 128, ppage),
            (128, HALF, 4, 1),
            sw.CU_TENSOR_MAP_SWIZZLE_128B,
            l2,
        )
        kval = tensormap.encode_map(
            ka,
            u8,
            (128, PAGE, 4, pages),
            (prow, 128, ppage),
            (128, PAGE, 1, 1),
            sw.CU_TENSOR_MAP_SWIZZLE_128B,
            l2,
        )
        krope = tensormap.encode_map(
            ka,
            u8,
            (576, PAGE, pages),
            (prow, ppage),
            (64, HALF, 1),
            sw.CU_TENSOR_MAP_SWIZZLE_64B,
            l2,
        )
        scale_log2 = scale * Float32(LOG2_E)
        self.kernel(
            qlat,
            qrope,
            khalf,
            kval,
            krope,
            blocks,
            lens,
            totals,
            workspace,
            words,
            wl,
            scale_log2,
        ).launch(
            grid=self.grid,
            block=(NT, 1, 1),
            cluster=(2, 1, 1),
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def local_bound(self, g):
        """Local keys of this rank among global positions [0, g)."""
        world, rank, inter = self.cp
        result = g
        if cutlass.const_expr(world > 1):
            cycle = world * inter
            full = g // cycle
            tail = g - full * cycle - rank * inter
            tail = cutlass.max(tail, Int32(0))
            tail = cutlass.min(tail, Int32(inter))
            result = full * inter + tail
        return result

    @cute.jit
    def item(self, wl, lens, totals, c, n):
        """Item n of cluster c: (request, split, q tile, local length, total, first
        page, pages); pages 0 for an empty or absent slot. Under LPC, ``wl`` is the
        shared LPC table and the split is the chunk's partial slot."""
        if cutlass.const_expr(self.lpc):
            return self.lpc_item(wl, lens, totals, c, n)
        idx = c + n * self.pairs
        word = Int32(WLIST_EMPTY_WORD)
        if idx < self.slots:
            word = Int32(wl[idx])
        qt = word & 255
        rq = (word >> 8) & 255
        s = (word >> 16) & 65535
        npages = Int32(0)
        L = Int32(0)
        G = Int32(0)
        page0 = Int32(0)
        if s != 32767:
            L = lens[rq]
            G = totals[rq]
            T = (L + 127) // 128
            per = long_per(T, self.split, self.long)
            k0 = s * per
            cnt = cutlass.max(cutlass.min(T, k0 + per) - k0, Int32(0))
            page0 = 2 * k0
            if cnt > 0:
                npages = cutlass.min(2 * cnt, (L + PAGE - 1) // PAGE - page0)
        return rq, s, qt, L, G, page0, npages

    @cute.jit
    def lpc_item(self, table, lens, totals, c, n):
        """Chunk n of cluster c's group (c // items_per), item stream c % items_per."""
        g = c // self.items_per
        rq, slot, page0, npages = lpc_chunk(table, lens, self.reqs, g, n)
        if g >= self.groups:
            npages = Int32(0)
        L = Int32(0)
        G = Int32(0)
        if npages > 0:
            L = lens[rq]
            G = totals[rq]
        return rq, slot, c % self.items_per, L, G, page0, npages

    @cute.jit
    def store_part_shfl(
        self,
        o,
        inv,
        part_base,
        rbase,
        tile,
        quarter,
        lane,
        dh,
        g,
        part: cutlass.Constexpr,
        s,
    ):
        """32 columns (``part``) of the warp's 32 rows: an 8 x 8 butterfly transpose of
        16 B chunks within each 8 lanes, then 8 stores of 4 rows x 128 B (lane l: chunk
        l % 8 of row 8 (l // 8) + j)."""
        v = [o[e] * inv for e in range(32)]
        i8 = lane % 8
        for bit in cutlass.range_constexpr(3):
            b = 1 << bit
            hi = (i8 & b) != 0
            for j0 in cutlass.range_constexpr(8):
                if cutlass.const_expr(j0 & b == 0):
                    j1 = j0 ^ b
                    for e in cutlass.range_constexpr(4):
                        a0, a1 = v[4 * j0 + e], v[4 * j1 + e]
                        recv = cute.arch.shuffle_sync_bfly(
                            cutlass.select_(hi, a0, a1), offset=b
                        )
                        v[4 * j0 + e] = cutlass.select_(hi, recv, a0)
                        v[4 * j1 + e] = cutlass.select_(hi, a1, recv)
        if cutlass.const_expr("noepi" not in DEBUG):
            col = Int64(256 * dh + 128 * g + 32 * part) * 4 + Int64(16 * i8)
            for j in cutlass.range_constexpr(8):
                pr = tile * ROWS + 32 * quarter + 8 * (lane // 8) + j
                if pr < self.R:
                    ro = (rbase + Int64(pr)) * Int64(self.split) + Int64(s)
                    store_f32x4_v(
                        part_base + col + ro * 2048,
                        tuple(v[4 * j + e] for e in range(4)),
                    )

    @cute.jit
    def role(self, qt, rank):
        """(q tile, value half, odd) of item ``qt`` in cluster rank ``rank``: tile pair
        qt // 2's tile 2 (qt // 2) + rank, value half qt % 2; past the pairs, the odd
        item: the last tile, value half ``rank``."""
        tile = 2 * (qt // 2) + rank
        dh = qt % 2
        odd = Int32(0)
        if cutlass.const_expr(self.odd == 1):  # noqa: SIM102
            if qt >= 2 * self.even:
                tile = Int32(self.tiles - 1)
                dh = rank
                odd = Int32(1)
        return tile, dh, odd

    @cute.jit
    def arm_x(self, words, cta, lane):
        chunks: cutlass.Constexpr = self.x_rows * self.x_columns // 8
        ctas: cutlass.Constexpr = self.grid[0]
        per: cutlass.Constexpr = -(-chunks // ctas)
        lo = Int64(Uint32(words[2])) | Int64(Uint32(words[3])) << 32
        pitch = Int64(words[4])
        for j in cutlass.range(lane, per, 32, unroll=1):
            i = cta * per + j
            if (i < chunks) & (lo != 0):  # (core_q.verify_table: no x words, no re-arm)
                row, col = i // (self.x_columns // 8), i % (self.x_columns // 8)
                ptr = cute.make_ptr(
                    Uint32,
                    lo + Int64(row) * pitch + Int64(col) * 16,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                store_words(ptr, (Uint32(X_SENT),) * 4)

    @cute.kernel
    def kernel(
        self,
        qlat: tensormap.GridMap,
        qrope: tensormap.GridMap,
        khalf: tensormap.GridMap,
        kval: tensormap.GridMap,
        krope: tensormap.GridMap,
        blocks: cute.Tensor,
        lens: cute.Tensor,
        totals: cute.Tensor,
        workspace: cute.Tensor,
        words: cute.Tensor,
        wl: cute.Tensor,
        scale_log2: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        bx, _, _ = cute.arch.block_idx()
        rank = bx % 2  # cluster rank: 0 issues the pair's MMAs and copies
        c = bx // 2

        smem = cutlass.utils.SmemAllocator()
        data = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(DATA), 1024)
        bars = smem.allocate_tensor(Int64, cute.make_layout(NBARS), 8)
        xch = smem.allocate_tensor(Float32, cute.make_layout((2 + 2 * NWG) * ROWS), 16)
        tslot = smem.allocate_tensor(Uint32, cute.make_layout(1), 4)
        if cutlass.const_expr(self.lpc):
            ltab = smem.allocate_tensor(Int32, cute.make_layout(self.reqs + 2), 16)
        base = Uint32(data.iterator.toint())
        qs = base
        rs = qs + QSTAGE
        vs = rs + NRD * RSLOT
        bar0 = Uint32(bars.iterator.toint())
        rfull, rempty = bar0, bar0 + 8 * NRD
        vfull = bar0 + 16 * NRD
        vempty = vfull + 8 * NVD
        sfree = vempty + 8 * NVD  # 2 (leader: both CTAs loaded S; count 8)
        pfull = sfree + 16  # NP (leader: both CTAs' P stored / page ready; count 8)
        pempty = pfull + 8 * NP  # NP (PV commits, both CTAs)
        ofinal = pempty + 8 * NP  # an item's last PV completed (both CTAs)
        ofree = ofinal + 8  # leader: both CTAs' epilogues read O (count 16)
        qsmem, qvalid = ofree + 8, ofree + 16  # this CTA's Q tile landed in the staging
        qboth = ofree + 24  # leader: both CTAs' Q staged (count 2)
        qstfree = ofree + 32  # the leader's Q copies read both stagings (both CTAs)
        mready = ofree + 40  # 4 (per TMEM lane quarter): the running max of a page
        sfull = mready + 64  # NWG: QK commits
        xbase = Uint32(xch.iterator.toint())
        canary = Uint32(0)
        if cutlass.const_expr("smemchk" in DEBUG):  # a 4 KB per-CTA canary at the top
            ctensor = smem.allocate_tensor(Uint32, cute.make_layout(1024), 16)
            canary = Uint32(ctensor.iterator.toint())
            if tid < 256:
                for i in cutlass.range_constexpr(4):
                    st_shared_u32(
                        canary + 16 * tid + 4 * i, Uint32(bx * 4096 + 4 * tid + i)
                    )
            if tid == 0:
                progress(words, bx, 27, Int64(smem_reg(0)), tid)
                progress(words, bx, 28, Int64(smem_reg(1)), tid)
                progress(words, bx, 29, Int64(canary - base) + Int64(4096), tid)
                progress(words, bx, 30, Int64(base), tid)

        if tid == 0:
            for i in cutlass.range_constexpr(NRD):
                mbar_init(rfull + 8 * i, 1)
                mbar_init(rempty + 8 * i, 1)
            for i in cutlass.range_constexpr(NVD):
                mbar_init(vfull + 8 * i, 1)
                mbar_init(vempty + 8 * i, 1)
            for i in cutlass.range_constexpr(2):
                mbar_init(sfree + 8 * i, 8)
            for i in cutlass.range_constexpr(NP):
                mbar_init(pfull + 8 * i, 8)
                mbar_init(pempty + 8 * i, 1)
            mbar_init(ofinal, 1)
            mbar_init(ofree, 16)
            mbar_init(qsmem, 1)
            mbar_init(qboth, 2)
            mbar_init(qstfree, 1)
            for i in cutlass.range_constexpr(8):  # per lane quarter x page parity
                mbar_init(mready + 8 * i, 1)
            for i in cutlass.range_constexpr(NWG):
                mbar_init(sfull + 8 * i, 1)
            init_valid(qvalid)
            cute.arch.mbarrier_init_fence()
        if warp == W_MMA:
            cute.arch.alloc_tmem(
                TMEM_COLS, tslot.iterator, is_two_cta=True, arch="sm_107"
            )
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
        fence_tmem(before=True)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        fence_tmem()
        tmem = tslot[0]
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr("prog" in DEBUG):
            progress(words, bx, 16, globaltimer(), tid)
        src = wl  # the work list, or under LPC the shared LPC table
        if cutlass.const_expr(self.lpc):
            table = Uint32(ltab.iterator.toint())
            if warp == 0:
                lpc_table(lens, self.reqs, self.groups, self.split, table, lane)
            cute.arch.sync_threads()
            src = table

        if warp < 4 * NWG:
            self.softmax(
                tmem,
                xbase,
                sfull,
                sfree,
                pfull,
                pempty,
                ofinal,
                ofree,
                mready,
                workspace,
                scale_log2,
                rank,
                c,
                warp,
                lane,
                src,
                lens,
                totals,
                words,
            )
        if (warp == W_MMA) & (rank == 0):
            self.issue_qks(
                tmem,
                qs,
                rs,
                rfull,
                rempty,
                sfull,
                sfree,
                qboth,
                qstfree,
                lane,
                c,
                src,
                lens,
                totals,
                words,
                bx,
            )
        if warp == W_PV:  # noqa: SIM102 (only the pair leader issues the PVs)
            if rank == 0:
                self.issue_pvs(
                    tmem,
                    vs,
                    vfull,
                    vempty,
                    pfull,
                    pempty,
                    ofinal,
                    ofree,
                    lane,
                    c,
                    src,
                    lens,
                    totals,
                    words,
                    bx,
                )
        if (warp == W_K0) | (warp == W_K0 + 1):
            self.load_k(
                warp,
                qs,
                rs,
                vs,
                rfull,
                rempty,
                vfull,
                vempty,
                pempty,
                sfree,
                qvalid,
                qboth,
                qstfree,
                qlat,
                qrope,
                khalf,
                kval,
                krope,
                blocks,
                rank,
                lane,
                c,
                src,
                lens,
                totals,
                words,
            )
            if warp == W_K0 + 1:
                self.arm_x(words, bx, lane)
                cute.arch.griddepcontrol_launch_dependents()

        fence_tmem(before=True)
        # neither CTA leaves while the other may still signal it
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if cutlass.const_expr("prog" in DEBUG):
            progress(words, bx, 17, globaltimer(), tid)
        if cutlass.const_expr("smemchk" in DEBUG):  # noqa: SIM102
            if tid < 256:
                bad = Int32(0)
                for i in cutlass.range_constexpr(4):
                    v = ld_shared_u32(canary + 16 * tid + 4 * i)
                    if v != Uint32(bx * 4096 + 4 * tid + i):
                        bad = bad + 1
                for k in cutlass.range_constexpr(5):
                    bad = bad + cute.arch.shuffle_sync_bfly(bad, offset=1 << k)
                progress(words, bx, 40 + warp, Int64(bad), lane)
        if warp == W_MMA:
            fence_tmem()
            dealloc_tmem2(tmem, Uint32(TMEM_COLS))

    @cute.jit
    def load_k(
        self,
        warp,
        qs,
        rs,
        vs,
        rfull,
        rempty,
        vfull,
        vempty,
        pempty,
        sfree,
        qvalid,
        qboth,
        qstfree,
        qlat,
        qrope,
        khalf,
        kval,
        krope,
        blocks,
        rank,
        lane,
        c,
        wl,
        lens,
        totals,
        words,
    ):
        """This CTA's QK half (32 keys, 576 dims; warp W_K0, which also loads each
        item's Q tile into the staging once the leader's copies of the previous one have
        read it) or value block (128 columns, 64 keys) of every page of every item, into
        rings that run on across items."""
        ldesc = tensormap.map_address(qlat)
        qrdesc = tensormap.map_address(qrope)
        phase = Int32(0)
        ni = Int32(0)
        kdesc = tensormap.map_address(khalf)
        vdesc = tensormap.map_address(kval)
        rdesc = tensormap.map_address(krope)
        if lane == 0:
            tensormap.prefetch_map(kdesc)
            tensormap.prefetch_map(vdesc)
            tensormap.prefetch_map(rdesc)
        jg = Int32(0)
        vj = Int32(0)  # value slots used (an odd item's pages take two)
        for n in cutlass.range(self.rounds, unroll=1):
            rq, s, qt, L, G, page0, npages = self.item(wl, lens, totals, c, n)
            tile, dh, odd = self.role(qt, rank)
            vb = 2 * dh + rank  # this CTA's value block (odd: blocks rank, 2 + rank)
            if odd != 0:
                vb = rank
            if (
                (warp == W_K0)
                & (npages > 0)
                & ((ni == 0) | cutlass.const_expr("noq" not in DEBUG))
            ):
                if ni > 0:
                    wait_bar(qstfree, (ni - 1) % 2)
                if lane == 0:
                    armed = Int32(1)
                    while armed != 0:
                        expect_valid(qvalid, ROWS * 576)
                        for kb in cutlass.range_constexpr(4):
                            tma_valid3(
                                qs + kb * QBLOCK,
                                ldesc,
                                128 * kb,
                                tile * ROWS,
                                rq,
                                qvalid,
                            )
                        tma_valid3(
                            qs + 4 * QBLOCK, qrdesc, 512, tile * ROWS, rq, qvalid
                        )
                        armed = wait_valid(qvalid, phase)
                        phase = phase ^ 1
                    leader_arrive(qboth)
                ni = ni + 1
            mine = Int32(0)
            for j in cutlass.range(npages, unroll=1):
                if j % 32 == 0:
                    mine = Int32(0)
                    if j + lane < npages:
                        mine = blocks[rq, page0 + j + lane]
                page = cute.arch.shuffle_sync(mine, j % 32)
                jj = jg + j
                if warp == W_K0:
                    slot = jj % NRD
                    if jj >= NRD:
                        wait_bar(rempty + 8 * slot, (jj // NRD - 1) % 2)
                    stamp(stamp_base(words, bx_of(rank, c)), 11, jj, lane)
                    if lane == 0:
                        dst = rs + slot * RSLOT
                        if rank == 0:
                            arm_tx(rfull + 8 * slot, 2 * RSLOT)
                        lead = mapa(rfull + 8 * slot, 0)
                        tma4_2(dst, kdesc, 0, HALF * rank, 0, page, lead)
                        tma3_2(dst + 4 * HBLOCK, rdesc, 512, HALF * rank, page, lead)
                else:
                    for h in cutlass.range(1 + odd, unroll=1):
                        v = vj + (1 + odd) * j + h
                        slot = v % NVD
                        if v >= NVD:
                            wait_bar(vempty + 8 * slot, (v // NVD - 1) % 2)
                        stamp(stamp_base(words, bx_of(rank, c)), 14, jj, lane)
                        if lane == 0:
                            dst = vs + slot * VSLOT
                            if rank == 0:
                                arm_tx(vfull + 8 * slot, 2 * VSLOT)
                            tma4_2(
                                dst,
                                vdesc,
                                0,
                                0,
                                vb + 2 * h,
                                page,
                                mapa(vfull + 8 * slot, 0),
                            )
            jg = jg + npages
            vj = vj + (1 + odd) * npages
        # drain: the pair's last commits (multicast) onto this CTA's barriers landed
        if warp == W_K0:
            if ni > 0:
                wait_bar(qstfree, (ni - 1) % 2)
            for i in cutlass.range_constexpr(NRD):
                last = jg - 1 - (jg - 1 - i) % NRD
                if (jg > i) & (last >= 0):
                    wait_bar(rempty + 8 * i, (last // NRD) % 2)
        else:
            for i in cutlass.range_constexpr(NVD):
                last = vj - 1 - (vj - 1 - i) % NVD
                if (vj > i) & (last >= 0):
                    wait_bar(vempty + 8 * i, (last // NVD) % 2)
            for i in cutlass.range_constexpr(NP):
                last = jg - 1 - (jg - 1 - i) % NP
                if (jg > i) & (last >= 0):
                    wait_bar(pempty + 8 * i, (last // NP) % 2)
            if rank == 0:
                for i in cutlass.range_constexpr(2):
                    last = jg - 1 - (jg - 1 - i) % 2
                    if (jg > i) & (last >= 0):
                        wait_bar(sfree + 8 * i, (last // 2) % 2)

    @cute.jit
    def issue_qks(
        self,
        tmem,
        qs,
        rs,
        rfull,
        rempty,
        sfull,
        sfree,
        qboth,
        qstfree,
        lane,
        c,
        wl,
        lens,
        totals,
        words,
        cta,
    ):
        """The leader's QK warp: per item, both CTAs' Q staging -> TMEM (after the
        previous item's QKs in issue order), then QK(j) once both halves of page j have
        landed and both CTAs have loaded S(j - 2)."""
        jg = Int32(0)
        ni = Int32(0)
        qd = sdesc(qs, 16, 1024, SW128)
        sbase = stamp_base(words, cta)
        for n in cutlass.range(self.rounds, unroll=1):
            rq, s, qt, L, G, page0, npages = self.item(wl, lens, totals, c, n)
            if npages > 0:
                if (ni == 0) | cutlass.const_expr("noq" not in DEBUG):
                    wait_bar(qboth, ni % 2)
                    stamp(sbase, 15, ni, lane)
                    fence_tmem()
                    cp_q2(tmem + T_Q, qd)
                    commit_pair(qstfree)
                for j in cutlass.range(npages, unroll=1):
                    jj = jg + j
                    st = jj % 2
                    rsl = jj % NRD
                    wait_bar(rfull + 8 * rsl, (jj // NRD) % 2)
                    stamp(sbase, 13, jj, lane)
                    if jj >= 2:
                        wait_bar(sfree + 8 * st, (jj // 2 - 1) % 2)
                    stamp(sbase, 0, jj, lane)
                    fence_tmem()
                    ra = rs + rsl * RSLOT
                    qk_pair(
                        tmem + T_S + 64 * st,
                        tmem + T_Q,
                        sdesc(ra, 16, 1024, SW128),
                        sdesc(ra + 4 * HBLOCK, 16, 512, SW64),
                        QK2_INSTR,
                        sfull + 8 * (jj % NWG),
                        rempty + 8 * rsl,
                    )
                    stamp(sbase, 1, jj, lane)
                jg = jg + npages
                ni = ni + 1
        if ni > 0:  # the peer's last Q arrival and O releases are consumed elsewhere
            pass

    @cute.jit
    def issue_pvs(
        self,
        tmem,
        vs,
        vfull,
        vempty,
        pfull,
        pempty,
        ofinal,
        ofree,
        lane,
        c,
        wl,
        lens,
        totals,
        words,
        cta,
    ):
        """The leader's PV warp: PV(j) once both CTAs' P(j) are stored and both value
        blocks have landed; an item's first PV also waits until both CTAs' epilogues
        have read the previous item's O; its last PV's commit reports O."""
        jg = Int32(0)
        vj = Int32(0)  # value slots used (an odd item's pages take two)
        ni = Int32(0)
        sbase = stamp_base(words, cta)
        for n in cutlass.range(self.rounds, unroll=1):
            rq, s, qt, L, G, page0, npages = self.item(wl, lens, totals, c, n)
            if npages > 0:
                tile, dh, odd = self.role(qt, Int32(0))
                for j in cutlass.range(npages, unroll=1):
                    jj = jg + j
                    ps = jj % NP
                    v0 = vj + (1 + odd) * j
                    vsl = v0 % NVD
                    vsl2 = (v0 + odd) % NVD
                    wait_bar(pfull + 8 * ps, (jj // NP) % 2)
                    wait_bar(vfull + 8 * vsl, (v0 // NVD) % 2)
                    if odd != 0:
                        wait_bar(vfull + 8 * vsl2, ((v0 + 1) // NVD) % 2)
                    stamp(sbase, 2, jj, lane)
                    if (j == 0) & (ni > 0):
                        wait_bar(ofree, (ni - 1) % 2)
                        stamp(sbase, 16, ni, lane)
                    fence_tmem()
                    pv_page(
                        tmem + T_O,
                        tmem + T_P + 16 * ps,
                        sdesc(vs + vsl * VSLOT, VSLOT, 1024, SW128),
                        sdesc(vs + vsl2 * VSLOT, VSLOT, 1024, SW128),
                        PV2_INSTR,
                        j > 0,
                        pempty + 8 * ps,
                        vempty + 8 * vsl,
                        vempty + 8 * vsl2,
                        ofinal,
                        j == npages - 1,
                        odd,
                    )
                    stamp(sbase, 3, jj, lane)
                jg = jg + npages
                vj = vj + (1 + odd) * npages
                ni = ni + 1
        if ni > 0:  # drain: the last epilogues' O releases landed
            wait_bar(ofree, (ni - 1) % 2)

    @cute.jit
    def softmax(
        self,
        tmem,
        xbase,
        sfull,
        sfree,
        pfull,
        pempty,
        ofinal,
        ofree,
        mready,
        workspace,
        scale_log2,
        rank,
        c,
        warp,
        lane,
        wl,
        lens,
        totals,
        words,
    ):
        """Warp group g runs the pages jj = g, g + NWG, .. of the global page sequence
        (all 64 keys of its 32 rows per warp); per item the running max passes page to
        page through shared memory, each group sums its own pages, and the groups merge
        at the item's end; then the epilogue reads O out of TMEM (releasing it for the
        next item's first PV) and stores the FP32 partials and the base-2 LSEs."""
        H, S = self.H, self.split
        sbase = stamp_base(words, rank + 2 * c)
        slane = lane + 32 * quarter_of(warp)
        quarter, g = warp % 4, warp // 4
        r = 32 * quarter + lane  # tile row (TMEM lane)
        lanes = tmem + Uint32(32 * quarter << 16)
        idb = 1 + quarter
        mbar = mready + 8 * quarter
        chain = xbase  # 2 x ROWS: the running max hand-off
        merge = xbase + 4 * (2 * ROWS)  # NWG x 2 x ROWS: the groups' (max, sum)
        jg = Int32(0)
        mb = Int32(0)  # mready phases completed before this item
        ni = Int32(0)
        for n in cutlass.range(self.rounds, unroll=1):
            rq, s, qt, L, G, page0, npages = self.item(wl, lens, totals, c, n)
            if npages > 0:
                tile, dh, odd = self.role(qt, rank)
                p = tile * ROWS + r  # packed row
                live = p < self.R
                bound = L
                low = L
                if cutlass.const_expr(self.causal):
                    t = cutlass.min(p // H, Int32(self.q - 1))
                    bound = self.local_bound(G - (self.q - 1) + t)
                    low = self.local_bound(G - (self.q - 1) + (tile * ROWS) // H)
                my_max = -Float32.inf
                my_sum = Float32(0.0)
                j0 = (g - jg % NWG + NWG) % NWG
                for j in cutlass.range(j0, npages, NWG, unroll=1):
                    jj = jg + j
                    st = jj % 2
                    wait_bar(sfull + 8 * g, (jj // NWG) % 2)
                    stamp(sbase, 4, jj, slane)
                    fence_tmem()
                    sc = cute.make_rmem_tensor(64, Float32)
                    lo = ld_red32(lanes + T_S + 64 * st)
                    hi = ld_red32(lanes + T_S + 64 * st + 32)
                    fence_tmem(before=True)
                    cute.arch.sync_warp()
                    if (
                        lane == 0
                    ):  # S is in registers: QK(jj + 2) may overwrite the stage
                        leader_arrive(sfree + 8 * st)
                    for cc in cutlass.range_constexpr(32):
                        sc[cc] = lo[cc]
                        sc[32 + cc] = hi[cc]
                    m = fmax(lo[32], hi[32])
                    key0 = (page0 + j) * PAGE
                    if key0 + PAGE > low:
                        for cc in cutlass.range_constexpr(64):
                            sc[cc] = cutlass.select_(
                                key0 + cc < bound, sc[cc], Float32(MASKED)
                            )
                        m = sc[0]
                        for cc in cutlass.range_constexpr(1, 64):
                            m = fmax(m, sc[cc])
                    prev = -Float32.inf
                    if j >= 1:  # the max after the item's page j - 1
                        k = mb + j - 1
                        wait_bar(mbar + 32 * (k % 2), (k // 2) % 2)
                        prev = ld_shared_f(chain + 4 * (((mb + j - 1) % 2) * ROWS + r))
                    m = fmax(prev, m)
                    skip = (m - prev) * scale_log2 <= Float32(SKIP)
                    rescale = Int32(0)
                    if skip:
                        m = prev
                    elif j > 0:
                        rescale = Int32(1)
                    if j + 1 < npages:
                        st_shared_f(chain + 4 * (((mb + j) % 2) * ROWS + r), m)
                        cute.arch.sync_warp()
                        if lane == 0:
                            mbar_arrive(mbar + 32 * ((mb + j) % 2))
                    stamp(sbase, 6, jj, slane)
                    own = Float32(1.0)
                    if m != my_max:
                        own = ex2((my_max - m) * scale_log2)
                    bias = (Float32(0.0) - m) * scale_log2
                    acc = (Float32(0.0), Float32(0.0))
                    packed = []
                    for c4 in cutlass.range_constexpr(16):
                        x0, x1 = ffma2(sc[4 * c4], sc[4 * c4 + 1], scale_log2, bias)
                        x2, x3 = ffma2(sc[4 * c4 + 2], sc[4 * c4 + 3], scale_log2, bias)
                        x0, x1, x2, x3 = ex2(x0), ex2(x1), ex2(x2), ex2(x3)
                        acc = add2(acc, (x0, x1))
                        acc = add2(acc, (x2, x3))
                        packed.append(e4m3x4(x0, x1, x2, x3))
                    my_sum = my_sum * own
                    my_sum = acc[0] + acc[1] + my_sum
                    my_max = m
                    if cute.arch.vote_any_sync(rescale != 0):
                        corr = ex2((prev - m) * scale_log2)
                        if rescale == 0:
                            corr = Float32(1.0)
                        pq = (jj - 1) % NP
                        wait_bar(pempty + 8 * pq, ((jj - 1) // NP) % 2)
                        fence_tmem()
                        for part in cutlass.range_constexpr(8):
                            col = lanes + T_O + 32 * part
                            o = ld_tmem32(col)
                            st_tmem_f(col, tuple(o[e] * corr for e in range(32)))
                    ps = jj % NP
                    if jj >= NP:  # PV(jj - NP) has read the stage
                        wait_bar(pempty + 8 * ps, (jj // NP - 1) % 2)
                        fence_tmem()
                    st_tmem(lanes + T_P + 16 * ps, tuple(packed))
                    fence_tmem(before=True)
                    cute.arch.sync_warp()
                    if lane == 0:
                        leader_arrive(pfull + 8 * ps)
                    stamp(sbase, 9, jj, slane)
                # the groups' (max, sum) -> the row's max and sum (group order)
                bar_sync(idb, 32 * NWG)
                mine = merge + 4 * (g * 2 * ROWS)
                st_shared_f(mine + 4 * r, my_max)
                st_shared_f(mine + 4 * (ROWS + r), my_sum)
                bar_sync(idb, 32 * NWG)
                row_max = ld_shared_f(merge + 4 * r)
                for gg in cutlass.range_constexpr(1, NWG):
                    row_max = fmax(
                        row_max, ld_shared_f(merge + 4 * (gg * 2 * ROWS + r))
                    )
                total = Float32(0.0)
                for gg in cutlass.range_constexpr(NWG):
                    mg = ld_shared_f(merge + 4 * (gg * 2 * ROWS + r))
                    sg = ld_shared_f(merge + 4 * (gg * 2 * ROWS + ROWS + r))
                    wg = Float32(0.0)
                    if sg > Float32(0.0):
                        wg = sg * ex2((mg - row_max) * scale_log2)
                    total = total + wg
                # epilogue (groups 0-1, 2 threads per row: columns 128 g ..): O out of
                # TMEM first (then the next item's first PV may overwrite it), then the
                # stores
                if g < 2:
                    wait_bar(ofinal, ni % 2)
                    stamp(sbase, 17, ni, lane + 32 * warp)
                    fence_tmem()
                    inv = cute.arch.rcp_approx(total)
                    part_base = workspace.iterator.toint()
                    rbase = Int64(rq) * Int64(self.x_rows * H)
                    row_off = (rbase + Int64(p)) * Int64(S) + Int64(s)
                    dst = part_base + Int64(256 * dh + 128 * g) * 4 + row_off * 2048
                    for part in cutlass.range_constexpr(4):
                        o = ld_tmem32(lanes + T_O + 128 * g + 32 * part)
                        if cutlass.const_expr(EPI == "shfl"):
                            self.store_part_shfl(
                                o,
                                inv,
                                part_base,
                                rbase,
                                tile,
                                quarter,
                                lane,
                                dh,
                                g,
                                part,
                                s,
                            )
                        elif live & cutlass.const_expr("noepi" not in DEBUG):
                            for e8 in cutlass.range_constexpr(4):
                                vals = tuple(o[8 * e8 + e] * inv for e in range(8))
                                store_f32x8_v(dst + (32 * part + 8 * e8) * 4, vals)
                    fence_tmem(before=True)
                    cute.arch.sync_warp()
                    if (
                        lane == 0
                    ):  # O is out of TMEM: the next item's first PV may write it
                        leader_arrive(ofree)
                    stamp(sbase, 18, ni, lane + 32 * warp)
                    if live & (dh == 0) & (g == 0):
                        lse = (
                            cute.math.log2(total, fastmath=True) + scale_log2 * row_max
                        )
                        lse_base = (
                            part_base + Int64(self.reqs * self.x_rows * H * S * 512) * 4
                        )
                        store_f32(lse_base + row_off * 4, lse)
                    stamp(sbase, 19, ni, lane + 32 * warp)
                    if cutlass.const_expr("prog" in DEBUG):  # noqa: SIM102
                        if warp == 0:
                            slot = 20 + cutlass.min(ni, Int32(40))
                            progress(words, rank + 2 * c, slot, globaltimer(), lane)
                jg = jg + npages
                mb = mb + npages - 1
                ni = ni + 1


def compile_core(core, kv_spec):
    """cute.compile of a P2Core (no device context: CPU compiles work too)."""
    from cutlass.cute.runtime import make_fake_stream

    from ..common.host import fake

    shape, stride = kv_spec
    query = fake((core.reqs, core.q, core.H, 576), None, cutlass.Uint8)
    kv = fake(shape, stride, cutlass.Uint8)
    blocks = fake((core.reqs, cute.sym_int()), (cute.sym_int(), 1), cutlass.Int32, 4)
    lens = fake((core.reqs,), None, cutlass.Int32, 4)
    totals = fake((core.reqs,), None, cutlass.Int32, 4)
    work = fake((cute.sym_int(),), None, cutlass.Int8, 16)
    words = fake((cute.sym_int(),), None, cutlass.Int32, 4)
    wl = fake((cute.sym_int(),), None, cutlass.Int32, 4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = "--gpu-arch=sm_107a --enable-tvm-ffi --opt-level 2"
    return cute.compile(
        core,
        query,
        kv,
        blocks,
        lens,
        totals,
        work,
        words,
        wl,
        Float32(1.0),
        stream,
        options=options,
    )


@cache
def compile_p2(
    heads, q, page, split, cp, device, causal, x_rows, long, kv_spec, reqs, slots,
    pairs, lpc=False,
):  # fmt: skip
    assert page == PAGE, "DS-T2 reads 64-key pages"
    core = P2Core(
        heads, q, split, cp, causal, x_rows, long, slots, pairs, reqs=reqs, lpc=lpc
    )
    with torch.accelerator.device_index(device):
        return compile_core(core, kv_spec)


def ds_partials(
    query, cache_kv, block_table, length, total, scales, out, lse, workspace, split,
    cp, table, x_rows=8, causal=True, long=None, wlist=None, lpc=False,
):  # fmt: skip
    """FP32 split partials and base-2 split LSEs of the requests' rows into DcpValue's
    rows workspace (lc_partials' interface). ``wlist``: (words int32, slots) of the
    mla.wlist work list built for ``pairs`` = sms / 2 persistent pairs at this ``split``
    and ``long``. ``lpc``: dcp.lpc_table's partition over the launched ``split`` slots
    instead (``wlist`` unused but for its slot count)."""
    reqs, q, heads, _ = query.shape
    kv = cache_kv.view(torch.uint8)
    pairs = torch.cuda.get_device_properties(query.device).multi_processor_count // 2
    kernel = compile_p2(
        heads, q, cache_kv.shape[1], split, tuple(cp), query.device.index, causal,
        x_rows, tuple(long), (tuple(kv.shape), tuple(kv.stride())), reqs, wlist[1],
        pairs, lpc,
    )  # fmt: skip
    import tvm_ffi

    softmax, _ = scales
    with tvm_ffi.use_torch_stream():
        kernel(
            query.view(torch.uint8),
            kv,
            block_table,
            length,
            total,
            workspace.view(torch.int8),
            table,
            wlist[0],
            softmax,
        )
