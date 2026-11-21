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
"""specDSA multi-CTA exact top-512: the top-K stage of the tail-attention hybrid, one thread-block cluster per row.

Selects the exact top-512 of every row of the fp32 logits that the indexer kernel writes (``logits`` [B, npad], row
``r`` valid up to ``ctx[r]``) and writes the positions to ``out`` [B, 512] int32: local positions of the row in no
particular order and, for a row of at most 512 positions, ``0 .. n - 1`` followed by -1 (what the production top-K
writes). The selection is exact for every input: its values are the 512 largest of the row (the multiset
``torch.topk`` returns); which of the positions that share the 512th value are taken is not specified (the fast path
takes the smaller positions, the radix fallback the ones its CTAs reach first). NaN logits are outside the contract
(the indexer kernel never writes one): they rank by their bit pattern.

A cluster of ``g`` CTAs (grid ``g * B``, 512 threads each, no wait between clusters) serves a row, which is read once
with addresses that do not depend on the row length:

1. Sample windows (8 floats at stride 896, rows longer than 8192 positions only) give the threshold T1, the value
   whose rank in the row is expected to be 3.5 x 512, and the span [T1, sample max] that defines 1024 candidate bins.
   The CTA's 512-byte chunks of the row (interleaved over the cluster) stream through a cp.async ring in shared memory
   while the threshold is computed.
2. Every value >= T1 goes (position, value) to a warp-private candidate list of its CTA, and the lists are counted per
   bin.
3. One all-gather of the per-CTA suffix-count tables (u16 x 1024, a bulk copy through distributed shared memory
   completing an mbarrier) tells every CTA the bin B* that holds the 512th value, the number of candidates above it
   and its own output offset.
4. Every CTA writes its candidates above B* straight to ``out``; the members of B* go to the lead CTA of the row,
   which ranks them by (value desc, position asc) and writes the missing ones.

Every CTA of the cluster checks the certificate of this fast path from the same all-gathered data: at least 512
candidates, no list overflow, at most 512 members in B*. When it fails (the sample threshold was too high, massive
ties) the whole cluster runs an exact 4-level radix select over the row instead (8-bit digits, one 256-bin all-gather
per level, a write pass with a tie quota per CTA). A row of at most 8192 positions has no sample: every value is a
candidate and the same certificate decides (a row of at most 512 positions is the identity followed by -1).

A CTA holds ~170 KB of shared memory, so the clusters of a launch have to co-reside in one wave on the 148 SMs
(clusters of 8 / 6 / 5 / 4 CTAs: 15 / 22 / 26 / 33 per B200); ``topk_multicta_op`` picks ``g`` from the batch size (1
to 26 rows). The launch is a plain stream-ordered launch. With ``trigger`` every CTA lets the dependent grid launch
(``griddepcontrol.launch_dependents``) as soon as its stream phase is done, which is what the tail attention launched
with the programmatic attribute (``tail_op``, ``pdl``) needs to run its prologue while this kernel ranks the crossing
bin; the production kernels around it never trigger a dependent launch early.

With ``arm`` the CTA 0 of every row marks the 512 entries of the row's output -2 at its start (a written entry is
>= -1; every path writes every entry exactly once): the tail attention launched behind it with ``poll``
(``tail_attention``) reads its entries as soon as they exist instead of waiting for the completion of this grid.
The order that makes this safe: the CTA 0 arrives at the cluster barrier with release semantics right before its own
trigger, so its marks are in memory before the dependent grid can start, and the other CTAs store entries only after
the cluster wait that follows that arrival (no mark can land after an entry); the other CTAs arrive early and relaxed,
as without ``arm``. A short row (no stream phase) arrives with release right after the marks.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op

from . import megakernel_lib as L
from .config import DS_V4_FLASH as SPEC

__all__ = ["SUPPORTED_CLUSTER_SIZES", "TopkMultiCta", "compile_topk_multicta"]

K_TOP = SPEC.topk  # 512
MAX_LEN = SPEC.max_index_len  # 262144: longest row
# CTAs per row that are built and validated (see ``topk_multicta_op`` for the batch sizes they serve).
SUPPORTED_CLUSTER_SIZES = (4, 5, 6, 8)

NT = 512  # threads per CTA
NW = NT // 32  # warps per CTA
U = 4  # 512-byte chunks per warp in one ring batch
NBUF = 3  # ring slots (batches in flight)
CH = 128  # floats per chunk (512 B, one warp-wide float4 load)
SMP_MAX = 292  # sample windows at MAX_LEN (the production top-K samples n / 896)
WST = 896  # window stride (floats): the sample windows are at j * WST whatever the row length
AIM = 1792  # expected number of candidates (>= T1) of a row, 3.5 x K_TOP
NBIN = 1024  # candidate bins
BPT = NBIN // NT  # bins per thread
NSB = 2048  # sample bins (key >> 21)
LST_BYTES = 32768  # warp-private candidate lists of the CTA
LCAPW = LST_BYTES // 8 // NW  # candidates one warp can hold
MCAP = 512  # crossing-bin members ranked by counting (XCAP entries are buffered)
XCAP = 1024  # crossing-bin member buffer (lead CTA)
NEG_INF = float("-inf")
SLOT = 2064  # R1 message bytes per source CTA (2048 of u16 suffix counts + 16 header)

# shared memory layout (bytes)
OFF_BAR = 0
OFF_MISC = 64  # 256 ints
M_BSTAR, M_SMAX, M_OVF, M_T1KEY, M_HAVE = 1, 2, 3, 8, 9
# fallback: running counters, crossing digit, strictly-above count, per-source above counts (10..17)
M_CABOVE, M_CTIE, M_FDIG, M_FABOVE, M_FA = 4, 5, 6, 7, 10
M_WCNT, M_WA, M_WM, M_WTOT = 32, 64, 96, 128  # per-warp counters (32 words each)
OFF_SHIST = OFF_MISC + 1024  # 2048 sample bins + 32 pad words
OFF_HIST = OFF_SHIST + (NSB + 32) * 4
OFF_LST = OFF_HIST + NBIN * 4  # warp-private lists
OFF_RX = OFF_LST + LST_BYTES
OFF_XIN = OFF_RX + 8 * SLOT
OFF_RING = OFF_XIN + XCAP * 8
ARENA_BYTES = OFF_RING + NBUF * U * NW * 512 + 1024
assert ARENA_BYTES <= 227 * 1024


@dsl_user_op
def lds_u64(saddr, *, loc=None, ip=None):
    return Uint64(
        L._asm(T.i64(), [L._v(saddr, loc, ip)], "ld.shared.u64 $0, [$1];", "=l,r", loc, ip)
    )


@dsl_user_op
def sts_u64(saddr, val, *, loc=None, ip=None):
    L._asm(
        None, [L._v(saddr, loc, ip), L._v(val, loc, ip)], "st.shared.u64 [$0], $1;", "r,l", loc, ip
    )


@dsl_user_op
def atom_add_sh(saddr, val, *, loc=None, ip=None):
    return Int32(
        L._asm(
            T.i32(),
            [L._v(saddr, loc, ip), L._v(val, loc, ip)],
            "atom.shared.add.u32 $0, [$1], $2;",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def red_add_sh(saddr, val, *, loc=None, ip=None):
    L._asm(
        None,
        [L._v(saddr, loc, ip), L._v(val, loc, ip)],
        "red.shared.add.u32 [$0], $1;",
        "r,r",
        loc,
        ip,
    )


@dsl_user_op
def red_max_sh(saddr, val, *, loc=None, ip=None):
    L._asm(
        None,
        [L._v(saddr, loc, ip), L._v(val, loc, ip)],
        "red.shared.max.u32 [$0], $1;",
        "r,r",
        loc,
        ip,
    )


@dsl_user_op
def lds_v2(saddr, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm

    st = llvm.StructType.get_literal([T.i32(), T.i32()])
    r = L._asm(st, [L._v(saddr, loc, ip)], "ld.shared.v2.u32 {$0, $1}, [$2];", "=r,=r,r", loc, ip)
    return tuple(Int32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def sts_v2(saddr, a, b, *, loc=None, ip=None):
    L._asm(
        None,
        [L._v(saddr, loc, ip), L._v(a, loc, ip), L._v(b, loc, ip)],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def cpa_commit(*, loc=None, ip=None):
    L._asm(None, [], "cp.async.commit_group;", "", loc, ip)


@dsl_user_op
def cpa_wait(n, *, loc=None, ip=None):
    L._asm(None, [], f"cp.async.wait_group {n};", "", loc, ip)


@dsl_user_op
def st_g_i32(addr, val, *, loc=None, ip=None):
    L._asm(
        None, [L._v(addr, loc, ip), L._v(val, loc, ip)], "st.global.u32 [$0], $1;", "l,r", loc, ip
    )


@cute.jit
def _umin(a, b):
    r = a
    if b < a:
        r = b
    return r


@cute.jit
def _issue_batch(
    ring,
    kbat,
    slot,
    rowbase,
    nlim,
    d,
    warp,
    lane,
    G: cutlass.Constexpr,
    U: cutlass.Constexpr,
    NW: cutlass.Constexpr,
):
    """cp.async the U chunks of this warp for batch ``kbat`` into ring slot ``slot`` (16 B per lane and chunk;
    chunks at or beyond ``nlim`` are skipped) and commit the group."""
    for j in cutlass.range_constexpr(U):
        k = kbat * Int32(U) + Int32(j)
        c = d + Int32(G) * (warp + Int32(NW) * k)
        if c < nlim:
            L.cp_async16(
                ring
                + slot * Int32(U * NW * 512)
                + Int32(j * NW * 512)
                + warp * Int32(512)
                + lane * Int32(16),
                rowbase + Int64(c * Int32(CH) + lane * Int32(4)) * Int64(4),
            )
    cpa_commit()


@cute.jit
def _kth_set_bit(mask, t):
    """Index of the t-th (0-based) set bit of ``mask`` (t < popc(mask))."""
    pos = Int32(0)
    m = mask
    tt = t
    for w in cutlass.range_constexpr(5):
        wd = 16 >> w
        low = m & Int32((1 << wd) - 1)
        c = Int32(cute.arch.popc(low))
        if tt >= c:
            tt = tt - c
            m = (m >> Int32(wd)) & Int32((1 << (32 - wd)) - 1)
            pos = pos + Int32(wd)
        else:
            m = low
    return pos


@cute.jit
def _consume_batch(
    ring,
    kbat,
    slot,
    n_ok,
    d,
    warp,
    lane,
    t1f,
    sb,
    wcnt,
    G: cutlass.Constexpr,
    U: cutlass.Constexpr,
    NW: cutlass.Constexpr,
    LCAPW: cutlass.Constexpr,
):
    """Values >= T1 of one batch (read from its ring slot) -> the warp's private candidate list (position, value);
    returns the warp's new list length. Phase 1: per chunk one ballot of the lanes that hold a candidate among
    their 4 values (no branch). Phase 2: the flagged (chunk, lane) pairs of the whole batch are numbered by the
    ballots' popcounts and handed to lanes 0..31 one pair per lane (extra rounds if there are more than 32): the lane
    re-reads the 4 values of its pair from the ring, tests them, a warp scan gives the slots, the lane stores. The
    cost of the batch follows the number of flagged pairs, not the number of chunks that contain one."""
    bbase = ring + slot * Int32(U * NW * 512) + warp * Int32(512)
    lst = sb + Int32(OFF_LST) + warp * Int32(LCAPW * 8)
    mcs = cute.make_rmem_tensor((U,), Int32)
    cums = cute.make_rmem_tensor((U + 1,), Int32)
    cums[0] = Int32(0)
    for j in cutlass.range_constexpr(U):
        v = L.lds_v4(bbase + Int32(j * NW * 512) + lane * Int32(16))
        x0 = L.f32_of_u32(Uint32(v[0]))
        x1 = L.f32_of_u32(Uint32(v[1]))
        x2 = L.f32_of_u32(Uint32(v[2]))
        x3 = L.f32_of_u32(Uint32(v[3]))
        mx = cute.arch.fmax(cute.arch.fmax(x0, x1), cute.arch.fmax(x2, x3))
        pj = Int32(0)
        if mx >= t1f:
            pj = Int32(1)
        mj = Int32(cute.arch.vote_ballot_sync(pj != Int32(0)))
        mcs[j] = mj
        cums[j + 1] = cums[j] + Int32(cute.arch.popc(mj))
    nitems = cums[U]
    if nitems != Int32(0):
        rd = Int32(0)
        while rd < nitems:
            r = rd + lane
            act = r < nitems
            # chunk of item r: the number of chunk starts (j >= 1) that are <= r
            jj = Int32(0)
            for j in cutlass.range_constexpr(1, U):
                if cums[j] <= r:
                    jj = jj + Int32(1)
            msk = Int32(0)
            st_j = Int32(0)
            for j in cutlass.range_constexpr(U):
                if jj == Int32(j):
                    msk = mcs[j]
                    st_j = cums[j]
            gl = Int32(0)
            g0 = Int32(0)
            g1 = Int32(0)
            g2 = Int32(0)
            g3 = Int32(0)
            pb = Int32(0)
            vv0 = Int32(0)
            vv1 = Int32(0)
            vv2 = Int32(0)
            vv3 = Int32(0)
            if act:
                gl = _kth_set_bit(msk, r - st_j)
                v = L.lds_v4(bbase + jj * Int32(NW * 512) + gl * Int32(16))
                vv0 = v[0]
                vv1 = v[1]
                vv2 = v[2]
                vv3 = v[3]
                k = kbat * Int32(U) + jj
                c = d + Int32(G) * (warp + Int32(NW) * k)
                pb = c * Int32(CH) + gl * Int32(4)
                lim = n_ok - pb
                if L.f32_of_u32(Uint32(vv0)) >= t1f:
                    if lim > Int32(0):
                        g0 = Int32(1)
                if L.f32_of_u32(Uint32(vv1)) >= t1f:
                    if lim > Int32(1):
                        g1 = Int32(1)
                if L.f32_of_u32(Uint32(vv2)) >= t1f:
                    if lim > Int32(2):
                        g2 = Int32(1)
                if L.f32_of_u32(Uint32(vv3)) >= t1f:
                    if lim > Int32(3):
                        g3 = Int32(1)
            cl = g0 + g1 + g2 + g3
            incl = L.warp_incl_scan_add(cl, lane)
            tot = cute.arch.shuffle_sync(incl, Int32(31))
            slot_c = wcnt + incl - cl
            if g0 != Int32(0):
                if slot_c < Int32(LCAPW):
                    sts_u64(
                        lst + slot_c * Int32(8),
                        (Uint64(Uint32(pb)) << Uint64(32)) | Uint64(Uint32(vv0)),
                    )
                slot_c = slot_c + Int32(1)
            if g1 != Int32(0):
                if slot_c < Int32(LCAPW):
                    sts_u64(
                        lst + slot_c * Int32(8),
                        (Uint64(Uint32(pb + Int32(1))) << Uint64(32)) | Uint64(Uint32(vv1)),
                    )
                slot_c = slot_c + Int32(1)
            if g2 != Int32(0):
                if slot_c < Int32(LCAPW):
                    sts_u64(
                        lst + slot_c * Int32(8),
                        (Uint64(Uint32(pb + Int32(2))) << Uint64(32)) | Uint64(Uint32(vv2)),
                    )
                slot_c = slot_c + Int32(1)
            if g3 != Int32(0):
                if slot_c < Int32(LCAPW):
                    sts_u64(
                        lst + slot_c * Int32(8),
                        (Uint64(Uint32(pb + Int32(3))) << Uint64(32)) | Uint64(Uint32(vv3)),
                    )
                slot_c = slot_c + Int32(1)
            wcnt = wcnt + tot
            rd = rd + Int32(32)
    return wcnt


@cute.jit
def _level_batch(
    ring,
    kbat,
    slot,
    n_ok,
    d,
    warp,
    lane,
    pfx,
    shp,
    dsh,
    sb,
    G: cutlass.Constexpr,
    U: cutlass.Constexpr,
    NW: cutlass.Constexpr,
):
    """Exact radix fallback, one level: histogram (256 bins, smem) of the 8-bit digit at bit ``dsh`` of the keys of this
    batch whose bits above ``dsh + 8`` equal ``pfx`` (``shp`` = number of low bits below the prefix: 32 - 8 * level)."""
    bbase = ring + slot * Int32(U * NW * 512) + warp * Int32(512) + lane * Int32(16)
    for j in cutlass.range_constexpr(U):
        k = kbat * Int32(U) + Int32(j)
        c = d + Int32(G) * (warp + Int32(NW) * k)
        lim = n_ok - (c * Int32(CH) + lane * Int32(4))
        v = L.lds_v4(bbase + Int32(j * NW * 512))
        for e in cutlass.range_constexpr(4):
            if Int32(e) < lim:
                kx = L.fkey_bits(Uint32(v[e]))
                if ((kx >> Uint32(1)) >> Uint32(shp - Int32(1))) == pfx:
                    red_add_sh(
                        sb + Int32(OFF_SHIST) + Int32((kx >> Uint32(dsh)) & Uint32(255)) * Int32(4),
                        Int32(1),
                    )


@cute.jit
def _final_batch(
    ring,
    kbat,
    slot,
    n_ok,
    d,
    warp,
    lane,
    kth,
    out_row,
    a_base,
    t_base,
    t_quota,
    sb,
    G: cutlass.Constexpr,
    U: cutlass.Constexpr,
    NW: cutlass.Constexpr,
):
    """Exact radix fallback, write pass: the elements whose key is above ``kth`` go to ``a_base + running count``, the
    first ``t_quota`` elements equal to ``kth`` of this CTA to ``t_base + ordinal``. The running counters live in smem
    (M_CABOVE / M_CTIE words); one atomic per candidate chunk step."""
    misc = sb + Int32(OFF_MISC)
    lm = Int32(cute.arch.lanemask_lt())
    bbase = ring + slot * Int32(U * NW * 512) + warp * Int32(512) + lane * Int32(16)
    for j in cutlass.range_constexpr(U):
        k = kbat * Int32(U) + Int32(j)
        c = d + Int32(G) * (warp + Int32(NW) * k)
        pbase = c * Int32(CH) + lane * Int32(4)
        lim = n_ok - pbase
        v = L.lds_v4(bbase + Int32(j * NW * 512))
        for e in cutlass.range_constexpr(4):
            pa = Int32(0)
            pt = Int32(0)
            if Int32(e) < lim:
                kx = L.fkey_bits(Uint32(v[e]))
                if kx > kth:
                    pa = Int32(1)
                if kx == kth:
                    pt = Int32(1)
            ma = Int32(cute.arch.vote_ballot_sync(pa != Int32(0)))
            mt = Int32(cute.arch.vote_ballot_sync(pt != Int32(0)))
            ba = Int32(0)
            bt = Int32(0)
            if lane == Int32(0):
                if ma != Int32(0):
                    ba = atom_add_sh(misc + Int32(4 * M_CABOVE), Int32(cute.arch.popc(ma)))
                if mt != Int32(0):
                    bt = atom_add_sh(misc + Int32(4 * M_CTIE), Int32(cute.arch.popc(mt)))
            ba = cute.arch.shuffle_sync(ba, Int32(0))
            bt = cute.arch.shuffle_sync(bt, Int32(0))
            if pa != Int32(0):
                st_g_i32(
                    out_row + Int64(a_base + ba + Int32(cute.arch.popc(ma & lm))) * Int64(4),
                    pbase + Int32(e),
                )
            if pt != Int32(0):
                ordn = bt + Int32(cute.arch.popc(mt & lm))
                if ordn < t_quota:
                    st_g_i32(out_row + Int64(t_base + ordn) * Int64(4), pbase + Int32(e))


class TopkMultiCta:
    """The multi-CTA top-512 kernel (see the module docstring); ``compile_topk_multicta`` compiles it.

    Args:
        g: CTAs (the cluster) per row, one of ``SUPPORTED_CLUSTER_SIZES``; another value raises ValueError.
        force_fallback: take the exact radix path on every row (a test hook: it is a different kernel, the default one
            takes it only when the certificate of the fast path fails).
        trigger: launch the dependent grid once the stream phase of the CTA is done (programmatic dependent launch).
            A CTA that takes another path (a short row) triggers when it exits.
        arm: mark the 512 output entries of every row -2 ("not written yet") at the start of the kernel, for a tail
            attention that polls them (needs ``trigger``).

    The compiled function takes, in order: ``logits`` float32 [B, npad] (unit column stride, row stride a multiple of
    128, 16-byte aligned base), ``ctx`` int32 [B] (valid length of each row) and ``out`` int32 [B, 512].
    """

    def __init__(
        self, g: int, force_fallback: bool = False, trigger: bool = False, arm: bool = False
    ):
        if g not in SUPPORTED_CLUSTER_SIZES:
            raise ValueError(
                f"cluster size {g} not supported (built and validated for {SUPPORTED_CLUSTER_SIZES})"
            )
        self.g = g
        self.force_fallback = force_fallback
        if arm and not trigger:
            raise ValueError(
                "arm needs trigger: the CTA 0 arrives at the cluster barrier before its trigger"
            )
        self.trigger = trigger
        self.arm = arm

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        ctx: cute.Tensor,
        out: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        b = logits.shape[0]
        self.kernel(logits, ctx, out).launch(
            grid=(b * self.g, 1, 1),
            block=[NT, 1, 1],
            cluster=(self.g, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        ctx: cute.Tensor,
        out: cute.Tensor,
    ):
        G = self.g
        NR = (SMP_MAX + NT - 1) // NT  # sample rounds per thread
        bidx, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        row = bidx // Int32(G)
        d = bidx % Int32(G)
        smem = cutlass.utils.SmemAllocator()
        arena = smem.allocate(ARENA_BYTES, byte_alignment=1024)
        sb = Int32(arena.toint())
        b_r1 = sb + Int32(OFF_BAR)
        b_r2 = sb + Int32(OFF_BAR + 8)
        misc = sb + Int32(OFF_MISC)

        # mbarrier init (fence before any load of this thread is outstanding)
        if warp == Int32(NW - 1):
            if lane == Int32(0):
                L.mbar_init(b_r1, Int32(1))
                L.mbar_init(b_r2, Int32(1))
                L.fence_mbar_init_cluster()
        # zero state (the barrier that publishes it sits before the first atomics)
        zi = tid
        while zi < Int32((OFF_LST - OFF_MISC) // 16):
            L.sts_v4(misc + zi * Int32(16), Int32(0), Int32(0), Int32(0), Int32(0))
            zi = zi + Int32(NT)
        # ---- t = 0: every global load whose address does not depend on n
        if cutlass.const_expr(self.arm):
            # the polling tail attention behind this kernel starts after every CTA has run its stream phase: mark the
            # row's entries "not written yet" (-2) now (released by the CTA 0's arrive right before its trigger, see the
            # module docstring)
            if d == Int32(0):
                st_g_i32(
                    out.iterator.toint() + (Int64(row) * Int64(K_TOP) + Int64(tid)) * Int64(4),
                    Int32(-2),
                )
        n = L.ld_cg_i32(ctx.iterator.toint() + Int64(row) * Int64(4))
        npad = Int32(logits.layout.stride[0])
        rowbase = logits.iterator.toint() + Int64(row) * Int64(logits.layout.stride[0]) * Int64(4)
        ncap = npad // Int32(CH)
        sv = cute.make_rmem_tensor((8 * NR,), Float32)
        for e in cutlass.range_constexpr(8 * NR):
            sv[e] = Float32(NEG_INF)
        for r in cutlass.range_constexpr(NR):
            jw0 = tid + Int32(NT * r)
            if jw0 < Int32(SMP_MAX):
                if jw0 * Int32(WST) + Int32(8) <= npad:
                    wa = rowbase + Int64(jw0 * Int32(WST)) * Int64(4)
                    a4 = L.ld_cg_f32x4(wa)
                    b4 = L.ld_cg_f32x4(wa + Int64(16))
                    for e in cutlass.range_constexpr(4):
                        sv[8 * r + e] = a4[e]
                        sv[8 * r + 4 + e] = b4[e]
        ring = sb + Int32(OFF_RING)

        n_ok = n
        if n > Int32(MAX_LEN):
            n_ok = Int32(MAX_LEN)
        if n_ok > npad:  # a length beyond the row buffer must not read past the row
            n_ok = npad
        if n_ok > Int32(K_TOP):
            # sample: the windows j * WST (8 floats each) that lie inside the row, at most SMP_MAX;
            # TGT = the rank in the sample that corresponds to AIM candidates of the row (aim 1792 =
            # 3.5 x K). No sample for rows <= 8192 (everything is a candidate)
            SMP = Int32(0)
            TGT = Int32(0)
            if n_ok > Int32(8192):
                SMP = (n_ok - Int32(8)) // Int32(WST) + Int32(1)
                if SMP > Int32(SMP_MAX):
                    SMP = Int32(SMP_MAX)
                TGT = Int32(Float32(AIM * 8) * Float32(SMP) * cute.arch.rcp_approx(Float32(n_ok)))
                if TGT < Int32(1):
                    TGT = Int32(1)
            # ---------------- sample -> T1, span ----------------
            L.bar(1, NT)
            if cutlass.const_expr(self.arm):
                if d != Int32(0):
                    L.cluster_arrive_relaxed()
            else:
                L.cluster_arrive_relaxed()
            smx = Uint32(0)
            for r in cutlass.range_constexpr(NR):
                jw = tid + Int32(NT * r)
                if jw < SMP:
                    for e in cutlass.range_constexpr(8):
                        kk = L.fkey(sv[8 * r + e])
                        bq = Int32(kk >> Uint32(21))
                        red_add_sh(
                            sb + Int32(OFF_SHIST) + (bq + (bq >> Int32(6))) * Int32(4), Int32(1)
                        )
                        if kk > smx:
                            smx = kk
            wmx = L.warp_max_u32(smx)
            if lane == Int32(0):
                red_max_sh(misc + Int32(4 * M_SMAX), Int32(wmx))
            L.bar(1, NT)
            # warps 1..: ring prefetch (the issue takes ~1 us of LSU time: it must not sit in front of
            # the sample / threshold chain)
            if warp != Int32(0):
                for pb in cutlass.range_constexpr(NBUF):
                    _issue_batch(ring, Int32(pb), Int32(pb), rowbase, ncap, d, warp, lane, G, U, NW)
            t1key = Uint32(0)
            have = Int32(0)
            if warp == Int32(0):
                # round 0: the 32 bins at and below the highest bin of the sample (the TGT-th largest
                # sample is there whenever the top of the row is positive: bins are quarter octaves of
                # the float key)
                smaxk0 = Uint32(L.lds_u32(misc + Int32(4 * M_SMAX)))
                b0 = Int32(smaxk0 >> Uint32(21))
                found = Int32(0)
                idx = b0 - lane
                hh = Int32(0)
                if idx >= Int32(0):
                    hh = L.lds_u32(sb + Int32(OFF_SHIST) + (idx + (idx >> Int32(6))) * Int32(4))
                pcum = L.warp_incl_scan_add(hh, lane)
                mk = Int32(cute.arch.vote_ballot_sync(pcum >= TGT))
                if mk != Int32(0):
                    lbq = Int32(cute.arch.popc((mk & (Int32(0) - mk)) - Int32(1)))
                    t1key = Uint32(b0 - lbq) << Uint32(21)
                    have = Int32(1)
                    found = Int32(1)
                if found == Int32(0):
                    # slow path (a signed row whose TGT-th largest sample is far below the max, e.g.
                    # mostly negative logits: the float key space has ~900 empty bins between the tiny
                    # positive and the tiny negative values, a 32-bin scan needs 30 rounds = 4.5 us):
                    # coarse level = 32 super-bins of 64 bins (lane l sums super-bin 31 - l), then the
                    # 64 bins of the hit
                    jc = Int32(31) - lane
                    cs = Int32(0)
                    for i_ in cutlass.range_constexpr(64):
                        cs = cs + L.lds_u32(
                            sb + Int32(OFF_SHIST) + (jc * Int32(65) + Int32(i_)) * Int32(4)
                        )
                    pc2 = L.warp_incl_scan_add(cs, lane)
                    mk2 = Int32(cute.arch.vote_ballot_sync(pc2 >= TGT))
                    if mk2 != Int32(0):
                        lb2 = Int32(cute.arch.popc((mk2 & (Int32(0) - mk2)) - Int32(1)))
                        jstar = Int32(31) - lb2
                        cumb = cute.arch.shuffle_sync(pc2 - cs, lb2)
                        # lane l: bins 63 - 2l and 62 - 2l of the super-bin (padded index = bin + super-bin number)
                        pa = jstar * Int32(65) + Int32(63) - lane * Int32(2)
                        ha = L.lds_u32(sb + Int32(OFF_SHIST) + pa * Int32(4))
                        hb2 = L.lds_u32(sb + Int32(OFF_SHIST) + (pa - Int32(1)) * Int32(4))
                        pc3 = cumb + L.warp_incl_scan_add(ha + hb2, lane)
                        mk3 = Int32(cute.arch.vote_ballot_sync(pc3 >= TGT))
                        lb3 = Int32(cute.arch.popc((mk3 & (Int32(0) - mk3)) - Int32(1)))
                        pre3 = cute.arch.shuffle_sync(pc3 - ha - hb2, lb3)
                        ha3 = cute.arch.shuffle_sync(ha, lb3)
                        tb = Int32(63) - lb3 * Int32(2)
                        if pre3 + ha3 < TGT:
                            tb = tb - Int32(1)
                        t1key = Uint32(jstar * Int32(64) + tb) << Uint32(21)
                        have = Int32(1)
                if TGT <= Int32(0):
                    have = Int32(0)
                if lane == Int32(0):
                    L.sts_u32(misc + Int32(4 * M_T1KEY), Int32(t1key))
                    L.sts_u32(misc + Int32(4 * M_HAVE), have)
                # publish T1 without waiting for the other warps' prefetch issue (arrive-only on barrier 3)
                L.bar_arrive(3, NT)
                for pb in cutlass.range_constexpr(NBUF):
                    _issue_batch(ring, Int32(pb), Int32(pb), rowbase, ncap, d, warp, lane, G, U, NW)
            if warp != Int32(0):
                L.bar(3, NT)
            t1key = Uint32(L.lds_u32(misc + Int32(4 * M_T1KEY)))
            have = L.lds_u32(misc + Int32(4 * M_HAVE))
            smaxk = Uint32(L.lds_u32(misc + Int32(4 * M_SMAX)))
            t1f = Float32(-3e38)
            k1 = Uint32(0)
            if have != Int32(0):
                t1f = L.invkey(t1key)
                k1 = t1key
            else:
                smaxk = Uint32(4294967295)
            span = Uint32(0)
            if smaxk > k1:
                span = smaxk - k1
            sh = Int32(0)
            bl = Int32(32) - Int32(cute.arch.clz(Int32(span)))
            if bl > Int32(10):
                sh = bl - Int32(10)

            # ---------------- stream the CTA's chunks, collect candidates ----------------
            nchunk = (n_ok + Int32(CH - 1)) // Int32(CH)
            nrem = nchunk - d
            if nrem < Int32(0):
                nrem = Int32(0)
            kc0 = (nrem + Int32(G * NW - 1)) // Int32(G * NW)
            nbat = (kc0 + Int32(U - 1)) // Int32(U)
            wcnt = Int32(0)
            kb = Int32(0)
            while kb < nbat:
                cpa_wait(NBUF - 1)
                slot_b = kb % Int32(NBUF)
                wcnt = _consume_batch(
                    ring, kb, slot_b, n_ok, d, warp, lane, t1f, sb, wcnt, G, U, NW, LCAPW
                )
                _issue_batch(
                    ring, kb + Int32(NBUF), slot_b, rowbase, nchunk, d, warp, lane, G, U, NW
                )
                kb = kb + Int32(1)
            if cutlass.const_expr(self.trigger):
                if cutlass.const_expr(self.arm):
                    # the marks are in memory (gpu scope) before any CTA of the dependent grid can start and before
                    # the other CTAs of the cluster, which wait for this arrival, store an entry
                    if d == Int32(0):
                        L.cluster_arrive()
                cute.arch.griddepcontrol_launch_dependents()
            # histogram of the warp's own candidates (32 entries per step)
            lsth = sb + Int32(OFF_LST) + warp * Int32(LCAPW * 8)
            cwh = _umin(wcnt, Int32(LCAPW))
            rh = Int32(0)
            while rh < cwh:
                ih = rh + lane
                if ih < cwh:
                    eh = lds_u64(lsth + ih * Int32(8))
                    kxh = L.fkey_bits(Uint32(eh & Uint64(4294967295)))
                    bqh = Int32(0)
                    if kxh > k1:
                        bqh = Int32((kxh - k1) >> Uint32(sh))
                    if bqh > Int32(NBIN - 1):
                        bqh = Int32(NBIN - 1)
                    red_add_sh(sb + Int32(OFF_HIST) + bqh * Int32(4), Int32(1))
                rh = rh + Int32(32)
            if lane == Int32(0):
                L.sts_u32(misc + Int32(4 * M_WCNT) + warp * Int32(4), wcnt)
                if wcnt > Int32(LCAPW):
                    red_add_sh(misc + Int32(4 * M_OVF), Int32(1))
            L.bar(1, NT)

            # ---------------- local suffix counts -> R1 all-gather -> crossing bin ----------------
            ovf_l = L.lds_u32(misc + Int32(4 * M_OVF))
            hv2 = lds_v2(sb + Int32(OFF_HIST) + tid * Int32(8))
            hb = [hv2[0], hv2[1]]
            sumt = hb[0]
            for j_ in cutlass.range_constexpr(1, BPT):
                sumt = sumt + hb[j_]
            vsuf = L.warp_suffix_scan_add(sumt, lane)
            if lane == Int32(0):
                L.sts_u32(misc + Int32(4 * M_WTOT) + warp * Int32(4), vsuf)
            L.bar(1, NT)
            hi_w = Int32(0)
            for w_ in cutlass.range_constexpr(NW):
                if Int32(w_) > warp:
                    hi_w = hi_w + L.lds_u32(misc + Int32(4 * (M_WTOT + w_)))
            ss = [Int32(0)] * BPT
            acc = (vsuf - sumt) + hi_w
            for j_ in cutlass.range_constexpr(BPT - 1, -1, -1):
                acc = acc + hb[j_]
                ss[j_] = _umin(acc, Int32(65535))
            mybase = sb + Int32(OFF_RX) + d * Int32(SLOT)
            L.sts_u32(mybase + tid * Int32(4), ss[0] | (ss[1] << Int32(16)))
            if tid == Int32(0):
                sts_v2(mybase + Int32(2048), ovf_l, Int32(0))
            if tid == Int32(0):
                L.mbar_arrive_expect_tx(b_r1, Int32((G - 1) * SLOT))
            L.fence_proxy_async_smem()
            L.cluster_wait()
            L.bar(1, NT)
            if warp == Int32(0):
                if lane < Int32(G - 1):
                    pr = (d + Int32(1) + lane) % Int32(G)
                    L.bulk_s2c(L.mapa_u32(mybase, pr), mybase, Int32(SLOT), L.mapa_u32(b_r1, pr))
            L.wait_cl(b_r1, Int32(0))
            # decode: totals per bin of this thread's bins, crossing bin, overflow flags
            tt = [Int32(0)] * BPT
            for g_ in cutlass.range_constexpr(G):
                xw = L.lds_u32(sb + Int32(OFF_RX) + Int32(g_ * SLOT) + tid * Int32(4))
                tt[0] = tt[0] + (xw & Int32(65535))
                tt[1] = tt[1] + ((xw >> Int32(16)) & Int32(65535))
            cand = Int32(0)
            for j_ in cutlass.range_constexpr(BPT):
                if tt[j_] >= Int32(K_TOP):
                    cand = tid * Int32(BPT) + Int32(j_ + 1)
            wmc = L.warp_max_u32(Uint32(cand))
            if lane == Int32(0):
                red_max_sh(misc + Int32(4 * M_BSTAR), Int32(wmc))
            ovf_any = Int32(0)
            for g_ in cutlass.range_constexpr(G):
                ovf_any = ovf_any | L.lds_u32(sb + Int32(OFF_RX) + Int32(g_ * SLOT + 2048))
            L.bar(1, NT)
            bsp1 = L.lds_u32(
                misc + Int32(4 * M_BSTAR)
            )  # B* + 1, 0 = the total is below K (undershoot)
            bstar = bsp1 - Int32(1)
            good = Int32(0)
            if bsp1 > Int32(0):
                if ovf_any == Int32(0):
                    good = Int32(1)
            above_tot = Int32(0)
            m_tot = Int32(0)
            off_d = Int32(0)
            moff_d = Int32(0)
            at_0 = Int32(0)
            if good != Int32(0):
                for g_ in cutlass.range_constexpr(G):
                    gb = sb + Int32(OFF_RX) + Int32(g_ * SLOT)
                    wa_ = L.lds_u32(gb + (bstar >> Int32(1)) * Int32(4))
                    sat = (wa_ >> ((bstar & Int32(1)) * Int32(16))) & Int32(65535)
                    sab = Int32(0)
                    if bstar < Int32(NBIN - 1):
                        wb_ = L.lds_u32(gb + ((bstar + Int32(1)) >> Int32(1)) * Int32(4))
                        sab = (wb_ >> (((bstar + Int32(1)) & Int32(1)) * Int32(16))) & Int32(65535)
                    atg = sat - sab
                    if Int32(g_) < d:
                        off_d = off_d + sab
                        moff_d = moff_d + atg
                    if g_ == 0:
                        at_0 = atg
                    above_tot = above_tot + sab
                    m_tot = m_tot + atg
            outrow = out.iterator.toint() + Int64(row) * Int64(K_TOP) * Int64(4)
            lstw = sb + Int32(OFF_LST) + warp * Int32(LCAPW * 8)
            lm2 = Int32(cute.arch.lanemask_lt())
            need_fb = Int32(0)
            if good == Int32(0):
                need_fb = Int32(1)
            if m_tot > Int32(MCAP):
                need_fb = Int32(1)
            if cutlass.const_expr(self.force_fallback):
                need_fb = Int32(1)
            if need_fb != Int32(0):
                # ---- exact radix fallback over the row (rare: undershoot, list overflow, crossing bin > MCAP) ----
                fpar = Int32(1)  # parity of the next phase of b_r1 (phase 0 was the R1 exchange)
                pfx = Uint32(0)
                need_r = Int32(K_TOP)
                lvl = Int32(0)
                far = cute.make_rmem_tensor((8,), Int32)
                tcr = cute.make_rmem_tensor((8,), Int32)
                hgr = cute.make_rmem_tensor((8,), Int32)
                for z_ in cutlass.range_constexpr(8):
                    far[z_] = Int32(0)
                    tcr[z_] = Int32(0)
                    hgr[z_] = Int32(0)
                while lvl < Int32(4):
                    shp = Int32(32) - lvl * Int32(8)
                    dsh = Int32(24) - lvl * Int32(8)
                    # zero the level histogram and the per-source above counters
                    if tid < Int32(256):
                        L.sts_u32(sb + Int32(OFF_SHIST) + tid * Int32(4), Int32(0))
                    if tid < Int32(8):
                        L.sts_u32(misc + Int32(4 * M_FA) + tid * Int32(4), Int32(0))
                    cpa_wait(0)
                    L.bar(1, NT)
                    for pbf in cutlass.range_constexpr(NBUF):
                        _issue_batch(
                            ring, Int32(pbf), Int32(pbf), rowbase, nchunk, d, warp, lane, G, U, NW
                        )
                    kbf = Int32(0)
                    while kbf < nbat:
                        cpa_wait(NBUF - 1)
                        slotf = kbf % Int32(NBUF)
                        _level_batch(
                            ring, kbf, slotf, n_ok, d, warp, lane, pfx, shp, dsh, sb, G, U, NW
                        )
                        _issue_batch(
                            ring, kbf + Int32(NBUF), slotf, rowbase, nchunk, d, warp, lane, G, U, NW
                        )
                        kbf = kbf + Int32(1)
                    cpa_wait(0)
                    L.bar(1, NT)
                    # push the 256-bin histogram (u32) to the cluster (cluster barrier first: the previous
                    # level's slots are read)
                    if tid < Int32(256):
                        L.sts_u32(
                            mybase + tid * Int32(4),
                            L.lds_u32(sb + Int32(OFF_SHIST) + tid * Int32(4)),
                        )
                    if tid == Int32(0):
                        L.mbar_arrive_expect_tx(b_r1, Int32((G - 1) * 1024))
                    L.fence_proxy_async_smem()
                    L.cluster_arrive()
                    L.cluster_wait()
                    L.bar(1, NT)
                    if warp == Int32(0):
                        if lane < Int32(G - 1):
                            prf = (d + Int32(1) + lane) % Int32(G)
                            L.bulk_s2c(
                                L.mapa_u32(mybase, prf), mybase, Int32(1024), L.mapa_u32(b_r1, prf)
                            )
                    L.wait_cl(b_r1, fpar)
                    # decode: totals per digit, crossing digit
                    tot_t = Int32(0)
                    for g_ in cutlass.range_constexpr(G):
                        xg = Int32(0)
                        if tid < Int32(256):
                            xg = L.lds_u32(sb + Int32(OFF_RX) + Int32(g_ * SLOT) + tid * Int32(4))
                        hgr[g_] = xg
                        tot_t = tot_t + xg
                    vs = L.warp_suffix_scan_add(tot_t, lane)
                    if lane == Int32(0):
                        L.sts_u32(misc + Int32(4 * M_WTOT) + warp * Int32(4), vs)
                    L.bar(1, NT)
                    hiw = Int32(0)
                    for w_ in cutlass.range_constexpr(NW):
                        if Int32(w_) > warp:
                            hiw = hiw + L.lds_u32(misc + Int32(4 * (M_WTOT + w_)))
                    s_ex = (vs - tot_t) + hiw
                    if tid < Int32(256):
                        if s_ex < need_r:
                            if s_ex + tot_t >= need_r:
                                L.sts_u32(misc + Int32(4 * M_FDIG), tid)
                                L.sts_u32(misc + Int32(4 * M_FABOVE), s_ex)
                    L.bar(1, NT)
                    dig = L.lds_u32(misc + Int32(4 * M_FDIG))
                    above_l = L.lds_u32(misc + Int32(4 * M_FABOVE))
                    if tid < Int32(256):
                        if tid > dig:
                            for g_ in cutlass.range_constexpr(G):
                                red_add_sh(misc + Int32(4 * (M_FA + g_)), hgr[g_])
                    L.bar(1, NT)
                    for g_ in cutlass.range_constexpr(G):
                        far[g_] = far[g_] + L.lds_u32(misc + Int32(4 * (M_FA + g_)))
                        tcr[g_] = L.lds_u32(sb + Int32(OFF_RX) + Int32(g_ * SLOT) + dig * Int32(4))
                    # every thread has read this level's M_FA words and RX slots: the next level zeroes M_FA and
                    # rewrites the own RX slot (without this barrier a slow warp can read the zeros)
                    L.bar(1, NT)
                    need_r = need_r - above_l
                    pfx = (pfx << Uint32(8)) | Uint32(dig)
                    fpar = fpar ^ Int32(1)
                    lvl = lvl + Int32(1)
                # kth key = pfx; need_r ties (keys equal to it) complete the selection
                a_pre = Int32(0)
                t_pre = Int32(0)
                a_tot = Int32(0)
                t_my = Int32(0)
                for g_ in cutlass.range_constexpr(G):
                    if Int32(g_) < d:
                        a_pre = a_pre + far[g_]
                        t_pre = t_pre + tcr[g_]
                    if Int32(g_) == d:
                        t_my = tcr[g_]
                    a_tot = a_tot + far[g_]
                quota = need_r - t_pre
                if quota < Int32(0):
                    quota = Int32(0)
                if quota > t_my:
                    quota = t_my
                if tid < Int32(8):
                    L.sts_u32(misc + Int32(4 * M_CABOVE) + tid * Int32(4), Int32(0))
                cpa_wait(0)
                L.bar(1, NT)
                for pbf2 in cutlass.range_constexpr(NBUF):
                    _issue_batch(
                        ring, Int32(pbf2), Int32(pbf2), rowbase, nchunk, d, warp, lane, G, U, NW
                    )
                kbw = Int32(0)
                while kbw < nbat:
                    cpa_wait(NBUF - 1)
                    slotw = kbw % Int32(NBUF)
                    _final_batch(
                        ring,
                        kbw,
                        slotw,
                        n_ok,
                        d,
                        warp,
                        lane,
                        pfx,
                        outrow,
                        a_pre,
                        a_tot + t_pre,
                        quota,
                        sb,
                        G,
                        U,
                        NW,
                    )
                    _issue_batch(
                        ring, kbw + Int32(NBUF), slotw, rowbase, nchunk, d, warp, lane, G, U, NW
                    )
                    kbw = kbw + Int32(1)
                cpa_wait(0)
            if need_fb == Int32(0):
                if d == Int32(0):
                    if tid == Int32(0):
                        L.mbar_arrive_expect_tx(b_r2, (m_tot - at_0) * Int32(8))
                cw = _umin(wcnt, Int32(LCAPW))
                # pass A: per-warp counts of the entries above B* and in B*
                na_w = Int32(0)
                nm_w = Int32(0)
                ra = Int32(0)
                while ra < cw:
                    ia = ra + lane
                    pa_ = Int32(0)
                    pm_ = Int32(0)
                    if ia < cw:
                        ea = lds_u64(lstw + ia * Int32(8))
                        ka = L.fkey_bits(Uint32(ea & Uint64(4294967295)))
                        qa = Int32(0)
                        if ka > k1:
                            qa = Int32((ka - k1) >> Uint32(sh))
                        if qa > Int32(NBIN - 1):
                            qa = Int32(NBIN - 1)
                        if qa > bstar:
                            pa_ = Int32(1)
                        if qa == bstar:
                            pm_ = Int32(1)
                    na_w = na_w + Int32(
                        cute.arch.popc(Int32(cute.arch.vote_ballot_sync(pa_ != Int32(0))))
                    )
                    nm_w = nm_w + Int32(
                        cute.arch.popc(Int32(cute.arch.vote_ballot_sync(pm_ != Int32(0))))
                    )
                    ra = ra + Int32(32)
                if lane == Int32(0):
                    L.sts_u32(misc + Int32(4 * M_WA) + warp * Int32(4), na_w)
                    L.sts_u32(misc + Int32(4 * M_WM) + warp * Int32(4), nm_w)
                L.bar(1, NT)
                pre_a = Int32(0)
                pre_m = Int32(0)
                for w2 in cutlass.range_constexpr(NW):
                    if Int32(w2) < warp:
                        pre_a = pre_a + L.lds_u32(misc + Int32(4 * (M_WA + w2)))
                        pre_m = pre_m + L.lds_u32(misc + Int32(4 * (M_WM + w2)))
                oa = off_d + pre_a
                om = moff_d + pre_m
                # pass B: the writes
                rb = Int32(0)
                while rb < cw:
                    ib = rb + lane
                    pa2 = Int32(0)
                    pm2 = Int32(0)
                    eb = Uint64(0)
                    if ib < cw:
                        eb = lds_u64(lstw + ib * Int32(8))
                        kb2 = L.fkey_bits(Uint32(eb & Uint64(4294967295)))
                        qb = Int32(0)
                        if kb2 > k1:
                            qb = Int32((kb2 - k1) >> Uint32(sh))
                        if qb > Int32(NBIN - 1):
                            qb = Int32(NBIN - 1)
                        if qb > bstar:
                            pa2 = Int32(1)
                        if qb == bstar:
                            pm2 = Int32(1)
                    ma2 = Int32(cute.arch.vote_ballot_sync(pa2 != Int32(0)))
                    mm2 = Int32(cute.arch.vote_ballot_sync(pm2 != Int32(0)))
                    if pa2 != Int32(0):
                        st_g_i32(
                            outrow + Int64(oa + Int32(cute.arch.popc(ma2 & lm2))) * Int64(4),
                            Int32(Uint32(eb >> Uint64(32))),
                        )
                    if pm2 != Int32(0):
                        mi = om + Int32(cute.arch.popc(mm2 & lm2))
                        if d == Int32(0):
                            sts_u64(sb + Int32(OFF_XIN) + mi * Int32(8), eb)
                        else:
                            L.st_async_v2(
                                L.mapa_u32(sb + Int32(OFF_XIN) + mi * Int32(8), Int32(0)),
                                L.mapa_u32(b_r2, Int32(0)),
                                Int32(Uint32(eb & Uint64(4294967295))),
                                Int32(Uint32(eb >> Uint64(32))),
                            )
                    oa = oa + Int32(cute.arch.popc(ma2))
                    om = om + Int32(cute.arch.popc(mm2))
                    rb = rb + Int32(32)
                if d == Int32(0):
                    L.bar(1, NT)
                    L.wait_cl(b_r2, Int32(0))
                    need = Int32(K_TOP) - above_tot
                    ok_m = Int32(0)
                    if m_tot <= Int32(MCAP):
                        ok_m = Int32(1)
                    if ok_m != Int32(0):
                        jj = tid
                        while jj < m_tot:
                            ej = lds_u64(sb + Int32(OFF_XIN) + jj * Int32(8))
                            rk = Int32(0)
                            if need < m_tot:
                                kj = L.entry_key(ej)
                                j2 = Int32(0)
                                while j2 < m_tot:
                                    if (
                                        L.entry_key(lds_u64(sb + Int32(OFF_XIN) + j2 * Int32(8)))
                                        > kj
                                    ):
                                        rk = rk + Int32(1)
                                    j2 = j2 + Int32(1)
                            else:
                                rk = jj
                            if rk < need:
                                st_g_i32(
                                    outrow + Int64(above_tot + rk) * Int64(4),
                                    Int32(Uint32(ej >> Uint64(32))),
                                )
                            jj = jj + Int32(NT)
        else:
            # short row: identity and a -1 tail (written by CTA 0 of the row's cluster)
            L.bar(1, NT)
            if cutlass.const_expr(self.arm):
                if d == Int32(0):
                    L.cluster_arrive()
                else:
                    L.cluster_arrive_relaxed()
            else:
                L.cluster_arrive_relaxed()
            if d == Int32(0):
                for r2 in cutlass.range_constexpr(512 // NT if NT < 512 else 1):
                    jq = tid + Int32(r2 * NT)
                    if jq < Int32(K_TOP):
                        val = Int32(-1)
                        if jq < n_ok:
                            val = jq
                        st_g_i32(
                            out.iterator.toint()
                            + (Int64(row) * Int64(K_TOP) + Int64(jq)) * Int64(4),
                            val,
                        )
            L.cluster_wait()


def compile_topk_multicta(kernel: TopkMultiCta):
    """Compile ``kernel`` (logits [B, npad] float32 with a dynamic row stride, ctx [B] int32, out [B, 512] int32)."""
    rt = cute.runtime
    sb = cute.sym_int()
    logits_fake = rt.make_fake_tensor(
        Float32, (sb, cute.sym_int()), stride=(cute.sym_int64(), 1), assumed_align=16
    )
    ctx_fake = rt.make_fake_compact_tensor(Int32, (sb,), stride_order=(0,))
    out_fake = rt.make_fake_compact_tensor(Int32, (sb, K_TOP), stride_order=(1, 0))
    stream = rt.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        logits_fake,
        ctx_fake,
        out_fake,
        stream,
        options="--enable-tvm-ffi --generate-line-info",
    )
