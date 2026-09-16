# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Four-value BF16 streaming specialization of self-sampling GVR decode.

Retains the four-value scan geometry for shapes where wider loads regress.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass.cute import runtime as _crt
from cutlass.utils.smem_allocator import SmemAllocator

from .gvr_topk_decode_self_sampling_bf16 import (
    _dtype_of,
    _fmaf,
    _ldcg_v2_i32,
    _ldg_f32_rs,
    _lds_v2_u64,
    _pin_i32,
    _pin_i64,
    _prefetch_l2,
    _red_shared_add1,
    _smem_addr_reg,
    _st_g_u32,
    _st_g_u64,
    _st_s_v2_u32,
    atomic_add_cta,
    atomic_add_u64_gpu,
    atomic_max_cta,
    atomic_min_cta,
    ballot,
    clz_i32,
    f2s_rz,
    f2u_rz,
    f32_of_i32,
    ffs_m1,
    fkey,
    fkey_bits,
    fmax_f32,
    fmin_f32,
    g2r_atom_f32,
    gather_hint,
    invkey,
    ld_g_bf16x4,
    ld_g_f32x4,
    ldg_bf16,
    ldg_bf16_rs,
    ldg_f32,
    popc,
    scan_cross0,
    threadfence_gpu,
    u32_of_f32,
    warp_incl_scan_add,
    warp_max_u32,
    warp_min_u32,
)

MAXC = 160  # multi-CTA SPLIT row cap


GCAP = 16384  # per-row slab capacity in int2


QUADC_CLUS = 288  # clus + gvr_main gate


IDXB = 22  # packed candidate index bits


IDXM = (1 << IDXB) - 1


GVR_WS_OFF_OFF = MAXC * 8  # workspace g_off byte offset


GVR_WS_BUF_OFF = 2048  # workspace g_buf byte offset


SENT_LO = -3.0e38


SENT_HI = 3.0e38


RES_B = 0


RES_M = 1


RES_ABOVE = 2


RES_TOT = 3


RES_B2 = 4


RES_B3 = 5


MAXC__main = MAXC


GCAP__main = GCAP


IDXB__main = IDXB


IDXM__main = IDXM


QUADC_CLUS__main = QUADC_CLUS


WS_BYTES = GVR_WS_BUF_OFF + MAXC__main * GCAP__main * 8  # 20,973,568


_NEG_INF = float("-inf")


class GvrMainKernel:
    """gvr_main<BLK, U, MINB, NBS, KPT, SPLIT> — streaming self-sampling GVR."""

    def __init__(
        self,
        blk: int,
        u: int,
        minb: int,
        nbs: int,
        kpt: int,
        split: bool,
        tshg: bool = False,
        varlen: bool = False,
        next_n: int = 1,
        cr_shift: int = 0,
        r_const: int = 1,
        hint_free: bool = False,
        prefill: bool = False,
        dtype=cutlass.Float32,
    ) -> None:
        assert nbs == 256, "SNB must stay 256"
        assert blk in (256, 512, 1024) and u in (1, 2, 4, 8)
        assert kpt in (1, 2, 4, 8) and minb in (1, 2, 4)
        # constexpr logits dtype: Float32 (verbatim arm) or BFloat16 (native
        # bf16 arm). The bf16 arm is varlen/hint-free only (gather_hint and
        # the prefill window base stay fp32-pinned).
        self.dtype = dtype
        assert dtype in (cutlass.Float32, cutlass.BFloat16)
        if dtype is not cutlass.Float32:
            assert bool(hint_free) and not bool(prefill), "bf16 arm is hint-free, non-prefill"
        self.blk = blk
        self.u = u
        self.minb = minb
        self.nbs = nbs
        self.kpt = kpt
        self.split = bool(split)
        # ---- per-row varlen mode (production heuristicTopKDecode contract) --
        # n and the sampling-ladder scalars are re-derived PER ROW inside the
        # kernel from a device kv_lens tensor (route_dynamic formula mirror);
        # the scalar n/SMP/TGT/Q/SS2/TGT2 launch args become dead.  next_n /
        # cr_shift (log2 compressRatio: 0 = DSv3.2, 2 = DSv4) / r_const (the
        # frozen grid.x) are compile-time so their divisions strength-reduce.
        self.varlen = bool(varlen)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        self.r_const = int(r_const)
        # hint-free: gather_hint sites compiled out (sentinel pass-through)
        self.hint_free = bool(hint_free)
        # prefill: per-row [ks, ke) window riding the kv_lens/pre_idx ABI slots,
        # base rounded down to 16B with the <=3 lead lanes masked, one CTA per
        # row (no SPLIT); every edit is const_expr-gated so other codegen is unchanged.
        self.prefill = bool(prefill)
        if self.prefill:
            assert (
                self.varlen
                and self.hint_free
                and self.next_n == 1
                and self.cr_shift == 0
                and not self.split
            )
        if self.varlen:
            assert self.next_n >= 1 and self.cr_shift in (0, 2) and self.r_const >= 1
        # TSH-floor staging arm.  SPLIT-only compile-time key; the CUDA form
        # is a grid-uniform runtime gate over the same predicate
        # (b > 15 && k <= 1024 && n4 <= 32768).  varlen mode compiles the
        # machinery in whenever SPLIT and gates it per row at runtime
        # (tsh_en && n4 <= 32768) — mirroring the CUDA runtime gate.
        if self.varlen:
            self.tshg = bool(split)
        else:
            self.tshg = bool(tshg) and bool(split)
        # derived constexprs (bit-identical to the CUDA)
        self.hb = nbs
        self.kbig = (kpt >= 2) and (kpt * blk >= 2048)
        self.scpb = (8192 if split else 16384) if blk >= 1024 else (8192 if self.kbig else 4096)
        self.cmpb = (4096 if self.kbig else 2048) if blk >= 1024 else 1024
        self.shd = not split
        self.vstg = split or blk >= 512
        self.pfd = (u if u < 4 else 4) if minb <= 2 else 0
        self.pf = self.pfd > 0
        self.natt = 1 if split else 3
        # smem blob byte map: cbuf/cbuf2 alias @0,
        # ck64 @ 4*(VSTG ? 2*(SCPB+4) : SCPB+4), size (CMPB+1)*8
        self.ck_off = 4 * ((2 * (self.scpb + 4)) if self.vstg else (self.scpb + 4))
        assert self.ck_off % 16 == 0, "ck64 must stay 16B aligned (ulonglong2)"
        self.dyn_bytes = self.ck_off + (self.cmpb + 1) * 8
        self.lb = self.nbs.bit_length() - 1  # log2(NBS)=8

    # ------------------------------------------------------------------
    # GVR_EMITC: classify+stage one survivor.
    # Returns pos+1. Branchless trash slot min(pos, SCPB).
    # ------------------------------------------------------------------
    @cute.jit
    def _emitc(self, xv, idx, pos, TF, SC, hb, cb2, s_hist, s_cbuf, s_cbuf2):
        SCPB = self.scpb
        NBS = self.nbs
        if cutlass.const_expr(not self.split):
            bn_u = f2u_rz((xv - TF) * SC)  # saturating cvt.rzi
            if bn_u > cutlass.Uint32(NBS - 1):
                bn_u = cutlass.Uint32(NBS - 1)
            bn = cutlass.Int32(bn_u)
            if cutlass.const_expr(self.vstg):
                # result unused -> resultless red off the pinned hist base
                # (no per-site smem-base refold)
                _red_shared_add1(hb + (bn << cutlass.Int32(2)))
            else:
                # VSTG=False tuples sit at the 64-register wall: keep the
                # original spelling, no pinned base here
                atomic_add_cta(s_hist.iterator + bn, cutlass.Int32(1))
            if cutlass.const_expr(not self.vstg):
                ps = pos
                if ps > cutlass.Int32(SCPB):
                    ps = cutlass.Int32(SCPB)  # trash slot (IMNMX)
                s_cbuf[ps] = cutlass.Int32(
                    (bn_u << cutlass.Uint32(IDXB__main)) | cutlass.Uint32(idx)
                )
        if cutlass.const_expr(self.vstg):
            ps = pos
            if ps > cutlass.Int32(SCPB):
                ps = cutlass.Int32(SCPB)
            # int2 {value bits, idx} via st.shared.v2.u32 — same bytes as the
            # (idx << 32) | bits u64 pack (+0=bits, +4=idx), but no i64
            # materialization inside the bit-walk; address = one LEA off the
            # pinned cb2 base
            _st_s_v2_u32(cb2 + ps * cutlass.Int32(8), u32_of_f32(xv), cutlass.Uint32(idx))
        return pos + cutlass.Int32(1)

    # ------------------------------------------------------------------
    # two-predicate warp-ballot emit step (shared by P6 and both degen
    # emits): q1 winners to out[base1+p] p<cap1, q2 ties to out[base2+p]
    # p<cap2. s_scal[1]=s_o1, s_scal[2]=s_o2.
    # ------------------------------------------------------------------
    @cute.jit
    def _ballot_pair_emit(self, p1, p2, idv, base1, cap1, base2, cap2, out_row, s_scal, lane):
        n1 = ballot(p1 != cutlass.Int32(0))
        n2 = ballot(p2 != cutlass.Int32(0))
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        if lane == cutlass.Int32(0):
            if n1 != cutlass.Int32(0):
                b1 = atomic_add_cta(s_scal.iterator + 1, cutlass.Int32(popc(n1)))
            if n2 != cutlass.Int32(0):
                b2 = atomic_add_cta(s_scal.iterator + 2, cutlass.Int32(popc(n2)))
        b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
        b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
        lm = cutlass.Int32(cute.arch.lanemask_lt())
        if p1 != cutlass.Int32(0):
            p = b1 + cutlass.Int32(popc(n1 & lm))
            if p < cap1:
                out_row[base1 + p] = idv
        if p2 != cutlass.Int32(0):
            p = b2 + cutlass.Int32(popc(n2 & lm))
            if p < cap2:
                out_row[base2 + p] = idv

    # ------------------------------------------------------------------
    # kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        out: cute.Tensor,
        ws: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        scap_dead: cutlass.Int32,
        cmp_dead: cutlass.Int32,
        R: cutlass.Int32,
        SMP: cutlass.Int32,
        TGT: cutlass.Int32,
        Q: cutlass.Int32,
        SS2: cutlass.Int32,
        TGT2: cutlass.Int32,
        kv_lens: cute.Tensor,
        aim_base: cutlass.Int32,
        sfac: cutlass.Int32,
        amin: cutlass.Int32,
        sd_en: cutlass.Int32,
        tsh_en: cutlass.Int32,
    ):
        BLK = self.blk
        U = self.u
        NBS = self.nbs
        KPT = self.kpt
        SCPB = self.scpb
        CMPB = self.cmpb
        PFD = self.pfd
        NATT = self.natt
        NW = BLK // 32

        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()  # 2-D grid (part, row)
        row = by
        part = cutlass.Int32(0)
        if cutlass.const_expr(self.split):
            part = bx
        lane = tidx & cutlass.Int32(31)

        # ================= per-row varlen prologue (varlen mode only) =========
        # Production contract: row r serves request r // next_n with
        # kv_len = kv_lens[r // next_n], n = (kv_len - next_n + r % next_n + 1)
        # >> cr_shift.  The sampling-ladder scalars are then re-derived from
        # this row's n by the EXACT route_dynamic() host formulas (the scalar
        # launch args are dead in this mode).  n <= k rows have no runtime
        # `return` in CuTe DSL: they run the body as a zero-work pass
        # (n = 0, TGT = INT_MAX so no rung ever accepts) and the identity/pad
        # emission happens in the epilogue at the end of the kernel.  Every
        # value below is a pure function of `row`, so all R split CTAs of a
        # row (and all threads) compute identical scalars — grid-uniform per
        # row by construction.
        short = cutlass.Int32(0)
        n_row = cutlass.Int32(0)
        tsh_run = cutlass.Int32(1)
        # prefill window offset: lead = ks & 3 low lanes masked, col0 = ks
        # rounded down to a float4 boundary (declared before the dynamic ifs
        # per the scoping rule; stay 0 in every non-prefill compile).
        lead = cutlass.Int32(0)
        col0 = cutlass.Int32(0)
        if cutlass.const_expr(self.varlen):
            if cutlass.const_expr(self.prefill):
                # ks/ke are already compressed column units (kv_lens = row_starts,
                # pre_idx = row_ends); the clamps are for memory safety only.
                ks = kv_lens[row]
                ke = pre_idx[row]
                if ks < cutlass.Int32(0):
                    ks = cutlass.Int32(0)
                if ke > npad:
                    ke = npad
                if ks > ke:
                    ks = ke
                nv = ke - ks
                lead = ks & cutlass.Int32(3)
                col0 = ks - lead
                if col0 > npad - cutlass.Int32(4):
                    col0 = npad - cutlass.Int32(4)
            else:
                req = row // cutlass.Int32(self.next_n)
                rr = row % cutlass.Int32(self.next_n)
                kvl = kv_lens[req]
                nv = (kvl - cutlass.Int32(self.next_n) + rr + cutlass.Int32(1)) >> cutlass.Int32(
                    self.cr_shift
                )
                if nv < cutlass.Int32(0):
                    nv = cutlass.Int32(0)
                if nv > npad:
                    nv = npad
            n_row = nv
            if nv <= k:
                short = cutlass.Int32(1)
                n = cutlass.Int32(0)
                SMP = cutlass.Int32(0)
                SS2 = cutlass.Int32(1)
                # "never accepts" sentinels; 2^30-1 so the TGT*2 scan target
                # stays positive (0x7FFFFFFF would overflow to -2 and flip
                # every tot0 >= TGT*2 gate on the all-zero histogram)
                TGT = cutlass.Int32(0x3FFFFFFF)
                TGT2 = cutlass.Int32(0x3FFFFFFF)
                Q = cutlass.Int32(0)
            if short == cutlass.Int32(0):
                if cutlass.const_expr(self.prefill):
                    # scan extent from the rounded-down base col0 spans the
                    # lead pad plus the real window: [col0, ke) = nv + lead.
                    n = nv + lead
                else:
                    n = nv
                n4v = n >> cutlass.Int32(2)
                # Ladder-scalar baselines only: the real SMP/SS2/TGT/TGT2 are
                # derived by warp0 alone in the block below (bit-identical
                # formulas) and published through s_lad — every thread's local
                # copies here are overwritten by the post-barrier smem read.
                SMP = cutlass.Int32(0)
                SS2 = cutlass.Int32(1)
                TGT = cutlass.Int32(0)
                TGT2 = cutlass.Int32(0)
                if cutlass.const_expr(self.split):
                    Q = (n4v + cutlass.Int32(self.r_const - 1)) // cutlass.Int32(self.r_const)
                else:
                    Q = cutlass.Int32(0)
            # per-row TSH-floor runtime gate (CUDA parity: b>15 && k<=1024 in
            # tsh_en, n4 <= 32768 per row)
            tsh_run = cutlass.Int32(0)
            if tsh_en != cutlass.Int32(0):
                if (n >> cutlass.Int32(2)) <= cutlass.Int32(32768):
                    tsh_run = cutlass.Int32(1)

        # ---- shared memory (one blob, compile-time offsets) ----
        smem = SmemAllocator()
        s_hist = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((self.hb,), order=(0,)), byte_alignment=128
        )
        s_ws = smem.allocate_tensor(  # unused; byte parity  # noqa: F841
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        s_wmn = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        s_wmx = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        # crossing-scan result slots (RES_B/M/ABOVE/TOT/B2/B3)
        s_res = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((8,), order=(0,)), byte_alignment=16
        )
        # scalar block: [0]=s_bufn [1]=s_o1 [2]=s_o2 [3]=s_base
        s_scal = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
        )
        s_pk = smem.allocate_tensor(
            cutlass.Int64, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=8
        )
        s_tsh = smem.allocate_tensor(
            cutlass.Float32, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=4
        )
        # STATIC smem word for the walk's byte stride — kept out of the blob
        # so dyn_bytes stays equal to the CUDA dispatch's smem. blk==512 VSTG
        # only.
        if cutlass.const_expr(self.vstg and self.blk == 512):
            s_x4 = smem.allocate_tensor(
                cutlass.Int32, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=4
            )
        s_kmm = smem.allocate_tensor(  # [0]=kmin [1]=kmax
            cutlass.Uint32, cute.make_ordered_layout((2,), order=(0,)), byte_alignment=8
        )
        if cutlass.const_expr(self.varlen):
            # ladder broadcast slots: [0]=SMP [1]=SS2 [2]=TGT [3]=TGT2
            # (static like s_x4, so dyn_bytes keeps CUDA dispatch parity)
            s_lad = smem.allocate_tensor(
                cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
            )
        if cutlass.const_expr(self.prefill):
            # per-row lead (0..3) broadcast slot: warp0 publishes it under the
            # existing s_lad barrier; every masked-lane site reloads it from
            # smem so no live register is carried on the 64-register arms.
            s_lead = smem.allocate_tensor(
                cutlass.Int32, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=4
            )
        blob = smem.allocate_tensor(  # dynamic-equivalent region
            cutlass.Int8, cute.make_ordered_layout((self.dyn_bytes,), order=(0,)), byte_alignment=16
        )
        sbase = blob.iterator.toint()
        s_cbuf = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, sbase, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((SCPB + 4,)),
        )
        s_cbuf2 = cute.make_tensor(
            cute.make_ptr(cutlass.Uint64, sbase, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((SCPB + 4,)),
        )
        ck_addr = sbase + cutlass.Int32(self.ck_off)
        s_ck64 = cute.make_tensor(
            cute.make_ptr(cutlass.Uint64, ck_addr, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((CMPB + 1,)),
        )

        # emission smem bases pinned ONCE, outside the attempt/tile loops
        # (asm identity mov) — LLVM otherwise refolds the shared-window
        # materialisation into every _emitc site inside the divergent
        # bit-walk. VSTG-only: the VSTG=False tuples keep their original
        # spellings untouched (64-register wall).
        hb_pin = cutlass.Int32(0)
        cb2_pin = cutlass.Int32(0)
        if cutlass.const_expr(self.vstg):
            hb_pin = _smem_addr_reg(s_hist.iterator.toint())
            cb2_pin = _smem_addr_reg(s_cbuf2.iterator.toint())
        # park the stride 4 in the dedicated smem word and load it back —
        # the LDS result is opaque to ptxas, so the walk's stride register
        # cannot be re-materialized in-loop (asm-mov and shfl forms are
        # folded by ptxas value-tracking). blk==512 family ONLY: the other
        # arms sit at the 64-register wall. Threads are converged here
        # (kernel prologue), so the one extra barrier is safe.
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            x4_pin = cutlass.Int32(4)
            if cutlass.const_expr(self.vstg and self.blk == 512):
                if tidx == cutlass.Int32(0):
                    s_x4[0] = cutlass.Int32(4)
                cute.arch.barrier()
                x4_pin = s_x4[0]
        else:
            # bf16: the parked walk stride is the 2-byte element size
            x4_pin = cutlass.Int32(2)
            if cutlass.const_expr(self.vstg and self.blk == 512):
                if tidx == cutlass.Int32(0):
                    s_x4[0] = cutlass.Int32(2)
                cute.arch.barrier()
                x4_pin = s_x4[0]

        # ---- row bases ----
        row64 = cutlass.Int64(row)
        # _pin_i64: keep the row base a REGISTER across the attempt/tile scf
        # regions (NVVM otherwise re-derives ld.param+%ctaid.y+mul per region)
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            if cutlass.const_expr(self.prefill):
                # base rounded down to the col0 float4 boundary (16B aligned since
                # the row base is 16B aligned and col0 is a multiple of 4).
                x_addr = _pin_i64(
                    logits.iterator.toint()
                    + (row64 * cutlass.Int64(npad) + cutlass.Int64(col0)) * cutlass.Int64(4)
                )
            else:
                x_addr = _pin_i64(
                    logits.iterator.toint() + row64 * cutlass.Int64(npad) * cutlass.Int64(4)
                )
        else:
            # bf16: 2-byte elements (prefill is fp32-pinned; ctor asserts)
            x_addr = _pin_i64(
                logits.iterator.toint() + row64 * cutlass.Int64(npad) * cutlass.Int64(2)
            )
        # varlen: pre_idx is REQUEST-level [num_rows/next_n, k] — a request's
        # next_n rows share one hint row (production contract); legacy mode
        # keeps the per-row mapping (next_n == 1 makes them identical). In
        # prefill the pre_idx slot is 1-D row_ends (already consumed in the
        # prologue), so this dead hint pointer is compiled out.
        if cutlass.const_expr(not self.prefill):
            prow64 = row64
            if cutlass.const_expr(self.varlen):
                prow64 = cutlass.Int64(row // cutlass.Int32(self.next_n))
            p_addr = pre_idx.iterator.toint() + prow64 * cutlass.Int64(k) * cutlass.Int64(4)
        out_row = out[row, None]
        ws_addr = ws.iterator.toint()
        gdon_addr = ws_addr  # slab views
        goff_addr = ws_addr + cutlass.Int64(GVR_WS_OFF_OFF)
        gbuf_addr = ws_addr + cutlass.Int64(GVR_WS_BUF_OFF)
        # SPLIT only: row-slab base pinned like x_addr above; the
        # publish/gather/P5/degen consumers spell gbuf_row + i*8 instead of
        # re-deriving gbuf_addr + (row64*GCAP__main + i)*8 per candidate
        # (value-identical by i64 distributivity).
        gbuf_row = cutlass.Int64(0)
        if cutlass.const_expr(self.split):
            gbuf_row = _pin_i64(gbuf_addr + row64 * cutlass.Int64(GCAP__main) * cutlass.Int64(8))

        n4 = n >> cutlass.Int32(2)
        c0 = cutlass.Int32(0)
        c1 = n4
        if cutlass.const_expr(self.split):
            c0 = part * Q
            c1 = c0 + Q
            if c1 > n4:
                c1 = n4
        tail0 = n4 << cutlass.Int32(2)
        tailn = cutlass.Int32(0)
        if part == cutlass.Int32(0):
            tailn = n - tail0

        if tidx == cutlass.Int32(0):
            s_scal[0] = cutlass.Int32(0)  # s_bufn
            s_res[RES_B2] = cutlass.Int32(-1)
            s_res[RES_B3] = cutlass.Int32(-1)
        if tidx < cutlass.Int32(self.hb):  # HB<=BLK always
            s_hist[tidx] = cutlass.Int32(0)

        # ===== varlen: warp0-only ladder mirror + register-free L2 hints =====
        # The sampling-ladder scalars are a pure function of the row; issuing
        # the mirror chain (runtime divides + isqrt fixups) per thread costs
        # more instructions than the rest of the kernel on 1-row launches.
        # warp0 alone walks the chain and publishes the four derived scalars
        # through s_lad; the other warps spend the wait issuing L2 prefetch
        # hints for this CTA's own P3 slice (register-free, so zero pressure
        # on the 64-register arms — the PRIME-LATE register loads below are
        # untouched and simply hit L2).  Values are bit-identical to the
        # per-thread derivation this replaces.
        if cutlass.const_expr(self.varlen):
            if tidx < cutlass.Int32(32):
                if short == cutlass.Int32(0):
                    # ---- aim ladder (cheap mirror) ----
                    # The ladder scalars steer the sampling rung only —
                    # exactness is schedule-invariant (retry/degen close every
                    # miss), so +-1 drift vs the host double form is allowed.
                    # Serial latency dominates (this chain sits in front of a
                    # barrier): runtime divides become MUFU.RCP multiplies and
                    # the isqrt fixup loops collapse to single steps (the f32
                    # sqrt of an exactly-representable int (6n <= 2^23) is
                    # within 1 of isqrt, so one correction per side suffices).
                    # Q (chunk ownership) stays exact — compile-time divisor.
                    x6 = cutlass.Int32(6) * n
                    ri = cutlass.Int32(cmath.sqrt(cutlass.Float32(x6)))
                    if ri * ri > x6:
                        ri = ri - cutlass.Int32(1)
                    if (ri + cutlass.Int32(1)) * (ri + cutlass.Int32(1)) <= x6:
                        ri = ri + cutlass.Int32(1)
                    r6 = ri
                    if x6 - ri * ri > ri:
                        r6 = ri + cutlass.Int32(1)
                    aim = aim_base
                    if r6 > aim:
                        aim = r6
                    if cutlass.const_expr(self.r_const > 1):
                        if aim < amin:
                            aim = amin
                    scap_c = cutlass.Int32(SCPB)  # SCAP == SCPB for gvr_main (proven identity)
                    if aim > (scap_c >> cutlass.Int32(1)):
                        aim = scap_c >> cutlass.Int32(1)
                    if aim < k:
                        aim = k
                    n4w = n >> cutlass.Int32(2)
                    # pair-sample gate: (n > SCAP or small_dense) and n4 >= 4;
                    # small_dense = k > 1024 and not big and n <= SCAP and n > 2k
                    # (k/big folded into the launch-constant sd_en flag).
                    gate = cutlass.Int32(0)
                    if n > scap_c:
                        gate = cutlass.Int32(1)
                    if sd_en != cutlass.Int32(0):
                        if n <= scap_c:
                            if n > (k << cutlass.Int32(1)):
                                gate = cutlass.Int32(1)
                    if n4w < cutlass.Int32(4):
                        gate = cutlass.Int32(0)
                    if gate != cutlass.Int32(0):
                        # sel = sfac*n // aim via rcp (sfac*n <= 2^24: f32-exact
                        # to the last unit; quotient error < 1 => +-1 drift)
                        sel = cutlass.Int32(
                            cutlass.Float32(sfac * n) * cute.arch.rcp_approx(cutlass.Float32(aim))
                        )
                        if sel < cutlass.Int32(256):
                            sel = cutlass.Int32(256)
                        nh = n >> cutlass.Int32(1)
                        if sel > nh:
                            sel = nh
                        pairs = sel >> cutlass.Int32(3)
                        if pairs < cutlass.Int32(1):
                            pairs = cutlass.Int32(1)
                        half = n4w >> cutlass.Int32(1)
                        if half < cutlass.Int32(1):
                            half = cutlass.Int32(1)
                        if pairs > half:
                            pairs = half
                        SS2 = cutlass.Int32(
                            cutlass.Float32(half) * cute.arch.rcp_approx(cutlass.Float32(pairs))
                        )
                        if SS2 < cutlass.Int32(1):
                            SS2 = cutlass.Int32(1)
                        SMP = cutlass.Int32(
                            cutlass.Float32(half) * cute.arch.rcp_approx(cutlass.Float32(SS2))
                        )
                        # sample-window guard: the P1 gather indexes up to
                        # ~SMP*SS2*2 f32x4 lines; keep SMP*SS2 <= half so the
                        # window never walks past the row (approx error is
                        # bounded by +1, one decrement closes it)
                        if SMP * SS2 > half:
                            SMP = SMP - cutlass.Int32(1)
                        if SMP < cutlass.Int32(1):
                            SMP = cutlass.Int32(1)
                        # TGT/TGT2: i64 products // n -> f32 mul + one rcp(n).
                        # aim/SMP/k/n are all f32-exact here (<= 2^20); the
                        # quotients are <= 8*aim ~ 2^16, so the approx error
                        # stays far below 1 unit — +-1 at worst on the floor.
                        rn_ = cute.arch.rcp_approx(cutlass.Float32(n))
                        smp8f = cutlass.Float32(SMP) * cutlass.Float32(8.0)
                        TGT = cutlass.Int32(cutlass.Float32(aim) * smp8f * rn_)
                        if TGT < cutlass.Int32(1):
                            TGT = cutlass.Int32(1)
                        TGT2 = cutlass.Int32(cutlass.Float32(k) * smp8f * rn_)
                        if TGT2 < cutlass.Int32(1):
                            TGT2 = cutlass.Int32(1)
                if tidx == cutlass.Int32(0):
                    s_lad[0] = SMP
                    s_lad[1] = SS2
                    s_lad[2] = TGT
                    s_lad[3] = TGT2
                    if cutlass.const_expr(self.prefill):
                        s_lead[0] = lead
            # Register-free L2 hints for the first U-batch of this CTA's own
            # P3 slice (clamped in-row): the data P3 touches first starts
            # flowing while warp0 walks the chain. Short rows clamp every
            # hint to the row's last line — harmless.
            # prefill: the base is shifted to col0, so clamp within the row's own
            # window [col0, ke) — n4-1 is the last full in-window float4.
            if cutlass.const_expr(self.prefill):
                plim4 = n4 - cutlass.Int32(1)
                if plim4 < cutlass.Int32(0):
                    plim4 = cutlass.Int32(0)
            else:
                plim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)
            for uu in cutlass.range_constexpr(U):
                # NOTE: names must not collide with the PRIME-LATE block's
                # i_/ic — the DSL kills inner-scope names at region exit and
                # a later same-name assignment inside a dynamic `if` trips
                # "is None prior to this if".
                pic = c0 + tidx + cutlass.Int32(uu * BLK)
                if pic >= c1:
                    pic = plim4
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    _prefetch_l2(x_addr + cutlass.Int64(pic) * cutlass.Int64(16))
                else:
                    _prefetch_l2(x_addr + cutlass.Int64(pic) * cutlass.Int64(8))
            cute.arch.barrier()  # publish s_lad (also covers the smem inits)
            SMP = s_lad[0]
            SS2 = s_lad[1]
            TGT = s_lad[2]
            TGT2 = s_lad[3]

        # ============ P1: sample prefetch (hint gather LAZY) =================
        atom128 = g2r_atom_f32(128, invariant=True)
        fsa = cute.make_rmem_tensor((4,), cutlass.Float32)
        fsb = cute.make_rmem_tensor((4,), cutlass.Float32)
        shas = cutlass.Int32(0)
        if tidx < SMP:
            shas = cutlass.Int32(1)
        if shas != cutlass.Int32(0):
            p4 = tidx * SS2 * cutlass.Int32(2)
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                ld_g_f32x4(atom128, x_addr, p4, fsa)
                ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fsb)
            else:
                ld_g_bf16x4(atom128, x_addr, p4, fsa)
                ld_g_bf16x4(atom128, x_addr, p4 + cutlass.Int32(1), fsb)
            if cutlass.const_expr(self.prefill):
                # only thread 0's float4 0 holds the <=3 lead lanes; substitute
                # lane 3 (a -inf would drive f2s_rz to INT_MIN and index the
                # sample histogram out of bounds).
                if tidx == cutlass.Int32(0):
                    ld_ = s_lead[0]
                    for q in cutlass.range_constexpr(3):
                        if cutlass.Int32(q) < ld_:
                            fsa[q] = fsa[3]

        # ============ P2: quantile rung from the sample ======================
        smn = cutlass.Float32(float("inf"))
        smx = cutlass.Float32(float("-inf"))
        if shas != cutlass.Int32(0):
            for t in cutlass.range_constexpr(4):
                smn = fmin_f32(smn, fsa[t])
                smx = fmax_f32(smx, fsa[t])
            for t in cutlass.range_constexpr(4):
                smn = fmin_f32(smn, fsb[t])
                smx = fmax_f32(smx, fsb[t])
        fma_ = cute.make_rmem_tensor((4,), cutlass.Float32)  # strided-tail pair bufs
        fmb_ = cute.make_rmem_tensor((4,), cutlass.Float32)
        j = tidx + cutlass.Int32(BLK)  # strided tail
        while j < SMP:
            p4 = j * SS2 * cutlass.Int32(2)
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                ld_g_f32x4(atom128, x_addr, p4, fma_)
                ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
            else:
                ld_g_bf16x4(atom128, x_addr, p4, fma_)
                ld_g_bf16x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
            for t in cutlass.range_constexpr(4):
                smn = fmin_f32(smn, fma_[t])
                smx = fmax_f32(smx, fma_[t])
            for t in cutlass.range_constexpr(4):
                smn = fmin_f32(smn, fmb_[t])
                smx = fmax_f32(smx, fmb_[t])
            j = j + cutlass.Int32(BLK)
        a0 = warp_min_u32(fkey(smn))
        c0m = warp_max_u32(fkey(smx))
        if lane == cutlass.Int32(0):
            s_wmn[tidx >> cutlass.Int32(5)] = a0
            s_wmx[tidx >> cutlass.Int32(5)] = c0m
        cute.arch.barrier()  # ---- barrier (sample redux publish) ----

        # PRIME-LATE prefetch block: strictly after the barrier.
        # prefill clamps to the last in-window float4 (see plim4 note above).
        if cutlass.const_expr(self.prefill):
            lim4 = n4 - cutlass.Int32(1)
            if lim4 < cutlass.Int32(0):
                lim4 = cutlass.Int32(0)
        else:
            lim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)
        pf = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(max(PFD, 1))]
        if cutlass.const_expr(self.pf):
            fullsl = cutlass.Int32(0)
            if (c1 - c0) >= cutlass.Int32(BLK * U):
                fullsl = cutlass.Int32(1)
            if fullsl != cutlass.Int32(0):  # prime, full slice
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    for uu in cutlass.range_constexpr(PFD):
                        ld_g_f32x4(atom128, x_addr, c0 + tidx + cutlass.Int32(uu * BLK), pf[uu])
                else:
                    for uu in cutlass.range_constexpr(PFD):
                        ld_g_bf16x4(atom128, x_addr, c0 + tidx + cutlass.Int32(uu * BLK), pf[uu])
            else:  # clamped prime
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    for uu in cutlass.range_constexpr(PFD):
                        i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                        ic = i_
                        if ic >= c1:
                            ic = lim4
                        ld_g_f32x4(atom128, x_addr, ic, pf[uu])
                else:
                    for uu in cutlass.range_constexpr(PFD):
                        i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                        ic = i_
                        if ic >= c1:
                            ic = lim4
                        ld_g_bf16x4(atom128, x_addr, ic, pf[uu])
            # asm prefetch site #1: gate (c1-c0)>=2*BLK*U && SMP>=160
            g1 = cutlass.Int32(0)
            if (c1 - c0) >= cutlass.Int32(2 * BLK * U):
                if SMP >= cutlass.Int32(160):
                    g1 = cutlass.Int32(1)
            if g1 != cutlass.Int32(0):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    for uu in cutlass.range_constexpr(PFD, U):
                        _prefetch_l2(
                            x_addr
                            + cutlass.Int64(c0 + tidx + cutlass.Int32(uu * BLK)) * cutlass.Int64(16)
                        )
                else:
                    for uu in cutlass.range_constexpr(PFD, U):
                        _prefetch_l2(
                            x_addr
                            + cutlass.Int64(c0 + tidx + cutlass.Int32(uu * BLK)) * cutlass.Int64(8)
                        )
        if cutlass.const_expr((not self.pf) and (not self.split)):
            fullsl = cutlass.Int32(0)
            if (c1 - c0) >= cutlass.Int32(BLK * U):
                fullsl = cutlass.Int32(1)
            if fullsl != cutlass.Int32(0):  # prefetch site #2
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    for uu in cutlass.range_constexpr(U):
                        _prefetch_l2(
                            x_addr
                            + cutlass.Int64(c0 + tidx + cutlass.Int32(uu * BLK)) * cutlass.Int64(16)
                        )
                else:
                    for uu in cutlass.range_constexpr(U):
                        _prefetch_l2(
                            x_addr
                            + cutlass.Int64(c0 + tidx + cutlass.Int32(uu * BLK)) * cutlass.Int64(8)
                        )
            else:
                if SMP > cutlass.Int32(0):  # prefetch site #3
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        for uu in cutlass.range_constexpr(U):
                            i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                            ic = i_
                            if ic >= c1:
                                ic = lim4
                            _prefetch_l2(x_addr + cutlass.Int64(ic) * cutlass.Int64(16))
                    else:
                        for uu in cutlass.range_constexpr(U):
                            i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                            ic = i_
                            if ic >= c1:
                                ic = lim4
                            _prefetch_l2(x_addr + cutlass.Int64(ic) * cutlass.Int64(8))

        # cross-warp sample reduce
        av = cutlass.Uint32(0xFFFFFFFF)
        cv = cutlass.Uint32(0)
        if lane < cutlass.Int32(NW):
            av = s_wmn[lane]
            cv = s_wmx[lane]
        SMIN = invkey(warp_min_u32(av))
        SMAX = invkey(warp_max_u32(cv))

        GMIN = cutlass.Float32(SENT_LO)  # sentinels
        GMAX = cutlass.Float32(SENT_HI)
        T = cutlass.Float32(_NEG_INF)
        HIC = cutlass.Float32(_NEG_INF)
        w = cutlass.Float32(0.0)
        sok = cutlass.Int32(0)
        if SMP > cutlass.Int32(0):
            if SMAX > SMIN:
                sok = cutlass.Int32(1)
        if sok != cutlass.Int32(0):  # sample histogram
            w = (SMAX - SMIN) * cutlass.Float32(1.0 / 256.0)
            # rcp.approx.ftz.f32 = the CUDA arm's --use_fast_math 1.0f/w
            # (bare MUFU.RCP, no Newton refinement) — bitwise-aligned scale
            sc_s = cute.arch.rcp_approx(w)
            if shas != cutlass.Int32(0):
                for t in cutlass.range_constexpr(4):
                    bq = f2s_rz((fsa[t] - SMIN) * sc_s)
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                for t in cutlass.range_constexpr(4):
                    bq = f2s_rz((fsb[t] - SMIN) * sc_s)
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
            j = tidx + cutlass.Int32(BLK)  # tail re-loads
            while j < SMP:
                p4 = j * SS2 * cutlass.Int32(2)
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    ld_g_f32x4(atom128, x_addr, p4, fma_)
                    ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
                else:
                    ld_g_bf16x4(atom128, x_addr, p4, fma_)
                    ld_g_bf16x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
                for t in cutlass.range_constexpr(4):
                    bq = f2s_rz((fma_[t] - SMIN) * sc_s)
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                for t in cutlass.range_constexpr(4):
                    bq = f2s_rz((fmb_[t] - SMIN) * sc_s)
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                j = j + cutlass.Int32(BLK)
        cute.arch.barrier()  # ---- barrier (sample histogram) ----
        # triple-target ZERO scan: TGT / TGT2 / 2*TGT
        # (THREE = SHD || gated-SPLIT)
        scan_cross0(
            s_hist,
            TGT,
            tidx,
            s_res,
            TGT2,
            TGT * cutlass.Int32(2),
            s_hist,
            nb=NBS,
            zero=True,
            two=True,
            three=(self.shd or self.tshg),
        )
        cute.arch.barrier()  # ---- barrier (scan publish) ----

        tot0 = s_res[RES_TOT]
        b1v = s_res[RES_B]
        if sok != cutlass.Int32(0):
            if tot0 >= TGT:
                T = _fmaf(cutlass.Float32(b1v), w, SMIN)
        Trung = T  # snapshot
        needg = cutlass.Int32(1)  # degenerate sample
        if T > cutlass.Float32(_NEG_INF):
            needg = cutlass.Int32(0)
        if needg != cutlass.Int32(0):
            if cutlass.const_expr(not self.hint_free):
                GMIN, GMAX = gather_hint(
                    x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk=BLK, kpt=KPT
                )  # 2 barriers inside
            T = GMIN
        if sok != cutlass.Int32(0):  # HIC tighten
            if tot0 >= TGT:
                b2v = s_res[RES_B2]
                if b2v >= cutlass.Int32(0):
                    Tk = _fmaf(cutlass.Float32(b2v), w, SMIN)
                    anch = T
                    if cutlass.const_expr(not self.split):
                        anch = fmin_f32(T, Trung)
                    d_ = fmax_f32(Tk - anch, cutlass.Float32(0.0))
                    HIC = fmax_f32(
                        _fmaf(cutlass.Float32(4.0), d_, T), _fmaf(cutlass.Float32(8.0), w, T)
                    )
        if cutlass.const_expr(self.shd or self.tshg):  # TSH floor (+gated SPLIT)
            if tidx == cutlass.Int32(0):
                t5 = cutlass.Float32(_NEG_INF)
                if sok != cutlass.Int32(0):
                    if tot0 >= TGT * cutlass.Int32(2):
                        b3v = s_res[RES_B3]
                        if b3v >= cutlass.Int32(0):
                            if T > GMIN:
                                T3 = _fmaf(cutlass.Float32(b3v), w, SMIN)
                                if T3 < T:
                                    t5 = T3
                s_tsh[0] = t5

        if cutlass.const_expr(self.tshg):
            # TSH-FLOOR STAGING: SPLIT has no retry ladder, so a rung
            # overshoot (count(>=T) < k) used to hand the LAST CTA a
            # single-CTA whole-row narrowing.  Stage at the sample's
            # rank-(2*TGT) floor instead: staged population ~aim -> ~2*aim,
            # and the merged histogram contains the k-crossing whenever
            # count(>=TSH) >= k.  TSH miss falls to GMIN/degen unchanged.
            cute.arch.barrier()
            t5s = s_tsh[0]
            # varlen: per-row runtime gate (tsh_run == 1 always in legacy
            # mode, so legacy codegen semantics are unchanged)
            if tsh_run != cutlass.Int32(0):
                if t5s > cutlass.Float32(_NEG_INF):
                    if t5s < T:
                        T = t5s

        # ============ attempt loop — MUST NOT unroll ============
        listN = cutlass.Int32(0)
        above = cutlass.Int32(0)
        m = cutlass.Int32(0)
        need = cutlass.Int32(0)
        B = cutlass.Int32(0)
        SC = cutlass.Float32(1.0)
        TF = T
        complete = cutlass.Int32(0)
        valid = cutlass.Int32(0)
        fromg = cutlass.Int32(0)
        alive = cutlass.Int32(1)

        fr = [
            cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(max(U - PFD, 1))
        ]  # explicit batch
        att = cutlass.Int32(0)
        running = cutlass.Int32(1)
        while running != cutlass.Int32(0):
            if cutlass.const_expr(not self.split):  # SPLIT never retries (NATT=1)
                if att > cutlass.Int32(0):  # retry reset
                    if cutlass.const_expr(self.pf):
                        # exactness: re-prime pf[] (holds stale roll data)
                        fullsl = cutlass.Int32(0)
                        if (c1 - c0) >= cutlass.Int32(BLK * U):
                            fullsl = cutlass.Int32(1)
                        if fullsl != cutlass.Int32(0):
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                for uu in cutlass.range_constexpr(PFD):
                                    ld_g_f32x4(
                                        atom128, x_addr, c0 + tidx + cutlass.Int32(uu * BLK), pf[uu]
                                    )
                            else:
                                for uu in cutlass.range_constexpr(PFD):
                                    ld_g_bf16x4(
                                        atom128, x_addr, c0 + tidx + cutlass.Int32(uu * BLK), pf[uu]
                                    )
                        else:
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                for uu in cutlass.range_constexpr(PFD):
                                    i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                                    ic = i_
                                    if ic >= c1:
                                        ic = lim4
                                    ld_g_f32x4(atom128, x_addr, ic, pf[uu])
                            else:
                                for uu in cutlass.range_constexpr(PFD):
                                    i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                                    ic = i_
                                    if ic >= c1:
                                        ic = lim4
                                    ld_g_bf16x4(atom128, x_addr, ic, pf[uu])
                    if tidx < cutlass.Int32(NBS):
                        s_hist[tidx] = cutlass.Int32(0)
                    if tidx == cutlass.Int32(0):
                        s_scal[0] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (retry reset) ----

            TF = T  # window
            hi = fmax_f32(GMAX, T)
            if HIC > T:
                if HIC < hi:
                    hi = HIC
            WD = (hi - T) * cutlass.Float32(1.0 / 256.0)
            wdok = cutlass.Int32(0)
            if WD > cutlass.Float32(0.0):
                wdok = cutlass.Int32(1)
            if wdok == cutlass.Int32(0):
                WD = cutlass.Float32(1e-30)
            # CUDA compiles its own `1.0f / WD` here to a bare MUFU.RCP
            # (approximate); div.rn's dependent rcp+Newton+CALL chain
            # serializes the attempt prologue. blk==512 ONLY: the
            # (256,8,4,·) family keeps the original div.rn spelling below.
            if cutlass.const_expr(self.blk == 512):
                SC = cute.arch.rcp_approx(WD)
            else:
                SC = cutlass.Float32(1.0) / WD

            # ---- P3 row pass ----
            span = c1 - c0
            step = cutlass.Int32(BLK * U)
            nFull = cutlass.Int32(0)
            rem = cutlass.Int32(0)
            if span > cutlass.Int32(0):  # peel
                nFull = span // step
                rem = span - nFull * step
            # _pin_i32: the isfull peel predicate reads nFull every tile iter;
            # unpinned, NVVM re-derives the whole ld.param+shr/sel div chain
            # at the loop head
            nFull = _pin_i32(nFull)
            nIt = nFull
            if rem > cutlass.Int32(0):
                nIt = nIt + cutlass.Int32(1)
            # _pin_i32: stop NVVM re-deriving the ceil-div bound (ld.param n +
            # shr/sel chain) inside the tile-loop condition region per iter
            nIt = _pin_i32(nIt)

            it = cutlass.Int32(0)
            while it < nIt:
                i0 = c0 + it * step + tidx
                M = cutlass.Int32(0)
                isfull = cutlass.Int32(0)
                if it < nFull:
                    isfull = cutlass.Int32(1)
                if isfull != cutlass.Int32(0):  # full body
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        for uu in cutlass.range_constexpr(PFD, U):
                            ld_g_f32x4(atom128, x_addr, i0 + cutlass.Int32(uu * BLK), fr[uu - PFD])
                    else:
                        for uu in cutlass.range_constexpr(PFD, U):
                            ld_g_bf16x4(atom128, x_addr, i0 + cutlass.Int32(uu * BLK), fr[uu - PFD])
                    for uu in cutlass.range_constexpr(U):
                        if cutlass.const_expr(uu < PFD):
                            vv = pf[uu]
                        else:
                            vv = fr[uu - PFD]
                        for q in cutlass.range_constexpr(4):
                            M = M | (cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q))
                else:  # partial body
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        for uu in cutlass.range_constexpr(PFD, U):
                            i_ = i0 + cutlass.Int32(uu * BLK)
                            ic = i_
                            if ic >= c1:
                                ic = lim4  # clamped address
                            ld_g_f32x4(atom128, x_addr, ic, fr[uu - PFD])
                    else:
                        for uu in cutlass.range_constexpr(PFD, U):
                            i_ = i0 + cutlass.Int32(uu * BLK)
                            ic = i_
                            if ic >= c1:
                                ic = lim4  # clamped address
                            ld_g_bf16x4(atom128, x_addr, ic, fr[uu - PFD])
                    for uu in cutlass.range_constexpr(U):
                        if cutlass.const_expr(uu < PFD):
                            vv = pf[uu]
                        else:
                            vv = fr[uu - PFD]
                        i_ = i0 + cutlass.Int32(uu * BLK)
                        okq = cutlass.Int32(0)
                        if i_ < c1:
                            okq = cutlass.Int32(1)
                        if okq != cutlass.Int32(0):  # ok-gated (+inf-pad escape)
                            for q in cutlass.range_constexpr(4):
                                M = M | (cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q))
                if cutlass.const_expr(self.prefill):
                    # the <=lead lead lanes live only in bits 0..lead-1 of the
                    # i0==0 tile (float4 0, thread 0, part 0); clear them so the
                    # reservation, survivor walk and re-reads never see them.
                    if i0 == cutlass.Int32(0):
                        M = M & (cutlass.Int32(-1) << s_lead[0])
                # prefetch roll-forward BEFORE reservation/walk
                if cutlass.const_expr(self.pf):
                    hasnext = cutlass.Int32(0)
                    if it + cutlass.Int32(1) < nIt:
                        hasnext = cutlass.Int32(1)
                    if hasnext != cutlass.Int32(0):
                        j0 = i0 + step
                        infull = cutlass.Int32(0)  # warp-uniform peel
                        if it + cutlass.Int32(1) < nFull:
                            infull = cutlass.Int32(1)
                        if infull != cutlass.Int32(0):
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                for uu in cutlass.range_constexpr(PFD):
                                    ld_g_f32x4(
                                        atom128, x_addr, j0 + cutlass.Int32(uu * BLK), pf[uu]
                                    )
                            else:
                                for uu in cutlass.range_constexpr(PFD):
                                    ld_g_bf16x4(
                                        atom128, x_addr, j0 + cutlass.Int32(uu * BLK), pf[uu]
                                    )
                        else:
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                for uu in cutlass.range_constexpr(PFD):
                                    j_ = j0 + cutlass.Int32(uu * BLK)
                                    jc = j_
                                    if jc >= c1:
                                        jc = lim4
                                    ld_g_f32x4(atom128, x_addr, jc, pf[uu])
                            else:
                                for uu in cutlass.range_constexpr(PFD):
                                    j_ = j0 + cutlass.Int32(uu * BLK)
                                    jc = j_
                                    if jc >= c1:
                                        jc = lim4
                                    ld_g_bf16x4(atom128, x_addr, jc, pf[uu])
                # warp-aggregated reservation
                cnt = cutlass.Int32(popc(M))
                inc = warp_incl_scan_add(cnt, lane)
                bpos = cutlass.Int32(0)
                if lane == cutlass.Int32(31):
                    if inc != cutlass.Int32(0):
                        bpos = atomic_add_cta(s_scal.iterator + 0, inc)
                pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) + (inc - cnt)
                # survivor bit-walk, software-pipelined ONE deep;
                # reload X[idx] — do NOT hold the U float4s (spills)
                if M != cutlass.Int32(0):
                    bp = ffs_m1(M)
                    M = M & (M - cutlass.Int32(1))
                    idx = (
                        (i0 + (bp >> cutlass.Int32(2)) * cutlass.Int32(BLK)) << cutlass.Int32(2)
                    ) + (bp & cutlass.Int32(3))
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        if cutlass.const_expr(self.vstg and self.blk == 512):
                            xv = _ldg_f32_rs(x_addr, idx, x4_pin)
                        else:
                            xv = ldg_f32(x_addr, idx)
                    else:
                        if cutlass.const_expr(self.vstg and self.blk == 512):
                            xv = ldg_bf16_rs(x_addr, idx, x4_pin)
                        else:
                            xv = ldg_bf16(x_addr, idx)
                    while M != cutlass.Int32(0):
                        bp2 = ffs_m1(M)
                        M = M & (M - cutlass.Int32(1))
                        idx2 = (
                            (i0 + (bp2 >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                            << cutlass.Int32(2)
                        ) + (bp2 & cutlass.Int32(3))
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            if cutlass.const_expr(self.vstg and self.blk == 512):
                                xv2 = _ldg_f32_rs(x_addr, idx2, x4_pin)
                            else:
                                xv2 = ldg_f32(x_addr, idx2)
                        else:
                            if cutlass.const_expr(self.vstg and self.blk == 512):
                                xv2 = ldg_bf16_rs(x_addr, idx2, x4_pin)
                            else:
                                xv2 = ldg_bf16(x_addr, idx2)
                        pos = self._emitc(
                            xv, idx, pos, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2
                        )
                        idx = idx2
                        xv = xv2
                    pos = self._emitc(
                        xv, idx, pos, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2
                    )
                it = it + cutlass.Int32(1)
            # scalar tail, part 0 only
            i = tidx
            while i < tailn:
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    x = ldg_f32(x_addr, tail0 + i)
                else:
                    x = ldg_bf16(x_addr, tail0 + i)
                if x >= TF:
                    post = atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                    post = self._emitc(
                        x, tail0 + i, post, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2
                    )
                i = i + cutlass.Int32(BLK)
            cute.arch.barrier()  # ---- barrier (row pass) ----
            myn = s_scal[0]

            if cutlass.const_expr(self.split):
                # ---- SLAB HAND-OFF; exactly ONE attempt ----
                if tidx == cutlass.Int32(0):
                    pgo = cute.make_ptr(
                        cutlass.Int32,
                        goff_addr + row64 * cutlass.Int64(4),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    s_scal[3] = cutlass.Int32(cute.arch.atomic_add(pgo, myn))
                cute.arch.barrier()  # ---- barrier (slab offset) ----
                base = s_scal[3]
                if myn <= cutlass.Int32(SCPB):  # coalesced publish
                    i = tidx
                    while i < myn:
                        p = base + i
                        if p < cutlass.Int32(GCAP__main):
                            _st_g_u64(gbuf_row + cutlass.Int64(p) * cutlass.Int64(8), s_cbuf2[i])
                        i = i + cutlass.Int32(BLK)
                else:  # overflow re-sweep
                    if tidx == cutlass.Int32(0):
                        s_scal[0] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (overflow reset) ----
                    lo2 = c0 << cutlass.Int32(2)
                    hi2 = c1 << cutlass.Int32(2)
                    i = lo2 + tidx
                    while i < hi2:
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            x = ldg_f32(x_addr, i)
                        else:
                            x = ldg_bf16(x_addr, i)
                        if x >= TF:
                            pq = atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                            p = base + pq
                            if p < cutlass.Int32(GCAP__main):
                                _st_g_u64(
                                    gbuf_row + cutlass.Int64(p) * cutlass.Int64(8),
                                    (cutlass.Uint64(cutlass.Uint32(i)) << cutlass.Uint64(32))
                                    | cutlass.Uint64(u32_of_f32(x)),
                                )
                        i = i + cutlass.Int32(BLK)
                    i = tidx  # true tail
                    while i < tailn:
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            x = ldg_f32(x_addr, tail0 + i)
                        else:
                            x = ldg_bf16(x_addr, tail0 + i)
                        if x >= TF:
                            pq = atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                            p = base + pq
                            if p < cutlass.Int32(GCAP__main):
                                _st_g_u64(
                                    gbuf_row + cutlass.Int64(p) * cutlass.Int64(8),
                                    (
                                        cutlass.Uint64(cutlass.Uint32(tail0 + i))
                                        << cutlass.Uint64(32)
                                    )
                                    | cutlass.Uint64(u32_of_f32(x)),
                                )
                        i = i + cutlass.Int32(BLK)
                cute.arch.barrier()  # ---- barrier (slab publish) ----
                if tidx == cutlass.Int32(0):  # release + RMW
                    threadfence_gpu()
                    pdon = cute.make_ptr(
                        cutlass.Int64,
                        gdon_addr + row64 * cutlass.Int64(8),
                        cute.AddressSpace.gmem,
                        assumed_align=8,
                    )
                    s_pk[0] = atomic_add_u64_gpu(pdon, cutlass.Int64(1 << 32) + cutlass.Int64(myn))
                cute.arch.barrier()  # ---- barrier (arrival word) ----
                pk = s_pk[0]
                alive = cutlass.Int32(0)  # last-CTA test
                if cutlass.Int32(pk >> cutlass.Int64(32)) == R - cutlass.Int32(1):
                    alive = cutlass.Int32(1)
                if alive != cutlass.Int32(0):
                    threadfence_gpu()  # acquire
                    if tidx == cutlass.Int32(0):  # ZERO-RESTORE
                        _st_g_u32(goff_addr + row64 * cutlass.Int64(4), cutlass.Int32(0))
                        _st_g_u64(gdon_addr + row64 * cutlass.Int64(8), cutlass.Uint64(0))
                    total = cutlass.Int32(pk & cutlass.Int64(0xFFFFFFFF)) + myn
                    if total <= cutlass.Int32(GCAP__main):  # one-pass consume
                        listN = total
                        if total > cutlass.Int32(SCPB):
                            fromg = cutlass.Int32(1)
                        i = tidx
                        while i < listN:
                            gvx, gvy = _ldcg_v2_i32(gbuf_row + cutlass.Int64(i) * cutlass.Int64(8))
                            if fromg == cutlass.Int32(0):
                                s_cbuf2[i] = (
                                    cutlass.Uint64(cutlass.Uint32(gvy)) << cutlass.Uint64(32)
                                ) | cutlass.Uint64(cutlass.Uint32(gvx))
                            bq = f2s_rz((f32_of_i32(gvx) - TF) * SC)
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                            # resultless red off the pinned hist base
                            _red_shared_add1(hb_pin + (bq << cutlass.Int32(2)))
                            i = i + cutlass.Int32(BLK)
                        cute.arch.barrier()  # ---- barrier (slab histogram) ----
                        scan_cross0(
                            s_hist,
                            k,
                            tidx,
                            s_res,
                            cutlass.Int32(0),
                            cutlass.Int32(0),
                            s_hist,
                            nb=NBS,
                            zero=False,
                        )
                        cute.arch.barrier()  # ---- barrier (scan publish) ----
                        if s_res[RES_TOT] >= k:
                            valid = cutlass.Int32(1)
                            complete = cutlass.Int32(1)
                            above = s_res[RES_ABOVE]
                            m = s_res[RES_M]
                            need = k - above
                            B = s_res[RES_B]
                running = cutlass.Int32(0)  # break (NATT==1)
            else:
                # ---- non-split verify + rung ladder ----
                scan_cross0(
                    s_hist,
                    k,
                    tidx,
                    s_res,
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    s_hist,
                    nb=NBS,
                    zero=False,
                )
                cute.arch.barrier()  # ---- barrier (verify scan) ----
                tot = s_res[RES_TOT]
                acc = cutlass.Int32(0)
                if tot >= k:
                    acc = cutlass.Int32(1)
                if acc != cutlass.Int32(0):  # accept
                    valid = cutlass.Int32(1)
                    complete = cutlass.Int32(0)
                    if myn <= cutlass.Int32(SCPB):
                        complete = cutlass.Int32(1)
                    listN = myn
                    above = s_res[RES_ABOVE]
                    m = s_res[RES_M]
                    need = k - above
                    B = s_res[RES_B]
                    running = cutlass.Int32(0)
                else:
                    if att == cutlass.Int32(NATT - 1):  # ladder exhausted
                        running = cutlass.Int32(0)
                    else:
                        tshtaken = cutlass.Int32(0)  # TSH retry
                        if cutlass.const_expr(self.shd):
                            if att == cutlass.Int32(0):
                                T5 = s_tsh[0]
                                if T5 > cutlass.Float32(_NEG_INF):
                                    if T5 < TF:
                                        T = T5
                                        tshtaken = cutlass.Int32(1)
                        if tshtaken != cutlass.Int32(0):
                            cute.arch.barrier()  # ---- barrier (TSH retry) ----
                        else:
                            # LAZY GATHER (sentinel equality flag)
                            if GMIN == cutlass.Float32(SENT_LO):
                                if cutlass.const_expr(not self.hint_free):
                                    GMIN, GMAX = gather_hint(
                                        x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk=BLK, kpt=KPT
                                    )
                            floorhit = cutlass.Int32(1)
                            if T > GMIN:
                                floorhit = cutlass.Int32(0)
                            if floorhit != cutlass.Int32(0):
                                running = cutlass.Int32(0)
                            else:
                                T = GMIN
                                cute.arch.barrier()  # ---- barrier (floor retry) ----
            att = att + cutlass.Int32(1)

        # ============ classification ============
        if alive != cutlass.Int32(0):
            whole = cutlass.Int32(0)
            if valid != cutlass.Int32(0):
                if need >= m:
                    whole = cutlass.Int32(1)
            lim1 = above
            if whole != cutlass.Int32(0):
                lim1 = above + m
            degen = cutlass.Int32(0)
            if valid == cutlass.Int32(0):
                degen = cutlass.Int32(1)
            if m > cutlass.Int32(CMPB):
                degen = cutlass.Int32(1)
            mc = cutlass.Int32(0)
            if degen == cutlass.Int32(0):
                mc = m

            if degen == cutlass.Int32(0):
                # ---- P5 cursor emit ----
                if complete != cutlass.Int32(0):
                    i = tidx
                    while i < listN:
                        idv = cutlass.Int32(0)
                        bq = cutlass.Int32(0)
                        xv = cutlass.Float32(0.0)
                        if cutlass.const_expr(self.vstg):
                            vx = cutlass.Int32(0)
                            vy = cutlass.Int32(0)
                            if cutlass.const_expr(self.split):
                                if fromg != cutlass.Int32(0):
                                    vx, vy = _ldcg_v2_i32(
                                        gbuf_row + cutlass.Int64(i) * cutlass.Int64(8)
                                    )
                                else:
                                    pk64 = s_cbuf2[i]
                                    vx = cutlass.Int32(
                                        cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                    )
                                    vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                            else:
                                pk64 = s_cbuf2[i]
                                vx = cutlass.Int32(
                                    cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                )
                                vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                            xv = f32_of_i32(vx)
                            idv = vy
                            bq = f2s_rz((xv - TF) * SC)
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                        else:
                            wpk = cutlass.Uint32(s_cbuf[i])
                            idv = cutlass.Int32(wpk & cutlass.Uint32(IDXM__main))
                            bq = cutlass.Int32(wpk >> cutlass.Uint32(IDXB__main))
                        if bq >= B:
                            p = atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                            if p < lim1:
                                # prefill: staged idx are in the col0 frame; the
                                # local output frame is relative to ks = col0+lead.
                                if cutlass.const_expr(self.prefill):
                                    out_row[p] = idv - s_lead[0]
                                else:
                                    out_row[p] = idv
                            else:
                                if whole == cutlass.Int32(0):
                                    q2 = p - above
                                    if q2 < cutlass.Int32(CMPB):
                                        if cutlass.const_expr(self.vstg):
                                            kk = fkey(xv)
                                        else:
                                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                                kk = fkey(ldg_f32(x_addr, idv))
                                            else:
                                                kk = fkey(ldg_bf16(x_addr, idv))
                                        s_ck64[q2] = (
                                            cutlass.Uint64(kk) << cutlass.Uint64(32)
                                        ) | cutlass.Uint64(cutlass.Uint32(idv))
                        i = i + cutlass.Int32(BLK)
                else:
                    # collect overflow: scalar re-sweep, exact tail remap —
                    # zero extra live registers by design
                    lo2 = c0 << cutlass.Int32(2)
                    hi2 = c1 << cutlass.Int32(2)
                    i0_ = lo2 + tidx
                    while i0_ < hi2 + tailn:
                        i_ = i0_
                        if i0_ >= hi2:
                            i_ = tail0 + (i0_ - hi2)
                        masked = cutlass.Int32(0)
                        if cutlass.const_expr(self.prefill):
                            # skip the <=lead lead lanes (col0-frame positions
                            # 0..lead-1 hold the previous request's finite logits)
                            if i_ < s_lead[0]:
                                masked = cutlass.Int32(1)
                        if masked == cutlass.Int32(0):
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                x = ldg_f32(x_addr, i_)
                            else:
                                x = ldg_bf16(x_addr, i_)
                            if x >= TF:
                                bq = f2s_rz((x - TF) * SC)
                                if bq > cutlass.Int32(NBS - 1):
                                    bq = cutlass.Int32(NBS - 1)
                                if bq >= B:
                                    p = atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                                    if p < lim1:
                                        if cutlass.const_expr(self.prefill):
                                            out_row[p] = i_ - s_lead[0]
                                        else:
                                            out_row[p] = i_
                                    else:
                                        if whole == cutlass.Int32(0):
                                            q2 = p - above
                                            if q2 < cutlass.Int32(CMPB):
                                                s_ck64[q2] = (
                                                    cutlass.Uint64(fkey(x)) << cutlass.Uint64(32)
                                                ) | cutlass.Uint64(cutlass.Uint32(i_))
                        i0_ = i0_ + cutlass.Int32(BLK)

                # ---- P6 refine ----
                if whole == cutlass.Int32(0):
                    cute.arch.barrier()  # ---- barrier (emit done) ----
                    if mc <= cutlass.Int32(QUADC_CLUS__main):  # O(mc^2) rank
                        mc2 = mc & cutlass.Int32(~1)
                        i = tidx
                        while i < mc:
                            # NOTE: values crossing a dynamic-while region are
                            # re-wrapped SIGNED by the DSL — every u64 compare
                            # must re-assert Uint64 at the USE site.
                            u64v = s_ck64[i]
                            r_ = cutlass.Int32(0)
                            jq = cutlass.Int32(0)
                            while jq < mc2:  # ulonglong2 16B reads
                                vlo, vhi = _lds_v2_u64(ck_addr + jq * cutlass.Int32(8))
                                r_ = (
                                    r_
                                    + cutlass.Int32(vlo > cutlass.Uint64(u64v))
                                    + cutlass.Int32(vhi > cutlass.Uint64(u64v))
                                )
                                jq = jq + cutlass.Int32(2)
                            if mc2 < mc:  # odd tail
                                r_ = r_ + cutlass.Int32(
                                    cutlass.Uint64(s_ck64[mc2]) > cutlass.Uint64(u64v)
                                )
                            if r_ < need:
                                idv6 = cutlass.Int32(
                                    cutlass.Uint32(
                                        cutlass.Uint64(u64v) & cutlass.Uint64(0xFFFFFFFF)
                                    )
                                )
                                if cutlass.const_expr(self.prefill):
                                    idv6 = idv6 - s_lead[0]
                                out_row[above + r_] = idv6
                            i = i + cutlass.Int32(BLK)
                    else:
                        # key-space narrowing over ck64
                        if tidx == cutlass.Int32(0):
                            s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                            s_kmm[1] = cutlass.Uint32(0)
                        if tidx < cutlass.Int32(NBS):  # cleared ONCE
                            s_hist[tidx] = cutlass.Int32(0)
                        cute.arch.barrier()  # ---- barrier (narrowing init) ----
                        i = tidx
                        while i < mc:
                            kk = cutlass.Uint32(s_ck64[i] >> cutlass.Uint64(32))
                            atomic_min_cta(s_kmm.iterator + 0, kk)
                            atomic_max_cta(s_kmm.iterator + 1, kk)
                            i = i + cutlass.Int32(BLK)
                        cute.arch.barrier()  # ---- barrier (key range) ----
                        rlo = s_kmm[0]
                        rhi = s_kmm[1]
                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                        aboveC = cutlass.Int32(0)
                        needC = need
                        mm = mc
                        brk = cutlass.Int32(0)
                        lev = cutlass.Int32(0)
                        while brk == cutlass.Int32(0):  # <=6 levels
                            if needC == mm:
                                ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                aboveC = aboveC + mm
                                needC = cutlass.Int32(0)
                                brk = cutlass.Int32(1)
                            elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                brk = cutlass.Int32(1)
                            elif lev >= cutlass.Int32(6):
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                brk = cutlass.Int32(1)
                            else:
                                d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                b2_ = cutlass.Int32(32) - clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1))
                                )
                                sh2 = b2_ - cutlass.Int32(self.lb)
                                if sh2 < cutlass.Int32(0):
                                    sh2 = cutlass.Int32(0)
                                sh2u = cutlass.Uint32(sh2)
                                i = tidx
                                while i < mc:  # re-bin
                                    uq = cutlass.Uint32(s_ck64[i] >> cutlass.Uint64(32))
                                    if uq >= cutlass.Uint32(rlo):
                                        if uq <= cutlass.Uint32(rhi):
                                            du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                            if du > cutlass.Uint32(NBS - 1):
                                                du = cutlass.Uint32(NBS - 1)
                                            atomic_add_cta(
                                                s_hist.iterator + cutlass.Int32(du),
                                                cutlass.Int32(1),
                                            )
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()  # ---- barrier (level hist) ----
                                scan_cross0(
                                    s_hist,
                                    needC,
                                    tidx,
                                    s_res,
                                    cutlass.Int32(0),
                                    cutlass.Int32(0),
                                    s_hist,
                                    nb=NBS,
                                    zero=True,
                                )
                                cute.arch.barrier()  # ---- barrier (level scan) ----
                                aboveC = aboveC + s_res[RES_ABOVE]
                                needC = needC - s_res[RES_ABOVE]
                                mm = s_res[RES_M]
                                sB = s_res[RES_B]
                                nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                                if sB != cutlass.Int32(NBS - 1):
                                    rhi = nlo + ((cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1))
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        if tidx == cutlass.Int32(0):
                            s_scal[1] = cutlass.Int32(0)
                            s_scal[2] = cutlass.Int32(0)
                        cute.arch.barrier()  # ---- barrier (emit counters) ----
                        it2 = (mc + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                        it = cutlass.Int32(0)
                        while it < it2:  # ballot emit
                            i = it * cutlass.Int32(BLK) + tidx
                            p1 = cutlass.Int32(0)
                            p2 = cutlass.Int32(0)
                            idv = cutlass.Int32(0)
                            if i < mc:
                                w64 = s_ck64[i]
                                iu = cutlass.Int64(cutlass.Uint32(w64 >> cutlass.Uint64(32)))
                                idv = cutlass.Int32(
                                    cutlass.Uint32(w64 & cutlass.Uint64(0xFFFFFFFF))
                                )
                                if iu > ethr:
                                    p1 = cutlass.Int32(1)
                                if iu == ethr:
                                    p2 = cutlass.Int32(1)
                            # staged idx are col0-frame; shift to the ks-relative
                            # local output frame (p1=p2=0 for i>=mc, so the -lead
                            # on the idv=0 default is never emitted).
                            if cutlass.const_expr(self.prefill):
                                idv = idv - s_lead[0]
                            self._ballot_pair_emit(
                                p1,
                                p2,
                                idv,
                                above,
                                aboveC,
                                above + aboveC,
                                needC,
                                out_row,
                                s_scal,
                                lane,
                            )
                            it = it + cutlass.Int32(1)
            else:
                dga = cutlass.Int32(0)  # gate: valid && complete
                if valid != cutlass.Int32(0):
                    if complete != cutlass.Int32(0):
                        dga = cutlass.Int32(1)
                if dga != cutlass.Int32(0):
                    # ---- degen A: narrowing over STAGED candidates ----
                    rlo = cutlass.Uint32(0)
                    rhi = cutlass.Uint32(0xFFFFFFFF)
                    above2 = cutlass.Int32(0)
                    need2 = k
                    m2 = listN
                    ethr = cutlass.Int64(0)
                    tie_m = cutlass.Int32(1)
                    if tidx < cutlass.Int32(NBS):
                        s_hist[tidx] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (degen A init) ----
                    brk = cutlass.Int32(0)
                    lev = cutlass.Int32(0)
                    while brk == cutlass.Int32(0):  # <=8 levels
                        if need2 == m2:
                            ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                            above2 = above2 + m2
                            need2 = cutlass.Int32(0)
                            tie_m = cutlass.Int32(0)
                            brk = cutlass.Int32(1)
                        elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        elif lev >= cutlass.Int32(8):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        else:
                            d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                            b2_ = cutlass.Int32(32) - clz_i32(cutlass.Int32(d2 | cutlass.Uint32(1)))
                            sh2 = b2_ - cutlass.Int32(self.lb)
                            if sh2 < cutlass.Int32(0):
                                sh2 = cutlass.Int32(0)
                            sh2u = cutlass.Uint32(sh2)
                            i = tidx
                            while i < listN:
                                uq = cutlass.Uint32(0)
                                if cutlass.const_expr(self.vstg):
                                    vx = cutlass.Int32(0)
                                    vy = cutlass.Int32(0)
                                    if cutlass.const_expr(self.split):
                                        if fromg != cutlass.Int32(0):
                                            vx, vy = _ldcg_v2_i32(
                                                gbuf_row + cutlass.Int64(i) * cutlass.Int64(8)
                                            )
                                        else:
                                            pk64 = s_cbuf2[i]
                                            vx = cutlass.Int32(
                                                cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                            )
                                    else:
                                        pk64 = s_cbuf2[i]
                                        vx = cutlass.Int32(
                                            cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                        )
                                    uq = fkey_bits(cutlass.Uint32(vx))
                                else:
                                    id0 = cutlass.Int32(
                                        cutlass.Uint32(s_cbuf[i]) & cutlass.Uint32(IDXM__main)
                                    )
                                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                                        uq = fkey(ldg_f32(x_addr, id0))
                                    else:
                                        uq = fkey(ldg_bf16(x_addr, id0))
                                if uq >= cutlass.Uint32(rlo):
                                    if uq <= cutlass.Uint32(rhi):
                                        du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                        if du > cutlass.Uint32(NBS - 1):
                                            du = cutlass.Uint32(NBS - 1)
                                        atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(du), cutlass.Int32(1)
                                        )
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # ---- barrier (level hist) ----
                            scan_cross0(
                                s_hist,
                                need2,
                                tidx,
                                s_res,
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                                s_hist,
                                nb=NBS,
                                zero=True,
                            )
                            cute.arch.barrier()  # ---- barrier (level scan) ----
                            above2 = above2 + s_res[RES_ABOVE]
                            need2 = need2 - s_res[RES_ABOVE]
                            m2 = s_res[RES_M]
                            sB = s_res[RES_B]
                            nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                            if sB != cutlass.Int32(NBS - 1):
                                rhi = nlo + ((cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1))
                            rlo = nlo
                            lev = lev + cutlass.Int32(1)
                    if tidx == cutlass.Int32(0):
                        s_scal[1] = cutlass.Int32(0)
                        s_scal[2] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (emit counters) ----
                    nA = k
                    nT = cutlass.Int32(0)
                    if tie_m != cutlass.Int32(0):
                        nA = above2
                        nT = need2
                    it2 = (listN + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                    it = cutlass.Int32(0)
                    while it < it2:
                        i = it * cutlass.Int32(BLK) + tidx
                        p1 = cutlass.Int32(0)
                        p2 = cutlass.Int32(0)
                        idv = cutlass.Int32(0)
                        if i < listN:
                            uq = cutlass.Uint32(0)
                            if cutlass.const_expr(self.vstg):
                                vx = cutlass.Int32(0)
                                vy = cutlass.Int32(0)
                                if cutlass.const_expr(self.split):
                                    if fromg != cutlass.Int32(0):
                                        vx, vy = _ldcg_v2_i32(
                                            gbuf_row + cutlass.Int64(i) * cutlass.Int64(8)
                                        )
                                    else:
                                        pk64 = s_cbuf2[i]
                                        vx = cutlass.Int32(
                                            cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                        )
                                        vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                                else:
                                    pk64 = s_cbuf2[i]
                                    vx = cutlass.Int32(
                                        cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                    )
                                    vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                                uq = fkey_bits(cutlass.Uint32(vx))
                                idv = vy
                            else:
                                idv = cutlass.Int32(
                                    cutlass.Uint32(s_cbuf[i]) & cutlass.Uint32(IDXM__main)
                                )
                                if cutlass.const_expr(self.dtype == cutlass.Float32):
                                    uq = fkey(ldg_f32(x_addr, idv))
                                else:
                                    uq = fkey(ldg_bf16(x_addr, idv))
                            iu = cutlass.Int64(uq)
                            if iu > ethr:
                                p1 = cutlass.Int32(1)
                            if tie_m != cutlass.Int32(0):
                                if iu == ethr:
                                    p2 = cutlass.Int32(1)
                        # staged idx (already >= lead via the P3 M-mask) -> local
                        # frame; the x_addr re-read above stays in the col0 frame.
                        if cutlass.const_expr(self.prefill):
                            idv = idv - s_lead[0]
                        self._ballot_pair_emit(
                            p1, p2, idv, cutlass.Int32(0), nA, nA, nT, out_row, s_scal, lane
                        )
                        it = it + cutlass.Int32(1)
                else:
                    # ---- degen B: whole-row narrowing ----
                    rlo = cutlass.Uint32(0)
                    rhi = cutlass.Uint32(0xFFFFFFFF)
                    above2 = cutlass.Int32(0)
                    need2 = k
                    # prefill: the genuine window is [lead, n); the <=lead lead
                    # lanes are excluded from the histogram, the emit and the
                    # candidate count so they never join a tie class.
                    lead_db = cutlass.Int32(0)
                    if cutlass.const_expr(self.prefill):
                        lead_db = s_lead[0]
                    m2 = n - lead_db
                    ethr = cutlass.Int64(0)
                    tie_m = cutlass.Int32(1)
                    if tidx < cutlass.Int32(NBS):
                        s_hist[tidx] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (degen B init) ----
                    brk = cutlass.Int32(0)
                    lev = cutlass.Int32(0)
                    while brk == cutlass.Int32(0):  # <=8 levels
                        if need2 == m2:
                            ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                            above2 = above2 + m2
                            need2 = cutlass.Int32(0)
                            tie_m = cutlass.Int32(0)
                            brk = cutlass.Int32(1)
                        elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        elif lev >= cutlass.Int32(8):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        else:
                            d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                            b2_ = cutlass.Int32(32) - clz_i32(cutlass.Int32(d2 | cutlass.Uint32(1)))
                            sh2 = b2_ - cutlass.Int32(self.lb)
                            if sh2 < cutlass.Int32(0):
                                sh2 = cutlass.Int32(0)
                            sh2u = cutlass.Uint32(sh2)
                            i = tidx + lead_db  # prefill: skip lead lanes
                            while i < n:  # whole row (window [lead, n))
                                if cutlass.const_expr(self.dtype == cutlass.Float32):
                                    uq = fkey(ldg_f32(x_addr, i))
                                else:
                                    uq = fkey(ldg_bf16(x_addr, i))
                                if uq >= cutlass.Uint32(rlo):
                                    if uq <= cutlass.Uint32(rhi):
                                        du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                        if du > cutlass.Uint32(NBS - 1):
                                            du = cutlass.Uint32(NBS - 1)
                                        atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(du), cutlass.Int32(1)
                                        )
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # ---- barrier (level hist) ----
                            scan_cross0(
                                s_hist,
                                need2,
                                tidx,
                                s_res,
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                                s_hist,
                                nb=NBS,
                                zero=True,
                            )
                            cute.arch.barrier()  # ---- barrier (level scan) ----
                            above2 = above2 + s_res[RES_ABOVE]
                            need2 = need2 - s_res[RES_ABOVE]
                            m2 = s_res[RES_M]
                            sB = s_res[RES_B]
                            nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                            if sB != cutlass.Int32(NBS - 1):
                                rhi = nlo + ((cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1))
                            rlo = nlo
                            lev = lev + cutlass.Int32(1)
                    if tidx == cutlass.Int32(0):
                        s_scal[1] = cutlass.Int32(0)
                        s_scal[2] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (emit counters) ----
                    nA = k
                    nT = cutlass.Int32(0)
                    if tie_m != cutlass.Int32(0):
                        nA = above2
                        nT = need2
                    it2 = (n + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                    it = cutlass.Int32(0)
                    while it < it2:
                        i = it * cutlass.Int32(BLK) + tidx
                        p1 = cutlass.Int32(0)
                        p2 = cutlass.Int32(0)
                        if i < n:
                            if i >= lead_db:  # prefill: exclude lead lanes
                                if cutlass.const_expr(self.dtype == cutlass.Float32):
                                    uq = fkey(ldg_f32(x_addr, i))
                                else:
                                    uq = fkey(ldg_bf16(x_addr, i))
                                iu = cutlass.Int64(uq)
                                if iu > ethr:
                                    p1 = cutlass.Int32(1)
                                if tie_m != cutlass.Int32(0):
                                    if iu == ethr:
                                        p2 = cutlass.Int32(1)
                        self._ballot_pair_emit(
                            p1, p2, i - lead_db, cutlass.Int32(0), nA, nA, nT, out_row, s_scal, lane
                        )
                        it = it + cutlass.Int32(1)

        # ---- varlen short-row epilogue (production heuristicTopKDecode
        # convention): every valid position is in the top-K — emit identity
        # indices and pad the tail with -1.  The body above ran as a
        # zero-work pass for these rows (n = 0, TGT = INT_MAX) so nothing
        # was written; only part 0 of a SPLIT row emits.
        if cutlass.const_expr(self.varlen):
            if short != cutlass.Int32(0):
                if part == cutlass.Int32(0):
                    i = tidx
                    while i < n_row:
                        out_row[i] = i
                        i = i + cutlass.Int32(BLK)
                    j = n_row + tidx
                    while j < k:
                        out_row[j] = cutlass.Int32(-1)
                        j = j + cutlass.Int32(BLK)

    # ------------------------------------------------------------------
    # host launcher (grid dim3(R, b); MINB wall via min_blocks_per_mp)
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        out: cute.Tensor,
        ws: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        scap_dead: cutlass.Int32,
        cmp_dead: cutlass.Int32,
        R: cutlass.Int32,
        SMP: cutlass.Int32,
        TGT: cutlass.Int32,
        Q: cutlass.Int32,
        SS2: cutlass.Int32,
        TGT2: cutlass.Int32,
        kv_lens: cute.Tensor,
        aim_base: cutlass.Int32,
        sfac: cutlass.Int32,
        amin: cutlass.Int32,
        sd_en: cutlass.Int32,
        tsh_en: cutlass.Int32,
        stream,
    ):
        b = logits.shape[0]
        self.kern(
            logits,
            pre_idx,
            out,
            ws,
            n,
            npad,
            k,
            scap_dead,
            cmp_dead,
            R,
            SMP,
            TGT,
            Q,
            SS2,
            TGT2,
            kv_lens,
            aim_base,
            sfac,
            amin,
            sd_en,
            tsh_en,
        ).launch(grid=(R, b, 1), block=(self.blk, 1, 1), stream=stream, min_blocks_per_mp=self.minb)


_COMPILE_CACHE = {}


def get_compiled(
    tpl: tuple,
    options_extra: str = "",
    hint_free: bool = False,
    prefill: bool = False,
    dtype: str = "f32",
) -> Any:
    """Compile (or fetch) the gvr_main variant for constexpr tuple
    tpl = (BLK, U, MINB, NBS, KPT, SPLIT, TSHG)                — legacy, or
    tpl = (BLK, U, MINB, NBS, KPT, SPLIT, TSHG, NEXT_N, CR_SHIFT, R_CONST)
    — per-row varlen mode (TSHG slot is ignored: varlen compiles the TSH
    machinery in whenever SPLIT and gates it per row at runtime).

    ``prefill`` selects the per-row window mode. It shares the varlen tuple
    (next_n=1, cr_shift=0) but has a distinct prologue, so it MUST be part of
    the cache key — otherwise a DSv3.2 decode varlen engine and the prefill
    engine collide on the same tuple. The prefill compile also retypes the
    pre_idx ABI slot to a 1-D align-4 fake (it carries 4B-aligned row_ends)."""
    key = (tuple(tpl), options_extra, bool(hint_free), bool(prefill), str(dtype))
    hit = _COMPILE_CACHE.get(key)
    if hit is not None:
        return hit
    if prefill:
        assert len(tpl) == 10, "prefill compile requires the varlen tuple"
    if len(tpl) == 7:
        blk, u, minb, nbs, kpt, split, tshg = tpl
        kern = GvrMainKernel(
            blk,
            u,
            minb,
            nbs,
            kpt,
            bool(split),
            bool(tshg),
            hint_free=bool(hint_free),
            dtype=_dtype_of(dtype),
        )
    else:
        blk, u, minb, nbs, kpt, split, tshg, next_n, cr_shift, r_const = tpl
        kern = GvrMainKernel(
            blk,
            u,
            minb,
            nbs,
            kpt,
            bool(split),
            bool(tshg),
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            r_const=r_const,
            hint_free=bool(hint_free),
            prefill=bool(prefill),
            dtype=_dtype_of(dtype),
        )
    r0, c0 = cute.sym_int(), cute.sym_int()
    r1, c1 = cute.sym_int(), cute.sym_int()
    r2, c2 = cute.sym_int(), cute.sym_int()
    w0 = cute.sym_int()
    v0 = cute.sym_int()
    if dtype == "f32":
        logits_fake = _crt.make_fake_compact_tensor(
            cutlass.Float32, (r0, c0), stride_order=(1, 0), assumed_align=16
        )
    else:
        logits_fake = _crt.make_fake_compact_tensor(
            cutlass.BFloat16, (r0, c0), stride_order=(1, 0), assumed_align=16
        )
    if prefill:
        # pre_idx slot carries row_ends [rows] int32 (4B-aligned slices).
        pre_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (r1,), stride_order=(0,), assumed_align=4
        )
    else:
        pre_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (r1, c1), stride_order=(1, 0), assumed_align=16
        )
    out_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r2, c2), stride_order=(1, 0), assumed_align=16
    )
    ws_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (w0,), stride_order=(0,), assumed_align=16
    )
    kv_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (v0,), stride_order=(0,), assumed_align=4
    )
    fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kern,
        logits_fake,
        pre_fake,
        out_fake,
        ws_fake,
        *([cutlass.Int32(0)] * 11),
        kv_fake,
        *([cutlass.Int32(0)] * 5),
        stream=fake_stream,
        options=("--enable-tvm-ffi " + options_extra).strip(),
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def workspace_bytes() -> int:
    return WS_BYTES


__all__ = ["GvrMainKernel", "get_compiled", "workspace_bytes"]
