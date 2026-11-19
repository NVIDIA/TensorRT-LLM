# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

# ===========================================================================
# ==== family: reg =============================================
# ===========================================================================
"""gvr_topk_reg — register-resident exact top-K, one CTA per row, histogram
bins in FLOAT space.

Template knobs (CUDA `gvr_topk_reg<BLK,VPT,MINB,KPT,CUR,DEG,IMGF,NBH>`):
ctor args of :class:`GvrTopkRegKernel`, read as `cutlass.const_expr(self.x)`
inside the kernel. Runtime args mirror the CUDA `(n, npad, k, CMP, IMGOFF,
QC)` — npad/k come from tensor shapes, IMGOFF is dropped (dispatch pins
IMGOFF == NBSEL == NBH at every site, asserted in the host wrapper).

Shared-memory map (single dynamic window, word offsets; the CUDA static
__shared__ block is folded into the first 512 B so occupancy accounting
matches nvcc's static+dynamic sum):

    [0..5]        s_res    (shared slot map RES_B/M/ABOVE/TOT/B2/B3)
    [6..7]        s_cnt    (s_o1, s_oc)
    [8..9]        s_kmm    (s_kmin, s_kmax — Uint32)
    [10..11]      s_e12    (s_e1, s_e2)
    [12..13]      long-register histogram transform reload (SC, CQ)
    [16..16+NW)   ws       (scan_cross_w workspace)
    [48..48+NW)   wmn      (Uint32 warp min partials)
    [80..80+NW)   wmx      (Uint32 warp max partials)
    [128..128+NBH)          hist
    [128+NBH..128+NBH+CMP)  ck     (Uint32 crossing keys; CMP dynamic)
    [128+NBH+CMP..+2CMP)    ci     (Int32 crossing indices)
    img/bm alias ck at word 128+NBH (IMGOFF==NBH)

Launch smem = 512 + dispatch_smem_bytes (dynamic Int32).

TOOLCHAIN GOTCHA: dynamic launch smem with min_blocks_per_mp>1 crashes
cutlass_dsl._build_kernel_attrs (host ceil() on a dynamic value while
computing the PREFERRED_SHARED_MEMORY_CARVEOUT hint). `_no_carveout()`
scopes a monkeypatch around cute.compile dropping ONLY that hint (CUDA
__launch_bounds__ sets no carveout either); `.reqntid`/`.minnctapersm`
(the register wall) are unaffected.
"""

import contextlib
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import T, dsl_user_op

# The shared device units, the main family and the families before this one, from the module that
# re-exports this one (``C`` as in that module, where it is the module itself).
try:
    from . import gvr_topk_decode as C
except ImportError:  # standalone dir
    import gvr_topk_decode as C

RES_ABOVE = C.RES_ABOVE
RES_B = C.RES_B
RES_M = C.RES_M
RES_TOT = C.RES_TOT
SENT_HI = C.SENT_HI
SENT_LO = C.SENT_LO
atomic_add_cta = C.atomic_add_cta
atomic_max_cta = C.atomic_max_cta
atomic_min_cta = C.atomic_min_cta
atomic_or_cta = C.atomic_or_cta
ballot = C.ballot
clz_i32 = C.clz_i32
f2s_rz = C.f2s_rz
f2u_rz = C.f2u_rz
f32_of_u32 = C.f32_of_u32
ffs_m1 = C.ffs_m1
find_cross = C.find_cross
fkey = C.fkey
fmax_f32 = C.fmax_f32
fmin_f32 = C.fmin_f32
g2r_atom_f32 = C.g2r_atom_f32
invkey = C.invkey
ld_g_f32x4 = C.ld_g_f32x4
ld_g_i32 = C.ld_g_i32
ldg_f32 = C.ldg_f32
popc = C.popc
scan_cross_w = C.scan_cross_w
u32_of_f32 = C.u32_of_f32
warp_add_i32 = C.warp_add_i32
warp_incl_scan_add2 = C.warp_incl_scan_add2
warp_max_u32 = C.warp_max_u32
warp_min_u32 = C.warp_min_u32


NB__reg = 1024  # NBH default
STATIC_WORDS = 128  # DSL smem prelude (static-__shared__ mirror)
STATIC_BYTES = STATIC_WORDS * 4
_NEG_INF__reg = float("-inf")
_POS_INF = float("inf")


# ---------------------------------------------------------------------------
# module-local FP spellings
# ---------------------------------------------------------------------------
@dsl_user_op
def _fmaf__reg(a, b, c, *, loc=None, ip=None):
    """CUDA fmaf: single fma.rn.f32."""
    return cutlass.Float32(
        mlir_math.fma(
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            fastmath=mlir_arith.FastMathFlags.none,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _submul_asm(v, t, sc, *, loc=None, ip=None):
    """(v - t) * sc with two roundings, opaque to CSE/contraction.

    Used at the !BRL classify site so no sub-expression is shared with the
    emit's `fmaf(v - T, SC, OFF)` — the CUDA deliberately spells the two
    sites differently to stop the compiler holding all S q's live across
    the barrier.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [v.ir_value(loc=loc, ip=ip), t.ir_value(loc=loc, ip=ip), sc.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .f32 rtmp;\n\tsub.rn.f32 rtmp, $1, $2;\n\tmul.rn.f32 $0, rtmp, $3;\n\t}",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _smem_addr_reg__reg(addr, *, loc=None, ip=None):
    """Pin a CTA-shared 32-bit byte address in ONE register.

    Identity `mov` behind an asm boundary: without it LLVM re-folds the
    `mov.b32 %r, __dynamic_shmem__0` symbol materialisation into EVERY use
    site, and ptxas then re-derives the CGA shared window (S2UR SR_CgaCtaId
    + UMOV + ULEA, 3 instructions) inside each divergent classify block.
    The asm result is not duplicable, so the window is materialised exactly
    once. Value-identical: a plain register copy.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value(loc=loc, ip=ip)],
            "mov.u32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _red_shared_add1__reg(addr, *, loc=None, ip=None):
    """CUDA classify `atomicAdd(&hist[bin], 1u)` with the result unused.

    `red` (not `atom`) is the result-less spelling — ptxas lowers it to the
    same ATOMS.POPC.INC.32 RZ the CUDA arm emits. Same ordering contract as
    atomic_add_cta: .relaxed scope .cta. Takes the final shared byte address
    as a plain Int32 so the address datapath stays ordinary IR (ptxas fuses
    the shl+add into one LEA against the pinned `_smem_addr_reg__reg` base).
    """
    llvm.inline_asm(
        res=None,
        operands_=[addr.ir_value(loc=loc, ip=ip)],
        asm_string="red.relaxed.cta.shared.add.u32 [$0], 1;",
        constraints="r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _umin_u32(a, b):
    """unsigned min(a, b) — CUDA min() on the bin clamp (IMNMX)."""
    r = a
    if b < a:
        r = b
    return r


@cute.jit
def _fabsf(x):
    """|x| via sign-bit clear (exact, matches fabsf)."""
    return f32_of_u32(u32_of_f32(x) & cutlass.Uint32(0x7FFFFFFF))


def _f32_smem_atom():
    return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)


def _sts128_f32(atom, frag, base_addr, byte_off):
    p = cute.make_ptr(
        cutlass.Float32, base_addr + byte_off, cute.AddressSpace.smem, assumed_align=16
    )
    cute.copy(atom, frag, cute.make_tensor(p, cute.make_layout((4,))))


def _smem_view(dtype, sbase, word_off: int, length: int, align: int = 16):
    """Typed tensor view at a constexpr word offset into the smem window."""
    p = cute.make_ptr(
        dtype, sbase + cutlass.Int32(word_off * 4), cute.AddressSpace.smem, assumed_align=align
    )
    return cute.make_tensor(p, cute.make_layout((length,)))


def _val(frags, s: int):
    """val[s] accessor over the float4[VPT] register batch (constexpr s)."""
    return frags[s // 4][s % 4]


@contextlib.contextmanager
def _no_carveout():
    """Scoped: drop the DSL's carveout hint (see module docstring)."""
    import cutlass.cutlass_dsl.cutlass as _cdsl

    orig = _cdsl._build_kernel_attrs
    _cdsl._build_kernel_attrs = lambda config: {}
    # DSL >= 4.6.1 also derives a carveout inside the compiled artifact from
    # the smem.max_smem_per_mp MLIR attribute (emitted when
    # min_blocks_per_mp > 1).  That calculation only sees the static smem, so
    # with dynamic launch smem it selects a 16 KiB shared-memory config and
    # pins the SM to one resident CTA.  Drop the attribute as well so the
    # driver default carveout stays in effect.
    base = getattr(_cdsl, "CutlassBaseDSL", None)
    orig_gen = getattr(base, "_generate_kernel_attrs", None)

    if orig_gen is not None:

        def _gen(self, config, _orig=orig_gen):
            ret = _orig(self, config)
            ret.pop("smem.max_smem_per_mp", None)
            return ret

        base._generate_kernel_attrs = _gen
    try:
        yield
    finally:
        _cdsl._build_kernel_attrs = orig
        if orig_gen is not None:
            base._generate_kernel_attrs = orig_gen


class GvrTopkRegKernel:
    """gvr_topk_reg<BLK, VPT, MINB, KPT, CUR, DEG, IMGF, NBH>."""

    def __init__(
        self,
        blk: int,
        vpt: int,
        minb: int,
        kpt: int,
        cur: bool,
        deg: bool,
        img: bool,
        nbh: int = NB__reg,
        varlen: bool = False,
        next_n: int = 1,
        cr_shift: int = 0,
        hint_free: bool = False,
    ) -> None:
        # Ballot and deterministic emission pack four bits per float4.
        assert blk in (256, 512, 1024) and 1 <= vpt <= 8
        assert nbh in (256, 512, 1024, 2048)
        assert nbh % blk == 0 or blk % nbh == 0
        self.blk = blk
        self.vpt = vpt
        self.minb = minb
        self.kpt = kpt
        self.cur = bool(cur)
        self.deg = bool(deg)
        self.img = bool(img)
        self.nbh = nbh
        # per-row varlen mode (production heuristicTopKDecode contract, same
        # semantics as GvrMainKernel/GvrRegClusKernel): n is re-derived PER
        # ROW in-kernel from a device kv_lens tensor; the scalar n launch arg
        # becomes the envelope clamp bound. next_n / cr_shift compile-time.
        self.varlen = bool(varlen)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        if self.varlen:
            assert self.next_n >= 1 and self.cr_shift in (0, 2)
        # derived compile-time constants
        self.S = vpt * 4
        self.lnbh = {256: 8, 512: 9, 2048: 11}.get(nbh, 10)
        # Hint-free brackets use resident values; hint gathers are disabled.
        self.hint_free = bool(hint_free)
        # Dense brackets (every hint-free plan) fold a prefix, or the complete row for DEG plans.
        self.dense_bracket = self.hint_free
        self.use_bm = (not deg) and (not img) and kpt >= 2 and vpt == 1 and (not hint_free)
        self.use_img = img and vpt == 1 and (not hint_free)
        self.brl = (minb * blk <= 1024) or (vpt == 1)
        self.zero_offset_brl = self.hint_free and blk == 256 and vpt == 3 and minb == 4

    # ------------------------------------------------------------------
    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        cmp_: cutlass.Int32,
        qc: cutlass.Int32,
        smem_bytes: cutlass.Int32,
    ):
        BLK = cutlass.const_expr(self.blk)
        VPT = cutlass.const_expr(self.vpt)
        KPT = cutlass.const_expr(self.kpt)
        NBH = cutlass.const_expr(self.nbh)  # noqa: F841
        S = cutlass.const_expr(self.S)
        LNBH = cutlass.const_expr(self.lnbh)
        NW = cutlass.const_expr(self.blk // 32)

        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tid & cutlass.Int32(31)
        dw = tid >> cutlass.Int32(5)  # DET SLOTS: warp index

        # ================= per-row varlen prologue (varlen mode only) =========
        # Production heuristicTopKDecode contract (GvrMainKernel /
        # GvrRegClusKernel discipline): row r serves request r // next_n with
        # n = (kv_lens[req] - next_n + r % next_n + 1) >> cr_shift, clamped to
        # the envelope launch arg n (the launcher admits this family only when
        # the envelope fits its capacity window, so per-row n never exceeds
        # capacity). One CTA per row, so the whole-body guard below is
        # trivially block-uniform. Short rows (n <= k) emit identity + (-1)
        # tail here (k can exceed BLK on this family -> strided loop, unlike
        # the reg_clus k <= BLK single predicate) and SKIP the body entirely:
        # a zero-work pass would reach the degenerate emitter and poison out.
        short = cutlass.Int32(0)
        prow = row
        if cutlass.const_expr(self.varlen):
            kq = cutlass.Int32(pre_idx.shape[1])
            req = row // cutlass.Int32(self.next_n)
            rr = row % cutlass.Int32(self.next_n)
            prow = req
            kvl = kv_lens[req]
            nv = (kvl - cutlass.Int32(self.next_n) + rr + cutlass.Int32(1)) >> cutlass.Int32(
                self.cr_shift
            )
            if nv < cutlass.Int32(0):
                nv = cutlass.Int32(0)
            if nv > n:
                nv = n
            if nv <= kq:
                short = cutlass.Int32(1)
            if short == cutlass.Int32(0):
                n = nv
            if short != cutlass.Int32(0):
                i = tid
                while i < kq:
                    ov = cutlass.Int32(-1)
                    if i < nv:
                        ov = i
                    out[row, i] = ov
                    i = i + cutlass.Int32(BLK)

        # ------------------------------------------------------------------
        # Predeclarations: the DSL AST transformer requires every scalar that
        # is (re)assigned under a dynamic if/while region to pre-exist with a
        # stable type at every enclosing region level. Constant inits are
        # sunk/dead-coded by LLVM, so this costs no registers.
        # ------------------------------------------------------------------
        i = cutlass.Int32(0)
        j = cutlass.Int32(0)
        r = cutlass.Int32(0)
        tinc = cutlass.Int32(0)
        nA = cutlass.Int32(0)
        nT = cutlass.Int32(0)
        n1 = cutlass.Int32(0)
        n2 = cutlass.Int32(0)
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        p1e = cutlass.Int32(0)
        p2e = cutlass.Int32(0)
        lml = cutlass.Int32(0)
        aboveC = cutlass.Int32(0)
        needC = cutlass.Int32(0)
        mm = cutlass.Int32(0)
        lev = cutlass.Int32(0)
        done = cutlass.Int32(0)
        b2w = cutlass.Int32(0)
        sh2 = cutlass.Int32(0)
        it = cutlass.Int32(0)
        it2 = cutlass.Int32(0)
        idv = cutlass.Int32(0)
        q1e = cutlass.Int32(0)
        q2e = cutlass.Int32(0)
        q1f = cutlass.Int32(0)
        q2f = cutlass.Int32(0)
        b_lv = cutlass.Int32(0)
        mc = cutlass.Int32(0)
        quad = cutlass.Int32(0)
        lim1 = cutlass.Int32(0)
        p = cutlass.Int32(0)
        idx = cutlass.Int32(0)
        m1 = cutlass.Int32(0)
        m2 = cutlass.Int32(0)
        t1 = cutlass.Int32(0)
        t2 = cutlass.Int32(0)
        c1 = cutlass.Int32(0)
        c2 = cutlass.Int32(0)
        p1 = cutlass.Int32(0)
        p2 = cutlass.Int32(0)
        nbw = cutlass.Int32(0)
        uq = cutlass.Uint32(0)
        vq = cutlass.Uint32(0)
        kv = cutlass.Uint32(0)
        rlo = cutlass.Uint32(0)
        rhi = cutlass.Uint32(0)
        d2 = cutlass.Uint32(0)
        unar = cutlass.Uint32(0)
        bnn = cutlass.Uint32(0)
        nlo = cutlass.Uint32(0)
        uke = cutlass.Uint32(0)
        uk = cutlass.Uint32(0)
        bn = cutlass.Uint32(0)
        w = cutlass.Uint32(0)
        wt = cutlass.Uint32(0)
        ethr = cutlass.Int64(0)
        u64 = cutlass.Int64(0)
        HIf = cutlass.Float32(0.0)
        LOf = cutlass.Float32(0.0)
        qt3 = cutlass.Float32(0.0)

        # wrap-scoping additions (varlen whole-body guard): names whose
        # first assignment moves under the dynamic `if short == 0` region
        # below and are reassigned deeper — same rule as the block above.
        a = cutlass.Uint32(0)
        c = cutlass.Uint32(0)
        lmin = cutlass.Uint32(0)
        lmax = cutlass.Uint32(0)
        esc = cutlass.Int32(0)
        okc = cutlass.Int32(0)
        whole = cutlass.Int32(0)
        tval = cutlass.Float32(0.0)
        wsel = cutlass.Float32(0.0)
        GMAX = cutlass.Float32(0.0)
        Tv = cutlass.Float32(0.0)
        lmn = cutlass.Float32(0.0)
        lmx = cutlass.Float32(0.0)

        if short == cutlass.Int32(0):
            npad = cutlass.Int32(logits.shape[1])  # noqa: F841
            k = cutlass.Int32(pre_idx.shape[1])
            x_addr = logits[row, None].iterator.toint()  # Int64 gmem byte base
            p_addr = pre_idx[prow, None].iterator.toint()  # request-level under varlen

            # ---- shared-memory window (map in module docstring) ----
            sptr = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
            sbase = sptr.toint()  # Int32 shared addr

            s_res = _smem_view(cutlass.Int32, sbase, 0, 6)
            s_cnt = _smem_view(cutlass.Int32, sbase, 6, 2)  # [0]=s_o1 [1]=s_oc
            s_kmm = _smem_view(cutlass.Uint32, sbase, 8, 2)  # [0]=s_kmin [1]=s_kmax
            s_e12 = _smem_view(cutlass.Int32, sbase, 10, 2)  # [0]=s_e1 [1]=s_e2
            s_ws = _smem_view(cutlass.Int32, sbase, 16, 32)
            s_wmn = _smem_view(cutlass.Uint32, sbase, 48, 32)
            s_wmx = _smem_view(cutlass.Uint32, sbase, 80, 32)
            s_dw = _smem_view(
                cutlass.Int32, sbase, 16, 64
            )  # DET SLOTS: per-warp class counts (aliases ws+wmn, idle during emits)
            s_hist = _smem_view(cutlass.Int32, sbase, STATIC_WORDS, self.nbh)
            ck_base = sbase + cutlass.Int32((STATIC_WORDS + self.nbh) * 4)
            ck = cute.make_tensor(
                cute.make_ptr(cutlass.Uint32, ck_base, cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout((65536,)),
            )  # typed view, no bound
            ci = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Int32,
                    ck_base + cmp_ * cutlass.Int32(4),
                    cute.AddressSpace.smem,
                    assumed_align=4,
                ),
                cute.make_layout((65536,)),
            )
            img_f = cute.make_tensor(  # aliases ck/ci
                cute.make_ptr(cutlass.Float32, ck_base, cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout((65536,)),
            )
            bm = cute.make_tensor(  # aliases ck
                cute.make_ptr(cutlass.Int32, ck_base, cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout((65536,)),
            )

            out_row = out[row, None]

            n4 = n >> cutlass.Int32(2)
            ntail = n - (n4 << cutlass.Int32(2))
            tix = (n4 << cutlass.Int32(2)) + tid  # CUDA `tidx`

            # ---- hint prefetch: KPT coalesced pre_idx words BEFORE any
            # dependent gather; compiled out under DEG.
            pvs = []
            if cutlass.const_expr(not (self.deg or self.hint_free)):
                for t in cutlass.range_constexpr(KPT):
                    pv = cutlass.Int32(-1)
                    j = tid + cutlass.Int32(t * self.blk)
                    if j < k:
                        pv = ld_g_i32(p_addr, j)
                    pvs.append(pv)

            # ---- row load: exact-fit peel + float4[VPT] register batch
            atom128 = g2r_atom_f32(128, invariant=True)
            frags = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(VPT)]
            if n4 >= cutlass.Int32(self.blk * self.vpt):  # block-uniform peel
                for u in cutlass.range_constexpr(VPT):
                    ld_g_f32x4(atom128, x_addr, tid + cutlass.Int32(u * self.blk), frags[u])
            else:  # predicated flat batch
                for u in cutlass.range_constexpr(VPT):
                    i = tid + cutlass.Int32(u * self.blk)
                    if i < n4:
                        ld_g_f32x4(atom128, x_addr, i, frags[u])
                for u in cutlass.range_constexpr(VPT):
                    i = tid + cutlass.Int32(u * self.blk)
                    if i >= n4:  # -INFINITY fill
                        for q in cutlass.range_constexpr(4):
                            frags[u][q] = cutlass.Float32(_NEG_INF__reg)

            tval = cutlass.Float32(_NEG_INF__reg)
            if tid < ntail:
                tval = ldg_f32(x_addr, tix)

            # ---- init
            if tid == cutlass.Int32(0):
                s_cnt[0] = cutlass.Int32(0)
                s_cnt[1] = cutlass.Int32(0)
            for z in cutlass.range_constexpr(self.nbh // self.blk):
                s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)

            # ---- bracket: 4 mutually exclusive compile-time arms
            lmin = cutlass.Uint32(0xFFFFFFFF)
            lmax = cutlass.Uint32(0)
            if cutlass.const_expr(self.use_img):
                fatom = _f32_smem_atom()
                for u in cutlass.range_constexpr(VPT):  # VPT == 1 here
                    i = tid + cutlass.Int32(u * self.blk)
                    if i < n4:
                        _sts128_f32(fatom, frags[u], ck_base, i * cutlass.Int32(16))
                if tid < ntail:
                    img_f[tix] = tval
                cute.arch.barrier()  # image staged
                for t in cutlass.range_constexpr(KPT):
                    p = pvs[t]
                    if cutlass.Uint32(p) < cutlass.Uint32(n):
                        uk = fkey(img_f[p])
                        if uk < lmin:
                            lmin = uk
                        if uk > lmax:
                            lmax = uk
                cute.arch.barrier()  # img dies
            elif cutlass.const_expr(self.use_bm):
                nbw = (n + cutlass.Int32(31)) >> cutlass.Int32(5)
                i = tid
                while i < nbw:  # bitmap clear
                    bm[i] = cutlass.Int32(0)
                    i = i + cutlass.Int32(BLK)
                cute.arch.barrier()  # bitmap cleared
                for t in cutlass.range_constexpr(KPT):
                    p = pvs[t]
                    if cutlass.Uint32(p) < cutlass.Uint32(n):
                        atomic_or_cta(
                            bm.iterator + (p >> cutlass.Int32(5)),
                            cutlass.Int32(1) << (p & cutlass.Int32(31)),
                        )
                cute.arch.barrier()  # bitmap set
                lmn = cutlass.Float32(_POS_INF)
                lmx = cutlass.Float32(_NEG_INF__reg)
                for u in cutlass.range_constexpr(VPT):
                    base = (tid + cutlass.Int32(u * self.blk)) << cutlass.Int32(2)
                    w = cutlass.Uint32(0)
                    if cutlass.Uint32(base) < cutlass.Uint32(n):
                        w = cutlass.Uint32(bm[base >> cutlass.Int32(5)]) >> cutlass.Uint32(
                            base & cutlass.Int32(31)
                        )
                    for cbit in cutlass.range_constexpr(4):
                        if (w & cutlass.Uint32(1 << cbit)) != cutlass.Uint32(0):
                            lmn = fmin_f32(lmn, _val(frags, 4 * u + cbit))
                            lmx = fmax_f32(lmx, _val(frags, 4 * u + cbit))
                if tid < ntail:
                    wt = cutlass.Uint32(bm[tix >> cutlass.Int32(5)]) >> cutlass.Uint32(
                        tix & cutlass.Int32(31)
                    )
                    if (wt & cutlass.Uint32(1)) != cutlass.Uint32(0):
                        lmn = fmin_f32(lmn, tval)
                        lmx = fmax_f32(lmx, tval)
                lmin = fkey(lmn)
                lmax = fkey(lmx)  # monotone
                cute.arch.barrier()  # bm dies
            elif cutlass.const_expr(self.dense_bracket and not self.deg):
                lmn = cutlass.Float32(_POS_INF)
                lmx = cutlass.Float32(_NEG_INF__reg)
                for s in cutlass.range_constexpr(S):
                    pos = (
                        (tid + cutlass.Int32((s // 4) * self.blk)) << cutlass.Int32(2)
                    ) + cutlass.Int32(s % 4)
                    if pos < k:
                        v = _val(frags, s)
                        lmn = fmin_f32(lmn, v)
                        lmx = fmax_f32(lmx, v)
                lmin = fkey(lmn)
                lmax = fkey(lmx)
            elif cutlass.const_expr(self.dense_bracket):
                lmn = cutlass.Float32(_POS_INF)
                lmx = cutlass.Float32(_NEG_INF__reg)
                for u in cutlass.range_constexpr(VPT):
                    pos4 = tid + cutlass.Int32(u * self.blk)
                    if pos4 < n4:
                        for component in cutlass.range_constexpr(4):
                            v = frags[u][component]
                            lmn = fmin_f32(lmn, v)
                            lmx = fmax_f32(lmx, v)
                if tid < ntail:
                    lmn = fmin_f32(lmn, tval)
                    lmx = fmax_f32(lmx, tval)
                lmin = fkey(lmn)
                lmax = fkey(lmx)
            elif cutlass.const_expr(self.deg):
                lmn = cutlass.Float32(_POS_INF)
                lmx = cutlass.Float32(_NEG_INF__reg)
                for s in cutlass.range_constexpr(S):
                    v = _val(frags, s)
                    if v > cutlass.Float32(_NEG_INF__reg):
                        lmn = fmin_f32(lmn, v)
                        lmx = fmax_f32(lmx, v)
                if tid < ntail:
                    lmn = fmin_f32(lmn, tval)
                    lmx = fmax_f32(lmx, tval)
                lmin = fkey(lmn)
                lmax = fkey(lmx)
            else:
                # default: KPT scattered fkey ldg gathers, batch-then-fold
                xs = []
                for t in cutlass.range_constexpr(KPT):
                    xv = cutlass.Float32(0.0)
                    if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):
                        xv = ldg_f32(x_addr, pvs[t])
                    xs.append(xv)
                for t in cutlass.range_constexpr(KPT):
                    if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):
                        uk = fkey(xs[t])
                        if uk < lmin:
                            lmin = uk
                        if uk > lmax:
                            lmax = uk

            # Reduce participating lanes before publishing warp bounds.
            lmin = warp_min_u32(lmin)
            lmax = warp_max_u32(lmax)
            if lane == cutlass.Int32(0):
                s_wmn[tid >> cutlass.Int32(5)] = lmin
                s_wmx[tid >> cutlass.Int32(5)] = lmax
            cute.arch.barrier()  # warp partials published
            a = cutlass.Uint32(0xFFFFFFFF)
            c = cutlass.Uint32(0)
            if lane < cutlass.Int32(NW):
                a = cutlass.Uint32(s_wmn[lane])
                c = cutlass.Uint32(s_wmx[lane])
            lmin = warp_min_u32(a)
            lmax = warp_max_u32(c)
            Tv = invkey(lmin)
            GMAX = invkey(lmax)

            # ---- collapse guard, NaN-safe. w_ < 3.5e38 (> FLT_MAX) also
            # rejects an infinite bracket width: an in-window +inf (GMAX=+inf)
            # or -inf (Tv=-inf) makes SC=0 and folds every value into bin 0.
            okc = cutlass.Int32(0)
            if Tv < GMAX:
                w_ = GMAX - Tv
                if w_ > cutlass.Float32(1e-30) and w_ < cutlass.Float32(3.5e38):
                    okc = cutlass.Int32(1)
            if okc == cutlass.Int32(0):
                Tv = cutlass.Float32(SENT_LO)
                GMAX = cutlass.Float32(SENT_HI)

            # ---- bin transform constants
            BRL = cutlass.const_expr(self.brl)  # noqa: F841
            offset_brl = self.brl and not self.zero_offset_brl
            OFFf = cutlass.Float32(1.0 if offset_brl else 0.0)
            recip = 1.0 / float(self.nbh - (2 if offset_brl else 0))
            WD = (GMAX - Tv) * cutlass.Float32(recip)
            wsel = cutlass.Float32(1e-30)
            if WD > cutlass.Float32(0.0):
                wsel = WD
            # rcp.approx (single MUFU.RCP) — the CUDA arm's exact lowering of
            # `1.0f / wsel`; a plain `1.0 / wsel` would emit the IEEE div.rn
            # Newton triple + slowpath CALL on the barrier-bounded chain
            # feeding all S classify FMULs. Output exactness is SC-invariant
            # (any SC > 0 preserves the sign/monotonicity invariants) and the
            # WD > 0 arm is bit-identical to CUDA's MUFU.RCP.
            SC = cute.arch.rcp_approx(wsel)
            QCAPf = cutlass.Float32(float(self.nbh - 1))
            CQ0 = OFFf - Tv * SC
            if cutlass.const_expr(self.zero_offset_brl):
                CQ = CQ0
            else:
                CQ = CQ0 + cutlass.Float32(1e-6) * (_fabsf(CQ0) + cutlass.Float32(1.0))

            # ---- histogram
            if cutlass.const_expr(self.brl):
                # BRL classify arm: hist base pinned ONCE via the same
                # _smem_addr_reg__reg identity-mov used in the !BRL arm below,
                # and the result-discarded classify atomics spelled as
                # resultless red.shared (_red_shared_add1__reg).
                # Value-identical: same +1 to the same byte address
                # (hb + 4*bn == &s_hist[bn]), same .relaxed.cta ordering; the
                # q/bn computations are untouched so classify/emit
                # bit-identity (BRL requirement) is preserved. Emit-path hist
                # atomics (results used) are NOT touched.
                hb = _smem_addr_reg__reg(sbase + cutlass.Int32(STATIC_WORDS * 4))
                for s in cutlass.range_constexpr(S):
                    q = _fmaf__reg(_val(frags, s), SC, CQ)
                    bn = _umin_u32(f2u_rz(q), cutlass.Uint32(self.nbh - 1))
                    if tid + cutlass.Int32((s // 4) * self.blk) < n4:
                        _red_shared_add1__reg(hb + (cutlass.Int32(bn) << cutlass.Int32(2)))
                qt = _fmaf__reg(tval, SC, CQ)
                bnt = _umin_u32(f2u_rz(qt), cutlass.Uint32(self.nbh - 1))
                if tid < ntail:
                    _red_shared_add1__reg(hb + (cutlass.Int32(bnt) << cutlass.Int32(2)))
            else:
                # hist base pinned ONCE (byte addr, +STATIC_BYTES = word 128 map);
                # each site below is then LEA + ATOMS exactly like the CUDA arm
                # instead of re-deriving the shared window per divergent block.
                hb = _smem_addr_reg__reg(sbase + cutlass.Int32(STATIC_WORDS * 4))
                for s in cutlass.range_constexpr(S):
                    q = _submul_asm(_val(frags, s), Tv, SC)  # anti-CSE classify
                    if q >= cutlass.Float32(0.0):
                        _red_shared_add1__reg(hb + (f2s_rz(fmin_f32(q, QCAPf)) << cutlass.Int32(2)))
                qt = _submul_asm(tval, Tv, SC)
                if qt >= cutlass.Float32(0.0):
                    _red_shared_add1__reg(hb + (f2s_rz(fmin_f32(qt, QCAPf)) << cutlass.Int32(2)))
            cute.arch.barrier()  # histogram done

            # ---- crossing-bin find
            if cutlass.const_expr(self.cur or self.nbh > 1024):
                scan_cross_w(s_hist, s_ws, k, tid, s_res, blk=self.blk, nb=self.nbh)
            else:
                find_cross(s_hist, k, tid, s_res, nb=self.nbh)
            cute.arch.barrier()  # crossing published
            above = s_res[RES_ABOVE]
            m = s_res[RES_M]
            Bv = s_res[RES_B]
            tot = s_res[RES_TOT]
            need = k - above
            whole = cutlass.Int32(0)
            if need >= m:
                whole = cutlass.Int32(1)

            # ---- ESCAPE: radix descent over the key space
            # Count-crossing enforcement: when the hint-derived bracket's low
            # edge sits above the true k-th value, the classify arms count
            # fewer than k entries (q < 0 is never histogrammed) and the
            # crossing scan pins its bin-0 fallback as a fake crossing — the
            # whole-bin emit would then stop at the histogram total and leave
            # out_row[tot:k) unwritten. Exactness must come from the
            # count-crossing invariant, never from the bracket estimate: when
            # the histogram never reaches k (tot < k), take the escape — it
            # ranks the FULL row in key space, independent of the bracket.
            esc = cutlass.Int32(0)
            if whole == cutlass.Int32(0):
                if m > cmp_:
                    esc = cutlass.Int32(1)
            if tot < k:
                esc = cutlass.Int32(1)
            # A degenerate bracket (okc==0) collapses the histogram into
            # bin 0; escape to the bracket-independent key-space rank, where
            # fkey(+inf) is the maximum key.
            if okc == cutlass.Int32(0):
                esc = cutlass.Int32(1)
            if cutlass.const_expr(self.brl):
                # Negative q enters bin 0 but fails q >= Bv; use exact full-row selection.
                if Bv == cutlass.Int32(0):
                    esc = cutlass.Int32(1)
            if esc == cutlass.Int32(1):
                if tid == cutlass.Int32(0):
                    s_cnt[0] = cutlass.Int32(0)
                    s_cnt[1] = cutlass.Int32(0)
                    # DEVIATION (race fix): the CUDA zeroes s_o1/s_oc again
                    # between the nA read and the emit with only ONE barrier
                    # pair around both — a read/write race. Emit instead
                    # through the path-exclusive s_e1/s_e2 slots, zeroed HERE
                    # under the existing barrier; the racy mid-emit rezero is
                    # dropped. Barrier count unchanged.
                    s_e12[0] = cutlass.Int32(0)
                    s_e12[1] = cutlass.Int32(0)
                cute.arch.barrier()  # escape init
                # Vector-lane bound: the register batch holds real data only
                # below 4*n4 — lanes in [4*n4, n) are the -inf FILL of the
                # last partial float4 (the real values there live in tval).
                # Bounding by n would count/emit each tail element twice
                # (once as a fill key, once via tval): benign while the tie
                # threshold is finite (a fill key loses every compare), but
                # it emits duplicate indices when the tie class is the -inf
                # key itself (in-window -inf entries are admissible).
                nvec = n4 << cutlass.Int32(2)
                # Narrow LNBH bits per level over the register batch — the
                # same descent the Phase-3 ck fallback below runs — instead of
                # one rescan per key bit: ceil(32/LNBH) levels at worst and
                # two in practice, since the first level's LNBH bits already
                # cut inside the exponent field, where the bisection took 32.
                # The exit arms carry the count crossing, so the old separate
                # above-count pass folds in.
                # Neither ethr nor aboveC is carried across the descent: ethr
                # is an Int64 derivable from rlo and the exit arm, and
                # aboveC + needC == k holds at every exit.
                rlo = cutlass.Uint32(0)
                rhi = cutlass.Uint32(0xFFFFFFFF)
                needC = k
                mm = n  # in-range count, carried from the previous level
                lev = cutlass.Int32(0)
                state = cutlass.Int32(0)  # 1 -> ethr = rlo, 2 -> ethr = rlo - 1
                while state == cutlass.Int32(0):
                    if needC == mm:
                        # whole in-range block is needed: threshold below it
                        needC = cutlass.Int32(0)
                        state = cutlass.Int32(2)
                    if state == cutlass.Int32(0):
                        if rlo >= rhi:  # single key left == the k-th key
                            state = cutlass.Int32(1)
                        if lev >= cutlass.Int32(34):  # non-binding: sh2 strictly shrinks
                            state = cutlass.Int32(1)
                    if state == cutlass.Int32(0):
                        d2 = rhi - rlo
                        b2w = cutlass.Int32(32) - clz_i32(cutlass.Int32(d2 | cutlass.Uint32(1)))
                        sh2 = cutlass.Int32(0)
                        if b2w > cutlass.Int32(LNBH):
                            sh2 = b2w - cutlass.Int32(LNBH)
                        for z in cutlass.range_constexpr(self.nbh // self.blk):
                            s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                        cute.arch.barrier()  # esc level clear
                        for s in cutlass.range_constexpr(S):
                            ix = (
                                (tid + cutlass.Int32((s // 4) * self.blk)) << cutlass.Int32(2)
                            ) + cutlass.Int32(s % 4)
                            if ix < nvec:
                                uev = fkey(_val(frags, s))
                                if uev >= rlo:
                                    if uev <= rhi:
                                        bne = _umin_u32(
                                            (uev - rlo) >> cutlass.Uint32(sh2),
                                            cutlass.Uint32(self.nbh - 1),
                                        )
                                        atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(bne),
                                            cutlass.Int32(1),
                                        )
                        if tid < ntail:
                            uev = fkey(tval)
                            if uev >= rlo:
                                if uev <= rhi:
                                    bne = _umin_u32(
                                        (uev - rlo) >> cutlass.Uint32(sh2),
                                        cutlass.Uint32(self.nbh - 1),
                                    )
                                    atomic_add_cta(
                                        s_hist.iterator + cutlass.Int32(bne), cutlass.Int32(1)
                                    )
                        cute.arch.barrier()  # esc level hist
                        if cutlass.const_expr(self.nbh > 1024):
                            scan_cross_w(s_hist, s_ws, needC, tid, s_res, blk=self.blk, nb=self.nbh)
                        else:
                            find_cross(s_hist, needC, tid, s_res, nb=self.nbh)
                        cute.arch.barrier()  # esc level scan
                        needC = needC - s_res[RES_ABOVE]
                        mm = s_res[RES_M]
                        b_lv = s_res[RES_B]
                        nlo = rlo + (cutlass.Uint32(b_lv) << cutlass.Uint32(sh2))
                        if b_lv != cutlass.Int32(self.nbh - 1):
                            rhi = nlo + (
                                (cutlass.Uint32(1) << cutlass.Uint32(sh2)) - cutlass.Uint32(1)
                            )
                        rlo = nlo
                        lev = lev + cutlass.Int32(1)
                nA = k - needC
                nT = needC
                ethr = cutlass.Int64(rlo)
                if state == cutlass.Int32(2):
                    ethr = ethr - cutlass.Int64(1)
                cute.arch.barrier()  # descent done
                det_lo = cutlass.Uint32(0xFFFFFFFF) - (
                    cutlass.Uint32(n) - cutlass.Uint32(1)
                )  # ~idx >= 2^32 - n
                det_hi = cutlass.Uint32(0xFFFFFFFF)
                det_need = needC
                det_thr = cutlass.Int64(-1)
                det_sh = cutlass.Uint32(0)
                det_ph = cutlass.Int32(1)
                if needC > cutlass.Int32(0):
                    if mm > needC:
                        det_ph = cutlass.Int32(0)
                while det_ph <= cutlass.Int32(1):
                    if det_ph == cutlass.Int32(0):
                        det_d = cutlass.Uint32(det_hi) - cutlass.Uint32(det_lo)
                        det_bw = cutlass.Int32(32) - clz_i32(
                            cutlass.Int32(det_d | cutlass.Uint32(1))
                        )
                        det_s = det_bw - cutlass.Int32(LNBH)
                        if det_s < cutlass.Int32(0):
                            det_s = cutlass.Int32(0)
                        det_sh = cutlass.Uint32(det_s)
                        for z in cutlass.range_constexpr(self.nbh // self.blk):
                            s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                        cute.arch.barrier()  # ---- barrier (det level clear) ----
                    lml = cutlass.Int32(cute.arch.lanemask_lt())
                    em1 = cutlass.Int32(0)  # DET SLOTS: class masks over the register batch
                    em2 = cutlass.Int32(0)
                    for s in cutlass.range_constexpr(S):
                        ixv = (
                            (tid + cutlass.Int32((s // 4) * self.blk)) << cutlass.Int32(2)
                        ) + cutlass.Int32(s % 4)
                        u64 = cutlass.Int64(-1)
                        if ixv < nvec:
                            u64 = cutlass.Int64(fkey(_val(frags, s)))
                        q1e = cutlass.Int32(0)
                        q2e = cutlass.Int32(0)
                        if u64 > ethr:
                            q1e = cutlass.Int32(1)
                        if u64 == ethr:
                            det_u = cutlass.Uint32(0xFFFFFFFF) - cutlass.Uint32(ixv)
                            if det_ph == cutlass.Int32(0):
                                if cutlass.Uint32(det_u) >= cutlass.Uint32(det_lo):
                                    if cutlass.Uint32(det_u) <= cutlass.Uint32(det_hi):
                                        det_b = (
                                            cutlass.Uint32(det_u) - cutlass.Uint32(det_lo)
                                        ) >> cutlass.Uint32(det_sh)
                                        if det_b > cutlass.Uint32(self.nbh - 1):
                                            det_b = cutlass.Uint32(self.nbh - 1)
                                        atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(det_b),
                                            cutlass.Int32(1),
                                        )
                            else:
                                if cutlass.Int64(cutlass.Uint32(det_u)) > det_thr:
                                    q2e = cutlass.Int32(1)
                        if q1e == cutlass.Int32(1):
                            em1 = em1 | cutlass.Int32(1 << s)
                        if q2e == cutlass.Int32(1):
                            em2 = em2 | cutlass.Int32(1 << s)
                    u64 = cutlass.Int64(-1)
                    if tid < ntail:
                        u64 = cutlass.Int64(fkey(tval))
                    q1e = cutlass.Int32(0)
                    q2e = cutlass.Int32(0)
                    if u64 > ethr:
                        q1e = cutlass.Int32(1)
                    if u64 == ethr:
                        det_u = cutlass.Uint32(0xFFFFFFFF) - cutlass.Uint32(tix)
                        if det_ph == cutlass.Int32(0):
                            if cutlass.Uint32(det_u) >= cutlass.Uint32(det_lo):
                                if cutlass.Uint32(det_u) <= cutlass.Uint32(det_hi):
                                    det_b = (
                                        cutlass.Uint32(det_u) - cutlass.Uint32(det_lo)
                                    ) >> cutlass.Uint32(det_sh)
                                    if det_b > cutlass.Uint32(self.nbh - 1):
                                        det_b = cutlass.Uint32(self.nbh - 1)
                                    atomic_add_cta(
                                        s_hist.iterator + cutlass.Int32(det_b), cutlass.Int32(1)
                                    )
                        else:
                            if cutlass.Int64(cutlass.Uint32(det_u)) > det_thr:
                                q2e = cutlass.Int32(1)
                    if det_ph == cutlass.Int32(1):
                        c1 = popc(em1) + q1e
                        c2 = popc(em2) + q2e
                        w1 = cutlass.Int32(warp_add_i32(c1))
                        w2 = cutlass.Int32(warp_add_i32(c2))
                        x1 = cutlass.Int32(0)
                        x1 = x1 + (
                            popc(
                                ballot(
                                    ((c1 >> cutlass.Int32(0)) & cutlass.Int32(1))
                                    == cutlass.Int32(1)
                                )
                                & lml
                            )
                            << cutlass.Int32(0)
                        )
                        x1 = x1 + (
                            popc(
                                ballot(
                                    ((c1 >> cutlass.Int32(1)) & cutlass.Int32(1))
                                    == cutlass.Int32(1)
                                )
                                & lml
                            )
                            << cutlass.Int32(1)
                        )
                        x1 = x1 + (
                            popc(
                                ballot(
                                    ((c1 >> cutlass.Int32(2)) & cutlass.Int32(1))
                                    == cutlass.Int32(1)
                                )
                                & lml
                            )
                            << cutlass.Int32(2)
                        )
                        x1 = x1 + (
                            popc(
                                ballot(
                                    ((c1 >> cutlass.Int32(3)) & cutlass.Int32(1))
                                    == cutlass.Int32(1)
                                )
                                & lml
                            )
                            << cutlass.Int32(3)
                        )
                        x1 = x1 + (
                            popc(
                                ballot(
                                    ((c1 >> cutlass.Int32(4)) & cutlass.Int32(1))
                                    == cutlass.Int32(1)
                                )
                                & lml
                            )
                            << cutlass.Int32(4)
                        )
                        if cutlass.const_expr(S + 1 >= 32):
                            x1 = x1 + (
                                popc(
                                    ballot(
                                        ((c1 >> cutlass.Int32(5)) & cutlass.Int32(1))
                                        == cutlass.Int32(1)
                                    )
                                    & lml
                                )
                                << cutlass.Int32(5)
                            )
                        x2 = cutlass.Int32(0)
                        x2 = x2 + (
                            popc(
                                ballot(
                                    ((c2 >> cutlass.Int32(0)) & cutlass.Int32(1))
                                    == cutlass.Int32(1)
                                )
                                & lml
                            )
                            << cutlass.Int32(0)
                        )
                        x2 = x2 + (
                            popc(
                                ballot(
                                    ((c2 >> cutlass.Int32(1)) & cutlass.Int32(1))
                                    == cutlass.Int32(1)
                                )
                                & lml
                            )
                            << cutlass.Int32(1)
                        )
                        x2 = x2 + (
                            popc(
                                ballot(
                                    ((c2 >> cutlass.Int32(2)) & cutlass.Int32(1))
                                    == cutlass.Int32(1)
                                )
                                & lml
                            )
                            << cutlass.Int32(2)
                        )
                        x2 = x2 + (
                            popc(
                                ballot(
                                    ((c2 >> cutlass.Int32(3)) & cutlass.Int32(1))
                                    == cutlass.Int32(1)
                                )
                                & lml
                            )
                            << cutlass.Int32(3)
                        )
                        x2 = x2 + (
                            popc(
                                ballot(
                                    ((c2 >> cutlass.Int32(4)) & cutlass.Int32(1))
                                    == cutlass.Int32(1)
                                )
                                & lml
                            )
                            << cutlass.Int32(4)
                        )
                        if cutlass.const_expr(S + 1 >= 32):
                            x2 = x2 + (
                                popc(
                                    ballot(
                                        ((c2 >> cutlass.Int32(5)) & cutlass.Int32(1))
                                        == cutlass.Int32(1)
                                    )
                                    & lml
                                )
                                << cutlass.Int32(5)
                            )
                        if lane == cutlass.Int32(0):
                            s_dw[dw] = w1
                            s_dw[32 + dw] = w2
                        cute.arch.barrier()  # DET SLOTS: warp totals published
                        v1 = cutlass.Int32(0)
                        v2 = cutlass.Int32(0)
                        if lane < cutlass.Int32(NW):
                            v1 = s_dw[lane]
                            v2 = s_dw[32 + lane]
                        tot1 = cutlass.Int32(warp_add_i32(v1))
                        tot2 = cutlass.Int32(warp_add_i32(v2))
                        if lane >= dw:
                            v1 = cutlass.Int32(0)
                            v2 = cutlass.Int32(0)
                        b1 = cutlass.Int32(warp_add_i32(v1))
                        b2 = cutlass.Int32(warp_add_i32(v2))
                        p1e = b1 + x1
                        p2e = b2 + x2
                        ewm = em1
                        while ewm != cutlass.Int32(0):
                            esd = ffs_m1(ewm)
                            exv = (
                                (tid + (esd >> cutlass.Int32(2)) * cutlass.Int32(self.blk))
                                << cutlass.Int32(2)
                            ) + (esd & cutlass.Int32(3))
                            if p1e < nA:
                                out_row[p1e] = exv
                            p1e = p1e + cutlass.Int32(1)
                            ewm = ewm & (ewm - cutlass.Int32(1))
                        if q1e == cutlass.Int32(1):
                            if p1e < nA:
                                out_row[p1e] = tix
                        ewm = em2
                        while ewm != cutlass.Int32(0):
                            esd = ffs_m1(ewm)
                            exv = (
                                (tid + (esd >> cutlass.Int32(2)) * cutlass.Int32(self.blk))
                                << cutlass.Int32(2)
                            ) + (esd & cutlass.Int32(3))
                            if p2e < nT:
                                out_row[nA + p2e] = exv
                            p2e = p2e + cutlass.Int32(1)
                            ewm = ewm & (ewm - cutlass.Int32(1))
                        if q2e == cutlass.Int32(1):
                            if p2e < nT:
                                out_row[nA + p2e] = tix
                    if det_ph == cutlass.Int32(0):
                        cute.arch.barrier()  # ---- barrier (det level hist) ----
                        if cutlass.const_expr(self.nbh > 1024):
                            scan_cross_w(
                                s_hist, s_ws, det_need, tid, s_res, blk=self.blk, nb=self.nbh
                            )
                        else:
                            find_cross(s_hist, det_need, tid, s_res, nb=self.nbh)
                        cute.arch.barrier()  # ---- barrier (det level scan) ----
                        det_need = det_need - s_res[RES_ABOVE]
                        det_m = s_res[RES_M]
                        det_bin = s_res[RES_B]
                        det_nlo = cutlass.Uint32(det_lo) + (
                            cutlass.Uint32(det_bin) << cutlass.Uint32(det_sh)
                        )
                        if det_bin != cutlass.Int32(self.nbh - 1):
                            det_top = det_nlo + (
                                (cutlass.Uint32(1) << cutlass.Uint32(det_sh)) - cutlass.Uint32(1)
                            )
                            # clamped top edge: the ~index range starts at 2^32 - n,
                            # so the top bin can wrap past 2^32 - 1
                            if cutlass.Uint32(det_top) >= cutlass.Uint32(det_nlo):
                                if cutlass.Uint32(det_top) < cutlass.Uint32(det_hi):
                                    det_hi = det_top
                        det_lo = det_nlo
                        if det_need == det_m:
                            det_thr = cutlass.Int64(cutlass.Uint32(det_lo)) - cutlass.Int64(1)
                            det_ph = cutlass.Int32(1)
                        elif cutlass.Uint32(det_lo) >= cutlass.Uint32(det_hi):
                            det_thr = cutlass.Int64(cutlass.Uint32(det_lo)) - cutlass.Int64(1)
                            det_ph = cutlass.Int32(1)
                    else:
                        det_ph = cutlass.Int32(2)
            else:
                if cutlass.const_expr(S <= 4 and self.nbh <= 1024):
                    HIf = cutlass.Float32(_POS_INF)
                    LOf = cutlass.Float32(_POS_INF)
                    if whole == cutlass.Int32(1):
                        HIf = cutlass.Float32(Bv)
                    else:
                        if Bv < cutlass.Int32(self.nbh - 1):
                            HIf = cutlass.Float32(Bv + cutlass.Int32(1))
                        LOf = cutlass.Float32(Bv)
                    m1 = cutlass.Int32(0)
                    m2 = cutlass.Int32(0)
                    for s in cutlass.range_constexpr(S):
                        if cutlass.const_expr(self.brl):
                            q = _fmaf__reg(_val(frags, s), SC, CQ)
                        else:
                            q = _fmaf__reg(_val(frags, s) - Tv, SC, OFFf)
                        if q >= HIf:
                            m1 = m1 | cutlass.Int32(1 << s)
                        else:
                            if q >= LOf:
                                m2 = m2 | cutlass.Int32(1 << s)
                    if cutlass.const_expr(self.brl):
                        qt3 = _fmaf__reg(tval, SC, CQ)
                    else:
                        qt3 = _fmaf__reg(tval - Tv, SC, OFFf)
                    t1 = cutlass.Int32(0)
                    t2 = cutlass.Int32(0)
                    if qt3 >= HIf:
                        t1 = cutlass.Int32(1)
                    else:
                        if qt3 >= LOf:
                            t2 = cutlass.Int32(1)
                    lml = cutlass.Int32(cute.arch.lanemask_lt())
                    w1 = cutlass.Int32(warp_add_i32(popc(m1) + t1))
                    w2 = cutlass.Int32(warp_add_i32(popc(m2) + t2))
                    if lane == cutlass.Int32(0):
                        s_dw[dw] = w1
                        s_dw[32 + dw] = w2
                    cute.arch.barrier()  # DET SLOTS: warp totals published
                    v1 = cutlass.Int32(0)
                    v2 = cutlass.Int32(0)
                    if lane < cutlass.Int32(NW):
                        v1 = s_dw[lane]
                        v2 = s_dw[32 + lane]
                    tot1 = cutlass.Int32(warp_add_i32(v1))
                    tot2 = cutlass.Int32(warp_add_i32(v2))
                    if lane >= dw:
                        v1 = cutlass.Int32(0)
                        v2 = cutlass.Int32(0)
                    b1 = cutlass.Int32(warp_add_i32(v1))
                    b2 = cutlass.Int32(warp_add_i32(v2))
                    if tid == cutlass.Int32(0):
                        s_cnt[0] = tot1
                        s_cnt[1] = tot2
                    lim1 = above
                    if whole == cutlass.Int32(1):
                        lim1 = k
                    for s in cutlass.range_constexpr(S):
                        bit1 = (m1 >> cutlass.Int32(s)) & cutlass.Int32(1)
                        bit2 = (m2 >> cutlass.Int32(s)) & cutlass.Int32(1)
                        n1 = ballot(bit1 == cutlass.Int32(1))
                        n2 = ballot(bit2 == cutlass.Int32(1))
                        idx = (
                            (tid + cutlass.Int32((s // 4) * self.blk)) << cutlass.Int32(2)
                        ) + cutlass.Int32(s % 4)
                        pp1 = b1 + popc(n1 & lml)
                        pp2 = b2 + popc(n2 & lml)
                        if pp1 >= lim1:
                            bit1 = cutlass.Int32(0)
                        if pp2 >= cmp_:
                            bit2 = cutlass.Int32(0)
                        if bit1 == cutlass.Int32(1):
                            out_row[pp1] = idx
                        if bit2 == cutlass.Int32(1):
                            ck[pp2] = fkey(_val(frags, s))
                        if bit2 == cutlass.Int32(1):
                            ci[pp2] = idx
                        b1 = b1 + popc(n1)
                        b2 = b2 + popc(n2)
                    n1 = ballot(t1 == cutlass.Int32(1))
                    n2 = ballot(t2 == cutlass.Int32(1))
                    pp1 = b1 + popc(n1 & lml)
                    pp2 = b2 + popc(n2 & lml)
                    if pp1 >= lim1:
                        t1 = cutlass.Int32(0)
                    if pp2 >= cmp_:
                        t2 = cutlass.Int32(0)
                    if t1 == cutlass.Int32(1):
                        out_row[pp1] = tix
                    if t2 == cutlass.Int32(1):
                        ck[pp2] = fkey(tval)
                    if t2 == cutlass.Int32(1):
                        ci[pp2] = tix

                else:
                    HIf = cutlass.Float32(_POS_INF)
                    LOf = cutlass.Float32(_POS_INF)
                    if whole == cutlass.Int32(1):
                        HIf = cutlass.Float32(Bv)
                    else:
                        if Bv < cutlass.Int32(self.nbh - 1):
                            HIf = cutlass.Float32(Bv + cutlass.Int32(1))
                        LOf = cutlass.Float32(Bv)
                    m1 = cutlass.Int32(0)
                    m2 = cutlass.Int32(0)
                    for s in cutlass.range_constexpr(S):
                        if cutlass.const_expr(self.brl):
                            q = _fmaf__reg(_val(frags, s), SC, CQ)
                        else:
                            q = _fmaf__reg(_val(frags, s) - Tv, SC, OFFf)
                        if q >= HIf:
                            m1 = m1 | cutlass.Int32(1 << s)
                        else:
                            if q >= LOf:
                                m2 = m2 | cutlass.Int32(1 << s)
                    if cutlass.const_expr(self.brl):
                        qt3 = _fmaf__reg(tval, SC, CQ)
                    else:
                        qt3 = _fmaf__reg(tval - Tv, SC, OFFf)
                    t1 = cutlass.Int32(0)
                    t2 = cutlass.Int32(0)
                    if qt3 >= HIf:
                        t1 = cutlass.Int32(1)
                    else:
                        if qt3 >= LOf:
                            t2 = cutlass.Int32(1)
                    lml = cutlass.Int32(cute.arch.lanemask_lt())
                    c1 = popc(m1) + t1
                    c2 = popc(m2) + t2
                    w1 = cutlass.Int32(warp_add_i32(c1))
                    w2 = cutlass.Int32(warp_add_i32(c2))
                    x1 = cutlass.Int32(0)
                    x1 = x1 + (
                        popc(
                            ballot(
                                ((c1 >> cutlass.Int32(0)) & cutlass.Int32(1)) == cutlass.Int32(1)
                            )
                            & lml
                        )
                        << cutlass.Int32(0)
                    )
                    x1 = x1 + (
                        popc(
                            ballot(
                                ((c1 >> cutlass.Int32(1)) & cutlass.Int32(1)) == cutlass.Int32(1)
                            )
                            & lml
                        )
                        << cutlass.Int32(1)
                    )
                    x1 = x1 + (
                        popc(
                            ballot(
                                ((c1 >> cutlass.Int32(2)) & cutlass.Int32(1)) == cutlass.Int32(1)
                            )
                            & lml
                        )
                        << cutlass.Int32(2)
                    )
                    x1 = x1 + (
                        popc(
                            ballot(
                                ((c1 >> cutlass.Int32(3)) & cutlass.Int32(1)) == cutlass.Int32(1)
                            )
                            & lml
                        )
                        << cutlass.Int32(3)
                    )
                    x1 = x1 + (
                        popc(
                            ballot(
                                ((c1 >> cutlass.Int32(4)) & cutlass.Int32(1)) == cutlass.Int32(1)
                            )
                            & lml
                        )
                        << cutlass.Int32(4)
                    )
                    if cutlass.const_expr(S + 1 >= 32):
                        x1 = x1 + (
                            popc(
                                ballot(
                                    ((c1 >> cutlass.Int32(5)) & cutlass.Int32(1))
                                    == cutlass.Int32(1)
                                )
                                & lml
                            )
                            << cutlass.Int32(5)
                        )
                    x2 = cutlass.Int32(0)
                    x2 = x2 + (
                        popc(
                            ballot(
                                ((c2 >> cutlass.Int32(0)) & cutlass.Int32(1)) == cutlass.Int32(1)
                            )
                            & lml
                        )
                        << cutlass.Int32(0)
                    )
                    x2 = x2 + (
                        popc(
                            ballot(
                                ((c2 >> cutlass.Int32(1)) & cutlass.Int32(1)) == cutlass.Int32(1)
                            )
                            & lml
                        )
                        << cutlass.Int32(1)
                    )
                    x2 = x2 + (
                        popc(
                            ballot(
                                ((c2 >> cutlass.Int32(2)) & cutlass.Int32(1)) == cutlass.Int32(1)
                            )
                            & lml
                        )
                        << cutlass.Int32(2)
                    )
                    x2 = x2 + (
                        popc(
                            ballot(
                                ((c2 >> cutlass.Int32(3)) & cutlass.Int32(1)) == cutlass.Int32(1)
                            )
                            & lml
                        )
                        << cutlass.Int32(3)
                    )
                    x2 = x2 + (
                        popc(
                            ballot(
                                ((c2 >> cutlass.Int32(4)) & cutlass.Int32(1)) == cutlass.Int32(1)
                            )
                            & lml
                        )
                        << cutlass.Int32(4)
                    )
                    if cutlass.const_expr(S + 1 >= 32):
                        x2 = x2 + (
                            popc(
                                ballot(
                                    ((c2 >> cutlass.Int32(5)) & cutlass.Int32(1))
                                    == cutlass.Int32(1)
                                )
                                & lml
                            )
                            << cutlass.Int32(5)
                        )
                    if lane == cutlass.Int32(0):
                        s_dw[dw] = w1
                        s_dw[32 + dw] = w2
                    cute.arch.barrier()  # DET SLOTS: warp totals published
                    v1 = cutlass.Int32(0)
                    v2 = cutlass.Int32(0)
                    if lane < cutlass.Int32(NW):
                        v1 = s_dw[lane]
                        v2 = s_dw[32 + lane]
                    tot1 = cutlass.Int32(warp_add_i32(v1))
                    tot2 = cutlass.Int32(warp_add_i32(v2))
                    if lane >= dw:
                        v1 = cutlass.Int32(0)
                        v2 = cutlass.Int32(0)
                    b1 = cutlass.Int32(warp_add_i32(v1))
                    b2 = cutlass.Int32(warp_add_i32(v2))
                    if tid == cutlass.Int32(0):
                        s_cnt[0] = tot1
                        s_cnt[1] = tot2
                    p1 = b1 + x1
                    p2 = b2 + x2
                    lim1 = above
                    if whole == cutlass.Int32(1):
                        lim1 = k
                    ewm = m1
                    while ewm != cutlass.Int32(0):
                        esd = ffs_m1(ewm)
                        idx = (
                            (tid + (esd >> cutlass.Int32(2)) * cutlass.Int32(self.blk))
                            << cutlass.Int32(2)
                        ) + (esd & cutlass.Int32(3))
                        if p1 < lim1:
                            out_row[p1] = idx
                        p1 = p1 + cutlass.Int32(1)
                        ewm = ewm & (ewm - cutlass.Int32(1))
                    if t1 == cutlass.Int32(1):
                        if p1 < lim1:
                            out_row[p1] = tix
                        p1 = p1 + cutlass.Int32(1)
                    if m2 != cutlass.Int32(0):  # static-unrolled
                        for s in cutlass.range_constexpr(S):
                            if (m2 & cutlass.Int32(1 << s)) != cutlass.Int32(0):
                                idx = (
                                    (tid + cutlass.Int32((s // 4) * self.blk)) << cutlass.Int32(2)
                                ) + cutlass.Int32(s % 4)
                                if p2 < cmp_:
                                    ck[p2] = fkey(_val(frags, s))
                                    ci[p2] = idx
                                p2 = p2 + cutlass.Int32(1)
                    if t2 == cutlass.Int32(1):
                        if p2 < cmp_:
                            ck[p2] = fkey(tval)
                            ci[p2] = tix
                        p2 = p2 + cutlass.Int32(1)
                if whole == cutlass.Int32(0):
                    cute.arch.barrier()  # emit done
                    mc = s_cnt[1]  # DET SLOTS: emitted crossing count (every template)
                    if mc > cmp_:
                        mc = cmp_
                    quad = cutlass.Int32(0)
                    if mc >= m:
                        if mc <= qc:
                            quad = cutlass.Int32(1)
                    if quad == cutlass.Int32(1):
                        i = tid
                        while i < mc:
                            uq = cutlass.Uint32(ck[i])
                            uci = ci[i]  # DET: logical-index tie-break
                            r = cutlass.Int32(0)
                            j = cutlass.Int32(0)
                            while j < mc:
                                vq = cutlass.Uint32(ck[j])
                                tinc = cutlass.Int32(0)
                                if vq > uq:
                                    tinc = cutlass.Int32(1)
                                if vq == uq:
                                    if ci[j] < uci:
                                        tinc = cutlass.Int32(1)
                                r = r + tinc
                                j = j + cutlass.Int32(1)
                            if r < need:
                                out_row[above + r] = uci
                            i = i + cutlass.Int32(BLK)
                    else:
                        if tid == cutlass.Int32(0):
                            s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                            s_kmm[1] = cutlass.Uint32(0)
                        cute.arch.barrier()  # kmm init
                        i = tid
                        while i < mc:
                            kv = cutlass.Uint32(ck[i])
                            atomic_min_cta(s_kmm.iterator, kv)
                            atomic_max_cta(s_kmm.iterator + 1, kv)
                            i = i + cutlass.Int32(BLK)
                        cute.arch.barrier()  # key range published
                        rlo = cutlass.Uint32(s_kmm[0])
                        rhi = cutlass.Uint32(s_kmm[1])
                        ethr = cutlass.Int64(rlo)
                        aboveC = cutlass.Int32(0)
                        needC = need
                        mm = mc
                        lev = cutlass.Int32(0)
                        done = cutlass.Int32(0)
                        while done == cutlass.Int32(0):
                            if needC == mm:
                                ethr = cutlass.Int64(rlo) - cutlass.Int64(1)
                                aboveC = aboveC + mm
                                needC = cutlass.Int32(0)
                                done = cutlass.Int32(1)
                            if done == cutlass.Int32(0):
                                if rlo >= rhi:
                                    ethr = cutlass.Int64(rlo)
                                    done = cutlass.Int32(1)
                                if lev >= cutlass.Int32(6):
                                    ethr = cutlass.Int64(rlo)
                                    done = cutlass.Int32(1)
                            if done == cutlass.Int32(0):
                                d2 = rhi - rlo
                                b2w = cutlass.Int32(32) - clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1))
                                )
                                sh2 = cutlass.Int32(0)
                                if b2w > cutlass.Int32(LNBH):
                                    sh2 = b2w - cutlass.Int32(LNBH)
                                for z in cutlass.range_constexpr(self.nbh // self.blk):
                                    s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                                cute.arch.barrier()  # level clear
                                i = tid
                                while i < mc:
                                    unar = cutlass.Uint32(ck[i])
                                    if unar >= rlo:
                                        if unar <= rhi:
                                            bnn = (unar - rlo) >> cutlass.Uint32(sh2)
                                            bnn = _umin_u32(bnn, cutlass.Uint32(self.nbh - 1))
                                            atomic_add_cta(
                                                s_hist.iterator + cutlass.Int32(bnn),
                                                cutlass.Int32(1),
                                            )
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()  # level hist
                                if cutlass.const_expr(self.nbh > 1024):
                                    scan_cross_w(
                                        s_hist,
                                        s_ws,
                                        needC,
                                        tid,
                                        s_res,
                                        blk=self.blk,
                                        nb=self.nbh,
                                    )
                                else:
                                    find_cross(s_hist, needC, tid, s_res, nb=self.nbh)
                                cute.arch.barrier()  # level scan
                                aboveC = aboveC + s_res[RES_ABOVE]
                                needC = needC - s_res[RES_ABOVE]
                                mm = s_res[RES_M]
                                b_lv = s_res[RES_B]
                                nlo = rlo + (cutlass.Uint32(b_lv) << cutlass.Uint32(sh2))
                                if b_lv != cutlass.Int32(self.nbh - 1):
                                    rhi = nlo + (
                                        (cutlass.Uint32(1) << cutlass.Uint32(sh2))
                                        - cutlass.Uint32(1)
                                    )
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        if tid == cutlass.Int32(0):
                            s_e12[0] = cutlass.Int32(0)
                            s_e12[1] = cutlass.Int32(0)
                        cute.arch.barrier()  # emit counters
                        lml = cutlass.Int32(cute.arch.lanemask_lt())
                        it2 = (mc + cutlass.Int32(self.blk - 1)) // cutlass.Int32(self.blk)
                        run1 = cutlass.Int32(0)  # DET SLOTS: running class offsets across it
                        run2 = cutlass.Int32(0)
                        det_lo = cutlass.Uint32(0xFFFFFFFF) - (
                            cutlass.Uint32(n) - cutlass.Uint32(1)
                        )  # ~idx >= 2^32 - n
                        det_hi = cutlass.Uint32(0xFFFFFFFF)
                        det_need = needC
                        det_thr = cutlass.Int64(-1)
                        det_sh = cutlass.Uint32(0)
                        det_ph = cutlass.Int32(1)
                        if needC > cutlass.Int32(0):
                            if mm > needC:
                                det_ph = cutlass.Int32(0)
                        while det_ph <= cutlass.Int32(1):
                            if det_ph == cutlass.Int32(0):
                                det_d = cutlass.Uint32(det_hi) - cutlass.Uint32(det_lo)
                                det_bw = cutlass.Int32(32) - clz_i32(
                                    cutlass.Int32(det_d | cutlass.Uint32(1))
                                )
                                det_s = det_bw - cutlass.Int32(LNBH)
                                if det_s < cutlass.Int32(0):
                                    det_s = cutlass.Int32(0)
                                det_sh = cutlass.Uint32(det_s)
                                for z in cutlass.range_constexpr(self.nbh // self.blk):
                                    s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                                cute.arch.barrier()  # ---- barrier (det level clear) ----
                            it = cutlass.Int32(0)
                            while it < it2:
                                i = it * cutlass.Int32(BLK) + tid
                                uke = cutlass.Uint32(0)
                                idv = cutlass.Int32(0)
                                if i < mc:
                                    uke = cutlass.Uint32(ck[i])
                                    idv = ci[i]
                                q1f = cutlass.Int32(0)
                                q2f = cutlass.Int32(0)
                                if i < mc:
                                    if cutlass.Int64(uke) > ethr:
                                        q1f = cutlass.Int32(1)
                                    if cutlass.Int64(uke) == ethr:
                                        det_u = cutlass.Uint32(0xFFFFFFFF) - cutlass.Uint32(idv)
                                        if det_ph == cutlass.Int32(0):
                                            if cutlass.Uint32(det_u) >= cutlass.Uint32(det_lo):
                                                if cutlass.Uint32(det_u) <= cutlass.Uint32(det_hi):
                                                    det_b = (
                                                        cutlass.Uint32(det_u)
                                                        - cutlass.Uint32(det_lo)
                                                    ) >> cutlass.Uint32(det_sh)
                                                    if det_b > cutlass.Uint32(self.nbh - 1):
                                                        det_b = cutlass.Uint32(self.nbh - 1)
                                                    atomic_add_cta(
                                                        s_hist.iterator + cutlass.Int32(det_b),
                                                        cutlass.Int32(1),
                                                    )
                                        else:
                                            if cutlass.Int64(cutlass.Uint32(det_u)) > det_thr:
                                                q2f = cutlass.Int32(1)
                                if det_ph == cutlass.Int32(1):
                                    n1 = ballot(q1f == cutlass.Int32(1))
                                    n2 = ballot(q2f == cutlass.Int32(1))
                                    if lane == cutlass.Int32(0):
                                        s_dw[dw] = popc(n1)
                                        s_dw[32 + dw] = popc(n2)
                                    cute.arch.barrier()  # DET SLOTS: warp totals published
                                    v1 = cutlass.Int32(0)
                                    v2 = cutlass.Int32(0)
                                    if lane < cutlass.Int32(NW):
                                        v1 = s_dw[lane]
                                        v2 = s_dw[32 + lane]
                                    tot1 = cutlass.Int32(warp_add_i32(v1))
                                    tot2 = cutlass.Int32(warp_add_i32(v2))
                                    if lane >= dw:
                                        v1 = cutlass.Int32(0)
                                        v2 = cutlass.Int32(0)
                                    b1 = cutlass.Int32(warp_add_i32(v1))
                                    b2 = cutlass.Int32(warp_add_i32(v2))
                                    b1 = run1 + b1
                                    b2 = run2 + b2
                                    run1 = run1 + tot1
                                    run2 = run2 + tot2
                                    if it + cutlass.Int32(1) < it2:
                                        cute.arch.barrier()  # DET SLOTS: scratch reusable by the next it
                                    p1e = b1 + popc(n1 & lml)
                                    p2e = b2 + popc(n2 & lml)
                                    if q1f == cutlass.Int32(1):
                                        if p1e < aboveC:
                                            out_row[above + p1e] = idv
                                    if q2f == cutlass.Int32(1):
                                        if p2e < needC:
                                            out_row[above + aboveC + p2e] = idv
                                it = it + cutlass.Int32(1)
                            if det_ph == cutlass.Int32(0):
                                cute.arch.barrier()  # ---- barrier (det level hist) ----
                                if cutlass.const_expr(self.nbh > 1024):
                                    scan_cross_w(
                                        s_hist,
                                        s_ws,
                                        det_need,
                                        tid,
                                        s_res,
                                        blk=self.blk,
                                        nb=self.nbh,
                                    )
                                else:
                                    find_cross(s_hist, det_need, tid, s_res, nb=self.nbh)
                                cute.arch.barrier()  # ---- barrier (det level scan) ----
                                det_need = det_need - s_res[RES_ABOVE]
                                det_m = s_res[RES_M]
                                det_bin = s_res[RES_B]
                                det_nlo = cutlass.Uint32(det_lo) + (
                                    cutlass.Uint32(det_bin) << cutlass.Uint32(det_sh)
                                )
                                if det_bin != cutlass.Int32(self.nbh - 1):
                                    det_top = det_nlo + (
                                        (cutlass.Uint32(1) << cutlass.Uint32(det_sh))
                                        - cutlass.Uint32(1)
                                    )
                                    # clamped top edge: the ~index range starts at 2^32 - n,
                                    # so the top bin can wrap past 2^32 - 1
                                    if cutlass.Uint32(det_top) >= cutlass.Uint32(det_nlo):
                                        if cutlass.Uint32(det_top) < cutlass.Uint32(det_hi):
                                            det_hi = det_top
                                det_lo = det_nlo
                                if det_need == det_m:
                                    det_thr = cutlass.Int64(cutlass.Uint32(det_lo)) - cutlass.Int64(
                                        1
                                    )
                                    det_ph = cutlass.Int32(1)
                                elif cutlass.Uint32(det_lo) >= cutlass.Uint32(det_hi):
                                    det_thr = cutlass.Int64(cutlass.Uint32(det_lo)) - cutlass.Int64(
                                        1
                                    )
                                    det_ph = cutlass.Int32(1)
                            else:
                                det_ph = cutlass.Int32(2)

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        cmp_: cutlass.Int32,
        qc: cutlass.Int32,
        smem_bytes: cutlass.Int32,
        stream,
    ):
        b = logits.shape[0]
        self.kern(logits, pre_idx, kv_lens, out, n, cmp_, qc, smem_bytes).launch(
            grid=(b, 1, 1),
            block=(self.blk, 1, 1),
            stream=stream,
            smem=smem_bytes,
            min_blocks_per_mp=self.minb,
        )


# ---------------------------------------------------------------------------
# host wrapper: compile cache + route()-driven entry
# ---------------------------------------------------------------------------
_COMPILE_CACHE__reg: dict = {}


def get_compiled__reg(
    tpl: tuple,
    varlen: bool = False,
    next_n: int = 1,
    cr_shift: int = 0,
    hint_free: bool = False,
) -> Any:
    """Compile (or fetch) the variant for constexpr tuple
    (BLK, VPT, MINB, KPT, CUR, DEG, IMG, NBH)."""
    key = (
        tuple(tpl),
        bool(varlen),
        int(next_n),
        int(cr_shift),
        bool(hint_free),
    )
    compiled = _COMPILE_CACHE__reg.get(key)
    if compiled is None:
        from cutlass.cute import runtime as _crt

        blk, vpt, minb, kpt, cur, deg, img, nbh = tpl
        kernel = GvrTopkRegKernel(
            blk,
            vpt,
            minb,
            kpt,
            cur,
            deg,
            img,
            nbh,
            varlen=varlen,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=hint_free,
        )
        nb_, nc_ = cute.sym_int(), cute.sym_int()
        nb2_, nc2_ = cute.sym_int(), cute.sym_int()
        nb3_, nc3_ = cute.sym_int(), cute.sym_int()
        lg_fake = _crt.make_fake_compact_tensor(
            cutlass.Float32, (nb_, nc_), stride_order=(1, 0), assumed_align=16
        )
        pi_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (nb2_, nc2_), stride_order=(1, 0), assumed_align=16
        )
        out_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (nb3_, nc3_), stride_order=(1, 0), assumed_align=16
        )
        v0_ = cute.sym_int()
        kv_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (v0_,), stride_order=(0,), assumed_align=4
        )
        fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
        with _no_carveout():
            compiled = cute.compile(
                kernel,
                lg_fake,
                pi_fake,
                kv_fake,
                out_fake,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                stream=fake_stream,
                options="--enable-tvm-ffi",
            )
        _COMPILE_CACHE__reg[key] = compiled
    return compiled
