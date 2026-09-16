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

"""Native BF16 self-sampling GVR decode kernels for Blackwell.

The four families share the original GVR crossing and emission helpers. BF16
loads, compact keys and dense-bin refinements preserve exact value selection.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass._mlir.dialects import llvm
from cutlass.cute import runtime as _crt
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

from .gvr_topk_decode_self_sampling import (
    _cluster_sync_aligned,
    _f32_smem_atom,
    _fabsf,
    _fabsf__regclus,
    _fmaf,
    _fmaf__clus,
    _fmaf__reg,
    _fmaf__regclus,
    _ld_shared_cluster_i32,
    _ld_shared_cluster_v4_u32,
    _ldcg_v2_i32,
    _ldg_f32_rs,
    _lds_v2_u64,
    _mapa_shared_cluster,
    _mapa_shared_cluster_addr,
    _no_carveout,
    _pin_i32,
    _pin_i64,
    _prefetch_l2,
    _red_shared_add1,
    _red_shared_add1__reg,
    _smem_addr_reg,
    _smem_addr_reg__reg,
    _smem_view,
    _smem_view__regclus,
    _st_g_u32,
    _st_g_u64,
    _st_s_v2_u32,
    _st_shared_cluster_i32,
    _st_shared_cluster_u64,
    _sts128_f32,
    _submul_asm,
    _umin_u32,
    _umin_u32__regclus,
    _val,
    _val__regclus,
    atomic_add_cta,
    atomic_add_u64_gpu,
    atomic_max_cta,
    atomic_min_cta,
    atomic_or_cta,
    ballot,
    clz_i32,
    f2s_rz,
    f2u_rz,
    f32_of_i32,
    f32_of_u32,
    ffs_m1,
    find_cross,
    fkey,
    fkey_bits,
    fmax_f32,
    fmin_f32,
    g2r_atom_f32,
    gather_hint,
    invkey,
    ld_g_f32x4,
    ld_g_i32,
    ldg_f32,
    popc,
    scan_cross,
    scan_cross0,
    scan_cross_w,
    smem_atom_i32_128,
    sts128_i32,
    threadfence_gpu,
    u32_of_f32,
    warp_incl_scan_add,
    warp_incl_scan_add2,
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


@dsl_user_op
def _ld_g_nc_v2_b32(gaddr, *, loc=None, ip=None):
    """Pinned `ld.global.nc.v2.b32` (CUDA `__ldg(const uint2*)`): one 8-byte
    vector = four bf16, returned as two packed Uint32 (lo = elems 0/1,
    hi = elems 2/3, little-endian). gaddr: Int64 byte address, 8B-aligned."""
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [gaddr.ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Uint32(llvm.extractvalue(T.i32(), ret, [0])),
        cutlass.Uint32(llvm.extractvalue(T.i32(), ret, [1])),
    )


def ld_g_bf16x4(copy_atom, base_addr, v_idx, frag):
    """bf16 twin of ld_g_f32x4: load bf16x4 vector #v_idx (8B units) from the
    gmem byte base and widen into frag[0..3] (f32, exact). copy_atom is kept
    for call-site parity with ld_g_f32x4 and is unused."""
    lo, hi = _ld_g_nc_v2_b32(base_addr + cutlass.Int64(v_idx) * cutlass.Int64(8))
    frag[0] = f32_of_u32(lo << cutlass.Uint32(16))
    frag[1] = f32_of_u32(lo & cutlass.Uint32(0xFFFF0000))
    frag[2] = f32_of_u32(hi << cutlass.Uint32(16))
    frag[3] = f32_of_u32(hi & cutlass.Uint32(0xFFFF0000))


@dsl_user_op
def _ld_g_nc_u16(gaddr, *, loc=None, ip=None):
    """`ld.global.nc.u16` scalar bf16 gather, zero-extended into a u32 reg."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [gaddr.ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u16 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_g_nc_v4_b32(gaddr, *, loc=None, ip=None):
    """Pinned `ld.global.nc.v4.b32` (CUDA `__ldg(const uint4*)`): one 16-byte
    vector = EIGHT bf16, returned as four packed Uint32 (little-endian pairs).
    gaddr: Int64 byte address, 16B-aligned."""
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [gaddr.ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(cutlass.Uint32(llvm.extractvalue(T.i32(), ret, [i])) for i in range(4))


def ld_g_bf16x8(copy_atom, base_addr, v_idx, frag):
    """bf16 16-byte vector loader: bf16x8 vector #v_idx (16B units) widened
    into frag[0..7] (f32, exact). Halves the P3 load-instruction count vs the
    8-byte form. copy_atom kept for call-site parity and unused."""
    a, b, c, d = _ld_g_nc_v4_b32(base_addr + cutlass.Int64(v_idx) * cutlass.Int64(16))
    frag[0] = f32_of_u32(a << cutlass.Uint32(16))
    frag[1] = f32_of_u32(a & cutlass.Uint32(0xFFFF0000))
    frag[2] = f32_of_u32(b << cutlass.Uint32(16))
    frag[3] = f32_of_u32(b & cutlass.Uint32(0xFFFF0000))
    frag[4] = f32_of_u32(c << cutlass.Uint32(16))
    frag[5] = f32_of_u32(c & cutlass.Uint32(0xFFFF0000))
    frag[6] = f32_of_u32(d << cutlass.Uint32(16))
    frag[7] = f32_of_u32(d & cutlass.Uint32(0xFFFF0000))


def ld_g_bf16x8_pk(copy_atom, base_addr, v_idx, pk):
    """bf16 16-byte vector loader, PACKED form: bf16x8 vector #v_idx lands in
    pk[0..3] as raw Uint32 pairs (no widening). The classify site unpacks in
    the compare, so a primed vector carries FOUR registers instead of eight —
    the prime/roll batches stay off the 64-register wall at full depth."""
    a, b, c, d = _ld_g_nc_v4_b32(base_addr + cutlass.Int64(v_idx) * cutlass.Int64(16))
    pk[0] = a
    pk[1] = b
    pk[2] = c
    pk[3] = d


@cute.jit
def bf16_ceilx2_from_f32(x):
    """Broadcast the least bf16 value >= ``x`` as packed bf16x2 bits."""
    bits = u32_of_f32(x)
    hi = bits >> cutlass.Uint32(16)
    # Clearing the low fp32 bits is round-toward-zero: already ceil for a
    # negative value, while a positive inexact value needs the next bf16.
    if (bits >> cutlass.Uint32(31)) == cutlass.Uint32(0):
        if (bits & cutlass.Uint32(0xFFFF)) != cutlass.Uint32(0):
            hi = hi + cutlass.Uint32(1)
    return hi | (hi << cutlass.Uint32(16))


@dsl_user_op
def bf16x2_ge_mask(pair, threshold_pair, *, loc=None, ip=None):
    """Two ordered bf16 comparisons returned in bits 0 and 1."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [pair.ir_value(loc=loc, ip=ip), threshold_pair.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p, q;\n\tsetp.ge.bf16x2 p|q, $1, $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t@q or.b32 $0, $0, 2;\n}",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def ld_g_bf16x8_pair(copy_atom, base_addr, v_idx, fa, fb):
    """bf16 16-byte vector split across two 4-wide f32 frags — the P1 sample
    (fsa, fsb) shape: ONE ld.global.nc.v4.b32 covers the same 8 elements the
    fp32 arm fetches with two float4 loads."""
    a, b, c, d = _ld_g_nc_v4_b32(base_addr + cutlass.Int64(v_idx) * cutlass.Int64(16))
    fa[0] = f32_of_u32(a << cutlass.Uint32(16))
    fa[1] = f32_of_u32(a & cutlass.Uint32(0xFFFF0000))
    fa[2] = f32_of_u32(b << cutlass.Uint32(16))
    fa[3] = f32_of_u32(b & cutlass.Uint32(0xFFFF0000))
    fb[0] = f32_of_u32(c << cutlass.Uint32(16))
    fb[1] = f32_of_u32(c & cutlass.Uint32(0xFFFF0000))
    fb[2] = f32_of_u32(d << cutlass.Uint32(16))
    fb[3] = f32_of_u32(d & cutlass.Uint32(0xFFFF0000))


def ldg_bf16(base_addr, idx):
    """bf16 twin of ldg_f32: scalar read-only 2B gather widened to f32."""
    u = _ld_g_nc_u16(base_addr + cutlass.Int64(idx) * cutlass.Int64(2))
    return f32_of_u32(u << cutlass.Uint32(16))


def ldg_bf16_rs(base_addr, idx, sc2):
    """bf16 twin of _ldg_f32_rs: byte stride (== 2) rides a caller register."""
    u = _ld_g_nc_u16(base_addr + cutlass.Int64(idx) * cutlass.Int64(sc2))
    return f32_of_u32(u << cutlass.Uint32(16))


@cute.jit
def merge_scan4_regclus(s_hist, s_mrg, s_ws, rank, target, tidx, s_res, cs: cutlass.Constexpr):
    """Merge 512 bins across ranks and publish biased cursors with four warps.

    All CTA threads must call this helper. It contains one CTA barrier; the
    caller supplies the final publication barrier before reading the results.
    """
    totals = cute.make_rmem_tensor((4,), cutlass.Int32)
    prefixes = cute.make_rmem_tensor((4,), cutlass.Int32)
    warp_prefix = cutlass.Int32(0)
    if tidx < cutlass.Int32(128):
        lane = tidx & cutlass.Int32(31)
        warp = tidx >> cutlass.Int32(5)
        byte_offset = tidx * cutlass.Int32(16)
        # Issue one aligned four-bin DSM load per peer before folding.
        peers = []
        for peer in cutlass.range_constexpr(cs):
            mapped = _mapa_shared_cluster(s_hist.iterator, cutlass.Int32(peer))
            peers.append(_ld_shared_cluster_v4_u32(mapped + byte_offset))
        for item in cutlass.range_constexpr(4):
            totals[item] = cutlass.Int32(0)
            prefixes[item] = cutlass.Int32(0)
        for peer in cutlass.range_constexpr(cs):
            for item in cutlass.range_constexpr(4):
                totals[item] = totals[item] + peers[peer][item]
                if cutlass.Int32(peer) < rank:
                    prefixes[item] = prefixes[item] + peers[peer][item]
        span_total = totals[0] + totals[1] + totals[2] + totals[3]
        warp_prefix = warp_incl_scan_add(span_total, lane)
        if lane == cutlass.Int32(31):
            s_ws[warp] = warp_prefix
    cute.arch.barrier()  # all four warp totals are available
    if tidx < cutlass.Int32(128):
        warp = tidx >> cutlass.Int32(5)
        global_total = cutlass.Int32(0)
        lower_warps = cutlass.Int32(0)
        for peer_warp in cutlass.range_constexpr(4):
            partial = s_ws[cutlass.Int32(peer_warp)]
            global_total = global_total + partial
            if cutlass.Int32(peer_warp) < warp:
                lower_warps = lower_warps + partial
        after = global_total - lower_warps - warp_prefix
        if tidx == cutlass.Int32(0):
            s_res[RES_TOT] = global_total
        cursors = cute.make_rmem_tensor((4,), cutlass.Int32)
        for item in cutlass.range_constexpr(3, -1, -1):
            count = totals[item]
            cursors[item] = after + prefixes[item]
            bin_index = tidx * cutlass.Int32(4) + cutlass.Int32(item)
            crossing = cutlass.Int32(0)
            if after < target:
                if after + count >= target:
                    crossing = cutlass.Int32(1)
                if bin_index == cutlass.Int32(0):
                    crossing = cutlass.Int32(1)
            if crossing != cutlass.Int32(0):
                s_res[RES_B] = bin_index
                s_res[RES_ABOVE] = after
                s_res[RES_M] = count
            after = after + count
        atom = smem_atom_i32_128()
        sts128_i32(atom, cursors, s_mrg.iterator.toint(), tidx * cutlass.Int32(16))


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
        v16: bool = False,
        dense: bool = False,
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
        # bf16-only value-in-word staging: exact bf16->f32 widening leaves
        # the low 16 value bits zero, so a cbuf word can hold value_hi16|idx16.
        # This removes scattered survivor rereads for eligible small envelopes.
        self.v16 = bool(v16) and dtype is not cutlass.Float32 and not self.split and not self.vstg
        if dtype is cutlass.Float32:
            self.pfd = (u if u < 4 else 4) if minb <= 2 else 0
        else:
            # bf16 primed slots are PACKED u32 pairs (4 regs per bf16x8
            # vector), so the fp32 prime-depth formula fits the register wall
            self.pfd = (u if u < 4 else 4) if minb <= 2 else 0
        self.pf = self.pfd > 0
        self.natt = 1 if split else 3
        # Dense bins are beneficial for K-heavy BF16 main-engine ladders.
        # Lighter K bands can overflow the window and lose to linear bins.
        self.dense = bool(dense) and dtype is cutlass.BFloat16 and not self.prefill
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
    def _dense_bin(self, xv, base):
        """Return an exact BF16 value bin, with the final bin as overflow.

        Negative and positive zeros may share bin zero after the lower clamp;
        they have identical numerical values and are interchangeable top-K ties.
        The base key is an exactly represented float scalar so the original
        classifier argument ABI and register lifetime remain unchanged.
        """
        delta = cutlass.Int32(fkey(xv) >> cutlass.Uint32(16)) - cutlass.Int32(base)
        if delta < cutlass.Int32(0):
            delta = cutlass.Int32(0)
        if delta > cutlass.Int32(self.nbs - 1):
            delta = cutlass.Int32(self.nbs - 1)
        return delta

    @cute.jit
    def _emitc(self, xv, idx, pos, TF, SC, hb, cb2, s_hist, s_cbuf, s_cbuf2):
        SCPB = self.scpb
        NBS = self.nbs
        if cutlass.const_expr(not self.split):
            if cutlass.const_expr(self.dense):
                bn_u = cutlass.Uint32(self._dense_bin(xv, SC))
            else:
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
                if cutlass.const_expr(self.v16):
                    s_cbuf[ps] = cutlass.Int32(u32_of_f32(xv) | cutlass.Uint32(idx))
                else:
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
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    n4v = n >> cutlass.Int32(2)
                else:
                    n4v = n >> cutlass.Int32(3)  # bf16: 8-elem 16B vectors
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
            _tshlim = 32768
            if cutlass.const_expr(self.dtype is not cutlass.Float32):
                _tshlim = 36864
            if tsh_en != cutlass.Int32(0):
                if (n >> cutlass.Int32(2)) <= cutlass.Int32(_tshlim):
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

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            n4 = n >> cutlass.Int32(2)
        else:
            n4 = n >> cutlass.Int32(3)  # bf16: 8-elem 16B vectors
        c0 = cutlass.Int32(0)
        c1 = n4
        if cutlass.const_expr(self.split):
            c0 = part * Q
            c1 = c0 + Q
            if c1 > n4:
                c1 = n4
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            tail0 = n4 << cutlass.Int32(2)
        else:
            tail0 = n4 << cutlass.Int32(3)
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
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    plim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)
                else:
                    plim4 = (npad >> cutlass.Int32(3)) - cutlass.Int32(1)
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
                    # bf16: 16B vectors again (8 elems each)
                    _prefetch_l2(x_addr + cutlass.Int64(pic) * cutlass.Int64(16))
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
                # one bf16x8 vector covers the same 8 sampled elements
                ld_g_bf16x8_pair(atom128, x_addr, tidx * SS2, fsa, fsb)
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
                ld_g_bf16x8_pair(atom128, x_addr, j * SS2, fma_, fmb_)
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
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                lim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)
            else:
                lim4 = (npad >> cutlass.Int32(3)) - cutlass.Int32(1)
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            pf = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(max(PFD, 1))]
        else:
            # PACKED u32 pairs (4 regs per bf16x8 vector; unpacked at classify)
            pf = [cute.make_rmem_tensor((4,), cutlass.Uint32) for _ in range(max(PFD, 1))]
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
                        ld_g_bf16x8_pk(atom128, x_addr, c0 + tidx + cutlass.Int32(uu * BLK), pf[uu])
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
                        ld_g_bf16x8_pk(atom128, x_addr, ic, pf[uu])
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
                            + cutlass.Int64(c0 + tidx + cutlass.Int32(uu * BLK)) * cutlass.Int64(16)
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
                            + cutlass.Int64(c0 + tidx + cutlass.Int32(uu * BLK)) * cutlass.Int64(16)
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
                            _prefetch_l2(x_addr + cutlass.Int64(ic) * cutlass.Int64(16))

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
                    ld_g_bf16x8_pair(atom128, x_addr, j * SS2, fma_, fmb_)
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

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            fr = [
                cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(max(U - PFD, 1))
            ]  # explicit batch
        else:
            fr = [
                cute.make_rmem_tensor((4,), cutlass.Uint32) for _ in range(max(U - PFD, 1))
            ]  # explicit batch (bf16x8 vector as PACKED u32 pairs)
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
                                    ld_g_bf16x8_pk(
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
                                    ld_g_bf16x8_pk(atom128, x_addr, ic, pf[uu])
                    if tidx < cutlass.Int32(NBS):
                        s_hist[tidx] = cutlass.Int32(0)
                    if tidx == cutlass.Int32(0):
                        s_scal[0] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (retry reset) ----

            TF = T  # window
            if cutlass.const_expr(self.dtype == cutlass.BFloat16):
                tf_bf16x2 = bf16_ceilx2_from_f32(TF)
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
            if cutlass.const_expr(self.dense):
                threshold_bits = tf_bf16x2 & cutlass.Uint32(0xFFFF0000)
                SC = cutlass.Float32(fkey_bits(threshold_bits) >> cutlass.Uint32(16))
            elif cutlass.const_expr(self.blk == 512):
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
            if cutlass.const_expr(
                self.dtype == cutlass.BFloat16 and not self.split and self.kpt >= 2
            ):
                # A single bf16x8 runt makes the whole CTA execute a partial
                # P3 tile.  Leave it for the already-coalesced scalar tail.
                if rem == cutlass.Int32(1):
                    nIt = nFull
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
                        for uu in cutlass.range_constexpr(U):
                            if cutlass.const_expr(uu < PFD):
                                vv = pf[uu]
                            else:
                                vv = fr[uu - PFD]
                            for q in cutlass.range_constexpr(4):
                                M = M | (cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q))
                    else:
                        for uu in cutlass.range_constexpr(PFD, U):
                            ld_g_bf16x8_pk(
                                atom128, x_addr, i0 + cutlass.Int32(uu * BLK), fr[uu - PFD]
                            )
                        for uu in cutlass.range_constexpr(U):
                            if cutlass.const_expr(uu < PFD):
                                vv = pf[uu]
                            else:
                                vv = fr[uu - PFD]
                            for q in cutlass.range_constexpr(4):
                                qm = bf16x2_ge_mask(vv[q], tf_bf16x2)
                                M = M | (cutlass.Int32(qm) << cutlass.Int32(uu * 8 + q * 2))
                else:  # partial body
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        for uu in cutlass.range_constexpr(PFD, U):
                            i_ = i0 + cutlass.Int32(uu * BLK)
                            ic = i_
                            if ic >= c1:
                                ic = lim4  # clamped address
                            ld_g_f32x4(atom128, x_addr, ic, fr[uu - PFD])
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
                                    M = M | (
                                        cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q)
                                    )
                    else:
                        for uu in cutlass.range_constexpr(PFD, U):
                            i_ = i0 + cutlass.Int32(uu * BLK)
                            ic = i_
                            if ic >= c1:
                                ic = lim4  # clamped address
                            ld_g_bf16x8_pk(atom128, x_addr, ic, fr[uu - PFD])
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
                                    qm = bf16x2_ge_mask(vv[q], tf_bf16x2)
                                    M = M | (cutlass.Int32(qm) << cutlass.Int32(uu * 8 + q * 2))
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
                                    ld_g_bf16x8_pk(
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
                                    ld_g_bf16x8_pk(atom128, x_addr, jc, pf[uu])
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
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        idx = (
                            (i0 + (bp >> cutlass.Int32(2)) * cutlass.Int32(BLK)) << cutlass.Int32(2)
                        ) + (bp & cutlass.Int32(3))
                    else:
                        idx = (
                            (i0 + (bp >> cutlass.Int32(3)) * cutlass.Int32(BLK)) << cutlass.Int32(3)
                        ) + (bp & cutlass.Int32(7))
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
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            idx2 = (
                                (i0 + (bp2 >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                                << cutlass.Int32(2)
                            ) + (bp2 & cutlass.Int32(3))
                        else:
                            idx2 = (
                                (i0 + (bp2 >> cutlass.Int32(3)) * cutlass.Int32(BLK))
                                << cutlass.Int32(3)
                            ) + (bp2 & cutlass.Int32(7))
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
            if cutlass.const_expr(
                self.dtype == cutlass.BFloat16 and not self.split and self.kpt >= 2
            ):
                # When the final bf16x8 vector was folded out of P3 above,
                # extend this tail eight elements to the left.
                if rem == cutlass.Int32(1):
                    i = i - cutlass.Int32(8)
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
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        lo2 = c0 << cutlass.Int32(2)
                        hi2 = c1 << cutlass.Int32(2)
                    else:
                        lo2 = c0 << cutlass.Int32(3)
                        hi2 = c1 << cutlass.Int32(3)
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
                            if cutlass.const_expr(self.dense):
                                bq = self._dense_bin(f32_of_i32(gvx), SC)
                            else:
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
            if cutlass.const_expr(self.dense):
                if valid != cutlass.Int32(0) and B < cutlass.Int32(NBS - 1):
                    # Every interior bin contains one exact BF16 value.
                    whole = cutlass.Int32(1)
            lim1 = above
            if whole != cutlass.Int32(0):
                if cutlass.const_expr(self.dense):
                    lim1 = above + need
                else:
                    lim1 = above + m
            degen = cutlass.Int32(0)
            if valid == cutlass.Int32(0):
                degen = cutlass.Int32(1)
            if m > cutlass.Int32(CMPB):
                if cutlass.const_expr(self.dense):
                    if whole == cutlass.Int32(0):
                        degen = cutlass.Int32(1)
                else:
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
                            if cutlass.const_expr(self.dense):
                                bq = self._dense_bin(xv, SC)
                            else:
                                bq = f2s_rz((xv - TF) * SC)
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                        else:
                            if cutlass.const_expr(self.v16):
                                wpk = cutlass.Uint32(s_cbuf[i])
                                idv = cutlass.Int32(wpk & cutlass.Uint32(0xFFFF))
                                xv = f32_of_u32(wpk & cutlass.Uint32(0xFFFF0000))
                                if cutlass.const_expr(self.dense):
                                    bq = self._dense_bin(xv, SC)
                                else:
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
                                        if cutlass.const_expr(self.vstg or self.v16):
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
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        lo2 = c0 << cutlass.Int32(2)
                        hi2 = c1 << cutlass.Int32(2)
                    else:
                        lo2 = c0 << cutlass.Int32(3)
                        hi2 = c1 << cutlass.Int32(3)
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
                                if cutlass.const_expr(self.dense):
                                    bq = self._dense_bin(x, SC)
                                else:
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
                                    if cutlass.const_expr(self.v16):
                                        wq = cutlass.Uint32(s_cbuf[i])
                                        uq = fkey_bits(wq & cutlass.Uint32(0xFFFF0000))
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
                                if cutlass.const_expr(self.v16):
                                    wq = cutlass.Uint32(s_cbuf[i])
                                    idv = cutlass.Int32(wq & cutlass.Uint32(0xFFFF))
                                    uq = fkey_bits(wq & cutlass.Uint32(0xFFFF0000))
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

                            # G2 (bf16 SPLIT only): the level histogram reads
                            # the row slab when it is complete, else the row.
                            # The row arm below is textually identical to the
                            # base module's loop (closures are not allowed
                            # inside staged regions, so it is duplicated).
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
                                            s_hist.iterator + cutlass.Int32(du),
                                            cutlass.Int32(1),
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
                            p1,
                            p2,
                            i - lead_db,
                            cutlass.Int32(0),
                            nA,
                            nA,
                            nT,
                            out_row,
                            s_scal,
                            lane,
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


def _dtype_of(name: str):
    """'f32' | 'bf16' -> cutlass numeric type (host module stays cutlass-free)."""
    assert name in ("f32", "bf16"), name
    return cutlass.Float32 if name == "f32" else cutlass.BFloat16


def get_compiled(
    tpl: tuple,
    options_extra: str = "",
    hint_free: bool = False,
    prefill: bool = False,
    dtype: str = "f32",
    v16: bool = False,
    dense: bool = False,
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
    key = (
        tuple(tpl),
        options_extra,
        bool(hint_free),
        bool(prefill),
        str(dtype),
        bool(v16),
        bool(dense),
    )
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
            v16=bool(v16),
            dense=bool(dense),
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
            v16=bool(v16),
            dense=bool(dense),
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


NB__reg = 1024  # NBH default


STATIC_WORDS = 128  # DSL smem prelude (static-__shared__ mirror)

STATIC_BYTES = STATIC_WORDS * 4


_NEG_INF__reg = float("-inf")


_POS_INF = float("inf")


@dsl_user_op
def _bf16_rne_bits__reg(value, *, loc=None, ip=None):
    """Return the raw BF16 encoding after round-to-nearest-even conversion."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.u32.u16 $0, h; }",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


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
        pdl: bool = False,
        varlen: bool = False,
        next_n: int = 1,
        cr_shift: int = 0,
        hint_free: bool = False,
        dtype=cutlass.Float32,
        pk16: bool = False,
        binproof: bool = False,
        packed_prefetch: bool = True,
    ) -> None:
        assert blk in (256, 512, 1024) and vpt in (1, 2, 4)
        assert nbh in (256, 512, 1024, 2048)
        assert nbh % blk == 0 or blk % nbh == 0
        # constexpr logits dtype (Float32 verbatim arm / BFloat16 native arm)
        self.dtype = dtype
        assert dtype in (cutlass.Float32, cutlass.BFloat16)
        self.blk = blk
        self.vpt = vpt
        self.minb = minb
        self.kpt = kpt
        self.cur = bool(cur)
        self.deg = bool(deg)
        self.img = bool(img)
        self.nbh = nbh
        self.pdl = bool(pdl)
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
        # hint-free: bracket = min/max fold of the first k row values
        # (already in registers); the hint-gather bracket arms are forced off
        self.hint_free = bool(hint_free)
        self.use_bm = (not deg) and (not img) and kpt >= 2 and vpt == 1 and (not hint_free)
        self.use_img = img and vpt == 1 and (not hint_free)
        self.brl = (minb * blk <= 1024) or (vpt == 1)
        # Native-bf16 crossing candidates can share one word when row indices
        # fit 16 bits: fkey's significant bf16 bits occupy the high half and
        # the low half carries the index.  The fp32 arm always keeps its
        # original two-word (ck, ci) representation.
        self.packed_prefetch = bool(packed_prefetch)
        self.pk16 = bool(pk16) and dtype != cutlass.Float32
        self.binproof = bool(binproof) and dtype == cutlass.BFloat16 and self.cur and self.brl
        self._prefetch_head = bool(
            dtype != cutlass.Float32
            and self.varlen
            and self.minb < 8
            and ((self.minb >= 2) or not (self.deg and self.blk == 1024))
        )
        self._prefetch_fragments = self.vpt

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

        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()  # knob default off

        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tid & cutlass.Int32(31)

        HP = cutlass.const_expr(self._prefetch_fragments)
        # Keep the prefetch packed across the varlen prologue except on the
        # short-row specialization, where eager widening has lower latency.
        if cutlass.const_expr(self.packed_prefetch):
            hpacked = [cute.make_rmem_tensor((2,), cutlass.Uint32) for _ in range(HP)]
            if cutlass.const_expr(self._prefetch_head):
                hx_addr = logits[row, None].iterator.toint()
                env4 = n >> cutlass.Int32(2)
                if env4 > cutlass.Int32(self.blk * self.vpt):
                    env4 = cutlass.Int32(self.blk * self.vpt)
                if env4 >= cutlass.Int32(self.blk * self.vpt):  # block-uniform peel
                    for u in cutlass.range_constexpr(HP):
                        hpacked[u][0], hpacked[u][1] = _ld_g_nc_v2_b32(
                            hx_addr
                            + cutlass.Int64(tid + cutlass.Int32(u * self.blk)) * cutlass.Int64(8)
                        )
                else:  # predicated flat batch (envelope window)
                    for u in cutlass.range_constexpr(HP):
                        hvi = tid + cutlass.Int32(u * self.blk)
                        if hvi < env4:
                            hpacked[u][0], hpacked[u][1] = _ld_g_nc_v2_b32(
                                hx_addr + cutlass.Int64(hvi) * cutlass.Int64(8)
                            )
        else:
            hfrags = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(VPT)]
            if cutlass.const_expr(self._prefetch_head):
                hx_addr = logits[row, None].iterator.toint()
                hatom = g2r_atom_f32(128, invariant=True)
                env4 = n >> cutlass.Int32(2)
                if env4 > cutlass.Int32(self.blk * self.vpt):
                    env4 = cutlass.Int32(self.blk * self.vpt)
                if env4 >= cutlass.Int32(self.blk * self.vpt):  # block-uniform peel
                    for u in cutlass.range_constexpr(HP):
                        ld_g_bf16x4(hatom, hx_addr, tid + cutlass.Int32(u * self.blk), hfrags[u])
                else:  # predicated flat batch (envelope window)
                    for u in cutlass.range_constexpr(HP):
                        hvi = tid + cutlass.Int32(u * self.blk)
                        if hvi < env4:
                            ld_g_bf16x4(hatom, hx_addr, hvi, hfrags[u])

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
        q2i = cutlass.Int32(0)
        idx = cutlass.Int32(0)
        m1 = cutlass.Int32(0)
        m2 = cutlass.Int32(0)
        t1 = cutlass.Int32(0)
        t2 = cutlass.Int32(0)
        c1 = cutlass.Int32(0)
        c2 = cutlass.Int32(0)
        s1 = cutlass.Int32(0)
        s2 = cutlass.Int32(0)
        p1 = cutlass.Int32(0)
        p2 = cutlass.Int32(0)
        wm = cutlass.Int32(0)
        sdyn = cutlass.Int32(0)
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
        LOQ = cutlass.Float32(0.0)
        HIf = cutlass.Float32(0.0)
        LOf = cutlass.Float32(0.0)
        qt2 = cutlass.Float32(0.0)
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
            out_row = out[row, None]
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

            if cutlass.const_expr(self.dtype == cutlass.Float32):
                n4 = n >> cutlass.Int32(2)
                ntail = n - (n4 << cutlass.Int32(2))
                tix = (n4 << cutlass.Int32(2)) + tid  # CUDA `tidx`
            else:
                # bf16 slot-exact window: cap the vector window at the register
                # batch (BLK*VPT vectors) and hand the overflow (route-enforced
                # <= BLK elements) to the existing scalar tail. Every phase of
                # the tail machinery is value-generic (`tid < ntail` guards),
                # so the bf16 route may halve VPT whenever BLK*VPT*4 + BLK
                # covers the envelope -- halving the long-lived value
                # registers and the per-attempt classify work instead of
                # carrying a half-idle -inf register batch.
                n4 = n >> cutlass.Int32(2)
                if n4 > cutlass.Int32(self.blk * self.vpt):
                    n4 = cutlass.Int32(self.blk * self.vpt)
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
            if cutlass.const_expr(self.dtype == cutlass.Float32):
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
            else:
                if cutlass.const_expr(self._prefetch_head):
                    # Batch already in flight from the kernel-head hoist:
                    # adopt it and overwrite lanes at/beyond the per-row
                    # vector window with -inf (identical register state to
                    # the unhoisted predicated batch below).  With a PARTIAL
                    # hoist (HP < VPT) the fragments that were not hoisted are
                    # loaded here with the ordinary predicated shape.
                    if cutlass.const_expr(self.packed_prefetch):
                        for u in cutlass.range_constexpr(HP):
                            frags[u][0] = f32_of_u32(hpacked[u][0] << cutlass.Uint32(16))
                            frags[u][1] = f32_of_u32(hpacked[u][0] & cutlass.Uint32(0xFFFF0000))
                            frags[u][2] = f32_of_u32(hpacked[u][1] << cutlass.Uint32(16))
                            frags[u][3] = f32_of_u32(hpacked[u][1] & cutlass.Uint32(0xFFFF0000))
                    else:
                        frags = [hfrags[u] if u < HP else frags[u] for u in range(VPT)]
                    if cutlass.const_expr(HP < VPT):
                        for u in cutlass.range_constexpr(VPT - HP):
                            i = tid + cutlass.Int32((u + HP) * self.blk)
                            if i < n4:
                                ld_g_bf16x4(atom128, x_addr, i, frags[u + HP])
                    for u in cutlass.range_constexpr(VPT):
                        hvj = tid + cutlass.Int32(u * self.blk)
                        if hvj >= n4:  # -INFINITY fill
                            for q in cutlass.range_constexpr(4):
                                frags[u][q] = cutlass.Float32(_NEG_INF__reg)
                else:
                    if n4 >= cutlass.Int32(self.blk * self.vpt):  # block-uniform peel
                        for u in cutlass.range_constexpr(VPT):
                            ld_g_bf16x4(
                                atom128, x_addr, tid + cutlass.Int32(u * self.blk), frags[u]
                            )
                    else:  # predicated flat batch
                        for u in cutlass.range_constexpr(VPT):
                            i = tid + cutlass.Int32(u * self.blk)
                            if i < n4:
                                ld_g_bf16x4(atom128, x_addr, i, frags[u])
                        for u in cutlass.range_constexpr(VPT):
                            i = tid + cutlass.Int32(u * self.blk)
                            if i >= n4:  # -INFINITY fill
                                for q in cutlass.range_constexpr(4):
                                    frags[u][q] = cutlass.Float32(_NEG_INF__reg)

                tval = cutlass.Float32(_NEG_INF__reg)
                if tid < ntail:
                    tval = ldg_bf16(x_addr, tix)

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
            elif cutlass.const_expr(self.hint_free and not self.deg):
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
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    for t in cutlass.range_constexpr(KPT):
                        xv = cutlass.Float32(0.0)
                        if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):
                            xv = ldg_f32(x_addr, pvs[t])
                        xs.append(xv)
                else:
                    for t in cutlass.range_constexpr(KPT):
                        xv = cutlass.Float32(0.0)
                        if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):
                            xv = ldg_bf16(x_addr, pvs[t])
                        xs.append(xv)
                for t in cutlass.range_constexpr(KPT):
                    if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):
                        uk = fkey(xs[t])
                        if uk < lmin:
                            lmin = uk
                        if uk > lmax:
                            lmax = uk

            # ---- block min/max in ONE barrier; publishes hist clear
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
            OFFf = cutlass.Float32(1.0 if self.brl else 0.0)
            recip = 1.0 / float(self.nbh - (2 if self.brl else 0))
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
                    _red_shared_add1__reg(hb + (cutlass.Int32(bn) << cutlass.Int32(2)))
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    qt = _fmaf__reg(tval, SC, CQ)  # unconditional
                    bnt = _umin_u32(f2u_rz(qt), cutlass.Uint32(self.nbh - 1))
                    _red_shared_add1__reg(hb + (cutlass.Int32(bnt) << cutlass.Int32(2)))
                else:
                    qt = cutlass.Float32(0.0)
                    bnt = cutlass.Uint32(0)
                    if tid < ntail:
                        qt = _fmaf__reg(tval, SC, CQ)
                        bnt = _umin_u32(f2u_rz(qt), cutlass.Uint32(self.nbh - 1))
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

            # Interior FMA bounds and a nonempty crossing bin prove that
            # only one BF16 value is possible between adjacent encodings.
            # CUR cursors place all higher bins before the arbitrary ties.
            if cutlass.const_expr(self.binproof):
                if whole == cutlass.Int32(0) and m > cutlass.Int32(0):
                    if Bv > cutlass.Int32(0) and Bv < cutlass.Int32(self.nbh - 1):
                        if SC > cutlass.Float32(0.0):
                            bin_distance = cutlass.Float32(Bv) + cutlass.Float32(0.5) - CQ
                            # Only skip the proof when value/bin-width ratio
                            # is small; the original exact path remains valid.
                            if _fabsf(bin_distance) >= cutlass.Float32(64.0):
                                center = bin_distance * cute.arch.rcp_approx(SC)
                                center_bits = _bf16_rne_bits__reg(center)
                                magnitude = center_bits & cutlass.Uint32(0x7FFF)
                                # Exclude signed zero, subnormal boundaries, and
                                # the maximal finite encodings next to infinity.
                                if magnitude > cutlass.Uint32(
                                    0x0080
                                ) and magnitude < cutlass.Uint32(0x7F7F):
                                    previous_bits = center_bits - cutlass.Uint32(1)
                                    next_bits = center_bits + cutlass.Uint32(1)
                                    if (center_bits & cutlass.Uint32(0x8000)) != cutlass.Uint32(0):
                                        previous_bits = center_bits + cutlass.Uint32(1)
                                        next_bits = center_bits - cutlass.Uint32(1)
                                    previous = f32_of_u32(previous_bits << cutlass.Uint32(16))
                                    following = f32_of_u32(next_bits << cutlass.Uint32(16))
                                    # Interior bin boundaries are exactly representable.
                                    # These FMA bounds imply the original f2u/clamp
                                    # classifications while avoiding three conversions.
                                    previous_q = _fmaf__reg(previous, SC, CQ)
                                    next_q = _fmaf__reg(following, SC, CQ)
                                    bin_low = cutlass.Float32(Bv)
                                    bin_high = cutlass.Float32(Bv + cutlass.Int32(1))
                                    # The bin is nonempty (m > 0). Monotonicity
                                    # and these neighbor bounds restrict it to
                                    # the candidate value, so checking that value
                                    # itself would repeat an established fact.
                                    if previous_q < bin_low and next_q >= bin_high:
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
                lml = cutlass.Int32(cute.arch.lanemask_lt())
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
                        q2e = cutlass.Int32(1)
                    n1 = ballot(q1e == cutlass.Int32(1))
                    n2 = ballot(q2e == cutlass.Int32(1))
                    b1 = cutlass.Int32(0)
                    b2 = cutlass.Int32(0)
                    if lane == cutlass.Int32(0):
                        if n1 != cutlass.Int32(0):
                            b1 = atomic_add_cta(s_e12.iterator, popc(n1))
                        if n2 != cutlass.Int32(0):
                            b2 = atomic_add_cta(s_e12.iterator + 1, popc(n2))
                    b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                    b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                    p1e = b1 + popc(n1 & lml)
                    p2e = b2 + popc(n2 & lml)
                    if q1e == cutlass.Int32(1):
                        if p1e < nA:
                            out_row[p1e] = ixv
                    if q2e == cutlass.Int32(1):
                        if p2e < nT:
                            out_row[nA + p2e] = ixv
                # tail element
                u64 = cutlass.Int64(-1)
                if tid < ntail:
                    u64 = cutlass.Int64(fkey(tval))
                q1e = cutlass.Int32(0)
                q2e = cutlass.Int32(0)
                if u64 > ethr:
                    q1e = cutlass.Int32(1)
                if u64 == ethr:
                    q2e = cutlass.Int32(1)
                n1 = ballot(q1e == cutlass.Int32(1))
                n2 = ballot(q2e == cutlass.Int32(1))
                b1 = cutlass.Int32(0)
                b2 = cutlass.Int32(0)
                if lane == cutlass.Int32(0):
                    if n1 != cutlass.Int32(0):
                        b1 = atomic_add_cta(s_e12.iterator, popc(n1))
                    if n2 != cutlass.Int32(0):
                        b2 = atomic_add_cta(s_e12.iterator + 1, popc(n2))
                b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                p1e = b1 + popc(n1 & lml)
                p2e = b2 + popc(n2 & lml)
                if q1e == cutlass.Int32(1):
                    if p1e < nA:
                        out_row[p1e] = tix
                if q2e == cutlass.Int32(1):
                    if p2e < nT:
                        out_row[nA + p2e] = tix
                # (CUDA returns here — everything below is the else-arm)
            else:
                # ---- emit
                if cutlass.const_expr(self.cur):
                    LOQ = cutlass.Float32(Bv)  # int->float cvt
                    lim1 = above
                    if whole == cutlass.Int32(1):
                        lim1 = above + m
                        if cutlass.const_expr(self.binproof):
                            if lim1 > k:
                                lim1 = k
                    for s in cutlass.range_constexpr(S):
                        if cutlass.const_expr(self.brl):
                            q = _fmaf__reg(_val(frags, s), SC, CQ)  # bit-identical to classify
                        else:
                            q = _fmaf__reg(_val(frags, s) - Tv, SC, OFFf)  # emit spelling
                        idx = (
                            (tid + cutlass.Int32((s // 4) * self.blk)) << cutlass.Int32(2)
                        ) + cutlass.Int32(s % 4)
                        p = cutlass.Int32(0)
                        if q >= LOQ:
                            bn = _umin_u32(f2u_rz(q), cutlass.Uint32(self.nbh - 1))
                            p = atomic_add_cta(
                                s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1)
                            )
                            if p < lim1:
                                out_row[p] = idx
                            else:
                                if whole == cutlass.Int32(0):
                                    q2i = p - above
                                    if q2i < cmp_:  # escape-made-safe guard
                                        if cutlass.const_expr(self.pk16):
                                            ck[q2i] = (
                                                fkey(_val(frags, s)) & cutlass.Uint32(0xFFFF0000)
                                            ) | cutlass.Uint32(idx)
                                        else:
                                            ck[q2i] = fkey(_val(frags, s))
                                            ci[q2i] = idx
                    # tail
                    if cutlass.const_expr(self.brl):
                        qt2 = _fmaf__reg(tval, SC, CQ)
                    else:
                        qt2 = _fmaf__reg(tval - Tv, SC, OFFf)
                    p = cutlass.Int32(0)
                    if qt2 >= LOQ:
                        bn = _umin_u32(f2u_rz(qt2), cutlass.Uint32(self.nbh - 1))
                        p = atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                        if p < lim1:
                            out_row[p] = tix
                        else:
                            if whole == cutlass.Int32(0):
                                q2i = p - above
                                if q2i < cmp_:
                                    if cutlass.const_expr(self.pk16):
                                        ck[q2i] = (
                                            fkey(tval) & cutlass.Uint32(0xFFFF0000)
                                        ) | cutlass.Uint32(tix)
                                    else:
                                        ck[q2i] = fkey(tval)
                                        ci[q2i] = tix
                else:
                    # two-mask ballot emit
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
                    c1 = popc(m1) + t1
                    c2 = popc(m2) + t2
                    s1, s2 = warp_incl_scan_add2(c1, c2, lane)
                    b1 = cutlass.Int32(0)
                    b2 = cutlass.Int32(0)
                    if lane == cutlass.Int32(31):
                        b1 = atomic_add_cta(s_cnt.iterator, s1)
                        b2 = atomic_add_cta(s_cnt.iterator + 1, s2)
                    b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(31))
                    b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(31))
                    p1 = b1 + (s1 - c1)
                    p2 = b2 + (s2 - c2)
                    lim1 = above
                    if whole == cutlass.Int32(1):
                        lim1 = k
                    wm = m1  # sparse set-bit walk
                    while wm != cutlass.Int32(0):
                        sdyn = ffs_m1(wm)
                        idx = (
                            (tid + (sdyn >> cutlass.Int32(2)) * cutlass.Int32(self.blk))
                            << cutlass.Int32(2)
                        ) + (sdyn & cutlass.Int32(3))
                        if p1 < lim1:
                            out_row[p1] = idx
                        p1 = p1 + cutlass.Int32(1)
                        wm = wm & (wm - cutlass.Int32(1))
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
                                    if cutlass.const_expr(self.pk16):
                                        ck[p2] = (
                                            fkey(_val(frags, s)) & cutlass.Uint32(0xFFFF0000)
                                        ) | cutlass.Uint32(idx)
                                    else:
                                        ck[p2] = fkey(_val(frags, s))
                                        ci[p2] = idx
                                p2 = p2 + cutlass.Int32(1)
                    if t2 == cutlass.Int32(1):
                        if p2 < cmp_:
                            if cutlass.const_expr(self.pk16):
                                ck[p2] = (fkey(tval) & cutlass.Uint32(0xFFFF0000)) | cutlass.Uint32(
                                    tix
                                )
                            else:
                                ck[p2] = fkey(tval)
                                ci[p2] = tix
                        p2 = p2 + cutlass.Int32(1)

                # ---- refine (skipped when whole — CUDA returned inside emit)
                if whole == cutlass.Int32(0):
                    cute.arch.barrier()  # emit done
                    if cutlass.const_expr(self.cur):
                        mc = m
                        if mc > cmp_:
                            mc = cmp_
                    else:
                        mc = s_cnt[1]
                        if mc > cmp_:
                            mc = cmp_
                    quad = cutlass.Int32(0)
                    if mc >= m:
                        if mc <= qc:
                            quad = cutlass.Int32(1)
                    if quad == cutlass.Int32(1):
                        # O(mc^2) index-tie-broken rank
                        i = tid
                        while i < mc:
                            uq = cutlass.Uint32(ck[i])
                            r = cutlass.Int32(0)
                            j = cutlass.Int32(0)
                            while j < mc:
                                vq = cutlass.Uint32(ck[j])
                                tinc = cutlass.Int32(0)
                                if cutlass.const_expr(self.pk16):
                                    # key_hi16|idx16 is unique within a row
                                    # (the idx half never repeats), so the
                                    # equality/index tie-break can never fire.
                                    if vq > uq:
                                        tinc = cutlass.Int32(1)
                                else:
                                    if vq > uq:
                                        tinc = cutlass.Int32(1)
                                    if vq == uq:
                                        if j < i:
                                            tinc = cutlass.Int32(1)
                                r = r + tinc
                                j = j + cutlass.Int32(1)
                            if r < need:
                                if cutlass.const_expr(self.pk16):
                                    out_row[above + r] = cutlass.Int32(
                                        cutlass.Uint32(ck[i]) & cutlass.Uint32(0xFFFF)
                                    )
                                else:
                                    out_row[above + r] = ci[i]
                            i = i + cutlass.Int32(BLK)
                    else:
                        # ---- fallback: exact key-space narrowing
                        if tid == cutlass.Int32(0):
                            s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                            s_kmm[1] = cutlass.Uint32(0)
                        cute.arch.barrier()  # kmm init
                        i = tid
                        while i < mc:
                            if cutlass.const_expr(self.pk16):
                                kv = cutlass.Uint32(ck[i]) & cutlass.Uint32(0xFFFF0000)
                            else:
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
                                    if cutlass.const_expr(self.pk16):
                                        unar = cutlass.Uint32(ck[i]) & cutlass.Uint32(0xFFFF0000)
                                    else:
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
                                        s_hist, s_ws, needC, tid, s_res, blk=self.blk, nb=self.nbh
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
                        # final two-predicate ballot emit
                        if tid == cutlass.Int32(0):
                            s_e12[0] = cutlass.Int32(0)
                            s_e12[1] = cutlass.Int32(0)
                        cute.arch.barrier()  # emit counters
                        lml = cutlass.Int32(cute.arch.lanemask_lt())
                        it2 = (mc + cutlass.Int32(self.blk - 1)) // cutlass.Int32(self.blk)
                        it = cutlass.Int32(0)
                        while it < it2:
                            i = it * cutlass.Int32(BLK) + tid
                            uke = cutlass.Uint32(0)
                            idv = cutlass.Int32(0)
                            if i < mc:
                                if cutlass.const_expr(self.pk16):
                                    w = cutlass.Uint32(ck[i])
                                    uke = w & cutlass.Uint32(0xFFFF0000)
                                    idv = cutlass.Int32(w & cutlass.Uint32(0xFFFF))
                                else:
                                    uke = cutlass.Uint32(ck[i])
                                    idv = ci[i]
                            q1f = cutlass.Int32(0)
                            q2f = cutlass.Int32(0)
                            if i < mc:
                                if cutlass.Int64(uke) > ethr:
                                    q1f = cutlass.Int32(1)
                                if cutlass.Int64(uke) == ethr:
                                    q2f = cutlass.Int32(1)
                            n1 = ballot(q1f == cutlass.Int32(1))
                            n2 = ballot(q2f == cutlass.Int32(1))
                            b1 = cutlass.Int32(0)
                            b2 = cutlass.Int32(0)
                            if lane == cutlass.Int32(0):
                                if n1 != cutlass.Int32(0):
                                    b1 = atomic_add_cta(s_e12.iterator, popc(n1))
                                if n2 != cutlass.Int32(0):
                                    b2 = atomic_add_cta(s_e12.iterator + 1, popc(n2))
                            b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                            b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                            p1e = b1 + popc(n1 & lml)
                            p2e = b2 + popc(n2 & lml)
                            if q1f == cutlass.Int32(1):
                                if p1e < aboveC:
                                    out_row[above + p1e] = idv
                            if q2f == cutlass.Int32(1):
                                if p2e < needC:
                                    out_row[above + aboveC + p2e] = idv
                            it = it + cutlass.Int32(1)

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
            use_pdl=self.pdl,
        )


_COMPILE_CACHE__reg: dict = {}


def get_compiled__reg(
    tpl: tuple,
    dump_dir: str | None = None,
    pdl: bool = False,
    varlen: bool = False,
    next_n: int = 1,
    cr_shift: int = 0,
    hint_free: bool = False,
    dtype: str = "f32",
    pk16: bool = False,
    binproof: bool = False,
    packed_prefetch: bool = True,
) -> Any:
    """Compile (or fetch) the variant for constexpr tuple
    (BLK, VPT, MINB, KPT, CUR, DEG, IMG, NBH).

    ``packed_prefetch=False`` uses eager BF16 widening for short rows. Other
    rows keep prefetched pairs packed until the valid row length is known.
    """
    key = (
        tuple(tpl),
        bool(pdl),
        bool(varlen),
        int(next_n),
        int(cr_shift),
        bool(hint_free),
        str(dtype),
        bool(pk16),
        bool(binproof),
        bool(packed_prefetch),
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
            pdl=pdl,
            varlen=varlen,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=hint_free,
            dtype=_dtype_of(dtype),
            pk16=pk16,
            binproof=binproof,
            packed_prefetch=packed_prefetch,
        )
        nb_, nc_ = cute.sym_int(), cute.sym_int()
        nb2_, nc2_ = cute.sym_int(), cute.sym_int()
        nb3_, nc3_ = cute.sym_int(), cute.sym_int()
        if dtype == "f32":
            lg_fake = _crt.make_fake_compact_tensor(
                cutlass.Float32, (nb_, nc_), stride_order=(1, 0), assumed_align=16
            )
        else:
            lg_fake = _crt.make_fake_compact_tensor(
                cutlass.BFloat16, (nb_, nc_), stride_order=(1, 0), assumed_align=16
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
        opts = "--enable-tvm-ffi"
        if dump_dir:
            opts += f" --keep-ptx --keep-cubin --dump-dir {dump_dir}"
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
                options=opts,
            )
        _COMPILE_CACHE__reg[key] = compiled
    return compiled


QUADC_CLUS__clus = QUADC_CLUS


_NEG_INF__clus = float("-inf")


class GvrClusKernel:
    """gvr_clus<BLK, U, MINB, NBS, CS> — clustered streaming GVR."""

    def __init__(
        self,
        blk: int,
        u: int,
        minb: int,
        nbs: int,
        cs: int,
        scap: int = 8192,
        cmp_: int = 2048,
        varlen: bool = False,
        next_n: int = 1,
        cr_shift: int = 0,
        hint_free: bool = False,
        dtype=cutlass.Float32,
    ) -> None:
        assert blk == 1024, "gvr_clus is always BLK=1024"
        assert minb == 1, "gvr_clus is __launch_bounds__(BLK, 1)"
        # constexpr logits dtype: bf16 arm is hint-free only (gather_hint
        # keeps its fp32-pinned scalar gathers).
        self.dtype = dtype
        assert dtype in (cutlass.Float32, cutlass.BFloat16)
        if dtype is not cutlass.Float32:
            assert bool(hint_free), "bf16 arm is hint-free"
        assert nbs == 256, "SNB must stay 256"
        assert u in (1, 2, 4, 8) and cs in (2, 4, 8)
        # per-row varlen mode (production heuristicTopKDecode contract, same
        # semantics as GvrMainKernel / GvrRegClusKernel): n and the sampling-
        # ladder scalars are re-derived PER ROW in-kernel from a device
        # kv_lens tensor; the scalar launch args become the envelope clamp
        # bound (n) and dead slots (SMP/TGT/Q/SS2/TGT2).
        self.varlen = bool(varlen)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        if self.varlen:
            assert self.next_n >= 1 and self.cr_shift in (0, 2)
        self.hint_free = bool(hint_free)  # hint-free: gather_hint sites compiled out
        self.lcs = cs.bit_length() - 1  # log2(CS) for the per-row Q shift
        self.blk = blk
        self.u = u
        self.minb = minb
        self.nbs = nbs
        self.cs = cs
        self.scap = scap  # smem extents only —
        self.cmp = cmp_  # value logic uses rt args
        self.hb = nbs
        self.stepc = blk * u
        if dtype is cutlass.Float32:
            self.pfd = u if u < 4 else 4  # PFD=min(U,4)
        else:
            self.pfd = u if u < 4 else 4  # bf16: packed u32 slots, full depth fits
        self.lb = nbs.bit_length() - 1  # log2(NBS)=8
        # dynamic-region byte map: hist | cbuf(int2) | ck64c | mrg
        self.cbuf_bytes = (scap + 4) * 8
        assert self.cbuf_bytes % 16 == 0
        self.ck_off = self.cbuf_bytes  # inside the blob
        self.dyn_bytes = nbs * 4 + self.cbuf_bytes + cmp_ * 8 + nbs * 4
        # == host smc = SNB*8 + (SCAP+4)*8 + CMP*8

    # ------------------------------------------------------------------
    # GVR_EMITK: classify+stage one survivor.
    # bn via UNSIGNED saturating convert (f2u_rz); staging store is ONE
    # u64; branchless trash slot min(pos, SCAP) (runtime SCAP). Returns pos+1.
    # ------------------------------------------------------------------
    @cute.jit
    def _emitk(self, xv, idx, pos, TF, SC, SCAP, s_hist, s_cbuf2):
        NBS = self.nbs
        bn_u = f2u_rz((xv - TF) * SC)
        if bn_u > cutlass.Uint32(NBS - 1):
            bn_u = cutlass.Uint32(NBS - 1)
        bn = cutlass.Int32(bn_u)
        atomic_add_cta(s_hist.iterator + bn, cutlass.Int32(1))
        ps = pos
        if ps > SCAP:
            ps = SCAP  # trash slot (IMNMX)
        s_cbuf2[ps] = (cutlass.Uint64(cutlass.Uint32(idx)) << cutlass.Uint64(32)) | cutlass.Uint64(
            u32_of_f32(xv)
        )
        return pos + cutlass.Int32(1)

    # ------------------------------------------------------------------
    # P5 emit step: bn via SIGNED rz convert (__float2int_rz); bn>=B gate;
    # LOCAL mrg atomicAdd whose result is a CLUSTER-GLOBAL position
    # (prefix-biased cursors from merge_scan0); overflow -> ONE packed u64
    # DSMEM store to rank-0 ck64c (never split into 4B stores).
    # ------------------------------------------------------------------
    @cute.jit
    def _p5_emit(self, xv, idv, TF, SC, B, above, lim1, whole, CMP, s_mrg, out_row, rk64):
        NBS = self.nbs
        bn = f2s_rz((xv - TF) * SC)
        if bn > cutlass.Int32(NBS - 1):
            bn = cutlass.Int32(NBS - 1)
        if bn >= B:
            p = atomic_add_cta(s_mrg.iterator + bn, cutlass.Int32(1))
            if p < lim1:
                out_row[p] = idv
            else:
                if whole == cutlass.Int32(0):
                    q2 = p - above
                    if q2 < CMP:
                        _st_shared_cluster_u64(
                            rk64 + q2 * cutlass.Int32(8),
                            (cutlass.Uint64(fkey(xv)) << cutlass.Uint64(32))
                            | cutlass.Uint64(cutlass.Uint32(idv)),
                        )

    # ------------------------------------------------------------------
    # LOCAL patched copy of merge_scan0: rematerializes mapa per (q, r)
    # exactly like the CUDA instead of holding CS mapped base addresses
    # across the whole merge. The hoisted-array form costs CS extra
    # long-lived registers; with the U>=4 sixteen-register pf prime batch
    # it tips ptxas into spilling the batch across the rung phase.
    # Semantics, the DSMEM v4 load spelling, the register accumulation and
    # the prefix-biased STS.128 cursor write are IDENTICAL to merge_scan0.
    # NO barrier inside (caller pays the merge-publish barrier).
    # ------------------------------------------------------------------
    @cute.jit
    def _merge_scan0_local(self, s_hist, s_mrg, rank, target, tidx, s_res):
        NBS = self.nbs
        CS = self.cs
        BPT = NBS // 32
        NV = BPT // 4
        if tidx < cutlass.Int32(32):
            lane = tidx
            atom = smem_atom_i32_128()
            hbase = s_hist.iterator.toint()
            # pass 1: remote v4 accumulation of tot/pre per vector
            tot_r = []
            pre_r = []
            sm = cutlass.Int32(0)
            for q in cutlass.range_constexpr(NV):
                boff = (lane * cutlass.Int32(BPT) + cutlass.Int32(4 * q)) * cutlass.Int32(4)
                t = [cutlass.Int32(0)] * 4
                p = [cutlass.Int32(0)] * 4
                for r in cutlass.range_constexpr(CS):
                    mapped = _mapa_shared_cluster_addr(
                        hbase + boff, cutlass.Int32(r)
                    )  # per-use mapa
                    v0, v1, v2, v3 = _ld_shared_cluster_v4_u32(mapped)
                    t[0] = t[0] + v0
                    t[1] = t[1] + v1
                    t[2] = t[2] + v2
                    t[3] = t[3] + v3
                    if cutlass.Int32(r) < rank:  # predicated adds
                        p[0] = p[0] + v0
                        p[1] = p[1] + v1
                        p[2] = p[2] + v2
                        p[3] = p[3] + v3
                tot_r.append(t)
                pre_r.append(p)
                sm = sm + t[0] + t[1] + t[2] + t[3]
            # inclusive scan + totals
            w = warp_incl_scan_add(sm, lane)
            tt = cute.arch.shuffle_sync(w, cutlass.Int32(31))
            after = tt - w
            if lane == cutlass.Int32(0):
                s_res[RES_TOT] = tt
            base = lane * cutlass.Int32(BPT)
            # descending walk: crossing pin + prefix-biased cursors into mrg
            for q in cutlass.range_constexpr(NV - 1, -1, -1):
                o4 = cute.make_rmem_tensor((4,), cutlass.Int32)
                for j in cutlass.range_constexpr(3, -1, -1):
                    cq = tot_r[q][j]
                    o4[j] = after + pre_r[q][j]
                    gb = base + cutlass.Int32(4 * q + j)
                    cross = cutlass.Int32(0)
                    if after < target:
                        if (after + cq) >= target:
                            cross = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross = cutlass.Int32(1)
                    if cross != cutlass.Int32(0):
                        s_res[RES_B] = gb
                        s_res[RES_ABOVE] = after
                        s_res[RES_M] = cq
                    after = after + cq
                boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
                sts128_i32(atom, o4, s_mrg.iterator.toint(), boff)

    # ------------------------------------------------------------------
    # two-predicate warp-ballot emit step (narrowing and degen emits) —
    # same helper as the main family. s_scal[1]=s_o1, s_scal[2]=s_o2.
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
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        SCAP: cutlass.Int32,
        CMP: cutlass.Int32,
        SMP: cutlass.Int32,
        TGT: cutlass.Int32,
        Q: cutlass.Int32,
        SS2: cutlass.Int32,
        TGT2: cutlass.Int32,
        bigf: cutlass.Int32,
    ):
        BLK = self.blk
        U = self.u
        NBS = self.nbs
        CS = self.cs
        PFD = self.pfd
        STEPC = self.stepc
        NW = BLK // 32

        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()  # (rank, row)
        rank = bx
        row = by
        lane = tidx & cutlass.Int32(31)

        # ============ per-row varlen prologue — shared contract lives in ======
        # GvrMainKernel's prologue (per-row n from kv_lens; ladder scalars are
        # dead launch args, re-derived here by the route_dynamic() clus
        # formulas). Clus deviation: QUAD sample geometry runs for every
        # non-short row (host never launches clus at n <= SCAP, so SMP == 0 is
        # untested; sampling only steers the rung — exactness is
        # schedule-invariant). All values are pure functions of `row`, so the
        # whole-body guard and the cluster barriers stay cluster-uniform.
        # Short rows (n <= k): rank 0 emits identity + (-1); body is SKIPped.
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
                # ---- aim ladder (cheap mirror, all-thread) ----
                # Schedule quantities only (exactness is schedule-invariant,
                # same argument as the always-sample deviation above): divides
                # become MUFU.RCP multiplies and the isqrt fixup loops collapse
                # to single steps (f32 sqrt of an exactly-representable int
                # (6n <= 2^23) is within 1 of isqrt).  All-thread on purpose:
                # this family's mirror redundancy is small, and a
                # warp0+barrier hoist EXPOSES the chain's serial latency at
                # ~1 CTA/SM, while the redundant form hides it across warps.
                # Q (chunk ownership) keeps its exact shift form.
                x6 = cutlass.Int32(6) * nv
                ri = cutlass.Int32(cmath.sqrt(cutlass.Float32(x6)))
                if ri * ri > x6:
                    ri = ri - cutlass.Int32(1)
                if (ri + cutlass.Int32(1)) * (ri + cutlass.Int32(1)) <= x6:
                    ri = ri + cutlass.Int32(1)
                r6 = ri
                if x6 - ri * ri > ri:
                    r6 = ri + cutlass.Int32(1)
                # aim_base: R = CS > 1 always for this family; bigf is the
                # launch-computed occupancy flag (num_rows * CS <= 148).
                aim = k << cutlass.Int32(1)
                if bigf == cutlass.Int32(0):
                    if k >= cutlass.Int32(1024):
                        aim = (cutlass.Int32(11) * k) >> cutlass.Int32(3)
                    else:
                        aim = (cutlass.Int32(3) * k) >> cutlass.Int32(1)
                if r6 > aim:
                    aim = r6
                amin = cutlass.Int32(3) * k
                if cutlass.const_expr(self.cs != 2):
                    amin = (cutlass.Int32(7) * k) >> cutlass.Int32(1)
                if aim < amin:
                    aim = amin
                if aim > (SCAP >> cutlass.Int32(1)):
                    aim = SCAP >> cutlass.Int32(1)
                if aim < k:
                    aim = k
                # ---- QUAD sample geometry (route_dynamic clus override,
                # always-on per the deviation note above; k <= 1024 for this
                # family so sfac has no k > 1024 arm).
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    n4v = nv >> cutlass.Int32(2)
                else:
                    n4v = nv >> cutlass.Int32(3)  # bf16: 8-elem vectors
                sel = cutlass.Int32(
                    cutlass.Float32(nv)
                    * cutlass.Float32(32.0 if self.cs == 2 else 16.0)
                    * cute.arch.rcp_approx(cutlass.Float32(aim))
                )
                if sel < cutlass.Int32(256):
                    sel = cutlass.Int32(256)
                nh = nv >> cutlass.Int32(1)
                if sel > nh:
                    sel = nh
                quads = sel >> cutlass.Int32(4)
                if quads < cutlass.Int32(1):
                    quads = cutlass.Int32(1)
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    quarter = n4v >> cutlass.Int32(2)
                else:
                    quarter = n4v >> cutlass.Int32(1)  # same 16-elem sample units
                if quarter < cutlass.Int32(1):
                    quarter = cutlass.Int32(1)
                if quads > quarter:
                    quads = quarter
                SS2 = cutlass.Int32(
                    cutlass.Float32(quarter) * cute.arch.rcp_approx(cutlass.Float32(quads))
                )
                if SS2 < cutlass.Int32(1):
                    SS2 = cutlass.Int32(1)
                SMP = cutlass.Int32(
                    cutlass.Float32(quarter) * cute.arch.rcp_approx(cutlass.Float32(SS2))
                )
                # sample-window guard: P1 indexes up to ~SMP*SS2*4 lines; keep
                # SMP*SS2 <= quarter (approx error <= +1, one step closes it)
                if SMP * SS2 > quarter:
                    SMP = SMP - cutlass.Int32(1)
                if SMP < cutlass.Int32(1):
                    SMP = cutlass.Int32(1)
                rn_ = cute.arch.rcp_approx(cutlass.Float32(nv))
                smp16f = cutlass.Float32(SMP) * cutlass.Float32(16.0)
                TGT = cutlass.Int32(cutlass.Float32(aim) * smp16f * rn_)
                if TGT < cutlass.Int32(1):
                    TGT = cutlass.Int32(1)
                TGT2 = cutlass.Int32(cutlass.Float32(k) * smp16f * rn_)
                if TGT2 < cutlass.Int32(1):
                    TGT2 = cutlass.Int32(1)
                # NB: the Q launch slot is dead in this family (no in-kernel
                # consumer); chunk ownership derives from n4/STEPC below.
            if short != cutlass.Int32(0):
                if rank == cutlass.Int32(0):
                    if tidx < kq:
                        ov = cutlass.Int32(-1)
                        if tidx < nv:
                            ov = tidx
                        out[row, tidx] = ov

        # ---- shared memory (CUDA dynamic map order, then static allocs) ----
        smem = SmemAllocator()
        s_hist = smem.allocate_tensor(  # hist[NBS] @ blob start
            cutlass.Int32, cute.make_ordered_layout((self.hb,), order=(0,)), byte_alignment=128
        )
        blob = smem.allocate_tensor(  # cbuf(int2) | ck64c
            cutlass.Int8,
            cute.make_ordered_layout((self.cbuf_bytes + self.cmp * 8,), order=(0,)),
            byte_alignment=16,
        )
        s_mrg = smem.allocate_tensor(  # mrg[NBS]
            cutlass.Int32, cute.make_ordered_layout((self.nbs,), order=(0,)), byte_alignment=16
        )
        s_ws = smem.allocate_tensor(  # degen scan only
            cutlass.Int32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        # scan_cross predeclares its second-stage partial as Int32 and reads
        # s_ws inside a dynamic if — a Uint32 ws tensor trips the DSL type-
        # stability check (counts < 2^31 so Int32 is exact).
        s_wmn = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        s_wmx = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        s_res = smem.allocate_tensor(  # shared slot map
            cutlass.Int32, cute.make_ordered_layout((8,), order=(0,)), byte_alignment=16
        )
        # scalar block: [0]=s_bufn [1]=s_o1 [2]=s_o2
        s_scal = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
        )
        s_tsh = smem.allocate_tensor(
            cutlass.Float32, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=4
        )
        s_kmm = smem.allocate_tensor(  # [0]=kmin [1]=kmax
            cutlass.Uint32, cute.make_ordered_layout((2,), order=(0,)), byte_alignment=8
        )
        sbase = blob.iterator.toint()
        s_cbuf2 = cute.make_tensor(  # int2 staged as u64
            cute.make_ptr(cutlass.Uint64, sbase, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.scap + 4,)),
        )
        ck_addr = sbase + cutlass.Int32(self.ck_off)
        s_ck64 = cute.make_tensor(
            cute.make_ptr(cutlass.Uint64, ck_addr, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.cmp,)),
        )

        # ---- whole-body short-row guard (cluster-uniform: `short` is a pure
        # function of `row`, identical across all CS ranks and threads, so
        # every cluster barrier below stays aligned; short rows already
        # emitted identity + -1 tail in the prologue) ----
        if short == cutlass.Int32(0):
            # ---- row bases (pre_idx is request-level under varlen) ----
            row64 = cutlass.Int64(row)
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                x_addr = logits.iterator.toint() + row64 * cutlass.Int64(npad) * cutlass.Int64(4)
            else:
                x_addr = logits.iterator.toint() + row64 * cutlass.Int64(npad) * cutlass.Int64(2)
            p_addr = pre_idx.iterator.toint() + cutlass.Int64(prow) * cutlass.Int64(
                k
            ) * cutlass.Int64(4)
            out_row = out[row, None]

            # ---- interleaved chunk ownership ----
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                n4 = n >> cutlass.Int32(2)
            else:
                n4 = n >> cutlass.Int32(3)  # bf16: 8-elem 16B vectors
            nCh = (n4 + cutlass.Int32(STEPC - 1)) // cutlass.Int32(STEPC)
            nFullG = n4 // cutlass.Int32(STEPC)
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                tail0 = n4 << cutlass.Int32(2)
            else:
                tail0 = n4 << cutlass.Int32(3)
            tailn = cutlass.Int32(0)
            if rank == cutlass.Int32(0):
                tailn = n - tail0

            if tidx == cutlass.Int32(0):
                s_res[RES_B2] = cutlass.Int32(-1)
                s_res[RES_B3] = cutlass.Int32(-1)
                s_scal[0] = cutlass.Int32(0)  # s_bufn
            if tidx < cutlass.Int32(self.hb):  # HB<=BLK
                s_hist[tidx] = cutlass.Int32(0)

            # ============ P1: QUAD sample (hint gather LAZY) =====================
            # one 64B line = 4 float4 per location, TWO threads: tid takes the
            # lower pair at p4, tid+SMP the upper pair at p4+2.
            atom128 = g2r_atom_f32(128, invariant=True)
            fsa = cute.make_rmem_tensor((4,), cutlass.Float32)
            fsb = cute.make_rmem_tensor((4,), cutlass.Float32)
            smp2 = SMP * cutlass.Int32(2)
            shas = cutlass.Int32(0)
            if tidx < smp2:
                shas = cutlass.Int32(1)
            if shas != cutlass.Int32(0):
                p4 = tidx * SS2 * cutlass.Int32(4)
                if tidx >= SMP:
                    p4 = (tidx - SMP) * SS2 * cutlass.Int32(4) + cutlass.Int32(2)
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    ld_g_f32x4(atom128, x_addr, p4, fsa)
                    ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fsb)
                else:
                    # ONE bf16x8 vector = the same 8 elements of the fp32 pair:
                    # lower half of the 16-elem unit for tid, upper for tid+SMP
                    p8 = tidx * SS2 * cutlass.Int32(2)
                    if tidx >= SMP:
                        p8 = (tidx - SMP) * SS2 * cutlass.Int32(2) + cutlass.Int32(1)
                    ld_g_bf16x8_pair(atom128, x_addr, p8, fsa, fsb)

            # ============ P2: quantile rung, redundant per CTA ===================
            smn = cutlass.Float32(float("inf"))
            smx = cutlass.Float32(float("-inf"))
            if shas != cutlass.Int32(0):
                for t in cutlass.range_constexpr(4):
                    smn = fmin_f32(smn, fsa[t])
                    smx = fmax_f32(smx, fsa[t])
                for t in cutlass.range_constexpr(4):
                    smn = fmin_f32(smn, fsb[t])
                    smx = fmax_f32(smx, fsb[t])
            fma_ = cute.make_rmem_tensor((4,), cutlass.Float32)  # mop-up pair bufs
            fmb_ = cute.make_rmem_tensor((4,), cutlass.Float32)
            j = tidx + cutlass.Int32(BLK)  # mop-up
            while j < smp2:
                p4 = j * SS2 * cutlass.Int32(4)
                if j >= SMP:
                    p4 = (j - SMP) * SS2 * cutlass.Int32(4) + cutlass.Int32(2)
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    ld_g_f32x4(atom128, x_addr, p4, fma_)
                    ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
                else:
                    p8 = j * SS2 * cutlass.Int32(2)
                    if j >= SMP:
                        p8 = (j - SMP) * SS2 * cutlass.Int32(2) + cutlass.Int32(1)
                    ld_g_bf16x8_pair(atom128, x_addr, p8, fma_, fmb_)
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

            # PRIME-LATE: every rank's sample has landed; prime NOW.
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                lim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)
                pf = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(PFD)]
            else:
                lim4 = (npad >> cutlass.Int32(3)) - cutlass.Int32(1)
                # PACKED u32 pairs (4 regs per bf16x8 vector; unpacked at classify)
                pf = [cute.make_rmem_tensor((4,), cutlass.Uint32) for _ in range(PFD)]
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                for uu in cutlass.range_constexpr(PFD):  # clamped prime
                    i_ = rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
                    ic = i_
                    if ic >= n4:
                        ic = lim4
                    ld_g_f32x4(atom128, x_addr, ic, pf[uu])
            else:
                for uu in cutlass.range_constexpr(PFD):  # clamped prime
                    i_ = rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
                    ic = i_
                    if ic >= n4:
                        ic = lim4
                    ld_g_bf16x8_pk(atom128, x_addr, ic, pf[uu])
            # asm prefetch gate: DEEP rows only; empty for U<=PFD
            if cutlass.const_expr(U > PFD):
                gpp = cutlass.Int32(0)
                if n4 >= cutlass.Int32(32768):
                    if (rank + cutlass.Int32(1)) * cutlass.Int32(STEPC) <= n4:
                        gpp = cutlass.Int32(1)
                if gpp != cutlass.Int32(0):
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        for uu in cutlass.range_constexpr(PFD, U):
                            _prefetch_l2(
                                x_addr
                                + cutlass.Int64(
                                    rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
                                )
                                * cutlass.Int64(16)
                            )
                    else:
                        for uu in cutlass.range_constexpr(PFD, U):
                            _prefetch_l2(
                                x_addr
                                + cutlass.Int64(
                                    rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
                                )
                                * cutlass.Int64(16)
                            )

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
            T = cutlass.Float32(_NEG_INF__clus)
            HIC = cutlass.Float32(_NEG_INF__clus)
            w = cutlass.Float32(0.0)
            sok = cutlass.Int32(0)
            if SMP > cutlass.Int32(0):
                if SMAX > SMIN:
                    sok = cutlass.Int32(1)
            if sok != cutlass.Int32(0):  # sample hist
                w = (SMAX - SMIN) * cutlass.Float32(1.0 / 256.0)
                # CUDA --use_fast_math lowers `1.0f / w` to a bare MUFU.RCP;
                # a plain `/` would emit the IEEE div.rn rcp+Newton+CALL
                # chain. w > 0 by the sok guard; bucketing is SC-invariant.
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
                j = tidx + cutlass.Int32(BLK)  # mop-up reloads
                while j < smp2:
                    p4 = j * SS2 * cutlass.Int32(4)
                    if j >= SMP:
                        p4 = (j - SMP) * SS2 * cutlass.Int32(4) + cutlass.Int32(2)
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        ld_g_f32x4(atom128, x_addr, p4, fma_)
                        ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
                    else:
                        p8 = j * SS2 * cutlass.Int32(2)
                        if j >= SMP:
                            p8 = (j - SMP) * SS2 * cutlass.Int32(2) + cutlass.Int32(1)
                        ld_g_bf16x8_pair(atom128, x_addr, p8, fma_, fmb_)
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
                three=True,
            )
            cute.arch.barrier()  # ---- barrier (scan publish) ----

            tot0 = s_res[RES_TOT]
            b1v = s_res[RES_B]
            if sok != cutlass.Int32(0):
                if tot0 >= TGT:
                    T = _fmaf__clus(cutlass.Float32(b1v), w, SMIN)
            needg = cutlass.Int32(1)
            if T > cutlass.Float32(_NEG_INF__clus):
                needg = cutlass.Int32(0)
            if needg != cutlass.Int32(0):
                # degenerate sample: identical on every rank of the cluster
                if cutlass.const_expr(not self.hint_free):
                    GMIN, GMAX = gather_hint(
                        x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk=BLK, kpt=1
                    )  # 2 barriers
                T = GMIN
            if sok != cutlass.Int32(0):  # HIC tighten
                if tot0 >= TGT:
                    b2v = s_res[RES_B2]
                    if b2v >= cutlass.Int32(0):
                        Tk = _fmaf__clus(cutlass.Float32(b2v), w, SMIN)
                        up = fmax_f32(Tk - T, cutlass.Float32(0.0))
                        # heavy-tail cap by T - T3 (rank-TGT..rank-2TGT distance)
                        if tot0 >= TGT * cutlass.Int32(2):
                            b3v = s_res[RES_B3]
                            if b3v >= cutlass.Int32(0):
                                T3 = _fmaf__clus(cutlass.Float32(b3v), w, SMIN)
                                if T > T3:
                                    up = fmin_f32(up, cutlass.Float32(2.0) * (T - T3))
                        HIC = fmax_f32(
                            _fmaf__clus(cutlass.Float32(4.0), up, T),
                            _fmaf__clus(cutlass.Float32(8.0), w, T),
                        )
            # ladder floor kept in SHARED (64-register wall)
            if tidx == cutlass.Int32(0):
                t5 = cutlass.Float32(_NEG_INF__clus)
                if sok != cutlass.Int32(0):
                    if tot0 >= TGT * cutlass.Int32(2):
                        b3v = s_res[RES_B3]
                        if b3v >= cutlass.Int32(0):
                            if T > GMIN:
                                T3 = _fmaf__clus(cutlass.Float32(b3v), w, SMIN)
                                if T3 < T:
                                    t5 = T3
                s_tsh[0] = t5

            # ============ attempt loop — MUST NOT unroll ===========
            listN = cutlass.Int32(0)
            above = cutlass.Int32(0)
            m = cutlass.Int32(0)
            need = cutlass.Int32(0)
            B = cutlass.Int32(0)
            SC = cutlass.Float32(1.0)
            TF = T
            complete = cutlass.Int32(0)
            valid = cutlass.Int32(0)

            if cutlass.const_expr(self.dtype == cutlass.Float32):
                fr = [
                    cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(U - PFD)
                ]  # explicit batch
            else:
                fr = [
                    cute.make_rmem_tensor((4,), cutlass.Uint32) for _ in range(U - PFD)
                ]  # explicit batch (bf16x8 vector as PACKED u32 pairs)
            # (empty for U<=PFD — every row-pass float4 then comes from pf[])
            att = cutlass.Int32(0)
            running = cutlass.Int32(1)
            while running != cutlass.Int32(0):
                if att > cutlass.Int32(0):  # retry preamble
                    # exactness: re-prime pf[] (holds stale roll data)
                    if rank < nFullG:
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            for uu in cutlass.range_constexpr(PFD):
                                ld_g_f32x4(
                                    atom128,
                                    x_addr,
                                    rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK),
                                    pf[uu],
                                )
                        else:
                            for uu in cutlass.range_constexpr(PFD):
                                ld_g_bf16x8_pk(
                                    atom128,
                                    x_addr,
                                    rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK),
                                    pf[uu],
                                )
                    else:
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            for uu in cutlass.range_constexpr(PFD):
                                i_ = rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
                                ic = i_
                                if ic >= n4:
                                    ic = lim4
                                ld_g_f32x4(atom128, x_addr, ic, pf[uu])
                        else:
                            for uu in cutlass.range_constexpr(PFD):
                                i_ = rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
                                ic = i_
                                if ic >= n4:
                                    ic = lim4
                                ld_g_bf16x8_pk(atom128, x_addr, ic, pf[uu])
                    _cluster_sync_aligned()  # ==== clus.sync (retry) ====
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
                # MUFU.RCP spelling: WD >= 1e-30 finite by the wdok clamp;
                # classify bucketing is SC-invariant for any SC > 0.
                SC = cute.arch.rcp_approx(WD)

                # ---- P3 row pass over OWNED CHUNKS ----
                g = rank + cutlass.Int32(0)
                while g < nCh:
                    i0 = g * cutlass.Int32(STEPC) + tidx
                    M = cutlass.Int32(0)
                    isfull = cutlass.Int32(0)
                    if g < nFullG:
                        isfull = cutlass.Int32(1)
                    if isfull != cutlass.Int32(0):  # full body
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            for uu in cutlass.range_constexpr(PFD, U):
                                ld_g_f32x4(
                                    atom128, x_addr, i0 + cutlass.Int32(uu * BLK), fr[uu - PFD]
                                )
                            for uu in cutlass.range_constexpr(U):
                                if cutlass.const_expr(uu < PFD):
                                    vv = pf[uu]
                                else:
                                    vv = fr[uu - PFD]
                                for q in cutlass.range_constexpr(4):
                                    M = M | (
                                        cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q)
                                    )
                        else:
                            for uu in cutlass.range_constexpr(PFD, U):
                                ld_g_bf16x8_pk(
                                    atom128, x_addr, i0 + cutlass.Int32(uu * BLK), fr[uu - PFD]
                                )
                            for uu in cutlass.range_constexpr(U):
                                if cutlass.const_expr(uu < PFD):
                                    vv = pf[uu]
                                else:
                                    vv = fr[uu - PFD]
                                for q in cutlass.range_constexpr(8):
                                    if cutlass.const_expr(q % 2 == 0):
                                        xq = f32_of_u32(vv[q // 2] << cutlass.Uint32(16))
                                    else:
                                        xq = f32_of_u32(vv[q // 2] & cutlass.Uint32(0xFFFF0000))
                                    M = M | (cutlass.Int32(xq >= TF) << cutlass.Int32(uu * 8 + q))
                    else:  # partial body
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            for uu in cutlass.range_constexpr(PFD, U):
                                i_ = i0 + cutlass.Int32(uu * BLK)
                                ic = i_
                                if ic >= n4:
                                    ic = lim4  # clamp in [n, npad)
                                ld_g_f32x4(atom128, x_addr, ic, fr[uu - PFD])
                            for uu in cutlass.range_constexpr(U):
                                if cutlass.const_expr(uu < PFD):
                                    vv = pf[uu]
                                else:
                                    vv = fr[uu - PFD]
                                i_ = i0 + cutlass.Int32(uu * BLK)
                                okq = cutlass.Int32(0)
                                if i_ < n4:
                                    okq = cutlass.Int32(1)
                                if okq != cutlass.Int32(0):  # +inf-pad escape, ok-gated
                                    for q in cutlass.range_constexpr(4):
                                        M = M | (
                                            cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q)
                                        )
                        else:
                            for uu in cutlass.range_constexpr(PFD, U):
                                i_ = i0 + cutlass.Int32(uu * BLK)
                                ic = i_
                                if ic >= n4:
                                    ic = lim4  # clamp in [n, npad)
                                ld_g_bf16x8_pk(atom128, x_addr, ic, fr[uu - PFD])
                            for uu in cutlass.range_constexpr(U):
                                if cutlass.const_expr(uu < PFD):
                                    vv = pf[uu]
                                else:
                                    vv = fr[uu - PFD]
                                i_ = i0 + cutlass.Int32(uu * BLK)
                                okq = cutlass.Int32(0)
                                if i_ < n4:
                                    okq = cutlass.Int32(1)
                                if okq != cutlass.Int32(0):  # +inf-pad escape, ok-gated
                                    for q in cutlass.range_constexpr(8):
                                        if cutlass.const_expr(q % 2 == 0):
                                            xq = f32_of_u32(vv[q // 2] << cutlass.Uint32(16))
                                        else:
                                            xq = f32_of_u32(vv[q // 2] & cutlass.Uint32(0xFFFF0000))
                                        M = M | (
                                            cutlass.Int32(xq >= TF) << cutlass.Int32(uu * 8 + q)
                                        )
                    # ROLL THE PREFETCH FORWARD: next OWNED chunk, issued
                    # before the reservation and the survivor walk.
                    g2 = g + cutlass.Int32(CS)
                    if g2 < nCh:
                        j0 = g2 * cutlass.Int32(STEPC) + tidx
                        infull = cutlass.Int32(0)
                        if g2 < nFullG:
                            infull = cutlass.Int32(1)
                        if infull != cutlass.Int32(0):
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                for uu in cutlass.range_constexpr(PFD):
                                    ld_g_f32x4(
                                        atom128, x_addr, j0 + cutlass.Int32(uu * BLK), pf[uu]
                                    )
                            else:
                                for uu in cutlass.range_constexpr(PFD):
                                    ld_g_bf16x8_pk(
                                        atom128, x_addr, j0 + cutlass.Int32(uu * BLK), pf[uu]
                                    )
                        else:
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                for uu in cutlass.range_constexpr(PFD):
                                    j_ = j0 + cutlass.Int32(uu * BLK)
                                    jc = j_
                                    if jc >= n4:
                                        jc = lim4
                                    ld_g_f32x4(atom128, x_addr, jc, pf[uu])
                            else:
                                for uu in cutlass.range_constexpr(PFD):
                                    j_ = j0 + cutlass.Int32(uu * BLK)
                                    jc = j_
                                    if jc >= n4:
                                        jc = lim4
                                    ld_g_bf16x8_pk(atom128, x_addr, jc, pf[uu])
                    # warp-aggregated slot reservation
                    cnt = cutlass.Int32(popc(M))
                    inc = warp_incl_scan_add(cnt, lane)
                    bpos = cutlass.Int32(0)
                    if lane == cutlass.Int32(31):
                        if inc != cutlass.Int32(0):
                            bpos = atomic_add_cta(s_scal.iterator + 0, inc)
                    pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) + (inc - cnt)
                    # survivor bit-walk, software-pipelined ONE deep;
                    # reload X[idx] — never hold the U float4s across the walk
                    if M != cutlass.Int32(0):
                        bp = ffs_m1(M)
                        M = M & (M - cutlass.Int32(1))
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            idx = (
                                (i0 + (bp >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                                << cutlass.Int32(2)
                            ) + (bp & cutlass.Int32(3))
                            xv = ldg_f32(x_addr, idx)
                        else:
                            idx = (
                                (i0 + (bp >> cutlass.Int32(3)) * cutlass.Int32(BLK))
                                << cutlass.Int32(3)
                            ) + (bp & cutlass.Int32(7))
                            xv = ldg_bf16(x_addr, idx)
                        while M != cutlass.Int32(0):
                            bp2 = ffs_m1(M)
                            M = M & (M - cutlass.Int32(1))
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                idx2 = (
                                    (i0 + (bp2 >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                                    << cutlass.Int32(2)
                                ) + (bp2 & cutlass.Int32(3))
                                xv2 = ldg_f32(x_addr, idx2)
                            else:
                                idx2 = (
                                    (i0 + (bp2 >> cutlass.Int32(3)) * cutlass.Int32(BLK))
                                    << cutlass.Int32(3)
                                ) + (bp2 & cutlass.Int32(7))
                                xv2 = ldg_bf16(x_addr, idx2)
                            pos = self._emitk(xv, idx, pos, TF, SC, SCAP, s_hist, s_cbuf2)
                            idx = idx2
                            xv = xv2
                        pos = self._emitk(xv, idx, pos, TF, SC, SCAP, s_hist, s_cbuf2)
                    g = g + cutlass.Int32(CS)
                # rank-0 scalar tail: per-thread atomics, bound-check
                i = tidx
                while i < tailn:
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        x = ldg_f32(x_addr, tail0 + i)
                    else:
                        x = ldg_bf16(x_addr, tail0 + i)
                    if x >= TF:
                        bq = f2s_rz((x - TF) * SC)  # signed form
                        if bq > cutlass.Int32(NBS - 1):
                            bq = cutlass.Int32(NBS - 1)
                        atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                        post = atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                        if post < SCAP:
                            s_cbuf2[post] = (
                                cutlass.Uint64(cutlass.Uint32(tail0 + i)) << cutlass.Uint64(32)
                            ) | cutlass.Uint64(u32_of_f32(x))
                    i = i + cutlass.Int32(BLK)

                # ---- cluster merge ----
                _cluster_sync_aligned()  # ==== clus.sync (merge) ====
                myn = s_scal[0]
                self._merge_scan0_local(s_hist, s_mrg, rank, k, tidx, s_res)
                cute.arch.barrier()  # ---- barrier (merge publish) ----
                tot = s_res[RES_TOT]
                acc = cutlass.Int32(0)
                if tot >= k:
                    acc = cutlass.Int32(1)
                if acc != cutlass.Int32(0):  # accept
                    valid = cutlass.Int32(1)
                    complete = cutlass.Int32(0)
                    if myn <= SCAP:
                        complete = cutlass.Int32(1)
                    listN = myn
                    above = s_res[RES_ABOVE]
                    m = s_res[RES_M]
                    need = k - s_res[RES_ABOVE]
                    B = s_res[RES_B]
                    running = cutlass.Int32(0)
                else:
                    if att == cutlass.Int32(2):  # ladder exhausted
                        running = cutlass.Int32(0)
                    else:
                        # rung ladder — cluster-uniform on every arm
                        tshtaken = cutlass.Int32(0)
                        if att == cutlass.Int32(0):
                            T5 = s_tsh[0]
                            if T5 > cutlass.Float32(_NEG_INF__clus):
                                if T5 < TF:
                                    T = T5
                                    tshtaken = cutlass.Int32(1)
                        if tshtaken == cutlass.Int32(0):
                            # LAZY GATHER — every rank computes identical GMIN
                            if GMIN == cutlass.Float32(SENT_LO):
                                if cutlass.const_expr(not self.hint_free):
                                    GMIN, GMAX = gather_hint(
                                        x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk=BLK, kpt=1
                                    )  # 2 barriers inside
                            floorhit = cutlass.Int32(1)
                            if T > GMIN:
                                floorhit = cutlass.Int32(0)
                            if floorhit != cutlass.Int32(0):
                                running = cutlass.Int32(0)
                            else:
                                T = GMIN
                att = att + cutlass.Int32(1)

            # ============ classification ============
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
            if m > CMP:
                degen = cutlass.Int32(1)
            mc = cutlass.Int32(0)
            if degen == cutlass.Int32(0):
                mc = m
            # crossing candidates land in RANK 0's ck64c via DSMEM
            rk64 = _mapa_shared_cluster_addr(ck_addr, cutlass.Int32(0))

            if degen == cutlass.Int32(0):
                if complete != cutlass.Int32(0):
                    # ---- P5 emit from staged cbuf ----
                    i = tidx
                    while i < listN:
                        pk64 = s_cbuf2[i]
                        vx = cutlass.Int32(cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF)))
                        idv = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                        xv = f32_of_i32(vx)
                        self._p5_emit(
                            xv, idv, TF, SC, B, above, lim1, whole, CMP, s_mrg, out_row, rk64
                        )
                        i = i + cutlass.Int32(BLK)
                else:
                    # ---- exactness re-sweep: OWNED CHUNKS + rank-0 true tail ----
                    g = rank + cutlass.Int32(0)
                    while g < nCh:
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            lo2 = (g * cutlass.Int32(STEPC)) << cutlass.Int32(2)
                        else:
                            lo2 = (g * cutlass.Int32(STEPC)) << cutlass.Int32(3)
                        e4 = (g + cutlass.Int32(1)) * cutlass.Int32(STEPC)
                        if e4 > n4:
                            e4 = n4
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            hi2 = e4 << cutlass.Int32(2)
                        else:
                            hi2 = e4 << cutlass.Int32(3)
                        i = lo2 + tidx
                        while i < hi2:
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                x = ldg_f32(x_addr, i)
                            else:
                                x = ldg_bf16(x_addr, i)
                            if x >= TF:
                                self._p5_emit(
                                    x, i, TF, SC, B, above, lim1, whole, CMP, s_mrg, out_row, rk64
                                )
                            i = i + cutlass.Int32(BLK)
                        g = g + cutlass.Int32(CS)
                    t2 = tidx
                    while t2 < tailn:
                        ii = tail0 + t2
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            x = ldg_f32(x_addr, ii)
                        else:
                            x = ldg_bf16(x_addr, ii)
                        if x >= TF:
                            self._p5_emit(
                                x, ii, TF, SC, B, above, lim1, whole, CMP, s_mrg, out_row, rk64
                            )
                        t2 = t2 + cutlass.Int32(BLK)

            # ============ EXIT RENDEZVOUS ============
            # all DSMEM traffic retired; the ONLY exit rendezvous. rank!=0
            # falls through to the kernel end (post-barrier asymmetric exit);
            # NO later cluster barrier.
            _cluster_sync_aligned()  # ==== clus.sync (exit) ====

            if rank == cutlass.Int32(0):
                if degen == cutlass.Int32(0):
                    if whole == cutlass.Int32(0):
                        # ---- P6 rank-0 refine ----
                        if mc <= cutlass.Int32(QUADC_CLUS__clus):  # O(mc^2)
                            mc2 = mc & cutlass.Int32(~1)
                            i = tidx
                            while i < mc:
                                # re-assert Uint64 at every unsigned compare
                                # in/after dynamic loops.
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
                                    out_row[above + r_] = cutlass.Int32(
                                        cutlass.Uint32(
                                            cutlass.Uint64(u64v) & cutlass.Uint64(0xFFFFFFFF)
                                        )
                                    )
                                i = i + cutlass.Int32(BLK)
                        else:
                            # key-space narrowing over ck64c
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
                                        rhi = nlo + (
                                            (cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1)
                                        )
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
                    # ---- degen fallback: whole-row key-space narrowing
                    # (per-level clear + scan_cross w/ ws) ----
                    rlo = cutlass.Uint32(0)
                    rhi = cutlass.Uint32(0xFFFFFFFF)
                    above2 = cutlass.Int32(0)
                    need2 = k
                    m2 = n
                    ethr = cutlass.Int64(0)
                    tie_m = cutlass.Int32(1)
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
                            if tidx < cutlass.Int32(NBS):  # per-level clear
                                s_hist[tidx] = cutlass.Int32(0)
                            cute.arch.barrier()  # ---- barrier (level clear) ----
                            i = tidx
                            while i < n:  # whole row
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
                            # block-parallel scan (ONE internal barrier; only
                            # use of ws in this kernel)
                            scan_cross(
                                s_hist,
                                s_ws,
                                need2,
                                tidx,
                                s_res,
                                cutlass.Int32(0),
                                blk=BLK,
                                nb=NBS,
                                two=False,
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
                            p1, p2, i, cutlass.Int32(0), nA, nA, nT, out_row, s_scal, lane
                        )
                        it = it + cutlass.Int32(1)

    # ------------------------------------------------------------------
    # host launcher: grid dim3(CS, b) + cluster (CS,1,1);
    # min_blocks_per_mp=1 == __launch_bounds__(1024, 1) 64-register wall.
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        SCAP: cutlass.Int32,
        CMP: cutlass.Int32,
        SMP: cutlass.Int32,
        TGT: cutlass.Int32,
        Q: cutlass.Int32,
        SS2: cutlass.Int32,
        TGT2: cutlass.Int32,
        stream,
    ):
        # varlen grid rows = out.shape[0] (logits row count == out row count in
        # both modes); bigf = the route() `big` occupancy flag, a pure function
        # of (rows, CS) so it is launch-computed, not an ABI scalar.
        b = out.shape[0]
        bigf = cutlass.Int32(0)
        if b * cutlass.Int32(self.cs) <= cutlass.Int32(148):
            bigf = cutlass.Int32(1)
        self.kern(
            logits, pre_idx, kv_lens, out, n, npad, k, SCAP, CMP, SMP, TGT, Q, SS2, TGT2, bigf
        ).launch(
            grid=(self.cs, b, 1),
            block=(self.blk, 1, 1),
            cluster=(self.cs, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.minb,
        )


_COMPILE_CACHE__clus = {}


def get_compiled__clus(
    tpl: tuple,
    scap: int = 8192,
    cmp_: int = 2048,
    options_extra: str = "",
    varlen: bool = False,
    next_n: int = 1,
    cr_shift: int = 0,
    hint_free: bool = False,
    dtype: str = "f32",
) -> Any:
    """Compile (or fetch) the gvr_clus variant for constexpr tuple
    tpl = (BLK, U, MINB, NBS, CS); scap/cmp are smem-extent keys (every
    reachable route has 8192/2048 — asserted by run__clus())."""
    key = (
        tuple(tpl),
        scap,
        cmp_,
        options_extra,
        bool(varlen),
        int(next_n),
        int(cr_shift),
        bool(hint_free),
        str(dtype),
    )
    hit = _COMPILE_CACHE__clus.get(key)
    if hit is not None:
        return hit
    blk, u, minb, nbs, cs = tpl
    kern = GvrClusKernel(
        blk,
        u,
        minb,
        nbs,
        cs,
        scap=scap,
        cmp_=cmp_,
        varlen=varlen,
        next_n=next_n,
        cr_shift=cr_shift,
        hint_free=hint_free,
        dtype=_dtype_of(dtype),
    )
    r0, c0 = cute.sym_int(), cute.sym_int()
    r1, c1 = cute.sym_int(), cute.sym_int()
    r2, c2 = cute.sym_int(), cute.sym_int()
    if dtype == "f32":
        logits_fake = _crt.make_fake_compact_tensor(
            cutlass.Float32, (r0, c0), stride_order=(1, 0), assumed_align=16
        )
    else:
        logits_fake = _crt.make_fake_compact_tensor(
            cutlass.BFloat16, (r0, c0), stride_order=(1, 0), assumed_align=16
        )
    pre_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r1, c1), stride_order=(1, 0), assumed_align=16
    )
    out_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r2, c2), stride_order=(1, 0), assumed_align=16
    )
    v0 = cute.sym_int()
    kv_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (v0,), stride_order=(0,), assumed_align=4
    )
    fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kern,
        logits_fake,
        pre_fake,
        kv_fake,
        out_fake,
        *([cutlass.Int32(0)] * 10),
        stream=fake_stream,
        options=("--enable-tvm-ffi " + options_extra).strip(),
    )
    _COMPILE_CACHE__clus[key] = compiled
    return compiled


NB__regclus = 1024  # histogram bins; == BLKC here


LNB = 10  # log2(NB__regclus) — reg_clus narrowing shift


QUADC__regclus = 96  # O(mc^2) rank gate


CMPC = 4096  # crossing slots PER CTA (pow2)


LCMPC = 12  # log2(CMPC)


BLKC = 1024  # CTA size


STATIC_WORDS__regclus = 128  # DSL smem prelude (static-__shared__ mirror)


STATIC_BYTES__regclus = STATIC_WORDS__regclus * 4


DYN_SMEM_BYTES = (3 * NB__regclus + 2 * CMPC) * 4  # 45,056


SMEM_BYTES = STATIC_BYTES__regclus + DYN_SMEM_BYTES  # 45,568


W_HIST = STATIC_WORDS__regclus


W_MRG = STATIC_WORDS__regclus + NB__regclus


W_HOFF = STATIC_WORDS__regclus + 2 * NB__regclus


W_CK = STATIC_WORDS__regclus + 3 * NB__regclus


W_CI = STATIC_WORDS__regclus + 3 * NB__regclus + CMPC


_NEG_INF__regclus = float("-inf")


def _val8__regclus(frags, s: int):
    """bf16 twin of _val__regclus: 8-wide (16B bf16x8) register batch."""
    return frags[s // 8][s % 8]


def _val8_pk__regclus(frags, s: int):
    """bf16 twin of _val__regclus over packed bf16x8 u32-pair batches."""
    w = frags[s // 8][(s % 8) // 2]
    if s % 2 == 0:
        return f32_of_u32(w << cutlass.Uint32(16))
    return f32_of_u32(w & cutlass.Uint32(0xFFFF0000))


class GvrRegClusKernel:
    """gvr_reg_clus<BLK, VPT, CS>."""

    def __init__(
        self,
        blk: int,
        vpt: int,
        cs: int,
        pdl: bool = False,
        varlen: bool = False,
        next_n: int = 1,
        cr_shift: int = 0,
        hint_free: bool = False,
        dtype=cutlass.Float32,
    ) -> None:
        assert blk == BLKC, "all instantiations BLK=BLKC=1024"
        # CS=16 is a bf16-route-only widening: the merge/striping phases are
        # already CS-parametric (range_constexpr(CS) mapa loops, rank slabs
        # q2i >> LCMPC with CS*CMPC capacity) and 16-CTA clusters are within
        # the SM100 nonportable limit. fp32 route() never emits cs > 8.
        assert vpt in (1, 2, 4) and cs in (2, 4, 8, 16)
        # constexpr logits dtype (Float32 verbatim arm / BFloat16 native arm)
        self.dtype = dtype
        assert dtype in (cutlass.Float32, cutlass.BFloat16)
        self.blk = blk
        self.vpt = vpt
        self.cs = cs
        self.pdl = bool(pdl)
        # per-row varlen mode (production heuristicTopKDecode contract, same
        # semantics as GvrMainKernel): n is re-derived PER ROW in-kernel from
        # a device kv_lens tensor; the scalar n launch arg becomes the
        # envelope clamp bound. next_n / cr_shift are compile-time.
        self.varlen = bool(varlen)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        if self.varlen:
            assert self.next_n >= 1 and self.cr_shift in (0, 2)
        # hint-free: P0 samples the first k row elements (coalesced) instead of the hint
        self.hint_free = bool(hint_free)
        if dtype is cutlass.Float32:
            self.S = vpt * 4
        else:
            self.S = vpt * 8  # bf16: 8 elems per 16B vector, same element count
        self.span = blk * vpt  # float4 per CTA

    # ------------------------------------------------------------------
    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
    ):
        BLK = cutlass.const_expr(self.blk)
        VPT = cutlass.const_expr(self.vpt)
        CS = cutlass.const_expr(self.cs)
        S = cutlass.const_expr(self.S)
        NW = cutlass.const_expr(self.blk // 32)
        MERGE4 = cutlass.const_expr(
            self.dtype == cutlass.BFloat16 and NB__regclus == 512 and self.vpt == 1 and self.cs == 4
        )

        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()  # knob default off

        tid, _, _ = cute.arch.thread_idx()
        rank, row, _ = cute.arch.block_idx()  # bx=rank
        lane = tid & cutlass.Int32(31)

        # ---- bf16 varlen head-latency hoist (see GvrTopkRegKernel): the P1
        # row batch is address-independent of kv_lens (the envelope arg n
        # bounds it inside the padded row) but in warp issue order it sits
        # BEHIND the kv_lens-dependent P0 bracket gather, so the whole batch
        # serializes on the kv_lens round trip. Issue it before the varlen
        # prologue; lanes at/beyond the per-row window are overwritten with
        # packed/widened -inf at P1 (register state identical to the
        # unhoisted batch; pad contents never consulted). fp32 arm:
        # constexpr-eliminated, codegen untouched.
        if cutlass.const_expr(self.dtype != cutlass.Float32 and self.varlen):
            hcx_addr = logits[row, None].iterator.toint()
            hcatom = g2r_atom_f32(128, invariant=True)
            envc8 = n >> cutlass.Int32(3)
            if envc8 > cutlass.Int32(self.cs * self.span):
                envc8 = cutlass.Int32(self.cs * self.span)
            hcbase = rank * cutlass.Int32(self.span)
            if cutlass.const_expr(VPT >= 2):
                hcfrags = [cute.make_rmem_tensor((4,), cutlass.Uint32) for _ in range(VPT)]
                for u in cutlass.range_constexpr(VPT):
                    hci = hcbase + tid + cutlass.Int32(u * self.blk)
                    if hci < envc8:
                        ld_g_bf16x8_pk(hcatom, hcx_addr, hci, hcfrags[u])
            else:
                hcfrags = [cute.make_rmem_tensor((8,), cutlass.Float32) for _ in range(VPT)]
                for u in cutlass.range_constexpr(VPT):
                    hci = hcbase + tid + cutlass.Int32(u * self.blk)
                    if hci < envc8:
                        ld_g_bf16x8(hcatom, hcx_addr, hci, hcfrags[u])
            # P0/P3 bracket gather hoist: the sample address is a pure
            # function of envelope-only quantities (tid, k, grid rows, the
            # envelope arg n), so the gather — the ladder's critical-path
            # load — issues here instead of serializing behind kv_lens.
            # Contribution at P3 keeps the per-row `< n` guard, so rows
            # shorter than the envelope contribute a SUBSET of the original
            # sample set (never a pad value); the bracket only initializes
            # the ladder, count-crossing still proves exactness.
            hpv = cutlass.Int32(-1)
            hgv = cutlass.Float32(0.0)
            if cutlass.const_expr(self.hint_free):
                if cutlass.const_expr(self.cs >= 16):
                    hpv = tid * (n >> cutlass.Int32(10))
                else:
                    _, hgrows, _ = cute.arch.grid_dim()
                    khq = cutlass.Int32(pre_idx.shape[1])
                    if khq > cutlass.Int32(self.blk) and hgrows > cutlass.Int32(1):
                        hpv = tid * (n >> cutlass.Int32(10))
                    else:
                        if tid < khq:
                            hpv = tid
                if cutlass.Uint32(hpv) < cutlass.Uint32(n):
                    hgv = ldg_bf16(hcx_addr, hpv)

        # ============ per-row varlen prologue — shared contract lives in ======
        # GvrMainKernel's prologue (per-row n from kv_lens, clamped to the
        # envelope arg; the launcher admits this family only when the envelope
        # fits its capacity window). Pure functions of `row` keep the body
        # guard and cluster barriers cluster-uniform. Short rows (n <= k):
        # rank 0 emits identity + (-1) and the body is SKIPped — a zero-work
        # pass would reach the crossing-overflow emitter and poison the output.
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
                if rank == cutlass.Int32(0):
                    # BF16 admits k > BLK; rank 0 must cover every output
                    # column, including the identity and -1 padding tail.
                    short_idx = tid
                    while short_idx < kq:
                        ov = cutlass.Int32(-1)
                        if short_idx < nv:
                            ov = short_idx
                        out[row, short_idx] = ov
                        short_idx = short_idx + cutlass.Int32(BLK)

        # ------------------------------------------------------------------
        # Predeclarations (DSL AST rule: every scalar (re)assigned under a
        # dynamic if/while must pre-exist with a stable type; constant inits
        # are dead-coded).
        # ------------------------------------------------------------------
        i = cutlass.Int32(0)
        j = cutlass.Int32(0)
        rnk = cutlass.Int32(0)
        tinc = cutlass.Int32(0)
        mc = cutlass.Int32(0)
        p = cutlass.Int32(0)
        q2i = cutlass.Int32(0)
        idx = cutlass.Int32(0)
        lim1 = cutlass.Int32(0)
        aboveC = cutlass.Int32(0)
        needC = cutlass.Int32(0)
        mm = cutlass.Int32(0)
        lev = cutlass.Int32(0)
        done = cutlass.Int32(0)
        b2w = cutlass.Int32(0)
        sh2 = cutlass.Int32(0)
        b_lv = cutlass.Int32(0)
        it = cutlass.Int32(0)
        it2 = cutlass.Int32(0)
        idv = cutlass.Int32(0)
        q1f = cutlass.Int32(0)
        q2f = cutlass.Int32(0)
        n1 = cutlass.Int32(0)
        n2 = cutlass.Int32(0)
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        p1e = cutlass.Int32(0)
        p2e = cutlass.Int32(0)
        lml = cutlass.Int32(0)
        nA = cutlass.Int32(0)
        nT = cutlass.Int32(0)
        tie_m = cutlass.Int32(0)
        pv0 = cutlass.Int32(-1)
        okc = cutlass.Int32(0)
        whole = cutlass.Int32(0)
        degen = cutlass.Int32(0)
        pre_a = cutlass.Int32(0)
        tot_a = cutlass.Int32(0)
        uk = cutlass.Uint32(0)
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
        bn = cutlass.Uint32(0)
        ethr = cutlass.Int64(0)
        tval = cutlass.Float32(_NEG_INF__regclus)
        LOQ = cutlass.Float32(0.0)
        qv = cutlass.Float32(0.0)

        if short == cutlass.Int32(0):
            npad = cutlass.Int32(logits.shape[1])  # noqa: F841
            k = cutlass.Int32(pre_idx.shape[1])
            out_row = out[row, None]
            x_addr = logits[row, None].iterator.toint()  # Int64 gmem byte base
            p_addr = pre_idx[prow, None].iterator.toint()  # request-level under varlen

            # ---- shared-memory window (map in module docstring) ----
            sptr = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
            sbase = sptr.toint()  # Int32 shared addr

            s_res = _smem_view__regclus(cutlass.Int32, sbase, 0, 6)
            s_cnt = _smem_view__regclus(cutlass.Int32, sbase, 6, 2)  # [0]=s_o1 [1]=s_o2
            s_kmm = _smem_view__regclus(cutlass.Uint32, sbase, 8, 2)  # [0]=s_kmin [1]=s_kmax
            s_ws = _smem_view__regclus(cutlass.Int32, sbase, 16, 32)
            s_wmn = _smem_view__regclus(cutlass.Uint32, sbase, 48, 32)
            s_wmx = _smem_view__regclus(cutlass.Uint32, sbase, 80, 32)
            s_hist = _smem_view__regclus(cutlass.Int32, sbase, W_HIST, NB__regclus)
            s_mrg = _smem_view__regclus(cutlass.Int32, sbase, W_MRG, NB__regclus)
            s_hoff = _smem_view__regclus(cutlass.Int32, sbase, W_HOFF, NB__regclus)
            s_ck = _smem_view__regclus(cutlass.Uint32, sbase, W_CK, CMPC)
            s_ci = _smem_view__regclus(cutlass.Int32, sbase, W_CI, CMPC, align=4)
            # raw byte bases for DSMEM (mapa) addressing
            hist_addr = sbase + cutlass.Int32(W_HIST * 4)
            ck_addr = sbase + cutlass.Int32(W_CK * 4)
            ci_addr = sbase + cutlass.Int32(W_CI * 4)

            if cutlass.const_expr(self.dtype == cutlass.Float32):
                n4 = n >> cutlass.Int32(2)
                ntail = n - (n4 << cutlass.Int32(2))
                base4 = rank * cutlass.Int32(self.span)
                tix = (n4 << cutlass.Int32(2)) + tid  # CUDA `tidx`
            else:
                # bf16 slot-exact window (see GvrTopkRegKernel): cap the
                # vector window at the cluster's register capacity (cs * span
                # vectors) and hand the (<= BLK, route-enforced) element
                # overflow to the existing rank-0 scalar tail — the tail is
                # value-generic (trash-bin histogram, tval==-inf elsewhere),
                # so the bf16 route may pick (v, cs) that under-cover by up
                # to BLK elements instead of doubling v or cs for a 64-pad
                # envelope overflow.
                n4 = n >> cutlass.Int32(3)  # bf16: 8-elem 16B vectors
                if n4 > cutlass.Int32(self.cs * self.span):
                    n4 = cutlass.Int32(self.cs * self.span)
                ntail = n - (n4 << cutlass.Int32(3))
                base4 = rank * cutlass.Int32(self.span)
                tix = (n4 << cutlass.Int32(3)) + tid

            # ---- P0: redundant hint gather, EVERY CTA (k<=BLK by dispatch
            # gate). One coalesced word per thread, NO cluster barrier —
            # GMIN/GMAX identical everywhere by construction.
            if cutlass.const_expr(self.hint_free):
                if cutlass.const_expr(self.dtype == cutlass.BFloat16):
                    _, grid_rows, _ = cute.arch.grid_dim()
                    if cutlass.const_expr(self.cs >= 16):
                        # 16-CTA envelopes (163K-262K columns): the first-K
                        # prefix covers < 1.3% of the row, and a collapsed
                        # bracket sends every row through the striped-slab
                        # narrowing (probe: pro_1024k_bs2 17.9us prefix vs the
                        # 11.9us split baseline). Whole-row 1/1024 stride
                        # restores a full-range bracket; the sample only
                        # initializes the ladder, count crossing stays exact.
                        pv0 = tid * (n >> cutlass.Int32(10))
                    else:
                        if k > cutlass.Int32(self.blk) and grid_rows > cutlass.Int32(1):
                            # Multi-row K=2048 captures need a whole-row bracket:
                            # their first-K prefix collapses the upper tail into
                            # one large crossing bin. The sample only initializes
                            # the ladder; count crossing still proves exactness.
                            pv0 = tid * (n >> cutlass.Int32(10))
                        else:
                            if tid < k:
                                pv0 = tid
                else:
                    if tid < k:
                        pv0 = tid
            else:
                if tid < k:
                    pv0 = ld_g_i32(p_addr, tid)

            # ---- P1: row load — predicated flat float4[VPT] batch (the CUDA
            # has NO exact-fit peel here, guard is per-load). Issue all loads
            # first, then -INFINITY-fill missed slots.
            atom128 = g2r_atom_f32(128, invariant=True)
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                frags = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(VPT)]
                for u in cutlass.range_constexpr(VPT):
                    i = base4 + tid + cutlass.Int32(u * self.blk)
                    if i < n4:
                        ld_g_f32x4(atom128, x_addr, i, frags[u])
                for u in cutlass.range_constexpr(VPT):
                    i = base4 + tid + cutlass.Int32(u * self.blk)
                    if i >= n4:  # -INFINITY fill
                        for z in cutlass.range_constexpr(4):
                            frags[u][z] = cutlass.Float32(_NEG_INF__regclus)
            else:
                if cutlass.const_expr(self.varlen):
                    # Batch already in flight from the kernel-head hoist:
                    # adopt it and overwrite lanes at/beyond the per-row
                    # window (identical register state to the unhoisted
                    # predicated batches below).
                    frags = hcfrags
                    if cutlass.const_expr(VPT >= 2):
                        for u in cutlass.range_constexpr(VPT):
                            hcj = base4 + tid + cutlass.Int32(u * self.blk)
                            if hcj >= n4:  # -INFINITY fill (packed bf16x2 -inf)
                                for z in cutlass.range_constexpr(4):
                                    frags[u][z] = cutlass.Uint32(0xFF80FF80)
                    else:
                        for u in cutlass.range_constexpr(VPT):
                            hcj = base4 + tid + cutlass.Int32(u * self.blk)
                            if hcj >= n4:  # -INFINITY fill
                                for z in cutlass.range_constexpr(8):
                                    frags[u][z] = cutlass.Float32(_NEG_INF__regclus)
                elif cutlass.const_expr(VPT >= 2):
                    # packed retention: bf16x8 vector = FOUR raw u32 pairs
                    # (not eight widened f32) held live across the classify
                    # phases; unpack (shift/mask, exact) happens per use in
                    # _val8_pk__regclus. Halves the long-lived value
                    # registers — the VPT=4 batch carries 16 u32 instead of
                    # 32 f32, staying under the 64-register launch wall.
                    frags = [cute.make_rmem_tensor((4,), cutlass.Uint32) for _ in range(VPT)]
                    for u in cutlass.range_constexpr(VPT):
                        i = base4 + tid + cutlass.Int32(u * self.blk)
                        if i < n4:
                            ld_g_bf16x8_pk(atom128, x_addr, i, frags[u])
                    for u in cutlass.range_constexpr(VPT):
                        i = base4 + tid + cutlass.Int32(u * self.blk)
                        if i >= n4:  # -INFINITY fill (packed bf16x2 -inf)
                            for z in cutlass.range_constexpr(4):
                                frags[u][z] = cutlass.Uint32(0xFF80FF80)
                else:
                    frags = [cute.make_rmem_tensor((8,), cutlass.Float32) for _ in range(VPT)]
                    for u in cutlass.range_constexpr(VPT):
                        i = base4 + tid + cutlass.Int32(u * self.blk)
                        if i < n4:
                            ld_g_bf16x8(atom128, x_addr, i, frags[u])
                    for u in cutlass.range_constexpr(VPT):
                        i = base4 + tid + cutlass.Int32(u * self.blk)
                        if i >= n4:  # -INFINITY fill
                            for z in cutlass.range_constexpr(8):
                                frags[u][z] = cutlass.Float32(_NEG_INF__regclus)
            # tail element: rank 0 only
            if rank == cutlass.Int32(0):
                if tid < ntail:
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        tval = ldg_f32(x_addr, tix)
                    else:
                        tval = ldg_bf16(x_addr, tix)

            # ---- P2: init. NB__regclus == BLK -> single-pass hist clear.
            if tid == cutlass.Int32(0):
                s_cnt[0] = cutlass.Int32(0)
                s_cnt[1] = cutlass.Int32(0)
            if cutlass.const_expr(NB__regclus < self.blk):
                if tid < cutlass.Int32(NB__regclus):
                    s_hist[tid] = cutlass.Int32(0)
            else:
                for z in cutlass.range_constexpr(NB__regclus // self.blk):
                    s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)

            # ---- P3: GMIN/GMAX from the hint, ONE barrier fold.
            lmin = cutlass.Uint32(0xFFFFFFFF)
            lmax = cutlass.Uint32(0)
            if cutlass.const_expr(
                self.dtype == cutlass.BFloat16 and self.varlen and self.hint_free
            ):
                # value preloaded at the kernel head (see gather hoist);
                # contribution stays per-row-guarded so short rows sample a
                # subset of the original set, never a pad column.
                if cutlass.Uint32(hpv) < cutlass.Uint32(n):
                    uk = fkey(hgv)
                    lmin = uk
                    lmax = uk
            else:
                if cutlass.Uint32(pv0) < cutlass.Uint32(n):
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        uk = fkey(ldg_f32(x_addr, pv0))  # __ldg(X+pv0)
                    else:
                        uk = fkey(ldg_bf16(x_addr, pv0))
                    lmin = uk
                    lmax = uk
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

            # ---- collapse guard, NaN-safe. Reject infinite width as well:
            # otherwise SC becomes zero and can collapse +inf into a finite
            # histogram bin.
            okc = cutlass.Int32(0)
            if Tv < GMAX:
                w_ = GMAX - Tv
                if w_ > cutlass.Float32(1e-30) and w_ < cutlass.Float32(3.5e38):
                    okc = cutlass.Int32(1)
            if okc == cutlass.Int32(0):
                Tv = cutlass.Float32(SENT_LO)
                GMAX = cutlass.Float32(SENT_HI)

            # ---- bin transform constants: branchless trash bin.
            WD = (GMAX - Tv) * cutlass.Float32(1.0 / float(NB__regclus - 2))
            wsel = cutlass.Float32(1e-30)
            if WD > cutlass.Float32(0.0):
                wsel = WD
            # MUFU.RCP spelling (mirrors the reg family's site):
            # wsel >= 1e-30 finite; bucketing is SC-invariant for any SC > 0.
            SC = cute.arch.rcp_approx(wsel)
            CQ0 = cutlass.Float32(1.0) - Tv * SC
            CQ = CQ0 + cutlass.Float32(1e-6) * (_fabsf__regclus(CQ0) + cutlass.Float32(1.0))

            # ---- P4: histogram; tval add UNCONDITIONAL (trash bin swallows
            # -INFINITY via the saturating cvt).
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                for s in cutlass.range_constexpr(S):
                    qv = _fmaf__regclus(_val__regclus(frags, s), SC, CQ)
                    bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                    atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))
            else:
                if cutlass.const_expr(self.cs >= 16):
                    if base4 >= n4:
                        # Fully-idle rank (window entirely beyond the row, only
                        # reachable on the slack ranks of the widened envelopes):
                        # every slot carries the -inf fill, so all S*BLK adds land
                        # in the saturating trash bin. One store writes the exact
                        # same histogram (hist was just cleared, no other writer
                        # before the cluster merge) without S*BLK same-address
                        # smem atomics convoying the P5 rendezvous.
                        if tid == cutlass.Int32(0):
                            s_hist[0] = cutlass.Int32(S * self.blk)
                    else:
                        for s in cutlass.range_constexpr(S):
                            if cutlass.const_expr(VPT >= 2):
                                qv = _fmaf__regclus(_val8_pk__regclus(frags, s), SC, CQ)
                            else:
                                qv = _fmaf__regclus(_val8__regclus(frags, s), SC, CQ)
                            bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                            atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                else:
                    for s in cutlass.range_constexpr(S):
                        if cutlass.const_expr(VPT >= 2):
                            qv = _fmaf__regclus(_val8_pk__regclus(frags, s), SC, CQ)
                        else:
                            qv = _fmaf__regclus(_val8__regclus(frags, s), SC, CQ)
                        bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                        atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                qv = _fmaf__regclus(tval, SC, CQ)
                bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))
            else:
                if rank == cutlass.Int32(0):
                    if tid < ntail:
                        qv = _fmaf__regclus(tval, SC, CQ)
                        bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                        atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))

            # ---- P5: cluster merge
            _cluster_sync_aligned()  # histograms complete on every rank
            if cutlass.const_expr(MERGE4):
                merge_scan4_regclus(s_hist, s_mrg, s_ws, rank, k, tid, s_res, cs=CS)
            else:
                if cutlass.const_expr(NB__regclus < self.blk):
                    if tid < cutlass.Int32(NB__regclus):
                        i = tid
                        hvals = []
                        for r in cutlass.range_constexpr(CS):
                            ma = _mapa_shared_cluster_addr(
                                hist_addr + (i << cutlass.Int32(2)), cutlass.Int32(r)
                            )
                            hvals.append(_ld_shared_cluster_i32(ma))
                        tot_a = cutlass.Int32(0)
                        pre_a = cutlass.Int32(0)
                        for r in cutlass.range_constexpr(CS):
                            if cutlass.Int32(r) < rank:
                                pre_a = pre_a + hvals[r]
                            tot_a = tot_a + hvals[r]
                        s_mrg[i] = tot_a
                        s_hoff[i] = pre_a
                else:
                    for z in cutlass.range_constexpr(NB__regclus // self.blk):
                        i = tid + cutlass.Int32(z * self.blk)
                        # CS-unrolled remote u32 loads: batch-issue, then fold
                        # (one mapa per (i, r) exactly like the CUDA)
                        hvals = []
                        for r in cutlass.range_constexpr(CS):
                            ma = _mapa_shared_cluster_addr(
                                hist_addr + (i << cutlass.Int32(2)), cutlass.Int32(r)
                            )
                            hvals.append(_ld_shared_cluster_i32(ma))
                        tot_a = cutlass.Int32(0)
                        pre_a = cutlass.Int32(0)
                        for r in cutlass.range_constexpr(CS):
                            if cutlass.Int32(r) < rank:
                                pre_a = pre_a + hvals[r]  # rank-exclusive
                            tot_a = tot_a + hvals[r]
                        s_mrg[i] = tot_a
                        s_hoff[i] = pre_a

                # ---- P6: scan
                cute.arch.barrier()  # merge published
                if cutlass.const_expr(NB__regclus < self.blk):
                    scan_cross(
                        s_mrg,
                        s_ws,
                        k,
                        tid,
                        s_res,
                        cutlass.Int32(0),
                        blk=self.blk,
                        nb=NB__regclus,
                        two=False,
                    )
                else:
                    scan_cross_w(s_mrg, s_ws, k, tid, s_res, blk=self.blk, nb=NB__regclus)
            cute.arch.barrier()  # scan published
            above = s_res[RES_ABOVE]
            m = s_res[RES_M]
            Bv = s_res[RES_B]
            need = k - above
            whole = cutlass.Int32(0)
            if need >= m:
                whole = cutlass.Int32(1)
            degen = cutlass.Int32(0)
            if m > cutlass.Int32(CS * CMPC):
                degen = cutlass.Int32(1)
            # Global per-bin cursors permit a singleton crossing bin to
            # emit any needed unique ties without DSM candidate staging.
            if cutlass.const_expr(self.dtype == cutlass.BFloat16):
                if whole == cutlass.Int32(0) and m > cutlass.Int32(0):
                    if Bv > cutlass.Int32(0) and Bv < cutlass.Int32(NB__regclus - 1):
                        if SC > cutlass.Float32(0.0):
                            center = (
                                cutlass.Float32(Bv) + cutlass.Float32(0.5) - CQ
                            ) * cute.arch.rcp_approx(SC)
                            center_bits = _bf16_rne_bits__reg(center)
                            magnitude = center_bits & cutlass.Uint32(0x7FFF)
                            # Exclude signed zero, subnormal boundaries, and
                            # the maximal finite encodings next to infinity.
                            if magnitude > cutlass.Uint32(0x0080) and magnitude < cutlass.Uint32(
                                0x7F7F
                            ):
                                previous_bits = center_bits - cutlass.Uint32(1)
                                next_bits = center_bits + cutlass.Uint32(1)
                                if (center_bits & cutlass.Uint32(0x8000)) != cutlass.Uint32(0):
                                    previous_bits = center_bits + cutlass.Uint32(1)
                                    next_bits = center_bits - cutlass.Uint32(1)
                                previous = f32_of_u32(previous_bits << cutlass.Uint32(16))
                                following = f32_of_u32(next_bits << cutlass.Uint32(16))
                                # Interior bin boundaries are exactly representable.
                                # These FMA bounds imply the original f2u/clamp
                                # classifications while avoiding three conversions.
                                previous_q = _fmaf__regclus(previous, SC, CQ)
                                next_q = _fmaf__regclus(following, SC, CQ)
                                bin_low = cutlass.Float32(Bv)
                                bin_high = cutlass.Float32(Bv + cutlass.Int32(1))
                                # The bin is nonempty (m > 0). Monotonicity
                                # and these neighbor bounds restrict it to
                                # the candidate value, so checking that value
                                # itself would repeat an established fact.
                                if previous_q < bin_low and next_q >= bin_high:
                                    whole = cutlass.Int32(1)
                                    degen = cutlass.Int32(0)

            # The sentinel bracket also has infinite width. Bypass its
            # collapsed histogram and enter the exact whole-row key-space
            # fallback on rank 0, where fkey(+inf) is the maximum key.
            if okc == cutlass.Int32(0):
                whole = cutlass.Int32(0)
                degen = cutlass.Int32(1)
            if cutlass.const_expr(not MERGE4):
                if cutlass.const_expr(NB__regclus < self.blk):
                    if tid < cutlass.Int32(NB__regclus):
                        s_mrg[tid] = s_mrg[tid] + s_hoff[tid]
                else:
                    for z in cutlass.range_constexpr(NB__regclus // self.blk):
                        i = tid + cutlass.Int32(z * self.blk)
                        s_mrg[i] = s_mrg[i] + s_hoff[i]  # global cursor
                cute.arch.barrier()  # cursors published

            # ---- P7: register sweep emit (!degen)
            if degen == cutlass.Int32(0):
                LOQ = cutlass.Float32(Bv)
                lim1 = above
                if whole == cutlass.Int32(1):
                    lim1 = above + m
                    if cutlass.const_expr(self.dtype == cutlass.BFloat16):
                        if lim1 > k:
                            lim1 = k
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    for s in cutlass.range_constexpr(S):
                        qv = _fmaf__regclus(_val__regclus(frags, s), SC, CQ)  # bit-identical
                        if qv >= LOQ:
                            bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                            p = atomic_add_cta(s_mrg.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                            idx = (
                                (base4 + tid + cutlass.Int32((s // 4) * self.blk))
                                << cutlass.Int32(2)
                            ) + cutlass.Int32(s % 4)
                            if p < lim1:
                                out_row[p] = idx
                            else:
                                if whole == cutlass.Int32(0):
                                    # crossing overflow -> striped DSMEM slabs; TWO
                                    # separate u32 remote stores (NOT packed)
                                    q2i = p - above
                                    rnk = q2i >> cutlass.Int32(LCMPC)
                                    j = (q2i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)
                                    _st_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(ck_addr + j, rnk),
                                        fkey(_val__regclus(frags, s)),
                                    )
                                    _st_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(ci_addr + j, rnk), idx
                                    )
                else:
                    for s in cutlass.range_constexpr(S):
                        if cutlass.const_expr(VPT >= 2):
                            xv = _val8_pk__regclus(frags, s)
                        else:
                            xv = _val8__regclus(frags, s)
                        qv = _fmaf__regclus(xv, SC, CQ)  # bit-identical
                        if qv >= LOQ:
                            bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                            p = atomic_add_cta(s_mrg.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                            idx = (
                                (base4 + tid + cutlass.Int32((s // 8) * self.blk))
                                << cutlass.Int32(3)
                            ) + cutlass.Int32(s % 8)
                            if p < lim1:
                                out_row[p] = idx
                            else:
                                if whole == cutlass.Int32(0):
                                    # BF16 keys retain their exact monotone order in
                                    # their upper 16 bits; refine omits empty low bits.
                                    q2i = p - above
                                    rnk = q2i >> cutlass.Int32(LCMPC)
                                    j = (q2i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)
                                    _st_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(ck_addr + j, rnk),
                                        fkey(xv) >> cutlass.Uint32(16),
                                    )
                                    _st_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(ci_addr + j, rnk), idx
                                    )
                # tail element: tval == -INF fails q>=LOQ elsewhere
                qv = _fmaf__regclus(tval, SC, CQ)
                if qv >= LOQ:
                    bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                    p = atomic_add_cta(s_mrg.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                    if p < lim1:
                        out_row[p] = tix
                    else:
                        if whole == cutlass.Int32(0):
                            q2i = p - above
                            rnk = q2i >> cutlass.Int32(LCMPC)
                            j = (q2i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)
                            tail_key = fkey(tval)
                            if cutlass.const_expr(self.dtype == cutlass.BFloat16):
                                tail_key = tail_key >> cutlass.Uint32(16)
                            _st_shared_cluster_i32(
                                _mapa_shared_cluster_addr(ck_addr + j, rnk), tail_key
                            )
                            _st_shared_cluster_i32(_mapa_shared_cluster_addr(ci_addr + j, rnk), tix)

            # ---- P8: release staging to rank 0
            cute.arch.barrier()
            _cluster_sync_aligned()

            # ---- P9: rank-0 selection
            if rank == cutlass.Int32(0):
                if whole == cutlass.Int32(0):
                    mc = m
                    if degen == cutlass.Int32(1):
                        mc = cutlass.Int32(0)
                    if degen == cutlass.Int32(0):
                        if mc <= cutlass.Int32(QUADC__regclus):
                            # (1) quad-96: all candidates LOCAL (96 < CMPC),
                            # O(mc^2) slot-order tie-broken rank
                            i = tid
                            while i < mc:
                                uq = cutlass.Uint32(s_ck[i])
                                rnk = cutlass.Int32(0)
                                j = cutlass.Int32(0)
                                while j < mc:
                                    vq = cutlass.Uint32(s_ck[j])
                                    tinc = cutlass.Int32(0)
                                    if vq > uq:
                                        tinc = cutlass.Int32(1)
                                    if vq == uq:
                                        if j < i:
                                            tinc = cutlass.Int32(1)
                                    rnk = rnk + tinc
                                    j = j + cutlass.Int32(1)
                                if rnk < need:
                                    out_row[above + rnk] = s_ci[i]
                                i = i + cutlass.Int32(BLK)
                        else:
                            # (2) key-space narrowing over striped DSMEM slabs:
                            # slot = i & (CMPC-1), rank = i >> LCMPC
                            if tid == cutlass.Int32(0):
                                s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                                s_kmm[1] = cutlass.Uint32(0)
                            cute.arch.barrier()  # kmm init
                            i = tid
                            while i < mc:
                                kv = cutlass.Uint32(
                                    _ld_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(
                                            ck_addr
                                            + ((i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)),
                                            i >> cutlass.Int32(LCMPC),
                                        )
                                    )
                                )
                                atomic_min_cta(s_kmm.iterator, kv)
                                atomic_max_cta(s_kmm.iterator + 1, kv)
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # key range published
                            rlo = cutlass.Uint32(s_kmm[0])
                            rhi = cutlass.Uint32(s_kmm[1])
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            aboveC = cutlass.Int32(0)
                            needC = need
                            mm = mc
                            lev = cutlass.Int32(0)
                            done = cutlass.Int32(0)
                            while done == cutlass.Int32(0):  # <=6 levels
                                if needC == mm:
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                    aboveC = aboveC + mm
                                    needC = cutlass.Int32(0)
                                    done = cutlass.Int32(1)
                                if done == cutlass.Int32(0):
                                    if cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                        done = cutlass.Int32(1)
                                    if lev >= cutlass.Int32(6):
                                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                        done = cutlass.Int32(1)
                                if done == cutlass.Int32(0):
                                    d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                    b2w = cutlass.Int32(32) - clz_i32(
                                        cutlass.Int32(d2 | cutlass.Uint32(1))
                                    )
                                    sh2 = cutlass.Int32(0)
                                    if b2w > cutlass.Int32(LNB):
                                        sh2 = b2w - cutlass.Int32(LNB)
                                    if cutlass.const_expr(NB__regclus < self.blk):
                                        if tid < cutlass.Int32(NB__regclus):
                                            s_hist[tid] = cutlass.Int32(0)
                                    else:
                                        for z in cutlass.range_constexpr(NB__regclus // self.blk):
                                            s_hist[tid + cutlass.Int32(z * self.blk)] = (
                                                cutlass.Int32(0)
                                            )
                                    cute.arch.barrier()  # level clear
                                    i = tid
                                    while i < mc:
                                        unar = cutlass.Uint32(
                                            _ld_shared_cluster_i32(
                                                _mapa_shared_cluster_addr(
                                                    ck_addr
                                                    + (
                                                        (i & cutlass.Int32(CMPC - 1))
                                                        << cutlass.Int32(2)
                                                    ),
                                                    i >> cutlass.Int32(LCMPC),
                                                )
                                            )
                                        )
                                        if cutlass.Uint32(unar) >= cutlass.Uint32(rlo):
                                            if cutlass.Uint32(unar) <= cutlass.Uint32(rhi):
                                                bnn = (
                                                    cutlass.Uint32(unar) - cutlass.Uint32(rlo)
                                                ) >> cutlass.Uint32(sh2)
                                                bnn = _umin_u32__regclus(
                                                    bnn, cutlass.Uint32(NB__regclus - 1)
                                                )
                                                atomic_add_cta(
                                                    s_hist.iterator + cutlass.Int32(bnn),
                                                    cutlass.Int32(1),
                                                )
                                        i = i + cutlass.Int32(BLK)
                                    cute.arch.barrier()  # level hist
                                    find_cross(s_hist, needC, tid, s_res, nb=NB__regclus)
                                    cute.arch.barrier()  # level scan
                                    aboveC = aboveC + s_res[RES_ABOVE]
                                    needC = needC - s_res[RES_ABOVE]
                                    mm = s_res[RES_M]
                                    b_lv = s_res[RES_B]
                                    nlo = cutlass.Uint32(rlo) + (
                                        cutlass.Uint32(b_lv) << cutlass.Uint32(sh2)
                                    )
                                    if b_lv != cutlass.Int32(NB__regclus - 1):
                                        rhi = nlo + (
                                            (cutlass.Uint32(1) << cutlass.Uint32(sh2))
                                            - cutlass.Uint32(1)
                                        )
                                    rlo = nlo
                                    lev = lev + cutlass.Int32(1)
                            cute.arch.barrier()  # narrowing done
                            # two-predicate ballot emit over the striped slabs
                            lml = cutlass.Int32(cute.arch.lanemask_lt())
                            it2 = (mc + cutlass.Int32(self.blk - 1)) // cutlass.Int32(self.blk)
                            it = cutlass.Int32(0)
                            while it < it2:
                                i = it * cutlass.Int32(BLK) + tid
                                uke = cutlass.Uint32(0)
                                idv = cutlass.Int32(0)
                                if i < mc:  # predicated remote
                                    uke = cutlass.Uint32(
                                        _ld_shared_cluster_i32(
                                            _mapa_shared_cluster_addr(
                                                ck_addr
                                                + (
                                                    (i & cutlass.Int32(CMPC - 1))
                                                    << cutlass.Int32(2)
                                                ),
                                                i >> cutlass.Int32(LCMPC),
                                            )
                                        )
                                    )
                                    idv = _ld_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(
                                            ci_addr
                                            + ((i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)),
                                            i >> cutlass.Int32(LCMPC),
                                        )
                                    )
                                q1f = cutlass.Int32(0)
                                q2f = cutlass.Int32(0)
                                if i < mc:
                                    if cutlass.Int64(cutlass.Uint32(uke)) > ethr:
                                        q1f = cutlass.Int32(1)
                                    if cutlass.Int64(cutlass.Uint32(uke)) == ethr:
                                        q2f = cutlass.Int32(1)
                                n1 = ballot(q1f == cutlass.Int32(1))
                                n2 = ballot(q2f == cutlass.Int32(1))
                                b1 = cutlass.Int32(0)
                                b2 = cutlass.Int32(0)
                                if lane == cutlass.Int32(0):
                                    if n1 != cutlass.Int32(0):
                                        b1 = atomic_add_cta(s_cnt.iterator, popc(n1))
                                    if n2 != cutlass.Int32(0):
                                        b2 = atomic_add_cta(s_cnt.iterator + 1, popc(n2))
                                b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                                b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                                p1e = b1 + popc(n1 & lml)
                                p2e = b2 + popc(n2 & lml)
                                if q1f == cutlass.Int32(1):
                                    if p1e < aboveC:
                                        out_row[above + p1e] = idv
                                if q2f == cutlass.Int32(1):
                                    if p2e < needC:
                                        out_row[above + aboveC + p2e] = idv
                                it = it + cutlass.Int32(1)
                    else:
                        # (3) degen safety net: crossing bin larger than the
                        # whole cluster buffer -> exact whole-row key-space
                        # narrowing by rank 0 alone, <=8 levels.
                        rlo = cutlass.Uint32(0)
                        rhi = cutlass.Uint32(0xFFFFFFFF)
                        if cutlass.const_expr(self.dtype == cutlass.BFloat16):
                            rhi = cutlass.Uint32(0xFFFF)
                        aboveC = cutlass.Int32(0)  # above2
                        needC = k  # need2
                        mm = n  # m2
                        ethr = cutlass.Int64(0)
                        tie_m = cutlass.Int32(1)
                        lev = cutlass.Int32(0)
                        done = cutlass.Int32(0)
                        while done == cutlass.Int32(0):
                            if needC == mm:
                                ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                aboveC = aboveC + mm
                                needC = cutlass.Int32(0)
                                tie_m = cutlass.Int32(0)
                                done = cutlass.Int32(1)
                            if done == cutlass.Int32(0):
                                if cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    done = cutlass.Int32(1)
                                if lev >= cutlass.Int32(8):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    done = cutlass.Int32(1)
                            if done == cutlass.Int32(0):
                                d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                b2w = cutlass.Int32(32) - clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1))
                                )
                                sh2 = cutlass.Int32(0)
                                if b2w > cutlass.Int32(LNB):
                                    sh2 = b2w - cutlass.Int32(LNB)
                                if cutlass.const_expr(NB__regclus < self.blk):
                                    if tid < cutlass.Int32(NB__regclus):
                                        s_hist[tid] = cutlass.Int32(0)
                                else:
                                    for z in cutlass.range_constexpr(NB__regclus // self.blk):
                                        s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                                cute.arch.barrier()  # level clear
                                i = tid
                                while i < n:  # whole-row bin
                                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                                        unar = fkey(ldg_f32(x_addr, i))
                                    else:
                                        unar = fkey(ldg_bf16(x_addr, i)) >> cutlass.Uint32(16)
                                    if cutlass.Uint32(unar) >= cutlass.Uint32(rlo):
                                        if cutlass.Uint32(unar) <= cutlass.Uint32(rhi):
                                            bnn = (
                                                cutlass.Uint32(unar) - cutlass.Uint32(rlo)
                                            ) >> cutlass.Uint32(sh2)
                                            bnn = _umin_u32__regclus(
                                                bnn, cutlass.Uint32(NB__regclus - 1)
                                            )
                                            atomic_add_cta(
                                                s_hist.iterator + cutlass.Int32(bnn),
                                                cutlass.Int32(1),
                                            )
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()  # level hist
                                find_cross(s_hist, needC, tid, s_res, nb=NB__regclus)
                                cute.arch.barrier()  # level scan
                                aboveC = aboveC + s_res[RES_ABOVE]
                                needC = needC - s_res[RES_ABOVE]
                                mm = s_res[RES_M]
                                b_lv = s_res[RES_B]
                                nlo = cutlass.Uint32(rlo) + (
                                    cutlass.Uint32(b_lv) << cutlass.Uint32(sh2)
                                )
                                if b_lv != cutlass.Int32(NB__regclus - 1):
                                    rhi = nlo + (
                                        (cutlass.Uint32(1) << cutlass.Uint32(sh2))
                                        - cutlass.Uint32(1)
                                    )
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        cute.arch.barrier()  # narrowing done
                        nA = k  # tie_m ? above2 : k
                        if tie_m == cutlass.Int32(1):
                            nA = aboveC
                        nT = cutlass.Int32(0)
                        if tie_m == cutlass.Int32(1):
                            nT = needC
                        lml = cutlass.Int32(cute.arch.lanemask_lt())
                        it2 = (n + cutlass.Int32(self.blk - 1)) // cutlass.Int32(self.blk)
                        it = cutlass.Int32(0)
                        while it < it2:
                            i = it * cutlass.Int32(BLK) + tid
                            uke = cutlass.Uint32(0)
                            if i < n:
                                if cutlass.const_expr(self.dtype == cutlass.Float32):
                                    uke = fkey(ldg_f32(x_addr, i))
                                else:
                                    uke = fkey(ldg_bf16(x_addr, i)) >> cutlass.Uint32(16)
                            q1f = cutlass.Int32(0)
                            q2f = cutlass.Int32(0)
                            if i < n:
                                if cutlass.Int64(cutlass.Uint32(uke)) > ethr:
                                    q1f = cutlass.Int32(1)
                                if tie_m == cutlass.Int32(1):
                                    if cutlass.Int64(cutlass.Uint32(uke)) == ethr:
                                        q2f = cutlass.Int32(1)
                            n1 = ballot(q1f == cutlass.Int32(1))
                            n2 = ballot(q2f == cutlass.Int32(1))
                            b1 = cutlass.Int32(0)
                            b2 = cutlass.Int32(0)
                            if lane == cutlass.Int32(0):
                                if n1 != cutlass.Int32(0):
                                    b1 = atomic_add_cta(s_cnt.iterator, popc(n1))
                                if n2 != cutlass.Int32(0):
                                    b2 = atomic_add_cta(s_cnt.iterator + 1, popc(n2))
                            b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                            b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                            p1e = b1 + popc(n1 & lml)
                            p2e = b2 + popc(n2 & lml)
                            if q1f == cutlass.Int32(1):
                                if p1e < nA:
                                    out_row[p1e] = i
                            if q2f == cutlass.Int32(1):
                                if p2e < nT:
                                    out_row[nA + p2e] = i
                            it = it + cutlass.Int32(1)

            # ---- P10: FINAL cluster rendezvous — ALL ranks reach it;
            # keeps peers resident until rank 0 has read their ck/ci.
            _cluster_sync_aligned()

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        stream,
    ):
        b = out.shape[0]
        self.kern(logits, pre_idx, kv_lens, out, n).launch(
            grid=(self.cs, b, 1),
            block=(self.blk, 1, 1),
            cluster=(self.cs, 1, 1),
            stream=stream,
            smem=SMEM_BYTES,
            min_blocks_per_mp=1,
            use_pdl=self.pdl,
        )


_COMPILE_CACHE__regclus: dict = {}


def get_compiled__regclus(
    tpl: tuple,
    dump_dir: str | None = None,
    pdl: bool = False,
    varlen: bool = False,
    next_n: int = 1,
    cr_shift: int = 0,
    hint_free: bool = False,
    dtype: str = "f32",
    nbh: int = 1024,
    quadc: int = 384,
) -> Any:
    """Compile (or fetch) the variant for constexpr tuple (BLK, VPT, CS)."""
    key = (
        tuple(tpl),
        bool(pdl),
        bool(varlen),
        int(next_n),
        int(cr_shift),
        bool(hint_free),
        str(dtype),
        int(nbh),
        int(quadc),
    )
    compiled = _COMPILE_CACHE__regclus.get(key)
    if compiled is None:
        from cutlass.cute import runtime as _crt

        blk, vpt, cs = tpl
        kernel = GvrRegClusKernel(
            blk,
            vpt,
            cs,
            pdl=pdl,
            varlen=varlen,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=hint_free,
            dtype=_dtype_of(dtype),
        )
        nb_, nc_ = cute.sym_int(), cute.sym_int()
        nb2_, nc2_ = cute.sym_int(), cute.sym_int()
        nb3_, nc3_ = cute.sym_int(), cute.sym_int()
        if dtype == "f32":
            lg_fake = _crt.make_fake_compact_tensor(
                cutlass.Float32, (nb_, nc_), stride_order=(1, 0), assumed_align=16
            )
        else:
            lg_fake = _crt.make_fake_compact_tensor(
                cutlass.BFloat16, (nb_, nc_), stride_order=(1, 0), assumed_align=16
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
        opts = "--enable-tvm-ffi"
        if dump_dir:
            opts += f" --keep-ptx --keep-cubin --dump-dir {dump_dir}"
        assert nbh in (512, 1024)
        assert quadc in (96, 384)
        if dtype == "f32":
            assert nbh == 1024
        global NB__regclus, LNB, DYN_SMEM_BYTES, SMEM_BYTES
        global W_MRG, W_HOFF, W_CK, W_CI, QUADC__regclus
        _cfg_saved = (
            NB__regclus,
            LNB,
            DYN_SMEM_BYTES,
            SMEM_BYTES,
            W_MRG,
            W_HOFF,
            W_CK,
            W_CI,
            QUADC__regclus,
        )
        if dtype != "f32" and nbh == 512:
            NB__regclus = 512
            LNB = 9
            DYN_SMEM_BYTES = (3 * NB__regclus + 2 * CMPC) * 4
            SMEM_BYTES = STATIC_BYTES__regclus + DYN_SMEM_BYTES
            W_MRG = STATIC_WORDS__regclus + NB__regclus
            W_HOFF = STATIC_WORDS__regclus + 2 * NB__regclus
            W_CK = STATIC_WORDS__regclus + 3 * NB__regclus
            W_CI = STATIC_WORDS__regclus + 3 * NB__regclus + CMPC
            QUADC__regclus = quadc
        try:
            compiled = cute.compile(
                kernel,
                lg_fake,
                pi_fake,
                kv_fake,
                out_fake,
                cutlass.Int32(0),
                stream=fake_stream,
                options=opts,
            )
        finally:
            (
                NB__regclus,
                LNB,
                DYN_SMEM_BYTES,
                SMEM_BYTES,
                W_MRG,
                W_HOFF,
                W_CK,
                W_CI,
                QUADC__regclus,
            ) = _cfg_saved
        _COMPILE_CACHE__regclus[key] = compiled
    return compiled


__all__ = [
    "GvrClusKernel",
    "GvrMainKernel",
    "GvrRegClusKernel",
    "GvrTopkRegKernel",
    "get_compiled",
    "get_compiled__clus",
    "get_compiled__reg",
    "get_compiled__regclus",
    "workspace_bytes",
]
