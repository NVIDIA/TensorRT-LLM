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

"""Throughput (tp) GVR Top-K tier — CuTe DSL, Blackwell SM100.

CuTe DSL translation of the CUDA ``gvr_topk_tp<TB,CS,AR,UF>``
throughput GVR top-K kernel (tuned CUDA head),
adapted for the production ``trtllm::cute_dsl_gvr_topk_decode`` contract:

* RAGGED N: production logits tails beyond the per-row valid length are
  stale garbage (NOT -FLT_MAX pad, unlike the standalone bench harness). A device
  ``seq_lens`` tensor is threaded into every tier and every global row read
  is predicated at the LANE level: element index >= N_eff substitutes
  -FLT_MAX AFTER the (always in-bounds, buffer width = logits.shape[1])
  float4 load. N_eff mirrors the in-tree ``run_one_row`` arithmetic:
  ``(seq_lens[req] - next_n + row % next_n + 1) // compress_ratio``.
* pre_idx hardening: the P1 hint gather clamps out-of-range hint indices
  into [0, N_eff-1] (production cold-start is all zeros; arbitrary garbage
  must neither fault nor corrupt the rung ladder).
* Degenerate rows (N_eff <= K): per-row in-kernel identity emit
  [0..N_eff-1] plus -1 padding to K, matching the in-tree kernel.

Faithful phase-by-phase port otherwise:
  P1  : hint gather + minmax + two-stage 64-bin histogram -> CCDF rung ladder
  P2a : uniform every-32nd-float4 sampled multi-rung count (+ per-rung
        float4 occupancy for a clustering-aware sigma) -> 4-stage pivot:
        stage 0 is the hint-ladder ADMISSION (in-tree R0 parity) —
        tightest rung whose sampled CI sits inside [K, 0.6*kC] (the
        legacy pivot-band hi) — stage 0b prefers the legacy band pick
        when it is strictly leaner and safe by its clustering-aware
        1.5-sigma lower CI, and the legacy 3-stage pick handles rows
        where no rung qualifies
  P2b : ONE fused streaming pass: exact counts at {pivot, rescue rung}
        + optimistic collect of packed (key<<32|idx) u64 candidates
        >= pivot into CTA0 smem (capped kC). Pivot count in [K, kC] =>
        the candidates are reused as-is (1-pass admission); pivot under K
        but rescue in-window => ONE collect re-stream
  P2c : multi-rung secant refine / max-below plateau descent when the pivot
        count misses [K, kC]
  P3  : candidate reuse (thr == tpush, no overflow) or one re-stream collect
  P4  : CTA0-solo 4x8-bit radix select + tie-aware ticketed emit
  (plus the trivial npad <= kC path and the plateau direct-emit path)

Templates -> compile-time ctor knobs; compile cache keyed (K, CS, AR, UF, TB).
``__launch_bounds__(TB, 2)`` -> ``.launch(min_blocks_per_mp=2)``.
grid dim3(CS, BS) -> 1-D grid BS*CS with cluster=(CS,1,1); row = bidx // CS.

DSMEM (CS > 1) via inline PTX: mapa.shared::cluster + ld/st.shared::cluster
+ atom.relaxed.cluster.shared::cluster.add.u32. Writer-side cluster syncs use
FULL cluster_arrive (release) — never relaxed (a relaxed arrive has no
release semantics, so peer CTAs could observe stale DSMEM: see
``cluster_arrive_relaxed`` DSMEM race observed in development).
"""

import math

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute import runtime as _crt
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.distributed import atomicAdd
from cutlass.utils.smem_allocator import SmemAllocator

RUNGS = 8
MAXPASS = 8
SS = 32  # P2a sample stride (float4s)

# DATA-ADAPTIVE admission (measured full-grid verdict).
# This is the baseline admission machinery
# BYTE-PARITY on every streaming pass, the fused R=2 count-collect, the
# reuse rule, the P4 select and the P2c driver (a dedicated-window-rung variant
# rungs and a later R=4 window variant both taxed the whole kernel 4-15%
# globally — even trivial-branch cells — via register pressure/extra
# compares/P4 prefilter, and the former additionally carried an adversarial
# under-emit bug). The adaptivity is a PICK-ONLY delta:
#   * stage 0c LEAN-PIVOT OVERRIDE: when the stage-0 CI admission FAILED
#     (the pick in hand is a band/2-sigma/overshoot fallback: fat or
#     undershoot-prone), npad <= 262144, and a strictly LEANER ladder
#     rung's sampled count lands in [K, kC] by its clustering-aware
#     1.5-sigma CI on both sides, that rung becomes the pivot (the
#     rescue is recomputed from it by the standard next-fatter rule).
#     A fat band pick (est ~3-4x K, the P3-push/P4-radix fatness the
#     mechanism probe identified as the whole residual-band gap) is
#     replaced by a lean CI-backed pivot; an undershoot costs ONE rescue
#     re-stream — the baseline's own economics.
#   * kC pinned to the flat budget 8192 (K>=2048) / 6144
#     (the K-scaled diet was falsified as pure harm).
#   * ladder quantiles pinned WIDE (baseline): ablation showed the re-placed
#     spread under this admission is a net residual harm.
#   * the C4 occupancy CS cut keeps the fix-head default (neutral).
FLT_MAX = 3.4028234663852886e38
INF = float("inf")


# ---------------------------------------------------------------------------
# DSMEM primitives (inline PTX)
# ---------------------------------------------------------------------------
@dsl_user_op
def _mapa_shared_cluster(smem_ptr, peer_rank, *, loc=None, ip=None):
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_shared_cluster_i32(mapped_addr, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [mapped_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_shared_cluster_f32(mapped_addr, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [mapped_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_cluster_i32(mapped_addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        res=None,
        operands_=[mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        asm_string="st.shared::cluster.u32 [$0], $1;",
        constraints="r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_cluster_u64(mapped_addr, val, *, loc=None, ip=None):
    """One 8B DSMEM candidate push (CUDA: dst[pos] = (u64)key<<32 | idx).

    Measured: a single packed u64 store halves remote-store
    transactions vs two 4B pushes under CS>1 cluster contention (matches
    the CUDA arm's ``unsigned long long cand[kC]``). Do NOT split back
    into key/idx 4B stores — the split form measurably regresses the
    K-scaled CS>1 band (pro/v32 BS16-64)."""
    llvm.inline_asm(
        res=None,
        operands_=[mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        asm_string="st.shared::cluster.u64 [$0], $1;",
        constraints="r,l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atom_shared_cluster_add_i32(mapped_addr, val, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "atom.relaxed.cluster.shared::cluster.add.u32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _f32_bits_u32(float_val):
    """Raw fp32 bits as Uint32 (bit-cast)."""
    return cutlass.Uint32(llvm.bitcast(cutlass.Uint32.mlir_type, float_val.ir_value()))


def _i32_bits_f32(int_val):
    return cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, int_val.ir_value()))


@dsl_user_op
def _fmin_f32(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lg2_f32(a, *, loc=None, ip=None):
    """lg2.approx.f32 — thread-0 serial pick section only (lean interp)."""
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [a.ir_value(loc=loc, ip=ip)],
            "lg2.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _f2u_bits(u):
    """Order-preserving fp32-bits -> Uint32 key: u ^ (sign ? 0xFFFFFFFF : 0x80000000)."""
    neg = cutlass.Uint32(0) - (u >> cutlass.Uint32(31))  # 0 or 0xFFFFFFFF
    return u ^ (neg | cutlass.Uint32(0x80000000))


@cute.jit
def _exp2i(e):
    """2.0**e for dynamic int e in [0, 127] via exponent-bit construction."""
    bits = (e + cutlass.Int32(127)) << cutlass.Int32(23)
    return _i32_bits_f32(bits)


@cute.jit
def _shfl_up_add(val, lane, offset: cutlass.Constexpr):
    """Inclusive-scan step: val += shfl_up(val, offset) gated by lane >= offset."""
    src = lane - cutlass.Int32(offset)
    if src < 0:
        src = cutlass.Int32(0)
    other = cute.arch.shuffle_sync(val, src)
    if lane >= cutlass.Int32(offset):
        val = val + other
    return val


@cute.jit
def _shfl_down_add(val, lane, offset: cutlass.Constexpr):
    """Suffix-scan step: val += shfl_down(val, offset) gated by lane + offset < 32."""
    src = lane + cutlass.Int32(offset)
    if src > 31:
        src = cutlass.Int32(31)
    other = cute.arch.shuffle_sync(val, src)
    if lane + cutlass.Int32(offset) < cutlass.Int32(32):
        val = val + other
    return val


@cute.jit
def _mask_tail(v, gidx, n_eff):
    """Ragged-N lane predicate: value at global element index ``gidx`` is
    replaced by -FLT_MAX when ``gidx >= n_eff``.

    The float4 LOAD itself is always in-bounds (the logits buffer is
    allocated to npad = logits.shape[1]); only the stale VALUE beyond the
    row's valid length must be masked so garbage can never enter counts,
    max-below, candidate pushes, or emits."""
    r = v
    if gidx >= n_eff:
        r = cutlass.Float32(-FLT_MAX)
    return r


class GvrTpKernel:
    """CuTe DSL port of gvr_topk_tp<TB, CS, AR, UF> (fp32, B200/B300).

    Ctor knobs mirror the CUDA template params plus the production
    ``next_n`` / ``compress_ratio`` contract (compile-time constexpr:
    N_eff arithmetic, request-level hint-row sharing and the cr==1 hint
    temporal shift all mirror the in-tree ``run_one_row`` /
    ``phase1_preidx_stats`` formulas exactly; the next_n==1 / cr==4 hot
    path traces identically to the v1 port).
    """

    WARP_SIZE = 32

    def __init__(
        self,
        top_k: int,
        kC: int,
        cluster_size: int = 1,
        ar: int = RUNGS,
        uf: int = 4,
        num_threads: int = 512,
        next_n: int = 1,
        compress_ratio: int = 4,
    ):
        assert num_threads % 32 == 0
        assert ar in (6, 8)
        assert cluster_size in (1, 2, 4, 8)
        # Lean-pivot interpolation target (per-K: K512 rows carry
        # 5-20x sparser P2a samples, so 2.0K keeps interpolation-error
        # margin; K>=1024 takes the measured-win 1.5K).
        self.lean_tgt = (2 * top_k) if top_k < 1024 else (3 * top_k) // 2
        self.top_k = top_k
        self.kC = kC
        self.cluster_size = cluster_size
        self.ar = ar
        self.uf = uf
        self.num_threads = num_threads
        self.num_warps = num_threads // 32
        self.next_n = next_n
        self.compress_ratio = compress_ratio

    # ------------------------------------------------------------------
    # float4 (128-bit) strided streaming load helper pieces
    # ------------------------------------------------------------------
    def _copy_atom(self):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
        )

    @cute.jit
    def _ld_float4(self, copy_atom, row_addr, v_idx, frag):
        """Load float4 #v_idx of the row into frag[0..3]."""
        p = cute.make_ptr(
            cutlass.Float32,
            row_addr + cutlass.Int64(v_idx) * cutlass.Int64(16),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        src = cute.make_tensor(p, cute.make_layout((4,)))
        cute.copy(copy_atom, src, frag)

    # ------------------------------------------------------------------
    # Per-row valid length (ragged N). Mirrors the in-tree run_one_row
    # arithmetic EXACTLY (gvr_topk_decode.py): seq_lens is request-level
    # and in uncompressed-token space; logits live in compressed space
    # when cr > 1.
    # ------------------------------------------------------------------
    @cute.jit
    def _row_n_eff(self, seq_lens: cute.Tensor, row):
        NN = cutlass.const_expr(self.next_n)
        seq_len = seq_lens[row // cutlass.Int32(NN)]
        actual_kv_len = seq_len - cutlass.Int32(NN) + (row % cutlass.Int32(NN)) + cutlass.Int32(1)
        if cutlass.const_expr(self.compress_ratio == 1):
            n_eff = actual_kv_len
        else:
            n_eff = actual_kv_len // cutlass.Int32(self.compress_ratio)
        return n_eff

    # ------------------------------------------------------------------
    # count_pass<TB, R, U>: R-rung exact count over [v0, v1) float4s,
    # per-thread counts -> s_ptcnt[r*TB + tid].
    # ------------------------------------------------------------------
    @cute.jit
    def count_pass(
        self,
        R: cutlass.Constexpr,
        U: cutlass.Constexpr,
        row_addr,
        v0,
        v1,
        n_eff,
        tidx,
        s_rungs,
        s_ptcnt,
    ):
        # Explicit U-batched loads (CUDA `float4 a[U]` idiom) — a
        # `cutlass.range(unroll=U)` loop leaves ONE load in flight per iter
        # and costs ~11% kernel time in the DRAM-bound regimes. Keep the
        # Python-unrolled register batch.
        TB = cutlass.const_expr(self.num_threads)
        copy_atom = self._copy_atom()
        tr = cute.make_rmem_tensor((R,), cutlass.Float32)
        cnt = cute.make_rmem_tensor((R,), cutlass.Int32)
        for r in cutlass.range_constexpr(R):
            tr[r] = s_rungs[r]
            cnt[r] = cutlass.Int32(0)
        frags = [
            cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(U)
        ]  # Python-unrolled register batch
        # Mask hoist: float4s below vmain = min(v1, n_eff >> 2) are fully
        # valid (4i+3 < n_eff), so the main loop runs mask-free (the
        # per-element gi compare + select was ~10% of whole-kernel
        # instructions); only [vmain, v1) — the n_eff boundary float4 and
        # the pad tail — takes the masked epilogue. Bit-exact: the mask
        # only changes values at gi >= n_eff, all of which live in
        # [vmain, v1).
        vmain = v1
        vfull = n_eff >> cutlass.Int32(2)
        if vmain > vfull:
            vmain = vfull
        i = v0 + tidx
        while i + cutlass.Int32((U - 1) * TB) < vmain:
            for u in cutlass.range_constexpr(U):
                self._ld_float4(copy_atom, row_addr, i + cutlass.Int32(u * TB), frags[u])
            for u in cutlass.range_constexpr(U):
                for q in cutlass.range_constexpr(4):
                    v = frags[u][q]
                    for r in cutlass.range_constexpr(R):
                        cnt[r] = cnt[r] + cutlass.Int32(v >= tr[r])
            i = i + cutlass.Int32(U * TB)
        while i < vmain:
            self._ld_float4(copy_atom, row_addr, i, frags[0])
            for q in cutlass.range_constexpr(4):
                v = frags[0][q]
                for r in cutlass.range_constexpr(R):
                    cnt[r] = cnt[r] + cutlass.Int32(v >= tr[r])
            i = i + cutlass.Int32(TB)
        while i < v1:
            self._ld_float4(copy_atom, row_addr, i, frags[0])
            gi = i << cutlass.Int32(2)
            for q in cutlass.range_constexpr(4):
                v = _mask_tail(frags[0][q], gi + cutlass.Int32(q), n_eff)
                for r in cutlass.range_constexpr(R):
                    cnt[r] = cnt[r] + cutlass.Int32(v >= tr[r])
            i = i + cutlass.Int32(TB)
        for r in cutlass.range_constexpr(R):
            s_ptcnt[r * TB + tidx] = cnt[r]

    # ------------------------------------------------------------------
    # sample_count<TB, R, SS>: uniform every-SS-th float4 over the slice.
    # Each per-thread accumulator PACKS the per-rung hit count (low 16
    # bits) with the per-rung float4 OCCUPANCY (high 16 bits) — the number
    # of sampled float4s with at least one hit. On spatially-clustered
    # real rows the 4 values of one float4 are strongly correlated, so the
    # effective independent sample count is the occupancy, not the hit
    # count: the admission stage sizes its confidence interval with the
    # compound-Poisson sigma cnt/sqrt(occ) (equal to the classic
    # sqrt(cnt) Poisson sigma on IID data where occ == cnt). Packing
    # keeps registers, SMEM and the exchange at their pre-admission
    # sizes; no field overflow: the tier envelope pins npad <= 262144, so
    # cluster-total cnt <= npad/32 = 8192 < 2^16 and occ <= npad/128.
    # ------------------------------------------------------------------
    @cute.jit
    def sample_count(self, R: cutlass.Constexpr, row_addr, v0, v1, n_eff, tidx, s_rungs, s_ptcnt):
        TB = cutlass.const_expr(self.num_threads)
        copy_atom = self._copy_atom()
        tr = cute.make_rmem_tensor((R,), cutlass.Float32)
        cnt = cute.make_rmem_tensor((R,), cutlass.Int32)
        for r in cutlass.range_constexpr(R):
            tr[r] = s_rungs[r]
            cnt[r] = cutlass.Int32(0)
        frag = cute.make_rmem_tensor((4,), cutlass.Float32)
        j = cutlass.Int32(tidx)
        while v0 + j * cutlass.Int32(SS) < v1:
            self._ld_float4(copy_atom, row_addr, v0 + j * cutlass.Int32(SS), frag)
            gi = (v0 + j * cutlass.Int32(SS)) << cutlass.Int32(2)
            v0m = _mask_tail(frag[0], gi, n_eff)
            v1m = _mask_tail(frag[1], gi + cutlass.Int32(1), n_eff)
            v2m = _mask_tail(frag[2], gi + cutlass.Int32(2), n_eff)
            v3m = _mask_tail(frag[3], gi + cutlass.Int32(3), n_eff)
            for r in cutlass.range_constexpr(R):
                c4 = (
                    cutlass.Int32(v0m >= tr[r])
                    + cutlass.Int32(v1m >= tr[r])
                    + cutlass.Int32(v2m >= tr[r])
                    + cutlass.Int32(v3m >= tr[r])
                )
                cnt[r] = cnt[r] + c4 + (cutlass.Int32(c4 > cutlass.Int32(0)) << cutlass.Int32(16))
            j = j + cutlass.Int32(TB)
        for r in cutlass.range_constexpr(R):
            s_ptcnt[r * TB + tidx] = cnt[r]

    # ------------------------------------------------------------------
    # exchange_counts<TB, CS, R>: warp-fused CTA reduce -> cluster sum + prefix.
    # rcnt[r] = cluster-wide count, rpre[r] = exclusive prefix over lower ranks.
    # P2a rows carry packed (occ << 16 | cnt) accumulators; the integer sum
    # distributes over the packed fields (no overflow: see sample_count).
    # ------------------------------------------------------------------
    @cute.jit
    def exchange_counts(
        self, R: cutlass.Constexpr, par, tidx, rank, s_ptcnt, s_rcnt, s_rpre, s_ipartial
    ):
        TB = cutlass.const_expr(self.num_threads)
        CS = cutlass.const_expr(self.cluster_size)
        cute.arch.barrier()  # ptcnt final
        lane = tidx & cutlass.Int32(31)
        wid = tidx >> cutlass.Int32(5)
        if wid < cutlass.Int32(R):
            s = cutlass.Int32(0)
            for k in cutlass.range_constexpr(TB // 32):
                s = s + s_ptcnt[wid * TB + lane + cutlass.Int32(32 * k)]
            s = cute.arch.warp_redux_sync(s, "add")
            if cutlass.const_expr(CS == 1):
                if lane == cutlass.Int32(0):
                    s_rcnt[wid] = s
                    s_rpre[wid] = cutlass.Int32(0)
            else:
                if lane == cutlass.Int32(0):
                    s_ipartial[par * cutlass.Int32(RUNGS) + wid] = s
        if cutlass.const_expr(CS == 1):
            cute.arch.barrier()
        else:
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if tidx < cutlass.Int32(R):
                tot = cutlass.Int32(0)
                pre = cutlass.Int32(0)
                local_ptr = s_ipartial.iterator + (par * cutlass.Int32(RUNGS) + tidx)
                for rr in cutlass.range_constexpr(CS):
                    a = _mapa_shared_cluster(local_ptr, cutlass.Int32(rr))
                    v = _ld_shared_cluster_i32(a)
                    tot = tot + v
                    if cutlass.Int32(rr) < rank:
                        pre = pre + v
                s_rcnt[tidx] = tot
                s_rpre[tidx] = pre
            cute.arch.barrier()

    # ------------------------------------------------------------------
    # max_below_pass<TB, CS>: largest value strictly below t_hi_bound (cluster-reduced).
    # ------------------------------------------------------------------
    @cute.jit
    def max_below_pass(self, row_addr, v0, v1, n_eff, t_hi_bound, par, tidx, s_fwred, s_fpartial):
        TB = cutlass.const_expr(self.num_threads)
        CS = cutlass.const_expr(self.cluster_size)
        NWARP = cutlass.const_expr(self.num_warps)
        copy_atom = self._copy_atom()
        m = cutlass.Float32(-FLT_MAX)
        # Explicit U=4 batched loads (CUDA `float4 a[4]` idiom).
        frags = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(4)]
        vmain = v1
        vfull = n_eff >> cutlass.Int32(2)
        if vmain > vfull:
            vmain = vfull
        i = v0 + tidx
        while i + cutlass.Int32(3 * TB) < vmain:
            for u in cutlass.range_constexpr(4):
                self._ld_float4(copy_atom, row_addr, i + cutlass.Int32(u * TB), frags[u])
            for u in cutlass.range_constexpr(4):
                for q in cutlass.range_constexpr(4):
                    v = frags[u][q]
                    if v < t_hi_bound:
                        m = cute.arch.fmax(m, v)
            i = i + cutlass.Int32(4 * TB)
        while i < vmain:
            self._ld_float4(copy_atom, row_addr, i, frags[0])
            for q in cutlass.range_constexpr(4):
                v = frags[0][q]
                if v < t_hi_bound:
                    m = cute.arch.fmax(m, v)
            i = i + cutlass.Int32(TB)
        while i < v1:
            self._ld_float4(copy_atom, row_addr, i, frags[0])
            gi = i << cutlass.Int32(2)
            for q in cutlass.range_constexpr(4):
                v = _mask_tail(frags[0][q], gi + cutlass.Int32(q), n_eff)
                if v < t_hi_bound:
                    m = cute.arch.fmax(m, v)
            i = i + cutlass.Int32(TB)
        m = cute.arch.warp_redux_sync(m, "fmax")
        lane = tidx & cutlass.Int32(31)
        wid = tidx >> cutlass.Int32(5)
        if lane == cutlass.Int32(0):
            s_fwred[wid] = m
        cute.arch.barrier()
        if tidx == cutlass.Int32(0):
            mm = cutlass.Float32(-FLT_MAX)
            for w in cutlass.range_constexpr(NWARP):
                mm = cute.arch.fmax(mm, s_fwred[w])
            s_fpartial[par] = mm
        res = cutlass.Float32(-FLT_MAX)
        if cutlass.const_expr(CS == 1):
            cute.arch.barrier()
            res = s_fpartial[par]
        else:
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            local_ptr = s_fpartial.iterator + par
            for rr in cutlass.range_constexpr(CS):
                a = _mapa_shared_cluster(local_ptr, cutlass.Int32(rr))
                res = cute.arch.fmax(res, _ld_shared_cluster_f32(a))
        return res

    # ------------------------------------------------------------------
    # phase1<TB, AR>: hint gather + stats + rung ladder from hint-value CCDF.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1(
        self,
        logits_row,
        pre_idx_row,
        n_eff,
        tidx,
        s_hist,
        s_fwred,
        s_hminmax,
        s_rungs,
        hint_off=None,  # cr==1 temporal shift ((row % next_n) + 1); None => 0
    ):
        TB = cutlass.const_expr(self.num_threads)
        AR = cutlass.const_expr(self.ar)
        NWARP = cutlass.const_expr(self.num_warps)
        K = cutlass.const_expr(self.top_k)
        if tidx < cutlass.Int32(64):
            s_hist[tidx] = cutlass.Int32(0)
        hv = cute.make_rmem_tensor((4,), cutlass.Float32)
        hok = cute.make_rmem_tensor((4,), cutlass.Int32)
        mn = cutlass.Float32(FLT_MAX)
        mx = cutlass.Float32(-FLT_MAX)
        for jj in cutlass.range_constexpr(4):
            j = tidx + cutlass.Int32(jj * TB)
            hok[jj] = cutlass.Int32(0)
            hv[jj] = cutlass.Float32(0.0)
            if j < cutlass.Int32(K):
                # pre_idx hardening: clamp hints into [0, N_eff-1] (not
                # [0, npad-1] as in the standalone bench port). Production
                # cold-start is all zeros and arbitrary garbage must not
                # crash or corrupt; clamping (rather than skipping, which
                # the in-tree phase1_preidx_stats does) also keeps the CCDF
                # ladder seeded with real in-range values so an all-OOR
                # hint cannot collapse the ladder onto FLT_MAX and force a
                # pathological one-value-per-pass plateau descent.
                hidx = pre_idx_row[j]
                # cr==1 temporal shift, mirroring the in-tree run_one_row
                # pre_idx_offset ((row % next_n) + 1); shifted-out-of-range
                # hints fall into the same clamp below. const_expr'd out on
                # cr>1 builds (hint_off is None => identical trace).
                if cutlass.const_expr(hint_off is not None):
                    hidx = hidx + hint_off
                if hidx < cutlass.Int32(0):
                    hidx = cutlass.Int32(0)
                if hidx > n_eff - cutlass.Int32(1):
                    hidx = n_eff - cutlass.Int32(1)
                v = logits_row[hidx]
                hv[jj] = v
                hok[jj] = cutlass.Int32(1)
                mn = _fmin_f32(mn, v)
                mx = cute.arch.fmax(mx, v)
        mn = cute.arch.warp_redux_sync(mn, "fmin")
        mx = cute.arch.warp_redux_sync(mx, "fmax")
        lane = tidx & cutlass.Int32(31)
        wid = tidx >> cutlass.Int32(5)
        if lane == cutlass.Int32(0):
            s_fwred[wid] = mn
            s_fwred[cutlass.Int32(NWARP) + wid] = mx
        cute.arch.barrier()
        if tidx == cutlass.Int32(0):
            a = cutlass.Float32(FLT_MAX)
            b = cutlass.Float32(-FLT_MAX)
            for w in cutlass.range_constexpr(NWARP):
                a = _fmin_f32(a, s_fwred[w])
                b = cute.arch.fmax(b, s_fwred[NWARP + w])
            s_hminmax[0] = a
            s_hminmax[1] = b
        cute.arch.barrier()
        hmin = s_hminmax[0]
        hmax = s_hminmax[1]
        if hmax - hmin > cutlass.Float32(0.0):
            # Stage 1: coarse 64-bin hist over [hmin, hmax] -> 97% trim point.
            scale = cutlass.Float32(64.0) / (hmax - hmin)
            for jj in cutlass.range_constexpr(4):
                if hok[jj] != cutlass.Int32(0):
                    b1 = cutlass.Int32((hv[jj] - hmin) * scale)
                    if b1 < cutlass.Int32(0):
                        b1 = cutlass.Int32(0)
                    if b1 > cutlass.Int32(63):
                        b1 = cutlass.Int32(63)
                    atomicAdd(s_hist.iterator + b1, cutlass.Int32(1))
            cute.arch.barrier()
            if tidx < cutlass.Int32(32):
                h0 = s_hist[cutlass.Int32(62) - cutlass.Int32(2) * tidx]
                h1 = s_hist[cutlass.Int32(63) - cutlass.Int32(2) * tidx]
                Ssum = h0 + h1
                x = Ssum
                for o in [1, 2, 4, 8, 16]:
                    x = _shfl_up_add(x, tidx, o)
                A = x - Ssum
                binw = (hmax - hmin) * cutlass.Float32(1.0 / 64.0)
                qtrim = cutlass.const_expr((K * 97) // 100)
                if A < cutlass.Int32(qtrim):
                    if x >= cutlass.Int32(qtrim):
                        b2 = cutlass.Int32(62) - cutlass.Int32(2) * tidx
                        if A + h1 >= cutlass.Int32(qtrim):
                            b2 = cutlass.Int32(63) - cutlass.Int32(2) * tidx
                        s_hminmax[0] = hmin + binw * cutlass.Float32(b2)
                if tidx == cutlass.Int32(31):
                    if x < cutlass.Int32(qtrim):
                        s_hminmax[0] = hmin
                if tidx == cutlass.Int32(0):
                    s_rungs[AR - 1] = hmin
                # barrier fold: re-zero both hist halves for stage 2
                s_hist[tidx] = cutlass.Int32(0)
                s_hist[tidx + cutlass.Int32(32)] = cutlass.Int32(0)
            cute.arch.barrier()
            tlow = s_hminmax[0]
            if hmax - tlow > cutlass.Float32(0.0):
                # Stage 2: fine 64-bin hist over [tlow, hmax] -> rung quantiles.
                scale2 = cutlass.Float32(64.0) / (hmax - tlow)
                for jj in cutlass.range_constexpr(4):
                    if hok[jj] != cutlass.Int32(0):
                        if hv[jj] >= tlow:
                            b3 = cutlass.Int32((hv[jj] - tlow) * scale2)
                            if b3 < cutlass.Int32(0):
                                b3 = cutlass.Int32(0)
                            if b3 > cutlass.Int32(63):
                                b3 = cutlass.Int32(63)
                            atomicAdd(s_hist.iterator + b3, cutlass.Int32(1))
                cute.arch.barrier()
                if tidx < cutlass.Int32(32):
                    h0 = s_hist[cutlass.Int32(62) - cutlass.Int32(2) * tidx]
                    h1 = s_hist[cutlass.Int32(63) - cutlass.Int32(2) * tidx]
                    Ssum = h0 + h1
                    x = Ssum
                    for o in [1, 2, 4, 8, 16]:
                        x = _shfl_up_add(x, tidx, o)
                    A = x - Ssum
                    binw2 = (hmax - tlow) * cutlass.Float32(1.0 / 64.0)
                    # WIDE (baseline) quantile spread, pinned — the
                    # P2A path must stay baseline-parity (ablation: the re-placed
                    # spread under this admission is a net residual harm).
                    if cutlass.const_expr(AR == 6):
                        qt = (
                            (K * 15) // 100,
                            (K * 40) // 100,
                            (K * 70) // 100,
                            (K * 92) // 100,
                        )
                    else:
                        qt = (
                            (K * 10) // 100,
                            (K * 25) // 100,
                            (K * 45) // 100,
                            (K * 65) // 100,
                            (K * 82) // 100,
                            (K * 94) // 100,
                        )
                    tot = cute.arch.shuffle_sync(x, cutlass.Int32(31))
                    for r in cutlass.range_constexpr(AR - 2):
                        qtr = cutlass.const_expr(qt[r])
                        if A < cutlass.Int32(qtr):
                            if x >= cutlass.Int32(qtr):
                                b4 = cutlass.Int32(62) - cutlass.Int32(2) * tidx
                                if A + h1 >= cutlass.Int32(qtr):
                                    b4 = cutlass.Int32(63) - cutlass.Int32(2) * tidx
                                s_rungs[r + 1] = tlow + binw2 * cutlass.Float32(b4)
                        if tidx == cutlass.Int32(31):
                            if tot < cutlass.Int32(qtr):
                                s_rungs[r + 1] = tlow
                    if tidx == cutlass.Int32(0):
                        s_rungs[0] = hmax + (hmax - tlow)
                cute.arch.barrier()
            else:
                # degenerate trim: all rungs above the floor = hmax
                if tidx < cutlass.Int32(AR - 1):
                    s_rungs[tidx] = hmax
                cute.arch.barrier()
        else:
            # degenerate: all hint values equal
            if tidx < cutlass.Int32(AR):
                s_rungs[tidx] = hmin
            cute.arch.barrier()

    # ------------------------------------------------------------------
    # fused_count_collect<TB, 2, UF>: exact counts at {rungs[0]=pivot,
    # rungs[1]=rescue} + push (key,idx) >= tpush into CTA0 cand, capped
    # kcap. The rescue rung (next fatter ladder rung) replaces HEAD's hmin
    # column at identical register cost: on a pivot undershoot the driver
    # accepts the rescue with ONE collect re-stream instead of the
    # multi-pass secant loop (hmin was a bracket-only data point; the
    # secant fallback re-derives its bracket from the rescue instead).
    # Keys are RAW fp32 bits (f2u happens in P4 round 0), matching CUDA.
    # Candidates are a single packed (key<<32 | idx) u64 array — CUDA's
    # `unsigned long long cand[kC]` — one 8B push per candidate.
    # ------------------------------------------------------------------
    @cute.jit
    def _push_cand(
        self,
        a_cnt,
        a_st,
        s_cand,
        s_isc,
        kcap: cutlass.Constexpr,
        capped: cutlass.Constexpr,
        v,
        gidx,
    ):
        """One candidate push: kv = (raw fp32 bits << 32) | index, single 8B
        store (st.shared::cluster.u64 remote at CS>1, local u64 store at CS1).
        a_cnt/a_st are the pre-mapa'd CTA0 addresses of cnt_c / cand[0]."""
        CS = cutlass.const_expr(self.cluster_size)
        kv = (cutlass.Uint64(_f32_bits_u32(v)) << cutlass.Uint64(32)) | cutlass.Uint64(
            cutlass.Uint32(gidx)
        )
        if cutlass.const_expr(CS > 1):
            p = _atom_shared_cluster_add_i32(a_cnt, cutlass.Int32(1))
            if cutlass.const_expr(capped):
                if p < cutlass.Int32(kcap):
                    _st_shared_cluster_u64(a_st + p * cutlass.Int32(8), kv)
            else:
                _st_shared_cluster_u64(a_st + p * cutlass.Int32(8), kv)
        else:
            p = atomicAdd(s_isc.iterator + cutlass.Int32(5), cutlass.Int32(1))
            if cutlass.const_expr(capped):
                if p < cutlass.Int32(kcap):
                    s_cand[p] = kv
            else:
                s_cand[p] = kv

    @cute.jit
    def fused_count_collect(
        self,
        U: cutlass.Constexpr,
        row_addr,
        v0,
        v1,
        n_eff,
        tpush,
        tidx,
        s_rungs,
        s_ptcnt,
        s_cand,
        s_isc,
    ):
        TB = cutlass.const_expr(self.num_threads)
        CS = cutlass.const_expr(self.cluster_size)
        kcap = cutlass.const_expr(self.kC)
        R = cutlass.const_expr(2)
        copy_atom = self._copy_atom()
        tr = cute.make_rmem_tensor((R,), cutlass.Float32)
        cnt = cute.make_rmem_tensor((R,), cutlass.Int32)
        for r in cutlass.range_constexpr(R):
            tr[r] = s_rungs[r]
            cnt[r] = cutlass.Int32(0)
        a_cnt = cutlass.Int32(0)
        a_st = cutlass.Int32(0)
        if cutlass.const_expr(CS > 1):
            a_cnt = _mapa_shared_cluster(s_isc.iterator + cutlass.Int32(5), cutlass.Int32(0))
            a_st = _mapa_shared_cluster(s_cand.iterator, cutlass.Int32(0))
        # Explicit U-batched loads (CUDA `float4 a[U]` idiom), main +
        # vec-tail while-loops (same fix as count_pass). Mask
        # hoist as in count_pass: [v0, vmain) mask-free (gi computed only
        # inside the rare push branch), [vmain, v1) masked epilogue.
        frags = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(U)]
        vmain = v1
        vfull = n_eff >> cutlass.Int32(2)
        if vmain > vfull:
            vmain = vfull
        i = v0 + tidx
        while i + cutlass.Int32((U - 1) * TB) < vmain:
            for u in cutlass.range_constexpr(U):
                self._ld_float4(copy_atom, row_addr, i + cutlass.Int32(u * TB), frags[u])
            for u in cutlass.range_constexpr(U):
                for q in cutlass.range_constexpr(4):
                    v = frags[u][q]
                    for r in cutlass.range_constexpr(R):
                        cnt[r] = cnt[r] + cutlass.Int32(v >= tr[r])
                    if v >= tpush:
                        self._push_cand(
                            a_cnt,
                            a_st,
                            s_cand,
                            s_isc,
                            kcap,
                            True,
                            v,
                            ((i + cutlass.Int32(u * TB)) << cutlass.Int32(2)) + cutlass.Int32(q),
                        )
            i = i + cutlass.Int32(U * TB)
        while i < vmain:
            self._ld_float4(copy_atom, row_addr, i, frags[0])
            for q in cutlass.range_constexpr(4):
                v = frags[0][q]
                for r in cutlass.range_constexpr(R):
                    cnt[r] = cnt[r] + cutlass.Int32(v >= tr[r])
                if v >= tpush:
                    self._push_cand(
                        a_cnt,
                        a_st,
                        s_cand,
                        s_isc,
                        kcap,
                        True,
                        v,
                        (i << cutlass.Int32(2)) + cutlass.Int32(q),
                    )
            i = i + cutlass.Int32(TB)
        while i < v1:
            self._ld_float4(copy_atom, row_addr, i, frags[0])
            gi = i << cutlass.Int32(2)
            for q in cutlass.range_constexpr(4):
                v = _mask_tail(frags[0][q], gi + cutlass.Int32(q), n_eff)
                for r in cutlass.range_constexpr(R):
                    cnt[r] = cnt[r] + cutlass.Int32(v >= tr[r])
                if v >= tpush:
                    self._push_cand(
                        a_cnt, a_st, s_cand, s_isc, kcap, True, v, gi + cutlass.Int32(q)
                    )
            i = i + cutlass.Int32(TB)
        for r in cutlass.range_constexpr(R):
            s_ptcnt[r * TB + tidx] = cnt[r]

    # ------------------------------------------------------------------
    # collect_at<TB>: plain streaming collect at thr (uncapped; caller
    # guarantees count(thr) <= kC).
    # ------------------------------------------------------------------
    @cute.jit
    def collect_at(self, row_addr, v0, v1, n_eff, thr, tidx, s_cand, s_isc):
        TB = cutlass.const_expr(self.num_threads)
        CS = cutlass.const_expr(self.cluster_size)
        kcap = cutlass.const_expr(self.kC)
        copy_atom = self._copy_atom()
        a_cnt = cutlass.Int32(0)
        a_st = cutlass.Int32(0)
        if cutlass.const_expr(CS > 1):
            a_cnt = _mapa_shared_cluster(s_isc.iterator + cutlass.Int32(5), cutlass.Int32(0))
            a_st = _mapa_shared_cluster(s_cand.iterator, cutlass.Int32(0))
        # Explicit U=4 batched loads. nvcc auto-unrolls the
        # CUDA collect_at loop 4x with 4 LDG.E.128 issued back-to-back; a
        # 1-deep loop was the dominant stall site (36% of all warp-stall
        # samples) on cells whose reuse check fails at big npad x big BS.
        frags = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(4)]
        vmain = v1
        vfull = n_eff >> cutlass.Int32(2)
        if vmain > vfull:
            vmain = vfull
        i = v0 + tidx
        while i + cutlass.Int32(3 * TB) < vmain:
            for u in cutlass.range_constexpr(4):
                self._ld_float4(copy_atom, row_addr, i + cutlass.Int32(u * TB), frags[u])
            for u in cutlass.range_constexpr(4):
                for q in cutlass.range_constexpr(4):
                    v = frags[u][q]
                    if v >= thr:
                        self._push_cand(
                            a_cnt,
                            a_st,
                            s_cand,
                            s_isc,
                            kcap,
                            False,
                            v,
                            ((i + cutlass.Int32(u * TB)) << cutlass.Int32(2)) + cutlass.Int32(q),
                        )
            i = i + cutlass.Int32(4 * TB)
        while i < vmain:
            self._ld_float4(copy_atom, row_addr, i, frags[0])
            for q in cutlass.range_constexpr(4):
                v = frags[0][q]
                if v >= thr:
                    self._push_cand(
                        a_cnt,
                        a_st,
                        s_cand,
                        s_isc,
                        kcap,
                        False,
                        v,
                        (i << cutlass.Int32(2)) + cutlass.Int32(q),
                    )
            i = i + cutlass.Int32(TB)
        while i < v1:
            self._ld_float4(copy_atom, row_addr, i, frags[0])
            gi = i << cutlass.Int32(2)
            for q in cutlass.range_constexpr(4):
                v = _mask_tail(frags[0][q], gi + cutlass.Int32(q), n_eff)
                if v >= thr:
                    self._push_cand(
                        a_cnt, a_st, s_cand, s_isc, kcap, False, v, gi + cutlass.Int32(q)
                    )
            i = i + cutlass.Int32(TB)

    # ------------------------------------------------------------------
    # kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def gvr_tp_kernel(
        self, logits: cute.Tensor, pre_idx: cute.Tensor, seq_lens: cute.Tensor, out_idx: cute.Tensor
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        TB = cutlass.const_expr(self.num_threads)
        CS = cutlass.const_expr(self.cluster_size)
        AR = cutlass.const_expr(self.ar)
        UF = cutlass.const_expr(self.uf)
        K = cutlass.const_expr(self.top_k)
        kC = cutlass.const_expr(self.kC)

        if cutlass.const_expr(CS > 1):
            row = bidx // cutlass.Int32(CS)
            rank = cute.arch.block_idx_in_cluster()
        else:
            row = bidx
            rank = cutlass.Int32(0)

        npad = cutlass.Int32(logits.shape[1])
        logits_row = logits[row, None]
        # Hint sharing: pre_idx is request-level ([num_rows // next_n, K]);
        # the next_n MTP rows of one request share the same hint row —
        # mirrors the in-tree run_one_row's pre_idx_row_idx = row // next_n.
        if cutlass.const_expr(self.next_n == 1):
            pre_idx_row = pre_idx[row, None]
        else:
            pre_idx_row = pre_idx[row // cutlass.Int32(self.next_n), None]
        out_row = out_idx[row, None]
        row_addr = logits_row.iterator.toint()

        # Ragged N: per-row valid length from seq_lens (see _row_n_eff).
        n_eff = self._row_n_eff(seq_lens, row)

        # ---- shared memory (order must be identical across CTAs for mapa) ----
        smem = SmemAllocator()
        # Packed (key<<32 | idx) u64 candidates = CUDA's
        # `unsigned long long cand[kC]` (one 8B push per candidate).
        s_cand = smem.allocate_tensor(
            element_type=cutlass.Uint64,
            layout=cute.make_ordered_layout((kC,), order=(0,)),
            byte_alignment=128,
        )
        s_ptcnt = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((RUNGS * self.num_threads,), order=(0,)),
            byte_alignment=128,
        )
        s_hist = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((256,), order=(0,)),
            byte_alignment=128,
        )
        s_rungs = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((RUNGS,), order=(0,)),
            byte_alignment=32,
        )
        s_rcnt = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((RUNGS,), order=(0,)),
            byte_alignment=32,
        )
        s_rpre = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((RUNGS,), order=(0,)),
            byte_alignment=32,
        )
        s_ipartial = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((2 * RUNGS,), order=(0,)),
            byte_alignment=32,
        )
        s_fpartial = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((2,), order=(0,)),
            byte_alignment=16,
        )
        s_fwred = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((2 * self.num_warps,), order=(0,)),
            byte_alignment=64,
        )
        s_hminmax = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((2,), order=(0,)),
            byte_alignment=16,
        )
        # iscalars: [0]=sel_bin [1]=sel_above [2]=sel_count [3]=cnt_m [4]=cnt_t [5]=cnt_c
        s_isc = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((8,), order=(0,)),
            byte_alignment=32,
        )

        # slice in 64-float units (npad multiple of 64); v0/v1 are float4 indices
        units = npad >> cutlass.Int32(6)
        u0 = (units * rank) // cutlass.Int32(CS)
        u1 = (units * (rank + cutlass.Int32(1))) // cutlass.Int32(CS)
        v0 = u0 << cutlass.Int32(4)
        v1 = u1 << cutlass.Int32(4)

        xch = cutlass.Int32(0)
        thr = cutlass.Float32(0.0)
        tpush = cutlass.Float32(0.0)
        C = cutlass.Int32(0)
        m_gt = cutlass.Int32(-1)
        span0 = cutlass.Float32(1e-3)

        # ---- Degenerate rows (N_eff <= K): identity emit + -1 pad. ----
        # Mirrors the in-tree kernel's degenerate branch. n_eff is uniform
        # across the cluster (all CTAs own the same row), so this is a
        # cluster-uniform branch; non-leader CTAs fall through to the exit
        # rendezvous (CuTe DSL has no runtime return).
        if n_eff <= cutlass.Int32(K):
            if rank == cutlass.Int32(0):
                jd = cutlass.Int32(tidx)
                while jd < n_eff:
                    out_row[jd] = jd
                    jd = jd + cutlass.Int32(TB)
                jp = n_eff + cutlass.Int32(tidx)
                if jp < cutlass.Int32(0):
                    jp = cutlass.Int32(tidx)  # n_eff < 0 (defensive): pad all
                while jp < cutlass.Int32(K):
                    out_row[jp] = cutlass.Int32(-1)
                    jp = jp + cutlass.Int32(TB)
        else:
            if rank == cutlass.Int32(0):
                if tidx == cutlass.Int32(0):
                    s_isc[5] = cutlass.Int32(0)  # cnt_c
            if cutlass.const_expr(CS > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

            if npad <= cutlass.Int32(kC):
                # trivial: whole row fits the candidate buffer. Masked tail
                # lanes push -FLT_MAX candidates; they can never displace a
                # real value because n_eff > K here (kth value > -FLT_MAX).
                if tidx < cutlass.Int32(RUNGS):
                    s_rungs[tidx] = cutlass.Float32(-FLT_MAX)
                cute.arch.barrier()
                self.count_pass(1, 8, row_addr, v0, v1, n_eff, tidx, s_rungs, s_ptcnt)
                self.exchange_counts(
                    RUNGS, cutlass.Int32(0), tidx, rank, s_ptcnt, s_rcnt, s_rpre, s_ipartial
                )
                xch = xch + cutlass.Int32(1)
                thr = cutlass.Float32(-FLT_MAX)
                C = s_rcnt[0]
                self.collect_at(row_addr, v0, v1, n_eff, thr, tidx, s_cand, s_isc)
                cute.arch.barrier()
            else:
                # cr==1 hint temporal shift (in-tree pre_idx_offset parity:
                # (row % next_n) + 1 maps prev-step indices into this step's
                # KV space); cr>1 keeps the exact pre-MTP trace (no offset
                # value is even computed).
                if cutlass.const_expr(self.compress_ratio == 1):
                    hint_off = (row % cutlass.Int32(self.next_n)) + cutlass.Int32(1)
                    self.phase1(
                        logits_row,
                        pre_idx_row,
                        n_eff,
                        tidx,
                        s_hist,
                        s_fwred,
                        s_hminmax,
                        s_rungs,
                        hint_off=hint_off,
                    )
                else:
                    self.phase1(
                        logits_row, pre_idx_row, n_eff, tidx, s_hist, s_fwred, s_hminmax, s_rungs
                    )
                hmin_floor = s_rungs[AR - 1]
                span0 = cute.arch.fmax(s_hminmax[1] - s_hminmax[0], cutlass.Float32(1e-3))
                # P2a: sampled ladder count -> pivot pick. The exchanged rows
                # carry packed (occ << 16 | cnt) values (admission sigma
                # input); every consumer below unpacks with & 0xFFFF / >> 16.
                self.sample_count(AR, row_addr, v0, v1, n_eff, tidx, s_rungs, s_ptcnt)
                self.exchange_counts(
                    AR, xch & cutlass.Int32(1), tidx, rank, s_ptcnt, s_rcnt, s_rpre, s_ipartial
                )
                xch = xch + cutlass.Int32(1)
                cute.arch.barrier()
                if tidx == cutlass.Int32(0):
                    # 4-stage pivot pick (R0-admission tightest-in-window ->
                    # band target -> 2-sigma bound -> gamble fallback)
                    lo = cutlass.const_expr((3 * K) // 2)
                    hi = cutlass.const_expr((6 * kC) // 10)
                    tgt_py = min(max(3 * K, (3 * K) // 2), (6 * kC) // 10)
                    tgt = cutlass.Int32(tgt_py)
                    best = cutlass.Int32(AR - 1)
                    bestd = cutlass.Int32(0x7FFFFFFF)
                    # Stage 0 — hint-ladder ADMISSION (in-tree R0
                    # parity): accept the TIGHTEST ladder rung
                    # (highest threshold = smallest admitted-candidate set)
                    # whose sampled-count confidence interval lies inside
                    # the [K, kC] acceptance window, mirroring the R0 rule
                    # "smallest exact count in [K, kC]". The old stage-1
                    # band ([1.5K, 0.6kC], target ~3K) systematically picks
                    # a FAT rung on real high-hit-rate rows (2-4x more P3
                    # pushes + P4 candidates than needed — the v32/pro
                    # BS>=16 losses vs the in-tree R0 kernel) or, when
                    # spatially-clustered real data inflates a sampled
                    # estimate into the band while the true count is < K,
                    # an undershooting one (whole-row secant + re-stream:
                    # the flash_512k 1.6-1.8x losses).
                    # Sigma is clustering-aware: est = SS*cnt, sigma =
                    # SS*cnt/sqrt(occ) (compound Poisson over occupied
                    # float4 clumps; equals the classic sqrt(SS*est) when
                    # occ == cnt, i.e. IID rows).
                    # Fix round (pr2 full-grid regressions), all changes on
                    # the PICK only — the fused pass / rescue / exactness
                    # machinery is byte-identical to the admission commit:
                    #  * upper acceptance bound tightened from kC to the
                    #    legacy pivot-band hi (0.6*kC): an in-window-but-
                    #    fat admitted rung inflates P3 pushes + P4
                    #    candidates 2-4x over the ~3K legacy target;
                    #  * K2048 lower margin raised 1.5 -> 1.85 sigma: at
                    #    1.5 the admitted rung's true count lands under K
                    #    on clustered rows (v32_32k_L23 rung est 2720,
                    #    sigma 374, true 1959 < K) and the rescue
                    #    re-stream costs more than the fat legacy pick it
                    #    displaced; 1.85 keeps the genuine tight admits
                    #    (v32_32k_L50: margin +1.92 sigma, true 2917);
                    #  * stage 0b below: when the legacy band pick is
                    #    strictly LEANER than the admitted rung and safe
                    #    by its own clustering-aware 1.5-sigma lower CI,
                    #    prefer it (mlo=2.0 rows whose next-leaner rung
                    #    sits at 1.8-2.0 sigma otherwise admit a 2.5x
                    #    fatter set: pro_64k_L06 4134 vs legacy 1671).
                    # Rungs are descending in j, so the first passing j is
                    # the tightest.
                    # All margin tests below are sqrt/div-free: with
                    # sigma = est/sqrt(occ), "est - m*sigma >= K" is
                    # "(est-K)^2 * occ >= m^2 * est^2" (est >= K), and
                    # "est + 2*sigma <= U" is "4*est^2 <= (U-est)^2 * occ"
                    # (est <= U). The pick is a THREAD-0 serial section
                    # between two CTA barriers; the sqrt+fdiv chain of the
                    # first admission cut measured ~2-5% whole-kernel on
                    # L2-resident accept-path rows.
                    mlo2 = cutlass.const_expr(1.85 * 1.85 if K >= 2048 else 4.0)
                    ubnd = cutlass.const_expr(float((6 * kC) // 10))
                    adm_est = cutlass.Int32(0x7FFFFFFF)
                    for j in cutlass.range_constexpr(AR):
                        cnt_j = s_rcnt[j] & cutlass.Int32(0xFFFF)
                        if cnt_j > cutlass.Int32(0):
                            if bestd == cutlass.Int32(0x7FFFFFFF):
                                occ_j = s_rcnt[j] >> cutlass.Int32(16)
                                if occ_j < cutlass.Int32(1):
                                    occ_j = cutlass.Int32(1)
                                focc = cutlass.Float32(occ_j)
                                fest = cutlass.Float32(cnt_j * cutlass.Int32(SS))
                                fe2 = fest * fest
                                a_lo = fest - cutlass.Float32(float(K))
                                bhi = cutlass.Float32(ubnd) - fest
                                if a_lo >= cutlass.Float32(0.0):
                                    if a_lo * a_lo * focc >= cutlass.Float32(mlo2) * fe2:
                                        if bhi >= cutlass.Float32(0.0):
                                            if cutlass.Float32(4.0) * fe2 <= bhi * bhi * focc:
                                                bestd = cutlass.Int32(0)
                                                best = cutlass.Int32(j)
                                                adm_est = cnt_j * cutlass.Int32(SS)
                    # Record the stage-0 CI-admission outcome
                    # BEFORE stages 0b/1/2 mutate bestd; a stage-0 admit
                    # is the "confident P2A" signal — those rows never
                    # take the stage-0c override.
                    s0_ok = cutlass.Int32(0)
                    if bestd == cutlass.Int32(0):
                        s0_ok = cutlass.Int32(1)
                    # Stage 0b — legacy band pick (min |est - tgt| in
                    # [1.5K, 0.6kC], EXACTLY the pr1 stage-1 rule), then
                    # combine: with no admission it is taken as-is (pr1
                    # parity); with an admitted rung it OVERRIDES only
                    # when strictly leaner AND safe by its own
                    # clustering-aware 1.5-sigma lower CI.
                    jb = cutlass.Int32(AR - 1)
                    jbd = cutlass.Int32(0x7FFFFFFF)
                    for j in cutlass.range_constexpr(AR):
                        est = (s_rcnt[j] & cutlass.Int32(0xFFFF)) * cutlass.Int32(SS)
                        if est >= cutlass.Int32(lo):
                            if est <= cutlass.Int32(hi):
                                dd = est - tgt
                                if dd < cutlass.Int32(0):
                                    dd = tgt - est
                                if dd < jbd:
                                    jbd = dd
                                    jb = cutlass.Int32(j)
                    if jbd != cutlass.Int32(0x7FFFFFFF):
                        if bestd == cutlass.Int32(0x7FFFFFFF):
                            bestd = jbd
                            best = jb
                        else:
                            cnt_b = s_rcnt[jb] & cutlass.Int32(0xFFFF)
                            est_b = cnt_b * cutlass.Int32(SS)
                            if est_b < adm_est:
                                occ_b = s_rcnt[jb] >> cutlass.Int32(16)
                                if occ_b < cutlass.Int32(1):
                                    occ_b = cutlass.Int32(1)
                                fb = cutlass.Float32(est_b)
                                ab = fb - cutlass.Float32(float(K))
                                if ab >= cutlass.Float32(0.0):
                                    if (
                                        ab * ab * cutlass.Float32(occ_b)
                                        >= cutlass.Float32(2.25) * fb * fb
                                    ):
                                        bestd = jbd
                                        best = jb
                    if bestd == cutlass.Int32(0x7FFFFFFF):
                        for j in cutlass.range_constexpr(AR):
                            est = (s_rcnt[j] & cutlass.Int32(0xFFFF)) * cutlass.Int32(SS)
                            if est > cutlass.Int32(0):
                                g = cutlass.Float32(2.0) * cmath.sqrt(
                                    cutlass.Float32(cutlass.Int32(SS) * est)
                                )
                                fest = cutlass.Float32(est)
                                if fest - g >= cutlass.Float32(float(K)):
                                    if fest + g <= cutlass.Float32(float(kC)):
                                        dd = est - tgt
                                        if dd < cutlass.Int32(0):
                                            dd = tgt - est
                                        if dd < bestd:
                                            bestd = dd
                                            best = cutlass.Int32(j)
                    if bestd == cutlass.Int32(0x7FFFFFFF):
                        if (s_rcnt[AR - 1] & cutlass.Int32(0xFFFF)) * cutlass.Int32(
                            SS
                        ) > cutlass.Int32(kC):
                            for j in cutlass.range_constexpr(AR):
                                est = (s_rcnt[j] & cutlass.Int32(0xFFFF)) * cutlass.Int32(SS)
                                if est > cutlass.Int32(0):
                                    dd = est - tgt
                                    if dd < cutlass.Int32(0):
                                        dd = tgt - est
                                    if dd < bestd:
                                        bestd = dd
                                        best = cutlass.Int32(j)
                    # ---- stage 0c: lean-pivot override ----
                    # When the stage-0 CI admission FAILED, npad <=
                    # 262144, and some ladder rung is (i) strictly LEANER
                    # than the pick in hand and (ii) lands in [K, kC] by
                    # its own clustering-aware 1.5-sigma CI on BOTH sides
                    # (a looser window than stage 0's [K, 0.6kC] at
                    # 1.85/2.0-sigma — rows passing THAT never get here),
                    # make IT the pivot. Rungs are descending in j: scan
                    # ascending and keep the FIRST (tightest) qualifier.
                    # Hint-unrepresentative rows (the measured harm
                    # band) have no CI-qualifying rung and keep the baseline
                    # pick. Same sqrt/div-free margin algebra as stage 0;
                    # everything downstream (fused R=2, rescue, reuse,
                    # P4, secant) is byte-parity with the baseline.
                    # npad floor 16384: below it the stride-32 sample sees
                    # <= ~128 float4s, the occ-aware CI fires on noise and
                    # misfire rescues cost 3-5% (measured, npad~8K probe);
                    # the absolute win-room there is small anyway.
                    gate_ok = cutlass.Int32(0)
                    if npad >= cutlass.Int32(16384):
                        if npad <= cutlass.Int32(262144):
                            gate_ok = cutlass.Int32(1)
                    if gate_ok != cutlass.Int32(0):
                        if s0_ok == cutlass.Int32(0):
                            est_pick = cutlass.Int32(0x7FFFFFFF)
                            if bestd != cutlass.Int32(0x7FFFFFFF):
                                est_pick = (s_rcnt[best] & cutlass.Int32(0xFFFF)) * cutlass.Int32(
                                    SS
                                )
                            jw = cutlass.Int32(AR)
                            for w_ in cutlass.range_constexpr(AR):
                                if jw == cutlass.Int32(AR):
                                    cnt_w = s_rcnt[w_] & cutlass.Int32(0xFFFF)
                                    if cnt_w > cutlass.Int32(0):
                                        estw = cnt_w * cutlass.Int32(SS)
                                        if estw < est_pick:
                                            occ_w = s_rcnt[w_] >> cutlass.Int32(16)
                                            if occ_w < cutlass.Int32(1):
                                                occ_w = cutlass.Int32(1)
                                            foccw = cutlass.Float32(occ_w)
                                            festw = cutlass.Float32(estw)
                                            few2 = festw * festw
                                            a_low = festw - cutlass.Float32(float(K))
                                            bhiw = cutlass.Float32(float(kC)) - festw
                                            if a_low >= cutlass.Float32(0.0):
                                                if (
                                                    a_low * a_low * foccw
                                                    >= cutlass.Float32(2.25) * few2
                                                ):
                                                    if bhiw >= cutlass.Float32(0.0):
                                                        if (
                                                            cutlass.Float32(2.25) * few2
                                                            <= bhiw * bhiw * foccw
                                                        ):
                                                            jw = cutlass.Int32(w_)
                            if jw < cutlass.Int32(AR):
                                best = jw
                                bestd = cutlass.Int32(0)
                    tp_ = s_rungs[best]
                    # Rescue rung: the next FATTER ladder rung below the
                    # pivot. If the pivot's exact count lands under K
                    # (sampling error on a clustered row), the driver can
                    # accept the rescue with ONE collect re-stream instead
                    # of the multi-pass secant loop. Read before the ladder
                    # slots are overwritten.
                    resc_ = hmin_floor
                    if best < cutlass.Int32(AR - 1):
                        resc_ = s_rungs[best + cutlass.Int32(1)]
                    # ---- lean pivot (per-K interpolated push threshold,
                    # composed AFTER stage 0c): the pick in hand — band,
                    # 2-sigma, overshoot fallback, or a stage-0c lean rung
                    # that is still fat — is structurally FAT on the
                    # 32k-128k residual band (exact Cp 2.4-4.7x K; P3
                    # pushes + P4 radix pay for it). When the picked
                    # rung's sampled est >= 1.8K, interpolate tpush
                    # between the pick and the next tighter rung with
                    # est < lean_tgt, targeting count ~= lean_tgt
                    # (log2-count interpolation: tails are ~exponential,
                    # the linear form undershoots). Rescue = the ORIGINAL
                    # pick, whose exact count the fused pass computes
                    # anyway: an undershoot costs one collect re-stream,
                    # cheap under the npad <= 98304 gate; the 256k-1024k
                    # guard band keeps the stock pick bit-for-bit.
                    if npad <= cutlass.Int32(98304):
                        estp = (s_rcnt[best] & cutlass.Int32(0xFFFF)) * cutlass.Int32(SS)
                        if estp >= cutlass.Int32((9 * K) // 5):
                            jm = cutlass.Int32(-1)
                            estm = cutlass.Int32(1)
                            for j in cutlass.range_constexpr(AR):
                                if cutlass.Int32(j) < best:
                                    cj = (s_rcnt[j] & cutlass.Int32(0xFFFF)) * cutlass.Int32(SS)
                                    if cj < cutlass.Int32(self.lean_tgt):
                                        jm = cutlass.Int32(j)
                                        estm = cj
                            if jm >= cutlass.Int32(0):
                                if estm < cutlass.Int32(1):
                                    estm = cutlass.Int32(1)
                                fp = _lg2_f32(cutlass.Float32(estp))
                                fm = _lg2_f32(cutlass.Float32(estm))
                                den = fp - fm
                                if den < cutlass.Float32(1e-6):
                                    den = cutlass.Float32(1e-6)
                                frac = (fp - cutlass.Float32(math.log2(self.lean_tgt))) / den
                                resc_ = tp_
                                tp_ = tp_ + (s_rungs[jm] - tp_) * frac
                    s_rungs[0] = tp_
                    s_rungs[1] = resc_
                cute.arch.barrier()
                tpush = s_rungs[0]
                self.fused_count_collect(
                    UF, row_addr, v0, v1, n_eff, tpush, tidx, s_rungs, s_ptcnt, s_cand, s_isc
                )
                self.exchange_counts(
                    2, xch & cutlass.Int32(1), tidx, rank, s_ptcnt, s_rcnt, s_rpre, s_ipartial
                )
                xch = xch + cutlass.Int32(1)

                # ---- P2c: secant refine driver (redundant on every thread) ----
                t_lo = cutlass.Float32(-FLT_MAX)
                t_hi = cutlass.Float32(INF)
                c_hi = cutlass.Int32(0)
                Rcur = cutlass.Int32(2)
                passno = cutlass.Int32(0)
                running = cutlass.Int32(1)
                descend_break = cutlass.Int32(0)
                while running != cutlass.Int32(0):
                    # first rung index j with rcnt[j] >= K (rcnt ascending in j)
                    j = Rcur
                    for r_ in range(RUNGS - 1, -1, -1):
                        if cutlass.Int32(r_) < Rcur:
                            if s_rcnt[r_] >= cutlass.Int32(K):
                                j = cutlass.Int32(r_)
                    jj = j
                    if jj > Rcur - cutlass.Int32(1):
                        jj = Rcur - cutlass.Int32(1)
                    cj = s_rcnt[jj]
                    rj = s_rungs[jj]
                    found = cutlass.Int32(0)
                    if j < Rcur:
                        if cj <= cutlass.Int32(kC):
                            found = cutlass.Int32(1)
                    if found != cutlass.Int32(0):
                        thr = rj
                        C = cj
                        running = cutlass.Int32(0)
                    else:
                        if j < Rcur:
                            if rj >= t_lo:
                                t_lo = rj
                        if j > cutlass.Int32(0):
                            jm = j - cutlass.Int32(1)
                            if s_rungs[jm] <= t_hi:
                                t_hi = s_rungs[jm]
                                c_hi = s_rcnt[jm]
                        descend = cutlass.Int32(0)
                        if passno >= cutlass.Int32(MAXPASS):
                            descend = cutlass.Int32(1)
                        # ladder params (uniform; recomputed per thread)
                        e3 = passno * cutlass.Int32(3)
                        if e3 > cutlass.Int32(24):
                            e3 = cutlass.Int32(24)
                        step = span0 * _exp2i(e3)
                        dt = cutlass.Float32(0.0)
                        mode = cutlass.Int32(2)  # 0=up-ladder,1=down-ladder,2=secant
                        if t_hi == cutlass.Float32(INF):
                            mode = cutlass.Int32(0)
                            nr0 = t_lo + step * cutlass.Float32(float(1 << (AR - 1)))
                            if nr0 == cutlass.Float32(INF):
                                descend = cutlass.Int32(1)
                        else:
                            if t_lo == cutlass.Float32(-FLT_MAX):
                                mode = cutlass.Int32(1)
                            else:
                                dt = (t_hi - t_lo) * cutlass.Float32(1.0 / float(AR + 1))
                                nr_last = t_hi - dt * cutlass.Float32(float(AR))
                                nr_first = t_hi - dt
                                ok = cutlass.Int32(0)
                                if nr_last > t_lo:
                                    if nr_first < t_hi:
                                        ok = cutlass.Int32(1)
                                if ok == cutlass.Int32(0):
                                    descend = cutlass.Int32(1)
                        if descend != cutlass.Int32(0):
                            running = cutlass.Int32(0)
                            descend_break = cutlass.Int32(1)
                        else:
                            cute.arch.barrier()
                            if tidx == cutlass.Int32(0):
                                for r_ in cutlass.range_constexpr(AR):
                                    nrv = cutlass.Float32(0.0)
                                    if mode == cutlass.Int32(0):
                                        nrv = t_lo + step * cutlass.Float32(
                                            float(1 << (AR - 1 - r_))
                                        )
                                    else:
                                        if mode == cutlass.Int32(1):
                                            nrv = t_hi - step * cutlass.Float32(float(1 << r_))
                                        else:
                                            nrv = t_hi - dt * cutlass.Float32(float(r_ + 1))
                                    s_rungs[r_] = nrv
                            cute.arch.barrier()
                            self.count_pass(AR, UF, row_addr, v0, v1, n_eff, tidx, s_rungs, s_ptcnt)
                            self.exchange_counts(
                                AR,
                                xch & cutlass.Int32(1),
                                tidx,
                                rank,
                                s_ptcnt,
                                s_rcnt,
                                s_rpre,
                                s_ipartial,
                            )
                            xch = xch + cutlass.Int32(1)
                            passno = passno + cutlass.Int32(1)
                            Rcur = cutlass.Int32(AR)

                # ---- plateau descent (exact max-below stepping) ----
                if descend_break != cutlass.Int32(0):
                    pl = cutlass.Int32(1)
                    while pl != cutlass.Int32(0):
                        vstar = self.max_below_pass(
                            row_addr,
                            v0,
                            v1,
                            n_eff,
                            t_hi,
                            xch & cutlass.Int32(1),
                            tidx,
                            s_fwred,
                            s_fpartial,
                        )
                        xch = xch + cutlass.Int32(1)
                        cute.arch.barrier()
                        if tidx == cutlass.Int32(0):
                            s_rungs[0] = vstar
                        cute.arch.barrier()
                        self.count_pass(1, 8, row_addr, v0, v1, n_eff, tidx, s_rungs, s_ptcnt)
                        self.exchange_counts(
                            RUNGS,
                            xch & cutlass.Int32(1),
                            tidx,
                            rank,
                            s_ptcnt,
                            s_rcnt,
                            s_rpre,
                            s_ipartial,
                        )
                        xch = xch + cutlass.Int32(1)
                        c = s_rcnt[0]
                        okc = cutlass.Int32(0)
                        if c >= cutlass.Int32(K):
                            if c <= cutlass.Int32(kC):
                                okc = cutlass.Int32(1)
                        if okc != cutlass.Int32(0):
                            thr = vstar
                            C = c
                            pl = cutlass.Int32(0)
                        else:
                            if c < cutlass.Int32(K):
                                t_hi = vstar
                                c_hi = c
                            else:
                                thr = vstar
                                m_gt = c_hi
                                pl = cutlass.Int32(0)

                # ---- candidate reuse check / re-stream collect ----
                if m_gt < cutlass.Int32(0):
                    if cutlass.const_expr(CS > 1):
                        cute.arch.cluster_arrive()
                        cute.arch.cluster_wait()
                    else:
                        cute.arch.barrier()
                    if cutlass.const_expr(CS > 1):
                        a_cnt0 = _mapa_shared_cluster(
                            s_isc.iterator + cutlass.Int32(5), cutlass.Int32(0)
                        )
                        dcnt = _ld_shared_cluster_i32(a_cnt0)
                    else:
                        dcnt = s_isc[5]
                    reuse = cutlass.Int32(0)
                    if thr == tpush:
                        if dcnt == C:
                            reuse = cutlass.Int32(1)
                    if reuse == cutlass.Int32(0):
                        if cutlass.const_expr(CS > 1):
                            cute.arch.cluster_arrive()
                            cute.arch.cluster_wait()
                        else:
                            cute.arch.barrier()
                        if rank == cutlass.Int32(0):
                            if tidx == cutlass.Int32(0):
                                s_isc[5] = cutlass.Int32(0)
                        if cutlass.const_expr(CS > 1):
                            cute.arch.cluster_arrive()
                            cute.arch.cluster_wait()
                        else:
                            cute.arch.barrier()
                        self.collect_at(row_addr, v0, v1, n_eff, thr, tidx, s_cand, s_isc)
                    cute.arch.barrier()

            # ---- plateau direct emit (all CTAs stream their slice) ----
            if m_gt >= cutlass.Int32(0):
                if rank == cutlass.Int32(0):
                    if tidx == cutlass.Int32(0):
                        s_isc[3] = cutlass.Int32(0)  # cnt_m
                        s_isc[4] = cutlass.Int32(0)  # cnt_t
                if cutlass.const_expr(CS > 1):
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                    a_m = _mapa_shared_cluster(s_isc.iterator + cutlass.Int32(3), cutlass.Int32(0))
                    a_t = _mapa_shared_cluster(s_isc.iterator + cutlass.Int32(4), cutlass.Int32(0))
                else:
                    cute.arch.barrier()
                nt = cutlass.Int32(K) - m_gt
                copy_atom = self._copy_atom()
                frag = cute.make_rmem_tensor((4,), cutlass.Float32)
                ii = v0 + tidx
                while ii < v1:
                    self._ld_float4(copy_atom, row_addr, ii, frag)
                    gi = ii << cutlass.Int32(2)
                    for q in cutlass.range_constexpr(4):
                        v = _mask_tail(frag[q], gi + cutlass.Int32(q), n_eff)
                        if v > thr:
                            if cutlass.const_expr(CS > 1):
                                p = _atom_shared_cluster_add_i32(a_m, cutlass.Int32(1))
                            else:
                                p = atomicAdd(s_isc.iterator + cutlass.Int32(3), cutlass.Int32(1))
                            out_row[p] = gi + cutlass.Int32(q)
                        else:
                            if v == thr:
                                if cutlass.const_expr(CS > 1):
                                    p = _atom_shared_cluster_add_i32(a_t, cutlass.Int32(1))
                                else:
                                    p = atomicAdd(
                                        s_isc.iterator + cutlass.Int32(4), cutlass.Int32(1)
                                    )
                                if p < nt:
                                    out_row[m_gt + p] = gi + cutlass.Int32(q)
                    ii = ii + cutlass.Int32(TB)
            else:
                # ---- P4 (CTA0 solo when CS>1) ----
                if cutlass.const_expr(CS > 1):
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                if rank == cutlass.Int32(0):
                    if C == cutlass.Int32(K):
                        ie = cutlass.Int32(tidx)
                        while ie < C:
                            out_row[ie] = cutlass.Int32(
                                cutlass.Uint32(s_cand[ie] & cutlass.Uint64(0xFFFFFFFF))
                            )
                            ie = ie + cutlass.Int32(TB)
                    else:
                        # 4x8-bit radix select over cand keys (f2u in round 0)
                        if tidx < cutlass.Int32(256):
                            s_hist[tidx] = cutlass.Int32(0)
                        cute.arch.barrier()
                        pref = cutlass.Uint32(0)
                        want = cutlass.Int32(K)
                        m = cutlass.Int32(0)
                        final_shift = cutlass.Uint32(0)
                        active = cutlass.Int32(1)
                        for r_ in cutlass.range_constexpr(4):
                            shift = cutlass.const_expr(24 - 8 * r_)
                            if active != cutlass.Int32(0):
                                ih = cutlass.Int32(tidx)
                                while ih < C:
                                    if cutlass.const_expr(r_ == 0):
                                        kv = s_cand[ih]
                                        raw = cutlass.Uint32(kv >> cutlass.Uint64(32))
                                        u = _f2u_bits(raw)
                                        s_cand[ih] = (cutlass.Uint64(u) << cutlass.Uint64(32)) | (
                                            kv & cutlass.Uint64(0xFFFFFFFF)
                                        )
                                        atomicAdd(
                                            s_hist.iterator
                                            + cutlass.Int32(u >> cutlass.Uint32(24)),
                                            cutlass.Int32(1),
                                        )
                                    else:
                                        u = cutlass.Uint32(s_cand[ih] >> cutlass.Uint64(32))
                                        if (u >> cutlass.Uint32(shift + 8)) == pref:
                                            b = cutlass.Int32(
                                                (u >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                                            )
                                            atomicAdd(s_hist.iterator + b, cutlass.Int32(1))
                                    ih = ih + cutlass.Int32(TB)
                                cute.arch.barrier()
                                if tidx < cutlass.Int32(32):
                                    b8 = tidx * cutlass.Int32(8)
                                    h = cute.make_rmem_tensor((8,), cutlass.Int32)
                                    Ssum = cutlass.Int32(0)
                                    for q in cutlass.range_constexpr(8):
                                        h[q] = s_hist[b8 + cutlass.Int32(q)]
                                        Ssum = Ssum + h[q]
                                        s_hist[b8 + cutlass.Int32(q)] = cutlass.Int32(0)
                                    x = Ssum
                                    for o in [1, 2, 4, 8, 16]:
                                        x = _shfl_down_add(x, tidx, o)
                                    A = x - Ssum
                                    if A < want:
                                        if want <= A + Ssum:
                                            run = A
                                            for q in range(7, -1, -1):
                                                if run < want:
                                                    if want <= run + h[q]:
                                                        s_isc[0] = b8 + cutlass.Int32(q)
                                                        s_isc[1] = run
                                                        s_isc[2] = h[q]
                                                run = run + h[q]
                                cute.arch.barrier()
                                sel = s_isc[0]
                                above = s_isc[1]
                                m = m + above
                                want = want - above
                                pref = (pref << cutlass.Uint32(8)) | cutlass.Uint32(sel)
                                if s_isc[2] == want:
                                    final_shift = cutlass.Uint32(shift)
                                    active = cutlass.Int32(0)
                        kth = pref
                        if tidx == cutlass.Int32(0):
                            s_isc[3] = cutlass.Int32(0)
                            s_isc[4] = cutlass.Int32(0)
                        cute.arch.barrier()
                        nt = cutlass.Int32(K) - m
                        ie = cutlass.Int32(tidx)
                        while ie < C:
                            kv = s_cand[ie]
                            u = cutlass.Uint32(kv >> cutlass.Uint64(32)) >> final_shift
                            idx = cutlass.Int32(cutlass.Uint32(kv & cutlass.Uint64(0xFFFFFFFF)))
                            if u > kth:
                                p = atomicAdd(s_isc.iterator + cutlass.Int32(3), cutlass.Int32(1))
                                out_row[p] = idx
                            else:
                                if u == kth:
                                    p = atomicAdd(
                                        s_isc.iterator + cutlass.Int32(4), cutlass.Int32(1)
                                    )
                                    if p < nt:
                                        out_row[m + p] = idx
                            ie = ie + cutlass.Int32(TB)

        # Exit rendezvous: DSMEM (mapa'd peer smem) must stay valid until every
        # CTA of the cluster is done issuing remote ops (plateau emit's remote
        # cnt_m/cnt_t atomics can land after CTA0 finishes its own slice). nvcc
        # inserts an implicit cluster barrier before ret for DSMEM kernels; CuTe
        # DSL does not — without this, plateau rows fault with a
        # timing-dependent cudaErrorLaunchFailure (CUDA 719, hidden under
        # compute-sanitizer synccheck). DO NOT remove or make conditional.
        if cutlass.const_expr(CS > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    # ------------------------------------------------------------------
    # host launcher
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        seq_lens: cute.Tensor,
        out_idx: cute.Tensor,
        stream,
    ):
        num_rows = logits.shape[0]
        CS = cutlass.const_expr(self.cluster_size)
        self.gvr_tp_kernel(logits, pre_idx, seq_lens, out_idx).launch(
            grid=(num_rows * CS, 1, 1),
            block=(self.num_threads, 1, 1),
            cluster=(CS, 1, 1) if cutlass.const_expr(CS > 1) else None,
            stream=stream,
            min_blocks_per_mp=2,
        )


# ---------------------------------------------------------------------------
# compile cache + public entry (mirrors launch_tp<512> selection exactly)
# ---------------------------------------------------------------------------
_LAUNCH_CACHE = {}


def _p2floor(x: int) -> int:
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def _two_waves_rows() -> int:
    """2 * SM count of the current device (cached): the co-residency row
    budget behind the CS selection below (min_blocks_per_mp=2). Mirrors
    ``_get_num_sms`` in ``cute_dsl_custom_ops.py`` (not importable from
    here — the import runs the other way). NOTE: only this term scales
    with the device; every other constant in ``tp_cluster_size`` and the
    dispatch band tables is frozen B200 calibration (the kernel family
    measured HW-invariant in earlier B200/B300 cross-arch A/Bs, so the
    bands are kept device-independent on purpose).
    """
    if not hasattr(_two_waves_rows, "_value"):
        _two_waves_rows._value = 2 * torch.cuda.get_device_properties().multi_processor_count
    return _two_waves_rows._value


def _get_compiled(K: int, CS: int, AR: int, UF: int, TB: int, next_n: int = 1, cr: int = 4):
    key = (K, CS, AR, UF, TB, next_n, cr)
    compiled = _LAUNCH_CACHE.get(key)
    if compiled is None:
        # Flat candidate budget, never the K-scaled
        # diet (falsified as pure harm).
        kC = 8192 if K >= 2048 else 6144
        kern = GvrTpKernel(
            top_k=K,
            kC=kC,
            cluster_size=CS,
            ar=AR,
            uf=UF,
            num_threads=TB,
            next_n=next_n,
            compress_ratio=cr,
        )
        n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
        # pre_idx is request-level: num_rows // next_n rows. At next_n == 1
        # keep the shared n_rows sym (identical compiled artifact to the v1
        # port); at next_n > 1 the row counts differ, so use a distinct sym.
        n_pre = n_rows if next_n == 1 else cute.sym_int()
        logits_fake = _crt.make_fake_compact_tensor(
            cutlass.Float32, (n_rows, n_cols), stride_order=(1, 0), assumed_align=16
        )
        pre_idx_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (n_pre, K), stride_order=(1, 0), assumed_align=16
        )
        seq_lens_fake = _crt.make_fake_compact_tensor(cutlass.Int32, (n_batch,), stride_order=(0,))
        out_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (n_rows, K), stride_order=(1, 0), assumed_align=16
        )
        fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            kern,
            logits_fake,
            pre_idx_fake,
            seq_lens_fake,
            out_fake,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )
        _LAUNCH_CACHE[key] = compiled
    return compiled


_FAST = {}  # (K, bs, npad) -> compiled variant (hot-path single dict hit)


def tp_cluster_size(bs: int, npad: int) -> int:
    """CS selection of launch_tp<512>: co-residency + slice floor.
    Occupancy cut for mid-N large-BS (measured neutral-to-positive)."""
    # C4 occupancy cut kept at fix-head default (ablation: neutral).
    if npad < 65536 and bs >= 64:
        return 1
    cs = 1
    if bs < 128:
        two_waves = _two_waves_rows()  # 296 on B200 (2 x 148 SMs)
        cs = _p2floor(two_waves // bs) if bs <= two_waves else 1
        capn = _p2floor(npad // 8192 if npad // 8192 > 0 else 1)
        if cs > capn:
            cs = capn
        if cs > 8:
            cs = 8
    return cs


def _dispatch(K: int, bs: int, npad: int, next_n: int = 1, cr: int = 4):
    """launch_tp<512> selection: CS by co-residency + slice floor, UF by depth."""
    assert npad % 64 == 0
    TB = 512
    AR = RUNGS
    cs = tp_cluster_size(bs, npad)
    if cs > 1:
        uf = 4
    else:
        uf = 8 if npad >= 16384 else 4
    return _get_compiled(K, cs, AR, uf, TB, next_n, cr)


def tp_topk(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    K: int,
    next_n: int = 1,
    cr: int = 4,
) -> None:
    """CuTe DSL gvr_topk_tp. logits [BS, npad] fp32 (npad mult of 64;
    tail beyond each row's N_eff may be garbage — masked in-kernel),
    pre_idx [BS // next_n, K] int32 hint (request-level), seq_lens
    [BS // next_n] int32 (uncompressed-token space), out [BS, K] int32.
    Launch-time CS/UF selection replicates launch_tp<512>."""
    sh = logits.shape
    key = (K, sh[0], sh[1], next_n, cr)
    fn = _FAST.get(key)
    if fn is None:
        fn = _dispatch(K, sh[0], sh[1], next_n, cr)
        _FAST[key] = fn
    fn(logits, pre_idx, seq_lens, out)
