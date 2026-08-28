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

"""GVR (Guess-Verify-Refine) Top-K kernel for Blackwell sm_100.

Unified single- and multi-CTA-per-row implementation. Each row is processed by
``cluster_size`` CTAs cooperating via a thread-block cluster; CTA ``r`` scans
``row[r*N/cs : (r+1)*N/cs]`` (vec_w-aligned split, last CTA absorbs remainder).
Per-iter cand_count is aggregated via DSMEM (``mapa.shared::cluster`` +
``ld.shared::cluster``) — no GMEM atomics. ``cluster_size=1`` degenerates to a
plain single-CTA path with all cluster code paths compiled out via
``const_expr``.

Supported (dtype, K): fp32 / bf16 / fp16 x 512 / 1024 / 2048.
cluster_size: 1 (default), 2, 4 (B200 GPC limit caps at ~16).
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.distributed import atomicAdd
from cutlass.utils.smem_allocator import SmemAllocator

from ..utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait
from .block_scan import warp_scan


def _env_flag(name: str) -> bool:
    """Debug-knob parse that never raises at import time: any of
    ""/0/false/off/no (case-insensitive) is False, everything else True."""
    return os.environ.get(name, "0").strip().lower() not in ("", "0", "false", "off", "no")


# Diagnostic knob: compile per-phase clock64 stamps of the list path
# into the spare xstate slots (harness-side analysis). Off by default;
# NEVER set in production.
_P4_TAIL_DBG = _env_flag("TRTLLM_GVR_P4_TAIL_DBG")
# P4 sub-phase clock64 breakdown -> xstate[1,2,4,5,6,7] (debug: clobbers
# the closed-loop thr/anch publish; single-shot cells only, not chains)
_P4_SUB_DBG = _env_flag("TRTLLM_GVR_P4_SUB_DBG")
# TRTLLM_GVR_P4_SUB_DBG=2: publish the P4 HEAD triple (minmax / histogram build /
# coarse search) instead of the tail triple, so the phase budget adds up.
_P4_SUB_HEAD = os.environ.get("TRTLLM_GVR_P4_SUB_DBG", "0").strip() == "2"
# Exact-tail small-class pair buffer. The scatter parks each member of the
# straddling tie class as (value bits, index) here, above the 256 digit bins
# the large-class radix zeroes and above its [256..258] scalars, so the repair
# never has to re-walk the candidates. The capacity is DERIVED from the bin
# count, never assumed: K=2048 with R0 shrinks the histogram to 512 bins (see
# the kNumBins override in __init__), where 260 + 2*128 would run 4 ints past
# the end of the allocation - into the per-thread count buffer that follows it.
# Rounded down to a power of two so the scatter can wrap the ordinal with a
# mask instead of a bounds branch; a class past the cap takes the large-class
# route, which does not use this buffer.
_PAIR_BASE = 260
_PAIR_MAX = 128


def _pair_cap_for(n_bins: int) -> int:
    """Largest power-of-two pair count that fits [_PAIR_BASE, n_bins)."""
    room = (n_bins - _PAIR_BASE) // 2
    if room < 1:
        return 0
    return min(_PAIR_MAX, 1 << (room.bit_length() - 1))


_SKIP_DBG = _env_flag("TRTLLM_GVR_SKIP_DBG")


# ---------------------------------------------------------------------------
# DSMEM primitives (inline PTX)
# Adapted from single_pass_multi_cta_radix_topk_cluster.py.
# ---------------------------------------------------------------------------
@dsl_user_op
def _mapa_shared_cluster(smem_ptr, peer_rank, *, loc=None, ip=None):
    """Map a local SMEM pointer to peer CTA's SMEM in cluster address space.

    PTX: ``mapa.shared::cluster.u32 %dst, %src, %peer_rank;``

    The returned i32 address can be passed to ``ld.shared::cluster.*`` and
    ``st.shared::cluster.*`` to read/write the peer's identically-laid-out
    SMEM allocation. CuTe DSL's high-level SMEM tensor ops do NOT lower to
    cluster-space loads, so DSMEM access must go through inline PTX.
    """
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


@cute.jit
def mapa_shared_cluster(smem_ptr, peer_rank):
    return _mapa_shared_cluster(smem_ptr, peer_rank)


@dsl_user_op
def _ld_shared_cluster_i32(mapped_addr, *, loc=None, ip=None):
    """Load an int32 from a peer CTA's SMEM via cluster mapped address."""
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


@cute.jit
def ld_shared_cluster_i32(mapped_addr):
    return _ld_shared_cluster_i32(mapped_addr)


@dsl_user_op
def _ld_shared_cluster_f32(mapped_addr, *, loc=None, ip=None):
    """Load an fp32 from a peer CTA's SMEM via cluster mapped address."""
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


@cute.jit
def ld_shared_cluster_f32(mapped_addr):
    return _ld_shared_cluster_f32(mapped_addr)


def float_as_uint32(float_val):
    """Interpret FP32 value as uint32 bit pattern (cuTe DSL bit-cast)."""
    return llvm.bitcast(cutlass.Uint32.mlir_type, float_val.ir_value())


def float_as_int32(float_val):
    """Interpret FP32 value as int32 bit pattern (cuTe DSL bit-cast)."""
    return cutlass.Int32(llvm.bitcast(cutlass.Int32.mlir_type, float_val.ir_value()))


def f32_order_key(float_val):
    """Order-preserving fp32 -> int32 key (unsigned-monotonic bit pattern).

    ``s ^ ((s >> 31) | 0x80000000)``: positive floats map to
    ``bits | 0x80000000``, negative floats to ``~bits`` — the standard radix
    transform whose UNSIGNED order equals fp32 order (NaN-free inputs). The
    returned Int32 must only be consumed digit-wise (``(k >> s) & 0xFF``) or
    via equality / prefix-equality; for a full ordered compare, flip the top
    bit first (``k ^ 0x80000000`` is signed-monotonic).
    """
    s = float_as_int32(float_val)
    return s ^ ((s >> cutlass.Int32(31)) | cutlass.Int32(-2147483648))


def f32_order_key_signed(float_val):
    """Signed-monotonic Int32 key (fp32 order == signed Int32 order), so the
    Phase-3 repair can bisect with provable collapse in <= 32 steps."""
    return f32_order_key(float_val) ^ cutlass.Int32(-2147483648)


def order_key_signed_to_f32(m):
    """Branchless inverse of :func:`f32_order_key_signed` (non-NaN keys)."""
    k = m ^ cutlass.Int32(-2147483648)
    top = k >> cutlass.Int32(31)  # -1 if top bit set, else 0
    mask = cutlass.Int32(-2147483648) | (~top & cutlass.Int32(2147483647))
    s = k ^ mask
    return cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, s.ir_value()))


def order_key_mid_f32(v_lo, v_hi):
    """Ordered-key midpoint; returns (mid_float, is_adjacent) where adjacency
    means no float strictly between v_lo and v_hi exists."""
    m_lo = f32_order_key_signed(v_lo)
    m_hi = f32_order_key_signed(v_hi)
    m_mid = (m_lo & m_hi) + ((m_lo ^ m_hi) >> cutlass.Int32(1))
    return order_key_signed_to_f32(m_mid), m_mid == m_lo




def _fmin_f32_inline(a, b):
    """Single PTX ``min.f32`` → one SASS FMNMX.

    cuTe DSL exposes ``cute.arch.fmax`` but not ``fmin``; the canonical
    ``-fmax(-a, -b)`` workaround lowers to 4 SASS insts and was worth
    ~8-10 µs of the prod-GVR gap at fp32 K=2048 BS=1.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [a.ir_value(), b.ir_value()],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# GvrParams<T, K> — parameters for different (dtype, K, compress_ratio) combinations.
# =============================================================================


@dataclass(frozen=True)
class GvrParams:
    kFTarget: int
    kC: int  # candidate buffer cap
    kNumBins: int  # histogram bin count

    @staticmethod
    def get(dtype_name: str, top_k: int, compress_ratio: int = 1) -> "GvrParams":
        """Per-(dtype, K, cr) tuning constants, mirroring CUDA's
        ``GvrParams<T, K>`` template specialization. For K ∈ {512, 1024}
        cr=1 (DSv3.2) and cr=4 (DSv4, PR #14413) use different kFTarget —
        V4 aligns kFTarget with kK to avoid upper-clamp saturation on
        tight-sigma layers (1.5-2.2x fewer P2 iters on swe-bench). K=2048 is
        identical across cr (V4 doesn't natively use it).
        """
        TABLE = {
            # --- cr = 1 (DSv3.2): tuned on V3.2 swe-bench data ---
            ("float32", 512, 1): GvrParams(kFTarget=384, kC=5120, kNumBins=1024),
            ("float32", 1024, 1): GvrParams(kFTarget=2560, kC=5120, kNumBins=1024),
            ("float32", 2048, 1): GvrParams(kFTarget=3072, kC=6144, kNumBins=2048),
            ("bfloat16", 512, 1): GvrParams(kFTarget=384, kC=5120, kNumBins=512),
            ("bfloat16", 1024, 1): GvrParams(kFTarget=2560, kC=5120, kNumBins=512),
            ("bfloat16", 2048, 1): GvrParams(kFTarget=4096, kC=5120, kNumBins=2048),
            ("float16", 512, 1): GvrParams(kFTarget=384, kC=5120, kNumBins=512),
            ("float16", 1024, 1): GvrParams(kFTarget=2560, kC=5120, kNumBins=1024),
            ("float16", 2048, 1): GvrParams(kFTarget=4096, kC=5120, kNumBins=2048),
            # --- cr = 4 (DSv4): tuned on V4 Flash/Pro swe-bench data ---
            ("float32", 512, 4): GvrParams(kFTarget=512, kC=5120, kNumBins=1024),
            ("float32", 1024, 4): GvrParams(kFTarget=1024, kC=5120, kNumBins=1024),
            ("float32", 2048, 4): GvrParams(kFTarget=3072, kC=6144, kNumBins=2048),
            ("bfloat16", 512, 4): GvrParams(kFTarget=512, kC=5120, kNumBins=512),
            ("bfloat16", 1024, 4): GvrParams(kFTarget=1024, kC=5120, kNumBins=512),
            ("bfloat16", 2048, 4): GvrParams(kFTarget=4096, kC=5120, kNumBins=2048),
            ("float16", 512, 4): GvrParams(kFTarget=512, kC=5120, kNumBins=512),
            ("float16", 1024, 4): GvrParams(kFTarget=1024, kC=5120, kNumBins=1024),
            ("float16", 2048, 4): GvrParams(kFTarget=4096, kC=5120, kNumBins=2048),
        }
        key = (dtype_name, top_k, compress_ratio)
        if key not in TABLE:
            raise ValueError(f"Unsupported GvrParams<{dtype_name}, {top_k}, cr={compress_ratio}>")
        return TABLE[key]


class GvrTopKKernel:
    """GVR (Guess-Verify-Refine) heuristic top-K kernel using cuTe DSL.

    One CTA processes one row.
    Block size = 512/1024, as specified by num_threads.
    Smem region sized to GvrParams<dtype, top_k>.

    Algorithm phases:
      P1: preIdx Min/Max/Mean → initial threshold
      P1b: 256-bin histogram over prev-topK gathered values → M rung
           thresholds (enable_r0 only)
      P2: threshold admission — default (enable_r0=True) is a single-pass
          multi-threshold rung-ladder; enable_r0=False keeps the classic
          secant threshold search loop (count-only), also the R0-miss fallback
      P3: Ballot-free candidate collect into smem keys[]/vals[]
      P4: rank-and-scatter (enable_r0) / histogram snap → exact top-K + writeback

    For different compress_ratio:
      cr = 1: preIdxOffset = (row_idx % next_n) + 1. V3.2 decode +1 temporal shift.
      cr = 4: preIdxOffset = 0. V4 decode no temporal shift.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        top_k: int,
        next_n: int = 1,
        num_threads: int = 512,
        enable_unroll_4: Optional[bool] = None,
        enable_phase3_unroll: Optional[bool] = None,
        use_constant_hint: bool = False,
        min_blocks_per_mp: int = 3,
        use_256bit_load: bool = False,
        enable_warp_parallel_reduce: Optional[bool] = None,
        compress_ratio: int = 1,
        return_output_values: bool = True,
        cluster_size: int = 1,
        enable_smem_cache: bool = False,
        smem_cache_elems: int = 32768,
        seqlen_sorted: bool = False,
        kc_diet: Optional[bool] = None,
        pdl_wait_late: bool = False,
        p4_fine_rangetest: Optional[bool] = None,
        p4_scat_rangetest: bool = False,
        enable_r0: bool = True,
        accept_cap: "int | None" = None,
        kc_override: "int | None" = None,
        self_scan: bool = False,
        cap_c: "int | None" = None,
        r0_qfracs: Optional[tuple] = None,
        mt_unroll: int = 4,
        p1b_cache: Optional[bool] = None,
        fb_fix: bool = True,
        fb_alpha: float = 0.2,
        r0_vseed: Optional[bool] = None,
        enable_p4_rank_scatter: Optional[bool] = None,
        enable_p4_rank_scatter_exact: Optional[bool] = None,
        p4_exact_tail: Optional[bool] = None,
        p4_tail_fast: Optional[bool] = None,  # [p4tt]
        p4_tail_v3: Optional[bool] = None,
        p4_no_fine: Optional[bool] = None,
        p1r_rescue: bool = True,
        num_bins: Optional[int] = None,
        p4_warp_redundant: bool = True,
        p2_warp_redundant: bool = True,
        enable_block_skip: bool = False,
        use_ext_counts: bool = False,
        ext_rungs: bool = False,
        use_ext_cand: bool = False,
        cand_cap: int = 5120,
        cand_rung: int = 0,
        emit_xstate: bool = False,
    ):
        # Redundant-warp sync reduction: every warp replays the block
        # reduce + decision from the same staged SMEM partials in the
        # same fp32 order, so results are bit-identical across warps and
        # the publish barrier + leader serialization disappear.
        #   p4_warp_redundant: P4 k-th bin search + snap loop (1 barrier/iter).
        #   p2_warp_redundant: P2 secant cadence (cluster_size == 1 only).
        # Both default ON; OFF restores the leader-based paths (A/B).
        self.p4_warp_redundant = p4_warp_redundant
        self.p2_warp_redundant = p2_warp_redundant
        # cluster_size: number of CTAs cooperating per row. 1 = single-CTA
        # path; 2/4 = thread-block cluster with DSMEM aggregation. Capped at
        # 16 by B200's per-GPC SM count.
        if cluster_size < 1 or cluster_size > 16:
            raise ValueError(
                f"cluster_size must be in [1, 16] (B200 GPC limit); got {cluster_size}"
            )
        self.cluster_size = cluster_size
        # When True, the kernel resolves the owning row per CTA via a
        # caller-provided ``order_row`` indirection — an LJF host-side
        # dispatch order so longer rows hit earlier waves. ``order_row``
        # is REQUEST-level: int32[batch_size = num_rows / next_n],
        # typically a descending argsort of seq_lens. The kernel
        # expands to row level as ``order_row[req] * next_n + nn`` so a
        # request's ``next_n`` rows stay contiguous. Compatible with
        # cluster_size > 1: all cs CTAs in a cluster see the same
        # cluster_id (= bidx // cluster_size), hence the same row.
        self.seqlen_sorted = seqlen_sorted
        # SMEM slice cache (optional): pre-stage each CTA's slice into SMEM
        # once between Phase 1 and Phase 2, so Phase 2/3's GE-count scans
        # read LDS instead of re-streaming GMEM. Caller is responsible for
        # ensuring slice_len <= smem_cache_elems; ``smem_cache_elems`` sets
        # the JIT-time alloc size (see TODO at _compile for host-side assert).
        if enable_smem_cache and smem_cache_elems <= 0:
            raise ValueError("smem_cache_elems must be > 0 when enable_smem_cache")
        self.enable_smem_cache = enable_smem_cache
        self.smem_cache_elems = smem_cache_elems
        # e.g., dtype = cutlass.Float32 / cutlass.BFloat16 / cutlass.Float16
        self.dtype = dtype
        self.top_k = top_k
        self.next_n = next_n
        # KV compression ratio:
        #   1 → DSv3.2; preIdxOffset = (row % next_n) + 1 to land prev-step
        #       indices in this step's KV space (with MTP windowing).
        #   4 → DSv4; logits/preIdx live in compressed-token-index space.
        #       New entries are appended at the end so prev indices stay
        #       valid → preIdxOffset = 0.
        assert compress_ratio in (1, 4), (
            f"compress_ratio must be 1 (V3.2) or 4 (V4); got {compress_ratio}"
        )
        self.compress_ratio = compress_ratio

        self.WARP_SIZE = 32
        self.num_threads = num_threads
        self.num_warps = num_threads // self.WARP_SIZE
        # __launch_bounds__(num_threads, min_blocks_per_mp) ptxas hint.
        # On B200 (65536 regs/SM, BS=512), max regs/thread is 128 at mb=1,
        # 64 at mb=2, 42 at mb=3. Pick low mb when num_rows ≤ #SMs so
        # ptxas can spend more regs covering LDG latency.
        self.min_blocks_per_mp = min_blocks_per_mp
        # Vector-load width for Phase 2/3 scans:
        #   False (default): 128-bit LDG  (fp32: 4 / bf16/fp16: 8 elems)
        #   True:            256-bit LDG  (fp32: 8 / bf16/fp16: 16 elems)
        # 256-bit halves the LDG count but needs 32B-aligned addresses
        # (we set assumed_align=32) and doubles fragment reg footprint.
        self.use_256bit_load = use_256bit_load
        self.vec_bits = 256 if use_256bit_load else 128
        self.vec_align_bytes = self.vec_bits // 8  # 32 for 256-bit, 16 for 128-bit
        # Vec-loop unroll switches.
        #   enable_unroll_4:        4-way fast path in block_count_ge.
        #   enable_phase3_unroll:   4-way fast path in phase3_collect.
        #     Independent of enable_unroll_4: Phase 3 has thread-local wc
        #     state + smem writes, so its fast-path trade-off differs.
        #   use_constant_hint:      True → CopyG2ROp(invariant=True) emits
        #     LDG.E.*.CONSTANT (read-only cache, == CUDA __ldg). False →
        #     plain CopyUniversalOp / LDG.E.*.
        if enable_unroll_4 is None:
            enable_unroll_4 = True
        if enable_phase3_unroll is None:
            enable_phase3_unroll = True
        self.enable_unroll_4 = enable_unroll_4
        self.enable_phase3_unroll = enable_phase3_unroll
        self.use_constant_hint = use_constant_hint
        # Replace tid==0 serial block-reduces with warp-parallel reduces
        # in warp 0. Auto-policy: on iff num_threads == 1024 (32 warps),
        # where the serial cost is meaningful; at 512 threads (16 warps)
        # the warp-parallel path regressed ~2pp on synth.
        if enable_warp_parallel_reduce is None:
            enable_warp_parallel_reduce = num_threads == 1024
        self.enable_warp_parallel_reduce = enable_warp_parallel_reduce

        # When False, skip all STG writes to ``output_values`` and accept
        # None at launch — saves LSU bandwidth + reg pressure for callers
        # that only consume top-K indices (e.g. the DSA indexer). When
        # True (default), values are written for bench / standalone use.
        self.return_output_values = return_output_values

        # Map cutlass dtype → GvrParams lookup name
        if dtype == cutlass.Float32:
            self._dtype_name = "float32"
        elif dtype == cutlass.BFloat16:
            self._dtype_name = "bfloat16"
        elif dtype == cutlass.Float16:
            self._dtype_name = "float16"
        else:
            raise ValueError(f"Unsupported dtype for GvrTopKKernel: {dtype}")

        params = GvrParams.get(self._dtype_name, top_k, self.compress_ratio)
        self.kC = params.kC
        # num_bins: the coarse histogram width. The bin search walks it
        # per warp, so it sets the barrier count of the P4 head.
        self.kNumBins = params.kNumBins if num_bins is None else int(num_bins)
        self.kFTarget = params.kFTarget

        # Kernel-wide constants.
        # self.MAX_REFINE_ITERS: Phase-2 secant refine iteration cap.
        # self.FLT_MAX / self.NEG_FLT_MAX: fp32 IEEE-754 max / negative-max
        # sentinels used as reduction identities and pad values.
        self.MAX_REFINE_ITERS = 15
        self.FLT_MAX = 3.4028235e38
        self.NEG_FLT_MAX = -self.FLT_MAX

        # --- op#26 R0 histogram-ladder admission (default ON) ---
        # enable_r0: replace the Phase-2 secant search with a single-pass
        #   multi-threshold "rung ladder" admission seeded by a 256-bin
        #   histogram over the prev-topK gathered values (P1b).
        #   DEFAULT True: validated on real DSv4/V3.2 decode-capture
        #   workloads (25-cell seq-len scan) where R0 wins 24/25 vs the
        #   secant baseline, geomean 1.33x (pro 128k 2.10x). Correctness is
        #   value-set-exact vs torch.topk (186/186 across dtype/K/N/BS/cluster
        #   + tie plateaus). The secant path is retained verbatim and remains
        #   reachable via enable_r0=False; it is the exact fallback for the
        #   large-N / cold-hint (low preIdx hit-rate) regime where R0 can
        #   regress on the synthetic worst axis — a follow-up PR adds a
        #   data-driven dispatch guard to route between the two. All R0 fields
        #   are const-foldable, so an enable_r0=False kernel is byte-identical
        #   to the pre-R0 upstream base.
        # r0_qfracs: descending h-space quantile fractions defining the M
        #   candidate rungs (ascending threshold values); None => no rungs.
        # r0_vseed: park P1's pmean (the secant init probe) as one extra
        #   "virtual seed" rung column in the M-ary count pass (no extra
        #   memory traffic or sync; the column reuses the secant per-thread
        #   count buffer, so SMEM does not grow). Adapts the admission
        #   ladder to the row's value distribution: fixes the fat-admission
        #   regime (a coarse quantile rung admitting ~kC candidates where
        #   pmean admits ~K) and donates a measured interior bracket point
        #   to the fallback refine on a full miss. None => enable_r0.
        # mt_unroll: 4-way unroll factor for block_count_ge_multi.
        # p1b_cache: stash the K gathered preIdx values in SMEM so P1b skips
        #   a second GMEM random gather (dtype-gated in a later commit).
        # fb_fix: R0-miss fallback re-measures the rung bracket ends before
        #   refining (excludes the R2-class unmeasured-seed failure mode).
        self.enable_r0 = bool(enable_r0)
        self.mt_unroll = int(mt_unroll)
        self.fb_fix = bool(fb_fix)
        # enable_block_skip: gate the R0 count pass and the Phase-3
        # stream-write on per-32-position upper bounds from the indexer
        # epilogue (block_max [num_rows, nb_pad*4] fp32; record r = max
        # over positions [r*32, r*32+32) of the stored logits). Lossless:
        # the active list is built at the loosest rung, so a skipped
        # block holds nothing >= any rung and every count equals its
        # dense value.
        self.enable_block_skip = bool(enable_block_skip)
        self.SKIP_BLOCK = 32
        self.SKIP_BLOCK_LOG2 = 5
        self.SKIP_MAX_BLOCKS = 8192  # smem active-list budget (32KB of local
        # ids); bounds block-skip to N_local <= 262144 at grain 32 - longer
        # rows run the dense fallback and owe nothing to block-skip
        self.SKIP_UNROLL = 2
        self.skip_order = "grouped"
        if enable_block_skip and num_threads not in (512, 1024):
            raise ValueError("enable_block_skip requires num_threads in {512, 1024}")
        if enable_block_skip and not enable_r0:
            # The compact machinery hangs off the R0 count pass and the
            # phase-3 stream-write; without R0 the 16KB list SMEM would be
            # allocated but the skip could never engage.
            raise ValueError("enable_block_skip requires enable_r0")
        # C7 dispatch (op#26 host policy folded into the ctor; all gated on
        # enable_r0 so an OFF kernel is byte-identical to the base):
        #  - qfracs default = M2D (0.85, 0.35): dispatch_r0_op26 ships M2D for
        #    every (dtype, K, N); the M=2 pass is ~free and the R1 falsi shot
        #    covers the 3-7% bracket misses. uh4 (M=4) was silicon-falsified
        #    (mc geomean 0.956 — admission != latency).
        #  - p1b_cache default is cs-aware:
        #      * cs>1 (cluster): ON for ALL dtypes. The SMEM gather-cache win
        #        holds and the fp32 occupancy regression that hurts the
        #        single-CTA path does NOT reproduce in the cluster kernel
        #        (latency-bound, different SMEM budget). nsys cs=4: K1024
        #        ~1.01x / K2048 ~1.02x / K512 wash, 0 losses, exact. Matches
        #        op26 dispatch_p1bc_mc (unconditional ON).
        #      * cs=1 (single-CTA): (dtype != fp32). The gather-cache wins
        #        +0.8-2.8% on 16-bit (random half-prec gather is the cost) but
        #        is flat/negative on fp32 (occupancy at kC=6144), so OFF there.
        #  - kC-diet: K512 single-CTA -> kC=3072 (saves 16KB SMEM; 16-bit win,
        #    fp32 neutral). kC>=2560 is the K512 16-bit tie-safety contract so
        #    3072 is safe; the cluster port and K1024/K2048 stay stock.
        if r0_vseed is None:
            r0_vseed = enable_r0
        if enable_r0 and r0_qfracs is None:
            # Per-K default (2026-07-16 vseed full-envelope audit, 2772
            # cells): with the virtual seed rung on, pmean covers q.35's
            # admission region for K512/K1024 (2 count columns = zero
            # column tax); K2048 keeps q.35 (kC/K = 2.5 makes a fat admit
            # costlier than a slim 2-pass miss). Without vseed, q.35 must
            # stay for all K (it is the only slim rung).
            # K2048 low rung 0.85 -> 0.6 (2026-07-19 real-content rung
            # recalibration + paired nsys cold-L2 A/B, B200): the shipped
            # 0.85 rung's admission straddles [K, kC] on real V3.2 decode
            # captures (bracket on 86% of steps -> one extra falsi pass);
            # 0.6 lands the first pass. Measured: real V3.2 geomean
            # +2.2-2.8% across fp32/bf16/fp16 and the full BS grid (8K
            # rung +10-13% at every BS, no loser cell), favorable
            # synthetic +9-11%, adverse synthetic wash, exact everywhere.
            # K512/K1024 unchanged: moving or widening their ladder
            # measured wash-to-loss (the extra count column costs 3-7%).
            if top_k == 2048:
                r0_qfracs = (0.6, 0.35) if r0_vseed else (0.85, 0.35)
            else:
                r0_qfracs = (0.85,) if r0_vseed else (0.85, 0.35)
        if enable_r0 and p1b_cache is None:
            if cluster_size > 1:
                p1b_cache = True
            else:
                p1b_cache = dtype != cutlass.Float32
        self.p1b_cache = bool(p1b_cache)
        # kc_diet: None → diet iff single-CTA (tuned default). The LB hybrid
        # kernel passes False for BOTH member instances so their SMEM layouts
        # stay byte-identical (the DSL sizes the launch from the last-traced
        # SmemAllocator only; see GvrTopKLBKernel).
        # pdl_wait_late: move the PDL wait past the prologue so that work
        # overlaps the producer's tail. Off by default: the entry-point
        # wait is what upstream emits.
        self.pdl_wait_late = bool(pdl_wait_late)
        if kc_diet is None:
            kc_diet = cluster_size == 1
        if enable_r0 and top_k == 512 and kc_diet and self.kC > 3072:
            self.kC = 3072
        # K2048 R0 Phase-4 histogram diet: 2048 -> 512 bins (2026-07-19
        # paired nsys cold-L2 A/B on B200, all cells exact). The P4 zero /
        # atomic build / serial scan all shrink 4x; the deeper boundary-bin
        # recursion costs less than the saved passes at kC=6144 candidates.
        # Measured vs this head: real V3.2 decode captures geomean +6.1%
        # (fp32) / +10.9% (bf16) / +6.3% (fp16); favorable synthetic
        # +5.2-11.0%, adverse synthetic +5.1-10.6%; no losing cell
        # (fp32 min 0.994, bf16 min 1.035, fp16 min 0.999). Gated on
        # enable_r0 so the retained secant path (which shares GvrParams
        # and its own P4 histogram) stays byte-identical. P1b reuses this
        # buffer and needs >= 256 bins, so 512 is safe. K512/K1024
        # measured as a wash under the same protocol and stay stock.
        if enable_r0 and top_k == 2048 and self.kNumBins > 512:
            self.kNumBins = 512
        # p4_fine_rangetest: filter the fine recursion by value range
        # instead of recomputing each candidate's bin. OFF - the two are
        # not fp32-equivalent, and the scatter still classifies by bin
        # recompute, so a candidate they disagree on is counted by one
        # pass and placed by the other, leaving an output slot unwritten.
        self.p4_fine_rangetest = False if p4_fine_rangetest is None else bool(p4_fine_rangetest)
        # p4_scat_rangetest: the scatter classifies each candidate as
        # above / inside / below the straddling bin and recomputed the
        # bin index to do it; the same value-range compare the fine
        # recursion uses answers it without the subtract, multiply and
        # two clamps per candidate.
        # DEFAULT OFF - same fp32 non-equivalence as p4_fine_rangetest,
        # in the opposite direction: the range test admits a candidate
        # the histogram binned elsewhere, so the scatter writes past the
        # rank it reserved and out-of-range indices reach the output.
        self.p4_scat_rangetest = bool(p4_scat_rangetest)
        self.r0_qfracs = tuple(float(q) for q in r0_qfracs) if r0_qfracs else ()
        if self.r0_qfracs:
            assert all(0.0 < q < 1.0 for q in self.r0_qfracs), self.r0_qfracs
            assert list(self.r0_qfracs) == sorted(self.r0_qfracs, reverse=True), (
                "r0_qfracs must be descending h (ascending threshold value)"
            )
        self.M_thr = len(self.r0_qfracs)
        # --- vseed (2026-07-16): fold P1's pmean (the secant init
        # probe) into the M-ary R0 count pass as one extra "virtual rung".
        # Fixes the flash-1M fat-admission regression (the coarse q.85 rung
        # admits ~4400 candidates where pmean admits ~630 -> 7x P3/P4 cand
        # cost) and, on a true miss, donates a measured interior bracket
        # point to the fallback refine. Const-folded: r0_vseed=False kernels
        # are byte-identical to before. M_qf = rungs P1b places from qneeds;
        # M_thr = total columns counted/admitted (M_qf + 1 when vseed).
        self.r0_vseed = bool(r0_vseed) and bool(enable_r0) and self.M_thr > 0
        self.M_qf = self.M_thr
        if self.r0_vseed:
            self.M_thr = self.M_qf + 1
        # need[m] = ceil(q_m * K) prev-topK values >= rung m.
        self.qneeds = tuple(max(1, int(math.ceil(q * self.top_k))) for q in self.r0_qfracs)
        # use_ext_counts (waterfall L1 admission): thresholds AND their
        # exact counts arrive from the indexer epilogue (interface v2) —
        # P1b and the M-ary count pass are skipped; an in-band rung is
        # re-measured ONCE through the seeded refine (per-thread hand-off)
        # and accepted; a miss seeds log-falsi with the external brackets.
        # emit_xstate: write the per-row closed-loop state at Phase 4 exit
        # (interface v2: [0] valid, [1] kth proxy, [2] accepted threshold,
        # [3] cand_count). cs==1 only (leader-gather rows land later).
        self.emit_xstate = bool(emit_xstate)
        # use_ext_cand (waterfall L2 direct-to-P4): pre-collected (value,
        # index) pairs from the epilogue land straight in smem_keys/vals —
        # no P1, no counting, no P3 scan. Eligible when void==0, claimed
        # <= cand_cap and the collect rung's exact count is in [K, kC];
        # ineligible rows fall through to the ext-counts path.
        self.use_ext_cand = bool(use_ext_cand)
        if kc_override is not None:
            # physical candidate-buffer capacity override (B* search)
            self.kC = int(kc_override)
        # acceptance band top B*: a cut whose count fits [K, B*] goes
        # straight to Phase 4. Physically bounded by kC.
        self.accept_cap = int(accept_cap) if accept_cap is not None else self.kC
        self.cand_cap = int(cand_cap)
        # list path: the score column is staged into a DEDICATED smem
        # region sized cand_cap fp32 (96KB at 24576). Budget note: this
        # coexists with everything except a simultaneously-enabled big
        # slice cache (128KB) - that combination exceeds the 227KB CTA
        # limit and fails loudly at compile time. Long rows (the list
        # path's target) cannot enable the slice cache anyway.
        # v5 bucketed layout: tensor width = 2 * accept_cap
        # (segments A, B) + segment-C capacity; the admission
        # bounds the ENTRY COUNT by C's capacity (the only
        # segment that can void).
        self.list_cap = max(0, int(cand_cap) - 2 * self.accept_cap)
        self.cand_rung = int(cand_rung)
        # self_scan (fused self-contained mode): the kernel streams the
        # row once against the three closed-loop lines, bucketing values
        # into on-chip segments (A >= t2 / B [t1,t2) / C [t0,t1) at bases
        # 0 / accept_cap / 2*accept_cap) and positions into cand_idx. A
        # line cut compacts the winning segment to the smem_keys prefix
        # and fills smem_vals with segment coordinates, after which the
        # list consumer runs unchanged. Ineligible rows take the stock
        # fallback.
        self.self_scan = bool(self_scan)
        if self.self_scan:
            if not use_ext_counts:
                raise ValueError("self_scan requires use_ext_counts")
            if use_ext_cand:
                raise ValueError("self_scan and use_ext_cand are exclusive")
            if dtype != cutlass.Float32:
                raise ValueError("self_scan is fp32-only (v1)")
            if self.enable_smem_cache:
                raise ValueError(
                    "self_scan and enable_smem_cache exceed the CTA smem budget together"
                )
            # on-chip segment budget: values only, 4B/entry; C sized so
            # keys(160KB) + vals(32KB) + hist + scratch stay under the
            # 227KB CTA limit with room for the stage-2 skip list.
            self.seg_total = 2 * self.accept_cap + int(cap_c if cap_c is not None else 16384)
            self.cap_c = self.seg_total - 2 * self.accept_cap
            # cp.async staging for the phase-0 dense scan: the pair-step
            # pipeline keeps 2 pairs x 2 slots in flight per thread, so
            # exactly 4 slot rows are required. The 64KB staging fits the
            # CTA budget only with the C segment trimmed to <= 16384
            # (keys 128KB + staging 64KB + hist/scratch); larger C would
            # silently overrun the alias, so reject it outright. Rows are
            # 16B slots; never fewer than vals holds so the alias always
            # covers it.
            if self.cap_c > 16384:
                raise ValueError("self_scan requires cap_c <= 16384 (staging budget)")
            self.stage_slots = 4
            self.stage_rows = max(self.stage_slots * self.num_threads, self.kC // 4)
        else:
            self.seg_total = self.kC
            self.cap_c = 0
        if use_ext_cand and not use_ext_counts:
            raise ValueError("use_ext_cand requires use_ext_counts")
        if (use_ext_counts or ext_rungs or use_ext_cand) and not enable_r0:
            # the effective flags below are and-ed with enable_r0; reject
            # instead of silently compiling the stock path
            raise ValueError("ext tiers require enable_r0")
        if use_ext_cand and self.list_cap <= 0:
            raise ValueError(
                f"cand_cap={cand_cap} leaves no C segment past 2*accept_cap={2 * self.accept_cap}"
            )
        # ext_rungs (two-pass variant B): closed-loop rung THRESHOLDS come
        # from the host (previous-step xstep lines); the kernel counts them
        # itself via the stock R0 multi-count and admits the tightest rung
        # in [K, kC]. Exclusive with use_ext_counts (which also imports
        # the counts and skips nothing else).
        self.ext_rungs = bool(ext_rungs) and bool(enable_r0)
        if self.ext_rungs and bool(use_ext_counts):
            raise ValueError("ext_rungs is exclusive with use_ext_counts")
        self.use_ext_counts = bool(use_ext_counts) and bool(enable_r0)
        if self.use_ext_counts:
            if not self.fb_fix:
                raise ValueError("use_ext_counts requires fb_fix")
            if self.M_thr != 3:
                raise ValueError("use_ext_counts expects exactly 3 seed rungs")
        if self.ext_rungs:
            if not self.fb_fix:
                raise ValueError("ext_rungs requires fb_fix")
            if self.M_thr != 3:
                raise ValueError("ext_rungs expects exactly 3 seed rungs")
            # cluster_size > 1 supported: the ext rungs/counts are
            # per-row (identical across the cluster), the stock multi
            # count pass cluster-merges as usual, and the L2 direct
            # loader runs leader-only (peers contribute zero candidates).
        # R1 inline shot aim in log2-count space: geometric center of the
        # [K, kC] acceptance window.
        self.log2_r1aim = math.log2(math.sqrt(self.top_k * self.kC)) if self.r0_qfracs else 0.0
        # fb_fix interior aim (HLS grid optimum): log2(K * (kC/K)**fb_alpha).
        self.log2_mstar = (
            math.log2(self.top_k * (self.kC / self.top_k) ** float(fb_alpha))
            if self.r0_qfracs
            else 0.0
        )

        # --- op#7 P4 fused rank-and-scatter (inert until enable_p4_rank_scatter) ---
        # Replaces phase4_histogram_snap's k-th-bin search + 2-pass writeback
        # with a single rank-and-scatter pass (op#7 PR#15709), cutting Phase-4
        # barriers ~14 -> ~7. On a latency-bound kernel that is a whole-kernel
        # win (~1.078x, HW-invariant). enable_p4_rank_scatter_exact adds ONE
        # fine-histogram recursion on the straddling coarse bin so the result is
        # bit-exact vs torch.topk (adds a few barriers back but still < snap).
        # Default ON with R0: nsys over the op22 4k-1M BS=1 best/worst envelope
        # gives geomean ~1.09x (K1024 1.12 / K2048 1.12 / K512 1.05) with NO
        # cell regressing >2%. Resolves to OFF when enable_r0 is False, so the
        # base kernel stays byte-identical to upstream.
        if enable_p4_rank_scatter is None:
            enable_p4_rank_scatter = bool(enable_r0)
        if enable_p4_rank_scatter_exact is None:
            enable_p4_rank_scatter_exact = bool(enable_p4_rank_scatter)
        self.enable_p4_rank_scatter = bool(enable_p4_rank_scatter)
        self.enable_p4_rank_scatter_exact = bool(enable_p4_rank_scatter_exact)
        # p4_exact_tail: ambiguity-gated exact tie-resolution for the fine
        # straddling bin (fp32 inputs only; see phase4_rank_scatter). The
        # fine recursion resolves values to range/(kNumBins*256); two fp32
        # values closer than that straddling the kK boundary inside one fine
        # bin were previously picked in arrival order (observed as |miss|=1
        # with |dv| ~ 3e-6 on real Pro 512k-ISL captures). Default ON for
        # fp32 rank-scatter-exact kernels; 16-bit inputs keep the arrival
        # fill (their upconverted keys are already fully resolved by the
        # two-level histogram, and 16-bit tie plateaus are bitwise-equal,
        # where arrival order is value-exact).
        if p4_exact_tail is None:
            p4_exact_tail = self.enable_p4_rank_scatter_exact and dtype == cutlass.Float32
        self.p4_exact_tail = bool(p4_exact_tail) and self.enable_p4_rank_scatter_exact
        # p4_tail_fast: tiny-tie COLLECT+SELECT fast path inside the
        # exact-tail fire branch: when the boundary tie class holds few
        # enough entries to buffer, one candidate pass replaces the radix
        # passes; larger classes fall through to the radix backstop, so
        # exactness is identical either way.
        if p4_tail_fast is None:  # [p4tt]
            # default follows p4_exact_tail for every K (the compacted
            # path pays for the boundary class only)
            p4_tail_fast = self.p4_exact_tail
        self.p4_tail_fast = bool(p4_tail_fast) and self.p4_exact_tail  # [p4tt]
        # p4_tail_v3: compacted-class repair (block-parallel radix +
        # pure-tie pre-check) in place of the stock thread0 serial
        # select. Default follows the configuration: a stock kernel gets
        # upstream's body, any emission-assisted one gets the rewrite.
        if p4_tail_v3 is None:
            p4_tail_v3 = bool(
                use_ext_counts or use_ext_cand or ext_rungs or self_scan or enable_block_skip
            )
        self.p4_tail_v3 = bool(p4_tail_v3)
        # p4_no_fine: drop the 256-bin fine level from phase 4 and let the
        # tail repair rank the whole straddling COARSE bin instead. A class
        # past the tail's pair buffer falls into its radix, which handles
        # any size. Gated to the same configuration as that small-class
        # route; the stock kernel keeps the fine level so its codegen is
        # untouched.
        if p4_no_fine is None:
            p4_no_fine = bool(
                self.p4_exact_tail and self.p4_tail_fast and self.p4_tail_v3
            ) and not (self.p4_fine_rangetest or self.p4_scat_rangetest)
        self.p4_no_fine = bool(p4_no_fine)
        if self.p4_no_fine:
            if not (self.p4_exact_tail and self.p4_tail_fast and self.p4_tail_v3):
                raise ValueError("p4_no_fine requires the exact-tail repair chain")
            if self.p4_fine_rangetest or self.p4_scat_rangetest:
                raise ValueError("p4_no_fine is incompatible with the range-test arms")
        if self.p4_exact_tail and self.p4_tail_fast and self.p4_tail_v3:
            if _pair_cap_for(self.kNumBins) < 1:
                raise ValueError(
                    f"kNumBins={self.kNumBins} leaves no room for the tail pair buffer"
                )
        # p1r_rescue: rebuild the refine bracket from the row when the
        # seed bracket is degenerate (e.g. the zero-init prev_topk every
        # request's first decode step feeds). ON by default.
        self.p1r_rescue = bool(p1r_rescue)

    # ------------------------------------------------------------------
    # SMEM slice cache loader. Streams this CTA's slice GMEM → SMEM via
    # LDG → STS so Phase 2/3 can read LDS instead of re-streaming GMEM.
    # Iteration pattern mirrors block_count_ge's scan so the SMEM layout
    # is naturally aligned for subsequent LDS reads
    # (smem_input[i] == input_row[slice_start + i]).
    # ------------------------------------------------------------------
    @cute.jit
    def load_slice_to_smem(
        self,
        input_row,
        slice_start,
        slice_end,
        smem_input,
        tidx,
    ):
        num_threads = cutlass.const_expr(self.num_threads)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        step_elem = cutlass.const_expr(num_threads * vec_w)

        copy_atom = self._make_load_copy_atom()
        row_addr = input_row.iterator.toint()
        smem_addr = smem_input.iterator.toint()

        slice_len = slice_end - slice_start
        i_local = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)
        n_aligned_local = (slice_len // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)

        # Vectorized GMEM→SMEM load via 4-way LDG.E.* unroll, mirroring
        # block_count_ge's fast path. ic_local indexes both GMEM
        # (input_row[slice_start + ic_local]) and SMEM (smem_input[ic_local]).
        if self.enable_unroll_4:
            rng_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
            big_iters = cutlass.Int32(0)
            if slice_len > i_local + cutlass.Int32(vec_w - 1):
                big_iters = (slice_len - i_local - cutlass.Int32(vec_w)) // cutlass.Int32(
                    step_elem
                ) + cutlass.Int32(1)
            for k in cutlass.range(big_iters, unroll=4):
                ic_local = i_local + k * cutlass.Int32(step_elem)
                src_ptr = cute.make_ptr(
                    self.dtype,
                    row_addr + cutlass.Int64(slice_start + ic_local) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.gmem,
                    assumed_align=vec_align,
                )
                src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
                cute.copy(copy_atom, src, rng_frag)
                dst_ptr = cute.make_ptr(
                    self.dtype,
                    smem_addr + cutlass.Int64(ic_local) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.smem,
                    assumed_align=vec_align,
                )
                dst = cute.make_tensor(dst_ptr, cute.make_layout((vec_w,)))
                cute.copy(copy_atom, rng_frag, dst)
            i_local = i_local + big_iters * cutlass.Int32(step_elem)

        # 1-way tail vec loop (slice_len mod step_elem residual).
        tail_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
        while i_local + cutlass.Int32(vec_w - 1) < slice_len:
            src_ptr = cute.make_ptr(
                self.dtype,
                row_addr + cutlass.Int64(slice_start + i_local) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.gmem,
                assumed_align=vec_align,
            )
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            dst_ptr = cute.make_ptr(
                self.dtype,
                smem_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.smem,
                assumed_align=vec_align,
            )
            dst = cute.make_tensor(dst_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, tail_frag, dst)
            i_local = i_local + step

        # Scalar tail (slice_len % vec_w). Each thread strides by num_threads.
        it_local = n_aligned_local + tidx
        while it_local < slice_len:
            smem_input[it_local] = input_row[slice_start + it_local]
            it_local = it_local + cutlass.Int32(num_threads)

        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Build a vectorized copy atom for the input scan loops. With
    # use_constant_hint=True we use CopyG2ROp+invariant to get
    # xxx.E.*.CONSTANT (read-only cache, matches CUDA __ldg). Defined as
    # a plain Python method (not @cute.jit) so the if-else branches both
    # bind copy_atom in the same trace scope.
    # ------------------------------------------------------------------
    def _make_load_copy_atom(self):
        # num_bits_per_copy matches self.vec_bits (128 default; 256 when
        # use_256bit_load=True).
        if self.use_constant_hint:
            return cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(),
                self.dtype,
                num_bits_per_copy=self.vec_bits,
                invariant=True,
            )
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=self.vec_bits,
        )

    # ------------------------------------------------------------------
    # Input load helper — casts to fp32 regardless of self.dtype.
    # ------------------------------------------------------------------
    @cute.jit
    def _load_fp32(self, ptr_view, idx):
        # TODO: instructions?
        v = ptr_view[idx]
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            return v
        else:
            return cutlass.Float32(v)

    # ------------------------------------------------------------------
    # Warp-level reductions
    #
    # ------------------------------------------------------------------
    @cute.jit
    def warp_reduce_sum_i32(self, val):
        # REDUX.SYNC.ADD.S32 (sm_80+)
        return cute.arch.warp_redux_sync(val, "add")

    @cute.jit
    def warp_reduce_sum_f32(self, val):
        # PTX redux.sync has no fadd.
        # will lower to SHFL.BFLY 5-step tree.
        return cute.arch.warp_reduction_sum(val)

    @cute.jit
    def warp_reduce_min_f32(self, val):
        # PTX redux.sync.fmin.f32 (sm_100).
        return cute.arch.warp_redux_sync(val, "fmin")

    @cute.jit
    def warp_reduce_max_f32(self, val):
        # PTX redux.sync.fmax.f32 (sm_100).
        return cute.arch.warp_redux_sync(val, "fmax")

    # ------------------------------------------------------------------
    # Raw-address SMEM scalar access through a pre-hoisted window base.
    #
    # Tensor-indexed SMEM access (smem_keys[i]) makes the compiler
    # re-derive the cluster SMEM window per access (S2R SR_CgaCtaId +
    # LEA<<24) — ncu shows this as the top single-instruction stall in
    # the P3 stream-write and P4 snap loops. Hoisting the base once via
    # iterator.toint() (one S2R per call site) turns every subsequent
    # access into plain integer addressing — the same pattern the P2
    # scan loops already use for smem_input, whose SASS regions show no
    # S2R at all.
    # ------------------------------------------------------------------
    @cute.jit
    def _smem_ref(self, dtype: cutlass.Constexpr, base_addr, idx):
        elem_bytes = cutlass.const_expr(dtype.width // 8)
        p = cute.make_ptr(
            dtype,
            base_addr + cutlass.Int64(idx) * cutlass.Int64(elem_bytes),
            cute.AddressSpace.smem,
            assumed_align=4,
        )
        return cute.make_tensor(p, cute.make_layout((1,)))

    @cute.jit
    def _smem_ld(self, dtype: cutlass.Constexpr, base_addr, idx):
        return self._smem_ref(dtype, base_addr, idx)[0]

    @cute.jit
    def _smem_st(self, dtype: cutlass.Constexpr, base_addr, idx, val):
        t = self._smem_ref(dtype, base_addr, idx)
        t[0] = val

    # ------------------------------------------------------------------
    # Phase 1: preIdx Min/Max/Mean -> initial threshold
    # ------------------------------------------------------------------
    @cute.jit
    def phase1_preidx_stats(
        self,
        input_row,  # cute.Tensor [N] fp32 (post-cast for half-prec)
        N,  # length of input_row
        pre_idx_row,  # cute.Tensor [M] int32
        pre_idx_count,
        pre_idx_offset,
        smem_wmin_f32,  # cute.Tensor [NUM_WARPS] float32
        smem_wmax_f32,  # cute.Tensor [NUM_WARPS] float32
        smem_wsum_f32,  # cute.Tensor [NUM_WARPS] float32
        smem_wcnt_i32,  # cute.Tensor [NUM_WARPS] int32
        s_thr,  # cute.Tensor [3] float32: [threshold, val_lo, val_hi]
        s_iscalars,  # cute.Tensor [6] int32: [cand_count, done, cnt_lo, cnt_hi, out_count, local_cand_count]
        tidx,
        warp_id,
        lane,
        smem_gath=None,  # cute.Tensor [top_k] f32 or None (p1b_cache): stash
        # the gathered value per preIdx slot so P1b skips a 2nd GMEM gather.
        s_mt_thr=None,  # r0_vseed: P1 also parks pmean in the last rung
        # column (visibility via P1's own trailing barrier -> zero extra sync).
    ):
        """preIdx scan + warp reduce + block aggregate + initial threshold.

        Smem layout split: floats kept in fp32 buffers, ints kept in int32
        buffers (no bit-cast tricks needed — avoids ArithValue/ir_value
        coupling and keeps types clean for the MLIR codegen).
        """
        local_min = cutlass.Float32(self.FLT_MAX)
        local_max = cutlass.Float32(self.NEG_FLT_MAX)
        local_sum = cutlass.Float32(0.0)
        local_cnt = cutlass.Int32(0)

        # Stride loop over preIdx with pre_idx_offset shift. pre_idx_count
        # is compile-time (= top_k). Two cases:
        #   K >= num_threads: every thread loads ≥1 preIdx; fully unrolled
        #     over n_iters = K // num_threads.
        #   K <  num_threads: only the first K threads load (guard below);
        #     remaining threads contribute identity values.
        if cutlass.const_expr(pre_idx_count >= self.num_threads):
            n_iters = cutlass.const_expr(pre_idx_count // self.num_threads)
            for u in cutlass.range_constexpr(n_iters):
                i = tidx + cutlass.Int32(u * self.num_threads)
                raw = pre_idx_row[i]
                idx = raw + pre_idx_offset
                if cutlass.const_expr(smem_gath is not None):
                    smem_gath[i] = cutlass.Float32(self.NEG_FLT_MAX)
                if idx >= 0 and idx < N:
                    v = self._load_fp32(input_row, idx)
                    if cutlass.const_expr(smem_gath is not None):
                        smem_gath[i] = v
                    local_max = cute.arch.fmax(local_max, v)
                    local_min = _fmin_f32_inline(local_min, v)
                    local_sum = local_sum + v
                    local_cnt = local_cnt + 1
        else:
            # K < num_threads — only first K threads load a preIdx.
            # cute DSL requires variables to exist before dynamic `if` blocks,
            # so predeclare `idx` with an out-of-range sentinel and update
            # it conditionally; the downstream `if idx >= 0 and idx < N`
            # gate handles the sentinel naturally.
            idx = cutlass.Int32(-1)
            if tidx < cutlass.Int32(pre_idx_count):
                idx = pre_idx_row[tidx] + pre_idx_offset
                if cutlass.const_expr(smem_gath is not None):
                    smem_gath[tidx] = cutlass.Float32(self.NEG_FLT_MAX)
            if idx >= 0 and idx < N:
                v = self._load_fp32(input_row, idx)
                if cutlass.const_expr(smem_gath is not None):
                    smem_gath[tidx] = v
                local_max = cute.arch.fmax(local_max, v)
                local_min = _fmin_f32_inline(local_min, v)
                local_sum = local_sum + v
                local_cnt = local_cnt + 1

        # Warp-level reductions + smem write. When K < num_threads only
        # the first ``active_preidx_warps`` warps have real data — skip
        # the rest to save ~30 cy/warp. K ∈ {512, 1024, 2048} divides
        # evenly by WARP_SIZE, so the clamp to num_warps just handles
        # K > num_threads (avoids OOB into smem[num_warps]).
        active_preidx_warps = cutlass.const_expr(
            min(pre_idx_count // self.WARP_SIZE, self.num_warps)
        )
        if cutlass.const_expr(active_preidx_warps < self.num_warps):
            if warp_id < cutlass.Int32(active_preidx_warps):
                wmin = self.warp_reduce_min_f32(local_min)
                wmax = self.warp_reduce_max_f32(local_max)
                wsum = self.warp_reduce_sum_f32(local_sum)
                wcnt = self.warp_reduce_sum_i32(local_cnt)
                if lane == 0:
                    smem_wmin_f32[warp_id] = wmin
                    smem_wmax_f32[warp_id] = wmax
                    smem_wsum_f32[warp_id] = wsum
                    smem_wcnt_i32[warp_id] = wcnt
        else:
            wmin = self.warp_reduce_min_f32(local_min)
            wmax = self.warp_reduce_max_f32(local_max)
            wsum = self.warp_reduce_sum_f32(local_sum)
            wcnt = self.warp_reduce_sum_i32(local_cnt)
            if lane == 0:
                smem_wmin_f32[warp_id] = wmin
                smem_wmax_f32[warp_id] = wmax
                smem_wsum_f32[warp_id] = wsum
                smem_wcnt_i32[warp_id] = wcnt
        cute.arch.barrier()

        # Block aggregate: 4 reductions across num_warps slots. Warp-parallel
        # path is gated by enable_warp_parallel_reduce (auto-on at 32 warps,
        # off at 16 warps — see __init__).
        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            # Warp-parallel 4-way reduce in warp 0. Only the first
            # `active_preidx_warps` slots are read (dummy warps skipped).
            if warp_id == cutlass.Int32(0):
                v_min = cutlass.Float32(self.FLT_MAX)
                v_max = cutlass.Float32(self.NEG_FLT_MAX)
                v_sum = cutlass.Float32(0.0)
                v_cnt = cutlass.Int32(0)
                if lane < cutlass.Int32(active_preidx_warps):
                    v_min = smem_wmin_f32[lane]
                    v_max = smem_wmax_f32[lane]
                    v_sum = smem_wsum_f32[lane]
                    v_cnt = smem_wcnt_i32[lane]
                pmin = self.warp_reduce_min_f32(v_min)
                pmax = self.warp_reduce_max_f32(v_max)
                psum = self.warp_reduce_sum_f32(v_sum)
                pcnt = self.warp_reduce_sum_i32(v_cnt)
                if lane == cutlass.Int32(0):
                    pmean = cutlass.Float32(0.0)
                    if pcnt > 0:
                        pmean = psum / cutlass.Float32(pcnt)
                    else:
                        pmean = (pmin + pmax) * cutlass.Float32(0.5)
                    cnt_lo_seed = pre_idx_count + (pre_idx_count >> 2)
                    s_thr[0] = pmean
                    if cutlass.const_expr(self.r0_vseed):
                        s_mt_thr[self.M_thr - 1] = pmean
                    s_thr[1] = pmin
                    s_thr[2] = pmax
                    s_iscalars[0] = cutlass.Int32(0)  # cand_count
                    s_iscalars[1] = cutlass.Int32(0)  # done
                    s_iscalars[2] = cutlass.Int32(cnt_lo_seed)  # cnt_lo
                    s_iscalars[3] = cutlass.Int32(1)  # cnt_hi
                    s_iscalars[4] = cutlass.Int32(0)  # out_count
        else:
            # tid==0 serial loop.
            if tidx == 0:
                pmin = cutlass.Float32(self.FLT_MAX)
                pmax = cutlass.Float32(self.NEG_FLT_MAX)
                psum = cutlass.Float32(0.0)
                pcnt = cutlass.Int32(0)
                # Iterate over active_preidx_warps (= num_warps when K >=
                # num_threads; smaller when K < num_threads since dummy warps
                # above no longer write smem).
                for w in cutlass.range_constexpr(active_preidx_warps):
                    v_min = smem_wmin_f32[w]
                    v_max = smem_wmax_f32[w]
                    v_sum = smem_wsum_f32[w]
                    v_cnt = smem_wcnt_i32[w]
                    pmax = cute.arch.fmax(pmax, v_max)
                    pmin = _fmin_f32_inline(pmin, v_min)
                    psum = psum + v_sum
                    pcnt = pcnt + v_cnt

                pmean = cutlass.Float32(0.0)
                if pcnt > 0:
                    pmean = psum / cutlass.Float32(pcnt)
                else:
                    pmean = (pmin + pmax) * cutlass.Float32(0.5)

                cnt_lo_seed = pre_idx_count + (pre_idx_count >> 2)
                s_thr[0] = pmean
                if cutlass.const_expr(self.r0_vseed):
                    s_mt_thr[self.M_thr - 1] = pmean
                s_thr[1] = pmin
                s_thr[2] = pmax
                s_iscalars[0] = cutlass.Int32(0)
                s_iscalars[1] = cutlass.Int32(0)
                s_iscalars[2] = cutlass.Int32(cnt_lo_seed)
                s_iscalars[3] = cutlass.Int32(1)
                s_iscalars[4] = cutlass.Int32(0)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # P1r — degenerate-seed rescue: rebuild the refine bracket from the
    # data itself. Runs only when the preIdx gather produced an unusable
    # bracket (duplicate or invalid preIdx: cold-start zero-init slots,
    # stale slots pointing past N, or an all-tied gather). A full-row
    # min/max restores the P2 invariant count(>= v_lo) >= K, so the
    # normal pipeline stays exact; the extra row scan is paid only by
    # the (rare) degenerate rows. Bounds are clamped to +-FLT_MAX/2 so
    # the secant range arithmetic stays finite against inf-laden rows.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1r_data_reseed(
        self,
        input_row,  # cute.Tensor [N] (row-major slice of the row)
        N,  # runtime row length
        smem_wmin_f32,  # cute.Tensor [NUM_WARPS] float32 (reused P1 buffer)
        smem_wmax_f32,  # cute.Tensor [NUM_WARPS] float32 (reused P1 buffer)
        s_thr,  # cute.Tensor [3] float32: [threshold, val_lo, val_hi]
        s_iscalars,  # [cand_count, done, cnt_lo, cnt_hi, out_count, ...]
        s_mt_thr,  # rung columns (r0_vseed parks the seed line here)
        tidx,
        warp_id,
        lane,
    ):
        local_min = cutlass.Float32(self.FLT_MAX)
        local_max = cutlass.Float32(self.NEG_FLT_MAX)
        i = cutlass.Int32(tidx)
        while i < N:
            v = self._load_fp32(input_row, i)
            local_max = cute.arch.fmax(local_max, v)
            local_min = _fmin_f32_inline(local_min, v)
            i = i + cutlass.Int32(self.num_threads)
        wmin = self.warp_reduce_min_f32(local_min)
        wmax = self.warp_reduce_max_f32(local_max)
        if lane == 0:
            smem_wmin_f32[warp_id] = wmin
            smem_wmax_f32[warp_id] = wmax
        cute.arch.barrier()
        if tidx == 0:
            rmin = cutlass.Float32(self.FLT_MAX)
            rmax = cutlass.Float32(self.NEG_FLT_MAX)
            for w in cutlass.range_constexpr(self.num_warps):
                rmax = cute.arch.fmax(rmax, smem_wmax_f32[w])
                rmin = _fmin_f32_inline(rmin, smem_wmin_f32[w])
            # finite clamp keeps rng = val_hi - val_lo representable; rows
            # with mass beyond +-FLT_MAX/2 are adversarial-only (production
            # indexer scores are small finite values).
            rmin = cute.arch.fmax(rmin, cutlass.Float32(self.NEG_FLT_MAX * 0.5))
            rmax = _fmin_f32_inline(rmax, cutlass.Float32(self.FLT_MAX * 0.5))
            mid = (rmin + rmax) * cutlass.Float32(0.5)
            s_thr[0] = mid
            s_thr[1] = rmin
            s_thr[2] = rmax
            if cutlass.const_expr(self.r0_vseed):
                s_mt_thr[self.M_thr - 1] = mid
            s_iscalars[0] = cutlass.Int32(0)  # cand_count
            s_iscalars[1] = cutlass.Int32(0)  # done
            s_iscalars[2] = N  # cnt_lo: count(>= row min) = N, truthful
            s_iscalars[3] = cutlass.Int32(1)  # cnt_hi seed (same as P1)
            s_iscalars[4] = cutlass.Int32(0)  # out_count
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # P1b — 256-bin SMEM histogram over the prev-topK gathered values
    # (band [v_lo, v_hi] = P1's pmin/pmax = s_thr[1]/s_thr[2]), then M
    # h-space quantile rungs into s_mt_thr (ascending value order). Reuses
    # the Phase-4 smem_hist buffer (kNumBins >= 512 >= 256 in every spec;
    # Phase 4 re-zeroes it later). Provides the R0 admission placement; it
    # is only invoked from the enable_r0 path (added in a follow-up commit),
    # so the base kernel is unaffected.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1b_hspace_rungs(
        self,
        input_row,
        N,
        pre_idx_row,
        pre_idx_count,
        pre_idx_offset,
        smem_hist,
        s_thr,
        s_mt_thr,
        tidx,
        warp_id,
        lane,
    ):
        M = cutlass.const_expr(self.M_qf)
        NB = cutlass.const_expr(256)
        SEG = cutlass.const_expr(8)  # NB / WARP_SIZE bins per lane
        num_threads = cutlass.const_expr(self.num_threads)

        jz = tidx
        while jz < cutlass.Int32(NB):
            smem_hist[jz] = cutlass.Int32(0)
            jz = jz + cutlass.Int32(num_threads)
        cute.arch.barrier()

        v_lo = s_thr[1]
        v_hi = s_thr[2]
        width = (v_hi - v_lo) / cutlass.Float32(NB)  # caller guards v_hi > v_lo
        inv_w = cutlass.Float32(1.0) / width

        ig = tidx
        while ig < cutlass.Int32(pre_idx_count):
            idx = pre_idx_row[ig] + pre_idx_offset
            if idx >= cutlass.Int32(0) and idx < N:
                v = cutlass.Float32(input_row[idx])
                bf = (v - v_lo) * inv_w
                b = cutlass.Int32(bf)
                if b < cutlass.Int32(0):
                    b = cutlass.Int32(0)
                if b > cutlass.Int32(NB - 1):
                    b = cutlass.Int32(NB - 1)
                atomicAdd(smem_hist.iterator + b, cutlass.Int32(1))
            ig = ig + cutlass.Int32(num_threads)
        cute.arch.barrier()

        # Warp-0-parallel rung extraction (a tid0 256-bin serial walk is a
        # ~10-15us per-CTA dependency chain). Lane l owns the SEG consecutive
        # bins descending from bin NB-1-l*SEG; segment sums -> 5-step shfl_up
        # inclusive scan gives each lane the cumulative count of all
        # higher-value bins; each lane then walks its SEG bins once and fires
        # rung m at the unique crossing bin (cum_before < qneeds[m] <=
        # cum_at). qfracs descending in h => thresholds ascending in m.
        if warp_id == cutlass.Int32(0):
            top = cutlass.Int32(NB - 1) - lane * cutlass.Int32(SEG)
            seg_frag = cute.make_rmem_tensor((SEG,), cutlass.Int32)
            part = cutlass.Int32(0)
            for j in cutlass.range_constexpr(SEG):
                v8 = smem_hist[top - cutlass.Int32(j)]
                seg_frag[j] = v8
                part = part + v8
            tp = part
            for off_i in cutlass.range_constexpr(5):
                off_v = cutlass.const_expr(1 << off_i)
                other = cute.arch.shuffle_sync_up(tp, off_v, mask_and_clamp=0)
                if lane >= cutlass.Int32(off_v):
                    tp = tp + other
            excl = tp - part  # cum of all bins above my segment
            total = cute.arch.shuffle_sync(tp, cutlass.Int32(self.WARP_SIZE - 1))
            run = cutlass.Int32(0)
            for j in cutlass.range_constexpr(SEG):
                run = run + seg_frag[j]
                cum_at = excl + run
                cum_before = cum_at - seg_frag[j]
                for m in cutlass.range_constexpr(M):
                    if cum_at >= cutlass.Int32(self.qneeds[m]) and cum_before < cutlass.Int32(
                        self.qneeds[m]
                    ):
                        s_mt_thr[m] = v_lo + cutlass.Float32(top - cutlass.Int32(j)) * width
            # unfired rungs (heavy invalid-preIdx rows: total < need): v_lo
            if lane == 0:
                for m in cutlass.range_constexpr(M):
                    if total < cutlass.Int32(self.qneeds[m]):
                        s_mt_thr[m] = v_lo
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # P1b (p1b_cache variant) — build the rung histogram from the SMEM
    # gathered values that P1 stashed (smem_gath), skipping P1b's second
    # GMEM random gather. Sentinel NEG_FLT_MAX marks invalid/out-of-range
    # preIdx slots. Rung extraction is identical to phase1b_hspace_rungs.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1b_hspace_rungs_cached(
        self, pre_idx_count, smem_gath, smem_hist, s_thr, s_mt_thr, tidx, warp_id, lane
    ):
        M = cutlass.const_expr(self.M_qf)
        NB = cutlass.const_expr(256)
        SEG = cutlass.const_expr(8)
        num_threads = cutlass.const_expr(self.num_threads)

        jz = tidx
        while jz < cutlass.Int32(NB):
            smem_hist[jz] = cutlass.Int32(0)
            jz = jz + cutlass.Int32(num_threads)
        cute.arch.barrier()

        v_lo = s_thr[1]
        v_hi = s_thr[2]
        width = (v_hi - v_lo) / cutlass.Float32(NB)
        inv_w = cutlass.Float32(1.0) / width

        ig = tidx
        while ig < cutlass.Int32(pre_idx_count):
            v = smem_gath[ig]
            if v > cutlass.Float32(self.NEG_FLT_MAX):
                bf = (v - v_lo) * inv_w
                b = cutlass.Int32(bf)
                if b < cutlass.Int32(0):
                    b = cutlass.Int32(0)
                if b > cutlass.Int32(NB - 1):
                    b = cutlass.Int32(NB - 1)
                atomicAdd(smem_hist.iterator + b, cutlass.Int32(1))
            ig = ig + cutlass.Int32(num_threads)
        cute.arch.barrier()

        if warp_id == cutlass.Int32(0):
            top = cutlass.Int32(NB - 1) - lane * cutlass.Int32(SEG)
            seg_frag = cute.make_rmem_tensor((SEG,), cutlass.Int32)
            part = cutlass.Int32(0)
            for j in cutlass.range_constexpr(SEG):
                v8 = smem_hist[top - cutlass.Int32(j)]
                seg_frag[j] = v8
                part = part + v8
            tp = part
            for off_i in cutlass.range_constexpr(5):
                off_v = cutlass.const_expr(1 << off_i)
                other = cute.arch.shuffle_sync_up(tp, off_v, mask_and_clamp=0)
                if lane >= cutlass.Int32(off_v):
                    tp = tp + other
            excl = tp - part
            total = cute.arch.shuffle_sync(tp, cutlass.Int32(self.WARP_SIZE - 1))
            run = cutlass.Int32(0)
            for j in cutlass.range_constexpr(SEG):
                run = run + seg_frag[j]
                cum_at = excl + run
                cum_before = cum_at - seg_frag[j]
                for m in cutlass.range_constexpr(M):
                    if cum_at >= cutlass.Int32(self.qneeds[m]) and cum_before < cutlass.Int32(
                        self.qneeds[m]
                    ):
                        s_mt_thr[m] = v_lo + cutlass.Float32(top - cutlass.Int32(j)) * width
            if lane == 0:
                for m in cutlass.range_constexpr(M):
                    if total < cutlass.Int32(self.qneeds[m]):
                        s_mt_thr[m] = v_lo
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # block_count_ge — GE-count of input vs threshold (shared by P2/P3).
    # Per-thread strided accumulate → smem_ptcnt[tid] (for P3 prefix sum)
    # → warp reduce → block reduce → s_iscalars[0] = cand_count.
    # Optionally DSMEM-aggregates across the cluster.
    # ------------------------------------------------------------------
    @cute.jit
    def phase0_scan_bucket(
        self,
        input_row,  # cute.Tensor [N] dtype (fp32; full row, cs==1 only)
        N,  # int32 valid length (pad tail beyond N is never read)
        seed_thr_row,  # [3] fp32 closed-loop lines, ascending t0 < t1 < t2
        smem_keys,  # [seg_total] fp32 value segments (A @0 / B @segA / C @2segA)
        cand_idx_row,  # [seg_total] int32 gmem POSITION column (write-only here)
        block_max_row,  # [nb_pad] fp32 per-32-position maxima, or None
        smem_stage,  # [stage_rows, 4] fp32 cp.async staging (aliases smem_vals)
        s_seg,  # [>=7] int32 scratch (reuses smem_wcnt_p1: P1 never runs
        #         on a row this phase succeeded on): [0..2] A/B/C claim
        #         cursors, [3] void, [4] n0, [5] n1, [6] n2
        tidx,
        warp_id,
        lane,
    ):
        """self_scan phase 0: ONE streaming pass buckets every element
        >= t0 into the on-chip value segments (tightest line passed picks
        the segment; a full segment spills to the next looser one) and
        writes each entry's POSITION to the same coordinate of the gmem
        column. The final cursor values ARE the line counts (attempts,
        uncapped), so {n0, void, n1, n2} fall out for free — the same
        contract the v5 emitter produced externally.

        The dense scan is a cp.async pipeline: each thread streams one
        16B vector per step into its private slot-major smem staging
        slot, keeping ``stage_slots`` steps in flight; classification
        reads the staged values and claims passers with per-element
        direct smem atomics (they don't synchronize the warp). Segment
        overflow is resolved in-claim by spilling to the next looser
        segment (a segment overflows at most once per row)."""
        num_threads = cutlass.const_expr(self.num_threads)
        segA = cutlass.const_expr(self.accept_cap)
        capC = cutlass.const_expr(self.cap_c)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        p0ck0 = cutlass.Int64(0)
        if cutlass.const_expr(_P4_SUB_DBG):
            p0ck0 = cute.arch.clock64()
        if tidx == cutlass.Int32(0):
            s_seg[0] = cutlass.Int32(0)
            s_seg[1] = cutlass.Int32(0)
            s_seg[2] = cutlass.Int32(0)
        cute.arch.barrier()
        t0_s = seed_thr_row[0]
        t1_s = seed_thr_row[1]
        t2_s = seed_thr_row[2]
        row_addr = input_row.iterator.toint()
        # ---- stage-2 BLOCK-SKIP variant: the GEMM tail left per-32-
        # position maxima; a block whose max < t0 contributes nothing to
        # any count or segment, so it is never read. One warp per block:
        # the bmax compare is warp-uniform (all lanes read the same
        # scalar), a passing block is one coalesced 128B load, claims
        # are the same non-synchronizing per-element atomics as the
        # dense loop - so the block loop needs no uniform trip counts.
        if cutlass.const_expr(self.enable_block_skip and block_max_row is not None):
            # 8 blocks per warp iteration: every lane vector-loads the
            # SAME 8 bmax values (L1 broadcast - the pass decisions are
            # warp-uniform registers), then the passing blocks are
            # loaded back-to-back as independent 128B coalesced loads.
            bm_addr = block_max_row.iterator.toint()
            nb0 = (N + cutlass.Int32(31)) >> cutlass.Int32(5)
            # Two-pass skip: (1) DENSE-scan the bmax array itself (it
            # is 1/32 of the row) with the tuned vector loop, compacting
            # PASSING BLOCK IDS into the idle C segment (single-band mode
            # never fills C; ids < 2^23 store exactly as floats);
            # (2) walk the compact list, 8 blocks per warp round issued
            # unguarded back-to-back. If the list overflows capC the row
            # falls back to the dense full scan.
            # pass-1 vectors: 128-bit (the bmax row base is only 16B
            # aligned: nb_pad %% 4)
            pass1_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float32,
                num_bits_per_copy=128,
            )
            p1w = cutlass.const_expr(4)
            frag_p = cute.make_rmem_tensor((p1w,), cutlass.Float32)
            nfb0 = nb0 >> cutlass.const_expr((num_threads * 4).bit_length() - 1)
            itp0 = cutlass.Int32(0)
            while itp0 < nfb0:
                ip0 = (itp0 * cutlass.Int32(num_threads) + tidx) * cutlass.Int32(p1w)
                pp0 = cute.make_ptr(
                    cutlass.Float32,
                    bm_addr + cutlass.Int64(ip0) * cutlass.Int64(4),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                cute.copy(
                    pass1_atom,
                    cute.make_tensor(pp0, cute.make_layout((p1w,))),
                    frag_p,
                )
                for _jp in cutlass.range_constexpr(p1w):
                    if cutlass.Float32(frag_p[_jp]) >= t2_s:
                        slp0 = atomicAdd(s_seg.iterator + cutlass.Int32(1), cutlass.Int32(1))
                        if slp0 < cutlass.Int32(capC):
                            smem_keys[cutlass.Int32(2 * segA) + slp0] = cutlass.Float32(
                                ip0 + cutlass.Int32(_jp)
                            )
                itp0 = itp0 + cutlass.Int32(1)
            ptb0 = nfb0 * cutlass.Int32(num_threads * 4) + tidx
            while ptb0 < nb0:
                bpt0 = cute.make_ptr(
                    cutlass.Float32,
                    bm_addr + cutlass.Int64(ptb0) * cutlass.Int64(4),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                if cute.make_tensor(bpt0, cute.make_layout((1,)))[0] >= t2_s:
                    slp0 = atomicAdd(s_seg.iterator + cutlass.Int32(1), cutlass.Int32(1))
                    if slp0 < cutlass.Int32(capC):
                        smem_keys[cutlass.Int32(2 * segA) + slp0] = cutlass.Float32(ptb0)
                ptb0 = ptb0 + cutlass.Int32(num_threads)
            cute.arch.barrier()
            nlist0 = s_seg[1]
            if nlist0 <= cutlass.Int32(capC):
                # pass 2: 8 listed blocks per warp round
                nwp = cutlass.const_expr(self.num_warps)
                lb0 = warp_id * cutlass.Int32(8)
                frag_v = cute.make_rmem_tensor((8,), cutlass.Float32)
                while lb0 < nlist0:
                    # LOAD phase first: eight independent block loads in
                    # flight before any atomic (claims are memory-ordered
                    # and would serialize the blocks otherwise)
                    p2b0 = cutlass.Int32(0)
                    p2b1 = cutlass.Int32(0)
                    p2b2 = cutlass.Int32(0)
                    p2b3 = cutlass.Int32(0)
                    p2b4 = cutlass.Int32(0)
                    p2b5 = cutlass.Int32(0)
                    p2b6 = cutlass.Int32(0)
                    p2b7 = cutlass.Int32(0)
                    for _jb in cutlass.range_constexpr(8):
                        li0 = lb0 + cutlass.Int32(_jb)
                        bid0 = cutlass.Int32(-1)
                        vv0 = cutlass.Float32(self.NEG_FLT_MAX)
                        if li0 < nlist0:
                            bid0 = cutlass.Int32(smem_keys[cutlass.Int32(2 * segA) + li0])
                            pos0 = (bid0 << cutlass.Int32(5)) + lane
                            if pos0 < N:
                                vp0 = cute.make_ptr(
                                    cutlass.Float32,
                                    row_addr + cutlass.Int64(pos0) * cutlass.Int64(4),
                                    cute.AddressSpace.gmem,
                                    assumed_align=4,
                                )
                                vv0 = cute.make_tensor(vp0, cute.make_layout((1,)))[0]
                        frag_v[_jb] = vv0
                        if cutlass.const_expr(_jb == 0):
                            p2b0 = bid0
                        elif cutlass.const_expr(_jb == 1):
                            p2b1 = bid0
                        elif cutlass.const_expr(_jb == 2):
                            p2b2 = bid0
                        elif cutlass.const_expr(_jb == 3):
                            p2b3 = bid0
                        elif cutlass.const_expr(_jb == 4):
                            p2b4 = bid0
                        elif cutlass.const_expr(_jb == 5):
                            p2b5 = bid0
                        elif cutlass.const_expr(_jb == 6):
                            p2b6 = bid0
                        else:
                            p2b7 = bid0
                    # CLAIM phase
                    for _jb in cutlass.range_constexpr(8):
                        bidc = (
                            p2b0
                            if _jb == 0
                            else p2b1
                            if _jb == 1
                            else p2b2
                            if _jb == 2
                            else p2b3
                            if _jb == 3
                            else p2b4
                            if _jb == 4
                            else p2b5
                            if _jb == 5
                            else p2b6
                            if _jb == 6
                            else p2b7
                        )
                        vvc = cutlass.Float32(frag_v[_jb])
                        if bidc >= cutlass.Int32(0) and vvc >= t2_s:
                            posc = (bidc << cutlass.Int32(5)) + lane
                            if posc < N:
                                sl0 = atomicAdd(s_seg.iterator, cutlass.Int32(1))
                                if sl0 < cutlass.Int32(segA):
                                    smem_keys[sl0] = vvc
                                    cand_idx_row[sl0] = posc
                    lb0 = lb0 + cutlass.Int32(nwp * 8)
            if nlist0 > cutlass.Int32(capC):
                # list overflow (pass rate too high for skip): dense full
                # scan backup - nothing was read yet, plain re-run
                itd0 = cutlass.Int32(0)
                nfd0 = N >> cutlass.const_expr((num_threads * 4).bit_length() - 1)
                while itd0 < nfd0:
                    idd0 = (itd0 * cutlass.Int32(num_threads) + tidx) * cutlass.Int32(p1w)
                    pd0 = cute.make_ptr(
                        self.dtype,
                        row_addr + cutlass.Int64(idd0) * cutlass.Int64(4),
                        cute.AddressSpace.gmem,
                        assumed_align=vec_align,
                    )
                    cute.copy(
                        pass1_atom,
                        cute.make_tensor(pd0, cute.make_layout((p1w,))),
                        frag_p,
                    )
                    for _jd in cutlass.range_constexpr(p1w):
                        vd0 = cutlass.Float32(frag_p[_jd])
                        if vd0 >= t2_s:
                            sl0 = atomicAdd(s_seg.iterator, cutlass.Int32(1))
                            if sl0 < cutlass.Int32(segA):
                                smem_keys[sl0] = vd0
                                cand_idx_row[sl0] = idd0 + cutlass.Int32(_jd)
                    itd0 = itd0 + cutlass.Int32(1)
                ptd0 = nfd0 * cutlass.Int32(num_threads * 4) + tidx
                while ptd0 < N:
                    pe0 = cute.make_ptr(
                        cutlass.Float32,
                        row_addr + cutlass.Int64(ptd0) * cutlass.Int64(4),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    ve0 = cute.make_tensor(pe0, cute.make_layout((1,)))[0]
                    if ve0 >= t2_s:
                        sl0 = atomicAdd(s_seg.iterator, cutlass.Int32(1))
                        if sl0 < cutlass.Int32(segA):
                            smem_keys[sl0] = ve0
                            cand_idx_row[sl0] = ptd0
                    ptd0 = ptd0 + cutlass.Int32(num_threads)
        cpw = cutlass.const_expr(4)  # cp.async caps at 16B per copy
        step1 = cutlass.const_expr(num_threads * cpw)
        st2log = cutlass.const_expr((2 * step1).bit_length() - 1)
        # Pair-step cp.async pipeline: each step processes TWO 16B
        # vectors per thread (2 pairs x 32B across the 4 staging slots).
        # One commit group per pair; wait_group(1) pops the oldest pair.
        # The staging buffer aliases smem_vals (only written after phase
        # 0); every non-empty group is drained inside the loop, so
        # nothing is in flight once the alias is read. FULL-pair steps
        # only in the hot loop; the remainder takes the scalar tail below.
        nfull = N >> st2log
        if cutlass.const_expr(self.enable_block_skip and block_max_row is not None):
            nfull = cutlass.Int32(0)
        g2s_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        stage_addr = smem_stage.iterator.toint()
        for _p in cutlass.range_constexpr(2):
            if cutlass.Int32(_p) < nfull:
                for _v in cutlass.range_constexpr(2):
                    pr0 = cutlass.Int32(2 * _p + _v) * cutlass.Int32(num_threads) + tidx
                    pp0 = cutlass.Int32(2 * _p + _v) * cutlass.Int32(step1) + tidx * cutlass.Int32(
                        cpw
                    )
                    pq0 = cute.make_ptr(
                        self.dtype,
                        row_addr + cutlass.Int64(pp0) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    dq0 = cute.make_ptr(
                        cutlass.Float32,
                        stage_addr + cutlass.Int64(pr0 * cutlass.Int32(cpw)) * cutlass.Int64(4),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    cute.copy(
                        g2s_atom,
                        cute.make_tensor(pq0, cute.make_layout((cpw,))),
                        cute.make_tensor(dq0, cute.make_layout((cpw,))),
                    )
            cute.arch.cp_async_commit_group()
        it0 = cutlass.Int32(0)
        while it0 < nfull:
            cute.arch.cp_async_wait_group(1)
            sp0 = (it0 & cutlass.Int32(1)) * cutlass.Int32(2 * num_threads) + tidx
            ia0 = it0 * cutlass.Int32(2 * step1) + tidx * cutlass.Int32(cpw)
            # v6: per-element DIRECT atomic claims for passers — smem
            # atomics do NOT synchronize the warp and hide under the
            # async copy stream.
            for _jh in cutlass.range_constexpr(2):
                sq0 = cute.make_ptr(
                    cutlass.Float32,
                    stage_addr
                    + cutlass.Int64(
                        (sp0 + cutlass.Int32(_jh) * cutlass.Int32(num_threads)) * cutlass.Int32(cpw)
                    )
                    * cutlass.Int64(4),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                srow = cute.make_tensor(sq0, cute.make_layout((cpw,)))
                for _jv in cutlass.range_constexpr(cpw):
                    v0 = cutlass.Float32(srow[_jv])
                    if v0 >= t0_s:
                        pos0 = ia0 + cutlass.Int32(_jh) * cutlass.Int32(step1) + cutlass.Int32(_jv)
                        c0 = cutlass.Int32(2)
                        if v0 >= t1_s:
                            c0 = cutlass.Int32(1)
                        if v0 >= t2_s:
                            c0 = cutlass.Int32(0)
                        while c0 >= cutlass.Int32(0) and c0 <= cutlass.Int32(2):
                            cap0 = cutlass.Int32(segA)
                            if c0 == cutlass.Int32(2):
                                cap0 = cutlass.Int32(capC)
                            sl0 = atomicAdd(s_seg.iterator + c0, cutlass.Int32(1))
                            if sl0 < cap0:
                                cd0 = c0 * cutlass.Int32(segA) + sl0
                                smem_keys[cd0] = v0
                                cand_idx_row[cd0] = pos0
                                c0 = cutlass.Int32(-1)
                            else:
                                c0 = c0 + cutlass.Int32(1)
            # reissue the just-consumed slot pair for step it0 + 2 (the
            # thread's own prior reads are ordered before the async write
            # begins, so no fence is needed)
            kn0 = it0 + cutlass.Int32(2)
            if kn0 < nfull:
                for _jh in cutlass.range_constexpr(2):
                    jr0 = sp0 + cutlass.Int32(_jh) * cutlass.Int32(num_threads)
                    jp0 = (
                        kn0 * cutlass.Int32(2 * step1)
                        + cutlass.Int32(_jh) * cutlass.Int32(step1)
                        + tidx * cutlass.Int32(cpw)
                    )
                    pq0 = cute.make_ptr(
                        self.dtype,
                        row_addr + cutlass.Int64(jp0) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    dq0 = cute.make_ptr(
                        cutlass.Float32,
                        stage_addr + cutlass.Int64(jr0 * cutlass.Int32(cpw)) * cutlass.Int64(4),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    cute.copy(
                        g2s_atom,
                        cute.make_tensor(pq0, cute.make_layout((cpw,))),
                        cute.make_tensor(dq0, cute.make_layout((cpw,))),
                    )
            cute.arch.cp_async_commit_group()
            it0 = it0 + cutlass.Int32(1)
        # scalar tail (< step1 elements): per-element DIRECT atomic
        # claims — divergent-safe, no warp collectives
        pt0 = (N >> st2log) * cutlass.Int32(2 * step1) + tidx
        if cutlass.const_expr(self.enable_block_skip and block_max_row is not None):
            pt0 = N
        while pt0 < N:
            spt = cute.make_ptr(
                cutlass.Float32,
                row_addr + cutlass.Int64(pt0) * cutlass.Int64(4),
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            vt0 = cute.make_tensor(spt, cute.make_layout((1,)))[0]
            if vt0 >= t0_s:
                ct0 = cutlass.Int32(2)
                if vt0 >= t1_s:
                    ct0 = cutlass.Int32(1)
                if vt0 >= t2_s:
                    ct0 = cutlass.Int32(0)
                while ct0 >= cutlass.Int32(0) and ct0 <= cutlass.Int32(2):
                    capt = cutlass.Int32(segA)
                    if ct0 == cutlass.Int32(2):
                        capt = cutlass.Int32(capC)
                    slt = atomicAdd(s_seg.iterator + ct0, cutlass.Int32(1))
                    if slt < capt:
                        cdt = ct0 * cutlass.Int32(segA) + slt
                        smem_keys[cdt] = vt0
                        cand_idx_row[cdt] = pt0
                        ct0 = cutlass.Int32(-1)
                    else:
                        ct0 = ct0 + cutlass.Int32(1)
            pt0 = pt0 + cutlass.Int32(num_threads)
        cute.arch.barrier()
        if tidx == cutlass.Int32(0):
            if cutlass.const_expr(self.enable_block_skip):
                # single-band mode: every count is the t2 cursor; a cut
                # can only land on t2 (in band), the sample-hist (over)
                # or the fallback (under) - exactly the v5 state machine
                # fed with n0 == n1 == n2
                curT0 = s_seg[0]
                # claims past segA were dropped by the walk: honor the
                # "void==0 means nothing was dropped" contract
                s_seg[3] = cutlass.Int32(0)
                if curT0 > cutlass.Int32(segA):
                    s_seg[3] = cutlass.Int32(1)
                s_seg[4] = curT0
                s_seg[5] = curT0
                s_seg[6] = curT0
            if cutlass.const_expr(_P4_SUB_DBG):
                s_seg[7] = cutlass.Int32(cute.arch.clock64() - p0ck0)
            if cutlass.const_expr(not self.enable_block_skip):
                curA0 = s_seg[0]
                curB0 = s_seg[1]
                curC0 = s_seg[2]
                spA0 = curA0 - cutlass.Int32(segA)
                if spA0 < cutlass.Int32(0):
                    spA0 = cutlass.Int32(0)
                spB0 = curB0 - cutlass.Int32(segA)
                if spB0 < cutlass.Int32(0):
                    spB0 = cutlass.Int32(0)
                n1_0 = curA0 + curB0 - spA0
                n0_0 = n1_0 + curC0 - spB0
                s_seg[3] = cutlass.Int32(0)
                if curC0 > cutlass.Int32(capC):
                    s_seg[3] = cutlass.Int32(1)
                s_seg[4] = n0_0
                s_seg[5] = n1_0
                s_seg[6] = curA0
        cute.arch.barrier()

    @cute.jit
    def block_count_ge(
        self,
        input_row,  # cute.Tensor [N] fp32 (full row; this CTA only scans its slice)
        slice_start,  # int32: index in input_row where this CTA's slice starts
        slice_end,  # int32: index in input_row where this CTA's slice ends
        threshold,  # cutlass.Float32 scalar
        smem_ptcnt,  # cute.Tensor [BLOCK_SIZE] int32 (P3 cache)
        smem_wcnt,  # cute.Tensor [NUM_WARPS] int32 (block reduce scratch)
        s_iscalars,  # cute.Tensor [6] int32 (writes [0] = cand_count)
        s_cluster_partial,  # cute.Tensor [1] int32 (per-CTA partial scratch for DSMEM)
        tidx,
        warp_id,
        lane,
        do_cluster_sync,  # bool: False = skip DSMEM aggregation (cs=1 / short-row degrade)
        smem_input=None,  # optional SMEM-cached slice (smem_input[i] == input_row[slice_start+i])
        redundant=False,  # trace-time: every-warp reduce, return the total
        wcnt_off=None,  # int32 staging bank offset into smem_wcnt (parity)
    ):
        """Count input[i] >= threshold across this CTA's row slice, then
        DSMEM-aggregate across the cluster.

        ``redundant=True`` (p2_warp_redundant, cluster_size == 1 only):
        after the staging barrier EVERY warp reduces the warp counts
        lane-parallel and the block total RETURNS in a register —
        bit-identical across warps — instead of a leader writing
        s_iscalars[0] for a barrier-published broadcast. ``wcnt_off``
        parity-banks the smem_wcnt staging so a warp that has moved on
        to the next Phase-2 round cannot clobber a slot a slower warp is
        still reading (the per-round staging barrier bounds the drift to
        one round).

        Vectorized scan: each thread loads vec_w elements per iter (128 or
        256 bits) over ``input_row[slice_start : slice_end)``; scalar tail
        handles the remainder.

        Cluster aggregation (cluster_size > 1): every CTA stages its
        slice-local count into ``s_cluster_partial[call & 1]`` (parity
        double-buffer; slot 2 is the tid0-private call counter), syncs the
        cluster, then DSMEM-reads every peer's slot and sums into
        ``s_iscalars[0]``.
        After this every CTA's ``s_iscalars[0]`` holds the same
        cluster-wide cand_count, so Phase 2's secant update stays a
        leader-only scalar op on a value all CTAs agree on.
        """
        num_threads = cutlass.const_expr(self.num_threads)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        cluster_size = cutlass.const_expr(self.cluster_size)
        c = cutlass.Int32(0)
        copy_atom = self._make_load_copy_atom()

        step_elem = cutlass.const_expr(num_threads * vec_w)

        row_addr = input_row.iterator.toint()
        slice_len = slice_end - slice_start
        # smem-cache path uses slice-LOCAL indices (smem_input[0] ==
        # input_row[slice_start]); GMEM path uses global indices. Set up
        # both upfront so the const_expr branches below stay flat.
        if cutlass.const_expr(smem_input is not None):
            smem_addr = smem_input.iterator.toint()
            n_aligned = (slice_len // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
            N = slice_len  # upper bound is slice-local
            i = tidx * cutlass.Int32(vec_w)
        else:
            n_aligned = slice_start + (slice_len // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
            N = slice_end  # global upper bound
            i = slice_start + tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        # Fast path: 4-way unroll for LSU-pipelining ILP.
        # Each iter loads 1 vec_w chunk; LLVM unrolls 4 iters at IR level
        # so 4 LDG.E.* stay in flight.
        if self.enable_unroll_4:
            rng_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
            # Number of complete vec_w-aligned loads this thread can do:
            #   need: i + k*step_elem + (vec_w - 1) < N
            #   max k: floor((N - i - vec_w) / step_elem)
            #   N_iters = max_k + 1
            big_iters = cutlass.Int32(0)
            if N > i + cutlass.Int32(vec_w - 1):
                big_iters = (N - i - cutlass.Int32(vec_w)) // cutlass.Int32(
                    step_elem
                ) + cutlass.Int32(1)

            for k in cutlass.range(big_iters, unroll=4):
                i_local = i + k * cutlass.Int32(step_elem)
                if cutlass.const_expr(smem_input is not None):
                    src_ptr_k = cute.make_ptr(
                        self.dtype,
                        smem_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.smem,
                        assumed_align=vec_align,
                    )
                else:
                    src_ptr_k = cute.make_ptr(
                        self.dtype,
                        row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem,
                        assumed_align=vec_align,
                    )
                src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                cute.copy(copy_atom, src_k, rng_frag)
                for j in cutlass.range_constexpr(vec_w):
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        vj = rng_frag[j]
                    else:
                        vj = cutlass.Float32(rng_frag[j])
                    if vj >= threshold:
                        c = c + cutlass.Int32(1)
            # Advance i past all consumed vec_w-aligned positions so the
            # medium/tail loops below correctly skip (they check i + ... < N).
            i = i + big_iters * cutlass.Int32(step_elem)

        # Tail vec loop: 1-way, handles remainder < 2*step (= remaining 1
        # full vec_w-stride or less). i is always vec_w-aligned here (it
        # advanced by multiples of step_elem = num_threads*vec_w), so the
        # same vec_align bytes hold.
        tail_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
        while i + cutlass.Int32(vec_w - 1) < N:
            if cutlass.const_expr(smem_input is not None):
                src_ptr = cute.make_ptr(
                    self.dtype,
                    smem_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.smem,
                    assumed_align=vec_align,
                )
            else:
                src_ptr = cute.make_ptr(
                    self.dtype,
                    row_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.gmem,
                    assumed_align=vec_align,
                )
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                if vj >= threshold:
                    c = c + cutlass.Int32(1)
            i = i + step

        # Tail scalar loop. SMEM path uses slice-local indexing
        # (smem_input[it]); GMEM path uses global indices (input_row[it]).
        it = n_aligned + tidx
        while it < N:
            if cutlass.const_expr(smem_input is not None):
                v = smem_input[it]
                if cutlass.const_expr(self.dtype != cutlass.Float32):
                    v = cutlass.Float32(v)
            else:
                v = self._load_fp32(input_row, it)
            if v >= threshold:
                c = c + cutlass.Int32(1)
            it = it + cutlass.Int32(num_threads)

        # Cache per-thread count for P3 retry-shrink reuse.
        smem_ptcnt[tidx] = c

        # Warp reduce + lane-0 write
        wc = self.warp_reduce_sum_i32(c)
        stage_base = cutlass.Int32(0)
        if cutlass.const_expr(wcnt_off is not None):
            stage_base = wcnt_off
        if lane == 0:
            smem_wcnt[stage_base + warp_id] = wc
        cute.arch.barrier()

        if cutlass.const_expr(redundant):
            # Every warp reduces the staged counts itself; no leader, no
            # publish barrier, no s_iscalars[0] round-trip.
            v_r = cutlass.Int32(0)
            if lane < cutlass.Int32(self.num_warps):
                v_r = smem_wcnt[stage_base + lane]
            total_r = self.warp_reduce_sum_i32(v_r)
            return total_r

        # Block aggregate (sum reduce over num_warps slots). No trailing
        # barrier: caller is expected to insert its own __syncthreads after
        # its post-processing of cand_count.
        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            # NEW: warp-parallel sum reduce in warp 0.
            if warp_id == cutlass.Int32(0):
                v = cutlass.Int32(0)
                if lane < cutlass.Int32(self.num_warps):
                    v = smem_wcnt[lane]
                total = self.warp_reduce_sum_i32(v)
                if lane == cutlass.Int32(0):
                    s_iscalars[0] = total
        else:
            # tid==0 serial sum.
            if tidx == 0:
                total = cutlass.Int32(0)
                for w in cutlass.range_constexpr(self.num_warps):
                    total = total + smem_wcnt[w]
                s_iscalars[0] = total

        # Snapshot local cand_count into s_iscalars[5] before the cluster
        # all-reduce overwrites s_iscalars[0]. Only needed when
        # do_cluster_sync=True: the DSMEM gather in Phase 4 reads peer
        # s_iscalars[5] values; skipped in short-row degrade (do_cluster_sync=False)
        # where s_iscalars[0] is never overwritten and the gather never fires.
        if cutlass.const_expr(cluster_size > 1):
            if do_cluster_sync:
                if tidx == cutlass.Int32(0):
                    s_iscalars[5] = s_iscalars[0]
                cute.arch.barrier()

        # Cluster all-reduce of cand_count. Skipped at cluster_size==1.
        # Also skipped at runtime when do_cluster_sync=False (short-row
        # degrade): CTA 0 is the only live CTA in the cluster and its
        # local count IS the total, so s_iscalars[0] already holds the
        # correct value with no DSMEM read needed.
        if cutlass.const_expr(cluster_size > 1):
            if do_cluster_sync:
                cute.arch.barrier()  # publish s_iscalars[0] to all threads of this CTA
                # Parity double-buffer: with a single slot, a straggler's
                # post-wait DSMEM read races the peer's next-call overwrite
                # (PTX-model data race). Writing call k into slot k&1 orders
                # the call-(k+2) overwrite after my call-k reads via the
                # call-(k+1) rendezvous. Slot 2 = tid0-private call counter
                # (zeroed per row); do_cluster_sync is row-uniform, so CTAs
                # step the counter in lockstep and parity stays aligned.
                par = cutlass.Int32(0)
                if tidx == cutlass.Int32(0):
                    par = s_cluster_partial[2]
                    s_cluster_partial[par & cutlass.Int32(1)] = s_iscalars[0]
                    s_cluster_partial[2] = par + cutlass.Int32(1)
                # Non-relaxed arrive: pairs with the peer cluster_wait acquire
                # to release s_cluster_partial writes so the DSMEM ld below
                # observes them. cluster_arrive_relaxed would skip the release
                # fence and risk stale peer reads on hardware that doesn't
                # eagerly publish shared writes.
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                if tidx == cutlass.Int32(0):
                    total = cutlass.Int32(0)
                    local_ptr = s_cluster_partial.iterator + (par & cutlass.Int32(1))
                    for peer in cutlass.range_constexpr(cluster_size):
                        peer_addr = mapa_shared_cluster(local_ptr, cutlass.Int32(peer))
                        total = total + ld_shared_cluster_i32(peer_addr)
                    s_iscalars[0] = total
                cute.arch.barrier()  # broadcast cluster total within this CTA

        return cutlass.Int32(0)

    # ------------------------------------------------------------------
    # block_count_ge_multi<M> — GE-count of the input row against M
    # thresholds in ONE vectorized scan, reusing block_count_ge's memory
    # path (same vec_w / 4-way-unroll / tail loops) with M static register
    # counters. Caches all M per-thread count columns in smem_ptcnt_multi so
    # the accepted rung's column seeds Phase 3 with zero rescan. This is the
    # R0 admission primitive (op#18 multithresh lineage); it is only invoked
    # from the enable_r0 path added in a later commit, so the base kernel is
    # unaffected. Slice + cluster form: each CTA scans [slice_start,
    # slice_end) and the M per-CTA totals are DSMEM all-reduced across the
    # cluster (cluster_size>1, do_cluster_sync) with a release cluster_arrive
    # mirroring block_count_ge; at cs==1 (or short-row degrade) the local
    # totals are the answer. smem_ptcnt_multi holds slice-local per-thread
    # columns (the accepted rung's column seeds Phase 3 per CTA).
    # ------------------------------------------------------------------

    # ---- block-skip machinery (enable_block_skip): grain 32, int16
    # list, grouped strided build, UN=2 scan ----

    @cute.jit
    def _list_ld(self, smem_active, idx):
        return cutlass.Int32(smem_active[idx])

    @cute.jit
    def _list_st(self, smem_active, idx, val):
        smem_active[idx] = cutlass.Int16(val)

    @cute.jit
    def _block_bound(self, bm_addr, blk_id):
        # grain 32: record blk_id IS the exact positional bound of
        # [blk_id*32, blk_id*32+32) (indexer TMEM partition contract) —
        # one 4B scalar load, no fold.
        bm_ptr = cute.make_ptr(
            cutlass.Float32,
            bm_addr + cutlass.Int64(blk_id) * cutlass.Int64(4),
            cute.AddressSpace.gmem,
            assumed_align=4,
        )
        return cute.make_tensor(bm_ptr, cute.make_layout((1,)))[0]

    @cute.jit
    def _full_build_active(
        self,
        block_max_row,
        slice_start,
        slice_end,
        threshold,
        smem_wcnt,
        smem_active,
        s_active_cnt,
        tidx,
        warp_id,
        lane,
    ):
        # Two-phase register-bitmask build. Each thread owns ids_per_thread
        # block ids STRIDED (t, t+T, t+2T, ...) so each pass's bound loads
        # coalesce warp-wide. Phase A flags+counts with no sync; phase B is
        # ONE block-wide exclusive scan over per-thread counts; phase C
        # scatters from the bitmask — 3 barriers TOTAL, independent of
        # nb_slice. List order is thread-grouped, not ascending: fine, since
        # the count scan and the Phase-3 stream-write walk the SAME list by
        # position (determinism contract).
        ids_per_thread = cutlass.const_expr(self.SKIP_MAX_BLOCKS // self.num_threads)
        num_threads = cutlass.const_expr(self.num_threads)
        # first FULL block: a block straddling slice_start would map list
        # positions outside this CTA's slice; the sub-block head region
        # [slice_start, blk_lo*32) is scanned separately by the callers.
        blk_lo = (slice_start + cutlass.Int32(self.SKIP_BLOCK - 1)) >> cutlass.Int32(
            self.SKIP_BLOCK_LOG2
        )
        blk_hi = (slice_end + cutlass.Int32(self.SKIP_BLOCK - 1)) >> cutlass.Int32(
            self.SKIP_BLOCK_LOG2
        )
        nb_slice = blk_hi - blk_lo
        bm_addr = block_max_row.iterator.toint()

        # Phase A (no sync): flag my ids into a register bitmask.
        bmask = cutlass.Int32(0)
        cnt = cutlass.Int32(0)
        for m in cutlass.range_constexpr(ids_per_thread):
            ib = tidx + cutlass.Int32(m * num_threads)
            if ib < nb_slice:
                bound = self._block_bound(bm_addr, blk_lo + ib)
                if bound >= threshold:
                    bmask = bmask | (cutlass.Int32(1) << cutlass.Int32(m))
                    cnt = cnt + cutlass.Int32(1)

        # Phase B: one block-wide exclusive scan over per-thread counts.
        tp = cnt
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            other = cute.arch.shuffle_sync_up(tp, off_v, mask_and_clamp=0)
            if lane >= cutlass.Int32(off_v):
                tp = tp + other
        excl = tp - cnt
        warp_total = cute.arch.shuffle_sync(tp, cutlass.Int32(self.WARP_SIZE - 1))
        if lane == 0:
            smem_wcnt[warp_id] = warp_total
        cute.arch.barrier()
        if tidx == 0:
            tot = cutlass.Int32(0)
            for w in cutlass.range_constexpr(self.num_warps):
                cw = smem_wcnt[w]
                smem_wcnt[w] = tot
                tot = tot + cw
            s_active_cnt[0] = tot
        cute.arch.barrier()

        # Phase C: scatter from the bitmask at deterministic offsets.
        pos_out = smem_wcnt[warp_id] + excl
        if bmask != cutlass.Int32(0):
            for m in cutlass.range_constexpr(ids_per_thread):
                if (bmask & (cutlass.Int32(1) << cutlass.Int32(m))) != cutlass.Int32(0):
                    ib_c = tidx + cutlass.Int32(m * num_threads)
                    self._list_st(smem_active, pos_out, blk_lo + ib_c)
                    pos_out = pos_out + cutlass.Int32(1)
        cute.arch.barrier()

    @cute.jit
    def block_count_ge_multi(
        self,
        input_row,
        slice_start,
        slice_end,
        s_mt_thr,
        smem_ptcnt_multi,
        smem_wcnt_multi,
        s_mt_cnt,
        s_cluster_partial_m,
        do_cluster_sync,
        tidx,
        warp_id,
        lane,
        smem_ptcnt=None,  # vseed: last column's per-thread counts land here
        block_max_row=None,  # block-skip: per-32-position upper bounds
        smem_active=None,  # block-skip: int16 active list
        s_active_cnt=None,  # block-skip: [0]=list length, [1]=list-current flag
    ):
        M = cutlass.const_expr(self.M_thr)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        cluster_size = cutlass.const_expr(self.cluster_size)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        copy_atom = self._make_load_copy_atom()
        step_elem = cutlass.const_expr(num_threads * vec_w)

        thr_frag = cute.make_rmem_tensor((M,), cutlass.Float32)
        cnt_frag = cute.make_rmem_tensor((M,), cutlass.Int32)
        for m in cutlass.range_constexpr(M):
            thr_frag[m] = s_mt_thr[m]
            cnt_frag[m] = cutlass.Int32(0)

        row_addr = input_row.iterator.toint()
        slice_len = slice_end - slice_start
        n_aligned = slice_start + (slice_len // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        i = slice_start + tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        # ---- block-skip compact iteration (lossless vs the dense path) ----
        # Build the active list at the LOOSEST threshold over all M columns:
        # a skipped block bounds every element below min(t_m), so all M
        # counts equal their dense values. The list covers FULL blocks only;
        # the sub-block head region of an unaligned slice start is counted
        # here separately (per-thread order contract: head elements FIRST,
        # then list entries — Phase 3's compact write replays the same).
        skip_ok = cutlass.Int32(0)
        dense_ok = True  # Python bool: no scf.if when block skip is off
        if cutlass.const_expr(self.enable_block_skip and block_max_row is not None):
            skip_ok = cutlass.Int32(1)
            # Capacity/id-width guard: the active list holds at most
            # SKIP_MAX_BLOCKS local ids and _list_st stores ABSOLUTE block
            # ids as int16. A slice over 8192 full blocks (N_local >
            # 262144) or reaching absolute id >= 32768 falls back to the
            # dense walk (lossless; the list-current flag is never set, so
            # phase3 stays dense too).
            blk_lo_g = (slice_start + cutlass.Int32(self.SKIP_BLOCK - 1)) >> cutlass.Int32(
                self.SKIP_BLOCK_LOG2
            )
            blk_hi_g = (slice_end + cutlass.Int32(self.SKIP_BLOCK - 1)) >> cutlass.Int32(
                self.SKIP_BLOCK_LOG2
            )
            if blk_hi_g - blk_lo_g > cutlass.Int32(self.SKIP_MAX_BLOCKS):
                skip_ok = cutlass.Int32(0)
            if blk_hi_g > cutlass.Int32(32767):
                skip_ok = cutlass.Int32(0)
            dense_ok = skip_ok == cutlass.Int32(0)
        if cutlass.const_expr(self.enable_block_skip and block_max_row is not None):
            if skip_ok == cutlass.Int32(1):
                head_end = (
                    (slice_start + cutlass.Int32(self.SKIP_BLOCK - 1))
                    >> cutlass.Int32(self.SKIP_BLOCK_LOG2)
                ) << cutlass.Int32(self.SKIP_BLOCK_LOG2)
                if head_end > slice_end:
                    head_end = slice_end
                hh = slice_start + tidx
                while hh < head_end:
                    vh = self._load_fp32(input_row, hh)
                    for m in cutlass.range_constexpr(M):
                        cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vh >= thr_frag[m])
                    hh = hh + cutlass.Int32(num_threads)
                # Rung-tightening build (cs==1): a rung whose active list
                # exceeds CAP blocks cannot be accepted (count >= list
                # length), so drop it and rebuild at the next tighter
                # threshold. A dropped rung is only an unmeasured probe,
                # recorded in the mask at s_active_cnt[2] and skipped by
                # classify and by the fallback seeding. Bounded by M-1
                # extra builds. At cs>1 the per-CTA list lengths differ
                # and the drop decision would diverge across the cluster,
                # so keep the plain loosest-rung build there.
                CAP_BLOCKS = cutlass.const_expr(3 * self.kC // 4)
                if cutlass.const_expr(cluster_size == 1):
                    build_done = cutlass.Int32(0)
                    for _attempt in cutlass.range_constexpr(M):
                        if build_done == cutlass.Int32(0):
                            dmask = s_active_cnt[2]
                            tcur = cutlass.Float32(self.FLT_MAX)
                            mcur = cutlass.Int32(-1)
                            kept = cutlass.Int32(0)
                            for m in cutlass.range_constexpr(M):
                                if (
                                    dmask & (cutlass.Int32(1) << cutlass.Int32(m))
                                ) == cutlass.Int32(0):
                                    kept = kept + cutlass.Int32(1)
                                    if thr_frag[m] < tcur:
                                        tcur = thr_frag[m]
                                        mcur = cutlass.Int32(m)
                            self._full_build_active(
                                block_max_row,
                                slice_start,
                                slice_end,
                                tcur,
                                smem_wcnt_multi,
                                smem_active,
                                s_active_cnt,
                                tidx,
                                warp_id,
                                lane,
                            )
                            if s_active_cnt[0] <= cutlass.Int32(
                                CAP_BLOCKS
                            ) or kept <= cutlass.Int32(1):
                                build_done = cutlass.Int32(1)
                            else:
                                if tidx == 0:
                                    s_active_cnt[2] = dmask | (cutlass.Int32(1) << mcur)
                                cute.arch.barrier()
                else:
                    tmin = thr_frag[0]
                    for m in cutlass.range_constexpr(M):
                        tmin = _fmin_f32_inline(tmin, thr_frag[m])
                    self._full_build_active(
                        block_max_row,
                        slice_start,
                        slice_end,
                        tmin,
                        smem_wcnt_multi,
                        smem_active,
                        s_active_cnt,
                        tidx,
                        warp_id,
                        lane,
                    )
                if tidx == 0:
                    s_active_cnt[1] = cutlass.Int32(1)  # list-current flag
                chunks_per_block = cutlass.const_expr(
                    self.SKIP_BLOCK // (self.vec_bits // self.dtype.width)
                )
                tpb = cutlass.const_expr(chunks_per_block)
                blocks_per_iter = cutlass.const_expr(self.num_threads // chunks_per_block)
                UN = cutlass.const_expr(self.SKIP_UNROLL)
                stride_un = cutlass.const_expr(blocks_per_iter * UN)
                my_blk_slot = tidx // cutlass.Int32(tpb)
                my_chunk0 = tidx % cutlass.Int32(tpb)
                cnt_active = s_active_cnt[0]
                frags = [cute.make_rmem_tensor((vec_w,), self.dtype) for _ in range(UN)]
                li = my_blk_slot
                while li < cnt_active:
                    poss = []
                    valids = []
                    for u in cutlass.range_constexpr(UN):
                        lu = li + cutlass.Int32(u * blocks_per_iter)
                        valid = lu < cnt_active
                        pos0 = cutlass.Int32(0)
                        if valid:
                            blk = self._list_ld(smem_active, lu)
                            pos0 = blk * cutlass.Int32(self.SKIP_BLOCK) + my_chunk0 * cutlass.Int32(
                                vec_w
                            )
                            # Vector-load only fully in-bounds chunks; the
                            # slice-end straddle re-reads scalars below (a
                            # tail block's chunks would otherwise read past
                            # the row/allocation when N % 32 != 0).
                            if pos0 + cutlass.Int32(vec_w) <= slice_end:
                                src_ptr_u = cute.make_ptr(
                                    self.dtype,
                                    row_addr + cutlass.Int64(pos0) * cutlass.Int64(elem_bytes),
                                    cute.AddressSpace.gmem,
                                    assumed_align=vec_align,
                                )
                                cute.copy(
                                    copy_atom,
                                    cute.make_tensor(src_ptr_u, cute.make_layout((vec_w,))),
                                    frags[u],
                                )
                        poss.append(pos0)
                        valids.append(valid)
                    for u in cutlass.range_constexpr(UN):
                        if valids[u]:
                            pos = poss[u]
                            if pos + cutlass.Int32(vec_w) <= slice_end:
                                for j in cutlass.range_constexpr(vec_w):
                                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                                        vj = frags[u][j]
                                    else:
                                        vj = cutlass.Float32(frags[u][j])
                                    for m in cutlass.range_constexpr(M):
                                        cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
                            else:
                                jj = pos
                                while jj < slice_end:
                                    vs = self._load_fp32(input_row, jj)
                                    for m in cutlass.range_constexpr(M):
                                        cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vs >= thr_frag[m])
                                    jj = jj + cutlass.Int32(1)
                    li = li + cutlass.Int32(stride_un)

        if self.enable_unroll_4 and dense_ok:
            rng_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
            big_iters = cutlass.Int32(0)
            if slice_end > i + cutlass.Int32(vec_w - 1):
                big_iters = (slice_end - i - cutlass.Int32(vec_w)) // cutlass.Int32(
                    step_elem
                ) + cutlass.Int32(1)
            for k in cutlass.range(big_iters, unroll=self.mt_unroll):
                i_local = i + k * cutlass.Int32(step_elem)
                src_ptr_k = cute.make_ptr(
                    self.dtype,
                    row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.gmem,
                    assumed_align=vec_align,
                )
                src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                cute.copy(copy_atom, src_k, rng_frag)
                for j in cutlass.range_constexpr(vec_w):
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        vj = rng_frag[j]
                    else:
                        vj = cutlass.Float32(rng_frag[j])
                    for m in cutlass.range_constexpr(M):
                        cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + big_iters * cutlass.Int32(step_elem)

        tail_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
        if dense_ok:
            while i + cutlass.Int32(vec_w - 1) < slice_end:
                src_ptr = cute.make_ptr(
                    self.dtype,
                    row_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.gmem,
                    assumed_align=vec_align,
                )
                src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
                cute.copy(copy_atom, src, tail_frag)
                for j in cutlass.range_constexpr(vec_w):
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        vj = tail_frag[j]
                    else:
                        vj = cutlass.Float32(tail_frag[j])
                    for m in cutlass.range_constexpr(M):
                        cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
                i = i + step

            it = n_aligned + tidx
            while it < slice_end:
                v = self._load_fp32(input_row, it)
                for m in cutlass.range_constexpr(M):
                    cnt_frag[m] = cnt_frag[m] + cutlass.Int32(v >= thr_frag[m])
                it = it + cutlass.Int32(num_threads)

        for m in cutlass.range_constexpr(M):
            if cutlass.const_expr(self.r0_vseed and m == self.M_qf):
                smem_ptcnt[tidx] = cnt_frag[m]
            else:
                smem_ptcnt_multi[m * num_threads + tidx] = cnt_frag[m]

        for m in cutlass.range_constexpr(M):
            wc = self.warp_reduce_sum_i32(cnt_frag[m])
            if lane == 0:
                smem_wcnt_multi[m * num_warps + warp_id] = wc
        cute.arch.barrier()
        # Block-reduce the M warp counts to this CTA's slice totals. Stage
        # into DSMEM scratch at cs>1 (for the cluster merge below), else
        # write straight to s_mt_cnt.
        if warp_id == cutlass.Int32(0):
            for m in cutlass.range_constexpr(M):
                v = cutlass.Int32(0)
                if lane < cutlass.Int32(num_warps):
                    v = smem_wcnt_multi[m * num_warps + lane]
                total = self.warp_reduce_sum_i32(v)
                if lane == cutlass.Int32(0):
                    if cutlass.const_expr(cluster_size > 1):
                        s_cluster_partial_m[m] = total
                    else:
                        s_mt_cnt[m] = total
        cute.arch.barrier()
        if cutlass.const_expr(cluster_size > 1):
            if do_cluster_sync:
                # Release arrive (NOT relaxed): pairs with the peer
                # cluster_wait acquire so the staged M totals are visible
                # before any CTA reads them over DSMEM.
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                if tidx == cutlass.Int32(0):
                    local_ptr = s_cluster_partial_m.iterator
                    for m in cutlass.range_constexpr(M):
                        total = cutlass.Int32(0)
                        for peer in cutlass.range_constexpr(cluster_size):
                            peer_addr = mapa_shared_cluster(
                                local_ptr + cutlass.Int32(m), cutlass.Int32(peer)
                            )
                            total = total + ld_shared_cluster_i32(peer_addr)
                        s_mt_cnt[m] = total
                cute.arch.barrier()
            else:
                # short-row degrade: this CTA's local totals are the answer.
                if tidx == cutlass.Int32(0):
                    for m in cutlass.range_constexpr(M):
                        s_mt_cnt[m] = s_cluster_partial_m[m]
                cute.arch.barrier()

    # ------------------------------------------------------------------
    # Phase 2: Secant-interpolation threshold search
    # Refines threshold to bring cand_count into [kK, kCC] using secant
    # interpolation on (val_lo, cnt_lo) / (val_hi, cnt_hi). At most
    # self.MAX_REFINE_ITERS iterations.
    # ------------------------------------------------------------------
    @cute.jit
    def phase2_secant_search(
        self,
        input_row,
        N,
        slice_start,
        slice_end,
        smem_ptcnt,
        smem_wcnt,
        s_thr,  # [threshold, val_lo, val_hi]
        s_iscalars,  # [cand_count, done, cnt_lo, cnt_hi, out_count]
        s_cluster_partial,  # [3] int32 cluster scratch (parity slots + counter)
        tidx,
        warp_id,
        lane,
        do_cluster_sync,  # bool: False = cs=1 / short-row degrade (skip cluster sync)
        smem_input=None,  # optional SMEM-cached slice
    ):
        """Refine s_thr[0] until cand_count lands in [kK, kCC].

        Each iter calls block_count_ge at the candidate threshold and
        updates the bracket (val_lo, val_hi, cnt_lo, cnt_hi). Sets
        s_iscalars[1] (done) = 1 on convergence, 2 on bracket exhaustion.
        """
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        kFTarget = cutlass.const_expr(self.kFTarget)

        if cutlass.const_expr(self.p2_warp_redundant and self.cluster_size == 1):
            # ---- Redundant-warp cadence: ONE barrier per round ----
            # The whole secant state (threshold, bracket, counts, done)
            # lives in registers; every warp reduces the staged warp
            # counts itself (block_count_ge redundant mode) and replays
            # the identical classify + secant update, so the per-round
            # publish barriers and every s_thr/s_iscalars SMEM round-trip
            # (with its per-access cluster-window S2R recompute)
            # disappear. Canonical exit state is written once for P3.
            nwp2 = cutlass.const_expr(self.num_warps)
            thr_r = s_thr[0]
            vlo_r = s_thr[1]
            vhi_r = s_thr[2]
            clo_r = s_iscalars[2]
            chi_r = s_iscalars[3]
            done_r = cutlass.Int32(0)
            par_r = cutlass.Int32(0)
            cnt_r = self.block_count_ge(
                input_row,
                slice_start,
                slice_end,
                thr_r,
                smem_ptcnt,
                smem_wcnt,
                s_iscalars,
                s_cluster_partial,
                tidx,
                warp_id,
                lane,
                cutlass.Boolean(False),  # do_cluster_sync (cs==1 gate)
                smem_input=smem_input,
                redundant=True,
                wcnt_off=par_r * cutlass.Int32(nwp2),
            )
            if cnt_r >= cutlass.Int32(kK) and cnt_r <= cutlass.Int32(kCC):
                done_r = cutlass.Int32(1)
            elif cnt_r > cutlass.Int32(kCC):
                vlo_r = thr_r
                clo_r = cnt_r
            else:
                vhi_r = thr_r
                chi_r = cnt_r
            it = cutlass.Int32(0)
            while it < cutlass.Int32(self.MAX_REFINE_ITERS) and done_r == cutlass.Int32(0):
                rng = vhi_r - vlo_r
                nv = cutlass.Float32(0.0)
                if clo_r > chi_r and rng > cutlass.Float32(1e-10):
                    f = cutlass.Float32(clo_r - cutlass.Int32(kFTarget)) / cutlass.Float32(
                        clo_r - chi_r
                    )
                    f = cute.arch.fmax(cutlass.Float32(0.05), f)
                    f = _fmin_f32_inline(f, cutlass.Float32(0.95))
                    if it == cutlass.Int32(0):
                        f = _fmin_f32_inline(f, cutlass.Float32(0.5))
                    nv = vlo_r + rng * f
                else:
                    nv = (vlo_r + vhi_r) * cutlass.Float32(0.5)
                if nv <= vlo_r:
                    nv = vlo_r + rng * cutlass.Float32(0.05)
                if nv >= vhi_r:
                    nv = vhi_r - rng * cutlass.Float32(0.05)
                if nv == vlo_r or nv == vhi_r:
                    nv = (vlo_r + vhi_r) * cutlass.Float32(0.5)
                    if nv == vlo_r or nv == vhi_r:
                        # ADJACENT-FLOAT bracket, same terminal as the
                        # leader path: a low side over the candidate
                        # buffer plus a high side under K means the
                        # boundary sits inside a bitwise-equal plateau
                        # wider than kC. Keep the sure-winner threshold
                        # and let Phase 4's plateau fill finish the row.
                        if clo_r > cutlass.Int32(kCC) and chi_r < cutlass.Int32(kK):
                            thr_r = vhi_r
                            done_r = cutlass.Int32(3)
                        else:
                            thr_r = vlo_r
                            done_r = cutlass.Int32(2)
                if done_r == cutlass.Int32(0):
                    thr_r = nv
                    par_r = par_r ^ cutlass.Int32(1)
                    cnt_r = self.block_count_ge(
                        input_row,
                        slice_start,
                        slice_end,
                        thr_r,
                        smem_ptcnt,
                        smem_wcnt,
                        s_iscalars,
                        s_cluster_partial,
                        tidx,
                        warp_id,
                        lane,
                        cutlass.Boolean(False),  # do_cluster_sync (cs==1 gate)
                        smem_input=smem_input,
                        redundant=True,
                        wcnt_off=par_r * cutlass.Int32(nwp2),
                    )
                    if cnt_r >= cutlass.Int32(kK) and cnt_r <= cutlass.Int32(kCC):
                        done_r = cutlass.Int32(1)
                    elif cnt_r > cutlass.Int32(kCC):
                        vlo_r = thr_r
                        clo_r = cnt_r
                    else:
                        vhi_r = thr_r
                        chi_r = cnt_r
                it = it + cutlass.Int32(1)
            # ---- Budget-exhausted plateau collapse (mirrors the leader
            # path): the refine budget can run out while the bracket is
            # still wide because a tie plateau wider than kC admits no
            # threshold. On exactly that signature, bisect to adjacent
            # floats so the plateau terminal is exact. Every thread
            # replays this from identical registers, so the branch stays
            # warp-uniform and block_count_ge keeps its barrier cadence.
            if (
                done_r == cutlass.Int32(0)
                and clo_r > cutlass.Int32(kCC)
                and chi_r >= cutlass.Int32(0)
                and chi_r < cutlass.Int32(kK)
            ):
                itc = cutlass.Int32(0)
                while itc < cutlass.Int32(64) and done_r == cutlass.Int32(0):
                    mid_c = (vlo_r + vhi_r) * cutlass.Float32(0.5)
                    if mid_c == vlo_r or mid_c == vhi_r:
                        thr_r = vhi_r
                        done_r = cutlass.Int32(3)
                    else:
                        thr_r = mid_c
                        par_r = par_r ^ cutlass.Int32(1)
                        cnt_r = self.block_count_ge(
                            input_row,
                            slice_start,
                            slice_end,
                            thr_r,
                            smem_ptcnt,
                            smem_wcnt,
                            s_iscalars,
                            s_cluster_partial,
                            tidx,
                            warp_id,
                            lane,
                            cutlass.Boolean(False),  # do_cluster_sync (cs==1)
                            smem_input=smem_input,
                            redundant=True,
                            wcnt_off=par_r * cutlass.Int32(nwp2),
                        )
                        if cnt_r >= cutlass.Int32(kK) and cnt_r <= cutlass.Int32(kCC):
                            done_r = cutlass.Int32(1)
                        elif cnt_r > cutlass.Int32(kCC):
                            vlo_r = thr_r
                            clo_r = cnt_r
                        else:
                            vhi_r = thr_r
                            chi_r = cnt_r
                    itc = itc + cutlass.Int32(1)
                if done_r == cutlass.Int32(3):
                    # recount at the terminal threshold so Phase 3 sees
                    # per-thread counts for the sure-winner set.
                    par_r = par_r ^ cutlass.Int32(1)
                    cnt_r = self.block_count_ge(
                        input_row,
                        slice_start,
                        slice_end,
                        thr_r,
                        smem_ptcnt,
                        smem_wcnt,
                        s_iscalars,
                        s_cluster_partial,
                        tidx,
                        warp_id,
                        lane,
                        cutlass.Boolean(False),  # do_cluster_sync (cs==1)
                        smem_input=smem_input,
                        redundant=True,
                        wcnt_off=par_r * cutlass.Int32(nwp2),
                    )
            if done_r == cutlass.Int32(0):
                if clo_r <= cutlass.Int32(kCC * 2):
                    thr_r = vlo_r
                else:
                    thr_r = vhi_r
                done_r = cutlass.Int32(2)
            # Canonical exit state for Phase 3/4 (byte-compatible with the
            # leader path), published once.
            if tidx == 0:
                s_thr[0] = thr_r
                s_thr[1] = vlo_r
                s_thr[2] = vhi_r
                s_iscalars[0] = cnt_r
                s_iscalars[1] = done_r
                s_iscalars[2] = clo_r
                s_iscalars[3] = chi_r
            cute.arch.barrier()
            return

        # ---- Initial count with the Phase-1 mean as threshold ----
        # TODO: smem_ptcnt is not always needed? only for the last block_count_ge.
        # Do we have methods to reduce its write?
        thr_init = s_thr[0]
        self.block_count_ge(
            input_row,
            slice_start,
            slice_end,
            thr_init,
            smem_ptcnt,
            smem_wcnt,
            s_iscalars,
            s_cluster_partial,
            tidx,
            warp_id,
            lane,
            smem_input=smem_input,
            do_cluster_sync=do_cluster_sync,
        )

        # tid==0 classifies the initial count.
        if tidx == 0:
            c0 = s_iscalars[0]
            t0 = s_thr[0]
            if c0 >= cutlass.Int32(kK) and c0 <= cutlass.Int32(kCC):
                s_iscalars[1] = cutlass.Int32(1)  # done = 1 (converged)
            elif c0 > cutlass.Int32(kCC):
                # too many → threshold is the new lower bound (search HIGHER)
                s_thr[1] = t0
                s_iscalars[2] = c0
            else:
                # too few → threshold is the new upper bound (search LOWER)
                s_thr[2] = t0
                s_iscalars[3] = c0
        cute.arch.barrier()

        # ---- Secant refinement loop ----
        it = cutlass.Int32(0)
        while it < cutlass.Int32(self.MAX_REFINE_ITERS) and s_iscalars[1] == cutlass.Int32(0):
            # tid==0 computes new threshold via secant interpolation.
            if tidx == 0:
                vlo = s_thr[1]
                vhi = s_thr[2]
                clo = s_iscalars[2]
                chi = s_iscalars[3]
                rng = vhi - vlo
                nv = cutlass.Float32(0.0)
                if clo > chi and rng > cutlass.Float32(1e-10):
                    f = cutlass.Float32(clo - cutlass.Int32(kFTarget)) / cutlass.Float32(clo - chi)
                    # clamp f to [0.05, 0.95]
                    f = cute.arch.fmax(cutlass.Float32(0.05), f)
                    f = _fmin_f32_inline(f, cutlass.Float32(0.95))
                    if it == cutlass.Int32(0):
                        # iter 0: f = min(f, 0.5)  — runtime compare (matches CUDA)
                        f = _fmin_f32_inline(f, cutlass.Float32(0.5))
                    nv = vlo + rng * f
                else:
                    nv = (vlo + vhi) * cutlass.Float32(0.5)

                # clamp nv into (vlo, vhi) range
                if nv <= vlo:
                    nv = vlo + rng * cutlass.Float32(0.05)
                if nv >= vhi:
                    nv = vhi - rng * cutlass.Float32(0.05)

                if nv == vlo or nv == vhi:
                    # Bracket exhausted — try midpoint, else terminal.
                    nv = (vlo + vhi) * cutlass.Float32(0.5)
                    if nv == vlo or nv == vhi:
                        # ADJACENT-FLOAT bracket: every value in
                        # [vlo, vhi) is bitwise-equal to vlo. Low side
                        # overflowing the candidate buffer AND high side
                        # undershooting K means the boundary sits inside
                        # a bitwise-equal plateau wider than kC — record
                        # the plateau terminal (done = 3) and keep the
                        # sure-winner threshold vhi.
                        if clo > cutlass.Int32(kCC) and chi < cutlass.Int32(kK):
                            s_thr[0] = vhi
                            s_iscalars[1] = cutlass.Int32(3)  # done = 3 (plateau)
                        else:
                            s_thr[0] = vlo
                            s_iscalars[1] = cutlass.Int32(2)  # done = 2 (give up)
                    else:
                        s_thr[0] = nv
                else:
                    s_thr[0] = nv
            cute.arch.barrier()

            # Re-check done (tid==0 may have set it to 2)
            if s_iscalars[1] == cutlass.Int32(0):
                new_thr = s_thr[0]
                self.block_count_ge(
                    input_row,
                    slice_start,
                    slice_end,
                    new_thr,
                    smem_ptcnt,
                    smem_wcnt,
                    s_iscalars,
                    s_cluster_partial,
                    tidx,
                    warp_id,
                    lane,
                    smem_input=smem_input,
                    do_cluster_sync=do_cluster_sync,
                )
                # tid==0 classifies the new count.
                if tidx == 0:
                    c_new = s_iscalars[0]
                    t_new = s_thr[0]
                    if c_new >= cutlass.Int32(kK) and c_new <= cutlass.Int32(kCC):
                        s_iscalars[1] = cutlass.Int32(1)
                    elif c_new > cutlass.Int32(kCC):
                        s_thr[1] = t_new
                        s_iscalars[2] = c_new
                    else:
                        s_thr[2] = t_new
                        s_iscalars[3] = c_new
                cute.arch.barrier()
            it = it + cutlass.Int32(1)

        # ---- Budget-exhausted plateau collapse ----
        # The refine budget can run out while the bracket is still wide: the
        # secant step keeps making progress (the bracket shrinks every
        # iteration) but a tie plateau wider than kC admits no threshold, so
        # the count never lands in [kK, kCC]. In exactly that signature -
        # count(>= v_lo) > kCC AND count(>= v_hi) < kK, both counts current -
        # collapse the bracket by pure bisection until the ends are ADJACENT
        # floats; every value in [v_lo, v_hi) is then bitwise-equal, so the
        # plateau terminal (done = 3) is exact and Phase 4 completes the row
        # from that tie class. A count landing in [kK, kCC] mid-collapse
        # converges normally. Anything else keeps the legacy give-up below.
        if (
            s_iscalars[1] == cutlass.Int32(0)
            and s_iscalars[2] > cutlass.Int32(kCC)
            and s_iscalars[3] >= cutlass.Int32(0)
            and s_iscalars[3] < cutlass.Int32(kK)
        ):
            itc = cutlass.Int32(0)
            while itc < cutlass.Int32(64) and s_iscalars[1] == cutlass.Int32(0):
                if tidx == 0:
                    vlo_c = s_thr[1]
                    vhi_c = s_thr[2]
                    mid_c = (vlo_c + vhi_c) * cutlass.Float32(0.5)
                    if mid_c == vlo_c or mid_c == vhi_c:
                        s_thr[0] = vhi_c
                        s_iscalars[1] = cutlass.Int32(3)  # plateau terminal
                    else:
                        s_thr[0] = mid_c
                cute.arch.barrier()
                if s_iscalars[1] == cutlass.Int32(0):
                    self.block_count_ge(
                        input_row,
                        slice_start,
                        slice_end,
                        s_thr[0],
                        smem_ptcnt,
                        smem_wcnt,
                        s_iscalars,
                        s_cluster_partial,
                        tidx,
                        warp_id,
                        lane,
                        smem_input=smem_input,
                        do_cluster_sync=do_cluster_sync,
                    )
                    if tidx == 0:
                        c_c = s_iscalars[0]
                        t_c = s_thr[0]
                        if c_c >= cutlass.Int32(kK) and c_c <= cutlass.Int32(kCC):
                            s_iscalars[1] = cutlass.Int32(1)
                        elif c_c > cutlass.Int32(kCC):
                            s_thr[1] = t_c
                            s_iscalars[2] = c_c
                        else:
                            s_thr[2] = t_c
                            s_iscalars[3] = c_c
                    cute.arch.barrier()
                itc = itc + cutlass.Int32(1)
            if s_iscalars[1] == cutlass.Int32(3):
                # recount at the terminal threshold so Phase 3's cached
                # per-thread counts describe the sure-winner set.
                self.block_count_ge(
                    input_row,
                    slice_start,
                    slice_end,
                    s_thr[0],
                    smem_ptcnt,
                    smem_wcnt,
                    s_iscalars,
                    s_cluster_partial,
                    tidx,
                    warp_id,
                    lane,
                    smem_input=smem_input,
                    do_cluster_sync=do_cluster_sync,
                )
                cute.arch.barrier()

        # ---- Post-loop fallback: if still not done, force threshold ----
        if tidx == 0:
            if s_iscalars[1] == cutlass.Int32(0):
                if s_iscalars[2] <= cutlass.Int32(kCC * 2):
                    s_thr[0] = s_thr[1]  # threshold = val_lo
                else:
                    s_thr[0] = s_thr[2]  # threshold = val_hi
                s_iscalars[1] = cutlass.Int32(2)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Phase 3: Ballot-free candidate collect
    # If P2 ended with done=2 (bracket exhausted), first run a retry-shrink
    # loop (≤10 iters) to bring cand_count <= kCC.
    # Then reuse cached smem_ptcnt → warp prefix sum → block prefix sum
    # → stream-write keys[]/vals[] for v >= threshold.
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_collect_candidates(
        self,
        input_row,
        N,
        slice_start,
        slice_end,
        smem_keys,
        smem_vals,
        smem_ptcnt,
        smem_wcnt,
        s_thr,
        s_iscalars,
        s_cluster_partial,
        tidx,
        warp_id,
        lane,
        do_cluster_sync,  # bool: False = cs=1 / short-row degrade (skip cluster sync)
        smem_input=None,  # optional SMEM-cached slice
        smem_active=None,  # block-skip: int16 active list (reused, not rebuilt)
        s_active_cnt=None,  # block-skip: [0]=list length, [1]=list-current flag
    ):
        """Retry-shrink (when P2 didn't converge) + prefix sum + stream-write.

        On exit, smem_keys[0 : cand_count] / smem_vals[0 : cand_count]
        hold every (value, index) pair with value >= threshold, in the
        scan order each thread produces them. Uses smem_ptcnt cached by
        the last block_count_ge in Phase 2 (or by the retry-shrink below).
        """
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        num_threads = cutlass.const_expr(self.num_threads)

        # ---- Retry-shrink loop (only if P2 didn't converge cleanly) ----
        # Phase 3 runs cluster-parallel — every CTA shrinks against its own
        # slice but must agree on the threshold update. block_count_ge
        # always aggregates across the cluster, so every CTA sees the same
        # cluster-wide cand_count; cs=1 makes the aggregation a no-op.
        if s_iscalars[1] != cutlass.Int32(1):
            # Any dense re-count invalidates the block-skip active list. The
            # list is built from the loosest kept rung of the probe that set
            # the flag, so it is a superset only at or above that rung, and
            # the repair below anchors and bisects underneath it; separately,
            # the compact stream-write replays the list walk against the
            # smem_ptcnt of its matching compact pass, which a dense re-count
            # overwrites. Cleared once here rather than per re-count: done==1
            # rows never reach this block, so the hot path keeps its compact
            # write.
            if cutlass.const_expr(self.enable_block_skip):
                if tidx == cutlass.Int32(0):
                    s_active_cnt[1] = cutlass.Int32(0)
                cute.arch.barrier()

            # Re-count with current threshold (may already have stale cand_count)
            cur_thr = s_thr[0]
            self.block_count_ge(
                input_row,
                slice_start,
                slice_end,
                cur_thr,
                smem_ptcnt,
                smem_wcnt,
                s_iscalars,
                s_cluster_partial,
                tidx,
                warp_id,
                lane,
                smem_input=smem_input,
                do_cluster_sync=do_cluster_sync,
            )
            # Two-sided repair: the old retry only guarded overflow, so an
            # undershooting threshold shipped a -1-padded, silently wrong
            # top-K. Anchor the untested bracket end at a float extreme, then
            # bisect on the signed order-key image (provable collapse).
            if tidx == 0:
                c0 = s_iscalars[0]
                if c0 > cutlass.Int32(kCC):
                    s_thr[1] = s_thr[0]
                    s_thr[2] = cutlass.Float32(self.FLT_MAX)
                elif c0 < cutlass.Int32(kK):
                    s_thr[2] = s_thr[0]
                    s_thr[1] = cutlass.Float32(self.NEG_FLT_MAX)
            cute.arch.barrier()

            rs = cutlass.Int32(0)
            collapsed = cutlass.Int32(0)
            while (
                rs < cutlass.Int32(48)
                and (s_iscalars[0] > cutlass.Int32(kCC) or s_iscalars[0] < cutlass.Int32(kK))
                and collapsed == cutlass.Int32(0)
            ):
                mid_f, adj = order_key_mid_f32(s_thr[1], s_thr[2])
                if adj:
                    collapsed = cutlass.Int32(1)
                if collapsed == cutlass.Int32(0):
                    if tidx == 0:
                        s_thr[0] = mid_f
                    cute.arch.barrier()
                    self.block_count_ge(
                        input_row,
                        slice_start,
                        slice_end,
                        s_thr[0],
                        smem_ptcnt,
                        smem_wcnt,
                        s_iscalars,
                        s_cluster_partial,
                        tidx,
                        warp_id,
                        lane,
                        smem_input=smem_input,
                        do_cluster_sync=do_cluster_sync,
                    )
                    if tidx == 0:
                        c_rs = s_iscalars[0]
                        if c_rs > cutlass.Int32(kCC):
                            s_thr[1] = s_thr[0]
                        elif c_rs < cutlass.Int32(kK):
                            s_thr[2] = s_thr[0]
                    cute.arch.barrier()
                rs = rs + cutlass.Int32(1)

        # ---- Warp prefix sum over smem_ptcnt ----
        # my_total_qual = per-thread count cached by last block_count_ge.
        my_total_qual = smem_ptcnt[tidx]
        tp = my_total_qual

        # 5-level shfl_up_sync inclusive scan within warp.
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            other = cute.arch.shuffle_sync_up(tp, off_v, mask_and_clamp=0)
            if lane >= cutlass.Int32(off_v):
                tp = tp + other

        my_excl_offset = tp - my_total_qual
        # Warp total = lane 31's tp; broadcast via shfl_sync_bfly (or
        # cross-lane read: shuffle_sync_op with lane=31).
        warp_total = cute.arch.shuffle_sync(tp, cutlass.Int32(self.WARP_SIZE - 1))

        if lane == 0:
            smem_wcnt[warp_id] = warp_total
        cute.arch.barrier()

        # Exclusive prefix sum over num_warps warp totals.
        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            # NEW: warp-parallel via block_scan.warp_scan (Hillis-Steele
            # inclusive scan, log2(num_warps) shfl_up steps). Exclusive
            # prefix = inclusive - val. Total = inclusive at last lane.
            if warp_id == cutlass.Int32(0):
                if lane < cutlass.Int32(self.num_warps):
                    val = smem_wcnt[lane]
                    inclusive = warp_scan(val, tidx, lane, num_threads_per_warp=self.num_warps)
                    smem_wcnt[lane] = inclusive - val  # exclusive prefix
                    if lane == cutlass.Int32(self.num_warps - 1):
                        s_iscalars[0] = inclusive  # cand_count (total)
        else:
            # tid==0 serial exclusive prefix.
            if tidx == 0:
                total = cutlass.Int32(0)
                for w in cutlass.range_constexpr(self.num_warps):
                    cnt = smem_wcnt[w]
                    smem_wcnt[w] = total
                    total = total + cnt
                s_iscalars[0] = total
        cute.arch.barrier()

        # Each thread's write base = warp-prefix + intra-warp exclusive offset.
        my_base = smem_wcnt[warp_id]
        my_write_pos = my_base + my_excl_offset

        # ---- Stream-write loop ----
        # Scan bound is this CTA's slice [slice_start, slice_end), not the
        # full row. Phase 2's last block_count_ge populated smem_ptcnt with
        # slice-local counts, so the prefix sum above already reflects
        # "candidates this thread will write" within the slice. After this
        # function returns, smem_keys[0 .. local_cand_count) holds this
        # CTA's slice candidates; the kernel-level handoff DSMEM-gathers
        # peers' chunks into the leader's smem_keys before Phase 4. At
        # cluster_size==1 the slice is [0, N) and behavior is identical to
        # the single-CTA path.
        thr_final = s_thr[0]
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        copy_atom = self._make_load_copy_atom()
        row_addr = input_row.iterator.toint()
        step_elem = cutlass.const_expr(num_threads * vec_w)
        # Hoisted SMEM window bases (one S2R here vs one per emitted
        # candidate below — this loop is the kernel's biggest instruction
        # region at production shapes).
        keys_base = smem_keys.iterator.toint()
        vals_base = smem_vals.iterator.toint()

        slice_len = slice_end - slice_start
        # When reading from the cached slice, scan indices are slice-LOCAL;
        # the GMEM path uses global indices. smem_vals always stores the
        # GLOBAL position so Phase 4 / writeback stays consistent.
        if cutlass.const_expr(smem_input is not None):
            smem_addr = smem_input.iterator.toint()
            n_aligned = (slice_len // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
            N_local = slice_len
            ic = tidx * cutlass.Int32(vec_w)
        else:
            n_aligned = slice_start + (slice_len // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
            N_local = slice_end
            ic = slice_start + tidx * cutlass.Int32(vec_w)
        wc = my_write_pos
        step = cutlass.Int32(step_elem)

        # ---- block-skip compact stream-write ----
        # Reuses the active list left by the R0 compact count pass (same
        # ownership walk: my_blk_slot's list slots in ascending order, so
        # each thread produces its candidates in the SAME per-thread order
        # the count pass counted them — the prefix-sum positions match).
        # Only taken when the list is CURRENT (s_active_cnt[1] == 1, set by
        # the build; cleared on any dense fallback re-count).
        skip_wr = cutlass.Int32(0)
        park_cursors = False  # Python bool: no scf.if when block skip is off
        if cutlass.const_expr(self.enable_block_skip and smem_active is not None):
            if s_active_cnt[1] == cutlass.Int32(1):
                skip_wr = cutlass.Int32(1)
            park_cursors = skip_wr == cutlass.Int32(1)
        if cutlass.const_expr(self.enable_block_skip and smem_active is not None):
            if skip_wr == cutlass.Int32(1):
                # head region first — same per-thread order as the count pass
                head_end_w = (
                    (slice_start + cutlass.Int32(self.SKIP_BLOCK - 1))
                    >> cutlass.Int32(self.SKIP_BLOCK_LOG2)
                ) << cutlass.Int32(self.SKIP_BLOCK_LOG2)
                if head_end_w > slice_end:
                    head_end_w = slice_end
                hh_w = slice_start + tidx
                while hh_w < head_end_w:
                    vh_w = self._load_fp32(input_row, hh_w)
                    if vh_w >= thr_final and wc < cutlass.Int32(kCC):
                        smem_keys[wc] = vh_w
                        smem_vals[wc] = hh_w
                        wc = wc + cutlass.Int32(1)
                    hh_w = hh_w + cutlass.Int32(num_threads)
                chunks_per_block_w = cutlass.const_expr(
                    self.SKIP_BLOCK // (self.vec_bits // self.dtype.width)
                )
                blocks_per_iter_w = cutlass.const_expr(self.num_threads // chunks_per_block_w)
                my_blk_slot_w = tidx // cutlass.Int32(chunks_per_block_w)
                my_chunk0_w = tidx % cutlass.Int32(chunks_per_block_w)
                cnt_active_w = s_active_cnt[0]
                wfrag = cute.make_rmem_tensor((vec_w,), self.dtype)
                li_w = my_blk_slot_w
                while li_w < cnt_active_w:
                    blk_w = self._list_ld(smem_active, li_w)
                    pos0_w = blk_w * cutlass.Int32(self.SKIP_BLOCK) + my_chunk0_w * cutlass.Int32(
                        vec_w
                    )
                    if pos0_w + cutlass.Int32(vec_w) <= slice_end:
                        src_ptr_w = cute.make_ptr(
                            self.dtype,
                            row_addr + cutlass.Int64(pos0_w) * cutlass.Int64(elem_bytes),
                            cute.AddressSpace.gmem,
                            assumed_align=vec_align,
                        )
                        cute.copy(
                            copy_atom,
                            cute.make_tensor(src_ptr_w, cute.make_layout((vec_w,))),
                            wfrag,
                        )
                        for j in cutlass.range_constexpr(vec_w):
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                vj = wfrag[j]
                            else:
                                vj = cutlass.Float32(wfrag[j])
                            if vj >= thr_final and wc < cutlass.Int32(kCC):
                                smem_keys[wc] = vj
                                smem_vals[wc] = pos0_w + cutlass.Int32(j)
                                wc = wc + cutlass.Int32(1)
                    else:
                        jj_w = pos0_w
                        while jj_w < slice_end:
                            v_w = self._load_fp32(input_row, jj_w)
                            if v_w >= thr_final and wc < cutlass.Int32(kCC):
                                smem_keys[wc] = v_w
                                smem_vals[wc] = jj_w
                                wc = wc + cutlass.Int32(1)
                            jj_w = jj_w + cutlass.Int32(1)
                    li_w = li_w + cutlass.Int32(blocks_per_iter_w)

        # When the compact write ran, park the dense cursors at the end so
        # all three dense loops below (4-way, vec tail, scalar tail) fall
        # through without re-indenting them.
        if park_cursors:
            ic = N_local
            n_aligned = N_local

        # Phase3 unrolling: master gated by self.enable_phase3_unroll.
        # When OFF, only the tail 1-way loop runs (matches the pre-unroll
        # state of phase3_collect). When ON, the inner enable_unroll_4
        # controls the 4-way fast path.
        if self.enable_phase3_unroll:
            # Fast path: 4-way unrolled vec loop (4 loading instructions in flight).
            if self.enable_unroll_4:
                rng_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
                big_iters = cutlass.Int32(0)
                if N_local > ic + cutlass.Int32(vec_w - 1):
                    big_iters = (N_local - ic - cutlass.Int32(vec_w)) // cutlass.Int32(
                        step_elem
                    ) + cutlass.Int32(1)

                for k in cutlass.range(big_iters, unroll=4):
                    ic_local = ic + k * cutlass.Int32(step_elem)
                    if cutlass.const_expr(smem_input is not None):
                        src_ptr_k = cute.make_ptr(
                            self.dtype,
                            smem_addr + cutlass.Int64(ic_local) * cutlass.Int64(elem_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=vec_align,
                        )
                        global_base = slice_start + ic_local
                    else:
                        src_ptr_k = cute.make_ptr(
                            self.dtype,
                            row_addr + cutlass.Int64(ic_local) * cutlass.Int64(elem_bytes),
                            cute.AddressSpace.gmem,
                            assumed_align=vec_align,
                        )
                        global_base = ic_local
                    src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                    cute.copy(copy_atom, src_k, rng_frag)
                    for j in cutlass.range_constexpr(vec_w):
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            vj = rng_frag[j]
                        else:
                            vj = cutlass.Float32(rng_frag[j])
                        if vj >= thr_final and wc < cutlass.Int32(kCC):
                            self._smem_st(cutlass.Float32, keys_base, wc, vj)
                            self._smem_st(
                                cutlass.Int32, vals_base, wc, global_base + cutlass.Int32(j)
                            )
                            wc = wc + cutlass.Int32(1)
                # Advance ic past all consumed vec_w-aligned positions.
                ic = ic + big_iters * cutlass.Int32(step_elem)

        # Tail vec loop: 1-way, handles remainder < 2*step.
        tail_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
        while ic + cutlass.Int32(vec_w - 1) < N_local:
            if cutlass.const_expr(smem_input is not None):
                src_ptr = cute.make_ptr(
                    self.dtype,
                    smem_addr + cutlass.Int64(ic) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.smem,
                    assumed_align=vec_align,
                )
                global_base_t = slice_start + ic
            else:
                src_ptr = cute.make_ptr(
                    self.dtype,
                    row_addr + cutlass.Int64(ic) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.gmem,
                    assumed_align=vec_align,
                )
                global_base_t = ic
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                if vj >= thr_final and wc < cutlass.Int32(kCC):
                    self._smem_st(cutlass.Float32, keys_base, wc, vj)
                    self._smem_st(cutlass.Int32, vals_base, wc, global_base_t + cutlass.Int32(j))
                    wc = wc + cutlass.Int32(1)
            ic = ic + step

        # Tail scalar loop (slice_len % vec_w)
        it = n_aligned + tidx
        while it < N_local:
            if cutlass.const_expr(smem_input is not None):
                v = smem_input[it]
                if cutlass.const_expr(self.dtype != cutlass.Float32):
                    v = cutlass.Float32(v)
                pos_global = slice_start + it
            else:
                v = self._load_fp32(input_row, it)
                pos_global = it
            if v >= thr_final and wc < cutlass.Int32(kCC):
                self._smem_st(cutlass.Float32, keys_base, wc, v)
                self._smem_st(cutlass.Int32, vals_base, wc, pos_global)
                wc = wc + cutlass.Int32(1)
            it = it + cutlass.Int32(num_threads)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # block_fused_snap_iter — P4 snap convergence inner step
    # ------------------------------------------------------------------
    @cute.jit
    def block_fused_snap_iter(
        self,
        keys_base,  # hoisted SMEM window base of smem_keys (iterator.toint())
        smem_wcnt,
        smem_hist,  # reused as scratch for s_up/s_down warp aggregates
        s_thr,
        s_iscalars,
        count,
        tidx,
        warp_id,
        lane,
    ):
        """One iteration of histogram snap. Updates s_iscalars[2]=cnt_lo (cge),
        s_iscalars[3]=cnt_hi (cgt), and s_thr[0]=threshold (moves toward
        the cnt-in-(kK_GT, kK_GE) bracket).
        """
        kK = cutlass.const_expr(self.top_k)
        num_threads = cutlass.const_expr(self.num_threads)
        thr = s_thr[0]

        lge = cutlass.Int32(0)
        lgt = cutlass.Int32(0)
        s_up = cutlass.Float32(self.FLT_MAX)
        s_down = cutlass.Float32(self.NEG_FLT_MAX)

        isi = tidx
        while isi < count:
            v = self._smem_ld(cutlass.Float32, keys_base, isi)
            if v >= thr:
                lge = lge + cutlass.Int32(1)
            if v > thr:
                lgt = lgt + cutlass.Int32(1)
                # s_up = min(s_up, v) — hot path in block_fused_snap_iter (~10us)
                s_up = _fmin_f32_inline(s_up, v)
            if v < thr:
                s_down = cute.arch.fmax(s_down, v)
            isi = isi + cutlass.Int32(num_threads)

        # Pack lge/lgt into one int32 so the warp reduce sums both counts
        # in a single shuffle. Safe as long as each per-warp count
        # stays < 2^16; lge/lgt are bounded by cand_count ≤ kC ≤ 6144
        # (GvrParams), so we're well clear. Bumping kC past 65535 would
        # silently corrupt this packing.
        packed = (lge << cutlass.Int32(16)) | lgt
        packed = self.warp_reduce_sum_i32(packed)
        s_up = self.warp_reduce_min_f32(s_up)
        s_down = self.warp_reduce_max_f32(s_down)

        # Lane 0 stages results into warp slots (smem_hist[0..NW-1] = s_up,
        # smem_hist[NW..2*NW-1] = s_down stored as int32 bit-cast).
        if lane == 0:
            smem_wcnt[warp_id] = packed
            smem_hist[warp_id] = float_as_uint32(s_up)
            smem_hist[self.num_warps + warp_id] = float_as_uint32(s_down)
        cute.arch.barrier()

        # 3-way block reduce + threshold bound update.
        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            # Warp-parallel 3-way reduce in warp 0.
            if warp_id == cutlass.Int32(0):
                v_tp = cutlass.Int32(0)
                v_up = cutlass.Float32(self.FLT_MAX)
                v_dn = cutlass.Float32(self.NEG_FLT_MAX)
                if lane < cutlass.Int32(self.num_warps):
                    v_tp = smem_wcnt[lane]
                    vu_bits = smem_hist[lane]
                    vd_bits = smem_hist[self.num_warps + lane]
                    v_up = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, vu_bits.ir_value())
                    )
                    v_dn = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, vd_bits.ir_value())
                    )
                tp = self.warp_reduce_sum_i32(v_tp)
                total_up = self.warp_reduce_min_f32(v_up)
                total_down = self.warp_reduce_max_f32(v_dn)
                if lane == cutlass.Int32(0):
                    cge = tp >> cutlass.Int32(16)
                    cgt = tp & cutlass.Int32(0xFFFF)
                    s_iscalars[2] = cge
                    s_iscalars[3] = cgt
                    if cgt >= cutlass.Int32(kK):
                        if total_up < cutlass.Float32(self.FLT_MAX):
                            s_thr[0] = total_up
                    elif cge < cutlass.Int32(kK):
                        if total_down > cutlass.Float32(self.NEG_FLT_MAX):
                            s_thr[0] = total_down
        else:
            # tid==0 serial 3-way reduce.
            if tidx == 0:
                tp = cutlass.Int32(0)
                total_up = cutlass.Float32(self.FLT_MAX)
                total_down = cutlass.Float32(self.NEG_FLT_MAX)
                for w in cutlass.range_constexpr(self.num_warps):
                    tp = tp + smem_wcnt[w]
                    vu = llvm.bitcast(cutlass.Float32.mlir_type, smem_hist[w].ir_value())
                    vd = llvm.bitcast(
                        cutlass.Float32.mlir_type, smem_hist[self.num_warps + w].ir_value()
                    )
                    vu_w = cutlass.Float32(vu)
                    vd_w = cutlass.Float32(vd)
                    total_up = _fmin_f32_inline(total_up, vu_w)
                    total_down = cute.arch.fmax(total_down, vd_w)

                cge = tp >> cutlass.Int32(16)
                cgt = tp & cutlass.Int32(0xFFFF)
                s_iscalars[2] = cge
                s_iscalars[3] = cgt
                if cgt >= cutlass.Int32(kK):
                    if total_up < cutlass.Float32(self.FLT_MAX):
                        s_thr[0] = total_up
                elif cge < cutlass.Int32(kK):
                    if total_down > cutlass.Float32(self.NEG_FLT_MAX):
                        s_thr[0] = total_down
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # P4 helpers: histogram build + parallel k-th bin search. Factored
    # out so the level-2 refinement can rerun both over a narrowed window.
    # ------------------------------------------------------------------
    @cute.jit
    def _hist_build(self, keys_base, smem_hist, cand_count, lo, inv, tidx):
        """Zero smem_hist[0:kBins], then histogram keys[0:cand_count] with
        bin = clamp(int((v - lo) * inv), 0, kBins-1). Out-of-window values
        clamp into the edge bins, which keeps cumulative counts from the
        top exact for the k-th search (everything above the window lands
        in the top bin). Barrier after the zero pass and after the build."""
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        i6 = tidx
        while i6 < cutlass.Int32(kBins):
            smem_hist[i6] = cutlass.Int32(0)
            i6 = i6 + cutlass.Int32(num_threads)
        cute.arch.barrier()
        i7 = tidx
        while i7 < cand_count:
            vk = self._smem_ld(cutlass.Float32, keys_base, i7)
            bin_f = (vk - lo) * inv
            # Clamp in the FLOAT domain before the int cast: fptosi is
            # undefined for out-of-range/NaN inputs at the IR level (PTX
            # cvt.rzi saturates, but LLVM may optimize on the poison).
            # fmax first canonicalizes NaN to 0; the pair keeps the
            # edge-bin clamping semantics bit-identical for in-range
            # values.
            bin_f = cute.arch.fmax(bin_f, cutlass.Float32(0.0))
            bin_f = _fmin_f32_inline(bin_f, cutlass.Float32(kBins - 1))
            bin_i = cutlass.Int32(bin_f)
            atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
            i7 = i7 + cutlass.Int32(num_threads)
        cute.arch.barrier()

    @cute.jit
    def _kth_bin_search(
        self, smem_hist, smem_wcnt, s_thr, s_iscalars, lo, binw, tidx, warp_id, lane
    ):
        """Parallel k-th bin search (3-step, high→low). Writes
        s_thr[0] = lower edge of the selected bin (lo + bidx*binw) and
        s_iscalars[4] = selected bin's count (gates the level-2 histogram
        refinement). Clobbers s_iscalars[2]/[3] as staging (both are
        rewritten by the snap loop before anyone else reads them).
        Trailing barrier."""
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        # Step 1: each warp sums BINS_PER_WARP bins (high→low slice).
        # Lane-parallel when the slice divides evenly across the warp:
        # each lane sums bins_per_warp/32 bins + one warp reduce, instead
        # of every lane redundantly walking a bins_per_warp-deep serial
        # LDS+IADD dependency chain (~7% of stall samples at N=8K).
        warp_bin_sum = cutlass.Int32(0)
        if cutlass.const_expr(bins_per_warp % self.WARP_SIZE == 0):
            for jm in cutlass.range_constexpr(bins_per_warp // self.WARP_SIZE):
                bidx_s = (
                    cutlass.Int32(kBins - 1)
                    - warp_id * cutlass.Int32(bins_per_warp)
                    - (lane + cutlass.Int32(jm * self.WARP_SIZE))
                )
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
            warp_bin_sum = self.warp_reduce_sum_i32(warp_bin_sum)
        else:
            for jb in cutlass.range_constexpr(bins_per_warp):
                bidx_s = (
                    cutlass.Int32(kBins - 1)
                    - warp_id * cutlass.Int32(bins_per_warp)
                    - cutlass.Int32(jb)
                )
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
        if lane == 0:
            smem_wcnt[warp_id] = warp_bin_sum
        cute.arch.barrier()

        # Step 2: tid==0 finds target warp; stores prefix-count + warp index
        # into s_iscalars[2] (=cnt_lo: prefix before target warp)
        # and s_iscalars[3] (=cnt_hi: target warp index)
        if tidx == 0:
            cum = cutlass.Int32(0)
            tw = cutlass.Int32(self.num_warps - 1)
            found = cutlass.Int32(0)
            for w2 in cutlass.range_constexpr(self.num_warps):
                cum = cum + smem_wcnt[w2]
                if cum >= cutlass.Int32(kK) and found == cutlass.Int32(0):
                    tw = cutlass.Int32(w2)
                    found = cutlass.Int32(1)
            # Recompute prefix BEFORE target warp
            cum2 = cutlass.Int32(0)
            for w3 in cutlass.range_constexpr(self.num_warps):
                if cutlass.Int32(w3) < tw:
                    cum2 = cum2 + smem_wcnt[w3]
            s_iscalars[2] = cum2  # prefix
            s_iscalars[3] = tw  # target warp index
        cute.arch.barrier()

        # Step 3: target warp's lane 0 scans BINS_PER_WARP bins →
        # threshold. Single-thread serial; the unrolled
        # range_constexpr beats a runtime `for+break` (tried it: -544
        # SASS insts but -7pp fp32 / -14pp bf16, since the
        # branch/counter overhead in a single thread dominates the
        # static math).
        target_warp = s_iscalars[3]
        if warp_id == target_warp and lane == cutlass.Int32(0):
            base_cum = s_iscalars[2]
            thr_local = lo
            sel_cnt = cutlass.Int32(0)
            set_done = cutlass.Int32(0)
            for jb2 in cutlass.range_constexpr(bins_per_warp):
                bidx2 = (
                    cutlass.Int32(kBins - 1)
                    - target_warp * cutlass.Int32(bins_per_warp)
                    - cutlass.Int32(jb2)
                )
                cnt_here = smem_hist[bidx2]
                base_cum = base_cum + cnt_here
                if base_cum >= cutlass.Int32(kK) and set_done == cutlass.Int32(0):
                    thr_local = lo + cutlass.Float32(bidx2) * binw
                    sel_cnt = cnt_here
                    set_done = cutlass.Int32(1)
            s_thr[0] = thr_local
            s_iscalars[4] = sel_cnt
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # _kth_bin_search_rw — redundant-warp variant (p4_warp_redundant).
    # Step 1 stages per-warp bin-slice sums exactly like _kth_bin_search
    # (the ONE barrier). Then EVERY warp redundantly (a) walks the
    # num_warps slot sums with broadcast SMEM reads + predicated adds to
    # locate the target warp, and (b) lane-parallel walks the target
    # slice — each lane owns a contiguous descending sub-range, a
    # shuffle-up prefix + the unique sub-range crossing test find the
    # k-th bin in O(bins_per_warp/32) LDS instead of a 64-deep serial
    # LDS+IADD chain in one thread. Same inputs in the same order on
    # every warp -> bit-identical results, so there is no leader, no
    # publish barrier, and no s_thr/s_iscalars staging; the selected
    # (threshold, bin count) return in registers.
    # ------------------------------------------------------------------
    @cute.jit
    def _kth_bin_search_rw(self, smem_hist, smem_wcnt, lo, binw, tidx, warp_id, lane):
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        # Step 1: identical staging to _kth_bin_search.
        warp_bin_sum = cutlass.Int32(0)
        if cutlass.const_expr(bins_per_warp % self.WARP_SIZE == 0):
            for jm in cutlass.range_constexpr(bins_per_warp // self.WARP_SIZE):
                bidx_s = (
                    cutlass.Int32(kBins - 1)
                    - warp_id * cutlass.Int32(bins_per_warp)
                    - (lane + cutlass.Int32(jm * self.WARP_SIZE))
                )
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
            warp_bin_sum = self.warp_reduce_sum_i32(warp_bin_sum)
        else:
            for jb in cutlass.range_constexpr(bins_per_warp):
                bidx_s = (
                    cutlass.Int32(kBins - 1)
                    - warp_id * cutlass.Int32(bins_per_warp)
                    - cutlass.Int32(jb)
                )
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
        if lane == 0:
            smem_wcnt[warp_id] = warp_bin_sum
        cute.arch.barrier()

        # Step 2 (every warp, lane-parallel): lane w holds slot w; an
        # inclusive idx-shuffle scan + ballot locate the target warp.
        # (shuffle_sync with a computed source lane is the working shfl
        # idiom; shuffle_sync_up ignores its offset — probed.)
        v_s = cutlass.Int32(0)
        if lane < cutlass.Int32(self.num_warps):
            v_s = smem_wcnt[lane]
        run2 = v_s
        for d2 in cutlass.range_constexpr(5):
            off2 = cutlass.const_expr(1 << d2)
            src2 = lane - cutlass.Int32(off2)
            if src2 < cutlass.Int32(0):
                src2 = cutlass.Int32(0)
            up2 = cute.arch.shuffle_sync(run2, src2)
            if lane >= cutlass.Int32(off2):
                run2 = run2 + up2
        m2 = cute.arch.vote_ballot_sync(run2 >= cutlass.Int32(kK))
        tw = cutlass.Int32(self.num_warps - 1)
        if m2 != cutlass.Uint32(0):
            low2 = m2 & (cutlass.Uint32(0) - m2)
            tw = cutlass.Int32(cute.arch.popc(low2 - cutlass.Uint32(1)))
        incl_tw = cute.arch.shuffle_sync(run2, tw)
        slot_tw = cute.arch.shuffle_sync(v_s, tw)
        prefix = incl_tw - slot_tw

        # Step 3 (every warp, lane-parallel): lane l owns the contiguous
        # descending positions [l*ppl, (l+1)*ppl) of the target slice.
        ppl = cutlass.const_expr((bins_per_warp + self.WARP_SIZE - 1) // self.WARP_SIZE)
        cnt_frag = cute.make_rmem_tensor((ppl,), cutlass.Int32)
        my_sum = cutlass.Int32(0)
        for j3 in cutlass.range_constexpr(ppl):
            pos = lane * cutlass.Int32(ppl) + cutlass.Int32(j3)
            cnt_j = cutlass.Int32(0)
            if pos < cutlass.Int32(bins_per_warp):
                bidx3 = cutlass.Int32(kBins - 1) - tw * cutlass.Int32(bins_per_warp) - pos
                cnt_j = smem_hist[bidx3]
            cnt_frag[j3] = cnt_j
            my_sum = my_sum + cnt_j
        # Exclusive cross-lane prefix of the lane partial sums via the
        # idx-shuffle scan (5 log-steps; shuffle_sync_up ignores its
        # offset — probed — so the scan uses computed source lanes).
        run3 = my_sum
        for d3 in cutlass.range_constexpr(5):
            off3 = cutlass.const_expr(1 << d3)
            src3 = lane - cutlass.Int32(off3)
            if src3 < cutlass.Int32(0):
                src3 = cutlass.Int32(0)
            up3 = cute.arch.shuffle_sync(run3, src3)
            if lane >= cutlass.Int32(off3):
                run3 = run3 + up3
        base3 = prefix + (run3 - my_sum)

        # Unique crossing: the lane where the running count passes kK.
        thr_loc = lo
        sel_loc = cutlass.Int32(0)
        hit = cutlass.Int32(0)
        r3 = base3
        for j4 in cutlass.range_constexpr(ppl):
            pos4 = lane * cutlass.Int32(ppl) + cutlass.Int32(j4)
            cnt4 = cnt_frag[j4]
            if (
                pos4 < cutlass.Int32(bins_per_warp)
                and r3 < cutlass.Int32(kK)
                and r3 + cnt4 >= cutlass.Int32(kK)
                and hit == cutlass.Int32(0)
            ):
                bidx4 = cutlass.Int32(kBins - 1) - tw * cutlass.Int32(bins_per_warp) - pos4
                thr_loc = lo + cutlass.Float32(bidx4) * binw
                sel_loc = cnt4
                hit = cutlass.Int32(1)
            r3 = r3 + cnt4
        # Broadcast from the (at most one) hitting lane; no hit keeps
        # (lo, 0) — same fallback as _kth_bin_search's set_done guard.
        mask3 = cute.arch.vote_ballot_sync(hit != cutlass.Int32(0))
        thr_out = lo
        sel_out = cutlass.Int32(0)
        if mask3 != cutlass.Uint32(0):
            low = mask3 & (cutlass.Uint32(0) - mask3)
            src = cutlass.Int32(cute.arch.popc(low - cutlass.Uint32(1)))
            thr_out = cute.arch.shuffle_sync(thr_loc, src)
            sel_out = cute.arch.shuffle_sync(sel_loc, src)
        return thr_out, sel_out

    # ------------------------------------------------------------------
    # _p4_coarse_rw - redundant-warp coarse bin search for the fused
    # rank-and-scatter path: returns the straddling bin and the count
    # strictly above it, resolved lane-parallel on every warp (an
    # idx-shuffle scan + ballot locate the target slice, a second scan
    # + the unique crossing test locate the bin inside it). Integer
    # sums are associative, so every warp lands on the same answer
    # bit-for-bit. Mirrors _kth_bin_search_rw (snap path).
    # ------------------------------------------------------------------
    @cute.jit
    def _p4_coarse_rw(self, smem_hist, smem_wcnt, warp_id, lane):
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        warp_bin_sum = cutlass.Int32(0)
        if cutlass.const_expr(bins_per_warp % self.WARP_SIZE == 0):
            for jm in cutlass.range_constexpr(bins_per_warp // self.WARP_SIZE):
                bidx_s = (
                    cutlass.Int32(kBins - 1)
                    - warp_id * cutlass.Int32(bins_per_warp)
                    - (lane + cutlass.Int32(jm * self.WARP_SIZE))
                )
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
            warp_bin_sum = self.warp_reduce_sum_i32(warp_bin_sum)
        else:
            for jb in cutlass.range_constexpr(bins_per_warp):
                bidx_s = (
                    cutlass.Int32(kBins - 1)
                    - warp_id * cutlass.Int32(bins_per_warp)
                    - cutlass.Int32(jb)
                )
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
        if lane == cutlass.Int32(0):
            smem_wcnt[warp_id] = warp_bin_sum
        cute.arch.barrier()

        # locate the target slice (lane w holds slot w)
        v_s = cutlass.Int32(0)
        if lane < cutlass.Int32(self.num_warps):
            v_s = smem_wcnt[lane]
        run2 = v_s
        for d2 in cutlass.range_constexpr(5):
            off2 = cutlass.const_expr(1 << d2)
            src2 = lane - cutlass.Int32(off2)
            if src2 < cutlass.Int32(0):
                src2 = cutlass.Int32(0)
            up2 = cute.arch.shuffle_sync(run2, src2)
            if lane >= cutlass.Int32(off2):
                run2 = run2 + up2
        m2 = cute.arch.vote_ballot_sync(run2 >= cutlass.Int32(kK))
        tw = cutlass.Int32(self.num_warps - 1)
        if m2 != cutlass.Uint32(0):
            low2 = m2 & (cutlass.Uint32(0) - m2)
            tw = cutlass.Int32(cute.arch.popc(low2 - cutlass.Uint32(1)))
        incl_tw = cute.arch.shuffle_sync(run2, tw)
        slot_tw = cute.arch.shuffle_sync(v_s, tw)
        prefix = incl_tw - slot_tw

        # locate the bin inside the target slice
        ppl = cutlass.const_expr((bins_per_warp + self.WARP_SIZE - 1) // self.WARP_SIZE)
        cnt_frag = cute.make_rmem_tensor((ppl,), cutlass.Int32)
        my_sum = cutlass.Int32(0)
        for j3 in cutlass.range_constexpr(ppl):
            pos = lane * cutlass.Int32(ppl) + cutlass.Int32(j3)
            cnt_j = cutlass.Int32(0)
            if pos < cutlass.Int32(bins_per_warp):
                bidx3 = cutlass.Int32(kBins - 1) - tw * cutlass.Int32(bins_per_warp) - pos
                cnt_j = smem_hist[bidx3]
            cnt_frag[j3] = cnt_j
            my_sum = my_sum + cnt_j
        run3 = my_sum
        for d3 in cutlass.range_constexpr(5):
            off3 = cutlass.const_expr(1 << d3)
            src3 = lane - cutlass.Int32(off3)
            if src3 < cutlass.Int32(0):
                src3 = cutlass.Int32(0)
            up3 = cute.arch.shuffle_sync(run3, src3)
            if lane >= cutlass.Int32(off3):
                run3 = run3 + up3
        base3 = prefix + (run3 - my_sum)

        b_loc = cutlass.Int32(kBins - 1)
        ra_loc = prefix
        hit = cutlass.Int32(0)
        r3 = base3
        for j4 in cutlass.range_constexpr(ppl):
            pos4 = lane * cutlass.Int32(ppl) + cutlass.Int32(j4)
            cnt4 = cnt_frag[j4]
            if (
                pos4 < cutlass.Int32(bins_per_warp)
                and r3 < cutlass.Int32(kK)
                and r3 + cnt4 >= cutlass.Int32(kK)
                and hit == cutlass.Int32(0)
            ):
                b_loc = cutlass.Int32(kBins - 1) - tw * cutlass.Int32(bins_per_warp) - pos4
                ra_loc = r3
                hit = cutlass.Int32(1)
            r3 = r3 + cnt4
        mask3 = cute.arch.vote_ballot_sync(hit != cutlass.Int32(0))
        b_out = cutlass.Int32(kBins - 1)
        ra_out = prefix
        if mask3 != cutlass.Uint32(0):
            low = mask3 & (cutlass.Uint32(0) - mask3)
            src = cutlass.Int32(cute.arch.popc(low - cutlass.Uint32(1)))
            b_out = cute.arch.shuffle_sync(b_loc, src)
            ra_out = cute.arch.shuffle_sync(ra_loc, src)
        return b_out, ra_out

    # ------------------------------------------------------------------
    # _p4_fine_rw - redundant-warp variant of the fine sub-bin search,
    # the same transformation _p4_coarse_rw applies one level up: every
    # warp resolves it from the staged per-warp sums with an idx-shuffle
    # scan and a ballot. Integer sums are associative: every warp lands
    # on the same answer.
    # ------------------------------------------------------------------
    @cute.jit
    def _p4_fine_rw(self, smem_hist, smem_wcnt, fbins, rank_above, warp_id, lane):
        kK = cutlass.const_expr(self.top_k)
        fbpw = cutlass.const_expr(fbins // self.num_warps)

        ws = cutlass.Int32(0)
        if cutlass.const_expr(fbpw <= self.WARP_SIZE):
            if lane < cutlass.Int32(fbpw):
                bif = cutlass.Int32(fbins - 1) - warp_id * cutlass.Int32(fbpw) - lane
                ws = smem_hist[bif]
            ws = self.warp_reduce_sum_i32(ws)
        else:
            for jm in cutlass.range_constexpr(fbpw):
                bif = cutlass.Int32(fbins - 1) - warp_id * cutlass.Int32(fbpw) - cutlass.Int32(jm)
                ws = ws + smem_hist[bif]
        if lane == cutlass.Int32(0):
            smem_wcnt[warp_id] = ws
        cute.arch.barrier()

        v_s = cutlass.Int32(0)
        if lane < cutlass.Int32(self.num_warps):
            v_s = smem_wcnt[lane]
        run2 = v_s
        for d2 in cutlass.range_constexpr(5):
            off2 = cutlass.const_expr(1 << d2)
            src2 = lane - cutlass.Int32(off2)
            if src2 < cutlass.Int32(0):
                src2 = cutlass.Int32(0)
            up2 = cute.arch.shuffle_sync(run2, src2)
            if lane >= cutlass.Int32(off2):
                run2 = run2 + up2
        m2 = cute.arch.vote_ballot_sync(rank_above + run2 >= cutlass.Int32(kK))
        tw = cutlass.Int32(self.num_warps - 1)
        if m2 != cutlass.Uint32(0):
            low2 = m2 & (cutlass.Uint32(0) - m2)
            tw = cutlass.Int32(cute.arch.popc(low2 - cutlass.Uint32(1)))
        incl_tw = cute.arch.shuffle_sync(run2, tw)
        slot_tw = cute.arch.shuffle_sync(v_s, tw)
        prefix = rank_above + (incl_tw - slot_tw)

        ppl = cutlass.const_expr((fbpw + self.WARP_SIZE - 1) // self.WARP_SIZE)
        cnt_frag = cute.make_rmem_tensor((ppl,), cutlass.Int32)
        my_sum = cutlass.Int32(0)
        for j3 in cutlass.range_constexpr(ppl):
            pos = lane * cutlass.Int32(ppl) + cutlass.Int32(j3)
            cj = cutlass.Int32(0)
            if pos < cutlass.Int32(fbpw):
                sbi = cutlass.Int32(fbins - 1) - tw * cutlass.Int32(fbpw) - pos
                cj = smem_hist[sbi]
            cnt_frag[j3] = cj
            my_sum = my_sum + cj
        run3 = my_sum
        for d3 in cutlass.range_constexpr(5):
            off3 = cutlass.const_expr(1 << d3)
            src3 = lane - cutlass.Int32(off3)
            if src3 < cutlass.Int32(0):
                src3 = cutlass.Int32(0)
            up3 = cute.arch.shuffle_sync(run3, src3)
            if lane >= cutlass.Int32(off3):
                run3 = run3 + up3
        base3 = prefix + (run3 - my_sum)

        sb_loc = cutlass.Int32(fbins - 1)
        ra_loc = prefix
        hit = cutlass.Int32(0)
        r3 = base3
        for j4 in cutlass.range_constexpr(ppl):
            pos4 = lane * cutlass.Int32(ppl) + cutlass.Int32(j4)
            c4 = cnt_frag[j4]
            if (
                pos4 < cutlass.Int32(fbpw)
                and r3 < cutlass.Int32(kK)
                and r3 + c4 >= cutlass.Int32(kK)
                and hit == cutlass.Int32(0)
            ):
                sb_loc = cutlass.Int32(fbins - 1) - tw * cutlass.Int32(fbpw) - pos4
                ra_loc = r3
                hit = cutlass.Int32(1)
            r3 = r3 + c4
        mask3 = cute.arch.vote_ballot_sync(hit != cutlass.Int32(0))
        sb_out = cutlass.Int32(fbins - 1)
        ra_out = prefix
        if mask3 != cutlass.Uint32(0):
            low = mask3 & (cutlass.Uint32(0) - mask3)
            src = cutlass.Int32(cute.arch.popc(low - cutlass.Uint32(1)))
            sb_out = cute.arch.shuffle_sync(sb_loc, src)
            ra_out = cute.arch.shuffle_sync(ra_loc, src)
        return sb_out, ra_out

    # ------------------------------------------------------------------
    # Phase 4 (alt): fused rank-and-scatter (enable_p4_rank_scatter).
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_rank_scatter(
        self,
        smem_keys,
        smem_vals,
        smem_hist,
        smem_wcnt,
        s_thr,
        s_iscalars,
        output_values_row,
        output_indices_row,
        cand_count,
        tidx,
        warp_id,
        lane,
        ext_range_flag=None,  # list rows: walk pre-staged range + hist zero
        ext_min=None,  # list rows: cut line == exact candidate minimum
    ):
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        pair_cap = cutlass.const_expr(_pair_cap_for(self.kNumBins))
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        if cand_count == cutlass.Int32(kK):
            i4 = tidx
            while i4 < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i4] = self.dtype(smem_keys[i4])
                output_indices_row[i4] = smem_vals[i4]
                i4 = i4 + cutlass.Int32(num_threads)
        elif cand_count > cutlass.Int32(kK):
            if cutlass.const_expr(_P4_SUB_DBG):
                sc1 = cutlass.Int64(0)
                sc2 = cutlass.Int64(0)
                sc3 = cutlass.Int64(0)
                sc4 = cutlass.Int64(0)
                sc5 = cutlass.Int64(0)
                sc6 = cutlass.Int64(0)
                sc0 = cute.arch.clock64()
            bmin_r = cutlass.Float32(self.FLT_MAX)
            bmax_r = cutlass.Float32(self.NEG_FLT_MAX)
            # a Python bool here, so with the feature off the stock body
            # below traces straight-line instead of inside an scf.if whose
            # predicate is a compile-time constant
            run_stock_range = True
            if cutlass.const_expr(ext_range_flag is not None):
                use_ext_r = ext_range_flag
                run_stock_range = use_ext_r == cutlass.Int32(0)
                if use_ext_r == cutlass.Int32(1):
                    # list rows: the take walk pre-zeroed the hist and staged
                    # per-warp maxima in smem_wcnt (its end barrier orders
                    # them); min := cut line by construction.
                    if cutlass.const_expr(ext_min is not None):
                        bmin_r = ext_min
                    for w in cutlass.range_constexpr(self.num_warps):
                        vmax = cutlass.Float32(
                            llvm.bitcast(cutlass.Float32.mlir_type, smem_wcnt[w].ir_value())
                        )
                        bmax_r = cute.arch.fmax(bmax_r, vmax)
                    if bmax_r <= bmin_r:
                        bmax_r = bmin_r + cutlass.Float32(1e-6)
            if run_stock_range:
                # ---- block min/max over candidates ----
                # The accepted threshold is an EXACT lower bound on every
                # candidate, so it can stand in for the min. Only the
                # assist tiers have a published threshold, hence the gate.
                use_thr_min = cutlass.const_expr(self.p4_no_fine)
                local_cmin = cutlass.Float32(self.FLT_MAX)
                local_cmax = cutlass.Float32(self.NEG_FLT_MAX)
                i5 = tidx
                while i5 < cand_count:
                    v = smem_keys[i5]
                    if cutlass.const_expr(not use_thr_min):
                        local_cmin = _fmin_f32_inline(local_cmin, v)
                    local_cmax = cute.arch.fmax(local_cmax, v)
                    i5 = i5 + cutlass.Int32(num_threads)
                cmin = cutlass.Float32(0.0)
                if cutlass.const_expr(not use_thr_min):
                    cmin = self.warp_reduce_min_f32(local_cmin)
                cmax = self.warp_reduce_max_f32(local_cmax)
                if lane == cutlass.Int32(0):
                    if cutlass.const_expr(not use_thr_min):
                        smem_wcnt[warp_id] = float_as_uint32(cmin)
                    smem_hist[warp_id] = float_as_uint32(cmax)
                cute.arch.barrier()
                # lane-parallel cross-warp fold: lane w holds slot w and one
                # warp reduce settles it. min/max reassociate freely, so the
                # result is bit-identical on every warp without a leader.
                pmn = cutlass.Float32(self.FLT_MAX)
                pmx = cutlass.Float32(self.NEG_FLT_MAX)
                if lane < cutlass.Int32(self.num_warps):
                    if cutlass.const_expr(not use_thr_min):
                        pmn = cutlass.Float32(
                            llvm.bitcast(cutlass.Float32.mlir_type, smem_wcnt[lane].ir_value())
                        )
                    pmx = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, smem_hist[lane].ir_value())
                    )
                if cutlass.const_expr(use_thr_min):
                    bmin_r = s_thr[0]
                else:
                    bmin_r = _fmin_f32_inline(bmin_r, self.warp_reduce_min_f32(pmn))
                bmax_r = cute.arch.fmax(bmax_r, self.warp_reduce_max_f32(pmx))
                if bmax_r <= bmin_r:
                    bmax_r = bmin_r + cutlass.Float32(1e-6)
                cute.arch.barrier()
                if cutlass.const_expr(_P4_SUB_DBG):
                    sc1 = cute.arch.clock64()
                # ---- zero + build histogram ----
                i6 = tidx
                while i6 < cutlass.Int32(kBins):
                    smem_hist[i6] = cutlass.Int32(0)
                    i6 = i6 + cutlass.Int32(num_threads)
                cute.arch.barrier()
            range1 = bmax_r - bmin_r
            inv1 = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range1
            i7 = tidx
            while i7 < cand_count:
                vk = smem_keys[i7]
                bin_i = cutlass.Int32((vk - bmin_r) * inv1)
                if bin_i < cutlass.Int32(0):
                    bin_i = cutlass.Int32(0)
                if bin_i > cutlass.Int32(kBins - 1):
                    bin_i = cutlass.Int32(kBins - 1)
                atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
                i7 = i7 + cutlass.Int32(num_threads)
            cute.arch.barrier()
            if cutlass.const_expr(_P4_SUB_DBG):
                sc2 = cute.arch.clock64()
            # ---- high→low bin search → straddling bin b* + rank_above ----
            if cutlass.const_expr(self.p4_warp_redundant):
                b_star, rank_above = self._p4_coarse_rw(smem_hist, smem_wcnt, warp_id, lane)
                if tidx == cutlass.Int32(0):
                    s_iscalars[4] = cutlass.Int32(0)  # cnt_above
                    s_iscalars[1] = cutlass.Int32(0)  # cnt_straddle
                cute.arch.barrier()
                if cutlass.const_expr(_P4_SUB_DBG):
                    sc3 = cute.arch.clock64()
            else:
                warp_bin_sum = cutlass.Int32(0)
                for jb in cutlass.range_constexpr(bins_per_warp):
                    bidx_s = (
                        cutlass.Int32(kBins - 1)
                        - warp_id * cutlass.Int32(bins_per_warp)
                        - cutlass.Int32(jb)
                    )
                    warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
                if lane == cutlass.Int32(0):
                    smem_wcnt[warp_id] = warp_bin_sum
                cute.arch.barrier()
                if tidx == cutlass.Int32(0):
                    cum = cutlass.Int32(0)
                    tw = cutlass.Int32(num_warps - 1)
                    found = cutlass.Int32(0)
                    for w2 in cutlass.range_constexpr(self.num_warps):
                        cum = cum + smem_wcnt[w2]
                        if cum >= cutlass.Int32(kK) and found == cutlass.Int32(0):
                            tw = cutlass.Int32(w2)
                            found = cutlass.Int32(1)
                    cum2 = cutlass.Int32(0)
                    for w3 in cutlass.range_constexpr(self.num_warps):
                        if cutlass.Int32(w3) < tw:
                            cum2 = cum2 + smem_wcnt[w3]
                    s_iscalars[2] = cum2  # prefix-count before target warp
                    s_iscalars[3] = tw
                cute.arch.barrier()
                target_warp = s_iscalars[3]
                if warp_id == target_warp and lane == cutlass.Int32(0):
                    base_cum = s_iscalars[2]
                    b_star_s = cutlass.Int32(kBins - 1)
                    rank_above_s = base_cum
                    set_d = cutlass.Int32(0)
                    for jb2 in cutlass.range_constexpr(bins_per_warp):
                        bidx2 = (
                            cutlass.Int32(kBins - 1)
                            - target_warp * cutlass.Int32(bins_per_warp)
                            - cutlass.Int32(jb2)
                        )
                        ra_before = base_cum
                        base_cum = base_cum + smem_hist[bidx2]
                        if base_cum >= cutlass.Int32(kK) and set_d == cutlass.Int32(0):
                            b_star_s = bidx2
                            rank_above_s = ra_before  # count in bins strictly above b*
                            set_d = cutlass.Int32(1)
                    s_iscalars[2] = rank_above_s
                    s_iscalars[3] = b_star_s
                    s_iscalars[4] = cutlass.Int32(0)  # cnt_above
                    s_iscalars[1] = cutlass.Int32(0)  # cnt_straddle
                cute.arch.barrier()
                if cutlass.const_expr(_P4_SUB_DBG):
                    sc3 = cute.arch.clock64()
                b_star = s_iscalars[3]
                rank_above = s_iscalars[2]

            # ---- EXACT: one fine-histogram recursion on the straddling bin b* ----
            if cutlass.const_expr(self.enable_p4_rank_scatter_exact):
                # FIXED small fine-bin count (independent of kNumBins): 256
                # sub-bins over bin b* gives kNumBins×256 effective resolution,
                # enough to resolve the straddling bin to ≤1 distinct value.
                fbins = cutlass.const_expr(256)
                # bin b* value range under the inv1 binning: [f_lo, f_lo + 1/inv1)
                f_lo = cutlass.Float32(0.0)
                finv = cutlass.Float32(0.0)
                if cutlass.const_expr(not self.p4_no_fine):
                    f_lo = bmin_r + cutlass.Float32(b_star) / inv1
                    finv = (cutlass.Float32(fbins - 1) + cutlass.Float32(0.99)) * inv1
                if cutlass.const_expr(self.p4_no_fine):
                    # Sub-binning collapsed: finv 0 sends every member of the
                    # straddling coarse bin to sub-bin 0 == sb*, so the
                    # scatter parks the whole coarse class and the tail ranks
                    # it. MUST stay a compile-time removal (the re-zero, build
                    # and search below untraced), not a runtime branch.
                    finv = cutlass.Float32(0.0)
                    sb_star = cutlass.Int32(0)
                    rank_above_fine = rank_above
                # bin b* spans [f_lo, f_hi); the clamped ends fold
                # out-of-range values into bin 0 and bin kBins-1, so those
                # two drop the matching side. Only the range-test arms read
                # these, so upstream's build must not compute them.
                if cutlass.const_expr(self.p4_fine_rangetest or self.p4_scat_rangetest):
                    f_hi = f_lo + cutlass.Float32(1.0) / inv1
                    lo_edge = b_star == cutlass.Int32(0)
                    hi_edge = b_star == cutlass.Int32(kBins - 1)
                if cutlass.const_expr(not self.p4_no_fine):
                    # re-zero (only fbins slots) + build fine sub-hist of bin-b* cands
                    iz = tidx
                    while iz < cutlass.Int32(fbins):
                        smem_hist[iz] = cutlass.Int32(0)
                        iz = iz + cutlass.Int32(num_threads)
                    cute.arch.barrier()
                    if cutlass.const_expr(self.p4_fine_rangetest):
                        # A candidate belongs to bin b* exactly when its value
                        # lies in [f_lo, f_hi). The clamped ends of the binning
                        # fold out-of-range values INTO bin 0 and bin kBins-1,
                        # so those two bins must drop the matching side of the
                        # range test to stay bit-identical.
                        ifb = tidx
                        while ifb < cand_count:
                            vf = smem_keys[ifb]
                            inb = vf >= f_lo and vf < f_hi
                            if lo_edge:
                                inb = vf < f_hi
                            if hi_edge:
                                inb = vf >= f_lo
                            if inb:
                                sb = cutlass.Int32((vf - f_lo) * finv)
                                if sb < cutlass.Int32(0):
                                    sb = cutlass.Int32(0)
                                if sb > cutlass.Int32(fbins - 1):
                                    sb = cutlass.Int32(fbins - 1)
                                atomicAdd(smem_hist.iterator + sb, cutlass.Int32(1))
                            ifb = ifb + cutlass.Int32(num_threads)
                    else:
                        ifb = tidx
                        while ifb < cand_count:
                            vfo = smem_keys[ifb]
                            cbo = cutlass.Int32((vfo - bmin_r) * inv1)
                            if cbo < cutlass.Int32(0):
                                cbo = cutlass.Int32(0)
                            if cbo > cutlass.Int32(kBins - 1):
                                cbo = cutlass.Int32(kBins - 1)
                            if cbo == b_star:
                                sbo = cutlass.Int32((vfo - f_lo) * finv)
                                if sbo < cutlass.Int32(0):
                                    sbo = cutlass.Int32(0)
                                if sbo > cutlass.Int32(fbins - 1):
                                    sbo = cutlass.Int32(fbins - 1)
                                atomicAdd(smem_hist.iterator + sbo, cutlass.Int32(1))
                            ifb = ifb + cutlass.Int32(num_threads)
                    cute.arch.barrier()
                    # fine sub-bin search, resolved lane-parallel on every
                    # warp (see _p4_fine_rw); the answer comes back in
                    # registers.
                    sb_star, rank_above_fine = self._p4_fine_rw(
                        smem_hist, smem_wcnt, fbins, rank_above, warp_id, lane
                    )
                if tidx == cutlass.Int32(0):
                    s_iscalars[4] = cutlass.Int32(0)  # cnt_above
                    s_iscalars[0] = cutlass.Int32(0)  # cnt_mid (b*, sub>sb*)
                    s_iscalars[1] = cutlass.Int32(0)  # cnt_strad (b*, sub==sb*)
                cute.arch.barrier()
                if cutlass.const_expr(_P4_SUB_DBG):
                    sc4 = cute.arch.clock64()
                isc = tidx
                while isc < cand_count:
                    v = smem_keys[isc]
                    if cutlass.const_expr(self.p4_scat_rangetest):
                        # same three-way split as the bin recompute, by value:
                        # above b* <=> v >= f_hi (impossible at the top bin,
                        # which absorbs everything higher), inside b* <=> v in
                        # [f_lo, f_hi) with the edge bins dropping their
                        # absorbed side. bin_i is only ever compared against
                        # b*, so encoding the class as b*-1 / b* / b*+1 keeps
                        # the branches below unchanged.
                        abv = v >= f_hi
                        inb2 = v >= f_lo and v < f_hi
                        if lo_edge:
                            inb2 = v < f_hi
                        if hi_edge:
                            abv = False
                            inb2 = v >= f_lo
                        bin_i = b_star - cutlass.Int32(1)
                        if abv:
                            bin_i = b_star + cutlass.Int32(1)
                        if inb2:
                            bin_i = b_star
                    else:
                        bin_i = cutlass.Int32((v - bmin_r) * inv1)
                        if bin_i < cutlass.Int32(0):
                            bin_i = cutlass.Int32(0)
                        if bin_i > cutlass.Int32(kBins - 1):
                            bin_i = cutlass.Int32(kBins - 1)
                    if bin_i > b_star:
                        pos = atomicAdd(s_iscalars.iterator + cutlass.Int32(4), cutlass.Int32(1))
                        if pos < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[pos] = self.dtype(v)
                            output_indices_row[pos] = smem_vals[isc]
                    elif bin_i == b_star:
                        # With the fine level compiled out the sub-bin is a
                        # constant 0 == sb*, so the whole three-way split
                        # below collapses to the park arm and the per-
                        # candidate recompute (a subtract, a multiply and two
                        # clamps) is dead. Keep it a compile-time constant so
                        # the scatter does not carry it.
                        sb = cutlass.Int32(0)
                        if cutlass.const_expr(not self.p4_no_fine):
                            sb = cutlass.Int32((v - f_lo) * finv)
                            if sb < cutlass.Int32(0):
                                sb = cutlass.Int32(0)
                            if sb > cutlass.Int32(fbins - 1):
                                sb = cutlass.Int32(fbins - 1)
                        if sb > sb_star:
                            o = atomicAdd(s_iscalars.iterator + cutlass.Int32(0), cutlass.Int32(1))
                            pos = rank_above + o
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                        elif sb == sb_star:
                            o = atomicAdd(s_iscalars.iterator + cutlass.Int32(1), cutlass.Int32(1))
                            if cutlass.const_expr(
                                self.p4_exact_tail and self.p4_tail_fast and self.p4_tail_v3
                            ):
                                # Park (value bits, index) so the exact-tail
                                # repair never re-walks the candidates. Pairs
                                # live above the digit bins the radix route
                                # zeroes; small classes are the only consumer.
                                # Unconditional, index wrapped: the buffer is
                                # only ever READ when the class fits it, and
                                # then the wrap is a no-op.
                                ow = (o & cutlass.Int32(pair_cap - 1)) * cutlass.Int32(2)
                                smem_hist[cutlass.Int32(_PAIR_BASE) + ow] = float_as_int32(v)
                                smem_hist[cutlass.Int32(_PAIR_BASE) + ow + cutlass.Int32(1)] = (
                                    smem_vals[isc]
                                )
                            pos = rank_above_fine + o
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                    isc = isc + cutlass.Int32(num_threads)
                cute.arch.barrier()
                if cutlass.const_expr(_P4_SUB_DBG):
                    sc5 = cute.arch.clock64()
                cnt_strad = s_iscalars[1]
                filled = rank_above_fine + cnt_strad
                if filled > cutlass.Int32(kK):
                    filled = cutlass.Int32(kK)
                ipad = filled + tidx
                while ipad < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[ipad] = cutlass.Int32(-1)
                    ipad = ipad + cutlass.Int32(num_threads)

                # ---- EXACT-TAIL repair (p4_exact_tail, fp32): the fine bin
                # resolves values to range/(kBins*fbins); two candidates
                # closer than that can straddle the kK boundary inside ONE
                # fine bin, and the arrival-order fill above then keeps an
                # arbitrary subset. Gated on the ONLY case where that is
                # ambiguous — the tie set overfills the remaining slots — this
                # re-ranks the (b*, sb*) tie set exactly via an MSB-first
                # 8-bit-digit radix select over the order-preserving integer
                # keys (4 levels = bit-exact for fp32) and rewrites the tie
                # slot range [rank_above_fine, kK). Unambiguous rows (the
                # overwhelming majority) pay two scalar compares; the counters
                # and the fine histogram are reused, so SMEM does not grow.
                # boundary-class repair: pure-tie classes (one key value)
                # exit on a warp-reduce precheck; classes parked in the
                # pair buffer rank block-parallel over the parked pairs;
                # anything larger takes the full-candidate radix below.
                if cutlass.const_expr(self.p4_exact_tail and self.p4_tail_fast):  # [p4tt]
                    if cutlass.const_expr(self.p4_tail_v3):
                        need0 = cutlass.Int32(kK) - rank_above_fine
                        # per-thread compact buffers, bounded by the
                        # strided trip count over the candidate array
                        nbuf7 = cutlass.const_expr(
                            (self.kC + self.num_threads - 1) // self.num_threads
                        )
                        rv7 = cute.make_rmem_tensor((nbuf7,), cutlass.Float32)
                        ri7 = cute.make_rmem_tensor((nbuf7,), cutlass.Int32)
                        if cutlass.const_expr(_P4_TAIL_DBG or _P4_SUB_DBG):
                            if tidx == cutlass.Int32(0):
                                s_thr[1] = cutlass.Float32(cnt_strad)
                                s_thr[2] = cutlass.Float32(need0)
                        fast_done = cutlass.Int32(1)
                        if cnt_strad > need0 and need0 > cutlass.Int32(0):
                            fast_done = cutlass.Int32(0)
                            if cnt_strad <= cutlass.Int32(pair_cap):
                                # Small mixed class: the scatter already parked
                                # every member as a (value bits, index) pair,
                                # so there is nothing to collect - go straight
                                # to the rank. Rank with the WHOLE BLOCK: each
                                # warp owns a stride of the class and its 32
                                # lanes split the comparisons.
                                e9 = warp_id
                                while e9 < cnt_strad:
                                    be9 = smem_hist[_PAIR_BASE + e9 + e9]
                                    ke9 = f32_order_key(
                                        cutlass.Float32(
                                            llvm.bitcast(cutlass.Float32.mlir_type, be9.ir_value())
                                        )
                                    ) ^ cutlass.Int32(-2147483648)
                                    c9 = cutlass.Int32(0)
                                    j9 = lane
                                    while j9 < cnt_strad:
                                        bj9 = smem_hist[_PAIR_BASE + j9 + j9]
                                        kj9 = f32_order_key(
                                            cutlass.Float32(
                                                llvm.bitcast(
                                                    cutlass.Float32.mlir_type, bj9.ir_value()
                                                )
                                            )
                                        ) ^ cutlass.Int32(-2147483648)
                                        if kj9 > ke9:
                                            c9 = c9 + cutlass.Int32(1)
                                        elif kj9 == ke9 and j9 < e9:
                                            c9 = c9 + cutlass.Int32(1)
                                        j9 = j9 + cutlass.Int32(32)
                                    r9 = self.warp_reduce_sum_i32(c9)
                                    if lane == cutlass.Int32(0) and r9 < need0:
                                        pos9 = rank_above_fine + r9
                                        if pos9 < cutlass.Int32(kK):
                                            if cutlass.const_expr(self.return_output_values):
                                                output_values_row[pos9] = self.dtype(
                                                    cutlass.Float32(
                                                        llvm.bitcast(
                                                            cutlass.Float32.mlir_type,
                                                            be9.ir_value(),
                                                        )
                                                    )
                                                )
                                            output_indices_row[pos9] = smem_hist[
                                                _PAIR_BASE + e9 + e9 + cutlass.Int32(1)
                                            ]
                                    e9 = e9 + cutlass.Int32(num_warps)
                                cute.arch.barrier()
                            else:
                                # Large class only: block-wide pure-tie check
                                # (min/max order key over the (b*, sb*) class)
                                # decides whether the radix can be skipped.
                                # A pure-tie class needs NO repair — the scatter's
                                # arrival fill of bit-equal values is already
                                # value-set exact.
                                # Staging mirrors the head min/max (wcnt + hist
                                # slots [0..31], both dead here; pairs live at
                                # 260+).
                                if tidx == cutlass.Int32(0):
                                    s_iscalars[0] = cutlass.Int32(0)
                                nh7 = cutlass.Int32(0)
                                kmn6 = cutlass.Int32(2147483647)
                                kmx6 = cutlass.Int32(-2147483648)
                                it6 = tidx
                                # The pure-tie pre-check SKIPs the repair when
                                # the class is bit-uniform; only the large-class
                                # route runs it (the small route is already
                                # value-exact on a pure tie).
                                while it6 < cand_count:
                                    v6 = smem_keys[it6]
                                    b6 = cutlass.Int32((v6 - bmin_r) * inv1)
                                    if b6 < cutlass.Int32(0):
                                        b6 = cutlass.Int32(0)
                                    if b6 > cutlass.Int32(kBins - 1):
                                        b6 = cutlass.Int32(kBins - 1)
                                    if b6 == b_star:
                                        # with the fine level compiled out the
                                        # whole coarse bin IS the class
                                        s6 = cutlass.Int32(0)
                                        if cutlass.const_expr(not self.p4_no_fine):
                                            s6 = cutlass.Int32((v6 - f_lo) * finv)
                                            if s6 < cutlass.Int32(0):
                                                s6 = cutlass.Int32(0)
                                            if s6 > cutlass.Int32(fbins - 1):
                                                s6 = cutlass.Int32(fbins - 1)
                                        if s6 == sb_star:
                                            k6 = f32_order_key(v6) ^ cutlass.Int32(-2147483648)
                                            if k6 < kmn6:
                                                kmn6 = k6
                                            if k6 > kmx6:
                                                kmx6 = k6
                                            # Buffer the member so the compaction
                                            # below needs no walk of its own; the
                                            # array is nbuf7 = kC/num_threads
                                            # deep (unrolled dynamic-index store).
                                            for sl7 in cutlass.range_constexpr(nbuf7):
                                                if cutlass.Int32(sl7) == nh7:
                                                    rv7[sl7] = v6
                                                    ri7[sl7] = smem_vals[it6]
                                            nh7 = nh7 + cutlass.Int32(1)
                                    it6 = it6 + cutlass.Int32(num_threads)
                                kmn6 = cute.arch.warp_redux_sync(kmn6, "min")
                                kmx6 = cute.arch.warp_redux_sync(kmx6, "max")
                                if lane == cutlass.Int32(0):
                                    smem_wcnt[warp_id] = kmn6
                                    smem_hist[warp_id] = kmx6
                                cute.arch.barrier()
                                # lane-parallel cross-warp fold: lane w holds
                                # slot w and one warp reduce settles it, instead
                                # of every thread walking all num_warps slots of
                                # two arrays with dependent SMEM reads. Same
                                # inputs in the same order on every warp, so the
                                # result stays bit-identical and leaderless.
                                pa8 = cutlass.Int32(2147483647)
                                pb8 = cutlass.Int32(-2147483648)
                                if lane < cutlass.Int32(self.num_warps):
                                    pa8 = smem_wcnt[lane]
                                    pb8 = smem_hist[lane]
                                kmn7 = cute.arch.warp_redux_sync(pa8, "min")
                                kmx7 = cute.arch.warp_redux_sync(pb8, "max")
                                if kmn7 == kmx7:
                                    fast_done = cutlass.Int32(1)
                                if fast_done == cutlass.Int32(0):
                                    # mixed class: compact the members buffered by
                                    # the pure-tie pass above into
                                    # smem_keys/vals[0..cnt_strad). The staging
                                    # barrier above orders the buffered reads
                                    # before these writes.
                                    # warp-aggregated claim: intra-warp exclusive
                                    # prefix via shfl scan + ONE atomic per warp
                                    # (same-address claims would serialize)
                                    pf7 = nh7
                                    for so3 in cutlass.range_constexpr(5):
                                        oth3 = cute.arch.shuffle_sync_up(
                                            pf7, cutlass.Int32(1 << so3), mask_and_clamp=0
                                        )
                                        if lane >= cutlass.Int32(1 << so3):
                                            pf7 = pf7 + oth3
                                    tot7 = cute.arch.shuffle_sync(pf7, cutlass.Int32(31))
                                    wb7 = cutlass.Int32(0)
                                    if lane == cutlass.Int32(31):
                                        if tot7 > cutlass.Int32(0):
                                            wb7 = atomicAdd(
                                                s_iscalars.iterator + cutlass.Int32(0), tot7
                                            )
                                    wb7 = cute.arch.shuffle_sync(wb7, cutlass.Int32(31))
                                    bs7 = wb7 + pf7 - nh7
                                    for sl8 in cutlass.range_constexpr(nbuf7):
                                        if cutlass.Int32(sl8) < nh7:
                                            smem_keys[bs7 + cutlass.Int32(sl8)] = rv7[sl8]
                                            smem_vals[bs7 + cutlass.Int32(sl8)] = ri7[sl8]
                                    cute.arch.barrier()
                                    # Large class only (the small one never
                                    # reaches here): block-parallel 4-level
                                    # MSB radix over the compacted class (scans
                                    # touch class pairs only; warp0 shuffle-scan
                                    # digit search).
                                    if tidx == cutlass.Int32(0):
                                        smem_hist[256] = cutlass.Int32(0)
                                        smem_hist[257] = need0
                                        smem_hist[258] = cutlass.Int32(0)
                                    cute.arch.barrier()
                                    for lvl2 in cutlass.range_constexpr(4):
                                        shift2 = cutlass.const_expr(24 - 8 * lvl2)
                                        iz3 = tidx
                                        while iz3 < cutlass.Int32(256):
                                            smem_hist[iz3] = cutlass.Int32(0)
                                            iz3 = iz3 + cutlass.Int32(num_threads)
                                        cute.arch.barrier()
                                        uthr_c2 = smem_hist[256]
                                        ic2 = tidx
                                        while ic2 < cnt_strad:
                                            uk3 = f32_order_key(smem_keys[ic2])
                                            pm2 = cutlass.Int32(1)
                                            if cutlass.const_expr(lvl2 > 0):
                                                if (uk3 >> cutlass.Int32(shift2 + 8)) != (
                                                    uthr_c2 >> cutlass.Int32(shift2 + 8)
                                                ):
                                                    pm2 = cutlass.Int32(0)
                                            if pm2 == cutlass.Int32(1):
                                                dg2 = (
                                                    uk3 >> cutlass.Int32(shift2)
                                                ) & cutlass.Int32(0xFF)
                                                atomicAdd(
                                                    smem_hist.iterator + dg2, cutlass.Int32(1)
                                                )
                                            ic2 = ic2 + cutlass.Int32(num_threads)
                                        cute.arch.barrier()
                                        if warp_id == cutlass.Int32(0):
                                            ws3 = cutlass.Int32(0)
                                            for jd3 in cutlass.range_constexpr(8):
                                                di3 = (
                                                    cutlass.Int32(255)
                                                    - lane * cutlass.Int32(8)
                                                    - cutlass.Int32(jd3)
                                                )
                                                ws3 = ws3 + smem_hist[di3]
                                            pre6 = ws3
                                            for so2 in cutlass.range_constexpr(5):
                                                oth2 = cute.arch.shuffle_sync_up(
                                                    pre6,
                                                    cutlass.Int32(1 << so2),
                                                    mask_and_clamp=0,
                                                )
                                                if lane >= cutlass.Int32(1 << so2):
                                                    pre6 = pre6 + oth2
                                            needl3 = smem_hist[257]
                                            if pre6 >= needl3 and (pre6 - ws3) < needl3:
                                                base5 = pre6 - ws3
                                                dstar2 = cutlass.Int32(0)
                                                above5 = base5
                                                sd5 = cutlass.Int32(0)
                                                for jd4 in cutlass.range_constexpr(8):
                                                    di4 = (
                                                        cutlass.Int32(255)
                                                        - lane * cutlass.Int32(8)
                                                        - cutlass.Int32(jd4)
                                                    )
                                                    ra5 = base5
                                                    base5 = base5 + smem_hist[di4]
                                                    if base5 >= needl3 and sd5 == cutlass.Int32(0):
                                                        dstar2 = di4
                                                        above5 = ra5
                                                        sd5 = cutlass.Int32(1)
                                                smem_hist[256] = uthr_c2 | (
                                                    dstar2 << cutlass.Int32(shift2)
                                                )
                                                smem_hist[257] = needl3 - above5
                                                smem_hist[258] = smem_hist[258] + above5
                                        cute.arch.barrier()
                                    u_thr2 = smem_hist[256]
                                    cnt_ab2 = smem_hist[258]
                                    need_eq2 = smem_hist[257]
                                    kthr2 = u_thr2 ^ cutlass.Int32(-2147483648)
                                    if tidx == cutlass.Int32(0):
                                        s_iscalars[4] = cutlass.Int32(0)
                                        s_iscalars[0] = cutlass.Int32(0)
                                    cute.arch.barrier()
                                    ir3 = tidx
                                    while ir3 < cnt_strad:
                                        vv3 = smem_keys[ir3]
                                        uk4 = f32_order_key(vv3)
                                        ks4 = uk4 ^ cutlass.Int32(-2147483648)
                                        if ks4 > kthr2:
                                            o4 = atomicAdd(
                                                s_iscalars.iterator + cutlass.Int32(4),
                                                cutlass.Int32(1),
                                            )
                                            pos = rank_above_fine + o4
                                            if pos < cutlass.Int32(kK):
                                                if cutlass.const_expr(self.return_output_values):
                                                    output_values_row[pos] = self.dtype(vv3)
                                                output_indices_row[pos] = smem_vals[ir3]
                                        elif ks4 == kthr2:
                                            q4 = atomicAdd(
                                                s_iscalars.iterator + cutlass.Int32(0),
                                                cutlass.Int32(1),
                                            )
                                            if q4 < need_eq2:
                                                pos = rank_above_fine + cnt_ab2 + q4
                                                if pos < cutlass.Int32(kK):
                                                    if cutlass.const_expr(
                                                        self.return_output_values
                                                    ):
                                                        output_values_row[pos] = self.dtype(vv3)
                                                    output_indices_row[pos] = smem_vals[ir3]
                                        ir3 = ir3 + cutlass.Int32(num_threads)
                                    cute.arch.barrier()
                    else:
                        need0_s = cutlass.Int32(kK) - rank_above_fine
                        if cnt_strad > need0_s and need0_s > cutlass.Int32(0):
                            if cnt_strad <= cutlass.Int32(128):
                                # SMEM: (value_bits, cand_idx) pairs at
                                # smem_hist[2*o]/[2*o+1], o < 128 (slots 0..255).
                                # The 256 digit bins are dead here (the fast path
                                # replaces the radix levels that used them); the
                                # sb_star/ra staging in slots 2/3 was read by
                                # every thread before the pre-scatter barrier.
                                # Persistent radix scalars [256..258] untouched.
                                # Collect counter = s_iscalars[0] (dead after the
                                # scatter; same reuse as the radix rewrite pass).
                                if tidx == cutlass.Int32(0):
                                    s_iscalars[0] = cutlass.Int32(0)
                                cute.arch.barrier()
                                itc = tidx
                                while itc < cand_count:
                                    tv = smem_keys[itc]
                                    tb = cutlass.Int32((tv - bmin_r) * inv1)
                                    if tb < cutlass.Int32(0):
                                        tb = cutlass.Int32(0)
                                    if tb > cutlass.Int32(kBins - 1):
                                        tb = cutlass.Int32(kBins - 1)
                                    if tb == b_star:
                                        ts = cutlass.Int32((tv - f_lo) * finv)
                                        if ts < cutlass.Int32(0):
                                            ts = cutlass.Int32(0)
                                        if ts > cutlass.Int32(fbins - 1):
                                            ts = cutlass.Int32(fbins - 1)
                                        if ts == sb_star:
                                            to = atomicAdd(
                                                s_iscalars.iterator + cutlass.Int32(0),
                                                cutlass.Int32(1),
                                            )
                                            if to < cutlass.Int32(128):
                                                smem_hist[to + to] = float_as_int32(tv)
                                                smem_hist[to + to + cutlass.Int32(1)] = smem_vals[
                                                    itc
                                                ]
                                    itc = itc + cutlass.Int32(num_threads)
                                cute.arch.barrier()
                                # thread0 exact top-need0_s select rewriting
                                # positions [rank_above_fine, kK). Consumed flag =
                                # the cand_idx slot set to -1 (indices are always
                                # >= 0), so a genuine -FLT_MAX value in the class
                                # remains selectable (no value sentinel). Ties
                                # (bit-equal values) pick arbitrarily: value-set
                                # exact.
                                if tidx == cutlass.Int32(0):
                                    tj = cutlass.Int32(0)
                                    while tj < need0_s:
                                        tbv = cutlass.Float32(self.NEG_FLT_MAX)
                                        tbi = cutlass.Int32(-1)
                                        ti = cutlass.Int32(0)
                                        while ti < cnt_strad:
                                            tvi = smem_hist[ti + ti + cutlass.Int32(1)]
                                            if tvi >= cutlass.Int32(0):
                                                tvb = smem_hist[ti + ti]
                                                tvv = cutlass.Float32(
                                                    llvm.bitcast(
                                                        cutlass.Float32.mlir_type,
                                                        tvb.ir_value(),
                                                    )
                                                )
                                                take = cutlass.Int32(0)
                                                if tbi < cutlass.Int32(0):
                                                    take = cutlass.Int32(1)
                                                elif tvv > tbv:
                                                    take = cutlass.Int32(1)
                                                if take == cutlass.Int32(1):
                                                    tbv = tvv
                                                    tbi = ti
                                            ti = ti + cutlass.Int32(1)
                                        pos_s = rank_above_fine + tj
                                        if cutlass.const_expr(self.return_output_values):
                                            output_values_row[pos_s] = self.dtype(tbv)
                                        output_indices_row[pos_s] = smem_hist[
                                            tbi + tbi + cutlass.Int32(1)
                                        ]
                                        smem_hist[tbi + tbi + cutlass.Int32(1)] = cutlass.Int32(-1)
                                        tj = tj + cutlass.Int32(1)
                                cute.arch.barrier()
                            else:
                                # Persistent scalars live above the 256 digit bins
                                # (kNumBins >= 512 always): [256] key prefix (chosen
                                # digits, remaining bits 0), [257] slots still to fill
                                # inside the current equal-prefix set, [258] ties
                                # strictly above the prefix (their slots precede it).
                                if tidx == cutlass.Int32(0):
                                    smem_hist[256] = cutlass.Int32(0)
                                    smem_hist[257] = need0_s
                                    smem_hist[258] = cutlass.Int32(0)
                                cute.arch.barrier()
                                for lvl in cutlass.range_constexpr(4):
                                    shift = cutlass.const_expr(24 - 8 * lvl)
                                    iz2 = tidx
                                    while iz2 < cutlass.Int32(256):
                                        smem_hist[iz2] = cutlass.Int32(0)
                                        iz2 = iz2 + cutlass.Int32(num_threads)
                                    cute.arch.barrier()
                                    uthr_cur = smem_hist[256]
                                    it2 = tidx
                                    while it2 < cand_count:
                                        vt = smem_keys[it2]
                                        bt = cutlass.Int32((vt - bmin_r) * inv1)
                                        if bt < cutlass.Int32(0):
                                            bt = cutlass.Int32(0)
                                        if bt > cutlass.Int32(kBins - 1):
                                            bt = cutlass.Int32(kBins - 1)
                                        if bt == b_star:
                                            st2 = cutlass.Int32((vt - f_lo) * finv)
                                            if st2 < cutlass.Int32(0):
                                                st2 = cutlass.Int32(0)
                                            if st2 > cutlass.Int32(fbins - 1):
                                                st2 = cutlass.Int32(fbins - 1)
                                            if st2 == sb_star:
                                                uk = f32_order_key(vt)
                                                pmatch = cutlass.Int32(1)
                                                if cutlass.const_expr(lvl > 0):
                                                    if (uk >> cutlass.Int32(shift + 8)) != (
                                                        uthr_cur >> cutlass.Int32(shift + 8)
                                                    ):
                                                        pmatch = cutlass.Int32(0)
                                                if pmatch == cutlass.Int32(1):
                                                    dg = (
                                                        uk >> cutlass.Int32(shift)
                                                    ) & cutlass.Int32(0xFF)
                                                    atomicAdd(
                                                        smem_hist.iterator + dg, cutlass.Int32(1)
                                                    )
                                        it2 = it2 + cutlass.Int32(num_threads)
                                    cute.arch.barrier()
                                    # Two-stage descending digit scan (mirrors the
                                    # fine 3-step search): per-warp partial sums,
                                    # thread0 picks the target warp, its lane0 walks
                                    # the warp's digit range.
                                    fdw = cutlass.const_expr(256 // self.num_warps)
                                    wsum2 = cutlass.Int32(0)
                                    for jd in cutlass.range_constexpr(fdw):
                                        dix = (
                                            cutlass.Int32(255)
                                            - warp_id * cutlass.Int32(fdw)
                                            - cutlass.Int32(jd)
                                        )
                                        wsum2 = wsum2 + smem_hist[dix]
                                    if lane == cutlass.Int32(0):
                                        smem_wcnt[warp_id] = wsum2
                                    cute.arch.barrier()
                                    if tidx == cutlass.Int32(0):
                                        needl = smem_hist[257]
                                        cw = cutlass.Int32(0)
                                        tw3 = cutlass.Int32(num_warps - 1)
                                        f3 = cutlass.Int32(0)
                                        for w4 in cutlass.range_constexpr(self.num_warps):
                                            cw = cw + smem_wcnt[w4]
                                            if cw >= needl and f3 == cutlass.Int32(0):
                                                tw3 = cutlass.Int32(w4)
                                                f3 = cutlass.Int32(1)
                                        pre3 = cutlass.Int32(0)
                                        for w5 in cutlass.range_constexpr(self.num_warps):
                                            if cutlass.Int32(w5) < tw3:
                                                pre3 = pre3 + smem_wcnt[w5]
                                        s_iscalars[4] = pre3  # prefix above target warp
                                        s_iscalars[0] = tw3  # target warp
                                    cute.arch.barrier()
                                    pre4 = s_iscalars[4]
                                    tw4 = s_iscalars[0]
                                    if warp_id == tw4 and lane == cutlass.Int32(0):
                                        needl2 = smem_hist[257]
                                        base4 = pre4
                                        dstar = cutlass.Int32(0)
                                        above_d = pre4
                                        sd4 = cutlass.Int32(0)
                                        for jd2 in cutlass.range_constexpr(fdw):
                                            dix2 = (
                                                cutlass.Int32(255)
                                                - tw4 * cutlass.Int32(fdw)
                                                - cutlass.Int32(jd2)
                                            )
                                            ra4 = base4
                                            base4 = base4 + smem_hist[dix2]
                                            if base4 >= needl2 and sd4 == cutlass.Int32(0):
                                                dstar = dix2
                                                above_d = ra4
                                                sd4 = cutlass.Int32(1)
                                        smem_hist[256] = uthr_cur | (dstar << cutlass.Int32(shift))
                                        smem_hist[257] = needl2 - above_d
                                        smem_hist[258] = smem_hist[258] + above_d
                                    cute.arch.barrier()
                                # Rewrite the tie slot range: ties with key > u_thr
                                # first (there are exactly cnt_ab of them), then the
                                # first need_eq bitwise-equal-to-u_thr ties in arrival
                                # order (value-exact by construction). Signed compare
                                # needs the top bit flipped (unsigned-monotonic key).
                                u_thr = smem_hist[256]
                                cnt_ab = smem_hist[258]
                                need_eq = smem_hist[257]
                                ks_thr = u_thr ^ cutlass.Int32(-2147483648)
                                if tidx == cutlass.Int32(0):
                                    s_iscalars[4] = cutlass.Int32(0)  # above-writer ctr
                                    s_iscalars[0] = cutlass.Int32(0)  # equal-writer ctr
                                cute.arch.barrier()
                                ir2 = tidx
                                while ir2 < cand_count:
                                    vr = smem_keys[ir2]
                                    br = cutlass.Int32((vr - bmin_r) * inv1)
                                    if br < cutlass.Int32(0):
                                        br = cutlass.Int32(0)
                                    if br > cutlass.Int32(kBins - 1):
                                        br = cutlass.Int32(kBins - 1)
                                    if br == b_star:
                                        sr = cutlass.Int32((vr - f_lo) * finv)
                                        if sr < cutlass.Int32(0):
                                            sr = cutlass.Int32(0)
                                        if sr > cutlass.Int32(fbins - 1):
                                            sr = cutlass.Int32(fbins - 1)
                                        if sr == sb_star:
                                            uk2 = f32_order_key(vr)
                                            ks2 = uk2 ^ cutlass.Int32(-2147483648)
                                            if ks2 > ks_thr:
                                                o2 = atomicAdd(
                                                    s_iscalars.iterator + cutlass.Int32(4),
                                                    cutlass.Int32(1),
                                                )
                                                pos_s = rank_above_fine + o2
                                                if pos_s < cutlass.Int32(kK):
                                                    if cutlass.const_expr(
                                                        self.return_output_values
                                                    ):
                                                        output_values_row[pos_s] = self.dtype(vr)
                                                    output_indices_row[pos_s] = smem_vals[ir2]
                                            elif ks2 == ks_thr:
                                                q2 = atomicAdd(
                                                    s_iscalars.iterator + cutlass.Int32(0),
                                                    cutlass.Int32(1),
                                                )
                                                if q2 < need_eq:
                                                    pos_s = rank_above_fine + cnt_ab + q2
                                                    if pos_s < cutlass.Int32(kK):
                                                        if cutlass.const_expr(
                                                            self.return_output_values
                                                        ):
                                                            output_values_row[pos_s] = self.dtype(
                                                                vr
                                                            )
                                                        output_indices_row[pos_s] = smem_vals[ir2]
                                    ir2 = ir2 + cutlass.Int32(num_threads)
                                cute.arch.barrier()
                elif cutlass.const_expr(self.p4_exact_tail):  # if->elif only
                    need0 = cutlass.Int32(kK) - rank_above_fine
                    if cnt_strad > need0 and need0 > cutlass.Int32(0):
                        # Persistent scalars live above the 256 digit bins
                        # (kNumBins >= 512 always): [256] key prefix (chosen
                        # digits, remaining bits 0), [257] slots still to fill
                        # inside the current equal-prefix set, [258] ties
                        # strictly above the prefix (their slots precede it).
                        if tidx == cutlass.Int32(0):
                            smem_hist[256] = cutlass.Int32(0)
                            smem_hist[257] = need0
                            smem_hist[258] = cutlass.Int32(0)
                        cute.arch.barrier()
                        for lvl in cutlass.range_constexpr(4):
                            shift = cutlass.const_expr(24 - 8 * lvl)
                            iz2 = tidx
                            while iz2 < cutlass.Int32(256):
                                smem_hist[iz2] = cutlass.Int32(0)
                                iz2 = iz2 + cutlass.Int32(num_threads)
                            cute.arch.barrier()
                            uthr_cur = smem_hist[256]
                            it2 = tidx
                            while it2 < cand_count:
                                vt = smem_keys[it2]
                                bt = cutlass.Int32((vt - bmin_r) * inv1)
                                if bt < cutlass.Int32(0):
                                    bt = cutlass.Int32(0)
                                if bt > cutlass.Int32(kBins - 1):
                                    bt = cutlass.Int32(kBins - 1)
                                if bt == b_star:
                                    st2 = cutlass.Int32((vt - f_lo) * finv)
                                    if st2 < cutlass.Int32(0):
                                        st2 = cutlass.Int32(0)
                                    if st2 > cutlass.Int32(fbins - 1):
                                        st2 = cutlass.Int32(fbins - 1)
                                    if st2 == sb_star:
                                        uk = f32_order_key(vt)
                                        pmatch = cutlass.Int32(1)
                                        if cutlass.const_expr(lvl > 0):
                                            if (uk >> cutlass.Int32(shift + 8)) != (
                                                uthr_cur >> cutlass.Int32(shift + 8)
                                            ):
                                                pmatch = cutlass.Int32(0)
                                        if pmatch == cutlass.Int32(1):
                                            dg = (uk >> cutlass.Int32(shift)) & cutlass.Int32(0xFF)
                                            atomicAdd(smem_hist.iterator + dg, cutlass.Int32(1))
                                it2 = it2 + cutlass.Int32(num_threads)
                            cute.arch.barrier()
                            # Two-stage descending digit scan (mirrors the
                            # fine 3-step search): per-warp partial sums,
                            # thread0 picks the target warp, its lane0 walks
                            # the warp's digit range — 2*num_warps serial
                            # steps instead of 256.
                            fdw = cutlass.const_expr(256 // self.num_warps)
                            wsum2 = cutlass.Int32(0)
                            for jd in cutlass.range_constexpr(fdw):
                                dix = (
                                    cutlass.Int32(255)
                                    - warp_id * cutlass.Int32(fdw)
                                    - cutlass.Int32(jd)
                                )
                                wsum2 = wsum2 + smem_hist[dix]
                            if lane == cutlass.Int32(0):
                                smem_wcnt[warp_id] = wsum2
                            cute.arch.barrier()
                            if tidx == cutlass.Int32(0):
                                needl = smem_hist[257]
                                cw = cutlass.Int32(0)
                                tw3 = cutlass.Int32(num_warps - 1)
                                f3 = cutlass.Int32(0)
                                for w4 in cutlass.range_constexpr(self.num_warps):
                                    cw = cw + smem_wcnt[w4]
                                    if cw >= needl and f3 == cutlass.Int32(0):
                                        tw3 = cutlass.Int32(w4)
                                        f3 = cutlass.Int32(1)
                                pre3 = cutlass.Int32(0)
                                for w5 in cutlass.range_constexpr(self.num_warps):
                                    if cutlass.Int32(w5) < tw3:
                                        pre3 = pre3 + smem_wcnt[w5]
                                s_iscalars[4] = pre3  # prefix above target warp
                                s_iscalars[0] = tw3  # target warp
                            cute.arch.barrier()
                            pre4 = s_iscalars[4]
                            tw4 = s_iscalars[0]
                            if warp_id == tw4 and lane == cutlass.Int32(0):
                                needl2 = smem_hist[257]
                                base4 = pre4
                                dstar = cutlass.Int32(0)
                                above_d = pre4
                                sd4 = cutlass.Int32(0)
                                for jd2 in cutlass.range_constexpr(fdw):
                                    dix2 = (
                                        cutlass.Int32(255)
                                        - tw4 * cutlass.Int32(fdw)
                                        - cutlass.Int32(jd2)
                                    )
                                    ra4 = base4
                                    base4 = base4 + smem_hist[dix2]
                                    if base4 >= needl2 and sd4 == cutlass.Int32(0):
                                        dstar = dix2
                                        above_d = ra4
                                        sd4 = cutlass.Int32(1)
                                smem_hist[256] = uthr_cur | (dstar << cutlass.Int32(shift))
                                smem_hist[257] = needl2 - above_d
                                smem_hist[258] = smem_hist[258] + above_d
                            cute.arch.barrier()
                        # Rewrite the tie slot range: ties with key > u_thr
                        # first (there are exactly cnt_ab of them), then the
                        # first need_eq bitwise-equal-to-u_thr ties in arrival
                        # order (value-exact by construction). Signed compare
                        # needs the top bit flipped (unsigned-monotonic key).
                        u_thr = smem_hist[256]
                        cnt_ab = smem_hist[258]
                        need_eq = smem_hist[257]
                        ks_thr = u_thr ^ cutlass.Int32(-2147483648)
                        if tidx == cutlass.Int32(0):
                            s_iscalars[4] = cutlass.Int32(0)  # above-writer ctr
                            s_iscalars[0] = cutlass.Int32(0)  # equal-writer ctr
                        cute.arch.barrier()
                        ir2 = tidx
                        while ir2 < cand_count:
                            vr = smem_keys[ir2]
                            br = cutlass.Int32((vr - bmin_r) * inv1)
                            if br < cutlass.Int32(0):
                                br = cutlass.Int32(0)
                            if br > cutlass.Int32(kBins - 1):
                                br = cutlass.Int32(kBins - 1)
                            if br == b_star:
                                sr = cutlass.Int32((vr - f_lo) * finv)
                                if sr < cutlass.Int32(0):
                                    sr = cutlass.Int32(0)
                                if sr > cutlass.Int32(fbins - 1):
                                    sr = cutlass.Int32(fbins - 1)
                                if sr == sb_star:
                                    uk2 = f32_order_key(vr)
                                    ks2 = uk2 ^ cutlass.Int32(-2147483648)
                                    if ks2 > ks_thr:
                                        o2 = atomicAdd(
                                            s_iscalars.iterator + cutlass.Int32(4),
                                            cutlass.Int32(1),
                                        )
                                        pos = rank_above_fine + o2
                                        if pos < cutlass.Int32(kK):
                                            if cutlass.const_expr(self.return_output_values):
                                                output_values_row[pos] = self.dtype(vr)
                                            output_indices_row[pos] = smem_vals[ir2]
                                    elif ks2 == ks_thr:
                                        q2 = atomicAdd(
                                            s_iscalars.iterator + cutlass.Int32(0),
                                            cutlass.Int32(1),
                                        )
                                        if q2 < need_eq:
                                            pos = rank_above_fine + cnt_ab + q2
                                            if pos < cutlass.Int32(kK):
                                                if cutlass.const_expr(self.return_output_values):
                                                    output_values_row[pos] = self.dtype(vr)
                                                output_indices_row[pos] = smem_vals[ir2]
                            ir2 = ir2 + cutlass.Int32(num_threads)
                        cute.arch.barrier()
            else:
                # ---- APPROX rank-and-scatter (single pass), arbitrary straddling order ----
                isc = tidx
                while isc < cand_count:
                    v = smem_keys[isc]
                    bin_i = cutlass.Int32((v - bmin_r) * inv1)
                    if bin_i < cutlass.Int32(0):
                        bin_i = cutlass.Int32(0)
                    if bin_i > cutlass.Int32(kBins - 1):
                        bin_i = cutlass.Int32(kBins - 1)
                    if bin_i > b_star:
                        pos = atomicAdd(s_iscalars.iterator + cutlass.Int32(4), cutlass.Int32(1))
                        if pos < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[pos] = self.dtype(v)
                            output_indices_row[pos] = smem_vals[isc]
                    elif bin_i == b_star:
                        off = atomicAdd(s_iscalars.iterator + cutlass.Int32(1), cutlass.Int32(1))
                        pos = rank_above + off
                        if pos < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[pos] = self.dtype(v)
                            output_indices_row[pos] = smem_vals[isc]
                    isc = isc + cutlass.Int32(num_threads)
                cute.arch.barrier()
                cnt_strad = s_iscalars[1]
                filled = rank_above + cnt_strad
                if filled > cutlass.Int32(kK):
                    filled = cutlass.Int32(kK)
                ipad = filled + tidx
                while ipad < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[ipad] = cutlass.Int32(-1)
                    ipad = ipad + cutlass.Int32(num_threads)
            if cutlass.const_expr(_P4_SUB_DBG):
                # smem_wcnt slots [8..13] are dead after the last warp-sum
                # use above; the take-block publish copies them to xstate.
                sc6 = cute.arch.clock64()
                if tidx == cutlass.Int32(0):
                    smem_wcnt[8] = cutlass.Int32(sc1 - sc0)  # minmax
                    smem_wcnt[9] = cutlass.Int32(sc2 - sc1)  # hist build
                    smem_wcnt[10] = cutlass.Int32(sc3 - sc2)  # coarse search
                    smem_wcnt[11] = cutlass.Int32(sc4 - sc3)  # fine recursion
                    smem_wcnt[12] = cutlass.Int32(sc5 - sc4)  # scatter
                    smem_wcnt[13] = cutlass.Int32(sc6 - sc5)  # tail repair+pad
        else:
            i10 = tidx
            while i10 < cand_count:
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i10] = self.dtype(smem_keys[i10])
                output_indices_row[i10] = smem_vals[i10]
                i10 = i10 + cutlass.Int32(num_threads)
            if s_iscalars[6] == cutlass.Int32(0):  # plateau fill completes done=3
                i11 = cand_count + tidx
                while i11 < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[i11] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[i11] = cutlass.Int32(-1)
                    i11 = i11 + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # Phase 4: Histogram-based k-th selection + two-pass writeback
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_histogram_snap(
        self,
        smem_keys,
        smem_vals,
        smem_hist,
        smem_wcnt,
        s_thr,
        s_iscalars,
        output_values_row,
        output_indices_row,
        cand_count,
        tidx,
        warp_id,
        lane,
    ):
        """Three branches by cand_count vs kK:
        == kK: direct emit (fast path)
        >  kK: histogram k-th bin search → snap → 2-pass writeback
        <  kK: emit cand_count + pad with -FLT_MAX
        """
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        # Hoisted SMEM window bases: every keys/vals element access below
        # goes through raw integer addressing (see _smem_ref rationale).
        keys_base = smem_keys.iterator.toint()
        vals_base = smem_vals.iterator.toint()
        # Scalars base for the snap-loop convergence check (read by ALL
        # threads once per snap iteration — a measured per-iteration
        # LDS hotspot).
        isc_base = s_iscalars.iterator.toint()

        # ----- Branch A: cand_count == kK (fast path) -----
        if cand_count == cutlass.Int32(kK):
            i4 = tidx
            while i4 < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i4] = self.dtype(
                        self._smem_ld(cutlass.Float32, keys_base, i4)
                    )
                output_indices_row[i4] = self._smem_ld(cutlass.Int32, vals_base, i4)
                i4 = i4 + cutlass.Int32(num_threads)
        elif cand_count > cutlass.Int32(kK):
            # ----- Branch B: cand_count > kK → histogram snap -----

            # ---- Histogram window ----
            # Fast path: reuse the P2 exit bracket [vlo, vhi) instead of
            # scanning candidates for min/max. P3 collected v >= s_thr[0]
            # and P2's exit sets s_thr[0] = vlo (= s_thr[1]), so vlo
            # lower-bounds every candidate; the bracket invariant
            # cnt(>= vhi) < kK puts the k-th value inside [vlo, vhi).
            # Out-of-window candidates (row max etc.) clamp into the edge
            # bins — cumulative counts from the top stay exact — and the
            # bracket is P2's acceptance band, far narrower than
            # [cand_min, cand_max], so level-1 bin resolution IMPROVES.
            # Any path that leaves the bracket stale (degenerate-bracket
            # fallback, probe variants) fails the guard and takes the
            # original min/max scan; a plausible-but-wrong bracket can
            # only cost extra snap/refinement steps, never exactness.
            # Uniform branch: SMEM scalars read after the P3-exit barrier.
            w_lo = s_thr[1]
            w_hi = s_thr[2]
            bmin_r = cutlass.Float32(0.0)
            bmax_r = cutlass.Float32(1e-6)
            if s_thr[0] == w_lo and w_hi > w_lo and w_hi < cutlass.Float32(self.FLT_MAX):
                bmin_r = w_lo
                bmax_r = w_hi
            else:
                # Block min/max over keys[0:cand_count]
                local_cmin = cutlass.Float32(self.FLT_MAX)
                local_cmax = cutlass.Float32(self.NEG_FLT_MAX)
                i5 = tidx
                while i5 < cand_count:
                    v = self._smem_ld(cutlass.Float32, keys_base, i5)
                    local_cmin = _fmin_f32_inline(local_cmin, v)
                    local_cmax = cute.arch.fmax(local_cmax, v)
                    i5 = i5 + cutlass.Int32(num_threads)
                cmin = self.warp_reduce_min_f32(local_cmin)
                cmax = self.warp_reduce_max_f32(local_cmax)
                # Stage warp results into smem_wcnt[w] (cmin) and smem_hist[w] (cmax)
                # as bit-cast int32. cmax stored at smem_hist[0..NW-1].
                if lane == 0:
                    smem_wcnt[warp_id] = float_as_uint32(cmin)
                    smem_hist[warp_id] = float_as_uint32(cmax)
                cute.arch.barrier()

                # Every thread independently recomputes block_min/block_max
                # from the warp-staged smem slots (CUDA heuristic_topk.cuh:891-898
                # pattern). No tid==0 → s_thr broadcast → saves a block barrier.
                bmin_r = cutlass.Float32(self.FLT_MAX)
                bmax_r = cutlass.Float32(self.NEG_FLT_MAX)
                # Unrolled num_warps times (16 or 32 — fixed at compile time).
                for w in cutlass.range_constexpr(self.num_warps):
                    vmin_bits = smem_wcnt[w]
                    vmax_bits = smem_hist[w]
                    vmin = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, vmin_bits.ir_value())
                    )
                    vmax = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, vmax_bits.ir_value())
                    )
                    bmin_r = _fmin_f32_inline(bmin_r, vmin)
                    bmax_r = cute.arch.fmax(bmax_r, vmax)
                if bmax_r <= bmin_r:
                    bmax_r = bmin_r + cutlass.Float32(1e-6)
                # Barrier required: smem_hist[0..NW-1] above doubles as cmax
                # scratch and below as the histogram. Without this sync the
                # zeroing pass below can clobber a cmax slot a later warp is
                # still reading → wrong bmax_r → all candidates squashed into
                # bin 0 (hit-rate-dependent race).
                cute.arch.barrier()

            range1 = bmax_r - bmin_r
            # Overflow hardening (pre-existing):
            # a candidate span > FLT_MAX (needs |v| ~ 1.7e38; fuzz-only for
            # real logits) overflows range1 to +inf → inv1 = +0 → every
            # candidate lands in bin 0 → thr = lo + 0*inf = NaN → all snap
            # comparisons false, the walk never moves, and the whole row
            # writes as padding. Clamp to FLT_MAX: the start threshold
            # stays ORDERED (±inf is fine — snap's monotone walk rescues
            # any ordered start; only NaN breaks it).
            if range1 > cutlass.Float32(self.FLT_MAX):
                range1 = cutlass.Float32(self.FLT_MAX)
            # inv1 = (kBins - 1 + 0.99) / range1  (range1 > 0 guaranteed by 1e-6 patch)
            inv1 = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range1
            binw1 = range1 / cutlass.Float32(kBins)

            # Predeclared register state for the redundant-warp path
            # (threshold / counts / staging parity live in registers; the
            # leader path below keeps them in s_thr/s_iscalars instead).
            thr_reg = bmin_r
            selc_reg = cutlass.Int32(0)
            thr_s = bmin_r
            cge_r = cutlass.Int32(0)
            cgt_r = cutlass.Int32(0)
            win_par = cutlass.Int32(0)

            # Level-1: histogram over [bmin, bmax] + k-th bin search.
            self._hist_build(keys_base, smem_hist, cand_count, bmin_r, inv1, tidx)
            if cutlass.const_expr(self.p4_warp_redundant):
                thr_reg, selc_reg = self._kth_bin_search_rw(
                    smem_hist, smem_wcnt, bmin_r, binw1, tidx, warp_id, lane
                )
            else:
                self._kth_bin_search(
                    smem_hist, smem_wcnt, s_thr, s_iscalars, bmin_r, binw1, tidx, warp_id, lane
                )

            # ---- Level-2 histogram refinement ----
            # The snap loop below steps ONE distinct value per iteration
            # (~0.45us each: full candidate re-scan + 2 barriers), and real
            # logits concentrate count mass right at the k-th boundary, so
            # the selected level-1 bin often holds tens of values → snap
            # stragglers of 10+ us set the wall clock at N<=32K. When the
            # selected bin is dense, re-histogram just that bin (bin width
            # shrinks kBins x) for ~1us of extra scan, leaving the snap
            # loop 0-2 steps. The snap loop converges monotonically from
            # any starting threshold, so this only moves the start point —
            # exactness is untouched (a level-2 edge-rounding error at
            # worst costs one extra snap step). Uniform branch: everyone
            # reads the same post-barrier SMEM scalar.
            # Level 2 fires when a snap walk would cost more than one
            # rebuild (~2 snap steps break even); level 3 only when level 2
            # failed to split the bin (>8: heavy ties or a sub-ulp-wide
            # window — both rare on real logits, where ties at the k-th
            # are ~1 and the acceptance band spans >>1 ulp).
            binw_cur = binw1
            for _lvl in cutlass.range_constexpr(2):
                if cutlass.const_expr(self.p4_warp_redundant):
                    sel_cnt_l = selc_reg
                else:
                    sel_cnt_l = s_iscalars[4]
                gate_l = cutlass.const_expr(2 if _lvl == 0 else 8)
                if sel_cnt_l > cutlass.Int32(gate_l):
                    if cutlass.const_expr(self.p4_warp_redundant):
                        thr_el = thr_reg
                        # _kth_bin_search_rw has no trailing barrier; the
                        # zero pass of the rebuild below must not clobber
                        # smem_hist under a warp still in its step 3.
                        cute.arch.barrier()
                    else:
                        thr_el = s_thr[0]
                    # 2% slop each side absorbs the inv-vs-binw rounding
                    # difference in the previous level's edge estimate.
                    lo_l = thr_el - cutlass.Float32(0.02) * binw_cur
                    range_l = cutlass.Float32(1.04) * binw_cur
                    inv_l = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range_l
                    binw_next = range_l / cutlass.Float32(kBins)
                    self._hist_build(keys_base, smem_hist, cand_count, lo_l, inv_l, tidx)
                    if cutlass.const_expr(self.p4_warp_redundant):
                        thr_l2, selc_l2 = self._kth_bin_search_rw(
                            smem_hist, smem_wcnt, lo_l, binw_next, tidx, warp_id, lane
                        )
                        thr_reg = thr_l2
                        selc_reg = selc_l2
                    else:
                        self._kth_bin_search(
                            smem_hist,
                            smem_wcnt,
                            s_thr,
                            s_iscalars,
                            lo_l,
                            binw_next,
                            tidx,
                            warp_id,
                            lane,
                        )
                    binw_cur = binw_next

            # ---- Snap convergence loop ----
            # Upper bound = cand_count (matches CUDA heuristic_topk.cuh:985).
            # Common path converges in 1-3 iters; the loose ceiling only
            # matters for adversarial cells where a tighter bound would
            # accept a non-converged threshold (~0.09% of distributions).
            snap_limit = cand_count

            # Runtime break via a guard flag — no `break` in cute.range.
            si = cutlass.Int32(0)
            done_snap = cutlass.Int32(0)
            if cutlass.const_expr(self.p4_warp_redundant):
                # Redundant-warp snap: threshold + convergence state live
                # in registers (every warp reduces the staged partials
                # itself, bit-identically), so each iteration needs ONE
                # barrier (staging visibility) instead of two. Staging is
                # parity double-buffered in smem_hist[par*3NW ..] so a
                # warp one iteration ahead writes the other bank while a
                # slow warp still reads the old one; the staging barrier
                # bounds the drift to a single iteration.
                cute.arch.barrier()  # rw-search step-3 readers vs staging
                nwc = cutlass.const_expr(self.num_warps)
                thr_s = thr_reg
                par4 = cutlass.Int32(0)
                while si < snap_limit and done_snap == cutlass.Int32(0):
                    lge4 = cutlass.Int32(0)
                    lgt4 = cutlass.Int32(0)
                    up4 = cutlass.Float32(self.FLT_MAX)
                    dn4 = cutlass.Float32(self.NEG_FLT_MAX)
                    isi4 = tidx
                    while isi4 < cand_count:
                        v4 = self._smem_ld(cutlass.Float32, keys_base, isi4)
                        if v4 >= thr_s:
                            lge4 = lge4 + cutlass.Int32(1)
                        if v4 > thr_s:
                            lgt4 = lgt4 + cutlass.Int32(1)
                            up4 = _fmin_f32_inline(up4, v4)
                        if v4 < thr_s:
                            dn4 = cute.arch.fmax(dn4, v4)
                        isi4 = isi4 + cutlass.Int32(num_threads)
                    packed4 = (lge4 << cutlass.Int32(16)) | lgt4
                    packed4 = self.warp_reduce_sum_i32(packed4)
                    up4 = self.warp_reduce_min_f32(up4)
                    dn4 = self.warp_reduce_max_f32(dn4)
                    off4 = par4 * cutlass.Int32(3 * nwc)
                    if lane == 0:
                        smem_hist[off4 + warp_id] = packed4
                        smem_hist[off4 + cutlass.Int32(nwc) + warp_id] = float_as_uint32(up4)
                        smem_hist[off4 + cutlass.Int32(2 * nwc) + warp_id] = float_as_uint32(dn4)
                    cute.arch.barrier()
                    v_tp = cutlass.Int32(0)
                    v_up = cutlass.Float32(self.FLT_MAX)
                    v_dn = cutlass.Float32(self.NEG_FLT_MAX)
                    if lane < cutlass.Int32(nwc):
                        v_tp = smem_hist[off4 + lane]
                        vu_b = smem_hist[off4 + cutlass.Int32(nwc) + lane]
                        vd_b = smem_hist[off4 + cutlass.Int32(2 * nwc) + lane]
                        v_up = cutlass.Float32(
                            llvm.bitcast(cutlass.Float32.mlir_type, vu_b.ir_value())
                        )
                        v_dn = cutlass.Float32(
                            llvm.bitcast(cutlass.Float32.mlir_type, vd_b.ir_value())
                        )
                    tp4 = self.warp_reduce_sum_i32(v_tp)
                    tup4 = self.warp_reduce_min_f32(v_up)
                    tdn4 = self.warp_reduce_max_f32(v_dn)
                    cge_r = tp4 >> cutlass.Int32(16)
                    cgt_r = tp4 & cutlass.Int32(0xFFFF)
                    win_par = par4
                    if cgt_r >= cutlass.Int32(kK):
                        if tup4 < cutlass.Float32(self.FLT_MAX):
                            thr_s = tup4
                    elif cge_r < cutlass.Int32(kK):
                        if tdn4 > cutlass.Float32(self.NEG_FLT_MAX):
                            thr_s = tdn4
                    if cgt_r < cutlass.Int32(kK) and cge_r >= cutlass.Int32(kK):
                        done_snap = cutlass.Int32(1)
                    par4 = par4 ^ cutlass.Int32(1)
                    si = si + cutlass.Int32(1)
            else:
                while si < snap_limit and done_snap == cutlass.Int32(0):
                    self.block_fused_snap_iter(
                        keys_base,
                        smem_wcnt,
                        smem_hist,
                        s_thr,
                        s_iscalars,
                        cand_count,
                        tidx,
                        warp_id,
                        lane,
                    )
                    # After block_fused_snap_iter, s_iscalars[2]=cge, s_iscalars[3]=cgt.
                    cgt_c = self._smem_ld(cutlass.Int32, isc_base, cutlass.Int32(3))
                    cge_c = self._smem_ld(cutlass.Int32, isc_base, cutlass.Int32(2))
                    if cgt_c < cutlass.Int32(kK) and cge_c >= cutlass.Int32(kK):
                        done_snap = cutlass.Int32(1)
                    si = si + cutlass.Int32(1)

            # ---- Writeback (ballot + popc) ----
            # Converged snap (the overwhelmingly common case): SINGLE pass.
            # The converged iteration's cgt (s_iscalars[3]) is the exact
            # strictly-greater count at sel_thr (block_fused_snap_iter does
            # not move the threshold when cgt < kK <= cge), so gt entries
            # can pack into [0, cgt) via counter s_iscalars[4] while
            # tie(==) entries start at offset cgt via counter s_iscalars[5]
            # — same [gt | eq | pad] output partition as the two-pass
            # original, one candidate sweep and one barrier fewer. The
            # non-converged fallback keeps the original two-pass (its cgt
            # would be stale: the last iter may have moved the threshold
            # after counting).
            if cutlass.const_expr(self.p4_warp_redundant):
                sel_thr = thr_s
            else:
                sel_thr = s_thr[0]
            if tidx == 0:
                s_iscalars[4] = cutlass.Int32(0)  # gt out_count
                # s_iscalars[5] (cluster-local scratch, consumed before P4)
                # is reused as the eq counter for the single-pass path.
                s_iscalars[5] = cutlass.Int32(0)
            cute.arch.barrier()

            if done_snap == cutlass.Int32(1):
                # Zero-atomic single pass. The converged snap iteration
                # staged each warp's packed(ge<<16|gt) counts AT sel_thr in
                # smem_wcnt[w] (nothing touches smem_wcnt between the snap
                # exit and here), and the snap scan's tidx-strided
                # partition covers exactly the same element set per warp
                # as this warp-chunk scan. So every warp derives its
                # deterministic output bases from a prefix over
                # smem_wcnt — the ~2*cand/32 serialized SMEM atomics of
                # the claim-based scheme (a top stall region in ncu at
                # N=8K) disappear. Output order within the [gt | eq]
                # segments changes (deterministic instead of claim order),
                # which the contract allows.
                if cutlass.const_expr(self.p4_warp_redundant):
                    cgt_base = cgt_r
                else:
                    cgt_base = s_iscalars[3]
                gt_run = cutlass.Int32(0)
                eq_run = cutlass.Int32(0)
                for wpre in cutlass.range_constexpr(self.num_warps):
                    if cutlass.const_expr(self.p4_warp_redundant):
                        # Converged iteration's packed counts live in the
                        # winning parity bank of smem_hist, not smem_wcnt.
                        pk_w = smem_hist[win_par * cutlass.Int32(3 * self.num_warps) + wpre]
                    else:
                        pk_w = smem_wcnt[wpre]
                    if cutlass.Int32(wpre) < warp_id:
                        wge_w = pk_w >> cutlass.Int32(16)
                        wgt_w = pk_w & cutlass.Int32(0xFFFF)
                        gt_run = gt_run + wgt_w
                        eq_run = eq_run + (wge_w - wgt_w)
                eq_run = cgt_base + eq_run
                base_w = warp_id * cutlass.Int32(self.WARP_SIZE)
                while base_w < cand_count:
                    ix1 = base_w + lane
                    emit_gt = cutlass.Int32(0)
                    emit_eq = cutlass.Int32(0)
                    v_p1 = cutlass.Float32(self.NEG_FLT_MAX)
                    if ix1 < cand_count:
                        v_p1 = self._smem_ld(cutlass.Float32, keys_base, ix1)
                        if v_p1 > sel_thr:
                            emit_gt = cutlass.Int32(1)
                        if v_p1 == sel_thr:
                            emit_eq = cutlass.Int32(1)
                    mask_gt = cute.arch.vote_ballot_sync(emit_gt != cutlass.Int32(0))
                    lane_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
                    if mask_gt != cutlass.Uint32(0):
                        moff_gt = cutlass.Int32(cute.arch.popc(mask_gt & lane_mask))
                        wpos_p1 = gt_run + moff_gt
                        if emit_gt != cutlass.Int32(0) and wpos_p1 < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[wpos_p1] = self.dtype(v_p1)
                            output_indices_row[wpos_p1] = self._smem_ld(
                                cutlass.Int32, vals_base, ix1
                            )
                        gt_run = gt_run + cutlass.Int32(cute.arch.popc(mask_gt))
                    mask_eq = cute.arch.vote_ballot_sync(emit_eq != cutlass.Int32(0))
                    if mask_eq != cutlass.Uint32(0):
                        moff_eq = cutlass.Int32(cute.arch.popc(mask_eq & lane_mask))
                        wpos_p2 = eq_run + moff_eq
                        if emit_eq != cutlass.Int32(0) and wpos_p2 < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[wpos_p2] = self.dtype(v_p1)
                            output_indices_row[wpos_p2] = self._smem_ld(
                                cutlass.Int32, vals_base, ix1
                            )
                        eq_run = eq_run + cutlass.Int32(cute.arch.popc(mask_eq))
                    base_w = base_w + cutlass.Int32(num_threads)
                cute.arch.barrier()
            else:
                # Pass 1: v > sel_thr, strided over (warp_id * WARP_SIZE, ...).
                base_w = warp_id * cutlass.Int32(self.WARP_SIZE)
                while base_w < cand_count:
                    ix1 = base_w + lane
                    emit_gt = cutlass.Int32(0)
                    v_p1 = cutlass.Float32(self.NEG_FLT_MAX)
                    if ix1 < cand_count:
                        v_p1 = self._smem_ld(cutlass.Float32, keys_base, ix1)
                        if v_p1 > sel_thr:
                            emit_gt = cutlass.Int32(1)
                    mask_gt = cute.arch.vote_ballot_sync(emit_gt != cutlass.Int32(0))
                    if mask_gt != cutlass.Uint32(0):
                        cnt_gt = cutlass.Int32(cute.arch.popc(mask_gt))
                        lane_mask_gt = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(
                            1
                        )
                        moff_gt = cutlass.Int32(cute.arch.popc(mask_gt & lane_mask_gt))
                        bp_gt = cutlass.Int32(0)
                        if lane == cutlass.Int32(0):
                            bp_gt = atomicAdd(
                                s_iscalars.iterator + cutlass.Int32(4),
                                cnt_gt,
                            )
                        bp_gt = cute.arch.shuffle_sync(bp_gt, cutlass.Int32(0))
                        wpos_p1 = bp_gt + moff_gt
                        if emit_gt != cutlass.Int32(0) and wpos_p1 < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[wpos_p1] = self.dtype(v_p1)
                            output_indices_row[wpos_p1] = self._smem_ld(
                                cutlass.Int32, vals_base, ix1
                            )
                    base_w = base_w + cutlass.Int32(num_threads)
                cute.arch.barrier()

                # Pass 2: v == sel_thr (same pattern + guard as Pass 1).
                base_w2 = warp_id * cutlass.Int32(self.WARP_SIZE)
                while base_w2 < cand_count:
                    ix2 = base_w2 + lane
                    emit_eq = cutlass.Int32(0)
                    v_p2 = cutlass.Float32(self.NEG_FLT_MAX)
                    if ix2 < cand_count:
                        v_p2 = self._smem_ld(cutlass.Float32, keys_base, ix2)
                        if v_p2 == sel_thr:
                            emit_eq = cutlass.Int32(1)
                    mask_eq = cute.arch.vote_ballot_sync(emit_eq != cutlass.Int32(0))
                    if mask_eq != cutlass.Uint32(0):
                        cnt_eq = cutlass.Int32(cute.arch.popc(mask_eq))
                        lane_mask_eq = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(
                            1
                        )
                        moff_eq = cutlass.Int32(cute.arch.popc(mask_eq & lane_mask_eq))
                        bp_eq = cutlass.Int32(0)
                        if lane == cutlass.Int32(0):
                            bp_eq = atomicAdd(
                                s_iscalars.iterator + cutlass.Int32(4),
                                cnt_eq,
                            )
                        bp_eq = cute.arch.shuffle_sync(bp_eq, cutlass.Int32(0))
                        wpos_p2 = bp_eq + moff_eq
                        if emit_eq != cutlass.Int32(0) and wpos_p2 < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[wpos_p2] = self.dtype(v_p2)
                            output_indices_row[wpos_p2] = self._smem_ld(
                                cutlass.Int32, vals_base, ix2
                            )
                    base_w2 = base_w2 + cutlass.Int32(num_threads)
                cute.arch.barrier()

            # Pad remainder with -self.FLT_MAX / -1. Single-pass filled =
            # cge (= cgt + total ties at sel_thr, from the converged snap
            # iteration; the zero-atomic path leaves counters untouched);
            # two-pass filled = counter [4] (gt + eq accumulated).
            filled_par = cutlass.Int32(0)
            if done_snap == cutlass.Int32(1):
                if cutlass.const_expr(self.p4_warp_redundant):
                    filled_par = cge_r
                else:
                    filled_par = s_iscalars[2]
            else:
                filled_par = s_iscalars[4]
            if filled_par > cutlass.Int32(kK):
                filled_par = cutlass.Int32(kK)
            ipad = filled_par + tidx
            while ipad < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                output_indices_row[ipad] = cutlass.Int32(-1)
                ipad = ipad + cutlass.Int32(num_threads)

        else:
            # ----- Branch C: cand_count < kK -----
            # Emit cand_count + pad
            i10 = tidx
            while i10 < cand_count:
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i10] = self.dtype(
                        self._smem_ld(cutlass.Float32, keys_base, i10)
                    )
                output_indices_row[i10] = self._smem_ld(cutlass.Int32, vals_base, i10)
                i10 = i10 + cutlass.Int32(num_threads)
            if s_iscalars[6] == cutlass.Int32(0):  # plateau fill completes done=3
                i11 = cand_count + tidx
                while i11 < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[i11] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[i11] = cutlass.Int32(-1)
                    i11 = i11 + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # Main kernel — one CTA per row
    # CUDA source: heuristicTopKDecode.cu:49-93 (heuristicTopKMultiRowKernel)
    # ------------------------------------------------------------------
    @cute.kernel
    def gvr_topk_kernel(
        self,
        input_data: cute.Tensor,  # [numRows, stride0] dtype
        pre_idx: cute.Tensor,  # [numRows / next_n, pre_idx_stride] int32
        seq_lens: cute.Tensor,  # [numRows / next_n] int32
        output_values: cute.Tensor,  # [numRows, top_k] dtype
        output_indices: cute.Tensor,  # [numRows, top_k] int32
        order_row: cute.Tensor,  # [batch_size] int32 (or None when seqlen_sorted=False)
        block_max: cute.Tensor,  # [numRows, nb_pad*4] fp32 (or None: no block-skip)
        seed_thr: cute.Tensor,  # [numRows, 3] fp32 (or None: no ext counts)
        seed_counts: cute.Tensor,  # [numRows, 3] int32 (or None)
        xstate: cute.Tensor,  # [numRows, 8] fp32 closed-loop state (or None)
        cand_vals: cute.Tensor,  # [numRows, CAP] fp32 scores (or None)
        cand_idx: cute.Tensor,  # [numRows, CAP] int32 positions (or None)
        cand_ctl: cute.Tensor,  # [numRows, 4] int32 {n0, void, n1, n2} (or None)
    ):
        """Thin entry: bidx → row_idx → run_one_row.

        grid = (num_rows * cluster_size,) where num_rows = batch_size *
        next_n. cluster_id = bidx // cluster_size, cta_in_cluster ∈
        [0, cluster_size). CTA r scans row[r * N / cs : (r+1) * N / cs]
        in Phase 2, so the per-row GE-count scales as 1 / cs. At
        cluster_size == 1 this collapses to one CTA per row scanning
        the whole row.

        When ``self.seqlen_sorted`` is True, the LJF dispatch order
        operates at REQUEST granularity (``order_row`` has length
        batch_size = num_rows / next_n). The owning row is resolved as
        ``order_row[cluster_id // next_n] * next_n + cluster_id % next_n``
        so the ``next_n`` rows of one request stay contiguous in
        dispatch order. All ``cluster_size`` CTAs within a cluster see
        the same ``cluster_id`` and therefore the same row, preserving
        cluster-sync semantics.

        Body is extracted into :meth:`run_one_row` so other entries (e.g.
        the LB load-balance variant) can resolve ``row_idx`` differently
        from the mappings used here.
        """
        bidx, _, _ = cute.arch.block_idx()
        cluster_size = cutlass.const_expr(self.cluster_size)
        seqlen_sorted = cutlass.const_expr(self.seqlen_sorted)
        next_n = cutlass.const_expr(self.next_n)
        if cutlass.const_expr(cluster_size > 1):
            cluster_id = bidx // cluster_size
        else:
            cluster_id = bidx
        if cutlass.const_expr(seqlen_sorted):
            # order_row is request-level (batch_size); expand to row-level
            # via req_id * next_n + nn so a request's next_n rows stay
            # contiguous in dispatch order (mirrors the LB main entry).
            if cutlass.const_expr(next_n == 1):
                row_idx = order_row[cluster_id]
            else:
                req_offset = cluster_id // cutlass.Int32(next_n)
                nn = cluster_id % cutlass.Int32(next_n)
                req_id = order_row[req_offset]
                row_idx = req_id * cutlass.Int32(next_n) + nn
        else:
            row_idx = cluster_id
        self.run_one_row(
            row_idx,
            input_data,
            pre_idx,
            seq_lens,
            output_values,
            output_indices,
            block_max=block_max,
            seed_thr=seed_thr,
            seed_counts=seed_counts,
            xstate=xstate,
            cand_vals=cand_vals,
            cand_idx=cand_idx,
            cand_ctl=cand_ctl,
        )

    @cute.jit
    def run_one_row(
        self,
        row_idx,  # int32, owning row in [0, num_rows)
        input_data: cute.Tensor,  # [numRows, stride0] dtype
        pre_idx: cute.Tensor,  # [numRows / next_n, pre_idx_stride] int32
        seq_lens: cute.Tensor,  # [numRows / next_n] int32
        output_values: cute.Tensor,  # [numRows, top_k] dtype, optional
        output_indices: cute.Tensor,  # [numRows, top_k] int32
        block_max: cute.Tensor = None,  # [numRows, nb_pad*4] fp32
        seed_thr: cute.Tensor = None,  # [numRows, 3] fp32 (ext counts)
        seed_counts: cute.Tensor = None,  # [numRows, 3] int32 (ext counts)
        xstate: cute.Tensor = None,  # [numRows, 8] fp32 (emit_xstate)
        cand_vals: cute.Tensor = None,  # [numRows, CAP] fp32 (ext cand)
        cand_idx: cute.Tensor = None,  # [numRows, CAP] int32 (ext cand)
        cand_ctl: cute.Tensor = None,  # [numRows, 4] int32 (ext cand)
    ):
        """Dispatch: compute per-row slice + cluster sync mode, call _run_phases.

        ``run_one_row`` only handles row resolution, SMEM allocation, and
        the per-row long-vs-short decision. Phase 1-4 are in
        :meth:`_run_phases`.

        Short-row degrade: when the actual row workload fits within ONE
        CTA's design slice (``ceil(max_seq_len / cluster_size)``), CTA 0
        solo-scans the row (do_cluster_sync=False, no cluster sync) and
        the other cluster CTAs fall through ``run_one_row`` without
        calling ``_run_phases``. CuTe DSL doesn't support runtime
        ``return``, so non-leader CTAs naturally reach
        ``griddepcontrol_launch_dependents`` at the end.
        """
        tidx, _, _ = cute.arch.thread_idx()

        next_n = cutlass.const_expr(self.next_n)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        kC = cutlass.const_expr(self.kC)
        kNumBins = cutlass.const_expr(self.kNumBins)
        cluster_size = cutlass.const_expr(self.cluster_size)

        warp_id = tidx // self.WARP_SIZE
        lane = tidx & (self.WARP_SIZE - 1)

        if cutlass.const_expr(cluster_size > 1):
            cta_in_cluster = cute.arch.block_idx_in_cluster()
        else:
            cta_in_cluster = cutlass.Int32(0)
        pre_idx_row_idx = row_idx // next_n
        # Temporal-shift offset, mirroring heuristicTopKDecode.cu PR #14219:
        #   cr == 1 (V3.2): (row % next_n) + 1 maps prev-step indices into this
        #     step's KV space (+1 for the newly appended token).
        #   cr  > 1 (V4):   0 — in compressed-index space, new entries are
        #     appended at the end so prev indices remain valid as-is.
        if cutlass.const_expr(self.compress_ratio == 1):
            pre_idx_offset = cutlass.Int32(row_idx % next_n) + cutlass.Int32(1)
        else:
            pre_idx_offset = cutlass.Int32(0)

        # Per-row length. seq_lens is in uncompressed-token space; logits/preIdx
        # live in compressed-token-index space when cr > 1 → divide by cr.
        seq_len = seq_lens[pre_idx_row_idx]
        actual_kv_len = (
            seq_len - cutlass.Int32(next_n) + cutlass.Int32(row_idx % next_n) + cutlass.Int32(1)
        )
        if cutlass.const_expr(self.compress_ratio == 1):
            N = actual_kv_len
        else:
            N = actual_kv_len // cutlass.Int32(self.compress_ratio)

        # Slice per-row views.
        input_row = input_data[row_idx, None]
        pre_idx_row = pre_idx[pre_idx_row_idx, None]
        # trace-time contract checks: a feature flag without its tensor
        # must fail HERE, not as a NoneType subscript deep in the phases
        if cutlass.const_expr((self.use_ext_counts or self.ext_rungs) and seed_thr is None):
            raise ValueError("use_ext_counts/ext_rungs kernels require seed_thr")
        if cutlass.const_expr(self.emit_xstate and xstate is None):
            raise ValueError("emit_xstate kernels require xstate")
        if cutlass.const_expr(
            self.use_ext_cand and (cand_vals is None or cand_idx is None or cand_ctl is None)
        ):
            raise ValueError("use_ext_cand kernels require cand_vals/cand_idx/cand_ctl")
        if cutlass.const_expr(self.enable_block_skip and block_max is None):
            raise ValueError("enable_block_skip kernels require block_max")
        if cutlass.const_expr(self.enable_block_skip and block_max is not None):
            block_max_row = block_max[row_idx, None]
        else:
            block_max_row = None
        if cutlass.const_expr(self.use_ext_counts and seed_thr is not None):
            # packed seed row [>=6] fp32: [0..2] lines, [3..5] counts as
            # floats (exact to 2^24) - ONE 32B sector serves both
            seed_thr_row = seed_thr[row_idx, None]
            seed_counts_row = None
        elif cutlass.const_expr(self.ext_rungs and seed_thr is not None):
            seed_thr_row = seed_thr[row_idx, None]
            seed_counts_row = None
        else:
            seed_thr_row = None
            seed_counts_row = None
        if cutlass.const_expr(self.emit_xstate and xstate is not None):
            xstate_row = xstate[row_idx, None]
        else:
            xstate_row = None
        if cutlass.const_expr(
            self.use_ext_cand
            and cand_vals is not None
            and cand_idx is not None
            and cand_ctl is not None
        ):
            cand_vals_row = cand_vals[row_idx, None]
            cand_idx_row = cand_idx[row_idx, None]
            cand_ctl_row = cand_ctl[row_idx, None]
        elif cutlass.const_expr(self.self_scan and cand_idx is not None):
            # self_scan: only the POSITION column exists (values live in
            # smem from birth; counts come from the phase-0 cursors)
            cand_vals_row = None
            cand_idx_row = cand_idx[row_idx, None]
            cand_ctl_row = None
        else:
            cand_vals_row = None
            cand_idx_row = None
            cand_ctl_row = None
        # When return_output_values=False, ``output_values`` is None at
        # launch and the gated writes below are compiled out; slicing into
        # None would crash so we keep the view None as well.
        if cutlass.const_expr(self.return_output_values):
            output_values_row = output_values[row_idx, None]
        else:
            output_values_row = None
        output_indices_row = output_indices[row_idx, None]
        pre_idx_count = pre_idx.shape[1]

        if cutlass.const_expr(not self.pdl_wait_late):
            griddepcontrol_wait()

        # ---- Shared memory allocation ----
        smem = SmemAllocator()
        # keys[kC] fp32 (P3 candidate values; smem keys always fp32 even for half-prec)
        # Use fp32 even for half-prec to make secant search algorithm keep the accuracy/precision and converge faster.
        # self_scan: enlarged to seg_total (three value segments at bases
        # 0 / accept_cap / 2*accept_cap); every later consumer only ever
        # touches a <= kC prefix after cut compaction.
        smem_keys = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((cutlass.const_expr(self.seg_total),), order=(0,)),
            byte_alignment=128,
        )
        # vals[kC] int32 (P3 candidate indices). self_scan: holds the
        # SEGMENT COORDINATE of each compacted candidate (identity for the
        # deferred position gather via cand_idx[coord]) — every consumer
        # (P4, tail repair, gather) works unchanged.
        if cutlass.const_expr(self.self_scan):
            # phase-0 cp.async staging, ALIASED over vals: vals is only
            # written after phase 0 completes and every in-flight group
            # is drained inside the dense loop, so the lifetimes never
            # overlap. Slot-major (slot s of thread t at row
            # s*num_threads + t): a warp's 16B reads/writes land on
            # consecutive banks, conflict-free.
            smem_stage = smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=cute.make_ordered_layout(
                    (cutlass.const_expr(self.stage_rows), 4), order=(1, 0)
                ),
                byte_alignment=128,
            )
            smem_vals = cute.make_tensor(
                cute.recast_ptr(smem_stage.iterator, dtype=cutlass.Int32),
                cute.make_ordered_layout((kC,), order=(0,)),
            )
        else:
            smem_stage = None
            smem_vals = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((kC,), order=(0,)),
                byte_alignment=128,
            )
        # histogram[kNumBins] int32 (P4 only)
        smem_hist = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((kNumBins,), order=(0,)),
            byte_alignment=128,
        )
        # per_thread_counts[BLOCK_SIZE] int32 (P2/P3 cached counts)
        smem_ptcnt = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((num_threads,), order=(0,)),
            byte_alignment=128,
        )
        # block-skip: int16 active list (16KB at 8192 entries) + control
        # ([0] list length, [1] list-current flag for the Phase-3 reuse).
        if cutlass.const_expr(self.enable_block_skip):
            smem_active = smem.allocate_tensor(
                element_type=cutlass.Int16,
                layout=cute.make_ordered_layout((self.SKIP_MAX_BLOCKS,), order=(0,)),
                byte_alignment=128,
            )
            s_active_cnt = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((4,), order=(0,)),
                byte_alignment=16,
            )
        else:
            smem_active = None
            s_active_cnt = None
        # warp_counts[NUM_WARPS] int32 (P3 prefix-sum scratch)
        # p2_warp_redundant parity-banks the Phase-2 staging (a warp one
        # round ahead writes the other half) — costs num_warps*4 bytes.
        smem_wcnt = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout(
                (2 * num_warps if cutlass.const_expr(self.p2_warp_redundant) else num_warps,),
                order=(0,),
            ),
            byte_alignment=128,
        )
        # Phase-1 warp aggregates (fp32 + int32; ~256 bytes total)
        smem_wmin = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=64,
        )
        smem_wmax = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=64,
        )
        smem_wsum = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=64,
        )
        smem_wcnt_p1 = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=64,
        )
        # Float scalars: threshold, val_lo, val_hi
        s_thr = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((3,), order=(0,)),
            byte_alignment=16,
        )
        # Int scalars:
        #   [0] cand_count   (cluster-aggregated total at cs>1; local total at cs=1)
        #   [1] done
        #   [2] cnt_lo
        #   [3] cnt_hi
        #   [4] out_count
        #   [5] local cand_count  (per-CTA snapshot before cluster all-reduce;
        #                          consumed by the kernel-level cluster handoff)
        #   [6] plateau terminal flag, captured from [1] BEFORE Phase 4
        #       (Phase 4 REUSES [1] as radix scratch, so the terminal must
        #       never be re-read from it afterwards)
        #   [7] plateau fill ticket (done == 3 only)
        s_iscalars = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((8,), order=(0,)),
            byte_alignment=16,
        )
        # Per-CTA DSMEM scratch for the cluster all-reduce of cand_count:
        # slots 0/1 = parity double-buffered count exchange (call k writes
        # slot k&1 — closes the straggler-read-vs-next-write DSMEM race),
        # slot 2 = tid0-private call counter. mapa.shared::cluster relies
        # on every CTA holding this block at the SAME SMEM offset, so it's
        # allocated once here. Only USED at cs>1 (uses are gated by
        # const_expr(cs>1)), but ALLOCATED unconditionally: the LB hybrid
        # kernel inlines a cs>1 and a cs=1 instance into one launch, and the
        # DSL sizes the launch SMEM from the last-traced SmemAllocator only —
        # the layouts must stay byte-identical across cluster_size (16B cost
        # at cs=1).
        s_cluster_partial = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((3,), order=(0,)),
            byte_alignment=16,
        )
        if cutlass.const_expr(cluster_size > 1):
            # Zero the call counter before any block_count_ge call. tid0-
            # private (same thread reads/increments it), so program order
            # suffices — but parity must start at 0 on EVERY CTA of the
            # cluster for lockstep alignment.
            if tidx == cutlass.Int32(0):
                s_cluster_partial[2] = cutlass.Int32(0)

        # SMEM slice cache (optional). Sized in ``self.dtype`` so the same
        # vec_w-wide LDG→STS→LDS pipeline works for fp32/bf16/fp16.
        # enable_smem_cache=False by default; caller ensures slice_len <=
        # smem_cache_elems before enabling (no runtime guard in kernel).
        if cutlass.const_expr(self.enable_smem_cache):
            smem_input = smem.allocate_tensor(
                element_type=self.dtype,
                layout=cute.make_ordered_layout((self.smem_cache_elems,), order=(0,)),
                byte_alignment=128,
            )
        else:
            smem_input = None

        # op#26 R0 admission scratch (single-CTA fast path). Allocated only
        # when enable_r0; None otherwise so the base SMEM layout is byte-for-
        # byte unchanged and these propagate harmlessly through _run_phases'
        # const_expr(enable_r0)-gated branch (same idiom as s_cluster_partial
        # / smem_input above). smem_ptcnt_multi caches M per-thread count
        # columns; s_r0col carries the accepted rung index tid0 -> all.
        if cutlass.const_expr(self.enable_r0):
            M_r0 = cutlass.const_expr(self.M_thr)
            # vseed (v3): the pmean column's per-thread counts reuse the
            # existing single-column smem_ptcnt buffer, so the BIG multi
            # buffer only holds the M_qf rung columns -> zero smem growth
            # (the round-1 +2-4KB column pushed 16-bit mb3/T1024 configs over
            # an occupancy cliff: K2048 fp16 BS1024 -26%).
            M_r0_pt = cutlass.const_expr(self.M_qf)
            s_mt_thr = smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=cute.make_ordered_layout((M_r0,), order=(0,)),
                byte_alignment=16,
            )
            smem_ptcnt_multi = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((M_r0_pt * num_threads,), order=(0,)),
                byte_alignment=128,
            )
            smem_wcnt_multi = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((M_r0 * num_warps,), order=(0,)),
                byte_alignment=64,
            )
            s_mt_cnt = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((M_r0,), order=(0,)),
                byte_alignment=16,
            )
            s_r0col = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=16,
            )
            # DSMEM scratch for the M-way cluster all-reduce of the R0 rung
            # counts (mapa.shared::cluster needs the same offset on every
            # CTA). Only USED at cs>1; allocated unconditionally so the
            # cs=1 / cs>1 SMEM layouts stay byte-identical for the LB
            # hybrid kernel (see s_cluster_partial above).
            s_cluster_partial_m = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((M_r0,), order=(0,)),
                byte_alignment=16,
            )
            # p1b_cache: P1 stashes the K gathered preIdx values here so P1b
            # skips a second GMEM random gather (dtype-gated: 16-bit only).
            if cutlass.const_expr(self.p1b_cache):
                smem_gath = smem.allocate_tensor(
                    element_type=cutlass.Float32,
                    layout=cute.make_ordered_layout((self.top_k,), order=(0,)),
                    byte_alignment=128,
                )
            else:
                smem_gath = None
        else:
            s_mt_thr = None
            smem_ptcnt_multi = None
            smem_wcnt_multi = None
            s_mt_cnt = None
            s_r0col = None
            s_cluster_partial_m = None
            smem_gath = None

        # PDL wait placed as late as possible so the prologue overlaps
        # the producer indexer's tail. INVARIANT: nothing above may read
        # producer-written data - seq_lens/pre_idx come from host-side
        # metadata and the feedback buffer; logits / block_max /
        # seed_thr / cand are first touched below.
        if cutlass.const_expr(self.pdl_wait_late):
            griddepcontrol_wait()

        # ---- Per-row dispatch ----
        # Three branches:
        #   1. Degenerate (N <= top_k): no GVR work, leader emits identity.
        #   2. cs>1 long row:           all cluster CTAs cooperate.
        #   3. cs>1 short row OR cs=1:  leader/single CTA runs solo.
        # Non-leader CTAs in (1)/(3) fall through to the function end (CuTe
        # DSL doesn't support runtime ``return``).
        top_k = cutlass.const_expr(self.top_k)
        if N <= cutlass.Int32(top_k):
            # Degenerate: no GVR, just emit [0..N-1] + (-1) padding.
            # Leader-only write (was an idempotent race across cluster CTAs).
            if cta_in_cluster == cutlass.Int32(0):
                jd = tidx
                while jd < N:
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[jd] = input_row[jd]
                    output_indices_row[jd] = cutlass.Int32(jd)
                    jd = jd + cutlass.Int32(num_threads)
                jp = N + cutlass.Int32(tidx)
                while jp < cutlass.Int32(top_k):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[jp] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[jp] = cutlass.Int32(-1)
                    jp = jp + cutlass.Int32(num_threads)
        else:
            # Normal GVR. Long vs short row decision threshold =
            # ceil(max_seq_len / cluster_size) = one CTA's design
            # workload. When actual seq_len fits within that, cluster
            # cooperation overhead exceeds the work saved → degrade to
            # CTA 0 solo.
            if cutlass.const_expr(cluster_size > 1):
                # max_slice_len: per-CTA slice upper bound when the row is
                # long enough to warrant cluster cooperation.
                max_slice_len = (
                    input_data.shape[1] + cutlass.Int32(cluster_size - 1)
                ) // cutlass.Int32(cluster_size)
                if N > max_slice_len:
                    # Long row: cluster cooperation, all cs CTAs scan
                    # N/cs. slice_base rounded DOWN to vec_w so each
                    # CTA's slice_start stays vec_w-aligned; the last
                    # CTA absorbs the N mod cs remainder.
                    vec_w_const = cutlass.const_expr(self.vec_bits // self.dtype.width)
                    raw_base = N // cutlass.Int32(cluster_size)
                    slice_base = (raw_base // cutlass.Int32(vec_w_const)) * cutlass.Int32(
                        vec_w_const
                    )
                    slice_start = cta_in_cluster * slice_base
                    slice_is_last = cta_in_cluster == cutlass.Int32(cluster_size - 1)
                    slice_end = N if slice_is_last else (slice_start + slice_base)
                    self._run_phases(
                        input_row,
                        pre_idx_row,
                        output_values_row,
                        output_indices_row,
                        N,
                        pre_idx_offset,
                        pre_idx_count,
                        slice_start,
                        slice_end,
                        cutlass.Boolean(True),
                        cta_in_cluster,
                        smem_keys,
                        smem_vals,
                        smem_hist,
                        smem_ptcnt,
                        smem_wcnt,
                        smem_wmin,
                        smem_wmax,
                        smem_wsum,
                        smem_wcnt_p1,
                        s_thr,
                        s_iscalars,
                        s_cluster_partial,
                        smem_input,
                        s_mt_thr,
                        smem_ptcnt_multi,
                        smem_wcnt_multi,
                        s_mt_cnt,
                        s_r0col,
                        s_cluster_partial_m,
                        smem_gath,
                        tidx,
                        warp_id,
                        lane,
                        block_max_row=block_max_row,
                        seed_thr_row=seed_thr_row,
                        seed_counts_row=seed_counts_row,
                        xstate_row=xstate_row,
                        cand_vals_row=cand_vals_row,
                        cand_idx_row=cand_idx_row,
                        cand_ctl_row=cand_ctl_row,
                        smem_active=smem_active,
                        s_active_cnt=s_active_cnt,
                        smem_stage=smem_stage,
                    )
                else:
                    # Short row: only CTA 0 scans the full row; the other
                    # (cluster_size - 1) CTAs fall through without entering
                    # _run_phases and naturally reach the function end.
                    if cta_in_cluster == cutlass.Int32(0):
                        self._run_phases(
                            input_row,
                            pre_idx_row,
                            output_values_row,
                            output_indices_row,
                            N,
                            pre_idx_offset,
                            pre_idx_count,
                            cutlass.Int32(0),
                            N,
                            cutlass.Boolean(False),
                            cta_in_cluster,
                            smem_keys,
                            smem_vals,
                            smem_hist,
                            smem_ptcnt,
                            smem_wcnt,
                            smem_wmin,
                            smem_wmax,
                            smem_wsum,
                            smem_wcnt_p1,
                            s_thr,
                            s_iscalars,
                            s_cluster_partial,
                            smem_input,
                            s_mt_thr,
                            smem_ptcnt_multi,
                            smem_wcnt_multi,
                            s_mt_cnt,
                            s_r0col,
                            s_cluster_partial_m,
                            smem_gath,
                            tidx,
                            warp_id,
                            lane,
                            block_max_row=block_max_row,
                            seed_thr_row=seed_thr_row,
                            seed_counts_row=seed_counts_row,
                            xstate_row=xstate_row,
                            cand_vals_row=cand_vals_row,
                            cand_idx_row=cand_idx_row,
                            cand_ctl_row=cand_ctl_row,
                            smem_active=smem_active,
                            s_active_cnt=s_active_cnt,
                            smem_stage=smem_stage,
                        )
            else:
                # cs=1: one CTA per row, no cluster sync.
                self._run_phases(
                    input_row,
                    pre_idx_row,
                    output_values_row,
                    output_indices_row,
                    N,
                    pre_idx_offset,
                    pre_idx_count,
                    cutlass.Int32(0),
                    N,
                    cutlass.Boolean(False),
                    cta_in_cluster,
                    smem_keys,
                    smem_vals,
                    smem_hist,
                    smem_ptcnt,
                    smem_wcnt,
                    smem_wmin,
                    smem_wmax,
                    smem_wsum,
                    smem_wcnt_p1,
                    s_thr,
                    s_iscalars,
                    s_cluster_partial,
                    smem_input,
                    s_mt_thr,
                    smem_ptcnt_multi,
                    smem_wcnt_multi,
                    s_mt_cnt,
                    s_r0col,
                    s_cluster_partial_m,
                    smem_gath,
                    tidx,
                    warp_id,
                    lane,
                    block_max_row=block_max_row,
                    seed_thr_row=seed_thr_row,
                    seed_counts_row=seed_counts_row,
                    xstate_row=xstate_row,
                    cand_vals_row=cand_vals_row,
                    cand_idx_row=cand_idx_row,
                    cand_ctl_row=cand_ctl_row,
                    smem_active=smem_active,
                    s_active_cnt=s_active_cnt,
                    smem_stage=smem_stage,
                )

        griddepcontrol_launch_dependents()

    @cute.jit
    def _run_phases(
        self,
        input_row,
        pre_idx_row,
        output_values_row,
        output_indices_row,
        N,
        pre_idx_offset,
        pre_idx_count,
        slice_start,
        slice_end,
        do_cluster_sync,
        cta_in_cluster,
        smem_keys,
        smem_vals,
        smem_hist,
        smem_ptcnt,
        smem_wcnt,
        smem_wmin,
        smem_wmax,
        smem_wsum,
        smem_wcnt_p1,
        s_thr,
        s_iscalars,
        s_cluster_partial,
        smem_input,
        s_mt_thr,
        smem_ptcnt_multi,
        smem_wcnt_multi,
        s_mt_cnt,
        s_r0col,
        s_cluster_partial_m,
        smem_gath,
        tidx,
        warp_id,
        lane,
        block_max_row=None,  # block-skip: this row's per-32-position bounds
        seed_thr_row=None,  # ext counts: this row's 3 seed thresholds (fp32)
        seed_counts_row=None,  # ext counts: this row's 3 exact counts (int32)
        xstate_row=None,  # emit_xstate: this row's [8] fp32 state slot
        cand_vals_row=None,  # ext cand: this row's [CAP] fp32 scores
        cand_idx_row=None,  # ext cand: this row's [CAP] int32 positions
        cand_ctl_row=None,  # ext cand: this row's [2] int32 {claimed, void}
        smem_active=None,
        s_active_cnt=None,
        smem_stage=None,  # self_scan: [stage_rows, 4] fp32 cp.async staging
    ):
        """Run Phase 1-4 + final cluster barrier on a given row slice.

        Caller (``run_one_row``) decides slice + do_cluster_sync per row:
          - cs=1                 → slice=[0,N), do_cluster_sync=False
          - cs>1, long row       → slice=N/cs per CTA, do_cluster_sync=True
          - cs>1, short row      → slice=[0,N), do_cluster_sync=False, CTA 0 only

        Non-leader CTAs in short-row mode never call this helper.
        """
        num_threads = cutlass.const_expr(self.num_threads)
        cluster_size = cutlass.const_expr(self.cluster_size)
        is_leader = cta_in_cluster == cutlass.Int32(0)

        # block-skip: the list-current flag starts INVALID every row; only
        # the R0 compact pass's build sets it. Ordered ahead of all readers
        # by Phase 1's internal barriers.
        if cutlass.const_expr(self.enable_block_skip):
            if tidx == cutlass.Int32(0):
                s_active_cnt[1] = cutlass.Int32(0)
                s_active_cnt[2] = cutlass.Int32(0)  # dropped-rung mask

        # ---- Per-row dynamic routing (ext counts) ----
        # Use the epilogue rungs ONLY when the row is valid (finite t_0,
        # xstate contract) AND some rung count already lies in [K, kC].
        # A miss/invalid row runs the full stock path (P1 + P1b + vseed +
        # count). All threads read the same control words, so the
        # predicate is CTA-uniform and the branches below stay convergent.
        if cutlass.const_expr(_P4_TAIL_DBG):
            ck0 = cutlass.Int64(0)
            ck1 = cutlass.Int64(0)
            ckE = cutlass.Int64(0)
            ckE = cute.arch.clock64()  # row-phase entry (device-residency ref)
        ext_row = cutlass.Int32(0)
        if cutlass.const_expr(self.use_ext_counts):
            # line validity mirrors ext_rungs: ALL THREE lines must be
            # finite and strictly ascending (a NaN in t1/t2 must not
            # reach the refine brackets); invalid rows fall to the stock
            # path, so exactness never rides on the host loop's line
            # quality.
            if (
                seed_thr_row[0] < cutlass.Float32(1e37)
                and seed_thr_row[0] > cutlass.Float32(-1e37)
                and seed_thr_row[1] > seed_thr_row[0]
                and seed_thr_row[2] > seed_thr_row[1]
            ):
                for m in cutlass.range_constexpr(cutlass.const_expr(self.M_thr)):
                    cm_e = cutlass.Int32(seed_thr_row[3 + m])
                    if cm_e >= cutlass.Int32(self.top_k) and cm_e <= cutlass.Int32(self.kC):
                        ext_row = cutlass.Int32(1)
            # list path preview: when the SoA candidate list will be taken
            # (count-only admission), Phase 1's gather buys nothing either
            # - reuse the same skip (the seed rungs still provide the
            # [t_0, t_2] bracket the degenerate check wants).
            if cutlass.const_expr(
                self.use_ext_cand
                and self.use_ext_counts
                and cluster_size == 1
                and self.dtype == cutlass.Float32
            ):
                claimed_p = cutlass.Int32(cand_ctl_row[0])
                void_p = cutlass.Int32(cand_ctl_row[1])
                # real (non-parked) lines only: the skip stages the raw
                # lines into the threshold scratch, and a parked line
                # (1e30) would poison every later bracket. Parked rows
                # keep Phase 1; the list take below is independent.
                if (
                    void_p == cutlass.Int32(0)
                    and claimed_p >= cutlass.Int32(self.top_k + 64)
                    and claimed_p <= cutlass.Int32(self.list_cap)
                    and seed_thr_row[0] < cutlass.Float32(1e37)
                    and seed_thr_row[0] > cutlass.Float32(-1e37)
                    and seed_thr_row[1] > seed_thr_row[0]
                    and seed_thr_row[2] > seed_thr_row[1]
                    and seed_thr_row[2] < cutlass.Float32(1e29)
                ):
                    ext_row = cutlass.Int32(1)
            # ---- self_scan phase 0: fused scan-bucket ----
            # The kernel streams the row itself (no external emitter);
            # eligibility mirrors the list contract: nothing dropped
            # (void == 0) and the loosest line provably covers the top-K
            # (n0 >= K, exact counts - no sentinel slack needed). The
            # seed-thr finite guard keeps the branch CTA-uniform, so the
            # barriers/ballots inside phase 0 stay convergent. Cursor
            # scratch = smem_wcnt_p1 (P1 only runs when this row is NOT
            # taken, so the reuse never overlaps live data).
            if cutlass.const_expr(
                self.self_scan and cluster_size == 1 and self.dtype == cutlass.Float32
            ):
                if seed_thr_row[0] < cutlass.Float32(1e37):
                    self.phase0_scan_bucket(
                        input_row,
                        N,
                        seed_thr_row,
                        smem_keys,
                        cand_idx_row,
                        block_max_row,
                        smem_stage,
                        smem_wcnt_p1,
                        tidx,
                        warp_id,
                        lane,
                    )
                    if smem_wcnt_p1[3] == cutlass.Int32(0) and smem_wcnt_p1[4] >= cutlass.Int32(
                        self.top_k
                    ):
                        ext_row = cutlass.Int32(1)

        # ---- Phase 1: preIdx Min/Max/Mean ----
        # ext counts: P1's only surviving products are the [v_lo, v_hi]
        # outer bracket and the scalar state init — the ext rungs provide
        # the bracket directly (host contract: t_0 < t_2, finite, all rows
        # valid), so the preIdx gather is skipped wholesale. A miss whose
        # target lies outside [t_0, t_2] recovers via the refine loop's
        # 8x bracket expansion (same fail-soft as the stock path).
        rungs_ok = cutlass.Int32(0)
        if cutlass.const_expr(self.ext_rungs):
            # runtime validity: finite AND strictly ascending; anything
            # else (cold start, dropped row, NaN from the host loop)
            # falls back to the stock seed path below - exactness never
            # rides on the host's line quality
            if (
                seed_thr_row[0] < cutlass.Float32(1e37)
                and seed_thr_row[0] > cutlass.Float32(-1e37)
                and seed_thr_row[1] > seed_thr_row[0]
                and seed_thr_row[2] > seed_thr_row[1]
            ):
                rungs_ok = cutlass.Int32(1)
        if cutlass.const_expr(self.ext_rungs):
            if rungs_ok == cutlass.Int32(1):
                # variant B: the rungs carry the bracket, so P1's gather
                # buys nothing - same seed-line init as the ext-counts
                # hit path.
                if tidx == cutlass.Int32(0):
                    s_thr[0] = seed_thr_row[1]
                    s_thr[1] = seed_thr_row[0]
                    s_thr[2] = seed_thr_row[2]
                    s_iscalars[0] = cutlass.Int32(0)  # cand_count
                    s_iscalars[1] = cutlass.Int32(0)  # done
                    s_iscalars[2] = cutlass.Int32(-1)  # cnt_lo (fb owns)
                    s_iscalars[3] = cutlass.Int32(-1)  # cnt_hi
                    s_iscalars[4] = cutlass.Int32(0)  # out_count
                cute.arch.barrier()
            if rungs_ok == cutlass.Int32(0):
                self.phase1_preidx_stats(
                    input_row,
                    N,
                    pre_idx_row,
                    pre_idx_count,
                    pre_idx_offset,
                    smem_wmin,
                    smem_wmax,
                    smem_wsum,
                    smem_wcnt_p1,
                    s_thr,
                    s_iscalars,
                    tidx,
                    warp_id,
                    lane,
                    smem_gath=smem_gath,
                    s_mt_thr=s_mt_thr,
                )
        if cutlass.const_expr(self.use_ext_counts):
            if ext_row == cutlass.Int32(1):
                if tidx == cutlass.Int32(0):
                    s_thr[0] = seed_thr_row[1]
                    s_thr[1] = seed_thr_row[0]
                    s_thr[2] = seed_thr_row[2]
                    s_iscalars[0] = cutlass.Int32(0)  # cand_count
                    s_iscalars[1] = cutlass.Int32(0)  # done
                    s_iscalars[2] = cutlass.Int32(-1)  # cnt_lo (fb seeding owns)
                    s_iscalars[3] = cutlass.Int32(-1)  # cnt_hi
                    s_iscalars[4] = cutlass.Int32(0)  # out_count
                    # rung parking folded into the SAME thread0 block
                    # (was a second thread0 block + barrier in the R0
                    # region): tightest in-band count picks the admitted
                    # line; single-column builds stage it in s_thr[0] and
                    # pre-mark the rung column (M_qf = accepted, -2 =
                    # defensive M-ary rerun).
                    bx_m = cutlass.Int32(-1)
                    bx_c = cutlass.Int32(2147483647)
                    for m in cutlass.range_constexpr(cutlass.const_expr(self.M_thr)):
                        cx = cutlass.Int32(seed_thr_row[3 + m])
                        if (
                            cx >= cutlass.Int32(self.top_k)
                            and cx <= cutlass.Int32(self.kC)
                            and cx < bx_c
                        ):
                            bx_m = cutlass.Int32(m)
                            bx_c = cx
                    for m in cutlass.range_constexpr(cutlass.const_expr(self.M_thr)):
                        if bx_m >= cutlass.Int32(0):
                            s_mt_thr[m] = seed_thr_row[bx_m]
                        else:
                            s_mt_thr[m] = seed_thr_row[m]
                    if cutlass.const_expr(not self.enable_block_skip):
                        if bx_m >= cutlass.Int32(0):
                            s_thr[0] = seed_thr_row[bx_m]
                            s_r0col[0] = cutlass.Int32(self.M_qf)
                        else:
                            s_r0col[0] = cutlass.Int32(-2)
                cute.arch.barrier()
            if ext_row == cutlass.Int32(0):
                self.phase1_preidx_stats(
                    input_row,
                    N,
                    pre_idx_row,
                    pre_idx_count,
                    pre_idx_offset,
                    smem_wmin,
                    smem_wmax,
                    smem_wsum,
                    smem_wcnt_p1,
                    s_thr,
                    s_iscalars,
                    tidx,
                    warp_id,
                    lane,
                    smem_gath=smem_gath,  # p1b_cache: stash gathered values (None-op OFF)
                    s_mt_thr=s_mt_thr,  # r0_vseed: park pmean in the last rung column
                )
        if cutlass.const_expr(not (self.use_ext_counts or self.ext_rungs)):
            self.phase1_preidx_stats(
                input_row,
                N,
                pre_idx_row,
                pre_idx_count,
                pre_idx_offset,
                smem_wmin,
                smem_wmax,
                smem_wsum,
                smem_wcnt_p1,
                s_thr,
                s_iscalars,
                tidx,
                warp_id,
                lane,
                smem_gath=smem_gath,  # p1b_cache: stash gathered values (None-op OFF)
                s_mt_thr=s_mt_thr,  # r0_vseed: park pmean in the last rung column
            )

        # Degenerate threshold init: val_hi <= -self.FLT_MAX or val_lo >= val_hi.
        # A duplicate/invalid preIdx gather (cold-start zero-init slots, stale
        # slots pointing past N, an all-tied gather) produces an unusable
        # bracket. When N > K real selection work remains, so rebuild the
        # bracket from the data itself (P1r) and run the normal pipeline;
        # an identity shortcut here would NOT be the top-K. If the bracket
        # is STILL degenerate after the rescue, every in-range value is
        # identical (or N <= K), and identity output is then exact — keep
        # the shortcut for exactly those rows.
        if cutlass.const_expr(self.p1r_rescue):
            v_lo = s_thr[1]
            v_hi = s_thr[2]
            if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
                if N > cutlass.Int32(self.top_k):
                    self.phase1r_data_reseed(
                        input_row,
                        N,
                        smem_wmin,
                        smem_wmax,
                        s_thr,
                        s_iscalars,
                        s_mt_thr,
                        tidx,
                        warp_id,
                        lane,
                    )
        v_lo = s_thr[1]
        v_hi = s_thr[2]
        if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
            if cutlass.const_expr(cluster_size == 1):
                if tidx == 0:
                    top_k = cutlass.const_expr(self.top_k)
                    # Emit identity output (first min(top_k, N) indices)
                    emit_count = cutlass.Int32(top_k) if cutlass.Int32(top_k) < N else N
                    je = cutlass.Int32(0)
                    while je < emit_count:
                        output_indices_row[je] = je
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[je] = input_row[je]
                        je = je + cutlass.Int32(1)
                    if cutlass.const_expr(self.emit_xstate):
                        xstate_row[0] = cutlass.Float32(0.0)  # degenerate
            else:
                # cs>1: all cluster CTAs enter _run_phases; only leader writes.
                if is_leader & (tidx == cutlass.Int32(0)):
                    top_k = cutlass.const_expr(self.top_k)
                    # Emit identity output (first min(top_k, N) indices)
                    emit_count = cutlass.Int32(top_k) if cutlass.Int32(top_k) < N else N
                    je = cutlass.Int32(0)
                    while je < emit_count:
                        output_indices_row[je] = je
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[je] = input_row[je]
                        je = je + cutlass.Int32(1)
                    if cutlass.const_expr(self.emit_xstate):
                        xstate_row[0] = cutlass.Float32(0.0)  # degenerate
        else:
            # ---- List path: known-counts admission ----
            # The emitter wrote an SoA list collected at t0 = seed_thr[0]
            # and counted the two tighter lines on the way out, so the
            # control words carry {n0, void, n1, n2} with n_i = #(>= t_i).
            # Admission is then a scalar lookup:
            #   1. some n_i in [K, B*] -> cut at the tightest, one pass
            #   2. every line straddles/overshoots -> histogram over the
            #      list clamped between the two known bracket lines
            #   3. void, or n0 < K + 64 (emitter sentinel bound) -> fall
            #      back. The slack also lets accepted cuts load without
            #      an overflow net: count and load are the same compare.
            # cs>1 and 16-bit dtypes keep the plain fallback.
            take_cand = cutlass.Int32(0)
            # Python bool: with every list/scan feature off, the stock
            # Phase 2 + Phase 3 below trace straight-line rather than as
            # one 470-line scf.if region with a large yield list.
            run_stock_p23 = True
            list_used = cutlass.Int32(0)  # list path taken (xstate publish)
            claimed_c = cutlass.Int32(0)
            if cutlass.const_expr(
                self.use_ext_cand
                and self.use_ext_counts
                and cluster_size == 1
                and self.dtype == cutlass.Float32
            ):
                # ---- List path: bucketed segments ----
                # The emitter classifies each entry by the tightest line
                # it passes and appends into one of three fixed segments
                # (A = [0, segA) holds >= t2, B = [segA, 2*segA) holds
                # [t1, t2), C = [2*segA, ...) holds [t0, t1)); a full
                # segment spills to the next looser one. Segment caps =
                # B*, so "line in the acceptance band" <=> "its segment
                # prefix group is complete" - the same condition the cut
                # selection already checks. A LINE cut therefore loads a
                # dense prefix of known length: a pure mapped copy, no
                # filtering, no ballots, no atomics, zero wasted reads.
                # Histogram fallbacks value-scan the mapped extents.
                claimed_c = cutlass.Int32(cand_ctl_row[0])
                void_c = cutlass.Int32(cand_ctl_row[1])
                n1_c = cutlass.Int32(cand_ctl_row[2])
                n2_c = cutlass.Int32(cand_ctl_row[3])
                # segment bases/extents follow the EMITTER geometry
                # (accept_cap); the admission bound is additionally
                # clamped by the physical candidate capacity kC.
                segA = cutlass.const_expr(self.accept_cap)
                bstar = cutlass.const_expr(min(self.accept_cap, self.kC))
                if cutlass.const_expr(_P4_TAIL_DBG):
                    ck0 = cute.arch.clock64()
                    ck1 = ck0
                # segment extents (pads live at C's tail: sentinel score
                # -inf slots, harmless to copy, never rank)
                lenA = n2_c
                if lenA > cutlass.Int32(segA):
                    lenA = cutlass.Int32(segA)
                spillA = n2_c - lenA
                lenB = n1_c - n2_c + spillA
                if lenB > cutlass.Int32(segA):
                    lenB = cutlass.Int32(segA)
                lenC = claimed_c - lenA - lenB
                total_l = claimed_c
                usable = cutlass.Int32(0)
                if (
                    void_c == cutlass.Int32(0)
                    and claimed_c >= cutlass.Int32(self.top_k + 64)
                    and claimed_c <= cutlass.Int32(self.list_cap)
                ):
                    usable = cutlass.Int32(1)
                kK_l = cutlass.Int32(self.top_k)
                bs_l = cutlass.Int32(bstar)
                cut_t = cutlass.Float32(0.0)
                cut_n = cutlass.Int32(0)
                have = cutlass.Int32(0)
                line_cut = cutlass.Int32(0)
                anch_t = cutlass.Float32(0.0)
                vbase = cutlass.Int64(0)
                ibase = cutlass.Int64(0)
                if cutlass.const_expr(True):
                    vbase = cand_vals_row.iterator.toint()
                    ibase = cand_idx_row.iterator.toint()
                if usable == cutlass.Int32(1):
                    # cut = tightest line in [K, B*]; anchor = loosest.
                    if n2_c >= kK_l and n2_c <= bs_l:
                        cut_t = seed_thr_row[2]
                        cut_n = n2_c
                        anch_t = seed_thr_row[2]
                        have = cutlass.Int32(1)
                        line_cut = cutlass.Int32(1)
                    if n1_c >= kK_l and n1_c <= bs_l:
                        if have == cutlass.Int32(0):
                            cut_t = seed_thr_row[1]
                            cut_n = n1_c
                        anch_t = seed_thr_row[1]
                        have = cutlass.Int32(1)
                        line_cut = cutlass.Int32(1)
                    if claimed_c <= bs_l:
                        if have == cutlass.Int32(0):
                            cut_t = seed_thr_row[0]
                            cut_n = claimed_c
                        anch_t = seed_thr_row[0]
                        have = cutlass.Int32(1)
                        line_cut = cutlass.Int32(1)
                    if have == cutlass.Int32(0):
                        # ---- clamped-histogram fallback over the mapped
                        # extents (bracket between two known lines; the
                        # all-above case takes one max pass first) ----
                        # histogram source = the bracket's own SEGMENT
                        # prefix: if the segment is full it is a value-
                        # blind (unbiased) SAMPLE of the band - scale the
                        # targets by band/segment and let the post-load
                        # count net verify; if not full it IS the exact
                        # band. Either way the scan shrinks from the
                        # whole list to <= one segment.
                        b_lo = seed_thr_row[0]
                        b_hi = seed_thr_row[1]
                        base_c = n1_c
                        hs_base = cutlass.Int32(2 * segA)  # segment C
                        hs_len = lenC
                        hs_band = claimed_c - n1_c
                        if n1_c > bs_l:
                            b_lo = seed_thr_row[1]
                            b_hi = seed_thr_row[2]
                            base_c = n2_c
                            hs_base = cutlass.Int32(segA)  # segment B
                            hs_len = lenB
                            hs_band = n1_c - n2_c
                        need_max = cutlass.Int32(0)
                        if n2_c > bs_l:
                            b_lo = seed_thr_row[2]
                            base_c = cutlass.Int32(0)
                            hs_base = cutlass.Int32(0)  # segment A
                            hs_len = lenA
                            hs_band = n2_c
                            need_max = cutlass.Int32(1)
                        if b_hi >= cutlass.Float32(1e29):
                            # parked upper line: bracket by the segment max
                            need_max = cutlass.Int32(1)
                        if hs_band < cutlass.Int32(1):
                            hs_band = cutlass.Int32(1)
                        samp_f = (cutlass.Float32(1.0) * hs_len) / hs_band
                        if need_max == cutlass.Int32(1):
                            lmax = cutlass.Float32(self.NEG_FLT_MAX)
                            i_m = tidx
                            while i_m < hs_len:
                                for _ju in cutlass.range_constexpr(4):
                                    j_m = i_m + cutlass.Int32(_ju * num_threads)
                                    if j_m < hs_len:
                                        src_m = hs_base + j_m
                                        vp_m = cute.make_ptr(
                                            cutlass.Float32,
                                            vbase + cutlass.Int64(src_m) * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        lmax = cute.arch.fmax(
                                            lmax, cute.make_tensor(vp_m, cute.make_layout((1,)))[0]
                                        )
                                i_m = i_m + cutlass.Int32(4 * num_threads)
                            wmax_l = self.warp_reduce_max_f32(lmax)
                            if lane == cutlass.Int32(0):
                                smem_wmax[warp_id] = wmax_l
                            cute.arch.barrier()
                            vmax_l = cutlass.Float32(self.NEG_FLT_MAX)
                            for _wr in cutlass.range_constexpr(self.num_warps):
                                vmax_l = cute.arch.fmax(vmax_l, smem_wmax[_wr])
                            b_hi = vmax_l + cutlass.Float32(1e-3)
                        NBL = cutlass.const_expr(self.kNumBins)
                        r_lo = b_lo
                        r_w = (b_hi - b_lo) / cutlass.Float32(NBL)
                        if r_w <= cutlass.Float32(0.0):
                            r_w = cutlass.Float32(1e-6)
                        # sample-unit targets (population targets scaled
                        # by segment/band); the descend base stays in
                        # sample units too - the post-load exact-count
                        # net absorbs the sampling error.
                        # fire target = 1.25x the K-need (headroom over
                        # the sampling noise)
                        kneedS = cutlass.Int32(
                            (cutlass.Float32(1.25) * (kK_l - base_c)) * samp_f
                            + cutlass.Float32(0.5)
                        )
                        if kneedS < cutlass.Int32(1):
                            kneedS = cutlass.Int32(1)
                        kfitS = cutlass.Int32((cutlass.Float32(1.0) * (bs_l - base_c)) * samp_f)
                        sbase = cutlass.Int32(0)
                        searching = cutlass.Int32(1)
                        for _rd in cutlass.range_constexpr(3):
                            if searching == cutlass.Int32(1):
                                jz_l = tidx
                                while jz_l < cutlass.Int32(NBL):
                                    smem_hist[jz_l] = cutlass.Int32(0)
                                    jz_l = jz_l + cutlass.Int32(num_threads)
                                if tidx == cutlass.Int32(0):
                                    s_iscalars[2] = cutlass.Int32(0)
                                cute.arch.barrier()
                                inv_wr = cutlass.Float32(1.0) / r_w
                                r_hi = r_lo + r_w * cutlass.Float32(NBL)
                                i_h = tidx
                                while i_h < hs_len:
                                    for _ju in cutlass.range_constexpr(4):
                                        j_h = i_h + cutlass.Int32(_ju * num_threads)
                                        if j_h < hs_len:
                                            src_h = hs_base + j_h
                                            vp_h = cute.make_ptr(
                                                cutlass.Float32,
                                                vbase + cutlass.Int64(src_h) * cutlass.Int64(4),
                                                cute.AddressSpace.gmem,
                                                assumed_align=4,
                                            )
                                            v_h = cute.make_tensor(vp_h, cute.make_layout((1,)))[0]
                                            if v_h >= r_lo and v_h < r_hi:
                                                b_h = cutlass.Int32((v_h - r_lo) * inv_wr)
                                                if b_h < cutlass.Int32(0):
                                                    b_h = cutlass.Int32(0)
                                                if b_h > cutlass.Int32(NBL - 1):
                                                    b_h = cutlass.Int32(NBL - 1)
                                                atomicAdd(
                                                    smem_hist.iterator + b_h, cutlass.Int32(1)
                                                )
                                    i_h = i_h + cutlass.Int32(4 * num_threads)
                                cute.arch.barrier()
                                if warp_id == cutlass.Int32(0):
                                    SEGL = cutlass.const_expr(NBL // self.WARP_SIZE)
                                    top_l = cutlass.Int32(NBL - 1) - lane * cutlass.Int32(SEGL)
                                    seg_l = cute.make_rmem_tensor((SEGL,), cutlass.Int32)
                                    part_l = cutlass.Int32(0)
                                    for _js in cutlass.range_constexpr(SEGL):
                                        v8_l = smem_hist[top_l - cutlass.Int32(_js)]
                                        seg_l[_js] = v8_l
                                        part_l = part_l + v8_l
                                    tp_l = part_l
                                    for _os in cutlass.range_constexpr(5):
                                        ov_l = cutlass.const_expr(1 << _os)
                                        oth_l = cute.arch.shuffle_sync_up(
                                            tp_l, ov_l, mask_and_clamp=0
                                        )
                                        if lane >= cutlass.Int32(ov_l):
                                            tp_l = tp_l + oth_l
                                    excl_l = tp_l - part_l
                                    kneed = kneedS - sbase
                                    kfit = kfitS - sbase
                                    run_l = cutlass.Int32(0)
                                    for _js in cutlass.range_constexpr(SEGL):
                                        run_l = run_l + seg_l[_js]
                                        cum_at = excl_l + run_l
                                        cum_bef = cum_at - seg_l[_js]
                                        if cum_bef < kneed and cum_at >= kneed:
                                            s_iscalars[3] = top_l - cutlass.Int32(_js)
                                            smem_wcnt[0] = cum_bef
                                            if cum_at <= kfit:
                                                s_iscalars[2] = cutlass.Int32(1)
                                            else:
                                                s_iscalars[2] = cutlass.Int32(2)
                                cute.arch.barrier()
                                st_l = s_iscalars[2]
                                if st_l == cutlass.Int32(1):
                                    cut_t = r_lo + cutlass.Float32(s_iscalars[3]) * r_w
                                    anch_t = cut_t
                                    have = cutlass.Int32(1)
                                    searching = cutlass.Int32(0)
                                if st_l == cutlass.Int32(2):
                                    sbase = sbase + smem_wcnt[0]
                                    r_lo = r_lo + cutlass.Float32(s_iscalars[3]) * r_w
                                    r_w = r_w / cutlass.Float32(NBL)
                                    if r_w <= cutlass.Float32(0.0):
                                        searching = cutlass.Int32(0)
                                if st_l == cutlass.Int32(0):
                                    searching = cutlass.Int32(0)
                                cute.arch.barrier()
                        if tidx == cutlass.Int32(0):
                            s_iscalars[2] = cutlass.Int32(-1)
                            s_iscalars[3] = cutlass.Int32(-1)
                        cute.arch.barrier()
                if usable == cutlass.Int32(1) and have == cutlass.Int32(1):
                    take_cand = cutlass.Int32(1)
                    list_used = cutlass.Int32(1)
                    if tidx == cutlass.Int32(0):
                        s_iscalars[0] = cutlass.Int32(0)
                        s_iscalars[2] = cutlass.Int32(0)  # non-sentinel count
                        s_thr[0] = anch_t
                        s_iscalars[1] = cutlass.Int32(1)  # done
                    cute.arch.barrier()
                    lane_c = tidx & cutlass.Int32(self.WARP_SIZE - 1)
                    # fused P4 prologue: zero the coarse hist here and
                    # accumulate the candidate max INSIDE the cut walk
                    # (min := cut line by construction); the staging
                    # rides this walk's own end barrier.
                    izh_c = tidx
                    while izh_c < cutlass.Int32(self.kNumBins):
                        smem_hist[izh_c] = cutlass.Int32(0)
                        izh_c = izh_c + cutlass.Int32(num_threads)
                    wmax_acc = cutlass.Float32(self.NEG_FLT_MAX)
                    if line_cut == cutlass.Int32(1):
                        # ---- LINE cut: dense mapped-prefix COPY of
                        # exactly cut_n entries. No filter, no ballots,
                        # no atomics - every read is a winner candidate.
                        if tidx == cutlass.Int32(0):
                            s_iscalars[0] = cut_n
                        nreal_c = cutlass.Int32(0)
                        i_c = tidx
                        while i_c < cut_n:
                            for _ju in cutlass.range_constexpr(4):
                                j_c = i_c + cutlass.Int32(_ju * num_threads)
                                if j_c < cut_n:
                                    src_c = j_c
                                    if j_c >= lenA:
                                        src_c = cutlass.Int32(segA) + j_c - lenA
                                    if j_c >= lenA + lenB:
                                        src_c = cutlass.Int32(2 * segA) + j_c - lenA - lenB
                                    vp_c = cute.make_ptr(
                                        cutlass.Float32,
                                        vbase + cutlass.Int64(src_c) * cutlass.Int64(4),
                                        cute.AddressSpace.gmem,
                                        assumed_align=4,
                                    )
                                    pv_c = cute.make_tensor(vp_c, cute.make_layout((1,)))[0]
                                    smem_keys[j_c] = pv_c
                                    wmax_acc = cute.arch.fmax(wmax_acc, pv_c)
                                    # eager position fetch: the idx column
                                    # rides the same ILP batch as the value
                                    # read, so vals hold TRUE positions and
                                    # the post-P4 slot swap disappears
                                    ip_c = cute.make_ptr(
                                        cutlass.Int32,
                                        ibase + cutlass.Int64(src_c) * cutlass.Int64(4),
                                        cute.AddressSpace.gmem,
                                        assumed_align=4,
                                    )
                                    iv_c = cute.make_tensor(ip_c, cute.make_layout((1,)))[0]
                                    smem_vals[j_c] = iv_c
                                    # sentinel pads carry idx -1; cut_n
                                    # (= claimed n0) counts them, so the
                                    # REAL candidate count must be
                                    # re-measured during the copy
                                    if iv_c >= cutlass.Int32(0):
                                        nreal_c = nreal_c + cutlass.Int32(1)
                            i_c = i_c + cutlass.Int32(4 * num_threads)
                        wsum_c = self.warp_reduce_sum_i32(nreal_c)
                        if lane_c == cutlass.Int32(0):
                            atomicAdd(s_iscalars.iterator + cutlass.Int32(2), wsum_c)
                        wmax_w = self.warp_reduce_max_f32(wmax_acc)
                        if lane_c == cutlass.Int32(0):
                            smem_wcnt[tidx // cutlass.Int32(32)] = float_as_uint32(wmax_w)
                        cute.arch.barrier()
                        # demote when the pad-inflated claim admitted a
                        # list that holds fewer than K real candidates;
                        # the stock path below recovers exactly
                        real_c = s_iscalars[2]
                        if real_c < cutlass.Int32(self.top_k):
                            take_cand = cutlass.Int32(0)
                            list_used = cutlass.Int32(0)
                            if tidx == cutlass.Int32(0):
                                s_iscalars[1] = cutlass.Int32(0)
                            cute.arch.barrier()
                    if line_cut == cutlass.Int32(0):
                        # ---- histogram-edge cut: value-filtered mapped
                        # walk with merged-ballot claims (float edges
                        # round independently -> demote net below).
                        i_c = tidx
                        while (i_c - lane_c) < total_l:
                            pvals = []
                            pidxs = []
                            keeps = []
                            for _ju in cutlass.range_constexpr(4):
                                j_c = i_c + cutlass.Int32(_ju * num_threads)
                                pval = cutlass.Float32(self.NEG_FLT_MAX)
                                pidx = cutlass.Int32(-1)
                                keep = cutlass.Int32(0)
                                if j_c < total_l:
                                    src_c = j_c
                                    if j_c >= lenA:
                                        src_c = cutlass.Int32(segA) + j_c - lenA
                                    if j_c >= lenA + lenB:
                                        src_c = cutlass.Int32(2 * segA) + j_c - lenA - lenB
                                    vp_c = cute.make_ptr(
                                        cutlass.Float32,
                                        vbase + cutlass.Int64(src_c) * cutlass.Int64(4),
                                        cute.AddressSpace.gmem,
                                        assumed_align=4,
                                    )
                                    pval = cute.make_tensor(vp_c, cute.make_layout((1,)))[0]
                                    # eager position fetch (unconditional:
                                    # keeps the load independent of the
                                    # value compare, same ILP batch)
                                    ip_c = cute.make_ptr(
                                        cutlass.Int32,
                                        ibase + cutlass.Int64(src_c) * cutlass.Int64(4),
                                        cute.AddressSpace.gmem,
                                        assumed_align=4,
                                    )
                                    pidx = cute.make_tensor(ip_c, cute.make_layout((1,)))[0]
                                    if pval >= cut_t:
                                        keep = cutlass.Int32(1)
                                wmax_acc = cute.arch.fmax(wmax_acc, pval)
                                pvals.append(pval)
                                pidxs.append(pidx)
                                keeps.append(keep)
                            m0 = cute.arch.vote_ballot_sync(keeps[0] != cutlass.Int32(0))
                            m1 = cute.arch.vote_ballot_sync(keeps[1] != cutlass.Int32(0))
                            m2 = cute.arch.vote_ballot_sync(keeps[2] != cutlass.Int32(0))
                            m3 = cute.arch.vote_ballot_sync(keeps[3] != cutlass.Int32(0))
                            nk = cutlass.Int32(
                                cute.arch.popc(m0)
                                + cute.arch.popc(m1)
                                + cute.arch.popc(m2)
                                + cute.arch.popc(m3)
                            )
                            bk = cutlass.Int32(0)
                            if nk > cutlass.Int32(0):
                                if lane_c == cutlass.Int32(0):
                                    bk = atomicAdd(s_iscalars.iterator + cutlass.Int32(0), nk)
                                bk = cute.arch.shuffle_sync(bk, cutlass.Int32(0))
                                lmk = (
                                    cutlass.Uint32(1) << cutlass.Uint32(lane_c)
                                ) - cutlass.Uint32(1)
                                off = bk
                                for _ju in cutlass.range_constexpr(4):
                                    mj = (
                                        m0
                                        if _ju == 0
                                        else m1
                                        if _ju == 1
                                        else m2
                                        if _ju == 2
                                        else m3
                                    )
                                    if keeps[_ju] != cutlass.Int32(0):
                                        wpos = off + cutlass.Int32(cute.arch.popc(mj & lmk))
                                        if wpos < cutlass.Int32(self.kC):
                                            smem_keys[wpos] = pvals[_ju]
                                            smem_vals[wpos] = pidxs[_ju]
                                    off = off + cutlass.Int32(cute.arch.popc(mj))
                            i_c = i_c + cutlass.Int32(4 * num_threads)
                        wmax_w2 = self.warp_reduce_max_f32(wmax_acc)
                        if lane_c == cutlass.Int32(0):
                            smem_wcnt[tidx // cutlass.Int32(32)] = float_as_uint32(wmax_w2)
                        cute.arch.barrier()
                        cnt_l = s_iscalars[0]
                        if cnt_l < cutlass.Int32(self.top_k) or cnt_l > cutlass.Int32(self.kC):
                            take_cand = cutlass.Int32(0)
                            list_used = cutlass.Int32(0)
                            if tidx == cutlass.Int32(0):
                                s_iscalars[1] = cutlass.Int32(0)
                            cute.arch.barrier()
                if cutlass.const_expr(_P4_TAIL_DBG):
                    ck1 = cute.arch.clock64()

            # ---- parked count-free take (ext counts, no list): the
            # emission already measured the admitted line's exact count,
            # so the count pass exists ONLY to build P3's placement
            # prefix. Claim-collect instead (v5 edge-cut walk shape:
            # 4-way strided reads, ballot-merged claims): ONE cold pass
            # replaces count(cold) + collect(L2-hot). P4 is candidate-
            # order agnostic (the v5 list path feeds it emission-claim
            # order already). The claim total re-measures the count; a
            # mismatch with the parked band (stale host state) leaves
            # take_cand=0 and the stock single-count path below recovers.
            if cutlass.const_expr(
                self.use_ext_counts
                and not self.use_ext_cand
                and not self.self_scan
                and not self.enable_block_skip
                and cluster_size == 1
                and self.dtype == cutlass.Float32
            ):
                if ext_row == cutlass.Int32(1) and N < cutlass.Int32(16384):
                    # short rows only (< 16k): fused claim-collect;
                    # longer rows keep the two-pass count-then-place path
                    cut_p = s_thr[0]  # parked line (staged at P1-init)
                    rbase = input_row.iterator.toint()
                    vw_p = cutlass.const_expr(self.vec_bits // self.dtype.width)
                    va_p = cutlass.const_expr(self.vec_align_bytes)
                    eb_p = cutlass.const_expr(self.dtype.width // 8)
                    cp_atom = self._make_load_copy_atom()
                    frag0 = cute.make_rmem_tensor((vw_p,), self.dtype)
                    frag1 = cute.make_rmem_tensor((vw_p,), self.dtype)
                    frag2 = cute.make_rmem_tensor((vw_p,), self.dtype)
                    frag3 = cute.make_rmem_tensor((vw_p,), self.dtype)
                    step4_p = cutlass.const_expr(4 * num_threads * vw_p)
                    nfull_p = (N // cutlass.Int32(step4_p)) * cutlass.Int32(step4_p)
                    it_p = tidx * cutlass.Int32(vw_p)
                    # 4 chunks in flight per iter (the count primitive's
                    # ILP shape); per-element DIRECT smem atomics for
                    # passers - no warp sync, claims hide under the reads
                    while it_p < nfull_p:
                        for _jf in cutlass.range_constexpr(4):
                            gp_p = cute.make_ptr(
                                self.dtype,
                                rbase
                                + cutlass.Int64(it_p + cutlass.Int32(_jf * num_threads * vw_p))
                                * cutlass.Int64(eb_p),
                                cute.AddressSpace.gmem,
                                assumed_align=va_p,
                            )
                            cute.copy(
                                cp_atom,
                                cute.make_tensor(gp_p, cute.make_layout((vw_p,))),
                                frag0
                                if _jf == 0
                                else frag1
                                if _jf == 1
                                else frag2
                                if _jf == 2
                                else frag3,
                            )
                        for _jf in cutlass.range_constexpr(4):
                            for _jv in cutlass.range_constexpr(vw_p):
                                v_p = cutlass.Float32(
                                    (
                                        frag0
                                        if _jf == 0
                                        else frag1
                                        if _jf == 1
                                        else frag2
                                        if _jf == 2
                                        else frag3
                                    )[_jv]
                                )
                                if v_p >= cut_p:
                                    sl_p = atomicAdd(
                                        s_iscalars.iterator + cutlass.Int32(0),
                                        cutlass.Int32(1),
                                    )
                                    if sl_p < cutlass.Int32(self.kC):
                                        smem_keys[sl_p] = v_p
                                        smem_vals[sl_p] = (
                                            it_p
                                            + cutlass.Int32(_jf * num_threads * vw_p)
                                            + cutlass.Int32(_jv)
                                        )
                        it_p = it_p + cutlass.Int32(step4_p)
                    i_p = nfull_p + tidx
                    while i_p < N:
                        vp_p = cute.make_ptr(
                            cutlass.Float32,
                            rbase + cutlass.Int64(i_p) * cutlass.Int64(4),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        pval = cute.make_tensor(vp_p, cute.make_layout((1,)))[0]
                        if pval >= cut_p:
                            sl_p = atomicAdd(
                                s_iscalars.iterator + cutlass.Int32(0),
                                cutlass.Int32(1),
                            )
                            if sl_p < cutlass.Int32(self.kC):
                                smem_keys[sl_p] = pval
                                smem_vals[sl_p] = i_p
                        i_p = i_p + cutlass.Int32(num_threads)
                    cute.arch.barrier()
                    cnt_p = s_iscalars[0]
                    if cnt_p >= cutlass.Int32(self.top_k) and cnt_p <= cutlass.Int32(self.kC):
                        take_cand = cutlass.Int32(1)
                        if tidx == cutlass.Int32(0):
                            s_iscalars[1] = cutlass.Int32(1)  # done
                        cute.arch.barrier()
                    if take_cand == cutlass.Int32(0):
                        # stale host counts: reset the claim counter so
                        # the stock single-count path re-measures cleanly
                        if tidx == cutlass.Int32(0):
                            s_iscalars[0] = cutlass.Int32(0)
                        cute.arch.barrier()

            # ---- self_scan take: cut straight from the phase-0 cursors ----
            # Same admission state machine as the v5 list (tightest line
            # whose count fits [K, B*] wins; anchor = loosest in-band
            # line), but the candidates already LIVE in smem: a line cut
            # is a same-buffer run compaction (sources at >= segA,
            # destinations below it - disjoint) plus the segment-coordinate
            # fill of smem_vals that the unchanged P4 / tail repair /
            # deferred position gather consume. Straddle / overshoot rows
            # (no line in band) fall through to the stock fallback (v1).
            if cutlass.const_expr(
                self.self_scan and cluster_size == 1 and self.dtype == cutlass.Float32
            ):
                segA_f = cutlass.const_expr(self.accept_cap)
                if cutlass.const_expr(_P4_TAIL_DBG):
                    ck0 = cute.arch.clock64()
                    ck1 = ck0
                n0_f = smem_wcnt_p1[4]
                n1_f = smem_wcnt_p1[5]
                n2_f = smem_wcnt_p1[6]
                kK_f = cutlass.Int32(self.top_k)
                bs_f = cutlass.Int32(segA_f)
                cut_n = cutlass.Int32(0)
                have_f = cutlass.Int32(0)
                anch_f = cutlass.Float32(0.0)
                if ext_row == cutlass.Int32(1):
                    if n2_f >= kK_f and n2_f <= bs_f:
                        if have_f == cutlass.Int32(0):
                            cut_n = n2_f
                        anch_f = seed_thr_row[2]
                        have_f = cutlass.Int32(1)
                    if n1_f >= kK_f and n1_f <= bs_f:
                        if have_f == cutlass.Int32(0):
                            cut_n = n1_f
                        anch_f = seed_thr_row[1]
                        have_f = cutlass.Int32(1)
                    if n0_f <= bs_f:
                        if have_f == cutlass.Int32(0):
                            cut_n = n0_f
                        anch_f = seed_thr_row[0]
                        have_f = cutlass.Int32(1)
                if have_f == cutlass.Int32(1):
                    take_cand = cutlass.Int32(1)
                    list_used = cutlass.Int32(1)
                    if tidx == cutlass.Int32(0):
                        s_iscalars[0] = cut_n
                        s_thr[0] = anch_f
                        s_iscalars[1] = cutlass.Int32(1)  # done
                    cute.arch.barrier()
                    # line cut => no segment spilled (n_cut <= B* bounds
                    # every tighter count too) => lenA = n2, lenB = n1-n2
                    j_f = tidx
                    while j_f < cut_n:
                        for _jw in cutlass.range_constexpr(4):
                            q_f = j_f + cutlass.Int32(_jw * num_threads)
                            if q_f < cut_n:
                                src_f = q_f
                                if q_f >= n2_f:
                                    src_f = cutlass.Int32(segA_f) + q_f - n2_f
                                if q_f >= n1_f:
                                    src_f = cutlass.Int32(2 * segA_f) + q_f - n1_f
                                if src_f != q_f:
                                    smem_keys[q_f] = smem_keys[src_f]
                                smem_vals[q_f] = src_f
                        j_f = j_f + cutlass.Int32(4 * num_threads)
                    cute.arch.barrier()
                if cutlass.const_expr(_P4_TAIL_DBG):
                    ck1 = cute.arch.clock64()

            if cutlass.const_expr(self.use_ext_cand or self.use_ext_counts or self.self_scan):
                run_stock_p23 = take_cand == cutlass.Int32(0)
            if run_stock_p23:
                # Stage this CTA's slice into SMEM once before Phase 2's
                # 6-10 secant iters re-scan it. Phase 1 (preIdx) uses
                # scatter-loads OUTSIDE this slice, so it stays on GMEM.
                if cutlass.const_expr(self.enable_smem_cache):
                    self.load_slice_to_smem(
                        input_row,
                        slice_start,
                        slice_end,
                        smem_input,
                        tidx,
                    )

                # ---- Phase 2: R0 histogram-ladder admission (single-CTA fast
                # path) or the secant threshold search ----
                # enable_r0 gates to cluster_size==1: R0 scans the full row in
                # one CTA; cs>1 keeps the secant path.
                if cutlass.const_expr(self.enable_r0):
                    # P1b rung placement -> ONE M-ary R0 count pass -> accept the
                    # tightest rung with count in [K, kC]. On a miss, fall back to
                    # the inline log-falsi R1 shot / fb_fix refine. At cs>1 each
                    # CTA scans its slice and block_count_ge_multi cluster-merges
                    # the rung counts (phase1b rungs are per-CTA identical since
                    # preIdx stats are full-row).
                    if cutlass.const_expr(self.use_ext_counts):
                        if ext_row == cutlass.Int32(1):
                            # ---- Waterfall L1 admission (ext rungs) ----
                            # Rung thresholds arrive from the indexer epilogue,
                            # so only P1b is skipped: the stock M-ary count pass
                            # runs on the ext rungs and the block-skip list
                            # build, rung tightening, hand-off and classify all
                            # compose unchanged. When an ext count already lies
                            # in [K, kC] the admitted threshold is parked in all
                            # rung slots, degenerating the pass to one compact
                            # single-threshold count; a full miss keeps the
                            # three distinct rungs as brackets for the refine.
                            # Parking and staging happen in the P1-init thread0
                            # block, one barrier for the whole prologue.
                            if cutlass.const_expr(not self.enable_block_skip):
                                # parked admission: count the parked threshold
                                # ONCE with the refine primitive - same
                                # per-thread ptcnt cache and cluster merge P3
                                # consumes - and accept in place.
                                if s_r0col[0] == cutlass.Int32(self.M_qf):
                                    self.block_count_ge(
                                        input_row,
                                        slice_start,
                                        slice_end,
                                        s_thr[0],
                                        smem_ptcnt,
                                        smem_wcnt,
                                        s_iscalars,
                                        s_cluster_partial,
                                        tidx,
                                        warp_id,
                                        lane,
                                        do_cluster_sync=do_cluster_sync,
                                        smem_input=smem_input,
                                    )
                                    cute.arch.barrier()
                                    if tidx == cutlass.Int32(0):
                                        cpar = s_iscalars[0]
                                        if cpar >= cutlass.Int32(
                                            self.top_k
                                        ) and cpar <= cutlass.Int32(self.kC):
                                            s_iscalars[1] = cutlass.Int32(1)
                                        else:
                                            # emission counts disagree with
                                            # the measured count (stale
                                            # host state): rerun the full
                                            # M-ary machinery on the three
                                            # distinct seed lines.
                                            s_r0col[0] = cutlass.Int32(-2)
                                            for m in cutlass.range_constexpr(
                                                cutlass.const_expr(self.M_thr)
                                            ):
                                                s_mt_thr[m] = seed_thr_row[m]
                                    cute.arch.barrier()
                        if ext_row == cutlass.Int32(0):
                            if cutlass.const_expr(self.p1b_cache):
                                # rungs from the SMEM gather-cache P1 stashed (no 2nd
                                # GMEM gather); 16-bit only.
                                self.phase1b_hspace_rungs_cached(
                                    pre_idx_count,
                                    smem_gath,
                                    smem_hist,
                                    s_thr,
                                    s_mt_thr,
                                    tidx,
                                    warp_id,
                                    lane,
                                )
                            else:
                                self.phase1b_hspace_rungs(
                                    input_row,
                                    N,
                                    pre_idx_row,
                                    pre_idx_count,
                                    pre_idx_offset,
                                    smem_hist,
                                    s_thr,
                                    s_mt_thr,
                                    tidx,
                                    warp_id,
                                    lane,
                                )
                    if cutlass.const_expr(self.ext_rungs):
                        # variant B: rung thresholds = the closed-loop seed
                        # lines verbatim; the stock multi-count measures
                        # them and the argmin admission below picks the
                        # tightest one in [K, kC]. Invalid lines fall back
                        # to the stock P1b quantile rungs (P1 stats ran on
                        # this row in that case).
                        if rungs_ok == cutlass.Int32(1):
                            if tidx == cutlass.Int32(0):
                                for m in cutlass.range_constexpr(cutlass.const_expr(self.M_thr)):
                                    s_mt_thr[m] = seed_thr_row[m]
                            cute.arch.barrier()
                        if rungs_ok == cutlass.Int32(0):
                            self.phase1b_hspace_rungs(
                                input_row,
                                N,
                                pre_idx_row,
                                pre_idx_count,
                                pre_idx_offset,
                                smem_hist,
                                s_thr,
                                s_mt_thr,
                                tidx,
                                warp_id,
                                lane,
                            )
                    if cutlass.const_expr(not (self.use_ext_counts or self.ext_rungs)):
                        if cutlass.const_expr(self.p1b_cache):
                            # rungs from the SMEM gather-cache P1 stashed (no 2nd
                            # GMEM gather); 16-bit only.
                            self.phase1b_hspace_rungs_cached(
                                pre_idx_count,
                                smem_gath,
                                smem_hist,
                                s_thr,
                                s_mt_thr,
                                tidx,
                                warp_id,
                                lane,
                            )
                        else:
                            self.phase1b_hspace_rungs(
                                input_row,
                                N,
                                pre_idx_row,
                                pre_idx_count,
                                pre_idx_offset,
                                smem_hist,
                                s_thr,
                                s_mt_thr,
                                tidx,
                                warp_id,
                                lane,
                            )
                    r0_par = cutlass.Int32(0)
                    run_mary = True  # Python bool: no scf.if without ext counts
                    if cutlass.const_expr(self.use_ext_counts and not self.enable_block_skip):
                        # single-column fast path accepted: the parked count
                        # is done and admitted; the M-ary pass, argmin,
                        # handoff and miss machinery all stand down
                        # (s_r0col == M_qf skips the copy and the refine).
                        if s_r0col[0] == cutlass.Int32(self.M_qf) and s_iscalars[
                            1
                        ] == cutlass.Int32(1):
                            r0_par = cutlass.Int32(1)
                        run_mary = r0_par == cutlass.Int32(0)
                    if run_mary:
                        self.block_count_ge_multi(
                            input_row,
                            slice_start,
                            slice_end,
                            s_mt_thr,
                            smem_ptcnt_multi,
                            smem_wcnt_multi,
                            s_mt_cnt,
                            s_cluster_partial_m,
                            do_cluster_sync,
                            tidx,
                            warp_id,
                            lane,
                            smem_ptcnt=smem_ptcnt,
                            block_max_row=block_max_row,
                            smem_active=smem_active,
                            s_active_cnt=s_active_cnt,
                        )
                    cute.arch.barrier()
                    if run_mary and tidx == 0:
                        # tightest admissible rung = SMALLEST count in [K, kC]
                        # (explicit argmin: with r0_vseed the pmean column is not
                        # sorted into the rung order). Dropped rungs (block-skip
                        # rung tightening) hold PARTIAL counts — never admissible.
                        dmask_c = cutlass.Int32(0)
                        if cutlass.const_expr(self.enable_block_skip):
                            dmask_c = s_active_cnt[2]
                        best_m = cutlass.Int32(-1)
                        best_c = cutlass.Int32(2147483647)
                        for m in cutlass.range_constexpr(cutlass.const_expr(self.M_thr)):
                            cm = s_mt_cnt[m]
                            if (
                                cm >= cutlass.Int32(self.top_k)
                                and cm <= cutlass.Int32(self.kC)
                                and cm < best_c
                                and (dmask_c & (cutlass.Int32(1) << cutlass.Int32(m)))
                                == cutlass.Int32(0)
                            ):
                                best_m = cutlass.Int32(m)
                                best_c = cm
                        s_r0col[0] = best_m
                        if best_m >= cutlass.Int32(0):
                            s_thr[0] = s_mt_thr[best_m]
                            s_iscalars[0] = s_mt_cnt[best_m]
                            # done=1: the threshold is admitted, so Phase 3 must
                            # SKIP its retry-shrink and honor s_thr[0]. (block_count
                            # _ge / secant leave done via their own path; the R0
                            # admission must set it explicitly or Phase 3 re-searches
                            # and the cluster collect diverges -> wrong output.)
                            s_iscalars[1] = cutlass.Int32(1)
                            # Snapshot this CTA's LOCAL slice count for the chosen
                            # rung into s_iscalars[5] — the per-CTA cand_count that
                            # Phase 3/4's cluster gather consumes (block_count_ge
                            # sets it too; the R0 admission must match). Without it
                            # the cluster collect under-counts -> wrong output.
                            if cutlass.const_expr(cluster_size > 1):
                                s_iscalars[5] = s_cluster_partial_m[best_m]

                    cute.arch.barrier()
                    bc = s_r0col[0]
                    if bc >= cutlass.Int32(0) and bc < cutlass.Int32(self.M_qf):
                        # accepted rung column: copy its cached per-thread counts
                        # into the secant hand-off buffer (zero rescan). The vseed
                        # column (bc == M_qf) is ALREADY in smem_ptcnt (v3 reuse).
                        smem_ptcnt[tidx] = smem_ptcnt_multi[bc * cutlass.Int32(num_threads) + tidx]
                    cute.arch.barrier()
                    # ---- R0 miss: SEEDED bounded log-falsi refine ----
                    # The refine must find a threshold with count in [K, kC]
                    # between the measured rungs. SEED the loop with the rung
                    # bracket AND its known counts (clo/chi) so it does
                    # log-count regula-falsi from iter 0 with no re-measure.
                    # done=1 on accept so Phase 3 skips its retry-shrink.
                    if bc < cutlass.Int32(0):
                        if cutlass.const_expr(self.enable_block_skip):
                            if tidx == cutlass.Int32(0):
                                s_active_cnt[1] = cutlass.Int32(0)
                        if cutlass.const_expr(self.fb_fix):
                            if tidx == cutlass.Int32(0):
                                M = cutlass.const_expr(self.M_thr)
                                blo = v_lo
                                bhi = v_hi
                                clo = cutlass.Int32(-1)
                                chi = cutlass.Int32(-1)
                                dmask_f = cutlass.Int32(0)
                                if cutlass.const_expr(self.enable_block_skip):
                                    dmask_f = s_active_cnt[2]
                                for m in cutlass.range_constexpr(M):
                                    cm = s_mt_cnt[m]
                                    tm = s_mt_thr[m]
                                    m_ok = (
                                        dmask_f & (cutlass.Int32(1) << cutlass.Int32(m))
                                    ) == cutlass.Int32(0)
                                    if (
                                        m_ok
                                        and cm > cutlass.Int32(self.kC)
                                        and (clo < cutlass.Int32(0) or tm > blo)
                                    ):
                                        blo = tm
                                        clo = cm
                                    if (
                                        m_ok
                                        and cm < cutlass.Int32(self.top_k)
                                        and (chi < cutlass.Int32(0) or tm < bhi)
                                    ):
                                        bhi = tm
                                        chi = cm
                                s_thr[1] = blo
                                s_thr[2] = bhi
                                s_iscalars[2] = clo  # SEED known rung counts
                                s_iscalars[3] = chi
                                s_iscalars[1] = cutlass.Int32(0)  # done=0
                                cand = (blo + bhi) * cutlass.Float32(0.5)
                                if clo > cutlass.Int32(0) and chi >= cutlass.Int32(0):
                                    chic = chi
                                    if chic < cutlass.Int32(1):
                                        chic = cutlass.Int32(1)
                                    l_lo = cmath.log2(cutlass.Float32(clo), fastmath=True)
                                    l_hi = cmath.log2(cutlass.Float32(chic), fastmath=True)
                                    den = l_lo - l_hi
                                    if den > cutlass.Float32(0.0):
                                        t3 = (cutlass.Float32(self.log2_mstar) - l_hi) / den
                                        cnd3 = bhi + t3 * (blo - bhi)
                                        if cnd3 > blo and cnd3 < bhi:
                                            cand = cnd3
                                elif chi < cutlass.Int32(0):
                                    cand = bhi
                                elif clo < cutlass.Int32(0):
                                    cand = blo
                                s_thr[0] = cand
                            cute.arch.barrier()
                            rs = cutlass.Int32(0)
                            while rs < cutlass.Int32(8) and s_iscalars[1] == cutlass.Int32(0):
                                if rs > cutlass.Int32(0):
                                    if tidx == cutlass.Int32(0):
                                        lo3 = s_thr[1]
                                        hi3 = s_thr[2]
                                        clo3 = s_iscalars[2]
                                        chi3 = s_iscalars[3]
                                        cand = (lo3 + hi3) * cutlass.Float32(0.5)
                                        if chi3 < cutlass.Int32(0):
                                            cand = hi3
                                        elif clo3 < cutlass.Int32(0):
                                            cand = lo3
                                        else:
                                            chic = chi3
                                            if chic < cutlass.Int32(1):
                                                chic = cutlass.Int32(1)
                                            l_lo = cmath.log2(cutlass.Float32(clo3), fastmath=True)
                                            l_hi = cmath.log2(cutlass.Float32(chic), fastmath=True)
                                            den3 = l_lo - l_hi
                                            if den3 > cutlass.Float32(0.0):
                                                t3 = (
                                                    cutlass.Float32(self.log2_mstar) - l_hi
                                                ) / den3
                                                cnd3 = hi3 + t3 * (lo3 - hi3)
                                                if cnd3 > lo3 and cnd3 < hi3:
                                                    cand = cnd3
                                        s_thr[0] = cand
                                    cute.arch.barrier()
                                self.block_count_ge(
                                    input_row,
                                    slice_start,
                                    slice_end,
                                    s_thr[0],
                                    smem_ptcnt,
                                    smem_wcnt,
                                    s_iscalars,
                                    s_cluster_partial,
                                    tidx,
                                    warp_id,
                                    lane,
                                    do_cluster_sync=do_cluster_sync,
                                    smem_input=smem_input,
                                )
                                cute.arch.barrier()
                                if tidx == cutlass.Int32(0):
                                    c3 = s_iscalars[0]
                                    t3v = s_thr[0]
                                    if c3 >= cutlass.Int32(self.top_k) and c3 <= cutlass.Int32(
                                        self.kC
                                    ):
                                        s_iscalars[1] = cutlass.Int32(1)  # accept
                                    elif c3 > cutlass.Int32(self.kC):
                                        s_thr[1] = t3v
                                        s_iscalars[2] = c3
                                        if t3v >= s_thr[2]:
                                            rng3 = s_thr[2] - s_thr[1]
                                            if rng3 < cutlass.Float32(1.0):
                                                rng3 = cutlass.Float32(1.0)
                                            s_thr[2] = s_thr[2] + rng3 * cutlass.Float32(8.0)
                                            s_iscalars[3] = cutlass.Int32(-1)
                                    else:
                                        s_thr[2] = t3v
                                        s_iscalars[3] = c3
                                        if t3v <= s_thr[1]:
                                            rng3 = s_thr[2] - s_thr[1]
                                            if rng3 < cutlass.Float32(1.0):
                                                rng3 = cutlass.Float32(1.0)
                                            s_thr[1] = s_thr[1] - rng3 * cutlass.Float32(8.0)
                                            s_iscalars[2] = cutlass.Int32(-1)
                                cute.arch.barrier()
                                rs = rs + cutlass.Int32(1)
                        if s_iscalars[1] != cutlass.Int32(1):
                            # The retry budget could not land in [K, kC].
                            # ONLY the coherent undershoot-overflow corner
                            # (count(>= lo) > kC AND 0 <= count(>= hi) < K,
                            # both counts CURRENT — the retry's bracket
                            # widening marks a side stale with -1 and thus
                            # fails this guard) collapses the bracket by
                            # pure bisection to ADJACENT floats, where the
                            # plateau terminal (done = 3, threshold = hi)
                            # is exact: Phase 4 emits the sure winners and
                            # the plateau fill completes the row from the
                            # tie class. A mid-collapse count landing in
                            # [K, kC] converges normally; anything else
                            # (incl. an exhausted collapse budget) falls
                            # through to the fail-soft terminal below.
                            it4 = cutlass.Int32(0)
                            if (
                                s_iscalars[2] <= cutlass.Int32(self.kC)
                                or s_iscalars[3] < cutlass.Int32(0)
                                or s_iscalars[3] >= cutlass.Int32(self.top_k)
                            ):
                                it4 = cutlass.Int32(40)  # guard: skip collapse
                            while it4 < cutlass.Int32(40) and s_iscalars[1] == cutlass.Int32(0):
                                if tidx == cutlass.Int32(0):
                                    lo4 = s_thr[1]
                                    hi4 = s_thr[2]
                                    mid4 = (lo4 + hi4) * cutlass.Float32(0.5)
                                    if mid4 == lo4 or mid4 == hi4:
                                        s_thr[0] = hi4
                                        s_iscalars[1] = cutlass.Int32(3)
                                    else:
                                        s_thr[0] = mid4
                                cute.arch.barrier()
                                if s_iscalars[1] == cutlass.Int32(0):
                                    self.block_count_ge(
                                        input_row,
                                        slice_start,
                                        slice_end,
                                        s_thr[0],
                                        smem_ptcnt,
                                        smem_wcnt,
                                        s_iscalars,
                                        s_cluster_partial,
                                        tidx,
                                        warp_id,
                                        lane,
                                        do_cluster_sync=do_cluster_sync,
                                        smem_input=smem_input,
                                    )
                                    cute.arch.barrier()
                                    if tidx == cutlass.Int32(0):
                                        c4 = s_iscalars[0]
                                        t4 = s_thr[0]
                                        if c4 >= cutlass.Int32(self.top_k) and c4 <= cutlass.Int32(
                                            self.kC
                                        ):
                                            s_iscalars[1] = cutlass.Int32(1)
                                        elif c4 > cutlass.Int32(self.kC):
                                            s_thr[1] = t4
                                            s_iscalars[2] = c4
                                        else:
                                            s_thr[2] = t4
                                            s_iscalars[3] = c4
                                    cute.arch.barrier()
                                it4 = it4 + cutlass.Int32(1)
                            if s_iscalars[1] == cutlass.Int32(3):
                                # recount at the terminal threshold so P3's
                                # cached per-thread counts describe the
                                # sure-winner set the fill completes.
                                self.block_count_ge(
                                    input_row,
                                    slice_start,
                                    slice_end,
                                    s_thr[0],
                                    smem_ptcnt,
                                    smem_wcnt,
                                    s_iscalars,
                                    s_cluster_partial,
                                    tidx,
                                    warp_id,
                                    lane,
                                    do_cluster_sync=do_cluster_sync,
                                    smem_input=smem_input,
                                )
                                cute.arch.barrier()
                            elif s_iscalars[1] != cutlass.Int32(1):
                                # Non-converged terminal on the leader path.
                                # This used to recount at the undershoot side
                                # and stamp done = 1, shipping a -1-padded row
                                # as a documented "non-convergence encoding" -
                                # which also hid the row from Phase 3, since
                                # done == 1 never enters the repair. Stamp
                                # done = 2 and let Phase 3's two-sided
                                # bisection own it; the recount is dropped
                                # because that bisection measures anyway.
                                if tidx == cutlass.Int32(0):
                                    s_thr[0] = s_thr[2]
                                    s_iscalars[1] = cutlass.Int32(2)
                                cute.arch.barrier()
                        else:
                            self.phase2_secant_search(
                                input_row,
                                N,
                                slice_start,
                                slice_end,
                                smem_ptcnt,
                                smem_wcnt,
                                s_thr,
                                s_iscalars,
                                s_cluster_partial,
                                tidx,
                                warp_id,
                                lane,
                                do_cluster_sync=do_cluster_sync,
                                smem_input=smem_input,
                            )
                else:
                    self.phase2_secant_search(
                        input_row,
                        N,
                        slice_start,
                        slice_end,
                        smem_ptcnt,
                        smem_wcnt,
                        s_thr,
                        s_iscalars,
                        s_cluster_partial,
                        tidx,
                        warp_id,
                        lane,
                        do_cluster_sync=do_cluster_sync,
                        smem_input=smem_input,
                    )

                # Cluster handoff #1 (end of Phase 2). Skipped when
                # do_cluster_sync is False (cs=1 or short-row degrade).
                if cutlass.const_expr(cluster_size > 1):
                    if do_cluster_sync:
                        cute.arch.cluster_arrive_relaxed()
                        cute.arch.cluster_wait()

                # ---- Phase 3: cluster-parallel candidate collect ----
                self.phase3_collect_candidates(
                    input_row,
                    N,
                    slice_start,
                    slice_end,
                    smem_keys,
                    smem_vals,
                    smem_ptcnt,
                    smem_wcnt,
                    s_thr,
                    s_iscalars,
                    s_cluster_partial,
                    tidx,
                    warp_id,
                    lane,
                    do_cluster_sync=do_cluster_sync,
                    smem_input=smem_input,
                    smem_active=smem_active,
                    s_active_cnt=s_active_cnt,
                )

            # Cluster handoff #2: leader's DSMEM gather of peer
            # smem_keys/smem_vals. Skipped at do_cluster_sync=False.
            if cutlass.const_expr(cluster_size > 1):
                if do_cluster_sync:
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

            # Phase 4 runs on the leader only. const_expr (compile-
            # time eliminated) split from runtime so cs=1 gets a flat
            # code path with no leader/sync checks.
            # Pre-init cand_count_p4 so CuTe DSL sees a stable Int32 type
            # across the runtime ``if is_leader:`` branch in cs>1 mode
            # (DSL forbids first-assigning a variable inside a dynamic if).
            ck2 = cutlass.Int64(0)
            if cutlass.const_expr(_P4_TAIL_DBG):
                ck2 = cute.arch.clock64()
            cand_count_p4 = cutlass.Int32(0)
            if cutlass.const_expr(cluster_size == 1):
                # cs=1: the single CTA per row IS the leader.
                # Capture the P2 terminal BEFORE Phase 4: P4 reuses
                # s_iscalars[1] as radix scratch.
                if tidx == cutlass.Int32(0):
                    s_iscalars[6] = cutlass.Int32(0)
                    if s_iscalars[1] == cutlass.Int32(3):
                        s_iscalars[6] = cutlass.Int32(1)
                cute.arch.barrier()
                cand_count_p4 = min(s_iscalars[0], cutlass.Int32(self.kC))
                if cutlass.const_expr(self.enable_p4_rank_scatter):
                    if cutlass.const_expr(
                        self.use_ext_cand and self.use_ext_counts and self.dtype == cutlass.Float32
                    ):
                        # list rows carry a walk-staged range + pre-zeroed
                        # hist (flag = list_used; fallback rows take the
                        # stock minmax path inside)
                        self.phase4_rank_scatter(
                            smem_keys,
                            smem_vals,
                            smem_hist,
                            smem_wcnt,
                            s_thr,
                            s_iscalars,
                            output_values_row,
                            output_indices_row,
                            cand_count_p4,
                            tidx,
                            warp_id,
                            lane,
                            ext_range_flag=list_used,
                            ext_min=cut_t,
                        )
                    else:
                        self.phase4_rank_scatter(
                            smem_keys,
                            smem_vals,
                            smem_hist,
                            smem_wcnt,
                            s_thr,
                            s_iscalars,
                            output_values_row,
                            output_indices_row,
                            cand_count_p4,
                            tidx,
                            warp_id,
                            lane,
                        )
                else:
                    self.phase4_histogram_snap(
                        smem_keys,
                        smem_vals,
                        smem_hist,
                        smem_wcnt,
                        s_thr,
                        s_iscalars,
                        output_values_row,
                        output_indices_row,
                        cand_count_p4,
                        tidx,
                        warp_id,
                        lane,
                    )
                # ---- plateau fill (done == 3): complete the row from the
                # bitwise-equal plateau class. The terminal is only set on an
                # ADJACENT-FLOAT bracket, so every value in [s_thr[1], s_thr[0])
                # is bitwise-equal; Phase 4 has already emitted the
                # cnt(>= s_thr[0]) sure winners, and ANY (K - count)-subset of
                # the tie class is a valid tie-aware completion. Ticket counter
                # lives in the DEDICATED s_iscalars[7].
                if s_iscalars[6] == cutlass.Int32(1):
                    pv_lo = s_thr[1]
                    pv_hi = s_thr[0]
                    if tidx == cutlass.Int32(0):
                        # cand_count_p4 was captured BEFORE Phase 4;
                        # s_iscalars[0] is radix scratch by now (same
                        # hazard as the flag).
                        s_iscalars[7] = cand_count_p4
                    cute.arch.barrier()
                    ifp = tidx
                    while ifp < N:
                        vfp = cutlass.Float32(0.0)
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            vfp = input_row[ifp]
                        else:
                            vfp = cutlass.Float32(input_row[ifp])
                        if vfp >= pv_lo and vfp < pv_hi:
                            pfill = atomicAdd(
                                s_iscalars.iterator + cutlass.Int32(7), cutlass.Int32(1)
                            )
                            if pfill < cutlass.Int32(self.top_k):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pfill] = self.dtype(vfp)
                                output_indices_row[pfill] = ifp
                        ifp = ifp + cutlass.Int32(self.num_threads)
                    cute.arch.barrier()
                ck_sw0 = cutlass.Int64(0)
                ck_sw1 = cutlass.Int64(0)
                if cutlass.const_expr(_P4_SUB_DBG):
                    ck_sw0 = cute.arch.clock64()
                if cutlass.const_expr(
                    self.self_scan and self.use_ext_counts and self.dtype == cutlass.Float32
                ):
                    # self_scan rows: the compact stored SEGMENT COORDS in
                    # the vals slots. Swap them for true positions with K
                    # fully-parallel gathers. Must precede the xstate
                    # publish, which reads output slot K-1 as a position.
                    # (ext_cand list rows translate EAGERLY in the take
                    # walk - the idx column rides the value ILP batch -
                    # so they never reach this loop.)
                    if list_used == cutlass.Int32(1):
                        io_r = tidx
                        while io_r < cutlass.Int32(self.top_k):
                            li_r = output_indices_row[io_r]
                            # slots are segmented offsets (may exceed the
                            # entry count); only sentinel -1 is invalid
                            if li_r >= cutlass.Int32(0):
                                ip_r = cute.make_ptr(
                                    cutlass.Int32,
                                    cand_idx_row.iterator.toint()
                                    + cutlass.Int64(li_r) * cutlass.Int64(4),
                                    cute.AddressSpace.gmem,
                                    assumed_align=4,
                                )
                                output_indices_row[io_r] = cute.make_tensor(
                                    ip_r, cute.make_layout((1,))
                                )[0]
                            io_r = io_r + cutlass.Int32(num_threads)
                        cute.arch.barrier()
                if cutlass.const_expr(_P4_SUB_DBG):
                    ck_sw1 = cute.arch.clock64()
                if cutlass.const_expr(self.emit_xstate):
                    # Closed-loop state (interface v2): [0] valid, [1] kth
                    # proxy (= accepted threshold; the tie-fill makes it a
                    # tight lower bound of the true kth), [2] accepted
                    # threshold, [3] cand_count. The next step derives its
                    # seed rung group from these.
                    if list_used == cutlass.Int32(1):
                        # the anchor below reads output_indices_row[K-1],
                        # written by peer threads in rank-scatter / tail
                        # repair; not every exit of that phase ends in a
                        # block barrier (the eager-position path dropped
                        # the swap loop's trailing one), so publish
                        # visibility explicitly. list_used is uniform
                        # (admission is decided from shared control
                        # words), so the barrier is block-safe.
                        cute.arch.barrier()
                    if tidx == cutlass.Int32(0):
                        xstate_row[0] = cutlass.Float32(1.0)
                        thr_pub = s_thr[0]
                        anch_pub = s_thr[0]
                        if list_used == cutlass.Int32(1):
                            # list rows: rank-scatter's output is rank-
                            # ordered, so slot K-1 holds the exact k-th
                            # boundary - a tighter, healthier closed-loop
                            # anchor than the loose collect line.
                            idx_k = output_indices_row[cutlass.Int32(self.top_k - 1)]
                            if idx_k >= cutlass.Int32(0) and idx_k < N:
                                thr_pub = cutlass.Float32(input_row[idx_k])
                        xstate_row[1] = thr_pub
                        xstate_row[2] = anch_pub
                        if cutlass.const_expr(_P4_TAIL_DBG):
                            ck3 = cute.arch.clock64()
                            # [1] device total (entry->publish), [2] true
                            # in-kernel prologue (entry->walk start): wall
                            # minus [1] = host/launch, NOT kernel work
                            xstate_row[1] = cutlass.Float32(cutlass.Int32(ck3 - ckE))
                            xstate_row[2] = cutlass.Float32(cutlass.Int32(ck0 - ckE))
                            xstate_row[4] = cutlass.Float32(cutlass.Int32(ck1 - ck0))  # walk+flags
                            xstate_row[5] = cutlass.Float32(cutlass.Int32(ck2 - ck1))  # P2/P3 gap
                            xstate_row[6] = cutlass.Float32(cutlass.Int32(ck3 - ck2))  # Phase 4
                            xstate_row[7] = s_thr[1]  # cnt_strad
                        if cutlass.const_expr(_P4_SUB_DBG):
                            xstate_row[2] = cutlass.Float32(smem_wcnt_p1[7])
                        if cutlass.const_expr(_P4_SUB_DBG):
                            # P4 sub-phase cycles staged by rank_scatter.
                            # Chain-safe layout: [2] (closed-loop anchor)
                            # untouched; [1] cnt_strad (tail class size),
                            # [4] fine, [5] scatter, [6] tail, [7] deferred-
                            # position swap. The small C-predictable phases
                            # (minmax/hist/coarse, wcnt[8..10]) are not
                            # published.
                            xstate_row[1] = s_thr[1]
                            if cutlass.const_expr(_P4_SUB_HEAD):
                                xstate_row[4] = cutlass.Float32(smem_wcnt[8])
                                xstate_row[5] = cutlass.Float32(smem_wcnt[9])
                                xstate_row[6] = cutlass.Float32(smem_wcnt[10])
                                xstate_row[7] = cutlass.Float32(smem_wcnt[11])
                            else:
                                xstate_row[4] = cutlass.Float32(smem_wcnt[11])
                                xstate_row[5] = cutlass.Float32(smem_wcnt[12])
                                xstate_row[6] = cutlass.Float32(smem_wcnt[13])
                                xstate_row[7] = cutlass.Float32(cutlass.Int32(ck_sw1 - ck_sw0))
                        # cand_count_p4 = pre-P4 snapshot (P4 repurposes
                        # the s_iscalars slots).
                        xstate_row[3] = cutlass.Float32(cand_count_p4)
                        if cutlass.const_expr(
                            self.ext_rungs and not _P4_SUB_DBG and not _P4_TAIL_DBG
                        ):
                            # closed-loop food: the three rung counts this
                            # step measured (exact, straight from the R0
                            # multi-count) - the host derives next-step
                            # lines from these instead of re-counting.
                            xstate_row[4] = cutlass.Float32(s_mt_cnt[0])
                            xstate_row[5] = cutlass.Float32(s_mt_cnt[1])
                            xstate_row[6] = cutlass.Float32(s_mt_cnt[2])
                        if cutlass.const_expr(_SKIP_DBG and self.enable_block_skip):
                            xstate_row[4] = cutlass.Float32(s_active_cnt[0])
                            xstate_row[5] = cutlass.Float32(s_active_cnt[1])
                            xstate_row[6] = cutlass.Float32(s_r0col[0])
                            xstate_row[7] = cutlass.Float32(s_active_cnt[2])
            else:
                # cs>1: only the leader (CTA 0 in cluster) runs Phase 4.
                if is_leader:
                    if do_cluster_sync:
                        # DSMEM-gather peer candidates into the leader's
                        # smem_keys/smem_vals. Layout: leader's chunk goes
                        # to [0 .. leader_local_cnt); each peer r's chunk
                        # appends the next peer_r_local_cnt entries.
                        local_cnt_self = s_iscalars[5]
                        local_iscalars_ptr = s_iscalars.iterator + cutlass.Int32(5)
                        smem_keys_iter = smem_keys.iterator
                        smem_vals_iter = smem_vals.iterator
                        base_offset = local_cnt_self
                        for peer in cutlass.range_constexpr(1, cluster_size):
                            peer_iscalars_addr = mapa_shared_cluster(
                                local_iscalars_ptr, cutlass.Int32(peer)
                            )
                            peer_cnt = ld_shared_cluster_i32(peer_iscalars_addr)
                            # Cap to kC (defense-in-depth vs. the
                            # done==2 bracket-exhaustion path).
                            peer_cnt = min(peer_cnt, cutlass.Int32(self.kC))
                            i_gather = tidx
                            while i_gather < peer_cnt:
                                peer_key_addr = mapa_shared_cluster(
                                    smem_keys_iter + i_gather, cutlass.Int32(peer)
                                )
                                peer_val_addr = mapa_shared_cluster(
                                    smem_vals_iter + i_gather, cutlass.Int32(peer)
                                )
                                k_val = ld_shared_cluster_f32(peer_key_addr)
                                v_val = ld_shared_cluster_i32(peer_val_addr)
                                dst = base_offset + i_gather
                                if dst < cutlass.Int32(self.kC):
                                    smem_keys[dst] = k_val
                                    smem_vals[dst] = v_val
                                i_gather = i_gather + cutlass.Int32(num_threads)
                            base_offset = base_offset + peer_cnt
                        # Reset s_iscalars[0] to cluster-wide cand_count.
                        if tidx == cutlass.Int32(0):
                            s_iscalars[0] = base_offset
                        cute.arch.barrier()
                    # else: short-row degrade — leader (CTA 0) already
                    # holds the full row's candidates in its own
                    # smem_keys/smem_vals (no peers to gather from).

                    # ---- Phase 4: histogram snap + writeback ----
                    # Capture the P2 terminal BEFORE Phase 4: P4
                    # reuses s_iscalars[1] as radix scratch.
                    if tidx == cutlass.Int32(0):
                        s_iscalars[6] = cutlass.Int32(0)
                        if s_iscalars[1] == cutlass.Int32(3):
                            s_iscalars[6] = cutlass.Int32(1)
                    cute.arch.barrier()
                    cand_count_p4 = min(s_iscalars[0], cutlass.Int32(self.kC))
                    if cutlass.const_expr(self.enable_p4_rank_scatter):
                        self.phase4_rank_scatter(
                            smem_keys,
                            smem_vals,
                            smem_hist,
                            smem_wcnt,
                            s_thr,
                            s_iscalars,
                            output_values_row,
                            output_indices_row,
                            cand_count_p4,
                            tidx,
                            warp_id,
                            lane,
                        )
                    else:
                        self.phase4_histogram_snap(
                            smem_keys,
                            smem_vals,
                            smem_hist,
                            smem_wcnt,
                            s_thr,
                            s_iscalars,
                            output_values_row,
                            output_indices_row,
                            cand_count_p4,
                            tidx,
                            warp_id,
                            lane,
                        )
                    if cutlass.const_expr(self.emit_xstate):
                        # closed-loop state, leader-only at cs > 1 (same
                        # layout as the cs == 1 exit).
                        if tidx == cutlass.Int32(0):
                            xstate_row[0] = cutlass.Float32(1.0)
                            xstate_row[1] = s_thr[0]
                            xstate_row[2] = s_thr[0]
                            xstate_row[3] = cutlass.Float32(cand_count_p4)
                            if cutlass.const_expr(
                                self.ext_rungs and not _P4_SUB_DBG and not _P4_TAIL_DBG
                            ):
                                # cluster-merged rung counts (identical on
                                # every CTA after the multi-count DSMEM
                                # aggregation)
                                xstate_row[4] = cutlass.Float32(s_mt_cnt[0])
                                xstate_row[5] = cutlass.Float32(s_mt_cnt[1])
                                xstate_row[6] = cutlass.Float32(s_mt_cnt[2])

                    # ---- plateau fill (done == 3): complete the row from the
                    # bitwise-equal plateau class. The terminal is only set on an
                    # ADJACENT-FLOAT bracket, so every value in [s_thr[1], s_thr[0])
                    # is bitwise-equal; Phase 4 has already emitted the
                    # cnt(>= s_thr[0]) sure winners, and ANY (K - count)-subset of
                    # the tie class is a valid tie-aware completion. Ticket counter
                    # lives in the DEDICATED s_iscalars[7].
                    if s_iscalars[6] == cutlass.Int32(1):
                        pv_lo = s_thr[1]
                        pv_hi = s_thr[0]
                        if tidx == cutlass.Int32(0):
                            # cand_count_p4 was captured BEFORE Phase 4;
                            # s_iscalars[0] is radix scratch by now (same
                            # hazard as the flag).
                            s_iscalars[7] = cand_count_p4
                        cute.arch.barrier()
                        ifp = tidx
                        while ifp < N:
                            vfp = cutlass.Float32(0.0)
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                vfp = input_row[ifp]
                            else:
                                vfp = cutlass.Float32(input_row[ifp])
                            if vfp >= pv_lo and vfp < pv_hi:
                                pfill = atomicAdd(
                                    s_iscalars.iterator + cutlass.Int32(7), cutlass.Int32(1)
                                )
                                if pfill < cutlass.Int32(self.top_k):
                                    if cutlass.const_expr(self.return_output_values):
                                        output_values_row[pfill] = self.dtype(vfp)
                                    output_indices_row[pfill] = ifp
                            ifp = ifp + cutlass.Int32(self.num_threads)
                        cute.arch.barrier()

        # Final cluster barrier: keep peer CTAs (and their SMEM) alive
        # until the leader's gather + Phase 4 finish. Skipped at
        # do_cluster_sync=False (no peers; short-row degrade non-leaders
        # already fell through ``run_one_row``).
        if cutlass.const_expr(cluster_size > 1):
            if do_cluster_sync:
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()

    # ------------------------------------------------------------------
    # Host-side launcher
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seq_lens: cute.Tensor,
        output_values: cute.Tensor,  # or None.
        output_indices: cute.Tensor,
        order_row: cute.Tensor,  # or None when seqlen_sorted=False
        stream,
        block_max: cute.Tensor = None,  # block-skip bounds; None = disabled
        seed_thr: cute.Tensor = None,  # [num_rows, 3] fp32 (ext counts)
        seed_counts: cute.Tensor = None,  # [num_rows, 3] int32 (ext counts)
        xstate: cute.Tensor = None,  # [num_rows, 8] fp32 (emit_xstate)
        cand_vals: cute.Tensor = None,  # [num_rows, CAP] fp32 (ext cand)
        cand_idx: cute.Tensor = None,  # [num_rows, CAP] int32 (ext cand)
        cand_ctl: cute.Tensor = None,  # [num_rows, 2] int32 (ext cand)
    ):
        num_rows = input_data.shape[0]
        cluster_size = cutlass.const_expr(self.cluster_size)
        # TODO: n_cols (= input_data.shape[1] = max_seq_len) is sym_int here
        # because the wrapper compiles with cute.sym_int() for n_cols. In
        # practice max_seq_len is static (from model config), so adding n_cols
        # to the wrapper cache key would allow a concrete-int fake tensor and
        # enable a real enable_smem_cache size assertion in _compile().

        # Grid = num_rows * cluster_size. Adjacent bidx in
        # [cluster_id*cs, (cluster_id+1)*cs) form one thread-block cluster
        # that owns row[cluster_id]. ``cluster=None`` at cs=1 keeps the
        # launch identical to a plain single-CTA-per-row kernel.
        total_ctas = num_rows * cluster_size
        self.gvr_topk_kernel(
            input_data,
            pre_idx,
            seq_lens,
            output_values,
            output_indices,
            order_row,
            block_max,
            seed_thr,
            seed_counts,
            xstate,
            cand_vals,
            cand_idx,
            cand_ctl,
        ).launch(
            grid=(total_ctas, 1, 1),
            block=(self.num_threads, 1, 1),
            cluster=(cluster_size, 1, 1) if cutlass.const_expr(cluster_size > 1) else None,
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )

    # ------------------------------------------------------------------ #
    #  Host-side launch-shape policy + self-contained launcher            #
    # ------------------------------------------------------------------ #
    # cluster_size / num_threads / min_blocks_per_mp / use_256bit_load are
    # compile-time ctor knobs: a compiled kernel cannot change its own grid
    # or cluster shape, so batch-size adaptation MUST happen at launch time
    # by picking a different compiled variant. ``pick_config`` is that
    # policy as a pure function colocated with the kernel (single source of
    # truth), and ``launch`` is a thin variant-cache wrapper so direct-drive
    # users (tests, benchmarks) get the same shapes production would pick.
    # The production custom op delegates here (``pick_cluster_size`` /
    # ``pick_tuning``) — one policy, two shells. Intentional shell
    # divergence: on a 32B-misaligned logits pointer the production runner
    # ASSERTS (contract violation), while ``launch`` silently downgrades to
    # 128-bit loads (dev convenience for ad-hoc tensors).

    _NUM_SMS: Optional[int] = None
    _LAUNCH_CACHE: dict = {}

    @staticmethod
    def _device_num_sms() -> int:
        if GvrTopKKernel._NUM_SMS is None:
            import torch  # local: keep the module importable without torch

            GvrTopKKernel._NUM_SMS = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        return GvrTopKKernel._NUM_SMS

    @staticmethod
    def pick_cluster_size(num_rows: int, n_row: int, num_sms: int) -> int:
        """Cluster-size policy: N < 64K -> 1 (sync unrecouped); tiny grid
        at large N -> 8; single-wave -> 4/2; multi-wave -> 1 (row
        parallelism already saturates the SMs; per-row splitting is pure
        overhead past one wave)."""
        if n_row < 65536:
            return 1
        if num_rows <= 4 and n_row >= 131072:
            return 8
        if num_rows * 4 <= num_sms:
            return 4
        if num_rows * 2 <= num_sms:
            return 2
        return 1

    @staticmethod
    def pick_tuning(
        torch_dtype,
        num_rows: int,
        n_per_cta: int,
        num_sms: int,
        graph_capture: bool,
    ) -> dict:
        """T / V / min_blocks_per_mp / warp-reduce policy at a given
        per-CTA row width (cluster split already applied).

        ``graph_capture``: raise the half-prec T=1024 bar so a small
        capture-N does not pin T=1024 onto small-N replays.
        Returns ``num_threads``, ``use_256bit_load``,
        ``min_blocks_per_mp``, ``enable_warp_parallel_reduce``.
        """
        import torch  # local: keep the module importable without torch

        is_fp32 = torch_dtype == torch.float32
        # T=1024 needs a 1 CTA/SM grid AND enough per-CTA vec work.
        n_thresh_t = 131072 if (graph_capture and not is_fp32) else 65536
        num_threads = 1024 if (num_rows <= num_sms and n_per_cta >= n_thresh_t) else 512
        # V=256-bit only helps fp32 at large N; half-prec cvt doubles reg
        # pressure. Requires a 32B-aligned contiguous tensor (see the
        # shell-divergence note above).
        use_256bit_load = is_fp32 and n_per_cta >= 16384
        enable_warp_parallel_reduce = num_threads == 1024

        # min_blocks_per_mp: reg-vs-occupancy tiers (fp32 wants ~70 regs
        # for 4-LDG ILP -> mb<=2; half-prec fits 40 regs -> mb=3 packs
        # 3 CTA/SM when rows oversubscribe the device).
        vec_bits = 256 if use_256bit_load else 128
        vec_w = vec_bits // (32 if is_fp32 else 16)
        n_vec_iters = max(1, n_per_cta // (num_threads * vec_w))
        if is_fp32:
            if n_vec_iters < 4:
                min_blocks_per_mp = 0
            elif num_rows <= num_sms:
                min_blocks_per_mp = 1
            elif num_sms * 2 < num_rows <= num_sms * 3 and n_per_cta <= 32768:
                min_blocks_per_mp = 3
            else:
                min_blocks_per_mp = 2
        else:
            if num_rows > num_sms:
                min_blocks_per_mp = 3
            elif n_vec_iters < 4:
                min_blocks_per_mp = 0
            else:
                min_blocks_per_mp = 1

        return dict(
            num_threads=num_threads,
            use_256bit_load=use_256bit_load,
            min_blocks_per_mp=min_blocks_per_mp,
            enable_warp_parallel_reduce=enable_warp_parallel_reduce,
        )

    @staticmethod
    def pick_config(
        torch_dtype,
        num_rows: int,
        num_candidates: int,
        max_seq_len: Optional[int] = None,
        num_sms: Optional[int] = None,
        has_block_max: bool = False,
    ) -> dict:
        """Pick the launch-shape ctor kwargs for ``(dtype, BS, N)``.

        Mirrors the production runner policy (cluster_size auto-pick +
        ``_pick_tuning``) so any caller instantiating the kernel directly
        gets the same shapes the custom op would use. Rationale (B200,
        nsys cold-L2, 2026-07-15 big-BS triage): a config frozen at the
        BS=1 optimum (cs = N>=65536 ? 4 : 1, T=1024, mbpm=1) is geomean
        2.27x slower (max 6.0x) than the op-bench anchor at BS in
        {64, 256, 1024}, while this policy is 0.95x (parity/better).
        Multi-CTA splitting only pays while the grid is a single wave
        (num_rows * cluster_size <= num_sms); past that, row parallelism
        already saturates the SMs and per-row splitting is pure overhead.

        ``max_seq_len``: pass the peak runtime N under CUDA-graph capture
        so the variant is picked for the replay shape, not the capture
        shape (same contract as the custom op's ``_pick_tuning``).

        Returns kwargs for ``GvrTopKKernel(...)``: ``cluster_size``,
        ``num_threads``, ``use_256bit_load``, ``min_blocks_per_mp``,
        ``enable_warp_parallel_reduce``.
        """
        if num_sms is None:
            num_sms = GvrTopKKernel._device_num_sms()
        n_row = max_seq_len if max_seq_len is not None else num_candidates
        if has_block_max and n_row >= 200_000:
            # Block-skip requires cs == 1 with a large per-CTA slice
            # (splitting shrinks each CTA's slice below the skip
            # break-even and disables rung tightening). Below 200k the
            # wrapper drops block_max anyway (skip_min_n gate) and the
            # stock picks apply.
            cluster_size = 1
        else:
            cluster_size = GvrTopKKernel.pick_cluster_size(num_rows, n_row, num_sms)
        cfg = GvrTopKKernel.pick_tuning(
            torch_dtype,
            num_rows,
            n_row // cluster_size,
            num_sms,
            graph_capture=max_seq_len is not None,
        )
        cfg["cluster_size"] = cluster_size
        return cfg

    @classmethod
    def launch(
        cls,
        logits,
        pre_idx,
        seq_lens,
        output_indices,
        top_k: int,
        next_n: int = 1,
        compress_ratio: int = 1,
        max_seq_len: Optional[int] = None,
        num_sms: Optional[int] = None,
        **kernel_overrides,
    ) -> None:
        """Compile-and-launch with ``pick_config`` shapes (indices-only path).

        Owns a class-level compiled-variant cache keyed by every ctor knob,
        so repeated calls at any (BS, N, dtype) reuse the right variant.
        ``kernel_overrides`` (e.g. ``enable_r0=False``, ``cluster_size=8``)
        override the picked config and participate in the cache key.
        Mirrors the custom op's compile contract: sym_int shapes, tvm-ffi
        env stream (launches on the ambient torch stream), fixed
        ``return_output_values=False`` / ``seqlen_sorted=False``.
        """
        import torch  # local: keep the module importable without torch
        from cutlass.cute import runtime as _crt

        _cute_dt = {
            torch.float32: cutlass.Float32,
            torch.float16: cutlass.Float16,
            torch.bfloat16: cutlass.BFloat16,
        }
        num_rows, num_candidates = logits.shape
        cfg = cls.pick_config(
            logits.dtype, num_rows, num_candidates, max_seq_len=max_seq_len, num_sms=num_sms
        )
        cfg.update(kernel_overrides)
        if cfg["cluster_size"] > 1:
            try:
                from .single_pass_multi_cta_radix_topk_cluster import _query_max_cluster_size

                cfg["cluster_size"] = min(cfg["cluster_size"], _query_max_cluster_size())
            except ImportError:
                pass  # standalone snapshot: trust the [1, 16] ctor bound
        if cfg.get("use_256bit_load") and logits.data_ptr() % 32 != 0:
            cfg["use_256bit_load"] = False  # 256-bit vec loads need 32B alignment

        key = (logits.dtype, top_k, next_n, compress_ratio) + tuple(sorted(cfg.items()))
        compiled = cls._LAUNCH_CACHE.get(key)
        if compiled is None:
            kernel = cls(
                dtype=_cute_dt[logits.dtype],
                top_k=top_k,
                next_n=next_n,
                compress_ratio=compress_ratio,
                return_output_values=False,
                **cfg,
            )
            n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
            in_align = 32 if cfg["use_256bit_load"] else 16
            input_fake = _crt.make_fake_compact_tensor(
                kernel.dtype, (n_rows, n_cols), stride_order=(1, 0), assumed_align=in_align
            )
            pre_idx_fake = _crt.make_fake_compact_tensor(
                cutlass.Int32, (n_batch, top_k), stride_order=(1, 0), assumed_align=16
            )
            seq_lens_fake = _crt.make_fake_compact_tensor(
                cutlass.Int32, (n_batch,), stride_order=(0,)
            )
            out_indices_fake = _crt.make_fake_compact_tensor(
                cutlass.Int32, (n_rows, top_k), stride_order=(1, 0), assumed_align=16
            )
            fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
            compiled = cute.compile(
                kernel,
                input_fake,
                pre_idx_fake,
                seq_lens_fake,
                None,
                out_indices_fake,
                None,
                stream=fake_stream,
                options="--enable-tvm-ffi",
            )
            cls._LAUNCH_CACHE[key] = compiled
        compiled(logits, pre_idx, seq_lens, None, output_indices, None)


__all__ = ["GvrTopKKernel", "GvrParams"]
