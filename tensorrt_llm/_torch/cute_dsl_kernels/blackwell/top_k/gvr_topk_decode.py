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

"""GVR (Guess-Verify-Refine) heuristic Top-K kernel using cuTe DSL on Blackwell sm_100.

Supported (dtype, K): fp32/bf16/fp16 x 512/1024/2048.

TODO: could see if smem_ptcnt part and fmin/fmax vectorization could be improved.
"""

from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.utils.distributed import atomicAdd
from cutlass.utils.smem_allocator import SmemAllocator

from ..utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait
from .block_scan import warp_scan


def float_as_uint32(float_val):
    """Interpret FP32 value as uint32 bit pattern (cuTe DSL bit-cast)."""
    return llvm.bitcast(cutlass.Uint32.mlir_type, float_val.ir_value())


def _fmin_f32_inline(a, b):
    """Emit single PTX `min.f32` -> single SASS FMNMX instruction.

    cuTe DSL has cute.arch.fmax but NOT cute.arch.fmin; the canonical
    workaround `-fmax(-a, -b)` lowers to a 3-instruction sequence
    (FADD R, RZ, -a; FADD R, RZ, -b; FMNMX R, ...; FADD R, RZ, -R). This pattern
    is concentrated in block_fused_snap_iter's inner loop and accounts
    for ~8-10 us of the cuTe vs prod GVR gap at fp32 K=2048 BS=1.
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
        """Returns the per-(dtype, K, cr) specialization parameters.

        Mirrors CUDA template specialization GvrParams<T, K>. For K ∈ {512, 1024},
        cr=1 (DSv3.2) and cr=4 (DSv4, PR #14413) use different kFTarget values
        — V4's kFTarget=kK alignment eliminates upper-clamp saturation on
        tight-σ + high-A2 layers; cross-prompt swe-bench shows 1.5-2.2×
        P2-iter reduction vs V3.2's kFTarget=384/2560. K=2048 is identical
        across cr (V4 doesn't natively use K=2048; values reused for safety).
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
      P2: Secant threshold search loop (count-only)
      P3: Ballot-free candidate collect into smem keys[]/vals[]
      P4: Histogram snap (cand → exact top-K) + writeback

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
    ):
        # e.g., dtype = cutlass.Float32 / cutlass.BFloat16 / cutlass.Float16
        self.dtype = dtype
        self.top_k = top_k
        self.next_n = next_n
        # KV compression ratio of the indexer feeding this kernel:
        #   1 → DSv3.2 (no compressor); preIdxOffset = (row % next_n) + 1
        #       reflects "newest token appended" + MTP windowing.
        #   4 → DSv4 (overlap compressor); logits / preIdx live in compressed-
        #       token-index space. New compressed entries are appended at the
        #       end so prev-step indices remain valid as-is → preIdxOffset = 0.
        assert compress_ratio in (1, 4), (
            f"compress_ratio must be 1 (V3.2) or 4 (V4); got {compress_ratio}"
        )
        self.compress_ratio = compress_ratio

        self.WARP_SIZE = 32
        self.num_threads = num_threads
        self.num_warps = num_threads // self.WARP_SIZE
        # __launch_bounds__(num_threads, min_blocks_per_mp) hint to ptxas.
        # B200 sm_100: 65536 regs/SM, BS=512 → max_regs_per_thread caps at:
        #   min_blocks=1: 128  (loosest — allows ptxas to use many regs)
        #   min_blocks=2:  64
        #   min_blocks=3:  42  (current default — keeps 3 CTA/SM occupancy)
        # When num_rows ≤ #SMs (e.g., 148 on B200), grid is undersubscribed so high
        # occupancy isn't useful → set min_blocks=1 to let ptxas spend more
        # regs covering LDG latency (4-LDG back-to-back pattern survives).
        self.min_blocks_per_mp = min_blocks_per_mp
        # Vector load width for Phase 2/3 vec loops:
        #   False (default): 128-bit per LDG → LDG.E.128 (fp32: 4 elems; bf16/fp16: 8)
        #   True:            256-bit per LDG → LDG.E.256 (fp32: 8 elems; bf16/fp16: 16)
        # 256-bit halves the LDG count per byte loaded, but requires:
        #   * 32-byte aligned addresses (we pass assumed_align=32 below)
        #   * sm_90+ (Blackwell sm_100 supports it)
        #   * fragment reg usage doubles → potentially more reg pressure
        self.use_256bit_load = use_256bit_load
        self.vec_bits = 256 if use_256bit_load else 128
        self.vec_align_bytes = self.vec_bits // 8  # 32 for 256-bit, 16 for 128-bit
        # Vec-loop unrolling switches.
        #   * enable_unroll_4: 4-way fast path in block_count_ge (4 LDG.E.128
        #     in flight). Controls block_count_ge's fast path only.
        #   * enable_phase3_unroll: 4-way fast path in phase3_collect only.
        #     Independent of enable_unroll_4 because the perf trade-offs
        #     differ — phase3 has thread-local wc state and smem writes,
        #     making fast-path more expensive at large grid.
        #   * use_constant_hint: True → CopyG2ROp(invariant=True) → emits
        #       SASS LDG.E.*.CONSTANT (read-only data cache, matches CUDA
        #       __ldg path). False → CopyUniversalOp → plain LDG.E.* (no
        #       .CONSTANT modifier).
        if enable_unroll_4 is None:
            enable_unroll_4 = True
        if enable_phase3_unroll is None:
            enable_phase3_unroll = True
        self.enable_unroll_4 = enable_unroll_4
        self.enable_phase3_unroll = enable_phase3_unroll
        self.use_constant_hint = use_constant_hint
        # Replace tid==0 serial loops over num_warps with warp-parallel
        # reduce/scan in warp 0. Default policy is auto-coupled to
        # num_threads: enabled iff num_threads == 1024 (32 warps), where
        # the serial tid==0 cost is meaningful. At num_threads == 512
        # (16 warps) the warp-parallel path measured a ~2pp regression on
        # synth, so it stays off. Explicit True / False overrides the
        # policy for A/B testing.
        if enable_warp_parallel_reduce is None:
            enable_warp_parallel_reduce = num_threads == 1024
        self.enable_warp_parallel_reduce = enable_warp_parallel_reduce

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
        self.kNumBins = params.kNumBins
        self.kFTarget = params.kFTarget

        # Kernel-wide constants.
        # self.MAX_REFINE_ITERS: Phase-2 secant refine iteration cap.
        # self.FLT_MAX / self.NEG_FLT_MAX: fp32 IEEE-754 max / negative-max
        # sentinels used as reduction identities and pad values.
        self.MAX_REFINE_ITERS = 15
        self.FLT_MAX = 3.4028235e38
        self.NEG_FLT_MAX = -self.FLT_MAX

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
        s_thr,  # cute.Tensor [3] float32: [threshold, val_lo, val_hi, pmax_saved]
        s_thr_extra,  # cute.Tensor [1] float32: [pmax_saved]
        s_iscalars,  # cute.Tensor [5] int32: [cand_count, done, cnt_lo, cnt_hi, out_count]
        tidx,
        warp_id,
        lane,
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

        # Stride loop over preIdx with pre_idx_offset shift.
        # pre_idx_count is compile-time constant (top_k from JIT key).
        # Two cases:
        #   (a) pre_idx_count >= num_threads: every thread loads ≥1 preIdx;
        #       n_iters = pre_idx_count // num_threads ∈ {1, 2, 4, ...}.
        #       Fully unrolled (straight-line code).
        #   (b) pre_idx_count < num_threads (e.g. K=512, BS=1024): only first
        #       pre_idx_count threads have a preIdx; guard with `if tidx<K`.
        if cutlass.const_expr(pre_idx_count >= self.num_threads):
            n_iters = cutlass.const_expr(pre_idx_count // self.num_threads)
            for u in cutlass.range_constexpr(n_iters):
                i = tidx + cutlass.Int32(u * self.num_threads)
                raw = pre_idx_row[i]
                idx = raw + pre_idx_offset
                if idx >= 0 and idx < N:
                    v = self._load_fp32(input_row, idx)
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
            if idx >= 0 and idx < N:
                v = self._load_fp32(input_row, idx)
                local_max = cute.arch.fmax(local_max, v)
                local_min = _fmin_f32_inline(local_min, v)
                local_sum = local_sum + v
                local_cnt = local_cnt + 1

        # Warp-level reductions + smem write.
        # When pre_idx_count < num_threads (e.g. K=512 with num_threads=1024),
        # only the first `active_preidx_warps` warps have real data; remaining
        # warps would just reduce identity values + write identity to smem.
        # Skip those dummy warps to save ~30 cy/warp. Full barrier below still
        # required so all threads reach Phase 2 entry together. K ∈ {512,
        # 1024, 2048} are all multiples of WARP_SIZE so no rounding needed.
        # Clamp to num_warps so K > num_threads case (e.g. K=2048, threads=512)
        # doesn't index past smem allocation (sized to num_warps).
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

        # Block aggregate: 4 reductions over num_warps slots. Gated by
        # self.enable_warp_parallel_reduce — True picks warp-parallel reduce
        # (warp 0 lanes do 4 REDUX.SYNC instructions); False keeps tid==0
        # serial loop (default, since num_threads=512 measured a regression
        # with the warp-parallel path; expected to win at num_threads=1024).
        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            # NEW: warp-parallel 4-way reduce in warp 0. Read only the first
            # `active_preidx_warps` slots (dummy warps above wrote nothing,
            # so those smem slots are uninitialized).
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
                    s_thr[1] = pmin
                    s_thr[2] = pmax
                    s_thr_extra[0] = pmax
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
                s_thr[1] = pmin
                s_thr[2] = pmax
                s_thr_extra[0] = pmax
                s_iscalars[0] = cutlass.Int32(0)
                s_iscalars[1] = cutlass.Int32(0)
                s_iscalars[2] = cutlass.Int32(cnt_lo_seed)
                s_iscalars[3] = cutlass.Int32(1)
                s_iscalars[4] = cutlass.Int32(0)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # block_count_ge — Phase 2 / Phase 3 GE-count over global input
    # Per-thread accumulate (4-element strided), cache to smem_ptcnt[tid]
    # for P3 reuse, warp-reduce, block-reduce → s_iscalars[0] = cand_count.
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_ge(
        self,
        input_row,  # cute.Tensor [N] fp32
        N,  # length of input_row
        threshold,  # cutlass.Float32 scalar
        smem_ptcnt,  # cute.Tensor [BLOCK_SIZE] int32 (P3 cache)
        smem_wcnt,  # cute.Tensor [NUM_WARPS] int32 (block reduce scratch)
        s_iscalars,  # cute.Tensor [5] int32 (writes [0] = cand_count)
        tidx,
        warp_id,
        lane,
    ):
        """Count input[i] >= threshold across N elements.

        Vectorized: each thread loads vec_w-bit per iter (e.g., 128 bits loading 4 fp32 / 8 bf16 / 8 fp16)
        via cute.copy + CopyUniversalOp. Falls back to scalar tail for the N-mod-vec_w remainder.
        """
        num_threads = cutlass.const_expr(self.num_threads)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        c = cutlass.Int32(0)
        copy_atom = self._make_load_copy_atom()

        step_elem = cutlass.const_expr(num_threads * vec_w)

        row_addr = input_row.iterator.toint()
        n_aligned = (N // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        i = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        # Fast path: 4-way unroll for LSU-pipelining ILP.
        if self.enable_unroll_4:
            # =================================================================
            # Each loop body loads 1 vec_w chunk; LLVM unrolls 4 iters at IR
            # level. After unroll, GVN/CSE has one common base ptr in scope and
            # MAY fold the 4 derived addresses into shared base + immediate
            # offsets, e.g., loading from [base+0x2000/0x4000/0x6000]).
            # =================================================================
            rng_frag = cute.make_fragment((vec_w,), self.dtype)
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
        tail_frag = cute.make_fragment((vec_w,), self.dtype)
        while i + cutlass.Int32(vec_w - 1) < N:
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

        # Tail scalar loop
        it = n_aligned + tidx
        while it < N:
            v = self._load_fp32(input_row, it)
            if v >= threshold:
                c = c + cutlass.Int32(1)
            it = it + cutlass.Int32(num_threads)

        # Cache per-thread count for P3 retry-shrink reuse.
        smem_ptcnt[tidx] = c

        # Warp reduce + lane-0 write
        wc = self.warp_reduce_sum_i32(c)
        if lane == 0:
            smem_wcnt[warp_id] = wc
        cute.arch.barrier()

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
        smem_ptcnt,
        smem_wcnt,
        s_thr,  # [threshold, val_lo, val_hi]
        s_iscalars,  # [cand_count, done, cnt_lo, cnt_hi, out_count]
        tidx,
        warp_id,
        lane,
    ):
        """Refine smem threshold to land cand_count in [kK, kCC] window.

        Calls block_count_ge to evaluate each candidate threshold and
        updates the bracket (val_lo, val_hi, cnt_lo, cnt_hi). Marks
        s_iscalars[1] (done) = 1 on convergence, 2 on bracket exhaustion.
        """
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        kFTarget = cutlass.const_expr(self.kFTarget)

        # ---- Initial count with the Phase-1 mean as threshold ----
        # TODO: smem_ptcnt is not always needed? only for the last block_count_ge.
        # Do we have methods to reduce its write?
        thr_init = s_thr[0]
        self.block_count_ge(
            input_row,
            N,
            thr_init,
            smem_ptcnt,
            smem_wcnt,
            s_iscalars,
            tidx,
            warp_id,
            lane,
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
                    # Bracket exhausted — try midpoint, else give up.
                    nv = (vlo + vhi) * cutlass.Float32(0.5)
                    if nv == vlo or nv == vhi:
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
                    N,
                    new_thr,
                    smem_ptcnt,
                    smem_wcnt,
                    s_iscalars,
                    tidx,
                    warp_id,
                    lane,
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
        smem_keys,
        smem_vals,
        smem_ptcnt,
        smem_wcnt,
        s_thr,
        s_iscalars,
        tidx,
        warp_id,
        lane,
    ):
        """Retry-shrink (if done!=1) + warp/block prefix sum + stream-write.

        After this fn, smem_keys[0:cand_count] contains all v >= threshold
        in some order (determined by tid's per-thread index within stream-write
        loop), and smem_vals[0:cand_count] holds matching original indices.
        smem_ptcnt is reused per-thread cached counts from the LAST
        block_count_ge inside P2 (or inside the retry-shrink below).
        """
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        num_threads = cutlass.const_expr(self.num_threads)

        # ---- Retry-shrink loop (only if P2 didn't converge cleanly) ----
        if s_iscalars[1] != cutlass.Int32(1):
            # Re-count with current threshold (may already have stale cand_count)
            cur_thr = s_thr[0]
            self.block_count_ge(
                input_row,
                N,
                cur_thr,
                smem_ptcnt,
                smem_wcnt,
                s_iscalars,
                tidx,
                warp_id,
                lane,
            )
            if tidx == 0:
                if s_iscalars[0] > cutlass.Int32(kCC):
                    s_thr[1] = s_thr[0]  # val_lo = threshold
            cute.arch.barrier()

            # 10-iter retry-shrink. Runtime while with `cand_count > kCC` in the
            # loop condition.
            rs = cutlass.Int32(0)
            while rs < cutlass.Int32(10) and s_iscalars[0] > cutlass.Int32(kCC):
                if tidx == 0:
                    lo = s_thr[1]
                    hi = s_thr[2]
                    mid = (lo + hi) * cutlass.Float32(0.5)
                    if mid == lo:
                        mid = hi
                    s_thr[0] = mid
                cute.arch.barrier()
                new_thr = s_thr[0]
                self.block_count_ge(
                    input_row,
                    N,
                    new_thr,
                    smem_ptcnt,
                    smem_wcnt,
                    s_iscalars,
                    tidx,
                    warp_id,
                    lane,
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
        thr_final = s_thr[0]
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        copy_atom = self._make_load_copy_atom()
        row_addr = input_row.iterator.toint()
        step_elem = cutlass.const_expr(num_threads * vec_w)

        n_aligned = (N // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        wc = my_write_pos
        ic = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        # Phase3 unrolling: master gated by self.enable_phase3_unroll.
        # When OFF, only the tail 1-way loop runs (matches the pre-unroll
        # state of phase3_collect). When ON, the inner enable_unroll_4
        # controls the 4-way fast path.
        if self.enable_phase3_unroll:
            # Fast path: 4-way unrolled vec loop (4 loading instructions in flight).
            if self.enable_unroll_4:
                # =============================================================
                # unroll: cutlass.range(unroll=4).
                # Each body loads 1 vec_w chunk; LLVM unrolls 4 iters at IR
                # level. Same intent as the Phase-2 rewrite above.
                # =============================================================
                rng_frag = cute.make_fragment((vec_w,), self.dtype)
                big_iters = cutlass.Int32(0)
                if N > ic + cutlass.Int32(vec_w - 1):
                    big_iters = (N - ic - cutlass.Int32(vec_w)) // cutlass.Int32(
                        step_elem
                    ) + cutlass.Int32(1)

                for k in cutlass.range(big_iters, unroll=4):
                    ic_local = ic + k * cutlass.Int32(step_elem)
                    src_ptr_k = cute.make_ptr(
                        self.dtype,
                        row_addr + cutlass.Int64(ic_local) * cutlass.Int64(elem_bytes),
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
                        if vj >= thr_final and wc < cutlass.Int32(kCC):
                            smem_keys[wc] = vj
                            smem_vals[wc] = ic_local + cutlass.Int32(j)
                            wc = wc + cutlass.Int32(1)
                # Advance ic past all consumed vec_w-aligned positions.
                ic = ic + big_iters * cutlass.Int32(step_elem)

        # Tail vec loop: 1-way, handles remainder < 2*step. ic stays vec_w-
        # aligned across the unroll loop (steps by num_threads*vec_w).
        tail_frag = cute.make_fragment((vec_w,), self.dtype)
        while ic + cutlass.Int32(vec_w - 1) < N:
            src_ptr = cute.make_ptr(
                self.dtype,
                row_addr + cutlass.Int64(ic) * cutlass.Int64(elem_bytes),
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
                if vj >= thr_final and wc < cutlass.Int32(kCC):
                    smem_keys[wc] = vj
                    smem_vals[wc] = ic + cutlass.Int32(j)
                    wc = wc + cutlass.Int32(1)
            ic = ic + step

        # Tail scalar loop (N % vec_w)
        it = n_aligned + tidx
        while it < N:
            v = self._load_fp32(input_row, it)
            if v >= thr_final and wc < cutlass.Int32(kCC):
                smem_keys[wc] = v
                smem_vals[wc] = it
                wc = wc + cutlass.Int32(1)
            it = it + cutlass.Int32(num_threads)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # block_fused_snap_iter — P4 snap convergence inner step
    # ------------------------------------------------------------------
    @cute.jit
    def block_fused_snap_iter(
        self,
        smem_keys,
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
            v = smem_keys[isi]
            if v >= thr:
                lge = lge + cutlass.Int32(1)
            if v > thr:
                lgt = lgt + cutlass.Int32(1)
                # s_up = min(s_up, v) — hot path in block_fused_snap_iter (~10us)
                s_up = _fmin_f32_inline(s_up, v)
            if v < thr:
                s_down = cute.arch.fmax(s_down, v)
            isi = isi + cutlass.Int32(num_threads)

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
            # NEW: warp-parallel 3-way reduce in warp 0.
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
        """Three branches by cand_count:
        == kK : direct emit (fast path)
        >  kK : histogram k-th bin search + snap + 2-pass writeback
        <  kK : emit cand_count + pad with -self.FLT_MAX
        """
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        # ----- Branch A: cand_count == kK (fast path) -----
        if cand_count == cutlass.Int32(kK):
            i4 = tidx
            while i4 < cutlass.Int32(kK):
                output_values_row[i4] = self.dtype(smem_keys[i4])
                output_indices_row[i4] = smem_vals[i4]
                i4 = i4 + cutlass.Int32(num_threads)
        elif cand_count > cutlass.Int32(kK):
            # ----- Branch B: cand_count > kK → histogram snap -----

            # Block min/max over keys[0:cand_count]
            local_cmin = cutlass.Float32(self.FLT_MAX)
            local_cmax = cutlass.Float32(self.NEG_FLT_MAX)
            i5 = tidx
            while i5 < cand_count:
                v = smem_keys[i5]
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
            # Note: unrolled for 64 times.
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

            # Zero histogram (must zero ALL slots since smem_hist[0..NW-1] was
            # used as cmax scratch above).
            i6 = tidx
            while i6 < cutlass.Int32(kBins):
                smem_hist[i6] = cutlass.Int32(0)
                i6 = i6 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            range1 = bmax_r - bmin_r
            # inv1 = (kBins - 1 + 0.99) / range1  (range1 > 0 guaranteed by 1e-6 patch)
            inv1 = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range1

            # Build histogram by atomicAdd.
            i7 = tidx
            while i7 < cand_count:
                vk = smem_keys[i7]
                bin_f = (vk - bmin_r) * inv1
                bin_i = cutlass.Int32(bin_f)
                if bin_i < cutlass.Int32(0):
                    bin_i = cutlass.Int32(0)
                if bin_i > cutlass.Int32(kBins - 1):
                    bin_i = cutlass.Int32(kBins - 1)
                atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
                i7 = i7 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            # ---- Parallel k-th bin search (3-step) ----
            # Step 1: each warp sums BINS_PER_WARP bins (high→low slice)
            warp_bin_sum = cutlass.Int32(0)
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
                tw = cutlass.Int32(num_warps - 1)
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

            # Step 3: target warp's lane 0 scans BINS_PER_WARP bins → threshold
            # NOTE: This loop runs in a single thread (warp 0 lane 0). Tried
            # changing range_constexpr → cutlass.range(unroll=1) to mirror
            # CUDA's `for+break` (gets nvcc to keep runtime branch). SASS
            # I2FP dropped 64→1, total inst -544 for fp32, but perf was
            # WORSE (-7pp on fp32 large N, -14pp on bf16 synth). The runtime
            # branch/counter overhead in a 1-thread serial path exceeds the
            # static I2FP/FMUL/FFMA waste — those 60+ fp ops are essentially
            # free for one thread. Keeping range_constexpr.
            target_warp = s_iscalars[3]
            if warp_id == target_warp and lane == cutlass.Int32(0):
                base_cum = s_iscalars[2]
                thr_local = bmin_r
                bmin_local = bmin_r
                set_done = cutlass.Int32(0)
                for jb2 in cutlass.range_constexpr(bins_per_warp):
                    bidx2 = (
                        cutlass.Int32(kBins - 1)
                        - target_warp * cutlass.Int32(bins_per_warp)
                        - cutlass.Int32(jb2)
                    )
                    base_cum = base_cum + smem_hist[bidx2]
                    if base_cum >= cutlass.Int32(kK) and set_done == cutlass.Int32(0):
                        thr_local = bmin_local + cutlass.Float32(bidx2) * range1 / cutlass.Float32(
                            kBins
                        )
                        set_done = cutlass.Int32(1)
                s_thr[0] = thr_local
            cute.arch.barrier()

            # ---- Snap convergence loop ----
            # snap_limit = cand_count (matches CUDA heuristic_topk.cuh:985).
            # The older `cand_count / 4` bound silently accepted a non-
            # converged threshold in ~0.09% of adversarial distributions
            # (Pass 1 then picked K from cgt > kK candidates in scan order,
            # missing some true top-K members). Common path still converges
            # in 1-3 iters; the higher upper bound only affects long-tail
            # cells.
            snap_limit = cand_count

            # Loop with runtime break-via-guard. We unroll up to a safe ceiling
            # then loop the rest with a while.
            # For simplicity: while loop with explicit break-flag.
            si = cutlass.Int32(0)
            done_snap = cutlass.Int32(0)
            while si < snap_limit and done_snap == cutlass.Int32(0):
                self.block_fused_snap_iter(
                    smem_keys,
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
                if s_iscalars[3] < cutlass.Int32(kK) and s_iscalars[2] >= cutlass.Int32(kK):
                    done_snap = cutlass.Int32(1)
                si = si + cutlass.Int32(1)

            # ---- Two-pass output writeback (ballot+popc, CUDA-style) ----
            # Per-iter: ballot collects emit flags into a 32-bit mask, popc gives
            # within-warp count, lane 0 atomicAdds to out_count, shuffle broadcasts
            # the base offset. No per-iter barriers — only 1 barrier between passes.
            sel_thr = s_thr[0]
            if tidx == 0:
                s_iscalars[4] = cutlass.Int32(0)  # out_count
            cute.arch.barrier()

            # Pass 1: v > sel_thr — strided over (warp_id*WARP_SIZE, ...) like CUDA.
            # `if mask_gt != 0` mirrors CUDA heuristic_topk.cuh:1020 — when no
            # lane in the warp emits, skip the popc + atomicAdd + shuffle round
            # trip (the atomicAdd alone is ~10-30 cycles on a SMEM atomic unit).
            base_w = warp_id * cutlass.Int32(self.WARP_SIZE)
            while base_w < cand_count:
                ix1 = base_w + lane
                emit_gt = cutlass.Int32(0)
                v_p1 = cutlass.Float32(self.NEG_FLT_MAX)
                if ix1 < cand_count:
                    v_p1 = smem_keys[ix1]
                    if v_p1 > sel_thr:
                        emit_gt = cutlass.Int32(1)
                mask_gt = cute.arch.vote_ballot_sync(emit_gt != cutlass.Int32(0))
                if mask_gt != cutlass.Uint32(0):
                    cnt_gt = cutlass.Int32(cute.arch.popc(mask_gt))
                    lane_mask_gt = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
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
                        output_values_row[wpos_p1] = self.dtype(v_p1)
                        output_indices_row[wpos_p1] = smem_vals[ix1]
                base_w = base_w + cutlass.Int32(num_threads)
            cute.arch.barrier()

            # Pass 2: v == sel_thr (same pattern as Pass 1, same `if mask` guard).
            # Empty-iter case is much more common here because only tie-values
            # at the K-th rank emit.
            base_w2 = warp_id * cutlass.Int32(self.WARP_SIZE)
            while base_w2 < cand_count:
                ix2 = base_w2 + lane
                emit_eq = cutlass.Int32(0)
                v_p2 = cutlass.Float32(self.NEG_FLT_MAX)
                if ix2 < cand_count:
                    v_p2 = smem_keys[ix2]
                    if v_p2 == sel_thr:
                        emit_eq = cutlass.Int32(1)
                mask_eq = cute.arch.vote_ballot_sync(emit_eq != cutlass.Int32(0))
                if mask_eq != cutlass.Uint32(0):
                    cnt_eq = cutlass.Int32(cute.arch.popc(mask_eq))
                    lane_mask_eq = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
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
                        output_values_row[wpos_p2] = self.dtype(v_p2)
                        output_indices_row[wpos_p2] = smem_vals[ix2]
                base_w2 = base_w2 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            # Pad remainder with -self.FLT_MAX / -1
            filled_par = s_iscalars[4]
            if filled_par > cutlass.Int32(kK):
                filled_par = cutlass.Int32(kK)
            ipad = filled_par + tidx
            while ipad < cutlass.Int32(kK):
                output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                output_indices_row[ipad] = cutlass.Int32(-1)
                ipad = ipad + cutlass.Int32(num_threads)
        else:
            # ----- Branch C: cand_count < kK -----
            # Emit cand_count + pad
            i10 = tidx
            while i10 < cand_count:
                output_values_row[i10] = self.dtype(smem_keys[i10])
                output_indices_row[i10] = smem_vals[i10]
                i10 = i10 + cutlass.Int32(num_threads)
            i11 = cand_count + tidx
            while i11 < cutlass.Int32(kK):
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
    ):
        """One CTA per row. Grid = (num_rows, 1, 1)."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        next_n = cutlass.const_expr(self.next_n)
        top_k = cutlass.const_expr(self.top_k)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        kC = cutlass.const_expr(self.kC)
        kNumBins = cutlass.const_expr(self.kNumBins)

        warp_id = tidx // self.WARP_SIZE
        lane = tidx & (self.WARP_SIZE - 1)

        row_idx = bidx
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
        # (For cr == 1, the divide is a no-op, but the explicit form mirrors
        # the CUDA branch and keeps the IR straightforward to read.)
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
        output_values_row = output_values[row_idx, None]
        output_indices_row = output_indices[row_idx, None]
        pre_idx_count = pre_idx.shape[1]

        griddepcontrol_wait()

        # ---- Shared memory allocation ----
        smem = SmemAllocator()
        # keys[kC] fp32 (P3 candidate values; smem keys always fp32 even for half-prec)
        # Use fp32 even for half-prec to make secant search algorithm keep the accuracy/precision and converge faster.
        smem_keys = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((kC,), order=(0,)),
            byte_alignment=128,
        )
        # vals[kC] int32 (P3 candidate indices)
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
        # warp_counts[NUM_WARPS] int32 (P3 prefix-sum scratch)
        smem_wcnt = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
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
        # Extra float scalar: pmax_saved (kept separate so s_thr stays exactly 3-wide)
        s_thr_extra = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((1,), order=(0,)),
            byte_alignment=16,
        )
        # Int scalars: cand_count, done, cnt_lo, cnt_hi, out_count
        s_iscalars = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((5,), order=(0,)),
            byte_alignment=16,
        )

        # ---- Degenerate path: N <= top_k → copy input as-is ----
        if N <= cutlass.Int32(top_k):
            jd = tidx
            while jd < N:
                output_values_row[jd] = input_row[jd]
                output_indices_row[jd] = cutlass.Int32(jd)
                jd = jd + cutlass.Int32(num_threads)
            jp = N + cutlass.Int32(tidx)
            while jp < cutlass.Int32(top_k):
                output_values_row[jp] = self.dtype(self.NEG_FLT_MAX)
                output_indices_row[jp] = cutlass.Int32(-1)
                jp = jp + cutlass.Int32(num_threads)
        else:
            # =================================================================
            # Phase 1 — preIdx Min/Max/Mean
            # =================================================================
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
                s_thr_extra,
                s_iscalars,
                tidx,
                warp_id,
                lane,
            )

            # Degenerate threshold init: val_hi <= -self.FLT_MAX or val_lo >= val_hi
            v_lo = s_thr[1]
            v_hi = s_thr[2]
            if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
                if tidx == 0:
                    # Emit identity output (first min(top_k, N) indices)
                    emit_count = cutlass.Int32(top_k) if cutlass.Int32(top_k) < N else N
                    je = cutlass.Int32(0)
                    while je < emit_count:
                        output_indices_row[je] = je
                        output_values_row[je] = input_row[je]
                        je = je + cutlass.Int32(1)
            else:
                # =============================================================
                # Phase 2 — Secant threshold search
                # =============================================================
                self.phase2_secant_search(
                    input_row,
                    N,
                    smem_ptcnt,
                    smem_wcnt,
                    s_thr,
                    s_iscalars,
                    tidx,
                    warp_id,
                    lane,
                )

                # =============================================================
                # Phase 3 — Ballot-free candidate collect
                # =============================================================
                self.phase3_collect_candidates(
                    input_row,
                    N,
                    smem_keys,
                    smem_vals,
                    smem_ptcnt,
                    smem_wcnt,
                    s_thr,
                    s_iscalars,
                    tidx,
                    warp_id,
                    lane,
                )

                # =============================================================
                # Phase 4 — Histogram snap + writeback (top-K from candidates)
                # =============================================================
                # cand_count = min(s_iscalars[0], kCC)
                cand_count_p4 = s_iscalars[0]
                if cand_count_p4 > cutlass.Int32(self.kC):
                    cand_count_p4 = cutlass.Int32(self.kC)

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

        griddepcontrol_launch_dependents()

    # ------------------------------------------------------------------
    # Host-side launcher
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seq_lens: cute.Tensor,
        output_values: cute.Tensor,
        output_indices: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        self.gvr_topk_kernel(
            input_data,
            pre_idx,
            seq_lens,
            output_values,
            output_indices,
        ).launch(
            grid=(num_rows, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )


__all__ = ["GvrTopKKernel", "GvrParams"]
