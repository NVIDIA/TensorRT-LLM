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

"""GVR (Guess-Verify-Refine) heuristic Top-K kernel — cuTe DSL port for
Blackwell sm_100.

Drop-in alternative to ``torch.ops.trtllm.indexer_topk_decode`` (CUDA
``heuristicTopKMultiRowKernel``). Single CTA per row, block size 512,
preIdxOffset = (row_idx % next_n) + 1 (V3.2 decode semantics).

Supported (dtype, K): fp32/bf16/fp16 × 512/1024/2048.


TODO:
1. 对齐cuda gvr的性能：看看有那里的实现，没对齐？
=》 check sass. 找差异.
(1) check instructions: ldg, stg, sts, lds...
(2) check smem usage.
(3) check register usage.

After aligned, next, continue to tune:
1. why use fp32 even for half-prec? I think we can use dtype for all.
2. tune num_threads_per_block.
3. optimize smem_ptcnt.
4. tune vec_size: 128->256.
5. block prefix sum parallelization.
6. fmin/fmax有向量化指令吗？如果有，可以向量化。
7.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

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
    (FADD R, RZ, -a; FADD R, RZ, -b; FMNMX R, ...; FADD R, RZ, -R).
    Per phase-resolved hotspot analysis 2026-05-11
    (.perfbot/learnings/20260511T150703-agent.yaml F004), this pattern
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
# Constants (mirror heuristic_topk.cuh:157-167)
# BLOCK_SIZE and WARP_SIZE moved into GvrTopKKernel class as class-level
# attributes — see class body below. (BLOCK_SIZE is configurable per-instance
# via num_threads; WARP_SIZE is a hardware constant.)
# =============================================================================

MAX_CANDIDATES = 6144  # K=2048 + SAFETY_MARGIN×2
MAX_REFINE_ITERS = 15
NUM_BINS_DEFAULT = 2048

FLT_MAX = 3.4028235e38
NEG_FLT_MAX = -FLT_MAX


# =============================================================================
# GvrParams<T, K> — mirror heuristic_topk.cuh:209-285
# =============================================================================


@dataclass(frozen=True)
class GvrParams:
    kFTarget: int
    kC: int  # candidate buffer cap
    kNumBins: int  # histogram bin count

    @staticmethod
    def get(dtype_name: str, top_k: int) -> "GvrParams":
        """Returns the per-(dtype, K) specialization parameters.

        Mirrors CUDA template specialization GvrParams<T, K>.
        """
        TABLE = {
            ("float32", 512): GvrParams(kFTarget=384, kC=5120, kNumBins=1024),
            ("float32", 1024): GvrParams(kFTarget=2560, kC=5120, kNumBins=1024),
            ("float32", 2048): GvrParams(kFTarget=3072, kC=6144, kNumBins=NUM_BINS_DEFAULT),
            ("bfloat16", 512): GvrParams(kFTarget=384, kC=5120, kNumBins=512),
            ("bfloat16", 1024): GvrParams(kFTarget=2560, kC=5120, kNumBins=512),
            ("bfloat16", 2048): GvrParams(kFTarget=4096, kC=5120, kNumBins=NUM_BINS_DEFAULT),
            ("float16", 512): GvrParams(kFTarget=384, kC=5120, kNumBins=512),
            ("float16", 1024): GvrParams(kFTarget=2560, kC=5120, kNumBins=1024),
            ("float16", 2048): GvrParams(kFTarget=4096, kC=5120, kNumBins=NUM_BINS_DEFAULT),
        }
        if (dtype_name, top_k) not in TABLE:
            raise ValueError(f"Unsupported GvrParams<{dtype_name}, {top_k}>")
        return TABLE[(dtype_name, top_k)]


# =============================================================================
# GvrTopKKernel — class-based cuTe DSL kernel matching TRTLLM idiom
# =============================================================================


class GvrTopKKernel:
    """GVR (Guess-Verify-Refine) heuristic top-K kernel, cuTe DSL port.

    One CTA per row (matches CUDA heuristicTopKMultiRowKernel BS=1 semantics).
    Block size = 512, smem region sized to GvrParams<dtype, top_k>.

    Algorithm phases (mirror heuristic_topk.cuh:627-1192):
      P1: preIdx Min/Max/Mean → initial threshold
      P2: Secant threshold search loop (count-only)
      P3: Ballot-free candidate collect into smem keys[]/vals[]
      P4: Histogram snap (cand → exact top-K) + writeback

    Production-path semantics: preIdxOffset = (row_idx % next_n) + 1
    (V3.2 decode +1 temporal shift; see heuristicTopKDecode.cu:89-93).
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
        enable_warp_parallel_reduce: bool = False,
    ):
        # cutlass.Numeric enum: cutlass.Float32 / cutlass.BFloat16 / cutlass.Float16
        self.dtype = dtype
        self.top_k = top_k
        self.next_n = next_n
        # WARP_SIZE = 32 is a hardware constant on all NVIDIA GPUs.
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
        # reduce/scan in warp 0. Beneficial when num_warps is large (e.g.
        # num_threads=1024 -> num_warps=32) so serial latency is meaningful;
        # at num_threads=512 (num_warps=16) measured ~2pp regression on
        # synth, so default off.
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

        params = GvrParams.get(self._dtype_name, top_k)
        self.kC = params.kC
        self.kNumBins = params.kNumBins
        self.kFTarget = params.kFTarget

    # ------------------------------------------------------------------
    # Build a 128-bit copy atom for the input scan loops. With
    # use_constant_hint=True we use CopyG2ROp+invariant to get
    # LDG.E.*.CONSTANT (read-only cache, matches CUDA __ldg). Defined as
    # a plain Python method (not @cute.jit) so the if-else branches both
    # bind copy_atom in the same trace scope.
    # ------------------------------------------------------------------
    def _make_load_copy_atom(self):
        # num_bits_per_copy matches self.vec_bits (128 default; 256 when
        # use_256bit_load=True) so cute emits LDG.E.128 or LDG.E.256 per copy.
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
    # Input load helper — casts to fp32 regardless of input dtype.
    # Mirrors CUDA's Trait::to_fp32 pattern (heuristic_topk.cuh:74-153).
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
    # Warp-level reductions (mirror heuristic_topk.cuh:336-398)
    #
    # Use cute.arch.warp_redux_sync — direct map to PTX redux.sync (sm_80+
    # for int32, sm_100 hardware for fp32). NCU on 2026-05-11 showed the
    # generic cute.arch.warp_reduction_{sum,max} lower to SHFL.BFLY 5-step
    # tree, not REDUX — accounting for ~29% (~7.7 us) of the cuTe vs CUDA
    # gap at K=2048 fp32 BS=1. warp_redux_sync emits redux.sync directly
    # (see cutlass/cute/arch/nvvm_wrappers.py:1611).
    # ------------------------------------------------------------------
    @cute.jit
    def warp_reduce_sum_i32(self, val):
        # REDUX.SYNC.ADD.S32 (sm_80+)
        return cute.arch.warp_redux_sync(val, "add")

    @cute.jit
    def warp_reduce_sum_f32(self, val):
        # PTX redux.sync has no fadd — keep shfl-tree fallback.
        return cute.arch.warp_reduction_sum(val)

    @cute.jit
    def warp_reduce_min_f32(self, val):
        # PTX redux.sync.fmin.f32 (sm_100). Single instruction; supersedes
        # the prior negation trick (-warp_reduction_max(-val)) which lowered
        # to 5x SHFL.BFLY + 2x FNEG.
        return cute.arch.warp_redux_sync(val, "fmin")

    @cute.jit
    def warp_reduce_max_f32(self, val):
        # PTX redux.sync.fmax.f32 (sm_100). Supersedes warp_reduction_max
        # (shfl tree) lowering.
        return cute.arch.warp_redux_sync(val, "fmax")

    # ------------------------------------------------------------------
    # Phase 1: preIdx Min/Max/Mean → initial threshold
    # CUDA source: heuristic_topk.cuh:645-714
    # ------------------------------------------------------------------
    @cute.jit
    def phase1_preidx_stats(
        self,
        input_row,  # cute.Tensor [N] fp32 (post-cast for half-prec)
        N,
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
        local_min = cutlass.Float32(FLT_MAX)
        local_max = cutlass.Float32(NEG_FLT_MAX)
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

        # tid==0 aggregates NUM_WARPS warp results → pmin, pmax, psum, pcnt → pmean.
        # Output scalars (float):
        #   s_thr[0] = threshold (initial = pmean)
        #   s_thr[1] = val_lo (= pmin)
        #   s_thr[2] = val_hi (= pmax)
        #   s_thr_extra[0] = pmax_saved (= pmax)
        # Output scalars (int):
        #   s_iscalars[0] = cand_count = 0
        #   s_iscalars[1] = done = 0
        #   s_iscalars[2] = cnt_lo seed = pre_idx_count + pre_idx_count // 4
        #   s_iscalars[3] = cnt_hi seed = 1
        #   s_iscalars[4] = out_count = 0
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
                v_min = cutlass.Float32(FLT_MAX)
                v_max = cutlass.Float32(NEG_FLT_MAX)
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
            # OLD: tid==0 serial loop.
            if tidx == 0:
                pmin = cutlass.Float32(FLT_MAX)
                pmax = cutlass.Float32(NEG_FLT_MAX)
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
    # CUDA source: heuristic_topk.cuh:400-429 (blockCountGE)
    #
    # Per-thread accumulate (4-element strided), cache to smem_ptcnt[tid]
    # for P3 reuse, warp-reduce, block-reduce → s_iscalars[0] = cand_count.
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_ge(
        self,
        input_row,  # cute.Tensor [N] fp32
        N,
        threshold,  # cutlass.Float32 scalar
        smem_ptcnt,  # cute.Tensor [BLOCK_SIZE] int32 (P3 cache)
        smem_wcnt,  # cute.Tensor [NUM_WARPS] int32 (block reduce scratch)
        s_iscalars,  # cute.Tensor [5] int32 (writes [0] = cand_count)
        tidx,
        warp_id,
        lane,
    ):
        """Count input[i] >= threshold across N elements.

        Vectorized: each thread loads 128-bit per iter (4 fp32 / 8 bf16 / 8 fp16)
        via cute.copy + CopyUniversalOp. Mirrors CUDA GVR's float4 / int4 LDG.128
        pattern (heuristic_topk.cuh:407 / :524). Falls back to scalar tail for
        the N-mod-vec_w remainder.
        """
        num_threads = cutlass.const_expr(self.num_threads)
        # Vec width = (128 or 256) / dtype_bits, controlled by use_256bit_load
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)  # 16 or 32
        c = cutlass.Int32(0)

        # Vectorized main loop using cute.copy. use_constant_hint=True
        # switches to CopyG2ROp+invariant which emits SASS LDG.E.*.CONSTANT
        # (read-only data cache, matches CUDA __ldg).
        copy_atom = self._make_load_copy_atom()

        # 4-way unrolled fast path: issue 4 independent LDG.E.128 per round
        # to engage LSU pipelining. Mirrors nvcc/ptxas auto-partial-unroll
        # on the CUDA side (LDG.E.128.CONSTANT R8/R12/R16/R20 with
        # +0x2000/+0x4000/+0x6000 strides).
        #
        # Strided-layout trick: instead of 4 separate cute.make_ptr calls
        # (which produce 4 independent base registers in SASS), use a
        # single base ptr + (UNROLL, vec_w) layout with stride=(step, 1).
        # cute can then emit 4 LDG.E.128 sharing one base register with
        # constexpr immediate offsets — matching the CUDA SASS pattern.
        # UNROLL = 4 / UNROLL_MID = 2 — only referenced in commented-out
        # manual-unroll code below; the active path uses cutlass.range(unroll=4).
        # Unrolling switches (read from self, set in __init__). Both gates
        # are Python-level (NOT cutlass.const_expr) so when disabled the
        # entire code block is invisible to cute's IR — no reg pressure or
        # SASS bloat. See __init__ docstring for default policy.

        step_elem = cutlass.const_expr(num_threads * vec_w)

        row_addr = input_row.iterator.toint()
        n_aligned = (N // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        i = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        # Fast path: 4-way unroll for LSU-pipelining ILP.
        # Two layout choices controlled by self.use_strided_layout:
        #   * True  (strided): single make_ptr + (UNROLL, vec_w) layout with
        #     stride=(step, 1) → cute emits 4 LDG.E.128 sharing base reg with
        #     +0x2000/+0x4000/+0x6000 imm offsets (CUDA __ldg pattern).
        #   * False (separate-ptrs): 4 independent make_ptr calls + 4 frags →
        #     cute emits 4 LDG.E.128 with 4 independent base regs (matches
        #     the b459a8ff2 commit's behavior). A/B knob for fp32 large-grid.
        if self.enable_unroll_4:
            # =================================================================
            # OLD (commented out for A/B comparison): manual 4-way unroll via
            # `while + range_constexpr(UNROLL)`. Trace-time generates 4 explicit
            # cute.copy + 16 explicit FSETP. SASS analysis showed: 4× LDG.E.128
            # emitted, but each with its own base reg + interleaved address calc
            # (VIADD + IMAD.WIDE per LDG) — NOT folded into the CUDA-style
            # shared base + +0x2000/+0x4000/+0x6000 immediate offsets pattern.
            # =================================================================
            # # Entry guard: last element of the last unrolled iter must fit.
            # fast_last_offset = cutlass.const_expr((UNROLL - 1) * step_elem + (vec_w - 1))
            # if self.use_strided_layout:
            #     big_layout = cute.make_layout((UNROLL, vec_w), stride=(step_elem, 1))
            #     big_frag = cute.make_fragment(big_layout, self.dtype)
            #     while i + cutlass.Int32(fast_last_offset) < N:
            #         big_src_ptr = cute.make_ptr(
            #             self.dtype,
            #             row_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
            #             cute.AddressSpace.gmem,
            #             assumed_align=16,
            #         )
            #         big_src = cute.make_tensor(big_src_ptr, big_layout)
            #         for u in cutlass.range_constexpr(UNROLL):
            #             src_u = big_src[u, None]
            #             frag_u = big_frag[u, None]
            #             cute.copy(copy_atom, src_u, frag_u)
            #         for u in cutlass.range_constexpr(UNROLL):
            #             for j in cutlass.range_constexpr(vec_w):
            #                 if cutlass.const_expr(self.dtype == cutlass.Float32):
            #                     vj = big_frag[u, j]
            #                 else:
            #                     vj = cutlass.Float32(big_frag[u, j])
            #                 if vj >= threshold:
            #                     c = c + cutlass.Int32(1)
            #         i = i + cutlass.Int32(UNROLL * step_elem)
            # else:
            #     # Separate-ptrs path: matches the b459a8ff2 commit's writing.
            #     frag_a = cute.make_fragment((vec_w,), self.dtype)
            #     frag_b = cute.make_fragment((vec_w,), self.dtype)
            #     frag_c = cute.make_fragment((vec_w,), self.dtype)
            #     frag_d = cute.make_fragment((vec_w,), self.dtype)
            #     frags_sep = (frag_a, frag_b, frag_c, frag_d)
            #     while i + cutlass.Int32(fast_last_offset) < N:
            #         for u in cutlass.range_constexpr(UNROLL):
            #             u_off = cutlass.const_expr(u * step_elem)
            #             src_ptr_u = cute.make_ptr(
            #                 self.dtype,
            #                 row_addr
            #                 + cutlass.Int64(i + cutlass.Int32(u_off)) * cutlass.Int64(elem_bytes),
            #                 cute.AddressSpace.gmem,
            #                 assumed_align=16,
            #             )
            #             src_u = cute.make_tensor(src_ptr_u, cute.make_layout((vec_w,)))
            #             cute.copy(copy_atom, src_u, frags_sep[u])
            #         for u in cutlass.range_constexpr(UNROLL):
            #             for j in cutlass.range_constexpr(vec_w):
            #                 if cutlass.const_expr(self.dtype == cutlass.Float32):
            #                     vj = frags_sep[u][j]
            #                 else:
            #                     vj = cutlass.Float32(frags_sep[u][j])
            #                 if vj >= threshold:
            #                     c = c + cutlass.Int32(1)
            #         i = i + cutlass.Int32(UNROLL * step_elem)

            # =================================================================
            # NEW (experimental): use cutlass.range(unroll=4).
            # Each loop body loads 1 vec_w chunk; LLVM unrolls 4 iters at IR
            # level. After unroll, GVN/CSE has one common base ptr in scope and
            # MAY fold the 4 derived addresses into shared base + immediate
            # offsets (matching CUDA's LDG.E.128 [base+0x2000/0x4000/0x6000]).
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
            # Unroll factor: 4 for 128-bit and fp32 256-bit (no cvt spill);
            # 2 for 256-bit bf16/fp16 to avoid reg spill from cvt-to-fp32 doubling.
            # 4 × 16 bf16/fp16 elems × cvt fp32 = 64 reg dest > 40 reg cap → spill.
            # 2 × 16 bf16/fp16 elems × cvt fp32 = 32 reg dest, fits.
            _UNROLL = cutlass.const_expr(
                2 if (self.use_256bit_load and self.dtype != cutlass.Float32) else 4
            )
            for k in cutlass.range(big_iters, unroll=_UNROLL):
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

        # =====================================================================
        # OLD (commented out): manual 2-way unroll medium path. Same scheduling
        # issue as the manual 4-way path — separate base reg per LDG.
        # Replaced by the cutlass.range(unroll=4) path above, which covers all
        # vec_w-aligned iterations (LLVM unroll handles the [0,4) epilog).
        # =====================================================================
        # # Medium path: 2-way unroll for the case where fast-path remainder is
        # # in [2*step, 4*step). Runs at most once (after this iter, remainder
        # # is < 2*step). Same strided-layout trick as fast path → 2 LDG.E.128
        # # in flight sharing base register. Skipped entirely when fast path
        # # exits with remainder < 2*step (e.g., N=16384 → fast 1, mid 0).
        # # Python-level gate: when False, the entire block is absent from
        # # the cute IR — no reg pressure or SASS bloat.
        # if self.enable_unroll_2:
        #     mid_layout = cute.make_layout((UNROLL_MID, vec_w), stride=(step_elem, 1))
        #     mid_frag = cute.make_fragment(mid_layout, self.dtype)
        #     mid_last_offset = cutlass.const_expr((UNROLL_MID - 1) * step_elem + (vec_w - 1))
        #     while i + cutlass.Int32(mid_last_offset) < N:
        #         mid_src_ptr = cute.make_ptr(
        #             self.dtype,
        #             row_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
        #             cute.AddressSpace.gmem,
        #             assumed_align=16,
        #         )
        #         mid_src = cute.make_tensor(mid_src_ptr, mid_layout)
        #         for u in cutlass.range_constexpr(UNROLL_MID):
        #             src_u = mid_src[u, None]
        #             frag_u = mid_frag[u, None]
        #             cute.copy(copy_atom, src_u, frag_u)
        #         for u in cutlass.range_constexpr(UNROLL_MID):
        #             for j in cutlass.range_constexpr(vec_w):
        #                 if cutlass.const_expr(self.dtype == cutlass.Float32):
        #                     vj = mid_frag[u, j]
        #                 else:
        #                     vj = cutlass.Float32(mid_frag[u, j])
        #                 if vj >= threshold:
        #                     c = c + cutlass.Int32(1)
        #         i = i + cutlass.Int32(UNROLL_MID * step_elem)

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
        # its post-processing of cand_count, matching CUDA blockCountGE
        # (heuristic_topk.cuh:413-441 — no sync at function end).
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
            # OLD: tid==0 serial sum.
            if tidx == 0:
                total = cutlass.Int32(0)
                for w in cutlass.range_constexpr(self.num_warps):
                    total = total + smem_wcnt[w]
                s_iscalars[0] = total

    # ------------------------------------------------------------------
    # Phase 2: Secant-interpolation threshold search
    # CUDA source: heuristic_topk.cuh:716-810
    #
    # Refines threshold to bring cand_count into [kK, kCC] using secant
    # interpolation on (val_lo, cnt_lo) / (val_hi, cnt_hi). At most
    # MAX_REFINE_ITERS iterations.
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
        # Do we have methods to reduce its write.
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
        # Runtime `while` with `done` check in the loop condition, matching
        # CUDA's `for(iter; iter < MAX_REFINE_ITERS; iter++) { if(done) break; }`
        # (heuristic_topk.cuh:683-743). Previous Python-unrolled `for it in
        # range(15)` ran the guard check for all 15 unrolled bodies even
        # when the kernel converged at iter 3, wasting ~12 LDS+ICMP+branch
        # per kernel call.
        it = cutlass.Int32(0)
        while it < cutlass.Int32(MAX_REFINE_ITERS) and s_iscalars[1] == cutlass.Int32(0):
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
    # CUDA source: heuristic_topk.cuh:813-912
    #
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
            # loop condition (matches CUDA `for(retry; retry<10 && cand>kCC;)`
            # at heuristic_topk.cuh:769). Previous `for rs in range(10)` Python
            # unroll ran 10 guard checks even after early convergence.
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

        # TODO: optimize this. using mindy's block prefix sum.
        # 5-level shfl_up_sync inclusive scan within warp.
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            # NOTE: mask_and_clamp=0 matches CUDA __shfl_up_sync semantics
            # (cuTe DSL default is 31 which makes shfl.up always return own
            # value, breaking the prefix sum). See SESSION_LOG session 4.
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
            # OLD: tid==0 serial exclusive prefix.
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
        # TODO: 这块的逻辑没有看懂？warp的start offset + ?
        my_write_pos = my_base + my_excl_offset

        # ---- Stream-write loop ----
        # 3-tier cascade mirroring block_count_ge: 4-way unrolled fast path
        # → 2-way medium → 1-way tail → scalar tail. Each conditional write
        # advances thread-local wc; LDGs themselves are independent and
        # cute can pipeline them. Switches gate via self.* (set in __init__).
        thr_final = s_thr[0]
        vec_w_p3 = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes_p3 = cutlass.const_expr(self.dtype.width // 8)
        vec_align_p3 = cutlass.const_expr(self.vec_align_bytes)
        # use_constant_hint=True → LDG.E.*.CONSTANT (same as block_count_ge).
        copy_atom_p3 = self._make_load_copy_atom()
        row_addr_p3 = input_row.iterator.toint()
        step_elem_p3 = cutlass.const_expr(num_threads * vec_w_p3)
        # UNROLL = 4 / UNROLL_MID = 2 — only referenced in commented-out manual
        # unroll code below; the active path uses cutlass.range(unroll=4).

        n_aligned = (N // cutlass.Int32(vec_w_p3)) * cutlass.Int32(vec_w_p3)
        wc = my_write_pos
        ic2 = tidx * cutlass.Int32(vec_w_p3)
        step = cutlass.Int32(step_elem_p3)

        # Phase3 unrolling: master gated by self.enable_phase3_unroll.
        # When OFF, only the tail 1-way loop runs (matches the pre-unroll
        # state of phase3_collect). When ON, the inner enable_unroll_4 and
        # enable_unroll_2 switches independently control the 4-way fast
        # path and 2-way medium path (same semantics as in block_count_ge).
        if self.enable_phase3_unroll:
            # Fast path: 4-way unrolled vec loop (4 LDG.E.128 in flight).
            # See block_count_ge for strided vs separate-ptrs trade-offs.
            if self.enable_unroll_4:
                # =============================================================
                # OLD (commented out for A/B comparison): manual 4-way unroll
                # via while + range_constexpr(UNROLL). Same issue as in
                # block_count_ge — 4× LDG.E.128 emitted but each with its own
                # base reg + VIADD/IMAD.WIDE address calc, NOT folded into
                # CUDA-style shared base + immediate offsets pattern.
                # =============================================================
                # fast_last_offset_p3 = cutlass.const_expr(
                #     (UNROLL - 1) * step_elem_p3 + (vec_w_p3 - 1)
                # )
                # if self.use_strided_layout:
                #     big_layout_p3 = cute.make_layout((UNROLL, vec_w_p3), stride=(step_elem_p3, 1))
                #     big_frag_p3 = cute.make_fragment(big_layout_p3, self.dtype)
                #     while ic2 + cutlass.Int32(fast_last_offset_p3) < N:
                #         big_src_ptr = cute.make_ptr(
                #             self.dtype,
                #             row_addr_p3 + cutlass.Int64(ic2) * cutlass.Int64(elem_bytes_p3),
                #             cute.AddressSpace.gmem,
                #             assumed_align=16,
                #         )
                #         big_src = cute.make_tensor(big_src_ptr, big_layout_p3)
                #         for u in cutlass.range_constexpr(UNROLL):
                #             cute.copy(
                #                 copy_atom_p3,
                #                 big_src[u, None],
                #                 big_frag_p3[u, None],
                #             )
                #         for u in cutlass.range_constexpr(UNROLL):
                #             u_off = cutlass.const_expr(u * step_elem_p3)
                #             for j in cutlass.range_constexpr(vec_w_p3):
                #                 if cutlass.const_expr(self.dtype == cutlass.Float32):
                #                     vj_p3 = big_frag_p3[u, j]
                #                 else:
                #                     vj_p3 = cutlass.Float32(big_frag_p3[u, j])
                #                 if vj_p3 >= thr_final and wc < cutlass.Int32(kCC):
                #                     smem_keys[wc] = vj_p3
                #                     smem_vals[wc] = ic2 + cutlass.Int32(u_off + j)
                #                     wc = wc + cutlass.Int32(1)
                #         ic2 = ic2 + cutlass.Int32(UNROLL * step_elem_p3)
                # else:
                #     # Separate-ptrs path (matches the b459a8ff2 commit style).
                #     frag_a = cute.make_fragment((vec_w_p3,), self.dtype)
                #     frag_b = cute.make_fragment((vec_w_p3,), self.dtype)
                #     frag_c = cute.make_fragment((vec_w_p3,), self.dtype)
                #     frag_d = cute.make_fragment((vec_w_p3,), self.dtype)
                #     frags_sep = (frag_a, frag_b, frag_c, frag_d)
                #     while ic2 + cutlass.Int32(fast_last_offset_p3) < N:
                #         for u in cutlass.range_constexpr(UNROLL):
                #             u_off = cutlass.const_expr(u * step_elem_p3)
                #             src_ptr_u = cute.make_ptr(
                #                 self.dtype,
                #                 row_addr_p3
                #                 + cutlass.Int64(ic2 + cutlass.Int32(u_off))
                #                 * cutlass.Int64(elem_bytes_p3),
                #                 cute.AddressSpace.gmem,
                #                 assumed_align=16,
                #             )
                #             src_u = cute.make_tensor(src_ptr_u, cute.make_layout((vec_w_p3,)))
                #             cute.copy(copy_atom_p3, src_u, frags_sep[u])
                #         for u in cutlass.range_constexpr(UNROLL):
                #             u_off = cutlass.const_expr(u * step_elem_p3)
                #             for j in cutlass.range_constexpr(vec_w_p3):
                #                 if cutlass.const_expr(self.dtype == cutlass.Float32):
                #                     vj_p3 = frags_sep[u][j]
                #                 else:
                #                     vj_p3 = cutlass.Float32(frags_sep[u][j])
                #                 if vj_p3 >= thr_final and wc < cutlass.Int32(kCC):
                #                     smem_keys[wc] = vj_p3
                #                     smem_vals[wc] = ic2 + cutlass.Int32(u_off + j)
                #                     wc = wc + cutlass.Int32(1)
                #         ic2 = ic2 + cutlass.Int32(UNROLL * step_elem_p3)

                # =============================================================
                # NEW (experimental): cutlass.range(unroll=4).
                # Each body loads 1 vec_w_p3 chunk; LLVM unrolls 4 iters at IR
                # level. Same intent as the Phase-2 rewrite above.
                # =============================================================
                rng_frag_p3 = cute.make_fragment((vec_w_p3,), self.dtype)
                big_iters_p3 = cutlass.Int32(0)
                if N > ic2 + cutlass.Int32(vec_w_p3 - 1):
                    big_iters_p3 = (N - ic2 - cutlass.Int32(vec_w_p3)) // cutlass.Int32(
                        step_elem_p3
                    ) + cutlass.Int32(1)
                # Unroll factor: same logic as block_count_ge (Phase 2):
                # 2 for 256-bit bf16/fp16 to avoid cvt-to-fp32 spill; else 4.
                _UNROLL_P3 = cutlass.const_expr(
                    2 if (self.use_256bit_load and self.dtype != cutlass.Float32) else 4
                )
                for k in cutlass.range(big_iters_p3, unroll=_UNROLL_P3):
                    ic2_local = ic2 + k * cutlass.Int32(step_elem_p3)
                    src_ptr_k = cute.make_ptr(
                        self.dtype,
                        row_addr_p3 + cutlass.Int64(ic2_local) * cutlass.Int64(elem_bytes_p3),
                        cute.AddressSpace.gmem,
                        assumed_align=vec_align_p3,
                    )
                    src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w_p3,)))
                    cute.copy(copy_atom_p3, src_k, rng_frag_p3)
                    for j in cutlass.range_constexpr(vec_w_p3):
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            vj_p3 = rng_frag_p3[j]
                        else:
                            vj_p3 = cutlass.Float32(rng_frag_p3[j])
                        if vj_p3 >= thr_final and wc < cutlass.Int32(kCC):
                            smem_keys[wc] = vj_p3
                            smem_vals[wc] = ic2_local + cutlass.Int32(j)
                            wc = wc + cutlass.Int32(1)
                # Advance ic2 past all consumed vec_w-aligned positions.
                ic2 = ic2 + big_iters_p3 * cutlass.Int32(step_elem_p3)

            # =================================================================
            # OLD (commented out): manual 2-way unroll medium path. Replaced by
            # the cutlass.range(unroll=4) path above which covers all
            # vec_w-aligned iterations.
            # =================================================================
            # # Medium path: 2-way unroll for the remainder when fast-path exits
            # # with [2*step, 4*step) elements left. Runs at most once.
            # if self.enable_unroll_2:
            #     mid_layout_p3 = cute.make_layout((UNROLL_MID, vec_w_p3), stride=(step_elem_p3, 1))
            #     mid_frag_p3 = cute.make_fragment(mid_layout_p3, self.dtype)
            #     mid_last_offset_p3 = cutlass.const_expr(
            #         (UNROLL_MID - 1) * step_elem_p3 + (vec_w_p3 - 1)
            #     )
            #     while ic2 + cutlass.Int32(mid_last_offset_p3) < N:
            #         mid_src_ptr = cute.make_ptr(
            #             self.dtype,
            #             row_addr_p3 + cutlass.Int64(ic2) * cutlass.Int64(elem_bytes_p3),
            #             cute.AddressSpace.gmem,
            #             assumed_align=16,
            #         )
            #         mid_src = cute.make_tensor(mid_src_ptr, mid_layout_p3)
            #         for u in cutlass.range_constexpr(UNROLL_MID):
            #             cute.copy(
            #                 copy_atom_p3,
            #                 mid_src[u, None],
            #                 mid_frag_p3[u, None],
            #             )
            #         for u in cutlass.range_constexpr(UNROLL_MID):
            #             u_off = cutlass.const_expr(u * step_elem_p3)
            #             for j in cutlass.range_constexpr(vec_w_p3):
            #                 if cutlass.const_expr(self.dtype == cutlass.Float32):
            #                     vj_p3 = mid_frag_p3[u, j]
            #                 else:
            #                     vj_p3 = cutlass.Float32(mid_frag_p3[u, j])
            #                 if vj_p3 >= thr_final and wc < cutlass.Int32(kCC):
            #                     smem_keys[wc] = vj_p3
            #                     smem_vals[wc] = ic2 + cutlass.Int32(u_off + j)
            #                     wc = wc + cutlass.Int32(1)
            #             ic2 = ic2 + cutlass.Int32(UNROLL_MID * step_elem_p3)

        # Tail vec loop: 1-way, handles remainder < 2*step. ic2 stays vec_w-
        # aligned across the unroll loop (steps by num_threads*vec_w_p3).
        tail_frag_p3 = cute.make_fragment((vec_w_p3,), self.dtype)
        while ic2 + cutlass.Int32(vec_w_p3 - 1) < N:
            src_ptr_p3 = cute.make_ptr(
                self.dtype,
                row_addr_p3 + cutlass.Int64(ic2) * cutlass.Int64(elem_bytes_p3),
                cute.AddressSpace.gmem,
                assumed_align=vec_align_p3,
            )
            src_p3 = cute.make_tensor(src_ptr_p3, cute.make_layout((vec_w_p3,)))
            cute.copy(copy_atom_p3, src_p3, tail_frag_p3)
            for j in cutlass.range_constexpr(vec_w_p3):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj_p3 = tail_frag_p3[j]
                else:
                    vj_p3 = cutlass.Float32(tail_frag_p3[j])
                if vj_p3 >= thr_final and wc < cutlass.Int32(kCC):
                    smem_keys[wc] = vj_p3
                    smem_vals[wc] = ic2 + cutlass.Int32(j)
                    wc = wc + cutlass.Int32(1)
            ic2 = ic2 + step

        # Tail scalar loop (N % vec_w)
        it2 = n_aligned + tidx
        while it2 < N:
            v = self._load_fp32(input_row, it2)
            if v >= thr_final and wc < cutlass.Int32(kCC):
                smem_keys[wc] = v
                smem_vals[wc] = it2
                wc = wc + cutlass.Int32(1)
            it2 = it2 + cutlass.Int32(num_threads)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # block_fused_snap_iter — P4 snap convergence inner step
    # CUDA source: heuristic_topk.cuh:436-510
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
        s_up = cutlass.Float32(FLT_MAX)
        s_down = cutlass.Float32(NEG_FLT_MAX)

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
                v_up = cutlass.Float32(FLT_MAX)
                v_dn = cutlass.Float32(NEG_FLT_MAX)
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
                        if total_up < cutlass.Float32(FLT_MAX):
                            s_thr[0] = total_up
                    elif cge < cutlass.Int32(kK):
                        if total_down > cutlass.Float32(NEG_FLT_MAX):
                            s_thr[0] = total_down
        else:
            # OLD: tid==0 serial 3-way reduce.
            if tidx == 0:
                tp = cutlass.Int32(0)
                total_up = cutlass.Float32(FLT_MAX)
                total_down = cutlass.Float32(NEG_FLT_MAX)
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
                    if total_up < cutlass.Float32(FLT_MAX):
                        s_thr[0] = total_up
                elif cge < cutlass.Int32(kK):
                    if total_down > cutlass.Float32(NEG_FLT_MAX):
                        s_thr[0] = total_down
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Phase 4: Histogram-based k-th selection + two-pass writeback
    # CUDA source: heuristic_topk.cuh:914-1128
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
        <  kK : emit cand_count + pad with -FLT_MAX
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
            local_cmin = cutlass.Float32(FLT_MAX)
            local_cmax = cutlass.Float32(NEG_FLT_MAX)
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
            bmin_r = cutlass.Float32(FLT_MAX)
            bmax_r = cutlass.Float32(NEG_FLT_MAX)
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
                v_p1 = cutlass.Float32(NEG_FLT_MAX)
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
                v_p2 = cutlass.Float32(NEG_FLT_MAX)
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

            # Pad remainder with -FLT_MAX / -1
            filled_par = s_iscalars[4]
            if filled_par > cutlass.Int32(kK):
                filled_par = cutlass.Int32(kK)
            ipad = filled_par + tidx
            while ipad < cutlass.Int32(kK):
                output_values_row[ipad] = self.dtype(NEG_FLT_MAX)
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
                output_values_row[i11] = self.dtype(NEG_FLT_MAX)
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
        pre_idx_offset = cutlass.Int32(row_idx % next_n) + cutlass.Int32(1)

        # Per-row length
        seq_len = seq_lens[pre_idx_row_idx]
        N = seq_len - cutlass.Int32(next_n) + cutlass.Int32(row_idx % next_n) + cutlass.Int32(1)

        # Slice per-row views.
        input_row = input_data[row_idx, None]
        pre_idx_row = pre_idx[pre_idx_row_idx, None]
        output_values_row = output_values[row_idx, None]
        output_indices_row = output_indices[row_idx, None]
        pre_idx_count = pre_idx.shape[1]

        griddepcontrol_wait()

        # ---- Shared memory allocation ----
        # Typed regions (no union/bit-cast hacks). Total ≈ same as CUDA
        # KernelSmemTplK (within SmemAllocator 128B alignment padding).
        smem = SmemAllocator()
        # keys[kC] fp32 (P3 candidate values; smem keys always fp32 even for half-prec)
        # TODO: why use fp32 even for half-prec?
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
        # cuTe DSL doesn't allow `return` from @cute.kernel under dynamic
        # predicates — restructure as if/else covering both branches.
        if N <= cutlass.Int32(top_k):
            jd = tidx
            while jd < N:
                output_values_row[jd] = input_row[jd]
                output_indices_row[jd] = cutlass.Int32(jd)
                jd = jd + cutlass.Int32(num_threads)
            jp = N + cutlass.Int32(tidx)
            while jp < cutlass.Int32(top_k):
                output_values_row[jp] = self.dtype(NEG_FLT_MAX)
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

            # Degenerate threshold init: val_hi <= -FLT_MAX or val_lo >= val_hi
            v_lo = s_thr[1]
            v_hi = s_thr[2]
            if v_hi <= cutlass.Float32(NEG_FLT_MAX) or v_lo >= v_hi:
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
                # CUDA source: heuristic_topk.cuh:716-810
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
                # CUDA source: heuristic_topk.cuh:813-912
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
                # CUDA source: heuristic_topk.cuh:914-1128
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


# =============================================================================
# Public Python entrypoint
# =============================================================================

import torch  # noqa: E402

_DTYPE_TORCH_TO_CUTE = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
}

# Compile cache keyed by (cute_dtype, top_k, next_n). Each entry is a JIT-compiled
# kernel callable; same (dtype, K, next_n) reuse avoids re-JIT.
_gvr_topk_compile_cache: dict = {}


def gvr_topk_decode(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int = 1,
    out_values: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
    num_sms: int = 148,  # default number of sms in a B200
    enable_unroll_4: Optional[bool] = None,
    enable_phase3_unroll: Optional[bool] = None,
    use_constant_hint: bool = False,
    min_blocks_per_mp: Optional[int] = None,
    use_256bit_load: bool = False,
    num_threads_per_block: int = 512,
    enable_warp_parallel_reduce: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cuTe DSL GVR Top-K, drop-in for ``torch.ops.trtllm.indexer_topk_decode``.

    Args:
        logits:    ``[num_rows, max_S]`` float32 / bfloat16 / float16.
        pre_idx:   ``[num_rows // next_n, pre_idx_count]`` int32.
                   ``pre_idx[..., 0]`` must be the argmax index — indexer invariant.
        seq_lens:  ``[num_rows // next_n]`` int32. Effective sequence length per group.
        top_k:     K ∈ {512, 1024, 2048} — compile-time specialized.
        next_n:    Temporal stride for V3.2 ``preIdxOffset = (row % next_n) + 1``.
        out_values, out_indices: Optional preallocated outputs.
        min_blocks_per_mp: Override ptxas ``__launch_bounds__(BS, min_blocks)``
            hint. ``None`` (default) → use the 3-tier shape-aware heuristic
            (see in-function comment). For dynamic-shape CUDA graphs where
            the captured shape may not match replay shape, pass an explicit
            value (e.g. ``3`` for a graph-safe single-kernel selection).

    Returns:
        Tuple ``(out_values, out_indices)`` where ``out_values`` is shaped
        ``[num_rows, top_k]`` with the same dtype as ``logits`` and
        ``out_indices`` is ``[num_rows, top_k]`` int32.

    Indices are unordered within each row (top-K set, not sorted).
    """
    assert logits.is_cuda, "logits must be on CUDA"
    assert logits.dim() == 2, f"logits must be 2D, got {logits.shape}"
    assert pre_idx.dim() == 2 and pre_idx.dtype == torch.int32
    assert seq_lens.dim() == 1 and seq_lens.dtype == torch.int32

    if logits.dtype not in _DTYPE_TORCH_TO_CUTE:
        raise ValueError(f"Unsupported logits dtype: {logits.dtype}")
    cute_dtype = _DTYPE_TORCH_TO_CUTE[logits.dtype]

    num_rows = logits.shape[0]
    if out_values is None:
        out_values = torch.empty((num_rows, top_k), dtype=logits.dtype, device=logits.device)
    if out_indices is None:
        out_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=logits.device)

    # Resolve defaults (must be concrete for cache key).
    # Dtype-based policy validated via A/B testing on B200:
    #   * bf16/fp16: strided layout + cascade (unroll_4 + unroll_2) gives
    #     consistent wins across all grid sizes (small +1pp, large +6pp).
    #   * fp32: strided layout REGRESSES large-grid configs by 30-60pp
    #     (worst: K=1024 BS=128 → 0.753 vs 1.364 separate-ptrs). Default
    #     to separate-ptrs unroll-4 without cascade.
    if enable_unroll_4 is None:
        enable_unroll_4 = True
    if enable_phase3_unroll is None:
        enable_phase3_unroll = True
    # 3-tier reg-vs-occupancy heuristic (B200 has 148 SMs):
    #   n_vec_iters < 4 (small N):
    #       → min_blocks=0 (no __launch_bounds__) — let ptxas pick natural
    #       reg count. With <4 vec iters the LLVM unroll=4 fold doesn't fire
    #       anyway, and any explicit cap (=1 or =3) just constrains the
    #       compiler against its own optimum → matches phase3_unroll baseline.
    #   n_vec_iters ≥ 4 AND num_rows ≤ 148:
    #       → min_blocks=1 (allow many regs) — grid fits 1 CTA/SM, ptxas
    #       gets 69+ regs and produces 4×LDG.E.128 back-to-back.
    #   n_vec_iters ≥ 4 AND num_rows > 148:
    #       → min_blocks=3 (keep 3 CTA/SM) — large grid needs warp diversity
    #       for latency hiding; cap regs at 42.
    #
    # vec_w by dtype: fp32 → 128/32 = 4, bf16/fp16 → 128/16 = 8.
    # num_threads = BLOCK_SIZE = 512.
    #
    # ── CUDA Graph behavior ─────────────────────────────────────────────
    # The heuristic reads `logits.shape[0]` and `logits.shape[1]` which are
    # host-side Python ints (tensor metadata). Reading them does NOT touch
    # the GPU or trigger a host-device sync, so it is safe inside a CUDA
    # graph capture region. The branch resolves at capture time and the
    # corresponding compiled kernel is recorded into the graph.
    #
    # Per-graph capture (e.g. TRT-LLM piecewise CUDA graphs that bucket by
    # batch size) is the intended usage — each capture sees the right
    # shape and selects the optimal kernel. The recorded launch is fixed
    # for that graph.
    #
    # Dynamic-shape graphs (single graph replayed at different shapes
    # than capture) bypass this heuristic at replay — the captured kernel
    # is reused regardless of the new shape's `min_blocks` "preference".
    # The kernel itself uses `sym_int` shape symbols so it remains
    # FUNCTIONALLY correct under any (num_rows, N), but may be sub-optimal
    # if the dominant runtime shape is in a different tier than capture.
    # Caller can override by passing `min_blocks_per_mp=` explicitly
    # (e.g. =3 for a graph-safe single-kernel pick).
    # ────────────────────────────────────────────────────────────────────
    if min_blocks_per_mp is None:
        # vec_bits depends on use_256bit_load; vec_w_host = vec_bits / dtype_bits
        vec_bits_host = 256 if use_256bit_load else 128
        vec_w_host = vec_bits_host // (32 if logits.dtype == torch.float32 else 16)
        n_vec_iters = max(1, logits.shape[1] // (num_threads_per_block * vec_w_host))
        if n_vec_iters < 4:
            min_blocks_per_mp = 0  # no launch_bounds — natural ptxas allocation
        elif num_rows <= num_sms:
            min_blocks_per_mp = 1
        else:
            min_blocks_per_mp = 3

    # Cache key includes the unrolling + layout switches so different
    # settings share separate compiled kernels. Shapes (num_rows, num_tokens,
    # batch) are made dynamic via `sym_int` placeholders.
    key = (
        cute_dtype,
        top_k,
        next_n,
        enable_unroll_4,
        enable_phase3_unroll,
        use_constant_hint,
        min_blocks_per_mp,
        use_256bit_load,
        num_threads_per_block,
        enable_warp_parallel_reduce,
    )
    if key not in _gvr_topk_compile_cache:
        n_rows = cute.sym_int()
        n_cols = cute.sym_int()
        n_batch = cute.sym_int()
        # 256-bit vec loads require 32-byte aligned input addresses (PyTorch
        # CUDA allocations are 256-byte aligned, and Phase 2/3 offsets are
        # multiples of vec_w*elem_bytes = 32 bytes when use_256bit_load).
        in_align = 32 if use_256bit_load else 16
        input_fake = cute.runtime.make_fake_compact_tensor(
            cute_dtype, (n_rows, n_cols), stride_order=(1, 0), assumed_align=in_align
        )
        pre_idx_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (n_batch, top_k), stride_order=(1, 0), assumed_align=16
        )
        seq_lens_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (n_batch,), stride_order=(0,)
        )
        out_values_fake = cute.runtime.make_fake_compact_tensor(
            cute_dtype, (n_rows, top_k), stride_order=(1, 0), assumed_align=16
        )
        out_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (n_rows, top_k), stride_order=(1, 0), assumed_align=16
        )
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        kernel = GvrTopKKernel(
            dtype=cute_dtype,
            top_k=top_k,
            next_n=next_n,
            num_threads=num_threads_per_block,
            enable_unroll_4=enable_unroll_4,
            enable_phase3_unroll=enable_phase3_unroll,
            use_constant_hint=use_constant_hint,
            min_blocks_per_mp=min_blocks_per_mp,
            use_256bit_load=use_256bit_load,
            enable_warp_parallel_reduce=enable_warp_parallel_reduce,
        )
        _gvr_topk_compile_cache[key] = cute.compile(
            kernel,
            input_fake,
            pre_idx_fake,
            seq_lens_fake,
            out_values_fake,
            out_indices_fake,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )

    # TVM FFI path: pass raw torch tensors directly (no from_dlpack), no
    # stream argument (env stream is picked up automatically).
    _gvr_topk_compile_cache[key](logits, pre_idx, seq_lens, out_values, out_indices)
    return out_values, out_indices


__all__ = ["GvrTopKKernel", "GvrParams", "gvr_topk_decode"]
