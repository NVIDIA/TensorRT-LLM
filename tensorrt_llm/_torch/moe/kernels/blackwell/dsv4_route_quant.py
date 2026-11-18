# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 sqrt-softplus routing and NVFP4 quantization of the MoE input in one launch, optionally also
MXFP8 for a shared expert fused into the MegaMoE kernel."""

import functools
from typing import Optional, Union

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL, griddepcontrol_wait

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

Int32, Int64, Uint8, Uint32, Float32 = (
    cutlass.Int32,
    cutlass.Int64,
    cutlass.Uint8,
    cutlass.Uint32,
    cutlass.Float32,
)

# A CTA routing one token takes at most an eighth of an SM's registers and a few KB of shared
# memory: eight per SM overlap their logit loads, and one still fits beside a concurrent kernel's
# CTA (an unfused shared expert's GEMM) in the shared-memory carveout that kernel chose.
CTAS_PER_SM = 8

# With tables one CTA routes every token: a thread per expert scores every token, then a warp per
# token picks its experts.
ROUTING_TABLES_MAX_TOKENS = 8
ROUTING_TABLES_MAX_EXPERTS = 1024

# TRTLLM-Gen FP4 tiles usable at these sizes: its smallest and the next two (computeSelectedTileN).
ROUTING_TABLE_TILES = (8, 16, 32)

# Bucket rows past the tokens that one CTA clears when the launch writes into a launch bucket.
TAIL_ROWS_PER_CTA = 32

# A token's CTA scores its experts with every warp only for a bank of at least 256 experts and up to
# 1280 tokens (b320 x 4), where it shortens the launch; otherwise one warp per token costs less.
CTA_SCORING_MIN_EXPERTS = 256
CTA_SCORING_MAX_TOKENS = 1280


def _cta_scoring(num_tokens: int, num_experts: int) -> bool:
    return num_experts >= CTA_SCORING_MIN_EXPERTS and num_tokens <= CTA_SCORING_MAX_TOKENS


def _inline_asm(result_type, operands, code: str, constraints: str, loc, ip):
    return llvm.inline_asm(
        result_type,
        [operand.ir_value(loc=loc, ip=ip) for operand in operands],
        code,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ordered(value: Float32, *, loc=None, ip=None) -> Int32:
    """An Int32 whose signed order is the float order of ``value`` (-0 below +0, NaN on top)."""
    return Int32(
        _inline_asm(
            T.i32(),
            [Float32(value)],
            "{\n.reg .b32 bits, mask;\nmov.b32 bits, $1;\nshr.s32 mask, bits, 31;\n"
            "and.b32 mask, mask, 2147483647;\nxor.b32 $0, bits, mask;\n}",
            "=r,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Round-to-nearest addition that keeps subnormals, as in non-fast-math CUDA C++."""
    return Float32(
        _inline_asm(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip)
    )


@dsl_user_op
def _mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Round-to-nearest multiplication that keeps subnormals, as in non-fast-math CUDA C++."""
    return Float32(
        _inline_asm(T.f32(), [Float32(a), Float32(b)], "mul.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip)
    )


@dsl_user_op
def _abs(value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_inline_asm(T.f32(), [Float32(value)], "abs.f32 $0, $1;", "=f,f", loc, ip))


@dsl_user_op
def _bits(value: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _from_bits(value: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(T.f32(), Int32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _rcp_approx_ftz(value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _inline_asm(T.f32(), [Float32(value)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", loc, ip)
    )


@dsl_user_op
def _to_e4m3(value: Float32, *, loc=None, ip=None) -> Uint32:
    """E4M3 code of ``value``, rounded to nearest and saturated to finite."""
    return Uint32(
        _inline_asm(
            T.i32(),
            [Float32(value)],
            "{\n.reg .b16 pair;\ncvt.rn.satfinite.e4m3x2.f32 pair, 0f00000000, $1;\n"
            "cvt.u32.u16 $0, pair;\n}",
            "=r,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _to_e4m3x4(v0, v1, v2, v3, *, loc=None, ip=None) -> Uint32:
    """Four E4M3 codes, rounded to nearest and saturated to finite; ``v0`` is the low byte."""
    return Uint32(
        _inline_asm(
            T.i32(),
            [Float32(v) for v in (v0, v1, v2, v3)],
            "{\n.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\nmov.b32 $0, {lo, hi};\n}",
            "=r,f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _e4m3_to_f32(code: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _inline_asm(
            T.f32(),
            [Uint32(code)],
            "{\n.reg .b16 pair, lo, hi;\n.reg .b32 halves;\ncvt.u16.u32 pair, $1;\n"
            "cvt.rn.f16x2.e4m3x2 halves, pair;\nmov.b32 {lo, hi}, halves;\ncvt.f32.f16 $0, lo;\n}",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _to_e2m1x8(v0, v1, v2, v3, v4, v5, v6, v7, *, loc=None, ip=None) -> Uint32:
    """Eight E2M1 values, rounded to nearest and saturated to finite; ``v0`` is the low nibble."""
    return Uint32(
        _inline_asm(
            T.i32(),
            [Float32(v) for v in (v0, v1, v2, v3, v4, v5, v6, v7)],
            "{\n.reg .b8 b0, b1, b2, b3;\ncvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;\ncvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;\nmov.b32 $0, {b0, b1, b2, b3};\n}",
            "=r,f,f,f,f,f,f,f,f",
            loc,
            ip,
        )
    )


class DSv4RouteNvfp4QuantKernel:
    """Route each token as ``gate_forward`` and NVFP4-quantize it as ``fp4_quantize``, bitwise;
    with ``routing_tiles`` one CTA routes every token and also writes TRTLLM-Gen's routing tables.
    ``staged`` writes into a launch bucket: ids also in the bucket's index type, and extra CTAs give
    the rows past the tokens expert -1 and weight 0. ``remap`` writes the bucket ids through an
    expert -> slot table, with ``replicas`` a table of copies (``replica_slot_table``). ``count``
    adds the routes of counted rows to a per-expert histogram (the warmup expert placement's rules:
    a flag, a live-row bound, with ``count_mask`` a row mask, and rows with all-zero logits
    skipped). ``mxfp8`` also writes each token as MXFP8. Without ``quant`` the launch only routes
    (an input the producer already quantized). With ``num_local`` the tables cover the experts
    ``[local_start, local_start + num_local)`` of an expert-parallel rank, numbered from 0. ``pdl``
    launches it as a programmatic dependent of the previous kernel. With ``splits`` > 1 the logits
    are K-chunk partials ``[splits, tokens, experts]``, summed s = 0 .. splits-1 as they load.
    """

    group_size = 16

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        hidden_size: int,
        routing_tiles: tuple[int, ...] = (),
        staged: bool = False,
        remap: bool = False,
        count: bool = False,
        count_mask: bool = False,
        mxfp8: bool = False,
        replicas: bool = False,
        quant: bool = True,
        num_local: Optional[int] = None,
        pdl: bool = False,
        cta_scoring: bool = True,
        splits: int = 1,
    ):
        warp = cute.arch.WARP_SIZE
        num_local = num_experts if num_local is None else num_local
        self.num_threads = num_experts if routing_tiles else 128
        if (
            num_experts % warp != 0
            or not 0 < top_k <= warp
            or hidden_size % self.group_size != 0
            or not 0 < num_local <= num_experts
            or (routing_tiles and num_experts > ROUTING_TABLES_MAX_EXPERTS)
            # A warp per token picks it.
            or (routing_tiles and num_experts < warp * ROUTING_TABLES_MAX_TOKENS)
            or (routing_tiles and (staged or count))
            or (remap and not staged)
            or (count_mask and not count)
            or (replicas and not remap)
        ):
            raise ValueError(
                f"expected experts divisible by {warp} ({warp * ROUTING_TABLES_MAX_TOKENS} to "
                f"{ROUTING_TABLES_MAX_EXPERTS} with routing tables), 0 < top_k <= {warp} and hidden "
                f"divisible by {self.group_size}; "
                f"got {num_experts}, {top_k}, {hidden_size}; routing tables exclude staged and "
                f"count, remap needs staged"
            )
        self.num_experts = num_experts
        self.num_local = num_local
        self.pdl = pdl
        self.splits = splits
        # The experts a thread scores for one token, and a lane holds when picking.
        self.experts_per_thread = (num_experts + self.num_threads - 1) // self.num_threads
        self.experts_per_lane = num_experts // warp
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.routing_tiles = routing_tiles
        # Plain bools for the DSL's compile-time branches (it rejects a tuple operand).
        self.has_tables = bool(routing_tiles)
        # Every warp of a token's CTA scores its experts (always with tables), else warp 0 alone.
        self.cta_scoring = self.has_tables or cta_scoring
        # Without the quantization CTAs, the routing CTA is the whole grid.
        self.tables_only = self.has_tables and not quant
        self.min_blocks = 2 if self.has_tables else CTAS_PER_SM
        self.scored_rows = ROUTING_TABLES_MAX_TOKENS if self.has_tables else 1
        self.groups = hidden_size // self.group_size
        self.quant = quant
        self.quant_ctas = (self.groups + self.num_threads - 1) // self.num_threads if quant else 0
        self.staged = staged
        self.remap = remap
        self.replicas = replicas
        self.count = count
        self.count_mask = count_mask
        # The two lanes of an MXFP8 block exchange their amax, so every warp's groups must exist.
        if mxfp8 and self.groups % warp != 0:
            raise ValueError(f"MXFP8 needs hidden divisible by {self.group_size * warp}")
        self.mxfp8 = mxfp8
        self.k_atoms = hidden_size // 128

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        x: Optional[cute.Tensor],
        global_scale: Optional[cute.Tensor],
        ids: cute.Tensor,
        weights: cute.Tensor,
        x_fp4: Optional[cute.Tensor],
        x_sf: Optional[cute.Tensor],
        tables: Optional[cute.Tensor],
        local_start: Int32,
        staged_ids: Optional[cute.Tensor],
        slot_of_expert: Optional[cute.Tensor],
        count_hist: Optional[cute.Tensor],
        count_flag: Optional[cute.Tensor],
        count_bound: Optional[cute.Tensor],
        count_rows: Optional[cute.Tensor],
        count_row_offset: Int32,
        x_fp8: Optional[cute.Tensor],
        x_sf8: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        rows = self.num_rows(logits)
        if cutlass.const_expr(self.staged):
            rows = rows + (staged_ids.shape[0] - rows + TAIL_ROWS_PER_CTA - 1) // TAIL_ROWS_PER_CTA
        if cutlass.const_expr(self.tables_only):
            rows = 1
        self.kernel(
            logits,
            bias,
            x,
            global_scale,
            ids,
            weights,
            x_fp4,
            x_sf,
            tables,
            local_start,
            staged_ids,
            slot_of_expert,
            count_hist,
            count_flag,
            count_bound,
            count_rows,
            count_row_offset,
            x_fp8,
            x_sf8,
        ).launch(
            grid=[rows, 1 + self.quant_ctas, 1],
            block=[self.num_threads, 1, 1],
            min_blocks_per_mp=self.min_blocks,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        x: Optional[cute.Tensor],
        global_scale: Optional[cute.Tensor],
        ids: cute.Tensor,
        weights: cute.Tensor,
        x_fp4: Optional[cute.Tensor],
        x_sf: Optional[cute.Tensor],
        tables: Optional[cute.Tensor],
        local_start: Int32,
        staged_ids: Optional[cute.Tensor],
        slot_of_expert: Optional[cute.Tensor],
        count_hist: Optional[cute.Tensor],
        count_flag: Optional[cute.Tensor],
        count_bound: Optional[cute.Tensor],
        count_rows: Optional[cute.Tensor],
        count_row_offset: Int32,
        x_fp8: Optional[cute.Tensor],
        x_sf8: Optional[cute.Tensor],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, part, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.pdl):
            # Nothing global is touched before the previous kernel completes: an earlier layer's
            # finalize may still read buffers this launch reuses. The dependent (the expert GEMMs,
            # hundreds of CTAs) is released only at exit, so it does not hold SMs the shared
            # expert's concurrent GEMMs need while this launch routes.
            griddepcontrol_wait()
        if cutlass.const_expr(self.cta_scoring):
            # Every token's (routing tables) or this CTA's token's selection keys and unbiased
            # scores.
            smem = SmemAllocator()
            scored = self.scored_rows
            keys_s = smem.allocate_tensor(Int32, cute.make_layout(scored * self.num_experts))
            scores_s = smem.allocate_tensor(Float32, cute.make_layout(scored * self.num_experts))
            if cutlass.const_expr(self.has_tables):
                # The local expert histogram (then one tile's first CTAs) and each warp's CTA
                # counts.
                experts_s = smem.allocate_tensor(Int32, cute.make_layout(self.num_local))
                warp_ctas_s = smem.allocate_tensor(
                    Int32,
                    cute.make_layout(
                        (len(self.routing_tiles), self.num_threads // cute.arch.WARP_SIZE)
                    ),
                )
            nonzero_s = smem.allocate_tensor(
                Int32, cute.make_layout(self.num_threads // cute.arch.WARP_SIZE)
            )
        if part == 0:
            if cutlass.const_expr(self.has_tables):
                if row == 0:
                    self.route_with_tables(
                        logits,
                        bias,
                        ids,
                        weights,
                        tables,
                        local_start,
                        keys_s,
                        scores_s,
                        experts_s,
                        warp_ctas_s,
                        tidx,
                    )
            elif row < self.num_rows(logits):
                if cutlass.const_expr(self.cta_scoring):
                    self.route(
                        logits,
                        bias,
                        ids,
                        weights,
                        row,
                        tidx,
                        keys_s,
                        scores_s,
                        nonzero_s,
                        staged_ids,
                        slot_of_expert,
                        count_hist,
                        count_flag,
                        count_bound,
                        count_rows,
                        count_row_offset,
                    )
                else:
                    if tidx < cute.arch.WARP_SIZE:
                        self.route_warp(
                            logits,
                            bias,
                            ids,
                            weights,
                            row,
                            tidx,
                            staged_ids,
                            slot_of_expert,
                            count_hist,
                            count_flag,
                            count_bound,
                            count_rows,
                            count_row_offset,
                        )
            elif cutlass.const_expr(self.staged):
                self.clear_tail(self.num_rows(logits), staged_ids, weights, row, tidx)
        else:
            if cutlass.const_expr(self.quant):
                if row < x.shape[0]:
                    group = (part - 1) * self.num_threads + tidx
                    if group < self.groups:
                        self.quantize(x, global_scale, x_fp4, x_sf, row, group, x_fp8, x_sf8)

    def num_rows(self, logits: cute.Tensor):
        """The tokens of ``logits`` (or of their K-chunk partials)."""
        return logits.shape[0] if self.splits == 1 else logits.shape[1]

    @cute.jit
    def logit(self, logits: cute.Tensor, row: Int32, expert: Int32) -> Float32:
        """Token ``row``'s logit of ``expert``; from partials their sum, s = 0 .. splits-1 in order."""
        total = Float32(0.0)
        if cutlass.const_expr(self.splits == 1):
            total = Float32(logits[row, expert])
        else:
            total = Float32(logits[0, row, expert])
            for s in cutlass.range_constexpr(1, self.splits):
                total = _add_rn(total, Float32(logits[s, row, expert]))
        return total

    @cute.jit
    def score(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        row: Int32,
        expert: Int32,
        keys_s: cute.Tensor,
        scores_s: cute.Tensor,
        slot: Int32,
    ) -> Int32:
        """Write ``expert``'s selection key and unbiased score for token ``row`` at ``slot``;
        returns 1 if its logit is non-zero."""
        return self.score_logit(
            self.logit(logits, row, expert), Float32(bias[expert]), keys_s, scores_s, slot
        )

    @cute.jit
    def score_logit(
        self,
        logit: Float32,
        bias: Float32,
        keys_s: cute.Tensor,
        scores_s: cute.Tensor,
        slot: Int32,
    ) -> Int32:
        """:meth:`score` for a loaded logit and selection bias."""
        softplus = cute.math.log1p(cute.math.exp(logit))
        if logit > Float32(20.0):
            softplus = logit
        score = cute.math.sqrt(softplus, rounding=cute.math.RoundingMode.NEAREST_EVEN)
        scores_s[slot] = score
        keys_s[slot] = _ordered(_add_rn(score, bias))
        nonzero = Int32(0)
        if logit != Float32(0.0):
            nonzero = Int32(1)
        return nonzero

    @cute.jit
    def route(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        row: Int32,
        tidx: Int32,
        keys_s: cute.Tensor,
        scores_s: cute.Tensor,
        nonzero_s: cute.Tensor,
        staged_ids: Optional[cute.Tensor] = None,
        slot_of_expert: Optional[cute.Tensor] = None,
        count_hist: Optional[cute.Tensor] = None,
        count_flag: Optional[cute.Tensor] = None,
        count_bound: Optional[cute.Tensor] = None,
        count_rows: Optional[cute.Tensor] = None,
        count_row_offset: Int32 = 0,
    ):
        """Route token ``row`` with the CTA: every thread scores some experts, warp 0 picks."""
        warp_size = cute.arch.WARP_SIZE
        nonzero = Int32(0)
        for v in cutlass.range_constexpr(self.experts_per_thread):
            expert = tidx + v * self.num_threads
            if cutlass.const_expr((v + 1) * self.num_threads <= self.num_experts):
                flag = self.score(logits, bias, row, expert, keys_s, scores_s, expert)
                if flag != Int32(0):
                    nonzero = Int32(1)
            else:
                if expert < self.num_experts:
                    flag = self.score(logits, bias, row, expert, keys_s, scores_s, expert)
                    if flag != Int32(0):
                        nonzero = Int32(1)
        if cutlass.const_expr(self.count):
            nonzero = cute.arch.warp_redux_sync(nonzero, "max")
            if tidx % warp_size == 0:
                nonzero_s[tidx // warp_size] = nonzero
        cute.arch.barrier()
        if tidx < warp_size:
            if cutlass.const_expr(self.count):
                for w in cutlass.range_constexpr(self.num_threads // warp_size):
                    if nonzero_s[w] != Int32(0):
                        nonzero = Int32(1)
            self.pick(
                keys_s,
                scores_s,
                0,
                ids,
                weights,
                row,
                tidx,
                nonzero,
                staged_ids,
                slot_of_expert,
                count_hist,
                count_flag,
                count_bound,
                count_rows,
                count_row_offset,
            )

    @cute.jit
    def pick(
        self,
        keys_s: cute.Tensor,
        scores_s: cute.Tensor,
        base: Int32,
        ids: cute.Tensor,
        weights: cute.Tensor,
        row: Int32,
        lane: Int32,
        nonzero: Int32,
        staged_ids: Optional[cute.Tensor] = None,
        slot_of_expert: Optional[cute.Tensor] = None,
        count_hist: Optional[cute.Tensor] = None,
        count_flag: Optional[cute.Tensor] = None,
        count_bound: Optional[cute.Tensor] = None,
        count_rows: Optional[cute.Tensor] = None,
        count_row_offset: Int32 = 0,
    ) -> Int32:
        """Pick token ``row``'s experts with one warp from the keys and scores at ``base``; lane
        ``k < top_k`` returns pick ``k``. ``nonzero``: some logit of the row is non-zero."""
        warp = cute.arch.WARP_SIZE
        taken = Int32(-(2**31))
        keys = [keys_s[base + lane + v * warp] for v in range(self.experts_per_lane)]
        picked = []
        for k in cutlass.range_constexpr(self.top_k):
            # The lane's best candidate; its experts ascend with v, so ">" keeps the lowest.
            best = keys[0]
            best_expert = lane
            for v in cutlass.range_constexpr(1, self.experts_per_lane):
                if keys[v] > best:
                    best = keys[v]
                    best_expert = lane + v * warp
            top = cute.arch.warp_redux_sync(best, "max")
            # Among the lanes holding the top key, the lowest expert (largest num_experts - e).
            remaining = Int32(0)
            if best == top:
                remaining = self.num_experts - best_expert
            expert = self.num_experts - cute.arch.warp_redux_sync(remaining, "max")
            picked.append(expert)
            for v in cutlass.range_constexpr(self.experts_per_lane):
                if lane + v * warp == expert:
                    keys[v] = taken
        expert = Int32(0)
        for k in cutlass.range_constexpr(self.top_k):
            if lane == k:
                expert = picked[k]
        # Lane k holds pick k's unbiased score, the other lanes zero, as gate_forward's sum.
        score = Float32(0.0)
        if lane < self.top_k:
            score = scores_s[base + expert]
        return self.store_picks(
            ids,
            weights,
            row,
            lane,
            expert,
            score,
            nonzero,
            staged_ids,
            slot_of_expert,
            count_hist,
            count_flag,
            count_bound,
            count_rows,
            count_row_offset,
        )

    @cute.jit
    def route_warp(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        row: Int32,
        lane: Int32,
        staged_ids: Optional[cute.Tensor] = None,
        slot_of_expert: Optional[cute.Tensor] = None,
        count_hist: Optional[cute.Tensor] = None,
        count_flag: Optional[cute.Tensor] = None,
        count_bound: Optional[cute.Tensor] = None,
        count_rows: Optional[cute.Tensor] = None,
        count_row_offset: Int32 = 0,
    ) -> Int32:
        """Route token ``row`` with one warp, its scores in registers; lane ``k < top_k`` returns
        pick ``k``."""
        warp = cute.arch.WARP_SIZE
        taken = Int32(-(2**31))
        scores = []
        keys = []
        nonzero = Int32(0)
        for v in cutlass.range_constexpr(self.experts_per_lane):
            expert = lane + v * warp
            logit = self.logit(logits, row, expert)
            if cutlass.const_expr(self.count):
                if logit != Float32(0.0):
                    nonzero = Int32(1)
            softplus = cute.math.log1p(cute.math.exp(logit))
            if logit > Float32(20.0):
                softplus = logit
            score = cute.math.sqrt(softplus, rounding=cute.math.RoundingMode.NEAREST_EVEN)
            scores.append(score)
            keys.append(_ordered(_add_rn(score, Float32(bias[expert]))))
        picked = []
        for k in cutlass.range_constexpr(self.top_k):
            # The lane's best candidate; its experts ascend with v, so ">" keeps the lowest.
            best = keys[0]
            best_expert = lane
            for v in cutlass.range_constexpr(1, self.experts_per_lane):
                if keys[v] > best:
                    best = keys[v]
                    best_expert = lane + v * warp
            top = cute.arch.warp_redux_sync(best, "max")
            # Among the lanes holding the top key, the lowest expert (largest num_experts - e).
            remaining = Int32(0)
            if best == top:
                remaining = self.num_experts - best_expert
            expert = self.num_experts - cute.arch.warp_redux_sync(remaining, "max")
            picked.append(expert)
            for v in cutlass.range_constexpr(self.experts_per_lane):
                if lane + v * warp == expert:
                    keys[v] = taken
        # Lane k holds pick k's unbiased score, the other lanes zero, as gate_forward's sum.
        score = Float32(0.0)
        expert = Int32(0)
        for k in cutlass.range_constexpr(self.top_k):
            owned = scores[0]
            for v in cutlass.range_constexpr(1, self.experts_per_lane):
                if picked[k] // warp == v:
                    owned = scores[v]
            owned = cute.arch.shuffle_sync(owned, offset=picked[k] % warp)
            if lane == k:
                score = owned
                expert = picked[k]
        return self.store_picks(
            ids,
            weights,
            row,
            lane,
            expert,
            score,
            nonzero,
            staged_ids,
            slot_of_expert,
            count_hist,
            count_flag,
            count_bound,
            count_rows,
            count_row_offset,
        )

    @cute.jit
    def store_picks(
        self,
        ids: cute.Tensor,
        weights: cute.Tensor,
        row: Int32,
        lane: Int32,
        expert: Int32,
        score: Float32,
        nonzero: Int32,
        staged_ids: Optional[cute.Tensor],
        slot_of_expert: Optional[cute.Tensor],
        count_hist: Optional[cute.Tensor],
        count_flag: Optional[cute.Tensor],
        count_bound: Optional[cute.Tensor],
        count_rows: Optional[cute.Tensor],
        count_row_offset: Int32,
    ) -> Int32:
        """Write token ``row``'s picks from the warp whose lane ``k < top_k`` holds pick ``k`` and
        its unbiased score (the other lanes zero): ids, staged ids, weights and counts."""
        total = score
        for offset in (16, 8, 4, 2, 1):
            total = _add_rn(total, cute.arch.shuffle_sync_bfly(total, offset=offset))
        counted = Int32(0)
        if cutlass.const_expr(self.count):
            counted = self.counts_row(
                row, nonzero, count_flag, count_bound, count_rows, count_row_offset
            )
        if lane < self.top_k:
            ids[row, lane] = expert
            if cutlass.const_expr(self.count):
                if counted != Int32(0):
                    cute.arch.atomic_add(
                        count_hist.iterator + expert, Int32(1), sem="relaxed", scope="gpu"
                    )
            if cutlass.const_expr(self.staged):
                bucket_id = expert
                if cutlass.const_expr(self.replicas):
                    # Copy h(row) % copies of the expert, h(row) = (row * 0x9E3779B1 mod 2^32) >> 16.
                    meta = Int32(slot_of_expert[expert])
                    copies = meta >> Int32(16)
                    hashed = (Int64(row) * Int64(0x9E3779B1)) & Int64(0xFFFFFFFF)
                    copy = Int32((hashed >> Int64(16)) % Int64(copies))
                    bucket_id = Int32(
                        slot_of_expert[self.num_experts + (meta & Int32(0xFFFF)) + copy]
                    )
                elif cutlass.const_expr(self.remap):
                    bucket_id = Int32(slot_of_expert[expert])
                staged_ids[row, lane] = bucket_id.to(staged_ids.element_type)
            weight = cute.math.div(
                score,
                _add_rn(total, Float32(1e-20)),
                rounding=cute.math.RoundingMode.NEAREST_EVEN,
            )
            weight = _mul_rn(weight, Float32(self.routed_scaling_factor))
            weights[row, lane] = weight.to(weights.element_type)
        return expert

    @cute.jit
    def counts_row(
        self,
        row: Int32,
        nonzero: Int32,
        count_flag: cute.Tensor,
        count_bound: cute.Tensor,
        count_rows: Optional[cute.Tensor],
        count_row_offset: Int32,
    ) -> Int32:
        """Whether the warp's row counts: the flag is set, some lane saw a non-zero logit, the row
        is below the live-row bound (-1: none) and, with a row mask, its byte is set."""
        counted = cute.arch.warp_redux_sync(nonzero, "max")
        if Int32(count_flag[0]) == Int32(0):
            counted = Int32(0)
        g = row + count_row_offset
        bound = Int32(count_bound[0])
        if bound >= Int32(0):
            if g >= bound:
                counted = Int32(0)
        if cutlass.const_expr(self.count_mask):
            if g < count_rows.shape[0]:
                if Int32(count_rows[g]) == Int32(0):
                    counted = Int32(0)
            else:
                counted = Int32(0)
        return counted

    @cute.jit
    def route_with_tables(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        tables: cute.Tensor,
        local_start: Int32,
        keys_s: cute.Tensor,
        scores_s: cute.Tensor,
        experts: cute.Tensor,
        warp_ctas: cute.Tensor,
        tidx: Int32,
    ):
        """Route every token (thread ``e`` scores expert ``e`` of every token, then warp ``t``
        picks token ``t``), then write ``tables[t]`` for tile ``t``: expanded-to-permuted,
        permuted-to-token, CTA experts, CTA limits, (CTAs, padded rows). Experts are numbered
        from ``local_start``; a pick of another rank's expert maps to -1. ``experts`` holds the
        local expert histogram, then one tile's first CTAs."""
        warp_size = cute.arch.WARP_SIZE
        num_warps = self.num_threads // warp_size
        warp = tidx // warp_size
        lane = tidx % warp_size
        num_tokens = self.num_rows(logits)
        # Every token's logit is loaded before any is scored, so the loads overlap; rows past the
        # tokens load the last token's.
        token_logits = []
        for r in cutlass.range_constexpr(ROUTING_TABLES_MAX_TOKENS):
            token = Int32(r)
            if token >= num_tokens:
                token = num_tokens - 1
            token_logits.append(self.logit(logits, token, tidx))
        expert_bias = Float32(bias[tidx])
        for r in cutlass.range_constexpr(ROUTING_TABLES_MAX_TOKENS):
            if r < num_tokens:
                self.score_logit(
                    token_logits[r], expert_bias, keys_s, scores_s, r * self.num_experts + tidx
                )
        for e in cutlass.range(tidx, self.num_local, self.num_threads):
            experts[e] = Int32(0)
        cute.arch.barrier()
        # Lane k < top_k keeps its token's pick k: the local expert and its row in the expert's
        # block, or rank -1 for another rank's expert.
        is_pick = Int32(0)
        pick_expert = Int32(0)
        pick_rank = Int32(-1)
        if warp < num_tokens:
            expert = self.pick(
                keys_s, scores_s, warp * self.num_experts, ids, weights, warp, lane, 0
            )
            if lane < self.top_k:
                is_pick = Int32(1)
                local = expert - local_start
                if local >= Int32(0):
                    if local < Int32(self.num_local):
                        pick_expert = local
                        # An expert's rows come in atomic order; the GEMMs and finalize do not
                        # depend on it.
                        pick_rank = cute.arch.atomic_add(
                            experts.iterator + local, Int32(1), scope="cta"
                        )
        cute.arch.barrier()
        # Thread tidx owns local expert tidx (one thread per expert).
        count = Int32(0)
        if tidx < self.num_local:
            count = experts[tidx]
        # All tiles at once: each expert's CTA count, scanned within the warp, then over warps.
        owned = [(count + tile - 1) // tile for tile in self.routing_tiles]
        scanned = list(owned)
        for offset in (1, 2, 4, 8, 16):
            for t in cutlass.range_constexpr(len(self.routing_tiles)):
                lower = cute.arch.shuffle_sync_up(scanned[t], offset)
                if lane >= offset:
                    scanned[t] = scanned[t] + lower
        if lane == warp_size - 1:
            for t in cutlass.range_constexpr(len(self.routing_tiles)):
                warp_ctas[t, warp] = scanned[t]
        cute.arch.barrier()
        for t in cutlass.range_constexpr(len(self.routing_tiles)):
            tile = self.routing_tiles[t]
            first = scanned[t] - owned[t]
            total = Int32(0)
            for w in cutlass.range_constexpr(num_warps):
                if w < warp:
                    first = first + warp_ctas[t, w]
                total = total + warp_ctas[t, w]
            if tidx < self.num_local:
                experts[tidx] = first
                for c in cutlass.range(owned[t]):
                    cta = first + c
                    limit = (cta + 1) * tile
                    end = first * tile + count
                    if end < limit:
                        limit = end
                    tables[t, 2, cta] = tidx
                    tables[t, 3, cta] = limit
            if tidx == 0:
                tables[t, 4, 0] = total
                tables[t, 4, 1] = total * tile
            cute.arch.barrier()
            pick = warp * self.top_k + lane
            if pick_rank >= 0:
                permuted = experts[pick_expert] * tile + pick_rank
                tables[t, 0, pick] = permuted
                tables[t, 1, permuted] = warp
            else:
                if is_pick != Int32(0):
                    tables[t, 0, pick] = Int32(-1)
            cute.arch.barrier()

    @cute.jit
    def clear_tail(
        self,
        num_tokens: Int32,
        staged_ids: cute.Tensor,
        weights: cute.Tensor,
        row: Int32,
        tidx: Int32,
    ):
        """Expert -1 and weight 0 for the bucket rows of tail CTA ``row - num_tokens``."""
        first = num_tokens + (row - num_tokens) * TAIL_ROWS_PER_CTA
        items = TAIL_ROWS_PER_CTA * self.top_k
        for step in cutlass.range_constexpr((items + self.num_threads - 1) // self.num_threads):
            item = tidx + step * self.num_threads
            tail_row = first + item // self.top_k
            if item < items:
                if tail_row < staged_ids.shape[0]:
                    staged_ids[tail_row, item % self.top_k] = staged_ids.element_type(-1)
                    weights[tail_row, item % self.top_k] = weights.element_type(0.0)

    @cute.jit
    def quantize(
        self,
        x: cute.Tensor,
        global_scale: cute.Tensor,
        x_fp4: cute.Tensor,
        x_sf: cute.Tensor,
        row: Int32,
        group: Int32,
        x_fp8: Optional[cute.Tensor],
        x_sf8: Optional[cute.Tensor],
    ):
        scale = Float32(global_scale[0])
        values = cute.make_rmem_tensor((self.group_size,), x.element_type)
        cute.autovec_copy(cute.zipped_divide(x[row, None], (self.group_size,))[None, group], values)
        elements = [values[i].to(Float32) for i in range(self.group_size)]
        amax = Float32(0.0)
        for element in elements:
            amax = cute.arch.fmax(amax, _abs(element))
        code = _to_e4m3(_mul_rn(scale, _mul_rn(amax, _rcp_approx_ftz(Float32(6.0)))))
        out_scale = Float32(0.0)
        if _bits(amax) != Int32(0):
            out_scale = _rcp_approx_ftz(_mul_rn(_e4m3_to_f32(code), _rcp_approx_ftz(scale)))
        scaled = [_mul_rn(element, out_scale) for element in elements]
        for word in cutlass.range_constexpr(self.group_size // 8):
            x_fp4[row, 2 * group + word] = _to_e2m1x8(*scaled[8 * word : 8 * word + 8])
        x_sf[row, group] = Uint8(code)
        if cutlass.const_expr(self.mxfp8):
            # This group's half of an MXFP8 block: E4M3 data, and a UE8M0 scale rounded up like
            # cvt.rp.ue8m0, in the 128x4 scale-atom layout of the MegaMoE MMA.
            amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
            bits = _bits(_mul_rn(amax, Float32(1.0 / 448.0)))
            exponent = bits >> 23
            if (bits & Int32(0x7FFFFF)) != Int32(0):
                exponent = exponent + Int32(1)
            if exponent > Int32(254):
                exponent = Int32(254)
            # 2**(127 - exponent): the exact reciprocal of the decode scale.
            encode = _from_bits((Int32(254) - exponent) << 23)
            scaled8 = [_mul_rn(element, encode) for element in elements]
            for word in cutlass.range_constexpr(self.group_size // 4):
                x_fp8[row, 4 * group + word] = _to_e4m3x4(*scaled8[4 * word : 4 * word + 4])
            if group % 2 == 0:
                block = group // 2
                x_sf8[
                    (row // 128) * (self.k_atoms * 512)
                    + (block // 4) * 512
                    + (row % 32) * 16
                    + ((row // 32) % 4) * 4
                    + block % 4
                ] = Uint8(exponent)


_WEIGHT_TYPES = {torch.float32: Float32, torch.bfloat16: cutlass.BFloat16}
_INDEX_TYPES = {torch.int32: Int32, torch.int64: Int64}


@functools.cache
def _compile(
    num_experts: int,
    top_k: int,
    routed_scaling_factor: float,
    hidden_size: int,
    weights_dtype: torch.dtype,
    routing_tiles: tuple[int, ...],
    staged_index_dtype: Optional[torch.dtype] = None,
    remap: bool = False,
    count: bool = False,
    count_mask: bool = False,
    mxfp8: bool = False,
    replicas: bool = False,
    quant: bool = True,
    num_local: Optional[int] = None,
    pdl: bool = False,
    cta_scoring: bool = True,
    splits: int = 1,
):
    tokens = cute.sym_int()
    # A launch bucket has its own row count, at least the tokens'.
    bucket = cute.sym_int() if staged_index_dtype is not None else tokens

    def rows(dtype, width, count=tokens, **kwargs):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (count, width), stride_order=(1, 0), **kwargs
        )

    tables = None
    if routing_tiles:
        tables = cute.runtime.make_fake_compact_tensor(
            Int32, (len(routing_tiles), 5, cute.sym_int()), stride_order=(2, 1, 0)
        )
    staged_ids = None
    if staged_index_dtype is not None:
        staged_ids = rows(_INDEX_TYPES[staged_index_dtype], top_k, bucket)

    def per_expert(enabled):
        if not enabled:
            return None
        return cute.runtime.make_fake_compact_tensor(Int32, (num_experts,), stride_order=(0,))

    def scalar(enabled):
        if not enabled:
            return None
        return cute.runtime.make_fake_compact_tensor(Int32, (1,), stride_order=(0,))

    count_rows = None
    if count_mask:
        count_rows = cute.runtime.make_fake_compact_tensor(
            cutlass.Int8, (cute.sym_int(),), stride_order=(0,)
        )

    x_fp8 = x_sf8 = None
    if mxfp8:
        x_fp8 = rows(Uint32, hidden_size // 4, cute.sym_int())
        x_sf8 = cute.runtime.make_fake_compact_tensor(Uint8, (cute.sym_int(),), stride_order=(0,))
    logits = rows(Float32, num_experts)
    if splits > 1:
        # K-chunk partials: rows of a partial are contiguous, partials sit a row count apart.
        logits = cute.runtime.make_fake_tensor(
            Float32,
            (splits, tokens, num_experts),
            stride=(cute.sym_int64(divisibility=1), num_experts, 1),
        )
    return cute.compile(
        DSv4RouteNvfp4QuantKernel(
            num_experts,
            top_k,
            routed_scaling_factor,
            hidden_size,
            routing_tiles,
            staged=staged_index_dtype is not None,
            remap=remap,
            count=count,
            count_mask=count_mask,
            mxfp8=mxfp8,
            replicas=replicas,
            quant=quant,
            num_local=num_local,
            pdl=pdl,
            cta_scoring=cta_scoring,
            splits=splits,
        ),
        logits,
        cute.runtime.make_fake_compact_tensor(Float32, (num_experts,), stride_order=(0,)),
        rows(cutlass.BFloat16, hidden_size, assumed_align=16) if quant else None,
        cute.runtime.make_fake_compact_tensor(Float32, (1,), stride_order=(0,)) if quant else None,
        rows(Int32, top_k),
        rows(_WEIGHT_TYPES[weights_dtype], top_k, bucket),
        rows(Uint32, hidden_size // 8, bucket) if quant else None,
        rows(Uint8, hidden_size // 16, bucket) if quant else None,
        tables,
        Int32(0),
        staged_ids,
        (
            cute.runtime.make_fake_compact_tensor(Int32, (cute.sym_int(),), stride_order=(0,))
            if replicas
            else per_expert(remap)
        ),
        per_expert(count),
        scalar(count),
        scalar(count),
        count_rows,
        Int32(0),
        x_fp8,
        x_sf8,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def can_build_routing_tables(num_tokens: int, num_experts: int, total_experts: int) -> bool:
    """Whether ``dsv4_route_nvfp4_quant`` can also write routing tables for this shape, with
    ``num_experts`` the experts the tables cover out of ``total_experts``."""
    return (
        0 < num_tokens <= ROUTING_TABLES_MAX_TOKENS
        and num_experts <= total_experts
        # A thread per expert, and a warp per token.
        and cute.arch.WARP_SIZE * ROUTING_TABLES_MAX_TOKENS
        <= total_experts
        <= ROUTING_TABLES_MAX_EXPERTS
    )


def _max_ctas(num_tokens: int, top_k: int, num_experts: int, tile: int) -> int:
    """TRTLLM-Gen's getMaxNumCtasInBatchDim: one CTA per routed expert, plus one per full tile."""
    expanded = num_tokens * top_k
    filled = min(num_experts, expanded)
    return filled + (expanded - filled) // tile


def _local_range(local_experts: Optional[tuple[int, int]], num_experts: int) -> tuple[int, int]:
    first, count = (0, num_experts) if local_experts is None else local_experts
    if not (0 <= first and 0 < count and first + count <= num_experts):
        raise ValueError(f"local experts [{first}, {first + count}) not in [0, {num_experts})")
    return first, count


def _routing_tables(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device,
    total_experts: int,
    tiles: tuple[int, ...] = ROUTING_TABLE_TILES,
) -> torch.Tensor:
    """The buffer of ``tiles``' tables for one launch over ``num_experts`` local experts."""
    if not can_build_routing_tables(num_tokens, num_experts, total_experts):
        raise ValueError(f"no routing tables for {num_tokens} tokens over {num_experts} experts")
    if not tiles or not set(tiles) <= set(ROUTING_TABLE_TILES):
        raise ValueError(f"routing tables are for tiles of {ROUTING_TABLE_TILES}, not {tiles}")
    # Route maps are read one past their padded rows (getRouteMapAllocCount).
    width = max(
        [num_tokens * top_k] + [_max_ctas(num_tokens, top_k, num_experts, t) * t + 1 for t in tiles]
    )
    return torch.empty((len(tiles), 5, width), dtype=torch.int32, device=device)


def dsv4_route_nvfp4_quant(
    router_logits: Union[torch.Tensor, "RouterLogitPartials"],  # noqa: F821
    selection_bias: torch.Tensor,
    x: torch.Tensor,
    global_scale: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    weights_dtype: torch.dtype,
    with_routing_tables: bool = False,
    staged: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    slot_of_expert: Optional[torch.Tensor] = None,
    counters: Optional[tuple] = None,
    mxfp8: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    replicas: bool = False,
    local_experts: Optional[tuple[int, int]] = None,
    routing_tiles: tuple[int, ...] = ROUTING_TABLE_TILES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """DeepSeek-V4 routing and NVFP4 input quantization in one launch, bitwise ``routing.apply``
    then ``fp4_quantize``; ``with_routing_tables`` also returns ``run_moe``'s routing tables for
    ``routing_tiles``, over the experts ``local_experts`` = (first, count) of an expert-parallel
    rank if given.

    ``staged`` = a launch bucket's (ids, weights, NVFP4 rows, scale rows) with at least as many rows
    as tokens: the launch also writes the ids there and writes the rest there instead, returning
    their token rows; the bucket's other rows get expert -1 and weight 0.

    ``slot_of_expert`` (int32 [num_experts], with ``staged``) writes the bucket ids as expert slots;
    the returned ids stay expert ids; with ``replicas`` it is a ``replica_slot_table`` and a pick in
    row r goes to copy ``h(r) % copies`` of its expert. ``counters`` = (histogram int32 [num_experts], flag int32
    [1], live-row bound int32 [1], row mask int8 or None, the launch's first row in the forward):
    the warmup expert placement's ``RouteCounters``, which get the picks of the counted rows.

    ``mxfp8`` = (E4M3 rows [>= tokens, hidden], UE8M0 scales in 128x4 atoms) also gets each token as
    MXFP8, exactly as ``mxfp8_quantize_megamoe`` writes it.

    ``router_logits`` may be ``RouterLogitPartials``: the launch routes on their sum, s = 0 .. S-1.
    """
    from tensorrt_llm._torch.custom_ops.router_logit_partials import RouterLogitPartials

    logits, splits = router_logits, 1
    if isinstance(router_logits, RouterLogitPartials):
        logits, splits = router_logits.parts, router_logits.parts.shape[0]
        if logits.stride(1) != logits.shape[2]:
            raise ValueError("router logit partials need each row contiguous over the experts")
    num_tokens, num_experts = logits.shape[-2:]
    hidden_size = x.shape[1]
    device = x.device
    ids = torch.empty((num_tokens, top_k), dtype=torch.int32, device=device)
    staged_ids = staged_index_dtype = None
    if staged is None:
        weights = torch.empty((num_tokens, top_k), dtype=weights_dtype, device=device)
        x_fp4 = torch.empty((num_tokens, hidden_size // 8), dtype=torch.int32, device=device)
        x_sf = torch.empty((num_tokens, hidden_size // 16), dtype=torch.uint8, device=device)
    else:
        staged_ids, weights, x_fp4, x_sf = staged
        x_fp4 = x_fp4.view(torch.int32)
        staged_index_dtype = staged_ids.dtype
        if with_routing_tables or not 0 < num_tokens <= staged_ids.shape[0]:
            raise ValueError(f"cannot stage {num_tokens} tokens into {staged_ids.shape[0]} rows")
    count_hist = count_flag = count_bound = count_rows = None
    count_row_offset = 0
    if counters is not None:
        count_hist, count_flag, count_bound, count_rows, count_row_offset = counters
    checked = (("counters", count_hist),)
    if not replicas:
        checked += (("slot_of_expert", slot_of_expert),)
    elif slot_of_expert is None or slot_of_expert.numel() <= num_experts:
        raise ValueError("replicas need a replica table")
    for name, table in checked:
        if table is not None and (
            table.dtype != torch.int32 or table.shape != (num_experts,) or not table.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous int32 [{num_experts}] tensor")
    if slot_of_expert is not None and staged is None:
        raise ValueError("slot_of_expert maps the staged ids; pass staged")
    if counters is not None and with_routing_tables:
        raise ValueError("counters are not supported with routing tables")
    x_fp8 = x_sf8 = None
    if mxfp8 is not None:
        x_fp8, x_sf8 = mxfp8
        if (
            x_fp8.shape[0] < num_tokens
            or x_fp8.shape[1] != hidden_size
            or not x_fp8.is_contiguous()
            or x_sf8.numel() < (num_tokens + 127) // 128 * 128 * hidden_size // 32
        ):
            raise ValueError(f"MXFP8 buffers are too small for {num_tokens} rows of {hidden_size}")
        x_fp8 = x_fp8.view(torch.int32)
    local_start, num_local = _local_range(local_experts, num_experts)
    tables = (
        _routing_tables(num_tokens, num_local, top_k, device, num_experts, routing_tiles)
        if with_routing_tables
        else None
    )
    routing_tiles = routing_tiles if with_routing_tables else ()
    if num_tokens > 0:
        _compile(
            num_experts,
            top_k,
            routed_scaling_factor,
            hidden_size,
            weights_dtype,
            routing_tiles,
            staged_index_dtype,
            slot_of_expert is not None,
            counters is not None,
            count_rows is not None,
            mxfp8 is not None,
            replicas,
            num_local=num_local,
            cta_scoring=_cta_scoring(num_tokens, num_experts),
            splits=splits,
        )(
            logits,
            selection_bias,
            x,
            global_scale.view(1),
            ids,
            weights,
            x_fp4,
            x_sf,
            tables,
            local_start,
            staged_ids,
            slot_of_expert,
            count_hist,
            count_flag,
            count_bound,
            count_rows,
            int(count_row_offset),
            x_fp8,
            x_sf8,
        )
    if staged is not None:
        rows = slice(0, num_tokens)
        return ids, weights[rows], x_fp4[rows].view(torch.uint8), x_sf[rows], None
    return ids, weights, x_fp4.view(torch.uint8), x_sf, tables


def dsv4_route(
    router_logits: torch.Tensor,
    selection_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    weights_dtype: torch.dtype,
    hidden_size: int,
    with_routing_tables: bool = False,
    local_experts: Optional[tuple[int, int]] = None,
    routing_tiles: tuple[int, ...] = ROUTING_TABLE_TILES,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """DeepSeek-V4 routing in one launch, bitwise ``routing.apply`` with the weights in
    ``weights_dtype``, for an input already quantized; ``with_routing_tables`` also returns
    ``run_moe``'s routing tables for ``routing_tiles`` (as ``dsv4_route_nvfp4_quant``). It launches
    as a programmatic dependent of the router GEMM."""
    if not isinstance(router_logits, torch.Tensor):
        raise TypeError("dsv4_route takes [tokens, experts] router logits")
    num_tokens, num_experts = router_logits.shape
    device = router_logits.device
    ids = torch.empty((num_tokens, top_k), dtype=torch.int32, device=device)
    weights = torch.empty((num_tokens, top_k), dtype=weights_dtype, device=device)
    local_start, num_local = _local_range(local_experts, num_experts)
    tables = (
        _routing_tables(num_tokens, num_local, top_k, device, num_experts, routing_tiles)
        if with_routing_tables
        else None
    )
    routing_tiles = routing_tiles if with_routing_tables else ()
    if num_tokens > 0:
        _compile(
            num_experts,
            top_k,
            routed_scaling_factor,
            hidden_size,
            weights_dtype,
            routing_tiles,
            quant=False,
            num_local=num_local,
            pdl=TRTLLM_ENABLE_PDL,
            cta_scoring=_cta_scoring(num_tokens, num_experts),
        )(
            router_logits,
            selection_bias,
            None,
            None,
            ids,
            weights,
            None,
            None,
            tables,
            local_start,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            None,
            None,
        )
    return ids, weights, tables
