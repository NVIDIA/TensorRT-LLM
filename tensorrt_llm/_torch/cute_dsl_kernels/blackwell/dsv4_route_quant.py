# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 sqrt-softplus routing and NVFP4 quantization of the MoE input in one launch."""

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

Int32, Uint8, Uint32, Float32 = cutlass.Int32, cutlass.Uint8, cutlass.Uint32, cutlass.Float32

# A CTA takes at most a quarter of an SM's registers and under 2 KB of shared memory, so it fits
# beside a concurrent kernel's CTA (the shared expert's GEMM), in the shared-memory carveout that
# kernel chose, instead of waiting for it to drain.
CTAS_PER_SM = 4

# With tables one CTA of 8 warps routes every token, a warp per token: one round of the warps, since
# a second round costs more than the routing kernel the tables replace.
ROUTING_TABLES_THREADS = 256
ROUTING_TABLES_MAX_TOKENS = ROUTING_TABLES_THREADS // 32
ROUTING_TABLES_MAX_EXPERTS = 4 * ROUTING_TABLES_THREADS

# TRTLLM-Gen FP4 tiles usable at these sizes: its smallest and the next two (computeSelectedTileN).
ROUTING_TABLE_TILES = (8, 16, 32)


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
    """

    group_size = 16

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        hidden_size: int,
        routing_tiles: tuple[int, ...] = (),
    ):
        warp = cute.arch.WARP_SIZE
        self.num_threads = ROUTING_TABLES_THREADS if routing_tiles else 128
        if (
            num_experts % warp != 0
            or not 0 < top_k <= warp
            or hidden_size % self.group_size != 0
            or (routing_tiles and num_experts > ROUTING_TABLES_MAX_EXPERTS)
        ):
            raise ValueError(
                f"expected experts divisible by {warp} (at most {ROUTING_TABLES_MAX_EXPERTS} with "
                f"routing tables), 0 < top_k <= {warp} and hidden divisible by {self.group_size}; "
                f"got {num_experts}, {top_k}, {hidden_size}"
            )
        self.num_experts = num_experts
        # A thread's contiguous experts in the table scan.
        self.experts_per_thread = (num_experts + self.num_threads - 1) // self.num_threads
        self.experts_per_lane = num_experts // warp
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.routing_tiles = routing_tiles
        self.groups = hidden_size // self.group_size
        self.quant_ctas = (self.groups + self.num_threads - 1) // self.num_threads

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        x: cute.Tensor,
        global_scale: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        x_fp4: cute.Tensor,
        x_sf: cute.Tensor,
        tables: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        self.kernel(logits, bias, x, global_scale, ids, weights, x_fp4, x_sf, tables).launch(
            grid=[logits.shape[0], 1 + self.quant_ctas, 1],
            block=[self.num_threads, 1, 1],
            min_blocks_per_mp=CTAS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        x: cute.Tensor,
        global_scale: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        x_fp4: cute.Tensor,
        x_sf: cute.Tensor,
        tables: Optional[cute.Tensor],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, part, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.routing_tiles):
            num_tiles = len(self.routing_tiles)
            smem = SmemAllocator()
            scratch = (
                smem.allocate_tensor(Int32, cute.make_layout(self.num_experts)),
                smem.allocate_tensor(
                    Int32, cute.make_layout((num_tiles, self.num_threads // cute.arch.WARP_SIZE))
                ),
            )
        if part == 0:
            if cutlass.const_expr(self.routing_tiles):
                if row == 0:
                    self.route_with_tables(logits, bias, ids, weights, tables, *scratch, tidx)
            elif tidx < cute.arch.WARP_SIZE:
                self.route(logits, bias, ids, weights, row, tidx)
        else:
            group = (part - 1) * self.num_threads + tidx
            if group < self.groups:
                self.quantize(x, global_scale, x_fp4, x_sf, row, group)

    @cute.jit
    def route(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        row: Int32,
        lane: Int32,
    ) -> Int32:
        """Route token ``row`` with one warp; lane ``k < top_k`` returns pick ``k``."""
        warp = cute.arch.WARP_SIZE
        taken = Int32(-(2**31))
        scores = []
        keys = []
        for v in cutlass.range_constexpr(self.experts_per_lane):
            expert = lane + v * warp
            logit = Float32(logits[row, expert])
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
        total = score
        for offset in (16, 8, 4, 2, 1):
            total = _add_rn(total, cute.arch.shuffle_sync_bfly(total, offset=offset))
        if lane < self.top_k:
            ids[row, lane] = expert
            weight = cute.math.div(
                score,
                _add_rn(total, Float32(1e-20)),
                rounding=cute.math.RoundingMode.NEAREST_EVEN,
            )
            weight = _mul_rn(weight, Float32(self.routed_scaling_factor))
            weights[row, lane] = weight.to(weights.element_type)
        return expert

    @cute.jit
    def route_with_tables(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        tables: cute.Tensor,
        experts: cute.Tensor,
        warp_ctas: cute.Tensor,
        tidx: Int32,
    ):
        """Route every token (warp ``t`` token ``t``), then write ``tables[t]`` for tile ``t``:
        expanded-to-permuted, permuted-to-token, CTA experts, CTA limits, (CTAs, padded rows).
        ``experts`` holds the expert histogram, then one tile's first CTAs."""
        warp_size = cute.arch.WARP_SIZE
        num_warps = self.num_threads // warp_size
        warp = tidx // warp_size
        lane = tidx % warp_size
        num_tokens = logits.shape[0]
        for e in cutlass.range(tidx, self.num_experts, self.num_threads):
            experts[e] = Int32(0)
        cute.arch.barrier()
        # Lane k < top_k keeps its token's pick k: the expert and its row in the expert's block.
        pick_expert = Int32(0)
        pick_rank = Int32(-1)
        if warp < num_tokens:
            expert = self.route(logits, bias, ids, weights, warp, lane)
            if lane < self.top_k:
                pick_expert = expert
                # An expert's rows come in atomic order; the GEMMs and finalize do not depend on it.
                pick_rank = cute.arch.atomic_add(experts.iterator + expert, Int32(1), scope="cta")
        cute.arch.barrier()
        # Thread tidx owns experts [tidx * per_thread, (tidx + 1) * per_thread).
        per_thread = self.experts_per_thread
        counts = []
        for j in cutlass.range_constexpr(per_thread):
            count = Int32(0)
            if tidx * per_thread + j < self.num_experts:
                count = experts[tidx * per_thread + j]
            counts.append(count)
        # All tiles at once: each expert's CTA count, summed per thread, scanned within the warp,
        # then over warps.
        ctas = [[(count + tile - 1) // tile for count in counts] for tile in self.routing_tiles]
        owned = [sum(tile_ctas[1:], tile_ctas[0]) for tile_ctas in ctas]
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
            for j in cutlass.range_constexpr(per_thread):
                expert = tidx * per_thread + j
                if expert < self.num_experts:
                    experts[expert] = first
                    for c in cutlass.range(ctas[t][j]):
                        cta = first + c
                        limit = (cta + 1) * tile
                        end = first * tile + counts[j]
                        if end < limit:
                            limit = end
                        tables[t, 2, cta] = expert
                        tables[t, 3, cta] = limit
                first = first + ctas[t][j]
            if tidx == 0:
                tables[t, 4, 0] = total
                tables[t, 4, 1] = total * tile
            cute.arch.barrier()
            if pick_rank >= 0:
                pick = warp * self.top_k + lane
                permuted = experts[pick_expert] * tile + pick_rank
                tables[t, 0, pick] = permuted
                tables[t, 1, permuted] = warp
            cute.arch.barrier()

    @cute.jit
    def quantize(
        self,
        x: cute.Tensor,
        global_scale: cute.Tensor,
        x_fp4: cute.Tensor,
        x_sf: cute.Tensor,
        row: Int32,
        group: Int32,
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


_WEIGHT_TYPES = {torch.float32: Float32, torch.bfloat16: cutlass.BFloat16}


@functools.cache
def _compile(
    num_experts: int,
    top_k: int,
    routed_scaling_factor: float,
    hidden_size: int,
    weights_dtype: torch.dtype,
    routing_tiles: tuple[int, ...],
):
    tokens = cute.sym_int()

    def rows(dtype, width, **kwargs):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (tokens, width), stride_order=(1, 0), **kwargs
        )

    tables = None
    if routing_tiles:
        tables = cute.runtime.make_fake_compact_tensor(
            Int32, (len(routing_tiles), 5, cute.sym_int()), stride_order=(2, 1, 0)
        )
    return cute.compile(
        DSv4RouteNvfp4QuantKernel(
            num_experts, top_k, routed_scaling_factor, hidden_size, routing_tiles
        ),
        rows(Float32, num_experts),
        cute.runtime.make_fake_compact_tensor(Float32, (num_experts,), stride_order=(0,)),
        rows(cutlass.BFloat16, hidden_size, assumed_align=16),
        cute.runtime.make_fake_compact_tensor(Float32, (1,), stride_order=(0,)),
        rows(Int32, top_k),
        rows(_WEIGHT_TYPES[weights_dtype], top_k),
        rows(Uint32, hidden_size // 8),
        rows(Uint8, hidden_size // 16),
        tables,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def can_build_routing_tables(num_tokens: int, num_experts: int) -> bool:
    """Whether ``dsv4_route_nvfp4_quant`` can also write routing tables for this shape."""
    return 0 < num_tokens <= ROUTING_TABLES_MAX_TOKENS and num_experts <= ROUTING_TABLES_MAX_EXPERTS


def _max_ctas(num_tokens: int, top_k: int, num_experts: int, tile: int) -> int:
    """TRTLLM-Gen's getMaxNumCtasInBatchDim: one CTA per routed expert, plus one per full tile."""
    expanded = num_tokens * top_k
    filled = min(num_experts, expanded)
    return filled + (expanded - filled) // tile


def dsv4_route_nvfp4_quant(
    router_logits: torch.Tensor,
    selection_bias: torch.Tensor,
    x: torch.Tensor,
    global_scale: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    weights_dtype: torch.dtype,
    with_routing_tables: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """DeepSeek-V4 routing and NVFP4 input quantization in one launch, bitwise ``routing.apply``
    then ``fp4_quantize``; ``with_routing_tables`` also returns ``run_moe``'s routing tables."""
    num_tokens, num_experts = router_logits.shape
    hidden_size = x.shape[1]
    device = x.device
    ids = torch.empty((num_tokens, top_k), dtype=torch.int32, device=device)
    weights = torch.empty((num_tokens, top_k), dtype=weights_dtype, device=device)
    x_fp4 = torch.empty((num_tokens, hidden_size // 8), dtype=torch.int32, device=device)
    x_sf = torch.empty((num_tokens, hidden_size // 16), dtype=torch.uint8, device=device)
    tables = None
    routing_tiles = ()
    if with_routing_tables:
        if not can_build_routing_tables(num_tokens, num_experts):
            raise ValueError(
                f"no routing tables for {num_tokens} tokens over {num_experts} experts"
            )
        routing_tiles = ROUTING_TABLE_TILES
        # Route maps are read one past their padded rows (getRouteMapAllocCount).
        width = max(
            [num_tokens * top_k]
            + [_max_ctas(num_tokens, top_k, num_experts, t) * t + 1 for t in routing_tiles]
        )
        tables = torch.empty((len(routing_tiles), 5, width), dtype=torch.int32, device=device)
    if num_tokens > 0:
        _compile(
            num_experts, top_k, routed_scaling_factor, hidden_size, weights_dtype, routing_tiles
        )(
            router_logits,
            selection_bias,
            x,
            global_scale.view(1),
            ids,
            weights,
            x_fp4,
            x_sf,
            tables,
        )
    return ids, weights, x_fp4.view(torch.uint8), x_sf, tables
