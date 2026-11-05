# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The last decoder layer's unpool gather, post-FFN gated residual and final model norm, in one launch.

The decoder layers before the last hand their unpool slice to ``unpool_gather_boundary``, whose
boundary also applies the next layer's pre-QKV norm. The last layer has no next layer, so the built-in
step runs three launches between its unpool GEMM and the LM head:

1. ``musespark_open_projection_gather._gather_columns``: the eight 1024-column slices exchanged through
   fabric memory into a ``[M, 8192]`` BF16 branch;
2. the post-FFN join ``MuseSparkOpenGatedResidual.add_after_rmsnorm`` (Triton
   ``_gated_residual_after_rmsnorm_kernel`` with ``FUSE_NEXT_NORM=False``) and the BF16 rounding of
   its FP32 output;
3. the final model norm (``RMSNorm`` without gemma: flashinfer's CuTe-DSL ``RMSNormKernel`` with
   ``weight_bias = 0``) on that BF16 rounding.

This op is the three in one kernel. Route 1 (``route=1``) computes the same values with the same
floating-point operations in the same order:

* the join, as Triton 3.8 compiles it for an 8192-wide row at ``num_warps=16`` (read off its PTX): the
  gain product as ``mul.rn.bf16x2``; Triton thread ``t`` (0..511) owns columns ``8t + e`` and
  ``4096 + 8t + e``; its 16 squares summed as ``E + O`` with ``E = ((s0^2 + s2^2) + (s4^2 + s6^2)) +
  ((s8^2 + s10^2) + (s12^2 + s14^2))`` (each pair one ``fma(a, a, b^2)``) and ``O`` the same over the
  odd registers; a warp butterfly (offsets 16, 8, 4, 2, 1); the 16 warp sums through shared memory,
  read as ``W[lane % 16]`` and combined by the butterfly offsets 8, 4, 2, 1; ``div.full.f32`` by 8192,
  ``add`` eps, ``rsqrt.approx.ftz``; ``normed = bf16(rstd * s)``;
  ``updated = fma(residual, alpha, normed * beta)``;
* the final norm, as flashinfer's ``RMSNormKernel`` compiles for an 8192-wide BF16 row (one CTA of 128
  threads; ``kernels/pf_pre/pf_pre.cu`` and ``ops/allreduce/drafter_hop.py`` certify the same
  statements): flashinfer thread ``T`` owns columns ``8T + 1024b + e``; one chain
  ``s = fma(x, x, s)`` from ``+0`` over ``b`` then ``e``; the warp butterfly (offsets 1, 2, 4, 8, 16);
  the four warp sums padded with ``+0`` and the same butterfly;
  ``rstd = rsqrt.approx.ftz(fma(s, 2^-13, eps))``; ``y = bf16((x * rstd) * (w + 0))`` on
  ``x = bf16(updated)``.

Every floating-point operation is an inline PTX instruction with its rounding spelled out, so ptxas can
neither contract nor reorder any of them.

Route 2 (``route=2``) keeps every rounding point and the join's reduction but sums the final norm's
squares in this kernel's own order (each thread's 16 values as the join's tree, then the join's warp and
cluster combine): an association change of one FP32 sum, no precision change. It drops route 1's
exact-order reduction and is therefore not value-identical.

Thread map. A row is one cluster of ``CLUSTER = 4`` CTAs of 128 threads; thread ``j`` of CTA ``c``
plays Triton thread ``t = 128 c + j`` whole, so each real warp is one Triton warp (warps ``4c .. 4c+3``)
and the thread's columns are ``1024 c + 8 j + e`` (words ``4j .. 4j+3`` of peer ``c``'s slice) and
``4096 + 1024 c + 8 j + e`` (the same words of peer ``4 + c``'s slice). flashinfer thread ``T``'s block
``b`` is chunk A of CTA ``b`` (``b < 4``) or chunk B of CTA ``b - 4``, always thread ``T``: every CTA
stores its two 16-byte BF16 chunks per thread into all four CTAs' shared memory (distributed shared
memory, one mbarrier), and every CTA then runs flashinfer's 128 chains and both butterflies itself.

Transport: the ``ProjectionInbox`` protocol of ``unpool_gather_boundary`` (tagged 64-bit words, two
slots per row alternating per call, readers clear what they consumed, per-row state read by every CTA
of the row and advanced by CTA 0 after the last cluster barrier). Thread ``j`` of CTA ``c`` publishes
this rank's words ``4j .. 4j+3`` to peers ``c`` and ``4 + c`` and polls the same words of those two
peers, which covers every (peer, word) pair once per call. The op shares the inbox (and its per-row
state) with the other layers' unpool boundaries: every call on it is one ordered chain per rank.

PDL: launched with programmatic stream serialization. Only the norm weight, the gain and the gated
residual's coefficients may be read before ``griddepcontrol.wait`` (``prewait``; the caller passes
them only when they are load-time constants); the local slice, the residual, the state and every
transport access come after it. ``trigger`` places ``griddepcontrol.launch_dependents``: ``"exit"``
after the last store (flashinfer's placement), ``"after_wait"``, or ``"entry"``. A dependent that reads
this kernel's output does so after its own wait, which returns once this grid has completed.
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .projection_inbox import HAS_CUTE_DSL, MAX_ROWS, ProjectionInbox

#: The TP size this kernel serves (the ``last_layer_boundary`` deployment); its inbox is the
#: 8-row TP8 one (``[slot][WORLD][MAX_ROWS][row words]``).
WORLD = 8

#: Width of the residual stream.
HIDDEN = 8192
#: Columns of the unpool output one rank computes.
LOCAL_COLUMNS = HIDDEN // WORLD
#: 32-bit payload words of one row of one rank's slice.
_ROW_WORDS = LOCAL_COLUMNS // 2
#: CTAs per row and threads per CTA (4 x 128 Triton threads x 2 chunks x 8 columns = 8192).
CLUSTER = 4
THREADS = 128
WARPS = THREADS // 32
#: Triton's warps per row (the join's 16).
ROW_WARPS = CLUSTER * WARPS
#: flashinfer's blocks per thread (8192 / (128 x 8)).
BLOCKS = 8
assert CLUSTER * THREADS * 2 * 8 == HIDDEN
#: Where the kernel releases its dependents (see the module docstring).
TRIGGERS = {"exit": 0, "after_wait": 1, "entry": 2}
#: 1: exact orders (value-identical); 2: this kernel's own order for the final norm's sum.
ROUTES = (1, 2)
#: Shared memory: 16 warp sums (16 B messages), the row's 16 KB of BF16 chunks, 4 warp sums,
#: two mbarriers, alignment slack.
_PARTIAL_BYTES = ROW_WARPS * 16
_CHUNK_BYTES = BLOCKS * THREADS * 16
_SMEM_BYTES = 2 * _PARTIAL_BYTES + _CHUNK_BYTES + WARPS * 4 + 16 + 64

if HAS_CUTE_DSL:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Float32, Int32, Int64, Uint32
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    from .projection_inbox import (
        any_stale,
        clear_words,
        load_state,
        load_words,
        next_tag,
        poll_words,
        poll_words_if,
        publish_words,
        store_state,
    )

    def _asm(result_type, operands, text, constraints, loc, ip, side_effects=True):
        return llvm.inline_asm(
            result_type,
            operands,
            text,
            constraints,
            has_side_effects=side_effects,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    def _f32(value, loc, ip):
        return Float32(value).ir_value(loc=loc, ip=ip)

    def _u32(value, loc, ip):
        return Uint32(value).ir_value(loc=loc, ip=ip)

    def _i64(value, loc, ip):
        return Int64(value).ir_value(loc=loc, ip=ip)

    def _f32_asm(text, operands, loc, ip):
        result = _asm(
            T.f32(),
            [_f32(x, loc, ip) for x in operands],
            text,
            "=f" + ",f" * len(operands),
            loc,
            ip,
            side_effects=False,
        )
        return Float32(result)

    # ---- arithmetic: one PTX instruction each, as the built-in kernels run them ----

    @dsl_user_op
    def _add_rn(a, b, *, loc=None, ip=None):
        return _f32_asm("add.rn.f32 $0, $1, $2;", (a, b), loc, ip)

    @dsl_user_op
    def _mul_rn(a, b, *, loc=None, ip=None):
        return _f32_asm("mul.rn.f32 $0, $1, $2;", (a, b), loc, ip)

    @dsl_user_op
    def _fma_rn(a, b, c, *, loc=None, ip=None):
        return _f32_asm("fma.rn.f32 $0, $1, $2, $3;", (a, b, c), loc, ip)

    @dsl_user_op
    def _div_full(a, b, *, loc=None, ip=None):
        """Triton's ``x / 8192.0`` as compiled: ``div.full.f32``."""
        return _f32_asm("div.full.f32 $0, $1, $2;", (a, b), loc, ip)

    @dsl_user_op
    def _rsqrt_ftz(a, *, loc=None, ip=None):
        return _f32_asm("rsqrt.approx.ftz.f32 $0, $1;", (a,), loc, ip)

    @dsl_user_op
    def _round_bf16(a, *, loc=None, ip=None):
        """``a`` rounded to BF16 (nearest even) and widened back, exactly."""
        return _f32_asm(
            "{\n.reg .b16 r;\ncvt.rn.bf16.f32 r, $1;\ncvt.f32.bf16 $0, r;\n}", (a,), loc, ip
        )

    @dsl_user_op
    def _shfl_bfly(value, lane_mask, *, loc=None, ip=None):
        result = _asm(
            T.f32(),
            [_f32(value, loc, ip), _u32(lane_mask, loc, ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            loc,
            ip,
        )
        return Float32(result)

    @dsl_user_op
    def _float_bits(value, *, loc=None, ip=None):
        bits = _asm(
            T.i32(), [_f32(value, loc, ip)], "mov.b32 $0, $1;", "=r,f", loc, ip, side_effects=False
        )
        return Int32(bits)

    @dsl_user_op
    def _bits_float(word, *, loc=None, ip=None):
        value = _asm(
            T.f32(), [_u32(word, loc, ip)], "mov.b32 $0, $1;", "=f,r", loc, ip, side_effects=False
        )
        return Float32(value)

    @dsl_user_op
    def _mul_bf16x2(a, b, *, loc=None, ip=None):
        """``mul.rn.bf16x2``: the join's gain product, rounded to BF16 once (Triton's instruction)."""
        result = _asm(
            T.i32(),
            [_u32(a, loc, ip), _u32(b, loc, ip)],
            "mul.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    @dsl_user_op
    def _unpack_bf16x2(word, *, loc=None, ip=None):
        """FP32 values of the low and high BF16 halves of ``word`` (exact)."""
        result = _asm(
            ir.Type.parse("!llvm.struct<(f32,f32)>"),
            [_u32(word, loc, ip)],
            "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xffff0000;\n"
            "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
            "=f,=f,r",
            loc,
            ip,
            side_effects=False,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        )

    @dsl_user_op
    def _pack_bf16x2(low, high, *, loc=None, ip=None):
        """Two FP32 values rounded to BF16 (nearest even), ``low`` in the low half."""
        result = _asm(
            T.i32(),
            [_f32(high, loc, ip), _f32(low, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    @dsl_user_op
    def _store_words(address, w0, w1, w2, w3, *, loc=None, ip=None):
        """Plain 16-byte global store."""
        _asm(
            None,
            [_i64(address, loc, ip)] + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
            "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r",
            loc,
            ip,
        )

    def _square_tree(s):
        """The join's per-thread sum of 16 squares (``E + O``, the PTX's packed tree)."""
        pairs = []
        for base in (0, 4, 8, 12):
            for odd in (0, 1):
                a, b = s[base + odd], s[base + 2 + odd]
                pairs.append(_fma_rn(a, a, _mul_rn(b, b)))
        # pairs = [E0, O0, E1, O1, E2, O2, E3, O3]
        even = _add_rn(_add_rn(pairs[0], pairs[2]), _add_rn(pairs[4], pairs[6]))
        odd = _add_rn(_add_rn(pairs[1], pairs[3]), _add_rn(pairs[5], pairs[7]))
        return _add_rn(even, odd)

    class _UnpoolFinalBoundaryKernel:
        """Gather + gated residual + final norm, one row per cluster of 4 CTAs."""

        def __init__(
            self,
            rank: int,
            post_eps: float,
            final_eps: float,
            route: int,
            trigger: int,
            prewait: bool,
            store_updated: bool,
        ):
            self.rank = rank
            self.post_eps = float(post_eps)
            self.final_eps = float(final_eps)
            self.route = route
            self.trigger = trigger
            self.prewait = prewait
            self.store_updated = store_updated
            # Transaction bytes of the second exchange: the row's BF16 chunks (route 1) or the
            # 16 warp sums (route 2).
            self.final_bytes = _CHUNK_BYTES if route == 1 else _PARTIAL_BYTES

        @cute.jit
        def __call__(
            self,
            local: Int64,
            counters: Int64,
            inbox: Int64,
            stride: Int64,
            residual: Int64,
            gain: Int64,
            alpha: Int64,
            beta: Int64,
            weight: Int64,
            output: Int64,
            updated: Int64,
            rows: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                local, counters, inbox, stride, residual, gain, alpha, beta, weight, output, updated
            ).launch(
                grid=[rows * CLUSTER, 1, 1],
                block=[THREADS, 1, 1],
                cluster=[CLUSTER, 1, 1],
                smem=_SMEM_BYTES,
                stream=stream,
                use_pdl=True,
            )

        @cute.kernel
        def kernel(
            self,
            local: Int64,
            counters: Int64,
            inbox: Int64,
            stride: Int64,
            residual: Int64,
            gain: Int64,
            alpha: Int64,
            beta: Int64,
            weight: Int64,
            output: Int64,
            updated: Int64,
        ):
            if cutlass.const_expr(self.trigger == TRIGGERS["entry"]):
                cute.arch.griddepcontrol_launch_dependents()
            tid = cute.arch.thread_idx()[0]
            lane = tid % 32
            warp = tid // 32
            crank = cute.arch.block_idx()[0] % CLUSTER
            row = cute.arch.block_idx()[0] // CLUSTER
            smem = utils.SmemAllocator()
            partials = smem.allocate_tensor(
                Float32, cute.make_layout((2 * ROW_WARPS * 4,)), byte_alignment=16
            )
            chunks = smem.allocate_tensor(
                Uint32, cute.make_layout((BLOCKS * THREADS * 4,)), byte_alignment=16
            )
            sums = smem.allocate_tensor(Float32, cute.make_layout((WARPS,)), byte_alignment=16)
            bars = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((2,)), byte_alignment=8)
            if tid == 0:
                cute.arch.mbarrier_init(bars.iterator, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator, _PARTIAL_BYTES)
            if tid == 1:
                cute.arch.mbarrier_init(bars.iterator + 1, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + 1, self.final_bytes)
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

            # This thread's columns: chunk A = 1024 c + 8 j (peer c), chunk B = 4096 + chunk A
            # (peer 4 + c); both are words 4j .. 4j+3 of their peer's slice.
            word = tid * 4
            column_a = crank * LOCAL_COLUMNS + tid * 8
            column_b = column_a + HIDDEN // 2
            peer_a = crank
            peer_b = crank + CLUSTER
            vec_a = Int64(column_a) * 2
            vec_b = Int64(column_b) * 2
            if cutlass.const_expr(self.prewait):
                ga = load_words(gain + vec_a)
                gb = load_words(gain + vec_b)
                wa = load_words(weight + vec_a)
                wb = load_words(weight + vec_b)
                aa = load_words(alpha + vec_a * 2) + load_words(alpha + vec_a * 2 + 16)
                ab = load_words(alpha + vec_b * 2) + load_words(alpha + vec_b * 2 + 16)
                ba = load_words(beta + vec_a * 2) + load_words(beta + vec_a * 2 + 16)
                bb = load_words(beta + vec_b * 2) + load_words(beta + vec_b * 2 + 16)
            # The local slice is the previous kernel's output; nothing below may read
            # before this wait.
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.trigger == TRIGGERS["after_wait"]):
                cute.arch.griddepcontrol_launch_dependents()

            state = counters + Int64(row) * 8
            last_tag, last_slot = load_state(state)
            tag = next_tag(last_tag)
            slot = (last_slot ^ Uint32(1)) & Uint32(1)
            l0, l1, l2, l3 = load_words(local + (Int64(row) * _ROW_WORDS + Int64(word)) * 4)
            sent = ((Int64(slot) * WORLD + self.rank) * MAX_ROWS + Int64(row)) * _ROW_WORDS + Int64(
                word
            )
            if peer_a != self.rank:
                publish_words(inbox + Int64(peer_a) * stride + sent * 8, l0, l1, l2, l3, tag)
            if peer_b != self.rank:
                publish_words(inbox + Int64(peer_b) * stride + sent * 8, l0, l1, l2, l3, tag)
            if cutlass.const_expr(not self.prewait):
                ga = load_words(gain + vec_a)
                gb = load_words(gain + vec_b)
                wa = load_words(weight + vec_a)
                wb = load_words(weight + vec_b)
                aa = load_words(alpha + vec_a * 2) + load_words(alpha + vec_a * 2 + 16)
                ab = load_words(alpha + vec_b * 2) + load_words(alpha + vec_b * 2 + 16)
                ba = load_words(beta + vec_a * 2) + load_words(beta + vec_a * 2 + 16)
                bb = load_words(beta + vec_b * 2) + load_words(beta + vec_b * 2 + 16)
            res_row = residual + Int64(row) * (HIDDEN * 4)
            ra = load_words(res_row + vec_a * 2) + load_words(res_row + vec_a * 2 + 16)
            rb = load_words(res_row + vec_b * 2) + load_words(res_row + vec_b * 2 + 16)

            # The two peers' words of this thread (this rank's own slice where the peer is us).
            base = inbox + Int64(self.rank) * stride
            box_a = (
                base
                + (
                    ((Int64(slot) * WORLD + Int64(peer_a)) * MAX_ROWS + Int64(row)) * _ROW_WORDS
                    + Int64(word)
                )
                * 8
            )
            box_b = (
                base
                + (
                    ((Int64(slot) * WORLD + Int64(peer_b)) * MAX_ROWS + Int64(row)) * _ROW_WORDS
                    + Int64(word)
                )
                * 8
            )
            xa0, xa1, xa2, xa3 = l0, l1, l2, l3
            xb0, xb1, xb2, xb3 = l0, l1, l2, l3
            ta0, ta1, ta2, ta3 = tag, tag, tag, tag
            tb0, tb1, tb2, tb3 = tag, tag, tag, tag
            if peer_a != self.rank:
                xa0, ta0, xa1, ta1, xa2, ta2, xa3, ta3 = poll_words(box_a)
            if peer_b != self.rank:
                xb0, tb0, xb1, tb1, xb2, tb2, xb3, tb3 = poll_words(box_b)
            stale_a = any_stale(ta0, ta1, ta2, ta3, tag)
            stale_b = any_stale(tb0, tb1, tb2, tb3, tag)
            while (stale_a | stale_b) != Uint32(0):
                xa0, ta0, xa1, ta1, xa2, ta2, xa3, ta3 = poll_words_if(
                    box_a, stale_a, xa0, ta0, xa1, ta1, xa2, ta2, xa3, ta3
                )
                xb0, tb0, xb1, tb1, xb2, tb2, xb3, tb3 = poll_words_if(
                    box_b, stale_b, xb0, tb0, xb1, tb1, xb2, tb2, xb3, tb3
                )
                stale_a = any_stale(ta0, ta1, ta2, ta3, tag)
                stale_b = any_stale(tb0, tb1, tb2, tb3, tag)
            if peer_a != self.rank:
                clear_words(box_a)
            if peer_b != self.rank:
                clear_words(box_b)

            # ---- the join (Triton's statements) ----
            branch = (xa0, xa1, xa2, xa3, xb0, xb1, xb2, xb3)
            gains = tuple(ga) + tuple(gb)
            scaled = []
            for k in cutlass.range_constexpr(8):
                low, high = _unpack_bf16x2(_mul_bf16x2(branch[k], gains[k]))
                scaled.append(low)
                scaled.append(high)
            total = _square_tree(scaled)
            for i in cutlass.range_constexpr(5):
                total = _add_rn(total, _shfl_bfly(total, Uint32(16 >> i)))
            # Triton warp 4 c + warp's sum, to every CTA of the row (16-byte messages).
            if lane < CLUSTER:
                slot_ptr = partials.iterator + (crank * WARPS + warp) * 4
                cute.arch.store_async_dsmem(
                    slot_ptr.align(16),
                    (_float_bits(total), Int32(0), Int32(0), Int32(0)),
                    bars.iterator,
                    lane,
                )
            cute.arch.mbarrier_wait(bars.iterator, 0)
            row_sum = partials[(lane % ROW_WARPS) * 4]
            for i in cutlass.range_constexpr(4):
                row_sum = _add_rn(row_sum, _shfl_bfly(row_sum, Uint32(8 >> i)))
            rstd = _rsqrt_ftz(_add_rn(Float32(self.post_eps), _div_full(row_sum, Float32(8192.0))))

            alphas = tuple(aa) + tuple(ab)
            betas = tuple(ba) + tuple(bb)
            resids = tuple(ra) + tuple(rb)
            new = []
            carried = []
            for i in cutlass.range_constexpr(16):
                normed = _round_bf16(_mul_rn(rstd, scaled[i]))
                u = _fma_rn(
                    _bits_float(resids[i]),
                    _bits_float(alphas[i]),
                    _mul_rn(normed, _bits_float(betas[i])),
                )
                new.append(u)
                carried.append(_round_bf16(u))
            packed = [_pack_bf16x2(carried[2 * k], carried[2 * k + 1]) for k in range(8)]

            # ---- the final norm's sum of squares ----
            if cutlass.const_expr(self.route == 1):
                # flashinfer thread ``tid``'s blocks: chunk A of CTA b at block b, chunk B of
                # CTA b at block 4 + b; stored into every CTA of the row.
                target_a = chunks.iterator + (crank * THREADS + tid) * 4
                target_b = chunks.iterator + ((crank + CLUSTER) * THREADS + tid) * 4
                for peer in cutlass.range_constexpr(CLUSTER):
                    cute.arch.store_async_dsmem(
                        target_a.align(16),
                        (Int32(packed[0]), Int32(packed[1]), Int32(packed[2]), Int32(packed[3])),
                        bars.iterator + 1,
                        peer,
                    )
                    cute.arch.store_async_dsmem(
                        target_b.align(16),
                        (Int32(packed[4]), Int32(packed[5]), Int32(packed[6]), Int32(packed[7])),
                        bars.iterator + 1,
                        peer,
                    )
                cute.arch.mbarrier_wait(bars.iterator + 1, 0)
                chain = Float32(0.0)
                for b in cutlass.range_constexpr(BLOCKS):
                    for k in cutlass.range_constexpr(4):
                        low, high = _unpack_bf16x2(chunks[(b * THREADS + tid) * 4 + k])
                        chain = _fma_rn(low, low, chain)
                        chain = _fma_rn(high, high, chain)
                for i in cutlass.range_constexpr(5):
                    chain = _add_rn(chain, _shfl_bfly(chain, Uint32(1 << i)))
                if lane == 0:
                    sums[warp] = chain
                cute.arch.barrier()
                final_sum = Float32(0.0)
                if lane < WARPS:
                    final_sum = sums[lane]
                for i in cutlass.range_constexpr(5):
                    final_sum = _add_rn(final_sum, _shfl_bfly(final_sum, Uint32(1 << i)))
            else:
                part = _square_tree(carried)
                for i in cutlass.range_constexpr(5):
                    part = _add_rn(part, _shfl_bfly(part, Uint32(16 >> i)))
                if lane < CLUSTER:
                    slot_ptr = partials.iterator + (ROW_WARPS + crank * WARPS + warp) * 4
                    cute.arch.store_async_dsmem(
                        slot_ptr.align(16),
                        (_float_bits(part), Int32(0), Int32(0), Int32(0)),
                        bars.iterator + 1,
                        lane,
                    )
                cute.arch.mbarrier_wait(bars.iterator + 1, 0)
                final_sum = partials[(ROW_WARPS + lane % ROW_WARPS) * 4]
                for i in cutlass.range_constexpr(4):
                    final_sum = _add_rn(final_sum, _shfl_bfly(final_sum, Uint32(8 >> i)))
            final_rstd = _rsqrt_ftz(
                _fma_rn(final_sum, Float32(1.0 / HIDDEN), Float32(self.final_eps))
            )

            # ---- the final norm's output (flashinfer's pass 2) ----
            weights = tuple(wa) + tuple(wb)
            out = []
            for k in cutlass.range_constexpr(8):
                w_low, w_high = _unpack_bf16x2(weights[k])
                low = _mul_rn(_mul_rn(carried[2 * k], final_rstd), _add_rn(w_low, Float32(0.0)))
                high = _mul_rn(
                    _mul_rn(carried[2 * k + 1], final_rstd), _add_rn(w_high, Float32(0.0))
                )
                out.append(_pack_bf16x2(low, high))
            out_row = output + Int64(row) * (HIDDEN * 2)
            _store_words(out_row + vec_a, out[0], out[1], out[2], out[3])
            _store_words(out_row + vec_b, out[4], out[5], out[6], out[7])
            if cutlass.const_expr(self.store_updated):
                upd_row = updated + Int64(row) * (HIDDEN * 4)
                bits = [Uint32(_float_bits(v)) for v in new]
                _store_words(upd_row + vec_a * 2, bits[0], bits[1], bits[2], bits[3])
                _store_words(upd_row + vec_a * 2 + 16, bits[4], bits[5], bits[6], bits[7])
                _store_words(upd_row + vec_b * 2, bits[8], bits[9], bits[10], bits[11])
                _store_words(upd_row + vec_b * 2 + 16, bits[12], bits[13], bits[14], bits[15])
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            # Every CTA of the row read the state before the cluster barriers above, so CTA 0
            # may now advance it.
            if crank == 0:
                if tid == 0:
                    store_state(state, tag, slot)
            if cutlass.const_expr(self.trigger == TRIGGERS["exit"]):
                cute.arch.griddepcontrol_launch_dependents()

    @functools.lru_cache(maxsize=64)
    def _compile(
        device_index: int,
        rank: int,
        post_eps: float,
        final_eps: float,
        route: int,
        trigger: int,
        prewait: bool,
        store_updated: bool,
    ):
        kernel = _UnpoolFinalBoundaryKernel(
            rank, post_eps, final_eps, route, trigger, prewait, store_updated
        )
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                kernel, *([Int64(0)] * 11), Int32(1), stream, options="--enable-tvm-ffi"
            )


def _reference(
    branch: torch.Tensor,
    residual: torch.Tensor,
    gain: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    weight: torch.Tensor,
    post_eps: float,
    final_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """float64 between the chain's BF16 rounding points, which it keeps."""
    scaled = (branch.double() * gain.double()).to(torch.bfloat16).double()
    normalized = scaled * torch.rsqrt(scaled.square().mean(-1, keepdim=True) + post_eps)
    normalized = normalized.to(torch.bfloat16).double()
    updated = alpha.double() * residual.double() + beta.double() * normalized
    carried = updated.float().to(torch.bfloat16).double()
    final = carried * torch.rsqrt(carried.square().mean(-1, keepdim=True) + final_eps)
    return updated.float(), (final * weight.double()).to(torch.bfloat16)


class UnpoolFinalBoundary(OpWrapper):
    """Gather the eight unpool slices, join the post-FFN residual and apply the final model norm.

    ``local`` is this rank's ``[M, 1024]`` BF16 unpool output (columns ``rank * 1024 ...``),
    ``residual`` the ``[M, 8192]`` FP32 residual stream, ``gain`` the post-FFN norm's ``1 + w``
    (BF16), ``alpha`` / ``beta`` the FP32 gated-residual coefficients, ``weight`` the final norm's
    weight (BF16, applied without a gemma offset). Returns the ``[M, 8192]`` BF16 LM-head input and,
    with ``return_updated``, the updated FP32 residual (else ``None``).

    A collective over the TP8 group on ``inbox``: every rank calls it with the same row count, in
    the same order relative to its other calls on ``inbox``. Launched with programmatic dependent
    launch.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DFlash2 K=7 verify: 8 rows, exact orders (value-identical route)",
            spec=dict(rows=8, route=1),
        ),
        Cell(
            why="a short block (3 rows), exact orders",
            spec=dict(rows=3, route=1),
        ),
        Cell(
            why="decode without drafts: 1 row, exact orders",
            spec=dict(rows=1, route=1),
        ),
        Cell(
            why="weights read before the dependency wait (load-time constants)",
            spec=dict(rows=8, route=1, prewait=True),
        ),
        Cell(
            why="dependents released at entry",
            spec=dict(rows=8, route=1, trigger="entry"),
        ),
        Cell(
            why="route 2: the final norm's sum in this kernel's order (not value-identical)",
            spec=dict(rows=8, route=2),
        ),
    )

    note = """
    Route 1 is certified against the built-in gather + gated residual + final norm bit for bit (the
    test compares raw bits through CUDA-graph replays with changing inputs, special values and a
    delayed rank). Route 2 differs from it only in the final norm's variance sum order and is gated
    against the float64 reference within a band. Rows 1..8 only (the inbox's capacity).
    """

    def __call__(
        self,
        local: torch.Tensor,
        inbox: ProjectionInbox,
        residual: torch.Tensor,
        gain: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        final_eps: float,
        route: int = 1,
        trigger: str = "exit",
        prewait: bool = False,
        return_updated: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        rows = local.shape[0]
        output = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=residual.device)
        updated = torch.empty_like(residual) if return_updated else None
        compiled = _compile(
            residual.device.index,
            inbox.rank,
            float(post_eps),
            float(final_eps),
            int(route),
            TRIGGERS[trigger],
            bool(prewait),
            bool(return_updated),
        )
        stream = cuda.CUstream(torch.cuda.current_stream(residual.device).cuda_stream)
        compiled(
            local.data_ptr(),
            inbox.counters.data_ptr(),
            inbox.base,
            inbox.stride,
            residual.data_ptr(),
            gain.data_ptr(),
            alpha.data_ptr(),
            beta.data_ptr(),
            weight.data_ptr(),
            output.data_ptr(),
            updated.data_ptr() if updated is not None else 0,
            rows,
            stream,
        )
        return output, updated

    def reference(
        self,
        per_rank_locals: Sequence[torch.Tensor],
        inbox: Optional[ProjectionInbox],
        residual: torch.Tensor,
        gain: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        final_eps: float,
        route: int = 1,
        trigger: str = "exit",
        prewait: bool = False,
        return_updated: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The chain on the concatenated slices, float64 between BF16 roundings.

        The collective exception to mirroring: ``per_rank_locals`` is every rank's ``local`` in
        ascending rank order. ``inbox``, ``route``, ``trigger`` and ``prewait`` are ignored.
        """
        branch = torch.cat(list(per_rank_locals), dim=1)
        updated, output = _reference(
            branch, residual, gain, alpha, beta, weight, post_eps, final_eps
        )
        return output, (updated if return_updated else None)

    def is_valid(
        self,
        local: torch.Tensor,
        inbox: ProjectionInbox,
        residual: torch.Tensor,
        gain: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        final_eps: float,
        route: int = 1,
        trigger: str = "exit",
        prewait: bool = False,
        return_updated: bool = False,
    ) -> None:
        """Every input is read through raw 16-byte vector accesses at fixed strides.

        A strided, misaligned or mistyped tensor is read as if it were packed, with no error; a row
        count outside 1..8 addresses outside the inbox.
        """
        rows = local.shape[0] if local.dim() == 2 else -1
        if not 1 <= rows <= MAX_ROWS:
            raise ValueError(
                f"local must be [1..{MAX_ROWS}, {LOCAL_COLUMNS}], got {tuple(local.shape)}"
            )
        if not fits(local, residual, gain, alpha, beta, weight):
            raise ValueError(
                "expected contiguous 16-byte aligned operands on one device: local BF16 "
                f"[M, {LOCAL_COLUMNS}], residual FP32 [M, {HIDDEN}], gain / weight BF16 [{HIDDEN}], "
                f"alpha / beta FP32 [{HIDDEN}]"
            )
        if inbox.row_words != _ROW_WORDS:
            raise ValueError(f"inbox carries {inbox.row_words}-word rows, this op {_ROW_WORDS}")
        if inbox.world != WORLD or inbox.max_rows != MAX_ROWS:
            raise ValueError(
                f"the inbox is TP{inbox.world} with {inbox.max_rows} rows; this op serves TP{WORLD} "
                f"with {MAX_ROWS}"
            )
        if route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {sorted(TRIGGERS)}, got {trigger!r}")


unpool_final_boundary = UnpoolFinalBoundary()


def fits(
    local: torch.Tensor,
    residual: torch.Tensor,
    gain: Optional[torch.Tensor],
    alpha: torch.Tensor,
    beta: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Whether the operands are the shapes, dtypes and layout the kernel addresses.

    Shapes, dtypes, contiguity and devices only, except the 16-byte alignment, which the
    caller asserts after choosing this path (pointer values may differ between ranks).
    """
    rows = local.shape[0] if local.dim() == 2 else 0
    if gain is None or not 1 <= rows <= MAX_ROWS:
        return False
    expected = (
        (local, (rows, LOCAL_COLUMNS), torch.bfloat16),
        (residual, (rows, HIDDEN), torch.float32),
        (gain, (HIDDEN,), torch.bfloat16),
        (alpha, (HIDDEN,), torch.float32),
        (beta, (HIDDEN,), torch.float32),
        (weight, (HIDDEN,), torch.bfloat16),
    )
    return all(
        tuple(t.shape) == shape
        and t.dtype == dtype
        and t.is_contiguous()
        and t.device == residual.device
        for t, shape, dtype in expected
    )


def available() -> bool:
    """Whether this image can build the kernel (CuTe DSL)."""
    return HAS_CUTE_DSL
