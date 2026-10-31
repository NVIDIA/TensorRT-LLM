# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Weight-streaming MXFP8 GEMV over an LM-head vocabulary shard, with the shard's argmax fused.

``C[M, N] = A[M, K] (bf16) @ dequant(B[N, K] e4m3, S[N, K/32] ue8m0).T`` for ``M <= 16`` (draft-head decode
rows). In ``argmax`` mode the logits are not stored; each row's argmax is folded into one packed int64 key

    key = (orderable_int32(value_fp32) << 32) | (0xFFFFFFFF - (vocab_offset + column))

with ``red.global.max.s64``, so a signed max over CTAs and then over vocab-shard ranks picks the largest
logit, ties to the smallest id (same packing as ``distributed/ll_exchange.py``; -0.0 ranks as +0.0).
``keys`` must hold ``INT64_MIN`` before the launch; the consumer resets it (``mxfp8_draft_head.decode_keys``).

* PDL: the first two weight tiles are issued before ``griddepcontrol.wait``; dependents are released once
  the activation is staged (they still wait for this grid's completion before reading the output).
* Persistent CTAs stream 8-row weight groups (CTA ``c`` takes ``c, c + C, ...``), two per 16-row tile, with
  one bulk copy per row into two double-buffered shared-memory slots.
* Weights are the M side of ``mma.sync.m16n8k16``, activations the N side; K is split over warps and the
  FP32 partials are summed through shared memory in a fixed order.
* Numerics: each activation row is scaled by ``2^e`` so its absmax lands in ``[2^14, 2^15)`` and converted
  to FP16; E4M3 -> FP16 is exact, so products are exact and only the FP32 summation order differs from a
  dequantized reference.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op

from .utils import griddepcontrol_launch_dependents

try:
    from cutlass.memory import SmemAllocator as _SmemAllocator
except ImportError:  # older CuTe DSL
    from cutlass.utils import SmemAllocator as _SmemAllocator

BLOCK = 32  # K elements per UE8M0 scale: the two m16n8k16 steps of one block
ROWS = 16  # weight rows per tile (the MMA's M)
GROUP_ROWS = 8  # weight rows per group: a tile is two groups
TOKENS = 8  # activation rows per MMA tile (the MMA's N)
HEADROOM_EXP = 14  # per-row activation max scaled into [2^14, 2^15), below the FP16 max
ROW_PAD = 32  # shared-memory row padding: rows 4 apart start 128 bytes apart (no bank conflicts)
SMEM_LIMIT = 227 * 1024
NO_COLUMN = 0x7FFFFFFF  # running-best column of a thread that has seen no column yet
LOWEST = -3.0e38  # running-best value of a thread that has seen no column yet


@dsl_user_op
def _gdc_wait(*, loc=None, ip=None) -> None:
    """``griddepcontrol.wait`` with a memory clobber: no load may be hoisted above it."""
    llvm.inline_asm(
        None,
        [],
        "griddepcontrol.wait;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mma_f16_m16n8k16(a, b, c: cute.Tensor, *, loc=None, ip=None):
    """``mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32``: ``a`` 8 FP16 (4 regs), ``b`` 4 FP16
    (2 regs), ``c`` 4 FP32; returns the 4 FP32 accumulators."""
    a32 = cute.recast_tensor(a, Int32, loc=loc, ip=ip)
    b32 = cute.recast_tensor(b, Int32, loc=loc, ip=ip)
    f32 = Float32.mlir_type
    out = llvm.inline_asm(
        llvm.StructType.get_literal([f32] * 4),
        [a32[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b32[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], f32, loc=loc),
        [llvm.extractvalue(f32, out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 4, Float32)


@dsl_user_op
def _fp8x4_to_fp16x4(x: Uint32, *, loc=None, ip=None):
    """Four packed E4M3 values -> four FP16 values, as two packed ``Uint32`` (exact)."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 $1, hi;\n\t"
        "}\n",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 2, Uint32)


@dsl_user_op
def _bulk_copy_g2s(dst, src, nbytes, mbar, *, loc=None, ip=None) -> None:
    """``nbytes`` (a multiple of 16, both addresses 16-byte aligned) from global ``src`` to shared
    ``dst``, completing as transaction bytes on the shared mbarrier ``mbar``."""
    llvm.inline_asm(
        None,
        [
            dst.toint(loc=loc, ip=ip).ir_value(),
            src.llvm_ptr,
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            mbar.toint(loc=loc, ip=ip).ir_value(),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atomic_max_shared_u32(ptr, value, *, loc=None, ip=None) -> None:
    """``*ptr = max(*ptr, value)`` on a shared-memory ``u32`` (non-negative FP32 bit patterns)."""
    llvm.inline_asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(), Uint32(value).ir_value(loc=loc, ip=ip)],
        "red.shared.max.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_max_s64_global(ptr, lo, hi, *, loc=None, ip=None) -> None:
    """``*ptr = max(*ptr, (hi << 32) | lo)`` on a global signed 64-bit word (relaxed, GPU scope)."""
    llvm.inline_asm(
        None,
        [
            ptr.llvm_ptr,
            Int32(lo).ir_value(loc=loc, ip=ip),
            Int32(hi).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b64 k;\n\t"
        "mov.b64 k, {$1, $2};\n\t"
        "red.relaxed.gpu.global.max.s64 [$0], k;\n\t"
        "}\n",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


class MXFP8ArgmaxGemvKernel:
    """``num_rows`` (<= 16) x K x N MXFP8 GEMV over ``num_ctas`` persistent CTAs, optionally with fused argmax."""

    def __init__(
        self,
        *,
        num_rows: int,
        k_extent: int,
        num_ctas: int,
        slice_blocks: int = 8,
        use_pdl: bool = True,
        argmax: bool = True,
    ) -> None:
        """``slice_blocks``: scale blocks of K per warp (2, 4, 8 or 16); the CTA has
        ``K / (32 * slice_blocks)`` warps."""
        if not 1 <= num_rows <= 2 * TOKENS:
            raise ValueError(f"num_rows={num_rows} must be in [1, {2 * TOKENS}]")
        if slice_blocks not in (2, 4, 8, 16):
            raise ValueError(f"slice_blocks={slice_blocks} must be 2, 4, 8 or 16")
        if k_extent % (BLOCK * slice_blocks) or k_extent % 512:
            raise ValueError(
                f"K={k_extent} must be a multiple of 512 and of {BLOCK * slice_blocks}"
            )
        self.warps = k_extent // (BLOCK * slice_blocks)
        if not 4 <= self.warps <= 32:
            raise ValueError(
                f"K={k_extent} with {slice_blocks} blocks per warp: {self.warps} warps"
            )
        self.num_rows = num_rows
        self.token_tiles = 1 if num_rows <= TOKENS else 2
        self.tokens_pad = TOKENS * self.token_tiles
        self.k_extent = k_extent
        self.slice_blocks = slice_blocks
        self.w_row_bytes = k_extent
        self.scale_row = k_extent // BLOCK
        self.row_stride = self.w_row_bytes + ROW_PAD
        self.num_ctas = num_ctas
        self.use_pdl = use_pdl
        self.argmax = argmax
        if self.smem_bytes() > SMEM_LIMIT:
            raise ValueError(f"K={k_extent}, M={num_rows}: {self.smem_bytes()} bytes of smem")

    def smem_bytes(self) -> int:
        weights = 2 * ROWS * self.row_stride
        scales = 2 * ROWS * self.scale_row
        partials = 2 * self.warps * 4 * self.token_tiles * 32 * 4
        # running-best exchange: (value, column) per (warp, lane quad)
        best = 2 * 4 * self.token_tiles * 4 * 4
        return weights + scales + partials + best + 16 + self.tokens_pad * 4 + 512

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gS: cute.Tensor,
        gOut: cute.Tensor,
        vocab_offset: Int32,
        stream,
    ) -> None:
        """``gOut``: logits ``[M, N]`` (``argmax=False``), or ``keys`` ``[M, 2]`` Int32 (the two words
        of one int64 per row, little endian; ``argmax=True``)."""
        copy_a = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        copy_w = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=64)
        copy_s = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            Uint8,
            num_bits_per_copy=8 * self.slice_blocks,
        )
        self.kernel(gA, gB, gS, gOut, vocab_offset, copy_a, copy_w, copy_s).launch(
            grid=[self.num_ctas, 1, 1],
            block=[32 * self.warps, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gS: cute.Tensor,
        gOut: cute.Tensor,
        vocab_offset: Int32,
        copy_a: cute.CopyAtom,
        copy_w: cute.CopyAtom,
        copy_s: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        g = lane // 4
        q = lane % 4

        num_rows: cutlass.Constexpr = self.num_rows
        tokens_pad: cutlass.Constexpr = self.tokens_pad
        token_tiles: cutlass.Constexpr = self.token_tiles
        slice_blocks: cutlass.Constexpr = self.slice_blocks
        warps: cutlass.Constexpr = self.warps
        num_ctas: cutlass.Constexpr = self.num_ctas

        n = cute.size(gB, mode=[0])
        num_groups = (n + GROUP_ROWS - 1) // GROUP_ROWS
        # The grid has at most num_groups CTAs, so every CTA has at least one group.
        my_groups = (num_groups - bidx + num_ctas - 1) // num_ctas
        my_tiles = (my_groups + 1) // 2

        smem = _SmemAllocator()
        sW = smem.allocate_tensor(
            Uint8,
            cute.make_layout(
                (2, ROWS, self.row_stride), stride=(ROWS * self.row_stride, self.row_stride, 1)
            ),
            byte_alignment=128,
        )
        sS = smem.allocate_tensor(
            Uint8,
            cute.make_layout(
                (2, ROWS, self.scale_row), stride=(ROWS * self.scale_row, self.scale_row, 1)
            ),
            byte_alignment=128,
        )
        # Partial tiles: (slot, warp, MMA tile * 4 + accumulator element, lane).
        sPart = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (2, warps, 4 * token_tiles, 32),
                stride=(warps * 4 * token_tiles * 32, 4 * token_tiles * 32, 32, 1),
            ),
            byte_alignment=16,
        )
        # Running bests of the reducing warps: (warp, lane quad q).
        sBestV = smem.allocate_tensor(
            Float32, cute.make_layout((4 * token_tiles, 4), stride=(4, 1)), byte_alignment=16
        )
        sBestI = smem.allocate_tensor(
            Int32, cute.make_layout((4 * token_tiles, 4), stride=(4, 1)), byte_alignment=16
        )
        mbar = smem.allocate_tensor(Int64, cute.make_layout((2,)), byte_alignment=16)
        sAmax = smem.allocate_tensor(Uint32, cute.make_layout((tokens_pad,)), byte_alignment=16)

        if tidx == 0:
            cute.arch.mbarrier_init(mbar.iterator, 1)
            cute.arch.mbarrier_init(mbar.iterator + 1, 1)
            cute.arch.mbarrier_init_fence()
        if tidx < tokens_pad:
            sAmax[tidx] = Uint32(0)
        cute.arch.sync_threads()

        # Weights do not depend on the producer: issue the first two tiles before the wait.
        if warp == 0:
            self._issue_tile(gB, gS, sW, sS, mbar, 0, Int32(0), my_groups, bidx, n, lane)
            if my_tiles > 1:
                self._issue_tile(gB, gS, sW, sS, mbar, 1, Int32(1), my_groups, bidx, n, lane)

        if const_expr(self.use_pdl):
            _gdc_wait()

        # act[t, b, :]: row 8t + g, K 32 * (warp * slice_blocks + b) + 8q + 0..7.
        act = cute.make_rmem_tensor(
            cute.make_layout((token_tiles, slice_blocks, 8), stride=(slice_blocks * 8, 8, 1)),
            Float16,
        )
        inv = self._stage_activation(gA, act, sAmax, copy_a, tidx, warp, g, q)
        if const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()

        a0 = cute.make_rmem_tensor(cute.make_layout((8,)), Float16)
        a1 = cute.make_rmem_tensor(cute.make_layout((8,)), Float16)
        a0_u32 = cute.recast_tensor(a0, Uint32)
        a1_u32 = cute.recast_tensor(a1, Uint32)
        b0 = cute.make_rmem_tensor(cute.make_layout((4,)), Float16)
        b1 = cute.make_rmem_tensor(cute.make_layout((4,)), Float16)
        b0_u32 = cute.recast_tensor(b0, Uint32)
        b1_u32 = cute.recast_tensor(b1, Uint32)
        zero = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        zero.fill(0.0)
        part = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        acc = cute.make_rmem_tensor(cute.make_layout((token_tiles, 4), stride=(4, 1)), Float32)
        w_bytes: cutlass.Constexpr = 8
        w_lo = cute.make_rmem_tensor(cute.make_layout((w_bytes,)), Uint8)
        w_hi = cute.make_rmem_tensor(cute.make_layout((w_bytes,)), Uint8)
        w_lo_u32 = cute.recast_tensor(w_lo, Uint32)
        w_hi_u32 = cute.recast_tensor(w_hi, Uint32)
        s_lo = cute.make_rmem_tensor(cute.make_layout((slice_blocks,)), Uint8)
        s_hi = cute.make_rmem_tensor(cute.make_layout((slice_blocks,)), Uint8)
        scale_bits = cute.make_rmem_tensor(cute.make_layout((2,)), Int32)
        scale = cute.recast_tensor(scale_bits, Float32)
        act_u32 = cute.recast_tensor(act, Uint32)
        # Running best (value, column) of the output this thread reduces in every tile.
        best_v = cute.make_rmem_tensor(cute.make_layout((1,)), Float32)
        best_i = cute.make_rmem_tensor(cute.make_layout((1,)), Int32)
        best_v[0] = Float32(LOWEST)
        best_i[0] = Int32(NO_COLUMN)

        sW_v = cute.logical_divide(sW, (None, None, w_bytes))
        sS_v = cute.logical_divide(sS, (None, None, slice_blocks))
        # This warp's first w_bytes-wide column of a weight row (a 32-element block is 4 such columns).
        col0 = warp * slice_blocks * 4

        for i in cutlass.range(0, my_tiles):
            slot = i % 2
            cute.arch.mbarrier_wait(mbar.iterator + slot, (i // 2) % 2)
            cute.copy(copy_s, sS_v[slot, g, (None, warp)], s_lo)
            cute.copy(copy_s, sS_v[slot, g + 8, (None, warp)], s_hi)
            acc.fill(0.0)
            for b in cutlass.range_constexpr(slice_blocks):
                cute.copy(copy_w, sW_v[slot, g, (None, col0 + 4 * b + q)], w_lo)
                cute.copy(copy_w, sW_v[slot, g + 8, (None, col0 + 4 * b + q)], w_hi)
                lo01 = _fp8x4_to_fp16x4(w_lo_u32[0])
                lo23 = _fp8x4_to_fp16x4(w_lo_u32[1])
                hi01 = _fp8x4_to_fp16x4(w_hi_u32[0])
                hi23 = _fp8x4_to_fp16x4(w_hi_u32[1])
                # Step j: (row g, 4j..4j+1), (row g+8, same), (row g, 4j+2..4j+3), (row g+8, same).
                a0_u32[0] = lo01[0]
                a0_u32[1] = hi01[0]
                a0_u32[2] = lo01[1]
                a0_u32[3] = hi01[1]
                a1_u32[0] = lo23[0]
                a1_u32[1] = hi23[0]
                a1_u32[2] = lo23[1]
                a1_u32[3] = hi23[1]
                # UE8M0 -> the bit pattern of 2^(s - 127).
                scale_bits[0] = s_lo[b].to(Int32) << 23
                scale_bits[1] = s_hi[b].to(Int32) << 23
                for t in cutlass.range_constexpr(token_tiles):
                    b0_u32[0] = act_u32[t, b, 0]
                    b0_u32[1] = act_u32[t, b, 1]
                    b1_u32[0] = act_u32[t, b, 2]
                    b1_u32[1] = act_u32[t, b, 3]
                    part.store(_mma_f16_m16n8k16(a0, b0, zero))
                    part.store(_mma_f16_m16n8k16(a1, b1, part))
                    for e in cutlass.range_constexpr(4):
                        acc[t, e] = acc[t, e] + scale[e // 2] * part[e]
            for t in cutlass.range_constexpr(token_tiles):
                for e in cutlass.range_constexpr(4):
                    sPart[slot, warp, 4 * t + e, lane] = acc[t, e]
            # Every warp is done with this slot's weights and has written its partial tile.
            cute.arch.sync_threads()
            if warp == 0:
                if i + 2 < my_tiles:
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self._issue_tile(gB, gS, sW, sS, mbar, slot, i + 2, my_groups, bidx, n, lane)
            if tidx < 128 * token_tiles:
                # Thread (t, e, lane) sums element e of the lane's MMA tile t over the warps.
                total = Float32(0.0)
                for w in cutlass.range_constexpr(warps):
                    total = total + sPart[slot, w, tidx // 32, lane]
                e = (tidx // 32) % 4
                token = TOKENS * (tidx // 128) + 2 * q + e % 2
                group = 2 * i + e // 2  # MMA rows 0..7: the tile's first group, 8..15: its second
                row = (bidx + group * num_ctas) * GROUP_ROWS + g
                if token < num_rows:
                    if group < my_groups:
                        if row < n:
                            value = total * inv
                            if const_expr(self.argmax):
                                # Columns of one thread increase with i: strict > keeps the first.
                                if value > best_v[0]:
                                    best_v[0] = value
                                    best_i[0] = row
                            else:
                                gOut[token, row] = value.to(gOut.element_type)

        if const_expr(self.argmax):
            self._reduce_and_publish(
                gOut, best_v, best_i, sBestV, sBestI, vocab_offset, tidx, warp, lane, q
            )

    @cute.jit
    def _reduce_and_publish(
        self, gOut, best_v, best_i, sBestV, sBestI, vocab_offset, tidx, warp, lane, q
    ) -> None:
        """CTA argmax per row: shuffle over the 8 lane groups that share a row (lanes 4g + q), then
        the two warps of that row through shared memory; one thread per row folds the packed key into
        ``gOut[row]`` (``red.global.max.s64``)."""
        num_rows: cutlass.Constexpr = self.num_rows
        token_tiles: cutlass.Constexpr = self.token_tiles
        for k in cutlass.range_constexpr(3):
            off = 4 << k
            ov = cute.arch.shuffle_sync_bfly(best_v[0], offset=off)
            oi = cute.arch.shuffle_sync_bfly(best_i[0], offset=off)
            if ov > best_v[0]:
                best_v[0] = ov
                best_i[0] = oi
            else:
                if ov == best_v[0]:
                    if oi < best_i[0]:
                        best_i[0] = oi
        if warp < 4 * token_tiles:
            if lane < 4:
                sBestV[warp, q] = best_v[0]
                sBestI[warp, q] = best_i[0]
        cute.arch.sync_threads()
        if tidx < num_rows:
            # Row r = 8 * tile + 2 * qq + p lives in warps 4 * tile + p and 4 * tile + p + 2, quad qq.
            tile = tidx // TOKENS
            within = tidx % TOKENS
            qq = within // 2
            w0 = 4 * tile + within % 2
            v = sBestV[w0, qq]
            c = sBestI[w0, qq]
            v1 = sBestV[w0 + 2, qq]
            c1 = sBestI[w0 + 2, qq]
            res_v = cute.make_rmem_tensor(cute.make_layout((1,)), Float32)
            res_i = cute.make_rmem_tensor(cute.make_layout((1,)), Int32)
            res_v[0] = v
            res_i[0] = c
            if v1 > v:
                res_v[0] = v1
                res_i[0] = c1
            else:
                if v1 == v:
                    if c1 < c:
                        res_i[0] = c1
            if res_i[0] != Int32(NO_COLUMN):
                bits = cute.make_rmem_tensor(cute.make_layout((1,)), Int32)
                bits_f = cute.recast_tensor(bits, Float32)
                bits_f[0] = res_v[0]
                if bits[0] == Int32(-2147483648):
                    bits[0] = Int32(0)  # -0.0 ranks as +0.0 (equal under the float compare)
                if bits[0] < 0:
                    bits[0] = bits[0] ^ Int32(0x7FFFFFFF)
                # 0xFFFFFFFF - id == ~id as a 32-bit word.
                lo = Int32(-1) - (vocab_offset + res_i[0])
                _red_max_s64_global(gOut[tidx, None].iterator, lo, bits[0])

    @cute.jit
    def _issue_tile(self, gB, gS, sW, sS, mbar, slot, tile, my_groups, bidx, n, lane) -> None:
        """Bulk copies of this CTA's tile ``tile`` (its groups ``2 tile`` and ``2 tile + 1``, the
        second only if it exists) into ``slot``: weight and scale rows, lanes 0..15 of the calling
        warp, one row each. Rows past N re-read row N - 1 (their outputs are not used)."""
        row_bytes: cutlass.Constexpr = self.w_row_bytes + self.scale_row
        if lane == 0:
            groups = cutlass.min(my_groups - 2 * tile, Int32(2))
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar.iterator + slot, groups * (GROUP_ROWS * row_bytes)
            )
        if lane < ROWS:
            group = 2 * tile + lane // GROUP_ROWS
            if group < my_groups:
                row = cutlass.min(
                    (bidx + group * self.num_ctas) * GROUP_ROWS + lane % GROUP_ROWS, n - 1
                )
                _bulk_copy_g2s(
                    sW[slot, lane, None].iterator,
                    gB[row, None].iterator,
                    self.w_row_bytes,
                    mbar.iterator + slot,
                )
                _bulk_copy_g2s(
                    sS[slot, lane, None].iterator,
                    gS[row, None].iterator,
                    self.scale_row,
                    mbar.iterator + slot,
                )

    @cute.jit
    def _stage_activation(self, gA, act, sAmax, copy_a, tidx, warp, g, q) -> Float32:
        """Loads this warp's K slice of A rows ``8t + g`` into ``act`` as FP16 times ``2^e_row``
        (zeros past M) and returns ``2^-e_row`` of the output row this thread reduces (row
        ``8 (tidx // 128) + 2q + (tidx // 32) % 2``; unused by the other threads)."""
        num_rows: cutlass.Constexpr = self.num_rows
        tokens_pad: cutlass.Constexpr = self.tokens_pad
        token_tiles: cutlass.Constexpr = self.token_tiles
        slice_blocks: cutlass.Constexpr = self.slice_blocks
        gA8 = cute.logical_divide(gA, (None, 8))
        raw = cute.make_rmem_tensor(
            cute.make_layout((token_tiles, slice_blocks, 8), stride=(slice_blocks * 8, 8, 1)),
            BFloat16,
        )
        raw.fill(0.0)
        amax_f = cute.make_rmem_tensor(cute.make_layout((1,)), Float32)
        amax_u = cute.recast_tensor(amax_f, Uint32)
        factor_bits = cute.make_rmem_tensor(cute.make_layout((1,)), Int32)
        factor = cute.recast_tensor(factor_bits, Float32)
        for t in cutlass.range_constexpr(token_tiles):
            if TOKENS * t + g < num_rows:
                for b in cutlass.range_constexpr(slice_blocks):
                    kv = (warp * slice_blocks + b) * 4 + q
                    cute.copy(copy_a, gA8[TOKENS * t + g, (None, kv)], raw[t, b, None])
        for t in cutlass.range_constexpr(token_tiles):
            amax = Float32(0.0)
            for b in cutlass.range_constexpr(slice_blocks):
                for e in cutlass.range_constexpr(8):
                    xv = raw[t, b, e].to(Float32)
                    amax = cute.arch.fmax(amax, cute.arch.fmax(xv, -xv))
            for i in cutlass.range_constexpr(2):
                amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << i))
            if q == 0:
                if TOKENS * t + g < num_rows:
                    amax_f[0] = amax
                    _atomic_max_shared_u32(sAmax.iterator + (TOKENS * t + g), amax_u[0])
        cute.arch.sync_threads()
        for t in cutlass.range_constexpr(token_tiles):
            shift = HEADROOM_EXP + 127 - Int32(sAmax[TOKENS * t + g] >> 23)
            shift = cutlass.max(cutlass.min(shift, Int32(126)), Int32(-126))
            factor_bits[0] = (shift + 127) << 23
            for b in cutlass.range_constexpr(slice_blocks):
                for e in cutlass.range_constexpr(8):
                    act[t, b, e] = (raw[t, b, e].to(Float32) * factor[0]).to(Float16)
        out_row = (TOKENS * (tidx // 128) + 2 * q + (tidx // 32) % 2) % tokens_pad
        shift = HEADROOM_EXP + 127 - Int32(sAmax[out_row] >> 23)
        shift = cutlass.max(cutlass.min(shift, Int32(126)), Int32(-126))
        factor_bits[0] = (127 - shift) << 23
        return factor[0]
