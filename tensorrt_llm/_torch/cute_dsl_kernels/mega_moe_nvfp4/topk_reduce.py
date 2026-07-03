# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device combine reduce: collapse per-(token, topk) fc2 cells into one row.

The combine step writes one fc2 output per ``(token, topk)`` cell; this reduces
over the topk axis into the token-centric ``(token, hidden)`` output. The wire
format is a :class:`~src.token_comm.CombineFormat`:

    bf16          -- no staging: bf16 terms reduced directly.
    32e4m3xe8m0   -- MXFP8: fp8 e4m3 data + per-32 e8m0 (power-of-2) scale.
    16e2m1xbf16   -- fp4 e2m1 data + per-16 bf16 amax (one level, no global);
                     dequant per element x = fp4 * (amax * (1 / 6)).

Task partition: each worker owns one ``(token, hidden_tile)`` and loops topk; the
flat worker index decodes into ``(token_idx, hidden_tile_idx)`` via a constant
divide by ``hidden_tiles``. The per-block scale is broadcast to a logical
per-hidden view (stride 0) so it tiles by the same worker index as the data. The
activation load stays in the topk loop (too large to hoist); the small scale and
score loads are hoisted ahead of the loop when topk is small.
"""

from __future__ import annotations

import os
from typing import ClassVar, Dict, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, T

from .megamoe_constants import Nvfp4E2M1RcpLimit
from .token_comm import CombineFormat

# ---------------------------------------------------------------------------
# fp4 (e2m1) -> fp32 register decode.
#
# Blackwell has no e2m1->f32 upconvert: the framework's ``term.load().to(f32)``
# lowers to an ALU subnormal-normalization path (~60% DRAM SOL). Both helpers
# below force a table-driven decode instead; ``e2m1_reg`` (N e2m1 codes, N % 8
# == 0) is read as packed b32 words and N fp32 values are written into
# ``fp32_reg`` in code order. The 16 e2m1 values are exact in fp32, so the two
# decoders are bit-for-bit identical (cross-check the optimal one against the
# cvt one over all 16 codes).
# ---------------------------------------------------------------------------


@cute.jit
def cvt_e2m1_to_fp32_cvt_ptx(e2m1_reg: cute.Tensor,
                             fp32_reg: cute.Tensor) -> None:
    """Decode via the e2m1->f16 HW cvt (``cvt.rn.f16x2.e2m1x2``) then widen f16->f32.

    Safe baseline: the per-pair ``cvt`` instruction is itself a HW PRMT+F2FP, so
    the e2m1->f16 step already avoids ALU normalization; f16->f32 is one cheap
    ``cvt.f32.f16`` per element.
    """
    src_words = cute.recast_tensor(e2m1_reg, Int32)  # (N,) e2m1 -> (N/8,) b32
    for w in cutlass.range_constexpr(cute.size(src_words)):
        res = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32()] * 8),
            [src_words[w].ir_value()],
            "{\n"
            "  .reg .b8  b0, b1, b2, b3;\n"
            "  .reg .b32 p0, p1, p2, p3;\n"
            "  .reg .b16 c0, d0, c1, d1, c2, d2, c3, d3;\n"
            "  mov.b32 {b0, b1, b2, b3}, $8;\n"
            "  cvt.rn.f16x2.e2m1x2 p0, b0;\n"
            "  cvt.rn.f16x2.e2m1x2 p1, b1;\n"
            "  cvt.rn.f16x2.e2m1x2 p2, b2;\n"
            "  cvt.rn.f16x2.e2m1x2 p3, b3;\n"
            "  mov.b32 {c0, d0}, p0;\n"
            "  mov.b32 {c1, d1}, p1;\n"
            "  mov.b32 {c2, d2}, p2;\n"
            "  mov.b32 {c3, d3}, p3;\n"
            "  cvt.f32.f16 $0, c0;\n"
            "  cvt.f32.f16 $1, d0;\n"
            "  cvt.f32.f16 $2, c1;\n"
            "  cvt.f32.f16 $3, d1;\n"
            "  cvt.f32.f16 $4, c2;\n"
            "  cvt.f32.f16 $5, d2;\n"
            "  cvt.f32.f16 $6, c3;\n"
            "  cvt.f32.f16 $7, d3;\n"
            "}",
            "=f,=f,=f,=f,=f,=f,=f,=f,r",
            has_side_effects=False,
        )
        for i in cutlass.range_constexpr(8):
            fp32_reg[w * 8 + i] = Float32(llvm.extractvalue(T.f32(), res, [i]))


@cute.jit
def cvt_e2m1_to_fp32_optimal_ptx(e2m1_reg: cute.Tensor,
                                 fp32_reg: cute.Tensor) -> None:
    """Decode via a register-resident bf16 LUT + PRMT, landing fp32 directly.

    The 8 e2m1 magnitudes ``{0,.5,1,1.5,2,3,4,6}`` are exact bf16, so a PRMT
    byte-gather of the per-magnitude hi/lo bytes builds the bf16; ``fp32 =
    bf16 << 16`` makes the widen free. Sign (nibble bit3) is spread to the 4
    output-byte MSBs with one ``prmt`` of ``{word<<4, word}`` (selector 0x5140),
    avoiding the non-uniform shift the 4-bit-vs-8-bit stride would otherwise need.
    """
    src_words = cute.recast_tensor(e2m1_reg, Int32)  # (N,) e2m1 -> (N/8,) b32
    dst_words = cute.recast_tensor(fp32_reg, Int32)  # write fp32 bit patterns
    for w in cutlass.range_constexpr(cute.size(src_words)):
        res = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32()] * 8),
            [src_words[w].ir_value()],
            "{\n"
            "  .reg .b32 ha, hb, la, lb, inh, wl, ih, il, hl, ll, hh, lh, sl, sh, p0, p1, p2, p3;\n"
            "  mov.b32 ha, 0x3F3F3F00;\n"  # hi byte LUT, magnitudes 0..3
            "  mov.b32 hb, 0x40404040;\n"  # hi byte LUT, magnitudes 4..7
            "  mov.b32 la, 0xC0800000;\n"  # lo byte LUT, magnitudes 0..3
            "  mov.b32 lb, 0xC0804000;\n"  # lo byte LUT, magnitudes 4..7
            "  shr.b32 inh, $8, 16;\n"  # high 4 elements -> low 16 bits
            "  and.b32 il, $8, 0x00007777;\n"  # low 4 magnitude indices (clear sign)
            "  and.b32 ih, inh, 0x00007777;\n"  # high 4 magnitude indices
            "  prmt.b32 hl, ha, hb, il;\n"  # hi bytes for e0..e3
            "  prmt.b32 ll, la, lb, il;\n"  # lo bytes for e0..e3
            "  prmt.b32 hh, ha, hb, ih;\n"  # hi bytes for e4..e7
            "  prmt.b32 lh, la, lb, ih;\n"  # lo bytes for e4..e7
            "  shl.b32 wl, $8, 4;\n"
            "  prmt.b32 sl, wl, $8, 0x5140;\n"  # gather s0..s3 to byte MSBs
            "  and.b32 sl, sl, 0x80808080;\n"
            "  or.b32  hl, hl, sl;\n"
            "  shl.b32 wl, inh, 4;\n"
            "  prmt.b32 sh, wl, inh, 0x5140;\n"  # gather s4..s7 to byte MSBs
            "  and.b32 sh, sh, 0x80808080;\n"
            "  or.b32  hh, hh, sh;\n"
            "  prmt.b32 p0, ll, hl, 0x5140;\n"  # {bf16(e0), bf16(e1)}
            "  prmt.b32 p1, ll, hl, 0x7362;\n"  # {bf16(e2), bf16(e3)}
            "  prmt.b32 p2, lh, hh, 0x5140;\n"  # {bf16(e4), bf16(e5)}
            "  prmt.b32 p3, lh, hh, 0x7362;\n"  # {bf16(e6), bf16(e7)}
            "  shl.b32 $0, p0, 16;\n"
            "  and.b32 $1, p0, 0xFFFF0000;\n"
            "  shl.b32 $2, p1, 16;\n"
            "  and.b32 $3, p1, 0xFFFF0000;\n"
            "  shl.b32 $4, p2, 16;\n"
            "  and.b32 $5, p2, 0xFFFF0000;\n"
            "  shl.b32 $6, p3, 16;\n"
            "  and.b32 $7, p3, 0xFFFF0000;\n"
            "}",
            "=r,=r,=r,=r,=r,=r,=r,=r,r",
            has_side_effects=False,
        )
        for i in cutlass.range_constexpr(8):
            dst_words[w * 8 + i] = Int32(llvm.extractvalue(T.i32(), res, [i]))


class TopkReduce:
    """Combine reduce for a fixed ``(hidden, num_topk, combine_format)``.

    ``__init__`` pins the static shape and format (and the derived launch
    geometry); ``__call__`` (a ``@cute.jit`` launcher) sizes a 1D grid from the
    runtime token count and dispatches the format's kernel. The caller owns the
    torch->cute conversion and the ``cute.compile`` / ``aot_compile``.
    """

    _threads: ClassVar[int] = 128
    # combine_format.name -> hidden elements per worker (one LDG of data:
    # bf16 8*2B=16B, e4m3 16*1B=16B, e2m1 16*0.5B=8B). For quantized formats this
    # stays <= the scale block, so each worker reads exactly one scale entry.
    _hidden_per_thread: ClassVar[Dict[str, int]] = {
        "bf16": 8,
        "32e4m3xe8m0": 16,
        "16e2m1xbf16": 16,
    }
    # topk count at/below which the scale + score loads are hoisted ahead of the
    # topk loop (small enough to not bloat registers; a CTA-broadcast read).
    _prefetch_limit: ClassVar[int] = 16

    def __init__(self, hidden: int, num_topk: int,
                 combine_format: CombineFormat) -> None:
        self.hidden = int(hidden)
        self.num_topk = int(num_topk)
        self.combine_format = combine_format
        self.hidden_per_thread = self._hidden_per_thread[combine_format.name]
        # hidden must tile cleanly both into worker slices and into scale blocks.
        align = max(combine_format.scale_block or self.hidden_per_thread,
                    self.hidden_per_thread)
        if self.hidden % align != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be divisible by max(scale_block, "
                f"hidden_per_thread) = {align} for combine_format {combine_format}."
            )
        self.hidden_tiles = self.hidden // self.hidden_per_thread
        # tail guard only needed when the worker count per token is not a whole
        # number of CTAs; prefetch only when topk is small enough to hoist.
        self.require_predicate = self.hidden_tiles % self._threads != 0
        self.prefetch = self.num_topk <= self._prefetch_limit

    # -- launcher -------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        combine_quant: cute.Tensor,  # (token, topk, hidden)
        combine_sf: Optional[cute.Tensor],  # (token, topk, hidden)
        reduced_output: cute.Tensor,  # (token, hidden)
        topk_score: Optional[cute.Tensor],  # (token, topk)
        stream: cuda.CUstream,
    ):
        threads = self._threads
        total_workers = reduced_output.shape[0] * self.hidden_tiles
        grid = [(total_workers + threads - 1) // threads, 1, 1]
        block = [threads, 1, 1]

        combine_quant = cute.make_tensor(
            combine_quant.iterator,
            cute.make_layout(
                (combine_quant.shape[0], self.num_topk, self.hidden),
                stride=combine_quant.stride))
        reduced_output = cute.make_tensor(
            reduced_output.iterator,
            cute.make_layout((reduced_output.shape[0], self.hidden),
                             stride=reduced_output.stride))
        if cutlass.const_expr(topk_score is not None):
            topk_score = cute.make_tensor(
                topk_score.iterator,
                cute.make_layout((topk_score.shape[0], self.num_topk),
                                 stride=topk_score.stride))

        if cutlass.const_expr(not self.combine_format.is_quantized):
            self._reduce_bf16(combine_quant, topk_score, reduced_output).launch(
                grid=grid,
                block=block,
                stream=stream,
            )
            return

        # The mega kernel hands sf in already as the depth-2 broadcast layout; a
        # plain (torch) sf is depth-1 and gets its hidden mode split into
        # (sf_vec, hidden/sf_vec):(0, s_h) so logical hidden h reads block h//sf_vec.
        sf_vec = self.combine_format.scale_block
        if cutlass.const_expr(cute.depth(combine_sf.layout) >= 2):
            sf = cute.make_tensor(
                combine_sf.iterator,
                cute.make_layout((combine_sf.shape[0], self.num_topk,
                                  (sf_vec, self.hidden // sf_vec)),
                                 stride=combine_sf.stride))
        else:
            sf = cute.make_tensor(
                combine_sf.iterator,
                cute.make_layout(
                    (combine_sf.shape[0], self.num_topk,
                     (sf_vec, self.hidden // sf_vec)),
                    stride=(combine_sf.stride[0], combine_sf.stride[1],
                            (0, combine_sf.stride[2])),
                ),
            )

        if cutlass.const_expr(
                self.combine_format.act_dtype is cutlass.Float8E4M3FN):
            self._reduce_mxfp8(combine_quant, sf, topk_score,
                               reduced_output).launch(
                                   grid=grid,
                                   block=block,
                                   stream=stream,
                               )
        else:
            self._reduce_fp4(combine_quant, sf, topk_score,
                             reduced_output).launch(
                                 grid=grid,
                                 block=block,
                                 stream=stream,
                             )

    @cute.jit
    def _mark_alignment(self, tensor: cute.Tensor,
                        align_bytes: int) -> cute.Tensor:
        p = tensor.iterator
        return cute.make_tensor(
            cute.make_ptr(p.dtype,
                          p.toint(),
                          p.memspace,
                          assumed_align=align_bytes),
            tensor.layout,
        )

    # -- kernels --------------------------------------------------------------

    @cute.kernel
    def _reduce_bf16(
        self,
        combine_output: cute.Tensor,
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
    ):
        threads = self._threads
        hidden_per_thread = self.hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        needs_guard = self.require_predicate
        prefetch = self.prefetch
        out_dtype = reduced_output.element_type

        worker_idx = cute.arch.block_idx()[0] * Int32(
            threads) + cute.arch.thread_idx()[0]
        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        score_dtype = topk_score.dtype if cutlass.const_expr(
            topk_score is not None) else cutlass.Float32
        score_reg = cute.make_rmem_tensor((num_topk, ), score_dtype)

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            terms = cute.zipped_divide(
                combine_output[token_idx, None, None],
                (num_topk, hidden_per_thread),
            )[(None, None), (0, hidden_tile_idx)]
            # (token, hidden) -> (hidden_per_thread)
            dst = cute.zipped_divide(
                reduced_output[token_idx, None],
                (hidden_per_thread, ),
            )[(None, ), (hidden_tile_idx, )]

            if cutlass.const_expr(topk_score is not None):
                if cutlass.const_expr(prefetch):
                    cute.autovec_copy(topk_score[token_idx, None], score_reg)
            else:
                for k in cutlass.range_constexpr(num_topk):
                    score_reg[k] = score_dtype(1)

            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.BFloat16,
                num_bits_per_copy=128,
            )
            acc = cute.make_rmem_tensor((hidden_per_thread, ), cutlass.Float32)

            for k in cutlass.range_constexpr(0, num_topk, 1):
                term = cute.make_rmem_tensor((hidden_per_thread, ),
                                             cutlass.BFloat16)
                cute.copy(load_atom, terms[k, None], term)
                if cutlass.const_expr(topk_score is not None and not prefetch):
                    score_reg[k] = topk_score[token_idx, Int32(k)]
                score_pair = (Float32(score_reg[k]), Float32(score_reg[k]))

                for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                    value_pair = (Float32(term[i]), Float32(term[i + 1]))
                    if cutlass.const_expr(k != 0):
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(
                            value_pair, score_pair, (acc[i], acc[i + 1]))
                    else:
                        if cutlass.const_expr(topk_score is not None):
                            acc[i], acc[i + 1] = cute.arch.mul_packed_f32x2(
                                value_pair, score_pair)
                        else:
                            acc[i] = value_pair[0]
                            acc[i + 1] = value_pair[1]

            out = cute.make_rmem_tensor((hidden_per_thread, ), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                    out_dtype,
                                    num_bits_per_copy=128),
                out,
                self._mark_alignment(dst,
                                     hidden_per_thread * out_dtype.width // 8),
            )

    @cute.kernel
    def _reduce_mxfp8(
        self,
        combine_quant: cute.Tensor,
        combine_sf: cute.
        Tensor,  # depth-2 broadcast view: logical (token, topk, hidden) e8m0
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
    ):
        threads = self._threads
        hidden_per_thread = self.hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        needs_guard = self.require_predicate
        prefetch = self.prefetch
        out_dtype = reduced_output.element_type

        worker_idx = cute.arch.block_idx()[0] * Int32(
            threads) + cute.arch.thread_idx()[0]
        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        score_dtype = topk_score.dtype if cutlass.const_expr(
            topk_score is not None) else cutlass.Float32
        score_reg = cute.make_rmem_tensor((num_topk, ), score_dtype)
        scale_reg = cute.make_rmem_tensor((num_topk, ), cutlass.Float8E8M0FNU)

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            codes = cute.zipped_divide(
                combine_quant[token_idx, None, None],
                (num_topk, hidden_per_thread),
            )[(None, None), (0, hidden_tile_idx)]
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            sf = cute.zipped_divide(
                combine_sf[token_idx, None, None],
                (num_topk, hidden_per_thread),
            )[(None, None), (0, hidden_tile_idx)]
            # (token, hidden) -> (hidden_per_thread)
            dst = cute.zipped_divide(
                reduced_output[token_idx, None],
                (hidden_per_thread, ),
            )[(None, ), (hidden_tile_idx, )]

            if cutlass.const_expr(topk_score is not None):
                if cutlass.const_expr(prefetch):
                    cute.autovec_copy(topk_score[token_idx, None], score_reg)
            else:
                for k in cutlass.range_constexpr(num_topk):
                    score_reg[k] = score_dtype(1)
            if cutlass.const_expr(prefetch):
                cute.autovec_copy(
                    sf[None, 0],
                    scale_reg)  # one scale per topk slot (stride-0 broadcast)

            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float8E4M3FN,
                num_bits_per_copy=128,
            )
            acc = cute.make_rmem_tensor((hidden_per_thread, ), cutlass.Float32)

            for k in cutlass.range_constexpr(0, num_topk, 1):
                term = cute.make_rmem_tensor((hidden_per_thread, ),
                                             cutlass.Float8E4M3FN)
                cute.copy(load_atom, codes[k, None], term)
                value = cute.make_rmem_tensor((hidden_per_thread, ),
                                              cutlass.Float32)
                value.store(term.load().to(cutlass.Float32))

                if cutlass.const_expr(not prefetch):
                    scale_reg[k] = sf[k, 0]
                    if cutlass.const_expr(topk_score is not None):
                        score_reg[k] = topk_score[token_idx, Int32(k)]

                scale = Float32(scale_reg[k])  # e8m0 -> f32
                scale_pair = (scale, scale)
                score_pair = (Float32(score_reg[k]), Float32(score_reg[k]))

                for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                    dequant_pair = cute.arch.mul_packed_f32x2(
                        (value[i], value[i + 1]), scale_pair)
                    if cutlass.const_expr(k != 0):
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(
                            dequant_pair, score_pair, (acc[i], acc[i + 1]))
                    else:
                        if cutlass.const_expr(topk_score is not None):
                            acc[i], acc[i + 1] = cute.arch.mul_packed_f32x2(
                                dequant_pair, score_pair)
                        else:
                            acc[i] = dequant_pair[0]
                            acc[i + 1] = dequant_pair[1]

            out = cute.make_rmem_tensor((hidden_per_thread, ), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                    out_dtype,
                                    num_bits_per_copy=256),
                out,
                self._mark_alignment(dst,
                                     hidden_per_thread * out_dtype.width // 8),
            )

    @cute.kernel
    def _reduce_fp4(
        self,
        combine_quant: cute.Tensor,  # (token, topk, hidden) e2m1 (logical)
        combine_sf: cute.
        Tensor,  # depth-2 broadcast view: logical (token, topk, hidden) bf16 amax
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
    ):
        threads = self._threads
        hidden_per_thread = self.hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        needs_guard = self.require_predicate
        prefetch = self.prefetch
        out_dtype = reduced_output.element_type

        worker_idx = cute.arch.block_idx()[0] * Int32(
            threads) + cute.arch.thread_idx()[0]
        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        score_dtype = topk_score.dtype if cutlass.const_expr(
            topk_score is not None) else cutlass.Float32
        score_reg = cute.make_rmem_tensor((num_topk, ), score_dtype)
        scale_reg = cute.make_rmem_tensor((num_topk, ), cutlass.BFloat16)

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            codes = cute.zipped_divide(
                combine_quant[token_idx, None, None],
                (num_topk, hidden_per_thread),
            )[(None, None), (0, hidden_tile_idx)]
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            sf = cute.zipped_divide(
                combine_sf[token_idx, None, None],
                (num_topk, hidden_per_thread),
            )[(None, None), (0, hidden_tile_idx)]
            # (token, hidden) -> (hidden_per_thread)
            dst = cute.zipped_divide(
                reduced_output[token_idx, None],
                (hidden_per_thread, ),
            )[(None, ), (hidden_tile_idx, )]

            if cutlass.const_expr(topk_score is not None):
                if cutlass.const_expr(prefetch):
                    cute.autovec_copy(topk_score[token_idx, None], score_reg)
            else:
                for k in cutlass.range_constexpr(num_topk):
                    score_reg[k] = score_dtype(1)
            if cutlass.const_expr(prefetch):
                cute.autovec_copy(sf[None, 0], scale_reg)

            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float4E2M1FN,
                num_bits_per_copy=64,
            )
            acc = cute.make_rmem_tensor((hidden_per_thread, ), cutlass.Float32)

            for k in cutlass.range_constexpr(0, num_topk, 1):
                term = cute.make_rmem_tensor((hidden_per_thread, ),
                                             cutlass.Float4E2M1FN)
                cute.copy(load_atom, codes[k, None], term)
                value = cute.make_rmem_tensor((hidden_per_thread, ),
                                              cutlass.Float32)
                # Dev-only knob (MEGA_F4CVT_USE_MANUAL): manual LUT/PRMT decode vs
                # the HW cvt path, so both SASS forms can be compared on device;
                # one is kept once chosen. Read inline on purpose -- never a
                # customer-facing option. Both decoders are bit-exact.
                if cutlass.const_expr(
                        os.environ.get("MEGA_F4CVT_USE_MANUAL", "0") == "1"):
                    cvt_e2m1_to_fp32_optimal_ptx(term, value)
                else:
                    cvt_e2m1_to_fp32_cvt_ptx(term, value)

                if cutlass.const_expr(not prefetch):
                    scale_reg[k] = sf[k, 0]
                    if cutlass.const_expr(topk_score is not None):
                        score_reg[k] = topk_score[token_idx, Int32(k)]

                # amax (bf16) -> per-element scale; (1/6) folds the fp4 grid max.
                scale = Float32(scale_reg[k]) * Float32(Nvfp4E2M1RcpLimit)
                scale_pair = (scale, scale)
                score_pair = (Float32(score_reg[k]), Float32(score_reg[k]))

                for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                    dequant_pair = cute.arch.mul_packed_f32x2(
                        (value[i], value[i + 1]), scale_pair)
                    if cutlass.const_expr(k != 0):
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(
                            dequant_pair, score_pair, (acc[i], acc[i + 1]))
                    elif cutlass.const_expr(topk_score is not None):
                        acc[i], acc[i + 1] = cute.arch.mul_packed_f32x2(
                            dequant_pair, score_pair)
                    else:
                        acc[i] = dequant_pair[0]
                        acc[i + 1] = dequant_pair[1]

            out = cute.make_rmem_tensor((hidden_per_thread, ), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                    out_dtype,
                                    num_bits_per_copy=256),
                out,
                self._mark_alignment(dst,
                                     hidden_per_thread * out_dtype.width // 8),
            )
