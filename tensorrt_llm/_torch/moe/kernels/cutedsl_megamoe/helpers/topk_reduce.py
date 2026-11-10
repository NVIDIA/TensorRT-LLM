# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device combine reduce: collapse per-(token, topk) fc2 cells into one row.

The combine step writes one fc2 output per ``(token, topk)`` cell; this reduces
over the topk axis into the token-centric ``(token, hidden)`` output. The wire
format is described by ``CombineFormat``:

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

import os
from typing import ClassVar, Optional, Tuple, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass.cutlass_dsl import Float32, Int32, Int64, T
from cutlass._mlir.dialects import llvm

from ..quant_def import CombineFormat
from .constants import Nvfp4E2M1RcpLimit
from .dsl_helpers import mark_alignment
from .ptx_helpers import nanosleep


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
def cvt_e2m1_to_fp32_cvt_ptx(e2m1_reg: cute.Tensor, fp32_reg: cute.Tensor) -> None:
    """Decode via the e2m1->f16 HW cvt (``cvt.rn.f16x2.e2m1x2``) then widen f16->f32.

    Safe baseline: the per-pair ``cvt`` instruction is itself a two-op hardware sequence, so
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
def cvt_e2m1_to_fp32_optimal_ptx(e2m1_reg: cute.Tensor, fp32_reg: cute.Tensor) -> None:
    """Decode via a register-resident bf16 LUT + byte permutes, landing fp32 directly.

    The 8 e2m1 magnitudes ``{0,.5,1,1.5,2,3,4,6}`` are exact bf16, so a byte permute
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


@cute.jit
def ld_relaxed_sys_v2_b64(address):
    """One strong 16-byte load (two 8-byte halves); never hoisted out of a polling loop."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i64(), T.i64()]),
        [address.ir_value()],
        "ld.relaxed.sys.global.v2.b64 {$0, $1}, [$2];",
        "=l,=l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,
    )
    return Int64(llvm.extractvalue(T.i64(), res, [0])), Int64(llvm.extractvalue(T.i64(), res, [1]))


@cute.jit
def _zero_half_mask(word):
    """Nonzero exactly when one of the word's four 16-bit halves is zero (exact for "any")."""
    return (word - Int64(0x0001000100010001)) & (word ^ Int64(-1)) & Int64(-0x7FFF7FFF7FFF8000)


def _mixed_add_bf16_asm(n: int):
    """PTX + constraint string for ``add_bf16_into_f32_ptx`` (plain Python: the
    DSL stages ``for`` loops inside ``@cute.jit`` bodies, so text is built here).

    Operand numbering: ``$0..$n-1`` f32 outputs, ``$n..$2n-1`` f32 accumulator
    inputs, ``$2n..$2n+n/2-1`` packed bf16x2 words.
    """
    nw = n // 2
    body = ["{", "  .reg .b16 lo, hi;"]
    for w in range(nw):
        body.append(f"  mov.b32 {{lo, hi}}, ${2 * n + w};")
        body.append(f"  add.rn.f32.bf16 ${2 * w}, lo, ${n + 2 * w};")
        body.append(f"  add.rn.f32.bf16 ${2 * w + 1}, hi, ${n + 2 * w + 1};")
    body.append("}")
    return "\n".join(body), ",".join(["=f"] * n + ["f"] * n + ["r"] * nw)


@cute.jit
def add_bf16_into_f32_ptx(term: cute.Tensor, acc: cute.Tensor, n: int) -> None:
    # ``acc[i] += term[i]`` for N bf16 terms via ``add.rn.f32.bf16`` (sm_100+).
    words = cute.recast_tensor(term, Int32)  # (N,) bf16 -> (N/2,) b32
    asm, constraints = _mixed_add_bf16_asm(n)
    operands = []
    for i in cutlass.range_constexpr(n):
        operands.append(acc[i].ir_value())
    for w in cutlass.range_constexpr(n // 2):
        operands.append(words[w].ir_value())
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * n),
        operands,
        asm,
        constraints,
        has_side_effects=False,
    )
    for i in cutlass.range_constexpr(n):
        acc[i] = Float32(llvm.extractvalue(T.f32(), res, [i]))


class TopkReduce:
    """Combine reduce for a fixed ``(hidden, num_topk, combine_format)``.

    ``__init__`` pins the static shape and format (and the derived launch
    geometry); ``__call__`` (a ``@cute.jit`` launcher) sizes a 1D grid from the
    runtime token count and dispatches the format's kernel. The caller owns the
    torch->cute conversion and the ``cute.compile`` / ``aot_compile``.

    ``persistent=True`` (opt in; default off and the static path below is
    untouched) swaps the one-worker-per-thread grid for a fixed
    ``ctas_per_sm * num_sms`` grid that pulls work from an L2-resident int32
    counter; ``ctas_per_sm`` defaults to the launch bound (``min_blocks_per_mp``),
    i.e. exactly one guaranteed-resident wave. The static grid is ``ceil(tokens * hidden_tiles / 128)``, so it
    tracks whatever token extent the caller hands in -- which for a MegaMoE
    workspace is the CAPACITY, making a 256-token decode pay the 4096-token
    launch. The persistent form reads the token count from device memory
    instead, so one launch geometry -- and one captured CUDA graph -- covers
    every token count.
    """

    _threads: ClassVar[int] = 128
    # Persistent scheduler work-unit granularity: one warp's worth of workers.
    # Warp- (not CTA-) granular so the work-id broadcast is a shuffle instead of
    # SMEM + two __syncthreads per fetch, and so one slow warp cannot stall its
    # CTA siblings.
    _Lanes: ClassVar[int] = 32
    # Per-thread vector width in bytes for the data-plane load: ONE load per
    # (thread, topk) term, at most 128 bits. 16 B is the widest -- there is no
    # two-load emulation of a 256-bit slice. Elements per worker follow from
    # this and the wire format, capped at the scale block so every worker still
    # reads exactly one scale entry:
    #   16 B: bf16 8 elems (one 128-bit load / one 128-bit store)
    #         e4m3 16 elems (one 128-bit load / two 128-bit stores: the bf16 result is 32 B)
    #         e2m1 16 elems (capped by its 16-wide scale block -> 64-bit load / two 128-bit stores)
    #    8 B: bf16 4, e4m3 8, e2m1 16 (64-bit load)
    #
    # Default (vector_bytes=None): 16 B on both paths, i.e. main's static
    # geometry -- the reasoning is in __init__ next to the choice.
    _vector_bytes_allowed: ClassVar[Tuple[int, ...]] = (8, 16)
    # bf16 plain-sum path on sm_100+: accumulate with add.rn.f32.bf16 (no
    # bf16->f32 cvt). Bit-identical to the convert-then-add form.
    _bf16_mixed_add: ClassVar[bool] = True
    # topk count at/below which the scale + score loads are hoisted ahead of the
    # topk loop (small enough to not bloat registers; a CTA-broadcast read).
    _prefetch_limit: ClassVar[int] = 16

    def __init__(
        self,
        hidden: int,
        num_topk: int,
        combine_format: CombineFormat,
        *,
        sm_arch: Optional[str] = None,
        num_experts: Optional[int] = None,
        ll_encoded: bool = False,
        vector_bytes: Optional[int] = None,
        min_blocks_per_mp: Optional[int] = None,
        persistent: bool = False,
        ctas_per_sm: Optional[int] = None,  # persistent grid per SM; None = min_blocks_per_mp
        num_sms: Optional[int] = None,
        max_fetch: int = 8,
        fetch_rounds: int = 4,
        internode: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """``internode = (experts_per_rank, ranks_per_node, remote_node_slots)`` enables the ordered reduce
        with one BF16 node partial per remote node (design invariant I11): a token's local top-k cells
        stay separate terms; the first top-k slot that names an expert of remote node ``n`` contributes
        that node's returned partial once and the node's other slots contribute nothing."""
        self.hidden = int(hidden)
        self.num_topk = int(num_topk)
        self.num_experts = None if num_experts is None else int(num_experts)
        self.combine_format = combine_format
        # ``sm_arch=None`` means "Blackwell or newer", the greenfield default.
        # SM90 has no packed f32x2 math, so the Hopper MegaMoE kernel passes its
        # arch explicitly and gets the scalar form of _fma/_fmul below.
        self.sm_arch = sm_arch
        self.use_scalar_math = False
        if sm_arch is not None:
            arch_code = sm_arch.removeprefix("sm_")
            if arch_code[-1:] in ("a", "f"):
                arch_code = arch_code[:-1]
            if not sm_arch.startswith("sm_") or not arch_code.isdigit():
                raise ValueError(f"sm_arch must have the form 'sm_XX', got {sm_arch!r}.")
            arch_number = int(arch_code)
            if arch_number < 90:
                raise ValueError(f"sm_arch must target SM90 or newer, got {sm_arch!r}.")
            self.use_scalar_math = arch_number < 100
            if self.use_scalar_math and combine_format.name != "bf16":
                raise ValueError(
                    f"sm_arch={sm_arch!r} only supports BF16 combine, got {combine_format.name!r}."
                )
        if vector_bytes is None:
            #   * static: 8 B only helps small token counts on B200, and Rubin
            #     is slower with it at large token counts (the DeepSeek V4 Pro
            #     CI guard regressed), so the static path keeps main's 16 B.
            #   * persistent wants the wider vector. Its overhead is per work
            #     unit (one atomic fetch + shuffle per 32 workers, once
            #     T * hidden_tiles / 32 exceeds the seed round) and halving the
            #     width doubles the unit count.
            #   * e2m1 is capped at 16 elems by its scale block (64-bit load) either way.
            vector_bytes = 16
        if vector_bytes not in self._vector_bytes_allowed:
            raise ValueError(f"vector_bytes must be one of {self._vector_bytes_allowed}, got {vector_bytes}.")
        self.vector_bytes = int(vector_bytes)
        self.hidden_per_thread = self.vector_bytes * 8 // combine_format.act_dtype.width
        if combine_format.scale_block:
            self.hidden_per_thread = min(self.hidden_per_thread, combine_format.scale_block)
        # Launch bound (__launch_bounds__ min blocks per SM). Left to itself
        # ptxas trades loads-in-flight for occupancy and sinks the last topk
        # loads below the first adds -- a second exposed DRAM round trip per
        # cell. Source-order hoisting does not change that; the register budget
        # does. Size the budget from what a cell actually needs to have every
        # load in flight at once: the fp32 accumulators, num_topk vector loads,
        # and ~16 for addressing/loop state.
        if min_blocks_per_mp is None:
            load_regs = self.hidden_per_thread * combine_format.act_dtype.width // 32
            regs_needed = self.hidden_per_thread + self.num_topk * load_regs + 16
            min_blocks_per_mp = max(4, min(8, 65536 // (128 * regs_needed)))
        self.min_blocks_per_mp = int(min_blocks_per_mp)
        self.internode = internode
        if internode is not None:
            if combine_format.is_quantized:
                raise ValueError("The internode node-partial reduce accepts the BF16 combine format only.")
            # _reduce_bf16_internode is a static-grid kernel with 128-bit bf16 loads and packed math.
            if persistent or self.use_scalar_math or self.vector_bytes != 16:
                raise ValueError(
                    "The internode node-partial reduce runs on the static path with the 16 B vector "
                    "on SM100+ only."
                )
        # bf16 cells stored as ``bits ^ 0xFFFF`` into a zeroed plane (see the FC2 UBLK
        # store): the reduce polls each cell instead of waiting for the producer grid,
        # and zeroes what it consumed for the next call. It skips top-k slots whose expert
        # index is outside [0, num_experts): the comm drops such routes (padding rows are
        # all -1), so no producer ever writes their cell.
        self.ll_encoded = bool(ll_encoded)
        if self.ll_encoded:
            if self.num_experts is None or num_topk > 31:
                raise ValueError("The polled bf16 reduce needs num_experts and at most 31 top-k slots.")
            if combine_format.is_quantized:
                raise ValueError("The self-validating combine plane is bf16 only.")
            if persistent or internode is not None or self.use_scalar_math or self.vector_bytes != 16:
                raise ValueError(
                    "The polled bf16 reduce is a static-grid kernel with 16 B cells on SM100+ (no persistent "
                    "scheduler, no internode partials)."
                )
        # hidden must tile cleanly both into worker slices and into scale blocks.
        align = max(combine_format.scale_block or self.hidden_per_thread, self.hidden_per_thread)
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

        # --- optional persistent scheduler (see the class docstring) ---
        self.persistent = bool(persistent)
        # Grid = ctas_per_sm x SMs. The default is the launch bound: exactly the
        # CTAs the register budget guarantees co-resident, so the fixed grid is
        # one full wave with no tail CTA queuing behind a finished one. An
        # explicit value is for experiments; the mega kernels do not set it.
        self.ctas_per_sm = int(ctas_per_sm) if ctas_per_sm is not None else self.min_blocks_per_mp
        self.max_fetch = int(max_fetch)
        self.fetch_rounds = int(fetch_rounds)
        if self.persistent:
            if self.ctas_per_sm < 1:
                raise ValueError(f"ctas_per_sm must be >= 1, got {self.ctas_per_sm}.")
            if self.max_fetch < 1:
                raise ValueError(f"max_fetch must be >= 1, got {self.max_fetch}.")
            if self.fetch_rounds < 1:
                raise ValueError(f"fetch_rounds must be >= 1, got {self.fetch_rounds}.")
            # Queried once here rather than per launch; callers that compile
            # without a device (aot_compile) must pass num_sms explicitly.
            # 0 / None both mean "ask the driver".
            self.num_sms = int(
                num_sms if num_sms else cutlass_utils.HardwareInfo().get_device_multiprocessor_count()
            )
            self.num_ctas = self.ctas_per_sm * self.num_sms
            self.warps_per_cta = self._threads // self._Lanes

    # -- arch-dispatched packed math -------------------------------------------

    _Float32Value = Union[Tuple[Float32, Float32], Float32]

    @cute.jit
    def _fma(self, lhs: _Float32Value, rhs: Float32, acc: _Float32Value) -> _Float32Value:
        """FP32 FMA with a scalar multiplier; packed on SM100+, scalar on SM90."""
        if cutlass.const_expr(self.use_scalar_math):
            if cutlass.const_expr(hasattr(cute.math, "fma")):
                return cute.math.fma(lhs, rhs, acc)
            return Float32(
                llvm.inline_asm(
                    T.f32(),
                    [lhs.ir_value(), rhs.ir_value(), acc.ir_value()],
                    "fma.rn.f32 $0, $1, $2, $3;",
                    "=f,f,f,f",
                    has_side_effects=False,
                )
            )
        return cute.arch.fma_packed_f32x2(lhs, (rhs, rhs), acc)

    @cute.jit
    def _fmul(self, lhs: _Float32Value, rhs: Float32) -> _Float32Value:
        """FP32 multiply with a scalar multiplier; packed on SM100+, scalar on SM90."""
        if cutlass.const_expr(self.use_scalar_math):
            if cutlass.const_expr(hasattr(cute.math, "mul")):
                return cute.math.mul(lhs, rhs)
            return Float32(
                llvm.inline_asm(
                    T.f32(),
                    [lhs.ir_value(), rhs.ir_value()],
                    "mul.rn.f32 $0, $1, $2;",
                    "=f,f,f",
                    has_side_effects=False,
                )
            )
        return cute.arch.mul_packed_f32x2(lhs, (rhs, rhs))

    # -- launcher -------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        combine_quant: cute.Tensor,  # (token, topk, hidden)
        combine_sf: Optional[cute.Tensor],  # (token, topk, hidden)
        reduced_output: cute.Tensor,  # (token, hidden)
        topk_score: Optional[cute.Tensor],  # (token, topk)
        stream: cuda.CUstream,
        counters: Optional[cute.Tensor] = None,  # persistent only: int32 x1 work cursor, caller-zeroed per launch
        num_tokens_dev: Optional[cute.Tensor] = None,  # persistent only: int32 x1 in gmem
        remote_partials: Optional[cute.Tensor] = None,  # (remote slot, message, hidden) BF16 node partials
        message_index: Optional[cute.Tensor] = None,  # (remote slot, token) -> message index
        topk_indices: Optional[cute.Tensor] = None,  # (token, topk) global expert ids
        local_rank: Optional[Int32] = None,
    ):
        """Launch the reduce.

        Persistent path only: the token count comes from device memory and
        nowhere else. Kernel parameters are frozen into a CUDA graph at
        capture, so a host scalar could never follow a per-replay token count;
        ``num_tokens_dev`` is one int32 in gmem that the caller writes and the
        kernel never modifies. The kernel clamps it to the output extent, so a
        stale or garbage value can never index past the workspace.
        """
        if cutlass.const_expr(self.internode is None and topk_indices is not None):
            if cutlass.const_expr(self.num_experts is None):
                raise ValueError("Masked combine requires a static expert count.")
        threads = self._threads
        if cutlass.const_expr(self.persistent):
            if cutlass.const_expr(counters is None or num_tokens_dev is None):
                raise ValueError(
                    "persistent=True requires both `counters` (one int32 in gmem, zeroed by "
                    "the caller before every launch) and `num_tokens_dev` (one int32 in gmem "
                    "holding the valid token count)."
                )
            grid = [self.num_ctas, 1, 1]
        else:
            if cutlass.const_expr(num_tokens_dev is not None):
                raise ValueError(
                    "num_tokens_dev requires persistent=True: the static reduce takes "
                    "its token count from the launch grid, which is host-side."
                )
            total_workers = reduced_output.shape[0] * self.hidden_tiles
            grid = [(total_workers + threads - 1) // threads, 1, 1]
        block = [threads, 1, 1]

        combine_quant = cute.make_tensor(
            combine_quant.iterator,
            cute.make_layout((combine_quant.shape[0], self.num_topk, self.hidden), stride=combine_quant.stride),
        )
        reduced_output = cute.make_tensor(
            reduced_output.iterator,
            cute.make_layout((reduced_output.shape[0], self.hidden), stride=reduced_output.stride),
        )
        if cutlass.const_expr(topk_score is not None):
            topk_score = cute.make_tensor(
                topk_score.iterator, cute.make_layout((topk_score.shape[0], self.num_topk), stride=topk_score.stride)
            )

        if cutlass.const_expr(self.ll_encoded):
            if cutlass.const_expr(topk_indices is None):
                raise ValueError("The polled bf16 reduce needs topk_indices to skip dropped slots.")
            self._reduce_bf16_ll(combine_quant, topk_score, reduced_output, topk_indices).launch(
                grid=grid, block=block, stream=stream, use_pdl=True
            )
            return
        if cutlass.const_expr(self.internode is not None):
            if cutlass.const_expr(remote_partials is None or message_index is None or topk_indices is None or local_rank is None):
                raise ValueError("The internode reduce needs remote_partials, message_index, topk_indices and local_rank.")
            self._reduce_bf16_internode(
                combine_quant, topk_score, reduced_output, remote_partials, message_index, topk_indices, local_rank
            ).launch(grid=grid, block=block, stream=stream)
            return

        # The mega kernel hands sf in already as the depth-2 broadcast layout; a
        # plain (torch) sf is depth-1 and gets its hidden mode split into
        # (sf_vec, hidden/sf_vec):(0, s_h) so logical hidden h reads block h//sf_vec.
        # Hoisted above the bf16 dispatch so the persistent path can share it;
        # `is_quantized` is a constexpr, so the bf16 form still emits nothing.
        sf = None
        if cutlass.const_expr(self.combine_format.is_quantized):
            sf_vec = self.combine_format.scale_block
            if cutlass.const_expr(cute.depth(combine_sf.layout) >= 2):
                sf = cute.make_tensor(
                    combine_sf.iterator,
                    cute.make_layout(
                        (combine_sf.shape[0], self.num_topk, (sf_vec, self.hidden // sf_vec)), stride=combine_sf.stride
                    ),
                )
            else:
                sf = cute.make_tensor(
                    combine_sf.iterator,
                    cute.make_layout(
                        (combine_sf.shape[0], self.num_topk, (sf_vec, self.hidden // sf_vec)),
                        stride=(combine_sf.stride[0], combine_sf.stride[1], (0, combine_sf.stride[2])),
                    ),
                )

        launch_kwargs = dict(grid=grid, block=block, stream=stream)
        if cutlass.const_expr(self.min_blocks_per_mp is not None):
            launch_kwargs["min_blocks_per_mp"] = self.min_blocks_per_mp
            # With a launch bound and no explicit carveout the DSL queries the
            # device's shared-memory size at COMPILE time (cuDeviceGet), which
            # an AOT compile worker without a CUDA context cannot do. The reduce
            # uses no shared memory, so state the carveout: 0 = prefer L1.
            launch_kwargs["preferred_smem_carveout"] = 0
        if cutlass.const_expr(self.persistent):
            self._reduce_persistent(
                combine_quant, sf, topk_score, reduced_output, counters, num_tokens_dev, topk_indices
            ).launch(**launch_kwargs)
        else:
            self._reduce_static(combine_quant, sf, topk_score, reduced_output, topk_indices).launch(
                **launch_kwargs
            )

    # -- kernels --------------------------------------------------------------

    @cute.kernel
    def _reduce_static(
        self,
        combine_quant: cute.Tensor,          # (token, topk, hidden)
        combine_sf: Optional[cute.Tensor],   # depth-2 broadcast view, or None
        topk_score: Optional[cute.Tensor],   # (token, topk)
        reduced_output: cute.Tensor,         # (token, hidden)
        topk_indices: Optional[cute.Tensor] = None,  # (token, topk)
    ):
        """One thread per (token, hidden_tile) worker; the grid carries the
        token count. This is the default path."""
        worker_idx = cute.arch.block_idx()[0] * Int32(self._threads) + cute.arch.thread_idx()[0]
        self._reduce_cell(
            worker_idx,
            combine_quant,
            combine_sf,
            topk_score,
            reduced_output,
            reduced_output.shape[0],
            self.require_predicate,
            topk_indices,
        )


    # -- persistent scheduler (opt in) ----------------------------------------

    @cute.kernel
    def _reduce_bf16_internode(
        self,
        combine_output: cute.Tensor,
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
        remote_partials: cute.Tensor,
        message_index: cute.Tensor,
        topk_indices: cute.Tensor,
        local_rank: Int32,
    ):
        threads = self._threads
        hidden_per_thread = self.hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        experts_per_rank, ranks_per_node, remote_slots = self.internode
        needs_guard = self.require_predicate
        out_dtype = reduced_output.element_type

        worker_idx = cute.arch.block_idx()[0] * Int32(threads) + cute.arch.thread_idx()[0]
        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles
        my_node = local_rank // Int32(ranks_per_node)

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            terms = cute.zipped_divide(combine_output[token_idx, None, None], (num_topk, hidden_per_thread))[
                (None, None), (0, hidden_tile_idx)
            ]
            dst = cute.zipped_divide(reduced_output[token_idx, None], (hidden_per_thread,))[(None,), (hidden_tile_idx,)]
            load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
            acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
            for i in cutlass.range_constexpr(hidden_per_thread):
                acc[i] = Float32(0.0)
            seen_nodes = Int32(0)
            for k in cutlass.range_constexpr(0, num_topk, 1):
                expert = Int32(topk_indices[token_idx, Int32(k)])
                node = (expert // Int32(experts_per_rank)) // Int32(ranks_per_node)
                term = cute.make_rmem_tensor((hidden_per_thread,), cutlass.BFloat16)
                score = Float32(1.0)
                contributes = cutlass.Boolean(True)
                if node == my_node:
                    cute.copy(
                        load_atom, mark_alignment(terms[k, None], hidden_per_thread * cutlass.BFloat16.width // 8), term
                    )
                    if cutlass.const_expr(topk_score is not None):
                        score = Float32(topk_score[token_idx, Int32(k)])
                else:
                    slot = node
                    if node > my_node:
                        slot = node - Int32(1)
                    node_bit = Int32(1) << slot
                    if (seen_nodes & node_bit) != Int32(0):
                        contributes = cutlass.Boolean(False)
                    seen_nodes = seen_nodes | node_bit
                    message = message_index[slot, token_idx]
                    partial_row = cute.zipped_divide(remote_partials[slot, message, None], (hidden_per_thread,))[
                        (None,), (hidden_tile_idx,)
                    ]
                    cute.copy(load_atom, mark_alignment(partial_row, hidden_per_thread * cutlass.BFloat16.width // 8), term)
                if contributes:
                    score_pair = (score, score)
                    for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                        value_pair = (Float32(term[i]), Float32(term[i + 1]))
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(value_pair, score_pair, (acc[i], acc[i + 1]))

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=128),
                out,
                mark_alignment(dst, hidden_per_thread * out_dtype.width // 8),
            )

    @cute.kernel
    def _reduce_bf16_ll(
        self,
        combine_output: cute.Tensor,
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
        topk_indices: cute.Tensor,
    ):
        """The bf16 ``_reduce_cell`` over the self-validating plane: poll, reduce, zero.

        A slot whose expert index is outside ``[0, num_experts)`` (a padding row, a dropped
        route) is never produced, so its cell stays zero forever: it is not waited for and
        contributes an exact zero term, as it did when the reduce ran after the producer grid.

        Started as a programmatic dependent but without ``griddepcontrol.wait``: a
        cell is final once none of its 16-bit halves reads zero, so the reduce runs
        alongside the producer's tail instead of after it, and may also end before
        that tail; the next call's communication kernel waits for the tail itself. All loads of a worker
        issue before any test; only a cell that is still in flight is loaded again.
        The arithmetic is the bf16 ``_reduce_cell``'s (convert-then-FMA form), term for term.
        """
        threads = self._threads
        hidden_per_thread = self.hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        needs_guard = self.require_predicate
        prefetch = self.prefetch
        out_dtype = reduced_output.element_type
        if cutlass.const_expr(hidden_per_thread * cutlass.BFloat16.width != 128):
            raise ValueError("The polled bf16 reduce reads one 16-byte cell per top-k slot.")

        worker_idx = cute.arch.block_idx()[0] * Int32(threads) + cute.arch.thread_idx()[0]
        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        score_dtype = topk_score.dtype if cutlass.const_expr(topk_score is not None) else cutlass.Float32
        score_reg = cute.make_rmem_tensor((num_topk,), score_dtype)

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            terms = cute.zipped_divide(combine_output[token_idx, None, None], (num_topk, hidden_per_thread))[
                (None, None), (0, hidden_tile_idx)
            ]
            dst = cute.zipped_divide(reduced_output[token_idx, None], (hidden_per_thread,))[(None,), (hidden_tile_idx,)]

            if cutlass.const_expr(topk_score is not None):
                if cutlass.const_expr(prefetch):
                    cute.autovec_copy(topk_score[token_idx, None], score_reg)
            else:
                for k in cutlass.range_constexpr(num_topk):
                    score_reg[k] = score_dtype(1)

            # Every cell load is issued first and unconditionally, as before the slot test
            # existed: the loads of one worker then overlap, and a dropped slot's cell (never
            # written, all zero) is loaded harmlessly. Gating the loads on the indices
            # serialized them behind the index loads and cost 2-3% on the last reduce waves.
            words = cute.make_rmem_tensor((num_topk * 2,), Int64)
            for k in cutlass.range_constexpr(num_topk):
                low, high = ld_relaxed_sys_v2_b64(terms[k, None].iterator.toint())
                words[2 * k] = low
                words[2 * k + 1] = high

            # Bit k set: slot k routes to a real expert. Same unsigned test as the comm's
            # ``_is_unowned_expert`` (so -1 is invalid too). A dropped slot's cell stays zero
            # forever, so its words are replaced by the encoded zero (-1: both halves nonzero,
            # decodes to 0.0) and it is neither waited for nor cleared.
            if cutlass.const_expr(hidden_tiles % 32 == 0):
                # The 32 lanes of a warp hold one token: lane k loads slot k and one ballot is the
                # mask, instead of num_topk index loads per thread (which cost up to 16% of this
                # load-bound kernel at top-k 10, tpr 1024).
                lane_idx = cute.arch.lane_idx()
                lane_valid = cutlass.Boolean(False)
                if lane_idx < Int32(num_topk):
                    index = topk_indices[token_idx, lane_idx]
                    if cutlass.const_expr(topk_indices.element_type is cutlass.Int32):
                        lane_valid = cutlass.Uint32(index) < cutlass.Uint32(self.num_experts)
                    else:
                        lane_valid = cutlass.Uint64(index) < cutlass.Uint64(self.num_experts)
                valid_slots = Int32(cute.arch.vote_ballot_sync(lane_valid))
            else:
                valid_slots = Int32(0)
                for k in cutlass.range_constexpr(num_topk):
                    index = topk_indices[token_idx, Int32(k)]
                    if cutlass.const_expr(topk_indices.element_type is cutlass.Int32):
                        slot_valid = cutlass.Uint32(index) < cutlass.Uint32(self.num_experts)
                    else:
                        slot_valid = cutlass.Uint64(index) < cutlass.Uint64(self.num_experts)
                    if slot_valid:
                        valid_slots = valid_slots | Int32(1 << k)
            for k in cutlass.range_constexpr(num_topk):
                if (valid_slots >> Int32(k)) & Int32(1) == Int32(0):
                    words[2 * k] = Int64(-1)
                    words[2 * k + 1] = Int64(-1)
            stale = Int64(0)
            for k in cutlass.range_constexpr(num_topk):
                stale = stale | _zero_half_mask(words[2 * k]) | _zero_half_mask(words[2 * k + 1])
            while stale != Int64(0):
                # Launched while the producer's last tiles still stream: one short sleep
                # per round, then every stale slot is reloaded at once (never one sleep
                # per slot, which would serialize the latency over the top-k).
                nanosleep(128)
                for k in cutlass.range_constexpr(num_topk):
                    if (_zero_half_mask(words[2 * k]) | _zero_half_mask(words[2 * k + 1])) != Int64(0):
                        low, high = ld_relaxed_sys_v2_b64(terms[k, None].iterator.toint())
                        words[2 * k] = low
                        words[2 * k + 1] = high
                stale = Int64(0)
                for k in cutlass.range_constexpr(num_topk):
                    stale = stale | _zero_half_mask(words[2 * k]) | _zero_half_mask(words[2 * k + 1])
            for i in cutlass.range_constexpr(num_topk * 2):
                words[i] = words[i] ^ Int64(-1)
            values = cute.recast_tensor(words, cutlass.BFloat16)

            acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
            for k in cutlass.range_constexpr(0, num_topk, 1):
                if cutlass.const_expr(topk_score is not None and not prefetch):
                    score_reg[k] = topk_score[token_idx, Int32(k)]
                score_pair = (Float32(score_reg[k]), Float32(score_reg[k]))

                for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                    value_pair = (
                        Float32(values[k * hidden_per_thread + i]),
                        Float32(values[k * hidden_per_thread + i + 1]),
                    )
                    if cutlass.const_expr(k != 0):
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(value_pair, score_pair, (acc[i], acc[i + 1]))
                    else:
                        # Include the +0 seed: direct assignment/multiply preserves -0.
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(
                            value_pair, score_pair, (Float32(0.0), Float32(0.0))
                        )

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            vector_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=128)
            cute.copy(vector_store, out, mark_alignment(dst, hidden_per_thread * out_dtype.width // 8))

            # Consumer reset: the next call's producers find zeroed cells. They write
            # this rank's cells only for tokens this rank pushes after this launch.
            cleared = cute.make_rmem_tensor((hidden_per_thread,), cutlass.BFloat16)
            cleared.fill(0.0)
            cell_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
            for k in cutlass.range_constexpr(num_topk):
                if (valid_slots >> Int32(k)) & Int32(1) != Int32(0):
                    cute.copy(cell_store, cleared, mark_alignment(terms[k, None], hidden_per_thread * 2))
    @cute.kernel
    def _reduce_persistent(
        self,
        combine_quant: cute.Tensor,  # (token, topk, hidden)
        combine_sf: Optional[cute.Tensor],  # depth-2 broadcast view, or None
        topk_score: Optional[cute.Tensor],  # (token, topk)
        reduced_output: cute.Tensor,  # (token, hidden)
        counters: cute.Tensor,  # int32 x1: work cursor, zeroed by the caller before every launch
        num_tokens_dev: cute.Tensor,  # int32 x1: valid token count
        topk_indices: Optional[cute.Tensor] = None,  # (token, topk)
    ):
        """Fixed-grid form of the three ``_reduce_*`` kernels above.
        """
        lanes = self._Lanes
        hidden_tiles = self.hidden_tiles

        tid = cute.arch.thread_idx()[0]
        lane = tid % Int32(lanes)
        warp = tid // Int32(lanes)
        bid = cute.arch.block_idx()[0]

        num_tokens = max(Int32(0), min(Int32(reduced_output.shape[0]), num_tokens_dev[0]))

        num_units = (num_tokens * Int32(hidden_tiles) + Int32(lanes - 1)) // Int32(lanes)

        counter_ptr = counters.iterator

        # Seed one unit per warp, STRIDED. Blocked seeding (warp w takes units
        # [w*fetch, (w+1)*fetch)) idles most warps when num_units is small --
        # exactly the decode case this path exists for.
        total_warps = Int32(self.num_ctas * self.warps_per_cta)
        unit = bid * Int32(self.warps_per_cta) + warp
        preclaimed = total_warps

        rounds = Int32(self.fetch_rounds)
        leftover = num_units - preclaimed
        fetch = (leftover + total_warps * rounds - Int32(1)) // (total_warps * rounds)
        fetch = max(Int32(1), min(Int32(self.max_fetch), fetch))

        res_lo = Int32(0)
        res_hi = Int32(0)

        has_leftover = leftover > Int32(0)  # kernel-uniform

        while unit < num_units:
            # Issue the refill BEFORE this unit's work and consume it after, so
            # the atomic's L2 round trip overlaps the loads/FMAs.
            need_fetch = has_leftover and res_lo >= res_hi  # warp-uniform
            base = Int32(0)
            if need_fetch and lane == Int32(0):
                base = cute.arch.atomic_add(counter_ptr, fetch)

            self._reduce_cell(
                unit * Int32(lanes) + lane,
                combine_quant,
                combine_sf,
                topk_score,
                reduced_output,
                num_tokens,
                True,
                topk_indices,
            )

            if need_fetch:
                base = cute.arch.shuffle_sync(base, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31)
                res_lo = preclaimed + base
                res_hi = res_lo + fetch
            if has_leftover:
                unit = res_lo
                res_lo = res_lo + Int32(1)
            else:
                unit = num_units  # seed unit was this warp's only unit


    @cute.jit
    def _reduce_cell(
        self,
        worker_idx: Int32,
        combine_quant: cute.Tensor,
        combine_sf: Optional[cute.Tensor],
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
        num_tokens: Int32,
        needs_guard: cutlass.Constexpr[bool] = True,
        topk_indices: Optional[cute.Tensor] = None,
    ):
        """One ``(token, hidden_tile)`` cell, format-dispatched at compile time.

        The single copy of the reduce arithmetic: both ``_reduce_static`` and
        ``_reduce_persistent`` drive it, and the bf16 / mxfp8 / nvfp4 wire
        formats are ``const_expr`` branches rather than separate kernels.

        ``needs_guard=False`` elides the bounds check entirely, which is what
        the static path wants when ``hidden_tiles`` is a whole number of CTAs.
        """
        hidden_per_thread = self.hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        prefetch = self.prefetch
        out_dtype = reduced_output.element_type
        fmt = self.combine_format

        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        score_dtype = topk_score.dtype if cutlass.const_expr(topk_score is not None) else cutlass.Float32
        score_reg = cute.make_rmem_tensor((num_topk,), score_dtype)

        # Static path: the tail CTA may run past the token count. Persistent
        # path: the last work unit is ragged because num_tokens * hidden_tiles
        # need not be a multiple of _Lanes.
        if (not needs_guard) or token_idx < num_tokens:
            codes = cute.zipped_divide(combine_quant[token_idx, None, None], (num_topk, hidden_per_thread))[
                (None, None), (0, hidden_tile_idx)
            ]
            dst = cute.zipped_divide(reduced_output[token_idx, None], (hidden_per_thread,))[(None,), (hidden_tile_idx,)]

            if cutlass.const_expr(topk_score is not None):
                if cutlass.const_expr(prefetch):
                    cute.autovec_copy(topk_score[token_idx, None], score_reg)
            else:
                for k in cutlass.range_constexpr(num_topk):
                    score_reg[k] = score_dtype(1)

            acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)

            # One load per (thread, topk) term, at most 128 bits, on every arch.
            # Stores may tile to two 128-bit stores when the bf16 result is wider than
            # the quantized input (e4m3 / e2m1 at 16 elems); that is the layout
            # the kernel always had.
            load_bits = min(128, hidden_per_thread * fmt.act_dtype.width)
            store_bits = min(128, hidden_per_thread * out_dtype.width)

            if cutlass.const_expr(not fmt.is_quantized):
                load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=load_bits)
                # Plain-sum path on sm_100+: the bf16 terms go straight into the
                # f32 adder (add.rn.f32.bf16), no cvt. Seed with +0.0 so k == 0
                # takes the same add and a -0.0 first term yields +0.0, exactly
                # like fma(v, score, +0.0) on the other paths (main's +0 seed).
                mixed_add = topk_score is None and not self.use_scalar_math and self._bf16_mixed_add
                if cutlass.const_expr(mixed_add):
                    for i in cutlass.range_constexpr(hidden_per_thread):
                        acc[i] = Float32(0.0)
                # Note on load scheduling: ptxas sinks the last one or two of the
                # num_topk loads below the first adds (a second exposed DRAM
                # round trip). Hoisting them in source does not change that -- the
                # emitted load positions were identical -- it is a register
                # budget decision, steerable only via min_blocks_per_mp.
                for k in cutlass.range_constexpr(0, num_topk, 1):
                    term = cute.make_rmem_tensor((hidden_per_thread,), cutlass.BFloat16)
                    if cutlass.const_expr(topk_indices is not None):
                        expert = topk_indices[token_idx, Int32(k)]
                        valid_route = (expert >= 0) & (expert < self.num_experts)
                        term.fill(0)
                        if valid_route:
                            cute.copy(
                                load_atom,
                                mark_alignment(codes[k, None], hidden_per_thread * cutlass.BFloat16.width // 8),
                                term,
                            )
                    else:
                        cute.copy(
                            load_atom,
                            mark_alignment(codes[k, None], hidden_per_thread * cutlass.BFloat16.width // 8),
                            term,
                        )
                    if cutlass.const_expr(topk_score is not None and not prefetch):
                        score_reg[k] = topk_score[token_idx, Int32(k)]
                    if cutlass.const_expr(topk_indices is not None):
                        # Zero both operands: a stale NaN score times zero is NaN.
                        if not valid_route:
                            score_reg[k] = score_dtype(0)
                    score = Float32(score_reg[k])

                    if cutlass.const_expr(self.use_scalar_math):
                        for i in cutlass.range_constexpr(hidden_per_thread):
                            value = Float32(term[i])
                            if cutlass.const_expr(k != 0):
                                acc[i] = self._fma(value, score, acc[i])
                            else:
                                # +0 seed: a -0.0 first term yields +0.0 (main's semantics).
                                acc[i] = self._fma(value, score, Float32(0.0))
                    elif cutlass.const_expr(mixed_add):
                        add_bf16_into_f32_ptx(term, acc, hidden_per_thread)
                    else:
                        for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                            value_pair = (Float32(term[i]), Float32(term[i + 1]))
                            if cutlass.const_expr(k != 0):
                                acc[i], acc[i + 1] = self._fma(value_pair, score, (acc[i], acc[i + 1]))
                            else:
                                # +0 seed: a -0.0 first term yields +0.0 (main's semantics).
                                acc[i], acc[i + 1] = self._fma(value_pair, score, (Float32(0.0), Float32(0.0)))
            else:
                sf = cute.zipped_divide(combine_sf[token_idx, None, None], (num_topk, hidden_per_thread))[
                    (None, None), (0, hidden_tile_idx)
                ]
                is_fp8 = cutlass.const_expr(fmt.act_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2))
                scale_dtype = cutlass.Float8E8M0FNU if cutlass.const_expr(is_fp8) else cutlass.BFloat16
                scale_reg = cute.make_rmem_tensor((num_topk,), scale_dtype)
                if cutlass.const_expr(prefetch):
                    cute.autovec_copy(sf[None, 0], scale_reg)

                act_dtype = fmt.act_dtype
                load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), act_dtype, num_bits_per_copy=load_bits)
                for k in cutlass.range_constexpr(0, num_topk, 1):
                    term = cute.make_rmem_tensor((hidden_per_thread,), act_dtype)
                    if cutlass.const_expr(topk_indices is not None):
                        expert = topk_indices[token_idx, Int32(k)]
                        valid_route = (expert >= 0) & (expert < self.num_experts)
                        # Packed FP4 has no scalar DSL conversion; clear its bits.
                        cute.recast_tensor(term, Int32).fill(0)
                        if valid_route:
                            cute.copy(
                                load_atom, mark_alignment(codes[k, None], hidden_per_thread * act_dtype.width // 8), term
                            )
                    else:
                        cute.copy(
                            load_atom, mark_alignment(codes[k, None], hidden_per_thread * act_dtype.width // 8), term
                        )
                    value = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
                    if cutlass.const_expr(is_fp8):
                        value.store(term.load().to(cutlass.Float32))
                    elif cutlass.const_expr(os.environ.get("MEGA_F4CVT_USE_MANUAL", "0") == "1"):
                        cvt_e2m1_to_fp32_optimal_ptx(term, value)
                    else:
                        cvt_e2m1_to_fp32_cvt_ptx(term, value)

                    if cutlass.const_expr(not prefetch):
                        scale_reg[k] = sf[k, 0]
                        if cutlass.const_expr(topk_score is not None):
                            score_reg[k] = topk_score[token_idx, Int32(k)]

                    if cutlass.const_expr(is_fp8):
                        scale = Float32(scale_reg[k])  # e8m0 -> f32
                    else:
                        # amax (bf16) -> per-element scale; (1/6) folds the fp4 grid max.
                        scale = Float32(scale_reg[k]) * Float32(Nvfp4E2M1RcpLimit)
                    score = Float32(score_reg[k])
                    if cutlass.const_expr(topk_indices is not None):
                        if not valid_route:
                            scale = Float32(0)
                            score = Float32(0)

                    for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                        dequant_pair = self._fmul((value[i], value[i + 1]), scale)
                        if cutlass.const_expr(k != 0):
                            acc[i], acc[i + 1] = self._fma(dequant_pair, score, (acc[i], acc[i + 1]))
                        else:
                            # +0 seed: a -0.0 first term yields +0.0 (main's semantics).
                            acc[i], acc[i + 1] = self._fma(dequant_pair, score, (Float32(0.0), Float32(0.0)))

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=store_bits),
                out,
                mark_alignment(dst, hidden_per_thread * out_dtype.width // 8),
            )
