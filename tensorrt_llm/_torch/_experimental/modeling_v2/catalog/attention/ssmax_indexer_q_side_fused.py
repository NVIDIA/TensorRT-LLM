# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral Large 4 DSA indexer's q side after its projection, SSMax included, in one Triton
launch: ``ssmax_scaling`` followed by ``cute_dsl_indexer_q_side_fused`` (interleaved rope), bit
for bit, reading the head weights straight out of the bf16 ``wk | weights_proj`` output."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.language.extra.cuda import gdc_wait

from .._op import Arch, Cell, OpWrapper
from .cute_dsl_indexer_q_side_fused import cute_dsl_indexer_q_side_fused
from .ssmax_scaling import ssmax_scaling

HEADS = 32
HEAD_DIM = 128
ROPE_DIM = 64

#: max_seq_len of the NVFP4 targets' 32K config: every position, and the rope table's rows.
MAX_SEQ_LEN = 32768
#: Their 1M config (the checkpoint's own context, chunked prefill in steps of at most
#: MAX_NUM_TOKENS rows): a 1048576-row table, a chunk's rows behind its request's cached
#: prefix, decode rows at any position up to 1048575.
LONG_MAX_SEQ_LEN = 1048576
MAX_NUM_TOKENS = 32768
#: Decode batches the targets capture (see `ssmax_scaling_fused_q.GRAPH_BATCHES`).
GRAPH_BATCHES = (1, 2, 3, 4, 8, 13, 16, 32, 64, 128, 256)

#: What every cell shares: the NVFP4 checkpoint's indexer s and b (44 layers x 32 heads, read
#: on disk as bf16: s in [-0.0437012, 0.0664062], b in [0.976562, 1.04688]) inside the draw
#: ranges with the four corners pinned; its interleaved YaRN (factor 2048) + LeRoPE table of
#: 32768 rows with cos and sin rounded through bf16, as the targets hold it; the head weights
#: as the targets pass them, the last 32 columns of the bf16 `[T, 160]` wk | weights_proj
#: output.
_ML4 = dict(
    s_range=(-0.05, 0.075),
    b_range=(0.97, 1.06),
    q="normal",
    table="yarn",
    wq="kw_bf16",
    max_pos=MAX_SEQ_LEN,
)
#: The same at the 1M config: the YaRN (factor 2048) + LeRoPE table of 1048576 rows.
_ML4_1M = dict(_ML4, max_pos=LONG_MAX_SEQ_LEN)


@triton.jit
def _round_bf16(x):
    """fp32 rounded to bf16 (to nearest, ties to even) and widened back. The rounding is an
    inline `cvt.rn.bf16.f32`: the compiler folds a bare `.to(bf16).to(fp32)` pair into
    nothing, which skips a rounding the chain specifies."""
    r = tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;", "=h,r", [x], dtype=tl.bfloat16, is_pure=True, pack=1
    )
    return r.to(tl.float32)


@triton.jit
def _round_bf16_ftz(x):
    """`_round_bf16`, then a bf16 subnormal (or zero) flushed to the zero of its sign: the
    q side's fast-math widening of its rope-half input."""
    bits = _round_bf16(x).to(tl.int32, bitcast=True)
    # bf16 shares fp32's exponent field: a bf16 subnormal widens to a zero exponent.
    flushed = tl.where((bits & 0x7F800000) == 0, bits & -2147483648, bits)
    return flushed.to(tl.float32, bitcast=True)


@triton.jit
def _max_nan(a, b):
    """max that returns NaN when either side is NaN, as the q side's amax and its floor do."""
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _ssmax_indexer_q_side_kernel(
    q_ptr,  # bf16 [T, 32, 128], unit stride over the head dim
    q_st,
    q_sh,
    pos_ptr,  # int32 [T]
    s_ptr,  # [32]
    s_st,
    b_ptr,  # [32]
    b_st,
    cs_ptr,  # fp32 [positions, 64]: cos(32) | sin(32), contiguous
    w_ptr,  # bf16 or fp32 [T, 32], unit stride over the heads
    w_st,
    weight_scale,  # fp32
    qf_ptr,  # e4m3 [T, 32, 128], contiguous
    wo_ptr,  # fp32 [T, 32], contiguous
    USE_PDL: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, 32)
    i = tl.arange(0, 32)
    j = tl.arange(0, 64)
    if USE_PDL:
        gdc_wait()
    # SSMax's scale, `ssmax_scaling`'s chain: the position to fp32, + 1, logf rounded to
    # bf16; * s and + b, each rounded to bf16.
    pos = tl.load(pos_ptr + t)
    lp = _round_bf16(libdevice.log(pos.to(tl.float32) + 1.0))
    s = _round_bf16(tl.load(s_ptr + h * s_st).to(tl.float32))
    b = _round_bf16(tl.load(b_ptr + h * b_st).to(tl.float32))
    scale = _round_bf16(_round_bf16(lp * s) + b)[:, None]
    row = q_ptr + t * q_st + h[:, None].to(tl.int64) * q_sh
    # The scaled query, rounded to bf16 (SSMax's output); the rope half then enters the
    # q side through its subnormal flush. Interleaved pairs: dims 2i and 2i + 1.
    xe = _round_bf16_ftz(tl.load(row + 2 * i[None, :]).to(tl.float32) * scale)
    xo = _round_bf16_ftz(tl.load(row + 2 * i[None, :] + 1).to(tl.float32) * scale)
    xn = _round_bf16(tl.load(row + 64 + j[None, :]).to(tl.float32) * scale)
    # The rope: fma(x, cos, (-/+ partner) * sin), rounded to bf16.
    tab = cs_ptr + pos.to(tl.int64) * 64
    c = tl.load(tab + i)[None, :]
    sn = tl.load(tab + 32 + i)[None, :]
    ye = _round_bf16(tl.fma(xe, c, -(xo * sn)))
    yo = _round_bf16(tl.fma(xo, c, xe * sn))
    # One power-of-two scale per (token, head) over [rope | nope]: max(amax, 1e-12) / 448
    # rounded up from its exponent bits; its inverse is exact. A NaN in the row (only an
    # SSMax overflow makes one) propagates through the amax and the floor, as there.
    amax = _max_nan(tl.reduce(tl.abs(ye), 1, _max_nan), tl.reduce(tl.abs(yo), 1, _max_nan))
    amax = _max_nan(_max_nan(amax, tl.reduce(tl.abs(xn), 1, _max_nan)), 1.0e-12)
    # fp32(1 / 448), the q side's constant, and one fp32 product.
    raw = (amax * tl.full([1], 1.0 / 448.0, tl.float32)).to(tl.int32, bitcast=True)
    exp_bits = raw & 0x7F800000
    exp_bits = tl.where((raw & 0x007FFFFF) != 0, exp_bits + 0x00800000, exp_bits)
    q_scale = exp_bits.to(tl.float32, bitcast=True)
    inv = (0x7F000000 - exp_bits).to(tl.float32, bitcast=True)[:, None]
    out = qf_ptr + t * (32 * 128) + h[:, None] * 128
    tl.store(out + 2 * i[None, :], (ye * inv).to(tl.float8e4nv))
    tl.store(out + 2 * i[None, :] + 1, (yo * inv).to(tl.float8e4nv))
    tl.store(out + 64 + j[None, :], (xn * inv).to(tl.float8e4nv))
    # The head weight: two fp32 products, left-associated.
    w = tl.load(w_ptr + t * w_st + h).to(tl.float32)
    tl.store(wo_ptr + t * 32 + h, (w * q_scale) * weight_scale)


class SSMaxIndexerQSideFused(OpWrapper):
    """SSMax-scale the indexer query, rope it, pack it to e4m3, fold its scale into the head
    weights.

    Computes ``cute_dsl_indexer_q_side_fused(ssmax_scaling(q, positions, s, b), positions,
    cos_sin, wq.float(), weight_scale, q_fp8_out, weights_out, is_neox=False)``: `q` is the
    unscaled, unrotated ``[T, 32, 128]`` bf16 indexer query, each head ``[pe(64) |
    nope(64)]``; `s` and `b` the layer's ``[32]`` SSMax parameters; `cos_sin` the fp32
    ``[positions, 64]`` ``[cos(32) | sin(32)]`` table, rotating interleaved pairs (dims
    ``2i``, ``2i + 1`` by column ``i``; an identity table on a NoPE indexer layer); `wq` the
    ``[T, 32]`` head weights, bf16 or fp32, any row stride (the targets pass the last 32
    columns of the bf16 ``[T, 160]`` wk | weights_proj output). Writes `q_fp8_out` (e4m3
    ``[T, 32, 128]``) and `weights_out` (fp32 ``[T, 32]``), both contiguous and
    caller-owned, and returns them. `q` is not modified.

    Fusion boundary: everything between the `wq_b` GEMM (and the wk | weights_proj GEMM) and
    the logits scorer. The projections, and the scorer's reads of the outputs, are the
    caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: `tokens` rows; `positions` "spread" (uniform over the table), "tail" (its last 64
    # rows) or "arange" (0..T - 1, or `cached`..`cached` + T - 1 behind a cached prefix);
    # `position_sweep` (lo, hi) drives every position lo..hi - 1 in calls of `tokens` rows,
    # each on fresh inputs; `table` "yarn" (the NVFP4 checkpoint's indexer table, a
    # seeded LeRoPE alpha, cos and sin rounded through bf16) or "identity"; `wq` "kw_bf16"
    # (the [T, 160] output's last 32 columns) or "fp32" (contiguous); s and b drawn over
    # [s_range] / [b_range] with the four corners on heads 0..3; `q` "normal" is 2 N(0, 1),
    # "wide" every bf16 binade from the subnormals up; `graph` captures the call and replays
    # it on new inputs.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"decode, graph batch {n:>3}, rope layer: captured, replayed on new inputs",
            spec=dict(_ML4, tokens=n, positions="spread", graph=True),
        )
        for n in GRAPH_BATCHES
    ) + (
        Cell(
            why="decode, graph batch 16 at positions in the table's last 64 rows",
            spec=dict(_ML4, tokens=16, positions="tail", graph=True),
        ),
        Cell(
            why="decode, graph batch 16, NoPE indexer layer: identity table, captured",
            spec=dict(_ML4, tokens=16, positions="spread", graph=True, table="identity"),
        ),
        Cell(
            why="decode, graph batch 256, NoPE indexer layer: identity table, captured",
            spec=dict(_ML4, tokens=256, positions="spread", graph=True, table="identity"),
        ),
        Cell(
            why="prefill of 32768 rows at positions 0..32767, rope layer: every table row",
            spec=dict(_ML4, tokens=32768, positions="arange", graph=False),
        ),
        Cell(
            why="prefill of 32768 rows, NoPE indexer layer: identity table over 0..32767",
            spec=dict(_ML4, tokens=32768, positions="arange", graph=False, table="identity"),
        ),
        Cell(
            why="an odd step of 300 rows, contiguous fp32 head weights",
            spec=dict(_ML4, tokens=300, positions="spread", graph=False, wq="fp32"),
        ),
        Cell(
            why="stress at 8192 rows, s in [-1, 1] and b in [-2, 2], q over every bf16 "
            "binade: subnormal flushes, zero rows, overflow to the e4m3 range",
            spec=dict(
                _ML4,
                tokens=8192,
                positions="spread",
                graph=False,
                q="wide",
                s_range=(-1.0, 1.0),
                b_range=(-2.0, 2.0),
            ),
        ),
    )

    # The 1M config (max_seq_len 1048576, chunked prefill, max_num_tokens 32768, 16 sequences):
    # the 1048576-row table, decode rows at any position, a chunk's rows behind a cached prefix,
    # and every table row swept in calls of 32768 rows.
    CELLS += tuple(
        Cell(
            why=f"1M decode, graph batch {n:>2}, rope layer: positions spread over the "
            "1048576-row table, captured, replayed on new inputs",
            spec=dict(_ML4_1M, tokens=n, positions="spread", graph=True),
        )
        for n in (1, 2, 3, 4, 8, 13, 16)
    )
    CELLS += (
        Cell(
            why="1M decode, graph batch 16 at positions in the 1048576-row table's last 64 rows",
            spec=dict(_ML4_1M, tokens=16, positions="tail", graph=True),
        ),
        Cell(
            why="1M decode, graph batch 16, NoPE indexer layer: identity table of 1048576 rows",
            spec=dict(_ML4_1M, tokens=16, positions="spread", graph=True, table="identity"),
        ),
        Cell(
            why="1M last chunk: 32768 rows at positions 1015808..1048575 behind a "
            "1015808-token cached prefix, rope layer",
            spec=dict(
                _ML4_1M,
                tokens=MAX_NUM_TOKENS,
                positions="arange",
                cached=LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,
                graph=False,
            ),
        ),
        Cell(
            why="1M last chunk, NoPE indexer layer: identity table, positions 1015808..1048575",
            spec=dict(
                _ML4_1M,
                tokens=MAX_NUM_TOKENS,
                positions="arange",
                cached=LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,
                graph=False,
                table="identity",
            ),
        ),
        Cell(
            why="1M mixed step of 32768 rows at positions spread over 0..1048575",
            spec=dict(_ML4_1M, tokens=MAX_NUM_TOKENS, positions="spread", graph=False),
        ),
        Cell(
            why="1M stress at 8192 rows spread over the table, s in [-1, 1] and b in [-2, 2], "
            "q over every bf16 binade",
            spec=dict(
                _ML4_1M,
                tokens=8192,
                positions="spread",
                graph=False,
                q="wide",
                s_range=(-1.0, 1.0),
                b_range=(-2.0, 2.0),
            ),
        ),
        Cell(
            why="1M every table row: positions 0..1048575 in 32 calls of 32768 rows, rope layer",
            spec=dict(
                _ML4_1M,
                tokens=MAX_NUM_TOKENS,
                positions="arange",
                graph=False,
                position_sweep=(0, LONG_MAX_SEQ_LEN),
            ),
        ),
    )

    CELLS += tuple(
        Cell(
            why=f"DSpark 1M verification: {n} rows at the table tail, {table}, captured",
            spec=dict(_ML4_1M, tokens=n, positions="tail", table=table, graph=True),
        )
        for n in (4, 52, 64)
        for table in ("yarn", "identity")
    )

    note = """
    Call sites: the mistral_large4 NVFP4 targets (tp4, tep4, dep4), every full-indexer layer
    (44 per step), decode and prefill, right after the `wq_b` GEMM. Replaces `ssmax_scaling`
    (seven torch launches), the head weights' fp32 copy (one) and
    `cute_dsl_indexer_q_side_fused` (one) with one launch.

    Numerics: the same operations as the composition it replaces, in the same order, at the
    same types, rounded at the same points; codes and weights are bitwise that composition's
    over every cell, on tensors captured from a served decode step, and over the NVFP4
    checkpoint's 44 indexer layers (their s and b, the 1048576-row table or the identity) at
    every position 0..1048575 (the test runs all three), and bitwise the composed references
    of the two entries. Stage by stage (the unfused op, then this kernel):
      ssmax_scaling              its seven ops (`ssmax_scaling_fused_q`'s note), bf16 output:
                                 the same stages, the same roundings
      .float().contiguous()      bf16 head weight widened to fp32 (exact): the same widening,
                                 at the load
      q side, rope half          bf16 to fp32, a subnormal flushed to its signed zero: the
                                 same flush on the widened value (zero exponent field)
      q side, nope half          bf16 to fp32 (exact): the same
      q side, rope               fma(x, cos, -(x' * sin)), fma(x', cos, x * sin) in fp32 with
                                 the partner product rounded first: tl.fma, the same operands
      q side, rope output        rounded to bf16 (round to nearest even): cvt.rn.bf16.f32
      q side, amax               max of |.| over the row, NaN propagating: the same max (exact)
      q side, scale              max(amax, 1e-12) * fp32(1 / 448), rounded up to a power of
                                 two from its exponent bits: the same fp32 product, the same
                                 integer arithmetic
      q side, codes              e4m3(x * 2^-k), cvt.rn.satfinite: the same conversion
      q side, weights            (wq * scale) * weight_scale, two fp32 products: the same
    Every intermediate rounding is an inline `cvt.rn.bf16.f32` (the compiler folds a bare
    bf16 round trip); the kernel is compiled without implicit FMA contraction
    (`enable_fp_fusion=False`), so the only fused multiply-adds are the rope's explicit ones,
    as in the q side. No fast-math function is used.

    Launch: Triton, grid (T,), one program per token over its 32 heads; programmatic
    dependent launch (waits on its predecessor before the first load). No host
    synchronization and no allocation: CUDA-graph capturable, a replay on new data writes
    what an eager call writes.

    `positions` is read on the device with no range check: a position at or past
    `cos_sin.shape[0]` reads past the table (`is_valid` states it).

    Hard limits (sm_103, T <= 32768): token offsets are int64; the table offset is
    position * 64 in int64; positions pass through fp32 for the log, exact below 2^24 (at most
    1048575 at the 1M config, its table's last row). None reached.

    Not certified: NeoX rope (no mode here), tables other than the NVFP4 checkpoint's indexer
    table and the identity table, positions past 1048575 (tables of more than 1048576 rows),
    more than 32768 rows, a q dtype other than bf16.
    """

    def __call__(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        s: torch.Tensor,
        b: torch.Tensor,
        cos_sin: torch.Tensor,
        wq: torch.Tensor,
        weight_scale: float,
        q_fp8_out: torch.Tensor,
        weights_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = q.shape[0]
        if tokens == 0:
            return q_fp8_out, weights_out
        _ssmax_indexer_q_side_kernel[(tokens,)](
            q, q.stride(0), q.stride(1), positions, s, s.stride(0), b, b.stride(0), cos_sin,
            wq, wq.stride(0), float(weight_scale), q_fp8_out, weights_out,
            USE_PDL=True, num_warps=4, launch_pdl=True, enable_fp_fusion=False,
        )  # fmt: skip
        return q_fp8_out, weights_out

    def reference(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        s: torch.Tensor,
        b: torch.Tensor,
        cos_sin: torch.Tensor,
        wq: torch.Tensor,
        weight_scale: float,
        q_fp8_out: torch.Tensor,
        weights_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fresh ``(q_fp8, weights)``: `ssmax_scaling`'s reference chain, then
        `cute_dsl_indexer_q_side_fused`'s reference (interleaved) on its bf16 output and the
        fp32 head weights; `q_fp8_out` and `weights_out` are not read."""
        scaled = ssmax_scaling.reference(q, positions, s, b)
        return cute_dsl_indexer_q_side_fused.reference(
            scaled, positions, cos_sin, wq.float(), weight_scale, q_fp8_out, weights_out,
            is_neox=False, compiled=None,
        )  # fmt: skip

    def is_valid(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        s: torch.Tensor,
        b: torch.Tensor,
        cos_sin: torch.Tensor,
        wq: torch.Tensor,
        weight_scale: float,
        q_fp8_out: torch.Tensor,
        weights_out: torch.Tensor,
    ) -> None:
        # Each is addressed through a layout the kernel assumes and answers wrongly otherwise.
        name = "ssmax_indexer_q_side_fused"
        if q.dtype != torch.bfloat16 or q.dim() != 3 or tuple(q.shape[1:]) != (HEADS, HEAD_DIM):
            raise ValueError(f"{name}: q must be bf16 [T, 32, 128], got {tuple(q.shape)}")
        tokens = q.shape[0]
        if q.stride(2) != 1:
            raise ValueError(f"{name}: q needs a unit stride over the head dim")
        if tuple(positions.shape) != (tokens,) or positions.dtype != torch.int32:
            raise ValueError(f"{name}: positions must be int32 [T] = [{tokens}]")
        for pname, t in (("s", s), ("b", b)):
            if tuple(t.shape) != (HEADS,):
                raise ValueError(f"{name}: {pname} must be [32]")
        if cos_sin.dtype != torch.float32 or cos_sin.dim() != 2 or cos_sin.shape[1] != 64:
            raise ValueError(f"{name}: cos_sin must be fp32 [positions, 64]")
        if not cos_sin.is_contiguous():
            raise ValueError(f"{name}: cos_sin must be contiguous")
        if (
            tuple(wq.shape) != (tokens, HEADS)
            or wq.dtype not in (torch.bfloat16, torch.float32)
            or wq.stride(1) != 1
        ):
            raise ValueError(f"{name}: wq must be bf16 or fp32 [T, 32] with a unit head stride")
        if (
            q_fp8_out.dtype != torch.float8_e4m3fn
            or tuple(q_fp8_out.shape) != (tokens, HEADS, HEAD_DIM)
            or not q_fp8_out.is_contiguous()
        ):
            raise ValueError(f"{name}: q_fp8_out must be contiguous e4m3 [T, 32, 128]")
        if (
            weights_out.dtype != torch.float32
            or tuple(weights_out.shape) != (tokens, HEADS)
            or not weights_out.is_contiguous()
        ):
            raise ValueError(f"{name}: weights_out must be contiguous fp32 [T, 32]")
        # Positions are read on the device with no bound.
        if positions.numel():
            lo, hi = int(positions.min()), int(positions.max())
            if lo < 0 or hi >= cos_sin.shape[0]:
                raise ValueError(
                    f"{name}: position_ids span [{lo}, {hi}] but the rope table has "
                    f"{cos_sin.shape[0]} rows; the kernel reads past it"
                )


ssmax_indexer_q_side_fused = SSMaxIndexerQSideFused()
