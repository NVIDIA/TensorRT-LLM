# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SSMax query scaling of the MLA query, with its rope half written into the absorbed query's
tail, in one Triton launch: ``ssmax_scaling``'s bf16 chain bit for bit, plus the copy (or, on a
NoPE layer, the zero fill) of the ``[T, H, 64]`` tail of the 576-wide fused query FlashMLA
reads."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.language.extra.cuda import gdc_wait

from .._op import Arch, Cell, OpWrapper
from .ssmax_scaling import ssmax_scaling

HEAD_DIM = 128
#: The query's two halves: [nope(64) | rope(64)] on a rope layer; a NoPE layer's 128 are all
#: nope and its tail is zero.
HALF = 64
#: Heads one program scales: a rank's 16 (tp4/tep4) or 8 (tp8/tep8) in one, dep4's 64 in four.
_HEADS_PER_PROGRAM = 16

#: max_seq_len of the NVFP4 targets' 32K config, which bounds every position there.
MAX_SEQ_LEN = 32768
#: Their 1M config: the checkpoint's own context (max_seq_len 1048576), prefilled in chunked
#: steps of at most MAX_NUM_TOKENS rows, a chunk's rows at cached .. cached + n - 1 behind its
#: request's cached prefix, decode rows at any position up to 1048575 (log 13.86).
LONG_MAX_SEQ_LEN = 1048576
MAX_NUM_TOKENS = 32768
#: The 1M config's decode batches (max_batch_size 16): the padded 1/2/4/8/16 and unpadded ones.
LONG_GRAPH_BATCHES = (1, 2, 3, 4, 5, 7, 8, 13, 16)
#: Decode batches the targets capture: the padded 1/2/4/8/16 and the unpadded 3..15 of a
#: max_batch_size-16 engine, then the larger padded batches a max_batch_size-256 engine
#: captures (the Pareto and decode-profile configurations).
GRAPH_BATCHES = (1, 2, 3, 4, 5, 7, 8, 13, 16, 32, 64, 128, 256)

#: The draw ranges hold the NVFP4 checkpoint's MLA s and b over all 72 layers x 64 heads
#: (read on disk, bf16): s in [-0.078125, 0.574219], b in [0.902344, 1.57031].
_MLA = dict(
    dtype=torch.bfloat16,
    positions_dtype=torch.int32,
    param_heads=64,
    s_range=(-0.08, 0.58),
    b_range=(0.9, 1.6),
    max_seq_len=MAX_SEQ_LEN,
    q="normal",
    corners=True,
    inplace=True,
)


@triton.jit
def _round_bf16(x):
    """fp32 rounded to bf16 (to nearest, ties to even) and widened back. The rounding is an
    inline `cvt.rn.bf16.f32`: the compiler folds a bare `.to(bf16).to(fp32)` pair into
    nothing, which skips the rounding the chain specifies."""
    r = tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;", "=h,r", [x], dtype=tl.bfloat16, is_pure=True, pack=1
    )
    return r.to(tl.float32)


@triton.jit
def _ssmax_fused_q_kernel(
    q_ptr,  # bf16 [T, H, 128], unit stride over the head dim
    q_st,
    q_sh,
    pos_ptr,  # int32/int64 [T]
    s_ptr,  # [H]
    s_st,
    b_ptr,  # [H]
    b_st,
    out_ptr,  # bf16 [T, H, 128], unit stride over the head dim; may be q itself
    o_st,
    o_sh,
    tail_ptr,  # bf16 [T, H, 64], unit stride over the last dim
    t_st,
    t_sh,
    heads,
    ROPE: tl.constexpr,
    HB: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * HB + tl.arange(0, HB)
    hmask = h < heads
    d = tl.arange(0, 64)
    m2 = hmask[:, None]
    if USE_PDL:
        gdc_wait()
    # The chain of `ssmax_scaling`, one rounding per torch op: the position to fp32, + 1,
    # logf, rounded to bf16; then * s and + b, each rounded to bf16; then q * scale.
    pos = tl.load(pos_ptr + t).to(tl.float32)
    lp = _round_bf16(libdevice.log(pos + 1.0))
    s = _round_bf16(tl.load(s_ptr + h * s_st, mask=hmask, other=0.0).to(tl.float32))
    b = _round_bf16(tl.load(b_ptr + h * b_st, mask=hmask, other=0.0).to(tl.float32))
    m = _round_bf16(lp * s)
    scale = _round_bf16(m + b)[:, None]
    q_row = q_ptr + t * q_st + h[:, None].to(tl.int64) * q_sh + d[None, :]
    lo = tl.load(q_row, mask=m2, other=0.0).to(tl.float32)
    hi = tl.load(q_row + 64, mask=m2, other=0.0).to(tl.float32)
    lo = (lo * scale).to(tl.bfloat16)
    hi = (hi * scale).to(tl.bfloat16)
    o_row = out_ptr + t * o_st + h[:, None].to(tl.int64) * o_sh + d[None, :]
    tl.store(o_row, lo, mask=m2)
    tl.store(o_row + 64, hi, mask=m2)
    t_row = tail_ptr + t * t_st + h[:, None].to(tl.int64) * t_sh + d[None, :]
    if ROPE:
        tl.store(t_row, hi, mask=m2)
    else:
        tl.store(t_row, tl.zeros([HB, 64], dtype=tl.bfloat16), mask=m2)


class SSMaxScalingFusedQ(OpWrapper):
    """``out = ssmax_scaling(q, positions, s, b)``, and the absorbed query's rope tail from it.

    `q` is the MLA query ``[T, H, 128]`` bf16, `positions` the step's ``[T]`` position ids,
    `s` and `b` this rank's ``[H]`` slice of the 64 per-head parameters (all 64 at dep4).
    `out` (``[T, H, 128]`` bf16, may be `q` itself) receives the scaled query, bitwise
    `ssmax_scaling`'s, and `tail` (``[T, H, 64]`` bf16, the ``[..., 512:576]`` view of this
    rank's heads of the 576-wide fused query) receives ``out[..., 64:]`` when `rope`, zeros
    otherwise (a NoPE layer, whose 128 dims are all nope and whose query tail is zero).
    Returns `out`.

    Fusion boundary: the scaling, the rope half's copy into the fused query and a NoPE
    layer's zero tail. The K-B absorption of ``out[..., :nope]`` into the fused query's first
    512 columns, and the rope rotation of the tail (``ds_mla_rope_append`` on owner layers),
    are the caller's; nothing here reads or writes those columns or the other heads.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: `heads` of a `param_heads` draw at rank `rank`'s slice, `tokens` rows at
    # positions: each context's 0..n-1 in `contexts` order (c..c + n - 1 behind a `cached`
    # prefix c), then `decode_rows` random positions in 1..max_seq_len - 1 (the first at
    # max_seq_len - 1); `position_sweep` (lo, hi) drives every position lo..hi - 1 in calls of
    # `tokens` rows, each on a fresh q, the same s and b. `corners` pins four of
    # the slice's heads to the (s, b) corners of the ranges. `rope` picks the tail's
    # contents; `inplace` passes `q` as `out` (the targets' call); `graph` captures the
    # call and replays it on new q and positions. `q` "wide" draws every bf16 binade.
    # `fq_heads` (64 when absent, FlashMLA's padded query) is the head count of the fused
    # query the tail is a view of: `heads` is the tp4 / tep4 decode targets' own
    # `[T, heads, 576]` scratch for cute_dsl_ds_mla_sparse_decode.
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"tp4/tep4 decode, graph batch {n:>3}, rank {n % 4}'s 16 heads, rope layer: "
                "captured, replayed on new q and positions",
                spec=dict(
                    _MLA,
                    heads=16,
                    rank=n % 4,
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    rope=True,
                    graph=True,
                ),
            )
            for n in GRAPH_BATCHES
        )
        + tuple(
            Cell(
                why=f"tp4/tep4 decode, graph batch {n:>3}, NoPE layer: zero tail, captured",
                spec=dict(
                    _MLA,
                    heads=16,
                    rank=1,
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    rope=False,
                    graph=True,
                ),
            )
            for n in (1, 16, 256)
        )
        + tuple(
            Cell(
                why=f"dep4 decode, graph batch {n:>3}, all 64 heads (four programs per row), "
                f"{'rope' if rope else 'NoPE'} layer, captured",
                spec=dict(
                    _MLA,
                    heads=64,
                    rank=0,
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    rope=rope,
                    graph=True,
                ),
            )
            for n in (1, 16, 64, 256)
            for rope in (True, False)
        )
        + tuple(
            Cell(
                why=f"tp8/tep8 decode, graph batch {n:>3}, rank {n % 8}'s 8 heads (a half-masked "
                "program), rope layer, captured",
                spec=dict(
                    _MLA,
                    heads=8,
                    rank=n % 8,
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    rope=True,
                    graph=True,
                ),
            )
            for n in (1, 7, 16, 256)
        )
        + (
            Cell(
                why="tp4 context of 32768 rows at positions 0..32767, rank 1's heads, rope layer: "
                "every position of the envelope",
                spec=dict(
                    _MLA,
                    heads=16,
                    rank=1,
                    tokens=32768,
                    contexts=(32768,),
                    decode_rows=0,
                    rope=True,
                    graph=False,
                ),
            ),
            Cell(
                why="tp4 mixed step of 32768 rows, 3 contexts + 13 decode rows, NoPE layer, "
                "out-of-place",
                spec=dict(
                    _MLA,
                    heads=16,
                    rank=2,
                    tokens=32768,
                    contexts=(17000, 11000, 4755),
                    decode_rows=13,
                    rope=False,
                    graph=False,
                    inplace=False,
                ),
            ),
            Cell(
                why="dep4 context of 32768 rows, all 64 heads, rope layer",
                spec=dict(
                    _MLA,
                    heads=64,
                    rank=0,
                    tokens=32768,
                    contexts=(32768,),
                    decode_rows=0,
                    rope=True,
                    graph=False,
                ),
            ),
            Cell(
                why="tp8 context of 32768 rows, rank 7's 8 heads, rope layer",
                spec=dict(
                    _MLA,
                    heads=8,
                    rank=7,
                    tokens=32768,
                    contexts=(32768,),
                    decode_rows=0,
                    rope=True,
                    graph=False,
                ),
            ),
            Cell(
                why="stress at 32768 rows, s in [-1, 1] and b in [-2, 2], q over every bf16 "
                "binade: cancellation, overflow, subnormals, out-of-place",
                spec=dict(
                    _MLA,
                    heads=16,
                    rank=3,
                    s_range=(-1.0, 1.0),
                    b_range=(-2.0, 2.0),
                    q="wide",
                    tokens=32768,
                    contexts=(32768,),
                    decode_rows=0,
                    rope=True,
                    graph=False,
                    inplace=False,
                ),
            ),
        )
        # The tp4 / tep4 decode step on cute_dsl_ds_mla_sparse_decode (16 heads, 8 at tp8 /
        # tep8): the tail is a view of the rank's unpadded `[T, heads, 576]` query scratch (row
        # stride heads x 576), every element of which the absorption BMM and this tail write
        # rewrite each layer.
        + tuple(
            Cell(
                why=f"{'tp4/tep4' if h == 16 else 'tp8/tep8'} decode {n:>3} "
                f"{'rope' if rope else 'NoPE'}, native [T, {h}, 576] scratch: the tail at row "
                f"stride {h} x 576, captured",
                spec=dict(
                    _MLA,
                    heads=h,
                    rank=n % (64 // h),
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    rope=rope,
                    graph=True,
                    fq_heads=h,
                ),
            )
            for h in (16, 8)
            for n in (1, 16, 64, 256)
            for rope in (True, False)
        )
    )
    # The 1M config (max_seq_len 1048576, chunked prefill, max_num_tokens 32768, 16 sequences):
    # decode rows at positions up to 1048575, chunk rows behind a cached prefix, mixed and packed
    # steps, and every position 0..1048575 swept in calls of 32768 rows (`position_sweep`), at
    # the rank's 16 heads (tp4 / tep4), 8 (tp8 / tep8) and all 64 (dep4), into FlashMLA's
    # 64-head padded fused query.
    CELLS += tuple(
        Cell(
            why=f"1M tp4/tep4 decode, graph batch {n:>2}, rank {n % 4}'s 16 heads, rope layer: "
            "positions up to 1048575, captured, replayed on new q and positions",
            spec=dict(
                _MLA,
                max_seq_len=LONG_MAX_SEQ_LEN,
                heads=16,
                rank=n % 4,
                tokens=n,
                contexts=(),
                decode_rows=n,
                rope=True,
                graph=True,
            ),
        )
        for n in LONG_GRAPH_BATCHES
    )
    CELLS += tuple(
        Cell(
            why=f"1M {name} decode, graph batch {n:>2}, {heads} heads, "
            f"{'rope' if rope else 'NoPE'} layer: positions up to 1048575, captured",
            spec=dict(
                _MLA,
                max_seq_len=LONG_MAX_SEQ_LEN,
                heads=heads,
                rank=n % (64 // heads),
                tokens=n,
                contexts=(),
                decode_rows=n,
                rope=rope,
                graph=True,
            ),
        )
        for name, heads in (("tp4/tep4", 16), ("tp8/tep8", 8), ("dep4", 64))
        for n in (1, 16)
        for rope in (True, False)
        if not (heads == 16 and rope)
    )
    CELLS += (
        Cell(
            why="1M tp4 last chunk: 32768 rows behind a 1015808-token cached prefix, positions "
            "1015808..1048575, rank 1's heads, rope layer",
            spec=dict(
                _MLA,
                max_seq_len=LONG_MAX_SEQ_LEN,
                heads=16,
                rank=1,
                tokens=MAX_NUM_TOKENS,
                contexts=(MAX_NUM_TOKENS,),
                cached=(LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,),
                decode_rows=0,
                rope=True,
                graph=False,
            ),
        ),
        Cell(
            why="1M dep4 last chunk, all 64 heads, positions 1015808..1048575, NoPE layer",
            spec=dict(
                _MLA,
                max_seq_len=LONG_MAX_SEQ_LEN,
                heads=64,
                rank=0,
                tokens=MAX_NUM_TOKENS,
                contexts=(MAX_NUM_TOKENS,),
                cached=(LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,),
                decode_rows=0,
                rope=False,
                graph=False,
            ),
        ),
        Cell(
            why="1M tp4 mixed step of 32768 rows, 3 chunk tails + 13 decode rows: chunks at "
            "positions 1031576.., 500000.. and 0.., decode rows up to 1048575, out-of-place",
            spec=dict(
                _MLA,
                max_seq_len=LONG_MAX_SEQ_LEN,
                heads=16,
                rank=2,
                tokens=MAX_NUM_TOKENS,
                contexts=(17000, 11000, 4755),
                cached=(LONG_MAX_SEQ_LEN - 17000, 500000, 0),
                decode_rows=13,
                rope=True,
                graph=False,
                inplace=False,
            ),
        ),
        Cell(
            why="1M tp8 packed step of 32768 rows, 4 chunks + 13 decode rows: chunks at "
            "1032192.., 524288.., 98304.. and 0.., rank 7's 8 heads, NoPE layer",
            spec=dict(
                _MLA,
                max_seq_len=LONG_MAX_SEQ_LEN,
                heads=8,
                rank=7,
                tokens=MAX_NUM_TOKENS,
                contexts=(16384, 8192, 4096, 4083),
                cached=(LONG_MAX_SEQ_LEN - 16384, 524288, 98304, 0),
                decode_rows=13,
                rope=False,
                graph=False,
            ),
        ),
        Cell(
            why="stress at the 1M last chunk, s in [-1, 1] and b in [-2, 2], q over every bf16 "
            "binade, positions 1015808..1048575: cancellation, overflow, subnormals",
            spec=dict(
                _MLA,
                max_seq_len=LONG_MAX_SEQ_LEN,
                heads=16,
                rank=3,
                s_range=(-1.0, 1.0),
                b_range=(-2.0, 2.0),
                q="wide",
                tokens=MAX_NUM_TOKENS,
                contexts=(MAX_NUM_TOKENS,),
                cached=(LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,),
                decode_rows=0,
                rope=True,
                graph=False,
                inplace=False,
            ),
        ),
    )
    CELLS += tuple(
        Cell(
            why=f"1M every position 0..1048575 in 32 calls of 32768 rows, {heads} heads "
            f"(rank {rank}), {'rope' if rope else 'NoPE'} layer, the range corners",
            spec=dict(
                _MLA,
                max_seq_len=LONG_MAX_SEQ_LEN,
                heads=heads,
                rank=rank,
                tokens=MAX_NUM_TOKENS,
                contexts=(MAX_NUM_TOKENS,),
                decode_rows=0,
                rope=rope,
                graph=False,
                position_sweep=(0, LONG_MAX_SEQ_LEN),
            ),
        )
        for heads, rank, rope in ((16, 0, True), (8, 5, False), (64, 0, True))
    )
    # The 1M config's decode step on cute_dsl_ds_mla_sparse_decode: the tail a view of the
    # rank's unpadded `[T, heads, 576]` query scratch (row stride heads x 576), positions up to
    # 1048575, and every position swept in that layout.
    CELLS += tuple(
        Cell(
            why=f"1M {'tp4/tep4' if h == 16 else 'tp8/tep8'} decode {n:>2} "
            f"{'rope' if rope else 'NoPE'}, native [T, {h}, 576] scratch: positions up to "
            "1048575, captured",
            spec=dict(
                _MLA,
                max_seq_len=LONG_MAX_SEQ_LEN,
                heads=h,
                rank=n % (64 // h),
                tokens=n,
                contexts=(),
                decode_rows=n,
                rope=rope,
                graph=True,
                fq_heads=h,
            ),
        )
        for h in (16, 8)
        for n in (1, 13, 16)
        for rope in (True, False)
    )
    CELLS += tuple(
        Cell(
            why=f"1M every position 0..1048575 in 32 calls, native [T, {h}, 576] scratch, rank "
            f"{rank}'s {h} heads, {'rope' if rope else 'NoPE'} layer",
            spec=dict(
                _MLA,
                max_seq_len=LONG_MAX_SEQ_LEN,
                heads=h,
                rank=rank,
                tokens=MAX_NUM_TOKENS,
                contexts=(MAX_NUM_TOKENS,),
                decode_rows=0,
                rope=rope,
                graph=False,
                position_sweep=(0, LONG_MAX_SEQ_LEN),
                fq_heads=h,
            ),
        )
        for h, rank, rope in ((16, 2, True), (8, 6, False))
    )

    CELLS += tuple(
        Cell(
            why=f"DSpark 1M verification: {n} rows, {h} heads, rope={rope}, captured",
            spec=dict(
                _MLA,
                max_seq_len=LONG_MAX_SEQ_LEN,
                heads=h,
                rank=0,
                tokens=n,
                contexts=(),
                decode_rows=n,
                rope=rope,
                graph=True,
                fq_heads=h,
            ),
        )
        for h in (8, 16, 64)
        for n in (52, 64)
        for rope in (True, False)
    )

    note = """
    Call sites: the mistral_large4 NVFP4 targets (tp4, tep4, dep4), every layer, context and
    generation rows, right after the `wq` GEMM; the targets pass `q` as `out`. Replaces
    `ssmax_scaling` (seven torch launches) plus the tail's strided copy or zero fill (one
    more) with one launch. The tail is a view of FlashMLA's 64-head padded fused query, or,
    on the tp4 / tep4 decode step that runs cute_dsl_ds_mla_sparse_decode, of the rank's own
    `[T, heads, 576]` query (the kernel takes the tail's row and head strides).

    Numerics: the same operations as the unfused ops, in the same order, at the same types,
    rounded at the same points; `out` and `tail` are bitwise those ops' results over every
    cell, on tensors captured from a served decode step, and over the NVFP4 checkpoint's own
    s and b of all 72 layers at every position 0..1048575 in every call site's layout of the
    fused query (the test runs all three). Stage by stage (the unfused op, then this
    kernel):
      positions.to(float32)      int to fp32 (exact below 2^24)   the same conversion
      + 1                        fp32 add                         fp32 add
      torch.log                  fp32 logf (libdevice __nv_logf)  libdevice.log, the same
                                                                  function: fp32-bitwise at
                                                                  every pos + 1 below 2^24
                                                                  and 2^24 random fp32 (tested)
      .to(bf16)                  round to nearest even            cvt.rn.bf16.f32
      s.to(bf16), b.to(bf16)     no-op on the bf16 parameters     cvt.rn.bf16.f32 (no-op)
      log_pos * s                fp32 product, rounded to bf16    fp32 product, cvt.rn.bf16
      + b                        fp32 sum, rounded to bf16        fp32 sum, cvt.rn.bf16
      q * scale                  fp32 product, rounded to bf16    fp32 product, cvt.rn.bf16
      tail copy_ / zero_         exact copy / zero fill           store of the same bf16 / 0
    Every intermediate rounding is an inline `cvt.rn.bf16.f32`, because the compiler folds a
    bare `.to(bf16).to(fp32)` pair into nothing; the kernel is compiled without implicit FMA
    contraction (`enable_fp_fusion=False`), so no product and sum are fused. No fast-math
    function is used. `ssmax_scaling`'s note derives where the chain sits from the closed
    form; nothing here changes it.

    Launch: Triton, grid (T, ceil(H / 16)), 16 heads (8 at tp8/tep8) per program, two
    64-wide halves per head; programmatic dependent launch (waits on its predecessor before
    the first load). No host synchronization and no allocation: CUDA-graph capturable, a
    replay on new q and positions writes what an eager call writes.

    Rows and heads are independent: a padded decode row reaches no live one. In place is
    safe: every element is loaded before the program stores it, and no program reads
    another's elements. `tail` must not overlap `q` or `out`.

    Hard limits (sm_103, T <= 32768 rows a call): row offsets are int64; positions pass
    through fp32, exact below 2^24 (positions <= 1048575 at the 1M config); the grid's T
    dimension is the x dimension (2^31 - 1). None reached.

    Not certified: a q dtype other than bf16, head_dim other than 128, heads other than 8,
    16 and 64, positions above 1048575, a non-unit stride over the head dim of q, out or
    tail.
    """

    def __call__(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        s: torch.Tensor,
        b: torch.Tensor,
        out: torch.Tensor,
        tail: torch.Tensor,
        rope: bool,
    ) -> torch.Tensor:
        tokens, heads = q.shape[0], q.shape[1]
        if tokens == 0:
            return out
        hb = min(_HEADS_PER_PROGRAM, triton.next_power_of_2(heads))
        _ssmax_fused_q_kernel[(tokens, triton.cdiv(heads, hb))](
            q, q.stride(0), q.stride(1), positions, s, s.stride(0), b, b.stride(0),
            out, out.stride(0), out.stride(1), tail, tail.stride(0), tail.stride(1), heads,
            ROPE=bool(rope), HB=hb, USE_PDL=True, num_warps=4, launch_pdl=True,
            enable_fp_fusion=False,
        )  # fmt: skip
        return out

    def reference(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        s: torch.Tensor,
        b: torch.Tensor,
        out: torch.Tensor,
        tail: torch.Tensor,
        rope: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fresh ``(out, tail)``: `ssmax_scaling`'s reference chain, then its rope half or
        zeros; `out` and `tail` are not read."""
        scaled = ssmax_scaling.reference(q, positions, s, b)
        new_tail = scaled[..., HALF:].clone() if rope else torch.zeros_like(scaled[..., HALF:])
        return scaled, new_tail

    def is_valid(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        s: torch.Tensor,
        b: torch.Tensor,
        out: torch.Tensor,
        tail: torch.Tensor,
        rope: bool,
    ) -> None:
        # Each is addressed through strides the kernel assumes, and answers wrongly otherwise.
        if q.dim() != 3 or q.shape[2] != HEAD_DIM or q.dtype != torch.bfloat16:
            raise ValueError(f"ssmax_scaling_fused_q: q must be bf16 [T, H, 128], got {q.shape}")
        tokens, heads = q.shape[0], q.shape[1]
        if tuple(positions.shape) != (tokens,):
            raise ValueError(f"ssmax_scaling_fused_q: positions must be [T] = [{tokens}]")
        for name, t in (("s", s), ("b", b)):
            if tuple(t.shape) != (heads,):
                raise ValueError(f"ssmax_scaling_fused_q: {name} must be [H] = [{heads}]")
        if tuple(out.shape) != tuple(q.shape) or out.dtype != torch.bfloat16:
            raise ValueError("ssmax_scaling_fused_q: out must be bf16 shaped like q")
        if tuple(tail.shape) != (tokens, heads, HALF) or tail.dtype != torch.bfloat16:
            raise ValueError(f"ssmax_scaling_fused_q: tail must be bf16 [T, H, {HALF}]")
        for name, t in (("q", q), ("out", out), ("tail", tail)):
            if t.stride(2) != 1:
                raise ValueError(f"ssmax_scaling_fused_q: {name} needs a unit last stride")
        if out.data_ptr() != q.data_ptr() and _overlap(out, q):
            raise ValueError("ssmax_scaling_fused_q: out must be q itself or not overlap it")
        if _overlap(tail, q) or _overlap(tail, out):
            raise ValueError("ssmax_scaling_fused_q: tail must not overlap q or out")


def _overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether the byte spans of `a` and `b` intersect."""
    a0, b0 = a.data_ptr(), b.data_ptr()
    return a0 < b0 + _span(b) and b0 < a0 + _span(a)


def _span(t: torch.Tensor) -> int:
    """Bytes from `t`'s first element to one past its last."""
    if t.numel() == 0:
        return 0
    last = sum((n - 1) * st for n, st in zip(t.shape, t.stride()))
    return (last + 1) * t.element_size()


ssmax_scaling_fused_q = SSMaxScalingFusedQ()
