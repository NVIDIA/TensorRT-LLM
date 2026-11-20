# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode rows (1..64) of q_b and o_proj on the MXF8 weights the context path keeps: a PDL
activation quant into a persistent 64-row operand and a tcgen05 block-scaled GEMM that
streams its weight before its PDL wait, or the GEMM alone when the producer wrote the
operand. `build_wide`: the same for o_proj at the MTP3 / MTP5 decode rows (65..192), on a
192-row operand."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common import self_attn_prefill_act_quant_mxf8 as _quant
from ...kernels.gemm import self_attn_decode_projections_mxf8 as _kernel

#: Entry -> the kernel's op.
ENTRIES: dict[str, str] = {"q_b": "q_b_proj", "o_proj": "o_proj"}
#: The largest row count: the decode CUDA graphs' batch.
ROWS_MAX = _kernel.ROWS_MAX
#: The entries `build_wide` serves, and the row counts (inclusive) of the MTP3 / MTP5 decode
#: steps they take: 32 sequences x widths 4 / 6 and the ragged batches of fewer sequences.
WIDE_ENTRIES: tuple[str, ...] = tuple(_kernel.WIDE_CONFIGS)
WIDE_ROWS = (_kernel.WIDE_ROWS_MIN, _kernel.WIDE_ROWS_MAX)
QuantizedWeight = _kernel.QuantizedWeight


@dataclass(frozen=True)
class DecodeMxf8Runner:
    """One entry's launcher. `operand` is its `(xq [R, K] uint8, sfa)` (R = 64, or 192 for a
    wide runner): the activation the GEMM reads, written by the launcher's quant or by a
    producer (allocated once, fixed addresses a producer kernel may bake in)."""

    kernel: str
    n: int
    k: int
    launch: _kernel.DecodeMxf8Projection | _kernel.WideDecodeMxf8Projection

    @property
    def operand(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.launch.xq, self.launch.sfa

    @property
    def rows(self) -> tuple[int, int]:
        """The row counts the launcher serves, inclusive."""
        return self.launch.rows_min, self.launch.rows_max


def build(kernel: str, weight: QuantizedWeight) -> DecodeMxf8Runner:
    """Compile `kernel`'s quant and GEMM and launch each at 64 rows and 1 row, so no decode
    step or graph capture meets a first module load. `weight` is any layer's weight."""
    dev = weight.q.device
    launch = _kernel.build(ENTRIES[kernel], dev.index)
    with torch.cuda.device(dev):
        for rows in (ROWS_MAX, 1):
            x = torch.zeros(rows, launch.k, dtype=torch.bfloat16, device=dev)
            launch(x, weight, torch.empty(rows, launch.n, dtype=torch.bfloat16, device=dev))
        torch.cuda.synchronize(dev)
    launch.hits = launch.pre_hits = 0
    return DecodeMxf8Runner(kernel, launch.n, launch.k, launch)


def build_wide(kernel: str, weight: QuantizedWeight) -> DecodeMxf8Runner:
    """`kernel`'s launcher for the MTP3 / MTP5 decode rows (`WIDE_ROWS`): compile its quant
    and GEMM instances and launch them at 192 and 128 rows (every instance once), so no
    decode step or graph capture meets a first module load. `weight` is any layer's weight."""
    dev = weight.q.device
    launch = _kernel.build_wide(ENTRIES[kernel], dev.index)
    with torch.cuda.device(dev):
        for rows in (launch.rows_max, 128):
            x = torch.zeros(rows, launch.k, dtype=torch.bfloat16, device=dev)
            launch(x, weight, torch.empty(rows, launch.n, dtype=torch.bfloat16, device=dev))
        torch.cuda.synchronize(dev)
    launch.hits = launch.pre_hits = 0
    return DecodeMxf8Runner(kernel, launch.n, launch.k, launch)


def shift_scales(w: QuantizedWeight, exponent: int) -> QuantizedWeight:
    """`w`'s e4m3 bytes with every block scale multiplied by `2^exponent`: the weight of
    `2^exponent * W`, exact while no code leaves 0..254. The target derives decode q_b from
    the fp8 attention's copy of `q_b / s_q` (s_q a power of two) this way."""
    sf = w.sf.to(torch.int32) + int(exponent)
    if int(sf.min()) < 0 or int(sf.max()) > 254:
        raise ValueError(f"a block scale leaves the e8m0 range shifted by 2^{exponent}")
    return QuantizedWeight(w.q, sf.to(torch.uint8))


class CuteDslProjectionMxf8Decode(OpWrapper):
    """`out = x @ W^T` for `x` bf16 `[M, K]`, M in the runner's `rows` (1..64; 65..192 for
    a `build_wide` runner), both operands MXF8 (the activation quant's rule), fp32
    accumulation in K order, bf16 out.

    `x` needs a unit inner stride, a row pitch that is a multiple of 8 and 16-byte
    alignment; `weight` is the `[N, K]` `QuantizedWeight` (128-byte aligned); `out` a
    contiguous bf16 `[M, N]`, written and returned. With `pre=True` the operand's first M
    rows (`runner.operand`) already hold `x` quantized -- its producer, the launch just
    before on the stream, wrote them -- and only the GEMM runs; `x` is then read for its
    row count only.

        q_b     [M, 2048] x [4096, 2048]
        o_proj  [M, 4096] x [6144, 4096]
    """

    ARCHS = frozenset({Arch.SM_103})

    # The decode graphs' row counts: 1, a ragged count, the full batch; each entry through
    # its own quant and on a producer's operand.
    CELLS: tuple[Cell, ...] = (
        Cell(why="q_b at one row", spec=dict(kernel="q_b", rows=1)),
        Cell(why="q_b at 37 rows, a ragged batch", spec=dict(kernel="q_b", rows=37)),
        Cell(
            why="q_b at 64 rows on the q_a norm's operand",
            spec=dict(kernel="q_b", rows=64, pre=True),
        ),
        Cell(why="o_proj at 2 rows, one sequence x MTP 2", spec=dict(kernel="o_proj", rows=2)),
        Cell(why="o_proj at 64 rows", spec=dict(kernel="o_proj", rows=64)),
        Cell(
            why="o_proj at 48 rows on the V-B GEMM's side output",
            spec=dict(kernel="o_proj", rows=48, pre=True),
        ),
        # The wide runner (MTP3 / MTP5 decode rows): the template (up to 128 rows) and the
        # context PDL GEMM (above), the first wide count, a ragged batch, through its own quant
        # and on the V-B's side output.
        Cell(
            why="wide o_proj at 128 rows, MTP3's 32 x 4, on the V-B side output",
            spec=dict(kernel="o_proj", rows=128, pre=True, wide=True),
        ),
        Cell(
            why="wide o_proj at 192 rows, MTP5's 32 x 6, on the V-B side output",
            spec=dict(kernel="o_proj", rows=192, pre=True, wide=True),
        ),
        Cell(
            why="wide o_proj at 65 rows, the first wide count, through its quant",
            spec=dict(kernel="o_proj", rows=65, wide=True),
        ),
        Cell(
            why="wide o_proj at 150 rows, 25 x 6 on the context PDL instance",
            spec=dict(kernel="o_proj", rows=150, wide=True),
        ),
    )

    note = """
    The target registers each trunk layer's o_proj weight (the context path's MXF8 copy)
    and q_b weight (the fp8 attention's copy with its Q scale shifted back out of the block
    scales, checked against a direct quantization at load) and takes this op for decode
    rows. On a pure decode step the operand is written by its producer: the q_a norm
    (`cute_dsl_qkv_a_rmsnorm_q8`) for q_b and the V-B GEMM's side output for o_proj, so the
    GEMM runs alone. Rows of a mixed step and the MTP layer keep the bf16 decode cells.
    On B300 behind a PDL predecessor: o_proj 11.69 -> 9.11 us, q_b 6.12 -> 5.65 us per layer
    at 64 rows, and 2.3 us per launch more where the producer writes the operand.

    An engine drafting more than one token also builds the wide o_proj runner
    (`build_wide`) and takes it for an MTP3 / MTP5 decode step's trunk o_proj rows above 64,
    on the operand the MTP group's V-B side output writes. Its K order and operands are the
    context MXF8 GEMM's, which those rows took before: the output is that GEMM's, bitwise.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: QuantizedWeight,
        out: torch.Tensor,
        runner: DecodeMxf8Runner,
        pre: bool = False,
    ) -> torch.Tensor:
        if pre:
            return runner.launch.gemm_only(weight, out, int(x.shape[0]))
        return runner.launch(x, weight, out)

    def reference(
        self,
        x: torch.Tensor,
        weight: QuantizedWeight,
        out: torch.Tensor,
        runner: DecodeMxf8Runner,
        pre: bool = False,
    ) -> torch.Tensor:
        """The same MXF8 operands multiplied in fp32 (`x` quantized by the act quant's torch
        twin, or the operand's rows as they are with `pre`), rounded to bf16, as a new
        tensor."""
        rows = int(x.shape[0])
        if pre:
            xq, sf = runner.operand
            a = _quant.dequantize(xq[:rows], sf, rows, runner.k)
        else:
            xq, sf = _quant.reference(x.contiguous())
            a = _quant.dequantize(xq, sf, rows, runner.k)
        b = _quant.dequantize(weight.q, weight.sf, runner.n, runner.k)
        return (a @ b.t()).to(torch.bfloat16)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: QuantizedWeight,
        out: torch.Tensor,
        runner: DecodeMxf8Runner,
        pre: bool = False,
    ) -> None:
        rows = int(x.shape[0])
        lo, hi = runner.rows
        assert lo <= rows <= hi, f"rows must be in {lo}..{hi}, got {rows}"
        if not pre:
            runner.launch.check(x, weight, out)
            return
        assert tuple(out.shape) == (rows, runner.n) and out.dtype == torch.bfloat16, (
            f"out must be bf16 [{rows}, {runner.n}]"
        )
        assert tuple(weight.q.shape) == (runner.n, runner.k), (
            f"weight must be the MXF8 [{runner.n}, {runner.k}]"
        )


cute_dsl_projection_mxf8_decode = CuteDslProjectionMxf8Decode()
