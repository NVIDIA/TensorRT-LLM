# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE shared expert's gate|up on context rows as one MXF8 block-scaled GEMM with the
SwiGLU in its epilogue: bf16 `[T, 512]` out, the operand the bf16 down projection reads."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.gemm import moe_prefill_shared_up_swiglu_mxf8 as _kernel
from ...kernels.gemm import self_attn_prefill_projections_mxf8 as _proj

CompiledSharedUpSwigluMxf8 = _kernel.Mxf8SwigluProjection
QuantizedWeight = _proj.QuantizedWeight
#: The layer input width, this rank's gate|up rows and the SwiGLU output width.
HIDDEN, GATE_UP, OUT = _kernel.HIDDEN, _kernel.GATE_UP, _kernel.OUT


def prepare_weight(gate_up: torch.Tensor) -> QuantizedWeight:
    """A layer's row-major bf16 gate|up weight `[1024, 6144]` (gate rows first) with its rows
    interleaved in blocks of 16 (the epilogue's pairing), quantized to MXF8. Once at load."""
    return _proj.quantize_weight(_kernel.interleave_gate_up(gate_up.contiguous()))


def compile(device_index: int) -> CompiledSharedUpSwigluMxf8:  # noqa: A001 -- the kernel's name
    """Compile the activation quant and both GEMM tiles for `device_index` and launch each
    once on zeros, so no context step meets a first compile or module load."""
    dev = torch.device("cuda", int(device_index))
    runner = _kernel.Mxf8SwigluProjection(int(device_index))
    w0 = prepare_weight(torch.zeros(GATE_UP, HIDDEN, dtype=torch.bfloat16, device=dev))
    with torch.cuda.device(dev):
        for rows in runner.warm_rows():
            x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
            runner(x, w0, torch.empty(rows, OUT, dtype=torch.bfloat16, device=dev))
        torch.cuda.synchronize(dev)
    runner.choice_hits = {t: 0 for t in runner.choice_hits}
    runner.pre_hits = 0
    return runner


def _deinterleave(gu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(gate, up)` columns of a `[T, 1024]` product on the interleaved weight."""
    rows = int(gu.shape[0])
    blocks = gu.reshape(rows, OUT // _kernel.HALF, 2, _kernel.HALF)
    return blocks[:, :, 0].reshape(rows, OUT), blocks[:, :, 1].reshape(rows, OUT)


class CuteDslSharedUpSwigluMxf8(OpWrapper):
    """`out = silu(gate) * up` of `x @ W^T`, both operands MXF8, fp32 accumulation.

    `x` is the bf16 `[T, 6144]` layer input (unit inner stride, 16-byte aligned); one launch
    quantizes it with the activation quant's rule, or `pre = (xq, sfa)` is that operand as
    its producer (the pre-MoE all-reduce) wrote it and the quant is skipped. `weight` is
    `prepare_weight` of this layer's gate|up. The product is rounded to bf16 as a plain
    bf16-out GEMM rounds it, then `bf16(silu(g) * u)` in fp32 with an exact divide: the bytes
    of the `shared_up_mxf8` GEMM into bf16 `[T, 1024]` followed by the rows SwiGLU. `out` is
    a contiguous bf16 `[T, 512]`, written and returned.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The trunk's context rows of the prefill MoE (513..8192): the 256x128 tile up to 4096
    # rows, 256x256 above; both with the quant launch and on the producer's operand.
    CELLS: tuple[Cell, ...] = (
        Cell(why="513 rows, the first prefill-MoE count: the 256x128 tile", spec=dict(rows=513)),
        Cell(why="4096 rows, the small tile's last count", spec=dict(rows=4096)),
        Cell(why="4097 rows, the 256x256 tile's first", spec=dict(rows=4097)),
        Cell(
            why="7310 rows on the pre-MoE all-reduce's MXF8 operand",
            spec=dict(rows=7310, pre=True),
        ),
        Cell(why="max_num_tokens on the producer's operand", spec=dict(rows=8192, pre=True)),
    )

    note = """
    The target quantizes every MoE layer's interleaved gate|up weight once at load and
    takes this op for the trunk's context rows that run the prefill shared expert, on the
    MXF8 operand the pre-MoE two-shot all-reduce writes; the bf16 down projection then
    reads `out`. Elsewhere (no operand, decode rows, the MTP layer) the bf16 shared expert
    runs. The launcher checks shapes, dtypes and alignment on every call and raises.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: QuantizedWeight,
        out: torch.Tensor,
        compiled: CompiledSharedUpSwigluMxf8,
        pre: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return compiled(x, weight, out, pre=pre)

    def reference(
        self,
        x: torch.Tensor,
        weight: QuantizedWeight,
        out: torch.Tensor,
        compiled: CompiledSharedUpSwigluMxf8,
        pre: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The same MXF8 operands multiplied in fp32 and rounded to bf16, then the fp64
        SwiGLU rounded once, as a new tensor; `out` is not written."""
        if pre is not None:
            gu = _proj.emulate(pre[0], weight, HIDDEN, sfa=pre[1])
        else:
            gu = _proj.emulate(x, weight, HIDDEN)
        gate, up = _deinterleave(gu.to(torch.bfloat16))
        g, u = gate.double(), up.double()
        return (g / (1 + torch.exp(-g)) * u).to(torch.bfloat16)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: QuantizedWeight,
        out: torch.Tensor,
        compiled: CompiledSharedUpSwigluMxf8,
        pre: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        rows = int(x.shape[0])
        assert x.dtype == torch.bfloat16 and tuple(x.shape) == (rows, HIDDEN), (
            f"x must be bf16 [T, {HIDDEN}], got {x.dtype} {tuple(x.shape)}"
        )
        assert tuple(weight.q.shape) == (GATE_UP, HIDDEN), (
            f"weight must be the MXF8 [{GATE_UP}, {HIDDEN}] of prepare_weight"
        )
        assert out.dtype == torch.bfloat16 and tuple(out.shape) == (rows, OUT), (
            f"out must be bf16 [{rows}, {OUT}]"
        )
        if pre is not None:
            assert tuple(pre[0].shape) == (rows, HIDDEN) and pre[0].dtype == torch.uint8, (
                f"pre must be the uint8 [{rows}, {HIDDEN}] operand and its scales"
            )


cute_dsl_shared_up_swiglu_mxf8 = CuteDslSharedUpSwigluMxf8()
