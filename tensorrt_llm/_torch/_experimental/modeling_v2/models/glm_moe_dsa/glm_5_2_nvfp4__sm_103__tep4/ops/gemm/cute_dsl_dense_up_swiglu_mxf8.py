# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A dense layer's gate|up on context rows as one MXF8 block-scaled GEMM whose epilogue writes
the SwiGLU as the down projection's MXF8 operand: no bf16 `[T, 6144]` intermediate, one
launch instead of the GEMM and `cute_dsl_swiglu_mxf8`."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common import self_attn_prefill_act_quant_mxf8 as _quant
from ...kernels.gemm import mlp_prefill_dense_up_swiglu_mxf8 as _kernel
from ...kernels.gemm import self_attn_prefill_projections_mxf8 as _proj

CompiledDenseUpSwigluMxf8 = _kernel.Mxf8DenseSwigluProjection
QuantizedWeight = _proj.QuantizedWeight
#: The layer input width, this rank's gate|up rows and the SwiGLU output width.
HIDDEN, GATE_UP, OUT = _kernel.HIDDEN, _kernel.GATE_UP, _kernel.OUT
sf_bytes = _quant.sf_bytes


def prepare_weight(gate_up: torch.Tensor) -> QuantizedWeight:
    """A dense layer's row-major bf16 gate|up weight `[6144, 6144]` (gate rows first) with its
    rows interleaved in blocks of 16 (the epilogue's pairing), quantized to MXF8. Once at
    load."""
    return _proj.quantize_weight(_kernel.interleave_gate_up(gate_up.contiguous()))


def operand(rows: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Buffers of the `[rows, 3072]` MXF8 output: e4m3 bytes and the blocked e8m0 scales."""
    return (
        torch.empty(rows, OUT, dtype=torch.uint8, device=device),
        torch.empty(sf_bytes(rows, OUT), dtype=torch.uint8, device=device),
    )


def compile(device_index: int) -> CompiledDenseUpSwigluMxf8:  # noqa: A001 -- the kernel's name
    """Compile the activation quant and both GEMM tiles for `device_index` and launch each
    once on zeros, so no context step meets a first compile or module load."""
    dev = torch.device("cuda", int(device_index))
    runner = _kernel.Mxf8DenseSwigluProjection(int(device_index))
    w0 = prepare_weight(torch.zeros(GATE_UP, HIDDEN, dtype=torch.bfloat16, device=dev))
    with torch.cuda.device(dev):
        for rows in runner.warm_rows():
            x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
            runner(x, w0, *operand(rows, dev))
        torch.cuda.synchronize(dev)
    runner.choice_hits = {t: 0 for t in runner.choice_hits}
    runner.pre_hits = 0
    return runner


def _deinterleave(gu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(gate, up)` columns of a `[T, 6144]` product on the interleaved weight."""
    rows = int(gu.shape[0])
    blocks = gu.reshape(rows, OUT // _kernel.HALF, 2, _kernel.HALF)
    return blocks[:, :, 0].reshape(rows, OUT), blocks[:, :, 1].reshape(rows, OUT)


class CuteDslDenseUpSwigluMxf8(OpWrapper):
    """`(q, sf)` = the MXF8 operand of `silu(gate) * up` of `x @ W^T`, both operands MXF8,
    fp32 accumulation.

    `x` is the bf16 `[T, 6144]` layer input (unit inner stride, 16-byte aligned); one launch
    quantizes it with the activation quant's rule, or `pre = (xq, sfa)` is that operand as its
    producer (the two-shot all-reduce) wrote it and the quant is skipped. `weight` is
    `prepare_weight` of this layer's gate|up. The product is rounded to bf16 as a plain
    bf16-out GEMM rounds it, then `cute_dsl_swiglu_mxf8`'s SwiGLU (approximate exp2 and
    reciprocal) and activation quant: the bytes of the `dense_gate_up_mxf8` GEMM into bf16
    `[T, 6144]` followed by `cute_dsl_swiglu_mxf8`. `q` is uint8 `[T, 3072]`, `sf` at least
    `sf_bytes(T, 3072)` bytes; both written and returned. Scale rows past T up to the next
    multiple of 128 get 2^0.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The dense layers' context rows: the 256x128 tile up to 1024 rows, 256x256 above, with
    # the quant launch and on the producer's operand.
    CELLS: tuple[Cell, ...] = (
        Cell(why="65 rows, the first context count: the 256x128 tile", spec=dict(rows=65)),
        Cell(why="1024 rows, the small tile's last count", spec=dict(rows=1024)),
        Cell(why="1025 rows, the 256x256 tile's first", spec=dict(rows=1025)),
        Cell(why="7310 rows on the all-reduce's MXF8 operand", spec=dict(rows=7310, pre=True)),
        Cell(why="max_num_tokens on the producer's operand", spec=dict(rows=8192, pre=True)),
    )

    note = """
    The target quantizes each dense layer's interleaved gate|up weight once at load and takes
    this op for the dense layers' context rows where the MXF8 dense MLP runs; the MXF8 down
    projection then reads `(q, sf)`. Decode rows keep the bf16 GEMMs and SwiGLU. The launcher
    checks shapes, dtypes and alignment on every call and raises.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: QuantizedWeight,
        q: torch.Tensor,
        sf: torch.Tensor,
        compiled: CompiledDenseUpSwigluMxf8,
        pre: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compiled(x, weight, q, sf, pre=pre)

    def reference(
        self,
        x: torch.Tensor,
        weight: QuantizedWeight,
        q: torch.Tensor,
        sf: torch.Tensor,
        compiled: CompiledDenseUpSwigluMxf8,
        pre: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The same MXF8 operands multiplied in fp32 and rounded to bf16, the fp64 SwiGLU
        rounded once to bf16, then the activation quant's torch twin, as new tensors; `q` and
        `sf` are not written. The kernel's approximate sigmoid moves a rare bf16 value by one
        ulp, which moves at most that element's e4m3 code."""
        if pre is not None:
            gu = _proj.emulate(pre[0], weight, HIDDEN, sfa=pre[1])
        else:
            gu = _proj.emulate(x, weight, HIDDEN)
        gate, up = _deinterleave(gu.to(torch.bfloat16))
        act = (torch.nn.functional.silu(gate.double()) * up.double()).to(torch.bfloat16)
        return _quant.reference(act.contiguous())

    def is_valid(
        self,
        x: torch.Tensor,
        weight: QuantizedWeight,
        q: torch.Tensor,
        sf: torch.Tensor,
        compiled: CompiledDenseUpSwigluMxf8,
        pre: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        rows = int(x.shape[0])
        assert x.dtype == torch.bfloat16 and tuple(x.shape) == (rows, HIDDEN), (
            f"x must be bf16 [T, {HIDDEN}], got {x.dtype} {tuple(x.shape)}"
        )
        assert tuple(weight.q.shape) == (GATE_UP, HIDDEN), (
            f"weight must be the MXF8 [{GATE_UP}, {HIDDEN}] of prepare_weight"
        )
        assert q.dtype == torch.uint8 and tuple(q.shape) == (rows, OUT), (
            f"q must be uint8 [{rows}, {OUT}]"
        )
        assert sf.dtype == torch.uint8 and sf.numel() >= sf_bytes(rows, OUT), (
            f"sf must hold sf_bytes({rows}, {OUT}) bytes"
        )
        if pre is not None:
            assert tuple(pre[0].shape) == (rows, HIDDEN) and pre[0].dtype == torch.uint8, (
                f"pre must be the uint8 [{rows}, {HIDDEN}] operand and its scales"
            )


cute_dsl_dense_up_swiglu_mxf8 = CuteDslDenseUpSwigluMxf8()
