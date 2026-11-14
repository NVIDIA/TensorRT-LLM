# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dense MLP's SwiGLU written as the MXF8 operand of the down projection that reads it:
the SwiGLU and the activation quant in one CuTe DSL launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.activation import mlp_prefill_swiglu_mxf8 as _kernel
from ...kernels.common import self_attn_prefill_act_quant_mxf8 as _quant

CompiledSwigluMxf8 = _kernel.SwigluFn
sf_bytes = _quant.sf_bytes


def compile(device_index: int) -> CompiledSwigluMxf8:  # noqa: A001 -- the kernel's own name
    """Compile the kernel for `device_index` with the dense SwiGLU's math; return the handle
    the op takes as `compiled`. Call once at load: row count and width are runtime values.
    Launches once on a one-row case, so no step meets a first module load."""
    fn = _kernel.compile_swiglu_mxf8(int(device_index), fast=True)
    dev = torch.device("cuda", int(device_index))
    with torch.cuda.device(dev):
        _kernel.swiglu_mxf8(fn, torch.zeros(1, 256, dtype=torch.bfloat16, device=dev))
        torch.cuda.synchronize(dev)
    return fn


def operand(rows: int, width: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Buffers of a `[rows, width]` MXF8 operand: e4m3 bytes and the blocked e8m0 scales."""
    return (
        torch.empty(rows, width, dtype=torch.uint8, device=device),
        torch.empty(sf_bytes(rows, width), dtype=torch.uint8, device=device),
    )


class CuteDslSwigluMxf8(OpWrapper):
    """`silu(source[:, :W]) * source[:, W:]`, rounded to bf16, written as the MXF8 operand
    `(q [M, W] uint8 e4m3, sf)`: per 32-element block the activation quant's power-of-two
    scale and e4m3 values. Returns `(q, sf)`.

    `source` is the bf16 `[M, 2W]` gate|up output (gate first), W a multiple of 128, unit
    inner stride, a row pitch that is a multiple of 8, 16-byte aligned. The SwiGLU is
    `cute_dsl_dense_swiglu`'s arithmetic (approximate exp2 and reciprocal), so the bytes are
    the activation quant of that op's output. The scale rows past M up to the next multiple
    of 128 get 2^0.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The dense layers' context rows: above the decode buckets up to max_num_tokens. The
    # grid covers 64 rows per CTA; the cells take a partial last CTA, a whole one, and the
    # token budget.
    CELLS: tuple[Cell, ...] = (
        Cell(why="65 rows, the first context count", spec=dict(rows=65, width=3072)),
        Cell(why="7310 rows, a served context step", spec=dict(rows=7310, width=3072)),
        Cell(why="max_num_tokens", spec=dict(rows=8192, width=3072)),
    )

    note = """
    The target takes this op on the dense layers' context rows when their gate|up and down
    run as MXF8 GEMMs (`cute_dsl_projection_mxf8` `dense_gate_up` / `dense_down`): the
    down GEMM reads the operand without its own activation quant. Decode rows keep the
    bf16 MLP. The kernel checks shapes, strides, dtypes and alignment on every call and
    raises.
    """

    def __call__(
        self,
        source: torch.Tensor,
        q: torch.Tensor,
        sf: torch.Tensor,
        compiled: CompiledSwigluMxf8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _kernel.swiglu_mxf8(compiled, source, q, sf)

    def reference(
        self,
        source: torch.Tensor,
        q: torch.Tensor,
        sf: torch.Tensor,
        compiled: CompiledSwigluMxf8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(q, sf)` as new tensors: the fp64 SwiGLU rounded once to bf16, then the activation
        quant's torch twin. The kernel's approximate sigmoid moves a bf16 value by one ulp near
        a rounding midpoint, so a few e4m3 bytes may differ by one code. `q`, `sf` and
        `compiled` are not used."""
        width = int(source.shape[1]) // 2
        gate, up = source.double()[:, :width], source.double()[:, width:]
        act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        return _quant.reference(act.contiguous())

    def is_valid(
        self,
        source: torch.Tensor,
        q: torch.Tensor,
        sf: torch.Tensor,
        compiled: CompiledSwigluMxf8,
    ) -> None:
        rows, width = int(source.shape[0]), int(source.shape[1]) // 2
        assert source.dtype == torch.bfloat16 and source.dim() == 2 and width % 128 == 0, (
            f"source must be bf16 [M, 2W] with W a multiple of 128, got {tuple(source.shape)}"
        )
        assert tuple(q.shape) == (rows, width) and q.dtype == torch.uint8 and q.is_contiguous(), (
            f"q must be a contiguous uint8 [{rows}, {width}]"
        )
        assert sf.dtype == torch.uint8 and sf.numel() >= sf_bytes(rows, width), (
            f"sf must hold at least {sf_bytes(rows, width)} bytes"
        )


cute_dsl_swiglu_mxf8 = CuteDslSwigluMxf8()
