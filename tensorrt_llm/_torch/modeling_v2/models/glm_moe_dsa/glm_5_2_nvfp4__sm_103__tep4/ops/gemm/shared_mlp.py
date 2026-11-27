# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared expert, this rank's slice, as two ops: `shared_gate_up_swiglu` (gate|up and the
SwiGLU) and `shared_down`.

A trunk layer's context rows take the prefill pair where it built: the gate|up GEMM with the
SwiGLU in its epilogue (on MXF8 operands where the pre-MoE collective wrote `x`'s and the
layer's MXF8 gate|up built, `cute_dsl_shared_up_swiglu_mxf8`; else bf16) into one buffer, then
the prefill down GEMM. Decode rows take the fused gate|up + SwiGLU cell at their row count and
the down projection; a `Phase.wide` step's rows above 64 the MTP decode cells (at 128 / 192,
else the next one up). What no cell takes runs the gate|up projection, `swiglu_rows` and the
down projection. The up op returns `(act, prefill)`: the SwiGLU output and whether it is the
prefill pair's, whose down GEMM then reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count, try_build
from ..activation.cute_dsl_swiglu_rows import SwigluRowsHandle, swiglu_rows
from . import cute_dsl_projection as _projection_op
from . import cute_dsl_shared_gate_up_swiglu as _shared_gu_swiglu_op
from . import cute_dsl_shared_up_swiglu_mxf8 as _shared_up8_op
from .cute_dsl_projection import cute_dsl_projection
from .cute_dsl_shared_gate_up_swiglu import cute_dsl_shared_gate_up_swiglu
from .cute_dsl_shared_up_swiglu_mxf8 import cute_dsl_shared_up_swiglu_mxf8
from .projection import DECODE_MAX_ROWS, ProjectionHandle, cell_up, projection, wide_cells

#: The prefill pair's row counts: the context steps above the decode cells, up to the token
#: budget.
PREFILL_MIN_ROWS = DECODE_MAX_ROWS + 1
PREFILL_MAX_ROWS = 8192


@dataclass(frozen=True)
class SharedMlpHandle:
    """What the shared expert's two ops launch.

    `cells`: the fused gate|up + SwiGLU cells by row count; `mtp_cells`: the MTP decode cells
    (128 / 192, `Phase.wide` only) and `wide` the row counts between them on the next one up
    (empty with buckets off); `up_cells`: an MTP3 / MTP5 engine's small-batch decode counts
    (`cell_up`). `up_pre` / `down_pre`: the prefill pair, both or neither; `mid` its SwiGLU
    buffer `[8192, inter]` (one is enough: up and down run back to back on the shared
    branch's stream, which forks again only after the layer's reduce); `mxf8` the MXF8
    gate|up of the pair. `gate_up` / `down` / `swiglu`: the two-launch path's handles."""

    cells: Mapping[int, Any]
    mtp_cells: Mapping[int, Any]
    wide: Mapping[int, Any]
    wide_bucket: bool
    up_cells: Mapping[int, Any]
    up_pre: Any
    down_pre: Any
    mid: torch.Tensor | None
    mxf8: Any
    gate_up: ProjectionHandle
    down: ProjectionHandle
    swiglu: SwigluRowsHandle
    count: Count


def build_prefill(gu_il: torch.Tensor, down: torch.Tensor) -> tuple[Any, Any]:
    """The prefill pair `(up, down)`, both or `(None, None)` with a warning: up writes the
    SwiGLU output down reads. Built against one MoE layer's weights (`gu_il` interleaved,
    `down` row-major); each call passes its own."""
    pair = try_build(
        "shared expert prefill pair",
        lambda: (
            _shared_gu_swiglu_op.build_prefill(gu_il),
            _projection_op.build_shared_down_prefill(down),
        ),
    )
    return pair if pair is not None else (None, None)


def build_mxf8(dev: int):
    """The prefill pair's MXF8 gate|up with the SwiGLU in its epilogue, or None with a
    warning (the pair's bf16 gate|up then serves). Its down stays bf16: the MXF8 down moved
    the output distribution more than the accuracy budget allows."""
    return try_build("shared gate|up MXF8", lambda: _shared_up8_op.compile(dev))


def build(
    *,
    cells: Mapping[int, Any],
    mtp_cells: Mapping[int, Any],
    up_cells: Mapping[int, Any],
    up_pre: Any,
    down_pre: Any,
    mid: torch.Tensor | None,
    mxf8: Any,
    gate_up: ProjectionHandle,
    down: ProjectionHandle,
    swiglu: SwigluRowsHandle,
    wide_bucket: bool,
    count: Count,
) -> SharedMlpHandle:
    """The shared expert's handle over the kernels built at load; maps each wide decode row
    count without an MTP cell onto the next one up (its split-K kernel takes any row count)."""
    return SharedMlpHandle(
        cells=dict(cells),
        mtp_cells=dict(mtp_cells),
        wide=wide_cells(mtp_cells) if wide_bucket else {},
        wide_bucket=wide_bucket,
        up_cells=dict(up_cells),
        up_pre=up_pre,
        down_pre=down_pre,
        mid=mid,
        mxf8=mxf8,
        gate_up=gate_up,
        down=down,
        swiglu=swiglu,
        count=count,
    )


def _direct_ok(t: torch.Tensor) -> bool:
    """Read in place rather than staged: a staging copy is a memcpy node, which a PDL edge
    cannot pass through, so it is skipped for a 128-byte aligned contiguous bf16 tensor -- the
    collective's output, whose graph-pool address is stable."""
    return t.dtype is torch.bfloat16 and t.is_contiguous() and t.data_ptr() % 128 == 0


class SharedGateUpSwiglu(OpWrapper):
    """`swiglu(x @ gu)` of the shared expert: `(act, prefill)`.

    `gu_il` is the layer's interleaved gate|up weight. A `Phase.wide` step's rows above 64:
    the MTP decode cell at 128 / 192, else the next one up (counted `dec_wide`). A trunk
    layer's (`trunk`) other rows in the prefill pair's range, 128-byte aligned: the pair's
    gate|up into the handle's `mid` (`prefill` True), on MXF8 operands with `x8` (`x`'s MXF8
    operand from the pre-MoE collective) and `w8` (the layer's MXF8 gate|up), counted
    `shared_mxf8`. Otherwise the fused cell at the row count, or for rows up to 64 without one
    where `cell_up` holds the one built at that count (`dec_cell_up`); else the gate|up
    projection then `swiglu_rows`."""

    def __call__(
        self,
        ph,
        x: torch.Tensor,
        gu: torch.Tensor,
        gu_il: torch.Tensor | None,
        w8,
        x8,
        handle: SharedMlpHandle,
        trunk: bool = False,
    ):
        rows = int(x.shape[0])
        wide = None
        if ph.wide and rows > DECODE_MAX_ROWS:
            wide = handle.mtp_cells.get(rows)
            if wide is None and handle.wide_bucket:
                wide = handle.wide.get(rows)
                handle.count("dec_wide_rows", wide is not None)
            handle.count("dec_wide", wide is not None)
        if (
            trunk
            and wide is None
            and handle.down_pre is not None
            and PREFILL_MIN_ROWS <= rows <= PREFILL_MAX_ROWS
        ):
            xs = x if _direct_ok(x) else x.contiguous()
            # The prefill kernels take 128-byte aligned rows; a contiguous view that is not
            # takes the two-launch path.
            if xs.dtype is torch.bfloat16 and xs.data_ptr() % 128 == 0:
                mid = handle.mid[:rows]
                if x8 is not None and w8 is not None:
                    cute_dsl_shared_up_swiglu_mxf8(xs, w8, mid, handle.mxf8, pre=x8)
                    handle.count("shared_mxf8", True)
                else:
                    cute_dsl_shared_gate_up_swiglu(xs, gu_il, mid, handle.up_pre)
                return mid, True
        runner = handle.cells.get(rows) if wide is None else wide
        if runner is None and handle.up_cells and cell_up(ph, rows):
            # An MTP3 / MTP5 decode count of 1..3 sequences: its own fused cell.
            runner = handle.up_cells.get(rows)
            handle.count("dec_cell_up", runner is not None)
        if runner is None:
            gate_up = projection(ph, x, gu, handle.gate_up)
            return swiglu_rows(gate_up, handle.swiglu), False
        act = torch.empty([rows, gu.shape[1] // 2], dtype=x.dtype, device=x.device)
        cute_dsl_shared_gate_up_swiglu(x, gu_il, act, runner)
        return act, False

    def reference(
        self,
        ph,
        x: torch.Tensor,
        gu: torch.Tensor,
        gu_il: torch.Tensor | None,
        w8,
        x8,
        handle: SharedMlpHandle,
        trunk: bool = False,
    ):
        """`(silu(gate) * up, False)` of `x @ gu` in fp32 (gate columns first), rounded once
        to bf16. Every argument but `x` and `gu` is ignored."""
        gate_up = x.float() @ gu.float()
        gate, up = gate_up.chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16), False


class SharedDown(OpWrapper):
    """The shared expert's down projection of `shared_gate_up_swiglu`'s `act`, a partial sum:
    the prefill pair's down GEMM where the up op ran the pair (`prefill`), else the down
    projection. `out` is the routed experts' output buffer when FC2 will accumulate onto it
    (else a new tensor)."""

    def __call__(
        self,
        ph,
        act: torch.Tensor,
        dn: torch.Tensor,
        handle: SharedMlpHandle,
        out: torch.Tensor | None = None,
        prefill: bool = False,
    ) -> torch.Tensor:
        if prefill:
            if out is None:
                out = torch.empty(
                    (int(act.shape[0]), dn.shape[1]), dtype=torch.bfloat16, device=act.device
                )
            return cute_dsl_projection(act, dn.t(), out, handle.down_pre)
        return projection(ph, act, dn, handle.down, out=out)

    def reference(
        self,
        ph,
        act: torch.Tensor,
        dn: torch.Tensor,
        handle: SharedMlpHandle,
        out: torch.Tensor | None = None,
        prefill: bool = False,
    ) -> torch.Tensor:
        """`act @ dn` in fp32, rounded once to bf16. Every other argument is ignored."""
        return (act.float() @ dn.float()).to(torch.bfloat16)


shared_gate_up_swiglu = SharedGateUpSwiglu()
shared_down = SharedDown()
