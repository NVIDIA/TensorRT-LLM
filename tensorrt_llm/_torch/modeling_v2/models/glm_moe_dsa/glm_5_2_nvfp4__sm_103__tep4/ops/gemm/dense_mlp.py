# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A dense layer's MLP, this rank's partial, as two ops: `dense_gate_up_swiglu` (gate|up and
the SwiGLU) and `dense_down`.

Context rows take the MXF8 trio where it built: the gate|up GEMM with the SwiGLU in its
epilogue writing the down GEMM's MXF8 operand (`cute_dsl_dense_up_swiglu_mxf8`; without it the
MXF8 gate|up then `cute_dsl_swiglu_mxf8`), then the MXF8 down GEMM: no bf16 intermediate. Every
other row count takes the bf16 projections and `cute_dsl_dense_swiglu`. The up op returns
`(act, act8)`: the bf16 SwiGLU output or its MXF8 operand, and the down op takes whichever it
got.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count, try_build
from ..activation import cute_dsl_swiglu_mxf8 as _swiglu8_op
from ..activation.cute_dsl_dense_swiglu import cute_dsl_dense_swiglu
from ..activation.cute_dsl_swiglu_mxf8 import cute_dsl_swiglu_mxf8
from . import cute_dsl_dense_up_swiglu_mxf8 as _dense_up8_op
from . import cute_dsl_projection_mxf8 as _mxf8_op
from .cute_dsl_dense_up_swiglu_mxf8 import cute_dsl_dense_up_swiglu_mxf8
from .cute_dsl_projection_mxf8 import cute_dsl_projection_mxf8
from .projection import ProjectionHandle, mxf8_takes, projection


@dataclass(frozen=True)
class DenseMlpHandle:
    """What the dense MLP's two ops launch: the gate|up and down projections' handles (their
    `mxf8` is the MXF8 trio's GEMMs), the MXF8 gate|up with the SwiGLU epilogue (`fused`),
    the SwiGLU into the down GEMM's MXF8 operand (`swiglu8`), the bf16 SwiGLU (`swiglu`),
    and the route counters. `fused` / `swiglu8` None: not built (with `swiglu8` None the
    layers have no MXF8 weights either and every row count takes the bf16 path)."""

    gate_up: ProjectionHandle
    down: ProjectionHandle
    fused: Any
    swiglu8: Any
    swiglu: Any
    count: Count


def build_mxf8(gate_up: list[torch.Tensor], down: list[torch.Tensor], dev: int):
    """The MXF8 trio for the dense layers' context rows, their weights quantized once here
    (`gate_up` / `down`: each dense layer's row-major bf16 weights). At 7-8k rows 0.82
    against 1.61 ms per context step in bf16.

    Returns None (every dense layer keeps the bf16 path) unless both GEMMs and the SwiGLU into
    MXF8 build; else `(runners, swiglu8, fused, weights)`: the two GEMMs by projection name,
    the SwiGLU, the gate|up GEMM with the SwiGLU in its epilogue (None where it does not
    build, or with TRTLLM_GLM52_CTX_DENSE_SWIGLU_EPI=0), and each layer's `(gate_up8,
    down8)`, plus its interleaved fused weight where `fused` built (+0.11 GB per rank: one
    launch instead of two and no bf16 `[T, 6144]`)."""
    wq = {
        "dense_gate_up": [_mxf8_op.quantize_weight(w) for w in gate_up],
        "dense_down": [_mxf8_op.quantize_weight(w) for w in down],
    }
    runners = {
        name: try_build(f"MXF8 {name}", lambda name=name: _mxf8_op.build(name, wq[name][0]))
        for name in wq
    }
    swiglu8 = try_build("SwiGLU into MXF8", lambda: _swiglu8_op.compile(dev))
    if swiglu8 is None or not all(runners.values()):
        return None
    weights = list(zip(wq["dense_gate_up"], wq["dense_down"]))
    fused = (
        try_build("dense gate|up + SwiGLU MXF8", lambda: _dense_up8_op.compile(dev))
        if os.environ.get("TRTLLM_GLM52_CTX_DENSE_SWIGLU_EPI", "1") == "1"
        else None
    )
    if fused is not None:
        weights = [
            (gu8, dn8, _dense_up8_op.prepare_weight(w)) for (gu8, dn8), w in zip(weights, gate_up)
        ]
    return SimpleNamespace(runners=runners, swiglu8=swiglu8, fused=fused, weights=weights)


def build(
    gate_up: ProjectionHandle,
    down: ProjectionHandle,
    *,
    fused: Any,
    swiglu8: Any,
    swiglu: Any,
    count: Count,
) -> DenseMlpHandle:
    """The dense MLP's handle over the kernels built at load."""
    return DenseMlpHandle(gate_up, down, fused, swiglu8, swiglu, count)


class DenseGateUpSwiglu(OpWrapper):
    """`swiglu(x @ gu)`, a dense layer's MLP up to the down GEMM's operand: `(act, act8)`.

    Where the layer has MXF8 weights (`w8`: `(gate_up8, down8[, fused8])`) and the MXF8 gate|up
    takes the rows (`mxf8_takes`, context rows): the gate|up GEMM with the SwiGLU into the
    down GEMM's MXF8 operand in its epilogue (counted `dense_up_swiglu`), or without that
    kernel the MXF8 gate|up then `cute_dsl_swiglu_mxf8`; `act8` is that operand, `act` None.
    `x8` is `x`'s MXF8 operand from the collective (else gate|up quantizes `x` itself).
    Elsewhere the bf16 gate|up projection and `cute_dsl_dense_swiglu`: `act` bf16, `act8`
    None."""

    def __call__(self, ph, x: torch.Tensor, gu: torch.Tensor, w8, x8, handle: DenseMlpHandle):
        if w8 is not None and mxf8_takes(ph, x, w8[0], None, handle.gate_up):
            rows = int(x.shape[0])
            if handle.fused is not None:
                act8 = cute_dsl_dense_up_swiglu_mxf8(
                    x, w8[2], *_dense_up8_op.operand(rows, x.device), handle.fused, pre=x8
                )
                handle.count("dense_up_swiglu", True)
            else:
                gate_up = projection(ph, x, gu, handle.gate_up, w8=w8[0], pre=x8)
                act8 = cute_dsl_swiglu_mxf8(
                    gate_up,
                    *_swiglu8_op.operand(rows, gate_up.shape[1] // 2, x.device),
                    handle.swiglu8,
                )
            return None, act8
        gate_up = projection(ph, x, gu, handle.gate_up)
        return cute_dsl_dense_swiglu(gate_up, handle.swiglu), None

    def reference(self, ph, x: torch.Tensor, gu: torch.Tensor, w8, x8, handle: DenseMlpHandle):
        """`(silu(gate) * up, None)` of `x @ gu` in fp32 (gate columns first), rounded once to
        bf16. Every argument but `x` and `gu` is ignored."""
        gate_up = x.float() @ gu.float()
        gate, up = gate_up.chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16), None


class DenseDown(OpWrapper):
    """A dense layer's down projection of `dense_gate_up_swiglu`'s output: the MXF8 down GEMM
    on `act8` where the up op wrote it (`w8[1]`, counted `dense_mxf8`), else the bf16 down
    projection of `act`."""

    def __call__(self, ph, act, act8, dn: torch.Tensor, w8, handle: DenseMlpHandle):
        if act8 is not None:
            down = handle.down.mxf8
            rows = int(act8[0].shape[0])
            out = torch.empty([rows, down.n], dtype=torch.bfloat16, device=act8[0].device)
            handle.count("dense_mxf8", True)
            return cute_dsl_projection_mxf8(act8[0], w8[1], out, down, pre=act8)
        return projection(ph, act, dn, handle.down)

    def reference(self, ph, act, act8, dn: torch.Tensor, w8, handle: DenseMlpHandle):
        """`act @ dn` in fp32, rounded once to bf16 (the bf16 operand; with `act8` its
        dequantized values). Every other argument is ignored."""
        return (act.float() @ dn.float()).to(torch.bfloat16)


dense_gate_up_swiglu = DenseGateUpSwiglu()
dense_down = DenseDown()
