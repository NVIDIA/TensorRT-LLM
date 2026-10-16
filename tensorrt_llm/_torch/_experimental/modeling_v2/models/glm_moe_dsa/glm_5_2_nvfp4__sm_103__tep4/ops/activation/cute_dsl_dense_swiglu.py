# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dense MLP's SwiGLU at GLM-5.2's TP4 width, in one CuTe DSL launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.activation import mlp_dense_swiglu_n3072_bf16 as _kernel


def compile() -> _kernel.CompiledDispatcher:
    """Compile the kernel's four column schedules. Call once, at load.

    Returns the dispatcher the op takes as `compiled`: it picks a precompiled
    schedule from the runtime row count, so one handle serves every M. Costs
    four CuTe DSL compiles the first time; the kernel module caches the result.
    """
    return _kernel.compile()


class CuteDslDenseSwiglu(OpWrapper):
    """`silu(source[:, :3072]) * source[:, 3072:]` as a new `[M, 3072]` tensor.

    `source` is the contiguous `[M, 6144]` bf16 output of the dense gate/up
    projection: gate in the first half, up in the second. That split is the
    layout the projection weight was loaded in, and nothing in the tensor shows
    it; a weight with the halves swapped computes `silu(up) * gate` without
    error.
    """

    ARCHS = frozenset({Arch.SM_103})

    # GLM-5.2 layers 0-2 call this at the step's full row count: decode up to
    # 64 rows, prefill chunks up to 8192. The handle dispatches on M among four
    # column schedules (M <= 24, 25..74 and 149..184, 75..97, 98..148 and > 184),
    # so the cells pick one row count in each.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, one token: the 512-column schedule", spec=dict(rows=1)),
        Cell(
            why="the full decode batch, 64 rows: the 1536-column schedule",
            spec=dict(rows=64),
        ),
        Cell(why="a small mixed batch, 96 rows: the 1024-column schedule", spec=dict(rows=96)),
        Cell(why="max_num_tokens: the 3072-column schedule", spec=dict(rows=8192)),
    )

    note = """
    The target compiles one handle at load and runs it at every row count; there
    is no stock-op fallback at the call site. Quantized output and a SwiGLU
    limit are not supported by the kernel.
    """

    def __call__(self, source: torch.Tensor, compiled: _kernel.CompiledDispatcher) -> torch.Tensor:
        out = torch.empty(
            [source.shape[0], _kernel.OUTPUT_FEATURES], dtype=source.dtype, device=source.device
        )
        _kernel.run(compiled, source, out)
        return out

    def reference(self, source: torch.Tensor, compiled: _kernel.CompiledDispatcher) -> torch.Tensor:
        """fp64 `silu(gate) * up`, rounded once. `compiled` is ignored."""
        gate, up = source.double().chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate) * up).to(source.dtype)


cute_dsl_dense_swiglu = CuteDslDenseSwiglu()
