# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`dst.copy_(src)` for 2-D row blocks, with dtype conversion, one CuTe DSL launch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.elementwise import tensor_copy_convert_2d as _kernel


@dataclass(frozen=True)
class CompiledCopy:
    """One compiled kernel per `(src dtype, dst dtype)` the op takes."""

    kernels: Mapping[tuple[torch.dtype, torch.dtype], Callable]


#: Same-dtype copies run on raw integers of the element's size; these conversions
#: convert. Everything else is `Tensor.copy_`'s.
PAIRS = tuple((t, t) for t in _kernel._RAW_BY_SIZE.values()) + tuple(
    sorted(_kernel._CONVERSIONS, key=str)
)


def compile() -> CompiledCopy:  # noqa: A001 -- the kernel module's own name for this step
    """Compile every pair in `PAIRS` and launch each once on a 2 x 3 copy, so no step
    meets a first compile or a first module load. Call once at load, before any capture."""
    kernels = {pair: _kernel._compiled(*pair) for pair in PAIRS}
    _kernel.warm()
    return CompiledCopy(kernels)


def takes(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """Whether the op takes this pair: same shape on one CUDA device, a dtype pair in
    `PAIRS`, both 2-D row blocks after collapsing leading dimensions (unit inner stride,
    one row pitch each), and no overlap. Anything else is `Tensor.copy_`'s."""
    return _kernel.plan(dst, src) is not None


class CuteDslCopy(OpWrapper):
    """`dst[r, c] = convert(src[r, c])`. Returns `dst`.

    Same dtype: bitwise, any 1-, 2-, 4- or 8-byte type. bf16 / fp16 -> fp32 exact;
    fp32 -> bf16 / fp16 round-to-nearest-even; int32 -> int64 sign extension;
    int64 -> int32 the low 32 bits: each as `Tensor.copy_` converts. One 256-thread
    CTA per 1024 columns of a row, every warp access coalesced.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The forward's copies: the MoE front end's staging of activations, weights and
    # ids into the slot buffers, a chunk's result into the step's output, and the
    # selection buffers. Rows from 1 to 8192; widths from 8 (top-k) to 6144.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="bf16 activation rows into a slot buffer",
            spec=dict(shape=(64, 6144), src=torch.bfloat16, dst=torch.bfloat16),
        ),
        Cell(
            why="fp32 top-k weights",
            spec=dict(shape=(1037, 8), src=torch.float32, dst=torch.float32),
        ),
        Cell(
            why="int64 ids narrowed to int32",
            spec=dict(shape=(512,), src=torch.int64, dst=torch.int32),
        ),
        Cell(
            why="int32 ids widened to int64",
            spec=dict(shape=(77, 8), src=torch.int32, dst=torch.int64),
        ),
        Cell(
            why="fp32 rounded to bf16",
            spec=dict(shape=(33, 6144), src=torch.float32, dst=torch.bfloat16),
        ),
        Cell(
            why="a strided source and destination (column slices of wider rows)",
            spec=dict(shape=(129, 1030), src=torch.bfloat16, dst=torch.bfloat16, strided=True),
        ),
        Cell(
            why="three dimensions collapsed into rows",
            spec=dict(shape=(4, 16, 2048), src=torch.int32, dst=torch.int32),
        ),
    )

    note = """
    The target compiles every pair at load and takes this op wherever `takes`
    holds; every other pair goes to `Tensor.copy_` at the call site. The op
    raises on a pair outside `takes` rather than fall back.
    """

    def __call__(
        self, dst: torch.Tensor, src: torch.Tensor, compiled: CompiledCopy
    ) -> torch.Tensor:
        plan = _kernel.plan(dst, src)
        if plan is None:
            raise ValueError(
                f"copy of {tuple(src.shape)} {src.dtype} into {tuple(dst.shape)} {dst.dtype}: "
                "not a pair this kernel takes"
            )
        d2, s2, sd, dd = plan
        rows, cols = d2.shape
        col_tiles = -(-int(cols) // _kernel._COLS_PER_CTA)
        compiled.kernels[(sd, dd)](
            s2,
            d2,
            col_tiles,
            int(rows) * col_tiles,
            torch.cuda.current_stream(d2.device).cuda_stream,
        )
        return dst

    def reference(
        self, dst: torch.Tensor, src: torch.Tensor, compiled: CompiledCopy
    ) -> torch.Tensor:
        """`Tensor.copy_` into a clone of `dst`. `compiled` is ignored."""
        return dst.clone().copy_(src)


cute_dsl_copy = CuteDslCopy()
