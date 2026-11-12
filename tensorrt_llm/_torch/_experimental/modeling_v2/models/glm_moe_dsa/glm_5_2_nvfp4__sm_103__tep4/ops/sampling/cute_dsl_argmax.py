# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Greedy token pick: row-wise argmax over GLM-5.2's full fp32 vocabulary, one CuTe DSL launch."""

from __future__ import annotations

from typing import Callable

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.sampling import sampler_argmax_v154880_fp32 as _kernel

VOCAB = _kernel.VOCAB


def compile() -> Callable:  # noqa: A001 -- the kernel module's own name for this step
    """Compile the kernel; return the handle the op takes as `compiled`. Call once at load.
    The row count is symbolic: one handle serves every call."""
    return _kernel.compile()


def takes(logits: torch.Tensor) -> bool:
    """Whether the op takes `logits`: contiguous, 16-byte aligned fp32 `[M >= 1, 154880]`.
    Anything else is `torch.argmax`'s."""
    return (
        logits.is_cuda
        and logits.dtype is torch.float32
        and logits.ndim == 2
        and logits.shape[0] >= 1
        and logits.shape[1] == VOCAB
        and logits.is_contiguous()
        and logits.data_ptr() % 16 == 0
    )


class CuteDslArgmax(OpWrapper):
    """`tokens[r] = argmax(logits[r])`, written into the caller's int32 `tokens [M]`.

    `torch.argmax` semantics: the first index of the maximum wins, `-0.0` equals
    `+0.0`, and any NaN is the maximum (the first NaN of a row wins). Every value
    is mapped to an order-preserving int32 key, so the result does not depend on
    the reduction order. One 8-CTA cluster per row. Returns `tokens`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The draft step's pick (one row per request) and the target's all-greedy
    # accept (MTP 2 rows per request) of a 32-request batch, plus the semantics
    # cells argmax is easy to get subtly wrong on.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one request", spec=dict(rows=1, case="random")),
        Cell(why="the draft pick of a full batch, 32 requests", spec=dict(rows=32, case="random")),
        Cell(why="the target accept of 32 requests x MTP 2", spec=dict(rows=64, case="random")),
        Cell(
            why="ties across the cluster's segments: the first index must win",
            spec=dict(rows=5, case="ties"),
        ),
        Cell(why="-0.0 and +0.0 compare equal", spec=dict(rows=3, case="signed_zero")),
        Cell(why="NaN is the maximum, the first NaN wins", spec=dict(rows=3, case="nan")),
        Cell(why="+inf and -inf rows", spec=dict(rows=4, case="inf")),
    )

    note = """
    The target compiles one handle at load and takes this op for the spec
    worker's all-greedy picks wherever `takes` holds; every other call keeps the
    worker's own sampling. The kernel checks the operands on the host and
    raises.
    """

    def __call__(
        self, logits: torch.Tensor, tokens: torch.Tensor, compiled: Callable
    ) -> torch.Tensor:
        return _kernel.run(compiled, logits, tokens)

    def reference(
        self, logits: torch.Tensor, tokens: torch.Tensor, compiled: Callable
    ) -> torch.Tensor:
        """`torch.argmax` as int32, a new tensor. `tokens` and `compiled` are ignored."""
        return torch.argmax(logits, dim=-1).to(torch.int32)


cute_dsl_argmax = CuteDslArgmax()
