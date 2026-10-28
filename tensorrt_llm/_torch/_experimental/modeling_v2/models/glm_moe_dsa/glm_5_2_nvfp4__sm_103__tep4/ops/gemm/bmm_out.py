# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched matmul writing into a caller-owned output, via the trtllm bmm_out op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class BmmOut(OpWrapper):
    """`out[i] = a[i] @ b[i]` for every batch index, in place, returns None.

    The point of the op over `torch.bmm` is the caller-owned `out`: both
    GLM call sites pass a transposed view of a buffer the rest of the forward
    already holds, so the product lands where it is needed with no copy. That
    is also the hazard -- see `is_valid`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # GLM-5.2's absorbed MLA at TP4 calls it twice per layer, batch = the 16
    # heads per rank. K-B absorbs the 192-wide nope half of q into the 512-wide
    # latent; V-B projects the latent attention output to the 256-wide v. Both
    # read and write transposed views of token-major buffers: q is
    # `[M, 16, 256]` (nope | rope), the latent query `[M, 16, 576]`, the
    # attention output `[M, 16, 512]` and v `[M, 16, 256]`. `a_row` / `out_row`
    # are those row widths. The target's own kernels take M in
    # {2, 4, 8, 16, 32, 48, 64}; every other M, and every prefill chunk, falls
    # back to this op.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="K-B decode, M=1: [16, 1, 192] @ [16, 192, 512] into a latent-q view",
            spec=dict(batch=16, m=1, k=192, n=512, a_row=256, out_row=576, dtype=torch.bfloat16),
        ),
        Cell(
            why="K-B at the full decode batch, 32 x MTP 2 = 64 rows",
            spec=dict(batch=16, m=64, k=192, n=512, a_row=256, out_row=576, dtype=torch.bfloat16),
        ),
        Cell(
            why="K-B at a row count the target kernels refuse, so it lands here",
            spec=dict(batch=16, m=37, k=192, n=512, a_row=256, out_row=576, dtype=torch.bfloat16),
        ),
        Cell(
            why="K-B prefill chunk",
            spec=dict(batch=16, m=2048, k=192, n=512, a_row=256, out_row=576, dtype=torch.bfloat16),
        ),
        Cell(
            why="V-B decode, M=1: [16, 1, 512] @ [16, 512, 256] into a v view",
            spec=dict(batch=16, m=1, k=512, n=256, a_row=512, out_row=256, dtype=torch.bfloat16),
        ),
        Cell(
            why="V-B at the full decode batch, 64 rows",
            spec=dict(batch=16, m=64, k=512, n=256, a_row=512, out_row=256, dtype=torch.bfloat16),
        ),
        Cell(
            why="V-B prefill chunk",
            spec=dict(batch=16, m=2048, k=512, n=256, a_row=512, out_row=256, dtype=torch.bfloat16),
        ),
        Cell(
            why="contiguous operands and out, to separate the op's own correctness "
            "from the strided-view path the target uses",
            spec=dict(batch=16, m=64, k=192, n=512, a_row=None, out_row=None, dtype=torch.bfloat16),
        ),
    )

    note = """
    `out` is written, not accumulated into: its prior contents are ignored.
    """

    def __call__(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        torch.ops.trtllm.bmm_out(a, b, out)
        return out

    def reference(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The product the op writes into `out`, fp32-accumulated.

        `out` is named to mirror `__call__`: the op takes a destination,
        the reference returns a value, and hiding the argument would hide that
        the op mutates one of its inputs.

        TF32 off: torch's default fp32 matmul on this hardware is a TF32
        product, ~1e-3 relative, which would make the reference the inaccurate
        side of the comparison.
        """
        with true_fp32_matmul():
            return (a.float() @ b.float()).to(a.dtype)

    def is_valid(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        # A wrong-shaped `out` is silently resized and re-allocated by the op,
        # detaching it from whatever buffer the caller aliased it with -- the
        # product then lands somewhere nobody reads, and the forward continues
        # with stale data rather than failing.
        assert a.dim() == 3 and b.dim() == 3 and out.dim() == 3, "a, b, out must be 3D"
        assert out.shape == (a.shape[0], a.shape[1], b.shape[2]), (
            "out must be [B, M, N] matching a [B, M, K] and b [B, K, N]"
        )
        # Mixed input dtypes are type-promoted instead of rejected, changing the
        # required out dtype; the catalog exposes only the single-dtype form.
        assert a.dtype == b.dtype == out.dtype, "a, b, out must share one dtype"


bmm_out = BmmOut()
