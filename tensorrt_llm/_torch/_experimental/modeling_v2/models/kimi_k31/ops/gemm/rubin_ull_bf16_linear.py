# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's plain BF16 projection: ``torch.nn.functional.linear`` without a bias (cuBLASLt)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _cell(why: str, rows: int, n: int, k: int, bank: tuple[int, int] | None = None) -> Cell:
    """``bank``: ``weight`` is rows ``[offset, offset + n)`` of a ``[bank_rows, k]`` bank."""
    return Cell(why=why, spec=dict(rows=rows, n=n, k=k, bank=bank))


class Bf16Linear(OpWrapper):
    """``x @ weight.T``: BF16 ``x`` [..., K] by the stored BF16 ``weight`` [N, K] (a row slice of a
    bank is fine), FP32 accumulation rounded once to BF16. Returns a new [..., N] tensor."""

    ARCHS = frozenset({Arch.SM_103})

    # The ULL GEN step's shapes: verify rows 8 / 16 / 32 (1..4 requests x next_n 8), q1 decode and
    # the drafter's rows; N and K are this rank's (TP16 / Helix CP16) shards.
    CELLS: tuple[Cell, ...] = (
        _cell("LM head vocabulary shard 163840 / 16, q1 decode row", 1, 10240, 7168),
        _cell("LM head vocabulary shard, b1 verify (8 rows)", 8, 10240, 7168),
        _cell("LM head vocabulary shard, b4 verify (32 rows)", 32, 10240, 7168),
        _cell("L0 KDA o-projection (6 local heads x 128 -> 7168), b1 verify", 8, 7168, 768),
        _cell("L0 KDA o-projection, b2 verify (16 rows)", 16, 7168, 768),
        _cell("L0 KDA o-projection, b4 verify (32 rows)", 32, 7168, 768),
        _cell("DSA o-projection (96 heads x 128 -> 7168), b1 verify", 8, 7168, 12288),
        _cell("DSA o-projection, b4 verify (32 rows)", 32, 7168, 12288),
        _cell("DSA attention Q rows of the packed q_b bank (96 x 192)", 8, 18432, 1536, (22528, 0)),
        _cell(
            "DSA indexer Q rows of the packed q_b bank (32 x 128)", 32, 4096, 1536, (22528, 18432)
        ),
        _cell("whole packed q_b bank (attention + indexer rows), b2 verify", 16, 22528, 1536),
        _cell("L0 dense gate/up TP16 slice, past the split-K kernels' 32 rows", 64, 4224, 7168),
        _cell("L0 dense down TP16 slice (I / 16 = 2112), past 32 rows", 64, 7168, 2112),
        _cell("drafter FC input shard, odd row count", 7, 7168, 1792),
    )

    note = """
    The cuBLASLt GEMM torch picks for the shape (no bias, no epilogue), on the current stream; the
    algorithm is cuBLAS's heuristic, so the summation order is not pinned across library versions.
    Every input is accepted: a non-BF16 operand computes in its own dtype, not a wrong answer, so
    there is nothing for `is_valid` to refuse. Graph-capturable (cuBLAS's workspace is torch's).
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight)

    def reference(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """The FP64 product, rounded once to ``x``'s dtype."""
        return (x.double() @ weight.double().t()).to(x.dtype)


bf16_linear = Bf16Linear()
