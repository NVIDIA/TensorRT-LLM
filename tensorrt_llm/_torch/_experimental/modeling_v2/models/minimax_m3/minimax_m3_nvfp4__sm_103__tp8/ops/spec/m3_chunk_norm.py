# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Eagle3 fc_norm over the captured layers in one launch: per-chunk RMSNorm of the concatenated hidden states
(``m3_spec_tail.chunk_norm``)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Largest hidden size the kernel's register tile holds (4 x 256 threads x 8 bf16).
MAX_HIDDEN = 8192


def rmsnorm_rows(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """bf16(x * rsqrt(mean(x^2) + eps) * w) per row, fp32 in between ((x * r) * w, as the kernels multiply)."""
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * r * weight.float()).to(torch.bfloat16)


class M3ChunkNorm(OpWrapper):
    """``dst[r, c*H:(c+1)*H] = rmsnorm(src[r, c*H:(c+1)*H]) * w[c]`` for the C chunks of ``w [C, H]``; ``src`` may
    have a row stride wider than C * H (the capture buffer). Writes ``dst [M, C*H]`` and returns it."""

    ARCHS = frozenset({Arch.SM_103})

    # The Eagle3 capture buffer of the target (3 captured layers x hidden 6144, buffer rows wider than the layers
    # when the final hidden is also captured), over the verify rows of draft_len 3: 4 / 8 / 32 rows.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request's 4 verify rows, dense [4, 3 x 6144]",
            spec=dict(m=4, c=3, h=6144, extra_cols=0),
        ),
        Cell(
            why="cap 8 (32 rows) read from the capture buffer with a fourth (final-hidden) slot in the row stride",
            spec=dict(m=32, c=3, h=6144, extra_cols=6144),
        ),
    )

    note = """
    fp32 statistic per chunk (block sum, rsqrtf), one bf16 rounding: an output moves by one bf16 ulp now and then
    against torch's statistic; the cells' band states it.
    """

    def __call__(
        self, src: torch.Tensor, w: torch.Tensor, eps: float, dst: torch.Tensor, pdl: bool = True
    ):
        torch.ops.trtllm.minimax_m3_chunk_norm(src, w, eps, dst, pdl)
        return dst

    def reference(
        self, src: torch.Tensor, w: torch.Tensor, eps: float, dst: torch.Tensor, pdl: bool = True
    ):
        """Each chunk normalized on its own; ``dst`` only says where the op writes, ``pdl`` shapes the launch."""
        c, h = w.shape
        x = src[:, : c * h].reshape(src.shape[0], c, h)
        return rmsnorm_rows(x, w, eps).reshape(src.shape[0], c * h)

    def is_valid(
        self, src: torch.Tensor, w: torch.Tensor, eps: float, dst: torch.Tensor, pdl: bool = True
    ) -> None:
        c, h = w.shape
        assert w.is_contiguous(), "w must be dense [C, H] (chunk c read at c * H)"
        assert h % 8 == 0 and h <= MAX_HIDDEN, "H must be a multiple of 8, at most 8192"


m3_chunk_norm = M3ChunkNorm()
