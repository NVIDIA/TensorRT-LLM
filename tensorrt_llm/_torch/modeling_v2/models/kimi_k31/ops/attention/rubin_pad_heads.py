# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Head padding of the sparse MLA calls at 128 heads (pad128): fill a padded head buffer (Triton).

TRTLLM-Gen's sparse MLA tiles a head group of more than 64 heads in Q64 tiles, which must divide the
group, so the 96 heads per rank of dep16 (attention DP) and helix8 (Helix CP) run padded to 128
heads with zero queries. Their context and generation calls fill the padded ``q_pe`` and query
buffers with this entry.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention.rubin_pad_heads import _pad_heads


class RubinPadHeads(OpWrapper):
    """Fill ``dst`` ``[n, pad, D]`` (contiguous): rows ``h < heads`` from ``src`` ``[n, >= heads,
    D]`` (last dim contiguous, ``dst``'s dtype) and rows ``h >= heads`` with zeros; with ``src``
    None only the rows ``h >= heads`` are zeroed. ``D`` in bytes must be a power of two. Returns
    ``dst``."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 prefill q_pe: 96 -> 128 heads of 64 BF16 rope columns, a 16k chunk",
            spec=dict(n=16384, pad=128, heads=96, d=64, dtype=torch.bfloat16, src=True),
        ),
        Cell(
            why="DEP16 prefill query: zero the padding heads as 64-column rows (576 x 128 / 64)",
            spec=dict(n=4096, pad=1152, heads=864, d=64, dtype=torch.bfloat16, src=False),
        ),
        Cell(
            why="one token",
            spec=dict(n=1, pad=128, heads=96, d=64, dtype=torch.bfloat16, src=True),
        ),
        Cell(
            why="DEP16 / helix8 generation q_pe: 16 requests x 8 verify rows, 96 -> 128 heads",
            spec=dict(n=128, pad=128, heads=96, d=64, dtype=torch.bfloat16, src=True),
        ),
        Cell(
            why="DEP16 / helix8 generation query: zero the padding heads of 128 verify rows",
            spec=dict(n=128, pad=1152, heads=864, d=64, dtype=torch.bfloat16, src=False),
        ),
    )
    note = """
    Two launches cover the pad128 buffers of a sparse MLA call (context or generation): ``q_pe``
    copied and padded, and the query's padding heads zeroed (the absorbed BMM writes the real
    heads' latent columns; the context attention op writes their RoPE columns, the generation RoPE
    kernel only into its FP8 copy of the query). Exact copies and zeros.
    """

    def __call__(self, dst: torch.Tensor, src: Optional[torch.Tensor], heads: int) -> torch.Tensor:
        _pad_heads(dst, src, heads)
        return dst

    def reference(self, dst: torch.Tensor, src: Optional[torch.Tensor], heads: int) -> torch.Tensor:
        """A new tensor: ``dst`` with its padding heads zero and, with ``src``, its first ``heads``
        heads copied from it."""
        out = dst.clone()
        out[:, heads:] = 0
        if src is not None:
            out[:, :heads] = src[:, :heads]
        return out

    def is_valid(self, dst: torch.Tensor, src: Optional[torch.Tensor], heads: int) -> None:
        # The kernel indexes dst as a packed [n, pad, D] block from its base pointer.
        assert dst.is_contiguous(), "dst must be contiguous"
        assert 0 <= heads <= dst.shape[1]
        if src is not None:
            # A head row read with another row stride, or another dtype reinterpreted.
            assert src.dtype == dst.dtype and src.stride(-1) == 1 and src.shape[1] >= heads
            assert src.shape[0] == dst.shape[0] and src.shape[-1] == dst.shape[-1]


rubin_pad_heads = RubinPadHeads()
