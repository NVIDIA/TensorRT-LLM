# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-Connections post-mapping in one launch: expand a sub-layer output and mix the residual in."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post import hc_post


class HcPostMapping(OpWrapper):
    """`-> [n, hc, d]` bf16: `post[t, i] * x[t] + sum_j comb[t, j, i] * residual[t, j]`.

    The engine's `mhc_post_mapping`; the same function as the plain-torch `hc_post` (bf16 stream).
    `comb` is indexed [token, source copy, destination copy].
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="decode rows 6", spec=dict(n=6, hc=4, d=5120)),
        Cell(why="decode rows 18", spec=dict(n=18, hc=4, d=5120)),
        Cell(why="decode rows 96", spec=dict(n=96, hc=4, d=5120)),
        Cell(why="decode rows 1280", spec=dict(n=1280, hc=4, d=5120)),
        Cell(why="prefill tile: 8192 tokens of a chunk", spec=dict(n=8192, hc=4, d=5120)),
        Cell(why="an attention-DP rank with no rows", spec=dict(n=0, hc=4, d=5120)),
    )
    note = "Elementwise in fp32, one bf16 rounding; allocates only its output and syncs nothing."

    def __call__(
        self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        n, hc, d = residual.shape
        out = torch.empty_like(residual)
        if n == 0:  # an attention-DP rank without rows
            return out
        torch.ops.trtllm.mhc_post_mapping(residual, x, post, comb, out, n, d)
        return out

    def reference(
        self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        """The plain-torch `hc_post` entry's reference (float64, the loops written out)."""
        return hc_post.reference(x, residual, post, comb)

    def is_valid(
        self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> None:
        n, hc, d = residual.shape
        assert hc == 4, "the kernel is instantiated for 4 residual copies"
        assert x.shape == (n, d) and post.shape == (n, hc) and comb.shape == (n, hc, hc)
        assert x.dtype == residual.dtype == torch.bfloat16
        assert post.dtype == comb.dtype == torch.float32
        assert all(t.is_cuda and t.is_contiguous() for t in (x, residual, post, comb))
        assert d % 8 == 0 and x.data_ptr() % 16 == 0 and residual.data_ptr() % 16 == 0


hc_post_mapping = HcPostMapping()
