# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MTP index share: draft step 0's TopK row of each request's last accepted token, in one launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._mtp_topk_stash import launch_stash_last_accepted_rows


def _spec(requests, accepted, **overrides) -> dict:
    spec = dict(requests=requests, next_n=8, n_cols=2048, accepted=tuple(accepted), dst="rows")
    spec.update(overrides)
    return spec


class GlmStashLastAcceptedRows(OpWrapper):
    """``dst[r] = gen_topk[r * next_n + clamp(num_accepted[r] - 1, 0, next_n - 1)]`` per request.

    Draft step 0 of the MTP draft loop computes the TopK of the verify step's ``next_n`` rows; the
    draft steps after it reuse, per request, the row of its last accepted token. ``num_accepted``
    is the verify step's accepted-token count per request (1 to next_n); out-of-range counts clamp.
    Returns ``dst``.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP4 first draft: three requests with mixed acceptance",
            spec=_spec(3, [1, 4, 6], next_n=6),
        ),
        Cell(
            why="TP8 first draft: two requests with mixed acceptance",
            spec=_spec(2, [1, 4], next_n=4, dst="wide"),
        ),
        Cell(why="c1: one request, 8 TopK rows, 3 tokens accepted", spec=_spec(1, [3])),
        Cell(why="c1: every draft token accepted (the last row)", spec=_spec(1, [8])),
        Cell(why="counts past the window clamp: 0 and 9", spec=_spec(2, [0, 9])),
        Cell(
            why="four requests into rows of a wider shared buffer",
            spec=_spec(4, [1, 5, 8, 2], dst="wide"),
        ),
    )

    note = """
    Metadata consumed: none. The target passes the draft step-0 TopK output
    ([num_requests * next_n, 2048] int32), the verify step's accepted counts per request (int32 on the
    device) and the rows [0, num_requests) of the persistent shared-TopK buffer. A plain launch
    (no PDL): at c1 it runs on its own stream behind an event recorded after the TopK, and the
    layer joins it at the end of its attention.
    """

    def __call__(
        self, gen_topk: torch.Tensor, num_accepted: torch.Tensor, dst: torch.Tensor, next_n: int
    ) -> torch.Tensor:
        launch_stash_last_accepted_rows(gen_topk, num_accepted, dst, next_n)
        return dst

    def reference(
        self, gen_topk: torch.Tensor, num_accepted: torch.Tensor, dst: torch.Tensor, next_n: int
    ) -> torch.Tensor:
        """The selected rows as a fresh tensor."""
        requests = dst.shape[0]
        offsets = (num_accepted.long() - 1).clamp(0, next_n - 1)
        index = torch.arange(requests, device=gen_topk.device) * next_n + offsets
        return gen_topk[index, : dst.shape[1]].clone()

    def is_valid(
        self, gen_topk: torch.Tensor, num_accepted: torch.Tensor, dst: torch.Tensor, next_n: int
    ) -> None:
        requests, n_cols = dst.shape
        # Rows are addressed as request * next_n + offset with unit column strides.
        assert gen_topk.shape == (requests * next_n, n_cols), (
            f"gen_topk must be [{requests} * {next_n}, {n_cols}], got {tuple(gen_topk.shape)}"
        )
        assert gen_topk.dtype == dst.dtype, (gen_topk.dtype, dst.dtype)
        assert gen_topk.stride(1) == 1 and dst.stride(1) == 1, "unit column strides"
        assert num_accepted.is_contiguous() and num_accepted.numel() >= requests, (
            "num_accepted must hold one contiguous count per request"
        )


stash_last_accepted_rows = GlmStashLastAcceptedRows()
