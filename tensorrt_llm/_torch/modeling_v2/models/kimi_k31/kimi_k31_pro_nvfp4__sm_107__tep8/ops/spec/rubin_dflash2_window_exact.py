# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's window attention in FP32: the exact decode the DEP16 and TEP8 drafters run.

Every query row attends its window's rows of the paged latent pool (``dflash2_window_rows``): a
gather of the rows, FP32 scores, a softmax and the FP32 value product, rounded once to the
query's dtype at the end. No kernel of its own: torch ops on the current stream.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell, OpWrapper

from ....ops.attention.dflash2_window import sparse_mla_rows_eager

#: Kimi K3.1 DFlash2 drafter: kv_lora_rank 512 + rope 64, window 1024, 64-token draft pages,
#: softmax scale mscale(32, 1)^2 / sqrt(192).
_LATENT = 512
_ROPE = 64
_WINDOW = 1024
_PAGE = 64
_SOFTMAX_SCALE = 0.13086080


def _spec(batch: int, block_size: int, heads: int, **overrides) -> dict:
    spec = dict(
        batch=batch,
        block_size=block_size,
        heads=heads,
        latent=_LATENT,
        rope=_ROPE,
        window=_WINDOW,
        page_size=_PAGE,
        scale=_SOFTMAX_SCALE,
        pool="fp8",
    )
    spec.update(overrides)
    return spec


class RubinDFlash2WindowExact(OpWrapper):
    """``out[r, h] = softmax_k(scale * query[r, h] . kv[k]) @ kv[k, :latent_dim]`` over the pool
    rows ``k`` of ``rows[r]`` (``-1`` entries are masked), in FP32; ``kv`` the pool's rows, an FP8
    pool read times ``kv_scale``. Returns a new ``[R, H, latent_dim]`` tensor in the query's dtype.

    ``query`` ``[R, H, latent_dim + rope]`` holds the absorbed queries (``[q_nope W_UK | q_pe]``),
    ``pool_rows`` ``[N, latent_dim + rope]`` the pool flattened to rows (``page * page_size +
    offset``), ``rows`` int32 ``[R, K]`` each query's window rows.
    """

    ARCHS = frozenset()

    CELLS: tuple[Cell, ...] = (
        Cell(why="TEP8 generation b1 k7: 8 query rows, 12 heads", spec=_spec(1, 8, 12)),
        Cell(why="DEP16 generation b12 k7: 96 query rows, 96 heads", spec=_spec(12, 8, 96)),
        Cell(why="DEP16 generation b16 k5: 96 query rows, 96 heads", spec=_spec(16, 6, 96)),
        Cell(why="a BF16 pool (no dequant scale)", spec=_spec(2, 8, 12, pool="bf16")),
    )

    note = """
    The window decode as the drafter's reference computes it: the gathered rows are read as FP32
    (an FP8 pool times `kv_scale`), the scores and the value product accumulate in FP32, and the
    output is rounded once to the query's dtype. A step allocates the gathered rows
    (`[R, K, latent_dim + rope]` FP32) and the scores; under a CUDA graph they come from the
    graph's pool. `reference` is the same arithmetic in float64.
    """

    def __call__(
        self,
        query: torch.Tensor,
        pool_rows: torch.Tensor,
        rows: torch.Tensor,
        softmax_scale: float,
        latent_dim: int,
        kv_scale: float,
    ) -> torch.Tensor:
        return sparse_mla_rows_eager(query, pool_rows, rows, softmax_scale, latent_dim, kv_scale)

    def reference(
        self,
        query: torch.Tensor,
        pool_rows: torch.Tensor,
        rows: torch.Tensor,
        softmax_scale: float,
        latent_dim: int,
        kv_scale: float,
    ) -> torch.Tensor:
        """The window decode in float64, rounded once to the query's dtype."""
        valid = rows >= 0
        kv = pool_rows[rows.clamp_min(0).long()].double() * kv_scale
        scores = torch.einsum("rhc,rkc->rhk", query.double(), kv) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("rhk,rkc->rhc", probs, kv[..., :latent_dim]).to(query.dtype)

    def is_valid(
        self,
        query: torch.Tensor,
        pool_rows: torch.Tensor,
        rows: torch.Tensor,
        softmax_scale: float,
        latent_dim: int,
        kv_scale: float,
    ) -> None:
        assert query.dim() == 3 and pool_rows.dim() == 2 and rows.dim() == 2, (
            "[R, H, D], [N, D], [R, K]"
        )
        assert rows.dtype == torch.int32 and rows.shape[0] == query.shape[0], (
            "one row list per query"
        )
        assert pool_rows.shape[1] == query.shape[2], "query and pool rows share their width"
        assert 0 < latent_dim <= query.shape[2], "the output columns lie inside the rows"
        # A row with no valid entry gives NaN (softmax over nothing): every query attends itself.
        assert bool((rows >= 0).any(dim=1).all()), "every query row attends at least one pool row"
        assert int(rows.max()) < pool_rows.shape[0], "a window row past the pool"


rubin_dflash2_window_exact = RubinDFlash2WindowExact()
