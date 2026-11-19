# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's exact FP32 window decode, its operands staged in two Triton launches.

The decode of DEP16 and TEP8 (``sparse_mla_rows_eager``): each block query row attends the pool
rows of its sliding window through two TF32 einsums and a softmax over FP32 operands. This entry
makes the same einsum, softmax and BF16 calls on the same FP32 operands; only their staging moves
to ``kernels/spec/rubin_dflash2_window_fp32`` (the window rows gathered and widened with the
query in one launch, the scores scaled and masked in place in another), where the torch decode
spends nine launches per layer.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ....ops.attention.dflash2_window import sparse_mla_rows_eager
from ...kernels.spec.rubin_dflash2_window_fp32 import window_fp32

#: Kimi K3.1 DFlash2 drafter: kv_lora_rank 512 + rope 64, window 1024, 64-token draft pages,
#: softmax scale mscale(32, 1)^2 / sqrt(192).
_LATENT = 512
_ROPE = 64
_WINDOW = 1024
_PAGE = 64
_SOFTMAX_SCALE = 0.13086080
#: One history per request: DEP16's 12 at the window's boundaries and far past it.
_DEP16_LENGTHS = (0, 1, 63, 65, 1017, 1023, 1024, 1031, 4097, 9000, 65536, 190000)


def _spec(lengths: tuple, heads: int, **overrides) -> dict:
    spec = dict(
        lengths=lengths,
        block_size=8,
        heads=heads,
        latent=_LATENT,
        rope=_ROPE,
        window=_WINDOW,
        page_size=_PAGE,
        scale=_SOFTMAX_SCALE,
        pool="fp8",
        kv_scale=1.0,
    )
    spec.update(overrides)
    return spec


class RubinDFlash2WindowFp32(OpWrapper):
    """Absorbed MLA of each query row over the pool rows of its window, in FP32.

    ``out[r, h] = softmax_k(scale * query[r, h] . kv[k]) @ kv[k, :latent_dim]`` over the rows
    ``rows[r]`` lists (``-1`` masked), ``kv`` the pool row widened to FP32 (times ``kv_scale``).
    ``query`` is ``[R, H, latent + rope]`` (contiguous BF16, ``[q_nope W_UK | q_pe]``),
    ``pool_rows`` one layer's ``[N, latent + rope]`` rows (FP8 E4M3 or BF16), ``rows`` the
    ``[R, K]`` int32 index lists of ``dflash2_window_rows``. Returns ``[R, H, latent_dim]`` in
    the query's dtype.
    """

    ARCHS: frozenset[Arch] = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TEP8 generation: one request at draft 7, 12 heads, a 131k history",
            spec=_spec((131072,), 12),
        ),
        Cell(
            why="TEP8, histories inside the first window: lists cut at position 0, -1 columns",
            spec=_spec((0, 5, 300, 1017, 1023, 1030), 12),
        ),
        Cell(
            why="DEP16 generation: 12 requests at draft 7, 96 heads",
            spec=_spec(_DEP16_LENGTHS, 96),
        ),
        Cell(
            why="BF16 history (a BF16 draft pool)",
            spec=_spec((1030, 131072), 12, pool="bf16"),
        ),
        Cell(
            why="an FP8 dequant scale other than 1",
            spec=_spec((4097, 9000), 12, kv_scale=0.5),
        ),
    )

    note = """
    Bit for bit `reference` (`sparse_mla_rows_eager`): the same TF32 einsums, softmax and BF16
    rounding over the same FP32 operands (FP8 and BF16 widen exactly; the scale is one FP32
    product in both), so a cell either matches every bit or fails. The one input the two read
    differently is invalid: an index past the pool's last row, which the torch gather rejects,
    reads the last row here (`is_valid`).
    """

    def __call__(
        self,
        query: torch.Tensor,
        pool_rows: torch.Tensor,
        rows: torch.Tensor,
        softmax_scale: float,
        latent_dim: int,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        return window_fp32(query, pool_rows, rows, softmax_scale, latent_dim, kv_scale)

    def reference(
        self,
        query: torch.Tensor,
        pool_rows: torch.Tensor,
        rows: torch.Tensor,
        softmax_scale: float,
        latent_dim: int,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        """The torch decode the drafter ran before this entry."""
        return sparse_mla_rows_eager(query, pool_rows, rows, softmax_scale, latent_dim, kv_scale)

    def is_valid(
        self,
        query: torch.Tensor,
        pool_rows: torch.Tensor,
        rows: torch.Tensor,
        softmax_scale: float,
        latent_dim: int,
        kv_scale: float = 1.0,
    ) -> None:
        # The torch decode widens the query in its own layout; the einsums see the same operands
        # only when that layout is the contiguous one this entry writes.
        assert query.dtype in (torch.bfloat16, torch.float16) and query.dim() == 3
        assert query.is_contiguous(), "a contiguous query"
        assert pool_rows.dim() == 2 and pool_rows.shape[1] == query.shape[2]
        assert pool_rows.stride(1) == 1, "pool rows with a unit inner stride"
        assert pool_rows.dtype in (torch.float8_e4m3fn, torch.bfloat16)
        assert rows.dtype == torch.int32 and rows.dim() == 2 and rows.is_contiguous()
        assert rows.shape[0] == query.shape[0], "one index list per query row"
        assert 0 < latent_dim <= query.shape[2]
        # The torch gather raises on an index past the pool; the kernel reads the last row.
        assert rows.numel() == 0 or int(rows.max()) < pool_rows.shape[0], "an index past the pool"


rubin_dflash2_window_fp32 = RubinDFlash2WindowFp32()

__all__ = ["rubin_dflash2_window_fp32"]
