# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's window attention on SM107 in Triton: split-KV body, then head and block.

Two launches per drafter layer over the step's plan (``rubin_dflash2_window_plan``): the body
kernel attends every query row of a request to one page-aligned chunk of its shared body, the
chunk count fixed by the plan's static shape; the tail kernel attends the head and block keys
and merges the chunk partials by log-sum-exp (``kernels/spec/rubin_dflash2_window``).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.rubin_dflash2_window import (
    BLOCK_DV,
    BLOCK_KC,
    BLOCK_M,
    NUM_WARPS,
    DFlash2WindowPlan,
    DFlash2WindowWorkspace,
    launch_window_triton,
    window_reference,
    window_vec_io_aligned,
    window_workspace,
)


@dataclasses.dataclass(frozen=True)
class WindowTiles:
    """Launch tiles of the body and tail kernels (``launch_window_triton``'s keywords).

    ``BLOCK_KC`` (the score dot's reduction chunk) is not a field: every tiling sums each score in
    the same order. The defaults are the kernel module's. Checked when made: power-of-two edges of
    16 or more, 1 to 16 warps, and a tile no larger than the default's BLOCK_M x BLOCK_DV: a
    64 x 512 tile at 4 or 8 warps ends the process with a misaligned-address fault (GB300).
    """

    block_m: int = BLOCK_M
    block_dv: int = BLOCK_DV
    num_warps: int = NUM_WARPS
    num_stages: int = 1

    def __post_init__(self) -> None:
        for edge in (self.block_m, self.block_dv):
            if edge < 16 or edge & (edge - 1):
                raise ValueError(f"window tile edge {edge}: a power of two of 16 or more")
        if self.block_m * self.block_dv > BLOCK_M * BLOCK_DV:
            raise ValueError(
                f"window tile {self.block_m} x {self.block_dv} is larger than {BLOCK_M} x {BLOCK_DV}"
            )
        if self.num_warps not in (1, 2, 4, 8, 16) or self.num_stages < 1:
            raise ValueError(f"window launch: {self.num_warps} warps, {self.num_stages} stages")

    def launch(self) -> dict:
        """``launch_window_triton``'s keywords."""
        return dict(
            block_m=self.block_m,
            block_dv=self.block_dv,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )


#: Kimi K3.1 DFlash2 drafter: kv_lora_rank 512 + rope 64, window 1024, 64-token draft pages,
#: softmax scale mscale(32, 1)^2 / sqrt(192).
_LATENT = 512
_ROPE = 64
_WINDOW = 1024
_PAGE = 64
_SOFTMAX_SCALE = 0.13086080
#: One history per request at the window's boundaries (see the plan entry): DEP16's 12, helix8's 16.
_DEP16_LENGTHS = (0, 1, 63, 65, 1017, 1023, 1024, 1031, 4097, 9000, 65536, 190000)
_HELIX8_LENGTHS = (0, 1, 5, 60, 63, 64, 66, 130, 1017, 1022, 1025, 1031, 4097, 9000, 65536, 190000)
#: The helix16 target's launch tiles (its modeling's ``_DRAFT_WINDOW_TILES``): one request's tail
#: grid at the default BLOCK_DV is 24 CTAs; BLOCK_DV 64 gives 96, bitwise the default.
_CP16_TILES = dict(block_m=64, block_dv=64, num_warps=8, num_stages=1)


def _spec(batch: int, block_size: int, heads: int, num_splits: int, **overrides) -> dict:
    spec = dict(
        batch=batch,
        block_size=block_size,
        heads=heads,
        num_splits=num_splits,
        latent=_LATENT,
        rope=_ROPE,
        window=_WINDOW,
        page_size=_PAGE,
        scale=_SOFTMAX_SCALE,
        pool="fp8",
        layers=1,
        cp_size=1,
        cp_rank=0,
        stats=False,
        lengths={12: _DEP16_LENGTHS, 16: _HELIX8_LENGTHS}.get(batch, _DEP16_LENGTHS[:batch]),
    )
    spec.update(overrides)
    return spec


class RubinDFlash2WindowTriton(OpWrapper):
    """Absorbed MLA of each block query row over its sliding window, as the plan splits it.

    ``out[r, h] = softmax_k(scale * query[r, h] . kv[k]) @ kv[k, :latent_dim]`` over the keys the
    plan gives row ``r = b * blk + j``: this rank's body pages, the head rows from
    ``head_first[b, j]`` on and the owned block rows ``k <= j`` of ``block_kv``. ``kv`` is an
    FP8 E4M3 or BF16 latent read times ``kv_scale``. Returns ``[R, H, latent_dim]`` in the
    query's dtype.

    ``query`` is ``[R, H, latent + rope]`` (BF16, ``[q_nope W_UK | q_pe]``), ``pool`` one layer's
    ``[pages, P, latent + rope]`` view (pages may be strided; rows contiguous), ``block_kv`` the
    ``[R, latent + rope]`` scratch holding the block's own latents in the pool's dtype.
    ``workspace`` comes from ``window_workspace`` (allocated before capture). With ``stats``
    (``[R, H, 2]`` float32, helix) the entry also writes each row's natural-log ``(max, sum)``,
    and a row with no key of this rank gets a zero output and ``(-inf, 0)``: the input
    ``_helix_post_process`` takes with ``zero_kv_mask=None``.

    ``vec_io`` (keyword, off by default): the kernels move their rows as 16-byte vectors where
    every stride allows (``window_vec_io_aligned`` says whether it does). ``pdl`` (keyword, off by
    default; under the engine-wide PDL switch): the tail launches as a programmatic dependent of the
    body. Neither changes an output byte.
    """

    ARCHS: frozenset[Arch] = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 generation, 12 requests at draft 7, 96 heads, FP8 pool",
            spec=_spec(12, 8, 96, 2),
        ),
        Cell(
            why="helix8 generation rank 3, 16 requests at draft 5, 96 heads, stats",
            spec=_spec(16, 6, 96, 2, cp_size=8, cp_rank=3, stats=True),
        ),
        Cell(
            why="helix8 rank 0: holds the head and short histories, other ranks' rows empty",
            spec=_spec(16, 6, 96, 2, cp_size=8, cp_rank=0, stats=True),
        ),
        Cell(
            why="TEP8 generation, one request at draft 7, 12 heads, 16 chunks",
            spec=_spec(1, 8, 12, 16, lengths=(190000,)),
        ),
        Cell(
            why="TEP8, short history: most chunks empty",
            spec=_spec(1, 8, 12, 16, lengths=(300,)),
        ),
        Cell(
            why="BF16 history (a BF16 draft pool)",
            spec=_spec(4, 8, 96, 2, pool="bf16", lengths=(1, 1030, 9000, 190000)),
        ),
        Cell(
            why="the pool's pages strided by 5 layers (the KV cache manager's layout)",
            spec=_spec(4, 8, 96, 3, layers=5, lengths=(64, 1025, 4097, 65536)),
        ),
        Cell(
            why="dummy rows of a padded graph: histories 0 .. blk - 1, empty bodies",
            spec=_spec(8, 8, 12, 2, lengths=tuple(range(8))),
        ),
        Cell(
            why="helix16 target rank 0, one request at draft 7, 96 heads, one chunk, its tiles",
            spec=_spec(
                1, 8, 96, 1, cp_size=16, cp_rank=0, stats=True, lengths=(262150,), tiles=_CP16_TILES
            ),
        ),
        Cell(
            why="helix16 target rank 9, a history just past the window, its tiles",
            spec=_spec(
                1, 8, 96, 1, cp_size=16, cp_rank=9, stats=True, lengths=(1031,), tiles=_CP16_TILES
            ),
        ),
    )

    note = """
    Not bitwise the FP32 decode: the probabilities enter P.V as BF16 and the chunks merge in
    another order; scores and accumulations are FP32 over BF16 queries and BF16 (or exactly
    widened FP8) keys. The test bounds it against `reference`, the exact FP32 softmax over the
    same keys, and separately checks the plan's keys against the dense window
    (`sparse_mla_rows_eager`). Whether that changes the drafter's acceptance length is
    checked on serving runs, not here.
    """

    def __call__(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        block_kv: torch.Tensor,
        plan: DFlash2WindowPlan,
        softmax_scale: float,
        latent_dim: int,
        kv_scale: float,
        workspace: DFlash2WindowWorkspace,
        stats: Optional[torch.Tensor] = None,
        tiles: Optional[WindowTiles] = None,
        *,
        vec_io: bool = False,
        pdl: bool = False,
    ) -> torch.Tensor:
        launch = {} if tiles is None else tiles.launch()
        return launch_window_triton(
            query,
            pool,
            block_kv,
            plan,
            softmax_scale,
            latent_dim,
            kv_scale,
            workspace,
            stats,
            **launch,
            vec_io=vec_io,
            pdl=pdl,
        )

    def reference(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        block_kv: torch.Tensor,
        plan: DFlash2WindowPlan,
        softmax_scale: float,
        latent_dim: int,
        kv_scale: float,
        workspace: DFlash2WindowWorkspace,
        stats: Optional[torch.Tensor] = None,
        tiles: Optional[WindowTiles] = None,
        *,
        vec_io: bool = False,
        pdl: bool = False,
    ) -> torch.Tensor:
        """One FP32 softmax per row over the plan's keys (``workspace``, ``tiles`` and the levers
        unused)."""
        del tiles
        return window_reference(
            query, pool, block_kv, plan, softmax_scale, latent_dim, kv_scale, stats
        )

    def is_valid(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        block_kv: torch.Tensor,
        plan: DFlash2WindowPlan,
        softmax_scale: float,
        latent_dim: int,
        kv_scale: float,
        workspace: DFlash2WindowWorkspace,
        stats: Optional[torch.Tensor] = None,
        tiles: Optional[WindowTiles] = None,
        *,
        vec_io: bool = False,
        pdl: bool = False,
    ) -> None:
        assert isinstance(vec_io, bool) and isinstance(pdl, bool), "launch switches"
        batch, block_size = plan.head_first.shape
        rows, heads, dim = query.shape
        rope = dim - latent_dim
        assert query.dtype in (torch.bfloat16, torch.float16), "half-precision query"
        assert query.stride(-1) == 1 and query.stride(1) == dim, "rows of contiguous heads"
        assert rows == batch * block_size, "request-major query rows, blk per request"
        assert pool.dim() == 3 and pool.shape[-1] == dim and pool.stride(-1) == 1
        assert pool.stride(1) == dim, "each page's rows must be contiguous"
        assert pool.dtype in (torch.float8_e4m3fn, torch.bfloat16)
        assert block_kv.shape == (rows, dim) and block_kv.dtype == pool.dtype
        assert block_kv.stride(-1) == 1
        page_size = pool.shape[1]
        for width in (page_size, rope, latent_dim):
            assert width >= 16 and width & (width - 1) == 0, "power-of-two tiles of 16 or more"
        assert latent_dim % min(BLOCK_KC, latent_dim) == 0
        num_splits = plan.chunk_bounds.shape[1] - 1
        assert (
            workspace.partial_out.shape[0] >= batch and workspace.partial_out.shape[1] >= num_splits
        )
        assert workspace.partial_out.shape[2:] == (block_size * heads, latent_dim)
        assert workspace.partial_stats.shape[2:] == (block_size * heads, 2)
        assert workspace.partial_out.dtype == workspace.partial_stats.dtype == torch.float32
        if stats is not None:
            assert stats.shape == (rows, heads, 2) and stats.dtype == torch.float32
            assert stats.stride(-1) == 1
        if tiles is not None:  # WindowTiles checked its own fields when made
            assert latent_dim % min(tiles.block_dv, latent_dim) == 0


rubin_dflash2_window_triton = RubinDFlash2WindowTriton()

__all__ = [
    "DFlash2WindowWorkspace",
    "WindowTiles",
    "rubin_dflash2_window_triton",
    "window_vec_io_aligned",
    "window_workspace",
]
