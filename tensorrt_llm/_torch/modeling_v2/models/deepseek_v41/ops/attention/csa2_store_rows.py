# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused quantize + store of a layer's cache rows, and in-place rope of latent heads.

Wraps the engine's one-launch quantize/scatter kernels (`row_job`, `footer_job`, `split_job`,
`quantize_scatter_jobs` of the CSA2 CuTe kernels; up to four jobs per launch) and
`torch.ops.trtllm.mla_rope_inplace`. One launch publishes the SWA rows (optionally RMS-normalised and
rotated on the way), the compressed main rows, the index-key rows in native footer pages, and quantizes
the index queries into separate data / scale bytes. The byte formats are the ones of `csa2_pack_rows`.
Static shapes, device-side slots: safe under CUDA-graph capture after one eager call (kernel JIT).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_pack_rows import csa2_pack_rows
from .csa2_rope import csa2_rope_tail
from .csa2_rows_io import csa2_scatter_rows, csa2_scatter_rows_footer

_FOOTER_PAGE_ROWS = 64
# The stock layout of a target-owned rope table, kept on the table itself (this attribute): a 1M-position table's is
# hundreds of MB, so it is built once per table, not once per call, and is released with the table.
_STOCK_LAYOUT_ATTR = "_csa2_stock_layout"


def csa2_stock_rope_table(table: torch.Tensor) -> torch.Tensor:
    """`[P, rd/2, 2]` (cos, sin) -> the engine's `[P, rd]` float32 (all cos, then all sin) layout.

    A setup helper, not an entry: a target converts its tables once and keeps only the result (a table already in
    the engine's layout is returned as is), and passes it to `csa2_rope_inplace` / `csa2_store_rows`.
    """
    if table.dim() == 2:  # already the engine's layout (what a target keeps)
        return table
    stock = getattr(table, _STOCK_LAYOUT_ATTR, None)
    if stock is None:
        stock = torch.cat([table[..., 0], table[..., 1]], dim=-1).contiguous()
        setattr(table, _STOCK_LAYOUT_ATTR, stock)
    return stock


def _table_of(cos_sin: torch.Tensor) -> torch.Tensor:
    """The `[P, rd/2, 2]` table `csa2_rope_tail` takes, from the engine's `[P, rd]` layout."""
    half = cos_sin.shape[-1] // 2
    return torch.stack([cos_sin[:, :half], cos_sin[:, half:]], dim=-1)


class Csa2RopeInplace(OpWrapper):
    """Rotate the trailing rope dims (interleaved pairs) of `x [n, h, D]` at `pos [n]`, in place.

    `x` is bf16 and contiguous, `pos` int32, `cos_sin` the engine's `[P, rd]` float32 layout
    (`csa2_stock_rope_table`); `inverse` rotates back. Returns `x`.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = tuple(
        Cell(why=why, spec=dict(n=300, heads=h, dim=d, inverse=inv))
        for h, d, what in (
            (64, 512, "all 64 query heads"),
            (32, 128, "32 index heads"),
            (8, 512, "8 heads of a tp8 rank"),
        )
        for inv, why in ((False, f"{what}: forward rotation"), (True, f"{what}: inverse rotation"))
    )
    note = "rd is 64 at every cell (the checkpoint's `qk_rope_head_dim`)."

    def __call__(
        self, x: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        n, h, d = x.shape
        rd = cos_sin.shape[-1]
        if n:
            torch.ops.trtllm.mla_rope_inplace(
                x, pos, cos_sin.view(-1, 2, rd // 2), h, d - rd, rd, inverse, False
            )
        return x

    def reference(
        self, x: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """`csa2_rope_tail.reference` (the checkpoint's complex-exponential form) on the same table;
        returns a new tensor and leaves `x` alone."""
        return csa2_rope_tail.reference(x, _table_of(cos_sin), pos, cos_sin.shape[-1], inverse)

    def is_valid(
        self, x: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor, inverse: bool = False
    ) -> None:
        assert x.dim() == 3 and x.dtype == torch.bfloat16 and x.is_contiguous(), (
            "x must be a contiguous bf16 [n, h, D]: the kernel rotates it in place through raw pointers"
        )
        assert pos.dtype == torch.int32 and pos.shape == (x.shape[0],), "pos is int32 [n]"
        assert cos_sin.dtype == torch.float32 and cos_sin.dim() == 2 and cos_sin.is_contiguous(), (
            "cos_sin is the engine layout [P, rd] float32 (csa2_stock_rope_table), all cos then all sin"
        )
        assert cos_sin.shape[-1] % 2 == 0 and x.shape[-1] >= cos_sin.shape[-1]


class Csa2StoreRows(OpWrapper):
    """One launch that normalises / rotates / quantizes rows and writes them into the cache pools.

    Each of `swa` / `main` / `index` is `(pool, slots, values, norm, rope)` (`swa` may be None when only the
    compressed rows are published, e.g. the boundary layer of a bounded decoder replay):

    - `pool`: SWA `[rows, 528]` / main `[rows, 288]` uint8 view of the pool, or the native index pages
      (uint8, any shape);
    - `slots` int64 `[n]`: flat row of each value row (`-1` skips it); `values` bf16 `[n, 512]`
      (index: `[n, 128]`);
    - `norm = (weight bf16, eps)` and `rope = (pos int32 [n], csa2_stock_rope_table, 64)` are applied
      to `values` first, or None.

    `index_q` bf16 `[n, heads, 128]` is quantized in the same launch; the op then returns
    `(data int8 [n, heads, 64], scales uint8 [n, heads, 4])`, else None.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="a 1280-row decode step of dep8 (batch 320 x 4 tokens): all four jobs",
            spec=dict(n=1280, main=True, index=True, index_q=True),
        ),
        Cell(
            why="a layer without compression: SWA rows and the index queries only",
            spec=dict(n=96, main=False, index=False, index_q=True),
        ),
        Cell(
            why="a 4096-row prefill chunk: SWA, main and index rows, no index queries",
            spec=dict(n=4096, main=True, index=True, index_q=False),
        ),
        Cell(
            why="boundary layer of a bounded decoder replay: only the main and index rows of 4096 rows, no SWA",
            spec=dict(n=4096, main=True, index=True, index_q=False, swa=False),
        ),
    )
    note = """
    The fused kernel and the reference may differ by one quantization step on rare ties, so the
    test compares decoded values and the share of differing bytes, not bytes.
    """

    def __call__(
        self,
        swa: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[tuple], Optional[tuple]]
        ],
        main: Optional[tuple] = None,
        index: Optional[tuple] = None,
        index_q: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import (
            footer_job,
            quantize_scatter_jobs,
            row_job,
            split_job,
        )

        jobs = [] if swa is None else [row_job(swa[0], swa[1], swa[2], "swa", swa[3], swa[4])]
        if main is not None:
            jobs.append(row_job(main[0], main[1], main[2], "main", main[3], main[4]))
        if index is not None:
            jobs.append(
                footer_job(index[0], index[1], index[2], _FOOTER_PAGE_ROWS, index[3], index[4])
            )
        packed = None
        if index_q is not None:
            n, h, d = index_q.shape
            data = torch.empty(n, h, 64, dtype=torch.uint8, device=index_q.device)
            scales = torch.empty(n, h, 4, dtype=torch.uint8, device=index_q.device)
            jobs.append(
                split_job(
                    index_q.reshape(n * h, d).contiguous(),
                    data.view(n * h, 64),
                    scales.view(n * h, 4),
                )
            )
            packed = (data.view(torch.int8), scales)
        first = next(t for t in (swa, main, index) if t is not None)
        with torch.cuda.device(first[2].device):
            quantize_scatter_jobs(jobs)
        return packed

    def reference(
        self,
        swa: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[tuple], Optional[tuple]]
        ],
        main: Optional[tuple] = None,
        index: Optional[tuple] = None,
        index_q: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """RMS-norm, rope (the checkpoint's complex form), `csa2_pack_rows`, then the masked scatter,
        written into the pools passed in (pass copies) exactly like the op."""

        def prep(values, norm, rope):
            x = values
            if norm is not None:
                v = x.float()
                x = (
                    v * torch.rsqrt(v.square().mean(-1, keepdim=True) + norm[1]) * norm[0].float()
                ).to(torch.bfloat16)
            if rope is not None:
                x = csa2_rope_tail.reference(x, _table_of(rope[1]), rope[0], rope[2])
            return x

        def store(spec, kind, scatter):
            pool, slots, values, norm, rope = spec
            if not bool((slots >= 0).any()):
                return
            rows = csa2_pack_rows.reference(prep(values, norm, rope), kind)
            scatter(pool, slots, rows, torch.ones_like(slots, dtype=torch.bool))

        if swa is not None:
            store(swa, "swa", csa2_scatter_rows)
        if main is not None:
            store(main, "main", csa2_scatter_rows)
        if index is not None:
            store(index, "index", csa2_scatter_rows_footer)
        if index_q is None:
            return None
        packed = csa2_pack_rows.reference(index_q.reshape(-1, 128), "index")
        n, h, _ = index_q.shape
        return (
            packed[:, :64].contiguous().view(torch.int8).view(n, h, 64),
            packed[:, 64:].contiguous().view(n, h, 4),
        )

    def is_valid(
        self,
        swa: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[tuple], Optional[tuple]]
        ],
        main: Optional[tuple] = None,
        index: Optional[tuple] = None,
        index_q: Optional[torch.Tensor] = None,
    ) -> None:
        assert any(t is not None for t in (swa, main, index)), "nothing to store"
        for name, spec, width in (("swa", swa, 512), ("main", main, 512), ("index", index, 128)):
            if spec is None:
                continue
            pool, slots, values, norm, rope = spec
            assert pool.dtype == torch.uint8, f"{name}: the pool is uint8 bytes"
            assert values.dtype == torch.bfloat16 and values.shape[1:] == (width,), (
                f"{name}: values are bf16 [n, {width}]"
            )
            assert slots.dtype == torch.int64 and slots.shape == (values.shape[0],), (
                f"{name}: slots are int64 [n] flat rows (-1 skips a row)"
            )
        if index_q is not None:
            assert (
                index_q.dtype == torch.bfloat16 and index_q.dim() == 3 and index_q.shape[-1] == 128
            )


csa2_rope_inplace = Csa2RopeInplace()
csa2_store_rows = Csa2StoreRows()
