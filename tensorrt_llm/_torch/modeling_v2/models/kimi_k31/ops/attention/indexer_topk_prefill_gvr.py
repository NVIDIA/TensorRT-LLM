# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer prefill top-k: the deterministic self-sampling GVR top-k (CuTe DSL) on every tile with a
selecting row, the CUDA radix on tiles of identity rows only."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.gvr_topk_decode_host import (  # noqa: E501
    run_prefill,
    warmup_prefill,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_prefill import (  # noqa: E501
    indexer_topk_prefill,
)

#: Largest key window (logits columns) the warmup compiles the GVR prefill engines for; every larger
#: window takes the engine of this envelope bucket.
_WARMUP_COLUMNS = 1 << 15
_WARMED: set = set()  # (device index, index_topk) whose GVR prefill engines are compiled


def _spec(requests, **overrides) -> dict:
    """``requests``: (cached keys, new query rows) per request of one prefill chunk."""
    spec = dict(requests=tuple(requests), index_topk=2048, data="distinct", logits_pad=0)
    spec.update(overrides)
    return spec


class IndexerTopkPrefillGvr(OpWrapper):
    """Writes into ``indices`` (int32 [M, index_topk], contiguous) the ``index_topk`` largest
    columns of each row's window ``[row_starts[r], row_ends[r])`` of ``logits`` (fp32 [M, N], any
    row stride), as offsets from ``row_starts[r]``; ties at the k-th value go to the smaller
    column. A window of at most ``index_topk`` columns yields ``0 .. len - 1``, then -1. The order
    within a row is unspecified (the caller orders the rows). Columns outside the windows are
    never read. ``tier_rows`` (GVR windows only): the engine plan of a ``tier_rows``-row launch, so
    a slice of a larger tile gets the tile's per-row output bit for bit.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TEP8 ctx q-split share: 2048 rows over a 120k-key cached prefix (GVR)",
            spec=_spec([(120_000, 2048)]),
        ),
        Cell(
            why="DEP16 ctx chunk: 4096 rows over a 100k-key prefix (GVR)",
            spec=_spec([(100_000, 4096)]),
        ),
        Cell(
            why="a fresh 16k-token prompt: windows 1..16384 (GVR, rows up to 2048 keys identity)",
            spec=_spec([(0, 16384)]),
        ),
        Cell(
            why="three packed cached requests in one chunk (windows start at each request's keys)",
            spec=_spec([(30_000, 300), (60_000, 700), (20_000, 64)]),
        ),
        Cell(
            why="threshold ties: integer-valued scores, the smaller columns win the ties",
            spec=_spec([(40_000, 512)], data="ties"),
        ),
        Cell(
            why="a fresh 3k-token prompt: rows past 2048 keys select (GVR), the rest are identity rows",
            spec=_spec([(0, 3000)]),
        ),
        Cell(
            why="ties in a fresh 2.1k-token prompt: integer-valued scores, the smaller columns win",
            spec=_spec([(0, 2100)], data="ties"),
        ),
        Cell(
            why="ties in a fresh 3k-token prompt: integer-valued scores, the smaller columns win",
            spec=_spec([(0, 3000)], data="ties"),
        ),
        Cell(
            why="a 2048-token prompt: identity rows only (the radix serves the tile)",
            spec=_spec([(0, 2048)]),
        ),
        Cell(
            why="a padded logits row stride (DeepGEMM's aligned prefill rows)",
            spec=_spec([(50_000, 256)], logits_pad=448),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).
    Every K3.1 target on SM 10.3 calls this in K31Indexer.forward_prepared in place of indexer_topk_prefill
    (same arguments): logits the fp8_mqa_logits tile of a prefill chunk (clean_logits False),
    row_starts / row_ends the chunk's cu_seqlen_ks / ke slices. A tile wider than index_topk
    logits columns (it may hold a selecting row) takes the deterministic GVR prefill engine: the
    selected set is a function of the scores (ties at the k-th score to the smaller column), as in
    the decode GVR. The CUDA radix picks among tied scores in atomic order, so a prompt of a few
    thousand tokens could select different keys from run to run; it serves only tiles of at most
    index_topk columns, whose rows are all identity rows. The first call compiles every GVR
    prefill engine (row tiers, envelope buckets up to _WARMUP_COLUMNS; it runs in the engine's
    warmup), so a CUDA-graph capture finds them compiled (run_prefill refuses to compile under
    capture). GB300 per prefill chunk, radix -> GVR: 2.1k columns 18.5 -> 48 us, 3k 33.5 -> 77,
    6k 108 -> 160, 8k 178 -> 203, 12k 313 -> 288, 16k 613 -> 385.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        index_topk: int,
        tier_rows: int | None = None,
    ) -> None:
        n = logits.shape[1]
        if n > index_topk:
            key = (logits.device.index, index_topk)
            if key not in _WARMED and not torch.cuda.is_current_stream_capturing():
                warmup_prefill(index_topk, _WARMUP_COLUMNS)
                _WARMED.add(key)
            run_prefill(
                logits,
                row_starts,
                row_ends,
                indices,
                max_row_len=n,
                tier_rows=tier_rows,
            )
            return
        indexer_topk_prefill(logits, row_starts, row_ends, indices, index_topk)

    def reference(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        index_topk: int,
        tier_rows: int | None = None,
    ) -> None:
        """Per row: a stable descending sort of the window, first index_topk offsets (``tier_rows``
        only picks the engine plan and is ignored)."""
        indices.fill_(-1)
        for r in range(logits.shape[0]):
            s, e = int(row_starts[r]), int(row_ends[r])
            if e - s <= index_topk:
                indices[r, : e - s] = torch.arange(e - s, dtype=torch.int32, device=logits.device)
                continue
            order = torch.sort(logits[r, s:e], descending=True, stable=True).indices
            indices[r] = order[:index_topk].int()

    def is_valid(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        index_topk: int,
        tier_rows: int | None = None,
    ) -> None:
        assert logits.dtype == torch.float32 and logits.stride(1) == 1
        assert indices.dtype == torch.int32 and indices.is_contiguous()
        assert indices.shape == (logits.shape[0], index_topk)
        assert index_topk % 4 == 0 and indices.data_ptr() % 16 == 0
        assert row_starts.dtype == torch.int32 and row_ends.dtype == torch.int32
        assert bool((row_starts >= 0).all()) and bool((row_ends >= row_starts).all())
        assert bool((row_ends <= logits.shape[1]).all()), "a window past the logits columns"


indexer_topk_prefill_gvr = IndexerTopkPrefillGvr()
