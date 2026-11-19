# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer prefill top-k: the top-k columns of each row's key window (CUDA)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: K3.1's DSA top-k width (dsa_config.index_topk).
INDEX_TOPK = 2048


def _spec(requests, **overrides) -> dict:
    """``requests``: (cached keys, new query rows) per request of one prefill chunk; the chunk's
    key matrix holds each request's cached and new keys back to back."""
    spec = dict(requests=tuple(requests), q_rows=None, index_topk=INDEX_TOPK, logits_pad=0)
    spec.update(overrides)
    return spec


class IndexerTopkPrefill(OpWrapper):
    """Writes into ``indices`` (int32 [M, index_topk], contiguous) the ``index_topk`` largest
    columns of each row's window ``[row_starts[r], row_ends[r])`` of ``logits`` (fp32 [M, N], any
    row stride), as offsets from ``row_starts[r]``, in unspecified order. A window of at most
    ``index_topk`` columns yields ``0 .. len - 1`` in order, then -1. Columns outside the window
    are never read. Returns None.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one-token prefill: a one-column window, 2047 slots of -1",
            spec=_spec([(0, 1)]),
        ),
        Cell(
            why="a prompt at the top-k edge: windows of 1..2100 columns (the short-row shortcut, "
            "then 52 full rows)",
            spec=_spec([(0, 2100)]),
        ),
        Cell(
            why="three packed requests: windows start at each request's first key in the chunk",
            spec=_spec([(0, 100), (0, 3000), (0, 700)]),
        ),
        Cell(
            why="chunked-prefill continuation: 512 rows over an 8192-key cached prefix",
            spec=_spec([(8192, 512)]),
        ),
        Cell(
            why="a packed chunk of cached requests (prefix reuse): 3 requests, 2k-20k cached keys",
            spec=_spec([(2000, 64), (20000, 300), (5000, 1)]),
        ),
        Cell(
            why="a query tile of a long request: rows 2000..2999 of a 4096-row chunk",
            spec=_spec([(0, 4096)], q_rows=(2000, 3000)),
        ),
        Cell(
            why="DeepGEMM's row-padded logits: a row stride past the chunk's key count",
            spec=_spec([(0, 2500)], logits_pad=60),
        ),
        Cell(
            why="a 12800-row chunk: rows 12288.. take the radix-sort kernel",
            spec=_spec([(0, 12800)]),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).

    The K3.1 general target reaches this on architectures other than SM 10.3 (where every target
    takes indexer_topk_prefill_gvr) from K31Indexer.forward_prepared, once per fp8_mqa_logits query
    tile of a prefill chunk: logits is that tile's [rows, chunk keys] output computed with
    clean_logits=False, row_starts / row_ends the tile's slice of the chunk's cu_seqlen_ks /
    cu_seqlen_ke, indices the tile's rows of the step's top-k buffer, index_topk 2048. Only the
    window is read, so the uncleaned logits outside it are never seen (the cells fill them with
    +inf, which any read would select). The offsets are request-local key positions: a window
    starts at the request's first key in the chunk. The order of the written indices varies from
    run to run, so the cells compare each row as a set over distinct logits; logits tied at the
    k-th value may be chosen either way. No workspace.

    ``indices`` must be exactly ``index_topk`` wide and contiguous: the kernel writes row r at
    ``r * index_topk`` whatever the tensor's row stride.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        index_topk: int,
    ) -> None:
        torch.ops.trtllm.indexer_topk_prefill(logits, row_starts, row_ends, indices, index_topk)

    def reference(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        index_topk: int,
    ) -> None:
        """torch.topk over each window (outside it masked to -inf), written into ``indices`` in
        descending logit order; short windows get the identity then -1."""
        cols = torch.arange(logits.shape[1], device=logits.device)
        starts = row_starts.long().unsqueeze(1)
        ends = row_ends.long().unsqueeze(1)
        inside = (cols >= starts) & (cols < ends)
        masked = torch.where(inside, logits.float(), float("-inf"))
        k = min(index_topk, logits.shape[1])
        top = masked.topk(k, dim=1).indices - starts
        lengths = (row_ends - row_starts).long()
        slots = torch.arange(index_topk, device=logits.device).unsqueeze(0)
        short = torch.where(slots < lengths.unsqueeze(1), slots, -1)
        out = torch.full_like(short, -1)
        out[:, :k] = top
        out = torch.where((lengths > index_topk).unsqueeze(1), out, short)
        indices[:, :index_topk] = out.to(indices.dtype)

    def is_valid(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        index_topk: int,
    ) -> None:
        # The kernel writes row r at r * index_topk: a wider or strided buffer gets its rows
        # interleaved.
        assert indices.is_contiguous() and indices.shape[1] == index_topk, (
            f"indices must be a contiguous [rows, {index_topk}] buffer, got shape "
            f"{tuple(indices.shape)} stride {indices.stride()}"
        )
        if row_ends.numel():
            # A window past the logits row reads the next row (or another allocation).
            assert int(row_ends.max()) <= logits.shape[1], "a row window ends past the logits row"
            # An inverted window gives a negative length, and the -1 fill then writes before
            # the row's output slots.
            assert bool((row_starts <= row_ends).all()), "a row window starts after it ends"


indexer_topk_prefill = IndexerTopkPrefill()
