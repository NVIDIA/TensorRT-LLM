# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact top-2048 of the key-sharded decode indexer logits, by an MNNVL histogram exchange."""

from __future__ import annotations

from typing import Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._hist_topk import MAX_SHARD, TOPK, shard_size

_TOKEN_BITS = 21


def _spec(group_size, kv_len, pattern="randn", rows=8, next_n=8) -> dict:
    return dict(group_size=group_size, kv_len=kv_len, pattern=pattern, rows=rows, next_n=next_n)


def canonical_rows(rows: torch.Tensor) -> torch.Tensor:
    """Each row's tokens in ascending order, the -1 slots last: the set a row holds, as a tensor."""
    largest = torch.iinfo(rows.dtype).max
    keyed = torch.where(rows < 0, torch.full_like(rows, largest), rows)
    ordered = keyed.sort(dim=1).values
    return torch.where(ordered == largest, torch.full_like(ordered, -1), ordered)


def ordered_keys(logits: torch.Tensor) -> torch.Tensor:
    """The kernel's order-preserving 32-bit key of each FP32 logit, as int64: the bits with the
    sign bit set when it is clear and all bits inverted when it is set, so -0 sorts below +0,
    positive NaNs above +inf and negative NaNs below -inf; the one key equal to the empty word
    (NaN 0x7FFFFFFF) is clamped one below it."""
    bits = logits.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    keys = torch.where(bits >= 0x80000000, (~bits) & 0xFFFFFFFF, bits | 0x80000000)
    return keys.clamp(max=0xFFFFFFFE)


class GlmHistTopk(OpWrapper):
    """Each verify row's exact top-2048 over the TP group's key shards; every rank gets the same
    rows.

    The request of ``kv_lens[0]`` indexer keys is split into contiguous shards of ``S =
    shard_size(kv_len, group_size)`` keys, rank ``r`` holding [r * S, r * S + n_r); ``logits``
    [num_rows, >= S] fp32 holds this rank's shard (its column ``c`` is key r * S + c). Verify row
    ``i`` of ``next_n`` sees the keys [0, kv_len - next_n + i + 1).

    ``out[i]`` [2048] int32 gets the row's top-2048 keys by (logit, key), the larger key first among
    equal logits, in an unspecified order that is the same on every rank; a row of at most 2048
    keys gets [0, n) and then -1. FP32 values are ordered by their bits: -0 below +0, NaNs above
    +inf (positive) or below -inf (negative).

    The exchange runs through ``comm_buffer`` (this rank's unicast view of the group's multicast
    workspace) and ``state`` (``_hist_topk.HistTopkWorkspace``), both private to the op: every rank
    of the group makes the same sequence of calls. Returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="16 ranks (c1): a 530,000-key request, 8 verify rows", spec=_spec(16, 530_000)),
        Cell(
            why="16 ranks: the c1 gen server's 1,048,596-key capacity",
            spec=_spec(16, 1_048_596),
        ),
        Cell(
            why="16 ranks: the largest shard, 98,304 keys of a 1,572,864-key request",
            spec=_spec(16, 16 * MAX_SHARD),
        ),
        Cell(why="4 ranks: a 300,000-key request, random logits", spec=_spec(4, 300_000)),
        Cell(why="4 ranks: ties at the cut, the larger keys taken", spec=_spec(4, 200_000, "ties")),
        Cell(why="4 ranks: every logit equal", spec=_spec(4, 50_000, "equal")),
        Cell(why="4 ranks: +-inf, +-0 and NaN among the logits", spec=_spec(4, 120_000, "special")),
        Cell(
            why="4 ranks: one histogram bin holds more than 65,535 keys of a shard",
            spec=_spec(4, 4 * MAX_SHARD, "saturated"),
        ),
        Cell(
            why="4 ranks: 2,051 keys, rows 0-4 without an exchange, rows 5-7 with",
            spec=_spec(4, 2_051),
        ),
        Cell(why="4 ranks: no row exceeds 2,048 keys", spec=_spec(4, 1_000)),
        Cell(why="4 ranks: one row of a next_n = 1 call", spec=_spec(4, 90_000, rows=1, next_n=1)),
    )

    note = """
    Metadata consumed: none. At c1 the target passes the shard logits of
    ``fp8_paged_mqa_logits`` (the view ``logits[:, :shard capacity]``), the generation request's
    indexer KV length ([1] int32 on the device), its [8, 2048] TopK buffer, the workspace's
    ``comm_buffer`` and ``state``, next_n = 8 and the TP size (16). The workspace is created once
    per process (``_hist_topk.get_hist_topk_workspace``) by every rank of the group, outside CUDA
    graph capture.

    Order and ties: the set is exact; the order within a row after a cut on the top 11 key bits
    follows shared-memory atomics and can change between calls with the same logits.

    The launch is PDL-enabled (``TRTLLM_ENABLE_PDL``): before its wait it initializes its cluster
    barriers and reads ``kv_lens`` (written by the step's metadata preparation, before the logits).
    A wait for a peer longer than 120 s records the error in ``state`` and aborts the kernel.

    The kernel refuses (loudly) a shard above 98,304 keys, more than 8 rows or 16 ranks, logits rows
    not 16-byte aligned, and a workspace of another group.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        kv_lens: torch.Tensor,
        out: torch.Tensor,
        comm_buffer: torch.Tensor,
        state: torch.Tensor,
        next_n: int,
        group_size: int,
    ) -> torch.Tensor:
        torch.ops.trtllm.glm_dsa_hist_topk(
            logits, kv_lens, out, comm_buffer, state, next_n, group_size
        )
        return out

    def reference(
        self,
        logits: Sequence[torch.Tensor],
        kv_lens: torch.Tensor,
        out: torch.Tensor,
        comm_buffer: torch.Tensor,
        state: torch.Tensor,
        next_n: int,
        group_size: int,
    ) -> torch.Tensor:
        """``canonical_rows`` of the expected ``out``, from the group's shard logits in rank order.
        The workspace only carries the exchange."""
        assert len(logits) == group_size, "the reference takes every rank's shard logits"
        kv_len = int(kv_lens[0])
        size = shard_size(kv_len, group_size)
        rows = out.shape[0]
        expected = torch.full((rows, TOPK), -1, dtype=torch.int32, device=out.device)
        whole = []
        for rank, shard in enumerate(logits):
            n = min(max(kv_len - rank * size, 0), size)
            whole.append(shard[:rows, :n].to(out.device))
        full = torch.cat(whole, dim=1)
        for row in range(rows):
            window = kv_len - next_n + row + 1
            if window <= TOPK:
                count = max(window, 0)
                expected[row, :count] = torch.arange(count, dtype=torch.int32, device=out.device)
                continue
            tokens = torch.arange(window, dtype=torch.int64, device=out.device)
            composite = (ordered_keys(full[row, :window]) << _TOKEN_BITS) | tokens
            top = composite.topk(TOPK).values & ((1 << _TOKEN_BITS) - 1)
            expected[row] = top.sort().values.int()
        return expected

    def is_valid(
        self,
        logits: torch.Tensor,
        kv_lens: torch.Tensor,
        out: torch.Tensor,
        comm_buffer: torch.Tensor,
        state: torch.Tensor,
        next_n: int,
        group_size: int,
    ) -> None:
        # The kernel reads kv_lens[0] only: a batch's lengths would select request 0's silently.
        assert kv_lens.numel() == 1, (
            "kv_lens must hold the one generation request's indexer KV length"
        )


hist_topk = GlmHistTopk()
