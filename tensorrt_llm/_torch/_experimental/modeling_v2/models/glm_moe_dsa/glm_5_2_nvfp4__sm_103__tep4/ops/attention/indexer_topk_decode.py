# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode top-k of the DSA indexer: per query row, the best keys up to its causal end."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: With fp32 logits, a row of 4096 or more columns is split across blocks and
#: merged in a second pass; below that one block owns a row. The split path
#: needs the caller's `radix_aux_*` scratch.
SPLIT_WORK_COLUMNS = 4096

#: Largest `index_topk` that launches in each regime: the kernels take
#: `4 * index_topk` (one block) or `8 * index_topk` (split) bytes of dynamic
#: shared memory on top of their static use, against the 48 KB default. Above
#: it the launch fails at configuration, `indices` keeps its old contents, and
#: nothing raises in a release build. Found by bisection on sm_100.
MAX_INDEX_TOPK_SINGLE_BLOCK = 8188
MAX_INDEX_TOPK_SPLIT_WORK = 3020


class IndexerTopkDecode(OpWrapper):
    """Select each decode row's `index_topk` best columns below its causal end.

    Row `b * next_n + t` belongs to sequence `b`'s `t`-th new token. Its
    candidates are columns `[0, end)` with `end = (seq_lens[b] - next_n + t +
    1) // compress_ratio`; nothing past `end` is read. With `n = min(index_topk,
    end)`, the `n` winners land in `indices[r, :n]` as plain column indices and
    `indices[r, n:]` is -1. The set of winners is the top-n by score; their
    order is unspecified, except that a row with `end <= index_topk` lists
    `0..end-1` in order.

    `indices` is a caller-allocated int32 out-parameter written in place; the
    call returns it.
    """

    ARCHS = frozenset({Arch.SM_103})

    # GLM's decode feeds it the paged scorer's output: fp32, one row per
    # generation token (32 sequences x MTP 2 at most), as wide as the engine's
    # max sequence length, row-padded to a multiple of 256 columns. `width` is
    # that column count; `seq_lens` the sequences' KV lengths after this step.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one sequence, one token, single-block regime (width < 4096)",
            spec=dict(next_n=1, seq_lens=[3000], width=3072, index_topk=2048),
        ),
        Cell(
            why="contexts shorter than top-k: identity rows plus -1 padding",
            spec=dict(next_n=2, seq_lens=[2, 700, 2049], width=3072, index_topk=2048),
        ),
        Cell(
            why="the full decode batch, 32 x MTP 2, split-work regime",
            spec=dict(
                next_n=2,
                seq_lens=[2100 + 190 * i for i in range(32)],
                width=8192,
                index_topk=2048,
            ),
        ),
        Cell(
            why="long contexts: 64K columns, split across several blocks per row",
            spec=dict(next_n=1, seq_lens=[65536, 40000, 12345], width=65536, index_topk=2048),
        ),
        Cell(
            why="a row-strided logits view, as the paged scorer returns it",
            spec=dict(next_n=2, seq_lens=[5000, 2500], width=5000, index_topk=2048, row_pad=256),
        ),
    )

    note = """
    Only fp32 logits are certified. bf16/fp16 logits take a separate entry
    with no split-work path.

    `compress_ratio` other than 1 and `row_kv_lens` (ragged rows) belong to
    other DSA models; GLM passes neither and neither is certified here.

    Logits are read at `logits.stride(0)`, so the paged scorer's row-padded
    view is a legal operand without a copy.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        index_topk: int = 2048,
        compress_ratio: int = 1,
        radix_aux_indices: Optional[torch.Tensor] = None,
        radix_aux_logits: Optional[torch.Tensor] = None,
        row_kv_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch.ops.trtllm.indexer_topk_decode(
            logits,
            seq_lens,
            indices,
            next_n,
            index_topk,
            compress_ratio,
            radix_aux_indices,
            radix_aux_logits,
            row_kv_lens,
        )
        return indices

    def reference(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        index_topk: int = 2048,
        compress_ratio: int = 1,
        radix_aux_indices: Optional[torch.Tensor] = None,
        radix_aux_logits: Optional[torch.Tensor] = None,
        row_kv_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The winners per row, as a fresh `[rows, index_topk]` int32 tensor.

        `indices` and the two scratch buffers are not read. Rows are judged on
        fp32 values with `torch.topk`, so the answer is exact only when a row
        has no tie at its k-th value; the test gates on set equality on that basis.
        """
        rows = logits.shape[0]
        out = torch.full((rows, index_topk), -1, dtype=torch.int32, device=logits.device)
        lens = seq_lens.tolist()
        per_row = None if row_kv_lens is None else row_kv_lens.tolist()
        for r in range(rows):
            kv = per_row[r] if per_row is not None else lens[r // next_n] - next_n + r % next_n + 1
            end = max(0, kv // compress_ratio)
            if end <= index_topk:
                out[r, :end] = torch.arange(end, dtype=torch.int32, device=logits.device)
            else:
                out[r] = torch.topk(logits[r, :end].float(), index_topk).indices.to(torch.int32)
        return out

    def is_valid(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        index_topk: int = 2048,
        compress_ratio: int = 1,
        radix_aux_indices: Optional[torch.Tensor] = None,
        radix_aux_logits: Optional[torch.Tensor] = None,
        row_kv_lens: Optional[torch.Tensor] = None,
    ) -> None:
        split = logits.dtype == torch.float32 and logits.shape[1] >= SPLIT_WORK_COLUMNS
        ceiling = MAX_INDEX_TOPK_SPLIT_WORK if split else MAX_INDEX_TOPK_SINGLE_BLOCK
        assert 1 <= index_topk <= ceiling, (
            f"index_topk must be in [1, {ceiling}] for a {logits.shape[1]}-column operand; "
            f"got {index_topk}, which fails to launch and raises nothing"
        )
        # The op checks `indices.size(1) >= index_topk`, but the kernel writes at
        # a row stride of `index_topk`: a wider buffer gets its rows packed into
        # the first few and the rest left at their old contents.
        assert indices.shape[1] == index_topk, (
            f"indices must be exactly {index_topk} columns wide (the kernel writes it "
            f"flat at a row stride of index_topk); got shape {tuple(indices.shape)}"
        )
        # `seq_lens` is read through an int32 pointer with no dtype check; int64
        # lengths are read as pairs of 32-bit halves.
        assert seq_lens.dtype == torch.int32, f"seq_lens must be int32, got {seq_lens.dtype}"
        # Only the column stride of 1 is driven; the split kernel's index
        # arithmetic for other strides was observed to select a wrong set.
        assert logits.stride(1) == 1, (
            f"logits rows must be unit-stride in the column dimension; got {logits.stride()}"
        )
        # A causal end past the logits width reads beyond the row.
        if row_kv_lens is None and seq_lens.numel():
            widest = int(seq_lens.max()) // compress_ratio
            assert widest <= logits.shape[1], (
                f"a causal end of {widest} exceeds the {logits.shape[1]}-column logits; "
                "the kernel reads past the row"
            )


indexer_topk_decode = IndexerTopkDecode()
