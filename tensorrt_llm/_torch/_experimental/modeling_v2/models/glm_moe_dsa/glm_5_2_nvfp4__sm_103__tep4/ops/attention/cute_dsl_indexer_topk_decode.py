# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode top-k of the DSA indexer in one CuTe DSL launch: per query row, the best keys."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import self_attn_indexer_topk_decode as _kernel

#: The compiled handle type: the kernel's own, which records its compile key.
CompiledIndexerTopkDecode = _kernel.CompiledIndexerTopKDecode


def build(
    device_id: int, *, max_columns: int, top_k: int, next_n: int
) -> CompiledIndexerTopkDecode:
    """Compile one `next_n` width and launch it once on zeros, ahead of any capture.

    `max_columns` is the widest logits row the handle accepts: the engine's
    `max_seq_len`, which is also the paged scorer's row width. Raises
    `ValueError` when `max_columns <= top_k` or `top_k != 2048`: the kernel is
    built for rows longer than top-k only. One compile of a few seconds plus a
    `[next_n, max_columns]` fp32 scratch launch; cached per process.
    """
    return _kernel.warmup_indexer_topk_decode(
        max_num_columns=int(max_columns),
        top_k=int(top_k),
        next_n=int(next_n),
        device_id=int(device_id),
    )


def _ordered_key(values: torch.Tensor) -> torch.Tensor:
    """fp32 values as int64 keys in the order of their ordered bit patterns (-0 < +0)."""
    bits = values.contiguous().view(torch.int32).long()
    return torch.where(bits < 0, -(bits & 0x7FFFFFFF) - 1, bits)


def select_rows(
    logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, top_k: int
) -> torch.Tensor:
    """`[rows, top_k]` int32 window-local winners of `logits[r, starts[r]:ends[r]]`.

    A window no longer than `top_k` is the identity `0..len-1` then -1. A longer
    one holds its `top_k` largest values, ties broken toward the lower column,
    in descending order. Shared with the prefill entry's reference.
    """
    rows, width = logits.shape
    dev = logits.device
    out = torch.full((rows, top_k), -1, dtype=torch.int32, device=dev)
    lengths = (ends - starts).clamp_min(0)
    cols = torch.arange(width, device=dev)
    short = lengths <= top_k
    if bool(short.any()):
        ident = torch.arange(top_k, device=dev, dtype=torch.int32).expand(int(short.sum()), -1)
        out[short] = torch.where(ident < lengths[short, None], ident, -1).to(torch.int32)
    long_rows = (~short).nonzero().flatten()
    # Bounded batches keep the int64 key matrix small at 64K columns.
    for chunk in long_rows.split(64):
        if chunk.numel() == 0:
            continue
        key = _ordered_key(logits[chunk].float())
        inside = (cols >= starts[chunk, None]) & (cols < ends[chunk, None])
        key = torch.where(inside, key, torch.iinfo(torch.int64).min)
        order = torch.sort(key, dim=1, descending=True, stable=True).indices[:, :top_k]
        out[chunk] = (order - starts[chunk, None]).to(torch.int32)
    return out


class CuteDslIndexerTopkDecode(OpWrapper):
    """Select each decode row's `top_k` best columns below its causal end.

    Row `r = b * next_n + t` is sequence `b`'s `t`-th new token. Its candidates
    are columns `[0, end)` with `end = clamp(seq_lens[b] - next_n + t + 1, 0,
    logits.shape[1])`. When `end <= top_k`, `indices[r]` is `0..end-1` in order
    then -1. Otherwise it holds the `top_k` columns with the largest values,
    ties broken toward the lower column (fp32 bit order, so -0 ranks below +0);
    the order of the winners in the row is unspecified.

    `indices` (`[rows, top_k]` int32) is written in place and returned.
    `top_k` and `next_n` are keyword-only: the stock trtllm op takes `(next_n,
    index_topk)` positionally and this kernel `(top_k, next_n)`, both small
    ints, so a positional swap would select the wrong positions.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target scores every generation token of a decode step with the paged
    # scorer and selects here: up to 32 sequences x MTP 2, fp32, each row as
    # wide as the engine's max_seq_len (the compile key), possibly row-padded.
    # It builds next_n 1 and 2. `seq_lens` are KV lengths after this step.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one sequence, one token, a context longer than top-k",
            spec=dict(next_n=1, max_columns=8192, seq_lens=[3000], width=8192),
        ),
        Cell(
            why="the full decode batch: 32 sequences x MTP 2 at 8192 columns",
            spec=dict(
                next_n=2,
                max_columns=8192,
                seq_lens=[2100 + 190 * i for i in range(32)],
                width=8192,
            ),
        ),
        Cell(
            why="contexts at and below top-k, and a first MTP row with no key at all",
            spec=dict(next_n=2, max_columns=8192, seq_lens=[1, 2, 700, 2049, 2050], width=8192),
        ),
        Cell(
            why="long contexts: 64K columns",
            spec=dict(next_n=1, max_columns=65536, seq_lens=[65536, 40000, 12345], width=65536),
        ),
        Cell(
            why="a row-padded logits view narrower than the compiled width",
            spec=dict(next_n=2, max_columns=8192, seq_lens=[5000, 2500], width=5000, row_pad=256),
        ),
        Cell(
            why="heavy ties at the k-th value: the lower column wins",
            spec=dict(next_n=2, max_columns=8192, seq_lens=[8192, 6000], width=8192, levels=7),
        ),
    )

    note = """
    `build(device_id, max_columns=, top_k=, next_n=)` compiles and warms one
    `next_n`; the target builds next_n 1 and 2 on its first forward (with
    `max_columns` = the engine's max_seq_len) and passes the handle for the
    step's width as `compiled`. A width with no handle -- `max_seq_len <=
    index_topk`, a failed build, or another MTP width -- takes the stock
    `indexer_topk_decode` op at the call site.

    The handle rejects, loudly, a different `top_k` / `next_n`, a logits row
    wider than `max_columns`, and non-int32 or non-contiguous `seq_lens` /
    `indices`. Logits are read at `logits.stride(0)`, so the paged scorer's
    row-padded view needs no copy.

    Unlike the stock op the kernel clamps a causal end past the logits width
    to the width instead of reading past the row; `is_valid` refuses such a
    call because the answer is a top-k of the wrong range.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        *,
        top_k: int,
        next_n: int,
        compiled: CompiledIndexerTopkDecode,
    ) -> torch.Tensor:
        return _kernel.indexer_topk_decode_run(
            logits, seq_lens, indices, compiled_fn=compiled, top_k=top_k, next_n=next_n
        )

    def reference(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        *,
        top_k: int,
        next_n: int,
        compiled: CompiledIndexerTopkDecode,
    ) -> torch.Tensor:
        """A fresh `[rows, top_k]` int32 selection; `indices` and `compiled` are not read.

        Exact under the kernel's tie rule, so the test compares sets with no slack.
        """
        rows, width = logits.shape
        r = torch.arange(rows, device=logits.device)
        ends = (seq_lens.long()[r // next_n] - next_n + r % next_n + 1).clamp(0, width)
        return select_rows(logits, torch.zeros_like(ends), ends, top_k)

    def is_valid(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        *,
        top_k: int,
        next_n: int,
        compiled: CompiledIndexerTopkDecode,
    ) -> None:
        # The kernel clamps a causal end past the row to the row width and
        # returns a top-k of the truncated range without complaint.
        if seq_lens.numel():
            widest = int(seq_lens.max())
            assert widest <= logits.shape[1], (
                f"a causal end of {widest} exceeds the {logits.shape[1]}-column logits; "
                "the kernel selects from a truncated row"
            )


cute_dsl_indexer_topk_decode = CuteDslIndexerTopkDecode()
