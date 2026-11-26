# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode top-k of the DSA indexer: per query row, the best keys up to its causal end."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: With fp32 logits, a row of 4096 or more columns is split across blocks and
#: merged in a second launch; below that one block owns a row. The split path
#: needs the caller's `radix_aux_*` scratch.
SPLIT_WORK_COLUMNS = 4096

#: Largest `index_topk` that launches in each regime: the kernels take
#: `4 * index_topk` (one block) or `8 * index_topk` (split) bytes of dynamic
#: shared memory on top of their static use, against the 48 KB default. Above
#: it the launch fails at configuration and `indices` keeps its old contents.
#: The call raises nothing, and torch does not report the failure either: the
#: launch goes through the CUDA runtime that libth_common links statically, so
#: its error never reaches torch's own runtime. Found by bisection on
#: sm_100 and again on sm_103 (the split ceiling at 4096 and 9216 columns,
#: and at 32768).
MAX_INDEX_TOPK_SINGLE_BLOCK = 8188
MAX_INDEX_TOPK_SPLIT_WORK = 3020

#: Mistral Large 4's decode shape: next_n 1 (no speculation), its max_seq_len as
#: the scorer's column count, index_topk 2048, and the DSA metadata's radix
#: scratch, `[max_num_sequences = 16, 10, index_topk]`, whatever the batch.
#: `_ML4` is the FP8 target's, whose envelope is 9216.
_ML4 = dict(next_n=1, width=9216, index_topk=2048, aux_rows=16)
#: The NVFP4 target's: the same call at its max_seq_len of 32768, a multiple
#: of the scorer's 256-column alignment, so the scorer's rows are 32768 wide.
_ML4_32K = dict(next_n=1, width=32768, index_topk=2048, aux_rows=16)
#: The NVFP4 tep4 deployment at its 1M envelope (max_seq_len 1048576, chunked
#: prefill): the scorer's rows are 1048576 wide at every decode step, the
#: short prompts' included, since its column count is frozen at max_seq_len.
#: The NVFP4 tp4 deployment at the same envelope makes the same call (scorer
#: width, decode batches, the 16-sequence arena), so these cells cover it too.
#: So does the NVFP4 tep8 deployment (the tep4 deployment at eight ranks, the indexer
#: replicated) at its 1M envelope: the same call on every rank.
_ML4_1M = dict(next_n=1, width=1048576, index_topk=2048, aux_rows=16)
#: The NVFP4 dep4 deployment at the checkpoint's own context: max_seq_len 1048576,
#: the scorer's column count (a multiple of 256, so its rows are contiguous).
#: From 200000 columns every row is split across 10 blocks whatever the row
#: count, so 16 rows fill the 16-sequence scratch exactly; `blocks_per_row` is
#: that split, which the test reads back off the scratch. Only the dep4
#: deployment's cells use it (its own file named it `_ML4_1M`): it is `_ML4_1M` plus
#: `blocks_per_row=10`, the key by which the test routes them to the dep4
#: deployment's drivers.
_ML4_1M_DEP4 = dict(next_n=1, width=1048576, index_topk=2048, aux_rows=16, blocks_per_row=10)
#: DSpark K=3 at the 1M envelope, including overlap/verification headroom.
#: The scorer pads each row to 1048832 columns; 16 requests verify 64 rows.
_ML4_1M_DSPARK = dict(
    next_n=4, width=1048584, row_pad=248, index_topk=2048, aux_rows=64, blocks_per_row=10
)


class IndexerTopkDecode(OpWrapper):
    """Select each decode row's `index_topk` best columns below its causal end.

    Row `b * next_n + t` belongs to sequence `b`'s `t`-th new token. Its
    candidates are columns `[0, end)` with `end = (seq_lens[b] - next_n + t +
    1) // compress_ratio`; nothing past `end` is read. With `n = min(index_topk,
    end)`, the `n` winners land in `indices[r, :n]` as plain column indices and
    `indices[r, n:]` is -1 (a score of -inf is the exception; see the note).

    Each row is a set. The winners are the top-n by score; where scores tie at
    the n-th value, which of the tied columns win is unspecified, but the
    selected scores are the reference's. Their order within `indices[r, :n]`
    is unspecified; on the split path it changes from call to call on
    identical input (20 calls, 20 orders, the set unchanged). Only on the
    single-block path (fp32 rows narrower than `SPLIT_WORK_COLUMNS`) does a
    row with `end <= index_topk` list `0..end-1` in order. Split rows, every
    Mistral Large 4 row included, list the same columns in no fixed order.

    `indices` is a caller-allocated int32 out-parameter written in place; the
    call returns it. The scorer's logits are consumed as they are: the call
    owns the selection, the caller owns the scoring and what reads the indices.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The decode feeds it the paged scorer's output: fp32, one row per
    # generation token, as wide as the engine's max sequence length,
    # row-padded to a multiple of 256 columns. `width` is that column count;
    # `seq_lens` the sequences' KV lengths after this step.
    #
    # Mistral Large 4's (`_ML4`, the FP8 target's 9216 envelope): next_n 1, 1..16
    # sequences, every row split across 4 blocks at width 9216, its scratch the
    # engine's 16-sequence arena. The NVFP4 target's (`_ML4_32K`, its 32768
    # envelope): the same call at width 32768, where a row is split across 10
    # blocks at 1..15 rows and 9 at 16, so its cells drive 15 and 16 rows. At
    # its 1M envelope (`_ML4_1M`, the tep4 deployment's and, at eight ranks, the tep8
    # deployment's): width 1048576, where every row count 1..16 splits 10 ways, so
    # 16 rows fill the 16-sequence arena exactly; seq_lens up to 1048576, the
    # decode graph captured at full length. The tp4 deployment at that envelope
    # adds ends at the per-block spans, the blocks' boundary columns as winners
    # and the graphs captured at the engine's own capture lengths.
    # The dep4 deployment at max_seq_len 1048576 (`_ML4_1M_DEP4`): width 1048576, 10
    # blocks per row at every row count 1..16, the scratch the arena's exact
    # 327680 elements at 16 rows. Its boot captures the long-family decode
    # graphs on a warm-up dummy of max_seq_len - 1 keys beside 1-key padding
    # rows, the first `_ML4_1M_DEP4` cell.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DSpark 1M verify causal/page ends, 64 rows and exact radix arena",
            spec=dict(
                _ML4_1M_DSPARK,
                seq_lens=[
                    3,
                    4,
                    5,
                    31,
                    32,
                    33,
                    2047,
                    2048,
                    2049,
                    32768,
                    65536,
                    131072,
                    524289,
                    1048575,
                    1048576,
                    1048584,
                ],
                batches=[1, 2, 4, 8, 15, 16],
            ),
        ),
        Cell(
            why="DSpark 1M verify mixed tail: 60 rows after 32708 context rows",
            spec=dict(
                _ML4_1M_DSPARK,
                tail_of=32768,
                seq_lens=[1048584, 2048, 2049, 32, 33] * 3,
            ),
        ),
        Cell(
            why="DSpark 1M verify tied winners and signed zero across four causal ends",
            spec=dict(_ML4_1M_DSPARK, seq_lens=[1048584, 2049, 4], scores="ties"),
        ),
        Cell(
            why="DSpark verify four causal rows per request, 64 rows and scorer stride 33024",
            spec=dict(
                next_n=4,
                width=32776,
                row_pad=248,
                index_topk=2048,
                aux_rows=64,
                seq_lens=[
                    3,
                    4,
                    5,
                    6,
                    31,
                    32,
                    33,
                    2047,
                    2048,
                    2049,
                    2050,
                    4096,
                    16385,
                    32767,
                    32768,
                    32776,
                ],
            ),
        ),
        Cell(
            why="DSpark verify graph batches 1..16 x 4, independent causal ends after replay",
            spec=dict(
                next_n=4,
                width=32776,
                row_pad=248,
                index_topk=2048,
                aux_rows=64,
                graph_batches=[1, 2, 4, 8, 16],
                seq_lens=[32776] * 16,
                replays=3,
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, captured decode batches 1, 2, 4, 8, 16: replayed on new "
            "logits and seq_lens up to 32768, the padding dummy's row at seq_len 1",
            spec=dict(_ML4_32K, graph_batches=[1, 2, 4, 8, 16], seq_lens=[32768] * 16, replays=3),
        ),
        Cell(
            why="Mistral Large 4 32768, tied scores at the k-th value at 15 and 16 rows: 16 "
            "levels, all +0.0 (relu's floor), mixed +-0.0; gated on the selected scores",
            spec=dict(
                _ML4_32K,
                scores="ties",
                batches=(15, 16),
                seq_lens=[32768, 32768, 32768, 20490, 20490, 20490, 18441, 18441]
                + [18441, 2049, 2049, 2049, 2048, 100, 1, 30000],
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, mixed step: 15 generation rows at rows 32753..32767 of the "
            "step's 32768-row selection buffer, at width 1048576 (10 blocks per row); the "
            "context rows untouched",
            spec=dict(
                _ML4_1M,
                tail_of=32768,
                seq_lens=[1048576, 1, 2049, 20481, 2048, 20490, 1048575, 3]
                + [524288, 7777, 4096, 1234, 1000000, 6145, 131073],
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, mixed step: 15 generation rows at rows 32753..32767 of the "
            "step's 32768-row selection buffer (10 blocks per row); the context rows "
            "untouched",
            spec=dict(
                _ML4_1M_DEP4,
                tail_of=32768,
                seq_lens=[1048576, 1, 2049, 20481, 2048, 20490, 1048575, 3]
                + [524289, 7777, 4096, 1234, 1000000, 6145, 262145],
            ),
        ),
    )

    note = """
    Radix top-k decode selector for DSA indexer scores (`torch.ops.trtllm.indexer_topk_decode`).
    Covers 32K and 1M widths, CUDA graph replay, DSpark verify steps, tied/extreme fp32 scores.
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
        has no tie at its k-th value; under a tie the selected scores, not the
        columns, are what the reference fixes.
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
        # Only the column stride of 1 is driven. On the split path another stride
        # selects a wrong set, with indices past the row's end.
        assert logits.stride(1) == 1, (
            f"logits rows must be unit-stride in the column dimension; got {logits.stride()}"
        )
        # A causal end past the logits width reads beyond the row. The lengths are
        # device data that a captured call reads only at replay, and reading them
        # here would sync inside the capture, so the check needs an eager call.
        capturing = torch.cuda.is_current_stream_capturing()
        if row_kv_lens is None and seq_lens.numel() and not capturing:
            assert int(seq_lens.min()) >= next_n - 1, (
                "uniform next_n produces a negative causal end; "
                "seq_lens must be at least next_n - 1"
            )
            widest = int(seq_lens.max()) // compress_ratio
            assert widest <= logits.shape[1], (
                f"a causal end of {widest} exceeds the {logits.shape[1]}-column logits; "
                "the kernel reads past the row"
            )


indexer_topk_decode = IndexerTopkDecode()
