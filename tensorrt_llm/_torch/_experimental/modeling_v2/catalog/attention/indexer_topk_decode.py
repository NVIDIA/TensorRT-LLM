# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode top-k of the DSA indexer: per query row, the best keys up to its causal end."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: With fp32 logits, a row of 4096 or more columns is split across blocks and
#: merged in a second launch; below that one block owns a row. The split path
#: needs the caller's `radix_aux_*` scratch.
SPLIT_WORK_COLUMNS = 4096

#: Largest `index_topk` that launches in each regime: the kernels take
#: `4 * index_topk` (one block) or `8 * index_topk` (split) bytes of dynamic
#: shared memory on top of their static use, against the 48 KB default. Above
#: it the launch fails at configuration and `indices` keeps its old contents.
#: The call raises nothing; the error stays pending and the next torch kernel
#: launch raises it as `CUDA error: invalid argument`. Found by bisection on
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
#: The NVFP4 tep4 target at its 1M envelope (max_seq_len 1048576, chunked
#: prefill): the scorer's rows are 1048576 wide at every decode step, the
#: short prompts' included, since its column count is frozen at max_seq_len.
#: The NVFP4 tp4 target at the same envelope makes the same call (scorer
#: width, decode batches, the 16-sequence arena), so these cells cover it too.
#: So does the NVFP4 tep8 target (the tep4 target at eight ranks, the indexer
#: replicated) at its 1M envelope: the same call on every rank.
_ML4_1M = dict(next_n=1, width=1048576, index_topk=2048, aux_rows=16)
#: The NVFP4 dep4 target at the checkpoint's own context: max_seq_len 1048576,
#: the scorer's column count (a multiple of 256, so its rows are contiguous).
#: From 200000 columns every row is split across 10 blocks whatever the row
#: count, so 16 rows fill the 16-sequence scratch exactly; `blocks_per_row` is
#: that split, which the test reads back off the scratch. Only the dep4
#: target's cells use it (its own file named it `_ML4_1M`): it is `_ML4_1M` plus
#: `blocks_per_row=10`, the key by which the test routes them to the dep4
#: target's drivers.
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
    # its 1M envelope (`_ML4_1M`, the tep4 target's and, at eight ranks, the tep8
    # target's): width 1048576, where every row count 1..16 splits 10 ways, so
    # 16 rows fill the 16-sequence arena exactly; seq_lens up to 1048576, the
    # decode graph captured at full length. The tp4 target at that envelope
    # adds ends at the per-block spans, the blocks' boundary columns as winners
    # and the graphs captured at the engine's own capture lengths.
    # The dep4 target at max_seq_len 1048576 (`_ML4_1M_DEP4`): width 1048576, 10
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
        *(
            Cell(
                why=f"DSpark 1M verify graph batch {batch}: boot, long replay and DEP padding",
                spec=dict(
                    _ML4_1M_DSPARK,
                    graph_batches=[batch],
                    seq_lens=[1048583] + [4] * 15,
                    replay_lens=(
                        [
                            1048584,
                            1048576,
                            1048575,
                            524289,
                            131073,
                            65537,
                            32769,
                            2049,
                            2048,
                            257,
                            65,
                            33,
                            32,
                            5,
                            4,
                            4,
                        ],
                        [1048576] + [4] * 15,
                    ),
                    replays=2,
                ),
            )
            for batch in (1, 2, 4, 8, 16)
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
            why="DSpark 1M verify split-edge winners: all 64 rows reach the last block",
            spec=dict(_ML4_1M_DSPARK, seq_lens=[1048584] * 16, scores="edges"),
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
            why="DSpark verify mixed topk: 60 rows after 32708 prompt rows",
            spec=dict(
                next_n=4,
                width=32776,
                row_pad=248,
                index_topk=2048,
                aux_rows=64,
                tail_of=32768,
                seq_lens=[32776, 2048, 2049, 32, 33] * 3,
            ),
        ),
        Cell(
            why="Mistral Large 4 batch 16 at width 9216, 4 blocks per row: seq_lens 1..9216, "
            "the 2048/2049 top-k edge, the 4-block edges 8192..8196",
            spec=dict(
                _ML4,
                seq_lens=[1, 2, 5, 2047, 2048, 2049, 4097, 6145]
                + [8191, 8192, 8193, 8195, 8196, 9000, 9215, 9216],
            ),
        ),
        Cell(
            why="Mistral Large 4 every row count 1..16 (mixed steps' generation rows; the default "
            "graph list 1..16, unpadded): the leading b rows of one batch",
            spec=dict(
                _ML4,
                batches=tuple(range(1, 17)),
                seq_lens=[9216, 2049, 8193, 1, 5000, 2048, 8196, 777]
                + [9215, 4097, 8192, 3, 6144, 2047, 8195, 7000],
            ),
        ),
        Cell(
            why="Mistral Large 4 captured decode batches 1, 2, 4, 8, 16: replayed on new logits "
            "and seq_lens, the padding dummy's row at seq_len 1",
            spec=dict(_ML4, graph_batches=(1, 2, 4, 8, 16), seq_lens=[9216] * 16, replays=3),
        ),
        Cell(
            why="Mistral Large 4 tied scores at the k-th value: 16 levels, all +0.0 (relu's floor), "
            "mixed +-0.0; gated on the selected scores",
            spec=dict(
                _ML4,
                scores="ties",
                seq_lens=[9216, 9216, 9216, 8196, 8196, 8196, 5000, 5000]
                + [5000, 2049, 2049, 2049, 2048, 100, 1, 9000],
            ),
        ),
        Cell(
            why="Mistral Large 4 scores past fp16's range and under its resolution, both signs: "
            "the fp16 first pass bins them together",
            spec=dict(
                _ML4,
                scores="wide",
                seq_lens=[9216, 9216, 9215, 8196, 8195, 8192, 6000, 4097]
                + [2049, 2048, 1000, 1, 9216, 7000, 3000, 9100],
            ),
        ),
        Cell(
            why="Mistral Large 4 mixed step: 15 generation rows at rows 9201..9215 of the "
            "step's 9216-row selection buffer; the context rows untouched",
            spec=dict(
                _ML4,
                tail_of=9216,
                seq_lens=[9216, 1, 2049, 8196, 2048, 5000, 9215, 3]
                + [8192, 7777, 4096, 1234, 9000, 6145, 8193],
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, batch 16 at width 32768, 9 blocks per row: seq_lens "
            "1..32768, the 2048/2049 top-k edge, the 9216 and 16384 marks",
            spec=dict(
                _ML4_32K,
                seq_lens=[1, 2, 5, 2047, 2048, 2049, 8192, 8196]
                + [9216, 16384, 16385, 24576, 30000, 32767, 32768, 12345],
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, every row count 1..16 (10 blocks per row to 15 rows, 9 "
            "at 16; mixed steps' generation rows, the default graph list unpadded): the "
            "per-block edges 20480/20481/20489/20490 and 18432/18433/18440/18441",
            spec=dict(
                _ML4_32K,
                batches=tuple(range(1, 17)),
                seq_lens=[20480, 20481, 20489, 20490, 32768, 2049, 1, 18432]
                + [18433, 18440, 18441, 2048, 32767, 16385, 3, 30000],
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
            why="Mistral Large 4 32768, scores past fp16's range and under its resolution at 15 "
            "and 16 rows, both signs: the fp16 first pass bins them together",
            spec=dict(
                _ML4_32K,
                scores="wide",
                batches=(15, 16),
                seq_lens=[32768, 32768, 32767, 20490, 20489, 20481, 20480, 18441]
                + [18433, 18432, 2049, 2048, 1000, 1, 16385, 9216],
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, mixed step: 15 generation rows at rows 32753..32767 of "
            "the step's 32768-row selection buffer (10 blocks per row); the context rows "
            "untouched",
            spec=dict(
                _ML4_32K,
                tail_of=32768,
                seq_lens=[32768, 1, 2049, 20481, 2048, 20490, 32767, 3]
                + [18441, 7777, 4096, 1234, 30000, 6145, 16385],
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, batch 16 at width 1048576, 10 blocks per row filling the "
            "16-sequence arena: seq_lens 1..1048576, the 2048/2049 top-k edge, the 10-block "
            "edges 20480/20481/20490, the 32768 and 200000 marks",
            spec=dict(
                _ML4_1M,
                seq_lens=[1, 2, 2047, 2048, 2049, 20480, 20481, 20490]
                + [32768, 32769, 131072, 200000, 524288, 1000000, 1048575, 1048576],
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, every row count 1..16 at width 1048576 (10 blocks per row at "
            "each; mixed steps' generation rows, the default graph list unpadded): full-length "
            "rows beside short ones",
            spec=dict(
                _ML4_1M,
                batches=tuple(range(1, 17)),
                seq_lens=[1048576, 2049, 20481, 1, 524288, 2048, 20490, 777]
                + [1048575, 32769, 200001, 3, 131072, 2047, 1000000, 65536],
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, captured decode batches 1, 2, 4, 8, 16 at width 1048576, "
            "captured at full length: replayed on new logits and seq_lens up to 1048576, the "
            "padding dummy's row at seq_len 1",
            spec=dict(_ML4_1M, graph_batches=[1, 2, 4, 8, 16], seq_lens=[1048576] * 16, replays=3),
        ),
        Cell(
            why="Mistral Large 4 1M, tied scores at the k-th value at 15 and 16 rows: 16 levels, "
            "all +0.0 (relu's floor), mixed +-0.0, up to 104863 tied columns a block; gated on "
            "the selected scores",
            spec=dict(
                _ML4_1M,
                scores="ties",
                batches=(15, 16),
                seq_lens=[1048576, 1048576, 1048576, 20490, 20490, 20490, 20481, 20481]
                + [20481, 2049, 2049, 2049, 2048, 100, 1, 524288],
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, scores past fp16's range and under its resolution at 15 and 16 "
            "rows, both signs, at width 1048576: the fp16 first pass bins them together",
            spec=dict(
                _ML4_1M,
                scores="wide",
                batches=(15, 16),
                seq_lens=[1048576, 1048576, 1048575, 524289, 200000, 20490, 20481, 20480]
                + [32769, 2049, 2048, 1000, 1, 131073, 9216, 65537],
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
            why="Mistral Large 4 1M, both sides of each 10-block edge at 16 rows: ends 20480/20481 "
            "(the last block starts selecting) and 20489/20490 (all 10 do), beside full-length "
            "rows whose 104857-column blocks start off the 16-byte grid",
            spec=dict(
                _ML4_1M,
                seq_lens=[20479, 20480, 20481, 20482, 20488, 20489, 20490, 20491]
                + [1048576, 1048575, 1048570, 1048569, 104857, 209715, 2048, 2049],
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, the per-block spans at batch 16, width 1048576: ends 104857/104858 "
            "(a full-length row's block width) and 943713 (where its last block starts), 12345 "
            "(every block under 2048 columns), 199999/200000 as ends, the 2048/2049 top-k edge, "
            "1048575/1048576",
            spec=dict(
                _ML4_1M,
                seq_lens=[1, 2047, 2048, 2049, 32768, 32769, 65536, 104857]
                + [104858, 199999, 200000, 524288, 943713, 1048575, 1048576, 12345],
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, the 10 blocks' boundary columns as winners at 16 rows: each "
            "block's first and last 4 columns scored at the top or straddling the 2048th score, "
            "blocks of floor(end / 10) = 0..3 mod 4 columns, so starts on and off the 16-byte grid",
            spec=dict(
                _ML4_1M,
                scores="edges",
                seq_lens=[1048576, 1048575, 1048570, 1048569, 1048560, 1048559, 1048550, 1048549]
                + [524288, 200000, 104857, 209715, 32769, 20490, 20491, 1000003],
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, graphs captured as the engine captures them: batches 1, 2, 4, "
            "8, 16 with one row at 1048575 and the rest at 1, replayed at the 2048/2049 and "
            "10-block edges, 1048575 and 1048576, the padding dummy's row at seq_len 1",
            spec=dict(
                _ML4_1M,
                graph_batches=[1, 2, 4, 8, 16],
                seq_lens=[1048575] + [1] * 15,
                replay_lens=(
                    [2049, 1048576, 2048, 20481, 20490, 1048575, 2047, 20480]
                    + [104857, 200000, 524288, 20489, 1, 32768, 1000000, 1048576],
                    [1048576, 2048, 1048575, 2049, 20490, 1, 20481, 20480]
                    + [1048576, 1048575, 1048576, 2049, 2048, 20489, 20491, 1048576],
                ),
                replays=2,
            ),
        ),
        Cell(
            why="Mistral Large 4 1M boot: decode graphs 1, 2, 4, 8, 16 captured on the warm-up "
            "dummy at max_seq_len - 1 = 1048575 keys ahead of 1-key padding rows (the "
            "long-family capture), the warm-up call gated; replayed on new logits and "
            "seq_lens up to 1048576",
            spec=dict(
                _ML4_1M_DEP4,
                graph_batches=(1, 2, 4, 8, 16),
                seq_lens=[1048575] + [1] * 15,
                replays=3,
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, batch 16 at width 1048576, 10 blocks per row (the arena "
            "full): seq_lens 1..1048576, the 2048/2049 top-k edge, 32768/32769, "
            "199999/200000, 1048575/1048576",
            spec=dict(
                _ML4_1M_DEP4,
                seq_lens=[1, 2, 5, 2047, 2048, 2049, 32768, 32769]
                + [131072, 199999, 200000, 262145, 524288, 777777, 1048575, 1048576],
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, every row count 1..16 (10 blocks per row at each, 16 "
            "included; mixed steps' generation rows, the default graph list unpadded): the "
            "per-block edges 20480/20481/20489/20490 and 204800/204801/204810/204811",
            spec=dict(
                _ML4_1M_DEP4,
                batches=tuple(range(1, 17)),
                seq_lens=[20480, 20481, 20489, 20490, 1048576, 2049, 1, 204800]
                + [204801, 204810, 204811, 2048, 1048575, 524289, 3, 1000000],
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, captured decode batches 1, 2, 4, 8, 16 at 1048576 keys: "
            "replayed on new logits and seq_lens up to 1048576, the padding dummy's row at "
            "seq_len 1",
            spec=dict(
                _ML4_1M_DEP4, graph_batches=[1, 2, 4, 8, 16], seq_lens=[1048576] * 16, replays=3
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, attention DP's one live row: graphs 1, 2, 4, 8, 16 captured "
            "at one 1048576-key row ahead of 1-key dummies, replayed with the live row at "
            "1048576, 1048575 and a drawn length, the rest of the batch seq_len-1 dummies",
            spec=dict(
                _ML4_1M_DEP4,
                graph_batches=[1, 2, 4, 8, 16],
                live=1,
                seq_lens=[1048576] + [1] * 15,
                replays=3,
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, tied scores at the k-th value at 15 and 16 rows: 16 levels, "
            "all +0.0 (relu's floor), mixed +-0.0; gated on the selected scores",
            spec=dict(
                _ML4_1M_DEP4,
                scores="ties",
                batches=(15, 16),
                seq_lens=[1048576, 1048576, 1048576, 20490, 20490, 20490, 204811, 204811]
                + [204811, 2049, 2049, 2049, 2048, 100, 1, 524289],
            ),
        ),
        Cell(
            why="Mistral Large 4 1M, scores past fp16's range and under its resolution at 15 and "
            "16 rows, both signs: the fp16 first pass bins them together",
            spec=dict(
                _ML4_1M_DEP4,
                scores="wide",
                batches=(15, 16),
                seq_lens=[1048576, 1048576, 1048575, 20490, 20489, 20481, 20480, 204811]
                + [204801, 204800, 2049, 2048, 1000, 1, 524289, 262145],
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
    DSpark's 1M cells verify four tokens per request at width 1048584 and
    stride 1048832, up to 64 rows. Every row splits 10 ways, including graph
    batches 1/2/4/8/16 and the 60-row tail after 32708 prompt rows. Boot
    capture starts at one 1048583-key request beside four-key dummies;
    replays reach 1048584 and return the eager call's sets on new logits.
    The exact scratch is [64, 10, 2048] (1310720 elements per buffer): every
    gated launch writes only its leading footprint and leaves the canary
    untouched. One element short is refused before any write. The cells
    cover causal/page ends, split-edge winners, tied scores and signed zero.

    Only fp32 logits are certified. bf16/fp16 logits take another path through
    the op, with no split work, that no cell drives. Nor does any cell drive
    fp32 rows narrower than `SPLIT_WORK_COLUMNS` (the single-block path): the
    op accepts them, but they are not certified here.

    `compress_ratio` other than 1 and `row_kv_lens` (ragged rows) belong to
    other DSA models; Mistral Large 4 passes neither and neither is certified
    here.

    Logits are read at `logits.stride(0)`, so the paged scorer's row-padded
    view is a legal operand without a copy. At the Mistral Large 4 targets'
    max_context_len, 9216 (FP8), 32768 (NVFP4) and 1048576 (NVFP4 at its 1M
    envelope, for dep4 the checkpoint's own context), all multiples of the
    scorer's 256-column alignment, that view is contiguous.

    Metadata consumed: none beyond the arguments. Preconditions:
    - How many blocks a row is split across is chosen on the host per call
      from the row count, the column count and the device's SM count
      (`computeIndexerTopKDecodeBlocksPerRow`, `kernels/indexerTopK.cu`),
      not from the lengths, so a captured graph keeps its capture's count.
      Measured from the scratch footprint on a 152-SM GB300 at 1..16 rows:
      4096 columns give 2 blocks; 8192 and 9216 give 4 (the FP8 target's
      width: every row at 4 blocks); 12288 give 6, 16384 give 8, 18432 give
      9; 20480, 24576 and 32768 give 10 at 1..15 rows and 9 at 16 rows (the
      NVFP4 target's width), and so do 131072, 199936 and 199999. From
      200000 columns (`kDefaultSplitWorkThreshold`) the host returns 10
      blocks before it reads the SM count, so every row count splits 10
      ways: 200000, 262144, 524288 and 1048576 (the NVFP4 target at its 1M
      envelope) give 10 at 1..16 rows. A test pins 200000 and 1048576, and
      every `_ML4_1M_DEP4` cell reads that split back off the scratch.
    - Each block takes floor(end / blocks) of the row's columns and the last
      block the rest. A block of at most index_topk columns hands all of
      them to the merge, and a wider one selects its own top index_topk
      (`topKPerRowDecode`). The edges depend on the block count, not the
      width: with 10 blocks (width 32768 at 1..15 rows, width 1048576 at
      every row count) the blocks start selecting at ends 20481 (the last
      block) and 20490 (all 10), with 9 at 18433 and 18441. The cells drive
      both sides of each edge, at width 32768 (10 and 9 blocks) and at
      1048576 (10). One-off sweeps (not cells) with distinct scores selected
      the reference's set in every row: every end 1..32768 at width 32768,
      at 15 and at 16 rows; and at width 1048576, 16 rows a call, every end
      1..32768, every end 1015809..1048576 and 32768 random ends between,
      plus 1024 random ends one row a call; and, in the dep4 target's sweep
      at width 1048576, 16 rows a call, every end 1..40960, 16384 drawn ends
      past it and the last 1024 ends to 1048576.
    - A block's columns are read in 16-byte vectors from its own start
      (`vectorized_process`, `kernels/indexerTopK.cu`), so a block that
      starts off that grid (a block width of 104857 columns, odd, puts most
      of a full-length row's starts there) reads its first and last few
      columns one at a time. At width 1048576 a cell scores each block's
      first and last 4 columns at the top of the row or either side of its
      2048th score, over block widths of every residue mod 4 (68 of its 160
      block starts off the grid), and every planted column lands on its side
      of the cut. One-off sweeps of the same planting at 16 rows a call (40
      ends 1048537..1048576, 512 ends 20490..21001 and 2048 random ends
      between) selected the reference's set in every row.
    - From 4096 fp32 columns the call needs `radix_aux_indices` (int32) and
      `radix_aux_logits` (fp32), contiguous, at least rows x blocks x
      index_topk elements each. At width 32768 that is 307200 at 15 rows and
      294912 at 16, inside the 16-sequence arena's 327680. At width 1048576
      it is the whole arena at 16 rows: every element of `[16, 10, 2048]` is
      written and nothing past it, and an arena one element short is
      refused (`got 327679 / 327679, need 327680`). At every call a
      `_ML4_1M_DEP4` cell gates, the first launch writes exactly rows x blocks
      x index_topk elements, a leading prefix, and nothing past it: those
      cells pass the arena at its exact size, `[16, 10, 2048]` of each, as
      the leading view of a buffer whose tail stayed untouched. A stable
      arena keeps a captured graph valid: the DSA metadata's `radix_aux_*`
      reserves one row per verification token, or one per sequence without
      speculation. Without them the op raises
      `RuntimeError: radix_aux_{indices,logits} must be pre-allocated by the caller when
      blocks_per_row > 1 (got blocks_per_row=4). Required for CUDA Graph
      safety.`; with too few elements, `RuntimeError:
      radix_aux_{indices,logits} must hold at least
      num_rows*blocks_per_row*index_topk elements (got 24576 / 24576, need
      32768)`.
    - `seq_lens` must be int32; the op refuses int64 itself with
      `RuntimeError: expected scalar type Int but found Long`.
    - Uniform `next_n` needs `seq_lens >= next_n - 1`: a negative causal end
      triggers an illegal address in the split kernel (measured at next_n=4,
      seq_lens=1/2, 32776 columns and stride 33024). The eager guard refuses
      this input at 32776 and 1048584 columns; captured calls must preserve
      the bound on replay. Runtime
      padding dummies have at least next_n tokens, including their drafts.
    - Treat each row as a set (see the class docstring). A captured graph's
      replay selects the same set as an eager call on the same data, in a
      different order. The lengths are read at replay: a graph captured as
      the engine captures it at width 1048576 (one row at 1048575, the
      others at 1) and replayed at ends from 1 to 1048576 (the 2048/2049
      and 10-block edges among them) selected an eager call's sets.
    - On the split path, a row with end <= index_topk returns a column scored
      -inf or -FLT_MAX as -1: it loses to the merge's -FLT_MAX padding
      (measured: 99 of 100 columns back). The paged scorer's finite
      relu-weighted sums reach neither value. A row whose end is 0 returns
      all -1. Measured again at width 32768, at 10 blocks and at 9, and at
      width 1048576 (10 blocks, 2 and 16 rows): 99 of 100 columns back for
      each value.

    Cost: the rows' ends, more than the width, set the cost. At width 1048576
    with every row at full length, 16 rows launch 160 blocks, more than one
    wave on sm_103.

    Hard limits (sm_103)
      logits row offset  rowIdx * stride0, int64              no 32-bit product
      indices offset     rowIdx * index_topk, int64           no 32-bit product. A mixed step
                                                              passes the tail rows of a
                                                              [32768, 2048] selection buffer as
                                                              a view, whose base offset torch
                                                              forms in 64 bits
      scratch offset     rowIdx * blocks * index_topk, int64  no 32-bit product
      rows, seq_lens     int32                                envelope: 16 rows at 1048576;
                                                              64 at 32776/1048584 (DSpark)
      column index       int32, below the column count        boundary 2^31 columns; envelope
                                                              1048584, not probed
      logits stride(0)   cast to int32 on the host            boundary 2^31; envelope 1048576,
                                                              DSpark 33024/1048832; not probed
      logits elements    rows x stride(0)                     envelope 16 x 1048576 = 2^24, far
                                                              below 2^31; probed one-off (not a
                                                              cell) at 2049 x 1048576, past
                                                              2^31: every row's set right, row
                                                              2048 (from element 2^31) included;
                                                              DSpark 64 x 1048832 = 67125248,
                                                              below 2^31 (2048 such rows)
      scratch elements   rows x blocks x index_topk           envelope 16 x 10 x 2048 = 327680,
                                                              the arena exactly at width
                                                              1048576; DSpark 64 x 10 x 2048
                                                              = 1310720 at width 1048584: driven
      top-k indices      int32 throughout, never through fp32 (the split path carries each
                         candidate's fp32 score beside its int32 column)
      index_topk         launch ceiling 8188 (one block),     probed at 8188/8189, and
                         3020 (split)                         3020/3021 at 9216, 32768 and
                                                              1048576 columns: is_valid
                                                              refuses past it
      KV pool            not read                             the op takes the scorer's logits,
                                                              so the pool's size does not bound
                                                              it
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
