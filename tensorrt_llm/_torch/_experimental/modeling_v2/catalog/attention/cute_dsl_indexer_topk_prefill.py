# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill top-k of the DSA indexer in CuTe DSL: per query row, the best keys of its window."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: What the indexer's prefill passes: top-k 2048, the "REREAD" overflow policy
#: the engine's `TopK` module keeps (exact), no shared-memory value cache.
_INDEXER = dict(top_k=2048, overflow_policy="REREAD", cache_smem_values=False)
#: The Mistral Large 4 targets' call site: fp8_mqa_logits' padded output at the
#: op's default 256-bit copy width (the cells marked 128 drive the width the
#: engine's `TopK` module passes).
_ML4_256 = dict(layout="padded", num_copy_bits=256, **_INDEXER)
_ML4_128 = dict(layout="padded", num_copy_bits=128, **_INDEXER)


class CuteDslIndexerTopkPrefill(OpWrapper):
    """Select each prefill row's `top_k` best keys inside its window, as window-local indices.

    `input_values` is `[rows, W]` fp32 scores over one K block; row `r` sees
    columns `[row_starts[r], row_ends[r])`. With `len = row_ends[r] -
    row_starts[r]`: when `len <= top_k`, `output_indices[r]` is `0..len-1` in
    order, then -1 (a window with `len <= 0` is all -1); otherwise it holds the
    `top_k` columns of the window with the largest scores, minus
    `row_starts[r]`, in unspecified order, and no -1. Nothing outside a row's
    window is read, its row padding included.

    One launch of `torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell`
    (registered in `tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`;
    kernel `FilteredTopKKernelVarlenPrefill` in
    `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/filtered_top_k_prefill_varlen.py`),
    one CTA per row: the CuTe DSL prefill top-k the engine's `TopK` module
    (`tensorrt_llm/_torch/modules/top_k.py`) runs for
    `TopKImplementation.CUTE_DSL_RADIX`. Selection only: the scores
    (`fp8_mqa_logits`) and the mapping of local indices to pool rows are the
    caller's.

    `output_indices` (`[rows, top_k]` int32) is written in place and returned.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Mistral Large 4 calls this per indexer prefill chunk, per row tile, on
    # fp8_mqa_logits' row-padded output: one K block holds the keys of one or
    # more prompts back to back, so a later prompt's windows start past zero.
    # Mistral Large 4 at max_seq_len 9216 (FP8) and 32768 (NVFP4) runs without KV
    # reuse or chunked prefill (no cached prefix, so W is the step's token count:
    # up to max_num_tokens 9216 for the FP8 target and 32768 for the NVFP4 one;
    # `num_copy_bits` the op's default 256 at the call site), tiling the rows at
    # floor(2^24 / W).
    # The NVFP4 tp4, tep4, dep4 and tep8 targets at max_seq_len 1048576 run
    # chunked prefill: tp4's cells are the two groups after the 256-bit copy
    # cells; tep4's and tep8's are the last two groups below (the last one
    # tep8's q-split shares) and the cells the tep4 group shares with the first
    # of tp4's; dep4's are that first group's cells but its `W 262144 wide`.
    # `blocks` lists one call per K block; a block lists `(cached, new,
    # rows_taken)` per prompt, `rows_taken` keeping each prompt's last rows (a
    # row tile). The op compiles one kernel per power-of-two width bucket, so the
    # cells drive every bucket from 1 to 32768 and both edges of the two widest,
    # both edges of every bucket from 65536 to 2097152, and bucket 4194304 from
    # its lower edge to the widest packed chunk the per-rank pool holds.
    # `values` names the score pattern the test writes into the windows;
    # `layout` is "padded" as fp8_mqa_logits returns it, or "compact".
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="widths 1..1025, one call per width bucket below top-k: every "
            "window is short, so each row lists 0..len-1 then -1",
            spec=dict(
                blocks=tuple(
                    [(0, w, None)] for w in (1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025)
                ),
                values="distinct",
                layout="padded",
                num_copy_bits=128,
                **_INDEXER,
            ),
        ),
        Cell(
            why="W 2048 (bucket 2048): the last window is exactly top-k long and still short",
            spec=dict(
                blocks=([(0, 2048, None)],),
                values="distinct",
                layout="padded",
                num_copy_bits=128,
                **_INDEXER,
            ),
        ),
        Cell(
            why="W 2049, bucket 4096's lower edge: a window one key past top-k is ranked",
            spec=dict(
                blocks=([(0, 2049, 64)],),
                values="distinct",
                layout="padded",
                num_copy_bits=128,
                **_INDEXER,
            ),
        ),
        Cell(
            why="two prompts filling bucket 4096: the second's windows start at 1500 "
            "and its last 548 rows are ranked",
            spec=dict(
                blocks=([(0, 1500, None), (0, 2596, None)],),
                values="distinct",
                layout="padded",
                num_copy_bits=128,
                **_INDEXER,
            ),
        ),
        Cell(
            why="W 8192, the widest bucket whose threshold bin always fits the "
            "8192-entry shared candidate buffer",
            spec=dict(
                blocks=([(0, 8192, 256)],),
                values="distinct",
                layout="padded",
                num_copy_bits=128,
                **_INDEXER,
            ),
        ),
        Cell(
            why="W 8193, bucket 16384's lower edge: narrow scores put the last "
            "window's 8193 keys in one coarse bin, one past the shared buffer, so "
            "REREAD's second pass runs",
            spec=dict(
                blocks=([(0, 8193, 64)],),
                values="narrow",
                layout="padded",
                num_copy_bits=128,
                **_INDEXER,
            ),
        ),
        Cell(
            why="W 9216, the FP8 target's widest K block: a 9216-token prompt's last "
            "row tile of floor(2^24 / 9216) = 1820 rows",
            spec=dict(
                blocks=([(0, 9216, 1820)],),
                values="distinct",
                layout="padded",
                num_copy_bits=128,
                **_INDEXER,
            ),
        ),
        Cell(
            why="W 9216 tile, narrow scores: 1024 windows of more than 8192 keys "
            "overflow the shared buffer and take REREAD",
            spec=dict(
                blocks=([(0, 9216, 1820)],),
                values="narrow",
                layout="padded",
                num_copy_bits=128,
                **_INDEXER,
            ),
        ),
        Cell(
            why="W 9216 tile, integer scores 0..3: thousands of exact ties at the "
            "k-th score, gated on the selected scores",
            spec=dict(
                blocks=([(0, 9216, 1820)],),
                values="ties",
                layout="padded",
                num_copy_bits=128,
                **_INDEXER,
            ),
        ),
        Cell(
            why="the FP8 target's max_num_tokens 9216 as two prompts in one call, all "
            "9216 rows, scores spanning +-1.6e6 (past fp16 in the coarse key)",
            spec=dict(
                blocks=([(0, 5000, None), (0, 4216, None)],),
                values="wide",
                layout="padded",
                num_copy_bits=128,
                **_INDEXER,
            ),
        ),
        Cell(
            why="max_batch_size 16 prompts of 576 in one 9216-column block: every "
            "window short, in bucket 16384",
            spec=dict(
                blocks=([(0, 576, None)] * 16,),
                values="distinct",
                layout="padded",
                num_copy_bits=128,
                **_INDEXER,
            ),
        ),
        Cell(
            why="the op's default 256-bit copy width (the engine's TopK module "
            "passes 128), two prompts at W 9216",
            spec=dict(
                blocks=([(0, 5000, None), (0, 4216, None)],),
                values="distinct",
                layout="padded",
                num_copy_bits=256,
                **_INDEXER,
            ),
        ),
        Cell(
            why="compact scores of odd width 3001: each row starts at its own "
            "4-byte phase, which the kernel's scalar prologue aligns",
            spec=dict(
                blocks=([(0, 2600, None), (0, 401, None)],),
                values="distinct",
                layout="compact",
                num_copy_bits=128,
                **_INDEXER,
            ),
        ),
        # Mistral Large 4 NVFP4 at max_num_tokens 32768. A K block of W keys is
        # bucket next_pow2(W): W 12001..16384 compiles bucket 16384 and W
        # 16385..32768 bucket 32768; both keep the 8192-entry shared candidate
        # buffer and the 16-bit candidate index of bucket 8192.
        Cell(
            why="W 16384, bucket 16384's upper edge: a 16384-token prompt's last row "
            "tile of floor(2^24 / 16384) = 1024 rows",
            spec=dict(blocks=([(0, 16384, 1024)],), values="distinct", **_ML4_128),
        ),
        Cell(
            why="W 16385, bucket 32768's lower edge: the last floor(2^24 / 16385) = 1023 "
            "rows, narrow scores overflow the shared buffer and take REREAD",
            spec=dict(blocks=([(0, 16385, 1023)],), values="narrow", **_ML4_128),
        ),
        Cell(
            why="W 32768, bucket 32768's upper edge: a 32768-token prompt's last row "
            "tile of floor(2^24 / 32768) = 512 rows",
            spec=dict(blocks=([(0, 32768, 512)],), values="distinct", **_ML4_128),
        ),
        Cell(
            why="W 32768 tile, narrow scores: every window of 32257..32768 keys in one "
            "coarse bin, four times the shared buffer, so REREAD rescans the row",
            spec=dict(blocks=([(0, 32768, 512)],), values="narrow", **_ML4_128),
        ),
        Cell(
            why="W 32768 tile, integer scores 0..3: about 8192 tied keys at the k-th "
            "score, the threshold bin on either side of the shared buffer's size",
            spec=dict(blocks=([(0, 32768, 512)],), values="ties", **_ML4_128),
        ),
        Cell(
            why="max_num_tokens 32768 as two prompts in one call, all 32768 rows: "
            "the second's windows start at 20000; scores spanning +-1.6e6",
            spec=dict(blocks=([(0, 20000, None), (0, 12768, None)],), values="wide", **_ML4_128),
        ),
        Cell(
            why="max_batch_size 16 prompts of 2048 in one 32768-column block, all "
            "rows: every window at most top-k long, in bucket 32768",
            spec=dict(blocks=([(0, 2048, None)] * 16,), values="distinct", **_ML4_128),
        ),
        Cell(
            why="call site's 256-bit copies, W 32768: the last 512-row tile",
            spec=dict(blocks=([(0, 32768, 512)],), values="distinct", **_ML4_256),
        ),
        Cell(
            why="call site's 256-bit copies, W 32768 tile, narrow scores: REREAD's "
            "rescan at 32-byte loads",
            spec=dict(blocks=([(0, 32768, 512)],), values="narrow", **_ML4_256),
        ),
        Cell(
            why="call site's 256-bit copies, a 32768-token prompt in one call: every "
            "window length 1..32768 in bucket 32768",
            spec=dict(blocks=([(0, 32768, None)],), values="distinct", **_ML4_256),
        ),
        Cell(
            why="call site's 256-bit copies, one 1024-token prompt: a K block no wider than top-k",
            spec=dict(blocks=([(0, 1024, None)],), values="distinct", **_ML4_256),
        ),
        Cell(
            why="call site's 256-bit copies, packed prompts behind cached prefixes, long and "
            "short rows, later windows past 0",
            spec=dict(
                blocks=([(2900, 100, None), (0, 1500, None), (2000, 600, None)],),
                values="distinct",
                **_ML4_256,
            ),
        ),
        Cell(
            why="call site's 256-bit copies, a 2800-column block in bucket 4096 behind a "
            "2500-key cached prefix",
            spec=dict(blocks=([(2500, 300, None)],), values="distinct", **_ML4_256),
        ),
        Cell(
            why="call site's 256-bit copies, a block of exactly 8192 columns: a prompt's "
            "last 256 rows",
            spec=dict(blocks=([(7000, 1192, 256)],), values="distinct", **_ML4_256),
        ),
        Cell(
            why="call site's 256-bit copies, a 12000-column block in bucket 16384 behind a "
            "cached prefix past 8192: a prompt's last 128 rows",
            spec=dict(blocks=([(11000, 1000, 128)],), values="distinct", **_ML4_256),
        ),
        # Mistral Large 4 NVFP4 at max_seq_len 1048576 with chunked prefill
        # (max_num_tokens 32768): a chunk scores its rows against its requests'
        # whole KV, cached + new, so W = n_k reaches 1048576 (the 32nd chunk of a
        # 1M prompt), and the target tiles the rows at floor(2^28 / W): 256 at
        # 1048576, 8192 at 32768. A chunk's or a rank's share of rows ends in a
        # shorter tile (the 64- and 16-row cells): at tp4 a step of 8192 context
        # rows or more is split over the four ranks, each scoring its share in
        # chunks of its own, a request cut by a share's edge alone. dep4 tiles
        # at floor(budget / W): at 1048576, 64 rows under a 2^26 budget and 16
        # under 2^24 (the same two cells). A packed chunk puts several requests'
        # keys back to back, its rows times W at most 2^28. Each bucket's edges
        # at the call site's 256-bit copies; `rereads` says which rows take
        # REREAD's second pass, counted by the test with the kernel's coarse key
        # and its buffer size.
        Cell(
            why="W 1048576 distinct, bucket 1048576's upper edge: the last 256-row "
            "tile (2^28 budget) of a 1M prompt's 32nd chunk, cached 1015808",
            spec=dict(
                blocks=([(1015808, 32768, 256)],), values="distinct", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="W 1048576 narrow: every row of the 256-row tile rescans its window "
            "of 1048321..1048576 keys at 32-bit candidate indices",
            spec=dict(
                blocks=([(1015808, 32768, 256)],), values="narrow", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="W 1048576 ties, integer scores 0..3: about 262144 keys tied at the "
            "k-th score in each row of the 256-row tile",
            spec=dict(blocks=([(1015808, 32768, 256)],), values="ties", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 1048576 wide, scores spanning +-1.6e6 (past fp16 in the coarse "
            "key): the 256-row tile",
            spec=dict(blocks=([(1015808, 32768, 256)],), values="wide", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 1048576 spread: scores falling geometrically from the top, so the "
            "threshold bin fits the 4096-entry buffer and no row of the 256-row tile "
            "rereads",
            spec=dict(
                blocks=([(1015808, 32768, 256)],), values="spread", rereads="none", **_ML4_256
            ),
        ),
        Cell(
            why="W 1048576 narrow, the 64-row tile of a 2^26 budget",
            spec=dict(blocks=([(1015808, 32768, 64)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 1048576 narrow, the 16-row tile of a 2^24 budget (16 CTAs)",
            spec=dict(blocks=([(1015808, 32768, 16)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 524289 narrow, bucket 1048576's lower edge: the last "
            "floor(2^28 / 524289) = 511 rows",
            spec=dict(blocks=([(491521, 32768, 511)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 524288 distinct, bucket 524288's upper edge: the last 512-row tile",
            spec=dict(
                blocks=([(491520, 32768, 512)],), values="distinct", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="W 524288 narrow, bucket 524288's upper edge: the last 512-row tile",
            spec=dict(blocks=([(491520, 32768, 512)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 262145 narrow, bucket 524288's lower edge: the last "
            "floor(2^28 / 262145) = 1023 rows",
            spec=dict(
                blocks=([(229377, 32768, 1023)],), values="narrow", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="W 262144 distinct, bucket 262144's upper edge: the last 1024-row tile",
            spec=dict(
                blocks=([(229376, 32768, 1024)],), values="distinct", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="W 262144 ties, integer scores 0..3, bucket 262144's upper edge: the "
            "last 1024-row tile",
            spec=dict(blocks=([(229376, 32768, 1024)],), values="ties", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 131073 narrow, bucket 262144's lower edge: the last "
            "floor(2^28 / 131073) = 2047 rows",
            spec=dict(blocks=([(98305, 32768, 2047)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 131072 distinct, bucket 131072's upper edge: the last 2048-row tile",
            spec=dict(
                blocks=([(98304, 32768, 2048)],), values="distinct", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="W 131072 ties, integer scores 0..3, bucket 131072's upper edge: the "
            "last 2048-row tile",
            spec=dict(blocks=([(98304, 32768, 2048)],), values="ties", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 65537 narrow, bucket 131072's lower edge, the narrowest 32-bit "
            "candidate index: the last floor(2^28 / 65537) = 4095 rows",
            spec=dict(blocks=([(32769, 32768, 4095)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 65537 spread, bucket 131072's lower edge: no row of the last 4095 "
            "overflows the 4096-entry buffer",
            spec=dict(
                blocks=([(32769, 32768, 4095)],), values="spread", rereads="none", **_ML4_256
            ),
        ),
        Cell(
            why="W 65536 distinct, bucket 65536's upper edge, the widest 16-bit "
            "candidate index (columns to 65535): the last 4096-row tile",
            spec=dict(
                blocks=([(32768, 32768, 4096)],), values="distinct", rereads="none", **_ML4_256
            ),
        ),
        Cell(
            why="W 65536 narrow, bucket 65536's upper edge: the last 4096-row tile",
            spec=dict(blocks=([(32768, 32768, 4096)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 65536 ties, integer scores 0..3, bucket 65536's upper edge: the "
            "last 4096-row tile",
            spec=dict(blocks=([(32768, 32768, 4096)],), values="ties", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 32769 narrow, bucket 65536's lower edge: the last "
            "floor(2^28 / 32769) = 8191 rows",
            spec=dict(blocks=([(1, 32768, 8191)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 32768 ties at 256-bit copies: the 8192-row tile (2^28 budget) of a "
            "32768-token first chunk",
            spec=dict(blocks=([(0, 32768, 8192)],), values="ties", **_ML4_256),
        ),
        Cell(
            why="W 32768 wide at 256-bit copies: the 8192-row tile (2^28 budget) of a "
            "32768-token first chunk",
            spec=dict(blocks=([(0, 32768, 8192)],), values="wide", rereads="none", **_ML4_256),
        ),
        Cell(
            why="call site's 256-bit copies, max_batch_size 16 prompts of 2048 in one "
            "32768-column block: every window at most top-k long",
            spec=dict(blocks=([(0, 2048, None)] * 16,), values="distinct", **_ML4_256),
        ),
        Cell(
            why="packed chunk, distinct: 16 requests' last 16 rows over 65520 cached "
            "keys each in one 1048576-column block, windows starting at k * 65536",
            spec=dict(
                blocks=([(65520, 16, None)] * 16,), values="distinct", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="packed chunk, narrow: 16 requests' last 16 rows over 65520 cached "
            "keys each in one 1048576-column block, every row rereads",
            spec=dict(
                blocks=([(65520, 16, None)] * 16,), values="narrow", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="packed chunk, spread: 16 requests' last 16 rows over 65520 cached "
            "keys each in one 1048576-column block, no row rereads",
            spec=dict(
                blocks=([(65520, 16, None)] * 16,), values="spread", rereads="none", **_ML4_256
            ),
        ),
        Cell(
            why="mixed packed chunk: a 1M prompt's last 16 rows (cached 983040) and a "
            "fresh 200-token prompt whose short windows start at 983056, W 983256",
            spec=dict(
                blocks=([(983040, 16, None), (0, 200, None)],),
                values="distinct",
                rereads="all",
                **_ML4_256,
            ),
        ),
        Cell(
            why="split packed chunk: one request's 5-token last chunk over 983040 "
            "cached keys scored alone, W 983045, 5 rows",
            spec=dict(blocks=([(983040, 5, None)],), values="distinct", rereads="all", **_ML4_256),
        ),
        Cell(
            why="W 262144 wide, scores spanning +-1.6e6, bucket 262144's upper edge: "
            "the last 1024-row tile",
            spec=dict(blocks=([(229376, 32768, 1024)],), values="wide", rereads="all", **_ML4_256),
        ),
        # Past 1048576 columns: the engine packs whole requests into one chunk
        # while their rows times their summed KV stay within 2^28, so a K block
        # holds two 1M requests' keys and more, up to the tokens the per-rank
        # pool holds (2,373,856 at the tp4 boot at max_seq_len 32768, 2,352,000
        # at 1048576), at most floor(2^28 / W) rows: buckets 2097152 and
        # 4194304, the 32-bit-index kernel compiled at two more widths.
        Cell(
            why="W 1048577 narrow, bucket 2097152's lower edge: a 1M prompt's last "
            "254 rows packed with a fresh 1-token prompt, 255 rows",
            spec=dict(
                blocks=([(1048322, 254, None), (0, 1, None)],),
                values="narrow",
                rereads="all",
                **_ML4_256,
            ),
        ),
        Cell(
            why="W 1048577 ties, integer scores 0..3, bucket 2097152's lower edge: a "
            "1M prompt's last 254 rows beside a 1-key window",
            spec=dict(
                blocks=([(1048322, 254, None), (0, 1, None)],),
                values="ties",
                rereads="all",
                **_ML4_256,
            ),
        ),
        Cell(
            why="W 1048577 wide, scores spanning +-1.6e6, bucket 2097152's lower edge: "
            "a 1M prompt's last 254 rows beside a 1-key window",
            spec=dict(
                blocks=([(1048322, 254, None), (0, 1, None)],),
                values="wide",
                rereads="all",
                **_ML4_256,
            ),
        ),
        Cell(
            why="W 1048577 spread, bucket 2097152's lower edge: no row of a 1M prompt's "
            "last 254 overflows the 4096-entry buffer",
            spec=dict(
                blocks=([(1048322, 254, None), (0, 1, None)],),
                values="spread",
                rereads="none",
                **_ML4_256,
            ),
        ),
        Cell(
            why="W 2097152 distinct, bucket 2097152's upper edge: two 1M requests' last "
            "rows (cached 1048575 each) packed, the second's window from 1048576",
            spec=dict(
                blocks=([(1048575, 1, None), (1048575, 1, None)],),
                values="distinct",
                rereads="all",
                **_ML4_256,
            ),
        ),
        Cell(
            why="W 2097153 distinct, bucket 4194304's lower edge: two 1M requests' last "
            "63 rows and a fresh 1-token prompt packed, 127 rows",
            spec=dict(
                blocks=([(1048513, 63, None), (1048513, 63, None), (0, 1, None)],),
                values="distinct",
                rereads="all",
                **_ML4_256,
            ),
        ),
        Cell(
            why="W 2373713 narrow, bucket 4194304 at the per-rank pool: two 1M requests' "
            "last 50 rows and a 277613-key request's last 13, 113 rows",
            spec=dict(
                blocks=([(1048000, 50, None), (1048000, 50, None), (277600, 13, None)],),
                values="narrow",
                rereads="all",
                **_ML4_256,
            ),
        ),
        Cell(
            why="W 2373713 ties, integer scores 0..3, bucket 4194304 at the per-rank "
            "pool: three requests packed, 113 rows",
            spec=dict(
                blocks=([(1048000, 50, None), (1048000, 50, None), (277600, 13, None)],),
                values="ties",
                rereads="all",
                **_ML4_256,
            ),
        ),
        Cell(
            why="W 2373713 wide, scores spanning +-1.6e6, bucket 4194304 at the "
            "per-rank pool: three requests packed, 113 rows",
            spec=dict(
                blocks=([(1048000, 50, None), (1048000, 50, None), (277600, 13, None)],),
                values="wide",
                rereads="all",
                **_ML4_256,
            ),
        ),
        Cell(
            why="W 2373713 spread, bucket 4194304 at the per-rank pool: no row of the "
            "three packed requests overflows the 4096-entry buffer",
            spec=dict(
                blocks=([(1048000, 50, None), (1048000, 50, None), (277600, 13, None)],),
                values="spread",
                rereads="none",
                **_ML4_256,
            ),
        ),
        # Mistral Large 4 NVFP4 tep4 at max_seq_len 1048576 with chunked prefill
        # (max_num_tokens 32768): a chunk scores its rows against its requests'
        # whole KV, cached + new, so W = n_k reaches 1048576 (the 32nd chunk of a
        # 1M prompt), and the target tiles the rows at floor(2^28 / W): 8192 at
        # 32768, 256 at 1048576. A step of 8192 or more context rows splits its
        # queries over the 4 TP ranks, 8192 rows each, and rank r scores its
        # share against the keys up to its last row (cached + 8192 (r + 1)). A
        # packed chunk puts several requests' keys back to back. Each bucket's
        # edges at the call site's 256-bit copies; `rereads` says which rows take
        # REREAD's second pass, counted by the test with the kernel's coarse key
        # and its buffer size. tep4's other cells, W 32768 to 1048576, are the
        # identical ones of the first chunked-prefill group above.
        Cell(
            why="W 32768 distinct at 256-bit copies: the 8192-row tile (2^28 budget) "
            "of a 32768-token first chunk, rank 3's whole q-split share",
            spec=dict(blocks=([(0, 32768, 8192)],), values="distinct", rereads="none", **_ML4_256),
        ),
        Cell(
            why="W 262144 narrow, bucket 262144's upper edge: the last 1024-row tile",
            spec=dict(
                blocks=([(229376, 32768, 1024)],), values="narrow", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="q-split rank 0's share of a 1M prompt's last chunk, W 1024000: the "
            "last 70 of its 8192 rows (31 tiles of 262, then 70)",
            spec=dict(
                blocks=([(1015808, 8192, 70)],), values="distinct", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="q-split ranks 0..2 of a 1M prompt's last chunk at W 1024000, "
            "1032192 and 1040384: each rank's full tile (262, 260, 258 rows) and "
            "the last tiles of ranks 1 and 2 (132, 194 rows)",
            spec=dict(
                blocks=(
                    [(1015808, 8192, 262)],
                    [(1024000, 8192, 260)],
                    [(1024000, 8192, 132)],
                    [(1032192, 8192, 258)],
                    [(1032192, 8192, 194)],
                ),
                values="distinct",
                rereads="all",
                **_ML4_256,
            ),
        ),
        Cell(
            why="packed chunk tails, distinct: 16 requests' last 16 rows over 61440 "
            "cached keys each in one 983296-column block, windows starting at k * "
            "61456 up to 921840",
            spec=dict(
                blocks=([(61440, 16, None)] * 16,), values="distinct", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="packed chunk tails, odd starts, distinct: 16 requests' last 16 rows "
            "over 61439 cached keys each, windows starting at k * 61455, every 4-byte "
            "phase of a 32-byte load",
            spec=dict(
                blocks=([(61439, 16, None)] * 16,), values="distinct", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="packed chunk tails, odd starts, spread: windows starting at k * 61455 "
            "of a 983280-column block, no row rereads",
            spec=dict(
                blocks=([(61439, 16, None)] * 16,), values="spread", rereads="none", **_ML4_256
            ),
        ),
        # Mistral Large 4 NVFP4 tep8 at the same envelope: a step of 8192 or more
        # context rows splits its queries over the 8 TP ranks (the engine's
        # `_split_prefill_queries`: rank r takes rows [count * r // 8,
        # count * (r + 1) // 8), 4096 of a 32768-row chunk) and rank r scores its
        # share against cached + 4096 (r + 1) keys, so rank 7's share spans the
        # chunk's whole K block and a call of a split step holds at most 4096
        # rows. Each cell is the last row tile floor(2^28 / W) cuts from a share
        # (the target's `_prefill_tiles`, from the share's first row). A step
        # below 8192 rows is not split, and a chunk the engine packs past
        # max_seq_len keys is cut per request by the target, so W stays within
        # 1048576.
        Cell(
            why="tep8 q-split rank 7's 4096-row share of a 32768-token first chunk "
            "at 8 ranks, W 32768 distinct: one tile under the 2^28 budget",
            spec=dict(blocks=([(0, 32768, 4096)],), values="distinct", rereads="none", **_ML4_256),
        ),
        Cell(
            why="tep8 q-split rank 7's 4096-row share at W 32769 narrow, bucket 65536's "
            "lower edge: one tile (floor(2^28 / 32769) = 8191 rows)",
            spec=dict(blocks=([(1, 32768, 4096)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="tep8 rank 7's last tile at W 65537 narrow, bucket 131072's lower edge: "
            "4096 = 4095 + 1 rows, so 1 row",
            spec=dict(blocks=([(32769, 32768, 1)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="tep8 rank 7's last tile at W 131073 narrow, bucket 262144's lower edge: "
            "4096 = 2 * 2047 + 2 rows, so 2 rows",
            spec=dict(blocks=([(98305, 32768, 2)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="tep8 rank 7's last tile at W 262145 distinct, bucket 524288's lower "
            "edge: 4096 = 4 * 1023 + 4 rows, so 4 rows",
            spec=dict(blocks=([(229377, 32768, 4)],), values="distinct", rereads="all", **_ML4_256),
        ),
        Cell(
            why="tep8 rank 7's last tile at W 524289 narrow, bucket 1048576's lower "
            "edge: 4096 = 8 * 511 + 8 rows, so 8 rows",
            spec=dict(blocks=([(491521, 32768, 8)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="tep8 q-split rank 0's share of a 1M prompt's last chunk at W 1019904 "
            "(1015808 cached + 4096): the last 151 of its 4096 rows (15 tiles of 263, "
            "then 151)",
            spec=dict(
                blocks=([(1015808, 4096, 151)],), values="distinct", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="tep8 envelope prompt's last chunk (1048559 tokens: 32751 rows behind "
            "1015808 cached), rank 7's 4094-row share [28657, 32751) at W 1048559: its "
            "last 254 rows (15 tiles of 256, then 254)",
            spec=dict(
                blocks=([(1015808, 32751, 254)],), values="distinct", rereads="all", **_ML4_256
            ),
        ),
        Cell(
            why="tep8 unsplit 8191-row last chunk (below the 8192-row q-split "
            "threshold, every rank selects every row) behind 1040385 cached, W 1048576 "
            "narrow: the last 255 rows (31 tiles of 256, then 255)",
            spec=dict(blocks=([(1040385, 8191, 255)],), values="narrow", rereads="all", **_ML4_256),
        ),
        Cell(
            why="tep8 one request's piece of a packed chunk past 1M keys: two 5-token "
            "tails over 983040 cached keys, packed by the engine into 1966090 keys and "
            "cut per request by the target, a 983045-column block from column 0",
            spec=dict(blocks=([(983040, 5, None)],), values="distinct", rereads="all", **_ML4_256),
        ),
    )

    note = """
    Metadata consumed: none. The windows are the caller's: an indexer prefill
    chunk's `cu_seqlen_ks` / `cu_seqlen_ke`, sliced to the row tile, in the K
    block's column coordinates.

    Preconditions, each measured to raise rather than answer, so nothing here
    guards them:
      - `input_values` fp32 `[rows, W]` with unit column stride and a 32-byte
        aligned base; any row stride, so fp8_mqa_logits' row-padded view is
        read in place. "Misaligned Tensor data on argument #0 ... expected data
        alignment=32 bytes" for a base at a 4-byte phase (a row slice of an
        odd-width compact tensor); "Mismatched input_values.strides[1] ...
        expected to be 1" for a column-major view.
      - `row_starts` / `row_ends` int32 `[rows]`: "Mismatched Tensor on
        argument #1 ... expected dtype=int32" for int64; "Mismatched
        row_starts.shape[0] ... expected to match input_values.shape[0]".
      - `output_indices` int32 `[rows, top_k]` with a row stride of exactly
        `top_k` (a row slice of a `[T, top_k]` buffer is fine):
        "Mismatched output_indices.shape[1] ... expected to be 2048" for a
        wider buffer, "Mismatched output_indices.strides[0] ... expected to be
        2048" for a column slice of one.
      - `rows >= 1`: zero rows raises `CUDADialectError` at launch,
        CUDA_LAUNCH_INVALID_CONFIG (cudaErrorInvalidValue; later calls in the
        process still ran). Skip empty tiles.
    The `ValueError`s come from the op's argument check, before any launch.

    The op compiles one kernel per (dtype, next power of two >= W, top_k,
    num_copy_bits, overflow_policy, cache_smem_values) on the first call that
    needs it and caches it for the process; measured here at 0.4 to 1.4 s per
    width bucket (bucket 32768: 0.69 s at 128-bit copies, 0.86 s at 256). A
    first use inside a CUDA-graph capture would compile inside it; prefill
    steps are eager. A caller that wants no first-use latency calls each
    bucket once at load: a Mistral Large 4 NVFP4 prompt of W keys reaches bucket
    next_pow2(W), so 1 to 32768; with chunked prefill at max_seq_len 1048576
    a chunk's K block holds its request's whole KV, so 1 to 1048576 (measured
    0.84 s per bucket from 32768 to 524288 at 256-bit copies, 1.07 s at
    1048576; in tep4's run, in one process at 256-bit copies, 0.872 to
    0.883 s for each of buckets 65536, 131072, 262144, 524288 and 1048576,
    4.38 s for the five; in the tep8 run's process, after a warm-up compile
    at bucket 1024, 0.854 to 0.922 s each, 4.44 s for the five), and a
    packed chunk several requests' whole KV, up to the per-rank pool's
    tokens: buckets 2097152 and 4194304 (first calls 0.88 and 0.85 s,
    compile included).

    Under "REREAD", buckets above 8192 compile a second pass over the row,
    taken when a row's threshold bin holds more than the 8192-entry shared
    candidate buffer; the result is exact either way. Buckets 16384 and 32768
    keep that 8192-entry buffer, 512 threads per row and a 16-bit candidate
    index (`FilteredTopKKernelVarlenPrefill`). Counted with the kernel's
    coarse key on the cells' own scores: every row of the narrow tiles at
    16385 and 32768 columns overflows the buffer (its threshold bin holds the
    whole window, up to 32768 keys), 100 of the 512 rows of the 32768-column
    ties tile do, and no row of the distinct or wide cells does. "TRUNCATE"
    drops those candidates instead, and was measured to select a wrong set on
    9216-, 16385- and 32768-column blocks whose windows overflow (every row
    of the narrow tiles): `is_valid` refuses it above 8192 columns. The
    winners of a long row can come back in a different order on two calls
    with the same input (all 512 rows of a 32768-column tile, the sets equal).

    Past 65536 columns (buckets 131072 to 4194304) the runner compiles the
    same kernel with a 32-bit candidate index, which halves the shared
    candidate buffer to 4096 entries at the same shared-memory budget; bucket
    65536 keeps the 16-bit index (columns to 65535) and 8192 entries. The
    bucket is the kernel's `max_num_cols`, which feeds only the index width,
    the thread count, the buffer size and whether the second pass is
    compiled, so buckets 131072 to 4194304 build alike (read off their kernel
    objects: 512 threads, 8-element copies, 4096 entries, REREAD's pass). The
    `rereads` cells read both off the kernel object for their bucket and
    count their rereading rows: at those widths every ranked row of the
    distinct, narrow, tied and wide cells rereads (scores from 65520 up, the
    distinct and wide tops, round to fp16 infinity, one coarse bin), the spread
    cells' rows (scores falling by 2^(1/8192) per rank, about 2640 keys at
    most per coarse bin) never do, and the 65536-column distinct tile fits
    the 8192-entry buffer (its threshold bin at most the 8192 integers
    57328..65519). On the packed tails (windows of 61441 to 61456 keys in a
    983296- or 983280-column block, the latter starting at every 4-byte
    phase) every row rereads with distinct scores and none with spread
    ones. At 256-bit copies in bucket 32768, 130 of the 8192 rows of the
    ties tile reread and none of the distinct or wide tiles. On the tep8
    cells no row of rank 7's 4096-row share at 32768 columns (distinct)
    rereads, and every ranked row of the others does: narrow at 32769,
    65537, 131073, 524289 and 1048576 columns, distinct at 262145, 983045,
    1019904 and 1048559.

    Cost on sm_103: rereading costs time. At 1048576 columns, where every
    row of the distinct and narrow tiles rereads and none of the spread ones
    does, the spread tiles take the least time. One call allocates nothing
    of its own: at [256, 1048576] its footprint is the caller's 1.000 GiB
    logits view (row stride 1048832) and 2 MiB output, at [113, 2373713]
    0.999 GiB (row stride 2374144) and 0.88 MiB (torch's peak and the device
    memory in use unchanged across the call). At [255, 1048576] the call
    allocated nothing either (+0.00 MiB over the caller's 0.996 GiB view).

    Measured, not a cell: a window with `row_end <= row_start` writes an all
    -1 row. Not certified: fp16 and bf16 scores (bf16 measured accepted), a
    `top_k` other than 2048 (1024 measured accepted), `num_copy_bits` other
    than 128 and 256 (64 measured accepted), the "GMEM_SPILL",
    "REREAD_ALWAYS" and "TRUNCATE" policies, `cache_smem_values=True`, W
    above 2373713 (bucket 4194304 past the widest packed cell; the tp4
    boot's per-rank pool holds 2,352,000 tokens at max_seq_len 1048576 and
    2,373,856 at 32768, and no K block is wider than the keys resident), and
    128-bit copies above 32768 columns. Buckets 2^21 and up (a packed chunk
    whose keys sum past 1048576) are certified only by tp4's cells, to W
    2373713; tep4's, dep4's and tep8's cells stop at W 1048576.
    Launched with PDL when `TRTLLM_ENABLE_PDL` is on; scheduling only.

    Hard limits (sm_103)
      input_values offset   row * stride(0), the row stride a 64-bit runtime
                            value; boundary 2^31 elements, 2^31 / 33024 =
                            65027.9 rows at the padded 32768 width
                            probed at 65100 rows of stride 33024 (72 rows past
                            2^31 elements) and at 3 rows of stride 2^30 + 256
                            (the last at 2^31 + 512): passed. Cells reach
                            32768 rows of 33024 (rows past 2^32 bytes): passed.
                            A Mistral Large 4 call stops at floor(2^24 / W) rows,
                            512 at W 32768; under a 2^28 budget 8192 rows at
                            W 32768 (270,532,608 elements), 256 at W
                            1048576, stride 1048832 (268,500,992), where 2^31
                            is 2047.5 rows, 262 at W 1024000, stride 1024256
                            (268,355,072), and 113 at W 2373713, stride
                            2374144 (268,278,272), where 2^31 is 904.5 rows:
                            not reached. At tep8 a q-split share's call holds
                            at most 4096 rows (135,266,304 elements at W
                            32768), inside the bounds above
      output_indices offset row * top_k      2^31 / 2048 = 1048576 rows
                            cells reach 32768 rows   unprobed, outside the envelope;
                            a tp4 call takes at most 8192 rows (a 2^28 tile at
                            W 32768; a rank's share of a 32768-row step under
                            the engine's query split from 8192 context rows)
      window columns        int32 offsets; per-row candidate index uint16 up
                            to a 65536-column bucket, uint32 above
                            cells reach 2373713 columns: uint16 to column
                            65535, uint32 from 65536 to 2373712 (past 2^21,
                            selections naming columns to 2373705): passed.
                            2^31 (int32) and 2^32 (uint32) unprobed, outside
                            the envelope
      fp32-exact 2^24       no index passes through fp32 (the coarse key
                            rounds scores, not columns, to fp16)
      pools                 none: the op reads the step's logits, not the KV pool
    """

    def __call__(
        self,
        input_values: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        num_copy_bits: int = 256,
        overflow_policy: str = "REREAD",
        cache_smem_values: bool = False,
    ) -> torch.Tensor:
        torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell(
            input_values,
            row_starts,
            row_ends,
            output_indices,
            top_k,
            num_copy_bits,
            overflow_policy,
            cache_smem_values,
        )
        return output_indices

    def reference(
        self,
        input_values: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        num_copy_bits: int = 256,
        overflow_policy: str = "REREAD",
        cache_smem_values: bool = False,
    ) -> torch.Tensor:
        """A fresh `[rows, top_k]` int32 selection.

        `output_indices`, `num_copy_bits`, `overflow_policy` and
        `cache_smem_values` are not read. A long row holds `torch.topk`'s
        winners in descending score order; the kernel promises no order and may
        break a tie at the k-th score differently, so the set is exact only on
        rows without such a tie.
        """
        rows, width = input_values.shape
        device = input_values.device
        starts = row_starts.long()[:, None]
        ends = row_ends.long()[:, None]
        cols = torch.arange(width, device=device)
        inside = (cols >= starts) & (cols < ends)
        scores = torch.where(inside, input_values.float(), float("-inf"))
        out = torch.full((rows, top_k), -1, dtype=torch.int64, device=device)
        ranked = min(top_k, width)
        if ranked:
            out[:, :ranked] = scores.topk(ranked, dim=1).indices - starts
        lengths = (ends - starts).squeeze(1)
        short = lengths <= top_k
        ident = torch.arange(top_k, device=device)
        out[short] = torch.where(ident < lengths[short, None], ident, -1)
        return out.to(torch.int32)

    def is_valid(
        self,
        input_values: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        num_copy_bits: int = 256,
        overflow_policy: str = "REREAD",
        cache_smem_values: bool = False,
    ) -> None:
        # "TRUNCATE" keeps only the threshold-bin candidates that fit the
        # 8192-entry shared buffer; a wider block can overflow it.
        assert overflow_policy != "TRUNCATE" or input_values.shape[1] <= 8192, (
            f"TRUNCATE drops candidates on a {input_values.shape[1]}-column block and "
            "selects a wrong set when a threshold bin overflows; use REREAD"
        )
        # The kernel reads each window from the row's base with no bound check,
        # and ranks whatever it reads as this row's scores. An end past W reads
        # the row padding or the next row and returns those columns as if they
        # were the window's; a start below zero reads the previous row's tail
        # and returns those columns as 65536 + i, wrapped through the kernel's
        # 16-bit candidate index, outside any window.
        if row_starts.numel():
            first = int(row_starts.min())
            assert first >= 0, f"a window starts at {first}, before its row"
            last = int(row_ends.max())
            assert last <= input_values.shape[1], (
                f"a window ends at {last}, past the {input_values.shape[1]}-column K block"
            )


cute_dsl_indexer_topk_prefill = CuteDslIndexerTopkPrefill()
