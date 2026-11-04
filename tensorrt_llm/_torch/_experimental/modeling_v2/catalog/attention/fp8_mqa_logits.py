# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ragged FP8 MQA logits: the DSA indexer's prefill scores, in one DeepGEMM call.

Wraps the pybind `tensorrt_llm.deep_gemm.fp8_mqa_logits`, not a
`torch.ops.trtllm.*` op: the ragged prefill scorer has no registered-op form,
and the engine's DSA indexer (`Indexer._call_mqa_logits` in
`tensorrt_llm/_torch/attention/backends/sparse/dsa/indexer.py`) calls the
pybind directly. A call given a split-KV schedule (`mqa_logits_schedule`) goes
through `fp8_fp4_mqa_logits`, of which the legacy pybind is the schedule-less
form: the same kernel, its keys spread over every SM.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm.deep_gemm import fp8_fp4_mqa_logits as _deep_gemm_fp8_fp4_mqa_logits
from tensorrt_llm.deep_gemm import fp8_mqa_logits as _deep_gemm_fp8_mqa_logits
from tensorrt_llm.deep_gemm import get_mqa_logits_metadata as _deep_gemm_mqa_logits_metadata
from tensorrt_llm.deep_gemm import get_num_sms as _deep_gemm_num_sms

from .._op import Arch, Cell, OpWrapper


def mqa_logits_schedule(
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    num_kv_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    """The split-KV schedule of one `fp8_mqa_logits` call over these windows and keys.

    DeepGEMM's `get_mqa_logits_metadata`: one single-CTA kernel on the current stream
    (no host sync) that prices each 4-row query block (`128 / num_heads` rows) by the
    256-key splits of its window span and cuts the summed splits into one contiguous run
    per SM, so a block's keys may be spread over several SMs. It returns an int32 tensor
    the call takes as `schedule_meta`. The schedule is a function of the windows, the
    key count, the head count and the device's SM count only: a caller that scores the
    same rows and windows against several layers' keys builds it once."""
    return _deep_gemm_mqa_logits_metadata(
        cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads
    )


#: The DSA indexer of Mistral Large 4: 32 heads of 128, fp8 keys with one fp32
#: scale per key.
_INDEXER = dict(heads=32, head_dim=128)

#: The Mistral Large 4 FP8 target's fullest context step: max_batch_size 16
#: requests packed into its 9216-token budget (no KV reuse, no chunked
#: prefill, so no cached prefix). Lengths are chosen so request boundaries fall
#: inside the kernel's 4-row query blocks, and the first request is a single
#: token.
_PACKED_16 = [
    (0, n) for n in (1, 7, 33, 100, 255, 257, 513, 1023, 1025, 2047, 2049, 931, 299, 301, 201, 174)
]

#: The Mistral Large 4 NVFP4 target's fullest context step: 16 requests packed into
#: its 32768-token budget, the same way. Request starts (and so window starts)
#: at keys 0, 1, 8, 41, 141, 396, 653, 1166, 2189, 3214, 5261, 7310, 11405,
#: 15502, 23693 and 31886; windows up to 8193 keys long.
_PACKED_16_32K = [
    (0, n)
    for n in (1, 7, 33, 100, 255, 257, 513, 1023, 1025, 2047, 2049, 4095, 4097, 8191, 8193, 882)
]

#: The NVFP4 tep4, tp4 and tep8 targets at their 1M envelope (max_seq_len 1048576,
#: chunked prefill in 32768-token steps): the last chunk of a 1048576-token
#: prompt, behind its 1015808-token cached prefix. Each chunk is scored against
#: the request's whole KV, cached + new, so this step's keys are 1048576. The
#: dep4 target's cells use it too: the 32nd and last indexer chunk of a
#: 1048576-token prompt at its 1M envelope (chunked prefill in 32768-token
#: chunks), 32768 new tokens over 1015808 cached, so the chunk gathers
#: n_k = 1048576 keys and its row t sees keys [0, 1015809 + t).
_LAST_CHUNK_1M = [(1015808, 32768)]

#: Under the engine's query split (tep4 and tp4, a step of 8192 or more context
#: query tokens) rank 0 scores the first 8192 rows of that chunk against the
#: keys up to its share's end, cached + 8192 = 1024000.
_RANK0_SHARE_1M = [(1015808, 8192)]

#: Rank 1's share of the same chunk: rows 8192..16383 against the keys up to
#: its share's end, cached + 16384 = 1032192.
_RANK1_SHARE_1M = [(1015808, 16384)]

#: 16 requests' 16-token chunk tails behind 61440-token prefixes, packed into
#: one indexer chunk as the engine packs them while sum(q) x sum(kv) stays
#: within its 2**28 packed-logits budget: 256 x 983296 = 251,723,776.
_TAILS_16_1M = [(61440, 16) for _ in range(16)]

#: Two 1048576-token prompts' 1-token final chunks, packed into one indexer
#: chunk: 2 x 2097152 = 4,194,304 packed logits. The second request's keys and
#: window start at 1048576 = 2**20.
_FINAL_TOKENS_2X1M = [(1048575, 1), (1048575, 1)]

#: Three requests' final chunks packed up to the 2**28 budget: two near-1M
#: prompts' 50-token tails and a 277613-token prompt's 13-token tail,
#: 113 x 2373713 = 268,229,569 packed logits. 2373713 keys sit just under the
#: 2,373,856-token per-rank pool of the tp4 target's delivered boot, which bounds
#: a packed chunk's keys while block reuse is off.
_FINAL_CHUNKS_POOL = [(1048000, 50), (1048000, 50), (277600, 13)]

#: A packed indexer chunk at the dep4 target's 1M envelope: the engine groups
#: whole requests into one chunk while sum(new) x sum(cached + new) <= 2**28,
#: so 16 requests' 16-token last chunks over 65520 cached keys each are one
#: 1048576-key block of 256 rows, each request's windows inside its own 65536
#: keys.
_PACKED_16_TAILS_1M = [(65520, 16) for _ in range(16)]

#: Under the tep8 target's query split (eight ranks, a step of 8192 or more
#: context query tokens) rank r scores rows [4096 r, 4096 (r + 1)) of a
#: 32768-token chunk against the keys up to its share's end, cached +
#: 4096 (r + 1): the request's rows up to there are the step. Rank 3's share of
#: the last chunk of a 1048576-token prompt sees 1032192 keys, rank 0's 1019904.
_TEP8_RANK3_SHARE_1M = [(1015808, 16384)]
_TEP8_RANK0_SHARE_1M = [(1015808, 4096)]


class Fp8MqaLogits(OpWrapper):
    """Score every query row against every key of one ragged prefill chunk.

    Returns `[T, N]` fp32 logits (a row-padded view, not contiguous) with

        logits[i, j] = k_scale[j] * sum_h weights[i, h] * relu(q[i, h, :] . k[j, :])

    for `cu_seq_len_k_start[i] <= j < cu_seq_len_k_end[i]`, and -inf
    elsewhere when `clean_logits`. `kv` is the pair `(k [N, D] fp8, k_scale [N]
    fp32)`. The fp8 codes of `q` enter the dot product raw: there is no q-side
    scale, so the caller folds it, with the indexer's `D**-0.5 * H**-0.5`, into
    `weights`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # `seqs` lists each request of one context step as (cached keys, new
    # tokens); token `t` of a request sees its own keys up to `cached + t`.
    # `rows`, when present, is the query tile `[lo, hi)` the call scores,
    # passed as views into the step's buffers the way a tiling caller slices
    # them; without it the call scores the whole step. The "Mistral Large 4 1M"
    # cells are the tep4 and tp4 targets' calls at their 1M envelope: each
    # scores every indexer prefill chunk in query tiles of floor(2**28 / n_k)
    # rows, cut from the chunk's first row. The "Mistral Large 4 tep8 1M" cells are
    # the tep8 target's, which tiles the same way from the first row of its
    # rank's share of the chunk under the engine's query split; `share`, when
    # present, is that share `[lo, hi)` of the step's rows (the whole step below
    # the split threshold), the rows a tile is cut from. Rank 7's last tile at
    # 1048576, 524288, 262144, 131072 and 65536 keys and the 16 packed chunk
    # tails are the "Mistral Large 4 1M" cells of the same calls. `budget`, when
    # present (the dep4 target's 1M cells), is the logits budget whose
    # floor(budget / keys)-row tiles the call is one of (the whole step when it
    # has no `rows`).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request, no cache: causal windows over the chunk's own keys",
            spec=dict(**_INDEXER, seqs=[(0, 2048)]),
        ),
        Cell(
            why="a cached prefix: every window starts at 0 and runs past the chunk",
            spec=dict(**_INDEXER, seqs=[(4096, 1024)]),
        ),
        Cell(
            why="two requests in one call: the second's windows start mid-row",
            spec=dict(**_INDEXER, seqs=[(0, 300), (1700, 500)]),
        ),
        Cell(
            why="a query count that is not a multiple of the 4-row query block",
            spec=dict(**_INDEXER, seqs=[(100, 37)]),
        ),
        Cell(
            why="Mistral Large 4 9216 keys in one call, T = N = 9216: the largest [T, N] the FP8 "
            "target's envelope (max_num_tokens = max_seq_len = 9216) admits, for a caller that "
            "does not tile",
            spec=dict(**_INDEXER, seqs=[(0, 9216)]),
        ),
        Cell(
            why="Mistral Large 4 9216 keys, the 116-row tail tile of the floor(2**24 / 9216) = "
            "1820-row query tiles: views 9100 rows into q and weights, windows reaching the "
            "last key",
            spec=dict(**_INDEXER, seqs=[(0, 9216)], rows=(9100, 9216)),
        ),
        Cell(
            why="Mistral Large 4 16 packed requests (max_batch_size) filling the FP8 target's 9216 "
            "tokens in one call: a 1-token request, request boundaries inside 4-row query blocks",
            spec=dict(**_INDEXER, seqs=_PACKED_16),
        ),
        Cell(
            why="Mistral Large 4 16 packed requests, the 1820-row tile at rows 7280..9099: "
            "windows start at keys 5261, 7310, 8241, 8540, 8841, 9042, mostly off the "
            "kernel's 4-key alignment",
            spec=dict(**_INDEXER, seqs=_PACKED_16, rows=(7280, 9100)),
        ),
        Cell(
            why="Mistral Large 4 32768 keys in one call, T = N = 32768: the largest [T, N] the NVFP4 "
            "target's envelope (max_num_tokens = max_seq_len = indexer_max_chunk_size = 32768) "
            "admits, for a caller that does not tile; its [32768, 33024]-element fp32 rows span "
            "4,328,521,728 bytes, rows 16257 on past 2**31 bytes and 32514 on past 2**32",
            spec=dict(**_INDEXER, seqs=[(0, 32768)]),
        ),
        Cell(
            why="Mistral Large 4 32768 keys, the last of the floor(2**24 / 32768) = 512-row query "
            "tiles the NVFP4 target cuts: views 32256 rows into q and weights, windows of "
            "32257..32768 keys reaching the last one",
            spec=dict(**_INDEXER, seqs=[(0, 32768)], rows=(32256, 32768)),
        ),
        Cell(
            why="Mistral Large 4 32768 tokens in one call, 16 packed requests (max_batch_size): a "
            "1-token request, windows up to 8193 keys, request boundaries inside 4-row query "
            "blocks",
            spec=dict(**_INDEXER, seqs=_PACKED_16_32K),
        ),
        Cell(
            why="Mistral Large 4 32768 tokens, the 512-row tile at rows 31744..32255 of the 16 "
            "packed requests: two requests' windows, starting at keys 23693 and 31886, off the "
            "kernel's 4-key alignment, the second request's first row mid query block",
            spec=dict(**_INDEXER, seqs=_PACKED_16_32K, rows=(31744, 32256)),
        ),
        Cell(
            why="Mistral Large 4 1M, first 256-row tile at 1048576 keys: rows 0..255 of the last "
            "32768-token chunk of a 1048576-token prompt, behind its 1015808-token cached "
            "prefix, windows of 1015809..1016064 keys; floor(2**28 / 1048576) = 256 rows, a "
            "256 x 1048832 fp32 tile of 1,074,003,968 bytes",
            spec=dict(**_INDEXER, seqs=_LAST_CHUNK_1M, rows=(0, 256)),
        ),
        Cell(
            why="Mistral Large 4 1M, last 256-row tile at 1048576 keys: rows 32512..32767 of the "
            "last chunk, windows of 1048321..1048576 keys reaching the last key at "
            "max_seq_len",
            spec=dict(**_INDEXER, seqs=_LAST_CHUNK_1M, rows=(32512, 32768)),
        ),
        Cell(
            why="Mistral Large 4 1M, q-split rank 3's first tile: rows 24576..24831 of the last "
            "chunk; under the engine's query split rank 3 scores rows 24576..32767 against "
            "all 1048576 keys, in 256-row tiles from its share's first row",
            spec=dict(**_INDEXER, seqs=_LAST_CHUNK_1M, rows=(24576, 24832)),
        ),
        Cell(
            why="Mistral Large 4 1M, q-split rank 0's last tile: the 70 rows 8122..8191 of rank "
            "0's share of the last chunk, against its share's 1024000 keys; floor(2**28 / "
            "1024000) = 262-row tiles, 31 of them and this tail, its last query block partial",
            spec=dict(**_INDEXER, seqs=_RANK0_SHARE_1M, rows=(8122, 8192)),
        ),
        Cell(
            why="Mistral Large 4 1M, last 512-row tile at 524288 keys: rows 32256..32767 of a "
            "chunk behind a 491520-token prefix, floor(2**28 / 524288) = 512 rows",
            spec=dict(**_INDEXER, seqs=[(491520, 32768)], rows=(32256, 32768)),
        ),
        Cell(
            why="Mistral Large 4 1M, last 1024-row tile at 262144 keys: rows 31744..32767 of a "
            "chunk behind a 229376-token prefix, floor(2**28 / 262144) = 1024 rows",
            spec=dict(**_INDEXER, seqs=[(229376, 32768)], rows=(31744, 32768)),
        ),
        Cell(
            why="Mistral Large 4 1M, last 2048-row tile at 131072 keys: rows 30720..32767 of a "
            "chunk behind a 98304-token prefix, floor(2**28 / 131072) = 2048 rows",
            spec=dict(**_INDEXER, seqs=[(98304, 32768)], rows=(30720, 32768)),
        ),
        Cell(
            why="Mistral Large 4 1M, last 4096-row tile at 65536 keys: rows 28672..32767 of the "
            "second chunk of a prompt, behind a 32768-token prefix, windows of 61441..65536 "
            "keys crossing 2**16",
            spec=dict(**_INDEXER, seqs=[(32768, 32768)], rows=(28672, 32768)),
        ),
        Cell(
            why="Mistral Large 4 1M, 4095 rows at 65537 keys: a floor(2**28 / 65537) = 4095-row "
            "tile ending at the step's last row, rows 28673..32767 behind a 32769-token "
            "prefix, windows of 61443..65537 keys, an odd key count, its last query block "
            "partial",
            spec=dict(**_INDEXER, seqs=[(32769, 32768)], rows=(28673, 32768)),
        ),
        Cell(
            why="Mistral Large 4 1M, last 8192-row tile at 32768 keys: rows 24576..32767 of a "
            "32768-token first chunk, floor(2**28 / 32768) = 8192 rows; the tile spans "
            "8192 x 33024 fp32, 1,082,130,432 bytes",
            spec=dict(**_INDEXER, seqs=[(0, 32768)], rows=(24576, 32768)),
        ),
        Cell(
            why="Mistral Large 4 1M, 16 packed chunk tails in one call: 16 requests of 16 new "
            "tokens behind 61440-token prefixes, 256 rows against 983296 keys, windows "
            "starting at the multiples of 61456 up to 921840",
            spec=dict(**_INDEXER, seqs=_TAILS_16_1M),
        ),
        Cell(
            why="Mistral Large 4 1M, a 15-token final chunk at 1048559 keys: the envelope gate's "
            "1048559-token prompt (max_seq_len - 16 - 1, 15 tokens past its last full "
            "32-token page) scored alone, 15 rows, a key count off the kernel's 4-key "
            "alignment",
            spec=dict(**_INDEXER, seqs=[(1048544, 15)]),
        ),
        Cell(
            why="Mistral Large 4 1M, the 256-row tile at rows 8192..8447 of the last chunk at "
            "1048576 keys, the 33rd tile of the unsplit chunk and the row the q-split's rank 1 "
            "share starts at: windows of 1024001..1024256 keys",
            spec=dict(**_INDEXER, seqs=_LAST_CHUNK_1M, rows=(8192, 8448)),
        ),
        Cell(
            why="Mistral Large 4 1M, q-split rank 1's first tile: rows 8192..8451 of its share "
            "8192..16383 of the last chunk, against the share's 1032192 keys (cached + 16384); "
            "floor(2**28 / 1032192) = 260-row tiles cut from the share's first row",
            spec=dict(**_INDEXER, seqs=_RANK1_SHARE_1M, rows=(8192, 8452)),
        ),
        Cell(
            why="Mistral Large 4 1M packed, two 1M prompts' 1-token final chunks in one call: 2 rows "
            "against 2097152 keys, the second window [1048576, 2097152) starting at 2**20 and "
            "ending at 2**21; floor(2**28 / 2097152) = 128 rows, one tile",
            spec=dict(**_INDEXER, seqs=_FINAL_TOKENS_2X1M, rows=(0, 2)),
        ),
        Cell(
            why="Mistral Large 4 1M packed, three final chunks at the 2**28 packing budget: 50 + 50 + "
            "13 rows against 2373713 keys (cached 1048000, 1048000, 277600), windows starting at "
            "0, 1048050 and 2096100; floor(2**28 / 2373713) = 113 rows, one tile of 113 x 2374144 "
            "fp32",
            spec=dict(**_INDEXER, seqs=_FINAL_CHUNKS_POOL, rows=(0, 113)),
        ),
        Cell(
            why="Mistral Large 4 1048576 keys, the last 256-row tile of the 2**28 logits budget "
            "(floor(2**28 / 1048576)): the 32nd chunk of a 1M prompt, 32768 new over 1015808 "
            "cached, windows of 1048321..1048576 keys; rows 1048832 fp32 wide",
            spec=dict(**_INDEXER, seqs=_LAST_CHUNK_1M, rows=(32512, 32768), budget=2**28),
        ),
        Cell(
            why="Mistral Large 4 1048576 keys, the first 256-row tile of the 2**28 budget: the "
            "32nd chunk's rows 0..255, windows of 1015809..1016064 keys",
            spec=dict(**_INDEXER, seqs=_LAST_CHUNK_1M, rows=(0, 256), budget=2**28),
        ),
        Cell(
            why="Mistral Large 4 1048576 keys, the last 64-row tile of the 2**26 budget (the "
            "fallback): the 32nd chunk's rows 32704..32767",
            spec=dict(**_INDEXER, seqs=_LAST_CHUNK_1M, rows=(32704, 32768), budget=2**26),
        ),
        Cell(
            why="Mistral Large 4 1048576 keys, the last 16-row tile of the 2**24 budget (the "
            "target's before the 1M envelope): the 32nd chunk's rows 32752..32767",
            spec=dict(**_INDEXER, seqs=_LAST_CHUNK_1M, rows=(32752, 32768), budget=2**24),
        ),
        Cell(
            why="Mistral Large 4 524288 keys, the last 512-row tile of the 2**28 budget: the 16th "
            "chunk of a 1M prompt, 32768 new over 491520 cached",
            spec=dict(**_INDEXER, seqs=[(491520, 32768)], rows=(32256, 32768), budget=2**28),
        ),
        Cell(
            why="Mistral Large 4 131072 keys, the last 2048-row tile of the 2**28 budget: the 4th "
            "chunk of a 1M prompt, 32768 new over 98304 cached",
            spec=dict(**_INDEXER, seqs=[(98304, 32768)], rows=(30720, 32768), budget=2**28),
        ),
        Cell(
            why="Mistral Large 4 98304 keys, a 2730-row tile of the 2**28 budget (floor(2**28 / "
            "98304), not a multiple of the 4-row query block): the 3rd chunk's rows "
            "30030..32759",
            spec=dict(**_INDEXER, seqs=[(65536, 32768)], rows=(30030, 32760), budget=2**28),
        ),
        Cell(
            why="Mistral Large 4 65536 keys, the last 4096-row tile of the 2**28 budget: the 2nd "
            "chunk of a 1M prompt, 32768 new over 32768 cached",
            spec=dict(**_INDEXER, seqs=[(32768, 32768)], rows=(28672, 32768), budget=2**28),
        ),
        Cell(
            why="Mistral Large 4 32768 keys, the last 8192-row tile of the 2**28 budget: a 1M "
            "prompt's first chunk, no cache, windows of 24577..32768 keys",
            spec=dict(**_INDEXER, seqs=[(0, 32768)], rows=(24576, 32768), budget=2**28),
        ),
        Cell(
            why="Mistral Large 4 16 packed chunk tails, 16 tokens over 65520 cached each: one "
            "1048576-key block of 256 rows (2**28 elements, the engine's packed-logits cap), "
            "windows starting at multiples of 65536",
            spec=dict(**_INDEXER, seqs=_PACKED_16_TAILS_1M, budget=2**28),
        ),
        Cell(
            why="Mistral Large 4 2 packed chunk tails, 144 over 262000 cached and 112 over 500000: "
            "762256 keys in one call, the second request's windows starting at key 262144",
            spec=dict(**_INDEXER, seqs=[(262000, 144), (500000, 112)], budget=2**28),
        ),
        Cell(
            why="Mistral Large 4 packed chunk split per request: a 5-token last chunk over 983040 "
            "cached, 983045 keys from key 0, the call each request of the engine's packed "
            "1966090-key chunk takes once the target splits it",
            spec=dict(**_INDEXER, seqs=[(983040, 5)], budget=2**28),
        ),
        Cell(
            why="Mistral Large 4 2 packed 5-token last chunks over 983040 cached each, unsplit: "
            "1966090 keys in one call, past max_seq_len, the second request's first row inside "
            "the first's 4-row query block",
            spec=dict(**_INDEXER, seqs=[(983040, 5), (983040, 5)], budget=2**28),
        ),
        # The dep4 target's calls (dep8 alike) with the split-KV schedule
        # (`split_kv`): its 2**26-element tiles at the 1M envelope and its
        # 2**24-element tiles at max_seq_len 32768, each scored with the schedule
        # `mqa_logits_schedule` builds over the tile's own windows and keys.
        Cell(
            why="dep4 split-KV, the last 64-row tile of the 2**26 budget at 1048576 keys: the "
            "32nd chunk of a 1M prompt, rows 32704..32767, 16 query blocks spread over every SM",
            spec=dict(
                **_INDEXER, seqs=_LAST_CHUNK_1M, rows=(32704, 32768), budget=2**26, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="dep4 split-KV, the first 64-row tile at 1048576 keys: the 32nd chunk's rows "
            "0..63, windows of 1015809..1015872 keys",
            spec=dict(**_INDEXER, seqs=_LAST_CHUNK_1M, rows=(0, 64), budget=2**26, split_kv=True),
        ),
        Cell(
            why="dep4 split-KV, a 15-row final chunk at 1048559 keys: the envelope prompt's "
            "15-token tail alone, a partial query block, a key count off the 4-key alignment",
            spec=dict(**_INDEXER, seqs=[(1048544, 15)], budget=2**26, split_kv=True),
        ),
        Cell(
            why="dep4 split-KV, the last 128-row tile at 524288 keys: a chunk behind a "
            "491520-token prefix, floor(2**26 / 524288) = 128 rows",
            spec=dict(
                **_INDEXER, seqs=[(491520, 32768)], rows=(32640, 32768), budget=2**26, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="dep4 split-KV, the last 256-row tile at 262144 keys: behind a 229376-token prefix",
            spec=dict(
                **_INDEXER, seqs=[(229376, 32768)], rows=(32512, 32768), budget=2**26, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="dep4 split-KV, a 682-row tile at 98304 keys (floor(2**26 / 98304), not a "
            "multiple of the 4-row query block): the 3rd chunk's rows 32054..32735",
            spec=dict(
                **_INDEXER, seqs=[(65536, 32768)], rows=(32054, 32736), budget=2**26, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="dep4 split-KV, the last 1024-row tile at 65536 keys: the 2nd chunk of a 1M "
            "prompt, windows crossing 2**16",
            spec=dict(
                **_INDEXER, seqs=[(32768, 32768)], rows=(31744, 32768), budget=2**26, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="dep4 split-KV, the first 2048-row tile at 32768 keys: a 1M prompt's first "
            "chunk, causal windows of 1..2048 keys, rows whose whole window is one split",
            spec=dict(**_INDEXER, seqs=[(0, 32768)], rows=(0, 2048), budget=2**26, split_kv=True),
        ),
        Cell(
            why="dep4 split-KV, the last 512-row tile of the 2**24 budget at 32768 keys: the "
            "32K config's whole-prompt step, windows of 32257..32768 keys",
            spec=dict(
                **_INDEXER, seqs=[(0, 32768)], rows=(32256, 32768), budget=2**24, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="dep4 split-KV, 16 packed requests at 32768 keys, the 512-row tile at rows "
            "31744..32255 (2**24): two requests' windows, request boundaries inside query blocks",
            spec=dict(
                **_INDEXER, seqs=_PACKED_16_32K, rows=(31744, 32256), budget=2**24, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="dep4 split-KV, 16 packed chunk tails in one 1048576-key block, the 64-row tile "
            "at rows 192..255 (2**26): four requests' windows starting at multiples of 65536",
            spec=dict(
                **_INDEXER, seqs=_PACKED_16_TAILS_1M, rows=(192, 256), budget=2**26, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="dep4 split-KV, 2 packed tails over 762256 keys, the 88-row tile at rows "
            "88..175 (2**26) across the request boundary at row 144",
            spec=dict(
                **_INDEXER,
                seqs=[(262000, 144), (500000, 112)],
                rows=(88, 176),
                budget=2**26,
                split_kv=True,
            ),  # fmt: skip
        ),
        Cell(
            why="dep4 split-KV, one request's part of the packed 1966090-key chunk: 5 rows "
            "against its own 983045 keys from key 0",
            spec=dict(**_INDEXER, seqs=[(983040, 5)], budget=2**26, split_kv=True),
        ),
        # The tep4 target's calls (tep8 alike) with the split-KV schedule: its 2**28-element
        # tiles at the 1M envelope, cut from the first row of the rank's share of a chunk under
        # the engine's query split, and the pieces of a packed chunk cut per request
        # past max_seq_len. At max_seq_len 32768 its 2**24 tiles are the dep4 cells' above.
        Cell(
            why="tep4 split-KV, q-split rank 3's first 256-row tile at 1048576 keys: rows "
            "24576..24831 of the last chunk of a 1M prompt",
            spec=dict(
                **_INDEXER, seqs=_LAST_CHUNK_1M, rows=(24576, 24832), budget=2**28, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="tep4 split-KV, q-split rank 0's 70-row last tile at 1024000 keys: a partial "
            "query block",
            spec=dict(
                **_INDEXER, seqs=_RANK0_SHARE_1M, rows=(8122, 8192), budget=2**28, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="tep4 split-KV, q-split rank 1's first 260-row tile at 1032192 keys",
            spec=dict(
                **_INDEXER, seqs=_RANK1_SHARE_1M, rows=(8192, 8452), budget=2**28, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="tep4 split-KV, q-split rank 3's first 512-row tile at 524288 keys: a chunk "
            "behind a 491520-token prefix",
            spec=dict(
                **_INDEXER, seqs=[(491520, 32768)], rows=(24576, 25088), budget=2**28, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="tep4 split-KV, q-split rank 3's whole share of a 1M prompt's first chunk: 8192 "
            "rows against 32768 keys in one tile",
            spec=dict(
                **_INDEXER, seqs=[(0, 32768)], rows=(24576, 32768), budget=2**28, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="tep4 split-KV, q-split rank 0's whole share of a first chunk: 8192 rows "
            "against its share's 8192 keys, causal windows",
            spec=dict(
                **_INDEXER, seqs=[(0, 8192)], rows=(0, 8192), budget=2**28, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="tep8 split-KV, q-split rank 0's 151-row last tile at 1019904 keys",
            spec=dict(
                **_INDEXER,
                seqs=_TEP8_RANK0_SHARE_1M,
                rows=(3945, 4096),
                budget=2**28,
                split_kv=True,
            ),  # fmt: skip
        ),
        Cell(
            why="tep8 split-KV, q-split rank 7's whole 4096-row share of a first chunk at "
            "32768 keys",
            spec=dict(
                **_INDEXER, seqs=[(0, 32768)], rows=(28672, 32768), budget=2**28, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="tep4 split-KV, an unsplit 8191-row tail chunk's last 255 rows at 1048576 "
            "keys, below the split threshold",
            spec=dict(
                **_INDEXER, seqs=[(1040385, 8191)], rows=(7936, 8191), budget=2**28, split_kv=True
            ),  # fmt: skip
        ),
        Cell(
            why="tep4 split-KV, 16 packed chunk tails, 256 rows against 983296 keys in one "
            "2**28 tile, windows starting at multiples of 61456",
            spec=dict(**_INDEXER, seqs=_TAILS_16_1M, budget=2**28, split_kv=True),
        ),
        Cell(
            why="tep4 split-KV, the packed-chunk cut: a ~1M request's 100-token tail against its "
            "own 1048500 keys, the piece of a 1048655-key chunk the target scores alone",
            spec=dict(**_INDEXER, seqs=[(1048400, 100)], budget=2**28, split_kv=True),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, rank 7 first tile at 1048576 keys: rows 28672..28927 of "
            "the last 32768-token chunk of a 1048576-token prompt behind its 1015808-token "
            "cached prefix, windows of 1044481..1044736 keys; under tep8's query split rank 7 "
            "scores rows 28672..32767 against all 1048576 keys in 256-row tiles",
            spec=dict(**_INDEXER, seqs=_LAST_CHUNK_1M, rows=(28672, 28928), share=(28672, 32768)),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, rank 3 last tile at 1032192 keys: the 196 rows "
            "16188..16383 of rank 3's share 12288..16383 of the last chunk, against the "
            "share's cached + 16384 keys; floor(2**28 / 1032192) = 260-row tiles, 15 of them "
            "and this tail, windows of 1031997..1032192 keys reaching the share's last key",
            spec=dict(
                **_INDEXER, seqs=_TEP8_RANK3_SHARE_1M, rows=(16188, 16384), share=(12288, 16384)
            ),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, rank 3 first tile at 1032192 keys: rows 12288..12547 of "
            "the last chunk, the first 260-row tile of rank 3's share, windows of "
            "1028097..1028356 keys",
            spec=dict(
                **_INDEXER, seqs=_TEP8_RANK3_SHARE_1M, rows=(12288, 12548), share=(12288, 16384)
            ),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, rank 0 last tile at 1019904 keys: the 151 rows 3945..4095 "
            "of rank 0's share 0..4095 of the last chunk, against cached + 4096 keys; "
            "floor(2**28 / 1019904) = 263-row tiles, 15 of them and this tail, its last query "
            "block partial",
            spec=dict(**_INDEXER, seqs=_TEP8_RANK0_SHARE_1M, rows=(3945, 4096), share=(0, 4096)),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, rank 0 first tile at 1019904 keys: rows 0..262 of the "
            "last chunk, an odd 263-row tile, windows of 1015809..1016071 keys",
            spec=dict(**_INDEXER, seqs=_TEP8_RANK0_SHARE_1M, rows=(0, 263), share=(0, 4096)),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, rank 7 first tile at 524288 keys: rows 28672..29183 of a "
            "chunk behind a 491520-token prefix, the first 512-row tile of rank 7's share, "
            "windows of 520193..520704 keys",
            spec=dict(
                **_INDEXER, seqs=[(491520, 32768)], rows=(28672, 29184), share=(28672, 32768)
            ),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, rank 0 last tile at 495616 keys: the 309 rows 3787..4095 "
            "of rank 0's share of a chunk behind a 491520-token prefix, against cached + 4096 "
            "keys; floor(2**28 / 495616) = 541-row tiles, 7 of them and this tail",
            spec=dict(**_INDEXER, seqs=[(491520, 4096)], rows=(3787, 4096), share=(0, 4096)),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, rank 0 first tile at 495616 keys: rows 0..540 of that "
            "chunk, an odd 541-row tile, windows of 491521..492061 keys",
            spec=dict(**_INDEXER, seqs=[(491520, 4096)], rows=(0, 541), share=(0, 4096)),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, rank 7 first tile at 262144 keys: rows 28672..29695 of a "
            "chunk behind a 229376-token prefix, the first 1024-row tile of rank 7's share, "
            "windows of 258049..259072 keys",
            spec=dict(
                **_INDEXER, seqs=[(229376, 32768)], rows=(28672, 29696), share=(28672, 32768)
            ),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, rank 7 first tile at 131072 keys: rows 28672..30719 of a "
            "chunk behind a 98304-token prefix, the first 2048-row tile of rank 7's share, "
            "windows of 126977..129024 keys",
            spec=dict(**_INDEXER, seqs=[(98304, 32768)], rows=(28672, 30720), share=(28672, 32768)),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, rank 7 whole share at 32768 keys: rows 28672..32767 of a "
            "prompt's 32768-token first chunk in one call (floor(2**28 / 32768) = 8192 rows "
            "exceed the 4096-row share), windows of 28673..32768 keys",
            spec=dict(**_INDEXER, seqs=[(0, 32768)], rows=(28672, 32768), share=(28672, 32768)),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, rank 0 whole share at 4096 keys: rows 0..4095 of a "
            "prompt's 32768-token first chunk, against the share's 4096 keys in one call, "
            "causal windows of 1..4096 keys",
            spec=dict(**_INDEXER, seqs=[(0, 4096)], rows=(0, 4096), share=(0, 4096)),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, the envelope prompt's last chunk: the 1048559-token "
            "prompt's 32751-token last chunk behind 1015808 cached, split 4093 + 7 x 4094; "
            "rank 7's share 28657..32750 in floor(2**28 / 1048559) = 256-row tiles, this its "
            "254-row tail 32497..32750, windows of 1048306..1048559 keys",
            spec=dict(
                **_INDEXER, seqs=[(1015808, 32751)], rows=(32497, 32751), share=(28657, 32751)
            ),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, an unsplit 8191-row tail chunk: 8191 new tokens behind "
            "1040385 cached, below the 8192-row split threshold, so one rank scores every row "
            "against 1048576 keys; 31 256-row tiles and this 255-row tail 7936..8190, windows "
            "of 1048322..1048576 keys",
            spec=dict(**_INDEXER, seqs=[(1040385, 8191)], rows=(7936, 8191), share=(0, 8191)),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, one request's part of a packed chunk past 1M keys: two "
            "5-token final chunks over 983040-token prefixes pack into one engine chunk of "
            "1966090 keys, which the tep8 target scores per request, each part 5 rows against "
            "its request's own 983045 keys, windows of 983041..983045",
            spec=dict(**_INDEXER, seqs=[(983040, 5)]),
        ),
    )

    note = """
    Metadata consumed: none. `cu_seq_len_k_start` / `cu_seq_len_k_end` are int32
    [T], one window per query row, in the row coordinates of `k`; packed requests
    are expressed through them alone. `q` [T, H, D] e4m3, `k` [N, D] e4m3 and
    `k_scale` [N] fp32 are contiguous (the op asserts it); `weights` [T, H] fp32.
    The output is the op's own allocation, returned as a [T, N] view of rows
    whose stride is `align(N + 256, 256)` fp32 (measured 33024 at N = 32768,
    9472 at N = 9216, 2304 at N = 2048, 512 at N = 137, 1048832 at N = 1048576,
    1024256 at N = 1024000, 66048 at N = 65537, 1032448 at N = 1032192, 2097408
    at N = 2097152, 2374144 at N = 2373713, 1966592 at N = 1966090, 1020160 at
    N = 1019904, 495872 at N = 495616, 1048832 at N = 1048559, 983552 at
    N = 983045, 4352 at N = 4096; every cell asserts it), so an untiled call at
    T = N = 32768 allocates 4,328,521,728 bytes.

    Only `clean_logits=True` with `max_seqlen_k=0` is certified, which is what
    Mistral Large 4 passes. `clean_logits=False` leaves out-of-window entries
    unspecified, and `max_seqlen_k > 0` switches to a compressed layout where
    column `j` of row `i` is key `cu_seq_len_k_start[i] + j`; `reference`
    implements both, no cell drives them.

    At the NVFP4 tep4 and tp4 targets' 1M envelope (max_seq_len 1048576,
    chunked prefill in 32768-token steps) each indexer prefill chunk is scored
    against its requests' whole KV, cached + new, so one request's N reaches
    1048576, in query tiles of floor(2**28 / N) rows cut from the chunk's first
    row: 256 rows at 1048576 keys, 262 at 1024000, 260 at 1032192, 512 at
    524288, 1024 at 262144, 2048 at 131072, 4096 at 65536, 4095 at 65537 and
    8192 at 32768, each about 1 GiB of fp32 logits. Under the engine's query
    split (a step of 8192 or more context query tokens; tep4 and tp4 alike,
    attention DP off) rank r scores the chunk's rows [8192 r, 8192 (r + 1))
    against the keys up to its share's end (`_split_prefill_queries` and
    `plan_mla_chunked_prefill_chunks` in
    `_torch/attention/backends/sparse/dsa/indexer.py`); the cells hold rank 1's
    and rank 3's first tiles and rank 0's 70-row tail, rank 2 (1040384 keys,
    258-row tiles) only through the uniformity that follows. The engine packs
    consecutive context requests into one chunk while sum(q) x sum(kv) <= 2**28
    (`group_prefill_requests_by_logits_budget`, same file), so a packed chunk's
    rows never exceed floor(2**28 / N) and the target scores it in one call; its
    later requests' windows start past key 0, at the sum of the earlier
    requests' KV. The cells hold two 1M prompts' 1-token final chunks (2 rows,
    2097152 keys, the second window starting at 2**20) and three final chunks
    at the packing budget (113 rows, 2373713 keys, windows from 0, 1048050 and
    2096100). With block reuse off a packed chunk's keys are bounded by the
    rank's KV pool, 2,373,856 tokens at the tp4 target's delivered boot, 143
    above the widest cell; with block reuse on, requests sharing a cached
    prefix each count it, and a chunk can reach 16 x 1048576 = 2**24 keys at
    the packing cap, outside the cells. The kernel reads each window bound and
    key index as an unsigned 32-bit integer, scans each 4-row query block's keys
    from its smallest window start, and forms every logit's offset as row x
    stride in 64 bits (`get_mqa_logits_kv_span`, `get_logits_row` and
    `get_logits_col` in DeepGEMM's `scheduler/sm100_mqa_logits.cuh`): N moves
    which keys a block scans and how wide the rows are, not the arithmetic. Not
    certified: N above 2373713 (tep4's and tep8's cells alone reach N = 1048576,
    dep4's 1966090, tp4's 2373713). The short final chunks of two near-1M
    prompts can share a chunk of about 2**21 keys (two 5-token tails over
    983040-token prefixes: 1,966,090, scored whole and split per request in
    dep4's cells, below, and split per request by the tep8 target); tep4's and
    tep8's cells score no packed chunk whole, tp4's two above do, at 2097152
    and 2373713 keys.

    The 1M envelope (Mistral Large 4 NVFP4 dep4: max_seq_len 1048576, chunked prefill
    in 32768-token chunks). Each engine indexer chunk is scored over its
    gathered keys, cached + new, up to n_k = 1048576 (a 1M prompt's 32nd chunk),
    in query tiles of floor(budget / n_k) rows. The `budget` cells are 2**28
    tiles at 32768 (8192 rows), 65536 (4096), 98304 (2730), 131072 (2048),
    524288 (512) and 1048576 keys (256, the first and the last tile), and the
    last 2**26 (64-row) and 2**24 (16-row) tiles at 1048576 keys. Every 2**26
    and 2**24 tile inside a 2**28 tile cell is bitwise its rows of it, so those
    budgets' tiles are certified at the same widths. A call's only device
    allocation is its output: 1,074,003,968 bytes at 256 x 1048576 and
    1,082,130,432 at 8192 x 32768 (measured peak); a caller that holds the
    previous tile's logits while the next call allocates peaks at two tiles,
    2,148,007,936 bytes at 2**28 and 1048576 keys.
    Packed chunks: the engine groups whole requests into one chunk while
    sum(new) x sum(cached + new) <= 2**28. dep4's cells hold 16 requests' 16-token
    tails over 65520 cached keys each in one 1048576-key block, two tails over
    762256 keys, and two 5-token tails over 983040 cached keys each, 1966090
    keys in one call, past max_seq_len. Split per request, each request's own
    call over its 983045 keys from key 0 (a cell) is bitwise its rows and keys
    of the packed call. A split must give each call a 16-byte-aligned
    `k_scale`: a view of the packed scales at a key offset that is not a
    multiple of 4 raises (below). So gather each request's keys into buffers of
    their own, or copy the scales.

    The NVFP4 tep8 target at the same envelope tiles the same way, at eight
    ranks: under the engine's query split (`_split_prefill_queries` in
    `_torch/attention/backends/sparse/dsa/indexer.py`, a step of 8192 or more
    context query tokens) rank r scores the chunk's rows [count r / 8,
    count (r + 1) / 8) against the keys up to its share's end (cached +
    4096 (r + 1) for a 32768-token chunk), in floor(2**28 / N)-row tiles cut
    from the share's first row, so a call holds at most 4096 rows of a split
    step. The "Mistral Large 4 tep8 1M" cells hold each share's first and last tile
    behind cached prefixes of 1015808 (ranks 0, 3 and 7: 263-row tiles at
    1019904 keys, 260 at 1032192, 256 at 1048576), 491520 (ranks 0 and 7: 541
    at 495616, 512 at 524288), 229376 and 98304 (rank 7), rank 7's and rank 0's
    whole shares of a first chunk (4096 rows at 32768 and at 4096 keys), rank
    7's 254-row tail of the 1048559-token envelope prompt's 32751-token last
    chunk (shares 4093 + 7 x 4094), the 255-row tail of an unsplit 8191-row
    chunk at 1048576 keys, and one request's 5 rows against its 983045 keys,
    the part of the packed 1966090-key chunk above that the tep8 target scores
    per request. Every tile of each of those shares is bitwise its rows of a
    call over up to four tiles of the share, and every rank's first and last
    tile of the 1M prompt's last chunk (32768 and 32751 tokens) over its
    share's keys is bitwise its rows and keys of the same rows scored against
    all the chunk's keys, which hold -inf past the share's last key: the split
    changes no score.

    Measured on sm_103, on the path the cells drive (the SM100 DeepGEMM kernel,
    fp32 weights, no schedule metadata, no autotuning), each kept true by a test:
    - repeated calls are bitwise equal, and each query tile is bitwise the same
      rows of one untiled call, so a caller may tile the queries at any row count
      (proved at the 1820-row tiles a 2**24-element logits budget cuts at 9216 keys
      and the 512-row tiles it cuts at 32768; from 32768 to 1048576 keys, where
      no untiled chunk fits a call, against the `budget` cells' 2**28 tiles as
      above); at the 2**28 budget each tile is bitwise the same rows of a call
      over the step's last four tiles, at every N of the "Mistral Large 4 1M" cells,
      tiles that start inside that call's 4-row query blocks included (262 rows
      at 1024000 keys, 4095 at 65537), and at 32768 keys that call is the whole
      step;
    - a window that ends past the last key is clamped to the keys passed and
      answers as `reference` does;
    - row-padded `weights` (stride(0) 64 at 32 heads) score bitwise like packed
      ones.
    Measured to raise: a `weights` row stride that is not a multiple of 16
    bytes, with `RuntimeError: Assertion error (.../jit_kernels/impls/runtime_utils.hpp:130):
    num_gmem_outer_stride_bytes % 16u == 0u`; bf16 `weights`, with
    `RuntimeError: Assertion error (.../csrc/apis/attention.hpp:144):
    weights.scalar_type() != torch::kBFloat16 or logits_dtype == torch::kBFloat16`;
    a `k_scale` whose data pointer is not 16-byte aligned (a view starting at
    a key offset of 1, 2 or 3 mod 4), with `RuntimeError: Assertion error
    (.../csrc/apis/../jit_kernels/impls/runtime_utils.hpp:129):
    reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16u == 0u`.
    `is_valid` refuses fp16 weights, a negative window start and an inverted
    window: the op accepts each and answers wrongly (see `is_valid`).

    Cost on sm_103: without schedule metadata the scheduler gives each SM whole
    4-row query blocks and never splits a block's keys (read in
    `scheduler/sm100_mqa_logits.cuh`), so a 256-row tile at 1048576 keys keeps
    64 SMs busy, each scanning its block's window to the end, and a
    tile of fewer rows at about 1M keys takes as long as 256 rows. A 4-row
    query block holding rows of two near-1M requests (a packed final chunk) is
    scanned by one SM from its smallest window start to its largest window
    end, about 2**21 keys, so such a call takes longer than a 1M tile.

    Hard limits (sm_103)
      logits   row offset: row index x row stride, formed in 64 bits (the scheduler's
               `get_logits_row` times a uint64 stride, DeepGEMM `sm100_mqa_logits.cuh`);
               the allocation's element count is torch's int64
               2**31 and 2**32 bytes: T = N = 32768 in one call (a cell) puts
               rows 16257 on past 2**31 bytes and 32514 on past 2**32  passed
               2**31 elements: crossed in row 65027 at N = 32768, outside the
               envelope (its largest call spans 1,082,130,176 elements);
               probed once at 65100 rows x 32768 keys (2,149,862,400
               elements), rows 63488..65099 against `reference`        passed
               at the 1M envelope (stride 1048832 at N = 1048576, 2**31
               elements inside row 2047) a 2**28-budget tile, whose rows
               never exceed its keys, spans at most 2**28 + 511 x 16384 =
               276,807,680 elements, 1,107,230,720 bytes; the 256-row tiles
               at 1048576 keys span 268,500,992 (1,074,003,968 bytes), the
               113-row packed chunk at 2373713 keys 268,278,272
               (1,073,113,088 bytes)                                   not reached
               the stride itself is a uint32 (`logits_stride`): 2**32 at
               N >= 2**32 - 511                                        not reached
               the 2**28 tile cells span at most 270,532,608 elements
               (8192 x 33024 at 32768 keys; 268,500,992 at 256 x 1048576)  no boundary
               at the 1M envelope (stride 1048832 at N = 1048576, 2**31
               elements inside row 2047) a 2**28-budget tile, whose rows
               never exceed its keys, spans at most 2**28 + 511 x 16384 =
               276,807,680 elements, 1,107,230,720 bytes; the 256-row tiles
               at 1048576 keys span 268,500,992 (1,074,003,968 bytes)  not reached
      q        T * H rows of D bytes, a 32-bit TMA row coordinate      2**31 rows
               envelope max 32768 * 32 = 1,048,576 rows untiled, 16384 * 32 =
               524,288 in a 2**28-budget tile                          not reached, unprobed
      k        N rows of D bytes, a 32-bit TMA row coordinate (`get_kv_tma_offset`)
               2**31 rows (key indices, spans and columns are uint32 in the
               kernel's scheduler); cells reach N = 2373713 (303,835,264 bytes
               of keys, under 2**31 bytes until N = 2**24), dep4's cells
               N = 1966090                                             not reached, unprobed
      windows  int32 key indices, < N <= 2373713 (cells reach N = 2373713, dep4's
               N <= 1966090; the per-rank pool bounds a packed chunk at 2,373,856
               with block reuse off), read as unsigned 32-bit by the kernel; none
               passes through fp32 (dep4's cells: 2**21 < 2**24 in any case), and
               every index stays below 2**24 + 1 regardless
      pool     none: the op reads the keys its caller gathered, not the KV
               pool, so no pool size reaches it

    Not certified: N above 2373713 (dep4's cells alone reach 1966090, see
    above); at 65536 keys and more, a query tile above 2**28 elements (no
    untiled chunk there is a cell); budgets other than 2**24, 2**26 and 2**28
    above 32768 keys; `k_scale` views (only buffers of their own and views at
    key offsets that are multiples of 4 were driven).
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv: Tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        clean_logits: bool = True,
        max_seqlen_k: int = 0,
        schedule_meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if schedule_meta is None:
            return _deep_gemm_fp8_mqa_logits(
                q,
                kv,
                weights,
                cu_seq_len_k_start,
                cu_seq_len_k_end,
                clean_logits,
                max_seqlen_k,
            )
        # The legacy pybind is exactly this call without the schedule (DeepGEMM
        # `csrc/apis/attention.hpp`): q unscaled, fp32 logits.
        return _deep_gemm_fp8_fp4_mqa_logits(
            (q, None),
            kv,
            weights,
            cu_seq_len_k_start,
            cu_seq_len_k_end,
            clean_logits,
            max_seqlen_k,
            torch.float32,
            schedule_meta,
        )

    def reference(
        self,
        q: torch.Tensor,
        kv: Tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        clean_logits: bool = True,
        max_seqlen_k: int = 0,
        schedule_meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The scores in fp64, rounded once to fp32.

        Outside each row's window the entry is -inf when `clean_logits`, else
        NaN, which the test's gate reads as "unspecified". `schedule_meta` is
        not read: it moves which SM computes a logit, not its value.
        """
        k, k_scale = kv
        qd, kd, wd = q.float().double(), k.float().double(), weights.double()
        n = k.shape[0]
        # One head at a time keeps the transient at [T, N] rather than [T, H, N].
        full = torch.zeros(q.shape[0], n, dtype=torch.float64, device=q.device)
        for h in range(q.shape[1]):
            full += (qd[:, h] @ kd.t()).relu() * wd[:, h : h + 1]
        full *= k_scale.double().unsqueeze(0)
        ks = cu_seq_len_k_start.long().unsqueeze(1)
        ke = cu_seq_len_k_end.long().unsqueeze(1)
        fill = float("-inf") if clean_logits else float("nan")
        if max_seqlen_k == 0:
            cols = torch.arange(n, device=q.device).unsqueeze(0)
            out = full.masked_fill((cols < ks) | (cols >= ke), fill)
        else:
            cols = torch.arange(max_seqlen_k, device=q.device).unsqueeze(0)
            src = (ks + cols).clamp(max=n - 1)
            out = torch.gather(full, 1, src).masked_fill(cols >= ke - ks, fill)
        return out.float()

    def is_valid(
        self,
        q: torch.Tensor,
        kv: Tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        clean_logits: bool = True,
        max_seqlen_k: int = 0,
        schedule_meta: Optional[torch.Tensor] = None,
    ) -> None:
        if schedule_meta is not None:
            # The kernel trusts the schedule: one built for other windows, another key
            # count or another SM count scores the wrong spans and answers wrongly. Its
            # live words (per-SM starts and split counts, then each query block's span)
            # must be the ones this call's own schedule holds; the pad between them is
            # never written.
            fresh = mqa_logits_schedule(
                cu_seq_len_k_start, cu_seq_len_k_end, kv[0].shape[0], q.shape[1]
            )
            sms = _deep_gemm_num_sms()
            span_off = (3 * sms + 1) // 2 * 2
            blocks = -(-q.shape[0] // (128 // q.shape[1]))
            live = [slice(0, 3 * sms), slice(span_off, span_off + 2 * blocks)]
            assert schedule_meta.dtype == torch.int32 and schedule_meta.numel() >= fresh.numel(), (
                f"schedule_meta {schedule_meta.dtype} [{schedule_meta.numel()}] is not this "
                f"call's int32 [{fresh.numel()}] schedule"
            )
            assert all(torch.equal(schedule_meta[s], fresh[s]) for s in live), (
                "schedule_meta was not built for this call's windows, keys and SM count"
            )
        # fp16 weights are accepted and select a second kernel, which answers
        # at half precision: measured relative RMS error 4.8e-4 against the
        # reference, where the fp32 path measures 6.8e-8.
        assert weights.dtype == torch.float32, (
            f"weights must be fp32; got {weights.dtype}, which selects a kernel that answers "
            f"at half precision"
        )
        if cu_seq_len_k_start.numel():
            # A negative start is not read as 0: with every start at -1, every
            # logit of the call came back -inf.
            assert int(cu_seq_len_k_start.min()) >= 0, "a window starts before key 0"
            # An inverted window is not read as empty: its row came back with
            # finite logits where an empty window gives -inf.
            assert bool((cu_seq_len_k_start <= cu_seq_len_k_end).all()), (
                "a window ends before it starts"
            )


fp8_mqa_logits = Fp8MqaLogits()
